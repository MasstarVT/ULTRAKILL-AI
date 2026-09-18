"""One size-bounded, timestamped event log per env worker: `runs/<run>/env_<port>.log`.

**Why this exists.** On 2026-09-17 the `campaign_gates` run froze for 19 minutes and the post-mortem could not
name the call that blocked, because every diagnostic the env prints went nowhere: the supervisor spawns the
trainer with `DETACHED_PROCESS`, and a detached `cmd /c "... >> log"` creates the file and then silently
discards the child's stdout (reproduced three ways on 2026-09-17; see the freeze gotcha in CLAUDE.md). The
trainer's own log had not been written to in seven hours and nobody noticed, because the supervisor prints its
tail on every restart and a stale tail looks exactly like a fresh one.

So the env no longer *prints* what happened to its bridge. It writes it, to its own file, named after the port
that owns it -- which is the one identifier that ties a Python worker, a TCP port, a game process and a game
window together. The next incident is attributable from `runs/<run>/env_*.log` alone.

Bounded: at `max_bytes` the file is rolled to `<name>.1` (replacing an older roll) and started again, so twelve
workers logging every reset cannot fill a disk overnight.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # per port; twelve ports roll at 24 MB plus one roll each


class EnvLog:
    """Append-only one-line events, with a clock and a filesystem that the tests replace.

    Every method is failure-tolerant on purpose: a log that raises would turn a recoverable bridge fault into
    a dead worker, which is the exact failure it exists to explain.
    """

    def __init__(self, path: str | os.PathLike[str] | None, port: int,
                 max_bytes: int = DEFAULT_MAX_BYTES, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else None
        self.port = port
        self.max_bytes = max_bytes
        self.clock = clock
        self.enabled = self.path is not None
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    # -- writing ------------------------------------------------------------------------------

    def event(self, kind: str, **fields: object) -> str:
        """Writes `<iso time> port=<n> <kind> k=v ...` and returns the line (without its newline).

        The line is returned even when nothing is written, so a caller can hand the same text to a print or an
        assertion without formatting it twice.
        """
        line = "%s port=%d %s%s" % (self._stamp(), self.port, kind, self._fields(fields))
        if not self.enabled:
            return line
        try:
            self._roll_if_needed()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        return line

    def _stamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))

    @staticmethod
    def _fields(fields: dict[str, object]) -> str:
        out = []
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                value = "%.2f" % value
            text = str(value)
            out.append("%s=%s" % (key, text.replace("\n", " ") if " " not in text else '"%s"' % text.replace("\n", " ")))
        return (" " + " ".join(out)) if out else ""

    def _roll_if_needed(self) -> None:
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
        except OSError:
            return
        try:
            os.replace(self.path, self.path.with_suffix(self.path.suffix + ".1"))
        except OSError:
            # Windows refuses the replace while something else holds the roll open; truncating is still bounded.
            try:
                self.path.write_text("", encoding="utf-8")
            except OSError:
                pass


def env_log_path(log_dir: str | os.PathLike[str] | None, port: int) -> Path | None:
    """`runs/<run>/env_<port>.log`, or None when the run has no log directory (tests, eval, bridge_test)."""
    return Path(log_dir) / ("env_%d.log" % port) if log_dir else None


def tail(path: str | os.PathLike[str], count: int = 5) -> list[str]:
    """The last `count` lines, reading only the tail. Used by the supervisor when it reports a sick run."""
    try:
        path = Path(path)
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - 16384))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-count:]
