"""Keeps a training run alive without an LLM watching it.

On the morning of 2026-09-17 the `campaign_gates` run died on its own: a SubprocVecEnv worker hit a
TimeoutError on a level reset, `train.py`'s `finally` wrote `latest.zip`, and the trainer then hung in
teardown. Nothing restarted it, and the machine sat idle. A monitor agent covered that with a manual
playbook; this is the same playbook as a plain process, so it costs no tokens and never sleeps.

    python scripts/supervise.py --run campaign_gates --config configs/campaign_gates_main.yaml \
        --count 12 --monitor 1

**PAUSING TRAINING ON PURPOSE.** The supervisor's whole job is to start training again, so a planned
stop looks exactly like a crash to it. Before any planned pause, either stop the supervisor or create
the pause file:

    New-Item runs/campaign_gates/SUPERVISOR_PAUSE            # PowerShell, from python/
    Remove-Item runs/campaign_gates/SUPERVISOR_PAUSE         # when the pause is over

While that file exists the supervisor does nothing at all: it does not restart the trainer, does not
touch the games and does not start the helpers. It logs the pause once, and logs once when it lifts.

**Health** is all three of:
  - a `train.py` process for THIS run exists (matched on the command line: `train.py` plus the config
    file name or the run name),
  - `runs/<run>/status.json` was modified within `--stale-seconds`,
  - and its `state` is `"running"`.

A trainer that exists while `status.json` stays stale for the whole window is HUNG: it is killed by PID
(with its multiprocessing workers) before the restart. A trainer that is gone is DEAD: restart straight
away. A trainer that exists with a fresh `status.json` whose state is `stopped`/`finished` is draining
(a Ctrl+C in progress) and is left alone until it either exits or goes stale.

**Matching, and the trap in it.** A command line that *mentions* the trainer is not the trainer. A
previous report script counted the PowerShell query listing the processes -- its own command line
contains the search string -- and reported a trainer that did not exist. `SELF_MARKERS` drops any
command line containing `supervise.py`, `Win32_Process`, `Get-CimInstance`, `tasklist` or `wmic`, and
the supervisor's own process tree is excluded by PID as well.

**A restart** logs the last 30 lines of `runs/<run>_train.log`, kills what is left of the trainer, stops
the games with `games.stop_all()`, relaunches `--count N --monitor M` through `games.launch()` (which
waits for readiness by reading netstat -- this never opens a TCP connection to a bridge port, because
the bridge drops its current client when a new one connects), picks the resume file with the MOST
timesteps among `latest.zip` and the newest `ckpt_*_steps.zip`, and starts the trainer detached with the
same `cmd /c ... >> log 2>&1` pattern a human would use. `poll_status.py` and `keep_best.py` are checked
on EVERY poll, not only during a restart, and restarted if they are missing.

`latest.zip` only updates on a graceful stop, so it is usually BEHIND the newest checkpoint -- hence
reading the step count out of both instead of trusting the name. `latest.zip`'s count comes from the
`num_timesteps` field of the `data` member SB3 writes inside the zip; ties go to `latest.zip`.

After `--max-restarts-per-hour` restarts inside one hour the supervisor gives up and exits non-zero:
something is wrong that restarting cannot fix, and a restart loop would grind the checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HOUR = 3600.0
DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
TAIL_LINES = 30
STOP_WAIT_S = 5.0  # between closing the games and relaunching them

# Command lines that are ABOUT a script rather than an instance of it. The query that lists the
# processes necessarily contains the words being searched for, and so does this supervisor's own
# command line; counting either one is how a previous script reported a trainer that was not there.
SELF_MARKERS = ("supervise.py", "win32_process", "get-ciminstance", "tasklist", "wmic ")

PS_LIST = (
    "$ErrorActionPreference='SilentlyContinue';"
    "@(Get-CimInstance -ClassName Win32_Process |"
    " Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress -Depth 2"
)


class Proc(NamedTuple):
    pid: int
    ppid: int
    cmdline: str


class Status(NamedTuple):
    age: float | None        # seconds since status.json last changed; None when there is no file
    state: str | None        # "running" / "stopped" / "finished"
    timesteps: float | None


# ---------------------------------------------------------------------------
# Pure helpers (every one of these is covered by tests/test_supervise.py)
# ---------------------------------------------------------------------------


def _norm(cmdline: str | None) -> str:
    """Lowercased, forward-slashed command line, so a match does not depend on quoting or slashes."""
    return (cmdline or "").replace("\\", "/").lower()


def matches_script(cmdline: str | None, script: str, run: str, config: str = "") -> bool:
    """True when this command line IS an instance of `script` working on `run`.

    Requires the script's file name and then either the run name or the config file's name, so a
    trainer for another run is not matched. Anything in SELF_MARKERS is rejected first.
    """
    c = _norm(cmdline)
    if not c or any(marker in c for marker in SELF_MARKERS):
        return False
    if script.lower() not in c:
        return False
    tokens = [run.lower()]
    if config:
        tokens.append(Path(config).name.lower())
    return any(token in c for token in tokens)


def self_and_ancestors(procs: Iterable[Proc], pid: int) -> set[int]:
    """This process and every parent of it, so the supervisor can never match its own tree."""
    parent = {p.pid: p.ppid for p in procs}
    seen, cur = {pid}, pid
    while cur in parent and parent[cur] not in seen:
        cur = parent[cur]
        seen.add(cur)
    return seen


def descendants(procs: Iterable[Proc], roots: Iterable[int]) -> list[int]:
    """Every process under `roots`, deepest first -- the SubprocVecEnv workers of a hung trainer.

    `taskkill /T` already walks the tree, but an orphaned worker whose parent died is not under any
    tree any more, so the pids are collected from the snapshot taken BEFORE anything is killed.
    """
    procs = list(procs)
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p.pid)
    out: list[int] = []
    stack = list(roots)
    seen = set(stack)
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in seen:
                seen.add(child)
                out.append(child)
                stack.append(child)
    return list(reversed(out))


def checkpoint_steps(name: str) -> int | None:
    """The step count in a `ckpt_<n>_steps.zip` file name, or None when the name is not one."""
    m = re.fullmatch(r"ckpt_(\d+)_steps\.zip", name)
    return int(m.group(1)) if m else None


def zip_timesteps(path: Path) -> int | None:
    """`num_timesteps` out of an SB3 .zip, or None when it cannot be read.

    SB3 stores the model's scalars as JSON in the archive's `data` member (see
    `stable_baselines3.common.save_util`), so the count is readable without loading torch -- which
    matters here, because the supervisor must stay cheap enough to poll every minute. A half-written
    zip (the trainer was killed mid-save) raises and reads as None, which drops it from the running.
    """
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data").decode("utf-8"))
        value = data.get("num_timesteps")
        return int(value) if isinstance(value, (int, float)) else None
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, UnicodeDecodeError):
        return None


def choose_resume(model_dir: Path, zip_steps: Callable[[Path], int | None] = zip_timesteps
                  ) -> tuple[Path | None, int | None]:
    """The checkpoint with the MOST timesteps among latest.zip and the ckpt_* files.

    `latest.zip` is only written on a graceful stop, so after a hard kill it is usually behind the
    newest checkpoint -- resuming from it would silently throw away the steps in between. After a
    graceful stop it is ahead. Ties go to `latest.zip`, which carries the final optimizer state.
    An unreadable `latest.zip` simply drops out and the newest checkpoint wins.
    """
    candidates: list[tuple[int, int, Path]] = []
    for path in sorted(model_dir.glob("ckpt_*_steps.zip")):
        steps = checkpoint_steps(path.name)
        if steps is not None:
            candidates.append((steps, 0, path))  # rank 0: loses a tie with latest.zip
    latest = model_dir / "latest.zip"
    if latest.exists():
        steps = zip_steps(latest)
        if steps is not None:
            candidates.append((steps, 1, latest))
    if not candidates:
        return None, None
    steps, _, path = max(candidates, key=lambda c: (c[0], c[1]))
    return path, steps


def shell_command(python: str, script: str, args: list[str], log_path: str) -> str:
    """The `cmd /c "<python> -u <script> ... >> <log> 2>&1"` line a human would type.

    `/s` makes cmd strip only the outermost pair of quotes, so the quoted interpreter path survives.
    """
    inner = " ".join(['"%s"' % python, "-u", script, *args, ">>", log_path, "2>&1"])
    return 'cmd.exe /s /c "%s"' % inner


def tail_lines(path: Path, count: int = TAIL_LINES) -> list[str]:
    """The last `count` lines of a file, reading only its tail (train logs run to megabytes)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 65536))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-count:]


# ---------------------------------------------------------------------------
# The live-system pieces, all injectable so the tests touch no real process
# ---------------------------------------------------------------------------


def list_processes() -> list[Proc]:
    """Every process with its parent and command line, read through CIM."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_LIST],
                             capture_output=True, text=True, timeout=120).stdout
        data = json.loads(out) if out.strip() else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if isinstance(data, dict):
        data = [data]
    procs = []
    for row in data:
        try:
            procs.append(Proc(int(row["ProcessId"]), int(row.get("ParentProcessId") or 0),
                              row.get("CommandLine") or ""))
        except (TypeError, ValueError, KeyError):
            continue
    return procs


def probe_status(path: Path, now: float) -> Status:
    try:
        age = now - path.stat().st_mtime
    except OSError:
        return Status(None, None, None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Status(age, None, None)  # mid-write: the age still says whether it is alive
    return Status(age, data.get("state"), data.get("timesteps"))


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def spawn_detached(command: str, cwd: Path) -> int:
    proc = subprocess.Popen(command, cwd=str(cwd), creationflags=DETACHED, close_fds=True)
    return proc.pid


@dataclass
class Config:
    run: str = "campaign_gates"
    config: str = "configs/campaign_gates_main.yaml"
    count: int = 12
    monitor: int = 1
    stale_seconds: float = 600.0
    poll_seconds: float = 60.0
    max_restarts_per_hour: int = 3
    start_grace_seconds: float = 900.0
    base_port: int = 47800
    python: str = sys.executable
    cwd: Path = field(default_factory=Path.cwd)
    runs_dir: str = "runs"
    models_dir: str = "models"
    dry_run: bool = False


class Supervisor:
    """One `tick()` per poll. Every side effect goes through an injected callable, so the tests run
    against fake processes, a fake clock and a fake launcher and never start a game or a trainer."""

    def __init__(self, cfg: Config, *,
                 processes: Callable[[], list[Proc]] = list_processes,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 probe: Callable[[Path, float], Status] = probe_status,
                 kill: Callable[[int], None] = kill_tree,
                 spawn: Callable[[str, Path], int] = spawn_detached,
                 stop_games: Callable[[], None] | None = None,
                 launch_games: Callable[[int, int], bool] | None = None,
                 zip_steps: Callable[[Path], int | None] = zip_timesteps,
                 pid: int | None = None):
        self.cfg = cfg
        self.processes, self.now, self.sleep = processes, now, sleep
        self.probe, self.kill, self.spawn, self.zip_steps = probe, kill, spawn, zip_steps
        self._stop_games = stop_games or self._default_stop_games
        self._launch_games = launch_games or self._default_launch_games
        self.pid = os.getpid() if pid is None else pid

        run_dir = cfg.cwd / cfg.runs_dir / cfg.run
        self.status_path = run_dir / "status.json"
        self.pause_path = run_dir / "SUPERVISOR_PAUSE"
        self.model_dir = cfg.cwd / cfg.models_dir / cfg.run
        self.train_log = cfg.cwd / cfg.runs_dir / f"{cfg.run}_train.log"
        self.log_path = cfg.cwd / cfg.runs_dir / f"{cfg.run}_supervisor.log"

        self.restarts: list[float] = []
        self.grace_until = 0.0
        self.last_heartbeat = 0.0
        self._last_state: str | None = None

    # -- logging ---------------------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Writes one timestamped line to stdout AND to the log file, and needs only one of them to work.

        Measured 2026-09-17: `cmd /c ... >> runs\\<run>_supervisor.log` holds that file with an exclusive
        share mode, so a second writer's `open(..., "a")` raises PermissionError for as long as the
        supervisor runs. Appending to the file directly therefore CANNOT be the only path, or every line
        vanishes when started the documented way. stdout is already redirected into that same file by the
        shell, so exactly one copy lands in it either way: redirected, the direct append fails and stdout
        carries the line; unredirected, stdout goes to the console and the append writes the file.
        """
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass  # detached with no console and no redirect
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # the shell holds it: stdout above already put this line in the file

    def log_once(self, key: str, message: str) -> None:
        """Logs only when the situation changes, so a long pause is one line and not one per poll."""
        if key != self._last_state:
            self.log(message)
        self._last_state = key

    # -- the live defaults ------------------------------------------------------------------------

    def _default_stop_games(self) -> None:
        import games  # imported late: it pulls in ctypes/winreg and the tests never need it

        games.stop_all()

    def _default_launch_games(self, count: int, monitor: int) -> bool:
        import games

        try:
            games.launch(count, self.cfg.base_port, 368, 207, 4.0, 240.0, monitor, 3)
            return True
        except SystemExit as exc:  # games.launch reports failure by exiting
            self.log("games.launch failed: %s" % exc)
            return False
        except Exception as exc:  # noqa: BLE001 - a launch failure must not kill the supervisor
            self.log("games.launch raised %s: %s" % (type(exc).__name__, exc))
            return False

    # -- one poll ---------------------------------------------------------------------------------

    def tick(self) -> str:
        now = self.now()
        if self.pause_path.exists():
            self.log_once("paused", "PAUSED: %s exists, doing nothing (delete it to resume supervision)"
                          % self.pause_path)
            return "paused"

        procs = self.processes()
        mine = self_and_ancestors(procs, self.pid)
        trainers = [p for p in procs
                    if p.pid not in mine and matches_script(p.cmdline, "train.py", self.cfg.run, self.cfg.config)]
        status = self.probe(self.status_path, now)
        fresh = status.age is not None and status.age <= self.cfg.stale_seconds

        if trainers and fresh and status.state == "running":
            self.ensure_helpers(procs)
            self.log_once("ok", "healthy: trainer pid %s, %s steps, status %.0fs old"
                          % (trainers[0].pid, self._steps(status), status.age))
            self.heartbeat(now, trainers, status)
            return "ok"

        if trainers and fresh and status.state in ("stopped", "finished"):
            # A graceful stop in progress: train.py has written state and is saving latest.zip. Leave
            # it alone; if it wedges, status.json goes stale and the hung branch below takes it.
            self.log_once("draining", "trainer pid %s is finishing up (state=%s); leaving it alone"
                          % (trainers[0].pid, status.state))
            return "draining"

        if not trainers and status.state == "finished" and fresh:
            self.log("run finished (%s steps): nothing left to supervise" % self._steps(status))
            return "finished"

        if now < self.grace_until:
            self.log_once("grace", "waiting out the start grace period (%.0fs left) before judging health"
                          % (self.grace_until - now))
            return "grace"

        reason = ("no train.py process for this run" if not trainers else
                  "trainer pid %s is up but status.json is %s" % (
                      trainers[0].pid,
                      "missing" if status.age is None else "%.0fs stale (state=%s)" % (status.age, status.state)))
        return self.restart(now, procs, trainers, reason)

    def _steps(self, status: Status) -> str:
        return "unknown" if status.timesteps is None else "{:,.0f}".format(status.timesteps)

    def heartbeat(self, now: float, trainers: list[Proc], status: Status) -> None:
        if now - self.last_heartbeat < HOUR:
            return
        self.last_heartbeat = now
        self.log("heartbeat: healthy, trainer pid %s, %s steps, status %.0fs old, %d restart(s) in the last hour"
                 % (trainers[0].pid, self._steps(status), status.age or 0.0, len(self.recent_restarts(now))))

    def recent_restarts(self, now: float) -> list[float]:
        self.restarts = [t for t in self.restarts if now - t < HOUR]
        return self.restarts

    # -- the restart ------------------------------------------------------------------------------

    def restart(self, now: float, procs: list[Proc], trainers: list[Proc], reason: str) -> str:
        self._last_state = "restart"
        self.log("UNHEALTHY: %s" % reason)
        for line in tail_lines(self.train_log):
            self.log("  | %s" % line)

        resume, steps = choose_resume(self.model_dir, self.zip_steps)
        if resume is None:
            self.log("NO RESUME FILE in %s: refusing to start a run from scratch. Stopping." % self.model_dir)
            return "no_resume"
        self.log("resume file: %s (%s steps)" % (resume.name, "{:,}".format(steps)))

        if len(self.recent_restarts(now)) >= self.cfg.max_restarts_per_hour:
            self.log("GIVING UP: %d restarts already in the last hour (--max-restarts-per-hour %d). "
                     "Something restarting cannot fix is wrong; a human should look. Supervisor exiting."
                     % (len(self.restarts), self.cfg.max_restarts_per_hour))
            return "budget"

        if self.cfg.dry_run:
            self.log("[dry-run] would kill %s, relaunch %d games on monitor %d and resume from %s"
                     % ([p.pid for p in trainers] or "nothing", self.cfg.count, self.cfg.monitor, resume.name))
            return "would_restart"

        self.restarts.append(now)  # a failed attempt costs budget too, or a broken launch loops forever
        if trainers:
            roots = [p.pid for p in trainers]
            for pid in descendants(procs, roots) + roots:
                self.log("killing pid %d" % pid)
                self.kill(pid)

        self.log("stopping the games")
        self._stop_games()
        self.sleep(STOP_WAIT_S)
        self.log("launching %d games on monitor %d" % (self.cfg.count, self.cfg.monitor))
        if not self._launch_games(self.cfg.count, self.cfg.monitor):
            self.log("games did not come up; will try again at the next poll")
            return "launch_failed"

        command = shell_command(
            self.cfg.python, "scripts/train.py",
            ["--config", self.cfg.config, "--resume", (self.model_dir / resume.name).as_posix()],
            "%s\\%s_train.log" % (self.cfg.runs_dir, self.cfg.run))
        pid = self.spawn(command, self.cfg.cwd)
        self.log("started trainer (pid %s): %s" % (pid, command))
        self.grace_until = self.now() + self.cfg.start_grace_seconds
        self.ensure_helpers(self.processes())
        return "restarted"

    # -- the helper processes ---------------------------------------------------------------------

    def ensure_helpers(self, procs: list[Proc]) -> list[str]:
        """Starts poll_status.py / keep_best.py when they are not running. Checked every poll: a
        helper can die on its own, and keep_best.py is what protects the weights."""
        started = []
        helpers = [("scripts/poll_status.py", ["--run", self.cfg.run], "%s_poll.log" % self.cfg.run),
                   ("scripts/keep_best.py", ["--run", self.cfg.run, "--metric", "campaign"],
                    "%s_keep_best.log" % self.cfg.run)]
        mine = self_and_ancestors(procs, self.pid)
        for script, args, log_name in helpers:
            name = Path(script).name
            if any(p.pid not in mine and matches_script(p.cmdline, name, self.cfg.run) for p in procs):
                continue
            command = shell_command(self.cfg.python, script, args, "%s\\%s" % (self.cfg.runs_dir, log_name))
            if self.cfg.dry_run:
                self.log("[dry-run] would start %s" % name)
            else:
                self.log("starting %s (pid %s)" % (name, self.spawn(command, self.cfg.cwd)))
            started.append(name)
        return started

    # -- the loop ---------------------------------------------------------------------------------

    def run(self, max_ticks: int | None = None) -> int:
        self.log("supervisor up: run=%s config=%s count=%d monitor=%d stale=%.0fs poll=%.0fs budget=%d/h"
                 % (self.cfg.run, self.cfg.config, self.cfg.count, self.cfg.monitor,
                    self.cfg.stale_seconds, self.cfg.poll_seconds, self.cfg.max_restarts_per_hour))
        self.log("pause with: New-Item %s" % self.pause_path)
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            try:
                action = self.tick()
            except Exception as exc:  # noqa: BLE001 - a supervisor that dies on a hiccup is worse than none
                self.log("tick failed (%s: %s); continuing" % (type(exc).__name__, exc))
                action = "error"
            if action in ("budget", "no_resume"):
                return 1
            if action == "finished":
                return 0
            if self.cfg.dry_run:
                return 0
            self.sleep(self.cfg.poll_seconds)
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Restarts a training run when it dies (see the module docstring).")
    ap.add_argument("--run", default="campaign_gates")
    ap.add_argument("--config", default="configs/campaign_gates_main.yaml")
    ap.add_argument("--count", type=int, default=12, help="game instances to relaunch")
    ap.add_argument("--monitor", type=int, default=1, help="Windows display number for the games")
    ap.add_argument("--stale-seconds", type=float, default=600.0,
                    help="status.json older than this means the trainer is not stepping")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--max-restarts-per-hour", type=int, default=3,
                    help="after this many restarts in an hour the supervisor gives up and exits non-zero")
    ap.add_argument("--start-grace-seconds", type=float, default=900.0,
                    help="quiet period after a restart, while the games load and status.json is still old")
    ap.add_argument("--base-port", type=int, default=47800)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--dry-run", action="store_true", help="report the health decision and exit, changing nothing")
    a = ap.parse_args()

    cfg = Config(run=a.run, config=a.config, count=a.count, monitor=a.monitor,
                 stale_seconds=a.stale_seconds, poll_seconds=a.poll_seconds,
                 max_restarts_per_hour=a.max_restarts_per_hour, start_grace_seconds=a.start_grace_seconds,
                 base_port=a.base_port, python=a.python, cwd=Path.cwd(),
                 runs_dir=a.runs_dir, models_dir=a.models_dir, dry_run=a.dry_run)
    sys.exit(Supervisor(cfg).run())


if __name__ == "__main__":
    main()
