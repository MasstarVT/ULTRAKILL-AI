"""Socket client for the UltrakillAIBridge mod (newline-delimited JSON, see docs/protocol.md).

**Every blocking call here is bounded, in every state.** That is not a nicety: a worker blocked on this socket
blocks all twelve games (vectorized envs step in lockstep), and a stall longer than the supervisor's patience
costs a full twelve-game relaunch instead of a local recovery. The 2026-09-17 17:52 freeze is the worked
example -- see the freeze gotcha in CLAUDE.md.

Three bounds used to be wrong, and all three are now pinned by tests:

  - `close()` set `settimeout(5.0)` and then called `recv()`, which set the timeout straight back to
    `self.timeout`. Against a frozen game it blocked 120 s, not 5 s, *per env* -- measured live on the private
    game on 2026-09-17. Twelve of those is a 24-minute teardown.
  - The `hello` handshake was bounded by `self.timeout` (120 s), not by anything handshake-shaped. `connect()`'s
    own 5 s only ever covered the TCP connect.
  - A socket timeout is STICKY. `reset()` asked for its 600 s and left it on the socket, so the next `send()`
    inherited 600 s. Every request now sets its own bound for both the send and the read.

A bound also has to be CLAMPABLE, not just finite. `reset(timeout=...)` lets the caller ask for less than
`reset_timeout`, which is what makes the env's recovery budget a total rather than a suggestion: the call in
flight when the budget expires ends with the budget instead of adding a full reset on top of it. And a connect
that never succeeds raises `BridgeClosed` (which is RECOVERABLE), not a bare `BridgeError`: a port that is not
listening yet is the ordinary state of a booting game, not a fatal error.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800

# A reset is a full Unity scene load and a step is one frame, so one socket timeout cannot serve both. They used
# to share 120 s -- the SAME number as `EpisodeController.resetTimeoutSeconds`, which is why the mod's own
# "reset timed out" error reply could never arrive: the mod starts its 120 s timer after the client has already
# started its own, so the client always gave up first and the graceful path was dead code. See the timeout
# gotcha in CLAUDE.md.
DEFAULT_STEP_TIMEOUT = 120.0  # deliberately the old shared value: no step has ever timed out, so nothing moves
# The mod gives up on a reset at `EpisodeController.resetTimeoutSeconds` (120 s) and answers with an error, so
# the client only has to outlast that, not out-wait a hypothetical scene load. 180 s does, and unlike the old
# 600 s it leaves room for a whole recovery ladder inside the supervisor's stale window (see env.py's
# `bridge_recovery_budget_s`). 600 s was on its own longer than the supervisor's entire patience, so a single
# unanswered reset always presented as a hung trainer.
DEFAULT_RESET_TIMEOUT = 180.0
DEFAULT_CONNECT_TIMEOUT = 10.0  # one TCP connect attempt, retried until `connect(retry_seconds=...)` runs out
DEFAULT_HANDSHAKE_TIMEOUT = 20.0  # hello and config: the mod answers these without touching the scene
DEFAULT_CLOSE_TIMEOUT = 5.0  # the release handshake on the way out; a teardown may never wait on a sick game
UNKNOWN_SCENE = "unknown scene"

_READ_CHUNK = 65536


class BridgeError(RuntimeError):
    pass


class BridgeTimeout(BridgeError):
    """The game did not answer a request in time. The connection is unusable afterwards (see `_fail`)."""


class BridgeClosed(BridgeError):
    """The connection dropped, or was used after it broke."""


class BridgeSceneUnknown(BridgeError):
    """The mod could not find that scene.

    Almost always a game that is still booting rather than a bad name: `EpisodeController.SceneExists` searches
    `Addressables.ResourceLocators`, which stay empty until boot finishes, while the bridge port is already
    listening. Callers should wait and ask again rather than give up.
    """


# Everything a caller can recover from by rebuilding the connection. `BridgeSceneUnknown` is here too because a
# half-booted game answers every reset with it and the cure is time, not a new exception reaching the trainer.
RECOVERABLE = (BridgeTimeout, BridgeClosed, BridgeSceneUnknown)


class BridgeIncompatible(RuntimeError):
    """The mod on the other end cannot serve what this client is configured for -- e.g. `tech_layout: v2` against
    a 0.7.2 DLL, which would otherwise train 530 inputs of which 51 are always zero.

    DELIBERATELY NOT a BridgeError and NOT in RECOVERABLE. `UltrakillEnv`'s recovery ladder retries every
    BridgeError for up to `bridge_recovery_budget_s` (~540 s) and its last rung relaunches the env's own game;
    against a wrong DLL both are wasted, since no retry can change what the DLL serves. This escapes every env
    handler instead, so the worker dies at once with this message in runs/<run>_train.log.

    The worker dying does not stop the FLEET: its SubprocVecEnv parent sees EOFError and the trainer exits, and
    both restart paths would start it again -- the driver's `ensure_trainer` at its next poll (no budget), and
    `supervise.Supervisor.restart` by stopping and relaunching all twelve games (up to `max_restarts_per_hour`).
    What stops the fleet is the file the refusal leaves behind: before raising, the env writes
    `runs/<run>/MOD_INCOMPATIBLE` (`MOD_INCOMPATIBLE_FILE`; only a training run has a run directory -- the
    `env_log_dir` train.py alone fills), and both paths read it before any trainer (re)start. While it exists
    neither relaunches a game or a trainer, and both exit with code 4 (`supervise.EXIT_MOD_INCOMPATIBLE`).
    NOTHING deletes it but the operator, after installing the mod. The driver's layout guard (the plan's Task 5)
    is the other half: it refuses a checkpoint whose shapes do not match `tech_layout`, without reading the DLL.
    """


# What a `tech_layout: v2` client needs in `hello.features` (docs/protocol.md). A 0.7.x DLL sends no such array.
TECH_LAYOUT_FEATURES = ("monotonic_input_clock", "macro.ssj", "obs.move_tech")


# The fleet-level half of BridgeIncompatible: `runs/<run>/MOD_INCOMPATIBLE`. Written by the refusing env, read by
# `supervise.Supervisor.restart`, `campaign_driver.Driver.ensure_trainer` and `check_run.py`, removed by the
# OPERATOR ONLY -- after installing the mod (or setting tech_layout back to v1). A file that cleared itself would
# re-arm the very restart loop it exists to stop. Stdlib only: the driver and the supervisor read it.
MOD_INCOMPATIBLE_FILE = "MOD_INCOMPATIBLE"


def write_mod_incompatible(run_dir: str | os.PathLike[str], text: str) -> Path | None:
    """Writes `<run_dir>/MOD_INCOMPATIBLE` atomically and returns its path, or None when it could not be written.

    Twelve workers refuse the same DLL within a second of each other, so each writes its own temp file and
    `os.replace`s it in: a reader sees one whole report, never two interleaved. A replace that loses the race (or
    any other OSError) is swallowed -- the caller raises BridgeIncompatible either way, and one winner is enough.
    """
    path = Path(run_dir) / MOD_INCOMPATIBLE_FILE
    tmp = path.with_name("%s.%d.tmp" % (MOD_INCOMPATIBLE_FILE, os.getpid()))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    return path


def read_mod_incompatible(run_dir: str | os.PathLike[str]) -> str | None:
    """The text of `<run_dir>/MOD_INCOMPATIBLE`, or None when there is no such file.

    PRESENCE is the signal: a file that exists but cannot be read right now (a writer's replace in flight) still
    refuses, with a placeholder text, rather than reading as absent.
    """
    path = Path(run_dir) / MOD_INCOMPATIBLE_FILE
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return "(%s exists but could not be read: %s: %s)" % (path, type(exc).__name__, exc)


def mod_features(hello: dict[str, Any] | None) -> frozenset[str]:
    """`hello.features` as a set of strings; missing (a 0.7.x DLL) or malformed reads as the empty set."""
    raw = (hello or {}).get("features")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(f for f in raw if isinstance(f, str))


class _LineReader:
    """Newline-delimited reads off a socket with a WALL-CLOCK bound, not a per-syscall one.

    `socket.makefile().readline()` restarts the socket timeout on every underlying `recv`, so a peer that
    dribbles one byte just inside the timeout keeps a "bounded" read alive indefinitely. This keeps its own
    deadline and re-arms the socket with whatever is left of it, so `readline(20)` cannot take 21 seconds.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = b""

    def readline(self, bound: float) -> str:
        deadline = time.monotonic() + max(0.0, bound)
        while True:
            newline = self._buf.find(b"\n")
            if newline >= 0:
                line, self._buf = self._buf[:newline + 1], self._buf[newline + 1:]
                return line.decode("utf-8")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out")
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(_READ_CHUNK)
            if not chunk:
                self._buf = b""  # a half line before a hang-up is still a hang-up, never a message
                return ""
            self._buf += chunk

    def close(self) -> None:
        self._buf = b""


class BridgeClient:
    """One TCP connection to one game's bridge.

    `timeout` covers ordinary requests (step, get_obs); `reset_timeout` covers `reset`, which blocks while the
    game loads a scene; `handshake_timeout` covers hello and config; `close_timeout` covers the release on the
    way out. A request that times out or drops marks the client broken and every later request raises until
    `connect()` is called again: a late reply to the timed-out request is still sitting in the socket, and
    reusing the stream would hand it back as the answer to the NEXT request, leaving every observation from
    then on one request stale.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = DEFAULT_STEP_TIMEOUT, reset_timeout: float = DEFAULT_RESET_TIMEOUT,
                 handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
                 close_timeout: float = DEFAULT_CLOSE_TIMEOUT,
                 connect_timeout: float = DEFAULT_CONNECT_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.handshake_timeout = handshake_timeout
        self.close_timeout = close_timeout
        self.connect_timeout = connect_timeout
        self.broken = False
        self._sock: socket.socket | None = None
        self._reader = None

    def connect(self, retry_seconds: float = 60.0) -> dict[str, Any]:
        """Connects (retrying while the game starts up) and performs the hello handshake.

        Bounded by `retry_seconds` for the TCP connect plus `handshake_timeout` for the hello. Nothing here
        may wait on `self.timeout`: a game that accepts the socket and then answers nothing is exactly the
        state a recovery is trying to escape.
        """
        deadline = time.monotonic() + retry_seconds
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
                break
            except OSError:
                if time.monotonic() > deadline:
                    # BridgeClosed, not a bare BridgeError: a port that is not listening yet is the ordinary
                    # state of a game that is still booting, and it is exactly what the recovery ladder exists
                    # to wait out. A bare BridgeError is not in RECOVERABLE, so `reset()` did not catch it and
                    # the worker died -- the 18:08 pathology, where the boot gate gives up on a cold instance
                    # and starts the trainer anyway. See the freeze gotcha in CLAUDE.md.
                    raise BridgeClosed(
                        f"Could not connect to the mod on {self.host}:{self.port}. "
                        "Is ULTRAKILL running with BepInEx and UltrakillAIBridge installed?"
                    )
                time.sleep(1.0)

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._reader = _LineReader(sock)
        self.broken = False  # a reconnect is the one thing that clears it

        hello = self.request({"type": "hello", "protocol": PROTOCOL_VERSION}, timeout=self.handshake_timeout)
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(f"Protocol mismatch: mod speaks {hello.get('protocol')}, client speaks {PROTOCOL_VERSION}")
        return hello

    def close(self) -> None:
        """Releases control and drops the socket, in at most `close_timeout`.

        The release reply is worth a short wait -- it hands control back so the game stops being driven -- but
        never more than that. A teardown that waits on a sick game is how a crashed trainer becomes a hung one.
        """
        if self._sock is None:
            return
        if not self.broken:
            try:
                # Skipped on a broken client: `request` would raise straight away, and there is nothing to hand
                # control back to.
                self.request({"type": "release"}, timeout=self.close_timeout)
            except (OSError, BridgeError, ValueError):
                pass
        self._drop()

    def _drop(self) -> None:
        """Closes the socket without the release handshake."""
        try:
            if self._reader is not None:
                self._reader.close()
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        finally:
            self._sock = None
            self._reader = None

    def _fail(self, exc: BridgeError) -> BridgeError:
        """Marks the connection unusable and returns the error to raise.

        Everything after a timeout or a drop is poisoned: the game may still write the reply this client gave
        up on, so the next `recv` would return it instead of the answer to the next request.
        """
        self.broken = True
        return exc

    def _arm(self, bound: float) -> None:
        """Puts `bound` on the socket for the call about to be made.

        Every send and every read arms its own bound. A socket timeout is sticky, so without this a `reset`'s
        long bound stayed on the socket and the next `send` inherited it.
        """
        if self._sock is not None:
            self._sock.settimeout(bound)

    def send(self, msg: dict[str, Any], timeout: float | None = None) -> None:
        if self._sock is None:
            raise BridgeClosed("Not connected")
        if self.broken:
            raise BridgeClosed("Bridge connection is broken; reconnect before using it again")
        bound = self.timeout if timeout is None else timeout
        try:
            self._arm(bound)
            self._sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8"))
        except TimeoutError as exc:
            raise self._fail(BridgeTimeout(f"timed out sending to {self.host}:{self.port}")) from exc
        except OSError as exc:
            raise self._fail(BridgeClosed(f"send to {self.host}:{self.port} failed: {exc}")) from exc

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        if self._reader is None or self._sock is None:
            raise BridgeClosed("Not connected")
        if self.broken:
            raise BridgeClosed("Bridge connection is broken; reconnect before using it again")
        bound = self.timeout if timeout is None else timeout
        try:
            self._arm(bound)
            line = self._readline(bound)
        except TimeoutError as exc:
            # socket.timeout is TimeoutError from 3.10 on, so this catches both spellings.
            raise self._fail(BridgeTimeout(
                f"no reply from {self.host}:{self.port} within {bound:.0f}s")) from exc
        except OSError as exc:
            raise self._fail(BridgeClosed(f"read from {self.host}:{self.port} failed: {exc}")) from exc
        if not line:
            raise self._fail(BridgeClosed("Connection closed by the game"))
        msg = json.loads(line)
        if msg.get("type") == "error":
            message = msg.get("message", "unknown error from mod")
            # An error reply leaves the stream in step, so the connection is NOT marked broken here.
            if message.startswith(UNKNOWN_SCENE):
                raise BridgeSceneUnknown(message)
            raise BridgeError(message)
        return msg

    def _readline(self, bound: float) -> str:
        """`_LineReader` gets the wall-clock bound; any other file-like reader (the tests' fake) does not."""
        reader = self._reader
        if isinstance(reader, _LineReader):
            return reader.readline(bound)
        return reader.readline()

    def request(self, msg: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """One round trip. `timeout` bounds the send AND the read, so a request can never outlive twice it."""
        self.send(msg, timeout=timeout)
        return self.recv(timeout)

    # Convenience wrappers -------------------------------------------------

    def configure(self, **settings: Any) -> None:
        self.request({"type": "config", **settings}, timeout=self.handshake_timeout)

    def get_obs(self) -> dict[str, Any]:
        return self.request({"type": "get_obs"})

    def reset(self, scene: str | None = None, checkpoint: bool = False,
              timeout: float | None = None) -> dict[str, Any]:
        """Loads a scene (or respawns at the checkpoint). Waits `reset_timeout`, not the step timeout: this
        blocks on a Unity scene load, which with a dozen games loading at once is minutes, not frames.

        `timeout` lets a caller ask for LESS than `reset_timeout`. A recovery ladder clamps each call to what
        is left of its wall-clock budget, so the call in flight when the budget expires ends with it instead of
        overshooting it by a whole reset -- the difference between a bounded recovery and a hung trainer.
        """
        msg: dict[str, Any] = {"type": "reset", "checkpoint": checkpoint}
        if scene:
            msg["scene"] = scene
        return self.request(msg, timeout=self.reset_timeout if timeout is None else timeout)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.request({"type": "step", "action": action})

    def teleport(self, pos) -> dict[str, Any]:
        return self.request({"type": "teleport", "pos": [float(v) for v in pos]})

    def kill(self) -> dict[str, Any]:
        """Kills the player (debug command for the in-game death check). With soft_death on, the mod heals instead."""
        return self.request({"type": "kill"})

    def release(self) -> None:
        self.request({"type": "release"}, timeout=self.close_timeout)

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
