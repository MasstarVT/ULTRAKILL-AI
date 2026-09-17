"""Socket client for the UltrakillAIBridge mod (newline-delimited JSON, see docs/protocol.md)."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800

# A reset is a full Unity scene load and a step is one frame, so one socket timeout cannot serve both. They used
# to share 120 s -- the SAME number as `EpisodeController.resetTimeoutSeconds`, which is why the mod's own
# "reset timed out" error reply could never arrive: the mod starts its 120 s timer after the client has already
# started its own, so the client always gave up first and the graceful path was dead code. See the timeout
# gotcha in CLAUDE.md.
DEFAULT_STEP_TIMEOUT = 120.0  # deliberately the old shared value: no step has ever timed out, so nothing moves
DEFAULT_RESET_TIMEOUT = 600.0  # a scene load with a dozen games loading at once, GC and a window losing the GPU
UNKNOWN_SCENE = "unknown scene"


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


class BridgeClient:
    """One TCP connection to one game's bridge.

    `timeout` covers ordinary requests (step, get_obs, config); `reset_timeout` covers `reset`, which blocks
    while the game loads a scene. A request that times out or drops marks the client broken and every later
    request raises until `connect()` is called again: a late reply to the timed-out request is still sitting in
    the socket, and reusing the stream would hand it back as the answer to the NEXT request, leaving every
    observation from then on one request stale.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = DEFAULT_STEP_TIMEOUT, reset_timeout: float = DEFAULT_RESET_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.broken = False
        self._sock: socket.socket | None = None
        self._reader = None

    def connect(self, retry_seconds: float = 60.0) -> dict[str, Any]:
        """Connects (retrying while the game starts up) and performs the hello handshake."""
        deadline = time.monotonic() + retry_seconds
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise BridgeError(
                        f"Could not connect to the mod on {self.host}:{self.port}. "
                        "Is ULTRAKILL running with BepInEx and UltrakillAIBridge installed?"
                    )
                time.sleep(1.0)

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self.broken = False  # a reconnect is the one thing that clears it

        hello = self.request({"type": "hello", "protocol": PROTOCOL_VERSION})
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(f"Protocol mismatch: mod speaks {hello.get('protocol')}, client speaks {PROTOCOL_VERSION}")
        return hello

    def close(self) -> None:
        if self._sock is None:
            return
        if not self.broken:
            try:
                # Wait for the reply so control is released before the socket closes. Skipped on a broken
                # client: `request` would raise straight away, and there is nothing to hand control back to.
                self._sock.settimeout(5.0)
                self.request({"type": "release"})
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

    def send(self, msg: dict[str, Any]) -> None:
        if self._sock is None:
            raise BridgeClosed("Not connected")
        if self.broken:
            raise BridgeClosed("Bridge connection is broken; reconnect before using it again")
        try:
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
        try:
            self._sock.settimeout(self.timeout if timeout is None else timeout)
            line = self._reader.readline()
        except TimeoutError as exc:
            # socket.timeout is TimeoutError from 3.10 on, so this catches both spellings.
            raise self._fail(BridgeTimeout(
                f"no reply from {self.host}:{self.port} within "
                f"{self.timeout if timeout is None else timeout:.0f}s")) from exc
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

    def request(self, msg: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.send(msg)
        return self.recv(timeout)

    # Convenience wrappers -------------------------------------------------

    def configure(self, **settings: Any) -> None:
        self.request({"type": "config", **settings})

    def get_obs(self) -> dict[str, Any]:
        return self.request({"type": "get_obs"})

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict[str, Any]:
        """Loads a scene (or respawns at the checkpoint). Waits `reset_timeout`, not the step timeout: this
        blocks on a Unity scene load, which with a dozen games loading at once is minutes, not frames."""
        msg: dict[str, Any] = {"type": "reset", "checkpoint": checkpoint}
        if scene:
            msg["scene"] = scene
        return self.request(msg, timeout=self.reset_timeout)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.request({"type": "step", "action": action})

    def teleport(self, pos) -> dict[str, Any]:
        return self.request({"type": "teleport", "pos": [float(v) for v in pos]})

    def kill(self) -> dict[str, Any]:
        """Kills the player (debug command for the in-game death check). With soft_death on, the mod heals instead."""
        return self.request({"type": "kill"})

    def release(self) -> None:
        self.request({"type": "release"})

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
