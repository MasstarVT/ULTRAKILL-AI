"""Socket client for the UltrakillAIBridge mod (newline-delimited JSON, see docs/protocol.md)."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
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

        hello = self.request({"type": "hello", "protocol": PROTOCOL_VERSION})
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(f"Protocol mismatch: mod speaks {hello.get('protocol')}, client speaks {PROTOCOL_VERSION}")
        return hello

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self.send({"type": "release"})
        except OSError:
            pass
        try:
            self._reader.close()
            self._sock.close()
        finally:
            self._sock = None
            self._reader = None

    def send(self, msg: dict[str, Any]) -> None:
        if self._sock is None:
            raise BridgeError("Not connected")
        self._sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8"))

    def recv(self) -> dict[str, Any]:
        if self._reader is None:
            raise BridgeError("Not connected")
        line = self._reader.readline()
        if not line:
            raise BridgeError("Connection closed by the game")
        msg = json.loads(line)
        if msg.get("type") == "error":
            raise BridgeError(msg.get("message", "unknown error from mod"))
        return msg

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.send(msg)
        return self.recv()

    # Convenience wrappers -------------------------------------------------

    def configure(self, **settings: Any) -> None:
        self.request({"type": "config", **settings})

    def get_obs(self) -> dict[str, Any]:
        return self.request({"type": "get_obs"})

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict[str, Any]:
        msg: dict[str, Any] = {"type": "reset", "checkpoint": checkpoint}
        if scene:
            msg["scene"] = scene
        return self.request(msg)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.request({"type": "step", "action": action})

    def release(self) -> None:
        self.request({"type": "release"})

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
