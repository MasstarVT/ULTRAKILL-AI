"""S7 env wiring: the mod handshake, the 0.8 config, the action on the wire, the macro counters, the info keys.

No game:  python tests/test_tech_env.py

`FakeTechLevel` is test_campaign_env's FakeLevel speaking mod 0.8.0: `hello` carries `features`, every frame with
a player carries `move_tech` once `obs_move_tech` has been configured, and a step that asked for a macro gets the
mod's `macro` report back. The v1 tests here are pins: a v1 env's configure call and wire action are today's,
key for key (the key sets below were read off FakeLevel on 2026-09-23, before this change).
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_campaign_env import LEVEL, FakeLevel, action, forward, make_env  # noqa: E402

import campaign_driver  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv, tech_gates  # noqa: E402
from ultrakill_ai.protocol import (  # noqa: E402
    RECOVERABLE,
    TECH_LAYOUT_FEATURES,
    BridgeError,
    BridgeIncompatible,
    mod_features,
)
from ultrakill_ai.spaces import HOOK_INDEX, MACRO_INDEX, MACROS, VARIANT_INDEX  # noqa: E402

V1_CONFIG_KEYS = {
    "block_human_input", "command_timeout_s", "difficulty", "fixed_fps", "frameskip", "ground_ray_length",
    "ground_rays", "horizontal_rays", "max_enemies", "mute", "ray_length", "render", "reset_settle_frames",
    "soft_death", "unlimited_fps", "unlock_all_gear", "window_height", "window_width", "windowed",
}
# The whole v1 `config` line, byte for byte as BridgeClient serializes it, for make_env()'s settings. Captured
# from the code BEFORE this change (2026-09-23, commit 6e5fbf0): the live fleet runs v1 against mod 0.7.2 and
# must keep receiving exactly this on every connect.
V1_CONFIG_LINE = (
    '{"type":"config","frameskip":2,"fixed_fps":30,"unlimited_fps":true,"mute":true,"block_human_input":true,'
    '"reset_settle_frames":10,"command_timeout_s":900,"soft_death":false,"render":false,"windowed":true,'
    '"window_width":640,"window_height":360,"difficulty":-1,"unlock_all_gear":false,"max_enemies":32,'
    '"horizontal_rays":16,"ground_rays":8,"ray_length":50.0,"ground_ray_length":30.0}'
)
V1_HELLO_LINE = '{"type":"hello","protocol":1}'
V1_RELEASE_LINE = '{"type":"release"}'
TECH_CONFIG_KEYS = {
    "obs_move_tech", "obs_weapon_tech", "obs_projectiles", "obs_input_clock", "macros", "macro_ssj_wall",
    "allow_reserved_macros", "macro_wall_lead_unsafe", "variant_switching", "ssj_gap_s", "ssj_indicator",
}
V1_WIRE_KEYS = {"move", "buttons", "slot", "look"}
V08_FEATURES = ("monotonic_input_clock", "macro.ssj", "macro.ssj_wall", "obs.move_tech", "obs.weapon_tech",
                "obs.projectiles", "action.variant", "ssj_instrument")
MOVE_TECH = {
    "heavy_fall": False, "slam_force": 1.0, "bounce_window": False, "coyote": 0.0, "wall_jumps": 0,
    "wall_available": False, "boost": False, "boost_left": 0.0, "pre_slide_speed": 0.5, "jump_cooldown": False,
    "slide_grace": 0.0, "riding_rocket": False, "slide_since": 3.2, "slide_timestamp": 101.5,
    "jump_timestamp": 99.0,
}
EXPECTED_A = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 / 3.0, 0.0, 0.0, 0.0]
LANDED = {"result": "ran", "reason": None, "note": None, "ssj_bucket": 1, "ssj_landed": True}
REFUSED = {"result": "refused", "reason": "not_sliding", "note": None, "ssj_bucket": -1, "ssj_landed": False}
OFF_08 = {"obs_move_tech": False, "obs_weapon_tech": False, "obs_projectiles": False, "obs_input_clock": False,
          "macros": True, "macro_ssj_wall": False, "allow_reserved_macros": False, "macro_wall_lead_unsafe": False,
          "variant_switching": False, "ssj_gap_s": 0.012, "ssj_indicator": False}


class FakeTechLevel(FakeLevel):
    """FakeLevel speaking mod 0.8.0 (see the module docstring)."""

    def __init__(self, features=V08_FEATURES, mod_version: str = "0.8.0", emit_move_tech: bool = True):
        super().__init__()
        self.features = list(features)
        self.mod_version = mod_version
        self.emit_move_tech = emit_move_tech
        self.move_tech = dict(MOVE_TECH)
        self.macro_outcome = dict(LANDED)

    def connect(self, retry_seconds: float = 60.0) -> dict:
        self.connects += 1
        return {"type": "hello", "protocol": 1, "mod_version": self.mod_version, "scene": LEVEL,
                "features": list(self.features)}

    def step(self, action: dict) -> dict:
        obs = super().step(action)
        macro = action.get("macro")
        if macro:
            obs["macro"] = {"requested": MACROS[macro], **self.macro_outcome}
        return obs

    def _obs(self, event=None) -> dict:
        obs = super()._obs(event)
        if self.emit_move_tech and obs.get("player") and self.settings.get("obs_move_tech"):
            obs["move_tech"] = dict(self.move_tech)
        return obs


def tech_env(level_cls=FakeTechLevel, **overrides):
    env, _ = make_env(**{"tech_layout": "v2", **overrides})
    env.client = level_cls()
    return env, env.client


def tech_action(*, macro: int = 0, variant: int = 0, hook: int = 0, move: bool = True, buttons=()):
    a = np.zeros(15, dtype=np.int64)
    a[:12] = action(move=move, buttons=tuple(buttons))
    a[MACRO_INDEX], a[VARIANT_INDEX], a[HOOK_INDEX] = macro, variant, hook
    return a


class LoopbackMod:
    """A real loopback socket speaking just enough of the bridge for a connect: it answers `hello` with the
    given hello and anything else with `ok`, and records every line the client sent, verbatim. It binds port 0
    -- an ephemeral port, never a training bridge port (47800-47812)."""

    def __init__(self, hello: dict):
        self.hello = hello
        self.lines: list[str] = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        assert not 47800 <= self.port <= 47812, self.port
        self._conn = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self.server.settimeout(20.0)
        try:
            self._conn, _ = self.server.accept()
            self._conn.settimeout(20.0)
            buf = b""
            while True:
                chunk = self._conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8")
                    self.lines.append(line)
                    reply = self.hello if json.loads(line).get("type") == "hello" else {"type": "ok"}
                    self._conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        except OSError:
            pass

    def close(self) -> None:
        for sock in (self._conn, self.server):
            if sock is not None:
                sock.close()
        self._thread.join(timeout=10.0)


def connect_over_loopback(hello: dict, **overrides) -> tuple[list[str], BaseException | None, bool]:
    """Runs a real env's connect and then its ordinary close -- the real BridgeClient, real JSON on a real
    socket -- against `hello`. Returns every line the mod received, the exception the connect raised (if any),
    and whether the connect left the client's socket open."""
    mod = LoopbackMod(hello)
    env = UltrakillEnv(EnvConfig(mode="campaign", level=LEVEL, fixed_fps=30, frameskip=2, port=mod.port,
                                 **overrides))
    env.client.handshake_timeout = 10.0
    raised = None
    try:
        env._ensure_connected()
    except Exception as exc:  # noqa: BLE001 - handed back to the test, which asserts on its type
        raised = exc
    left_open = env.client._sock is not None
    try:
        env.close()
    finally:
        env.client._drop()  # only ever does anything if close() did not hang up, so a failing test cannot leak
        # The client has hung up, so the server thread has read everything it will ever get before join returns.
        mod.close()
    return list(mod.lines), raised, left_open


# ---------------------------------------------------------------------------------------------
# Task 2: the handshake and the config
# ---------------------------------------------------------------------------------------------


def test_a_v1_env_sends_todays_configure_call_exactly():
    env, fake = make_env()
    try:
        env.reset()
        assert set(fake.settings) == V1_CONFIG_KEYS, sorted(set(fake.settings) ^ V1_CONFIG_KEYS)
        # Byte for byte, key ORDER included, as BridgeClient.configure puts it on the wire.
        line = json.dumps({"type": "config", **fake.settings}, separators=(",", ":"))
        assert line == V1_CONFIG_LINE, line
    finally:
        env.close()


def test_the_connect_bytes_on_a_real_socket():
    """The same pins through the REAL client: hello, the features check and configure, as JSON on a socket."""
    hello_072 = {"type": "hello", "protocol": 1, "mod_version": "0.7.2", "scene": LEVEL}
    hello_080 = {**hello_072, "mod_version": "0.8.0", "features": list(V08_FEATURES)}

    # v1 against 0.7.2 -- the live fleet today: exactly today's lines, the close's release included.
    lines, raised, _ = connect_over_loopback(hello_072)
    assert raised is None, raised
    assert lines == [V1_HELLO_LINE, V1_CONFIG_LINE, V1_RELEASE_LINE], lines

    # v1 against 0.8.0: today's config line, then every 0.8 switch appended, OFF.
    lines, raised, _ = connect_over_loopback(hello_080)
    assert raised is None, raised
    assert len(lines) == 3 and lines[0] == V1_HELLO_LINE and lines[2] == V1_RELEASE_LINE, lines
    assert lines[1].startswith(V1_CONFIG_LINE[:-1] + ","), lines[1]
    sent = json.loads(lines[1])
    assert set(sent) == {"type"} | V1_CONFIG_KEYS | TECH_CONFIG_KEYS
    assert {k: sent[k] for k in TECH_CONFIG_KEYS} == OFF_08

    # v2 against 0.7.2: the hello and nothing after it -- the refusal comes before any configure, and it hangs
    # up rather than leave an open socket behind `_connected = False`, which close() would skip.
    lines, raised, left_open = connect_over_loopback(hello_072, tech_layout="v2")
    assert isinstance(raised, BridgeIncompatible), repr(raised)
    assert not left_open, "a refused mod's socket must be dropped before the error is raised"
    assert lines == [V1_HELLO_LINE], lines


def test_a_v1_env_on_a_0_8_dll_switches_every_0_8_feature_off():
    env, _ = make_env()
    env.client = fake = FakeTechLevel()
    try:
        obs, _ = env.reset()
        assert set(fake.settings) == V1_CONFIG_KEYS | TECH_CONFIG_KEYS
        assert {k: fake.settings[k] for k in TECH_CONFIG_KEYS} == OFF_08
        assert obs.shape == (479,) and "move_tech" not in env._raw
    finally:
        env.close()


def test_a_v2_env_on_a_0_7_2_mod_fails_loudly_at_connect():
    env, fake = make_env(tech_layout="v2")  # FakeLevel: mod 0.5.0, no `features` at all
    try:
        env.reset()
    except BridgeIncompatible as exc:
        text = str(exc)
        for feature in TECH_LAYOUT_FEATURES:
            assert feature in text, text
        assert "0.5.0" in text and "tech_layout" in text
    else:
        raise AssertionError("a v2 env must refuse a mod without the 0.8.0 features")
    finally:
        env.close()
    assert fake.configures == 0, "nothing may be configured on a mod that cannot serve the layout"
    assert fake.drops == 1, "the refused connection is hung up, not left open"


def test_the_mod_is_checked_on_every_connect_not_only_the_first():
    """A game relaunched onto an old DLL, or a DLL swapped under a running fleet, must be refused at the
    reconnect too: a check at the first connect alone would train v2 against a mod that no longer serves it."""
    env, fake = tech_env()
    try:
        env.reset()
        assert fake.configures == 1
        fake.features, fake.mod_version = [], "0.7.2"
        try:
            env._try_reconnect_until(time.monotonic() + 30.0)
        except BridgeIncompatible as exc:
            assert "0.7.2" in str(exc), exc
        else:
            raise AssertionError("a reconnect onto a mod without the 0.8.0 features was accepted")
        assert fake.configures == 1, "the refused mod was configured"
        assert fake.drops == 1 and not env._connected
    finally:
        env.close()


def test_a_memory_recycle_onto_an_old_dll_is_refused_loudly():
    """`_recycle_own_game_if_fat` swallows every other failure; this one must escape it, not be logged as a
    memory error and latch the recycle off."""
    env, fake = tech_env(bridge_relaunch=True, game_memory_growth_gb=1.0)
    # Both hooks are stubbed BEFORE the first reset: the real ones read the live fleet's memory and relaunch a
    # real game on this port.
    fleet = {env.cfg.port: int(1.4e9), env.cfg.port + 1: int(1.4e9)}
    env._fleet_memory_hook = lambda: dict(fleet)

    def relaunch_onto_an_old_dll(deadline):
        fake.features, fake.mod_version = [], "0.7.2"
        fleet[env.cfg.port] = int(1.4e9)
        return True

    env._relaunch_own_game = relaunch_onto_an_old_dll
    events: list[str] = []
    try:
        env.reset()  # a healthy fleet: nothing is recycled
        assert fake.mod_version == "0.8.0"
        env.envlog.event = lambda name, **_: events.append(name)
        fleet[env.cfg.port] = int(6e9)
        try:
            env._recycle_own_game_if_fat(time.monotonic() + 30.0)
        except BridgeIncompatible:
            pass
        else:
            raise AssertionError("the recycle swallowed an incompatible mod")
        assert "mem_recycle_error" not in events and "mod_incompatible" in events, events
        assert env._mem_recycle_off is False
    finally:
        env.close()


def test_bridge_incompatible_escapes_every_recovery_handler():
    assert not issubclass(BridgeIncompatible, BridgeError), "the reconnect ladder retries every BridgeError"
    assert not issubclass(BridgeIncompatible, OSError)
    assert not any(issubclass(BridgeIncompatible, kind) for kind in RECOVERABLE)
    env, _ = make_env(tech_layout="v2")
    try:
        env._try_reconnect_until(time.monotonic() + 30.0)
    except BridgeIncompatible:
        pass
    else:
        raise AssertionError("the reconnect ladder swallowed an incompatible mod instead of ending the worker")
    finally:
        env.close()


def test_a_v2_env_on_a_0_8_mod_sends_the_tech_config():
    env, fake = tech_env()
    try:
        obs, _ = env.reset()
        assert obs.shape == (530,) and len(env.action_space.nvec) == 15
        assert {k: fake.settings[k] for k in TECH_CONFIG_KEYS} == {**OFF_08, "obs_move_tech": True}
    finally:
        env.close()


def test_the_gates_reach_the_mod_config():
    env, fake = tech_env(macro_ssj_wall=True, variant_switching=True)
    try:
        env.reset()
        assert fake.settings["macro_ssj_wall"] is True and fake.settings["variant_switching"] is True
        assert fake.settings["allow_reserved_macros"] is False, "3-5 have no switch: they are not built"
    finally:
        env.close()


def test_mod_features_reads_the_hello_array():
    assert mod_features({"features": ["a", 3, "b"]}) == frozenset({"a", "b"})
    assert mod_features({"protocol": 1}) == frozenset()
    assert mod_features(None) == frozenset()
    assert mod_features({"features": "obs.move_tech"}) == frozenset(), "a string is not an array"


def test_the_tech_gates_follow_the_config():
    assert tech_gates(EnvConfig()) == (frozenset({1}), False, False), "S7: M1 alone"
    assert tech_gates(EnvConfig(macro_ssj=False)).macros == frozenset()
    assert tech_gates(EnvConfig(macro_ssj_wall=True, variant_switching=True, hook_action=True)) == (
        frozenset({1, 2}), True, True)


def test_tech_layout_is_validated_at_init():
    for cfg in (EnvConfig(mode="campaign", tech_layout="v3"), EnvConfig(mode="cybergrind", tech_layout="v2")):
        try:
            UltrakillEnv(cfg).close()
        except ValueError as exc:
            assert "tech_layout" in str(exc)
        else:
            raise AssertionError(f"accepted {cfg.mode} / {cfg.tech_layout}")


def test_the_tech_fields_round_trip_through_a_config_file():
    cfg = EnvConfig(mode="campaign", tech_layout="v2", macro_ssj=True, macro_ssj_wall=False,
                    variant_switching=False, hook_action=True)
    back = EnvConfig.from_dict(cfg.to_dict())
    assert (back.tech_layout, back.macro_ssj, back.macro_ssj_wall, back.variant_switching, back.hook_action) == (
        "v2", True, False, False, True)


def test_the_plan_validator_accepts_the_tech_fields_and_refuses_a_bad_layout():
    assert campaign_driver._speed_env("plan.yaml", {"tech_layout": "v2", "macro_ssj": True}) == {
        "tech_layout": "v2", "macro_ssj": True}
    for block, needle in (({"tech_layout": "v3"}, "tech_layout"), ({"hook_action": "false"}, "boolean")):
        try:
            campaign_driver._speed_env("plan.yaml", block)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"accepted {block}")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
