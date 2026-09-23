"""S7 env wiring: the mod handshake, the 0.8 config, the action on the wire, the macro counters, the info keys.

No game:  python tests/test_tech_env.py

`FakeTechLevel` is test_campaign_env's FakeLevel speaking mod 0.8.0: `hello` carries `features`, every frame with
a player carries `move_tech` once `obs_move_tech` has been configured, and a step that asked for a macro gets the
mod's `macro` report back. The v1 tests here are pins: a v1 env's configure call and wire action are today's,
key for key (the key sets below were read off FakeLevel on 2026-09-23, before this change).
"""

from __future__ import annotations

import functools
import hashlib
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
    BridgeClosed,
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


# ---------------------------------------------------------------------------------------------
# Task 3: the wire, block A / D through the env, the counters, info, status.json
# ---------------------------------------------------------------------------------------------


def test_a_v1_step_puts_todays_keys_on_the_wire():
    env, fake = make_env()
    try:
        env.reset()
        env.step(forward())
        assert set(fake.last_action) == V1_WIRE_KEYS
    finally:
        env.close()


def test_v2_forwards_the_ssj_macro_and_masks_every_reserved_value():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(macro=1))
        assert fake.last_action["macro"] == 1
        for value in (0, 2, 3, 4, 5):
            env.step(tech_action(macro=value))
            assert set(fake.last_action) == V1_WIRE_KEYS, value
    finally:
        env.close()


def test_v2_masks_variant_and_hook_until_their_stages_open():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(variant=2, hook=1))
        assert set(fake.last_action) == V1_WIRE_KEYS and "hook" not in fake.last_action["buttons"]
    finally:
        env.close()
    env, fake = tech_env(variant_switching=True, hook_action=True, macro_ssj_wall=True)
    try:
        env.reset()
        env.step(tech_action(macro=2, variant=2, hook=1))
        assert fake.last_action["macro"] == 2 and fake.last_action["variant"] == 2
        assert "hook" in fake.last_action["buttons"]
    finally:
        env.close()


def test_block_a_and_block_d_are_packed_from_the_mod_reports():
    env, fake = tech_env()
    try:
        obs, _ = env.reset()
        assert np.allclose(obs[479:491], EXPECTED_A, atol=1e-6)
        obs, *_ = env.step(tech_action(macro=1))
        assert np.allclose(obs[519:522], [1.0, 0.0, 1.0 / 3.0], atol=1e-6)
        obs, *_ = env.step(tech_action())
        assert not obs[519:522].any(), "no macro sent, no report: zeros"
        fake.macro_outcome = dict(REFUSED)
        obs, *_ = env.step(tech_action(macro=1))
        assert np.allclose(obs[519:522], [0.0, 1.0, 0.0])
        assert not obs[491:519].any() and not obs[522:530].any()
    finally:
        env.close()


def test_the_macro_report_survives_an_input_lock_after_the_step():
    env, fake = tech_env()
    try:
        env.reset()
        fake.lock_steps = 3
        before = fake.steps
        obs, *_ = env.step(tech_action(macro=1))
        assert fake.steps - before > 1, "the env stepped through the lock"
        assert obs[519] == 1.0, "block D kept the macro step's own report"
        assert env._behaviour["macro_ran"] == 1
    finally:
        env.close()


def test_the_counters_and_the_v2_info_keys():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(macro=1))  # sent, ran, landed
        fake.macro_outcome = dict(REFUSED)
        env.step(tech_action(macro=1))  # sent, refused: not_sliding
        env.step(tech_action(macro=4))  # requested, masked
        _, _, _, _, info = env.step(tech_action(variant=1, hook=1))
        assert info["macro_request_frac"] == 3 / 4 and info["macro_sent_frac"] == 2 / 4
        assert info["macro_ran_frac"] == 1 / 4 and info["macro_refused_frac"] == 1 / 4
        assert info["macro_landed_frac"] == 1 / 4 and info["macro_ran_share"] == 1 / 2
        assert info["variant_request_frac"] == 1 / 4 and info["hook_request_frac"] == 1 / 4
        assert info["macro_refusal_reasons"] == {"not_sliding": 1}
    finally:
        env.close()


def test_a_v1_info_carries_no_tech_keys():
    env, _ = make_env()
    try:
        env.reset()
        _, _, _, _, info = env.step(forward())
        assert not [k for k in info if k.startswith(("macro_", "variant_request", "hook_request"))]
    finally:
        env.close()


def test_a_missing_move_tech_is_warned_once_and_packs_zeros():
    env, _ = tech_env(level_cls=lambda: FakeTechLevel(emit_move_tech=False))
    try:
        obs, _ = env.reset()
        for _ in range(40):
            obs, *_ = env.step(tech_action(move=False))
        assert env._move_tech_warned and not obs[479:491].any()
    finally:
        env.close()


def test_progress_carries_the_macro_channel_only_when_the_env_reports_it():
    from test_progress import episode_info  # noqa: PLC0415

    from ultrakill_ai.progress import ProgressCallback  # noqa: PLC0415

    tech = {"macro_request_frac": 0.1, "macro_sent_frac": 0.02, "macro_ran_frac": 0.005,
            "macro_refused_frac": 0.015, "macro_landed_frac": 0.005, "macro_ran_share": 0.25,
            "variant_request_frac": 0.15, "hook_request_frac": 0.1,
            "macro_refusal_reasons": {"not_sliding": 3}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra in (("v1", {}), ("v2", tech)):
            run = Path(tmp) / name
            cb = ProgressCallback(run / "status.json", 1000, name, 1, update_every_s=0.0)
            cb._on_training_start()
            cb._record_episode(0, {**episode_info("Level 0-1", completed=1, seconds=90.0), **extra})
            cb._write(time.time())
            mean = json.loads((run / "status.json").read_text(encoding="utf-8"))["mean_100"]
            line = json.loads((run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            if extra:
                assert mean["macro_sent_frac"] == 0.02 and line["macro_refusal_reasons"] == {"not_sliding": 3}
            else:
                assert "macro_sent_frac" not in mean and "macro_refusal_reasons" not in line


# The macro channel as the env reports it, for the ProgressCallback tests below.
TECH_INFO = {"macro_request_frac": 0.1, "macro_sent_frac": 0.02, "macro_ran_frac": 0.005,
             "macro_refused_frac": 0.015, "macro_landed_frac": 0.005, "macro_ran_share": 0.25,
             "variant_request_frac": 0.15, "hook_request_frac": 0.1, "macro_refusal_reasons": {}}


def test_macro_ran_share_is_none_when_nothing_was_sent_and_only_senders_are_averaged():
    from test_progress import episode_info  # noqa: PLC0415

    from ultrakill_ai.progress import ProgressCallback  # noqa: PLC0415

    env, _ = tech_env()
    try:
        env.reset()
        _, _, _, _, info = env.step(tech_action(macro=4))  # requested, masked: nothing sent
        assert info["macro_request_frac"] == 1.0 and info["macro_sent_frac"] == 0.0
        assert info["macro_ran_share"] is None, "a no-send episode has no share, not a share of 0"
    finally:
        env.close()

    def window(shares):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "v2"
            cb = ProgressCallback(run / "status.json", 1000, "v2", 1, update_every_s=0.0)
            cb._on_training_start()
            for share in shares:
                cb._record_episode(0, {**episode_info("Level 0-1"), **TECH_INFO, "macro_ran_share": share})
            cb._write(time.time())
            mean = json.loads((run / "status.json").read_text(encoding="utf-8"))["mean_100"]
            lines = [json.loads(s) for s in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        return mean, [line["macro_ran_share"] for line in lines]

    mean, logged = window([None, 0.5, None, 1.0])
    assert mean["macro_ran_share"] == 0.75, "only the two senders are averaged"
    assert logged == [None, 0.5, None, 1.0], "episodes.jsonl records a no-send episode's share as null"
    mean, _ = window([None, None])
    assert "macro_ran_share" not in mean and mean["macro_sent_frac"] == 0.02, mean


def test_a_bridge_fault_inside_the_lock_leaves_every_sent_macro_with_a_verdict():
    env, fake = tech_env(bridge_backoff_s=0.0, bridge_retries=3)
    step = fake.step

    def fault_inside_the_lock(command):
        obs = step(command)
        if command.get("macro"):
            fake.fail_next_steps(BridgeClosed("dropped while the lock was stepped through"))
        return obs

    fake.step = fault_inside_the_lock
    try:
        env.reset()
        fake.lock_steps = 3
        _, _, _, truncated, info = env.step(tech_action(macro=1))
        assert truncated and info["end_reason"] == "bridge_reset", info.get("end_reason")
        b = env._behaviour
        assert b["macro_sent"] == 1 and b["macro_sent"] == b["macro_ran"] + b["macro_refused"], b
        assert info["macro_sent_frac"] == info["macro_ran_frac"] + info["macro_refused_frac"]
    finally:
        env.close()


def test_a_death_step_counts_the_macro_but_the_respawn_frame_packs_no_block_d():
    """Deliberate (the comment at `_respawn` in `_step`): the respawn frame is a new life at the checkpoint, so its
    block D reads "no macro"; the counters, which the S8 gate does not read from the frame, still record it."""
    env, fake = tech_env()
    try:
        env.reset()
        fake.kill_next = True
        obs, *_ = env.step(tech_action(macro=1))
        assert env._deaths == 1, "the step died and respawned"
        assert env._behaviour["macro_ran"] == 1 and env._behaviour["macro_landed"] == 1
        assert not obs[519:522].any(), obs[519:522]
    finally:
        env.close()


def test_an_action_of_the_other_layouts_width_fails_loudly():
    for build, wrong in ((tech_env, lambda: action(move=True)), (make_env, tech_action)):
        env, fake = build()
        try:
            env.reset()
            before = fake.steps
            try:
                env.step(wrong())
            except AssertionError as exc:
                assert "tech_layout" in str(exc), exc
            else:
                raise AssertionError(f"a {env.cfg.tech_layout} env accepted a {len(wrong())}-wide action")
            assert fake.steps == before, "nothing reached the game"
        finally:
            env.close()


def test_a_move_tech_field_sent_as_null_counts_as_missing():
    env, fake = tech_env()
    fake.move_tech["slide_grace"] = None
    events: list[tuple[str, dict]] = []
    env.envlog.event = lambda name, **kw: events.append((name, kw))
    try:
        obs, _ = env.reset()
        for _ in range(40):
            obs, *_ = env.step(tech_action(move=False))
        assert env._move_tech_warned
        warned = [kw for name, kw in events if name == "move_tech_missing"]
        assert len(warned) == 1 and "slide_grace" in warned[0]["what"], warned
        assert obs[489] == 0.0 and np.allclose(obs[479:489], EXPECTED_A[:10], atol=1e-6)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------
# Task 3's v1 pins: the step bytes, the step outputs, the info keys and the status.json / episodes.jsonl keys
# of a v1 env, against the BASE COMMIT 0be539f (the code the live fleet imports). A v1 env's step, info and
# status files may not change by one byte; if one of these fails, the v1 path moved -- find what moved it.
# The constants are what `tests/tools/tech_v1_pins.py` prints when run against the base commit's package (it
# extracts it with `git archive`); run against the working tree, it proves nothing. `v1_pins()` is the one
# computation both use, and a failure names every pin that differs.
# ---------------------------------------------------------------------------------------------

V1_SCRIPT_SEED, V1_SCRIPT_STEPS = 20260923, 400
# `forward()` as BridgeClient.send puts it on the socket.
V1_FORWARD_LINE = '{"type":"step","action":{"move":[0,1],"buttons":[],"slot":0,"look":[0.0,0.0]}}'
V1_WIRE_LINES = 420  # 400 decisions + 20 input-lock skip steps
# every step line of the script, "\n"-joined
V1_WIRE_SHA256 = "9dffae29607482b005bd25ace5ceaa6269ddfbb81832b1297b25b47aad48f8a1"
# every (obs bytes, reward, terminated, truncated, info) of the script, in order
V1_OUTPUTS_SHA256 = "4781c3d7928659e8d26feb0cda80a4ced1453aa910aace33fb94068695ab3a75"
# the sorted union of the script's info keys (79 of them)
V1_INFO_KEYS_SHA256 = "41a187e9ada19713cc2b1af49020a165d9d0c7e2e9288c5a75fcdd76056d725a"
# status.json's top-level, mean_100 (59) and campaign keys, in order, and every episodes.jsonl line's keys
V1_PROGRESS_KEYS_SHA256 = "8178818d96a63d0c11054f144c22242ae35a6a3bb8aeb94307cc12eb1e192112"
V1_PIN_NAMES = ("V1_FORWARD_LINE", "V1_WIRE_LINES", "V1_WIRE_SHA256", "V1_OUTPUTS_SHA256", "V1_INFO_KEYS_SHA256",
                "V1_PROGRESS_KEYS_SHA256")
# What each pin is, for a failure message: which of the step line, the wire, the outputs, or a key set moved.
V1_PIN_WHAT = {"V1_FORWARD_LINE": "the step line for forward()", "V1_WIRE_LINES": "the number of step lines",
               "V1_WIRE_SHA256": "the step lines", "V1_OUTPUTS_SHA256": "the step outputs (obs, reward, info)",
               "V1_INFO_KEYS_SHA256": "the info key set", "V1_PROGRESS_KEYS_SHA256": "the status.json / "
               "episodes.jsonl key sets"}


class WireLevel(FakeLevel):
    """FakeLevel keeping every step line exactly as `BridgeClient.step` / `send` serialize it onto the socket."""

    def __init__(self):
        super().__init__()
        self.wire: list[str] = []

    def step(self, action: dict) -> dict:
        self.wire.append(json.dumps({"type": "step", "action": action}, separators=(",", ":")))
        return super().step(action)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def v1_script() -> dict:
    """A v1 env driven through a fixed seeded script against FakeLevel -- mod 0.5.0, no `features`, so nothing 0.8
    is configured: the live fleet's situation. Random actions over the whole 12-dim space (look modes 1 / 2 and
    slot presses included), input locks, deaths and one-shot kills injected at fixed steps, and a reset after
    every episode end. Returns the wire lines, a digest of every step output, and every info dict. Cached: three
    tests read it, and nothing mutates what it returns."""
    env, _ = make_env()
    env.client = fake = WireLevel()
    rng = np.random.default_rng(V1_SCRIPT_SEED)
    outputs = hashlib.sha256()
    infos: list[dict] = []

    def record(obs, info, *step):
        # `reset_seconds` is a perf_counter reading: the one wall-clock value in an info.
        info = {k: v for k, v in info.items() if k != "reset_seconds"}
        infos.append(info)
        outputs.update(np.asarray(obs, dtype=np.float32).tobytes())
        outputs.update(json.dumps([*step, info], separators=(",", ":"), default=str).encode("utf-8"))

    try:
        record(*env.reset())
        for i in range(V1_SCRIPT_STEPS):
            if i % 41 == 7:
                fake.lock_steps = 2
            if i % 67 == 30:
                fake.kill_next = True
            if i % 53 == 11:
                fake.kill_enemy_next = True
            obs, reward, terminated, truncated, info = env.step(rng.integers(0, env.action_space.nvec))
            record(obs, info, reward, terminated, truncated)
            if terminated or truncated:
                record(*env.reset())
    finally:
        env.close()
    return {"wire": fake.wire, "outputs": outputs.hexdigest(), "infos": infos}


def v1_info_keys() -> list[str]:
    return sorted({k for info in v1_script()["infos"] for k in info})


@functools.lru_cache(maxsize=1)
def v1_progress_keys() -> dict:
    """The key shape of status.json and episodes.jsonl after ProgressCallback records the script's episodes."""
    from ultrakill_ai.progress import ProgressCallback  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "v1"
        cb = ProgressCallback(run / "status.json", 1000, "v1", 1, update_every_s=0.0)
        cb._on_training_start()
        for info in v1_script()["infos"]:
            if "end_reason" in info:
                cb._record_episode(0, {**info, "episode": {"r": 1.0, "l": 100.0}})
        cb._write(time.time())
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        lines = [json.loads(s) for s in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"status": list(status), "mean_100": list(status["mean_100"]),
            "campaign": list(status.get("campaign") or {}), "episodes": len(lines),
            "line": [list(line) for line in lines]}


@functools.lru_cache(maxsize=1)
def v1_pins() -> dict:
    """Every v1 pin's value as the imported `ultrakill_ai` computes it, by constant name."""
    env, _ = make_env()
    env.client = fake = WireLevel()
    try:
        env.reset()
        env.step(forward())
    finally:
        env.close()
    wire = v1_script()["wire"]
    return {"V1_FORWARD_LINE": fake.wire[0] if len(fake.wire) == 1 else fake.wire,
            "V1_WIRE_LINES": len(wire),
            "V1_WIRE_SHA256": hashlib.sha256("\n".join(wire).encode("utf-8")).hexdigest(),
            "V1_OUTPUTS_SHA256": v1_script()["outputs"],
            "V1_INFO_KEYS_SHA256": _sha(v1_info_keys()),
            "V1_PROGRESS_KEYS_SHA256": _sha(v1_progress_keys())}


def v1_pin_mismatches() -> list[str]:
    """One line per pin that differs, naming what it pins. Every pin, whichever test asks: a failure then says at
    once whether the step line, the wire, the outputs or a key set moved (or several)."""
    now = v1_pins()
    return [f"{name} ({V1_PIN_WHAT[name]}) moved: pinned {globals()[name]!r}, now {now[name]!r}"
            for name in V1_PIN_NAMES if now[name] != globals()[name]]


def _assert_pins(names: tuple[str, ...], *detail) -> None:
    bad = v1_pin_mismatches()
    if any(line.split(" ", 1)[0] in names for line in bad):
        raise AssertionError("\n".join([*bad, *map(str, detail)]))


def test_the_v1_step_bytes_are_the_base_commits():
    _assert_pins(("V1_FORWARD_LINE", "V1_WIRE_LINES", "V1_WIRE_SHA256"))


def test_the_v1_step_outputs_and_info_keys_are_the_base_commits():
    keys = v1_info_keys()
    _assert_pins(("V1_OUTPUTS_SHA256", "V1_INFO_KEYS_SHA256"), f"info keys now: {keys}")
    assert not [k for k in keys if k.startswith(("macro_", "variant_request", "hook_request"))], keys


def test_the_v1_status_json_and_episode_line_keys_are_the_base_commits():
    shape = v1_progress_keys()
    assert shape["episodes"] >= 2, "the script must end several episodes for the pin to mean anything"
    _assert_pins(("V1_PROGRESS_KEYS_SHA256",), f"key shape now: {shape}")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
