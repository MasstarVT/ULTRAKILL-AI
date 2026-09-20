"""STAGE S5: the sticky weapon slot, and the pin that it is DORMANT until a config says otherwise.

No game needed:  python tests/test_sticky_slot.py   (or pytest)

Stage S5 of docs/superpowers/specs/2026-09-20-speedrun-tech.md. §3.4: `GunControl.SwitchWeapon` with
`targetSlotIndex == currentSlotIndex` reads `PrefsManager`'s `WeaponRedrawBehaviour`, whose default is 0 =
cycle to the next variation, so pressing the slot already held RE-DRAWS the weapon -- `Revolver.OnEnable` sets
`gunReady = false`, only the `ReadyGun()` animation event clears it, and `Revolver.Update` gates firing on it.
A probe of the promoted `Level_0-1.zip` measured that press on 76.0% of steps.

THIS LEVER IS NOT TURNED ON HERE. `EnvConfig.sticky_weapon_slot` defaults False, no shipped config sets it,
and `test_the_lever_is_off_everywhere_that_ships` is the pin. The 76% is from a different checkpoint and the
spec says so; S0's `slot_same_frac` is the live re-measurement this switch waits on.

WHAT IS NOT MEASURED, and is said here because a reader will otherwise take it for a finding: the weapon draw
animation's real length. The game's clips are serialized and cannot be read from the decompiled C#, so
`sticky_slot_switch_every`'s default of 3 decisions is derived from the nearest documented window in the
spec's own table (the 200 ms `JumpReady` cooldown) at 66.7 ms per decision. It is a starting value.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_driver  # noqa: E402
import train  # noqa: E402
from test_campaign_env import action, forward, idle, make_env  # noqa: E402
from test_slot_counters import slot_action  # noqa: E402

from ultrakill_ai.env import EnvConfig  # noqa: E402

CONFIGS = ROOT / "configs"


def sticky_env(**overrides):
    return make_env(sticky_weapon_slot=True, **overrides)


def slots_sent(client) -> int:
    return int(client.last_action["slot"])


def test_the_lever_is_off_by_default():
    cfg = EnvConfig()
    assert cfg.sticky_weapon_slot is False
    assert cfg.sticky_slot_switch_every == 3


def test_the_lever_is_off_everywhere_that_ships():
    """No shipped config, and no stage the driver generates from the plan, turns it on."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        env_cfg, _ = train.load_config(str(path))
        assert not env_cfg.get("sticky_weapon_slot"), path.name
    plan = campaign_driver.load_plan(CONFIGS / "specialists.yaml")
    assert plan.speed_env == {}, "the plan ships no speed.env block at all"
    for kind in campaign_driver.STAGE_KINDS:
        env = campaign_driver.stage_config(plan, "Level 0-1", kind=kind)["env"]
        assert "sticky_weapon_slot" not in env
        assert EnvConfig.from_dict(env).sticky_weapon_slot is False


def test_off_is_the_action_stream_byte_for_byte():
    """The pin. Identical actions, one env with the lever off and one with it never constructed at all."""
    plain, plain_client = make_env()
    same, same_client = make_env(sticky_weapon_slot=False, sticky_slot_switch_every=1)
    try:
        for env, client in ((plain, plain_client), (same, same_client)):
            client.weapon_slot = 2
        plain.reset()
        same.reset()
        script = [slot_action(3), slot_action(3), slot_action(1), slot_action(3, move=True),
                  slot_action(0), forward(), slot_action(2), slot_action(5), action(buttons=("fire1",))]
        for i, a in enumerate(script):
            a_obs, a_reward, a_term, a_trunc, _ = plain.step(a)
            b_obs, b_reward, b_term, b_trunc, _ = same.step(a)
            assert plain_client.last_action == same_client.last_action, f"wire command differs at step {i}"
            assert a_obs.tobytes() == b_obs.tobytes() and a_reward == b_reward
            assert (a_term, a_trunc) == (b_term, b_trunc)
    finally:
        plain.close()
        same.close()


def test_on_a_press_of_the_held_slot_is_sent_as_keep():
    """Rule 1. `weapon_slot` is 0-based, the action's `slot` is a slot KEY, so "held" is `slot == index + 1`."""
    env, client = sticky_env()
    try:
        client.weapon_slot = 2  # holding slot 3
        env.reset()
        env.step(slot_action(3))
        assert slots_sent(client) == 0, "the redraw press never reaches the game"
        _, _, _, _, info = env.step(idle())
        assert info["slot_dropped_frac"] == 1 / 2
        assert info["slot_same_frac"] == 1 / 2, "the POLICY's intent is still measured, before the rewrite"
    finally:
        env.close()


def test_on_a_switch_is_honoured_then_rate_limited():
    """Rule 2: at `sticky_slot_switch_every: 3`, one switch is honoured and the next two are not."""
    env, client = sticky_env(sticky_slot_switch_every=3)
    try:
        client.weapon_slot = 0  # holding slot 1, and the corridor never changes it
        env.reset()
        sent = []
        for _ in range(7):
            env.step(slot_action(4))
            sent.append(slots_sent(client))
        assert sent == [4, 0, 0, 4, 0, 0, 4], sent
        _, _, _, _, info = env.step(idle())
        assert info["slot_blocked_frac"] == 4 / 8 and info["slot_dropped_frac"] == 0.0
    finally:
        env.close()


def test_on_a_cooldown_of_one_honours_every_switch():
    """`sticky_slot_switch_every` 0 or 1 is the drop rule alone, which is what the spec's S5 line describes."""
    for every in (0, 1):
        env, client = sticky_env(sticky_slot_switch_every=every)
        try:
            client.weapon_slot = 0
            env.reset()
            sent = []
            for _ in range(4):
                env.step(slot_action(4))
                sent.append(slots_sent(client))
            assert sent == [4, 4, 4, 4], (every, sent)
        finally:
            env.close()


def test_on_the_lever_never_selects_a_slot_the_policy_did_not_ask_for():
    """It can only ever turn a press into `keep`: no press is added, and no slot is substituted."""
    env, client = sticky_env(sticky_slot_switch_every=5)
    try:
        client.weapon_slot = 1
        env.reset()
        for a in (slot_action(0), slot_action(2), slot_action(5), slot_action(3), slot_action(0), forward()):
            env.step(a)
            asked, sent = int(a[8]), slots_sent(client)
            assert sent in (0, asked), (asked, sent)
    finally:
        env.close()


def test_on_an_unknown_held_slot_is_passed_through_untouched():
    """`weapon_slot` -1 is "GunControl has not started": nothing is known, so nothing is refused."""
    env, client = sticky_env(sticky_slot_switch_every=10)
    try:
        client.weapon_slot = -1
        env.reset()
        for _ in range(4):
            env.step(slot_action(2))
            assert slots_sent(client) == 2
        _, _, _, _, info = env.step(idle())
        assert info["slot_dropped_frac"] == 0.0 and info["slot_blocked_frac"] == 0.0
    finally:
        env.close()


def test_on_the_cooldown_is_cleared_by_a_level_load():
    """A load re-draws the weapon anyway, so a rate limit carried across one would be measuring nothing."""
    env, client = sticky_env(sticky_slot_switch_every=10)
    try:
        client.weapon_slot = 0
        env.reset()
        env.step(slot_action(4))
        assert slots_sent(client) == 4 and env._slot_cooldown == 9
        env.reset()
        assert env._slot_cooldown == 0
        env.step(slot_action(4))
        assert slots_sent(client) == 4
    finally:
        env.close()


def test_the_lever_is_reachable_from_a_plan_for_a_speed_stage_only():
    """`speed.env:` is the documented path. A complete stage's generated config must not move."""
    plan = campaign_driver.load_plan(CONFIGS / "specialists.yaml")
    lever = dataclasses.replace(plan, speed_env={"sticky_weapon_slot": True, "sticky_slot_switch_every": 4})
    speed = campaign_driver.stage_config(lever, "Level 0-1", kind=campaign_driver.SPEED)["env"]
    complete = campaign_driver.stage_config(lever, "Level 0-1")["env"]
    assert EnvConfig.from_dict(speed).sticky_weapon_slot is True
    assert EnvConfig.from_dict(speed).sticky_slot_switch_every == 4
    assert complete == campaign_driver.stage_config(plan, "Level 0-1")["env"], \
        "a complete stage's config is byte-identical with or without the block"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
