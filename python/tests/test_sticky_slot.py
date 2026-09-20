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
    """Rule 1. `weapon_slot` is the 1-BASED `GunControl.currentSlotIndex` and the action's `slot` is the same
    KEY, so "held" is `slot == weapon_slot` -- no offset. Until 2026-09-20 this compared `weapon_slot + 1`,
    and rule 1 therefore fired on a switch to the NEXT slot and never on a real redraw at all."""
    for key in (1, 3, 5):
        env, client = sticky_env()
        try:
            client.weapon_slot = key
            env.reset()
            env.step(slot_action(key))
            assert slots_sent(client) == 0, f"the redraw press never reaches the game (slot {key})"
            _, _, _, _, info = env.step(idle())
            assert info["slot_dropped_frac"] == 1 / 2, key
            assert info["slot_same_frac"] == 1 / 2, "the POLICY's intent is still measured, before the rewrite"
        finally:
            env.close()


def test_on_a_switch_to_the_next_slot_up_is_a_switch_and_not_a_redraw():
    """The exact shape of the 2026-09-20 off-by-one, pinned so it cannot come back.

    `slot == weapon_slot + 1` is a press of the slot ONE ABOVE the one in hand -- a perfectly ordinary
    switch. The broken rule dropped it as though it were the redraw, which both suppressed a real weapon
    change and left the actual redraw (`slot == weapon_slot`) reaching the game on every press.
    """
    env, client = sticky_env(sticky_slot_switch_every=1)
    try:
        client.weapon_slot = 2          # holding slot 2 ...
        env.reset()
        env.step(slot_action(3))        # ... and pressing slot 3: a switch, and honoured
        assert slots_sent(client) == 3, "a switch to the next slot up must reach the game"
        _, _, _, _, info = env.step(idle())
        assert info["slot_dropped_frac"] == 0.0, "nothing here is a redraw"
        assert info["slot_switch_frac"] == 1 / 2 and info["slot_same_frac"] == 0.0
    finally:
        env.close()


def test_on_a_switch_is_honoured_then_rate_limited():
    """Rule 2: at `sticky_slot_switch_every: 3`, one switch is honoured and the next two are not."""
    env, client = sticky_env(sticky_slot_switch_every=3)
    try:
        client.weapon_slot = 1  # holding slot 1, and the corridor never changes it
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
            client.weapon_slot = 1
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


def test_on_a_press_of_an_EMPTY_slot_costs_no_cooldown():
    """The cooldown rations SWITCHES, and the game cannot switch into a slot with no weapon in it.

    0-1's opening is the case: the revolver is picked up first and the other four slots stay empty for most of
    the level, so a policy that presses them -- which the S0 counter `slot_unowned_frac` measures -- would
    otherwise spend its whole switch budget on switches `GunControl.SwitchWeapon` never performed, cutting
    real switching by up to `sticky_slot_switch_every` times while `slot_blocked_frac` read as though the
    cooldown were working. The S5 round could then not separate "sticky slot did not help" from "the cooldown
    was spent on non-events".
    """
    env, client = sticky_env(sticky_slot_switch_every=3)
    try:
        client.weapon_slot = 1          # holding slot 1, the revolver
        client.slot_counts = [1, 0, 0, 0, 0]  # ... and nothing else has been picked up yet
        env.reset()
        for _ in range(4):
            env.step(slot_action(3))    # a slot the game has no weapon in
            assert slots_sent(client) == 3, "a press the game ignores is not refused either"
        assert env._slot_cooldown == 0, "and it costs nothing, so a REAL switch is still available"
        env.step(slot_action(2))        # ... which this proves: slot 2 is empty too, so still free
        client.slot_counts = [1, 1, 0, 0, 0]  # the shotgun is picked up
        env.step(idle())                # one step for that to reach the observation the next decision reads
        env.step(slot_action(2))        # now it is a real switch: honoured, and charged
        assert slots_sent(client) == 2 and env._slot_cooldown == 2
        env.step(slot_action(2))
        assert slots_sent(client) == 0, "the real switch is rate-limited as before"
    finally:
        env.close()


def test_on_holding_slot_6_makes_every_press_a_switch_and_that_is_correct():
    """Rule 1 cannot fire while slot KEY 6 is held, and must not: the policy cannot press key 6.

    The game really has six slots -- `GunControl` binds `Slot1`..`Slot6` and compares `currentSlotIndex != 6`
    -- but `NUM_WEAPON_CHOICES` is 6, keep plus keys 1..5, so `slot == key` is unreachable at `key == 6`.
    There is no redraw to drop there, because the policy has no way to ask for one; every key it CAN press
    while holding slot 6 names a different slot, so treating it as a switch is the right answer rather than a
    gap in the rule. (Slot 6 also ships empty in this build, so nothing reaches it in game either.)
    """
    env, client = sticky_env(sticky_slot_switch_every=3)
    try:
        client.weapon_slot = 6  # slot KEY 6: a real game slot no action can name
        client.slot_counts = [1, 1, 1, 1, 1, 1]
        env.reset()
        sent = [(env.step(slot_action(5)), slots_sent(client))[1] for _ in range(4)]
        assert sent == [5, 0, 0, 5], sent
        _, _, _, _, info = env.step(idle())
        assert info["slot_dropped_frac"] == 0.0, "nothing is a redraw at slot 6"
        assert info["slot_switch_frac"] == 4 / 5
    finally:
        env.close()


def test_on_an_unanswerable_slot_counts_is_still_charged():
    """Unknown is not "empty". An old mod sends no `slot_counts`, and guessing would disable the rule."""
    env, client = sticky_env(sticky_slot_switch_every=3)
    try:
        client.weapon_slot = 1
        client.slot_counts = []  # what BuildPlayer sends before GunControl starts, and all a v0.4 mod sends
        env.reset()
        env.step(slot_action(3))
        assert slots_sent(client) == 3 and env._slot_cooldown == 2
        env.step(slot_action(3))
        assert slots_sent(client) == 0, "unknown falls through to the conservative half of the rule"
    finally:
        env.close()


def test_on_the_cooldown_is_cleared_by_a_level_load():
    """A load re-draws the weapon anyway, so a rate limit carried across one would be measuring nothing."""
    env, client = sticky_env(sticky_slot_switch_every=10)
    try:
        client.weapon_slot = 1
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
