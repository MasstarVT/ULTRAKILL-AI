"""STAGE S0: the weapon-channel counters, and the pin that they change nothing.

No game needed:  python tests/test_slot_counters.py   (or pytest)

Stage S0 of docs/superpowers/specs/2026-09-20-speedrun-tech.md is "measure first". Section 3.4 of that spec
reports that a probe of the PROMOTED `Level_0-1.zip` pressed the slot key of the weapon it was ALREADY HOLDING
on 76.0% of steps (39,371 of 51,772) -- which, with `WeaponRedrawBehaviour` defaulting to 0 (cycle variation),
would mean the agent keeps its weapon in the draw animation and suppresses its own primary fire. The spec is
explicit that this figure is from a different checkpoint and that `status.json` carries no slot histogram to
corroborate it, so S5 (the sticky slot) may not be switched on until the number has been re-taken live.

These counters are that re-measurement. They are PASSIVE: nothing here is read by a reward, an observation, an
action or a promotion rule, and `test_the_counters_cannot_change_a_single_step_output` is the pin that says so
-- it runs the same actions through two envs, one of which has the whole behaviour-counting layer replaced by
no-ops, and requires the observation bytes, the reward, the two end flags and the command sent over the wire to
be identical on every step.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_campaign_env import action, forward, idle, make_env  # noqa: E402

from ultrakill_ai.spaces import NUM_WEAPON_SLOTS  # noqa: E402

# Every key `info` carried BEFORE stage S0, taken from the main tree at commit 259a2c5 by stepping this same
# FakeLevel corridor. Pinned literally, because "the counters added nothing else" is the claim being made.
BASELINE_INFO_KEYS = {
    "bridge_resets", "cells_new", "checkpoints_level", "completed", "completion_bonus", "deaths", "end_pos",
    "enemy_angle_mean", "enemy_close_frac", "enemy_dist_mean", "enemy_elev_abs_mean", "enemy_elev_mean",
    "enemy_elev_over15_frac", "enemy_pitch_err_mean", "enemy_visible_frac", "enemy_yaw_angle_mean",
    "exit_banished", "exit_dist_min", "exit_ground_dist_min", "firing_frac", "firing_on_target_frac",
    "fresh_start", "gate_hops_best", "gates_reached", "hp", "kills", "ladder_collapsed", "level",
    "level_seconds", "level_started", "look_enemy_frac", "look_free_frac", "look_gate_frac", "look_up_mean",
    "on_target_frac", "oob_frac", "pitch_abs_mean", "pitch_mean", "pitch_track", "reward_parts",
    "route_source", "route_source_name", "s_rank_seconds", "slide_forced_frac", "start_checkpoint", "style",
    "target_seconds", "targets_parked", "wave", "wedged_steps", "yaw_per_step_mean", "yaw_track",
}
# ... and exactly what S0 adds. `slot_press_per_s` is not here because, like `kills_per_min`, it is only set
# on the step that ENDS the episode -- it is the one counter denominated in game seconds rather than decisions.
NEW_INFO_KEYS = {
    "slot_press_frac", "slot_same_frac", "slot_switch_frac", "slot_unowned_frac",
    "fire1_frac", "fire2_frac", "punch_frac",
    "slot_dropped_frac", "slot_blocked_frac",
    "slot_held_frac", "slot_held_top_frac", "slot_kills", "slot_known_frac",
    "held_variation_frac", "variation0_frac", "variation_known_frac",
}


def slot_action(slot: int, *, move: bool = False):
    """An action that presses weapon slot key `slot` (0 keeps the current weapon)."""
    a = action(move=move)
    a[2 + 6] = slot  # [move x2, six buttons, SLOT, yaw, pitch, look_mode]
    return a


def test_the_info_gains_the_slot_counters_and_nothing_else():
    env, _ = make_env()
    try:
        env.reset()
        _, _, _, _, info = env.step(forward())
        assert set(info) == BASELINE_INFO_KEYS | NEW_INFO_KEYS, sorted(set(info) ^ (BASELINE_INFO_KEYS | NEW_INFO_KEYS))
    finally:
        env.close()


def test_the_counters_cannot_change_a_single_step_output():
    """THE PIN. Identical actions through two envs, one with the counting layer gone: identical outputs.

    "Outputs" is everything that leaves the env or reaches the game: the packed observation's BYTES, the
    reward, `terminated`, `truncated`, and the command dict the client was handed. `info` is deliberately not
    compared -- it is where the counters are reported, and reporting them is the whole point.
    """
    plain, plain_client = make_env()
    counted, counted_client = make_env()
    try:
        plain._note_behaviour = lambda *a, **k: None       # the whole behaviour layer, old counters and new
        plain._note_slot_kills = lambda *a, **k: None
        plain.reset()
        counted.reset()
        script = [forward(), slot_action(1, move=True), slot_action(3), action(buttons=("fire1", "punch")),
                  slot_action(2, move=True), idle(), action(buttons=("fire2",), move=True), forward()]
        for i, a in enumerate(script):
            a_obs, a_reward, a_term, a_trunc, _ = plain.step(a)
            b_obs, b_reward, b_term, b_trunc, _ = counted.step(a)
            assert a_obs.tobytes() == b_obs.tobytes(), f"observation differs at step {i}"
            assert a_reward == b_reward and (a_term, a_trunc) == (b_term, b_trunc), f"reward/flags differ at step {i}"
            assert plain_client.last_action == counted_client.last_action, f"wire command differs at step {i}"
    finally:
        plain.close()
        counted.close()


def test_a_press_of_the_held_slot_is_counted_as_the_redraw():
    """`weapon_slot` is the 1-BASED `GunControl.currentSlotIndex`; the action's `slot` is the same KEY.

    Deliberately UNBALANCED -- two redraws and one switch -- because a one-of-each script cannot tell the
    fixed code from the off-by-one that shipped on 2026-09-20: that version compared `slot == weapon_slot + 1`
    and so labelled these exact presses 1 same / 2 switch, the mirror image of the truth.
    """
    env, client = make_env()
    try:
        client.weapon_slot = 3  # holding slot 3, as the mod sends it
        env.reset()
        env.step(slot_action(3))  # the slot already held: the redraw press
        env.step(slot_action(3))  # ... and again
        env.step(slot_action(4))  # a different slot: a real switch
        for _ in range(4):
            env.step(slot_action(0))  # keep: no press at all
        _, _, _, _, info = env.step(idle())
        # Eight decisions, so the three shares are exact in binary and the sum below can be an equality.
        assert info["slot_press_frac"] == 3 / 8
        assert info["slot_same_frac"] == 2 / 8
        assert info["slot_switch_frac"] == 1 / 8
        assert info["slot_known_frac"] == 1.0
        # `slot_same_frac` is denominated in DECISIONS, which is the denominator §3.4's 76% used.
        assert info["slot_same_frac"] + info["slot_switch_frac"] == info["slot_press_frac"]
    finally:
        env.close()


def test_the_held_slot_is_read_as_the_1_based_key_the_mod_really_sends():
    """THE REGRESSION TEST for the 2026-09-20 off-by-one, at every slot the policy can press.

    `GunControl.currentSlotIndex` is 1-based -- seeded `PlayerPrefs.GetInt("CurSlo", 1)`, reset to 1 rather
    than 0 when it runs past `slots.Count`, and indexed `slots[currentSlotIndex - 1]` throughout the game --
    and `ObservationBuilder` sends it RAW. Pressing the key equal to it IS the redraw, at every one of them.
    """
    for key in (1, 2, 3, 4, 5):
        env, client = make_env()
        try:
            client.weapon_slot = key
            env.reset()
            env.step(slot_action(key))                    # the redraw
            env.step(slot_action(key % 5 + 1))            # any other key: a switch
            _, _, _, _, info = env.step(idle())
            assert info["slot_same_frac"] == 1 / 3, key
            assert info["slot_switch_frac"] == 1 / 3, key
            # ... and the time share lands on that key's own entry, which is KEY - 1.
            assert info["slot_held_frac"][key - 1] == 1.0, key
        finally:
            env.close()


def test_slot_0_is_unknown_because_the_game_never_reports_it():
    """0 is not slot 1. The game's own out-of-range reset is `currentSlotIndex = 1`, never 0, so a 0 on the
    wire would mean a build this code has never seen -- and the rule everywhere else here applies: never
    guess a slot from a value that is not one."""
    env, client = make_env()
    try:
        client.weapon_slot = 0
        env.reset()
        for _ in range(3):
            env.step(slot_action(1))
        _, _, _, _, info = env.step(idle())
        assert info["slot_known_frac"] == 0.0
        assert info["slot_same_frac"] == 0.0 and info["slot_switch_frac"] == 0.0
        assert info["slot_held_frac"] == [0.0] * NUM_WEAPON_SLOTS
    finally:
        env.close()


def test_a_press_of_an_empty_slot_is_counted_as_unowned():
    """`player.slot_counts` says which slots hold a weapon, and the game cannot switch into an empty one.

    This is measured PASSIVELY, with the sticky lever off, because it is the number that decides whether the
    lever's ownership gate matters: on 0-1 the revolver is the only weapon for most of the level, so a policy
    that presses 2..5 there is pressing keys that do nothing at all. It is also the precondition for reading
    an S5 round -- a cooldown charged for those presses would ration switches that never happened.
    """
    env, client = make_env()
    try:
        client.weapon_slot = 1  # holding slot 1, the revolver
        client.slot_counts = [1, 0, 0, 0, 0]  # 0-1 after the revolver pickup and nothing else
        env.reset()
        env.step(slot_action(1))   # owned: the redraw
        env.step(slot_action(4))   # empty
        env.step(slot_action(5))   # empty
        env.step(slot_action(0))   # no press at all
        _, _, _, _, info = env.step(idle())
        assert info["slot_press_frac"] == 3 / 5
        assert info["slot_unowned_frac"] == 2 / 5
        assert info["slot_dropped_frac"] == 0.0, "the lever is off: nothing is rewritten"
    finally:
        env.close()


def test_an_unanswerable_slot_counts_is_not_counted_as_unowned():
    """Unknown is a third outcome, never "empty": an empty array is what the mod sends before GunControl."""
    env, client = make_env()
    try:
        client.weapon_slot = 1
        client.slot_counts = []
        env.reset()
        for _ in range(3):
            env.step(slot_action(4))
        _, _, _, _, info = env.step(idle())
        assert info["slot_press_frac"] == 3 / 4 and info["slot_unowned_frac"] == 0.0
    finally:
        env.close()


def test_the_variation_held_is_measured_before_anything_freezes_it():
    """`weapon_variation` has been on the wire and thrown away (§3.2). S5 freezes it, so it is read first.

    Variation 0 is Piercer / Core Eject / Electric Railcannon / Freezeframe (§3.5) -- the set every technique
    in the spec is built on. A sticky-slot round that happened to freeze on the Marksman would show a worse
    `kills_per_min` and `firing_on_target_frac` for a reason that has nothing to do with the sticky slot, and
    without this counter nothing in status.json, metrics_log.csv or episodes.jsonl could tell the two apart.
    """
    env, client = make_env()
    try:
        client.weapon_variation = 0
        env.reset()
        env.step(forward())
        client.weapon_variation = 1
        env.step(forward())
        env.step(forward())
        _, _, _, _, info = env.step(idle())
        assert info["variation_known_frac"] == 1.0
        # Two of the four decisions were TAKEN while variation 0 was held. Like every other counter here the
        # read is off `prev` -- the observation the decision was made on -- so the change made after step 1
        # first shows in step 2's OBSERVATION and so in step 3's count. That lag is the intended semantics:
        # the number answers "what was the policy holding when it chose", which is what a round is judged on.
        assert info["variation0_frac"] == 2 / 4
        assert info["held_variation_frac"] == [2 / 4, 2 / 4, 0.0]
        assert info["variation0_frac"] == info["held_variation_frac"][0]
    finally:
        env.close()


def test_an_unknown_variation_is_never_guessed():
    """-1 is "GunControl has not started", exactly as it is for `weapon_slot`: it is not variation 0."""
    env, client = make_env()
    try:
        client.weapon_variation = -1
        env.reset()
        for _ in range(3):
            env.step(forward())
        _, _, _, _, info = env.step(idle())
        assert info["variation_known_frac"] == 0.0
        assert info["variation0_frac"] == 0.0
        assert info["held_variation_frac"] == [0.0, 0.0, 0.0]
    finally:
        env.close()


def test_an_unknown_held_slot_is_never_guessed():
    """`weapon_slot` is -1 until `GunControl` starts -- on 0-1 there is no weapon at all until the pickup."""
    env, client = make_env()
    try:
        client.weapon_slot = -1
        env.reset()
        for _ in range(4):
            env.step(slot_action(1))
        _, _, _, _, info = env.step(idle())
        assert info["slot_press_frac"] == 4 / 5, "the press itself is still counted"
        assert info["slot_same_frac"] == 0.0 and info["slot_switch_frac"] == 0.0, "nothing is known to compare"
        assert info["slot_known_frac"] == 0.0
        assert info["slot_held_frac"] == [0.0] * NUM_WEAPON_SLOTS
    finally:
        env.close()


def test_the_time_share_per_held_slot_is_over_the_steps_the_slot_was_known():
    """Entry i is slot KEY i + 1, so slot 3 lands at index 2. Before the 2026-09-20 fix the raw 1-based field
    was used as the index directly and every share sat one place to the right of its own name."""
    env, client = make_env()
    try:
        client.weapon_slot = 3  # slot KEY 3 -> entry 2
        env.reset()
        env.step(idle())
        env.step(idle())
        client.weapon_slot = 5  # the obs THIS step produces carries slot 5 ...
        env.step(idle())
        _, _, _, _, info = env.step(idle())  # ... and this step is the first to act on it
        # The slot is read from the frame the policy ACTED ON, so a switch shows one step later.
        shares = info["slot_held_frac"]
        assert len(shares) == NUM_WEAPON_SLOTS and abs(sum(shares) - 1.0) < 1e-9
        assert shares[2] == 3 / 4 and shares[4] == 1 / 4, shares
        assert info["slot_held_top_frac"] == 3 / 4
    finally:
        env.close()


def test_the_revolver_is_entry_0_of_the_per_slot_lists():
    """The named case, because it is the one a human reads off episodes.jsonl: 0-1 opens on the revolver
    (slot KEY 1), and its share and its kills belong in entry 0, not entry 1."""
    env, client = make_env()
    try:
        client.weapon_slot = 1
        env.reset()
        client.kill_enemy_next = True
        _, _, _, _, info = env.step(idle())
        assert info["slot_held_frac"][0] == 1.0, info["slot_held_frac"]
        assert info["slot_kills"][0] == 1, info["slot_kills"]
        assert sum(info["slot_kills"]) == 1
    finally:
        env.close()


def test_the_button_press_rates_separate_fire1_from_fire2():
    """`firing_frac` is the OR of the two and could never tell them apart; `punch` was not counted at all."""
    env, _ = make_env()
    try:
        env.reset()
        env.step(action(buttons=("fire1",)))
        env.step(action(buttons=("fire2",)))
        env.step(action(buttons=("fire1", "fire2", "punch")))
        _, _, _, _, info = env.step(idle())
        assert info["fire1_frac"] == 2 / 4 and info["fire2_frac"] == 2 / 4 and info["punch_frac"] == 1 / 4
        assert info["firing_frac"] == 3 / 4, "unchanged: fire1 OR fire2"
    finally:
        env.close()


def test_kills_are_credited_to_the_slot_that_was_held_when_the_shot_went_out():
    env, client = make_env()
    try:
        client.weapon_slot = 2  # before reset: the slot read is the one on the frame the policy ACTED ON
        env.reset()
        client.kill_enemy_next = True
        _, _, _, _, info = env.step(idle())  # the kill lands here; slot KEY 2 was held when it fired
        kills = info["slot_kills"]
        assert len(kills) == NUM_WEAPON_SLOTS
        assert kills[1] == 1 and sum(kills) == 1, kills  # KEY 2 -> entry 1
    finally:
        env.close()


def test_a_kill_counter_rollback_never_produces_a_negative_credit():
    """The game's `kills` rolls back to the checkpoint value on a respawn (133 rollbacks over 126 deaths in
    runs/probe_0-2_speed), so this uses the same `max(0, delta)` `compute_reward` does."""
    env, client = make_env()
    try:
        env.reset()
        client.weapon_slot = 1  # the revolver: slot KEY 1 -> entry 0
        client.kills = 10
        _, _, _, _, info = env.step(idle())
        assert info["slot_kills"][0] == 10
        client.kills = 2  # the rollback
        _, _, _, _, info = env.step(idle())
        assert info["slot_kills"] == [10, 0, 0, 0, 0, 0], "a rollback contributes nothing, never a subtraction"
    finally:
        env.close()


def test_slot_presses_per_game_second_are_reported_when_the_episode_ends():
    """Denominated in GAME seconds, like `kills_per_min`, so two runs at different frameskip are comparable."""
    env, client = make_env()
    try:
        env.reset()
        info = None
        for _ in range(200):
            _, _, term, trunc, info = env.step(slot_action(1, move=True))
            if term or trunc:
                break
        assert info is not None and "end_reason" in info, "the corridor's exit ends the episode"
        seconds = info["episode_seconds"]
        assert seconds > 0
        presses = env._behaviour["slot_press"]
        assert presses > 0
        assert abs(info["slot_press_per_s"] - presses / seconds) < 1e-9
    finally:
        env.close()


def test_the_counters_reach_status_json_the_csv_and_episodes_jsonl():
    """The three files the stage is read off. Imports stable_baselines3, so it is last in this file."""
    import poll_status  # noqa: PLC0415
    from ultrakill_ai.progress import SLOT_METRICS, ProgressCallback  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        status = Path(tmp) / "status.json"
        cb = ProgressCallback(status, 1_000_000, "spec_0-1_speed", 1)
        info = {
            "episode": {"r": 10.0, "l": 100},
            "kills": 3, "deaths": 0, "fresh_start": 1, "completed": 1, "level": "Level 0-1",
            "end_reason": "level_complete", "reward_parts": {},
            "slot_press_frac": 0.8, "slot_same_frac": 0.76, "slot_switch_frac": 0.04,
            "slot_unowned_frac": 0.02,
            "slot_press_per_s": 12.0, "fire1_frac": 0.3, "fire2_frac": 0.1, "punch_frac": 0.5,
            "slot_held_top_frac": 0.9, "slot_known_frac": 1.0,
            "slot_dropped_frac": 0.0, "slot_blocked_frac": 0.0,
            "variation0_frac": 1.0, "variation_known_frac": 0.95,
            "slot_held_frac": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0], "slot_kills": [2, 1, 0, 0, 0, 0],
            "held_variation_frac": [1.0, 0.0, 0.0],
        }
        cb._record_episode(0, info)
        snapshot = cb._snapshot(time.time())
        for name in SLOT_METRICS:
            assert snapshot["mean_100"][name] == info[name], name

        line = json.loads((Path(tmp) / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert line["slot_same_frac"] == 0.76 and line["slot_press_per_s"] == 12.0
        assert line["slot_held_frac"] == [0.9, 0.1, 0.0, 0.0, 0.0, 0.0]
        assert line["slot_kills"] == [2, 1, 0, 0, 0, 0]
        assert line["held_variation_frac"] == [1.0, 0.0, 0.0]

        row = poll_status.row(snapshot)
        for name in SLOT_METRICS:
            assert name in row and row[name] == info[name], name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
