"""Campaign reward terms. No game needed:  python tests/test_campaign_rewards.py  (or pytest)."""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.rewards import (  # noqa: E402
    SPEED_BONUS_MAX, SPEED_BONUS_MIN, CampaignStep, RewardConfig, completion_bonus, compute_reward)

CAMPAIGN_WEIGHTS = {"time": 0.01, "checkpoint": 10.0, "arena_clear": 10.0, "door_unlock": 3.0, "novelty": 0.5, "path": 0.1}
CAMPAIGN_PARTS = ("time", "checkpoint", "arena_clear", "door_unlock", "novelty", "path", "gate", "gate_approach")
# The weights configs/campaign_0-1.yaml runs with, for the arithmetic below.
GATES_WEIGHTS = {"time": 0.02, "level_complete": 100.0, "checkpoint": 10.0, "arena_clear": 10.0, "door_unlock": 15.0,
                 "gate": 15.0, "gate_approach": 0.15, "novelty": 0.2, "path": 0.0, "kill": 0.5, "damage_dealt": 0.5,
                 "damage_taken": 0.01, "death": 5.0, "style": 0.0, "punch": 0.01}


def snapshot(level_complete: bool = False, hp: int = 100, kills: int = 0) -> dict:
    """A quiet step the way the mod reports it: a player, no enemies, only the given stats."""
    return {
        "player": {"pos": [0.0, 0.0, 0.0], "hp": hp, "dead": False},
        "enemies": [],
        "stats": {"kills": kills, "style": 0, "level_complete": level_complete},
    }


def test_campaign_step_defaults_to_nothing_happened():
    step = CampaignStep()
    assert (step.checkpoints, step.arenas, step.doors, step.novelty, step.path_gain) == (0, 0, 0, 0.0, 0.0)
    assert (step.gates, step.gate_approach) == (0, 0.0)
    assert (step.item_pickups, step.item_placements) == (0, 0)


def test_item_terms_default_to_zero_and_scale_by_their_weights():
    """The skull-carry milestones ship at 0.0 and stay there until the in-game checks pass (§8 of the spec)."""
    cfg = RewardConfig()
    assert cfg.item_pickup == 0.0 and cfg.item_placed == 0.0
    off = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(item_pickups=1, item_placements=1))
    assert "item_pickup" not in off.parts and "item_placed" not in off.parts
    on = RewardConfig(**GATES_WEIGHTS, item_pickup=15.0, item_placed=15.0)
    parts = compute_reward(on, snapshot(), snapshot(), {}, campaign=CampaignStep(item_pickups=2, item_placements=1)).parts
    assert abs(parts["item_pickup"] - 30.0) < 1e-9 and abs(parts["item_placed"] - 15.0) < 1e-9


def test_item_terms_are_paid_on_a_step_without_a_player():
    """Like the other milestones: the env has already marked them paid, so a player-less frame must not drop them."""
    cfg = RewardConfig(item_pickup=15.0, item_placed=15.0)
    parts = compute_reward(cfg, {}, {}, {}, campaign=CampaignStep(item_pickups=1, item_placements=1)).parts
    assert abs(parts["item_pickup"] - 15.0) < 1e-9 and abs(parts["item_placed"] - 15.0) < 1e-9


def test_gate_terms_default_to_zero():
    cfg = RewardConfig()
    assert cfg.gate == 0.0 and cfg.gate_approach == 0.0  # Cyber Grind and every old config are unchanged
    off = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(gates=3, gate_approach=40.0))
    assert off.parts == {}


def test_gate_terms_scale_by_their_weights():
    cfg = RewardConfig(**GATES_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(gates=2, gate_approach=12.0)).parts
    assert abs(parts["gate"] - 30.0) < 1e-9
    assert abs(parts["gate_approach"] - 1.8) < 1e-9
    quiet = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep()).parts
    assert "gate" not in quiet and "gate_approach" not in quiet


def test_gate_terms_are_paid_on_a_step_without_a_player():
    """The env has already marked the ladder paid before the reward runs, so a player-less frame must not drop it."""
    cfg = RewardConfig(**GATES_WEIGHTS)
    loading = {"enemies": [], "stats": {}}
    parts = compute_reward(cfg, snapshot(), loading, {}, campaign=CampaignStep(gates=1, gate_approach=4.0)).parts
    assert set(parts) == {"time", "gate", "gate_approach"}


def test_the_route_is_the_largest_positive_group_for_a_route_walker():
    """The arithmetic of the design spec: 0-1 walked at human pace, 146.58 s = 2199 decisions at 15/s.

    10 hop values, a 723 m gate-to-gate polyline, 4 arena-held gate doors, 6 checkpoints, 3 distinct arena
    clear keys, 59 kills and ~60 health bars, ~200 raw novelty units through mostly-visited start rooms.
    """
    cfg = RewardConfig(**GATES_WEIGHTS)
    decisions = 2199
    parts: dict[str, float] = {}
    for name, step in (("route", CampaignStep(gates=10, gate_approach=723.0, doors=4)),
                       ("milestones", CampaignStep(checkpoints=6, arenas=3, novelty=200.0))):
        for key, value in compute_reward(cfg, snapshot(), snapshot(), {}, campaign=step).parts.items():
            parts[key] = parts.get(key, 0.0) + value
        assert name  # both steps contributed
    route = parts["gate"] + parts["gate_approach"] + parts["door_unlock"]
    assert abs(route - (150.0 + 108.45 + 60.0)) < 1e-6
    assert route > 100.0  # larger than finishing
    assert route > parts["checkpoint"] + parts["arena_clear"]  # and than the milestones
    assert route > parts["novelty"] * 1.5
    time_cost = decisions * cfg.time
    assert abs(time_cost - 43.98) < 1e-9 and route > 7 * time_cost

    # A wanderer that never leaves the start area: 9000 decisions, one gate approach, a measured 346 raw novelty.
    wander = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(gate_approach=64.8, novelty=346.0)).parts
    wandered = wander["gate_approach"] + wander["novelty"] - 9000 * cfg.time
    assert wandered < 0.0, f"wandering to the cap must not pay for itself, got {wandered:+.1f}"


def test_time_is_only_charged_with_a_campaign_step():
    cfg = RewardConfig(time=0.01)
    quiet = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep())
    assert quiet.parts == {"time": -0.01} and abs(quiet.total + 0.01) < 1e-12
    assert "time" not in compute_reward(cfg, snapshot(), snapshot(), {}).parts


def test_each_milestone_weight_multiplies_its_count():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(checkpoints=2, arenas=3, doors=4)).parts
    assert abs(parts["checkpoint"] - 20.0) < 1e-9
    assert abs(parts["arena_clear"] - 30.0) < 1e-9
    assert abs(parts["door_unlock"] - 12.0) < 1e-9
    assert "novelty" not in parts and "path" not in parts  # nothing new this step pays nothing
    single = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(doors=1)).parts
    assert abs(single["door_unlock"] - 3.0) < 1e-9 and "checkpoint" not in single and "arena_clear" not in single
    # The campaign config raises door_unlock to 15: the fight that unlocks a gate is the one place the route
    # signal is flat, and MilestoneTracker already pays it once per level load per key.
    raised = compute_reward(RewardConfig(**GATES_WEIGHTS), snapshot(), snapshot(), {}, campaign=CampaignStep(doors=4)).parts
    assert abs(raised["door_unlock"] - 60.0) < 1e-9


def test_novelty_and_path_scale_by_their_weights():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    novelty = 1.0 + 1.0 / 2.0 ** 0.5  # a cell no earlier episode entered, and one that one earlier episode did
    result = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(novelty=novelty, path_gain=12.5))
    assert abs(result.parts["novelty"] - 0.5 * novelty) < 1e-9
    assert abs(result.parts["path"] - 1.25) < 1e-9
    assert abs(result.total - sum(result.parts.values())) < 1e-9
    off = compute_reward(RewardConfig(), snapshot(), snapshot(), {}, campaign=CampaignStep(checkpoints=1, novelty=novelty, path_gain=12.5))
    assert off.parts == {}  # every campaign weight defaults to 0


def test_finishing_within_the_cap_beats_timing_out():
    # The campaign config charges time 0.01 per decision over a 9000-decision cap (10 game minutes).
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    max_steps = 9000
    tick = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep()).total
    last = compute_reward(cfg, snapshot(), snapshot(level_complete=True), {}, campaign=CampaignStep()).total
    timed_out = max_steps * tick
    finished_on_the_last_decision = (max_steps - 1) * tick + last
    assert abs(timed_out + 90.0) < 1e-6 and abs(last - 99.99) < 1e-9
    assert finished_on_the_last_decision > timed_out
    assert (max_steps // 2 - 1) * tick + last > finished_on_the_last_decision  # finishing sooner always pays more


def test_level_complete_pays_100_on_the_rising_edge_only():
    cfg = RewardConfig()
    assert cfg.level_complete == 100.0
    for campaign in (None, CampaignStep()):  # the same rule in both modes
        rising = compute_reward(cfg, snapshot(), snapshot(level_complete=True), {}, campaign=campaign).parts
        assert rising["level_complete"] == 100.0
        held = compute_reward(cfg, snapshot(level_complete=True), snapshot(level_complete=True), {}, campaign=campaign).parts
        assert "level_complete" not in held
        falling = compute_reward(cfg, snapshot(level_complete=True), snapshot(), {}, campaign=campaign).parts
        assert "level_complete" not in falling


def test_cybergrind_call_has_no_campaign_parts():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(kills=1), {}).parts
    assert parts["kill"] == cfg.kill
    assert not set(CAMPAIGN_PARTS) & set(parts)


def test_campaign_terms_are_paid_on_a_step_without_a_player():
    # The env marks milestones paid before it computes the reward, so a step that arrives without a player
    # (a level load) must still pay them or they are lost for the whole level load.
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    loading = {"enemies": [], "stats": {}}
    result = compute_reward(cfg, snapshot(), loading, {}, campaign=CampaignStep(checkpoints=1))
    assert set(result.parts) == {"time", "checkpoint"}
    assert compute_reward(cfg, snapshot(), loading, {}).parts == {}  # Cyber Grind: no player, no reward, as before


def test_died_is_a_keyword_after_enemy_max_health():
    cfg = RewardConfig(death=5.0, damage_taken=0.01)
    parts = compute_reward(cfg, snapshot(hp=40), snapshot(hp=40), {}, died=True).parts
    assert parts["death"] == -5.0
    assert abs(parts["damage_taken"] + 0.4) < 1e-9  # the lethal hit is the HP the player had left


def test_route_terms_are_retired():
    names = {f.name for f in fields(RewardConfig)}
    assert not {"route_point", "stuck"} & names
    assert not hasattr(RewardConfig(), "route_point") and not hasattr(RewardConfig(), "stuck")
    assert list(inspect.signature(compute_reward).parameters) == [
        "cfg", "prev", "cur", "enemy_max_health", "died", "campaign", "buttons",
        # The speed stage's two numbers, both keyword-only and both None outside one.
        "target_seconds", "official_seconds"]


def test_punch_is_charged_per_press_and_only_when_pressed():
    cfg = RewardConfig(punch=0.01)
    assert compute_reward(cfg, snapshot(), snapshot(), {}, buttons=("punch",)).parts["punch"] == -0.01
    assert "punch" not in compute_reward(cfg, snapshot(), snapshot(), {}, buttons=("fire1", "jump")).parts
    assert "punch" not in compute_reward(cfg, snapshot(), snapshot(), {}).parts  # default: no buttons, no charge
    # A zero weight (Cyber Grind, and any config that does not set it) pays nothing even when pressed.
    assert "punch" not in compute_reward(RewardConfig(), snapshot(), snapshot(), {}, buttons=("punch",)).parts


def test_level_complete_pays_even_when_the_frame_has_no_player():
    """The frame that ends a level arrives as the scene unloads and often has no player object.

    Before this, compute_reward returned early on a missing player and the +100 was simply never paid -- so the
    one outcome the campaign run exists to produce could have scored zero.
    """
    cfg = RewardConfig(level_complete=100.0)
    cur = dict(snapshot(level_complete=True))
    cur.pop("player")  # no player at all
    assert compute_reward(cfg, snapshot(), cur, {}).parts["level_complete"] == 100.0


# ---------------------------------------------------------------------------
# The speed stage's time-scaled completion bonus
# (docs/superpowers/specs/2026-09-18-speed-stages.md §2)
# ---------------------------------------------------------------------------


def test_the_plain_bonus_is_byte_identical_when_no_target_is_set():
    """Every run that is not a speed stage must be unchanged, so the default path may not even round."""
    cfg = RewardConfig(level_complete=100.0)
    assert completion_bonus(cfg) == cfg.level_complete
    assert completion_bonus(cfg, None, 131.25) == cfg.level_complete
    assert completion_bonus(cfg, 95.0, None) == cfg.level_complete, "a target with no official time pays plain"
    assert completion_bonus(cfg, 0.0, 131.25) == cfg.level_complete
    assert completion_bonus(cfg, 95.0, 0.0) == cfg.level_complete, "a completion frame with no clock pays plain"
    assert compute_reward(cfg, snapshot(), snapshot(level_complete=True), {}).parts["level_complete"] == 100.0


def test_the_bonus_scales_with_the_target_over_the_official_time():
    cfg = RewardConfig(level_complete=100.0)
    assert completion_bonus(cfg, 95.0, 95.0) == 100.0, "exactly the S-rank time pays the plain weight"
    assert completion_bonus(cfg, 95.0, 47.5) == 200.0, "half the target is the 2.0 ceiling"
    assert abs(completion_bonus(cfg, 95.0, 76.0) - 125.0) < 1e-9
    # Above the target the ratio is mapped into (MIN, 1] rather than used raw, so the floor is an asymptote:
    # 0.25 + 0.75 * 0.5 = 0.625.
    assert abs(completion_bonus(cfg, 95.0, 190.0) - 62.5) < 1e-9, "twice the target pays well under the plain weight"


def test_the_bonus_is_bounded_so_finishing_always_beats_not_finishing():
    """The floor is the whole safety property: the void-farming post-mortem says a completion must never be
    worth less than staying in the level, whatever the clock says."""
    cfg = RewardConfig(level_complete=100.0)
    assert completion_bonus(cfg, 95.0, 95.0 * 40) > 25.0, "a crawl bottoms out ABOVE 0.25, never at or below"
    assert completion_bonus(cfg, 95.0, 95.0 * 1e6) > 25.0, "however slow, the floor is never actually reached"
    assert completion_bonus(cfg, 95.0, 1e-3) == 200.0, "and a bogus near-zero time cannot pay unbounded"
    assert (SPEED_BONUS_MIN, SPEED_BONUS_MAX) == (0.25, 2.0)


def test_a_slower_completion_always_pays_strictly_less_than_a_faster_one():
    """The 2026-09-18 review's finding: a hard `max(ratio, 0.25)` clip is FLAT wherever the policy actually is.

    Measured on the live spec_0-1 log: 68 fresh Level 0-1 completions, median 481.9 s, against a 90 s target
    (0.75 x an S-rank 120 s). Under the clip 58 of those 68 paid exactly 25.0 with d(bonus)/d(time) == 0 -- a
    flat 75% pay cut carrying no information about the clock at all. Every one of them must now be separated.
    """
    cfg = RewardConfig(level_complete=100.0)
    target = 90.0
    times = [900.0, 600.0, 481.9, 400.0, 300.0, 243.4, 180.0, 120.0, 90.0, 60.0]
    paid = [completion_bonus(cfg, target, t) for t in times]
    assert all(a < b for a, b in zip(paid, paid[1:])), "strictly decreasing in the clock over the whole range"
    assert len({round(p, 6) for p in paid}) == len(times), "no two of them may pay the same"
    # The two numbers the live run sits between, and the plain weight at exactly the target.
    assert abs(completion_bonus(cfg, target, 481.9) - 39.01) < 0.01
    assert abs(completion_bonus(cfg, target, 243.4) - 52.74) < 0.01
    assert completion_bonus(cfg, target, target) == 100.0


def test_compute_reward_pays_the_scaled_bonus_on_the_completion_edge():
    cfg = RewardConfig(**GATES_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(level_complete=True), {},
                           campaign=CampaignStep(), target_seconds=95.0, official_seconds=47.5).parts
    assert parts["level_complete"] == 200.0
    # Not on a step that is not the rising edge, and not on a step that never completed.
    both = compute_reward(cfg, snapshot(level_complete=True), snapshot(level_complete=True), {},
                          target_seconds=95.0, official_seconds=47.5).parts
    assert "level_complete" not in both
    # ... and still paid on a player-less completion frame, which is when a real one usually arrives.
    cur = dict(snapshot(level_complete=True))
    cur.pop("player")
    assert compute_reward(cfg, snapshot(), cur, {}, target_seconds=95.0,
                          official_seconds=190.0).parts["level_complete"] == 62.5


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
