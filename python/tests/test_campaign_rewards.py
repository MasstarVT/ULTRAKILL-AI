"""Campaign reward terms. No game needed:  python tests/test_campaign_rewards.py  (or pytest)."""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.rewards import CampaignStep, RewardConfig, compute_reward  # noqa: E402

CAMPAIGN_WEIGHTS = {"time": 0.01, "checkpoint": 10.0, "arena_clear": 10.0, "door_unlock": 3.0, "novelty": 0.5, "path": 0.1}
CAMPAIGN_PARTS = ("time", "checkpoint", "arena_clear", "door_unlock", "novelty", "path")


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
        "cfg", "prev", "cur", "enemy_max_health", "died", "campaign", "buttons"]


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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
