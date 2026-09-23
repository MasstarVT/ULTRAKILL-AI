"""A SPEED stage's `fall_hp` weight (2026-09-22): the rule, the constant, the detector, and the farm bounds.

No game needed:  python tests/test_speed_fall_hp_weight.py   (or pytest)

WHY THE WEIGHT EXISTS, in one paragraph. Level 0-1's pit and shaft are NON-INSTAKILL DeathZones: the game hurts
the player by min(50, hp - 1), puts them back on the walkway, and only `FakeHurt`s at 1 HP
(decompiled/DeathZone.cs). So two falls leave 1 HP, and 89% of the rung-85 recordings got there. On Brutal a
1-HP player dies to the next hit: 18 of the 31 recorded deaths (runs/probe_0-1_rung85; 16 of 32 in
runs/probe_0-1_brutal) had a rescue as their last HP drop, a median ~300 decisions after it -- 21 of 28
recordings entered arena 2 at exactly 1 HP and 13 of them died there; the 7 that entered at 50-100 HP died
there 0 times (one-sided Fisher p = 0.0054). The fall was charged `damage_taken` 0.01/HP -- 0.5, 0.49, then
nothing -- and the death it sets up arrives where GAE's direct trace is ~1e-7. `fall_hp` charges the HP a
rescue removes, at the fall. Deaths are the head of the rung-85 loss: +8.07 s each (live OLS, n=1,712).

THE LEVER SHIPPED DORMANT (2026-09-22), WAS SWITCHED ON 2026-09-23, AND WAS REVERTED THE SAME DAY. It was built,
reviewed and tested for the 85 s rung, and first NOT switched on: the live policy had started an unexplained slide
just before the switch (bucket medians 96.8-102.1 s for seven buckets, then 110.2 and 108.4 after the 55.65M
trainer restart), and a new reward term judged against a moving baseline proves nothing (docs/project-log.md,
2026-09-22). The 2026-09-23 replay from 52.25M showed the slide was the 55.645M weights, not the setup, and four
flat buckets (medians 100.1-103.0 s) became the baseline; the weight went live as ONE line in `speed.rewards:`
for round 15 of spec_0-1_speed (docs/project-log.md, 2026-09-23). It was REMOVED at the round-16 boundary by the
pre-registered INERT rule: over 4 full post-switch buckets `rescue_hp` per episode (94.9 / 105.3 / 96.5 / 102.3)
and floors (1.05 / 1.22 / 1.25 / 1.20) never made two consecutive buckets below the baseline band (< 91.3 or
< 1.09); weights kept (docs/project-log.md, 2026-09-23 13:30). `test_the_lever_ships_dormant_again...` pins
that the line is gone and that it is still the whole switch; `plan()` below is the shipped plan PLUS the line, so
every sizing and farm test here still reads the lever at its designed weight.

WHAT THIS FILE PINS. The measured constants and the sizing band, the ceiling, the detector (a rescue is a
>= 12 m one-decision move on a non-death step, and nothing else is), that the weight at 0 changes no step,
that the readings reach episodes.jsonl, and the farm table: the term never pays, a fall stays cheaper than a
death, refusing the pit never pays, finishing and faster are untouched, and the one known asymmetry (a tilt
against bringing HP to a fall) is bounded.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_driver  # noqa: E402
from test_campaign_env import FakeLevel, add_parts, forward, idle, make_env  # noqa: E402
from ultrakill_ai import progress as progress_mod  # noqa: E402
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, RESCUE_JUMP_M, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.rewards import (SPEED_BONUS_MIN, CampaignStep, RewardConfig,  # noqa: E402
                                  completion_bonus, compute_reward)

PLAN = ROOT / "configs" / "specialists.yaml"
WEIGHT = 0.04

# ---------------------------------------------------------------------------------------------------------
# MEASURED CONSTANTS, re-derived with the env's own rule from runs/probe_0-1_rung85 (NEW, ckpt 55,695,046, 28
# recordings) and runs/probe_0-1_brutal (PRE, ckpt 45,685,042, 26). If a re-measurement moves one of these,
# the weight has to be re-derived.
# ---------------------------------------------------------------------------------------------------------
ZONE_DAMAGE = 50               # DeathZone.damage; every recorded HP-costing rescue is -50 or a drop to 1
DECISION_PAIRS = 114_877       # NEW + PRE
TELEPORTS_NEW, TELEPORTS_PRE = 375, 483       # >= 12 m, non-death: every rescue, free ones included
HP_COSTING_NEW, HP_COSTING_PRE = 75, 71       # ... of which removed HP (146, all on the known rescue points)
FLOORS_NEW, FLOORS_PRE = 37, 32               # ... of which took the player from above 1 HP to 1
RECORDINGS_NEW, RECORDINGS_PRE = 28, 26
DEATHS_NEW, DEATHS_PRE = 31, 32
LINKED_NEW, LINKED_PRE = 18, 16               # deaths whose last HP drop was a rescue
GAP_NEW, GAP_PRE = 299.5, 203.0               # median decisions from that rescue to the death
RESCUE_HP_MEDIAN = 99.0                       # HP rescues removed per recording, both sets
RESCUE_HP_MEAN_NEW, RESCUE_HP_MEAN_PRE = 117.6, 119.7
ARENA2_AT_1HP, ARENA2_AT_1HP_DIED = 21, 13    # NEW: entered arena 2 at exactly 1 HP / recordings that died there
ARENA2_HIGH, ARENA2_HIGH_DIED = 7, 0          # NEW: entered at 50-100 HP
SECONDS_PER_DEATH = 8.07                      # live OLS since 52.25M, kills and oob held fixed
GAMMA, GAE_LAMBDA = 0.998, 0.95
DECISIONS_PER_S = 15.0
TARGET, MEDIAN = 85.0, 100.1                  # the focus rung, and the probe's (= live) median completion
RUNGS_AFTER_THE_PIT = 6


def shipped_plan():
    return campaign_driver.load_plan(PLAN)


def _plan_with(weight: float | None):
    """The shipped plan with `speed.rewards.fall_hp` set to `weight`, or removed when it is None.

    Written to a temp file and read back through `load_plan`, so the edit goes through the same validation
    against RewardConfig's fields that a real plan edit does.
    """
    data = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    rewards = data["speed"].setdefault("rewards", {})
    if weight is None:
        rewards.pop("fall_hp", None)
    else:
        rewards["fall_hp"] = weight
    out = Path(tempfile.mkdtemp()) / "plan.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return campaign_driver.load_plan(out)


def plan():
    """The plan with the lever on at the designed weight: the shipped plan plus the one `fall_hp` line.

    That was the shipped plan itself during round 15 (2026-09-23); since the same day's revert it is the form a
    re-activation would ship, and every sizing and farm test in this file reads the lever through it.
    """
    return _plan_with(WEIGHT)


def dormant_plan():
    """The shipped plan with the one `fall_hp` line removed: the 2026-09-22 form, and again since the 2026-09-23
    revert -- so today it builds exactly what the shipped plan builds."""
    return _plan_with(None)


def speed_env(level: str = "Level 0-1") -> dict:
    return campaign_driver.stage_config(plan(), level, kind=campaign_driver.SPEED)["env"]


def speed_rewards() -> RewardConfig:
    return EnvConfig.from_dict(speed_env()).rewards


def step(cfg: RewardConfig, rescue_hp: float):
    """One campaign decision's reward with nothing happening but a rescue of `rescue_hp`."""
    frame = {"stats": {}, "player": {"pos": (0.0, 0.0, 0.0), "hp": 100, "dead": False}}
    return compute_reward(cfg, frame, frame, {}, died=False, campaign=CampaignStep(rescue_hp=rescue_hp))


class RescueLevel(FakeLevel):
    """FakeLevel with health and the game's non-instakill DeathZone.

    `rescue_next` is DeathZone.OnTriggerEnter on a notInstakill zone: hurt by min(50, hp - 1) (nothing at 1 HP),
    then teleport -- here 20 m along the corridor, back if there is room, else forward. `hit_next` is an
    ordinary hit while the player moves as usual. `heal_teleport_next` is a teleport that RAISES health.
    A checkpoint respawn heals to 100, as the game's does.
    """

    def __init__(self):
        super().__init__()
        self.hp = 100.0
        self.rescue_next = False
        self.hit_next = 0.0
        self.heal_teleport_next = False

    def _load(self) -> None:
        super()._load()
        self.hp = 100.0

    def reset(self, *args, **kwargs):
        self.hp = 100.0
        return super().reset(*args, **kwargs)

    def _teleport(self) -> None:
        self.z = self.z - 20.0 if self.z >= 20.0 else self.z + 20.0

    def step(self, action: dict) -> dict:
        if self.rescue_next:
            self.rescue_next = False
            if self.hp > ZONE_DAMAGE:
                self.hp -= ZONE_DAMAGE
            elif self.hp > 1:
                self.hp = 1.0
            self._teleport()
        if self.heal_teleport_next:
            self.heal_teleport_next = False
            self.hp = 100.0
            self._teleport()
        if self.hit_next:
            self.hp = max(0.0, self.hp - self.hit_next)
            self.hit_next = 0.0
        return super().step(action)

    def _obs(self, event: str | None = None) -> dict:
        obs = super()._obs(event)
        obs["player"]["hp"] = 0 if self.dead else self.hp
        return obs


def rescue_env(weight: float = WEIGHT):
    env, _ = make_env(RewardConfig(time=0.0, damage_taken=0.01, death=12.0, fall_hp=weight))
    level = RescueLevel()
    env.client = level
    return env, level


def walk(env, n: int, parts: dict) -> dict:
    info = {}
    for _ in range(n):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        assert not terminated and not truncated
    return info


def one(env, parts: dict) -> dict:
    _, _, terminated, truncated, info = env.step(idle())
    add_parts(parts, info)
    assert not terminated and not truncated
    return info


# ---------------------------------------------------------------------------------------------------------
# THE RULE AND THE CONSTANT
# ---------------------------------------------------------------------------------------------------------
def test_the_lever_ships_dormant_again_since_the_2026_09_23_revert_and_is_one_plan_line():
    """Live for round 15 only (2026-09-23), REVERTED at the round-16 boundary by the pre-registered INERT rule."""
    shipped = shipped_plan()
    assert "fall_hp" not in shipped.speed_rewards, \
        "REVERTED 2026-09-23 (inert at 4 full buckets): the shipped plan does not name the weight"
    off = campaign_driver.stage_config(shipped, "Level 0-1", kind=campaign_driver.SPEED)["env"]
    assert "fall_hp" not in off["rewards"], "so the shipped speed config carries no `fall_hp` key"
    assert EnvConfig.from_dict(off).rewards.fall_hp == 0.0, "... and builds the inert default"
    assert RewardConfig().fall_hp == 0.0, "the default is inert"
    complete = EnvConfig.from_dict(campaign_driver.stage_config(shipped, "Level 0-1")["env"]).rewards
    assert complete.fall_hp == 0.0, "a COMPLETE stage is untouched, and so is every config that never names it"
    dormant = campaign_driver.stage_config(dormant_plan(), "Level 0-1", kind=campaign_driver.SPEED)["env"]
    assert dormant == off, "removing the line from the shipped plan changes nothing: it is already gone"
    # The line is the whole switch: set (the round-15 form, and any re-activation) the speed stage builds the
    # weight, and the two generated speed configs differ in that one weight and nothing else.
    live = campaign_driver.stage_config(plan(), "Level 0-1", kind=campaign_driver.SPEED)["env"]
    assert live["rewards"]["fall_hp"] == WEIGHT and EnvConfig.from_dict(live).rewards.fall_hp == WEIGHT
    assert speed_rewards().fall_hp == WEIGHT, "the sizing tests below read the lever at its designed weight"
    assert {k: v for k, v in live.items() if k != "rewards"} == {k: v for k, v in off.items() if k != "rewards"}
    assert {k: v for k, v in live["rewards"].items() if k != "fall_hp"} == off["rewards"]


def test_every_shipped_non_speed_config_builds_the_inert_default():
    import yaml
    for name in ("campaign_gates_full.yaml", "campaign_0-1.yaml", "cybergrind.yaml"):
        path = ROOT / "configs" / name
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rewards = ((data.get("env") or {}).get("rewards")) or {}
        assert "fall_hp" not in rewards, name
        assert EnvConfig.from_dict(data.get("env") or {}).rewards.fall_hp == 0.0, name


def test_the_override_reaches_every_speed_stage_and_no_complete_stage():
    """With the line set (`plan()`, the re-activation form), every speed stage gets it and no complete stage."""
    p = plan()
    for level in ("Level 0-1", "Level 0-2", "Level 0-3"):
        assert campaign_driver.stage_config(p, level, kind=campaign_driver.SPEED)["env"]["rewards"]["fall_hp"] == WEIGHT
        assert "fall_hp" not in campaign_driver.stage_config(p, level)["env"]["rewards"], level


def test_an_env_config_saved_before_the_field_existed_still_loads_as_zero():
    """models/<run>/env_config.yaml files written before 2026-09-22 carry no `fall_hp`: they must load inert."""
    data = dict(speed_env())
    data["rewards"] = {k: v for k, v in data["rewards"].items() if k != "fall_hp"}
    assert EnvConfig.from_dict(data).rewards.fall_hp == 0.0


def test_the_observation_and_action_spaces_do_not_move():
    env = UltrakillEnv(EnvConfig.from_dict(speed_env()))
    try:
        assert env.observation_space.shape == (479,)
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3], "12 dimensions, 45 logits"
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------
# THE DETECTOR, end to end on the fake corridor
# ---------------------------------------------------------------------------------------------------------
def test_two_falls_are_charged_per_hp_and_a_third_at_1_hp_is_free():
    env, level = rescue_env()
    env.reset()
    parts: dict[str, float] = {}
    walk(env, 20, parts)                       # z 40
    level.rescue_next = True
    info = one(env, parts)                     # 100 -> 50, z 40 -> 20
    assert math.isclose(info["reward_parts"]["fall_hp"], -50 * WEIGHT)
    assert math.isclose(info["reward_parts"]["damage_taken"], -0.5)
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"]) == (1, 50.0, 0)
    level.rescue_next = True
    info = one(env, parts)                     # 50 -> 1, z 20 -> 0
    assert math.isclose(info["reward_parts"]["fall_hp"], -49 * WEIGHT)
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"]) == (2, 99.0, 1)
    level.rescue_next = True
    info = one(env, parts)                     # 1 -> 1 (FakeHurt), z 0 -> 20
    assert "fall_hp" not in info["reward_parts"] and "damage_taken" not in info["reward_parts"]
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"]) == (3, 99.0, 1)
    assert math.isclose(parts["fall_hp"], -99 * WEIGHT)
    assert info["hp_lost_other"] == 0.0
    env.close()


def test_the_death_step_and_the_respawn_teleport_are_never_charged():
    """The lethal hit is `damage_taken` + `death`'s; the respawn (40 -> 20 m, healed) is never compared."""
    env, level = rescue_env()
    env.reset()
    parts: dict[str, float] = {}
    walk(env, 20, parts)                       # z 40, past the checkpoint at 20
    level.hit_next = 70.0
    walk(env, 1, parts)                        # 100 -> 30: an ordinary hit
    level.kill_next = True
    info = one(env, parts)                     # dies at z 42, respawns at z 20 on 100 HP
    assert info["deaths"] == 1 and "death" in info["reward_parts"]
    assert math.isclose(info["reward_parts"]["damage_taken"], -0.3), "the lethal hit is the HP left"
    walk(env, 3, parts)
    info = one(env, parts)
    assert "fall_hp" not in parts
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"]) == (0, 0.0, 0)
    assert info["hp_lost_other"] == 70.0, "the ordinary hit, and not the lethal one"
    env.close()


def test_a_teleport_that_heals_counts_but_charges_nothing():
    env, level = rescue_env()
    env.reset()
    parts: dict[str, float] = {}
    walk(env, 5, parts)
    level.hit_next = 82.0
    walk(env, 1, parts)                        # 100 -> 18
    level.heal_teleport_next = True
    info = one(env, parts)                     # 18 -> 100 across a 20 m jump
    assert "fall_hp" not in parts
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"]) == (1, 0.0, 0)
    env.close()


def test_an_ordinary_hit_is_other_damage_and_not_a_rescue():
    env, level = rescue_env()
    env.reset()
    parts: dict[str, float] = {}
    walk(env, 3, parts)
    level.hit_next = 30.0
    info = walk(env, 1, parts)                 # a 2 m move with a 30 HP hit
    assert "fall_hp" not in parts
    assert math.isclose(info["reward_parts"]["damage_taken"], -0.3)
    assert (info["rescues"], info["rescue_hp"], info["hp_lost_other"]) == (0, 0.0, 30.0)
    env.close()


def test_the_threshold_is_twelve_metres_and_needs_both_frames():
    """Direct on `_note_rescue`: 11.99 m with an HP drop is other damage; 12.0 m is a rescue; nothing without HP."""
    env, _ = make_env()
    assert RESCUE_JUMP_M == 12.0

    def frame(z, hp, dead=False):
        return {"player": {"pos": [0.0, 1.0, z], "hp": hp, "dead": dead}}

    assert env._note_rescue(frame(0.0, 100), frame(11.99, 50), died=False) == 0.0
    assert env._hp_lost_other == 50.0 and env._rescues == 0
    assert env._note_rescue(frame(0.0, 100), frame(12.0, 50), died=False) == 50.0
    assert env._rescues == 1 and env._rescue_hp == 50.0
    assert env._note_rescue(frame(0.0, 100), frame(40.0, 0, dead=True), died=True) == 0.0, "a death"
    assert env._note_rescue(frame(0.0, 100), frame(40.0, 50), died=True) == 0.0, "the env's own death verdict"
    assert env._note_rescue(frame(0.0, 100), {"player": None}, died=False) == 0.0
    assert env._note_rescue(frame(0.0, None), frame(40.0, 50), died=False) == 0.0, "no HP reading: no charge"
    assert env._note_rescue({}, frame(40.0, 50), died=False) == 0.0
    assert (env._rescues, env._rescue_hp, env._hp_lost_other) == (1, 50.0, 50.0), "none of those counted"
    env.close()


def test_the_counters_reset_with_the_episode():
    env, level = rescue_env()
    env.reset()
    parts: dict[str, float] = {}
    walk(env, 20, parts)
    level.rescue_next = True
    one(env, parts)
    env.reset()
    info = one(env, {})
    assert (info["rescues"], info["rescue_hp"], info["rescue_floored"], info["hp_lost_other"]) == (0, 0.0, 0, 0.0)
    env.close()


def test_the_weight_at_zero_changes_no_step_and_the_readings_do_not_depend_on_it():
    runs = []
    for weight in (0.0, WEIGHT):
        env, level = rescue_env(weight)
        env.reset()
        parts: dict[str, float] = {}
        walk(env, 20, parts)
        level.rescue_next = True
        one(env, parts)
        level.hit_next = 10.0
        walk(env, 2, parts)
        level.rescue_next = True
        info = one(env, parts)
        env.close()
        runs.append((parts, {k: info[k] for k in ("rescues", "rescue_hp", "rescue_floored", "hp_lost_other")}))
    (zero, zero_info), (live, live_info) = runs
    assert "fall_hp" not in zero, "a zero weight must not even create the key"
    assert math.isclose(live["fall_hp"], -WEIGHT * (50 + 39))
    assert {k: v for k, v in live.items() if k != "fall_hp"} == zero, "nothing else moved"
    assert zero_info == live_info == {"rescues": 2, "rescue_hp": 89.0, "rescue_floored": 1, "hp_lost_other": 10.0}


def test_the_readings_reach_episodes_jsonl_and_not_the_monitor():
    for key in ("rescues", "rescue_hp", "rescue_floored", "hp_lost_other"):
        assert key in progress_mod.EPISODE_LOG_RAW, key
        assert key not in CAMPAIGN_INFO_KEYS, "the Monitor's columns are unchanged"


# ---------------------------------------------------------------------------------------------------------
# THE SIZING
# ---------------------------------------------------------------------------------------------------------
def per_hp(linked: int, recordings: int, gap: float | None, hp: float = RESCUE_HP_MEDIAN) -> float:
    """The expected `death` penalty a rescue's HP sets up, per HP, net of the `damage_taken` already charged."""
    discount = GAMMA ** gap if gap is not None else 1.0
    return linked / recordings * speed_rewards().death * discount / hp - speed_rewards().damage_taken


def test_the_band_the_weight_was_chosen_from():
    new, pre = per_hp(LINKED_NEW, RECORDINGS_NEW, GAP_NEW), per_hp(LINKED_PRE, RECORDINGS_PRE, GAP_PRE)
    low = per_hp(LINKED_NEW, RECORDINGS_NEW, GAP_NEW, hp=RESCUE_HP_MEAN_NEW)
    assert 0.032 < new < 0.034 and 0.039 < pre < 0.041 and 0.025 < low < 0.027, (new, pre, low)
    assert low < WEIGHT <= max(new, pre) + 0.001, "the top of the discounted band"
    undiscounted = (per_hp(LINKED_NEW, RECORDINGS_NEW, None), per_hp(LINKED_PRE, RECORDINGS_PRE, None))
    assert all(0.064 < u < 0.069 for u in undiscounted), undiscounted
    assert WEIGHT < min(undiscounted), "and well under the undiscounted price"
    # Why the charge has to be AT the fall: the death arrives where GAE's direct trace is gone.
    assert (GAMMA * GAE_LAMBDA) ** GAP_NEW < 1e-6 and (GAMMA * GAE_LAMBDA) ** GAP_PRE < 1e-4


def test_the_weight_stays_under_the_ceiling():
    """(fall_hp + damage_taken) x 99 < death: the two falls to 1 HP always cost less than the death they risk."""
    cfg = speed_rewards()
    ceiling = cfg.death / RESCUE_HP_MEDIAN - cfg.damage_taken
    assert 0.110 < ceiling < 0.112, ceiling
    assert cfg.fall_hp < 0.5 * ceiling
    assert (cfg.fall_hp + cfg.damage_taken) * RESCUE_HP_MEDIAN < cfg.death + cfg.damage_taken * 1


def test_what_one_fall_now_costs():
    cfg = speed_rewards()
    first = (cfg.fall_hp + cfg.damage_taken) * ZONE_DAMAGE
    second = (cfg.fall_hp + cfg.damage_taken) * (ZONE_DAMAGE - 1)
    assert math.isclose(first, 2.5) and math.isclose(second, 2.45)
    assert math.isclose(cfg.damage_taken * ZONE_DAMAGE, 0.5), "was 0.5, and 0 at 1 HP"
    assert "fall_hp" not in step(cfg, 0.0).parts, "a fall at 1 HP removes nothing and is charged nothing more"


def test_the_evidence_the_mechanism_rests_on():
    """Pinned so a re-measurement that moves it is noticed. Associations, not causes."""
    assert LINKED_NEW / DEATHS_NEW > 0.5 and LINKED_PRE / DEATHS_PRE == 0.5
    assert HP_COSTING_NEW + HP_COSTING_PRE == 146 and FLOORS_NEW / RECORDINGS_NEW > 1.0
    # One-sided Fisher on the NEW arena-2 table: 13 of 21 died at 1 HP, 0 of 7 at 50-100 HP.
    n, k, high = ARENA2_AT_1HP + ARENA2_HIGH, ARENA2_AT_1HP_DIED + ARENA2_HIGH_DIED, ARENA2_HIGH
    p = math.comb(n - k, high) / math.comb(n, high)
    assert ARENA2_HIGH_DIED == 0 and 0.005 < p < 0.006, p


def test_the_episode_scale_of_the_term():
    cfg = speed_rewards()
    mean = RESCUE_HP_MEAN_NEW * cfg.fall_hp
    assert 4.5 < mean < 5.0, mean
    assert RESCUE_HP_MEDIAN * cfg.fall_hp < 4.0
    assert mean / 418.64 < 0.02, "about 1% of the episode return, not a rewrite of it"
    assert mean < cfg.death, "smaller than a single death"


# ---------------------------------------------------------------------------------------------------------
# THE FARM TABLE. Finishing must beat not finishing, faster must beat slower, and living must beat dying.
# ---------------------------------------------------------------------------------------------------------
def test_the_term_is_non_positive_in_every_state_and_at_every_weight():
    for weight in (0.0, WEIGHT, 1.0, 100.0):
        cfg = RewardConfig(time=0.0, fall_hp=weight)
        for rescue_hp in (-50.0, 0.0, 1.0, 49.0, 50.0, 1e6, float("nan"), float("-inf")):
            assert step(cfg, rescue_hp).total <= 0.0, (weight, rescue_hp)
        assert step(cfg, -50.0).total == 0.0, "a negative reading is clamped, never paid"
        assert step(cfg, 50.0).total == -50.0 * weight


def test_a_fall_is_always_cheaper_than_a_death():
    cfg = speed_rewards()
    worst_fall = (cfg.fall_hp + cfg.damage_taken) * ZONE_DAMAGE
    cheapest_death = cfg.death + cfg.damage_taken * 1.0
    assert worst_fall * 4 < cheapest_death
    life = (cfg.fall_hp + cfg.damage_taken) * RESCUE_HP_MEDIAN
    assert life < cheapest_death / 2, "every fall a life can take, together, is under half a death"
    # Dying to come back on 100 HP buys nothing: the charge is per HP lost, so a full bar has MORE to lose.
    assert cfg.fall_hp * 100 > cfg.fall_hp * 1


def test_refusing_to_cross_the_pit_can_never_pay():
    """Counted pessimistically: the six rungs past the pit and the completion bonus at its floor only."""
    cfg = speed_rewards()
    forfeit = cfg.gate * RUNGS_AFTER_THE_PIT + SPEED_BONUS_MIN * cfg.level_complete
    saved = cfg.fall_hp * RESCUE_HP_MEDIAN
    assert forfeit > 25 * saved, (forfeit, saved)
    # A run sent back behind the pit by a death has been paid its early rungs, but 4 remain plus the bonus.
    assert cfg.gate * 4 + SPEED_BONUS_MIN * cfg.level_complete > 20 * saved
    assert EnvConfig.from_dict(speed_env()).stuck_seconds > 0, "and standing still is truncated"


def test_faster_still_beats_slower_and_the_extra_care_it_can_buy_is_bounded():
    """The charge does not depend on the clock, so the clock's own ordering is untouched.

    The most caution it can justify: one avoided 50-HP fall is worth `fall_hp` x 50 = 2.0, against the
    objective price of a second at the rung, `time` x 15 + the bonus slope = 0.94/s -- about 2.1 s.
    """
    cfg = speed_rewards()
    prev = None
    for seconds in (61.996, 80.0, 85.0, 100.1, 150.0, 400.0):
        value = completion_bonus(cfg, TARGET, seconds) - cfg.time * seconds * DECISIONS_PER_S
        if prev is not None:
            assert value < prev, seconds
        prev = value
    per_second = cfg.time * DECISIONS_PER_S + cfg.level_complete * (1 - SPEED_BONUS_MIN) * TARGET / MEDIAN ** 2
    assert 0.93 < per_second < 0.95, per_second
    assert cfg.fall_hp * ZONE_DAMAGE / per_second < 2.5
    assert cfg.fall_hp * RESCUE_HP_MEDIAN / per_second < 4.5, "a whole life's falls: under 4.5 s of care"
    assert cfg.fall_hp * RESCUE_HP_MEDIAN / per_second < SECONDS_PER_DEATH, "less than one death's seconds"


def test_the_known_asymmetry_is_bounded():
    """Per HP, a run that floors anyway pays less the less HP it brings: at most (fall_hp - damage_taken) x 49.

    Entering the pit on 51 HP instead of 100 saves 49 HP of `fall_hp` and pays those 49 HP as enemy damage at
    `damage_taken`. Watched live by `hp_lost_other`; bounded here at well under a death.
    """
    cfg = speed_rewards()
    tilt = (cfg.fall_hp - cfg.damage_taken) * (ZONE_DAMAGE - 1)
    assert 1.4 < tilt < 1.5, tilt
    assert tilt < cfg.death / 8


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
