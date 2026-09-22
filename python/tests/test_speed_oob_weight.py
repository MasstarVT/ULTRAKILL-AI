"""A SPEED stage's `oob` weight (2026-09-22): the rule, the constant, the wiring, and the farm bounds.

No game needed:  python tests/test_speed_oob_weight.py   (or pytest)

WHY THE WEIGHT EXISTS, in one paragraph. The per-leg time budget of Level 0-1 ON BRUTAL (26 recorded episodes
in runs/probe_0-1_brutal, 21 completions, median official 111.43 s) put the largest single named loss on THE
PIT: the stretch between gate rungs 6 (40,-9,552) and 4 (66,21,640), where the agent falls off the walkway and
the game teleports it back without killing it. 217 rescues over the 21 completions, every completion had at
least one, and the probe's slowest third spends 15.6 s more than its fastest third on that one leg -- 73% of
the whole fast/slow gap. It is not a death and it is not a wedge: one fall costs a median 1.40 s and buys a
median 3.9 m of net displacement. The env has COUNTED the condition every step since the exploration archive
was keyed on ground (`_oob_steps`, reported as `oob_frac`) and nothing ever read it back, so a fall was paid
`time` -0.02 and nothing else: `novelty` pays 0 with no ground under the player (that is the 2026-09-16 fix
that stopped diving off the map being an unbounded income stream) and `gate_approach` pays only a NEW best
closeness, which falling never sets. `oob` puts the cost at the decision the mistake is made, which is the one
thing S1 proved this policy cannot get from the deferred completion channel at gamma 0.998.

WHAT THIS FILE PINS. Every measured constant, the sizing band, the wiring (the reward's condition IS the
metric's condition, so `oob_frac` can be used to judge the lever), and the farm table: the term can never pay,
it cannot make dying cheaper than being rescued, it cannot make refusing to cross the pit worth more than
crossing it, finishing still beats not finishing and faster still beats slower. It also pins the bound the
weight was actually sized by -- the SHAFT leg, where airborne decisions legitimately out-earn grounded ones
and a weight above the crossover would push the policy off a climb it has already learned.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_driver  # noqa: E402
from ultrakill_ai.env import EnvConfig  # noqa: E402
from ultrakill_ai.rewards import (SPEED_BONUS_MIN, CampaignStep, RewardConfig,  # noqa: E402
                                  completion_bonus, compute_reward)

PLAN = ROOT / "configs" / "specialists.yaml"

# ---------------------------------------------------------------------------------------------------------
# MEASURED CONSTANTS. Every one re-derived per decision from runs/probe_0-1_brutal with the env's OWN rule
# (`ground_drop is None or >= ground_ray_length - 0.5`), completions only unless stated. If a re-measurement
# moves one of these, the weight has to be re-derived.
# ---------------------------------------------------------------------------------------------------------
COMPLETIONS = 21               # of 26 recordings (0.81; the live fresh rate over the same era is 0.90-0.95)
OFFICIAL_MEDIAN = 111.43       # s, median official time of those completions
LIVE_POOLED_MEDIAN = 111.88    # s, the live fresh completions of the 8 full 500k buckets since 42,185,602
OOB_MEDIAN = 167               # decisions per completion with no ground within 30 m
OOB_MEAN = 236.3
OOB_MIN, OOB_MAX = 64, 896     # rollout_13 (87.71 s) and rollout_14 (159.44 s)
OLS_INTERCEPT, OLS_SLOPE, OLS_R2 = 93.90, 0.0985, 0.581     # seconds = a + b * oob_decisions
SPELLS, SPELL_MEAN, SPELL_MAX = 498, 10.0, 89               # contiguous oob runs
RESCUES, DECISIONS_PER_RESCUE = 217, 22.87                  # spells ending in a >10 m one-decision teleport
# Share of all oob decisions by gate leg (the recording's own `target_hops`): the pit is 84.7% of the charge.
SHARE_PIT_EDGE, SHARE_PIT, SHARE_SHAFT = 0.558, 0.289, 0.109
# Mean reward PER DECISION, oob vs grounded, on the three legs that carry the charge (all 26 recordings).
SHAFT_OOB, SHAFT_GROUND = 0.1386, 0.0931        # the shaft climb EARNS MORE airborne: the bound below
PIT_EDGE_OOB, PIT_EDGE_GROUND = 0.0291, 0.1344
PIT_OOB, PIT_GROUND = 0.0186, 0.0396
# The clock and the horizon.
DECISIONS_PER_S = 15.0         # fixed_fps 30 / frameskip 2, exactly
FALL_SECONDS = 1.40            # median seconds airborne in one fall
PIT_TO_EXIT = 1180             # decisions from the pit (crossed 25-40 s in) to a median 111.4 s completion
GAMMA = 0.998
TARGET = 100.0                 # the focus rung the weight was derived at (configs/generated/spec_0-1_speed.yaml)
# The ladder a policy that REFUSED to cross the pit would forfeit, at the plan's own weights: rungs 5..0 of
# the gate ladder, and the completion bonus at its floor. Everything else it gives up (2 arenas, their door
# unlocks, the remaining checkpoints, ~400 m of gate_approach) is deliberately NOT counted, so the margin
# below is the pessimistic one.
RUNGS_AFTER_THE_PIT = 6


def plan():
    return campaign_driver.load_plan(PLAN)


def speed_env(level: str = "Level 0-1") -> dict:
    return campaign_driver.stage_config(plan(), level, kind=campaign_driver.SPEED)["env"]


def speed_rewards() -> RewardConfig:
    return EnvConfig.from_dict(speed_env()).rewards


def bonus(cfg: RewardConfig, seconds: float, target: float = TARGET) -> float:
    return completion_bonus(cfg, target, seconds)


def bonus_slope(cfg: RewardConfig, seconds: float = OFFICIAL_MEDIAN, target: float = TARGET) -> float:
    """How much of the completion bonus ONE MORE SECOND of official time gives up, at `seconds`."""
    return cfg.level_complete * (1.0 - SPEED_BONUS_MIN) * target / (seconds * seconds)


def objective_price(cfg: RewardConfig, seconds_lost: float, official: float = OFFICIAL_MEDIAN) -> float:
    """What one oob decision costs the OBJECTIVE: its `time` charge plus the completion bonus it gives up."""
    return cfg.time + bonus_slope(cfg, official) * seconds_lost


def felt_price(cfg: RewardConfig, seconds_lost: float, official: float = OFFICIAL_MEDIAN) -> float:
    """... and what the policy receives for it, with the bonus discounted across PIT_TO_EXIT decisions."""
    return cfg.time + bonus_slope(cfg, official) * seconds_lost * GAMMA ** PIT_TO_EXIT


def shortfall(cfg: RewardConfig, seconds_lost: float, official: float = OFFICIAL_MEDIAN) -> float:
    return objective_price(cfg, seconds_lost, official) - felt_price(cfg, seconds_lost, official)


def step(cfg: RewardConfig, *, oob: int = 0, **campaign):
    """One campaign decision's reward, with nothing happening but what `campaign` names."""
    frame = {"stats": {}, "player": {"pos": (0.0, 0.0, 0.0), "hp": 100, "dead": False}}
    return compute_reward(cfg, frame, frame, {}, died=False,
                          campaign=CampaignStep(oob_steps=oob, **campaign))


# ---------------------------------------------------------------------------------------------------------
# THE RULE AND THE CONSTANT
# ---------------------------------------------------------------------------------------------------------
def test_the_speed_stage_charges_an_oob_decision_at_0_035():
    p = plan()
    assert p.speed_rewards["oob"] == 0.035, "the plan names the weight"
    assert speed_rewards().oob == 0.035
    complete = EnvConfig.from_dict(campaign_driver.stage_config(p, "Level 0-1")["env"]).rewards
    assert complete.oob == 0.0, "a COMPLETE stage is untouched, and so is every config that never names it"


def test_every_shipped_non_speed_config_builds_the_inert_default():
    """`RewardConfig.oob` defaults to 0.0, so campaign_gates_full, campaign_0-1 and cybergrind are unchanged."""
    import yaml
    for name in ("campaign_gates_full.yaml", "campaign_0-1.yaml", "cybergrind.yaml"):
        path = ROOT / "configs" / name
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rewards = ((data.get("env") or {}).get("rewards")) or {}
        assert "oob" not in rewards, name
        assert EnvConfig.from_dict(data.get("env") or {}).rewards.oob == 0.0, name


def test_the_override_reaches_every_speed_stage_and_no_complete_stage():
    p = plan()
    for level in ("Level 0-1", "Level 0-2", "Level 0-3"):
        assert campaign_driver.stage_config(p, level, kind=campaign_driver.SPEED)["env"]["rewards"]["oob"] == 0.035
        assert "oob" not in campaign_driver.stage_config(p, level)["env"]["rewards"], level


# ---------------------------------------------------------------------------------------------------------
# THE WIRING: the charged condition IS the measured condition, which is what makes `oob_frac` the judge
# ---------------------------------------------------------------------------------------------------------
def test_a_void_fall_is_charged_once_per_decision_and_pays_no_novelty():
    """End to end on the fake level, against the same `falling` switch test_campaign_env.py's novelty pin uses.

    The charge must equal the metric exactly: `oob_frac * length` is `_oob_steps`, and the summed `oob` part
    is that number times the weight. If those two ever disagree the judging plan is measuring a different
    thing from the one being paid for.
    """
    from test_campaign_env import add_parts, forward, make_env

    env, level = make_env(RewardConfig(time=0.02, novelty=0.2, oob=0.035))
    env.reset()
    level.falling = True
    parts: dict[str, float] = {}
    steps = 0
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    assert level.y <= -300.0, "the fake level should have fallen well below the map"
    assert "novelty" not in parts, "a fall must still pay no novelty"
    assert info["oob_frac"] > 0.9
    charged = round(-parts["oob"] / 0.035)
    assert charged == round(info["oob_frac"] * steps), (charged, info["oob_frac"], steps)
    assert abs(parts["oob"]) <= 0.035 * steps + 1e-9, "at most one charge per decision"


def test_the_weight_at_zero_changes_no_step_at_all():
    """Every run before this one, and every complete stage: `r.add` skips zeros, so not even a key appears."""
    from test_campaign_env import add_parts, forward, make_env

    totals = []
    for weight in (0.0, 0.035):
        env, level = make_env(RewardConfig(time=0.02, novelty=0.2, oob=weight))
        env.reset()
        level.falling = True
        parts: dict[str, float] = {}
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        env.close()
        totals.append(parts)
    assert "oob" not in totals[0], "a zero weight must not even create the key"
    assert "oob" in totals[1]
    assert {k: v for k, v in totals[1].items() if k != "oob"} == totals[0], "nothing else moved"


def test_the_term_is_non_positive_in_every_state_and_at_every_weight():
    """It cannot be farmed: `oob_steps` is 0 or 1 and the weight is subtracted. There is no input that pays."""
    for weight in (0.0, 0.035, 1.0, 100.0):
        cfg = RewardConfig(time=0.0, oob=weight)
        assert step(cfg, oob=0).total == 0.0
        assert step(cfg, oob=1).total == -weight
        assert step(cfg, oob=1).total <= 0.0


# ---------------------------------------------------------------------------------------------------------
# THE SIZING
# ---------------------------------------------------------------------------------------------------------
def test_the_band_the_weight_was_chosen_from():
    """0.035 is under the MECHANICAL shortfall and well under the ASSOCIATIONAL one: conservative both ways."""
    cfg = speed_rewards()
    assert abs(bonus_slope(cfg) - 0.604) < 0.005, bonus_slope(cfg)      # per official second, at the median
    mechanical = shortfall(cfg, 1.0 / DECISIONS_PER_S)                  # 0.0667 s of dead time per decision
    associational = shortfall(cfg, OLS_SLOPE)                           # the measured 0.0985 s per decision
    assert 0.035 < mechanical < 0.038, mechanical
    assert 0.052 < associational < 0.056, associational
    assert cfg.oob < mechanical < associational, "the weight under-prices even the mechanical dead time"
    # The shortfall exists because the bonus is delivered PIT_TO_EXIT decisions later at ~9% of face value --
    # the same discounting S1 failed to fix by lengthening the horizon.
    assert 0.08 < GAMMA ** PIT_TO_EXIT < 0.11
    assert felt_price(cfg, 1.0 / DECISIONS_PER_S) < 0.5 * objective_price(cfg, 1.0 / DECISIONS_PER_S)


def test_the_weight_stays_under_the_shaft_crossover():
    """THE BOUND THAT SET THE CONSTANT, and the reason it is not the band's midpoint.

    10.9% of the charge lands on the shaft leg (192,31,594 -> 202,56,452), which is the level's third largest
    loss and which THIS policy solved (150.5 s in the 2026-09-19 Violent budget, 15.4 s now). There, airborne
    decisions earn MORE than grounded ones -- the climb is made in the air -- so a weight at or above the
    crossover would invert that ordering and push the policy off a climb it has learned. The two pit legs must
    still flip negative, which is the whole point.
    """
    cfg = speed_rewards()
    crossover = SHAFT_OOB - SHAFT_GROUND
    assert abs(crossover - 0.0455) < 0.0005, crossover
    assert cfg.oob < crossover, "the shaft's airborne stream must still out-earn its grounded one"
    assert SHAFT_OOB - cfg.oob > SHAFT_GROUND
    assert (crossover - cfg.oob) / crossover > 0.20, "and with better than a 20% margin"
    for airborne, grounded in ((PIT_EDGE_OOB, PIT_EDGE_GROUND), (PIT_OOB, PIT_GROUND)):
        assert airborne - cfg.oob < 0.0, "the pit legs' airborne stream must go net negative"
        assert airborne < grounded, "... and it was already the worse of the two"


def test_the_charge_lands_on_the_pit_and_not_on_the_fighting():
    """Targeting, as measured: 84.7% of all oob decisions are the two pit legs, 95.6% with the shaft."""
    assert SHARE_PIT_EDGE + SHARE_PIT > 0.84
    assert SHARE_PIT_EDGE + SHARE_PIT + SHARE_SHAFT > 0.95
    # ... which leaves under 5% for everything else, including both arena legs (8 decisions of 16,868) and
    # the ~31 s of arena fighting the time budget called close to irreducible.
    assert 1.0 - (SHARE_PIT_EDGE + SHARE_PIT + SHARE_SHAFT) < 0.05
    # And it predicts the clock better than the channel that was already priced: deaths alone are R2 0.132
    # over 1,680 live completions.
    assert OLS_R2 > 0.5 and OLS_SLOPE > 0.0


def test_the_episode_scale_of_the_term():
    """What it is worth per episode, so nobody has to guess whether it is a rounding error or a rewrite.

    Live `mean_100` at 46.2M steps: oob_frac 0.1259 over 2,035.5 decisions = 256 charged decisions, against a
    mean episode reward of 418.6. At the median RECORDED completion (167) it is 5.8.
    """
    cfg = speed_rewards()
    live = 0.1259 * 2035.5 * cfg.oob
    assert 8.0 < live < 10.0, live
    assert live / 418.64 < 0.03, "a few percent of the episode return, not a rewrite of it"
    assert OOB_MEDIAN * cfg.oob < 6.0
    # The worst completion ever recorded still pays less than a third of one death.
    assert OOB_MAX * cfg.oob < 31.5


# ---------------------------------------------------------------------------------------------------------
# THE FARM TABLE. Finishing must beat not finishing, faster must beat slower, and living must beat dying.
# ---------------------------------------------------------------------------------------------------------
def test_dying_never_becomes_cheaper_than_being_rescued():
    """The one new failure mode a fall charge could buy: dive to die rather than take the teleport.

    A death costs the speed stage's own `death` weight plus the lethal hit, and buys a median 6.77 s rewind
    and a 67.7 m setback on top. One rescue costs DECISIONS_PER_RESCUE charges.
    """
    cfg = speed_rewards()
    per_rescue = DECISIONS_PER_RESCUE * cfg.oob
    per_death = cfg.death + cfg.damage_taken * 100.0
    assert per_death > per_rescue * 12.0, (per_death, per_rescue)
    inversion = per_death / DECISIONS_PER_RESCUE
    assert inversion > 0.5 and cfg.oob < inversion / 10.0, inversion
    # ... and even the LONGEST spell ever recorded (89 decisions) is far cheaper than one death.
    assert SPELL_MAX * cfg.oob < per_death / 3.0


def test_refusing_to_cross_the_pit_can_never_pay():
    """The other new failure mode: stop at rung 6 and earn 0 from the term forever.

    Counted pessimistically -- only the six remaining gate milestones and the completion bonus AT ITS FLOOR,
    none of the two arenas, the door unlocks, the checkpoints or the ~400 m of gate_approach also given up.
    """
    cfg = speed_rewards()
    forfeit = cfg.gate * RUNGS_AFTER_THE_PIT + SPEED_BONUS_MIN * cfg.level_complete
    assert forfeit > 100.0
    assert forfeit > OOB_MAX * cfg.oob * 3.0, "even the worst crossing ever recorded is worth taking"
    assert forfeit > OOB_MEDIAN * cfg.oob * 15.0, "and the typical one by an order of magnitude"
    # Standing still is not an escape either: an episode that stops progressing is truncated by `stuck_seconds`
    # and an oob decision pays no novelty, no approach and no path, so a fall FEEDS that clock.
    assert EnvConfig.from_dict(speed_env()).stuck_seconds > 0


def test_finishing_still_beats_not_finishing_by_more_than_it_did():
    """The charge is bigger on the runs that do NOT finish, so the gap widens rather than narrows.

    rollout_24 (a non-completion that ended in the pit corridor) paid 2,736 oob decisions; the most any
    completion paid was 896, and the median completion paid 167.
    """
    cfg = speed_rewards()
    worst_completion_charge = OOB_MAX * cfg.oob
    non_completion_charge = 2736 * cfg.oob
    assert non_completion_charge > worst_completion_charge * 3.0
    # The completion bonus alone is of the same order as the worst crossing ever recorded (25 at the floor
    # against 31.4), which is why the ladder is counted with it: the six rungs past the pit are 90 more.
    floor = SPEED_BONUS_MIN * cfg.level_complete
    assert bonus(cfg, 10_000.0) > floor, "the floor is approached, never reached"
    assert floor + cfg.gate * RUNGS_AFTER_THE_PIT > worst_completion_charge * 3.0
    assert bonus(cfg, OFFICIAL_MEDIAN) > worst_completion_charge * 2.5, "and a typical completion pays 89.8"


def test_faster_still_beats_slower_at_the_same_falls():
    """`oob` is monotone non-decreasing in dead time, so it can only reinforce the clock's own ordering."""
    cfg = speed_rewards()
    prev = None
    for seconds in (66.66, 82.48, 100.0, 111.43, 150.0, 202.79, 400.0):
        value = bonus(cfg, seconds) - cfg.time * seconds * DECISIONS_PER_S
        if prev is not None:
            assert value < prev, seconds
        prev = value
    for charged in range(0, 900, 100):
        assert step(cfg, oob=1).total * charged <= 0.0


def test_a_fall_buys_nothing_back_through_any_other_channel():
    """No new loop: the channels a fall could have paid are exactly the ones that are silent while airborne.

    `novelty` is 0 with no ground under the player (the 2026-09-16 archive fix) and `gate_approach` pays only
    a NEW best closeness to the current rung, which falling away from it never sets. Measured: of 498 spells,
    the ones that bank anything bank it on the way back UP, not on the way down.
    """
    cfg = speed_rewards()
    assert step(cfg, oob=1, novelty=0.0, gate_approach=0.0).total < 0.0
    # A spell that does bank approach still nets out negative only if the approach is small; the term must
    # never be able to cancel a genuine milestone, which is the direction that matters.
    assert step(cfg, oob=1, gates=1).total > 0.0, "reaching a rung while airborne must still pay"
    assert cfg.gate > cfg.oob * SPELL_MAX


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
