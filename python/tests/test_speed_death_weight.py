"""A SPEED stage's own `death` weight (2026-09-20): the rule, the constant, and the farm bounds.

No game needed:  python tests/test_speed_death_weight.py   (or pytest)

WHY THE WEIGHT MOVES, in one paragraph. On Level 0-2's speed stage the clock IS deaths: over 2,067 live fresh
completions in runs/spec_0-2_speed/episodes.jsonl, `seconds = 121.33 + 22.32 * deaths` (r +0.86), and a
completed run averages 4.75 of them. One death objectively costs 14.27 of reward -- 6.99 of `time` over the
349.6 extra decisions it adds, plus 7.28 of `level_complete` given up by the slower clock. The policy feels
about -6.5 of that: `death` 5.0 and `damage_taken` 1.0 land at the decision, the time stream discounts to 5.02
at gamma 0.998, the completion bonus arrives ~1,484 decisions later at 5.1% of face value, and a respawn
RE-PAYS +6.17 of combat reward because the game's kill counter rolls back to the checkpoint value and the
resurrected arena is killed again (measured over the 126 deaths in runs/probe_0-2_speed: 133 rollbacks, 2,434
paid kill events against 1,126 final kills). `death: 12.0` makes the felt cost match the objective one.

WHAT THIS FILE PINS. Every constant and every rule above, and -- because rewards the policy could farm have
burned this project before -- the farm table: never-moving, dying at once, camping in front of the hazard and
suiciding for a checkpoint shortcut all earn less than any completion; faster always beats slower; and one
more death is always worth less than one fewer, INCLUDING the re-pay. The ledger below is built from the
plan's own generated config, so it re-derives at whatever weight the plan carries and states which bounds hold
at every weight and which ones are bought by this one.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_driver  # noqa: E402
from ultrakill_ai.env import EnvConfig  # noqa: E402
from ultrakill_ai.rewards import RewardConfig, completion_bonus  # noqa: E402

PLAN = ROOT / "configs" / "specialists.yaml"

# ---------------------------------------------------------------------------------------------------------
# MEASURED CONSTANTS. Every one of these is a number read off the live log or the recordings, not a guess, and
# each is quoted where it was measured so a later reader can re-take it. They are pinned because the sizing
# argument is only as good as they are: if a re-measurement moves one, the weight has to be re-derived.
# ---------------------------------------------------------------------------------------------------------
# runs/spec_0-2_speed/episodes.jsonl, 2,067 timed fresh completions, 18.78M-27.31M steps:
SECONDS_INTERCEPT = 121.33     # official seconds of a zero-death completion (OLS)
SECONDS_PER_DEATH = 22.32      # ... and what each death adds (r +0.864)
DECISIONS_PER_DEATH = 349.6    # `length = 1959.0 + 349.6 * deaths`
DECISIONS_PER_SECOND = 15.63   # median length / official seconds
MEAN_DEATHS = 4.75             # mean deaths per completed fresh run
DEATHS_TO_EXIT = 1484          # decisions from a median death to the completion that pays the bonus
# runs/probe_0-2_speed, 24 episodes / 126 deaths, summed per-step `reward_parts`:
REPAY_PER_DEATH = 6.17         # kill 4.18 + damage_dealt 1.99, r +0.854 -- the kill counter rolls back
REPAY_DECISIONS = 300          # ... measured over the 300 decisions after each restart (kill 4.78, dmg 2.83)
# The two non-completing outcomes an episode can actually reach, from the same live log (means, at death 5.0):
STUCK_REWARD, STUCK_DEATHS = 200.0, 3.26        # n=140, end_reason "stuck"
LOOP_REWARD, LOOP_DEATHS = 295.4, 24.33         # n=3, end_reason "max_steps": the death loop, live
STUCK_DECISIONS = 675          # the stuck rule: 45 game s * 30 fps / frameskip 2 -- the never-move ceiling
GAMMA = 0.998


def plan():
    return campaign_driver.load_plan(PLAN)


def speed_env(level: str = "Level 0-2") -> dict:
    return campaign_driver.stage_config(plan(), level, kind=campaign_driver.SPEED)["env"]


def speed_rewards() -> RewardConfig:
    return EnvConfig.from_dict(speed_env()).rewards


def discounted(per_decision: float, n: int) -> float:
    """The geometric sum of a per-decision stream, which is how the policy actually receives `time`."""
    return per_decision * (1.0 - GAMMA ** n) / (1.0 - GAMMA)


def spread_discount(n: int) -> float:
    """What a lump spread EVENLY over the next `n` decisions is worth at the decision before them.

    The combat re-pay is not a payment at decision `n`; it arrives across the whole post-respawn stretch as
    the resurrected arena is fought again, so it discounts like a stream (0.75 over 300), not like a bullet
    at the end (0.55). Using the harsher figure would under-state the re-pay and over-size the weight.
    """
    return (1.0 - GAMMA ** n) / (n * (1.0 - GAMMA))


def bonus(cfg: RewardConfig, seconds: float, target: float = 120.0) -> float:
    return completion_bonus(cfg, target, seconds)


def objective_cost(cfg: RewardConfig, deaths_before: float = 0.0, target: float = 120.0) -> float:
    """What ONE MORE death costs in the reward's own currency: lost clock plus lost completion bonus.

    The bonus is hyperbolic in time, so this shrinks with every death already taken: 18.52 for the first,
    11.07 for the fifth. Pricing the first death would need a weight near 17; see the sizing test.
    """
    t0 = SECONDS_INTERCEPT + SECONDS_PER_DEATH * deaths_before
    t1 = t0 + SECONDS_PER_DEATH
    return (bonus(cfg, t0, target) - bonus(cfg, t1, target)) + cfg.time * DECISIONS_PER_DEATH


def objective_cost_average(cfg: RewardConfig, target: float = 120.0) -> float:
    """... and what a death costs ON AVERAGE over today's operating range, 0 -> MEAN_DEATHS.

    This is the number the weight is sized against, because it is the death the policy actually has.
    """
    t0 = SECONDS_INTERCEPT
    t1 = SECONDS_INTERCEPT + SECONDS_PER_DEATH * MEAN_DEATHS
    return (bonus(cfg, t0, target) - bonus(cfg, t1, target)) / MEAN_DEATHS + cfg.time * DECISIONS_PER_DEATH


def felt_cost(cfg: RewardConfig, target: float = 120.0) -> float:
    """What the policy receives AT the death decision, discounted -- the number the weight is sized against.

    `death` and the lethal hit's `damage_taken` land immediately; the re-traversal's `time` is a stream over
    the next DECISIONS_PER_DEATH decisions; the combat re-pay lands across the ~300 decisions after the
    respawn; the completion bonus is DEATHS_TO_EXIT decisions away and is worth 5.1% of its face value there.
    """
    t0 = SECONDS_INTERCEPT + SECONDS_PER_DEATH * MEAN_DEATHS
    bonus_loss = (bonus(cfg, SECONDS_INTERCEPT, target) - bonus(cfg, t0, target)) / MEAN_DEATHS
    return -(cfg.death
             + cfg.damage_taken * 100.0
             + discounted(cfg.time, int(DECISIONS_PER_DEATH))
             + bonus_loss * GAMMA ** DEATHS_TO_EXIT
             - REPAY_PER_DEATH * spread_discount(REPAY_DECISIONS))


def completion_reward(cfg: RewardConfig, deaths: float, progress: float = 400.0,
                      target: float = 120.0) -> float:
    """A whole completed episode's return at `deaths`, from the weights themselves.

    `progress` is the route/arena income a completion earns whatever happens (door_unlock 137 + gate 116 +
    arena_clear 46 + kills on the live log); it is a constant here because it does not vary with deaths --
    measured `gates_reached = 8.00 + 0.0000 * deaths` and `checkpoints_level = 2.00 + 0.0001 * deaths`, i.e.
    a respawn re-pays NO milestone (`mark_paid` / `new_level_load(keep_paid=True)`).
    """
    seconds = SECONDS_INTERCEPT + SECONDS_PER_DEATH * deaths
    decisions = seconds * DECISIONS_PER_SECOND
    return (progress
            + bonus(cfg, seconds, target)
            + REPAY_PER_DEATH * deaths                      # the re-pay, counted IN FAVOUR of dying
            - cfg.death * deaths
            - cfg.damage_taken * 100.0 * deaths
            - cfg.time * decisions)


# ---------------------------------------------------------------------------------------------------------
# THE RULE AND THE CONSTANT
# ---------------------------------------------------------------------------------------------------------
def test_the_speed_stage_prices_a_death_at_twelve():
    p = plan()
    assert p.speed_rewards["death"] == 12.0, "the plan names the weight"
    assert speed_rewards().death == 12.0
    assert EnvConfig.from_dict(campaign_driver.stage_config(p, "Level 0-2")["env"]).rewards.death == 5.0, \
        "a COMPLETE stage is untouched"


def test_the_override_reaches_every_speed_stage_and_no_complete_stage():
    """`speed:` is per-KIND, so 0-1's and 0-3's speed stages carry it too. Stated, not discovered later."""
    p = plan()
    for level in ("Level 0-1", "Level 0-2", "Level 0-3"):
        speed = campaign_driver.stage_config(p, level, kind=campaign_driver.SPEED)["env"]
        assert speed["rewards"]["death"] == 12.0, level
    for level in ("Level 0-1", "Level 0-4", "Level 5-2"):
        assert campaign_driver.stage_config(p, level)["env"]["rewards"]["death"] == 5.0, level


def test_generating_a_speed_config_does_not_contaminate_the_next_complete_config():
    """THE ALIASING GUARD. `env` in stage_config is a SHALLOW copy, so `env["rewards"]` IS the plan's own
    dict. An in-place `.update()` would leak the speed weight into every complete stage generated afterwards
    -- and comparing the two generated configs would NOT catch it, because both would be the one mutated
    object. So: generate the speed config FIRST, then the complete one, from the same Plan.
    """
    p = plan()
    before = dict(p.env["rewards"])
    campaign_driver.stage_config(p, "Level 0-2", kind=campaign_driver.SPEED)
    after = campaign_driver.stage_config(p, "Level 0-2")["env"]["rewards"]
    assert after["death"] == 5.0, "the speed stage mutated the plan's shared rewards dict"
    assert p.env["rewards"] == before, "the plan itself was modified"
    assert dict(after) == before


def test_load_plan_refuses_a_misspelt_speed_weight(tmp_path=None):
    """A weight EnvConfig.from_dict does not know would silently train at the shared value."""
    import yaml
    data = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    data["speed"]["rewards"] = {"deaths": 12.0}          # the plural: not a RewardConfig field
    out = Path(tmp_path or ROOT / "runs") / "_bad_plan.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    try:
        campaign_driver.load_plan(out)
    except ValueError as exc:
        assert "deaths" in str(exc)
    else:
        raise AssertionError("load_plan accepted an unknown speed reward weight")
    finally:
        out.unlink(missing_ok=True)


def test_a_plan_without_the_block_generates_exactly_what_it_did_before():
    """Every older plan, and every test that loads one, must mean what it meant."""
    p = plan()
    bare = dataclasses.replace(p, speed_rewards={})
    speed = campaign_driver.stage_config(bare, "Level 0-2", kind=campaign_driver.SPEED)["env"]
    complete = campaign_driver.stage_config(bare, "Level 0-2")["env"]
    assert speed["rewards"] == complete["rewards"] == p.env["rewards"]


# ---------------------------------------------------------------------------------------------------------
# THE SIZING
# ---------------------------------------------------------------------------------------------------------
def test_twelve_is_what_makes_the_felt_cost_of_a_death_its_objective_cost():
    cfg = speed_rewards()
    objective = objective_cost_average(cfg)                     # the AVERAGE death of today's run
    marginal = objective_cost(cfg, 0.0)                         # ... and the FIRST one
    assert 13.5 < objective < 15.0, objective                   # 14.28 as derived
    assert 17.5 < marginal < 19.5, marginal                     # 18.52
    # The bonus is hyperbolic, so the marginal death gets cheaper as the run gets slower. 12.0 therefore
    # prices the death the policy HAS and UNDER-prices the one it will have when it is fast: conservative.
    assert objective_cost(cfg, 5.0) < objective < marginal
    assert abs(felt_cost(cfg)) < objective * 1.10, "12.0 must not OVER-price the average death"
    assert abs(felt_cost(cfg)) > objective * 0.85, "... and must come within 15% of it"
    # The old weight felt less than HALF the truth -- which is the defect, stated as a number.
    assert abs(felt_cost(dataclasses.replace(cfg, death=5.0))) < objective * 0.55
    # And the conservative direction is kept: pricing the MARGINAL death would need ~17.
    assert cfg.death < marginal - 1.0


def test_the_completion_bonus_channel_is_the_one_gamma_hides():
    """The reason the weight moves at all: the terminal channel is delivered at 5.1% of face value."""
    cfg = speed_rewards()
    assert 0.04 < GAMMA ** DEATHS_TO_EXIT < 0.07
    # ... while the `time` stream, the other half of the objective cost, is 70%+ visible over 349 decisions.
    assert discounted(cfg.time, int(DECISIONS_PER_DEATH)) > 0.70 * cfg.time * DECISIONS_PER_DEATH


def test_a_respawn_re_pays_combat_reward_which_the_weight_has_to_cover():
    """The measured +6.17: without it the sizing is wrong by nearly half a death."""
    cfg = speed_rewards()
    # At the OLD weight the immediate channels (death + the lethal hit) were SMALLER than the re-pay, so a
    # death paid for itself on everything the policy feels promptly. That is the defect, as an inequality.
    assert 5.0 + 0.01 * 100.0 < REPAY_PER_DEATH, "a death used to pay for itself on the immediate channels"
    assert cfg.death + cfg.damage_taken * 100.0 > REPAY_PER_DEATH * 1.5, "and now it does not"
    # Ignoring the re-pay would over-state the pain the old weight delivered by a third, and would have sized
    # the new one too low -- so it is carried in the ledger, in favour of dying, everywhere in this file.
    naive = -(cfg.death + cfg.damage_taken * 100.0 + discounted(cfg.time, int(DECISIONS_PER_DEATH)))
    assert abs(naive) > abs(felt_cost(cfg)) * 1.15, "the re-pay is a material part of the ledger"


# ---------------------------------------------------------------------------------------------------------
# THE FARM TABLE. Finishing must always beat not finishing; faster must always beat slower; and dying must
# never pay more than living.
# ---------------------------------------------------------------------------------------------------------
def test_never_moving_is_the_worst_outcome_there_is():
    """The stuck rule truncates a motionless episode at 675 decisions of pure `time` and nothing else."""
    cfg = speed_rewards()
    never_move = -cfg.time * STUCK_DECISIONS
    assert never_move < -13.0
    for deaths in range(0, 13):
        assert completion_reward(cfg, deaths) > never_move + 250.0, deaths


def test_dying_at_once_cannot_end_an_episode_and_buys_nothing():
    """env.py respawns unconditionally in campaign mode (`end_episode_on_death` is not consulted there), and
    0 of 2,392 live episodes ended with end_reason "death". "Die to end a bad episode" is not available; the
    policy just pays the weight and carries on, so the cost is monotone in the weight.
    """
    cfg = speed_rewards()
    cost_now = cfg.death + cfg.damage_taken * 100.0 - REPAY_PER_DEATH
    cost_before = 5.0 + 1.0 - REPAY_PER_DEATH
    assert cost_now > cost_before and cost_now > 0.0, "the immediate ledger of a death is now negative"


def test_camping_in_front_of_the_hazard_is_strictly_negative():
    """The kill volumes are TRIGGERS: every observation ray passes QueryTriggerInteraction.Ignore, there is no
    hazard channel in the 479 floats and no frame stack, so p(death | crossing) = 0.115 is memoryless and
    waiting cannot lower it. Waiting therefore buys exactly zero and costs `time` per decision, at every
    weight -- this is the bound the profile asked for before, not after.
    """
    cfg = speed_rewards()
    per_second = cfg.time * DECISIONS_PER_SECOND
    assert per_second > 0.3
    for waited in (1.0, 5.0, 20.0):
        # Same deaths (waiting does not change them), more seconds: strictly worse.
        base = completion_reward(cfg, MEAN_DEATHS)
        camped = base - per_second * waited - (
            bonus(cfg, SECONDS_INTERCEPT + SECONDS_PER_DEATH * MEAN_DEATHS)
            - bonus(cfg, SECONDS_INTERCEPT + SECONDS_PER_DEATH * MEAN_DEATHS + waited))
        assert camped < base, waited


def test_suiciding_for_a_checkpoint_shortcut_never_pays():
    """A respawn never advances the player and the official clock runs through it, so the bonus can only fall;
    and no milestone is re-paid (`gates_reached` slope 0.0000 over 2,067 completions). The only thing a death
    gives back is the combat re-pay, and it is smaller than the weight.
    """
    cfg = speed_rewards()
    for deaths in range(0, 12):
        assert completion_reward(cfg, deaths + 1) < completion_reward(cfg, deaths), deaths
    # ... and the per-death slope is negative even if `death` were ZERO, which is the structural invariant:
    zero = dataclasses.replace(cfg, death=0.0)
    assert completion_reward(zero, 5) < completion_reward(zero, 4)


def test_faster_always_beats_slower_at_the_same_deaths():
    cfg = speed_rewards()
    prev = None
    for seconds in (86.74, 100.0, 120.0, 121.33, 150.0, 203.9, 300.0, 600.0):
        value = bonus(cfg, seconds) - cfg.time * seconds * DECISIONS_PER_SECOND
        if prev is not None:
            assert value < prev, seconds
        prev = value
    # The bonus itself is strictly decreasing with no flat region -- the floor is approached, never reached.
    assert bonus(cfg, 10_000.0) > 0.25 * cfg.level_complete


def test_the_live_death_loop_now_earns_less_than_giving_up():
    """THE ORDERING THIS CHANGE BUYS, on the two outcomes the live log actually contains.

    Three episodes hit `max_steps` with a mean of 24.33 deaths and, at `death: 5.0`, a mean return of 295.4 --
    MORE than the 140 "stuck" episodes' 200.0. The combat re-pay funds the loop and only the `time` bleed kept
    it negative. Re-scoring both at the configured weight (which is exact: `death` is a linear per-event term)
    inverts that, and the crossover is at about 9.5.
    """
    cfg = speed_rewards()

    def rescored(reward, deaths):
        return reward - (cfg.death - 5.0) * deaths

    assert LOOP_REWARD > STUCK_REWARD, "at 5.0 the death loop out-earned giving up -- this is the defect"
    assert rescored(LOOP_REWARD, LOOP_DEATHS) < rescored(STUCK_REWARD, STUCK_DEATHS), \
        "at the configured weight the death loop must be the worse outcome"
    crossover = 5.0 + (LOOP_REWARD - STUCK_REWARD) / (LOOP_DEATHS - STUCK_DEATHS)
    assert 9.0 < crossover < 10.0 and cfg.death > crossover


def test_finishing_beats_every_way_of_not_finishing():
    cfg = speed_rewards()

    def rescored(reward, deaths):
        return reward - (cfg.death - 5.0) * deaths

    worst_completion = completion_reward(cfg, 12)
    for name, reward, deaths in (("stuck", STUCK_REWARD, STUCK_DEATHS),
                                 ("max_steps", LOOP_REWARD, LOOP_DEATHS)):
        assert worst_completion > rescored(reward, deaths) + 50.0, name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
