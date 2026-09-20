"""configs/specialists.yaml: the per-level specialist plan, pinned against the config it descends from.

No game needed:  python tests/test_specialists_config.py   (or pytest)

The whole point of a specialist ladder is that ONLY the mixture changes. Each stage is the existing
single-level campaign env, initialised from the shared run's policy, so every reward weight, every env setting
and every hyperparameter has to be character for character `configs/campaign_gates_full.yaml`'s -- and the
differences that are allowed are named here, one by one, exactly as `test_campaign_config.py` does for the
chain of campaign configs. Anything else is a change nobody decided.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_driver  # noqa: E402
import train  # noqa: E402
from ultrakill_ai.campaign import CAMPAIGN_LEVELS, CAMPAIGN_LEVELS_SHIPPED  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402

PLAN = ROOT / "configs" / "specialists.yaml"
GATES_FULL = ROOT / "configs" / "campaign_gates_full.yaml"
# The keys a single-level stage must not carry, and the EnvConfig fields their absence is allowed to change.
# `unlock_rate`, `unlock_window` and `level_weight_floor` are dropped too, but the full config sets them to
# EnvConfig's own defaults, so removing them changes nothing and they are not in this set.
CURRICULUM_FIELDS = {"levels", "unlock_after_fresh_episodes", "curriculum_weighting"}


def plan():
    return campaign_driver.load_plan(PLAN)


def test_the_order_is_the_full_configs_levels_in_the_same_order():
    env_dict, _ = train.load_config(str(GATES_FULL))
    assert plan().order == env_dict["levels"], "the ladder is the routed campaign, unchanged and in mission order"
    assert len(plan().order) == 30 and len(set(plan().order)) == 30
    assert all(level in CAMPAIGN_LEVELS_SHIPPED for level in plan().order)
    order = [CAMPAIGN_LEVELS.index(level) for level in plan().order]
    assert order == sorted(order), "mission order, so a specialist is initialised from the level before it"
    assert plan().order[0] == "Level 0-1"


def test_the_stage_template_is_the_full_config_minus_the_multi_level_keys():
    """Every env setting that is not a curriculum key is the live config's, and every reward weight is."""
    full_env, full_train = train.load_config(str(GATES_FULL))
    p = plan()

    # The raw YAML first: which keys were dropped, and that nothing was added.
    assert set(full_env) - set(p.env) == {"levels", "unlock_rate", "unlock_window",
                                          "unlock_after_fresh_episodes", "level_weight_floor",
                                          "curriculum_weighting"}
    assert set(p.env) - set(full_env) == {"level"}, "a stage names one level instead of a list"
    assert {k for k in set(p.env) & set(full_env) if p.env[k] != full_env[k]} == set()

    # And the parsed effect: only the curriculum fields may differ.
    full, cfg = EnvConfig.from_dict(full_env), EnvConfig.from_dict(dict(p.env))
    differing = {f.name for f in dataclasses.fields(EnvConfig)
                 if getattr(cfg, f.name) != getattr(full, f.name)}
    assert differing == CURRICULUM_FIELDS, f"env settings changed besides the curriculum: {sorted(differing)}"
    assert cfg.levels == [] and cfg.curriculum_path == "", "a stage is a single-level run, by construction"
    assert cfg.rewards == full.rewards, "no reward weight may change: the same policy continues"
    assert (cfg.difficulty, cfg.unlock_all_gear, cfg.max_steps, cfg.pitch_limit_deg) == (3, True, 12000, 45.0)
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (30, 2, False, False)
    assert cfg.gate_patience_mode == "collapsed" and cfg.prefer_route_when_collapsed is True


def test_the_train_template_is_the_full_configs_optimiser():
    full_env, full_train = train.load_config(str(GATES_FULL))
    p = plan()
    assert set(full_train) - set(p.train) == {"run_name", "timesteps"}, "both are per stage"
    assert set(p.train) - set(full_train) == set()
    assert {k for k in p.train if p.train[k] != full_train[k]} == set(), \
        "the optimiser, the policy shape, num_envs and the entropy floor are the live run's"
    assert p.train["num_envs"] == 12 and p.train["hyperparams"]["ent_coef"] == 0.004
    assert (p.train["ent_floor"], p.train["ent_coef_max"]) == (6.5, 0.02)


def test_every_plan_setting_is_a_real_field():
    """EnvConfig.from_dict drops keys it does not know, so a misspelt setting would silently use its default."""
    p = plan()
    names = {f.name for f in dataclasses.fields(EnvConfig)}
    assert not sorted(set(p.env) - names), sorted(set(p.env) - names)
    reward_names = {f.name for f in dataclasses.fields(RewardConfig)}
    assert not sorted(set(p.env["rewards"]) - reward_names)
    # ... including a SPEED stage's own weights, which are merged over those and go down the same from_dict.
    assert not sorted(set(p.speed_rewards) - reward_names)
    # And the stage knobs: load_plan refuses one it does not know rather than defaulting it.
    text = PLAN.read_text(encoding="utf-8")
    assert "target_rate" in text and "settle_steps" in text and "max_steps_per_stage" in text


def test_the_stage_rule_numbers_are_the_ones_the_driver_documents():
    rule = plan().rule
    assert (rule.target_rate, rule.min_fresh_window) == (0.5, 30)
    assert (rule.settle_steps, rule.max_steps_per_stage) == (300_000, 6_000_000)
    assert (rule.count, rule.monitor) == (12, 1)
    assert rule.max_rounds == 0, "no round cap: a HELD exit idles twelve games with nobody watching"
    assert rule.seed_explore_from == "campaign_gates", "the shared run's archives seed each stage"


def test_the_three_speed_stages_sit_before_any_0_4_stage():
    """The lead's 2026-09-18 instruction, pinned: nothing promotes to 0-4 until 0-1..0-3 have better times."""
    p = plan()
    keys = [s.key for s in p.stages]
    speed_at = [i for i, key in enumerate(keys) if key[1] == "speed"]
    later = [i for i, key in enumerate(keys) if key[0] not in ("Level 0-1", "Level 0-2", "Level 0-3")]
    assert max(speed_at) < min(later), "every speed stage runs before the first level past 0-3"
    assert len(keys) == 33 and len(set(keys)) == 33, "30 complete stages plus the three speed ones"


def test_the_plan_is_depth_first_by_level():
    """2026-09-19, the user: "shouldnt we just work on 0-1 untell its finished before working on the other
    levels". `hold_order: sequential` walks the `order:` list in order, so the LIST is the instruction: each
    level's speed stage sits directly after its own complete stage, and the machine finishes one level
    (complete, then speed) before the next level's first stage can start."""
    p = plan()
    keys = [s.key for s in p.stages]
    assert p.hold_order == campaign_driver.SEQUENTIAL, "depth-first, not the 2026-09-18 round robin"
    assert keys[:6] == [("Level 0-1", "complete"), ("Level 0-1", "speed"),
                        ("Level 0-2", "complete"), ("Level 0-2", "speed"),
                        ("Level 0-3", "complete"), ("Level 0-3", "speed")]
    # Level-major: every stage of a level is contiguous, so "the first not-done stage in plan order" is
    # always a stage of the earliest unfinished level.
    firsts = {}
    for i, (level, _) in enumerate(keys):
        firsts.setdefault(level, i)
    assert [level for level, _ in keys] == sorted((level for level, _ in keys), key=lambda l: firsts[l])
    # ... and the distinct levels full_run.py plays are untouched by the reordering.
    assert p.order[:4] == ["Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4"]


def test_the_speed_rule_is_the_stage_rule_with_the_clock_added():
    p = plan()
    speed = p.rule_for(campaign_driver.SPEED)
    assert (speed.target_rate, speed.max_steps_per_stage) == (0.4, 8_000_000)
    assert (speed.min_fresh_window, speed.settle_steps) == (30, 300_000), "inherited from `stage:`"
    assert (speed.count, speed.monitor, speed.seed_explore_from) == (12, 1, "campaign_gates")
    assert p.targets == {}, "no hand-picked seconds: every level's target is its own S-rank time"
    # 2026-09-18, the lead: the gate is the MEDIAN of the last 50 fresh completions, not the run's lifetime
    # best, and against a median the level's own S-rank time is already demanding (spec_0-1's median is 482 s
    # against an S of 120 s). Scale 1.0 is "the typical run S-ranks the clock".
    assert p.target_scale == 1.0, "the level's own S-rank time, unscaled (§8b)"
    # And NO round cap, under either kind: the driver's only alternative is a "HELD" exit, and nobody watches
    # this run, so that exit leaves twelve games idle for hours. Training the held stages beats idling.
    assert (speed.max_rounds, p.rule.max_rounds) == (0, 0), "unbounded rounds: never idle the machine"
    assert p.rule_for(campaign_driver.COMPLETE) == p.rule, "a complete stage's rule is untouched"


def test_the_hold_line_sits_in_front_of_every_stage_past_0_3():
    """§8a, and the reason the plan file has the key at all: the ladder may not reach 0-4 on slow policies."""
    p = plan()
    assert p.hold_before == "Level 0-4"
    hold = p.hold_index()
    assert hold == 6, "the six 0-1..0-3 stages are in front of it and nothing else is"
    assert {s.level for s in p.stages[:hold]} == {"Level 0-1", "Level 0-2", "Level 0-3"}
    assert all(s.level not in ("Level 0-1", "Level 0-2", "Level 0-3") for s in p.stages[hold:])


def test_a_speed_stage_only_adds_the_bonus_switch_and_the_death_weight():
    """No observation or action change, so the level's own specialist loads into it unchanged.

    ONE reward weight moves, and only for this kind: `death` 5.0 -> 12.0 (2026-09-20). A reward weight is not
    a policy parameter -- it is a scalar read into `RewardConfig` at env construction -- so every existing
    checkpoint still loads. The derivation is in configs/specialists.yaml beside the key and in
    docs/project-log.md; the farm bounds are pinned in tests/test_speed_death_weight.py.
    """
    p = plan()
    complete = campaign_driver.stage_config(p, "Level 0-2")["env"]
    speed = campaign_driver.stage_config(p, "Level 0-2", kind=campaign_driver.SPEED)["env"]
    added = {"speed_bonus", "speed_target_scale"}
    assert set(speed) - set(complete) == added and speed["speed_bonus"] is True
    # ... plus exactly ONE changed value: every episode is a fresh level load (2026-09-18 review). The bonus is
    # scaled only on a fresh-start completion, and the stage is scored only on fresh-start episodes, so at the
    # complete stage's 0.2 a checkpoint respawn would pay the full unscaled weight for the outcome the stage
    # does not measure -- an unobservable 4x split in the terminal reward.
    changed = {k for k in set(speed) & set(complete) if speed[k] != complete[k]}
    assert changed == {"fresh_start_prob", "rewards"}
    assert speed["fresh_start_prob"] == 1.0 and complete["fresh_start_prob"] == 0.2
    # ... and inside `rewards`, exactly one weight, by the plan's `speed.rewards:` block and nothing else.
    assert set(speed["rewards"]) == set(complete["rewards"]), "no weight appears or disappears"
    moved = {k for k in speed["rewards"] if speed["rewards"][k] != complete["rewards"][k]}
    assert moved == {"death"} == set(p.speed_rewards)
    assert (speed["rewards"]["death"], complete["rewards"]["death"]) == (12.0, 5.0)
    assert {k: v for k, v in speed.items() if k not in added | changed} == \
        {k: v for k, v in complete.items() if k not in changed}
    cfg = EnvConfig.from_dict(speed)
    assert cfg.speed_bonus is True and cfg.speed_target_seconds == 0.0, "0 = read the level's own S-rank time"
    assert cfg.fresh_start_prob == 1.0
    assert cfg.speed_target_scale == 1.0, "the plan's target_scale (§8b), applied in the env and nowhere else"
    base = EnvConfig.from_dict(complete).rewards
    assert dataclasses.replace(cfg.rewards, death=base.death) == base, \
        "every reward weight but `death` is the complete stage's"
    env = UltrakillEnv(cfg)
    try:
        assert env.observation_space.shape == (479,), "the same policy shape: an existing checkpoint loads"
    finally:
        env.close()


def test_a_generated_stage_config_builds_a_479_input_single_level_env():
    p = plan()
    generated = campaign_driver.stage_config(p, "Level 2-3", init_steps=12_000_000)
    assert not (set(generated["env"]) & set(campaign_driver.MULTI_LEVEL_KEYS))
    assert generated["env"]["level"] == "Level 2-3"
    assert generated["train"]["run_name"] == "spec_2-3"
    # `timesteps` is the run TOTAL in train.py, so a stage's budget is measured from where it starts.
    assert generated["train"]["timesteps"] == 12_000_000 + p.rule.max_steps_per_stage + campaign_driver.TIMESTEPS_SLACK

    cfg = EnvConfig.from_dict(generated["env"])
    env = UltrakillEnv(cfg)
    try:
        assert env.observation_space.shape == (479,)
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]
        assert env.level == "Level 2-3" and env.cfg.levels == []
        assert env._max_steps() == 12000
    finally:
        env.close()


def test_a_specialist_checkpoint_has_the_same_shape_as_the_shared_policy():
    """The first stage resumes from the shared run's own weights, so the spaces must be identical."""
    full_env, _ = train.load_config(str(GATES_FULL))
    shared = UltrakillEnv(EnvConfig.from_dict(full_env))
    stage = UltrakillEnv(EnvConfig.from_dict(campaign_driver.stage_config(plan(), "Level 0-1")["env"]))
    try:
        assert shared.observation_space.shape == stage.observation_space.shape == (479,)
        assert list(shared.action_space.nvec) == list(stage.action_space.nvec)
    finally:
        shared.close()
        stage.close()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
