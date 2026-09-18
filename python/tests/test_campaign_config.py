"""The campaign 0-1 config and train.py's campaign wiring. No game needed:  python tests/test_campaign_config.py  (or pytest)."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import train  # noqa: E402
from ultrakill_ai.campaign import CAMPAIGN_LEVELS, CAMPAIGN_LEVELS_SHIPPED  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.progress import PPO_METRICS  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402

CONFIG = ROOT / "configs" / "campaign_0-1.yaml"
PRELUDE = ROOT / "configs" / "campaign_prelude.yaml"
LEVEL_1_1 = ROOT / "configs" / "campaign_1-1.yaml"
GATES_PRELUDE = ROOT / "configs" / "campaign_gates_prelude.yaml"
GATES_MAIN = ROOT / "configs" / "campaign_gates_main.yaml"
GATES_FULL = ROOT / "configs" / "campaign_gates_full.yaml"
RUN_NAME = "campaign_gates"
MODEL_DIR = Path("models") / RUN_NAME
RUN_DIR = Path("runs") / RUN_NAME


def field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_campaign_env_has_479_inputs_and_the_yaml_settings():
    env_dict, _ = train.load_config(str(CONFIG))
    cfg = EnvConfig.from_dict(env_dict)
    assert (cfg.mode, cfg.level, cfg.difficulty, cfg.unlock_all_gear) == ("campaign", "Level 0-1", 3, True), (cfg.mode, cfg.level, cfg.difficulty, cfg.unlock_all_gear)
    # The spec's speed settings. If in-game check 6 falls back to 60/4 or rendering on, change the yaml and this together.
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (30, 2, False, False)
    assert (cfg.window_width, cfg.window_height, cfg.max_steps) == (368, 207, 9000)
    assert (cfg.fresh_start_prob, cfg.stuck_seconds, cfg.stuck_repeats, cfg.cell_size) == (0.2, 45, 3, 4.0)
    assert cfg.pitch_limit_deg == 45.0  # the campaign run left this off and the camera sat at a mean of -78 degrees
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # train.py fills these per run
    for name, value in env_dict["rewards"].items():
        assert getattr(cfg.rewards, name) == value, name
    assert cfg.rewards.style == 0.0 and cfg.rewards.time == 0.02 and cfg.rewards.level_complete == 100.0
    assert cfg.rewards.punch == 0.01  # the punch button is charged for, so the policy stops flailing
    # The route: gates and their approach shaping, and the door unlock that opens an arena-held gate.
    assert (cfg.rewards.gate, cfg.rewards.gate_approach, cfg.rewards.door_unlock) == (15.0, 0.15, 15.0)
    assert cfg.rewards.path == 0.0  # the NavMesh hint never read "complete"; its slots are the target's now
    assert cfg.rewards.novelty == 0.2  # demoted to a secondary explorer
    # The gate defaults come from EnvConfig, not the yaml, so a change there is visible here.
    assert (cfg.gate_reach_m, cfg.gate_reach_v_m, cfg.gate_min_gain_m) == (8.0, 6.0, 0.5)
    assert (cfg.wedge_seconds, cfg.wedge_creep_mps, cfg.slide_min_hold) == (3.0, 1.5, 0)

    env = UltrakillEnv(cfg)  # builds without a game: nothing connects until the first reset
    try:
        assert env.cfg.layout.campaign
        assert env.observation_space.shape == (479,)
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]  # the look mode is 12th
        assert env.cfg.rewards == cfg.rewards
    finally:
        env.close()


def test_the_0_1_config_is_still_a_single_level_run():
    """The live campaign_gates run must be untouched: no levels list, so nothing in S1 or S4 can reach it."""
    env_dict, _ = train.load_config(str(CONFIG))
    cfg = EnvConfig.from_dict(env_dict)
    assert cfg.levels == [] and cfg.curriculum_path == ""
    assert cfg.rewards.item_pickup == 0.0 and cfg.rewards.item_placed == 0.0
    assert cfg.gate_hops_min_frac == 0.5  # 0-1 is 11/11 gates with hops, so the guard never fires on it
    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert filled.curriculum_path == "", "a single-level run never opens a curriculum file"


def test_the_gates_prelude_config_carries_the_live_run_forward():
    """The config the live run resumes on (2026-09-17): the curriculum, the SAME run name and weights, ent_coef cut.

    Every reward weight and every env setting has to match configs/campaign_0-1.yaml, which is what
    models/campaign_gates/env_config.yaml records for the 4.8M steps already trained. Only three things may
    differ: the levels list (and its three knobs), num_envs and ent_coef.
    """
    live_env, live_train = train.load_config(str(CONFIG))
    env_dict, train_cfg = train.load_config(str(GATES_PRELUDE))
    live, cfg = EnvConfig.from_dict(live_env), EnvConfig.from_dict(env_dict)

    assert cfg.rewards == live.rewards, "no reward weight may change while the same policy continues"
    assert cfg.rewards.item_pickup == 0.0 and cfg.rewards.item_placed == 0.0, "S4 stays dormant"
    curriculum = {"level", "levels", "unlock_rate", "unlock_window", "level_weight_floor"}
    differing = {f.name for f in dataclasses.fields(EnvConfig)
                 if getattr(cfg, f.name) != getattr(live, f.name)}
    assert differing <= curriculum, f"env settings changed besides the curriculum: {sorted(differing - curriculum)}"
    assert cfg.levels == ["Level 0-1", "Level 0-3", "Level 0-4"] and cfg.levels[0] == "Level 0-1"
    assert all(lv in CAMPAIGN_LEVELS_SHIPPED for lv in cfg.levels)
    assert (cfg.unlock_rate, cfg.unlock_window, cfg.level_weight_floor) == (0.5, 20, 0.1)

    # The run name is deliberately NOT changed: models/campaign_gates/ holds the weights, best.zip and the
    # exploration archives, and runs/campaign_gates/status.json the chart history.
    assert train_cfg["run_name"] == RUN_NAME == live_train["run_name"]
    assert train_cfg["num_envs"] == 8, "eight games, which needs [Logging.Disk] Enabled = false in BepInEx.cfg"
    hyper, live_hyper = dict(train_cfg["hyperparams"]), dict(live_train["hyperparams"])
    assert hyper.pop("ent_coef") == 0.004 and live_hyper.pop("ent_coef") == 0.01, "the one training change"
    assert hyper == live_hyper, "ent_coef is the only hyperparameter that moves"
    assert train_cfg["policy_kwargs"] == live_train["policy_kwargs"]

    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert filled.curriculum_path == f"runs/{RUN_NAME}/curriculum.json"
    assert filled.explore_dir == f"models/{RUN_NAME}", "the eight archives stay where they are"
    header = GATES_PRELUDE.read_text(encoding="utf-8")
    assert "ent_coef" in header and "6.0" in header, "the header has to carry the falsifier and the tripwire"


def test_the_main_config_carries_the_gates_prelude_run_forward():
    """The config the live run resumes on after integration pause #2 (2026-09-17).

    The same run name and the same weights continue, so the burden is on every setting that is NOT one of the
    four deliberate changes to be character-for-character what the prelude config had. Four things may move:
    `levels` (and the new safety valve beside it), `max_steps`, and the two skull reward weights.
    """
    prelude_env, prelude_train = train.load_config(str(GATES_PRELUDE))
    env_dict, train_cfg = train.load_config(str(GATES_MAIN))
    prelude, cfg = EnvConfig.from_dict(prelude_env), EnvConfig.from_dict(env_dict)

    allowed = {"levels", "unlock_after_fresh_episodes", "max_steps", "rewards"}
    differing = {f.name for f in dataclasses.fields(EnvConfig)
                 if getattr(cfg, f.name) != getattr(prelude, f.name)}
    assert differing <= allowed, f"env settings changed besides the four: {sorted(differing - allowed)}"

    # Reward weights: only the two skull terms move, and everything else is identical to what the 6.8M steps
    # already on the clock were trained under.
    changed = {f.name for f in dataclasses.fields(RewardConfig)
               if getattr(cfg.rewards, f.name) != getattr(prelude.rewards, f.name)}
    assert changed == {"item_pickup", "item_placed"}, sorted(changed)
    assert (cfg.rewards.item_pickup, cfg.rewards.item_placed) == (10.0, 20.0)
    assert cfg.rewards.item_placed > cfg.rewards.item_pickup, "or collect-and-abandon is worth it alone"
    # The two bounds the skull-fixes checklist's arithmetic actually sets, which are about the DETOUR an
    # off-route skull could buy. 0-2 ships exactly one such skull, at a secret altar whose gate reports
    # `hops: null` and so can never become a target (verified in game, 2026-09-17).
    #   - a bare pickup must not out-earn a route rung, or wandering off to hold a skull beats walking the
    #     ladder: item_pickup <= gate;
    #   - the whole puzzle is worth two rungs and no more: item_pickup + item_placed <= 2 * gate. At 30 that is
    #     also ~100 game seconds of `time` (0.3/s), so a long detour to a skull the route does not need roughly
    #     breaks even at best.
    # `item_placed` alone is allowed above `gate`: it is the terminal event and has to out-weigh the pickup, and
    # it is only ever paid at an altar the agent is already standing at, never for travel.
    assert cfg.rewards.item_pickup <= cfg.rewards.gate
    assert cfg.rewards.item_pickup + cfg.rewards.item_placed <= 2 * cfg.rewards.gate

    # The 11 levels: mission order, Tier A + Tier B, starting where the run already is.
    assert cfg.levels[0] == "Level 0-1", "the first rung is always unlocked, so it must be the trained level"
    assert cfg.levels == ["Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4", "Level 1-1", "Level 1-2",
                          "Level 2-1", "Level 2-2", "Level 2-3", "Level 3-1", "Level 4-1"]
    assert all(lv in CAMPAIGN_LEVELS_SHIPPED for lv in cfg.levels)
    assert len(set(cfg.levels)) == len(cfg.levels)
    order = [CAMPAIGN_LEVELS.index(lv) for lv in cfg.levels]
    assert order == sorted(order), "mission order, so the ladder matches how the game unlocks"
    assert set(prelude.levels) <= set(cfg.levels), "no level the run is already training may be dropped"

    # The safety valve, and the truncation cap.
    assert cfg.unlock_after_fresh_episodes == 600 and prelude.unlock_after_fresh_episodes == 0
    assert (cfg.unlock_rate, cfg.unlock_window, cfg.level_weight_floor) == (0.5, 20, 0.1)
    assert cfg.max_steps == 12000 and prelude.max_steps == 9000
    assert cfg.max_steps_per_level == {}, "one cap for every level; a per-level cap is a separate decision"
    assert cfg.stuck_seconds == prelude.stuck_seconds == 45, "a stuck episode must still end early"

    # Same run, same weights, same optimiser: nothing in the train section moves at all.
    assert train_cfg["run_name"] == RUN_NAME == prelude_train["run_name"]
    assert train_cfg["hyperparams"] == prelude_train["hyperparams"], "ent_coef included: 0.004 stays"
    assert train_cfg["hyperparams"]["ent_coef"] == 0.004
    assert train_cfg["policy_kwargs"] == prelude_train["policy_kwargs"]
    # 12 games, up from 8, on the throughput A/B measured at this pause (188.6 -> 236.5 steps/s steady state).
    # `n_steps` in the config is the TOTAL rollout -- train.py divides it by num_envs -- so the PPO batch, and
    # therefore the update, is unchanged by the instance count.
    assert (train_cfg["num_envs"], prelude_train["num_envs"]) == (12, 8)
    assert train_cfg["hyperparams"]["n_steps"] == 2048

    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert filled.curriculum_path == f"runs/{RUN_NAME}/curriculum.json"
    assert filled.explore_dir == f"models/{RUN_NAME}", "the archives stay where they are"
    header = GATES_MAIN.read_text(encoding="utf-8")
    assert "6854" in header, "the max_steps change has to carry the measurement it came from"
    assert "6.0" in header, "and the entropy tripwire has to survive the config change"


def test_the_full_config_is_the_main_config_with_more_levels():
    """configs/campaign_gates_full.yaml: the whole routed campaign, and FOUR listed differences from main.

    The route fallback adds no reward term and no hyperparameter -- a room rung pays through `gate` and
    `gate_approach` exactly as a door does -- so the differences this config is allowed to carry are named one
    by one, and anything else is a change nobody decided. The three beyond the levels list are the 2026-09-17
    revision: where target patience may act, which layer a collapsed level uses, and the entropy floor.
    `ent_coef` itself is untouched -- the floor raises it only while the policy is sharpening past 5 nats.
    """
    main_env, main_train = train.load_config(str(GATES_MAIN))
    env_dict, train_cfg = train.load_config(str(GATES_FULL))
    main, cfg = EnvConfig.from_dict(main_env), EnvConfig.from_dict(env_dict)

    # The raw YAML, which is where "written out on purpose" lives: two of these three are the defaults, and
    # stating them is what makes flipping one a deliberate act rather than a silent inherit.
    assert set(env_dict) - set(main_env) == {"gate_patience_mode", "prefer_route_when_collapsed"}
    assert set(main_env) - set(env_dict) == set()
    assert {k for k in set(env_dict) & set(main_env) if env_dict[k] != main_env[k]} == {"levels"}
    assert set(train_cfg) - set(main_train) == {"ent_floor", "ent_coef_max"}
    assert {k for k in set(train_cfg) & set(main_train) if train_cfg[k] != main_train[k]} == set(), \
        "the optimiser, the run name and num_envs are character for character main's"

    # And the parsed effect of those keys: patience is level-conditional (its default), and the route preference
    # is ON since 2026-09-17 21:58 (0-3 sat at 0 of 75 fresh completions under gates + patience), so the EnvConfig
    # fields that differ are the levels list and that one flag.
    differing = {f.name for f in dataclasses.fields(EnvConfig)
                 if getattr(cfg, f.name) != getattr(main, f.name)}
    assert differing == {"levels", "prefer_route_when_collapsed"},         f"only the levels list and the route preference may change, got {sorted(differing)}"
    assert cfg.gate_patience_mode == "collapsed" and cfg.prefer_route_when_collapsed is True
    assert cfg.gate_target_patience_s == 20.0, "the window itself does not move; only where it may act"
    assert cfg.rewards == main.rewards, "no reward weight moves: the fallback adds no term"
    assert train_cfg["num_envs"] == 12 and train_cfg["run_name"] == RUN_NAME
    assert train_cfg["hyperparams"]["ent_coef"] == 0.004, "the BASE coefficient is unchanged"
    assert (train_cfg["ent_floor"], train_cfg["ent_coef_max"]) == (5.0, 0.02)

    # The 30 levels: every shipped level with a gate ladder OR a route file, in mission order.
    unrouted = {"Level 1-3", "Level 5-4", "Level 6-2"}  # spec §11.1: no route signal of any kind
    assert set(cfg.levels) == CAMPAIGN_LEVELS_SHIPPED - unrouted
    assert len(cfg.levels) == 30 and len(set(cfg.levels)) == 30
    assert not unrouted & set(cfg.levels), "a level with no route signal would train on the exit vector alone"
    assert "Level 9-1" not in cfg.levels and "Level 9-2" not in cfg.levels, "no scene bundle in this build"
    order = [CAMPAIGN_LEVELS.index(lv) for lv in cfg.levels]
    assert order == sorted(order), "mission order, because unlock_next walks the list as a chained ladder"
    assert cfg.levels[0] == "Level 0-1", "levels[0] is always unlocked, so it must be the trained level"
    assert set(main.levels) <= set(cfg.levels), "no level the run is already training may be dropped"
    # The 12 levels the route files cover, and the four that stop at a skull lock until S3 (spec §10).
    rooms = {"Level 0-5", "Level 1-4", "Level 2-4", "Level 4-2", "Level 4-4", "Level 5-2",
             "Level 7-1", "Level 7-2", "Level 7-3", "Level 7-4", "Level 8-3", "Level 8-4"}
    assert rooms <= set(cfg.levels) and len(set(cfg.levels) - rooms) == 18, "18 gates levels + 12 room levels"
    assert cfg.route_fallback is True and cfg.route_dir == "", "the packaged routes/ folder, on by default"
    assert (cfg.route_exit_tol_m, cfg.route_seed_m) == (5.0, 150.0)
    assert cfg.max_steps_per_level == {}, "one cap for every level; a per-level cap is its own decision"

    header = GATES_FULL.read_text(encoding="utf-8")
    for needed in ("1-3", "5-4", "6-2", "6x", "rungs", "route_source"):
        assert needed in header, f"the header has to carry {needed}: why the list is this list, and the levers"


def test_the_full_config_builds_a_479_input_curriculum_env():
    env_dict, _ = train.load_config(str(GATES_FULL))
    cfg = train.fill_campaign_dirs(EnvConfig.from_dict(env_dict), MODEL_DIR, RUN_DIR)
    env = UltrakillEnv(dataclasses.replace(cfg, explore_dir="", best_runs_dir="", curriculum_path=""))
    try:
        assert env.observation_space.shape == (479,), "no level id enters the observation, by design"
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]
        assert env.level == "Level 0-1" and env._max_steps() == 12000
        # No route file ships for 0-1, so a worker that starts on the first rung is on layer 1, unchanged.
        assert env.gates._route is None and env.gates.route_source == 0
    finally:
        env.close()


def test_the_main_config_builds_a_479_input_curriculum_env():
    env_dict, _ = train.load_config(str(GATES_MAIN))
    cfg = train.fill_campaign_dirs(EnvConfig.from_dict(env_dict), MODEL_DIR, RUN_DIR)
    env = UltrakillEnv(dataclasses.replace(cfg, explore_dir="", best_runs_dir="", curriculum_path=""))
    try:
        assert env.observation_space.shape == (479,), "no level id enters the observation, by design"
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]
        assert env.level == "Level 0-1", "with no curriculum file to read, a worker starts on the first rung"
        assert env._max_steps() == 12000
    finally:
        env.close()


def test_every_campaign_setting_is_a_real_field():
    # EnvConfig.from_dict drops keys it does not know, so a misspelt setting would silently use its default.
    for path in (CONFIG, PRELUDE, LEVEL_1_1, GATES_PRELUDE, GATES_MAIN, GATES_FULL):
        env_dict, _ = train.load_config(str(path))
        unknown = sorted(set(env_dict) - field_names(EnvConfig))
        assert not unknown, f"{path.name}: env keys EnvConfig does not know: {unknown}"
        unknown = sorted(set(env_dict["rewards"]) - field_names(RewardConfig))
        assert not unknown, f"{path.name}: reward keys RewardConfig does not know: {unknown}"


def test_prelude_config_builds_a_479_input_curriculum_env():
    env_dict, train_cfg = train.load_config(str(PRELUDE))
    cfg = EnvConfig.from_dict(env_dict)
    assert cfg.mode == "campaign"
    # The survey's Tier A Prelude, in mission order. 0-2 needs the skull carry and 0-5 has no goal room at all.
    assert cfg.levels == ["Level 0-1", "Level 0-3", "Level 0-4"]
    assert cfg.levels[0] == "Level 0-1" and all(lv in CAMPAIGN_LEVELS_SHIPPED for lv in cfg.levels)
    assert (cfg.unlock_rate, cfg.unlock_window, cfg.level_weight_floor) == (0.5, 20, 0.1)
    assert cfg.curriculum_path == ""  # train.py fills it per run
    assert train_cfg["run_name"] == "campaign_prelude", "a new run name: the campaign block changes meaning"
    env = UltrakillEnv(cfg)
    try:
        assert env.observation_space.shape == (479,), "no level id enters the observation, by design"
        assert list(env.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]
        assert env.level == "Level 0-1"
    finally:
        env.close()


def test_level_1_1_config_ships_the_skull_weights_at_zero():
    env_dict, train_cfg = train.load_config(str(LEVEL_1_1))
    cfg = EnvConfig.from_dict(env_dict)
    assert cfg.level == "Level 1-1" and cfg.levels == []
    # Both stay 0.0 until the three in-game checks pass: a skull that cannot physically be picked up turns S4
    # from a fix into a 134 m detour the agent still pays ~20 gate_approach to start.
    assert cfg.rewards.item_pickup == 0.0 and cfg.rewards.item_placed == 0.0
    assert cfg.subgoal_punch_range_m == 4.0  # Punch.ActiveFrame's own reach
    assert train_cfg["run_name"] == "campaign_1-1"
    header = LEVEL_1_1.read_text(encoding="utf-8")
    assert "15.0" in header and "§8" in header, "the header has to say when and why the weights go up"
    env = UltrakillEnv(cfg)
    try:
        assert env.observation_space.shape == (479,) and env.level == "Level 1-1"
    finally:
        env.close()


def test_fill_campaign_dirs_fills_the_curriculum_path_only_for_a_levels_run():
    multi = EnvConfig(mode="campaign", levels=["Level 0-1", "Level 0-3"])
    filled = train.fill_campaign_dirs(multi, Path("models") / "campaign_prelude", Path("runs") / "campaign_prelude")
    assert filled.curriculum_path == "runs/campaign_prelude/curriculum.json"
    assert multi.curriculum_path == "", "the config passed in is not modified"
    assert train.fill_campaign_dirs(EnvConfig(mode="campaign"), MODEL_DIR, RUN_DIR).curriculum_path == ""
    custom = EnvConfig(mode="campaign", levels=["Level 0-1"], curriculum_path="D:/elsewhere.json")
    assert train.fill_campaign_dirs(custom, MODEL_DIR, RUN_DIR).curriculum_path == "D:/elsewhere.json"
    # The parent directory name is what the workers check the file's run_name against, so it must be the run.
    assert Path(filled.curriculum_path).parent.name == "campaign_prelude"


def test_eval_pins_one_level_from_a_curriculum_config():
    """`--record-times` writes a row only a FASTER time can replace, so an eval must never sample a level."""
    eval_path = ROOT / "scripts" / "eval.py"
    source = eval_path.read_text(encoding="utf-8")
    assert "cfg.levels = []" in source or "cfg.levels" in source

    def resolve(cfg: EnvConfig, level: str | None) -> EnvConfig:
        """The block eval.main runs, applied to a copy so this test needs neither a model nor a game."""
        cfg = dataclasses.replace(cfg)
        if level:
            cfg.level, cfg.levels = level, []
        elif cfg.levels:
            cfg.level, cfg.levels = cfg.levels[0], []
        cfg.curriculum_path = ""
        return cfg

    import tempfile

    env_dict, _ = train.load_config(str(PRELUDE))
    with tempfile.TemporaryDirectory() as tmp:
        # A temp model directory, because close() saves this env's exploration archives into `explore_dir` and
        # a test must not write into the repo's models/.
        base = train.fill_campaign_dirs(EnvConfig.from_dict(env_dict), Path(tmp) / "models", Path(tmp) / "runs")
        assert base.curriculum_path and base.levels

        default = UltrakillEnv(resolve(base, None))
        try:
            assert default.level == "Level 0-1" and default.cfg.levels == [] and default.cfg.curriculum_path == ""
        finally:
            default.close()

        chosen = UltrakillEnv(resolve(base, "Level 0-4"))
        try:
            assert chosen.level == "Level 0-4" and chosen.cfg.levels == []
        finally:
            chosen.close()
        # Nothing was written anywhere but the temp directory.
        assert sorted(p.name for p in (Path(tmp) / "models").glob("*.npz")) == [
            "explore_Level_0-1_47800.npz", "explore_Level_0-4_47800.npz"]


def test_train_section_matches_the_spec():
    _, t = train.load_config(str(CONFIG))
    # A new run name is required: part_novelty, cells_new, part_door_unlock and the action space all change meaning.
    assert (t["algo"], t["num_envs"], t["run_name"], t["timesteps"], t["save_every"]) == ("ppo", 5, RUN_NAME, 20_000_000, 50_000)
    assert t["policy_kwargs"] == {"net_arch": [512, 512]}
    assert t["hyperparams"] == {
        "learning_rate": 0.0002, "n_steps": 2048, "batch_size": 512, "n_epochs": 5,
        "gamma": 0.998, "gae_lambda": 0.95, "ent_coef": 0.01, "target_kl": 0.03,
    }


def test_fill_campaign_dirs_fills_only_empty_dirs():
    cfg = EnvConfig(mode="campaign")
    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert (filled.explore_dir, filled.best_runs_dir) == (f"models/{RUN_NAME}", f"runs/{RUN_NAME}/best_runs")
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # the config passed in is not modified

    custom = EnvConfig(mode="campaign", explore_dir="D:/archives", best_runs_dir="D:/best")
    kept = train.fill_campaign_dirs(custom, MODEL_DIR, RUN_DIR)
    assert (kept.explore_dir, kept.best_runs_dir) == ("D:/archives", "D:/best")

    half = train.fill_campaign_dirs(EnvConfig(mode="campaign", explore_dir="D:/archives"), MODEL_DIR, RUN_DIR)
    assert (half.explore_dir, half.best_runs_dir) == ("D:/archives", f"runs/{RUN_NAME}/best_runs")


def test_importing_the_trainer_sets_the_thread_and_validation_switches():
    """Both have to be in place before torch is used: OpenMP reads its variable at library load."""
    assert os.environ.get("KMP_BLOCKTIME") == "1"  # 11.03 -> 1.43 cores busy, same update time
    assert os.environ.get("OMP_NUM_THREADS") is None  # setting it to 1 costs 2.5x on the update
    assert torch.distributions.Distribution._validate_args is False  # 0.95 ms of a 3.1 ms forward pass


def test_action_entropy_callback_measures_each_look_dimension():
    """Total entropy stops being interpretable once a look mode leaves the yaw and pitch heads inert."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ultrakill_ai.spaces import PITCH_BINS, YAW_BINS, ObsLayout, action_space

    class Fake(__import__("gymnasium").Env):
        def __init__(self):
            self.observation_space = ObsLayout(campaign=True).space()
            self.action_space = action_space(campaign=True)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    torch.manual_seed(0)
    model = PPO("MlpPolicy", DummyVecEnv([Fake]), n_steps=128, batch_size=64, n_epochs=1,
                policy_kwargs={"net_arch": [16]}, device="cpu", verbose=0)
    callback = train.ActionEntropyCallback(sample=32)
    model.learn(total_timesteps=256, callback=callback)
    assert callback.enabled, "a diagnostic that fails should disable itself, but it should not fail here"
    import math
    assert abs(callback.entropies["look_mode"] - math.log(3)) < 0.01  # a fresh head is uniform
    assert abs(callback.entropies["yaw"] - math.log(len(YAW_BINS))) < 0.01
    assert abs(callback.entropies["pitch"] - math.log(len(PITCH_BINS))) < 0.01
    for name in ("yaw", "pitch", "look_mode"):
        assert f"train/entropy_{name}" in PPO_METRICS, name  # so they reach status.json and the dashboard


class _FakeLogger:
    def __init__(self, entropy_loss=None):
        self.name_to_value = {} if entropy_loss is None else {"train/entropy_loss": entropy_loss}
        self.recorded: dict[str, float] = {}

    def record(self, key, value):
        self.recorded[key] = value


class _FakeModel:
    """The two attributes EntropyFloorCallback touches: SB3's PPO reads `ent_coef` on every update."""

    def __init__(self, logger):
        self.logger = logger
        self.ent_coef = None


def _floor_step(callback, entropy):
    """One rollout boundary with `entropy` nats of total policy entropy behind it."""
    callback.model = _FakeModel(_FakeLogger(None if entropy is None else -entropy))
    callback._on_rollout_start()
    return callback.model


def test_the_entropy_floor_raises_the_coefficient_below_the_floor_and_stops_at_the_cap():
    cb = train.EntropyFloorCallback(floor=5.0, base=0.004, maximum=0.02)
    assert cb.live == 0.004
    model = _floor_step(cb, 4.2)
    assert abs(cb.live - 0.004 * 1.10) < 1e-12, cb.live
    assert model.ent_coef == cb.live, "the coefficient PPO reads is written, not just logged"
    assert abs(model.logger.recorded["train/ent_coef_live"] - cb.live) < 1e-12
    for _ in range(60):
        _floor_step(cb, 4.2)
    assert cb.live == 0.02, "capped at ent_coef_max however long entropy stays under the floor"


def test_the_entropy_floor_decays_back_to_the_base_and_no_further():
    cb = train.EntropyFloorCallback(floor=5.0, base=0.004, maximum=0.02)
    for _ in range(60):
        _floor_step(cb, 4.2)
    assert cb.live == 0.02
    _floor_step(cb, 6.5)  # a nat clear of the floor: the controller starts letting go
    assert abs(cb.live - 0.02 * 0.95) < 1e-12, cb.live
    for _ in range(200):
        _floor_step(cb, 9.0)
    assert cb.live == 0.004, "never below the config's own ent_coef"


def test_the_entropy_floor_holds_still_inside_the_band_and_does_nothing_at_all_when_off():
    cb = train.EntropyFloorCallback(floor=5.0, base=0.004, maximum=0.02)
    _floor_step(cb, 4.0)
    held = cb.live
    for entropy in (5.0, 5.5, 6.0):  # floor .. floor + 1: neither raise nor decay
        _floor_step(cb, entropy)
        assert cb.live == held, entropy
    off = train.EntropyFloorCallback(floor=0.0, base=0.004, maximum=0.02)
    model = _floor_step(off, 0.5)  # far below any floor, and still untouched
    assert off.live == 0.004 and model.ent_coef is None, "ent_floor 0 never writes the coefficient"
    assert model.logger.recorded["train/ent_coef_live"] == 0.004, "it still reports what PPO is using"
    blank = train.EntropyFloorCallback(floor=5.0, base=0.004)
    model = _floor_step(blank, None)  # the first rollout: no update has run, so there is no entropy yet
    assert blank.live == 0.004 and model.ent_coef is None
    assert blank.maximum == 0.02, "the default cap"


def test_the_entropy_floor_is_wired_into_the_run_and_reported():
    """Config -> callback -> status.json, and the resume behaviour stated in the docstring."""
    _, t = train.load_config(str(GATES_FULL))
    cb = train.EntropyFloorCallback(t["ent_floor"], t["hyperparams"]["ent_coef"], t.get("ent_coef_max", 0.02))
    assert (cb.floor, cb.base, cb.maximum) == (5.0, 0.004, 0.02)
    assert "train/ent_coef_live" in PPO_METRICS, "so it reaches status.json, the dashboard and poll_status"
    assert "does not survive a resume" in train.EntropyFloorCallback.__doc__, \
        "the resume behaviour is deliberate and has to be stated where it is read"


def test_a_campaign_checkpoint_loads_against_every_campaign_config():
    """The live campaign_gates run's weights must keep loading: 479 inputs, 12 action dimensions, unchanged.

    S1 adds no observation value (no level id, deliberately) and S4 adds none either (a sub-goal rides the
    existing target slots), so a checkpoint saved before either one loads into a curriculum run untouched.
    Built here rather than read from models/, so this needs no checkpoint on disk and cannot disturb a live run.
    """
    import tempfile

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    configs = {path.name: EnvConfig.from_dict(train.load_config(str(path))[0])
               for path in (CONFIG, PRELUDE, LEVEL_1_1, GATES_PRELUDE, GATES_MAIN, GATES_FULL)}
    envs = {}
    try:
        for name, cfg in configs.items():
            envs[name] = UltrakillEnv(cfg)
        spaces = {(env.observation_space.shape, tuple(env.action_space.nvec)) for env in envs.values()}
        assert len(spaces) == 1, f"the configs disagree on the spaces: {spaces}"
        shape, nvec = spaces.pop()
        assert shape == (479,) and nvec == (3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3)

        class Fake(gym.Env):
            def __init__(self, env):
                self.observation_space, self.action_space = env.observation_space, env.action_space

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        model = PPO("MlpPolicy", DummyVecEnv([lambda: Fake(envs[CONFIG.name])]), n_steps=64, batch_size=32,
                    n_epochs=1, policy_kwargs={"net_arch": [16]}, device="cpu", verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gates.zip"
            model.save(path)
            for name, env in envs.items():
                loaded = PPO.load(path, env=DummyVecEnv([lambda env=env: Fake(env)]), device="cpu")
                assert loaded.observation_space.shape == (479,), name
                assert list(loaded.action_space.nvec) == list(nvec), name
    finally:
        for env in envs.values():
            env.close()


def test_fill_campaign_dirs_leaves_cybergrind_alone():
    cfg = EnvConfig(mode="cybergrind")
    same = train.fill_campaign_dirs(cfg, Path("models") / "cybergrind_ppo_v2", Path("runs") / "cybergrind_ppo_v2")
    assert same == cfg and (same.explore_dir, same.best_runs_dir) == ("", "")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
