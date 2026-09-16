"""The campaign 0-1 config and train.py's campaign wiring. No game needed:  python tests/test_campaign_config.py  (or pytest)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402

CONFIG = ROOT / "configs" / "campaign_0-1.yaml"
MODEL_DIR = Path("models") / "campaign_ppo"
RUN_DIR = Path("runs") / "campaign_ppo"


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
    assert cfg.pitch_limit_deg == 0.0  # the pitch band is a Cyber Grind aiming fix, off for the campaign
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # train.py fills these per run
    for name, value in env_dict["rewards"].items():
        assert getattr(cfg.rewards, name) == value, name
    assert cfg.rewards.style == 0.0 and cfg.rewards.time == 0.01 and cfg.rewards.level_complete == 100.0

    env = UltrakillEnv(cfg)  # builds without a game: nothing connects until the first reset
    try:
        assert env.cfg.layout.campaign
        assert env.observation_space.shape == (479,)
        assert env.cfg.rewards == cfg.rewards
    finally:
        env.close()


def test_every_campaign_setting_is_a_real_field():
    # EnvConfig.from_dict drops keys it does not know, so a misspelt setting would silently use its default.
    env_dict, _ = train.load_config(str(CONFIG))
    unknown = sorted(set(env_dict) - field_names(EnvConfig))
    assert not unknown, f"env keys EnvConfig does not know: {unknown}"
    unknown = sorted(set(env_dict["rewards"]) - field_names(RewardConfig))
    assert not unknown, f"reward keys RewardConfig does not know: {unknown}"


def test_train_section_matches_the_spec():
    _, t = train.load_config(str(CONFIG))
    assert (t["algo"], t["num_envs"], t["run_name"], t["timesteps"], t["save_every"]) == ("ppo", 5, "campaign_ppo", 20_000_000, 50_000)
    assert t["policy_kwargs"] == {"net_arch": [512, 512]}
    assert t["hyperparams"] == {
        "learning_rate": 0.0002, "n_steps": 2048, "batch_size": 512, "n_epochs": 5,
        "gamma": 0.998, "gae_lambda": 0.95, "ent_coef": 0.01, "target_kl": 0.03,
    }


def test_fill_campaign_dirs_fills_only_empty_dirs():
    cfg = EnvConfig(mode="campaign")
    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert (filled.explore_dir, filled.best_runs_dir) == ("models/campaign_ppo", "runs/campaign_ppo/best_runs")
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # the config passed in is not modified

    custom = EnvConfig(mode="campaign", explore_dir="D:/archives", best_runs_dir="D:/best")
    kept = train.fill_campaign_dirs(custom, MODEL_DIR, RUN_DIR)
    assert (kept.explore_dir, kept.best_runs_dir) == ("D:/archives", "D:/best")

    half = train.fill_campaign_dirs(EnvConfig(mode="campaign", explore_dir="D:/archives"), MODEL_DIR, RUN_DIR)
    assert (half.explore_dir, half.best_runs_dir) == ("D:/archives", "runs/campaign_ppo/best_runs")


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
