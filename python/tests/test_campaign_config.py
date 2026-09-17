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
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.progress import PPO_METRICS  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402

CONFIG = ROOT / "configs" / "campaign_0-1.yaml"
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


def test_every_campaign_setting_is_a_real_field():
    # EnvConfig.from_dict drops keys it does not know, so a misspelt setting would silently use its default.
    env_dict, _ = train.load_config(str(CONFIG))
    unknown = sorted(set(env_dict) - field_names(EnvConfig))
    assert not unknown, f"env keys EnvConfig does not know: {unknown}"
    unknown = sorted(set(env_dict["rewards"]) - field_names(RewardConfig))
    assert not unknown, f"reward keys RewardConfig does not know: {unknown}"


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
