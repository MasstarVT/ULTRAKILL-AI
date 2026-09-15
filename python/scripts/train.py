"""Trains a PPO agent.

    python scripts/train.py --config configs/cybergrind.yaml
    python scripts/train.py --config configs/campaign_0-1.yaml --algo rppo
    python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind/latest.zip

Watch progress with:  tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402


class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode reward components and end reasons to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.parts = defaultdict(float)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for name, value in info.get("reward_parts", {}).items():
                self.parts[name] += value
            if done:
                for name, value in self.parts.items():
                    self.logger.record_mean(f"reward_parts/{name}", value)
                if "end_reason" in info:
                    self.logger.record_mean(f"end_reason/{info['end_reason']}", 1.0)
                self.parts.clear()
        return True


def load_config(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get("env", {}), data.get("train", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML file with env and train sections")
    parser.add_argument("--algo", choices=["ppo", "rppo"], help="ppo = MLP, rppo = LSTM (sb3-contrib RecurrentPPO)")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="path to a saved model .zip to continue training")
    parser.add_argument("--device", default="cpu", help="cpu is usually fastest for MLP policies")
    args = parser.parse_args()

    env_cfg_dict, train_cfg = load_config(args.config)
    env_cfg = EnvConfig.from_dict(env_cfg_dict)

    algo = args.algo or train_cfg.get("algo", "ppo")
    timesteps = args.timesteps or train_cfg.get("timesteps", 5_000_000)
    run_name = args.run_name or train_cfg.get("run_name") or f"{env_cfg.mode}_{algo}"
    model_dir = Path("models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")

    info_keywords = ("kills", "wave", "style") if env_cfg.mode == "cybergrind" else ("kills", "style", "route_progress")
    venv = DummyVecEnv([lambda: Monitor(UltrakillEnv(env_cfg), info_keywords=info_keywords)])

    hyper = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        **train_cfg.get("hyperparams", {}),
    }
    policy_kwargs = train_cfg.get("policy_kwargs", {"net_arch": [512, 512]})

    if algo == "rppo":
        from sb3_contrib import RecurrentPPO

        cls, policy = RecurrentPPO, "MlpLstmPolicy"
    else:
        cls, policy = PPO, "MlpPolicy"

    if args.resume:
        model = cls.load(args.resume, env=venv, device=args.device, tensorboard_log="runs")
    else:
        model = cls(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log="runs", device=args.device, verbose=1, **hyper)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=train_cfg.get("save_every", 50_000), save_path=str(model_dir), name_prefix="ckpt"),
        EpisodeStatsCallback(),
    ])

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name=run_name, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("Interrupted, saving.")
    finally:
        model.save(model_dir / "latest")
        venv.close()
        print(f"Saved {model_dir / 'latest.zip'}")


if __name__ == "__main__":
    main()
