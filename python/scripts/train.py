"""Trains a PPO agent.

    python scripts/train.py --config configs/cybergrind.yaml
    python scripts/train.py --config configs/campaign_0-1.yaml --algo rppo
    python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind/latest.zip

Parallel training with several game instances (see scripts/games.py):
    python scripts/games.py launch --count 5
    python scripts/train.py --config configs/cybergrind.yaml --num-envs 5

Watch progress with:  python scripts/dashboard.py   (or: tensorboard --logdir runs)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402


class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode reward components and end reasons to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.parts: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            parts = self.parts[i]
            for name, value in info.get("reward_parts", {}).items():
                parts[name] += value
            if done:
                for name, value in parts.items():
                    self.logger.record_mean(f"reward_parts/{name}", value)
                if "end_reason" in info:
                    self.logger.record_mean(f"end_reason/{info['end_reason']}", 1.0)
                if "reset_seconds" in info:
                    self.logger.record_mean("time/reset_seconds", info["reset_seconds"])
                parts.clear()
        return True


def make_env(cfg: EnvConfig, info_keywords: tuple[str, ...]):
    def _init():
        return Monitor(UltrakillEnv(cfg), info_keywords=info_keywords)

    return _init


def load_config(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get("env", {}), data.get("train", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML file with env and train sections")
    parser.add_argument("--algo", choices=["ppo", "rppo"], help="ppo = MLP, rppo = LSTM (sb3-contrib RecurrentPPO)")
    parser.add_argument("--timesteps", type=int, help="total training steps for the run (resuming trains only the rest)")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="path to a saved model .zip to continue training")
    parser.add_argument("--device", default="cpu", help="cpu is usually fastest for MLP policies")
    parser.add_argument("--num-envs", type=int, help="game instances to train on in parallel (ports base-port..)")
    parser.add_argument("--base-port", type=int, help="port of the first game instance (default: env.port)")
    args = parser.parse_args()

    env_cfg_dict, train_cfg = load_config(args.config)
    env_cfg = EnvConfig.from_dict(env_cfg_dict)

    algo = args.algo or train_cfg.get("algo", "ppo")
    timesteps = args.timesteps or train_cfg.get("timesteps", 5_000_000)
    run_name = args.run_name or train_cfg.get("run_name") or f"{env_cfg.mode}_{algo}"
    model_dir = Path("models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")

    info_keywords = ("kills", "wave", "style", "deaths", "firing_frac", "on_target_frac") if env_cfg.mode == "cybergrind" else ("kills", "style", "route_progress")
    num_envs = args.num_envs or train_cfg.get("num_envs", 1)
    base_port = args.base_port or env_cfg.port
    env_fns = [make_env(replace(env_cfg, port=base_port + i), info_keywords) for i in range(num_envs)]
    venv = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

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
    # n_steps in the config is the total rollout size; SB3 counts it per environment.
    hyper["n_steps"] = max(64, hyper["n_steps"] // num_envs)
    policy_kwargs = train_cfg.get("policy_kwargs", {"net_arch": [512, 512]})

    if algo == "rppo":
        from sb3_contrib import RecurrentPPO

        cls, policy = RecurrentPPO, "MlpLstmPolicy"
    else:
        cls, policy = PPO, "MlpPolicy"

    if args.resume:
        # The rollout buffer is rebuilt for the current number of environments, and the config's
        # hyperparameters override the saved ones so tuning applies when resuming.
        model = cls.load(args.resume, env=venv, device=args.device, tensorboard_log="runs", **hyper)
    else:
        model = cls(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log="runs", device=args.device, verbose=1, **hyper)

    # timesteps is the total for the run. learn() adds its argument to the loaded step count when
    # resuming, so only the remaining steps are requested.
    remaining = max(0, timesteps - model.num_timesteps) if args.resume else timesteps
    progress = ProgressCallback(Path("runs") / run_name / "status.json", timesteps, run_name, num_envs)
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1, train_cfg.get("save_every", 50_000) // num_envs), save_path=str(model_dir), name_prefix="ckpt"),
        EpisodeStatsCallback(),
        progress,
    ])

    try:
        model.learn(total_timesteps=remaining, callback=callbacks, tb_log_name=run_name, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("Interrupted, saving.")
    finally:
        progress.mark_stopped()  # no-op if training finished normally
        model.save(model_dir / "latest")
        print(f"Saved {model_dir / 'latest.zip'}")
        try:
            venv.close()
        except (EOFError, BrokenPipeError, ConnectionError):
            pass  # A worker already died with its game.


if __name__ == "__main__":
    main()
