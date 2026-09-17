"""Trains a PPO agent.

    python scripts/train.py --config configs/cybergrind.yaml
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip
    python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind/latest.zip

Parallel training with several game instances (see scripts/games.py):
    python scripts/games.py launch --count 5
    python scripts/train.py --config configs/cybergrind.yaml --num-envs 5

Watch progress with:  python scripts/dashboard.py   (or: tensorboard --logdir runs)
"""

from __future__ import annotations

import os

# Before anything imports torch (stable_baselines3 does, at module import) -- OpenMP reads this at library load.
# Measured: 11.03 -> 1.43 cores busy with no change to the update time, so the trainer stops burning 11 cores on
# spin-wait next to five games on the same 12-core CPU. Do NOT set OMP_NUM_THREADS=1: that costs 2.5x per update.
os.environ.setdefault("KMP_BLOCKTIME", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import obs_as_tensor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402

# Argument validation in torch.distributions costs 0.95 ms of a 3.1 ms forward pass, about 2.9% of wall time.
torch.distributions.Distribution.set_default_validate_args(False)

ENTROPY_DIMS = {-1: "look_mode", -2: "pitch", -3: "yaw"}  # the action dimensions worth naming, from the end
ENTROPY_SAMPLE = 256  # rollout rows per entropy measurement; one forward pass an update is ~3 ms


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


class ActionEntropyCallback(BaseCallback):
    """Logs per-dimension action entropy, so total entropy stays readable once the look modes exist.

    On a step where look mode 1 or 2 aims, the yaw and pitch dimensions are causally inert: no advantage flows
    back into them and only the entropy bonus acts, pushing them toward uniform in proportion to the non-mode-0
    share. Total entropy then mixes a head being trained with a head being pushed around, and `ent_coef` stops
    being an interpretable signal. These three are read next to `look_free_frac` on the dashboard.

    Measured on a sample of the finished rollout, so it costs one forward pass an update. Any failure (a
    recurrent policy, a future distribution type) disables it rather than interrupting training.
    """

    def __init__(self, sample: int = ENTROPY_SAMPLE):
        super().__init__()
        self.sample = sample
        self.enabled = True
        self.entropies: dict[str, float] = {}

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Measures on the finished rollout; the values are recorded at the next rollout start (see below)."""
        if not self.enabled:
            return
        try:
            buffer = self.model.rollout_buffer
            flat = buffer.observations.reshape(-1, buffer.obs_shape[0])
            obs = obs_as_tensor(flat[: self.sample], self.model.device)
            with torch.no_grad():
                dists = self.model.policy.get_distribution(obs).distribution
            self.entropies = {name: float(dists[index].entropy().mean())
                              for index, name in ENTROPY_DIMS.items() if len(dists) >= -index}
        except Exception as exc:  # never let a diagnostic stop a run
            self.enabled = False
            self.entropies = {}
            print(f"ActionEntropyCallback disabled: {exc}")

    def _on_rollout_start(self) -> None:
        # SB3 dumps the logger between rollout end and train(), so anything recorded at rollout end is wiped
        # before it reaches TensorBoard or ProgressCallback. Recording here gives these the same lifetime as
        # PPO's own train/* metrics, and this callback runs before ProgressCallback in the list that reads them.
        for name, value in self.entropies.items():
            self.logger.record(f"train/entropy_{name}", value)


class EntropyFloorCallback(BaseCallback):
    """Holds total policy entropy above a floor by adapting `ent_coef` between updates.

    This run's failure mode, twice over, is a policy that sharpens past its own peak and then gets worse: the
    Cyber Grind run peaked near 3.5 nats and degraded while entropy kept falling (CLAUDE.md, "second peak and
    second collapse"), and the live campaign run went |entropy_loss| 7.88 at 9.58M steps to 5.57 at 11.0M on a
    FIXED `ent_coef` of 0.004. A fixed coefficient cannot defend a floor: the entropy bonus is a constant pull
    against an advantage signal that grows as the policy gets confident, so the same 0.004 that held 7.9 nats
    holds nothing at all later. This is the smallest controller that can -- one scalar, read once per update.

        floor = train.ent_floor       total entropy in NATS, summed over the action dimensions. 0 = off,
                                      and off means this callback never writes `ent_coef` at all.
        base  = train.hyperparams.ent_coef      the floor it never goes below, i.e. today's fixed value
        cap   = train.ent_coef_max    default 0.02, five times the base: the most it is ever allowed to push

    Each rollout, reading the entropy of the update that just finished (SB3 logs `train/entropy_loss`, which is
    NEGATED mean total entropy):

        entropy <  floor          ->  ent_coef *= 1.10, capped at `cap`
        entropy >  floor + 1.0    ->  ent_coef *= 0.95, never below `base`
        otherwise                 ->  unchanged (the 1 nat band is what stops it oscillating)

    Multiplicative both ways, so it is scale-free and slow: from 0.004 it takes 17 updates to reach the 0.02
    cap and 34 to come back, against ~2000 updates in a 20M-step run. It is a floor, never a ceiling -- above
    the band it only ever returns to `base`, so a run that is comfortably entropic trains at exactly the
    coefficient its config asks for and this callback is invisible.

    **It does not survive a resume, deliberately.** `ent_coef` is not in the checkpoint (SB3 stores it as a
    hyperparameter and `train.py` overrides it from the config on every resume anyway), so a resumed run starts
    at `base` and re-adapts over the first few updates. That is the honest behaviour: the state it would carry
    is a claim about a policy that has just been reloaded, and re-measuring costs a few thousand steps.
    """

    RAISE = 1.10
    DECAY = 0.95
    BAND = 1.0  # nats above the floor before the coefficient starts coming back down

    def __init__(self, floor: float, base: float, maximum: float = 0.02):
        super().__init__()
        self.floor = max(0.0, float(floor))
        self.base = float(base)
        self.maximum = max(float(base), float(maximum))
        self.live = float(base)
        self.entropy: float | None = None  # total entropy of the last update, for the log line

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # PPO's train() records into the logger after the rollout ends and the values live until the next dump,
        # so this reads the update that just ran -- the same window ProgressCallback reads its PPO metrics in.
        # Recorded here too (not at rollout end), so `train/ent_coef_live` has the same lifetime as train/*.
        values = getattr(self.logger, "name_to_value", {})
        entropy_loss = values.get("train/entropy_loss")
        if self.floor > 0 and entropy_loss is not None:
            try:
                entropy = -float(entropy_loss)
            except (TypeError, ValueError):
                entropy = None
            if entropy is not None and math.isfinite(entropy):
                self.entropy = entropy
                if entropy < self.floor:
                    self.live = min(self.maximum, self.live * self.RAISE)
                elif entropy > self.floor + self.BAND:
                    self.live = max(self.base, self.live * self.DECAY)
                self.model.ent_coef = self.live
        self.logger.record("train/ent_coef_live", self.live)


def make_env(cfg: EnvConfig, info_keywords: tuple[str, ...]):
    def _init():
        return Monitor(UltrakillEnv(cfg), info_keywords=info_keywords)

    return _init


def load_config(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get("env", {}), data.get("train", {})


def fill_campaign_dirs(cfg: EnvConfig, model_dir: Path, run_dir: Path) -> EnvConfig:
    """Campaign runs keep their exploration archives beside the checkpoints and their best runs beside the logs.

    Only empty settings are filled, so a config can still point either one elsewhere. Cyber Grind is returned as
    is. `curriculum_path` is filled only for a multi-level run: with no `levels` it stays empty and no worker
    ever opens a curriculum file.
    """
    if cfg.mode != "campaign":
        return cfg
    return replace(
        cfg,
        explore_dir=cfg.explore_dir or model_dir.as_posix(),
        best_runs_dir=cfg.best_runs_dir or (run_dir / "best_runs").as_posix(),
        curriculum_path=cfg.curriculum_path or ((run_dir / "curriculum.json").as_posix() if cfg.levels else ""),
    )


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
    env_cfg = fill_campaign_dirs(env_cfg, model_dir, Path("runs") / run_name)
    if env_cfg.best_runs_dir:
        Path(env_cfg.best_runs_dir).mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")

    info_keywords = ("kills", "wave", "style", "deaths", "firing_frac", "on_target_frac") if env_cfg.mode == "cybergrind" else CAMPAIGN_INFO_KEYS
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

    verbose = train_cfg.get("verbose", 1)
    if args.resume:
        # The rollout buffer is rebuilt for the current number of environments, and the config's
        # hyperparameters override the saved ones so tuning applies when resuming. `verbose` belongs on this
        # branch too: without it a resumed run's train log is 0 bytes, which is most of this project's runs.
        model = cls.load(args.resume, env=venv, device=args.device, tensorboard_log="runs", verbose=verbose, **hyper)
    else:
        model = cls(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log="runs", device=args.device, verbose=verbose, **hyper)

    # timesteps is the total for the run. learn() adds its argument to the loaded step count when
    # resuming, so only the remaining steps are requested.
    remaining = max(0, timesteps - model.num_timesteps) if args.resume else timesteps
    progress = ProgressCallback(Path("runs") / run_name / "status.json", timesteps, run_name, num_envs,
                                levels=env_cfg.levels, curriculum_path=env_cfg.curriculum_path,
                                unlock_rate=env_cfg.unlock_rate, unlock_window=env_cfg.unlock_window,
                                unlock_after_fresh_episodes=env_cfg.unlock_after_fresh_episodes,
                                level_weight_floor=env_cfg.level_weight_floor)
    if env_cfg.levels:
        # Written before learn(), because SB3's _setup_learn calls env.reset() before _on_training_start ever
        # runs: without this the workers' first fresh start would read whatever happened to be on disk. The
        # unlock set is printed because _restore bails when status.json's run_name differs, so a run rename
        # silently re-locks every level and this is where that shows.
        progress.write_curriculum()
        print(f"curriculum: {len(env_cfg.levels)} levels, unlocked {progress.unlocked_levels or [env_cfg.levels[0]]}")
    # The entropy floor, if the config asks for one. Before `progress` in the list, so the coefficient it
    # records is in the logger when ProgressCallback copies PPO_METRICS into status.json.
    entropy_floor = EntropyFloorCallback(train_cfg.get("ent_floor", 0.0), hyper["ent_coef"],
                                         train_cfg.get("ent_coef_max", 0.02))
    if entropy_floor.floor > 0:
        print(f"entropy floor: {entropy_floor.floor} nats, ent_coef {entropy_floor.base} -> at most "
              f"{entropy_floor.maximum} (starts at the base on every resume and re-adapts)")
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1, train_cfg.get("save_every", 50_000) // num_envs), save_path=str(model_dir), name_prefix="ckpt"),
        EpisodeStatsCallback(),
        ActionEntropyCallback(),
        entropy_floor,
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
