"""The trainer: PPO wiring, the training callbacks and `main()`.

**Everything heavy in a training run lives here rather than in `scripts/train.py`, and that is a memory
fix, not tidying.** `SubprocVecEnv` starts its workers with multiprocessing's *spawn* method, and a spawn
child re-executes the parent's `__main__` module top to bottom (as `__mp_main__`) before it unpickles
anything. While `import torch` and `from stable_baselines3 import PPO` sat at the top of
`scripts/train.py`, all twelve workers paid for them: measured at 0.91 GB of committed private bytes each
for 0.2 GB of working set, on a box whose commit limit had already been hit twice.

`scripts/train.py` is now a launcher whose module body imports nothing heavier than `pathlib`, so the
re-execution in each worker is free, and the heavy imports happen only where `main()` is actually called.
"""

from __future__ import annotations

import os

# Before anything imports torch (stable_baselines3 does, at module import) -- OpenMP reads this at library load.
# Measured: 11.03 -> 1.43 cores busy with no change to the update time, so the trainer stops burning 11 cores on
# spin-wait next to five games on the same 12-core CPU. Do NOT set OMP_NUM_THREADS=1: that costs 2.5x per update.
os.environ.setdefault("KMP_BLOCKTIME", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.utils import obs_as_tensor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig  # noqa: E402
from ultrakill_ai.envfactory import make_env  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402

# Argument validation in torch.distributions costs 0.95 ms of a 3.1 ms forward pass, about 2.9% of wall time.
torch.distributions.Distribution.set_default_validate_args(False)

ENTROPY_DIMS = {-1: "look_mode", -2: "pitch", -3: "yaw"}  # the action dimensions worth naming, from the end
ENTROPY_SAMPLE = 256  # rollout rows per entropy measurement; one forward pass an update is ~3 ms

# The PPO defaults a config's `train.hyperparams` block is merged OVER. Lifted out of `main()` unchanged so the
# resume path below can name the two that a rollout buffer is built from; every value is what it has always been.
DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 512,
    "n_epochs": 5,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}
# Hyperparameters that are BAKED INTO THE ROLLOUT BUFFER at construction, not read from the model per update.
# `RolloutBuffer.compute_returns_and_advantage` uses `self.gamma` and `self.gae_lambda`, the buffer's own copies.
BUFFER_HYPERPARAMS = ("gamma", "gae_lambda")


def apply_resume_hyperparams(model, hyper: dict) -> dict[str, tuple[float, float]]:
    """Forces the config's `gamma`/`gae_lambda` onto a RESUMED model and onto its rollout buffer.

    WHY THIS EXISTS. A saved SB3 zip carries the hyperparameters it was trained with, and the buffer that
    computes GAE keeps its OWN copies of `gamma` and `gae_lambda` taken at construction. So "the config says
    0.999 now" is only true of a resumed run if something actually writes it in both places -- and stages S1
    and S2 of docs/superpowers/specs/2026-09-20-speedrun-tech.md are exactly a gamma change applied to a run
    that only ever resumes (the driver re-execs `train.py --resume` every round).

    WHAT WAS MEASURED, 2026-09-20, against the installed stable_baselines3 2.9.0: `BaseAlgorithm.load` does
    `model.__dict__.update(data)` (the saved values) and then `model.__dict__.update(kwargs)` (ours) BEFORE
    calling `model._setup_model()`, and `OnPolicyAlgorithm._setup_model` builds the rollout buffer from
    `self.gamma` / `self.gae_lambda`. So on this version the config ALREADY wins, in the model and in the
    buffer, and this function is a no-op that returns nothing changed. It is kept because that is an
    undocumented ordering inside a third-party library, one line of which moving would silently train a gamma
    stage at the old gamma and read as "the change did nothing" -- and because `pinned by
    tests/test_resume_hyperparams.py` is cheaper than re-deriving the ordering at the next upgrade.

    Returns `{name: (before, after)}` for every value it actually had to move, which is normally empty.
    """
    moved: dict[str, tuple[float, float]] = {}
    buffer = getattr(model, "rollout_buffer", None)
    for name in BUFFER_HYPERPARAMS:
        if name not in hyper:
            continue
        want = float(hyper[name])
        for holder in (model, buffer):
            if holder is None:
                continue
            have = getattr(holder, name, None)
            if have is None or float(have) == want:
                continue
            moved.setdefault(name, (float(have), want))
            setattr(holder, name, want)
    return moved


def hyperparams_in_force(model, hyper: dict) -> str:
    """One log line of the values ACTUALLY in force once the model exists, read off the model, not the config.

    The buffer's own `gamma`/`gae_lambda` are printed separately because they are the ones GAE is computed
    with, and a run whose two disagree is the failure this line exists to make visible at a glance.
    """
    buffer = getattr(model, "rollout_buffer", None)
    live = ", ".join("%s=%s" % (name, getattr(model, name, "?"))
                     for name in ("gamma", "gae_lambda", "n_steps", "batch_size", "n_epochs", "ent_coef",
                                  "target_kl")
                     if name in hyper or hasattr(model, name))
    return "hyperparameters in force: %s | rollout buffer: gamma=%s, gae_lambda=%s" % (
        live, getattr(buffer, "gamma", "?"), getattr(buffer, "gae_lambda", "?"))


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


def close_vec_env(venv, timeout: float = 90.0) -> str:
    """Shuts the vectorized env down in bounded time, however wedged a worker is.

    `SubprocVecEnv.close()` cannot be trusted here and this is not a style preference. It does
    `if self.waiting: remote.recv()` on every pipe and then `process.join()` with no timeout, so ONE worker
    stuck in a bridge call keeps the trainer alive forever -- which is exactly how the 2026-09-17 freezes
    presented: the supervisor found a HUNG trainer at the ten-minute mark instead of a DEAD one at the
    one-minute mark, and paid a full twelve-game restart for it. A crash must reach the supervisor as an exit
    code, promptly. SB3 marks the workers daemonic, so terminating them is safe and the parent may exit.

    Returns what happened, for the log: "clean", "terminated" or "none".
    """
    processes = list(getattr(venv, "processes", []) or [])
    if not processes:  # DummyVecEnv, or an env that never started workers
        try:
            venv.close()
        except Exception:  # noqa: BLE001 - teardown may not raise over a run that is already ending
            pass
        return "none"

    venv.waiting = False  # never block reading a reply from a worker that may be wedged
    for remote in getattr(venv, "remotes", []):
        try:
            remote.send(("close", None))
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            pass  # a worker that already died needs no goodbye
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    stragglers = [p for p in processes if p.is_alive()]
    for process in stragglers:
        process.terminate()
    for process in stragglers:
        process.join(5.0)
    for process in processes:
        if process.is_alive():
            process.kill()
    venv.closed = True
    return "terminated" if stragglers else "clean"


def load_config(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get("env", {}), data.get("train", {})


def fill_run_dirs(cfg: EnvConfig, run_dir: Path) -> EnvConfig:
    """Settings only a training run may have, in either mode.

    `env_log_dir` is the per-worker attribution log (`runs/<run>/env_<port>.log`) and `bridge_relaunch` lets a
    worker restart its OWN game as the last rung of its recovery ladder. Both are filled here and nowhere else,
    on purpose: eval.py, bridge_test.py, campaign_check.py and every test build an `EnvConfig` of their own and
    must stay incapable of restarting a game process or of writing into a live run's log directory.
    """
    return replace(cfg,
                   env_log_dir=cfg.env_log_dir or run_dir.as_posix(),
                   bridge_relaunch=True)


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
    env_cfg = fill_run_dirs(env_cfg, Path("runs") / run_name)
    Path(env_cfg.env_log_dir).mkdir(parents=True, exist_ok=True)
    if env_cfg.best_runs_dir:
        Path(env_cfg.best_runs_dir).mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")

    info_keywords = ("kills", "wave", "style", "deaths", "firing_frac", "on_target_frac") if env_cfg.mode == "cybergrind" else CAMPAIGN_INFO_KEYS
    num_envs = args.num_envs or train_cfg.get("num_envs", 1)
    base_port = args.base_port or env_cfg.port
    env_fns = [make_env(replace(env_cfg, port=base_port + i), info_keywords) for i in range(num_envs)]
    venv = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

    hyper = {**DEFAULT_HYPERPARAMS, **train_cfg.get("hyperparams", {})}
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
        # ... and forced again onto the model AND its rollout buffer, because the buffer keeps its own copies
        # of gamma/gae_lambda and the zip carries the values the run was saved with. A no-op on
        # stable_baselines3 2.9.0, which already applies them in this order; see `apply_resume_hyperparams`.
        moved = apply_resume_hyperparams(model, hyper)
        for name, (before, after) in moved.items():
            print(f"resume: forced {name} {before} -> {after} (the saved model disagreed with the config)")
    else:
        model = cls(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log="runs", device=args.device, verbose=verbose, **hyper)
    # Printed on both paths, from the MODEL rather than from the config: a gamma stage (S1/S2 of the speedrun
    # spec) is judged on a number that has to be verifiable in the train log of the round it ran in.
    print(hyperparams_in_force(model, hyper), flush=True)

    # timesteps is the total for the run. learn() adds its argument to the loaded step count when
    # resuming, so only the remaining steps are requested.
    remaining = max(0, timesteps - model.num_timesteps) if args.resume else timesteps
    progress = ProgressCallback(Path("runs") / run_name / "status.json", timesteps, run_name, num_envs,
                                levels=env_cfg.levels, curriculum_path=env_cfg.curriculum_path,
                                unlock_rate=env_cfg.unlock_rate, unlock_window=env_cfg.unlock_window,
                                unlock_after_fresh_episodes=env_cfg.unlock_after_fresh_episodes,
                                level_weight_floor=env_cfg.level_weight_floor,
                                curriculum_weighting=env_cfg.curriculum_weighting,
                                curriculum_weight_cap=env_cfg.curriculum_weight_cap,
                                curriculum_blocked_fresh_episodes=env_cfg.curriculum_blocked_fresh_episodes)
    if env_cfg.levels:
        # Written before learn(), because SB3's _setup_learn calls env.reset() before _on_training_start ever
        # runs: without this the workers' first fresh start would read whatever happened to be on disk. The
        # unlock set is printed because _restore bails when status.json's run_name differs, so a run rename
        # silently re-locks every level and this is where that shows.
        progress.write_curriculum()
        print(f"curriculum: {len(env_cfg.levels)} levels, unlocked {progress.unlocked_levels or [env_cfg.levels[0]]}"
              f", weighting {env_cfg.curriculum_weighting}")
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
        try:
            model.save(model_dir / "latest")
            print(f"Saved {model_dir / 'latest.zip'}")
        except Exception as exc:  # noqa: BLE001 - a failed save may not block the teardown below
            print(f"Could not save latest.zip: {type(exc).__name__}: {exc}", flush=True)
        print(f"vec env teardown: {close_vec_env(venv)}", flush=True)
