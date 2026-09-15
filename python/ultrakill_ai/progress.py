"""Live training progress for scripts/dashboard.py.

`ProgressCallback` collects episode stats during `model.learn()` and writes a small JSON status file
(atomically, at most every few seconds). The dashboard only reads that file, so it never touches the
training process.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

EPISODE_WINDOW = 100
RATE_WINDOW_S = 45.0  # steps/s is measured over this much recent wall time
HISTORY_EVERY_S = 20.0
HISTORY_MAX = 2000
PPO_METRICS = (
    "train/entropy_loss",
    "train/approx_kl",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance",
    "train/clip_fraction",
    "train/loss",
    "train/learning_rate",
)


def _num(value: Any) -> float | None:
    """JSON-safe float (numpy scalars, NaN and inf become None)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader may hold the file open for a moment.
            if attempt == 4:
                raise
            time.sleep(0.05)


class ProgressCallback(BaseCallback):
    """Writes training progress to `status_path` as JSON for the dashboard.

    `target_timesteps` is the absolute step count at which training ends (on resume that is
    `model.num_timesteps + timesteps`).
    """

    def __init__(self, status_path: Path, target_timesteps: int, run_name: str, num_envs: int, update_every_s: float = 2.0):
        super().__init__()
        self.status_path = Path(status_path)
        self.target_timesteps = int(target_timesteps)
        self.run_name = run_name
        self.num_envs = int(num_envs)
        self.update_every_s = update_every_s

        self.state = "starting"
        self.start_time = time.time()
        self.start_timesteps = 0
        self._next_write = 0.0
        self._next_history = 0.0
        self._rate_samples: deque[tuple[float, int]] = deque()
        self._parts: dict[int, dict[str, float]] = {}

        self.episodes = 0
        self.episodes_recent: deque[dict] = deque(maxlen=EPISODE_WINDOW)
        self.best_reward: float | None = None
        self.best_wave: float | None = None
        self.per_env: dict[int, dict] = {}
        self.history: list[dict] = []
        self.ppo_metrics: dict[str, float] = {}
        self.previous_elapsed_s = 0.0
        self._restore()

    def _restore(self) -> None:
        """Carries totals and charts over from an earlier run, so restarting doesn't wipe the dashboard."""
        try:
            old = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if old.get("run_name") != self.run_name:
            return
        self.history = [p for p in old.get("history", []) if isinstance(p, dict)]
        self.episodes = int(old.get("episodes") or 0)
        self.best_reward = _num(old.get("best_reward"))
        self.best_wave = _num(old.get("best_wave"))
        self.previous_elapsed_s = _num(old.get("elapsed_s")) or 0.0
        self.ppo_metrics = {k: v for k, v in (old.get("ppo") or {}).items() if isinstance(v, (int, float))}

    # -- SB3 hooks ---------------------------------------------------------------------------

    def _on_training_start(self) -> None:
        now = time.time()
        self.state = "running"
        self.start_time = now
        self.start_timesteps = self.num_timesteps
        if self.target_timesteps <= 0:
            self.target_timesteps = int(getattr(self.model, "_total_timesteps", 0))
        self._rate_samples.clear()
        self._rate_samples.append((now, self.num_timesteps))
        self._next_history = now + HISTORY_EVERY_S
        self._write(now)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for i, (info, done) in enumerate(zip(infos, dones)):
            parts = info.get("reward_parts")
            if parts:
                acc = self._parts.setdefault(i, {})
                for name, value in parts.items():
                    acc[name] = acc.get(name, 0.0) + value
            if done:
                self._record_episode(i, info)

        now = time.time()
        if now >= self._next_write:
            self._write(now)
        return True

    def _on_rollout_start(self) -> None:
        # PPO's train() records its metrics after rollout end; they stay in the logger until the
        # next dump, so they're readable here.
        values = getattr(self.logger, "name_to_value", {})
        for key in PPO_METRICS:
            if key in values:
                v = _num(values[key])
                if v is not None:
                    self.ppo_metrics[key.split("/", 1)[1]] = v
        self._write(time.time())  # also refreshes updated_at after the (step-free) update

    def _on_training_end(self) -> None:
        self.state = "finished"
        self._write(time.time())

    def mark_stopped(self) -> None:
        """Call when training exits without finishing (exception, Ctrl+C)."""
        if self.state == "finished":
            return
        self.state = "stopped"
        try:
            self._write(time.time())
        except Exception as exc:  # never mask the real error
            print(f"ProgressCallback: could not write status: {exc}")

    # -- internals ---------------------------------------------------------------------------

    def _record_episode(self, env_index: int, info: dict) -> None:
        ep = info.get("episode") or {}
        parts = self._parts.pop(env_index, {})

        def field(name):
            return _num(ep.get(name, info.get(name)))

        stats = {
            "reward": _num(ep.get("r")),
            "length": _num(ep.get("l")),
            "kills": field("kills"),
            "kills_per_min": field("kills_per_min"),
            "wave": field("wave"),
            "style": field("style"),
            "route_progress": field("route_progress"),
            "end_reason": info.get("end_reason"),
            "reset_seconds": _num(info.get("reset_seconds")),
            "reward_parts": parts,
        }
        self.episodes += 1
        self.episodes_recent.append(stats)
        if stats["reward"] is not None and (self.best_reward is None or stats["reward"] > self.best_reward):
            self.best_reward = stats["reward"]
        if stats["wave"] is not None and (self.best_wave is None or stats["wave"] > self.best_wave):
            self.best_wave = stats["wave"]
        self.per_env[env_index] = {
            "reward": stats["reward"],
            "length": stats["length"],
            "kills": stats["kills"],
            "wave": stats["wave"],
            "end_reason": stats["end_reason"],
            "ended_at": time.time(),
            "episodes": self.per_env.get(env_index, {}).get("episodes", 0) + 1,
        }

    def _steps_per_s(self, now: float) -> float | None:
        samples = self._rate_samples
        samples.append((now, self.num_timesteps))
        while len(samples) > 2 and now - samples[1][0] >= RATE_WINDOW_S:
            samples.popleft()
        t0, s0 = samples[0]
        dt = now - t0
        return (self.num_timesteps - s0) / dt if dt > 0.5 else None

    def _recent_mean(self, key: str) -> float | None:
        return _mean(ep[key] for ep in self.episodes_recent)

    def _add_history(self, point: dict) -> None:
        self.history.append(point)
        if len(self.history) > HISTORY_MAX:
            # Halve the resolution of the older half; recent points stay dense.
            half = len(self.history) // 2
            self.history = self.history[:half:2] + self.history[half:]

    def _snapshot(self, now: float) -> dict:
        steps_per_s = self._steps_per_s(now)
        remaining = max(0, self.target_timesteps - self.num_timesteps)
        eta = remaining / steps_per_s if steps_per_s and self.state == "running" else None

        recent = {key: self._recent_mean(key) for key in ("reward", "length", "kills", "kills_per_min", "wave", "style", "route_progress", "reset_seconds")}
        part_names = sorted({name for ep in self.episodes_recent for name in ep["reward_parts"]})
        n = len(self.episodes_recent)
        parts_mean = {name: sum(ep["reward_parts"].get(name, 0.0) for ep in self.episodes_recent) / n for name in part_names} if n else {}
        end_reasons = Counter(ep["end_reason"] for ep in self.episodes_recent if ep["end_reason"])

        if now >= self._next_history and self.state == "running":
            self._next_history = now + HISTORY_EVERY_S
            self._add_history({
                "timesteps": self.num_timesteps,
                "wall_time": now,
                "mean_reward_100": recent["reward"],
                "mean_kills_100": recent["kills"],
                "mean_kills_per_min_100": recent["kills_per_min"],
                "mean_wave_100": recent["wave"],
                "steps_per_s": steps_per_s,
            })

        envs = []
        for i in range(max(self.num_envs, len(self.per_env))):
            e = self.per_env.get(i)
            if e is None:
                envs.append({"env": i, "episodes": 0, "age_s": now - self.start_time})
            else:
                envs.append({"env": i, **{k: v for k, v in e.items() if k != "ended_at"}, "age_s": now - e["ended_at"], "last_episode_at": e["ended_at"]})

        return {
            "version": 1,
            "run_name": self.run_name,
            "state": self.state,
            "updated_at": now,
            "started_at": self.start_time,
            "elapsed_s": self.previous_elapsed_s + (now - self.start_time),
            "session_elapsed_s": now - self.start_time,
            "num_envs": self.num_envs,
            "timesteps": self.num_timesteps,
            "start_timesteps": self.start_timesteps,
            "target_timesteps": self.target_timesteps,
            "progress": self.num_timesteps / self.target_timesteps if self.target_timesteps > 0 else None,
            "steps_per_s": steps_per_s,
            "eta_s": eta,
            "episodes": self.episodes,
            "window": n,
            "mean_100": recent,
            "reward_parts_mean_100": parts_mean,
            "best_reward": self.best_reward,
            "best_wave": self.best_wave,
            "end_reasons_100": dict(end_reasons.most_common()),
            "envs": envs,
            "ppo": self.ppo_metrics,
            "history": self.history,
        }

    def _write(self, now: float) -> None:
        self._next_write = now + self.update_every_s
        try:
            write_json_atomic(self.status_path, self._snapshot(now))
        except OSError as exc:  # a status file must never stop training
            print(f"ProgressCallback: could not write {self.status_path}: {exc}")
