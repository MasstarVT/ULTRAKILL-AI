"""Live training progress for scripts/dashboard.py.

`ProgressCallback` collects episode stats during `model.learn()` and writes a small JSON status file
(atomically, at most every few seconds). The dashboard only reads that file, so it never touches the
training process.

Campaign runs also get a `campaign` block. Its completion rate and median time count fresh starts only:
a checkpoint respawn starts partway through the level, and the official timer carries over from earlier
episodes, so neither says how the agent does on a whole level.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from ultrakill_ai.campaign import CURRICULUM_VERSION, level_weights, unlock_next

EPISODE_WINDOW = 100
FRESH_WINDOW = 50  # campaign completion rate and median time are over this many fresh-start episodes
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
    # Per-dimension entropy (scripts/train.py). Once the look modes exist, total entropy stops being an
    # interpretable ent_coef signal on its own: on a step where a look mode aims, the yaw and pitch heads are
    # causally inert and only the entropy bonus acts on them. These are read next to look_free_frac.
    "train/entropy_yaw",
    "train/entropy_pitch",
    "train/entropy_look_mode",
)
# Per-episode fields carried straight into runs/<run>/episodes.jsonl. `field()` routes everything through
# _num(), which returns None for anything float() rejects, so a string checkpoint id and a [x, y, z] list have
# to bypass it -- otherwise the two fields that say where an episode died would both be written as null.
EPISODE_LOG_RAW = ("level", "start_checkpoint", "end_pos", "end_reason", "level_seconds", "gate_hops_best",
                   "bridge_resets")


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

    def __init__(self, status_path: Path, target_timesteps: int, run_name: str, num_envs: int, update_every_s: float = 2.0,
                 *, levels=(), curriculum_path=None, unlock_rate: float = 0.5, unlock_window: int = 20,
                 unlock_after_fresh_episodes: int = 0, level_weight_floor: float = 0.1):
        super().__init__()
        self.status_path = Path(status_path)
        # One JSON object per finished episode, appended. Written here rather than in the env because with
        # SubprocVecEnv five workers would interleave appends to one file, while this callback sees every
        # finished episode in the main process.
        self.episodes_path = self.status_path.with_name("episodes.jsonl")
        self._episode_log_warned = False
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
        self.best_checkpoints_level: float | None = None
        # Route gates, over fresh starts only: a respawn episode inherits both from its level load, and the
        # pre-fix run's corr(fresh_start, checkpoints_level) = -0.975 produced a "peak by depth" that was purely
        # that inheritance. best_gate_hops is a MINIMUM (lower is better), so it needs its own comparison.
        self.best_gates_reached: float | None = None
        self.best_gate_hops: float | None = None
        self.best_time: float | None = None  # fastest fresh-start completion, official level seconds
        self.fresh_recent: deque[tuple[float, float | None]] = deque(maxlen=FRESH_WINDOW)  # (completed, level_seconds)
        self.campaign = False  # set once an episode reports fresh_start
        self.per_env: dict[int, dict] = {}
        self.history: list[dict] = []
        self.ppo_metrics: dict[str, float] = {}
        self.previous_elapsed_s = 0.0

        # The multi-level curriculum. Empty `levels` is a single-level run: nothing below is written, and the
        # campaign block stays byte-identical to what it was before the curriculum existed.
        self.order: list[str] = [str(lv) for lv in levels]
        self.curriculum_path = Path(curriculum_path) if curriculum_path else None
        self.unlock_rate = float(unlock_rate)
        self.unlock_window = int(unlock_window)
        self.unlock_after_fresh_episodes = int(unlock_after_fresh_episodes)
        self.level_weight_floor = float(level_weight_floor)
        self.per_level: dict[str, dict] = {}
        self._curriculum_written: dict | None = None
        self._curriculum_warned = False
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
        self.best_checkpoints_level = _num(old.get("best_checkpoints_level"))
        self.best_gates_reached = _num(old.get("best_gates_reached"))
        self.best_gate_hops = _num(old.get("best_gate_hops"))
        if isinstance(old.get("campaign"), dict):
            self.campaign = True
            self.best_time = _num(old["campaign"].get("best_time"))
            self._restore_levels(old["campaign"].get("levels"))
        self.previous_elapsed_s = _num(old.get("elapsed_s")) or 0.0
        self.ppo_metrics = {k: v for k, v in (old.get("ppo") or {}).items() if isinstance(v, (int, float))}

    def _restore_levels(self, levels) -> None:
        """Carries the per-level table over from an earlier run of the same name.

        `unlocked` is a latch and is the whole point: without it every level re-locks on a restart and the
        levels that were already being trained stall. `best_time` and `episodes` are cumulative, so they carry
        too. The fresh-episode windows deliberately do NOT, exactly as the pooled `fresh_recent` deque does not:
        a rate carried without the episodes behind it would let `unlock_next` unlock a level on evidence that is
        no longer in the window. Every level therefore reads rate None for a while after a restart, which makes
        the sampling weights uniform until the windows refill -- wasteful but self-correcting and honest.
        """
        if not isinstance(levels, dict):
            return
        for name, old in levels.items():
            if not isinstance(old, dict):
                continue
            record = self._level_record(str(name))
            record["unlocked"] = bool(old.get("unlocked")) or record["unlocked"]
            record["best_time"] = _num(old.get("best_time"))
            record["episodes"] = int(_num(old.get("episodes")) or 0)
            # Cumulative, so it carries like `episodes` and NOT like the windows: the safety valve counts how
            # many fresh tries a level has had in total, and a restart must not hand it a clean slate or a run
            # that is stopped every few hours can never reach the valve at all.
            record["fresh_episodes"] = int(_num(old.get("fresh_episodes")) or 0)
            carried = {key: _num(old.get(key)) for key in ("checkpoints_level", "gates_reached")}
            if any(v is not None for v in carried.values()):
                record["recent"].append(carried)  # one synthetic episode, so the panel is not blank on restart

    def _level_record(self, level: str) -> dict:
        """This level's record, created (locked, unless it is the first) the first time it is asked for."""
        record = self.per_level.get(level)
        if record is None:
            record = {
                "unlocked": bool(self.order) and level == self.order[0],
                "fresh": deque(maxlen=FRESH_WINDOW),  # (completed, level_seconds) of its own fresh starts
                "recent": deque(maxlen=EPISODE_WINDOW),  # the two early-progress signals, all episodes
                "best_time": None,
                "episodes": 0,
                "fresh_episodes": 0,  # cumulative fresh starts; what `unlock_after_fresh_episodes` counts
            }
            self.per_level[level] = record
        return record

    def _level_table(self, *, with_weights: bool = False) -> dict[str, dict]:
        """The per-level stats `curriculum.json` and `status.json` both publish, for every level in `order`.

        Every level in `order` is listed even before it has produced an episode, so no reader has to invent a
        record: `campaign._level_stat` exists for readers of a table written by an older version.
        """
        table: dict[str, dict] = {}
        for level in self.order or sorted(self.per_level):
            record = self._level_record(level)
            fresh = record["fresh"]
            table[level] = {
                "unlocked": bool(record["unlocked"]),
                "fresh_window": len(fresh),
                "fresh_completion_rate": (sum(c for c, _ in fresh) / len(fresh)) if fresh else None,
                "best_time": record["best_time"],
                "episodes": int(record["episodes"]),
                "fresh_episodes": int(record["fresh_episodes"]),
            }
        if not with_weights:
            return table
        times = {level: [s for c, s in self._level_record(level)["fresh"] if c and s is not None] for level in table}
        weights = dict(level_weights(list(table), table, floor=self.level_weight_floor))
        total = sum(weights.values())
        for level, row in table.items():
            row["weight"] = (weights[level] / total) if level in weights and total else 0.0
            row["median_time_50"] = statistics.median(times[level]) if times[level] else None
            for key in ("checkpoints_level", "gates_reached"):
                row[key] = _mean(ep.get(key) for ep in self._level_record(level)["recent"])
        return table

    def write_curriculum(self) -> None:
        """Publishes the per-level table for the SubprocVecEnv workers to sample from.

        One writer (this process, in the trainer's main thread), N readers, no lock: `write_json_atomic` replaces
        the file in one operation, so a worker either reads the old whole file or the new one, and a worker that
        catches a torn or missing read keeps its last good copy. Workers never write it, which is what makes the
        lockless design correct. Skipped entirely when the run has no `levels`.
        """
        if not self.order or self.curriculum_path is None:
            return
        table = self._level_table()
        if table == self._curriculum_written:
            return
        data = {"version": CURRICULUM_VERSION, "updated_at": time.time(), "run_name": self.run_name,
                "order": list(self.order), "levels": table}
        try:
            write_json_atomic(self.curriculum_path, data)
            self._curriculum_written = table
        except OSError as exc:  # a curriculum file must never stop training; the workers fall back to levels[0]
            if not self._curriculum_warned:
                self._curriculum_warned = True
                print(f"ProgressCallback: could not write {self.curriculum_path}: {exc}")

    @property
    def unlocked_levels(self) -> list[str]:
        return [level for level in self.order if self._level_record(level)["unlocked"]]

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
        if self.order:
            self._level_record(self.order[0])["unlocked"] = True  # the ladder always has a starting rung
            self.write_curriculum()
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
            "deaths": field("deaths"),
            "firing_frac": field("firing_frac"),
            "on_target_frac": field("on_target_frac"),
            "firing_on_target_frac": field("firing_on_target_frac"),
            "enemy_visible_frac": field("enemy_visible_frac"),
            "enemy_angle_mean": field("enemy_angle_mean"),
            "enemy_dist_mean": field("enemy_dist_mean"),
            "enemy_close_frac": field("enemy_close_frac"),
            "yaw_per_step_mean": field("yaw_per_step_mean"),
            "enemy_yaw_angle_mean": field("enemy_yaw_angle_mean"),
            "enemy_pitch_err_mean": field("enemy_pitch_err_mean"),
            "yaw_track": field("yaw_track"),
            "pitch_track": field("pitch_track"),
            "pitch_abs_mean": field("pitch_abs_mean"),
            "pitch_mean": field("pitch_mean"),
            "look_up_mean": field("look_up_mean"),
            "enemy_elev_mean": field("enemy_elev_mean"),
            "enemy_elev_abs_mean": field("enemy_elev_abs_mean"),
            "enemy_elev_over15_frac": field("enemy_elev_over15_frac"),
            "wave": field("wave"),
            "style": field("style"),
            "completed": field("completed"),
            "fresh_start": field("fresh_start"),
            "level_seconds": field("level_seconds"),
            "checkpoints_level": field("checkpoints_level"),
            "cells_new": field("cells_new"),
            "oob_frac": field("oob_frac"),
            "exit_dist_min": field("exit_dist_min"),
            "gates_reached": field("gates_reached"),
            # The 2026-09-17 patience/exit-guard spec's two mechanism counters: parks this episode, and whether
            # the exit guard rejected a banished FinalPit report. Both 0 on a healthy monotone level.
            "targets_parked": field("targets_parked"),
            "exit_banished": field("exit_banished"),
            "wedged_steps": field("wedged_steps"),
            "level_started": field("level_started"),
            "look_free_frac": field("look_free_frac"),
            "look_enemy_frac": field("look_enemy_frac"),
            "look_gate_frac": field("look_gate_frac"),
            "slide_forced_frac": field("slide_forced_frac"),
            "end_reason": info.get("end_reason"),
            "reset_seconds": _num(info.get("reset_seconds")),
            "reward_parts": parts,
            # The level the episode RAN on, not the one the env is about to reset into. A string, so it goes
            # through info directly rather than through field()/_num().
            "level": info.get("level"),
        }
        self.episodes += 1
        self.episodes_recent.append(stats)
        if stats["reward"] is not None and (self.best_reward is None or stats["reward"] > self.best_reward):
            self.best_reward = stats["reward"]
        if stats["wave"] is not None and (self.best_wave is None or stats["wave"] > self.best_wave):
            self.best_wave = stats["wave"]
        if stats["checkpoints_level"] is not None and (self.best_checkpoints_level is None or stats["checkpoints_level"] > self.best_checkpoints_level):
            self.best_checkpoints_level = stats["checkpoints_level"]
        if stats["fresh_start"] is not None:
            self.campaign = True
            if stats["fresh_start"]:
                completed, seconds = stats["completed"] or 0.0, stats["level_seconds"]
                self.fresh_recent.append((completed, seconds))
                if completed and seconds is not None and (self.best_time is None or seconds < self.best_time):
                    self.best_time = seconds
                reached = stats["gates_reached"]
                if reached is not None and (self.best_gates_reached is None or reached > self.best_gates_reached):
                    self.best_gates_reached = reached
                hops = _num(info.get("gate_hops_best"))
                if hops is not None and (self.best_gate_hops is None or hops < self.best_gate_hops):
                    self.best_gate_hops = hops
        self._record_level_episode(stats)
        self.per_env[env_index] = {
            "level": stats["level"],
            "reward": stats["reward"],
            "length": stats["length"],
            "kills": stats["kills"],
            "wave": stats["wave"],
            "checkpoints_level": stats["checkpoints_level"],
            "end_reason": stats["end_reason"],
            "ended_at": time.time(),
            "episodes": self.per_env.get(env_index, {}).get("episodes", 0) + 1,
        }
        self._log_episode(env_index, info, stats)

    def _record_level_episode(self, stats: dict) -> None:
        """Folds one finished episode into its level's record, then unlocks the next level if it has been earned.

        Only a fresh start counts toward a level's completion rate: a checkpoint respawn starts partway through
        and its official timer carries over from an earlier episode, so neither says how a whole level goes.
        Unlocking is checked once per episode, is chained, and is a latch -- a level never re-locks as its rate
        falls, which would stall the learning that was in progress on it. `fresh_episodes` counts the same fresh
        starts cumulatively (the `fresh` deque only keeps the last `FRESH_WINDOW`), which is what the
        `unlock_after_fresh_episodes` safety valve reads.
        """
        level = stats.get("level")
        if not self.order or not isinstance(level, str):
            return
        record = self._level_record(level)
        record["episodes"] += 1
        record["recent"].append({"checkpoints_level": stats["checkpoints_level"], "gates_reached": stats["gates_reached"]})
        if stats["fresh_start"]:
            record["fresh_episodes"] += 1
            completed, seconds = stats["completed"] or 0.0, stats["level_seconds"]
            record["fresh"].append((completed, seconds))
            if completed and seconds is not None and (record["best_time"] is None or seconds < record["best_time"]):
                record["best_time"] = seconds
        nxt = unlock_next(self.order, self._level_table(), unlock_rate=self.unlock_rate,
                          unlock_window=self.unlock_window,
                          unlock_after_fresh_episodes=self.unlock_after_fresh_episodes)
        if nxt is not None:
            self._level_record(nxt)["unlocked"] = True

    def _log_episode(self, env_index: int, info: dict, stats: dict) -> None:
        """Appends one line to runs/<run>/episodes.jsonl: what happened, and where it ended.

        This is the only record of an individual episode. `status.json` keeps 100-episode means that a restart
        throws away, so without this a failure mode is invisible unless someone is watching live.
        """
        line = {
            "t": round(time.time(), 1),
            "env": env_index,
            "timesteps": self.num_timesteps,
            "reward": stats["reward"],
            "length": stats["length"],
            "fresh_start": stats["fresh_start"],
            "kills": stats["kills"],
            "deaths": stats["deaths"],
            "checkpoints_level": stats["checkpoints_level"],
            "gates_reached": stats["gates_reached"],
            "targets_parked": stats["targets_parked"],
            "exit_banished": stats["exit_banished"],
            "level_started": stats["level_started"],
            "wedged_steps": stats["wedged_steps"],
            "completed": stats["completed"],
        }
        line.update({name: info.get(name) for name in EPISODE_LOG_RAW})
        try:
            self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
            with self.episodes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, separators=(",", ":"), default=str) + "\n")
        except (OSError, TypeError, ValueError) as exc:  # an episode log must never stop training
            if not self._episode_log_warned:
                self._episode_log_warned = True
                print(f"ProgressCallback: could not write {self.episodes_path}: {exc}")

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

    def _fresh_mean(self, key: str) -> float | None:
        """The same mean over fresh-start episodes only: what run gates 2 and 3 are judged on."""
        return _mean(ep[key] for ep in self.episodes_recent if ep.get("fresh_start"))

    def _campaign_stats(self) -> dict:
        n = len(self.fresh_recent)
        times = [seconds for completed, seconds in self.fresh_recent if completed and seconds is not None]
        stats = {
            "fresh_window": n,
            "fresh_completion_rate": sum(completed for completed, _ in self.fresh_recent) / n if n else None,
            "median_time_50": statistics.median(times) if times else None,
            "best_time": self.best_time,
        }
        if not self.order:
            return stats  # single level: byte-identical to what this block has always been
        # Multi-level. Only `fresh_completion_rate` changes meaning, and it becomes a SCORE, not a rate: the sum
        # over unlocked levels of each one's completions shrunk by max(its window, unlock_window). That shape is
        # forced by keep_best.py, which only ever replaces best.zip on a strict improvement and is never edited
        # for this: a plain mean over unlocked levels DROPS at every unlock (~0.52 -> ~0.26 at the second level),
        # which would freeze best.zip on a single-level policy and print the "15% below best" warning forever.
        # A shrunk sum cannot drop -- a newly unlocked level contributes 0 -- and with one unlocked level at
        # window >= unlock_window it is exactly the rate this field has always been. `fresh_window` stays the
        # POOLED deque length so keep_best's MIN_FRESH_WINDOW guard keeps passing; a per-level minimum would
        # stall forever, since the slowest window to refill after a restart is the mastered level at weight 0.1.
        table = self._level_table(with_weights=True)
        contributing = [row for row in table.values() if row["unlocked"] and row["fresh_window"]]
        if contributing:
            stats["fresh_completion_rate"] = sum(
                row["fresh_completion_rate"] * row["fresh_window"] / max(row["fresh_window"], self.unlock_window)
                for row in contributing)
        stats["order"] = list(self.order)
        stats["levels"] = table
        return stats

    def _add_history(self, point: dict) -> None:
        # Keep `timesteps` strictly increasing. `_restore` carries the whole old history over, but resuming from
        # an older checkpoint rewinds num_timesteps, so the restored tail can sit ahead of the point being added
        # and the dashboard draws a line that doubles back on itself. Dropping the superseded tail keeps the run
        # that is actually continuing and discards the abandoned one, which is what the chart should show.
        steps = _num(point.get("timesteps"))
        if steps is not None:
            while self.history:
                last = _num(self.history[-1].get("timesteps"))
                if last is None or last < steps:
                    break
                self.history.pop()
        self.history.append(point)
        if len(self.history) > HISTORY_MAX:
            # Halve the resolution of the older half; recent points stay dense.
            half = len(self.history) // 2
            self.history = self.history[:half:2] + self.history[half:]

    def _snapshot(self, now: float) -> dict:
        steps_per_s = self._steps_per_s(now)
        remaining = max(0, self.target_timesteps - self.num_timesteps)
        eta = remaining / steps_per_s if steps_per_s and self.state == "running" else None

        recent = {key: self._recent_mean(key) for key in ("reward", "length", "kills", "kills_per_min", "deaths", "wave", "style", "reset_seconds",
                                                 "completed", "fresh_start", "level_seconds", "checkpoints_level", "cells_new", "oob_frac", "exit_dist_min",
                                                 "firing_frac", "on_target_frac", "firing_on_target_frac",
                                                 "enemy_visible_frac", "enemy_angle_mean", "enemy_dist_mean",
                                                 "enemy_close_frac", "yaw_per_step_mean", "enemy_yaw_angle_mean", "enemy_pitch_err_mean", "pitch_abs_mean",
                                                 "yaw_track", "pitch_track",
                                                 "pitch_mean", "look_up_mean", "enemy_elev_mean", "enemy_elev_abs_mean", "enemy_elev_over15_frac",
                                                 "gates_reached", "wedged_steps", "level_started", "slide_forced_frac",
                                                 "targets_parked", "exit_banished",
                                                 "look_free_frac", "look_enemy_frac", "look_gate_frac")}
        fresh_recent = {key: self._fresh_mean(key) for key in ("gates_reached", "checkpoints_level", "completed", "wedged_steps")}
        part_names = sorted({name for ep in self.episodes_recent for name in ep["reward_parts"]})
        n = len(self.episodes_recent)
        parts_mean = {name: sum(ep["reward_parts"].get(name, 0.0) for ep in self.episodes_recent) / n for name in part_names} if n else {}
        end_reasons = Counter(ep["end_reason"] for ep in self.episodes_recent if ep["end_reason"])
        campaign = self._campaign_stats() if self.campaign else None

        if now >= self._next_history and self.state == "running":
            self._next_history = now + HISTORY_EVERY_S
            self._add_history({
                "timesteps": self.num_timesteps,
                "wall_time": now,
                "mean_reward_100": recent["reward"],
                "mean_kills_100": recent["kills"],
                "mean_kills_per_min_100": recent["kills_per_min"],
                "mean_wave_100": recent["wave"],
                "completion_rate_fresh_50": campaign["fresh_completion_rate"] if campaign else None,
                "mean_checkpoints_level_100": recent["checkpoints_level"],
                "mean_gates_reached_100": recent["gates_reached"],
                "steps_per_s": steps_per_s,
            })

        envs = []
        for i in range(max(self.num_envs, len(self.per_env))):
            e = self.per_env.get(i)
            if e is None:
                envs.append({"env": i, "episodes": 0, "age_s": now - self.start_time})
            else:
                envs.append({"env": i, **{k: v for k, v in e.items() if k != "ended_at"}, "age_s": now - e["ended_at"], "last_episode_at": e["ended_at"]})

        status = {
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
            "mean_fresh_100": fresh_recent,
            "reward_parts_mean_100": parts_mean,
            "best_reward": self.best_reward,
            "best_wave": self.best_wave,
            "best_checkpoints_level": self.best_checkpoints_level,
            "best_gates_reached": self.best_gates_reached,
            "best_gate_hops": self.best_gate_hops,
            "end_reasons_100": dict(end_reasons.most_common()),
            "envs": envs,
            "ppo": self.ppo_metrics,
            "history": self.history,
        }
        if campaign is not None:
            status["campaign"] = campaign
        return status

    def _write(self, now: float) -> None:
        self._next_write = now + self.update_every_s
        self.write_curriculum()  # same 2 s cadence, and only when the table actually changed
        try:
            write_json_atomic(self.status_path, self._snapshot(now))
        except OSError as exc:  # a status file must never stop training
            print(f"ProgressCallback: could not write {self.status_path}: {exc}")
