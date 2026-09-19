"""Tests for ultrakill_ai.progress.ProgressCallback, the dashboard and poll_status.py, without the game.

    .venv\\Scripts\\python -m pytest tests -q     (if pytest is installed)
    .venv\\Scripts\\python tests\\test_progress.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import subprocess
import os
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai import progress as progress_mod  # noqa: E402
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402
from ultrakill_ai.spaces import ObsLayout, action_space  # noqa: E402


class FakeGrindEnv(gym.Env):
    """Random observations and rewards, random episode ends, Cyber Grind-style infos."""

    def __init__(self, seed: int = 0):
        self.observation_space = ObsLayout().space()
        self.action_space = action_space()
        self.rng = np.random.default_rng(seed)
        self.kills = 0
        self.wave = 1

    def _obs(self):
        return self.rng.uniform(-1, 1, self.observation_space.shape).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.kills, self.wave = 0, 1
        return self._obs(), {"reset_seconds": 0.5}

    def step(self, action):
        reward = float(self.rng.normal(0.0, 0.1))
        if self.rng.random() < 0.05:
            self.kills += 1
            reward += 1.0
        if self.rng.random() < 0.01:
            self.wave += 1
        terminated = bool(self.rng.random() < 0.02)
        truncated = False
        info = {
            "kills": self.kills,
            "wave": self.wave,
            "style": self.kills * 100,
            "reward_parts": {"kill": reward if reward > 0.5 else 0.0, "noise": reward},
        }
        if terminated:
            info["end_reason"] = "death" if self.rng.random() < 0.7 else "max_steps"
            info["reset_seconds"] = 0.5
        return self._obs(), reward, terminated, truncated, info


class FakeCampaignEnv(gym.Env):
    """Campaign-style infos: every other episode is a fresh start, and about half the episodes finish the level."""

    def __init__(self, seed: int, log: list[dict]):
        self.observation_space = ObsLayout().space()
        self.action_space = action_space(campaign=True)
        self.rng = np.random.default_rng(seed)
        self.log = log  # final info of every finished episode, in the order the callback records them
        self.episode = 0
        self.steps = 0
        self.fresh = 1

    def _obs(self):
        return self.rng.uniform(-1, 1, self.observation_space.shape).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.fresh = 1 if self.episode % 2 == 0 else 0
        self.episode += 1
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        terminated = bool(self.rng.random() < 0.1)
        completed = int(terminated and self.rng.random() < 0.5)
        info = {
            "kills": 0,
            "style": 0,
            "wave": 0,  # the real env keeps this Cyber Grind key in campaign mode
            "deaths": self.steps // 8,
            "completed": completed,
            "fresh_start": self.fresh,
            # The env reports the official time only for fresh-start completions. The 60 s base keeps these
            # fake times in the range a real level run can produce: a time at or under a second is what the
            # 2026-09-19 bug looked like, and the trainer now discards those (times.valid_official_seconds).
            "level_seconds": round(60.0 + self.steps * 2 / 15, 3) if completed and self.fresh else None,
            "checkpoints_level": min(3, self.steps // 4),
            "cells_new": self.steps * 2,
            "oob_frac": 0.1,  # fraction of steps with no ground under the player
            "exit_dist_min": max(0.0, 60.0 - self.steps),
            # ... and the same measure to the STANDABLE point beside the pit (mod 0.7.2). It is the
            # smaller of the two, because the pit's own transform sits far below anything walkable.
            "exit_ground_dist_min": max(0.0, 55.0 - self.steps),
            # The route gates and the wedge detector (§9.7 of the design spec).
            "gates_reached": min(4, self.steps // 3),
            "gate_hops_best": max(0, 9 - self.steps // 3) if self.steps >= 3 else None,
            # The ladder-patience and exit-guard mechanisms (2026-09-17 spec).
            "targets_parked": self.steps // 6,
            "exit_banished": int(self.steps % 7 == 0),
            # ... and the detector's verdict, which is what lets parking act at all. 1 here, so the per-level
            # row and the pooled mean both have something to carry; on a real 0-1 episode it is 0.
            "ladder_collapsed": 1,
            # The route layer (route-fallback spec): 1 is the mod's gate ladder, the layer 0-1 runs on.
            "route_source": 1,
            "route_source_name": "gates",
            "wedged_steps": 45 if self.steps % 5 == 0 else 0,
            "level_started": 1,
            "look_free_frac": 0.5,
            "look_enemy_frac": 0.2,
            "look_gate_frac": 0.3,
            "slide_forced_frac": 0.0,
            "start_checkpoint": None if self.fresh else "40,-2,414",
            # A speed stage: the target, the raw S threshold it was scaled from, and what the completion edge
            # paid. 0-1's real thresholds are [300, 240, 180, 120], so S is 120 s and the 0.75 scale (§8b)
            # makes the target 90 s.
            "target_seconds": 90.0,
            "s_rank_seconds": 120.0,
            "completion_bonus": 200.0 if completed else 0.0,
            "end_pos": [40.0, -0.5, 361.2 + self.steps],
            "reward_parts": {"time": -0.01, "novelty": 0.5, "gate": 15.0},
        }
        if terminated:
            info["end_reason"] = "level_complete" if completed else "wedged" if self.steps % 4 == 0 else "stuck"
            self.log.append(info)
        return self._obs(), 0.49, terminated, False, info


def _make(seed):
    return lambda: Monitor(FakeGrindEnv(seed), info_keywords=("kills", "wave", "style"))


def _make_campaign(seed, log):
    return lambda: Monitor(FakeCampaignEnv(seed, log), info_keywords=CAMPAIGN_INFO_KEYS)


def _load_dashboard():
    """Imports scripts/dashboard.py as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location("dashboard", ROOT / "scripts" / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(status_path: Path, timesteps: int, model=None, resume=False, env_fns=None, run_name="test_run"):
    venv = DummyVecEnv(env_fns or [_make(1), _make(2)])
    if model is None:
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=128, n_epochs=2, policy_kwargs={"net_arch": [32]}, device="cpu", verbose=0)
    else:
        model.set_env(venv)
    target = (model.num_timesteps if resume else 0) + timesteps
    cb = ProgressCallback(status_path, target, run_name, 2, update_every_s=0.2)
    model.learn(total_timesteps=timesteps, callback=cb, reset_num_timesteps=not resume)
    return model, cb


def test_progress_callback_writes_status():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "runs" / "test_run" / "status.json"
        old_every = progress_mod.HISTORY_EVERY_S
        progress_mod.HISTORY_EVERY_S = 0.5  # get some chart points in a short run
        try:
            model, cb = _train(status_path, 3000)
        finally:
            progress_mod.HISTORY_EVERY_S = old_every

        s = json.loads(status_path.read_text(encoding="utf-8"))
        assert s["state"] == "finished"
        assert s["run_name"] == "test_run"
        assert s["num_envs"] == 2
        assert s["target_timesteps"] == 3000
        assert s["timesteps"] == model.num_timesteps >= 3000
        assert 0.99 <= s["progress"] <= 1.2
        assert s["episodes"] > 10
        assert s["window"] == min(100, s["episodes"])
        assert abs(time.time() - s["updated_at"]) < 60
        m = s["mean_100"]
        assert m["reward"] is not None and m["length"] > 1
        assert m["kills"] >= 0 and m["wave"] >= 1 and m["style"] >= 0
        assert "route_progress" not in m, "route_progress is retired"
        assert m["completed"] is None and m["checkpoints_level"] is None
        assert "campaign" not in s, "a Cyber Grind run has no campaign block"
        assert s["best_checkpoints_level"] is None
        assert s["best_reward"] >= m["reward"]
        assert s["best_wave"] >= m["wave"]
        assert set(s["end_reasons_100"]) <= {"death", "max_steps"}
        assert sum(s["end_reasons_100"].values()) == s["window"]
        assert set(s["reward_parts_mean_100"]) == {"kill", "noise"}
        assert len(s["envs"]) == 2
        for e in s["envs"]:
            assert e["episodes"] > 0 and e["age_s"] >= 0 and "kills" in e
        assert "entropy_loss" in s["ppo"] and "approx_kl" in s["ppo"]
        assert s["history"], "expected chart history points"
        assert all(p["timesteps"] <= s["timesteps"] for p in s["history"])
        assert all(p["completion_rate_fresh_50"] is None for p in s["history"])
        assert not list(status_path.parent.glob("*.tmp"))

        # Resume: the target is absolute, counting on from the loaded model.
        start = model.num_timesteps
        model, cb = _train(status_path, 1000, model=model, resume=True)
        s = json.loads(status_path.read_text(encoding="utf-8"))
        assert s["target_timesteps"] == start + 1000
        assert s["start_timesteps"] == start
        assert s["timesteps"] >= start + 1000

        # A crash after finishing leaves "finished"; a fresh callback that never finished says "stopped".
        cb.mark_stopped()
        assert json.loads(status_path.read_text(encoding="utf-8"))["state"] == "finished"
        stopped = ProgressCallback(status_path, 10, "test_run", 2)
        stopped.mark_stopped()
        assert json.loads(status_path.read_text(encoding="utf-8"))["state"] == "stopped"


def test_history_is_capped():
    cb = ProgressCallback(Path(tempfile.gettempdir()) / "unused.json", 100, "x", 1)
    for i in range(progress_mod.HISTORY_MAX * 3):
        cb._add_history({"timesteps": i})
    assert len(cb.history) <= progress_mod.HISTORY_MAX
    ts = [p["timesteps"] for p in cb.history]
    assert ts == sorted(ts) and ts[-1] == progress_mod.HISTORY_MAX * 3 - 1


def test_history_drops_the_tail_a_rewound_resume_supersedes():
    """Resuming from an older checkpoint must not leave the chart's x axis running backwards.

    `_restore` carries the whole old history over, but a resume from an older checkpoint rewinds
    num_timesteps, so the restored tail sits ahead of the points that follow it and the dashboard draws a line
    that doubles back. This happened for real on campaign_ppo_ground (2,691,640 -> 2,654,560).
    """
    cb = ProgressCallback(Path(tempfile.gettempdir()) / "unused.json", 100, "x", 1)
    for step in (100, 200, 300, 400):
        cb._add_history({"timesteps": step, "mean_reward_100": step / 10.0})
    cb._add_history({"timesteps": 250, "mean_reward_100": 99.0})  # resumed from the 250 checkpoint
    ts = [p["timesteps"] for p in cb.history]
    assert ts == [100, 200, 250], f"superseded points should be dropped, got {ts}"
    assert ts == sorted(ts)
    cb._add_history({"timesteps": 300, "mean_reward_100": 1.0})
    assert [p["timesteps"] for p in cb.history] == [100, 200, 250, 300]
    # The surviving point at 300 is the one from the continuing run, not the abandoned one.
    assert cb.history[-1]["mean_reward_100"] == 1.0


def test_dashboard_renders_a_history_that_already_rewound():
    """The dashboard is a viewer: it must cope with a status.json written before the callback fix."""
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        _train(status_path, 600)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        history = status.get("history") or []
        if len(history) >= 2:  # splice a rewind in, the way a resume from an older checkpoint leaves one
            rewound = dict(history[-1])
            rewound["timesteps"] = int(history[0].get("timesteps") or 0)
            history.append(rewound)
        status["history"] = history
        status_path.write_text(json.dumps(status), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_dashboard_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        _train(status_path, 600)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_campaign_progress():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "runs" / "campaign_run" / "status.json"
        log: list[dict] = []
        old_every = progress_mod.HISTORY_EVERY_S
        progress_mod.HISTORY_EVERY_S = 0.5
        try:
            _train(status_path, 3000, env_fns=[_make_campaign(1, log), _make_campaign(2, log)], run_name="campaign_run")
        finally:
            progress_mod.HISTORY_EVERY_S = old_every

        s = json.loads(status_path.read_text(encoding="utf-8"))
        assert "campaign" in s, "expected a campaign block once episodes report fresh_start"
        assert s["episodes"] == len(log)
        fresh = [ep for ep in log if ep["fresh_start"]]
        window = fresh[-progress_mod.FRESH_WINDOW:]
        fresh_times = [ep["level_seconds"] for ep in fresh if ep["completed"]]
        assert len(fresh) > progress_mod.FRESH_WINDOW and fresh_times, "the fake run is too short"

        c = s["campaign"]
        assert set(c) == {"fresh_window", "fresh_completion_rate", "median_time_50", "best_time",
                          "target_seconds", "s_rank_seconds"}, \
            "a single-level run's campaign block, plus the target and the S threshold behind it"
        # The target is what the env computed live (0.75 x the level's own S-rank time), carried straight
        # through: it is how campaign_driver learns the number at all -- the driver only ever reads files.
        assert c["target_seconds"] == 90.0 and c["s_rank_seconds"] == 120.0
        assert c["fresh_window"] == progress_mod.FRESH_WINDOW
        assert 0.0 <= c["fresh_completion_rate"] <= 1.0
        assert abs(c["fresh_completion_rate"] - sum(ep["completed"] for ep in window) / len(window)) < 1e-9
        assert c["best_time"] == min(fresh_times)
        assert c["median_time_50"] == statistics.median(ep["level_seconds"] for ep in window if ep["completed"])
        assert s["best_checkpoints_level"] == max(ep["checkpoints_level"] for ep in log)
        # The route bests are over fresh starts only: a respawn episode inherits both from its level load.
        assert s["best_gates_reached"] == max(ep["gates_reached"] for ep in fresh)
        assert s["best_gate_hops"] == min(ep["gate_hops_best"] for ep in fresh if ep["gate_hops_best"] is not None)

        m = s["mean_100"]
        assert "route_progress" not in m
        for key in ("completed", "fresh_start", "level_seconds", "checkpoints_level", "cells_new", "exit_dist_min",
                    "exit_ground_dist_min",
                    "gates_reached", "wedged_steps", "level_started", "look_gate_frac", "slide_forced_frac",
                    "targets_parked", "exit_banished", "route_source", "ladder_collapsed"):
            assert m[key] is not None, key
        assert m["ladder_collapsed"] == 1.0, "the whole window ran on a level whose ladder is collapsed"
        assert m["route_source"] == 1.0, "the whole window ran on the gate ladder, so the mean is the layer"
        assert s["mean_fresh_100"]["gates_reached"] is not None
        assert set(s["end_reasons_100"]) <= {"level_complete", "stuck", "wedged"}
        assert set(s["reward_parts_mean_100"]) == {"time", "novelty", "gate"}
        for e in s["envs"]:
            assert e["episodes"] > 0 and e["checkpoints_level"] is not None
        assert s["history"], "expected chart history points"
        assert all("completion_rate_fresh_50" in p and "mean_checkpoints_level_100" in p for p in s["history"])
        assert all("mean_gates_reached_100" in p for p in s["history"])  # dashboard.py reads it by this name
        assert all(p["mean_route_source_100"] == 1.0 for p in s["history"]), "the route layer is charted too"
        rates = [p["completion_rate_fresh_50"] for p in s["history"] if p["completion_rate_fresh_50"] is not None]
        assert rates and all(0.0 <= r <= 1.0 for r in rates)

        # One JSON line per finished episode, with the fields status.json's means throw away.
        lines = [json.loads(line) for line in (status_path.parent / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(lines) == s["episodes"]
        for entry in lines:
            for key in ("t", "env", "timesteps", "reward", "length", "fresh_start", "start_checkpoint", "end_reason",
                        "kills", "deaths", "checkpoints_level", "gates_reached", "gate_hops_best", "level_started",
                        "wedged_steps", "end_pos", "level_seconds", "completed", "targets_parked", "exit_banished",
                        "ladder_collapsed", "route_source", "route_source_name"):
                assert key in entry, key
            # The integer is charted; the string is the readable half, and it has to bypass _num() to survive.
            assert entry["route_source"] == 1 and entry["route_source_name"] == "gates"
            assert isinstance(entry["end_pos"], list) and len(entry["end_pos"]) == 3
            assert all(isinstance(v, float) for v in entry["end_pos"])
            assert entry["end_reason"] in ("level_complete", "stuck", "wedged")
        respawned = [e for e in lines if not e["fresh_start"]]
        assert respawned and all(e["start_checkpoint"] == "40,-2,414" for e in respawned)
        assert any(e["gate_hops_best"] is not None for e in lines)

        # A restart keeps the best time, the best checkpoint count and both route bests; the 50-episode window
        # starts empty.
        restored = ProgressCallback(status_path, 10, "campaign_run", 2)
        restored.mark_stopped()
        r = json.loads(status_path.read_text(encoding="utf-8"))
        assert r["campaign"]["best_time"] == c["best_time"]
        assert r["campaign"]["fresh_window"] == 0 and r["campaign"]["fresh_completion_rate"] is None
        assert r["campaign"]["median_time_50"] is None
        assert r["best_checkpoints_level"] == s["best_checkpoints_level"]
        assert r["best_gates_reached"] == s["best_gates_reached"]
        assert r["best_gate_hops"] == s["best_gate_hops"], "a minimum must survive a resume too"


# ---------------------------------------------------------------------------------------------
# The multi-level curriculum
# ---------------------------------------------------------------------------------------------

CURRICULUM_LEVELS = ["Level 0-1", "Level 0-3", "Level 0-4"]


def episode_info(level, *, fresh=1, completed=0, seconds=None, checkpoints=2, gates=3, hops=1):
    """One finished campaign episode as the env reports it."""
    return {
        "episode": {"r": 1.0, "l": 100.0}, "level": level, "kills": 0, "deaths": 0, "wave": 0, "style": 0,
        "completed": completed, "fresh_start": fresh, "level_seconds": seconds,
        "checkpoints_level": checkpoints, "cells_new": 10, "oob_frac": 0.0, "exit_dist_min": 5.0,
        "exit_ground_dist_min": 3.0,
        "gates_reached": gates, "gate_hops_best": hops, "wedged_steps": 0, "level_started": 1,
        "look_free_frac": 0.5, "look_enemy_frac": 0.2, "look_gate_frac": 0.3, "slide_forced_frac": 0.0,
        "start_checkpoint": None, "end_pos": [0.0, 1.0, 2.0],
        "end_reason": "level_complete" if completed else "stuck", "reward_parts": {"gate": 15.0},
    }


def curriculum_callback(tmp: Path, **overrides) -> ProgressCallback:
    run = tmp / "runs" / "campaign_multi"
    cb = ProgressCallback(run / "status.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                          levels=CURRICULUM_LEVELS, curriculum_path=run / "curriculum.json", **overrides)
    cb._on_training_start()
    return cb


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Official times the game never reported (2026-09-19)
# ---------------------------------------------------------------------------------------------


def test_an_official_time_the_game_never_reported_is_not_a_record():
    """A completion frame read after the level stats reset gave 0.0 s on a 4,120-decision episode.

    It is still a COMPLETION -- the rate must not move -- but it is not a time: not the best, not in the
    median window (which is a speed stage's promotion gate), not anywhere a real run could be measured against.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "spec_0-2_speed"
        cb = ProgressCallback(run / "status.json", 1000, "spec_0-2_speed", 1, update_every_s=0.0)
        cb._on_training_start()
        for seconds in (150.0, 200.0, 250.0):
            cb._record_episode(0, episode_info("Level 0-2", completed=1, seconds=seconds))
        cb._record_episode(0, episode_info("Level 0-2", completed=1, seconds=0.0))
        cb._write(time.time())
        c = read_json(run / "status.json")["campaign"]
        assert c["best_time"] == 150.0, "0.0 is not the fastest completion, it is no completion time at all"
        assert c["median_time_50"] == 200.0, "the median is over the three real times, not four"
        assert c["fresh_window"] == 4 and c["fresh_completion_rate"] == 1.0, "the completion itself still counts"


def test_a_poisoned_best_time_heals_when_the_trainer_restarts():
    """`_restore` carries `best_time` across every bounce, so a 0.0 left in status.json is forever otherwise."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "spec_0-2_speed"
        run.mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({
            "run_name": "spec_0-2_speed", "episodes": 312, "best_reward": 531.99,
            "campaign": {"fresh_window": 50, "fresh_completion_rate": 0.86, "median_time_50": 0.0,
                         "best_time": 0.0, "target_seconds": 120.0, "s_rank_seconds": 120.0,
                         "levels": {"Level 0-2": {"unlocked": True, "best_time": 0.0, "episodes": 312,
                                                  "fresh_episodes": 120}}}}), encoding="utf-8")
        cb = ProgressCallback(run / "status.json", 1000, "spec_0-2_speed", 1, update_every_s=0.0,
                              levels=["Level 0-2"], curriculum_path=run / "curriculum.json")
        assert cb.best_time is None, "the poisoned lifetime minimum must not be restored"
        assert cb._level_record("Level 0-2")["best_time"] is None
        assert cb.episodes == 312 and cb._level_record("Level 0-2")["unlocked"] is True, "everything else carries"
        assert cb.target_seconds == 120.0, "and so does the speed target"
        cb._on_training_start()
        cb._record_episode(0, episode_info("Level 0-2", completed=1, seconds=123.5368))
        cb._write(time.time())
        c = read_json(run / "status.json")["campaign"]
        assert c["best_time"] == 123.5368, "the first real completion after the bounce IS the best again"
        assert c["levels"]["Level 0-2"]["best_time"] == 123.5368


def test_curriculum_file_is_written_before_any_episode_and_lists_every_level():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp))
        data = read_json(run / "curriculum.json")
        assert data["run_name"] == "campaign_multi" and data["order"] == CURRICULUM_LEVELS
        assert list(data["levels"]) == CURRICULUM_LEVELS, "every level is listed before it has any episodes"
        assert data["levels"]["Level 0-1"]["unlocked"] is True, "the ladder always has a starting rung"
        assert [data["levels"][lv]["unlocked"] for lv in CURRICULUM_LEVELS[1:]] == [False, False]
        assert data["levels"]["Level 0-3"] == {
            "unlocked": False, "fresh_window": 0, "fresh_completion_rate": None, "best_time": None,
            "episodes": 0, "fresh_episodes": 0,
            # The learning-progress fields, all empty before the level has run: additive, so a reader from
            # before they existed (an older env.py on a resumed run) ignores them and samples as it always did.
            "progress_fast": None, "progress_slow": None, "progress_best": None, "progress_samples": 0,
            "gates_total": None, "fresh_completions": 0, "dry_fresh_episodes": 0}
        assert cb.unlocked_levels == ["Level 0-1"]
        assert not list((run).glob("*.tmp"))


def test_a_level_unlocks_the_next_one_and_the_score_never_drops():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp))
        for i in range(19):  # 19 fresh completions: one short of the window, so nothing unlocks yet
            cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=100.0 + i))
        cb._write(time.time())
        assert cb.unlocked_levels == ["Level 0-1"], "19 fresh episodes is not evidence"
        cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=90.0))
        cb._write(time.time())
        assert cb.unlocked_levels == ["Level 0-1", "Level 0-3"]
        assert read_json(run / "curriculum.json")["levels"]["Level 0-3"]["unlocked"] is True

        s = read_json(run / "status.json")["campaign"]
        assert s["order"] == CURRICULUM_LEVELS
        assert abs(s["fresh_completion_rate"] - 1.0) < 1e-9, "one unlocked level at the window IS today's rate"
        assert s["best_time"] == 90.0 and s["fresh_window"] == 20
        table = s["levels"]
        assert table["Level 0-1"]["fresh_completion_rate"] == 1.0 and table["Level 0-1"]["best_time"] == 90.0
        assert table["Level 0-1"]["episodes"] == 20 and table["Level 0-1"]["checkpoints_level"] == 2.0
        assert table["Level 0-1"]["gates_reached"] == 3.0
        # A mastered level falls to the floor weight; the newly unlocked one takes the rest.
        assert abs(table["Level 0-1"]["weight"] - 0.1 / 1.1) < 1e-9
        assert abs(table["Level 0-3"]["weight"] - 1.0 / 1.1) < 1e-9
        assert table["Level 0-4"]["weight"] == 0.0

        # The unlock itself, and then a level that completes nothing, must never lower the headline: keep_best
        # only ever replaces best.zip on a strict improvement, so a score that drops would freeze it forever.
        before = s["fresh_completion_rate"]
        for _ in range(20):
            cb._record_episode(1, episode_info("Level 0-3"))
            cb._write(time.time())
            now = read_json(run / "status.json")["campaign"]["fresh_completion_rate"]
            assert now >= before - 1e-9, (before, now)
        for i in range(10):
            cb._record_episode(1, episode_info("Level 0-3", completed=1, seconds=200.0 + i))
        cb._write(time.time())
        s = read_json(run / "status.json")["campaign"]
        # A SCORE, not a rate: each level contributes completions / max(its window, unlock_window). 0-1 is
        # 20/20 = 1.0 and 0-3 is now 10 completions over 30 fresh episodes.
        assert abs(s["fresh_completion_rate"] - (1.0 + 10 / 30)) < 1e-9, s["fresh_completion_rate"]
        assert s["fresh_completion_rate"] > 1.0, "it can exceed 1.0, which is why the dashboard calls it a score"
        assert s["best_time"] == 90.0, "the global minimum, as it has always been"
        assert s["fresh_window"] == progress_mod.FRESH_WINDOW, "the POOLED deque, so keep_best's guard keeps passing"


def test_the_unlock_latch_survives_a_restart_and_a_falling_rate():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp))
        for i in range(20):
            cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=120.0 + i))
        cb._write(time.time())
        assert cb.unlocked_levels == ["Level 0-1", "Level 0-3"]

        again = ProgressCallback(run / "status.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                                 levels=CURRICULUM_LEVELS, curriculum_path=run / "curriculum.json")
        assert again.unlocked_levels == ["Level 0-1", "Level 0-3"], "without the latch every level re-locks"
        again._on_training_start()
        table = read_json(run / "status.json")["campaign"]["levels"]
        assert table["Level 0-1"]["best_time"] == 120.0 and table["Level 0-1"]["episodes"] == 20
        assert table["Level 0-1"]["fresh_window"] == 0, "the windows start empty, exactly as the pooled deque does"
        assert table["Level 0-1"]["checkpoints_level"] == 2.0, "the early-progress signals are carried"
        # A collapsing rate must not re-lock 0-3 and stall the learning in progress on it.
        for _ in range(25):
            again._record_episode(0, episode_info("Level 0-1"))
        again._write(time.time())
        assert again.unlocked_levels == ["Level 0-1", "Level 0-3"]
        assert read_json(run / "status.json")["campaign"]["levels"]["Level 0-1"]["fresh_completion_rate"] == 0.0


def test_the_safety_valve_opens_the_next_level_and_survives_a_restart():
    """`unlock_after_fresh_episodes` through the callback, including across a stop: the counter is cumulative.

    The realistic shape, and the reason the counter is not the `fresh` deque: a level is run 12 times with a 0.0
    rate, the trainer is restarted (which empties every window), and the remaining tries still add up to the
    valve. Without carrying `fresh_episodes` a run stopped every few hours could never reach it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp), unlock_after_fresh_episodes=20)
        for _ in range(12):
            cb._record_episode(0, episode_info("Level 0-1"))  # fresh, never completed
        cb._write(time.time())
        assert cb.unlocked_levels == ["Level 0-1"], "12 tries is not 20"
        assert read_json(run / "status.json")["campaign"]["levels"]["Level 0-1"]["fresh_episodes"] == 12

        again = ProgressCallback(run / "status.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                                 levels=CURRICULUM_LEVELS, curriculum_path=run / "curriculum.json",
                                 unlock_after_fresh_episodes=20)
        again._on_training_start()
        assert again._level_record("Level 0-1")["fresh_episodes"] == 12, "cumulative, unlike the windows"
        for _ in range(7):
            again._record_episode(0, episode_info("Level 0-1"))
        assert again.unlocked_levels == ["Level 0-1"], "19"
        again._record_episode(0, episode_info("Level 0-1"))
        assert again.unlocked_levels == ["Level 0-1", "Level 0-3"], "the 20th fresh try opens the valve"
        again._write(time.time())
        assert read_json(run / "curriculum.json")["levels"]["Level 0-3"]["unlocked"] is True


def test_the_safety_valve_is_off_by_default():
    """Every run written before this existed must behave exactly as it did: no valve, no unlock."""
    with tempfile.TemporaryDirectory() as tmp:
        cb = curriculum_callback(Path(tmp))
        assert cb.unlock_after_fresh_episodes == 0
        for _ in range(200):
            cb._record_episode(0, episode_info("Level 0-1"))
        assert cb.unlocked_levels == ["Level 0-1"]


def test_inserting_a_level_before_an_unlocked_one_keeps_it_unlocked():
    """The 2026-09-17 pause's own move: 0-1 and 0-3 are unlocked, then 0-2 is inserted between them.

    `_restore_levels` latches by NAME, so reordering cannot re-lock anything, and `_level_table` republishes the
    new order with both old levels still open. The inserted level is locked and waits its turn on 0-1's rate.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp))
        for i in range(20):
            cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=120.0 + i))
        cb._record_episode(1, episode_info("Level 0-3"))
        cb._write(time.time())
        assert cb.unlocked_levels == ["Level 0-1", "Level 0-3"]

        reordered = ["Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4"]
        again = ProgressCallback(run / "status.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                                 levels=reordered, curriculum_path=run / "curriculum.json")
        again._on_training_start()
        assert again.unlocked_levels == ["Level 0-1", "Level 0-3"], "0-3 must not re-lock behind the new 0-2"
        table = read_json(run / "curriculum.json")
        assert table["order"] == reordered
        assert [lv for lv, row in table["levels"].items() if row["unlocked"]] == ["Level 0-1", "Level 0-3"]
        # 0-1's window restarts empty, so 0-2 unlocks once the rate is re-earned -- not before.
        for i in range(19):
            again._record_episode(0, episode_info("Level 0-1", completed=1, seconds=200.0 + i))
        assert again.unlocked_levels == ["Level 0-1", "Level 0-3"], "19 fresh episodes is not the window"
        again._record_episode(0, episode_info("Level 0-1", completed=1, seconds=199.0))
        assert again.unlocked_levels == ["Level 0-1", "Level 0-2", "Level 0-3"]


def test_the_curriculum_publishes_the_learning_progress_statistics():
    """Per level: the two averages, the samples behind them, the ladder length and the dry-episode counter.

    curriculum.json carries the raw statistics because the SubprocVecEnv workers weight their own draws from it;
    status.json adds the two derived numbers (`progress_score`, `learning_progress`) because the dashboard reads
    those and nothing else does.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp), curriculum_weighting="progress")
        for _ in range(10):  # 2 of 10 rungs, nothing finished
            cb._record_episode(0, episode_info("Level 0-1", gates=2, hops=8, checkpoints=1))
        for _ in range(15):  # then it starts finishing the level
            cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=150.0, gates=10, hops=0))
        cb._write(time.time())
        row = read_json(run / "curriculum.json")["levels"]["Level 0-1"]
        assert row["gates_total"] == 10, "gates_reached + gate_hops_best, ratcheted to the ladder's own length"
        assert row["progress_samples"] == 25 and row["fresh_completions"] == 15
        assert row["dry_fresh_episodes"] == 0, "it completed the level on its last fresh try"
        assert row["progress_fast"] > row["progress_slow"], "the fast average leads a level that is improving"
        assert 0.0 <= row["progress_slow"] <= row["progress_fast"] <= 1.0

        published = read_json(run / "status.json")["campaign"]["levels"]["Level 0-1"]
        assert abs(published["progress_score"] - row["progress_fast"]) < 1e-9, "the score IS the fast average"
        assert abs(published["learning_progress"] - (row["progress_fast"] - row["progress_slow"])) < 1e-9
        assert published["weight"] > 0.0


def test_a_blocked_level_stops_taking_the_fresh_starts():
    """The live run's bug, end to end: 0-3 blocked at 0 completions while 0-1 completes and improves.

    Under today's rule 0-3 would take the larger share of every fresh draw for having the worse rate. Under
    `curriculum_weighting: "progress"` it keeps the retention floor and 0-1 gets the rest.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp), curriculum_weighting="progress")
        cb._level_record("Level 0-3")["unlocked"] = True
        for i in range(40):  # 0-1: climbing, and finishing the level more and more often
            done = 1 if i >= 20 else 0
            cb._record_episode(0, episode_info("Level 0-1", completed=done, seconds=160.0 - i if done else None,
                                               gates=10 if done else 4, hops=0 if done else 6))
        # 0-3, the shape the live level actually had: it got a good way down the ladder for a while, then slid
        # back and stayed there, and never once finished. 140 fresh tries, no completion.
        for i in range(140):
            gates = 7 if i < 50 else 3 + (i % 3)
            cb._record_episode(1, episode_info("Level 0-3", gates=gates, hops=10 - gates, checkpoints=0))
        cb._write(time.time())
        table = read_json(run / "status.json")["campaign"]["levels"]
        assert table["Level 0-3"]["dry_fresh_episodes"] == 140
        assert abs(table["Level 0-3"]["weight"] - 0.1) < 1e-9, "the floor, not 50% of the run"
        assert table["Level 0-1"]["weight"] > 0.85
        assert table["Level 0-1"]["fresh_completion_rate"] > table["Level 0-3"]["fresh_completion_rate"]

        # The same table under today's rule: the blocked level takes MORE than the level that is learning.
        inverse = ProgressCallback(run / "status_inverse.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                                   levels=CURRICULUM_LEVELS, curriculum_path=run / "curriculum_inverse.json")
        inverse.per_level = cb.per_level
        inverse.order = cb.order
        weights = inverse._level_table(with_weights=True)
        assert weights["Level 0-3"]["weight"] > weights["Level 0-1"]["weight"]


def test_the_progress_statistics_survive_a_restart():
    """They have to: the supervisor bounces this trainer, and the windows deliberately do not carry.

    Without carrying the two averages every restart would drop the rule back to `inverse_rate` for the next 20
    fresh episodes per level -- which is the starvation this change exists to stop, handed back every few hours.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp), curriculum_weighting="progress")
        for i in range(30):
            cb._record_episode(0, episode_info("Level 0-1", completed=1 if i > 10 else 0,
                                               seconds=150.0 if i > 10 else None,
                                               gates=10 if i > 10 else 3, hops=0 if i > 10 else 7))
        cb._write(time.time())
        before = read_json(run / "curriculum.json")["levels"]["Level 0-1"]

        again = ProgressCallback(run / "status.json", 1000, "campaign_multi", 2, update_every_s=0.0,
                                 levels=CURRICULUM_LEVELS, curriculum_path=run / "curriculum.json",
                                 curriculum_weighting="progress")
        again._on_training_start()
        after = read_json(run / "curriculum.json")["levels"]["Level 0-1"]
        for key in ("progress_fast", "progress_slow", "progress_samples", "gates_total",
                    "fresh_completions", "dry_fresh_episodes"):
            assert after[key] == before[key], key
        assert after["fresh_window"] == 0, "the windows still start empty, exactly as they always have"


def test_the_weighting_rule_defaults_to_the_one_that_is_running():
    """Only the full-campaign config opts in, so every other run's draws are byte for byte what they were."""
    with tempfile.TemporaryDirectory() as tmp:
        cb = curriculum_callback(Path(tmp))
        assert cb.curriculum_weighting == "inverse_rate"
        cb._level_record("Level 0-3")["unlocked"] = True
        for _ in range(30):
            cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=120.0))
        for _ in range(30):
            cb._record_episode(1, episode_info("Level 0-3"))
        table = cb._level_table(with_weights=True)
        assert abs(table["Level 0-1"]["weight"] - 0.1 / 1.1) < 1e-9, "mastered: the floor weight, as before"
        assert abs(table["Level 0-3"]["weight"] - 1.0 / 1.1) < 1e-9, "rate 0.0: the whole rest, as before"
        assert table["Level 0-1"]["progress_score"] is not None, "the statistics are kept either way"


def test_replay_curriculum_scores_both_rules_from_an_episode_log():
    """scripts/replay_curriculum.py: the evidence tool, and read-only -- it must not touch the run it reads.

    Two levels in the log: one completing and improving, one that got a good way in and then stalled for 140
    fresh tries. The replay has to say that today's rule hands the stalled one the bigger share and the new one
    hands it the smaller, which is the whole claim the rule change rests on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "episodes.jsonl"
        rows = []
        clock = 1_000_000.0
        for i in range(60):
            done = i >= 20
            rows.append({"t": clock + i, "env": 0, "level": "Level 0-1", "fresh_start": 1,
                         "completed": 1 if done else 0, "level_seconds": 150.0 if done else None,
                         "gates_reached": 10 if done else 4, "gate_hops_best": 0 if done else 6,
                         "checkpoints_level": 4, "reward": 1.0, "length": 100.0})
        for i in range(250):
            gates = 7 if i < 50 else 3 + (i % 3)
            rows.append({"t": clock + 1000 + i * 30, "env": 1, "level": "Level 0-3", "fresh_start": 1,
                         "completed": 0, "level_seconds": None, "gates_reached": gates,
                         "gate_hops_best": 10 - gates, "checkpoints_level": 0, "reward": 1.0, "length": 100.0})
        rows.sort(key=lambda r: r["t"])
        log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        before = log.read_bytes()

        # The script refuses to start when system commit is over 85%, because offline analysis has twice taken
        # this machine down beside a live run. A 60-row fixture is not that hazard, so the test overrides it
        # the same documented way an operator would.
        env = dict(os.environ, ULTRAKILL_AI_IGNORE_COMMIT="1")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "replay_curriculum.py"), "--episodes", str(log),
             "--hours", "0.5", "--drift"],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT), env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert log.read_bytes() == before, "the replay must never write to the log it reads"
        assert [p.name for p in Path(tmp).glob("*")] == ["episodes.jsonl"], "and it writes nothing beside it"
        # hour, level, fresh%, inverse_rate, progress, prog, blocked, dry
        rows_out = {line.split()[1]: line.split() for line in result.stdout.splitlines()
                    if line.strip().startswith("-0h")}
        assert set(rows_out) == {"0-1", "0-3"}, result.stdout
        inverse = {lv: float(cols[3].rstrip("%")) for lv, cols in rows_out.items()}
        progress = {lv: float(cols[4].rstrip("%")) for lv, cols in rows_out.items()}
        assert inverse["0-3"] > inverse["0-1"], "today's rule: the stalled level takes the bigger share"
        assert progress["0-3"] < progress["0-1"], "learning progress: it takes the smaller one"
        assert progress["0-3"] <= 11.0, "and that share is the retention floor"
        assert rows_out["0-3"][6] == "100%", "blocked for the whole hour"
        assert "signed drift" in result.stdout


def test_a_run_with_no_levels_writes_no_curriculum_file():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "single"
        cb = ProgressCallback(run / "status.json", 1000, "single", 1, update_every_s=0.0,
                              curriculum_path=run / "curriculum.json")
        cb._on_training_start()
        cb._record_episode(0, episode_info("Level 0-1", completed=1, seconds=100.0))
        cb._write(time.time())
        assert not (run / "curriculum.json").exists()
        s = read_json(run / "status.json")["campaign"]
        assert set(s) == {"fresh_window", "fresh_completion_rate", "median_time_50", "best_time",
                          "target_seconds", "s_rank_seconds"}
        assert s["target_seconds"] is None and s["s_rank_seconds"] is None, \
            "None until a speed stage's env reports them"


def test_episodes_jsonl_carries_the_level_it_ran_on():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "runs" / "campaign_multi"
        cb = curriculum_callback(Path(tmp))
        cb._record_episode(0, episode_info("Level 0-1"))
        cb._record_episode(1, episode_info("Level 0-3", completed=1, seconds=42.0))
        lines = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [line["level"] for line in lines] == ["Level 0-1", "Level 0-3"]
        assert "level" in progress_mod.EPISODE_LOG_RAW, "a string field must bypass _num()"
        cb._write(time.time())
        envs = read_json(run / "status.json")["envs"]
        assert [e.get("level") for e in envs] == ["Level 0-1", "Level 0-3"]


def test_dashboard_renders_the_per_level_block():
    dashboard = _load_dashboard()
    campaign = {
        "fresh_window": 50, "fresh_completion_rate": 1.34, "median_time_50": 152.0, "best_time": 141.2,
        "order": CURRICULUM_LEVELS,
        "levels": {
            # 0-1's ladder is monotone, so `col 0` and `pk 0.0`; 0-3's collapses, so it is allowed to park.
            "Level 0-1": {"unlocked": True, "fresh_window": 50, "fresh_completion_rate": 0.62, "best_time": 141.2,
                          "episodes": 812, "weight": 0.38, "checkpoints_level": 4.1, "gates_reached": 3.2,
                          "ladder_collapsed": 0.0, "targets_parked": 0.0, "progress_score": 0.66,
                          "learning_progress": 0.063},
            "Level 0-3": {"unlocked": True, "fresh_window": 23, "fresh_completion_rate": 0.13, "best_time": None,
                          "episodes": 188, "weight": 0.62, "checkpoints_level": 1.0, "gates_reached": 0.4,
                          "ladder_collapsed": 1.0, "targets_parked": 8.3, "progress_score": None,
                          "learning_progress": None},
            "Level 0-4": {"unlocked": False, "fresh_window": 0, "fresh_completion_rate": None, "best_time": None,
                          "episodes": 0, "weight": 0.0, "checkpoints_level": None, "gates_reached": None},
        },
    }
    lines = dashboard.campaign_lines(campaign, {"completed": 0.4, "checkpoints_level": 2.14})
    assert lines[0] == "  fresh score      1.34 / 2 levels", lines[0]
    assert "  levels" in lines
    block = lines[lines.index("  levels") + 1:]
    # `w` is the share of fresh draws and `prog` the progress score the weighting rule reads (a dash while the
    # statistics are cold, which is also what an older status.json shows).
    assert block[:2] == [
        "    0-1  fresh 62% (50)  best 02:21.200  cp 4.1  w 0.38  prog 0.66  col 0  pk 0.0",
        "    0-3  fresh 13% (23)  best —  cp 1.0  w 0.62  prog —  col 1  pk 8.3",
    ], block
    assert not any("0-4" in line for line in block), "a locked level has no rows to show"
    # A single-level run keeps the old headline and grows no block at all.
    single = dashboard.campaign_lines({"fresh_window": 50, "fresh_completion_rate": 0.62}, {})
    assert single[0] == "  fresh completed  62% of last 50"
    assert "  levels" not in single

    # The whole window against a real multi-level status, so the games table's level column renders too.
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        cb = curriculum_callback(Path(tmp) / "unused")
        for i in range(25):
            cb._record_episode(i % 2, episode_info(CURRICULUM_LEVELS[i % 2], completed=i % 3 == 0, seconds=90.0 + i))
        cb.status_path = status_path
        cb._write(time.time())
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["campaign"]["levels"] and any(e.get("level") for e in status["envs"])
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_poll_status_logs_the_unlocked_level_count():
    dashboard_free = {
        "state": "running", "timesteps": 10, "episodes": 5, "window": 5, "steps_per_s": 100.0,
        "mean_100": {"reward": 1.0}, "campaign": {"fresh_window": 50, "fresh_completion_rate": 1.34,
                                                  "levels": {"a": {"unlocked": True}, "b": {"unlocked": True},
                                                             "c": {"unlocked": False}}},
        "reward_parts_mean_100": {"item_pickup": 15.0, "item_placed": 15.0},
    }
    spec = importlib.util.spec_from_file_location("poll_status", ROOT / "scripts" / "poll_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = module.row(dashboard_free)
    assert row["levels_unlocked"] == 2
    assert row["part_item_pickup"] == 15.0 and row["part_item_placed"] == 15.0
    assert module.row({"mean_100": {}})["levels_unlocked"] is None, "a single-level run has no such column value"


def test_episode_log_write_failure_does_not_stop_training():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "runs" / "broken" / "status.json"
        cb = ProgressCallback(status_path, 10, "broken", 1)
        cb.episodes_path = Path(tmp) / "runs" / "broken"  # a directory: every open() for append raises
        cb.episodes_path.mkdir(parents=True, exist_ok=True)
        cb._record_episode(0, {"episode": {"r": 1.0, "l": 2}, "kills": 0, "end_reason": "stuck"})
        assert cb.episodes == 1  # the episode was still recorded


def test_dashboard_campaign_panel():
    dashboard = _load_dashboard()
    lines = dashboard.campaign_lines(
        {"fresh_window": 50, "fresh_completion_rate": 0.62, "median_time_50": 59.9996, "best_time": 83.25},
        {"completed": 0.4, "checkpoints_level": 2.14, "cells_new": 84.4, "deaths": 1.3, "exit_dist_min": 12.2,
         "gates_reached": 1.82, "wedged_steps": 612.4, "look_free_frac": 0.51, "look_gate_frac": 0.29,
         "targets_parked": 0.34, "exit_banished": 0.02, "route_source": 2, "ladder_collapsed": 1.0},
        {"time": -9.0, "checkpoint": 20.0, "novelty": 5.04, "path": 0.8, "level_complete": 50.0, "death": None},
        {"best_gates_reached": 4, "best_gate_hops": 2},
        {"gates_reached": 0.91},
        {"entropy_yaw": 2.3, "entropy_pitch": 1.8, "entropy_look_mode": 1.09},
    )
    assert lines == [
        "  fresh completed  62% of last 50",
        "  all completed    40%",
        "  best time        01:23.250",
        "  median time      01:00.000",  # whole milliseconds, never "00:60.000"
        "  gates/load       1.8 fresh 0.9 best 4 hops 2",
        "  checkpoints/load 2.1",
        "  parked/ep        0.34  ladder collapsed 100%  exit banished 2%",
        "  route layer      rooms",
        "  wedged/ep        612",
        "  new cells/ep     84",
        "  deaths/ep        1.30",
        "  closest to exit  12m",
        "  look free/gate   51%/29%  ent y/p/m 2.3/1.8/1.09",
        "  reward parts/ep",
        "    level_complete +50.0",
        "    checkpoint     +20.0",
        "    time           -9.0",
        "    novelty        +5.0",
    ], lines
    empty = dashboard.campaign_lines({"fresh_window": 0, "fresh_completion_rate": None, "median_time_50": None, "best_time": None}, {})
    assert empty[0] == "  fresh completed  — of last 0", empty
    assert len(empty) == 13, empty
    assert all("—" in line for line in empty[1:]), empty
    # The layer reads as a name only when the whole window agrees; a mixed curriculum shows the number.
    assert "  route layer      gates" in dashboard.campaign_lines({}, {"route_source": 1})
    assert "  route layer      exit vector" in dashboard.campaign_lines({}, {"route_source": 0})
    assert "  route layer      mixed 1.42" in dashboard.campaign_lines({}, {"route_source": 1.42})

    # The whole window on a real campaign status (charts, games table, Campaign panel).
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        old_every = progress_mod.HISTORY_EVERY_S
        progress_mod.HISTORY_EVERY_S = 0.2
        try:
            _train(status_path, 600, env_fns=[_make_campaign(1, []), _make_campaign(2, [])], run_name="campaign_run")
        finally:
            progress_mod.HISTORY_EVERY_S = old_every
        assert "campaign" in json.loads(status_path.read_text(encoding="utf-8"))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_poll_status_keeps_old_header():
    status = {
        "state": "running", "timesteps": 123456, "episodes": 300, "window": 100, "steps_per_s": 190.0,
        "mean_100": {"reward": 12.5, "kills": 3.0, "deaths": 1.2, "completed": 0.4, "fresh_start": 0.2,
                     "checkpoints_level": 1.5, "cells_new": 60.0, "exit_dist_min": 20.0,
                     "gates_reached": 1.8, "wedged_steps": 612.0, "look_gate_frac": 0.3,
                     "targets_parked": 0.34, "exit_banished": 0.02, "route_source": 2.0},
        "mean_fresh_100": {"gates_reached": 0.9, "checkpoints_level": 0.4, "completed": 0.0},
        "campaign": {"fresh_window": 50, "fresh_completion_rate": 0.4, "median_time_50": 95.0, "best_time": 83.25},
        "best_checkpoints_level": 3,
        "best_gates_reached": 4,
        "best_gate_hops": 2,
        "ppo": {"entropy_loss": -8.0, "entropy_yaw": 2.3},
        "reward_parts_mean_100": {"time": -9.0, "checkpoint": 20.0, "novelty": 5.0, "gate": 30.0, "gate_approach": 8.5},
    }
    # The header an older poll_status.py wrote: none of the campaign columns.
    old_header = ["wall_time", "timesteps", "episodes", "window", "steps_per_s", "state",
                  "reward", "kills", "ppo_entropy_loss", "part_kill", "part_total", "aim_share"]
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        for run in ("old_run", "new_run"):
            (runs / run).mkdir(parents=True)
            (runs / run / "status.json").write_text(json.dumps(status), encoding="utf-8")
        with (runs / "old_run" / "metrics_log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(old_header)
            w.writerow(["2026-09-15 23:00:00", 1000, 10, 10, 150.0, "running", 1.0, 0.5, -9.0, 0.5, 1.0, 0.0])

        for run in ("old_run", "new_run"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "poll_status.py"), "--run", run, "--runs-dir", str(runs), "--once"],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr

        with (runs / "old_run" / "metrics_log.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == old_header, rows[0]
        assert len(rows) == 3, rows
        assert len(rows[2]) == len(old_header), (len(rows[2]), len(old_header))
        appended = dict(zip(old_header, rows[2]))
        assert appended["timesteps"] == "123456" and appended["reward"] == "12.5" and appended["part_total"] == "54.5", appended

        # A new log gets every column, campaign ones included.
        with (runs / "new_run" / "metrics_log.csv").open(newline="", encoding="utf-8") as f:
            new_rows = list(csv.DictReader(f))
        assert len(new_rows) == 1, new_rows
        logged = new_rows[0]
        assert logged["fresh_window"] == "50" and logged["fresh_completion_rate"] == "0.4", logged
        assert logged["median_time_50"] == "95.0" and logged["best_time"] == "83.25", logged
        assert logged["best_checkpoints_level"] == "3" and logged["checkpoints_level"] == "1.5", logged
        assert logged["part_time"] == "-9.0" and logged["part_novelty"] == "5.0" and logged["part_level_complete"] == "", logged
        # The route columns, and the fresh-start split gates 2 and 3 are judged on.
        assert logged["gates_reached"] == "1.8" and logged["gates_reached_fresh"] == "0.9", logged
        assert logged["wedged_steps"] == "612.0" and logged["look_gate_frac"] == "0.3", logged
        assert logged["best_gates_reached"] == "4" and logged["best_gate_hops"] == "2", logged
        assert logged["part_gate"] == "30.0" and logged["part_gate_approach"] == "8.5", logged
        assert logged["ppo_entropy_yaw"] == "2.3", logged
        # The ladder-patience and exit-guard mechanisms. Both are absent from `old_header`, which is exactly
        # why an existing metrics_log.csv has to be moved aside to see them.
        assert logged["targets_parked"] == "0.34" and logged["exit_banished"] == "0.02", logged
        assert "targets_parked" not in appended and "exit_banished" not in appended
        # And the route layer, for the same reason: it is a new column on an existing log.
        assert logged["route_source"] == "2.0", logged
        assert "route_source" not in appended


def test_the_entropy_floors_live_coefficient_reaches_status_and_the_csv():
    """`train/ent_coef_live` is recorded by scripts/train.py's EntropyFloorCallback once per update.

    It only reaches anyone if `ProgressCallback` copies it out of the logger like PPO's own metrics and
    `poll_status.py` has a column for it, so both halves are pinned here. Read it against `ppo_entropy_loss`:
    entropy is the negation of that, and the coefficient is what is holding it up.
    """
    assert "train/ent_coef_live" in progress_mod.PPO_METRICS
    spec = importlib.util.spec_from_file_location("poll_status", ROOT / "scripts" / "poll_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = module.row({"ppo": {"ent_coef_live": 0.0044, "entropy_loss": -5.2}, "mean_100": {"ladder_collapsed": 1.0}})
    assert row["ppo_ent_coef_live"] == 0.0044 and row["ppo_entropy_loss"] == -5.2
    assert row["ladder_collapsed"] == 1.0, "and the detector's verdict is a column too"


def test_every_campaign_info_key_survives_the_numeric_pipeline():
    """§7.4 of the route spec: `CAMPAIGN_INFO_KEYS` must all be NUMERIC, and that was only ever prose.

    `env.py` documents the invariant ("they go through `Monitor(info_keywords=...)` and
    `ProgressCallback._num` and must all be numeric") and nothing enforced it. Revision 1 of the route spec put
    the STRING `route_source` in that tuple, which `_num` turns into None on every episode -- silently killing
    the one metric every recovery path in §12 reads, with no test failing anywhere. So the invariant is pinned
    here, against a representative campaign info: an int, a float, a bool and a None-valued key all pass, and
    a string does not.
    """
    info = {
        "kills": 12, "style": 900, "deaths": 1, "completed": 0, "fresh_start": 1, "level_seconds": 146.582,
        "checkpoints_level": 6, "cells_new": 421, "exit_dist_min": 12.25, "exit_ground_dist_min": 4.5,
        "oob_frac": 0.09,
        "gates_reached": 4, "wedged_steps": 0, "level_started": True, "look_gate_frac": 0.31,
        "slide_forced_frac": 0.0, "targets_parked": 0, "exit_banished": 0, "route_source": 2,
        "ladder_collapsed": 1,
        # The speed stage's three. They are None on every run that is not one, which is a missing VALUE, not a
        # non-numeric type -- the same shape `level_seconds` has had all along.
        "target_seconds": 22.5, "s_rank_seconds": 30.0, "completion_bonus": 200.0,
    }
    assert set(info) == set(CAMPAIGN_INFO_KEYS), "the representative info has to cover exactly the tuple"
    for key in CAMPAIGN_INFO_KEYS:
        assert progress_mod._num(info[key]) is not None, f"{key} does not survive _num(): it must be numeric"
    # `level_seconds` is None on every episode that did not complete, which is fine -- a missing VALUE is not a
    # non-numeric TYPE -- but a string is not, whatever it holds.
    assert progress_mod._num(None) is None and progress_mod._num("rooms") is None
    assert "route_source_name" in progress_mod.EPISODE_LOG_RAW, "the string half bypasses _num() instead"
    assert "route_source_name" not in CAMPAIGN_INFO_KEYS


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...", flush=True)
            fn()
            print(f"{name} ok")
    print("all tests passed")
