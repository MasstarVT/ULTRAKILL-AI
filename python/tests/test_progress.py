"""Tests for ultrakill_ai.progress.ProgressCallback, without the game.

    .venv\\Scripts\\python -m pytest tests -q     (if pytest is installed)
    .venv\\Scripts\\python tests\\test_progress.py
"""

from __future__ import annotations

import json
import subprocess
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


def _make(seed):
    return lambda: Monitor(FakeGrindEnv(seed), info_keywords=("kills", "wave", "style"))


def _train(status_path: Path, timesteps: int, model=None, resume=False):
    venv = DummyVecEnv([_make(1), _make(2)])
    if model is None:
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=128, n_epochs=2, policy_kwargs={"net_arch": [32]}, device="cpu", verbose=0)
    else:
        model.set_env(venv)
    target = (model.num_timesteps if resume else 0) + timesteps
    cb = ProgressCallback(status_path, target, "test_run", 2, update_every_s=0.2)
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
        assert m["route_progress"] is None
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


def test_dashboard_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        _train(status_path, 600)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...", flush=True)
            fn()
            print(f"{name} ok")
    print("all tests passed")
