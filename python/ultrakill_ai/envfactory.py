"""The env factory the `SubprocVecEnv` workers run, and nothing else.

**This module exists to keep torch and stable_baselines3 out of the twelve worker processes.** SB3 pickles
each `env_fn` with cloudpickle; a closure is pickled by value, but the module-level names it *references*
are pickled by reference, so the worker imports whatever module those names live in. While `make_env`
lived in `scripts/train.py` and wrapped the env in `stable_baselines3.common.monitor.Monitor`, unpickling
one env_fn imported `stable_baselines3`, whose package `__init__` imports PPO and therefore torch --
measured at 175 MB of committed private bytes per worker, for a process that only talks to a game over a
socket and packs a 479-float array.

`EpisodeMonitor` below is SB3's `Monitor` for the one configuration this project uses (no csv file, early
resets allowed, no reset keywords), reimplemented so it can live in a package that imports nothing heavier
than gymnasium. It produces a byte-identical `info["episode"]` -- the same keys, the same `round(., 6)`,
the same float64 sum -- so `rollout/ep_rew_mean`, `ProgressCallback` and everything downstream read
exactly what they read before. The alternative, SB3's own `VecMonitor` in the trainer process, was
rejected: it accumulates the return in float32 and times episodes from the vec env's construction rather
than each env's, which would have moved two numbers this project has years of history on.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym


class EpisodeMonitor(gym.Wrapper):
    """Records episode return, length, time and chosen info keys into `info["episode"]` on the last step.

    A faithful stand-in for `stable_baselines3.common.monitor.Monitor` with `filename=None`,
    `allow_early_resets=True` and `reset_keywords=()`. The accessors SB3 and its users reach for
    (`get_episode_rewards`, `get_episode_lengths`, `get_episode_times`, `get_total_steps`) are kept.
    """

    def __init__(self, env: gym.Env, info_keywords: tuple[str, ...] = ()):
        super().__init__(env)
        self.t_start = time.time()
        self.info_keywords = tuple(info_keywords)
        self.rewards: list[float] = []
        self.needs_reset = True
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.total_steps = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def get_total_steps(self) -> int:
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        return self.episode_times


def make_env(cfg, info_keywords: tuple[str, ...]):
    """The `env_fn` SB3 pickles into a worker. Every name it closes over must come from a light module."""

    def _init():
        from ultrakill_ai.env import UltrakillEnv  # imported in the worker, not at pickling time

        return EpisodeMonitor(UltrakillEnv(cfg), info_keywords=info_keywords)

    return _init
