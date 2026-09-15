"""Smoke test: random actions with automatic resets.

    python scripts/random_agent.py --mode cybergrind --episodes 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cybergrind", choices=["cybergrind", "campaign"])
    parser.add_argument("--level", default="Level 0-1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--port", type=int, default=47800)
    args = parser.parse_args()

    env = UltrakillEnv(EnvConfig(mode=args.mode, level=args.level, max_steps=args.max_steps, port=args.port))
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            assert env.observation_space.contains(obs), "observation outside declared space"
            total, steps, start = 0.0, 0, time.perf_counter()
            while True:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - start
            print(
                f"episode {ep}: steps={steps} reward={total:.2f} kills={info['kills']} wave={info['wave']} "
                f"end={info.get('end_reason')} ({steps / elapsed:.0f} steps/s)"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
