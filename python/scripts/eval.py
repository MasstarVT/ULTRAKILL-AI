"""Runs a trained model so you can watch it or measure it.

    python scripts/eval.py models/cybergrind_ppo/latest.zip --episodes 3 --realtime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to a saved .zip")
    parser.add_argument("--algo", choices=["ppo", "rppo"], default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--realtime", action="store_true", help="normal speed with sound")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    args = parser.parse_args()

    cfg_path = Path(args.model).parent / "env_config.yaml"
    cfg = EnvConfig.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else EnvConfig()
    if args.realtime:
        cfg.unlimited_fps = False
        cfg.mute = False

    if args.algo == "rppo":
        from sb3_contrib import RecurrentPPO as cls
    else:
        from stable_baselines3 import PPO as cls
    model = cls.load(args.model, device="cpu")

    env = UltrakillEnv(cfg)
    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            state, start = None, np.ones((1,), dtype=bool)
            total, steps = 0.0, 0
            while True:
                action, state = model.predict(obs, state=state, episode_start=start, deterministic=not args.stochastic)
                start = np.zeros((1,), dtype=bool)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            results.append((total, info))
            extra = f" route={info['route_progress']:.0%}" if "route_progress" in info else f" wave={info['wave']}"
            print(f"episode {ep}: reward={total:.1f} steps={steps} kills={info['kills']}{extra} end={info.get('end_reason')}")
    finally:
        env.close()

    print(f"mean reward {np.mean([r for r, _ in results]):.1f} over {len(results)} episodes")


if __name__ == "__main__":
    main()
