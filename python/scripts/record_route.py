"""Records your own run through a campaign level as a route for navigation rewards.

    python scripts/record_route.py --level "Level 0-1"

Load the level, run this, then play through it normally. Recording stops when the level is
completed (or on Ctrl+C) and the route is saved to routes/<level>.json. If you die, points recorded
after the spot you respawn at are discarded so the route follows your successful attempt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.protocol import BridgeClient  # noqa: E402
from ultrakill_ai.routes import Route, route_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, help='scene name, e.g. "Level 0-1"')
    parser.add_argument("--spacing", type=float, default=1.0, help="metres between route points")
    parser.add_argument("--route-dir", default="routes")
    parser.add_argument("--port", type=int, default=47800)
    parser.add_argument("--hz", type=float, default=20.0, help="polling rate")
    args = parser.parse_args()

    points: list[np.ndarray] = []
    was_dead = False

    with BridgeClient(port=args.port) as client:
        client.connect()
        print(f"Recording {args.level}. Play through the level; Ctrl+C to stop early.")
        try:
            while True:
                obs = client.get_obs()
                p = obs.get("player")
                if obs.get("scene") != args.level or not p:
                    time.sleep(1.0 / args.hz)
                    continue

                if p["dead"]:
                    was_dead = True
                elif obs.get("ready"):
                    pos = np.asarray(p["pos"], dtype=np.float32)
                    if was_dead and points:
                        # Respawned at a checkpoint: rewind the route to the nearest recorded point.
                        d = np.linalg.norm(np.stack(points) - pos, axis=1)
                        keep = int(np.argmin(d)) + 1
                        print(f"Respawn detected, discarding {len(points) - keep} points")
                        del points[keep:]
                    was_dead = False

                    if not points or np.linalg.norm(points[-1] - pos) >= args.spacing:
                        points.append(pos)
                        if len(points) % 50 == 0:
                            print(f"{len(points)} points")

                if obs.get("stats", {}).get("level_complete"):
                    print("Level complete.")
                    break
                time.sleep(1.0 / args.hz)
        except KeyboardInterrupt:
            print("Stopped.")

    if len(points) < 2:
        print("Not enough points recorded, nothing saved.")
        return
    path = route_path(args.route_dir, args.level)
    Route(scene=args.level, points=np.stack(points)).save(path, args.spacing)
    print(f"Saved {len(points)} points to {path}")


if __name__ == "__main__":
    main()
