"""Appends live training metrics to a CSV so they survive restarts.

`status.json` only holds a snapshot: `mean_100` is a 100-episode deque that `ProgressCallback` does NOT
carry across a restart, and `history[]` keeps only reward/kills/wave. Every aiming diagnostic
(`on_target_frac`, `yaw_track`, `pitch_track`, `pitch_mean`, `look_up_mean`, `enemy_*`) exists only in
the current snapshot, so a restart throws the record away. That is how a 5-episode window got mistaken
for a trend on the night of 2026-09-15.

Read-only: it polls a file and never touches the bridge ports, so it is safe to leave running
alongside training.

    python scripts/poll_status.py --run cybergrind_ppo_v2            # every 30 s until stopped
    python scripts/poll_status.py --run cybergrind_ppo_v2 --once
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

FIELDS = [
    "reward", "length", "kills", "kills_per_min", "deaths", "wave", "style",
    "on_target_frac", "firing_frac", "firing_on_target_frac", "enemy_visible_frac",
    "yaw_track", "pitch_track", "pitch_mean", "look_up_mean", "pitch_abs_mean",
    "enemy_angle_mean", "enemy_yaw_angle_mean", "enemy_pitch_err_mean",
    "enemy_elev_mean", "enemy_elev_abs_mean", "enemy_elev_over15_frac",
    "enemy_dist_mean", "enemy_close_frac", "reset_seconds",
]
PPO_FIELDS = ["entropy_loss", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "learning_rate"]
PART_FIELDS = ["aim_yaw", "aim_pitch", "aim_locked", "aim", "kill", "damage_dealt", "damage_taken", "death", "wave", "style", "step"]


def row(status: dict) -> dict:
    m = status.get("mean_100") or {}
    ppo = status.get("ppo") or {}
    parts = status.get("reward_parts_mean_100") or {}
    out = {
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timesteps": status.get("timesteps"),
        "episodes": status.get("episodes"),
        # The number of episodes actually behind the means. Anything under ~50 is not a trend.
        "window": status.get("window"),
        "steps_per_s": status.get("steps_per_s"),
        "state": status.get("state"),
    }
    out.update({k: m.get(k) for k in FIELDS})
    out.update({f"ppo_{k}": ppo.get(k) for k in PPO_FIELDS})
    out.update({f"part_{k}": parts.get(k) for k in PART_FIELDS})
    total = sum(v for v in parts.values() if isinstance(v, (int, float)))
    aim = sum(v for k, v in parts.items() if k.startswith("aim") and isinstance(v, (int, float)))
    out["part_total"] = total
    out["aim_share"] = (aim / total) if total else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cybergrind_ppo_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    status_path = Path(a.runs_dir) / a.run / "status.json"
    out_path = Path(a.runs_dir) / a.run / "metrics_log.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = list(row({}).keys())
    new = not out_path.exists()
    last_steps = None
    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = None  # mid-write or not started yet; try again next tick
        if status and status.get("timesteps") != last_steps:
            last_steps = status.get("timesteps")
            with out_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if new:
                    w.writeheader()
                    new = False
                w.writerow(row(status))
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
