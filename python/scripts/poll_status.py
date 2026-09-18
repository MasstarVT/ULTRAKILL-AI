"""Appends live training metrics to a CSV so they survive restarts.

`status.json` only holds a snapshot: `mean_100` is a 100-episode deque that `ProgressCallback` does NOT
carry across a restart, and `history[]` keeps only reward/kills/wave. Every aiming diagnostic
(`on_target_frac`, `yaw_track`, `pitch_track`, `pitch_mean`, `look_up_mean`, `enemy_*`) exists only in
the current snapshot, so a restart throws the record away. That is how a 5-episode window got mistaken
for a trend on the night of 2026-09-15.

Read-only: it polls a file and never touches the bridge ports, so it is safe to leave running
alongside training.

An existing `metrics_log.csv` keeps its header. Rows are written under the columns it already has, and
values for columns it lacks are dropped, so adding a column here never shifts an old log out of
alignment. Move the old file aside to start logging the new columns.

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
    "completed", "fresh_start", "checkpoints_level", "cells_new", "oob_frac", "exit_dist_min",
    # Mod 0.7.2: the same measure to the nearest STANDABLE ground beside the pit rather than to the pit's own
    # transform, which sits 61-75 m under the floor on 0-2 and gives `exit_dist_min` a floor it can never go
    # under. This is the column to read for "did the agent get near the exit". A new column, so an existing
    # metrics_log.csv must be moved aside to get it.
    "exit_ground_dist_min",
    "gates_reached", "wedged_steps", "level_started", "look_free_frac", "look_gate_frac", "slide_forced_frac",
    # The 2026-09-17 patience/exit-guard mechanisms. An existing metrics_log.csv keeps its own header, so move
    # the old file aside to get these two columns.
    "targets_parked", "exit_banished",
    # 1 when the level being played has a collapsed gate ladder, which is what lets `targets_parked` move at
    # all under `gate_patience_mode: collapsed`. On a curriculum it is the share of the window spent on
    # collapsed levels; `targets_parked` above 0 while this reads 0 would mean the detector let patience run
    # on a healthy ladder, which is the 0-1 regression this column exists to catch.
    "ladder_collapsed",
    # Which route layer the window ran on: 0 exit vector, 1 gate ladder, 2 offline room trunk. Read
    # `part_gate_approach` against `part_level_complete` on any window where this is above 1 -- the route
    # spec's §12.6 tripwire is 6x, and the lever is the route file, never the weight.
    "route_source",
]
# The same means over fresh-start episodes only (status["mean_fresh_100"]): a respawn episode inherits
# gates_reached and checkpoints_level from its level load, so only these two say how a whole run goes.
FRESH_FIELDS = ["gates_reached", "checkpoints_level", "completed"]
# Campaign runs only (status["campaign"]): completion rate and median time over the last 50 fresh starts.
CAMPAIGN_FIELDS = ["fresh_window", "fresh_completion_rate", "median_time_50", "best_time"]
BEST_FIELDS = ["best_checkpoints_level", "best_gates_reached", "best_gate_hops"]
PPO_FIELDS = ["entropy_loss", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "learning_rate",
              "entropy_yaw", "entropy_pitch", "entropy_look_mode",
              # The adaptive floor's live coefficient: equal to the config's `ent_coef` while total entropy
              # (|ppo_entropy_loss|) sits above `ent_floor` + 1, and climbing while it does not.
              "ent_coef_live"]
PART_FIELDS = ["aim_yaw", "aim_pitch", "aim_locked", "aim", "kill", "damage_dealt", "damage_taken", "death", "wave", "style", "step",
               "time", "checkpoint", "arena_clear", "door_unlock", "novelty", "path", "level_complete", "punch",
               "gate", "gate_approach", "item_pickup", "item_placed"]


def row(status: dict) -> dict:
    m = status.get("mean_100") or {}
    ppo = status.get("ppo") or {}
    parts = status.get("reward_parts_mean_100") or {}
    campaign = status.get("campaign") or {}
    out = {
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timesteps": status.get("timesteps"),
        "episodes": status.get("episodes"),
        # The number of episodes actually behind the means. Anything under ~50 is not a trend.
        "window": status.get("window"),
        "steps_per_s": status.get("steps_per_s"),
        "state": status.get("state"),
    }
    fresh = status.get("mean_fresh_100") or {}
    out.update({k: m.get(k) for k in FIELDS})
    out.update({f"{k}_fresh": fresh.get(k) for k in FRESH_FIELDS})
    out.update({k: campaign.get(k) for k in CAMPAIGN_FIELDS})
    # A multi-level run only: how many levels are unlocked, so `fresh_completion_rate` (a shrunk SUM over them,
    # not a rate) can be read against the right denominator. Deliberately one column and not one per level: a
    # column per level would break this file's "keep the existing header" rule the moment a level unlocked, and
    # the per-level numbers live in status.json's campaign.levels table.
    levels = campaign.get("levels")
    out["levels_unlocked"] = (sum(1 for row in levels.values() if isinstance(row, dict) and row.get("unlocked"))
                              if isinstance(levels, dict) else None)
    out.update({k: status.get(k) for k in BEST_FIELDS})
    out.update({f"ppo_{k}": ppo.get(k) for k in PPO_FIELDS})
    out.update({f"part_{k}": parts.get(k) for k in PART_FIELDS})
    total = sum(v for v in parts.values() if isinstance(v, (int, float)))
    aim = sum(v for k, v in parts.items() if k.startswith("aim") and isinstance(v, (int, float)))
    out["part_total"] = total
    out["aim_share"] = (aim / total) if total else None
    return out


def existing_header(path: Path) -> list[str] | None:
    """The header row of an existing CSV, or None when there is no file or it is empty."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None) or None
    except OSError:
        return None


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

    header = existing_header(out_path)
    new = header is None
    if new:
        header = list(row({}).keys())
    last_steps = None
    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = None  # mid-write or not started yet; try again next tick
        if status and status.get("timesteps") != last_steps:
            last_steps = status.get("timesteps")
            with out_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if new:
                    w.writeheader()
                    new = False
                w.writerow(row(status))
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
