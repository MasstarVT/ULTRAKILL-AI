"""Preserves the best-performing checkpoint of a run, automatically.

PPO does not improve monotonically. On the night of 2026-09-15 the run peaked at 4.40M steps
(kills/min 6.62, reward 217, deaths 1.33) and then degraded for half a million steps as entropy
collapsed, down to 5.36 kills/min by 4.93M. Nothing was watching, and `latest.zip` tracks the LAST
policy, not the best one, so the peak would have been lost with the next checkpoint rotation.

This copies the checkpoint nearest each new best to `best.zip` and records how it was chosen in
`best.json`. It only reads `metrics_log.csv` (written by poll_status.py) and copies files, so it is
safe to leave running alongside training and never touches the bridge ports.

Scoring: kills per game-minute, smoothed over several consecutive samples, and only from samples
backed by a full 100-episode window. Ties break on lower deaths.

    python scripts/keep_best.py --run cybergrind_ppo_v2            # watch until stopped
    python scripts/keep_best.py --run cybergrind_ppo_v2 --once     # report and exit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import time
from pathlib import Path

SMOOTH = 9          # samples per smoothing window (~4.5 min at a 30 s poll)
MIN_WINDOW = 100    # require a full episode window behind every mean


def num(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def scored(csv_path: Path) -> list[tuple[float, float, float, float]]:
    """(score, deaths, reward, timesteps), smoothed, newest last."""
    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    except OSError:
        return []
    rows = [r for r in rows if (num(r, "window") or 0) >= MIN_WINDOW and num(r, "kills_per_min") is not None]
    out = []
    half = SMOOTH // 2
    for i in range(half, len(rows) - half):
        w = rows[i - half:i + half + 1]
        kpm = [num(r, "kills_per_min") for r in w]
        dth = [num(r, "deaths") for r in w]
        rew = [num(r, "reward") for r in w]
        if any(v is None for v in kpm):
            continue
        out.append((
            statistics.mean(kpm),
            statistics.mean([d for d in dth if d is not None] or [0.0]),
            statistics.mean([v for v in rew if v is not None] or [0.0]),
            num(rows[i], "timesteps") or 0.0,
        ))
    return out


def nearest_checkpoint(model_dir: Path, timesteps: float) -> Path | None:
    """Newest checkpoint at or before `timesteps` -- the weights that produced that score."""
    cks = []
    for p in model_dir.glob("ckpt_*_steps.zip"):
        m = re.search(r"ckpt_(\d+)_steps", p.name)
        if m:
            cks.append((int(m.group(1)), p))
    cks.sort()
    at_or_before = [p for n, p in cks if n <= timesteps]
    return at_or_before[-1] if at_or_before else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cybergrind_ppo_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    csv_path = Path(a.runs_dir) / a.run / "metrics_log.csv"
    model_dir = Path(a.models_dir) / a.run
    best_zip = model_dir / "best.zip"
    best_json = model_dir / "best.json"

    best_score = -1.0
    if best_json.exists():
        try:
            best_score = float(json.loads(best_json.read_text(encoding="utf-8")).get("score", -1.0))
        except (OSError, ValueError, json.JSONDecodeError):
            best_score = -1.0

    while True:
        series = scored(csv_path)
        if series:
            score, deaths, reward, ts = max(series, key=lambda s: (s[0], -s[1]))
            newest = series[-1]
            if score > best_score + 1e-9:
                src = nearest_checkpoint(model_dir, ts)
                if src and src.exists():
                    shutil.copy2(src, best_zip)
                    best_json.write_text(json.dumps({
                        "score_metric": "kills_per_min, smoothed over %d samples" % SMOOTH,
                        "score": round(score, 4),
                        "deaths": round(deaths, 4),
                        "reward": round(reward, 3),
                        "at_timesteps": ts,
                        "checkpoint": src.name,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, indent=2), encoding="utf-8")
                    best_score = score
                    print(f"[keep_best] new best {score:.2f} kills/min at {ts:,.0f} -> {src.name}", flush=True)
            # Warn when the current policy has fallen well below the best: that is the signal to roll back.
            if best_score > 0 and newest[0] < best_score * 0.85:
                print(f"[keep_best] WARNING current {newest[0]:.2f} kills/min is "
                      f"{(1 - newest[0]/best_score)*100:.0f}% below best {best_score:.2f} "
                      f"(best.zip holds the good weights)", flush=True)
        if a.once:
            if best_json.exists():
                print(best_json.read_text(encoding="utf-8"))
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
