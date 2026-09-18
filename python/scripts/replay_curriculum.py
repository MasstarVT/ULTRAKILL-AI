"""Replay a run's episodes.jsonl through both curriculum weighting rules and print what each would have done.

Read-only: it opens one `episodes.jsonl`, never a checkpoint, never a port, never the run's own state. Safe to
point at a LIVE run's log while it is training (that is what it was written for).

    python scripts/replay_curriculum.py --episodes ../../ULTRAKILL-AI/python/runs/campaign_gates/episodes.jsonl
    python scripts/replay_curriculum.py --run campaign_gates --hours 8

For each of the last `--hours` hours it prints, per unlocked level, the share of fresh draws the level would
have had under `inverse_rate` (today's rule) and under `progress` (learning progress with a retention floor, a
cap and the blocked damping), plus the progress score and the signed drift the second rule reads. `--drift` adds
the distribution of that drift per level, which is how `PROGRESS_IMPROVEMENT` was chosen.

The statistics come from `ProgressCallback` itself -- the same accumulator the trainer runs -- so the only thing
this script adds is the replay loop and the table. A level counts as unlocked from its first episode in the log,
which is what the run itself did.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CURRICULUM_BLOCKED_FRESH_EPISODES,
    CURRICULUM_WEIGHT_CAP,
    PROGRESS_IMPROVEMENT,
    level_blocked,
    level_weights,
    progress_drift,
)
from ultrakill_ai.progress import ProgressCallback  # noqa: E402
from ultrakill_ai.times import short_level  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    """Every episode in the log that names its level, oldest first. A torn last line is skipped."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # the trainer may be appending as we read
            if isinstance(row, dict) and isinstance(row.get("level"), str) and row.get("t") is not None:
                rows.append(row)
    rows.sort(key=lambda r: r["t"])
    return rows


def as_info(row: dict) -> dict:
    """One episodes.jsonl row as the info dict `ProgressCallback._record_episode` expects."""
    return {
        "episode": {"r": row.get("reward"), "l": row.get("length")},
        "level": row["level"],
        "completed": row.get("completed"),
        "fresh_start": row.get("fresh_start"),
        "level_seconds": row.get("level_seconds"),
        "checkpoints_level": row.get("checkpoints_level"),
        "gates_reached": row.get("gates_reached"),
        "gate_hops_best": row.get("gate_hops_best"),
        "kills": row.get("kills"),
        "deaths": row.get("deaths"),
        "end_reason": row.get("end_reason"),
    }


def both_rules(table: dict, *, floor: float, cap: float, blocked: int) -> dict[str, tuple[float, float]]:
    """{level: (inverse_rate share, progress share)} over the unlocked levels, both normalised for comparison."""
    out = {}
    for rule in ("inverse_rate", "progress"):
        weights = dict(level_weights(list(table), table, floor=floor, rule=rule, cap=cap,
                                     blocked_fresh_episodes=blocked))
        total = sum(weights.values()) or 1.0
        for level, weight in weights.items():
            share = out.setdefault(level, [0.0, 0.0])
            share[0 if rule == "inverse_rate" else 1] = weight / total
    return {level: (shares[0], shares[1]) for level, shares in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="campaign_gates", help="run name, for the default episodes path")
    ap.add_argument("--episodes", default=None, help="path to episodes.jsonl (default runs/<run>/episodes.jsonl)")
    ap.add_argument("--hours", type=float, default=8.0, help="how many trailing hours to report (0 = all)")
    ap.add_argument("--floor", type=float, default=0.1, help="level_weight_floor, as the config sets it")
    ap.add_argument("--cap", type=float, default=CURRICULUM_WEIGHT_CAP)
    ap.add_argument("--blocked-fresh-episodes", type=int, default=CURRICULUM_BLOCKED_FRESH_EPISODES)
    ap.add_argument("--drift", action="store_true", help="also print the signed-drift distribution per level")
    args = ap.parse_args()

    path = Path(args.episodes) if args.episodes else Path("runs") / args.run / "episodes.jsonl"
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    rows = load_rows(path)
    if not rows:
        print(f"{path}: no episodes with a level name")
        return 2
    end = rows[-1]["t"]
    start = end - args.hours * 3600 if args.hours > 0 else rows[0]["t"]

    with tempfile.TemporaryDirectory() as tmp:
        cb = ProgressCallback(Path(tmp) / "status.json", 1, args.run, 1, update_every_s=1e9,
                              levels=[], curriculum_path=None)
        cb.episodes_path = Path(tmp) / "episodes.jsonl"
        order: list[str] = []
        drifts: dict[str, list[float]] = {}
        # Per hour and per level: the running sum of everything printed, because a weight applies to every draw
        # in the hour, not to the one episode that happened to end at the hour boundary.
        buckets: dict[int, dict[str, dict]] = {}
        for row in rows:
            level = row["level"]
            if level not in order:
                order.append(level)
                cb._level_record(level)["unlocked"] = True  # in the ladder from its first episode, as the run had it
            cb.order = list(order)
            cb._record_episode(int(row.get("env") or 0), as_info(row))
            if row["t"] < start:
                continue
            table = cb._level_table(with_weights=True)
            shares = both_rules(table, floor=args.floor, cap=args.cap, blocked=args.blocked_fresh_episodes)
            hour = buckets.setdefault(int((end - row["t"]) // 3600), {})
            for lv, record in table.items():
                drift = progress_drift(record)
                if drift is not None:
                    drifts.setdefault(lv, []).append(drift)
                if lv not in shares:
                    continue
                acc = hour.setdefault(lv, {"n": 0, "inverse": 0.0, "progress": 0.0, "rate": 0.0, "prog": 0.0,
                                           "blocked": 0, "dry": 0})
                acc["n"] += 1
                acc["inverse"] += shares[lv][0]
                acc["progress"] += shares[lv][1]
                acc["rate"] += record.get("fresh_completion_rate") or 0.0
                acc["prog"] += record.get("progress_score") or 0.0
                acc["blocked"] += 1 if level_blocked(
                    record, blocked_fresh_episodes=args.blocked_fresh_episodes) else 0
                acc["dry"] = record.get("dry_fresh_episodes") or 0

    print(f"{path}  {len(rows)} episodes with a level, replaying the last {args.hours:g} h "
          f"(floor {args.floor}, cap {args.cap}, blocked after {args.blocked_fresh_episodes} dry fresh episodes)")
    print("Mean share of the fresh draws each unlocked level would have had, per hour, under each rule.")
    print("  hour  level  fresh%   inverse_rate   progress   prog  blocked   dry")
    for hour in sorted(buckets, reverse=True):
        for level, acc in buckets[hour].items():
            n = acc["n"]
            print(f"  -{hour}h   {short_level(level):<4}  {acc['rate'] / n:>5.2f}   "
                  f"{acc['inverse'] / n:>11.0%}   {acc['progress'] / n:>8.0%}   {acc['prog'] / n:>4.2f}  "
                  f"{acc['blocked'] / n:>7.0%}  {acc['dry']:>4}")
        print()

    if args.drift:
        print(f"signed drift (fast - slow) per level, improvement threshold {PROGRESS_IMPROVEMENT}:")
        for level, values in sorted(drifts.items()):
            over = sum(1 for v in values if v > PROGRESS_IMPROVEMENT) / len(values)
            print(f"  {short_level(level):<4} n {len(values):>5}  mean {statistics.mean(values):+.3f}  "
                  f"median {statistics.median(values):+.3f}  min {min(values):+.3f}  max {max(values):+.3f}  "
                  f"share above the threshold {over:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
