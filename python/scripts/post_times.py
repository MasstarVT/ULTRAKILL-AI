"""Post a training run's best official level times to times.md. Read-only on the run: safe while training.

`eval.py --record-times` needs a game and a free bridge port; this reads what training has already saved.
`runs/<run>/best_runs/<level>.json` holds the fastest FRESH-START completion per level (official time, rank,
kills, deaths, difficulty), and `runs/<run>/episodes.jsonl` gives the step count it happened at. A level is
posted only when its time beats the row times.md already holds, so running this repeatedly adds nothing.

    python scripts/post_times.py --run campaign_gates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.times import (  # noqa: E402
    LEADERBOARD_HEADING, TimeEntry, _data_rows, _ms, _table, format_time, parse_time, record_file, short_level)

TIMES_MD = ROOT.parent / "times.md"


def held_time(markdown: str, level: str) -> float | None:
    """The leaderboard's current time for `level` (scene name), or None when the level has no row."""
    lines = markdown.splitlines()
    start, end = _table(lines, LEADERBOARD_HEADING)
    for cells in _data_rows(lines[start:end]):
        if cells[0] == short_level(level):
            return parse_time(cells[1])
    return None


def steps_of(episodes: Path, level: str, seconds: float) -> int | None:
    """The training step count of the fresh-start completion that set `seconds` on `level`."""
    if not episodes.exists():
        return None
    with episodes.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (row.get("completed") and row.get("fresh_start")):
                continue
            if row.get("level", level) != level or row.get("level_seconds") is None:
                continue
            if abs(float(row["level_seconds"]) - seconds) < 1e-3:
                return int(row["timesteps"])
    return None


def generation(run: str, steps: int | None) -> str:
    return run if steps is None else f"{run}@{steps / 1e6:.2f}M"


def post(run_dir: Path, times_md: Path, run: str) -> list[str]:
    """Posts every level whose best run beats times.md. Returns one printed line per level posted."""
    posted = []
    for path in sorted((run_dir / "best_runs").glob("*.json")):
        best = json.loads(path.read_text(encoding="utf-8"))
        level, seconds = best["level"], float(best["seconds"])
        held = held_time(times_md.read_text(encoding="utf-8"), level)
        if held is not None and _ms(seconds) >= _ms(held):
            continue
        entry = TimeEntry(
            level=level, seconds=seconds, rank=best.get("rank") or "",
            generation=generation(run, steps_of(run_dir / "episodes.jsonl", level, seconds)),
            difficulty=int(best.get("difficulty", 3)),
            date=(best.get("saved_at") or time.strftime("%Y-%m-%d"))[:10],
            kills=int(best.get("kills", 0)), deaths=int(best.get("deaths", 0)),
            notes="training episode (sampled actions), fresh start")
        record_file(times_md, entry)
        posted.append(f"{short_level(level)}: {format_time(seconds)} rank {entry.rank or '-'} ({entry.generation})")
    return posted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under runs/")
    parser.add_argument("--times", default=str(TIMES_MD), help="times.md to update")
    args = parser.parse_args()
    posted = post(ROOT / "runs" / args.run, Path(args.times), args.run)
    for line in posted:
        print("posted", line)
    if not posted:
        print("nothing new: times.md already holds every best run")


if __name__ == "__main__":
    main()
