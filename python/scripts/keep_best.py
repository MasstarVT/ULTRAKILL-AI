"""Preserves the best-performing checkpoint of a run, automatically.

PPO does not improve monotonically. On the night of 2026-09-15 the run peaked at 4.40M steps
(kills/min 6.62, reward 217, deaths 1.33) and then degraded for half a million steps as entropy
collapsed, down to 5.36 kills/min by 4.93M. Nothing was watching, and `latest.zip` tracks the LAST
policy, not the best one, so the peak would have been lost with the next checkpoint rotation.

This copies the checkpoint nearest each new best to `best.zip` and records how it was chosen in
`best.json`. It only reads `metrics_log.csv` (written by poll_status.py) and copies files, so it is
safe to leave running alongside training and never touches the bridge ports.

Scoring, smoothed over several consecutive samples in every mode:
- `--metric kills_per_min` (default, Cyber Grind): kills per game-minute, only from samples backed by a
  full 100-episode window. Ties break on lower deaths.
- `--metric campaign`: the completion rate over the last 50 fresh-start episodes, only from samples
  backed by at least 20 of them. Ties break on the lower best official time.
- `--metric time` (a SPEED STAGE, 2026-09-18): the lowest MEDIAN official time over the last 50 fresh episodes,
  but only from samples whose completion rate is at least `--min-rate` (0.3) -- a fast time set by one lucky
  load in a hundred is not a policy, and neither is the run's all-time best, which never moves back up. Ties
  break on the higher completion rate.

A run's `best.json` records which tie-break chose it (`penalty_name`), and this refuses to start against a
file written by another metric: the three scores are not comparable, and overwriting one would throw away a
run's best weights.

    python scripts/keep_best.py --run cybergrind_ppo_v2            # watch until stopped
    python scripts/keep_best.py --run cybergrind_ppo_v2 --once     # report and exit
    python scripts/keep_best.py --run campaign_ppo --metric campaign
    python scripts/keep_best.py --run spec_0-1_speed --metric time
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.procmem import cap_blas_threads  # noqa: E402

cap_blas_threads()  # before ultrakill_ai.times, which reaches numpy through ultrakill_ai.campaign

from ultrakill_ai.times import valid_official_seconds  # noqa: E402

SMOOTH = 9          # samples per smoothing window (~4.5 min at a 30 s poll)
MIN_WINDOW = 100    # kills_per_min: require a full episode window behind every mean
MIN_FRESH_WINDOW = 20  # campaign: fresh-start episodes behind the completion rate (its window holds 50)
MIN_RATE = 0.3      # time: the completion rate a sample needs before its clock is worth ranking at all


class Metric(NamedTuple):
    window: str     # CSV column counting the episodes behind the means
    min_window: int
    score: str      # CSV column, ranked higher-is-better AFTER `score_sign`
    penalty: str    # CSV column that breaks ties, lower-is-better AFTER `penalty_sign`; also `penalty_name`
    missing: float  # penalty when no sample in a smoothing window has one (already signed)
    unit: str
    # -1 turns a column where LOWER is better into the shape `rank_key` wants, without touching the ranking
    # itself. Both default to +1, so `kills_per_min` and `campaign` behave exactly as they always have.
    score_sign: float = 1.0
    penalty_sign: float = 1.0
    gate: str = ""       # an extra column a sample must reach to count at all ("" = no gate)
    min_gate: float = 0.0
    # Which of the two columns hold an OFFICIAL LEVEL TIME, and so have to pass `valid_official_seconds`
    # before they are ranked on. metrics_log.csv is an append-only history: rows written before the
    # 2026-09-19 fix still carry the impossible times that bug produced, and `--metric time` ranks on the
    # LOWEST median -- a single zero in the history would otherwise be the unbeatable best forever.
    score_is_time: bool = False
    penalty_is_time: bool = False


METRICS = {
    "kills_per_min": Metric("window", MIN_WINDOW, "kills_per_min", "deaths", 0.0, "kills/min"),
    # No completed run means no best time: an infinite penalty, so the first finite time wins the tie.
    "campaign": Metric("fresh_window", MIN_FRESH_WINDOW, "fresh_completion_rate", "best_time", math.inf, "fresh completion rate",
                       penalty_is_time=True),
    # A speed stage. The score is the official time NEGATED so that "higher is better" still holds everywhere
    # below, and the tie-break is the completion rate negated for the same reason -- a faster time wins, and at
    # equal times the more reliable policy does. `gate` is what stops a single lucky load setting the record:
    # a sample whose completion rate is under `min_gate` is not scored at all. `missing` is 0.0 because a
    # window with no rate at all cannot pass the gate in the first place.
    # SCORED ON THE MEDIAN, NOT ON `best_time` (2026-09-18 review). `best_time` is the run's LIFETIME MINIMUM:
    # it only ever falls, so `-best_time` is monotone non-decreasing, every sample after the last record ties
    # at the maximum, and `rank_key` is then decided entirely by the tie-break -- `--metric time` would quietly
    # behave as `--metric campaign` with a gate, and the "current is well below best" warning below could never
    # fire, because `raw_new > raw_best` cannot hold for a running minimum. `median_time_50` is a real
    # per-sample statistic of the policy that produced the window, it moves in both directions, and it is
    # already a column of metrics_log.csv (poll_status.CAMPAIGN_FIELDS).
    "time": Metric("fresh_window", MIN_FRESH_WINDOW, "median_time_50", "fresh_completion_rate", 0.0,
                   "s (median official time)",
                   score_sign=-1.0, penalty_sign=-1.0, gate="fresh_completion_rate", min_gate=MIN_RATE,
                   score_is_time=True),
}


def num(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _score(row: dict, m: Metric) -> float | None:
    """The sample's score column, or None when it is a time the game cannot have reported."""
    v = num(row, m.score)
    return valid_official_seconds(v) if m.score_is_time else v


def _penalty(row: dict, m: Metric) -> float | None:
    """The sample's tie-break column, on the same terms: an invalid time reads as "this sample has none"."""
    v = num(row, m.penalty)
    return valid_official_seconds(v) if m.penalty_is_time else v


def scored(csv_path: Path, metric: str = "kills_per_min", *,
           min_rate: float | None = None) -> list[tuple[float, float, float, float]]:
    """(score, penalty, reward, timesteps), smoothed, newest last. Higher score and lower penalty are better.

    `min_rate` overrides a gated metric's own `min_gate` (`--metric time` only); it is ignored by the others,
    which carry no gate.
    """
    m = METRICS[metric]
    gate_at = m.min_gate if min_rate is None else float(min_rate)
    try:
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    rows = [r for r in rows if (num(r, m.window) or 0) >= m.min_window and _score(r, m) is not None
            and (not m.gate or (num(r, m.gate) or 0.0) >= gate_at)]
    out = []
    half = SMOOTH // 2
    for i in range(half, len(rows) - half):
        w = rows[i - half:i + half + 1]
        score = [_score(r, m) for r in w]
        pen = [_penalty(r, m) for r in w]
        rew = [num(r, "reward") for r in w]
        if any(v is None for v in score):
            continue
        out.append((
            m.score_sign * statistics.mean(score),
            m.penalty_sign * statistics.mean([p for p in pen if p is not None] or [m.missing]),
            statistics.mean([v for v in rew if v is not None] or [0.0]),
            num(rows[i], "timesteps") or 0.0,
        ))
    return out


def rank_key(score: float, penalty: float) -> tuple[float, float]:
    """Higher is better: the score, then the negated penalty, both at the 4 decimals best.json keeps.

    best_of and is_better share it, so the sample picked as best is always the one compared with the stored best,
    and a restart does not re-save the best it already holds (a raw score against its own rounded copy).
    """
    return round(score, 4), -round(penalty, 4)


def best_of(series: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """The sample with the highest score, ties broken on the lowest penalty."""
    return max(series, key=lambda s: rank_key(s[0], s[1]))


def is_better(candidate: tuple[float, float], stored: tuple[float, float] | None) -> bool:
    """True when (score, penalty) strictly beats the stored best: a higher score, or the same score and a lower penalty."""
    if stored is None:
        return True
    return rank_key(*candidate) > rank_key(*stored)


def stored_best(best_json: Path) -> tuple[float, float] | None:
    """(score, penalty) of the saved best, or None. best.json files written before --metric keep the penalty as `deaths`."""
    try:
        data = json.loads(best_json.read_text(encoding="utf-8"))
        penalty = data.get("penalty", data.get("deaths"))
        return float(data["score"]), (math.inf if penalty is None else float(penalty))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def stored_penalty_name(best_json: Path) -> str | None:
    """The tie-break the saved best was chosen with (`deaths` for files written before --metric), or None without one."""
    try:
        return str(json.loads(best_json.read_text(encoding="utf-8")).get("penalty_name", "deaths"))
    except (OSError, ValueError, AttributeError):
        return None


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


def save_if_better(series: list[tuple[float, float, float, float]], model_dir: Path, metric: str,
                   best: tuple[float, float] | None) -> tuple[float, float] | None:
    """Copies the checkpoint behind the series' best sample to best.zip when it beats `best`. Returns the best now held."""
    m = METRICS[metric]
    score, penalty, reward, ts = best_of(series)
    if not is_better((score, penalty), best):
        return best
    src = nearest_checkpoint(model_dir, ts)
    if not (src and src.exists()):
        return best
    shutil.copy2(src, model_dir / "best.zip")
    (model_dir / "best.json").write_text(json.dumps({
        # `score` and `penalty` are stored SIGNED, the way they are ranked, so a restart compares like with
        # like. The string says which way round they read, because -118.5 in a file is otherwise a puzzle.
        "score_metric": ("%s, smoothed over %d samples" % (m.score, SMOOTH) if m.score_sign > 0 else
                         "-%s (lower is better), smoothed over %d samples" % (m.score, SMOOTH)),
        "score": round(score, 4),
        "penalty": round(penalty, 4) if math.isfinite(penalty) else None,  # null: no completed run yet
        "penalty_name": m.penalty,
        "reward": round(reward, 3),
        "at_timesteps": ts,
        "checkpoint": src.name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[keep_best] new best {m.score_sign * score:.2f} {m.unit} "
          f"({m.penalty} {m.penalty_sign * penalty:.2f}) at {ts:,.0f} -> {src.name}", flush=True)
    return score, penalty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cybergrind_ppo_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--metric", choices=sorted(METRICS), default="kills_per_min",
                    help="kills_per_min (Cyber Grind), campaign (fresh-start completion rate, then best "
                         "time) or time (a speed stage: the best official time, gated on --min-rate)")
    ap.add_argument("--min-rate", type=float, default=MIN_RATE,
                    help="--metric time only: the fresh completion rate a sample needs before its time counts")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    metric = METRICS[a.metric]
    csv_path = Path(a.runs_dir) / a.run / "metrics_log.csv"
    model_dir = Path(a.models_dir) / a.run
    best_json = model_dir / "best.json"
    # A best chosen by the other metric is not comparable (7.16 kills/min would outrank any completion rate), and
    # overwriting it would silently replace that run's best.zip, so refuse instead.
    held = stored_penalty_name(best_json)
    if held is not None and held != metric.penalty:
        ap.error(f"{best_json} holds a best chosen with the {held!r} tie-break, not {metric.penalty!r}: "
                 f"pass the --metric that run was scored with")
    best = stored_best(best_json)

    while True:
        series = scored(csv_path, a.metric, min_rate=a.min_rate)
        if series:
            best = save_if_better(series, model_dir, a.metric, best)
            newest = series[-1]
            # Warn when the current policy has fallen well below the best: that is the signal to roll back.
            # Measured on the RAW column, so a negated score (`--metric time`) reads the same way round as the
            # others: 15% off the best time is 15% off, whichever sign it is stored with.
            raw_best = metric.score_sign * best[0] if best is not None else None
            raw_new = metric.score_sign * newest[0]
            worse = raw_best is not None and raw_best > 0 and (
                raw_new < raw_best * 0.85 if metric.score_sign > 0 else raw_new > raw_best / 0.85)
            if worse:
                print(f"[keep_best] WARNING current {raw_new:.2f} {metric.unit} is "
                      f"{abs(1 - raw_new/raw_best)*100:.0f}% off best {raw_best:.2f} "
                      f"(best.zip holds the good weights)", flush=True)
        if a.once:
            if best_json.exists():
                print(best_json.read_text(encoding="utf-8"))
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
