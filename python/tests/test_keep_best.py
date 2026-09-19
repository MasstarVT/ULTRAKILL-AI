"""keep_best.py scoring for Cyber Grind and the campaign. No game needed:  python tests/test_keep_best.py  (or pytest)."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import keep_best  # noqa: E402

# The metrics_log.csv columns keep_best reads, as poll_status.py writes them.
COLUMNS = ["wall_time", "timesteps", "episodes", "window", "reward", "kills_per_min", "deaths",
           "fresh_window", "fresh_completion_rate", "median_time_50", "best_time"]

# models/cybergrind_ppo_v2/best.json as keep_best.py wrote it before --metric existed.
OLD_BEST_JSON = {
    "score_metric": "kills_per_min, smoothed over 9 samples",
    "score": 7.1567,
    "deaths": 1.5478,
    "reward": 231.472,
    "at_timesteps": 4697515.0,
    "checkpoint": "ckpt_4679085_steps.zip",
    "saved_at": "2026-09-16 04:22:09",
}


def write_log(folder: Path, rows: list[dict]) -> Path:
    """A metrics_log.csv the way poll_status.py writes it: None and missing values become empty cells."""
    path = folder / "metrics_log.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def grind_rows(kills_per_min: list[float], window: int = 100) -> list[dict]:
    return [{"timesteps": 10_000 * (i + 1), "window": window, "reward": 100.0 + i, "kills_per_min": k, "deaths": 2.0 - 0.1 * i}
            for i, k in enumerate(kills_per_min)]


def campaign_rows(rates: list[float], best_times: list[float | None], fresh_window: int = 50,
                  median_times: list[float | None] | None = None) -> list[dict]:
    """`median_time_50` defaults to `best_time`, which is what a run that never varies would write.

    The two are separate columns because they are separate statistics: `best_time` is the run's lifetime
    minimum and `median_time_50` is the median over the completions in the last 50 fresh episodes. Every test
    that cares about the difference passes both.
    """
    medians = list(median_times) if median_times is not None else list(best_times)
    return [{"timesteps": 10_000 * (i + 1), "window": 100, "reward": 10.0 * i, "fresh_window": fresh_window,
             "fresh_completion_rate": rate, "best_time": best_time, "median_time_50": median}
            for i, (rate, best_time, median) in enumerate(zip(rates, best_times, medians))]


def test_kills_per_min_needs_a_full_window():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_log(Path(tmp), grind_rows([5.0] * 12, window=99))
        assert keep_best.scored(path, "kills_per_min") == []
        path = write_log(Path(tmp), grind_rows([5.0] * 8, window=99) + grind_rows([6.0] * 9))
        series = keep_best.scored(path, "kills_per_min")
        assert len(series) == 1 and abs(series[0][0] - 6.0) < 1e-9
        assert keep_best.scored(path) == series  # kills_per_min stays the default


def test_kills_per_min_smoothing_is_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        series = keep_best.scored(write_log(Path(tmp), grind_rows([float(k) for k in range(1, 12)])), "kills_per_min")
        assert [round(s[0], 9) for s in series] == [5.0, 6.0, 7.0]  # 11 samples give 3 full 9-sample windows
        score, deaths, reward, timesteps = series[0]
        assert abs(deaths - 1.6) < 1e-9  # mean of 2.0 - 0.1 * i over i = 0..8
        assert abs(reward - 104.0) < 1e-9 and timesteps == 50_000  # timesteps of the window's middle sample


def test_campaign_needs_twenty_fresh_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        assert keep_best.scored(write_log(Path(tmp), campaign_rows([0.5] * 12, [80.0] * 12, fresh_window=19)), "campaign") == []
        ready = campaign_rows([0.5] * 9, [80.0] * 9, fresh_window=20)
        for row in ready:
            row["window"] = 3  # the 100-episode gate belongs to kills_per_min only
        path = write_log(Path(tmp), ready)
        series = keep_best.scored(path, "campaign")
        assert len(series) == 1 and abs(series[0][0] - 0.5) < 1e-9 and abs(series[0][1] - 80.0) < 1e-9
        assert keep_best.scored(path, "kills_per_min") == []


def test_campaign_missing_best_time_is_infinite():
    with tempfile.TemporaryDirectory() as tmp:
        series = keep_best.scored(write_log(Path(tmp), campaign_rows([0.0] * 9, [None] * 9)), "campaign")
        assert len(series) == 1 and series[0][0] == 0.0 and math.isinf(series[0][1])


def test_higher_completion_rate_wins():
    assert keep_best.is_better((0.6, 120.0), (0.5, 80.0))  # a slower best time does not protect a lower rate
    assert not keep_best.is_better((0.4, 60.0), (0.5, 80.0))
    assert keep_best.is_better((0.0, math.inf), None)  # nothing stored yet
    series = [(0.5, 80.0, 0.0, 1.0), (0.6, 120.0, 0.0, 2.0), (0.4, 60.0, 0.0, 3.0)]
    assert keep_best.best_of(series)[3] == 2.0


def test_equal_rate_lower_best_time_wins():
    assert keep_best.is_better((0.5, 75.0), (0.5, 80.0))
    assert not keep_best.is_better((0.5, 85.0), (0.5, 80.0))
    assert not keep_best.is_better((0.5, 80.0), (0.5, 80.0))  # strictly better only
    assert keep_best.is_better((0.5, 80.0), (0.5, math.inf))  # the first finite time beats none
    assert not keep_best.is_better((0.5, math.inf), (0.5, math.inf))
    series = [(0.5, 90.0, 0.0, 1.0), (0.5, 75.0, 0.0, 2.0), (0.5, math.inf, 0.0, 3.0)]
    assert keep_best.best_of(series)[3] == 2.0
    with tempfile.TemporaryDirectory() as tmp:
        rows = campaign_rows([0.5] * 18, [90.0] * 9 + [75.0] * 9)
        best = keep_best.best_of(keep_best.scored(write_log(Path(tmp), rows), "campaign"))
        assert abs(best[1] - 75.0) < 1e-9 and best[3] == 140_000  # the last window is all 75 s


def test_old_best_json_reads_deaths_as_penalty():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.json"
        path.write_text(json.dumps(OLD_BEST_JSON, indent=2), encoding="utf-8")
        stored = keep_best.stored_best(path)
        assert stored == (7.1567, 1.5478)
        assert keep_best.is_better((7.1567, 1.40), stored) and not keep_best.is_better((7.1567, 1.60), stored)
        assert not keep_best.is_better((7.15674, 1.5478), stored)  # compared at the precision best.json keeps
        assert keep_best.stored_best(Path(tmp) / "missing.json") is None
        assert keep_best.stored_penalty_name(path) == "deaths"  # so --metric kills_per_min still accepts it
        assert keep_best.stored_penalty_name(Path(tmp) / "missing.json") is None
        # best_of ranks at the same precision: a score higher only past 4 decimals does not hide a lower penalty.
        assert keep_best.best_of([(7.15674, 1.5478, 0.0, 1.0), (7.1567, 1.40, 0.0, 2.0)])[3] == 2.0


def test_new_best_copies_checkpoint_and_records_penalty():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "ckpt_100000_steps.zip").write_bytes(b"early")
        (model_dir / "ckpt_200000_steps.zip").write_bytes(b"late")
        series = [(0.25, 95.0, 40.0, 150_000.0), (0.5, 81.5, 60.0, 230_000.0)]
        assert keep_best.save_if_better(series, model_dir, "campaign", None) == (0.5, 81.5)
        assert (model_dir / "best.zip").read_bytes() == b"late"  # newest checkpoint at or before 230k steps
        saved = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
        assert set(saved) == {"score_metric", "score", "penalty", "penalty_name", "reward", "at_timesteps", "checkpoint", "saved_at"}
        assert saved["score_metric"] == "fresh_completion_rate, smoothed over 9 samples"
        assert saved["score"] == 0.5 and saved["penalty"] == 81.5 and saved["penalty_name"] == "best_time"
        assert saved["reward"] == 60.0 and saved["at_timesteps"] == 230_000.0 and saved["checkpoint"] == "ckpt_200000_steps.zip"
        assert keep_best.stored_best(model_dir / "best.json") == (0.5, 81.5)
        assert keep_best.stored_penalty_name(model_dir / "best.json") == "best_time"


# ---------------------------------------------------------------------------
# --metric time: the speed stage (docs/superpowers/specs/2026-09-18-speed-stages.md §5)
# ---------------------------------------------------------------------------


def test_time_needs_a_minimum_completion_rate_behind_it():
    """A 40-second time set by the one lucky load in a hundred is not a policy. The gate is the whole metric."""
    with tempfile.TemporaryDirectory() as tmp:
        cold = campaign_rows([0.05] * 12, [42.0] * 12)
        assert keep_best.scored(write_log(Path(tmp), cold), "time") == []
        # ... and the window gate is still there: a rate over the bar on 19 fresh episodes does not count.
        thin = campaign_rows([0.6] * 12, [42.0] * 12, fresh_window=19)
        assert keep_best.scored(write_log(Path(tmp), thin), "time") == []
        ready = campaign_rows([0.6] * 9, [42.0] * 9, fresh_window=20)
        series = keep_best.scored(write_log(Path(tmp), ready), "time")
        assert len(series) == 1
        assert abs(series[0][0] + 42.0) < 1e-9, "the score is the NEGATED time, so higher is still better"
        assert abs(series[0][1] + 0.6) < 1e-9, "and the tie-break is the negated rate, so higher wins"
        assert keep_best.scored(write_log(Path(tmp), ready), "time", min_rate=0.9) == []


def test_a_row_with_no_best_time_never_scores_on_time():
    """Before the first completion there is no time to rank, whatever the rate says."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = campaign_rows([0.5] * 12, [None] * 12)
        assert keep_best.scored(write_log(Path(tmp), rows), "time") == []


def test_time_ranks_the_policys_median_and_not_the_runs_lifetime_best():
    """The 2026-09-18 review: `best_time` only ever falls, so scoring it makes `--metric time` inert.

    After one lucky load sets the record every later sample ties at the same score, `rank_key` falls through
    to the tie-break -- the completion rate -- and `best.zip` tracks the most RELIABLE checkpoint while the
    policy gets slower. `median_time_50` is a real per-sample statistic and moves in both directions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # One 88 s load early on, then a policy that is reliable and slow, then one that is slightly less
        # reliable and much faster. `best_time` is 88 for every row after the record, exactly as the live
        # ProgressCallback writes it.
        rates = [0.31] * 9 + [0.52] * 9 + [0.45] * 9
        best = [250.0] * 4 + [88.0] * 23
        medians = [150.0] * 9 + [240.0] * 9 + [120.0] * 9
        path = write_log(Path(tmp), campaign_rows(rates, best, median_times=medians))
        series = keep_best.scored(path, "time")
        scores = sorted({round(s[0], 4) for s in series})
        assert len(scores) > 1, "a lifetime minimum would make every post-record sample score the same"
        fastest = keep_best.best_of(series)
        assert abs(fastest[0] + 120.0) < 1e-6, "the 120 s median wins, not the 240 s one behind the same best"
        # ... and the degradation warning is reachable again: a later, slower sample IS worse than the stored
        # best, which can never happen while the score column is a running minimum.
        assert keep_best.is_better((-120.0, -0.45), (-240.0, -0.52))
        assert not keep_best.is_better((-240.0, -0.52), (-120.0, -0.45))


def test_a_median_the_game_cannot_have_produced_never_scores_on_time():
    """metrics_log.csv is append-only: rows written before the 2026-09-19 fix still hold impossible times.

    `--metric time` ranks on the LOWEST median, so a zero would be the unbeatable best for the rest of the
    run -- and it would drag every smoothing window that contains it down with it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rows = campaign_rows([0.6] * 10, [88.0] * 10, median_times=[150.0] * 5 + [0.0] + [150.0] * 4)
        series = keep_best.scored(write_log(Path(tmp), rows), "time")
        assert len(series) == 1 and abs(series[0][0] + 150.0) < 1e-9, "the zero sample is dropped, not averaged"
        allzero = campaign_rows([0.6] * 12, [88.0] * 12, median_times=[0.0] * 12)
        assert keep_best.scored(write_log(Path(tmp), allzero), "time") == []
        # `--metric campaign` breaks ties on `best_time`, where a 0.0 would be the best tie-break forever.
        camp = campaign_rows([0.5] * 9, [0.0] * 9)
        series = keep_best.scored(write_log(Path(tmp), camp), "campaign")
        assert len(series) == 1 and series[0][1] == math.inf, "an impossible best time reads as no time at all"


def test_a_faster_time_wins_and_the_rate_breaks_the_tie():
    assert keep_best.is_better((-118.0, -0.5), (-131.0, -0.5)), "118 s beats 131 s"
    assert not keep_best.is_better((-140.0, -0.9), (-131.0, -0.5)), "a higher rate does not buy a slower time"
    assert keep_best.is_better((-131.0, -0.7), (-131.0, -0.5)), "same time, the higher rate wins"
    assert not keep_best.is_better((-131.0, -0.5), (-131.0, -0.5))  # strictly better only
    series = [(-131.0, -0.5, 0.0, 1.0), (-118.0, -0.4, 0.0, 2.0), (-125.0, -0.9, 0.0, 3.0)]
    assert keep_best.best_of(series)[3] == 2.0, "the fastest sample, not the most reliable one"


def test_time_saves_the_checkpoint_and_guards_the_other_metrics_best_json():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "ckpt_100000_steps.zip").write_bytes(b"early")
        (model_dir / "ckpt_200000_steps.zip").write_bytes(b"late")
        series = [(-131.25, -0.42, 40.0, 150_000.0), (-118.50, -0.44, 60.0, 230_000.0)]
        assert keep_best.save_if_better(series, model_dir, "time", None) == (-118.5, -0.44)
        assert (model_dir / "best.zip").read_bytes() == b"late"
        saved = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
        assert saved["score"] == -118.5 and saved["penalty"] == -0.44
        assert saved["penalty_name"] == "fresh_completion_rate"
        assert saved["score_metric"] == "-median_time_50 (lower is better), smoothed over 9 samples"
        # The cross-metric guard: this file may not be read, or overwritten, by the campaign metric.
        assert keep_best.stored_penalty_name(model_dir / "best.json") == "fresh_completion_rate"
        assert keep_best.METRICS["campaign"].penalty == "best_time" != "fresh_completion_rate"
        assert keep_best.METRICS["time"].penalty == "fresh_completion_rate"
        # ... and a slower later run does not move best.zip.
        assert keep_best.save_if_better([(-140.0, -0.9, 0.0, 240_000.0)], model_dir, "time",
                                        (-118.5, -0.44)) == (-118.5, -0.44)


def test_the_campaign_and_kills_metrics_are_untouched_by_the_signs():
    """Both signs default to +1, so `scored` returns exactly what it always did for the other two metrics."""
    for name in ("kills_per_min", "campaign"):
        m = keep_best.METRICS[name]
        assert (m.score_sign, m.penalty_sign) == (1.0, 1.0) and m.gate == ""
    with tempfile.TemporaryDirectory() as tmp:
        rows = campaign_rows([0.5] * 9, [80.0] * 9)
        series = keep_best.scored(write_log(Path(tmp), rows), "campaign")
        assert len(series) == 1 and abs(series[0][0] - 0.5) < 1e-9 and abs(series[0][1] - 80.0) < 1e-9


def test_restart_does_not_resave_the_same_best():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "ckpt_50000_steps.zip").write_bytes(b"weights")
        series = [(0.0, math.inf, -30.0, 60_000.0)]  # no completed run yet
        keep_best.save_if_better(series, model_dir, "campaign", None)
        saved = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
        assert saved["penalty"] is None and saved["penalty_name"] == "best_time"  # strict JSON has no infinity
        stored = keep_best.stored_best(model_dir / "best.json")
        assert stored[0] == 0.0 and math.isinf(stored[1])
        (model_dir / "best.zip").unlink()
        assert keep_best.save_if_better(series, model_dir, "campaign", stored) == stored
        assert not (model_dir / "best.zip").exists()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
