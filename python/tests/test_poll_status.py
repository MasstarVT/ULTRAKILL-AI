"""poll_status.py's metrics_log.csv header handling. No game needed:  python tests/test_poll_status.py  (or pytest).

The one behaviour worth pinning here is the one that failed silently on 2026-09-20: rows are written with
`extrasaction="ignore"`, so a value for a column the existing header lacks is DROPPED WITHOUT A WORD. That
turned the Brutal switch's `difficulty` column into nothing, and `keep_best.py --metric time` -- whose whole
difficulty clause reads that column -- would have kept `best.zip` on Violent-trained weights for the entire
Brutal round.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import poll_status  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def test_a_log_that_already_has_every_column_is_left_exactly_as_it_is():
    """The ordinary case, and the one that must not rotate: no column is dropped, so there is nothing to fix."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics_log.csv"
        wanted = list(poll_status.row({}).keys())
        write_csv(path, wanted, [{"timesteps": 1000}])
        header, needs_header = poll_status.open_log(path, wanted)
        assert header == wanted and needs_header is False
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["metrics_log.csv"], "nothing was moved"


def test_extra_columns_alone_never_rotate_a_log():
    """A header with columns we no longer write drops nothing, so the history is kept rather than rotated."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics_log.csv"
        wanted = list(poll_status.row({}).keys())
        write_csv(path, wanted + ["a_column_we_stopped_writing"], [{"timesteps": 1000}])
        header, needs_header = poll_status.open_log(path, wanted)
        assert needs_header is False
        assert "a_column_we_stopped_writing" in header, "the column is kept, and simply written empty"
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["metrics_log.csv"]


def test_a_log_missing_the_difficulty_column_is_rotated_aside_and_restarted():
    """THE 2026-09-20 failure. A pre-switch header silently swallowed every `difficulty` value.

    The live `runs/spec_0-1_speed/metrics_log.csv` was in exactly this state: 3,292 rows whose header ended
    `fresh_window,fresh_completion_rate,median_time_50,best_time`. Rotating is what makes the column real
    without an operator remembering to move a file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics_log.csv"
        wanted = list(poll_status.row({}).keys())
        assert "difficulty" in wanted, "poll_status writes it; that is the whole point"
        old_header = [c for c in wanted if c != "difficulty"]
        write_csv(path, old_header, [{"timesteps": 1000}, {"timesteps": 2000}])

        assert poll_status.missing_columns(old_header, wanted) == ["difficulty"]
        header, needs_header = poll_status.open_log(path, wanted)
        assert header == wanted and needs_header is True, "the new log takes the full header"

        moved = [p for p in Path(tmp).iterdir() if p.name != "metrics_log.csv"]
        assert len(moved) == 1, moved
        assert not path.exists(), "the new log is not written until the first row"
        kept_header, kept_rows = read_csv(moved[0])
        assert "difficulty" not in kept_header and len(kept_rows) == 2, "the old log is KEPT, not deleted"


def test_the_difficulty_a_row_carries_survives_the_rotation():
    """End to end: the same status.json that lost its difficulty against the old header keeps it after."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics_log.csv"
        wanted = list(poll_status.row({}).keys())
        write_csv(path, [c for c in wanted if c != "difficulty"], [{"timesteps": 1000}])
        status = {"timesteps": 2000, "campaign": {"difficulty": 4, "fresh_window": 31,
                                                  "fresh_completion_rate": 0.55, "median_time_50": 160.0}}

        header, needs_header = poll_status.open_log(path, wanted)
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if needs_header:
                w.writeheader()
            w.writerow(poll_status.row(status))
        _, rows = read_csv(path)
        assert rows[0]["difficulty"] == "4", "the Brutal tag reaches the log keep_best reads"
        assert rows[0]["median_time_50"] == "160.0"


def test_rotating_twice_in_the_same_second_does_not_overwrite_the_first_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics_log.csv"
        write_csv(path, ["timesteps"], [{"timesteps": 1}])
        first = poll_status.rotate_aside(path)
        write_csv(path, ["timesteps"], [{"timesteps": 2}])
        second = poll_status.rotate_aside(path)
        assert first != second and first.exists() and second.exists()
        assert read_csv(first)[1][0]["timesteps"] == "1"
        assert read_csv(second)[1][0]["timesteps"] == "2"


def test_the_campaign_row_carries_the_difficulty_at_all():
    """`row()` has to lift `campaign.difficulty` out of status.json, or none of the above matters."""
    assert poll_status.row({"campaign": {"difficulty": 4}})["difficulty"] == 4
    assert poll_status.row({"campaign": {}})["difficulty"] is None
    assert poll_status.row({})["difficulty"] is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
