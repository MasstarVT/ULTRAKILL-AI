"""times.md bookkeeping. No game needed:  python tests/test_times.py  (or pytest)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.times import (  # noqa: E402
    TimeEntry, format_delta, format_time, parse_time, record, record_file, valid_official_seconds)

# times.md at the repo root, exactly as committed before any campaign run.
TIMES_MD = """# Times

Leaderboard of the AI's level completion times across training generations.
Times are in-game level time (`mm:ss.mmm`), not wall-clock training time.

## Leaderboard

Best time per level. A generation only takes a spot by beating the current record.

| Level | Time | Rank | Generation | Difficulty | Date | Notes |
|-------|------|------|------------|------------|------|-------|
| — | — | — | — | — | — | No completed runs yet |

## Generation history

Best run from each generation, newest first. Keep every generation here, even ones that didn't set a record.

| Generation | Level | Time | Rank | Kills | Deaths | Δ vs previous | Date | Notes |
|------------|-------|------|------|-------|--------|---------------|------|-------|
| — | — | — | — | — | — | — | — | No generations trained yet |

<!--
How to add an entry:
- Generation history: add a row at the top of the table for the new generation's best run.
- Leaderboard: if that run beats the level's record, replace the level's row (one row per level, sorted by level order).
- Rank is the in-game style rank (D, C, B, A, S, P). Δ vs previous is the time change from the previous generation on the same level, e.g. -1.250s.
-->
"""


def entry(level="Level 0-1", seconds=83.25, rank="S", generation="campaign_ppo@1.00M", kills=12, deaths=1,
          notes="3/5 eval runs completed") -> TimeEntry:
    return TimeEntry(level=level, seconds=seconds, rank=rank, generation=generation, difficulty=3, date="2026-09-20",
                     kills=kills, deaths=deaths, notes=notes)


def table_rows(markdown: str, heading: str) -> list[str]:
    """Data rows (below the header and separator) of the table under a heading."""
    lines = markdown.splitlines()
    i = lines.index(heading) + 1
    while not lines[i].startswith("|"):
        i += 1
    end = i
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return lines[i + 2:end]


def column(rows: list[str], index: int) -> list[str]:
    return [row.strip().strip("|").split("|")[index].strip() for row in rows]


def skeleton(markdown: str) -> list[str]:
    """Every line except table data rows: headings, prose, table headers and the closing comment."""
    out, position = [], 0
    for line in markdown.splitlines():
        position = position + 1 if line.startswith("|") else 0
        if position <= 2:
            out.append(line)
    return out


def test_first_record_replaces_both_placeholders():
    out = record(TIMES_MD, entry())
    assert table_rows(out, "## Leaderboard") == [
        "| 0-1 | 01:23.250 | S | campaign_ppo@1.00M | Violent | 2026-09-20 | 3/5 eval runs completed |"]
    assert table_rows(out, "## Generation history") == [
        "| campaign_ppo@1.00M | 0-1 | 01:23.250 | S | 12 | 1 | — | 2026-09-20 | 3/5 eval runs completed |"]
    assert "No completed runs yet" not in out and "No generations trained yet" not in out
    assert skeleton(out) == skeleton(TIMES_MD)
    assert record(TIMES_MD.replace("\n", "\r\n"), entry()) == out  # a CRLF checkout gives the same result


def test_slower_run_keeps_the_record():
    first = record(TIMES_MD, entry())
    out = record(first, entry(seconds=83.75, rank="A", generation="campaign_ppo@2.00M", kills=10, deaths=2,
                              notes="4/5 eval runs completed"))
    assert table_rows(out, "## Leaderboard") == table_rows(first, "## Leaderboard")
    assert table_rows(out, "## Generation history") == [
        "| campaign_ppo@2.00M | 0-1 | 01:23.750 | A | 10 | 2 | +0.500s | 2026-09-20 | 4/5 eval runs completed |",
        "| campaign_ppo@1.00M | 0-1 | 01:23.250 | S | 12 | 1 | — | 2026-09-20 | 3/5 eval runs completed |",
    ]
    tie = record(first, entry(generation="campaign_ppo@2.00M"))
    assert table_rows(tie, "## Leaderboard") == table_rows(first, "## Leaderboard")  # only a faster time takes the spot
    assert column(table_rows(tie, "## Generation history"), 6) == ["+0.000s", "—"]


def test_faster_run_takes_the_record():
    slower = record(record(TIMES_MD, entry()), entry(seconds=83.75, generation="campaign_ppo@2.00M"))
    out = record(slower, entry(seconds=82.0, generation="campaign_ppo@3.00M", kills=11, deaths=0,
                               notes="5/5 eval runs completed"))
    assert table_rows(out, "## Leaderboard") == [
        "| 0-1 | 01:22.000 | S | campaign_ppo@3.00M | Violent | 2026-09-20 | 5/5 eval runs completed |"]
    history = table_rows(out, "## Generation history")
    assert len(history) == 3
    # Against the newest earlier run on the level (83.75 s), not against the record (83.25 s).
    assert history[0] == "| campaign_ppo@3.00M | 0-1 | 01:22.000 | S | 11 | 0 | -1.750s | 2026-09-20 | 5/5 eval runs completed |"


def test_leaderboard_stays_in_level_order():
    out = TIMES_MD
    for level, seconds in (("Level 0-2", 95.5), ("Level 1-1", 200.0), ("Level 0-1", 83.25)):
        out = record(out, entry(level=level, seconds=seconds))
    assert column(table_rows(out, "## Leaderboard"), 0) == ["0-1", "0-2", "1-1"]
    history = table_rows(out, "## Generation history")
    assert column(history, 1) == ["0-1", "1-1", "0-2"]  # newest first
    assert column(history, 6) == ["—", "—", "—"]  # no earlier run on any of these levels


def test_format_and_parse_time_round_trip():
    assert format_time(83.25) == "01:23.250"
    assert format_time(0.0) == "00:00.000" and format_time(59.9996) == "01:00.000"
    assert format_time(3725.5) == "62:05.500"
    for seconds in (0.0, 7.5, 83.25, 599.999, 3725.5):
        assert abs(parse_time(format_time(seconds)) - seconds) < 1e-9
    for text in ("00:00.000", "01:23.250", "09:59.999", "62:05.500"):
        assert format_time(parse_time(text)) == text
    assert parse_time("—") is None and parse_time("") is None and parse_time("1:2:3") is None
    assert format_delta(-1.25) == "-1.250s" and format_delta(0.5) == "+0.500s"


def test_html_comment_survives():
    comment = TIMES_MD[TIMES_MD.index("<!--"):]
    out = TIMES_MD
    for seconds in (90.0, 85.0, 88.0):
        out = record(out, entry(seconds=seconds))
    assert out.endswith(comment) and out.count("<!--") == 1
    assert skeleton(out) == skeleton(TIMES_MD)


PLACEHOLDER = "| — | — | — | — | — | — | No completed runs yet |"
BOGUS_ROW = "| 0-2 | 00:00.000 | A | spec_0-2_speed@18.80M | Violent | 2026-09-19 | training episode |"


def test_a_time_the_game_never_reported_is_refused():
    """2026-09-19: a completion frame read after the level stats reset produced a 00:00.000 leaderboard row."""
    for seconds in (0.0, 1.0, -5.0, float("nan"), float("inf")):
        try:
            record(TIMES_MD, entry(seconds=seconds))
        except ValueError:
            continue
        raise AssertionError(f"{seconds!r} was recorded as an official time")
    assert "01:23.250" in record(TIMES_MD, entry(seconds=83.25)), "a real time still records"
    assert valid_official_seconds(1.001) == 1.001 and valid_official_seconds(6.6) == 6.6, "the human IL floor"
    assert valid_official_seconds(None) is None and valid_official_seconds("") is None


def test_an_invalid_leaderboard_row_is_replaced_by_a_real_time():
    """Nothing is faster than zero, so without this the bogus row would hold the level's spot forever."""
    poisoned = TIMES_MD.replace(PLACEHOLDER, BOGUS_ROW)
    out = record(poisoned, entry(level="Level 0-2", seconds=123.5368, rank="S", generation="spec_0-2_speed@19.00M"))
    rows = table_rows(out, "## Leaderboard")
    assert len(rows) == 1 and "00:00.000" not in rows[0]
    assert "| 0-2 | 02:03.537 | S | spec_0-2_speed@19.00M |" in rows[0]
    # ... and a valid held row still wins against a slower run, exactly as before.
    kept = record(out, entry(level="Level 0-2", seconds=200.0, rank="B"))
    assert "| 0-2 | 02:03.537 |" in table_rows(kept, "## Leaderboard")[0]


def test_record_file_rewrites_in_place():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "times.md"
        path.write_text(TIMES_MD, encoding="utf-8")
        record_file(path, entry())
        assert path.read_text(encoding="utf-8") == record(TIMES_MD, entry())


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
