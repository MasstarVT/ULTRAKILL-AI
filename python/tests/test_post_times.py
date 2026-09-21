"""post_times.py: a run's best official times into times.md. No game needed:  python tests/test_post_times.py"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import post_times  # noqa: E402
from test_times import TIMES_MD  # noqa: E402


def make_run(tmp: Path, seconds: float, steps: int = 5_571_302, difficulty=3) -> tuple[Path, Path]:
    run = tmp / "runs" / "campaign_gates"
    (run / "best_runs").mkdir(parents=True, exist_ok=True)
    best = {"level": "Level 0-1", "seconds": seconds, "kills": 64, "deaths": 1, "rank": "B",
            "difficulty": difficulty, "saved_at": "2026-09-17 06:35:52", "positions": [[0, 0, 0]]}
    if difficulty is None:
        best.pop("difficulty")
    (run / "best_runs" / "Level_0-1.json").write_text(json.dumps(best), encoding="utf-8")
    rows = [
        {"completed": True, "fresh_start": False, "level": "Level 0-1", "level_seconds": None, "timesteps": 1},
        {"completed": True, "fresh_start": True, "level": "Level 0-1", "level_seconds": 522.1206, "timesteps": 5_392_862},
        {"completed": True, "fresh_start": True, "level": "Level 0-1", "level_seconds": seconds, "timesteps": steps},
    ]
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    times = tmp / "times.md"
    if not times.exists():
        times.write_text(TIMES_MD, encoding="utf-8")
    return run, times


def test_first_post_fills_both_tables_with_the_step_count():
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 440.7143)
        posted = post_times.post(run, times, "campaign_gates")
        text = times.read_text(encoding="utf-8")
        assert len(posted) == 1 and "07:20.714" in posted[0]
        assert text.count("| 0-1 | 07:20.714 | B | campaign_gates@5.57M | Violent | 2026-09-17 |") == 1
        assert "| campaign_gates@5.57M | 0-1 | 07:20.714 | B | 64 | 1 |" in text
        assert "No completed runs yet" not in text


def test_posting_again_adds_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 440.7143)
        post_times.post(run, times, "campaign_gates")
        before = times.read_text(encoding="utf-8")
        assert post_times.post(run, times, "campaign_gates") == []
        assert times.read_text(encoding="utf-8") == before


def test_only_a_faster_best_run_is_posted():
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 440.7143)
        post_times.post(run, times, "campaign_gates")
        make_run(Path(tmp), 300.5, steps=7_000_000)
        posted = post_times.post(run, times, "campaign_gates")
        text = times.read_text(encoding="utf-8")
        assert len(posted) == 1
        assert "| 0-1 | 05:00.500 | B | campaign_gates@7.00M |" in text  # leaderboard row replaced
        assert "| campaign_gates@5.57M | 0-1 | 07:20.714 |" in text  # history keeps the older generation
        assert "-140.214s" in text


def test_missing_episode_row_still_posts_without_a_step_count():
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 440.7143)
        (run / "episodes.jsonl").unlink()
        posted = post_times.post(run, times, "campaign_gates")
        assert len(posted) == 1 and "(campaign_gates)" in posted[0]


def test_a_best_run_with_no_official_time_is_never_posted():
    """The 2026-09-19 bug in one line: `best_runs/<level>.json` held a 0.0 s completion of a 4,120-step episode."""
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 0.0)
        before = times.read_text(encoding="utf-8")
        assert post_times.post(run, times, "spec_0-2_speed") == []
        assert times.read_text(encoding="utf-8") == before
        assert "00:00.000" not in times.read_text(encoding="utf-8")


def test_a_real_time_replaces_an_invalid_leaderboard_row():
    """Only a FASTER time is ever posted, so a 00:00.000 row has to read as no row at all."""
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 300.5, steps=7_000_000)
        bogus = "| 0-1 | 00:00.000 | A | spec_0-1_speed@18.80M | Violent | 2026-09-19 | training episode |"
        times.write_text(TIMES_MD.replace("| — | — | — | — | — | — | No completed runs yet |", bogus),
                         encoding="utf-8")
        assert post_times.held_time(times.read_text(encoding="utf-8"), "Level 0-1") is None
        posted = post_times.post(run, times, "campaign_gates")
        text = times.read_text(encoding="utf-8")
        assert len(posted) == 1 and "| 0-1 | 05:00.500 | B | campaign_gates@7.00M |" in text
        assert "| 0-1 | 00:00.000 |" not in text


def test_push_stages_only_times_md():
    calls = []

    class Done:
        returncode = 0

    def fake_run(cmd, check=False):
        calls.append(cmd)
        return Done()

    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.md"
        times.write_text(TIMES_MD, encoding="utf-8")
        assert post_times.push_times(times, ["0-1: 03:03.628 rank A (campaign_gates@6.85M)"], run=fake_run)
    add, commit, push = calls
    assert add[-2:] == ["--", "times.md"] and "-A" not in add and "." not in add
    assert commit[-2:] == ["--", "times.md"] and "0-1: 03:03.628" in commit[commit.index("-m") + 1]
    assert push[-3:] == ["push", "origin", "HEAD"]


def test_a_slower_brutal_best_run_is_posted_over_a_faster_violent_row():
    """post_times must apply the SAME rule as times.record, or the switch to Brutal posts nothing for a round.

    Before the difficulty clause this was the blocking case: `spec_0-1_speed` holds a 81.5 s Violent row, and
    every early Brutal completion is slower, so a difficulty-blind "only a faster time is posted" would have
    left times.md claiming Violent for as long as the Brutal round ran.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 81.46, difficulty=3)
        assert post_times.post(run, times, "campaign_gates"), "the Violent row is posted first"
        assert "| 0-1 | 01:21.460 | B | campaign_gates@5.57M | Violent |" in times.read_text(encoding="utf-8")
        held_difficulty, held_seconds = post_times.held_record(times.read_text(encoding="utf-8"), "Level 0-1")
        assert held_difficulty == 3 and abs(held_seconds - 81.46) < 1e-6

        run, times = make_run(Path(tmp), 150.0, difficulty=4)
        posted = post_times.post(run, times, "campaign_gates")
        text = times.read_text(encoding="utf-8")
        assert len(posted) == 1 and "Brutal" in posted[0], posted
        assert "| 0-1 | 02:30.000 |" in text and "| Brutal |" in text
        assert post_times.held_record(text, "Level 0-1") == (4, 150.0)


def test_a_faster_violent_best_run_is_not_posted_over_a_brutal_row():
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 150.0, difficulty=4)
        post_times.post(run, times, "campaign_gates")
        held = times.read_text(encoding="utf-8")

        run, times = make_run(Path(tmp), 60.0, difficulty=3)
        assert post_times.post(run, times, "campaign_gates") == [], "an easier difficulty never takes the row"
        assert times.read_text(encoding="utf-8") == held, "and nothing at all is written, not even the history"


def test_a_best_run_without_a_difficulty_is_not_recorded_as_violent():
    """It used to default to 3. A row that says Violent because nobody asked is the bug being fixed."""
    with tempfile.TemporaryDirectory() as tmp:
        run, times = make_run(Path(tmp), 440.7143, difficulty=None)
        post_times.post(run, times, "campaign_gates")
        row = next(line for line in times.read_text(encoding="utf-8").splitlines()
                   if line.startswith("| 0-1 |"))
        assert "Violent" not in row, row
        assert post_times.held_record(times.read_text(encoding="utf-8"), "Level 0-1")[0] == -1


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
