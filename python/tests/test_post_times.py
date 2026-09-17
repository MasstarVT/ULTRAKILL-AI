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


def make_run(tmp: Path, seconds: float, steps: int = 5_571_302) -> tuple[Path, Path]:
    run = tmp / "runs" / "campaign_gates"
    (run / "best_runs").mkdir(parents=True, exist_ok=True)
    best = {"level": "Level 0-1", "seconds": seconds, "kills": 64, "deaths": 1, "rank": "B", "difficulty": 3,
            "saved_at": "2026-09-17 06:35:52", "positions": [[0, 0, 0]]}
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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
