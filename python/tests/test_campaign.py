"""Campaign helpers (ultrakill_ai/campaign.py). No game needed:  python tests/test_campaign.py  (or pytest)."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CAMPAIGN_LEVELS,
    RANK_LETTERS,
    ExplorationArchive,
    compute_rank,
    grade,
    safe_name,
)

RANKS = {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}


def test_campaign_levels():
    assert len(CAMPAIGN_LEVELS) == 35 and len(set(CAMPAIGN_LEVELS)) == 35
    assert CAMPAIGN_LEVELS[0] == "Level 0-1"
    assert CAMPAIGN_LEVELS[4] == "Level 0-5" and CAMPAIGN_LEVELS[5] == "Level 1-1"
    assert CAMPAIGN_LEVELS[14] == "Level 3-2"
    assert CAMPAIGN_LEVELS[-1] == "Level 9-2"


def test_safe_name():
    assert safe_name("Level 0-1") == "Level_0-1"
    assert safe_name("Level 3-2") == "Level_3-2"
    assert safe_name("a  b/c") == "a_b_c"


def test_grade_counts_thresholds_in_order():
    assert grade([10, 20, 30, 40], 0, reverse=False) == 0
    assert grade([10, 20, 30, 40], 10, reverse=False) == 1  # meeting a threshold exactly counts
    assert grade([10, 20, 30, 40], 25, reverse=False) == 2
    assert grade([300, 240, 180, 120], 301.0, reverse=True) == 0
    assert grade([300, 240, 180, 120], 240.0, reverse=True) == 2
    assert grade([300, 240, 180, 120], 150.5, reverse=True) == 3


def test_grade_stops_at_the_first_miss():
    assert grade([10, 5, 1, 0], 7, reverse=False) == 0  # 5, 1 and 0 are met, but 10 is checked first
    assert grade([100, 50, 300, 400], 60.0, reverse=True) == 1  # 300 and 400 are met after the miss at 50


def test_grade_all_thresholds_met_is_4():
    assert grade([10, 20, 30, 40], 40, reverse=False) == 4
    assert grade([10, 20, 30, 40], 1e6, reverse=False) == 4
    assert grade([300, 240, 180, 120], 12.5, reverse=True) == 4


def test_compute_rank_letters():
    assert RANK_LETTERS == ("D", "C", "B", "A", "S")
    cases = [
        ((400.0, 0, 0), "D"),  # 0 + 0 + 0
        ((300.0, 0, 0), "D"),  # 1: 1/3 rounds down
        ((300.0, 10, 0), "C"),  # 2: 2/3 rounds up
        ((240.0, 20, 1000), "B"),  # 2 + 2 + 1 = 5
        ((180.0, 30, 2000), "A"),  # 3 + 3 + 2 = 8
        ((120.0, 40, 3000), "S"),  # 4 + 4 + 3 = 11
    ]
    for (seconds, kills, style), letter in cases:
        assert compute_rank(seconds, kills, style, 0, RANKS) == letter, (seconds, kills, style)


def test_compute_rank_restarts_lower_the_rank():
    assert compute_rank(180.0, 30, 3000, 0, RANKS) == "A"  # 3 + 3 + 3 = 9
    assert compute_rank(180.0, 30, 3000, 2, RANKS) == "B"  # 7
    assert compute_rank(180.0, 30, 3000, 5, RANKS) == "C"  # 4
    assert compute_rank(400.0, 10, 1000, 9, RANKS) == "D"  # 2 - 9 floors at 0


def test_compute_rank_p_needs_12_without_restarts():
    assert compute_rank(100.0, 45, 5000, 0, RANKS) == "P"
    assert compute_rank(100.0, 45, 5000, 1, RANKS) == "S"  # 12 - 1 = 11


def test_archive_cell_floors_each_axis():
    archive = ExplorationArchive(cell_size=4.0)
    assert archive.cell((0.0, 3.99, 4.0)) == (0, 0, 1)
    assert archive.cell([-0.5, -4.0, -4.01]) == (-1, -1, -2)


def test_archive_novelty_first_entry_per_episode():
    archive = ExplorationArchive(cell_size=4.0)
    archive.start_episode()
    assert archive.visit((1.0, 1.0, 1.0)) == 1.0
    assert archive.visit((3.0, 2.0, 0.5)) == 0.0  # same cell, same episode
    assert archive.visit((5.0, 1.0, 1.0)) == 1.0  # a different cell
    assert archive.episode_cells == 2
    archive.start_episode()
    assert archive.episode_cells == 0
    assert abs(archive.visit((2.0, 2.0, 2.0)) - 1 / math.sqrt(2)) < 1e-12
    assert archive.visit((2.0, 2.0, 2.0)) == 0.0
    archive.start_episode()
    assert abs(archive.visit((2.0, 2.0, 2.0)) - 1 / math.sqrt(3)) < 1e-12
    assert archive.counts[(0, 0, 0)] == 3 and archive.counts[(1, 0, 0)] == 1


def test_archive_features_straight_ahead_and_clockwise():
    archive = ExplorationArchive(cell_size=4.0)
    archive.counts[(0, 0, 1)] = 10**6  # +z of the player's cell; saturates at 1.0
    archive.counts[(1, 0, 0)] = 1  # +x of the player's cell
    one = math.log(2.0) / math.log(1001.0)
    pos = (2.0, 2.0, 2.0)  # the centre of cell (0, 0, 0)

    ahead = archive.features(pos, 0.0)
    assert len(ahead) == 9
    assert ahead[1] == 1.0  # k = 0: yaw 0 faces +z
    assert abs(ahead[3] - one) < 1e-12  # k = 2: 90 degrees to the right of +z is +x
    assert all(v == 0.0 for i, v in enumerate(ahead) if i not in (1, 3))  # index 0 is the player's own cell

    turned = archive.features(pos, 90.0)
    assert len(turned) == 9
    assert abs(turned[1] - one) < 1e-12  # k = 0: yaw 90 faces +x
    assert turned[7] == 1.0  # k = 6: 270 degrees clockwise from +x is +z, on the player's left
    assert all(v == 0.0 for i, v in enumerate(turned) if i not in (1, 7))


def test_archive_save_load_round_trip():
    archive = ExplorationArchive(cell_size=4.0)
    archive.start_episode()
    for pos in ((1.0, 1.0, 1.0), (-7.0, 0.5, 30.0), (40000.0, -3.0, -2.0)):
        archive.visit(pos)
    archive.start_episode()
    archive.visit((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models" / "explore_Level_0-1_47800.npz"
        archive.save(path)
        assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp file left, no second .npz suffix
        loaded = ExplorationArchive.load(path, cell_size=4.0)
        empty = Path(tmp) / "empty.npz"
        ExplorationArchive(cell_size=4.0).save(empty)
        assert ExplorationArchive.load(empty, cell_size=4.0).counts == {}
    assert loaded.cell_size == 4.0 and loaded.episode_cells == 0
    assert loaded.counts == {(0, 0, 0): 2, (-2, 0, 7): 1, (10000, -1, -1): 1}
    assert all(type(v) is int for cell in loaded.counts for v in cell)
    assert all(type(n) is int for n in loaded.counts.values())


def test_archive_save_retries_a_briefly_locked_file():
    archive = ExplorationArchive(cell_size=4.0)
    archive.visit((1.0, 1.0, 1.0))
    real_replace, calls = os.replace, []

    def replace_locked_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:  # Windows refuses the replace while another process has the file open
            raise PermissionError(13, "The process cannot access the file")
        real_replace(src, dst)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explore.npz"
        with mock.patch("os.replace", replace_locked_twice):
            archive.save(path)
        assert len(calls) == 3
        assert [p.name for p in path.parent.iterdir()] == ["explore.npz"]
        assert ExplorationArchive.load(path, cell_size=4.0).counts == {(0, 0, 0): 1}
        with mock.patch("os.replace", side_effect=PermissionError(13, "locked")) as always_locked:
            try:
                archive.save(Path(tmp) / "stuck.npz")
            except PermissionError:
                pass
            else:
                raise AssertionError("save should give up and raise")
        assert always_locked.call_count == 5


def test_archive_load_missing_or_unreadable_is_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert ExplorationArchive.load(Path(tmp) / "missing.npz", cell_size=4.0).counts == {}
        bad = Path(tmp) / "bad.npz"
        for data in (b"not an npz file", b"PK\x03\x04 truncated zip", b""):
            bad.write_bytes(data)
            loaded = ExplorationArchive.load(bad, cell_size=4.0)
            assert loaded.counts == {} and loaded.cell_size == 4.0, data


def test_archive_load_cell_size_mismatch_is_empty():
    archive = ExplorationArchive(cell_size=4.0)
    archive.visit((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explore.npz"
        archive.save(path)
        assert ExplorationArchive.load(path, cell_size=4.0).counts == {(0, 0, 0): 1}
        loaded = ExplorationArchive.load(path, cell_size=2.0)
    assert loaded.counts == {} and loaded.cell_size == 2.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
