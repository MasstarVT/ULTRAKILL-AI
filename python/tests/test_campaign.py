"""Campaign helpers (ultrakill_ai/campaign.py). No game needed:  python tests/test_campaign.py  (or pytest)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CAMPAIGN_LEVELS,
    RANK_LETTERS,
    ExplorationArchive,
    MilestoneTracker,
    PathProgress,
    choose_fresh_start,
    compute_rank,
    grade,
    safe_name,
    save_best_run,
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


def test_archive_cell_is_none_for_a_non_finite_position():
    archive = ExplorationArchive(cell_size=4.0)
    assert archive.cell((float("nan"), 0.0, 0.0)) is None
    assert archive.cell((0.0, float("inf"), 0.0)) is None
    assert archive.cell((0.0, 0.0, float("-inf"))) is None
    assert archive.cell((1.0, 2.0, 3.0)) == (0, 0, 0)  # unaffected coordinates still work


def test_archive_visit_ignores_a_non_finite_position():
    # math.floor raises ValueError on NaN and OverflowError on infinity; one glitched physics frame must not
    # crash the training worker, so a non-finite position simply pays no novelty.
    archive = ExplorationArchive(cell_size=4.0)
    archive.start_episode()
    assert archive.visit((float("nan"), 0.0, 0.0)) == 0.0
    assert archive.visit((0.0, float("inf"), 0.0)) == 0.0
    assert archive.visit((0.0, 0.0, float("-inf"))) == 0.0
    assert archive.episode_cells == 0  # nothing was recorded as visited
    assert archive.counts == {}  # and nothing polluted the saved counts
    # a later finite visit still works normally
    assert archive.visit((1.0, 1.0, 1.0)) == 1.0


def test_archive_features_ignores_a_non_finite_position_or_yaw():
    archive = ExplorationArchive(cell_size=4.0)
    archive.counts[(0, 0, 0)] = 5  # would otherwise show up as nonzero at the player's own cell

    nan_pos = archive.features((float("nan"), 0.0, 0.0), 0.0)
    assert nan_pos == [0.0] * 9

    inf_pos = archive.features((0.0, 0.0, float("inf")), 0.0)
    assert inf_pos == [0.0] * 9

    # a non-finite yaw only breaks the neighbours (which depend on yaw); the player's own cell is unaffected
    nan_yaw = archive.features((0.0, 0.0, 0.0), float("nan"))
    assert nan_yaw[0] == min(1.0, math.log1p(5) / math.log(1001.0))
    assert nan_yaw[1:] == [0.0] * 8


def test_archive_load_cell_size_mismatch_is_empty():
    archive = ExplorationArchive(cell_size=4.0)
    archive.visit((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explore.npz"
        archive.save(path)
        assert ExplorationArchive.load(path, cell_size=4.0).counts == {(0, 0, 0): 1}
        loaded = ExplorationArchive.load(path, cell_size=2.0)
    assert loaded.counts == {} and loaded.cell_size == 2.0


def milestones_block(checkpoints=(), arenas=(), doors=()):
    """A campaign block holding only the milestone keys. `checkpoints` holds (id, activated, current) tuples."""
    return {
        "checkpoints": [{"id": cid, "pos": [0.0, 0.0, 0.0], "activated": act, "current": cur} for cid, act, cur in checkpoints],
        "cleared_arenas": list(arenas),
        "unlocked_doors": list(doors),
    }


def test_milestones_pay_a_checkpoint_once_per_level_load():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (1, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (0, 0, 0)
    assert m.checkpoints_reached == 1
    further = milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(further) == (1, 1, 1)
    assert m.update(further) == (0, 0, 0)
    assert m.checkpoints_reached == 2
    assert m.update(None) == (0, 0, 0)  # a step without the block pays nothing and forgets nothing
    assert m.update(further) == (0, 0, 0)


def test_milestones_level_load_baseline_pays_nothing():
    m = MilestoneTracker()
    loaded = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.new_level_load(loaded)
    assert m.update(loaded) == (0, 0, 0)
    assert m.checkpoints_reached == 1
    m.new_level_load(None)
    assert m.checkpoints_reached == 0


def test_milestones_mark_paid_absorbs_a_respawn_unlock():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"])) == (1, 1, 0)
    # Restart() at the checkpoint unlocked a door: absorbed, never paid
    respawned = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.mark_paid(respawned)
    assert m.update(respawned) == (0, 0, 0)
    later = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25", "9,0,50"])
    assert m.update(later) == (0, 0, 1)
    assert m.checkpoints_reached == 1


def test_milestones_current_counts_as_activated():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False), ("0,1,80", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", False, True), ("0,1,80", False, False)])) == (1, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", False, False)])) == (0, 0, 0)
    assert m.checkpoints_reached == 1


def test_milestones_new_level_load_pays_again():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    reached = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(reached) == (1, 1, 1)
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.checkpoints_reached == 0
    assert m.update(reached) == (1, 1, 1)


def nav_path(length, status="complete"):
    return {"status": status, "length": length, "next_corner": [0.0, 0.0, 0.0]}


def test_path_progress_baseline_pays_nothing():
    p = PathProgress()
    p.reset()
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(70.0)) == 0.0  # a longer path is not progress and does not move the best
    assert p.update(nav_path(58.5)) == 1.5  # still measured from 60


def test_path_progress_pays_a_gain_over_min_gain():
    assert PathProgress.MIN_GAIN == 1.0
    p = PathProgress()
    p.reset()
    p.update(nav_path(60.0))
    assert p.update(nav_path(52.0)) == 8.0
    assert p.update(nav_path(52.0)) == 0.0
    assert p.update(nav_path(51.0)) == 0.0  # exactly MIN_GAIN is not enough
    p.reset()
    assert p.update(nav_path(40.0)) == 0.0  # a new episode starts from a new baseline
    assert p.update(nav_path(30.0)) == 10.0


def test_path_progress_partial_or_missing_path_pays_nothing():
    p = PathProgress()
    p.reset()
    assert p.update(None) == 0.0
    assert p.update({"status": "none"}) == 0.0
    assert p.update(nav_path(10.0, status="partial")) == 0.0  # does not set the baseline either
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(20.0, status="partial")) == 0.0
    assert p.update({"status": "none"}) == 0.0
    assert p.update(nav_path(55.0)) == 5.0


def test_path_progress_sub_metre_gains_accumulate():
    p = PathProgress()
    p.reset()
    p.update(nav_path(60.0))
    assert p.update(nav_path(59.6)) == 0.0
    assert p.update(nav_path(59.2)) == 0.0
    assert abs(p.update(nav_path(58.8)) - 1.2) < 1e-9  # measured from 60, not from 59.2
    assert p.update(nav_path(58.8)) == 0.0


def fresh(rng, fresh_prob, **overrides):
    """choose_fresh_start for a game mid-level with an active checkpoint, with some conditions overridden."""
    kwargs = {"in_level": True, "level_over": False, "has_checkpoint": True, "stuck_streak": 0, "stuck_limit": 3}
    kwargs.update(overrides)
    return choose_fresh_start(rng, fresh_prob=fresh_prob, **kwargs)


def test_fresh_start_when_not_in_the_level():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, in_level=False) is True


def test_fresh_start_after_the_level_is_over():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, level_over=True) is True


def test_fresh_start_without_a_checkpoint():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, has_checkpoint=False) is True


def test_fresh_start_after_repeated_stuck_episodes():
    assert fresh(random.Random(0), 0.0, stuck_streak=2) is False
    assert fresh(random.Random(0), 0.0, stuck_streak=3) is True
    assert fresh(random.Random(0), 0.0, stuck_streak=4) is True
    assert fresh(random.Random(0), 1.0, stuck_streak=3) is True


def test_fresh_start_otherwise_by_probability():
    assert fresh(random.Random(0), 0.0) is False
    assert fresh(random.Random(0), 1.0) is True
    rng, twin = random.Random(123), random.Random(123)
    picks = [fresh(rng, 0.2) for _ in range(2000)]
    assert picks == [twin.random() < 0.2 for _ in range(2000)]  # one draw per decision from the seeded source
    assert 300 < sum(picks) < 500
    forced, untouched = random.Random(7), random.Random(7)
    assert fresh(forced, 0.2, in_level=False) is True
    assert forced.random() == untouched.random()  # a forced reload does not draw


def best_run(seconds, **extra):
    run = {
        "level": "Level 0-1", "seconds": seconds, "kills": 5, "style": 300, "restarts": 0, "deaths": 0, "rank": "A",
        "difficulty": 3, "positions": [[0, 1, 2], [0, 1, 4]], "saved_at": "2026-09-16T12:00:00",
    }
    run.update(extra)
    return run


def test_save_best_run_writes_the_first_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_runs" / "Level_0-1.json"
        assert save_best_run(path, best_run(95.5)) is True
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(95.5)
        assert [p.name for p in path.parent.iterdir()] == ["Level_0-1.json"]  # no temp file left behind


def test_save_best_run_refuses_a_slower_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        assert save_best_run(path, best_run(95.5)) is True
        assert save_best_run(path, best_run(120.0, kills=9)) is False
        assert save_best_run(path, best_run(95.5, kills=9)) is False  # a tie keeps the stored run
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(95.5)


def test_save_best_run_overwrites_with_a_faster_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        save_best_run(path, best_run(95.5))
        assert save_best_run(path, best_run(80.25, rank="S")) is True
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(80.25, rank="S")
        assert [p.name for p in path.parent.iterdir()] == ["Level_0-1.json"]


def test_save_best_run_overwrites_an_unreadable_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        path.write_text("{truncated", encoding="utf-8")
        assert save_best_run(path, best_run(300.0)) is True
        assert json.loads(path.read_text(encoding="utf-8"))["seconds"] == 300.0
        path.write_text('{"level": "Level 0-1"}', encoding="utf-8")  # readable JSON without a time
        assert save_best_run(path, best_run(310.0)) is True
        assert json.loads(path.read_text(encoding="utf-8"))["seconds"] == 310.0


def test_save_best_run_serialises_two_racing_writers():
    """Two processes finishing at nearly the same time must not let the slower one win the write.

    Without locking the whole read-compare-write, both could read the same stale stored value, both decide
    independently that they are faster, and whichever writes last wins regardless of which run is actually
    faster. This uses real threads (so the actual O_CREAT | O_EXCL lock file is exercised, not a mock standing
    in for it) with a sleep forced into the middle of each writer's critical section -- in the style of
    test_archive_save_retries_a_briefly_locked_file's monkeypatching, but timed rather than counted, since the
    two writers must genuinely overlap for the race to be worth testing. A shared counter proves the lock kept
    them from ever being inside their critical section at the same time, and the stored file must end up with
    the faster of the two runs no matter which thread's write lands last.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        assert save_best_run(path, best_run(200.0)) is True  # a slow baseline already on disk

        real_read_text = Path.read_text
        active = {"n": 0, "max": 0}
        guard = threading.Lock()

        def slow_read_text(self, *a, **kw):
            with guard:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            try:
                time.sleep(0.05)  # hold the critical section open long enough for the other thread to try it
                return real_read_text(self, *a, **kw)
            finally:
                with guard:
                    active["n"] -= 1

        results: dict[str, bool] = {}

        def writer(name: str, seconds: float) -> None:
            results[name] = save_best_run(path, best_run(seconds), lock_timeout=5.0, lock_poll=0.01)

        with mock.patch("pathlib.Path.read_text", slow_read_text):
            slow_thread = threading.Thread(target=writer, args=("slow", 150.0))
            fast_thread = threading.Thread(target=writer, args=("fast", 90.0))
            slow_thread.start()
            fast_thread.start()
            slow_thread.join()
            fast_thread.join()

        assert active["max"] == 1  # the lock never let both writers' critical sections overlap
        assert json.loads(path.read_text(encoding="utf-8"))["seconds"] == 90.0  # the faster run survived
        assert not (path.parent / f"{path.name}.lock").exists()  # the lock file is always cleaned up


def test_save_best_run_gives_up_on_a_stale_lock_instead_of_blocking_forever():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        stale_lock = path.with_name(f"{path.name}.lock")
        stale_lock.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(stale_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # a lock never removed by a killed process
        started = time.monotonic()
        assert save_best_run(path, best_run(80.0), lock_timeout=0.2, lock_poll=0.02) is False
        assert time.monotonic() - started < 2.0  # gave up quickly instead of blocking forever
        assert not path.exists()  # nothing was written
        assert stale_lock.exists()  # a stale lock is left for whoever created it to clean up, not deleted by a waiter


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
