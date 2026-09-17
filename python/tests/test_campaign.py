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
    CAMPAIGN_LEVELS_SHIPPED,
    GATE_EXIT_KEY,
    RANK_LETTERS,
    SUBGOAL_ALTAR,
    SUBGOAL_ITEM,
    ExitGuard,
    ExplorationArchive,
    GateProgress,
    MilestoneTracker,
    PathProgress,
    altar_aim_point,
    choose_fresh_start,
    choose_level,
    compute_rank,
    dead_twin,
    grade,
    level_weights,
    read_curriculum,
    safe_name,
    save_best_run,
    unlock_next,
    wanting_altars,
)

RANKS = {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}


def test_campaign_levels():
    assert len(CAMPAIGN_LEVELS) == 35 and len(set(CAMPAIGN_LEVELS)) == 35
    assert CAMPAIGN_LEVELS[0] == "Level 0-1"
    assert CAMPAIGN_LEVELS[4] == "Level 0-5" and CAMPAIGN_LEVELS[5] == "Level 1-1"
    assert CAMPAIGN_LEVELS[14] == "Level 3-2"
    assert CAMPAIGN_LEVELS[-1] == "Level 9-2"


def test_shipped_levels_exclude_the_two_with_no_scene_bundle():
    assert len(CAMPAIGN_LEVELS_SHIPPED) == 33
    assert CAMPAIGN_LEVELS_SHIPPED < set(CAMPAIGN_LEVELS)
    assert "Level 9-1" not in CAMPAIGN_LEVELS_SHIPPED and "Level 9-2" not in CAMPAIGN_LEVELS_SHIPPED
    assert "Level 0-1" in CAMPAIGN_LEVELS_SHIPPED and "Level 8-4" in CAMPAIGN_LEVELS_SHIPPED


# ---------------------------------------------------------------------------
# The multi-level curriculum
# ---------------------------------------------------------------------------

ORDER = ["Level 0-1", "Level 0-3", "Level 0-4"]


def record(*, unlocked=True, window=50, rate=None, best=None, episodes=0, fresh_episodes=0):
    return {"unlocked": unlocked, "fresh_window": window, "fresh_completion_rate": rate,
            "best_time": best, "episodes": episodes, "fresh_episodes": fresh_episodes}


def draws(stats, n=4000, seed=0, floor=0.1):
    """The share of fresh draws each level gets, so the weights can be checked through choose_level itself."""
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for _ in range(n):
        level = choose_level(rng, ORDER, stats, floor=floor)
        counts[level] = counts.get(level, 0) + 1
    return {level: count / n for level, count in counts.items()}


def test_choose_level_gives_an_unseen_level_the_full_weight():
    # 0-1 mastered (rate 1.0 -> floor 0.1), 0-3 just unlocked and never played (rate None -> weight 1.0).
    stats = {"Level 0-1": record(rate=1.0), "Level 0-3": record(window=0), "Level 0-4": record(unlocked=False)}
    assert dict(level_weights(ORDER, stats)) == {"Level 0-1": 0.1, "Level 0-3": 1.0}
    share = draws(stats)
    assert "Level 0-4" not in share, "a locked level is never drawn"
    assert abs(share["Level 0-3"] - 1.0 / 1.1) < 0.03 and abs(share["Level 0-1"] - 0.1 / 1.1) < 0.03


def test_choose_level_is_uniform_once_every_level_is_mastered():
    stats = {level: record(rate=1.0) for level in ORDER}
    assert [w for _, w in level_weights(ORDER, stats)] == [0.1, 0.1, 0.1]
    share = draws(stats)
    assert all(abs(share[level] - 1 / 3) < 0.03 for level in ORDER), share


def test_choose_level_always_has_the_first_level_in_the_pool():
    # Empty stats, and a table that explicitly locks order[0]: both still draw it, so a fresh run can start.
    for stats in ({}, {"Level 0-1": record(unlocked=False)}, None):
        assert choose_level(random.Random(0), ORDER, stats) == "Level 0-1"


def test_unlock_next_does_not_raise_on_an_empty_table():
    assert unlock_next(ORDER, {}) is None
    assert unlock_next(ORDER, None) is None
    assert unlock_next([], {}) is None
    assert unlock_next(["Level 0-1"], {}) is None


def test_unlock_next_needs_the_rate_and_the_window():
    below_rate = {"Level 0-1": record(window=50, rate=0.49)}
    assert unlock_next(ORDER, below_rate) is None
    below_window = {"Level 0-1": record(window=19, rate=1.0)}
    assert unlock_next(ORDER, below_window) is None, "19 fresh episodes is not evidence"
    earned = {"Level 0-1": record(window=20, rate=0.5)}
    assert unlock_next(ORDER, earned) == "Level 0-3", "order[0] is unlocked by the default record"


def test_unlock_next_is_chained_and_stops_at_the_first_locked_level():
    stats = {"Level 0-1": record(rate=0.9), "Level 0-3": record(unlocked=False), "Level 0-4": record(unlocked=False)}
    assert unlock_next(ORDER, stats) == "Level 0-3"
    stats["Level 0-3"] = record(rate=0.1)  # now unlocked but nowhere near the bar
    assert unlock_next(ORDER, stats) is None, "0-4 must not unlock before 0-3 has earned it"
    stats["Level 0-3"] = record(rate=0.6)
    assert unlock_next(ORDER, stats) == "Level 0-4"
    stats["Level 0-4"] = record(rate=0.0)
    assert unlock_next(ORDER, stats) is None  # everything is unlocked


def test_unlock_is_a_latch_the_caller_holds():
    """`unlock_next` never returns an already-unlocked level, so a falling rate cannot re-lock one.

    Without the latch a level re-locks as its completion rate falls and the learning in progress on it stalls.
    """
    stats = {"Level 0-1": record(rate=0.9), "Level 0-3": record(unlocked=True, rate=0.0, window=50)}
    assert unlock_next(ORDER, stats) is None
    assert "Level 0-3" in dict(level_weights(ORDER, stats))


def test_a_level_inserted_before_an_unlocked_one_does_not_re_lock_it():
    """Inserting 0-2 between 0-1 and 0-3 must leave 0-3 unlocked and still sampled.

    This is the 2026-09-17 pause's own case: the live run had 0-1 and 0-3 unlocked, and the new order puts 0-2
    between them. `unlock_next` walks the order and stops at the first LOCKED level, so with 0-2 locked it never
    looks at 0-3 -- and it has no way to lock anything, since it only ever names a level for the caller to
    unlock. The danger would be `level_weights` dropping 0-3 for having a locked predecessor; it does not, it
    reads each level's own `unlocked` flag.
    """
    order = ["Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4"]
    stats = {"Level 0-1": record(window=0, rate=None), "Level 0-2": record(unlocked=False),
             "Level 0-3": record(unlocked=True, window=9, rate=0.0), "Level 0-4": record(unlocked=False)}
    assert unlock_next(order, stats) is None, "0-1's window is empty after a restart, so nothing unlocks yet"
    assert set(dict(level_weights(order, stats))) == {"Level 0-1", "Level 0-3"}, "0-3 keeps its sampling weight"
    stats["Level 0-1"] = record(window=20, rate=0.54)  # the window refills at the old rate
    assert unlock_next(order, stats) == "Level 0-2", "and then the inserted level unlocks normally"
    stats["Level 0-2"] = record(unlocked=True, window=0, rate=None)
    assert unlock_next(order, stats) is None, "0-3 is already unlocked; 0-4 waits on 0-3's own rate"


def test_the_safety_valve_unlocks_on_fresh_episodes_alone():
    """`unlock_after_fresh_episodes` stops one hard level blocking the whole campaign.

    Off by default (0), so every existing run is byte for byte unchanged. When set, a level that has spent that
    many of its own fresh episodes without reaching `unlock_rate` opens its successor anyway -- the rate bar is
    the fast path, this is the slow one. It counts CUMULATIVE fresh episodes, not the 50-deep window, because a
    window saturates at 50 and could never express "600 tries".
    """
    stats = {"Level 0-1": record(window=50, rate=0.1, fresh_episodes=599)}
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) is None
    assert unlock_next(ORDER, stats) is None, "and with the valve off it stays shut forever"
    stats["Level 0-1"]["fresh_episodes"] = 600
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) == "Level 0-3"
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=0) is None, "0 means off, not 'always'"


def test_the_safety_valve_is_chained_like_the_rate_bar():
    """It opens exactly one rung: the successor of the level that ran out of patience, not the whole ladder."""
    stats = {"Level 0-1": record(window=50, rate=0.0, fresh_episodes=900),
             "Level 0-3": record(unlocked=False), "Level 0-4": record(unlocked=False)}
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) == "Level 0-3"
    stats["Level 0-3"] = record(unlocked=True, window=50, rate=0.0, fresh_episodes=10)
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) is None, "0-4 waits its own 600 out"
    stats["Level 0-3"]["fresh_episodes"] = 600
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) == "Level 0-4"


def test_the_safety_valve_still_needs_the_predecessor_unlocked():
    """A locked level cannot accumulate fresh episodes, but a hand-edited table must not open a hole either."""
    stats = {"Level 0-1": record(window=50, rate=0.0, fresh_episodes=5),
             "Level 0-3": record(unlocked=False, fresh_episodes=9999), "Level 0-4": record(unlocked=False)}
    assert unlock_next(ORDER, stats, unlock_after_fresh_episodes=600) is None


def test_a_table_written_before_the_safety_valve_existed_still_loads():
    """Old `curriculum.json` / `status.json` rows have no `fresh_episodes`; they must read as 0, not raise."""
    old = {"Level 0-1": {"unlocked": True, "fresh_window": 50, "fresh_completion_rate": 0.1,
                         "best_time": None, "episodes": 400}}
    assert unlock_next(ORDER, old, unlock_after_fresh_episodes=600) is None
    old["Level 0-1"]["fresh_completion_rate"] = 0.6
    assert unlock_next(ORDER, old, unlock_after_fresh_episodes=600) == "Level 0-3", "the rate bar still works"


def curriculum_file(path, *, order=ORDER, run_name="campaign_prelude", levels=None):
    data = {"version": 1, "updated_at": 1.0, "run_name": run_name, "order": list(order),
            "levels": levels if levels is not None else {level: record() for level in order}}
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def test_read_curriculum_rejects_anything_it_cannot_trust():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "curriculum.json"
        assert read_curriculum(path, order=ORDER) is None, "missing"
        path.write_text('{"version": 1, "order": ["Level 0-1"], "lev', encoding="utf-8")
        assert read_curriculum(path, order=ORDER) is None, "torn"
        path.write_text("[]", encoding="utf-8")
        assert read_curriculum(path, order=ORDER) is None, "not an object"
        curriculum_file(path, levels="nope")
        assert read_curriculum(path, order=ORDER) is None, "levels is not a table"
        curriculum_file(path, order=["Level 0-1", "Level 0-4"])
        assert read_curriculum(path, order=ORDER) is None, "another run's level order"
        curriculum_file(path)
        assert read_curriculum(path, order=ORDER, run_name="other_run") is None, "another run's name"
        assert set(read_curriculum(path, order=ORDER, run_name="campaign_prelude")) == set(ORDER)
        assert read_curriculum(path) is not None, "with no order to check against, the file is usable"


def test_read_curriculum_drops_rows_that_are_not_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "curriculum.json"
        curriculum_file(path, levels={"Level 0-1": record(), "Level 0-3": 5, "Level 0-4": None})
        table = read_curriculum(path, order=ORDER)
        assert list(table) == ["Level 0-1"]
        # The default record is what keeps the two dropped rows usable anyway.
        assert choose_level(random.Random(0), ORDER, table) in ("Level 0-1",)


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


def milestones_block(checkpoints=(), arenas=(), doors=(), altars=(), items=()):
    """A campaign block holding only the milestone keys. `checkpoints` holds (id, activated, current) tuples.

    `altars` and `items` are passed through as the mod sends them (see `altar` and `item` below); a mod older
    than 0.7.0 sends neither key at all, which is what leaving them out reproduces.
    """
    block = {
        "checkpoints": [{"id": cid, "pos": [0.0, 0.0, 0.0], "activated": act, "current": cur} for cid, act, cur in checkpoints],
        "cleared_arenas": list(arenas),
        "unlocked_doors": list(doors),
    }
    if altars:
        block["altars"] = list(altars)
    if items:
        block["items"] = list(items)
    return block


def test_milestones_pay_a_checkpoint_once_per_level_load():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (1, 0, 0, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (0, 0, 0, 0, 0)
    assert m.checkpoints_reached == 1
    further = milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(further) == (1, 1, 1, 0, 0)
    assert m.update(further) == (0, 0, 0, 0, 0)
    assert m.checkpoints_reached == 2
    assert m.update(None) == (0, 0, 0, 0, 0)  # a step without the block pays nothing and forgets nothing
    assert m.update(further) == (0, 0, 0, 0, 0)


def test_milestones_level_load_baseline_pays_nothing():
    m = MilestoneTracker()
    loaded = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.new_level_load(loaded)
    assert m.update(loaded) == (0, 0, 0, 0, 0)
    assert m.checkpoints_reached == 1
    m.new_level_load(None)
    assert m.checkpoints_reached == 0


def test_milestones_mark_paid_absorbs_a_respawn_unlock():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"])) == (1, 1, 0, 0, 0)
    # Restart() at the checkpoint unlocked a door: absorbed, never paid
    respawned = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.mark_paid(respawned)
    assert m.update(respawned) == (0, 0, 0, 0, 0)
    later = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25", "9,0,50"])
    assert m.update(later) == (0, 0, 1, 0, 0)
    assert m.checkpoints_reached == 1


def test_milestones_current_counts_as_activated():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False), ("0,1,80", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", False, True), ("0,1,80", False, False)])) == (1, 0, 0, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", False, False)])) == (0, 0, 0, 0, 0)
    assert m.checkpoints_reached == 1


def test_milestones_new_level_load_pays_again():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    reached = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(reached) == (1, 1, 1, 0, 0)
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.checkpoints_reached == 0
    assert m.update(reached) == (1, 1, 1, 0, 0)


# -- the two skull-carry milestones ---------------------------------------------------------

def zone(key, kind, *, filled=False, doors=("20,-10,381",)):
    return {"key": key, "pos": [0.0, 0.0, 0.0], "item": kind, "filled": filled, "active": True,
            "inactive_ancestors": 1, "doors": [{"key": k, "pos": [0.0, 0.0, 0.0]} for k in doors],
            "reverse_doors": []}


def carryable(key, kind, *, held=False, placed_in=None):
    """An `items[]` entry. `placed_in` is the altar key it is resting in, which is what a placement is read from."""
    return {"key": key, "pos": [0.0, 0.0, 0.0], "item": kind, "held": held, "placed": placed_in is not None,
            "placed_in": placed_in, "active": True, "active_self": True, "inactive_ancestors": 1}


def skull_milestones(altars, items):
    return milestones_block(altars=altars, items=items)


def test_item_pickup_pays_once_per_type_and_only_for_accepted_types():
    m = MilestoneTracker()
    altars = [zone("a", "SkullRed")]
    m.new_level_load(skull_milestones(altars, [carryable("s1", "SkullRed")]))
    assert m.update(skull_milestones(altars, [carryable("s1", "SkullRed", held=True)]))[3] == 1
    assert m.update(skull_milestones(altars, [carryable("s1", "SkullRed", held=True)]))[3] == 0
    # A respawn re-instantiates the skull under a new mod-side key: type-keyed, so it cannot pay again.
    assert m.update(skull_milestones(altars, [carryable("s1#2", "SkullRed", held=True)]))[3] == 0
    # 8-2 ships 22 CustomKey1 props and 6-1 eleven; no altar accepts them, so picking them up is not a milestone.
    props = [carryable("p", "CustomKey1", held=True), carryable("t", "Torch", held=True)]
    assert m.update(skull_milestones(altars, props))[3] == 0


def test_item_placed_pays_once_per_puzzle_not_once_per_zone():
    """Coincident duplicate zones 0.2 m apart drive the same door; one puzzle pays once.

    A placement is physical: `Punch.PlaceHeldObject` reparents the skull to the zone and clears `pickedUp` in
    one call, so on the paying step the item reads `held: false, placed_in: <that zone>` and the step before it
    read `held: true`. The item never reads held while the altar it is in reads filled.
    """
    m = MilestoneTracker()
    twins = [zone("0,-7,381", "SkullRed"), zone("0,-7,381#2", "SkullRed")]
    m.new_level_load(skull_milestones(twins, [carryable("s1", "SkullRed", held=True)]))
    filled_one = [zone("0,-7,381", "SkullRed", filled=True), twins[1]]
    assert m.update(skull_milestones(filled_one, [carryable("s1", "SkullRed", placed_in="0,-7,381")]))[4] == 1
    # Punch it back out and place it in the twin: the same puzzle, the same key, nothing more to earn.
    m.update(skull_milestones(twins, [carryable("s1", "SkullRed", held=True)]))
    both = [twins[0], zone("0,-7,381#2", "SkullRed", filled=True)]
    assert m.update(skull_milestones(both, [carryable("s1", "SkullRed", placed_in="0,-7,381#2")]))[4] == 0, \
        "punching it out and into the twin earns nothing"
    # A different puzzle (another door) is its own key and does pay.
    blue_altar = zone("9,9,9", "SkullBlue", doors=("81,-6,240",))
    red_done = carryable("s1", "SkullRed", placed_in="0,-7,381#2")
    carrying = [carryable("s2", "SkullBlue", held=True), red_done]
    assert m.update(skull_milestones(both + [blue_altar], carrying))[3] == 1  # the blue skull is picked up
    placed = both + [zone("9,9,9", "SkullBlue", filled=True, doors=("81,-6,240",))]
    assert m.update(skull_milestones(placed, [carryable("s2", "SkullBlue", placed_in="9,9,9"), red_done]))[4] == 1


def test_an_altar_that_fills_with_nothing_ever_held_pays_nothing():
    """M11: 31 source pedestals campaign-wide read filled: true the moment their room switches on.

    `mark_paid` cannot catch those -- it only runs at a reset or a respawn, never when a room activates
    mid-episode -- so the payment rule itself has to require that the agent put the item there.
    """
    m = MilestoneTracker()
    altars = [zone("81,-2,275", "SkullRed", doors=())]
    m.new_level_load(skull_milestones(altars, [carryable("s1", "SkullRed")]))
    switched_on = [zone("81,-2,275", "SkullRed", filled=True, doors=())]
    assert m.update(skull_milestones(switched_on, [carryable("s1", "SkullRed", placed_in="81,-2,275")]))[4] == 0
    # And it is absorbed, not merely skipped: it can never pay later either.
    m.update(skull_milestones(switched_on, [carryable("s1", "SkullRed", held=True)]))
    assert m.update(skull_milestones(switched_on, [carryable("s1", "SkullRed", held=True)]))[4] == 0


def test_a_pedestal_switching_on_mid_carry_pays_nothing():
    """The other half of M11, and the one a type-keyed rule cannot see.

    Shaped like 1-4: four blue source pedestals whose rooms stream in one at a time. "Something of this type
    was held a step ago" is true for the whole duration of any legitimate carry, so every pre-filled pedestal
    the agent walks past while carrying a skull paid +15 for nothing -- three on 1-4, three on 5-1 and 7-1,
    and up to 31 zones campaign-wide. The payment has to follow the instance, not the type.
    """
    m = MilestoneTracker()
    target = zone("target", "SkullBlue", doors=("99,9,99",))
    sources = {k: zone(k, "SkullBlue", filled=False, doors=()) for k in ("srcA", "srcB", "srcC")}
    resting = [carryable(f"skull{k[-1]}", "SkullBlue", placed_in=k) for k in sources]

    def block(filled, items):
        altars = [target] + [zone(k, "SkullBlue", filled=(k in filled), doors=()) for k in sources]
        return skull_milestones(altars, items)

    m.new_level_load(block({"srcA"}, resting))  # A's room is on at load; B and C are still switched off
    carrying = [carryable("skullA", "SkullBlue", held=True)] + resting[1:]
    assert m.update(block(set(), carrying))[3] == 1, "A's skull is picked up"
    assert m.update(block({"srcB"}, carrying))[4] == 0, "B's room switches on while the agent carries A's skull"
    assert m.update(block({"srcB", "srcC"}, carrying))[4] == 0, "and C's"
    # The real placement, into the altar that drives a door, still pays.
    done = [carryable("skullA", "SkullBlue", placed_in="target")] + resting[1:]
    altars = [zone("target", "SkullBlue", filled=True, doors=("99,9,99",)),
              zone("srcA", "SkullBlue", doors=()), zone("srcB", "SkullBlue", filled=True, doors=()),
              zone("srcC", "SkullBlue", filled=True, doors=())]
    assert m.update(skull_milestones(altars, done))[4] == 1


def test_a_respawns_item_keys_are_absorbed_not_paid():
    m = MilestoneTracker()
    altars = [zone("a", "SkullRed", filled=True)]
    held = [carryable("s1", "SkullRed", held=True)]
    m.new_level_load(skull_milestones(altars, held))  # the level load absorbs everything it already reports
    assert m.update(skull_milestones(altars, held)) == (0, 0, 0, 0, 0)
    m.mark_paid(skull_milestones(altars, held))
    assert m.update(skull_milestones(altars, held)) == (0, 0, 0, 0, 0)
    m.new_level_load(skull_milestones([zone("a", "SkullRed")], [carryable("s1", "SkullRed")]))
    assert m.update(skull_milestones([zone("a", "SkullRed")], held))[3] == 1, "a fresh load pays again"


def test_item_milestones_are_silent_against_a_mod_that_sends_neither_array():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (1, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# GateProgress: the door-graph route
# ---------------------------------------------------------------------------

def gate(key, pos, hops, *, open=False, locked=False, active=True, controller=True):
    g = {"key": key, "pos": list(pos), "hops": hops, "open": open, "locked": locked, "active": active}
    if controller is not None:
        g["controller_active"] = controller
    return g


def gates_block(gates, *, ordered=True, truncated=False, exit_pos=(0.0, 0.0, 200.0)):
    block = {"gates_ordered": ordered, "gates_truncated": truncated, "gates": list(gates)}
    if exit_pos is not None:
        block["exit"] = {"pos": list(exit_pos), "active": False}
    return block


# A corridor along +z: one gate every 20 m, hops counting down to the exit.
LADDER = [gate(f"0,0,{20 * (9 - h)}", (0.0, 0.0, 20.0 * (9 - h)), h) for h in range(9, -1, -1)]


def walk(progress: GateProgress, camp: dict, positions) -> tuple[int, float]:
    """Steps the tracker through a list of positions and returns the totals it paid."""
    paid, approach = 0, 0.0
    for pos in positions:
        g, a = progress.update(camp, pos)
        paid += g
        approach += a
    return paid, approach


def test_gate_hops_pays_once_per_level_load():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -30.0))
    paid, _ = walk(p, camp, [(0.0, 0.0, float(z)) for z in range(0, 190, 2)])
    assert paid == 10 and p.best_hops == 0 and p.gates_reached == 10
    p.mark_paid(camp, (0.0, 0.0, 180.0))
    p.reset_episode()
    again, _ = walk(p, camp, [(0.0, 0.0, float(z)) for z in range(0, 190, 2)])
    assert again == 0, "the ladder is per level load, however often it is re-walked"


def test_gate_skipped_hops_pay_per_hop():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -30.0))
    assert p.update(camp, (0.0, 0.0, 0.0))[0] == 1  # the hops 9 gate: the first instalment
    assert p.update(camp, (0.0, 0.0, 60.0))[0] == 3  # straight to hops 6: three rungs crossed
    assert p.best_hops == 6


def test_gate_approach_pays_only_new_best():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    _, forward = walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 11)])  # 20 m closer
    assert abs(forward - 20.0) < 1e-6
    _, back = walk(p, camp, [(0.0, 0.0, -40.0 - 2.0 * k) for k in range(1, 11)])
    assert back == 0.0
    _, again = walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 11)])
    assert again == 0.0, "only new best closeness pays"
    assert forward + back + again <= 60.0  # bounded by the distance the target was first seen at


def test_gate_approach_does_not_pay_for_flapping_between_two_gates_at_the_same_hops():
    """The exploit the per-gate dict closes: a tier of two gates 60 m apart, walked back and forth forever."""
    tier = [gate("a", (0.0, 0.0, 0.0), 2), gate("b", (60.0, 0.0, 0.0), 2), gate("c", (0.0, 0.0, 300.0), 1)]
    camp = gates_block(tier)
    p = GateProgress()
    p.new_level_load(camp, (30.0, 0.0, 0.0))  # the bisector, 30 m from each and outside both reach radii
    p.reset_episode()
    p.retarget(camp, (30.0, 0.0, 0.0))
    assert p.best_hops is None and p.target["key"] in ("a", "b")
    cycle = list(range(30, 10, -2)) + list(range(10, 50, 2)) + list(range(50, 28, -2))

    def cycles(n: int) -> float:
        return sum(p.update(camp, (float(x), 0.0, 0.0))[1] for _ in range(n) for x in cycle)

    first = cycles(5)
    # Bounded by the sum over distinct targets of the distance each was first targeted at, never by the number
    # of crossings: the old scalar best_dist, re-seeded on every target change, paid 0.065 a decision forever.
    assert first <= 60.0 + 1e-6, f"a tier must pay each gate's first approach once, got {first}"
    assert cycles(5) == 0.0, "crossing the tier again pays nothing at all"
    assert p.best_hops is None  # neither gate was ever actually reached


def test_gate_approach_survives_a_respawn():
    """A death inside an episode must not re-earn the ground the episode already covered."""
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 31)])
    p.mark_paid(camp, (0.0, 0.0, -60.0))  # respawned back at the start, inside the same episode
    p.retarget(camp, (0.0, 0.0, -60.0))
    _, again = walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 31)])
    assert again == 0.0


def test_gate_approach_is_re_earned_in_a_new_episode():
    """Lead ruling R1: `best_dist` is episode scoped, so a new episode's dense signal starts again.

    The `gate` ladder stays level-load scoped either way, and an episode's approach is bounded by the route
    length, so this is not farmable -- a truncation bootstraps the same state, so ending an episode early can
    never pay.
    """
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    _, first = walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 11)])
    p.mark_paid(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    _, second = walk(p, camp, [(0.0, 0.0, -60.0 + 2.0 * k) for k in range(1, 11)])
    assert abs(first - 20.0) < 1e-6 and abs(second - first) < 1e-6
    assert p.paid_hops == p.best_hops  # the ladder itself was absorbed, not re-paid


def test_target_changes_do_not_pay_approach():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    before = p.target["key"]
    _, approach = p.update(camp, (0.0, 0.0, 0.0))  # arrives at the first gate: the target moves on
    assert p.target["key"] != before and approach == 0.0


def test_target_before_any_gate_is_the_nearest():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.retarget(camp, (0.0, 0.0, -60.0))
    assert p.target["hops"] == 9 and p.best_hops is None


def test_target_after_hops_zero_is_the_exit():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.retarget(camp, (0.0, 0.0, 180.0))  # standing at the hops 0 gate
    assert p.best_hops == 0 and p.target["key"] == GATE_EXIT_KEY and p.target["hops"] is None
    no_exit = gates_block(LADDER, exit_pos=None)
    q = GateProgress()
    q.new_level_load(no_exit, (0.0, 0.0, 180.0))
    q.retarget(no_exit, (0.0, 0.0, 180.0))
    assert q.target is not None and q.target["hops"] == 0  # no exit reported: hold the last gate


def test_retarget_pays_nothing():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    for pos in ((0.0, 0.0, -60.0), (0.0, 0.0, 0.0), (0.0, 0.0, 100.0)):
        assert p.retarget(camp, pos) is None
    assert p.paid_hops is None and p.best_hops == 4  # the z 100 gate: reached, noted, never paid


def test_unordered_gates_have_no_target():
    camp = gates_block([gate("a", (0.0, 0.0, 10.0), None)], ordered=False)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.target is None
    assert p.update(camp, (0.0, 0.0, 5.0)) == (0, 0.0) and p.gates_reached == 0


def test_empty_gates_have_no_target():
    for camp in (gates_block([]), {"exit": {"pos": [0.0, 0.0, 10.0], "active": True}}, {}, None):
        p = GateProgress()
        p.new_level_load(camp, (0.0, 0.0, 0.0))
        p.reset_episode()
        p.retarget(camp, (0.0, 0.0, 0.0))
        assert p.target is None
        assert p.update(camp, (0.0, 0.0, 5.0)) == (0, 0.0)


# -- S2: the gates usability guard ----------------------------------------------------------

def ratio_block(with_hops: int, total: int, *, altar_only: int = 0, altar_only_hops: int = 0):
    """A gate array where exactly `with_hops` of `total` phase-1 gates carry a hops value, the rest null.

    `altar_only` appends that many phase-2 gates -- the altar-driven one-room doors the mod started appending
    once skull gates existed -- of which `altar_only_hops` carry a hops value. Measured, only 1-1, 5-3 and 8-1
    have one whose room is in the door graph; everywhere else the extra gates are null-hops.
    """
    gates = [gate(f"0,0,{20 * i}", (0.0, 0.0, 20.0 * i), i if i < with_hops else None) for i in range(total)]
    gates += [dict(gate(f"9,9,{i}", (9.0, 9.0, float(i)), 0 if i < altar_only_hops else None), altar_only=True)
              for i in range(altar_only)]
    return gates_block(gates)


def usable(camp, *, hops_min_frac=0.5) -> bool:
    return bool(GateProgress(hops_min_frac=hops_min_frac)._gates(camp))


def test_gates_guard_keeps_the_levels_whose_ladder_is_usable():
    """Measured ratios from the level survey, each with the phase-2 gates that level's altars add.

    The guard's threshold was calibrated on the phase-1 array, but the mod now also appends altar-driven
    one-room doors, which carry `hops: null` on every level but 1-1, 5-3 and 8-1. Counted in the ratio those
    would push 6-1 from 6/12 = 0.500 to 6/13 = 0.462 and drop the ladder the threshold exists to keep, and
    5-1 from 1.000 to 3/6 = 0.500. The ratio is therefore taken over the phase-1 gates alone, and each case
    below carries its measured phase-2 count.

    `gates_ordered` is true on 7-2 (1 of 4) and 8-3 (1 of 32), where the old code locked onto that single
    ordered door -- 892 m from the start on 8-3 -- and never retargeted, so the ladder was worse than none.
    """
    kept = ((11, 11, 0, 0, "0-1"), (24, 25, 1, 0, "8-2"), (3, 5, 0, 0, "4-3"), (28, 52, 1, 1, "8-1"),
            (6, 12, 1, 0, "6-1"), (3, 3, 3, 0, "5-1"), (13, 13, 1, 1, "1-1"), (20, 20, 2, 1, "5-3"))
    for with_hops, total, extra, extra_hops, level in kept:
        assert usable(ratio_block(with_hops, total, altar_only=extra, altar_only_hops=extra_hops)), level
    for with_hops, total, extra, level in ((1, 4, 1, "7-2"), (1, 32, 1, "8-3")):
        assert not usable(ratio_block(with_hops, total, altar_only=extra)), level


def test_gates_guard_ignores_phase_two_gates_in_the_ratio_but_still_returns_them():
    """A phase-2 gate adds no node and no edge, so it cannot make the ladder it is appended to less trustworthy.

    6-1 is the case that decides this: its 6-of-12 ladder sits exactly on the threshold, and its one altar door
    is enough to drop it if counted.
    """
    assert usable(ratio_block(6, 12, altar_only=1)), "6-1: the altar door must not drop the ladder"
    assert usable(ratio_block(3, 3, altar_only=3)), "5-1: three altar doors against a perfect ladder"
    both = GateProgress()._gates(ratio_block(6, 12, altar_only=1))
    assert len(both) == 13 and sum(1 for g in both if g.get("altar_only")) == 1, "kept ladders return every gate"
    # A phase-1 gate with no hops still counts against the ratio: only the phase-2 ones are exempt.
    assert not usable(ratio_block(3, 7, altar_only=4, altar_only_hops=4)), "3 of 7 is still under the threshold"
    # 7-1 has four altar-driven doors and no phase-1 gate at all: with no ladder to dilute, the whole array is
    # the ratio, which is the unchanged rule. All four carry hops: null, so it is dropped as it always was.
    assert not usable(ratio_block(0, 0, altar_only=4)), "7-1: four altar doors, none with hops"
    assert usable(ratio_block(0, 0, altar_only=4, altar_only_hops=4)), \
        "but an altar-only array that IS ordered is judged by the same ratio, not discarded unread"


def test_gates_guard_keeps_both_older_guards():
    assert not usable(None), "no campaign block at all (a scene load, or the mod's build threw)"
    assert not usable({}), "a campaign block with no gates key (a 0.5.x mod)"
    assert not usable(gates_block([gate("a", (0.0, 0.0, 10.0), 3)], ordered=False)), "gates_ordered false"
    assert not usable(gates_block([])), "an empty array must not divide by zero"
    assert not usable(gates_block([{"key": "a", "hops": 0}])), "a gate with no pos is not a gate"


def test_gates_guard_threshold_is_configurable():
    quarter = ratio_block(1, 4)
    assert not usable(quarter) and usable(quarter, hops_min_frac=0.25)
    assert not usable(ratio_block(6, 12), hops_min_frac=0.6), "the first knob if 6-1 ever wedges"
    assert not usable(ratio_block(6, 12, altar_only=1), hops_min_frac=0.6), "and it still bites with a phase-2 gate"


def test_a_dropped_ladder_pays_nothing_and_has_no_target():
    camp = ratio_block(1, 32)  # 8-3
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.target is None and p.best_hops is None
    assert p.update(camp, (0.0, 0.0, 1.0)) == (0, 0.0) and p.gates_reached == 0


# -- S4: the skull-carry sub-goal ----------------------------------------------------------
#
# Modelled on Level 1-1's red leg as parsed from the scene bundle: the gate at 20,-10,381 (hops 2) is held shut
# by the altar at 0,-7,381, whose only live source is the pedestal skull 133.5 m away at (81, -2.2, 275). The
# level also ships a dead twin of the altar, a phantom item under a permanently disabled node, and a decoration
# skull inside the destination altar -- all three must be ignored.

GATE_KEY = "20,-10,381"


def altar(key, pos, item, *, filled=False, doors=(GATE_KEY,), ancestors=1, reverse=()):
    return {"key": key, "pos": list(pos), "item": item, "filled": filled, "active": True,
            "inactive_ancestors": ancestors,
            "doors": [{"key": k, "pos": [0.0, 0.0, 0.0]} for k in doors],
            "reverse_doors": [{"key": k, "pos": [0.0, 0.0, 0.0]} for k in reverse]}


def item(key, pos, kind, *, held=False, placed=True, placed_in=None, ancestors=1, active_self=True):
    return {"key": key, "pos": list(pos), "item": kind, "held": held, "placed": placed,
            "placed_in": placed_in, "active": True, "active_self": active_self, "inactive_ancestors": ancestors}


def skull_block(*, altars, items, filled_gate=False, needs="SkullRed"):
    """A one-gate level whose gate needs an item. `needs` None is a mod that sends no needs_item at all."""
    g = gate(GATE_KEY, (0.0, 0.0, 100.0), 2)
    if needs is not None:
        g["needs_item"] = None if filled_gate else needs
    block = gates_block([g])
    block["altars"], block["items"] = list(altars), list(items)
    return block


PEDESTAL = item("81,-2,275", (81.0, -2.2, 275.0), "SkullRed", placed_in="81,-2,275")
DECORATION = item("0,-7,381", (0.0, -6.86, 381.0), "SkullRed", placed_in="0,-7,381", ancestors=2, active_self=False)
LIVE_ALTAR = altar("0,-7,381", (0.0, -6.76, 381.0), "SkullRed")
DEAD_TWIN = altar("0,-7,381#2", (0.0, -6.6, 381.0), "SkullRed", ancestors=2)


def targeted(camp, pos=(0.0, 0.0, 0.0)):
    p = GateProgress()
    p.new_level_load(camp, pos)
    p.reset_episode()
    p.retarget(camp, pos)
    return p.target


def test_subgoal_walks_fetch_then_carry_then_the_gate():
    fetch = targeted(skull_block(altars=[LIVE_ALTAR, DEAD_TWIN], items=[PEDESTAL, DECORATION]))
    assert fetch["subgoal"] == SUBGOAL_ITEM and fetch["gate_key"] == GATE_KEY
    assert fetch["key"] == "item:SkullRed" and fetch["pos"] == [81.0, -2.2, 275.0]
    assert fetch["hops"] == 2, "the sub-goal inherits the gate's hops, so it packs through the gate branch"

    carried = dict(PEDESTAL, held=True, placed=False, placed_in=None)
    carry = targeted(skull_block(altars=[LIVE_ALTAR, DEAD_TWIN], items=[carried, DECORATION]))
    assert carry["subgoal"] == SUBGOAL_ALTAR and carry["key"] == f"altar:SkullRed:{GATE_KEY}"
    assert carry["item"] == "SkullRed", "the held type, so carry protection and release share one source"
    assert carry["pos"] == [0.0, -7.76, 381.0], \
        "the live altar, never the dead twin -- and its COLLIDER centre, 1 m under the reported position"

    placed = item("81,-2,275", (0.0, -6.86, 381.0), "SkullRed", placed_in="0,-7,381")
    done = skull_block(altars=[dict(LIVE_ALTAR, filled=True), DEAD_TWIN], items=[placed], filled_gate=True)
    assert targeted(done)["key"] == GATE_KEY and "subgoal" not in targeted(done)


# Level 1-1's full ladder as parsed from the scene bundle: both middle rungs are skull-locked.
L11_BLUE_GATE, L11_RED_GATE = "81,-6,240", "20,-10,381"


def level_1_1():
    gates = [
        dict(gate("81,-6,188", (81.0, -6.0, 188.5), 0), needs_item=None),
        dict(gate(L11_BLUE_GATE, (81.0, -6.0, 239.5), 1, controller=False), needs_item="SkullBlue",
             altar_only=True),
        dict(gate(L11_RED_GATE, (20.5, -9.5, 381.0), 2, controller=False), needs_item="SkullRed"),
        dict(gate("16,20,427", (15.5, 20.5, 427.0), 3), needs_item=None),
    ]
    block = gates_block(gates, exit_pos=(81.0, -76.1, 91.0))
    block["altars"] = [altar("81,-4,251", (81.0, -3.76, 251.0), "SkullBlue", doors=(L11_BLUE_GATE,)),
                       altar("0,-7,381", (0.0, -6.76, 381.0), "SkullRed", doors=(L11_RED_GATE,))]
    block["items"] = [item("-15,27,427", (-15.0, 26.64, 427.0), "SkullBlue", placed_in="-15,27,427"),
                      item("81,-2,275", (81.0, -2.2, 275.0), "SkullRed", placed_in="81,-2,275")]
    return block


def test_standing_at_a_skull_locked_gate_is_not_reaching_it():
    """A door the agent physically cannot pass must not pay `gate` or move `best_hops`.

    It did both, and the consequence was worse than the stray +15: with `best_hops` at 1, `_choose_target`
    picks the rung BELOW the lock (`81,-6,188`, which carries no `needs_item`), `_subgoal` returns it
    unchanged, and the fetch/carry machine never fires again for the rest of the level load -- including every
    checkpoint-respawn episode, since `best_hops` and `reached` are level-load scoped.
    """
    camp = level_1_1()
    p = GateProgress()
    spawn = (0.0, 105.0, 253.0)
    p.new_level_load(camp, spawn)
    p.reset_episode()
    p.retarget(camp, spawn)
    assert p.target["key"] == "item:SkullBlue", "the sub-goal engages at spawn"

    paid, _ = p.update(camp, (81.0, -6.0, 243.0))  # 3.5 m from the blue-locked door
    assert paid == 0 and p.best_hops is None and p.hops_reached == set()
    assert p.reached == set() and p.gates_reached == 0
    assert p.target["key"] == "item:SkullBlue", "and it still points at the skull, not past the lock"

    # Once the altar is filled the mod drops needs_item, and the gate reaches and pays exactly as any other.
    opened = level_1_1()
    opened["gates"][1]["needs_item"] = None
    opened["altars"][0]["filled"] = True
    q = GateProgress()
    q.new_level_load(opened, spawn)
    q.reset_episode()
    assert q.update(opened, (81.0, -6.0, 243.0))[0] == 1 and q.best_hops == 1


def test_a_skull_locked_gate_does_not_block_the_rest_of_the_ladder():
    """Only the locked gate is unreachable: an ordinary gate at any rung still counts and still retargets."""
    camp = level_1_1()
    p = GateProgress()
    p.new_level_load(camp, (0.0, 105.0, 253.0))
    p.reset_episode()
    assert p.update(camp, (15.5, 20.5, 427.0))[0] == 1  # the hops-3 walk-up door
    assert p.best_hops == 3
    assert p.target["key"] == "item:SkullRed", "the hops-2 rung is red-locked, so the target is the red skull"


def test_the_dead_twin_never_becomes_the_target_after_a_placement():
    """X1 of the design spec: a dead zone can never fill, so counting it would undo the puzzle.

    The machine would drop back to "carry", find the just-placed skull, send the agent to punch it out of the
    altar -- and `ItemPlaceZone.CheckItem`'s empty branch calls Close() on the door it had opened.
    """
    placed = item("81,-2,275", (0.0, -6.86, 381.0), "SkullRed", placed_in="0,-7,381")
    # The mod still reports needs_item here only if it counted the dead twin; Python must not target it either.
    camp = skull_block(altars=[dict(LIVE_ALTAR, filled=True), DEAD_TWIN], items=[placed], needs="SkullRed")
    target = targeted(camp)
    assert target["key"] == GATE_KEY and "subgoal" not in target


def test_an_altar_is_aimed_at_its_collider_centre_not_its_transform():
    """F2. `Punch.AltHit` only places when the ray hits the GameObject carrying the `ItemPlaceZone`, and every
    one of the campaign's 104 zones is the same prefab: a trigger box of local size (2.2, 3.5, 2.2) centred at
    (0, -1.25, 0) on a transform scaled (0.9, 0.8, 0.8). The collider centre is therefore 1 m below the reported
    position, which is itself only 0.4 m under the box lid. Aiming at the reported position did not place in
    game; aiming ~1.25 m below it did.
    """
    assert altar_aim_point(LIVE_ALTAR) == [0.0, -7.76, 381.0], "no aim_pos: the measured 1 m drop"
    exact = dict(LIVE_ALTAR, aim_pos=[0.0, -7.6, 381.0])
    assert altar_aim_point(exact) == [0.0, -7.6, 381.0], "the mod's own figure wins, so the odd zone is right too"

    carried = dict(PEDESTAL, held=True, placed=False, placed_in=None)
    camp = skull_block(altars=[exact, DEAD_TWIN], items=[carried, DECORATION])
    assert targeted(camp)["pos"] == [0.0, -7.6, 381.0], "and it is what the carry leg walks to and aims at"


def test_a_dead_twin_is_judged_relative_to_its_twins_not_against_a_constant():
    """F3/M14. `inactive_ancestors` counts the zone's chain PLUS whatever of the room above it is switched off,
    so it shifts when the room lights: measured on 1-1, the live altar and its twin read 1/2 on a fresh load and
    0/1 after the checkpoint respawn switched that room on. The old absolute test (`> 1`) therefore stopped
    filtering the twin exactly when the player arrived, and the gate kept `needs_item` however often the puzzle
    was solved. A shared room contributes equally to both halves, so the relative test cannot be shifted.
    """
    def pair(live_anc, twin_anc, *, live_filled=False):
        live = dict(LIVE_ALTAR, inactive_ancestors=live_anc, filled=live_filled)
        twin = dict(DEAD_TWIN, inactive_ancestors=twin_anc)
        return live, twin, [live, twin]

    for live_anc, twin_anc in ((1, 2), (0, 1), (2, 3)):  # fresh load, room lit, and a room off two levels deep
        live, twin, altars = pair(live_anc, twin_anc)
        assert not dead_twin(live, altars) and dead_twin(twin, altars), f"{live_anc}/{twin_anc}"

    # The bug itself: with the live half FILLED the twin must still be filtered, or the lock never clears.
    live, twin, altars = pair(0, 1, live_filled=True)
    camp = {"altars": altars}
    assert wanting_altars(camp, "SkullRed") == [], "nothing still wants a red skull: the gate is solved"

    # A lone zone under a switched-off room is NOT a twin, however many inactive ancestors it reports. The old
    # rule dropped it at 2 and would have lost the lock entirely.
    lonely = dict(LIVE_ALTAR, inactive_ancestors=4)
    assert not dead_twin(lonely, [lonely])
    assert wanting_altars({"altars": [lonely]}, "SkullRed") == [lonely]

    # Same item and counts but a different door set, or a different position: two puzzles, not a pair.
    elsewhere = altar("500,0,500", (500.0, 0.0, 500.0), "SkullRed", ancestors=3)
    assert not dead_twin(elsewhere, [LIVE_ALTAR, elsewhere]), "far away: not a co-located twin"
    other_door = dict(DEAD_TWIN, doors=[{"key": "9,9,9", "pos": [0.0, 0.0, 0.0]}])
    assert not dead_twin(other_door, [LIVE_ALTAR, other_door]), "drives another door: its own lock"

    # An older mod sends no inactive_ancestors at all: every entry reads 0, nothing is strictly fewer, and the
    # filter is inert.
    bare = [{"key": "a", "pos": [0.0, 0.0, 0.0], "item": "SkullRed", "filled": False, "doors": []},
            {"key": "a#2", "pos": [0.0, 0.0, 0.0], "item": "SkullRed", "filled": False, "doors": []}]
    assert [dead_twin(a, bare) for a in bare] == [False, False]


def test_subgoal_returns_the_gate_when_there_is_nothing_to_fetch():
    plain = gates_block([gate(GATE_KEY, (0.0, 0.0, 100.0), 2)])
    assert targeted(plain)["key"] == GATE_KEY, "no needs_item: an old mod, or an ordinary door"
    assert targeted(skull_block(altars=[], items=[]))["key"] == GATE_KEY, "no altars"
    assert targeted(skull_block(altars=[LIVE_ALTAR], items=[]))["key"] == GATE_KEY, "no items"
    wrong_gate = altar("9,9,9", (5.0, 0.0, 5.0), "SkullRed", doors=("0,0,0",))
    assert targeted(skull_block(altars=[wrong_gate], items=[PEDESTAL]))["key"] == GATE_KEY, "wired to another door"
    reverse = altar("9,9,9", (5.0, 0.0, 5.0), "SkullRed", doors=(), reverse=(GATE_KEY,))
    assert targeted(skull_block(altars=[reverse], items=[PEDESTAL]))["key"] == GATE_KEY, "a reverse door closes it"
    other_type = skull_block(altars=[altar("a", (5.0, 0.0, 5.0), "SkullBlue")], items=[PEDESTAL], needs="SkullBlue")
    assert targeted(other_type)["key"] == GATE_KEY, "no source of the type the altar accepts"


def test_subgoal_rejects_the_phantom_and_the_decoration_but_takes_the_pedestal():
    phantom = item("x", (1.0, 0.0, 1.0), "SkullRed", placed_in="x", ancestors=2)  # under a disabled Altar node
    camp = skull_block(altars=[LIVE_ALTAR], items=[phantom, DECORATION, PEDESTAL])
    target = targeted(camp)
    assert target["subgoal"] == SUBGOAL_ITEM and target["pos"] == [81.0, -2.2, 275.0], \
        "the phantom is 1 m away and the pedestal 290 m, so only the liveness filter can be choosing"
    # "Not held and not placed" finds nothing on 1-1: every ItemIdentifier starts inside an ItemPlaceZone.
    assert PEDESTAL["placed"] and PEDESTAL["placed_in"] == "81,-2,275"
    in_the_target_altar = item("y", (1.0, 0.0, 1.0), "SkullRed", placed_in=LIVE_ALTAR["key"])
    assert targeted(skull_block(altars=[LIVE_ALTAR], items=[in_the_target_altar]))["key"] == GATE_KEY


def test_subgoal_needs_a_player_and_keeps_an_existing_sub_goal():
    camp = skull_block(altars=[LIVE_ALTAR], items=[PEDESTAL])
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.target["subgoal"] == SUBGOAL_ITEM
    p.retarget(camp, None)  # a frame with no player keeps the target as it was
    assert p.target["subgoal"] == SUBGOAL_ITEM and p.target["key"] == "item:SkullRed"


def test_subgoal_keys_are_by_type_so_a_re_keyed_instance_cannot_re_seed_the_budget():
    """`best_dist` is seeded once per key per episode and never re-seeded; a respawn re-instantiates skulls.

    With instance keys, dying next to the skull would hand back the whole ~20-point approach every time.
    """
    camp = skull_block(altars=[LIVE_ALTAR], items=[PEDESTAL])
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    seeded = dict(p.best_dist)
    assert set(seeded) == {"item:SkullRed"}
    # Same skull, brand-new mod-side key and object: the target key, and so the budget, is unchanged.
    rekeyed = skull_block(altars=[LIVE_ALTAR], items=[dict(PEDESTAL, key="81,-2,275#7")])
    p.retarget(rekeyed, (0.0, 0.0, 0.0))
    assert dict(p.best_dist) == seeded and p.target["key"] == "item:SkullRed"


def test_subgoal_keys_cannot_collide_with_a_gate_or_the_exit():
    for key in ("item:SkullRed", f"altar:SkullRed:{GATE_KEY}"):
        assert not key.replace(":", "").isdigit() and key != GATE_EXIT_KEY
        assert ":" in key, "a real Door key is always x,y,z with no colon"


def test_target_kind_slots_are_off_by_default():
    camp = skull_block(altars=[LIVE_ALTAR], items=[PEDESTAL])
    off = GateProgress()
    off.new_level_load(camp, (0.0, 0.0, 0.0))
    off.retarget(camp, (0.0, 0.0, 0.0))
    assert off.target["open"] is False and off.target["locked"] is False

    on = GateProgress(target_kind_slots=True)
    on.new_level_load(camp, (0.0, 0.0, 0.0))
    on.retarget(camp, (0.0, 0.0, 0.0))
    assert on.target["open"] is True and on.target["locked"] is False  # "the target is an item"
    carried = skull_block(altars=[LIVE_ALTAR], items=[dict(PEDESTAL, held=True, placed=False, placed_in=None)])
    on.retarget(carried, (0.0, 0.0, 0.0))
    assert on.target["open"] is False and on.target["locked"] is True  # "the target is an altar"


def test_shuttling_between_the_skull_and_the_altar_pays_each_leg_once():
    free = skull_block(altars=[LIVE_ALTAR], items=[PEDESTAL])
    carried = skull_block(altars=[LIVE_ALTAR], items=[dict(PEDESTAL, held=True, placed=False, placed_in=None)])
    p = GateProgress()
    p.new_level_load(free, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(free, (0.0, 0.0, 0.0))
    # Walk to the skull, pick it up, walk back, drop it, walk to the skull again.
    out = sum(p.update(free, (x, -2.2, 275.0 * x / 81.0))[1] for x in (20.0, 40.0, 60.0, 81.0))
    back = sum(p.update(carried, (x, -6.76, 381.0))[1] for x in (60.0, 30.0, 0.0))
    again = sum(p.update(free, (x, -2.2, 275.0))[1] for x in (30.0, 60.0, 81.0))
    assert out > 0 and back > 0
    assert again == 0.0, "the fetch leg's budget was spent the first time; shuttling earns nothing"


def test_open_gate_counts_as_reached_at_double_range():
    closed = gates_block([gate("a", (0.0, 0.0, 0.0), 1)])
    p = GateProgress()
    p.new_level_load(closed, (0.0, 0.0, 100.0))
    p.retarget(closed, (0.0, 0.0, 12.0))
    assert p.best_hops is None  # 12 m away, outside the 8 m radius
    opened = gates_block([gate("a", (0.0, 0.0, 0.0), 1, open=True)])
    q = GateProgress()
    q.new_level_load(opened, (0.0, 0.0, 100.0))
    q.retarget(opened, (0.0, 0.0, 12.0))
    assert q.best_hops == 1  # an open door doubles both radii


def test_reach_is_a_cylinder():
    camp = gates_block([gate("a", (0.0, 0.0, 0.0), 1)])
    roof = GateProgress()
    roof.new_level_load(camp, (0.0, 0.0, 100.0))
    roof.retarget(camp, (3.0, 12.0, 0.0))
    assert roof.best_hops is None, "standing on the roof over a door is not passing it"
    beside = GateProgress()
    beside.new_level_load(camp, (0.0, 0.0, 100.0))
    beside.retarget(camp, (7.0, 4.0, 0.0))
    assert beside.best_hops == 1


def test_inactive_gates_are_ignored():
    camp = gates_block([gate("a", (0.0, 0.0, 0.0), 1, active=False), gate("b", (0.0, 0.0, 80.0), 0)])
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 100.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.best_hops is None and p.target["key"] == "b"


def test_hops_shifting_without_moving_pays_nothing():
    """A Scan() that renumbers the graph must not itself pay: best_hops only falls when a gate is reached."""
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    assert p.update(camp, (0.0, 0.0, 0.0))[0] == 1
    lower = gates_block([gate(g["key"], g["pos"], max(0, g["hops"] - 1)) for g in LADDER])
    assert p.update(lower, (0.0, 0.0, 0.0))[0] == 1  # the same gate now reads hops 8: one rung, paid once
    assert p.update(lower, (0.0, 0.0, 0.0))[0] == 0


def test_duplicate_gate_keys_do_not_freeze_the_target():
    """CampaignPatches.Key rounds to whole metres, so the mod appends #2 to a collision. Keys are opaque."""
    camp = gates_block([gate("40,1,408", (40.0, 1.0, 408.0), 3), gate("40,1,408#2", (40.3, 1.0, 408.4), 2),
                        gate("40,1,500", (40.0, 1.0, 500.0), 1)])
    p = GateProgress()
    p.new_level_load(camp, (40.0, 1.0, 300.0))
    p.reset_episode()
    p.retarget(camp, (40.0, 1.0, 300.0))
    assert p.target["key"] == "40,1,408"
    assert p.update(camp, (40.0, 1.0, 408.0))[0] == 1  # both doors are within reach; the first rung pays once
    assert p.best_hops == 2 and p.reached == {"40,1,408", "40,1,408#2"}
    assert p.target["key"] == "40,1,500"  # the duplicate key did not freeze it


def test_truncated_gates_still_target():
    camp = gates_block(LADDER[:4], truncated=True)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    assert p.target is not None
    assert p.update(camp, (0.0, 0.0, 0.0))[0] == 1


def test_gates_without_controller_active_still_work():
    """Graceful degradation: a mod that does not report controller_active must not break targeting."""
    camp = gates_block([gate("a", (0.0, 0.0, 0.0), 1, controller=None)])
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, 100.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 100.0))
    assert p.target["key"] == "a"
    assert p.update(camp, (0.0, 0.0, 0.0))[0] == 1


def test_gate_progress_survives_a_step_without_a_player():
    camp = gates_block(LADDER)
    p = GateProgress()
    p.new_level_load(camp, (0.0, 0.0, -60.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, -60.0))
    held = p.target
    assert p.update(camp, None) == (0, 0.0)
    assert p.target is held  # a frame with no player keeps pointing where it was


# ---------------------------------------------------------------------------
# Target patience, parking and the A2 fallback
# (docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md)
# ---------------------------------------------------------------------------

# A collapsed ladder in miniature, with Level 0-3's shape: the gate nearest the spawn is already hops 1, the
# hops 0 door is unreachable straight up, and the walkable route runs AWAY from the exit through hops 2 and 3
# before it comes back. `PATIENCE` is the default 20 game seconds at 30 fps / frameskip 2.
PATIENCE = 300
COLLAPSED = [
    gate("0,80,0", (0.0, 80.0, 0.0), 0),       # 80 m straight up, the ladder's answer and unreachable
    gate("0,0,10", (0.0, 0.0, 10.0), 1),       # touched from below at the spawn, which locks best_hops at 1
    gate("0,0,60", (0.0, 0.0, 60.0), 2),       # the real next leg, further from the exit
    gate("0,0,110", (0.0, 0.0, 110.0), 3),     # and the one after it
]


def patient(**kw) -> GateProgress:
    kw.setdefault("patience_steps", PATIENCE)
    return GateProgress(**kw)


def dwell(progress: GateProgress, camp: dict, pos, steps: int, **kw) -> tuple[int, float]:
    """Stands still for `steps` decisions, which is what runs the patience clock down."""
    return walk_kw(progress, camp, [pos] * steps, **kw)


def walk_kw(progress: GateProgress, camp: dict, positions, **kw) -> tuple[int, float]:
    paid, approach = 0, 0.0
    for pos in positions:
        g, a = progress.update(camp, pos, **kw)
        paid += g
        approach += a
    return paid, approach


def test_patience_is_off_by_default_so_the_tracker_is_the_pre_patience_one():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = GateProgress()  # no patience_steps
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), 3 * PATIENCE)
    assert p.target["key"] == "0,80,0" and not p.parked, "with patience 0 nothing is ever parked"


def test_a_target_that_never_gets_closer_is_parked_and_the_fallback_takes_over():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))  # the spawn already touches the hops 1 gate
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    assert p.best_hops == 1 and p.target["key"] == "0,80,0"
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE - 1)
    assert p.target["key"] == "0,80,0", "the clock has not run out yet"
    dwell(p, camp, (0.0, 0.0, 10.0), 1)
    assert p.parked == {"0,80,0"} and p.parks == 1
    assert p.target["key"] == "0,0,60", "the nearest unreached, unparked gate at any hop count"


def test_parking_needs_somewhere_better_to_go():
    """The whole reason Level 0-1 is left alone: with nothing nearer unreached, the ladder's answer stands."""
    camp = gates_block([gate("0,0,0", (0.0, 0.0, 0.0), 0), gate("0,0,40", (0.0, 0.0, 40.0), 1)], exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 40.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 40.0))
    assert p.target["key"] == "0,0,0"
    dwell(p, camp, (0.0, 0.0, 200.0), 4 * PATIENCE)  # far from both, and the only other gate is reached
    assert not p.parked and p.parks == 0
    assert p.target["key"] == "0,0,0"


def test_an_arena_holding_the_door_shut_suspends_the_clock():
    camp = gates_block(COLLAPSED, exit_pos=None)
    held = dict(camp, arena_enemies_alive=4)
    p = patient()
    p.new_level_load(held, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(held, (0.0, 0.0, 10.0))
    dwell(p, held, (0.0, 0.0, 10.0), 2 * PATIENCE - 1)  # just inside the bound the suspension is allowed
    assert not p.parked, "a door an ActivateArena wave is holding cannot be approached; that is not the ladder's fault"
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)  # the wave is dead and the door still never gets closer
    assert p.parked == {"0,80,0"}


def test_an_arena_that_never_dies_still_parks_the_gate_eventually():
    """The suspension is BOUNDED, because `arena_enemies_alive` is not scoped to the target's own arena.

    The mod counts every enemy whose `ActivateNextWave` has not run anywhere on the level, so a wave the agent
    walks away from and never clears used to freeze the clock for the rest of the level load: an unreachable
    door then stayed the target forever and `targets_parked` read 0, which on the dashboard looks exactly like
    a monotone level with the mechanism correctly inert. Three of the four collapsed levels in the live
    curriculum (1-1, 1-2, 2-3) have arenas, and no probe data exists for any of them.
    """
    held = dict(gates_block(COLLAPSED, exit_pos=None), arena_enemies_alive=4)
    p = patient()
    p.new_level_load(held, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(held, (0.0, 0.0, 10.0))
    dwell(p, held, (0.0, 0.0, 10.0), 2 * PATIENCE)
    assert p.parked == {"0,80,0"} and p.parks == 1, "forced at clock + suspended == 2 * patience_steps"
    assert p.target["key"] == "0,0,60", "and the fallback takes over, as on any other level"
    # 600 decisions is still inside the 675 of `stuck_seconds` 45 at 30 fps / frameskip 2, so the episode's own
    # budget remains the outer bound and the park happens with room to act on it.
    assert 2 * PATIENCE < 675


def test_a_kill_or_a_style_event_suspends_the_clock():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    for _ in range(2 * PATIENCE - 1):
        p.update(camp, (0.0, 0.0, 10.0), fought=True)
    assert not p.parked
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.parked == {"0,80,0"}


def test_an_endless_fight_under_the_door_still_parks_it():
    """`fought` is bounded by the same counter: farming enemies under an unreachable door is not progress."""
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    for _ in range(2 * PATIENCE):
        p.update(camp, (0.0, 0.0, 10.0), fought=True)
    assert p.parked == {"0,80,0"} and p.target["key"] == "0,0,60"


def test_an_approach_restarts_the_clock():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    for k in range(1, 7):
        dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE - 2)
        assert p.update(camp, (0.0, 10.0 * k, 10.0))[1] > 0.0  # climbing 10 m further each time is a new best
        p.update(camp, (0.0, 0.0, 10.0))
    assert not p.parked, "every window was broken by a real approach"


def test_a_parked_gate_comes_back_when_the_player_gets_nearer_than_it_was_parked_at():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient(unpark_m=2.0)
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    parked_at = math.dist((0.0, 0.0, 10.0), (0.0, 80.0, 0.0))
    assert p.parked == {"0,80,0"} and abs(p.park_best["0,80,0"] - parked_at) < 1e-6
    p.update(camp, (0.0, 0.0, 10.0))
    assert p.parked == {"0,80,0"}, "standing still is not getting closer"
    p.update(camp, (0.0, 1.9, 10.0))
    assert p.parked == {"0,80,0"}, "under the 2 m threshold"
    p.update(camp, (0.0, 4.0, 10.0))
    assert not p.parked
    assert p.park_best["0,80,0"] == parked_at, \
        "the baseline outlives the park: a re-park can only lower it, never read a farther spot as a fresh start"
    assert p.target["key"] == "0,80,0", "the ladder takes it back the moment it is un-parked"


def test_parks_survive_an_episode_reset_but_the_clock_restarts():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.parked == {"0,80,0"}
    p.mark_paid(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    assert p.parked == {"0,80,0"}, "a park is a fact about this level load, not about one episode"
    assert p.target["key"] == "0,0,60"
    assert p._clock == 0, "the clock measures the current attempt, whose approach baseline was just cleared"


def test_a_new_level_load_forgets_every_park():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.parked and p.parks == 1
    p.new_level_load(camp, (0.0, 0.0, -50.0))
    assert not p.parked and not p.park_best and not p.paid_fallback
    assert p.parks == 1, "the counter is an env-lifetime total; the env reports parks per episode by difference"


def test_reaching_a_fallback_target_pays_a_gate_instalment_once_per_level_load():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    assert dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)[0] == 0
    assert p.target["key"] == "0,0,60"
    paid, _ = walk_kw(p, camp, [(0.0, 0.0, float(z)) for z in range(12, 62, 2)])
    assert paid == 1, "a forward leg away from the exit earns something, which the monotone rule never did"
    assert p.best_hops == 1, "and it does NOT move the ladder: hops 2 is above the floor"
    back, _ = walk_kw(p, camp, [(0.0, 0.0, 10.0), (0.0, 0.0, 60.0)])
    assert back == 0, "once per key per level load"


def test_a_door_off_the_ladder_is_never_a_target_and_never_pays():
    """0-2's secret arena shape: an `altar_only` door with no `hops` is not on the exit chain at all."""
    secret = gate("6,0,60", (6.0, 0.0, 60.0), None)
    secret["altar_only"] = True
    camp = gates_block(COLLAPSED + [secret], exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    paid, _ = walk_kw(p, camp, [(0.0, 0.0, float(z)) for z in range(12, 62, 2)])
    assert paid == 1 and "6,0,60" not in p.reached, "a door with no hops is neither targeted nor paid"


def test_a_second_door_on_a_rung_already_reached_pays_nothing():
    """A4's stated goal is "no incentive to tour doors", and per-KEY payment does not deliver it.

    `_pick` re-selects the nearest unreached, unparked gate every step, so door after door becomes the fallback
    target in turn and used to pay 15 each: measured on eight side doors sharing one rung, a tour collected six
    instalments where the ladder's depth was two, and on 8-1's 52 phase-1 gates the ceiling was 780 against
    `level_complete` 100. The instalment is bounded by the ladder's own unit instead, so the twin here is worth
    nothing while every genuine forward leg still pays (see the 0-3 route test in test_ladder_replay.py).
    """
    twin = gate("6,0,60", (6.0, 0.0, 60.0), 2)  # 6 m from the route gate, inside the same reach cylinder
    camp = gates_block(COLLAPSED + [twin], exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.target["key"] == "0,0,60", "the nearest unreached gate, by a third of a metre"
    walk = [(0.0, 0.0, float(z)) for z in range(12, 62, 2)]
    paid, _ = walk_kw(p, camp, walk)
    assert paid == 1 and {"0,0,60", "6,0,60"} <= p.reached, "one rung reached, one instalment, two doors"
    assert walk_kw(p, camp, [(0.0, 0.0, 10.0)] + walk)[0] == 0, "re-walking the pair pays nothing"


def test_touring_many_doors_on_one_rung_cannot_out_earn_the_ladder():
    """The ceiling is the ladder depth, not the gate count. Eight doors on one rung; the tour pays once."""
    hub = (0.0, 0.0, 300.0)
    ring = [gate(f"side{i}", (hub[0] + 60.0 * math.cos(i * math.pi / 4), 0.0,
                              hub[2] + 60.0 * math.sin(i * math.pi / 4)), 4) for i in range(8)]
    camp = gates_block([gate("0,120,300", (0.0, 120.0, 300.0), 3),  # the unreachable ladder answer, 120 m up
                        gate("0,0,0", (0.0, 0.0, 0.0), 5)] + ring, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 0.0))  # touch the hops 5 door: best_hops 5, so rung 4 is the ring
    assert p.best_hops == 5, "the ring is not within reach of the spawn"
    p.reset_episode()
    p.retarget(camp, hub)
    paid = 0
    for g in ring:  # reach every door on the rung, lingering at each so the sticky fallback settles on it
        paid += walk_kw(p, camp, [tuple(g["pos"])] * 40)[0]
        paid += dwell(p, camp, hub, 40)[0]
    assert {g["key"] for g in ring} <= p.reached, "all eight were walked through"
    assert len(p.paid_fallback) >= 4, "several of them really did become the fallback target in turn"
    assert paid == 1, f"one new rung, one instalment, however many of its doors were toured (paid {paid})"


def test_a_respawn_onto_the_fallback_target_is_absorbed_not_paid():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.target["key"] == "0,0,60"
    p.mark_paid(camp, (0.0, 0.0, 60.0))  # the respawn puts the player on the gate the fallback was pointing at
    assert p.update(camp, (0.0, 0.0, 60.0))[0] == 0


def test_the_fallback_is_sticky_within_the_hysteresis():
    twins = [gate("0,80,0", (0.0, 80.0, 0.0), 0), gate("0,0,10", (0.0, 0.0, 10.0), 1),
             gate("-30,0,60", (-30.0, 0.0, 60.0), 2), gate("30,0,60", (30.0, 0.0, 60.0), 2)]
    camp = gates_block(twins, exit_pos=None)
    p = patient(fallback_hysteresis_m=10.0)
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    first = p.target["key"]
    p.update(camp, (5.0 if first == "-30,0,60" else -5.0, 0.0, 10.0))  # a step toward the other twin
    assert p.target["key"] == first, "a rival must beat the sticky choice by more than the hysteresis"
    p.update(camp, (25.0 if first == "-30,0,60" else -25.0, 0.0, 10.0))
    assert p.target["key"] != first, "far enough, and the nearest wins again"


def test_the_exit_is_never_parked():
    camp = gates_block([gate("0,0,0", (0.0, 0.0, 0.0), 0)], exit_pos=(0.0, 0.0, 200.0))
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.target["key"] == GATE_EXIT_KEY
    dwell(p, camp, (0.0, 0.0, 0.0), 4 * PATIENCE)
    assert not p.parked and p.target["key"] == GATE_EXIT_KEY


def test_a_stalled_skull_leg_is_never_parked_at_all():
    """A carry is exempt from the clock, and this one is a safety rule rather than a tuning choice.

    `env._protect_carry` drops the punch button only while `gates.target` is the ALTAR sub-goal of a held item,
    because protection and release have to come from one source or the button can be taken away and never given
    back. Parking the gate deletes the leg, `_subgoal` never rebuilds it, and the protection switches off with
    the skull still in the player's hands -- and the live policy presses punch on ~35% of decisions, which
    `Punch.ActiveStart` turns into a throw. On 1-1 the gate is the only route forward, so the level load is lost.
    A skull-locked door is held shut by its altar, not by geometry, exactly like the arena case §3 exempts.
    """
    camp = skull_block(altars=[LIVE_ALTAR], items=[PEDESTAL])
    camp["gates"].append(gate("0,0,60", (0.0, 0.0, 60.0), 3))  # somewhere nearer the fallback could have gone
    here = (0.0, 0.0, 90.0)
    p = patient()
    p.new_level_load(camp, here)
    p.reset_episode()
    p.retarget(camp, here)
    assert p.target["subgoal"] == SUBGOAL_ITEM and p.target["gate_key"] == GATE_KEY
    dwell(p, camp, here, 6 * PATIENCE)
    assert not p.parked and p.parks == 0, "a fetch leg is not parked, however long it stalls"
    assert p.target["gate_key"] == GATE_KEY, "so the carry machine is still pointing at the skull"


def test_a_gate_still_wanting_a_skull_is_never_parked():
    """The same rule keyed on the block rather than on the leg, for the states where `_subgoal` returns the gate
    itself (no free source, or only dead twins left): its approach is blocked by the altar, not by geometry."""
    camp = skull_block(altars=[LIVE_ALTAR], items=[])  # nothing to fetch, so the target IS the locked gate
    camp["gates"].append(gate("0,0,60", (0.0, 0.0, 60.0), 3))
    here = (0.0, 0.0, 90.0)
    p = patient()
    p.new_level_load(camp, here)
    p.reset_episode()
    p.retarget(camp, here)
    assert p.target["key"] == GATE_KEY and not p.target.get("subgoal")
    dwell(p, camp, here, 6 * PATIENCE)
    assert not p.parked and p.target["key"] == GATE_KEY


def test_a_fallback_target_that_never_gets_closer_is_parked_too():
    """Finding 1: the patience mechanism must not protect exactly one hand-over and then switch itself off.

    `_pick` chooses the fallback as the nearest unreached, unparked, active gate and `_nearer_unreached` filters
    that identical set, so nothing can ever be strictly nearer than a fallback target. Requiring "somewhere
    better to go" for it made every fallback permanent: reproduced as 4,000 consecutive decisions aimed through
    a ceiling with `targets_parked` reading 1, i.e. the wedge moved one door over and the telemetry showed the
    mechanism as having fired. A fallback hands over to the next candidate instead, which is the A2 rule already.
    """
    camp = gates_block([gate("0,30,0", (0.0, 30.0, 0.0), 1),     # 30 m up through the ceiling
                        gate("0,60,0", (0.0, 60.0, 0.0), 2),     # the ladder's answer, 60 m up
                        gate("0,0,200", (0.0, 0.0, 200.0), 3),   # touched from the wrong side, so best_hops 3
                        gate("0,0,400", (0.0, 0.0, 400.0), 4)], exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 200.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.target["key"] == "0,60,0", "the ladder's rung, and unreachable"
    dwell(p, camp, (0.0, 0.0, 0.0), PATIENCE)
    assert p.parked == {"0,60,0"} and p.target["key"] == "0,30,0", "the fallback, and also unreachable"
    dwell(p, camp, (0.0, 0.0, 0.0), PATIENCE)
    assert p.target["key"] != "0,30,0", "which is parked in its turn rather than held for the level load"
    assert p.parked == {"0,60,0", "0,30,0"} and p.parks == 2


def test_when_every_gate_is_parked_the_target_becomes_the_exit():
    """A2's tail: "then the exit when none remain". Only reachable now that a fallback can be parked."""
    camp = gates_block([gate("0,30,0", (0.0, 30.0, 0.0), 0), gate("0,60,0", (0.0, 60.0, 0.0), 1),
                        gate("0,0,200", (0.0, 0.0, 200.0), 2)], exit_pos=(0.0, 0.0, 600.0))
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 200.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    dwell(p, camp, (0.0, 0.0, 0.0), 4 * PATIENCE)
    assert p.parked == {"0,30,0", "0,60,0"}
    assert p.target["key"] == GATE_EXIT_KEY, "aiming at the real goal beats aiming at a proved dead end"
    dwell(p, camp, (0.0, 0.0, 0.0), 4 * PATIENCE)
    assert p.target["key"] == GATE_EXIT_KEY and p.parks == 2, "and the exit is still never parked"


def test_wandering_away_and_stalling_again_cannot_buy_a_cheap_un_park():
    """Finding 3's real loophole: the un-park bar was measured from wherever the player happened to be standing
    when the clock next ran out, so the baseline could be RAISED by walking away.

    Measured on the recorded 0-3 probe: the first park recorded 40.9 m, and the second was taken at 77.5 m after
    the agent had wandered off -- at which point simply walking back into the pit cleared the bar for nothing.
    `park_best` is now the closest the player has been at any park of that key this level load, and a re-park
    can only lower it.
    """
    camp = gates_block(COLLAPSED, exit_pos=None)
    near, far = (0.0, 0.0, 10.0), (0.0, 0.0, 200.0)
    parked_at = math.dist(near, (0.0, 80.0, 0.0))
    p = patient(unpark_m=2.0)
    p.new_level_load(camp, near)
    p.reset_episode()
    p.retarget(camp, near)
    dwell(p, camp, near, PATIENCE)
    assert p.parked == {"0,80,0"} and p.park_best["0,80,0"] == parked_at
    p.update(camp, (0.0, 4.0, 10.0))  # a real 4 m gain clears the first bar
    assert not p.parked
    dwell(p, camp, far, PATIENCE)  # stall again, this time 215 m off, and the gate is parked a second time
    assert p.parked == {"0,80,0"} and p.park_best["0,80,0"] <= parked_at, "the baseline was not raised"
    assert dwell(p, camp, near, 5) and p.parked == {"0,80,0"}, \
        "so walking all the way back to where it was first parked is not 'getting closer than it has ever been'"


def test_bobbing_under_a_parked_door_cannot_un_park_it():
    """And inside the bar, nothing at all happens: the door stays parked while the agent shuffles beneath it."""
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient(unpark_m=2.0)
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.parked == {"0,80,0"}
    for y in (1.5, 0.0, 1.9, 0.5, 1.8, 0.0):  # every one of these is under a 2 m gain on the baseline
        p.update(camp, (0.0, y, 10.0))
        assert p.parked == {"0,80,0"}, f"y {y} is not a change of situation"


def test_the_un_park_bar_doubles_each_time_a_door_is_parked_again():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient(unpark_m=2.0)
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    assert p.park_count["0,80,0"] == 1
    p.update(camp, (0.0, 4.0, 10.0))  # a 3.9 m gain clears the first 2 m bar
    assert not p.parked
    dwell(p, camp, (0.0, 4.0, 10.0), PATIENCE)
    assert p.parked == {"0,80,0"} and p.park_count["0,80,0"] == 2
    p.update(camp, (0.0, 7.0, 10.0))  # a 2.8 m gain would have cleared 2 m; it does not clear 4 m
    assert p.parked == {"0,80,0"}, "a door already proved unreachable twice needs a real change of situation"
    p.update(camp, (0.0, 12.0, 10.0))
    assert not p.parked, "and genuinely climbing 7 m toward it is one"


def test_a_death_respawn_does_not_park_the_door_the_agent_is_walking_at():
    """Finding 4: the clock cannot be timed off `gate_approach`, which ruling R1 makes episode scoped.

    `env._respawn` calls `mark_paid` and deliberately not `reset_episode`, so `best_dist` keeps the pre-death
    minimum and the whole walk back over ground already covered pays nothing by design. A clock reading the
    reward therefore ran out on a target the agent was walking straight at -- measured, parked at decision 299
    of a flawless 389 m approach with 101 m still to go, and `park_best` was set from the stale pre-death
    minimum, so the correct door only came back once the agent was nearer than it had ever been.
    """
    camp = gates_block([gate("0,0,400", (0.0, 0.0, 400.0), 1), gate("40,0,330", (40.0, 0.0, 330.0), 0),
                        gate("0,0,0", (0.0, 0.0, 0.0), 2)], exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 0.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 0.0))
    approach = [(0.0, 0.0, float(z)) for z in range(0, 389)]
    walk_kw(p, camp, approach)
    assert p.target["key"] == "0,0,400" and p.best_dist["0,0,400"] == 12.0
    p.mark_paid(camp, (0.0, 0.0, 0.0))  # the death respawn, exactly as env._respawn drives it
    p.retarget(camp, (0.0, 0.0, 0.0))
    assert p.best_dist["0,0,400"] == 12.0, "R1: the approach already earned is not re-earned"
    paid, gained = walk_kw(p, camp, approach)
    assert gained == 0.0, "and the re-walk pays nothing, which is what used to run the clock out"
    assert not p.parked and p.target["key"] == "0,0,400", "the clock measures THIS attempt, so nothing is parked"


def test_a_respawn_restarts_the_patience_clock():
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE - 1)
    p.mark_paid(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE - 1)
    assert not p.parked, "a respawn is a new attempt, so the window starts again"
    dwell(p, camp, (0.0, 0.0, 10.0), 2)
    assert p.parked == {"0,80,0"}, "and it still runs out when the new attempt stalls too"


def test_a_fallback_gate_reporting_no_hops_does_not_raise():
    """Finding 8: `_gate_by_key` searches the full array, which carries `altar_only` doors with `hops: null`.

    An unguarded `int(gate["hops"])` raises out of `update`, past `env.step` (which catches only
    `BridgeRecovered`), past SubprocVecEnv's worker and into all twelve games -- the failure mode commit e9a53d4
    exists to prevent. One line of guard against the whole run.
    """
    camp = gates_block(COLLAPSED, exit_pos=None)
    p = patient()
    p.new_level_load(camp, (0.0, 0.0, 10.0))
    p.reset_episode()
    p.retarget(camp, (0.0, 0.0, 10.0))
    dwell(p, camp, (0.0, 0.0, 10.0), PATIENCE)
    target = p.target["key"]
    assert target == "0,0,60" and p._fallback
    for mutate in (lambda g: g.update(hops=None), lambda g: g.pop("hops")):
        broken = gates_block([dict(g) for g in COLLAPSED], exit_pos=None)
        mutate(next(g for g in broken["gates"] if g["key"] == target))
        assert p.update(broken, (0.0, 0.0, 60.0)) == (0, 0.0), "no payment, and above all no exception"


# ---------------------------------------------------------------------------
# ExitGuard: the banished FinalPit
# ---------------------------------------------------------------------------

L02_EXIT = [-199.0, -86.1, 277.0]  # Level 0-2's real FinalPit, measured on a fresh load
BANISH = 10000.0  # CheckPoint.Start:132 / ResetRoom:681


def exit_block(pos):
    return {"exit": {"pos": list(pos), "active": False}}


def test_exit_guard_freezes_the_first_report_of_a_level_load():
    g = ExitGuard()
    g.new_level_load()
    camp = exit_block(L02_EXIT)
    assert g.apply(camp) is False and camp["exit"]["pos"] == L02_EXIT
    assert g.frozen == L02_EXIT and not g.banished


def test_exit_guard_ignores_the_ten_thousand_metre_banish():
    g = ExitGuard()
    g.new_level_load()
    g.apply(exit_block(L02_EXIT))
    camp = exit_block([L02_EXIT[0] + BANISH, L02_EXIT[1], L02_EXIT[2]])
    assert g.apply(camp) is True
    assert camp["exit"]["pos"] == L02_EXIT, "the block is rewritten, so every consumer sees the real pit"
    assert g.banished and g.rejections == 1


def test_exit_guard_ignores_a_repeated_banish():
    """ResetRoom runs again on every respawn, so the offset is k * 10000."""
    g = ExitGuard()
    g.new_level_load()
    g.apply(exit_block(L02_EXIT))
    for k in (1, 2, 3):
        camp = exit_block([L02_EXIT[0] + k * BANISH, L02_EXIT[1], L02_EXIT[2]])
        assert g.apply(camp) is True and camp["exit"]["pos"] == L02_EXIT
    assert g.rejections == 3


def test_exit_guard_ignores_any_implausible_jump():
    g = ExitGuard(max_shift_m=100.0)
    g.new_level_load()
    g.apply(exit_block(L02_EXIT))
    camp = exit_block([L02_EXIT[0], L02_EXIT[1] + 500.0, L02_EXIT[2]])
    assert g.apply(camp) is True and camp["exit"]["pos"] == L02_EXIT


def test_exit_guard_accepts_a_small_move():
    g = ExitGuard(max_shift_m=100.0)
    g.new_level_load()
    g.apply(exit_block(L02_EXIT))
    moved = [L02_EXIT[0] + 3.0, L02_EXIT[1], L02_EXIT[2]]
    camp = exit_block(moved)
    assert g.apply(camp) is False and camp["exit"]["pos"] == moved and g.frozen == moved


def test_exit_guard_recovers_when_the_first_report_was_already_banished():
    """The banish only ever ADDS to x, so a whole multiple back toward 0 is the live pit, not a twin."""
    g = ExitGuard()
    g.new_level_load()
    g.apply(exit_block([L02_EXIT[0] + BANISH, L02_EXIT[1], L02_EXIT[2]]))
    camp = exit_block(L02_EXIT)
    assert g.apply(camp) is False and g.frozen == L02_EXIT and camp["exit"]["pos"] == L02_EXIT


def test_exit_guard_accepts_a_different_exit_after_a_new_level_load():
    g = ExitGuard()
    g.new_level_load()
    g.apply(exit_block(L02_EXIT))
    g.new_level_load()
    other = [157.0, 28.0, 640.0]
    camp = exit_block(other)
    assert g.apply(camp) is False and g.frozen == other and not g.banished


def test_exit_guard_is_silent_without_an_exit():
    g = ExitGuard()
    g.new_level_load()
    assert g.apply(None) is False
    assert g.apply({}) is False
    assert g.apply({"exit": None}) is False
    assert g.apply({"exit": {"pos": None}}) is False
    assert g.apply({"exit": {"pos": [float("nan"), 0.0, 0.0]}}) is False
    assert g.frozen is None


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
