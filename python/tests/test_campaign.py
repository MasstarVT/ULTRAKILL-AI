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
    GATE_EXIT_KEY,
    RANK_LETTERS,
    ExplorationArchive,
    GateProgress,
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
