"""`build_routes.apply_inserts`, the manual waypoint list. No game, no scene bundles:

    python tests/test_route_inserts.py     (or pytest)

The sibling of `test_route_drops.py`, and the same trick: the functions under test are lifted out of
`scripts/build_routes.py` by AST so importing them cannot pull in numpy, the bundle reader or a
scene.

What is pinned is rules 7-17 of the OVERRIDES_NAME comment, the ones an `insert_after` entry adds --
one test per rule, each of which fails if its rule is deleted -- plus the two properties the rest of
the pipeline relies on:

  * the waypoint lands immediately behind its anchor, so `hops` (assigned from the row order in
    `build_level`) stays a gapless n-1..0 and the ladder keeps its total order;
  * ALL-OR-NOTHING: one refused entry withholds every entry for that level, because half a waypoint
    chain leaves exactly the near-vertical leg the chain exists to remove, one rung further up.

The last section walks the AI's own recorded 0-3 completion through `GateProgress._is_reached`'s own
cylinder and asserts the six shipped rungs are credited IN ORDER, and that the places the live policy
actually stalls credit nothing. The trace samples are literals here on purpose: `python/runs/` is
gitignored, so a test that read the file would pass on this machine and vanish on any other.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "build_routes.py"
ROUTES = ROOT / "ultrakill_ai" / "routes"


def _lift(*names):
    """The named top-level functions of build_routes.py, compiled into a fresh namespace."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    want = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            want[node.name] = node
    missing = [n for n, v in want.items() if v is None]
    assert not missing, "build_routes.py no longer defines %r" % missing
    ns = {
        "math": math,
        "re": __import__("re"),
        "MIN_RUNGS": 3,
        "MAX_RUNGS": 24,
        "SEP": 16.0,
        "REACH_H": 8.0,
        "REACH_V": 6.0,
        "OVERRIDE_WAS_TOL_M": 1.0,
        "WAYPOINT_PREFIX": __import__("re").compile(r"^WP\d+ - \S"),
        "SEED_MARGIN_M": 4.0,
        "SEED_MARGIN_FRAC": 0.10,
        "dist3": lambda a, b: math.dist(a, b),
        "standability": lambda geom, rows: [{"name": r["name"], "reach": r["name"] not in geom}
                                            for r in rows],
    }
    module = ast.Module(body=[want[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)  # noqa: S102
    return [ns[n] for n in names]


apply_inserts, apply_overrides, co_credit, _vec3 = _lift(
    "apply_inserts", "apply_overrides", "co_credit", "_vec3")

REACH_H, REACH_V, SEP = 8.0, 6.0, 16.0


def rows(*names):
    """Rungs 100 m apart along z, which clears both SEP and the co-credit box by a wide margin."""
    return [{"name": n, "pos": [0.0, 0.0, 100.0 * i]} for i, n in enumerate(names)]


def names(rs):
    return [r["name"] for r in rs]


def wp(name, after, pos, **kw):
    entry = {"name": name, "insert_after": after, "pos": list(pos), "why": "measured"}
    if after and not after.startswith("WP"):
        entry["anchor_was"] = kw.pop("anchor_was", None)
    entry.update(kw)
    return {k: v for k, v in entry.items() if v is not None}


def refusals(notes):
    return [n for n in notes if "REFUSED" in n]


# ---------------------------------------------------------------- the normal case

def test_a_level_with_no_insert_list_is_untouched():
    """The thirteen route files with no waypoints must regenerate byte for byte, which starts here:
    with no `insert_after` entry the rows object itself is handed straight back."""
    before = rows("a", "b", "c")
    after, notes, inserted = apply_inserts("0-1", before, {})
    assert after is before and notes == [] and inserted == []
    after, notes, inserted = apply_inserts("0-1", before, {"0-1": [
        {"name": "a", "pos": [1.0, 2.0, 3.0], "was": [1.0, 2.0, 4.0]},
        {"name": "b", "drop": True}]})
    assert after is before, "a MOVE or a DROP entry must not be read as an insert"
    assert notes == [] and inserted == []


def test_the_waypoint_lands_immediately_behind_its_anchor():
    """Rule 7's half that `hops` depends on: `build_level` assigns hops from the row order, so an
    insert that landed anywhere else would silently re-aim the ladder."""
    after, notes, inserted = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert not refusals(notes), notes
    assert names(after) == ["a", "b", "WP1 - x", "c"]
    assert after[2]["waypoint"] is True and after[2]["pos"] == [0.0, 40.0, 100.0]
    assert inserted == [{"name": "WP1 - x", "after": "b", "pos": [0.0, 40.0, 100.0], "source": ""}]
    assert not any("waypoint" in r for r in after if r["name"] != "WP1 - x")


def test_the_rows_that_were_there_are_not_mutated():
    """`apply_inserts` copies before it splices: the caller's list is the fallback every refusal
    returns, so mutating it in place would make all-or-nothing a lie."""
    before = rows("a", "b", "c")
    snapshot = [dict(r) for r in before]
    after, notes, _ = apply_inserts("0-3", before, {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert not refusals(notes), notes
    assert before == snapshot and after is not before


def test_hops_stay_gapless_when_build_level_assigns_them_from_the_row_order():
    """The property `test_route_files.test_hops_are_a_strict_total_order` checks on the file, proved
    here on the mechanism: hops is n-1-k over the row order AFTER the splice, so it cannot gap."""
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c", "d"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
        wp("WP2 - y", "WP1 - x", [0.0, 80.0, 100.0])]})
    assert not refusals(notes), notes
    n = len(after)
    assert [n - 1 - k for k in range(n)] == list(range(n - 1, -1, -1))
    assert names(after) == ["a", "b", "WP1 - x", "WP2 - y", "c", "d"]


def test_several_waypoints_chain_through_each_other_in_list_order():
    """Rule 8. Three waypoints on one climb are ordered by each naming the last, NOT by a tie-break
    on a shared anchor -- which would splice them in REVERSE and send the agent down the climb."""
    after, notes, inserted = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - low", "b", [0.0, 30.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
        wp("WP2 - mid", "WP1 - low", [0.0, 60.0, 100.0]),
        wp("WP3 - high", "WP2 - mid", [0.0, 90.0, 100.0])]})
    assert not refusals(notes), notes
    assert names(after) == ["a", "b", "WP1 - low", "WP2 - mid", "WP3 - high", "c"]
    assert [i["after"] for i in inserted] == ["b", "WP1 - low", "WP2 - mid"]


def test_an_insert_anchored_on_an_insert_carries_no_anchor_was():
    """Rule 10: `anchor_was` means "the position the generator produced for this anchor". An
    inserted rung's position is authored and cannot drift, so recording one is a stale-data trap."""
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - low", "b", [0.0, 30.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
        {"name": "WP2 - mid", "insert_after": "WP1 - low", "pos": [0.0, 60.0, 100.0],
         "anchor_was": [0.0, 30.0, 100.0]}]})
    assert names(after) == ["a", "b", "c"], "all-or-nothing"
    assert any("cannot drift" in n for n in refusals(notes)), notes


# ---------------------------------------------------------------- one test per refusal rule

def test_rule_9_a_name_outside_the_reserved_namespace_is_refused():
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("11 - Somewhere", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "c"]
    assert any("reserved" in n for n in refusals(notes)), notes
    for good in ("WP1 - x", "WP12 - a long name"):
        _after, notes, _ins = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
            wp(good, "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
        assert not refusals(notes), (good, notes)


def test_rule_9_a_generated_room_inside_the_namespace_refuses_the_whole_list():
    """The other half: if the survey ever produces a `WP<n> - ` room the namespace has collided and
    every duplicate and anchor rule below it is unsound, so nothing is applied."""
    after, notes, _ = apply_inserts("0-3", rows("a", "WP1 - a real room", "c"), {"0-3": [
        wp("WP2 - x", "a", [0.0, 40.0, 50.0], anchor_was=[0.0, 0.0, 0.0])]})
    assert names(after) == ["a", "WP1 - a real room", "c"]
    assert any("namespace" in n for n in refusals(notes)), notes


def test_rule_10_an_anchor_that_has_drifted_is_refused():
    """The waypoint was measured in geometry its anchor sat in: if the anchor moved, the rooms did,
    and the hand-picked point is no longer known to be on the climb."""
    ok, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.9, 100.0])]})
    assert not refusals(notes), "0.9 m is inside OVERRIDE_WAS_TOL_M"
    assert names(ok) == ["a", "b", "WP1 - x", "c"]
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 109.0])]})
    assert names(after) == ["a", "b", "c"]
    assert any("9.0 m from the recorded `anchor_was`" in n for n in refusals(notes)), notes


def test_rule_10_the_anchor_is_matched_against_its_MOVED_position():
    """Rule 7 in one assertion. `apply_inserts` runs after `apply_overrides`, so an anchor that is
    itself a move target is matched against where it SHIPS. Recording the generator's own position
    for such an anchor is the mistake this pins: 0-3's Side Hallway generates at (0,10,331) and
    ships at (0,10,340), nine times the tolerance apart."""
    rs = rows("a", "2 - Side Hallway - Floor 1", "c")
    rs[1]["pos"] = [0.0, 10.0, 331.0]
    moved, notes = apply_overrides("0-3", rs, {"0-3": [
        {"name": "2 - Side Hallway - Floor 1", "was": [0.0, 10.0, 331.0],
         "pos": [0.0, 10.0, 340.0], "why": "measured"}]}, geom=None)
    assert not refusals(notes), notes
    after, notes, _ = apply_inserts("0-3", moved, {"0-3": [
        wp("WP1 - x", "2 - Side Hallway - Floor 1", [-10.7, 21.6, 327.2],
           anchor_was=[0.0, 10.0, 340.0])]})
    assert not refusals(notes), notes
    assert names(after) == ["a", "2 - Side Hallway - Floor 1", "WP1 - x", "c"]
    after, notes, _ = apply_inserts("0-3", moved, {"0-3": [
        wp("WP1 - x", "2 - Side Hallway - Floor 1", [-10.7, 21.6, 327.2],
           anchor_was=[0.0, 10.0, 331.0])]})
    assert names(after) == ["a", "2 - Side Hallway - Floor 1", "c"], \
        "the PRE-move position must not match: inserts run after the moves"
    assert refusals(notes)


def test_rule_11_a_waypoint_inside_I2s_separation_is_refused():
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 0.0, 115.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "c"]
    assert any("I2's 16 m separation" in n for n in refusals(notes)), notes


def test_rule_11_two_cylinders_that_share_a_point_are_refused_even_past_I2():
    """The rule this incident buys, and the one I2 cannot express. The reviewed proposal put two
    waypoints 13.5 m apart horizontally and 10.8 m vertically -- 17.3 m in a straight line, so I2's
    16 m passed them -- while (-4.0, 27.0, 328.3) lies inside BOTH 8 m x 6 m cylinders, and one
    arrival there pays two ladder instalments for one place."""
    a, b = [-10.7, 21.6, 327.1], [2.6, 32.4, 329.5]
    assert math.dist(a, b) > SEP, "the pair really does clear I2"
    assert co_credit(a, b)
    rs = rows("a", "b")
    rs[1]["pos"] = list(a)
    after, notes, _ = apply_inserts("0-3", rs, {"0-3": [
        wp("WP1 - x", "b", b, anchor_was=list(a))]})
    assert names(after) == ["a", "b"]
    assert any("reach cylinder overlaps" in n for n in refusals(notes)), notes


def test_co_credit_is_the_cylinder_test_and_not_a_sphere():
    """Two cylinders meet iff horizontal <= 2*REACH_H AND vertical <= 2*REACH_V. Both bounds, and
    neither halved: a pair 10 m apart in y DOES share points, because each cylinder reaches 6 m."""
    o = [0.0, 0.0, 0.0]
    assert co_credit(o, [15.9, 0.0, 0.0]) and not co_credit(o, [16.1, 0.0, 0.0])
    assert co_credit(o, [0.0, 11.9, 0.0]) and not co_credit(o, [0.0, 12.1, 0.0])
    assert co_credit(o, [10.0, 10.0, 0.0]), "inside both bounds: the cylinders share a point"
    assert not co_credit(o, [17.0, 10.0, 0.0]), "clear horizontally: no shared point"
    assert not co_credit(o, [10.0, 13.0, 0.0]), "clear vertically: no shared point"
    # ... and a point inside both really does exist wherever it says so.
    a, b = [0.0, 0.0, 0.0], [10.0, 10.0, 0.0]
    p = [5.0, 5.0, 0.0]
    for r in (a, b):
        assert math.hypot(p[0] - r[0], p[2] - r[2]) <= REACH_H and abs(p[1] - r[1]) <= REACH_V


def test_rule_12_a_waypoint_R4_cannot_stand_on_is_refused():
    """`geom` here is the stub `standability` the lifter installs: it reports reach=False for any
    rung whose name is in the object it is handed."""
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]},
        geom={"WP1 - x"})
    assert names(after) == ["a", "b", "c"]
    assert any("no standable cell" in n for n in refusals(notes)), notes
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]},
        geom=set())
    assert names(after) == ["a", "b", "WP1 - x", "c"]


def test_rule_12_a_no_probe_run_ships_the_waypoint_with_a_loud_note():
    """`geom=None` is `--no-probe`, the same concession `apply_overrides` already makes -- but it
    must SAY it shipped something unprobed."""
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]}, geom=None)
    assert names(after) == ["a", "b", "WP1 - x", "c"]
    assert not refusals(notes)
    assert any("UNPROBED" in n for n in notes), notes


def test_rule_13_a_waypoint_that_would_steal_the_seed_is_refused():
    """`GateProgress._seed_start` absorbs the rung nearest the spawn and forgoes every instalment
    above it. A waypoint that lands nearer the spawn than the trunk's own first rung therefore
    deletes the ladder above itself in silence -- which is the whole mechanism, not a warning."""
    # `a` at z 0, `b` at z 100, `c` at z 200; the spawn 48 m from `a`, the nearest of them. The
    # candidates sit on a ray running +x out of the spawn, so their distance from every generated
    # rung is at least 48 m and rules 11 and 12 cannot fire -- only rule 13 can.
    start = [0.0, 0.0, -48.0]

    def at(x):
        return apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
            wp("WP1 - x", "b", [x, 0.0, -48.0], anchor_was=[0.0, 0.0, 100.0], start_pos=start)]})

    after, notes, _ = at(28.0)
    assert names(after) == ["a", "b", "c"]
    assert any("_seed_start" in n for n in refusals(notes)), notes
    # The bar is 48 + max(4.0, 10% of 48 = 4.8) = 52.8 m, so 52 m fails and 53 m passes. 52 m is
    # what pins the FRACTION: with the 4 m floor alone the bar would be 52.0 and 52 m would pass.
    for x, want_ok in ((52.0, False), (53.0, True)):
        _after, notes, _ins = at(x)
        assert (not refusals(notes)) is want_ok, (x, notes)


def test_rule_13_without_a_start_pos_the_check_is_skipped_with_a_note():
    """The offline parse has no real per-level spawn, so the entry has to supply one -- and its
    absence has to be visible rather than silently safe."""
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "WP1 - x", "c"]
    assert any("seed-margin check" in n for n in notes), notes


def test_rule_13_measures_against_generated_rungs_only():
    """An earlier waypoint is not a baseline. `_seed_start` absorbs whichever rung is nearest the
    spawn, waypoints included, so measuring each new one against the last would let a pair walk the
    seed down the ladder together -- and an entry that carries no `start_pos` skips the check
    entirely, which is exactly how one gets near the spawn in the first place.

    Here `WP1 - near` has no `start_pos` and lands 10 m from the spawn. `WP2 - x` at 30 m is under
    the 52.8 m bar the four GENERATED rungs set, and must be refused; chained through `WP1 - near`
    the bar would be 14 m and it would ship.
    """
    start = [0.0, 0.0, -48.0]
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - near", "b", [10.0, 0.0, -48.0], anchor_was=[0.0, 0.0, 100.0]),
        wp("WP2 - x", "WP1 - near", [30.0, 0.0, -48.0], start_pos=start)]})
    assert names(after) == ["a", "b", "c"], "all-or-nothing takes WP1 down with it"
    assert any("_seed_start" in n and "WP2 - x" in n for n in refusals(notes)), notes


def test_rule_14_a_duplicate_name_is_refused():
    after, notes, _ = apply_inserts("0-3", rows("a", "WP1 - x", "c"), {"0-3": [
        wp("WP1 - x", "a", [0.0, 40.0, 0.0], anchor_was=[0.0, 0.0, 0.0])]})
    assert names(after) == ["a", "WP1 - x", "c"]
    assert refusals(notes), notes
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
        wp("WP1 - x", "b", [0.0, 80.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "c"]
    assert any("already a rung" in n for n in refusals(notes)), notes


def test_rule_15_an_unknown_or_dropped_anchor_is_refused():
    after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - x", "ghost", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "c"]
    assert any("not a rung here" in n for n in refusals(notes)), notes


def test_rule_16_going_over_the_budget_cap_refuses_the_whole_list():
    """I3's cheapest-detour trimming has already run by the time inserts land, so without this the
    waypoints would push a trunk room off the end of the route instead of failing."""
    rs = rows(*["r%02d" % i for i in range(24)])
    after, notes, _ = apply_inserts("0-3", rs, {"0-3": [
        wp("WP1 - x", "r05", [0.0, 40.0, 500.0], anchor_was=[0.0, 0.0, 500.0])]})
    assert len(after) == 24 and not any(r.get("waypoint") for r in after)
    assert any("over I3's cap" in n for n in refusals(notes)), notes


def test_rule_17_one_refusal_withholds_every_waypoint_for_that_level():
    """A half-applied chain is worse than none: it leaves exactly the near-vertical leg the chain
    exists to remove, one rung further up."""
    after, notes, inserted = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [
        wp("WP1 - good", "b", [0.0, 30.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
        wp("WP2 - bad", "ghost", [0.0, 60.0, 100.0], anchor_was=[0.0, 0.0, 100.0])]})
    assert names(after) == ["a", "b", "c"], "the GOOD one must not ship either"
    assert inserted == []
    assert any("all 2 waypoints withheld" in n for n in notes), notes
    assert not any("insert applied" in n for n in notes), notes


def test_a_shape_error_is_a_refusal_and_not_an_exception():
    for entry in ({"name": "", "insert_after": "b", "pos": [0.0, 1.0, 2.0]},
                  {"name": "WP1 - x", "insert_after": "b", "pos": [0.0, 1.0]},
                  {"name": "WP1 - x", "insert_after": "b", "pos": "nowhere"},
                  {"name": "WP1 - x", "insert_after": "b", "pos": [0.0, float("nan"), 2.0]}):
        after, notes, _ = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [entry]})
        assert names(after) == ["a", "b", "c"], entry
        assert refusals(notes), entry
    assert _vec3([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert _vec3([1, 2]) is None and _vec3(None) is None and _vec3([1, 2, float("inf")]) is None


def test_an_empty_insert_after_is_not_an_insert_and_still_cannot_be_lost():
    """The kind of an entry is decided by the PRESENCE of a truthy `insert_after`, exactly as
    `drop: true` decides a drop. So `"insert_after": ""` is not an insert at all -- and the thing
    that matters is that it cannot then fall between the two passes and be silently ignored: it
    lands in the move path, which refuses it for having no `was`."""
    entry = {"name": "WP1 - x", "insert_after": "", "pos": [0.0, 1.0, 2.0]}
    after, notes, inserted = apply_inserts("0-3", rows("a", "b", "c"), {"0-3": [entry]})
    assert names(after) == ["a", "b", "c"] and notes == [] and inserted == []
    _after, notes = apply_overrides("0-3", rows("a", "b", "c"), {"0-3": [entry]}, geom=None)
    assert refusals(notes), notes


def test_apply_overrides_steps_over_an_insert_entry():
    """The twin of `test_route_drops.test_apply_overrides_steps_over_a_drop_entry`. Without the skip
    an insert entry lands in the MOVE path, fails its `pos`/`was` shape check and REFUSES -- turning
    `--validate` red on every run for a change that worked."""
    rs = rows("a", "b", "c")
    entries = [wp("WP1 - x", "b", [0.0, 40.0, 100.0], anchor_was=[0.0, 0.0, 100.0]),
               {"name": "b", "pos": [0.0, 0.0, 120.0], "was": [0.0, 0.0, 100.0], "why": "measured"}]
    after, notes = apply_overrides("0-3", rs, {"0-3": entries}, geom=None)
    assert not refusals(notes), notes
    assert dict((r["name"], r["pos"]) for r in after)["b"] == [0.0, 0.0, 120.0]
    assert "WP1 - x" not in names(after), "apply_overrides must not add a rung"


# ---------------------------------------------------------------- the 0-3 ladder, on the real trace

# `GateProgress._is_reached` for a route rung: `_parse_route` forces open=false, so the cylinder is
# never doubled -- 8.0 m horizontal, 6.0 m vertical, flat.
SHIPPED_0_3 = [
    ("1 - Main Room - Floor 1", [0.0, -10.0, 300.0]),
    ("2 - Side Hallway - Floor 1", [0.0, 10.0, 340.0]),
    ("WP1 - Main Room Stack Foot", [-10.7, 21.6, 327.2]),
    ("WP2 - Main Room Stack Top", [5.3, 36.7, 328.9]),
    ("10 - Main Room - Floor 2", [-7.1, 48.5, 315.2]),
    ("10B - Second Encounter + 11 - Boss Arena - Floor 2", [-82.0, 90.0, 315.0]),
]

# Samples copied out of runs/campaign_gates/best_runs/Level_0-3.json -- the AI's OWN 263.904 s
# completion of Level 0-3 (4064 positions at 15.4 Hz, rank B, difficulty 3, saved 2026-09-18
# 01:17:26), NOT a human demo. `runs/` is gitignored, so the numbers live here as literals: the
# index of each sample is kept so the file can be re-derived, and they are in trace order.
TRACE_0_3 = [
    (0, [0.0, 0.5, 253.0]),            # the level start
    (34, [2.79, -6.26, 296.25]),       # first credit: 1 - Main Room - Floor 1
    (130, [-14.24, 9.05, 309.78]),     # 8.4 s in, on the low pipe staircase -- credits NOTHING
    (651, [-0.07, 10.92, 332.8]),      # first credit: 2 - Side Hallway - Floor 1
    (688, [-9.51, 15.62, 325.06]),     # first credit: WP1
    (1004, [-5.41, 13.71, 303.0]),     # the spiral ramp, climbing
    (1010, [-13.38, 17.12, 314.46]),
    (1014, [-10.72, 21.56, 327.15]),   # WP1 was measured here
    (1019, [-0.29, 29.2, 329.35]),
    (1021, [2.59, 32.38, 329.46]),     # first credit: WP2
    (1024, [5.3, 36.67, 328.85]),      # WP2 was measured here
    (1030, [-1.81, 41.39, 325.33]),
    (1034, [-7.33, 44.17, 323.17]),    # first credit: 10 - Main Room - Floor 2
    (1040, [-14.35, 47.17, 317.11]),
    (1100, [-8.32, 48.95, 319.09]),    # the Floor 2 dwell
    (1241, [-74.54, 91.54, 316.84]),   # first credit: the boss arena
    (4063, [-177.81, 81.32, 313.63]),  # the exit
]


def credits(pos, rung_pos):
    return (math.hypot(pos[0] - rung_pos[0], pos[2] - rung_pos[2]) <= REACH_H
            and abs(pos[1] - rung_pos[1]) <= REACH_V)


def test_the_shipped_0_3_ladder_is_what_this_file_walks():
    """The literals above are only evidence while they are the positions that ship."""
    doc = json.loads((ROUTES / "route_Level_0-3.json").read_text(encoding="utf-8"))
    assert [(r["name"], r["pos"]) for r in doc["rungs"]] == SHIPPED_0_3, doc["rungs"]
    assert [r["hops"] for r in doc["rungs"]] == [5, 4, 3, 2, 1, 0]


def test_the_ai_s_own_completion_credits_the_six_rungs_in_order():
    """The ladder's one hard requirement: along the only recorded 0-3 completion every rung is first
    credited strictly after the one above it. A rung credited out of order lowers `best_hops` early,
    pays `paid_hops - best_hops` instalments for ground the agent has not covered, and takes the
    skipped rung out of `_choose_target` for the rest of the level load.

    The reviewed proposal failed exactly here: its first waypoint sat where the trace passes at
    index 130, 8.4 s in and 521 samples BEFORE the Side Hallway it was anchored behind.
    """
    first = {}
    for idx, pos in TRACE_0_3:
        for name, rung in SHIPPED_0_3:
            if name not in first and credits(pos, rung):
                first[name] = idx
    missing = [n for n, _ in SHIPPED_0_3 if n not in first]
    assert not missing, "never credited along the trace: %r" % missing
    order = [first[n] for n, _ in SHIPPED_0_3]
    assert order == sorted(order) and len(set(order)) == len(order), \
        dict(zip([n for n, _ in SHIPPED_0_3], order))
    assert order == [34, 651, 688, 1021, 1034, 1241], order


def test_the_early_pipe_staircase_sample_credits_nothing():
    """Trace index 130 is 8.4 s into the level, on the low staircase the agent walks over on its way
    into the main room. A waypoint reachable from there is collected before the ladder has asked for
    anything, which is what inverted the reviewed proposal's hop order."""
    pos = dict(TRACE_0_3)[130]
    assert [n for n, r in SHIPPED_0_3 if credits(pos, r)] == []


def test_the_live_stall_points_credit_nothing():
    """Where the policy actually ends its episodes today: the four densest 10 m end-position cells
    of the 264 fresh `spec_0-3` episodes since timesteps 21,701,566, plus the exact point the task
    named. None of them may pay -- a waypoint credited from the wedge is a farm, not a route."""
    for stall in ([0.0, 20.0, 340.0], [10.0, 20.0, 340.0], [10.0, 20.0, 330.0],
                  [10.0, 20.0, 350.0], [-10.0, 20.0, 340.0], [0.0, 10.0, 330.0]):
        got = [n for n, r in SHIPPED_0_3 if credits(stall, r)]
        assert got == [], "%s credits %r" % (stall, got)


def test_no_waypoint_is_creditable_from_the_floor_below_or_above_it():
    """A rung 6 m of cylinder tall is reachable from a full jump's 15 m apex if it is stacked over
    somewhere the agent already stands, so each waypoint is checked from directly under it on the
    main-room floor (y -10) and from directly over it on Floor 2 (y 48.5)."""
    for name, rung in SHIPPED_0_3:
        if not name.startswith("WP"):
            continue
        for y in (-10.0, 48.5):
            probe = [rung[0], y, rung[2]]
            assert not credits(probe, rung), "%s credits from y %.1f directly %s it" % (
                name, y, "below" if y < rung[1] else "above")


def test_no_two_shipped_0_3_rungs_share_a_reach_cylinder():
    """I2 and rule 11 on the file that ships, not on a fixture."""
    pts = [p for _, p in SHIPPED_0_3]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert math.dist(pts[i], pts[j]) >= SEP, (SHIPPED_0_3[i][0], SHIPPED_0_3[j][0])
            assert not co_credit(pts[i], pts[j]), (SHIPPED_0_3[i][0], SHIPPED_0_3[j][0])


def test_every_leg_of_the_0_3_ladder_carries_a_usable_heading():
    """The mechanism the change exists for. The leg it replaced -- the Side Hallway (0,10,340) to
    the Floor 2 centroid (0,50,330) -- was 10.0 m of horizontal against 40.0 m of vertical, 76.0
    degrees, so the unit target vector the policy reads was almost entirely 'up'."""
    old = math.degrees(math.atan2(40.0, 10.0))
    assert round(old, 1) == 76.0
    pts = [p for _, p in SHIPPED_0_3]
    angles = [math.degrees(math.atan2(abs(pts[i + 1][1] - pts[i][1]),
                                      math.hypot(pts[i + 1][0] - pts[i][0],
                                                 pts[i + 1][2] - pts[i][2])))
              for i in range(len(pts) - 1)]
    assert max(angles) < 45.0, [round(a, 1) for a in angles]
    assert round(max(angles), 1) == 43.2, [round(a, 1) for a in angles]


def test_the_ladder_is_still_five_hops_deep():
    """The arithmetic that condemned the 0-3 side wing, held on the new ladder. `_seed_start`
    absorbs the rung nearest the spawn -- rung 1, 48.2 m away against the nearest waypoint's 77.9 m
    -- so a level load can earn `gate` 15.0 x 5 = 75.0 by walking the whole ladder and never
    finishing, against `level_complete` 100.0. Six rungs, five hops: unchanged from the four-rung
    file, because the seed and the exit rung are both unpaid."""
    doc = json.loads((ROUTES / "route_Level_0-3.json").read_text(encoding="utf-8"))
    rungs = doc["rungs"]
    start = [0.0, 0.5, 253.0]
    by_dist = sorted(rungs, key=lambda r: math.dist(start, r["pos"]))
    assert by_dist[0]["name"] == "1 - Main Room - Floor 1"
    assert by_dist[0]["hops"] == len(rungs) - 1 == 5
    near, second = (math.dist(start, r["pos"]) for r in by_dist[:2])
    assert round(near, 1) == 48.2 and second - near >= max(4.0, 0.10 * near), (near, second)
    assert by_dist[0]["hops"] * 15.0 == 75.0


def test_the_shipped_file_records_its_waypoints():
    """`trunk_inserted` beside `trunk_dropped`: a route carrying rungs the room survey never
    produced has to say so in the file, and every such rung carries `waypoint: true`."""
    doc = json.loads((ROUTES / "route_Level_0-3.json").read_text(encoding="utf-8"))
    flagged = [r["name"] for r in doc["rungs"] if r.get("waypoint")]
    assert flagged == ["WP1 - Main Room Stack Foot", "WP2 - Main Room Stack Top"], flagged
    recorded = doc.get("trunk_inserted") or []
    assert [i["name"] for i in recorded] == flagged, recorded
    assert [i["after"] for i in recorded] == ["2 - Side Hallway - Floor 1",
                                              "WP1 - Main Room Stack Foot"], recorded
    for item in recorded:
        assert item["source"].startswith("runs/campaign_gates/best_runs/"), item
    for rung in doc["rungs"]:
        if not rung.get("waypoint"):
            assert "waypoint" not in rung, rung["name"]


def test_only_0_3_ships_waypoints():
    """`apply_inserts` is a no-op for a level with no entries, so the other thirteen files must
    carry neither key. One appearing means a waypoint list was added without a row in
    `tests/test_route_files.py`'s ladder table."""
    for path in sorted(ROUTES.glob("route_Level_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "route_Level_0-3.json":
            continue
        assert "trunk_inserted" not in doc, "%s records inserts" % path.name
        assert not any(r.get("waypoint") for r in doc["rungs"]), path.name


def test_every_insert_entry_is_present_in_the_file_that_ships():
    """The detector for a regeneration that silently stopped applying the list: every entry in
    `rung_overrides.json` ships, at its own position, flagged, and behind the anchor it names."""
    entries = json.loads((ROUTES / "rung_overrides.json").read_text(encoding="utf-8"))
    doc = json.loads((ROUTES / "route_Level_0-3.json").read_text(encoding="utf-8"))
    order = [r["name"] for r in doc["rungs"]]
    by_name = {r["name"]: r for r in doc["rungs"]}
    inserts = [e for e in entries["0-3"] if e.get("insert_after")]
    assert len(inserts) == 2, inserts
    for entry in inserts:
        name = entry["name"]
        assert name in by_name, "%r is not a rung in the shipped file" % name
        assert by_name[name]["pos"] == [round(float(v), 1) for v in entry["pos"]], name
        assert by_name[name].get("waypoint") is True, name
        assert order.index(name) == order.index(entry["insert_after"]) + 1, name
        assert "was" not in entry and "drop" not in entry, \
            "%r mixes an insert with another entry kind" % name
        assert entry.get("why"), "%r records no reason" % name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
    sys.exit(0)
