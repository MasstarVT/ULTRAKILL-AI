"""`build_routes.apply_drops`, the manual rung-drop list. No game, no scene bundles:

    python tests/test_route_drops.py     (or pytest)

The two functions under test are lifted out of `scripts/build_routes.py` by AST rather than
imported. That module calls `refuse_if_commit_high("build_routes.py")` at IMPORT time -- correctly,
because a real run parses every shipped scene bundle and is the heaviest offline script in the repo
-- so importing it here would make this file fail whenever the training box is busy, which is
exactly when the no-game suite gets run. Lifting the function compiles the same source the
generator executes, with none of that weight.

What is pinned is the three rules of the OVERRIDES_NAME comment that a drop entry adds:

  4. drops are applied between I2 and R1, so `hops` (assigned from the surviving row order) stays
     gapless and R1 counts what actually ships -- tested here as "order is preserved, nothing else
     moves";
  5. a name that is no longer a rung is REFUSED, not ignored;
  6. a list that would take a level under MIN_RUNGS is refused WHOLE, and the generated trunk ships.

Rule 6 is the one with teeth: `GateProgress._parse_route` reads a file with fewer than
ROUTE_MIN_RUNGS rungs as NO file, so a drop list that overshot would silently move the level from
its trunk to its gate ladder -- a far larger change than a drop list is allowed to make by accident.
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
        "MIN_RUNGS": 3,
        "SEP": 16.0,
        "OVERRIDE_WAS_TOL_M": 1.0,
        "dist3": lambda a, b: math.dist(a, b),
        "standability": None,  # only reached with geom is not None, which no test here passes
    }
    module = ast.Module(body=[want[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)  # noqa: S102
    return [ns[n] for n in names]


apply_drops, apply_overrides = _lift("apply_drops", "apply_overrides")


def rows(*names):
    return [{"name": n, "pos": [0.0, 0.0, 100.0 * i]} for i, n in enumerate(names)]


def names(rs):
    return [r["name"] for r in rs]


# ---------------------------------------------------------------- the normal case

def test_a_level_with_no_drop_list_is_untouched():
    before = rows("a", "b", "c", "d")
    after, notes, dropped = apply_drops("0-1", before, {})
    assert after is before and notes == [] and dropped == []
    after, notes, dropped = apply_drops("0-1", before, {"0-1": [{"name": "a", "pos": [1, 2, 3],
                                                                "was": [0, 0, 0]}]})
    assert after is before, "a MOVE entry must not be read as a drop"
    assert notes == [] and dropped == []


def test_the_named_rungs_go_and_the_order_of_the_rest_is_kept():
    """Rule 4: `hops` is assigned from the surviving row order in `build_level`, so the one thing a
    drop must never do is reorder what is left."""
    after, notes, dropped = apply_drops("0-3", rows("a", "b", "c", "d", "e"),
                                        {"0-3": [{"name": "d", "drop": True, "why": "x"},
                                                 {"name": "b", "drop": True, "why": "y"}]})
    assert names(after) == ["a", "c", "e"]
    assert dropped == ["d", "b"], "the report lists them in the order the file asks for"
    assert not [n for n in notes if "REFUSED" in n]
    assert sum(1 for n in notes if n.startswith("drop applied")) == 2
    assert "(x)" in notes[0] and notes[0].startswith("drop applied: 0-3 'd'"), notes


def test_the_reason_is_carried_into_the_note():
    """`--validate` prints these, and a drop with no recorded reason is how a route silently loses a
    room. A missing `why` is allowed but says so out loud."""
    _after, notes, _dropped = apply_drops("0-3", rows("a", "b", "c", "d"),
                                          {"0-3": [{"name": "b", "drop": True}]})
    assert notes == ["drop applied: 0-3 'b' (no reason recorded)"], notes


# ---------------------------------------------------------------- rule 5, an unknown name

def test_a_name_that_is_not_a_rung_is_refused_not_ignored():
    after, notes, dropped = apply_drops("0-3", rows("a", "b", "c", "d"),
                                        {"0-3": [{"name": "ghost", "drop": True},
                                                 {"name": "b", "drop": True}]})
    assert names(after) == ["a", "c", "d"], "the entries that DO match still apply"
    assert dropped == ["b"]
    refused = [n for n in notes if "REFUSED" in n]
    assert len(refused) == 1 and "ghost" in refused[0], notes


def test_a_list_of_nothing_but_unknown_names_changes_nothing():
    before = rows("a", "b", "c")
    after, notes, dropped = apply_drops("0-3", before, {"0-3": [{"name": "ghost", "drop": True}]})
    assert after is before and dropped == []
    assert len(notes) == 1 and "REFUSED" in notes[0]


# ---------------------------------------------------------------- rule 6, the MIN_RUNGS floor

def test_a_drop_list_that_would_go_under_min_rungs_is_refused_whole():
    """Not partially applied, and not applied down to the floor: `_parse_route` would read the
    result as no file at all and the level would fall to its gate ladder in silence."""
    before = rows("a", "b", "c", "d")
    after, notes, dropped = apply_drops("0-3", before,
                                        {"0-3": [{"name": "b", "drop": True},
                                                 {"name": "c", "drop": True}]})
    assert after is before, "the generated trunk ships untouched"
    assert dropped == []
    refused = [n for n in notes if "REFUSED" in n]
    assert len(refused) == 1 and "under R1's 3" in refused[0], notes


def test_exactly_min_rungs_is_allowed():
    after, notes, dropped = apply_drops("0-3", rows("a", "b", "c", "d"),
                                        {"0-3": [{"name": "b", "drop": True}]})
    assert names(after) == ["a", "c", "d"] and dropped == ["b"]
    assert not [n for n in notes if "REFUSED" in n]


# ---------------------------------------------------------------- the move path is not confused

def test_apply_overrides_steps_over_a_drop_entry():
    """Both kinds live in one per-level list, and `apply_overrides` runs LAST, after the drop has
    already happened. Without the skip a drop entry would land in the move path, fail its
    `pos`/`was` shape check and REFUSE -- turning `--validate` red for a change that worked."""
    rs = rows("a", "b", "c", "d")
    entries = [{"name": "ghost", "drop": True}, {"name": "b", "pos": [0.0, 0.0, 120.0],
                                                 "was": [0.0, 0.0, 100.0], "why": "measured"}]
    after, notes = apply_overrides("0-3", rs, {"0-3": entries}, geom=None)
    assert not [n for n in notes if "REFUSED" in n], notes
    assert dict((r["name"], r["pos"]) for r in after)["b"] == [0.0, 0.0, 120.0]


# ---------------------------------------------------------------- what actually ships

def test_the_shipped_0_3_file_is_what_its_drop_list_asks_for():
    """End to end on the committed data, without regenerating it: the seven names the override file
    drops are absent from the route, what remains is in trunk order with gapless hops, and the file
    records the drop. (`tests/test_route_files.py` pins the positions and the diagnostics.)

    Six rungs, not the four the drop list leaves: `apply_inserts` splices two waypoints onto the
    main-room climb (2026-09-18, `tests/test_route_inserts.py`). Counting the ROOMS is what this
    test is for, so the waypoints are excluded from that count rather than folded into it -- a drop
    that stopped being applied must still show up here even while the insert list grows.
    """
    doc = json.loads((ROUTES / "route_Level_0-3.json").read_text(encoding="utf-8"))
    entries = json.loads((ROUTES / "rung_overrides.json").read_text(encoding="utf-8"))["0-3"]
    dropped = [e["name"] for e in entries if e.get("drop")]
    assert len(dropped) == 7, dropped
    shipped = [r["name"] for r in doc["rungs"]]
    assert not set(dropped) & set(shipped), sorted(set(dropped) & set(shipped))
    rooms = [r["name"] for r in doc["rungs"] if not r.get("waypoint")]
    assert len(rooms) == 4, rooms
    assert len(shipped) == 6, shipped
    assert [r["hops"] for r in doc["rungs"]] == [5, 4, 3, 2, 1, 0]
    assert sorted(doc["trunk_dropped"]) == sorted(dropped)


def test_the_other_route_files_record_no_drops():
    """`trunk_dropped` is emitted only when a drop list was applied, so the thirteen files with none
    stay byte for byte what they were. A key appearing on one of them means a drop list was added
    without a row in `tests/test_route_files.py`'s ladder table."""
    for path in sorted(ROUTES.glob("route_Level_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if path.name != "route_Level_0-3.json":
            assert "trunk_dropped" not in doc, "%s records drops" % path.name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
    sys.exit(0)
