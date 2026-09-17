"""A0 of docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md §8, offline:
    python tests/test_route_replay.py    (or pytest). No game needed.

A0 is the check that runs before anything else, and it exists because of the measurement that changed the
design (§2.6). `GateProgress` carries ONE `best_hops`: `_note_reached` scans every rung each step and collapses
it to the minimum over all of them, so on a branching ladder the branch the agent enters SECOND permanently
loses its signal. Re-run with the real tracker on revision 1's files, entering one of 1-4's five side rooms
first paid four instalments in a single step and then pointed the observation at the boss door while three
required rooms were still unvisited. Invariant T is the fix -- ship only the trunk, the rooms every
playthrough must pass -- and this file is T's regression test:

  1. **the trunk is a chain**: every rung is its own hop tier, and no two rungs share an ordinal with different
     suffixes, so there is exactly ONE order the level allows;
  2. **replayed in every order that order permits** (and from every rung the player could start at), the target
     is never a rung already visited and no step pays more than one instalment;
  3. **the assertions can fail**: the same replay over a synthetic STAR reproduces §2.6's four-at-once, so a
     regeneration that lost T would be caught here rather than in a run.

And the other half of A0's job, the one that protects the live run: `GateProgress` with a route document
present reproduces `tests/test_ladder_replay.py`'s golden file -- 32,022 recorded Level 0-1 decisions and the
two Level 0-3 probes, produced by the implementation from BEFORE any of this existed. That is a pin against
recorded data rather than against this code talking to itself.

The fixtures are hand-written until `build_routes.py` ships the real files; `route_Level_0-5.json` is the spec's
own literal example (§4.2), which is measured level data. Every committed `routes/route_*.json` is picked up
automatically as soon as it exists, so the wrap-up stage runs A0 over the real 12 without editing this file.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_ladder_replay as ladder  # noqa: E402  (the recorded fixtures and the golden file)
from ultrakill_ai.campaign import (  # noqa: E402
    ROUTE_DIR,
    ROUTE_SOURCE_ROOMS,
    ROUTE_VERSION,
    GateProgress,
    _read_route,
    load_route,
    route_path,
)

# The whole of `route_Level_0-5.json` as §4.2 prints it: Tier C level 0-5, five rungs, Cerberus's arena at
# hops 1, the exit's own room at hops 0 and the pit 211 m past it.
SPEC_0_5 = {
    "level": "Level 0-5", "version": 2, "source": "room-trunk offline v2",
    "exit": {"pos": [391.5, -121.6, 382.0], "target": "Level 1-1"},
    "start_room": "1 - Opening Hallway", "trunk_collapsed": [],
    "tour_ratio": 1.0, "checkpoints_within_60m": "1/1", "legs_witnessed": "3/5",
    "last_rung_to_exit_m": 211.1,
    "rungs": [
        {"key": "0,-10,300", "pos": [0.0, -10.0, 300.0], "hops": 4, "name": "1 - Opening Hallway",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "0,0,351", "pos": [0.0, 0.0, 351.0], "hops": 3, "name": "2 - Lava Foundry",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "134,-6,382", "pos": [133.5, -6.0, 382.0], "hops": 2, "name": "3 - Smallway",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "174,-6,382", "pos": [174.5, -6.0, 382.0], "hops": 1, "name": "4 - Cerberus Arena",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "214,-6,382", "pos": [214.5, -6.5, 382.0], "hops": 0, "name": "5 - Final Hallway",
         "gated_by": [], "open": False, "locked": False, "active": True},
    ],
}

# A second chain, long enough that the budget cap and the tier logic are exercised over more than five rungs:
# 7-2's shape, 15 rungs at ~80 m spacing, as the §10 table measures it.
LONG_CHAIN = {
    "level": "Level 7-2", "version": ROUTE_VERSION, "source": "room-trunk offline v2 (test fixture)",
    "exit": {"pos": [0.0, 0.0, 1400.0], "target": "Level 7-3"},
    "start_room": "1 - Start", "trunk_collapsed": [],
    "tour_ratio": 1.26, "checkpoints_within_60m": "5/6", "legs_witnessed": "10/15",
    "last_rung_to_exit_m": 168.0,
    "rungs": [{"key": f"0,0,{80 * i}", "pos": [0.0, 0.0, 80.0 * i], "hops": 14 - i,
               "name": f"{i + 1} - Room {i + 1}", "gated_by": [], "open": False, "locked": False, "active": True}
              for i in range(15)],
}

# §2.6's own shape, which invariant T exists to remove: `3 - Main Hall` and then FIVE side rooms sharing
# ordinal 3, which the emitter's alphabetical suffix order turned into hop tiers 4, 3, 2, 1, 0. The names are
# 1-4's real ones.
STAR = {
    "level": "Level 1-4", "version": ROUTE_VERSION, "source": "revision 1, before invariant T",
    "exit": {"pos": [0.0, 0.0, 400.0], "target": "Level 2-1"},
    "start_room": "3 - Main Hall", "trunk_collapsed": [],
    "tour_ratio": 1.0, "checkpoints_within_60m": "2/2", "legs_witnessed": "2/4", "last_rung_to_exit_m": 150.0,
    "rungs": [
        {"key": "0,0,0", "pos": [0.0, 0.0, 0.0], "hops": 5, "name": "3 - Main Hall",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "0,0,40", "pos": [0.0, 0.0, 40.0], "hops": 4, "name": "3C - Cube Room",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "40,0,80", "pos": [40.0, 0.0, 80.0], "hops": 3, "name": "3DL - Down Left",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "-40,0,80", "pos": [-40.0, 0.0, 80.0], "hops": 2, "name": "3DR - Down Right",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "40,0,120", "pos": [40.0, 0.0, 120.0], "hops": 1, "name": "3TL - Top Left",
         "gated_by": [], "open": False, "locked": False, "active": True},
        {"key": "-40,0,120", "pos": [-40.0, 0.0, 120.0], "hops": 0, "name": "3TR - Shelf Room",
         "gated_by": [], "open": False, "locked": False, "active": True},
    ],
}

FIXTURES = (SPEC_0_5, LONG_CHAIN)
NAME = re.compile(r"^\s*(\d+)([A-Za-z]*)\s*-\s*(.+)$")
MAX_ORDERS = 120  # permutations replayed per parallel set; a 5-room star is 120 orders exactly


def write(doc: dict, directory) -> dict | None:
    """Writes `doc` as its level's route file in `directory` and loads it back through the real loader."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    route_path(doc["level"], str(directory)).write_text(json.dumps(doc), encoding="utf-8")
    _read_route.cache_clear()
    return load_route(doc["level"], str(directory))


def block(route: dict) -> dict:
    """The campaign block a level on layer 2 reports: no usable gates, and the pit the trunk was built against."""
    return {"gates_ordered": False, "gates": [], "exit": {"pos": list(route["exit_pos"]), "active": False}}


def parallel_sets(rungs) -> list[list[str]]:
    """Invariant T's own test: groups of >= 2 SUFFIXED rungs sharing one ordinal.

    T's definition, and the distinction matters both ways. `3C/3DL/3DR/3TL/3TR` on 1-4 is a star of five side
    rooms off one ordinal and is what T removes. A LONE suffix is a chain and is kept -- 5-2's
    `7 - Broken Pier` -> `7B - Crooked Cabin`, 8-3's `10B - Night Street`, 4-4's merged `3A` -- so a group only
    counts when at least TWO of its members carry a suffix. (Measured while writing this: requiring only "two
    members with different suffixes" flags 5-2's `7`/`7B` pair, which the spec keeps on purpose.)

    The emitter applies T inside a numbering SCOPE and the shipped files do not carry the scope, so this reads
    ordinals across the whole file. That is the conservative direction -- it can only over-report -- and no
    shipped level has a suffixed same-ordinal pair straddling two scopes (spec §4.3 T).
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for rung in rungs:
        match = NAME.match(str(rung.get("name") or ""))
        if match:
            groups.setdefault(match.group(1), []).append((match.group(2), str(rung["key"])))
    return [[key for _, key in members] for members in groups.values()
            if sum(1 for suffix, _ in members if suffix) >= 2]


def allowed_orders(rungs) -> list[list[dict]]:
    """Every visit order the level's structure permits, highest `hops` first.

    On a chain -- which is what invariant T guarantees -- that is exactly one order, and A0's assertions are
    then trivially true, which is the point. On a star, the members of a parallel set may be entered in any
    order, so every permutation of each set is replayed: that is what caught revision 1's defect.
    """
    chain = sorted(rungs, key=lambda r: -int(r["hops"]))
    sets = parallel_sets(rungs)
    if not sets:
        return [chain]
    orders = [chain]
    for keys in sets:
        members = [r for r in chain if str(r["key"]) in set(keys)]
        rest = [r for r in chain if str(r["key"]) not in set(keys)]
        first = chain.index(members[0])
        for order in list(permutations(members))[:MAX_ORDERS]:
            orders.append(rest[:first] + list(order) + rest[first:])
    return orders


def replay(route: dict, order, *, patience: int = 0) -> dict:
    """Walks the rungs in `order`, one decision each, the way `UltrakillEnv` drives the tracker.

    The player starts on the first rung of the order, which is what the level's own spawn does on a real trunk
    (measured: the spawn is inside the first rung's reach cylinder, or within `route_seed_m` of it), so the
    first rung is absorbed rather than earned by both the ladder and the seeding rule.
    """
    camp = block(route)
    progress = GateProgress(route=route, patience_steps=patience)
    start = list(order[0]["pos"])
    progress.new_level_load(camp, start)
    progress.reset_episode()
    progress.retarget(camp, start)
    steps = []
    for rung in order:
        paid, _ = progress.update(camp, list(rung["pos"]))
        target = progress.target or {}
        hops = target.get("hops")
        steps.append({"entered": str(rung["key"]), "paid": paid, "target": str(target.get("key")),
                      # The exit sentinel carries `hops: None`; -1 is "one past the last rung", which is what
                      # it is, and it makes the ordering test below a single comparison.
                      "target_hops": -1 if hops is None else int(hops), "best_hops": progress.best_hops})
    return {"steps": steps, "paid": sum(s["paid"] for s in steps), "progress": progress}


def past_unvisited(order, index: int, target_hops: int) -> list[str]:
    """The rungs this step's target has jumped over: unvisited rungs FURTHER from the exit than the target.

    `hops` descends toward the exit, so a target with fewer hops than a rung nobody has entered means the
    route signal has skipped that rung -- and on a branching ladder it never comes back to it, which is the
    signal death §2.6 measured. The exit counts as -1, so handing the target to the pit while any rung is
    unvisited is the most extreme form of the same failure.
    """
    return [str(r["key"]) for r in order[index + 1:] if int(r["hops"]) > target_hops]


# ---------------------------------------------------------------------------
# A0, part 1: the trunk is a chain and every order it allows behaves
# ---------------------------------------------------------------------------

def test_a0_every_shipped_trunk_is_a_chain():
    """Invariant I1 and invariant T, asserted on the loaded document rather than on the file's prose."""
    with tempfile.TemporaryDirectory() as tmp:
        for doc in FIXTURES + tuple(real_routes()):
            route = write(doc, tmp)
            assert route is not None, f"{doc['level']}: the loader refused its own shipped file"
            hops = [int(r["hops"]) for r in route["rungs"]]
            assert sorted(hops, reverse=True) == list(range(len(hops) - 1, -1, -1)), \
                f"{doc['level']}: hops must be n-1 .. 0 with no duplicates, got {hops}"
            assert parallel_sets(route["rungs"]) == [], \
                f"{doc['level']}: invariant T is broken -- a parallel set shipped: {parallel_sets(route['rungs'])}"
            assert len(allowed_orders(route["rungs"])) == 1, f"{doc['level']}: a chain allows one order"


def test_a0_the_target_never_advances_past_an_unvisited_rung():
    """A0's first assertion, over every order the level allows, with parking off and on."""
    with tempfile.TemporaryDirectory() as tmp:
        for doc in FIXTURES + tuple(real_routes()):
            route = write(doc, tmp)
            for order in allowed_orders(route["rungs"]):
                for patience in (0, 300):
                    result = replay(route, order, patience=patience)
                    for i, step in enumerate(result["steps"]):
                        skipped = past_unvisited(order, i, step["target_hops"])
                        assert not skipped, (
                            f"{doc['level']}: after entering {step['entered']} the target is "
                            f"{step['target']} (hops {step['target_hops']}), past the unvisited {skipped}")


def test_a0_no_step_pays_more_than_one_instalment():
    """A0's second assertion. On a chain the whole trunk pays exactly one instalment per rung descended."""
    with tempfile.TemporaryDirectory() as tmp:
        for doc in FIXTURES + tuple(real_routes()):
            route = write(doc, tmp)
            for order in allowed_orders(route["rungs"]):
                result = replay(route, order)
                for step in result["steps"]:
                    assert step["paid"] <= 1, f"{doc['level']}: {step['entered']} paid {step['paid']} at once"
                assert result["paid"] == len(route["rungs"]) - 1, \
                    f"{doc['level']}: {result['paid']} instalments over {len(route['rungs'])} rungs"
                assert result["progress"].best_hops == 0, f"{doc['level']}: the trunk ends at hops 0"
                assert result["progress"].route_source == ROUTE_SOURCE_ROOMS


def test_a0_starting_anywhere_on_the_trunk_pays_the_rungs_ahead_and_no_more():
    """The seeding rule means the player can start at any rung, so every starting point is replayed too.

    `route_seed_m` is 150 m and unvalidated against the real spawns of 11 of the 12 levels (spec §2.8), so
    "the trunk behaves from wherever the load puts the player" is the property that has to hold, not "the
    player starts at rung n-1".
    """
    with tempfile.TemporaryDirectory() as tmp:
        for doc in FIXTURES + tuple(real_routes()):
            route = write(doc, tmp)
            chain = allowed_orders(route["rungs"])[0]
            for start in range(len(chain)):
                result = replay(route, chain[start:])
                ahead = len(chain) - start - 1
                assert result["paid"] == ahead, \
                    f"{doc['level']}: starting at rung {start} paid {result['paid']}, expected {ahead}"
                for step in result["steps"]:
                    assert step["paid"] <= 1


def test_a0_can_fail_the_star_pays_four_instalments_at_once():
    """§2.6 reproduced: the assertions above are not vacuous, they are what invariant T buys.

    Revision 1's 1-4 ladder, entered as the spec measured it -- one of the five side rooms first -- pays four
    instalments in a single step and then points the target past three rungs that were never visited. Both of
    A0's assertions fail on it, which is exactly why the shipped ladder is the trunk alone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        route = write(STAR, tmp)
        assert len(parallel_sets(route["rungs"])) == 1, "the fixture has to BE a star"
        orders = allowed_orders(route["rungs"])
        assert len(orders) > 1, "so more than one visit order is allowed"
        shelf = next(r for r in route["rungs"] if r["name"].startswith("3TR"))
        main = next(r for r in route["rungs"] if r["name"].startswith("3 -"))
        order = [main, shelf] + [r for r in route["rungs"] if r not in (main, shelf)]
        result = replay(route, order)
        at_once = max(step["paid"] for step in result["steps"])
        assert at_once >= 4, f"expected the four-at-once of §2.6, got {at_once}"
        assert [s["paid"] for s in result["steps"][2:]] == [0] * 4, \
            "and then the rest of the star pays nothing at all: the signal is dead, not doubled"
        # The same total either way, which is §2.6's point: this is signal death, not extra reward.
        in_order = replay(route, allowed_orders(route["rungs"])[0])
        assert in_order["paid"] == result["paid"] == len(route["rungs"]) - 1
        # And A0's own assertions are what catch it: the target hands over to the pit with four required side
        # rooms unentered, which is revision 1's measured failure exactly.
        skipped = [past_unvisited(order, i, s["target_hops"]) for i, s in enumerate(result["steps"])]
        assert any(skipped), "A0's target assertion has to be the thing that fails on a star"
        assert len(skipped[1]) == 4 and result["steps"][1]["target"] == "exit"
        assert max(s["paid"] for s in in_order["steps"]) == 1, "while the same ladder in order pays one at a time"


# ---------------------------------------------------------------------------
# A0, part 2: the gates levels are untouched, against recorded data
# ---------------------------------------------------------------------------

def test_a0_a_gates_level_is_byte_for_byte_identical_with_a_route_loaded():
    """`tests/test_ladder_replay.py`'s golden file, replayed with a route document in hand.

    The golden file was produced by the PRE-patience implementation over 32,022 recorded Level 0-1 decisions
    and the two Level 0-3 probes, so this compares the route code against a recording of the behaviour before
    any of it existed -- not against itself. Every one of those levels reports a usable gate ladder, so the
    trunk must never be read, and `A3` in game is the same claim measured live.
    """
    want = ladder.golden()
    with tempfile.TemporaryDirectory() as tmp:
        route = write(SPEC_0_5, tmp)
        for tag, level, name in (("level_0-1_run", "0-1", "level_0-1_run.jsonl.gz"),
                                 ("level_0-3_probe", "0-3", "level_0-3_probe.jsonl.gz"),
                                 ("level_0-3_scripted", "0-3", "level_0-3_scripted.jsonl.gz")):
            camp, eps = ladder.gates(level), ladder.episodes(name)
            for ep, steps in eps.items():
                with_route = ladder.replay(camp, steps, route=route)
                assert ladder.comparable(with_route) == want[tag][ep], f"{tag} ep {ep}: patience off"
                on = ladder.replay(camp, steps, route=route, **ladder.PATIENT)
                off = ladder.replay(camp, steps, **ladder.PATIENT)
                assert ladder.comparable(on) == ladder.comparable(off), f"{tag} ep {ep}: patience on"
                assert on["park_log"] == off["park_log"] and on["parks"] == off["parks"], f"{tag} ep {ep}"


def test_a0_a_gates_level_never_reads_the_trunk_at_all():
    """The same claim from the other side: not "the same answer", but "the file was not consulted"."""
    with tempfile.TemporaryDirectory() as tmp:
        route = write(SPEC_0_5, tmp)
        camp = ladder.gates("0-1")
        steps = ladder.episodes("level_0-1_run.jsonl.gz")["5"]
        progress = GateProgress(route=route, **ladder.PATIENT)
        progress.new_level_load(camp, steps[0][:3])
        progress.reset_episode()
        progress.retarget(camp, steps[0][:3])
        for row in steps[:2000]:
            camp["arena_enemies_alive"] = row[3]
            progress.update(camp, row[:3])
        assert progress.route_reads == 0, "0-1's gate ladder is usable, so the trunk is never the ladder"
        assert progress._hops_source == "gates" and progress.route_source == 1
        assert progress._route_ok is None, "guard I5 is never even evaluated on a gates level"


def test_the_default_route_directory_is_the_packaged_one():
    """`route_dir` "" means `ultrakill_ai/routes/`, which is where the 12 committed files live."""
    assert route_path("Level 0-5") == ROUTE_DIR / "route_Level_0-5.json"
    assert ROUTE_DIR.name == "routes" and ROUTE_DIR.parent.name == "ultrakill_ai"
    assert route_path("Level 0-5", "") == route_path("Level 0-5")
    assert load_route("Level 0-5") is None or ROUTE_DIR.is_dir(), "a missing folder is simply layer 3"


def test_the_hops_slot_is_bounded_so_a_deep_trunk_cannot_pin_it():
    """`spaces.campaign_block` slot 12 is `min(hops, 20) / 20`, a bound rather than a behaviour change.

    A proven no-op on everything that exists -- the deepest gate ladder in the campaign is 13 hops and the
    longest shipped trunk 14 -- but a regenerated file with 25 rungs would otherwise feed a learned input
    column a value above 1.0, on a run whose weights were trained when 13/20 was the maximum.
    """
    from ultrakill_ai.spaces import campaign_block

    obs = {"campaign": {"seconds": 0.0}, "player": {"pos": [0.0, 0.0, 0.0], "yaw": 0.0}}
    def slot12(hops):
        target = {"key": "0,0,10", "pos": [0.0, 0.0, 10.0], "hops": hops, "open": False, "locked": False}
        return campaign_block(obs, target=target)[12]
    assert slot12(14) == 0.7 and slot12(13) == 0.65, "every real ladder is unchanged"
    assert slot12(20) == 1.0 and slot12(25) == 1.0 and slot12(400) == 1.0, "and nothing can read above 1.0"
    assert slot12(0) == 0.0


def real_routes() -> list[dict]:
    """Every committed `routes/route_*.json`, so A0 covers the real files as soon as S1 ships them."""
    out = []
    for path in sorted(ROUTE_DIR.glob("route_*.json")) if ROUTE_DIR.is_dir() else ():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # test_route_files.py is what reports a broken file properly
            raise AssertionError(f"{path.name} is not readable JSON: {exc}") from exc
        out.append(doc)
    return out


if __name__ == "__main__":
    shipped = real_routes()
    print(f"A0 over {len(FIXTURES)} fixtures and {len(shipped)} shipped route files"
          + (f": {', '.join(d.get('level', '?') for d in shipped)}" if shipped else " (none yet)"))
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
