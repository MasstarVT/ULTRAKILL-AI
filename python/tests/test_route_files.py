"""The committed room-trunk route files. No game needed:  python tests/test_route_files.py  (or pytest).

These are the acceptance tests of the route-fallback spec, section 7.1. They read only what is
committed under `ultrakill_ai/routes/` and re-derive every guard from the shipped `pos` list, so a
bug in `scripts/build_routes.py`'s own helpers cannot hide behind them: nothing here imports the
generator, and the checks are written from the spec's wording rather than from its code.

Layer 2 fires only where the live door-gate ladder fails its guard, so a file appearing or vanishing
is a coverage change, never a detail. `test_shipped_set_is_exactly_the_twelve` is what makes a
regeneration that loses a level fail a test instead of a training run.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import CAMPAIGN_LEVELS, safe_name  # noqa: E402

ROUTES = Path(__file__).resolve().parents[1] / "ultrakill_ai" / "routes"

# Spec section 10. Exactly these levels ship a room trunk: 18 stay on the gate ladder and
# 1-3, 5-4 and 6-2 fall through to the exit vector.
SHIPPED = ("Level 0-5", "Level 1-4", "Level 2-4", "Level 4-2", "Level 4-4", "Level 5-2",
           "Level 7-1", "Level 7-2", "Level 7-3", "Level 7-4", "Level 8-3", "Level 8-4")
UNROUTED = ("Level 1-3", "Level 5-4", "Level 6-2")

VERSION = 2
SEP = 16.0              # I2: 2 x GateProgress.gate_reach_m
MIN_RUNGS = 3           # R1
MAX_RUNGS = 24          # I3
MAX_TOUR = 1.35         # R2
PIT_MAX_M = 70.0        # the `Pit` exception

# Optional-area names, spec 4.3 I6. Pinned by name because both rules were added on a sample of
# exactly three campaign-wide occurrences: `6S - P Door` (3-1), `1S - P Door` (6-2) and
# `10S - Secret Arena` (8-3). Revision 1 shipped `1S - P Door` as 6-2's hops-1 rung -- the Prime
# Sanctum door, which needs every level in the layer at P rank and is never on the route to the exit
# -- and 6-2's whole claim to a route rested on it. `S - Secret Fight` is 1-3's.
BANNED_NAMES = ("1S - P Door", "6S - P Door", "10S - Secret Arena", "S - Secret Fight")

RX_ORD = re.compile(r"^(\d+)([A-Za-z][A-Za-z0-9]?)?\s*-\s+(.*)$")
RX_BR = re.compile(r"^([A-Z])(\d{1,2})\s*-\s+(.*)$")
RX_SECRET = re.compile(r"\b(secret|bonus)\b", re.I)

REQUIRED_KEYS = ("level", "version", "source", "exit", "start_room", "trunk_collapsed",
                 "tour_ratio", "checkpoints_within_60m", "legs_witnessed",
                 "last_rung_to_exit_m", "rungs")
RUNG_KEYS = ("key", "pos", "hops", "name", "gated_by", "open", "locked", "active")


# ---------------------------------------------------------------- helpers, deliberately independent

def files() -> dict[str, Path]:
    return {p.name: p for p in sorted(ROUTES.glob("route_*.json"))}


def docs() -> dict[str, dict]:
    out = {}
    for name, path in files().items():
        with open(path, encoding="utf8") as f:
            out[name] = json.load(f)
    return out


def shipped_docs() -> dict[str, dict]:
    """{level: document} for the files that carry rungs."""
    return {d["level"]: d for d in docs().values() if d.get("rungs")}


def lead(name: str) -> str:
    """A rung's own name: the part before any `+` an I2 merge concatenated."""
    return name.split(" + ")[0].strip()


def name_kind(name: str) -> tuple[str, int | None, str]:
    m = RX_ORD.match(name)
    if m:
        return ("ord", int(m.group(1)), m.group(2) or "")
    m = RX_BR.match(name)
    if m:
        return ("branch", int(m.group(2)), m.group(1))
    return ("other", None, "")


def dist3(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def tour_ratio(pts: list[list[float]]) -> float:
    """The ladder's own polyline over a greedy nearest-neighbour tour of the same points, from the
    same start. Re-derived here from the spec's wording rather than imported."""
    if len(pts) < 3:
        return 1.0
    poly = sum(dist3(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    left, cur, tour = list(range(1, len(pts))), 0, 0.0
    while left:
        j = min(left, key=lambda k: dist3(pts[cur], pts[k]))
        tour += dist3(pts[cur], pts[j])
        left.remove(j)
        cur = j
    return poly / tour if tour else 1.0


# ---------------------------------------------------------------- 1. schema

def test_schema():
    assert ROUTES.is_dir(), "no routes directory at %s" % ROUTES
    for name, doc in docs().items():
        for k in REQUIRED_KEYS:
            assert k in doc, "%s is missing %r" % (name, k)
        assert doc["version"] == VERSION, "%s ships version %r" % (name, doc["version"])
        assert name == "route_%s.json" % safe_name(doc["level"]), \
            "%s is labelled %r" % (name, doc["level"])
        assert doc["level"] in CAMPAIGN_LEVELS, "%s is not a campaign level" % doc["level"]
        assert doc["rungs"], "%s ships no rungs (delete the file instead)" % name
        assert isinstance(doc["source"], str) and doc["source"]
        ex = doc["exit"]
        assert isinstance(ex["pos"], list) and len(ex["pos"]) == 3
        assert all(isinstance(v, (int, float)) for v in ex["pos"])
        assert isinstance(ex["target"], str) and ex["target"]
        for k in ("checkpoints_within_60m", "legs_witnessed"):
            assert re.fullmatch(r"\d+/\d+", doc[k]), "%s %s = %r" % (name, k, doc[k])
        assert isinstance(doc["trunk_collapsed"], list)
        assert isinstance(doc["last_rung_to_exit_m"], (int, float))
        for r in doc["rungs"]:
            for k in RUNG_KEYS:
                assert k in r, "%s rung %r is missing %r" % (name, r.get("name"), k)
            assert isinstance(r["pos"], list) and len(r["pos"]) == 3
            assert all(isinstance(v, (int, float)) for v in r["pos"])
            assert isinstance(r["name"], str) and r["name"]


# ---------------------------------------------------------------- 2. I1 total order

def test_hops_are_a_strict_total_order():
    for level, doc in shipped_docs().items():
        hops = [r["hops"] for r in doc["rungs"]]
        assert all(isinstance(h, int) for h in hops), "%s has a non-integer hops" % level
        assert hops == list(range(len(hops) - 1, -1, -1)), \
            "%s hops are %r, not %d..0" % (level, hops, len(hops) - 1)


# ---------------------------------------------------------------- 3. I2 separation

def test_no_two_rungs_share_a_reach_cylinder():
    """`_is_reached` marks every rung within 8 m horizontal / 6 m vertical, so two rungs inside
    2 x gate_reach_m would be marked by one arrival and skip a hop for free."""
    for level, doc in shipped_docs().items():
        for a, b in itertools.combinations(doc["rungs"], 2):
            d = dist3(a["pos"], b["pos"])
            assert d >= SEP, "%s: %r and %r are %.1f m apart" % (level, a["name"], b["name"], d)


# ---------------------------------------------------------------- 4. I3 budget cap

def test_budget_cap():
    for level, doc in shipped_docs().items():
        assert len(doc["rungs"]) <= MAX_RUNGS, "%s ships %d rungs" % (level, len(doc["rungs"]))


# ---------------------------------------------------------------- 5. R1 and R2

def test_at_least_three_rungs():
    for level, doc in shipped_docs().items():
        assert len(doc["rungs"]) >= MIN_RUNGS, "%s ships %d rungs" % (level, len(doc["rungs"]))


def test_tour_ratio_passes_and_is_reproduced_from_the_shipped_positions():
    """R2 is judged on what SHIPS. Revision 1 stored 8-3's ratio for a 28-rung chain that the budget
    cap then cut to 24, and those 24 scored 1.421 -- over the threshold the stored 1.264 claimed to
    pass. A guard evaluated on data that does not ship is not a guard, so the stored value has to be
    reproducible from the `pos` list in the same file."""
    for level, doc in shipped_docs().items():
        repro = tour_ratio([r["pos"] for r in doc["rungs"]])
        assert abs(repro - doc["tour_ratio"]) <= 5e-4, \
            "%s stores tour_ratio %.4f but its own positions score %.4f" \
            % (level, doc["tour_ratio"], repro)
        assert doc["tour_ratio"] <= MAX_TOUR, "%s tour_ratio %.3f" % (level, doc["tour_ratio"])


# ---------------------------------------------------------------- 6. I6 optional areas

def test_no_optional_area_is_on_the_route():
    for level, doc in shipped_docs().items():
        for r in doc["rungs"]:
            for piece in (p.strip() for p in r["name"].split(" + ")):
                _kind, _ordn, suf = name_kind(piece)
                low = piece.lower()
                assert not RX_SECRET.search(piece), "%s: %r is a secret/bonus area" % (level, piece)
                assert not suf.upper().startswith("S"), "%s: %r carries an S suffix" % (level, piece)
                assert "p door" not in low, "%s: %r is a Prime Sanctum door" % (level, piece)
                assert "prime" not in low, "%s: %r is Prime content" % (level, piece)


def test_the_named_i6_regressions_stay_fixed():
    """1-3's `S - Secret Fight` must not reappear at hops 1 and 6-2's `1S - P Door` must not reappear
    as a route rung -- 6-2's whole claim to a route rested on it."""
    every = {piece
             for doc in shipped_docs().values()
             for r in doc["rungs"]
             for piece in (p.strip() for p in r["name"].split(" + "))}
    for banned in BANNED_NAMES:
        assert banned not in every, "%r is back on the route" % banned


# ---------------------------------------------------------------- 7. T trunk only

def test_only_the_trunk_ships():
    """`GateProgress` carries one `best_hops` and `_note_reached` collapses it to the minimum over
    every rung, so a parallel set cannot be expressed: whichever member the agent enters first pays
    the whole set at once and the rest go dark. A same-ordinal suffix group and a multi-prefix branch
    region are therefore collapsed onto the rung they hang off."""
    for level, doc in shipped_docs().items():
        by_ord: dict[int, set[str]] = {}
        prefixes: set[str] = set()
        for r in doc["rungs"]:
            kind, ordn, suf = name_kind(lead(r["name"]))
            if kind == "ord":
                by_ord.setdefault(ordn, set()).add(suf)
            elif kind == "branch":
                prefixes.add(suf)
        for ordn, sufs in sorted(by_ord.items()):
            lettered = sorted(s for s in sufs if s)
            assert len(lettered) < 2, \
                "%s: ordinal %d ships the suffixes %r" % (level, ordn, lettered)
        assert len(prefixes) <= 1, "%s ships %d branch prefixes: %r" % (level, len(prefixes),
                                                                        sorted(prefixes))


def test_7_1_ships_the_ordinal_fixed_order():
    """`ActivateNextWave` containers match the `<N> - <Name>` room regex, and on 7-1 `1 - Wave 1` and
    `1 - Wave 2` sorted as ordinal 1 and dragged `2 - Left Arena` down with them through the I2 merge,
    shipping `1, 3, 4, 5, 2, 1, 2, 3, 4, 5` -- room 2 after room 5 inside one numbering, which R2 does
    not catch because those rooms are spatially clustered. Excluding wave containers from the room
    inventory fixes it; this pins the fix, because nothing else can see it."""
    doc = shipped_docs()["Level 7-1"]
    names = [r["name"] for r in doc["rungs"]]
    assert names == ["1 - Pillar Hall", "2 - Left Arena", "3 - Spiral Staircase",
                     "4 - Interior Exterior", "5 - Exit", "1 - Entry",
                     "2 - Underground Tunnels", "3 - Fight Segment", "4 - Outro",
                     "5 - End Arena"], names
    assert not any("Wave" in n for n in names), names
    ordinals = [name_kind(lead(n))[1] for n in names]
    assert ordinals == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], ordinals


# ---------------------------------------------------------------- 8. keys and flags

def test_key_is_the_position_rounded_and_the_flags_are_the_exit_convention():
    """`open` is false, not true: `_is_reached` doubles its reach cylinder when `open` is true, which
    would collide rungs and silently widen R4's own criterion, and `open` is observation slot 453 --
    shipping true would pin a learned input column to 1.0 on every fallback level. false is what
    `GateProgress._exit` itself uses."""
    for level, doc in shipped_docs().items():
        for r in doc["rungs"]:
            want = "%d,%d,%d" % tuple(int(round(v)) for v in r["pos"])
            assert r["key"] == want, "%s: %r has key %r, want %r" % (level, r["name"], r["key"], want)
            assert r["open"] is False, "%s: %r ships open true" % (level, r["name"])
            assert r["locked"] is False, "%s: %r ships locked true" % (level, r["name"])
            assert r["active"] is True, "%s: %r ships active false" % (level, r["name"])
            assert isinstance(r["gated_by"], list), "%s: %r gated_by" % (level, r["name"])
            for g in r["gated_by"]:                     # empty until stage S3 fills it
                assert isinstance(g, list) and len(g) == 3 and all(
                    isinstance(v, (int, float)) for v in g), \
                    "%s: %r gated_by entry %r is not a position" % (level, r["name"], g)


# ---------------------------------------------------------------- 9. the Pit exception

def test_a_pit_rung_is_only_ever_the_last_one():
    """7-4 and 8-4 both ship a hops-0 rung named `Pit`: it is the FinalPit prefab's own root, 62.2 m
    from the pit on both, and the voxel probe finds standable floor there. "Make the object named
    `Pit` the goal" is on this project's list of previously-rejected rules, so the exception is pinned
    rather than left to look like that rule sneaking back in."""
    for level, doc in shipped_docs().items():
        for r in doc["rungs"]:
            if lead(r["name"]) != "Pit":
                continue
            assert r["hops"] == 0, "%s: a Pit rung at hops %d" % (level, r["hops"])
            d = dist3(r["pos"], doc["exit"]["pos"])
            assert d < PIT_MAX_M, "%s: the Pit rung is %.1f m from the pit" % (level, d)


# ---------------------------------------------------------------- 10. the coverage set

def test_shipped_set_is_exactly_the_twelve():
    got = tuple(sorted(shipped_docs(), key=CAMPAIGN_LEVELS.index))
    assert got == SHIPPED, "shipped %r, expected %r" % (got, SHIPPED)


def test_the_unrouted_levels_ship_nothing():
    """1-3 and 6-2 lose their route to the guards and 5-4 has one rung, no numbered rooms and no
    gate-candidate doors. All three must fall through to the exit vector, which means no file at all
    or a file whose `rungs` is empty -- `_read_route` refuses both."""
    present = docs()
    for level in UNROUTED:
        name = "route_%s.json" % safe_name(level)
        if name in present:
            assert not present[name].get("rungs"), \
                "%s ships %d rungs; it must fall to layer 3" % (level, len(present[name]["rungs"]))


def test_no_file_ships_for_a_gates_level():
    """The 18 levels layer 1 routes never read a file -- `_rooms()` is only reached on the three
    `return []` arms of `_gates()` -- so a file there would be dead weight and a coverage change."""
    for level in CAMPAIGN_LEVELS:
        if level in SHIPPED or level in UNROUTED:
            continue
        name = "route_%s.json" % safe_name(level)
        assert name not in files(), "%s is on the gate ladder but ships %s" % (level, name)


# ---------------------------------------------------------------- 11. the shape of each ladder

# Every guard above says a ladder is WELL FORMED. None of them says it is the SAME ladder, and the
# coverage set cannot see a change inside a file that still ships. `scripts/build_routes.py` rewrites
# these files in place and spec risk 3 says to rerun it after every game update, so a patch that
# moves one room, or a room the parser stops recognising, would re-aim the agent at a different
# sequence of places with every structural invariant still passing. This table is the pin: the
# ladders as measured on 2026-09-17, one line per rung, `key  name` -- the key being the position
# rounded to whole metres, exactly the string `CampaignPatches.Key()` builds and `_is_reached`
# matches on. `build_routes.py`'s `EXPECTED_SHAPE` pins the same thing from the generator's side, on
# the numbers this file cannot recompute without the scene bundles; both have to be edited, with a
# measured reason, for a level's route to change.
#
#   level -> (tour_ratio, legs_witnessed, checkpoints_within_60m, last_rung_to_exit_m, [rungs])
EXPECTED_LADDERS = {
    "Level 0-5":   (1.000, "3/5",  "1/1",    211.1, [
        "0,-10,300  1 - Opening Hallway",
        "0,0,351  2 - Lava Foundry",
        "134,-6,382  3 - Smallway",
        "174,-6,382  4 - Cerberus Arena",
        "214,-6,382  5 - Final Hallway",
    ]),
    "Level 1-4":   (1.000, "2/4",  "2/2",    149.5, [
        "0,-10,300  1 - Opener",
        "0,-15,381  2 - Bridge",
        "0,-15,446  3 - Main Hall",
        "0,-17,563  V2 - Arena",
    ]),
    "Level 2-4":   (1.000, "1/4",  "1/2",    111.2, [
        "0,-10,0  1 - Tram Tunnel",
        "14,-10,430  2 - Sidehalls",
        "0,-12,650  0 - Tram + 3 - First Encounter",
        "425,-10,650  4 - Second Encounter",
    ]),
    "Level 4-2":   (1.000, "4/6",  "2/4",    162.9, [
        "0,-10,300  1 - Opening",
        "43,-14,466  2 - Cerberus Arena",
        "66,-18,515  3 - More Platforming",
        "-2,-18,774  4 - Arena",
        "8,15,904  6 - Solarium",
        "8,-15,1158  7 - Boss Arena",
    ]),
    "Level 4-4":   (1.000, "4/8",  "2/3",    539.9, [
        "0,-10,300  1 - Underground",
        "0,0,362  2 - Elevator 1",
        "0,320,415  3 - Ground Floor + 3A - Elevator 2",
        "-11,639,472  4 - Pit Bridge",
        "108,648,465  5 - Window Hallway",
        "118,664,355  6 - Boss Entrance",
        "118,664,323  7 - Boss Arena 1",
        "1065,254,702  8 - Outro",
    ]),
    "Level 5-2":   (1.000, "6/7",  "3/4",    372.8, [
        "0,-10,300  1A - Opening + 1B - Second Rock",
        "4,-20,707  2 - Fort",
        "-10,-5,834  3 - Ferryman's Cabin",
        "-10,-5,866  4 - Dark Hallway",
        "-10,-5,898  5 - Library Loft",
        "52,-3,960  7 - Broken Pier",
        "104,-11,947  7B - Crooked Cabin",
    ]),
    "Level 7-1":   (1.000, "5/10", "2/5",    127.9, [
        "126,21,483  1 - Pillar Hall",
        "164,24,509  2 - Left Arena",
        "202,21,483  3 - Spiral Staircase",
        "217,1,467  4 - Interior Exterior",
        "217,-22,438  5 - Exit",
        "-240,0,300  1 - Entry",
        "-242,-114,324  2 - Underground Tunnels",
        "-242,0,0  3 - Fight Segment",
        "-242,80,0  4 - Outro",
        "-242,95,-376  5 - End Arena",
    ]),
    "Level 7-2":   (1.255, "10/15", "5/6",   168.3, [
        "0,-10,300  1 - Empty Hall",
        "0,-8,340  2 - Whiplash Course",
        "0,28,340  3 - Switch Tutorial",
        "-15,28,274  4 - Obstacle Course",
        "-100,28,274  5 - Corner Staircase",
        "-115,38,304  6 - Gutterman Intro",
        "-115,55,422  7 - Outdoors Start",
        "-120,34,552  8 - Sunken Pagoda and Fallen Tower",
        "-24,38,806  9 - Tram Station",
        "-218,28,862  10 - Ambush Station",
        "-306,30,620  11 - Bomb Station",
        "-273,30,570  12 - Red Skull Trench",
        "-306,30,550  13 - Last Obstacle Course",
        "46,6,701  14 - Broken Hallway",
        "88,6,701  15 - Forgotten Archive",
    ]),
    "Level 7-3":   (1.174, "8/12", "4/4",    224.9, [
        "0,-10,300  1 - Dark Path",
        "-13,-10,484  2 - Garden Maze",
        "-58,-10,484  3 - Central Plaza",
        "-146,-20,539  4 - Forest Path",
        "-110,-15,608  5 - Exterior Arena",
        "-16,8,604  6 - Interior Garden",
        "-176,-2,514  7 - Central Plaza Upper",
        "-96,8,409  8 - Upper Garden Battlefield",
        "-148,8,276  9 - Circular Garden",
        "-188,8,316  10 - Garden Corridor",
        "-146,-15,449  11 - Stairs House",
        "-212,-35,484  12 - Grand Hall",
    ]),
    "Level 7-4":   (1.000, "3/4",  "5/7",     62.2, [
        "28,470,746  2 - Security Checkpoint",
        "0,506,671  3 - Entrance Checkpoint",
        "0,793,619  4 - Brain Checkpoint + 5 - Return Checkpoint",
        "-2,124,1247  Pit",
    ]),
    "Level 8-3":   (1.239, "12/16", "8/13", 1689.6, [
        "-10,6,450  1 - Escher Entrance",
        "-41,1,427  2 - Shifting Hallway",
        "-152,1,427  3 - Shifting Arena",
        "-293,1,427  4 - Fallway",
        "-363,1,473  5 - Upside-Down Center",
        "-368,10,684  6 - Rotating Hallway",
        "-68,10,765  7 - Heart Chamber",
        "200,80,0  8 - Garage Loop",
        "200,45,46  9 - Loop Drop",
        "200,48,213  10 - Split Color Door",
        "355,68,213  10B - Night Street",
        "300,40,615  11 - Fake End + 12 - Space Platforming",
        "513,254,941  13 - Space Streets",
        "417,288,733  14 - Space House",
        "201,155,928  15 - Space Mass",
        "308,190,1027  16 - Space Tree",
    ]),
    "Level 8-4":   (1.000, "0/4",  "0/1",     62.2, [
        "0,-10,300  1 - Lower Intro",
        "-28,0,290  2 - Maintenance",
        "-28,50,249  3 - Upper Intro",
        "-200,-52,482  Pit",
    ]),
}

TOTAL_RUNGS = 95


def test_the_ladder_table_covers_exactly_the_shipped_levels():
    """A level added to `SHIPPED` without a row here would ship unpinned, which is the whole hole
    this table closes."""
    assert tuple(EXPECTED_LADDERS) == SHIPPED, \
        "EXPECTED_LADDERS covers %r, SHIPPED is %r" % (tuple(EXPECTED_LADDERS), SHIPPED)
    assert sum(len(v[4]) for v in EXPECTED_LADDERS.values()) == TOTAL_RUNGS


def test_each_ladder_ships_exactly_the_recorded_rungs():
    """The rungs themselves, in order, by key and name.

    8-3 is the row to read before editing this table: it ships 16 rungs where spec section 10 says
    13, because the design measured it on revision 1's file, which the budget cap had already cut to
    24 rungs -- dropping `2 - Shifting Hallway`, `3 - Shifting Arena` and `10 - Split Color Door` --
    before the trunk collapse ran. Spec section 4.3 moves I3 last, so after T the cap binds on
    nothing and those three rooms survive. That deviation was found by a human reading the table, not
    by a tool, and this test exists so the next one is not.
    """
    for level, doc in shipped_docs().items():
        assert level in EXPECTED_LADDERS, "%s ships a ladder with no row in EXPECTED_LADDERS" % level
        want = EXPECTED_LADDERS[level][4]
        got = ["%s  %s" % (r["key"], r["name"]) for r in doc["rungs"]]
        assert len(got) == len(want), \
            "%s ships %d rungs, expected %d:\n  got  %s\n  want %s" \
            % (level, len(got), len(want), "\n       ".join(got), "\n       ".join(want))
        for i, (g, w) in enumerate(zip(got, want)):
            assert g == w, "%s rung %d (hops %d) is %r, expected %r" \
                % (level, i, len(got) - 1 - i, g, w)


def test_each_ladders_stored_diagnostics_are_the_measured_ones():
    """`tour_ratio` is re-derived from `pos` above, but `legs_witnessed`, `checkpoints_within_60m`
    and `last_rung_to_exit_m` need the scene bundles, so offline they can only be pinned. They are
    the columns spec section 10 is read by, and a silent move in any of them is a coverage change."""
    for level, doc in shipped_docs().items():
        assert level in EXPECTED_LADDERS, "%s ships a ladder with no row in EXPECTED_LADDERS" % level
        tour, legs, cps, last, _rungs = EXPECTED_LADDERS[level]
        assert abs(doc["tour_ratio"] - tour) <= 5e-4, \
            "%s tour_ratio %.3f, expected %.3f" % (level, doc["tour_ratio"], tour)
        assert doc["legs_witnessed"] == legs, \
            "%s legs_witnessed %s, expected %s" % (level, doc["legs_witnessed"], legs)
        assert doc["checkpoints_within_60m"] == cps, \
            "%s checkpoints_within_60m %s, expected %s" % (level, doc["checkpoints_within_60m"], cps)
        assert abs(doc["last_rung_to_exit_m"] - last) <= 0.05, \
            "%s last_rung_to_exit_m %.1f, expected %.1f" % (level, doc["last_rung_to_exit_m"], last)


def test_files_stay_small():
    """95 rungs over 12 files, 17.1 KB measured. They are read once per level load and committed, so a
    file that grew by an order of magnitude means the trunk collapse stopped working."""
    sizes = {name: path.stat().st_size for name, path in files().items()}
    for name, n in sizes.items():
        assert n <= 8 * 1024, "%s is %d bytes" % (name, n)
    total = sum(sizes.values())
    assert total <= 32 * 1024, "the route files total %d bytes" % total


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
