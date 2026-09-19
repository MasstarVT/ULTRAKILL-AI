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

# Spec section 10, plus the 2026-09-17 lead ruling. Twelve levels ship a trunk because their gate
# ladder is unusable; 0-3 and 4-3 ship one BESIDE a usable-but-collapsed ladder, read only by a run
# with `prefer_route_when_collapsed: true` (default false). 16 levels stay on the gate ladder and
# 1-3, 5-4 and 6-2 fall through to the exit vector.
SHIPPED = ("Level 0-3", "Level 0-5", "Level 1-4", "Level 2-4", "Level 4-2", "Level 4-3", "Level 4-4",
           "Level 5-2", "Level 7-1", "Level 7-2", "Level 7-3", "Level 7-4", "Level 8-3", "Level 8-4")
# The two of those that also have a gate ladder, and the only levels allowed to have both.
COLLAPSED_SHIP = ("Level 0-3", "Level 4-3")
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

def test_shipped_set_is_exactly_the_fourteen():
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
    """A level layer 1 routes never reads a file, so a file there would be dead weight and a coverage
    change -- with exactly two exceptions, `COLLAPSED_SHIP`, whose ladder is usable but collapses at
    the spawn. Their file is inert unless a run sets `prefer_route_when_collapsed`, and
    `GateProgress` gives a HEALTHY level's file no way in at all whatever that flag says."""
    for level in CAMPAIGN_LEVELS:
        if level in SHIPPED or level in UNROUTED:
            continue
        name = "route_%s.json" % safe_name(level)
        assert name not in files(), "%s is on the gate ladder but ships %s" % (level, name)


def test_the_collapsed_levels_that_ship_are_the_two_the_lead_approved():
    """0-3 and 4-3 only. 1-1 waits for S3 to stamp its skull locks; 1-2, 2-3 and 8-1 fail the trunk
    guards (tour 1.426, tour 1.393, and an exit room guard T drops 1513 m short). A third collapsed
    level appearing here would be a coverage change made by editing `build_routes.COLLAPSED_SHIP`."""
    for level in COLLAPSED_SHIP:
        assert level in SHIPPED, level
    for level in ("Level 1-1", "Level 1-2", "Level 2-3", "Level 8-1"):
        assert "route_%s.json" % safe_name(level) not in files(), level


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
    # The two collapsed-ladder levels. 0-3 ships FOUR of the survey's eleven rooms: 'Double Down'
    # forks into Path 1 (y 5 -> -15) and Path 2 (y 50), guard T cannot see a fork spelled mid-name,
    # and the generator chained the two mutually exclusive branches in series. The seven rungs of that
    # side wing were dropped on 2026-09-18 by the `drop` list in rung_overrides.json; the four here
    # are the ones the level's only recorded completions credited (gates_reached 3 and 4, never more).
    # `test_the_0_3_wing_stays_off_the_route` below names them individually.
    # The four became SIX on 2026-09-18: the leg from the Side Hallway to Floor 2 was 76.0 degrees
    # (10 m horizontal against 40 m vertical), so the target vector carried no heading and 219 of 264
    # fresh episodes stalled on the hallway rung with zero completions in 4.3M steps. Two waypoints
    # from the AI's own completing run were spliced onto the spiral ramp by `apply_inserts`, and
    # Floor 2 moved off its centroid onto the floor the run actually walks. Steepest leg now 43.2
    # degrees; the ladder is still five hops deep, so the gate income a load can earn without
    # finishing is unchanged. `tests/test_route_inserts.py` holds the evidence and walks the trace.
    "Level 0-3":   (0.972, "2/6", "1/4", 119.7, [
        "0,-10,300  1 - Main Room - Floor 1",
        # Moved from the room centroid 0,10,331 by route_overrides.json, 2026-09-18: the centroid sat
        # 1.5 m past the main room's far wall (z 329.5), so the reach cylinder covered the wall face
        # and the rung was credited from the air on the main-room side. See the override's `why`.
        "0,10,340  2 - Side Hallway - Floor 1",
        "-11,22,327  WP1 - Main Room Stack Foot",
        "5,37,329  WP2 - Main Room Stack Top",
        # Moved from the room centroid 0,50,330: the centroid is at the EDGE of the second floor and
        # the AI's own completion enters its cylinder once, on the way out.
        "-7,48,315  10 - Main Room - Floor 2",
        "-82,90,315  10B - Second Encounter + 11 - Boss Arena - Floor 2",
    ]),
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
    "Level 4-3":   (1.000, "6/8", "3/3", 218.5, [
        "0,-10,300  1 - First Chambers",
        "2,-40,548  2 - Torches Arena",
        "2,-40,630  3 - Traitor Hallway",
        "-17,-40,676  3B - Tomb of Kings",
        "47,-50,676  4 - Pit Room",
        "62,-30,624  5 - Cerberus Room",
        "129,-29,589  6 - Generator Room Hallway",
        "191,-19,589  7 - Generator Room",
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

TOTAL_RUNGS = 109  # 95 over the twelve unrouted levels, plus 0-3's 6 (4 rooms + 2 waypoints) and 4-3's 8


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


# ---------------------------------------------------------------- 12. the manual rung overrides

OVERRIDES = ROUTES / "rung_overrides.json"


def overrides() -> dict[str, list[dict]]:
    if not OVERRIDES.exists():
        return {}
    with open(OVERRIDES, encoding="utf8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def test_the_overrides_file_is_not_picked_up_as_a_route():
    """It lives beside the routes, so it must not match the `route_*.json` glob every reader uses --
    `files()` here, and `docs()`, which would try to parse it as a route document."""
    assert "rung_overrides.json" not in files()
    for name in files():
        assert name.startswith("route_Level_"), name


def test_every_override_is_present_in_the_file_that_ships():
    """The override mechanism's whole purpose is that a REGENERATION keeps a measured fix. Nothing in
    the emitted JSON records that a rung was overridden, so if `build_routes.py` ever stopped applying
    them -- a refused `was` check after a game update, a renamed room, a lost call -- the files would
    quietly go back to the centroid the fix exists to avoid, and every other test here would pass.

    This is the detector: each entry's `pos` has to be the position that actually ships, and its `was`
    must NOT be, or the override is doing nothing and should be deleted rather than left as decoration.
    """
    docs_by_level = shipped_docs()
    for level_short, entries in overrides().items():
        level = "Level %s" % level_short
        assert level in docs_by_level, "%s has overrides but ships no route file" % level
        rungs = {r["name"]: r for r in docs_by_level[level]["rungs"]}
        for entry in entries:
            name = entry["name"]
            if entry.get("drop"):
                # The other kind of entry, and the same detector the other way round: a drop that
                # stopped being applied would put the rung back on the route in silence.
                assert name not in rungs, \
                    "%s override asks to drop %r, which is still a rung in the shipped file -- the " \
                    "regeneration did not apply it (check --validate for a REFUSED note)" % (level, name)
                assert "pos" not in entry and "was" not in entry, \
                    "%s %r is a drop entry and must carry neither `pos` nor `was`" % (level, name)
                assert name in (docs_by_level[level].get("trunk_dropped") or ()), \
                    "%s drops %r but the shipped file does not record it in `trunk_dropped`" \
                    % (level, name)
                continue
            if entry.get("insert_after"):
                # The third kind. It has no `was` -- there is no generated position for a rung the
                # generator did not produce -- so it must not fall into the move branch below.
                # `tests/test_route_inserts.py` pins the anchor order and the flags; what matters
                # here is the same detector as the other two: the entry reached the file that ships.
                assert name in rungs, \
                    "%s inserts %r, which is not a rung in the shipped file -- the regeneration " \
                    "did not apply it (check --validate for a REFUSED note)" % (level, name)
                assert "was" not in entry and "drop" not in entry, \
                    "%s %r is an insert entry and must carry neither `was` nor `drop`" % (level, name)
                assert rungs[name]["pos"] == [round(float(v), 1) for v in entry["pos"]], \
                    "%s %r ships at %s, not the %s the insert asks for" \
                    % (level, name, rungs[name]["pos"], entry["pos"])
                assert rungs[name].get("waypoint") is True, \
                    "%s %r ships without `waypoint: true`" % (level, name)
                assert name in [i["name"] for i in (docs_by_level[level].get("trunk_inserted") or ())], \
                    "%s inserts %r but the shipped file does not record it in `trunk_inserted`" \
                    % (level, name)
                continue
            assert name in rungs, \
                "%s override names %r, which is not a rung in the shipped file" % (level, name)
            got = rungs[name]["pos"]
            assert got == entry["pos"], \
                "%s %r ships at %s, but the override asks for %s -- the regeneration did not apply " \
                "it (check --validate for a REFUSED note)" % (level, name, got, entry["pos"])
            assert entry["was"] != entry["pos"], \
                "%s %r overrides a position to itself; delete the entry" % (level, name)
            assert got == [round(float(v), 1) for v in entry["pos"]], \
                "%s %r: route positions are stored to 0.1 m" % (level, name)
            assert rungs[name]["key"] == ",".join(str(int(round(v))) for v in got), \
                "%s %r: the key does not match its own position" % (level, name)


WING_0_3 = ("3 - Side Arena - Floor 1", "4 - Side Stairway - Floor 1-2",
            "5 - Path 1 - First Encounter", "6 - Path 1 - Boss Arena",
            "7 - Path 2 - Menacing Room", "8 - Path 2 - Menacing Hallway", "9 - Windtunnel")


def test_the_0_3_wing_stays_off_the_route():
    """`test_only_the_trunk_ships` CANNOT catch 0-3, which is why this row exists beside it.

    Guard T collapses a parallel branch only when the branch marker is a LEADING letter
    (`RX_G_BR = ^([A-Z])(\\d{1,2})\\s*-\\s+`). 0-3 spells its fork in the middle of the room name --
    `5 - Path 1 - First Encounter` against `7 - Path 2 - Menacing Room` -- so every one of its rooms
    parses as a plain ordinal with an empty suffix, no rung is kind `branch`, and that test's
    `len(prefixes) <= 1` is vacuous on exactly the level that needed it. The two mutually exclusive
    branches therefore shipped chained in series until 2026-09-18.

    Measured before the drop: 238 fresh `spec_0-3` episodes over 2.83M steps, zero fresh completions,
    115 of them (48%) walking the wing to `7 - Path 2 - Menacing Room` for 7 x 15.0 = 105 reward --
    more than the 100.0 `level_complete` pays -- and 200 of the 263 episodes that got there ending
    back in the Path 1 bowl. All four recorded 0-3 completions credited 3 or 4 rungs and none of them
    ever entered the wing. Restoring any of these seven re-creates that trade.
    """
    doc = shipped_docs()["Level 0-3"]
    names = {r["name"] for r in doc["rungs"]}
    for banned in WING_0_3:
        assert banned not in names, "0-3 ships %r again; see rung_overrides.json" % banned
    assert [r["name"] for r in doc["rungs"]] == [
        "1 - Main Room - Floor 1", "2 - Side Hallway - Floor 1",
        # The two waypoints `apply_inserts` splices onto the main-room spiral ramp. They are the
        # only rungs on this level the room survey did not produce, and the only reason the list
        # above is longer than the four rooms the drop list left.
        "WP1 - Main Room Stack Foot", "WP2 - Main Room Stack Top",
        "10 - Main Room - Floor 2",
        "10B - Second Encounter + 11 - Boss Arena - Floor 2"], [r["name"] for r in doc["rungs"]]
    assert sorted(doc["trunk_dropped"]) == sorted(WING_0_3), doc["trunk_dropped"]


def test_a_drop_entry_names_a_room_that_was_really_there():
    """The drop list is measured against a SHAPE, so a name that never appears in the survey's trunk
    for that level is a typo the generator cannot tell from a room the guards took first: both look
    like 'no rung named X'. `build_routes.py` REFUSES such an entry rather than ignoring it, and
    `--validate` turns the refusal into a non-zero exit -- but that needs the scene bundles, so pin
    the one thing that can be checked offline: every dropped name is recorded by the file that ships,
    and no name is both dropped and kept."""
    for level_short, entries in overrides().items():
        doc = shipped_docs()["Level %s" % level_short]
        dropped = [e["name"] for e in entries if e.get("drop")]
        assert len(set(dropped)) == len(dropped), "Level %s drops a name twice: %r" % (level_short,
                                                                                       dropped)
        recorded = list(doc.get("trunk_dropped") or ())
        assert sorted(recorded) == sorted(dropped), \
            "Level %s drops %r but the file records %r" % (level_short, sorted(dropped),
                                                           sorted(recorded))
        kept = {r["name"] for r in doc["rungs"]}
        assert not (set(dropped) & kept), \
            "Level %s both drops and ships %r" % (level_short, sorted(set(dropped) & kept))


def test_no_two_rungs_of_an_overridden_level_share_a_cylinder():
    """I2 is run BEFORE the overrides in the pipeline, so an override is the one way a rung could be
    moved inside a neighbour's 16 m reach cylinder after the guard has passed. `apply_overrides`
    re-checks it and reverts; this is the same property asserted on what ships.
    (`test_no_two_rungs_share_a_reach_cylinder` covers every level, including these -- this row exists
    so the failure names the override as the suspect.)"""
    for level_short in overrides():
        doc = shipped_docs().get("Level %s" % level_short)
        assert doc is not None
        for a, b in itertools.combinations(doc["rungs"], 2):
            assert dist3(a["pos"], b["pos"]) >= SEP, \
                "Level %s: %r and %r are %.1f m apart after the overrides" \
                % (level_short, a["name"], b["name"], dist3(a["pos"], b["pos"]))


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
