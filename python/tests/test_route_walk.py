"""Every SHIPPED room trunk, walked end to end by the real loader and the real `GateProgress`:
    python tests/test_route_walk.py    (or pytest). No game needed.

This is the wrap-up stage's own check of S1 x S2 -- the offline data and the Python that consumes it, run
against each other over all fourteen committed files (the twelve with no gate ladder, plus 0-3
and 4-3, whose ladder is usable but collapsed and whose trunk ships for `prefer_route_when_collapsed`). `tests/test_route_files.py` judges the FILES against the
spec's invariants and imports nothing that reads them; `tests/test_route_replay.py` (A0 of the spec's section
8) judges the TRACKER on a safety property -- over every visit order the level allows, the target must never
advance past a rung nobody has entered. Neither one walks a trunk.

So the property here is the liveness one, and it is the claim the whole route fallback rests on:

> loaded from the packaged `routes/` folder exactly as a training run loads it, with the campaign block a
> fallback level really reports (`gates_ordered: false`, no gates, a live exit), a player who walks the
> trunk in order gets EVERY rung as its target in turn, pays each one exactly one instalment, and is handed
> over to the exit after the last one.

The trajectory is continuous rather than a teleport per rung -- `step_m` metres at a time along each leg,
through the rungs' own positions and on to the pit -- because the two failure modes it is here to catch are
both invisible to a teleport: a leg whose reach cylinders overlap enough to mark two rungs in one decision
(invariant I2 is a 16 m separation and the cylinder is 8 m x 6 m, so overlap is geometrically possible), and a
target that flaps or parks somewhere along a leg rather than at its ends. It is still a SYNTHETIC path and it
proves nothing about whether a leg is walkable in game: the spec says plainly (section 2.8) that no offline
method can answer that, and section 12.4's recovery path -- set a level's `"rungs": []` -- exists for it.

Four controls keep the assertions from being vacuous:

  - the trunk walked BACKWARDS pays nothing, so "pays once per rung" is a fact about the order, not about
    touching rungs (`test_walking_the_trunk_backwards_pays_nothing`);
  - a second lap inside the same level load pays nothing, which is `gate`'s "once per level load" rule
    (`test_a_second_lap_in_the_same_level_load_pays_nothing`);
  - the same walk with the live patience setting (300 decisions) must park nothing and produce the identical
    target sequence, which is the config's claim that parking is inert on a healthy trunk
    (`test_the_live_patience_setting_parks_nothing_on_a_healthy_trunk`);
  - and `GateProgress(route=None)` over the same walk pays nothing at all and reports layer 3, which is what
    "the 21 levels with no file are untouched" means (`test_without_a_route_the_same_walk_is_todays_behaviour`).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    GATE_EXIT_KEY,
    ROUTE_DIR,
    ROUTE_SOURCE_NONE,
    ROUTE_SOURCE_ROOMS,
    GateProgress,
    _dist3,
    load_route,
)

# The live setting: `gate_target_patience_s` 20.0 at `fixed_fps` 30 / `frameskip` 2 is 300 decisions
# (env.py builds it as round(20.0 * 30 / 2)). Same numbers as tests/test_ladder_replay.py's PATIENT.
PATIENT = {"patience_steps": 300, "unpark_m": 2.0, "fallback_hysteresis_m": 10.0}
# Metres per decision along a leg. Well above `min_gain_m` 0.5, so every step of a straight approach banks
# `gate_approach` and restarts the patience clock, and well under the 8 m reach cylinder, so no step can jump
# a rung: a coarser walk could skip past one and would make "every rung becomes the target" untestable.
STEP_M = 4.0


def shipped() -> list[tuple[str, dict]]:
    """(scene, loaded document) for every committed route file, through the PACKAGED path.

    `load_route(scene)` with no directory is the call `UltrakillEnv.__init__` makes, so this exercises
    `ROUTE_DIR` resolution, the version/level checks, the `open`/`locked`/`active` normalisation and the
    per-instance deep copy -- the whole loader, not a fixture written into a temp folder.
    """
    out = []
    for path in sorted(ROUTE_DIR.glob("route_*.json")):
        scene = path.stem[len("route_"):].replace("_", " ")
        doc = load_route(scene)
        assert doc is not None, f"{path.name}: the shipped file did not survive its own loader"
        out.append((scene, doc))
    assert out, f"no route files under {ROUTE_DIR}"
    return out


def chain(route: dict) -> list[dict]:
    """The trunk from the spawn end to the exit end: hops n-1 down to 0."""
    return sorted(route["rungs"], key=lambda r: -int(r["hops"]))


def block(route: dict, *, gates_ordered: bool = False) -> dict:
    """The campaign block a fallback level reports: no usable gate ladder, and the pit the trunk was built
    against (so guard I5 passes and `_exit` has something to hand over to)."""
    return {"gates_ordered": gates_ordered, "gates": [],
            "exit": {"pos": list(route["exit_pos"]), "active": False},
            "arena_enemies_alive": 0}


def trajectory(points, step_m: float = STEP_M) -> list[list[float]]:
    """A continuous path through `points`, in steps of at most `step_m`, landing exactly on each one."""
    path = [list(points[0])]
    for nxt in points[1:]:
        here = path[-1]
        span = _dist3(here, nxt)
        steps = max(1, int(math.ceil(span / step_m)))
        for i in range(1, steps + 1):
            t = i / steps
            path.append([here[a] + (nxt[a] - here[a]) * t for a in range(3)])
    return path


def walk(route: dict, points, *, patience: dict | None = None, progress: GateProgress | None = None,
         campaign: dict | None = None, ground: tuple | None = None, **kwargs) -> dict:
    """Drives `GateProgress` along `points` the way `UltrakillEnv` drives it: retarget, then update per step.

    Pass `progress` to continue an existing level load (the second-lap control); otherwise a fresh tracker
    starts a level load at the first point, which is what `env._campaign_reset` does on a fresh level.
    """
    camp = campaign if campaign is not None else block(route)
    path = trajectory(points)
    if progress is None:
        progress = GateProgress(route=route, **(patience or {}), **kwargs)
        progress.new_level_load(camp, list(path[0]))
        progress.reset_episode()
        progress.retarget(camp, list(path[0]))
    first = progress.target
    steps = []
    for pos in path[1:]:
        paid, approach = progress.update(camp, pos, ground=ground)
        steps.append({"pos": pos, "paid": paid, "approach": approach,
                      "target": None if progress.target is None else str(progress.target.get("key"))})
    # Every distinct target in the order it was held, starting with the one chosen before the first step.
    order, last = [], None
    for key in [None if first is None else str(first.get("key"))] + [s["target"] for s in steps]:
        if key is not None and key != last:
            order.append(key)
        last = key
    return {"progress": progress, "steps": steps, "targets": order,
            "paid": sum(s["paid"] for s in steps), "approach": sum(s["approach"] for s in steps),
            "walked": sum(_dist3(path[i], path[i + 1]) for i in range(len(path) - 1))}


def full_walk(route: dict, **kwargs) -> dict:
    """The whole trunk from its first rung to the pit."""
    rungs = chain(route)
    return walk(route, [r["pos"] for r in rungs] + [route["exit_pos"]], **kwargs)


# ---------------------------------------------------------------------------
# The walk itself
# ---------------------------------------------------------------------------

def test_every_shipped_trunk_hands_out_every_rung_in_order_then_the_exit():
    """The headline. Each rung below the one the player starts at becomes the target in turn, and after the
    last rung the exit does -- which is the handover `_choose_target` only makes once `best_hops` is 0, i.e.
    once the hops-0 rung has been physically stood in.

    The first rung never appears: `new_level_load` absorbs the rung the player is standing in (`mark_paid`),
    so the trunk aims at the rung BELOW from the first decision. That is the same absorb the seeding rule of
    section 6 performs for a player who spawns merely NEAR a rung, and it is why the ladder is worth
    `len(rungs) - 1` instalments rather than `len(rungs)`.
    """
    for scene, route in shipped():
        rungs = chain(route)
        result = full_walk(route)
        want = [str(r["key"]) for r in rungs[1:]] + [GATE_EXIT_KEY]
        assert result["targets"] == want, (
            f"{scene}: target order was {result['targets']}, expected {want}")
        assert result["steps"][-1]["target"] == GATE_EXIT_KEY, f"{scene}: the walk must end aimed at the pit"


def test_every_rung_pays_exactly_one_instalment_and_no_step_pays_two():
    """`gate` is once per new lower `hops` per level load, and a walk of the trunk collects the trunk's depth.

    The per-step bound is the one that matters: two rungs paid in one decision is invariant I2 failing (the
    reach cylinders overlapping), and it is the shape of the signal death the spec's section 2.6 measured on
    revision 1's branching ladders.
    """
    for scene, route in shipped():
        result = full_walk(route)
        depth = len(route["rungs"]) - 1
        assert result["paid"] == depth, f"{scene}: walked the trunk for {result['paid']} instalments, want {depth}"
        for step in result["steps"]:
            assert step["paid"] <= 1, f"{scene}: one decision paid {step['paid']} instalments at {step['pos']}"
        progress = result["progress"]
        assert progress.best_hops == 0 and progress.paid_hops == 0, f"{scene}: the trunk did not reach hops 0"
        assert progress.gates_reached == len(route["rungs"]), \
            f"{scene}: {progress.gates_reached} of {len(route['rungs'])} rungs were ever stood in"


def test_the_source_is_the_room_trunk_and_the_file_was_really_read():
    """Layer selection, from both sides: the reported layer and a counter on the loader itself.

    `route_source` is what every recovery path in the spec's section 12 reads, and `_hops_source` is what the
    guard against a mid-load layer flip keys on. `route_reads` proves the trunk was the ladder rather than
    merely present -- the same counter `test_route_replay.py` uses to prove the opposite on 0-1.
    """
    for scene, route in shipped():
        result = full_walk(route)
        progress = result["progress"]
        assert progress.route_source == ROUTE_SOURCE_ROOMS, f"{scene}: route_source {progress.route_source}"
        assert progress._hops_source == "rooms", f"{scene}: hop scale came from {progress._hops_source}"
        assert progress.route_reads > 0, f"{scene}: the trunk was never read"
        assert progress._route_ok is True, f"{scene}: guard I5 refused a file whose exit it was built against"


def test_the_approach_paid_is_the_distance_walked_and_no_more():
    """`gate_approach` pays metres of new best closeness, so a straight walk collects the path and nothing on
    top of it. This is the anti-farm bound: the ladder cannot pay more than the trunk is long.

    The walked path is slightly longer than the sum of the legs, because the target switches when the player
    ENTERS a rung's 8 m cylinder while the path carries on through the rung's own position; the approach
    banked for the next leg is measured from that switch point. So the two agree to a few metres per rung
    rather than exactly, and the assertion is a bound plus a 10% band.
    """
    for scene, route in shipped():
        result = full_walk(route)
        assert result["approach"] > 0.0, f"{scene}: a walk of the whole trunk paid no approach at all"
        assert result["approach"] <= result["walked"] + 1.0, \
            f"{scene}: paid {result['approach']:.1f} m of approach over {result['walked']:.1f} m walked"
        assert result["approach"] >= 0.90 * result["walked"], \
            f"{scene}: only {result['approach']:.1f} m of {result['walked']:.1f} m walked was ever paid"


def test_the_exit_leg_pays_approach_but_never_another_instalment():
    """After the hops-0 rung the trunk is done and the pit is the target: the last leg is the exit vector,
    which is the spec's section 10 `last rung -> pit` column and is unaided by design.

    The leg starts one step AFTER the handover, not at it. The decision that enters the hops-0 rung's
    cylinder is the decision that pays the trunk's last instalment AND the one whose `retarget` hands the
    target to the pit -- `_note_reached` runs before `_choose_target` inside `update` -- so counting from the
    handover step itself would attribute the ladder's own last payment to the exit leg.
    """
    for scene, route in shipped():
        result = full_walk(route)
        handover = next((i for i, s in enumerate(result["steps"]) if s["target"] == GATE_EXIT_KEY), None)
        assert handover is not None, f"{scene}: the exit never became the target"
        assert result["steps"][handover]["paid"] == 1, \
            f"{scene}: the handover step is the one that pays the hops-0 rung"
        tail = result["steps"][handover + 1:]
        assert tail, f"{scene}: nothing was walked after the handover"
        assert sum(s["paid"] for s in tail) == 0, f"{scene}: the exit leg paid a ladder instalment"
        assert sum(s["approach"] for s in tail) > 0.0, f"{scene}: the exit leg paid no approach"
        assert all(s["target"] == GATE_EXIT_KEY for s in tail), f"{scene}: the exit was handed back"


# ---------------------------------------------------------------------------
# The controls: what makes the assertions above non-vacuous
# ---------------------------------------------------------------------------

def test_walking_the_trunk_backwards_pays_nothing():
    """The ladder is a direction, not a set of places. Starting at the hops-0 rung absorbs it, `best_hops`
    is 0 from the first decision, the target is the pit, and walking back up the level pays no instalment
    however many rungs are stood in on the way.

    That is `mark_paid` semantics working as intended -- the player did not travel to the rung it started at
    -- and it is the reason the seeding rule of section 6 is worth having: without an absorb, a spawn near
    the wrong end of a total order aims the agent at the room it just left.
    """
    for scene, route in shipped():
        rungs = chain(route)
        result = walk(route, [r["pos"] for r in reversed(rungs)])
        assert result["paid"] == 0, f"{scene}: walking the trunk backwards paid {result['paid']} instalments"
        assert result["targets"] == [GATE_EXIT_KEY], f"{scene}: backwards targets {result['targets']}"
        assert result["progress"].gates_reached == len(rungs), f"{scene}: the walk did stand in every rung"


def test_a_second_lap_in_the_same_level_load_pays_nothing():
    """`gate` is once per LEVEL LOAD. Re-walking the trunk after a death that respawned at a checkpoint must
    pay no instalment for ground already covered, which is what keeps the ladder bounded by its own depth."""
    for scene, route in shipped():
        rungs = chain(route)
        points = [r["pos"] for r in rungs] + [route["exit_pos"]]
        first = walk(route, points)
        again = walk(route, points, progress=first["progress"])
        assert again["paid"] == 0, f"{scene}: a second lap paid {again['paid']} instalments"
        assert again["targets"] == [GATE_EXIT_KEY], f"{scene}: a second lap retargeted to {again['targets']}"


def test_the_live_patience_setting_parks_nothing_on_a_healthy_trunk():
    """`configs/campaign_gates_full.yaml`'s own claim, asserted: a monotone trunk's rung-below is also its
    nearest unreached rung, so `_nearer_unreached` keeps the ladder's pick and the patience rule is inert.

    This is why `targets_parked` 0 on a level reading `route_source` 2 is NOT evidence that the trunk is
    fine -- the dashboard note says so, and this is the measurement behind it. A walk that reaches every rung
    parks nothing; so would a walk that reaches none, and only the data can tell those apart.
    """
    for scene, route in shipped():
        plain = full_walk(route)
        patient = full_walk(route, patience=PATIENT)
        assert patient["progress"].parks == 0, f"{scene}: parked {patient['progress'].parks} times on a clean walk"
        assert patient["targets"] == plain["targets"], f"{scene}: patience changed the target order"
        assert patient["paid"] == plain["paid"], f"{scene}: patience changed the instalments paid"


def test_without_a_route_the_same_walk_is_todays_behaviour():
    """`GateProgress(route=None)` -- the 21 levels with no file, and any run with `route_fallback: false`.

    No ladder, no target, no payment: the level runs on the exit vector and exploration, exactly as every
    fallback level does today. The walk is identical, so the difference is the route document and nothing else.
    """
    for scene, route in shipped():
        rungs = chain(route)
        points = [r["pos"] for r in rungs] + [route["exit_pos"]]
        camp = block(route)
        progress = GateProgress(route=None, **PATIENT)
        progress.new_level_load(camp, list(points[0]))
        progress.reset_episode()
        progress.retarget(camp, list(points[0]))
        paid = 0
        for pos in trajectory(points)[1:]:
            got, _ = progress.update(camp, pos)
            paid += got
        assert paid == 0, f"{scene}: a tracker with no route paid {paid} instalments"
        assert progress.target is None, f"{scene}: a tracker with no route held {progress.target}"
        assert progress.route_source == ROUTE_SOURCE_NONE and progress.gates_reached == 0, f"{scene}"


def test_a_good_gate_ladder_beats_the_trunk_on_the_same_level():
    """Precedence, on the files themselves: layer 1 wins whenever it can, and the trunk is not even read.

    Twelve of the fourteen shipped levels have no gate ladder at all; 0-3 and 4-3 have one that is usable but
    COLLAPSED, and their file is read only when a run sets `prefer_route_when_collapsed` (default false, the
    lead's ruling). Either way the precedence below is the same and is a property of the CODE: with the flag at
    its default a level that HAS a usable ladder never reads its file, which is checked here by handing every
    route level a gate ladder it does not have.
    """
    for scene, route in shipped():
        rungs = chain(route)
        gates = [{"key": f"g{i}", "pos": list(r["pos"]), "hops": int(r["hops"]),
                  "open": False, "locked": False, "active": True} for i, r in enumerate(rungs)]
        camp = dict(block(route, gates_ordered=True), gates=gates)
        result = walk(route, [r["pos"] for r in rungs] + [route["exit_pos"]], campaign=camp)
        progress = result["progress"]
        assert progress.route_reads == 0, f"{scene}: the trunk was read on a level with a good gate ladder"
        assert progress._hops_source == "gates", f"{scene}: hop scale came from {progress._hops_source}"
        assert all(key == GATE_EXIT_KEY or key.startswith("g") for key in result["targets"]), \
            f"{scene}: a room rung was targeted over a gate: {result['targets']}"


# ---------------------------------------------------------------------------
# The ground rule: a room rung is credited only where the player could have landed
# ---------------------------------------------------------------------------
#
# `GateProgress._on_ground`, added 2026-09-18. A room rung is a CENTROID, and a centroid carries no promise
# that its 8 m x 6 m reach cylinder stays inside the room it names. On `Level 0-3` one did not: the
# `2 - Side Hallway` rung sat 1.5 m past the main room's far wall, so the cylinder covered the wall FACE and
# 680 of that rung's 801 live credit steps were the player hanging against the wall on the WRONG SIDE of it,
# 10-20 m above the floor below. The rung itself has been moved (`rung_overrides.json`), but the general
# defence is the rule these tests cover. Measured bound, from those same three episodes:
#
#     legitimate airborne credits, per rung  max 5.8 5.9 6.0 6.0 6.0 7.9  (six rungs)
#     against the wall face, first credit of each episode          9.5 10.5 10.5
#     scripted runs that never crossed the wall, first credit      8.8 9.1
#
# so ROUTE_GROUND_M 8.0 is the only round number above every legitimate reading and below every bad one.

GROUNDED = (True, 1.5)          # standing: the centre ray from the player's middle to the floor
AIRBORNE_OK = (False, 7.9)      # the largest legitimate airborne reading measured, on 0-3's bowl rung
AIRBORNE_BAD = (False, 9.5)     # the smallest wall-face reading measured
VOID = (False, 30.0)            # `ground_ray_length` exactly: the mod's "the ray hit nothing" sentinel


def test_a_grounded_walk_of_every_shipped_trunk_is_unchanged_by_the_ground_rule():
    """The control, and the one that matters most: a player who walks the trunk on the floor must get byte
    for byte what they got before the rule existed. Compared over all fourteen files, on the instalments, the
    approach metres and the whole target sequence."""
    for scene, route in shipped():
        before = full_walk(route)
        for name, reading in (("grounded", GROUNDED), ("a legal hop", AIRBORNE_OK)):
            after = full_walk(route, ground=reading)
            assert after["paid"] == before["paid"], f"{scene} ({name}): paid {after['paid']} vs {before['paid']}"
            assert abs(after["approach"] - before["approach"]) < 1e-9, f"{scene} ({name}): approach moved"
            assert after["targets"] == before["targets"], f"{scene} ({name}): the target sequence moved"


def test_an_airborne_walk_with_nothing_under_it_credits_no_rung():
    """The rule's whole point. The same trajectory, flown rather than walked, pays nothing and advances the
    ladder by nothing -- so a cylinder that overhangs a wall or a void cannot be cashed from the wrong side.

    The ladder does not start empty: `new_level_load` ABSORBS the rung the walk begins at, permissively and
    on purpose (see `test_a_respawn_still_absorbs_a_rung_from_the_air`). So the assertion is that the flight
    adds nothing to what the load already knew, which is `1` rung and the trunk's top hop count.
    """
    for scene, route in shipped():
        top = int(chain(route)[0]["hops"])
        for name, reading in (("the wall face", AIRBORNE_BAD), ("the void", VOID)):
            result = full_walk(route, ground=reading)
            assert result["paid"] == 0, f"{scene} ({name}): paid {result['paid']} instalments in the air"
            progress = result["progress"]
            assert progress.gates_reached == 1, \
                f"{scene} ({name}): reached {progress.gates_reached} rungs, only the seeded one was allowed"
            assert progress.best_hops == top, f"{scene} ({name}): best_hops {progress.best_hops}, not {top}"


def test_the_bound_is_the_measured_one():
    """8.0 m exactly: 7.9 (the largest legitimate airborne credit measured) is in, 8.1 is out. Pinned so the
    constant cannot drift away from the data without a test saying so."""
    route = dict(shipped())["Level 0-3"]
    assert full_walk(route, ground=(False, 8.0))["paid"] == full_walk(route)["paid"]
    assert full_walk(route, ground=(False, 7.9))["paid"] == full_walk(route)["paid"]
    assert full_walk(route, ground=(False, 8.1))["paid"] == 0


def test_the_grounded_flag_beats_the_ray():
    """`grounded` is the game's own controller state and short-circuits the ray. It has to: on 0-3's
    second-floor walkway the centre ray reads the full 30 m while the player is provably standing on it
    (12 of that rung's 30 live credit steps)."""
    route = dict(shipped())["Level 0-3"]
    assert full_walk(route, ground=(True, 30.0))["paid"] == full_walk(route)["paid"]


def test_a_mod_that_reports_no_ground_at_all_is_todays_behaviour():
    """Permissive when it cannot answer: an observation with neither the flag nor a ray must not silently
    stop paying a trunk. That is the pre-0.6.0 mod, and it is also every caller that passes no `ground`."""
    for scene, route in shipped():
        assert full_walk(route, ground=(None, None))["paid"] == full_walk(route)["paid"], scene


def test_route_ground_m_zero_switches_the_rule_off():
    for scene, route in shipped():
        flown = full_walk(route, ground=VOID, route_ground_m=0.0)
        assert flown["paid"] == full_walk(route)["paid"], scene


def test_a_gate_ladder_is_never_subject_to_the_ground_rule():
    """A door is a place you pass THROUGH and the mod's own graph says what that means; only a room centroid
    is a guess. So the same flown walk over a gate ladder must pay exactly what it always did -- this is the
    property that makes the change inert on the 18 levels layer 1 routes and on all of Cyber Grind."""
    for scene, route in shipped():
        rungs = chain(route)
        gates = [{"key": f"g{i}", "pos": list(r["pos"]), "hops": int(r["hops"]),
                  "open": False, "locked": False, "active": True} for i, r in enumerate(rungs)]
        camp = dict(block(route, gates_ordered=True), gates=gates)
        points = [r["pos"] for r in rungs] + [route["exit_pos"]]
        grounded = walk(route, points, campaign=camp)
        flown = walk(route, points, campaign=camp, ground=VOID)
        assert flown["paid"] == grounded["paid"], f"{scene}: gates paid {flown['paid']} vs {grounded['paid']}"
        assert flown["targets"] == grounded["targets"], f"{scene}: the gate target sequence moved"


def test_a_respawn_still_absorbs_a_rung_from_the_air():
    """ABSORBING must stay permissive, which is why `self._ground` is only set inside `update()`.

    `mark_paid` records what a respawn REVEALS so it can never be paid for again. If the ground rule reached
    it, a respawn that put the player anywhere the rule dislikes would leave the rung looking unreached, and
    walking two metres to it afterwards would pay an instalment for ground the episode never covered -- a
    farm, which is the exact opposite of what the rule is for.
    """
    route = dict(shipped())["Level 0-3"]
    rungs = chain(route)
    progress = GateProgress(route=route)
    camp = block(route)
    # A respawn that lands on the second rung, mid-air with nothing under it.
    progress.new_level_load(camp, list(rungs[1]["pos"]))
    assert str(rungs[1]["key"]) in progress.reached, "the respawn's own rung was not absorbed"
    assert progress.paid_hops == int(rungs[1]["hops"]) == progress.best_hops
    # ... and walking back onto it, grounded, pays nothing, because absorbing already recorded it.
    result = walk(route, [rungs[1]["pos"], rungs[1]["pos"]], progress=progress, ground=GROUNDED)
    assert result["paid"] == 0, "an absorbed rung was paid for after all"


def test_the_0_3_rung_the_rule_was_measured_on_is_where_the_override_put_it():
    """Belt and braces with `tests/test_route_files.py`: the rung move and the ground rule were measured
    together and neither one alone fixed 0-3 (the bound alone still credited from a ledge on the main-room
    side at 3-5 m; the move alone left every other centroid unguarded). If the data half is ever lost, this
    fails here too, next to the rule it partners."""
    route = dict(shipped())["Level 0-3"]
    hallway = next(r for r in route["rungs"] if r["name"].startswith("2 - Side Hallway"))
    assert hallway["pos"] == [0.0, 10.0, 340.0], hallway["pos"]
    assert hallway["key"] == "0,10,340", hallway["key"]


if __name__ == "__main__":
    routes = shipped()
    print(f"walking {len(routes)} shipped trunks at {STEP_M:.0f} m per decision")
    for scene, route in routes:
        result = full_walk(route)
        # `gate` 15.0 and `gate_approach` 0.15 are the shipped weights; this reproduces the spec's section 6
        # table from the tracker rather than from the polyline, which is the arithmetic that table asserts.
        print(f"  {scene:<12} {len(route['rungs']):>2} rungs  {len(result['steps']):>5} decisions  "
              f"paid {result['paid']:>2}  walked {result['walked']:>7.0f} m  "
              f"approach {result['approach']:>7.0f} m  "
              f"gate {result['paid'] * 15.0:>5.0f} + approach {result['approach'] * 0.15:>6.1f}")
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
