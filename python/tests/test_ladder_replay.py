"""The proof obligations of docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md, replayed
against recorded game data:  python tests/test_ladder_replay.py  (or pytest). No game needed.

A6, Level 0-1 must not change. A7, Level 0-3 must stop locking onto the door it cannot walk to.

The fixtures under tests/fixtures/ are trimmed probe logs (positions rounded to 0.1 m, plus
`arena_enemies_alive`, `kills` and `style`, one line per episode, gzipped) and `ladder_golden.json`, which was
produced by the PRE-patience `GateProgress` over exactly those rounded positions, so A6.1 is a regression pin
against the old implementation rather than against the new one talking to itself.

  level_0-1_run.jsonl.gz   32,022 decisions of campaign_ppo_ground/latest.zip playing Level 0-1, 7 episodes.
                           Three of them walk the ladder to hops 3; one holds an arena for 2,543 decisions.
  level_0-3_probe.jsonl.gz two episodes of the live 0-3 probe, the sampled policy that produced the bug report.
  level_0-3_scripted.jsonl.gz the scripted crossing of 0-3's first room, which walks through the hops 2 door.
  level_0-1_gates.json     0-1's eleven gates, verbatim from the gates design spec's literal example (S3.6).
  level_0-3_gates.json     0-3's twelve gates, verbatim from a fresh load's campaign block.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import GateProgress  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PATIENCE = 300  # the default 20 game seconds at 30 fps / frameskip 2
PATIENT = {"patience_steps": PATIENCE, "unpark_m": 2.0, "fallback_hysteresis_m": 10.0}
UNREACHABLE = "-16,73,315"  # 0-3's hops 1 door, 66 m straight up from the pit the agent locks under
ROUTE_0_3 = {"0,13,362", "0,53,330", "0,13,402", "-20,8,413", "-86,-12,413", "76,53,398"}


def gates(level: str) -> dict:
    return json.loads((FIXTURES / f"level_{level}_gates.json").read_text(encoding="utf-8"))


def episodes(name: str) -> dict[str, list[list[float]]]:
    """{episode: [[x, y, z, arena_enemies_alive, kills, style], ...]}, in recorded order."""
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {str(r["ep"]): r["steps"] for r in rows}


def golden() -> dict:
    return json.loads((FIXTURES / "ladder_golden.json").read_text(encoding="utf-8"))


def replay(camp: dict, steps: list[list[float]], **kw) -> dict:
    """One episode through GateProgress, the way UltrakillEnv drives it, with the recorded arena and fight state."""
    camp = json.loads(json.dumps(camp))  # a private copy: arena_enemies_alive is written per step
    progress = GateProgress(**kw)
    start = steps[0][:3]
    progress.new_level_load(camp, start)
    progress.reset_episode()
    progress.retarget(camp, start)
    targets: list[list] = []
    paid, approach = 0, 0.0
    parked_while_arena = unparks = 0
    was_parked: set[str] = set()
    park_log: list[tuple] = []
    for i, row in enumerate(steps):
        camp["arena_enemies_alive"] = row[3]
        fought = i > 0 and (row[4] > steps[i - 1][4] or row[5] > steps[i - 1][5])
        before = progress.parks
        gained, closer = progress.update(camp, row[:3], fought=fought)
        paid += gained
        approach += closer
        if progress.parks > before and row[3] > 0:
            parked_while_arena += 1
        for k in sorted(progress.parked - was_parked):
            park_log.append(("park", k, i, round(progress.park_best[k], 2)))
        for k in sorted(was_parked - progress.parked):
            park_log.append(("unpark", k, i, round(_dist(row[:3], camp, k), 2)))
        unparks += len(was_parked - progress.parked)
        was_parked = set(progress.parked)
        key = progress.target and progress.target.get("key")
        if targets and targets[-1][0] == key:
            targets[-1][1] += 1
        else:
            targets.append([key, 1])
    return {"targets": targets, "gate_paid": paid, "gate_approach": round(approach, 6),
            "best_hops": progress.best_hops, "reached": sorted(progress.reached),
            "parks": progress.parks, "parked_while_arena": parked_while_arena,
            "parked_keys": sorted(progress.parked), "unparks": unparks, "park_log": park_log}


def _dist(pos, camp: dict, key: str) -> float:
    gate = next(g for g in camp["gates"] if g["key"] == key)
    return math.dist(pos, gate["pos"])


def held(result: dict, key: str) -> int:
    """Decisions this episode spent pointing at `key`."""
    return sum(n for k, n in result["targets"] if k == key)


def comparable(result: dict) -> dict:
    return {k: result[k] for k in ("targets", "gate_paid", "gate_approach", "best_hops", "reached")}


# ---------------------------------------------------------------------------
# A6: Level 0-1 is inert
# ---------------------------------------------------------------------------

def test_a6_1_patience_off_reproduces_the_old_implementation_exactly():
    """Every recorded episode of all three fixtures, against the golden file the OLD tracker wrote.

    This is what pins that nothing else moved: the ladder, the reach cylinder, the approach accounting, the
    hand-over to the exit and the `gates_ordered` guards are all byte for byte what they were.
    """
    want = golden()
    for tag, level, name in (("level_0-1_run", "0-1", "level_0-1_run.jsonl.gz"),
                             ("level_0-3_probe", "0-3", "level_0-3_probe.jsonl.gz"),
                             ("level_0-3_scripted", "0-3", "level_0-3_scripted.jsonl.gz")):
        camp, eps = gates(level), episodes(name)
        assert set(eps) == set(want[tag]), tag
        for ep, steps in eps.items():
            assert comparable(replay(camp, steps)) == want[tag][ep], f"{tag} ep {ep}"


def test_a6_2_a_monotone_ladder_walked_inside_the_window_is_untouched_by_patience():
    """0-1's own eleven gates, walked hops 9 -> 0 and out of the pit, every target reached inside the window.

    The spec's guarantee in its exact form: where every target is reached before the patience runs out, the
    patient tracker and the pre-patience one agree on the target sequence, on every `gate` instalment and on
    `gate_approach` to the last decimal.
    """
    camp = gates("0-1")
    route = [(39.7, -0.5, 343.7)] + [tuple(g["pos"]) for g in sorted(camp["gates"], key=lambda g: -g["hops"])]
    route.append(tuple(camp["exit"]["pos"]))
    steps = [list(p) + [0, 0, 0] for p in walk(route, 1.0)]
    assert len(steps) > 2 * PATIENCE, "the walk has to be long enough for patience to have had a chance"
    off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
    assert max(n for _, n in off["targets"]) < PATIENCE, "the precondition: every target reached inside the window"
    assert on["parks"] == 0
    assert comparable(on) == comparable(off)
    assert off["best_hops"] == 0 and off["gate_paid"] == 10, off["gate_paid"]
    assert off["targets"][-1][0] == "exit"


def test_a6_3_an_arena_holding_a_door_shut_never_parks_it_on_the_recorded_run():
    """The measured reason the clock is suspended: four recorded 0-1 stalls, up to 2,543 decisions, ran with an
    ActivateArena wave alive on 96-100% of their steps, in front of the correct gate."""
    camp, eps = gates("0-1"), episodes("level_0-1_run.jsonl.gz")
    arena_steps = sum(1 for steps in eps.values() for row in steps if row[3] > 0)
    assert arena_steps > 3000, "the fixture has to contain the arena stalls this rule exists for"
    for ep, steps in eps.items():
        assert replay(camp, steps, **PATIENT)["parked_while_arena"] == 0, ep


def test_a6_4_the_recorded_run_keeps_its_payments_and_its_ladder():
    """Six of the seven recorded episodes are byte-identical. The seventh is measured here in full rather than
    described, because `gate_approach` IS a payment (0.15 per metre) and it is the one number that moves.

    The honest statement of A6 on recorded data, and the reason the guarantee in `test_a6_2` is written as a
    conditional: the patience window EXPIRES 46 times across these 32,022 decisions -- the agent stalls, dies
    and wanders, and 0-1's ladder being monotone does not stop that. 45 of the 46 are held by
    `_nearer_unreached`, which keeps a LADDER pick when there is nothing nearer to try. One becomes a real park,
    on episode 5, and what it does is hand over to `66,21,640` 78 decisions early -- the same gate the ladder
    was going to pick next, in the same order -- for +10.363 m of approach and not one extra instalment.
    """
    camp, eps = gates("0-1"), episodes("level_0-1_run.jsonl.gz")
    identical, parks, deltas = 0, 0, {}
    for ep, steps in eps.items():
        off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
        identical += comparable(off) == comparable(on)
        parks += on["parks"]
        deltas[ep] = round(on["gate_approach"] - off["gate_approach"], 3)
        assert on["gate_paid"] == off["gate_paid"], f"ep {ep}: no extra `gate` instalment on a monotone level"
        assert on["reached"] == off["reached"] and on["best_hops"] == off["best_hops"], ep
        assert [k for k, _ in on["targets"]] == [k for k, _ in off["targets"]], \
            f"ep {ep}: the same targets in the same order; at most a boundary between two of them moves"
    assert identical == 6 and parks == 1, (identical, parks)
    assert deltas == {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 10.363, "6": 0.0}, deltas


def test_a6_5_the_one_park_on_0_1_only_moves_a_hand_over_earlier():
    """What that single park costs, to the decision: episode 5's fifth and sixth targets swap 78 decisions."""
    camp, steps = gates("0-1"), episodes("level_0-1_run.jsonl.gz")["5"]
    off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
    assert [n for _, n in off["targets"]] == [134, 3034, 42, 181, 3183, 1092, 743, 592]
    assert [n for _, n in on["targets"]] == [134, 3034, 42, 181, 3105, 1170, 743, 592]
    assert sum(n for _, n in on["targets"]) == sum(n for _, n in off["targets"]) == len(steps)
    assert on["gate_paid"] == off["gate_paid"] == 7, "the extra 78 decisions on `66,21,640` pay no instalment"


# ---------------------------------------------------------------------------
# A7: Level 0-3 stops locking onto the door it cannot walk to
# ---------------------------------------------------------------------------

def test_a7_1_the_sampled_policy_stops_targeting_the_unreachable_door():
    """A7 on the recorded probe: the unreachable door goes from 82-94% of the episode's decisions to 40% and
    34%, and every time it comes back the agent has got strictly closer to it than ever before.

    Two things in that number are A1 and A3 working as specified rather than slack, and neither is reachable by
    tuning. The FIRST lock (380 and 841 decisions) runs before any park because the clock is reset by every
    genuine new best, and the sampled policy does stumble closer to that door on its way round the pit. Episode
    0 then un-parks three times, at 38.4, 29.2 and 17.6 m against a baseline that started at 40.9 -- a monotone
    24 m closing, which A3's own wording ("the agent has climbed toward the high door by another way") makes an
    un-park. The doubling bar is what makes the sequence converge: those cost 2, 4 and 8 m, and a fourth park
    would need the agent to stand within 2 m of the door.
    """
    camp, eps = gates("0-3"), episodes("level_0-3_probe.jsonl.gz")
    for ep, steps in eps.items():
        off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
        assert held(off, UNREACHABLE) > 2000, f"ep {ep}: the bug, as recorded"
        first = [k for k, _ in on["targets"]]
        assert first[1] == UNREACHABLE and on["targets"][1][1] <= 3 * PATIENCE, \
            f"ep {ep}: it is dropped within a few patience windows, not held for the episode"
        assert held(on, UNREACHABLE) <= 0.5 * held(off, UNREACHABLE), \
            f"ep {ep}: {held(on, UNREACHABLE)} of {len(steps)}, against {held(off, UNREACHABLE)}"
        assert ROUTE_0_3 & set(first), f"ep {ep}: the target moves to a gate on the walkable route"
        assert on["gate_paid"] >= off["gate_paid"], ep
        assert on["parks"] >= 1
        # Every re-entry is earned: each park of that door records a strictly closer baseline than the last.
        bests = [b for kind, key, _, b in on["park_log"] if kind == "park" and key == UNREACHABLE]
        assert bests == sorted(bests, reverse=True) and len(set(bests)) == len(bests), (ep, bests)
        assert on["unparks"] <= 3, f"ep {ep}: {on['unparks']} un-parks"


def test_a7_1b_the_probe_episodes_park_exactly_where_measured():
    """The two recorded episodes to the decision, so a later change to the park rules cannot move them unseen."""
    camp, eps = gates("0-3"), episodes("level_0-3_probe.jsonl.gz")
    zero = replay(camp, eps["0"], **PATIENT)
    # (step, baseline the bar is measured from) at each park, and (step, distance) at each un-park. The bars
    # those un-parks had to beat were 2, 4 and 8 m against 40.87, 35.98 and 26.13.
    assert [(s, b) for kind, key, s, b in zero["park_log"] if key == UNREACHABLE and kind == "park"] \
        == [(516, 40.87), (1159, 35.98), (1925, 26.13)]
    assert [(s, b) for kind, key, s, b in zero["park_log"] if key == UNREACHABLE and kind == "unpark"] \
        == [(854, 38.16), (1624, 29.16), (2168, 17.41)]
    assert held(zero, UNREACHABLE) == 1006 and zero["unparks"] == 3
    one = replay(camp, eps["1"], **PATIENT)
    assert [(s, b) for kind, key, s, b in one["park_log"] if key == UNREACHABLE] == [(991, 29.2)]
    assert held(one, UNREACHABLE) == 841 and one["unparks"] == 0
    assert UNREACHABLE in one["parked_keys"], "and in this one it never comes back at all"


def test_a7_2_the_scripted_crossing_gets_a_sensible_target_sequence():
    """What this fixture is matters for reading the result: it walks through 0-3's hops 2 door in 77 decisions
    and then mills about inside a 20 m box for the remaining 2,120, never approaching anything again.

    So the old tracker's answer -- one target for the whole run -- and the patient tracker's answer -- try each
    candidate for a window, give up, try the next -- are both "the agent is going nowhere". The second is the
    one that can escape: every gate it parks carries a 2 m un-park bar, so the moment the scripted path did walk
    toward one it would come straight back. The forward door it DID pass still pays its instalment.
    """
    camp = gates("0-3")
    steps = episodes("level_0-3_scripted.jsonl.gz")["0"]
    off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
    assert [k for k, _ in off["targets"]] == ["0,13,330", UNREACHABLE], off["targets"]
    assert off["targets"][1][1] == 2120, "the old tracker spends the whole run pointing through the ceiling"
    seq = [k for k, _ in on["targets"]]
    assert seq[0] == "0,13,330" and on["targets"][0][1] == 77, "the door it actually walks through"
    assert seq[1] == UNREACHABLE and on["targets"][1][1] <= 2 * PATIENCE, "held for a window, not for the run"
    assert UNREACHABLE not in seq[2:], "and never targeted again: the park sticks"
    assert len(ROUTE_0_3 & set(seq[2:])) >= 3, f"the rest is spent on gates the route uses: {seq[2:]}"
    assert max(n for _, n in on["targets"][1:]) <= 2 * PATIENCE, "no candidate is held longer than the window"
    assert on["gate_paid"] == off["gate_paid"] == 1, "the one gate it passed, paid once, by either rule"
    assert on["reached"] == off["reached"] == ["0,13,330"]


def test_a7_3_the_forward_legs_of_the_real_route_pay_gate():
    """0-3's walkable route, hops 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 2 -> 1 -> 0, at the pit's own gate positions.

    The monotone rule pays 3 instalments for the whole level, all of them at the end; with patience the forward
    legs -- which run AWAY from the exit -- pay too, which is the only reason a policy could learn to walk them.
    """
    camp = gates("0-3")
    by_key = {g["key"]: tuple(g["pos"]) for g in camp["gates"]}
    route = [(0.0, 0.5, 253.0), by_key["0,13,330"]]
    tail = ["0,13,362", "0,13,402", "-20,8,413", "-86,-12,413", "76,53,398", "0,53,330",
            UNREACHABLE, "-82,93,315"]
    steps = [list(p) + [0, 0, 0] for p in walk(route, 1.0)]
    steps += [list(by_key["0,13,330"]) + [0, 0, 0]] * (PATIENCE + 20)  # the agent mills about below the door
    steps += [list(p) + [0, 0, 0] for p in walk([by_key["0,13,330"]] + [by_key[k] for k in tail]
                                                + [tuple(camp["exit"]["pos"])], 1.0)]
    off, on = replay(camp, steps), replay(camp, steps, **PATIENT)
    assert off["gate_paid"] == 3, off["gate_paid"]
    assert on["gate_paid"] > off["gate_paid"], (on["gate_paid"], off["gate_paid"])
    assert on["gate_approach"] > off["gate_approach"]
    assert on["best_hops"] == 0 and on["targets"][-1][0] == "exit", "and it still finishes down the ladder"


def test_a7_4_the_unreachable_door_comes_back_once_the_agent_is_actually_near_it():
    """A3 in the field: the park is not permanent, it is un-done by getting genuinely closer."""
    camp = gates("0-3")
    by_key = {g["key"]: tuple(g["pos"]) for g in camp["gates"]}
    steps = [list(by_key["0,13,330"]) + [0, 0, 0]] * (PATIENCE + 5)
    steps += [list(p) + [0, 0, 0] for p in walk([by_key["0,13,330"], by_key["0,53,330"], by_key[UNREACHABLE]], 1.0)]
    progress = GateProgress(**PATIENT)
    camp = json.loads(json.dumps(camp))
    progress.new_level_load(camp, steps[0][:3])
    progress.reset_episode()
    progress.retarget(camp, steps[0][:3])
    seen_parked = False
    for row in steps:
        progress.update(camp, row[:3])
        seen_parked = seen_parked or UNREACHABLE in progress.parked
    assert seen_parked and UNREACHABLE not in progress.parked, "parked while below it, back once climbed to"
    assert UNREACHABLE in progress.reached and progress.best_hops == 1


def walk(points, step_m: float) -> list[tuple[float, float, float]]:
    """A straight-line walk through `points` at `step_m` per decision, which is 0-3's own measured pace."""
    out = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        n = max(1, int(math.dist(a, b) / step_m))
        out.extend(tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3)) for i in range(1, n + 1))
    return out


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
