# Route fallback and boss levels: design

Date: 2026-09-17. **Revision 2**, rewritten against two independent reviews (`rl`, `data`); every
disputed fact was re-measured by the author of this revision before being accepted or rejected, and the
dispositions are in §13. Status: engineering contract. Mod version target **0.7.1**; protocol stays **1**
(obs gains one block, nothing removed or changed). Observation width stays **479**, the action space is
unchanged, and **no checkpoint is invalidated**.

Follows `2026-09-16-campaign-gates-unwedge-design.md` (the gates spec) and
`2026-09-17-multi-level-and-skull-gates-design.md`. Inputs: `docs/level-survey.md`, the shipped scene bundles
(re-parsed, not trusted from the proposals), and `F:\Github\ULTRAKILL-AI\decompiled` (read-only).
No human demo and no recorded route is used anywhere: every rung is derived from level data alone.

**Every number called *measured* below was produced or reproduced by this revision**, with the script named.
Scripts for revision 2 are in `…\scratchpad\route\spec-rev\` (`reach_verify.py`, `recompute.py`, `legs.py`,
`lockprobe.py`); revision 1's are in `…\scratchpad\route\judge\`.

---

## 1. The decision

**A three-layer route signal with a strict precedence rule, evaluated per level load, top to bottom.**

| # | layer | fires when | source | levels |
|---|---|---|---|---|
| 1 | **gates ladder** | `gates_ordered` and `hops`-carrying share of the phase-1 ladder ≥ `gate_hops_min_frac` (0.5) — **today's guard, unchanged** | the mod's live `campaign.gates` | **18** |
| 2 | **room trunk** | layer 1 fails **and** the level's shipped route file passes guards R1-R4 (§4.3) | offline per-level JSON, read by Python | **12** |
| 3 | **exit vector** | both fail | `campaign.exit` + exploration — today's behaviour | **3** (1-3, 5-4, 6-2) |

Layer 1 is byte for byte what ships today, because the layer-2 read happens only on the three
`return []` arms of `GateProgress._gates()`. Nothing on a gates level ever reads a route file, and
**revision 2 removes the one change that did reach a gates level** (the global boss stuck-clock clause,
§5.4), so the byte-for-byte claim is now true of every line in this spec rather than of most of them.

**Measured outcome: 18 gates + 12 rooms + 3 nothing, against today's 18 + 15 nothing.** Of the 12 gained
levels, **8 get a complete trunk from the first room to the exit** and **4 stop at a skull/altar lock**
partway (1-4, 4-4, 7-1, 7-2 — §4.5). Campaign-wide that is **26 of 33 levels routed spawn-to-exit, 4 routed
spawn-to-lock, 3 unrouted.** Full table in §10.

**What revision 1 got wrong, and why the count fell from 14 to 12.** Revision 1 shipped a *strict total
order over every room*. Both reviews independently found that `GateProgress` cannot express "both required":
`_note_reached` scans every rung each step and collapses `best_hops` to the minimum over all of them, so on a
branching level the branch the agent enters *second* permanently loses its signal. Re-run with the real
`GateProgress` (§2.6) this is not a rounding error — on 1-4, entering one side room first pays four
instalments at once and then points the observation at the V2 door while three required side rooms are still
unvisited. The fix is to ship only the **trunk**: the rooms every playthrough must pass, in order. That is
coarser and it costs 1-3 and 6-2, and it is never wrong.

**Rejected as the primary signal: the unified activation graph** (§2.3) and **the geometry geodesic** (§2.4).
Three pieces are adopted and are load-bearing: the activation graph's Tier C boss parse (§5), its
altar→door wiring (§4.5), and the geometry work's voxel standability probe, **promoted from an optional
reviewer's aid to hard guard R4** (§4.3) after it found five shipped rungs the agent could never reach.

---

## 2. Why — the measurements that decided it

### 2.1 The room numbering is real, and a room's transform is the doorway

Independent scan of the bundles (`judge/check_rooms.py`, no proposal code loaded): 0-1 has 15 numbered rooms
`1 … 13` with `A`/`B` suffixes plus the `12 - Stairway OLD` dead variant; 7-3 has 12, `1 - Dark Path` …
`12 - Grand Hall`. Rooms ahead of the player are `activeSelf == false` at load, as the survey says — though
not universally: on 0-1, 10 of the 11 rooms past `3 - Gun Room` are off, and `8 - Curved Hallway` is on.

The load-bearing claim tested against the gate keys **the mod actually publishes** (`judge/check_doorway.py`):

- **0-1: 11 of 11 gates have a numbered room within 10.0 m; median 3.0 m.** Every rung of the ladder that
  took 0-1 to a 50% fresh-start completion rate is reproduced by a room transform.
- Campaign-wide: **174 of 297 gates (59%) within 12 m; median of per-level medians 3.0 m.**

**Both figures are measured on levels that have gates, i.e. by construction not on the levels the fallback
fires on** — the `rl` review is right to call that out, and revision 1 leaned on it as if it were general.
The property that matters on a fallback level is not "a gate is near a room" (there are no gates) but "the
rung is somewhere the player can stand", which is now measured directly and guarded (§2.7, R4).

### 2.2 Rule-independent test: does the trunk pass through the level?

Checkpoints are authored progression markers and **neither ordering rule uses them**, so they are free ground
truth (`judge/check_cp.py`, 40 m, all 33 levels; `spec-rev/legs.py` for the leg measure):

| ladder | checkpoints within 40 m of a rung |
|---|---|
| today's gates, over the 18 levels it routes | 70/99 (71%) |
| room ladder, over all 33 levels it can emit | 107/142 (75%) |
| activation graph, one rung per hop tier | 49/142 (35%) |
| activation graph, all 64 shipped sub-goals | 109/142 (77%) |

Both figures come from `checkpoint-chain/routes`, i.e. the raw chain **before** I6/T/R4 — they are a verdict
on the *rule*, not on what ships. What ships is measured separately in §2.7 and in §10's `cp≤60` column.
(Revision 1 reported "50/62 (81%) on the 15 levels the fallback fires on"; that number was over a 15-level set
that included 5-4, which has never had a file. Restated here rather than reused.)

### 2.3 Why the activation graph is not the primary signal

Genuine contribution: 26 routed / 7 partial / 0 none against today's 14/4/15, reproduced exactly
(`coverage.py`, verdicts identical); its 0-1 validation reproduces all 11 gate hop values. But `GateProgress`
consumes a **ladder**:

- **The ladder is 4 rungs deep.** Median hop tiers per level 4 (max 10 on 0-1); on the fallback levels the
  whole route is 58 rungs — 3.9 per level. 8-3 gets 2 tiers, 8-4 gets 1.
- **The tiers are enormous, and `_choose_target` picks the nearest member of one.** Median largest tier 12,
  max 49 (8-1). **27 of the 58 fallback rungs have ≥ 10 alternatives, median 7, max 98.** "Walk toward the
  nearest of 98 co-tier objects" flaps as the player moves and seeds a fresh `best_dist` key each time.
- **5 of 58 fallback rungs are enemy-death edges** whose position is the enemy's *authored spawn*, and 11 are
  `kind: boss|fight`. Named rungs include `Trasher`, `Gutterman`, `SuicideTree4`, `Light` and `Body`.
- **It compresses the gates ladder where gates are good** (5-3 13 tiers → 5, 4-1 8 → 5, Spearman 0.38).

Its `requires` wiring is adopted (§4.5); its ordering is not.

### 2.4 Why the geometry geodesic is not the primary signal, and what it is kept for

Reproduced exactly (`evaluate.py 0-1 0-5`). Its author already measured the fine geodesic dead (median 56% of
walkable surface reaches the exit; 0-1 only closes at a physically impossible 20 m jump). The coarse
room-adjacency layer is also unfit as primary: **`partial` on 0-1** (13/25 rooms, `FirstRoom` not connected to
the exit room), **loses 2-1 and 5-3** which gates handle today, room AABBs polluted (0-1's `7 - Fan Room` box
spans 536 m of height), precision 44% / recall 34%, 55 min to produce.

**But it produced the best independent corroboration in the exercise**, and revision 2 promotes its voxel
work from a footnote to a guard. Its 0-1 hop assignment, derived from collision voxels with no name used for
ordering, is monotone in the room ordinal: `4 → 5 → 6 → 7 → 8/8-9 → 9 → 10/11 → 12 → 12B → 13`. Its
`ukgeom` + `voxel` standability pass is what R4 (§4.3) now runs.

### 2.5 Triangulation

`judge/triangulate.py` matches rungs between ladders by nearest pair within 30 m, Kendall tau. Activation
graph vs room ladder: 20 levels comparable, **17 with tau ≥ 0.6** (mostly +1.00), 3 negative, all three on
≤ 5 matched pairs. Room ladder vs today's gates: +1.00 on 0-1, 0-2, 0-4, 2-1, 2-2, 2-3, 3-1, 3-2, 4-1, 4-3,
5-1, 6-1. The hops-0 rung sits **3 m** from today's hops-0 gate on 0-1, 0-2, 0-3, 0-4 and 0 m on 1-2, 3-2,
4-3, 6-1.

### 2.6 The measurement that changed the design: a total order cannot express a branch

`GateProgress` carries one `best_hops`. `_note_reached` scans **every** rung each step and takes the minimum
hop value of any rung the player has stood in; `_choose_target` then targets `max({h : h < best_hops})`. So
touching any rung skips every rung above it, whether or not the level required those rooms.

Run with the **real** `GateProgress` against the shipped ladders, with §4.4's `_rooms` and §6's seeding
implemented exactly as revision 1 stated them (`…\scratchpad\route\rev\sim.py`, re-run here):

| level | order | instalments | what happens |
|---|---|---|---|
| 1-4 | alphabetical (the emitter's) | 8 | one per rung, as designed |
| 1-4 | player enters `3TR - Shelf Room` first | 8 | **reaching hops 1 pays 4 at once**; hops 2, 3, 4 then pay **0, 0, 0**; target jumps to `V2 - Arena` with three required blue-skull rooms unvisited |
| 1-3 | Blue wing first | 6 | reaching hops 2 pays 4 at once; the whole Red wing (5, 4, 3) pays 0 and is never targeted |
| 8-3 | R branch first | 23 | reaching hops 10 pays 7 at once; the entire B branch (16…11) pays 0 while the target sits frozen |

Total instalments are identical in both orders (8/8, 6/6, 23/23), so this is **not** extra reward. It is the
signal going dead on the half of the level the agent has not done — the brief's "a wrong route is worse than
none", and the observation vector (slots 448-451) pointing at a door that will not open.

The emitter's own docstring says why: the order is "scope order, then ordinal, then suffix … The wired edges
are deliberately NOT what produces the order." Alphabetical suffix order is a fabrication on a star.

**Therefore the ladder ships only the trunk** (invariant T, §4.3): a parallel set — same-ordinal suffixed
rooms, or a multi-prefix branch region — is collapsed onto the rung it hangs off. Coarser, never wrong.

### 2.7 The measurement that added a guard: five shipped rungs were unreachable

`_note_reached` is the only thing that advances `best_hops`, and it needs the player inside `_is_reached`'s
cylinder: **8 m horizontal, 6 m vertical** (`margin` 1.0, because a room rung ships `open: false`). A room's
transform is a prefab pivot, not a doorway the player walks through, so this has to be measured.

`spec-rev/reach_verify.py` voxelises the collision geometry within ±14 m of each rung (geometry-geodesic's own
`ukgeom` + `voxel`), extracts standable cells and reports the nearest one in the cylinder metric. Over all 14
revision-1 files:

| level | rung | verdict |
|---|---|---|
| 5-2 | `8 - Ship`, hops **0** | **no standable cell within ±40 m**; nearest collider of any kind 77.4 m, the hull 83 m above the pivot |
| 4-2 | `5 - Temple Entrance`, hops 4 | nearest standable cell 19.8 m horizontally |
| 7-4 | `0 - Leg Checkpoint`, hops 6 | **zero triangles within ±40 m** |
| 7-4 | `1 - Front Checkpoint`, hops 5 | **zero triangles within ±40 m** |
| 7-4 | `3B - Secret Entrance Checkpoint`, hops 2 | nearest standable cell 11.5 m > 8 m |

**Control** (`review-data/r03_control.py`, re-run): the same probe on 60 gates from ladders that demonstrably
work in game — 0-1 11/11, 0-3 12/12, 4-3 3/3, 6-1 6/6, 8-1 28/28 — reports **0 unreachable, dxz 0.4 m on
every one**. 111 of the 116 room rungs also sit on standable floor at dxz ≤ 2 m. So the verdict is about those
five rungs, not about the probe, and the pivot convention itself is sound.

5-2 was the serious one: with its hops-**0** rung unreachable, `best_hops` can never become 0,
`_choose_target` never returns `_exit(campaign)`, and **the pit is never the target on 5-2 for the whole
run**, while `gate_approach` keeps paying toward the unreachable pivot and `best_dist` is re-seeded every
episode. That is a stable non-progress income stream of the same shape as the void-novelty faucet this project
already paid for once.

R4 drops those five and renumbers. 5-2 8→7 rungs, which also **fixes** the dead end: `7B - Crooked Cabin`
becomes the lowest rung, so reaching it sets `best_hops` 0 and the exit becomes the target. 4-2 9→8 before
the trunk collapse, 7-4 7→5.

### 2.8 Connectivity, measured

The brief requires connectivity be measured; revision 1 measured rung counts and gaps instead. Three measures,
all on **what ships after every filter** (`spec-rev/reach_verify.py`, `legs.py`, `recompute.py`):

- **Per-rung standability: 92/92 rungs pass R4** — by construction, since R4 is now a guard.
- **Leg witness: 57 of 92 legs (62%)** have an authored checkpoint within 40 m of the segment between their
  two rungs. Per level in §10. 8-4 scores 0/4 (it has one checkpoint) and 2-4 1/4 (two checkpoints across
  400 m rooms); 5-2 6/7, 8-3 11/13, 7-3 8/12.
- **Ordering: tour ratio ≤ 1.35 on all 12** (max 1.255 on 7-2), plus §2.5's triangulation.

**What is *not* measured, and cannot be offline: whether a leg is traversable in that direction.** The fine
geodesic was measured dead (§2.4) and a straight knee-height segment test is `partial` on 0-1 itself, so no
offline method can answer it. The honest statement is that the trunk is ordered, standable and corroborated,
not that each leg is walkable. §12.4's recovery path exists for exactly this.

**One connectivity claim revision 1 could not make and revision 2 still cannot: "from the spawn".** The
offline parse has no per-level spawn — `raw/*.detail.json`'s `spawn` field reads `[0, 105, 253]` on **every
one of the 33 levels**, i.e. it is the player prefab's authored transform, not the level's spawn point. The
first rung is the lowest ordinal of the first numbering scope, which is the author's own start marker, and
§6's seeding rule absorbs whatever rung the player actually starts near. That is the argument; it is not a
measurement, and `route_seed_m` is unvalidated against real spawns on 11 of the 12 levels.

---

## 3. Where the data lives, and the mod/Python split

**The route files are read by Python, not by the mod.** This is the largest single cost decision:

- A room rung needs **no live state** in the base design: it is always `open: false, locked: false,
  active: true`. Routing it through the mod buys nothing and costs a DLL rebuild, which needs the game closed
  while a run is live. (The one place live state *is* needed — the altar stamp, §4.5 — reads a block the mod
  already publishes, so it still needs no mod change.)
- The guards are whole-level statistics (trunk structure, tour ratio, voxel standability) the mod cannot
  compute per frame and would have to be told anyway.
- A per-level file can be hand-corrected (`"rungs": []` disables one level) without a rebuild.
- It is testable end to end with no game and no mod: `test_campaign_env.py` can hand `FakeLevel` a route file.

**Therefore the route fallback needs zero mod work.** The mod work in this spec is confined to the Tier C
boss block (§5), which is genuinely new live state, and which revision 2 moves off the critical path entirely.

Data lives at `python/ultrakill_ai/routes/route_<scene>.json`, committed: **12 files**, ~14 KB total.
Levels with no file fall through to layer 3.

---

## 4. The data-file interface

### 4.1 Schema

```
{
  "level":       "Level 0-5",            # must equal the scene name; a mismatch rejects the file
  "version":     2,                       # bumped by revision 2; Python refuses an unknown version
  "source":      "room-trunk offline v2",
  "exit":        {"pos": [x,y,z],         # the FinalPit this trunk was built against: the staleness check
                  "target": "Level 1-1"},
  "start_room":  "1 - Opening Hallway",   # diagnostic only
  "trunk_collapsed": [],                  # diagnostic: the parallel sets invariant T removed, by name
  "tour_ratio":        1.0,               # guard R2, computed on the rungs THAT SHIP (post-I3)
  "checkpoints_within_60m": "1/1",        # diagnostic only, §2.2
  "legs_witnessed":        "3/5",         # diagnostic: connectivity, §2.8
  "last_rung_to_exit_m":   211.1,         # diagnostic only; the last leg is always the exit vector
  "rungs": [ {"key": "0,-10,300",         # CampaignPatches.Key() form: position rounded to whole metres
              "pos": [0.0,-10.0,300.0],
              "hops": 4,                  # rooms still to traverse; 0 is the last room before the pit
              "name": "1 - Opening Hallway",
              "gated_by": [],             # §4.5, stage S3: altar positions that must be filled first
              "open": false, "locked": false, "active": true}, ... ]
}
```

Every field `GateProgress._choose_target`, `_is_reached`, `_subgoal` and `spaces.campaign_block` read is
present with the same meaning it has on a gate. `name`, `gated_by` and the diagnostics are extra; `gated_by`
is ignored by every consumer until S3 ships.

`open` is **false**, not true. `_is_reached` doubles its reach cylinder when `open` is true (16 m × 12 m
instead of 8 m × 6 m), which collides rungs (§4.3 I2) and would silently widen R4's own criterion, and `open`
is observation slot 453 — shipping `true` would pin a learned input column to 1.0 on every fallback level.
`false` matches the exit sentinel's own convention in `GateProgress._exit`.

**Observation saturation, stated rather than fixed.** `spaces.campaign_block` packs a target with a non-null
`hops` at rel/50 and dist/100, clipped per axis to ±4.0 — so a leg beyond 200 m per axis or 400 m of distance
is pinned, and per-axis clipping distorts the direction when more than one axis saturates. Measured over the
92 shipped legs: **33 are over 200 m (36%) and 11 over 400 m (12%)**; per level 8-3 10/13 and 4/13, 4-4 4/8
and 2/8, 2-4 3/4 and 2/4, 7-4 2/4 and 1/4. Re-scaling would change a learned input column's meaning on the 18
gates levels, so it is **not** done: the fallback's target slots are direction-only on a long leg, and
`gate_approach` — a scalar on the true distance — is unaffected either way.

### 4.2 Literal example — the whole of `route_Level_0-5.json`

```json
{"level":"Level 0-5","version":2,"source":"room-trunk offline v2",
 "exit":{"pos":[391.5,-121.6,382.0],"target":"Level 1-1"},
 "start_room":"1 - Opening Hallway","trunk_collapsed":[],
 "tour_ratio":1.0,"checkpoints_within_60m":"1/1","legs_witnessed":"3/5","last_rung_to_exit_m":211.1,
 "rungs":[
  {"key":"0,-10,300","pos":[0.0,-10.0,300.0],"hops":4,"name":"1 - Opening Hallway","gated_by":[],"open":false,"locked":false,"active":true},
  {"key":"0,0,351","pos":[0.0,0.0,351.0],"hops":3,"name":"2 - Lava Foundry","gated_by":[],"open":false,"locked":false,"active":true},
  {"key":"134,-6,382","pos":[133.5,-6.0,382.0],"hops":2,"name":"3 - Smallway","gated_by":[],"open":false,"locked":false,"active":true},
  {"key":"174,-6,382","pos":[174.5,-6.0,382.0],"hops":1,"name":"4 - Cerberus Arena","gated_by":[],"open":false,"locked":false,"active":true},
  {"key":"214,-6,382","pos":[214.5,-6.5,382.0],"hops":0,"name":"5 - Final Hallway","gated_by":[],"open":false,"locked":false,"active":true}]}
```

Tier C level 0-5, which has **zero gate candidates today**: 5 rungs, all standable (R4), no parallel set to
collapse, no altar on the route, median gap 52 m, Cerberus's arena at hops 1, the exit's own room at hops 0,
and the pit 211 m past it.

### 4.3 Invariants the emitter guarantees and the tests assert

Order of operations: **I6 → T → R4 → I2 → R1 → I3 → R2**. Revision 1 ran R2 *before* the budget cap so that
"the cap cannot change a level's verdict"; revision 2 reverses that, because the review measured the
consequence: 8-3's stored `tour_ratio` 1.264 describes a 28-rung chain that was then cut to 24, and **the 24
rungs that actually shipped score 1.421**, over R2's own 1.35 threshold. A guard that is evaluated on data
that does not ship is not a guard. R2 is now judged on what ships, and §7.1 test 5 reproduces it from `pos`.

- **I1 total order.** `hops` are `len(rungs)-1 … 0`, strictly descending, no duplicates.
- **I6 optional areas are not the route.** A rung whose own name (the part before any `+` a merge added)
  matches `\b(secret|bonus)\b`, **or carries an `S` ordinal suffix, or contains `P Door` or `Prime`**, is
  dropped. Revision 1's `checkpoint` carve-out is **deleted**.
  *(Revision 2 measurement: the `<N>S` suffix occurs exactly three times campaign-wide — `6S - P Door` on
  3-1, `1S - P Door` on 6-2, `10S - Secret Arena` on 8-3 — and all three are optional areas. The Prime
  Sanctum door needs every level in the layer at P rank and is never on the route to the exit; revision 1
  shipped it as 6-2's hops-1 rung, one rung before the exit room, and 6-2's whole claim to a route rested on
  it. The carve-out is deleted because being a checkpoint does not make a secret area on-route: it kept
  `3B - Secret Entrance Checkpoint` at hops 2 on 7-4, and R4 independently found that same rung unreachable.
  Effect on what ships: 6-2 loses its middle rung and falls to layer 3; 7-4 loses `3B`.)*
- **T trunk only. (New in revision 2, and the reason the design changed — §2.6.)** A **parallel set** is
  (a) a group of ≥ 2 suffixed rooms sharing one ordinal *within one numbering scope*, or (b) the whole
  branch-prefix region when the level carries more than one distinct branch prefix. Every member is dropped;
  the rung the set hangs off is kept. *(Measured effect: 1-4 loses `3C/3DL/3DR/3TL/3TR` → 4 rungs;
  8-3 loses `B1…B8` and `R1…R6` → 13 rungs; 4-2 loses `6A/6B` → 7 before R4; 1-3 loses both wings → 2 rungs,
  failing R1. A lone suffix is a chain, not a star, and is kept: 5-2's `7B - Crooked Cabin`, 8-3's
  `10B - Night Street`, 4-4's already-merged `3A`. A single branch prefix is a chain too: 1-4's `V2 - Arena`
  is kept, which matters — it is the boss room the level ends in.)*
  **The test is scope-aware and the emitter has the scope; the shipped files do not carry it.** No shipped
  level has a suffixed same-ordinal pair straddling two scopes, so revision 2's measurement is unaffected,
  but `build_routes.py` must apply T inside a scope or 7-1's two 1…5 scopes would collapse into each other.
- **R4 reachability. (New in revision 2, promoted from revision 1's §7.4 — §2.7.)** A rung is dropped when the voxel pass
  finds no standable cell inside `_is_reached`'s own cylinder (8 m horizontal, 6 m vertical, `open: false`)
  around it; `hops` are renumbered afterward. *(Measured: 5 rungs dropped across the campaign — 4-2 ×1,
  5-2 ×1, 7-4 ×3 — and 0 on the 60-gate control.)* This is the guard the `data` review's blocking item asked
  for and it is not optional: without it, 5-2 can never target its own exit.
- **I2 separation.** No two rungs within `SEP = 16.0 m` = 2 × `gate_reach_m`. *(This was a real defect in the
  raw chain: 12 of 33 levels had rung pairs inside the reach cylinder, seven at 0.0 m. On 7-1,
  `2 - Left Arena`, `1 - Wave 1` and `1 - Wave 2` share one point, so arriving there would silently mark
  three rungs reached and skip two. The emitter merges the closest pair into the later (lower-hops) rung,
  concatenating names, and renumbers, repeating until clean.)*
  **Revision 2 adds: `ActivateNextWave` containers are excluded from the room inventory before chaining.**
  They match the `<N> - <Name>` room regex, so on 7-1 `1 - Wave 1` / `1 - Wave 2` sorted as ordinal 1 and the
  I2 merge dragged `2 - Left Arena` down with them, producing the shipped order `1, 3, 4, 5, 2, 1, 2, 3, 4,
  5` — room 2 after room 5 inside one scope, which R2 does not catch because the rooms are spatially
  clustered (tour 1.039). *(Measured effect of the fix on 7-1: order becomes `1, 2, 3, 4, 5 | 1, 2, 3, 4, 5`,
  tour 1.039 → 1.000, polyline 1 613 → 1 553 m, median gap 100 → 80 m.)*
- **I3 budget cap.** At most `MAX_RUNGS = 24` rungs, dropped cheapest-detour-first, never the first or last.
  *(After T it binds on **no** level: the longest trunk is 7-2 at 15 rungs. It stays as a bound, not as a
  behaviour.)*
- **I4 guards.** A file is emitted only if, after I6, T, R4 and I2:
  - **R1** ≥ 3 rungs. *(Drops 5-4: one rung, no numbered rooms, no doors. Drops 1-3 and 6-2 in revision 2.)*
  - **R2** `tour_ratio ≤ 1.35` on the shipped rungs. *(Measured over the 12: max **1.255** on 7-2, then 7-3
    1.174, 8-3 1.247, 7-1 1.000 after the wave-container fix, 1.000 on the other eight.)*
  - **R3** `level` equals the scene name and `version` is known.
- **I5 staleness.** Python refuses the file when `campaign.exit.pos` is present and more than
  `route_exit_tol_m` (5.0) from `exit.pos`. **On refusal it also clears `target` and `best_dist` once**, so
  the level really does fall to layer-3 behaviour; revision 1 left `_choose_target`'s
  `if not active: return self.target` holding the last stale rung while `gate_approach` kept paying toward
  it. When `campaign.exit` is absent (a frame mid-load) the ladder is used unvalidated.
  *(Verified: all 12 files' `exit.pos` match the survey's reproduction of the mod's own FinalPit rule to
  0.0 m, including the four levels with 4-6 pits, so I5 will not misfire on emitter/mod disagreement.)*

### 4.4 The `GateProgress` change, in pseudocode

The whole Python route change is the loader, the three `return []` arms, the seeding rule and the source
guard. **The `return gates` success path is not touched**, so a level on layer 1 gets the identical list it
gets today.

```python
# campaign.py — new, module level
@lru_cache(maxsize=64)
def _read_route(scene, route_dir):                     # route_dir "" = the packaged routes/ folder
    doc = read_json(route_dir_or_default(route_dir) / f"route_{safe_name(scene)}.json")
    if doc is None or doc.get("version") != 2 or doc.get("level") != scene:
        return None                                    # missing / unknown / mislabelled -> layer 3
    rungs = [r for r in doc["rungs"] if r.get("pos") and r.get("hops") is not None]
    return {"exit_pos": doc["exit"]["pos"], "rungs": rungs} if len(rungs) >= 3 else None


def load_route(scene, route_dir):
    """A per-instance DEEP COPY: `_choose_target` hands its result out as `self.target`, and an
    lru_cache'd document would then be aliased by every env in the process."""
    doc = _read_route(scene, route_dir)
    return None if doc is None else deepcopy(doc)


class GateProgress:
    def __init__(self, ..., route=None):               # env passes load_route(self.level, cfg.route_dir)
        self._route = route
        self._route_ok = None                          # I5 verdict, decided once per level load
        self._hops_source = None                       # "gates" | "rooms": what set best_hops

    # -- the fallback -------------------------------------------------------------------
    def _rooms(self, campaign):
        """The shipped room trunk, or [] — today's behaviour — when there is none or it is stale."""
        if not self._route:
            return []
        if self._route_ok is None:
            ex = (campaign or {}).get("exit") or {}
            if ex.get("pos") is None:
                return self._route["rungs"]            # mid-load frame: use it, decide the check later
            self._route_ok = _dist3(ex["pos"], self._route["exit_pos"]) <= ROUTE_EXIT_TOL_M
            if not self._route_ok:
                log_once("route file for this level is stale (exit moved); falling back to the exit vector")
                self.target = None                     # I5: really fall back, do not hold the stale rung
                self.best_dist.clear()
        return self._route["rungs"] if self._route_ok else []

    def _gates(self, campaign):
        if not campaign or not campaign.get("gates_ordered"):
            return self._rooms(campaign)                                  # was: return []
        gates = [g for g in (campaign.get("gates") or ()) if isinstance(g, dict) and g.get("pos")]
        ladder = [g for g in gates if not g.get("altar_only")] or gates
        if not ladder:
            return self._rooms(campaign)                                  # was: return []
        if sum(1 for g in ladder if g.get("hops") is not None) / len(ladder) < self.hops_min_frac:
            return self._rooms(campaign)                                  # was: return []
        return gates                                                      # UNCHANGED, byte for byte

    def _is_room_ladder(self, rungs):
        # identity against the loaded document, never a field on the entries, so nothing in the data can
        # turn the seeding rule on for a gate. Short-circuits, so a gates level never subscripts None.
        return self._route is not None and rungs is self._route["rungs"]

    # -- source guard: a mid-load layer flip must not pay the hops difference -------------
    def _note_reached(self, campaign, pos):
        rungs = self._gates(campaign)
        source = "rooms" if self._is_room_ladder(rungs) else "gates"
        if self._hops_source is not None and source != self._hops_source:
            self.best_hops = self.paid_hops = None     # different ladder, different hop scale
            self.reached.clear(); self.hops_reached.clear()
        self._hops_source = source
        ...                                            # the rest of _note_reached is unchanged

    # -- seeding (§6): rooms path only ---------------------------------------------------
    def _choose_target(self, campaign, pos):
        if pos is None:
            return self.target
        rungs = self._gates(campaign)
        active = [g for g in rungs if g.get("hops") is not None and g.get("active")]
        if not active:
            return self.target
        if self.best_hops is None and self._is_room_ladder(rungs):
            near = self._nearest(active, pos)
            if _dist3(pos, near["pos"]) <= ROUTE_SEED_M:
                # absorb, never pay: the player did not travel to the rung it started at
                self.best_hops = self.paid_hops = int(near["hops"])
        ...                                            # the rest of _choose_target is unchanged

    def new_level_load(self, campaign, player_pos=None):
        self._route_ok = None                          # re-check staleness on every level load
        self._hops_source = None
        ...                                            # the rest is unchanged
```

`_note_reached`, `_is_reached`, `_subgoal`, `update`, `mark_paid` and `reset_episode` are otherwise
**unchanged**.

**The exit becomes the target only after the lowest rung is physically touched**, and revision 2 checked the
proposed data-only fix for it. `_choose_target` with `best_hops == 1` computes `lower = {0}` and returns the
hops-0 rung; only `best_hops == 0` returns `_exit(campaign)`. Renumbering so the lowest rung is hops 1 was
proposed as a no-code fix — **it is a no-op**: with rungs `n…1`, `best_hops == 2` gives `lower = {1}` and the
same rung, and the `lower` empty arm is reached at `best_hops == 1`, i.e. still only after touching the lowest
rung (verified directly against `_choose_target` on 8-3's file, both numberings). The underlying gap is real
and is a data property, not a code one: it is the `last rung → pit` column in §10 (1 690 m on 8-3, 540 m on
4-4, 373 m on 5-2 after R4). Nothing in this spec closes it; the exit vector in slots 0-4 is what carries it.

### 4.5 Skull and altar locks on the trunk

**Measured, not assumed** (`spec-rev/lockprobe.py`: an altar with a real `acceptedItemType`, the doors it
drives from the offline parse, and those doors' distance to the trunk's own segments):

| level | ships | altar-driven door on the trunk |
|---|---|---|
| 0-5, 2-4, 4-2, 5-2, 7-3, 7-4, 8-3, 8-4 | yes | **none** — the trunk is clear spawn to exit |
| 1-4 | yes | leg 2 (`3 - Main Hall` → `V2 - Arena`): three `SkullBlue` doors at 0-2 m |
| 4-4 | yes | leg 4: one `SkullBlue` door at 6 m |
| 7-1 | yes | legs 0 and 2: two `SkullBlue` doors at 0-1 m |
| 7-2 | yes | leg 11: one `SkullRed` door at 1 m |
| 1-3 | no (R1) | leg 0: one door wanting `SkullBlue`/`SkullRed` at 2 m |

**This is a direct consequence of the trunk collapse and mostly a benefit.** The skull-fetch rooms *are* the
parallel sets: collapsing 8-3's `B`/`R` branches, 4-2's `6A/6B` and 1-4's five side rooms removed most of the
lock exposure with them. Only four shipped levels still cross a lock, against eight in revision 1.

**Base behaviour (stages S1-S2): the trunk routes the agent to the lock and stops there.** A room rung carries
no `needs_item`, so `_subgoal` returns it untouched, no fetch/carry sub-goal is produced, and `_protect_carry`
— whose `carrying` test is `target.get("subgoal") == SUBGOAL_ALTAR and target.get("item") in held` — never
sees a carry to protect. (Its other arm, the filled-altar guard against `Punch.AltHit`, still fires; only the
carry half goes inert.) The rung behind the lock is a correct destination the agent cannot yet reach, so
`best_hops` freezes one rung above it and `gate_approach` saturates. That is **not** a regression — those
levels have no route signal at all today — but it is not spawn-to-exit either, and §10 says so per level.

**Stage S3 closes it with no mod change and no new code path.** The mod already publishes `campaign.altars`
with `pos`, `item`, `filled` and `doors[]` for the skull-gates work. So each rung carries `gated_by`: the
offline positions of the altars that must be filled before its room is passable, and `_rooms()` stamps
`needs_item` onto a copy of the rung each step from live state:

> for each `gated_by` position, find the live altar within `ALTAR_MATCH_M` (3.0 m); if any matched altar
> reports `filled: false`, set `needs_item` to the lowest `item` type among them, exactly as the mod's own
> gate rule does. If nothing matches, stamp nothing.

`_subgoal`, `_protect_carry` and `_is_reached` then work byte for byte as they do on a skull-locked gate, and
the failure mode is safe by construction: **no live match means no stamp means today's behaviour**, never a
wedge. Position matching rather than key matching is deliberate — the mod's altar keys carry `#N` suffixes for
co-located twins and a dead-twin filter whose verdict shifts as rooms light up, and the offline parse cannot
reproduce either; the zone's `transform.position` is the one thing both sides read identically.

S3 is sequenced after S2 because it is the only part that needs the altar→room attribution to be right, and a
wrong `needs_item` on a rung that is *not* locked would make that rung permanently unreachable
(`_is_reached` refuses a gate with `needs_item` set). Its acceptance test is therefore an offline one
(§7.5) before any live run.

---

## 5. Tier C: boss levels

### 5.1 What the route does

A boss is not a special case for the trunk. On all seven Tier C levels the boss stands on the last or
second-to-last rung, so the trunk walks the agent to the fight and stops. Verified by two independent parses
that **agree on every level** (`bossc.py` re-run on 0-5 and 5-2; `bossends.json`; `review-data/r08_boss.py`
re-run here):

| level | boss (`BossHealthBar`) | boss rung | what ends the fight | what that opens |
|---|---|---|---|---|
| 0-5 | Cerberus ×2 @`208,-6,370` / `208,-6,394` | `4 - Cerberus Arena`, hops **1** of 4 | `ActivateNextWaveHP` at hp ≤ 40, then `lastWave` @`174,-6,382` | `toActivate` = **`5 - Final Hallway`**, the exit's own room, + `DelayedDoorActivation` |
| 2-4 | Corpse of King Minos @`280,-599,575` | `4 - Second Encounter`, hops **0** | **no wave at all** — the boss's own death event | `MinosBoss.onDeathImpact` → `4 - Second Encounter`; `onDeathOver` → `DeadMinos` |
| 3-2 | Gabriel, Judge of Hell @`0,-151,262` | `4 - Heart Chamber` (**gates level**, hops 0 gate) | no wave; `Machine.onDeath` / `Enemy.onDeath`, then `GabrielOutro.onDisappear` | re-activates `4 - Heart Chamber` / `GabrielOutro` — the level *is* the fight |
| 4-2 | Sisyphean Insurrectionist @`8,-15,1244` | `7 - Boss Arena`, hops **0** | `lastWave` @`8,-15,1158` | `FinalDoorOpener` + `CheckPointsReEnabler` |
| 5-2 | Ferryman @`-60,30,1240` | `7B - Crooked Cabin`, hops **0** after R4 | `lastWave` @`88,-53,1240` | `FightEnd`; two `Idol` deaths → `8 - Ship` + `ShipRise` |
| 6-2 | Gabriel, Apostate of Hate @`-299,28,350` | — (**6-2 falls to layer 3**, §4.3 I6) | no wave; boss death event | `Door "ExitRaiser"` → `FinalRoom` |
| 7-4 | 1000-THR Earthmover @`0,838,644` (+ Defence System) | `4 - Brain + 5 - Return Checkpoint`, hops **1** of 3 | `ActivateNextWaveHP` on `Brain` at hp ≤ 50 | **nothing** — see below |

Six of the seven end with a one-room `Door "DoorLeft"` whose single `activatedRooms` entry is the object
`Pit` — which is exactly *why* they have no gate ladder. The trunk never needs the pit's room to be a node.

**Revision 2 correction, 7-4.** Revision 1 credited the Earthmover fight with `arena_clear` and
`door_unlock`. Re-measured: the Brain's trigger is an `ActivateNextWaveHP` with `lastWave = false` and empty
`toActivate`/`doors`, and `ActivateNextWaveHP` is a **separate `MonoBehaviour`, not a subclass** of
`ActivateNextWave` (`decompiled/ActivateNextWaveHP.cs`), while `CampaignPatches.cs:78` patches
`ActivateNextWave.EndWaves` only. 7-4's one `lastWave` `ActivateNextWave`, the one carrying
`doors=['Quakedoor']`, sits at `0,458.5,649.8` — the earlier arena, reached before the boss. So on 7-4 the
milestones are paid *before* the fight and the Earthmover itself pays only `kill` + `damage_dealt`. 7-4 moves
into the 2-4 / 3-2 group in §5.3 and into the `boss_down` revisit trigger.

**Target during the fight.** Once `best_hops` reaches the boss rung's value, `_choose_target` picks the rung
below, and at hops 0 it returns `_exit(campaign)`. On 2-4, 4-2 and 5-2, where the boss is at hops 0, the
target during the fight is the pit itself. That is accepted, not fixed: the pit is unreachable until the fight
ends, `gate_approach` to it is bounded by a single new-best-distance budget, and a "hold the target on the
boss" rule would need live boss state inside `GateProgress`, which §3 deliberately keeps out.

### 5.2 What the mod must report (the only mod work in this spec, and it is off the critical path)

A new `campaign.bosses` array, built on the same 30-obs rescan as `gates`, from every `BossHealthBar` in the
scene (`decompiled/BossHealthBar.cs`: `bossName`, `secondaryBar`, `healthLayers[]`, and
`source: IEnemyHealthDetails` with `FullName`, `Health`, `Dead`):

```
"bosses": [{"key": "208,-6,370", "name": "CERBERUS, GUARDIAN OF HELL", "pos": [207.5,-6.5,369.5],
            "health": 1.0, "health_max": null, "dead": false, "secondary": false, "active": false}],
"boss_active": false          # any non-secondary bar exists, is active and is not dead
```

**Revision 2 rewrote this block; revision 1's version would have thrown on every campaign level.**
`BossHealthBar.source` is `[HideInInspector]` — not serialized — and is assigned in `Awake()`, which Unity
does not run on a component of an inactive GameObject. Measured over the seven Tier C levels plus every gates
level (`review-data/r08_boss.py`, `r17_gateslevels.py`, `r18_01boss.py`, all re-run): **every BossHealthBar in
the campaign reports `activeInHierarchy == false` at load** — both 0-5 Cerberus bars, 2-4 Minos, 3-2 Gabriel,
4-2 Sisyphus, 5-2 Ferryman, 6-2 Gabriel 2nd, both 7-4 bars, 0-1 Malicious Face, 0-2/0-3 Swordsmachine, all
four 6-1 bars, 8-2 Mirror Reaper. So `FindObjectsOfType<BossHealthBar>(true)` returns bars whose `source` is
null, and `source.Health` / `source.Dead` / `source.FullName` throw. With revision 1's "any throw yields
`bosses: []`", one un-activated bar blanked the entire array — i.e. the array would have been empty on every
level, always. The rules:

- **Wrap each bar individually, never the whole array.** Skip a bar whose `source` is null or whose
  `gameObject.activeInHierarchy` is false; emit the rest. A skipped bar is not an error.
- **`health_max` is `healthLayers.Sum(l => l.health)` taken at the first scan on which `source != null`**, not
  at the first scan, and is `null` until then, so Python's "a missing `health_max` means unknown" rule really
  fires. Revision 1 latched it at the first scan, which reads **0** on every bar shipping with an empty
  `healthLayers` — measured: 0-5's second Cerberus (`StatueFake/Cerberus`), both 7-4 bars, and all four 6-1
  bars. `Awake` is what fills that array (`if (healthLayers.Length == 0) healthLayers = new HealthLayer[1] {
  health = source.Health }`), so before activation there is nothing to sum.
- **`dead` is `source.Dead`, and it is the only end-of-fight signal.** Revision 1 said "a bar removed by
  `DisappearBar()` simply stops appearing in the array". It does not: `DisappearBar()` only calls
  `BossBarManager.ExpireImmediately(bossBarId)` (`BossHealthBar.cs:109-115`); the component is never
  destroyed and `FindObjectsOfType(true)` keeps returning it.
- `key` is `CampaignPatches.Key()` of the bar's own transform. **It is not an identity.** The bar lives on the
  enemy, so the key moves as the boss moves, and 0-5's two Cerberus bars start 24 m apart and can collide onto
  one key. Nothing in this spec consumes it; do not let a later change use it as one.
- `BossHealthBarTemplate` is a separate MonoBehaviour, not a subclass, so the scan will not pick up UI
  templates. *(Verified.)*

Nothing else is added. The fight-over signal on 0-5, 4-2 and 5-2 already exists: the last wave's
`ActivateNextWave` key appears in `cleared_arenas` via `ActivateNextWave.EndWaves`. It does **not** exist on
7-4 (above), 2-4, 3-2 or 6-2.

### 5.3 What pays during a boss fight

Nothing new. Every term already exists and is already wired:

| what | weight | when | Tier C levels |
|---|---|---|---|
| `arena_clear` | 10.0 | the boss arena's last wave clears (`ActivateNextWave.EndWaves` → `cleared_arenas`) | 0-5, 4-2, 5-2 |
| `door_unlock` | 15.0 | the `FinalDoor` the fight unlocks | 0-5, 4-2 |
| `gate` | 15.0 | the rung below the boss rung | 0-5 (elsewhere the boss is at hops 0 and there is no rung left) |
| `kill`, `damage_dealt` | 0.5, 0.5 | throughout | all |
| `level_complete` | 100.0 | the pit | all |

**No `boss_down` term is added, and that is a decision, not an omission.** On 2-4, 3-2 and **7-4** there is no
wave the fight ends, so winning pays only `kill` + `damage_dealt` until `level_complete` — but on 2-4 and 3-2
the boss's room **is** the exit's room, so `level_complete` follows within metres of the kill. 7-4 is the
weakest case (its `Pit` rung is 62 m from the FinalPit but the Brain is 600 m above). Adding a reward term
would forfeit the "no change to `rewards.py`" property that makes this spec safe to merge next to a live run.
**Trigger to revisit:** if a 2-4, 3-2 or 7-4 run reaches the boss reliably (`gates_reached` at maximum in
> 50% of fresh starts) and still never completes, add `boss_down` — §5.2's block already carries everything
it needs.

### 5.4 Episode length and the stuck clock

A boss fight is long and produces no `kills` until it ends. **Revision 1 added a global clause restarting the
stuck clock on a fall in any non-secondary boss's `health`. Revision 2 removes it from the merge**, for two
measured reasons:

1. **It is very nearly a no-op.** `_campaign_progress`'s existing test is
   `fought = stats.kills > before.kills or stats.style > before.style` (`env.py:1024`), and a boss fight
   generates style on every hit. Where `fought` is already true the new clause changes nothing; where it is
   false, the clause is the claim that needs measuring, and nobody has measured it.
2. **It fires on 0-1, which is the merge gate.** Measured: **9 of the 18 gates-layer levels carry a
   `BossHealthBar`** — 0-1, 0-2, 0-3, 1-2, 2-3, 3-2, 4-3, 6-1, 8-2 — and 0-1's `MALICIOUS FACE` bar is
   `secondaryBar = 0`, exactly what the clause keys on. A3 (§8) demands that completion count, official times
   and `gates_reached` be *identical* over 10 deterministic `eval.py` episodes on 0-1; an agent at 0-1's ~50%
   completion rate reaches the Malicious Face regularly, so an episode whose termination moved from `stuck` to
   `max_steps` would fail A3 for a reason that has nothing to do with routing. This was the **only**
   behavioural change in revision 1 that reached a gates level — `spaces.py`'s slot-12 clamp is provably a
   no-op (campaign-wide max gate hops 13 < 20, and the longest trunk after T has max hops 14), a
   `GateProgress` built with `route=None` leaves the gates path untouched, and no route file ships for any of
   the 18. Removing the clause restores §1's byte-for-byte claim in full.

If it is wanted later it belongs in stage S4, gated on `route_source == "rooms"`, with `end_reason`
distributions on 0-5 measured before and after.

- `max_steps_per_level` already exists; the Tier C levels get entries when they are first trained (0-5 is the
  only Tier C level in the Prelude). Sizing: 0-5's trunk polyline is 270 m and the human reference for the
  comparable 0-1 is 146 s, so 0-5 starts at the same `max_steps` as 0-1 and is raised only if `stuck` rather
  than `max_steps` is what ends its episodes.

---

## 6. Reward arithmetic

**No new reward weight, no change to `rewards.py`, no change to the anti-farm rules.** A room rung is shaped
exactly like a gate, so `gate` (15.0) still pays once per new lower `hops` per **level load**, and
`gate_approach` (0.15) still pays metres of new best closeness to the current target with `best_dist` keyed
per target and seeded once per episode.

Measured per level on the shipped trunks (`spec-rev/recompute.py`; weights read from
`configs/campaign_gates_prelude.yaml`: `gate` 15.0, `gate_approach` 0.15, `level_complete` 100.0). `approach`
is the whole polyline **plus the last rung → pit leg**, which is its own `best_dist` key and which revision 1
left out of its headline:

| level | rungs | `gate` total | polyline m | last leg m | `gate_approach` | ladder ÷ `level_complete` |
|---|---|---|---|---|---|---|
| 0-5 | 5 | 60 | 270 | 211 | 72 | 1.3× |
| 1-4 | 4 | 45 | 263 | 150 | 62 | 1.1× |
| 2-4 | 4 | 45 | 1 076 | 111 | 178 | 2.2× |
| 4-2 | 6 | 75 | 883 | 163 | 157 | 2.3× |
| 4-4 | 8 | 105 | 2 073 | 540 | 392 | 5.0× |
| 5-2 | 7 | 90 | 742 | 373 | 167 | 2.6× |
| 7-1 | 10 | 135 | 1 553 | 128 | 252 | 3.9× |
| 7-2 | 15 | 210 | 1 771 | 168 | 291 | 5.0× |
| 7-3 | 12 | 165 | 1 244 | 225 | 220 | 3.9× |
| 7-4 | 4 | 45 | 1 296 | 62 | 204 | 2.5× |
| **8-3** | **13** | **180** | **3 542** | **1 690** | **785** | **9.7×** |
| 8-4 | 4 | 45 | 403 | 62 | 70 | 1.2× |

**The trunk collapse cut the worst case nearly in half**: 8-3 was 24 rungs, a 6 278 m polyline and 1 195 of
`gate_approach` (15.6× `level_complete`) in revision 1. It is still the largest single term in this spec and
the first thing to check against `part_gate_approach` in `metrics_log.csv`. `gate` is once per level load;
`gate_approach` is re-earned every episode (`reset_episode` clears `best_dist`) and at `fresh_start_prob` 0.2
most episodes are respawns, so the per-episode number is the one that matters and is what the table reports.

**Two known free-shaping cases, stated because §10 lists the gaps without noting they pay.** 4-4's 1 099 m
hops-4→3 step and 7-4's 917 m step are authored relocations (a post-boss outro and a walking machine), so
165 and 138 of `gate_approach` are collected in the single step the cutscene fires. Not farmable — once per
traverse, and `min_gain` does not help — but it is reward for a scripted move.

**Levers, in order of preference:** the trunk itself (regenerate one file), `MAX_RUNGS` in the data, then
`gate` in the config. There is deliberately **no per-level lever for `gate_approach`**: it is a global weight
shared with the live `campaign_gates` run on 0-1/0-3/0-4, and adding a per-level scale would mean a reward
change, which is the property §1 is buying. The tripwire is §12.6.

**One behaviour change, on the fallback path only — seeding.** `_choose_target` with `best_hops is None`
returns the *nearest* rung. On a total order that rung can be behind the player: at 0-1's real spawn
`(39.7, −0.5, 343.7)` the nearest rung is `2B - Hallway` 31.6 m **behind**, while the correct next rung is
`3 - Gun Room` 34.4 m ahead. So on a rooms ladder, and only there:

> when `best_hops is None` and the nearest rung is within `route_seed_m` (150.0 m), **absorb** it — set
> `best_hops` and `paid_hops` to that rung's `hops` without paying — and target the rung below.

This is `mark_paid` semantics applied to the rung the player starts at, which is correct: you did not travel
to it. It cannot pay anything, so it cannot be farmed. On a gates ladder the rule is not reached.

**Revision 2 corrects the rule's stated rationale.** Revision 1 justified it partly with "every checkpoint
respawn starts mid-ladder with `best_hops` None". It does not: `best_hops`/`paid_hops` are **level-load**
scoped (GateProgress's own class docstring), `env.py:884-888` calls `mark_paid` on a respawn, and `mark_paid`
only sets `paid_hops = best_hops` — only `new_level_load` clears them. So the rule fires **once per level
load**, not per respawn. The first motivation stands on its own; the frequency §12.7 reasons about was wrong.

**An adjacent real bug this makes worse, to fix in S2.** `_respawn` (`env.py:892-906`) also calls `mark_paid`,
not `new_level_load`, **even on the branch where `StatsManager.Restart` reloaded the whole level** because
there was no checkpoint yet. After that reload `best_hops` is still low, so `_choose_target` targets a rung
far ahead of a player standing at spawn and `gate` can never pay again for that load. It exists today on gates
ladders; a 13-rung trunk makes it bite much harder. Fix: on the no-checkpoint branch `_respawn` already
detects (`self._current_checkpoint(raw) is None`), call `self.gates.new_level_load(...)` instead of
`mark_paid`.

---

## 7. Offline tests (no game)

### 7.1 `python/tests/test_route_files.py` — new

Over every committed `routes/route_*.json`:

1. schema: required keys present, `version == 2`, `level` matches the filename, `rungs` non-empty;
2. **I1**: `hops` are `n-1 … 0`, strictly descending, unique;
3. **I2**: no two rungs within 16.0 m (3-D);
4. **I3**: `len(rungs) <= 24`;
5. **I4/R1**: `len(rungs) >= 3`; **R2**: `tour_ratio <= 1.35` **and the stored value is reproduced from the
   shipped `pos` list to 3 decimals** — this is the assertion that caught 8-3's 1.264-vs-1.421 gap, and it
   only works because §4.3 now evaluates R2 after I3;
6. **I6**: no rung's leading name matches `\b(secret|bonus)\b`, carries an `S` ordinal suffix, or contains
   `P Door`/`Prime` (the 1-3 regression: `S - Secret Fight` must not reappear at hops 1; the 6-2 regression:
   `1S - P Door` must not reappear as a route rung);
7. **T**: no two rungs share an ordinal with different letter suffixes, and at most one branch prefix appears;
8. every rung's `key` equals its own `pos` rounded to whole metres, and `open`/`locked` are `false`,
   `active` is `true`;
9. **the `Pit` exception is explicit**: a rung may be named `Pit` only as the hops-0 rung, and then its
   distance to `exit.pos` must be under 70 m. *(7-4 and 8-4 both ship this: the hops-0 rung is the FinalPit
   prefab's own root, 62.2 m from the pit on both, which is why both files carry the identical
   `last_rung_to_exit_m`. The probe finds standable floor at both. Revision 1 had no test for it, and "make
   the object named `Pit` the goal" is on the project's list of previously-rejected rules, so it is pinned
   rather than left to look like that rule sneaking back.)*
10. the set of shipped levels is exactly the 12 of §10 — a file appearing or vanishing is a test failure, not
    a silent coverage change.

### 7.2 `python/tests/test_campaign_env.py` — extended (`FakeLevel` growth)

`FakeLevel` gains a `route` mode: it reports a campaign block with **no usable gates**
(`gates_ordered: false`) while the env is given a route file for its scene. New cases:

- `test_route_fallback_pays_the_ladder`: walking the fake corridor's rungs pays `gate` once per rung and
  nothing on a second walk in the same level load;
- `test_route_fallback_is_not_used_when_gates_are_good`: with `gates_ordered: true` and full `hops`, the
  target keys are gate keys and the route file is never opened (assert via a counter on the loader);
- `test_route_seeding_absorbs_the_nearest_rung`: a player spawned between rungs 3 and 4 targets rung 3 (one
  below the nearest), and the absorbed rung pays 0;
- `test_route_respawn_mid_ladder_targets_forward`: after `mark_paid` at a mid-level checkpoint the target is
  the next lower rung, not the nearest overall;
- `test_route_reload_without_checkpoint_reseeds`: the `_respawn` bug in §6 — a death with no checkpoint
  reloads the level and `best_hops` must be cleared, not carried;
- `test_route_rejected_when_exit_moved`: a campaign block whose `exit.pos` is 20 m from the file's yields no
  target **and clears a target already held** (I5), and pays nothing;
- `test_route_source_flip_clears_the_ladder`: a block that starts `gates_ordered: false` and later reports a
  good gate ladder must not pay the hop difference between the two scales;
- `test_route_absent_is_todays_behaviour`: a scene with no file produces `target is None`, `gate == 0`,
  `gate_approach == 0` — byte for byte the current path.

### 7.3 `python/tests/test_campaign.py` — extended

`GateProgress` unit cases for the three `_gates()` arms, asserting the gates success path returns the
identical list object it returns today, and that `_is_room_ladder` short-circuits when `self._route is None`.

### 7.4 `python/tests/test_progress.py` — extended

One assertion, because the invariant it pins is currently only prose: **every key in `CAMPAIGN_INFO_KEYS`
survives `progress._num`** on a representative campaign `info`. `env.py:1131-1135` documents that those keys
"go through `Monitor(info_keywords=...)` and `ProgressCallback._num` and must all be numeric", and revision 1
put the string `route_source` in that tuple, which would have logged `None` on every episode and killed the
one metric every recovery path in §12 reads. §9 now emits an integer there and the string in
`EPISODE_LOG_RAW`, next to `level` and `end_reason`.

### 7.5 `build_routes.py --validate` — the offline reachability and lock checks

Not a unit test: the regeneration's own acceptance run, single-threaded, no game, no port.

- **R4**, the voxel standability pass (§2.7), which is now a guard rather than a report. It is the only part
  of the pipeline that needs the scene bundles' collision geometry, so it is also the slowest (2-21 s per
  level, 21 s on 8-3).
- **The connectivity report** §2.8 publishes: per-rung standability, leg witness, tour ratio.
- **S3 only, and blocking for S3:** for every rung carrying `gated_by`, assert the altar positions resolve
  against the offline parse to a zone with a real `acceptedItemType` that drives a door whose `activatedRooms`
  contains that rung's room, or that lies within 25 m of the leg into it. A `gated_by` that does not resolve
  is a build failure, because a wrong stamp makes the rung permanently unreachable.

---

## 8. In-game acceptance checks

Run on **one** game, nothing else on its port, exactly as `campaign_check.py` is run today
(`games.py launch --count 1 --monitor 1`). Never against a game a trainer is using.

- **A0 — offline, and it runs before anything else.** Replay the real `GateProgress` over each shipped trunk
  in every branch order the level allows and assert the target never advances past an unvisited rung, and that
  no step pays more than one instalment. This is what caught revision 1's blocking defect (§2.6) and it needs
  no game; `…\scratchpad\route\rev\sim.py` is the prototype. After invariant T there is no parallel set left,
  so the assertion should hold trivially — which is the point: it is the regression test for T.
- **A1 — 0-5 route, read-only.** `bridge_test.py --campaign` on a fresh `Level 0-5` load. Expect
  `gates_ordered: false` (today's known state) and, from the Python side, a target at `0,0,351` (hops 3) once
  the player leaves the opening hallway. The exit reported by the mod must be within 5 m of
  `(391.5, −121.6, 382.0)`, or I5 has fired and the data is stale.
- **A2 — the trunk advances.** `random_agent.py --mode campaign --level "Level 0-5"`: `gates_reached` must
  exceed 0 within 20 episodes. Today it is identically 0 on every Tier C level. **This is weak evidence** and
  is labelled as such: it shows a rung's cylinder is touchable by random walking near spawn, not that a policy
  follows the route. A0 and A3 are the checks that carry weight.
- **A3 — no regression on 0-1.** `eval.py models/campaign_gates/best.zip --level "Level 0-1" --episodes 10`
  before and after the change. Completion count, official times and `gates_reached` must be **identical**;
  0-1 has no route file and must not take the new code path. **This is the only gate that can block the
  merge**, and revision 2's removal of the global boss stuck-clock clause (§5.4) is what makes it a fair test
  — under revision 1 it could have failed on 0-1's Malicious Face for reasons unrelated to routing.
- **A4 — boss block (stage S4 only).** `campaign_check.py --level "Level 0-5"` gains **check 7**, not "a
  sixth check": `NAMES` already holds six (`campaign_check.py:47`). It asserts only that `bosses` is present
  and well-formed and that every entry carries a numeric `health` with `health_max` either numeric or null.
  It does **not** assert `health > 0` inside the arena, because `campaign_check.py` reaches places by
  `env.client.teleport` and a teleport into a switched-off room activates nothing — the documented reason
  check 5 FAILs on every level. Confirming a live bar needs the read-only `watch_completion.py` pattern over a
  human play-through, and that is how it will be confirmed.
- **A5 — stale-data detector.** Edit **one shipped route file's** `exit.pos` by 50 m, confirm the env logs the
  refusal once and that the target is `None` thereafter, then revert. Run it on the level whose file was
  edited — revision 1 compared against A3's 0-1 baseline, which has no route file and so could not detect the
  stale-target-held bug §4.3 I5 now fixes.

---

## 9. File-by-file change list

No file appears in two lists. **Stage labels are in §11.5.**

### MOD WORK (C#, mod 0.7.1) — stage S4, off the critical path

| file | change |
|---|---|
| `mod/UltrakillAIBridge/Obs/CampaignObserver.cs` | `ScanBosses()` beside `ScanGates()` on the same rescan; publish `bosses[]` and `boss_active` with the per-bar defensive rules of §5.2. ~80 lines, no new Unity API. |
| `mod/UltrakillAIBridge/Plugin.cs` | version string → `0.7.1`. |

### PYTHON WORK

| file | change | stage |
|---|---|---|
| `python/ultrakill_ai/campaign.py` | `_read_route` / `load_route` with the LRU cache and a per-instance deep copy; `GateProgress.__init__` takes `route`; the three `return []` arms of `_gates()`; `_is_room_ladder` with its `None` short-circuit; the `_hops_source` guard; the §6 seeding rule; I5 clearing `target`/`best_dist` on refusal. ~80 lines. | S2 |
| `python/ultrakill_ai/env.py` | build `GateProgress` with `route=load_route(self.level, cfg.route_dir) if cfg.route_fallback else None`; reload it in `_switch_level`; `route_source` as an **integer** (0 none / 1 gates / 2 rooms) in `info` and `CAMPAIGN_INFO_KEYS`, and the string `route_source_name` in `EPISODE_LOG_RAW`; fix `_respawn` to call `new_level_load` on the no-checkpoint reload branch (§6); `EnvConfig` gains `route_fallback: bool = True`, `route_dir: str = ""`, `route_exit_tol_m: float = 5.0`, `route_seed_m: float = 150.0`. `EnvConfig.from_dict` already ignores retired keys. | S2 |
| `python/ultrakill_ai/spaces.py` | slot 12 becomes `min(hops, 20.0) / 20.0`. Provably identical on every shipped ladder (campaign-wide max gate hops 13, longest trunk max hops 14) — a bound, not a behaviour change. | S2 |
| `python/ultrakill_ai/progress.py` | chart `route_source` as the numeric series; carry `route_source_name` through `EPISODE_LOG_RAW`. | S2 |
| `python/ultrakill_ai/campaign.py` (S3) | `_rooms()` stamps `needs_item` from live `campaign.altars` against each rung's `gated_by` (§4.5). ~25 lines, no new consumer. | S3 |
| `python/scripts/campaign_check.py` | check **7**, the boss block (A4). | S4 |
| `python/tests/test_route_files.py` | **new** (§7.1). | S1 |
| `python/tests/test_campaign_env.py` | `FakeLevel` route mode and the eight cases in §7.2. | S2 |
| `python/tests/test_campaign.py` | §7.3. | S2 |
| `python/tests/test_progress.py` | §7.4, the `CAMPAIGN_INFO_KEYS`-are-numeric assertion. | S2 |
| `docs/protocol.md` | document `bosses[]` / `boss_active`. | S4 |

### OFFLINE-DATA WORK

| file | change | stage |
|---|---|---|
| `python/ultrakill_ai/routes/route_Level_*.json` | **new**, 12 files, ~14 KB, the §10 levels. | S1 |
| `python/scripts/build_routes.py` | **new**: regenerates the files from the scene bundles — room inventory (excluding `ActivateNextWave` containers), numbering scopes, chain assembly, then I6, T, R4, I2, R1, I3, R2 in that order, plus the diagnostics and `--validate` (§7.5). Derived from `…\scratchpad\route\checkpoint-chain\ladder.py`, `…\scratchpad\route\judge\final_table.py` and `…\scratchpad\route\spec-rev\recompute.py`; the voxel pass from `…\scratchpad\route\geometry-geodesic\ukgeom.py`/`voxel.py`; the bundle path from `mod/GamePaths.props`. Single-threaded, ~4 min for all 33 levels (R4 dominates), no game, no port. | S1 |
| `docs/level-survey.md` | append §10 as the route-coverage section. | S1 |
| `python/configs/campaign_*.yaml` | no change — the fallback is on by default and inert on every level these configs name. | — |

`CLAUDE.md` is in none of the three lists on purpose: both sides touch it, and the project rule is that
whoever merges last brings it up to date.

**Note on the live configuration.** `configs/campaign_gates_prelude.yaml` names 0-1/0-3/0-4, none of which has
a route file, so §9's "no change" is true — and it means **the live run never exercises layer 2**. A0 and the
`FakeLevel` cases are therefore the whole of the pre-merge evidence for the fallback, and A3 is the whole of
the evidence that layer 1 is untouched. That is stated rather than implied.

---

## 10. Per-level coverage of the final design

All 33 shipped levels. `signal` is the layer that fires. Gates rows are today's behaviour, listed only to
show they are untouched; their `rungs` is hop tiers and `maxgap` is `route_max_tier_step_m`, both from the
survey's reproduction of the mod's own rules. Room rows are measured by `spec-rev/recompute.py` and
`legs.py` on the rungs **that ship**. `legs` is the connectivity column §2.8 defines: legs with an authored
checkpoint within 40 m of the segment, out of all legs including the last one into the pit.

| Level | Tier | signal | rungs | med gap m | max gap m | tour | legs wit. | last rung → pit m | cp≤60 | reaches | what breaks it |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | A | gates | 10 | – | 144 | – | – | – | – | exit | unchanged |
| 0-2 | B | gates | 8 | – | 73 | – | – | – | – | exit | unchanged |
| 0-3 | A | gates | 7 | – | 69 | – | – | – | – | exit | unchanged |
| 0-4 | A | gates | 6 | – | 82 | – | – | – | – | exit | unchanged |
| **0-5** | C | **rooms** | 5 | 46 | 137 | 1.00 | 3/5 | 211 | 1/1 | **exit** | – (gained; Cerberus at hops 1) |
| 1-1 | B | gates | 6 | – | 190 | – | – | – | – | exit | unchanged; its gate graph may collapse the level (§12.5) |
| 1-2 | B | gates | 5 | – | 61 | – | – | – | – | exit | unchanged |
| **1-3** | D | **exit vector** | – | – | – | – | – | – | – | – | **R1 after T**: Red and Blue wings are a parallel set (§2.6); collapsing both leaves 2 rungs |
| **1-4** | D | **rooms** | 4 | 81 | 117 | 1.00 | 2/4 | 150 | 2/2 | **lock** | trunk ends at the `V2 - Arena` door: 3 `SkullBlue` altars on leg 2 (S3) |
| 2-1 | A | gates | 4 | – | 339 | – | – | – | – | exit | unchanged |
| 2-2 | A | gates | 3 | – | 111 | – | – | – | – | exit | unchanged |
| 2-3 | B | gates | 3 | – | 92 | – | – | – | – | exit | unchanged |
| **2-4** | C | **rooms** | 4 | 425 | 430 | 1.00 | 1/4 | 111 | 1/2 | **exit** | tram level, 4 enormous rooms: barely denser than the exit vector; hops-1 rung is a tram |
| 3-1 | A | gates | 8 | – | 124 | – | – | – | – | exit | unchanged |
| 3-2 | C | gates | 4 | – | 344 | – | – | – | – | exit | unchanged; boss is in the exit room, the level *is* the fight |
| 4-1 | A | gates | 9 | – | 182 | – | – | – | – | exit | unchanged |
| **4-2** | C | **rooms** | 6 | 172 | 268 | 1.00 | 4/6 | 163 | 2/4 | **exit** | – (gained; Insurrectionist at hops 0). `5 - Temple Entrance` dropped by R4, `6A`/`6B` by T |
| 4-3 | D | gates | 3 | – | 76 | – | – | – | – | exit | unchanged (3/5 gates carry hops, passes at 0.60) |
| **4-4** | D | **rooms** | 8 | 119 | **1099** | 1.00 | 4/8 | **540** | 2/3 | **lock** | one `SkullBlue` altar on leg 4; the 1 099 m step and the 540 m final leg are the post-boss outro relocation, and two rungs are elevators |
| 5-1 | B | gates | 3 | – | 223 | – | – | – | – | exit | unchanged |
| **5-2** | C | **rooms** | 7 | 71 | 407 | 1.00 | 6/7 | **373** | 3/4 | **exit** | `8 - Ship` dropped by R4 (unreachable, and it was a hard wedge — §2.7), so the last leg is 373 m |
| 5-3 | B | gates | 14 | – | 110 | – | – | – | – | exit | unchanged |
| **5-4** | E | **exit vector** | – | – | – | – | – | – | – | – | **R1**: 1 rung. No numbered rooms, no gate-candidate doors, 1 checkpoint |
| 6-1 | D | gates | 3 | – | 231 | – | – | – | – | exit | unchanged (6/12, exactly on the 0.5 threshold) |
| **6-2** | C | **exit vector** | – | – | – | – | – | – | – | – | **R1 after I6**: only 3 rooms exist and the middle one is `1S - P Door`, the Prime Sanctum door, which is never on the route |
| **7-1** | E | **rooms** | 10 | 80 | 478 | 1.00 | 5/10 | 128 | 2/5 | **lock** | two `SkullBlue` altars on legs 0 and 2; 478 m tram step no component encodes; order fixed by excluding wave containers (§4.3 I2) |
| **7-2** | E | **rooms** | 15 | 76 | 384 | 1.26 | 10/15 | 168 | 5/6 | **lock** | one `SkullRed` altar on leg 11; 384 m tram step no component encodes |
| **7-3** | D | **rooms** | 12 | 104 | 184 | 1.17 | 8/12 | 225 | 4/4 | **exit** | – (gained; 46 doors gave today 0 gate candidates). 7 locked doors, none altar-driven |
| **7-4** | C | **rooms** | 4 | 291 | **917** | 1.00 | 3/4 | 62 | 5/7 | **exit** | 3 of 7 rungs dropped (2 by R4, `3B` by I6); the rest are checkpoint markers, not rooms; the 917 m step is the walking machine |
| 8-1 | D | gates | 5 | – | 289 | – | – | – | – | exit | unchanged (28/52, passes at 0.538) |
| 8-2 | B | gates | 9 | – | 834 | – | – | – | – | exit | unchanged; its room chain scores tour 1.55 and is **not** shipped |
| **8-3** | E | **rooms** | 13 | 258 | 814 | 1.25 | 11/13 | **1690** | 6/13 | **exit** | `B1…B8` and `R1…R6` collapsed by T — both dimensions are required and the ladder cannot express that (§2.6), so the 814 m step spans them unaided; the last 1 690 m is the exit vector |
| **8-4** | D | **rooms** | 4 | 65 | 307 | 1.00 | **0/4** | 62 | 0/1 | **exit** | the two halves are joined by a scripted fall no component encodes; one checkpoint, and it witnesses no leg |

**Totals: gates 18, rooms 12, exit vector only 3.** Today: gates 18, nothing 15.
Of the 12 room levels, **8 reach the exit and 4 stop at a skull lock** until S3 lands.
Campaign-wide: **26 of 33 routed spawn-to-exit, 4 spawn-to-lock, 3 unrouted.**

By tier, counting only the levels the fallback can act on:
**Tier C** (0-5, 2-4, 4-2, 5-2, 7-4 gained; 3-2 already on gates; 6-2 unrouted) — **5 of 6** newly routed,
all to the exit. **Tier D** (1-4, 4-4, 7-3, 8-4 gained; 4-3, 6-1, 8-1 already on gates; 1-3 unrouted) —
**4 of 5**, two of them to a lock. **Tier E** (7-1, 7-2, 8-3 gained; 5-4 unrouted) — **3 of 4**, two to a
lock.

12 route files, ~14 KB, **92 rungs**. Trunk gaps over the 12: median of per-level medians **92 m**, max
1 099 m. Trunk polylines: 270 m (0-5), 1 244 m (7-3), 3 542 m (8-3). Connectivity: **92/92 rungs standable**,
**57/92 legs checkpoint-witnessed (62%)**.

---

## 11. Unrouted, and the disagreements on record

### 11.1 The three unrouted levels

- **5-4** — the Leviathan arena. Zero numbered rooms, zero gate-candidate doors, one checkpoint, no authored
  activation structure (the activation graph gets 2 rungs of `MusicActivator` and `Audio Source`; the geometry
  route gets 1). Nothing in the level data describes a route. **No further work is proposed on 5-4; it is a
  swimming boss arena and the navigation problem is not the hard part of it.**
- **1-3** — lost in revision 2, and the loss is the design working. Its Red and Blue wings are both required
  and neither precedes the other; §2.6 measured what shipping a guessed order does. After T collapses both,
  1-3 has `1 - Water Hall` and `Pit`: 2 rungs, under R1. It is the best candidate for the first extension
  (§12.2), because it is the smallest level where "both required" is the only obstacle.
- **6-2** — lost in revision 2 because its only middle rung was `1S - P Door`, the Prime Sanctum door, which
  requires every level in the layer at P rank and is never on the route to the exit. 6-2 genuinely has only
  two numbered rooms. `2 - Organ Hall` alone would still be a real waypoint 181 m before the pit, and lowering
  R1 to 2 would ship it — **not done**, because a threshold moved to rescue one level is exactly the failure
  mode this project has logged twice, and because a 2-rung file is barely distinguishable from the exit
  vector it would replace.

### 11.2 Shipped but known weak

- **8-4** — **0/4 legs witnessed**, 4 rungs, both halves joined by a scripted fall that no component encodes.
  All three proposals fail on 8-4 for the same reason. Shipped because 4 ordered, standable rungs beat none
  and the failure is recoverable (§12.4). It is the first level to check if a fallback run wanders.
- **2-4** — 1/4 legs witnessed, 4 rungs across rooms hundreds of metres wide, and the hops-1 rung is a tram
  (`0 - Tram + 3 - First Encounter`), so its 8 m × 6 m cylinder sits where the tram started.
- **8-3** — the 814 m step spans two required dimensions the ladder cannot order (§2.6), and the last 1 690 m
  is unaided. Still 11/13 legs witnessed, the best connectivity score of the long levels.
- **7-1** — 2/5 checkpoints within 60 m and 5/10 legs witnessed, which no guard catches. Its order was wrong
  in revision 1 and is fixed by excluding wave containers; the fix is asserted offline, not by a live read.
  Flagged for a live read before 7-1 is trained.
- **4-4, 7-1, 7-4** — rung positions are **load-time transforms and some rungs are movers**: 4-4 runs through
  `2 - Elevator 1` and `3A - Elevator 2`, 7-1 and 2-4 through trams. On these the reach cylinder sits where
  the mover started, not where it is.
- **7-4** — after R4 and I6 its four rungs are all `Checkpoints/...` marker objects with no colliders under
  them, not rooms. They are standable and on route (4/7 checkpoints, 3/4 legs witnessed), but the level's
  route signal is a checkpoint chain wearing a room ladder's schema, and it should be read that way.

### 11.3 Levels where the room order and the gate order disagree

1-1 (tau −0.23), 8-1 (−0.60), 8-2 (−0.06) — all loop levels, all three still on the gates layer, so nothing
changes. `checkpoint-chain` argues 1-1's gate graph is the one that is wrong, because 1-1's final door lists
the spawn field in `activatedRooms`, which would put the spawn one hop from the goal. **That is an argument,
not a measurement, and it is not acted on here.** It is recorded in §12.5 as a falsifiable prediction.

### 11.4 Where a proposal's prose and its own artifact disagree

Recorded so the next reader does not chase it: `geometry-geodesic`'s coverage.md says 0-1's flagged 6.1 m link
"joins `7 - Fan Room` to `9 - Projectile Arena` and skips `8 - Curved Hallway`". Its shipped
`artifacts/route_0-1.json` says the 6.1 m link joins `10 - Combo Hallway` (hops 3) to `12 - Boss Hallway`
(hops 2). The phenomenon is real and `gap_m` does expose it; the cited example does not match the data.
Separately, `activation-graph`'s `out/bosses.json` holds only the last level of the last run (a write bug in
`bossc.py`); the script itself reproduces correctly and its printed table is what §5.1 uses.

### 11.5 The build plan, split

Four stages. **S1 and S2 are the spec; S3 and S4 are optional and separately mergeable.**

| stage | what | needs a game? | merge gate | unlocks |
|---|---|---|---|---|
| **S1** | `build_routes.py` with the I6 → T → R4 → I2 → R1 → I3 → R2 pipeline; the 12 route files; `test_route_files.py`; the survey appendix | no | §7.1 passes; `--validate` reproduces §10 | nothing at runtime — pure data, zero risk to the live run |
| **S2** | `load_route`, the three `_gates()` arms, the seeding rule, the `_hops_source` guard, I5's clear, the `_respawn` fix, `route_source`, the `FakeLevel` cases | no for the tests, yes for A1-A3 | **A0 offline, then A3 on 0-1 must be identical** | layer 2 on 12 levels: 8 to the exit, 4 to a lock |
| **S3** | `gated_by` in the data; `_rooms()` stamps `needs_item` live from `campaign.altars`; §7.5's blocking resolve check | no | §7.5 resolves every `gated_by`; then A1 on 1-4 | 1-4, 4-4, 7-1, 7-2 from lock to exit |
| **S4** | `ScanBosses()` (mod 0.7.1), `campaign_check.py` check 7, `docs/protocol.md`; optionally the boss stuck-clock clause gated on `route_source == "rooms"` | yes (rebuild, game closed) | A4 well-formedness, then a read-only human play-through | boss telemetry; the `boss_down` decision in §5.3 |

S1 can land while a run is live. S2 is the only stage that changes behaviour on a level the live run touches,
and A3 is the check that says it does not. S4 requires the game closed for the DLL, so it waits for a natural
pause regardless.

---

## 12. Risks

1. **The name rules are heuristics over an author convention.** `<N>[suffix] - <Name>` holds on all 33 shipped
   levels and was re-derived independently, but a future level could name rooms differently and silently
   produce a short or missing chain. Mitigations: the fallback only runs where there is no gate ladder; a
   missing file is today's behaviour; §7.1 asserts the exact shipped level set, so a regeneration that loses a
   level fails a test rather than a run. **Revision 2 added two name rules** (the `S` suffix, `P Door`) on a
   sample of exactly three campaign-wide occurrences; that is a small sample and it is why §7.1 test 6 pins
   both regressions by name.
2. **The trunk is coarse where the level branches, and 8-3 and 1-3 are the price.** `GateProgress` has one
   `best_hops` and cannot express "both required" (§2.6). The right extension is a per-rung `requires` set —
   the activation graph's own field, already computed in
   `…\scratchpad\route\activation-graph\routes\route_*.json`, **do not re-derive it** — plus a `best_hops`
   that only advances when a rung's prerequisites are satisfied. That is a change to `GateProgress`'s core
   invariant and it is out of scope here; it is the first thing to build after S3, and 1-3 is its test case.
3. **A game update invalidates the data.** I5 catches a moved FinalPit within 5 m and disables the fallback
   for that level. It does **not** catch a level whose rooms moved but whose pit did not. `build_routes.py` is
   one command and ~4 min, and should be run after every game update alongside `campaign_check.py`.
4. **A bad rung is worse than no rung, and there is no live evidence yet.** Nothing in this spec has been
   tested in game; every number is parsed from the bundles or reproduced offline. Revision 2 removed the two
   worst classes of bad rung (unreachable, §2.7; out-of-order across a branch, §2.6) with guards rather than
   with prose, but "the leg is traversable" remains unmeasurable offline (§2.8). The recovery path is
   deliberately trivial: set `"rungs": []` in one level's file and that level falls to layer 3 with no code
   change and no retrain. `route_source` is the metric that says which layer fired, which is why §7.4 exists.
5. **1-1 is the next level to train and its gate ladder may be collapsed** by the end room's `activatedRooms`
   listing the spawn field. Falsifiable: if a 1-1 run shows `gates_reached` saturating in the first seconds
   while the agent wanders, that is this bug, and the room trunk for 1-1 is the fix already in hand. It would
   be a *replacement* on a gates level, which this spec does not authorise.
6. **`gate_approach` dominance on the long levels.** 8-3 is 785 of approach plus 180 of ladder against
   `level_complete` 100 — **9.7×**, down from revision 1's 15.6× but still the largest number here, and
   `gate_approach` is re-earned every episode. There is deliberately no per-level lever (§6). **Tripwire:** if
   `part_gate_approach` exceeds 6× `part_level_complete` over a 50-episode window on any fallback level while
   the completion rate stays 0, regenerate that level's file with a lower `MAX_RUNGS` before touching any
   weight — the data lever is reversible and level-local, the weight lever is neither.
7. **The seeding rule (§6) is the only behaviour change to `GateProgress` logic**, and it is guarded to the
   rooms path by object identity. It fires **once per level load**, not per respawn (§6 corrects revision 1
   here). Its 150 m tolerance is calibrated against 0-1's real spawn and is unvalidated on the other 11
   levels, because the offline parse has no per-level spawn (§2.8).
8. **Checkpoint coverage is a diagnostic, not a guard**, and the leg-witness measure inherits that. 8-4 ships
   at 0/4 legs with four sane rungs and 7-1 at 6/10 with a fixed order, so no threshold is enforced on either;
   they are numbers to point at when a level wedges.
9. **S3's stamp is the one place a data error becomes a permanent wedge.** `_is_reached` refuses a gate
   carrying `needs_item`, so a `gated_by` on a rung that is not actually locked makes that rung unreachable
   for the whole run. §7.5's resolve check is blocking for S3 for that reason, and the stamp is written to
   fail safe: no live altar match means no stamp means today's behaviour.

---

## 13. Review dispositions

Both reviews were re-measured before being accepted. Scripts named are the ones run for this revision.

### 13.1 Accepted, and the spec changed

| # | claim | verification | change |
|---|---|---|---|
| rl-1 | the name-sorted order breaks on every branching level | **Confirmed**, `rev/sim.py` re-run: 1-4 entering `3TR` first pays 4 instalments at once then 0, 0, 0; 1-3 and 8-3 the same shape. Instalment totals identical in both orders, so it is signal death, not extra reward | §2.6 (new), invariant **T** (§4.3), §10 recount, A0 (§8). 1-3 and 8-3's "gained" verdicts restated |
| rl-2 | the route crosses skull locks the fallback cannot solve; `_protect_carry` goes inert | **Confirmed with a correction.** `_subgoal` does return a room rung untouched, and `_protect_carry`'s `carrying` test is `target.get("subgoal") == SUBGOAL_ALTAR`, never true on a rung — but its *filled-altar* arm still fires, so only the carry half goes inert. Re-measured the exposure: after T it is **4 shipped levels, not 8** (`lockprobe.py`) | §4.5 (new): the lock is measured per level, `reaches` column in §10, and stage S3 closes it by stamping `needs_item` live from `campaign.altars` — neither of the review's two options, because a static `needs_item` would wedge the rung permanently |
| rl-3 | connectivity is never measured | **Confirmed as a gap.** Rejected the two proposed guards: `contained_frac` tests the doorway convention, not reachability (it is 1.00 on 0-1 and low on levels whose rooms are authored as siblings), and a knee-height segment test is `partial` on 0-1 itself | §2.8 (new): per-rung standability (R4, now a guard, 92/92), leg witness (57/92), tour ratio; `legs_witnessed` in the schema and a `legs wit.` column in §10; and an explicit statement of what cannot be measured offline |
| rl-4 | I6 is under-inclusive; 6-2 passes R1 only via the Prime door | **Confirmed.** Scanned all 33 `*.anc.json`: the `<N>S` suffix occurs exactly three times — `6S - P Door` (3-1), `1S - P Door` (6-2), `10S - Secret Arena` (8-3) — all optional | I6 extended to the `S` suffix and `P Door`/`Prime`; the `checkpoint` carve-out deleted. **6-2 falls to layer 3**; 7-4 loses `3B`, which R4 independently found unreachable |
| rl-5 | `route_source` as a string in `CAMPAIGN_INFO_KEYS` is logged as `None` | **Confirmed** at `env.py:1131-1135` (the invariant, stated for `info["level"]`), `progress.py:51` (`EPISODE_LOG_RAW`, the escape hatch) and `progress.py:54-59` (`_num`) | integer `route_source` in `CAMPAIGN_INFO_KEYS`, string `route_source_name` in `EPISODE_LOG_RAW`, and §7.4 asserts the invariant so it stops being prose |
| data-1 | five shipped rungs are outside `_is_reached`'s cylinder; 5-2 is a hard wedge | **Confirmed independently** (`spec-rev/reach_verify.py`, own run over all 14 files): 5-2 `8 - Ship` no standable cell within ±40 m, 4-2 `5 - Temple Entrance` 19.8 m, 7-4 `0 -`/`1 -` zero triangles, 7-4 `3B` 11.5 m. Control: 0 unreachable over 60 gates on 0-1/0-3/4-3/6-1/8-1, dxz 0.4 m on every one | **R4 promoted from revision 1's §7.4 "never a guard" to a hard guard**; §2.7 (new); 5-2 8→7 rungs, which also fixes the never-target-the-exit wedge |
| data-2 | §7.1 test 5 fails on 8-3 because R2 is judged pre-cap | **Confirmed** (`review-data/r12_invariants.py` re-run): 8-3 stored 1.264, shipped 24 rungs recompute **1.421**, over R2's 1.35; all 13 other levels reproduce exactly | order of operations reversed to **… I3 → R2**; §7.1 test 5 reproduces from the shipped `pos`. Moot for 8-3 in the end — T cuts it to 13 rungs at tour 1.247 and the cap no longer binds anywhere |
| data-3 | the boss block cannot read what it reads | **Confirmed.** `decompiled/BossHealthBar.cs:13-14` (`[HideInInspector] public IEnemyHealthDetails source`), `:28-43` (`Awake` assigns it and fills `healthLayers`), `:109-115` (`DisappearBar` only calls `ExpireImmediately`; nothing is destroyed). Scene scan: **every** campaign BossHealthBar is `activeInHierarchy == false` at load; 0-5's second Cerberus, both 7-4 bars and all four 6-1 bars ship `healthLayers` empty | §5.2 rewritten: per-bar try/catch, skip null `source`, `health_max` deferred to the first scan with a live `source` and `null` until then, `dead` named as the only end-of-fight signal, `key` flagged as not an identity |
| data-4 | the stuck-clock clause fires on 0-1 and can fail the merge gate | **Confirmed**: 9 of the 18 gates levels carry a bar; 0-1's is `secondaryBar = 0`; `fought` at `env.py:1024` already includes `style` | the clause is **removed from the merge** (§5.4) and deferred to S4 gated on `route_source == "rooms"`. This restores §1's byte-for-byte claim in full |
| rl-nb | §6's second rationale is false: `best_hops` survives a respawn | **Confirmed** by reading `mark_paid` / `new_level_load` (`campaign.py:618-631`) | §6 corrected; the rule stands on its first motivation |
| data-nb | `_respawn` calls `mark_paid` even after `StatsManager.Restart` reloaded the level | **Confirmed** at `env.py:892-906` | fixed in S2; `test_route_reload_without_checkpoint_reseeds` added |
| rl-nb | 7-1 is misordered inside one scope; wave containers cause it | **Order confirmed** from the shipped file itself (`1, 3, 4, 5, 2, …`, room 2 after room 5). The wave-container cause is consistent with §4.3 I2's own record of the 3-way merge at one point, and is recorded as the likely cause rather than as a measurement | `ActivateNextWave` containers excluded from the room inventory (§4.3 I2). Measured effect: tour 1.039 → 1.000, polyline 1 613 → 1 553 m, median gap 100 → 80 m |
| rl-nb | a mid-load layer flip would pay the hops difference | **Confirmed** by reading `_gates`'s per-call evaluation and 7-2/8-3's third-arm status | `_hops_source` guard in §4.4; `test_route_source_flip_clears_the_ladder` |
| rl-nb | I5's stale path holds the stale target instead of falling back | **Confirmed**: `_choose_target`'s `if not active: return self.target` | I5 clears `target` and `best_dist` on refusal; A5 restated to run on the edited level |
| rl-nb | `_is_room_ladder` subscripts `None`; `target` aliases an `lru_cache`d document | **Confirmed by inspection** of the revision-1 pseudocode | both fixed in §4.4: `self._route is not None and …`, and `load_route` deep-copies per instance |
| rl-nb | §6 understates the `gate_approach` ceiling by omitting the final leg | **Confirmed** (`review-data/r15_reward.py`, and recomputed on the trunks) | §6's table now reports polyline **plus** last leg per level; 8-3 785 (9.7×), was 1 195 (15.6×) on the 24-rung ladder |
| data-nb | target slots saturate on long legs | **Confirmed** at `spaces.py:170-177` (rel/50, dist/100, per-axis clip ±4.0). Re-measured on the trunks: **33 of 92 legs over 200 m, 11 over 400 m** | stated in §4.1 as a known property with the numbers; **not fixed**, because re-scaling changes a learned input column's meaning on the 18 gates levels |
| data-nb | 7-4's boss rewards are mis-attributed | **Confirmed**: the Brain's trigger is an `ActivateNextWaveHP` with `lastWave false` and empty `toActivate`/`doors`; `ActivateNextWaveHP` is a separate MonoBehaviour; `CampaignPatches.cs:78` patches `ActivateNextWave.EndWaves` only; 7-4's one `lastWave` with `doors=['Quakedoor']` is the earlier arena at `0,458.5,649.8` | §5.1 and §5.3 corrected; 7-4 moved into the 2-4/3-2 group and into the `boss_down` revisit trigger |
| data-nb | A4 is check 7, and its teleport cannot confirm a live bar | **Confirmed**: `campaign_check.py:47` `NAMES` already has six; the teleport limitation is the documented reason check 5 FAILs | A4 renumbered and restated as a well-formedness check plus a read-only play-through |
| data-nb | `Pit` is the hops-0 rung on 1-3, 7-4 and 8-4 at exactly 62.2 m | **Confirmed** from the files (identical `last_rung_to_exit_m`); it is the FinalPit prefab's root, and the probe finds standable floor at all three | §7.1 test 9 pins the exception by name and distance, so it cannot be mistaken for the previously-rejected "make `Pit` the goal" rule |
| data-nb | rung positions are load-time transforms and some rungs are movers | **Confirmed** from the names (2-4's tram, 4-4's two elevators, 7-1's tram) | §11.2 |
| data-nb | boss `key` is not an identity | **Confirmed**: the bar lives on the enemy; 0-5's two Cerberus bars start 24 m apart | §5.2 |
| data-nb | §2.2's headline is over a differently-defined set | **Confirmed**: revision 1's 50/62 was over a 15-level set including 5-4 | §2.2 restated, and labelled as a verdict on the rule rather than on what ships |
| both | the live configs never exercise layer 2, so A2 is weak pre-merge evidence | **Confirmed**: `campaign_gates_prelude.yaml` names 0-1/0-3/0-4, none of which has a file | §9's closing note; **A0 added** as the offline branch-order regression test; A2 relabelled as weak |

### 13.2 Verified and rejected

| claim | why it is rejected |
|---|---|
| rl-nb: **renumber so the lowest rung is hops 1**, and `_choose_target` will fall through to `_exit` earlier | **It is a no-op.** Run against `_choose_target` on 8-3's file in both numberings: with rungs `n…0`, `best_hops == 1` targets the hops-0 rung and `best_hops == 0` returns the exit; with rungs `n…1`, `best_hops == 2` targets the hops-1 rung and `best_hops == 1` returns the exit. In both cases the exit becomes the target only after the lowest rung is physically touched. The underlying gap — the long unaided final leg — is real and is §10's `last rung → pit` column; renumbering does not close it |
| rl-3 alternative: ship `contained_frac` from `doorway.json` as guard R4 | It measures whether a rung's pivot lies inside the *previous* room's AABB, i.e. the doorway convention, not whether the rung is reachable. A pivot can be perfectly standable and on route while sitting outside the previous room's box whenever rooms are authored as siblings rather than nested. R4's voxel standability test is a direct test of the condition `_is_reached` actually evaluates, and its control scores 60/60 on ladders proven in game |
| rl-2 option (a) as stated: **have `build_routes.py` write a static `needs_item` on a locked rung** | `_is_reached` returns false for any gate carrying `needs_item`, and a static field never clears. The rung would become permanently unreachable and `best_hops` would freeze above it — a hard wedge, strictly worse than no stamp. §4.5's S3 stamps it *live* from `campaign.altars` instead, which is what the mod itself does for a gate |
| rl-2 option (b): **refuse to emit any level whose chain crosses an `ItemPlaceZone`-driven door** | Measured after T the exposure is 4 levels, not 8, and on all four the trunk is correct up to the lock — 1-4 gets 3 of 4 rungs, 7-1 gets 10. Dropping them would trade a partial gain for nothing on levels that have no signal at all today |
| rl-1 option (b): **reject any chain with more than one suffix on an ordinal** | Over-broad. 4-2's `6 → 6A → 6B` is a chain, not a star, and 5-2's `7B` and 8-3's `10B` are lone suffixes; rejecting on the name alone would drop 4-2 and cost three more levels than the structural test. Invariant T rejects a *set of ≥ 2* suffixed rooms at one ordinal and keeps a lone suffix, which is the distinction the data supports |
| rl-nb: a per-level `gate_approach` scale in the route file | It is a reward change, and "no change to `rewards.py`" is the property that makes this spec safe to merge beside a live run. The reversible, level-local lever is `MAX_RUNGS`/the trunk in the data; §12.6 makes that the tripwire's first action |

### 13.3 Verified correct, so a later reader does not re-litigate them

From the `data` review, re-checked and confirmed: §10 of revision 1 reproduces byte for byte
(`r07_reproduce.py`, 14 files identical); the gates guard split reproduces from the survey JSON (PASS 18 at
ratios 1.000×14, 8-2 0.960, 4-3 0.600, 8-1 0.538, 6-1 0.500; FAIL 15), so 4-3, 6-1 and 8-1 stay on layer 1 and
no route file exists for any of the 18; the slot-12 clamp is a proven no-op (campaign max gate hops 13, longest
trunk max hops 14); §2.1's 0-1 claim reproduces exactly (11/11 within 10.0 m, median 3.0 m); all route files'
`exit.pos` match the survey's reproduction of the mod's FinalPit rule to 0.0 m, including the four levels with
4-6 pits; `key` == Python `round(pos)` matches `Mathf.RoundToInt`; `BossHealthBarTemplate` is not a
`BossHealthBar` subclass; and §5.1's non-7-4 rows check out (0-5's two `lastWave` at `174.5,-6,382` →
`DelayedDoorActivation` / `5 - Final Hallway`; 4-2's at `8,-15,1157.5` → `FinalDoorOpener`,
`CheckPointsReEnabler`; 5-2's at `87.5,-53,1240` → `FightEnd`; 2-4/3-2/6-2 have no `ActivateNextWave` at all;
6-2's `Door "ExitRaiser"` at `-295,7.5,350` → `FinalRoom`; six of seven end with a one-room `DoorLeft` whose
single `activatedRooms` entry is `Pit`).
