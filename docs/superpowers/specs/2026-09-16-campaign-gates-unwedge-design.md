# Campaign gates, un-wedge and look modes: design

Date: 2026-09-16. Status: approved, revised against three adversarial reviews (§14). Mod version target: **0.6.0**.
Protocol version stays **1** (obs gains fields; no request or reply is removed or changed).

Follows `2026-09-16-campaign-foundation-design.md`. Driven by the five-investigator diagnosis of the 0-1 pilot
(0 completions in 2.9M steps), each report adversarially verified.

**The mod and the Python side are implemented in parallel by two engineers who cannot talk to each other.** The JSON
interface in §3 is the contract between them. Both sides must degrade gracefully when the other has not landed yet
(§3.7), so either can be merged and run first.

Every number in this spec that is called *measured* was re-measured against the code or the diagnosis data while
revising it. §14 records which review claims were adopted, corrected or rejected.

---

## 1. Goals and non-goals

### Goals

| # | Goal | Done when |
|---|---|---|
| G1 | The absorbing `slowMode`/`heavyFall` movement state cannot hold the agent | Both wedge flavours recover within 1 s of game time, with no teleport and no enemy damage; sampled-policy vent pass rate ≥ 80% over 15 fresh loads (§11.2 A9) |
| G2 | A route signal exists that is not the NavMesh | `campaign.gates` orders 0-1's 11 doors 0..9 hops from the exit, live |
| G3 | The route signal drives the reward and the observation | `gate` + `gate_approach` + `door_unlock` are the largest positive reward group for a policy that walks the route |
| G4 | The agent can point the camera at an enemy or at the route without learning to aim first | `look_mode` exists, is learned, and the run continues from the current weights with a measured, bounded policy displacement |
| G5 | Failures are visible without a live watcher | `runs/<run>/episodes.jsonl` exists; `gates_reached`, `wedged_steps`, `level_started` reach the dashboard |
| G6 | The trainer stops burning 11 cores on spin-wait and logs on resume | `KMP_BLOCKTIME=1`, `validate_args(False)`, `verbose=1` |

### Non-goals (out of scope for this spec)

More than 5 games; multi-level training; skull keys and item gates (sub-project 3); RecurrentPPO or any policy
architecture change; mod timing instrumentation; PPO hyperparameter changes (measured median `approx_kl` 0.020 —
leave `learning_rate`, `target_kl`, `n_epochs`, `gamma`, `ent_coef` alone); a NavMesh triangulation dump (rejected:
0-1's baked mesh is 412 polys in 47 islands with nothing before z 378.5, and no bridging rule connects start to exit
below a 17 m gap / 12 m climb limit, which would also bridge through walls); a straight-line distance-to-exit reward
(0-1's exit is *nearer* to spawn than to 5 of its 6 checkpoints); a scripted opener in `_campaign_reset`; changing
the Cyber Grind action space (§7.1).

---

## 2. What the design rests on

Verified findings, condensed. Full reports:
`C:\Users\tyler\.claude\projects\F--Github-ULTRAKILL-AI\02d63a84-4e41-4935-b073-3ab480256b6d\subagents\workflows\wf_e0f27ec6-ab7\journal.jsonl`.

- **F1 WEDGE.** ~70% of sampled fresh loads (8/26 passed at 30 fps / frameskip 2, 3/15 at 60/4) end in an absorbing
  state. A slide that ends while the player is airborne, or a jump out of a slide, where the game's stand-up test
  fails, sets `slowMode` (`NewMovement.HandleSlideState`, decompiled lines 921-928). Afterwards `grounded` is never
  true, velocity reads `(0,-100,0)` (a ground slam that never lands) or ~0, stamina is frozen, jump/dash/slide do
  nothing, and only a 0.477 m/s creep remains. Mechanism confirmed from the code: stamina regen is gated on
  `!sliding && !slowMode` (:755), dash on `!slowMode` (:1036), slide start on `(!slowMode || crouching)` plus ground
  (:1031), walk speed is `slowMode ? 1.25 : 2.75` × `walkSpeed` (:1332), and `slowMode` is set at exactly one place
  (:927). It happens in 0-1's slide vent (x = 40.0, z 349 → 377.5, 1.0 m wide), at the spawn-room wall, and at
  (145.4, 45.5, 664.2) after checkpoint 3 — it is a general state, not a vent property. Scripted **grounded** slides
  pass the vent every time (22 decisions, 1.5 s).
  Re-measured on the 32,022-decision live log (`scratchpad/diag/run_steps.jsonl`, 7 episodes): exactly **five**
  runs of ≥ 45 consecutive wedge-signature decisions exist (2976, 1108, 729, 694, 672), and nothing else exceeds 29.
- **F2 NO ROUTE SIGNAL.** The baked NavMesh is an enemy-walking mesh. `path.status` is never `complete`, so the
  `path` reward never paid in 656 logged rows. But `Door.activatedRooms` (`GameObject[]`, decompiled `Door.cs:72`)
  forms a room graph, and a BFS from the room holding the `FinalPit` orders every door in 0-1, 0-2, 0-3, 0-4 and
  1-1. It fails in 0-5 (no room node is an ancestor of the exit pit). Doors are proximity-opened (DoorController
  type 0, 8×6×6 m trigger) unless an `ActivateArena` holds them locked; the last `ActivateNextWave` unlocks.
  Re-derived from the scene files with the exact rules of §3.1 (`scratchpad/diag/modside/bfs2.py`): 0-1 → 11 gates,
  hops 0..9, identical to §3.6; 0-2 → 17 gates, hops 0..7; 0-3 → 12, 0..6; 0-4 → 7, 0..5; 1-1 → 13, 0..5;
  0-5 → 3 gates, `gates_ordered: false`.
- **F3 COMBAT GATES DOORS AND THE AGENT CANNOT AIM.** The campaign config sets no `pitch_limit_deg`, so the camera
  drifted to a mean of −78° (`on_target_frac` 1.5%). All three live episodes that passed checkpoint 3 died in the
  Stray arena between locked doors (66,21,640) and (146,31,640) with 0-1 kills in 35-193 s.
- **F4 SMALLER.** The stuck clock never resets on kills and `_respawn` does not reset it; the ground run never saved
  an exploration archive; there is no per-episode log; `_ground_point` takes the minimum of an 8-ray 4 m ring
  because the mod has no centre ground ray (measured ring spread among hits: p50 0.5 m, **p90 10.8 m**);
  `ChooseExit` can pick a secret-level `FinalPit` in 0-2 and 1-1; the trainer's OpenMP threads spin-wait on 11.0
  cores (`KMP_BLOCKTIME=1` → 1.43 cores, same update time); `Distribution.set_default_validate_args(False)` saves
  0.95 ms of the 3.1 ms forward pass (~2.9% of wall time); the resume path passes no `verbose`, so resumed runs log
  nothing.

**Correction to one premise of the plan, re-measured.** D3 assumed the NavMesh path slots "have read zero/none for
the whole run". They have not. Over the 32,022 logged decisions of `campaign_ppo_ground/latest.zip` playing 0-1,
`path.status` was **`partial` on 26,318 rows (82.19%)** with a `next_corner` on every one of them, and `none` on
5,704 (17.81%). So inputs 448-455 have carried real, varying values for that run's ~465k steps. Measured on
`models/campaign_ppo_ground/latest.zip` (first-layer column norms, both hidden stacks):

| abs index | block | what it carried | policy-stack column norm | value-stack column norm |
|---|---|---|---|---|
| 448-450 | 5-7 | path next corner rel xyz / 50 | 1.023, 0.698, 0.920 | 1.043, 2.866, 1.099 |
| 451 | 8 | path next corner dist / 50 | 0.755 | 2.170 |
| 452 | 9 | path length / 300 | 0.570 | 2.677 |
| **453** | 10 | status `complete` — **never once true** | **0.000** | **0.000** |
| 454 | 11 | status `partial` (mean 0.821) | 0.476 | 2.970 |
| 455 | 12 | status `none` (mean 0.179 — **not constant**) | 0.801 | 2.304 |

`‖W[:, 448:456]‖_F` = 2.04 against `‖W‖_F` = 33.78. The slot reuse stands (D3), but it is **not free**, and §7.3
states the measured cost and the mitigation instead of claiming bit-identity.

---

## 3. JSON interface (the mod ↔ Python contract)

Everything below is added to the existing `campaign` block and obs, in `mod/UltrakillAIBridge/Obs/`. No existing
field changes meaning, type or units.

### 3.1 `campaign.gates`, `gates_ordered`, `gates_truncated`

Present whenever the `campaign` block is present.

| field | type | units / convention | when |
|---|---|---|---|
| `gates_ordered` | bool | `true` when a goal room was found and at least one gate has a non-null `hops` | always |
| `gates_truncated` | bool | `true` when the level had more gate candidates than fit the cap | always |
| `gates` | array (may be empty) | at most **64** entries, sorted by `hops` ascending with `null` last, then by `key` ascending | always |
| `gates[].key` | string | `CampaignPatches.Key(closed world position)` — the **existing** position-key convention, `"x,y,z"` whole metres, plus a `#2`/`#3` suffix on collision (rule 9) | always |
| `gates[].pos` | `[x,y,z]` floats | the **closed** world position of the `Door` component's own GameObject, metres (rule 8) | always |
| `gates[].hops` | int or `null` | rooms still to traverse **after** passing this door: the minimum room-hop count over the door's `activatedRooms`. `0` = the door into the exit's room. `null` = this door is not in the exit's connected component | always |
| `gates[].open` | bool | `Door.open` — "currently open or opening", **not** "has been passed" (rule 11) | always |
| `gates[].locked` | bool | `Door.locked` | always |
| `gates[].active` | bool | `door.gameObject.activeInHierarchy` | always |
| `gates[].controller_active` | bool | `true` when at least one of the door's `DoorController`s is `activeInHierarchy` (rule 10) | always |

Rules:

1. A door is a **gate candidate** iff its `activatedRooms` holds ≥ 2 distinct non-null GameObjects. Room templates
   are excluded by the existing `IsTemplate` ancestry test.
2. **Room nodes** are exactly the GameObjects that appear in some gate candidate's `activatedRooms` — nothing else.
   (This matters: 0-1, 0-2, 0-3, 0-4, 0-5 and 1-1 each have a GameObject literally named `Pit` on the exit pit's
   ancestor chain, and on several of them a *one-room* door references it. Admitting rooms from non-gate-candidate
   doors would make `Pit` a room node and break rule 4.)
3. **Node identity** is `CampaignPatches.Key(room.transform.position)`, the same "x,y,z" whole-metre convention as
   `gates[].key`. Reference identity must **not** be used: `CheckPoint.Start` clones each of its `rooms` entries,
   deactivates the original and moves it +10,000 m in X (`decompiled/CheckPoint.cs:108-132`), and `ResetRoom()`
   destroys and re-instantiates the clone on every respawn, so the same physical room is a different object across
   a respawn. The position key separates the banished templates automatically (+10,000 X) and is stable across
   re-instantiation.
4. **Goal room:** walk up the chosen `FinalPit`'s transform chain, starting at the pit itself, and take the **first**
   transform that is a room node. That single node is the goal room (`hops` 0). If no ancestor is a room node, the
   level is unordered (rule 7). Never "every room under a common root": a level whose rooms sit under one container
   would otherwise report every gate at `hops` 0, which looks valid and is meaningless.
5. **Adjacency:** rooms A and B are adjacent iff some gate candidate lists both.
6. **BFS** from the goal room over that adjacency gives each room its `hops`.
   `gate.hops = min(room.hops for room in gate.activatedRooms if room has hops)`, or `null` when none has one.
7. When there is no goal room (0-5), or `ChooseExit` returns null, every gate reports `"hops": null` and
   `gates_ordered` is `false` — **unless** a previous `Scan()` in this level load produced a hops map, in which case
   that map is re-applied by `key` and `gates_ordered` stays `true`. (0-1's live `FinalPit` is inside a
   checkpoint-owned room — `13 Content` belongs to `Checkpoint (2)` — so a pre-boss respawn destroys and recreates
   it, and a `Scan()` that lands mid-recreation must not blank the route.)
8. **`key` and `pos` are the CLOSED position and are frozen for the level load.** `Door.Update` moves the `Door`
   component's **own** transform when `doorType == Normal`
   (`base.transform.localPosition = Vector3.MoveTowards(..., targetPos, ...)`, `Door.cs:341-351`, where
   `targetPos = openPosRelative = closedPos + openPos`, `Door.cs:309-318`). Measured from the scene files: **every**
   gate candidate on 0-1 (11/11), 0-2 (17/17), 0-3 (12/12), 0-4 (7/7) and 0-5 (3/3) is `doorType 0` with
   `openPos = (0, 5.75, 0)`; only 1-1 is immune (13/13 are `doorType 1`, `openPos` zero, and `Door.Update` returns
   early for non-Normal). A rescan while the player stands in the 8×6×6 proximity trigger would otherwise capture
   the open position and flip the pre-boss key `"202,56,432"` → `"202,62,432"`. Read it as:
   ```csharp
   Vector3 p = door.gotPos
       ? (door.transform.parent != null ? door.transform.parent.TransformPoint(door.closedPos) : door.closedPos)
       : door.transform.position;
   ```
   `gotPos` (`Door.cs:22`) and `closedPos` (`Door.cs:25`, a **local** position set in `GetPos()`) are both public.
   A door whose `Awake` has not run is still sitting at its closed position, so the fallback is correct. This also
   keeps `key` compatible with `CampaignPatches.RecordDoorUnlock`, which keys `Door.Unlock` while the door is closed.
   The `Door Object` child carries the `NavMeshObstacle` and is **not** what moves — the old spec had this backwards.
9. **`key` must be unique within one `gates` array.** `CampaignPatches.Key` rounds to whole metres, so two doors
   within a metre collide. On a collision append `#2`, `#3`, … in scan order and log one warning per level load.
   Python uses `key` alone to detect a target change, so a duplicate would freeze the target.
10. `controller_active` disambiguates the `(open: false, locked: false)` pair at load. 0-1's gun-room gate reports
    exactly that, but it cannot be opened: its `DoorController` is inactive in the hierarchy until
    `ActivateArena.Activate` runs, which then calls `door.gameObject.SetActive(true)` and `door.Lock()`. Without
    this flag "walk up to it" and "a fight gates it" are indistinguishable.
11. `open` is transient. `DoorController.Update` opens on `playerIn || enemyIn` and calls `Close()` as soon as both
    go false; enemies open doors (`dc.Open(enemy: true)`); and `Door.Open()` force-closes every other open door that
    has a type-0 controller and no enemy in it (`Door.cs:492-514`), so the door behind you shuts when you open the
    next one. It means "currently open or opening", never "has been passed".
12. **Statics vs per-step.** `key`, `pos`, `hops`, and each gate's `JObject` skeleton are computed inside the
    existing `Scan()` (which already runs on a new scene, after a respawn, on a checkpoint state change and every
    `RescanEvery` = 30 builds) and cached; the BFS is recomputed on every `Scan()` and **`hops` for a given `key`
    must be stable across those recomputations within one level load**. Only `open`, `locked`, `active` and
    `controller_active` are read per step, assigned into the cached `JObject`s. The array is sent in full every
    step: ≤ 64 entries is ≈ 3 KB of JSON against an obs line that is already 3.3-7.1 KB and a 32.9 ms lockstep
    round, and a stateless wire format removes every cache-coherence failure mode between the two implementations.
    Pre-building the objects keeps the allocation cost near zero at 150-180 steps/s across 5 games.
13. **The cap.** Measured gate-candidate counts: 0-1 11, 0-2 17, 0-3 12, 0-4 7, 0-5 3, 1-1 13; total `Door`
    components 33/28/20/20/7/24. 64 is well clear. If a level ever exceeds it, keep the 64 gates **nearest the
    player** (not the 64 lowest `hops`, which would drop the start-of-level gates the agent needs first) and set
    `gates_truncated: true`.

### 3.2 `campaign.exit`: exclude secret pits and pick the next mission (`ChooseExit` fix)

`CampaignObserver.ChooseExit` currently returns the first active `FinalPit`, else the first inactive one in
`FindObjectsOfType` order. Measured from the scene files: 0-2 has a real pit targeting `Level 0-3` and one targeting
`Level 0-S`; 1-1 has one targeting `Level 1-2` and two targeting `Level 1-S`. All are inactive at load, so the
choice is arbitrary today.

New selection, in order:

1. Discard `fakeEnd`, `secondPit`, `rankless`, templates (unchanged), **and** any pit whose `targetLevelName` is
   empty or ends with `-S` (ordinal, case-insensitive).
2. Prefer a pit whose `targetLevelName` is the **mission successor** of the current scene. Parse both as
   `"Level <a>-<b>"` with `a`, `b` integers; a target `(c,d)` is a successor of the current `(a,b)` iff
   `(c,d) == (a, b+1)` or `(c,d) == (a+1, 1)`. Scenes or targets that do not parse simply do not rank.
   **No 35-name mission table.** The game reviewer re-ran the pit parser: after step 1, exactly **one** pit survives
   on each of 0-1, 0-2, 0-3, 0-4, 0-5 and 1-1, so the table would be dead weight duplicating
   `python/ultrakill_ai/campaign.py: CAMPAIGN_LEVELS`; the successor rule is a 6-line total order that needs no
   table and resolves 0-5 → 1-1 correctly.
3. Then prefer an active pit over an inactive one; then `FindObjectsOfType` order.
4. **Freeze the choice for the level load.** Rule 3 is time-varying (a pit's room activates mid-run), and every
   `hops` value depends on which pit was chosen, so re-running it mid-load could silently renumber the route. Keep
   the chosen instance until it is destroyed or the scene changes; on destruction re-run 1-3, and if that
   transiently yields nothing, rule 7 of §3.1 keeps the previous hops map.
5. `Level 9-2` has no successor and no non-secret target; it falls through to rule 3, which is correct (one real
   pit). If step 1 leaves nothing, `exit` is `null` and `gates_ordered` is `false`.

0-1 is unaffected (a single real pit, target `Level 0-2`).

### 3.3 `ground_ray_center`

New **top-level** obs field, sibling of `ground_rays`:

```
"ground_ray_center": 1.62
```

A single downward raycast from `player.pos + up * 1` along `-up`, same environment mask and
`QueryTriggerInteraction.Ignore` as `BuildGroundRays` (`ObservationBuilder.cs:251-262`), reporting
`hit.distance - 1f`, and reporting `ground_ray_length` **exactly** when it hits nothing — identical convention to
the ring, including negative values when the ground is above the player's feet.

> **Do not add it to the `ground_rays` array.** `ObsLayout.ground_rays` is 8 and the packed vector is 479. A 9th ring
> value would change the observation size and invalidate every checkpoint. `ground_ray_center` is read by
> `env._ground_point` only and is never packed.

### 3.4 `player`: raw movement flags

Added to the `player` object (all always present once the mod is 0.6.0):

| field | type | source |
|---|---|---|
| `slow_mode` | bool | `NewMovement.slowMode` (public, `NewMovement.cs:177`) |
| `heavy_fall` | bool | `NewMovement.gc.heavyFall` (public getter on `GroundCheckGroup`) |
| `crouching` | bool | `NewMovement.crouching` — **private** (`NewMovement.cs:154`), read with `AccessTools.FieldRefAccess`, resolved lazily and degrading to `false` with one logged warning if a game update renames it (same pattern as `ObservationBuilder.GetAnw`) |

### 3.5 What Python must never do

- Never recompute a `key` from `pos`: `Mathf.RoundToInt` rounds halves to even and 0-1's doors sit on exact `.5` z
  values (`408.5 → 408`, `469.5 → 470`). Keys are opaque strings, including any `#N` suffix.
- Never treat `open == true` as "passed", or `locked == false` as "passable" (use `controller_active`).
- Never assume `gates` is the whole level (`gates_truncated`).

### 3.6 Literal example: `Level 0-1`, first obs of a fresh load

Player at the `unlock_all_gear` alternate spawn `(39.7, -0.5, 343.7)`, nothing opened yet.

```json
{
  "type": "obs", "step": 0, "frame": 4181, "time": 9.82, "scene": "Level 0-1", "ready": true, "event": "reset",
  "player": {
    "pos": [39.7, -0.5, 343.7], "vel": [0.0, 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0],
    "forward": [0.0, 0.0, 1.0], "yaw": 0.001, "pitch": 0.0, "hp": 100, "anti_hp": 0.0, "stamina": 300.0,
    "grounded": true, "sliding": false, "dead": false, "activated": true, "level_over": false,
    "slow_mode": false, "heavy_fall": false, "crouching": false,
    "weapon_slot": 1, "weapon_variation": 2, "slot_counts": [0, 0, 0, 0, 0, 0],
    "soft_deaths": 0, "soft_death_instakill": false
  },
  "enemies": [],
  "rays": [3.5, 3.8, 4.9, 4.6, 4.1, 5.7, 5.4, 6.2, 5.7, 6.2, 6.6, 5.1, 4.7, 5.1, 4.9, 4.7],
  "ground_rays": [-0.9, 0.3, 1.2, 0.8, 0.4, 0.8, 1.5, 1.5],
  "ground_ray_center": 0.0,
  "stats": {"kills": 0, "style": 0, "seconds": 0.0, "restarts": 0, "level_complete": false},
  "campaign": {
    "mission": 1, "difficulty": 3, "seconds": 0.0, "timer_running": false, "level_started": false,
    "level_over": false, "restarts": 0, "input_locked": false,
    "exit": {"pos": [202.0, -17.1, 354.0], "active": false},
    "checkpoints": [
      {"id": "40,-2,414", "pos": [40.0, -2.0, 414.0], "activated": false, "current": false},
      {"id": "40,-12,485", "pos": [40.0, -12.0, 485.0], "activated": false, "current": false},
      {"id": "76,18,640", "pos": [76.0, 18.0, 640.0], "activated": false, "current": false},
      {"id": "157,28,640", "pos": [157.0, 28.0, 640.0], "activated": false, "current": false}
    ],
    "path": {"status": "none"},
    "locked_doors": [],
    "arena_enemies_alive": 0,
    "cleared_arenas": [],
    "unlocked_doors": [],
    "gates_ordered": true,
    "gates_truncated": false,
    "gates": [
      {"key": "202,56,432", "pos": [202.0, 56.0, 431.5], "hops": 0, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "202,56,452", "pos": [202.0, 56.0, 452.5], "hops": 1, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "192,31,594", "pos": [192.0, 31.0, 594.5], "hops": 2, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "202,56,534", "pos": [202.0, 56.0, 533.5], "hops": 2, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "146,31,640", "pos": [146.5, 31.0, 640.0], "hops": 3, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "66,21,640",  "pos": [65.5, 21.0, 640.0],  "hops": 4, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "40,11,624",  "pos": [40.0, 11.0, 624.5],  "hops": 5, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "40,-9,552",  "pos": [40.0, -9.0, 551.5],  "hops": 6, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "40,-9,490",  "pos": [40.0, -9.0, 490.5],  "hops": 7, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "40,1,470",   "pos": [40.0, 1.0, 469.5],   "hops": 8, "open": false, "locked": false, "active": true,  "controller_active": true},
      {"key": "40,1,408",   "pos": [40.0, 1.0, 408.5],   "hops": 9, "open": false, "locked": false, "active": true,  "controller_active": false}
    ],
    "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}
  }
}
```

Notes on the example, all reproduced from the level data with §3.1's rules
(`scratchpad/diag/modside/bfs2.py`, `gates_0-1.txt`):

- 11 gates, 10 distinct hop values (0..9). Two gates share `hops` 2 because rooms 11 and 12 are both two rooms from
  the exit — a genuine fork, not a duplicate. **Python must handle a multi-gate tier** (§5): 0-2 has tiers of 3, 3,
  2, 2 and 4 gates.
- The gate at `hops` 9 is the gun-room door, 64.8 m from spawn; the gate at `hops` 0 is the pre-boss door.
- `controller_active` is `false` only on the `hops` 9 gate: its `ActivateArena` ("3 - Gun Room/Enemies/Trigger",
  waves of 3 then 8 enemies) has not run yet, so the door is neither locked nor approachable.
- The checkpoint list is abbreviated (0-1 has 6; `id` values are the existing rounded keys).
- `path.status` is `"none"` at this position; it reads `"partial"` on 82.19% of logged steps.

### 3.7 Graceful degradation (hard requirement, both sides)

Either engineer's work must run against the other's old code.

- Python: `campaign.gates` missing or `gates_ordered` false → `GateProgress` reports no target, the gate reward
  terms pay 0 and observation slots 448-455 read 0. `gates[].controller_active` missing → treated as `true`.
  `player.slow_mode` missing → the wedge detector falls back to the position rule (§9.3). `ground_ray_center`
  missing → `_ground_point` keeps today's ring minimum.
- Mod: unknown keys in a `config` or `action` message are already ignored; no new request type is added.

---

## 4. Observation slot mapping

`ObsLayout(campaign=True).size` stays **479**. Absolute indices (unchanged prefix: player 0-16, enemies 17-416, rays
417-432, ground rays 433-440, Cyber Grind 441-442, campaign block 443-478).

Only block indices 5-12 (**absolute 448-455**) change. Every other index keeps its meaning and its packing formula.

| abs | block | old meaning | **new meaning** |
|---|---|---|---|
| 443-447 | 0-4 | exit rel xyz / 100, dist / 200, mask | unchanged |
| **448** | 5 | path next corner rel x / 50 | target rel **x** in the player's yaw frame / S<sub>rel</sub> |
| **449** | 6 | path next corner rel y / 50 | target rel **y** / S<sub>rel</sub> |
| **450** | 7 | path next corner rel z / 50 | target rel **z** / S<sub>rel</sub> |
| **451** | 8 | path next corner dist / 50 | 3-D distance to the target / S<sub>dist</sub> |
| **452** | 9 | path length / 300 | target mask: 1.0 when a target exists, else 0.0 |
| **453** | 10 | path status `complete` | target gate `open` (1.0/0.0; 0.0 when the target is the exit) |
| **454** | 11 | path status `partial` | target gate `locked` (1.0/0.0; 0.0 when the target is the exit) |
| **455** | 12 | path status `none` | `hops` of the target gate / 20 (0.0 when the target is the exit or absent) |
| 456-460 | 13-17 | nearest pending checkpoint | unchanged |
| 461-465 | 18-22 | first locked door | unchanged |
| 466-469 | 23-26 | arena alive / timer / input lock / seconds | unchanged |
| 470-478 | 27-35 | exploration map | unchanged |

- **Scales.** A **gate** target uses S<sub>rel</sub> = 50, S<sub>dist</sub> = 100. The **exit** target uses
  S<sub>rel</sub> = 100, S<sub>dist</sub> = 200 — the same scales slots 443-447 already use for it, because 0-1's
  exit is ~195 m from spawn and the gate scales would put 448-451 near 4.0, far outside the range the rest of the
  vector uses. Every one of 448-451 is then clipped to ±4.0 as a guard (never reached on 0-1).
- The yaw frame is `spaces.yaw_frame`: +x right, +y up, +z forward, so "ahead" is +z whichever way the player faces
  (pinned by `tests/test_spaces.py`).
- When the target is the **exit** (after the `hops` 0 gate is reached), 448-451 carry the exit's relative vector and
  distance at the exit scales, 452 is 1.0, and 453-455 are 0.0.
- **Threading the target.** `campaign_block` and `pack_observation` derive everything else from the raw obs, but
  `GateProgress` lives in the env. Signatures become
  `campaign_block(obs, explore=None, target=None)` and
  `pack_observation(obs, layout, enemy_max_health, explore=None, target=None)`, with `target` the `GateProgress`
  target dict (or `None`). `target=None` ⇒ slots 448-455 are all 0.0. `env._pack` passes `self.gates.target`.
- `campaign_block`'s docstring must be rewritten; the `path` block is no longer packed at all.
- **Consistency requirement:** the target written into 448-455 and the target used by look mode 2 (§7.2) are the same
  object. `GateProgress.retarget()` runs before every observation is packed (§5 lifecycle), including the one
  `reset()` returns, so slot 452 reads 1.0 on the first decision of every episode on an ordered level and mode 2 is
  never dead on step 1.

---

## 5. `GateProgress`

New class in `python/ultrakill_ai/campaign.py`, alongside `MilestoneTracker` and `PathProgress`, same shape and
lifecycle rules.

```python
REACH_H  = cfg.gate_reach_m        # default 8.0   horizontal
REACH_V  = cfg.gate_reach_v_m      # default 6.0   vertical
MIN_GAIN = cfg.gate_min_gain_m     # default 0.5

class GateProgress:
    # ---- per LEVEL LOAD ----------------------------------------------------
    best_hops  = None    # lowest hops reached
    paid_hops  = None    # lowest hops already paid or absorbed
    reached    = set()   # gate keys ever reached this level load
    # ---- per EPISODE (best_dist: lead ruling R1, see the scope bullet below) ----
    best_dist  = {}      # gate key (or "exit") -> closest 3-D approach made this episode
    target     = None    # a gate dict, the exit sentinel, or None

    # ---- lifecycle ---------------------------------------------------------
    def new_level_load(camp, player_pos):      # fresh level load only
        best_hops = paid_hops = None
        reached.clear(); best_dist.clear(); target = None
        mark_paid(camp, player_pos)

    def mark_paid(camp, player_pos):           # checkpoint respawn, and any reset
        h = _note_reached(camp, player_pos)    # updates `reached` and `best_hops`
        paid_hops = best_hops                  # absorbed, never paid
        # best_dist is deliberately NOT cleared: re-walking after a death INSIDE an episode must not pay again

    def reset_episode():                       # env.reset() only
        target = None
        best_dist.clear()                      # R1: episode scoped, so a new episode re-earns its approach

    def retarget(camp, player_pos):            # pays NOTHING; call before packing any observation
        _note_reached(camp, player_pos)
        target = _choose_target(camp, player_pos)
        k = _key_of(target)
        if k is not None and k not in best_dist:
            best_dist[k] = dist3(player_pos, target["pos"])   # seeded once, never re-seeded

    # ---- per step ----------------------------------------------------------
    def update(camp, player_pos) -> (gates: int, approach_m: float):
        prev_key  = _key_of(target)
        prev_best = best_dist.get(prev_key, inf)
        retarget(camp, player_pos)

        # 1. pay each new lower hops value once per level load
        paid = 0
        if best_hops is not None:
            if   paid_hops is None:     paid = 1
            elif best_hops < paid_hops: paid = paid_hops - best_hops
            paid_hops = best_hops if paid_hops is None else min(paid_hops, best_hops)

        # 2. approach: a new best 3-D closeness to a target that did not change this step
        approach = 0.0
        k = _key_of(target)
        if k is not None and k == prev_key:
            d = dist3(player_pos, target["pos"])
            if d < prev_best - MIN_GAIN:
                approach, best_dist[k] = prev_best - d, d
        return paid, approach

    # ---- helpers -----------------------------------------------------------
    def _is_reached(g, pos):
        if g["hops"] is None or not g["active"]: return False
        dh = hypot(pos[0] - g["pos"][0], pos[2] - g["pos"][2])
        dv = abs(pos[1] - g["pos"][1])
        m  = 2.0 if g["open"] else 1.0
        return dh <= m * REACH_H and dv <= m * REACH_V

    def _note_reached(camp, pos):
        for g in gates:
            if _is_reached(g, pos):
                reached.add(g["key"])
                if best_hops is None or g["hops"] < best_hops: best_hops = g["hops"]

    def _choose_target(camp, pos):
        act = [g for g in gates if g["hops"] is not None and g["active"]]
        if not act:                 return target          # keep the previous one rather than flapping
        if best_hops is None:       return nearest(act, pos)
        if best_hops == 0:          return EXIT(camp) or nearest([g for g in act if g["hops"] == 0], pos)
        lower = [h for h in {g["hops"] for g in act} if h < best_hops]
        if not lower:               return EXIT(camp) or target
        return nearest([g for g in act if g["hops"] == max(lower)], pos)
```

`EXIT(camp)` is the sentinel `{"key": "exit", "pos": camp["exit"]["pos"], "hops": None, "open": False,
"locked": False, "active": True}`, or `None` when `campaign.exit` is null. `_key_of(None)` is `None`.

**Exact call order** (two implementers must not order these differently):

| site | order |
|---|---|
| `reset()` (both modes of `_campaign_reset`) | `_campaign_reset` → `new_level_load` (fresh) **or** `mark_paid` (respawn) → `reset_episode` → `retarget` → pack the observation |
| `_respawn()` (death inside an episode) | `mark_paid` → `retarget` → continue; **no** `reset_episode` |
| `_campaign_progress(raw)` (every step) | `update` → its two values go into `CampaignStep` |

Behaviour this pins down:

- **Empty or unordered `gates`** (0-5, or a 0.5.x mod): no target, both reward terms 0, slots 448-455 read 0, the
  straight-line exit vector at 443-447 is the only route input — exactly today's behaviour.
- **Before any gate is reached** the target is the active gate nearest the player, which on a fresh 0-1 load is the
  `hops` 9 gun-room door 64.8 m ahead.
- **A multi-gate tier cannot be farmed.** `best_dist` is **per gate key**, seeded the first time that key becomes
  the target and **never re-seeded**, so walking A → B → A → B across a tier pays A's approach once and B's once
  and nothing thereafter. The old scalar `best_dist`, re-seeded on every target change, was the exploit: at the
  measured median horizontal speed of 17.1 m/s (1.14 m per decision at 15 decisions/s) the loop netted
  0.0855 − 0.02 = +0.065 per decision, worth +32.5 discounted at `gamma` 0.998, against +1.22 for the `+100`
  completion seen from spawn. It is reachable on 0-1 (its `hops` 2 tier is two gates 66.7 m apart, so the bisector
  sits 33 m from each, well outside any reach radius) and on every tier of 0-2, 0-3, 0-4 and 1-1.
- **`gate_approach` is bounded by the route.** Total over a level load ≤ Σ over distinct targets of the distance at
  which each was first targeted ≈ the gate-to-gate polyline, measured at **723-736 m** on 0-1.
- **`best_dist` is EPISODE scoped (lead ruling R1, overriding this spec's original level-load scope).** It is
  cleared by `reset_episode` — R12's rollback, taken as the default. A death *inside* an episode still does not
  clear it (`mark_paid` leaves it alone), so re-walking after a respawn pays nothing within the episode. The
  `gate` ladder (`best_hops`/`paid_hops`/`reached`) stays level-load scoped exactly as written above, like
  `MilestoneTracker`'s checkpoint/arena/door sets.
  *Rationale for the ruling:* PPO returns are computed within an episode and a truncation bootstraps
  `gamma * V(s_T)` on the same state, so ending an episode early can never be profitable; and most episodes are
  checkpoint respawns, which under level-load scope would get no dense signal at all on ground an earlier
  episode already covered. It is still not farmable — with the per-gate dict an episode's total is bounded by
  the gate-to-gate polyline (723-736 m on 0-1) either way.
- **Skipped hop values** (a lucky shortcut from 9 to 6) pay `paid_hops - best_hops` = 3, one per hop crossed.
- **`hops` shifting without the player moving pays nothing.** `best_hops` only ever falls inside `_note_reached`,
  i.e. because a gate was newly *reached*; a `Scan()` that renumbers the graph cannot itself trigger a payment.
  (§3.1 rule 7 also stops the graph renumbering in the first place.)
- **Reach is a cylinder, not a sphere.** `gate_reach_m` 8.0 horizontal is the DoorController proximity trigger's
  largest half-extent (8×6×6 ⇒ 4 m) plus the up to ~3 m the `Door` transform sits above the walkable floor;
  `gate_reach_v_m` 6.0 vertical keeps a player who is on the roof above a door from "reaching" it (both measured
  deterministic episodes spent 32-130 s at y 10-18 over the `hops` 9 gate at y 1, which is not passing it). The
  `open` case doubles both. `open` alone never counts, because enemies open doors too.
- 0-1's gate spacing is safe against a through-wall hop skip: the closest pair with different `hops` is 21 m
  ((202,56,431.5) and (202,56,452.5)), outside both radii. That is a property of 0-1, not of the rule.

---

## 6. Reward arithmetic

### 6.1 Weights

`RewardConfig` gains two fields (both default `0.0`, so Cyber Grind and old configs are unchanged):

```python
gate: float = 0.0           # per new lower hops value reached, once per level load
gate_approach: float = 0.0  # per metre of new best 3-D closeness to the current target
```

`configs/campaign_0-1.yaml` changes:

| key | old | new | why |
|---|---|---|---|
| `gate` | — | **15.0** | §6.2 |
| `gate_approach` | — | **0.15** | §6.2 |
| `door_unlock` | 3.0 | **15.0** | the fight that unlocks a gate is the one place the route signal is flat (§6.3). `door_unlock` already exists, is keyed by `Door.Unlock`'s rounded position, and `MilestoneTracker` already pays it once per level load with the respawn/`mark_paid` rules — so this needs no new mechanism, no new test and adds no new farm surface |
| `novelty` | 0.5 | **0.2** | demoted to a secondary explorer now that a real forward signal exists |
| `path` | 0.1 | **0.0** | it has never paid, and its observation slots are now the target's. The term stays in `RewardConfig` and in `compute_reward` |
| `pitch_limit_deg` | absent (0 = off) | **45.0** | F3: the camera sits at −78° |
| `run_name` | `campaign_ppo_ground` | **`campaign_gates`** | §10 |
| header run commands | reference `campaign_ppo` | the §10 block | the header is stale already |

Everything else unchanged: `time` 0.02, `level_complete` 100, `checkpoint` 10, `arena_clear` 10, `kill` 0.5,
`damage_dealt` 0.5, `damage_taken` 0.01, `death` 5, `style` 0, `punch` 0.01.

### 6.2 The arithmetic

Units: 0-1 at `fixed_fps` 30 / `frameskip` 2 = **15 decisions per game second**; `max_steps` 9000 = 600 game
seconds. Human reference run **146.58 s = 2199 decisions**, 59 kills, 6 checkpoints. Gate-to-gate polyline
**723-736 m** (the checkpoint polyline is 677 m). 10 distinct hop values, 4 arena-held gate doors, 3 distinct
`ActivateNextWave` clear keys (0-1 has 4 waves with `last = 1`, two of them on the same GameObject at (56,18,640),
so they share one rounded key).

**Novelty, measured, both regimes** (`scratchpad/diag/ground_episodes.json`, the ground run's own episodes):

| regime | decisions | new cells | novelty at weight 0.5 | raw units | at weight 0.2 |
|---|---|---|---|---|---|
| mean over all 47 episodes | 3741 | 289 | 51.3 | 103 | +21 |
| the 6 full-length episodes | 9000 | 743 | 172.9 | 346 | **+69** |

**A policy that walks the route and finishes at human pace:**

| term | arithmetic | value |
|---|---|---|
| `gate` | 10 hop values × 15 | **+150** |
| `gate_approach` | 0.15 × ~723 m | **+108** |
| `door_unlock` | 4 arena-held gate doors × 15 | **+60** |
| **route subtotal** | | **+318** |
| `level_complete` | once | +100 |
| `checkpoint` | 6 × 10 | +60 |
| `arena_clear` | 3 distinct keys × 10 (that is the ceiling, not a floor) | +30 |
| `novelty` | a 2199-decision route walk through mostly-visited start rooms and fresh later rooms: ~200 raw × 0.2 | +40 |
| `kill` + `damage_dealt` | 59 × 0.5, plus ~60 health bars × 0.5 | +60 |
| `time` | −0.02 × 2199 | −44 |
| `punch` | −0.01 × ~750 presses | −8 |
| **total** | | **≈ +556** |

Route progress (+318) is the largest positive group — larger than finishing (+100), the checkpoints (+60), combat
(+60) and novelty (+40), which is 13% of the gross positive against 93.9% before the ground fix and 76.8% after it.

**A policy that wanders the start area to `max_steps` (today's median episode):**

| term | arithmetic | value |
|---|---|---|
| `gate_approach` | 0.15 × 64.8 m (spawn → the first gate, once per level load) | +10 |
| `novelty` | 346 raw units × 0.2 (measured, full-length episode) | +69 |
| `time` | −0.02 × 9000 | −180 |
| **total** | | **≈ −101** |

**Read that second table honestly.** For a policy that never leaves the start area — the regime this run will
occupy for its first million steps — novelty is still **87% of gross positive income** (69 of 79). The design does
not remove one-term dominance in that regime; it makes the regime unprofitable (−101, against **+501** for the
same behaviour before the ground fix) and gives the alternative a bigger prize. The novelty tripwire in §11.3 is
therefore written against the *wanderer* regime, not the route-walker one.

**Finishing beats loitering.** Finishing at decision 2199 pays +100 and stops the clock. Every gate, approach,
checkpoint, unlock and arena term is one-shot per level load, so after the route is walked the only remaining
income is decaying novelty (≤ 0.2 per new cell) against a permanent −0.02 per decision. Staying to `max_steps`
costs an extra (9000 − 2199) × 0.02 = **−136** on top of forfeiting the **+100**: loitering is **−236** relative to
finishing.

**Each gate pays for itself.** A ~70 m segment at the measured median 17.1 m/s is ~62 decisions = −1.24 of time,
against 15 + 0.15 × 70 = **+25.5**. Ratio 20 : 1.

**Novelty can no longer out-earn the route.** Beating the +318 route subtotal on novelty alone at weight 0.2 needs
1590 raw units in one episode — 4.6× the measured 346 of a full 9000-decision wander, and the ground-keyed measure
makes every unit cost real floor.

### 6.3 Why `door_unlock` moved

The shaping is dense along corridors and exactly flat where the run actually fails. On 0-1 the agent is paid +15 of
`gate` and ~+9.7 of approach for walking 64.8 m to the gun-room door, after which the route signal points through a
door that only 11 kills will open: `ActivateArena` "3 - Gun Room/Enemies/Trigger" at (40,13,393) holds
`Door (Large)@(40,1,408.5)`, unlocked by `Wave 2` (`last = 1`, `enemyCount = 8`) after a 3-enemy first wave. At the
old weights the entire reward for that fight was `kill` 5.5 + `damage_dealt` ~5.5 + `arena_clear` 10 +
`door_unlock` 3 ≈ +24 against ~30 s of time (−9) and −5 per death — F3 unchanged. At `door_unlock` 15 it is ≈ +36,
and the term that pays it is a route term keyed to the gate the agent is being steered at.

This is also why §11.3 gate 2 is explicitly a **combat** gate, not a navigation gate (§11.3).

---

## 7. Look modes

### 7.1 Action space: campaign only

The new dimension is **appended at the end of the campaign action vector**, so the new action-head logits are
appended as rows and every existing row keeps its index. **The Cyber Grind action space does not change.**

```python
LOOK_MODES = 3   # 0 free, 1 enemy, 2 gate
BASE_NVEC   = (3, 3, *([2] * len(BUTTONS)), NUM_WEAPON_CHOICES, len(YAW_BINS), len(PITCH_BINS))
ACTION_NVEC          = np.array(BASE_NVEC, dtype=np.int64)                 # 11 dims, sum 42 — unchanged
ACTION_NVEC_CAMPAIGN = np.array((*BASE_NVEC, LOOK_MODES), dtype=np.int64)  # 12 dims, sum 45

def action_space(campaign: bool = False) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC)
```

- `env.py:122` becomes `self.action_space = action_space(self.cfg.mode == "campaign")`.
- `decode_action(a)` infers the width from `len(a)`: 12 ⇒ `"look_mode": int(a[11])`, 11 ⇒ `"look_mode": 0`.
- **`noop_action()` must stop using negative indices.** `a[-2] = YAW_BINS.index(0.0)` and
  `a[-1] = PITCH_BINS.index(0.0)` address the wrong slots in the 12-wide vector. Signature becomes
  `noop_action(campaign: bool = False)`, using explicit indices and leaving the look-mode slot 0. Its only callers
  are `scripts/campaign_check.py` and the tests; `bridge_test.py` and `walk_to_exit.py` build mod commands directly.
- **Why campaign-only.** `spaces.action_space()` is read by `env.py` for both modes and by
  `scripts/transfer_weights.py:79` (`SpacesEnv(ObsLayout(campaign=True).space(), src.action_space)`). Widening the
  module constant unconditionally would make every Cyber Grind checkpoint unloadable against its own env and make
  `transfer_weights.py` emit an 11-dimension model that `train.py --resume` cannot use against a 12-dimension env —
  and CLAUDE.md documents rerunning that command as the recovery path when the first update's entropy is out of
  band. Look mode 1 is an auto-aim; handing it to the Cyber Grind run would short-circuit the exact capability that
  run has spent 6.5M steps failing to learn. **Look mode 1 is not available in Cyber Grind.**
- `transfer_weights.py` must be updated to build the destination with `action_space(campaign=True)` and copy the
  source's 42 action rows × `--action-scale` into rows 0-41, leaving rows 42-44 zero — the same widening
  `add_look_mode.py` does, so the documented Cyber-Grind → campaign path still produces a loadable model.
- The wire `action` message is unchanged. `env.step` **pops `look_mode` out of `command`** before
  `client.step(command)`; the mod would ignore it either way (§3.7), but the mod engineer should not have to infer
  that.

### 7.2 Geometry

Resolved in `env.step`, after `decode_action` and **before** `_note_behaviour`, against `prev` — the observation the
policy acted on. Sign conventions as verified: **positive `rotationX` looks up**; `yaw_frame` gives +x right, +z
forward, and increasing `rotationY` turns right.

```python
YAW_CAP   = max(abs(b) for b in YAW_BINS)    # 90.0 deg per decision
PITCH_CAP = max(abs(b) for b in PITCH_BINS)  # 20.0 deg per decision
MODE1_PITCH_LIMIT = 85.0                     # inside the game's own +-90 clamp
PITCH_BAND = cfg.pitch_limit_deg or MODE1_PITCH_LIMIT   # 0 means "off" everywhere else in the codebase
```

**Mode 0 — free look.** Unchanged: the sampled bins, then `clamp_pitch_command(prev.player.pitch, pitch,
cfg.pitch_limit_deg)` with the limit now 45.

**Mode 1 — nearest visible enemy.** Target = the first entry with `visible == true` among the **first
`layout.max_enemies` (8)** entries of `prev["enemies"]`, which the mod sorts by distance. The 8-entry restriction
matters: the env asks the mod for 32 enemies so damage rewards are not missed, but the policy only ever sees 8, and
mode 1 must not aim at something the observation does not contain. Aim point = that entry's `rel`, the enemy's
`GetCenter()` in **camera space, origin at the camera** (`ObservationBuilder.cs:221`), which is the right origin for
aiming (+x right, +y up, +z forward). Un-pitching by the current pitch `p` gives the player's yaw frame:

```python
x, y, z = enemy["rel"]
p  = prev["player"]["pitch"]
cp, sp = cos(radians(p)), sin(radians(p))
xf, yf, zf = x, y * cp + z * sp, -y * sp + z * cp        # yaw frame
yaw_cmd   = clamp(degrees(atan2(xf, zf)), -YAW_CAP, YAW_CAP)          # + turns right
elevation = degrees(atan2(yf, hypot(xf, zf)))                          # + above the horizon
pitch_cmd = clamp(elevation - p, -PITCH_CAP, PITCH_CAP)                # + looks up
pitch_cmd = clamp(p + pitch_cmd, -MODE1_PITCH_LIMIT, MODE1_PITCH_LIMIT) - p
```

Check: `rel = (0,0,1)` (dead ahead) ⇒ `yaw_cmd` 0, `elevation` = p, `pitch_cmd` 0. `rel = (1,0,0)` ⇒ `yaw_cmd` +90
(turn right). Mode 1 deliberately ignores `pitch_limit_deg` (an enemy overhead in an arena is exactly the case the
band blocks) but never exceeds ±85, inside the ±90 clamp `ActionInjector.ApplyLook` applies per frame
(`ActionInjector.cs:270`, after dividing the command by the frameskip at :175-178).

The un-pitch is **exact only up to camera roll**, which the old spec said did not exist. It does:
`CameraController.ApplyRotations` (`CameraController.cs:413`) is
`AngleAxis(-rotationX, right) * AngleAxis(transitionRotationZ, forward) * AngleAxis(tiltRotationZ, forward) * rotationOffset`.
`tiltRotationZ` targets `movementHor * -1` normally and `movementHor * -5` while boosting (:372-380), so it is ≤ 1°
strafing and ≤ 5° during a dash; `transitionRotationZ` only moves on gravity transitions, which 0-1 has none of.
The residual aim error is a few degrees and only at high elevation. Do not try to correct it.

`visible` is a line-of-sight test with **no frustum check** (`ObservationBuilder.cs:213`), so the nearest visible
enemy can be behind the player and mode 1 will turn 180° toward it, up to the 90°-per-decision cap. That is intended
(turn and fight) and must not be "fixed" into a forward-only filter.

**Fallback:** no player, no visible enemy in the first 8, or `rel` shorter than 1e-6 ⇒ behave exactly as mode 0.

**Mode 2 — current target.** Target = `GateProgress.target` — the same object packed into observation slots
448-455, with the same frozen closed-door `pos`.

```python
gx, gy, gz = yaw_frame((target["pos"] - prev["player"]["pos"]), prev["player"]["yaw"])
yaw_cmd      = clamp(degrees(atan2(gx, gz)), -YAW_CAP, YAW_CAP)
elevation    = degrees(atan2(gy, hypot(gx, gz)))
target_pitch = clamp(elevation, -PITCH_BAND, PITCH_BAND)
pitch_cmd    = clamp(target_pitch - p, -PITCH_CAP, PITCH_CAP)
```

**Fallback:** no target (unordered/empty gates, or no player) ⇒ behave exactly as mode 0.

**Movement is unchanged** — still camera-relative, as the game computes it. Turning toward a gate therefore also
turns "forward" toward it, which is the point.

**Diagnostics.** `look_free_frac`, `look_enemy_frac`, `look_gate_frac` per episode. `yaw_track` and `pitch_track`
must only count steps where `look_mode == 0`, or they become ~1.0 by construction and stop measuring the look head.

**All four count the mode that APPLIED, not the one the policy sampled** (clarified during implementation review;
it follows from "behave exactly as mode 0" above, and is how it is implemented and tested —
`test_look_mode_counters_record_the_mode_that_applied`). A fall-back step is a mode-0 step in every way that
matters here: the sampled bins drove the camera, so grading it cannot score ~1.0 by construction, and it is a
genuine look-head sample. The fall-backs are not rare — mode 2 falls back on *every* step of an unordered level or
against a 0.5.x mod (§3.7), so the run's first window can have no gates at all, and mode 1 falls back on the ~30% of
steps with nothing visible. Counting the request instead would report `look_free_frac` ≈ 0 for a window whose yaw
and pitch heads were causally live on every step, which is exactly the number R6 mandates reading the per-dimension
entropy against.

**Known and accepted:** on a step where mode 1 or 2 is selected the yaw and pitch dimensions are causally inert, so
`E[∇ log π(yaw) · A] = 0` there and only the entropy bonus acts on those two heads, pushing them toward uniform in
proportion to the non-mode-0 share. Today's entropy is 10.6 of 12.49 nats, i.e. the head is near-uniform anyway, so
the practical harm is small; the real cost is that total entropy stops being an interpretable `ent_coef` signal once
the ceiling moves to 13.59 and part of the head is unconstrained. Mitigation, not a fix: **log per-dimension entropy
alongside the total** and put `look_free_frac` in the same dashboard panel, so the two are always read together.

### 7.3 Weight-surgery contract

New `python/scripts/add_look_mode.py`, modelled on `scripts/transfer_weights.py` and its test.

```
python scripts/add_look_mode.py models/campaign_ppo_ground/latest.zip \
                                models/campaign_gates/look_init.zip
```

| tensor | contract |
|---|---|
| `mlp_extractor.policy_net.*`, `mlp_extractor.value_net.*` (layers 0 and 2) | bit-identical, **except** first-layer `.0.weight` columns 448-455 set to 0 and `.0.bias` compensated (below), in **both** stacks |
| `action_net.weight` | shape `(42, H)` → `(45, H)`; rows 0-41 bit-identical, rows 42-44 **zero** |
| `action_net.bias` | shape `(42,)` → `(45,)`; entries 0-41 bit-identical, entries 42-44 **zero** |
| `value_net.*` (the critic output layer) | bit-identical |
| observation space | unchanged, `(479,)` |
| action space | `MultiDiscrete([3,3,2,2,2,2,2,2,6,11,7])` → `MultiDiscrete([3,3,2,2,2,2,2,2,6,11,7,3])` |
| `num_timesteps`, `_n_updates` | preserved from the source, so the new run's step axis is continuous |
| optimizer | preferred: every parameter's Adam `exp_avg`/`exp_avg_sq` carried over, padded with **zero rows** for the three new action rows, zeroed for the eight changed input columns, `step` preserved. Acceptable fallback: a fresh optimizer (what `transfer_weights.py` does) — note it in the run log if taken |

**Mean-fold, required.** Before zeroing, fold the removed columns' mean contribution into the first-layer bias in
each stack:

```python
MU = (-0.054, -0.098, -0.034, 0.324, 0.114, 0.000, 0.821, 0.179)   # per-column mean of inputs 448-455
for W, b in ((policy_W0, policy_b0), (value_W0, value_b0)):
    b += W[:, 448:456] @ MU
    W[:, 448:456] = 0.0
```

`MU` is measured over the 436 real `Level 0-1` observations in `scratchpad/diag/run_raw.jsonl` packed through
`pack_observation(ObsLayout(campaign=True))`. The script recomputes it from that file (or from a file given with
`--stats`) rather than hard-coding it, and prints it.

**Measured effect on `models/campaign_ppo_ground/latest.zip` over those 436 states** (reproduced while revising this
spec):

| variant | mean KL(old‖new) | greedy action changed | mean \|ΔV\| |
|---|---|---|---|
| zero only | 0.0449 | 45.6% | 1.32 |
| **zero + mean-fold** | **0.0262** | **29.6%** | **0.77** |

for reference `target_kl` is 0.03, the run's measured median `approx_kl` is 0.020, and V spans −5.99 … +64.05.
Column 453 (`status == "complete"`) already has an exactly **zero** norm in both stacks — that input was never once
true, so it never received gradient and was left at the transfer's zero — so seven columns actually carry weight.

**Do not claim bit-identity.** The new run does **not** continue from the current policy unchanged; it continues
from a policy displaced by slightly less than one PPO update, and §11.3's gates are read against that. The critic's
`explained_variance` will dip on resume for the same reason.
`tests/test_look_mode_transfer.py` asserts the **KL bound** (mean KL ≤ 0.03 on the recorded observations) and the
tensor-level contract, not distribution identity.

**The action-space change invalidates every existing campaign checkpoint.** Any `.zip` not run through this script
will fail to load against the new campaign env. `eval.py` and `keep_best.py` must be pointed at the new run only.
Cyber Grind checkpoints are unaffected (§7.1).

---

## 8. MOD: un-wedge (D1)

Active **only while `EpisodeController.InControl`**, so a human playing the same install is never affected.

### 8.1 What is actually true (the old §8.1 was wrong in three places)

Observable signature: `!gc.onGround`, `!sliding`, `slowMode` true, horizontal speed ≈ 0, velocity `(0,-100,0)` or
~0, stamina frozen, jump/dash/slide inert, a 0.477 m/s creep.

What the decompiled code supports:

1. **The state is not a geometry mismatch, and step 2 of the old fix was a no-op on it.** The early-return branch is
   only reachable when `playerCollider.height != 3.5f` (`NewMovement.cs:921`), which is only true after
   `StartSlide()` or a `forceCrouch` ground. `StartSlide` already sets `playerCollider.height = 1.25f`,
   `transform.position += up * -0.5f * (height - 1.25f)` and `gc.SetLocalPosition(groundCheckPos + Vector3.up * 1.125f)`
   (:2267-2274) — the same triple the `forceCrouch` path sets at :902-908. Collider height, transform and
   ground-check offset are therefore **already mutually consistent** when `crouching = true; slowMode = true; return;`
   runs at :926-928.
2. **`TryStartSlam` does not re-apply the slam velocity.** Its entry condition ends `&& !gc.heavyFall`
   (`NewMovement.cs:1116`), so it cannot re-enter. The velocity is re-applied by
   `if (gc.heavyFall) { if (!slamStorage) rb.velocity = rb.GetGravityDirection() * 100f; }` in `Update`
   (:701-704). This matters for placement: `Update` calls `HandleSlideState()` at **:740**, *after* :701, so a
   postfix on `HandleSlideState` that clears `heavyFall` and zeroes the downward velocity holds for the frame.
3. **The one hypothesis the code supports for why `onGround` never recovers:** `GroundCheck` maintains `onGround`
   from trigger callbacks over a `cols` list (`GroundCheck.cs:296` enter, `:256` exit, `:117` `UpdateState`). A
   collider the ground-check capsule is *already overlapping* when the state begins never fires `OnTriggerEnter`
   again, so `cols` stays empty and `UpdateState()` can never set `onGround`. `GroundCheck.ForceGroundCheck()`
   (`GroundCheck.cs:97-115`) exists for exactly this: it clears `cols`, runs `Physics.OverlapCapsule` on the
   ground-check capsule, calls `OnTriggerEnter` for every overlap, then `UpdateState()`.
4. `slowMode` blocks dash (:1036) and stamina regen (:755) and cuts walk speed to `slowMode ? 1.25 : 2.75` (:1332).
   Slide start (:1031) is `(!slowMode || crouching)` and `crouching` is true here, so **slide was never the blocked
   thing** — dash and stamina were.

### 8.2 Preferred fix

A Harmony **postfix on the private `NewMovement.HandleSlideState`** (patch by name), running only while in control.

*Guard:* act only when `!nm.gc.onGround && nm.slowMode`, **held for `unwedge_frames` consecutive frames**
(default **10**, clamped to at least 1; the counter resets on any frame the guard fails, including the frames the
fix is off or the AI does not have control). A legitimate **grounded** crouch under a low ceiling keeps `slowMode`
true and is untouched (and still reports `slow_mode: true` in the obs — A8).

**Why the hold is on the preferred fix and not only on §8.3's fallback (added during implementation review).** The
guard alone cannot tell the wedge from a legitimate ground slam on frame 1. `Update` runs `HandleInputs()` (:686)
before the `heavyFall` block (:701) and `HandleSlideState()` (:740), and `TryStartSlam` (:1108) calls `StopSlide()`,
which does **not** restore `playerCollider.height` (:2448-2474) — the stand-up inside `HandleSlideState` does, on a
later frame. So a slam started out of a slide arrives at :921 with `height == 1.25f`, and under a vent roof, a
doorway lintel or any overhang the stand-up raycast (3.5 m) or spherecast (0.5 r, 2 m) hits and the game sets
`crouching = true; slowMode = true;` on the very frame `gc.heavyFall` and `rb.velocity = (0,-100,0)` were set.
Steps 1-2 on that frame silently cancel the slam: no `LandingImpact`, no ground-slam enemy damage, and no
`Breakable.Break(2f)` from `Update` (:708-716) — one of the few ways through the two weak planks that seal 0-1's
`unlock_all_gear` starting room (§13). `heavyFall` cannot come back without a fresh Slide press with
`slamCooldown == 0`. A8 cannot catch it: its slide/slam/crouch steps all run **grounded** (`a8_steps.jsonl` records
`grounded: true` on every `D_slam` step), where the `!onGround` guard stops the patch running at all.
G1 only asks for recovery within 1.0 s of game time, so frame 1 buys nothing. 10 frames is 0.33 s at `fixed_fps` 30
and 0.17 s at 60 — inside G1 either way, and two orders of magnitude below the shortest measured wedge (672
decisions). The separation is clean because a slam that can make progress lands (`onGround`) or falls out of the
3.5 m ceiling ray within a frame or two, resetting the counter, while the wedge holds `slowMode` continuously:
`NewMovement.slowMode` is written at exactly three places (:927 true, :931 and :2203 false), so nothing clears it
while the stand-up test keeps failing. `Recoveries` is now counted once per armed run rather than once per frame.

*Then, in this exact order (the order is normative, not incidental):*

1. `nm.gc.heavyFall = false` — the `GroundCheckGroup.heavyFall` setter writes every instance.
2. If `nm.rb.velocity.y <= -99f`, set the velocity to `(vx, 0, vz)`.
3. Defensive and idempotent, written only if the value differs (§8.1.1 says the game normally already has these;
   they are here so an unforeseen entry path cannot leave an inconsistent collider):
   `playerCollider.height = 1.25f`; `gc.SetLocalPosition(groundCheckPos + Vector3.up * 1.125f)` (`groundCheckPos` is
   private, `NewMovement.cs:85` — `AccessTools.FieldRefAccess<NewMovement, Vector3>`); `crouching = true`
   (private, :154).
4. `nm.slowMode = false`.
5. Force the ground check on every instance:
   ```csharp
   var instances = AccessTools.FieldRefAccess<GroundCheckGroup, List<GroundCheck>>("instances")(nm.gc);
   foreach (var g in instances) if (g != null && g.isActiveAndEnabled) g.ForceGroundCheck();
   ```
   **`GroundCheckGroup` has no `ForceGroundCheck` and no `cols`.** `NewMovement.gc` is a `GroundCheckGroup`
   (`NewMovement.cs:213`), whose whole public surface is `onGround`, `touchingGround`, `hasImpacted`, `heavyFall`,
   `canJump`, `sinceLastGrounded`, `forcedOff`, `superJumpChance`, `bounceChance`, `extraJumpChance`, `AddInstance`,
   `RemoveInstance`, `SetLocalPosition`, `ForceOff`, `StopForceOff`, `Update`. `ForceGroundCheck()` is on
   `GroundCheck`, and the group's list is `[SerializeField] private List<GroundCheck> instances`.
   (`nm.gc.GetComponentsInChildren<GroundCheck>(true)` is an acceptable alternative.)
   **Steps 1-2 must precede this.** `GroundCheck.OnTriggerEnter` has a `if (heavyFall)` branch
   (`GroundCheck.cs:333-385`) that calls `eid.DeliverDamage(..., ×5000f, ..., 2, tryForExplode: true)` on every
   overlapping enemy collider and can call `Bounce(...)`, which does `nmov.transform.position = position`
   (`GroundCheck.cs:240-245`) — a teleport this section forbids.

*The postfix re-clears `slowMode` on every frame the game's own stand-up test keeps failing.* That is intended: it
re-enables dash and stamina regen while leaving the player legitimately crouched, and it stops by itself the moment
the game's success branch runs (`slowMode = false`, :931).

**Must not:** change `transform.position` at all; set `playerCollider.height` to 3.5; damage any enemy; run outside
AI control.

**Deleted from the old spec:** "if the game's own stand-up test now passes, let the game's own success path run". A
postfix runs after the method returned; the success path is ~20 lines of private state (`travellerPosition`,
`portalAwareCollider`, `standing`, `cc.defaultTarget`, :930-990), and re-invoking the private method from its own
postfix recurses through the patch. It is also unnecessary: the branch is re-entered every frame while
`height != 3.5f`, so the game stands the player up by itself on the first frame its own raycast/spherecast passes.

### 8.3 Fallback watchdog

Not needed: §8.2 holds, and its hold counter is what §8.3 proposed. `unwedge_frames` is therefore the config key of
the §8.2 hold (default **10**, minimum 1), not of a separate watchdog, and `unwedge` stays the only off switch.

If a broader signature is ever needed, the watchdog form is:
*Signature:* `!gc.onGround && !nm.sliding && (nm.slowMode || nm.gc.heavyFall) && horizontal speed < 1.0 m/s`, held
for `unwedge_frames` consecutive frames (30 frames = 1.0 s of game time at `fixed_fps` 30, 0.5 s at 60).
*Action:* steps 1-5 of §8.2, in that order. *Reset:* the counter clears on any frame the signature is false.

### 8.4 Reproducing the two flavours

Both are reproducible on demand with `scratchpad/diag/watch_probe4.py` (slam flavour, velocity `(0,-100,0)`,
stamina frozen at 24.7) and `watch_probe5.py` (non-slam, velocity ~`(0,-0.3,0)`, stamina 84.7).

---

## 9. PYTHON: env fixes (D5)

### 9.1 New `EnvConfig` fields

```python
gate_reach_m: float = 8.0          # horizontal radius at which a gate counts as reached (2x when it is open)
gate_reach_v_m: float = 6.0        # vertical half-height of the same test
gate_min_gain_m: float = 0.5       # metres of new best closeness below which gate_approach pays nothing
wedge_seconds: float = 3.0         # game seconds wedged before the episode ends (0 = detector off, still counted)
wedge_creep_mps: float = 1.5       # movement below this counts as not moving, in the wedge test only
slide_min_hold: int = 0            # decisions slide stays held once pressed (0 = off)
archive_save_steps: int = 20000    # env lifetime steps between exploration-archive saves
archive_save_seconds: float = 600.0
```

`EnvConfig.from_dict` already drops unknown keys, so an old `env_config.yaml` still loads.

### 9.2 Stuck clock

`_campaign_progress` takes `prev` as well as `cur`. It resets `_steps_since_progress` on any of: a checkpoint, an
arena clear, a door unlock, `novelty > 0`, `path_gain > 0`, **a new gate hops value**, **`gate_approach > 0`**,
**`cur.stats.kills > prev.stats.kills`**, or **`cur.stats.style > prev.stats.style`**.

`_respawn()` sets `_steps_since_progress = 0` (a respawn moves the player; the pre-death clock is meaningless).

**Damage dealt must not reset the clock, and `damage_dealt_units` is not extracted.** `rewards.py:163-177` computes
`dealt` by comparing each tracked enemy's health to the previous step with **no attribution to the player**, and
ULTRAKILL enemies damage each other with crossfire and take environmental damage. Under an unattributed rule, the
Stray-arena tail the live probe recorded — 193 s after checkpoint 3 with `arena_enemies_alive > 0` on 2,858 of
2,899 decisions and exactly 1 kill — would never truncate, growing a 675-decision stuck tail into the full
9,000-decision `max_steps`. The throughput report already measures 51% of samples coming from timeout episodes and
13.6% from stuck tails; that change would move the 13.6% into the 51%. `kills` and `style` only move on player
actions. (Separately noted, not changed here: the existing `damage_dealt` 0.5 reward is already payable for enemy
crossfire.)

Why it matters: 4 of 0-1's gate doors are held behind kill gates inside locked rooms where novelty runs out. The
motivating case is the **post-checkpoint-3 Stray arena**, not the window mean — of the ground run's 47 episodes, 29
have exactly 0 kills and only 2 have 1-10, so the "mean 8.8 kills, below the 11 the first door needs" reading in
the previous draft was a mixture artifact and is withdrawn.

### 9.3 Wedge detector

Per step, from the observation. `CREEP_M = cfg.wedge_creep_mps * frameskip / fixed_fps` (= **0.10 m** per decision
at both 30/2 and 60/4):

```python
moved  = dist3(pos, prev_pos) if prev_pos is not None else 0.0
still  = hypot(vel[0], vel[2]) < 1.0 and moved < CREEP_M
wedged = (player is not None and not player["grounded"] and not player["sliding"] and still
          and (player.get("slow_mode") or player.get("heavy_fall")))
```

Fallback when `slow_mode` and `heavy_fall` are both absent (mod < 0.6.0): the same predicate without the flag test.
The two branches now differ **only** in the flag test.

`wedged_steps` counts steps **inside a consecutive run that reached the hold length** `HOLD = wedge_seconds ×
fixed_fps / frameskip` (3.0 s ⇒ 45 decisions), crediting the whole run retroactively when the threshold trips. Raw
candidate steps are not counted.

**Why both details are load-bearing**, measured on `scratchpad/diag/run_steps.jsonl` (32,022 decisions, 7 episodes):

| predicate | raw flagged / episode | credited (runs ≥ 45) / episode | real wedge runs found |
|---|---|---|---|
| flags + speed only, raw count | 1228.7 | — | — |
| flags + speed only, run-credited | 1228.7 | 882.7 | **5 of 5** |
| + `moved < 0.05 m` | 856.4 | 778.3 | **4 of 5** ✗ |
| + `moved < 0.10 m` (**chosen**) | 892.4 | **877.7** | **5 of 5** |
| + `moved < 0.20 m` | 900.1 | 881.7 | 5 of 5 |

Without the movement term, ordinary falls and slams flag on 26.9% of all decisions (mean **346** per episode in runs
of ≤ 29), which alone would make §11.3 gate 1's "< 20 per episode" unpassable. With `0.05 m` — the constant one
review proposed — the 694-decision wedge at (145.4, 45.5, 664.2), the post-checkpoint-3 one F1 cites as proof the
state is general rather than a vent property, fragments below the hold and is **lost**: its per-decision movement
has p50 0.030 m and p90 0.055 m. `0.10 m` catches all five and leaves 14.7 uncredited candidate steps per episode,
which run-crediting removes entirely. **Baseline for §11.3 gate 1: 877.7 `wedged_steps` per episode.**

Consecutive wedged steps ≥ `HOLD` also end the episode with `truncated = True` and `end_reason = "wedged"` — a
distinct reason, not folded into `stuck`, so the dashboard can tell "cannot move" from "moving but making no
progress". This is a safety net that should almost never fire once §8 works; if `wedged_steps` stays high after the
mod lands, §8 is not working and no reward change will help.

**Precedence in the env's end-of-step chain:** `level_complete` (terminated) → `wedged` → `stuck` → `max_steps`.

**`wedged` feeds the `stuck_repeats` escape hatch.** `env.py:469` is the only input to `_stuck_streak`, and that
streak is what `choose_fresh_start(stuck_limit=...)` uses to force a fresh level load after three failures at the
same checkpoint. Change it to:

```python
if self._last_end_reason in ("stuck", "wedged"):
    self._stuck_streak = self._stuck_streak + 1 if current == self._stuck_checkpoint else 1
    self._stuck_checkpoint = current
```

Keep the two reasons distinct everywhere else (info, dashboard, `episodes.jsonl`). Without this, a checkpoint whose
respawn point is wedge-prone — exactly the (145.4, 45.5, 664.2) case, 28.2 m from the locked door (146,31,640), with
`has_checkpoint` true — can never escalate to a reload. Vent wedges on a fresh load are saved only incidentally,
because `has_checkpoint` is false there.

### 9.4 Exploration archive saves

Keep the "every 20 episodes" save and add a schedule checked inside `step()` every 500 steps: save when
`lifetime_steps - last_save_steps >= archive_save_steps` **or** `monotonic() - last_save_time >=
archive_save_seconds`. `UltrakillEnv` has no such counter today (`self._steps` is reset to 0 in `reset()`,
`env.py:210`), so add `self._lifetime_steps`, incremented in `step()` and never reset; the check runs regardless of
episode boundaries. Measured cost: 16 ms per save of a 41,791-cell archive — negligible at once per 10 minutes.
This is why the whole ground run saved nothing: episodes are ~3900 decisions, no worker reached 20 episodes before a
restart, and `SubprocVecEnv` workers never call `close()` on Ctrl+C.

### 9.5 `slide_min_hold`

In `env.step`, after the look mode is resolved:

- `"slide" in command["buttons"]` ⇒ set the latch to `slide_min_hold - 1` remaining decisions.
- Else if the latch > 0 ⇒ append `"slide"` to `command["buttons"]` and decrement.
- The latch is cleared on `reset()`, on `_respawn()` and whenever `campaign.input_locked` is true.
- Never removes a slide the policy asked for; only adds.
- Diagnostic `slide_forced_frac` per episode.

Rationale, and the caveat: passing 0-1's vent needs slide held over a run of 3-18 (mean 11.3) consecutive decisions,
the policy presses slide on 90% of sliding-in-vent decisions, and `0.90^11.3 = 0.30` is close to the observed 31%
pass rate. **That arithmetic over-fits.** The live probe classifies the 22 wedged episodes as 14 slide-release,
2 jump-out-of-a-slide, 6 that entered the vent un-crouched or never sliding, plus 1 dash at stamina ≥ 100 — so at
most ~64% of wedges are the mechanism this latch addresses. Worse, the latch holds slide while the policy presses
jump on 58-81% of decisions, and "a jump out of a grounded slide" is one of F1's two wedge paths — **the latch may
raise the jump-out wedge rate**, which is the same hazard cited when rejecting an "until jump is pressed" latch.
Default stays **0 (off)**: it is an experiment, and A9 must be re-measured with it on before it is enabled.

### 9.6 `_ground_point`

Prefer `ground_ray_center`:

```python
c = raw.get("ground_ray_center")
drop = c if c is not None else min(raw["ground_rays"])
if drop >= layout.ground_ray_length - 0.5:  return None     # no ground under the player
return (x, y - drop, z)
```

The centre ray is the correct measure: the 8-ray ring's minimum can be a ledge 4 m away rather than the floor, and
the measured ring spread among hit rays is p50 0.5 m but **p90 10.8 m**.

**This re-keys the novelty archive, and that is accepted, not hidden.** On the 32,022-row live log the 4 m cell key
differs between the ring minimum and the ring median on **23.2%** of rows with any ground hit, and 23.8% of rows
have all eight rays missing. So a fifth of the carried `explore_*.npz` counts will address cells the new key never
visits, and novelty income rises slightly. Carry the archives over anyway (§10) — the bulk of the counts are floor
cells where centre and ring agree. **`oob_frac`, `cells_new` and `part_novelty` are not comparable with the ground
run's baselines** (`oob_frac` 0.090); they are not comparable regardless, because `novelty` goes 0.5 → 0.2 and the
run name changes. Take a fresh `oob_frac` baseline from the new run's first 100 episodes and judge against that.

### 9.7 Info keys and the per-episode log

New keys in every campaign `info` (all set unconditionally in `_info`):

| key | type | meaning |
|---|---|---|
| `gates_reached` | int | distinct hop values reached this level load (0 when none) — always numeric, chartable |
| `gate_hops_best` | int or `None` | lowest hops reached this level load |
| `wedged_steps` | int | §9.3, run-credited |
| `level_started` | int | 1 if `campaign.level_started` was ever true this episode |
| `look_free_frac`, `look_enemy_frac`, `look_gate_frac` | float | §7.2 |
| `slide_forced_frac` | float | §9.5 |
| `start_checkpoint` | str or `None` | the checkpoint id the episode began at (`None` on a fresh load) |
| `end_pos` | `[x,y,z]` | the player's final position, rounded to 2 dp |

`CAMPAIGN_INFO_KEYS` gains only the numeric, always-present ones: `gates_reached`, `wedged_steps`, `level_started`,
`look_gate_frac`, `slide_forced_frac`. The rest are read from `info` by the `episodes.jsonl` writer.

**`runs/<run>/episodes.jsonl`** is written by `ProgressCallback`, not by the env: with `SubprocVecEnv` five workers
would interleave appends to one file, while the callback sees every finished episode in the main process. One JSON
object per line, appended, flushed per line:

```json
{"t": 1757980000.4, "env": 2, "timesteps": 2914785, "reward": -25.5, "length": 1368,
 "fresh_start": 1, "start_checkpoint": null, "end_reason": "wedged", "kills": 0, "deaths": 0,
 "checkpoints_level": 0, "gates_reached": 0, "gate_hops_best": null, "level_started": 0,
 "wedged_steps": 612, "end_pos": [40.0, -0.5, 361.2], "level_seconds": null, "completed": 0}
```

**The writer must read `info.get(...)` directly for the non-numeric fields.** `ProgressCallback`'s existing helper
is `def field(name): return _num(ep.get(name, info.get(name)))` (`progress.py:188-190`), and `_num` (:42-48) returns
`None` for anything `float()` rejects — so routing `start_checkpoint` (a string) and `end_pos` (a list) through it
would write `null` for exactly the two fields that say where an episode died. `start_checkpoint`, `end_pos`,
`end_reason` and `level_seconds` bypass `field()`; `gate_hops_best` is `None` or an int.

A write failure prints a warning and never stops training, like `write_json_atomic`.

### 9.8 `status.json`, `poll_status.py`, dashboard

- `progress.py`: add `gates_reached`, `wedged_steps`, `level_started`, `look_gate_frac`, `slide_forced_frac` to the
  `_snapshot` `recent` key tuple; add top-level `best_gates_reached` (max) and `best_gate_hops` (**min** — lower is
  better, so it needs its own comparison, unlike `best_checkpoints_level`). **Both must also be read back in
  `_restore` (`progress.py:107-124`)**, `best_gate_hops` with the reverse comparison, or they reset on every
  resume — `_restore` is what carries `best_checkpoints_level` across a restart today.
- History: add `mean_gates_reached_100` to the fixed key set `_snapshot` writes into `history`
  (`progress.py:314-324`); keep writing `mean_checkpoints_level_100`. `dashboard.py:433-452` reads by that name, so
  the two must agree.
- `poll_status.py`: add the new means to `FIELDS`, `best_gates_reached`/`best_gate_hops` beside
  `best_checkpoints_level`, and `"gate"`, `"gate_approach"` to `PART_FIELDS`. It keeps an existing header and drops
  unknown columns, so **move `metrics_log.csv` aside** at the start of the new run or the new columns are silently
  discarded.
- `dashboard.py`: two rows in `campaign_lines` — `gates/load` (`mean.gates_reached`, best) and `wedged/ep`
  (`mean.wedged_steps`) — and swap the second chart series from `mean_checkpoints_level_100` to
  `mean_gates_reached_100`.
- **Chart and judge `gates_reached` on fresh starts.** Like `checkpoints_level`, it is inherited by a respawn
  episode from its level load. CLAUDE.md records that `corr(fresh_start, checkpoints_level) = -0.975` produced the
  false "peak by depth" reading at 1.33M of the pre-fix run. The dashboard row shows the all-episode mean **and**
  the fresh-start mean side by side, and `best_gates_reached` is a max over fresh starts only.

---

## 10. Trainer and config (D6)

`scripts/train.py`:

1. **First statement after `from __future__ import annotations`**, before any `stable_baselines3`/`torch` import
   (sb3 imports torch at module import, and OpenMP reads the variable at library load):
   ```python
   import os
   os.environ.setdefault("KMP_BLOCKTIME", "1")
   ```
   Measured: 11.03 → 1.43 cores busy, forward pass 3.24 → 3.40 ms, 20 minibatch updates 0.32 → 0.33 s. The five
   games run below-normal priority on the same 12-core CPU, so the live gain needs a 10-minute A/B — but the burn is
   free to remove either way. Do **not** set `OMP_NUM_THREADS=1`: it costs 2.5× on the update.
2. After the torch import, before the model is built:
   ```python
   import torch
   torch.distributions.Distribution.set_default_validate_args(False)
   ```
   Measured: forward pass 3.08 → 2.13 ms, ≈ 2.9% of wall time.
3. `verbose` on **both** branches: `cls.load(..., verbose=train_cfg.get("verbose", 1), **hyper)` as well as the
   fresh branch. Today `campaign_ppo_train.log` is 0 bytes because only the fresh branch passes it.
4. No other change. **No PPO hyperparameter changes.**

New run: **`campaign_gates`** (a new name is required: `part_novelty`, `cells_new`, `part_door_unlock` and the
action space all change meaning). Before starting it, copy `models/campaign_ppo_ground/explore_*.npz` into
`models/campaign_gates/` — `explore_dir` follows `run_name`, and those counts carry the `1/sqrt(N)` decay that
makes the agent push outward — and move `runs/campaign_gates/metrics_log.csv` aside if one exists.

Run order:

```
python scripts/add_look_mode.py models/campaign_ppo_ground/latest.zip models/campaign_gates/look_init.zip
python scripts/games.py launch --count 5
python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_gates/look_init.zip
python scripts/poll_status.py --run campaign_gates
python scripts/keep_best.py --run campaign_gates --metric campaign
python scripts/dashboard.py --run campaign_gates
```

---

## 11. Tests

### 11.1 Offline (no game) — every one must pass before the run starts

**`tests/test_campaign.py`** (`GateProgress`, new cases):

| test | asserts |
|---|---|
| `test_gate_hops_pays_once_per_level_load` | walking 9 → 0 pays 10 gate steps total; re-walking after `mark_paid` pays 0 |
| `test_gate_skipped_hops_pay_per_hop` | 9 → 6 in one step pays 3 |
| `test_gate_approach_pays_only_new_best` | forward-back-forward pays the forward distance once; total ≤ the initial distance |
| `test_gate_approach_does_not_pay_for_flapping_between_two_gates_at_the_same_hops` | two gates at hops 2 that are 60 m apart, 5 A→B→A cycles that never come within reach: total approach ≤ the larger of the two first-seen distances; the second crossing pays 0 |
| `test_gate_approach_survives_a_respawn` | `mark_paid` then re-walking the same 60 m pays 0 |
| `test_gate_approach_is_re_earned_in_a_new_episode` | `reset_episode` then re-walking the same 60 m pays the same again, and the `gate` ladder is still absorbed (ruling R1) |
| `test_target_changes_do_not_pay_approach` | the step a target changes pays 0 approach |
| `test_target_before_any_gate_is_the_nearest` | fresh spawn ⇒ the `hops` 9 gate |
| `test_target_after_hops_zero_is_the_exit` | `best_hops == 0` ⇒ the exit sentinel; and `None` when `campaign.exit` is null |
| `test_retarget_pays_nothing` | `retarget` on an arbitrary position returns nothing and leaves `paid_hops` alone |
| `test_unordered_gates_have_no_target` | `gates_ordered: false` ⇒ no target, 0 and 0.0 |
| `test_empty_gates_have_no_target` | `gates: []` ⇒ same, and no exception |
| `test_open_gate_counts_as_reached_at_double_range` | `open` at 12 m horizontal reaches, closed at 12 m does not |
| `test_reach_is_a_cylinder` | 3 m horizontal but 12 m above the gate does not reach; 7 m horizontal and 4 m above does |
| `test_inactive_gates_are_ignored` | an inactive gate within reach sets neither `best_hops` nor the target |
| `test_hops_shifting_without_moving_pays_nothing` | the same gate array re-sent with every `hops` one lower pays 0 |
| `test_duplicate_gate_keys_do_not_freeze_the_target` | two gates whose keys differ only by the `#2` suffix retarget normally |
| `test_truncated_gates_still_target` | `gates_truncated: true` with a partial array still produces a target and pays |

**`tests/test_campaign_rewards.py`**: `gate` and `gate_approach` are paid from `CampaignStep` before the
missing-player early return, are 0 by default, and the arithmetic of §6.2 for a synthetic route run;
`door_unlock` at 15.0 still pays once per key per level load.

**`tests/test_spaces.py`**: `ObsLayout(campaign=True).size == 479` still; indices 448-455 carry the target in the
yaw frame with the documented scales and signs (gate scales 50/100, exit scales 100/200, ±4 clip), mirroring the
existing exit-vector sign tests; `target=None` ⇒ 448-455 all zero; 443-447 and 456-478 are byte-identical to today
for the same input; the Cyber Grind 448-dim path and its 11-dimension `ACTION_NVEC` are untouched;
`ACTION_NVEC_CAMPAIGN` is length 12 summing to 45; `noop_action()` and `noop_action(campaign=True)` both return
yaw/pitch neutrals at the right indices and look mode 0; `decode_action` on an 11-wide vector returns
`look_mode == 0`.

**`tests/test_campaign_env.py`** — `FakeLevel` **grows**:

- three doors along the corridor at z 15, 35 and 55 with `hops` 2, 1, 0, `gates_ordered: true`,
  `gates_truncated: false`, `controller_active: true`, `open` flipping true within 8 m;
- a two-gate tier variant for the flapping test;
- `ground_ray_center`, `slow_mode`, `heavy_fall`, `crouching` in the player block;
- a **`wedged` switch** that freezes `z` and reports `grounded: False`, `slow_mode: True`, `vel ≈ 0` — today
  `FakeLevel` reports `"grounded": True` and `"vel": [0,0,0]` unconditionally, so none of the wedge tests can be
  written against it;
- a **`falling` variant** that reports the same flags but keeps moving, so the movement term is exercised;
- `self.last_action`, recording the command dict, so the look-mode tests have something to assert against — today
  `step()` ignores everything but `action["move"]`.

| test | asserts |
|---|---|
| `test_walking_the_corridor_pays_each_gate_once` | 3 gate payments, 0 on a second pass after a respawn |
| `test_gate_slots_track_the_target` | 448-455 point at the next gate and switch when it is reached; the observation `reset()` returns already has slot 452 == 1.0 and `reset` pays 0 |
| `test_unordered_level_falls_back_to_the_exit_vector` | `gates_ordered: false` ⇒ 448-455 all 0, episode still runs |
| `test_a_wedged_player_ends_the_episode` | `slow_mode` true + airborne + still for 45 decisions ⇒ `end_reason == "wedged"`, `wedged_steps == 45` |
| `test_a_short_wedge_is_not_counted` | 44 wedged decisions then movement ⇒ `wedged_steps == 0` |
| `test_falling_is_not_wedged` | airborne, `heavy_fall` true, moving 1 m per decision ⇒ `wedged_steps == 0` |
| `test_wedge_fallback_without_mod_flags` | same as the first, with `slow_mode`/`heavy_fall` absent |
| `test_three_wedges_force_a_fresh_start` | three consecutive `wedged` ends at the same checkpoint ⇒ `fresh_start == 1` |
| `test_kills_reset_the_stuck_clock` | a kill with no novelty keeps the episode alive past `stuck_seconds` |
| `test_enemy_crossfire_does_not_reset_the_stuck_clock` | an enemy losing health with no kill and no style ⇒ the episode still ends `stuck` |
| `test_respawn_resets_the_stuck_clock` | `_steps_since_progress == 0` after `_respawn` |
| `test_slide_min_hold_holds_slide` | one press ⇒ slide in the next N-1 commands; latch cleared on reset |
| `test_look_mode_enemy_turns_toward_the_enemy` | mode 1 with an enemy 90° right ⇒ `look[0] == +90`; overhead ⇒ positive pitch; no visible enemy ⇒ identical to mode 0 |
| `test_look_mode_gate_turns_toward_the_target` | mode 2 yaw sign matches the target's yaw-frame x; pitch clamped to the band; `pitch_limit_deg == 0` uses the 85° fallback, not 0; no target ⇒ mode 0 |
| `test_look_mode_is_popped_from_the_command` | the dict handed to the client has no `look_mode` key |
| `test_look_mode_counters_record_the_mode_that_applied` | a mode that finds nothing to aim at counts as `look_free` and is graded by `yaw_track`; a mode that really aims counts as its own and is not graded (§7.2) |
| `test_ground_ray_center_is_preferred` | a centre hit different from the ring minimum decides the ground point; a centre miss with ring hits ⇒ `None` and `oob_frac` counts it |
| `test_archive_saved_on_a_step_schedule` | `archive_save_steps` small ⇒ the `.npz` exists mid-episode |
| `test_episode_info_has_the_new_keys` | every key in §9.7 present at every episode end |

**`tests/test_look_mode_transfer.py`** (new, modelled on `tests/test_transfer.py`): on small PPO models — action head
widened 42 → 45 with the last three rows and biases zero and every earlier row bit-identical; hidden stacks
identical except first-layer columns 448-455 (zero) and the first-layer bias (folded); the saved model loads with a
12-dimension `MultiDiscrete` and a 479 observation space; `num_timesteps` preserved; the three new logits are equal
on random inputs (uniform over modes); **mean KL(old‖new) over a recorded observation batch ≤ 0.03**, and the
mean-fold variant's KL is strictly lower than the zero-only variant's.

**`tests/test_progress.py`**: the new `mean_100` keys, `best_gates_reached` and `best_gate_hops` (minimum, not
maximum) appear in `status.json` **and survive `_restore`**; `mean_gates_reached_100` is in `history`;
`episodes.jsonl` gets one valid JSON line per finished episode with every §9.7 field, `end_pos` a 3-element list of
floats and `start_checkpoint` the checkpoint id string on a respawn episode; a write failure does not raise.

**`tests/test_campaign_config.py`**: `configs/campaign_0-1.yaml` still builds a 479-input env with a 12-dimension
action space; `gate`, `gate_approach`, `pitch_limit_deg`, `door_unlock` are real fields with the §6.1 values;
`path` is 0; `run_name` is `campaign_gates`.

Run all: from `python/`, `Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }`.

### 11.2 In game (one game, `games.py launch --count 1 --monitor 1`, nothing else on the port)

| # | check | pass criterion |
|---|---|---|
| A1 | `bridge_test.py --campaign` on `Level 0-1` prints the gates block, **on a fresh load and again after a checkpoint respawn** | 11 gates, hops 0..9 present, keys and positions matching §3.6 both times, `gates_truncated: false` |
| A2 | Same on `Level 1-1` and `Level 0-5` | 1-1 ordered (13 gates, hops 0..5); 0-5 `gates_ordered: false` with every `hops` null, and no exception |
| A3 | `ChooseExit` on 0-2 and 1-1 | `campaign.exit.pos` is the non-secret pit ((-199,-86.1,277) on 0-2, (81,-76.1,91) on 1-1) |
| A4 | `campaign_check.py` (0-1 and 1-1), plus a new check 6 for the gates block, run **again after a checkpoint respawn** | all previous verdicts unchanged; check 6 PASS both times |
| A5 | **Un-wedge, slam flavour** (`watch_probe4.py`) | recovery within 1.0 s of game time: `grounded` true or free fall resumes, `slow_mode` false, `heavy_fall` false; **`player.pos` unchanged (±0 m) across the recovery frame and no enemy loses health** |
| A6 | **Un-wedge, non-slam flavour** (`watch_probe5.py`) | same |
| A7 | Slide works after recovery | a held slide from the recovered state moves > 5 m in 10 decisions |
| A8 | No regression elsewhere | scripted slide, ground slam and crouch on open ground behave as before (probe B/C/D/E distances within 0.5 m of their recorded values); a **grounded** crouch under a low ceiling still reports `slow_mode: true` |
| A9 | **Vent pass rate** (`watch_tunnel_stats.py`, 15 sampled fresh loads at 30 fps / frameskip 2, **1500 decisions per load** — the budget the 8/26 = 31% baseline was measured at) | ≥ 80% reach z > 380 |
| A10 | Gate reward live | a scripted grounded slide + walk to the gun-room door pays exactly one `gate` and a positive `gate_approach`, and `gates_reached` reads 1 |
| A11 | Look mode 2 live | forcing `look_mode = 2` from spawn turns the camera to within 10° of the `hops` 9 gate within 3 decisions |
| A12 | Throughput | ≥ 150 steps/s with 5 games on 0-1 (today: 152) |

**A9 is a policy statistic, not a mod statistic**, because the un-wedge fix converts an absorbing failure into a
retryable one and the resulting pass rate then depends on the policy and the per-load decision budget. Pre-committed
outcome if it lands **between 31% and 80%**: the mod fix ships anyway (A5-A8 are the mod's own acceptance tests),
the §9.3 Python detector stays as the bound, and `slide_min_hold` becomes the next experiment with A9 re-measured
with it on (§9.5).

### 11.3 Run gates (judge on none of them early; `window >= 50` on every comparison)

1. **+200k steps:** `wedged_steps` mean < 20 per episode (baseline **877.7** under §9.3's run-credited definition)
   and `end_reason == "wedged"` under 5% of ends. Mechanism check only. If not, §8 is not working — debug the mod,
   do not tune weights.
2. **+500k steps:** `gates_reached` mean ≥ 3 **on fresh starts**, and `fresh_start` ≤ 0.32 (the pre-fix run ended at
   0.49, i.e. 36% forced reloads). **This is a combat gate, not a navigation gate.** On 0-1, `gates_reached ≥ 1`
   requires 11 kills in the gun room and `≥ 3` also requires the projectile arena, against F3's measured 0-1 kills
   per episode. A failure here does **not** mean the route signal is not working; read it with §6.3 and the `kill`
   and `door_unlock` parts.
3. **+1.5M steps:** `best_gate_hops` ≤ 2 on fresh starts, i.e. the agent reaches the pre-boss area on some episode.
4. **+2.5M steps:** `campaign.fresh_completion_rate > 0` with `fresh_window >= 20`.

**Novelty tripwire (wanderer regime, §6.2):** if at +1.5M `gates_reached` is still ≤ 1 on fresh starts **and**
`part_novelty` is above 70% of gross positive reward, cut `novelty` 0.2 → 0.1 and give it 400k steps before judging.

**Falsifier for the whole design:** if `gates_reached` is still ≤ 1 at +1.5M while `wedged_steps` is near 0 and
`part_gate_approach` is being paid, the route signal is not the binding constraint and the next suspect is combat
(F3: 0-1 kills in the post-checkpoint-3 Stray arena), not more shaping.

---

## 12. File-by-file change list

No file appears in both buckets.

### MOD WORK (C#, `mod/UltrakillAIBridge/`)

| file | change |
|---|---|
| `Plugin.cs` | `Version = "0.6.0"` |
| `Obs/CampaignObserver.cs` | room graph from `Door.activatedRooms` keyed by rounded room position, BFS from the nearest room-node ancestor of the chosen `FinalPit`, per-gate `hops` (§3.1) built and cached in `Scan()` with the closed-door `key`/`pos`; `BuildGates()` assigning `open`/`locked`/`active`/`controller_active` into the cached `JObject`s every step; `gates_ordered`, `gates_truncated`; `ChooseExit` secret-pit + mission-successor fix, frozen per level load (§3.2) |
| `Obs/ObservationBuilder.cs` | `ground_ray_center` (§3.3); `slow_mode`, `heavy_fall`, `crouching` in `BuildPlayer` (§3.4, `crouching` via `AccessTools`) |
| `Env/TrainingSpeed.cs` | the un-wedge postfix/watchdog (§8) — it is the existing home of "optional training behaviour active only while in control" |
| `Env/EpisodeController.cs` | `unwedge` and `unwedge_frames` config keys (§8.2's hold; the §8.3 watchdog was not needed) |
| `docs/protocol.md` | document `gates`, `gates_ordered`, `gates_truncated`, `ground_ray_center`, the three player flags, the `ChooseExit` rule and `unwedge_frames`; fix the stale "The player end snaps within 6 m" at line 120 (it is `PlayerSnapDistance = 25f`) |
| `docs/game-internals.md` | `Door.activatedRooms` as the room graph; **a Normal `Door` moves its OWN transform by `openPos`, measured (0, 5.75, 0) on every gate door of 0-1..0-5, while the `NavMeshObstacle` sits on the `Door Object` child and the `DoorController`s are siblings under `Door (Large) With Controllers (N)`**; `CheckPoint.Start` clones each room, deactivates the original and moves it +10,000 X, and `ResetRoom()` re-instantiates the clone on every respawn; `NewMovement.gc` is a `GroundCheckGroup` with **no** `ForceGroundCheck` — that method is on `GroundCheck`, reached through the private `instances` list; the `slowMode` wedge and the fields it depends on |

### PYTHON WORK (`python/`)

| file | change |
|---|---|
| `ultrakill_ai/campaign.py` | `GateProgress` (§5) |
| `ultrakill_ai/spaces.py` | slots 448-455 and the `target=` parameter on `campaign_block`/`pack_observation` (§4); `ACTION_NVEC_CAMPAIGN`, `action_space(campaign=)`, `decode_action` width inference, `noop_action(campaign=)` (§7.1) |
| `ultrakill_ai/rewards.py` | `gate`, `gate_approach` in `RewardConfig` and `CampaignStep`; paid before the missing-player return |
| `ultrakill_ai/env.py` | new config fields (§9.1); `action_space(campaign)`; look-mode resolution and the `look_mode` pop (§7.2); `GateProgress` lifecycle and call order in `reset`/`_campaign_reset`/`_respawn`/`_campaign_progress` (§5); target threaded into `_pack`; stuck-clock fixes (§9.2); wedge detector, end-reason precedence and the `stuck_streak` inclusion (§9.3); `_lifetime_steps` and the archive schedule (§9.4); `slide_min_hold` (§9.5); `_ground_point` centre ray (§9.6); info keys and `CAMPAIGN_INFO_KEYS` (§9.7) |
| `ultrakill_ai/progress.py` | new mean keys, `best_gates_reached`/`best_gate_hops` in `_snapshot` **and `_restore`**, `mean_gates_reached_100` in `history`, the `episodes.jsonl` writer reading `info` directly for non-numeric fields (§9.7, §9.8) |
| `scripts/train.py` | `KMP_BLOCKTIME`, `validate_args(False)`, `verbose` on resume (§10) |
| `scripts/add_look_mode.py` | **new** — the weight surgery with the mean-fold (§7.3) |
| `scripts/transfer_weights.py` | build the destination with `action_space(campaign=True)` and zero-pad the three new action rows (§7.1) |
| `scripts/poll_status.py` | new columns (§9.8) |
| `scripts/dashboard.py` | gates and wedged rows, fresh-start split, chart series (§9.8) |
| `scripts/campaign_check.py` | new check 6: the gates block is present, ordered, hop-monotone and unchanged after a respawn |
| `configs/campaign_0-1.yaml` | §6.1 weights, `pitch_limit_deg` 45, `door_unlock` 15, `run_name: campaign_gates`, header run commands |
| `tests/test_campaign.py`, `test_campaign_rewards.py`, `test_spaces.py`, `test_campaign_env.py`, `test_progress.py`, `test_campaign_config.py`, `test_look_mode_transfer.py`, `test_transfer.py`, `test_campaign_check.py` | §11.1 |

### INTEGRATION (after both land)

`CLAUDE.md` — layout, commands, the new run, the findings above, and the reward-weight table. Owned by whoever
merges second, so neither engineer conflicts on it.

---

## 13. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | **Inputs 448-455 carry real weight** (82.19% `partial`, §2), so reusing them displaces a working policy | §7.3 zeroes those first-layer columns in both stacks and folds their mean into the bias. Measured mean KL 0.0449 → **0.0262** and greedy-action change 45.6% → 29.6% — under one `target_kl` of displacement, but **not zero**. §11.3's gates are read against a policy that moved before step 0 |
| R2 | The door graph does not order a level | `gates_ordered: false` is a first-class case, tested offline and live (0-5, A2). Python falls back to today's exit vector. Levels beyond 1-1 are unverified |
| R3 | Skull-locked doors (0-2 onward) are unmodelled | Out of scope (sub-project 3). `locked` is already reported per gate, so a target can still be chosen; only the *why* is missing. On skull levels the ordering is well-defined and **still wrong as a route**: 1-1 requires fetching the blue skull from a `hops` 3 room and backtracking to the `hops` 2 altar, so monotonically decreasing hops is unwalkable there. A2 only asserts "1-1 ordered", which will pass |
| R4 | The `Door` transform sits metres above the walkable floor, so the reach test misses | The cylinder test (8 m horizontal, 6 m vertical), the `open` 2× case, and A10 measures it live on the real first gate |
| R5 | §8 cannot be made to hold with a postfix | The §8.3 watchdog is the fallback, and the §9.3 Python detector bounds the damage either way (45 decisions instead of 675) |
| R6 | Look mode 1 becomes an aim crutch and the free-look head never improves | `look_free/enemy/gate_frac` and per-dimension entropy are logged; `yaw_track`/`pitch_track` count mode 0 only. If mode 1 dominates and kills rise, that is a win for this sub-project and a question for the next |
| R7 | The action-space change invalidates every existing **campaign** checkpoint | §7.3 is the only migration path; `eval.py`/`keep_best.py` point at the new run only. Cyber Grind is untouched by design (§7.1), and its `best.zip` stays loadable by both the old and the new code |
| R8 | `poll_status.py` silently drops the new columns | Move `metrics_log.csv` aside when the run starts (§10) |
| R9 | Cutting `novelty` 0.5 → 0.2 removes forward pressure before the gate terms prove themselves | The route group is ~8× novelty for a route-walker (§6.2), and run gate 2 fails fast if not. Reverting `novelty` to 0.5 is a one-line rollback. **But in the wanderer regime novelty is still 87% of gross positive income** — the design makes that regime unprofitable rather than un-dominated, and the §11.3 tripwire watches it |
| R10 | The mod and Python land at different times | §3.7 degradation is a hard requirement on both sides, with an offline test for each fallback (§11.1) |
| R11 | Key strings recomputed on the Python side would not match | §3.5: keys are opaque, including the `#N` suffix |
| R12 | Level-load-scoped `best_dist` leaves respawn episodes with no approach income | **Taken, by lead ruling R1: `best_dist` is cleared in `reset_episode()`, i.e. episode scoped.** The per-gate dict bounds an episode's total by the route length either way, and a truncation bootstraps the same state, so ending an episode early cannot pay. The residual risk is the mirror image — a respawn episode re-earning ground an earlier one covered — which is bounded by the same polyline and is the price of giving the majority of episodes (checkpoint respawns) a dense signal at all. Rollback is one line: drop the `best_dist.clear()` from `reset_episode` |
| R13 | `_ground_point`'s centre ray re-keys 23.2% of the carried archive cells | Accepted and stated (§9.6). Novelty income rises slightly; `oob_frac` gets a fresh baseline from the new run's first 100 episodes |

### Deliberately not included (flagged for the lead, not decided here)

- **Do not charge `punch` before `campaign.level_started`.** 0-1's alternate start (`unlock_all_gear` ⇒
  `GearCheckEnabler` swaps in the short starting room) is sealed by two weak `Breakable` planks at (40,0,347.5) and
  (40,2,347.5); punching through them or sliding under the lower one (1.5 m gap against a 1.25 m slide height) is the
  only way out, and the same opening is the "slide vent" the wedge happens in. A punch charge before the player owns
  a weapon prices the only tool they have. One line in `compute_reward`; not added because it is outside D1-D6.
- **BepInEx's 5-copy cap is configurable** (`[Logging.Disk] Enabled = false`); more than 5 games is explicitly out of
  scope here.
- **The 8.87 s title wait** after the revolver pickup is dead time on every fresh 0-1 load and is only skippable with
  the Pause input. Mod-side, low priority.
- **The player-side NavMesh sample fails at 0-1's own spawn.** Pre-existing, unrelated to the gates work, and the
  `path` term is now 0 anyway.

---

## 14. Review dispositions

### Lead rulings during implementation (these override the text above where they conflict)

- **R1. `GateProgress.best_dist` is EPISODE scoped, not level-load scoped.** It is cleared in `reset_episode`
  (R12's rollback, taken as the default) and is *not* cleared by an in-episode death respawn. The `gate`
  milestone ladder stays level-load scoped exactly as §5 describes. Rationale: PPO returns are computed within
  an episode and a truncation bootstraps the same state, so ending an episode cannot be profitable; and most
  episodes are checkpoint respawns, which need the dense signal on ground earlier episodes already covered.
  §5, §11.1 and R12 were updated to match; implemented and tested
  (`test_gate_approach_is_re_earned_in_a_new_episode`, `test_gate_approach_survives_a_respawn`).
- **R2.** `ObsLayout` stays 479 with the §4 slot reuse and the mandatory zero + mean-fold surgery (the 487-input
  variant flagged at the end of this section stays flagged, not taken).
- **R3.** The items under "Deliberately not included" stay out of scope: no `punch` change before
  `level_started`, no more than 5 games, no title-wait skip.
- **R4 (integration).** The new run is named **`campaign_gates`**, not `campaign_ppo_gates`, and its starting
  weights are **`models/campaign_gates/look_init.zip`**, not `init.zip`. `explore_dir` follows `run_name`, so
  the name and the directory the archives are copied into have to agree or the run restarts exploration from
  nothing. §6.1, §10, §11.1 and `configs/campaign_0-1.yaml` were updated to match.

### From the three review lenses

Three lenses reviewed the previous draft. Everything below was re-verified against the code or the diagnosis data
while revising; the measurement scripts are in the scratchpad alongside the diagnosis artifacts.

### Adopted in full

| claim | where fixed | verification |
|---|---|---|
| `gate_approach`'s scalar `best_dist`, re-seeded on every target change, is a farming loop | §5 (per-gate dict, never re-seeded), §11.1 flapping test | Both reviewers derived it independently. The tier geometry is real: 0-1 has two `hops` 2 gates 66.7 m apart and 0-2/0-3/0-4/1-1 have multi-gate tiers at nearly every level (`bfs2.py`) |
| The weight surgery is not behaviour-preserving; "bit-identical on every input" is false | §2 table, §7.3 (measured KL + mandatory mean-fold), §11.1, R1 | Reproduced exactly: zero-only KL 0.0449 / 45.6% greedy change / \|ΔV\| 1.32; mean-fold 0.0262 / 29.6% / 0.77. `‖W[:,448:456]‖_F` 2.04 of 33.78 |
| `GateProgress` has no way to pick a target without paying, so slot 452 is 0 on every episode's first decision | §5 (`retarget` split, exact call order table), §4, §11.1 | `env.py:231` packs `reset()`'s observation with no campaign-progress call |
| `"wedged"` is invisible to the `stuck_repeats` escape hatch | §9.3 | `env.py:469` is the only feed into `_stuck_streak` |
| The wedge predicate counts ordinary airborne time | §9.3 | Measured 26.9% of all decisions, 1228.7/episode raw |
| `Door` transform moves 5.75 m when open, so `key`/`pos` are not static and the key flips | §3.1 rule 8, §12 game-internals row | `Door.cs:341-351` + scene dump: 0-1 11/11, 0-2 17/17, 0-3 12/12, 0-4 7/7, 0-5 3/3 all `doorType 0`, `openPos (0,5.75,0)`; 1-1 13/13 `doorType 1`, zero |
| `GroundCheckGroup.ForceGroundCheck()` does not exist and the call is unsafe before `heavyFall` is cleared | §8.2 step 5 + normative ordering, A5/A6 | `GroundCheckGroup.cs` has no such member; `GroundCheck.cs:97` does; `OnTriggerEnter`'s `heavyFall` branch (:333-385) damages enemies and `Bounce` (:240) teleports |
| §8.1's root cause and §8.2 steps 1-2 are contradicted by the code | §8.1 rewritten, step 1 deleted, step 2 labelled defensive | `NewMovement.cs:921` gate, `StartSlide` :2267-2274, `TryStartSlam` :1116, `Update` :701-704 and :740 |
| Goal-room rule 3 collapses the graph when rooms share a parent | §3.1 rule 4 (nearest room-node ancestor) + rule 2 (room nodes from gate candidates only) | Re-ran the BFS with the new rules: 0-1 reproduces §3.6 exactly (11 gates, 0..9); 0-5 unordered. Every level has a `Pit` GameObject on the pit's chain, so rule 2 is load-bearing |
| Room-node identity is undefined and `Scan()` reruns after rooms are destroyed | §3.1 rules 3, 7, 12 + the Python-side defence in §5 | `CampaignObserver.cs:12-24` documents the rescan triggers; `CheckPoint.cs:108-132` the clone-and-banish |
| `ProgressCallback.field()` would write `null` for `start_checkpoint` and `end_pos` | §9.7, §11.1 | `progress.py:188-190` and `_num` at :42-48 |
| `ACTION_NVEC` is unconditional and would break Cyber Grind and `transfer_weights.py` | §7.1 (`ACTION_NVEC_CAMPAIGN`), §12 | `env.py:122`, `transfer_weights.py:79` |
| The 24-entry cap truncates the wrong end; no truncation flag; `key` uniqueness unspecified | §3.1 rules 9 and 13, §3.6, §11.1 | Measured gate counts 11/17/12/7/3/13 |
| `campaign_block`'s signature never receives the target | §4 | `spaces.py:119` derives everything from the raw obs |
| The exit target saturates the gate scales | §4 (exit scales + ±4 clip) | 0-1's exit is ~195 m from spawn |
| `FakeLevel` cannot express the wedge or look-mode tests | §11.1 FakeLevel change list | `tests/test_campaign_env.py:122-135` |
| `pitch_limit_deg == 0` would weld mode 2's camera level | §7.2 `PITCH_BAND` | `EnvConfig` line 68 and `env.py:237` treat 0 as "off" |
| Archive schedule has no lifetime step counter | §9.4 | `env.py:210` resets `self._steps` |
| `_restore` and the history key were unnamed | §9.8 | `progress.py:107-124`, `:314-324`, `dashboard.py:433-452` |
| `gates_reached` is inherited by respawn episodes and will reproduce the false "peak by depth" | §9.8, §11.3 gates 2-3 | CLAUDE.md records `corr(fresh_start, checkpoints_level) = -0.975` |
| `look_mode`'s fate in `command` is unstated | §7.1 | `env.py:241` |
| §6.1 understates the config change (run name, header) | §6.1 | `configs/campaign_0-1.yaml` already says `campaign_ppo_ground` with a `campaign_ppo` header |
| `ChooseExit` rule 3 is time-varying; 9-2 and empty targets unhandled | §3.2 rules 4-5 | — |
| `active` filter missing from `reached` | §5 `_is_reached` | — |
| `open` is far more transient than the draft implied; `(open:false, locked:false)` is ambiguous at load | §3.1 rules 10-11, `controller_active` | `DoorController.cs:48-122`, `Door.cs:492-514`; `gates_0-1.txt` shows the gun-room door's controller inactive |
| The 35-name mission table duplicates `CAMPAIGN_LEVELS` and is unnecessary | §3.2 rule 2 (successor rule, no table) | After the `-S` filter exactly one pit survives on each of the six levels |
| Look modes 1/2 make the look head causally inert on those steps | §7.2 "Known and accepted" | — |
| `slide_min_hold`'s rationale over-fits and the latch may raise the jump-out wedge rate | §9.5 | — |
| A9 is a policy statistic with an unstated budget | §11.2 note under A9 | — |
| §9.2's kill/damage rationale cited a mixture artifact | §9.2 | 29 of 47 episodes have 0 kills, 2 have 1-10 |
| §11.3 gate 2 is a combat gate and will be misread as a route failure | §6.3, §11.3 gate 2 | `gates_0-1.txt`: the `hops` 9 door needs 3 + 8 kills |
| Arithmetic and citation nits: total is +488 not "~480"; the `arena_clear` "+160" ceiling is unsupported; index 455 is not constant; `visible` is at `ObservationBuilder.cs:213` | §6.2 rebuilt, §2 table, §7.2 | 0-1 has 4 `ActivateNextWave` with `last = 1`, two sharing the key `56,18,640` ⇒ 3 distinct keys ⇒ +30 is the ceiling; index 455 is 1.0 on 17.81% of states |

### Adopted with a correction to the reviewer's number

- **Wedge movement threshold: 0.10 m per decision, not 0.05 m.** One review proposed `dist3(pos, prev_pos) < 0.05`.
  Measured on all 32,022 decisions, that threshold **loses one of the five real wedge runs** — the 694-decision one
  at (145.4, 45.5, 664.2), whose per-decision movement has p50 0.030 m and p90 0.055 m, so it fragments below the
  45-decision hold. That is precisely the wedge F1 cites as evidence the state is general rather than a vent
  property. `0.10 m` keeps all five and leaves 14.7 uncredited candidates per episode, which run-crediting removes.
  §9.1 expresses it as `wedge_creep_mps = 1.5` so it is frameskip-independent. Table in §9.3.
- **Novelty in §6.2.** One review put the full-length-episode figure at "+163..+173 at weight 0.5". Re-measured from
  `ground_episodes.json`: the six full-length episodes mean **172.9** at weight 0.5 = **345.8 raw**, 743 new cells.
  Both tables in §6.2 now carry the measured number and the regime it belongs to.
- **Route length.** The two reviews measured the gate-to-gate polyline at 723 m and 736 m; §6.2 uses "723-736 m"
  and computes `gate_approach` from 723.
- **The flapping exploit is reachable on 0-1.** One review argued 0-1 is accidentally safe because reaching either
  `hops` 2 gate sets `best_hops = 2`. That only holds if the agent comes within reach: the bisector between the two
  gates is 33 m from each, far outside the 8 m radius, so the tier can be targeted and flapped indefinitely while
  `best_hops == 3`. The fix is the same either way; §5 states it as reachable on 0-1.

### Adopted in substance, implemented differently

- **"Pay a second `gate` instalment when the current target's `locked` goes false."** The mechanism is right — the
  route gradient is exactly flat at every arena-held door, which is F3 — but it needs no new term. **`door_unlock`
  goes 3.0 → 15.0** (§6.1, §6.3). It is already keyed by `Door.Unlock`'s rounded position, already paid once per
  level load by `MilestoneTracker` with the `mark_paid` rules, already tested, and on 0-1 the four arena-held doors
  are exactly the four gate doors. No new mechanism, no new farm surface, no new test.
- **"Append 8 new inputs (479 → 487) so KL is 0 by construction."** Rejected as the default: D3 fixes the layout at
  479 and every index unchanged, and the alternative would have to re-widen the first layer of a 2.9M-step policy
  anyway. The reviewer is right that D3's *stated premise* for the reuse is refuted, and that is now recorded in §2
  and R1. The cost is measured, bounded at 0.026 KL after the mean-fold, and stated rather than claimed away. **If
  the lead prefers zero displacement, the 487-input variant is a drop-in replacement for §4 and §7.3** — it changes
  `ObsLayout.campaign_block` to 44, `SHARED_INPUTS` stays 443, and `add_look_mode.py` zeroes the new columns exactly
  as `transfer_weights.py` already does. Flagged, not taken.
- **`best_dist` scope.** The two reviews disagreed: one wanted level-load scope, the other per-episode. This
  draft took **level-load**, matching `paid_hops` and `MilestoneTracker` exactly; **the lead then ruled for
  per-episode (R1 above), which is what is implemented.** Per-episode is not farmable once the dict is per-gate
  (an episode's total is bounded by the route length either way); the reason the draft hesitated was that it
  lets an agent which can cheaply end its own episode re-earn, and this project has twice paid millions of steps
  for an income stream nobody bounded. The ruling's counter-argument is that a truncation bootstraps
  `gamma * V(s_T)` on the same state, so there is no cheap way to end an episode in the first place.

### Rejected, with reasons

- **"Reset the stuck clock on damage dealt."** The previous draft's own proposal, now removed (§9.2). `dealt` is
  unattributed (`rewards.py:163-177`), enemies damage each other, and the measured Stray-arena tail would stop
  truncating — moving 13.6% of samples from stuck tails into the 51% already coming from 9,000-step timeouts.
  `kills` and `style` do the job and only move on player actions.
- **"Keep `min(ground_rays)` as the archive key and use the centre ray only for the out-of-bounds test."** The ring
  minimum is a measurement error, not a policy: its spread among hit rays is p90 **10.8 m**, so it regularly reports
  a ledge 4 m away as the ground under the player. Protecting comparability with an `oob_frac` baseline that is
  already broken by `novelty` 0.5 → 0.2 and a new run name is the wrong trade. §9.6 adopts the centre ray and states
  the 23.2% re-key and the fresh baseline instead.
- **"§5's `reached_hops` has no `active` filter, which could make `lower` empty and fall through to the exit — the
  rejected distance-to-exit reward."** The `active` filter is adopted (it is correct), but the stated consequence is
  wrong: `_choose_target` falls through to `EXIT` only when `best_hops` is 0 or no active gate has a lower `hops`,
  and in the latter case the exit is genuinely the only remaining target. It is not a distance-to-exit *reward* —
  `gate_approach` to the exit is still a monotone new-best measure bounded by one first-seen distance, not a
  per-step distance term.
