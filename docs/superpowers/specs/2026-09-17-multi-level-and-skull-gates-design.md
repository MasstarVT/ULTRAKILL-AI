# Multi-level curriculum, gates usability guard, 6-2 exit, skull-carry gates: design

Date: 2026-09-17. Revised 2026-09-17 against two independent reviews (`rl-contract`, `game`); every disputed
fact was re-measured before the change was written, and §11 records the disposition of each item. Status:
engineering contract; scope fixed by the project lead (S1-S4). Mod version target **0.7.0**; protocol stays
**1** (obs gains fields, nothing removed or changed).

Follows `2026-09-16-campaign-gates-unwedge-design.md` ("the gates spec") and
`2026-09-16-campaign-foundation-design.md`. Inputs: `docs/level-survey.md` and its raw
`…/scratchpad/levels/level_survey.json` (all 33 shipped levels, parsed offline), the scene bundles themselves
(re-parsed for §2's `activeSelf` facts), and `F:\Github\ULTRAKILL-AI\decompiled` (read-only).

**The mod and Python sides are implemented by two engineers who cannot talk to each other.** §2, §5 and §6.1 are
the contract. Each side must run against the other's old code (§3.6, §6.9), with an offline test per fallback.
Every number called *measured* was read out of the scene files or the decompiled source while writing this.

---

## 1. Scope

| # | Item | Contract |
|---|---|---|
| **S1** | Multi-level training with an automatic curriculum | optional ordered `env.levels`; the level changes only on a fresh load in `_campaign_reset`; sampling and unlocking driven by a small JSON file written by `ProgressCallback` |
| **S2** | Gates usability guard (Python only) | gates count as unordered unless at least **half** carry a non-null `hops` |
| **S3** | 6-2 exit tie-break (mod only) | an `Intermission*` target is a successor; a `Level P-n` Prime Sanctum pit is never the exit; a rank tie is logged |
| **S4** | Skull-carry gates from level data only | mod widens gate candidates to altar-driven doors, emits `campaign.altars[]`, `campaign.items[]`, `gates[].needs_item`; `GateProgress` grows a three-state sub-goal; two new milestones; the env protects a carry from the policy's own punch |

**Out of scope, decided:** checkpoint-chain route fallback for Tier D; boss end conditions; >5 game instances;
trams and water; any observation-width or action-space change; PPO hyperparameter changes.

**Invariant.** With `env.levels` absent, on a level with **no `ItemPlaceZone`** whose gates are all ordered, the
packed observation, every reward part, the action actually sent and every `status.json` field are **identical to
today, byte for byte**. Measured, `Level 0-1` is such a level (11/11 gates carry `hops`, 0 `ItemPlaceZone`), so
the live `campaign_gates` run is untouched by S1/S2/S4. S3 does not touch it either (one surviving pit, target
`Level 0-2`). Every new mechanism in §6 is gated on `gates[].needs_item` being non-null somewhere in the level,
which on 0-1 never happens.

## 2. What this rests on (measured)

- **M1 — the gates ladder is a Prelude/Act-I mechanism.** `gates_with_hops / gates` is 1.000 on 14 levels
  (0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 3-2, 4-1, 5-1, 5-3), then 0.960 (8-2), 0.600 (4-3),
  0.538 (8-1, 28/52), **0.500 (6-1, 6/12)**, **0.250 (7-2, 1/4)**, **0.031 (8-3, 1/32)**, and 0.000 on
  **13** levels — seven with gates but no hops (0-5, 1-3, 1-4, 4-2, 4-4, 7-4, 8-4) and six with no gates at all
  (2-4, 5-2, 5-4, 6-2, 7-1, 7-3). `gates_ordered` is `true` on 7-2 and 8-3, where `GateProgress` locks onto one
  far door and never retargets (8-3's single ordered gate is 892 m from the start). A `>= 0.5` guard keeps 6-1,
  drops 7-2 and 8-3. The largest gate array that ships is 8-1's 52, below `MaxGates` 64, so no shipped level is
  truncated (§4).
- **M2 — four levels are exit-ambiguous, and S3 fixes one of them.** Re-running today's filter
  (`fakeEnd || secondPit || rankless || empty target || "-S"`) and today's rank over `pits_all` for all 33
  levels: 29 leave exactly one candidate. The four that do not:
  | level | candidates | today | after S3 |
  |---|---|---|---|
  | **6-2** | `Intermission2`, `Level P-2` ×2 | all rank 3, tie → `FindObjectsOfType` order | P-2 filtered out, `Intermission2` ranks 1 — **unique** |
  | **3-2** | `Intermission1` ×2 | both rank 3, tie | both rank 1, **still tied** — benign only because both pits sit at the identical position `(0, -264.5, 262.5)` |
  | **2-4** | `Level 3-1` ×2 | both rank 1, tie | unchanged, **still tied** — benign: 1.4 m apart, `(363, -80.1, 590.0)` and `(363, -80.1, 591.4)`, same room |
  | **3-1** | `Level 3-2`, `Level P-1` | ranks 1 and 3 — already resolved | unchanged (the P- filter drops the loser earlier) |
  8-4's `EarlyAccessEnd` parses as neither a level nor an Intermission and is picked by **uniqueness alone**, not
  by the successor test. Campaign-wide only 3-2 and 6-2 ship an `Intermission*` pit and only 3-1 and 6-2 ship a
  `Level P-*` pit, and no main level's legitimate successor starts with `Level P-`.
- **M3 — altars and the gate ladder.** **21** of 33 levels carry at least one `ItemPlaceZone` with
  `acceptedItemType != None` (0-2, 1-1, 1-2, 1-3, 1-4, 2-3, 2-4, 4-2, 4-3, 4-4, 5-1, 5-2, 5-3, 6-1, 7-1, 7-2,
  7-3, 8-1, 8-2, 8-3, 8-4). Counting distinct forward doors those altars drive: **46 altar→door wirings over 28
  distinct doors, of which only 6 are gates today** (1-1 `20,-10,381` hops 2; 1-2 `0,-15,380` hops 2; 2-3
  `-67,8,375` hops 0 and `47,-9,385` hops 1; 4-4 `108,648,425` **hops null**; 8-2 `285,-10,431` hops 6). A
  skull-locked door is overwhelmingly a **one-room streaming door**, which `ScanGates` rejects at
  `if (roomIds.Count < 2) continue;` (`CampaignObserver.cs:286`), so it can never carry `needs_item`. **This is
  why §6.1 widens the gate candidate set**; see M4.
- **M4 — widening pays off on exactly three levels, one of which is 1-1.** Reproducing `ComputeHops` offline
  with the room graph built **only** from ≥2-room doors (unchanged) and then reading `min(roomHops[r])` over the
  rooms an altar-driven one-room door does list, the door gains a usable `hops` on 1-1 (`81,-6,240` → **hops
  1**, room `81,-6,239` "12 - Hall", which sits between the Skull Field and the hops-0 gate `81,-6,188` into
  "13 - End Room"), 5-3 (`-66,15,394` → hops 10) and 8-1 (`205,110,476` → hops 3). On every other level the
  room it lists is not in the room graph, so `hops` stays null and `_choose_target` drops it exactly as today.
  **Coverage after the widening** (gate with `hops` *and* `needs_item`): 1-1 **both legs**, 2-3 **both legs**,
  1-2 red only, 8-2 blue only, 5-3, 8-1. 4-4's altar-gate carries `hops: null` and 4-4's ratio is 0.000, so S2
  drops its gates anyway.
- **M5 — a skull-locked door looks exactly like a walk-up door today.** `ItemPlaceZone.ColorDoors` →
  `Door.AltarControlled()` (`decompiled/Door.cs:320-335`) **deactivates every `DoorController`** of the door it
  drives, so the gate reports `open: false, locked: false, controller_active: false` — the same signature the
  gates spec gave 0-1's arena-held gun-room door. `needs_item` is the only field that separates "kill 11
  enemies" from "fetch a skull".
- **M6 — `punch` already picks up and places; no new button, but four physical facts have to hold.**
  `Punch.ActiveFrame` raycasts 4 m from `cc.GetDefaultPos()` along `camObj.transform.forward` against
  `environmentMask` and calls `AltHit(hit.transform)` (`Punch.cs:798-812`); it first tries
  `Physics.CheckSphere(cc.GetDefaultPos(), 0.01f, environmentMask, QueryTriggerInteraction.Collide)` and calls
  `AltHit` on everything overlapping the camera. `AltHit` (`Punch.cs:1335-1362`) acts on `layer == 22`:
  holding + the target has `ItemPlaceZone`s → `PlaceHeldObject`; not holding + the target has an
  `ItemIdentifier` → `ForceHold`. `ActionInjector.TapButtons` and `spaces.BUTTONS` already carry `punch`.
  Verified while writing this: (a) `Punch.Awake` builds `environmentMask = LayerMaskDefaults.Get(LMD.Environment)
  | 0x400000`, i.e. layer 22 is explicitly included (`Punch.cs:127-128`); (b) every 1-1 `ItemPlaceZone` carries a
  **trigger** `BoxCollider` on layer 22 and every `ItemIdentifier` a non-trigger one, and the 4 m raycast passes
  no `QueryTriggerInteraction`, so it depends on `Physics.queriesHitTriggers`, which is **never assigned
  anywhere in `decompiled/`** and therefore keeps Unity's default `true`; (c) the `AltHit` block is outside the
  `FistType.Standard` guard (the guard at `Punch.cs:549` closes before the block at `798`), so either fist
  works; (d) **`AltHit` returns immediately when the hit transform itself carries an `ItemIdentifier` and the
  player is holding** (`if ((bool)itemIdentifier && hasHeldItem) return;`), so a placement punch must hit the
  zone's own `Cube`, not a skull already resting on it.
- **M7 — `ActiveStart`/`ActiveEnd` are Unity AnimationEvents, and that is S4's single point of failure.**
  Grepping all 1214 decompiled files finds no C# caller for either (only `CancelAttack` → `ActiveEnd`).
  `ActiveFrame` — the only `AltHit` call site, i.e. the only pickup and the only placement — runs from
  `ActiveStart` and from `FixedUpdate` **only while `activeFrames > 0`**, which `ActiveStart` sets. Training runs
  with `render: false`, and `TrainingSpeed.ApplyRendering` disables every `Camera`; the file already forces
  **enemy** Animators to `AlwaysAnimate` "so animation-driven attacks don't freeze when nothing is rendered"
  (`TrainingSpeed.cs:82-86`) but does not touch the player's fist Animator. §6.8 makes it unconditional and §8
  check 1 proves it in game before anything else in S4 is trusted.
- **M8 — punch cadence and the throw.** Gated on `ready && fc.fistCooldown <= 0 && punchStamina >= 1`
  (`Punch.cs:432`); a Standard punch sets `fc.fistCooldown = cooldownCost * 0.25f` = **0.5** and spends
  `cooldownCost / 2` = 1 of 2 stamina (`Punch.cs:497`), `FistControl.Update` drains the cooldown at
  `Time.deltaTime * 2f` so it clears in **0.25 s**, and stamina regenerates at 1.25/s
  (`WeaponCharges.cs:257-259`) — a sustained ~1 punch per 0.8 s with bursts of two, not "one per 0.5-0.8 s".
  **`Punch.ActiveStart` throws whatever it is holding when the first active frame did not place it**
  (`bool num = holding; … if (num && holding && heldItem != null) ForceThrow();`, `Punch.cs:514-531`;
  `ForceThrow` adds `(forward + up*0.1) * 5000`). The live policy presses punch on ~35% of decisions (~5/s at
  15 decisions/s), so an unprotected 134 m carry survives for a fraction of a second. **Reverse direction:**
  `AltHit` with `!holding && itemIdentifier != null` calls `ForceHold`, whose tail re-runs `CheckItem()` on the
  zones the item was parented to (`Punch.cs:395-403`), and `ItemPlaceZone.CheckItem`'s empty branch calls
  `doors[i].Close()` (`ItemPlaceZone.cs:268-275`) — punching a filled altar takes the skull back out **and shuts
  the gate it opened**. §6.7 is the answer to both.
- **M9 — the arm is available.** `FistControl` accepts an arm when
  `PrefsManager.GetInt("weapon."+name, 1) == 1 && GameProgressSaver.CheckGear(name) == 1` (`FistControl.cs:294`);
  `CampaignPatches.OverridePrefInt` and `UnlockGear` already force both to 1 under `unlock_all_gear`.
- **M10 — `activeSelf` alone does not identify a carryable; `activeSelf` + one inactive ancestor does.**
  Re-parsed from `campaign_scenes_level1-1.bundle`: **three** of 1-1's five `ItemIdentifier`s have
  `activeSelf == true`, not one. Walking each item's ancestor chain and counting inactive GameObjects
  (the item's own included) separates them cleanly:
  | item | pos | `activeSelf` | inactive ancestors | what it is |
  |---|---|---|---|---|
  | SkullRed | `(81.0, -2.2, 275.0)` | true | **1** (`3 - Skull Field`) | the real red source, on the pedestal |
  | SkullBlue | `(-15.0, 26.64, 427.0)` | true | **1** (`11 - Blue Skull Room`) | the real blue source |
  | SkullBlue | `(-15.0, 26.8, 427.0)` | true | **2** (`Altar`, `11 - Blue Skull Room`) | phantom: its parent `Altar` node is switched off, so it is never `activeInHierarchy` even after the room loads |
  | SkullBlue | `(81.0, -3.86, 251.0)` | false | 2 | `activateOnSuccess` decoration in the destination altar |
  | SkullRed | `(0.0, -6.86, 381.0)` | false | 2 | ditto |
  The rule **`active_self && inactive_ancestors <= 1`** (one inactive ancestor is the room switch; two is a dead
  branch) was checked against every `ItemIdentifier` on all 21 altar levels: combined with "the type is accepted
  by some altar in this level" it leaves **at most one source per accepted type on 17 of them**, and a small
  correct set on the four puzzle levels (2-3: 2 red + 2 blue, 1-4: 4 blue for 6 blue altars, 5-1: 3 blue,
  7-1: 2 + 2 + 1). On 8-2 it reduces 30 `ItemIdentifier`s (22 of them `CustomKey1` props; 13 with
  `activeSelf == true`) to one red and one blue.
- **M11 — pre-filled altars are real and must not pay.** An `ItemPlaceZone` whose own room switch is its only
  inactive ancestor and which holds a matching live `ItemIdentifier` reads `filled: true` the moment its room
  activates, because `ItemPlaceZone.CheckItem` uses `GetComponentInChildren<ItemIdentifier>()` **without**
  `includeInactive` (`decompiled/ItemPlaceZone.cs:160`). Counted campaign-wide with that rule: **31** such zones
  over 18 levels (1-1: 2 — the red pedestal `81,-2,275` and the blue source `-15,27,427`; 1-2: 2; 2-3: 3;
  5-1: 3; 7-1: 3; 8-1/8-2/8-3/4-2/5-2: 2 each; …). None of them is touched by the agent. §6.5 is why they pay
  nothing.
- **M12 — door keys do not collide across the gate boundary.** Rounding every `Door`'s authored world position
  with the mod's `Mathf.RoundToInt` rule on all 33 levels: **zero** gate keys collide with a non-gate door key,
  so widening the key namespace to every door (A2) never re-suffixes an existing gate key — 0-1's eleven keys
  are provably untouched. Exactly one level has a gate-internal collision, 1-2's `0,20,380` ×2, which is the one
  place A3's deterministic ordering changes an existing behaviour (see A3).
- **M13 — no shipped level has an `infiniteSource` item.** Checked every `ItemIdentifier` on all 33 levels:
  `infiniteSource` is false everywhere. `AltHit` mints a fresh `ItemIdentifier` per punch for such an item
  (`Punch.cs:1346-1349`), which would be an unbounded key stream, so the mod skips them (§6.1) rather than
  relying on the measurement.
- **M14 — one altar of every duplicate pair is a dead branch that can never be filled, and this is not
  cosmetic** (found while re-measuring for this revision; neither review raised it). Applying A6's chain count to
  `ItemPlaceZone`s: **20 of the campaign's 104 functional zones have `inactive_ancestors > 1`**, spread over 11
  levels (5-3: 5, 1-1: 3, 1-4: 3, 1-2: 2, and one each on 0-2, 2-3, 2-4, 4-3, 5-2, 7-1, 7-2). On 1-1 every
  duplicate pair is exactly one live `Altar (… Skull) Variant` (`inactive_ancestors: 1`) plus one dead plain
  `Altar` (`2`). A dead zone never activates, so its `CheckItem` never runs and it reads `filled: false`
  **forever**. If `needs_item` counted it, the lock would never clear after a successful placement: the machine
  would drop back to state 1, find the just-placed skull "free" (its `placed_in` is the *live* twin, not the
  unfilled dead one), send the agent to punch it out of the altar — and `ItemPlaceZone.CheckItem`'s empty branch
  would **close the gate again** (M8). Hence A6 applies to altars as well as items, on both sides. Measured
  safety check: **no door anywhere in the campaign is driven only by dead zones**, so filtering them never
  removes a lock that has no live alternative.

## 3. S1 — multi-level training with an automatic curriculum

### 3.1 Config

`EnvConfig` gains (campaign-only, all inert when `levels` is empty):

```python
levels: list[str] = field(default_factory=list)  # ordered; empty = single-level, today's behaviour
curriculum_path: str = ""      # runs/<run>/curriculum.json; "" = no curriculum
unlock_rate: float = 0.5       # fresh completion rate that unlocks the next level
unlock_window: int = 20        # fresh episodes a level needs before its rate can unlock anything
level_weight_floor: float = 0.1
```

`level` stays and remains the single-level setting; when `levels` is non-empty it is ignored and `levels[0]` is
the start. Every name must be in **`campaign.CAMPAIGN_LEVELS_SHIPPED`**, a new frozenset of the 33 levels whose
scene bundle ships — *not* `CAMPAIGN_LEVELS`, which still lists `Level 9-1` and `Level 9-2` (no bundle ships, so
they cannot load). `UltrakillEnv.__init__` raises otherwise. `train.fill_campaign_dirs` fills `curriculum_path`
with `(run_dir / "curriculum.json").as_posix()` when `mode == "campaign"` and `levels` is non-empty and the
config left it empty — same pattern as `explore_dir`.

### 3.2 `runs/<run>/curriculum.json`

Single writer: `ProgressCallback` in the trainer's main process, via the existing `progress.write_json_atomic`
(temp file + `os.replace`, already retried on Windows `PermissionError`). Readers are the `SubprocVecEnv`
workers: one `read_text` + `json.loads` in `try/except (OSError, ValueError)`.

```json
{"version": 1, "updated_at": 1789000000.0, "run_name": "campaign_prelude",
 "order": ["Level 0-1", "Level 0-2", "Level 0-3"],
 "levels": {
   "Level 0-1": {"unlocked": true,  "fresh_window": 50, "fresh_completion_rate": 0.62, "best_time": 141.2, "episodes": 812},
   "Level 0-2": {"unlocked": true,  "fresh_window": 23, "fresh_completion_rate": 0.13, "best_time": null,  "episodes": 188},
   "Level 0-3": {"unlocked": false, "fresh_window": 0,  "fresh_completion_rate": null, "best_time": null,  "episodes": 0}}}
```

- `levels` always lists **every** name in `order`, including ones that have never produced an episode, so no
  reader has to invent a record.
- `fresh_window` / `fresh_completion_rate` are over that level's own last `FRESH_WINDOW` (50) fresh-start
  episodes.
- `best_time` is the fastest fresh-start official time ever seen for the level in this run, or `null`.
- `unlocked` is a **latch**: never set back to `false` within a run, and it survives a restart because
  `ProgressCallback._restore` reads it back from `status.json`. Without the latch a level re-locks as its rate
  falls and its in-progress learning stalls.
- `order` is `cfg.levels` as the trainer saw it. A worker whose `cfg.levels` disagrees ignores the file (a stale
  file from another run) and uses `cfg.levels[0]`.
- **`train.py` writes the first `curriculum.json` before `model.learn()`**, because SB3's `_setup_learn` calls
  `env.reset()` before `_on_training_start` ever runs: without it the workers' first fresh start reads whatever
  is on disk, possibly a file from an aborted attempt with the same run name. `train.py` also prints the
  restored unlock set at startup, so a `run_name` change silently re-locking every level (`_restore` bails
  whenever `status.json`'s `run_name` differs, `progress.py:133`) is visible in the log.

### 3.3 Sampling and unlocking

Two pure functions in `campaign.py`, tested offline. **Both read `stats` through one default record**, because
`ProgressCallback.per_level` only gains an entry once a level has finished an episode:

```python
def _level_stat(stats, lv, first):
    st = stats.get(lv)
    if not isinstance(st, dict):
        return {"unlocked": lv == first, "fresh_window": 0, "fresh_completion_rate": None,
                "best_time": None, "episodes": 0}
    return st

def choose_level(rng, order, stats, *, floor=0.1) -> str:
    """The level the next fresh load uses."""
    unlocked = [lv for lv in order if lv == order[0] or _level_stat(stats, lv, order[0])["unlocked"]]
    weights  = [max(floor, 1.0 - (_level_stat(stats, lv, order[0])["fresh_completion_rate"] or 0.0))
                for lv in unlocked]
    return rng.choices(unlocked, weights=weights, k=1)[0]

def unlock_next(order, stats, *, unlock_rate=0.5, unlock_window=20) -> str | None:
    """The first still-locked level whose predecessor has earned it, or None. Called once per finished episode."""
    for i in range(1, len(order)):
        if _level_stat(stats, order[i], order[0])["unlocked"]:
            continue
        prev = _level_stat(stats, order[i - 1], order[0])
        if (prev["unlocked"] and prev["fresh_window"] >= unlock_window
                and (prev["fresh_completion_rate"] or 0.0) >= unlock_rate):
            return order[i]
        return None            # the chain stops at the first locked level
    return None
```

`order[0]` is unlocked by the default record, so the chain always starts and an empty `stats` returns `None`
rather than raising. `ProgressCallback` also creates `per_level[order[0]]` with `unlocked=True` at
`_on_training_start`. A level with no data has rate `None` → weight 1.0 (maximum attention). All rates 1.0 →
every weight is `floor` → uniform retention sampling, intended. Unlocking is chained: 0-3 cannot unlock before
0-2 has.

**The weight controls fresh *draws*, not episodes.** A worker re-draws only at a fresh start, and
`choose_fresh_start` keeps it on the level it reached a checkpoint in for ~`1/fresh_start_prob` = 5 episodes,
while a level where no checkpoint is ever reached forces a reload — and so a re-draw — every episode. Realised
episode share is weight × dwell. It points the right way (a completion sets `level_over`, which also forces a
redraw), but **the number to read is `campaign.levels[*].episodes`**, not the weight. The workers' `self._rng`
is unseeded today (`train.py` never calls `set_random_seed`), which is what keeps the five draws independent;
adding a `--seed` later would make all five workers walk the same level sequence in lockstep.

### 3.4 Env changes

`self.cfg.level` is read in **eight** places (`env.py:196` `scene`, `596` `_archive_path`, `651` the
`choose_fresh_start(in_level=…)` test, `658` `_campaign_reset`'s reset call, `675` `_respawn`'s reset call,
`832`/`845`/`856` `_save_best_run`). Replace all with a **mutable `self.level`**, initialised to
`cfg.levels[0] if cfg.levels else cfg.level`; `UltrakillEnv.scene` returns it.

`_campaign_reset`, in order: (1) evaluate `choose_fresh_start` **against the outgoing level** — line 651's
`in_level` test must compare `prev["scene"]` with `self.level` *before* any switch; (2) **only when `fresh` and
`cfg.levels`** read `curriculum_path` (last good cached copy on failure, else `{}`), call `choose_level` and
compute `new_level`; (3) if `new_level != self.level`, call `_switch_level(self.level, new_level)`, which saves
the outgoing archive under the **outgoing** level's path and only then assigns `self.level = new_level`;
(4) `client.reset(self.level, checkpoint=not fresh)`; (5) `new_level_load` / `mark_paid` as today.

`_switch_level(previous_level, new_level)` takes both names explicitly — an implementation that reads
`self.level` for the save path after assigning writes the outgoing archive under the incoming level's filename
and corrupts both. It saves the outgoing archive (guarded by the existing `except OSError` print), swaps
`self.archive` out of `self._archives: dict[str, ExplorationArchive]` (loaded from disk on a level's first use,
kept in memory after, so switching back does not re-read), and clears `self._stuck_streak` /
`_stuck_checkpoint`, which are level-scoped. `_save_archive`, `_save_archive_on_schedule` and `close()` iterate
`self._archives` and save all of them. `_archive_path(level)` and `_save_best_run` take the level explicitly;
`best_runs_dir` already keys by `safe_name(level)`.

**`_respawn` never re-samples.** When there is no current checkpoint it calls `client.reset(self.level,
checkpoint=True)` and the mod's `StatsManager.Restart` reloads the whole level *inside* the episode. That is a
fresh level load and it must keep the level it is on: sampling lives in `_campaign_reset` and nowhere else.
Sampling there would change level mid-episode and corrupt `info["level"]`, the best-run positions and the
archive.

`_info` adds `info["level"] = self.level` in campaign mode. **`level` is not added to `CAMPAIGN_INFO_KEYS`** —
those go through `Monitor(info_keywords=…)` and `ProgressCallback._num`, both numeric; `ProgressCallback` reads
`info["level"]` directly and lists it in `EPISODE_LOG_RAW`, which already bypasses `_num`. `info` is built in
`step()` before `SubprocVecEnv` calls `reset()`, so a switch episode's row carries the level it **ran on**; §8
pins that. **No level id enters the observation**: generalisation is the point, and a one-hot would change the
width.

### 3.5 Per-level stats, and who reads them

`ProgressCallback` gains `per_level: dict[str, dict]` with, per level: `fresh` (a `deque(maxlen=FRESH_WINDOW)`
of `(completed, seconds)`), `best_time`, `unlocked`, `episodes`, and the running means of `checkpoints_level`
and `gates_reached`. Restored in `_restore`, rewritten to `curriculum.json` when it changes (at most at the
existing 2 s `_write` cadence) and once in `_on_training_start`.

`status.json`'s existing `campaign` block gains `order` and a `levels` table:

```json
"levels": {"Level 0-1": {"unlocked": true, "fresh_window": 50, "fresh_completion_rate": 0.62,
                         "median_time_50": 152.0, "best_time": 141.2, "episodes": 812, "weight": 0.38,
                         "checkpoints_level": 4.1, "gates_reached": 3.2}}
```

`checkpoints_level` and `gates_reached` are in the table because they are the early-progress signals that judge
a run before any completion exists, and the pooled `mean_100` versions of them (and of `cells_new`, `oob_frac`,
`exit_dist_min`, `best_checkpoints_level`, `best_gates_reached`, `best_gate_hops`) become cross-level
maxima/means that mean nothing once two levels of different size run together. `poll_status.py` still logs the
pooled columns; **read the table, not the headline** (R6).

**With `cfg.levels` empty the `campaign` block is byte-identical to today** — no `order`, no `levels`, and the
four existing numbers computed exactly as they are now. With `levels` set:

| key | multi-level definition | why |
|---|---|---|
| `fresh_completion_rate` | `Σ_unlocked completions_i / max(n_i, unlock_window)` | a **monotone shrunk sum**: a newly unlocked level contributes 0, so an unlock can never lower it; it rises whenever any level improves; and with one unlocked level at `n ≥ unlock_window` it is exactly today's rate |
| `fresh_window` | `len(pooled fresh deque)` — today's exact meaning | keeps `keep_best`'s `MIN_FRESH_WINDOW = 20` guard from stalling: a per-level **minimum** would stall forever, because `_restore` does not restore `fresh_recent` and the slowest window to refill after a restart is the *mastered* level at weight 0.1 |
| `median_time_50` | median over the pooled fresh completions of unlocked levels | unchanged shape |
| `best_time` | global minimum over all fresh completions — today's exact meaning | monotone; it is only `keep_best`'s tie-break on an equal 4-dp score |

That aggregate is what makes **`keep_best.py` need no edit**: `keep_best.main` loads `stored_best` once and only
replaces it through `is_better` (strict `>` on `rank_key`, `keep_best.py:108-112,196`), so any aggregate that
*drops* at an unlock — a mean over unlocked levels does exactly that, from ~0.52 to ~0.26 — would freeze
`best.zip` on a single-level policy and print the "15% below best" warning forever. The sum cannot drop. Note
the number is a **score, not a rate**, once more than one level is unlocked (it can exceed 1.0); the dashboard
prints it as `score 1.34 / 3 levels` and shows the real rates per level.

`poll_status.py` gains one column, `levels_unlocked` (int), plus `part_item_pickup` / `part_item_placed`, and
nothing else — a column per level would break its "keep the existing header" rule the moment a level unlocked.
`dashboard.py` gains one Campaign-panel line per unlocked level
(`0-1  fresh 62% (50)  best 2:21.2  cp 4.1  w 0.38`, via `times.short_level`) and the short level name in the
per-game row.

`eval.py` must not silently run the wrong level, because `--record-times` writes a row that only a *faster* time
can ever replace:

- `--level L` given: `cfg.level = L` **and `cfg.levels = []`**;
- `--level` absent and `cfg.levels` non-empty: `cfg.level = cfg.levels[0]`, then `cfg.levels = []`, and print it;
- both cases: clear `cfg.curriculum_path` alongside the `explore_dir` / `best_runs_dir` it already clears, so an
  eval never reads or perturbs a live run's curriculum;
- belt and braces: `report_campaign`, the `times.md` entry and the `explore_*.npz` path read **`env.level`**,
  never `cfg.level`.

### 3.6 Races, atomicity, and what this does not solve

One writer, N readers, no lock — the write is atomic, and a reader that catches a torn or missing file keeps its
last good copy. A worker re-reads at every fresh start, so it is at most one episode stale (irrelevant at
thousands of decisions). Workers never write the file, which is what makes the lockless design correct. Missing,
unreadable, empty, wrong `run_name`, or `order` disagreeing with `cfg.levels` ⇒ **first level only**.

`max_steps` stays one number for every level. 0-1's ladder is 500 m; 5-3's is 797 m; 8-2's is 836 m with a
1644 m fork, so a curriculum reaching Act III on a 9000-decision cap truncates those levels by construction (R6).

**Degradation:** S1 is entirely Python-side, so new Python against an old mod and old Python against a new mod
are both unaffected; a config with no `levels` key gives `levels == []`, which skips every path above.

## 4. S2 — gates usability guard (Python only)

`EnvConfig.gate_hops_min_frac: float = 0.5`. `GateProgress._gates` **keeps both existing guards** and gains one
condition, applied to the array **as sent**:

```python
if not campaign or not campaign.get("gates_ordered"):
    return []                                      # unchanged first line: campaign is None on many frames
gates = [g for g in (campaign.get("gates") or ()) if isinstance(g, dict) and g.get("pos")]
if not gates:
    return []
with_hops = sum(1 for g in gates if g.get("hops") is not None)
if with_hops / len(gates) < self.hops_min_frac:   # `<`, so 6-1's exact 0.500 is kept
    return []
return gates
```

The `campaign is None` line is not optional: `env._campaign_progress` passes `raw.get("campaign")`, which is
absent whenever `CampaignObserver.IsCampaignScene` is false or a frame lands during a scene load, and
`tests/test_campaign_env.py:399` drives exactly that path with `fake.drop_campaign_steps = 2`. The
`gates_ordered` test is now **redundant but kept**: the mod sets `gatesOrdered = gates[0].Hops.HasValue` after a
hops-first sort (`CampaignObserver.cs:344`), so `gates_ordered == false` is exactly `with_hops == 0`, which the
ratio guard already rejects.

Measured: no change on the 14 fully-ordered levels or on 8-2 (0.960), 4-3 (0.600), 8-1 (0.538), 6-1 (0.500);
**7-2 (0.250) and 8-3 (0.031) fall back to the exit vector**, which is the point. 0-1 is 1.000 and
byte-identical. **6-1 is kept on the threshold, not on evidence** — six of its twelve gates carry hops and
nothing in the survey says the reachable half is the half on the route; if 6-1 ever wedges, this is the first
knob. The ratio is computed on the array after the mod's nearest-64 truncation, which would make the guard
player-position-dependent on a level with more than 64 gates; the largest that ships is 8-1's 52, so this is
latent, not handled. Evaluate on every call, never cached: a `Scan()` can change the ratio mid-load as rooms
activate, and ≤64 dicts per call is negligible.

## 5. S3 — 6-2 exit tie-break (mod only)

Three edits, all inside `CampaignObserver`:

1. The `FinalPit` loop in `Scan` also discards a pit whose `targetLevelName` starts with `"Level P-"` (ordinal,
   case-insensitive) — a Prime Sanctum. This joins the existing empty / `-S` tests.
2. **The Intermission test goes in `ChooseExit` (line ~532), not in `IsSuccessor`.** `IsSuccessor` receives
   `(int act, int mission)?` from `MissionNumber(pit.targetLevelName)`, which is `null` for `"Intermission2"`,
   so it has no target string to test and the rule cannot live inside it without a signature change. The rank
   line becomes:
   ```csharp
   var target = pit.targetLevelName;
   bool successor = (target != null && target.StartsWith("Intermission", StringComparison.OrdinalIgnoreCase))
                    || (current.HasValue && IsSuccessor(current.Value, MissionNumber(target)));
   ```
3. `ChooseExit` logs one warning when two candidates tie at the best rank, so the next 6-2-shaped bug announces
   itself instead of being found by a stalled run. `rank >= bestRank → continue` and its `FindObjectsOfType`
   tie-break are otherwise unchanged.

Measured (M2): 6-2 drops both `Level P-2` pits at step 1 and ranks its single surviving `Intermission2` pit as a
successor, so the choice is deterministic. **3-2 is *not* fixed** — both of its surviving pits target
`Intermission1`, so the new rule ranks them equally and the tie still falls through to `FindObjectsOfType`
order; it is benign only because both sit at the identical position `(0, -264.5, 262.5)`. **2-4 is not touched**
and keeps a genuine two-way tie 1.4 m apart, same room and same successor. 3-1 already resolved. 8-4's
`EarlyAccessEnd` is still not a successor and is still picked by uniqueness alone — a silent dependence on
uniqueness in exactly the class of level (an act finale) where 3-2's and 6-2's duplicates live, which is why
rule 3 exists. `docs/protocol.md`'s `exit` bullet gains all three rules.

## 6. S4 — skull-carry gates

### 6.0 Gate candidates widen to altar-driven doors (mod)

`ScanGates` runs in two phases so the room graph, the BFS and every existing gate keep their exact current
values:

1. **Phase 1, unchanged:** doors whose `activatedRooms` hold ≥2 distinct GameObjects become gates, contribute
   the graph's nodes and edges, and `ComputeHops` BFSes from `GoalRoom()`.
2. **Phase 2, new:** every door referenced by the `doors` list (forward doors only) of an `ItemPlaceZone` that is
   not a dead branch (A6/M14) and that phase 1
   rejected is appended as a gate with `altar_only: true`, its `hops` read as `min(roomHops[r])` over the rooms
   it *does* list (`null` when none of them is in the graph), its key from the shared door-key table (A2), and
   its `Controllers` found the same way. It adds **no** node and **no** edge, so no existing hop count can
   change. `MaxGates` truncation, `CompareGates` and `gatesOrdered` then run over the combined list as today.

Measured (M4): this is what makes 1-1's blue leg exist at all (`81,-6,240` → hops 1), and it also lights up 5-3
and 8-1. On every other altar level the door's room is outside the graph, `hops` stays `null`, and
`_choose_target`'s `hops is not None` filter drops it exactly as today. On a level with no `ItemPlaceZone`
(0-1) phase 2 appends nothing.

### 6.1 JSON interface

Added to the `campaign` block; nothing existing changes meaning, type or units.

**`campaign.altars`** — one entry per `ItemPlaceZone` in the level with `acceptedItemType != ItemType.None`,
found with `FindObjectsOfType<ItemPlaceZone>(true)` (inactive included) and filtered by the existing
`IsTemplate` ancestry test. Dead branches are reported too, with their `inactive_ancestors`, and Python filters
them — only `needs_item` (computed mod-side) applies A6 on the wire. Rebuilt in `Scan()`; only `filled`,
`active` and `inactive_ancestors` are read per step. Sorted by `key` ascending.

| field | type | meaning |
|---|---|---|
| `key` | string | `CampaignPatches.Key(zone.transform.position)`, `#2`/`#3` on collision (A3). Opaque |
| `pos` | `[x,y,z]` | the zone's own world position (the trigger `Cube` — what the punch raycast must hit) |
| `item` | string | `acceptedItemType` enum name: `SkullBlue`, `SkullRed`, `SkullGreen`, `Readable`, `Torch`, `Soap`, `CustomKey1..3` |
| `filled` | bool | rule A1 |
| `active` | bool | `gameObject.activeInHierarchy` |
| `inactive_ancestors` | int | inactive GameObjects on its own chain, itself included (A6) |
| `doors` | array | `[{"key","pos"}]` per non-null `ItemPlaceZone.doors` entry (the doors it **opens**), keys from the shared door-key table (A2) |
| `reverse_doors` | array | same shape, for `reverseDoors` (the doors it **closes**) |

**`campaign.items`** — one entry per `ItemIdentifier` in the level with `itemType != ItemType.None` **and
`infiniteSource == false`** (M13), same search (inactive included) and template filter. `pos`, `held`, `placed`, `placed_in`,
`active` and `inactive_ancestors` are read **per step** (a carried skull moves, and `activateOnSuccess` flips
ancestors); `key`, `item` and `active_self` come from the scan. Sorted by `key` ascending.

| field | type | meaning |
|---|---|---|
| `key` | string | rule A4. Opaque, unique within a step |
| `pos` | `[x,y,z]` | live world position, every step |
| `item` | string | `itemType` enum name |
| `held` | bool | `ItemIdentifier.pickedUp` |
| `placed` | bool | `ipz != null \|\| GetComponentInParent<ItemPlaceZone>(true) != null` (A5) |
| `placed_in` | string or null | that zone's `altars[].key` when `placed`, else `null` |
| `active` | bool | `gameObject.activeInHierarchy` |
| `active_self` | bool | `gameObject.activeSelf` |
| `inactive_ancestors` | int | inactive GameObjects on its own chain, itself included — **with `active_self` this is the carryable test** (A6, M10) |

**`gates[].needs_item`** — new nullable string on each existing `gates[]` entry: the `item` of any **unfilled,
non-dead** altar (`inactive_ancestors <= 1`, A6/M14) whose `doors` (never `reverse_doors`) contains that gate's
`key`; `null` when there is none. With several such altars take the lowest `item` name, deterministically. A
dead altar can never be filled, so counting it would leave the lock set forever (M14).

Rules:

- **A1 `filled`** is computed exactly as `ItemPlaceZone.CheckItem` does (`decompiled/ItemPlaceZone.cs:160`):
  `GetComponentInChildren<ItemIdentifier>()` **without** `includeInactive`, then `itemType == acceptedItemType`.
  With `includeInactive: true` every `Altar (… Skull) Variant` would report as already filled, because it
  carries a disabled decoration skull. Consequences to accept: while a room is switched off, every altar in it
  reads `filled: false` and every item `active: false` (conservative and self-correcting — `needs_item` says
  "yes" before the room loads, which is the right routing answer); and **31 source pedestals campaign-wide flip
  to `filled: true` the moment their room activates, with no agent action** (M11), which is what §6.5's payment
  rule exists to ignore.
- **A2 one door-key table.** `Scan()` builds `Dictionary<Door, string> doorKeys` over **every** door in the
  level (not only gate candidates) from the existing `ClosedPosition(door)` and `CampaignPatches.Key`. Both
  gate phases and `altars[].doors[].key` read out of that table, so the strings **match by construction**,
  including for one-room doors. Measured (M12): zero gate/non-gate collisions on all 33 levels, so no existing
  gate key changes; a regression test pins 0-1's 11 keys.
- **A3 `#N` suffixing becomes deterministic, which is a fix and not a neutral change.** Keys are suffixed
  `#2`, `#3`, … in ascending `(x, y, z)` of the closed position, falling back to `GetInstanceID()` so the order
  is a strict total order even for two doors authored at the same point. Today `ScanGates` suffixes in
  `FindObjectsOfType` order (`CampaignObserver.cs:292-301`), which is not stable across rescans while
  `hopsByKey` is keyed on the suffixed string — so a respawn can hand a gate the other gate's hops. Measured,
  the only level with a gate-internal collision is 1-2 (`0,20,380` ×2), so that is the only existing behaviour
  this changes. Altar keys collide the same way and use the same rule: measured on 1-1, `-15,27,427`,
  `0,-7,381` and `81,-4,251` each carry two `ItemPlaceZone`s ~0.2 m apart. Log one warning per level load,
  exactly as the gates array does.
- **A4 item keys** are memoized by `GetInstanceID()` in a dictionary cleared on scene change; a new instance
  takes `Key(pos)` at the scan that first sees it, `#N` if a live instance already holds that string. This is a
  **convenience, not an invariant**: nothing in Python depends on a key surviving a respawn any more (§6.4 keys
  approach budgets on the item *type* and §6.5 keys the milestones on type and door set), because
  `CheckPoint.ResetRoom` destroys and re-instantiates rooms and key stability across that cannot be tested
  offline. The only requirement is that keys are unique within a step.
- **A5 `placed`** must not rely on `ipz` alone: `ItemPlaceZone.Start` sets it, and `Start` has not run while the
  room is off, so a pedestal skull would read `placed: false` at load.
- **A6 carryability is `active_self && inactive_ancestors <= 1`** (M10), replacing the withdrawn "exactly one
  `ItemIdentifier` has `activeSelf == true`" rule, which is false: 1-1 has three, one of which is a phantom
  under a permanently disabled `Altar` node, and 8-2 has thirteen. **The same chain count applies to altars**:
  a zone with `inactive_ancestors > 1` is a dead branch that can never activate or fill (20 of 104 campaign-wide,
  M14), so it is excluded from `needs_item` on the mod side and from `_subgoal`'s altar list on the Python side.
  The mod reports the fields and Python applies the tests, so the threshold can move without a mod rebuild.
- `MaxAltars = MaxItems = 64`, same guard and "nearest the player first" truncation as gates; both arrays are
  empty on 0-1 and on the 12 levels with no altars, so this is a wire-size guard only. 1-1's 7 altars + 5 items
  add ~1.2 KB to a 3.6 KB obs line; sent in full every step, stateless, same reasoning as gates.

### 6.2 Literal example — `Level 1-1`, first obs of a fresh load

Re-parsed from `campaign_scenes_level1-1.bundle` (positions to 2 dp as authored). `gates` is abbreviated to the
four entries that matter (the real array has 13 today plus the one `altar_only` gate §6.0 adds).

```json
"campaign": {
  "mission": 6, "difficulty": 3, "seconds": 0.0, "timer_running": false, "level_started": false,
  "level_over": false, "restarts": 0, "input_locked": false,
  "exit": {"pos": [81.0, -76.1, 91.0], "active": false},
  "gates_ordered": true, "gates_truncated": false,
  "gates": [
    {"key": "81,-6,188",  "pos": [81.0, -6.0, 188.5], "hops": 0, "open": false, "locked": false, "active": true, "controller_active": true,  "needs_item": null},
    {"key": "81,-6,240",  "pos": [81.0, -6.0, 239.5], "hops": 1, "open": false, "locked": false, "active": true, "controller_active": false, "altar_only": true, "needs_item": "SkullBlue"},
    {"key": "20,-10,381", "pos": [20.5, -9.5, 381.0], "hops": 2, "open": false, "locked": false, "active": true, "controller_active": false, "needs_item": "SkullRed"},
    {"key": "16,20,427",  "pos": [15.5, 20.5, 427.0], "hops": 3, "open": false, "locked": false, "active": true, "controller_active": true,  "needs_item": null}
  ],
  "altars": [
    {"key": "-15,27,427",   "pos": [-15.0, 26.74, 427.0], "item": "SkullBlue", "filled": false, "active": false, "inactive_ancestors": 1, "doors": [], "reverse_doors": [{"key": "16,20,427", "pos": [15.5, 20.5, 427.0]}]},
    {"key": "-15,27,427#2", "pos": [-15.0, 26.9,  427.0], "item": "SkullBlue", "filled": false, "active": false, "inactive_ancestors": 2, "doors": [], "reverse_doors": [{"key": "16,20,427", "pos": [15.5, 20.5, 427.0]}]},
    {"key": "0,-7,381",     "pos": [0.0,  -6.76, 381.0],  "item": "SkullRed",  "filled": false, "active": false, "inactive_ancestors": 1, "doors": [{"key": "20,-10,381", "pos": [20.5, -9.5, 381.0]}], "reverse_doors": []},
    {"key": "0,-7,381#2",   "pos": [0.0,  -6.6,  381.0],  "item": "SkullRed",  "filled": false, "active": false, "inactive_ancestors": 2, "doors": [{"key": "20,-10,381", "pos": [20.5, -9.5, 381.0]}], "reverse_doors": []},
    {"key": "81,-2,275",    "pos": [81.0, -2.1,  275.0],  "item": "SkullRed",  "filled": false, "active": false, "inactive_ancestors": 1, "doors": [], "reverse_doors": []},
    {"key": "81,-4,251",    "pos": [81.0, -3.76, 251.0],  "item": "SkullBlue", "filled": false, "active": false, "inactive_ancestors": 1, "doors": [{"key": "81,-6,240", "pos": [81.0, -6.0, 239.5]}], "reverse_doors": []},
    {"key": "81,-4,251#2",  "pos": [81.0, -3.6,  251.0],  "item": "SkullBlue", "filled": false, "active": false, "inactive_ancestors": 2, "doors": [{"key": "81,-6,240", "pos": [81.0, -6.0, 239.5]}], "reverse_doors": []}
  ],
  "items": [
    {"key": "-15,27,427",   "pos": [-15.0, 26.64, 427.0], "item": "SkullBlue", "held": false, "placed": true, "placed_in": "-15,27,427",   "active": false, "active_self": true,  "inactive_ancestors": 1},
    {"key": "-15,27,427#2", "pos": [-15.0, 26.8,  427.0], "item": "SkullBlue", "held": false, "placed": true, "placed_in": "-15,27,427#2", "active": false, "active_self": true,  "inactive_ancestors": 2},
    {"key": "0,-7,381",     "pos": [0.0,  -6.86, 381.0],  "item": "SkullRed",  "held": false, "placed": true, "placed_in": "0,-7,381",     "active": false, "active_self": false, "inactive_ancestors": 2},
    {"key": "81,-2,275",    "pos": [81.0, -2.2,  275.0],  "item": "SkullRed",  "held": false, "placed": true, "placed_in": "81,-2,275",    "active": false, "active_self": true,  "inactive_ancestors": 1},
    {"key": "81,-4,251",    "pos": [81.0, -3.86, 251.0],  "item": "SkullBlue", "held": false, "placed": true, "placed_in": "81,-4,251",    "active": false, "active_self": false, "inactive_ancestors": 2}
  ]
}
```

What the machine does with it, all from the parsed data:

- **Red leg.** Gate `20,-10,381` (hops 2) needs `SkullRed`; of its two wired altars `0,-7,381` is live and
  `0,-7,381#2` is a dead twin (M14), so only the first counts. Candidate sources of type `SkullRed`: the pedestal
  skull at `(81, -2.2, 275)` passes A6 (`active_self`, 1 inactive ancestor) and is not in a wired altar; the
  decoration at `(0, -6.86, 381)` fails A6. **Exactly one source, 133.5 m from the altar.**
- **Blue leg.** Gate `81,-6,240` (hops 1, the `altar_only` gate §6.0 adds) needs `SkullBlue`; its live wired
  altar is `81,-4,251`, `81,-4,251#2` being the dead twin. Candidates: the real source at `(-15, 26.64, 427)`
  passes A6; the phantom at `(-15, 26.8, 427)` fails on `inactive_ancestors: 2`; the decoration at
  `(81, -3.86, 251)` fails on `active_self`. **Exactly one source, 202.8 m from the altar.** Under the withdrawn
  `active_self`-only rule the phantom would also have qualified.
- **Both legs terminate.** Once the live altar reads `filled: true` its gate's `needs_item` goes `null` and the
  machine returns to the gate (state 3). Counting the dead twin instead would leave `needs_item` set after a
  successful placement and send the agent to punch the skull back out of the altar it just filled (M14).
- `16,20,427` (hops 3) is a **reverse** door of the blue-skull altars — placing a blue skull *closes* it — so it
  gets `needs_item: null`. `needs_item` comes from `doors` only. (The survey's Table 2 counted reverse doors;
  this contract deliberately does not.)
- **Every `ItemIdentifier` on 1-1 starts inside an `ItemPlaceZone`**, so a rule of "nearest item not held **and
  not placed**" finds zero sources and the sub-goal never leaves the altar. §6.4 uses "not in an altar we are
  trying to fill" instead.
- `active` is `false` everywhere at load because 1-1's rooms are off until the player reaches them (A1). The
  sub-goal deliberately filters on A6 rather than on `active`, so it can point at a source whose room has not
  streamed in yet — on 1-1 that is the only thing that gives the blue leg a heading before the castle is
  reached.

### 6.3 How a skull is physically picked up and placed

**The existing `punch` button does both. No action-space change, and therefore no weight-surgery migration.**
Movement is camera-relative, `AltHit` fires on the first environment-mask hit within 4 m of the camera along
camera forward (M6), and look mode 2 already turns the camera at `GateProgress.target` — so "face the sub-goal,
walk to it, press punch" is exactly what the existing action space expresses. Three things stop that from being
true by itself, and §6.7 and §6.8 fix them rather than hoping:

- the policy **chooses** the look mode, and historically volunteers a useful aim ~3% of the time;
- `_look_at_target` clamps to `band = cfg.pitch_limit_deg` (45° in the campaign configs), while an altar cube at
  floor level within ~1.5 m horizontally needs a steeper look than −45°, so the ray goes over it;
- a punch that does not place **throws** the carried skull, and a punch at a filled altar takes it back out and
  re-closes the door (M8).

`punch` is still charged `RewardConfig.punch` (0.01) per pressing decision, small against the new milestones.
`FistType` does not matter (M6); `change_fist` is not needed.

### 6.4 `GateProgress` sub-goal state machine

`_choose_target` is unchanged; a new wrapper runs **after** it and may substitute a sub-goal. All state is in
this step's `campaign` block, so there is nothing to keep in sync and nothing to reset.

```python
SUBGOAL_ITEM, SUBGOAL_ALTAR = "item", "altar"

def _live(entry, *, need_active_self=True) -> bool:
    """A6: a live source or a fillable altar, not a decoration and not a dead branch (M14).

    An old mod sends neither field, so both default to "usable" and the machine behaves as before.
    """
    if need_active_self and not entry.get("active_self", True):
        return False
    return int(entry.get("inactive_ancestors", 0) or 0) <= 1

def _subgoal(self, campaign, gate, pos):
    """The gate, or the item/altar rung below it. Pure: reads only this step's campaign block."""
    campaign = campaign or {}
    if not gate or gate.get("subgoal"):
        return gate                                     # already a sub-goal (a kept target from a player-less frame)
    need = gate.get("needs_item")
    if not need or pos is None:
        return gate                                     # no lock, an old mod, or no player
    altars = [a for a in (campaign.get("altars") or ()) if isinstance(a, dict) and a.get("pos")
              and a.get("item") == need and not a.get("filled")
              and _live(a, need_active_self=False)          # a dead twin never fills (M14)
              and any(d.get("key") == gate["key"] for d in (a.get("doors") or ()))]
    if not altars:
        return gate                                     # 3. every wired altar is filled: the lock is open
    items = [i for i in (campaign.get("items") or ()) if isinstance(i, dict) and i.get("pos")
             and i.get("item") == need]
    if any(i.get("held") for i in items):
        target, kind = self._nearest(altars, pos), SUBGOAL_ALTAR      # 2. carrying
    else:
        altar_keys = {a["key"] for a in altars}
        free = [i for i in items if not i.get("held") and _live(i)
                and i.get("placed_in") not in altar_keys]             # not already in the altar we want
        if not free:
            return gate                                 # no reachable source: fall back to the gate
        target, kind = self._nearest(free, pos), SUBGOAL_ITEM         # 1. fetch
    key = f"{kind}:{need}" if kind == SUBGOAL_ITEM else f"{kind}:{need}:{gate['key']}"
    return {"key": key, "pos": list(target["pos"]), "hops": gate.get("hops"),
            "open": False, "locked": False, "active": True, "subgoal": kind, "gate_key": gate["key"]}
```

`retarget` becomes `self.target = self._subgoal(campaign, self._choose_target(campaign, player_pos), player_pos)`.

| state | condition | target | why it cannot be farmed |
|---|---|---|---|
| 1 fetch | gate has `needs_item T`, nothing of type `T` held | nearest live free `T` item | `best_dist["item:T"]` is seeded once per episode and never re-seeded (unchanged rule) |
| 2 carry | an item of type `T` is held | nearest unfilled wired `T` altar | `best_dist["altar:T:<gate key>"]`, likewise |
| 3 open | every wired altar for the gate is `filled` | the gate | the gate's own `best_dist` entry, likewise |

- **The sub-goal key is keyed on the item *type* and the gate, never on the instance.** That is deliberate: the
  anti-farm argument is `if key not in self.best_dist` (`campaign.py:347-348`), which only holds while the key
  string is stable, and a checkpoint respawn re-instantiates skulls (`CheckPoint.ResetRoom`) so instance-key
  stability across a respawn cannot be established offline. With type-and-gate keys, dying next to the skull
  cannot re-seed the budget and re-earn the ~20-point approach. Accepted trade-off: two same-type sources share
  one approach budget within an episode, which is strictly anti-farm. If the nearest instance changes, `pos`
  moves under a stable key — `update` only pays when the new distance beats the running best, so a switch to a
  farther instance pays nothing and a switch to a nearer one pays the difference once.
- **Key namespacing.** Sub-goal keys start `item:` / `altar:`, so they can never collide with a gate key
  (`"x,y,z"`) or the `"exit"` sentinel. A target change is still detected by `_key_of` alone, so `update`'s
  "only an unchanged target pays approach" rule needs no edit.
- **`hops` is inherited from the gate**, so the sub-goal packs through `campaign_block`'s existing gate branch
  (`hops is not None` ⇒ scales 50/100, `open`/`locked` 0.0, mask 1.0) and **`spaces.py` is not edited**. Slot
  455 therefore reports the *gate's* hops while the target is a fetch or carry leg, which may be off-route
  entirely (1-1's red pedestal at z 275 sits past the hops-2 door it opens at z 381). Harmless, but slot 455 is
  not a route distance for a sub-goal.
- **`reset_episode` / `mark_paid` / `new_level_load` are unchanged.** The machine is stateless; the only
  per-episode state it touches is `best_dist`, through the existing rules, so a respawn (`mark_paid` +
  `retarget`) re-derives everything and re-earns nothing.
- **Shuttling is bounded:** within one episode the fetch key, the carry key and the gate key each pay their
  approach once, so a skull↔altar loop pays at most the one-way distance of each leg.
- **Look mode 2** aims at `target["pos"]` whatever the target is; §6.8 is what makes that reliable. **An old
  mod** sends no `needs_item`, `altars` or `items`, so `need` is `None` and the gate is returned unchanged —
  byte-identical to today.

### 6.5 New milestones, and why geometry does not pay

`MilestoneTracker` gains two sets, paid once per level load and absorbed by `mark_paid` exactly like
checkpoints, arenas and doors. Neither is keyed on an item instance:

- **`item_pickup`** is keyed on the **item type**: the set of `items[].item` values reading `held: true`,
  **restricted to types some altar in this level accepts** (`{a["item"] for a in altars}`). Type-keying kills
  two farms at once — a respawn that re-instantiates a skull under a new key cannot re-pay, and neither can the
  duplicate instances a level ships. The accepted-type filter is what stops 8-2's 22 `CustomKey1` props
  (+330 at 15.0) and 6-1's 11 (+165) from being a side quest with no route value; measured, `Readable`, `Torch`,
  `Soap` and `CustomKey1..3` props appear as unaccepted types on 15 levels.
- **`item_placed`** is keyed on the altar's `item` plus the sorted keys of the doors it drives
  (`"SkullRed|20,-10,381"`), falling back to the altar's own key when it drives no door. One puzzle therefore
  pays once even though coincident duplicate zones 0.2 m apart are wired to the same
  door (several levels ship such pairs; 1-1 has three) — otherwise a policy could punch the placed skull back out (`AltHit`'s `!holding && itemIdentifier !=
  null → ForceHold`) and place it in the twin for a second +15. **And a `filled: false → true` transition pays
  only when an item of that altar's type read `held: true` in the block the tracker saw on the previous step;
  otherwise the key is absorbed silently.** That is what keeps M11's 31 pre-filled source pedestals from paying
  +15 each for a room switching on: `mark_paid` only runs at a reset or a respawn, never when a room activates
  mid-episode, so absorption alone does not catch them.

`_keys` returns five sets, `update` a 5-tuple, and `CampaignStep` gains `item_pickups: int = 0` and
`item_placements: int = 0`. `env._campaign_progress`'s stuck-clock predicate (`env.py:799`) gains both counts
alongside `checkpoints or arenas or doors …`.

### 6.6 Reward arithmetic

`RewardConfig` gains `item_pickup: float = 0.0` and `item_placed: float = 0.0`, paid in `compute_reward` next to
`checkpoint`/`arena_clear`/`door_unlock`, i.e. **before** the missing-player early return, because the env has
already marked them paid.

At the tier of `gate` and `door_unlock` (15.0), at 15 decisions/s and the measured median 17.1 m/s, 1-1's two
legs price out like this:

| term | red leg (133.5 m) | blue leg (202.8 m) |
|---|---|---|
| `item_pickup` | +15 | +15 |
| `item_placed` | +15 | +15 |
| `gate_approach` out (0.15/m, once) | +20.0 | +30.4 |
| `gate_approach` back (once) | +20.0 | +30.4 |
| `time` for the detour (0.02 × 2·d/17.1 × 15) | −4.7 | −7.1 |
| **subtotal** | **≈ +65** | **≈ +84** |

Each unlocked gate then pays another +15 on the `gate` ladder. Without `needs_item` both gates are dead ends the
agent pays `gate_approach` to reach and can never pass — the failure mode this removes. **Not farmable:** both
terms are once-per-level-load keys with the §6.5 restrictions, and both approach legs are once-per-episode per
target key, so pick-up/throw/pick-up earns nothing after the first.

**The weights ship at 0.0** until §8's in-game checks 1-3 pass (M7 and M6 are both unverified in the live game,
and a skull that cannot physically be picked up turns S4 from a fix into a 134 m detour the agent still pays
~20 `gate_approach` to start).

### 6.7 Env-side carry protection

Two rules in `UltrakillEnv`, both **conditioned on the current target being a sub-goal**, so they are inert on
0-1, on any level with no `ItemPlaceZone`, and against an old mod. New config, campaign-only:

```python
subgoal_punch_range_m: float = 4.0   # 0 disables both rules below
```

1. **Punch gating.** On a step where any `items[]` entry reads `held: true`, `punch` is dropped from the
   command unless the current target is a `SUBGOAL_ALTAR` and the player is within `subgoal_punch_range_m` of
   it. Without this, at ~5 punch presses per second against a ~8 s carry, the skull is thrown almost
   immediately (M8), and a punch at an altar that is already filled un-places the skull and shuts the gate.
   The policy still chooses whether to press; the env only removes presses that can only do harm.
2. **Forced aim on the decisive step.** When the current target is a sub-goal and the player is within
   `subgoal_punch_range_m` of it, the env resolves look mode 2 regardless of the sampled look mode, and uses
   `MODE1_PITCH_LIMIT` (85°) instead of `cfg.pitch_limit_deg` for that step — the same exemption mode 1 already
   takes for overhead enemies (`env.py:477`). The 4 m raycast runs along camera forward from the camera's
   default position, so a floor-level cube a metre away is unreachable inside a 45° band.

Both are applied in `step()` where `_hold_slide` already edits the command, and the applied look mode is what
`_note_behaviour` records, as today.

### 6.8 Animation events under `render: false` (mod)

`ActiveStart` is an AnimationEvent and nothing else calls it (M7), so **S4 is inert if the player's fist
Animator is culled**. `TrainingSpeed.ApplyRendering` also walks `MonoSingleton<FistControl>.Instance` and
`MonoSingleton<CameraController>.Instance` with `GetComponentsInChildren<Animator>(true)`, sets
`cullingMode = AnimatorCullingMode.AlwaysAnimate`, and remembers the previous value so `RestoreRendering`
restores it exactly as `disabledCameras` is restored. Cheap, unconditional, and §8 check 1 is what proves it.

### 6.9 Observation, and degradation

**No observation change.** `ObsLayout(campaign=True).size` stays 479, every index keeps its meaning, no
checkpoint is invalidated; the sub-goal rides slots 448-455 through the existing gate branch. The policy
therefore cannot tell a skull from a door — deliberate, the same generalisation argument as "no level id". If
that proves binding, the width-preserving escape hatch is to repurpose slots 453 and 454 (`open` / `locked`,
already 0.0 for every non-gate target) as "target is an item" / "target is an altar", behind a new
`EnvConfig.target_kind_slots: bool = False`. **Default off**: turning it on changes what two learned input
columns mean and needs its own decision.

Degradation, both directions:

- **New Python, old mod** (no `altars`/`items`/`needs_item`/`altar_only`): `_subgoal` returns the gate on its
  `need` test, `_live` defaults both missing fields to "carryable" but is never reached, the two new reward
  terms are never paid (`MilestoneTracker` sees empty sets), §6.7 never fires because there is no sub-goal, and
  `bridge_test --campaign` prints `altars: -`. Test: `test_old_mod_without_altars_still_runs`.
- **New mod, old Python**: the new fields are ignored by `campaign_block`, `GateProgress` and
  `MilestoneTracker`, all of which read named keys; an `altar_only` gate is just another gate. Test: a FakeLevel
  sending the new fields against a `GateProgress` with the sub-goal disabled behaves as today.
- Mod: unknown `config` and `action` keys are already ignored; no new request type.

## 7. Config files

- `configs/campaign_1-1.yaml` (new): `level: "Level 1-1"`, `run_name: campaign_1-1`, everything else copied from
  `campaign_0-1.yaml`, plus **`item_pickup: 0.0`, `item_placed: 0.0`** and `subgoal_punch_range_m: 4.0`. Its
  header states that both weights go to **15.0 only after §8's in-game checks 1-3 pass**, and why.
- `configs/campaign_prelude.yaml` (new): `levels: ["Level 0-1", "Level 0-3", "Level 0-4"]` (Tier A Prelude per
  the survey's §7.1 — 0-2 needs skull carry and 0-5 is unordered, so both are held back), `unlock_rate: 0.5`,
  `run_name: campaign_prelude`, same rewards as `campaign_0-1.yaml`. Its header carries the run commands and the
  "copy `explore_*.npz` into the new model dir" warning.
- **`campaign_0-1.yaml` is not edited**: the live run keeps its config byte for byte.

## 8. Tests

Offline (`python tests/test_x.py`, no game; all must pass before any run starts):

| file | test |
|---|---|
| `test_campaign.py` | `choose_level` weights: unseen level 1.0, rate 1.0 → floor, all-mastered → uniform, `order[0]` always in the pool; **`unlock_next` on an empty stats dict returns `None` and does not raise**; `order[0]` unlocked by default lets `order[1]` unlock; the latch holds; below `unlock_window` refuses |
| | gates guard: 1.000 / 0.960 / 0.600 / 0.538 / **0.500 kept**; **0.250 and 0.031 dropped**; empty array dropped; **`campaign=None` returns `[]`** |
| | `_subgoal` walks fetch → carry → open; returns the gate with no `needs_item`, no altars, no items, a `campaign` of `None`, or no A6-live source; ignores an item already in the target altar but accepts one on a pedestal (the 1-1 shape); rejects the phantom (`inactive_ancestors: 2`) and the decoration (`active_self: false`); its keys are type-and-gate, so a re-keyed instance does not re-seed `best_dist` |
| | **the dead-twin case (M14): with one live altar filled and its `inactive_ancestors: 2` twin still unfilled, the machine returns the gate and never targets the skull it just placed** |
| | `MilestoneTracker`: pickup pays once per **type** and only for accepted types; placement pays once per item+door-set; **an altar that becomes filled while nothing of its type was ever held pays nothing** (the M11 pedestal case); a respawn's keys are absorbed |
| `test_campaign_env.py` | `FakeLevel` grows a **skull room**: an `altar_only` gate with `needs_item`, a pedestal altar holding the source, a wired destination altar, `punch` within 4 m picking up and placing |
| | walking the corridor with the skull pays `item_pickup` and `item_placed` exactly once; **punch is dropped while carrying outside 4 m of the altar and kept inside it**; look mode 2 is forced inside that range; throwing it (punch allowed, no altar hit) retargets and pays no second approach; dying while holding pays nothing twice |
| | `FakeLevel` grows **two levels**: a fresh start switches level, a checkpoint respawn never does, **and `_respawn`'s no-checkpoint level reload never does**; each level keeps its own archive across a switch and back; `best_runs/` writes one file per level; a switch episode's `episodes.jsonl` row carries the level it **ran on** |
| | missing / torn / wrong-`run_name` / wrong-`order` `curriculum.json` ⇒ `levels[0]` |
| | `test_old_mod_without_altars_still_runs`; `test_new_mod_fields_ignored_by_old_rules`; the 479-value observation is unchanged when the target is a sub-goal |
| `test_progress.py` | per-level table in `status.json`; `curriculum.json` written atomically and re-read; the unlock latch survives `_restore`; **a single-level run's `campaign` block is byte-identical to today**; **one unlocked level with `fresh_window ≥ 20` reproduces today's `fresh_completion_rate`**, and **unlocking a second level never lowers the score**; `episodes.jsonl` carries `level`; the dashboard renders a per-level block |
| `test_campaign_config.py` | both new configs build a 479-input env; every key is a real field; `fill_campaign_dirs` fills `curriculum_path` only when `levels` is set; **`UltrakillEnv(cfg).level == args.level` for a `levels` config with eval's `--level` applied**; a `levels` entry outside `CAMPAIGN_LEVELS_SHIPPED` raises |
| `test_spaces.py` | unchanged sizes and index ranges (a sub-goal packs through the gate branch) |
| `test_keep_best.py` | unchanged and still passing — `keep_best.py` is not edited |

**Mod:** a clean `dotnet build -c Release -p:InstallPlugin=false` is the mod-side test. The running games hold
the installed DLL, so the copy step must be off.

**In game** (one game on port 47800, nothing else connected). Checks 1-3 gate the 1-1 run and the two 15.0
weights:

1. **Animation events survive `render: false` (M7).** `Level 1-1`, rendering off, reach the red pedestal
   `(81.0, -2.2, 275.0)`, face it, press punch: `items[]` must show that entry `held: true`. Run it once with
   rendering on as the control; if it works rendered and still fails unrendered **with §6.8 applied**, S4 is
   inert under training settings and nothing else in the spec matters. **Getting there matters:** the pedestal's
   room (`3 - Skull Field`) starts switched off, and this project has already measured that a bare teleport into
   a switched-off room activates nothing (the 0-1 teleport probe). Drive there — spawn `(0, 105, 253)` is 136 m
   away and the Skull Field is adjacent to the spawn room — or activate the room first the way
   `campaign_check.py` does (1-1's checkpoints sit at `(46, 0.5, 388.5)`, `(81, -6, 231)`, `(46, 11.5, 550)`,
   `(3.5, 20.5, 427)`, all at the scene root). Reading `items[].active` before punching tells you whether the
   room is on.
2. **Placement works and survives punch spam (M6d, M8).** Carry that skull to `(0.0, -6.76, 381.0)`, punch:
   that altar must report `filled: true`, gate `20,-10,381` must report `needs_item: null` (the dead twin
   `0,-7,381#2` must not keep it set, M14), and both must still hold after 100 further steps of punch spam with
   §6.7's gating on.
3. **Held-skull-on-death.** Carry a skull, die, and read `items[]` on the next step — how many entries of that
   type exist and which is `held`. The one thing that cannot be settled offline.
4. `bridge_test.py --campaign` on `Level 1-1`: 7 altars with §6.2's keys, 5 items, `81,-6,240` present as a gate
   with `altar_only: true`, `hops: 1` and `needs_item: "SkullBlue"`, and `20,-10,381` with
   `needs_item: "SkullRed"`.
5. `campaign_check.py --level "Level 6-2"`: `exit.pos` is the `Intermission2` pit on three consecutive fresh
   loads (the bug is a `FindObjectsOfType`-order tie, so one load proves nothing).

## 9. File-by-file change list

No file appears in both halves.

### MOD WORK (C#, `mod/UltrakillAIBridge/`, plus the protocol doc)

| file | change |
|---|---|
| `Obs/CampaignObserver.cs` | S3: `"Level P-"` filter in `Scan`'s `FinalPit` loop, the `Intermission` test in `ChooseExit`, a warning on a rank tie. S4: `doorKeys` table over all doors in `Scan` (A2) with deterministic `#N` (A3); `ScanGates` phase 2 for altar-driven doors (`altar_only`, §6.0); `altars`/`items` scan lists plus `BuildAltars()` / `BuildItems()` including `inactive_ancestors`; `needs_item` per gate; `MaxAltars`/`MaxItems` 64 |
| `Env/TrainingSpeed.cs` | §6.8: force `AlwaysAnimate` on the player's fist and camera Animators while rendering is disabled, restore on `RestoreRendering` |
| `UltrakillAIBridge.csproj` | add `<InstallPlugin Condition="'$(InstallPlugin)' == ''">true</InstallPlugin>` and require `'$(InstallPlugin)' == 'true'` on the `CopyToPlugins` target, so `-p:InstallPlugin=false` compiles without touching the locked game folder. Default behaviour unchanged |
| `Plugin.cs` | mod version `0.6.0` → `0.7.0` |
| `docs/protocol.md` | `campaign.altars`, `campaign.items`, `gates[].needs_item`, `gates[].altar_only`; the three new `exit` rules; the shared door-key table and its ordering; a note that `filled`/`active` are `false` while a room is switched off |

### PYTHON WORK (`python/`)

| file | change |
|---|---|
| `ultrakill_ai/campaign.py` | S1 `choose_level`, `unlock_next`, `_level_stat`, `read_curriculum(path)`, `CAMPAIGN_LEVELS_SHIPPED`; S2 the `_gates` guard (both existing guards kept) and its `hops_min_frac` ctor arg; S4 `_subgoal`, `_live`, the `SUBGOAL_*` prefixes; `MilestoneTracker` gains the two item sets, the previous-step held types and a 5-tuple return |
| `ultrakill_ai/env.py` | `EnvConfig`: `levels`, `curriculum_path`, `unlock_rate`, `unlock_window`, `level_weight_floor`, `gate_hops_min_frac`, `subgoal_punch_range_m`, `target_kind_slots`. Mutable `self.level` (8 sites); `self._archives`; `_switch_level(previous, new)`; level sampling in `_campaign_reset` only; `info["level"]`; §6.7's punch gating and forced aim; the two new `CampaignStep` fields; the stuck-clock predicate |
| `ultrakill_ai/rewards.py` | `RewardConfig.item_pickup` / `.item_placed`; `CampaignStep.item_pickups` / `.item_placements`; both paid before the missing-player return |
| `ultrakill_ai/progress.py` | `per_level` stats (incl. `checkpoints_level`, `gates_reached`), restore from `status.json`, `campaign.levels` / `campaign.order`, §3.5's four aggregate rules (single-level path byte-identical), the `curriculum.json` writer, `level` in `EPISODE_LOG_RAW` |
| `scripts/train.py` | `fill_campaign_dirs` fills `curriculum_path` when `levels` is set; write the first `curriculum.json` **before** `model.learn()`; print the restored unlock set |
| `scripts/poll_status.py` | new column `levels_unlocked`; `part_item_pickup` / `part_item_placed` in `PART_FIELDS` |
| `scripts/dashboard.py` | per-level lines in the Campaign panel; the aggregate labelled a score; short level name in the per-game row |
| `scripts/eval.py` | §3.5's level resolution; clear `cfg.curriculum_path`; read `env.level` for times, reporting and the archive path |
| `scripts/bridge_test.py` | `--campaign` prints altars, items and each gate's `needs_item` / `altar_only` |
| `configs/campaign_1-1.yaml`, `configs/campaign_prelude.yaml` | new (§7) |
| `tests/test_campaign.py`, `test_campaign_env.py`, `test_progress.py`, `test_campaign_config.py`, `test_spaces.py` | §8 |

**Not edited, deliberately:** `scripts/keep_best.py` (§3.5), `ultrakill_ai/spaces.py` (§6.9),
`scripts/add_look_mode.py` and `scripts/transfer_weights.py` (no width change), `configs/campaign_0-1.yaml`
(§7), `ultrakill_ai/times.py`.

## 10. Risks

| # | Risk | Mitigation / status |
|---|---|---|
| R1 | Widening the door-key namespace to all doors (A2) could suffix a key that is bare today | Measured clean on all 33 levels (M12). Regression test pins 0-1's 11 keys. Keys are level-load scoped and never persisted |
| R2 | `filled` and `active` read `false` for a whole room until it switches on | Accepted and stated (A1); conservative for routing. Never read `filled: false` as evidence the altar is *reachable* |
| R3 | The policy's own punch destroys a carry, or un-places a placed skull | §6.7 gates the button while carrying; §8 check 2 proves it in game. The fallback lever is `RewardConfig.punch`, not a new button |
| R4 | The held-skull-on-death behaviour is unknown offline | §8 check 3 before trusting 1-1. Type-keyed milestones and approach budgets (§6.4, §6.5) mean a duplicate or re-keyed instance cannot pay twice either way |
| R5 | The policy cannot see that a target is a skull rather than a door | Deliberate (§6.9); escape hatch specified and defaulted off |
| R6 | One `max_steps` for every level truncates Act III ladders (§3.6), and the pooled `mean_100` numbers stop meaning anything across levels | Out of scope. Judge a multi-level run per level (`campaign.levels`, which now carries `checkpoints_level` and `gates_reached`), never on the pooled number |
| R7 | `fresh_completion_rate` changes meaning between a single-level and a multi-level run | The multi-level aggregate is a monotone shrunk **sum**, so an unlock can never lower it and `keep_best` keeps working unedited. Still **start a new `run_name`** when moving from a single-level config to a multi-level one, and copy the `explore_*.npz` files across as CLAUDE.md requires |
| R8 | The curriculum keeps sampling a level that is unwinnable for a structural reason | `levels` is hand-ordered, Tier A only for the first run. The floor caps waste at 10% of fresh **draws** per *mastered* level (realised episodes are weight × dwell, §3.3), but a *broken* level holds weight 1.0 forever — watch `campaign.levels[*].fresh_completion_rate` and `episodes`, and remove it from the config |
| R9 | Coincident duplicate `ItemPlaceZone`s (11 levels, 20 zones) | One of each pair is a **dead branch that can never fill** (M14), so A6 excludes it from `needs_item` and from `_subgoal`; measured, no door is driven only by dead zones. `CheckItem` opens the door from whichever zone is filled, and `item_placed`'s item+door-set key stops the twin paying a second time (§6.5) |
| R10 | Extra JSON per step across 5 games | Measured ~1.2 KB on 1-1 against a 3.6 KB line; both arrays are empty on 0-1 and on the 12 levels with no altars |
| R11 | S4's whole mechanism depends on Unity AnimationEvents under `render: false` | §6.8 forces `AlwaysAnimate`; §8 check 1 is the first thing run and gates everything else |

### Flagged — cannot work as proposed

- **The level survey's §8.4 target rule is wrong** and must not be implemented as written. "Nearest item that is
  not held **and not placed**" finds zero sources on 1-1, because every `ItemIdentifier` in that level starts
  inside an `ItemPlaceZone`. §6.4 replaces "not placed" with "not in an altar we are trying to fill", plus the
  A6 liveness filter.
- **`needs_item` cannot come from `reverse_doors`.** Those altars *close* the door — on 1-1 that is gate
  `16,20,427` (hops 3), and treating it as a lock would send the agent to fetch a skull that shuts the route.
- **A skull-locked door is usually not a gate, and widening only fixes three levels.** Measured (M3, M4): of 28
  distinct altar-driven forward doors, 6 are gates today and the phase-2 widening adds 3 more with a usable
  `hops` (1-1, 5-3, 8-1); the other 19 stay unreachable rungs. Everywhere else the door's room is outside the
  room graph, so `needs_item` exists but nothing carries it and the sub-goal never fires. Levels where a skull
  leg is therefore still invisible: 0-2, 1-3, 1-4, 4-2, 4-4, 5-1, 6-1, 7-1, 7-2, 7-3, 8-3, 8-4, plus 1-2's blue
  leg and 8-2's red leg. Those need the out-of-scope route fallback, not more of S4.
- **S4 does not unblock the levels with no usable door route** (0-5, 1-3, 1-4, 2-4, 4-2, 4-4, 5-2, 5-4, 6-2,
  7-1, 7-3, 7-4, 8-4, plus 7-2 and 8-3 once S2 drops them). Skull carry and route repair are independent; the
  checkpoint-chain fallback is explicitly out of scope.
- **`Level 9-1` and `Level 9-2` are not in this game build** (no scene bundle ships). They stay in
  `CAMPAIGN_LEVELS` but are excluded from `CAMPAIGN_LEVELS_SHIPPED`, which is what `env.levels` validates
  against; `eval.py --level "Level 9-1"` still fails at load. 33 of the 35 are trainable.

## 11. Review dispositions

Two independent reviews (`rl-contract`, `game`). Every disputed fact below was re-measured for this revision —
from `level_survey.json`, from the scene bundles directly (`m_IsActive` and ancestor chains), or from
`decompiled/` and the mod source. **Accepted** means the contract changed; where the fix
differs from the one the review proposed, the entry says which and why.

IDs: `rl-N` is the Nth blocking item of the `rl-contract` review, `g-N` the Nth of the `game` review,
`nb-N` a non-blocking item from either.

| # | Finding | Disposition |
|---|---|---|
| rl-1 | `item_placed` pays for pre-filled pedestals; §6.5 and A1 contradicted each other | **Accepted.** Re-measured: **31** zones campaign-wide flip to `filled` on room activation (1-1: 2, both reviewers' 1-1 number; my campaign-wide count is stricter than the review's 49 because it also requires the zone's own branch to be able to activate). §6.5 now pays a `filled` transition only when an item of that type read `held: true` on the previous step |
| rl-2 | `keep_best --metric campaign` freezes `best.zip` at the first unlock under a mean | **Accepted**, with `rl-contract`'s monotone shrunk sum; `fresh_window` stays the pooled deque length and `best_time` the global minimum, so `keep_best.py` still needs no edit (§3.5). `game`'s alternative (a `levels_unlocked` guard in `best.json`) was not needed once the aggregate cannot drop |
| rl-3 | `eval.py --level` ignored when the config has `levels`; `--record-times` writes the wrong row | **Accepted** verbatim, plus reading `env.level` in `report_campaign`, the times entry and the archive path (§3.5) |
| rl-4, g-6 | `unlock_next` KeyErrors on a sparse stats dict; `order[0]` may never unlock | **Accepted.** Both functions read through `_level_stat`, `curriculum.json` always lists every level, `per_level[order[0]]` is created unlocked at `_on_training_start` (§3.3) |
| rl-5, g-6 | §4's `_gates` and §6.4's `_subgoal` dropped the `campaign is None` guard | **Accepted.** Both guards restored; `gates_ordered` kept and its redundancy stated (§4); `_subgoal` does `campaign or {}` and returns an existing sub-goal unchanged |
| rl-6 | The anti-farm argument rests on mod-side item-key stability (A4), untestable offline | **Accepted.** `best_dist` keys are now `item:<type>` and `altar:<type>:<gate>`; A4 demoted to a convenience, and the milestones are type/door-set keyed too, so nothing depends on key stability across a respawn (§6.4, §6.5) |
| rl-7 | Nothing verifies the punch can pick up or place in the live game | **Accepted.** §8 in-game checks 1-3 added and made gates; `campaign_1-1.yaml` ships both weights at 0.0 (§6.6, §7) |
| g-1 | S4 cannot unlock 1-1: skull-locked doors are one-room doors, never gates | **Accepted with the smaller of the two proposed fixes.** Re-measured: 6 of 28 altar-driven forward doors are gates; the phase-2 widening (§6.0) gives 1-1's blue door **hops 1** and also lights up 5-3 and 8-1, while leaving the room graph, the BFS and every existing hop untouched. Everywhere else the widened door still has `hops: null` — stated in the Flagged section rather than papered over. 4-4's altar-gate carries `hops: null`, as the review said |
| g-2 | The milestones pay for geometry: pre-filled altars, and pickups of prop types | **Accepted.** Both narrow rules adopted (§6.5); the accepted-type filter is measured against 8-2's 22 `CustomKey1` and 6-1's 11 |
| g-3 | `punch` destroys a carry; a punch at a filled altar un-places it | **Accepted**, implemented as §6.7's env-side gating plus forced look mode 2, which is inert without a sub-goal. Cadence corrected in M8 (cooldown clears in 0.25 s, not 0.5 s) |
| g-4 | `ActiveStart` is an AnimationEvent; training runs with every camera disabled | **Accepted.** Confirmed: no C# caller for `ActiveStart`/`ActiveEnd` anywhere in `decompiled/`. §6.8 forces `AlwaysAnimate` on the fist Animators; §8 check 1 runs first |
| g-5 | A6's `active_self` rule is false and admits a phantom | **Accepted.** Re-parsed: three of 1-1's five items have `activeSelf == true`, and the `(-15, 26.8, 427)` one sits under a permanently disabled `Altar`. A6 is now `active_self && inactive_ancestors <= 1`, checked against every item on all 21 altar levels (M10). §6.2's example table was rebuilt from the parsed values |
| nb-1 | M8's "exactly one `activeSelf`" is wrong | **Accepted** — see G5. Both reviews were right; the spec's original claim is withdrawn |
| nb-2 | M2 misstates which levels are exit-ambiguous; 3-2 is not fixed; 2-4 unmentioned | **Accepted.** Re-measured: 2-4, 3-1, 3-2, 6-2 are the four; only 6-2 is fixed; 3-2 and 2-4 are benign-but-tied and now named (M2, §5) |
| nb-3 | §5 rule 2 cannot live inside `IsSuccessor` | **Accepted.** The test moved into `ChooseExit` with the exact code (§5) |
| nb-4 | Look mode 2's 45° pitch band can stop the punch connecting | **Accepted**, folded into §6.7 rule 2 (`MODE1_PITCH_LIMIT` within range) |
| nb-5 | Coincident duplicate altars let one puzzle pay `item_placed` twice | **Accepted.** `item_placed` is keyed on item + sorted door keys (§6.5) |
| nb-6 | `infiniteSource` items would mint a key per punch | **Accepted.** Verified no shipped level has one (M13); the mod skips them anyway |
| nb-7 | The stuck-clock predicate must gain the two new counts | **Accepted** (§6.5, §9) |
| nb-8 | `_switch_level` ordering hazard | **Accepted.** Signature is `_switch_level(previous_level, new_level)` (§3.4) |
| nb-9 | Cross-level `mean_100` / `best_*` numbers stop meaning anything | **Accepted.** `checkpoints_level` and `gates_reached` added to the per-level table; R6 rewritten |
| nb-10 | The curriculum weight controls draws, not episodes | **Accepted**, stated in §3.3 with `episodes` named as the number to read |
| nb-11 | `curriculum.json` is written after SB3's first `reset()`; a `run_name` change silently re-locks | **Accepted.** `train.py` writes it before `model.learn()` and prints the restored unlock set (§3.2) |
| nb-12 | `CAMPAIGN_LEVELS` still lists 9-1/9-2; S2's ratio is computed after truncation | **Accepted.** `CAMPAIGN_LEVELS_SHIPPED` added (§3.1); the truncation caveat stated with the measured maximum of 52 gates (8-1) (§4) |
| nb-13 | The sub-goal inherits `hops`, so slot 455 is not a route distance | **Accepted**, one sentence in §6.4 |
| nb-14 | The workers' `_rng` is unseeded, which is what keeps the draws independent | **Accepted**, noted in §3.3 |
| nb-15 | A2/A3's `#N` reordering is a behaviour change, and `(x,y,z)` is not a total order | **Accepted.** A3 now states it as a fix, names 1-2 as the only level it changes, and falls back to `GetInstanceID()` |
| nb-16 | The placement-mechanics facts (layer 22, trigger colliders, `queriesHitTriggers`, the guard position) | **Accepted**, folded into M6, plus one the reviews missed: `AltHit` returns early when the hit transform itself carries an `ItemIdentifier` while holding, so a placement punch must hit the zone's `Cube` |
| nb-17 | M3's "17 of 33 levels" is 21; M6's cadence is optimistic; §6.2's decimals are off; `cfg.level` is read in 8 places | **Accepted**, all four corrected (M3, M8, §6.2, §3.4) |
| nb-18 | 6-1 is kept by the 0.5 threshold on no evidence | **Accepted**, stated in §4 as the first knob if 6-1 wedges |
| nb-19 | `_respawn`'s no-checkpoint reload must not re-sample; `info["level"]` needs a test | **Accepted** (§3.4, §8) |
| nb-20 | 8-4's exit is picked by uniqueness, not by the successor test | **Accepted** as a risk rather than a code change: `ChooseExit` logs a warning on a rank tie (§5 rule 3), which is the general form of the request |
| **X1** | **Found during this revision, raised by neither review: the dead twin of a duplicate altar can never be filled, so `needs_item` would never clear and the machine would send the agent to punch the skull back out of the altar it had just filled, re-closing the gate** (M8's `ForceHold` → `CheckItem` → `doors[i].Close()`) | **Contract changed.** A6 now applies to altars; `needs_item` (mod) and `_subgoal`'s altar list (Python) both require `inactive_ancestors <= 1`. Measured: 20 of 104 zones are dead, on 11 levels, and no door is driven only by dead zones, so the filter never removes a real lock. Offline test added to §8 |
