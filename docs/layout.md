# Layout — file-by-file detail

Moved verbatim out of `CLAUDE.md` on 2026-09-18 (documentation restructure). This is the full
per-file map of the mod, the Python package, the scripts, the tests, the configs and the route data.
Entries are as they were written when each piece landed, so read a date-stamped claim as of that date
and check `docs/project-log.md` for anything later.

## Layout
- `times.md`: the AI's level-time leaderboard (best time per level plus per-generation history). No entries yet.
- `mod/UltrakillAIBridge/`: BepInEx 5 plugin (C#, netstandard2.1).
  - `Plugin.cs`: entry point and config (port 47800, panic key F8).
  - `Net/BridgeServer.cs`: TCP server, newline JSON.
  - `Env/EpisodeController.cs`: lockstep, resets, time settings, and the `unwedge` / `unwedge_frames` config keys.
  - `Act/ActionInjector.cs`: virtual Input System keyboard and mouse; camera look via `CameraController.rotationX/Y`.
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot. `ground_ray_center` (a single ray straight down from the player, outside the 8-ray ring and outside the array, so the packed size stays 479) and the raw movement flags `player.slow_mode` / `heavy_fall` / `crouching` (`crouching` is private, read with `AccessTools` and degrading to `false` with one warning).
  - `Obs/CampaignObserver.cs`: the obs `campaign` block in the 35 main levels (exit, checkpoints, NavMesh path to the exit, locked doors, arena enemies, milestone keys, rank thresholds); room templates are skipped by `CheckPoint.defaultRooms` ancestry. Since v0.6.0 also the **route**: `campaign.gates`, the door graph built from `Door.activatedRooms` and BFS'd from the exit's room, plus `gates_ordered` / `gates_truncated` (see the gates gotcha), and a `ChooseExit` that drops secret-level pits and prefers the mission successor, frozen per level load. Since v0.7.0 (branch `next-levels`) also the **skull carry**: `campaign.altars[]`, `campaign.items[]` and `gates[].needs_item`, one shared door-key table so an altar's door key string-matches a gate key by construction, a phase-2 gate pass that appends altar-driven one-room doors as `altar_only` gates, and a `ChooseExit` that drops `Level P-` Prime Sanctum pits, counts an `Intermission*` target as leading onward and warns once per level load on a rank tie. Branch `skull-fixes` adds `altars[].aim_pos` (the zone's own collider centre, which is what a placement punch has to hit) and replaces the dead-twin filter with the relative `IsDeadTwin` rule. See the branch entries under Status. **Since v0.7.1 the chosen exit is RE-RESOLVED on every `Scan` (`ResolveExit`)** instead of the reference being cached: `CheckPoint.Start` clones the exit's room and banishes the original +10,000 on x, so a cached reference followed the banished twin. The choice *rule* (`ChooseExitByRule`) still runs once per level load — its rank reads `activeInHierarchy` and every gate's hops are measured from the chosen pit's room — but what it freezes is the pit's `targetLevelName` plus the position it was picked at, and each scan re-attaches that verdict to the live pit nearest that anchor (the clone is 0 m away, the banished twin 10,000 m). Python's `ExitGuard` stays as the older stopgap.
  - `Env/SafetyPatches.cs`: blocks leaderboard submissions.
  - `Env/SteamPatches.cs` (v0.7.1): under `-aibridge-nosteam`, a Harmony prefix that skips Facepunch's `SteamClient.Init(uint, bool)`, so the instance never registers with Steam. Reported in the handshake as `steam_hidden` (whether the skip is in place, not merely asked for). One patch is enough because `Init` is the only thing that calls `SteamAPI.Init()` and every consumer gates itself on `SteamClient.IsValid`; the audit is in `docs/game-internals.md`. Applied inside its own try/catch in `Plugin.Awake`, so losing invisibility can never cost the bridge.
  - `Env/TimePatches.cs`: frame-based hitstop during lockstep.
  - `Env/BackgroundPatches.cs`: keeps the cursor free and audio muted while the AI has control.
  - `Env/TrainingSpeed.cs`: soft death (Harmony prefix on `NewMovement.GetHurt` heals instead of a lethal hit, counted in obs `player.soft_deaths`), camera disabling, and enemy Animators set to `AlwaysAnimate`. Since v0.7.0 `ForcePlayerAnimators` also forces `AlwaysAnimate` on every Animator under `FistControl` and `CameraController` while rendering is disabled, because `Punch.ActiveStart` / `ActiveEnd` are Unity **AnimationEvents** with no C# caller: a culled fist Animator means nothing can ever be picked up under `render: false`. Its cache is keyed on the two singleton instance ids **plus every direct child's**, and it re-walks when any forced Animator has become null, because `FistControl.ResetFists` replaces the arms mid-level (an arm pickup) without changing the child count. Also `UnwedgePatch`, a separate Harmony class (so a game update renaming the private method it binds cannot take soft death down with it): a postfix on the private `NewMovement.HandleSlideState` that breaks the absorbing airborne `slowMode` state. See the wedge gotcha.
  - `Env/InstancePatches.cs`: training instances (`-aibridge-port N`) open prefs read-only and skip prefs and save writes; save writes are also skipped whenever the AI has control.
  - `Env/CampaignPatches.cs`: campaign config held in memory while the AI has control (`difficulty` override; `unlock_all_gear` makes every weapon, variant and arm read as owned); arena clears (`ActivateNextWave.EndWaves`) and door unlocks (`Door.Unlock`) recorded as rounded-position keys, cleared on every scene load; and every `StatsManager.Restart` not made by the bridge blocked while the AI has control (the game restarts by itself when a dead player presses Fire1, which would respawn mid-step without Python seeing the death).
- `mod/GamePaths.props`: local game path (gitignored; copy from `.example`). Build copies the DLL into `<game>/BepInEx/plugins/UltrakillAIBridge/` — **unless `-p:InstallPlugin=false`**, which skips the copy so the mod can be compiled while the running games hold the installed DLL open. The property defaults to `true`, so the ordinary build is unchanged.
- `python/ultrakill_ai/`:
  - `protocol.py`: socket client; `kill()` kills the player (debug command for the in-game death check).
    **Two timeouts, not one:** `timeout` (120 s) covers step/get_obs/config and `reset_timeout` (600 s) covers
    `reset`, which blocks on a Unity scene load. They shared 120 s until 2026-09-17, which was also the mod's own
    `resetTimeoutSeconds` — see the timeout gotcha. `BridgeError` now has three subclasses, so a caller can tell
    the cases apart while every existing `except BridgeError` still catches all of them: `BridgeTimeout` (no
    reply in time), `BridgeClosed` (socket dropped, or used after it broke) and `BridgeSceneUnknown` (the mod
    cannot find the scene — almost always a game that is still booting). `RECOVERABLE` is the tuple of the three.
    A timed-out or dropped request sets `client.broken` and every later request raises until `connect()` runs
    again: the reply the game was still writing would otherwise be read as the answer to the *next* request and
    every observation after it would be one request stale.
  - `env.py`: `UltrakillEnv` / `EnvConfig` (`pitch_limit_deg` keeps the camera near level; `camera_height_m` 0.9 is where every ray starts, `_eye`; `max_steps_per_level` overrides `max_steps` per scene name and is empty by default). `EnvConfig.from_dict` ignores keys that are no longer fields, so an old `env_config.yaml` with retired settings still loads. Campaign mode (`mode: campaign`, 479 inputs):
    - **Bridge resilience (2026-09-17): a sick game ends its episode, never the run.** Every bridge call in
      `reset()` and `step()` is wrapped. On a `RECOVERABLE` error the env reconnects to its **own** port (the mod
      accepts a new client and drops the old one, so no other worker's game is touched), re-sends its config,
      reloads the level and returns a fresh observation with the episode **truncated** and `end_reason`
      `"bridge_reset"` — which reaches `episodes.jsonl` and `status.json`'s `end_reasons_100` like any other
      reason, with a per-worker `bridge_resets` count beside it. A recovered step pays **0**: there is no `cur`
      frame to grade, and paying nothing is what keeps the accounting honest. `_adopt_fresh_load` then re-baselines
      `MilestoneTracker`, `GateProgress` and `PathProgress` on the reload exactly as a fresh level load does, so
      nothing it reveals is paid twice and nothing already paid is lost; the exploration archive keeps its visit
      counts (they span episodes and carry the `1/sqrt(N)` decay) and only starts a new episode. `info` for the
      truncated step is built from the **last good frame**, so the episode's own counters still describe the
      episode that was lost. Knobs: `step_timeout_s`, `reset_timeout_s`, `bridge_retries` (3), `bridge_backoff_s`
      (5), `unknown_scene_wait_s` (300), `connect_retry_s` (180). `unknown scene` on a reset is **waited out**
      rather than raised, because it means the game is still booting. Retries are bounded: after them the error is
      raised for real. A recovery blocks the worker, and vec envs step in lockstep, so a worst case stalls every
      game for ~5.5 min — under `supervise.py`'s 600 s `--stale-seconds`, and far cheaper than the crash it replaces.
    - Reset: a fresh level load or a respawn at the current checkpoint. Fresh after a completion, with no checkpoint yet in this level load, after `stuck_repeats` (3) stuck episodes in a row at one checkpoint, otherwise with probability `fresh_start_prob` (0.2).
    - **Multi-level curriculum** (branch `next-levels`): with `levels` set, a **fresh load and only a fresh load**
      picks the level, sampled from the unlocked set with weight `max(level_weight_floor, 1 - that level's fresh
      completion rate)`. `self.level` is mutable from then on; `_switch_level` saves the outgoing archive under the
      **outgoing** level's path before moving (both names are arguments so an implementation cannot write the old
      counts under the new name), keeps one `ExplorationArchive` per level visited, and clears the level-scoped
      stuck streak. `levels` is validated against `CAMPAIGN_LEVELS_SHIPPED`, so a curriculum cannot stall a worker
      on 9-1/9-2, whose scene bundles this build does not ship. Empty `levels` is today's behaviour exactly: no
      curriculum file is opened and the level never changes.
    - **Carry protection** (`subgoal_punch_range_m`, default 4.0 = `Punch.ActiveFrame`'s own reach): inside punch
      range of a sub-goal the camera uses the game's own pitch clamp instead of the 45 degree band, and outside it
      `punch` is dropped while a **carry** is in progress or a filled altar is in reach. Only two presses survive —
      the one that places and the one that picks up — because `Punch.ActiveStart` **throws** a held item and
      punching a filled altar `ForceHold`s the skull back out, closing the door it opened. A dropped press is not
      charged `punch` either. A carry is a **held item some live unfilled altar accepts**, not merely
      `any(items[].held)`: `campaign.wanting_altars` is the one filter, shared with `GateProgress._subgoal`, so the
      two agree by construction. Both harms need a destination — an item nothing wants can be thrown and picked up
      again, and the not-holding `ForceHold` branch is covered by the filled-altar test on its own. The release
      stays the spec's (within range of the **sub-goal** altar), deliberately narrower than "any altar that accepts
      this", because the source pedestal is itself an unfilled zone that accepts the item and releasing beside it
      would let the first press after the pickup throw the skull straight back down.
    - Input-locked frames (landing, cutscenes) are stepped through with an empty action and never reach the policy (`max_locked_skip_s`).
    - A death pays `death`, respawns at the checkpoint (or reloads the level when there is none) and the episode continues. Doors, arenas and checkpoints a respawn itself changes pay nothing.
    - Ends, in this precedence: `level_complete` (terminated), `wedged`, `stuck`, `max_steps` (all truncated).
      `stuck` is `stuck_seconds` with no milestone, new cell, shorter exit path, **new gate hops value,
      `gate_approach`, kill or style**. `wedged` is its own reason, never folded into `stuck`, so the dashboard
      can tell "cannot move" from "moving but making no progress"; both feed the `stuck_repeats` streak, because
      a checkpoint whose respawn point is wedge-prone could otherwise never escalate to a fresh load.
    - Look modes (campaign only, the 12th action dimension): 0 free look, 1 the nearest **visible** enemy among
      the first 8 the observation carries, 2 the current `GateProgress` target. Resolved in `step` against the
      observation the policy acted on and popped out of the command, so the wire `action` message is unchanged.
      A mode that finds nothing to aim at behaves exactly as mode 0, and **all four look counters record the mode
      that APPLIED, not the one sampled** -- a fall-back step is a genuine look-head sample and is graded by
      `yaw_track`/`pitch_track`, while a really-aimed step is not.
    - Wedge detector: airborne, not sliding, horizontal speed < 1 m/s, moved < `wedge_creep_mps` (1.5 m/s = 0.10 m
      per decision at both speed settings), with the mod's `slow_mode` or `heavy_fall` set (an older mod falls back
      to the same predicate without the flags). `wedged_steps` credits the whole run retroactively once it reaches
      `wedge_seconds` (45 decisions), so ordinary airborne time never enters it; `wedge_seconds` 0 turns the episode
      end off while the counting continues on the 3.0 s default hold.
    - Novelty is keyed on the **ground under the player**, not the player: `_ground_point` prefers the mod's
      `ground_ray_center` (the ring's minimum can be a ledge 4 m away rather than the floor -- measured spread
      among rays that hit is p50 0.5 m but **p90 10.8 m**) and falls back to the ring minimum on a 0.5.x mod. It
      pays nothing at all when there is no ground within range (the mod writes `ground_ray_length` exactly, so
      "off the map" is unambiguous). So a fall pays nothing, jumping on the spot pays once instead of once per
      cell of height, and running forward still pays for new floor while airborne. See the void-farming entry
      under Status for why.
    - The exploration archive also saves on a **lifetime-step schedule** (`archive_save_steps` 20000 /
      `archive_save_seconds` 600, checked every 500 steps), not only every 20 episodes: campaign episodes are
      thousands of decisions and `SubprocVecEnv` workers never call `close()` on Ctrl+C, which is why the whole
      ground run saved nothing.
    - `slide_min_hold` (default 0 = off) holds slide for N decisions once pressed. An experiment, not enabled:
      it also holds slide while the policy presses jump, and a jump out of a grounded slide is one of the two
      ways into the wedge. A9 must be re-measured with it on before it is turned on.
    - `level_complete` is graded **before** death and scene change, because the frame that ends a level arrives as
      the scene unloads and can have no player: grading a death first would pay 0 and lose the completion.
    - Info: `CAMPAIGN_INFO_KEYS` (`completed`, `fresh_start`, `level_seconds`, `checkpoints_level`, `cells_new`,
      `oob_frac`, `exit_dist_min`, `gates_reached`, `wedged_steps`, `level_started`, `look_gate_frac`,
      `slide_forced_frac`, plus `kills`, `style`, `deaths`), and `rank` / `restarts` on a fresh-start completion.
      `info` also carries the non-numeric `gate_hops_best`, `start_checkpoint`, `end_pos`, `look_free_frac` and
      `look_enemy_frac`, which only `episodes.jsonl` reads. `oob_frac` is the share of steps with no ground
      beneath, i.e. falling or off the map.
    - `difficulty` and `unlock_all_gear` are sent on connect. With `explore_dir` the exploration archive is saved to `explore_<level>_<port>.npz` every 20 episodes and on close; with `best_runs_dir` the fastest fresh-start completion goes to `<level>.json` (positions every step, official time, rank).
  - `spaces.py`: obs packing and `MultiDiscrete` actions. Cyber Grind is 448 dims, with 5 zeros where the retired route waypoint was so its checkpoints load; `ObsLayout(campaign=True)` is 479, replacing those 5 with the 36-value `campaign_block` (exit, **route target**, nearest checkpoint neither activated nor current, first locked door, arena enemies / timer / input lock / level seconds, and 9 exploration-map values). Every index before it is unchanged. Slot 12 is `min(hops, 20)/20` — a **bound, not a behaviour change** (the deepest gate ladder in the campaign is 13 hops and the longest shipped room trunk 15 rungs, so it is a proven no-op on everything that exists), there so a regenerated file with 25 rungs cannot feed a learned input column a value above 1.0.
    - **Absolute 448-455 changed meaning** (block 5-12): they carried the NavMesh path hint and now carry the
      `GateProgress` target -- rel xyz, 3-D distance, mask, gate `open`, `locked`, `hops`/20. Gate scales 50/100,
      exit scales 100/200 (0-1's exit is ~195 m from spawn, which the gate scales would push near 4.0), all four
      relative values clipped to +-4.0. `campaign_block(obs, explore, target)` and `pack_observation(..., target=)`
      take the target from the env, because `GateProgress` lives there; `target=None` leaves all eight at 0.0.
      The layout stays 479 and every other index is byte-identical, which is why the resume needs weight surgery
      (`scripts/add_look_mode.py`) rather than a fresh start.
    - Actions: `ACTION_NVEC` is 11 dims / 42 logits (Cyber Grind, **unchanged**); `ACTION_NVEC_CAMPAIGN` appends
      the look mode as a 12th dim, 45 logits. `action_space(campaign=)`, `decode_action` infers the width from
      `len(a)`, and `noop_action(campaign=)` uses explicit indices (the old negative ones addressed the wrong
      slots at width 12). Look mode 1 is an auto-aim and is deliberately **not** available in Cyber Grind.
  - `rewards.py`: reward weights and computation; `aim_errors` gives the 3-D, yaw and pitch angles off an enemy, `horizon_elevation` the enemy elevation above the horizontal (diagnostics, convention-free). Campaign terms (`time` per decision, `checkpoint`, `arena_clear`, `door_unlock`, `gate`, `gate_approach`, `item_pickup`, `item_placed`, `novelty`, `path`, all 0 by default) are paid only when the env passes a `CampaignStep` to `compute_reward`, and are paid even on a step without a player because the env has already marked those milestones paid. `level_complete` (default 100) pays once on the rising edge in both modes, and is paid **before** the missing-player
    early return so a completion on a player-less frame still scores. `punch` is charged per decision that presses the
    punch button: the button is an independent coin flip with no cost and, outside a parry, no effect, so nothing ever
    taught the policy to stop pressing it. The human-route terms `route_point` and `stuck` are gone.
  - `campaign.py`: campaign helpers: `CAMPAIGN_LEVELS` (the 35 main scene names in mission order), `safe_name`, the game's rank maths (`grade`, `compute_rank`; P needs 12 with no restarts), `ExplorationArchive` (per-game visit counts over 4 m cells: novelty `1/sqrt(N+1)` on a cell's first entry per episode, the 9-value exploration map around the player, atomic `.npz` save/load; the archive itself is position-agnostic — `env._ground_point` is what decides that the position handed to `visit`/`features` is the ground under the player, not the player), `MilestoneTracker` (pays each checkpoint, arena clear and door unlock once per level load; `mark_paid` absorbs what a reset or respawn reveals), `PathProgress` (metres of new best complete NavMesh path to the exit; gains count once they exceed 1 m), `choose_fresh_start` (level reload or checkpoint respawn for the next episode) and `save_best_run` (keeps the fastest run per level as JSON, written atomically). Branch `next-levels` adds:
    - **The curriculum** (`CAMPAIGN_LEVELS_SHIPPED`, `level_weights`, `choose_level`, `unlock_next`,
      `read_curriculum`): the 33 levels whose scene bundle this build ships, the sampling weights, the unlock
      latch, and a reader that falls back to `levels[0]` on a missing, torn, wrong-run or wrong-order file.
      **Unlocking is one-way by construction.** `unlock_next` only ever *names* a level for `ProgressCallback`
      to latch open — no code path anywhere sets `unlocked` back to False — so a level whose rate later collapses
      keeps training, and `ProgressCallback._restore_levels` latches by **name**, not by position. That is what
      makes **inserting a level into `order` safe**: with the new level locked the chained walk stops there and
      never revisits the unlocked levels behind it, and `level_weights` reads each level's own flag, so an
      already-unlocked later level keeps its sampling weight. Done live at integration pause #2 (0-2 inserted
      between an unlocked 0-1 and an unlocked 0-3) and pinned by
      `test_a_level_inserted_before_an_unlocked_one_does_not_re_lock_it` and
      `test_inserting_a_level_before_an_unlocked_one_keeps_it_unlocked`.
      **The safety valve, `unlock_after_fresh_episodes`** (0 = off; 600 in the live config): a level also opens
      its successor once it has spent that many of its own **cumulative** fresh episodes, whatever its rate. The
      ladder is strictly chained, so without it one level the policy cannot crack blocks every level behind it
      for the rest of the run — and no metric distinguishes "needs 2M more steps" from "needs a mechanic nobody
      has built". Cumulative, because `fresh_window` saturates at 50 and cannot express "600 tries"; the counter
      is `fresh_episodes`, carried across restarts with `episodes` (not with the windows, which deliberately
      start empty), published in `curriculum.json` and `status.json`, and read as 0 in any table written before
      it existed.
      **Learning-progress weighting** (branch `curriculum-progress`, `curriculum_weighting: "progress"`, the
      new default in `configs/campaign_gates_full.yaml` and nowhere else): `level_weights` gained a `rule`
      argument. `"inverse_rate"` is unchanged and still the default everywhere else — weight
      `max(floor, 1 - fresh completion rate)`, unnormalised. `"progress"` returns shares that sum to 1 and is
      built in a strict order of priority: every unlocked level gets `_floor_share` (the retention floor,
      `level_weight_floor` per level but shrunk so the floors together never exceed `CURRICULUM_FLOOR_MASS` 0.5
      — at 10+ unlocked levels 0.10 each would eat the whole distribution and the rule would silently become
      uniform); a **blocked** level gets that and nothing more; the rest is split in proportion to each level's
      **learning progress** `|progress_fast - progress_slow|`; and `CURRICULUM_WEIGHT_CAP` 0.5 then clamps each
      share, the surplus going to the other levels that are still learning. The cap is the one soft rule: with
      nobody else able to use the surplus it yields rather than pushing mass onto a level that cannot. An
      unknown rule string is today's rule, and `UltrakillEnv.__init__` refuses one outright
      (`CURRICULUM_WEIGHTINGS`) so a typo cannot silently restore the starvation.
      **The progress score** (`progress_score`) is 0..1 per **fresh** episode: a completion is 1.0, else
      `gates_reached / gates_total` where `gates_total` is `gates_reached + gate_hops_best` ratcheted per level
      (the ladder's own length, so 0-1's 10 rungs and 0-2's 8 are comparable), else `checkpoints_level / 6` as
      the rough fallback. `ProgressCallback` keeps two EMAs of it per level (`PROGRESS_FAST_SPAN` 10,
      `PROGRESS_SLOW_SPAN` 50), a high-water mark of the slow one, `progress_samples` (`PROGRESS_MIN_SAMPLES`
      20 before the pair is trusted at all) and `dry_fresh_episodes`, the fresh episodes since the last
      completion. **All of them carry across a restart** — unlike the windows — because the supervisor bounces
      this trainer and dropping them would hand the rule back to `inverse_rate` every few hours.
      **`level_blocked`** is the damping: `dry_fresh_episodes >= CURRICULUM_BLOCKED_FRESH_EPISODES` (100) AND
      not progressing, where "not progressing" is `progress_best - progress_slow > PROGRESS_IMPROVEMENT` (0.02,
      the level has fallen below the best progress it ever held) **or** `|fast - slow| <= PROGRESS_FLAT` (0.005,
      a score that does not move at all — the hard wall the high-water test cannot see, since a constant slow
      average is its own record). A cold level is never blocked: None means no evidence.
      **The obvious design was rejected on the data.** `fast - slow <= 0.02` ("is it climbing right now") blocks
      only 65% of Level 0-3's 301 dry samples, because a 10-deep average of a swinging score crosses any small
      threshold about a third of the time; the high-water test blocks 93% of them. `|fast - slow|` alone fixes
      nothing at all: 0-3's was 0.095, as large as 0-1's 0.063.
    - **`altar_aim_point` and `dead_twin`** (branch `skull-fixes`): where to aim a punch at a zone (the mod's
      `aim_pos`, else 1 m below `pos`), and the relative dead-twin rule the mod now applies too, so both sides
      agree about which zone a gate is waiting on. See the branch entry under Status.
    - **Skull milestones** in `MilestoneTracker`: `item_pickup` keyed on the item **type** and only for types some
      altar in this level accepts, `item_placed` keyed on `altar_placement_key` (the item type plus the sorted
      keys of the doors that altar opens), and a `filled` false -> true edge pays only when the item now in the
      altar is the one carried on the previous step — so a pedestal that reads filled at load pays nothing.
    - **`GateProgress` sub-goals**: with a target gate carrying `needs_item T` the target becomes the nearest free
      item of type T, then the altar of type T wired to that gate, then the gate again. `best_dist` is keyed per
      target, so shuttling between them cannot be farmed. Dead twins (`dead_twin`, the relative rule) and decorations
      (`active_self: false`) are filtered out, or the machine sends the agent to punch the skull back out of the
      altar it just filled. That altar filter is `wanting_altars(campaign, types)` — live, unfilled, accepts one of
      these types — in one place because `env._protect_carry` must use the identical test to decide a held item is
      a carry worth silencing the punch button for. The **usability guard** drops a whole ladder when fewer than half its phase-1 gates
      carry a `hops` value (7-2 is 1 of 4, 8-3 1 of 32), falling back to the exit vector.
    - `GateProgress` walks the `hops` ladder of `campaign.gates`. `retarget()` picks a target and pays **nothing**
      (it runs before every observation is packed, including `reset()`'s, so slot 452 reads 1.0 on the first
      decision of every episode and look mode 2 is never dead on step 1); `update()` returns (new hop values
      crossed, metres of new best closeness). Reach is a **cylinder**, 8 m horizontal by 6 m vertical, doubled
      while the gate is `open`, so a player on the roof over a door has not "reached" it. `best_hops`/`paid_hops`/
      `reached` are **level-load** scoped like `MilestoneTracker`; `best_dist` is **episode** scoped (ruling R1)
      and is keyed **per gate**, seeded once and never re-seeded, which is what stops a multi-gate tier being
      flapped A->B->A for income. An episode's total approach is bounded by the gate-to-gate polyline either way
      (723-736 m on 0-1).
    - **Target patience, parking and the fallback** (branch `ladder-fix`, spec
      `2026-09-17-ladder-patience-and-exit-guard.md`). `hops` is a shortest-path **lower bound** in a room graph
      that multi-room doors over-connect, so on 0-3, 1-1, 1-2, 2-3, 4-3 and 8-1 the gate nearest the spawn already
      sits near `hops` 0 and the monotone rule locks onto a door that cannot be walked to. A target that goes
      `patience_steps` decisions without getting closer is **parked** for the level load; the target then becomes
      the nearest **unreached, unparked** active gate at any hop count (sticky within `fallback_hysteresis_m`),
      then the exit once none remain. Reaching a gate the fallback chose pays a `gate` instalment **once per new
      `hops` value**, so the forward legs of a non-monotone route earn something while touring the other doors on
      a rung already reached earns nothing (the `gate` income of a level load is bounded by the ladder depth under
      both rules). Six rules keep the park from becoming a second wedge, each of them a reproduced failure:
      - a park is only a **switch** for a LADDER pick (it needs some unreached unparked gate strictly nearer),
        and that is what leaves 0-1 alone; a **FALLBACK** pick parks unconditionally, because `_pick` chose it as
        the nearest candidate and the same filter can never find one nearer, so the test would make every
        fallback target permanent and move the wedge one door over;
      - `park_best` is the **closest the player has been at any park of that key**: kept across an un-park and
        only ever lowered, so wandering off and stalling somewhere farther cannot buy a cheap un-park. It is NOT
        tracked down every step, which would be beaten by one decision's travel and make the park permanent;
      - the un-park bar **doubles per park** of that key (2, 4, 8, 16 m), so a door that has proved unreachable
        twice needs a real change of situation;
      - the clock runs on its own baseline, **never on `gate_approach`**, and `mark_paid` restarts it: `best_dist`
        is episode scoped by R1, so a post-death re-walk pays nothing and a clock reading the reward would park
        the one door the route needs;
      - the arena / kill-or-style suspension is **bounded** at `2 * patience_steps` (600 decisions, inside
        `stuck_seconds` 45's 675), because `arena_enemies_alive` is scoped to the whole level, not to the target's
        arena;
      - a **fetch or carry leg, and any gate reporting `needs_item`, is never parked at all** — parking it
        deletes the altar sub-goal that `env._protect_carry` keys the punch-drop on, and the next punch throws
        the skull.

      Parks survive `reset_episode` and die with the level load. `patience_steps` 0 restores the pre-patience
      tracker byte for byte, which is how the 0-1 proof is written and the per-config opt-out.
      Config: `gate_target_patience_s` (20 s), `gate_unpark_m` (2), `gate_fallback_hysteresis_m` (10).
    - **...and it only runs where the ladder is COLLAPSED** (`detect_collapsed_ladder`, `patience_active`,
      config `gate_patience_mode`: `collapsed` (default) | `always` | `off`; branch `route-fallback`,
      2026-09-17). Unconditional patience regressed the one level whose ladder is right: live, **0-1's fresh
      completion went 0.55 (n=43) -> 0.30 (n=70)** while 0-3's fresh `gates_reached` went 1.0 -> 4.5. The
      detector is a **pure function of the gates block and the player's first position on a fresh load**:
      collapsed when the gate nearest the spawn carries `hops <= max_hops / 2`. Measured over all 18 levels with
      a usable gate ladder it flags exactly the six the spec names (0-3 2/6, 1-1 1/5, 1-2 2/4, 2-3 0/2, 4-3 1/2,
      8-1 0/4) and none of the twelve healthy ones (ten at max/max, 5-3 12/13, 8-2 5/8). Decided **once per
      level load**, before anything reads it, from the layer precedence *without* the route preference; a
      checkpoint respawn keeps it, because a respawn starts the player half-way through a level where the
      nearest gate says nothing. `None` (no ladder or no player yet) reads as OFF. Reported per episode and per
      level as `ladder_collapsed`.
    - **`prefer_route_when_collapsed`** (default **false**, the lead's ruling): on a level that has both a
      collapsed gate ladder and a shipped trunk — 0-3 and 4-3, the only two — `true` hands it the trunk
      instead. A level whose ladder is HEALTHY can never read its file whatever the flag says, because the
      verdict gates it. With the flag false those two levels behave exactly as they do today (gates + patience),
      which is what 0-3 is currently progressing on.
    - **`ExitGuard`**: freezes `campaign.exit.pos` per level load and rewrites the block in place, so the gate
      target, observation slots 0-4 and `exit_dist_min` all stop following a `FinalPit` that `CheckPoint.Start` /
      `ResetRoom` banished by `x + 10000f`. A later report a whole multiple of 10,000 *below* the frozen one is
      accepted (the banish only ever adds, so that is the live pit coming back); anything beyond
      `exit_max_shift_m` is ignored. `env._guard_exit` runs it on every campaign observation the env adopts.
    - **The route fallback — layer 2, the offline room trunk** (branch `route-fallback`, spec
      `2026-09-17-route-fallback-and-boss-levels-design.md`). `_read_route` (lru-cached parse) / `load_route`
      (a per-instance **deep copy**, because `_choose_target` hands a rung straight out as `self.target`) read
      `ultrakill_ai/routes/route_<scene>.json`, and `GateProgress(route=...)` consults it on the **three
      `return []` arms of `_gates()` and nowhere else** — so the `return gates` success path is byte for byte
      what it was and no level with a usable gate ladder can reach any of this. Precedence per level load:
      **gates ladder (18 levels) → room trunk (12) → exit vector (3)**. Five rules hold the two layers apart:
      `_is_room_ladder` is **object identity** against the loaded document (nothing in the data can switch a
      rooms-only rule on for a gate); `_hops_source` clears the ladder on a mid-load layer flip, because the
      two carry different hop scales; guard **I5** refuses a file whose `exit.pos` is more than
      `route_exit_tol_m` (5 m) from the pit the game reports **and clears `target` / `best_dist`**, so a stale
      file really falls back instead of holding its last rung while `gate_approach` pays toward it; `_seed_start`
      **absorbs** the rung the player loads near (within `route_seed_m`, 150 m) in all four of `mark_paid`'s
      places rather than only the two hop floors, so the trunk aims at the rung *below* from the first decision
      and can pay nothing for it; and the loader **normalises** `open`/`locked`/`active` and strips any static
      `needs_item` rather than trusting the file (`open: true` would double `_is_reached`'s cylinder to
      16 × 12 m and pin observation slot 453; a static `needs_item` is a permanent wedge). `route_source` is an
      **integer** — 0 exit vector / 1 gates / 2 rooms — because everything in `CAMPAIGN_INFO_KEYS` goes through
      `ProgressCallback._num`; the string rides `EPISODE_LOG_RAW` as `route_source_name`. `set_route` swaps the
      trunk when the curriculum switches level. `route=None` (Cyber Grind, `route_fallback: false`, and the 19
      levels with no file) is provably the tracker without this spec.
      **Fourteen files ship, not twelve** (2026-09-17): the twelve levels with no usable ladder, plus **0-3 and
      4-3**, whose ladder is usable but collapsed. Those two are the only levels with both, which made one
      documented-as-impossible hazard real and it is now handled: a frame the mod sends with **no `campaign`
      block** used to take `_gates()`'s first arm, hand out the trunk unvalidated, flip `_hops_source` and clear
      a ladder that was fine. The layer is therefore **latched per load** — once a load's ladder has come from
      the gates, a block-less frame returns `[]` rather than the trunk. The twelve trunk-only levels are
      untouched, because their source is "rooms" from their first frame.
      **The ground rule, a sixth rule added 2026-09-18** (`_on_ground` / `_is_route_rung`): a **room-trunk**
      rung is credited only when the player has ground under them — `player.grounded`, or the centre ground
      ray within `route_ground_m` (8.0 m). A gate is never subject to it, by **object identity** against the
      loaded document exactly as `_is_room_ladder` is, so all 18 gates levels and Cyber Grind are provably
      untouched (`test_ladder_replay.py` and `test_a_gate_ladder_is_never_subject_to_the_ground_rule`). The
      reading is held only for the duration of `update()`, which is why every **absorbing** call —
      `mark_paid`, `new_level_load`, `retarget`, all outside it — stays permissive: refusing to absorb what a
      respawn reveals would re-arm a rung, which is a farm. `ground=None` (every caller but the campaign env)
      is the tracker exactly as it was. See the transform-vs-ground gotcha for the measurement.
  - `progress.py`: `ProgressCallback`. **Chart history keeps `timesteps` strictly increasing**: `_restore` carries
  the whole old history over, but resuming from an *older* checkpoint rewinds `num_timesteps`, so the restored tail
  would sit ahead of the points that follow it and the dashboard would draw a line doubling back on itself (seen
  live on `campaign_ppo_ground`: 2,691,640 -> 2,654,560). `_add_history` drops the superseded tail, keeping the run
  that is continuing. `dashboard.py` applies the same rule when building each chart series, because it is a viewer
  and has to render a `status.json` written before this fix. It writes live training stats to
  `runs/<run_name>/status.json` (atomic, every 2 s; `state` running/finished/stopped). Campaign runs add a `campaign` block (completion rate and median official time over the last 50 fresh starts, best time over all of them), `best_checkpoints_level`, `best_gates_reached` (max) and `best_gate_hops` (**min** -- lower is better, so it has its own comparison); all of them survive a restart through `_restore`. `mean_fresh_100` carries `gates_reached`, `checkpoints_level`, `completed` and `wedged_steps` over **fresh starts only**, because a respawn episode inherits them from its level load -- that inheritance is what produced the false "peak by depth" reading of the pre-fix run.
  On a **multi-level** run (branch `next-levels`) it also writes `runs/<run_name>/curriculum.json` atomically
  (per-level fresh window, completion rate, best time, `unlocked` latch), which is the only shared state the
  `SubprocVecEnv` workers read at a fresh start; `status.json` gains `campaign.levels`, and
  `campaign.fresh_completion_rate` becomes a **shrunk sum over the unlocked levels, not a rate** — it can exceed
  1.0. That shape is forced by `keep_best.py`, which only replaces `best.zip` on a strict improvement and is
  deliberately not edited: a plain mean drops at every unlock (~0.52 -> ~0.26), which would freeze `best.zip` on a
  single-level policy forever. With one unlocked level at `fresh_window >= 20` it is exactly the rate it has
  always been, and a single-level run's block is byte-identical to before the curriculum existed.
  Branch `curriculum-progress` adds the learning-progress statistics to each row of both files —
  `progress_fast`, `progress_slow`, `progress_best`, `progress_samples`, `gates_total`, `fresh_completions`,
  `dry_fresh_episodes` — plus the two derived numbers `progress_score` (the fast average) and
  `learning_progress` (signed `fast - slow`) in `status.json` only, since the dashboard is their only reader.
  Every one of them is **additive**: `CURRICULUM_VERSION` stays 1 and an env built before they existed reads the
  same file and weights by rate exactly as it always did, which is what makes the switch safe mid-run.
  It also writes **`runs/<run_name>/episodes.jsonl`**, one JSON object per finished episode, appended and flushed
  per line (with the `level` the episode actually ran on): it is written by the callback rather than the env because five `SubprocVecEnv` workers would interleave
  appends to one file. `start_checkpoint`, `end_pos`, `end_reason` and `level_seconds` bypass the numeric `field()`
  helper, which would write `null` for exactly the two fields that say where an episode died. A write failure
  warns and never stops training.
- `python/scripts/`: `bridge_test.py` (`--drive`, `--campaign`), `random_agent.py` (`--mode campaign` prints completed / checkpoints_level / cells_new per episode), `train.py` (PPO / RecurrentPPO, `--num-envs` uses SubprocVecEnv; **`EntropyFloorCallback`** adapts `ent_coef` between updates to hold total policy entropy above `train.ent_floor` nats — below the floor it multiplies by 1.10 up to `train.ent_coef_max` (0.02), a nat above it decays by 0.95 back to the config's own `ent_coef` and never below, and `ent_floor: 0` never touches the coefficient at all; the live value is logged as `train/ent_coef_live` and reaches `status.json`'s `ppo` block, the dashboard's PPO panel and `poll_status.py`. It does **not** survive a resume: every restart begins at the base and re-adapts over the first few updates), `eval.py`, `games.py` (launch/tile/status/stop/**relaunch** training instances; `relaunch --port N` restarts ONE instance, found by the pid listening on that port, leaving the others running — `launch` cannot, it calls `stop_all()` first), `dashboard.py` (Tkinter live view of `status.json`).
- `python/ultrakill_ai/routes/route_Level_*.json`: **the committed room trunks, 12 files, 95 rungs, ~17 KB**
  (0-5, 1-4, 2-4, 4-2, 4-4, 5-2, 7-1, 7-2, 7-3, 7-4, 8-3, 8-4) — layer 2's whole data set. Each is a ladder of
  `{key, pos, hops, name, gated_by, open, locked, active}` rungs shaped exactly like a gate array, plus the
  `exit` it was built against (guard I5's staleness check) and the diagnostics `tour_ratio`,
  `checkpoints_within_60m`, `legs_witnessed`, `last_rung_to_exit_m` and `trunk_collapsed`. **The recovery path
  for a bad rung is this data, never a knob:** set a level's `"rungs": []` and it falls to the exit vector with
  no code change and no retrain. `gated_by` is written but ignored by every consumer until stage S3.
- `python/ultrakill_ai/routes/rung_overrides.json`: **hand-measured rung positions the generator applies
  itself**, so a regeneration after a game update keeps a measured fix instead of silently undoing it (a plain
  edit of the emitted JSON would not survive `build_routes.py`). Deliberately NOT named `route_*.json`, which
  is the glob every reader of the route files uses. One entry today, 0-3's `2 - Side Hallway - Floor 1`; see
  the transform-vs-ground gotcha for why. Three rules make it safe: it is applied **last**, after every guard
  and **before** every diagnostic, so `tour_ratio` and the rest describe what ships; it only **moves** a rung,
  never adds, removes or reorders one, so `hops` and the total order cannot break; and each entry carries
  `was`, the position the generator itself produced when the override was measured — a drift of more than 1 m
  means the rooms have moved and the override is **REFUSED** with a note that `--validate` turns into a
  failure. Going last means the moved point skipped R4 and I2, so both are re-checked on the moved rungs alone
  and either failing reverts that one override. `tests/test_route_files.py` fails if an entry is not in the
  file that ships, which is the detector for an override that stopped being applied.
- `python/scripts/build_routes.py`: the only thing that writes those files — regenerates them from the shipped
  scene bundles with the spec's pipeline in order **I6 → T → R4 → I2 → R1 → I3 → R2**, then
  `rung_overrides.json`. One self-contained file
  (it inlines the pure-Python UnityFS/SerializedFile readers, the MonoScript map, the room-trunk chain assembly
  and the voxel standability probe), so regenerating needs nothing but the repo and the game install; the game
  path comes from `mod/GamePaths.props`. Read-only with respect to the game, single-threaded, no port, ~2 min
  for all 33 levels, **safe to run beside a live run**. `--validate` is the acceptance report, `--dry-run`
  measures without writing, `--levels` narrows, `--analyse-gates-levels` produces the gates-level comparison
  under Gotchas. **Rerun it after every game update** (risk 3: I5 catches a moved `FinalPit` within 5 m but not
  rooms that moved while the pit did not).
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
- `python/scripts/campaign_check.py`: in-game campaign checks on one game (difficulty, arsenal, checkpoint trigger, death respawn, exit and official time). `python/tests/test_campaign_check.py` runs the same five checks against a fake level (no game needed).
- `python/scripts/watch_completion.py`: watches a **human** playthrough read-only (never takes control, never
  configures, never writes) and reports when `level_started`, `exit.active` and `level_over` change, so a real run
  can prove the completion signal works. This is what cleared the exit blocker on 0-1; reach for it whenever a
  question is about what real play does, since the AI cannot yet reach the end of a level.
- `python/scripts/walk_to_exit.py`: drives a level toward its exit with scripted movement (waypoints are the
  level's checkpoints in nearest-neighbour order, shooting what gets in the way) and reports whether the level
  reports complete. Kept for the diagnosis it produced rather than for routine use: it cannot leave 0-1's sealed
  starting room, and its teleport ancestor showed that teleporting never activates rooms at all. Both failures
  are documented in the pilot entry under Status. Its `go_to` mover is now also `skull_check.py`'s, and steps
  through input-locked frames (the opening drop, a cutscene, a respawn) with an empty action instead of counting
  them as a stall.
- `python/scripts/skull_check.py` (branch `next-levels`): the in-game skull-carry probe, section 8 checks 1-3 of
  the multi-level/skull spec — the three that gate raising `item_pickup` / `item_placed` off 0.0. On one game,
  taking control at the training settings with rendering **off**, it **walks** (never teleports; a bare teleport
  into a switched-off room activates nothing) to 1-1's red pedestal `(81.0, -2.2, 275.0)`, faces it inside the
  game's own 85 degree pitch clamp and punches; carries to `(0.0, -6.76, 381.0)` and punches; then takes the
  skull back out and dies holding it. Reports 1 pickup (`items[].held`, and `items[].active` **before** the punch,
  so "the room is still switched off" is never mistaken for "the punch does not work"), 2 placement
  (`altars[].filled`, gate `20,-10,381`'s `needs_item`, then 100 decisions of punch spam pressed **through
  `env.step`**, so `_protect_carry` is what is under test — every decision is inspected, not just the last, since
  an unprotected punch pulls the skull out and the next one puts it back), 3 what a held skull does across a
  death. `--target/--altar/--gate/--via/--level/--port/--render/--teleport-assist/--skip-kill`, plus, added
  2026-09-17 when it was first pointed at the game: `--from-checkpoint X Y Z` (teleport onto a checkpoint, let it
  activate, **then respawn there** — the respawn is what switches the pedestal's room on), `--approach` /
  `--altar-approach` (stage by teleport into an already-lit room, still walking the last metres) and
  `--camera-height` (the 0.9 m from `player.pos` up to where the punch ray actually starts; `eye()` applies it).
  PASS/FAIL/SKIP per check, a `summary:` line and exit 1 on any FAIL, like `campaign_check.py`, whose helpers it
  reuses. `python/tests/test_skull_check.py` runs all three against `FakeLevel`'s skull room, so the script is
  proven before a game is ever launched.
- `python/scripts/transfer_weights.py`: campaign starting weights from a Cyber Grind checkpoint. Widens the 448 inputs to 479 (first-layer columns 0-442 copied, 443-478 zero, so Cyber Grind inputs give the same hidden features), scales the action head by `--action-scale` (default 0.5) to raise entropy, and keeps a fresh final value layer and optimizer. It now also builds the destination with the **campaign** action space and zero-pads the three new logit rows, so the documented Cyber Grind -> campaign path still produces a model `train.py --resume` can load. Output: `models/campaign_ppo/transfer_init.zip` (committed).
- `python/scripts/add_look_mode.py`: migrates a **campaign** checkpoint across the look-mode action dimension and the re-used target slots, so the run continues instead of restarting. Appends three zero action rows (all modes equally likely), zeroes first-layer columns 448-455 in both hidden stacks and **folds the removed inputs' mean into the first-layer bias** -- the fold is required, not optional: it roughly halves the displacement. Carries `num_timesteps`, `_n_updates` and the full Adam state (zero rows for the new logits, zeroed moments for the changed columns, `step` preserved). `--stats <run_raw.jsonl>` recomputes MU and measures the displacement on real observations. Output: `models/campaign_gates/look_init.zip` (committed). **This is the only migration path**: the action-space change makes every earlier campaign checkpoint unloadable. Cyber Grind is untouched.
- `python/ultrakill_ai/times.py`: reads and updates `times.md` (`record` is pure, `record_file` edits in place). Level cells use the short form (`0-1`), times are `mm:ss.mmm`, the leaderboard is sorted by campaign order and only a faster time replaces a row.
- `python/tests/test_progress.py`: `ProgressCallback`, dashboard and `poll_status.py` tests against fake Cyber Grind and campaign envs (no game needed).
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_transfer.py`: weight transfer on small PPO models: hidden features unchanged on Cyber Grind inputs, logits scaled, value head kept fresh, the saved model loads with 479 inputs (no game needed).
- `python/tests/test_look_mode_transfer.py`: the `add_look_mode.py` contract on small PPO models: the action head widened 42 -> 45 with the last three rows zero and every earlier row bit-identical, the hidden stacks identical except columns 448-455 and the folded bias, a 12-dimension action space and 479 inputs on reload, `num_timesteps` preserved, the three new logits equal on random inputs, and **mean KL(old||new) <= 0.03** on a recorded observation batch with the mean-fold strictly below the zero-only variant (no game needed).
- `python/tests/test_campaign_config.py`: `configs/campaign_0-1.yaml` builds a 479-input env with its reward weights, every key is a real config field, and `train.fill_campaign_dirs` (no game needed).
- `python/tests/test_keep_best.py`: `keep_best.py` scoring for both metrics on synthetic `metrics_log.csv` rows, and old `best.json` files (no game needed; `python tests/test_keep_best.py`).
- `python/tests/test_games.py`: `games.py`'s instance-count guard — that `disk_logging_enabled` reads `Enabled` from `[Logging.Disk]` and not from the `[Logging.Console]` section above it, and that a missing file or key means BepInEx's default (on). This is what keeps `launch --count 8` from silently starting three copies whose plugin never loads. Also the launch readiness verdict and the two parsers behind it: `parse_listening` (port → owning pid, ignoring ESTABLISHED rows), `parse_working_sets` (thousands separators and `N/A`) and `startup_verdict`, which is what stops the false "exited during startup" (no game needed).
- `python/tests/test_bridge_recovery.py`: **one sick game must not kill a twelve-game run.** The protocol half
  pins the two timeouts apart, that a timed-out request poisons its connection, that a dropped socket reads as
  `BridgeClosed` and that `unknown scene` is its own error which leaves the connection usable. The env half
  injects a timeout, a closed socket and `unknown scene` into `FakeLevel` (`fail_next_steps` / `fail_next_resets`,
  queues of exceptions popped one per call) and asserts the episode dies instead of the worker: truncated with
  `end_reason` `"bridge_reset"`, reward 0, one reconnect, config re-sent; a booting game waited out; a dead one
  finally raising after bounded retries; a death respawn that loses the bridge ending the episode; and the four
  consistency tests that matter — the reload re-baselines the trackers, its milestones pay again and only once,
  the archive keeps its counts, and the truncated episode's `info` describes the episode that was lost
  (no game needed; `python tests/test_bridge_recovery.py`).
- `python/tests/test_freeze_recovery.py` also holds **`_fake_clock_ladder`**, which runs a real `reset()` on a
  fake clock with every blocking call charged its REAL bound (a wedged game costs a full reset timeout, a dead
  port costs a full connect window) and returns the elapsed seconds, whether the relaunch rung fired and how it
  ended. Reach for it before changing any recovery bound: the suite's older fakes fail *instantly*, so a ladder
  that overshot its budget by 15 s and never reached its last rung passed them cleanly for a day. Anything that
  claims a time bound should be asserted on measured elapsed, not on a sum of the knobs.
- **The recovery ladder's budget (`env.py`) is authoritative, not advisory.** `_begin_recovery` returns
  `(deadline, opened_here)` and **only the frame that opened a budget may close it** — `reset()` opens one at the
  top (so the first connect, the ladder and the reset tail all share it) and `step()` opens one in its catch.
  `_clamp(want, deadline)` cuts every blocking call to what is left, so the call in flight when the budget
  expires ends *with* it: `client.reset(timeout=…)`, `connect(retry_seconds=…)`. Rung 2 (relaunch this env's own
  game) gets a reserve rung 1 may not spend (`bridge_relaunch_reserve_s`), because otherwise three slow attempts
  always spent the budget first and the rung was unreachable in the one shape it exists for. At most
  `bridge_relaunch_slots` workers relaunch at once, through `O_EXCL` permit files in `env_log_dir`
  (`acquire_relaunch_slot`, stale-age takeover so a crash cannot wedge the rung shut), plus a per-port stagger.
  `EnvConfig.RUN_ONLY_FIELDS` keeps `bridge_relaunch` and `env_log_dir` OUT of `to_dict`, so the
  `env_config.yaml` train.py writes can never hand `eval.py` the power to kill a training game.
- `python/ultrakill_ai/envlog.py`: `EnvLog`, one size-bounded timestamped event log per worker at
  `runs/<run>/env_<port>.log` (reset start/end with level and duration, recovery start/end, reconnect failures,
  relaunches, boot gate, budget spent). Filled in by `train.py` alone (`env_log_dir`), so eval and the tests
  write nothing. It exists because the 2026-09-17 freeze could not be attributed: the trainer's stdout was being
  thrown away by the supervisor's detached spawn and nobody could tell, since the stale log tail looked fresh.
  Rolls to `<name>.1` at 2 MB and never raises — a log that failed would turn a recoverable fault into a dead
  worker. `supervise.py` quotes each file's tail into its restart report.
- `python/scripts/post_times.py`: posts a training run's best official level times to `times.md` from files the run already writes (no game, idempotent). `python/tests/test_post_times.py` covers first post, repeat post, only-faster and a missing `episodes.jsonl` (no game needed).
- `python/scripts/replay_curriculum.py` (branch `curriculum-progress`): replays a run's `episodes.jsonl` through
  **both** curriculum weighting rules and prints, per hour, the mean share of the fresh draws each unlocked level
  would have had under each — the evidence a weighting change has to produce before it is switched on. The
  statistics come from `ProgressCallback` itself (the same accumulator the trainer runs), so the script is only
  the replay loop and the table; `--drift` adds the signed-drift distribution per level, which is how the
  blocked test was chosen over the obvious one. **Read-only and safe against a live run**: it opens one
  `episodes.jsonl`, never a port, a checkpoint or the run's own state, and `test_replay_curriculum_scores_both_rules_from_an_episode_log`
  asserts the log is byte-identical afterwards.
- `python/scripts/supervise.py`: **the crash supervisor** — restarts the run when it dies, with no LLM and no
  tokens. Health is four things at once: a `train.py` process for this run exists (matched on the command line),
  `runs/<run>/status.json` moved within `--stale-seconds`, **its `timesteps` moved within the same window**, and
  its `state` is `running`. **Honestly about the step test:** it would NOT have fired first on 2026-09-17 —
  `ProgressCallback` writes `status.json` only from SB3's own callbacks, so a worker blocked inside `env.step()`
  stops the writes too and the mtime went stale by itself. It is there because mtime is a proxy for liveness
  while `timesteps` is the thing actually being asked about, and because the file is already written off the step
  loop in places (`_on_rollout_start` after a step-free PPO update, `mark_stopped` on the way out) — a heartbeat
  thread, the obvious next improvement, would blind the mtime test completely. Any change counts, up or down,
  because a restart resumes from a checkpoint and the count goes backwards. A draining trainer (`state` `stopped`/`finished`) is exempt: it is *supposed* to have stopped
  stepping. **`--stale-seconds` is 900 s, and the number is derived, not chosen**: the step that faults pays its
  own 120 s bound, then `EnvConfig.bridge_recovery_budget_s` 540 s caps everything after it (reconnects,
  backoffs, resets and the relaunch, every call clamped to what is left of the budget), plus one 60 s poll =
  **720 s**. It was 600 s, which was shorter than a single old reset timeout, so the supervisor killed every
  recovery that was about to fix one game locally and paid twelve games for it. The derivation is not taken on
  trust: `test_the_measured_ladder_fits_inside_the_supervisors_patience` runs the real ladder on a fake clock,
  each blocking call charged its real bound, and asserts the measured elapsed fits inside the window.
  A process that exists while the run stops progressing is HUNG and is killed by PID with its SubprocVecEnv
  workers; a process that is gone is DEAD. **Before any restart it writes an attribution block**: one line per
  bridge port with pid, working set, ESTABLISHED connection count and CPU cores over a 5 s window — all from
  `netstat` and CIM, never by connecting to a bridge port, which would drop that game's trainer — then the tail
  of every `runs/<run>/env_<port>.log`. A restart logs the last 30 lines of the train log, stops the games, relaunches `--count N
  --monitor M` through `games.launch()` (readiness from netstat — it never opens a TCP connection to a bridge
  port), resumes from whichever of `latest.zip` and the newest `ckpt_*_steps.zip` holds **more timesteps**
  (`latest.zip`'s count is read out of the `num_timesteps` field of the `data` member inside the zip, so it costs
  no torch import), and starts the trainer with the same `cmd /c ... >> log 2>&1` line a human would type —
  **without `DETACHED_PROCESS`**, which silently discards everything that line's program prints (`CREATE_NEW_
  PROCESS_GROUP | CREATE_NO_WINDOW` instead; see the freeze gotcha). **A boot health gate runs between the launch and the trainer** (`await_boot`): a listening port is not a
  booted game, so it waits until every copy's working set has been over `--boot-min-mb` (600 MB; a booted copy
  sits near 1 GB) for `--boot-polls` consecutive polls, and restarts a laggard **on its own port** with
  `games.relaunch_one`, leaving the other eleven running. A laggard that never opened a port cannot be addressed
  that way and is only waited out; if anything is still cold at the deadline the trainer starts anyway, because
  the env now waits out `unknown scene` instead of dying on it. `poll_status.py` and `keep_best.py` are restarted whenever they are missing, on every poll. Matching
  rejects any command line containing `supervise.py`, `Win32_Process`, `Get-CimInstance`, `tasklist` or `wmic`,
  and the supervisor's own process tree by PID: a command line that *mentions* the trainer is not the trainer,
  which is how an earlier report script counted a PowerShell query as a running run. **`runs/<run>/SUPERVISOR_PAUSE`
  turns it off** (see Commands). Budget: after `--max-restarts-per-hour` restarts in an hour it logs `GIVING UP`
  and exits non-zero, because a restart loop grinds the checkpoints.
- `python/tests/test_supervise.py`: the supervisor's decisions against injected processes, clock, status probe,
  killer and launcher — healthy leaves everything alone, dead restarts once with the right resume file and count,
  hung kills the workers first, the pause file blocks everything, a graceful stop mid-save is not killed, the
  budget exits non-zero, the grace period stops a second trainer, helpers are started only when missing, the
  resume choice in all three shapes (latest ahead, checkpoint ahead, latest unreadable), the self-match trap and
  the boot health gate — `BootGate`'s two-consecutive-polls rule, a game that falls back under the line losing
  its streak, a vanished game being forgotten, `laggard_ports` naming only laggards it can address, a single
  laggard being restarted on its own port while the others keep running, and a game that never boots not
  blocking training forever (no game, no real process; `python tests/test_supervise.py`).
- `python/scripts/campaign_driver.py` (branch `specialists`): **the per-level specialist driver — one policy per
  level, trained sequentially.** It REPLACES `supervise.py` while it runs (never run both: they would fight over
  the games and over which trainer should exist) and embeds a `supervise.Supervisor` per stage, so the health
  test, the hung/dead/draining split, the sick report, the boot gate, the resume-file rule and the restart budget
  are the supervisor's own code, not a second copy. `StageSupervisor` adds exactly one thing: `post_times.py
  --watch --push` to the helper set, beside `poll_status.py` and `keep_best.py --metric campaign`. The seam for
  that is `Supervisor.helper_specs()`, a method rather than a literal, which is the only change made to
  `supervise.py`.
  - **The stage rule** (`stage_verdict`, pure, numbers in `configs/specialists.yaml`). A stage ends when BOTH:
    the level's fresh completion rate over its last 50 fresh episodes reaches `target_rate` 0.5 over at least
    `min_fresh_window` 30 of them — **latched**, because a later dip must not deadlock a stage that peaked and
    then collapsed (this project has watched exactly that twice), and `keep_best.py` is holding the peak — AND
    `settle_steps` 300k have passed since the **later** of that moment and the last time `keep_best.py` moved
    `best.zip` (`best.json`'s `at_timesteps`). A run still setting new bests keeps resetting its own clock and
    keeps training. Without the "later of", a best saved long before the target would satisfy the settle the
    instant the target was reached. OR the stage has consumed `max_steps_per_stage` 6M, at which point it moves
    on **regardless** and is recorded `"unfinished"` so it can be revisited: one blocked level may not block 29.
  - **Per stage**: run `spec_<short level>` (`spec_0-1`), `models/spec_0-1/`, `runs/spec_0-1/`, the generated
    config at `configs/generated/spec_0-1.yaml` (written from the plan, with `timesteps` budgeted **from where
    the stage starts** — `timesteps` is the run total in `train.py` and every stage resumes from the last one).
    The shared run's `explore_<level>_*.npz` are copied in when the stage has none of its own. The init
    checkpoint is also seeded as the stage's `latest.zip`, so `choose_resume` can answer from the first tick —
    without it a crash inside the first 50k steps, before the first checkpoint rotation, meets `NO RESUME FILE`
    and stops the supervisor dead.
  - **On stage end**: kill the trainer, its workers and its three helpers (never another run's — the live shared
    trainer is a `train.py` on another config and matching is by run name AND config file), then promote
    `best.zip` — or, with no `best.zip`, the newest checkpoint by `choose_resume`'s rule, since a killed trainer
    leaves `latest.zip` stale — to `models/specialists/<level>.zip` with a JSON sidecar (rate, best time, steps,
    source checkpoint, difficulty). The next stage resumes from that file. **The games are NOT relaunched
    between stages**: twelve cold starts are four minutes and the env loads the new scene on its next reset.
  - State in `runs/specialists/driver_state.json` (current stage, its start step count, the history), so
    restarting the driver resumes the stage it was on. `runs/specialists/DRIVER_PAUSE` is its `SUPERVISOR_PAUSE`.
    `--start-at <level>` and `--init <checkpoint>` are first-launch only; after that the state file wins.
- `python/configs/specialists.yaml`: the plan the driver reads — the level order (the 30 levels of
  `campaign_gates_full.yaml`, mission order) and a per-stage template that is that config's `env:` and `train:`
  **minus the multi-level keys** (`levels`, `unlock_rate`, `unlock_window`, `unlock_after_fresh_episodes`,
  `level_weight_floor`, `curriculum_weighting`) and minus the per-stage ones (`run_name`, `timesteps`). Not a
  training config: do not pass it to `train.py`.
- `python/scripts/full_run.py`: **chains the specialists over one game** — walks the plan's order, loads each
  level's specialist, plays it from a fresh load, records the official time and prints the table plus the total.
  A level with no specialist is skipped and reported. Read-only like `eval.py`: it reads each specialist's
  exploration archive and never writes it, never writes a `best_runs` file and never touches a curriculum file.
  `--record-times` posts each completed level through `ultrakill_ai.times` under the generation
  `specialists@<date>`; the leaderboard row only moves when the time is faster, which is that helper's own rule.
  One game, one port, and **never a port a trainer is using** — the bridge drops its current client.
- `python/scripts/specialists_status.py`: the driver's state in one screen (stage, steps into it, fresh rate and
  window, best time, how much settle is left, the promoted specialists table, what is left). Read-only: it opens
  files, never a port, so it is safe beside the driver and beside a trainer.
- `python/tests/test_campaign_driver.py`: the stage rule in every shape (below target, too few fresh episodes,
  the settle, a new best restarting the settle, the latch, the cap, no status at all), the plan loader refusing
  an unshipped level and a misspelt knob, the generated config being single-level and budgeted from its start,
  promotion preferring `best.zip` and falling back to the newest checkpoint, state round-tripping — and the
  driver end to end against fakes: first tick prepares and starts a stage, a healthy stage is left alone, the
  rule promotes and starts the next level, an unfinished stage does not block the ladder, a driver restart
  resumes its stage without starting a second trainer, the pause file, a dry run, games launched only when a
  port is missing, and **another run's trainer never matched or killed** (no game, no real process, no real clock).
- `python/tests/test_full_run.py`: the chaining logic against an injected `play` (order, skips, the total, the
  table, the times.md posting) and `play_level` itself against `test_campaign_env.FakeLevel` — a completed level
  reports an official time, a truncated one reports none and invents nothing (no game needed).
- `python/tests/test_specialists_config.py`: pins `configs/specialists.yaml` equal to `campaign_gates_full.yaml`
  — the order is that config's levels list, every env setting that is not a curriculum key is identical, every
  reward weight is identical, the train section is identical bar `run_name`/`timesteps`, every key is a real
  field, and a generated stage config builds a 479-input single-level env with the same action space as the
  shared policy it is initialised from (no game needed).
- `python/tests/test_freeze_recovery.py`: **the 17:52 freeze, and every bound that now stops it** (no game, no
  socket, no process, no real clock). The protocol bounds one by one — `close()` honours `close_timeout` and not
  `self.timeout`, a timed-out release still drops the socket without raising, a broken client skips the release
  entirely, the handshake has its own bound, a reset leaves no sticky bound on the socket, `_LineReader` bounds
  wall clock rather than each `recv`, and the reset bound sits between the mod's own 120 s give-up and its 300 s
  client drop. Then the recovery: one total budget, shared by `reset()`'s outer retry so it is not paid twice;
  the relaunch rung restarting **this env's port and no other**; no relaunch when the budget could not boot a
  game; and `bridge_relaunch` off unless `train.py` turned it on. Then the attribution log, the bounded teardown
  (a wedged worker terminated, a healthy one clean, never a blocking `recv`), the supervisor (a frozen run with a
  perfectly fresh status file caught by its step count, a backwards resume counted as progress, a draining
  trainer exempt, the stale-window arithmetic pinned against the env's budget, the per-port sick report written
  before anything is killed, and `DETACHED_PROCESS` asserted absent from the spawn flags), and `games.py`
  repairing one laggard port instead of failing a whole launch (`python tests/test_freeze_recovery.py`).
- `python/tests/test_times.py`: `times.md` updates against the committed file's exact text: placeholders, records, deltas, level order (no game needed; `python tests/test_times.py`).
- `python/tests/test_campaign_env.py`: campaign episodes against `FakeLevel`, a fake corridor level standing in for the bridge: completion and best run, no official time for a completion after a checkpoint respawn, respawn and reload after a death, the stuck rule, input-lock skipping, the 479 observation, retired config keys, archive save and load, and the two novelty-measure tests that pin the void exploit shut (`test_falling_off_the_map_pays_no_novelty`, `test_novelty_pays_for_new_ground_not_for_height`). `FakeLevel` reports ground rays the way the mod does, so its floor is at y 1 and `falling` makes every ray miss (no game needed). Its `enable_skulls(fields=, altars=, item_type=)` plus `item_active` cover the shapes a carryable comes in: the wired puzzle, a 0.6.x mod, 0-4's altar-free `CustomKey1`, an item no zone accepts, and an item whose room is still switched off.
- `python/tests/test_skull_check.py`: `skull_check.py`'s three checks against `FakeLevel`'s skull room — the whole carry green, an item whose room is off named as the reason, a placement undone by the spam caught as a FAIL (run with the carry protection disabled, which is what makes it the regression test for `_protect_carry`), a pre-0.7.0 mod, `--skip-kill`, the `--render` control run and the default coordinates (no game needed).
- `python/tests/test_campaign.py`: campaign helpers: level list, rank maths, exploration archive, milestones, path progress, the fresh-start rule and best runs, plus target patience / parking / the fallback payment and the `ExitGuard` arithmetic (no game needed). The patience section pins each anti-wedge rule against the failure it was written for: a **fallback** target parks in its turn and hands over (and the exit takes over when nothing is left), a second door on a rung already reached pays nothing and an eight-door tour still pays one instalment, a far-off re-park cannot buy a cheap un-park and the bar doubles per park, a death respawn does not park the door the agent is walking at, an arena that never dies still parks the gate at the `2 * patience_steps` bound, a carry leg and any `needs_item` gate are never parked, and a fallback gate reporting `hops: null` or no `hops` key returns 0 instead of raising.
- `python/tests/test_ladder_replay.py`: the A6/A7 proof obligations of the ladder-patience spec, replayed
  against recorded game data in `python/tests/fixtures/` (gzipped probe logs, ~190 KB in total, plus
  `ladder_golden.json`, which the PRE-patience `GateProgress` wrote). A6: patience off reproduces the golden
  file step for step over all 32,022 recorded `Level 0-1` decisions; a monotone ladder walked inside the window
  is untouched by patience; no gate is ever parked while an arena holds it; **6 of the 7 recorded episodes are
  byte-identical and the seventh is asserted in full** -- per-episode `gate_approach` deltas
  (`{5: +10.363, rest: 0.0}`), identical instalments, identical `reached` / `best_hops`, an identical target
  ORDER in every episode, and episode 5's run lengths to the decision, so the one park's whole effect is a
  78-decision boundary shift. A7: the 0-3 lock is broken within a few patience windows and held for at most half
  as many decisions, every park of the high door records a strictly closer baseline (the park log is pinned
  exactly), the target moves onto the walkable route, the forward legs pay `gate` (7 against 3) and the high door
  is un-parked once the agent actually climbs to it (no game needed).
  **Since the level-conditional revision** the A6 tests that measure what unconditional patience DID to 0-1 run
  with `patience_mode="always"` and are the record of a retired configuration; the guarantee that ships is
  `test_a6_0b` — under the SHIPPED settings all seven recorded 0-1 episodes reproduce the golden file exactly
  with **zero parks**, so the +10.363 m no longer happens at all. A7 (0-3) runs at the shipped default
  unchanged, which is the proof that the default keeps the level the mechanism exists for.
- `python/tests/test_route_files.py`: the **data**, judged against the route spec's section 7.1 and importing
  nothing that produced it — every guard is re-derived from the spec's wording and recomputed from the shipped
  `pos` list, so a bug in `build_routes.py`'s own helpers cannot hide behind it. Schema and version, I1
  (`hops` n-1..0, strictly descending, unique), I2 (16 m separation), I3 (≤ 24 rungs), R1 (≥ 3), **R2 with the
  stored `tour_ratio` reproduced from `pos` to 3 decimals** (the assertion that caught revision 1's
  1.264-vs-1.421 gap), I6 and T pinned by name, `key` == `round(pos)`, the `Pit` exception (hops-0 only, under
  70 m from the pit), and **`test_shipped_set_is_exactly_the_fourteen`**, which makes a regeneration that gains or
  loses a level a test failure rather than a silent coverage change, plus
  `test_the_collapsed_levels_that_ship_are_the_two_the_lead_approved` (0-3 and 4-3 only) (no game needed).
- `python/tests/test_route_replay.py`: **A0 of the route spec's section 8**, the safety property, and the
  regression test for invariant T. Replays the real `GateProgress` over each shipped trunk **in every visit
  order the level allows** and from every rung the player could start at, asserting the target never advances
  past an unvisited rung and no step pays more than one instalment; a synthetic **STAR** fixture reproduces
  section 2.6's four-instalments-in-one-step so the assertions are demonstrably not vacuous. Its other half
  protects the live run: `GateProgress` **with a route document in hand** reproduces `test_ladder_replay.py`'s
  golden file over all 32,022 recorded 0-1 decisions and both 0-3 probes, and `route_reads == 0` proves the
  trunk was never even consulted on a gates level (no game needed).
- `python/tests/test_route_walk.py`: the **liveness** property S1 and S2 owe each other, and the wrap-up
  stage's own check. Loads all 14 committed files through the packaged path `load_route(scene)` exactly as
  `UltrakillEnv` does, then walks each trunk with a **continuous** synthetic trajectory (4 m per decision,
  through every rung and on to the pit): every rung below the one the player starts at must become the target
  in turn, each pays **exactly one** instalment, and after the hops-0 rung the target is handed to the exit and
  never comes back. A continuous walk rather than a teleport per rung because the two failures it is here to
  catch are invisible to a teleport — overlapping reach cylinders marking two rungs in one decision, and a
  target that flaps or parks mid-leg. Four controls keep it honest: the trunk walked **backwards** pays 0, a
  **second lap** in the same level load pays 0, the **live patience setting** (300 decisions) parks nothing and
  changes no target, and `GateProgress(route=None)` over the identical walk pays 0 and reports layer 3. The
  measured `gate_approach` reproduces the spec's section 6 table to within a few percent from the tracker
  rather than from the polyline (no game needed). **It also owns the ground rule** (2026-09-18): the same
  walk **flown** at the wall-face reading (9.5 m) or over the void (30.0 m) pays nothing and advances the
  ladder by nothing on all 14 trunks, while a grounded walk and a legal 7.9 m hop are **byte-identical** to
  before the rule existed — instalments, approach metres and the whole target sequence. Plus the bound pinned
  at exactly 8.0 (7.9 in, 8.1 out), `grounded` beating the ray, a mod reporting neither being permissive,
  `route_ground_m: 0` switching it off, the **gate** ladder flown over the same trunks paying in full, and
  a respawn still absorbing a rung from the air (the anti-farm half).
- `python/tests/test_campaign_rewards.py`: campaign reward terms, finishing within the cap beating a timeout, the `level_complete` edge and the retired route terms (no game needed).
- `python/tests/test_spaces.py`: layout sizes (448 / 479) and every index range of the campaign block, including the yaw-frame signs (no game needed).
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml` (campaign 0-1: Violent, all gear unlocked in memory, run `campaign_gates`; its header lists the run commands). `campaign_prelude.yaml` (the multi-level curriculum 0-1 / 0-3 / 0-4 under a new run name `campaign_prelude`, kept as the reference) and `campaign_1-1.yaml` (the first skull-carry level, run `campaign_1-1`, **`item_pickup` and `item_placed` pinned at 0.0** until the three in-game checks pass). `campaign_gates_prelude.yaml` carried the live run from 4.8M to 6.77M steps (curriculum 0-1 / 0-3 / 0-4, `num_envs` 8, `ent_coef` 0.004) and is kept as history and as the partial rollback. **`campaign_gates_main.yaml` is the live one** (integration pause #2, 2026-09-17): the same `run_name: campaign_gates`, so the 6.77M-step weights, `best.zip` and the exploration archives carry on, with the 11-level Tier A + Tier B ladder, `unlock_after_fresh_episodes: 600`, `max_steps` 12000, `item_pickup` 10.0 / `item_placed` 20.0, `num_envs` 12 and `ent_coef` still 0.004. **`campaign_gates_full.yaml` is the live one since 2026-09-17 16:51** (branch `route-fallback`, merged): `campaign_gates_main.yaml` with **five** named changes and nothing else, each pinned by `test_the_full_config_is_the_main_config_with_more_levels` — the `levels` list 11 → 30, `gate_patience_mode: collapsed`, `prefer_route_when_collapsed: false` (both written out although they are the defaults, so flipping one is a deliberate act) and `ent_floor: 5.0` + `ent_coef_max: 0.02` (the adaptive entropy floor; `ent_coef` itself stays 0.004), plus **`curriculum_weighting: progress`** (2026-09-18, branch `curriculum-progress`) — the only config that opts into learning-progress weighting, which changes which level a fresh load draws and nothing else. Same `run_name: campaign_gates`, same weights, same policy — 18 of its levels are on layer 1, 12 on the offline room trunk, and 1-3 / 5-4 / 6-2 are left out because they have no route signal of any kind. Its header carries the three things to watch. Each config's header lists its own run commands, and **`tests/test_campaign_config.py` pins every setting and every reward weight that is NOT a named change equal to the config before it**, so nothing can drift while the same policy continues.
- `docs/level-survey.md`: all 35 levels parsed offline from the scene files — exits, checkpoints, door graphs and gate ladders, altars and carryables, arenas, bosses and hazards, plus a tier table and the recommended order of sub-projects. This is what the multi-level and skull-gates design was built from; check it before assuming anything about a level nobody has trained on. **Section 9 is the route-coverage appendix** (stage S1, 2026-09-17): which layer fires on each of the 33 levels, and per shipped trunk its rungs, gaps, tour ratio, legs witnessed and last-rung-to-pit distance.
- `docs/protocol.md`: the socket protocol.
- `docs/game-internals.md`: game classes and fields the mod relies on (check after game updates).
- `docs/superpowers/specs/`: approved design specs. `2026-09-16-campaign-foundation-design.md` is the campaign design; `2026-09-16-campaign-gates-unwedge-design.md` is the route-gates / un-wedge / look-modes design that followed the 0-1 pilot's diagnosis (implemented in mod v0.6.0 and the `campaign_gates` run; its section 14 records the lead rulings R1-R4); `2026-09-17-multi-level-and-skull-gates-design.md` is the multi-level curriculum / gates guard / 6-2 exit / skull-carry design (implemented on branch `next-levels` and mod v0.7.0; its section 8 lists the five in-game checks, section 10 the risks and section 11 the review dispositions); `2026-09-17-ladder-patience-and-exit-guard.md` is the collapsed-ladder / banished-exit design (implemented on branch `ladder-fix`, Python only; its section 3 records the five deviations from the brief that measurement forced and why the reach test was left alone, section 6 the proof obligations and the tests that discharge them, and **section 7 the disposition of the eight adversarial-review findings** — six bugs fixed, one proposed cure rejected with a reproduction of its own failure, and one claim about 0-1 rebutted by measurement). **`2026-09-17-route-fallback-and-boss-levels-design.md`** is the route-fallback / boss-levels design, revision 2, 1115 lines (branch `route-fallback`; stages S1 and S2 built, S3 and S4 not): three layers with a strict precedence rule, the offline room trunk that routes the 12 levels the gate ladder cannot, and the boss block that is deliberately off the critical path. Read section 2.6 before touching the ordering (a total order cannot express a branch, which is why only the **trunk** ships), 4.3 for the guard pipeline, 4.5 for the skull locks four levels stop at until S3, section 8 for the acceptance checks A0-A5, **11.5 for the four-stage build plan**, 12 for the nine risks and 13.2 for the five things already tried and rejected — a straight-line distance-to-exit reward, a static `needs_item`, renumbering so the lowest rung is hops 1, a per-level `gate_approach` scale and refusing every level that crosses an altar door.
- `docs/superpowers/plans/`: implementation plans. `2026-09-16-campaign-foundation.md` is the 16-task plan for the campaign foundation and the 0-1 pilot (Tasks 0-13 dry-run in a scratch copy: mod builds clean, all tests pass). `2026-09-17-next-levels-integration.md` is the ordered checklist for merging branch `next-levels` into main at a training pause and verifying it in the live game (merge, build and install, the six in-game readouts, the 20k-step smoke run, how to carry the live 0-1 policy into the multi-level run, and rollback).
- `.tools/` (gitignored): local `ilspycmd` install used to regenerate `decompiled/`.

