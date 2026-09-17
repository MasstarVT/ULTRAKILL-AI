# ULTRAKILL AI

Reinforcement-learning agent for ULTRAKILL (Cyber Grind + campaign). Repo: github.com/MasstarVT/ULTRAKILL-AI.

## Workflow rules
- **Always push to GitHub** after completing a change: commit, then `git push origin main` (remote: https://github.com/MasstarVT/ULTRAKILL-AI).
- **Always update this CLAUDE.md** as part of every change so it reflects the current state of the project (structure, setup, commands, conventions, decisions).
- Never commit game assemblies or decompiled game code (`.gitignore` covers `*.dll`, `decompiled/`).
- Checkpoints under `python/models/` are committed (since 2026-09-15, so a run can move between machines). Each is ~12 MB; commit `latest.zip` after a training session, and prune old `ckpt_*` files before committing if a run produces many.
- Update `times.md` whenever a training generation finishes (instructions are in an HTML comment at the bottom of that file). For campaign levels `python scripts/eval.py <model> --level "Level 0-1" --record-times` does it: the fastest completion goes into the generation history, and onto the leaderboard when it is a record.
- **Post times while training too** (user instruction, 2026-09-17): `python scripts/post_times.py --run campaign_gates` reads the run's `best_runs/<level>.json` (fastest fresh-start completion, official time and rank) plus `episodes.jsonl` for the step count, and posts a level only when it beats the row `times.md` already holds, so it is safe to run on every monitoring check and needs no game. After it posts, commit `times.md` and push. Rows it writes are marked `training episode (sampled actions), fresh start`; an `eval.py --record-times` row is the deterministic counterpart.

## Layout
- `times.md`: the AI's level-time leaderboard (best time per level plus per-generation history). No entries yet.
- `mod/UltrakillAIBridge/`: BepInEx 5 plugin (C#, netstandard2.1).
  - `Plugin.cs`: entry point and config (port 47800, panic key F8).
  - `Net/BridgeServer.cs`: TCP server, newline JSON.
  - `Env/EpisodeController.cs`: lockstep, resets, time settings, and the `unwedge` / `unwedge_frames` config keys.
  - `Act/ActionInjector.cs`: virtual Input System keyboard and mouse; camera look via `CameraController.rotationX/Y`.
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot. `ground_ray_center` (a single ray straight down from the player, outside the 8-ray ring and outside the array, so the packed size stays 479) and the raw movement flags `player.slow_mode` / `heavy_fall` / `crouching` (`crouching` is private, read with `AccessTools` and degrading to `false` with one warning).
  - `Obs/CampaignObserver.cs`: the obs `campaign` block in the 35 main levels (exit, checkpoints, NavMesh path to the exit, locked doors, arena enemies, milestone keys, rank thresholds); room templates are skipped by `CheckPoint.defaultRooms` ancestry. Since v0.6.0 also the **route**: `campaign.gates`, the door graph built from `Door.activatedRooms` and BFS'd from the exit's room, plus `gates_ordered` / `gates_truncated` (see the gates gotcha), and a `ChooseExit` that drops secret-level pits and prefers the mission successor, frozen per level load. Since v0.7.0 (branch `next-levels`) also the **skull carry**: `campaign.altars[]`, `campaign.items[]` and `gates[].needs_item`, one shared door-key table so an altar's door key string-matches a gate key by construction, a phase-2 gate pass that appends altar-driven one-room doors as `altar_only` gates, and a `ChooseExit` that drops `Level P-` Prime Sanctum pits, counts an `Intermission*` target as leading onward and warns once per level load on a rank tie. Branch `skull-fixes` adds `altars[].aim_pos` (the zone's own collider centre, which is what a placement punch has to hit) and replaces the dead-twin filter with the relative `IsDeadTwin` rule. See the branch entries under Status.
  - `Env/SafetyPatches.cs`: blocks leaderboard submissions.
  - `Env/TimePatches.cs`: frame-based hitstop during lockstep.
  - `Env/BackgroundPatches.cs`: keeps the cursor free and audio muted while the AI has control.
  - `Env/TrainingSpeed.cs`: soft death (Harmony prefix on `NewMovement.GetHurt` heals instead of a lethal hit, counted in obs `player.soft_deaths`), camera disabling, and enemy Animators set to `AlwaysAnimate`. Since v0.7.0 `ForcePlayerAnimators` also forces `AlwaysAnimate` on every Animator under `FistControl` and `CameraController` while rendering is disabled, because `Punch.ActiveStart` / `ActiveEnd` are Unity **AnimationEvents** with no C# caller: a culled fist Animator means nothing can ever be picked up under `render: false`. Its cache is keyed on the two singleton instance ids **plus every direct child's**, and it re-walks when any forced Animator has become null, because `FistControl.ResetFists` replaces the arms mid-level (an arm pickup) without changing the child count. Also `UnwedgePatch`, a separate Harmony class (so a game update renaming the private method it binds cannot take soft death down with it): a postfix on the private `NewMovement.HandleSlideState` that breaks the absorbing airborne `slowMode` state. See the wedge gotcha.
  - `Env/InstancePatches.cs`: training instances (`-aibridge-port N`) open prefs read-only and skip prefs and save writes; save writes are also skipped whenever the AI has control.
  - `Env/CampaignPatches.cs`: campaign config held in memory while the AI has control (`difficulty` override; `unlock_all_gear` makes every weapon, variant and arm read as owned); arena clears (`ActivateNextWave.EndWaves`) and door unlocks (`Door.Unlock`) recorded as rounded-position keys, cleared on every scene load; and every `StatsManager.Restart` not made by the bridge blocked while the AI has control (the game restarts by itself when a dead player presses Fire1, which would respawn mid-step without Python seeing the death).
- `mod/GamePaths.props`: local game path (gitignored; copy from `.example`). Build copies the DLL into `<game>/BepInEx/plugins/UltrakillAIBridge/` — **unless `-p:InstallPlugin=false`**, which skips the copy so the mod can be compiled while the running games hold the installed DLL open. The property defaults to `true`, so the ordinary build is unchanged.
- `python/ultrakill_ai/`:
  - `protocol.py`: socket client; `kill()` kills the player (debug command for the in-game death check).
  - `env.py`: `UltrakillEnv` / `EnvConfig` (`pitch_limit_deg` keeps the camera near level; `camera_height_m` 0.9 is where every ray starts, `_eye`; `max_steps_per_level` overrides `max_steps` per scene name and is empty by default). `EnvConfig.from_dict` ignores keys that are no longer fields, so an old `env_config.yaml` with retired settings still loads. Campaign mode (`mode: campaign`, 479 inputs):
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
  - `spaces.py`: obs packing and `MultiDiscrete` actions. Cyber Grind is 448 dims, with 5 zeros where the retired route waypoint was so its checkpoints load; `ObsLayout(campaign=True)` is 479, replacing those 5 with the 36-value `campaign_block` (exit, **route target**, nearest checkpoint neither activated nor current, first locked door, arena enemies / timer / input lock / level seconds, and 9 exploration-map values). Every index before it is unchanged.
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
  It also writes **`runs/<run_name>/episodes.jsonl`**, one JSON object per finished episode, appended and flushed
  per line (with the `level` the episode actually ran on): it is written by the callback rather than the env because five `SubprocVecEnv` workers would interleave
  appends to one file. `start_checkpoint`, `end_pos`, `end_reason` and `level_seconds` bypass the numeric `field()`
  helper, which would write `null` for exactly the two fields that say where an episode died. A write failure
  warns and never stops training.
- `python/scripts/`: `bridge_test.py` (`--drive`, `--campaign`), `random_agent.py` (`--mode campaign` prints completed / checkpoints_level / cells_new per episode), `train.py` (PPO / RecurrentPPO, `--num-envs` uses SubprocVecEnv), `eval.py`, `games.py` (launch/tile/status/stop training instances), `dashboard.py` (Tkinter live view of `status.json`).
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
- `python/tests/test_games.py`: `games.py`'s instance-count guard — that `disk_logging_enabled` reads `Enabled` from `[Logging.Disk]` and not from the `[Logging.Console]` section above it, and that a missing file or key means BepInEx's default (on). This is what keeps `launch --count 8` from silently starting three copies whose plugin never loads (no game needed).
- `python/scripts/post_times.py`: posts a training run's best official level times to `times.md` from files the run already writes (no game, idempotent). `python/tests/test_post_times.py` covers first post, repeat post, only-faster and a missing `episodes.jsonl` (no game needed).
- `python/tests/test_times.py`: `times.md` updates against the committed file's exact text: placeholders, records, deltas, level order (no game needed; `python tests/test_times.py`).
- `python/tests/test_campaign_env.py`: campaign episodes against `FakeLevel`, a fake corridor level standing in for the bridge: completion and best run, no official time for a completion after a checkpoint respawn, respawn and reload after a death, the stuck rule, input-lock skipping, the 479 observation, retired config keys, archive save and load, and the two novelty-measure tests that pin the void exploit shut (`test_falling_off_the_map_pays_no_novelty`, `test_novelty_pays_for_new_ground_not_for_height`). `FakeLevel` reports ground rays the way the mod does, so its floor is at y 1 and `falling` makes every ray miss (no game needed). Its `enable_skulls(fields=, altars=, item_type=)` plus `item_active` cover the shapes a carryable comes in: the wired puzzle, a 0.6.x mod, 0-4's altar-free `CustomKey1`, an item no zone accepts, and an item whose room is still switched off.
- `python/tests/test_skull_check.py`: `skull_check.py`'s three checks against `FakeLevel`'s skull room — the whole carry green, an item whose room is off named as the reason, a placement undone by the spam caught as a FAIL (run with the carry protection disabled, which is what makes it the regression test for `_protect_carry`), a pre-0.7.0 mod, `--skip-kill`, the `--render` control run and the default coordinates (no game needed).
- `python/tests/test_campaign.py`: campaign helpers: level list, rank maths, exploration archive, milestones, path progress, the fresh-start rule and best runs (no game needed).
- `python/tests/test_campaign_rewards.py`: campaign reward terms, finishing within the cap beating a timeout, the `level_complete` edge and the retired route terms (no game needed).
- `python/tests/test_spaces.py`: layout sizes (448 / 479) and every index range of the campaign block, including the yaw-frame signs (no game needed).
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml` (campaign 0-1: Violent, all gear unlocked in memory, run `campaign_gates`; its header lists the run commands). `campaign_prelude.yaml` (the multi-level curriculum 0-1 / 0-3 / 0-4 under a new run name `campaign_prelude`, kept as the reference) and `campaign_1-1.yaml` (the first skull-carry level, run `campaign_1-1`, **`item_pickup` and `item_placed` pinned at 0.0** until the three in-game checks pass). `campaign_gates_prelude.yaml` carried the live run from 4.8M to 6.77M steps (curriculum 0-1 / 0-3 / 0-4, `num_envs` 8, `ent_coef` 0.004) and is kept as history and as the partial rollback. **`campaign_gates_main.yaml` is the live one** (integration pause #2, 2026-09-17): the same `run_name: campaign_gates`, so the 6.77M-step weights, `best.zip` and the exploration archives carry on, with the 11-level Tier A + Tier B ladder, `unlock_after_fresh_episodes: 600`, `max_steps` 12000, `item_pickup` 10.0 / `item_placed` 20.0, `num_envs` 12 and `ent_coef` still 0.004. Each config's header lists its own run commands, and **`tests/test_campaign_config.py` pins every setting and every reward weight that is NOT a named change equal to the config before it**, so nothing can drift while the same policy continues.
- `docs/level-survey.md`: all 35 levels parsed offline from the scene files — exits, checkpoints, door graphs and gate ladders, altars and carryables, arenas, bosses and hazards, plus a tier table and the recommended order of sub-projects. This is what the multi-level and skull-gates design was built from; check it before assuming anything about a level nobody has trained on.
- `docs/protocol.md`: the socket protocol.
- `docs/game-internals.md`: game classes and fields the mod relies on (check after game updates).
- `docs/superpowers/specs/`: approved design specs. `2026-09-16-campaign-foundation-design.md` is the campaign design; `2026-09-16-campaign-gates-unwedge-design.md` is the route-gates / un-wedge / look-modes design that followed the 0-1 pilot's diagnosis (implemented in mod v0.6.0 and the `campaign_gates` run; its section 14 records the lead rulings R1-R4); `2026-09-17-multi-level-and-skull-gates-design.md` is the multi-level curriculum / gates guard / 6-2 exit / skull-carry design (implemented on branch `next-levels` and mod v0.7.0; its section 8 lists the five in-game checks, section 10 the risks and section 11 the review dispositions).
- `docs/superpowers/plans/`: implementation plans. `2026-09-16-campaign-foundation.md` is the 16-task plan for the campaign foundation and the 0-1 pilot (Tasks 0-13 dry-run in a scratch copy: mod builds clean, all tests pass). `2026-09-17-next-levels-integration.md` is the ordered checklist for merging branch `next-levels` into main at a training pause and verifying it in the live game (merge, build and install, the six in-game readouts, the 20k-step smoke run, how to carry the live 0-1 policy into the multi-level run, and rollback).
- `.tools/` (gitignored): local `ilspycmd` install used to regenerate `decompiled/`.

## Commands
- Build and install the mod: `cd mod/UltrakillAIBridge && dotnet build -c Release`. The game must be closed, or the DLL is locked. To **compile without installing** (a syntax check while training runs), add `-p:InstallPlugin=false`. `dotnet` is not on `PATH` in every shell here; `& "C:\Program Files\dotnet\dotnet.exe"` works.
- **Which checkpoint to resume from:** `best.zip` (see `best.json` for its score and source). `latest.zip` only
  updates on a GRACEFUL stop (Ctrl+C); every hard kill leaves it stale, and it sat at 1.18M steps for a whole
  night while the run reached 6.5M. Helpers to keep running alongside training, both read-only and safe:
  `python scripts/poll_status.py` (metrics to `runs/<run>/metrics_log.csv`) and `python scripts/keep_best.py`
  (maintains `best.zip`, warns when the current policy falls more than 15% below it).
  For the campaign run use `python scripts/keep_best.py --run campaign_gates --metric campaign`: it scores the
  completion rate over the last 50 fresh-start episodes (samples need `fresh_window >= 20`), breaks ties on the
  lower best official time, and `best.json` records the tie-break as `penalty` / `penalty_name` (older files
  that only have `deaths` still load). It refuses to start when `best.json` was written by the other metric, so
  forgetting `--metric campaign` cannot overwrite the campaign `best.zip`.
- Python env: `cd python && .venv\Scripts\activate` (created with `pip install torch` + `pip install -e .`).
- Bridge check (game open, in a level): `python scripts/bridge_test.py --drive`. In a campaign level, `python scripts/bridge_test.py --campaign` prints the `campaign` block (exit, checkpoints, path, locked doors, arena state, ranks, weapons per slot, and on branch `next-levels` every **gate** with its hops and the hops-coverage ratio the usability guard tests, every **altar** with its wiring and every **item** with its live flags) without taking control, so its `difficulty` is the game's own setting: the override applies only while the AI has control. It **cannot load a level** — pair it with a `campaign_check.py --level X` run, which does, and read the block while that scene is still up. Never run it against a game a trainer is using.
- Parallel training:
  1. `python scripts/games.py launch --count 5` (monitor 3 by default; see the instance-count gotcha for >5)
  2. `python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind_ppo_v2/best.zip` (`num_envs` 5 in config; `timesteps` is the run total, so resuming trains only the rest; drop `--resume` for a fresh run, and give it a new `run_name` so `status.json` does not inherit the old episodes)
  3. `python scripts/games.py stop`
  - `python scripts/games.py status` is safe during training (it reads netstat, it does not connect).
- **Campaign training — the live run. `configs/campaign_gates_main.yaml`, run `campaign_gates`, TWELVE games**
  (since integration pause #2, 2026-09-17; it uses every game, so Cyber Grind stays paused). This is the same run
  that started on 0-1: same `run_name`, same `models/campaign_gates/`, same weights, now a **curriculum** over
  the level survey's Tier A + Tier B — 0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 4-1 — with
  `ent_coef` 0.004, `max_steps` 12000 and the skull weights live.
  `configs/campaign_gates_prelude.yaml` is the previous config, kept as history and as the partial rollback.
  1. `python scripts/games.py launch --count 12 --monitor 1` (needs `[Logging.Disk] Enabled = false`; see the
     instance-count gotcha. `games.py launch` refuses `--count > 5` while disk logging is on, and says why.)
  2. `python scripts/train.py --config configs/campaign_gates_main.yaml --resume models/campaign_gates/latest.zip`
     To continue a stopped run, resume from `models/campaign_gates/latest.zip` after a graceful Ctrl+C, else
     from the newest `ckpt_*_steps.zip` — and **check which of the two actually holds more steps**: `games.py stop`
     kills the trainer with a `BridgeError`, whose `finally` still writes `latest.zip`, so after that stop
     `latest.zip` is the newer of the two (4,805,630 vs the 4,764,785 checkpoint on 2026-09-17).
     Not `best.zip` until completions have peaked and fallen: see the note under the pilot entry.
     **Nothing older than `look_init.zip` will load**: the 12th action dimension makes every earlier campaign
     checkpoint incompatible with the campaign env, and `add_look_mode.py` is the only migration path.
  3. Alongside it: `python scripts/poll_status.py --run campaign_gates`, `python scripts/keep_best.py --run campaign_gates --metric campaign` and `python scripts/dashboard.py --run campaign_gates --monitor 1`.
  4. `python scripts/games.py stop`
  - To start all four detached on the one-display PC, each appending to its own log (this is what is running):
    ```powershell
    Start-Process -FilePath cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized `
      -ArgumentList '/c', '"F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python.exe" -u scripts/train.py --config configs/campaign_gates_main.yaml --resume models/campaign_gates/latest.zip >> runs\campaign_gates_train.log 2>&1'
    ```
    and the same shape for `-u scripts/poll_status.py --run campaign_gates >> runs\campaign_gates_poll.log 2>&1`
    and `-u scripts/keep_best.py --run campaign_gates --metric campaign >> runs\campaign_gates_keep_best.log 2>&1`;
    the dashboard runs from `pythonw.exe scripts/dashboard.py --run campaign_gates --monitor 1`.
  - **Judge a curriculum run per level**, on the dashboard's `levels` rows and `status.json`'s `campaign.levels`
    table, **never on the pooled numbers**: `max_steps` is one value for every level, so a longer level is
    truncated by construction, and `campaign.fresh_completion_rate` becomes a shrunk **sum** over the unlocked
    levels — a score that can exceed 1.0, not a rate. With only 0-1 unlocked it is the number it always was.
  - `configs/campaign_0-1.yaml` is the single-level config this replaced. It still works and is the partial
    rollback: with no `levels` key no curriculum file is opened and the campaign block is byte-identical.
  - `train.py` fills `explore_dir` (the per-game exploration archives, `models/<run_name>/explore_*.npz`) and `best_runs_dir` (`runs/<run_name>/best_runs/`) when the config leaves them empty, before writing `env_config.yaml`.
  - **Renaming a run orphans its exploration archives**, because `explore_dir` defaults to `models/<run_name>/`.
    Copy the `explore_*.npz` files into the new model directory before starting, or the run restarts exploration
    from nothing: their floor counts carry the `1/sqrt(N)` decay that makes the agent push outward at all.
  - Before the first `campaign_gates` run, **copy `models/campaign_ppo_ground/explore_*.npz` into
    `models/campaign_gates/`** (done, committed) and move any `runs/campaign_gates/metrics_log.csv` aside:
    `poll_status.py` keeps an existing header and silently drops columns it lacks, and this change adds several.
  - Commit `look_init.zip` / `transfer_init.zip`, `best.zip` + `best.json`, `latest.zip`, `env_config.yaml` and the `explore_*.npz` archives; the numbered `ckpt_*` files and `models/campaign_smoke/` are gitignored.
  - Judge the run on the gates in the design spec's 11.3, in order, `window >= 50` on every comparison, and never
    on `novelty`/`cells_new`/`oob_frac` against the old run's numbers -- the centre ground ray re-keys ~23% of the
    carried archive cells, `novelty` went 0.5 -> 0.2 and the run name changed, so all three need a fresh baseline
    from the new run's first 100 episodes.
- **Starting a curriculum run under a NEW name** (`configs/campaign_prelude.yaml`, run `campaign_prelude`, the
  same three levels) — kept as the reference config; the live run uses `campaign_gates_main.yaml` instead so
  the weights, `best.zip` and the archives carry on. If you ever do rename:
  1. Copy `models/campaign_gates/explore_*.npz` into `models/<new run>/` **before the first start** and move any
     old `metrics_log.csv` aside (`poll_status.py` keeps an existing header and would drop `levels_unlocked`).
  2. `python scripts/train.py --config configs/campaign_prelude.yaml --resume models/campaign_gates/best.zip`.
     The 0-1 policy carries unchanged: 479 inputs, the same action space, the same reward weights, and only
     `Level 0-1` unlocked at the start — so early on this **is** the 0-1 run continuing.
  The merge-and-verify checklist that was followed is
  `docs/superpowers/plans/2026-09-17-next-levels-integration.md` (done 2026-09-17; see the integration entry
  under Status for what its in-game readouts actually returned).
- TensorBoard: `tensorboard --logdir runs`.
- Campaign in-game check (one game, nothing else connected to its port): `python scripts/games.py launch --count 1 --monitor 1`, then `python scripts/campaign_check.py` (Level 0-1) and `python scripts/campaign_check.py --level "Level 1-1"` (full arsenal), then `python scripts/games.py stop`. Six checks: level load, arsenal, checkpoint trigger, death respawn, exit, and **the gates block** (present, ordered, hop-monotone and unchanged after a respawn; SKIP rather than FAIL against a pre-0.6.0 mod). Prints PASS/FAIL/SKIP per check and a `summary:` line, exits 1 on any FAIL -- **check 5 (exit) is a known standing FAIL on both levels** (the exit's room is switched off at load, so a teleport onto its collider fires nothing; real play does trigger it, confirmed on a human run), so exit code 1 is expected today. `--fixed-fps 60 --frameskip 4` and `--render` repeat the trigger checks at other speed settings. Rerun after game updates and after mod changes to the campaign block; `python tests/test_campaign_check.py` tests the script without the game.
- Skull-carry in-game check (one game, nothing else connected to its port). **All three checks passed on
  2026-09-17** -- see the S4 entry under Status for the readouts and for the two fixes still owed before
  `item_pickup` / `item_placed` may leave 0.0 -- **both are fixed on branch `skull-fixes`**; see its entry at the
  bottom of Status. The command that reproduces it, on Level 1-1:
  ```
  python scripts/skull_check.py --port 47808 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231 \
      --approach 81 -1 255 --altar-approach 0 -4 374 --budget 700
  ```
  `--from-checkpoint` teleports onto a checkpoint, waits for it to activate **and then respawns there**, which is
  what actually switches the pedestal's room on; `--approach` / `--altar-approach` stage into a room already lit
  and the last metres are still walked. Since branch `skull-fixes` the **`--altar 0 -8 381` offset is gone**:
  `--altar` takes the zone's own reported position and the script aims at its collider centre (`aim_pos`, else
  1 m below), so both hacks the first in-game run needed are unnecessary.
  `--render` is the control run, `--via X Y Z` a walked waypoint, `--camera-height` the 0.9 m eye offset the
  punch ray starts from. Check 1 failing while the pre-punch line says `active=False` means the room is still
  switched off and the **walk** failed, not the punch. `python tests/test_skull_check.py` tests it without a game.
  - **To run it while a training run is live, never use `games.py launch`/`stop`** -- both call `stop_all()` and
    kill every game. Start one extra instance by hand on a free port with games.py's own arguments
    (`-aibridge-port N -screen-fullscreen 0 -screen-width 368 -screen-height 207 -job-worker-count 3`,
    `SteamAppId`/`SteamGameId` 1229490, detached + below-normal, cwd the game folder), note its PID, and at the
    end `taskkill /PID <pid> /F` that PID alone. Verify before and after with `games.py status` (netstat-based).
- Campaign eval (one game on port 47800, e.g. `python scripts/games.py launch --count 1 --monitor 1`):
  `python scripts/eval.py models/campaign_gates/best.zip --level "Level 0-1" --episodes 10`. Fresh level loads,
  deterministic actions, real deaths; prints completed, official time, rank, kills, style, restarts and deaths per
  episode, then the completion count and the fastest run. It reads the exploration counts the first training game
  saved next to the model (`explore_Level_0-1_47800.npz`, printed as a cell count; 0 cells means the policy sees
  an unexplored map) and never writes them. Add `--record-times` to write the fastest completion to `times.md`.
- Live dashboard: `python scripts/dashboard.py` (newest run) or `--run cybergrind_ppo_v2`; opens on monitor 3 below the game row (`--monitor`, `--reserve-top`); `--smoke-test` renders once and exits. A campaign run replaces the Shooting panel with a Campaign panel (fresh and all-episode completion rate, best and median official time, **gates per load** and checkpoints per load with the all-episode and fresh-start means side by side, **wedged steps per episode**, a **look free/gate row carrying the three per-dimension entropies**, new cells, deaths, closest to the exit, the four largest reward parts), charts fresh completion % and **gates per load** instead of kills/min and wave, and lists checkpoints instead of waves per game. On branch `next-levels` a **multi-level** run adds a `levels` block (one row per unlocked level: fresh rate and window, best time, checkpoints per load, sampling weight) and relabels the headline `fresh score N / K levels`, because the pooled figure is then a shrunk sum and can exceed 1.0.
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py`, `python tests/test_campaign_rewards.py`, `python tests/test_spaces.py`, `python tests/test_campaign_env.py`, `python tests/test_keep_best.py` and `python tests/test_times.py` (pytest is not installed; the files also work under pytest). Also `python tests/test_transfer.py` and `python tests/test_look_mode_transfer.py` (weight surgery) and `python tests/test_campaign_config.py` (the campaign config and `train.py` wiring). All of them at once, from `python/` in PowerShell: `Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }` (**15 files; 304 named tests** as of 2026-09-17, of which `test_progress.py`'s 19 print no count; ~2 min). `tests/test_games.py` covers `games.py`'s instance-count guard. `test_campaign_check.py` and `test_skull_check.py` print `[FAIL]` lines from their own fake levels on purpose -- they are asserting that a broken level is reported as broken -- so judge them on their last line and their exit code.
  **In a git worktree**, run them with the main venv but with `PYTHONPATH` pointed at the worktree: the package is an editable install pointing at the main tree, so without it you silently test the wrong code. Verify once with `python -c "import ultrakill_ai; print(ultrakill_ai.__file__)"`.

## Key design decisions
- **Lockstep:** the mod blocks Unity's main thread between steps. `Time.captureDeltaTime = 1/60` fixes game time per frame, and uncapped FPS makes training faster than real time. Game speed is not controlled through `Time.timeScale`, which `TimeController` owns for hitstop.
- **Input:** injected through virtual Input System devices rather than Harmony patches on `InputActionState` getters, which Mono may inline.
- **Rewards and observations:** computed in Python from raw mod data, so tuning needs no mod rebuild.

## Moving the project to another machine
- Checkpoints come with the repo (`python/models/`). `runs/` is gitignored, so copy `python/runs/cybergrind_ppo_v2/status.json` (dashboard history) and `python/runs/cybergrind_ppo_v2_1/` (TensorBoard curves) by hand if you want them.
- On the new machine: install BepInEx 5 into the game, copy `mod/GamePaths.props.example` to `mod/GamePaths.props` with the game path, build the mod (Commands above), create the venv (`pip install torch` then `pip install -e .`), run the no-game tests, then `games.py launch --count 5` and `train.py --config configs/cybergrind.yaml --resume models/cybergrind_ppo_v2/best.zip`.
- Re-check the monitor layout: `games.py` tiles on monitor 3 by default (`--monitor`), and the dashboard follows it.

## Local machine state (the original PC, 2026-09-15)
- **Game:** `E:\SteamLibrary\steamapps\common\ULTRAKILL`, Unity 2022.3.29 Mono.
- **Mods:** BepInEx 5.4.23.5 installed 2026-09-15; UltrakillAIBridge plugin installed.
- **Save backup:** `C:\Users\tyler\Documents\ULTRAKILL-Saves-Backup-2026-09-15`.

## Local machine state (the second PC, 2026-09-15)
- **Game:** `C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL`, Unity 2022.3.29 Mono (same build).
- **Hardware:** Ryzen 9 3900X (12C/24T), 32 GB, RTX 2080 SUPER. One display only, so launch with
  `games.py launch --count 5 --monitor 1` and `dashboard.py --monitor 1` (the default `--monitor 3` falls
  back to the primary with a printed note, but pass it explicitly).
- **Toolchain installed 2026-09-15:** Python 3.12.10 and .NET SDK 8.0.425 (winget), BepInEx 5.4.23.5,
  venv with torch 2.14.0+cpu / sb3 2.9.0 / gymnasium 1.3.0 / numpy 2.5.3. `GamePaths.props` is a
  straight copy of the example: the example's path already matches this machine.
- **Save backup:** `C:\Users\tyler\Documents\ULTRAKILL-Saves-Backup-2026-09-15-newpc`.
- **Migration gap:** `runs/` is gitignored and was not carried over, so the v2 dashboard history and the
  TensorBoard curves before 1.179M steps are gone. `models/` is committed now, so the weights survived.

## Gotchas found in the live game
- **Manager object destroyed:** ULTRAKILL destroys BepInEx's manager GameObject. The bridge runs on its own `HideAndDontSave` + `DontDestroyOnLoad` object (`BridgeRunner` in `Plugin.cs`).
- **Background running:** the game ships with `runInBackground` off. The plugin turns it on so the bridge answers while the window is unfocused.
- **Frame cap returns:** scene loads re-enable vsync or a frame cap, so time settings are re-applied on every step and reset.
- **Lockstep timing:** runs at end of frame (`WaitForEndOfFrame`), so obs reflect the finished frame and queued input lands next frame.
- **Hitstop:** the game waits with `WaitForSecondsRealtime`; `TimePatches` makes it frame-based during lockstep.
- **Stuck buttons:** virtual devices need `InputSystem.ResetDevice` before removal, or actions stay stuck pressed.
- **Cyber Grind start:** the player spawns on a ledge (≈ z -47, y 100.5), and waves start only on entering the `EndlessGrid` trigger collider. The mod reports it as `cybergrind.start_trigger`, and the env teleports into its center (lands ≈ (2.5, 26.5, 65)). Walking off the ledge remains as a fallback.
- **Background play:** while in control the game switches to a 640x360 window (`windowed`, `window_width/height`), unlocks the cursor every frame and after `GameStateManager.EvaluateState`, and re-mutes after every scene load, because `GameStateManager.IntroCheck` restores the volume. The original resolution and volume are restored on release and on quit.
- **Single-client bridge:** `BridgeServer` drops its current client whenever a new one connects, so any TCP connection to a training instance's port kicks the trainer off that game and the run dies with `Connection closed by the game` in every env. `games.py status` (and the launch readiness wait) therefore read listening ports from `netstat` instead of connecting. Never poke the ports with another client while training.
- **Multiple instances:**
  - The game holds `Preferences/Prefs.json` open with exclusive write access, so a second copy crashes unless `InstancePatches` opens it read-only.
  - Launching the exe directly works (Facepunch `SteamClient.Init`; the launcher sets `SteamAppId`).
  - The game re-applies resolution from `LocalPrefs.json` in `InitGame` at startup, so Unity's registry `Screenmanager*` values barely matter.
  - BepInEx writes `LogOutput.log.1..3` for extra instances.
- **The five-copy cap was BepInEx's disk log, not the game (measured and lifted 2026-09-17).** BepInEx's
  `DiskLogListener` opens `LogOutput.log` plus `.1`-`.4`, so a sixth copy could not open a log file and the
  plugin never loaded. Setting `[Logging.Disk] Enabled = false` in
  `C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\config\BepInEx.cfg` removes the listener and
  the cap with it: **8 copies load, serve the bridge and train**. Nothing in `games.py` capped it either — ports
  are `base_port + i`, readiness is read from netstat and `tile` wraps onto a second row — so the only change
  needed was a guard: `launch --count N > 5` now refuses while disk logging is on and names the file to edit
  (`games.py:disk_logging_enabled` reads `Enabled` inside `[Logging.Disk]` only, because `[Logging.Console]` has
  an `Enabled` key of its own a few lines above that is `false` by default; `tests/test_games.py` pins that).
  **To undo:** the pre-change file is kept as `BepInEx.cfg.bak-preN` beside it — restore it (or set `Enabled`
  back to `true` in the `[Logging.Disk]` section) and put `num_envs` back to 5. **The cost:** with the disk
  listener off the plugin only logs to the console listener, so a crash leaves no file behind. Turn it back on
  (and drop to 5 copies) when diagnosing one — that is where `campaign_check.py`'s key-collision and exit-tie
  warnings go.
- **Throughput scales well past five games (measured 2026-09-17, Ryzen 9 3900X 12C/24T, 32 GB).** Same
  checkpoint, same config, ~15k steps each, steady-state steps/s from SB3's own counters between iteration 1 and
  iteration 8 (`n_steps` in the config is the TOTAL rollout size and `train.py` divides it by `num_envs`, so the
  PPO batch is identical at every N and the comparison is fair):
  | games | steps/s | vs 5 | CPU |
  |---|---|---|---|
  | 5 | 152 | — | — |
  | 7 | 183 | +20% | 31% |
  | 8 | 235 | +54% | 33% |
  The second A/B (integration pause #2, same day, from the real 6.77M resume checkpoint on
  `configs/campaign_gates_main.yaml`, ~18k new steps per leg into a scratch `campaign_smoke` run whose
  exploration archives were seeded from the live ones):
  | games | steps/s | vs 8 | notes |
  |---|---|---|---|
  | 8 | 188.6 | — | 9 rollouts, 0 bridge errors |
  | 10 | 220.5 | +16.9% | 10 rollouts, 0 bridge errors |
  | 12 | 236.5 | **+25.4%** | 10 rollouts, 0 bridge errors; 9.6 GB of 31.9 GB, 16.6 GB free |
  **12 was adopted** (the bar was +10% over 8). Read these as ratios, not absolutes: the two A/Bs disagree on
  8 games (235 vs 188.6) because the first measured SB3's cumulative `fps` on short legs that saw few level
  loads, while this one takes the slope between rollout 1 and the last, excluding start-up. In the live run at
  12 games the first rollouts settled at **~270 steps/s**. Scaling still had not flattened at 12.
  **The aggregate rate IS the per-game responsiveness check**: `SubprocVecEnv` steps every env in lockstep, so
  one slow or wedged game drags the whole vector down and would show up here as a collapse, not as a quiet loss.
  Nothing in `games.py` needed changing for N > 8 — ports are `base_port + i`, `tile` wraps onto more rows, and
  `status` scans 16 ports. **Seed a new port's exploration archives** from an existing port's before using it
  (`models/campaign_gates/explore_<level>_<port>.npz`), or it re-pays novelty for ground the run has covered;
  47808-47811 were seeded from 47800-47803 exactly as 47805-47807 were from 47802-47804.
- **Training speed findings** (random actions, 4–5 games):
  - The PPO update is only ~0.2 s per 2048 samples on CPU, so the GPU wouldn't help. The time is in the games and resets.
  - 60 fps / frameskip 4: 138 steps/s. 30 fps / frameskip 2 (same 15 decisions per game second; game logic is deltaTime-based): 199.
  - Rendering off: 217. Soft death: 297. Both: 328. With 5 games: 380.
  - A 6th copy fails because BepInEx opens at most `LogOutput.log` plus `.1`–`.4` — **unless disk logging is
    switched off**; see the five-copy-cap gotcha above.
- **Lockstep stalls:** vectorized envs step together, so a reset in one game stalls all of them. Keep resets short (teleport entry, `reset_settle_frames` 10).
- **Speed:** about 600 fps / 150 steps/s in an empty scene and about 100 steps/s with enemies (frameskip 4, RTX 5070).
- **Campaign checkpoint triggers fire at training speed; the exit was never a trigger-timing test:** checkpoint triggers activate correctly at 30 fps / frameskip 2 with rendering off (`campaign_check.py` check 3 PASS on 0-1 and 1-1, 2026-09-16), so no speed-setting change was needed. Check 5 (exit) FAILed on both levels, but not from timing: see the exit-room gotcha below.
- **Campaign throughput:** 177 steps/s with 5 games on 0-1 at 30 fps / frameskip 2 with rendering off (20k-step smoke run from fresh weights; SB3's `fps` over the whole run, level loads included). At that rate 1M steps take 1.6 h and the 20M-step pilot 31 h.
- **Exit room switched off at load; checkpoint activation does not bring it up (0-1 and 1-1, 2026-09-16):** a fresh load reports the exit `active: false` on both levels, so `campaign_check.py` check 5 fails there. Automated stand-in for playing through, since `CheckPoint.ActivateCheckPoint` calls `SetActive(true)` on its own room (`decompiled/CheckPoint.cs`): teleported onto and activated every checkpoint in spawn-distance order. 0-1 activated 5 of 6 (the nearest did not activate within 30 decisions on this attempt); 1-1 activated 4 of 4. The exit stayed `active: false` on both afterward -- so checkpoint activation does not switch on the exit's own room; it sits well past the last checkpoint (293 m on 0-1, 469 m on 1-1 from the last checkpoint tried). Not verified by hand (no human play in this session). Treated as a known open item, not a blocker: check 5's code path is covered by `test_campaign_env.py` against the fake bridge, and the first real level completion during training exercises it live. (The `path.status` staying `none` the whole walk turned out to be a separate NavMesh-sampling bug, not caused by the room being inactive -- see the exit-sample fix below; `FinalPit` is findable and its `transform.position` valid via `FindObjectsOfType(true)` regardless of active state.)

- **Campaign NavMesh path, measured 2026-09-16.** The path hint only resolves near the exit. At `Level 0-1`'s
  spawn the player is grounded and snaps onto the mesh fine, but the mesh island holding the exit is not
  connected to the start area, so `NavMesh.CalculatePath` finds nothing and the block reports `status: "none"`;
  from the level's farthest checkpoint it reports `partial`, 72.4 m. `Level 1-1` reports `partial` from spawn
  (16.6 m). So the `path` reward pays nothing over the first part of a level by design, and exploration plus the
  milestones carry it; the straight-line exit vector is in the observation either way. `PlayerSnapDistance` is
  25 m rather than 6 m so the hint does not drop out every time the agent is airborne, which in this game is
  most of the time.

- **The baked NavMesh is an ENEMY-walking mesh, and `path` was retired because of it (2026-09-16).** 0-1's mesh
  is 412 polys in 47 islands with **nothing before z 378.5**, and no bridging rule connects the start area to the
  exit below a 17 m gap / 12 m climb limit — one that did would also bridge through walls. Over 32,022 logged
  decisions `path.status` was `partial` on 82.19% of rows and **never once `complete`**, so the `path` reward
  paid **0 in 656 logged rows** and its `status == "complete"` observation input had an exactly zero first-layer
  column norm in both hidden stacks. `path` is now weight **0** and its observation slots carry the route target
  instead. A NavMesh triangulation dump was considered and rejected for the same reason. Do not reach for the
  NavMesh as a route signal again: `campaign.gates` is the route signal.

- **`campaign.gates` is the route, and it is a door graph, not a path.** A gate is a `Door` whose
  `activatedRooms` holds >= 2 distinct rooms; those rooms (and only those) are the graph's nodes, keyed by
  **rounded world position** — never by reference, because `CheckPoint.Start` clones each room, deactivates the
  original and banishes it +10,000 m in X, and `ResetRoom()` re-instantiates the clone on every respawn, so the
  same physical room is a different object across a death. BFS from the exit's nearest room-node ancestor gives
  each door its `hops`. Measured live: 0-1 11 gates 0..9, 0-2 17 gates 0..7, 1-1 13 gates 0..5, **0-5 unordered**
  (no room node on the pit's ancestor chain) — Python falls back to the straight-line exit vector there.
  Rules worth remembering: `key`/`pos` are the **closed** position and frozen for the level load (a Normal door
  moves its OWN transform by `openPos`, measured (0, 5.75, 0) on every gate door of 0-1..0-5, so a rescan while
  the player stands in the proximity trigger would otherwise flip the key); `key` is **opaque**, including any
  `#2` suffix, because `Mathf.RoundToInt` rounds halves to even and 0-1's doors sit on exact `.5` z values;
  `open` means "currently open or opening", never "has been passed" (enemies open doors, and opening one
  force-closes the others); and `controller_active` is what separates "walk up to it" from "a fight gates it" —
  0-1's gun-room gate reports `open: false, locked: false` at load and still cannot be opened.
  Levels beyond 1-1 are unverified apart from 0-2 and 0-5.

- **The absorbing `slowMode` wedge, and the fix (mod v0.6.0).** A slide that ends while the player is airborne,
  or a jump out of a slide where the game's stand-up test fails, sets `NewMovement.slowMode` and leaves a state
  nothing can end: `grounded` never comes back (`GroundCheck` keeps `onGround` from trigger callbacks over a
  `cols` list, and a collider the capsule already overlaps never fires `OnTriggerEnter` again), velocity reads
  `(0,-100,0)` or ~0, stamina is frozen, jump/dash/slide are inert, and a **0.477 m/s creep** is all that
  remains. It is a general state, not a vent property: measured in 0-1's slide vent, at the spawn-room wall and
  at (145.4, 45.5, 664.2) after checkpoint 3. On the 32,022-decision live log exactly **five** runs of >= 45
  consecutive wedge decisions exist (2976, 1108, 729, 694, 672) and nothing else exceeds 29.
  The fix is `UnwedgePatch`, a postfix on the private `NewMovement.HandleSlideState`, active only while
  `EpisodeController.InControl`: clear `heavyFall`, zero a `<= -99` downward velocity, re-assert the crouch
  triple defensively, clear `slowMode`, then `ForceGroundCheck()` on every `GroundCheck` instance — **in that
  order**, because `GroundCheck.OnTriggerEnter`'s `heavyFall` branch deals x5000 damage to overlapping enemies
  and can `Bounce()`, which teleports the player.
  - **The hold is load-bearing.** The guard is held for `unwedge_frames` (**10**) consecutive frames. `Update`
    runs `HandleInputs()` before the `heavyFall` block and before `HandleSlideState()`, and `TryStartSlam` calls
    `StopSlide()`, which does **not** restore `playerCollider.height` — so a legitimate ground slam started out
    of a slide arrives at the stand-up test with height 1.25 still set and, under any ceiling, gets
    `crouching`/`slowMode` set on the very frame `heavyFall` and `(0,-100,0)` were. Acting on frame 1 silently
    cancelled the slam: no `LandingImpact`, no ground-slam enemy damage, and no `Breakable.Break(2f)` — one of
    the few ways through the planks that seal 0-1's starting room. Measured A/B: at `unwedge_frames` 1 the slam
    is dead on decision 1; at 10 it survives 4 decisions and the real wedge still breaks at 0.400 s.
  - `unwedge` (bool, default true) is a kill switch and `unwedge_frames` its hold. **Both are sticky per game
    process**, like `soft_death`: a client that sends `unwedge:false` and disconnects leaves it false for the
    next client on that port. Never send `unwedge:false` to a port a trainer might share.
  - A **grounded** crouch under a low ceiling is the same `slowMode` and is deliberately untouched; it still
    reports `slow_mode: true`.

- **0-1's `unlock_all_gear` alternate start, and the vent.** With `unlock_all_gear` on, `GearCheckEnabler` swaps
  in a short starting room at (39.7, -0.5, 343.7), sealed by two weak `Breakable` planks at (40,0,347.5) and
  (40,2,347.5). Punching through them or sliding under the lower one (a 1.5 m gap against a 1.25 m slide height)
  is the only way out, and that same opening is the **slide vent** (x = 40.0, z 349 -> 377.5, 1.0 m wide) where
  the wedge happens. **A scripted grounded slide passes it every time** — measured again at integration: forward
  + slide from spawn reached the gun-room door at z 401 in 42 decisions. So the older note that "0-1 spawns you
  sealed in" was wrong as stated: the room is sealed to a *teleport* and to a *walk*, not to a slide, which is
  why the scripted `walk_to_exit.py` could not leave it. `punch` is still charged before `level_started`, which
  prices the only other tool a weaponless player has; left alone deliberately, flagged in the design spec.
- **Reward rebalance at 329k steps:**
  - Problem: with kill 2 / death 10 the agent learned to avoid fights. Reward went -9.7 → -1.8 and episode length 124 → 772 steps, but kills per game-minute fell from ~22 to ~3.
  - Now kill 5 / death 5 / damage_dealt 2.
  - One-shot kills credit the vanished enemy's remaining health as damage dealt.
  - The dashboard shows kills/min (`info["kills_per_min"]`).
- **Aim shaping at 820k steps:** kills/min only recovered 2.5 → 4.2 and the aim reward was 0.04 per episode, i.e. the agent rarely faced an enemy. Raised `aim` 0.02 → 0.06 with a 25° cone and lowered `ent_coef` 0.01 → 0.005. Config hyperparameters now override the saved ones when resuming.
- **Dashboard layout:** games tile in one row (368x207) along the top of monitor 3; the dashboard fills the space below. `ProgressCallback` reloads the existing `status.json`, so restarting training keeps episodes, charts, bests and elapsed time.
- **Timed episodes at 1.03M steps:** ending episodes on death made hiding optimal — reward climbed 13 → 23 while kills/min stayed ~3.5 and episodes stretched to 65 s. Cyber Grind needs kills to advance waves, so episodes are now a fixed 2 minutes of game time (`max_steps` 1800, `end_episode_on_death: false`) and continue through soft deaths; the return therefore measures kills per minute. `death` back to 8, `max_wave` 0, and `hard_reset_above_wave` 5 reloads the arena so early waves keep appearing. Dashboard shows deaths per episode.
- **Behaviour diagnostics (1.68M steps):** per-episode `firing_frac`, `on_target_frac`, `firing_on_target_frac` (env → Monitor → dashboard "Shooting" panel). Measured: firing 75% of steps, enemy within 15° of the crosshair only 1.1%. The agent sprays and kills at close range instead of aiming, so `aim` went 0.06 → 0.15 with a 20° cone. Lower it again once `on_target_frac` climbs.
- **Timed episodes helped:** kills/min 2.5 → 3.5, reward 25 → 45, deaths/ep 1.6 → 1.2 between 1.05M and 1.68M steps.
- **Aim shaping needs a gradient (1.9M steps):** a cone-only aim reward paid zero beyond the cone, and the agent was inside it 0.8% of the time, so raising the weight (0.06 → 0.15) changed nothing (on-target went 1.1% → 0.8%). `aim` is now a slope from facing away (0) to facing straight at the nearest visible enemy, with `aim_locked` extra inside the cone.
- **Weight audit at 2.3M steps (why it never aims):** probing the policy offline and simulating its look dynamics from the weights alone reproduces the live stats (enemy angle 89°, on-target 0%, aim 0.040 per visible step, vs 88.8°, 0.9%, 0.041 in training). The pitch head has carried a constant +3°/step bias since ~300k steps, independent of weapon, health and enemy position, so within 5 game seconds the camera pins at the +90° clamp (the sky) and every enemy sits ~90° off the crosshair whatever the yaw does. Under random yaw the aim slope pays the same 0.5 level or pinned, so nothing ever corrected the drift. The yaw head only began tracking enemies in the last 300k steps (corr(rel.x, E[yaw]) 0 → 0.39, right sign) and the pin masks it: with pitch forced level the same weights are on target 12% of steps. The rest of the policy is fire always, dash often, lean backwards (entropy 11.7 → 5.9). Updates are noise-dominated: Adam gradient SNR sits at the noise floor and the checkpoint-to-checkpoint weight drift is a random walk (path/net ≈ 6–8 ≈ √40 over 40 checkpoints). Candidate fixes: clamp or auto-level pitch in Cyber Grind, penalise |pitch|, or split the aim reward into yaw and pitch terms.
- **Pitch fix (2.30M–2.45M steps):** `pitch_limit_deg` clamps the camera to within a band of level on the Python side (the pitch command is trimmed against the last observed pitch, so no mod change and the camera snaps back into the band in one step), and the aim reward is split per look axis so each head gets its own gradient: `aim_yaw` 0.06 on the ground-plane heading error and `aim_pitch` 0.06 on the enemy elevation in camera space, with `aim` 0 and `aim_locked` unchanged. New per-episode diagnostics `pitch_abs_mean` and `enemy_yaw_angle_mean` flow into status.json and the dashboard Shooting panel. A first pass with a 40° band did not help by itself: after 150k steps the policy just leaned on the new edge (pitch 35°, on-target 1.6%), which still put eye-level enemies outside the aim cone and the hitbox. So the band is now 15° and the pitch head of `ckpt_2447005` was zeroed (output rows, bias and their Adam moments, giving uniform pitch actions) into `models/cybergrind_ppo/pitch_reset_2447005.zip`. That resume ran only minutes before the decision to start v2 fresh instead. Expected for v2: camera pitch under 15°, on-target climbing from ~1% (random heading alone gives ~8% inside 15°, and the same weights reached 12% with a level camera), then kills/min.
- **Mod:** v0.6.0 (background play, training instances, teleport, soft death, rendering off; campaign block, `difficulty` and `unlock_all_gear` config, `kill` debug command, `player.slot_counts`, game-initiated restarts blocked while in control; **the `campaign.gates` route block with its room-graph BFS, the `ChooseExit` secret-pit and mission-successor fix frozen per level load, `ground_ray_center`, `player.slow_mode`/`heavy_fall`/`crouching`, and the `UnwedgePatch` postfix with its `unwedge` / `unwedge_frames` config keys**). Verified in game: plugin load, handshake, Cyber Grind reset, movement/look/jump/dash, observations (enemies, waves, damage, death), leaderboard block, and every v0.6.0 addition (acceptance A1-A12 under Status). Protocol version is still **1**: obs gains fields, nothing is removed or changed.
- **Python:** env, training and eval verified against a mock and partly in game; campaign episodes verified against a fake level (`tests/test_campaign_env.py`).
- **In game:** `UltrakillEnv` auto-enters the Cyber Grind arena on reset (`auto_enter_arena`). The random-agent smoke test passes at about 70 steps/s.
- **Training:** first Cyber Grind PPO run (`cybergrind_ppo`) started 2026-09-15 and was abandoned at 2.45M steps after the weight audit: its updates had been noise-dominated and it carried locked-in reflexes (fire always, dash, walk backwards, look up), so a resume was not worth the carry-over.
  - v1 output (kept for reference): `python/runs/cybergrind_ppo_1` (TensorBoard), log `python/runs/cybergrind_ppo_train.log`; of its checkpoints only `latest.zip` (2.30M) and the hand-edited `pitch_reset_2447005.zip` are in git, the numbered ones are ignored.
  - **v2, fresh start 2026-09-15** (`cybergrind_ppo_v2`): corrected reward (per-axis aim, `pitch_limit_deg` 15), `ent_coef` 0.01, `target_kl` 0.02, 5M steps. Output: `python/runs/cybergrind_ppo_v2_1`, log `python/runs/cybergrind_ppo_v2_train.log`, checkpoints in `python/models/cybergrind_ppo_v2/`.
    - Early signal at 344k steps: camera pitch 11° (band 15°), heading error to the nearest enemy 84° → 70° between 130k and 344k steps, so yaw tracking is emerging; kills/min 1.6 and deaths 2.8/episode (a fresh policy has not learned to back off yet).
    - 850k steps: heading error 49°, yaw tracking in the weights grew (corr(rel.x, E[yaw]) 0.21 → 0.61), deaths 2.2, but on-target stuck at 3% and kills/min 1.8. Checkpoint probes show the pitch head is not tracking elevation (corr ≈ 0); it developed a plain downward bias that the band holds at −15°, and the pitch reward implies enemies average ~27° off the horizon. Added per-episode `enemy_elev_mean`, `enemy_elev_abs_mean`, `enemy_elev_over15_frac`, `pitch_mean` and `look_up_mean` (status.json, dashboard) to size the band from data; resumed from `ckpt_850000`.
    - 1.0M steps, elevation measured: the nearest visible enemy sits more than 15° off the horizon 65% of the time (mean |elevation| 26°, signed mean +8°), so the 15° band blocked aiming most of the time and capped on-target at 4%. Band widened to 45° (`pitch_limit_deg`) and `enemy_pitch_err_mean` (vertical miss in camera space) added to the diagnostics; resumed from `ckpt_1006680`. Also confirmed from `pitch_mean` vs `look_up_mean`: positive `rotationX` looks up. kills/min 2.75, reward 139, entropy 9.6 at this point.
    - **Stopped 2026-09-15 at 1,179,085 steps** (graceful Ctrl+C, `latest.zip` saved) to move the project to another PC. Resume with `--resume models/cybergrind_ppo_v2/latest.zip`; the first thing to read on the dashboard is `pitch_track` vs `yaw_track`.
    - 1.06M: first read after widening showed the camera going straight to the 45° edges (pitch 30° mixed over old and new episodes) with the vertical miss unchanged at 29°. Added live tracking scores `yaw_track` and `pitch_track` (−1..1: sign agreement between the unclamped look command and the nearest visible enemy offset; random play scores 0) to see whether each look axis actually turns toward enemies; resumed from `ckpt_1056680`.
  - Throughput:
    - 1 game: about 36 steps/s
    - 4 games with walk-in resets: about 80 steps/s
    - 4 games with teleport resets: about 120 steps/s, CPU ~90%, resets ~0.6 s
    - 5 games, 30 fps / frameskip 2, soft death, no rendering, 480x270 windows: ~255+ steps/s in real training
  - Games run on monitor 3 (`\.\DISPLAY3`, x 1920–3840).
- **Resumed on the second PC at 1.179M steps (2026-09-15).** Throughput ~190 steps/s with 5 games
  (vs ~255 on the original PC). An offline probe of `latest.zip` (`probe_policy.py`: synthetic observations
  with one visible enemy, reading the yaw/pitch bin distributions) shows v2 has drifted into v1's failure
  mode: both look heads have the right slope (corr(rel.x, E[yaw]) +0.74, corr(rel.y, E[pitch]) +0.68) but
  carry a large constant offset, so `E[yaw]` is positive at every azimuth (+19 deg/step at azimuth 0, still
  +15 at -90 deg) and `E[pitch]` is positive at every elevation (+3 deg/step, v1's exact bias). Sign
  agreement is 50% on both axes, i.e. random: the camera spins right at ~285 deg/s and pitch pins against
  the 45 deg band. Live metrics agree: `yaw_track` 0.13, `pitch_track` 0.02, `on_target_frac` 3.2%,
  `pitch_abs_mean` 27 deg, `firing_frac` 86%.
- **Why the bias survives: the aim reward is mostly an unconditional floor.** Per-episode reward parts at
  1.19M steps: `aim_pitch` 65.0, `aim_yaw` 58.3, `kill` 31.0, `death` -12.8, `damage_dealt` 12.7,
  `aim_locked` 5.3, `damage_taken` -4.4, `style` 2.6 (total ~158). Aim shaping is 78% of the return, but the
  slope `w * (1 - err/180)` pays `w/2` per step at a uniformly random heading: about 54 points an episode
  per axis that no behaviour can avoid. Only ~15 of the ~123 aim points vary with aim quality, spread over
  1800 steps, so the gradient that would cancel the constant offset is far smaller than the kill term's own
  variance -- the same "noise-dominated updates" as the v1 audit, with the cause now identified. The one
  genuinely contingent term, `aim_locked` (paid only inside the 20 deg cone), is worth just 5.3.
- **Reward rebalance at 1.70M steps (2026-09-15), the change now running.** Baseline over 100 episodes at
  1,697,290 steps: reward 138.8, kills/min 2.64, deaths 1.91, `on_target_frac` 3.9%, `yaw_track` 0.149,
  `pitch_track` 0.065, `enemy_visible_frac` 0.748, entropy 9.66, `approx_kl` 0.031 (over `target_kl` 0.02).
  Parts: `aim_pitch` 57.2, `aim_yaw` 53.1, `kill` 26.4, `death` -15.3, `damage_dealt` 11.0, `aim_locked` 6.1,
  `damage_taken` -5.2, `wave` 3.0, `style` 2.4 -- aim shaping 84% of the return. Four independent analyses
  and four adversarial checks produced three findings that changed the plan:
  - **`visible` is line-of-sight only.** `ObservationBuilder.cs:165` is `!Physics.Linecast(cam.position,
    center, envMask, ...)` with no frustum test, and the list is distance-sorted, so `visible[0]` in
    `rewards.py` is the nearest unoccluded enemy whether or not it is on screen -- an enemy behind the player
    counts. The slope terms therefore paid ~100 points an episode for standing where an enemy can see you.
    That is an exposure subsidy, and exposure is what drives `damage_taken`; deaths rising 1.60 -> 2.38 was
    this term working as specified, not a mispriced `death`.
  - **A linear slope on a uniformly-swept error is gradient-free**, which is the real mechanism (the earlier
    "unconditional floor" framing was wrong: a constant per-step reward is absorbed by the value baseline,
    and `explained_variance` is 0.96). With the probed bias c = +19 deg/step and tracking gain
    k = (19-15)/90 = 0.044, the closed loop az <- (1-k)az - c has its fixed point at -c/k = -430 deg, outside
    [-180,180]. No lock-on exists, so the camera sweeps, yaw error goes uniform on [0,180], and the mean
    reward is identical for every c in (8, 19]. Lock-on only begins below c = k*180 = 8, so no value of
    `aim_yaw` can move the policy off the sweep -- the slopes had to be demoted, not retuned.
  - **`wave` was the most underpriced term.** Cyber Grind spawns nothing until the wave is cleared, so fewer
    kills -> no wave advance -> fewer enemies -> less line of sight -> less reward and fewer targets. That
    self-reinforcing spiral is what drove `enemy_visible_frac` 0.857 -> 0.669. Waves cannot be farmed without
    killing, so the term is purely contingent.
  Applied: `aim_yaw` 0.06 -> 0.02, `aim_pitch` 0.06 -> 0.01 (half of yaw, so the per-degree gradients match
  at 1.11e-4), `damage_dealt` 2 -> 4, `kill` 5 -> 8, `death` 8 -> 6, `wave` 5 -> 20. `aim_locked` stays 0.15,
  `aim_cone_deg` 20 and `pitch_limit_deg` 45 unchanged. **All four proposals wanted `aim_locked` raised (to
  0.25-0.8) and all four adversarial checks rejected that**: above w = 0.18 a tracker that holds an enemy in
  the cone and never kills it out-earns a policy that fights, because the cone subtends a fixed angle and
  close combat is the worst case for holding a target inside it. Cutting the slopes raises `aim_locked`'s
  share without touching its ceiling. Freezing the cone also keeps every predicted part an exact rescaling
  of a measured one. Predicted composition with behaviour unchanged: total 138.8 -> 95.4, aim share
  84% -> 35%, combat (`kill` + `damage_dealt` + `wave`) 80% of the return; a good combat policy scores ~719
  an episode against the best non-killing tracker's ~194.
  Resumed from `ckpt_1679085_steps.zip` (not `latest.zip`, which was left stale at 1.179M by the hard stop),
  `timesteps` raised 5M -> 8M so an overnight run does not idle.
  - **Falsifiers.** Do NOT judge this by total reward: it falls to ~95 by construction and its composition
    changed. Primary: `on_target_frac` must reach >= 0.09 by 2.18M steps and >= 0.15 by 2.68M; if it is
    still <= 0.06 at 2.18M the floor:signal hypothesis is not the binding constraint and the next suspect is
    the look-head parameterisation itself (the offset lives in the bias, so that needs a head reset like
    v1's `pitch_reset`, not another weight pass). Early tripwire at 1.83M: `yaw_track` >= 0.30, up from
    0.149 -- it moves before `on_target_frac` does. Guardrails: deaths/ep > 5.0 means the `death` cut was
    too deep; `enemy_dist_mean` > 32 with `on_target_frac` rising means `aim_locked` is paying for easy
    long-range tracking; and kills/min must not sit below 1.95 at 2.68M, since aim improving while kills
    fall is the signature of a tracking policy that stopped engaging.
- **Failed experiment at 1.98M: a pitch-head bias tilt, applied and reverted the same night.** Worth keeping
  because the failure mode is a trap this project can fall into again. An offline probe of
  `ckpt_1979085_steps.zip` reported the pitch head commanding -5.7 deg/step at *every* camera pitch from
  -40 to +40 and sign agreement 44-46% (below chance), i.e. an open loop driving the camera into the floor
  of the band. Note the bias itself was NOT the culprit: decomposing the logits showed `b` contributed only
  +0.06 of that -3.75 deg/step and `W @ h` the rest, so v1's `pitch_reset` approach (zeroing the head) would
  have been a no-op here and would also have destroyed a genuinely good slope (corr +0.95). Instead a single
  scalar tilt was added to the pitch bias (`logit_i += lam * PITCH_BINS[i]`, lam 0.0469, Adam moments for
  those rows cleared), chosen so the command at elevation 0 was exactly zero. Offline this looked right:
  the loop gained a fixed point near -8 deg, inside the band, and sign agreement went 45% -> 55%.
  **Live it was clearly worse**: `look_up_mean` -0.010 -> +0.447, `pitch_mean` -0.62 -> +27.9,
  `on_target_frac` 0.039 -> 0.016, `pitch_track` 0.065 -> -0.134. Reverted to `ckpt_1979085_steps.zip`
  within ~10k steps; the bad checkpoint is kept as `REVERTED_pitch_centred_1979085.zip.bad`.
  **Why it was wrong:** the live camera was ALREADY level on average (`pitch_mean` -0.62) and enemies sit
  essentially ON the horizon on average (`enemy_elev_mean` +0.80), so the live mean pitch command was already
  ~0. The probe's synthetic states (one enemy, seven empty enemy slots, no walls, zero velocity, camera pitch
  0, full health) are not the live distribution, and the "systematic downward command" was an artifact of
  them. `enemy_pitch_err_mean` 26.97 versus `enemy_elev_abs_mean` 26.34 says the same thing from the other
  side: a camera welded level would score what the real one scores, so the pitch loop is useless but it was
  not biased, and there was nothing for a constant correction to fix.
  **Rules adopted:** never judge a policy edit from an offline probe alone -- confirm against live
  `status.json` metrics; after any change, revert if `on_target_frac` or `kills_per_min` is worse at the next
  two checks; prefer reward-weight changes to editing weights directly; change one thing at a time and give
  it at least 400k steps.
- **Update size at 1.98M.** `approx_kl` had been running 0.029-0.034 against `target_kl` 0.02 every iteration,
  so SB3 was breaking out of the epoch loop early and only 1-2 of the 5 epochs ever ran. Set
  `learning_rate` 3e-4 -> 2e-4 and `target_kl` 0.02 -> 0.03 (one change: both control update size).
- **Open question: `yaw_track` is 0.149 live but +0.846 in the offline probe** on the same checkpoint. Until
  that gap is explained, neither number should drive a decision. The leading suspect is that `visible` is
  line-of-sight only, so the nearest visible enemy is often BEHIND the player, where turning either way is
  equally correct and the sign comparison becomes a coin flip -- which would drag the live metric toward 0
  for a policy that is actually fine.
- **The `yaw_track` gap explained, and the probe's exact lie (2.07M).** Two independent audits closed both
  open questions from the reverted pitch experiment.
  - **Trust the live number.** The offline probe scored `sign(E[yaw])`, the mean action; the live metric
    scores the sign of the SAMPLED action (`env.py` reads `command["look"][0]` from `decode_action`). With
    entropy at 9.47 of 12.49 nats the head is soft, so the two differ enormously: at azimuth -10 deg the
    policy puts P(left) 0.748, so the mean is right but sampled agreement is only 0.502. Reproducing the
    probe's own states gives mean-action 0.85 and sampled 0.566 -- half the gap is sampling alone.
  - **The metric graded the wrong enemy.** It scored against `visible[0]`, but `pack_observation` feeds the
    policy the nearest 8 by distance REGARDLESS of visibility, so whenever the nearest enemy was occluded
    the score compared the look command against a farther enemy at an unrelated azimuth. Monte-Carlo at the
    observed `enemy_visible_frac` 0.669-0.748 reproduces yaw_track 0.121-0.150 and pitch_track 0.031 --
    i.e. the whole live value -- from a true score of about 0.24. Fixed in `env.py`: the two tracking scores
    now only count steps where the nearest enemy IS the visible one, and the deadzone is angular (5 deg)
    rather than `abs(x) > 0.5` metres, which silently meant 1.4 deg at 20 m but 4.8 deg at 6 m. The gate guards
    only those two counters, never `on_target_frac`/`dist`/`elev`. Scores before and after this fix are not
    comparable.
  - **The "enemies behind" hypothesis was wrong.** Enemies behind score better, not worse (`rel.x` is large
    there). Dragging 0.85 to 0.15 by coin flips would need 82% of steps behind; `enemy_yaw_angle_mean` 62.6
    implies about 26%. The aim reward IS paid on unshootable enemies behind the player, but it is only
    about 4 of 131 reward (3%).
  - **Why the synthetic probe invented a pitch bias.** All 8 ground rays were set to 1.0 while `grounded`
    was also 1.0 -- and the mod documents a near-max ground ray as "a pit", so every probe sample described
    a player standing on nothing over a 30 m drop, a state that never occurs. All 16 wall rays were 1.0 (no
    geometry within 50 m), velocity was zero, health full. Camera pitch and enemy camera-space elevation
    were also swept INDEPENDENTLY, though live they are rigidly coupled (camera-space elevation is roughly
    world elevation minus camera pitch), so the grid contained impossible joint states like an implied world
    elevation of +85 deg. Each marginal looked plausible; the joint was fiction. **The free falsification:**
    -5.7 deg/step against a +/-45 clamp pins the camera within 8 steps, so `pitch_abs_mean` would read ~45;
    it read 27.2. The probe was refuted by data already on disk before any weight was touched.
- **Pitch is not the binding constraint; yaw is.** `enemy_yaw_angle_mean` is 62.8 deg. Zeroing the pitch
  error entirely would move the 3-D angle only 64.7 -> 62.8, so the 15 deg cone still fails on yaw alone.
  The pitch loop scores 0.6 deg WORSE than a camera welded level (`enemy_pitch_err_mean` 26.97 vs
  `enemy_elev_abs_mean` 26.34) while burning 27.2 deg of travel, and `pitch_abs_mean` 27.2 exceeds the 22.5
  of a uniform sweep over the band -- that is a random walk piling up on the clamp, not a biased head. Mean
  elevation is +0.8 deg but mean-absolute is 26.3: the residual is zero-mean and conditional, so no constant
  could ever capture it and the bias tilt was structurally incapable of helping.
- **`ent_coef` 0.01 -> 0.004 at 2.08M.** Measured per head as entropy pull (`ent_coef * ln bins`) against the
  normalised advantage over the GAE horizon: yaw 0.0240 vs 0.0512 (2.1x) and pitch 0.0195 vs 0.0246 (1.26x).
  The signal wins, but far too narrowly for a precision task, which is why the mean action is right 86% of
  the time while the sampled one is right ~24%. A 3x margin needs `ent_coef` <= 0.0071 for yaw and <= 0.0042
  for pitch. Unlike v1's cut to 0.005, this is not being used to paper over a broken reward: the reward was
  rebalanced onto combat first.
- **`scripts/poll_status.py`** appends `status.json` to `runs/<run>/metrics_log.csv` every 30 s. `mean_100`
  is a deque that does NOT survive a restart and `history[]` keeps only reward/kills/wave, so every aiming
  diagnostic was being thrown away on each resume -- which is how a 5-episode window got mistaken for a
  trend. Gate every comparison on `window >= 50`.
- **Overnight run 2.12M -> 4.93M (2026-09-16), and the first clear win.** Ran 5.8 h unattended at ~175
  steps/s with no crashes. **kills/min 2.64 (1.70M baseline) -> 6.62 at the 4.40M peak**, which beats v1's
  all-time best of 5.0; reward 136 -> 217, deaths 1.91 -> 1.33, `wave` 3.7 -> 4.2, aim share 84% -> 16%.
  The reward rebalance worked: combat is now ~78% of the return and the agent is actually fighting.
- **But the aim hypothesis is FALSIFIED.** The stated falsifier was `on_target_frac` >= 0.09 by 2.48M. It
  ran 0.038 -> 0.031 across the whole night and sat at 0.029 by 4.93M, with `firing_on_target_frac` equally
  flat at ~0.028. The agent got much better at killing WITHOUT aiming better -- `enemy_dist_mean` fell
  25.3 -> 14.5, so it is winning by closing distance and spraying, not by pointing at things. Some yaw
  progress is real (`enemy_yaw_angle_mean` 63.8 -> 51.3, `yaw_track` 0.154 -> 0.214) but nowhere near the
  ~15 deg the cone needs. Neither pitch band setting has ever moved this number (15 deg capped it at 4%,
  45 deg leaves it at 3%), which matches the audit finding that **yaw, not pitch, is the binding
  constraint**: at 51 deg mean heading error the 3-D angle cannot reach the cone whatever pitch does.
- **`ent_coef` 0.004 overshot, and the run peaked then degraded.** Entropy fell 9.47 -> 2.51 and was still
  falling. After the 4.40M peak the policy went deterministic and got worse for half a million steps:
  by 4.93M kills/min 6.62 -> 5.36, reward 217 -> 166, deaths 1.33 -> 2.15, and the camera drifted
  `pitch_mean` -6.9 -> -24.6 with `pitch_abs_mean` 28 -> 34 and `pitch_track` dead at 0.003. Rolled back to
  `ckpt_4379085_steps.zip` (kept as `best_6.6kpm_4379085.zip`) and raised `ent_coef` to 0.006 to hold
  entropy near the 3.60 the peak ran at. On resume entropy came back to exactly 3.60 and kills/min to 6.00.
- **`scripts/keep_best.py`** now preserves the best checkpoint automatically (`best.zip` + `best.json`),
  scored on smoothed kills/min from `metrics_log.csv` with a full 100-episode window, and warns when the
  current policy falls more than 15% below it. PPO does not improve monotonically and `latest.zip` tracks
  the LAST policy, not the best one, so without this a peak is lost at the next checkpoint rotation --
  which is exactly what nearly happened overnight.
- **The 15-minute monitor did not fire overnight.** Training survived on its own, but nothing was tuning or
  watching it, so the post-peak degradation ran unchecked for ~500k steps. Cron jobs here are session-only
  and only fire while the session is idle; do not rely on them for unattended work. The durable substitute
  is the pair of always-on helper processes (`poll_status.py`, `keep_best.py`), which need no scheduler.
- **Second peak and second collapse (4.42M -> 6.53M), then PAUSED 2026-09-16.** After the rollback the run set
  a new best: **7.16 kills/min at 4.70M steps** (reward 231, deaths 1.55; `ckpt_4679085_steps.zip`), caught by
  `keep_best.py`. It then degraded exactly as before, to **3.79 kills/min, reward 125.7, entropy 2.94** by
  6.53M. So raising `ent_coef` 0.004 -> 0.006 did NOT stop the collapse; it only moved the peak. The pattern is
  now twice-observed and should be treated as the defining problem of this run: the policy improves to a
  peak around entropy ~3.5, then keeps sharpening past it and gets worse, while `on_target_frac` never moves
  off ~3%. Paused by request at 6,532,470 steps with games stopped and display settings restored.
  **Resume from `best.zip` (4,679,085 steps, 7.16 kills/min), not from the final weights**, which are roughly
  half as good. `latest.zip` has been refreshed to the same file so the old command is not a trap.
  Suggested order on resume: (1) confirm `best.zip` reproduces ~7 kills/min before changing anything;
  (2) stop the collapse -- either hold entropy with a higher `ent_coef` (0.008-0.01) or anneal the learning
  rate toward zero after the peak so the policy stops moving once it is good; (3) only then attack yaw, the
  binding constraint, with a zero-floor yaw shaping term.
- **Next steps (Cyber Grind, paused while the campaign runs):**
  - **Yaw is the one thing to fix.** `enemy_yaw_angle_mean` 51 deg is what keeps `on_target_frac` at 3%;
    pitch changes have been tried at 15 and 45 deg and moved nothing. Consider a yaw-only shaping term with
    a zero floor (`w * max(0, 1 - yaw_err/T)` with T near the current mean so improvement always pays),
    rather than another pitch pass.
  - Hold `ent_coef` so entropy stays near 3.5-4.0; below ~3 the policy went deterministic and regressed.
  - Watch the 1.70M rebalance against the falsifiers above; `yaw_track` then `on_target_frac` lead, kills/min follows.
  - If `on_target_frac` is still <= 0.06 at 2.18M, reset the look-head output bias (as `pitch_reset_2447005.zip` did for v1) rather than tuning weights again: a constant offset in the bias is not reachable from a reward slope that is flat across the whole sweep range.
  - `approx_kl` runs 0.029-0.031 against `target_kl` 0.02, so every update is being truncated. Worth a pass once the reward change has been judged, but not at the same time as it.
  - Raise `max_wave` as the agent improves.
- **Campaign: designed 2026-09-16, foundation built (mod v0.6.0).** Goal: finish all 35 main levels (`Level 0-1` to `Level 9-2`)
  as fast as possible, learning alone. Spec: `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md`.
  - **Decisions:**
    - No human demos or recorded routes: `routes.py` and `record_route.py` were deleted 2026-09-16 (git history keeps them).
    - Violent difficulty, all weapons unlocked in memory only.
    - Deaths respawn at the checkpoint inside the episode.
  - **Approach:**
    - The mod reads level structure: the real `FinalPit` exit, checkpoints, locked doors, arena clears, the
      **door-graph route (`campaign.gates`)** and the official timer. The NavMesh path hint is still reported but
      is no longer a reward: it is an enemy mesh and never resolved (see the NavMesh gotcha).
    - Rewards: the gate ladder and its approach shaping, time, milestones and per-cell novelty as a secondary explorer.
    - Episodes mostly respawn at each game's current checkpoint, so training concentrates on the frontier.
  - **Sub-projects:**
    1. Foundation + 0-1 pilot. Success bar: 50% fresh-start completion on Violent.
    2. The rest of the Prelude.
    3. Skull keys and switches.
    4. Acts 1-3.
    5. Speed and movement-tech actions.
  - **Starting weights:** Cyber Grind `best.zip`, widened to 479 inputs (`transfer_weights.py`), then carried
    across the look-mode action dimension by `add_look_mode.py` — see the `campaign_gates` entry below.
  - Cyber Grind training stays paused while the 5 games run the campaign.
  - `decompiled/` was regenerated on the second PC for this (ilspycmd 9.1.0.7988 in `.tools/`).
  - Spec amended during planning (2026-09-16): arena clears and door unlocks are position keys paid once per level
    load, `checkpoints_level` replaces `furthest_checkpoint`, room templates are filtered by `defaultRooms`
    ancestry, `level_seconds` is only reported for fresh starts, the campaign style reward is 0, and the mod blocks
    game-initiated restarts while the AI has control.
  - **Campaign progress tooling is in.** Episodes record `completed`, `fresh_start`, `level_seconds`,
    `checkpoints_level`, `cells_new` and `exit_dist_min`. `status.json` gains `best_checkpoints_level` and, once
    any episode reports `fresh_start`, a `campaign` block: `fresh_window`, `fresh_completion_rate` and
    `median_time_50` over the last 50 fresh starts, and `best_time` over all of them. Only fresh starts count
    because a checkpoint respawn starts partway through the level and the official timer carries over.
    `poll_status.py` logs these as columns (`part_time`, `part_checkpoint`, ... for the reward parts), and now
    keeps an existing `metrics_log.csv` header, dropping columns that header lacks: move an old log aside to get
    the new columns. `route_progress` is gone from `status.json`.
- **Campaign verified in game (2026-09-16), mod v0.5.0, one game at 30 fps / frameskip 2 with rendering off**
  (superseded by the v0.6.0 acceptance run at the end of this section, which repeats all of it plus check 6):
  - Level 0-1: 1 level load PASS | 2 arsenal SKIP | 3 checkpoint PASS | 4 death respawn PASS | 5 exit FAIL (exit
    room switched off at load; see the gotcha above).
  - Level 1-1: 1 level load PASS | 2 arsenal PASS | 3 checkpoint PASS | 4 death respawn PASS | 5 exit FAIL (exit
    room switched off at load; see the gotcha above).
  - Throughput smoke run with 5 games (`train.py --config configs/campaign_0-1.yaml --timesteps 20000
    --run-name campaign_smoke`, outputs deleted): finished cleanly at 177 steps/s; 13 episodes ended (stuck 13).
  - Next: the 0-1 pilot run (`campaign_ppo`).
- **`campaign.path` exit-sample fix, verified in game (2026-09-16).** `CampaignObserver.UpdatePath` sampled
  the NavMesh at `exit.transform.position` with a 20 m radius, but a `FinalPit`'s transform sits inside the
  drop it triggers, not on walkable ground, so that sample always missed and `path.status` read `none`
  everywhere on both 0-1 and 1-1 (matching the read above). Logging every `NavMesh.SamplePosition` candidate
  in game showed the raw depth figures (17.1 m below the floor on 0-1, 76.1 m on 1-1) do not tell you where
  to look: the true nearest NavMesh point was 47.5 m away **straight down** from the pit on 0-1, and 84.6 m
  away **up and 46.8 m to the side** on 1-1 -- not "a few metres directly above" as first assumed, so an
  initial fix that only probed upward (heights 18/30/50/80 at a tight 6 m radius) found nothing on either
  level. Replaced with `CampaignObserver.SampleExit`: keep the unchanged 20 m try at the pit's own position,
  then escalate the search radius itself (55 m, then 95 m) at that same position, letting NavMesh's own
  nearest-point search find whichever direction actually holds mesh. Both measured points connect back to the
  player's side only as `NavMeshPathStatus.PathPartial`, which `BuildPath` already reports as `"partial"` --
  expected, since a `FinalPit` deliberately sits off the walkable graph. `length`/`next_corner` are still
  measured to the snapped point (distance left to walk to standable ground near the exit), never a straight
  line into the pit. Verified live with one game (`games.py launch --count 1 --monitor 1`,
  `campaign_check.py`'s teleport-to-checkpoint approach against the level's farthest pending checkpoint):
  - Level 0-1: spawn `path.status` stays `none` (unrelated, pre-existing: the player-side 6 m sample already
    fails on the player's own spawn position, left untouched per scope). After teleporting to and activating
    the farthest checkpoint (`157,28,640`, 319.9 m from spawn): `partial`, length 72.4 m, a real `next_corner`.
  - Level 1-1: `partial` from spawn already (length 16.6 m) and after activating the farthest checkpoint
    (`46,12,550`, 300.7 m from spawn): `partial`, length 16.8 m, `next_corner` updated toward it. Before the
    fix both levels read `none` in both cases (confirmed by rebuilding and re-running against the pre-fix code
    first).
  - Noted, not changed (out of scope): the player-side NavMesh sample failing at 0-1's own spawn point.

- **Campaign pilot `campaign_ppo` (0-1, Violent), started 2026-09-16.**
  - Start: `models/campaign_ppo/transfer_init.zip`, the Cyber Grind v2 `best.zip` (7.16 kills/min, `ckpt_4679085_steps`) widened to 479 inputs by `transfer_weights.py --action-scale 0.5`. First-update entropy 6.93 nats against a 6-10 target; 199 steps/s with 5 games.
  - Stop with Ctrl+C in the training console and wait for `train.py` to exit (it prints `Saved models\campaign_ppo\latest.zip` last; `runs/campaign_ppo_train.log` is appended across restarts, so an older such line proves nothing), then resume with `python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/latest.zip`. After a hard stop resume from the newest `ckpt_*_steps.zip`, not `best.zip`: `keep_best.py --metric campaign` only replaces `best.zip` when the smoothed fresh completion rate beats its record, so until levels are being completed it holds the first checkpoint scored (rate 0) and resuming from it would throw the run away. Use `best.zip` to roll back once completions have peaked and fallen (keep_best warns). Helpers: `poll_status.py --run campaign_ppo`, `keep_best.py --run campaign_ppo --metric campaign`, `dashboard.py --run campaign_ppo --monitor 1`.
  - Ctrl+C leaves the exploration archives slightly stale: SB3's subprocess workers exit on KeyboardInterrupt without calling `env.close()`, so each `explore_*.npz` holds its last 20-episode save.
  - Checks: `best_checkpoints_level` >= 1 by 250k steps; >= 2 by 1M steps, else a 1M-step fresh-weights comparison run (`campaign_ppo_fresh`); eval gate once `campaign.fresh_completion_rate` >= 0.5 with `fresh_window` >= 20.
  - 1M-step falsifier passed: `best_checkpoints_level` 4, checkpoints per level load 0.13 -> 2.19 between 250k and
    1M steps, kills/episode 0 -> 15.1, deaths 0.03 -> 0.50, entropy 9.12. The transferred weights reach well past
    the first checkpoint, so no fresh-weights comparison run was needed.
  - **Watch novelty's share.** At 1M the parts are `novelty` 539.6, `time` -62.3, `kill` 9.3, `damage_dealt` 9.3,
    `checkpoint` 6.2, `arena_clear` 3.3, `door_unlock` 0.9 of 501 total: exploration is about 90% of the gross
    positive reward, the same one-term dominance that broke the Cyber Grind run's aim shaping. It is not yet
    decaying (new cells/episode 425 -> 1492) because 0-1 is big and the agent keeps finding real ground, and
    progress is genuinely improving, so it stays for now. Tripwire: if the fresh completion rate is still 0 at
    3M steps while `novelty` is above 70% of the gross positive reward, cut `novelty` 0.5 -> 0.15 so the
    milestones and the time cost drive instead, and give it 400k steps before judging.
  - **STOPPED at 2.49M steps, 2026-09-16, and the reward was wrong. The agent was farming the void.** Zero
    completions in 397 episodes; `best_checkpoints_level` frozen at 5 since 1.36M; checkpoints per level load
    peaked 2.54 at 1.33M and fell to 1.73; `exit_dist_min` 157 m -> 188 m. The cause, found by a 9-agent
    adversarial review and then confirmed straight from the five committed `explore_*.npz` archives:
    **`ExplorationArchive` was keyed on the raw 3-D player position with no bound on y**, so novelty paid for
    occupying *volume*, not for covering *ground* -- and the cheapest volume in a 3-D game is vertical.
    - Measured over all 364,984 novelty units the run ever paid: **47.1% below y = -60 m** (the void under the
      map, reaching **y = -57,304 m**), of whose cells **99.98% were entered exactly once**; another **47.7%**
      for air cells stacked above ground already paid for (the playable footprint was 1,397 XZ columns but
      15,251 cells, ~11 stacked per column). **Only 5.3% was ever paid for actual ground covered.**
    - Why it was stable: a fall enters a brand-new 4 m cell every decision at the full `1/sqrt(0+1)` = 1.0, the
      supply never runs out, those cells are never revisited so the decay never bites, there is no kill plane,
      and any `novelty > 0` reset the stuck clock -- so diving off the level was an unbounded income stream that
      could not be truncated. Finishing pays +100 once and *terminates*, forfeiting the stream. Under that
      reward, completing the level was never the argmax, and the policy was correct to refuse.
    - Do not mistake this for the shape of the level: `corr(fresh_start, checkpoints_level) = -0.975`, so the
      1.33M "peak by depth" is an artifact of respawn episodes inheriting a checkpoint count, not a better policy.
      A fresh level load essentially never reached even the first checkpoint
      (`checkpoints_level = 3.748 - 4.434*fresh_start`, ~0 at `fresh_start = 1`).
    - **Fixed** by keying novelty on the ground point under the player (`env._ground_point`), which closes the
      void faucet and the air slab with one bounded mechanism while still paying for new floor covered while
      airborne -- which matters, because fast movement in this game is airborne. `oob_frac` was added so the
      mechanism is directly observable. Also fixed: `level_complete` graded and paid before death/scene-change,
      so the first real completion cannot be booked as a death.
    - **Rejected during review, do not retry without new evidence:** a straight-line distance-to-exit reward
      (measured `exit_dist_min = 224.9 - 90.5*fresh_start`, i.e. the checkpoints sit ~90 m *farther* from the pit
      than spawn does, so it would pay the agent to reverse, and it is farmable by a void glide); a 2-D cell key
      (breaks the saved 3-tuple archives and deletes the vertical structure of a vertical level); cutting
      `novelty` 0.5 -> 0.15 as well (the measure change already cuts the income ~5-10x; doing both removes the
      only forward pressure the run has -- hold it in reserve); cutting `max_steps`; and touching `gamma`,
      `ent_coef` or `pitch_limit_deg` (entropy is 9.7 nats and rising, `approx_kl` 0.021 vs `target_kl` 0.03, and
      pitch is causally irrelevant to navigation -- `NewMovement` builds `inputDir` from horizontal projections).
      - **`pitch_limit_deg` was reinstated at 45 on 2026-09-16 and the "pitch is causally irrelevant" reason was
        wrong as stated.** Movement is indeed horizontal-only, but *shooting* is not, and the campaign config set
        no band at all, so the camera drifted to a mean of **-78 deg** with `on_target_frac` 1.5% — and all three
        live episodes that got past checkpoint 3 then died in the Stray arena between two locked doors with 0-1
        kills in 35-193 s. Four of 0-1's gate doors are held behind kill gates, so the agent has to be able to
        aim to walk the route at all. Pitch is irrelevant to *where you go* and decisive for *whether you get
        through the door*.
    - Keep the five `explore_*.npz` archives: their floor counts carry the `1/sqrt(N)` decay history that is the
      frontier-seeking engine, and a standing player's cell and its ground-point cell are the same cell. The ~347k
      air and void keys just become dead weight. They were copied into `models/campaign_ppo_ground/` for the
      restart, because `explore_dir` follows `run_name`.
- **Campaign run `campaign_ppo_ground` (0-1, Violent), started 2026-09-16** from
  `models/campaign_ppo/ckpt_2450000_steps.zip` -- the pre-fix run's own weights, since the reward was wrong, not
  the policy. New run name because `part_novelty` and `cells_new` change meaning across the fix and
  `poll_status.py` keeps an existing CSV header (it would silently drop the new `oob_frac` column).
  - Baseline at 28 episodes / 2.59M steps, against the pre-fix run's last window: `novelty` 395.6 -> **75.1**
    (93.9% -> **76.8%** of gross positive), `time` -59.1 -> **-94.7** and now the largest single term,
    `checkpoint` 5.10 -> **6.79** (1.2% -> 6.9%), `punch` **-17.13** (still pressing on ~35% of steps, now priced),
    total reward +501 -> **-16.9**, `oob_frac` **0.090**, episode length 5950 -> 4836, deaths 1.0 -> 0.19,
    entropy 10.6 and rising. A negative total is intended: wandering no longer pays for itself, and SB3 bootstraps
    `gamma * V(s_T)` on both `stuck` and `max_steps`, so the agent cannot escape the stream by ending the episode.
    `level_complete` is the only termination that stops the clock, and it also pays +100.
  - **Gates, in order. Judge on none of them early, and on no gate satisfied by the accounting change alone:**
    `novelty` and `cells_new` fall by construction and prove nothing on their own.
    1. **+400k steps (~2.99M): `oob_frac` well below the 0.090 baseline**, on a 100-episode window. Mechanism check
       only. If it does not fall, the ground-ray sentinel is not being read as intended -- debug it, do not tune
       weights. Some floor is legitimate: 0-1 has real drops, so do not expect zero.
    2. **+1.0M steps (~3.59M): `fresh_start` <= 0.32 and `best_checkpoints_level` >= 6.** The real gate.
       `choose_fresh_start` forces a reload whenever the previous episode ended with no current checkpoint, so the
       measured `fresh_start` rate is a direct readout of how often a fresh load fails to reach any checkpoint at
       all: `f_measured = f_forced + (1 - f_forced) * 0.2`. The pre-fix run ended at 0.49, i.e. 36% forced reloads,
       up from 10% at 1.13M.
    3. **+2.0M steps (~4.59M): `campaign.fresh_completion_rate` > 0** with `fresh_window` >= 20.
    Gate every comparison on `window >= 50`.
  - **BLOCKER CLEARED 2026-09-16: a completion does register through the bridge, confirmed on a human
    playthrough of 0-1.** Watched read-only with `scripts/watch_completion.py` while the user played:
    `level_started` false -> true at t 0.23 s; **`exit.active` false -> true at t 122.2 s**, after all 6
    checkpoints; `level_over` true at **t 146.582 s** with 59 kills. So the exit room *is* switched on by real
    play, `stats.level_complete || campaign.level_over` fires in the pit, and the whole path `env.py` grades a
    completion on works end to end. `campaign_check.py` check 5's FAIL was an artifact of the check itself
    (teleporting onto a collider in a room that is still switched off), not a broken chain.
    - **Human reference time for 0-1: 2:26.58 (146.58 s), 59 kills, 6 checkpoints.** That is the bar.
    - The run reported `cleared_arenas` 0 and `unlocked_doors` 0, which is expected and not a discrepancy:
      `CampaignPatches.RecordArenaClear` and `RecordDoorUnlock` are both gated on `EpisodeController.InControl`,
      so they only record while the AI is driving. The training run's `arena_clear` / `door_unlock` payments are real.
    - **Two automated attempts could not answer this, and the reasons are worth keeping.** (1) *Teleporting cannot
      play the level.* Hops of 5 m walked the player through 0-1's entire geometry to the pit while `enemies`,
      `locked_doors` and `cleared_arenas` all stayed empty and the exit stayed inactive: rooms ahead of the player
      are switched off, so their trigger volumes are off too, and a teleport into dead space fires nothing. The
      player then fell to y -507 and kept going, independently confirming there is no kill plane under this level.
      (2) *Scripted movement cannot leave the starting room.* Dropping to 1.2 m hops jammed the player against a
      closed door at z 347 that the 5 m hops had been tunnelling through, and a scripted walk-and-shoot spent
      63,000 decisions inside a 6 m box with 0 kills. A probe at spawn shows why: `level_started` false,
      `slot_counts` all zero, and walls 3.5-6.6 m away in all 16 ray directions.
      - **Correction (2026-09-16): "0-1 spawns you sealed in" is wrong as stated.** The starting room is sealed
        to a *teleport* and to a *walk*, not to a **slide**: the lower `Breakable` plank leaves a 1.5 m gap
        against a 1.25 m slide height, and a scripted grounded slide goes through it every time. Measured at
        integration: forward + slide from spawn reached the gun-room door at z 401 in 42 decisions, and the
        sampled policy passes the vent on 67-93% of fresh loads. The walk-and-shoot script failed because it
        never slid, not because there is no way out. See the alternate-start gotcha.
  - **Was open, now answered by the above: nobody had ever seen a completion register through the
    bridge.** `part_level_complete` is blank in all 544 rows of the pre-fix run and `best_runs/` is empty. The code
    path reads right -- `FinalPit.OnTriggerEnter` sets `nmov.levelOver`, `ObservationBuilder.cs:274` reads
    `sm.infoSent || nm.levelOver` (so detection does not depend on the campaign `exit` block), and `Door.Open()`
    (`decompiled/Door.cs:432`, loop 458-465) is what calls `SetActive(true)` on `activatedRooms`, which is why
    check 5's teleport-onto-an-inactive-collider failure is expected rather than evidence of a broken chain. But it
    is unverified. To settle it: one game, `python scripts/bridge_test.py --campaign` (it polls without taking
    control), then play 0-1 to the end by hand and watch `exit.active` flip true near the final room and
    `level_over` go true in the pit. If `level_over` never fires, no reward change can move the completion rate and
    the next work is mod-side.
  - `exit_dist_min` went 157.9 m -> 193.6 m over the same window. That is straight-line distance, and 0-1's route
    does not run straight at the pit, so it is not yet evidence of anything; it matters only if it keeps rising
    while checkpoints stop.

- **`campaign_ppo_ground` STOPPED at 2,914,785 steps, 2026-09-16, and diagnosed. Zero completions in 2.9M steps
  across two runs.** A five-investigator diagnosis, each report adversarially verified, found four causes. Spec:
  `docs/superpowers/specs/2026-09-16-campaign-gates-unwedge-design.md` (mod v0.6.0). Findings:
  - **F1 the wedge.** ~70% of sampled fresh loads (8/26 passed at 30/2, 3/15 at 60/4) ended in the absorbing
    airborne `slowMode` state. See the wedge gotcha. This alone was most of the run.
  - **F2 no route signal.** The NavMesh is an enemy mesh and `path` never paid. See the NavMesh gotcha.
  - **F3 combat gates the doors and the agent cannot aim.** The campaign config set no `pitch_limit_deg`, the
    camera sat at a mean of -78 deg, `on_target_frac` 1.5%, and all three live episodes that passed checkpoint 3
    died in the Stray arena between locked doors with 0-1 kills in 35-193 s.
  - **F4 smaller things.** The stuck clock never reset on kills and `_respawn` did not reset it; the ground run
    **saved no exploration archive at all** (episodes are ~3900 decisions, no worker reached 20 of them, and
    `SubprocVecEnv` workers never call `close()` on Ctrl+C); no per-episode log; `_ground_point` took the minimum
    of an 8-ray ring whose spread among hits is p90 **10.8 m**; `ChooseExit` could pick a secret-level pit on 0-2
    and 1-1; the trainer's OpenMP threads spin-waited on **11.0 cores** (`KMP_BLOCKTIME=1` -> 1.43, same update
    time); `Distribution.set_default_validate_args(False)` saves 0.95 ms of a 3.1 ms forward pass; and the resume
    path passed no `verbose`, which is why `campaign_ppo_train.log` was 0 bytes.
  - **One premise of the plan was refuted and is recorded rather than claimed away.** Observation inputs 448-455
    were said to have "read zero/none for the whole run". They had not: `path.status` was `partial` on **82.19%**
    of 32,022 logged decisions with a `next_corner` on every one, and `||W[:, 448:456]||_F` is 2.04 of 33.78. So
    re-using those slots displaces a working policy, which is why `add_look_mode.py` zeroes them **and folds
    their mean into the bias**, and why the new run's gates are read against a policy that moved before step 0.
    (Only column 453, `status == "complete"`, was genuinely dead — exactly zero norm in both stacks.)

- **Campaign run `campaign_gates` (0-1, Violent), integrated and ready 2026-09-17; NOT started.** Starting weights
  `models/campaign_gates/look_init.zip` = `campaign_ppo_ground/latest.zip` (2,914,785 steps) through
  `add_look_mode.py`. Measured on the 436 recorded 0-1 states: MU reproduced the spec's values exactly, mean KL
  **0.0232** (max 0.2079), greedy action changed **27.98%**, mean |dV| 0.627 over a value range -6.61..+64.05,
  optimizer carried. Zero-only, for comparison: 0.0388 / 46.33% / 1.130. So the fold roughly halves the
  displacement and the result is inside the run's `target_kl` of 0.03; the zero-only variant is outside it.
  - **Reward weights now:** `gate` 15.0, `gate_approach` 0.15, `door_unlock` 3 -> **15.0**, `novelty` 0.5 -> **0.2**,
    `path` 0.1 -> **0.0**, `pitch_limit_deg` absent -> **45.0**. Unchanged: `time` 0.02, `level_complete` 100,
    `checkpoint` 10, `arena_clear` 10, `kill` 0.5, `damage_dealt` 0.5, `damage_taken` 0.01, `death` 5, `style` 0,
    `punch` 0.01. Arithmetic for a route-walker at human pace: route (`gate` 150 + `gate_approach` 108 +
    `door_unlock` 60) = **+318**, the largest positive group, against `level_complete` +100, checkpoints +60,
    combat +60, novelty +40, time -44 — total ~+556. A wanderer that never leaves the start area scores **-101**.
    `door_unlock` moved to 15 because the route gradient is exactly **flat** at every arena-held door, which is
    where F3 says the run actually fails; on 0-1 the four arena-held doors are exactly the four gate doors.
    **Read the wanderer case honestly:** novelty is still 87% of gross positive income in that regime (69 of 79).
    The design makes the regime unprofitable, it does not make it un-dominated.
  - **Lead rulings.** R1: `GateProgress.best_dist` is **episode** scoped (cleared in `reset_episode`, *not* by an
    in-episode death respawn); the `gate` ladder stays level-load scoped. PPO returns are computed within an
    episode and a truncation bootstraps `gamma * V(s_T)` on the same state, so ending an episode can never be
    profitable, and most episodes are checkpoint respawns that would otherwise get no dense signal on ground an
    earlier episode covered. Rollback is one line. R2: `ObsLayout` stays **479** with the slot re-use and the
    mandatory zero + mean-fold (the 487-input variant is a drop-in alternative, flagged and not taken). R3: no
    `punch` change before `level_started`, no more than 5 games, no title-wait skip. R4: the run is
    **`campaign_gates`** and the weights `look_init.zip`, so `explore_dir` and the copied archives agree.
  - **Acceptance, measured at integration (one game, 30 fps / frameskip 2, rendering off, monitor 1):**
    - A1 gates on 0-1, fresh load **and** after a checkpoint respawn at `40,-12,485`: **PASS** — 11 gates,
      hops 0..9, every `key`, `pos` and `controller_active` identical to the spec's literal example both times,
      `gates_truncated` false, and `controller_active` false only on the hops 9 gun-room gate.
    - A2 1-1 ordered (13 gates, 0..5) and 0-5 unordered (3 gates, all `hops` null, no exception): **PASS**.
    - A3 `ChooseExit`: **PASS** — 0-2 picks (-199, -86.1, 277) and 1-1 picks (81, -76.1, 91), the non-secret pits.
    - A4 `campaign_check.py` on 0-1 and 1-1: **PASS** — every earlier verdict unchanged and the new check 6 PASS
      on both, fresh and after a respawn. (Check 5 remains the known standing exit FAIL.)
    - A5/A6 un-wedge, both flavours: **PASS** — each reproduced with `unwedge:false`, absorbing over 45 decisions
      of move+jump+dash (max axis movement 1.407 m and 1.419 m = exactly the 0.47 m/s creep), then **recovered on
      decision 6 = 0.400 s of game time** with no input at all, `slow_mode` and `heavy_fall` false and free fall
      resumed. Total position shift across the whole recovery 0.101 m and 0.122 m — gravity resuming over two
      frames, not a teleport (`GroundCheck.Bounce` would move the player metres). No enemy lost health, though
      the vent has no enemies in it, so that half is vacuous in this sample.
    - A7 slide after recovery: **PASS** — 15.04 m and 14.81 m in 10 decisions, against a 5 m bar.
    - A8 no regression elsewhere: **PASS** (mod side) — 27 of 28 probe tags within 0.5 m of the v0.5.0 baseline,
      and the legitimate grounded low-ceiling crouch still reports `slow_mode: true` on the same 230 steps.
    - A10 gate reward live: **PASS** — a scripted grounded slide through the vent reached the hops 9 gun-room door
      in 42 decisions and paid exactly one `gate` (15.0) and `gate_approach` **9.42** (0.15 x 64.8 m less the
      0.5 m deadzone), `gates_reached` 1, and the target advanced `40,1,408` -> `40,1,470`.
    - A11 look mode 2 live: **PASS** — from **185.6 deg off** (camera deliberately turned 180 deg away and pitched
      to the -45 band edge) and with a hostile +90 yaw / -20 pitch free-look bin underneath it every decision,
      mode 2 reached 6.33 deg on decision 2 and 0.00 deg on decision 3. The two steps are exactly the 90 deg and
      20 deg per-decision caps. On a fresh spawn the target slots read
      `[0.006, 0.030, 1.296, 0.648, 1.0, 0.0, 0.0, 0.45]` — rel z 64.8/50, dist 64.8/100, mask, and hops 9/20.
    - Look mode 1 live: **PASS** — camera-space yaw +26.84 -> -5.12 and pitch -14.51 -> -0.12 in one decision,
      both axes the right sign, converging over the next four.
    - A12 throughput: **PASS** — **184-185 steps/s** with 5 games on 0-1, against a 150 bar (152 before).
    - A9 vent pass rate, 15 fresh loads x 1500 decisions: **PARTIAL, 10/15 = 67%** against an 80% bar — see below.
  - **A9 is the one number below its bar, and the cause is measured, not guessed.** The same widened model, same
    build, same archive and same budget scores **14/15 = 93% with the look-mode dimension pinned to 0**, against
    10/15 = 67% with it sampled. So the drop is the look modes commandeering the camera, **not** the weight
    surgery — the widened weights on free look reproduce the mod side's 93% exactly. The smoke run confirms the
    mechanism: `look_free_frac` 0.652, `look_gate_frac` 0.348, `look_enemy_frac` 0.000 (mode 1 falls back on every
    step of a spawn room with no enemies), so ~35% of decisions have the camera taken over, and that alone costs
    26 points of pass rate. Movement is camera-relative, so overriding the look head also overrides whatever
    movement reflex 2.9M steps built around it. The action head starts uniform (three zero logit rows), so this is
    expected to be transient — learning which mode to select is goal G4. Baseline for comparison: **31%** before
    the un-wedge fix, 93-100% with it and the old policy.
    Spec 11.2's pre-committed outcome for a 31-80% landing applies: the mod fix ships, the Python wedge detector
    stays as the bound, and `slide_min_hold` is the next experiment with A9 re-measured with it on.
  - **The un-wedge fix itself is working perfectly.** `wedged_steps` — run gate 1's own metric, against a baseline
    of **877.7 per episode** — read **0 on all 15 A9 episodes** and 0 on all 3 smoke episodes. The 5 A9 failures
    are not wedges: 4 are the policy climbing to y 13-21 in the spawn room and never entering the vent (the mod
    side saw the same failure mode at y ~116), and 1 entered the tunnel at decision 1489 and ran out of the 1500
    budget 11 decisions later.
  - **Smoke run, 5 games, 20,450 new steps, finished cleanly:** 184-185 steps/s, `approx_kl` 0.020-0.024 against
    `target_kl` 0.03, `explained_variance` 0.93, `entropy_look_mode` 1.06 (ln 3 = 1.0986, i.e. still near-uniform
    and starting to move), `entropy_yaw` 1.86, `entropy_pitch` 1.44. `episodes.jsonl` written with every spec 9.7
    field; `gate_approach` paid live; `status.json` carries `best_gates_reached`, `best_gate_hops`,
    `mean_fresh_100` and the three per-dimension entropies; `mean_gates_reached_100` is in `history`; the
    dashboard renders. Outputs deleted.
  - **Run gates, in order** (spec 11.3; judge on none of them early, `window >= 50` on every comparison):
    1. **+200k:** `wedged_steps` mean < 20 per episode (baseline 877.7) and `end_reason == "wedged"` under 5% of
       ends. Mechanism check only — if it fails, debug the mod, do not tune weights.
    2. **+500k:** `gates_reached` mean >= 3 **on fresh starts** and `fresh_start` <= 0.32. **This is a combat
       gate, not a navigation gate:** on 0-1 `gates_reached >= 1` needs 11 kills in the gun room and `>= 3` also
       needs the projectile arena, against F3's measured 0-1 kills per episode. A failure here does not mean the
       route signal is broken — read it with the `kill` and `door_unlock` parts.
    3. **+1.5M:** `best_gate_hops` <= 2 on fresh starts.
    4. **+2.5M:** `campaign.fresh_completion_rate > 0` with `fresh_window >= 20`.
    - **Novelty tripwire:** if at +1.5M `gates_reached` is still <= 1 on fresh starts **and** `part_novelty` is
      above 70% of gross positive reward, cut `novelty` 0.2 -> 0.1 and give it 400k steps.
    - **Falsifier for the whole design:** if `gates_reached` is still <= 1 at +1.5M while `wedged_steps` is near 0
      and `part_gate_approach` is being paid, the route signal is not the binding constraint and the next suspect
      is combat (F3), not more shaping.
  - **`fresh_start` as a gate is confounded and must be read as a ratio, not as a rate.**
    `choose_fresh_start` forces a reload whenever the previous episode ended with no current checkpoint, so the
    measured rate is `f_measured = f_forced + (1 - f_forced) * 0.2` — it is a readout of how often a fresh load
    fails to reach any checkpoint, not an independent setting. The same inheritance confounds `gates_reached` and
    `checkpoints_level`, which a respawn episode inherits from its level load: the pre-fix run's
    `corr(fresh_start, checkpoints_level) = -0.975` is what produced the false "peak by depth" at 1.33M. Chart
    and judge both on **fresh starts only** — that is what `mean_fresh_100` and `best_gates_reached` are for.
  - Known open, carried into the run: levels beyond 1-1 are unverified for gates (only 0-1, 0-2, 0-5 and 1-1 were
    checked live); `controller_active` is false for a door with no `DoorController` at all, which reads as "a
    fight gates it" but never fires on 0-1..1-1; and the spawn-room climb that costs ~27% of fresh loads is
    unrelated to the wedge and still unaddressed.

- **Branch `next-levels` (mod v0.7.0) — MERGED into `main` and verified in the game, 2026-09-17** (merge commit
  on top of `c20e97d`, which is the rollback target; see the integration entry below for the readouts). It was
  built in worktree `F:\Github\ULTRAKILL-AI-next` off `7da1204`. Four pieces from
  `docs/superpowers/specs/2026-09-17-multi-level-and-skull-gates-design.md`, built while the `campaign_gates` 0-1
  run stayed live and untouched:
  - **S1 multi-level curriculum.** Optional ordered `env.levels`; a level changes only on a **fresh** load and is
    sampled from the unlocked set with weight `max(floor, 1 - that level's fresh completion rate)`, floor 0.1.
    Level k+1 unlocks when level k's fresh completion rate over its last 20+ fresh episodes reaches
    `unlock_rate` (0.5); `levels[0]` is always unlocked and the unlock is a latch that survives `_restore`. The
    workers are `SubprocVecEnv` subprocesses, so the only shared state is `runs/<run>/curriculum.json`, written
    atomically by `ProgressCallback` and re-read at each fresh start; missing, torn, wrong-run or wrong-order
    falls back to `levels[0]` with one warning per worker. **No level id enters the observation** — generalisation
    by design. Per-level exploration archives, `best_runs/`, `episodes.jsonl` rows, `status.json`'s
    `campaign.levels` table and the dashboard's `levels` block all key on the level name.
  - **S2 gates usability guard** (Python only): a ladder whose phase-1 gates carry a `hops` value on fewer than
    half of them is ignored entirely and `GateProgress` falls back to the exit vector. 7-2 reports
    `gates_ordered: true` with 1 of 4 and 8-3 with 1 of 32, where the old code locked onto that single ordered
    door — 892 m from the start on 8-3 — and never retargeted.
  - **S3 6-2 exit tie-break** (mod): `Scan` discards a `FinalPit` targeting `Level P-` (a Prime Sanctum) as it
    already did `-S`, `ChooseExit` counts an `Intermission*` target as leading onward (an act finale's real exit
    drops into an intermission, and `Intermission2` does not parse as `Level a-b`), and a rank tie is logged once
    per level load. Before this, 6-2's three candidates ranked equally and the exit was whichever
    `FindObjectsOfType` returned first.
  - **S4 skull-carry gates** (mod + Python): `campaign.altars[]`, `campaign.items[]` and `gates[].needs_item`;
    one shared door-key table so an altar's door key string-matches a gate key by construction; a phase-2 gate
    pass appending altar-driven one-room doors as `altar_only` gates (adding no node and no edge, so no existing
    `hops` moves); a `GateProgress` sub-goal machine walking fetch -> carry -> open through observation slots
    448-455; two once-per-level-load milestones `item_pickup` / `item_placed`; and env-side carry protection.
  - **The action space did not have to change, and this was the one thing S4 could have broken.**
    `Punch.AltHit` does both pick-up (`ForceHold`) and place (`PlaceHeldObject`) off the existing `punch` button,
    which `ActionInjector` already binds to the game's own `input.Punch`. So no new button, **no weight surgery
    and no `add_look_mode.py`-style migration**: `models/campaign_gates/best.zip` loads into the new env as is.
  - **Verified:** `dotnet build -c Release -p:InstallPlugin=false` at 0 warnings / 0 errors, and all twelve
    no-game test files pass (248 named tests). Degradation is tested in both directions — new Python reads a
    missing `altars`/`items`/`needs_item` as "no altars", an old Python ignores the new fields.
  - **Nothing was verified in the game while the branch was written** (no game launched, no port opened). The
    merge-and-verify checklist `docs/superpowers/plans/2026-09-17-next-levels-integration.md` was run at the
    2026-09-17 pause; its results are the integration entry below. S1+S2+S3 are verified live; **S4 is not**, and
    its two weights stay at 0.0.
  - **The S4 tooling blocker is cleared: `scripts/skull_check.py` exists** (2026-09-17). It was the one gap — the
    spec's in-game checks 1-3 need an AI-driven punch at 1-1's red pedestal **with rendering off**, which a human
    playthrough cannot test (`render: false` only applies while the AI has control) and which no existing script
    could do (`walk_to_exit.py`'s waypoints are the level's checkpoints and it never punches; `campaign_check.py`
    teleports and no-ops; `bridge_test.py --campaign` is read-only). It walks rather than teleports, reads
    `items[].active` before punching so a switched-off room is never mistaken for a broken `ActiveStart`, and is
    green against `FakeLevel`'s skull room in `tests/test_skull_check.py`, so the pause only has to run it. **It
    has still never been pointed at the game**, and check 1 remains the kill switch: if the fist Animator is
    culled under `render: false` **with** `TrainingSpeed.ForcePlayerAnimators` applied, S4 is inert under training
    settings and nothing else in it matters.
  - **Known, deliberate deviation from the brief:** `keep_best --metric campaign` was asked to score the **mean**
    fresh completion rate over unlocked levels. It scores a **shrunk sum** instead, and `keep_best.py` is not
    edited at all. A mean drops at every unlock (~0.52 -> ~0.26 when the second level joins), and `keep_best.py`
    only replaces `best.zip` on a strict improvement — so a mean would freeze `best.zip` on a single-level policy
    and print the "15% below best" warning forever. A sum cannot drop, since a new level contributes 0 and grows;
    with one unlocked level at `fresh_window >= 20` it is exactly the rate that field has always been. The
    per-level means are all in `status.json`'s `campaign.levels`.
  - **Fixed 2026-09-17: `_protect_carry` was gating on "anything held".** It dropped `punch` on any step where any
    `items[]` entry read `held: true`, so on 0-4 — in the prelude curriculum, shipping a `CustomKey1` and **zero**
    `ItemPlaceZone`s — picking the key up cost the agent the button for the rest of the carry, including the throw
    that would have given it back (throwing is itself a punch). It now gates on a **carry**: a held item some live
    unfilled altar accepts, through `campaign.wanting_altars`, the same filter `GateProgress._subgoal` chooses a
    destination with. Both harms the rule exists for need a destination, so where nothing accepts the item there is
    nothing to protect. Pinned by `test_punch_passes_through_while_carrying_an_item_no_altar_wants` (0-4's shape)
    and `..._the_altars_do_not_accept` (the type filter, which is what 1-1's two skull colours need); the wired
    case still gates, unchanged. **The forced look mode 2 was checked and is not over-broad**: it keys on the
    target carrying a `subgoal` field, which only `_subgoal` sets and only for a gate with `needs_item` and a live
    wired altar, so it is already inert on 0-4 — asserted in the same test.
  - Also carried: S3 fixes 6-2 only. 3-2 keeps two `Intermission1` pits at an identical position and 2-4 two
    `Level 3-1` pits 1.4 m apart, and 8-4's `EarlyAccessEnd` pit parses as neither a level nor an intermission and
    is still picked by uniqueness alone — all three now visible through the rank-tie warning rather than silent.
    Gate phase 2 gives a usable `hops` on only three levels (1-1, 5-3, 8-1); elsewhere the widened door's rooms
    are outside the room graph, so `needs_item` exists but nothing on the route carries it. That is the
    out-of-scope checkpoint-chain fallback, not a defect.

- **Integration pause, 2026-09-17: `next-levels` merged, verified in game, 8 games, `ent_coef` cut, run resumed.**
  The live 0-1 run was stopped at **4,805,630 steps** (343 episodes, 12 completions, all from checkpoint
  respawns; fresh `gates_reached` 4.2 of 10, `wedged_steps` 0, `on_target_frac` 5.7%) and resumed on the merged
  code as the **same run** — same `run_name`, same `models/campaign_gates/`, same weights — now multi-level.
  - **Stopping it: `latest.zip` beat the newest checkpoint.** `games.py stop` kills the trainer with an
    `EOFError`/`BridgeError`, and `train.py`'s `finally` still saved `latest.zip` — 4,805,630 steps against
    `ckpt_4764785_steps.zip`'s 4,764,785. Check both before resuming; do not assume `latest.zip` is stale after a
    non-graceful stop, because this kind of stop is not the hard kill that leaves it behind.
  - **Merge:** clean, no conflicts (`main` had moved only under `python/models/`, which the branch never touches).
    Full no-game suite on `main` after the merge: **13 files, 258 tests, all pass**; after the pause's own changes
    **14 files, 263 tests**. Mod built and installed at **69,632 bytes** (0.6.0 was 58,880).
  - **In-game readouts** (one game, port 47800, 30 fps / frameskip 2, rendering off, Violent, all gear):
    - **0-1 regression — PASS, identical to v0.6.0.** `summary: 1 PASS | 2 SKIP | 3 PASS | 4 PASS | 5 FAIL |
      6 PASS`, 11 gates, hops 0..9, one inactive controller, unchanged after a respawn; `gates: 11 ordered=True
      truncated=False with hops=11 (1.000)`; **`altars: 0` and `items: 0`** as present-and-empty lists, no
      `needs_item` anywhere. S4 is invisible on the level the policy has 3.2M steps on, which is what mattered.
      (Check 5 is the known standing exit FAIL on every level; it was not chased.)
    - **1-1 altars/items/needs_item — PASS on 5 of the 6 predictions.** `altars: 7` with exactly the predicted
      keys; `items: 5`, all `placed=true`; gate `81,-6,240` present with `altar_only`, `hops=1`, `needs=SkullBlue`;
      gate `20,-10,381` `hops=2` `needs=SkullRed`; gate `16,20,427` carries no `needs` (a reverse door).
      `campaign_check.py` check 6 reports **14 gates** = 13 phase-1 plus the one `altar_only`, as predicted.
      **The sixth is a partial miss and matters for S4 only:** the dead-twin marker was predicted to be
      `inactive_ancestors=2` on all three `#2` altars. `-15,27,427#2` and `0,-7,381#2` read 2, but
      **`81,-4,251#2` reads 1**, and `CampaignObserver.NeedsItem` skips an altar only above 1 — so on the blue
      leg an unfilled twin would keep `needs_item` set after the real altar is filled (spec risk M14). Whether
      that twin is a dead duplicate or a second altar the level really requires is **unresolved**; it blocks
      raising the item weights, which are staying at 0.0 anyway.
    - **0-2 — PASS on altars/items, one prediction contradicted.** `altars: 2` and `items: 2`, all `SkullBlue`,
      as predicted. But the plan expected **no** `needs_item` anywhere, and the phase-2 pass adds an `altar_only`
      gate `-60,-6,236` with `needs=SkullBlue` and **`hops=None`**. It is off the ladder (`gates: 18 ordered=True
      ... (0.944)`), so the route is unaffected, and 0-2 is not in the curriculum. Worth knowing before 0-2 is
      trained: a carry there would make `_protect_carry` gate the punch, since the unfilled altar reads
      `inactive_ancestors=1` and so counts as live.
    - **0-4 — PASS.** `altars: 0`, `items: 1` (`CustomKey1`), no `needs_item`, `gates: 7 ordered=True (1.000)`.
      With no altars `campaign.wanting_altars` is empty, so the punch-gating bug fixed on this branch cannot fire
      — and the key did read `held=True` in the game, so a carry on 0-4 is real, not hypothetical.
    - **6-2 exit tie-break (S3) — PASS.** Three separate loads all report the **same** exit
      `(-342.0, -70.1, 350.0)`, and **no rank-tie warning** in `LogOutput.log` — the two `Level P-` Prime Sanctum
      pits are discarded before ranking, leaving one candidate at the best rank. Check 6 reporting 0 gates on 6-2
      is expected (no gate candidates). A benign `2 door position key(s) collide in Level 6-2` warning does
      appear, which also proves warnings reach the log at all.
    - **7-2 usability guard (S2) — PASS.** `gates: 5 ordered=True truncated=False with hops=1 (0.200)`. The plan
      predicted ~4 gates at 0.250; the count differs, the verdict does not — 0.200 is below the 0.5 threshold, so
      the ladder is discarded and the env falls back to the exit vector.
    - **Skull checks (S4, spec §8 checks 1-3) — FAIL on check 1, 2 and 3 SKIP. The walk failed, not the punch.**
      Two targets were tried inside the time box. (a) The spec's pedestal `81,-2,275`: the walker stalled
      **68.4 m short** at `(14.0, 0.2, 261.4)` after 4000 decisions. A compass probe from 1-1's spawn
      `(0, 0.5, 253)` explains it — **only heading 0 (+z) is open**, reaching `z 400.6` in 150 decisions, while
      all seven other headings stop within 15 m. The pedestal sits at the far end of the route (its gate is
      `hops=1`, i.e. next to the exit), so it is not reachable on foot from spawn without playing the level.
      (b) The red skull sitting in the red altar at `0,-7,381`, straight up that open corridor: reached to
      **1.4 m** in 159 decisions and punched 56 times, but it reads **`active_self=False`, `inactive_ancestors=1`**
      even with the player standing on it — the object itself is disabled, so nothing could pick it up. **So the
      `Punch.ActiveFrame`-under-`render:false` question (check 1's kill switch) is still unanswered**, and
      `item_pickup` / `item_placed` stay at **0.0**. S4 is not needed for the prelude levels. Next attempt should
      start from an activated checkpoint rather than a fresh spawn, since the pedestal reads `active=True
      inactive_ancestors=0` once checkpoint `81,-6,231` has been activated.
  - **Smoke run (plan §6) — PASS on every item**, folded into the throughput A/B rather than run separately:
    `curriculum: 3 levels, unlocked ['Level 0-1']` printed before learning; `curriculum.json` names all three in
    order with only 0-1 unlocked; every `episodes.jsonl` row carries `"level": "Level 0-1"`; one level-keyed
    `explore_Level_0-1_478NN.npz` per game; `dashboard.py --smoke-test` renders; `metrics_log.csv` carries
    `levels_unlocked`, `part_item_pickup` and `part_item_placed`; `part_gate_approach` 8.4 (**not** collapsed, so
    the S2 guard is correctly not firing on 0-1) and `part_item_*` absent, i.e. 0.
  - **The one training change: `ent_coef` 0.01 -> 0.004.** Evidence at the stop: total entropy **11.8 nats of a
    13.59 maximum and still rising**, with all three look heads statistically uniform (yaw 2.19 of ~2.20, pitch
    1.78 of ~1.79, look mode 1.04 of ~1.10), `on_target_frac` 5.7% and fresh `gates_reached` flat at ~4.7 of 10
    with **no fresh-start completion in 343 episodes**. The Cyber Grind precedent is the same cut giving that
    run's biggest single gain, and its failure mode is known and guarded: below ~3 nats it went deterministic and
    regressed, so `keep_best.py --metric campaign` holds the weights and there is an entropy floor tripwire.
    Config hyperparameters do override a checkpoint's saved ones on `--resume` — `train.py` passes them into
    `PPO.load`, whose `model.__dict__.update(kwargs)` applies them — and it was confirmed live: within the first
    seven updates entropy went **11.8 -> 11.4/11.5 and stopped rising**, yaw 2.19 -> 2.07-2.12, pitch 1.78 ->
    1.68-1.73, look mode 1.04 -> 0.985-1.01. **Falsifier:** entropy should fall toward **6-8 nats within ~1M
    steps** *while* fresh `gates_reached` rises above 5 and fresh completions appear. Entropy falling without
    `gates_reached` rising means the cut is not the binding constraint — put it back. **Tripwire: total entropy
    below 6.0** — stop and raise `ent_coef`.
  - **Eight games instead of five**, after the BepInEx disk-log cap turned out to be the only limit. 235 steps/s
    against 5 games' 152 (+54%) at 33% CPU — see the two instance-count gotchas for the table and how to undo it.
    In the live run, which reloads levels far more often than a 90-second A/B does, it settles at **~186 steps/s
    per rollout** against the pre-pause run's 159 with 5 games. Read the A/B as the ratio, not as an absolute:
    both A/B legs ran short enough to see few level loads, which is what makes the absolute numbers optimistic.
  - **Resumed and running** on `configs/campaign_gates_prelude.yaml` from `models/campaign_gates/latest.zip`
    (4,805,630 steps) with 8 games; `poll_status.py`, `keep_best.py --metric campaign` and the dashboard alongside.
    The old `runs/campaign_gates/metrics_log.csv` was moved to `metrics_log_pre_multilevel.csv` so the new columns
    are written; the three new ports' exploration archives (47805-47807) were **seeded from 47802-47804's**, since
    a from-zero archive would re-pay novelty for ground the run has already covered.
  - Watch for, in order: the entropy falsifier above; fresh `gates_reached` past 5; the first **fresh-start**
    completion (the 12 so far are all checkpoint respawns); and then `levels_unlocked` going 1 -> 2, at which
    point the pooled `fresh_completion_rate` stops being readable as a percentage and only the per-level rows
    mean anything.
  - **ent_coef falsifier PASSED at +727k steps, and the first FRESH-START completions (2026-09-17 06:32,
    5,532,670 steps).** Entropy 11.43 -> 10.16 -> 9.19 with all three look heads falling together (yaw 2.09 -> 1.71,
    pitch 1.79 -> 1.24, look_mode 1.03 -> 0.81), fresh `gates_reached` 4.56 -> 4.88 -> 5.74, completions per 100
    episodes 2 -> 5 -> 17, `approx_kl` 0.0216, `explained_variance` 0.90. **Two fresh-start completions of 0-1:
    479.71 s official (64 kills, 2 deaths; `campaign.best_time`) and 522.12 s** (human reference 146.58 s).
    Weights kept as `models/campaign_gates/first_fresh_completion_5505630.zip`; `fresh_completion_rate` 0.04 on a
    full 50 window; 29 completions all-time.
    - Entropy is still descending at about 1.0 nat per 300k steps, which reaches the 6.0 tripwire near 6.5M steps.
      Plan at that point: raise `ent_coef` (0.006-0.008) to hold entropy around 6-7 rather than let it run down.
    - **Fresh episodes are bimodal** (last 100 fresh): 38 end early (gates 0-2, `stuck`, ~3,850 decisions; 22%
      never reach checkpoint 1) and 45 get deep (gates 7-8) but end on `max_steps`. The two completions used 7,967
      decisions on average, 88% of the 9,000 cap (600 game seconds), so deep runs are near-misses against the
      clock. Candidate for the next planned pause, as its own change: `max_steps` 9,000 -> 12,000. Not done
      mid-climb because it needs a trainer restart and the run is improving on its own.

- **S4 skull carry VERIFIED IN GAME on Level 1-1, 2026-09-17 — all three of spec section 8's checks pass, and
  the blocking question was never the one everyone was watching.** Run on a **ninth** game instance on port
  47808, launched by hand with `games.py`'s own arguments (`-aibridge-port 47808 -screen-fullscreen 0
  -screen-width 368 -screen-height 207 -job-worker-count 3`, `SteamAppId` set, detached + below-normal) while
  the `campaign_gates` run kept using 47800-47807 untouched. **Never run `games.py launch` or `stop` for this:
  both call `stop_all()`, which kills every game including the trainer's eight.** Training held 158-175 steps/s
  throughout.
  - **Check 1 pickup — PASS with `render: false`.** One aimed punch at 1-1's red pedestal `81,-2,275`
    (reading `active=True active_self=True inactive_ancestors=0`) flips `items[].held` true. **So the
    `ActiveStart`/AnimationEvent worry is dead**: `TrainingSpeed.ForcePlayerAnimators` does keep the fist
    Animator running with every camera disabled, and S4 is not inert under training settings. The `--render`
    control behaves identically, which is how the real cause was found.
  - **Check 2 placement — PASS.** Carried to the red altar `0,-7,381`: one aimed punch sets `filled: true` and
    **the gate `20,-10,381` goes `open: true`** — the door really unlocks. The placement then survived
    **100 decisions of punch spam issued through `env.step`**, so **`_protect_carry` holds in the live game**,
    not just against `FakeLevel`.
  - **Check 3 held skull on death — PASS, and the answer is "it comes with you".** `kill()` while holding, then
    the respawn: the skull reads `held: true` still, at the checkpoint (`81.7,-3.6,229.6`, 2.3 m from the
    player), `deaths` 0 -> 1, the episode does not end. The level's other `SkullRed` entry is the destination
    altar's decoration (`active_self: false`). **A death mid-carry costs the carry nothing** — no re-fetch, and
    no unsolvable-puzzle failure mode to design around.
  - **THE REAL BUG, and it is Python-side: every aim at a skull or an altar is taken from the wrong point.**
    `Punch.ActiveFrame` (`decompiled/Punch.cs:547`) rays from **`cc.GetDefaultPos()` — the camera** — along
    camera forward for 4 m. `skull_check.py` and **`env._look_at_target` (look mode 2)** both compute the
    elevation from `player.pos`, i.e. `NewMovement.transform.position`, which sits **0.9 m below the camera**.
    At punch range that is a ~30 degree error and the ray passes clean over the target: measured on the pedestal
    at 1.57 m, aiming from `player.pos` asks for +27.0 deg and fails, the camera wants -3.0 deg and picks it up;
    at 2.71 m, +11.3 fails and +8.0 works. Both solve to **0.88-0.90 m**. The two earlier in-game attempts —
    including the one that concluded "the walk failed, not the punch" — were **59 aimed punches at a live
    pedestal 1.8 m away that all flew overhead**, with rendering on *and* off.
    - `scripts/skull_check.py` is fixed: new `eye(player, height)` helper, `CAMERA_HEIGHT = 0.9`, a
      `--camera-height` override (the offline tests pass `0`, since `FakeLevel` has no camera).
    - **`ultrakill_ai/env.py` is NOT fixed and needs the same correction before `item_pickup`/`item_placed` ever
      leave 0.0.** `_look_at_target` must aim from `player.pos + (0, 0.9, 0)` whenever `wide` is set, and
      `_near_subgoal`'s range test is measured from `player.pos` too. Left alone deliberately: the trainer
      imports this file and the live curriculum (0-1/0-3/0-4) has no wired altar, so mode 2 only ever aims at
      doors, where 0.9 m is under 3 degrees at 20 m and harmless. **It is not harmless at 4 m**, so as it stands
      the agent could never place a skull even with the weights raised.
    - **A second aim offset, not yet explained.** Aiming at `altars[].pos` exactly does not place: the placement
      only fired when aimed **~1.25 m below** the zone's reported position (`0,-7,381` -> aim y -8.0). Likely
      the ray hits the zone's decoration skull, which carries an `ItemIdentifier`, and `AltHit` bails on
      `if (itemIdentifier && hasHeldItem) return;` (`decompiled/Punch.cs:1341`). Pickup needs no such offset.
      Whatever `env` ends up doing for mode 2 has to clear this too; `skull_check.py` takes it as `--altar`.
  - **Data oddity 1 (M14) CONFIRMED AND REPRODUCED, on the red leg as well: `needs_item` does NOT clear after a
    real placement.** With `0,-7,381` filled and its door open, gate `20,-10,381` still read
    `needs_item: 'SkullRed'`. Cause measured directly: **`inactive_ancestors` conflates "my room is switched
    off" with "I am a dead twin"**, and the number moves when the room switches on. On a fresh load the live
    altar `81,-4,251` reads 1 and its twin `#2` reads 2; **after the checkpoint respawn switches that room on,
    the live one reads 0 and the twin reads 1**. `CampaignObserver.NeedsItem` skips only `> 1`, so the twin
    stops being filtered exactly when the player arrives, and holds the lock set forever. This is a **mod fix**:
    the dead-twin test must be relative, not a constant — a zone is dead when another zone with the same `item`
    and the same `doors` reports strictly fewer `inactive_ancestors` (equivalently, once the room is on, the
    live zone is the `active` one). Until it is fixed a skull-locked gate never reports itself solved, so
    `GateProgress._is_reached` refuses the gate forever and the fetch/carry machine cannot advance to `open`.
  - **Data oddity 2 (0-2's `-60,-6,236 needs=SkullBlue hops=None`) — cannot become a target; no mod fix needed
    for that.** `GateProgress._choose_target` filters on `g.get("hops") is not None` (campaign.py:709) and
    `_is_reached` returns False for `hops is None`, so a `hops: null` gate is invisible to targeting and to the
    ladder. **But there is a latent trap if 0-2 is ever added to the curriculum:** `_protect_carry` keys on
    `campaign.wanting_altars`, which does not look at gates at all, so picking that blue skull up drops the
    punch button — while `_near_subgoal` can never release it, because releasing needs a `subgoal` target and
    0-2 has no targetable gate to build one from. Fix when 0-2 is trained, not before: either give the phase-2
    gate a usable `hops` mod-side, or release the punch near any unfilled altar that accepts the held item when
    there is no sub-goal.
  - **How to reach the pedestal at all** (the previous attempt's blocker, now solved and folded into the
    script): 1-1's route to `81,-2,275` runs **through the skull-locked door `81,-6,240` itself**, so no walk
    from spawn can get there. `--from-checkpoint 81 -6 231` teleports onto that checkpoint and lets it activate,
    **and then respawns** — and the respawn is the part that matters: after the teleport alone the pedestal
    still reads `active: false`, and only `reset(checkpoint=True)` brings the room up to
    `active: true, inactive_ancestors: 0`. Activating `46,0,388` first also lights the red altar's room, and it
    survives the later respawn elsewhere. `--approach` / `--altar-approach` then stage into those now-live rooms
    and the last metres are still walked. The whole run:
    `python scripts/skull_check.py --port 47808 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231
    --approach 81 -1 255 --altar-approach 0 -4 374 --altar 0 -8 381 --budget 700`
  - **`item_pickup` / `item_placed` stay at 0.0** — the lead decides, and the `NeedsItem` mod fix and the
    `env._look_at_target` aim fix both have to land first or a raised weight buys nothing.

- **Branch `skull-fixes`, 2026-09-17: the four skull blockers fixed, three of them verified in game on a ninth
  instance.** Built in the worktree `F:\Github\ULTRAKILL-AI-skull` while `campaign_gates` kept training on
  47800-47807; the mod compiles (`dotnet build -c Release -p:InstallPlugin=false`) but could not be installed,
  because the running games hold the DLL. **Not merged into `main`.**
  - **F1 -- aim from the EYE, not the feet (Python).** `Punch.ActiveFrame` (`decompiled/Punch.cs:562`) rays from
    `cc.GetDefaultPos()`, the camera, which sits 0.9 m above `player.pos` (`NewMovement.transform.position`).
    `env._look_at_target` (look mode 2) and `_near_subgoal`'s reach test both measured from the feet, so every
    aim at a skull or an altar pointed over the top of it. New `EnvConfig.camera_height_m` (0.9; 0 restores the
    old behaviour) and `env._eye`.
    - The error is a short-range one and the correction is in the right direction everywhere: `atan(0.9/r)` is
      **29.8 deg at 1.57 m**, 16.7 at 3 m, **5.1 at 10 m, 1.7 at 30 m**. So a gate 10-30 m away moves by a couple
      of degrees, far inside any aim tolerance, and moves the correct way -- a camera 0.9 m up really does have to
      look slightly down at a door sill at its own feet's height. At punch range it is the whole answer.
    - **Look mode 1 needed no change, and that was verified rather than assumed.** `enemies[].rel` is
      `cam.InverseTransformPoint(centre)` (`ObservationBuilder.cs:246`) -- already measured from the camera -- and
      the only other use of `player.pitch` there is un-pitching `rel` into the yaw frame, a rotation about the
      same origin. `test_look_mode_1_is_already_camera_relative_and_is_left_alone` pins it at three camera
      heights: adding an eye offset there would introduce the very error mode 2 had. The mod reports **no camera
      position**, so the 0.9 is a constant; a future `player.cam_pos` would make it exact while crouched.
  - **F2 -- the placement offset explained, and it was NOT the decoration skull.** The `AltHit` `ItemIdentifier`
    early-return hypothesis is **wrong for this case**: 1-1's live red altar (`4 - Altar Field/Altar/Cube`) has no
    child skull at all. The real cause is geometric and uniform. `AltHit` calls
    `target.GetComponents<ItemPlaceZone>()` on the transform the raycast returned, and `ItemPlaceZone.Start`
    reads its own `GetComponent<Collider>()`, so the ray must hit **the zone's own collider** -- and that collider
    is not centred on the transform. Read out of the scene bundles, **all 104 campaign zones are one prefab**:
    layer 22, a single trigger `BoxCollider`, local size `(2.2, 3.5, 2.2)`, local centre `(0, -1.25, 0)`, on a
    transform scaled `(0.9, 0.8, 0.8)`. So the box centre sits **1.0 m below `pos`** (103 of 104; one is 0.625 m)
    and the box spans `pos.y-2.4 .. pos.y+0.4`: **`pos` is only 0.4 m under the lid, with no margin**, while the
    centre has 1.4 m. That is the "extra ~1.25 m downward" the first in-game run found by hand.
    - Fixed at the mod layer as the right one: `CampaignObserver.AimPoint` reports `altars[].aim_pos` =
      `transform.TransformPoint(box.center)` (`TransformPoint`, not `Collider.bounds`, because a zone in a room
      that has not streamed in is inactive and its bounds are meaningless). Python's `campaign.altar_aim_point`
      prefers it and falls back to `pos` minus 1 m, and `GateProgress._subgoal` makes it the altar sub-goal's
      `pos`, so the observation, the aim, the approach and the reach test all use one point.
  - **F3 -- the `NeedsItem` dead-twin filter is now RELATIVE (mod).** `inactive_ancestors` is not a property of
    the zone: it is the zone's own chain plus however much of the room above it is switched off, so it **shifts
    when the room lights** (measured on 1-1: live/twin 1 and 2 on a fresh load, **0 and 1** after the respawn).
    The absolute `> 1` test therefore stopped filtering the twin exactly when the player arrived. Replaced by
    `CampaignObserver.IsDeadTwin` and `campaign.dead_twin`, applied on **both** sides and everywhere altars are
    filtered (`NeedsItem`, `ScanGates`' phase 2, `wanting_altars`): a zone is dead when another zone with the
    same item type, the same door set **and the same rounded position** reports strictly fewer inactive
    ancestors. A shared room contributes equally to both halves of a pair, so the answer cannot be shifted, and
    the minimum of each group always survives, so a lock can never be filtered away.
    - **Re-validated offline against all 21 altar levels** (scene bundles, 104 zones, 28 wired doors). Dropped
      zones **20 (old) -> 20 (new)**; **locks lost: 0** on every level; doors still demanding an item after the
      puzzle is solved **1 -> 0**, and after a room lights **11 -> 0**. Position is part of the identity because
      twins are co-located (~0.2 m, which is why the keys need `#N`); without it the rule also drops five zones
      that are not twins but separate altars of one item type driving no door (4-2 x2, 4-3, 5-3, 7-1).
  - **F4 -- the 0-2 carry trap, closed by construction (Python).** `_protect_carry` keyed on
    `campaign.wanting_altars`, which does not look at gates, while `_near_subgoal` can only release on a
    `GateProgress` sub-goal. Both now key on the sub-goal: a press is dropped only while an altar sub-goal exists
    for something actually held, so the button can never be taken away by something that cannot give it back.
    - **What 0-2's off-ladder altar should do: nothing, and that is now what happens.** Its door `-60,-6,236` is
      a **"Secret Wall"** whose only `activatedRooms` entry is `-60,-11,236`, which appears nowhere on 0-2's pit
      chain (`9 - Crushers Arena` -> `FinalRoom` -> `Pit`). It is a secret arena, off the route, which is why its
      `hops` is null -- correct, not a mod bug. The agent keeps punch, may carry the skull or throw it away, and
      the route loses nothing.
  - **F5 -- `max_steps_per_level`** (a dict keyed by scene name, empty by default) so a `levels` ladder whose
    rungs differ in size is not stuck with one cap. `max_steps` itself is unchanged.
  - **Live verification, 2026-09-17, on a NINTH game on port 47808** started by hand with `games.py`'s own
    arguments (PID recorded, killed by PID alone; **never `games.py launch`/`stop`, both call `stop_all()`**).
    47800-47807 stayed listening throughout and `runs/campaign_gates/status.json` kept updating at 160-173
    steps/s. The installed DLL is the **old** v0.7.0, so this ran F2 through the **Python fallback** -- which is
    the stronger test, since it needed no mod at all.
    - `skull_check.py --port 47808 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231 --approach 81 -1 255
      --altar-approach 0 -4 374 --budget 700` -- **check 1 pickup PASS in ONE aimed punch** and **check 3 PASS**,
      with **no `--camera-height` and no `--altar` offset**: both hacks are gone.
    - **Check 2 placement: the altar filled and the door opened in ONE aimed punch**, aimed at the collider
      centre `(0, -7.8, 381)` derived from the reported `(0, -6.8, 381)`. Re-run with an empty `--gate` it is a
      clean **PASS**, including **100 decisions of punch spam through `env.step`** -- so F4's rekeying did not
      weaken the 1-1 carry protection. With the gate assertion on it reports FAIL, and that FAIL **is F3**:
      `altar 0,-7,381 filled=True` and `gate 20,-10,381 open=True` while `needs_item='SkullRed'` -- M14 captured
      live, from the old DLL, and exactly what the installed fix will clear.
    - **F4 on real 0-2 data:** the live block reports `gate -60,-6,236 hops=None altar_only=True`,
      `wanting_altars(SkullBlue) = ['-45,-6,236']` (non-empty -- the old rule **would** have engaged) and
      `gates.target` an ordinary gate with no sub-goal. With the skull marked held, `_protect_carry` **keeps**
      the punch, and the old `wanting_altars` key **would have dropped** it. The physical pickup did not land
      because the skull reads `active=False` -- its room is still switched off -- which is the known
      teleport-into-a-dark-room case, not the carry rule.
  - **At the next pause, in order:**
    1. Stop training gracefully (Ctrl+C, wait for `Saved ... latest.zip`), then `games.py stop`.
    2. Merge `skull-fixes` into `main`, then `cd mod/UltrakillAIBridge && dotnet build -c Release` to install the
       DLL (it only fails while the games hold it).
    3. **The F3 readout:** relaunch one game and rerun the `skull_check.py` line above **with** its default
       `--gate 20,-10,381`. It must now print
       `summary: 1 pickup PASS | 2 placement PASS | 3 held skull on death PASS`; the single thing that changes is
       `gate 20,-10,381 needs_item=None` once the altar reads `filled=True`. Also confirm `altars[].aim_pos` is
       now present (`bridge_test.py --campaign`), so the 1 m fallback stops being used.
    4. Recommended weights: **`item_pickup` 10.0, `item_placed` 20.0** (30 for the puzzle, two `gate` rungs at
       15). Arithmetic: `time` 0.02 per decision at 15 decisions/s is **0.3 per game second**, so 1-1's red leg
       -- roughly 35 m to the pedestal, 133 m to the altar, 20 m back to the gate, ~190 m, 24-48 s -- costs
       **7.2-14.4**. 30 clears that with margin and is 30% of `level_complete` (100) for a leg 1-1 cannot be
       finished without, next to `checkpoint`/`arena_clear` at 10. Weight the terminal event higher so
       collect-and-abandon is not worth it alone; 15/15 is the symmetric fallback. Do **not** go above ~`gate`,
       or fetching a skull the route does not need (0-2's) becomes attractive.
    5. Not farmable at any of these: both are paid once per level load (`MilestoneTracker`, pinned by
       `test_a_placement_pays_once_even_when_the_skull_is_pulled_back_out`). Note `door_unlock` does **not** fire
       for an altar door -- `ItemPlaceZone.CheckItem` calls `Door.Open()`, not `Unlock()` -- so `gate` is the
       only other term the leg earns.
    6. Then resume, and read `part_item_pickup` / `part_item_placed` against `part_gate` in `metrics_log.csv`
       for the first 400k steps before touching them again.

- **Integration pause #2, 2026-09-17: `skull-fixes` merged, S4 verified in game, 12 games, the campaign opened
  to 11 levels, run resumed.** The live run was stopped at **6,771,574 steps** (678 episodes; Level 0-1 at a
  **0.54** fresh-start completion rate over its last 50 fresh starts, best official time **260.28 s** against the
  human 146.58 s; Level 0-3 unlocked with 9 episodes) and resumed on the merged code as the **same run** — same
  `run_name`, same `models/campaign_gates/`, same weights. Rollback point: `4243ad4`; merge commit `faecc29`.
  - **It had already died on its own, four minutes before the pause.** One `SubprocVecEnv` worker raised
    `TimeoutError` out of `protocol.recv` waiting for a level reset, and `train.py`'s `finally` still wrote
    `latest.zip` at 6,771,574 — **more than the newest checkpoint** (`ckpt_6755630`), so `latest.zip` was again
    the right resume file. Check both every time; this kind of stop is not the hard kill that leaves it stale.
    The trainer then hung in teardown with its workers stuck and had to be killed by PID after `games.py stop`.
    Worth remembering: a bridge timeout on a reset kills the whole run, and nothing restarts it.
  - **Merge:** clean, no conflicts. Full no-game suite on `main` after the merge: 15 files, all pass; after the
    pause's own changes **15 files, 304 named tests, all pass** (`test_progress.py`'s 19 print no count, so the
    per-file lines add to 285). Mod rebuilt and installed at **70,144 bytes** (the pre-merge build was 69,632).
  - **In-game readouts** (one game, port 47800, 30 fps / frameskip 2, rendering off, Violent, all gear):
    - **0-1 regression — PASS, unchanged.** `summary: 1 PASS | 2 SKIP | 3 PASS | 4 PASS | 5 FAIL | 6 PASS`;
      `gates: 11 ordered=True truncated=False with hops=11 (1.000)`, hops 0..9, and **`altars: 0` / `items: 0`**
      as present-and-empty lists. S4 is still invisible on the level the policy has most of its steps on, which
      is the thing that mattered. (Check 5 is the known standing exit FAIL on every level.)
    - **F3, the merge's whole point — PASS, all three checks.**
      `skull_check.py --port 47800 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231 --approach 81 -1 255
      --altar-approach 0 -4 374 --budget 700` with its **default `--gate 20,-10,381`** prints
      `summary: 1 pickup PASS | 2 placement PASS | 3 held skull on death PASS`. Pickup in **one** aimed punch;
      placement in **one**, aimed at the collider centre `(0, -7.8, 381)` derived from the reported
      `(0, -6.8, 381)`, with **no `--camera-height` and no `--altar` hack** — both are gone. And the M14 line
      that was the blocker now reads **`gate 20,-10,381 needs_item=None open=True locked=False`** with the altar
      `filled=True`, where before the fix it kept `needs_item='SkullRed'` forever. It survived **100 decisions of
      punch spam through `env.step`**, so `_protect_carry`'s rekeying holds in the live game.
    - **`altars[].aim_pos` arrives.** All 7 of 1-1's altars carry it, exactly 1.0 m below `pos`, so the Python
      1 m fallback is no longer load-bearing. `bridge_test.py --campaign` now prints it (`aim=(x, y, z)`, or `-`
      on an older mod). The relative dead-twin rule is visible in the same readout: with the room lit, the live
      `0,-7,381` reads `inactive_ancestors=0` and its twin `#2` reads 1 — the exact pair the old absolute `> 1`
      test got backwards.
    - **0-2 — gates PASS, the off-route altar is inert as designed, one physical leg blocked.**
      `gates: 18 ordered=True truncated=False with hops=17 (0.944)` and check 6 PASS; `altars: 2` / `items: 2`,
      all `SkullBlue`. Running the **real** `GateProgress` against the live block from all 18 gate positions and
      the player's own: the secret-wall gate `-60,-6,236` (`hops=None altar_only needs=SkullBlue`) **never became
      the target from any of them**. And with the skull marked held, `wanting_altars(SkullBlue)` is non-empty
      (`['-45,-6,236']`, so the OLD gate-blind rule **would** have dropped the punch) while the merged
      `_protect_carry` **keeps** it — F4 confirmed on real 0-2 data, not just against `FakeLevel`.
      **The scripted carry: pickup PASS in one aimed punch** on the loose blue skull `-4,22,173`; **placement
      FAIL**, and the output names why — `altar -45,-6,236 ... active=False inactive_ancestors=1`. Its room is
      switched off, the known teleport-into-a-dark-room case, not a carry-code failure. 0-2's altar is an
      off-route secret, so nothing on the route depends on it.
  - **12 games instead of 8**, on a second throughput A/B (table and method in the instance-count gotcha):
    188.6 -> 220.5 -> **236.5 steps/s** at 8 / 10 / 12, zero bridge errors in every leg, 9.6 GB of 31.9 GB. The
    live run settles around **270 steps/s** on its early rollouts. 47808-47811's exploration archives were seeded
    from 47800-47803's.
  - **`max_steps` 9000 -> 12000, on measurement.** Of the last 100 **fresh** episodes before the pause,
    **24 ended at the 9000 cap** and 23 of those held **7-10 of 10 gates**, while the 41 that completed took a
    **median 6898 decisions**. Deep runs were losing to the clock, not to the level. **This is not a reward
    change:** a `max_steps` end is a truncation, so SB3 bootstraps `gamma * V(s_T)` and nothing pays for the
    extra time. `stuck_seconds` stays at 45, so the early-stuck cohort still ends early and the longer cap costs
    nothing there.
  - **`item_pickup` 10.0 and `item_placed` 20.0**, off 0.0 at last, on the F3 PASS above. The arithmetic is in
    the skull-fixes checklist; the two bounds now pinned by a test are `item_pickup <= gate` and
    `item_pickup + item_placed <= 2 * gate`, which is what keeps an off-route skull from being worth the detour.
    `item_placed` alone is allowed above `gate`: it is the terminal event, it has to out-weigh the pickup so that
    collect-and-abandon is not worth it alone, and it is only ever paid at an altar the agent already stands at.
  - **The curriculum opened to 11 levels** — 0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 4-1, the level
    survey's Tier A and Tier B in mission order — with **`unlock_after_fresh_episodes: 600`**, the new safety
    valve. 0-1 and 0-3 were already unlocked and **stayed** unlocked across the reorder that inserted 0-2 between
    them; the trainer printed `curriculum: 11 levels, unlocked ['Level 0-1', 'Level 0-3']` on start. See the
    curriculum bullet under Layout for why that is safe by construction and which tests pin it.
  - **`best.zip` was preserved before the pooled score changed meaning.** `keep_best --metric campaign` scores
    the pooled rate, which is a shrunk **sum** over unlocked levels; once several levels are unlocked a policy
    that is *worse at 0-1* but better overall can beat the stored record and take `best.zip`. That is arguably
    the right behaviour for a campaign run, but it loses the best 0-1 policy, so the current one was copied to
    **`models/campaign_gates/best_0-1_only_6755630.zip`** (+ `.json`, score 0.5111) first.
  - **Resumed and running** on `configs/campaign_gates_main.yaml` from `models/campaign_gates/latest.zip`
    (6,771,574 steps) with 12 games, `poll_status.py`, `keep_best.py --metric campaign` and the dashboard
    alongside. `runs/campaign_gates/metrics_log.csv` was moved to `metrics_log_pre_pause2.csv` so the new
    `fresh_episodes` and per-level columns are written — `poll_status.py` keeps an existing header and silently
    drops columns it lacks.
  - **Watch for, in order:**
    1. **Per-level fresh rates**, never the pooled one. 0-1 has to refill its 20-episode window at >= 0.5 before
       0-2 unlocks; the windows deliberately start empty after a restart, so every level reads rate None for a
       while and the sampling weights are uniform until they refill.
    2. **`part_item_pickup` / `part_item_placed` against `part_gate`** in `metrics_log.csv` for the first 400k
       steps, on the skull levels only. Both are paid once per level load, so they should stay small next to
       `gate`; a large `part_item_pickup` means skulls are being fetched that the route does not need.
    3. **The entropy tripwire is unchanged: total entropy below 6.0** — stop and raise `ent_coef` to 0.006-0.008.
       It read 7.68-7.83 through the resume and was flat.
    4. **0-1's early-stuck cohort is the main cap on its rate, and it is ~25%.** Of the last 100 fresh 0-1
       episodes, **25 reached 2 gates or fewer** and 12 of those killed nothing at all: the agent never picks the
       revolver up, or dies in the gun room. Nothing in this pause addresses it — `max_steps` helps the *deep*
       runs, not these — so 0-1's rate stays roughly capped near 0.75 until it is. That is the next thing to
       attack on 0-1, and it is a separate change from anything shipped here.
