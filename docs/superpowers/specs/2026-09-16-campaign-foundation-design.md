# Campaign foundation, piloted on 0-1: design

Date: 2026-09-16. Status: approved in brainstorming, awaiting spec review.

## Goal

Train an agent that finishes ULTRAKILL campaign levels as fast as possible, learning on its own: no human
demonstrations or recorded routes. This spec is the first of several sub-projects. It builds campaign support
in the mod and the Python side and proves it on a single level, 0-1.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Human demos? | None. Progress signals come from game objects the mod reads at runtime. |
| Difficulty | Violent (3), set in memory on the training games. |
| Weapons | Every weapon and variant unlocked, in memory only. Saves and prefs are never written. |
| Approach | Game milestones + exploration reward + frontier curriculum through native checkpoint respawns (approach A). Rejected: curiosity only (B), scripted NavMesh steering with RL combat (C). |
| Level scope | The 35 main levels, `Level 0-1` to `Level 9-2`. Prime Sanctums (P-1..P-3) and encores (0-E..9-E) are out of scope for now. |
| Official time | `StatsManager.seconds` at the moment the player enters the real `FinalPit`, which is when the game stops the timer. |
| Games | All 5 training games run the campaign. Cyber Grind training stays paused. |
| Deaths | Do not end an episode. The player respawns at the checkpoint and the clock keeps running, as in real play. |

## Sub-projects

1. **Campaign foundation, piloted on 0-1 (this spec).**
2. The rest of the Prelude (0-2 to 0-5): one policy trained on several levels, adding levels as earlier ones
   are cleared. 0-5 brings the first boss.
3. Skull keys and switches: item-carrying observations (item held, altar positions).
4. Acts 1 to 3 (1-1 to 9-2), bosses handled as they come.
5. Speed: once levels are cleared reliably, move the reward almost entirely onto time. Add `hook` (whiplash)
   and `change_fist` to the action space here for movement tech; this spec leaves the action space unchanged.

## Success criteria for this spec

- The agent finishes `Level 0-1` on Violent from a fresh level load in at least 50% of deterministic
  `eval.py` runs, with real deaths.
- The best official time and rank are recorded in `times.md`.

## Game facts this design relies on

All from the decompiled `Assembly-CSharp.dll` (Unity 2022.3.29 build). Checked 2026-09-16.

- **Exit.** `FinalPit` is a trigger baked into the scene. On entry it sets `NewMovement.levelOver = true` and calls
  `StatsManager.StopTimer()`, then sends results 5 s later. Decoys carry `fakeEnd`, `secondPit` or `rankless`.
  The pit may sit in a room that starts inactive, so search with `FindObjectsOfType<FinalPit>(true)`.
- **Room templates.** `CheckPoint.Start()` clones each room in `rooms` and moves the disabled original +10000 on
  X. Searches that include inactive objects must filter those copies out.
- **Timer.** `StatsManager.seconds += Time.deltaTime * GameStateManager.Instance.TimerModifier` while `timer`.
  Checkpoint respawns do not reset it. Cutscenes do not stop it.
- **Checkpoints.** `CheckPoint.activated`, `ActivateCheckPoint()`, `StatsManager.currentCheckPoint`.
  `StatsManager.Restart()` respawns at the current checkpoint (and reloads the scene if there is none). That
  resets the rooms the checkpoint owns, unlocks `doorsToUnlock` and increments `restarts`.
- **Arenas and doors.** `ActivateNextWave.EndWaves()` runs when an arena's last wave dies. It unlocks its doors
  and opens `doorForward`. `Door.locked`, `Door.open`, `Door.Unlock()`.
- **NavMesh.** Enemies use `NavMeshAgent`, so campaign scenes have a baked NavMesh. It is a ground mesh: jumps
  and gaps are not connected, and doors carry `NavMeshObstacle`s.
- **Level start.** The player drops in with the `"pit-falling"` game state (camera locked).
  `NewMovement.activated` goes true on landing. `GameStateManager.Instance.PlayerInputLocked` covers cutscenes.
- **Difficulty.** `PrefsManager.GetInt("difficulty")`: 0 Harmless, 1 Lenient, 2 Standard, 3 Violent, 4 Brutal.
- **Weapons.** `GunSetter.ResetWeapons()` adds a variant when `GameProgressSaver.CheckGear(name) > 0` and the pref
  `weapon.<name>` is above 0. Early levels start with `GunSetter` disabled until the first `WeaponPickUp`.
- **Rank.** `StatsManager.timeRanks`, `killRanks` and `styleRanks` each hold 4 thresholds per scene. Each category
  scores 0 (D) to 4 (S). The total is the sum minus `restarts`; 12 with no cheats is P, otherwise
  `RoundToInt(total / 3)`.
- **No teleport detection.** Nothing checks player position for cheating, so mod-side resets do not stop a
  level from completing.

## Mod changes (v0.5.0)

### Observation: `campaign` block

Present only in campaign scenes (mission numbers 1 to 35, from `StatsManager.levelNumber`).

| Field | Content |
|---|---|
| `mission`, `difficulty` | `StatsManager.levelNumber`, and the difficulty the game actually reads |
| `seconds`, `timer_running`, `level_started`, `level_over`, `restarts` | from `StatsManager` / `NewMovement` |
| `input_locked` | `PlayerInputLocked` or `!NewMovement.activated` |
| `exit` | `{pos}` of the real `FinalPit`, or `null`. Skips `fakeEnd`, `secondPit`, `rankless` and +10000 X templates. |
| `checkpoints` | list of `{id, pos, activated, current}`, template copies excluded |
| `path` | `{status: "complete" \| "partial" \| "none", length, next_corner}`: `NavMesh.CalculatePath` from the player to the exit, each end snapped with `NavMesh.SamplePosition`. Recalculated every 4 steps and cached in between. |
| `locked_doors` | nearest 4 `{pos, dist}` with `Door.locked` |
| `arena_enemies_alive` | live enemies whose parent `ActivateNextWave` is activated and not finished |
| `arenas_cleared`, `doors_unlocked` | counters incremented by Harmony postfixes on `ActivateNextWave.EndWaves` and `Door.Unlock`. They reset to 0 on scene load. |
| `ranks` | `{time: [4], kills: [4], style: [4]}` from `StatsManager` |

### Config settings

| Key | Default | Meaning |
|---|---|---|
| `difficulty` | unset | if set, a Harmony prefix on `PrefsManager.GetInt` returns it for `"difficulty"` while the AI has control |
| `unlock_all_gear` | false | `GameProgressSaver.CheckGear` returns 1, and `weapon.*` prefs read as 1, while the AI has control |

Both must be sent before the level loads, because enemies and `GunSetter` read them at scene start. Save and
prefs writes are already skipped while the AI has control (`InstancePatches`).

### Resets

No new reset types. `reset(checkpoint=true)` uses `StatsManager.Restart()`. `reset(checkpoint=false)` reloads the
scene. Mid-level teleport starts (the game's `TeleportCheat` recipe) are deliberately left out: that recipe does
not restore earlier doors, arena flags or objects switched on by level events, so the state could not be
trusted.

### Docs

Update `docs/protocol.md` (new block and config keys) and `docs/game-internals.md` (the facts above).

## Python changes

### Layout

- `ObsLayout` gains a mode. Cyber Grind stays exactly 448 dims (its 5 zero waypoint values included), so its
  checkpoints keep loading.
- Campaign replaces the 5 waypoint values with a 36-value block, giving 479 dims. Every index before it
  (player 0-16, enemies 17-416, rays 417-432, ground rays 433-440, Cyber Grind 441-442) is unchanged.

| Values | Content |
|---|---|
| 5 | exit: yaw-frame relative xyz / 100, distance / 200, mask |
| 8 | path next corner: yaw-frame relative xyz / 50, distance / 50; path length / 300; status one-hot (complete, partial, none) |
| 5 | nearest checkpoint not yet activated: relative xyz / 100, distance / 200, mask |
| 5 | nearest locked door: relative xyz / 50, distance / 100, mask |
| 4 | arena enemies alive / 20, timer running, input locked, level seconds / 600 |
| 9 | exploration map: `min(1, log1p(N) / log(1001))` for the player's cell and its 8 horizontal neighbours |

### Episodes (`UltrakillEnv`, campaign mode)

- **Start.** Choose a fresh level load when any of these holds:
  - the previous episode completed the level;
  - no checkpoint has been activated in the current level load;
  - three episodes in a row ended stuck at the same current checkpoint;
  - otherwise, with probability `fresh_start_prob` (0.2).

  If none holds, respawn at the current checkpoint (`reset(checkpoint=true)`). Each game drifts toward its own
  frontier, so most training lands on the part of the level it cannot do yet.
- **Input locked.** While `input_locked` is set, the environment steps with an empty action inside `reset()`
  and `step()` and never hands those frames to the policy.
- **Death.** Apply the `death` penalty, call `reset(checkpoint=true)` inside the same episode, wait for the
  player, and continue.
- **End.**
  - `level_over` means terminated, with reason `level_complete`.
  - A game-time cap (`max_steps` 9000, i.e. 10 min at 15 decisions/s) means truncated.
  - `stuck_seconds` (45 s) with no progress means truncated. Progress is a checkpoint, arena clear, door unlock,
    new best path distance, or entering a cell not yet visited this episode.
- **Speed settings.** Same as Cyber Grind: `fixed_fps` 30, `frameskip` 2, `render` false, and 368x207
  windows. `soft_death` is off (real deaths drive respawns).

### Rewards (`RewardConfig`, campaign terms)

| Term | Start value | Rule |
|---|---|---|
| `time` | -0.01 per decision | constant, so -9 per game minute |
| `level_complete` | +100 | once. Finishing within the 10 min cap always beats not finishing. |
| `checkpoint` | +10 | each checkpoint's first activation per level load |
| `arena_clear` | +10 | per increment of `arenas_cleared` |
| `door_unlock` | +3 | per increment of `doors_unlocked`. Counter changes across a reset or respawn are ignored. |
| `novelty` | +0.5 / sqrt(N + 1) | on first entry to a 4 m cell this episode. N = earlier episodes of this game that entered the cell (per level). |
| `path` | +0.1 per metre | when the complete-path length to the exit reaches a new minimum this episode |
| `kill`, `damage_dealt` | +0.5, +0.5 | existing computation |
| `damage_taken`, `death` | -0.01 per HP, -5 | existing computation |

The exploration archive (cell visit counts) is kept in memory per environment and per level, and saved next
to the checkpoints (`models/<run>/explore_<env>.npz`) so resuming keeps it.

### Episode info (dashboard, Monitor, `status.json`)

- `completed` (0/1), `fresh_start` (0/1), `level_seconds` (official time, set on completion).
- `checkpoints_reached`, `furthest_checkpoint` (checkpoint index in hierarchy order).
- `cells_new`, `deaths`, `exit_dist_min`, `end_reason`.

### Best runs

When a fresh-start episode completes faster than the stored best, write positions (every step), official time,
kills, style, restarts and the computed rank to `runs/<run>/best_runs/<scene>.json`.

### Retired

`python/ultrakill_ai/routes.py`, `python/scripts/record_route.py` and the `route_point` / `stuck` reward terms
existed only for human-recorded routes. Delete them (git history keeps them). `configs/campaign_0-1.yaml` is
rewritten.

## Training

### Starting weights: `scripts/transfer_weights.py`

- Build a fresh 479-input PPO policy.
- Load `models/cybergrind_ppo_v2/best.zip` (kills/min 7.16 at 4.70M steps).
- Copy both hidden stacks (policy and value, 512x512).
- First-layer columns 0-442 come from the old weights. Columns 443-478 are zero, so on existing inputs the
  network computes exactly what it did.
- Scale `action_net` weights and bias by 0.5 to raise entropy.
- Re-initialise the final `value_net` layer, because the reward scale is unrelated.
- Start the optimizer fresh.
- Save to `models/campaign_ppo/transfer_init.zip`.
- If the first live rollout's entropy is outside 6-10 nats, adjust the scale once and restart.

**Falsifier.** If no game has activated a second checkpoint in 0-1 by 1M steps, run 1M steps from fresh weights
and compare `furthest_checkpoint` and `cells_new`.

### `configs/campaign_0-1.yaml`

- `algo` ppo, `net_arch` [512, 512], `n_steps` 2048 (total), `batch_size` 512, `n_epochs` 5.
- `learning_rate` 2e-4, `gamma` 0.998, `gae_lambda` 0.95, `ent_coef` 0.01, `target_kl` 0.03.
- `num_envs` 5, `timesteps` 20M, `run_name` `campaign_ppo`, `save_every` 50k.

## Tooling

- **`progress.py` / dashboard.** Add a campaign panel:
  - completion rate over the last 50 fresh-start episodes and over all episodes;
  - best and median official time;
  - furthest checkpoint per game;
  - cells explored per episode, deaths per episode, reward parts.

  The Cyber Grind panels stay as they are.
- **`poll_status.py`** logs the new keys to `metrics_log.csv`.
- **`keep_best.py`** gets a `--metric campaign` mode: fresh-start completion rate first, then best official time.
  Kills/min stays the default.
- **`eval.py --level "Level 0-1"`:**
  - fresh loads, deterministic actions, real deaths;
  - prints official time, kills, style, restarts and rank computed from `ranks`;
  - `--record-times` adds the best run to `times.md` (generation history, and the leaderboard if it is a record).
- **`bridge_test.py --campaign`** prints the `campaign` block.
- **`.gitignore`** adds `.tools/` (local ILSpy install).

## Testing

### Without the game: `tests/test_campaign.py`

Runs with plain `python` and under pytest, like the existing tests.

- **Rewards:**
  - a checkpoint pays once per level load;
  - novelty pays only on first entry per episode and falls as 1/sqrt(N + 1);
  - path pays only on a new minimum, and nothing for a partial or missing path;
  - counter changes across a respawn pay nothing;
  - finishing within the cap nets more than timing out.
- **Observations:** the campaign layout is 479 and Cyber Grind stays 448; masks are 0 when exit, checkpoint or
  door are missing; yaw-frame vectors match `yaw_frame`.
- **Episode start:** each fresh-load rule, with a seeded random source.
- **Rank:** thresholds to rank, including restarts and the P case.
- **Weight transfer:** on a random 448-dim input padded with 36 zeros, the transferred policy's latent features
  equal the source's, checked before the action head is scaled.

### In game, before any training (0-1, one game)

1. After the mod builds, `bridge_test.py --campaign` shows an exit (or a documented `null` until its room loads),
   checkpoints, and a path status. `difficulty` reads 3.
2. After the revolver pickup, stepping with `slot` 1 to 5 moves `weapon_slot` to each of those slots, so the full
   arsenal is present.
3. Teleporting to the exit sets `level_over`, stops the timer, and ends the episode with `level_complete` and the
   expected `level_seconds`.
4. A death mid-episode respawns at the checkpoint and the episode continues.
5. A 2-minute random-agent run finishes without errors and gives a steps/s figure for 5 games.
6. Check that checkpoint and exit triggers still fire at 30 fps / frameskip 2 with rendering off. If not, fall
   back to 60/4 or turn rendering on, and record which.

## Risks

- **Exit not yet present.** The exit room may be inactive at load, so `exit` is `null` and the path is `none`
  until it loads. Exploration and milestone rewards carry the agent until then.
- **Respawn traps.** A checkpoint respawn can leave a door locked behind the player. Three stuck episodes in a
  row at the same checkpoint force a fresh load.
- **Throughput.** Campaign scenes are heavier than Cyber Grind, and a level load stalls all 5 games (lockstep).
  Measured in in-game check 5 before starting the 20M run.
- **Rendering off.** A level event that depends on rendering (for example `OnBecameVisible`) might not fire. In-game
  check 6 covers the main triggers. If a later level stalls at one spot, retry that level with rendering on.
- **Cyber Grind habits.** Always firing and walking backwards may slow early navigation. The softened action head
  and the falsifier above cover this.

## Housekeeping

Update `CLAUDE.md` with each change: layout, commands, the mod version and the campaign run's status. Commit and
push to `origin main` after each change, as the project rules require.
