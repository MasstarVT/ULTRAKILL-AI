# Campaign foundation (0-1 pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the agent to finish ULTRAKILL campaign levels with no human demonstrations, and prove it on `Level 0-1` at Violent difficulty.

**Architecture:** The mod reports each level's own structure (the real `FinalPit` exit, checkpoints, locked doors, arena clears, a NavMesh path hint, the official timer) and applies Violent difficulty and a full arsenal in memory only. Python turns that into a 36-value campaign observation block (479 inputs), a reward of time plus milestones plus per-cell novelty plus path progress, and episodes that mostly respawn at each game's frontier checkpoint and continue through deaths. Training starts from the Cyber Grind policy widened to the new inputs.

**Tech Stack:** C# / BepInEx 5 / HarmonyX / Unity 2022.3 (mod), Python 3.12 with Gymnasium, Stable-Baselines3 PPO, NumPy, Tkinter (training side).

**Design spec:** `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md`

**Verification already done on this plan:** every Python task (3-13) was applied step by step in a throwaway copy of the repo with each test run at the point the plan runs it, and the mod tasks (1-2) were compiled there with the plugin copy redirected away from the game folder. All ten test files pass and the mod builds with 0 warnings and 0 errors. Tasks 14 and 15 need the real game and are not dry-run.

---

### Task 0: Amend the spec with the planning refinements

**Files:**
- Modify: `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md` (status line, game facts, observation and config tables, debug additions, speed settings, rewards table, archive name, episode info, retired settings, falsifier, tooling, tests, in-game checks, new "Refinements decided during planning" section)
- Modify: `CLAUDE.md` (campaign status bullet)

This task changes documentation only, so it has no test run. Every "Replace" block below is an exact quote of
the current file; apply each with a single exact-string replacement.

- [ ] **Step 1: Mark the spec as amended**

In `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md`:

Replace:

```markdown
Date: 2026-09-16. Status: approved in brainstorming, awaiting spec review.
```

With:

```markdown
Date: 2026-09-16. Status: approved. Amended the same day with the refinements decided while writing the
implementation plan (listed at the end, and already applied to the sections below).
```

- [ ] **Step 2: Room templates are filtered by ancestry (game facts)**

Replace:

```markdown
- **Room templates.** `CheckPoint.Start()` clones each room in `rooms` and moves the disabled original +10000 on
  X. Searches that include inactive objects must filter those copies out.
```

With:

```markdown
- **Room templates.** `CheckPoint.Start()` clones each room in `rooms` and moves the disabled original +10000 on
  X, relative to wherever the room was, so no X threshold separates templates from live rooms. Searches that
  include inactive objects filter by ancestry instead: an object is a template when it or any ancestor is an
  entry of some checkpoint's `defaultRooms`. A respawn (`ResetRoom`) re-instantiates each template at the live
  room's position.
```

- [ ] **Step 3: EndWaves runs several times per arena (game facts)**

Replace:

```markdown
- **Arenas and doors.** `ActivateNextWave.EndWaves()` runs when an arena's last wave dies. It unlocks its doors
  and opens `doorForward`. `Door.locked`, `Door.open`, `Door.Unlock()`.
```

With:

```markdown
- **Arenas and doors.** `ActivateNextWave.EndWaves()` runs when an arena's last wave dies. It unlocks its doors
  and opens `doorForward`. It is invoked repeatedly (once per door, then a final call that destroys the
  component), so it cannot be counted. `Door.locked`, `Door.open`, `Door.Unlock()`.
```

- [ ] **Step 4: Observation table, `exit` and `checkpoints` rows**

Replace:

```markdown
| `exit` | `{pos}` of the real `FinalPit`, or `null`. Skips `fakeEnd`, `secondPit`, `rankless` and +10000 X templates. |
| `checkpoints` | list of `{id, pos, activated, current}`, template copies excluded |
```

With:

```markdown
| `exit` | `{pos, active}` of the real `FinalPit` (an active one preferred), or `null`. Skips `fakeEnd`, `secondPit`, `rankless` and room templates. |
| `checkpoints` | list of `{id, pos, activated, current}`, room templates excluded. `id` is the position key (see `cleared_arenas`). |
```

- [ ] **Step 5: Observation table, arena enemies and the milestone keys that replace the counters**

Replace:

```markdown
| `arena_enemies_alive` | live enemies whose parent `ActivateNextWave` is activated and not finished |
| `arenas_cleared`, `doors_unlocked` | counters incremented by Harmony postfixes on `ActivateNextWave.EndWaves` and `Door.Unlock`. They reset to 0 on scene load. |
```

With:

```markdown
| `arena_enemies_alive` | live enemies under an `ActivateNextWave` whose wave is not cleared yet (component present, `activated` false) |
| `cleared_arenas`, `unlocked_doors` | lists of position keys (`"x,y,z"`, whole metres) recorded by Harmony prefixes on `ActivateNextWave.EndWaves` and `Door.Unlock` (doors that were locked). Cleared on scene load. A checkpoint respawn re-creates rooms at the same positions, so an arena cleared again after a death gives the same key and cannot pay twice. |
```

- [ ] **Step 6: Config settings table**

Replace:

```markdown
| `difficulty` | unset | if set, a Harmony prefix on `PrefsManager.GetInt` returns it for `"difficulty"` while the AI has control |
| `unlock_all_gear` | false | `GameProgressSaver.CheckGear` returns 1, and `weapon.*` prefs read as 1, while the AI has control |
```

With:

```markdown
| `difficulty` | -1 (keep the game's setting) | if 0 or more, a Harmony postfix on `PrefsManager.GetInt` returns it for `"difficulty"` while the AI has control |
| `unlock_all_gear` | false | a `GameProgressSaver.CheckGear` result of 0, and a `weapon.<name>` pref of 0, read as 1 while the AI has control (2, the alternate version, is kept) |
```

- [ ] **Step 7: Add `slot_counts`, the `kill` command and the restart guard**

Replace:

```markdown
prefs writes are already skipped while the AI has control (`InstancePatches`).

### Resets
```

With:

```markdown
prefs writes are already skipped while the AI has control (`InstancePatches`).

### Debug additions

- `player.slot_counts`: the number of weapons in each of `GunControl.slots`, slot 1 first. The in-game gear
  check reads it instead of cycling weapons.
- `kill` request: a lethal `NewMovement.GetHurt(999)` through the normal damage path, for the in-game death
  check. With `soft_death` on it heals instead, like any other lethal hit.

While the AI has control the mod also blocks every `StatsManager.Restart` call the bridge did not make. Outside
Cyber Grind the game restarts by itself when a dead player presses Fire1 (or R). That would respawn the player, or
reload the level when there is no checkpoint, in the middle of a step before Python has seen the death.

### Resets
```

- [ ] **Step 8: Docs line**

Replace:

```markdown
Update `docs/protocol.md` (new block and config keys) and `docs/game-internals.md` (the facts above).
```

With:

```markdown
Update `docs/protocol.md` (new block, config keys, the `kill` request and `player.slot_counts`) and
`docs/game-internals.md` (the facts above).
```

- [ ] **Step 9: Campaign pitch clamp off**

Replace:

```markdown
- **Speed settings.** Same as Cyber Grind: `fixed_fps` 30, `frameskip` 2, `render` false, and 368x207
  windows. `soft_death` is off (real deaths drive respawns).
```

With:

```markdown
- **Speed settings.** Same as Cyber Grind: `fixed_fps` 30, `frameskip` 2, `render` false, and 368x207
  windows. `soft_death` is off (real deaths drive respawns). `pitch_limit_deg` is 0 (off): levels need the
  camera to look up and down.
```

- [ ] **Step 10: Rewards table, arena and door rows pay keys once per level load**

Replace:

```markdown
| `arena_clear` | +10 | per increment of `arenas_cleared` |
| `door_unlock` | +3 | per increment of `doors_unlocked`. Counter changes across a reset or respawn are ignored. |
```

With:

```markdown
| `arena_clear` | +10 | per new key in `cleared_arenas`, once per level load |
| `door_unlock` | +3 | per new key in `unlocked_doors`, once per level load. Keys that appear during a reset or respawn are marked paid without paying. |
```

- [ ] **Step 11: Rewards table, style row, and the archive file name**

Replace:

```markdown
| `damage_taken`, `death` | -0.01 per HP, -5 | existing computation |

The exploration archive (cell visit counts) is kept in memory per environment and per level, and saved next
to the checkpoints (`models/<run>/explore_<env>.npz`) so resuming keeps it.
```

With:

```markdown
| `damage_taken`, `death` | -0.01 per HP, -5 | existing computation |
| `style` | 0 | off: the campaign is judged on time, not style |

The exploration archive (cell visit counts) is kept in memory per environment and per level, and saved next
to the checkpoints (`models/<run>/explore_<level>_<port>.npz`) so resuming keeps it.
```

- [ ] **Step 12: Episode info, `level_seconds` for fresh starts and `checkpoints_level`**

Replace:

```markdown
- `completed` (0/1), `fresh_start` (0/1), `level_seconds` (official time, set on completion).
- `checkpoints_reached`, `furthest_checkpoint` (checkpoint index in hierarchy order).
```

With:

```markdown
- `completed` (0/1), `fresh_start` (0/1), `level_seconds` (official time, set only when a fresh-start episode
  completes: the timer carries across respawn episodes, so a respawn episode's time is not an official run).
- `checkpoints_level`: distinct checkpoints activated in the current level load. The mod has no reliable
  checkpoint order, so there is no "furthest checkpoint".
```

- [ ] **Step 13: Retired settings still load**

Replace:

```markdown
existed only for human-recorded routes. Delete them (git history keeps them). `configs/campaign_0-1.yaml` is
rewritten.
```

With:

```markdown
existed only for human-recorded routes. Delete them (git history keeps them). `configs/campaign_0-1.yaml` is
rewritten. `EnvConfig.from_dict` ignores unknown keys, so an old `env_config.yaml` that still carries retired
settings (`checkpoint_resets`, `stuck_steps`, `route_dir`, `route_point`, `stuck`) keeps loading.
```

- [ ] **Step 14: Falsifier reads `checkpoints_level`**

Replace:

```markdown
**Falsifier.** If no game has activated a second checkpoint in 0-1 by 1M steps, run 1M steps from fresh weights
and compare `furthest_checkpoint` and `cells_new`.
```

With:

```markdown
**Falsifier.** If no game has activated a second checkpoint in 0-1 by 1M steps (`best_checkpoints_level` < 2),
run 1M steps from fresh weights and compare `checkpoints_level` and `cells_new`.
```

- [ ] **Step 15: Dashboard panel bullet**

Replace:

```markdown
  - furthest checkpoint per game;
```

With:

```markdown
  - checkpoints activated in the current level load per game (`checkpoints_level`), and the best over all
    episodes (`best_checkpoints_level`);
```

- [ ] **Step 16: `poll_status.py` keeps the CSV header**

Replace:

```markdown
- **`poll_status.py`** logs the new keys to `metrics_log.csv`.
```

With:

```markdown
- **`poll_status.py`** logs the new keys to `metrics_log.csv`. An existing CSV keeps its header
  (`DictWriter(..., extrasaction="ignore")`), so new columns never misalign an old `metrics_log.csv`; they
  appear once a run starts a new file.
```

- [ ] **Step 17: Reward test bullet**

Replace:

```markdown
  - counter changes across a respawn pay nothing;
```

With:

```markdown
  - keys that appear during a respawn pay nothing, and an arena cleared again after a death pays nothing;
```

- [ ] **Step 18: In-game checks 2 to 4**

Replace:

```markdown
2. After the revolver pickup, stepping with `slot` 1 to 5 moves `weapon_slot` to each of those slots, so the full
   arsenal is present.
3. Teleporting to the exit sets `level_over`, stops the timer, and ends the episode with `level_complete` and the
   expected `level_seconds`.
4. A death mid-episode respawns at the checkpoint and the episode continues.
```

With:

```markdown
2. `player.slot_counts` shows at least one weapon in each of slots 1 to 5, so the full arsenal is present. 0-1
   has no weapons until the revolver pickup, so this is read on `Level 1-1`.
3. Teleporting to the exit sets `level_over`, stops the timer, and ends a fresh-start episode with
   `level_complete` and `level_seconds` equal to the block's `seconds`.
4. A death mid-episode (the `kill` request) respawns at the checkpoint and the episode continues.
```

- [ ] **Step 19: Tooling, the `.gitignore` bullet (`.tools/` is already ignored)**

Replace:

```markdown
- **`.gitignore`** adds `.tools/` (local ILSpy install).
```

With:

```markdown
- **`.gitignore`** already ignores `.tools/` (local ILSpy install). It adds the campaign run's numbered checkpoints
  (`python/models/campaign_ppo/ckpt_*.zip`) and `python/models/campaign_smoke/`.
```

- [ ] **Step 20: Testing, the weight-transfer input sizes (448 + 36 is not 479)**

Replace:

```markdown
- **Weight transfer:** on a random 448-dim input padded with 36 zeros, the transferred policy's latent features
  equal the source's, checked before the action head is scaled.
```

With:

```markdown
- **Weight transfer** (`tests/test_transfer.py`): random values in the 443 shared inputs, followed by 5 zeros for
  the 448-input source and 36 zeros for the 479-input transferred policy, give the same latent features in both.
```

- [ ] **Step 21: Add the refinements section before Housekeeping**

Replace:

```markdown
## Housekeeping
```

With:

```markdown
## Refinements decided during planning

Decided on 2026-09-16 while writing the implementation plan, and applied to the sections above.

1. Arena clears and door unlocks are reported as lists of rounded-position keys (`cleared_arenas`,
   `unlocked_doors`), not counters. A checkpoint respawn re-instantiates rooms at the same position, so the same
   arena re-cleared after a death yields the same key and cannot pay twice. Python pays each key once per level
   load, and anything that appears during a reset or respawn is marked paid without paying.
2. `furthest_checkpoint` is replaced by `checkpoints_level`: distinct checkpoints activated in the current level
   load (the mod has no reliable checkpoint order).
3. Room templates are filtered by ancestry (an object under any `CheckPoint.defaultRooms` entry), not by an X
   threshold: `CheckPoint.Start` moves templates +10000 relative to their original X, which can be anywhere.
4. The mod adds `player.slot_counts` (weapons in each slot, for the gear check) and a `kill` debug command (for
   the in-game death check).
5. `level_seconds` is only reported for fresh-start completions: the timer carries across respawn episodes.
6. `poll_status.py` keeps an existing CSV's header (`extrasaction="ignore"`), so new columns never misalign an
   old `metrics_log.csv`.
7. `EnvConfig.from_dict` ignores unknown keys, so old `env_config.yaml` files with retired settings still load.
8. Campaign style reward is 0 (speed matters, not style), `pitch_limit_deg` 0 (off) for the campaign.

Also aligned with the plan while applying these: `arena_enemies_alive` counts enemies of waves not yet cleared,
the `difficulty` config default is -1 and both config patches are Harmony postfixes, the exploration archive file
is named per level and port, in-game checks 2 to 4 name the observation and request they use, the `.gitignore`
bullet lists the campaign entries (`.tools/` was already ignored), and the weight-transfer test uses the real
input sizes (443 shared inputs, then 5 or 36 zeros). Plan review added one mod behaviour: game-initiated
`StatsManager.Restart` calls are blocked while the AI has control (see Debug additions).

## Housekeeping
```

- [ ] **Step 22: Check that no retired name is left outside the refinements list**

Run (PowerShell):

```powershell
Set-Location F:\Github\ULTRAKILL-AI
Select-String -Path docs\superpowers\specs\2026-09-16-campaign-foundation-design.md -Pattern 'arenas_cleared', 'doors_unlocked', 'furthest_checkpoint', 'checkpoints_reached', 'X templates', 'awaiting spec review' | ForEach-Object { "$($_.LineNumber): $($_.Line)" }
```

Expected: exactly one line, the refinements list entry (line number about 305):

```
305: 2. `furthest_checkpoint` is replaced by `checkpoints_level`: distinct checkpoints activated in the current level
```

- [ ] **Step 23: Update CLAUDE.md**

In `CLAUDE.md` (Status, Campaign bullet):

Replace:

```markdown
  - `decompiled/` was regenerated on the second PC for this (ilspycmd 9.1.0.7988 in `.tools/`).
```

With:

```markdown
  - `decompiled/` was regenerated on the second PC for this (ilspycmd 9.1.0.7988 in `.tools/`).
  - Spec amended during planning (2026-09-16): arena clears and door unlocks are position keys paid once per level
    load, `checkpoints_level` replaces `furthest_checkpoint`, room templates are filtered by `defaultRooms`
    ancestry, `level_seconds` is only reported for fresh starts, the campaign style reward is 0, and the mod blocks
    game-initiated restarts while the AI has control.
```

- [ ] **Step 24: Commit and push**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add docs/superpowers/specs/2026-09-16-campaign-foundation-design.md CLAUDE.md
git commit -m "Amend the campaign spec with the planning refinements" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the commit lists 2 files changed, and the push ends with `main -> main`.

### Task 1: Mod: campaign config patches, milestone keys, `kill` command, `slot_counts`

**Files:**
- Create: `mod/UltrakillAIBridge/Env/CampaignPatches.cs`
- Modify: `mod/UltrakillAIBridge/Plugin.cs` (register `CampaignPatches`, clear its keys on scene load)
- Modify: `mod/UltrakillAIBridge/Env/EpisodeController.cs` (`difficulty` and `unlock_all_gear` config keys, `kill` command, `BridgeRestart` around its own checkpoint restart)
- Modify: `mod/UltrakillAIBridge/Obs/ObservationBuilder.cs` (`player.slot_counts`)
- Modify: `CLAUDE.md` (Layout line for the new file)
- Test: none. The mod has no unit tests; this task's check is a clean Release build with the game closed. The
  in-game behaviour (difficulty 3 read back, arsenal present, keys appearing, `kill` respawn) is verified in Task 14.
  `docs/protocol.md` is updated in Task 2 together with the campaign block.

- [ ] **Step 1: Confirm the game is closed and the game members exist**

```powershell
@(Get-Process ULTRAKILL -ErrorAction SilentlyContinue).Count
Set-Location F:\Github\ULTRAKILL-AI
Select-String -Path decompiled\PrefsManager.cs, decompiled\GameProgressSaver.cs, decompiled\ActivateNextWave.cs, decompiled\Door.cs, decompiled\GunControl.cs, decompiled\NewMovement.cs, decompiled\StatsManager.cs -Pattern 'public int GetInt\(string key', 'public static int CheckGear\(string gear\)', 'private void EndWaves\(\)', 'public void Unlock\(\)', 'public bool locked;', 'public List<List<GameObject>> slots', 'public void GetHurt\(int damage', 'public void Restart\(\)' | ForEach-Object { "$($_.Filename): $($_.Line.Trim())" }
```

Expected: `0`, then these 8 lines. If the count is not 0, stop the games first (from `python\`:
`.venv\Scripts\python scripts\games.py stop`), because the build cannot overwrite a DLL the game has loaded.

```
PrefsManager.cs: public int GetInt(string key, int fallback = 0)
GameProgressSaver.cs: public static int CheckGear(string gear)
ActivateNextWave.cs: private void EndWaves()
Door.cs: public bool locked;
Door.cs: public void Unlock()
GunControl.cs: public List<List<GameObject>> slots = new List<List<GameObject>>();
NewMovement.cs: public void GetHurt(int damage, bool invincible, float scoreLossMultiplier = 1f, bool explosion = false, bool instablack = false, float hardDamageMultiplier = 0.35f, bool ignoreInvincibility = false)
StatsManager.cs: public void Restart()
```

- [ ] **Step 2: Create `mod/UltrakillAIBridge/Env/CampaignPatches.cs`**

```csharp
using System;
using System.Collections.Generic;
using HarmonyLib;
using UnityEngine;

namespace UltrakillAIBridge.Env
{
    /// <summary>
    /// Campaign support, active only while the AI has control.
    ///
    /// Config: <see cref="DifficultyOverride"/> replaces the difficulty the game reads from its prefs, and
    /// <see cref="UnlockAllGear"/> makes every weapon, variant and arm read as owned and switched on. Both live
    /// in memory only (prefs and save writes are skipped while the AI has control, see InstancePatches) and
    /// must be sent before the level loads, because enemies and GunSetter read them at scene start.
    ///
    /// Milestones: an arena's last wave clearing (ActivateNextWave.EndWaves) and a locked door unlocking
    /// (Door.Unlock) are recorded as rounded-position keys, not counters. A checkpoint respawn re-creates the
    /// rooms it owns at the same positions, so an arena cleared again after a death gives the same key and
    /// Python pays each key once per level load. EndWaves is invoked several times per arena (once per door,
    /// then a final call), which a set absorbs. Both sets are cleared when a scene loads.
    ///
    /// Respawns: while the AI has control only the bridge's own checkpoint reset may call StatsManager.Restart.
    /// Outside Cyber Grind, StatsManager.Update restarts by itself when the player is dead and Fire1 is newly
    /// pressed (or R), and the pause menu can restart too; that would respawn the player, or reload the level
    /// when there is no checkpoint, in the middle of a step without Python ever seeing the death.
    /// </summary>
    [HarmonyPatch]
    internal static class CampaignPatches
    {
        /// <summary>Difficulty the game reads while the AI has control (0 Harmless .. 4 Brutal); -1 leaves the game's setting.</summary>
        internal static int DifficultyOverride = -1;
        internal static bool UnlockAllGear;

        /// <summary>True only while EpisodeController itself calls StatsManager.Restart (reset with checkpoint=true).</summary>
        internal static bool BridgeRestart;

        private static readonly HashSet<string> clearedArenas = new HashSet<string>();
        private static readonly HashSet<string> unlockedDoors = new HashSet<string>();

        internal static IReadOnlyCollection<string> ClearedArenas => clearedArenas;
        internal static IReadOnlyCollection<string> UnlockedDoors => unlockedDoors;

        /// <summary>Position key shared with Python: whole metres, "x,y,z" (also used as checkpoint ids).</summary>
        internal static string Key(Vector3 p) =>
            FormattableString.Invariant($"{Mathf.RoundToInt(p.x)},{Mathf.RoundToInt(p.y)},{Mathf.RoundToInt(p.z)}");

        /// <summary>Called on every single-mode scene load (subscribed in Plugin).</summary>
        internal static void OnSceneLoaded()
        {
            clearedArenas.Clear();
            unlockedDoors.Clear();
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(PrefsManager), nameof(PrefsManager.GetInt))]
        private static void OverridePrefInt(string key, ref int __result)
        {
            if (!EpisodeController.InControl || key == null) return;
            if (key == "difficulty" && DifficultyOverride >= 0)
            {
                __result = DifficultyOverride;
            }
            else if (UnlockAllGear && __result == 0 && key.StartsWith("weapon.", StringComparison.Ordinal) && key.IndexOf('.', 7) < 0)
            {
                // "weapon.rev0" style keys: 0 = switched off. 2 (the alternate version) is left alone.
                __result = 1;
            }
        }

        [HarmonyPostfix]
        [HarmonyPatch(typeof(GameProgressSaver), nameof(GameProgressSaver.CheckGear))]
        private static void UnlockGear(ref int __result)
        {
            if (EpisodeController.InControl && UnlockAllGear && __result == 0) __result = 1;
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(ActivateNextWave), "EndWaves")]
        private static void RecordArenaClear(ActivateNextWave __instance)
        {
            if (EpisodeController.InControl && __instance != null) clearedArenas.Add(Key(__instance.transform.position));
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(Door), nameof(Door.Unlock))]
        private static void RecordDoorUnlock(Door __instance)
        {
            if (EpisodeController.InControl && __instance != null && __instance.locked) unlockedDoors.Add(Key(__instance.transform.position));
        }

        [HarmonyPrefix]
        [HarmonyPatch(typeof(StatsManager), nameof(StatsManager.Restart))]
        private static bool OnlyBridgeRestarts() => !EpisodeController.InControl || BridgeRestart;
    }
}
```

Notes for the reviewer: `EndWaves` is private, so it is patched by name. `Door.Unlock` records only doors that
were locked, so re-unlocking an open door adds nothing. `FormattableString.Invariant` keeps the key's minus sign
ASCII whatever the Windows culture. The key format is the contract `"x,y,z"` in whole metres.

`OnlyBridgeRestarts` is one addition beyond the contract's four patches. `StatsManager.Update` (outside Cyber Grind)
calls `Restart()` when `nm.hp <= 0` and `Fire1.WasPerformedThisFrame` (or legacy R). Held buttons stay down across
steps, so every step that starts firing after a step without fire is a new press on its first frame. A death on that
same frame (damage in `FixedUpdate` or an earlier `Update`) would respawn the player mid-step. With no checkpoint
it would reload the scene mid-step instead. Python would see neither the death nor its penalty. The other callers
of `Restart` are the pause menu's `RestartCheckpoint`, the debug console and `PlatformerMovement` (not a main level).
The bridge's own call in `BeginReset` sets `BridgeRestart` (Step 9).

- [ ] **Step 3: `Plugin.cs`, import the scene manager**

In `mod/UltrakillAIBridge/Plugin.cs`:

Replace:

```csharp
using UltrakillAIBridge.Net;
using UnityEngine;
```

With:

```csharp
using UltrakillAIBridge.Net;
using UnityEngine;
using UnityEngine.SceneManagement;
```

- [ ] **Step 4: `Plugin.cs`, register the patches and clear the keys on scene load**

Replace:

```csharp
            harmony.PatchAll(typeof(TrainingSpeed));
            EnemyTracker.onEnemyAdded += TrainingSpeed.OnEnemyAdded;
```

With:

```csharp
            harmony.PatchAll(typeof(TrainingSpeed));
            harmony.PatchAll(typeof(CampaignPatches));
            EnemyTracker.onEnemyAdded += TrainingSpeed.OnEnemyAdded;

            // Arena and door keys belong to one level load. Checkpoint respawns don't load a scene, so the
            // keys survive them, which is what lets Python ignore an arena cleared again after a death.
            SceneManager.sceneLoaded += (scene, mode) =>
            {
                if (mode == LoadSceneMode.Single) CampaignPatches.OnSceneLoaded();
            };
```

- [ ] **Step 5: `EpisodeController.cs`, list the commands in the class comment**

In `mod/UltrakillAIBridge/Env/EpisodeController.cs`:

Replace:

```csharp
    ///   step                    - takes control, runs one step, replies with an obs
    ///   release                 - gives control back to the human
```

With:

```csharp
    ///   step                    - takes control, runs one step, replies with an obs
    ///   teleport, kill          - take control, reply with an obs straight away (kill is a debug command)
    ///   release                 - gives control back to the human
```

- [ ] **Step 6: `EpisodeController.cs`, add the `kill` command**

Replace:

```csharp
                    Send(observer.Build(step, "teleport"));
                    break;
```

With:

```csharp
                    Send(observer.Build(step, "teleport"));
                    break;

                case "kill":
                    TakeControl(incoming.ClientId);
                    Kill();
                    Send(observer.Build(step, "kill"));
                    break;
```

- [ ] **Step 7: `EpisodeController.cs`, read the two campaign config keys**

Replace:

```csharp
            TrainingSpeed.SoftDeathEnabled = msg["soft_death"]?.Value<bool>() ?? TrainingSpeed.SoftDeathEnabled;
```

With:

```csharp
            TrainingSpeed.SoftDeathEnabled = msg["soft_death"]?.Value<bool>() ?? TrainingSpeed.SoftDeathEnabled;
            CampaignPatches.DifficultyOverride = msg["difficulty"]?.Value<int>() ?? CampaignPatches.DifficultyOverride;
            CampaignPatches.UnlockAllGear = msg["unlock_all_gear"]?.Value<bool>() ?? CampaignPatches.UnlockAllGear;
```

- [ ] **Step 8: `EpisodeController.cs`, the `Kill` helper (next to `Teleport`)**

Replace:

```csharp
            Physics.SyncTransforms();
        }
```

With:

```csharp
            Physics.SyncTransforms();
        }

        /// <summary>
        /// Debug command for the in-game death check: a lethal hit through the normal damage path. With
        /// soft_death on, TrainingSpeed heals it instead and counts a soft death. GetHurt ignores it once the
        /// level is over.
        /// </summary>
        private static void Kill()
        {
            var nm = MonoSingleton<NewMovement>.Instance;
            if (nm == null || nm.dead) throw new InvalidOperationException("kill needs a living player");
            nm.GetHurt(999, invincible: false, ignoreInvincibility: true);
        }
```

- [ ] **Step 9: `EpisodeController.cs`, let the bridge's own checkpoint restart through**

Replace:

```csharp
                UnpauseIfNeeded();
                sm.Restart();
                sceneRequested = true;
```

With:

```csharp
                UnpauseIfNeeded();
                CampaignPatches.BridgeRestart = true; // CampaignPatches blocks every other Restart while in control
                try
                {
                    sm.Restart();
                }
                finally
                {
                    CampaignPatches.BridgeRestart = false;
                }
                sceneRequested = true;
```

- [ ] **Step 10: `ObservationBuilder.cs`, report `slot_counts`**

In `mod/UltrakillAIBridge/Obs/ObservationBuilder.cs`:

Replace:

```csharp
                ["weapon_variation"] = gun != null ? gun.currentVariationIndex : -1,
```

With:

```csharp
                ["weapon_variation"] = gun != null ? gun.currentVariationIndex : -1,
                ["slot_counts"] = SlotCounts(gun),
```

- [ ] **Step 11: `ObservationBuilder.cs`, the `SlotCounts` helper**

Replace:

```csharp
        private JArray BuildEnemies(Transform cam, int envMask)
```

With:

```csharp
        /// <summary>Weapons in each slot, slot 1 first (empty until GunControl has started).</summary>
        private static JArray SlotCounts(GunControl gun)
        {
            var arr = new JArray();
            if (gun == null || gun.slots == null) return arr;
            foreach (var slot in gun.slots)
            {
                arr.Add(slot != null ? slot.Count : 0);
            }
            return arr;
        }

        private JArray BuildEnemies(Transform cam, int envMask)
```

- [ ] **Step 12: Build and install the mod**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge; & "C:\Program Files\dotnet\dotnet.exe" build -c Release
```

Expected, at the end of the output:

```
  UltrakillAIBridge -> F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge\bin\Release\UltrakillAIBridge.dll
  Copied UltrakillAIBridge.dll to C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\plugins\UltrakillAIBridge

Build succeeded.
    0 Warning(s)
    0 Error(s)
```

An `MSB3027`/`MSB3021` "being used by another process" error means a game copy is still running (Step 1). A
`CS0103`/`CS0246` error means an edit was missed: compare against Steps 2 to 11.

- [ ] **Step 13: Update CLAUDE.md**

In `CLAUDE.md` (Layout, under `mod/UltrakillAIBridge/`):

Replace:

```markdown
  - `Env/InstancePatches.cs`: training instances (`-aibridge-port N`) open prefs read-only and skip prefs and save writes; save writes are also skipped whenever the AI has control.
```

With:

```markdown
  - `Env/InstancePatches.cs`: training instances (`-aibridge-port N`) open prefs read-only and skip prefs and save writes; save writes are also skipped whenever the AI has control.
  - `Env/CampaignPatches.cs`: campaign config held in memory while the AI has control (`difficulty` override; `unlock_all_gear` makes every weapon, variant and arm read as owned); arena clears (`ActivateNextWave.EndWaves`) and door unlocks (`Door.Unlock`) recorded as rounded-position keys, cleared on every scene load; and every `StatsManager.Restart` not made by the bridge blocked while the AI has control (the game restarts by itself when a dead player presses Fire1, which would respawn mid-step without Python seeing the death).
```

- [ ] **Step 14: Commit and push**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add mod/UltrakillAIBridge/Env/CampaignPatches.cs mod/UltrakillAIBridge/Plugin.cs mod/UltrakillAIBridge/Env/EpisodeController.cs mod/UltrakillAIBridge/Obs/ObservationBuilder.cs CLAUDE.md
git commit -m "Mod: campaign config patches, milestone keys, kill command, slot_counts" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: 5 files changed (1 created), and the push ends with `main -> main`. `git status --short mod` shows
nothing afterwards (`bin/` and `obj/` are ignored).

### Task 2: Mod: campaign observation block, AIModule reference, v0.5.0, protocol and internals docs

**Files:**
- Create: `mod/UltrakillAIBridge/Obs/CampaignObserver.cs`
- Modify: `mod/UltrakillAIBridge/UltrakillAIBridge.csproj` (reference `UnityEngine.AIModule`)
- Modify: `mod/UltrakillAIBridge/Obs/ObservationBuilder.cs` (build the `campaign` block, `Vec` internal)
- Modify: `mod/UltrakillAIBridge/Plugin.cs` (`Version` 0.5.0)
- Modify: `docs/protocol.md` (`kill` request, `difficulty` and `unlock_all_gear` config keys, `slot_counts`, the `campaign` block)
- Modify: `docs/game-internals.md` (campaign game facts)
- Modify: `CLAUDE.md` (Layout lines, Status "Mod:" line, Campaign status heading)
- Test: none. The mod has no unit tests; this task's check is a clean Release build with the game closed. The
  block's content in a real level (exit, checkpoints, path status, keys, `difficulty` 3) is verified in Task 14.

- [ ] **Step 1: Confirm the game is closed, the game members exist and AIModule is present**

```powershell
@(Get-Process ULTRAKILL -ErrorAction SilentlyContinue).Count
Set-Location F:\Github\ULTRAKILL-AI
Select-String -Path decompiled\FinalPit.cs, decompiled\CheckPoint.cs, decompiled\StatsManager.cs, decompiled\ActivateNextWave.cs, decompiled\GameStateManager.cs -Pattern 'public bool (rankless|secondPit|fakeEnd|activated|lastWave|timer|levelStarted);', 'public List<GameObject> defaultRooms', 'public CheckPoint currentCheckPoint', 'public int (levelNumber|restarts);', 'public int\[\] (timeRanks|killRanks|styleRanks);', 'public bool PlayerInputLocked' | ForEach-Object { "$($_.Filename): $($_.Line.Trim())" }
Test-Path "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\ULTRAKILL_Data\Managed\UnityEngine.AIModule.dll"
```

Expected: `0` (otherwise stop the games as in Task 1 Step 1), then these 16 lines, then `True`:

```
FinalPit.cs: public bool rankless;
FinalPit.cs: public bool secondPit;
FinalPit.cs: public bool fakeEnd;
CheckPoint.cs: public bool activated;
CheckPoint.cs: public List<GameObject> defaultRooms = new List<GameObject>();
StatsManager.cs: public CheckPoint currentCheckPoint;
StatsManager.cs: public int levelNumber;
StatsManager.cs: public int restarts;
StatsManager.cs: public bool timer;
StatsManager.cs: public bool levelStarted;
StatsManager.cs: public int[] timeRanks;
StatsManager.cs: public int[] killRanks;
StatsManager.cs: public int[] styleRanks;
ActivateNextWave.cs: public bool lastWave;
ActivateNextWave.cs: public bool activated;
GameStateManager.cs: public bool PlayerInputLocked { get; private set; }
```

- [ ] **Step 2: Reference `UnityEngine.AIModule` (NavMesh)**

In `mod/UltrakillAIBridge/UltrakillAIBridge.csproj`:

Replace:

```xml
    <Reference Include="UnityEngine.PhysicsModule"><HintPath>$(ManagedDir)\UnityEngine.PhysicsModule.dll</HintPath><Private>false</Private></Reference>
```

With:

```xml
    <Reference Include="UnityEngine.PhysicsModule"><HintPath>$(ManagedDir)\UnityEngine.PhysicsModule.dll</HintPath><Private>false</Private></Reference>
    <Reference Include="UnityEngine.AIModule"><HintPath>$(ManagedDir)\UnityEngine.AIModule.dll</HintPath><Private>false</Private></Reference>
```

- [ ] **Step 3: Create `mod/UltrakillAIBridge/Obs/CampaignObserver.cs`**

```csharp
using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UltrakillAIBridge.Env;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

namespace UltrakillAIBridge.Obs
{
    /// <summary>
    /// Builds the obs "campaign" block in the 35 main levels: the exit, checkpoints, a NavMesh path hint to
    /// the exit, locked doors, arena enemies, milestone keys and rank thresholds (keys in docs/protocol.md).
    ///
    /// The exit and later checkpoints can sit in rooms that start inactive, so scene objects are found with
    /// FindObjectsOfType(includeInactive). That also returns the disabled room templates CheckPoint.Start
    /// keeps (moved +10000 on X from wherever the room was, so no X threshold separates them), so anything
    /// under an entry of some checkpoint's defaultRooms is skipped. The search is cached and repeated every
    /// <see cref="RescanEvery"/> builds, on a new scene, after a checkpoint respawn (which destroys and
    /// re-creates the rooms it owns), and when a checkpoint changes state: CheckPoint.Start and a repeated
    /// ActivateCheckPoint (InheritRoom) turn live rooms into templates, and a stale cache would report the
    /// objects inside them at +10000 X under new ids.
    /// </summary>
    public sealed class CampaignObserver
    {
        private const int RescanEvery = 30;
        private const int PathEvery = 4;
        private const int MaxLockedDoors = 4;
        private const float PlayerSnapDistance = 6f;
        private const float ExitSnapDistance = 20f;
        private const float CornerReachedDistance = 1.5f;

        private readonly List<FinalPit> pits = new List<FinalPit>();
        private readonly List<CheckPoint> checkpoints = new List<CheckPoint>();
        private readonly List<Door> doors = new List<Door>();
        private readonly HashSet<Transform> templates = new HashSet<Transform>();
        private readonly List<(Door door, float dist)> lockedDoors = new List<(Door, float)>();

        private bool scanned;
        private int sceneHandle;
        private int lastRestarts;
        private int checkpointSignature;
        private int builds;

        // NavMesh path to the exit, recalculated every PathEvery builds (pathStatus "none" is a cached result too).
        private NavMeshPath navPath;
        private bool pathCached;
        private string pathStatus = "none";
        private float pathLength;
        private Vector3 nextCorner;

        public static bool IsCampaignScene(StatsManager sm)
        {
            var scene = SceneHelper.CurrentScene;
            return sm != null && sm.levelNumber >= 1 && sm.levelNumber <= 35
                   && scene != null && scene.StartsWith("Level ", StringComparison.Ordinal);
        }

        public JObject Build(NewMovement nm, StatsManager sm)
        {
            int handle = SceneManager.GetActiveScene().handle;
            if (!scanned || handle != sceneHandle || sm.restarts != lastRestarts)
            {
                // A new scene, or a checkpoint respawn that re-created rooms and moved the player.
                sceneHandle = handle;
                builds = 0;
                pathCached = false;
            }
            else if (CheckpointSignature(sm) != checkpointSignature)
            {
                // A checkpoint activated, took over rooms or became current: live rooms may now be templates.
                builds = 0;
            }
            if (builds % RescanEvery == 0) Scan(sm);

            var playerPos = nm.transform.position;
            var exit = ChooseExit();
            if (!pathCached || builds % PathEvery == 0) UpdatePath(playerPos, exit);
            builds++;

            var prefs = MonoSingleton<PrefsManager>.Instance;
            var gsm = GameStateManager.Instance;
            return new JObject
            {
                ["mission"] = sm.levelNumber,
                ["difficulty"] = prefs != null ? prefs.GetInt("difficulty") : -1,
                ["seconds"] = sm.seconds,
                ["timer_running"] = sm.timer,
                ["level_started"] = sm.levelStarted,
                ["level_over"] = nm.levelOver,
                ["restarts"] = sm.restarts,
                ["input_locked"] = (gsm != null && gsm.PlayerInputLocked) || !nm.activated,
                ["exit"] = exit != null
                    ? new JObject { ["pos"] = ObservationBuilder.Vec(exit.transform.position), ["active"] = exit.gameObject.activeInHierarchy }
                    : null,
                ["checkpoints"] = BuildCheckpoints(sm),
                ["path"] = BuildPath(),
                ["locked_doors"] = BuildLockedDoors(playerPos),
                ["arena_enemies_alive"] = ArenaEnemiesAlive(),
                ["cleared_arenas"] = Strings(CampaignPatches.ClearedArenas),
                ["unlocked_doors"] = Strings(CampaignPatches.UnlockedDoors),
                ["ranks"] = new JObject
                {
                    ["time"] = Ints(sm.timeRanks),
                    ["kills"] = Ints(sm.killRanks),
                    ["style"] = Ints(sm.styleRanks),
                },
            };
        }

        private void Scan(StatsManager sm)
        {
            scanned = true;
            lastRestarts = sm.restarts;

            var allCheckpoints = Object.FindObjectsOfType<CheckPoint>(true);
            templates.Clear();
            foreach (var cp in allCheckpoints)
            {
                if (cp == null || cp.defaultRooms == null) continue;
                foreach (var room in cp.defaultRooms)
                {
                    if (room != null) templates.Add(room.transform);
                }
            }

            checkpoints.Clear();
            foreach (var cp in allCheckpoints)
            {
                if (cp != null && !IsTemplate(cp.transform)) checkpoints.Add(cp);
            }
            pits.Clear();
            foreach (var pit in Object.FindObjectsOfType<FinalPit>(true))
            {
                if (pit != null && !pit.fakeEnd && !pit.secondPit && !pit.rankless && !IsTemplate(pit.transform)) pits.Add(pit);
            }
            doors.Clear();
            foreach (var door in Object.FindObjectsOfType<Door>(true))
            {
                if (door != null && !IsTemplate(door.transform)) doors.Add(door);
            }
            checkpointSignature = CheckpointSignature(sm);
        }

        /// <summary>
        /// Summary of the cached checkpoints' state (current checkpoint, activated flags, owned room counts,
        /// destroyed entries). It only changes on checkpoint events, so comparing it every build is cheap.
        /// </summary>
        private int CheckpointSignature(StatsManager sm)
        {
            unchecked
            {
                int sig = sm.currentCheckPoint != null ? sm.currentCheckPoint.GetInstanceID() : 0;
                foreach (var cp in checkpoints)
                {
                    if (cp == null)
                    {
                        sig = sig * 31 + 1;
                        continue;
                    }
                    sig = sig * 31 + (cp.activated ? 3 : 2);
                    sig = sig * 31 + (cp.defaultRooms != null ? cp.defaultRooms.Count : 0);
                }
                return sig;
            }
        }

        /// <summary>True when the object or any ancestor is a room template kept by a checkpoint.</summary>
        private bool IsTemplate(Transform t)
        {
            for (; t != null; t = t.parent)
            {
                if (templates.Contains(t)) return true;
            }
            return false;
        }

        /// <summary>The real exit; an active pit is preferred over one in a room that hasn't loaded yet.</summary>
        private FinalPit ChooseExit()
        {
            FinalPit inactive = null;
            foreach (var pit in pits)
            {
                if (pit == null) continue;
                if (pit.gameObject.activeInHierarchy) return pit;
                if (inactive == null) inactive = pit;
            }
            return inactive;
        }

        /// <summary>
        /// NavMesh path from the player to the exit, both ends snapped onto the mesh. The mesh only covers
        /// walkable ground (jumps and gaps are not linked, doors carry obstacles), so a partial path is normal.
        /// </summary>
        private void UpdatePath(Vector3 playerPos, FinalPit exit)
        {
            pathCached = true;
            pathStatus = "none";
            if (exit == null) return;
            if (!NavMesh.SamplePosition(playerPos, out var from, PlayerSnapDistance, NavMesh.AllAreas)) return;
            if (!NavMesh.SamplePosition(exit.transform.position, out var to, ExitSnapDistance, NavMesh.AllAreas)) return;

            if (navPath == null) navPath = new NavMeshPath();
            if (!NavMesh.CalculatePath(from.position, to.position, NavMesh.AllAreas, navPath)) return;
            if (navPath.status == NavMeshPathStatus.PathInvalid) return;
            var corners = navPath.corners;
            if (corners.Length == 0) return;

            pathLength = Vector3.Distance(playerPos, corners[0]);
            for (int i = 1; i < corners.Length; i++)
            {
                pathLength += Vector3.Distance(corners[i - 1], corners[i]);
            }
            nextCorner = corners[corners.Length - 1];
            foreach (var corner in corners)
            {
                var flat = corner - playerPos;
                flat.y = 0f;
                if (flat.magnitude > CornerReachedDistance)
                {
                    nextCorner = corner;
                    break;
                }
            }
            pathStatus = navPath.status == NavMeshPathStatus.PathComplete ? "complete" : "partial";
        }

        private JObject BuildPath()
        {
            if (pathStatus == "none") return new JObject { ["status"] = "none" };
            return new JObject
            {
                ["status"] = pathStatus,
                ["length"] = pathLength,
                ["next_corner"] = ObservationBuilder.Vec(nextCorner),
            };
        }

        private JArray BuildCheckpoints(StatsManager sm)
        {
            var arr = new JArray();
            foreach (var cp in checkpoints)
            {
                if (cp == null) continue;
                var pos = cp.transform.position;
                arr.Add(new JObject
                {
                    ["id"] = CampaignPatches.Key(pos),
                    ["pos"] = ObservationBuilder.Vec(pos),
                    ["activated"] = cp.activated,
                    ["current"] = sm.currentCheckPoint == cp,
                });
            }
            return arr;
        }

        private JArray BuildLockedDoors(Vector3 playerPos)
        {
            lockedDoors.Clear();
            foreach (var door in doors)
            {
                if (door == null || !door.locked || !door.gameObject.activeInHierarchy) continue;
                lockedDoors.Add((door, Vector3.Distance(playerPos, door.transform.position)));
            }
            lockedDoors.Sort((a, b) => a.dist.CompareTo(b.dist));

            var arr = new JArray();
            for (int i = 0; i < lockedDoors.Count && i < MaxLockedDoors; i++)
            {
                var (door, dist) = lockedDoors[i];
                arr.Add(new JObject
                {
                    ["pos"] = ObservationBuilder.Vec(door.transform.position),
                    ["dist"] = dist,
                });
            }
            return arr;
        }

        /// <summary>Live enemies belonging to an arena wave that hasn't been cleared yet.</summary>
        private static int ArenaEnemiesAlive()
        {
            var tracker = MonoSingleton<EnemyTracker>.Instance;
            if (tracker == null) return 0;
            int alive = 0;
            foreach (var eid in tracker.GetCurrentEnemies())
            {
                if (eid == null || eid.dead) continue;
                var wave = eid.GetComponentInParent<ActivateNextWave>();
                if (wave != null && !wave.activated) alive++;
            }
            return alive;
        }

        private static JArray Strings(IEnumerable<string> values)
        {
            var arr = new JArray();
            foreach (var value in values)
            {
                arr.Add(value);
            }
            return arr;
        }

        private static JArray Ints(int[] values)
        {
            var arr = new JArray();
            if (values == null) return arr;
            foreach (var value in values)
            {
                arr.Add(value);
            }
            return arr;
        }
    }
}
```

Notes for the reviewer: besides the contract's triggers (every 30 builds, a new scene handle), the cache is also
rebuilt and the path recalculated when `StatsManager.restarts` changes. A checkpoint respawn destroys and
re-instantiates the rooms it owns and moves the player, so without this the reset obs after a respawn could list a
destroyed checkpoint as missing and carry a path measured from where the player died. The cache is also rebuilt
when `CheckpointSignature` changes (a checkpoint activates, becomes current, starts owning rooms or is destroyed).
`CheckPoint.Start` in a room that loads late, and a repeated `ActivateCheckPoint` (its `InheritRoom` path disables
the live room and moves it +10000 on X), turn cached live objects into templates; for up to 30 builds a stale cache
would report an activated checkpoint inside such a room under a new +10000 id, which Python would pay as a new
checkpoint, and could point `exit` at a template. The NavMesh API was checked
against `UnityEngine.AIModule.dll`: `NavMesh.AllAreas` is `-1`, `SamplePosition(Vector3, out NavMeshHit, float,
int)`, `CalculatePath(Vector3, Vector3, int, NavMeshPath)` returns false when no path is found (a partial path
returns true with `PathPartial`).

- [ ] **Step 4: `ObservationBuilder.cs`, own a `CampaignObserver`**

In `mod/UltrakillAIBridge/Obs/ObservationBuilder.cs`:

Replace:

```csharp
        private readonly List<(EnemyIdentifier eid, float dist)> sorted = new List<(EnemyIdentifier, float)>();
```

With:

```csharp
        private readonly List<(EnemyIdentifier eid, float dist)> sorted = new List<(EnemyIdentifier, float)>();
        private readonly CampaignObserver campaign = new CampaignObserver();
```

- [ ] **Step 5: `ObservationBuilder.cs`, add the block in campaign scenes**

Replace:

```csharp
            obs["stats"] = BuildStats(nm);
```

With:

```csharp
            obs["stats"] = BuildStats(nm);

            var sm = MonoSingleton<StatsManager>.Instance;
            if (CampaignObserver.IsCampaignScene(sm)) obs["campaign"] = campaign.Build(nm, sm);
```

- [ ] **Step 6: `ObservationBuilder.cs`, share `Vec`**

Replace:

```csharp
        private static JArray Vec(Vector3 v) => new JArray(v.x, v.y, v.z);
```

With:

```csharp
        internal static JArray Vec(Vector3 v) => new JArray(v.x, v.y, v.z);
```

- [ ] **Step 7: `Plugin.cs`, bump the version**

In `mod/UltrakillAIBridge/Plugin.cs`:

Replace:

```csharp
        public const string Version = "0.4.0";
```

With:

```csharp
        public const string Version = "0.5.0";
```

- [ ] **Step 8: Build and install the mod**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge; & "C:\Program Files\dotnet\dotnet.exe" build -c Release
```

Expected, at the end of the output:

```
  UltrakillAIBridge -> F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge\bin\Release\UltrakillAIBridge.dll
  Copied UltrakillAIBridge.dll to C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\plugins\UltrakillAIBridge

Build succeeded.
    0 Warning(s)
    0 Error(s)
```

`CS0234 The type or namespace name 'AI' does not exist in the namespace 'UnityEngine'` means Step 2 is missing;
`CS0122 'ObservationBuilder.Vec(Vector3)' is inaccessible` means Step 6 is missing.

- [ ] **Step 9: `docs/protocol.md`, the `kill` request**

In `docs/protocol.md` (Requests table):

Replace:

```markdown
| `teleport` | `pos` `[x,y,z]` | an `obs` with `"event":"teleport"` (takes control; moves the player and zeroes velocity) |
```

With:

```markdown
| `teleport` | `pos` `[x,y,z]` | an `obs` with `"event":"teleport"` (takes control; moves the player and zeroes velocity) |
| `kill` | | an `obs` with `"event":"kill"` (takes control; debug: a lethal `NewMovement.GetHurt(999)` for the in-game death check. With `soft_death` on it is healed like any lethal hit; ignored once the level is over; an error when there is no living player). While in control a dead player stays dead until a `reset`: the game's own restarts (Fire1 or R while dead, the pause menu) are blocked |
```

- [ ] **Step 10: `docs/protocol.md`, the two campaign config keys**

Replace:

```markdown
| `command_timeout_s` | 300 | drop the client if no command arrives for this long |
```

With:

```markdown
| `command_timeout_s` | 300 | drop the client if no command arrives for this long |
| `difficulty` | -1 | difficulty the game reads while in control: 0 Harmless, 1 Lenient, 2 Standard, 3 Violent, 4 Brutal; -1 keeps the game's own setting. In memory only. Send it before the level loads: enemies and the player read it at scene start |
| `unlock_all_gear` | false | while in control, every weapon, variant and arm reads as owned and switched on (`GameProgressSaver.CheckGear` 0 and `weapon.<name>` prefs of 0 read as 1). In memory only. Send it before the level loads: `GunSetter` builds the arsenal at scene start |
```

- [ ] **Step 11: `docs/protocol.md`, `slot_counts` in the obs example**

Replace:

```markdown
             "activated": true, "level_over": false, "weapon_slot": 0, "weapon_variation": 1},
```

With:

```markdown
             "activated": true, "level_over": false, "weapon_slot": 0, "weapon_variation": 1,
             "slot_counts": [3, 3, 3, 3, 3, 0]},
```

- [ ] **Step 12: `docs/protocol.md`, describe `slot_counts` and the `campaign` block**

Replace:

```markdown
- **`player`:** `null` when no player exists (e.g. the main menu).
```

With:

````markdown
- **`player`:** `null` when no player exists (e.g. the main menu).
- **`player.slot_counts`:** weapons in each of the six slots, slot 1 first. Empty until `GunControl` has started; 0-1 has none until the revolver pickup.

## campaign

Present only in the 35 main levels (`StatsManager.levelNumber` 1 to 35, in a scene whose name starts with `Level`) when a player exists:

```json
"campaign": {
  "mission": 1, "difficulty": 3, "seconds": 12.3, "timer_running": true, "level_started": true,
  "level_over": false, "restarts": 0, "input_locked": false,
  "exit": {"pos": [x, y, z], "active": true},
  "checkpoints": [{"id": "12,3,-40", "pos": [x, y, z], "activated": false, "current": false}],
  "path": {"status": "complete", "length": 84.2, "next_corner": [x, y, z]},
  "locked_doors": [{"pos": [x, y, z], "dist": 9.5}],
  "arena_enemies_alive": 0,
  "cleared_arenas": ["30,1,5"],
  "unlocked_doors": ["22,0,17"],
  "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}
}
```

- **Position keys** (`checkpoints[].id`, `cleared_arenas`, `unlocked_doors`): `"x,y,z"`, each coordinate rounded to whole metres. Objects that round to the same metre share a key (for example two waves' `ActivateNextWave` components on one GameObject, or wave containers placed at the same point), so only the first of them pays.
- **`mission`:** `StatsManager.levelNumber` (1 = 0-1 through 35 = 9-2).
- **`difficulty`:** the difficulty the game reads right now, so it shows the `difficulty` config override while in control.
- **`seconds`, `timer_running`, `level_started`, `restarts`:** from `StatsManager`. The timer keeps running through checkpoint respawns and cutscenes, and stops when the player enters the exit.
- **`level_over`:** `NewMovement.levelOver`, set on entering the real exit.
- **`input_locked`:** `GameStateManager.PlayerInputLocked` (any registered game state that locks player input: the pause, cheat and spawn menus, the console, and scene objects with `AutoRegisterState`) or the player not activated (the level-start drop, after entering the exit, and while dead).
- **`exit`:** the real `FinalPit`, or `null`. Decoys (`fakeEnd`, `secondPit`, `rankless`) and room templates are skipped, and an active pit is preferred; `active` is false while its room has not loaded.
- **`checkpoints`:** every checkpoint except room templates. `current` marks `StatsManager.currentCheckPoint`.
- **`path`:** NavMesh path from the player to the exit, each end snapped onto the mesh (within 6 m of the player, 20 m of the exit), recalculated every 4 obs. `length` runs from the player through every corner; `next_corner` is the first corner more than 1.5 m away horizontally, else the last. `partial` is common: the mesh does not link jumps or gaps, and doors carry obstacles. `{"status": "none"}` when there is no exit or no path.
- **`locked_doors`:** the nearest 4 active doors that are locked.
- **`arena_enemies_alive`:** live enemies under an `ActivateNextWave` whose wave has not been cleared.
- **`cleared_arenas`, `unlocked_doors`:** keys of arenas whose last wave was cleared (`ActivateNextWave.EndWaves`) and of doors that went from locked to unlocked (`Door.Unlock`), recorded while in control and emptied on every scene load. A checkpoint respawn re-creates rooms at the same positions, so an arena cleared again after a death gives the same key; a respawn also unlocks the checkpoint's doors, which adds keys. Pay each key once per level load.
- **`ranks`:** the 4 thresholds per category from `StatsManager`: `time` in seconds (lower is better), `kills` and `style` (higher is better).
- **Scene objects** are cached and searched again every 30 obs, on a new scene, after a checkpoint respawn, and when a checkpoint activates, becomes current or takes over rooms.
````

- [ ] **Step 13: `docs/game-internals.md`, where game-specific code lives**

In `docs/game-internals.md`:

Replace:

```markdown
When a game update breaks the mod, check these first. Everything game-specific is in
`ObservationBuilder.cs`, `ActionInjector.cs` and `EpisodeController.cs`.
```

With:

```markdown
When a game update breaks the mod, check these first. Everything game-specific is in
`ObservationBuilder.cs`, `CampaignObserver.cs`, `ActionInjector.cs`, `EpisodeController.cs` and the Harmony
patch classes in `Env/`.
```

- [ ] **Step 14: `docs/game-internals.md`, weapon slots and the gear checks**

Replace:

```markdown
- `currentSlotIndex`, `currentVariationIndex`, `slot1..slot6` lists
```

With:

```markdown
- `currentSlotIndex`, `currentVariationIndex`, `slot1..slot6` lists, and `slots` (`List<List<GameObject>>`, the six lists in order, filled in `Start`)
- `GunSetter.CheckWeapon` adds a variant when `PrefsManager.GetInt("weapon." + name, 1)` is above 0 (0 off, 1 on, 2 alternate) and `GameProgressSaver.CheckGear(name) > 0`. `FistControl` needs the pref `== 1` and `CheckGear(name) == 1`. Early levels keep `GunSetter` disabled until the first `WeaponPickUp`.
```

- [ ] **Step 15: `docs/game-internals.md`, StatsManager campaign fields**

Replace:

```markdown
- `StatsManager`: `kills`, `stylePoints`, `seconds`, `restarts`, `infoSent` (true once the level-end screen has sent results), `Restart()` (respawn at checkpoint, or reload if there is none)
```

With:

```markdown
- `StatsManager`: `kills`, `stylePoints`, `seconds`, `restarts`, `infoSent` (true once the level-end screen has sent results), `Restart()` (respawn at checkpoint, or reload if there is none)
  - Campaign fields: `levelNumber` (mission number), `currentCheckPoint`, `timer`, `levelStarted`, `timeRanks`, `killRanks`, `styleRanks`
```

- [ ] **Step 16: `docs/game-internals.md`, the Campaign section**

Replace:

```markdown
## Time: `TimeController`
```

With:

```markdown
## Campaign
- **Exit:** `FinalPit` is a trigger baked into the scene. Entering it sets `NewMovement.levelOver`, clears `activated`, calls `StatsManager.StopTimer()` and sends results 5 s later. Decoys carry `fakeEnd`, `secondPit` or `rankless`. The pit can sit in a room that starts inactive, so the mod searches with `FindObjectsOfType<FinalPit>(true)`.
- **Room templates:** `CheckPoint.Start()` adds each of `rooms` to `defaultRooms`, clones it in place (`newRooms`), disables the original and moves it +10000 on X relative to wherever it was. A respawn (`ResetRoom`) destroys the live copy and instantiates the template again at the live copy's position. Inactive-object searches skip anything under a `defaultRooms` entry.
- **Checkpoints:** `CheckPoint.activated`, `ActivateCheckPoint()`, `StatsManager.currentCheckPoint`. `StatsManager.Restart()` respawns at the current checkpoint and adds 1 to `restarts` (`CheckPoint.OnRespawn` unlocks `doorsToUnlock` and resets the rooms it owns), or reloads the scene when there is none. `startOff` checkpoints start activated.
- **Timer:** `StatsManager.seconds += Time.deltaTime * GameStateManager.Instance.TimerModifier` while `timer`. Respawns do not reset it and cutscenes do not stop it.
- **Arenas:** `ActivateNextWave` (`lastWave`, `activated`, `deadEnemies`, `enemyCount`). Enemies find their wave with `GetComponentInParent<ActivateNextWave>()`. `activated` goes true once `deadEnemies` reaches `enemyCount`. A non-last wave then unlocks its `doors`, spawns the next wave and destroys itself; the last wave invokes the private `EndWaves()` repeatedly (once per entry of `doors`, unlocking it and opening `doorForward`, then a final call that destroys the component).
- **Doors:** `Door.locked`, `Door.open`, `Door.Unlock()` (clears `locked`, opens when `openOnUnlock`). Doors carry a `NavMeshObstacle`.
- **NavMesh:** enemies use `NavMeshAgent`, so campaign scenes have a baked ground mesh (`UnityEngine.AIModule.dll`: `NavMesh.SamplePosition`, `NavMesh.CalculatePath`, `NavMeshPath`). Jumps and gaps are not linked.
- **Level start:** the player drops in under the `"pit-falling"` game state (`PlayerActivatorRelay`), which locks the camera and cursor but not player input; `PlayerActivator` pops it and sets `NewMovement.activated` on landing. `GameStateManager.Instance.PlayerInputLocked` is set only by game states registered with `playerInputLock` (the pause, cheat and spawn menus, the console, `AutoRegisterState` objects), so the campaign block's `input_locked` also checks `!activated`.
- **Difficulty:** `PrefsManager.GetInt(string key, int fallback = 0)` (instance) with `"difficulty"`: 0 Harmless, 1 Lenient, 2 Standard, 3 Violent, 4 Brutal. Enemies and `NewMovement` read it at scene start.
- **Gear:** `GameProgressSaver.CheckGear(string gear)` (static) reads the int field of `GameProgressMoneyAndGear` with that name (`rev0..rev3`, `revalt`, `sho0..3`, `shoalt`, `nai0..3`, `naialt`, `rai0..3`, `rock0..3`, `beam0..3`, `arm1..3`), 0 when missing. `GearCheckEnabler` switches scene objects on owned gear, so with `unlock_all_gear` a level looks the way it does for a player who owns everything. `WeaponPickUp` skips its prefs and save writes when the gear already reads as owned.
- **Death:** `NewMovement.GetHurt(int damage, bool invincible, float scoreLossMultiplier = 1f, bool explosion = false, bool instablack = false, float hardDamageMultiplier = 0.35f, bool ignoreInvincibility = false)` sets `dead` and clears `activated` at 0 hp, and returns early when `dead` or `levelOver`. Outside Cyber Grind, `StatsManager.Update` calls `Restart()` by itself when the player is at 0 hp and Fire1 is newly pressed (or R), unless paused; the pause menu's `RestartCheckpoint` and the debug console call it too. While the AI has control `CampaignPatches` blocks every `Restart` except the bridge's own checkpoint reset, so a death always stays visible until Python respawns.
- **Rank:** `timeRanks`, `killRanks` and `styleRanks` hold 4 thresholds each. Per category, count the thresholds met in order (time: `seconds <= t`; kills and style: `value >= t`), stopping at the first miss; meeting all 4 scores 4. The total is the sum minus `restarts`, floored at 0. 12 with no cheats is P, otherwise `RoundToInt(total / 3)` indexes D, C, B, A, S.
- **Mission numbers** (`GetMissionName`): 1-5 `Level 0-1`..`Level 0-5`, 6-9 `1-1`..`1-4`, 10-13 `2-1`..`2-4`, 14-15 `3-1`, `3-2`, 16-19 `4-1`..`4-4`, 20-23 `5-1`..`5-4`, 24-25 `6-1`, `6-2`, 26-29 `7-1`..`7-4`, 30-33 `8-1`..`8-4`, 34-35 `9-1`, `9-2`.

## Time: `TimeController`
```

- [ ] **Step 17: Update CLAUDE.md, Layout**

In `CLAUDE.md` (Layout, under `mod/UltrakillAIBridge/`):

Replace:

```markdown
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot.
```

With:

```markdown
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot.
  - `Obs/CampaignObserver.cs`: the obs `campaign` block in the 35 main levels (exit, checkpoints, NavMesh path to the exit, locked doors, arena enemies, milestone keys, rank thresholds); room templates are skipped by `CheckPoint.defaultRooms` ancestry.
```

- [ ] **Step 18: Update CLAUDE.md, Status "Mod:" line**

Replace:

```markdown
- **Mod:** v0.4.0 (background play, training instances, teleport, soft death, rendering off). Verified in game: plugin load, handshake, Cyber Grind reset, movement/look/jump/dash, observations (enemies, waves, damage, death), leaderboard block.
```

With:

```markdown
- **Mod:** v0.5.0 (background play, training instances, teleport, soft death, rendering off; campaign block, `difficulty` and `unlock_all_gear` config, `kill` debug command, `player.slot_counts`, game-initiated restarts blocked while in control). Verified in game: plugin load, handshake, Cyber Grind reset, movement/look/jump/dash, observations (enemies, waves, damage, death), leaderboard block. The v0.5.0 campaign additions only have a clean build so far; the in-game campaign check verifies them.
```

- [ ] **Step 19: Update CLAUDE.md, the campaign is no longer "not built yet"**

In `CLAUDE.md` (Layout, the specs line):

Replace:

```markdown
- `docs/superpowers/specs/`: approved design specs. `2026-09-16-campaign-foundation-design.md` is the campaign plan (not built yet).
```

With:

```markdown
- `docs/superpowers/specs/`: approved design specs. `2026-09-16-campaign-foundation-design.md` is the campaign plan (being built: the mod side is done in v0.5.0).
```

In `CLAUDE.md` (Status, the Campaign bullet's first line):

Replace:

```markdown
- **Campaign: designed 2026-09-16, not built yet.** Goal: finish all 35 main levels (`Level 0-1` to `Level 9-2`)
```

With:

```markdown
- **Campaign: designed 2026-09-16, being built (mod side done in v0.5.0).** Goal: finish all 35 main levels (`Level 0-1` to `Level 9-2`)
```

- [ ] **Step 20: Commit and push**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add mod/UltrakillAIBridge/Obs/CampaignObserver.cs mod/UltrakillAIBridge/UltrakillAIBridge.csproj mod/UltrakillAIBridge/Obs/ObservationBuilder.cs mod/UltrakillAIBridge/Plugin.cs docs/protocol.md docs/game-internals.md CLAUDE.md
git commit -m "Mod v0.5.0: campaign observation block, protocol and internals docs" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: 7 files changed (1 created), and the push ends with `main -> main`.

### Task 3: Python `campaign.py`: level list, ranks, exploration archive

**Files:**
- Create: `python/ultrakill_ai/campaign.py`
- Create: `python/tests/test_campaign.py`
- Modify: `CLAUDE.md` (Layout: `campaign.py` and `tests/test_campaign.py` lines; Commands: the no-game test list)
- Test: `python/tests/test_campaign.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_campaign.py` with exactly this content:

```python
"""Campaign helpers (ultrakill_ai/campaign.py). No game needed:  python tests/test_campaign.py  (or pytest)."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CAMPAIGN_LEVELS,
    RANK_LETTERS,
    ExplorationArchive,
    compute_rank,
    grade,
    safe_name,
)

RANKS = {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}


def test_campaign_levels():
    assert len(CAMPAIGN_LEVELS) == 35 and len(set(CAMPAIGN_LEVELS)) == 35
    assert CAMPAIGN_LEVELS[0] == "Level 0-1"
    assert CAMPAIGN_LEVELS[4] == "Level 0-5" and CAMPAIGN_LEVELS[5] == "Level 1-1"
    assert CAMPAIGN_LEVELS[14] == "Level 3-2"
    assert CAMPAIGN_LEVELS[-1] == "Level 9-2"


def test_safe_name():
    assert safe_name("Level 0-1") == "Level_0-1"
    assert safe_name("Level 3-2") == "Level_3-2"
    assert safe_name("a  b/c") == "a_b_c"


def test_grade_counts_thresholds_in_order():
    assert grade([10, 20, 30, 40], 0, reverse=False) == 0
    assert grade([10, 20, 30, 40], 10, reverse=False) == 1  # meeting a threshold exactly counts
    assert grade([10, 20, 30, 40], 25, reverse=False) == 2
    assert grade([300, 240, 180, 120], 301.0, reverse=True) == 0
    assert grade([300, 240, 180, 120], 240.0, reverse=True) == 2
    assert grade([300, 240, 180, 120], 150.5, reverse=True) == 3


def test_grade_stops_at_the_first_miss():
    assert grade([10, 5, 1, 0], 7, reverse=False) == 0  # 5, 1 and 0 are met, but 10 is checked first
    assert grade([100, 50, 300, 400], 60.0, reverse=True) == 1  # 300 and 400 are met after the miss at 50


def test_grade_all_thresholds_met_is_4():
    assert grade([10, 20, 30, 40], 40, reverse=False) == 4
    assert grade([10, 20, 30, 40], 1e6, reverse=False) == 4
    assert grade([300, 240, 180, 120], 12.5, reverse=True) == 4


def test_compute_rank_letters():
    assert RANK_LETTERS == ("D", "C", "B", "A", "S")
    cases = [
        ((400.0, 0, 0), "D"),  # 0 + 0 + 0
        ((300.0, 0, 0), "D"),  # 1: 1/3 rounds down
        ((300.0, 10, 0), "C"),  # 2: 2/3 rounds up
        ((240.0, 20, 1000), "B"),  # 2 + 2 + 1 = 5
        ((180.0, 30, 2000), "A"),  # 3 + 3 + 2 = 8
        ((120.0, 40, 3000), "S"),  # 4 + 4 + 3 = 11
    ]
    for (seconds, kills, style), letter in cases:
        assert compute_rank(seconds, kills, style, 0, RANKS) == letter, (seconds, kills, style)


def test_compute_rank_restarts_lower_the_rank():
    assert compute_rank(180.0, 30, 3000, 0, RANKS) == "A"  # 3 + 3 + 3 = 9
    assert compute_rank(180.0, 30, 3000, 2, RANKS) == "B"  # 7
    assert compute_rank(180.0, 30, 3000, 5, RANKS) == "C"  # 4
    assert compute_rank(400.0, 10, 1000, 9, RANKS) == "D"  # 2 - 9 floors at 0


def test_compute_rank_p_needs_12_without_restarts():
    assert compute_rank(100.0, 45, 5000, 0, RANKS) == "P"
    assert compute_rank(100.0, 45, 5000, 1, RANKS) == "S"  # 12 - 1 = 11


def test_archive_cell_floors_each_axis():
    archive = ExplorationArchive(cell_size=4.0)
    assert archive.cell((0.0, 3.99, 4.0)) == (0, 0, 1)
    assert archive.cell([-0.5, -4.0, -4.01]) == (-1, -1, -2)


def test_archive_novelty_first_entry_per_episode():
    archive = ExplorationArchive(cell_size=4.0)
    archive.start_episode()
    assert archive.visit((1.0, 1.0, 1.0)) == 1.0
    assert archive.visit((3.0, 2.0, 0.5)) == 0.0  # same cell, same episode
    assert archive.visit((5.0, 1.0, 1.0)) == 1.0  # a different cell
    assert archive.episode_cells == 2
    archive.start_episode()
    assert archive.episode_cells == 0
    assert abs(archive.visit((2.0, 2.0, 2.0)) - 1 / math.sqrt(2)) < 1e-12
    assert archive.visit((2.0, 2.0, 2.0)) == 0.0
    archive.start_episode()
    assert abs(archive.visit((2.0, 2.0, 2.0)) - 1 / math.sqrt(3)) < 1e-12
    assert archive.counts[(0, 0, 0)] == 3 and archive.counts[(1, 0, 0)] == 1


def test_archive_features_straight_ahead_and_clockwise():
    archive = ExplorationArchive(cell_size=4.0)
    archive.counts[(0, 0, 1)] = 10**6  # +z of the player's cell; saturates at 1.0
    archive.counts[(1, 0, 0)] = 1  # +x of the player's cell
    one = math.log(2.0) / math.log(1001.0)
    pos = (2.0, 2.0, 2.0)  # the centre of cell (0, 0, 0)

    ahead = archive.features(pos, 0.0)
    assert len(ahead) == 9
    assert ahead[1] == 1.0  # k = 0: yaw 0 faces +z
    assert abs(ahead[3] - one) < 1e-12  # k = 2: 90 degrees to the right of +z is +x
    assert all(v == 0.0 for i, v in enumerate(ahead) if i not in (1, 3))  # index 0 is the player's own cell

    turned = archive.features(pos, 90.0)
    assert len(turned) == 9
    assert abs(turned[1] - one) < 1e-12  # k = 0: yaw 90 faces +x
    assert turned[7] == 1.0  # k = 6: 270 degrees clockwise from +x is +z, on the player's left
    assert all(v == 0.0 for i, v in enumerate(turned) if i not in (1, 7))


def test_archive_save_load_round_trip():
    archive = ExplorationArchive(cell_size=4.0)
    archive.start_episode()
    for pos in ((1.0, 1.0, 1.0), (-7.0, 0.5, 30.0), (40000.0, -3.0, -2.0)):
        archive.visit(pos)
    archive.start_episode()
    archive.visit((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models" / "explore_Level_0-1_47800.npz"
        archive.save(path)
        assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp file left, no second .npz suffix
        loaded = ExplorationArchive.load(path, cell_size=4.0)
        empty = Path(tmp) / "empty.npz"
        ExplorationArchive(cell_size=4.0).save(empty)
        assert ExplorationArchive.load(empty, cell_size=4.0).counts == {}
    assert loaded.cell_size == 4.0 and loaded.episode_cells == 0
    assert loaded.counts == {(0, 0, 0): 2, (-2, 0, 7): 1, (10000, -1, -1): 1}
    assert all(type(v) is int for cell in loaded.counts for v in cell)
    assert all(type(n) is int for n in loaded.counts.values())


def test_archive_save_retries_a_briefly_locked_file():
    archive = ExplorationArchive(cell_size=4.0)
    archive.visit((1.0, 1.0, 1.0))
    real_replace, calls = os.replace, []

    def replace_locked_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:  # Windows refuses the replace while another process has the file open
            raise PermissionError(13, "The process cannot access the file")
        real_replace(src, dst)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explore.npz"
        with mock.patch("os.replace", replace_locked_twice):
            archive.save(path)
        assert len(calls) == 3
        assert [p.name for p in path.parent.iterdir()] == ["explore.npz"]
        assert ExplorationArchive.load(path, cell_size=4.0).counts == {(0, 0, 0): 1}
        with mock.patch("os.replace", side_effect=PermissionError(13, "locked")) as always_locked:
            try:
                archive.save(Path(tmp) / "stuck.npz")
            except PermissionError:
                pass
            else:
                raise AssertionError("save should give up and raise")
        assert always_locked.call_count == 5


def test_archive_load_missing_or_unreadable_is_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert ExplorationArchive.load(Path(tmp) / "missing.npz", cell_size=4.0).counts == {}
        bad = Path(tmp) / "bad.npz"
        for data in (b"not an npz file", b"PK\x03\x04 truncated zip", b""):
            bad.write_bytes(data)
            loaded = ExplorationArchive.load(bad, cell_size=4.0)
            assert loaded.counts == {} and loaded.cell_size == 4.0, data


def test_archive_load_cell_size_mismatch_is_empty():
    archive = ExplorationArchive(cell_size=4.0)
    archive.visit((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explore.npz"
        archive.save(path)
        assert ExplorationArchive.load(path, cell_size=4.0).counts == {(0, 0, 0): 1}
        loaded = ExplorationArchive.load(path, cell_size=2.0)
    assert loaded.counts == {} and loaded.cell_size == 2.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign.py
```

Expected: a traceback ending in `ModuleNotFoundError: No module named 'ultrakill_ai.campaign'`.

- [ ] **Step 3: Write the implementation**

Create `python/ultrakill_ai/campaign.py` with exactly this content:

```python
"""Campaign helpers: level names, rank maths, and the measures behind the campaign rewards.

Plain Python plus numpy (for the archive file), so all of it is tested without the game in
tests/test_campaign.py. The inputs come from the mod's `campaign` observation block (docs/protocol.md).
"""

from __future__ import annotations

import math
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np


def _replace_file(tmp: Path, path: Path) -> None:
    """os.replace(tmp, path), retried briefly as progress.write_json_atomic does.

    On Windows the replace fails with PermissionError while another process has `path` open for a moment (another
    training game reading the same best-run file, an editor, a virus scan).
    """
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Levels and ranks
# ---------------------------------------------------------------------------

# The 35 main levels in mission order (StatsManager.levelNumber 1..35, see GetMissionName). Prime Sanctums and
# encores are out of scope.
CAMPAIGN_LEVELS: tuple[str, ...] = (
    "Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4", "Level 0-5",
    "Level 1-1", "Level 1-2", "Level 1-3", "Level 1-4",
    "Level 2-1", "Level 2-2", "Level 2-3", "Level 2-4",
    "Level 3-1", "Level 3-2",
    "Level 4-1", "Level 4-2", "Level 4-3", "Level 4-4",
    "Level 5-1", "Level 5-2", "Level 5-3", "Level 5-4",
    "Level 6-1", "Level 6-2",
    "Level 7-1", "Level 7-2", "Level 7-3", "Level 7-4",
    "Level 8-1", "Level 8-2", "Level 8-3", "Level 8-4",
    "Level 9-1", "Level 9-2",
)

RANK_LETTERS = ("D", "C", "B", "A", "S")


def safe_name(scene: str) -> str:
    """A scene name usable in file names ("Level 0-1" -> "Level_0-1")."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scene)


def grade(thresholds: list[int], value: float, reverse: bool) -> int:
    """One rank category scored the way StatsManager.GetRanks does it: 0 (D) to 4 (S).

    Thresholds are checked in order and counting stops at the first one missed; meeting all of them scores 4.
    `reverse` is for time, where lower is better (`value <= t`); kills and style need `value >= t`.
    """
    for i, t in enumerate(thresholds):
        met = value <= t if reverse else value >= t
        if not met:
            return i
    return 4


def compute_rank(seconds: float, kills: int, style: int, restarts: int, ranks: dict) -> str:
    """The level rank StatsManager.GetFinalRank gives without cheats or major assists: "D".."S", or "P".

    `ranks` is the campaign block's {"time": [4], "kills": [4], "style": [4]}. The three category scores are
    summed and restarts subtracted (floored at 0); 12 is P, anything else rounds total / 3 to a letter. Thirds
    never land on .5, so round-half-up matches Unity's RoundToInt.
    """
    total = (
        grade(ranks["time"], seconds, reverse=True)
        + grade(ranks["kills"], kills, reverse=False)
        + grade(ranks["style"], style, reverse=False)
    )
    total = max(0, total - int(restarts))
    if total == 12:
        return "P"
    return RANK_LETTERS[math.floor(total / 3 + 0.5)]


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

EXPLORE_LOG_SCALE = math.log(1001.0)  # a cell entered in 1000 episodes reads 1.0 on the exploration map


class ExplorationArchive:
    """Visit counts over a grid of cubic cells in one level, kept per game.

    `counts[cell]` is the number of episodes that entered the cell (the current one included, once it has).
    `visit` pays novelty 1/sqrt(N + 1) on a cell's first entry in an episode, so a cell pays 1.0 the first time
    any episode reaches it and less each time after. `features` shows the policy the same counts around the
    player as a 9-value map.
    """

    def __init__(self, cell_size: float = 4.0):
        self.cell_size = float(cell_size)
        self.counts: dict[tuple[int, int, int], int] = {}
        self._episode: set[tuple[int, int, int]] = set()

    def cell(self, pos) -> tuple[int, int, int]:
        s = self.cell_size
        return (math.floor(pos[0] / s), math.floor(pos[1] / s), math.floor(pos[2] / s))

    def start_episode(self) -> None:
        self._episode.clear()

    def visit(self, pos) -> float:
        """Novelty for being at `pos`: 1/sqrt(N + 1) on the first entry of its cell this episode, else 0."""
        cell = self.cell(pos)
        if cell in self._episode:
            return 0.0
        self._episode.add(cell)
        n = self.counts.get(cell, 0)
        self.counts[cell] = n + 1
        return 1.0 / math.sqrt(n + 1)

    @property
    def episode_cells(self) -> int:
        """Cells entered so far this episode."""
        return len(self._episode)

    def features(self, pos, yaw_deg: float) -> list[float]:
        """The exploration map: the player's cell, then the 8 horizontal neighbours one cell_size away.

        Neighbour k sits at yaw + 45k degrees (k = 0 straight ahead, then clockwise to the right), the same yaw
        convention as spaces.yaw_frame, so the map turns with the player. Each value is min(1, log1p(N) / log(1001)).
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        points = [(x, y, z)]
        for k in range(8):
            a = math.radians(yaw_deg + 45.0 * k)
            points.append((x + self.cell_size * math.sin(a), y, z + self.cell_size * math.cos(a)))
        return [min(1.0, math.log1p(self.counts.get(self.cell(p), 0)) / EXPLORE_LOG_SCALE) for p in points]

    def save(self, path) -> None:
        """Writes the counts to an .npz file atomically (a temp file, then a rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = np.array(list(self.counts.keys()), dtype=np.int32).reshape(-1, 3)
        counts = np.array(list(self.counts.values()), dtype=np.int64)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:  # a file object, so numpy does not append .npz to the temp name
            np.savez(f, cells=cells, counts=counts, cell_size=np.float64(self.cell_size))
        _replace_file(tmp, path)

    @classmethod
    def load(cls, path, cell_size: float) -> "ExplorationArchive":
        """The archive saved at `path`, or an empty one when the file is missing, unreadable or used another cell size."""
        archive = cls(cell_size)
        try:
            with np.load(Path(path)) as data:
                if float(data["cell_size"]) != archive.cell_size:
                    return archive
                cells, counts = data["cells"], data["counts"]
            loaded = {(int(c[0]), int(c[1]), int(c[2])): int(n) for c, n in zip(cells, counts)}
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            return archive
        archive.counts = loaded
        return archive
```

- [ ] **Step 4: Run the test to verify it passes**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign.py
```

Expected:

```
ok test_archive_cell_floors_each_axis
ok test_archive_features_straight_ahead_and_clockwise
ok test_archive_load_cell_size_mismatch_is_empty
ok test_archive_load_missing_or_unreadable_is_empty
ok test_archive_novelty_first_entry_per_episode
ok test_archive_save_load_round_trip
ok test_archive_save_retries_a_briefly_locked_file
ok test_campaign_levels
ok test_compute_rank_letters
ok test_compute_rank_p_needs_12_without_restarts
ok test_compute_rank_restarts_lower_the_rank
ok test_grade_all_thresholds_met_is_4
ok test_grade_counts_thresholds_in_order
ok test_grade_stops_at_the_first_miss
ok test_safe_name
15 tests passed
```

- [ ] **Step 5: Run the regression suites**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_aim.py
.venv\Scripts\python tests\test_progress.py
```

Expected: `9 tests passed` from `test_aim.py`, and `all tests passed` as the last line from `test_progress.py`.

- [ ] **Step 6: Update CLAUDE.md**

In `CLAUDE.md` (Layout), replace this line:

```markdown
  - `routes.py`: campaign route tracking.
```

with:

```markdown
  - `routes.py`: campaign route tracking.
  - `campaign.py`: campaign helpers: `CAMPAIGN_LEVELS` (the 35 main scene names in mission order), `safe_name`, the game's rank maths (`grade`, `compute_rank`; P needs 12 with no restarts) and `ExplorationArchive` (per-game visit counts over 4 m cells: novelty `1/sqrt(N+1)` on a cell's first entry per episode, the 9-value exploration map around the player, atomic `.npz` save/load).
```

In `CLAUDE.md` (Layout), replace this line:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
```

with:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_campaign.py`: campaign helpers: level list, rank maths and the exploration archive (no game needed).
```

In `CLAUDE.md` (Commands), replace this line:

```markdown
- Tests (no game): `python tests/test_progress.py` and `python tests/test_aim.py` (pytest is not installed; the files also work under pytest).
```

with:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py` and `python tests/test_campaign.py` (pytest is not installed; the files also work under pytest).
```

- [ ] **Step 7: Commit and push**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/campaign.py python/tests/test_campaign.py CLAUDE.md
git commit -m "Add campaign helpers: level list, ranks and exploration archive" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

### Task 4: Python `campaign.py`: milestones, path progress, fresh-start rule, best runs

**Files:**
- Modify: `python/ultrakill_ai/campaign.py` (imports; append `MilestoneTracker`, `PathProgress`, `choose_fresh_start`, `save_best_run`)
- Modify: `python/tests/test_campaign.py` (imports; append the Task 4 tests)
- Modify: `CLAUDE.md` (Layout: the `campaign.py` and `tests/test_campaign.py` lines)
- Test: `python/tests/test_campaign.py`

- [ ] **Step 1: Extend the test imports**

In `python/tests/test_campaign.py`, replace:

```python
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CAMPAIGN_LEVELS,
    RANK_LETTERS,
    ExplorationArchive,
    compute_rank,
    grade,
    safe_name,
)
```

with:

```python
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    CAMPAIGN_LEVELS,
    RANK_LETTERS,
    ExplorationArchive,
    MilestoneTracker,
    PathProgress,
    choose_fresh_start,
    compute_rank,
    grade,
    safe_name,
    save_best_run,
)
```

- [ ] **Step 2: Write the failing tests**

In `python/tests/test_campaign.py`, replace:

```python
if __name__ == "__main__":
```

with:

```python
def milestones_block(checkpoints=(), arenas=(), doors=()):
    """A campaign block holding only the milestone keys. `checkpoints` holds (id, activated, current) tuples."""
    return {
        "checkpoints": [{"id": cid, "pos": [0.0, 0.0, 0.0], "activated": act, "current": cur} for cid, act, cur in checkpoints],
        "cleared_arenas": list(arenas),
        "unlocked_doors": list(doors),
    }


def test_milestones_pay_a_checkpoint_once_per_level_load():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (1, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)])) == (0, 0, 0)
    assert m.checkpoints_reached == 1
    further = milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(further) == (1, 1, 1)
    assert m.update(further) == (0, 0, 0)
    assert m.checkpoints_reached == 2
    assert m.update(None) == (0, 0, 0)  # a step without the block pays nothing and forgets nothing
    assert m.update(further) == (0, 0, 0)


def test_milestones_level_load_baseline_pays_nothing():
    m = MilestoneTracker()
    loaded = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.new_level_load(loaded)
    assert m.update(loaded) == (0, 0, 0)
    assert m.checkpoints_reached == 1
    m.new_level_load(None)
    assert m.checkpoints_reached == 0


def test_milestones_mark_paid_absorbs_a_respawn_unlock():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"])) == (1, 1, 0)
    # Restart() at the checkpoint unlocked a door: absorbed, never paid
    respawned = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25"])
    m.mark_paid(respawned)
    assert m.update(respawned) == (0, 0, 0)
    later = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["0,0,25", "9,0,50"])
    assert m.update(later) == (0, 0, 1)
    assert m.checkpoints_reached == 1


def test_milestones_current_counts_as_activated():
    m = MilestoneTracker()
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False), ("0,1,80", False, False)]))
    assert m.update(milestones_block(checkpoints=[("0,1,20", False, True), ("0,1,80", False, False)])) == (1, 0, 0)
    assert m.update(milestones_block(checkpoints=[("0,1,20", True, False), ("0,1,80", False, False)])) == (0, 0, 0)
    assert m.checkpoints_reached == 1


def test_milestones_new_level_load_pays_again():
    m = MilestoneTracker()
    m.new_level_load(milestones_block())
    reached = milestones_block(checkpoints=[("0,1,20", True, True)], arenas=["0,1,30"], doors=["5,0,40"])
    assert m.update(reached) == (1, 1, 1)
    m.new_level_load(milestones_block(checkpoints=[("0,1,20", False, False)]))
    assert m.checkpoints_reached == 0
    assert m.update(reached) == (1, 1, 1)


def nav_path(length, status="complete"):
    return {"status": status, "length": length, "next_corner": [0.0, 0.0, 0.0]}


def test_path_progress_baseline_pays_nothing():
    p = PathProgress()
    p.reset()
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(70.0)) == 0.0  # a longer path is not progress and does not move the best
    assert p.update(nav_path(58.5)) == 1.5  # still measured from 60


def test_path_progress_pays_a_gain_over_min_gain():
    assert PathProgress.MIN_GAIN == 1.0
    p = PathProgress()
    p.reset()
    p.update(nav_path(60.0))
    assert p.update(nav_path(52.0)) == 8.0
    assert p.update(nav_path(52.0)) == 0.0
    assert p.update(nav_path(51.0)) == 0.0  # exactly MIN_GAIN is not enough
    p.reset()
    assert p.update(nav_path(40.0)) == 0.0  # a new episode starts from a new baseline
    assert p.update(nav_path(30.0)) == 10.0


def test_path_progress_partial_or_missing_path_pays_nothing():
    p = PathProgress()
    p.reset()
    assert p.update(None) == 0.0
    assert p.update({"status": "none"}) == 0.0
    assert p.update(nav_path(10.0, status="partial")) == 0.0  # does not set the baseline either
    assert p.update(nav_path(60.0)) == 0.0
    assert p.update(nav_path(20.0, status="partial")) == 0.0
    assert p.update({"status": "none"}) == 0.0
    assert p.update(nav_path(55.0)) == 5.0


def test_path_progress_sub_metre_gains_accumulate():
    p = PathProgress()
    p.reset()
    p.update(nav_path(60.0))
    assert p.update(nav_path(59.6)) == 0.0
    assert p.update(nav_path(59.2)) == 0.0
    assert abs(p.update(nav_path(58.8)) - 1.2) < 1e-9  # measured from 60, not from 59.2
    assert p.update(nav_path(58.8)) == 0.0


def fresh(rng, fresh_prob, **overrides):
    """choose_fresh_start for a game mid-level with an active checkpoint, with some conditions overridden."""
    kwargs = {"in_level": True, "level_over": False, "has_checkpoint": True, "stuck_streak": 0, "stuck_limit": 3}
    kwargs.update(overrides)
    return choose_fresh_start(rng, fresh_prob=fresh_prob, **kwargs)


def test_fresh_start_when_not_in_the_level():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, in_level=False) is True


def test_fresh_start_after_the_level_is_over():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, level_over=True) is True


def test_fresh_start_without_a_checkpoint():
    for prob in (0.0, 1.0):
        assert fresh(random.Random(0), prob, has_checkpoint=False) is True


def test_fresh_start_after_repeated_stuck_episodes():
    assert fresh(random.Random(0), 0.0, stuck_streak=2) is False
    assert fresh(random.Random(0), 0.0, stuck_streak=3) is True
    assert fresh(random.Random(0), 0.0, stuck_streak=4) is True
    assert fresh(random.Random(0), 1.0, stuck_streak=3) is True


def test_fresh_start_otherwise_by_probability():
    assert fresh(random.Random(0), 0.0) is False
    assert fresh(random.Random(0), 1.0) is True
    rng, twin = random.Random(123), random.Random(123)
    picks = [fresh(rng, 0.2) for _ in range(2000)]
    assert picks == [twin.random() < 0.2 for _ in range(2000)]  # one draw per decision from the seeded source
    assert 300 < sum(picks) < 500
    forced, untouched = random.Random(7), random.Random(7)
    assert fresh(forced, 0.2, in_level=False) is True
    assert forced.random() == untouched.random()  # a forced reload does not draw


def best_run(seconds, **extra):
    run = {
        "level": "Level 0-1", "seconds": seconds, "kills": 5, "style": 300, "restarts": 0, "deaths": 0, "rank": "A",
        "difficulty": 3, "positions": [[0, 1, 2], [0, 1, 4]], "saved_at": "2026-09-16T12:00:00",
    }
    run.update(extra)
    return run


def test_save_best_run_writes_the_first_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_runs" / "Level_0-1.json"
        assert save_best_run(path, best_run(95.5)) is True
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(95.5)
        assert [p.name for p in path.parent.iterdir()] == ["Level_0-1.json"]  # no temp file left behind


def test_save_best_run_refuses_a_slower_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        assert save_best_run(path, best_run(95.5)) is True
        assert save_best_run(path, best_run(120.0, kills=9)) is False
        assert save_best_run(path, best_run(95.5, kills=9)) is False  # a tie keeps the stored run
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(95.5)


def test_save_best_run_overwrites_with_a_faster_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        save_best_run(path, best_run(95.5))
        assert save_best_run(path, best_run(80.25, rank="S")) is True
        assert json.loads(path.read_text(encoding="utf-8")) == best_run(80.25, rank="S")
        assert [p.name for p in path.parent.iterdir()] == ["Level_0-1.json"]


def test_save_best_run_overwrites_an_unreadable_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Level_0-1.json"
        path.write_text("{truncated", encoding="utf-8")
        assert save_best_run(path, best_run(300.0)) is True
        assert json.loads(path.read_text(encoding="utf-8"))["seconds"] == 300.0
        path.write_text('{"level": "Level 0-1"}', encoding="utf-8")  # readable JSON without a time
        assert save_best_run(path, best_run(310.0)) is True
        assert json.loads(path.read_text(encoding="utf-8"))["seconds"] == 310.0


if __name__ == "__main__":
```

- [ ] **Step 3: Run the tests to verify they fail**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign.py
```

Expected: a traceback ending in `ImportError: cannot import name 'MilestoneTracker' from 'ultrakill_ai.campaign' (F:\Github\ULTRAKILL-AI\python\ultrakill_ai\campaign.py)`.

- [ ] **Step 4: Extend the module imports**

In `python/ultrakill_ai/campaign.py`, replace:

```python
import math
import os
import re
import time
import zipfile
from pathlib import Path
```

with:

```python
import json
import math
import os
import random
import re
import time
import zipfile
from pathlib import Path
```

- [ ] **Step 5: Write the implementation**

In `python/ultrakill_ai/campaign.py`, replace the end of `ExplorationArchive.load`:

```python
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            return archive
        archive.counts = loaded
        return archive
```

with:

```python
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            return archive
        archive.counts = loaded
        return archive


# ---------------------------------------------------------------------------
# Milestones, path progress and episode starts
# ---------------------------------------------------------------------------


class MilestoneTracker:
    """Pays each checkpoint, arena clear and door unlock once per level load.

    The mod reports arenas and doors as rounded-position keys. A checkpoint respawn re-instantiates rooms at the
    same position, so an arena cleared again after a death gives the same key and cannot pay twice. Keys that
    show up during a reset or respawn (Restart() unlocks the checkpoint's doors) are absorbed with `mark_paid`
    instead of paid. A checkpoint counts once it is activated or current.
    """

    def __init__(self):
        self._checkpoints: set[str] = set()
        self._arenas: set[str] = set()
        self._doors: set[str] = set()

    @staticmethod
    def _keys(campaign: dict | None) -> tuple[set[str], set[str], set[str]]:
        if not campaign:
            return set(), set(), set()
        checkpoints = {str(cp["id"]) for cp in campaign.get("checkpoints") or () if cp.get("activated") or cp.get("current")}
        return checkpoints, set(campaign.get("cleared_arenas") or ()), set(campaign.get("unlocked_doors") or ())

    def new_level_load(self, campaign: dict | None) -> None:
        """Starts a level load: forgets what was paid, then absorbs whatever the fresh level already reports."""
        self._checkpoints.clear()
        self._arenas.clear()
        self._doors.clear()
        self.mark_paid(campaign)

    def mark_paid(self, campaign: dict | None) -> None:
        """Absorbs the block's current keys without paying for them."""
        self.update(campaign)

    def update(self, campaign: dict | None) -> tuple[int, int, int]:
        """(new checkpoints, new arena clears, new door unlocks): keys not yet paid or absorbed this level load."""
        checkpoints, arenas, doors = self._keys(campaign)
        new = (len(checkpoints - self._checkpoints), len(arenas - self._arenas), len(doors - self._doors))
        self._checkpoints |= checkpoints
        self._arenas |= arenas
        self._doors |= doors
        return new

    @property
    def checkpoints_reached(self) -> int:
        """Distinct checkpoints activated in this level load (paid or absorbed)."""
        return len(self._checkpoints)


class PathProgress:
    """Pays metres of new best NavMesh path length to the exit within an episode.

    Only complete paths count: a partial path stops wherever the NavMesh does, so its length says nothing about
    the distance left. The first complete path of an episode is the baseline and pays nothing. A new best must
    beat the old one by more than MIN_GAIN, and smaller improvements leave the best where it was, so they add up
    instead of paying jitter from the snapped path ends.
    """

    MIN_GAIN = 1.0

    def __init__(self):
        self.best = math.inf

    def reset(self) -> None:
        self.best = math.inf

    def update(self, path: dict | None) -> float:
        if not path or path.get("status") != "complete" or path.get("length") is None:
            return 0.0
        length = float(path["length"])
        if math.isinf(self.best):
            self.best = length
            return 0.0
        if length < self.best - self.MIN_GAIN:
            gain = self.best - length
            self.best = length
            return gain
        return 0.0


def choose_fresh_start(
    rng: random.Random,
    *,
    in_level: bool,
    level_over: bool,
    has_checkpoint: bool,
    stuck_streak: int,
    stuck_limit: int,
    fresh_prob: float,
) -> bool:
    """Whether the next campaign episode reloads the level (True) or respawns at the current checkpoint.

    A reload is forced when the game is not in the level with a player, the level is over, there is no current
    checkpoint to respawn at, or `stuck_limit` episodes in a row ended stuck at the same checkpoint (a respawn can
    leave a door locked behind the player). Otherwise it reloads with probability `fresh_prob`, so fresh-start
    completions keep being measured while most episodes land on each game's frontier. A forced reload does not
    draw from `rng`.
    """
    if not in_level or level_over or not has_checkpoint or stuck_streak >= stuck_limit:
        return True
    return rng.random() < fresh_prob


def save_best_run(path, run: dict) -> bool:
    """Stores `run` as the level's best run if it is faster than the stored one. Returns whether it wrote.

    A missing or unreadable file, or one without a time, counts as no stored run. The file is replaced
    atomically (retrying briefly while another training game has it open), so a crash mid-write never leaves a
    broken best run behind.
    """
    path = Path(path)
    try:
        stored = float(json.loads(path.read_text(encoding="utf-8"))["seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        stored = math.inf
    if not float(run["seconds"]) < stored:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
    _replace_file(tmp, path)
    return True
```

- [ ] **Step 6: Run the tests to verify they pass**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign.py
```

Expected:

```
ok test_archive_cell_floors_each_axis
ok test_archive_features_straight_ahead_and_clockwise
ok test_archive_load_cell_size_mismatch_is_empty
ok test_archive_load_missing_or_unreadable_is_empty
ok test_archive_novelty_first_entry_per_episode
ok test_archive_save_load_round_trip
ok test_archive_save_retries_a_briefly_locked_file
ok test_campaign_levels
ok test_compute_rank_letters
ok test_compute_rank_p_needs_12_without_restarts
ok test_compute_rank_restarts_lower_the_rank
ok test_fresh_start_after_repeated_stuck_episodes
ok test_fresh_start_after_the_level_is_over
ok test_fresh_start_otherwise_by_probability
ok test_fresh_start_when_not_in_the_level
ok test_fresh_start_without_a_checkpoint
ok test_grade_all_thresholds_met_is_4
ok test_grade_counts_thresholds_in_order
ok test_grade_stops_at_the_first_miss
ok test_milestones_current_counts_as_activated
ok test_milestones_level_load_baseline_pays_nothing
ok test_milestones_mark_paid_absorbs_a_respawn_unlock
ok test_milestones_new_level_load_pays_again
ok test_milestones_pay_a_checkpoint_once_per_level_load
ok test_path_progress_baseline_pays_nothing
ok test_path_progress_partial_or_missing_path_pays_nothing
ok test_path_progress_pays_a_gain_over_min_gain
ok test_path_progress_sub_metre_gains_accumulate
ok test_safe_name
ok test_save_best_run_overwrites_an_unreadable_file
ok test_save_best_run_overwrites_with_a_faster_run
ok test_save_best_run_refuses_a_slower_run
ok test_save_best_run_writes_the_first_run
33 tests passed
```

- [ ] **Step 7: Run the regression suites**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_aim.py
.venv\Scripts\python tests\test_progress.py
```

Expected: `9 tests passed` from `test_aim.py`, and `all tests passed` as the last line from `test_progress.py`.

- [ ] **Step 8: Update CLAUDE.md**

In `CLAUDE.md` (Layout), replace this line (written in Task 3):

```markdown
  - `campaign.py`: campaign helpers: `CAMPAIGN_LEVELS` (the 35 main scene names in mission order), `safe_name`, the game's rank maths (`grade`, `compute_rank`; P needs 12 with no restarts) and `ExplorationArchive` (per-game visit counts over 4 m cells: novelty `1/sqrt(N+1)` on a cell's first entry per episode, the 9-value exploration map around the player, atomic `.npz` save/load).
```

with:

```markdown
  - `campaign.py`: campaign helpers: `CAMPAIGN_LEVELS` (the 35 main scene names in mission order), `safe_name`, the game's rank maths (`grade`, `compute_rank`; P needs 12 with no restarts), `ExplorationArchive` (per-game visit counts over 4 m cells: novelty `1/sqrt(N+1)` on a cell's first entry per episode, the 9-value exploration map around the player, atomic `.npz` save/load), `MilestoneTracker` (pays each checkpoint, arena clear and door unlock once per level load; `mark_paid` absorbs what a reset or respawn reveals), `PathProgress` (metres of new best complete NavMesh path to the exit; gains count once they exceed 1 m), `choose_fresh_start` (level reload or checkpoint respawn for the next episode) and `save_best_run` (keeps the fastest run per level as JSON, written atomically).
```

In `CLAUDE.md` (Layout), replace this line (written in Task 3):

```markdown
- `python/tests/test_campaign.py`: campaign helpers: level list, rank maths and the exploration archive (no game needed).
```

with:

```markdown
- `python/tests/test_campaign.py`: campaign helpers: level list, rank maths, exploration archive, milestones, path progress, the fresh-start rule and best runs (no game needed).
```

- [ ] **Step 9: Commit and push**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/campaign.py python/tests/test_campaign.py CLAUDE.md
git commit -m "Add campaign milestones, path progress, fresh-start rule and best runs" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

### Task 5: Python `rewards.py`: campaign reward terms, retire route/stuck terms

The campaign rewards (time, milestones, novelty, path) are paid from a `CampaignStep` the env measures; the
human-route terms `route_point` and `stuck` go away. `env.py` gets only its one `compute_reward` call changed, so
the Cyber Grind path keeps working until Task 7 rewrites the file.

**Files:**
- Modify: `python/ultrakill_ai/rewards.py` (`RewardConfig` campaign fields, new `CampaignStep`, `compute_reward` signature and campaign terms)
- Modify: `python/ultrakill_ai/env.py` (the single `compute_reward` call in `step`)
- Modify: `CLAUDE.md` (Layout: `rewards.py` and the new test; Commands: the tests bullet)
- Modify: `README.md` (How it works: the campaign rewards bullet)
- Test: `python/tests/test_campaign_rewards.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_campaign_rewards.py`:

```python
"""Campaign reward terms. No game needed:  python tests/test_campaign_rewards.py  (or pytest)."""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.rewards import CampaignStep, RewardConfig, compute_reward  # noqa: E402

CAMPAIGN_WEIGHTS = {"time": 0.01, "checkpoint": 10.0, "arena_clear": 10.0, "door_unlock": 3.0, "novelty": 0.5, "path": 0.1}
CAMPAIGN_PARTS = ("time", "checkpoint", "arena_clear", "door_unlock", "novelty", "path")


def snapshot(level_complete: bool = False, hp: int = 100, kills: int = 0) -> dict:
    """A quiet step the way the mod reports it: a player, no enemies, only the given stats."""
    return {
        "player": {"pos": [0.0, 0.0, 0.0], "hp": hp, "dead": False},
        "enemies": [],
        "stats": {"kills": kills, "style": 0, "level_complete": level_complete},
    }


def test_campaign_step_defaults_to_nothing_happened():
    step = CampaignStep()
    assert (step.checkpoints, step.arenas, step.doors, step.novelty, step.path_gain) == (0, 0, 0, 0.0, 0.0)


def test_time_is_only_charged_with_a_campaign_step():
    cfg = RewardConfig(time=0.01)
    quiet = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep())
    assert quiet.parts == {"time": -0.01} and abs(quiet.total + 0.01) < 1e-12
    assert "time" not in compute_reward(cfg, snapshot(), snapshot(), {}).parts


def test_each_milestone_weight_multiplies_its_count():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(checkpoints=2, arenas=3, doors=4)).parts
    assert abs(parts["checkpoint"] - 20.0) < 1e-9
    assert abs(parts["arena_clear"] - 30.0) < 1e-9
    assert abs(parts["door_unlock"] - 12.0) < 1e-9
    assert "novelty" not in parts and "path" not in parts  # nothing new this step pays nothing
    single = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(doors=1)).parts
    assert abs(single["door_unlock"] - 3.0) < 1e-9 and "checkpoint" not in single and "arena_clear" not in single


def test_novelty_and_path_scale_by_their_weights():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    novelty = 1.0 + 1.0 / 2.0 ** 0.5  # a cell no earlier episode entered, and one that one earlier episode did
    result = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep(novelty=novelty, path_gain=12.5))
    assert abs(result.parts["novelty"] - 0.5 * novelty) < 1e-9
    assert abs(result.parts["path"] - 1.25) < 1e-9
    assert abs(result.total - sum(result.parts.values())) < 1e-9
    off = compute_reward(RewardConfig(), snapshot(), snapshot(), {}, campaign=CampaignStep(checkpoints=1, novelty=novelty, path_gain=12.5))
    assert off.parts == {}  # every campaign weight defaults to 0


def test_finishing_within_the_cap_beats_timing_out():
    # The campaign config charges time 0.01 per decision over a 9000-decision cap (10 game minutes).
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    max_steps = 9000
    tick = compute_reward(cfg, snapshot(), snapshot(), {}, campaign=CampaignStep()).total
    last = compute_reward(cfg, snapshot(), snapshot(level_complete=True), {}, campaign=CampaignStep()).total
    timed_out = max_steps * tick
    finished_on_the_last_decision = (max_steps - 1) * tick + last
    assert abs(timed_out + 90.0) < 1e-6 and abs(last - 99.99) < 1e-9
    assert finished_on_the_last_decision > timed_out
    assert (max_steps // 2 - 1) * tick + last > finished_on_the_last_decision  # finishing sooner always pays more


def test_level_complete_pays_100_on_the_rising_edge_only():
    cfg = RewardConfig()
    assert cfg.level_complete == 100.0
    for campaign in (None, CampaignStep()):  # the same rule in both modes
        rising = compute_reward(cfg, snapshot(), snapshot(level_complete=True), {}, campaign=campaign).parts
        assert rising["level_complete"] == 100.0
        held = compute_reward(cfg, snapshot(level_complete=True), snapshot(level_complete=True), {}, campaign=campaign).parts
        assert "level_complete" not in held
        falling = compute_reward(cfg, snapshot(level_complete=True), snapshot(), {}, campaign=campaign).parts
        assert "level_complete" not in falling


def test_cybergrind_call_has_no_campaign_parts():
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    parts = compute_reward(cfg, snapshot(), snapshot(kills=1), {}).parts
    assert parts["kill"] == cfg.kill
    assert not set(CAMPAIGN_PARTS) & set(parts)


def test_campaign_terms_are_paid_on_a_step_without_a_player():
    # The env marks milestones paid before it computes the reward, so a step that arrives without a player
    # (a level load) must still pay them or they are lost for the whole level load.
    cfg = RewardConfig(**CAMPAIGN_WEIGHTS)
    loading = {"enemies": [], "stats": {}}
    result = compute_reward(cfg, snapshot(), loading, {}, campaign=CampaignStep(checkpoints=1))
    assert set(result.parts) == {"time", "checkpoint"}
    assert compute_reward(cfg, snapshot(), loading, {}).parts == {}  # Cyber Grind: no player, no reward, as before


def test_died_is_a_keyword_after_enemy_max_health():
    cfg = RewardConfig(death=5.0, damage_taken=0.01)
    parts = compute_reward(cfg, snapshot(hp=40), snapshot(hp=40), {}, died=True).parts
    assert parts["death"] == -5.0
    assert abs(parts["damage_taken"] + 0.4) < 1e-9  # the lethal hit is the HP the player had left


def test_route_terms_are_retired():
    names = {f.name for f in fields(RewardConfig)}
    assert not {"route_point", "stuck"} & names
    assert not hasattr(RewardConfig(), "route_point") and not hasattr(RewardConfig(), "stuck")
    assert list(inspect.signature(compute_reward).parameters) == ["cfg", "prev", "cur", "enemy_max_health", "died", "campaign"]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test and confirm it fails**

From `python/`:

```powershell
.venv\Scripts\python tests\test_campaign_rewards.py
```

Expected: a traceback ending in
`ImportError: cannot import name 'CampaignStep' from 'ultrakill_ai.rewards' (F:\Github\ULTRAKILL-AI\python\ultrakill_ai\rewards.py)`.

- [ ] **Step 3: Replace the route-era campaign fields and add `CampaignStep`**

In `python/ultrakill_ai/rewards.py`, replace:

```python
    # Campaign
    route_point: float = 0.1  # per route point reached (~1 m of progress)
    level_complete: float = 50.0
    stuck: float = 1.0


@dataclass
class RewardResult:
```

with:

```python
    # Campaign. The new terms default to 0, so Cyber Grind runs are unchanged; campaign configs set them.
    level_complete: float = 100.0  # once, on the step the level ends (both modes)
    time: float = 0.0  # charged per decision, so finishing sooner always pays
    checkpoint: float = 0.0  # per checkpoint first reached in a level load
    arena_clear: float = 0.0  # per arena whose last wave died, once per level load
    door_unlock: float = 0.0  # per door unlocked by play, once per level load (respawn unlocks never pay)
    novelty: float = 0.0  # times CampaignStep.novelty, the summed 1/sqrt(N+1) of cells new this episode
    path: float = 0.0  # per metre of new best NavMesh distance to the exit


@dataclass
class CampaignStep:
    """What the level did this step, measured by the env (see campaign.py)."""

    checkpoints: int = 0
    arenas: int = 0
    doors: int = 0
    novelty: float = 0.0  # sum of 1/sqrt(N+1) over cells entered for the first time this episode
    path_gain: float = 0.0  # metres of new best NavMesh distance to the exit


@dataclass
class RewardResult:
```

- [ ] **Step 4: Change the `compute_reward` signature and pay the campaign terms**

In `python/ultrakill_ai/rewards.py`, replace:

```python
    enemy_max_health: dict[int, float],
    route_gain: int = 0,
    stuck: bool = False,
    died: bool | None = None,
) -> RewardResult:
    r = RewardResult()
    pp, cp = prev.get("player"), cur.get("player")
    if not pp or not cp:
        return r
```

with:

```python
    enemy_max_health: dict[int, float],
    died: bool | None = None,
    campaign: CampaignStep | None = None,
) -> RewardResult:
    r = RewardResult()
    if campaign is not None:
        # Paid before the player check: the env has already marked these milestones paid, so a step that
        # arrives without a player (a level load) must not drop them.
        r.add("time", -cfg.time)
        r.add("checkpoint", cfg.checkpoint * campaign.checkpoints)
        r.add("arena_clear", cfg.arena_clear * campaign.arenas)
        r.add("door_unlock", cfg.door_unlock * campaign.doors)
        r.add("novelty", cfg.novelty * campaign.novelty)
        r.add("path", cfg.path * campaign.path_gain)

    pp, cp = prev.get("player"), cur.get("player")
    if not pp or not cp:
        return r
```

- [ ] **Step 5: Remove the route and stuck terms at the end of `compute_reward`**

In `python/ultrakill_ai/rewards.py`, replace:

```python
    r.add("route", cfg.route_point * route_gain)
    if cs.get("level_complete") and not ps.get("level_complete"):
        r.add("level_complete", cfg.level_complete)
    if stuck:
        r.add("stuck", -cfg.stuck)

    return r
```

with:

```python
    if cs.get("level_complete") and not ps.get("level_complete"):
        r.add("level_complete", cfg.level_complete)

    return r
```

- [ ] **Step 6: Update the one call in `env.py`**

In `python/ultrakill_ai/env.py` (`UltrakillEnv.step`), replace:

```python
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, route_gain, stuck, died)
```

with:

```python
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, died=died)
```

Nothing else in `env.py` changes: `route_gain` and `stuck` are still computed there (the `stuck` truncation keeps
working) until Task 7 rewrites the file.

- [ ] **Step 7: Run the new test and confirm it passes**

From `python/`:

```powershell
.venv\Scripts\python tests\test_campaign_rewards.py
```

Expected:

```
ok test_campaign_step_defaults_to_nothing_happened
ok test_campaign_terms_are_paid_on_a_step_without_a_player
ok test_cybergrind_call_has_no_campaign_parts
ok test_died_is_a_keyword_after_enemy_max_health
ok test_each_milestone_weight_multiplies_its_count
ok test_finishing_within_the_cap_beats_timing_out
ok test_level_complete_pays_100_on_the_rising_edge_only
ok test_novelty_and_path_scale_by_their_weights
ok test_route_terms_are_retired
ok test_time_is_only_charged_with_a_campaign_step
10 tests passed
```

- [ ] **Step 8: Check the env call site with a fake bridge (no game)**

The call only runs inside `UltrakillEnv.step`, which no test reaches yet. From `python/`:

```powershell
@'
from ultrakill_ai.env import EnvConfig, UltrakillEnv
from ultrakill_ai.spaces import noop_action

PLAYER = {"pos": [0.0, 0.0, 0.0], "yaw": 0.0, "pitch": 0.0, "forward": [0.0, 0.0, 1.0], "local_vel": [0.0, 0.0, 0.0],
          "hp": 100, "anti_hp": 0, "stamina": 300.0, "grounded": True, "sliding": False, "weapon_slot": 0, "dead": False}


def raw(kills):
    return {"scene": "Endless", "player": dict(PLAYER), "enemies": [], "rays": [], "ground_rays": [],
            "stats": {"kills": kills, "style": 0}, "cybergrind": {"wave": 1, "enemies_left": 3}}


class FakeClient:
    def connect(self):
        pass

    def configure(self, **settings):
        pass

    def reset(self, scene, checkpoint=False):
        return raw(0)

    def step(self, command):
        return raw(1)


env = UltrakillEnv(EnvConfig(auto_enter_arena=False))
env.client = FakeClient()
env.reset()
obs, reward, terminated, truncated, info = env.step(noop_action())
print(obs.shape, reward, terminated, truncated, info["reward_parts"])
'@ | .venv\Scripts\python -
```

Expected: `(448,) 2.0 False False {'kill': 2.0}`

- [ ] **Step 9: Run the regression suites**

From `python/`:

```powershell
.venv\Scripts\python tests\test_aim.py
.venv\Scripts\python tests\test_progress.py
.venv\Scripts\python tests\test_campaign.py
```

Expected: `test_aim.py` ends with `9 tests passed` (its `test_reward_parts_split_by_axis` calls
`compute_reward(cfg, cur, cur, {1: 10.0})`, which the new signature still accepts); `test_progress.py` ends with
`all tests passed`; `test_campaign.py` (Tasks 3-4) ends with the count Task 4's run expects (`33 tests passed`).
No tracebacks.

- [ ] **Step 10: Update CLAUDE.md**

In `CLAUDE.md` (`## Layout`), replace:

```markdown
  - `rewards.py`: reward weights and computation; `aim_errors` gives the 3-D, yaw and pitch angles off an enemy, `horizon_elevation` the enemy elevation above the horizontal (diagnostics, convention-free).
```

with:

```markdown
  - `rewards.py`: reward weights and computation; `aim_errors` gives the 3-D, yaw and pitch angles off an enemy, `horizon_elevation` the enemy elevation above the horizontal (diagnostics, convention-free). Campaign terms (`time` per decision, `checkpoint`, `arena_clear`, `door_unlock`, `novelty`, `path`, all 0 by default) are paid only when the env passes a `CampaignStep` to `compute_reward`, and are paid even on a step without a player because the env has already marked those milestones paid. `level_complete` (default 100) pays once on the rising edge in both modes. The human-route terms `route_point` and `stuck` are gone.
```

Then replace:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
```

with:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_campaign_rewards.py`: campaign reward terms, finishing within the cap beating a timeout, the `level_complete` edge and the retired route terms (no game needed).
```

Then, in `## Commands`, replace the tests bullet as Task 3 left it:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py` and `python tests/test_campaign.py` (pytest is not installed; the files also work under pytest).
```

with:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py` and `python tests/test_campaign_rewards.py` (pytest is not installed; the files also work under pytest).
```

If the bullet reads differently (an earlier task listed other files), keep every file already listed and add
`python tests/test_campaign_rewards.py` as the last one.

- [ ] **Step 11: Update README.md**

In `README.md` (`## How it works`, the Rewards list), replace:

```markdown
  - Campaign: progress along the recorded route, level completion, a penalty for getting stuck.
```

with:

```markdown
  - Campaign: a small cost per decision, level completion, each checkpoint's first activation, arena clears and door unlocks, entering cells not yet visited this episode, and new best NavMesh distance to the exit.
```

- [ ] **Step 12: Commit and push**

From the repo root:

```powershell
git add python/ultrakill_ai/rewards.py python/ultrakill_ai/env.py python/tests/test_campaign_rewards.py CLAUDE.md README.md
git commit -m "Add campaign reward terms and retire the route and stuck rewards" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the push reports `main -> main`.

### Task 6: Python `spaces.py`: mode-dependent layout and the 36-value campaign block

`ObsLayout(campaign=True)` replaces the 5 retired route values with a 36-value level block (479 inputs). Cyber Grind
keeps its 5 zeros and stays 448, so its checkpoints still load. `env.py` gets only `_pack` changed.

**Files:**
- Modify: `python/ultrakill_ai/spaces.py` (`CAMPAIGN_BLOCK`, `ObsLayout.campaign`, `mode_size`, new `campaign_block`, `pack_observation` signature and tail)
- Modify: `python/ultrakill_ai/env.py` (`UltrakillEnv._pack` only)
- Modify: `CLAUDE.md` (Layout: `spaces.py` and the new test; Commands: the tests bullet)
- Modify: `README.md` (How it works: the observations list)
- Test: `python/tests/test_spaces.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_spaces.py`:

```python
"""Observation layouts and the campaign block. No game needed:  python tests/test_spaces.py  (or pytest)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import CAMPAIGN_BLOCK, ObsLayout, campaign_block, pack_observation, yaw_frame  # noqa: E402

EXPLORE = [0.05 * (k + 1) for k in range(9)]


def player(pos=(0.0, 0.0, 0.0), yaw: float = 0.0) -> dict:
    return {
        "pos": list(pos), "yaw": yaw, "pitch": 0.0, "local_vel": [0.0, 0.0, 0.0], "hp": 100, "anti_hp": 0,
        "stamina": 300.0, "grounded": True, "sliding": False, "weapon_slot": 1, "dead": False,
    }


def level(**values) -> dict:
    """A campaign block the way the mod reports it, empty apart from the given values."""
    block = {
        "mission": 1, "difficulty": 3, "seconds": 0.0, "timer_running": False, "level_started": True,
        "level_over": False, "restarts": 0, "input_locked": False, "exit": None, "checkpoints": [],
        "path": {"status": "none"}, "locked_doors": [], "arena_enemies_alive": 0, "cleared_arenas": [],
        "unlocked_doors": [], "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]},
    }
    block.update(values)
    return block


def snapshot(campaign: dict | None = None, pos=(0.0, 0.0, 0.0), yaw: float = 0.0, cybergrind: dict | None = None) -> dict:
    obs = {"player": player(pos, yaw), "enemies": [], "rays": [10.0] * 16, "ground_rays": [1.0] * 8, "stats": {}}
    if campaign is not None:
        obs["campaign"] = campaign
    if cybergrind is not None:
        obs["cybergrind"] = cybergrind
    return obs


def exit_at(x: float, y: float, z: float) -> dict:
    return {"pos": [x, y, z], "active": True}


def close(actual, expected, tol: float = 1e-5) -> bool:
    return len(actual) == len(expected) and all(abs(a - e) <= tol for a, e in zip(actual, expected))


def test_layout_sizes():
    assert CAMPAIGN_BLOCK == 36
    assert ObsLayout().size == 448 and ObsLayout().mode_size == 7
    assert ObsLayout(campaign=True).size == 479 and ObsLayout(campaign=True).mode_size == 38
    assert ObsLayout(campaign=True).space().shape == (479,)
    assert "campaign" not in ObsLayout(campaign=True).mod_config()  # the mod sends the block whatever the layout


def test_cybergrind_packing_leaves_the_last_5_values_zero():
    obs = snapshot(level(exit=exit_at(0.0, 0.0, 50.0), timer_running=True, seconds=60.0), cybergrind={"wave": 3, "enemies_left": 6})
    out = pack_observation(obs, ObsLayout(), {}, explore=EXPLORE)
    assert out.shape == (448,)
    assert close(out[441:443], [0.1, 0.2])
    assert not out[443:].any()


def test_campaign_block_is_packed_after_the_unchanged_prefix():
    obs = snapshot(level(exit=exit_at(0.0, 0.0, 50.0), timer_running=True, seconds=60.0), pos=(3.0, 1.0, -2.0), yaw=20.0)
    grind = pack_observation(obs, ObsLayout(), {})
    out = pack_observation(obs, ObsLayout(campaign=True), {}, explore=EXPLORE)
    assert out.shape == (479,)
    assert np.array_equal(out[:443], grind[:443])  # player 0-16, enemies 17-416, rays 417-432, ground rays 433-440, Cyber Grind 441-442
    assert close(out[443:], campaign_block(obs, EXPLORE))
    assert out[443 + 4] == 1.0 and out[443 + 24] == 1.0  # exit mask, timer running


def test_exit_straight_ahead_at_yaw_0():
    block = campaign_block(snapshot(level(exit=exit_at(10.0, 2.0, 45.0)), pos=(10.0, 2.0, -5.0)))
    assert close(block[0:5], [0.0, 0.0, 0.5, 0.25, 1.0])


def test_exit_follows_the_yaw_frame():
    ahead = campaign_block(snapshot(level(exit=exit_at(50.0, 0.0, 0.0)), yaw=90.0))
    assert close(ahead[0:5], [0.0, 0.0, 0.5, 0.25, 1.0])  # yaw 90 faces +x
    right = campaign_block(snapshot(level(exit=exit_at(0.0, 0.0, -30.0)), yaw=90.0))
    assert close(right[0:5], [0.3, 0.0, 0.0, 0.15, 1.0])  # facing +x, -z is on the right
    assert close(campaign_block(snapshot(level(exit=exit_at(30.0, 0.0, 0.0))))[0:3], [0.3, 0.0, 0.0])  # yaw 0: +x is right
    above = campaign_block(snapshot(level(exit=exit_at(0.0, 20.0, 0.0)), yaw=-135.0))
    assert close(above[0:4], [0.0, 0.2, 0.0, 0.1])
    delta, yaw = (12.0, -4.0, 30.0), 37.0
    exact = campaign_block(snapshot(level(exit=exit_at(*delta)), yaw=yaw))
    assert close(exact[0:4], [v / 100.0 for v in yaw_frame(delta, yaw)] + [math.sqrt(sum(d * d for d in delta)) / 200.0])


def test_every_target_uses_the_yaw_frame():
    pos, yaw, target = (4.0, 1.0, -3.0), 90.0, (4.0, 3.0, 17.0)  # 20 m along +z and 2 m up; facing +x, +z is on the left
    rel = yaw_frame(tuple(t - p for t, p in zip(target, pos)), yaw)
    assert close(rel, [-20.0, 2.0, 0.0])
    block = campaign_block(snapshot(level(
        exit=exit_at(*target),
        path={"status": "complete", "length": 40.0, "next_corner": list(target)},
        checkpoints=[{"id": "4,3,17", "pos": list(target), "activated": False, "current": False}],
        locked_doors=[{"pos": list(target), "dist": 20.1}],
    ), pos=pos, yaw=yaw))
    for start, scale in ((0, 100.0), (5, 50.0), (13, 100.0), (18, 50.0)):  # exit, path corner, checkpoint, door
        assert close(block[start : start + 3], [v / scale for v in rel])


def test_missing_exit_gives_mask_0():
    assert close(campaign_block(snapshot(level(exit=None)))[0:5], [0.0] * 5)
    no_key = level()
    del no_key["exit"]
    assert close(campaign_block(snapshot(no_key))[0:5], [0.0] * 5)


def test_path_complete_partial_and_none():
    complete = campaign_block(snapshot(level(path={"status": "complete", "length": 84.2, "next_corner": [0.0, 0.0, 10.0]})))
    assert close(complete[5:13], [0.0, 0.0, 0.2, 0.2, 84.2 / 300.0, 1.0, 0.0, 0.0])
    partial = campaign_block(snapshot(level(path={"status": "partial", "length": 30.0, "next_corner": [-5.0, 0.0, 0.0]})))
    assert close(partial[5:13], [-0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 1.0, 0.0])
    none = campaign_block(snapshot(level(path={"status": "none"})))
    assert close(none[5:13], [0.0] * 5 + [0.0, 0.0, 1.0])
    missing = level()
    del missing["path"]
    assert close(campaign_block(snapshot(missing))[5:13], [0.0] * 5 + [0.0, 0.0, 1.0])


def test_checkpoint_slot_picks_the_nearest_pending_checkpoint():
    checkpoints = [
        {"id": "0,0,5", "pos": [0.0, 0.0, 5.0], "activated": True, "current": False},
        {"id": "0,0,-8", "pos": [0.0, 0.0, -8.0], "activated": False, "current": True},
        {"id": "0,0,90", "pos": [0.0, 0.0, 90.0], "activated": False, "current": False},
        {"id": "0,0,-40", "pos": [0.0, 0.0, -40.0], "activated": False, "current": False},
    ]
    block = campaign_block(snapshot(level(checkpoints=checkpoints)))
    assert close(block[13:18], [0.0, 0.0, -0.4, 0.2, 1.0])
    reached = campaign_block(snapshot(level(checkpoints=checkpoints[:2])))
    assert close(reached[13:18], [0.0] * 5)


def test_locked_door_slot_uses_the_first_entry():
    doors = [{"pos": [5.0, 0.0, 0.0], "dist": 5.0}, {"pos": [0.0, 0.0, 2.0], "dist": 2.0}]
    block = campaign_block(snapshot(level(locked_doors=doors)))
    assert close(block[18:23], [0.1, 0.0, 0.0, 0.05, 1.0])  # the mod sorts them nearest first; the block keeps its order
    assert close(campaign_block(snapshot(level(locked_doors=[])))[18:23], [0.0] * 5)


def test_arena_timer_input_and_seconds():
    block = campaign_block(snapshot(level(arena_enemies_alive=5, timer_running=True, input_locked=True, seconds=120.0)))
    assert close(block[23:27], [0.25, 1.0, 1.0, 0.2])
    assert close(campaign_block(snapshot(level()))[23:27], [0.0, 0.0, 0.0, 0.0])


def test_explore_features_fill_the_last_9():
    assert close(campaign_block(snapshot(level()), EXPLORE)[27:36], EXPLORE)
    assert close(campaign_block(snapshot(level()))[27:36], [0.0] * 9)
    longer = campaign_block(snapshot(level()), EXPLORE + [1.0])
    assert len(longer) == CAMPAIGN_BLOCK and close(longer[27:36], EXPLORE)


def test_no_campaign_block_gives_36_zeros():
    assert campaign_block(snapshot(), EXPLORE) == [0.0] * 36
    no_player = snapshot(level(exit=exit_at(0.0, 0.0, 50.0)))
    del no_player["player"]
    assert campaign_block(no_player, EXPLORE) == [0.0] * 36
    out = pack_observation(snapshot(), ObsLayout(campaign=True), {}, explore=EXPLORE)
    assert out.shape == (479,) and not out[443:].any()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test and confirm it fails**

From `python/`:

```powershell
.venv\Scripts\python tests\test_spaces.py
```

Expected: a traceback ending in
`ImportError: cannot import name 'CAMPAIGN_BLOCK' from 'ultrakill_ai.spaces' (F:\Github\ULTRAKILL-AI\python\ultrakill_ai\spaces.py)`.

- [ ] **Step 3: Add `CAMPAIGN_BLOCK`**

In `python/ultrakill_ai/spaces.py`, replace:

```python
NUM_ENEMY_TYPES = 43  # EnemyType enum values 0..42 in the current game build
NUM_WEAPON_SLOTS = 6
```

with:

```python
NUM_ENEMY_TYPES = 43  # EnemyType enum values 0..42 in the current game build
NUM_WEAPON_SLOTS = 6
CAMPAIGN_BLOCK = 36  # campaign values packed after the Cyber Grind two (see campaign_block)
```

- [ ] **Step 4: Add `ObsLayout.campaign`**

In `python/ultrakill_ai/spaces.py`, replace:

```python
    ray_length: float = 50.0
    ground_ray_length: float = 30.0

    @property
    def player_size(self) -> int:
```

with:

```python
    ray_length: float = 50.0
    ground_ray_length: float = 30.0
    campaign: bool = False  # campaign levels: the level block replaces the 5 retired route values (448 -> 479)

    @property
    def player_size(self) -> int:
```

- [ ] **Step 5: Make `mode_size` depend on the mode**

In `python/ultrakill_ai/spaces.py`, replace:

```python
    @property
    def mode_size(self) -> int:
        return 2 + 5  # cybergrind (wave, enemies_left) + campaign (waypoint rel xyz, dist, progress)
```

with:

```python
    @property
    def mode_size(self) -> int:
        # Cyber Grind (wave, enemies_left), then the campaign block, or 5 zeros where the retired route waypoint
        # used to be, so 448-input Cyber Grind checkpoints still load.
        return 2 + (CAMPAIGN_BLOCK if self.campaign else 5)
```

- [ ] **Step 6: Add `campaign_block` and change the `pack_observation` signature**

In `python/ultrakill_ai/spaces.py`, replace:

```python
def pack_observation(
    obs: dict[str, Any],
    layout: ObsLayout,
    enemy_max_health: dict[int, float],
    waypoint: tuple[float, float, float] | None = None,
    route_progress: float = 0.0,
) -> np.ndarray:
```

with:

```python
def campaign_block(obs: dict[str, Any], explore: list[float] | None = None) -> list[float]:
    """The 36 campaign values, all zero without a `campaign` block or a player.

    Targets are relative to the player in its yaw frame (x right, y up, z forward), so "ahead" is +z whichever way
    the player faces.

        0-4    exit: rel xyz / 100, distance / 200, mask
        5-9    path next corner: rel xyz / 50, distance / 50, path length / 300 (complete or partial path only)
        10-12  path status one-hot: complete, partial, none
        13-17  nearest checkpoint neither activated nor current: rel xyz / 100, distance / 200, mask
        18-22  first locked door (the mod sends them nearest first): rel xyz / 50, distance / 100, mask
        23-26  arena enemies alive / 20, timer running, input locked, level seconds / 600
        27-35  exploration map (ExplorationArchive.features: current cell, then 8 neighbours from straight ahead)
    """
    out = [0.0] * CAMPAIGN_BLOCK
    c, p = obs.get("campaign"), obs.get("player")
    if not c or not p:
        return out
    pos, yaw = p["pos"], p["yaw"]

    def relative(point, rel_scale: float, dist_scale: float) -> list[float]:
        x, y, z = yaw_frame((point[0] - pos[0], point[1] - pos[1], point[2] - pos[2]), yaw)
        return [x / rel_scale, y / rel_scale, z / rel_scale, math.sqrt(x * x + y * y + z * z) / dist_scale]

    exit_ = c.get("exit")
    if exit_:
        out[0:5] = relative(exit_["pos"], 100.0, 200.0) + [1.0]

    path = c.get("path") or {}
    status = path.get("status", "none")
    if status in ("complete", "partial") and path.get("next_corner"):
        out[5:10] = relative(path["next_corner"], 50.0, 50.0) + [path.get("length", 0.0) / 300.0]
    out[10:13] = [float(status == "complete"), float(status == "partial"), float(status not in ("complete", "partial"))]

    pending = [cp for cp in (c.get("checkpoints") or []) if not cp["activated"] and not cp["current"]]
    if pending:
        nearest = min(pending, key=lambda cp: math.dist(cp["pos"], pos))
        out[13:18] = relative(nearest["pos"], 100.0, 200.0) + [1.0]

    doors = c.get("locked_doors") or []
    if doors:
        out[18:23] = relative(doors[0]["pos"], 50.0, 100.0) + [1.0]

    out[23:27] = [
        c.get("arena_enemies_alive", 0) / 20.0,
        float(bool(c.get("timer_running"))),
        float(bool(c.get("input_locked"))),
        c.get("seconds", 0.0) / 600.0,
    ]
    if explore is not None:
        values = [float(v) for v in explore[:9]]
        out[27 : 27 + len(values)] = values
    return out


def pack_observation(
    obs: dict[str, Any],
    layout: ObsLayout,
    enemy_max_health: dict[int, float],
    explore: list[float] | None = None,
) -> np.ndarray:
```

- [ ] **Step 7: Pack the campaign block (or the 5 zeros) at the end**

In `python/ultrakill_ai/spaces.py`, replace:

```python
    cg = obs.get("cybergrind")
    put([cg["wave"] / 30.0, max(cg["enemies_left"], 0) / 30.0] if cg else [0.0, 0.0])

    if waypoint is not None:
        pos = p["pos"]
        delta = (waypoint[0] - pos[0], waypoint[1] - pos[1], waypoint[2] - pos[2])
        lx, ly, lz = yaw_frame(delta, p["yaw"])
        dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        put([lx / 50.0, ly / 50.0, lz / 50.0, dist / 100.0, route_progress])
    else:
        put([0.0] * 5)
```

with:

```python
    cg = obs.get("cybergrind")
    put([cg["wave"] / 30.0, max(cg["enemies_left"], 0) / 30.0] if cg else [0.0, 0.0])

    # Cyber Grind keeps 5 zeros where the route waypoint used to be, so its 448-input checkpoints still load.
    put(campaign_block(obs, explore) if layout.campaign else [0.0] * 5)
```

- [ ] **Step 8: Update `_pack` in `env.py`**

In `python/ultrakill_ai/env.py`, replace:

```python
    def _pack(self, raw: dict[str, Any]) -> np.ndarray:
        waypoint = self.route_tracker.waypoint if self.route_tracker else None
        progress = self.route_tracker.progress if self.route_tracker else 0.0
        return pack_observation(raw, self.cfg.layout, self._enemy_max_health, waypoint, progress)
```

with:

```python
    def _pack(self, raw: dict[str, Any]) -> np.ndarray:
        return pack_observation(raw, self.cfg.layout, self._enemy_max_health)
```

Nothing else in `env.py` changes (Task 7 adds the exploration features).

- [ ] **Step 9: Run the new test and confirm it passes**

From `python/`:

```powershell
.venv\Scripts\python tests\test_spaces.py
```

Expected:

```
ok test_arena_timer_input_and_seconds
ok test_campaign_block_is_packed_after_the_unchanged_prefix
ok test_checkpoint_slot_picks_the_nearest_pending_checkpoint
ok test_cybergrind_packing_leaves_the_last_5_values_zero
ok test_every_target_uses_the_yaw_frame
ok test_exit_follows_the_yaw_frame
ok test_exit_straight_ahead_at_yaw_0
ok test_explore_features_fill_the_last_9
ok test_layout_sizes
ok test_locked_door_slot_uses_the_first_entry
ok test_missing_exit_gives_mask_0
ok test_no_campaign_block_gives_36_zeros
ok test_path_complete_partial_and_none
13 tests passed
```

- [ ] **Step 10: Check `_pack` with a fake bridge (no game)**

From `python/`:

```powershell
@'
from ultrakill_ai.env import EnvConfig, UltrakillEnv
from ultrakill_ai.spaces import noop_action

PLAYER = {"pos": [0.0, 0.0, 0.0], "yaw": 0.0, "pitch": 0.0, "forward": [0.0, 0.0, 1.0], "local_vel": [0.0, 0.0, 0.0],
          "hp": 100, "anti_hp": 0, "stamina": 300.0, "grounded": True, "sliding": False, "weapon_slot": 0, "dead": False}


def raw(kills):
    return {"scene": "Endless", "player": dict(PLAYER), "enemies": [], "rays": [], "ground_rays": [],
            "stats": {"kills": kills, "style": 0}, "cybergrind": {"wave": 1, "enemies_left": 3}}


class FakeClient:
    def connect(self):
        pass

    def configure(self, **settings):
        pass

    def reset(self, scene, checkpoint=False):
        return raw(0)

    def step(self, command):
        return raw(1)


env = UltrakillEnv(EnvConfig(auto_enter_arena=False))
env.client = FakeClient()
env.reset()
obs, reward, terminated, truncated, info = env.step(noop_action())
print(obs.shape, reward, terminated, truncated, info["reward_parts"])
'@ | .venv\Scripts\python -
```

Expected: `(448,) 2.0 False False {'kill': 2.0}`

- [ ] **Step 11: Run the regression suites**

From `python/`:

```powershell
.venv\Scripts\python tests\test_aim.py
.venv\Scripts\python tests\test_progress.py
.venv\Scripts\python tests\test_campaign.py
.venv\Scripts\python tests\test_campaign_rewards.py
```

Expected: `test_aim.py` ends with `9 tests passed`; `test_progress.py` ends with `all tests passed` (its fake env
still builds a 448-dim `ObsLayout().space()`); `test_campaign.py` ends with the count Task 4's run expects
(`33 tests passed`); `test_campaign_rewards.py` ends with `10 tests passed`. No tracebacks.

- [ ] **Step 12: Update CLAUDE.md**

In `CLAUDE.md` (`## Layout`), replace:

```markdown
  - `spaces.py`: 448-dim obs packing, `MultiDiscrete` actions.
```

with:

```markdown
  - `spaces.py`: obs packing and `MultiDiscrete` actions. Cyber Grind is 448 dims, with 5 zeros where the retired route waypoint was so its checkpoints load; `ObsLayout(campaign=True)` is 479, replacing those 5 with the 36-value `campaign_block` (exit, path next corner and status, nearest checkpoint neither activated nor current, first locked door, arena enemies / timer / input lock / level seconds, and 9 exploration-map values). Every index before it is unchanged.
```

Then replace:

```markdown
- `python/tests/test_campaign_rewards.py`: campaign reward terms, finishing within the cap beating a timeout, the `level_complete` edge and the retired route terms (no game needed).
```

with:

```markdown
- `python/tests/test_campaign_rewards.py`: campaign reward terms, finishing within the cap beating a timeout, the `level_complete` edge and the retired route terms (no game needed).
- `python/tests/test_spaces.py`: layout sizes (448 / 479) and every index range of the campaign block, including the yaw-frame signs (no game needed).
```

Then, in `## Commands`, replace the tests bullet as Task 5 left it:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py` and `python tests/test_campaign_rewards.py` (pytest is not installed; the files also work under pytest).
```

with:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py`, `python tests/test_campaign_rewards.py` and `python tests/test_spaces.py` (pytest is not installed; the files also work under pytest).
```

If the bullet reads differently (an earlier task listed other files), keep every file already listed and add
`python tests/test_spaces.py` as the last one.

- [ ] **Step 13: Update README.md**

In `README.md` (`## How it works`, the Observations list), replace:

```markdown
- **Observations:** the policy gets a fixed 448-float vector.
```

with:

```markdown
- **Observations:** the policy gets a fixed float vector: 448 values in Cyber Grind, 479 in campaign levels.
```

Then replace:

```markdown
  - Campaign only: direction to the next route waypoint.
```

with:

```markdown
  - Campaign only (36 values): the exit, the next NavMesh path corner and path status, the nearest checkpoint not yet reached, the nearest locked door, arena enemies, timer and input lock, and a 9-cell exploration map. Cyber Grind keeps 5 zeros in their place.
```

- [ ] **Step 14: Commit and push**

From the repo root:

```powershell
git add python/ultrakill_ai/spaces.py python/ultrakill_ai/env.py python/tests/test_spaces.py CLAUDE.md README.md
git commit -m "Pack a 36-value campaign block in a 479-input campaign layout" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the push reports `main -> main`.

### Task 7: Python `env.py`: campaign episodes (with a fake-bridge test)

**Files:**
- Modify: `python/ultrakill_ai/env.py` (complete new file: campaign episodes, `CAMPAIGN_INFO_KEYS`, `_known_fields`; the route tracker is removed; every Cyber Grind behaviour and comment is kept)
- Create: `python/tests/test_campaign_env.py`
- Modify: `CLAUDE.md` (Layout: the `env.py` entry and the new test file; Commands: the tests bullet)
- Test: `python/tests/test_campaign_env.py`

Before this task `env.py` still imports `ultrakill_ai.routes` and carries the Task 5 and Task 6 one-line edits (`compute_reward(..., died=died)` and the 3-argument `pack_observation` call). `campaign.py` (Tasks 3-4), `CampaignStep` and the campaign reward terms (Task 5) and `ObsLayout.campaign` / the 36-value block (Task 6) exist.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_campaign_env.py`. `FakeLevel` implements the BridgeClient methods the env calls (`connect`, `configure`, `reset`, `step`, `get_obs`, `close`) for a corridor along +z, as the contract describes. Besides `level_over` it sets `stats.level_complete`, because the env's end rule reads the stats flag (the mod reports both). It also carries a `kills` counter that a reload clears, for the death-before-any-checkpoint test.

```python
"""Campaign episodes in UltrakillEnv, against a fake level instead of the game:  python tests/test_campaign_env.py  (or pytest)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402
from ultrakill_ai.spaces import noop_action  # noqa: E402

LEVEL = "Level 0-1"
EXIT_Z = 60.0
CHECKPOINT_ID = "0,1,20"
ARENA_KEY = "0,1,30"
RESPAWN_DOOR_KEY = "0,0,25"
RANKS = {"time": [120, 90, 60, 30], "kills": [0, 1, 2, 3], "style": [0, 100, 200, 300]}


class FakeLevel:
    """Stands in for BridgeClient: a straight corridor along +z.

    Walking forward covers 2 m a step. The checkpoint at z 20 activates (and becomes current) on arrival, the
    arena at z 30 clears on arrival, and the exit is at z 60. A checkpoint respawn puts the player back at z 20
    and unlocks a door, the way `StatsManager.Restart` unlocks `doorsToUnlock`: that unlock must never pay.
    """

    def __init__(self):
        self.resets: list[bool] = []  # the checkpoint flag of every reset
        self.steps = 0
        self.kill_next = False  # the next step returns a dead player
        self.lock_steps = 0  # the next N steps report input_locked
        self._load()

    def _load(self) -> None:
        self.z = 0.0
        self.seconds = 0.0
        self.kills = 0
        self.restarts = 0
        self.dead = False
        self.locked = False
        self.checkpoint = False
        self.arenas: list[str] = []
        self.doors: list[str] = []

    # The BridgeClient methods UltrakillEnv uses ----------------------------

    def connect(self, retry_seconds: float = 60.0) -> dict:
        return {"type": "hello", "protocol": 1, "mod_version": "0.5.0", "scene": LEVEL}

    def configure(self, **settings) -> None:
        self.settings = settings

    def close(self) -> None:
        pass

    def get_obs(self) -> dict:
        return self._obs()

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict:
        self.resets.append(checkpoint)
        if checkpoint and self.checkpoint:
            self.z = 20.0
            self.dead = False
            self.restarts += 1
            if RESPAWN_DOOR_KEY not in self.doors:
                self.doors.append(RESPAWN_DOOR_KEY)
        else:
            self._load()
        return self._obs("reset")

    def step(self, action: dict) -> dict:
        self.steps += 1
        self.locked = self.lock_steps > 0
        if self.locked:
            self.lock_steps -= 1
        if not self.dead and self.z < EXIT_Z:
            if self.kill_next:
                self.dead, self.kill_next = True, False
            elif not self.locked and action.get("move", [0, 0])[1] > 0:
                self.z = min(EXIT_Z, self.z + 2.0)
            self.seconds += 2.0 / 15.0
        if self.z >= 20.0:
            self.checkpoint = True
        if self.z >= 30.0 and ARENA_KEY not in self.arenas:
            self.arenas.append(ARENA_KEY)
        return self._obs()

    def _obs(self, event: str | None = None) -> dict:
        over = self.z >= EXIT_Z
        obs = {
            "type": "obs",
            "step": self.steps,
            "scene": LEVEL,
            "ready": not self.dead,
            "player": {
                "pos": [0.0, 1.0, self.z], "vel": [0.0, 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, 1.0], "yaw": 0.0, "pitch": 0.0, "hp": 0 if self.dead else 100,
                "anti_hp": 0.0, "stamina": 300.0, "grounded": True, "sliding": False, "dead": self.dead,
                "activated": True, "level_over": over, "weapon_slot": 0, "weapon_variation": 0,
                "soft_deaths": 0, "soft_death_instakill": False, "slot_counts": [1, 0, 0, 0, 0],
            },
            "enemies": [],
            "rays": [50.0] * 16,
            "ground_rays": [0.0] * 8,
            "stats": {"kills": self.kills, "style": 0, "seconds": self.seconds, "restarts": self.restarts, "level_complete": over},
            "campaign": {
                "mission": 1, "difficulty": 3, "seconds": self.seconds, "timer_running": not over,
                "level_started": True, "level_over": over, "restarts": self.restarts, "input_locked": self.locked,
                "exit": {"pos": [0.0, 1.0, EXIT_Z], "active": True},
                "checkpoints": [{"id": CHECKPOINT_ID, "pos": [0.0, 1.0, 20.0], "activated": self.checkpoint, "current": self.checkpoint}],
                "path": {"status": "complete", "length": EXIT_Z - self.z, "next_corner": [0.0, 1.0, EXIT_Z]},
                "locked_doors": [],
                "arena_enemies_alive": 0,
                "cleared_arenas": list(self.arenas),
                "unlocked_doors": list(self.doors),
                "ranks": RANKS,
            },
        }
        if event:
            obs["event"] = event
        return obs


def make_env(**overrides) -> tuple[UltrakillEnv, FakeLevel]:
    rewards = RewardConfig(time=0.01, checkpoint=10.0, arena_clear=10.0, door_unlock=3.0, novelty=0.5, path=0.1)
    cfg = EnvConfig(mode="campaign", level=LEVEL, fixed_fps=30, frameskip=2, rewards=rewards, **overrides)
    env = UltrakillEnv(cfg)
    env.client = FakeLevel()
    return env, env.client


def forward():
    a = noop_action()
    a[0] = 2  # move forward
    return a


def add_parts(total: dict[str, float], info: dict) -> None:
    for name, value in info["reward_parts"].items():
        total[name] = total.get(name, 0.0) + value


def run_until_end(env: UltrakillEnv, limit: int = 500) -> dict:
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(noop_action())
        if terminated or truncated:
            return info
    raise AssertionError(f"episode did not end within {limit} steps")


def test_fresh_start_walked_to_the_exit_completes_the_level():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(best_runs_dir=str(Path(tmp) / "best_runs"), difficulty=3, unlock_all_gear=True)
        _, info = env.reset(seed=0)
        assert fake.resets == [False]
        assert fake.settings["difficulty"] == 3 and fake.settings["unlock_all_gear"] is True
        assert fake.settings["soft_death"] is False  # real deaths drive respawns in the campaign
        assert info["fresh_start"] == 1 and info["completed"] == 0 and info["level_seconds"] is None
        parts: dict[str, float] = {}
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["end_reason"] == "level_complete"
        assert {"checkpoint", "arena_clear", "level_complete", "time"} <= set(parts)
        assert parts["checkpoint"] == 10.0 and parts["arena_clear"] == 10.0  # each paid exactly once
        assert all(key in info for key in CAMPAIGN_INFO_KEYS)
        assert info["completed"] == 1 and info["fresh_start"] == 1
        assert info["level_seconds"] == fake.seconds
        assert isinstance(info["rank"], str)
        assert info["checkpoints_level"] == 1 and info["cells_new"] > 0 and info["exit_dist_min"] == 0.0
        env.close()

        run = json.loads((Path(tmp) / "best_runs" / "Level_0-1.json").read_text(encoding="utf-8"))
        assert run["level"] == LEVEL and run["seconds"] == fake.seconds and run["deaths"] == 0
        assert run["positions"][0] == [0.0, 1.0, 0.0] and run["positions"][-1] == [0.0, 1.0, EXIT_Z]


def test_completion_after_a_checkpoint_respawn_has_no_official_time():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(fresh_start_prob=0.0, max_steps=25, best_runs_dir=tmp)
        env.reset(seed=0)
        for _ in range(25):  # z 50: past the checkpoint, short of the exit, then the 25-decision cap
            _, _, terminated, truncated, info = env.step(forward())
        assert truncated and info["end_reason"] == "max_steps"
        _, info = env.reset()
        assert fake.resets == [False, True] and info["fresh_start"] == 0 and fake.z == 20.0
        for _ in range(25):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["completed"] == 1
        assert info["level_seconds"] is None and "rank" not in info  # the timer ran on from the earlier episode
        env.close()
        assert not (Path(tmp) / "Level_0-1.json").exists()


def test_death_after_the_checkpoint_respawns_inside_the_episode():
    env, fake = make_env()
    env.reset(seed=0)
    parts: dict[str, float] = {}
    for _ in range(11):  # z 22, just past the checkpoint
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
    fake.kill_next = True
    _, _, terminated, truncated, info = env.step(noop_action())
    add_parts(parts, info)
    assert not terminated and not truncated
    assert fake.resets == [False, True]
    assert info["deaths"] == 1 and "death" in info["reward_parts"]
    assert RESPAWN_DOOR_KEY in env._raw["campaign"]["unlocked_doors"]
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert "door_unlock" not in parts  # the respawn's own unlock pays nothing
    assert parts["checkpoint"] == 10.0  # and the respawn checkpoint is not paid again
    assert info["end_reason"] == "level_complete" and info["deaths"] == 1 and info["checkpoints_level"] == 1
    env.close()


def test_death_before_any_checkpoint_reloads_the_level_inside_the_episode():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(best_runs_dir=tmp)
        env.reset(seed=0)
        for _ in range(5):  # z 10, short of the checkpoint
            env.step(forward())
        fake.kills = 2
        fake.kill_next = True
        _, _, terminated, truncated, info = env.step(noop_action())
        assert not terminated and not truncated
        assert fake.resets == [False, True] and fake.z == 0.0  # no checkpoint yet, so the level reloaded
        assert info["deaths"] == 1 and info["kills"] == 2  # kills from before the reload still count
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["level_seconds"] == fake.seconds and info["kills"] == 2
        env.close()
        run = json.loads((Path(tmp) / "Level_0-1.json").read_text(encoding="utf-8"))
        assert len(run["positions"]) == 31 and run["positions"][0] == [0.0, 1.0, 0.0]  # only the attempt that finished


def test_three_stuck_episodes_at_one_checkpoint_force_a_fresh_load():
    env, fake = make_env(fresh_start_prob=0.0, stuck_seconds=1.0)  # 15 decisions without progress
    env.reset(seed=0)
    for _ in range(10):  # z 20: the checkpoint activates
        env.step(forward())
    assert run_until_end(env)["end_reason"] == "stuck"
    for _ in range(2):
        _, info = env.reset()
        assert info["fresh_start"] == 0
        assert run_until_end(env)["end_reason"] == "stuck"
    _, info = env.reset()
    assert fake.resets == [False, True, True, False]
    assert info["fresh_start"] == 1
    env.close()


def test_input_locked_frames_are_skipped():
    env, fake = make_env()
    env.reset(seed=0)
    fake_steps, env_steps = fake.steps, env._steps
    fake.lock_steps = 3
    env.step(noop_action())
    assert fake.steps - fake_steps == 3 + 1  # the policy's step, then empty steps until the lock ends
    assert env._steps - env_steps == 1
    assert env._raw["campaign"]["input_locked"] is False
    env.close()


def test_campaign_observation_is_479_values():
    env, _ = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (479,) and env.observation_space.shape == (479,)
    assert env.observation_space.contains(obs)
    assert obs[443 + 4] == 1.0  # exit mask: the campaign block starts after the 2 Cyber Grind values
    assert obs[443 + 27] > 0.0  # exploration map: the spawn cell has been entered once
    obs, *_ = env.step(forward())
    assert env.observation_space.contains(obs)
    env.close()


def test_cybergrind_default_stays_448_without_connecting():
    assert EnvConfig().layout.size == 448
    env = UltrakillEnv(EnvConfig())
    assert env.observation_space.shape == (448,)
    assert not env._connected and env.client._sock is None
    cfg = EnvConfig(mode="campaign")
    campaign = UltrakillEnv(cfg)
    assert campaign.observation_space.shape == (479,) and not campaign._connected
    assert campaign.cfg.layout.campaign and not cfg.layout.campaign  # the caller's config is left alone


def test_from_dict_ignores_retired_keys():
    cfg = EnvConfig.from_dict({
        "mode": "campaign",
        "fixed_fps": 30,
        "checkpoint_resets": True,
        "stuck_steps": 450,
        "route_dir": "routes",
        "layout": {"max_enemies": 8, "waypoint_offset": 3},
        "rewards": {"kill": 0.5, "route_point": 0.1, "stuck": 1.0},
    })
    assert cfg.mode == "campaign" and cfg.fixed_fps == 30 and cfg.layout.max_enemies == 8 and cfg.rewards.kill == 0.5
    for retired in ("checkpoint_resets", "stuck_steps", "route_dir"):
        assert not hasattr(cfg, retired)
    assert not hasattr(cfg.rewards, "route_point") and not hasattr(cfg.rewards, "stuck")
    assert EnvConfig.from_dict(cfg.to_dict()) == cfg


def test_exploration_archive_is_saved_on_close_and_loaded_again():
    with tempfile.TemporaryDirectory() as tmp:
        env, _ = make_env(explore_dir=tmp)
        env.reset(seed=0)
        for _ in range(5):
            env.step(forward())
        counts = dict(env.archive.counts)
        assert counts
        env.close()
        assert (Path(tmp) / f"explore_Level_0-1_{env.cfg.port}.npz").exists()
        again = UltrakillEnv(env.cfg)
        assert again.archive.counts == counts


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_env.py
```

Expected: a traceback ending in

```text
ImportError: cannot import name 'CAMPAIGN_INFO_KEYS' from 'ultrakill_ai.env' (F:\Github\ULTRAKILL-AI\python\ultrakill_ai\env.py)
```

- [ ] **Step 3: Replace `python/ultrakill_ai/env.py` with the complete new file**

What changes against the current file (everything else is byte-for-byte the same code and comments):

- Imports: `random`, `fields`, `replace`, the `campaign` helpers and `CampaignStep`; the `routes` import is gone.
- `CAMPAIGN_INFO_KEYS` and `_known_fields` at module level; `EnvConfig.from_dict` filters `EnvConfig`, `ObsLayout` and `RewardConfig` keys through it.
- `EnvConfig`: `checkpoint_resets`, `stuck_steps` and `route_dir` deleted; the nine campaign settings added.
- `__init__`: the layout follows the mode (a copy via `dataclasses.replace`, so the caller's config and the layout object that `train.py`'s per-port `replace` copies share are never mutated); campaign state replaces the route-file check.
- `_ensure_connected` also sends `difficulty` and `unlock_all_gear`.
- `reset`: campaign goes through `_campaign_reset`, then starts the archive episode (the spawn cell is visited without reward), path progress, positions, `cells_new` and `exit_dist_min`. The Cyber Grind branch is the old code with `checkpoint=False` (it was always False outside the campaign) and without the now-redundant `mode == "cybergrind"` test.
- `step`: campaign skips input-locked frames, measures `_campaign_progress` before the reward, respawns after a death; the soft-death teleport branch is unchanged and now sits under `elif`, which only ever ran in Cyber Grind (campaign never enables soft death). `stuck` is the game-time rule. Episode ends call `_end_campaign_episode`.
- `close` saves the archive first (in a `try`/`finally`, so the game is released even if the save fails).
- New campaign helpers between `_track_enemies` and `_pack`; `_pack` passes the exploration features; `_info` drops `route_progress` and adds the campaign keys.
- One addition beyond the contract, in `_respawn`: when the respawn raw reports no current checkpoint, `StatsManager.Restart` reloaded the level (it calls `SceneHelper.RestartSceneAsync` with no checkpoint), so the episode's `kills`/`style` baseline is rebased (the game's counters restart at 0) and a best run's `positions` restart at the spawn, matching the reloaded attempt's official time. Milestones are still only marked paid, so re-clearing an arena after a reload cannot farm reward.

```python
"""Gymnasium environment for ULTRAKILL (Cyber Grind and campaign levels)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ultrakill_ai.campaign import (
    ExplorationArchive,
    MilestoneTracker,
    PathProgress,
    choose_fresh_start,
    compute_rank,
    safe_name,
    save_best_run,
)
from ultrakill_ai.protocol import DEFAULT_PORT, BridgeClient
from ultrakill_ai.rewards import CampaignStep, RewardConfig, aim_errors, compute_reward, horizon_elevation
from ultrakill_ai.spaces import ObsLayout, action_space, decode_action, pack_observation

CYBERGRIND_SCENE = "Endless"
# Per-episode info the campaign Monitor records (scripts/train.py); every key is in every campaign info.
CAMPAIGN_INFO_KEYS = ("kills", "style", "deaths", "completed", "fresh_start", "level_seconds",
                      "checkpoints_level", "cells_new", "exit_dist_min")


def _known_fields(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Keeps the keys that are fields of dataclass `cls`, so configs saved with retired settings still load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class EnvConfig:
    mode: str = "cybergrind"  # "cybergrind" or "campaign"
    level: str = "Level 0-1"  # campaign scene name
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    # Game speed / stepping
    frameskip: int = 4  # frames per decision (4 frames at 60 fps = 15 decisions per game second)
    fixed_fps: float = 60.0
    unlimited_fps: bool = True  # render as fast as possible so training runs faster than real time
    mute: bool = True
    block_human_input: bool = True
    windowed: bool = True  # run in a small window while training so it doesn't hold your mouse or screen
    window_width: int = 640
    window_height: int = 360

    # Episodes
    max_steps: int = 4500  # 5 minutes of game time at 15 decisions/s
    max_wave: int = 0  # Cyber Grind curriculum: end the episode once this many waves are cleared (0 = off)
    hard_reset_above_wave: int = 5  # reload the arena once waves get this far, so training keeps seeing early waves
    auto_enter_arena: bool = True  # Cyber Grind: put the player into the arena on reset so wave 1 starts
    reset_settle_frames: int = 10  # frames the player must be spawned before a reset completes
    end_episode_on_death: bool = True  # False (with soft_death) keeps the run going after a death, so episodes are a
    # fixed slice of time: total reward then measures kills per minute instead of rewarding hiding
    soft_death: bool = True  # Cyber Grind: lethal hits heal instead of killing; the episode still ends with the death
    # penalty, but the next one continues in the same arena without a scene reload
    render: bool = False  # the agent never sees pixels; turning cameras off saves CPU/GPU
    pitch_limit_deg: float = 0.0  # keep the camera within ±this many degrees of level (0 = off). Without it the
    # policy drifted to the +90° clamp and stared at the sky, so no yaw ever put an enemy near the crosshair

    # Campaign (docs/superpowers/specs/2026-09-16-campaign-foundation-design.md)
    difficulty: int = -1  # difficulty the game reads while the AI has control (3 = Violent, -1 = leave the game's own)
    unlock_all_gear: bool = False  # every weapon and variant while the AI has control, in memory only
    fresh_start_prob: float = 0.2  # chance of a fresh level load when a checkpoint respawn would also do
    stuck_seconds: float = 45.0  # game seconds without progress (milestone, new cell, shorter path) before truncating
    stuck_repeats: int = 3  # episodes in a row stuck at the same checkpoint before a fresh load is forced
    cell_size: float = 4.0  # metres per exploration cell
    max_locked_skip_s: float = 120.0  # longest input lock (landing, cutscene) stepped through without the policy
    explore_dir: str = ""  # folder for the exploration archive, so a resumed run keeps its visit counts ("" = memory only)
    best_runs_dir: str = ""  # folder for the fastest fresh-start completion of each level ("" = off)

    layout: ObsLayout = field(default_factory=ObsLayout)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        layout = ObsLayout(**_known_fields(ObsLayout, d.pop("layout", None) or {}))
        rewards = RewardConfig(**_known_fields(RewardConfig, d.pop("rewards", None) or {}))
        return cls(**_known_fields(cls, d), layout=layout, rewards=rewards)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BEHAVIOUR_KEYS = ("steps", "firing", "on_target", "firing_on_target", "enemy_visible", "close", "angle_sum", "yaw_err_sum", "dist_sum", "yaw_sum",
                  "pitch_steps", "pitch_sum", "pitch_signed_sum", "look_up_sum", "elev_steps", "elev_sum", "elev_abs_sum", "elev_over15", "pitch_err_sum",
                  "yaw_track", "yaw_track_n", "pitch_track", "pitch_track_n")


def clamp_pitch_command(current_pitch: float, pitch_cmd: float, limit: float) -> float:
    """Trims a pitch command so the camera ends within ±limit degrees of level (snapping back if it is outside)."""
    target = max(-limit, min(limit, current_pitch + pitch_cmd))
    return target - current_pitch


class UltrakillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        if self.cfg.mode not in ("cybergrind", "campaign"):
            raise ValueError(f"Unknown mode {self.cfg.mode!r}")
        campaign = self.cfg.mode == "campaign"
        if self.cfg.layout.campaign != campaign:
            # The layout follows the mode (Cyber Grind 448 inputs, campaign 479). Replaced, not edited in place:
            # train.py builds each game's config with dataclasses.replace, so they all share one layout object.
            self.cfg = replace(self.cfg, layout=replace(self.cfg.layout, campaign=campaign))

        self.observation_space = self.cfg.layout.space()
        self.action_space = action_space()

        self.client = BridgeClient(self.cfg.host, self.cfg.port)
        self._connected = False

        self._raw: dict[str, Any] = {}
        self._enemy_max_health: dict[int, float] = {}
        self._steps = 0
        self._steps_since_progress = 0
        self._last_end_reason = ""
        self._reset_seconds = 0.0
        self._deaths = 0
        self._behaviour = dict.fromkeys(BEHAVIOUR_KEYS, 0)
        self._arena_spawn: list[float] | None = None
        self._episode_start_stats: dict[str, Any] = {}

        # Campaign state. The exploration archive counts, per game and per level, how many earlier episodes
        # entered each cell, so the novelty reward fades where this game has already been.
        if campaign and self.cfg.explore_dir:
            self.archive = ExplorationArchive.load(self._archive_path(), self.cfg.cell_size)
        else:
            self.archive = ExplorationArchive(self.cfg.cell_size)
        self.milestones = MilestoneTracker()
        self.path_progress = PathProgress()
        self._rng = random.Random()
        self._stuck_streak = 0  # episodes in a row that ended stuck at the same current checkpoint
        self._stuck_checkpoint: str | None = None
        self._fresh_start = False  # this episode began with a fresh level load
        self._positions: list[list[float]] = []  # fresh-start episodes only, for best runs
        self._cells_new = 0
        self._exit_dist_min = math.inf
        self._episodes = 0

    @property
    def scene(self) -> str:
        return CYBERGRIND_SCENE if self.cfg.mode == "cybergrind" else self.cfg.level

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self.client.connect()
        mod_layout = self.cfg.layout.mod_config()
        # Ask the mod for more enemies than the policy sees so damage rewards aren't missed.
        mod_layout["max_enemies"] = max(32, self.cfg.layout.max_enemies)
        self.client.configure(
            frameskip=self.cfg.frameskip,
            fixed_fps=self.cfg.fixed_fps,
            unlimited_fps=self.cfg.unlimited_fps,
            mute=self.cfg.mute,
            block_human_input=self.cfg.block_human_input,
            reset_settle_frames=self.cfg.reset_settle_frames,
            soft_death=self.cfg.soft_death and self.cfg.mode == "cybergrind",
            render=self.cfg.render,
            windowed=self.cfg.windowed,
            window_width=self.cfg.window_width,
            window_height=self.cfg.window_height,
            # Sent before the first reset: enemies and GunSetter read both when the level loads.
            difficulty=self.cfg.difficulty,
            unlock_all_gear=self.cfg.unlock_all_gear,
            **mod_layout,
        )
        self._connected = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._ensure_connected()

        start = time.perf_counter()
        if self.cfg.mode == "campaign":
            self._raw = self._campaign_reset()
        elif self._can_soft_reset():
            self._raw = self._soft_reset()
        else:
            self._raw = self.client.reset(self.scene, checkpoint=False)
            if self.cfg.auto_enter_arena:
                self._raw = self._enter_arena(self._raw)
            if self._raw.get("player"):
                self._arena_spawn = list(self._raw["player"]["pos"])
        self._reset_seconds = time.perf_counter() - start
        self._episode_start_stats = dict(self._raw.get("stats", {}))
        self._enemy_max_health = {}
        self._track_enemies(self._raw)
        self._steps = 0
        self._steps_since_progress = 0
        self._deaths = 0
        self._behaviour = dict.fromkeys(BEHAVIOUR_KEYS, 0)
        self._last_end_reason = ""

        if self.cfg.mode == "campaign":
            self.archive.start_episode()
            self.path_progress.reset()
            self._positions = []
            self._cells_new = 0
            self._exit_dist_min = math.inf
            player = self._raw.get("player")
            if player:
                self.archive.visit(player["pos"])  # the spawn cell is entered, but pays nothing
                if self._fresh_start:
                    self._positions.append(self._rounded(player["pos"]))

        return self._pack(self._raw), self._info(self._raw)

    def step(self, action):
        prev = self._raw
        command = decode_action(action)
        raw_pitch_cmd = command["look"][1]
        if self.cfg.pitch_limit_deg and prev.get("player"):
            command["look"][1] = clamp_pitch_command(prev["player"]["pitch"], command["look"][1], self.cfg.pitch_limit_deg)
        self._note_behaviour(prev, command, raw_pitch_cmd)
        campaign = self.cfg.mode == "campaign"
        cur = self.client.step(command)
        if campaign:
            cur = self._skip_locked(cur)
        self._raw = cur
        self._steps += 1
        self._track_enemies(cur)

        player = cur.get("player")
        prev_player = prev.get("player") or {}
        died = player is None or player["dead"] or (
            player.get("soft_deaths", 0) > prev_player.get("soft_deaths", player.get("soft_deaths", 0))
        )
        # Milestones, novelty and path progress are measured before the reward, from the frame the policy caused.
        campaign_step = self._campaign_progress(cur) if campaign else None
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, died=died, campaign=campaign_step)

        if campaign:
            if died and player is not None and not (cur.get("campaign") or {}).get("level_over"):
                # A death does not end a campaign episode: the penalty is paid above, then the player respawns at
                # the checkpoint and the level clock keeps running, as in real play.
                cur = self._respawn()
                self._raw = cur
                player = cur.get("player")
                self._deaths += 1
                died = False
        elif died and not self.cfg.end_episode_on_death and player is not None and not player["dead"]:
            # Soft death inside a timed episode: stay in the run, but get out of the pit that killed us.
            if player.get("soft_death_instakill") and self._arena_spawn is not None:
                x, y, z = self._arena_spawn
                cur = self.client.teleport([x, y + 10.0, z])
                self._raw = cur
                player = cur.get("player")
                self._track_enemies(cur)
            self._deaths += 1
            died = False

        stuck = campaign and self._steps_since_progress >= int(self.cfg.stuck_seconds * self.cfg.fixed_fps / self.cfg.frameskip)
        terminated, truncated, reason = False, False, ""
        stats = cur.get("stats", {})
        if died:
            terminated, reason = True, "death"
        elif cur.get("scene") != self.scene:
            terminated, reason = True, "scene_changed"
        elif self.cfg.mode == "campaign" and stats.get("level_complete"):
            terminated, reason = True, "level_complete"
        elif self.cfg.mode == "cybergrind" and self.cfg.max_wave and (cur.get("cybergrind") or {}).get("wave", 0) > self.cfg.max_wave:
            terminated, reason = True, "max_wave"
        elif stuck:
            truncated, reason = True, "stuck"
        elif self._steps >= self.cfg.max_steps:
            truncated, reason = True, "max_steps"
        self._last_end_reason = reason

        info = self._info(cur)
        info["reward_parts"] = reward.parts
        if reason:
            info["end_reason"] = reason
            info["reset_seconds"] = self._reset_seconds
            seconds = self._steps * self.cfg.frameskip / self.cfg.fixed_fps
            info["episode_seconds"] = seconds
            info["kills_per_min"] = info["kills"] / seconds * 60.0 if seconds > 0 else 0.0
            if campaign:
                self._end_campaign_episode(cur, reason, info)
        return self._pack(cur), float(reward.total), terminated, truncated, info

    def close(self) -> None:
        try:
            self._save_archive()
        finally:
            # Release the game even if the archive could not be written, so it never stays in lockstep.
            if self._connected:
                self.client.close()
                self._connected = False

    # ------------------------------------------------------------------

    def _can_soft_reset(self) -> bool:
        """After a soft death or a timeout, Cyber Grind can continue in place instead of reloading the scene."""
        player = self._raw.get("player") if self._raw else None
        return (
            self.cfg.mode == "cybergrind"
            and self.cfg.soft_death
            and self._last_end_reason in ("death", "max_steps")
            and self._raw.get("scene") == self.scene
            and player is not None
            and not player["dead"]
            and 1 <= (self._raw.get("cybergrind") or {}).get("wave", 0) <= (self.cfg.hard_reset_above_wave or 10 ** 9)
        )

    def _soft_reset(self) -> dict[str, Any]:
        player = self._raw["player"]
        if player.get("soft_death_instakill") and self._arena_spawn is not None:
            # Died in a pit: put the player back above the arena so they don't keep falling.
            x, y, z = self._arena_spawn
            return self.client.teleport([x, y + 10.0, z])
        return self.client.get_obs()

    def _enter_arena(self, raw: dict[str, Any]) -> dict[str, Any]:
        """The Cyber Grind spawn is a ledge above the arena; waves only start once the player enters the grid's trigger.

        Teleports straight into the trigger when the mod reports it (fast, so parallel games aren't held
        up by resets), otherwise walks off the ledge. Times are in game seconds so this doesn't depend on
        frameskip.
        """
        steps_per_second = self.cfg.fixed_fps / self.cfg.frameskip
        trigger = (raw.get("cybergrind") or {}).get("start_trigger")
        if trigger:
            raw = self.client.teleport(trigger["center"])
            for _ in range(int(2 * steps_per_second)):
                raw = self.client.step({})
                if (raw.get("cybergrind") or {}).get("wave", 0) >= 1:
                    return raw
        forward = {"move": [0, 1]}
        for _ in range(int(2.7 * steps_per_second)):
            raw = self.client.step(forward)
        raw = self.client.step({"move": [0, 1], "buttons": ["jump"]})
        for _ in range(int(10 * steps_per_second)):
            if (raw.get("cybergrind") or {}).get("wave", 0) >= 1:
                return raw
            raw = self.client.step(forward)
        raise RuntimeError("Failed to enter the Cyber Grind arena after reset")

    def _note_behaviour(self, raw: dict[str, Any], command: dict[str, Any], raw_pitch_cmd: float) -> None:
        """Per-episode diagnostics: is the agent shooting, is it shooting at anything, and does it turn toward enemies?"""
        self._behaviour["steps"] += 1
        firing = "fire1" in command["buttons"] or "fire2" in command["buttons"]
        self._behaviour["firing"] += firing
        self._behaviour["yaw_sum"] += abs(command["look"][0])
        player = raw.get("player")
        if not player:
            return
        self._behaviour["pitch_steps"] += 1
        self._behaviour["pitch_sum"] += abs(player["pitch"])
        self._behaviour["pitch_signed_sum"] += player["pitch"]
        self._behaviour["look_up_sum"] += player.get("forward", (0.0, 0.0, 0.0))[1]
        visible = [e for e in raw.get("enemies", []) if e["visible"]]
        if not visible:
            return
        self._behaviour["enemy_visible"] += 1
        errors = aim_errors(player, visible[0])
        if errors is None:
            return
        angle, yaw_err, pitch_err = errors
        self._behaviour["angle_sum"] += angle
        self._behaviour["yaw_err_sum"] += yaw_err
        self._behaviour["pitch_err_sum"] += pitch_err
        # Tracking scores: +1 when the look command turns toward the nearest visible enemy, -1 when away
        # (random play scores 0). Uses the unclamped pitch command so the band does not bias it.
        #
        # Two corrections after the 2.07M-step audit, which found the raw score reproduced almost entirely
        # by measurement artefacts rather than by policy behaviour:
        #  - Grade only when the nearest enemy overall is the visible one. `pack_observation` feeds the
        #    policy the nearest 8 by distance REGARDLESS of visibility, so when the nearest is occluded the
        #    old score graded the look command against a farther enemy at an unrelated azimuth. Simulating
        #    that alone reproduced the live 0.121-0.150 at the observed visibility, from a true ~0.24.
        #  - Use an ANGULAR deadzone. `abs(x) > 0.5` was 0.5 metres, i.e. 1.4 degrees off at 20 m but
        #    4.8 degrees at 6 m, so the threshold silently tightened with range.
        # Note these scores are still computed on the SAMPLED action, so with a soft policy they sit well
        # below the mean-action agreement (86% at 2.0M): the gap between them is the exploration noise.
        x, y, z = visible[0]["rel"]
        yaw_cmd = command["look"][0]
        if visible[0] is raw["enemies"][0]:  # gates only the two tracking scores, not the metrics below
            if yaw_cmd and abs(math.degrees(math.atan2(x, z))) > 5.0:
                self._behaviour["yaw_track_n"] += 1
                self._behaviour["yaw_track"] += 1 if (yaw_cmd > 0) == (x > 0) else -1
            if raw_pitch_cmd and abs(math.degrees(math.atan2(y, math.hypot(x, z)))) > 5.0:
                self._behaviour["pitch_track_n"] += 1
                self._behaviour["pitch_track"] += 1 if (raw_pitch_cmd > 0) == (y > 0) else -1
        elev = horizon_elevation(player, visible[0])
        if elev is not None:
            self._behaviour["elev_steps"] += 1
            self._behaviour["elev_sum"] += elev
            self._behaviour["elev_abs_sum"] += abs(elev)
            self._behaviour["elev_over15"] += abs(elev) > 15.0
        self._behaviour["dist_sum"] += visible[0]["dist"]
        self._behaviour["close"] += visible[0]["dist"] <= 5.0
        on_target = angle <= 15.0
        self._behaviour["on_target"] += on_target
        self._behaviour["firing_on_target"] += on_target and firing

    def _track_enemies(self, raw: dict[str, Any]) -> None:
        for e in raw.get("enemies", []):
            if e["health"] > self._enemy_max_health.get(e["id"], 0.0):
                self._enemy_max_health[e["id"]] = e["health"]

    # Campaign ----------------------------------------------------------

    def _archive_path(self) -> Path:
        return Path(self.cfg.explore_dir) / f"explore_{safe_name(self.cfg.level)}_{self.cfg.port}.npz"

    def _save_archive(self) -> None:
        if self.cfg.mode != "campaign" or not self.cfg.explore_dir:
            return
        path = self._archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.archive.save(path)

    @staticmethod
    def _current_checkpoint(raw: dict[str, Any]) -> str | None:
        """Id of the checkpoint a respawn would return to, or None when no checkpoint is active in this level load."""
        for cp in (raw.get("campaign") or {}).get("checkpoints", []):
            if cp.get("current"):
                return cp["id"]
        return None

    @staticmethod
    def _rounded(pos) -> list[float]:
        return [round(float(v), 2) for v in pos]

    def _campaign_reset(self) -> dict[str, Any]:
        """Fresh level load or checkpoint respawn, so each game drifts toward the part of the level it cannot do yet."""
        prev = self._raw or {}
        camp = prev.get("campaign") or {}
        current = self._current_checkpoint(prev)
        if self._last_end_reason == "stuck":
            # A respawn can leave a door locked behind the player; three stuck episodes in a row at the same
            # checkpoint force a fresh load.
            self._stuck_streak = self._stuck_streak + 1 if current == self._stuck_checkpoint else 1
            self._stuck_checkpoint = current
        else:
            self._stuck_streak, self._stuck_checkpoint = 0, None
        fresh = choose_fresh_start(
            self._rng,
            in_level=prev.get("scene") == self.cfg.level and prev.get("player") is not None,
            level_over=bool(camp.get("level_over")),
            has_checkpoint=current is not None,
            stuck_streak=self._stuck_streak,
            stuck_limit=self.cfg.stuck_repeats,
            fresh_prob=self.cfg.fresh_start_prob,
        )
        raw = self._skip_locked(self.client.reset(self.cfg.level, checkpoint=not fresh))
        if fresh:
            self.milestones.new_level_load(raw.get("campaign"))
            self._stuck_streak, self._stuck_checkpoint = 0, None
        else:
            # Whatever the respawn itself changes (doors it unlocks, rooms it resets) pays nothing.
            self.milestones.mark_paid(raw.get("campaign"))
        self._fresh_start = fresh
        return raw

    def _respawn(self) -> dict[str, Any]:
        """Respawns after a death inside the same episode. Milestones the respawn itself changes pay nothing."""
        before = self._raw.get("stats", {})
        raw = self._skip_locked(self.client.reset(self.cfg.level, checkpoint=True))
        self.milestones.mark_paid(raw.get("campaign"))
        self._track_enemies(raw)
        player = raw.get("player")
        if player and self._current_checkpoint(raw) is None:
            # No checkpoint yet, so StatsManager.Restart reloaded the level and its counters started again. Keep
            # the kills and style from before the death in this episode's info, and start a best run's positions
            # again from the spawn: the official time is the reloaded attempt's.
            after = raw.get("stats", {})
            for key in ("kills", "style"):
                self._episode_start_stats[key] = self._episode_start_stats.get(key, 0) - before.get(key, 0) + after.get(key, 0)
            self._positions = [self._rounded(player["pos"])] if self._fresh_start else []
        return raw

    def _skip_locked(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Steps with an empty action while input is locked (landing, cutscenes), so the policy never sees those frames."""
        for _ in range(int(self.cfg.max_locked_skip_s * self.cfg.fixed_fps / self.cfg.frameskip)):
            camp = raw.get("campaign") or {}
            player = raw.get("player")
            if not camp.get("input_locked") or camp.get("level_over") or not player or player["dead"]:
                break
            raw = self.client.step({})
            self._track_enemies(raw)
        return raw

    def _campaign_progress(self, raw: dict[str, Any]) -> CampaignStep:
        """What the level did this step. Any progress restarts the stuck clock."""
        camp = raw.get("campaign") or {}
        checkpoints, arenas, doors = self.milestones.update(raw.get("campaign"))
        novelty = 0.0
        player = raw.get("player")
        if player:
            pos = player["pos"]
            novelty = self.archive.visit(pos)
            if novelty > 0:
                self._cells_new += 1
            if self._fresh_start:
                self._positions.append(self._rounded(pos))
            if camp.get("exit"):
                self._exit_dist_min = min(self._exit_dist_min, math.dist(pos, camp["exit"]["pos"]))
        path_gain = self.path_progress.update(camp.get("path"))
        if checkpoints or arenas or doors or novelty > 0 or path_gain > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        return CampaignStep(checkpoints=checkpoints, arenas=arenas, doors=doors, novelty=novelty, path_gain=path_gain)

    def _level_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Official time, kills, style, restarts and rank as the game's results screen would count them."""
        camp = raw.get("campaign") or {}
        stats = raw.get("stats", {})
        seconds = camp.get("seconds", stats.get("seconds", 0.0))
        restarts = camp.get("restarts", stats.get("restarts", 0))
        kills, style = stats.get("kills", 0), stats.get("style", 0)
        ranks = camp.get("ranks")
        rank = compute_rank(seconds, kills, style, restarts, ranks) if ranks else None
        return {"seconds": seconds, "kills": kills, "style": style, "restarts": restarts, "rank": rank}

    def _end_campaign_episode(self, raw: dict[str, Any], reason: str, info: dict[str, Any]) -> None:
        if reason == "level_complete":
            info["completed"] = 1
            if self._fresh_start:
                # Only a fresh load has a meaningful official time: the timer carries across respawn episodes.
                result = self._level_result(raw)
                info["level_seconds"] = result["seconds"]
                info["restarts"] = result["restarts"]
                info["rank"] = result["rank"]
                self._save_best_run(raw)
        self._episodes += 1
        if self._episodes % 20 == 0:
            self._save_archive()

    def _save_best_run(self, raw: dict[str, Any]) -> None:
        if not self.cfg.best_runs_dir:
            return
        result = self._level_result(raw)
        run = {
            "level": self.cfg.level,
            "seconds": result["seconds"],
            "kills": result["kills"],
            "style": result["style"],
            "restarts": result["restarts"],
            "deaths": self._deaths,
            "rank": result["rank"],
            "difficulty": (raw.get("campaign") or {}).get("difficulty"),
            "positions": self._positions,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = Path(self.cfg.best_runs_dir) / f"{safe_name(self.cfg.level)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_best_run(path, run)

    def _pack(self, raw: dict[str, Any]) -> np.ndarray:
        player = raw.get("player")
        explore = self.archive.features(player["pos"], player["yaw"]) if self.cfg.mode == "campaign" and player else None
        return pack_observation(raw, self.cfg.layout, self._enemy_max_health, explore)

    def _info(self, raw: dict[str, Any]) -> dict[str, Any]:
        stats = raw.get("stats", {})
        player = raw.get("player") or {}
        start = self._episode_start_stats
        info = {
            "kills": stats.get("kills", 0) - start.get("kills", 0),
            "style": stats.get("style", 0) - start.get("style", 0),
            "hp": player.get("hp", 0),
            "wave": (raw.get("cybergrind") or {}).get("wave", 0),
            "deaths": self._deaths,
        }
        b = self._behaviour
        steps = max(1, b["steps"])
        seen = max(1, b["enemy_visible"])
        info["firing_frac"] = b["firing"] / steps
        info["on_target_frac"] = b["on_target"] / steps
        info["firing_on_target_frac"] = b["firing_on_target"] / steps
        info["enemy_visible_frac"] = b["enemy_visible"] / steps
        info["enemy_angle_mean"] = b["angle_sum"] / seen  # degrees off the crosshair
        info["enemy_yaw_angle_mean"] = b["yaw_err_sum"] / seen  # heading error only, ignoring pitch
        info["enemy_pitch_err_mean"] = b["pitch_err_sum"] / seen  # vertical miss: enemy elevation in camera space
        info["yaw_track"] = b["yaw_track"] / max(1, b["yaw_track_n"])  # -1..1, turns toward the enemy horizontally
        info["pitch_track"] = b["pitch_track"] / max(1, b["pitch_track_n"])  # -1..1, vertically
        info["pitch_abs_mean"] = b["pitch_sum"] / max(1, b["pitch_steps"])  # camera pitch away from level
        info["pitch_mean"] = b["pitch_signed_sum"] / max(1, b["pitch_steps"])  # signed rotationX
        info["look_up_mean"] = b["look_up_sum"] / max(1, b["pitch_steps"])  # mean camera forward.y (>0 = looking up)
        elev_steps = max(1, b["elev_steps"])
        info["enemy_elev_mean"] = b["elev_sum"] / elev_steps  # nearest visible enemy, degrees above the horizon
        info["enemy_elev_abs_mean"] = b["elev_abs_sum"] / elev_steps
        info["enemy_elev_over15_frac"] = b["elev_over15"] / elev_steps  # share of steps the pitch band cannot reach
        info["enemy_dist_mean"] = b["dist_sum"] / seen
        info["enemy_close_frac"] = b["close"] / steps  # nearest visible enemy within 5 m
        info["yaw_per_step_mean"] = b["yaw_sum"] / steps  # degrees turned per decision
        if self.cfg.mode == "campaign":
            info["fresh_start"] = int(self._fresh_start)
            info["checkpoints_level"] = self.milestones.checkpoints_reached  # distinct checkpoints this level load
            info["cells_new"] = self._cells_new  # cells entered for the first time this episode
            info["exit_dist_min"] = None if math.isinf(self._exit_dist_min) else self._exit_dist_min
            info["completed"] = 0  # set to 1 on the step that ends with level_complete
            info["level_seconds"] = None  # official time, only for a fresh-start completion
        return info
```

- [ ] **Step 4: Run the new tests to verify they pass**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_env.py
```

Expected:

```text
ok test_campaign_observation_is_479_values
ok test_completion_after_a_checkpoint_respawn_has_no_official_time
ok test_cybergrind_default_stays_448_without_connecting
ok test_death_after_the_checkpoint_respawns_inside_the_episode
ok test_death_before_any_checkpoint_reloads_the_level_inside_the_episode
ok test_exploration_archive_is_saved_on_close_and_loaded_again
ok test_fresh_start_walked_to_the_exit_completes_the_level
ok test_from_dict_ignores_retired_keys
ok test_input_locked_frames_are_skipped
ok test_three_stuck_episodes_at_one_checkpoint_force_a_fresh_load
10 tests passed
```

- [ ] **Step 5: Run every no-game test file**

```powershell
cd F:\Github\ULTRAKILL-AI\python
foreach ($t in Get-ChildItem tests\test_*.py) { .venv\Scripts\python $t.FullName; if ($LASTEXITCODE -ne 0) { throw "$($t.Name) failed" } }
```

Expected: no exception. `test_aim.py` ends with `9 tests passed`, `test_campaign_env.py` with `10 tests passed`, `test_progress.py` with `all tests passed`, and each test file added by Tasks 3-6 (`test_campaign.py`, `test_campaign_rewards.py`, `test_spaces.py`) with its own `N tests passed` line.

- [ ] **Step 6: Check that a saved config with retired keys still loads**

`models/cybergrind_ppo_v2/env_config.yaml` still holds `checkpoint_resets`, `stuck_steps`, `route_dir`, `rewards.route_point` and `rewards.stuck`; `eval.py` loads it through `EnvConfig.from_dict`.

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python -c "import yaml; from ultrakill_ai.env import EnvConfig, UltrakillEnv; c = EnvConfig.from_dict(yaml.safe_load(open('models/cybergrind_ppo_v2/env_config.yaml', encoding='utf-8'))); print(c.mode, UltrakillEnv(c).observation_space.shape)"
```

Expected: `cybergrind (448,)`

- [ ] **Step 7: Review the diff for Cyber Grind regressions**

```powershell
cd F:\Github\ULTRAKILL-AI
git diff -U0 python/ultrakill_ai/env.py | Select-String '^-'
```

Expected: apart from the `--- a/` header and bare `-` lines (removed blank lines), only these removed lines: the `dataclasses`, `rewards` and `routes` import lines; the `checkpoint_resets`, `stuck_steps` and `route_dir` fields; the three old `from_dict` body lines; the `route_tracker` block in `__init__` (`self.route_tracker: RouteTracker | None = None` through `self.route_tracker = RouteTracker(Route.load(path))`); in `reset` the three `checkpoint = (...)` lines, `if self._can_soft_reset():`, `self._raw = self.client.reset(self.scene, checkpoint=checkpoint)` and `if self.cfg.mode == "cybergrind" and self.cfg.auto_enter_arena:`, and the two `route_tracker.start` lines; in `step` `route_gain = 0`, the three `route_tracker` lines after it, the old `stuck = ...` line, the Task 5 `reward = compute_reward(...)` line and `if died and not self.cfg.end_episode_on_death ...` (now `elif`); in `close` the three lines from `if self._connected:` to `self._connected = False` (now indented under `finally:`); the Task 6 `_pack` body; the two `route_progress` lines in `_info`. Any other removed line is a Cyber Grind regression: restore it.

- [ ] **Step 8: Update CLAUDE.md**

Edit 1. In `CLAUDE.md`, replace:

```markdown
  - `env.py`: `UltrakillEnv` / `EnvConfig` (`pitch_limit_deg` keeps the camera near level).
```

with:

```markdown
  - `env.py`: `UltrakillEnv` / `EnvConfig` (`pitch_limit_deg` keeps the camera near level). `EnvConfig.from_dict` ignores keys that are no longer fields, so an old `env_config.yaml` with retired settings still loads. Campaign mode (`mode: campaign`, 479 inputs):
    - Reset: a fresh level load or a respawn at the current checkpoint. Fresh after a completion, with no checkpoint yet in this level load, after `stuck_repeats` (3) stuck episodes in a row at one checkpoint, otherwise with probability `fresh_start_prob` (0.2).
    - Input-locked frames (landing, cutscenes) are stepped through with an empty action and never reach the policy (`max_locked_skip_s`).
    - A death pays `death`, respawns at the checkpoint (or reloads the level when there is none) and the episode continues. Doors, arenas and checkpoints a respawn itself changes pay nothing.
    - Ends: `level_complete` (terminated), `stuck` (`stuck_seconds` without a milestone, a new cell or a shorter exit path) or `max_steps` (truncated).
    - Info: `CAMPAIGN_INFO_KEYS` (`completed`, `fresh_start`, `level_seconds`, `checkpoints_level`, `cells_new`, `exit_dist_min`, plus `kills`, `style`, `deaths`), and `rank` / `restarts` on a fresh-start completion.
    - `difficulty` and `unlock_all_gear` are sent on connect. With `explore_dir` the exploration archive is saved to `explore_<level>_<port>.npz` every 20 episodes and on close; with `best_runs_dir` the fastest fresh-start completion goes to `<level>.json` (positions every step, official time, rank).
```

Edit 2. In `CLAUDE.md`, replace:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
```

with:

```markdown
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_campaign_env.py`: campaign episodes against `FakeLevel`, a fake corridor level standing in for the bridge: completion and best run, no official time for a completion after a checkpoint respawn, respawn and reload after a death, the stuck rule, input-lock skipping, the 479 observation, retired config keys, archive save and load (no game needed).
```

Edit 3. In `CLAUDE.md` (`## Commands`), replace the tests bullet as Task 6 left it:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py`, `python tests/test_campaign_rewards.py` and `python tests/test_spaces.py` (pytest is not installed; the files also work under pytest).
```

with:

```markdown
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py`, `python tests/test_campaign_rewards.py`, `python tests/test_spaces.py` and `python tests/test_campaign_env.py` (pytest is not installed; the files also work under pytest).
```

If the bullet reads differently (an earlier task listed other files), keep every file already listed and add
`python tests/test_campaign_env.py` as the last one.

- [ ] **Step 9: Commit and push**

```powershell
cd F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/env.py python/tests/test_campaign_env.py CLAUDE.md
git commit -m "Run campaign episodes in UltrakillEnv: fresh starts, respawns, milestones, best runs" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the commit lists 3 files and the push reports `main -> main`.

### Task 8: Retire human routes; `protocol.kill`; `random_agent.py` and `bridge_test.py --campaign`

**Files:**
- Delete: `python/ultrakill_ai/routes.py`, `python/scripts/record_route.py`
- Modify: `python/ultrakill_ai/protocol.py` (`BridgeClient.kill`)
- Modify: `python/scripts/random_agent.py` (campaign episodes print completed / checkpoints_level / cells_new)
- Modify: `python/scripts/bridge_test.py` (`--campaign`, read-only)
- Modify: `python/tests/test_campaign_env.py` (two tests)
- Modify: `CLAUDE.md` (layout, bridge check command, status)
- Modify: `README.md` (Usage: the campaign training paragraph no longer records a route)
- Test: `python/tests/test_campaign_env.py`

Task 7 removed the last import of `routes`, so both files can go. `python/configs/campaign_0-1.yaml` still names `record_route.py` in its header comment; Task 13 rewrites that file.

- [ ] **Step 1: Write the failing tests**

Edit 1. In `python/tests/test_campaign_env.py`, replace:

```python
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
```

with:

```python
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.protocol import BridgeClient  # noqa: E402
```

Edit 2. In `python/tests/test_campaign_env.py` (the end of `test_exploration_archive_is_saved_on_close_and_loaded_again`), replace:

```python
        again = UltrakillEnv(env.cfg)
        assert again.archive.counts == counts
```

with:

```python
        again = UltrakillEnv(env.cfg)
        assert again.archive.counts == counts


def test_bridge_client_kill_sends_the_kill_command():
    client = BridgeClient()
    sent = []
    client.request = lambda msg: sent.append(msg) or {"type": "obs", "event": "kill"}
    assert client.kill() == {"type": "obs", "event": "kill"}
    assert sent == [{"type": "kill"}]


def test_human_routes_are_retired():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "ultrakill_ai" / "routes.py").exists()
    assert not (root / "scripts" / "record_route.py").exists()
```

The route test checks the files next to the test rather than calling `importlib.util.find_spec`: the venv holds an editable install of `ultrakill_ai`, and its finder resolves a submodule that is missing from the imported copy of the package to the same file under `F:\Github\ULTRAKILL-AI\python`, so `find_spec` answers for the install target, not for the checkout being tested.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_env.py
```

Expected: a traceback ending in

```text
AttributeError: 'BridgeClient' object has no attribute 'kill'
```

- [ ] **Step 3: Add `BridgeClient.kill`**

In `python/ultrakill_ai/protocol.py`, replace:

```python
    def teleport(self, pos) -> dict[str, Any]:
        return self.request({"type": "teleport", "pos": [float(v) for v in pos]})
```

with:

```python
    def teleport(self, pos) -> dict[str, Any]:
        return self.request({"type": "teleport", "pos": [float(v) for v in pos]})

    def kill(self) -> dict[str, Any]:
        """Kills the player (debug command for the in-game death check). With soft_death on, the mod heals instead."""
        return self.request({"type": "kill"})
```

- [ ] **Step 4: Run the tests to verify the route test still fails**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_env.py
```

Expected: `ok test_bridge_client_kill_sends_the_kill_command` and the other passing tests up to `ok test_from_dict_ignores_retired_keys`, then a traceback in `test_human_routes_are_retired` that points at this line (Python 3.12 underlines it with `^` marks) and ends in a bare `AssertionError`:

```text
    assert not (root / "ultrakill_ai" / "routes.py").exists()
```

- [ ] **Step 5: Delete the route modules**

```powershell
cd F:\Github\ULTRAKILL-AI
git rm python/ultrakill_ai/routes.py python/scripts/record_route.py
```

Expected: `rm 'python/scripts/record_route.py'` and `rm 'python/ultrakill_ai/routes.py'`.

- [ ] **Step 6: Run the tests to verify they pass**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_env.py
```

Expected:

```text
ok test_bridge_client_kill_sends_the_kill_command
ok test_campaign_observation_is_479_values
ok test_completion_after_a_checkpoint_respawn_has_no_official_time
ok test_cybergrind_default_stays_448_without_connecting
ok test_death_after_the_checkpoint_respawns_inside_the_episode
ok test_death_before_any_checkpoint_reloads_the_level_inside_the_episode
ok test_exploration_archive_is_saved_on_close_and_loaded_again
ok test_fresh_start_walked_to_the_exit_completes_the_level
ok test_from_dict_ignores_retired_keys
ok test_human_routes_are_retired
ok test_input_locked_frames_are_skipped
ok test_three_stuck_episodes_at_one_checkpoint_force_a_fresh_load
12 tests passed
```

- [ ] **Step 7: Prove nothing imports the route code**

```powershell
cd F:\Github\ULTRAKILL-AI
git grep -n -E "^\s*(from|import)\s+ultrakill_ai(\.routes|\s+import\s+.*\broutes\b)" -- python
git grep -n -E "record_route|RouteTracker|route_path" -- python/ultrakill_ai python/scripts
```

Expected: no output from either command (exit code 1 each). Before Step 5 the first command listed only `python/scripts/record_route.py:22` (Task 7 already removed the `env.py` import; before Task 7 it also listed `python/ultrakill_ai/env.py:16`).

- [ ] **Step 8: Print campaign progress in `random_agent.py`**

Edit 1. In `python/scripts/random_agent.py`, replace:

```python
"""Smoke test: random actions with automatic resets.

    python scripts/random_agent.py --mode cybergrind --episodes 5
"""
```

with:

```python
"""Smoke test: random actions with automatic resets.

    python scripts/random_agent.py --mode cybergrind --episodes 5
    python scripts/random_agent.py --mode campaign --level "Level 0-1" --episodes 2
"""
```

Edit 2. In `python/scripts/random_agent.py`, replace:

```python
            elapsed = time.perf_counter() - start
            print(
                f"episode {ep}: steps={steps} reward={total:.2f} kills={info['kills']} wave={info['wave']} "
```

with:

```python
            elapsed = time.perf_counter() - start
            if env.cfg.mode == "campaign":
                progress = f"completed={info['completed']} checkpoints_level={info['checkpoints_level']} cells_new={info['cells_new']}"
            else:
                progress = f"wave={info['wave']}"
            print(
                f"episode {ep}: steps={steps} reward={total:.2f} kills={info['kills']} {progress} "
```

- [ ] **Step 9: Add `--campaign` to `bridge_test.py`**

It prints the block from the same `get_obs` the plain check uses and never sends `reset`, `step` or `config`, so it never takes control; `--drive` and `--campaign` are mutually exclusive.

Edit 1. In `python/scripts/bridge_test.py`, replace:

```python
    python scripts/bridge_test.py            # print what the mod sees (you keep control)
    python scripts/bridge_test.py --drive    # also drive the player through a scripted sequence

Start a level or Cyber Grind first, then run this.
```

with:

```python
    python scripts/bridge_test.py              # print what the mod sees (you keep control)
    python scripts/bridge_test.py --campaign   # also print the campaign block (you keep control)
    python scripts/bridge_test.py --drive      # also drive the player through a scripted sequence

Start a level or Cyber Grind first, then run this. The bridge serves one client at a time, so never point
it at a game a trainer is using: connecting drops the trainer.

--campaign only reads (`get_obs`), so the mod's difficulty and gear overrides, which apply only while the AI
has control, are not in effect: `difficulty` shows the game's own setting here.
```

Edit 2. In `python/scripts/bridge_test.py` (insert the campaign summary after `summarize`, before `SEQUENCE`), replace:

```python
SEQUENCE = [
```

with:

```python
def vec(v) -> str:
    return ", ".join(f"{x:.1f}" for x in v)


def summarize_campaign(obs: dict) -> str:
    c = obs.get("campaign")
    if not c:
        return "no campaign block (not in a campaign level, or the mod is older than 0.5.0)"
    lines = [
        f"mission={c['mission']} difficulty={c['difficulty']} seconds={c['seconds']:.2f} "
        f"timer_running={c['timer_running']} level_started={c['level_started']} level_over={c['level_over']} "
        f"restarts={c['restarts']} input_locked={c['input_locked']}"
    ]
    exit_ = c.get("exit")
    lines.append(f"exit: ({vec(exit_['pos'])}) active={exit_['active']}" if exit_ else "exit: null (its room may not be loaded yet)")
    checkpoints = c.get("checkpoints", [])
    lines.append(f"checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        flags = (" activated" if cp["activated"] else "") + (" current" if cp["current"] else "")
        lines.append(f"  {cp['id']} at ({vec(cp['pos'])}){flags}")
    path = c.get("path") or {"status": "none"}
    if path["status"] == "none":
        lines.append("path: none")
    else:
        lines.append(f"path: {path['status']} length={path['length']:.1f}m next_corner=({vec(path['next_corner'])})")
    doors = c.get("locked_doors", [])
    lines.append("locked doors: " + (", ".join(f"({vec(d['pos'])}) {d['dist']:.1f}m" for d in doors) or "none"))
    lines.append(f"arena enemies alive: {c['arena_enemies_alive']}")
    lines.append(f"cleared arenas: {c.get('cleared_arenas') or 'none'}  unlocked doors: {c.get('unlocked_doors') or 'none'}")
    ranks = c.get("ranks") or {}
    lines.append(f"ranks: time={ranks.get('time')} kills={ranks.get('kills')} style={ranks.get('style')}")
    p = obs.get("player")
    if p and "slot_counts" in p:
        lines.append(f"weapons per slot: {p['slot_counts']}")
    return "\n".join(lines)


SEQUENCE = [
```

Edit 3. In `python/scripts/bridge_test.py`, replace:

```python
    parser.add_argument("--drive", action="store_true", help="take control and run a scripted input sequence")
```

with:

```python
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drive", action="store_true", help="take control and run a scripted input sequence")
    mode.add_argument("--campaign", action="store_true", help="also print the campaign block (read-only, never takes control)")
```

Edit 4. In `python/scripts/bridge_test.py`, replace:

```python
        print(summarize(client.get_obs()))
```

with:

```python
        obs = client.get_obs()
        print(summarize(obs))
        if args.campaign:
            print(summarize_campaign(obs))
```

- [ ] **Step 10: Check the two scripts**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python -m py_compile scripts\random_agent.py scripts\bridge_test.py
.venv\Scripts\python scripts\bridge_test.py --help
.venv\Scripts\python scripts\bridge_test.py --drive --campaign
.venv\Scripts\python -c "import sys; sys.path[:0] = ['scripts', 'tests']; import bridge_test, test_campaign_env as t; f = t.FakeLevel(); [f.step({'move': [0, 1]}) for _ in range(16)]; print(bridge_test.summarize_campaign(f.get_obs())); print(bridge_test.summarize_campaign({'scene': 'Endless'}))"
```

Expected: `py_compile` prints nothing. The help shows `usage: bridge_test.py [-h] [--port PORT] [--drive | --campaign] [--realtime]` and `--campaign   also print the campaign block (read-only, never takes control)`. The combined flags exit with `bridge_test.py: error: argument --campaign: not allowed with argument --drive`. The summary of the fake level prints:

```text
mission=1 difficulty=3 seconds=2.13 timer_running=True level_started=True level_over=False restarts=0 input_locked=False
exit: (0.0, 1.0, 60.0) active=True
checkpoints: 1
  0,1,20 at (0.0, 1.0, 20.0) activated current
path: complete length=28.0m next_corner=(0.0, 1.0, 60.0)
locked doors: none
arena enemies alive: 0
cleared arenas: ['0,1,30']  unlocked doors: none
ranks: time=[120, 90, 60, 30] kills=[0, 1, 2, 3] style=[0, 100, 200, 300]
weapons per slot: [1, 0, 0, 0, 0]
no campaign block (not in a campaign level, or the mod is older than 0.5.0)
```

- [ ] **Step 11: Run every no-game test file**

```powershell
cd F:\Github\ULTRAKILL-AI\python
foreach ($t in Get-ChildItem tests\test_*.py) { .venv\Scripts\python $t.FullName; if ($LASTEXITCODE -ne 0) { throw "$($t.Name) failed" } }
```

Expected: no exception. `test_aim.py` ends with `9 tests passed`, `test_campaign_env.py` with `12 tests passed`, `test_progress.py` with `all tests passed`, and the files Tasks 3-6 added (`test_campaign.py`, `test_campaign_rewards.py`, `test_spaces.py`) each with its own `N tests passed` line.

- [ ] **Step 12: Update CLAUDE.md**

Edit 1. In `CLAUDE.md`, replace:

```markdown
  - `protocol.py`: socket client.
```

with:

```markdown
  - `protocol.py`: socket client; `kill()` kills the player (debug command for the in-game death check).
```

Edit 2. In `CLAUDE.md`, delete this whole line (Task 3 put the `campaign.py` entry on the line right after it; that entry stays as it is):

```markdown
  - `routes.py`: campaign route tracking.
```

Edit 3. In `CLAUDE.md`, replace:

```markdown
- `python/scripts/`: `bridge_test.py`, `random_agent.py`, `record_route.py`, `train.py` (PPO / RecurrentPPO, `--num-envs` uses SubprocVecEnv), `eval.py`, `games.py` (launch/tile/status/stop training instances), `dashboard.py` (Tkinter live view of `status.json`).
```

with:

```markdown
- `python/scripts/`: `bridge_test.py` (`--drive`, `--campaign`), `random_agent.py` (`--mode campaign` prints completed / checkpoints_level / cells_new per episode), `train.py` (PPO / RecurrentPPO, `--num-envs` uses SubprocVecEnv), `eval.py`, `games.py` (launch/tile/status/stop training instances), `dashboard.py` (Tkinter live view of `status.json`).
```

Edit 4. In `CLAUDE.md`, replace:

```markdown
- Bridge check (game open, in a level): `python scripts/bridge_test.py --drive`.
```

with:

```markdown
- Bridge check (game open, in a level): `python scripts/bridge_test.py --drive`. In a campaign level, `python scripts/bridge_test.py --campaign` prints the `campaign` block (exit, checkpoints, path, locked doors, arena state, ranks, weapons per slot) without taking control, so its `difficulty` is the game's own setting: the override applies only while the AI has control. Never run it against a game a trainer is using.
```

Edit 5. In `CLAUDE.md`, replace:

```markdown
- **Python:** env, training, eval and route tracking verified against a mock and partly in game.
```

with:

```markdown
- **Python:** env, training and eval verified against a mock and partly in game; campaign episodes verified against a fake level (`tests/test_campaign_env.py`).
```

Edit 6. In `CLAUDE.md`, replace:

```markdown
    - No human demos or recorded routes, so `routes.py` and `record_route.py` are to be retired.
```

with:

```markdown
    - No human demos or recorded routes: `routes.py` and `record_route.py` were deleted 2026-09-16 (git history keeps them).
```

- [ ] **Step 13: Update README.md**

The Usage section still tells the reader to record a route with the script Step 5 deleted. In `README.md` (`## Usage`), replace:

````markdown
**Train on a campaign level.** Record a route by playing the level yourself first.
```bash
python scripts/record_route.py --level "Level 0-1"
python scripts/train.py --config configs/campaign_0-1.yaml
```
````

with:

````markdown
**Train on a campaign level.** No recorded route is needed: the agent learns from the level itself (checkpoints, arena clears, door unlocks, ground it has not covered yet, and the NavMesh distance to the exit). To see what the mod reports in a campaign level without taking control, run `python scripts/bridge_test.py --campaign` while playing it.
```bash
python scripts/train.py --config configs/campaign_0-1.yaml
```
````

Then check that no documentation, config or code still tells anyone to run the deleted script:

```powershell
cd F:\Github\ULTRAKILL-AI
git grep -n "record_route" -- README.md CLAUDE.md docs/protocol.md docs/game-internals.md python/configs python/scripts python/ultrakill_ai
```

Expected: exactly two lines, the `CLAUDE.md` status line that records the deletion (Step 12, Edit 6) and the old config's header comment, which Task 13 rewrites. The spec is not searched (its "Retired" section names the script on purpose), nor `python/tests` (`test_human_routes_are_retired` names the file it checks is gone).

```text
CLAUDE.md:<line>:    - No human demos or recorded routes: `routes.py` and `record_route.py` were deleted 2026-09-16 (git history keeps them).
python/configs/campaign_0-1.yaml:2:#   python scripts/record_route.py --level "Level 0-1"
```

- [ ] **Step 14: Commit and push**

```powershell
cd F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/protocol.py python/scripts/random_agent.py python/scripts/bridge_test.py python/tests/test_campaign_env.py CLAUDE.md README.md
git diff --cached --name-status
git commit -m "Retire human routes, add BridgeClient.kill and bridge_test --campaign" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected `git diff --cached --name-status` before the commit:

```text
M	CLAUDE.md
M	README.md
M	python/scripts/bridge_test.py
M	python/scripts/random_agent.py
D	python/scripts/record_route.py
M	python/tests/test_campaign_env.py
M	python/ultrakill_ai/protocol.py
D	python/ultrakill_ai/routes.py
```

The push reports `main -> main`.

### Task 9: Progress, `poll_status.py` and dashboard campaign panel

**Files:**
- Modify: `python/ultrakill_ai/progress.py` (record the campaign episode fields and drop `route_progress`; `FRESH_WINDOW`; the `campaign` status block; `best_checkpoints_level`; new history and per-game keys)
- Modify: `python/scripts/poll_status.py` (campaign columns; keep an existing CSV header with `extrasaction="ignore"`)
- Modify: `python/scripts/dashboard.py` (Campaign panel with completion, times and the largest reward parts; fresh completion chart; checkpoints column in the games table)
- Modify: `CLAUDE.md` (layout, commands, status)
- Modify: `README.md` (Usage: the "Watch training" paragraph)
- Test: `python/tests/test_progress.py`

Depends on Task 7 (`CAMPAIGN_INFO_KEYS` in `ultrakill_ai/env.py`). All commands run from `python/` in PowerShell, except the commit step, which changes to the repo root.

- [ ] **Step 1: Test helpers: a fake campaign env, a dashboard loader, and a more general `_train`**

In `python/tests/test_progress.py`, replace the module docstring's first line:

```python
"""Tests for ultrakill_ai.progress.ProgressCallback, without the game.
```

with:

```python
"""Tests for ultrakill_ai.progress.ProgressCallback, the dashboard and poll_status.py, without the game.
```

Replace the imports:

```python
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
```

with:

```python
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
```

Replace:

```python
from ultrakill_ai import progress as progress_mod  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402
```

with:

```python
from ultrakill_ai import progress as progress_mod  # noqa: E402
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS  # noqa: E402
from ultrakill_ai.progress import ProgressCallback  # noqa: E402
```

Replace:

```python
def _make(seed):
    return lambda: Monitor(FakeGrindEnv(seed), info_keywords=("kills", "wave", "style"))


def _train(status_path: Path, timesteps: int, model=None, resume=False):
    venv = DummyVecEnv([_make(1), _make(2)])
```

with:

```python
class FakeCampaignEnv(gym.Env):
    """Campaign-style infos: every other episode is a fresh start, and about half the episodes finish the level."""

    def __init__(self, seed: int, log: list[dict]):
        self.observation_space = ObsLayout().space()
        self.action_space = action_space()
        self.rng = np.random.default_rng(seed)
        self.log = log  # final info of every finished episode, in the order the callback records them
        self.episode = 0
        self.steps = 0
        self.fresh = 1

    def _obs(self):
        return self.rng.uniform(-1, 1, self.observation_space.shape).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.fresh = 1 if self.episode % 2 == 0 else 0
        self.episode += 1
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        terminated = bool(self.rng.random() < 0.1)
        completed = int(terminated and self.rng.random() < 0.5)
        info = {
            "kills": 0,
            "style": 0,
            "wave": 0,  # the real env keeps this Cyber Grind key in campaign mode
            "deaths": self.steps // 8,
            "completed": completed,
            "fresh_start": self.fresh,
            # The env reports the official time only for fresh-start completions.
            "level_seconds": round(self.steps * 2 / 15, 3) if completed and self.fresh else None,
            "checkpoints_level": min(3, self.steps // 4),
            "cells_new": self.steps * 2,
            "exit_dist_min": max(0.0, 60.0 - self.steps),
            "reward_parts": {"time": -0.01, "novelty": 0.5},
        }
        if terminated:
            info["end_reason"] = "level_complete" if completed else "stuck"
            self.log.append(info)
        return self._obs(), 0.49, terminated, False, info


def _make(seed):
    return lambda: Monitor(FakeGrindEnv(seed), info_keywords=("kills", "wave", "style"))


def _make_campaign(seed, log):
    return lambda: Monitor(FakeCampaignEnv(seed, log), info_keywords=CAMPAIGN_INFO_KEYS)


def _load_dashboard():
    """Imports scripts/dashboard.py as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location("dashboard", ROOT / "scripts" / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(status_path: Path, timesteps: int, model=None, resume=False, env_fns=None, run_name="test_run"):
    venv = DummyVecEnv(env_fns or [_make(1), _make(2)])
```

Replace (inside `_train`):

```python
    cb = ProgressCallback(status_path, target, "test_run", 2, update_every_s=0.2)
```

with:

```python
    cb = ProgressCallback(status_path, target, run_name, 2, update_every_s=0.2)
```

- [ ] **Step 2: The Cyber Grind test asserts the retired key is gone and there is no campaign block**

In `test_progress_callback_writes_status`, replace:

```python
        assert m["route_progress"] is None
```

with:

```python
        assert "route_progress" not in m, "route_progress is retired"
        assert m["completed"] is None and m["checkpoints_level"] is None
        assert "campaign" not in s, "a Cyber Grind run has no campaign block"
        assert s["best_checkpoints_level"] is None
```

and replace:

```python
        assert all(p["timesteps"] <= s["timesteps"] for p in s["history"])
```

with:

```python
        assert all(p["timesteps"] <= s["timesteps"] for p in s["history"])
        assert all(p["completion_rate_fresh_50"] is None for p in s["history"])
```

- [ ] **Step 3: Add the campaign, dashboard-panel and poll_status tests**

Replace the runner's first line:

```python
if __name__ == "__main__":
```

with the three new tests followed by that same line (the rest of the runner stays as it is):

```python
def test_campaign_progress():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "runs" / "campaign_run" / "status.json"
        log: list[dict] = []
        old_every = progress_mod.HISTORY_EVERY_S
        progress_mod.HISTORY_EVERY_S = 0.5
        try:
            _train(status_path, 3000, env_fns=[_make_campaign(1, log), _make_campaign(2, log)], run_name="campaign_run")
        finally:
            progress_mod.HISTORY_EVERY_S = old_every

        s = json.loads(status_path.read_text(encoding="utf-8"))
        assert "campaign" in s, "expected a campaign block once episodes report fresh_start"
        assert s["episodes"] == len(log)
        fresh = [ep for ep in log if ep["fresh_start"]]
        window = fresh[-progress_mod.FRESH_WINDOW:]
        fresh_times = [ep["level_seconds"] for ep in fresh if ep["completed"]]
        assert len(fresh) > progress_mod.FRESH_WINDOW and fresh_times, "the fake run is too short"

        c = s["campaign"]
        assert c["fresh_window"] == progress_mod.FRESH_WINDOW
        assert 0.0 <= c["fresh_completion_rate"] <= 1.0
        assert abs(c["fresh_completion_rate"] - sum(ep["completed"] for ep in window) / len(window)) < 1e-9
        assert c["best_time"] == min(fresh_times)
        assert c["median_time_50"] == statistics.median(ep["level_seconds"] for ep in window if ep["completed"])
        assert s["best_checkpoints_level"] == max(ep["checkpoints_level"] for ep in log)

        m = s["mean_100"]
        assert "route_progress" not in m
        for key in ("completed", "fresh_start", "level_seconds", "checkpoints_level", "cells_new", "exit_dist_min"):
            assert m[key] is not None, key
        assert set(s["end_reasons_100"]) <= {"level_complete", "stuck"}
        assert set(s["reward_parts_mean_100"]) == {"time", "novelty"}
        for e in s["envs"]:
            assert e["episodes"] > 0 and e["checkpoints_level"] is not None
        assert s["history"], "expected chart history points"
        assert all("completion_rate_fresh_50" in p and "mean_checkpoints_level_100" in p for p in s["history"])
        rates = [p["completion_rate_fresh_50"] for p in s["history"] if p["completion_rate_fresh_50"] is not None]
        assert rates and all(0.0 <= r <= 1.0 for r in rates)

        # A restart keeps the best time and the best checkpoint count; the 50-episode window starts empty.
        restored = ProgressCallback(status_path, 10, "campaign_run", 2)
        restored.mark_stopped()
        r = json.loads(status_path.read_text(encoding="utf-8"))
        assert r["campaign"]["best_time"] == c["best_time"]
        assert r["campaign"]["fresh_window"] == 0 and r["campaign"]["fresh_completion_rate"] is None
        assert r["campaign"]["median_time_50"] is None
        assert r["best_checkpoints_level"] == s["best_checkpoints_level"]


def test_dashboard_campaign_panel():
    dashboard = _load_dashboard()
    lines = dashboard.campaign_lines(
        {"fresh_window": 50, "fresh_completion_rate": 0.62, "median_time_50": 59.9996, "best_time": 83.25},
        {"completed": 0.4, "checkpoints_level": 2.14, "cells_new": 84.4, "deaths": 1.3, "exit_dist_min": 12.2},
        {"time": -9.0, "checkpoint": 20.0, "novelty": 5.04, "path": 0.8, "level_complete": 50.0, "death": None},
    )
    assert lines == [
        "  fresh completed  62% of last 50",
        "  all completed    40%",
        "  best time        01:23.250",
        "  median time      01:00.000",  # whole milliseconds, never "00:60.000"
        "  checkpoints/load 2.1",
        "  new cells/ep     84",
        "  deaths/ep        1.30",
        "  closest to exit  12m",
        "  reward parts/ep",
        "    level_complete +50.0",
        "    checkpoint     +20.0",
        "    time           -9.0",
        "    novelty        +5.0",
    ], lines
    empty = dashboard.campaign_lines({"fresh_window": 0, "fresh_completion_rate": None, "median_time_50": None, "best_time": None}, {})
    assert empty[0] == "  fresh completed  — of last 0", empty
    assert [line.split()[-1] for line in empty[1:]] == ["—"] * 7, empty

    # The whole window on a real campaign status (charts, games table, Campaign panel).
    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        old_every = progress_mod.HISTORY_EVERY_S
        progress_mod.HISTORY_EVERY_S = 0.2
        try:
            _train(status_path, 600, env_fns=[_make_campaign(1, []), _make_campaign(2, [])], run_name="campaign_run")
        finally:
            progress_mod.HISTORY_EVERY_S = old_every
        assert "campaign" in json.loads(status_path.read_text(encoding="utf-8"))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--file", str(status_path), "--smoke-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_poll_status_keeps_old_header():
    status = {
        "state": "running", "timesteps": 123456, "episodes": 300, "window": 100, "steps_per_s": 190.0,
        "mean_100": {"reward": 12.5, "kills": 3.0, "deaths": 1.2, "completed": 0.4, "fresh_start": 0.2,
                     "checkpoints_level": 1.5, "cells_new": 60.0, "exit_dist_min": 20.0},
        "campaign": {"fresh_window": 50, "fresh_completion_rate": 0.4, "median_time_50": 95.0, "best_time": 83.25},
        "best_checkpoints_level": 3,
        "ppo": {"entropy_loss": -8.0},
        "reward_parts_mean_100": {"time": -9.0, "checkpoint": 20.0, "novelty": 5.0},
    }
    # The header an older poll_status.py wrote: none of the campaign columns.
    old_header = ["wall_time", "timesteps", "episodes", "window", "steps_per_s", "state",
                  "reward", "kills", "ppo_entropy_loss", "part_kill", "part_total", "aim_share"]
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        for run in ("old_run", "new_run"):
            (runs / run).mkdir(parents=True)
            (runs / run / "status.json").write_text(json.dumps(status), encoding="utf-8")
        with (runs / "old_run" / "metrics_log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(old_header)
            w.writerow(["2026-09-15 23:00:00", 1000, 10, 10, 150.0, "running", 1.0, 0.5, -9.0, 0.5, 1.0, 0.0])

        for run in ("old_run", "new_run"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "poll_status.py"), "--run", run, "--runs-dir", str(runs), "--once"],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr

        with (runs / "old_run" / "metrics_log.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == old_header, rows[0]
        assert len(rows) == 3, rows
        assert len(rows[2]) == len(old_header), (len(rows[2]), len(old_header))
        appended = dict(zip(old_header, rows[2]))
        assert appended["timesteps"] == "123456" and appended["reward"] == "12.5" and appended["part_total"] == "16.0", appended

        # A new log gets every column, campaign ones included.
        with (runs / "new_run" / "metrics_log.csv").open(newline="", encoding="utf-8") as f:
            new_rows = list(csv.DictReader(f))
        assert len(new_rows) == 1, new_rows
        logged = new_rows[0]
        assert logged["fresh_window"] == "50" and logged["fresh_completion_rate"] == "0.4", logged
        assert logged["median_time_50"] == "95.0" and logged["best_time"] == "83.25", logged
        assert logged["best_checkpoints_level"] == "3" and logged["checkpoints_level"] == "1.5", logged
        assert logged["part_time"] == "-9.0" and logged["part_novelty"] == "5.0" and logged["part_level_complete"] == "", logged


if __name__ == "__main__":
```

- [ ] **Step 4: Run the four changed tests and watch each fail**

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_progress_callback_writes_status()"
```

Expected: `AssertionError: route_progress is retired`.

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_campaign_progress()"
```

Expected: `AssertionError: expected a campaign block once episodes report fresh_start`.

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_dashboard_campaign_panel()"
```

Expected: `AttributeError: module 'dashboard' has no attribute 'campaign_lines'`.

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_poll_status_keeps_old_header()"
```

Expected: `AssertionError: (50, 12)` (the old script writes all 50 of its columns under the 12-column header).

- [ ] **Step 5: `progress.py`: record the campaign fields and build the `campaign` block**

In `python/ultrakill_ai/progress.py`, replace the end of the module docstring:

```python
`ProgressCallback` collects episode stats during `model.learn()` and writes a small JSON status file
(atomically, at most every few seconds). The dashboard only reads that file, so it never touches the
training process.
"""
```

with:

```python
`ProgressCallback` collects episode stats during `model.learn()` and writes a small JSON status file
(atomically, at most every few seconds). The dashboard only reads that file, so it never touches the
training process.

Campaign runs also get a `campaign` block. Its completion rate and median time count fresh starts only:
a checkpoint respawn starts partway through the level, and the official timer carries over from earlier
episodes, so neither says how the agent does on a whole level.
"""
```

Replace:

```python
import json
import math
import os
import time
```

with:

```python
import json
import math
import os
import statistics
import time
```

Replace:

```python
EPISODE_WINDOW = 100
```

with:

```python
EPISODE_WINDOW = 100
FRESH_WINDOW = 50  # campaign completion rate and median time are over this many fresh-start episodes
```

Replace (in `__init__`):

```python
        self.best_wave: float | None = None
        self.per_env: dict[int, dict] = {}
```

with:

```python
        self.best_wave: float | None = None
        self.best_checkpoints_level: float | None = None
        self.best_time: float | None = None  # fastest fresh-start completion, official level seconds
        self.fresh_recent: deque[tuple[float, float | None]] = deque(maxlen=FRESH_WINDOW)  # (completed, level_seconds)
        self.campaign = False  # set once an episode reports fresh_start
        self.per_env: dict[int, dict] = {}
```

Replace (in `_restore`):

```python
        self.best_wave = _num(old.get("best_wave"))
        self.previous_elapsed_s = _num(old.get("elapsed_s")) or 0.0
```

with:

```python
        self.best_wave = _num(old.get("best_wave"))
        self.best_checkpoints_level = _num(old.get("best_checkpoints_level"))
        if isinstance(old.get("campaign"), dict):
            self.campaign = True
            self.best_time = _num(old["campaign"].get("best_time"))
        self.previous_elapsed_s = _num(old.get("elapsed_s")) or 0.0
```

Replace (in `_record_episode`'s `stats`):

```python
            "route_progress": field("route_progress"),
```

with:

```python
            "completed": field("completed"),
            "fresh_start": field("fresh_start"),
            "level_seconds": field("level_seconds"),
            "checkpoints_level": field("checkpoints_level"),
            "cells_new": field("cells_new"),
            "exit_dist_min": field("exit_dist_min"),
```

Replace:

```python
        if stats["wave"] is not None and (self.best_wave is None or stats["wave"] > self.best_wave):
            self.best_wave = stats["wave"]
```

with:

```python
        if stats["wave"] is not None and (self.best_wave is None or stats["wave"] > self.best_wave):
            self.best_wave = stats["wave"]
        if stats["checkpoints_level"] is not None and (self.best_checkpoints_level is None or stats["checkpoints_level"] > self.best_checkpoints_level):
            self.best_checkpoints_level = stats["checkpoints_level"]
        if stats["fresh_start"] is not None:
            self.campaign = True
            if stats["fresh_start"]:
                completed, seconds = stats["completed"] or 0.0, stats["level_seconds"]
                self.fresh_recent.append((completed, seconds))
                if completed and seconds is not None and (self.best_time is None or seconds < self.best_time):
                    self.best_time = seconds
```

Replace (in `self.per_env[env_index] = {...}`):

```python
            "wave": stats["wave"],
            "end_reason": stats["end_reason"],
```

with:

```python
            "wave": stats["wave"],
            "checkpoints_level": stats["checkpoints_level"],
            "end_reason": stats["end_reason"],
```

Replace:

```python
    def _recent_mean(self, key: str) -> float | None:
        return _mean(ep[key] for ep in self.episodes_recent)
```

with:

```python
    def _recent_mean(self, key: str) -> float | None:
        return _mean(ep[key] for ep in self.episodes_recent)

    def _campaign_stats(self) -> dict:
        n = len(self.fresh_recent)
        times = [seconds for completed, seconds in self.fresh_recent if completed and seconds is not None]
        return {
            "fresh_window": n,
            "fresh_completion_rate": sum(completed for completed, _ in self.fresh_recent) / n if n else None,
            "median_time_50": statistics.median(times) if times else None,
            "best_time": self.best_time,
        }
```

Replace (first line of the `recent` comprehension in `_snapshot`):

```python
        recent = {key: self._recent_mean(key) for key in ("reward", "length", "kills", "kills_per_min", "deaths", "wave", "style", "route_progress", "reset_seconds",
```

with:

```python
        recent = {key: self._recent_mean(key) for key in ("reward", "length", "kills", "kills_per_min", "deaths", "wave", "style", "reset_seconds",
                                                 "completed", "fresh_start", "level_seconds", "checkpoints_level", "cells_new", "exit_dist_min",
```

Replace:

```python
        end_reasons = Counter(ep["end_reason"] for ep in self.episodes_recent if ep["end_reason"])
```

with:

```python
        end_reasons = Counter(ep["end_reason"] for ep in self.episodes_recent if ep["end_reason"])
        campaign = self._campaign_stats() if self.campaign else None
```

Replace (in the history point):

```python
                "mean_wave_100": recent["wave"],
                "steps_per_s": steps_per_s,
```

with:

```python
                "mean_wave_100": recent["wave"],
                "completion_rate_fresh_50": campaign["fresh_completion_rate"] if campaign else None,
                "mean_checkpoints_level_100": recent["checkpoints_level"],
                "steps_per_s": steps_per_s,
```

Replace:

```python
        return {
            "version": 1,
```

with:

```python
        status = {
            "version": 1,
```

Replace (the end of that dict):

```python
            "best_wave": self.best_wave,
            "end_reasons_100": dict(end_reasons.most_common()),
            "envs": envs,
            "ppo": self.ppo_metrics,
            "history": self.history,
        }
```

with:

```python
            "best_wave": self.best_wave,
            "best_checkpoints_level": self.best_checkpoints_level,
            "end_reasons_100": dict(end_reasons.most_common()),
            "envs": envs,
            "ppo": self.ppo_metrics,
            "history": self.history,
        }
        if campaign is not None:
            status["campaign"] = campaign
        return status
```

- [ ] **Step 6: Run the progress tests and watch them pass**

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_progress_callback_writes_status(); t.test_campaign_progress(); print('progress ok')"
```

Expected: `progress ok` (about 10 s).

- [ ] **Step 7: `poll_status.py`: campaign columns, and an existing header is kept**

In `python/scripts/poll_status.py`, replace:

```python
Read-only: it polls a file and never touches the bridge ports, so it is safe to leave running
alongside training.
```

with:

```python
Read-only: it polls a file and never touches the bridge ports, so it is safe to leave running
alongside training.

An existing `metrics_log.csv` keeps its header. Rows are written under the columns it already has, and
values for columns it lacks are dropped, so adding a column here never shifts an old log out of
alignment. Move the old file aside to start logging the new columns.
```

Replace:

```python
    "enemy_dist_mean", "enemy_close_frac", "reset_seconds",
]
PPO_FIELDS = ["entropy_loss", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "learning_rate"]
PART_FIELDS = ["aim_yaw", "aim_pitch", "aim_locked", "aim", "kill", "damage_dealt", "damage_taken", "death", "wave", "style", "step"]
```

with:

```python
    "enemy_dist_mean", "enemy_close_frac", "reset_seconds",
    "completed", "fresh_start", "checkpoints_level", "cells_new", "exit_dist_min",
]
# Campaign runs only (status["campaign"]): completion rate and median time over the last 50 fresh starts.
CAMPAIGN_FIELDS = ["fresh_window", "fresh_completion_rate", "median_time_50", "best_time"]
PPO_FIELDS = ["entropy_loss", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "learning_rate"]
PART_FIELDS = ["aim_yaw", "aim_pitch", "aim_locked", "aim", "kill", "damage_dealt", "damage_taken", "death", "wave", "style", "step",
               "time", "checkpoint", "arena_clear", "door_unlock", "novelty", "path", "level_complete"]
```

Replace:

```python
    parts = status.get("reward_parts_mean_100") or {}
    out = {
```

with:

```python
    parts = status.get("reward_parts_mean_100") or {}
    campaign = status.get("campaign") or {}
    out = {
```

Replace:

```python
    out.update({k: m.get(k) for k in FIELDS})
```

with:

```python
    out.update({k: m.get(k) for k in FIELDS})
    out.update({k: campaign.get(k) for k in CAMPAIGN_FIELDS})
    out["best_checkpoints_level"] = status.get("best_checkpoints_level")
```

Replace:

```python
    return out


def main() -> None:
```

with:

```python
    return out


def existing_header(path: Path) -> list[str] | None:
    """The header row of an existing CSV, or None when there is no file or it is empty."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None) or None
    except OSError:
        return None


def main() -> None:
```

Replace:

```python
    header = list(row({}).keys())
    new = not out_path.exists()
```

with:

```python
    header = existing_header(out_path)
    new = header is None
    if new:
        header = list(row({}).keys())
```

Replace:

```python
                w = csv.DictWriter(f, fieldnames=header)
```

with:

```python
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
```

- [ ] **Step 8: Run the poll_status test and watch it pass**

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_poll_status_keeps_old_header(); print('poll ok')"
```

Expected: `poll ok`.

- [ ] **Step 9: `dashboard.py`: Campaign panel, completion chart and checkpoints column**

In `python/scripts/dashboard.py`, replace the end of the Formatting section:

```python
    d, h = divmod(h, 24)
    return f"{d}d {h}h"
```

with:

```python
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def fmt_time(seconds) -> str:
    """Official level time as mm:ss.mmm, the format times.md uses."""
    s = num(seconds)
    if s is None:
        return "—"
    try:
        from ultrakill_ai.times import format_time
    except ImportError:  # a checkout from before times.py: the same whole-millisecond format
        ms = round(max(0.0, s) * 1000)
        return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"
    return format_time(s)


def fmt_pct(value) -> str:
    v = num(value)
    return "—" if v is None else f"{v * 100:.0f}%"


def campaign_lines(campaign: dict, mean: dict, parts: dict | None = None) -> list[str]:
    """Campaign panel rows: completion and official times, per-episode means over the last 100, then the largest reward parts."""
    exit_dist = fmt_float(mean.get("exit_dist_min"), 0)
    rows = (
        ("fresh completed ", f"{fmt_pct(campaign.get('fresh_completion_rate'))} of last {fmt_int(campaign.get('fresh_window'))}"),
        ("all completed   ", fmt_pct(mean.get("completed"))),  # fresh starts and checkpoint respawns alike
        ("best time       ", fmt_time(campaign.get("best_time"))),
        ("median time     ", fmt_time(campaign.get("median_time_50"))),
        ("checkpoints/load", fmt_float(mean.get("checkpoints_level"), 1)),
        ("new cells/ep    ", fmt_float(mean.get("cells_new"), 0)),
        ("deaths/ep       ", fmt_float(mean.get("deaths"))),
        ("closest to exit ", exit_dist if exit_dist == "—" else f"{exit_dist}m"),
    )
    lines = [f"  {label} {value}" for label, value in rows]
    # The four largest reward parts by size, so a term that dominates the return shows up at a glance.
    values = [(str(name), v) for name, v in ((name, num(value)) for name, value in (parts or {}).items()) if v is not None]
    if values:
        lines.append("  reward parts/ep")
        lines.extend(f"    {name[:14]:<14} {v:+.1f}" for name, v in sorted(values, key=lambda nv: -abs(nv[1]))[:4])
    return lines
```

Replace (in `Dashboard.__init__`):

```python
        for c, head in enumerate(("#", "last reward", "kills", "wave", "last episode")):
            self.games_table.columnconfigure(c, weight=1 if c else 0)
            tk.Label(self.games_table, text=head, bg=PANEL, fg=MUTED, font=FONT_SMALL, anchor="w").grid(row=0, column=c, sticky="w", padx=(0, 8))
```

with:

```python
        self.games_heads: list[tk.Label] = []
        for c, head in enumerate(("#", "last reward", "kills", "wave", "last episode")):
            self.games_table.columnconfigure(c, weight=1 if c else 0)
            head_label = tk.Label(self.games_table, text=head, bg=PANEL, fg=MUTED, font=FONT_SMALL, anchor="w")
            head_label.grid(row=0, column=c, sticky="w", padx=(0, 8))
            self.games_heads.append(head_label)
```

Replace (at the top of `render`):

```python
        get = d.get
        mean = get("mean_100") if isinstance(get("mean_100"), dict) else {}
```

with:

```python
        get = d.get
        mean = get("mean_100") if isinstance(get("mean_100"), dict) else {}
        campaign = get("campaign") if isinstance(get("campaign"), dict) else None
```

Replace:

```python
        self.chart_kw.set_series([("kills/min", RED, series("mean_kills_per_min_100")), ("wave", YELLOW, series("mean_wave_100"))])
```

with:

```python
        if campaign is not None:
            self.chart_kw.title = "Fresh completion % & checkpoints"
            fresh_pct = [(x, y * 100.0) for x, y in series("completion_rate_fresh_50")]
            self.chart_kw.set_series([("fresh %", PURPLE, fresh_pct), ("checkpoints", YELLOW, series("mean_checkpoints_level_100"))])
        else:
            self.chart_kw.title = "Kills/min & wave (100 ep)"
            self.chart_kw.set_series([("kills/min", RED, series("mean_kills_per_min_100")), ("wave", YELLOW, series("mean_wave_100"))])
```

Replace:

```python
        self._render_games(get("envs") or [], age if state == "running" else None)
```

with:

```python
        self.games_heads[3].config(text="checkpoints" if campaign is not None else "wave")
        self._render_games(get("envs") or [], age if state == "running" else None)
```

Replace:

```python
        self.behaviour.config(text="Shooting (last 100)\n" + "\n".join(shooting) if shooting else "")
```

with:

```python
        if campaign is not None:
            parts = get("reward_parts_mean_100") if isinstance(get("reward_parts_mean_100"), dict) else {}
            self.behaviour.config(text="Campaign (last 100)\n" + "\n".join(campaign_lines(campaign, mean, parts)))
        else:
            self.behaviour.config(text="Shooting (last 100)\n" + "\n".join(shooting) if shooting else "")
```

Replace (in `_render_games`):

```python
            values = (fmt_int(e.get("env")), fmt_float(e.get("reward")), fmt_int(e.get("kills")), fmt_int(e.get("wave")),
```

with:

```python
            # Campaign games report the checkpoints of the level load instead of a wave (their wave stays 0).
            progress = e.get("checkpoints_level") if e.get("checkpoints_level") is not None else e.get("wave")
            values = (fmt_int(e.get("env")), fmt_float(e.get("reward")), fmt_int(e.get("kills")), fmt_int(progress),
```

- [ ] **Step 10: Run the dashboard test and watch it pass**

```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); import test_progress as t; t.test_dashboard_campaign_panel(); print('dashboard ok')"
```

Expected: `dashboard ok` (a dashboard window flashes for about 1.5 s).

- [ ] **Step 11: Run the whole progress suite and the regression suites**

```powershell
.venv\Scripts\python tests\test_progress.py
```

Expected (about 25 s):

```
test_progress_callback_writes_status ...
test_progress_callback_writes_status ok
test_history_is_capped ...
test_history_is_capped ok
test_dashboard_smoke ...
test_dashboard_smoke ok
test_campaign_progress ...
test_campaign_progress ok
test_dashboard_campaign_panel ...
test_dashboard_campaign_panel ok
test_poll_status_keeps_old_header ...
test_poll_status_keeps_old_header ok
all tests passed
```

Then every no-game suite, which covers `test_aim.py` and the files Tasks 3-7 added (`test_campaign.py`,
`test_campaign_rewards.py`, `test_spaces.py`, `test_campaign_env.py`):

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
```

Expected: every file ends with its pass line (`N tests passed`, or `all tests passed` for `test_progress.py`) and the
loop does not throw.

- [ ] **Step 12: Update CLAUDE.md**

Replace:

```markdown
  - `progress.py`: `ProgressCallback`, which writes live training stats to `runs/<run_name>/status.json` (atomic, every 2 s; `state` running/finished/stopped).
```

with:

```markdown
  - `progress.py`: `ProgressCallback`, which writes live training stats to `runs/<run_name>/status.json` (atomic, every 2 s; `state` running/finished/stopped). Campaign runs add a `campaign` block (completion rate and median official time over the last 50 fresh starts, best time over all of them) and `best_checkpoints_level`; both bests survive a restart.
```

Replace:

```markdown
- `python/tests/test_progress.py`: `ProgressCallback` and dashboard smoke tests against a fake env (no game needed).
```

with:

```markdown
- `python/tests/test_progress.py`: `ProgressCallback`, dashboard and `poll_status.py` tests against fake Cyber Grind and campaign envs (no game needed).
```

Replace:

```markdown
- Live dashboard: `python scripts/dashboard.py` (newest run) or `--run cybergrind_ppo_v2`; opens on monitor 3 below the game row (`--monitor`, `--reserve-top`); `--smoke-test` renders once and exits.
```

with:

```markdown
- Live dashboard: `python scripts/dashboard.py` (newest run) or `--run cybergrind_ppo_v2`; opens on monitor 3 below the game row (`--monitor`, `--reserve-top`); `--smoke-test` renders once and exits. A campaign run replaces the Shooting panel with a Campaign panel (fresh and all-episode completion rate, best and median official time, checkpoints per load, new cells, deaths, closest to the exit, the four largest reward parts), charts fresh completion % and checkpoints per load instead of kills/min and wave, and lists checkpoints instead of waves per game.
```

Append at the end of the file (it becomes the last sub-bullet of the **Campaign** status block):

```markdown
  - **Campaign progress tooling is in.** Episodes record `completed`, `fresh_start`, `level_seconds`,
    `checkpoints_level`, `cells_new` and `exit_dist_min`. `status.json` gains `best_checkpoints_level` and, once
    any episode reports `fresh_start`, a `campaign` block: `fresh_window`, `fresh_completion_rate` and
    `median_time_50` over the last 50 fresh starts, and `best_time` over all of them. Only fresh starts count
    because a checkpoint respawn starts partway through the level and the official timer carries over.
    `poll_status.py` logs these as columns (`part_time`, `part_checkpoint`, ... for the reward parts), and now
    keeps an existing `metrics_log.csv` header, dropping columns that header lacks: move an old log aside to get
    the new columns. `route_progress` is gone from `status.json`.
```

- [ ] **Step 13: Update README.md**

In `README.md` (`## Usage`), replace:

```markdown
**Watch training.** A small desktop window shows live progress: steps and ETA, recent reward, kills and waves, charts, and which games have stalled. It only reads `runs/<run_name>/status.json`, which `train.py` writes, so you can open and close it at any time.
```

with:

```markdown
**Watch training.** A small desktop window shows live progress: steps and ETA, recent reward, kills and waves, charts, and which games have stalled. On a campaign run it shows the fresh-start completion rate, best and median official level time, and checkpoints reached instead of the shooting stats and waves. It only reads `runs/<run_name>/status.json`, which `train.py` writes, so you can open and close it at any time.
```

- [ ] **Step 14: Commit and push**

```powershell
cd F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/progress.py python/scripts/poll_status.py python/scripts/dashboard.py python/tests/test_progress.py CLAUDE.md README.md
git commit -m "Show campaign progress in status.json, the dashboard and metrics_log.csv" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the push reports `main -> main`.

### Task 10: `keep_best.py --metric campaign`

`keep_best.py` keeps scoring Cyber Grind on smoothed kills/min (full 100-episode windows, ties on fewer deaths). A
`--metric campaign` mode scores the fresh-start completion rate instead (samples need `fresh_window >= 20`) and
breaks ties on the lower best official time. The "is this better" decision, the `best.json` reader and the
copy-and-record step become small functions, so the tests can exercise them without a loop or a game.

**Files:**
- Modify: `python/scripts/keep_best.py` (whole file: `--metric`, `Metric`/`METRICS`, `scored(csv_path, metric)`, `rank_key`, `best_of`, `is_better`, `stored_best`, `stored_penalty_name`, `save_if_better`, and a refusal when `best.json` was written by the other metric; `num` and `nearest_checkpoint` are unchanged)
- Create: `python/tests/test_keep_best.py`
- Modify: `CLAUDE.md` (Commands: campaign usage and the test list; Layout: the new test file)
- Test: `python/tests/test_keep_best.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_keep_best.py`:

```python
"""keep_best.py scoring for Cyber Grind and the campaign. No game needed:  python tests/test_keep_best.py  (or pytest)."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import keep_best  # noqa: E402

# The metrics_log.csv columns keep_best reads, as poll_status.py writes them.
COLUMNS = ["wall_time", "timesteps", "episodes", "window", "reward", "kills_per_min", "deaths",
           "fresh_window", "fresh_completion_rate", "median_time_50", "best_time"]

# models/cybergrind_ppo_v2/best.json as keep_best.py wrote it before --metric existed.
OLD_BEST_JSON = {
    "score_metric": "kills_per_min, smoothed over 9 samples",
    "score": 7.1567,
    "deaths": 1.5478,
    "reward": 231.472,
    "at_timesteps": 4697515.0,
    "checkpoint": "ckpt_4679085_steps.zip",
    "saved_at": "2026-09-16 04:22:09",
}


def write_log(folder: Path, rows: list[dict]) -> Path:
    """A metrics_log.csv the way poll_status.py writes it: None and missing values become empty cells."""
    path = folder / "metrics_log.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def grind_rows(kills_per_min: list[float], window: int = 100) -> list[dict]:
    return [{"timesteps": 10_000 * (i + 1), "window": window, "reward": 100.0 + i, "kills_per_min": k, "deaths": 2.0 - 0.1 * i}
            for i, k in enumerate(kills_per_min)]


def campaign_rows(rates: list[float], best_times: list[float | None], fresh_window: int = 50) -> list[dict]:
    return [{"timesteps": 10_000 * (i + 1), "window": 100, "reward": 10.0 * i, "fresh_window": fresh_window,
             "fresh_completion_rate": rate, "best_time": best_time}
            for i, (rate, best_time) in enumerate(zip(rates, best_times))]


def test_kills_per_min_needs_a_full_window():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_log(Path(tmp), grind_rows([5.0] * 12, window=99))
        assert keep_best.scored(path, "kills_per_min") == []
        path = write_log(Path(tmp), grind_rows([5.0] * 8, window=99) + grind_rows([6.0] * 9))
        series = keep_best.scored(path, "kills_per_min")
        assert len(series) == 1 and abs(series[0][0] - 6.0) < 1e-9
        assert keep_best.scored(path) == series  # kills_per_min stays the default


def test_kills_per_min_smoothing_is_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        series = keep_best.scored(write_log(Path(tmp), grind_rows([float(k) for k in range(1, 12)])), "kills_per_min")
        assert [round(s[0], 9) for s in series] == [5.0, 6.0, 7.0]  # 11 samples give 3 full 9-sample windows
        score, deaths, reward, timesteps = series[0]
        assert abs(deaths - 1.6) < 1e-9  # mean of 2.0 - 0.1 * i over i = 0..8
        assert abs(reward - 104.0) < 1e-9 and timesteps == 50_000  # timesteps of the window's middle sample


def test_campaign_needs_twenty_fresh_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        assert keep_best.scored(write_log(Path(tmp), campaign_rows([0.5] * 12, [80.0] * 12, fresh_window=19)), "campaign") == []
        ready = campaign_rows([0.5] * 9, [80.0] * 9, fresh_window=20)
        for row in ready:
            row["window"] = 3  # the 100-episode gate belongs to kills_per_min only
        path = write_log(Path(tmp), ready)
        series = keep_best.scored(path, "campaign")
        assert len(series) == 1 and abs(series[0][0] - 0.5) < 1e-9 and abs(series[0][1] - 80.0) < 1e-9
        assert keep_best.scored(path, "kills_per_min") == []


def test_campaign_missing_best_time_is_infinite():
    with tempfile.TemporaryDirectory() as tmp:
        series = keep_best.scored(write_log(Path(tmp), campaign_rows([0.0] * 9, [None] * 9)), "campaign")
        assert len(series) == 1 and series[0][0] == 0.0 and math.isinf(series[0][1])


def test_higher_completion_rate_wins():
    assert keep_best.is_better((0.6, 120.0), (0.5, 80.0))  # a slower best time does not protect a lower rate
    assert not keep_best.is_better((0.4, 60.0), (0.5, 80.0))
    assert keep_best.is_better((0.0, math.inf), None)  # nothing stored yet
    series = [(0.5, 80.0, 0.0, 1.0), (0.6, 120.0, 0.0, 2.0), (0.4, 60.0, 0.0, 3.0)]
    assert keep_best.best_of(series)[3] == 2.0


def test_equal_rate_lower_best_time_wins():
    assert keep_best.is_better((0.5, 75.0), (0.5, 80.0))
    assert not keep_best.is_better((0.5, 85.0), (0.5, 80.0))
    assert not keep_best.is_better((0.5, 80.0), (0.5, 80.0))  # strictly better only
    assert keep_best.is_better((0.5, 80.0), (0.5, math.inf))  # the first finite time beats none
    assert not keep_best.is_better((0.5, math.inf), (0.5, math.inf))
    series = [(0.5, 90.0, 0.0, 1.0), (0.5, 75.0, 0.0, 2.0), (0.5, math.inf, 0.0, 3.0)]
    assert keep_best.best_of(series)[3] == 2.0
    with tempfile.TemporaryDirectory() as tmp:
        rows = campaign_rows([0.5] * 18, [90.0] * 9 + [75.0] * 9)
        best = keep_best.best_of(keep_best.scored(write_log(Path(tmp), rows), "campaign"))
        assert abs(best[1] - 75.0) < 1e-9 and best[3] == 140_000  # the last window is all 75 s


def test_old_best_json_reads_deaths_as_penalty():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.json"
        path.write_text(json.dumps(OLD_BEST_JSON, indent=2), encoding="utf-8")
        stored = keep_best.stored_best(path)
        assert stored == (7.1567, 1.5478)
        assert keep_best.is_better((7.1567, 1.40), stored) and not keep_best.is_better((7.1567, 1.60), stored)
        assert not keep_best.is_better((7.15674, 1.5478), stored)  # compared at the precision best.json keeps
        assert keep_best.stored_best(Path(tmp) / "missing.json") is None
        assert keep_best.stored_penalty_name(path) == "deaths"  # so --metric kills_per_min still accepts it
        assert keep_best.stored_penalty_name(Path(tmp) / "missing.json") is None
        # best_of ranks at the same precision: a score higher only past 4 decimals does not hide a lower penalty.
        assert keep_best.best_of([(7.15674, 1.5478, 0.0, 1.0), (7.1567, 1.40, 0.0, 2.0)])[3] == 2.0


def test_new_best_copies_checkpoint_and_records_penalty():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "ckpt_100000_steps.zip").write_bytes(b"early")
        (model_dir / "ckpt_200000_steps.zip").write_bytes(b"late")
        series = [(0.25, 95.0, 40.0, 150_000.0), (0.5, 81.5, 60.0, 230_000.0)]
        assert keep_best.save_if_better(series, model_dir, "campaign", None) == (0.5, 81.5)
        assert (model_dir / "best.zip").read_bytes() == b"late"  # newest checkpoint at or before 230k steps
        saved = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
        assert set(saved) == {"score_metric", "score", "penalty", "penalty_name", "reward", "at_timesteps", "checkpoint", "saved_at"}
        assert saved["score_metric"] == "fresh_completion_rate, smoothed over 9 samples"
        assert saved["score"] == 0.5 and saved["penalty"] == 81.5 and saved["penalty_name"] == "best_time"
        assert saved["reward"] == 60.0 and saved["at_timesteps"] == 230_000.0 and saved["checkpoint"] == "ckpt_200000_steps.zip"
        assert keep_best.stored_best(model_dir / "best.json") == (0.5, 81.5)
        assert keep_best.stored_penalty_name(model_dir / "best.json") == "best_time"


def test_restart_does_not_resave_the_same_best():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        (model_dir / "ckpt_50000_steps.zip").write_bytes(b"weights")
        series = [(0.0, math.inf, -30.0, 60_000.0)]  # no completed run yet
        keep_best.save_if_better(series, model_dir, "campaign", None)
        saved = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
        assert saved["penalty"] is None and saved["penalty_name"] == "best_time"  # strict JSON has no infinity
        stored = keep_best.stored_best(model_dir / "best.json")
        assert stored[0] == 0.0 and math.isinf(stored[1])
        (model_dir / "best.zip").unlink()
        assert keep_best.save_if_better(series, model_dir, "campaign", stored) == stored
        assert not (model_dir / "best.zip").exists()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

Run (PowerShell; every later command in this task up to the commit runs from `python\`):

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_keep_best.py
```

Expected: FAIL in the first test alphabetically, `test_campaign_missing_best_time_is_infinite`, with
`TypeError: scored() takes 1 positional argument but 2 were given` (the current `scored` has no `metric`).

- [ ] **Step 3: Rewrite `keep_best.py`**

Replace the entire contents of `python/scripts/keep_best.py` (135 lines, from
`"""Preserves the best-performing checkpoint of a run, automatically.` to the final `    main()`) with the file
below. `num` and `nearest_checkpoint` are copied unchanged. Three deliberate behaviour changes for kills_per_min:
an equal score with fewer deaths now replaces the stored best (the contract's `(score, -penalty)` rule); picking
the best sample and comparing it with the stored best both happen at the 4-decimal precision `best.json` stores
(`rank_key`), so restarting the script no longer re-saves the best it already holds (the old
`score > best_score + 1e-9` compared an unrounded score with a rounded one); and the script refuses to run when
`best.json` was written by the other metric (its `penalty_name`, or `deaths` for files from before `--metric`),
because running the campaign run without `--metric campaign` would otherwise overwrite its `best.zip` with a
kills/min pick and leave a score of ~7 that no completion rate can ever beat.

```python
"""Preserves the best-performing checkpoint of a run, automatically.

PPO does not improve monotonically. On the night of 2026-09-15 the run peaked at 4.40M steps
(kills/min 6.62, reward 217, deaths 1.33) and then degraded for half a million steps as entropy
collapsed, down to 5.36 kills/min by 4.93M. Nothing was watching, and `latest.zip` tracks the LAST
policy, not the best one, so the peak would have been lost with the next checkpoint rotation.

This copies the checkpoint nearest each new best to `best.zip` and records how it was chosen in
`best.json`. It only reads `metrics_log.csv` (written by poll_status.py) and copies files, so it is
safe to leave running alongside training and never touches the bridge ports.

Scoring, smoothed over several consecutive samples in both modes:
- `--metric kills_per_min` (default, Cyber Grind): kills per game-minute, only from samples backed by a
  full 100-episode window. Ties break on lower deaths.
- `--metric campaign`: the completion rate over the last 50 fresh-start episodes, only from samples
  backed by at least 20 of them. Ties break on the lower best official time.

    python scripts/keep_best.py --run cybergrind_ppo_v2            # watch until stopped
    python scripts/keep_best.py --run cybergrind_ppo_v2 --once     # report and exit
    python scripts/keep_best.py --run campaign_ppo --metric campaign
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import NamedTuple

SMOOTH = 9          # samples per smoothing window (~4.5 min at a 30 s poll)
MIN_WINDOW = 100    # kills_per_min: require a full episode window behind every mean
MIN_FRESH_WINDOW = 20  # campaign: fresh-start episodes behind the completion rate (its window holds 50)


class Metric(NamedTuple):
    window: str     # CSV column counting the episodes behind the means
    min_window: int
    score: str      # CSV column, higher is better
    penalty: str    # CSV column that breaks ties, lower is better; also `penalty_name` in best.json
    missing: float  # penalty when no sample in a smoothing window has one
    unit: str


METRICS = {
    "kills_per_min": Metric("window", MIN_WINDOW, "kills_per_min", "deaths", 0.0, "kills/min"),
    # No completed run means no best time: an infinite penalty, so the first finite time wins the tie.
    "campaign": Metric("fresh_window", MIN_FRESH_WINDOW, "fresh_completion_rate", "best_time", math.inf, "fresh completion rate"),
}


def num(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def scored(csv_path: Path, metric: str = "kills_per_min") -> list[tuple[float, float, float, float]]:
    """(score, penalty, reward, timesteps), smoothed, newest last."""
    m = METRICS[metric]
    try:
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    rows = [r for r in rows if (num(r, m.window) or 0) >= m.min_window and num(r, m.score) is not None]
    out = []
    half = SMOOTH // 2
    for i in range(half, len(rows) - half):
        w = rows[i - half:i + half + 1]
        score = [num(r, m.score) for r in w]
        pen = [num(r, m.penalty) for r in w]
        rew = [num(r, "reward") for r in w]
        if any(v is None for v in score):
            continue
        out.append((
            statistics.mean(score),
            statistics.mean([p for p in pen if p is not None] or [m.missing]),
            statistics.mean([v for v in rew if v is not None] or [0.0]),
            num(rows[i], "timesteps") or 0.0,
        ))
    return out


def rank_key(score: float, penalty: float) -> tuple[float, float]:
    """Higher is better: the score, then the negated penalty, both at the 4 decimals best.json keeps.

    best_of and is_better share it, so the sample picked as best is always the one compared with the stored best,
    and a restart does not re-save the best it already holds (a raw score against its own rounded copy).
    """
    return round(score, 4), -round(penalty, 4)


def best_of(series: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """The sample with the highest score, ties broken on the lowest penalty."""
    return max(series, key=lambda s: rank_key(s[0], s[1]))


def is_better(candidate: tuple[float, float], stored: tuple[float, float] | None) -> bool:
    """True when (score, penalty) strictly beats the stored best: a higher score, or the same score and a lower penalty."""
    if stored is None:
        return True
    return rank_key(*candidate) > rank_key(*stored)


def stored_best(best_json: Path) -> tuple[float, float] | None:
    """(score, penalty) of the saved best, or None. best.json files written before --metric keep the penalty as `deaths`."""
    try:
        data = json.loads(best_json.read_text(encoding="utf-8"))
        penalty = data.get("penalty", data.get("deaths"))
        return float(data["score"]), (math.inf if penalty is None else float(penalty))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def stored_penalty_name(best_json: Path) -> str | None:
    """The tie-break the saved best was chosen with (`deaths` for files written before --metric), or None without one."""
    try:
        return str(json.loads(best_json.read_text(encoding="utf-8")).get("penalty_name", "deaths"))
    except (OSError, ValueError, AttributeError):
        return None


def nearest_checkpoint(model_dir: Path, timesteps: float) -> Path | None:
    """Newest checkpoint at or before `timesteps` -- the weights that produced that score."""
    cks = []
    for p in model_dir.glob("ckpt_*_steps.zip"):
        m = re.search(r"ckpt_(\d+)_steps", p.name)
        if m:
            cks.append((int(m.group(1)), p))
    cks.sort()
    at_or_before = [p for n, p in cks if n <= timesteps]
    return at_or_before[-1] if at_or_before else None


def save_if_better(series: list[tuple[float, float, float, float]], model_dir: Path, metric: str,
                   best: tuple[float, float] | None) -> tuple[float, float] | None:
    """Copies the checkpoint behind the series' best sample to best.zip when it beats `best`. Returns the best now held."""
    m = METRICS[metric]
    score, penalty, reward, ts = best_of(series)
    if not is_better((score, penalty), best):
        return best
    src = nearest_checkpoint(model_dir, ts)
    if not (src and src.exists()):
        return best
    shutil.copy2(src, model_dir / "best.zip")
    (model_dir / "best.json").write_text(json.dumps({
        "score_metric": "%s, smoothed over %d samples" % (m.score, SMOOTH),
        "score": round(score, 4),
        "penalty": round(penalty, 4) if math.isfinite(penalty) else None,  # null: no completed run yet
        "penalty_name": m.penalty,
        "reward": round(reward, 3),
        "at_timesteps": ts,
        "checkpoint": src.name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[keep_best] new best {score:.2f} {m.unit} ({m.penalty} {penalty:.2f}) at {ts:,.0f} -> {src.name}", flush=True)
    return score, penalty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cybergrind_ppo_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--metric", choices=sorted(METRICS), default="kills_per_min",
                    help="kills_per_min (Cyber Grind) or campaign (fresh-start completion rate, then best time)")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    metric = METRICS[a.metric]
    csv_path = Path(a.runs_dir) / a.run / "metrics_log.csv"
    model_dir = Path(a.models_dir) / a.run
    best_json = model_dir / "best.json"
    # A best chosen by the other metric is not comparable (7.16 kills/min would outrank any completion rate), and
    # overwriting it would silently replace that run's best.zip, so refuse instead.
    held = stored_penalty_name(best_json)
    if held is not None and held != metric.penalty:
        ap.error(f"{best_json} holds a best chosen with the {held!r} tie-break, not {metric.penalty!r}: "
                 f"pass the --metric that run was scored with")
    best = stored_best(best_json)

    while True:
        series = scored(csv_path, a.metric)
        if series:
            best = save_if_better(series, model_dir, a.metric, best)
            newest = series[-1]
            # Warn when the current policy has fallen well below the best: that is the signal to roll back.
            if best is not None and best[0] > 0 and newest[0] < best[0] * 0.85:
                print(f"[keep_best] WARNING current {newest[0]:.2f} {metric.unit} is "
                      f"{(1 - newest[0]/best[0])*100:.0f}% below best {best[0]:.2f} "
                      f"(best.zip holds the good weights)", flush=True)
        if a.once:
            if best_json.exists():
                print(best_json.read_text(encoding="utf-8"))
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `python\`):

```powershell
.venv\Scripts\python tests\test_keep_best.py
```

Expected: nine `ok test_...` lines (two of the tests also print a `[keep_best] new best ...` line) and
`9 tests passed`.

- [ ] **Step 5: Check the CLI against the real Cyber Grind best**

Run (from `python\`):

```powershell
.venv\Scripts\python scripts\keep_best.py --help
.venv\Scripts\python scripts\keep_best.py --run cybergrind_ppo_v2 --once
.venv\Scripts\python scripts\keep_best.py --run cybergrind_ppo_v2 --metric campaign --once
git -C .. diff --stat -- python/models
```

Expected: the help lists `--metric {campaign,kills_per_min}`. `--once` reads the real
`runs/cybergrind_ppo_v2/metrics_log.csv` (945 lines, 864 smoothed samples): its best sample is 7.15667 kills/min
with 1.54778 deaths at 4,697,515 steps, which rounds to exactly the stored best (7.1567, 1.5478), so nothing is
copied. It prints `[keep_best] WARNING current 3.86 kills/min is 46% below best 7.16 (best.zip holds the good
weights)` (the paused run's last samples), then the existing `best.json` unchanged, still with `deaths` and no
`penalty`. The `--metric campaign` run exits with code 2 and `keep_best.py: error: models\cybergrind_ppo_v2\best.json
holds a best chosen with the 'deaths' tie-break, not 'best_time': pass the --metric that run was scored with`.
`git diff --stat` prints nothing: `best.zip` and `best.json` are untouched.

- [ ] **Step 6: Run the regression suites**

Run (from `python\`; every test file in `tests\`, stopping at the first failure):

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { & .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
```

Expected: nothing is thrown; `test_progress.py` ends with `all tests passed` (it has its own runner) and every other
file ends with `N tests passed`, `test_keep_best.py` with `9 tests passed`.

- [ ] **Step 7: Update CLAUDE.md**

In the Commands section, replace:

```
  (maintains `best.zip`, warns when the current policy falls more than 15% below it).
```

with:

```
  (maintains `best.zip`, warns when the current policy falls more than 15% below it).
  For the campaign run use `python scripts/keep_best.py --run campaign_ppo --metric campaign`: it scores the
  completion rate over the last 50 fresh-start episodes (samples need `fresh_window >= 20`), breaks ties on the
  lower best official time, and `best.json` records the tie-break as `penalty` / `penalty_name` (older files
  that only have `deaths` still load). It refuses to start when `best.json` was written by the other metric, so
  forgetting `--metric campaign` cannot overwrite the campaign `best.zip`.
```

In the Layout section, replace:

```
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
```

with:

```
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_keep_best.py`: `keep_best.py` scoring for both metrics on synthetic `metrics_log.csv` rows, and old `best.json` files (no game needed; `python tests/test_keep_best.py`).
```

In the Commands section, add `python tests/test_keep_best.py` as the last file of the one-line `- Tests (no game):`
bullet. Tasks 3, 5, 6 and 7 each appended their own file to it, so it ends as Task 7 left it. Replace this exact
text (the end of that bullet, not a whole line; it occurs once):

```
`python tests/test_spaces.py` and `python tests/test_campaign_env.py` (pytest is not installed; the files also work under pytest).
```

with:

```
`python tests/test_spaces.py`, `python tests/test_campaign_env.py` and `python tests/test_keep_best.py` (pytest is not installed; the files also work under pytest).
```

If an earlier task left the bullet listing other files, keep every file it lists and add
`python tests/test_keep_best.py` as the last one, with ` and ` only in front of it.

- [ ] **Step 8: Commit and push**

Run:

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/scripts/keep_best.py python/tests/test_keep_best.py CLAUDE.md
git commit -m "Add a campaign metric to keep_best: fresh completion rate, then best time" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: one commit with three files; the push reaches `main`.

### Task 11: `times.py` and `eval.py --level/--record-times`

`times.py` owns the `times.md` format: the leaderboard keeps one row per level (sorted by campaign order,
replaced only by a faster time) and the generation history gets every recorded run, newest first, with its time
change against the newest earlier run on the same level. `eval.py` gains `--level` and `--record-times`: campaign
evaluation always starts from a fresh level load with real deaths, reads (never writes) the first training game's
exploration counts because the policy takes them as inputs, prints completed, official time, kills, style,
restarts, rank and deaths per episode (the spec's list plus the contract's), and records the fastest completion.

**Files:**
- Create: `python/ultrakill_ai/times.py`
- Create: `python/tests/test_times.py`
- Modify: `python/scripts/eval.py` (docstring, imports, `report_campaign`, `--level`, `--record-times`, campaign overrides, read-only exploration counts, per-episode print)
- Modify: `CLAUDE.md` (Workflow rules, Layout, Commands)
- Test: `python/tests/test_times.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_times.py` (the `TIMES_MD` string is `times.md` exactly as committed, em dashes and `Δ`
included):

```python
"""times.md bookkeeping. No game needed:  python tests/test_times.py  (or pytest)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.times import TimeEntry, format_delta, format_time, parse_time, record, record_file  # noqa: E402

# times.md at the repo root, exactly as committed before any campaign run.
TIMES_MD = """# Times

Leaderboard of the AI's level completion times across training generations.
Times are in-game level time (`mm:ss.mmm`), not wall-clock training time.

## Leaderboard

Best time per level. A generation only takes a spot by beating the current record.

| Level | Time | Rank | Generation | Difficulty | Date | Notes |
|-------|------|------|------------|------------|------|-------|
| — | — | — | — | — | — | No completed runs yet |

## Generation history

Best run from each generation, newest first. Keep every generation here, even ones that didn't set a record.

| Generation | Level | Time | Rank | Kills | Deaths | Δ vs previous | Date | Notes |
|------------|-------|------|------|-------|--------|---------------|------|-------|
| — | — | — | — | — | — | — | — | No generations trained yet |

<!--
How to add an entry:
- Generation history: add a row at the top of the table for the new generation's best run.
- Leaderboard: if that run beats the level's record, replace the level's row (one row per level, sorted by level order).
- Rank is the in-game style rank (D, C, B, A, S, P). Δ vs previous is the time change from the previous generation on the same level, e.g. -1.250s.
-->
"""


def entry(level="Level 0-1", seconds=83.25, rank="S", generation="campaign_ppo@1.00M", kills=12, deaths=1,
          notes="3/5 eval runs completed") -> TimeEntry:
    return TimeEntry(level=level, seconds=seconds, rank=rank, generation=generation, difficulty=3, date="2026-09-20",
                     kills=kills, deaths=deaths, notes=notes)


def table_rows(markdown: str, heading: str) -> list[str]:
    """Data rows (below the header and separator) of the table under a heading."""
    lines = markdown.splitlines()
    i = lines.index(heading) + 1
    while not lines[i].startswith("|"):
        i += 1
    end = i
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return lines[i + 2:end]


def column(rows: list[str], index: int) -> list[str]:
    return [row.strip().strip("|").split("|")[index].strip() for row in rows]


def skeleton(markdown: str) -> list[str]:
    """Every line except table data rows: headings, prose, table headers and the closing comment."""
    out, position = [], 0
    for line in markdown.splitlines():
        position = position + 1 if line.startswith("|") else 0
        if position <= 2:
            out.append(line)
    return out


def test_first_record_replaces_both_placeholders():
    out = record(TIMES_MD, entry())
    assert table_rows(out, "## Leaderboard") == [
        "| 0-1 | 01:23.250 | S | campaign_ppo@1.00M | Violent | 2026-09-20 | 3/5 eval runs completed |"]
    assert table_rows(out, "## Generation history") == [
        "| campaign_ppo@1.00M | 0-1 | 01:23.250 | S | 12 | 1 | — | 2026-09-20 | 3/5 eval runs completed |"]
    assert "No completed runs yet" not in out and "No generations trained yet" not in out
    assert skeleton(out) == skeleton(TIMES_MD)
    assert record(TIMES_MD.replace("\n", "\r\n"), entry()) == out  # a CRLF checkout gives the same result


def test_slower_run_keeps_the_record():
    first = record(TIMES_MD, entry())
    out = record(first, entry(seconds=83.75, rank="A", generation="campaign_ppo@2.00M", kills=10, deaths=2,
                              notes="4/5 eval runs completed"))
    assert table_rows(out, "## Leaderboard") == table_rows(first, "## Leaderboard")
    assert table_rows(out, "## Generation history") == [
        "| campaign_ppo@2.00M | 0-1 | 01:23.750 | A | 10 | 2 | +0.500s | 2026-09-20 | 4/5 eval runs completed |",
        "| campaign_ppo@1.00M | 0-1 | 01:23.250 | S | 12 | 1 | — | 2026-09-20 | 3/5 eval runs completed |",
    ]
    tie = record(first, entry(generation="campaign_ppo@2.00M"))
    assert table_rows(tie, "## Leaderboard") == table_rows(first, "## Leaderboard")  # only a faster time takes the spot
    assert column(table_rows(tie, "## Generation history"), 6) == ["+0.000s", "—"]


def test_faster_run_takes_the_record():
    slower = record(record(TIMES_MD, entry()), entry(seconds=83.75, generation="campaign_ppo@2.00M"))
    out = record(slower, entry(seconds=82.0, generation="campaign_ppo@3.00M", kills=11, deaths=0,
                               notes="5/5 eval runs completed"))
    assert table_rows(out, "## Leaderboard") == [
        "| 0-1 | 01:22.000 | S | campaign_ppo@3.00M | Violent | 2026-09-20 | 5/5 eval runs completed |"]
    history = table_rows(out, "## Generation history")
    assert len(history) == 3
    # Against the newest earlier run on the level (83.75 s), not against the record (83.25 s).
    assert history[0] == "| campaign_ppo@3.00M | 0-1 | 01:22.000 | S | 11 | 0 | -1.750s | 2026-09-20 | 5/5 eval runs completed |"


def test_leaderboard_stays_in_level_order():
    out = TIMES_MD
    for level, seconds in (("Level 0-2", 95.5), ("Level 1-1", 200.0), ("Level 0-1", 83.25)):
        out = record(out, entry(level=level, seconds=seconds))
    assert column(table_rows(out, "## Leaderboard"), 0) == ["0-1", "0-2", "1-1"]
    history = table_rows(out, "## Generation history")
    assert column(history, 1) == ["0-1", "1-1", "0-2"]  # newest first
    assert column(history, 6) == ["—", "—", "—"]  # no earlier run on any of these levels


def test_format_and_parse_time_round_trip():
    assert format_time(83.25) == "01:23.250"
    assert format_time(0.0) == "00:00.000" and format_time(59.9996) == "01:00.000"
    assert format_time(3725.5) == "62:05.500"
    for seconds in (0.0, 7.5, 83.25, 599.999, 3725.5):
        assert abs(parse_time(format_time(seconds)) - seconds) < 1e-9
    for text in ("00:00.000", "01:23.250", "09:59.999", "62:05.500"):
        assert format_time(parse_time(text)) == text
    assert parse_time("—") is None and parse_time("") is None and parse_time("1:2:3") is None
    assert format_delta(-1.25) == "-1.250s" and format_delta(0.5) == "+0.500s"


def test_html_comment_survives():
    comment = TIMES_MD[TIMES_MD.index("<!--"):]
    out = TIMES_MD
    for seconds in (90.0, 85.0, 88.0):
        out = record(out, entry(seconds=seconds))
    assert out.endswith(comment) and out.count("<!--") == 1
    assert skeleton(out) == skeleton(TIMES_MD)


def test_record_file_rewrites_in_place():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "times.md"
        path.write_text(TIMES_MD, encoding="utf-8")
        record_file(path, entry())
        assert path.read_text(encoding="utf-8") == record(TIMES_MD, entry())


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

Run (PowerShell; every later command in this task up to the commit runs from `python\`):

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_times.py
```

Expected: FAIL at import with `ModuleNotFoundError: No module named 'ultrakill_ai.times'`.

- [ ] **Step 3: Write `times.py`**

Create `python/ultrakill_ai/times.py`:

```python
"""The AI's level-time leaderboard: `times.md` at the repo root.

`record` is pure (markdown in, markdown out), so it is tested without touching the real file, and
`record_file` applies it in place. It follows the rules in the file's closing HTML comment: the
generation history gets every recorded run, newest first, with the time change against the newest
earlier run on the same level; the leaderboard keeps one row per level, sorted by campaign order and
replaced only by a faster time. Level cells use the short form (`0-1`) and times are `mm:ss.mmm`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ultrakill_ai.campaign import CAMPAIGN_LEVELS

DIFFICULTY_NAMES = ("Harmless", "Lenient", "Standard", "Violent", "Brutal")
LEADERBOARD_HEADING = "## Leaderboard"
HISTORY_HEADING = "## Generation history"
EMPTY = "—"  # em dash: an empty cell; a row whose first cell is one is a placeholder
_TIME = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")
_LEVEL_ORDER = {scene.removeprefix("Level "): i for i, scene in enumerate(CAMPAIGN_LEVELS)}


@dataclass
class TimeEntry:
    """One evaluated run, as it goes into both tables."""

    level: str  # scene name, e.g. "Level 0-1"
    seconds: float  # official level time: StatsManager.seconds when the real FinalPit stopped the timer
    rank: str  # "D".."S" or "P" ("" when the level reported no rank thresholds)
    generation: str  # run name and step count, e.g. "campaign_ppo@1.25M"
    difficulty: int  # 0 Harmless .. 4 Brutal
    date: str  # YYYY-MM-DD
    kills: int
    deaths: int
    notes: str = ""


def format_time(seconds: float) -> str:
    """83.25 -> "01:23.250", in whole milliseconds."""
    ms = _ms(max(0.0, seconds))
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def parse_time(text: str) -> float | None:
    """The inverse of format_time: "01:23.250" -> 83.25, and None for anything else (such as the placeholder dash)."""
    m = _TIME.match(text.strip())
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def format_delta(seconds: float) -> str:
    """A signed time change: -1.25 -> "-1.250s", 0.5 -> "+0.500s"."""
    return f"{seconds:+.3f}s"


def short_level(scene: str) -> str:
    """The level cell times.md uses: "Level 0-1" -> "0-1"."""
    return scene.removeprefix("Level ")


def record(markdown: str, entry: TimeEntry) -> str:
    """Returns `markdown` with `entry` added to both tables. Everything outside the table rows is kept as it is."""
    lines = markdown.splitlines()
    level = short_level(entry.level)
    time_text = format_time(entry.seconds)
    rank = entry.rank or EMPTY
    if 0 <= entry.difficulty < len(DIFFICULTY_NAMES):
        difficulty = DIFFICULTY_NAMES[entry.difficulty]
    else:
        difficulty = str(entry.difficulty)

    # Leaderboard: one row per level, replaced only by a strictly faster time.
    start, end = _table(lines, LEADERBOARD_HEADING)
    rows = _data_rows(lines[start:end])
    row = [level, time_text, rank, entry.generation, difficulty, entry.date, entry.notes]
    held = next((i for i, cells in enumerate(rows) if cells[0] == level), None)
    if held is None:
        rows.append(row)
    else:
        record_time = parse_time(rows[held][1])
        if record_time is None or _ms(entry.seconds) < _ms(record_time):
            rows[held] = row
    rows.sort(key=lambda cells: _LEVEL_ORDER.get(cells[0], len(_LEVEL_ORDER)))
    lines[start:end] = [_format_row(cells) for cells in rows]

    # Generation history: every run, newest first, compared with the newest earlier run on the same level.
    start, end = _table(lines, HISTORY_HEADING)
    rows = _data_rows(lines[start:end])
    previous = next((parse_time(cells[2]) for cells in rows if len(cells) > 2 and cells[1] == level), None)
    delta = EMPTY if previous is None else format_delta((_ms(entry.seconds) - _ms(previous)) / 1000)
    rows.insert(0, [entry.generation, level, time_text, rank, str(entry.kills), str(entry.deaths), delta, entry.date, entry.notes])
    lines[start:end] = [_format_row(cells) for cells in rows]
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def record_file(path, entry: TimeEntry) -> None:
    """Adds `entry` to the times.md at `path`, in place."""
    path = Path(path)
    path.write_text(record(path.read_text(encoding="utf-8"), entry), encoding="utf-8")


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _format_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"


def _data_rows(lines: list[str]) -> list[list[str]]:
    """Table rows as cell lists, placeholder rows dropped."""
    rows = [_cells(line) for line in lines]
    return [cells for cells in rows if cells[0] != EMPTY]


def _table(lines: list[str], heading: str) -> tuple[int, int]:
    """[start, end) line range of the data rows (below the header and separator) of the table under `heading`."""
    if heading not in lines:
        raise ValueError(f"times.md has no {heading!r} section")
    i = lines.index(heading) + 1
    while i < len(lines) and not lines[i].startswith(("|", "#")):
        i += 1
    if i + 1 >= len(lines) or not lines[i].startswith("|"):
        raise ValueError(f"times.md has no table under {heading!r}")
    end = i + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return i + 2, end
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `python\`):

```powershell
.venv\Scripts\python tests\test_times.py
```

Expected: seven `ok test_...` lines and `7 tests passed`.

- [ ] **Step 5: Add `--level` and `--record-times` to `eval.py`**

Six replacements in `python/scripts/eval.py`.

(a) Replace the docstring:

```python
"""Runs a trained model so you can watch it or measure it.

    python scripts/eval.py models/cybergrind_ppo/latest.zip --episodes 3 --realtime
"""
```

with:

```python
"""Runs a trained model so you can watch it or measure it.

    python scripts/eval.py models/cybergrind_ppo/latest.zip --episodes 3 --realtime
    python scripts/eval.py models/campaign_ppo/best.zip --level "Level 0-1" --episodes 10 --record-times

Campaign evaluation always starts from a fresh level load with real deaths. It reads the first training game's
exploration counts (the policy was trained with them as inputs) but never writes them back, and never writes the
training runs' best-run files. `--record-times` adds the fastest completion to the repo-root times.md
(generation history, plus the leaderboard when it is a record).
"""
```

(b) Replace:

```python
import argparse
import sys
from pathlib import Path
```

with:

```python
import argparse
import sys
import time
from pathlib import Path
```

(c) Replace:

```python
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
```

with:

```python
from ultrakill_ai.campaign import CAMPAIGN_LEVELS, ExplorationArchive, safe_name  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.times import TimeEntry, format_time, record_file  # noqa: E402

TIMES_MD = Path(__file__).resolve().parents[2] / "times.md"


def report_campaign(args: argparse.Namespace, cfg: EnvConfig, model, results: list[tuple[float, dict]]) -> None:
    """Completion summary for a campaign eval, and the times.md entry for its fastest completion."""
    completions = sum(1 for _, info in results if info.get("completed"))
    print(f"completed {completions}/{len(results)} fresh runs of {cfg.level}")
    timed = [info for _, info in results if info.get("completed") and info.get("level_seconds") is not None]
    if not timed:
        if args.record_times:
            print("no completed run, times.md unchanged")
        return
    best = min(timed, key=lambda info: info["level_seconds"])
    print(f"fastest {format_time(best['level_seconds'])} rank={best.get('rank') or '-'} kills={best['kills']}"
          f" style={best.get('style', 0)} restarts={best.get('restarts', '-')} deaths={best['deaths']}")
    if not args.record_times:
        return
    entry = TimeEntry(
        level=cfg.level,
        seconds=best["level_seconds"],
        rank=best.get("rank") or "",
        generation=f"{Path(args.model).parent.name}@{model.num_timesteps / 1e6:.2f}M",
        difficulty=best["difficulty"],
        date=time.strftime("%Y-%m-%d"),
        kills=best["kills"],
        deaths=best["deaths"],
        notes=f"{completions}/{len(results)} eval runs completed",
    )
    record_file(TIMES_MD, entry)
    print(f"recorded {format_time(entry.seconds)} ({entry.generation}) in {TIMES_MD}")
```

(d) Replace:

```python
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    args = parser.parse_args()

    cfg_path = Path(args.model).parent / "env_config.yaml"
    cfg = EnvConfig.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else EnvConfig()
    if args.realtime:
        cfg.unlimited_fps = False
        cfg.mute = False
        cfg.windowed = False
        cfg.render = True
    cfg.soft_death = False  # evaluate with real deaths
```

with:

```python
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    parser.add_argument("--level", help='campaign scene, e.g. "Level 0-1" (default: the level in env_config.yaml next to the model)')
    parser.add_argument("--record-times", action="store_true", help="campaign: add the fastest completion to times.md")
    args = parser.parse_args()

    cfg_path = Path(args.model).parent / "env_config.yaml"
    cfg = EnvConfig.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else EnvConfig()
    if (args.level or args.record_times) and cfg.mode != "campaign":
        parser.error(f"--level and --record-times need a campaign model; the env config next to it gives mode {cfg.mode!r}")
    if args.level and args.level not in CAMPAIGN_LEVELS:
        parser.error(f"--level {args.level!r} is not a campaign scene name (Level 0-1 .. Level 9-2)")
    if args.level:
        cfg.level = args.level
    if args.realtime:
        cfg.unlimited_fps = False
        cfg.mute = False
        cfg.windowed = False
        cfg.render = True
    cfg.soft_death = False  # evaluate with real deaths
    if cfg.mode == "campaign":
        # Every episode is a fresh level load, so each one is a whole run with an official time. The exploration
        # archive and best-run files belong to the training games, so eval never saves either of them.
        cfg.fresh_start_prob = 1.0
        cfg.explore_dir = ""
        cfg.best_runs_dir = ""
```

(e) Replace:

```python
    env = UltrakillEnv(cfg)
    results = []
```

with:

```python
    env = UltrakillEnv(cfg)
    if cfg.mode == "campaign":
        # The last 9 campaign inputs are the game's visit counts, which in training come from thousands of earlier
        # episodes. A fresh archive would show the policy an all-unexplored map it never trained on, so eval reads the
        # counts the first training game (the config's base port) saved next to the model. explore_dir stays empty,
        # so the env never writes them back. A missing file gives an empty archive, and the count below shows it.
        archive_path = Path(args.model).parent / f"explore_{safe_name(cfg.level)}_{cfg.port}.npz"
        env.archive = ExplorationArchive.load(archive_path, cfg.cell_size)
        print(f"exploration counts: {len(env.archive.counts)} cells from {archive_path}")
    results = []
```

(f) Replace:

```python
                if terminated or truncated:
                    break
            results.append((total, info))
            extra = f" route={info['route_progress']:.0%}" if "route_progress" in info else f" wave={info['wave']}"
            print(f"episode {ep}: reward={total:.1f} steps={steps} kills={info['kills']}{extra} end={info.get('end_reason')}")
    finally:
        env.close()

    print(f"mean reward {np.mean([r for r, _ in results]):.1f} over {len(results)} episodes")
```

with:

```python
                if terminated or truncated:
                    break
            if cfg.mode == "campaign":
                # The difficulty the game actually read this run: the campaign block reports the override, if any.
                info = dict(info, difficulty=(env._raw.get("campaign") or {}).get("difficulty", cfg.difficulty))
                seconds = info.get("level_seconds")
                extra = (f" completed={info.get('completed', 0)} time={format_time(seconds) if seconds is not None else '-'}"
                         f" rank={info.get('rank') or '-'} style={info.get('style', 0)}"
                         f" restarts={info.get('restarts', '-')} deaths={info['deaths']}")
            else:
                extra = f" wave={info['wave']}"
            results.append((total, info))
            print(f"episode {ep}: reward={total:.1f} steps={steps} kills={info['kills']}{extra} end={info.get('end_reason')}")
    finally:
        env.close()

    print(f"mean reward {np.mean([r for r, _ in results]):.1f} over {len(results)} episodes")
    if cfg.mode == "campaign":
        report_campaign(args, cfg, model, results)
```

The console output stays ASCII (`-` for a missing value, not `—`): Windows Python prints to a pipe in cp1252, and a
redirected eval log must not crash on a character.

- [ ] **Step 6: Verify `eval.py` compiles and parses its arguments**

Run (from `python\`):

```powershell
.venv\Scripts\python -m py_compile scripts\eval.py
.venv\Scripts\python scripts\eval.py --help
```

Expected: `py_compile` prints nothing and exits 0. `--help` exits 0 and its usage lists `--level LEVEL` and
`--record-times` alongside the existing `--algo {ppo,rppo}`, `--episodes EPISODES`, `--realtime` and
`--stochastic`. (The in-game `--record-times` run belongs to Task 15.)

- [ ] **Step 7: Run the regression suites**

Run (from `python\`):

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { & .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
git -C .. diff --stat -- times.md
```

Expected: nothing is thrown; `test_progress.py` ends with `all tests passed` (it has its own runner) and every other
file ends with `N tests passed` (including `9 tests passed` for `test_keep_best.py` and `7 tests passed` for
`test_times.py`), and the diff for `times.md` is empty: the tests only write temporary copies.

- [ ] **Step 8: Update CLAUDE.md**

In Workflow rules, replace:

```
- Update `times.md` whenever a training generation finishes (instructions are in an HTML comment at the bottom of that file).
```

with:

```
- Update `times.md` whenever a training generation finishes (instructions are in an HTML comment at the bottom of that file). For campaign levels `python scripts/eval.py <model> --level "Level 0-1" --record-times` does it: the fastest completion goes into the generation history, and onto the leaderboard when it is a record.
```

In the Layout section, replace:

```
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
```

with:

```
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
- `python/ultrakill_ai/times.py`: reads and updates `times.md` (`record` is pure, `record_file` edits in place). Level cells use the short form (`0-1`), times are `mm:ss.mmm`, the leaderboard is sorted by campaign order and only a faster time replaces a row.
```

In the Layout section, replace the line Task 10 added:

```
- `python/tests/test_keep_best.py`: `keep_best.py` scoring for both metrics on synthetic `metrics_log.csv` rows, and old `best.json` files (no game needed; `python tests/test_keep_best.py`).
```

with:

```
- `python/tests/test_keep_best.py`: `keep_best.py` scoring for both metrics on synthetic `metrics_log.csv` rows, and old `best.json` files (no game needed; `python tests/test_keep_best.py`).
- `python/tests/test_times.py`: `times.md` updates against the committed file's exact text: placeholders, records, deltas, level order (no game needed; `python tests/test_times.py`).
```

In the Commands section, replace:

```
- TensorBoard: `tensorboard --logdir runs`.
```

with:

```
- TensorBoard: `tensorboard --logdir runs`.
- Campaign eval (one game on port 47800, e.g. `python scripts/games.py launch --count 1 --monitor 1`):
  `python scripts/eval.py models/campaign_ppo/best.zip --level "Level 0-1" --episodes 10`. Fresh level loads,
  deterministic actions, real deaths; prints completed, official time, rank, kills, style, restarts and deaths per
  episode, then the completion count and the fastest run. It reads the exploration counts the first training game
  saved next to the model (`explore_Level_0-1_47800.npz`, printed as a cell count; 0 cells means the policy sees
  an unexplored map) and never writes them. Add `--record-times` to write the fastest completion to `times.md`.
```

In the Commands section, add `python tests/test_times.py` as the last file of the one-line `- Tests (no game):`
bullet, the same way Task 10 added `test_keep_best.py`. Replace this exact text (the end of that bullet, as Task 10
left it; it occurs once):

```
`python tests/test_campaign_env.py` and `python tests/test_keep_best.py` (pytest is not installed; the files also work under pytest).
```

with:

```
`python tests/test_campaign_env.py`, `python tests/test_keep_best.py` and `python tests/test_times.py` (pytest is not installed; the files also work under pytest).
```

If an earlier task left the bullet listing other files, keep every file it lists and add
`python tests/test_times.py` as the last one, with ` and ` only in front of it.

- [ ] **Step 9: Commit and push**

Run:

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/ultrakill_ai/times.py python/tests/test_times.py python/scripts/eval.py CLAUDE.md
git commit -m "Add times.md bookkeeping and campaign eval with --level and --record-times" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: one commit with four files; the push reaches `main`.

### Task 12: `transfer_weights.py`

Builds the campaign's starting weights from the Cyber Grind `best.zip`: a 479-input PPO policy whose hidden stacks
are the Cyber Grind ones, with first-layer columns 443-478 at zero (so every Cyber Grind input gives exactly the old
hidden features), the action head scaled by 0.5, and a fresh final value layer and optimizer. Depends on Task 6
(`ObsLayout(campaign=True)`, `CAMPAIGN_BLOCK`).

**Files:**
- Create: `python/scripts/transfer_weights.py`
- Create: `python/tests/test_transfer.py`
- Create: `python/models/campaign_ppo/transfer_init.zip` (generated by the script in Step 6, committed)
- Modify: `CLAUDE.md` (Layout: the script and the test; Commands: the no-game tests bullet)
- Test: `python/tests/test_transfer.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_transfer.py`:

```python
"""Weight transfer from the Cyber Grind policy to the campaign layout. No game needed:  python tests/test_transfer.py  (or pytest)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transfer_weights import SHARED_INPUTS, SpacesEnv, transfer, widen_state_dict  # noqa: E402
from ultrakill_ai.spaces import CAMPAIGN_BLOCK, ObsLayout  # noqa: E402

NET_ARCH = [16, 16]


def small_model(campaign: bool, seed: int) -> PPO:
    torch.manual_seed(seed)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=campaign).space()), policy_kwargs={"net_arch": NET_ARCH}, device="cpu")


def inputs(rows: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random Cyber Grind inputs (retired route values zero, as in live play) and the same inputs in the campaign layout."""
    rng = np.random.default_rng(seed)
    grind = rng.uniform(-1.0, 1.0, (rows, ObsLayout().size)).astype(np.float32)
    grind[:, SHARED_INPUTS:] = 0.0
    campaign = np.concatenate([grind[:, :SHARED_INPUTS], np.zeros((rows, CAMPAIGN_BLOCK), dtype=np.float32)], axis=1)
    return torch.from_numpy(grind), torch.from_numpy(campaign)


def outputs(policy, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(policy latent, value latent, action logits)."""
    with torch.no_grad():
        latent_pi, latent_vf = policy.mlp_extractor(x)
        return latent_pi, latent_vf, policy.action_net(latent_pi)


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, rtol=0.0, atol=1e-6)


def widened(action_scale: float):
    """(source model, destination model with the widened weights loaded)."""
    src, dst = small_model(campaign=False, seed=1), small_model(campaign=True, seed=2)
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), SHARED_INPUTS, action_scale))
    return src, dst


def test_layouts_share_the_first_443_inputs():
    assert ObsLayout().size == 448
    assert ObsLayout(campaign=True).size == SHARED_INPUTS + CAMPAIGN_BLOCK == 479


def test_widened_latents_match_the_source():
    src, dst = widened(1.0)
    grind, campaign = inputs()
    src_pi, src_vf, src_logits = outputs(src.policy, grind)
    dst_pi, dst_vf, dst_logits = outputs(dst.policy, campaign)
    assert close(src_pi, dst_pi) and close(src_vf, dst_vf)
    assert close(src_logits, dst_logits)


def test_action_scale_halves_the_logits():
    src, dst = widened(0.5)
    grind, campaign = inputs(seed=1)
    src_pi, src_vf, src_logits = outputs(src.policy, grind)
    dst_pi, dst_vf, dst_logits = outputs(dst.policy, campaign)
    assert close(src_pi, dst_pi) and close(src_vf, dst_vf)
    assert close(dst_logits, 0.5 * src_logits)
    assert (src_logits - dst_logits).abs().max() > 1e-4  # the scale really changed something


def test_widen_keeps_the_destinations_value_head_and_zeroes_new_inputs():
    src, dst = small_model(campaign=False, seed=1), small_model(campaign=True, seed=2)
    src_state, dst_state = src.policy.state_dict(), dst.policy.state_dict()
    dst_before = {name: value.clone() for name, value in dst_state.items()}
    out = widen_state_dict(src_state, dst_state, SHARED_INPUTS, 0.5)
    assert set(out) == set(dst_state)
    for name in ("value_net.weight", "value_net.bias"):
        assert torch.equal(out[name], dst_before[name])
    assert not torch.equal(out["value_net.weight"], src_state["value_net.weight"])
    for net in ("policy_net", "value_net"):
        first = out[f"mlp_extractor.{net}.0.weight"]
        assert first.shape == (NET_ARCH[0], 479)
        assert torch.equal(first[:, :SHARED_INPUTS], src_state[f"mlp_extractor.{net}.0.weight"][:, :SHARED_INPUTS])
        assert not first[:, SHARED_INPUTS:].any()
        assert torch.equal(out[f"mlp_extractor.{net}.2.weight"], src_state[f"mlp_extractor.{net}.2.weight"])
    for name, value in dst_before.items():  # the destination dict itself was not written to
        assert torch.equal(dst_state[name], value)


def test_transfer_saves_a_loadable_campaign_model():
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp) / "grind.zip", Path(tmp) / "campaign" / "transfer_init.zip"
        small_model(campaign=False, seed=1).save(source)
        transfer(source, dest, action_scale=0.5, seed=3)
        loaded = PPO.load(dest, device="cpu")
        assert loaded.observation_space.shape == (479,)
        assert loaded.policy_kwargs == {"net_arch": NET_ARCH}
        assert loaded.seed is None
        grind, campaign = inputs(seed=2)
        _, _, src_logits = outputs(PPO.load(source, device="cpu").policy, grind)
        _, _, dst_logits = outputs(loaded.policy, campaign)
        assert close(dst_logits, 0.5 * src_logits)

        again = Path(tmp) / "again.zip"
        transfer(source, again, action_scale=0.5, seed=3)
        assert torch.equal(PPO.load(again, device="cpu").policy.state_dict()["value_net.weight"], loaded.policy.state_dict()["value_net.weight"])

        try:
            transfer(dest, Path(tmp) / "twice.zip")
        except ValueError as e:
            assert "Cyber Grind layout" in str(e)
        else:
            raise AssertionError("a 479-input source must be rejected")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_transfer.py
```

Expected: FAIL at import, the traceback ending in `ModuleNotFoundError: No module named 'transfer_weights'`.

- [ ] **Step 3: Write `transfer_weights.py`**

Create `python/scripts/transfer_weights.py`:

```python
"""Builds the campaign's starting weights from a Cyber Grind checkpoint.

The campaign observation layout is the Cyber Grind one with its 5 retired route values (inputs 443-447,
always zero in Cyber Grind) replaced by the 36-value campaign block, so inputs 0-442 mean the same thing
in both. The widened policy:
  - copies both hidden stacks (policy and value) from the source;
  - takes first-layer columns 0-442 from the source and leaves columns 443-478 at zero, so on any Cyber
    Grind input it computes exactly what the source did and the campaign block starts disconnected;
  - scales the action head (weights and bias) by --action-scale, which softens every logit and raises
    entropy, so the fighting reflexes carry over without locking in "fire always, walk backwards";
  - keeps its own freshly initialised final value layer, because the campaign's reward scale has nothing
    to do with Cyber Grind's;
  - starts with a fresh optimizer.

    python scripts/transfer_weights.py models/cybergrind_ppo_v2/best.zip models/campaign_ppo/transfer_init.zip
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip

If the first live rollout's entropy is outside 6-10 nats, rerun this once with another --action-scale and
restart training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import ObsLayout, action_space  # noqa: E402

SHARED_INPUTS = 443  # player 0-16, enemies 17-416, rays 417-432, ground rays 433-440, Cyber Grind 441-442
HIDDEN_LAYERS = tuple(
    f"mlp_extractor.{net}.{layer}.{kind}" for net in ("policy_net", "value_net") for layer in (0, 2) for kind in ("weight", "bias")
)
ACTION_HEAD = ("action_net.weight", "action_net.bias")


class SpacesEnv(gym.Env):
    """Holds an observation and action space so PPO can build a policy without a game. Never reset or stepped."""

    def __init__(self, observation_space: gym.spaces.Space, actions: gym.spaces.Space | None = None):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = actions if actions is not None else action_space()


def widen_state_dict(src: dict, dst: dict, shared_inputs: int, action_scale: float) -> dict:
    """`dst` (a policy state dict) with the source's weights widened into it; neither input is modified.

    First-layer columns from `shared_inputs` on are zero; `value_net.*` stays the destination's own.
    """
    out = {name: value.clone() for name, value in dst.items()}
    for name in HIDDEN_LAYERS:
        if name.endswith(".0.weight"):
            widened = torch.zeros_like(dst[name])
            widened[:, :shared_inputs] = src[name][:, :shared_inputs]
            out[name] = widened
        else:
            out[name] = src[name].clone()
    for name in ACTION_HEAD:
        out[name] = src[name] * action_scale
    return out


def transfer(source: Path, dest: Path, action_scale: float = 0.5, seed: int = 0) -> tuple[PPO, PPO]:
    """Loads a Cyber Grind PPO checkpoint, widens it to the campaign layout and saves it to `dest`."""
    src = PPO.load(source, device="cpu")
    expected = ObsLayout().space().shape
    if src.observation_space.shape != expected:
        raise ValueError(f"{source} takes inputs of shape {src.observation_space.shape}, not the Cyber Grind layout's {expected}")
    # Seed torch here instead of passing seed= to PPO: a seed stored in the model would reseed every
    # later resume of the campaign run to the same random sequence.
    torch.manual_seed(seed)
    dst = PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), src.action_space), policy_kwargs=src.policy_kwargs, device="cpu")
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), SHARED_INPUTS, action_scale))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dst.save(dest)
    return src, dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="Cyber Grind PPO checkpoint, e.g. models/cybergrind_ppo_v2/best.zip")
    ap.add_argument("dest", type=Path, help="where to save the campaign starting weights")
    ap.add_argument("--action-scale", type=float, default=0.5, help="multiplies the action logits; below 1 raises entropy")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the fresh value head")
    a = ap.parse_args()

    src, dst = transfer(a.source, a.dest, a.action_scale, a.seed)
    src_state = src.policy.state_dict()
    for name, value in dst.policy.state_dict().items():
        if name.endswith(".0.weight"):
            note = f"columns 0-{SHARED_INPUTS - 1} copied, {SHARED_INPUTS}+ zero"
        elif name in ACTION_HEAD:
            note = f"copied x{a.action_scale}"
        elif name in HIDDEN_LAYERS:
            note = "copied"
        else:
            note = "fresh"
        print(f"{name:34s} {str(tuple(src_state[name].shape)):>10s} -> {str(tuple(value.shape)):10s} {note}")
    print(f"Saved {a.dest} ({src.num_timesteps:,} source steps, optimizer fresh)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_transfer.py
```

Expected (about 4 s):

```
ok test_action_scale_halves_the_logits
ok test_layouts_share_the_first_443_inputs
ok test_transfer_saves_a_loadable_campaign_model
ok test_widen_keeps_the_destinations_value_head_and_zeroes_new_inputs
ok test_widened_latents_match_the_source
5 tests passed
```

- [ ] **Step 5: Run every no-game suite (regressions)**

```powershell
cd F:\Github\ULTRAKILL-AI\python
Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
```

Expected: every file ends with its pass line (`N tests passed`, or `all tests passed` for `test_progress.py`) and the
loop does not throw. This covers `test_aim.py`, `test_progress.py`, `test_transfer.py` and every test file added by
Tasks 3-11.

- [ ] **Step 6: Build the real starting weights from `best.zip`**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\transfer_weights.py models\cybergrind_ppo_v2\best.zip models\campaign_ppo\transfer_init.zip
```

Expected (the source step count is `best.zip`'s, 4,679,085 as of 2026-09-16):

```
mlp_extractor.policy_net.0.weight  (512, 448) -> (512, 479) columns 0-442 copied, 443+ zero
mlp_extractor.policy_net.0.bias        (512,) -> (512,)     copied
mlp_extractor.policy_net.2.weight  (512, 512) -> (512, 512) copied
mlp_extractor.policy_net.2.bias        (512,) -> (512,)     copied
mlp_extractor.value_net.0.weight   (512, 448) -> (512, 479) columns 0-442 copied, 443+ zero
mlp_extractor.value_net.0.bias         (512,) -> (512,)     copied
mlp_extractor.value_net.2.weight   (512, 512) -> (512, 512) copied
mlp_extractor.value_net.2.bias         (512,) -> (512,)     copied
action_net.weight                   (42, 512) -> (42, 512)  copied x0.5
action_net.bias                         (42,) -> (42,)      copied x0.5
value_net.weight                     (1, 512) -> (1, 512)   fresh
value_net.bias                           (1,) -> (1,)       fresh
Saved models\campaign_ppo\transfer_init.zip (4,679,085 source steps, optimizer fresh)
```

- [ ] **Step 7: Check the saved model**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python -c "from stable_baselines3 import PPO; m = PPO.load('models/campaign_ppo/transfer_init.zip', device='cpu'); print(m.observation_space.shape, m.policy_kwargs, m.num_timesteps, m.seed)"
(Get-Item models\campaign_ppo\transfer_init.zip).Length
```

Expected: `(479,) {'net_arch': [512, 512]} 0 None`, then a size of about 4,190,000 bytes (the 12 MB source carries
Adam moments; the fresh optimizer has none).

- [ ] **Step 8: Update CLAUDE.md (Layout and Commands)**

Replace this exact line:

```text
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
```

with:

```text
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
- `python/scripts/transfer_weights.py`: campaign starting weights from a Cyber Grind checkpoint. Widens the 448 inputs to 479 (first-layer columns 0-442 copied, 443-478 zero, so Cyber Grind inputs give the same hidden features), scales the action head by `--action-scale` (default 0.5) to raise entropy, and keeps a fresh final value layer and optimizer. Output: `models/campaign_ppo/transfer_init.zip` (committed).
```

Replace this exact line:

```text
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
```

with:

```text
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/tests/test_transfer.py`: weight transfer on small PPO models: hidden features unchanged on Cyber Grind inputs, logits scaled, value head kept fresh, the saved model loads with 479 inputs (no game needed).
```

In the Commands section, the `- Tests (no game):` bullet lists the no-game test files, and Tasks 3-11 add their
own files to that list, so its full text depends on them. Its closing parenthesis does not change, and it occurs
exactly once in `CLAUDE.md`. Replace this exact text (the end of that bullet, not a whole line):

```text
(pytest is not installed; the files also work under pytest).
```

with:

```text
(pytest is not installed; the files also work under pytest). Also `python tests/test_transfer.py` (campaign weight transfer). All of them at once, from `python/` in PowerShell: `Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }`.
```

- [ ] **Step 9: Commit and push**

```powershell
cd F:\Github\ULTRAKILL-AI
git add python/scripts/transfer_weights.py python/tests/test_transfer.py python/models/campaign_ppo/transfer_init.zip CLAUDE.md
git commit -m "Add transfer_weights.py: widen the Cyber Grind policy to the campaign layout" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the commit lists exactly those 4 files and the push reaches `origin/main`.

### Task 13: `configs/campaign_0-1.yaml`, `train.py` wiring, `.gitignore`

Replaces the human-route campaign config with the self-taught one, has `train.py` give campaign runs their
exploration-archive and best-run directories and the campaign Monitor keywords, and keeps the campaign's numbered
checkpoints out of git. Depends on Task 7 (`EnvConfig` campaign fields, `_known_fields`, `CAMPAIGN_INFO_KEYS`, the
forced campaign layout) and Task 12 (`transfer_init.zip`).

**Files:**
- Modify: `python/configs/campaign_0-1.yaml` (replace the whole file)
- Modify: `python/scripts/train.py` (docstring example, `CAMPAIGN_INFO_KEYS` import, new `fill_campaign_dirs`, its call before `env_config.yaml` is written, campaign Monitor keywords)
- Modify: `.gitignore` (campaign checkpoints and smoke runs)
- Modify: `CLAUDE.md` (Layout: configs line and the new test; Commands: the no-game tests bullet and the campaign training sequence)
- Test: `python/tests/test_campaign_config.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_campaign_config.py`:

```python
"""The campaign 0-1 config and train.py's campaign wiring. No game needed:  python tests/test_campaign_config.py  (or pytest)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402

CONFIG = ROOT / "configs" / "campaign_0-1.yaml"
MODEL_DIR = Path("models") / "campaign_ppo"
RUN_DIR = Path("runs") / "campaign_ppo"


def field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_campaign_env_has_479_inputs_and_the_yaml_settings():
    env_dict, _ = train.load_config(str(CONFIG))
    cfg = EnvConfig.from_dict(env_dict)
    assert (cfg.mode, cfg.level, cfg.difficulty, cfg.unlock_all_gear) == ("campaign", "Level 0-1", 3, True), (cfg.mode, cfg.level, cfg.difficulty, cfg.unlock_all_gear)
    # The spec's speed settings. If in-game check 6 falls back to 60/4 or rendering on, change the yaml and this together.
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (30, 2, False, False)
    assert (cfg.window_width, cfg.window_height, cfg.max_steps) == (368, 207, 9000)
    assert (cfg.fresh_start_prob, cfg.stuck_seconds, cfg.stuck_repeats, cfg.cell_size) == (0.2, 45, 3, 4.0)
    assert cfg.pitch_limit_deg == 0.0  # the pitch band is a Cyber Grind aiming fix, off for the campaign
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # train.py fills these per run
    for name, value in env_dict["rewards"].items():
        assert getattr(cfg.rewards, name) == value, name
    assert cfg.rewards.style == 0.0 and cfg.rewards.time == 0.01 and cfg.rewards.level_complete == 100.0

    env = UltrakillEnv(cfg)  # builds without a game: nothing connects until the first reset
    try:
        assert env.cfg.layout.campaign
        assert env.observation_space.shape == (479,)
        assert env.cfg.rewards == cfg.rewards
    finally:
        env.close()


def test_every_campaign_setting_is_a_real_field():
    # EnvConfig.from_dict drops keys it does not know, so a misspelt setting would silently use its default.
    env_dict, _ = train.load_config(str(CONFIG))
    unknown = sorted(set(env_dict) - field_names(EnvConfig))
    assert not unknown, f"env keys EnvConfig does not know: {unknown}"
    unknown = sorted(set(env_dict["rewards"]) - field_names(RewardConfig))
    assert not unknown, f"reward keys RewardConfig does not know: {unknown}"


def test_train_section_matches_the_spec():
    _, t = train.load_config(str(CONFIG))
    assert (t["algo"], t["num_envs"], t["run_name"], t["timesteps"], t["save_every"]) == ("ppo", 5, "campaign_ppo", 20_000_000, 50_000)
    assert t["policy_kwargs"] == {"net_arch": [512, 512]}
    assert t["hyperparams"] == {
        "learning_rate": 0.0002, "n_steps": 2048, "batch_size": 512, "n_epochs": 5,
        "gamma": 0.998, "gae_lambda": 0.95, "ent_coef": 0.01, "target_kl": 0.03,
    }


def test_fill_campaign_dirs_fills_only_empty_dirs():
    cfg = EnvConfig(mode="campaign")
    filled = train.fill_campaign_dirs(cfg, MODEL_DIR, RUN_DIR)
    assert (filled.explore_dir, filled.best_runs_dir) == ("models/campaign_ppo", "runs/campaign_ppo/best_runs")
    assert (cfg.explore_dir, cfg.best_runs_dir) == ("", "")  # the config passed in is not modified

    custom = EnvConfig(mode="campaign", explore_dir="D:/archives", best_runs_dir="D:/best")
    kept = train.fill_campaign_dirs(custom, MODEL_DIR, RUN_DIR)
    assert (kept.explore_dir, kept.best_runs_dir) == ("D:/archives", "D:/best")

    half = train.fill_campaign_dirs(EnvConfig(mode="campaign", explore_dir="D:/archives"), MODEL_DIR, RUN_DIR)
    assert (half.explore_dir, half.best_runs_dir) == ("D:/archives", "runs/campaign_ppo/best_runs")


def test_fill_campaign_dirs_leaves_cybergrind_alone():
    cfg = EnvConfig(mode="cybergrind")
    same = train.fill_campaign_dirs(cfg, Path("models") / "cybergrind_ppo_v2", Path("runs") / "cybergrind_ppo_v2")
    assert same == cfg and (same.explore_dir, same.best_runs_dir) == ("", "")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_config.py
```

Expected: FAIL in the first test, the traceback ending in `AssertionError: ('campaign', 'Level 0-1', -1, False)`. The
old route-based yaml sets neither `difficulty` nor `unlock_all_gear`, and `EnvConfig.from_dict` quietly drops its
retired `checkpoint_resets` / `stuck_steps` / `route_point` / `stuck` keys.

- [ ] **Step 3: Replace `python/configs/campaign_0-1.yaml`**

Replace the whole file with:

```yaml
# Campaign level 0-1 on Violent with every weapon unlocked in memory, learned without human routes: the rewards
# come from checkpoints, arena clears, door unlocks, exploration and the NavMesh distance to the exit.
# From python/ (on a one-display PC add --monitor 1 to games.py and dashboard.py):
#   python scripts/transfer_weights.py models/cybergrind_ppo_v2/best.zip models/campaign_ppo/transfer_init.zip
#   python scripts/games.py launch --count 5
#   python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip
#   python scripts/poll_status.py --run campaign_ppo
#   python scripts/keep_best.py --run campaign_ppo --metric campaign
#   python scripts/dashboard.py --run campaign_ppo
#   python scripts/games.py stop
# To continue a stopped run, resume from models/campaign_ppo/best.zip once keep_best.py has written it, else
# from the newest ckpt_*_steps.zip (latest.zip is only current after a graceful Ctrl+C).
env:
  mode: campaign
  level: "Level 0-1"
  difficulty: 3
  unlock_all_gear: true
  fixed_fps: 30
  frameskip: 2
  render: false
  soft_death: false
  window_width: 368
  window_height: 207
  max_steps: 9000
  fresh_start_prob: 0.2
  stuck_seconds: 45
  stuck_repeats: 3
  cell_size: 4.0
  rewards: {time: 0.01, level_complete: 100.0, checkpoint: 10.0, arena_clear: 10.0, door_unlock: 3.0,
            novelty: 0.5, path: 0.1, kill: 0.5, damage_dealt: 0.5, damage_taken: 0.01, death: 5.0, style: 0.0}
train:
  algo: ppo
  num_envs: 5
  run_name: campaign_ppo
  timesteps: 20000000
  save_every: 50000
  policy_kwargs: {net_arch: [512, 512]}
  hyperparams: {learning_rate: 0.0002, n_steps: 2048, batch_size: 512, n_epochs: 5, gamma: 0.998,
                gae_lambda: 0.95, ent_coef: 0.01, target_kl: 0.03}
```

- [ ] **Step 4: Run the test to verify the next failure**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_config.py
```

Expected: `ok test_campaign_env_has_479_inputs_and_the_yaml_settings` and `ok test_every_campaign_setting_is_a_real_field`,
then FAIL with the traceback ending in `AttributeError: module 'train' has no attribute 'fill_campaign_dirs'`.

- [ ] **Step 5: Wire the campaign into `python/scripts/train.py`**

Five exact replacements.

(a) Docstring example. Replace:

```python
    python scripts/train.py --config configs/campaign_0-1.yaml --algo rppo
```

with:

```python
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip
```

(b) Import. Replace:

```python
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
```

with:

```python
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
```

(c) New function after `load_config`. Replace:

```python
    return data.get("env", {}), data.get("train", {})
```

with:

```python
    return data.get("env", {}), data.get("train", {})


def fill_campaign_dirs(cfg: EnvConfig, model_dir: Path, run_dir: Path) -> EnvConfig:
    """Campaign runs keep their exploration archives beside the checkpoints and their best runs beside the logs.

    Only empty settings are filled, so a config can still point either one elsewhere. Cyber Grind is returned as is.
    """
    if cfg.mode != "campaign":
        return cfg
    return replace(
        cfg,
        explore_dir=cfg.explore_dir or model_dir.as_posix(),
        best_runs_dir=cfg.best_runs_dir or (run_dir / "best_runs").as_posix(),
    )
```

(d) Fill the directories before `env_config.yaml` is written, so the saved file records the directories the games
actually used (`eval.py` blanks both for its own runs, and a resumed run fills them again from the config).
Replace:

```python
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")
```

with:

```python
    model_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = fill_campaign_dirs(env_cfg, model_dir, Path("runs") / run_name)
    if env_cfg.best_runs_dir:
        Path(env_cfg.best_runs_dir).mkdir(parents=True, exist_ok=True)
    (model_dir / "env_config.yaml").write_text(yaml.safe_dump(env_cfg.to_dict()), encoding="utf-8")
```

(e) Campaign Monitor keywords. Replace:

```python
    info_keywords = ("kills", "wave", "style", "deaths", "firing_frac", "on_target_frac") if env_cfg.mode == "cybergrind" else ("kills", "style", "route_progress")
```

with:

```python
    info_keywords = ("kills", "wave", "style", "deaths", "firing_frac", "on_target_frac") if env_cfg.mode == "cybergrind" else CAMPAIGN_INFO_KEYS
```

- [ ] **Step 6: Run the test to verify it passes**

```powershell
cd F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_config.py
```

Expected:

```
ok test_campaign_env_has_479_inputs_and_the_yaml_settings
ok test_every_campaign_setting_is_a_real_field
ok test_fill_campaign_dirs_fills_only_empty_dirs
ok test_fill_campaign_dirs_leaves_cybergrind_alone
ok test_train_section_matches_the_spec
5 tests passed
```

- [ ] **Step 7: Run every no-game suite (regressions)**

```powershell
cd F:\Github\ULTRAKILL-AI\python
Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
```

Expected: every file ends with its pass line (`N tests passed`, or `all tests passed` for `test_progress.py`) and the
loop does not throw.

- [ ] **Step 8: Ignore the campaign's numbered checkpoints and smoke runs**

In `.gitignore`, replace this exact line:

```text
python/models/cybergrind_ppo_v2/*.bad
```

with:

```text
python/models/cybergrind_ppo_v2/*.bad
# Campaign: numbered checkpoints stay local, as for v2. transfer_init.zip (the widened Cyber Grind weights),
# best.zip + best.json, latest.zip, env_config.yaml and the explore_*.npz exploration archives are committed.
python/models/campaign_ppo/ckpt_*.zip
# Throwaway in-game smoke runs
python/models/campaign_smoke/
```

- [ ] **Step 9: Check the ignore rules**

```powershell
cd F:\Github\ULTRAKILL-AI
git check-ignore -v python/models/campaign_ppo/ckpt_50000_steps.zip python/models/campaign_smoke/latest.zip python/models/campaign_ppo/transfer_init.zip python/models/campaign_ppo/explore_Level_0-1_47800.npz
```

Expected: exactly two lines, `.gitignore:37:python/models/campaign_ppo/ckpt_*.zip	python/models/campaign_ppo/ckpt_50000_steps.zip`
and `.gitignore:39:python/models/campaign_smoke/	python/models/campaign_smoke/latest.zip`; nothing for
`transfer_init.zip` or the `explore_*.npz` archive.

- [ ] **Step 10: Update CLAUDE.md (Layout and Commands)**

Replace this exact line:

```text
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml`.
```

with:

```text
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml` (campaign pilot: Violent, all gear unlocked in memory, run `campaign_ppo`; its header lists the run commands).
```

Replace this exact line (added by Task 12):

```text
- `python/tests/test_transfer.py`: weight transfer on small PPO models: hidden features unchanged on Cyber Grind inputs, logits scaled, value head kept fresh, the saved model loads with 479 inputs (no game needed).
```

with:

```text
- `python/tests/test_transfer.py`: weight transfer on small PPO models: hidden features unchanged on Cyber Grind inputs, logits scaled, value head kept fresh, the saved model loads with 479 inputs (no game needed).
- `python/tests/test_campaign_config.py`: `configs/campaign_0-1.yaml` builds a 479-input env with its reward weights, every key is a real config field, and `train.fill_campaign_dirs` (no game needed).
```

In the Commands section's `- Tests (no game):` bullet, replace this exact text (added by Task 12, part of a line):

```text
Also `python tests/test_transfer.py` (campaign weight transfer).
```

with:

```text
Also `python tests/test_transfer.py` (campaign weight transfer) and `python tests/test_campaign_config.py` (the campaign config and `train.py` wiring).
```

Replace this exact line:

```text
  - `python scripts/games.py status` is safe during training (it reads netstat, it does not connect).
```

with:

```text
  - `python scripts/games.py status` is safe during training (it reads netstat, it does not connect).
- Campaign training (0-1, `configs/campaign_0-1.yaml`; it uses the same five games, so Cyber Grind stays paused):
  1. `python scripts/transfer_weights.py models/cybergrind_ppo_v2/best.zip models/campaign_ppo/transfer_init.zip` (done once and committed; rerun only to change `--action-scale`)
  2. `python scripts/games.py launch --count 5` (add `--monitor 1` on the one-display PC)
  3. `python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip` (a new run: the file is at 0 steps). To continue a stopped run, resume from `models/campaign_ppo/best.zip` once `keep_best.py` has written it, else from the newest `ckpt_*_steps.zip`.
  4. Alongside it: `python scripts/poll_status.py --run campaign_ppo`, `python scripts/keep_best.py --run campaign_ppo --metric campaign` and `python scripts/dashboard.py --run campaign_ppo` (`--monitor 1` on the one-display PC).
  5. `python scripts/games.py stop`
  - `train.py` fills `explore_dir` (the per-game exploration archives, `models/campaign_ppo/explore_*.npz`) and `best_runs_dir` (`runs/campaign_ppo/best_runs/`) when the config leaves them empty, before writing `env_config.yaml`.
  - Commit `transfer_init.zip`, `best.zip` + `best.json`, `latest.zip`, `env_config.yaml` and the `explore_*.npz` archives; the numbered `ckpt_*` files and `models/campaign_smoke/` are gitignored.
  - If the first update's entropy (the dashboard's PPO panel shows it negated, as `entropy`) is outside 6-10 nats: stop, rerun step 1 once with another `--action-scale` and commit the new `transfer_init.zip`, delete `runs/campaign_ppo/` (dashboard history), the `runs/campaign_ppo_*` TensorBoard folders (a `--resume` run keeps writing into the newest one) and the run's `ckpt_*` and `explore_*.npz` files so the restart does not inherit them, then start step 3 again.
```

- [ ] **Step 11: Commit and push**

```powershell
cd F:\Github\ULTRAKILL-AI
git add python/configs/campaign_0-1.yaml python/scripts/train.py python/tests/test_campaign_config.py .gitignore CLAUDE.md
git commit -m "Rewrite the 0-1 campaign config and wire campaign runs into train.py" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

Expected: the commit lists exactly those 5 files and the push reaches `origin/main`.

### Task 14: In-game verification (`campaign_check.py`, smoke run)

**Files:**
- Create: `python/scripts/campaign_check.py`
- Create: `python/tests/test_campaign_check.py`
- Modify: `CLAUDE.md` (Layout, Commands, Gotchas, Status)
- Modify: `python/configs/campaign_0-1.yaml` and `python/tests/test_campaign_config.py` (only if Step 11 (d) finds that the triggers need 60 fps / frameskip 4 or rendering)
- Test: `python/tests/test_campaign_check.py`

`campaign_check.py` runs the spec's in-game checks through `UltrakillEnv` on one game, with the campaign config's env settings, fresh level loads only, and no files written. The test drives the same five checks against `FakeGame`, a stand-in for `BridgeClient` shaped like the mod's v0.5.0 observations, so the script is proven before a game is involved. Three small additions beyond the contract wording, all to avoid false results in the game: check 3 tries up to the 3 nearest pending checkpoints (one may sit in a room that is still switched off); check 4 requires the kill reply to show a dead player and the death count to rise by exactly one (a fall onto a switched-off checkpoint in check 3 can already have cost a death, and a kill healed by soft death would otherwise still pass, because the env counts and respawns soft deaths too); and check 5 also confirms the official time has not moved one game second after the finish.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_campaign_check.py`:

```python
"""campaign_check.py's five in-game checks, run against a fake campaign level. No game needed:  python tests/test_campaign_check.py  (or pytest)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.env import UltrakillEnv  # noqa: E402

SPAWN = (0.0, 1.0, 0.0)
CHECKPOINTS = ((0.0, 0.0, 20.0), (0.0, 0.0, 40.0))
EXIT = (0.0, 0.0, 80.0)
TRIGGER_RADIUS = 2.0


def load_script():
    spec = importlib.util.spec_from_file_location("campaign_check", ROOT / "scripts" / "campaign_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGame:
    """A campaign level as the mod reports it: checkpoints at z 20 and z 40 along +z, and an exit pit at z 80.

    Stands in for BridgeClient. Triggers fire on the step after the player comes within 2 m, as Unity trigger
    callbacks do on the next physics update. The timer stops when the exit fires. `apply_difficulty`, `triggers`
    and `exit` switch single features off for the failure cases. `void_checkpoint` is a checkpoint whose room is
    still switched off: arriving on it is a fall to death, not an activation. `heal_lethal` is soft death healing
    the kill (a living player in the reply, counted in `soft_deaths`).
    """

    def __init__(self, *, scene="Level 0-1", slot_counts=(0, 0, 0, 0, 0, 0), apply_difficulty=True, triggers=True, exit=EXIT,
                 void_checkpoint=None, heal_lethal=False):
        self.scene = scene
        self.slot_counts = list(slot_counts)
        self.apply_difficulty = apply_difficulty
        self.triggers = triggers
        self.exit = exit
        self.void_checkpoint = void_checkpoint
        self.heal_lethal = heal_lethal
        self.soft_deaths = 0  # the mod's counter lives across level loads
        self.settings: dict = {}
        self.resets: list[bool] = []
        self._load()

    def _load(self):
        self.pos = list(SPAWN)
        self.dead = False
        self.activated: set[int] = set()
        self.current: int | None = None
        self.seconds = 0.0
        self.level_over = False
        self.restarts = 0
        self.steps = 0

    def connect(self):
        return {"type": "hello", "protocol": 1, "mod_version": "0.5.0", "scene": "Main Menu"}

    def configure(self, **settings):
        self.settings.update(settings)

    def close(self):
        pass

    def reset(self, scene=None, checkpoint=False):
        self.resets.append(checkpoint)
        if checkpoint and self.current is not None:
            x, y, z = CHECKPOINTS[self.current]
            self.pos = [x, y + 1.25, z]  # CheckPoint respawns the player 1.25 m above its origin
            self.dead = False
            self.restarts += 1
        else:
            self._load()
        return self._obs("reset")

    def step(self, action):
        self.steps += 1
        if not self.level_over:
            self.seconds += 2 / 30
        if self.triggers and not self.dead:
            for i, cp in enumerate(CHECKPOINTS):
                if i not in self.activated and math.dist(self.pos, cp) <= TRIGGER_RADIUS:
                    if i == self.void_checkpoint:
                        self.dead = True
                        break
                    self.activated.add(i)
                    self.current = i
            if self.exit is not None and math.dist(self.pos, self.exit) <= TRIGGER_RADIUS:
                self.level_over = True
        return self._obs()

    def teleport(self, pos):
        self.pos = [float(v) for v in pos]
        return self._obs("teleport")

    def kill(self):
        if self.heal_lethal:
            self.soft_deaths += 1
        else:
            self.dead = True
        return self._obs("kill")

    def get_obs(self):
        return self._obs()

    def _obs(self, event=None):
        difficulty = self.settings.get("difficulty", -1) if self.apply_difficulty else -1
        player = {
            "pos": list(self.pos), "vel": [0.0, 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0], "forward": [0.0, 0.0, 1.0],
            "yaw": 0.0, "pitch": 0.0, "hp": 0 if self.dead else 100, "anti_hp": 0.0, "stamina": 300.0,
            "grounded": True, "sliding": False, "dead": self.dead, "activated": not self.level_over,
            "level_over": self.level_over, "weapon_slot": -1, "weapon_variation": -1, "soft_deaths": self.soft_deaths,
            "soft_death_instakill": False, "slot_counts": list(self.slot_counts),
        }
        campaign = {
            "mission": 1, "difficulty": difficulty if difficulty >= 0 else 2, "seconds": self.seconds,
            "timer_running": not self.level_over, "level_started": True, "level_over": self.level_over,
            "restarts": self.restarts, "input_locked": self.level_over,
            "exit": {"pos": list(self.exit), "active": True} if self.exit is not None else None,
            "checkpoints": [
                {"id": f"{x:.0f},{y:.0f},{z:.0f}", "pos": [x, y, z], "activated": i in self.activated, "current": i == self.current}
                for i, (x, y, z) in enumerate(CHECKPOINTS)
            ],
            "path": {"status": "complete", "length": 80.0 - self.pos[2], "next_corner": list(self.exit)} if self.exit else {"status": "none"},
            "locked_doors": [], "arena_enemies_alive": 0, "cleared_arenas": [], "unlocked_doors": [],
            "ranks": {"time": [120, 90, 60, 30], "kills": [0, 1, 2, 3], "style": [0, 100, 200, 300]},
        }
        obs = {
            "type": "obs", "step": self.steps, "scene": self.scene, "ready": not self.dead, "player": player,
            "enemies": [], "rays": [50.0] * 16, "ground_rays": [0.0] * 8,
            "stats": {"kills": 0, "style": 0, "seconds": self.seconds, "restarts": self.restarts, "level_complete": self.level_over},
            "campaign": campaign,
        }
        if event:
            obs["event"] = event
        return obs


def run(game: FakeGame, *args: str):
    script = load_script()
    env = UltrakillEnv(script.build_config(script.parse_args(["--level", game.scene, *args])))
    env.client = game
    try:
        results = script.run_checks(env)
    finally:
        env.close()
    return script, results


def statuses(results):
    return [status for _, _, status, _ in results]


def test_every_check_passes_on_a_working_level():
    game = FakeGame()
    script, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "PASS"], results
    assert game.settings["difficulty"] == 3 and game.settings["unlock_all_gear"] is True
    assert game.resets == [False, True]  # one fresh load, then the respawn after the kill
    assert "revolver pickup" in results[1][3]
    assert script.exit_code(results) == 0


def test_the_arsenal_passes_when_every_weapon_slot_is_filled():
    _, results = run(FakeGame(scene="Level 1-1", slot_counts=(4, 3, 3, 3, 3, 0)))
    assert statuses(results) == ["PASS", "PASS", "PASS", "PASS", "PASS"], results


def test_a_difficulty_the_game_ignores_fails_the_level_load():
    script, results = run(FakeGame(apply_difficulty=False))
    assert statuses(results)[0] == "FAIL"
    assert "expected 3" in results[0][3]
    assert script.exit_code(results) == 1


def test_triggers_that_never_fire_fail_the_checkpoint_and_exit_and_skip_the_death():
    game = FakeGame(triggers=False)
    script, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "FAIL", "SKIP", "FAIL"], results
    assert results[2][3].count("did not activate") == len(CHECKPOINTS)  # every pending checkpoint was tried
    assert "did not fire" in results[4][3]
    assert script.exit_code(results) == 1


def test_a_null_exit_fails_the_exit_check():
    _, results = run(FakeGame(exit=None))
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "FAIL"], results
    assert results[4][3].startswith("exit null")


def test_a_fall_during_the_checkpoint_check_does_not_fail_the_death_check():
    game = FakeGame(void_checkpoint=0)
    _, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "PASS"], results
    assert game.resets == [False, True, True]  # the fresh load, the reload after the fall, the respawn after the kill
    assert "0,0,20 did not activate" in results[2][3] and "0,0,40 activated" in results[2][3]
    assert "deaths 1 -> 2" in results[3][3]


def test_a_kill_that_soft_death_heals_fails_the_death_check():
    script, results = run(FakeGame(heal_lethal=True))
    assert statuses(results) == ["PASS", "SKIP", "PASS", "FAIL", "PASS"], results
    assert "kill reply dead=False, deaths 0 -> 1" in results[3][3]  # the env still counted and respawned it
    assert script.exit_code(results) == 1


def test_command_line_overrides_and_config_defaults():
    script = load_script()
    cfg = script.build_config(script.parse_args(["--level", "Level 1-1", "--port", "47801", "--fixed-fps", "60", "--frameskip", "4", "--render"]))
    assert (cfg.mode, cfg.level, cfg.port, cfg.fixed_fps, cfg.frameskip, cfg.render) == ("campaign", "Level 1-1", 47801, 60.0, 4, True)
    assert (cfg.fresh_start_prob, cfg.explore_dir, cfg.best_runs_dir) == (1.0, "", "")
    assert cfg.difficulty == 3 and cfg.unlock_all_gear is True

    env = yaml.safe_load((ROOT / "configs" / "campaign_0-1.yaml").read_text(encoding="utf-8"))["env"]
    default = script.build_config(script.parse_args([]))
    assert (default.level, default.port) == ("Level 0-1", 47800)
    assert (default.fixed_fps, default.frameskip, default.render) == (env["fixed_fps"], env["frameskip"], env["render"])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run it and watch it fail**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_check.py
```

Expected: a traceback ending in
`FileNotFoundError: [Errno 2] No such file or directory: 'F:\\Github\\ULTRAKILL-AI\\python\\scripts\\campaign_check.py'`

- [ ] **Step 3: Write `campaign_check.py`**

Create `python/scripts/campaign_check.py`:

```python
"""In-game checks for campaign support, run on one game before any campaign training.

    python scripts/games.py launch --count 1 --monitor 1
    python scripts/campaign_check.py                                 # Level 0-1 at the training speed settings
    python scripts/campaign_check.py --level "Level 1-1"             # the full arsenal
    python scripts/campaign_check.py --fixed-fps 60 --frameskip 4    # when a trigger did not fire at 30 fps / frameskip 2
    python scripts/campaign_check.py --render                        # when it did not fire at 60 / 4 either
    python scripts/games.py stop

Uses the env settings of configs/campaign_0-1.yaml, except that every reset is a fresh level load and nothing is
written (no exploration archive, no best runs). Each check prints PASS, FAIL or SKIP, then a one-line summary; the
exit code is 1 when any check failed. The bridge is single-client, so never run this against a port a trainer uses.

  1 level load     a fresh reset has a campaign block with difficulty 3 (Violent); prints exit, checkpoints, path
  2 arsenal        player.slot_counts has a weapon in each of slots 1-5 (0-1 has none before the revolver pickup)
  3 checkpoint     teleporting onto the nearest pending checkpoint activates it within 30 decisions
  4 death respawn  a real (not healed) death respawns at that checkpoint inside the same episode
  5 exit           teleporting into the exit ends the episode with level_complete, and the official time stops
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.spaces import noop_action  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "campaign_0-1.yaml"
VIOLENT = 3
CHECKPOINT_TRIES = 3  # pending checkpoints to try, nearest first: one may sit in a room that is still switched off
CHECKPOINT_STEPS = 30  # decisions to wait on each (2 game seconds at 15 decisions/s)
EXIT_STEPS = 60  # decisions to wait for the exit trigger (4 game seconds, room to drop into the pit)
TIMER_STEPS = 15  # decisions after the finish during which the official time must not move (1 game second)
RESPAWN_RADIUS = 10.0  # metres from the checkpoint a respawn must land within
NAMES = ("level load", "arsenal", "checkpoint", "death respawn", "exit")
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

Result = tuple[int, str, str, str]  # (number, name, status, detail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-game checks for campaign support (one game).")
    parser.add_argument("--level", default="Level 0-1", help="campaign scene name")
    parser.add_argument("--port", type=int, default=47800)
    parser.add_argument("--fixed-fps", type=float, help="override the config's fixed_fps (60 with --frameskip 4 keeps 15 decisions/s)")
    parser.add_argument("--frameskip", type=int, help="override the config's frameskip")
    parser.add_argument("--render", action="store_true", help="turn the cameras on, for triggers that may depend on rendering")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> EnvConfig:
    """configs/campaign_0-1.yaml's env settings, with fresh level loads only and no files written."""
    env = dict((yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("env", {}))
    env.update(level=args.level, port=args.port, fresh_start_prob=1.0, explore_dir="", best_runs_dir="")
    if args.fixed_fps:
        env["fixed_fps"] = args.fixed_fps
    if args.frameskip:
        env["frameskip"] = args.frameskip
    if args.render:
        env["render"] = True
    return EnvConfig.from_dict(env)


def block(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("campaign") or {}


def vec(v) -> str:
    return "(" + ", ".join(f"{c:.1f}" for c in v) + ")"


def secs(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def is_activated(raw: dict[str, Any], checkpoint_id: str) -> bool:
    return any(cp["id"] == checkpoint_id and (cp["activated"] or cp["current"]) for cp in block(raw).get("checkpoints", []))


def pending_checkpoints(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Checkpoints that are neither activated nor current, nearest to the player first."""
    pos = raw["player"]["pos"]
    pending = [cp for cp in block(raw).get("checkpoints", []) if not cp["activated"] and not cp["current"]]
    return sorted(pending, key=lambda cp: math.dist(cp["pos"], pos))


def run_noops(env: UltrakillEnv, steps: int, until: Callable[[dict[str, Any]], bool]) -> tuple[bool, dict[str, Any], bool]:
    """Steps no-op decisions until `until(raw)` holds or the episode ends: (condition met, last info, episode ended)."""
    info: dict[str, Any] = {}
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(noop_action())
        if until(env._raw):  # the raw snapshot the env just received from the mod
            return True, info, terminated or truncated
        if terminated or truncated:
            return False, info, True
    return False, info, False


def check_level_load(raw: dict[str, Any]) -> tuple[str, str]:
    player, cb = raw.get("player"), raw.get("campaign")
    if not player or not cb:
        missing = "player" if not player else "campaign block"
        return FAIL, f"scene {raw.get('scene')!r} has no {missing} after a fresh reset (mod older than v0.5.0, or not a campaign scene)"
    ext, path = cb.get("exit"), cb.get("path") or {}
    exit_text = f"exit {vec(ext['pos'])} active={ext['active']}" if ext else "exit null"
    path_text = f"path {path.get('status')}" + (f" {path['length']:.1f} m" if path.get("length") is not None else "")
    checkpoints = sorted(cb.get("checkpoints", []), key=lambda cp: math.dist(cp["pos"], player["pos"]))
    detail = (f"mission {cb.get('mission')}, difficulty {cb.get('difficulty')}, {exit_text}, {len(checkpoints)} checkpoints, "
              f"{path_text}, {len(cb.get('locked_doors', []))} locked doors listed, input_locked={cb.get('input_locked')}")
    for cp in checkpoints:
        detail += (f"\n      checkpoint {cp['id']} {math.dist(cp['pos'], player['pos']):.1f} m away, "
                   f"activated={cp['activated']} current={cp['current']}")
    if cb.get("difficulty") != VIOLENT:
        return FAIL, detail + f"\n      difficulty reads {cb.get('difficulty')}, expected {VIOLENT} (Violent)"
    return PASS, detail


def check_arsenal(raw: dict[str, Any], level: str) -> tuple[str, str]:
    counts = (raw.get("player") or {}).get("slot_counts")
    if counts is None:
        return FAIL, "player.slot_counts is missing (mod older than v0.5.0)"
    if len(counts) >= 5 and all(c > 0 for c in counts[:5]):
        return PASS, f"slot_counts {counts}"
    if level == "Level 0-1":
        return SKIP, f"slot_counts {counts}: 0-1 has no weapons until the revolver pickup; run with --level \"Level 1-1\" to see the arsenal"
    return SKIP, f"slot_counts {counts}: not every weapon slot is filled, so unlock_all_gear did not reach GunSetter"


def check_checkpoint(env: UltrakillEnv) -> tuple[str, str, dict[str, Any] | None, bool]:
    """(status, detail, the checkpoint that activated, episode ended)."""
    raw = env._raw
    if not raw.get("player") or not raw.get("campaign"):
        return SKIP, "needs a player and the campaign block (check 1)", None, False
    candidates = pending_checkpoints(raw)[:CHECKPOINT_TRIES]
    if not candidates:
        return SKIP, "no pending checkpoint in this level", None, False
    notes = []
    for cp in candidates:
        x, y, z = cp["pos"]
        env.client.teleport([x, y + 1.0, z])
        reached, info, ended = run_noops(env, CHECKPOINT_STEPS, lambda r, cid=cp["id"]: is_activated(r, cid))
        if reached:
            notes.append(f"checkpoint {cp['id']} activated (checkpoints_level {info.get('checkpoints_level')})")
            return PASS, "; ".join(notes), cp, ended
        notes.append(f"checkpoint {cp['id']} did not activate in {CHECKPOINT_STEPS} decisions "
                     f"(arena_enemies_alive {block(env._raw).get('arena_enemies_alive')})")
        if ended:
            notes.append(f"episode ended: {info.get('end_reason')}")
            return FAIL, "; ".join(notes), None, True
    return FAIL, "; ".join(notes), None, False


def check_death(env: UltrakillEnv, checkpoint: dict[str, Any] | None) -> tuple[str, str, bool]:
    """(status, detail, episode ended)."""
    if checkpoint is None:
        return SKIP, "needs the checkpoint activated in check 3", False
    # Check 3 can already have cost a death (a checkpoint in a room that is still switched off has no floor under
    # it), so the kill must add exactly one to the episode's count rather than make it 1.
    deaths_before = env._deaths
    killed = env.client.kill()
    # A real death only: soft death heals the lethal hit, and the env would still count and respawn that one.
    killed_dead = (killed.get("player") or {}).get("dead")
    _, _, terminated, truncated, info = env.step(noop_action())
    ended = terminated or truncated
    player = env._raw.get("player")
    alive = player is not None and not player["dead"]
    dist = math.dist(player["pos"], checkpoint["pos"]) if player else math.inf
    detail = (f"kill reply dead={killed_dead}, deaths {deaths_before} -> {info.get('deaths')}, "
              f"episode ended={ended}" + (f" ({info.get('end_reason')})" if ended else "")
              + f", player {'alive' if alive else 'missing or dead'} {dist:.1f} m from checkpoint {checkpoint['id']}")
    ok = (killed_dead is True and info.get("deaths") == deaths_before + 1 and not ended and alive
          and dist <= RESPAWN_RADIUS)
    return (PASS if ok else FAIL), detail, ended


def check_exit(env: UltrakillEnv) -> tuple[str, str]:
    raw = env._raw
    if not raw.get("player") or not raw.get("campaign"):
        return SKIP, "needs a player and the campaign block (check 1)"
    ext = block(raw).get("exit")
    if not ext:
        return FAIL, "exit null: no real FinalPit was found (decoy or template filter, or the pit is not in the scene)"
    env.client.teleport(ext["pos"])
    _, info, ended = run_noops(env, EXIT_STEPS, lambda r: False)
    if not ended:
        return FAIL, f"exit {vec(ext['pos'])} active={ext['active']}: the trigger did not fire in {EXIT_STEPS} decisions"
    reason, level_seconds = info.get("end_reason"), info.get("level_seconds")
    final = later = block(env._raw).get("seconds")
    for _ in range(TIMER_STEPS):
        later = block(env.client.step({})).get("seconds")
    detail = (f"end_reason {reason}, level_seconds {secs(level_seconds)}, block seconds {secs(final)} at the finish and "
              f"{secs(later)} one game second later, rank {info.get('rank')}")
    ok = (reason == "level_complete" and None not in (level_seconds, final, later)
          and abs(level_seconds - final) < 1e-3 and abs(later - final) < 1e-3)
    return (PASS if ok else FAIL), detail


def run_checks(env: UltrakillEnv) -> list[Result]:
    results: list[Result] = []

    def record(number: int, status: str, detail: str) -> None:
        results.append((number, NAMES[number - 1], status, detail))
        print(f"[{status}] {number} {NAMES[number - 1]}: {detail}", flush=True)

    env.reset()
    record(1, *check_level_load(env._raw))
    record(2, *check_arsenal(env._raw, env.cfg.level))
    status, detail, checkpoint, ended = check_checkpoint(env)
    record(3, status, detail)
    if ended:
        record(4, SKIP, "the episode ended during check 3")
    else:
        status, detail, ended = check_death(env, checkpoint)
        record(4, status, detail)
    if ended:
        env.reset()  # the exit check needs a running episode; this is another fresh level load
    record(5, *check_exit(env))
    return results


def exit_code(results: list[Result]) -> int:
    return 1 if any(status == FAIL for _, _, status, _ in results) else 0


def main() -> int:
    cfg = build_config(parse_args())
    print(f"{cfg.level} on port {cfg.port}: fixed_fps {cfg.fixed_fps:g}, frameskip {cfg.frameskip}, render {cfg.render}, "
          f"difficulty {cfg.difficulty}, unlock_all_gear {cfg.unlock_all_gear}", flush=True)
    env = UltrakillEnv(cfg)
    try:
        results = run_checks(env)
    finally:
        env.close()
    print("summary: " + " | ".join(f"{number} {name} {status}" for number, name, status, _ in results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test and watch it pass**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python tests\test_campaign_check.py
```

Expected: the checks' own `[PASS]` / `[SKIP]` / `[FAIL]` lines (the FAILs come from the failure-case tests), eight `ok test_...` lines, and last `8 tests passed`.

- [ ] **Step 5: Run every no-game suite**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
foreach ($t in Get-ChildItem tests\test_*.py) { .venv\Scripts\python $t.FullName | Select-Object -Last 1; if ($LASTEXITCODE -ne 0) { throw "$($t.Name) failed" } }
```

Expected: one pass line per file (`N tests passed`, or `all tests passed` for `test_progress.py`) for `test_aim.py`, `test_progress.py`, `test_campaign_check.py` and every test file added by Tasks 3-13, and no `failed` exception.

- [ ] **Step 6: Document the script in CLAUDE.md**

In `CLAUDE.md`, find this Layout line:

```
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
```

and replace it with:

```
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
- `python/scripts/campaign_check.py`: in-game campaign checks on one game (difficulty, arsenal, checkpoint trigger, death respawn, exit and official time). `python/tests/test_campaign_check.py` runs the same five checks against a fake level (no game needed).
```

Then find this Commands line:

```
- TensorBoard: `tensorboard --logdir runs`.
```

and replace it with:

```
- TensorBoard: `tensorboard --logdir runs`.
- Campaign in-game check (one game, nothing else connected to its port): `python scripts/games.py launch --count 1 --monitor 1`, then `python scripts/campaign_check.py` (Level 0-1) and `python scripts/campaign_check.py --level "Level 1-1"` (full arsenal), then `python scripts/games.py stop`. Prints PASS/FAIL/SKIP per check and a `summary:` line, exits 1 on any FAIL. `--fixed-fps 60 --frameskip 4` and `--render` repeat the trigger checks at other speed settings. Rerun after game updates and after mod changes to the campaign block; `python tests/test_campaign_check.py` tests the script without the game.
```

- [ ] **Step 7: Commit and push the script**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/scripts/campaign_check.py python/tests/test_campaign_check.py CLAUDE.md
git commit -m "Add the in-game campaign check script" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 8: Close the games and install the current mod build**

The game must be closed, or the DLL is locked.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py stop
Set-Location F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge
& "C:\Program Files\dotnet\dotnet.exe" build -c Release
Set-Location F:\Github\ULTRAKILL-AI\python
```

Expected: `Build succeeded` and `Copied UltrakillAIBridge.dll to C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\plugins\UltrakillAIBridge`. An error that the DLL is in use means a copy of ULTRAKILL is still open: close it and build again.

- [ ] **Step 9: Launch one training game and confirm the mod version**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py launch --count 1 --monitor 1
.venv\Scripts\python scripts\bridge_test.py
```

Expected: `All 1 instances ready on ports 47800-47800`, then a line starting `Connected: mod 0.5.0, protocol 1`, then a `scene=... (no player)` summary for the menu scene the game is in. Any other mod version means Step 8 did not install the build: repeat Step 8 with every ULTRAKILL window closed.

- [ ] **Step 10: Run the checks on Level 0-1 and Level 1-1**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\campaign_check.py
$LASTEXITCODE
.venv\Scripts\python scripts\campaign_check.py --level "Level 1-1"
$LASTEXITCODE
```

Expected shape for Level 0-1 (the numbers below are illustrative: positions, distances, counts and times come from the level):

```
Level 0-1 on port 47800: fixed_fps 30, frameskip 2, render False, difficulty 3, unlock_all_gear True
[PASS] 1 level load: mission 1, difficulty 3, exit (-4.0, -32.0, 481.5) active=True, 3 checkpoints, path complete 212.4 m, 0 locked doors listed, input_locked=False
      checkpoint 12,3,-40 38.2 m away, activated=False current=False
[SKIP] 2 arsenal: slot_counts [0, 0, 0, 0, 0, 0]: 0-1 has no weapons until the revolver pickup; run with --level "Level 1-1" to see the arsenal
[PASS] 3 checkpoint: checkpoint 12,3,-40 activated (checkpoints_level 1)
[PASS] 4 death respawn: kill reply dead=True, deaths 0 -> 1, episode ended=False, player alive 1.3 m from checkpoint 12,3,-40
[PASS] 5 exit: end_reason level_complete, level_seconds 14.733, block seconds 14.733 at the finish and 14.733 one game second later, rank D
summary: 1 level load PASS | 2 arsenal SKIP | 3 checkpoint PASS | 4 death respawn PASS | 5 exit PASS
0
```

Check 1 prints one indented `checkpoint <id> ... away, activated=... current=...` line per checkpoint in the
level; only the first is shown above. (`slot_counts` may also read `[]` in 0-1 when the level has no gun
controller yet.) Expected for Level 1-1: `mission 6`, check 2 `[PASS] 2 arsenal: slot_counts [...]` with the first
five counts above 0, all five checks PASS, and exit code `0`.

Then exercise the random agent's campaign path (the spec's in-game check 5 in its one-game form; Step 13 measures
the five-game throughput figure). It uses `EnvConfig`'s own defaults, not the campaign config, so it runs at
60 fps / frameskip 4:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\random_agent.py --mode campaign --level "Level 0-1" --episodes 1 --max-steps 900
```

Expected: no traceback, and one line of the shape

```
episode 0: steps=675 reward=-1.23 kills=0 completed=0 checkpoints_level=0 cells_new=12 end=stuck (35 steps/s)
```

(`end` is `stuck` after 45 game seconds without progress, or `max_steps` at 900 decisions; `completed` is 0
unless random actions happened to reach the exit). A `KeyError: 'completed'` means Task 8's campaign branch of
`random_agent.py` is missing.

Keep both `summary:` lines for Step 15. If both runs match, skip Step 11.

- [ ] **Step 11: Resolve any FAIL (only when Step 10 did not match)**

Work through the matching case, then rerun Step 10 for both levels at the default settings. If a FAIL remains after its case's actions (other than a check 5 FAIL that case (e) verified by hand), stop: do not run Steps 12-17 or Task 15, append this line at the end of `CLAUDE.md`, commit and push it with `CLAUDE.md` only, and debug with superpowers:systematic-debugging.

```
- **Campaign in-game verification blocked (2026-09-16):** `campaign_check.py` still fails after the plan's fallbacks (see its output); the 0-1 pilot waits for the fix.
```

(a) `[FAIL] 1 level load: scene 'Level 0-1' has no campaign block after a fresh reset`. The running mod is older than Task 2, or `CampaignObserver.IsCampaignScene` rejects the scene. Repeat Steps 8-9 and confirm `mod 0.5.0`. If the version is right, check what `StatsManager.levelNumber` the scene reports against the `1 <= levelNumber <= 35` rule in `mod/UltrakillAIBridge/Obs/CampaignObserver.cs`.

(b) `difficulty reads N, expected 3 (Violent)`. The override was not in effect when the level loaded. The env must send `difficulty` in the `config` message on connect (before the first `reset` takes control and loads the scene), the mod must store it, and the `PrefsManager.GetInt` postfix must be patched. Check each in order:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
Select-String -Path ultrakill_ai\env.py -Pattern "difficulty=self.cfg.difficulty"
Select-String -Path ..\mod\UltrakillAIBridge\Env\EpisodeController.cs -Pattern "DifficultyOverride"
Select-String -Path "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\LogOutput.log" -Pattern "Harmony|CampaignPatches|Exception"
```

Expected: one match in each of the first two files and no patching error in the log. Fix whichever is missing (Task 7 for `env.py`, Task 1 for the mod), rebuild with Step 8 when the mod changed, Step 9, then Step 10.

(c) Level 1-1 prints `[SKIP] 2 arsenal: ... not every weapon slot is filled, so unlock_all_gear did not reach GunSetter`. Treat this as a failure. Check the same chain for the gear unlock:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
Select-String -Path ultrakill_ai\env.py -Pattern "unlock_all_gear=self.cfg.unlock_all_gear"
Select-String -Path ..\mod\UltrakillAIBridge\Env\EpisodeController.cs -Pattern "UnlockAllGear"
Select-String -Path "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\LogOutput.log" -Pattern "Harmony|CampaignPatches|Exception"
```

Fix the missing piece, rebuild if the mod changed (Step 8), Step 9, Step 10.

(d) The triggers do not fire at 30 fps / frameskip 2: check 3 says every tried checkpoint `did not activate`, or check 5 says `active=True: the trigger did not fire`. Rerun with one setting changed at a time, stopping at the first run where checks 3 and 5 both PASS:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\campaign_check.py --fixed-fps 60 --frameskip 4
```

```powershell
.venv\Scripts\python scripts\campaign_check.py --render
```

```powershell
.venv\Scripts\python scripts\campaign_check.py --fixed-fps 60 --frameskip 4 --render
```

Then make `python/configs/campaign_0-1.yaml` use the settings that worked. Replace these lines of the `env:` section:

```yaml
  fixed_fps: 30
  frameskip: 2
  render: false
```

with, when `--fixed-fps 60 --frameskip 4` worked:

```yaml
  fixed_fps: 60   # checkpoint/exit triggers did not fire at 30 / 2 (campaign_check.py, 2026-09-16); still 15 decisions per game second
  frameskip: 4
  render: false
```

or, when `--render` worked:

```yaml
  fixed_fps: 30
  frameskip: 2
  render: true    # checkpoint/exit triggers did not fire with the cameras off (campaign_check.py, 2026-09-16)
```

or, when only both together worked:

```yaml
  fixed_fps: 60   # checkpoint/exit triggers only fired at 60 / 4 with the cameras on (campaign_check.py, 2026-09-16)
  frameskip: 4
  render: true
```

Task 13's `python/tests/test_campaign_config.py` pins the speed settings, so change it with the YAML. Replace this line:

```python
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (30, 2, False, False)
```

with, when `--fixed-fps 60 --frameskip 4` worked:

```python
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (60, 4, False, False)
```

or, when `--render` worked:

```python
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (30, 2, True, False)
```

or, when only both together worked:

```python
    assert (cfg.fixed_fps, cfg.frameskip, cfg.render, cfg.soft_death) == (60, 4, True, False)
```

Run every no-game suite (`test_campaign_check.py` reads its defaults from the YAML, so it stays at `8 tests passed`, and `test_campaign_config.py` at `5 tests passed`):

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
foreach ($t in Get-ChildItem tests\test_*.py) { .venv\Scripts\python $t.FullName | Select-Object -Last 1; if ($LASTEXITCODE -ne 0) { throw "$($t.Name) failed" } }
```

Then Step 10 again with no flags. If none of the three runs passed, the cause is not the speed settings: it is blocked (see the top of this step).

(e) `[FAIL] 5 exit: exit null: ...`, or `[FAIL] 5 exit: exit (...) active=False: the trigger did not fire`. The exit's room is switched off at load (or the pit is filtered out). Retry after walking further, by hand, on a training instance (it never writes saves or prefs). Relaunch with a playable window:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py stop
.venv\Scripts\python scripts\games.py launch --count 1 --monitor 1 --width 1280 --height 720
```

In that window, start the failing level from the main menu and play to the room with the exit pit; stop at its edge without falling in. Print the campaign block without taking control:

```powershell
.venv\Scripts\python scripts\bridge_test.py --campaign
```

If `exit` is present there with `active` true, the pit's room was off at load: run the exit trigger and the timer from this spot with the snippet below. It takes control for 150 decisions (10 game seconds, which covers the results screen at 5 s) and then releases it. When case (d) changed the config, first give the snippet's `configure` call the same values: `fixed_fps=60, frameskip=4` for 60 / 4, and `render=True` when rendering was turned on.

```powershell
@'
import sys
sys.path.insert(0, ".")
from ultrakill_ai.protocol import BridgeClient
with BridgeClient(port=47800) as client:
    client.connect()
    client.configure(fixed_fps=30, frameskip=2, render=False)
    raw = client.get_obs()
    client.teleport(raw["campaign"]["exit"]["pos"])
    for _ in range(60):
        raw = client.step({})
        if raw["stats"].get("level_complete"):
            break
    finish = raw["campaign"]["seconds"]
    for _ in range(90):
        raw = client.step({})
    print("level_complete", raw["stats"].get("level_complete"), "timer_running", raw["campaign"]["timer_running"],
          "seconds at finish", round(finish, 3), "six seconds later", round(raw["campaign"]["seconds"], 3))
'@ | .venv\Scripts\python -
```

Expected: `level_complete True timer_running False` and the two seconds values equal. That verifies the exit and the official time by hand: the level's `5 exit FAIL` in `campaign_check.py` stays as a known limitation, and Step 15 adds the exit-room gotcha. `level_complete False`, or seconds that moved, is a real failure: blocked.

If `exit` is still null in the pit room, the exit filter dropped the real pit (every `FinalPit` in the scene looks like a decoy or a room template to `CampaignObserver`). Temporarily log each `FinalPit` that the exit choice in `mod/UltrakillAIBridge/Obs/CampaignObserver.cs` considers (name, position, `fakeEnd`, `secondPit`, `rankless`, whether it counted as a template, `activeInHierarchy`) with `Plugin.Log.LogInfo`, rebuild (Step 8), play to the pit room again as above, and read the lines:

```powershell
Select-String -Path "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\LogOutput.log" -Pattern "FinalPit"
```

Correct the filter from what the flags show, remove the logging, rebuild, and rerun Step 10; Step 17 commits the fix.

When case (e) is done, return to the small window before any further `campaign_check.py` run:

```powershell
.venv\Scripts\python scripts\games.py stop
.venv\Scripts\python scripts\games.py launch --count 1 --monitor 1
```

(f) `[FAIL] 4 death respawn`:
- `kill reply dead=False` (typically with `deaths 0 -> 1`, because the env counts a soft death and respawns after it too): soft death is on, so the mod healed the lethal hit and there was no real death. The env must send `soft_death` only for Cyber Grind and the campaign config has `soft_death: false`:

  ```powershell
  Set-Location F:\Github\ULTRAKILL-AI\python
  Select-String -Path ultrakill_ai\env.py -Pattern 'soft_death=self.cfg.soft_death and self.cfg.mode == "cybergrind"'
  Select-String -Path configs\campaign_0-1.yaml -Pattern "soft_death"
  ```

- `episode ended=True (death)`: the env ended the episode instead of respawning, so the campaign death branch of Task 7 (`_respawn`) did not run. Run `.venv\Scripts\python tests\test_campaign_env.py`, fix `env.py` until it passes, then Step 10.

- [ ] **Step 12: Switch to five games**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py stop
.venv\Scripts\python scripts\games.py launch --count 5 --monitor 1
```

Expected: `Started instance 0` to `Started instance 4`, then `All 5 instances ready on ports 47800-47804`.

- [ ] **Step 13: Run the 20k-step throughput smoke run**

Fresh weights, the real campaign config and all five games. It blocks for a few minutes.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
New-Item -ItemType Directory -Force runs | Out-Null   # gitignored, so it is missing on a fresh clone
cmd /c ".venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --timesteps 20000 --run-name campaign_smoke > runs\campaign_smoke_train.log 2>&1"
$LASTEXITCODE
```

Expected: exit code `0`. (A second PowerShell window can follow it with `Get-Content F:\Github\ULTRAKILL-AI\python\runs\campaign_smoke_train.log -Tail 20 -Wait`.)

- [ ] **Step 14: Read the throughput and confirm a clean finish**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$fps = [int](Select-String -Path runs\campaign_smoke_train.log -Pattern '^\|\s+fps\s+\|\s+(\d+)' | Select-Object -Last 1).Matches[0].Groups[1].Value
"fps $fps  ->  1M steps {0:N1} h, 20M steps {1:N0} h" -f (1e6 / $fps / 3600), (2e7 / $fps / 3600)
Select-String -Path runs\campaign_smoke_train.log -Pattern "Traceback", "^Saved models"
$s = Get-Content runs\campaign_smoke\status.json -Raw | ConvertFrom-Json
"state {0}  timesteps {1}  episodes {2}" -f $s.state, $s.timesteps, $s.episodes
$s.end_reasons_100 | Format-List
Get-ChildItem models\campaign_smoke | Select-Object Name, Length
```

Expected:
- the `fps` line (SB3's steps per second over the whole run, level loads included) and the hours it implies;
- exactly one `Saved models\campaign_smoke\latest.zip` match and no `Traceback`;
- `state finished`, `timesteps` about 20450 (10 rollouts of 409 steps x 5 games);
- `episodes` may be 0: random weights rarely end a 10-minute episode inside ~4,090 steps per game, unless 45 s pass without progress (`stuck`);
- `models\campaign_smoke` holds `env_config.yaml`, `latest.zip` and five `explore_Level_0-1_4780N.npz` (N 0-4), written by `env.close()` on a normal finish.

A `Traceback` or missing archives means the campaign pipeline is not ready for Task 15: debug with superpowers:systematic-debugging before going on. Note `fps`, the two hour figures, `episodes` and the end reasons for Step 15.

- [ ] **Step 15: Record the verification and throughput in CLAUDE.md**

Fill-ins (all measured, nothing else in the text changes):
- `{SPEED}`: `30 fps / frameskip 2 with rendering off`, unless Step 11 (d) changed the config: then `60 fps / frameskip 4 with rendering off`, `30 fps / frameskip 2 with rendering on` or `60 fps / frameskip 4 with rendering on`.
- `{FPS}`, `{H1M}` (one decimal), `{H20M}` (whole hours): the first line printed by Step 14.
- `{SUMMARY_0_1}`, `{SUMMARY_1_1}`: the final `summary:` lines of Step 10 (after any Step 11 rerun), without the `summary: ` prefix.
- `{EPISODES}`: Step 14's `episodes`. `{END_REASONS}`: its `end_reasons_100` entries written `name count`, joined with `, ` (`none` when empty).
- `{LEVEL}`: the level whose exit Step 11 case (e) verified by hand.

In `CLAUDE.md`, find the last Gotchas line:

```
- **Speed:** about 600 fps / 150 steps/s in an empty scene and about 100 steps/s with enemies (frameskip 4, RTX 5070).
```

and replace it with:

```
- **Speed:** about 600 fps / 150 steps/s in an empty scene and about 100 steps/s with enemies (frameskip 4, RTX 5070).
- **Campaign triggers at training speed:** checkpoint and exit triggers fire at {SPEED} (`campaign_check.py` on 0-1 and 1-1, 2026-09-16), which is what `configs/campaign_0-1.yaml` uses.
- **Campaign throughput:** {FPS} steps/s with 5 games on 0-1 at {SPEED} (20k-step smoke run from fresh weights; SB3's `fps` over the whole run, level loads included). At that rate 1M steps take {H1M} h and the 20M-step pilot {H20M} h.
```

Only when Step 11 case (e) verified an exit by hand, add this line directly after the throughput line:

```
- **Exit room switched off at load ({LEVEL}):** a fresh load reports the exit with `active` false, so teleporting to it cannot finish the level and `campaign_check.py` check 5 fails there. Verified by hand instead: played to the pit room in a 1280x720 training instance, then teleported in (`level_complete`, timer stopped). The agent has to reach that room before the exit can fire.
```

Then append at the end of `CLAUDE.md` (after its current last line):

```
- **Campaign verified in game (2026-09-16), mod v0.5.0, one game at {SPEED}:**
  - Level 0-1: {SUMMARY_0_1}.
  - Level 1-1: {SUMMARY_1_1}.
  - Throughput smoke run with 5 games (`train.py --config configs/campaign_0-1.yaml --timesteps 20000 --run-name campaign_smoke`, outputs deleted): finished cleanly at {FPS} steps/s; {EPISODES} episodes ended ({END_REASONS}).
  - Next: the 0-1 pilot run (`campaign_ppo`).
```

- [ ] **Step 16: Stop the games and delete the smoke run**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py stop
foreach ($p in 'models\campaign_smoke', 'runs\campaign_smoke', 'runs\campaign_smoke_1', 'runs\campaign_smoke_train.log') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
Test-Path models\campaign_smoke, runs\campaign_smoke, runs\campaign_smoke_1, runs\campaign_smoke_train.log
```

Expected: `Closing 5 running game(s)...`, `Restored your display settings`, then `False` four times.

- [ ] **Step 17: Commit and push the results**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add CLAUDE.md
git status --short -- python/configs python/ultrakill_ai python/scripts python/tests mod/UltrakillAIBridge docs
```

Add each file the last command lists by its path (files there only change when Step 11 applied a fix), for example `git add python/configs/campaign_0-1.yaml python/tests/test_campaign_config.py` after case (d). Then:

```powershell
git commit -m "Verify campaign support in game and measure campaign throughput" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

### Task 15: Pilot training run, monitoring, falsifier, `times.md`

**Files:**
- Create: `python/models/campaign_ppo/env_config.yaml`, `latest.zip`, `best.zip`, `best.json` and `explore_Level_0-1_47800.npz` to `explore_Level_0-1_47804.npz` (training outputs, committed)
- Modify: `python/models/campaign_ppo/transfer_init.zip` (only when Step 7 rebuilds it with another `--action-scale`; Task 12 created and committed it)
- Modify: `times.md` (written by `eval.py --record-times`)
- Modify: `CLAUDE.md` (Status entries, appended)
- Modify: `.gitignore` (only when Step 13 switches the pilot to fresh weights)
- Test: none (a live training run; every check below reads `status.json` or `metrics_log.csv`)

Ground rules, learned on the Cyber Grind runs (CLAUDE.md Status):
- Judge the run only from live `runs/<run>/status.json` and `runs/<run>/metrics_log.csv`, never from an offline probe of the weights.
- Means need `window` >= 50 behind them; the completion gate needs `campaign.fresh_window` >= 20.
- Change one thing at a time. The only planned changes are the single entropy retry (Step 7) and the falsifier's switch to fresh weights (Step 13). Anything else gets at least 400k steps before it is judged.
- Never connect anything to ports 47800-47804 while a trainer runs (no `bridge_test.py`, `campaign_check.py` or `eval.py`): the single-client bridge drops the trainer and the run dies.
- Stop training only with Ctrl+C in its console, then wait for the `train.py` process to exit (it prints `Saved models\<run>\latest.zip` last). A hard stop (closing the window, killing the process) leaves `latest.zip` stale; resume from the newest `ckpt_*_steps.zip` then.

- [ ] **Step 1: Check the starting state**

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git status --short -- CLAUDE.md python/scripts python/ultrakill_ai python/configs mod
git log --oneline -3
Set-Location python
.venv\Scripts\python scripts\games.py status
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'train\.py|eval\.py|poll_status|keep_best' } | Select-Object ProcessId, CommandLine
```

Expected: no changed tracked files under those paths, Task 14's results commit on top, `0 game process(es) running: []` with no `bridge listening on` line, and no `train.py` or `eval.py` process (one would still own the games: stop it first). Stop any leftover helper from the Cyber Grind run:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'poll_status|keep_best' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Confirm:$false -ErrorAction Stop } catch {} }
```

- [ ] **Step 2: Check the starting weights**

Task 12 built `models/campaign_ppo/transfer_init.zip` (`--action-scale 0.5`) and committed it. Confirm it loads with 479 inputs; build it only if it is missing, because rebuilding an existing one gives the same weights in a zip with new timestamps, a spurious binary change in git:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
if (-not (Test-Path models\campaign_ppo\transfer_init.zip)) { .venv\Scripts\python scripts\transfer_weights.py models\cybergrind_ppo_v2\best.zip models\campaign_ppo\transfer_init.zip; if ($LASTEXITCODE -ne 0) { throw "transfer_weights.py failed" } }
.venv\Scripts\python -c "from stable_baselines3 import PPO; m = PPO.load('models/campaign_ppo/transfer_init.zip', device='cpu'); print(m.observation_space.shape, m.num_timesteps)"
Get-Item models\campaign_ppo\transfer_init.zip | Select-Object Name, Length
```

Expected: `(479,) 0`, and a `Length` above 3 MB (about 4,190,000 bytes: both 479-512-512 hidden stacks, no optimizer moments yet).

- [ ] **Step 3: Launch five games**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py launch --count 5 --monitor 1
```

Expected: `All 5 instances ready on ports 47800-47804`.

- [ ] **Step 4: Start training in its own console**

A separate console, so Ctrl+C can stop it gracefully while this PowerShell stays free. Output goes to `runs\campaign_ppo_train.log`.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
New-Item -ItemType Directory -Force runs | Out-Null   # gitignored, so it is missing on a fresh clone
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_ppo\transfer_init.zip >> runs\campaign_ppo_train.log 2>&1'
```

Expected within about a minute: `Get-Content F:\Github\ULTRAKILL-AI\python\runs\campaign_ppo_train.log -Tail 20` shows no `Traceback`, and `runs\campaign_ppo\status.json` exists with `"state":"running"`.

- [ ] **Step 5: Start the helpers and the dashboard**

```powershell
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\poll_status.py --run campaign_ppo >> runs\campaign_ppo_poll.log 2>&1'
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\keep_best.py --run campaign_ppo --metric campaign >> runs\campaign_ppo_keep_best.log 2>&1'
Start-Process F:\Github\ULTRAKILL-AI\python\.venv\Scripts\pythonw.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList 'scripts\dashboard.py', '--run', 'campaign_ppo', '--monitor', '1'
```

Expected: the dashboard opens on monitor 1 below the game row with the campaign panel (or `Waiting for ...status.json` for the first seconds), and within a few minutes `runs\campaign_ppo\metrics_log.csv` exists.

- [ ] **Step 6: Check the first-update entropy (6 to 10 nats)**

SB3 logs `train/entropy_loss` as minus the mean entropy of the MultiDiscrete action distribution (at most 12.49 nats, the sum of `ln` of the head sizes). `status.json` has it once the first PPO update has run, about two rollouts in.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$s = $null
while ($null -eq $s -or ($null -eq $s.ppo.entropy_loss -and $s.state -ne 'stopped')) {
    Start-Sleep -Seconds 15
    try { $s = Get-Content runs\campaign_ppo\status.json -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $s = $null }
}
"state {3}  timesteps {0:N0}  first-update entropy {1:N2} nats  steps/s {2:N0}" -f $s.timesteps, (-$s.ppo.entropy_loss), $s.steps_per_s, $s.state
```

If it printed `state stopped`, the trainer died before its first update: read the `Traceback` in `runs\campaign_ppo_train.log`, fix the cause, restart with Step 4's command and run this step again.

Decide:
- 6.00 to 10.00: note the entropy and steps/s, and go to Step 8 with scale 0.5.
- Below 6.00: the transferred action head is still too sharp. Step 7 with `--action-scale 0.3`.
- Above 10.00: too flat. Step 7 with `--action-scale 0.7`.
- If this reading already comes from the Step 7 retry, go to Step 8 whatever the value (the spec allows one adjustment).

- [ ] **Step 7: Retry the transfer once (only when Step 6 was out of range)**

1. Press Ctrl+C in the `campaign_ppo training` console, then wait for the trainer to exit (it saves `latest.zip` on the way out; the log is appended across restarts, so the process, not an old `Saved` line, is what to wait on):

   ```powershell
   Set-Location F:\Github\ULTRAKILL-AI\python
   while (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'train\.py' }) { Start-Sleep -Seconds 5 }
   Select-String -Path runs\campaign_ppo_train.log -Pattern '^Saved models' | Select-Object -Last 1
   ```

2. Stop the run's helpers (a deleted CSV under a running `poll_status.py` would be rewritten without its header):

   ```powershell
   Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match '(poll_status|keep_best)\.py --run campaign_ppo(\s|$)' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Confirm:$false -ErrorAction Stop } catch {} }
   ```

3. Delete everything the short run wrote, keeping `transfer_init.zip` to overwrite. Its TensorBoard folder is `runs\campaign_ppo_0`: SB3 continues the newest `<run>_N` folder when `train.py` resumes (and `--resume transfer_init.zip` is a resume), which with none yet is `_0`:

   ```powershell
   foreach ($p in 'runs\campaign_ppo', 'runs\campaign_ppo_0', 'runs\campaign_ppo_train.log', 'runs\campaign_ppo_poll.log', 'runs\campaign_ppo_keep_best.log', 'models\campaign_ppo\latest.zip', 'models\campaign_ppo\env_config.yaml') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
   Get-ChildItem models\campaign_ppo -Recurse -Include explore_*.npz, ckpt_*.zip, best.zip, best.json | Remove-Item -Force
   ```

4. Rebuild the starting weights with the new scale (0.3 when entropy was below 6, 0.7 when above 10); the command for 0.3:

   ```powershell
   .venv\Scripts\python scripts\transfer_weights.py models\cybergrind_ppo_v2\best.zip models\campaign_ppo\transfer_init.zip --action-scale 0.3
   ```

   and for 0.7:

   ```powershell
   .venv\Scripts\python scripts\transfer_weights.py models\cybergrind_ppo_v2\best.zip models\campaign_ppo\transfer_init.zip --action-scale 0.7
   ```

5. Restart training and the helpers (the dashboard stays open and picks the new file up):

   ```powershell
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_ppo\transfer_init.zip >> runs\campaign_ppo_train.log 2>&1'
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\poll_status.py --run campaign_ppo >> runs\campaign_ppo_poll.log 2>&1'
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\keep_best.py --run campaign_ppo --metric campaign >> runs\campaign_ppo_keep_best.log 2>&1'
   ```

6. Close the old training console window, then run Step 6 again (it will send you to Step 8).

- [ ] **Step 8: Record the start in CLAUDE.md, commit and push**

Fill-ins: `{SCALE}` is the `--action-scale` of the `transfer_init.zip` now training (0.5, or 0.3 / 0.7 after Step 7). `{ENTROPY}` (two decimals) and `{SPS}` (whole number) come from the last Step 6 reading. `{RETRY}` is empty, or after Step 7 exactly `; the first try at --action-scale 0.5 read {FIRST_ENTROPY} nats, so the scale was changed once` with `{FIRST_ENTROPY}` the first Step 6 reading.

Append at the end of `CLAUDE.md`:

```
- **Campaign pilot `campaign_ppo` (0-1, Violent), started 2026-09-16.**
  - Start: `models/campaign_ppo/transfer_init.zip`, the Cyber Grind v2 `best.zip` (7.16 kills/min, `ckpt_4679085_steps`) widened to 479 inputs by `transfer_weights.py --action-scale {SCALE}`. First-update entropy {ENTROPY} nats against a 6-10 target{RETRY}; {SPS} steps/s with 5 games.
  - Stop with Ctrl+C in the training console and wait for `train.py` to exit (it prints `Saved models\campaign_ppo\latest.zip` last; `runs/campaign_ppo_train.log` is appended across restarts, so an older such line proves nothing), then resume with `python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/latest.zip`. After a hard stop resume from the newest `ckpt_*_steps.zip`, not `best.zip`: `keep_best.py --metric campaign` only replaces `best.zip` when the smoothed fresh completion rate beats its record, so until levels are being completed it holds the first checkpoint scored (rate 0) and resuming from it would throw the run away. Use `best.zip` to roll back once completions have peaked and fallen (keep_best warns). Helpers: `poll_status.py --run campaign_ppo`, `keep_best.py --run campaign_ppo --metric campaign`, `dashboard.py --run campaign_ppo --monitor 1`.
  - Ctrl+C leaves the exploration archives slightly stale: SB3's subprocess workers exit on KeyboardInterrupt without calling `env.close()`, so each `explore_*.npz` holds its last 20-episode save.
  - Checks: `best_checkpoints_level` >= 1 by 250k steps; >= 2 by 1M steps, else a 1M-step fresh-weights comparison run (`campaign_ppo_fresh`); eval gate once `campaign.fresh_completion_rate` >= 0.5 with `fresh_window` >= 20.
```

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add python/models/campaign_ppo/transfer_init.zip python/models/campaign_ppo/env_config.yaml CLAUDE.md
git commit -m "Start the campaign pilot on 0-1 from widened Cyber Grind weights" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 9: Reading the run, and recovering from a crash**

The status readout used by the checks below (safe at any time: it only reads a file):

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$s = Get-Content runs\campaign_ppo\status.json -Raw | ConvertFrom-Json
"{0}  steps {1:N0}  steps/s {2:N0}  episodes {3}  window {4}  entropy {5:N2}" -f $s.state, $s.timesteps, $s.steps_per_s, $s.episodes, $s.window, (-$s.ppo.entropy_loss)
"best_checkpoints_level {0}  checkpoints/load {1:N2}  new cells/ep {2:N1}  deaths/ep {3:N2}  closest to exit {4:N1} m" -f $s.best_checkpoints_level, $s.mean_100.checkpoints_level, $s.mean_100.cells_new, $s.mean_100.deaths, $s.mean_100.exit_dist_min
"fresh completion {0:P0} over {1} fresh episodes  best time {2}  median time {3}" -f $s.campaign.fresh_completion_rate, $s.campaign.fresh_window, $s.campaign.best_time, $s.campaign.median_time_50
$s.end_reasons_100 | Format-List
```

If `state` reads `stopped` without a Ctrl+C, or the log shows a `Traceback` (typically `BridgeError: Connection closed by the game` after a game crashed), the trainer's `finally` block has already saved `latest.zip`. Close its console, relaunch the games and resume:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
.venv\Scripts\python scripts\games.py launch --count 5 --monitor 1
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_ppo\latest.zip >> runs\campaign_ppo_train.log 2>&1'
```

After Step 13 has switched the pilot to `campaign_ppo_fresh`, read `runs\campaign_ppo_fresh\status.json` instead, and resume with Step 16's command (`$run = 'campaign_ppo_fresh'`) after relaunching the games.

- [ ] **Step 10: 250k-step check: at least one checkpoint activated**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$s = $null
while ($null -eq $s -or ($s.timesteps -lt 250000 -and $s.state -ne 'stopped')) {
    Start-Sleep -Seconds 60
    try { $s = Get-Content runs\campaign_ppo\status.json -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $s = $null }
}
"{0} at {1:N0} steps" -f $s.state, $s.timesteps
```

`stopped` below 250,000 steps means the run crashed: recover with Step 9 and run this wait again. Otherwise run the Step 9 readout and take the first matching case (an empty `best_checkpoints_level` or `new cells/ep` means no episode has ended yet: count it as 0).

**`best_checkpoints_level` >= 1.** The pipeline reaches a milestone. `{ACTION}` below is `Continued unchanged.`

**`best_checkpoints_level` 0 with `new cells/ep` 10 or more** (it explores but stops short, usually `stuck` in the end reasons). Compare exploration over the last 150k steps:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$rows = Import-Csv runs\campaign_ppo\metrics_log.csv | Where-Object { [double]$_.window -ge 50 }
$early = @($rows | Where-Object { [double]$_.timesteps -ge 100000 -and [double]$_.timesteps -lt 175000 })
$late = @($rows | Where-Object { [double]$_.timesteps -ge 175000 -and [double]$_.timesteps -le 260000 })
"new cells/ep 100k-175k {0:N1} ({1} rows), 175k-260k {2:N1} ({3} rows)" -f ($early | Measure-Object cells_new -Average).Average, $early.Count, ($late | Measure-Object cells_new -Average).Average, $late.Count
```

If both ranges have rows and the later mean is more than 25% below the earlier one, exploration is collapsing: do the spawn investigation of the next case. Otherwise (rising, flat, or too few 50-episode windows yet) change nothing and let the 1M falsifier decide; `{ACTION}` is `No checkpoint yet with {CN} new cells per episode; continued unchanged to the 1M falsifier.`

**`best_checkpoints_level` 0 with `new cells/ep` below 10** (under ~40 m of distinct ground per episode): the player is not leaving the start. Investigate the spawn. Press Ctrl+C in the training console and wait for the save:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
while (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'train\.py' }) { Start-Sleep -Seconds 5 }
Select-String -Path runs\campaign_ppo_train.log -Pattern '^Saved models' | Select-Object -Last 1
```

With the trainer stopped, port 47800 is free. Confirm that resets still hand over a free player (check 1 PASS with `input_locked=False`), then watch one episode of the current policy at normal speed on game 0:

```powershell
.venv\Scripts\python scripts\campaign_check.py
.venv\Scripts\python scripts\eval.py models\campaign_ppo\latest.zip --episodes 1 --level "Level 0-1" --realtime --stochastic
```

If the player never gets control (check 1 shows `input_locked=True`, or the camera stays frozen through the episode), fix the input-lock skipping in `python/ultrakill_ai/env.py` (`_skip_locked`) test-first, with a failing case added to `python/tests/test_campaign_env.py`, and run every no-game suite. The fix is committed below together with its CLAUDE.md entry (project rule: CLAUDE.md is updated in the same commit):

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
foreach ($t in Get-ChildItem tests\test_*.py) { .venv\Scripts\python $t.FullName | Select-Object -Last 1; if ($LASTEXITCODE -ne 0) { throw "$($t.Name) failed" } }
```

If the player moves but circles the start, change nothing. Either way, resume:

```powershell
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_ppo\latest.zip >> runs\campaign_ppo_train.log 2>&1'
```

`{ACTION}` is `Watched an episode: the player moves but stays near the start; continued unchanged to the 1M falsifier.`, or one sentence saying what check 1 and the realtime episode showed and naming the `env.py` fix and its test.

Fill-ins from the Step 9 readout: `{B}` best_checkpoints_level, `{CPL}` checkpoints/load, `{CN}` new cells/ep, `{D}` deaths/ep, `{SPS}` steps/s, `{ER}` the end reasons as `name count` joined with `, `. Append at the end of `CLAUDE.md`, then commit and push:

```
  - 250k steps: `best_checkpoints_level` {B}, {CPL} checkpoints per level load, {CN} new cells and {D} deaths per episode, end reasons {ER}, {SPS} steps/s. {ACTION}
```

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add CLAUDE.md
git status --short -- python/ultrakill_ai/env.py python/tests/test_campaign_env.py
```

When the spawn investigation fixed `env.py`, the last command lists both files: add them too with `git add python/ultrakill_ai/env.py python/tests/test_campaign_env.py`. Then:

```powershell
git commit -m "Record the campaign pilot's 250k-step check" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 11: 1M-step falsifier: a second checkpoint**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$s = $null
while ($null -eq $s -or ($s.timesteps -lt 1000000 -and $s.state -ne 'stopped')) {
    Start-Sleep -Seconds 300
    try { $s = Get-Content runs\campaign_ppo\status.json -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $s = $null }
}
"{0} at {1:N0} steps: best_checkpoints_level {2}" -f $s.state, $s.timesteps, $s.best_checkpoints_level
```

`stopped` below 1,000,000 steps means the run crashed: recover with Step 9 and run this wait again.

**Raise the threshold by any checkpoint that was already activated at level load.** `CheckPoint.Start` marks
`startOff` checkpoints activated straight away, and `checkpoints_level` counts those too, so on a level that has
them the number starts above 0 and "2" would not mean the agent reached a second one. Task 14's first check
printed every checkpoint with its flags: let `{A}` be how many were already `activated` on a fresh load (0 for
`Level 0-1` unless that check said otherwise). The falsifier threshold below is `2 + {A}`, and the 250k check in
Step 10 was `1 + {A}`.

When `best_checkpoints_level` >= 2 + `{A}` the falsifier passes: append this line at the end of `CLAUDE.md` (`{B}` is the printed value), commit and push, and go to Step 14.

```
  - 1M-step falsifier passed: `best_checkpoints_level` {B}, so the transferred weights reach past the first checkpoint.
```

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add CLAUDE.md
git commit -m "Record the campaign pilot's 1M-step falsifier" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

When `best_checkpoints_level` < 2 + `{A}` (no game activated a second checkpoint), run Steps 12-13.

- [ ] **Step 12: Run 1M steps from fresh weights (only when Step 11 failed)**

1. Ctrl+C in the `campaign_ppo training` console and wait for the save:

   ```powershell
   Set-Location F:\Github\ULTRAKILL-AI\python
   while (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'train\.py' }) { Start-Sleep -Seconds 5 }
   Select-String -Path runs\campaign_ppo_train.log -Pattern '^Saved models' | Select-Object -Last 1
   ```

2. Start the fresh-weights run (no `--resume`, its own run name, 1M steps), its CSV logger and its dashboard, and close the `campaign_ppo` dashboard window. The `campaign_ppo` helpers may keep running: nothing changes their files while that run is stopped.

   ```powershell
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo_fresh training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --run-name campaign_ppo_fresh --timesteps 1000000 >> runs\campaign_ppo_fresh_train.log 2>&1'
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\poll_status.py --run campaign_ppo_fresh >> runs\campaign_ppo_fresh_poll.log 2>&1'
   Start-Process F:\Github\ULTRAKILL-AI\python\.venv\Scripts\pythonw.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList 'scripts\dashboard.py', '--run', 'campaign_ppo_fresh', '--monitor', '1'
   ```

3. Wait for it to finish (about as long as the first 1M steps took):

   ```powershell
   Set-Location F:\Github\ULTRAKILL-AI\python
   $f = $null
   while ($null -eq $f -or $f.state -eq 'running' -or $f.state -eq 'starting') {
       Start-Sleep -Seconds 300
       try { $f = Get-Content runs\campaign_ppo_fresh\status.json -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $f = $null }
   }
   "{0} at {1:N0} steps" -f $f.state, $f.timesteps
   ```

   `finished` means done. `stopped` means it crashed before 1M: relaunch the games with `.venv\Scripts\python scripts\games.py launch --count 5 --monitor 1`, close the old console, resume with the command below, and run this wait again.

   ```powershell
   Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo_fresh training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --run-name campaign_ppo_fresh --timesteps 1000000 --resume models\campaign_ppo_fresh\latest.zip >> runs\campaign_ppo_fresh_train.log 2>&1'
   ```

4. Compare both runs over 0.9M-1.05M steps (50-episode windows only) and print the decision:

   ```powershell
   Set-Location F:\Github\ULTRAKILL-AI\python
   $t = @(Import-Csv runs\campaign_ppo\metrics_log.csv | Where-Object { [double]$_.window -ge 50 -and [double]$_.timesteps -ge 900000 -and [double]$_.timesteps -le 1050000 })
   $r = @(Import-Csv runs\campaign_ppo_fresh\metrics_log.csv | Where-Object { [double]$_.window -ge 50 -and [double]$_.timesteps -ge 900000 -and [double]$_.timesteps -le 1050000 })
   $tb = [int](Get-Content runs\campaign_ppo\status.json -Raw | ConvertFrom-Json).best_checkpoints_level
   $fb = [int](Get-Content runs\campaign_ppo_fresh\status.json -Raw | ConvertFrom-Json).best_checkpoints_level
   $tc = ($t | Measure-Object cells_new -Average).Average
   $fc = ($r | Measure-Object cells_new -Average).Average
   "transfer: best_checkpoints_level {0}, new cells/ep {1:N1}, checkpoints/load {2:N2} ({3} rows)" -f $tb, $tc, ($t | Measure-Object checkpoints_level -Average).Average, $t.Count
   "fresh:    best_checkpoints_level {0}, new cells/ep {1:N1}, checkpoints/load {2:N2} ({3} rows)" -f $fb, $fc, ($r | Measure-Object checkpoints_level -Average).Average, $r.Count
   if ($fb -gt $tb -or ($fb -eq $tb -and $fc -ge 1.25 * $tc)) { "fresh weights did better: Step 13, first case" } else { "fresh weights did no better: Step 13, second case" }
   ```

- [ ] **Step 13: Act on the comparison (only after Step 12)**

Fill-ins from Step 12's comparison: `{TB}` and `{FB}` are the two `best_checkpoints_level` values, `{TC}` and `{FC}` the two new cells per episode (one decimal).

**First case, fresh weights did better** (a higher `best_checkpoints_level`, or the same with at least 25% more new cells per episode). The Cyber Grind habits are the obstacle, so the pilot continues from `campaign_ppo_fresh`. In `.gitignore`, find the line

```
python/models/campaign_smoke/
```

and replace it with

```
python/models/campaign_smoke/
python/models/campaign_ppo_fresh/ckpt_*.zip
```

Move the best-checkpoint keeper to the new run:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match '(poll_status|keep_best)\.py --run campaign_ppo(\s|$)' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Confirm:$false -ErrorAction Stop } catch {} }
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized -ArgumentList '/c', '.venv\Scripts\python.exe -u scripts\keep_best.py --run campaign_ppo_fresh --metric campaign >> runs\campaign_ppo_fresh_keep_best.log 2>&1'
```

Append at the end of `CLAUDE.md`:

```
  - 1M-step falsifier FAILED on the transfer run (`best_checkpoints_level` {TB}, {TC} new cells per episode over 0.9-1.05M steps), and 1M steps from fresh weights (`campaign_ppo_fresh`) did better ({FB}, {FC}). The Cyber Grind habits were the obstacle, so the pilot continues as `campaign_ppo_fresh`: resume with `--run-name campaign_ppo_fresh --resume models/campaign_ppo_fresh/latest.zip`, and point the helpers and the dashboard at `--run campaign_ppo_fresh`. The transfer run stays in git at 1M steps (`models/campaign_ppo/latest.zip`) for reference.
```

Commit and push (the transfer run's files are stable while it is stopped), then close the finished `campaign_ppo_fresh training` console and continue the fresh run to the config's 20M steps:

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add .gitignore CLAUDE.md python/models/campaign_ppo/latest.zip "python/models/campaign_ppo/explore_*.npz"
git commit -m "Switch the campaign pilot to fresh weights after the 1M-step falsifier" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo_fresh training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --run-name campaign_ppo_fresh --resume models\campaign_ppo_fresh\latest.zip >> runs\campaign_ppo_fresh_train.log 2>&1'
```

**Second case, fresh weights did no better.** The transferred weights are not what blocks the second checkpoint, so the transfer run resumes unchanged. Close the finished `campaign_ppo_fresh training` console and its dashboard, then stop the fresh run's logger, delete its outputs, and resume `campaign_ppo`:

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'poll_status\.py --run campaign_ppo_fresh' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Confirm:$false -ErrorAction Stop } catch {} }
foreach ($p in 'models\campaign_ppo_fresh', 'runs\campaign_ppo_fresh', 'runs\campaign_ppo_fresh_1', 'runs\campaign_ppo_fresh_train.log', 'runs\campaign_ppo_fresh_poll.log') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', 'title campaign_ppo training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_ppo\latest.zip >> runs\campaign_ppo_train.log 2>&1'
Start-Process F:\Github\ULTRAKILL-AI\python\.venv\Scripts\pythonw.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList 'scripts\dashboard.py', '--run', 'campaign_ppo', '--monitor', '1'
```

Append at the end of `CLAUDE.md`:

```
  - 1M-step falsifier FAILED (`best_checkpoints_level` {TB}, {TC} new cells per episode over 0.9-1.05M steps), but 1M steps from fresh weights did no better ({FB}, {FC}; that run was deleted), so the transferred weights are not what blocks the second checkpoint. Resumed `campaign_ppo` unchanged. If `best_checkpoints_level` is still below 2 at 2M steps, stop and watch an episode (`eval.py models/campaign_ppo/latest.zip --episodes 1 --level "Level 0-1" --realtime --stochastic`) before choosing the one next change.
```

```powershell
Set-Location F:\Github\ULTRAKILL-AI
git add CLAUDE.md
git commit -m "Record the campaign pilot's 1M-step falsifier comparison" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 14: Eval gate: at least half of fresh starts completing**

`$run` is `campaign_ppo`, or `campaign_ppo_fresh` when Step 13 switched runs. `$minSteps` is 0 for the first gate, and the previous gate's `timesteps` plus 500000 when re-running after a failed eval (Step 16). `$evaluated` is `''` for the first gate, and on a re-run the `{CKPT}` recorded in the failed eval's CLAUDE.md entry.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$run = 'campaign_ppo'
$minSteps = 0
$evaluated = ''
$s = $null
while ($null -eq $s -or ($s.state -ne 'stopped' -and ($s.timesteps -lt $minSteps -or -not ($s.campaign.fresh_completion_rate -ge 0.5 -and $s.campaign.fresh_window -ge 20)))) {
    Start-Sleep -Seconds 300
    try { $s = Get-Content "runs\$run\status.json" -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $s = $null }
}
"{3}: gate at {0:N0} steps: fresh completion {1:P0} over {2} fresh episodes" -f $s.timesteps, $s.campaign.fresh_completion_rate, $s.campaign.fresh_window, $s.state
$b = $null
while ($s.state -ne 'stopped' -and ($null -eq $b -or $b.score -lt 0.5)) {
    Start-Sleep -Seconds 60
    try { $b = Get-Content "models\$run\best.json" -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $b = $null }
    # Re-read the run each time: a crash after the gate would otherwise leave this loop waiting forever.
    try { $s = Get-Content "runs\$run\status.json" -Raw -ErrorAction Stop | ConvertFrom-Json } catch { }
}
"{0}: best.json score {1}" -f $s.state, $b.score
$b
```

Either printed line starting `stopped:` means the run crashed (before the gate, or between the gate and `best.json` catching up): recover with Step 9 (Step 16's command resumes either run) and run this step again. The second loop waits until `keep_best.py` has copied a checkpoint whose smoothed completion rate is at least 0.5 into `best.zip`, so the eval measures the gated policy and not an older one. Its score is `keep_best.py`'s 9-sample mean of `fresh_completion_rate`, so it reaches 0.5 a few polls after `status.json` does. On a re-run, `keep_best.py` replaces `best.zip` only when the smoothed rate beats its record, so `best.json` may still name the checkpoint the last eval failed; the eval then takes the current policy, `latest.zip`, instead (saved by the Ctrl+C below):

```powershell
$model = if ($b.checkpoint -eq $evaluated) { "models\$run\latest.zip" } else { "models\$run\best.zip" }
"eval model $model (best.json checkpoint $($b.checkpoint))"
```

1. Ctrl+C in the training console and wait for the save:

   ```powershell
   while (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'train\.py' }) { Start-Sleep -Seconds 5 }
   Select-String -Path "runs\${run}_train.log" -Pattern '^Saved models' | Select-Object -Last 1
   ```

2. Evaluate `$model` on game 0 (deterministic actions, fresh loads, real deaths, the run's own Violent config), writing the fastest completion to `times.md`:

   ```powershell
   .venv\Scripts\python scripts\eval.py $model --episodes 10 --level "Level 0-1" --record-times
   Get-Content ..\times.md -Encoding UTF8 | Select-String "eval runs completed"
   ```

   The eval prints one `episode N: ... completed=0|1 time=... rank=...` line per run, then `completed N/10 fresh runs of Level 0-1`, and after any completion `fastest mm:ss.mmm rank=X ...` and `recorded mm:ss.mmm (<run>@<steps>M) in ...times.md`; without one it prints `no completed run, times.md unchanged`. The new generation-history row in `times.md` carries the same count in its notes as `N/10 eval runs completed`.
3. Success is at least 5 of 10. On fewer, run the same policy with sampled actions once, without recording, to see whether determinism is the gap (read its `completed N/10` line):

   ```powershell
   .venv\Scripts\python scripts\eval.py $model --episodes 10 --level "Level 0-1" --stochastic
   ```

- [ ] **Step 15: Record the eval, commit and push the pilot's state**

Fill-ins: `{RUN}` is `$run`; `{MODEL}` is the file name in `$model` (`best.zip` or `latest.zip`); `{CKPT}` is `$b.checkpoint`, the checkpoint `best.json` named at this gate; `{STEPS}` (whole number with thousands separators), `{RATE}` and `{WINDOW}` come from the gate line of Step 14; `{N}` is the deterministic completion count (`completed N/10`); `{TIME}` and `{RANK}` are the fastest completion's official time (`mm:ss.mmm`) and rank as written to `times.md`; `{NS}` is the sampled-action completion count.

When `{N}` >= 5, append at the end of `CLAUDE.md`:

```
  - Eval gate at {STEPS} steps: training showed {RATE} fresh-start completion over {WINDOW} fresh episodes; `eval.py models/{RUN}/{MODEL} --episodes 10 --level "Level 0-1" --record-times` (deterministic, fresh loads, real deaths, Violent; `best.json` checkpoint `{CKPT}`) completed {N}/10, fastest {TIME} rank {RANK}, recorded in `times.md`. **The foundation spec's success bar (at least 5/10) is met.** Training resumed from `latest.zip`.
```

When `{N}` < 5, append instead:

```
  - Eval gate at {STEPS} steps: training showed {RATE} fresh-start completion over {WINDOW} fresh episodes, but the deterministic eval of `models/{RUN}/{MODEL}` (`best.json` checkpoint `{CKPT}`) completed only {N}/10 ({NS}/10 with sampled actions). Success bar not met; training resumed from `latest.zip`, and the gate is re-run once it still holds at least 500k steps later (evaluating `latest.zip` if `best.json` still names `{CKPT}`).
```

Commit before training resumes, so no checkpoint or archive changes mid-commit. `transfer_init.zip` is unchanged since Task 12 (or Step 8 after a retry); `times.md` only changes when a run completed.

```powershell
Set-Location F:\Github\ULTRAKILL-AI
$run = 'campaign_ppo'
git add "python/models/$run/latest.zip" "python/models/$run/best.zip" "python/models/$run/best.json" "python/models/$run/env_config.yaml" "python/models/$run/explore_*.npz" python/models/campaign_ppo/transfer_init.zip times.md CLAUDE.md
git commit -m "Record the campaign pilot's eval gate and commit its weights" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

(Set `$run = 'campaign_ppo_fresh'` in the second line when Step 13 switched runs.)

- [ ] **Step 16: Resume training**

```powershell
Set-Location F:\Github\ULTRAKILL-AI\python
$run = 'campaign_ppo'
Start-Process cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -ArgumentList '/k', "title $run training - press Ctrl+C here to stop and save && .venv\Scripts\python.exe -u scripts\train.py --config configs\campaign_0-1.yaml --run-name $run --resume models\$run\latest.zip >> runs\${run}_train.log 2>&1"
```

(Again `$run = 'campaign_ppo_fresh'` after a switch.) Close the old training console. After a failed eval, repeat Steps 14-16 with `$minSteps` set to this gate's `timesteps` plus 500000 and `$evaluated` set to this gate's `{CKPT}`; after a successful one the run continues toward its 20M steps, and Task 15 is complete.

