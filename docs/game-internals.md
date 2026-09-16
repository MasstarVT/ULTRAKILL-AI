# Game internals used by the mod

Verified against the Steam build using Unity 2022.3.29 (Mono), decompiled with ILSpy (`ilspycmd`).
When a game update breaks the mod, check these first. Everything game-specific is in
`ObservationBuilder.cs`, `CampaignObserver.cs`, `ActionInjector.cs`, `EpisodeController.cs` and the Harmony
patch classes in `Env/`.

To decompile locally (never commit the output):

```bash
dotnet tool install ilspycmd --version 9.1.0.7988 --tool-path .tools
.tools/ilspycmd -p -o decompiled -r "<ULTRAKILL>/ULTRAKILL_Data/Managed" "<ULTRAKILL>/ULTRAKILL_Data/Managed/Assembly-CSharp.dll"
```

## Singletons
Most managers inherit `MonoSingleton<T>` and are accessed as `MonoSingleton<T>.Instance`. `GameStateManager.Instance` is a plain static property.

## Player: `NewMovement`
- `hp` (int, 100 max), `antiHp` (hard damage), `boostCharge` (float, 300 = 3 dashes), `dead`, `activated`, `sliding`, `levelOver`
- `rb` (Rigidbody), `gc` (`GroundCheckGroup`, `.onGround`)
- Movement input: `InputSource.Move.ReadValue<Vector2>()` in `NewMovement.Update`

## Camera: `CameraController`
- `rotationY` (yaw, degrees), `rotationX` (pitch, degrees, positive = up, clamped to ±90), `activated`, `platformerCamera`, `cam`
- Look input: `InputSource.Look.ReadValue<Vector2>() * mouseSensitivity / 10`, then `ApplyRotations()`
- `GameStateManager.Instance.CameraLocked` blocks look during cutscenes

## Input: `InputManager.Instance.InputSource` (`PlayerInput`)
- Contains `InputActionState` fields: `Move, Look, Punch, Hook, Fire1, Fire2, Jump, Slide, Dodge, ChangeFist, Slot1..Slot6, NextVariation, ...`
- `InputActionState` subscribes to `InputAction.started` and `canceled`, and records `IsPressed` and `PerformedFrame`. That's why the mod drives input through **virtual Input System devices** rather than Harmony patches on the getters. Mono can inline those tiny auto-property getters, so patches on them wouldn't reliably take effect.

## Weapons: `GunControl`
- `currentSlotIndex`, `currentVariationIndex`, `slot1..slot6` lists, and `slots` (`List<List<GameObject>>`, the six lists in order, filled in `Start`)
- `GunSetter.CheckWeapon` adds a variant when `PrefsManager.GetInt("weapon." + name, 1)` is above 0 (0 off, 1 on, 2 alternate) and `GameProgressSaver.CheckGear(name) > 0`. `FistControl` needs the pref `== 1` and `CheckGear(name) == 1`. Early levels keep `GunSetter` disabled until the first `WeaponPickUp`.

## Enemies
- `EnemyTracker.GetCurrentEnemies()` returns the active, non-dead `EnemyIdentifier`s
- `EnemyIdentifier`: `enemyType` (`EnemyType` enum, values 0–42), `health`, `dead`, `blessed`, `GetCenter()`

## Stats and levels
- `StatsManager`: `kills`, `stylePoints`, `seconds`, `restarts`, `infoSent` (true once the level-end screen has sent results), `Restart()` (respawn at checkpoint, or reload if there is none)
  - Campaign fields: `levelNumber` (mission number), `currentCheckPoint`, `timer`, `levelStarted`, `timeRanks`, `killRanks`, `styleRanks`
- `SceneHelper.LoadScene(name)`, `SceneHelper.CurrentScene`, `SceneHelper.PendingScene` (non-null while loading)
  - The loading coroutine disables every MonoBehaviour outside `DontDestroyOnLoad`. The BepInEx manager object lives in `DontDestroyOnLoad`, so the plugin keeps running.
- Scene names: `"Main Menu"`, `"Endless"` (Cyber Grind), `"Level 0-1"`, `"Level 0-2"`, ...

## Cyber Grind: `EndlessGrid`
- `currentWave`, `tempEnemyAmount`; private `anw` (`ActivateNextWave`) with `deadEnemies`
- Waves start when the player enters the grid's trigger (`OnTriggerEnter`)

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
- Sets `Time.timeScale = timeScale * timeScaleModifier` and uses 0 for hitstop. The mod leaves `timeScale` alone and controls game speed with `Time.captureDeltaTime` instead.

## Leaderboards: `LeaderboardController`
- `SubmitCyberGrindScore`, `SubmitLevelScore`, `SubmitFishSize` are always blocked by `SafetyPatches`.

## Layers
- `LayerMaskDefaults.Get(LMD.Environment)` is used for rays and enemy visibility checks.
