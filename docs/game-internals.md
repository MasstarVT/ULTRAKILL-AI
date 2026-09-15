# Game internals used by the mod

Verified against the Steam build using Unity 2022.3.29 (Mono), decompiled with ILSpy (`ilspycmd`).
When a game update breaks the mod, check these first. Everything game-specific is in
`ObservationBuilder.cs`, `ActionInjector.cs` and `EpisodeController.cs`.

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
- `currentSlotIndex`, `currentVariationIndex`, `slot1..slot6` lists

## Enemies
- `EnemyTracker.GetCurrentEnemies()` returns the active, non-dead `EnemyIdentifier`s
- `EnemyIdentifier`: `enemyType` (`EnemyType` enum, values 0–42), `health`, `dead`, `blessed`, `GetCenter()`

## Stats and levels
- `StatsManager`: `kills`, `stylePoints`, `seconds`, `restarts`, `infoSent` (true once the level-end screen has sent results), `Restart()` (respawn at checkpoint, or reload if there is none)
- `SceneHelper.LoadScene(name)`, `SceneHelper.CurrentScene`, `SceneHelper.PendingScene` (non-null while loading)
  - The loading coroutine disables every MonoBehaviour outside `DontDestroyOnLoad`. The BepInEx manager object lives in `DontDestroyOnLoad`, so the plugin keeps running.
- Scene names: `"Main Menu"`, `"Endless"` (Cyber Grind), `"Level 0-1"`, `"Level 0-2"`, ...

## Cyber Grind: `EndlessGrid`
- `currentWave`, `tempEnemyAmount`; private `anw` (`ActivateNextWave`) with `deadEnemies`
- Waves start when the player enters the grid's trigger (`OnTriggerEnter`)

## Time: `TimeController`
- Sets `Time.timeScale = timeScale * timeScaleModifier` and uses 0 for hitstop. The mod leaves `timeScale` alone and controls game speed with `Time.captureDeltaTime` instead.

## Leaderboards: `LeaderboardController`
- `SubmitCyberGrindScore`, `SubmitLevelScore`, `SubmitFishSize` are always blocked by `SafetyPatches`.

## Layers
- `LayerMaskDefaults.Get(LMD.Environment)` is used for rays and enemy visibility checks.
