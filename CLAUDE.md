# ULTRAKILL AI

Reinforcement-learning agent for ULTRAKILL (Cyber Grind + campaign). Repo: github.com/MasstarVT/ULTRAKILL-AI.

## Workflow rules
- After every change: update this file to reflect the current state, then commit and push to `origin main`.
- Never commit game assemblies or decompiled game code (`.gitignore` covers `*.dll`, `decompiled/`).

## Layout
- `mod/UltrakillAIBridge/`: BepInEx 5 plugin (C#, netstandard2.1).
  - `Plugin.cs`: entry point and config (port 47800, panic key F8).
  - `Net/BridgeServer.cs`: TCP server, newline JSON.
  - `Env/EpisodeController.cs`: lockstep, resets, time settings.
  - `Act/ActionInjector.cs`: virtual Input System keyboard and mouse; camera look via `CameraController.rotationX/Y`.
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot.
  - `Env/SafetyPatches.cs`: blocks leaderboard submissions.
- `mod/GamePaths.props`: local game path (gitignored; copy from `.example`). Build copies the DLL into `<game>/BepInEx/plugins/UltrakillAIBridge/`.
- `python/ultrakill_ai/`:
  - `protocol.py`: socket client.
  - `env.py`: `UltrakillEnv` / `EnvConfig`.
  - `spaces.py`: 448-dim obs packing, `MultiDiscrete` actions.
  - `rewards.py`: reward weights and computation.
  - `routes.py`: campaign route tracking.
- `python/scripts/`: `bridge_test.py`, `random_agent.py`, `record_route.py`, `train.py` (PPO / RecurrentPPO), `eval.py`.
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml`.
- `docs/protocol.md`: the socket protocol.
- `docs/game-internals.md`: game classes and fields the mod relies on (check after game updates).

## Commands
- Build and install the mod: `cd mod/UltrakillAIBridge && dotnet build -c Release`. The game must be closed, or the DLL is locked.
- Python env: `cd python && .venv\Scripts\activate` (created with `pip install torch` + `pip install -e .`).
- Bridge check (game open, in a level): `python scripts/bridge_test.py --drive`.
- Train: `python scripts/train.py --config configs/cybergrind.yaml`; TensorBoard: `tensorboard --logdir runs`.

## Key design decisions
- **Lockstep:** the mod blocks Unity's main thread between steps. `Time.captureDeltaTime = 1/60` fixes game time per frame, and uncapped FPS makes training faster than real time. Game speed is not controlled through `Time.timeScale`, which `TimeController` owns for hitstop.
- **Input:** injected through virtual Input System devices rather than Harmony patches on `InputActionState` getters, which Mono may inline.
- **Rewards and observations:** computed in Python from raw mod data, so tuning needs no mod rebuild.

## Local machine state
- **Game:** `E:\SteamLibrary\steamapps\common\ULTRAKILL`, Unity 2022.3.29 Mono.
- **Mods:** BepInEx 5.4.23.5 installed 2026-09-15; UltrakillAIBridge plugin installed.
- **Save backup:** `C:\Users\tyler\Documents\ULTRAKILL-Saves-Backup-2026-09-15`.

## Status
- **Mod:** builds.
- **Python:** env, training, eval and route tracking verified against a mock of the mod protocol.
- **Not yet verified in the real game:**
  - Plugin load
  - Input injection
  - Resets
  - Timing

  Next step: launch the game, check `BepInEx/LogOutput.log`, then run `bridge_test.py --drive`.
