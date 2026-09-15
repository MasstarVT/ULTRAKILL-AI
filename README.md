# ULTRAKILL AI

A reinforcement-learning agent that teaches itself to play [ULTRAKILL](https://store.steampowered.com/app/1229490/ULTRAKILL/), starting in Cyber Grind and moving on to campaign levels.

```
 ULTRAKILL + BepInEx mod (C#)                Python
 ┌───────────────────────────────┐   TCP    ┌──────────────────────────────┐
 │ ObservationBuilder  ──obs────▶│ ───────▶ │ UltrakillEnv (Gymnasium)     │
 │ ActionInjector     ◀─action───│ ◀─────── │ PPO / RecurrentPPO (SB3)     │
 │ EpisodeController (lockstep,  │ localhost│ rewards, routes, TensorBoard │
 │   resets, game speed)         │          │                              │
 └───────────────────────────────┘          └──────────────────────────────┘
```

- **Mod (`mod/UltrakillAIBridge`):** a BepInEx plugin.
  - Reads game state: player, enemies, raycasts, Cyber Grind waves, level stats.
  - Plays through a virtual keyboard and mouse that use your key bindings.
  - Runs the game in lockstep with Python, so a slow policy never costs reaction time and training can run faster than real time.
  - Blocks Steam leaderboard submissions.
- **Python (`python/`):** a Gymnasium env, reward shaping, campaign route tracking, and training/eval scripts built on Stable-Baselines3.

All learning logic lives in Python, so tuning rewards or observations never needs a mod rebuild.

## Setup

### 1. Back up your save
The AI plays on whatever save slot is active. Copy `ULTRAKILL/Saves` somewhere safe, and pick an empty slot in the game's menu for AI runs.

### 2. Install BepInEx 5
1. Download **BepInEx 5.4.x (win x64)** from the [BepInEx releases](https://github.com/BepInEx/BepInEx/releases).
2. Extract it into the ULTRAKILL folder, next to `ULTRAKILL.exe`.
3. Launch the game once so BepInEx creates its folders, then quit.

### 3. Build the mod
Requires the .NET SDK (6 or newer).

```bash
cd mod
copy GamePaths.props.example GamePaths.props   # then edit the path to your ULTRAKILL folder
cd UltrakillAIBridge
dotnet build -c Release
```

The build copies `UltrakillAIBridge.dll` into `BepInEx/plugins/UltrakillAIBridge/`. Launch the game, then check `BepInEx/LogOutput.log` for `Listening on 127.0.0.1:47800`.

### 4. Python environment
Python 3.10+.

```bash
cd python
python -m venv .venv
.venv\Scripts\activate
pip install torch            # CPU build is fine for MLP policies
pip install -e .
```

The CPU build of PyTorch is enough: small MLP policies train about as fast on CPU, and the GPU stays free for the game. For the CUDA build (RTX 50-series needs CUDA 12.8+):
`pip install torch --index-url https://download.pytorch.org/whl/cu128`

## Usage

Run everything from the `python/` folder with the game open.

**Check the bridge.** Load any level first.
```bash
python scripts/bridge_test.py              # prints what the mod sees
python scripts/bridge_test.py --drive      # drives the player through a scripted sequence
```

**Smoke test with random actions.**
```bash
python scripts/random_agent.py --mode cybergrind --episodes 3
```

**Train on Cyber Grind.**
```bash
python scripts/train.py --config configs/cybergrind.yaml
tensorboard --logdir runs
```

**Train faster with several games at once.** `games.py` starts training copies of the game on ports 47800+, in small windows on monitor 3 (`--monitor`), at below-normal CPU priority. They never write your settings or saves.
```bash
python scripts/games.py launch --count 4
python scripts/train.py --config configs/cybergrind.yaml --num-envs 4
python scripts/games.py stop
```

**Train on a campaign level.** Record a route by playing the level yourself first.
```bash
python scripts/record_route.py --level "Level 0-1"
python scripts/train.py --config configs/campaign_0-1.yaml
```

**Watch a trained agent.**
```bash
python scripts/eval.py models/cybergrind_ppo/latest.zip --realtime
```

**Take back control:** press **F8** in game to disconnect the Python client.

## How it works

- **Observations:** the policy gets a fixed 448-float vector.
  - Player: velocity, HP, stamina, grounded/sliding, look angles, weapon slot.
  - The 8 nearest enemies: camera-space position, distance, health fraction, visibility, type.
  - 16 wall-distance rays and 8 pit-detection rays.
  - Cyber Grind wave info.
  - Campaign only: direction to the next route waypoint.
- **Actions:** a `MultiDiscrete` space.
  - Move forward/back and strafe; jump, dash, slide, fire1, fire2, punch; weapon slot.
  - Yaw and pitch in nonlinear degree bins, for both fine aim and fast turns.
- **Rewards** (`python/ultrakill_ai/rewards.py`):
  - Positive: damage dealt (normalised per enemy), kills, style, waves cleared.
  - Negative: damage taken, death.
  - Campaign: progress along the recorded route, level completion, a penalty for getting stuck.
  - Early training can also use an optional aim-assist shaping bonus.
- **Curriculum:**
  1. Cyber Grind with `max_wave: 3`. Raise it as the agent improves, then set 0 for full runs.
  2. Campaign 0-1 with checkpoint resets.
  3. More levels.

More detail: [docs/protocol.md](docs/protocol.md) covers the socket protocol, and [docs/game-internals.md](docs/game-internals.md) lists which game classes the mod touches, which helps after game updates.

## Notes

- **Background play:** while the AI has control, the game shrinks to a 640x360 window, stays muted and leaves your mouse free, so you can keep using your PC. Your display settings and volume come back when control is released.
- **Speed:** while the AI has control the game renders uncapped, and each frame is a fixed 1/60 s of game time. Training speed depends on how fast your PC renders, so lowering resolution and graphics settings helps.
- **Waiting during training:** the game window stops responding while Python runs a PPO update. That's expected; it resumes on the next step.
- **Leaderboards:** submissions are always blocked while the mod is installed.
- **Game code:** this repo contains no game code or assemblies. The mod references them from your local install at build time.
