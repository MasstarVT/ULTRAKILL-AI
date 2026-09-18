# Machines and migration

Moved verbatim out of `CLAUDE.md` on 2026-09-18 (documentation restructure).

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

