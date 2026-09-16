# ULTRAKILL AI

Reinforcement-learning agent for ULTRAKILL (Cyber Grind + campaign). Repo: github.com/MasstarVT/ULTRAKILL-AI.

## Workflow rules
- **Always push to GitHub** after completing a change: commit, then `git push origin main` (remote: https://github.com/MasstarVT/ULTRAKILL-AI).
- **Always update this CLAUDE.md** as part of every change so it reflects the current state of the project (structure, setup, commands, conventions, decisions).
- Never commit game assemblies or decompiled game code (`.gitignore` covers `*.dll`, `decompiled/`).
- Checkpoints under `python/models/` are committed (since 2026-09-15, so a run can move between machines). Each is ~12 MB; commit `latest.zip` after a training session, and prune old `ckpt_*` files before committing if a run produces many.
- Update `times.md` whenever a training generation finishes (instructions are in an HTML comment at the bottom of that file).

## Layout
- `times.md`: the AI's level-time leaderboard (best time per level plus per-generation history). No entries yet.
- `mod/UltrakillAIBridge/`: BepInEx 5 plugin (C#, netstandard2.1).
  - `Plugin.cs`: entry point and config (port 47800, panic key F8).
  - `Net/BridgeServer.cs`: TCP server, newline JSON.
  - `Env/EpisodeController.cs`: lockstep, resets, time settings.
  - `Act/ActionInjector.cs`: virtual Input System keyboard and mouse; camera look via `CameraController.rotationX/Y`.
  - `Obs/ObservationBuilder.cs`: raw game-state snapshot.
  - `Env/SafetyPatches.cs`: blocks leaderboard submissions.
  - `Env/TimePatches.cs`: frame-based hitstop during lockstep.
  - `Env/BackgroundPatches.cs`: keeps the cursor free and audio muted while the AI has control.
  - `Env/TrainingSpeed.cs`: soft death (Harmony prefix on `NewMovement.GetHurt` heals instead of a lethal hit, counted in obs `player.soft_deaths`), camera disabling, and enemy Animators set to `AlwaysAnimate`.
  - `Env/InstancePatches.cs`: training instances (`-aibridge-port N`) open prefs read-only and skip prefs and save writes; save writes are also skipped whenever the AI has control.
- `mod/GamePaths.props`: local game path (gitignored; copy from `.example`). Build copies the DLL into `<game>/BepInEx/plugins/UltrakillAIBridge/`.
- `python/ultrakill_ai/`:
  - `protocol.py`: socket client.
  - `env.py`: `UltrakillEnv` / `EnvConfig` (`pitch_limit_deg` keeps the camera near level).
  - `spaces.py`: 448-dim obs packing, `MultiDiscrete` actions.
  - `rewards.py`: reward weights and computation; `aim_errors` gives the 3-D, yaw and pitch angles off an enemy, `horizon_elevation` the enemy elevation above the horizontal (diagnostics, convention-free).
  - `routes.py`: campaign route tracking.
  - `progress.py`: `ProgressCallback`, which writes live training stats to `runs/<run_name>/status.json` (atomic, every 2 s; `state` running/finished/stopped).
- `python/scripts/`: `bridge_test.py`, `random_agent.py`, `record_route.py`, `train.py` (PPO / RecurrentPPO, `--num-envs` uses SubprocVecEnv), `eval.py`, `games.py` (launch/tile/status/stop training instances), `dashboard.py` (Tkinter live view of `status.json`).
- `python/ultrakill_ai/windows.py`: monitor work-area lookup shared by `games.py` and `dashboard.py`.
- `python/tests/test_progress.py`: `ProgressCallback` and dashboard smoke tests against a fake env (no game needed).
- `python/tests/test_aim.py`: aim-reward geometry (the yaw/pitch split) and the pitch clamp (no game needed).
- `python/configs/`: `cybergrind.yaml`, `campaign_0-1.yaml`.
- `docs/protocol.md`: the socket protocol.
- `docs/game-internals.md`: game classes and fields the mod relies on (check after game updates).
- `docs/superpowers/specs/`: approved design specs. `2026-09-16-campaign-foundation-design.md` is the campaign design (not built yet).
- `docs/superpowers/plans/`: implementation plans. `2026-09-16-campaign-foundation.md` is the 16-task plan for the campaign foundation and the 0-1 pilot (Tasks 0-13 dry-run in a scratch copy: mod builds clean, all tests pass).
- `.tools/` (gitignored): local `ilspycmd` install used to regenerate `decompiled/`.

## Commands
- Build and install the mod: `cd mod/UltrakillAIBridge && dotnet build -c Release`. The game must be closed, or the DLL is locked.
- **Which checkpoint to resume from:** `best.zip` (see `best.json` for its score and source). `latest.zip` only
  updates on a GRACEFUL stop (Ctrl+C); every hard kill leaves it stale, and it sat at 1.18M steps for a whole
  night while the run reached 6.5M. Helpers to keep running alongside training, both read-only and safe:
  `python scripts/poll_status.py` (metrics to `runs/<run>/metrics_log.csv`) and `python scripts/keep_best.py`
  (maintains `best.zip`, warns when the current policy falls more than 15% below it).
- Python env: `cd python && .venv\Scripts\activate` (created with `pip install torch` + `pip install -e .`).
- Bridge check (game open, in a level): `python scripts/bridge_test.py --drive`.
- Parallel training:
  1. `python scripts/games.py launch --count 5` (monitor 3 by default; BepInEx supports at most 5 copies)
  2. `python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind_ppo_v2/best.zip` (`num_envs` 5 in config; `timesteps` is the run total, so resuming trains only the rest; drop `--resume` for a fresh run, and give it a new `run_name` so `status.json` does not inherit the old episodes)
  3. `python scripts/games.py stop`
  - `python scripts/games.py status` is safe during training (it reads netstat, it does not connect).
- TensorBoard: `tensorboard --logdir runs`.
- Live dashboard: `python scripts/dashboard.py` (newest run) or `--run cybergrind_ppo_v2`; opens on monitor 3 below the game row (`--monitor`, `--reserve-top`); `--smoke-test` renders once and exits.
- Tests (no game): `python tests/test_progress.py` and `python tests/test_aim.py` (pytest is not installed; the files also work under pytest).

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
- **Training speed findings** (random actions, 4–5 games):
  - The PPO update is only ~0.2 s per 2048 samples on CPU, so the GPU wouldn't help. The time is in the games and resets.
  - 60 fps / frameskip 4: 138 steps/s. 30 fps / frameskip 2 (same 15 decisions per game second; game logic is deltaTime-based): 199.
  - Rendering off: 217. Soft death: 297. Both: 328. With 5 games: 380.
  - A 6th copy fails because BepInEx opens at most `LogOutput.log` plus `.1`–`.4`.
- **Lockstep stalls:** vectorized envs step together, so a reset in one game stalls all of them. Keep resets short (teleport entry, `reset_settle_frames` 10).
- **Speed:** about 600 fps / 150 steps/s in an empty scene and about 100 steps/s with enemies (frameskip 4, RTX 5070).

## Status
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
- **Mod:** v0.4.0 (background play, training instances, teleport, soft death, rendering off). Verified in game: plugin load, handshake, Cyber Grind reset, movement/look/jump/dash, observations (enemies, waves, damage, death), leaderboard block.
- **Python:** env, training, eval and route tracking verified against a mock and partly in game.
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
- **Next steps:**
  - **Yaw is the one thing to fix.** `enemy_yaw_angle_mean` 51 deg is what keeps `on_target_frac` at 3%;
    pitch changes have been tried at 15 and 45 deg and moved nothing. Consider a yaw-only shaping term with
    a zero floor (`w * max(0, 1 - yaw_err/T)` with T near the current mean so improvement always pays),
    rather than another pitch pass.
  - Hold `ent_coef` so entropy stays near 3.5-4.0; below ~3 the policy went deterministic and regressed.
  - Watch the 1.70M rebalance against the falsifiers above; `yaw_track` then `on_target_frac` lead, kills/min follows.
  - If `on_target_frac` is still <= 0.06 at 2.18M, reset the look-head output bias (as `pitch_reset_2447005.zip` did for v1) rather than tuning weights again: a constant offset in the bias is not reachable from a reward slope that is flat across the whole sweep range.
  - `approx_kl` runs 0.029-0.031 against `target_kl` 0.02, so every update is being truncated. Worth a pass once the reward change has been judged, but not at the same time as it.
  - Raise `max_wave` as the agent improves.
- **Campaign: designed 2026-09-16, not built yet.** Goal: finish all 35 main levels (`Level 0-1` to `Level 9-2`)
  as fast as possible, learning alone. Spec: `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md`.
  - **Decisions:**
    - No human demos or recorded routes, so `routes.py` and `record_route.py` are to be retired.
    - Violent difficulty, all weapons unlocked in memory only.
    - Deaths respawn at the checkpoint inside the episode.
  - **Approach:**
    - The mod reads level structure: the real `FinalPit` exit, checkpoints, locked doors, arena clears, a NavMesh path hint and the official timer.
    - Rewards: time, milestones, per-cell novelty and path progress.
    - Episodes mostly respawn at each game's current checkpoint, so training concentrates on the frontier.
  - **Sub-projects:**
    1. Foundation + 0-1 pilot. Success bar: 50% fresh-start completion on Violent.
    2. The rest of the Prelude.
    3. Skull keys and switches.
    4. Acts 1-3.
    5. Speed and movement-tech actions.
  - **Starting weights:** Cyber Grind `best.zip`, widened to 479 inputs.
  - Cyber Grind training stays paused while the 5 games run the campaign.
  - `decompiled/` was regenerated on the second PC for this (ilspycmd 9.1.0.7988 in `.tools/`).
  - Spec amended during planning (2026-09-16): arena clears and door unlocks are position keys paid once per level
    load, `checkpoints_level` replaces `furthest_checkpoint`, room templates are filtered by `defaultRooms`
    ancestry, `level_seconds` is only reported for fresh starts, the campaign style reward is 0, and the mod blocks
    game-initiated restarts while the AI has control.
