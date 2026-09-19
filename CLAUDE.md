# ULTRAKILL AI

Reinforcement-learning agent that plays the ULTRAKILL campaign (Cyber Grind is paused). Repo:
github.com/MasstarVT/ULTRAKILL-AI. A BepInEx 5 C# mod (`mod/`) exposes the game over a TCP bridge in lockstep;
Python (`python/`) builds observations and rewards from the raw game state and trains PPO against it.

## The goal, in the user's terms
- **Finish every shipped main level, as fast as possible.** 33 of the 35 campaign levels ship in this build
  (9-1 and 9-2 do not). **Speed, not kills** — combat rewards exist only because some doors are arena-gated.
  The next tuning step is a completion bonus scaled by official time, then movement tech.
- **It learns alone.** No human demos and no recorded human routes (`routes.py` / `record_route.py` were
  deleted 2026-09-16). Route signal comes from the game's own door graph and from offline room trunks.
- **Violent now; Brutal (difficulty 4) for the later speed pass**, switched on together with the time-scaled
  completion bonus, so combat is re-learned once on the enemy set the records will be set on. Not before
  completion is reliable.
- **Training is hidden from Steam**: every training game launches with `-aibridge-nosteam` (`--steam` opts
  out). No playtime is credited for training; the user accepted that.
- **Check on the run, cheaply.** No always-on LLM monitor agents (the driver, `mem_guard.py` and
  `post_times.py --watch` restart things token-free), but the lead session checks hourly with
  `scripts/check_run.py` and acts on stalls — the user asked for that on 2026-09-18.
- **Do not promote past 0-3 until 0-1..0-3 are much faster.** Speed stages (S-rank time targets, time-scaled
  bonus) come before Level 0-4 gets a specialist.

## Workflow rules
- **Always push after a change**: commit, then `git push origin main`.
- **Keep this file under 220 lines.** A change updates at most a few lines here (layout / commands / current
  state); every measurement, incident, rationale and per-branch history goes to `docs/project-log.md`
  (append-only, dated) or the topical docs file, and is **linked, not inlined**. Never paste a long Status
  entry here.
- Never commit game assemblies or decompiled game code (`.gitignore` covers `*.dll`, `decompiled/`).
- Checkpoints under `python/models/` are committed so a run can move between machines (~12 MB each): commit
  `latest.zip` and the promoted specialists, and prune old `ckpt_*` files before committing.
- **Stage explicitly (`git add CLAUDE.md docs`), never `git add -A`**, while a run is live: the trainer
  rewrites `python/models/` continuously.
- **Times**: `python scripts/post_times.py --run <run>` posts a level to `times.md` only when it beats the row
  already there (no game, idempotent); `--watch 600 --push` is the token-free watcher, which commits
  `times.md` alone. `eval.py --record-times` is the deterministic counterpart. Instructions for the file's own
  format are in an HTML comment at the bottom of `times.md`.

## Layout — one line each; file-by-file detail in `docs/layout.md`
- `mod/UltrakillAIBridge/` — the BepInEx plugin (C#, netstandard2.1): TCP bridge (port 47800+), lockstep
  episode control, virtual-device input, raw observations, the `campaign` block (exit, checkpoints, gates,
  altars, items), and the safety / time / training-speed / instance / Steam patches.
- `mod/GamePaths.props` — local game path, gitignored; copy from `.example`.
- `python/ultrakill_ai/` — the env package: `protocol.py` (socket client), `env.py` (`UltrakillEnv`, campaign
  mode, bridge recovery), `spaces.py` (obs packing, action spaces), `rewards.py`, `campaign.py` (gate ladder,
  milestones, exploration archive, curriculum), `progress.py` (`status.json` / `episodes.jsonl`), `times.py`,
  `envlog.py`, `windows.py`, and `routes/` (14 committed room trunks plus `rung_overrides.json`, whose
  entries move, drop or — since 2026-09-18 — INSERT a rung; the generator stays the only writer).
- `python/scripts/` — `train.py`, `eval.py`, `games.py`, `campaign_driver.py` (the specialist driver),
  `supervise.py`, `mem_guard.py`, `full_run.py`, `specialists_status.py`, `poll_status.py`, `keep_best.py`,
  `post_times.py`, `dashboard.py`, `build_routes.py`, `campaign_check.py`, `skull_check.py`, `bridge_test.py`.
- `python/tests/` — 30 no-game test files, ~774 named tests, about 3 minutes for the lot.
- `python/configs/` — `specialists.yaml` (the specialist plan; **not** a training config),
  `campaign_gates_full.yaml` (the shared-run config it is pinned against), `campaign_0-1.yaml`,
  `cybergrind.yaml`, `il_records.yaml`, and the earlier campaign configs kept as rollbacks.
- `python/models/` (committed) and `python/runs/` (gitignored) — checkpoints, and per-run state:
  `status.json`, `episodes.jsonl`, `curriculum.json`, `env_<port>.log`, `best_runs/`, `metrics_log.csv`.
- `times.md` — the AI's level-time leaderboard plus per-generation history.
- `.tools/` (gitignored) — local `ilspycmd`, used to regenerate `decompiled/`.

## Commands in use now — one line each; full reference in `docs/commands.md`
- Python env: `cd python && .venv\Scripts\activate`.
- Build and install the mod: `cd mod/UltrakillAIBridge && dotnet build -c Release` (the game must be closed).
  Compile only, while games run: add `-p:InstallPlugin=false`.
- All no-game tests, from `python/`: `Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python
  $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }`.
- **The live trainer is the specialist driver** (`scripts/campaign_driver.py`): start it detached through
  `runs/start_driver.cmd`, which carries **no flags** — `driver_state.json` decides the stage, and `--start-at`
  is refused once a stage has run. `--dry-run` decides and exits. Pause: `New-Item runs\specialists\DRIVER_PAUSE`.
- Watch it: `python scripts/check_run.py` (one cheap health reading with an ALERTS line; exit 1 = look),
  `python scripts/specialists_status.py` (read-only), and per stage
  `python scripts/dashboard.py --run spec_0-3 --monitor 1`.
- Memory guard, required beside the driver:
  `python scripts/mem_guard.py --run spec_0-3 --game-limit-gb 2.5 --commit-limit-frac 0.93`.
- Crash supervisor — for a SHARED (non-specialist) run only, never beside the driver:
  `python scripts/supervise.py --run <run> --config <cfg> --count 12 --monitor 1`; `--dry-run` first; pause
  with `runs/<run>/SUPERVISOR_PAUSE`.
- Games: `python scripts/games.py status` is safe during training (netstat, no connection);
  `python scripts/games.py relaunch --port 47808` restarts ONE instance; `launch --count N` and `stop` are for
  when no run is live.
- Times: `python scripts/post_times.py --run spec_0-3 [--watch 600 --push]`.
- Chain the promoted specialists over one game (never a trainer's port):
  `games.py launch --count 1 --monitor 1`, `python scripts/full_run.py --record-times`, `games.py stop`.
- Eval one policy: `python scripts/eval.py models/specialists/Level_0-1.zip --level "Level 0-1" --episodes 10`
  (`--record-times` to post the fastest completion).
- In-game checks on one private game: `python scripts/campaign_check.py --level "Level 0-1"` (check 5, the
  exit, is a known standing FAIL — exit code 1 is expected) and `python scripts/skull_check.py --port 47812 …`.
- Route data, no game and safe beside a live run: `python scripts/build_routes.py --validate`, then
  `python tests/test_route_files.py`. Rerun after every game update.
- TensorBoard: `tensorboard --logdir runs`.

## Safety-critical gotchas — the rest, with the measurements, in `docs/gotchas.md`
- **Never connect anything to a training bridge port.** The bridge is single-client: any new connection drops
  the trainer's and kills that worker. Read `netstat` (`games.py status`) and files, never a socket.
- **Never run `games.py launch` or `games.py stop` while a run is live** — both call `stop_all()` and take down
  all twelve games. Repair one port with `relaunch --port N`, or hand-start a private instance on **47812**.
- **Create the pause file before ANY planned pause**: `runs/specialists/DRIVER_PAUSE` (driver) or
  `runs/<run>/SUPERVISOR_PAUSE` (supervisor). A deliberate Ctrl+C is indistinguishable from a crash and it
  will relaunch the games and the trainer underneath you.
- **Never run `supervise.py` and `campaign_driver.py` together** — they fight over the games and over which
  trainer should exist.
- **`games.py stop` no longer ends the trainer** (each env retries its lost game for minutes), so `latest.zip`
  is NOT written: resume from the newest `ckpt_*_steps.zip`, at most 50k steps old. The driver and supervisor
  pick the file with the most steps by themselves.
- **The mod DLL is locked while games run** — use `-p:InstallPlugin=false` to compile without installing.
- **Memory is the binding constraint.** Twelve games plus the trainer sit at ~**66-75% of a 60 GB commit
  limit** (all Python ~4 GB since today's fix, was 15 GB), and the games leak ~**790 MB per game-hour** from
  ~1 GB at boot. `mem_guard.py` recycles a game at 2.5 GB and must be running; any offline analysis script must
  **stream** and stay under ~4 GB. See `docs/notes/2026-09-18-memory.md`.
- **The twelve games are CHILD processes of the driver's tree** — never `taskkill /T` the driver or its
  `start_driver.cmd` wrapper, or they die with it. Enumerate by command line (`Get-CimInstance Win32_Process`)
  and `Stop-Process -Id` each python pid individually; no `ULTRAKILL.exe` should ever be killed.
- **A replay proves a mechanism only for the policy that was recorded.** An offline replay "proved" target
  patience inert on 0-1; live, on the current policy, it halved 0-1's completion rate.
- **A transform is not a place you can stand.** A `FinalPit`'s transform sits 62.2 m below its own room's
  floor, and a room centroid can be through a wall. Aim at ground (`exit.ground_pos`, `altars[].aim_pos`) and
  always ask what the player would be standing on if they reached the target.
- **Never judge a policy edit from an offline probe alone** — confirm against live `status.json`; revert if the
  headline metric is worse at the next two checks; change one thing at a time and give it >= 400k steps.
- **In a git worktree, set `PYTHONPATH`** to that worktree's `python/`, or the editable install silently makes
  every test measure the main tree.
- **PowerShell rejects a long combined command** that mixes `Remove-Item` with a `cmd /c` argument list — run
  those as separate commands.

## Key design decisions
- **Lockstep:** the mod blocks Unity's main thread between steps. `Time.captureDeltaTime = 1/60` fixes game time per frame, and uncapped FPS makes training faster than real time. Game speed is not controlled through `Time.timeScale`, which `TimeController` owns for hitstop.
- **Input:** injected through virtual Input System devices rather than Harmony patches on `InputActionState` getters, which Mono may inline.
- **Rewards and observations:** computed in Python from raw mod data, so tuning needs no mod rebuild.

## Current state (2026-09-18)
- **Per-level specialists: one policy per level, trained sequentially** by `scripts/campaign_driver.py`, which
  replaces `supervise.py` and supervises each stage with the supervisor's own code. The shared multi-level run
  `campaign_gates` was stopped at 17,002,318 steps because the mixture thrashed — whichever level got the
  fresh-start share improved while the others regressed.
- A stage ends when that level's fresh completion rate reaches 0.5 over its last 50 fresh episodes (window
  >= 30, latched) **and** 300k steps have passed since `keep_best.py` last moved `best.zip`; else at the 6M-step
  cap, recorded `"unfinished"`. It then promotes `best.zip` to `models/specialists/<level>.zip` with a JSON
  sidecar, and the next stage resumes from that file. Games are not relaunched between stages.
- A **speed stage** (`{level, kind: speed}`) adds the clock: its `median_time_50` — never `best_time`, a
  lifetime minimum one lucky load sets for good — must also reach the level's own S-rank time (`target_scale:
  1.0`, since the gate is a median), at rate 0.4, cap 8M. It does NOT overwrite the level's specialist unless
  it ends `"done"`. The **hold line** `hold_before: "Level 0-4"` stops the ladder there and round-robins
  0-1..0-3 for as long as it takes (`max_rounds: 0`): a `HELD` exit would idle twelve games nobody watches.
  Spec and the review that shaped it: `docs/superpowers/specs/2026-09-18-speed-stages.md`.
- **The 0-3 stage ended at its 6M cap** (24,757,714 steps, recorded `"unfinished"`, 0 fresh completions); the
  driver is now on the **`Level 0-1` speed** stage, 12 games on ports 47800-47811, `mem_guard.py` alongside.
  Round-robin order means the next 0-3 window is **round 2**, after the 0-1 and 0-2 speed stages.
- **Parking is off on a collapsed-ladder trunk load** (2026-09-18, `patience_active` + `_prefers_route`): on
  0-3 a park aimed the agent 27.3 m back DOWN off Floor 2 and `gate_approach` paid +3.87 for descents over 13
  recorded rollouts. The proposed waypoint move was REFUSED — it fails the generator's own `SEP`/`co_credit`
  guards and credits earlier and lower. Baseline, live signal and revert trigger: `docs/project-log.md`.
- 0-3's route went 4 rungs -> **6** on 2026-09-18 (two `insert_after` waypoints on the main-room spiral ramp
  plus the Floor-2 rung moved off its centroid): the hallway -> Floor 2 leg was 76 degrees, so the target
  vector carried no heading and 219 of 264 fresh episodes wedged at y ~20 with zero completions in 4.3M
  steps. Steepest leg now 43.2 degrees; the ladder is still 5 hops deep, so the no-finish pay ceiling is
  unchanged at 75. Baseline, live signal and revert triggers: `docs/project-log.md`, 2026-09-18.
- Promoted so far: **0-1** (fresh rate 0.48 over 50, best 4:03.4) and **0-2** (0.72, best 2:19.5). Both are
  still **slower** than the shared run's leaderboard rows below — which is exactly why nothing promotes past
  0-3 until the speed work lands.
- `times.md` leaderboard, all Violent, all set by `campaign_gates`: **0-1 03:03.628 (A)**, **0-2 02:07.927
  (S)**, **0-3 04:23.904 (B)**. The human reference playthrough of 0-1 is **146.58 s**; human IL records, the
  real speed targets, are in `docs/il-records.md`.
- 0-3's trunk shipped BOTH branches of the level's fork chained in series, and a 7-rung detour tour paid 105
  against a completion's 100. Trimmed to the 4 rungs every recorded completion used (2026-09-18, `route-0-3`);
  `rung_overrides.json` now carries `drop` entries; trainer bounced onto it at **21,701,566** steps. Judge at
  +400k on wing endings and `exit_ground_dist_min` (131.8 m at the bounce), never on `gates_reached` — see the log.
- In flight: **`speed-stages` is merged** (2026-09-18, after an adversarial review that fixed one blocker and
  five majors — §9 of the spec, log entry 2026-09-18) and the driver runs it; no speed stage has been TRAINED
  yet, so none of the speed mechanism is validated in game. Also: guard T's mid-name-fork blind spot still
  unfixed campaign-wide; a **0-3 mid-level wall probe** still to run; and the memory work above — another
  engineer owns `python/`, `mod/` and `docs/notes/2026-09-18-memory.md`.
- Numbers a newcomer needs: observation **479** floats; campaign action space **12 dimensions / 45 logits**
  (Cyber Grind 11 / 42, and look mode 1 is deliberately campaign-only); mod **v0.7.2** installed; **33 of 35**
  levels ship; Violent, with all weapons unlocked in memory for AI runs only.
- Machines (backups and migration steps in `docs/machines.md`). This PC:
  `C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL`, Ryzen 9 3900X (12C/24T), 32 GB, RTX 2080 SUPER,
  **one display** — pass `--monitor 1` everywhere. Original PC:
  `E:\SteamLibrary\steamapps\common\ULTRAKILL`, where games tile on monitor 3 by default. Both run Unity
  2022.3.29 Mono with BepInEx 5.4.23.5; `runs/` is gitignored and does not travel, `python/models/` does.

## Docs index
- `docs/project-log.md` — the whole former Status section: every dated incident, measurement, branch history
  and decision, chronological and verbatim. **Append new entries here**, dated.
- `docs/gotchas.md` — every gotcha found in the live game, verbatim.
- `docs/layout.md` — the file-by-file layout, including the test-by-test list.
- `docs/commands.md` — the full command reference, including the activation checklists.
- `docs/machines.md` — both machines' state, and moving the project between them.
- `docs/protocol.md` — the socket protocol. `docs/game-internals.md` — game classes and fields the mod relies
  on (check after game updates).
- `docs/level-survey.md` — all 35 levels parsed offline: exits, checkpoints, door graphs, gate ladders,
  altars, arenas, bosses, the tier table, and section 9's route-coverage appendix.
- `docs/il-records.md` — human individual-level speedrun records, the speed targets (`configs/il_records.yaml`
  is the machine-readable copy).
- `docs/notes/2026-09-18-memory.md` — the memory investigation (owned by another engineer).
- `docs/superpowers/specs/` — approved design specs (campaign foundation, gates/un-wedge/look modes,
  multi-level and skull gates, ladder patience and exit guard, route fallback and boss levels).
  `docs/superpowers/plans/` — the implementation plans and integration checklists.
