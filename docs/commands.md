# Commands — the full reference

Moved verbatim out of `CLAUDE.md` on 2026-09-18 (documentation restructure): the whole Commands section,
including the activation checklists, plus the original long-form workflow rules that lived above it.
Read a date-stamped claim as of its date; `docs/project-log.md` has anything later.

## Workflow rules (the original long-form text)
- **Always push to GitHub** after completing a change: commit, then `git push origin main` (remote: https://github.com/MasstarVT/ULTRAKILL-AI).
- **Always update this CLAUDE.md** as part of every change so it reflects the current state of the project (structure, setup, commands, conventions, decisions).
- Never commit game assemblies or decompiled game code (`.gitignore` covers `*.dll`, `decompiled/`).
- Checkpoints under `python/models/` are committed (since 2026-09-15, so a run can move between machines). Each is ~12 MB; commit `latest.zip` after a training session, and prune old `ckpt_*` files before committing if a run produces many.
- Update `times.md` whenever a training generation finishes (instructions are in an HTML comment at the bottom of that file). For campaign levels `python scripts/eval.py <model> --level "Level 0-1" --record-times` does it: the fastest completion goes into the generation history, and onto the leaderboard when it is a record.
- **Post times while training too** (user instruction, 2026-09-17): `python scripts/post_times.py --run campaign_gates` reads the run's `best_runs/<level>.json` (fastest fresh-start completion, official time and rank) plus `episodes.jsonl` for the step count, and posts a level only when it beats the row `times.md` already holds, so it is safe to run on every monitoring check and needs no game. After it posts, commit `times.md` and push. **Token-free watcher (running since 2026-09-17, after the user stopped the LLM monitor):** `python scripts/post_times.py --run campaign_gates --watch 600 --push` checks every 10 minutes and commits `times.md` ALONE (`git add -- times.md`, never `-A`) and pushes, rebasing once if main moved; log `runs/campaign_gates_times.log`. Together with `supervise.py` this means the run needs no monitoring agent: **do not start LLM monitors unless the user asks** (they stopped monitor #6 by hand). Rows it writes are marked `training episode (sampled actions), fresh start`; an `eval.py --record-times` row is the deterministic counterpart.


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
- **Training is hidden from Steam** (user instruction, 2026-09-17; mod v0.7.1). `games.py launch` and
  `relaunch`, and every game `supervise.py` starts, add `-aibridge-nosteam`, so a training copy never
  registers with Steam: no playtime is credited and Steam does not list ULTRAKILL as running. `launch` prints
  `Launching N instance(s) HIDDEN from Steam` and the handshake carries `steam_hidden`.
  - **`--steam` is the opt-out**, on `games.py launch`, `games.py relaunch` and `supervise.py`. `--no-steam`
    is accepted everywhere as a **no-op alias** (it was the opt-IN for about an hour on 2026-09-17), so an old
    command line still means hidden rather than failing to parse.
  - `launch` writes the per-port flags to `python/runs/instance_flags.json` and `relaunch_one` reads them back,
    so **the env's own-game recovery rebuilds the same command line**: a recovery cannot silently change
    whether Steam can see that game. It has to be a file — the recovery runs in an SB3 worker that never
    called `launch`, in another process. A no-game test must never write that file: `test_freeze_recovery.py`
    drives the real `games.launch` and now stubs `record_instance_flags` for exactly this reason.
  - **To switch a running setup** (the config lives on the SUPERVISOR's command line, so this restarts it):
    `New-Item runs\campaign_gates\SUPERVISOR_PAUSE` → stop `supervise.py` → kill the `train.py` /
    multiprocessing / `poll_status.py` processes → `python scripts/games.py stop` →
    `Remove-Item runs\campaign_gates\SUPERVISOR_PAUSE` → start the supervisor again, with or without
    `--steam`. Nothing hides or unhides a game in place; only a relaunch does.
- **Campaign training — the live run. `configs/campaign_gates_main.yaml`, run `campaign_gates`, TWELVE games**
  (since integration pause #2, 2026-09-17; it uses every game, so Cyber Grind stays paused). This is the same run
  that started on 0-1: same `run_name`, same `models/campaign_gates/`, same weights, now a **curriculum** over
  the level survey's Tier A + Tier B — 0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 4-1 — with
  `ent_coef` 0.004, `max_steps` 12000 and the skull weights live.
  `configs/campaign_gates_prelude.yaml` is the previous config, kept as history and as the partial rollback.
  1. `python scripts/games.py launch --count 12 --monitor 1` (needs `[Logging.Disk] Enabled = false`; see the
     instance-count gotcha. `games.py launch` refuses `--count > 5` while disk logging is on, and says why.)
     **Then check every copy is actually booted before starting the trainer**, with `games.py status`: it now
     prints the pid and working set behind each listening port, and a copy at ~56 MB instead of ~1 GB will
     answer every reset `unknown scene`. Restart just that one with
     `python scripts/games.py relaunch --port 47808` — never `launch`/`stop`, which take down all twelve.
     `supervise.py` does this gate automatically on every restart it makes.
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
- **The crash supervisor — keep it running, and pause it before you pause training.** The run died on its own on
  2026-09-17 (a worker's `TimeoutError` on a level reset) and nothing restarted it for hours. `supervise.py`
  is the standing fix; start it the same detached way, fifth:
  ```powershell
  Start-Process -FilePath cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized `
    -ArgumentList '/c', '"F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python.exe" -u scripts/supervise.py --run campaign_gates --config configs/campaign_gates_main.yaml --count 12 --monitor 1 >> runs\campaign_gates_supervisor.log 2>&1'
  ```
  `python scripts/supervise.py ... --dry-run` reports the health decision and exits without changing anything —
  run that first, and expect one `healthy: trainer pid N, ... steps, status Ns old` line. It appends to
  `runs/campaign_gates_supervisor.log`, logs once per change of situation and one heartbeat an hour.
  - **STOP THE SUPERVISOR, OR CREATE THE PAUSE FILE, BEFORE ANY PLANNED PAUSE.** A deliberate Ctrl+C, a
    `games.py stop`, an integration pause — all of them look exactly like a crash to it, and it will stop your
    games and start the trainer again underneath you. Either kill it, or:
    ```powershell
    New-Item runs\campaign_gates\SUPERVISOR_PAUSE       # from python/; it then does nothing at all
    Remove-Item runs\campaign_gates\SUPERVISOR_PAUSE    # when the pause is over
    ```
    While that file exists it does not restart the trainer, does not touch the games and does not start the
    helpers. It logs the pause once and logs once when it lifts. A trainer that is mid-teardown with a fresh
    `status.json` (`state` already `stopped`) is left alone anyway, but only until the file goes stale, so the
    pause file is the only thing that holds across a whole pause.
  - It restarts `poll_status.py` and `keep_best.py` whenever they are missing, so do not count on stopping a
    helper by hand while the supervisor is up.
  - **Activating a Python-side change on the live run** (this is what branch `curriculum-progress` needs): the
    trainer only reads the code at start, so merge to `main`, then bounce it through the supervisor —
    `New-Item runs\campaign_gates\SUPERVISOR_PAUSE`, Ctrl+C the trainer and wait for `Saved ... latest.zip`,
    `Remove-Item runs\campaign_gates\SUPERVISOR_PAUSE`, and the supervisor restarts it on the new code with the
    config it was given. No games are stopped and no weights move. Confirm from the first lines of the trainer
    log: `curriculum: 30 levels, unlocked [...], weighting progress`.
  - **The supervisor's own code changes are not picked up by a trainer restart.** It is a long-lived process
    started by hand, so a change to `supervise.py` needs the supervisor itself stopped and started again — see
    the freeze-fix activation steps below.
- **Per-level specialist training — the driver (branch `specialists`).** One policy per level, trained
  sequentially on all twelve games, each initialised from the previous level's best. **It replaces
  `supervise.py` while it runs; never run both.** It supervises each stage itself, using the supervisor's own
  code, and it starts `poll_status.py`, `keep_best.py --metric campaign` and `post_times.py --watch --push` for
  the stage's run.
  ```powershell
  # FIRST launch only -- the one time --start-at/--init mean anything
  Start-Process -FilePath cmd.exe -WorkingDirectory F:\Github\ULTRAKILL-AI\python -WindowStyle Minimized `
    -ArgumentList '/c', '"F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python.exe" -u scripts/campaign_driver.py --start-at "Level 0-1" --init models\campaign_gates\best.zip --monitor 1 >> runs\specialists_driver.log 2>&1'
  # EVERY restart after that: the launcher script, which carries no flags at all
  Start-Process -FilePath "F:\Github\ULTRAKILL-AI\python\runs\start_driver.cmd" -WindowStyle Hidden
  ```
  `--start-at` and `--init` are read on the FIRST launch only; after that `runs/specialists/driver_state.json`
  decides, so a restart of the driver resumes the stage it was on rather than the top of the ladder. Run it once
  with `--dry-run` first: it reports the decision and exits, changing nothing.
  - **`runs\start_driver.cmd` is gitignored, so its exact content is recorded here.** The two flags it used to
    carry (`--start-at "Level 0-1" --init models\campaign_gates\ckpt_17002318_steps.zip`) were REMOVED on
    2026-09-18: the driver refuses `--start-at` for a stage that has already run, so the file as it stood would
    have failed any restart made after the 0-1 stage finished. Flagless, it resumes `current` and otherwise
    picks the next stage itself. Keep it a `.cmd` file: a bare `Start-Process` mangles a quoted `"Level 0-1"`.
    ```bat
    @echo off
    cd /d F:\Github\ULTRAKILL-AI\python
    "F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python.exe" -u scripts\campaign_driver.py >> runs\specialists_driver.log 2>&1
    ```
  - **`--start-at` is refused in two cases** (2026-09-18 review), because it bypasses `choose_stage`, which is
    where every other rule lives. It will not start a stage at or after the plan's `hold_before` while a stage
    in front of the line is not `"done"` — pass `--ignore-hold` to lift the line for that one start, which the
    driver logs — and it will not start a stage that has already run, which is what a restart of
    `runs/start_driver.cmd` after a `"held"` exit would otherwise do (`current` is null then, exactly as on a
    first launch); pass `--rerun-stage` to give a finished stage another round on purpose. The message names
    the flag in both cases.
  - **A `"held"` exit (code 1) is not a crash**, and since 2026-09-18 it is also not expected: both
    `max_rounds` are **0** (no cap), because a HELD exit stops the driver and leaves twelve games idle with
    nobody watching, which is strictly worse than going on training the held stages. The round robin now runs
    until the targets are met or a human lifts `hold_before`. A HELD exit therefore means something the driver
    cannot train its way out of — a speed stage whose complete stage is not `"done"`, or no specialist file —
    and the log line names the stage and the reason. Do not restart the driver into it; fix the reason, or
    retune `speed.target_scale` / add a per-level `speed.targets` override in `configs/specialists.yaml`.
    Every round's weights are still in `models/spec_<level>_speed/`.
  - **Pause it before any planned pause**, exactly as with the supervisor and for the same reason (a deliberate
    stop looks like a crash to it):
    ```powershell
    New-Item runs\specialists\DRIVER_PAUSE        # from python/; it then does nothing at all
    Remove-Item runs\specialists\DRIVER_PAUSE     # when the pause is over
    ```
  - **Watch it**: `python scripts/specialists_status.py` (stage, steps into it, rate and window, the MEDIAN
    official time over the last 50 fresh episodes beside the best one ever recorded, the hold line and what it
    is waiting on, settle left, the promoted table) and the monitor's `report.py --run spec_0-1` for the
    stage's own numbers. A speed stage promotes on the **median**, never on the best: `campaign.best_time` is
    a run-lifetime minimum that one lucky load sets for good, and on the live runs it is about half the median
    (see `docs/superpowers/specs/2026-09-18-speed-stages.md` §9a). `keep_best --metric time` scores the median
    for the same reason.
    `python scripts/dashboard.py --run spec_0-1 --monitor 1` works unchanged — a stage is a single-level run, so
    its campaign block is the one the dashboard has always drawn.
  - **Chain the specialists into a full-game run**, on ONE game and never a port a trainer is using:
    ```
    python scripts/games.py launch --count 1 --monitor 1
    python scripts/full_run.py --record-times
    python scripts/games.py stop
    ```
    It prints a per-level table (time, rank, kills, deaths) and the total of the completed times, skips and
    reports any level with no specialist yet, and with `--record-times` posts each completed level to `times.md`
    under the generation `specialists@<date>`. `--levels "Level 0-1" "Level 0-2"` runs part of the ladder and
    `--episodes N` takes the fastest of N attempts per level.
  - **ACTIVATION CHECKLIST — switching the live shared run over to the driver** (the lead picks the exact init
    file; `models/campaign_gates/best.zip` is the default choice, the newest `ckpt_*_steps.zip` the alternative
    when `best.json` is behind):
    1. `New-Item runs\campaign_gates\SUPERVISOR_PAUSE` — or the supervisor will restart the shared trainer under
       you the moment you stop it.
    2. Stop `supervise.py` (it is a long-lived process started by hand; a pause file alone is not enough if you
       want it gone), then Ctrl+C the shared trainer and wait for `Saved models\campaign_gates\latest.zip`.
    3. Kill the shared run's helpers: `poll_status.py`, `keep_best.py`, `post_times.py`, `dashboard.py`.
    4. `python scripts/games.py stop` (the driver launches its own twelve; it will not relaunch them between
       stages, only when a port is not listening).
    5. Start the driver with `--start-at "Level 0-1" --init models\campaign_gates\<best or newest ckpt>`.
    6. Confirm from `runs/specialists_driver.log`: a `STAGE 1/30 Level 0-1: run spec_0-1` line, then
       `started trainer for Level 0-1`, then `specialists_status.py` showing the stage with a rising step count.
- **Reading a stall (2026-09-17 onward).** Three files, none of which needs a game or a console:
  `runs/<run>_supervisor.log` (the decision and, before every restart, one line per port with pid, working set,
  ESTABLISHED connections and CPU cores), `runs/<run>/env_<port>.log` (that worker's own account: reset start
  and end with durations, `recover_start`/`bridge_reset`/`reconnect_failed`/`relaunch_*`/`recover_end`), and
  `runs/<run>/episodes.jsonl` (`end_reason` `bridge_reset` and the `bridge_resets` counter per episode).
  **Do not trust `runs/<run>_train.log`'s tail without checking its `LastWriteTime`** — that is exactly what
  made the 17:52 post-mortem quote a seven-hour-old traceback as current.
  - After `--max-restarts-per-hour` (3) restarts inside an hour it logs `GIVING UP` and exits non-zero: read
    `runs/campaign_gates_supervisor.log` and the train-log tail it copied in, because restarting is not the fix.
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
- Record a policy playing, one JSON line per decision (**a PRIVATE port only -- never 47800-47811**):
  `python scripts/probe_rollout.py --model <ckpt.zip> --config configs/generated/spec_0-3.yaml --port 47812
  --episodes 10 --out runs/probe_0-3`. It writes `pos`, `vel`, `hspeed`, `grounded`, `ground_ray_center`, the
  action, the reward parts and the whole target/ladder state per step, which is how the 0-3 climb was diagnosed
  in game (`docs/project-log.md`, 2026-09-18). Since 2026-09-19 it also writes the fields a TIME BUDGET needs:
  the official clock (`level_seconds`, `timer_running`), `kills`/`style`/`restarts`, `dead`/`hp`/`anti_hp`,
  the enemies on screen (`n_enemies`, `n_visible`, `nearest_dist`), `arena_alive`, `cleared_arenas`,
  `unlocked_doors`, `n_locked_doors`, and `beh` -- the per-decision diff of the env's own behaviour counters,
  which is the only way to read the APPLIED look mode and whether a shot was on target.
  `--deterministic` is argmax instead of sampling and `--scripted` drives
  by hand instead of by policy (the geometry probe). `--model` must be a **COPY** of a checkpoint, never the
  trainer's own file. Start the game by hand per the bullet above; the probe process peaks at ~330 MB.
- Campaign eval (one game on port 47800, e.g. `python scripts/games.py launch --count 1 --monitor 1`):
  `python scripts/eval.py models/campaign_gates/best.zip --level "Level 0-1" --episodes 10`. Fresh level loads,
  deterministic actions, real deaths; prints completed, official time, rank, kills, style, restarts and deaths per
  episode, then the completion count and the fastest run. It reads the exploration counts the first training game
  saved next to the model (`explore_Level_0-1_47800.npz`, printed as a cell count; 0 cells means the policy sees
  an unexplored map) and never writes them. Add `--record-times` to write the fastest completion to `times.md`.
- Live dashboard: `python scripts/dashboard.py` (newest run) or `--run cybergrind_ppo_v2`; opens on monitor 3 below the game row (`--monitor`, `--reserve-top`); `--smoke-test` renders once and exits. A campaign run replaces the Shooting panel with a Campaign panel (fresh and all-episode completion rate, best and median official time, **gates per load** and checkpoints per load with the all-episode and fresh-start means side by side, **wedged steps per episode**, a **parked/ep + exit-banished row** for the two ladder-patience mechanisms, a **look free/gate row carrying the three per-dimension entropies**, new cells, deaths, closest to the exit, the four largest reward parts), charts fresh completion % and **gates per load** instead of kills/min and wave, and lists checkpoints instead of waves per game. On branch `next-levels` a **multi-level** run adds a `levels` block (one row per unlocked level: fresh rate and window, best time, checkpoints per load, sampling weight `w` and, since branch `curriculum-progress`, the progress score `prog` the weighting rule reads -- a `w` at the retention floor next to a flat `prog` is a level the rule has damped for being blocked) and relabels the headline `fresh score N / K levels`, because the pooled figure is then a shrunk sum and can exceed 1.0.
- Tests (no game): `python tests/test_progress.py`, `python tests/test_aim.py`, `python tests/test_campaign.py`, `python tests/test_campaign_rewards.py`, `python tests/test_spaces.py`, `python tests/test_campaign_env.py`, `python tests/test_ladder_replay.py`, `python tests/test_keep_best.py` and `python tests/test_times.py` (pytest is not installed; the files also work under pytest). Also `python tests/test_transfer.py` and `python tests/test_look_mode_transfer.py` (weight surgery), `python tests/test_campaign_config.py` (the campaign config and `train.py` wiring) and the three route-fallback files `python tests/test_route_files.py` (the data), `python tests/test_route_replay.py` (A0, the branch-order safety property) and `python tests/test_route_walk.py` (the 14 trunks walked end to end), and the three specialist files `python tests/test_specialists_config.py` (the plan pinned against `campaign_gates_full.yaml`), `python tests/test_campaign_driver.py` (the stage rule and the driver against fakes) and `python tests/test_full_run.py` (the chaining, and `play_level` against `FakeLevel`). All of them at once, from `python/` in PowerShell: `Get-ChildItem tests\test_*.py | ForEach-Object { .venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }` (**25 files; 612 named tests** as of 2026-09-18 on branch `specialists`, plus `test_progress.py`'s 26 that print no count; ~2 min). `tests/test_games.py` covers `games.py`'s instance-count guard and launch readiness, `tests/test_supervise.py` the crash supervisor and its boot health gate, `tests/test_bridge_recovery.py` the bridge-failure recovery that keeps one sick game from killing a twelve-game run, and `tests/test_freeze_recovery.py` the bounds that keep a sick game from FREEZING it (the 17:52 incident). `test_campaign_check.py` and `test_skull_check.py` print `[FAIL]` lines from their own fake levels on purpose -- they are asserting that a broken level is reported as broken -- so judge them on their last line and their exit code. **`test_campaign.py`'s `test_save_best_run_serialises_two_racing_writers` is load-sensitive**: it races two real threads against a 5 s lock timeout, and on a box already running twelve games it failed once in eight suite runs on 2026-09-17 (then passed 6/6 when re-run on its own). A single failure of that one test under load is not a regression -- re-run the file before believing it -- but it is worth making the race deterministic rather than timed if it recurs.
  **In a worktree, set `PYTHONPATH` to that worktree's `python/`** or `import ultrakill_ai` resolves to the main tree and every test measures the wrong code: `$env:PYTHONPATH = "F:\Github\ULTRAKILL-AI-route\python"`.
- Regenerate the route data (no game, no port, safe beside a live run): `python scripts/build_routes.py --validate` from `python/`. ~2 min for all 33 levels; it rewrites only `ultrakill_ai/routes/` and exits non-zero if any shipped level changes shape. Run it after every game update, then `python tests/test_route_files.py`.
  **In a git worktree**, run them with the main venv but with `PYTHONPATH` pointed at the worktree: the package is an editable install pointing at the main tree, so without it you silently test the wrong code. Verify once with `python -c "import ultrakill_ai; print(ultrakill_ai.__file__)"`.

