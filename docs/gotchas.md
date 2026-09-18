# Gotchas found in the live game

Moved verbatim out of `CLAUDE.md` on 2026-09-18 (documentation restructure). Every entry is as it was
written; `CLAUDE.md` keeps only the one-line safety-critical summaries and points here.

- **Manager object destroyed:** ULTRAKILL destroys BepInEx's manager GameObject. The bridge runs on its own `HideAndDontSave` + `DontDestroyOnLoad` object (`BridgeRunner` in `Plugin.cs`).
- **Background running:** the game ships with `runInBackground` off. The plugin turns it on so the bridge answers while the window is unfocused.
- **Frame cap returns:** scene loads re-enable vsync or a frame cap, so time settings are re-applied on every step and reset.
- **Lockstep timing:** runs at end of frame (`WaitForEndOfFrame`), so obs reflect the finished frame and queued input lands next frame.
- **Hitstop:** the game waits with `WaitForSecondsRealtime`; `TimePatches` makes it frame-based during lockstep.
- **Stuck buttons:** virtual devices need `InputSystem.ResetDevice` before removal, or actions stay stuck pressed.
- **Cyber Grind start:** the player spawns on a ledge (≈ z -47, y 100.5), and waves start only on entering the `EndlessGrid` trigger collider. The mod reports it as `cybergrind.start_trigger`, and the env teleports into its center (lands ≈ (2.5, 26.5, 65)). Walking off the ledge remains as a fallback.
- **Background play:** while in control the game switches to a 640x360 window (`windowed`, `window_width/height`), unlocks the cursor every frame and after `GameStateManager.EvaluateState`, and re-mutes after every scene load, because `GameStateManager.IntroCheck` restores the volume. The original resolution and volume are restored on release and on quit.
- **Single-client bridge:** `BridgeServer` drops its current client whenever a new one connects, so any TCP connection to a training instance's port kicks the trainer off that game and the run dies with `Connection closed by the game` in every env. `games.py status` (and the launch readiness wait) therefore read listening ports from `netstat` instead of connecting. Never poke the ports with another client while training. (The same behaviour is what makes recovery possible from the *inside*: a worker whose socket broke reconnects to its own port and the mod hands the game over, without touching any other worker.)
- **The reset timeout crash, and why the mod's own escape hatch was dead code (killed the run 3x on 2026-09-17).**
  Verified from the tracebacks in `runs/campaign_gates_train.log`: **every** `TimeoutError` bottomed out in
  `protocol.py`'s `recv` under `client.reset`, never under `client.step`. Two were `_campaign_reset` at an episode
  boundary; the one that looked "mid-rollout" was `_respawn` (env.py), which is also a reset request, just issued
  from inside `step()`. The numbers are the tell: `BridgeClient.timeout` was **120.0** and
  `EpisodeController.resetTimeoutSeconds` is **120f** — the same value. The mod starts its reset timer *after*
  the client has already started its socket clock, so the mod's graceful `reset to '<scene>' timed out` error
  reply can never arrive in time and that whole branch was unreachable; there was also no margin at all for a
  slow load. What plausibly takes >120 s (inferred, not instrumented): a reset is a full Unity scene load, vec
  envs step in lockstep so several games hit an episode boundary together and contend for one disk/CPU/GPU, the
  curriculum now loads 11 different and often uncached scenes rather than one, the games run at
  `BELOW_NORMAL_PRIORITY_CLASS`, and a GC pause or a window losing the GPU lands on top. **Fixed** by splitting
  the timeouts (step 120 s unchanged, reset 600 s) and by making the env survive the error instead of dying on it
  — see the bridge-resilience entry under `env.py`. The blast radius was the real bug: SB3's `SubprocVecEnv._worker`
  does not catch, so the worker process exited, the parent read `BrokenPipeError`/`EOFError`, and all twelve games
  went down together while `train.py` hung in teardown.
- **THE 17:52 FREEZE (2026-09-17): the run stalled for 19 minutes and nothing in the logs could say why.**
  Reported by the user as "the AI has stalled". `campaign_gates` stopped at **12,441,598 steps at 17:52:29** with
  `state: "running"`; the supervisor did not act until **18:02:58** (629 s), relaunched all twelve games, and the
  relaunch itself then failed and had to be repeated twice — training resumed at **18:11:41**. It was the sixth
  stall of the day (episode gaps of 2097, 2057, 1571, 1483, 1057 and 1334 s are in `episodes.jsonl`).
  **What was measured during the freeze:** game pid 14896 on port 47800 `Responding=False`, CPU delta over 5 s
  **0.00**, working set 1397 MB (fully booted), and the **only** port of the twelve with an ESTABLISHED
  connection; the other eleven `Responding=True`, ~0.7-1.0 core each, **no connection at all**.
  - **The eleven free-running games were a CONSEQUENCE, not the cause.** `EpisodeController.commandTimeoutMs` is
    300 s: the mod drops a client it has heard nothing from for five minutes and releases control, after which
    the game runs its normal loop at ~1 core. Vec envs step in lockstep, so when one worker blocked, the other
    eleven sat idle behind it and were dropped one by one at 17:57. A game at **0.00 cores and Not Responding is
    the mod in lockstep waiting for a command** — that is the healthy idle state, not a hang.
  - **The exact blocking call is NOT determined, and the reason it is not is itself a bug.** Two evidence
    channels were both dead. (1) The supervisor spawns the trainer with `DETACHED_PROCESS` (0x8), and a detached
    `cmd /s /c "... >> log 2>&1"` **creates the log file and then silently discards everything the program
    prints**: with no console and no inherited handles the grandchild's `sys.stdout` is `None` and every `print`
    is a no-op. Reproduced three ways on 2026-09-17 (0x8 → 0 bytes; 0x200, 0x08000000 and 0 → every line
    written). So `campaign_gates_train.log` had **not been written to since 10:57** — its last
    `total_timesteps` is 7,602,826, it contains no `entropy floor:` line and no `curriculum: 30 levels` line,
    both of which every run since 16:43 prints — while the supervisor printed its seven-hour-old tail as
    evidence on every restart. **The "TimeoutError on reset → BrokenPipeError → EOFError" traceback quoted in
    the 17:52 report is from the 10:57 crash, 4.8M steps earlier.** (2) BepInEx disk logging is off for 12
    games, so the mod's own `No command for 300s, dropping client` warnings had nowhere to go either.
  - **What IS established, by reading and by reproduction on a private game:** the client had no total time
    bound, so any bridge hiccup presented as a hang. Measured live with the game suspended via
    `NtSuspendProcess`: `close()` blocked **20.0 s** against a client whose `timeout` was 20 (it set
    `settimeout(5.0)` and `recv()` armed `self.timeout` straight back over it) — 120 s live, ×12 envs in a
    teardown; the `hello` handshake was bounded by the same 120 s; a socket timeout is **sticky**, so a
    `reset`'s 600 s stayed on the socket for the next `send`. Composed, `_resilient_reset`'s three attempts of
    (close + connect + handshake + reset) ran to **~70 minutes**, and `reset()` retried the whole ladder on top
    — against a supervisor that gave up at 600 s. **Every local recovery this machinery existed for was killed
    mid-flight and paid for as a twelve-game restart.** `_resilient_reset` also ended in a bare `BridgeError`,
    which is *not* in `RECOVERABLE`, so the final failure killed the worker instead of truncating an episode —
    and `SubprocVecEnv.close()` does `remote.recv()` and `process.join()` with no timeout, so `train.py`'s
    `finally` then **hung**, turning a crash the supervisor catches in one poll into a hang it catches in ten.
  - **A single laggard port cost the other 14 minutes.** `games.launch` waits for ALL ports and `sys.exit`s if
    one is missing; the supervisor answers a failed launch with another full stop-and-launch. At 18:08 eleven
    games were up and healthy and the run still would not start, because pid 2588 on 47800 sat at 242 MB
    burning a core with no bridge port. Three such rounds exhaust `--max-restarts-per-hour` and the supervisor
    exits for the night.
  - **Fixed (2026-09-17, branch `freeze-fix`), all of it verified live on one private game on port 47812:**
    every blocking call in `protocol.py` is bounded in every state (`close_timeout` 5 s, `handshake_timeout`
    20 s, `connect_timeout` 10 s, step 120 s, reset **600 → 180** s, and `_LineReader` bounds wall clock rather
    than each `recv`); `reset_timeout` 180 s is comfortably above the mod's own 120 s give-up and below its
    300 s client drop — real resets on the live 12-game run are **median 1.94 s, max 5.38 s** over 181 samples,
    so 600 s was 110× the observed maximum; one **total** `bridge_recovery_budget_s` (540 s) covers every
    attempt, backoff and layer of a recovery; the last rung **relaunches this env's own game** via
    `games.relaunch_one` and waits out a working-set boot gate; the mod is told `command_timeout_s: 900` so one
    worker's recovery no longer gets its eleven siblings dropped (accepted by the installed mod v0.7.0, no
    rebuild); `train.py` tears the vec env down in bounded time and terminates a wedged worker; `games.launch`
    repairs a laggard port instead of failing the whole launch; the supervisor judges **steps, not mtime**,
    spawns without `DETACHED_PROCESS`, and writes a per-port sick report before it kills anything. Measured
    after the fix: `close()` 5.00 s, handshake 20.00 s, a whole failed ladder 51 s against its 60 s budget, and
    a game **killed outright** mid-episode recovered in 47.7 s with all twelve training ports' owning pids
    unchanged.
  - **And the attribution hole is closed:** every worker writes `runs/<run>/env_<port>.log` (see `envlog.py`).
    The next incident is readable from that file plus the supervisor's sick report, with no console attached.
  - **IT HAPPENED AGAIN AT 19:13 WHILE THE FIX WAS BEING WRITTEN, and the recurrence is the best evidence we
    have** — because main was still running the pre-fix code and the whole shape was captured live. The run
    froze at **13,261,282 steps at 19:13:09**; the supervisor killed the trainer at **19:23:26** (617 s) and
    relaunched all twelve. The socket table during the freeze, from `netstat` alone (never connect to a
    training port):

    | what | port 47809 | the other eleven |
    |---|---|---|
    | game side | **ESTABLISHED** | `FIN_WAIT_2` |
    | worker side | `ESTABLISHED` | `CLOSE_WAIT` |
    | `Responding` | **False** | True |
    | CPU delta / 5 s | **0.109** | ~1 core each |

    `FIN_WAIT_2` on the game side with `CLOSE_WAIT` on the worker side means **the mod sent FIN and the worker
    never closed** — the mod's 300 s `commandTimeoutMs` dropping eleven clients that were merely idle behind
    one blocked worker. It pins the causal direction the first post-mortem could only infer: one sick game,
    eleven dropped siblings, twelve recoveries. `mod_command_timeout_s: 900` is the fix, and
    `EpisodeController.cs:266` does read the key, so it is live without a rebuild.
    **The deadlock itself:** the frozen game sat at 0.109 cores / 5 s (idle in lockstep, *waiting for a
    command*) while its worker and the trainer parent both showed a CPU delta of **exactly 0.00** (blocked in
    a syscall). Both sides waiting on the other, with an empty receive buffer on the client — so the reply was
    never sent, not merely missed. That is a genuine request/reply deadlock against a wedged Unity main
    thread, not a lost message.
  - **Second-pass audit (2026-09-17): six findings, all six verified by measurement before being fixed.** The
    first pass got the bounds right but left the *total* advisory, so four of these are the same bug wearing
    different hats. Every number below was measured on a fake clock (`_fake_clock_ladder`, no game, no
    sockets), with each blocking call charged its real bound:
    1. **The "one TOTAL budget" was not a total.** `_resilient_reset` checked the budget only at the TOP of
       each attempt, so an attempt admitted with a moment left ran its own full bound anyway — and `request`
       arms the bound for the send *and* the read. Measured: a game that answered `hello` and never answered a
       reset ran **555.6 s** against a 540 s budget, and up to 780 s in the worst case. Worse, the ladder
       closed every budget it touched, including one opened above it, so `reset()`'s tail handler — the one
       that fires when `_skip_locked` hits a frozen game after a reset that DID answer — found a cleared
       deadline and opened a **second full budget** (measured: two, 1000→1540 and 1183→1723, for one fault).
       Fixed by making the deadline authoritative: `_begin_recovery` returns `(deadline, opened_here)` and
       only the opening frame may close it, the budget opens at the top of `reset()` and in `step()`'s catch,
       and `_clamp` cuts every call — `client.reset(timeout=…)`, `connect(retry_seconds=…)` — to what is left.
       Measured after: **540.0 s exactly**, one budget, for every shape.
    2. **The relaunch rung was unreachable in the fault shape it exists for.** Rung 1 was gated on the budget
       left *after* rung 0's three attempts, and those attempts each cost a full reset timeout against a
       wedged game, so the budget was always negative and the loop broke. Measured: relaunch called on a
       fast-failing bridge (which is what the test faked), **not called** on a slow-failing one (which is what
       production sees) — the only covered path was the one that does not occur. Fixed with an explicit
       `bridge_relaunch_reserve_s` (240 s) that rung 0 may not spend. Measured after: reached in both shapes.
    3. **`train.py` wrote the eval a loaded gun.** `fill_run_dirs` sets `bridge_relaunch=True` and
       `env_log_dir`, and those went into `models/<run>/env_config.yaml`, which `eval.py` loads verbatim.
       Round-trip verified: `bridge_relaunch=True`, `env_log_dir='runs/campaign_gates'` on the far side. One
       `BridgeTimeout` during the documented eval command and the env would `taskkill` a **training** game on
       port 47800. Fixed with `EnvConfig.RUN_ONLY_FIELDS`, excluded from `to_dict`, plus a defensive clear in
       `eval.py`.
    4. **Every `subprocess.run` in `games.py` was unbounded** — `tasklist`, `netstat`, `taskkill`, seven of
       them, one with a timeout. These are now called from *inside* the recovery whose module docstring
       promises it cannot hang, and from the supervisor. `netstat` blocking under Tcpip/WMI contention is
       normal on exactly the loaded box this runs on. All bounded at 30 s, `TimeoutExpired` caught and
       returned as an empty map so the caller falls to its next rung, and the snapshots are **cached with a
       2 s TTL** (twelve workers polling once a second was ~18 process spawns a second).
    5. **`connect_retry_s` had been cut 180 → 60 with no justification**, and a connect failure raised a bare
       `BridgeError`, which is *not* in `RECOVERABLE` — so the first `_ensure_connected`, which runs above the
       ladder, killed the worker outright. Measured: a port that was not listening killed the worker in
       **60 s** with no ladder and no relaunch. That is precisely the state `supervise.await_boot` creates on
       purpose when it gives up on a cold copy and starts the trainer anyway, on the stated grounds that the
       env waits a cold game out. Fixed three ways: `connect` raises `BridgeClosed` (which *is* recoverable),
       `reset()`'s tail reconnects through `_reconnect_within` (which retries inside the budget and then
       relaunches), and the retry window is back to 180 s. Measured after: **540 s of patience, and the
       instance gets relaunched.**
    6. **Nothing serialized the per-worker relaunch.** Twelve workers step in lockstep, so a fault that drops
       the games reaches all twelve within one step, and twelve simultaneous cold Unity starts boot none of
       them in time — so every `relaunch_one` returns False, every worker dies anyway, and twelve games have
       been killed behind the supervisor's back for nothing. Now at most `bridge_relaunch_slots` (2) workers
       hold a relaunch at once, through `O_EXCL` permit files in `env_log_dir` with a stale-age takeover, plus
       a per-port stagger. Separately, `relaunch_one`'s "no process is listening; starting one anyway" branch
       started a replacement **without reaping the wedged portless instance** (`launch` calls
       `kill_unlistening` for exactly this; the env path did not), leaving it at ~1 core for the rest of the
       run; it now reaps first.
    **Not changed, and why:** the worker still *dies* when a recovery genuinely fails. That is the design —
    `train.py`'s teardown is bounded, so the supervisor sees a DEAD trainer within a poll instead of a HUNG
    one after ten — and it is what makes a 540 s cap safe to enforce.
    **The arithmetic, re-derived:** a worker's worst silence is the step that faults (120 s, paid before the
    budget opens) + the recovery budget (540 s) = **660 s**; the supervisor may notice one poll late (60 s) =
    **720 s** against its 900 s window. `test_the_measured_ladder_fits_inside_the_supervisors_patience` does
    not take that sum on trust — it runs the real ladder on a fake clock for each fault shape and asserts the
    measured elapsed fits.
- **A listening bridge port is NOT a booted game.** The plugin opens its socket early in startup, while
  Addressables' resource locators are still empty — and `EpisodeController.SceneExists` searches exactly those
  locators. So a half-booted instance answers *every* reset with `unknown scene 'Level 0-1'`, which used to reach
  the trainer as a plain `BridgeError` and kill the whole vec env at `env.reset()`. Measured 2026-09-17: the
  stuck copy sat at a **56 MB** working set while the other eleven sat near **1 GB**, and two trainer starts were
  burned before anyone looked at memory. Two defences now: `supervise.py`'s boot health gate waits for every copy
  to hold >600 MB for two consecutive polls (and restarts a laggard on its own port with `games.py relaunch`),
  and the env waits `unknown scene` out for up to 5 minutes instead of raising.
- **"Instance(s) [...] exited during startup" is a false alarm after a dirty stop.** `games.py launch` used to
  judge readiness from the `Popen` handles it created, but the game **re-execs itself under a new pid**, so the
  original handle reaps a dead process while the instance comes up perfectly well; the ports appear 2-3 min
  later. It printed this twice during the 2026-09-17 recovery. Readiness is now judged by port (netstat) and by
  whether *any* `ULTRAKILL.exe` is running (`startup_verdict`), never by the launch pids — those are only launch
  handles and are not an instance's identity. Use `listening_pids()[port]` when you need to address one instance.
- **Multiple instances:**
  - The game holds `Preferences/Prefs.json` open with exclusive write access, so a second copy crashes unless `InstancePatches` opens it read-only.
  - Launching the exe directly works (Facepunch `SteamClient.Init`; the launcher sets `SteamAppId`).
  - The game re-applies resolution from `LocalPrefs.json` in `InitGame` at startup, so Unity's registry `Screenmanager*` values barely matter.
  - BepInEx writes `LogOutput.log.1..3` for extra instances.
- **Steam only credits playtime for instances it can SEE, and since 2026-09-17 training is not one of them.**
  Registering with Steam happens in exactly one place — Facepunch's `SteamClient.Init` calling
  `SteamAPI.Init()` — and mod v0.7.1's `-aibridge-nosteam` skips it, which `games.py`/`supervise.py` now pass
  by default. So **no training hours are credited to ULTRAKILL on Steam**, and Steam does not show the game as
  running while twelve copies train. **Your own sessions are unaffected**: nothing adds the flag to a game
  started from Steam or from the launcher, and the patch is per-process, applied at `Plugin.Awake` from that
  process's own command line. `--steam` puts a training launch back on the books.
  - **How to check what Steam believes**, without reading the client UI: `HKCU\Software\Valve\Steam\Apps\1229490\Running`
    (and `HKCU\Software\Valve\Steam\RunningAppID`). Measured 2026-09-17 with the Steam client up: 0 with no game,
    **1 / 1229490** about 10 s after a `--steam` instance passes ~1 GB working set, and **0 for the whole two
    minutes** a hidden instance sat fully booted at ~950 MB. The value lags the launch by a boot, so reading it
    the instant a port opens proves nothing — wait for the working set.
  - Not verified: the Steam client's own window and tray, which are GUI state this session could not read.
    The registry pair above is what was measured, plus the mod's in-band `steam_hidden` in the handshake.
- **The five-copy cap was BepInEx's disk log, not the game (measured and lifted 2026-09-17).** BepInEx's
  `DiskLogListener` opens `LogOutput.log` plus `.1`-`.4`, so a sixth copy could not open a log file and the
  plugin never loaded. Setting `[Logging.Disk] Enabled = false` in
  `C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\config\BepInEx.cfg` removes the listener and
  the cap with it: **8 copies load, serve the bridge and train**. Nothing in `games.py` capped it either — ports
  are `base_port + i`, readiness is read from netstat and `tile` wraps onto a second row — so the only change
  needed was a guard: `launch --count N > 5` now refuses while disk logging is on and names the file to edit
  (`games.py:disk_logging_enabled` reads `Enabled` inside `[Logging.Disk]` only, because `[Logging.Console]` has
  an `Enabled` key of its own a few lines above that is `false` by default; `tests/test_games.py` pins that).
  **To undo:** the pre-change file is kept as `BepInEx.cfg.bak-preN` beside it — restore it (or set `Enabled`
  back to `true` in the `[Logging.Disk]` section) and put `num_envs` back to 5. **The cost:** with the disk
  listener off the plugin only logs to the console listener, so a crash leaves no file behind. Turn it back on
  (and drop to 5 copies) when diagnosing one — that is where `campaign_check.py`'s key-collision and exit-tie
  warnings go.
- **Throughput scales well past five games (measured 2026-09-17, Ryzen 9 3900X 12C/24T, 32 GB).** Same
  checkpoint, same config, ~15k steps each, steady-state steps/s from SB3's own counters between iteration 1 and
  iteration 8 (`n_steps` in the config is the TOTAL rollout size and `train.py` divides it by `num_envs`, so the
  PPO batch is identical at every N and the comparison is fair):
  | games | steps/s | vs 5 | CPU |
  |---|---|---|---|
  | 5 | 152 | — | — |
  | 7 | 183 | +20% | 31% |
  | 8 | 235 | +54% | 33% |
  The second A/B (integration pause #2, same day, from the real 6.77M resume checkpoint on
  `configs/campaign_gates_main.yaml`, ~18k new steps per leg into a scratch `campaign_smoke` run whose
  exploration archives were seeded from the live ones):
  | games | steps/s | vs 8 | notes |
  |---|---|---|---|
  | 8 | 188.6 | — | 9 rollouts, 0 bridge errors |
  | 10 | 220.5 | +16.9% | 10 rollouts, 0 bridge errors |
  | 12 | 236.5 | **+25.4%** | 10 rollouts, 0 bridge errors; 9.6 GB of 31.9 GB, 16.6 GB free |
  **12 was adopted** (the bar was +10% over 8). Read these as ratios, not absolutes: the two A/Bs disagree on
  8 games (235 vs 188.6) because the first measured SB3's cumulative `fps` on short legs that saw few level
  loads, while this one takes the slope between rollout 1 and the last, excluding start-up. In the live run at
  12 games the first rollouts settled at **~270 steps/s**. Scaling still had not flattened at 12.
  **The aggregate rate IS the per-game responsiveness check**: `SubprocVecEnv` steps every env in lockstep, so
  one slow or wedged game drags the whole vector down and would show up here as a collapse, not as a quiet loss.
  Nothing in `games.py` needed changing for N > 8 — ports are `base_port + i`, `tile` wraps onto more rows, and
  `status` scans 16 ports. **Seed a new port's exploration archives** from an existing port's before using it
  (`models/campaign_gates/explore_<level>_<port>.npz`), or it re-pays novelty for ground the run has covered;
  47808-47811 were seeded from 47800-47803 exactly as 47805-47807 were from 47802-47804.
- **Training speed findings** (random actions, 4–5 games):
  - The PPO update is only ~0.2 s per 2048 samples on CPU, so the GPU wouldn't help. The time is in the games and resets.
  - 60 fps / frameskip 4: 138 steps/s. 30 fps / frameskip 2 (same 15 decisions per game second; game logic is deltaTime-based): 199.
  - Rendering off: 217. Soft death: 297. Both: 328. With 5 games: 380.
  - A 6th copy fails because BepInEx opens at most `LogOutput.log` plus `.1`–`.4` — **unless disk logging is
    switched off**; see the five-copy-cap gotcha above.
- **Lockstep stalls:** vectorized envs step together, so a reset in one game stalls all of them. Keep resets short (teleport entry, `reset_settle_frames` 10).
- **Speed:** about 600 fps / 150 steps/s in an empty scene and about 100 steps/s with enemies (frameskip 4, RTX 5070).
- **Campaign checkpoint triggers fire at training speed; the exit was never a trigger-timing test:** checkpoint triggers activate correctly at 30 fps / frameskip 2 with rendering off (`campaign_check.py` check 3 PASS on 0-1 and 1-1, 2026-09-16), so no speed-setting change was needed. Check 5 (exit) FAILed on both levels, but not from timing: see the exit-room gotcha below.
- **Campaign throughput:** 177 steps/s with 5 games on 0-1 at 30 fps / frameskip 2 with rendering off (20k-step smoke run from fresh weights; SB3's `fps` over the whole run, level loads included). At that rate 1M steps take 1.6 h and the 20M-step pilot 31 h.
- **Exit room switched off at load; checkpoint activation does not bring it up (0-1 and 1-1, 2026-09-16):** a fresh load reports the exit `active: false` on both levels, so `campaign_check.py` check 5 fails there. Automated stand-in for playing through, since `CheckPoint.ActivateCheckPoint` calls `SetActive(true)` on its own room (`decompiled/CheckPoint.cs`): teleported onto and activated every checkpoint in spawn-distance order. 0-1 activated 5 of 6 (the nearest did not activate within 30 decisions on this attempt); 1-1 activated 4 of 4. The exit stayed `active: false` on both afterward -- so checkpoint activation does not switch on the exit's own room; it sits well past the last checkpoint (293 m on 0-1, 469 m on 1-1 from the last checkpoint tried). Not verified by hand (no human play in this session). Treated as a known open item, not a blocker: check 5's code path is covered by `test_campaign_env.py` against the fake bridge, and the first real level completion during training exercises it live. (The `path.status` staying `none` the whole walk turned out to be a separate NavMesh-sampling bug, not caused by the room being inactive -- see the exit-sample fix below; `FinalPit` is findable and its `transform.position` valid via `FindObjectsOfType(true)` regardless of active state.)

- **Campaign NavMesh path, measured 2026-09-16.** The path hint only resolves near the exit. At `Level 0-1`'s
  spawn the player is grounded and snaps onto the mesh fine, but the mesh island holding the exit is not
  connected to the start area, so `NavMesh.CalculatePath` finds nothing and the block reports `status: "none"`;
  from the level's farthest checkpoint it reports `partial`, 72.4 m. `Level 1-1` reports `partial` from spawn
  (16.6 m). So the `path` reward pays nothing over the first part of a level by design, and exploration plus the
  milestones carry it; the straight-line exit vector is in the observation either way. `PlayerSnapDistance` is
  25 m rather than 6 m so the hint does not drop out every time the agent is airborne, which in this game is
  most of the time.

- **A TRANSFORM is not a place you can stand, and two live bugs came from treating one as a target
  (measured on a private game 2026-09-18, fixed in mod 0.7.2 + Python).** Both are the same mistake in two
  places: a position that identifies an object is not a position the agent can be told to go to. When adding
  any new target, ask what the player would be standing on if they reached it.
  1. **The exit target was the `FinalPit`'s own transform, which sits INSIDE the drop it triggers.** Measured
     **61-75 m below standable ground on 0-2 and 70 m on 0-3**. Everything that drives the agent somewhere
     reads `GateProgress._exit` — look mode 2, `gate_approach` and the target slots 448-455 — so once the
     ladder handed over to the exit the agent was being aimed down a killing fall: **11 of one 0-2 episode's
     18 respawns were falls taken at full health**, and `exit_dist_min` could never go below that offset.
     Both of 0-2's real completions triggered at **(-197.5, -25, 276) / (-199, -27.5, 277)** while `exit.pos`
     read **(-199, -86.1, 277)**. The fix was already half-built: `CampaignObserver.SampleExit` has snapped
     the path hint's exit end onto the NavMesh since 2026-09-16, so mod 0.7.2 just keeps that point and
     reports it as **`exit.ground_pos`**, and `_exit` prefers it. **Observation slots 0-4 still read
     `exit.pos`** — that raw vector is a learned input column the policy has read since the run began.
     `exit_dist_min` keeps its definition so its history stays comparable; **`exit_ground_dist_min`** is the
     new one, and it is the column to read for "did the agent get near the exit".
  2. **A room CENTROID is not inside its room, near a wall.** 0-3's trunk rung `2 - Side Hallway - Floor 1`
     sat at z 331, **1.5 m past the main room's far wall (face z 329.5)**, while the hallway floor is y 10.4
     spanning z 332-362. `_is_reached`'s 8 m x 6 m cylinder therefore covered the **wall face** down to y 4,
     so the rung was credited with the player hanging against the wall in the room **before** it: **680 of
     801 live credit steps airborne**, the first credit of every one of three fresh episodes at
     `ground_ray_center` 9.5-10.5 m, and a **scripted hold-forward-and-jump run that never crossed the wall
     (max z 329.5) credited it 31 times**. The ladder then advanced the target to the next rung **through the
     wall**, and the agent dropped into the bowl — **26 of 66 fresh route episodes ended there**. Two
     independent fixes, because **neither alone is enough**: the rung moved to **z 340** via
     `rung_overrides.json` (every first credit then lands inside the hallway, z 332.9-333.9, and the scripted
     non-crossing runs credit 0 times; `tour_ratio` 0.9439 → 0.9438, every other diagnostic unchanged), and
     `GateProgress._on_ground` refuses to credit any **room-trunk** rung from the air with no ground within
     `route_ground_m`. The bound is measured, not chosen: legitimate airborne credits on 0-3's six healthy
     rungs max out at **5.8, 5.9, 6.0, 6.0, 6.0 and 7.9 m**; the wall-face first credits are **9.5, 10.5,
     10.5**; the scripted ones **8.8 and 9.1**; and standing over the main room's void reads **30.0**, the
     `ground_ray_length` sentinel. **8.0** is the only round number above every good reading and below every
     bad one. Measuring it the other way round is what showed neither fix is redundant: the bound **alone**,
     with the rung left at z 331, still credited from a ledge on the main-room side at 3-5 m.

- **The baked NavMesh is an ENEMY-walking mesh, and `path` was retired because of it (2026-09-16).** 0-1's mesh
  is 412 polys in 47 islands with **nothing before z 378.5**, and no bridging rule connects the start area to the
  exit below a 17 m gap / 12 m climb limit — one that did would also bridge through walls. Over 32,022 logged
  decisions `path.status` was `partial` on 82.19% of rows and **never once `complete`**, so the `path` reward
  paid **0 in 656 logged rows** and its `status == "complete"` observation input had an exactly zero first-layer
  column norm in both hidden stacks. `path` is now weight **0** and its observation slots carry the route target
  instead. A NavMesh triangulation dump was considered and rejected for the same reason. Do not reach for the
  NavMesh as a route signal again: `campaign.gates` is the route signal.

- **`campaign.gates` is the route, and it is a door graph, not a path.** A gate is a `Door` whose
  `activatedRooms` holds >= 2 distinct rooms; those rooms (and only those) are the graph's nodes, keyed by
  **rounded world position** — never by reference, because `CheckPoint.Start` clones each room, deactivates the
  original and banishes it +10,000 m in X, and `ResetRoom()` re-instantiates the clone on every respawn, so the
  same physical room is a different object across a death. BFS from the exit's nearest room-node ancestor gives
  each door its `hops`. Measured live: 0-1 11 gates 0..9, 0-2 17 gates 0..7, 1-1 13 gates 0..5, **0-5 unordered**
  (no room node on the pit's ancestor chain) — Python falls back to the straight-line exit vector there.
  Rules worth remembering: `key`/`pos` are the **closed** position and frozen for the level load (a Normal door
  moves its OWN transform by `openPos`, measured (0, 5.75, 0) on every gate door of 0-1..0-5, so a rescan while
  the player stands in the proximity trigger would otherwise flip the key); `key` is **opaque**, including any
  `#2` suffix, because `Mathf.RoundToInt` rounds halves to even and 0-1's doors sit on exact `.5` z values;
  `open` means "currently open or opening", never "has been passed" (enemies open doors, and opening one
  force-closes the others); and `controller_active` is what separates "walk up to it" from "a fight gates it" —
  0-1's gun-room gate reports `open: false, locked: false` at load and still cannot be opened.
  Levels beyond 1-1 are unverified apart from 0-2 and 0-5.

- **`hops` is a shortest-path LOWER BOUND, not a route, and treating it as a monotone ladder collapses on six
  levels (diagnosed live 2026-09-17).** Multi-room doors over-connect the room graph, so the gate nearest the
  spawn can already sit near `hops` 0. Measured collapsed on **0-3, 1-1, 1-2, 2-3, 4-3 and 8-1** — four of them
  in the live 11-level curriculum. On 0-3 the spawn-side door `0,13,330` (hops 2) is "reached" from the pit
  **below** it — dy −4.1 to −6.0 m, dh 6.0-8.0 m, airborne, in all six probe episodes, so the 8 m / 6 m reach
  cylinder counts it without the player ever passing it — `best_hops` locks at 2 and the target becomes
  `-16,73,315`, **66 m straight up through a ceiling**, for 2,052-2,423 of every 2,501 decisions. 175 of 177
  training fresh episodes ended stuck beneath it with zero checkpoints. The walkable route is
  **2 → 3 → 4 → 5 → 6 → 3 → 2 → 1 → 0**, which is not monotone, so every forward leg paid nothing and
  observation slots 448-455 pointed at the wrong door. 0-1 is monotone (`spawn_h == max_hops == 9`) and works,
  which is why the bug went unseen for a whole run. Target patience (branch `ladder-fix`) is the mitigation, not
  a cure: it parks a target that stops getting closer and falls back to the nearest unreached gate at any hop
  count. If the falsifier under Status fails, the ladder needs a real per-level route, not another knob.

- **The room trunk IS that per-level route for three of the six, measured 2026-09-17 — and it is deliberately
  not shipped.** Stage S1 ran its pipeline over the six collapsed levels as a scratch measurement
  (`python scripts/build_routes.py --levels 0-3 1-1 1-2 2-3 4-3 8-1 0-1 --analyse-gates-levels --dry-run`;
  no file is written for any of them, because the emitter refuses to emit where the live gate guard passes,
  and the route spec's risk 5 does not authorise replacing a gates ladder). The control first: **0-1**'s trunk
  is 13 rungs, tour **1.000**, 13/13 legs witnessed, 6/6 checkpoints, 8/8 authored directed facts, and the gate
  `hops` beside each trunk rung read `9,9,9,8,7,6,5,4,3,2,2,1,0` — monotone, reproducing the working ladder rung
  for rung. Then:
  - **0-3 — yes, and it would fix the collapse by construction.** Trunk 11 rungs, tour 0.944, 9/11 legs, 4/4
    checkpoints. The gate `hops` along the trunk are `2,2,3,4,5,6,4,4,3,2,0` — the same out-and-back shape as
    the measured walkable sequence above — while the trunk itself expresses that route as a strictly monotone
    `10..0`, so **every forward leg pays**. It cannot collapse the same way either: the gate that collapses 0-3
    (`0,13,330`, reached from the floor below) corresponds to trunk rung 2 of 11, and the floor-2 room above it
    sits **40 m higher**. Over all 19 levels probed, the closest pair of trunk rungs within `_is_reached`'s 6 m
    of vertical is **21 m apart horizontally** against an 8 m cylinder — invariant I2's 16 m separation plus the
    room numbering keeps floors apart in a way the door graph does not.
  - **4-3 — the strongest of the six.** 8 rungs, tour 1.000, 6/8 legs, 3/3 checkpoints, every guard passed. The
    live ladder only covers the second half (3 of 5 gates carry `hops`, ratio 0.600, barely over the 0.5
    threshold; the first four trunk rungs have no gate hops at all); the trunk covers the level from spawn.
  - **1-1 — yes, and it measures the route spec's risk 5 directly.** 12 rungs, tour 1.058, 8/12 legs, 4/4
    checkpoints. The gate nearest the spawn carries **`hops` 1**, i.e. 1-1's final door really does list the
    spawn field in `activatedRooms` and put the spawn one hop from the goal. The trunk orders it `11..0`. It
    would stop at a lock until S3 (SkullRed on leg 2, SkullBlue on leg 9).
  - **8-1 — plausible order, but it loses the exit room.** 20 rungs, tour 1.245, 15/20 legs, 7/7 checkpoints,
    but only one directed fact to score against, and invariant **T drops the exit's own room
    `2C - Bathroom Ending`** (a member of a 429 m wide parallel set), leaving the last rung **1,513 m** from the
    pit. Its first three rungs also sit beside `hops` 0 gates — the loop its ladder collapses on. Not a drop-in.
  - **1-2 and 2-3 — refused by the spec's own tour guard.** 1-2 scores **1.426** and 2-3 **1.393** against R2's
    1.35, so even if layer 2 were allowed to fire there neither would get a file.

  **The order for the lead**, if the `ladder-fix` falsifier fails on 0-3: the trunk is the fix in hand for 0-3,
  4-3 and 1-1, and taking it means a deliberate ruling to let layer 2 **replace** a passing gate ladder on
  named levels — a change to `_gates()`'s precedence, not a data change, and it must be judged against A3
  (0-1 identical) exactly as S2 was.

- **`campaign.exit.pos` gets BANISHED +10,000 m in X on 0-2, and the proper fix is mod-side.**
  `CheckPoint.Start` (`decompiled/CheckPoint.cs:132`) and `CheckPoint.ResetRoom` (`:681`) clone every room the
  checkpoint owns and then move the **original** by `transform.position.x + 10000f` — x only, y and z untouched.
  The live clone and the real `FinalPit` trigger stay put, but the mod's frozen `FinalPit` reference follows the
  banished original, so on 0-2 the reported exit jumps (−199.0, −86.1, 277.0) → (9801.0, −86.1, 277.0) the
  moment checkpoint `-55,-11,277` activates — which is exactly when the gate ladder hands the target to the
  exit. `ResetRoom` runs again on every respawn, so the offset is k × 10,000 for k ≥ 1. 0-2 otherwise runs
  cleanly to `gates_reached` 8.
  `ExitGuard` (Python, branch `ladder-fix`) freezes the exit per level load and is a stopgap. **The real fix is
  mod-side at the next rebuild: re-resolve the `FinalPit` on every `Scan()` instead of caching the reference,
  so a rescan picks up the live clone** — the same rule the room-node keys already follow (rounded world
  position, never a reference, for exactly this reason). Until then, watch `exit_banished`.

- **The absorbing `slowMode` wedge, and the fix (mod v0.6.0).** A slide that ends while the player is airborne,
  or a jump out of a slide where the game's stand-up test fails, sets `NewMovement.slowMode` and leaves a state
  nothing can end: `grounded` never comes back (`GroundCheck` keeps `onGround` from trigger callbacks over a
  `cols` list, and a collider the capsule already overlaps never fires `OnTriggerEnter` again), velocity reads
  `(0,-100,0)` or ~0, stamina is frozen, jump/dash/slide are inert, and a **0.477 m/s creep** is all that
  remains. It is a general state, not a vent property: measured in 0-1's slide vent, at the spawn-room wall and
  at (145.4, 45.5, 664.2) after checkpoint 3. On the 32,022-decision live log exactly **five** runs of >= 45
  consecutive wedge decisions exist (2976, 1108, 729, 694, 672) and nothing else exceeds 29.
  The fix is `UnwedgePatch`, a postfix on the private `NewMovement.HandleSlideState`, active only while
  `EpisodeController.InControl`: clear `heavyFall`, zero a `<= -99` downward velocity, re-assert the crouch
  triple defensively, clear `slowMode`, then `ForceGroundCheck()` on every `GroundCheck` instance — **in that
  order**, because `GroundCheck.OnTriggerEnter`'s `heavyFall` branch deals x5000 damage to overlapping enemies
  and can `Bounce()`, which teleports the player.
  - **The hold is load-bearing.** The guard is held for `unwedge_frames` (**10**) consecutive frames. `Update`
    runs `HandleInputs()` before the `heavyFall` block and before `HandleSlideState()`, and `TryStartSlam` calls
    `StopSlide()`, which does **not** restore `playerCollider.height` — so a legitimate ground slam started out
    of a slide arrives at the stand-up test with height 1.25 still set and, under any ceiling, gets
    `crouching`/`slowMode` set on the very frame `heavyFall` and `(0,-100,0)` were. Acting on frame 1 silently
    cancelled the slam: no `LandingImpact`, no ground-slam enemy damage, and no `Breakable.Break(2f)` — one of
    the few ways through the planks that seal 0-1's starting room. Measured A/B: at `unwedge_frames` 1 the slam
    is dead on decision 1; at 10 it survives 4 decisions and the real wedge still breaks at 0.400 s.
  - `unwedge` (bool, default true) is a kill switch and `unwedge_frames` its hold. **Both are sticky per game
    process**, like `soft_death`: a client that sends `unwedge:false` and disconnects leaves it false for the
    next client on that port. Never send `unwedge:false` to a port a trainer might share.
  - A **grounded** crouch under a low ceiling is the same `slowMode` and is deliberately untouched; it still
    reports `slow_mode: true`.

- **0-1's `unlock_all_gear` alternate start, and the vent.** With `unlock_all_gear` on, `GearCheckEnabler` swaps
  in a short starting room at (39.7, -0.5, 343.7), sealed by two weak `Breakable` planks at (40,0,347.5) and
  (40,2,347.5). Punching through them or sliding under the lower one (a 1.5 m gap against a 1.25 m slide height)
  is the only way out, and that same opening is the **slide vent** (x = 40.0, z 349 -> 377.5, 1.0 m wide) where
  the wedge happens. **A scripted grounded slide passes it every time** — measured again at integration: forward
  + slide from spawn reached the gun-room door at z 401 in 42 decisions. So the older note that "0-1 spawns you
  sealed in" was wrong as stated: the room is sealed to a *teleport* and to a *walk*, not to a slide, which is
  why the scripted `walk_to_exit.py` could not leave it. `punch` is still charged before `level_started`, which
  prices the only other tool a weaponless player has; left alone deliberately, flagged in the design spec.
