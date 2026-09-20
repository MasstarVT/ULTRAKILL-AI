# Project log

The whole of the old `CLAUDE.md` Status section, moved on 2026-09-18 (documentation restructure):
every dated incident, measurement, branch history and decision, in the chronological order it was
written in, otherwise **verbatim**. The date headings below are the only thing added — no entry text
was edited, so a claim inside an entry is true as of that entry and may have been superseded later
in this file. Append new entries at the BOTTOM, dated, and keep `CLAUDE.md` to a few lines of
current state.

## 2026-09-15 → 2026-09-16 — Cyber Grind: reward rebalances, the aim audits, v1 and v2

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
- **Mod:** v0.6.0 (background play, training instances, teleport, soft death, rendering off; campaign block, `difficulty` and `unlock_all_gear` config, `kill` debug command, `player.slot_counts`, game-initiated restarts blocked while in control; **the `campaign.gates` route block with its room-graph BFS, the `ChooseExit` secret-pit and mission-successor fix frozen per level load, `ground_ray_center`, `player.slow_mode`/`heavy_fall`/`crouching`, and the `UnwedgePatch` postfix with its `unwedge` / `unwedge_frames` config keys**). Verified in game: plugin load, handshake, Cyber Grind reset, movement/look/jump/dash, observations (enemies, waves, damage, death), leaderboard block, and every v0.6.0 addition (acceptance A1-A12 under Status). Protocol version is still **1**: obs gains fields, nothing is removed or changed. **Installed version is now v0.7.2** (see the 0.7.1 and 0.7.2 entries at the end of Status).
- **Python:** env, training and eval verified against a mock and partly in game; campaign episodes verified against a fake level (`tests/test_campaign_env.py`).
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
- **Next steps (Cyber Grind, paused while the campaign runs):**
  - **Yaw is the one thing to fix.** `enemy_yaw_angle_mean` 51 deg is what keeps `on_target_frac` at 3%;
    pitch changes have been tried at 15 and 45 deg and moved nothing. Consider a yaw-only shaping term with
    a zero floor (`w * max(0, 1 - yaw_err/T)` with T near the current mean so improvement always pays),
    rather than another pitch pass.
  - Hold `ent_coef` so entropy stays near 3.5-4.0; below ~3 the policy went deterministic and regressed.
  - Watch the 1.70M rebalance against the falsifiers above; `yaw_track` then `on_target_frac` lead, kills/min follows.
  - If `on_target_frac` is still <= 0.06 at 2.18M, reset the look-head output bias (as `pitch_reset_2447005.zip` did for v1) rather than tuning weights again: a constant offset in the bias is not reachable from a reward slope that is flat across the whole sweep range.
  - `approx_kl` runs 0.029-0.031 against `target_kl` 0.02, so every update is being truncated. Worth a pass once the reward change has been judged, but not at the same time as it.
  - Raise `max_wave` as the agent improves.

## 2026-09-16 — Campaign foundation, the in-game checks and the 0-1 pilots

- **Campaign: designed 2026-09-16, foundation built (mod v0.6.0).** Goal: finish all 35 main levels (`Level 0-1` to `Level 9-2`)
  as fast as possible, learning alone. Spec: `docs/superpowers/specs/2026-09-16-campaign-foundation-design.md`.
  - **Decisions:**
    - No human demos or recorded routes: `routes.py` and `record_route.py` were deleted 2026-09-16 (git history keeps them).
    - Violent difficulty, all weapons unlocked in memory only.
    - Deaths respawn at the checkpoint inside the episode.
  - **Approach:**
    - The mod reads level structure: the real `FinalPit` exit, checkpoints, locked doors, arena clears, the
      **door-graph route (`campaign.gates`)** and the official timer. The NavMesh path hint is still reported but
      is no longer a reward: it is an enemy mesh and never resolved (see the NavMesh gotcha).
    - Rewards: the gate ladder and its approach shaping, time, milestones and per-cell novelty as a secondary explorer.
    - Episodes mostly respawn at each game's current checkpoint, so training concentrates on the frontier.
  - **Sub-projects:**
    1. Foundation + 0-1 pilot. Success bar: 50% fresh-start completion on Violent.
    2. The rest of the Prelude.
    3. Skull keys and switches.
    4. Acts 1-3.
    5. Speed and movement-tech actions.
  - **Starting weights:** Cyber Grind `best.zip`, widened to 479 inputs (`transfer_weights.py`), then carried
    across the look-mode action dimension by `add_look_mode.py` — see the `campaign_gates` entry below.
  - Cyber Grind training stays paused while the 5 games run the campaign.
  - `decompiled/` was regenerated on the second PC for this (ilspycmd 9.1.0.7988 in `.tools/`).
  - Spec amended during planning (2026-09-16): arena clears and door unlocks are position keys paid once per level
    load, `checkpoints_level` replaces `furthest_checkpoint`, room templates are filtered by `defaultRooms`
    ancestry, `level_seconds` is only reported for fresh starts, the campaign style reward is 0, and the mod blocks
    game-initiated restarts while the AI has control.
  - **Campaign progress tooling is in.** Episodes record `completed`, `fresh_start`, `level_seconds`,
    `checkpoints_level`, `cells_new` and `exit_dist_min`. `status.json` gains `best_checkpoints_level` and, once
    any episode reports `fresh_start`, a `campaign` block: `fresh_window`, `fresh_completion_rate` and
    `median_time_50` over the last 50 fresh starts, and `best_time` over all of them. Only fresh starts count
    because a checkpoint respawn starts partway through the level and the official timer carries over.
    `poll_status.py` logs these as columns (`part_time`, `part_checkpoint`, ... for the reward parts), and now
    keeps an existing `metrics_log.csv` header, dropping columns that header lacks: move an old log aside to get
    the new columns. `route_progress` is gone from `status.json`.
- **Campaign verified in game (2026-09-16), mod v0.5.0, one game at 30 fps / frameskip 2 with rendering off**
  (superseded by the v0.6.0 acceptance run at the end of this section, which repeats all of it plus check 6):
  - Level 0-1: 1 level load PASS | 2 arsenal SKIP | 3 checkpoint PASS | 4 death respawn PASS | 5 exit FAIL (exit
    room switched off at load; see the gotcha above).
  - Level 1-1: 1 level load PASS | 2 arsenal PASS | 3 checkpoint PASS | 4 death respawn PASS | 5 exit FAIL (exit
    room switched off at load; see the gotcha above).
  - Throughput smoke run with 5 games (`train.py --config configs/campaign_0-1.yaml --timesteps 20000
    --run-name campaign_smoke`, outputs deleted): finished cleanly at 177 steps/s; 13 episodes ended (stuck 13).
  - Next: the 0-1 pilot run (`campaign_ppo`).
- **`campaign.path` exit-sample fix, verified in game (2026-09-16).** `CampaignObserver.UpdatePath` sampled
  the NavMesh at `exit.transform.position` with a 20 m radius, but a `FinalPit`'s transform sits inside the
  drop it triggers, not on walkable ground, so that sample always missed and `path.status` read `none`
  everywhere on both 0-1 and 1-1 (matching the read above). Logging every `NavMesh.SamplePosition` candidate
  in game showed the raw depth figures (17.1 m below the floor on 0-1, 76.1 m on 1-1) do not tell you where
  to look: the true nearest NavMesh point was 47.5 m away **straight down** from the pit on 0-1, and 84.6 m
  away **up and 46.8 m to the side** on 1-1 -- not "a few metres directly above" as first assumed, so an
  initial fix that only probed upward (heights 18/30/50/80 at a tight 6 m radius) found nothing on either
  level. Replaced with `CampaignObserver.SampleExit`: keep the unchanged 20 m try at the pit's own position,
  then escalate the search radius itself (55 m, then 95 m) at that same position, letting NavMesh's own
  nearest-point search find whichever direction actually holds mesh. Both measured points connect back to the
  player's side only as `NavMeshPathStatus.PathPartial`, which `BuildPath` already reports as `"partial"` --
  expected, since a `FinalPit` deliberately sits off the walkable graph. `length`/`next_corner` are still
  measured to the snapped point (distance left to walk to standable ground near the exit), never a straight
  line into the pit. Verified live with one game (`games.py launch --count 1 --monitor 1`,
  `campaign_check.py`'s teleport-to-checkpoint approach against the level's farthest pending checkpoint):
  - Level 0-1: spawn `path.status` stays `none` (unrelated, pre-existing: the player-side 6 m sample already
    fails on the player's own spawn position, left untouched per scope). After teleporting to and activating
    the farthest checkpoint (`157,28,640`, 319.9 m from spawn): `partial`, length 72.4 m, a real `next_corner`.
  - Level 1-1: `partial` from spawn already (length 16.6 m) and after activating the farthest checkpoint
    (`46,12,550`, 300.7 m from spawn): `partial`, length 16.8 m, `next_corner` updated toward it. Before the
    fix both levels read `none` in both cases (confirmed by rebuilding and re-running against the pre-fix code
    first).
  - Noted, not changed (out of scope): the player-side NavMesh sample failing at 0-1's own spawn point.

- **Campaign pilot `campaign_ppo` (0-1, Violent), started 2026-09-16.**
  - Start: `models/campaign_ppo/transfer_init.zip`, the Cyber Grind v2 `best.zip` (7.16 kills/min, `ckpt_4679085_steps`) widened to 479 inputs by `transfer_weights.py --action-scale 0.5`. First-update entropy 6.93 nats against a 6-10 target; 199 steps/s with 5 games.
  - Stop with Ctrl+C in the training console and wait for `train.py` to exit (it prints `Saved models\campaign_ppo\latest.zip` last; `runs/campaign_ppo_train.log` is appended across restarts, so an older such line proves nothing), then resume with `python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/latest.zip`. After a hard stop resume from the newest `ckpt_*_steps.zip`, not `best.zip`: `keep_best.py --metric campaign` only replaces `best.zip` when the smoothed fresh completion rate beats its record, so until levels are being completed it holds the first checkpoint scored (rate 0) and resuming from it would throw the run away. Use `best.zip` to roll back once completions have peaked and fallen (keep_best warns). Helpers: `poll_status.py --run campaign_ppo`, `keep_best.py --run campaign_ppo --metric campaign`, `dashboard.py --run campaign_ppo --monitor 1`.
  - Ctrl+C leaves the exploration archives slightly stale: SB3's subprocess workers exit on KeyboardInterrupt without calling `env.close()`, so each `explore_*.npz` holds its last 20-episode save.
  - Checks: `best_checkpoints_level` >= 1 by 250k steps; >= 2 by 1M steps, else a 1M-step fresh-weights comparison run (`campaign_ppo_fresh`); eval gate once `campaign.fresh_completion_rate` >= 0.5 with `fresh_window` >= 20.
  - 1M-step falsifier passed: `best_checkpoints_level` 4, checkpoints per level load 0.13 -> 2.19 between 250k and
    1M steps, kills/episode 0 -> 15.1, deaths 0.03 -> 0.50, entropy 9.12. The transferred weights reach well past
    the first checkpoint, so no fresh-weights comparison run was needed.
  - **Watch novelty's share.** At 1M the parts are `novelty` 539.6, `time` -62.3, `kill` 9.3, `damage_dealt` 9.3,
    `checkpoint` 6.2, `arena_clear` 3.3, `door_unlock` 0.9 of 501 total: exploration is about 90% of the gross
    positive reward, the same one-term dominance that broke the Cyber Grind run's aim shaping. It is not yet
    decaying (new cells/episode 425 -> 1492) because 0-1 is big and the agent keeps finding real ground, and
    progress is genuinely improving, so it stays for now. Tripwire: if the fresh completion rate is still 0 at
    3M steps while `novelty` is above 70% of the gross positive reward, cut `novelty` 0.5 -> 0.15 so the
    milestones and the time cost drive instead, and give it 400k steps before judging.
  - **STOPPED at 2.49M steps, 2026-09-16, and the reward was wrong. The agent was farming the void.** Zero
    completions in 397 episodes; `best_checkpoints_level` frozen at 5 since 1.36M; checkpoints per level load
    peaked 2.54 at 1.33M and fell to 1.73; `exit_dist_min` 157 m -> 188 m. The cause, found by a 9-agent
    adversarial review and then confirmed straight from the five committed `explore_*.npz` archives:
    **`ExplorationArchive` was keyed on the raw 3-D player position with no bound on y**, so novelty paid for
    occupying *volume*, not for covering *ground* -- and the cheapest volume in a 3-D game is vertical.
    - Measured over all 364,984 novelty units the run ever paid: **47.1% below y = -60 m** (the void under the
      map, reaching **y = -57,304 m**), of whose cells **99.98% were entered exactly once**; another **47.7%**
      for air cells stacked above ground already paid for (the playable footprint was 1,397 XZ columns but
      15,251 cells, ~11 stacked per column). **Only 5.3% was ever paid for actual ground covered.**
    - Why it was stable: a fall enters a brand-new 4 m cell every decision at the full `1/sqrt(0+1)` = 1.0, the
      supply never runs out, those cells are never revisited so the decay never bites, there is no kill plane,
      and any `novelty > 0` reset the stuck clock -- so diving off the level was an unbounded income stream that
      could not be truncated. Finishing pays +100 once and *terminates*, forfeiting the stream. Under that
      reward, completing the level was never the argmax, and the policy was correct to refuse.
    - Do not mistake this for the shape of the level: `corr(fresh_start, checkpoints_level) = -0.975`, so the
      1.33M "peak by depth" is an artifact of respawn episodes inheriting a checkpoint count, not a better policy.
      A fresh level load essentially never reached even the first checkpoint
      (`checkpoints_level = 3.748 - 4.434*fresh_start`, ~0 at `fresh_start = 1`).
    - **Fixed** by keying novelty on the ground point under the player (`env._ground_point`), which closes the
      void faucet and the air slab with one bounded mechanism while still paying for new floor covered while
      airborne -- which matters, because fast movement in this game is airborne. `oob_frac` was added so the
      mechanism is directly observable. Also fixed: `level_complete` graded and paid before death/scene-change,
      so the first real completion cannot be booked as a death.
    - **Rejected during review, do not retry without new evidence:** a straight-line distance-to-exit reward
      (measured `exit_dist_min = 224.9 - 90.5*fresh_start`, i.e. the checkpoints sit ~90 m *farther* from the pit
      than spawn does, so it would pay the agent to reverse, and it is farmable by a void glide); a 2-D cell key
      (breaks the saved 3-tuple archives and deletes the vertical structure of a vertical level); cutting
      `novelty` 0.5 -> 0.15 as well (the measure change already cuts the income ~5-10x; doing both removes the
      only forward pressure the run has -- hold it in reserve); cutting `max_steps`; and touching `gamma`,
      `ent_coef` or `pitch_limit_deg` (entropy is 9.7 nats and rising, `approx_kl` 0.021 vs `target_kl` 0.03, and
      pitch is causally irrelevant to navigation -- `NewMovement` builds `inputDir` from horizontal projections).
      - **`pitch_limit_deg` was reinstated at 45 on 2026-09-16 and the "pitch is causally irrelevant" reason was
        wrong as stated.** Movement is indeed horizontal-only, but *shooting* is not, and the campaign config set
        no band at all, so the camera drifted to a mean of **-78 deg** with `on_target_frac` 1.5% — and all three
        live episodes that got past checkpoint 3 then died in the Stray arena between two locked doors with 0-1
        kills in 35-193 s. Four of 0-1's gate doors are held behind kill gates, so the agent has to be able to
        aim to walk the route at all. Pitch is irrelevant to *where you go* and decisive for *whether you get
        through the door*.
    - Keep the five `explore_*.npz` archives: their floor counts carry the `1/sqrt(N)` decay history that is the
      frontier-seeking engine, and a standing player's cell and its ground-point cell are the same cell. The ~347k
      air and void keys just become dead weight. They were copied into `models/campaign_ppo_ground/` for the
      restart, because `explore_dir` follows `run_name`.
- **Campaign run `campaign_ppo_ground` (0-1, Violent), started 2026-09-16** from
  `models/campaign_ppo/ckpt_2450000_steps.zip` -- the pre-fix run's own weights, since the reward was wrong, not
  the policy. New run name because `part_novelty` and `cells_new` change meaning across the fix and
  `poll_status.py` keeps an existing CSV header (it would silently drop the new `oob_frac` column).
  - Baseline at 28 episodes / 2.59M steps, against the pre-fix run's last window: `novelty` 395.6 -> **75.1**
    (93.9% -> **76.8%** of gross positive), `time` -59.1 -> **-94.7** and now the largest single term,
    `checkpoint` 5.10 -> **6.79** (1.2% -> 6.9%), `punch` **-17.13** (still pressing on ~35% of steps, now priced),
    total reward +501 -> **-16.9**, `oob_frac` **0.090**, episode length 5950 -> 4836, deaths 1.0 -> 0.19,
    entropy 10.6 and rising. A negative total is intended: wandering no longer pays for itself, and SB3 bootstraps
    `gamma * V(s_T)` on both `stuck` and `max_steps`, so the agent cannot escape the stream by ending the episode.
    `level_complete` is the only termination that stops the clock, and it also pays +100.
  - **Gates, in order. Judge on none of them early, and on no gate satisfied by the accounting change alone:**
    `novelty` and `cells_new` fall by construction and prove nothing on their own.
    1. **+400k steps (~2.99M): `oob_frac` well below the 0.090 baseline**, on a 100-episode window. Mechanism check
       only. If it does not fall, the ground-ray sentinel is not being read as intended -- debug it, do not tune
       weights. Some floor is legitimate: 0-1 has real drops, so do not expect zero.
    2. **+1.0M steps (~3.59M): `fresh_start` <= 0.32 and `best_checkpoints_level` >= 6.** The real gate.
       `choose_fresh_start` forces a reload whenever the previous episode ended with no current checkpoint, so the
       measured `fresh_start` rate is a direct readout of how often a fresh load fails to reach any checkpoint at
       all: `f_measured = f_forced + (1 - f_forced) * 0.2`. The pre-fix run ended at 0.49, i.e. 36% forced reloads,
       up from 10% at 1.13M.
    3. **+2.0M steps (~4.59M): `campaign.fresh_completion_rate` > 0** with `fresh_window` >= 20.
    Gate every comparison on `window >= 50`.
  - **BLOCKER CLEARED 2026-09-16: a completion does register through the bridge, confirmed on a human
    playthrough of 0-1.** Watched read-only with `scripts/watch_completion.py` while the user played:
    `level_started` false -> true at t 0.23 s; **`exit.active` false -> true at t 122.2 s**, after all 6
    checkpoints; `level_over` true at **t 146.582 s** with 59 kills. So the exit room *is* switched on by real
    play, `stats.level_complete || campaign.level_over` fires in the pit, and the whole path `env.py` grades a
    completion on works end to end. `campaign_check.py` check 5's FAIL was an artifact of the check itself
    (teleporting onto a collider in a room that is still switched off), not a broken chain.
    - **Human reference time for 0-1: 2:26.58 (146.58 s), 59 kills, 6 checkpoints.** That is the bar.
    - The run reported `cleared_arenas` 0 and `unlocked_doors` 0, which is expected and not a discrepancy:
      `CampaignPatches.RecordArenaClear` and `RecordDoorUnlock` are both gated on `EpisodeController.InControl`,
      so they only record while the AI is driving. The training run's `arena_clear` / `door_unlock` payments are real.
    - **Two automated attempts could not answer this, and the reasons are worth keeping.** (1) *Teleporting cannot
      play the level.* Hops of 5 m walked the player through 0-1's entire geometry to the pit while `enemies`,
      `locked_doors` and `cleared_arenas` all stayed empty and the exit stayed inactive: rooms ahead of the player
      are switched off, so their trigger volumes are off too, and a teleport into dead space fires nothing. The
      player then fell to y -507 and kept going, independently confirming there is no kill plane under this level.
      (2) *Scripted movement cannot leave the starting room.* Dropping to 1.2 m hops jammed the player against a
      closed door at z 347 that the 5 m hops had been tunnelling through, and a scripted walk-and-shoot spent
      63,000 decisions inside a 6 m box with 0 kills. A probe at spawn shows why: `level_started` false,
      `slot_counts` all zero, and walls 3.5-6.6 m away in all 16 ray directions.
      - **Correction (2026-09-16): "0-1 spawns you sealed in" is wrong as stated.** The starting room is sealed
        to a *teleport* and to a *walk*, not to a **slide**: the lower `Breakable` plank leaves a 1.5 m gap
        against a 1.25 m slide height, and a scripted grounded slide goes through it every time. Measured at
        integration: forward + slide from spawn reached the gun-room door at z 401 in 42 decisions, and the
        sampled policy passes the vent on 67-93% of fresh loads. The walk-and-shoot script failed because it
        never slid, not because there is no way out. See the alternate-start gotcha.
  - **Was open, now answered by the above: nobody had ever seen a completion register through the
    bridge.** `part_level_complete` is blank in all 544 rows of the pre-fix run and `best_runs/` is empty. The code
    path reads right -- `FinalPit.OnTriggerEnter` sets `nmov.levelOver`, `ObservationBuilder.cs:274` reads
    `sm.infoSent || nm.levelOver` (so detection does not depend on the campaign `exit` block), and `Door.Open()`
    (`decompiled/Door.cs:432`, loop 458-465) is what calls `SetActive(true)` on `activatedRooms`, which is why
    check 5's teleport-onto-an-inactive-collider failure is expected rather than evidence of a broken chain. But it
    is unverified. To settle it: one game, `python scripts/bridge_test.py --campaign` (it polls without taking
    control), then play 0-1 to the end by hand and watch `exit.active` flip true near the final room and
    `level_over` go true in the pit. If `level_over` never fires, no reward change can move the completion rate and
    the next work is mod-side.
  - `exit_dist_min` went 157.9 m -> 193.6 m over the same window. That is straight-line distance, and 0-1's route
    does not run straight at the pit, so it is not yet evidence of anything; it matters only if it keeps rising
    while checkpoints stop.

- **`campaign_ppo_ground` STOPPED at 2,914,785 steps, 2026-09-16, and diagnosed. Zero completions in 2.9M steps
  across two runs.** A five-investigator diagnosis, each report adversarially verified, found four causes. Spec:
  `docs/superpowers/specs/2026-09-16-campaign-gates-unwedge-design.md` (mod v0.6.0). Findings:
  - **F1 the wedge.** ~70% of sampled fresh loads (8/26 passed at 30/2, 3/15 at 60/4) ended in the absorbing
    airborne `slowMode` state. See the wedge gotcha. This alone was most of the run.
  - **F2 no route signal.** The NavMesh is an enemy mesh and `path` never paid. See the NavMesh gotcha.
  - **F3 combat gates the doors and the agent cannot aim.** The campaign config set no `pitch_limit_deg`, the
    camera sat at a mean of -78 deg, `on_target_frac` 1.5%, and all three live episodes that passed checkpoint 3
    died in the Stray arena between locked doors with 0-1 kills in 35-193 s.
  - **F4 smaller things.** The stuck clock never reset on kills and `_respawn` did not reset it; the ground run
    **saved no exploration archive at all** (episodes are ~3900 decisions, no worker reached 20 of them, and
    `SubprocVecEnv` workers never call `close()` on Ctrl+C); no per-episode log; `_ground_point` took the minimum
    of an 8-ray ring whose spread among hits is p90 **10.8 m**; `ChooseExit` could pick a secret-level pit on 0-2
    and 1-1; the trainer's OpenMP threads spin-waited on **11.0 cores** (`KMP_BLOCKTIME=1` -> 1.43, same update
    time); `Distribution.set_default_validate_args(False)` saves 0.95 ms of a 3.1 ms forward pass; and the resume
    path passed no `verbose`, which is why `campaign_ppo_train.log` was 0 bytes.
  - **One premise of the plan was refuted and is recorded rather than claimed away.** Observation inputs 448-455
    were said to have "read zero/none for the whole run". They had not: `path.status` was `partial` on **82.19%**
    of 32,022 logged decisions with a `next_corner` on every one, and `||W[:, 448:456]||_F` is 2.04 of 33.78. So
    re-using those slots displaces a working policy, which is why `add_look_mode.py` zeroes them **and folds
    their mean into the bias**, and why the new run's gates are read against a policy that moved before step 0.
    (Only column 453, `status == "complete"`, was genuinely dead — exactly zero norm in both stacks.)


## 2026-09-17 — The `campaign_gates` run: gates, un-wedge, look modes

- **Campaign run `campaign_gates` (0-1, Violent), integrated and ready 2026-09-17; NOT started.** Starting weights
  `models/campaign_gates/look_init.zip` = `campaign_ppo_ground/latest.zip` (2,914,785 steps) through
  `add_look_mode.py`. Measured on the 436 recorded 0-1 states: MU reproduced the spec's values exactly, mean KL
  **0.0232** (max 0.2079), greedy action changed **27.98%**, mean |dV| 0.627 over a value range -6.61..+64.05,
  optimizer carried. Zero-only, for comparison: 0.0388 / 46.33% / 1.130. So the fold roughly halves the
  displacement and the result is inside the run's `target_kl` of 0.03; the zero-only variant is outside it.
  - **Reward weights now:** `gate` 15.0, `gate_approach` 0.15, `door_unlock` 3 -> **15.0**, `novelty` 0.5 -> **0.2**,
    `path` 0.1 -> **0.0**, `pitch_limit_deg` absent -> **45.0**. Unchanged: `time` 0.02, `level_complete` 100,
    `checkpoint` 10, `arena_clear` 10, `kill` 0.5, `damage_dealt` 0.5, `damage_taken` 0.01, `death` 5, `style` 0,
    `punch` 0.01. Arithmetic for a route-walker at human pace: route (`gate` 150 + `gate_approach` 108 +
    `door_unlock` 60) = **+318**, the largest positive group, against `level_complete` +100, checkpoints +60,
    combat +60, novelty +40, time -44 — total ~+556. A wanderer that never leaves the start area scores **-101**.
    `door_unlock` moved to 15 because the route gradient is exactly **flat** at every arena-held door, which is
    where F3 says the run actually fails; on 0-1 the four arena-held doors are exactly the four gate doors.
    **Read the wanderer case honestly:** novelty is still 87% of gross positive income in that regime (69 of 79).
    The design makes the regime unprofitable, it does not make it un-dominated.
  - **Lead rulings.** R1: `GateProgress.best_dist` is **episode** scoped (cleared in `reset_episode`, *not* by an
    in-episode death respawn); the `gate` ladder stays level-load scoped. PPO returns are computed within an
    episode and a truncation bootstraps `gamma * V(s_T)` on the same state, so ending an episode can never be
    profitable, and most episodes are checkpoint respawns that would otherwise get no dense signal on ground an
    earlier episode covered. Rollback is one line. R2: `ObsLayout` stays **479** with the slot re-use and the
    mandatory zero + mean-fold (the 487-input variant is a drop-in alternative, flagged and not taken). R3: no
    `punch` change before `level_started`, no more than 5 games, no title-wait skip. R4: the run is
    **`campaign_gates`** and the weights `look_init.zip`, so `explore_dir` and the copied archives agree.
  - **Acceptance, measured at integration (one game, 30 fps / frameskip 2, rendering off, monitor 1):**
    - A1 gates on 0-1, fresh load **and** after a checkpoint respawn at `40,-12,485`: **PASS** — 11 gates,
      hops 0..9, every `key`, `pos` and `controller_active` identical to the spec's literal example both times,
      `gates_truncated` false, and `controller_active` false only on the hops 9 gun-room gate.
    - A2 1-1 ordered (13 gates, 0..5) and 0-5 unordered (3 gates, all `hops` null, no exception): **PASS**.
    - A3 `ChooseExit`: **PASS** — 0-2 picks (-199, -86.1, 277) and 1-1 picks (81, -76.1, 91), the non-secret pits.
    - A4 `campaign_check.py` on 0-1 and 1-1: **PASS** — every earlier verdict unchanged and the new check 6 PASS
      on both, fresh and after a respawn. (Check 5 remains the known standing exit FAIL.)
    - A5/A6 un-wedge, both flavours: **PASS** — each reproduced with `unwedge:false`, absorbing over 45 decisions
      of move+jump+dash (max axis movement 1.407 m and 1.419 m = exactly the 0.47 m/s creep), then **recovered on
      decision 6 = 0.400 s of game time** with no input at all, `slow_mode` and `heavy_fall` false and free fall
      resumed. Total position shift across the whole recovery 0.101 m and 0.122 m — gravity resuming over two
      frames, not a teleport (`GroundCheck.Bounce` would move the player metres). No enemy lost health, though
      the vent has no enemies in it, so that half is vacuous in this sample.
    - A7 slide after recovery: **PASS** — 15.04 m and 14.81 m in 10 decisions, against a 5 m bar.
    - A8 no regression elsewhere: **PASS** (mod side) — 27 of 28 probe tags within 0.5 m of the v0.5.0 baseline,
      and the legitimate grounded low-ceiling crouch still reports `slow_mode: true` on the same 230 steps.
    - A10 gate reward live: **PASS** — a scripted grounded slide through the vent reached the hops 9 gun-room door
      in 42 decisions and paid exactly one `gate` (15.0) and `gate_approach` **9.42** (0.15 x 64.8 m less the
      0.5 m deadzone), `gates_reached` 1, and the target advanced `40,1,408` -> `40,1,470`.
    - A11 look mode 2 live: **PASS** — from **185.6 deg off** (camera deliberately turned 180 deg away and pitched
      to the -45 band edge) and with a hostile +90 yaw / -20 pitch free-look bin underneath it every decision,
      mode 2 reached 6.33 deg on decision 2 and 0.00 deg on decision 3. The two steps are exactly the 90 deg and
      20 deg per-decision caps. On a fresh spawn the target slots read
      `[0.006, 0.030, 1.296, 0.648, 1.0, 0.0, 0.0, 0.45]` — rel z 64.8/50, dist 64.8/100, mask, and hops 9/20.
    - Look mode 1 live: **PASS** — camera-space yaw +26.84 -> -5.12 and pitch -14.51 -> -0.12 in one decision,
      both axes the right sign, converging over the next four.
    - A12 throughput: **PASS** — **184-185 steps/s** with 5 games on 0-1, against a 150 bar (152 before).
    - A9 vent pass rate, 15 fresh loads x 1500 decisions: **PARTIAL, 10/15 = 67%** against an 80% bar — see below.
  - **A9 is the one number below its bar, and the cause is measured, not guessed.** The same widened model, same
    build, same archive and same budget scores **14/15 = 93% with the look-mode dimension pinned to 0**, against
    10/15 = 67% with it sampled. So the drop is the look modes commandeering the camera, **not** the weight
    surgery — the widened weights on free look reproduce the mod side's 93% exactly. The smoke run confirms the
    mechanism: `look_free_frac` 0.652, `look_gate_frac` 0.348, `look_enemy_frac` 0.000 (mode 1 falls back on every
    step of a spawn room with no enemies), so ~35% of decisions have the camera taken over, and that alone costs
    26 points of pass rate. Movement is camera-relative, so overriding the look head also overrides whatever
    movement reflex 2.9M steps built around it. The action head starts uniform (three zero logit rows), so this is
    expected to be transient — learning which mode to select is goal G4. Baseline for comparison: **31%** before
    the un-wedge fix, 93-100% with it and the old policy.
    Spec 11.2's pre-committed outcome for a 31-80% landing applies: the mod fix ships, the Python wedge detector
    stays as the bound, and `slide_min_hold` is the next experiment with A9 re-measured with it on.
  - **The un-wedge fix itself is working perfectly.** `wedged_steps` — run gate 1's own metric, against a baseline
    of **877.7 per episode** — read **0 on all 15 A9 episodes** and 0 on all 3 smoke episodes. The 5 A9 failures
    are not wedges: 4 are the policy climbing to y 13-21 in the spawn room and never entering the vent (the mod
    side saw the same failure mode at y ~116), and 1 entered the tunnel at decision 1489 and ran out of the 1500
    budget 11 decisions later.
  - **Smoke run, 5 games, 20,450 new steps, finished cleanly:** 184-185 steps/s, `approx_kl` 0.020-0.024 against
    `target_kl` 0.03, `explained_variance` 0.93, `entropy_look_mode` 1.06 (ln 3 = 1.0986, i.e. still near-uniform
    and starting to move), `entropy_yaw` 1.86, `entropy_pitch` 1.44. `episodes.jsonl` written with every spec 9.7
    field; `gate_approach` paid live; `status.json` carries `best_gates_reached`, `best_gate_hops`,
    `mean_fresh_100` and the three per-dimension entropies; `mean_gates_reached_100` is in `history`; the
    dashboard renders. Outputs deleted.
  - **Run gates, in order** (spec 11.3; judge on none of them early, `window >= 50` on every comparison):
    1. **+200k:** `wedged_steps` mean < 20 per episode (baseline 877.7) and `end_reason == "wedged"` under 5% of
       ends. Mechanism check only — if it fails, debug the mod, do not tune weights.
    2. **+500k:** `gates_reached` mean >= 3 **on fresh starts** and `fresh_start` <= 0.32. **This is a combat
       gate, not a navigation gate:** on 0-1 `gates_reached >= 1` needs 11 kills in the gun room and `>= 3` also
       needs the projectile arena, against F3's measured 0-1 kills per episode. A failure here does not mean the
       route signal is broken — read it with the `kill` and `door_unlock` parts.
    3. **+1.5M:** `best_gate_hops` <= 2 on fresh starts.
    4. **+2.5M:** `campaign.fresh_completion_rate > 0` with `fresh_window >= 20`.
    - **Novelty tripwire:** if at +1.5M `gates_reached` is still <= 1 on fresh starts **and** `part_novelty` is
      above 70% of gross positive reward, cut `novelty` 0.2 -> 0.1 and give it 400k steps.
    - **Falsifier for the whole design:** if `gates_reached` is still <= 1 at +1.5M while `wedged_steps` is near 0
      and `part_gate_approach` is being paid, the route signal is not the binding constraint and the next suspect
      is combat (F3), not more shaping.
  - **`fresh_start` as a gate is confounded and must be read as a ratio, not as a rate.**
    `choose_fresh_start` forces a reload whenever the previous episode ended with no current checkpoint, so the
    measured rate is `f_measured = f_forced + (1 - f_forced) * 0.2` — it is a readout of how often a fresh load
    fails to reach any checkpoint, not an independent setting. The same inheritance confounds `gates_reached` and
    `checkpoints_level`, which a respawn episode inherits from its level load: the pre-fix run's
    `corr(fresh_start, checkpoints_level) = -0.975` is what produced the false "peak by depth" at 1.33M. Chart
    and judge both on **fresh starts only** — that is what `mean_fresh_100` and `best_gates_reached` are for.
  - Known open, carried into the run: levels beyond 1-1 are unverified for gates (only 0-1, 0-2, 0-5 and 1-1 were
    checked live); `controller_active` is false for a door with no `DoorController` at all, which reads as "a
    fight gates it" but never fires on 0-1..1-1; and the spawn-room climb that costs ~27% of fresh loads is
    unrelated to the wedge and still unaddressed.


## 2026-09-17 — Branch `next-levels` / mod v0.7.0, and integration pause #1

- **Branch `next-levels` (mod v0.7.0) — MERGED into `main` and verified in the game, 2026-09-17** (merge commit
  on top of `c20e97d`, which is the rollback target; see the integration entry below for the readouts). It was
  built in worktree `F:\Github\ULTRAKILL-AI-next` off `7da1204`. Four pieces from
  `docs/superpowers/specs/2026-09-17-multi-level-and-skull-gates-design.md`, built while the `campaign_gates` 0-1
  run stayed live and untouched:
  - **S1 multi-level curriculum.** Optional ordered `env.levels`; a level changes only on a **fresh** load and is
    sampled from the unlocked set with weight `max(floor, 1 - that level's fresh completion rate)`, floor 0.1.
    Level k+1 unlocks when level k's fresh completion rate over its last 20+ fresh episodes reaches
    `unlock_rate` (0.5); `levels[0]` is always unlocked and the unlock is a latch that survives `_restore`. The
    workers are `SubprocVecEnv` subprocesses, so the only shared state is `runs/<run>/curriculum.json`, written
    atomically by `ProgressCallback` and re-read at each fresh start; missing, torn, wrong-run or wrong-order
    falls back to `levels[0]` with one warning per worker. **No level id enters the observation** — generalisation
    by design. Per-level exploration archives, `best_runs/`, `episodes.jsonl` rows, `status.json`'s
    `campaign.levels` table and the dashboard's `levels` block all key on the level name.
  - **S2 gates usability guard** (Python only): a ladder whose phase-1 gates carry a `hops` value on fewer than
    half of them is ignored entirely and `GateProgress` falls back to the exit vector. 7-2 reports
    `gates_ordered: true` with 1 of 4 and 8-3 with 1 of 32, where the old code locked onto that single ordered
    door — 892 m from the start on 8-3 — and never retargeted.
  - **S3 6-2 exit tie-break** (mod): `Scan` discards a `FinalPit` targeting `Level P-` (a Prime Sanctum) as it
    already did `-S`, `ChooseExit` counts an `Intermission*` target as leading onward (an act finale's real exit
    drops into an intermission, and `Intermission2` does not parse as `Level a-b`), and a rank tie is logged once
    per level load. Before this, 6-2's three candidates ranked equally and the exit was whichever
    `FindObjectsOfType` returned first.
  - **S4 skull-carry gates** (mod + Python): `campaign.altars[]`, `campaign.items[]` and `gates[].needs_item`;
    one shared door-key table so an altar's door key string-matches a gate key by construction; a phase-2 gate
    pass appending altar-driven one-room doors as `altar_only` gates (adding no node and no edge, so no existing
    `hops` moves); a `GateProgress` sub-goal machine walking fetch -> carry -> open through observation slots
    448-455; two once-per-level-load milestones `item_pickup` / `item_placed`; and env-side carry protection.
  - **The action space did not have to change, and this was the one thing S4 could have broken.**
    `Punch.AltHit` does both pick-up (`ForceHold`) and place (`PlaceHeldObject`) off the existing `punch` button,
    which `ActionInjector` already binds to the game's own `input.Punch`. So no new button, **no weight surgery
    and no `add_look_mode.py`-style migration**: `models/campaign_gates/best.zip` loads into the new env as is.
  - **Verified:** `dotnet build -c Release -p:InstallPlugin=false` at 0 warnings / 0 errors, and all twelve
    no-game test files pass (248 named tests). Degradation is tested in both directions — new Python reads a
    missing `altars`/`items`/`needs_item` as "no altars", an old Python ignores the new fields.
  - **Nothing was verified in the game while the branch was written** (no game launched, no port opened). The
    merge-and-verify checklist `docs/superpowers/plans/2026-09-17-next-levels-integration.md` was run at the
    2026-09-17 pause; its results are the integration entry below. S1+S2+S3 are verified live; **S4 is not**, and
    its two weights stay at 0.0.
  - **The S4 tooling blocker is cleared: `scripts/skull_check.py` exists** (2026-09-17). It was the one gap — the
    spec's in-game checks 1-3 need an AI-driven punch at 1-1's red pedestal **with rendering off**, which a human
    playthrough cannot test (`render: false` only applies while the AI has control) and which no existing script
    could do (`walk_to_exit.py`'s waypoints are the level's checkpoints and it never punches; `campaign_check.py`
    teleports and no-ops; `bridge_test.py --campaign` is read-only). It walks rather than teleports, reads
    `items[].active` before punching so a switched-off room is never mistaken for a broken `ActiveStart`, and is
    green against `FakeLevel`'s skull room in `tests/test_skull_check.py`, so the pause only has to run it. **It
    has still never been pointed at the game**, and check 1 remains the kill switch: if the fist Animator is
    culled under `render: false` **with** `TrainingSpeed.ForcePlayerAnimators` applied, S4 is inert under training
    settings and nothing else in it matters.
  - **Known, deliberate deviation from the brief:** `keep_best --metric campaign` was asked to score the **mean**
    fresh completion rate over unlocked levels. It scores a **shrunk sum** instead, and `keep_best.py` is not
    edited at all. A mean drops at every unlock (~0.52 -> ~0.26 when the second level joins), and `keep_best.py`
    only replaces `best.zip` on a strict improvement — so a mean would freeze `best.zip` on a single-level policy
    and print the "15% below best" warning forever. A sum cannot drop, since a new level contributes 0 and grows;
    with one unlocked level at `fresh_window >= 20` it is exactly the rate that field has always been. The
    per-level means are all in `status.json`'s `campaign.levels`.
  - **Fixed 2026-09-17: `_protect_carry` was gating on "anything held".** It dropped `punch` on any step where any
    `items[]` entry read `held: true`, so on 0-4 — in the prelude curriculum, shipping a `CustomKey1` and **zero**
    `ItemPlaceZone`s — picking the key up cost the agent the button for the rest of the carry, including the throw
    that would have given it back (throwing is itself a punch). It now gates on a **carry**: a held item some live
    unfilled altar accepts, through `campaign.wanting_altars`, the same filter `GateProgress._subgoal` chooses a
    destination with. Both harms the rule exists for need a destination, so where nothing accepts the item there is
    nothing to protect. Pinned by `test_punch_passes_through_while_carrying_an_item_no_altar_wants` (0-4's shape)
    and `..._the_altars_do_not_accept` (the type filter, which is what 1-1's two skull colours need); the wired
    case still gates, unchanged. **The forced look mode 2 was checked and is not over-broad**: it keys on the
    target carrying a `subgoal` field, which only `_subgoal` sets and only for a gate with `needs_item` and a live
    wired altar, so it is already inert on 0-4 — asserted in the same test.
  - Also carried: S3 fixes 6-2 only. 3-2 keeps two `Intermission1` pits at an identical position and 2-4 two
    `Level 3-1` pits 1.4 m apart, and 8-4's `EarlyAccessEnd` pit parses as neither a level nor an intermission and
    is still picked by uniqueness alone — all three now visible through the rank-tie warning rather than silent.
    Gate phase 2 gives a usable `hops` on only three levels (1-1, 5-3, 8-1); elsewhere the widened door's rooms
    are outside the room graph, so `needs_item` exists but nothing on the route carries it. That is the
    out-of-scope checkpoint-chain fallback, not a defect.

- **Integration pause, 2026-09-17: `next-levels` merged, verified in game, 8 games, `ent_coef` cut, run resumed.**
  The live 0-1 run was stopped at **4,805,630 steps** (343 episodes, 12 completions, all from checkpoint
  respawns; fresh `gates_reached` 4.2 of 10, `wedged_steps` 0, `on_target_frac` 5.7%) and resumed on the merged
  code as the **same run** — same `run_name`, same `models/campaign_gates/`, same weights — now multi-level.
  - **Stopping it: `latest.zip` beat the newest checkpoint.** `games.py stop` kills the trainer with an
    `EOFError`/`BridgeError`, and `train.py`'s `finally` still saved `latest.zip` — 4,805,630 steps against
    `ckpt_4764785_steps.zip`'s 4,764,785. Check both before resuming; do not assume `latest.zip` is stale after a
    non-graceful stop, because this kind of stop is not the hard kill that leaves it behind.
  - **Merge:** clean, no conflicts (`main` had moved only under `python/models/`, which the branch never touches).
    Full no-game suite on `main` after the merge: **13 files, 258 tests, all pass**; after the pause's own changes
    **14 files, 263 tests**. Mod built and installed at **69,632 bytes** (0.6.0 was 58,880).
  - **In-game readouts** (one game, port 47800, 30 fps / frameskip 2, rendering off, Violent, all gear):
    - **0-1 regression — PASS, identical to v0.6.0.** `summary: 1 PASS | 2 SKIP | 3 PASS | 4 PASS | 5 FAIL |
      6 PASS`, 11 gates, hops 0..9, one inactive controller, unchanged after a respawn; `gates: 11 ordered=True
      truncated=False with hops=11 (1.000)`; **`altars: 0` and `items: 0`** as present-and-empty lists, no
      `needs_item` anywhere. S4 is invisible on the level the policy has 3.2M steps on, which is what mattered.
      (Check 5 is the known standing exit FAIL on every level; it was not chased.)
    - **1-1 altars/items/needs_item — PASS on 5 of the 6 predictions.** `altars: 7` with exactly the predicted
      keys; `items: 5`, all `placed=true`; gate `81,-6,240` present with `altar_only`, `hops=1`, `needs=SkullBlue`;
      gate `20,-10,381` `hops=2` `needs=SkullRed`; gate `16,20,427` carries no `needs` (a reverse door).
      `campaign_check.py` check 6 reports **14 gates** = 13 phase-1 plus the one `altar_only`, as predicted.
      **The sixth is a partial miss and matters for S4 only:** the dead-twin marker was predicted to be
      `inactive_ancestors=2` on all three `#2` altars. `-15,27,427#2` and `0,-7,381#2` read 2, but
      **`81,-4,251#2` reads 1**, and `CampaignObserver.NeedsItem` skips an altar only above 1 — so on the blue
      leg an unfilled twin would keep `needs_item` set after the real altar is filled (spec risk M14). Whether
      that twin is a dead duplicate or a second altar the level really requires is **unresolved**; it blocks
      raising the item weights, which are staying at 0.0 anyway.
    - **0-2 — PASS on altars/items, one prediction contradicted.** `altars: 2` and `items: 2`, all `SkullBlue`,
      as predicted. But the plan expected **no** `needs_item` anywhere, and the phase-2 pass adds an `altar_only`
      gate `-60,-6,236` with `needs=SkullBlue` and **`hops=None`**. It is off the ladder (`gates: 18 ordered=True
      ... (0.944)`), so the route is unaffected, and 0-2 is not in the curriculum. Worth knowing before 0-2 is
      trained: a carry there would make `_protect_carry` gate the punch, since the unfilled altar reads
      `inactive_ancestors=1` and so counts as live.
    - **0-4 — PASS.** `altars: 0`, `items: 1` (`CustomKey1`), no `needs_item`, `gates: 7 ordered=True (1.000)`.
      With no altars `campaign.wanting_altars` is empty, so the punch-gating bug fixed on this branch cannot fire
      — and the key did read `held=True` in the game, so a carry on 0-4 is real, not hypothetical.
    - **6-2 exit tie-break (S3) — PASS.** Three separate loads all report the **same** exit
      `(-342.0, -70.1, 350.0)`, and **no rank-tie warning** in `LogOutput.log` — the two `Level P-` Prime Sanctum
      pits are discarded before ranking, leaving one candidate at the best rank. Check 6 reporting 0 gates on 6-2
      is expected (no gate candidates). A benign `2 door position key(s) collide in Level 6-2` warning does
      appear, which also proves warnings reach the log at all.
    - **7-2 usability guard (S2) — PASS.** `gates: 5 ordered=True truncated=False with hops=1 (0.200)`. The plan
      predicted ~4 gates at 0.250; the count differs, the verdict does not — 0.200 is below the 0.5 threshold, so
      the ladder is discarded and the env falls back to the exit vector.
    - **Skull checks (S4, spec §8 checks 1-3) — FAIL on check 1, 2 and 3 SKIP. The walk failed, not the punch.**
      Two targets were tried inside the time box. (a) The spec's pedestal `81,-2,275`: the walker stalled
      **68.4 m short** at `(14.0, 0.2, 261.4)` after 4000 decisions. A compass probe from 1-1's spawn
      `(0, 0.5, 253)` explains it — **only heading 0 (+z) is open**, reaching `z 400.6` in 150 decisions, while
      all seven other headings stop within 15 m. The pedestal sits at the far end of the route (its gate is
      `hops=1`, i.e. next to the exit), so it is not reachable on foot from spawn without playing the level.
      (b) The red skull sitting in the red altar at `0,-7,381`, straight up that open corridor: reached to
      **1.4 m** in 159 decisions and punched 56 times, but it reads **`active_self=False`, `inactive_ancestors=1`**
      even with the player standing on it — the object itself is disabled, so nothing could pick it up. **So the
      `Punch.ActiveFrame`-under-`render:false` question (check 1's kill switch) is still unanswered**, and
      `item_pickup` / `item_placed` stay at **0.0**. S4 is not needed for the prelude levels. Next attempt should
      start from an activated checkpoint rather than a fresh spawn, since the pedestal reads `active=True
      inactive_ancestors=0` once checkpoint `81,-6,231` has been activated.
  - **Smoke run (plan §6) — PASS on every item**, folded into the throughput A/B rather than run separately:
    `curriculum: 3 levels, unlocked ['Level 0-1']` printed before learning; `curriculum.json` names all three in
    order with only 0-1 unlocked; every `episodes.jsonl` row carries `"level": "Level 0-1"`; one level-keyed
    `explore_Level_0-1_478NN.npz` per game; `dashboard.py --smoke-test` renders; `metrics_log.csv` carries
    `levels_unlocked`, `part_item_pickup` and `part_item_placed`; `part_gate_approach` 8.4 (**not** collapsed, so
    the S2 guard is correctly not firing on 0-1) and `part_item_*` absent, i.e. 0.
  - **The one training change: `ent_coef` 0.01 -> 0.004.** Evidence at the stop: total entropy **11.8 nats of a
    13.59 maximum and still rising**, with all three look heads statistically uniform (yaw 2.19 of ~2.20, pitch
    1.78 of ~1.79, look mode 1.04 of ~1.10), `on_target_frac` 5.7% and fresh `gates_reached` flat at ~4.7 of 10
    with **no fresh-start completion in 343 episodes**. The Cyber Grind precedent is the same cut giving that
    run's biggest single gain, and its failure mode is known and guarded: below ~3 nats it went deterministic and
    regressed, so `keep_best.py --metric campaign` holds the weights and there is an entropy floor tripwire.
    Config hyperparameters do override a checkpoint's saved ones on `--resume` — `train.py` passes them into
    `PPO.load`, whose `model.__dict__.update(kwargs)` applies them — and it was confirmed live: within the first
    seven updates entropy went **11.8 -> 11.4/11.5 and stopped rising**, yaw 2.19 -> 2.07-2.12, pitch 1.78 ->
    1.68-1.73, look mode 1.04 -> 0.985-1.01. **Falsifier:** entropy should fall toward **6-8 nats within ~1M
    steps** *while* fresh `gates_reached` rises above 5 and fresh completions appear. Entropy falling without
    `gates_reached` rising means the cut is not the binding constraint — put it back. **Tripwire: total entropy
    below 6.0** — stop and raise `ent_coef`.
  - **Eight games instead of five**, after the BepInEx disk-log cap turned out to be the only limit. 235 steps/s
    against 5 games' 152 (+54%) at 33% CPU — see the two instance-count gotchas for the table and how to undo it.
    In the live run, which reloads levels far more often than a 90-second A/B does, it settles at **~186 steps/s
    per rollout** against the pre-pause run's 159 with 5 games. Read the A/B as the ratio, not as an absolute:
    both A/B legs ran short enough to see few level loads, which is what makes the absolute numbers optimistic.
  - **Resumed and running** on `configs/campaign_gates_prelude.yaml` from `models/campaign_gates/latest.zip`
    (4,805,630 steps) with 8 games; `poll_status.py`, `keep_best.py --metric campaign` and the dashboard alongside.
    The old `runs/campaign_gates/metrics_log.csv` was moved to `metrics_log_pre_multilevel.csv` so the new columns
    are written; the three new ports' exploration archives (47805-47807) were **seeded from 47802-47804's**, since
    a from-zero archive would re-pay novelty for ground the run has already covered.
  - Watch for, in order: the entropy falsifier above; fresh `gates_reached` past 5; the first **fresh-start**
    completion (the 12 so far are all checkpoint respawns); and then `levels_unlocked` going 1 -> 2, at which
    point the pooled `fresh_completion_rate` stops being readable as a percentage and only the per-level rows
    mean anything.
  - **ent_coef falsifier PASSED at +727k steps, and the first FRESH-START completions (2026-09-17 06:32,
    5,532,670 steps).** Entropy 11.43 -> 10.16 -> 9.19 with all three look heads falling together (yaw 2.09 -> 1.71,
    pitch 1.79 -> 1.24, look_mode 1.03 -> 0.81), fresh `gates_reached` 4.56 -> 4.88 -> 5.74, completions per 100
    episodes 2 -> 5 -> 17, `approx_kl` 0.0216, `explained_variance` 0.90. **Two fresh-start completions of 0-1:
    479.71 s official (64 kills, 2 deaths; `campaign.best_time`) and 522.12 s** (human reference 146.58 s).
    Weights kept as `models/campaign_gates/first_fresh_completion_5505630.zip`; `fresh_completion_rate` 0.04 on a
    full 50 window; 29 completions all-time.
    - Entropy is still descending at about 1.0 nat per 300k steps, which reaches the 6.0 tripwire near 6.5M steps.
      Plan at that point: raise `ent_coef` (0.006-0.008) to hold entropy around 6-7 rather than let it run down.
    - **Fresh episodes are bimodal** (last 100 fresh): 38 end early (gates 0-2, `stuck`, ~3,850 decisions; 22%
      never reach checkpoint 1) and 45 get deep (gates 7-8) but end on `max_steps`. The two completions used 7,967
      decisions on average, 88% of the 9,000 cap (600 game seconds), so deep runs are near-misses against the
      clock. Candidate for the next planned pause, as its own change: `max_steps` 9,000 -> 12,000. Not done
      mid-climb because it needs a trainer restart and the run is improving on its own.


## 2026-09-17 — Skull carry: S4 verified in game, and branch `skull-fixes`

- **S4 skull carry VERIFIED IN GAME on Level 1-1, 2026-09-17 — all three of spec section 8's checks pass, and
  the blocking question was never the one everyone was watching.** Run on a **ninth** game instance on port
  47808, launched by hand with `games.py`'s own arguments (`-aibridge-port 47808 -screen-fullscreen 0
  -screen-width 368 -screen-height 207 -job-worker-count 3`, `SteamAppId` set, detached + below-normal) while
  the `campaign_gates` run kept using 47800-47807 untouched. **Never run `games.py launch` or `stop` for this:
  both call `stop_all()`, which kills every game including the trainer's eight.** Training held 158-175 steps/s
  throughout.
  - **Check 1 pickup — PASS with `render: false`.** One aimed punch at 1-1's red pedestal `81,-2,275`
    (reading `active=True active_self=True inactive_ancestors=0`) flips `items[].held` true. **So the
    `ActiveStart`/AnimationEvent worry is dead**: `TrainingSpeed.ForcePlayerAnimators` does keep the fist
    Animator running with every camera disabled, and S4 is not inert under training settings. The `--render`
    control behaves identically, which is how the real cause was found.
  - **Check 2 placement — PASS.** Carried to the red altar `0,-7,381`: one aimed punch sets `filled: true` and
    **the gate `20,-10,381` goes `open: true`** — the door really unlocks. The placement then survived
    **100 decisions of punch spam issued through `env.step`**, so **`_protect_carry` holds in the live game**,
    not just against `FakeLevel`.
  - **Check 3 held skull on death — PASS, and the answer is "it comes with you".** `kill()` while holding, then
    the respawn: the skull reads `held: true` still, at the checkpoint (`81.7,-3.6,229.6`, 2.3 m from the
    player), `deaths` 0 -> 1, the episode does not end. The level's other `SkullRed` entry is the destination
    altar's decoration (`active_self: false`). **A death mid-carry costs the carry nothing** — no re-fetch, and
    no unsolvable-puzzle failure mode to design around.
  - **THE REAL BUG, and it is Python-side: every aim at a skull or an altar is taken from the wrong point.**
    `Punch.ActiveFrame` (`decompiled/Punch.cs:547`) rays from **`cc.GetDefaultPos()` — the camera** — along
    camera forward for 4 m. `skull_check.py` and **`env._look_at_target` (look mode 2)** both compute the
    elevation from `player.pos`, i.e. `NewMovement.transform.position`, which sits **0.9 m below the camera**.
    At punch range that is a ~30 degree error and the ray passes clean over the target: measured on the pedestal
    at 1.57 m, aiming from `player.pos` asks for +27.0 deg and fails, the camera wants -3.0 deg and picks it up;
    at 2.71 m, +11.3 fails and +8.0 works. Both solve to **0.88-0.90 m**. The two earlier in-game attempts —
    including the one that concluded "the walk failed, not the punch" — were **59 aimed punches at a live
    pedestal 1.8 m away that all flew overhead**, with rendering on *and* off.
    - `scripts/skull_check.py` is fixed: new `eye(player, height)` helper, `CAMERA_HEIGHT = 0.9`, a
      `--camera-height` override (the offline tests pass `0`, since `FakeLevel` has no camera).
    - **`ultrakill_ai/env.py` is NOT fixed and needs the same correction before `item_pickup`/`item_placed` ever
      leave 0.0.** `_look_at_target` must aim from `player.pos + (0, 0.9, 0)` whenever `wide` is set, and
      `_near_subgoal`'s range test is measured from `player.pos` too. Left alone deliberately: the trainer
      imports this file and the live curriculum (0-1/0-3/0-4) has no wired altar, so mode 2 only ever aims at
      doors, where 0.9 m is under 3 degrees at 20 m and harmless. **It is not harmless at 4 m**, so as it stands
      the agent could never place a skull even with the weights raised.
    - **A second aim offset, not yet explained.** Aiming at `altars[].pos` exactly does not place: the placement
      only fired when aimed **~1.25 m below** the zone's reported position (`0,-7,381` -> aim y -8.0). Likely
      the ray hits the zone's decoration skull, which carries an `ItemIdentifier`, and `AltHit` bails on
      `if (itemIdentifier && hasHeldItem) return;` (`decompiled/Punch.cs:1341`). Pickup needs no such offset.
      Whatever `env` ends up doing for mode 2 has to clear this too; `skull_check.py` takes it as `--altar`.
  - **Data oddity 1 (M14) CONFIRMED AND REPRODUCED, on the red leg as well: `needs_item` does NOT clear after a
    real placement.** With `0,-7,381` filled and its door open, gate `20,-10,381` still read
    `needs_item: 'SkullRed'`. Cause measured directly: **`inactive_ancestors` conflates "my room is switched
    off" with "I am a dead twin"**, and the number moves when the room switches on. On a fresh load the live
    altar `81,-4,251` reads 1 and its twin `#2` reads 2; **after the checkpoint respawn switches that room on,
    the live one reads 0 and the twin reads 1**. `CampaignObserver.NeedsItem` skips only `> 1`, so the twin
    stops being filtered exactly when the player arrives, and holds the lock set forever. This is a **mod fix**:
    the dead-twin test must be relative, not a constant — a zone is dead when another zone with the same `item`
    and the same `doors` reports strictly fewer `inactive_ancestors` (equivalently, once the room is on, the
    live zone is the `active` one). Until it is fixed a skull-locked gate never reports itself solved, so
    `GateProgress._is_reached` refuses the gate forever and the fetch/carry machine cannot advance to `open`.
  - **Data oddity 2 (0-2's `-60,-6,236 needs=SkullBlue hops=None`) — cannot become a target; no mod fix needed
    for that.** `GateProgress._choose_target` filters on `g.get("hops") is not None` (campaign.py:709) and
    `_is_reached` returns False for `hops is None`, so a `hops: null` gate is invisible to targeting and to the
    ladder. **But there is a latent trap if 0-2 is ever added to the curriculum:** `_protect_carry` keys on
    `campaign.wanting_altars`, which does not look at gates at all, so picking that blue skull up drops the
    punch button — while `_near_subgoal` can never release it, because releasing needs a `subgoal` target and
    0-2 has no targetable gate to build one from. Fix when 0-2 is trained, not before: either give the phase-2
    gate a usable `hops` mod-side, or release the punch near any unfilled altar that accepts the held item when
    there is no sub-goal.
  - **How to reach the pedestal at all** (the previous attempt's blocker, now solved and folded into the
    script): 1-1's route to `81,-2,275` runs **through the skull-locked door `81,-6,240` itself**, so no walk
    from spawn can get there. `--from-checkpoint 81 -6 231` teleports onto that checkpoint and lets it activate,
    **and then respawns** — and the respawn is the part that matters: after the teleport alone the pedestal
    still reads `active: false`, and only `reset(checkpoint=True)` brings the room up to
    `active: true, inactive_ancestors: 0`. Activating `46,0,388` first also lights the red altar's room, and it
    survives the later respawn elsewhere. `--approach` / `--altar-approach` then stage into those now-live rooms
    and the last metres are still walked. The whole run:
    `python scripts/skull_check.py --port 47808 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231
    --approach 81 -1 255 --altar-approach 0 -4 374 --altar 0 -8 381 --budget 700`
  - **`item_pickup` / `item_placed` stay at 0.0** — the lead decides, and the `NeedsItem` mod fix and the
    `env._look_at_target` aim fix both have to land first or a raised weight buys nothing.

- **Branch `skull-fixes`, 2026-09-17: the four skull blockers fixed, three of them verified in game on a ninth
  instance.** Built in the worktree `F:\Github\ULTRAKILL-AI-skull` while `campaign_gates` kept training on
  47800-47807; the mod compiles (`dotnet build -c Release -p:InstallPlugin=false`) but could not be installed,
  because the running games hold the DLL. **Not merged into `main`.**
  - **F1 -- aim from the EYE, not the feet (Python).** `Punch.ActiveFrame` (`decompiled/Punch.cs:562`) rays from
    `cc.GetDefaultPos()`, the camera, which sits 0.9 m above `player.pos` (`NewMovement.transform.position`).
    `env._look_at_target` (look mode 2) and `_near_subgoal`'s reach test both measured from the feet, so every
    aim at a skull or an altar pointed over the top of it. New `EnvConfig.camera_height_m` (0.9; 0 restores the
    old behaviour) and `env._eye`.
    - The error is a short-range one and the correction is in the right direction everywhere: `atan(0.9/r)` is
      **29.8 deg at 1.57 m**, 16.7 at 3 m, **5.1 at 10 m, 1.7 at 30 m**. So a gate 10-30 m away moves by a couple
      of degrees, far inside any aim tolerance, and moves the correct way -- a camera 0.9 m up really does have to
      look slightly down at a door sill at its own feet's height. At punch range it is the whole answer.
    - **Look mode 1 needed no change, and that was verified rather than assumed.** `enemies[].rel` is
      `cam.InverseTransformPoint(centre)` (`ObservationBuilder.cs:246`) -- already measured from the camera -- and
      the only other use of `player.pitch` there is un-pitching `rel` into the yaw frame, a rotation about the
      same origin. `test_look_mode_1_is_already_camera_relative_and_is_left_alone` pins it at three camera
      heights: adding an eye offset there would introduce the very error mode 2 had. The mod reports **no camera
      position**, so the 0.9 is a constant; a future `player.cam_pos` would make it exact while crouched.
  - **F2 -- the placement offset explained, and it was NOT the decoration skull.** The `AltHit` `ItemIdentifier`
    early-return hypothesis is **wrong for this case**: 1-1's live red altar (`4 - Altar Field/Altar/Cube`) has no
    child skull at all. The real cause is geometric and uniform. `AltHit` calls
    `target.GetComponents<ItemPlaceZone>()` on the transform the raycast returned, and `ItemPlaceZone.Start`
    reads its own `GetComponent<Collider>()`, so the ray must hit **the zone's own collider** -- and that collider
    is not centred on the transform. Read out of the scene bundles, **all 104 campaign zones are one prefab**:
    layer 22, a single trigger `BoxCollider`, local size `(2.2, 3.5, 2.2)`, local centre `(0, -1.25, 0)`, on a
    transform scaled `(0.9, 0.8, 0.8)`. So the box centre sits **1.0 m below `pos`** (103 of 104; one is 0.625 m)
    and the box spans `pos.y-2.4 .. pos.y+0.4`: **`pos` is only 0.4 m under the lid, with no margin**, while the
    centre has 1.4 m. That is the "extra ~1.25 m downward" the first in-game run found by hand.
    - Fixed at the mod layer as the right one: `CampaignObserver.AimPoint` reports `altars[].aim_pos` =
      `transform.TransformPoint(box.center)` (`TransformPoint`, not `Collider.bounds`, because a zone in a room
      that has not streamed in is inactive and its bounds are meaningless). Python's `campaign.altar_aim_point`
      prefers it and falls back to `pos` minus 1 m, and `GateProgress._subgoal` makes it the altar sub-goal's
      `pos`, so the observation, the aim, the approach and the reach test all use one point.
  - **F3 -- the `NeedsItem` dead-twin filter is now RELATIVE (mod).** `inactive_ancestors` is not a property of
    the zone: it is the zone's own chain plus however much of the room above it is switched off, so it **shifts
    when the room lights** (measured on 1-1: live/twin 1 and 2 on a fresh load, **0 and 1** after the respawn).
    The absolute `> 1` test therefore stopped filtering the twin exactly when the player arrived. Replaced by
    `CampaignObserver.IsDeadTwin` and `campaign.dead_twin`, applied on **both** sides and everywhere altars are
    filtered (`NeedsItem`, `ScanGates`' phase 2, `wanting_altars`): a zone is dead when another zone with the
    same item type, the same door set **and the same rounded position** reports strictly fewer inactive
    ancestors. A shared room contributes equally to both halves of a pair, so the answer cannot be shifted, and
    the minimum of each group always survives, so a lock can never be filtered away.
    - **Re-validated offline against all 21 altar levels** (scene bundles, 104 zones, 28 wired doors). Dropped
      zones **20 (old) -> 20 (new)**; **locks lost: 0** on every level; doors still demanding an item after the
      puzzle is solved **1 -> 0**, and after a room lights **11 -> 0**. Position is part of the identity because
      twins are co-located (~0.2 m, which is why the keys need `#N`); without it the rule also drops five zones
      that are not twins but separate altars of one item type driving no door (4-2 x2, 4-3, 5-3, 7-1).
  - **F4 -- the 0-2 carry trap, closed by construction (Python).** `_protect_carry` keyed on
    `campaign.wanting_altars`, which does not look at gates, while `_near_subgoal` can only release on a
    `GateProgress` sub-goal. Both now key on the sub-goal: a press is dropped only while an altar sub-goal exists
    for something actually held, so the button can never be taken away by something that cannot give it back.
    - **What 0-2's off-ladder altar should do: nothing, and that is now what happens.** Its door `-60,-6,236` is
      a **"Secret Wall"** whose only `activatedRooms` entry is `-60,-11,236`, which appears nowhere on 0-2's pit
      chain (`9 - Crushers Arena` -> `FinalRoom` -> `Pit`). It is a secret arena, off the route, which is why its
      `hops` is null -- correct, not a mod bug. The agent keeps punch, may carry the skull or throw it away, and
      the route loses nothing.
  - **F5 -- `max_steps_per_level`** (a dict keyed by scene name, empty by default) so a `levels` ladder whose
    rungs differ in size is not stuck with one cap. `max_steps` itself is unchanged.
  - **Live verification, 2026-09-17, on a NINTH game on port 47808** started by hand with `games.py`'s own
    arguments (PID recorded, killed by PID alone; **never `games.py launch`/`stop`, both call `stop_all()`**).
    47800-47807 stayed listening throughout and `runs/campaign_gates/status.json` kept updating at 160-173
    steps/s. The installed DLL is the **old** v0.7.0, so this ran F2 through the **Python fallback** -- which is
    the stronger test, since it needed no mod at all.
    - `skull_check.py --port 47808 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231 --approach 81 -1 255
      --altar-approach 0 -4 374 --budget 700` -- **check 1 pickup PASS in ONE aimed punch** and **check 3 PASS**,
      with **no `--camera-height` and no `--altar` offset**: both hacks are gone.
    - **Check 2 placement: the altar filled and the door opened in ONE aimed punch**, aimed at the collider
      centre `(0, -7.8, 381)` derived from the reported `(0, -6.8, 381)`. Re-run with an empty `--gate` it is a
      clean **PASS**, including **100 decisions of punch spam through `env.step`** -- so F4's rekeying did not
      weaken the 1-1 carry protection. With the gate assertion on it reports FAIL, and that FAIL **is F3**:
      `altar 0,-7,381 filled=True` and `gate 20,-10,381 open=True` while `needs_item='SkullRed'` -- M14 captured
      live, from the old DLL, and exactly what the installed fix will clear.
    - **F4 on real 0-2 data:** the live block reports `gate -60,-6,236 hops=None altar_only=True`,
      `wanting_altars(SkullBlue) = ['-45,-6,236']` (non-empty -- the old rule **would** have engaged) and
      `gates.target` an ordinary gate with no sub-goal. With the skull marked held, `_protect_carry` **keeps**
      the punch, and the old `wanting_altars` key **would have dropped** it. The physical pickup did not land
      because the skull reads `active=False` -- its room is still switched off -- which is the known
      teleport-into-a-dark-room case, not the carry rule.
  - **At the next pause, in order:**
    1. Stop training gracefully (Ctrl+C, wait for `Saved ... latest.zip`), then `games.py stop`.
    2. Merge `skull-fixes` into `main`, then `cd mod/UltrakillAIBridge && dotnet build -c Release` to install the
       DLL (it only fails while the games hold it).
    3. **The F3 readout:** relaunch one game and rerun the `skull_check.py` line above **with** its default
       `--gate 20,-10,381`. It must now print
       `summary: 1 pickup PASS | 2 placement PASS | 3 held skull on death PASS`; the single thing that changes is
       `gate 20,-10,381 needs_item=None` once the altar reads `filled=True`. Also confirm `altars[].aim_pos` is
       now present (`bridge_test.py --campaign`), so the 1 m fallback stops being used.
    4. Recommended weights: **`item_pickup` 10.0, `item_placed` 20.0** (30 for the puzzle, two `gate` rungs at
       15). Arithmetic: `time` 0.02 per decision at 15 decisions/s is **0.3 per game second**, so 1-1's red leg
       -- roughly 35 m to the pedestal, 133 m to the altar, 20 m back to the gate, ~190 m, 24-48 s -- costs
       **7.2-14.4**. 30 clears that with margin and is 30% of `level_complete` (100) for a leg 1-1 cannot be
       finished without, next to `checkpoint`/`arena_clear` at 10. Weight the terminal event higher so
       collect-and-abandon is not worth it alone; 15/15 is the symmetric fallback. Do **not** go above ~`gate`,
       or fetching a skull the route does not need (0-2's) becomes attractive.
    5. Not farmable at any of these: both are paid once per level load (`MilestoneTracker`, pinned by
       `test_a_placement_pays_once_even_when_the_skull_is_pulled_back_out`). Note `door_unlock` does **not** fire
       for an altar door -- `ItemPlaceZone.CheckItem` calls `Door.Open()`, not `Unlock()` -- so `gate` is the
       only other term the leg earns.
    6. Then resume, and read `part_item_pickup` / `part_item_placed` against `part_gate` in `metrics_log.csv`
       for the first 400k steps before touching them again.


## 2026-09-17 — Integration pause #2, the crash supervisor, bridge robustness

- **Integration pause #2, 2026-09-17: `skull-fixes` merged, S4 verified in game, 12 games, the campaign opened
  to 11 levels, run resumed.** The live run was stopped at **6,771,574 steps** (678 episodes; Level 0-1 at a
  **0.54** fresh-start completion rate over its last 50 fresh starts, best official time **260.28 s** against the
  human 146.58 s; Level 0-3 unlocked with 9 episodes) and resumed on the merged code as the **same run** — same
  `run_name`, same `models/campaign_gates/`, same weights. Rollback point: `4243ad4`; merge commit `faecc29`.
  - **It had already died on its own, four minutes before the pause.** One `SubprocVecEnv` worker raised
    `TimeoutError` out of `protocol.recv` waiting for a level reset, and `train.py`'s `finally` still wrote
    `latest.zip` at 6,771,574 — **more than the newest checkpoint** (`ckpt_6755630`), so `latest.zip` was again
    the right resume file. Check both every time; this kind of stop is not the hard kill that leaves it stale.
    The trainer then hung in teardown with its workers stuck and had to be killed by PID after `games.py stop`.
    Worth remembering: a bridge timeout on a reset kills the whole run, and nothing restarts it.
  - **Merge:** clean, no conflicts. Full no-game suite on `main` after the merge: 15 files, all pass; after the
    pause's own changes **15 files, 304 named tests, all pass** (`test_progress.py`'s 19 print no count, so the
    per-file lines add to 285). Mod rebuilt and installed at **70,144 bytes** (the pre-merge build was 69,632).
  - **In-game readouts** (one game, port 47800, 30 fps / frameskip 2, rendering off, Violent, all gear):
    - **0-1 regression — PASS, unchanged.** `summary: 1 PASS | 2 SKIP | 3 PASS | 4 PASS | 5 FAIL | 6 PASS`;
      `gates: 11 ordered=True truncated=False with hops=11 (1.000)`, hops 0..9, and **`altars: 0` / `items: 0`**
      as present-and-empty lists. S4 is still invisible on the level the policy has most of its steps on, which
      is the thing that mattered. (Check 5 is the known standing exit FAIL on every level.)
    - **F3, the merge's whole point — PASS, all three checks.**
      `skull_check.py --port 47800 --from-checkpoint 46 0 388 --from-checkpoint 81 -6 231 --approach 81 -1 255
      --altar-approach 0 -4 374 --budget 700` with its **default `--gate 20,-10,381`** prints
      `summary: 1 pickup PASS | 2 placement PASS | 3 held skull on death PASS`. Pickup in **one** aimed punch;
      placement in **one**, aimed at the collider centre `(0, -7.8, 381)` derived from the reported
      `(0, -6.8, 381)`, with **no `--camera-height` and no `--altar` hack** — both are gone. And the M14 line
      that was the blocker now reads **`gate 20,-10,381 needs_item=None open=True locked=False`** with the altar
      `filled=True`, where before the fix it kept `needs_item='SkullRed'` forever. It survived **100 decisions of
      punch spam through `env.step`**, so `_protect_carry`'s rekeying holds in the live game.
    - **`altars[].aim_pos` arrives.** All 7 of 1-1's altars carry it, exactly 1.0 m below `pos`, so the Python
      1 m fallback is no longer load-bearing. `bridge_test.py --campaign` now prints it (`aim=(x, y, z)`, or `-`
      on an older mod). The relative dead-twin rule is visible in the same readout: with the room lit, the live
      `0,-7,381` reads `inactive_ancestors=0` and its twin `#2` reads 1 — the exact pair the old absolute `> 1`
      test got backwards.
    - **0-2 — gates PASS, the off-route altar is inert as designed, one physical leg blocked.**
      `gates: 18 ordered=True truncated=False with hops=17 (0.944)` and check 6 PASS; `altars: 2` / `items: 2`,
      all `SkullBlue`. Running the **real** `GateProgress` against the live block from all 18 gate positions and
      the player's own: the secret-wall gate `-60,-6,236` (`hops=None altar_only needs=SkullBlue`) **never became
      the target from any of them**. And with the skull marked held, `wanting_altars(SkullBlue)` is non-empty
      (`['-45,-6,236']`, so the OLD gate-blind rule **would** have dropped the punch) while the merged
      `_protect_carry` **keeps** it — F4 confirmed on real 0-2 data, not just against `FakeLevel`.
      **The scripted carry: pickup PASS in one aimed punch** on the loose blue skull `-4,22,173`; **placement
      FAIL**, and the output names why — `altar -45,-6,236 ... active=False inactive_ancestors=1`. Its room is
      switched off, the known teleport-into-a-dark-room case, not a carry-code failure. 0-2's altar is an
      off-route secret, so nothing on the route depends on it.
  - **12 games instead of 8**, on a second throughput A/B (table and method in the instance-count gotcha):
    188.6 -> 220.5 -> **236.5 steps/s** at 8 / 10 / 12, zero bridge errors in every leg, 9.6 GB of 31.9 GB. The
    live run settles around **270 steps/s** on its early rollouts. 47808-47811's exploration archives were seeded
    from 47800-47803's.
  - **`max_steps` 9000 -> 12000, on measurement.** Of the last 100 **fresh** episodes before the pause,
    **24 ended at the 9000 cap** and 23 of those held **7-10 of 10 gates**, while the 41 that completed took a
    **median 6898 decisions**. Deep runs were losing to the clock, not to the level. **This is not a reward
    change:** a `max_steps` end is a truncation, so SB3 bootstraps `gamma * V(s_T)` and nothing pays for the
    extra time. `stuck_seconds` stays at 45, so the early-stuck cohort still ends early and the longer cap costs
    nothing there.
  - **`item_pickup` 10.0 and `item_placed` 20.0**, off 0.0 at last, on the F3 PASS above. The arithmetic is in
    the skull-fixes checklist; the two bounds now pinned by a test are `item_pickup <= gate` and
    `item_pickup + item_placed <= 2 * gate`, which is what keeps an off-route skull from being worth the detour.
    `item_placed` alone is allowed above `gate`: it is the terminal event, it has to out-weigh the pickup so that
    collect-and-abandon is not worth it alone, and it is only ever paid at an altar the agent already stands at.
  - **The curriculum opened to 11 levels** — 0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 4-1, the level
    survey's Tier A and Tier B in mission order — with **`unlock_after_fresh_episodes: 600`**, the new safety
    valve. 0-1 and 0-3 were already unlocked and **stayed** unlocked across the reorder that inserted 0-2 between
    them; the trainer printed `curriculum: 11 levels, unlocked ['Level 0-1', 'Level 0-3']` on start. See the
    curriculum bullet under Layout for why that is safe by construction and which tests pin it.
  - **`best.zip` was preserved before the pooled score changed meaning.** `keep_best --metric campaign` scores
    the pooled rate, which is a shrunk **sum** over unlocked levels; once several levels are unlocked a policy
    that is *worse at 0-1* but better overall can beat the stored record and take `best.zip`. That is arguably
    the right behaviour for a campaign run, but it loses the best 0-1 policy, so the current one was copied to
    **`models/campaign_gates/best_0-1_only_6755630.zip`** (+ `.json`, score 0.5111) first.
  - **Resumed and running** on `configs/campaign_gates_main.yaml` from `models/campaign_gates/latest.zip`
    (6,771,574 steps) with 12 games, `poll_status.py`, `keep_best.py --metric campaign` and the dashboard
    alongside. `runs/campaign_gates/metrics_log.csv` was moved to `metrics_log_pre_pause2.csv` so the new
    `fresh_episodes` and per-level columns are written — `poll_status.py` keeps an existing header and silently
    drops columns it lacks.
  - **Watch for, in order:**
    1. **Per-level fresh rates**, never the pooled one. 0-1 has to refill its 20-episode window at >= 0.5 before
       0-2 unlocks; the windows deliberately start empty after a restart, so every level reads rate None for a
       while and the sampling weights are uniform until they refill.
    2. **`part_item_pickup` / `part_item_placed` against `part_gate`** in `metrics_log.csv` for the first 400k
       steps, on the skull levels only. Both are paid once per level load, so they should stay small next to
       `gate`; a large `part_item_pickup` means skulls are being fetched that the route does not need.
    3. **The entropy tripwire is unchanged: total entropy below 6.0** — stop and raise `ent_coef` to 0.006-0.008.
       It read 7.68-7.83 through the resume and was flat.
    4. **0-1's early-stuck cohort is the main cap on its rate, and it is ~25%.** Of the last 100 fresh 0-1
       episodes, **25 reached 2 gates or fewer** and 12 of those killed nothing at all: the agent never picks the
       revolver up, or dies in the gun room. Nothing in this pause addresses it — `max_steps` helps the *deep*
       runs, not these — so 0-1's rate stays roughly capped near 0.75 until it is. That is the next thing to
       attack on 0-1, and it is a separate change from anything shipped here.
- **The run now supervises itself: `scripts/supervise.py`, live since 2026-09-17 09:26.** On the morning of
  2026-09-17 `campaign_gates` died on its own — a SubprocVecEnv worker hit `TimeoutError` on a level reset,
  `train.py`'s `finally` wrote `latest.zip`, and the trainer then wedged in teardown — and nothing restarted it.
  An LLM monitor agent was covering that with a manual playbook; this is the same playbook as a plain process, so
  it costs no tokens, never sleeps and cannot forget. Health is a `train.py` process for this run **and** a
  `status.json` younger than `--stale-seconds` (600) **and** `state: running`; a process that is up while
  `status.json` is stale is HUNG and is killed with its workers first. See the Layout entry for the mechanism and
  Commands for how it is started.
  - **The pause file is the whole safety story: `runs/campaign_gates/SUPERVISOR_PAUSE`.** Stop the supervisor or
    create that file BEFORE any planned pause, because a deliberate Ctrl+C is indistinguishable from a crash from
    the outside and it will relaunch the games and the trainer underneath you. A teardown in progress (fresh
    `status.json`, `state` already `stopped`) is left alone, but only until it goes stale.
  - Verified before going live: the full no-game suite (**16 files, 323 named tests**, 0 failures) and a
    `--dry-run` against the live run, which reported `healthy: trainer pid 23480, 7,000,054 steps, status 1s old`
    and took no action. Confirmed over the first polls that the trainer PID (23480, started 09:05:56) and
    `status.json`'s step count were untouched.
  - **The bug that first start found, and why the log write is doubled.** `cmd /c ... >> runs\<run>_supervisor.log`
    holds that file with an exclusive share mode, so the supervisor's own `open(..., "a")` raises PermissionError
    for as long as it runs: started the documented way, **every log line silently vanished**. `log()` now writes
    the line to stdout (which the shell redirects into that same file) *and* tries the direct append, and needs
    only one of them to land — exactly one copy reaches the file either way. Pinned by
    `test_logging_survives_the_shell_holding_the_log_file`. Any future helper that logs to a file it is also
    redirected into has the same trap waiting.
  - **Matching a process is not matching a string.** The supervisor rejects any command line containing
    `supervise.py`, `Win32_Process`, `Get-CimInstance`, `tasklist` or `wmic`, and its own process tree by PID,
    because a query that *lists* the trainer contains the trainer's name — the miscount an earlier report script
    made. Measured live while building this: the probe's own Bash wrapper matched on `train.py` + the run name
    and was caught only by the ancestor-PID rule, so both halves of that guard earn their place.
- **Bridge robustness, branch `robust-bridge` (2026-09-17). Not merged yet; the lead merges and the supervisor
  picks it up at the next restart.** The run had died three times in one day on the same fault and a fourth time
  on a half-booted game. Root cause, verified from the tracebacks and the two sources: **every** `TimeoutError`
  was on a `reset` request (a Unity scene load), never on a `step`, and the client's socket timeout was the same
  120 s as the mod's own `EpisodeController.resetTimeoutSeconds` — so the mod's graceful error reply could never
  arrive and there was no margin for a slow load. See the two new gotchas for the full reasoning and for what is
  inferred rather than measured (the *reason* loads run long: 12 simultaneous loads, 11 curriculum scenes,
  below-normal priority, GC).
  - **No observation, reward or action semantics change.** `step()`/`reset()` became thin wrappers around the
    unchanged bodies; on the happy path every request, every reward term and every info key is byte-identical,
    and the step timeout was deliberately left at its old 120 s. The only new reward behaviour is on a path that
    used to crash: a recovered step pays 0 and truncates.
  - What changed: two timeouts in `protocol.py` plus three `BridgeError` subclasses and a broken-connection
    latch; reconnect-and-reload recovery in `env.py` with `end_reason` `"bridge_reset"`, bounded retries, and
    `unknown scene` waited out; `games.py` readiness judged by port and process instead of the `Popen` handles,
    a `relaunch --port N` that restarts one instance, and working-set reporting in `status`; a boot health gate
    in `supervise.py` that will not start the trainer until every copy is over 600 MB for two polls.
  - Verified: the full no-game suite, **17 files, 336 named tests, 0 failures**, run from the worktree with
    `PYTHONPATH` pointed at it. Not verified in game — the live run was never touched, per the brief.

## 2026-09-17 — The collapsed ladder, the banished exit, and the route fallback

- **The collapsed ladder and the banished exit, branch `ladder-fix` (2026-09-17). Python only (the DLL is
  locked); not merged — the lead merges and the supervisor picks it up at the next restart.** Two bugs diagnosed
  live on a private game at port 47812 and fixed here. Spec:
  `docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md`.
  - **A, the collapsed ladder. `hops` is a shortest-path LOWER BOUND, not a route.** Multi-room doors
    over-connect the room graph, so on **0-3, 1-1, 1-2, 2-3, 4-3 and 8-1** — four of them in the live 11-level
    curriculum — the gate nearest the spawn already sits near `hops` 0. On 0-3 the spawn-side door `0,13,330`
    (hops 2) is "reached" from the pit **below** it (measured dy −4.1 to −6.0 m, dh 6.0-8.0 m, airborne, in all
    six probe episodes), `best_hops` locks at 2, and the target becomes `-16,73,315` — 66 m straight up through a
    ceiling — for **2,052-2,423 of every 2,501 decisions**. 175 of 177 training fresh episodes ended stuck under
    it with zero checkpoints. The walkable route is 2 → 3 → 4 → 5 → 6 → 3 → 2 → 1 → 0, non-monotone, so every
    forward leg paid nothing and observation slots 448-455 pointed at the wrong door.
  - **B, the banished exit.** `CheckPoint.Start` (`decompiled/CheckPoint.cs:132`) and `CheckPoint.ResetRoom`
    (`:681`) clone each room and move the **original** by `x + 10000f`; the mod's frozen `FinalPit` reference
    follows the original. On 0-2 `campaign.exit.pos` jumps (−199.0, −86.1, 277.0) → (9801.0, −86.1, 277.0) the
    moment checkpoint `-55,-11,277` activates — which is exactly when the ladder hands the target to the exit.
    `ResetRoom` runs again per respawn, so the offset is k × 10,000. The proper fix is mod-side at the next
    rebuild; `ExitGuard` is the Python guard, and it also **recovers** from a first report that is already
    banished, because the banish only ever adds to x.
  - **Five deviations from the brief, each forced by measurement, all in section 3 of the spec.** (1) The
    patience clock is suspended while `arena_enemies_alive > 0` or on a kill/style step — four stalls of up to
    2,543 decisions in the recorded 0-1 run ran with an arena alive on **96-100%** of their steps, in front of
    the *correct* gate, where no approach is possible however well the agent plays, and on 0-3 the arena count
    is 0 on **100%** of the locked steps — but the suspension is **bounded** at `2 * patience_steps`.
    (2) A **ladder** pick is parked only when some unreached, unparked gate is strictly nearer, which takes the
    recorded 0-1 run from 0-13 parks per episode to 0-1 and leaves **6 of its 7 episodes byte-identical**; a
    **fallback** pick is parked unconditionally, because the same filter chose it as the nearest and the test
    would make it permanent. (3) The fallback target is sticky within 10 m, because pure "nearest" flaps between
    two near-equidistant doors **23-59 times per 0-3 episode** (10 m: 9-20) with no `gate` instalment lost.
    (4) The clock runs on its own baseline, not on `gate_approach`, and `mark_paid` restarts it. (5) A fetch or
    carry leg, and any gate reporting `needs_item`, is never parked at all.
  - **An adversarial review found eight defects in the first implementation; all eight were reproduced against
    the code before anything changed, and section 7 of the spec is the appendix.** Six were outright bugs: a
    **fallback target could never be parked** (the same filter picks it and tests it, so the mechanism protected
    one hand-over and then switched itself off — 4,000 decisions aimed through a ceiling with
    `targets_parked` reading 1); the instalment was **per door rather than per rung**, so an eight-door tour of
    one rung collected 6 instalments against a ladder depth of 2 (780 available on 8-1's 52 gates, against
    `level_complete` 100); the clock **read `gate_approach`**, which R1 makes episode scoped, so a post-death
    re-walk parked the right door at decision 299 of a flawless 389 m approach; the arena suspension was
    **unbounded and level-wide**, so an un-cleared wave anywhere froze the clock for the level load (20,000
    decisions, zero parks, telemetry reading `targets_parked = 0` — indistinguishable from "correctly inert");
    parking a skull gate **silently disabled the punch carry-protection**, so the next punch threw the skull and
    lost the level load; and `_pay_fallback`'s unguarded `int(gate["hops"])` **raised out of `update`** on a
    `hops: null` door, past `env.step`, into all twelve games. The other two were half right: the un-park bar
    was too easily cleared (a far-off re-park re-read the baseline), but the reviewer's proposed cure — track the
    baseline down every step — is beaten by one decision's travel and made the park permanent, failing A3; and
    0-1's inertness *was* overstated, though the claim that relaxing the nearness test adds 45 parks there is
    measurably wrong (**all 46 of 0-1's window expiries are on ladder targets, none on a fallback**, so 0-1 has
    exactly one park either way).
  - **The reach test was left alone, as A5 directs.** Adding a 3-D bound to the cylinder does reject all six of
    0-3's corner reaches (8.49-9.27 m), but replayed over the 32,022 recorded 0-1 decisions it is **not inert**
    (ep 5 never reaches `146,31,640` and reaches `40,11,624` 934 decisions late), so A5's precondition fails. It
    would not have fixed bug A anyway: on 0-3 it only delays the bad reach by 66-1,069 decisions.
  - **Result on the recorded data, after the review fixes.** 0-3: the lock on the unreachable door breaks after
    380-841 decisions in both probe episodes and the target moves to `0,13,362` / `0,53,330`, gates on the
    walkable route; the share of decisions spent on it falls 82-94% → **40.2% and 33.6%**, and every time it
    comes back the agent has got strictly closer than ever before (parks at 40.87, 35.98, 26.13 m against bars
    of 2, 4, 8 m — a monotone 24 m closing; episode 1 parks once and it never returns). A walk of the real route
    pays **7** `gate` instalments against the monotone rule's 3, with `gate_approach` 469.5 m against 316.5.
    0-1: 6 of 7 recorded episodes byte-identical; the seventh's single park hands over to `66,21,640` — the gate
    the ladder was going to pick next — **78 decisions early**, for `gate_approach` +10.363 m (+1.55 reward) and
    **zero** extra instalments, with the same target order, reached set and `best_hops`.
  - New per-episode info, both in `CAMPAIGN_INFO_KEYS` and so in `status.json`, `episodes.jsonl`, the dashboard's
    campaign panel and `poll_status.py`: **`targets_parked`** (parks this episode) and **`exit_banished`**.
    `poll_status.py` keeps an existing header, so **move the live run's `metrics_log.csv` aside** to get the two
    new columns.
  - **What to watch once this is merged**, in this order:
    1. `targets_parked` — expect **~0 on 0-1** (one park in seven recorded episodes) and **> 0 on 0-3, 1-1, 1-2,
       2-3, 4-3 and 8-1**. A collapsed level reading 0 means the mechanism is not firing there, not that the
       level is fine: that is exactly what the unbounded arena suspension used to look like, and it is also the
       **known residual** — a ladder pick is only parked when some unreached gate is strictly nearer, so a level
       whose unreachable rung-below door is *also* the nearest unreached gate still wedges. 0-3 is not that
       shape; 1-1, 1-2, 2-3, 4-3 and 8-1 have no probe data either way. Do not answer it by dropping that test:
       it is the only thing holding 0-1's other 45 window expiries.
    2. `exit_banished` — expect **1 on 0-2** once checkpoint `-55,-11,277` activates, and **0 everywhere else**.
       A 1 on another level means a second banish site nobody has looked at.
    3. **0-3 and 0-2 fresh** `gates_reached` and `checkpoints_level`. Judge these on fresh starts only, on a
       window of >= 50 episodes; a respawn episode inherits both from its level load.
    4. `part_gate` against `part_gate_approach` in `metrics_log.csv`. `gate` is bounded by the ladder depth per
       level load under both rules, so `part_gate` above ~10 × 15 on a prelude level means the bound has a hole.
  - **Falsifier.** Within ~300 fresh 0-3 episodes after activation, mean `gates_reached` should rise **above 2**
    and mean `checkpoints_level` **above 0**. If it does not, target patience is not enough and the ladder needs
    the route-path redesign (a per-level ordering learned from where the agent has actually been, rather than
    `hops` as a proxy for it) — not another patience knob.
  - No observation width, no action-space change, no reward-weight change: 479 inputs either way, and every
    existing checkpoint and `env_config.yaml` loads unchanged (`from_dict` drops unknown keys, the four new
    settings default) — verified against the live run's own `models/campaign_gates/env_config.yaml`, 51 keys,
    none of them new. `gate_target_patience_s: 0` restores the pre-patience tracker exactly, which is how the
    0-1 inertness proof is written and what to set if the mechanism ever misbehaves live. 0-1's inertness is
    **measured, not structural** — the tests in section 6 of the spec are what measure it.
  - Verified: the full no-game suite, **18 files, 408 named tests, 0 failures**, run from the worktree with
    `PYTHONPATH` pointed at it, plus each of the eight review findings re-run against the fixed code. Not
    verified in game — the live run was never touched, per the brief.

- **Branch `route-fallback` — the offline room trunk, stages S1 and S2, built 2026-09-17 and READY TO ACTIVATE.**
  Spec: `docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md` (revision 2). Python and
  data only: **no mod change, no protocol change, no reward weight, no observation width change, no action-space
  change, and no checkpoint invalidated.** 479 inputs either way. The live run was never touched.
  - **What it buys.** 15 of the 33 shipped levels have no usable gate ladder at all, so `GateProgress._gates()`
    returns `[]` and they have never had a route signal of any kind. 12 of them now ship an offline **room
    trunk** — the numbered rooms every playthrough must pass, in order, shaped exactly like a gate array.
    Campaign-wide: **18 gates + 12 rooms + 3 nothing**, against today's 18 + 15 nothing. Of the 12, **8 get a
    trunk from the first room to the exit and 4 stop at a skull/altar lock** (1-4, 4-4, 7-1, 7-2) until stage S3
    stamps `needs_item` live.
  - **Why only the trunk.** Revision 1 shipped a strict total order over every room, and two independent reviews
    found that `GateProgress` carries one `best_hops`: `_note_reached` collapses it to the minimum over every
    rung each step, so **on a branching level the branch the agent enters second permanently loses its signal**.
    Re-run with the real tracker, entering one of 1-4's five side rooms first pays **four instalments in a
    single step** and then points the observation at the boss door with three required rooms unvisited — the
    same total instalments either way, so it is signal death, not extra reward. Invariant **T** is the fix: a
    parallel set is collapsed onto the rung it hangs off. Coarser, never wrong, and it costs 1-3 and 6-2.
  - **Stages. S1 and S2 are built; S3 and S4 are out of scope and separately mergeable.** S3 is the live
    `needs_item` stamp from `campaign.altars` that takes the four locked levels through to their exit; S4 is the
    mod's boss block (`ScanBosses`, mod 0.7.1), which needs the game closed for a DLL rebuild and waits for a
    natural pause regardless.
  - **Evidence, all of it offline — and that is stated rather than implied.** The live configs name no level
    with a route file, so **A0 and the `FakeLevel` cases are the whole of the pre-merge evidence for layer 2,
    and A3 is the whole of the evidence that layer 1 is untouched.**
    - The full no-game suite: **21 files, 477 named tests, 0 failures**, run from the worktree with `PYTHONPATH`
      pointed at it.
    - **A0** (`tests/test_route_replay.py`): the real `GateProgress` replayed over each shipped trunk in every
      visit order the level allows and from every possible starting rung — the target never advances past an
      unvisited rung, no step pays more than one instalment. Non-vacuous by construction: the same replay over a
      synthetic star reproduces the four-at-once above.
    - **Layer 1 is untouched, twice over.** With a route document in hand, `GateProgress` reproduces
      `test_ladder_replay.py`'s golden file — 32,022 recorded 0-1 decisions and both 0-3 probes, produced by the
      implementation from *before* any of this existed — and `route_reads == 0` proves the trunk was not merely
      equal but never read. `_gates()`'s `return gates` success path is byte for byte what it was.
    - **The trunks walk** (`tests/test_route_walk.py`, this stage's own check): all 12 files loaded through the
      packaged path and walked end to end with a continuous synthetic trajectory. Every rung becomes the target
      in order, each pays exactly one instalment, the exit takes over after the hops-0 rung and never hands back.
      Backwards pays 0, a second lap pays 0, the live patience setting parks nothing, and `route=None` pays 0.
    - The measured `gate_approach` reproduces the spec's section 6 table from the tracker rather than from the
      polyline: 0-5 **68.8** (spec 72), 1-4 60.0 (62), 2-4 173.2 (178), 4-2 153.4 (157), 4-4 382.3 (392), 5-2
      160.9 (167), 7-1 241.3 (252), 7-2 273.1 (291), 7-3 203.8 (220), 7-4 200.1 (204), 8-3 **776.6** (785), 8-4
      64.8 (70). Consistently a few percent under, because `best_dist` is seeded where the player enters a rung's
      cylinder rather than at its centre.
  - **Known deviation from the spec's section 10, and it is the pipeline working.** **8-3 ships 16 rungs, not
    13**, so the campaign total is **95 rungs, not 92**. Section 10's 13 was measured on revision 1's file, which
    the budget cap had already cut before invariant T ran; revision 2 reversed the order so R2 is judged on what
    ships, and after T the cap binds on nothing. Median gap 258 -> 167 m, tour 1.247 -> 1.239, R2 still passes.
    It does raise 8-3's ladder: `gate` 180 -> **225**, so `gate` + `gate_approach` is **1,002 against
    `level_complete` 100 — 10.0x**, above the 9.7x the spec quotes. See the tripwire below.
  - **One other deviation, flagged for a ruling.** Invariant T's parallel-set predicate now additionally
    requires its members to be **distinct places** (spread >= 16 m). Read literally, the stated order
    I6 -> T -> R4 -> I2 would drop 5-2's `1A - Opening` and `1B - Second Rock`, which sit at *exactly* the same
    point, and 5-2 would lose the rung at its own spawn. A group inside one reach cylinder is not a choice the
    agent can take in either order — arriving marks every member at once — so T's justification does not apply
    and I2 is what should collapse it. **Exactly one group campaign-wide is affected**; every other same-ordinal
    group is 41 m or wider. With the clause 5-2 reproduces section 10 exactly.

  ### Activation checklist — how to put this in front of the live run

  Eight steps, in order. Steps 1-2 need no pause; steps 3-7 are the switch, and the stop is graceful, not a
  kill. **The config lives on the SUPERVISOR's command line**, so switching configs means restarting the
  supervisor, not just the trainer — the pause file alone does not do it.

  1. **Merge `route-fallback` into `main`** — Python and data only, no mod build, so the games stay up:
     `git -C F:\Github\ULTRAKILL-AI merge route-fallback`. Then from `python/` run the whole no-game suite
     against the merged main (21 files, 499 named tests, 0 failures) and push. The supervisor restarts the
     trainer from main's code, so **main must be left in a state that runs**.
  2. **Move the metrics log aside** for the two new columns (`ladder_collapsed`, `ppo_ent_coef_live`);
     `poll_status.py` keeps an existing header and would silently drop them:
     `Move-Item runs\campaign_gates\metrics_log.csv runs\campaign_gates\metrics_log.pre-route.csv`.
  3. **Pause the supervisor**: `New-Item runs\campaign_gates\SUPERVISOR_PAUSE`.
  4. **Stop the supervisor** — Ctrl+C its own console and wait for it to exit.
  5. **Stop the trainer** — Ctrl+C and wait for `Saved models\campaign_gates\latest.zip`.
  6. **Lift the pause**: `Remove-Item runs\campaign_gates\SUPERVISOR_PAUSE`.
  7. **Restart the supervisor on the new config**, from `python/`:
     `python scripts/supervise.py --run campaign_gates --config configs/campaign_gates_full.yaml --count 12 --monitor 1`.
     It finds no trainer, treats it as DEAD, relaunches the twelve games, waits out its boot health gate and
     starts `train.py --config configs/campaign_gates_full.yaml --resume <the checkpoint with the most
     timesteps>`. Same run name, same weights, same `best.zip`, same exploration archives.
  8. **Read the first update**: `ladder_collapsed` 0 on 0-1 and 1 on 0-3, `targets_parked` 0 on 0-1, and
     `ent_coef_live` 0.004 while total entropy is above 6. **Rollback is steps 3-7 with
     `--config configs/campaign_gates_main.yaml`.**

  The switch is also safe because of the unlock ladder rather than because of the code: at ~11.2M steps only
  0-1, 0-2 and 0-3 are unlocked and **0-4 is still locked**, so 0-5 — the first trunk-routed level — cannot
  open until 0-4 does, and nothing on layer 2 runs on the day of the switch. 0-3 and 4-3 do ship a file now,
  but `prefer_route_when_collapsed` is false, so neither of them reads it.


  ### What to watch, in this order

  0. **The conditional-patience readouts, first, because they are the ones that changed live behaviour.**
     `ladder_collapsed` per level (`status.json`'s `campaign.levels[...]`, the dashboard's `col` column and
     `poll_status.py`'s column) must read **0 on 0-1, 0-2, 0-4, 2-1, 2-2, 3-1 and 4-1** and **1 on 0-3** —
     it is a property of the level, so anything between 0 and 1 on one level means two workers disagree.
     **`targets_parked` on 0-1 must read 0**; anything above that is the regression coming back. Then 0-1's
     fresh completion rate recovering toward **0.55+** (window >= 50, fresh starts only) while 0-3 keeps its
     post-fix `gates_reached` near **4.5**. And `ent_coef_live` against `|entropy_loss|`: the coefficient
     should sit at 0.004 while entropy is above 6 and only climb if it falls under 5.
  1. **`route_source` per level** — the dashboard's **`route layer`** row, `poll_status.py`'s `route_source`
     column, `status.json`'s `mean_100.route_source` and the per-episode string in `episodes.jsonl`
     (`route_source_name`). Expect **1 ("gates")** for weeks. A window reading **"mixed"** is the first
     evidence a rooms level is running; a **2** on any of the 18 gates levels is a bug and the merge should be
     rolled back, because it means `_gates()`'s success path was reached and rejected.
  2. **`targets_parked`** — expect **0 on every rooms level**, and understand that this proves nothing on its
     own. A monotone trunk's rung-below is usually also its nearest unreached rung, so `_nearer_unreached` keeps
     the ladder's pick and the patience rule is structurally inert there. A walk that reaches every rung parks
     nothing; so does a walk that reaches none. **The detector for a bad trunk is `route_source` 2 with fresh
     `gates_reached` stuck AND `targets_parked` 0** — not either half alone.
  3. **Per-level `gates_reached` on the first rooms level to unlock, which is 0-5.** Read
     `status.json`'s `campaign.levels["Level 0-5"].gates_reached` (a 100-episode mean, per level) and the
     dashboard's `levels` rows — **never the pooled `fresh_completion_rate`**, which is a shrunk sum over the
     unlocked levels and can exceed 1.0. 0-5's trunk is 5 rungs, so `gates_reached` should climb toward 5 and
     `fresh_completion_rate` off 0. **Judge on fresh starts only, on a window of >= 50**: a respawn episode
     inherits `gates_reached` from its level load. A 0-5 that sits at `gates_reached` 1 with `targets_parked` 0
     is the "bad rung" case, and the cure is 0-5's own file (`"rungs": []`), not a weight.
  4. **The `gate_approach` dominance tripwire (spec section 12.6).** `gate_approach` is a **global** weight
     shared with the levels the live run is already training and there is deliberately **no per-level lever** —
     adding one would be a reward change, which is the property this whole spec buys. **If `part_gate_approach`
     exceeds 6x `part_level_complete` over a 50-episode window on a fallback level whose completion rate is
     still 0, regenerate THAT LEVEL'S file with a lower `MAX_RUNGS`** before touching any weight: the data lever
     is reversible and level-local, the weight lever is neither. **8-3 is expected to trip it** — its perfect
     walk is 777 of approach against `level_complete` 100, i.e. 7.8x before a single completion exists — so plan
     for that one rather than treating it as a surprise. 4-4 (4.9x) and 7-2 (4.8x) are the next two; every other
     trunk is under 4x.
  5. **8-4, 2-4 and 7-1 are the three weak trunks to check first** if a fallback run wanders: 8-4 is **0 of 4**
     legs checkpoint-witnessed, 2-4 is 1 of 4 with a **tram** as its hops-1 rung, and 7-1 is 5 of 10 with an
     order that was fixed offline and has never been read live. 4-4, 7-1 and 2-4 also carry rungs that are
     **movers** (two elevators, two trams): the reach cylinder sits where the mover started, not where it is.

  - **The S1 engineer's finding on the six collapsed-ladder gates levels has been RULED ON** (lead,
    2026-09-17; spec section 13.4). The room trunk reproduces a sensible walkable order on **0-3, 4-3 and
    1-1**, 8-1 loses its exit room to invariant T, and 1-2 and 2-3 are refused by the tour guard. Files now
    ship for **0-3 and 4-3 only** (`build_routes.COLLAPSED_SHIP`) — not 1-1, whose skull locks wait for S3 —
    and they are read only when `prefer_route_when_collapsed` is **true**, which it is not by default, because
    0-3 is currently progressing on gates plus patience. Flipping it is a per-run decision on evidence; a level
    whose ladder is healthy cannot reach its file whatever the flag says.
  - **What is NOT measured, and cannot be offline: whether a leg is walkable in that direction.** The fine
    geodesic was measured dead and a straight knee-height segment test is `partial` on 0-1 itself. The trunk is
    ordered, standable (100 of 100 probed places pass the voxel guard R4 over 95 rungs) and corroborated (58 of
    95 legs have an authored checkpoint within 40 m); it is not proved traversable. That is why the recovery
    path is deliberately trivial and why `route_source` exists.

- **Target patience is now LEVEL-CONDITIONAL, and two lead rulings landed with it (branch `route-fallback`,
  2026-09-17). Python and data only; not merged — the lead merges and the supervisor picks it up at its next
  restart.** The ladder-patience fix went live at **9.59M steps** and the live run measured both halves of it
  inside 1.5M steps:
  - **0-3 (collapsed ladder): the fix works.** Fresh `gates_reached` **1.0 -> 4.5**, checkpoints per load
    **0 -> 1.1**.
  - **0-2: the exit guard works.** Fresh completion **0.03 -> 0.41-0.62**, best **163.2 s**.
  - **0-1 (monotone, CORRECT ladder): it REGRESSED.** Fresh completion **0.55 (n=43, 7.5M-9.58M) -> 0.30
    (n=70 after)**. Before: `targets_parked` 0.00 per fresh episode, gates histogram `{1:3, 2:1, 5:1, 6:1,
    7:8, 8:6, 10:23}`, one stuck episode near (40, 480). After: `targets_parked` mean **1.17**, **34%** of
    fresh episodes park at least once, histogram `{0:2, 1:7, 2:19, 4:4, 5:2, 7:5, 8:10, 10:21}` — the mass
    moved from 10 to 2 — and **14 of 39** stuck episodes end at (40, ~480) with exactly 2 gates.
  - **Cause.** The offline replay that "proved" 0-1 inert used recordings of a much older policy
    (`campaign_ppo_ground/latest.zip`). Today's policy holds a target through much longer arena fights, the
    **bounded** arena suspension (2 x patience, the correction the adversarial review forced) expires, the
    CORRECT target is parked and the fallback sends the agent somewhere wrong. On a level whose ladder is
    right there is nothing better to switch to, so a park can only mislead.
  - **The fix is a detector, not a knob.** `campaign.detect_collapsed_ladder(rungs, spawn)` is a pure function
    of the `campaign.gates` block and the player's first position on a fresh load: **collapsed when the gate
    nearest the spawn carries `hops <= max_hops / 2`**. Measured over all 18 shipped levels with a usable gate
    ladder (`scratchpad/collapse_detect_scan.py`, from the offline level survey, with the two live spawns for
    0-1 and 0-3 and the level's first authored room elsewhere):

    | verdict | levels (nearest-gate hops / max hops) |
    |---|---|
    | COLLAPSED | 0-3 2/6, 1-1 1/5, 1-2 2/4, 2-3 0/2, 4-3 1/2, 8-1 0/4 — ratios 0.00-0.50 |
    | healthy | 0-1 9/9, 0-2 7/7, 0-4 5/5, 2-1 3/3, 2-2 2/2, 3-1 7/7, 3-2 3/3, 4-1 8/8, 5-1 2/2, 6-1 2/2, 5-3 12/13, 8-2 5/8 — ratios 0.62-1.00 |

    The worst collapsed level sits exactly at 0.50 and the nearest healthy one at 0.625 (**8-2**, whose spawn
    is the least certain of the eighteen: at each of its three plausible spawn points it reads 5, 5 or 7 of 8,
    so it stays healthy whichever is right). **Nothing else separated them**: the absolute deficit
    `max - near` puts 8-2 (3) above three collapsed levels (4-3 1, 1-2 2, 2-3 2) and has no working threshold.
    The verdict is taken **once per level load**, before anything reads it; a checkpoint respawn keeps it
    (a respawn starts the player half-way through a level, where the nearest gate says nothing); undecided
    reads as OFF. Reported as `ladder_collapsed` per episode and per level.
  - **Config: `gate_patience_mode`** — `collapsed` (default) | `always` (the mechanism as first shipped) |
    `off`. `gate_target_patience_s: 0.0` is still the global off switch.
  - **Confirmed against the live run, as far as `episodes.jsonl` allows** (it has no per-step positions, so
    this is a summary replay, not a step replay — `scratchpad/replay_live_0_1.py`): the detector reads
    **collapsed = False** on 0-1's real gate array at its real spawn (nearest gate `40,1,408`, hops 9 of 9),
    so patience is off on **every** 0-1 level load and all **125 parks** the run took on 0-1 after the fix
    (24 of 70 fresh episodes affected) could not have happened. The step-level proof is `test_a6_0b`: under
    the shipped settings all seven recorded 0-1 episodes reproduce `ladder_golden.json` exactly with zero
    parks, where the unconditional mechanism moved episode 5 by +10.363 m of `gate_approach`.
  - **Ruling (a): invariant T's distinct-places clause is ACCEPTED exactly as built** (spec section 13.4). A
    same-ordinal suffixed group is dropped only when its members are >= 16 m apart, the distance `_is_reached`
    cannot tell apart; a group inside one reach cylinder is one PLACE, not a choice, and I2 collapses it
    instead. Exactly one group campaign-wide is affected — 5-2's `1A - Opening` / `1B - Second Rock`, both at
    (0, -10, 300) — and without the clause 5-2 loses the rung at its own spawn.
  - **Ruling (b): 0-3 and 4-3 ship a trunk, and nothing prefers it yet.** `build_routes.COLLAPSED_SHIP` emits
    a file for those two beside their collapsed gate ladder (0-3 11 rungs, tour 0.944, 9/11 legs witnessed;
    4-3 8 rungs, tour 1.000, 6/8) — **not** 1-1 (skull locks wait for S3), **not** 1-2 / 2-3 / 8-1 (guards).
    `prefer_route_when_collapsed` selects it, and it **defaults to false** because 0-3 is progressing on gates
    plus patience today; the lead flips it per run on evidence. **14 route files ship, not 12.** A level whose
    ladder is HEALTHY can never read its file whatever the flag says — the verdict gates it, not the file.
  - **The one hazard that became real, and is handled.** 0-3 and 4-3 are now the only levels with BOTH a
    ladder and a file, so a frame the mod sends with no `campaign` block would have taken `_gates()`'s first
    arm, handed out the trunk unvalidated, flipped `_hops_source` and cleared a ladder that was fine — losing
    that load's `gate` income on the level the patience fix is rescuing. The layer is now **latched per load**:
    once a load's ladder has come from the gates, a block-less frame returns `[]`.
  - **Adaptive entropy floor (`train.ent_floor`, `train.ent_coef_max`).** The live run went `|entropy_loss|`
    **7.88 at 9.58M steps -> 5.57 at 11.0M** (heads yaw 1.06 / pitch 0.97 / look_mode 0.43) on a FIXED
    `ent_coef` of 0.004, and this project's Cyber Grind run collapsed exactly this way **twice** — peak near
    3.5 nats, then worse while it kept sharpening. A fixed coefficient cannot defend a floor: the bonus is a
    constant pull against an advantage signal that grows as the policy gets confident. `EntropyFloorCallback`
    reads the update that just finished and, per rollout: **entropy < floor -> `ent_coef *= 1.10`, capped at
    `ent_coef_max` (0.02); entropy > floor + 1.0 -> `*= 0.95`, never below the config's `ent_coef`; inside the
    band, unchanged.** Multiplicative and slow (17 updates from 0.004 to the cap, 34 back, against ~2000
    updates in a 20M-step run) and a floor, never a ceiling. `ent_floor: 5.0` in
    `configs/campaign_gates_full.yaml`, just under the live value, so it is inert while nothing is wrong. It
    **does not survive a resume** by design: every restart starts at the base and re-adapts over a few updates.
    Live value logged as `train/ent_coef_live` -> `status.json`'s `ppo` block, the dashboard's PPO panel and
    `poll_status.py`'s `ppo_ent_coef_live`.
  - **`configs/campaign_gates_full.yaml` now differs from `campaign_gates_main.yaml` in exactly five things**,
    each pinned by `test_the_full_config_is_the_main_config_with_more_levels`: `levels` 11 -> 30,
    `gate_patience_mode: collapsed`, `prefer_route_when_collapsed: false` (true since 2026-09-17 21:58),
    `ent_floor: 5.0` + `ent_coef_max: 0.02`, and `curriculum_weighting: progress` (2026-09-18). `num_envs`
    stays 12, `ent_coef` stays 0.004 and no reward weight moves.
  - **Evidence: the whole no-game suite, 21 files, 499 named tests, 0 failures.** New: the detector's
    separation table as a test, the three modes, the verdict's per-load scope and late-ladder case, 0-1 inert
    and 0-3 patient under the shipped default (unit, env and recorded-replay), both settings of
    `prefer_route_when_collapsed` including "a healthy ladder never reads its file", the block-less-frame
    latch, the two new route files against `EXPECTED_LADDERS`, and four entropy-floor tests against a fake
    model (raise, decay, cap, base, band, off).
  - **What to watch after activation**, in order: `ladder_collapsed` per level (0 on 0-1, 1 on 0-3; a value
    between them means workers disagree); **`targets_parked` on 0-1 must read 0**; 0-1's fresh completion
    recovering toward **0.55+** on a window of >= 50 fresh starts while 0-3 holds `gates_reached` near 4.5;
    and `ent_coef_live` against `|entropy_loss|` — 0.004 while entropy is above 6, climbing only under 5.

## 2026-09-17 — Full-campaign config live, the weights rollback, and `freeze-fix`

- **Full-campaign config live (2026-09-17 16:51, resumed at 11,626,642 steps).** `route-fallback` merged (`8d2c4c5`): 14
  room-trunk route files under `python/ultrakill_ai/routes/` (12 for levels gates cannot route, plus 0-3 and 4-3 shipped
  but unused while `prefer_route_when_collapsed` is false), level-conditional target patience (`gate_patience_mode:
  collapsed`: nearest-gate hops <= max_hops / 2 at level load; COLLAPSED on 0-3, 1-1, 1-2, 2-3, 4-3, 8-1, healthy on the
  other twelve gates levels), and the adaptive entropy floor (`ent_floor` 5.0, base `ent_coef` 0.004, x1.10 per rollout
  below the floor up to 0.02, x0.95 back toward the base above floor + 1). The run is now
  `configs/campaign_gates_full.yaml`: 30 levels in mission order (everything shipped except the unrouted 1-3, 5-4, 6-2),
  12 games, supervised by `supervise.py --config configs/campaign_gates_full.yaml`.
  - **Why conditional patience:** the unconditional version (live 9.59M-11.63M) fixed 0-3 (fresh gates 1.0 -> 4.5,
    checkpoints 0 -> 1.1) but REGRESSED 0-1 from 0.55 to 0.30 fresh completion: 34% of fresh 0-1 episodes parked a correct
    target during long arena fights and 12 of 32 stuck episodes ended at (40,0,480) with exactly 2 gates. The offline
    replay had called 0-1 inert because its recordings came from a much older policy. Lesson: a replay proves a
    mechanism only for the policy that was recorded.
  - State at the switch: 0-1 best 3:03.628 (rank A), 0-2 best 2:43.211 with ~0.6 fresh completion since the exit guard,
    0-3 no completion yet, 233+ completions all-time, ~231 steps/s.
  - Watch after activation: `targets_parked` on 0-1 must read 0 and its fresh rate should recover toward 0.55+;
    `ladder_collapsed` 0 on 0-1 / 1 on 0-3; `ent_coef_live` and total entropy holding >= ~5; 0-3 gates_reached still
    rising (if it plateaus without completions, set `prefer_route_when_collapsed: true` to move 0-3 onto its room trunk);
    `route_source` 2 on the first rooms-routed level to unlock (0-5).
  - **Gotcha, planned pauses since the bridge-recovery merge:** `games.py stop` no longer ends the trainer (each env now
    retries its lost game for minutes instead of raising), so `latest.zip` is NOT written. For a planned pause: create
    `runs/<run>/SUPERVISOR_PAUSE`, stop the supervisor, `games.py stop`, kill the `train.py` processes, and resume from
    the newest `ckpt_*_steps.zip` (at most 50k steps old; the supervisor picks the file with the most steps by itself).
    Also: PowerShell's safety check rejects a long combined command that mixes `Remove-Item` with a `cmd /c` argument
    list; run those as separate commands.
- **Weights rolled back to `ckpt_9553510` (2026-09-17 19:30), code kept.** The 1.4M steps of unconditional patience
  (9.59M-11.63M) taught 0-1 a bad habit that 1.6M steps of corrected targeting did not undo: fresh 0-1 since 16:51
  n=39, completion 0.13 (0.55 before 9.58M), 19 of 39 dying at gates 1-2 near (40,0,400-480) against 4 of 43 before,
  kills 47 -> 32; 0-2 0.22 and 0-3 0.05 over the same window. So training resumed from the last checkpoint before the
  misdirection with every fix in place (conditional patience, exit guard, route files, entropy floor). The 73
  post-9.55M checkpoints and the 12.44M `latest.zip` are in `models/campaign_gates/rolled_back_2026-09-17/`
  (gitignored); `best.zip` (10.93M, pooled score 0.70) is from inside the bad window and should NOT be used to resume.
  Falsifier for the rollback: within ~1M steps 0-1's fresh rate should sit near 0.5 again and 0-3's gates_reached should
  climb past 2 under patience; if 0-1 degrades again WITHOUT parking, the cause is multi-level interference or
  entropy, not the misdirection.
- **User priority restated 2026-09-17: speed, not kills.** The objective is finishing each level, and the game, as fast
  as possible. Kill / damage rewards stay minimal (arena-gated doors are the only reason they exist); once a level's
  fresh completion rate is reliable, the next tuning is a completion bonus scaled by official time, then movement tech.
- **`freeze-fix`: bounded bridge recovery, attributable stalls (branch, 2026-09-17, NOT YET MERGED).** Answers the
  17:52 freeze — read its gotcha entry above for the evidence and the root cause. Ten changes, no observation,
  reward or action semantics touched, and no mod rebuild:
  1. `protocol.py`: every blocking call bounded in every state (`close_timeout` 5 s, `handshake_timeout` 20 s,
     `connect_timeout` 10 s, reset **600 → 180** s), each request arming its own bound for the send as well as
     the read (a socket timeout is sticky), and `_LineReader` bounding wall clock instead of each `recv`.
  2. `env.py`: one **total** `bridge_recovery_budget_s` (540 s) across every attempt, backoff and layer,
     shared with `reset()`'s outer retry so it cannot be paid twice.
  3. `env.py`: the last rung **relaunches this env's own game** (`games.relaunch_one`, never another port) and
     waits out a working-set boot gate. Off by default; `train.py` alone turns it on, so nothing else that
     builds an `EnvConfig` can restart a game.
  4. `env.py`: `mod_command_timeout_s` 900 is sent on connect, so the mod's 300 s silent-client drop no longer
     costs the other eleven games their connections while one worker recovers.
  5. `envlog.py` + `train.py`: `runs/<run>/env_<port>.log`, the per-worker attribution log.
  6. `train.py`: `close_vec_env`, a bounded teardown that terminates a wedged worker, so a crash reaches the
     supervisor as an exit code in one poll rather than a hang in ten.
  7. `supervise.py`: health judged on **`timesteps` moving** as well as on `status.json`'s mtime (defence in
     depth, not the thing that would have caught 17:52 — see the supervisor's Layout entry).
  8. `supervise.py`: `--stale-seconds` 600 → **900**, derived as step 120 + budget 540 + poll 60 = **720** plus
     margin, and pinned by a test that RUNS the ladder on a fake clock rather than re-adding the knobs.
  9. `supervise.py`: a per-port sick report (pid, working set, ESTABLISHED connections, CPU cores) plus every
     `env_*.log` tail, written **before** the kill; and the spawn no longer uses `DETACHED_PROCESS`, which had
     been discarding the trainer's entire stdout.
  10. `games.py`: a laggard port is repaired with `relaunch_one` (and a wedged portless instance killed) instead
      of failing the whole launch and costing another full stop-and-launch round.
  - **Second pass (2026-09-17, same branch): six more findings, all six measured, all six fixed.** The first
    ten got the individual bounds right and left the *total* advisory, which is why four of the six are the
    same bug in different places. Full detail and the measurements are in the freeze gotcha; in brief:
    11. `env.py`: the deadline is **authoritative**, not advisory — `_begin_recovery` returns
        `(deadline, opened_here)`, only the opening frame closes it (the reset tail was opening a second full
        budget), and `_clamp` cuts `client.reset(timeout=…)` and `connect(retry_seconds=…)` to what is left.
        Measured 555.6 s → **540.0 s exactly**, one budget instead of two.
    12. `env.py`: `bridge_relaunch_reserve_s` (240 s) that rung 0 may not spend, because the relaunch rung was
        **unreachable** whenever the bridge failed slowly — i.e. in the shape it exists for.
    13. `env.py` + `eval.py`: `EnvConfig.RUN_ONLY_FIELDS` keeps `bridge_relaunch` and `env_log_dir` out of
        `env_config.yaml`, so an eval can no longer `taskkill` a **training** game or write into a live run's
        attribution log.
    14. `games.py`: every `subprocess.run` bounded at 30 s and the `netstat`/`tasklist` snapshots cached with a
        2 s TTL — these are called from inside the recovery that promises it cannot hang.
    15. `protocol.py` + `env.py`: a connect failure is `BridgeClosed` (RECOVERABLE), `connect_retry_s` back to
        180 s, and `reset()`'s tail reconnects through `_reconnect_within`, which retries inside the budget and
        then relaunches. A cold port used to kill the worker in 60 s flat — the 18:08 pathology.
    16. `env.py`: at most 2 workers relaunch at once (`O_EXCL` permits in `env_log_dir`, stale-age takeover)
        with a per-port stagger; and `relaunch_one` reaps a wedged portless instance before replacing it.
  - **Verified (second pass):** the full no-game suite green — 22 files, **0 failures** — with the ladder's
    behaviour pinned by measurement rather than by arithmetic: `test_the_measured_ladder_fits_inside_the_
    supervisors_patience`, `test_the_relaunch_rung_is_reached_when_the_bridge_fails_SLOWLY`,
    `test_a_cold_port_at_startup_does_not_kill_the_worker`, `test_run_only_settings_never_travel_in_the_
    config_file`, and `test_against_a_real_silent_socket_every_bound_is_honoured` (a real loopback peer that
    accepts and then says nothing — the client's-eye view of the frozen game). No private game was needed:
    **the incident reproduced itself on main at 19:13** while this was being written, and the socket-state
    capture from that recurrence is stronger evidence than a synthetic freeze would have been.
  - **Verified:** the full no-game suite (22 files, 534 named tests, 0 failures) plus five live reproductions on
    ONE private game on port 47812 while the 12-game run kept training at 209-234 steps/s — `close()` 5.00 s
    (was 20.0 s at `timeout=20`), handshake 20.00 s (was 120), no sticky bound after a reset, a step timing out
    at its own bound and reconnecting, a whole failed ladder in 51 s against its 60 s budget, and a game
    **killed outright** mid-episode recovered in 47.7 s with all twelve training ports' owning pids unchanged.
  - **ACTIVATION (the lead does this; it restarts the supervisor, which the usual procedure does not).**
    ```powershell
    # from python/  --  1. pause, so nothing restarts underneath the merge
    New-Item runs\campaign_gates\SUPERVISOR_PAUSE
    ```
    2. **Stop `supervise.py` itself** — Ctrl+C its console and wait for it to exit. This is the step the normal
       config-switch procedure does not have: `supervise.py` is a long-lived process and a trainer restart does
       not reload its code, so the running one would keep the old 600 s window and the old detached spawn.
    3. **Merge `freeze-fix` into main and run the full no-game suite against the merged tree.** The supervisor
       restarts the trainer from main's code, so main must be left in a state that runs.
    4. **Stop the trainer** — Ctrl+C and wait for `Saved models\campaign_gates\latest.zip`, then for the new
       `vec env teardown: clean` line. After a hard stop, resume from the newest `ckpt_*_steps.zip` instead.
    5. `Remove-Item runs\campaign_gates\SUPERVISOR_PAUSE`
    6. **Start the supervisor again** (this is what picks up the new code), from `python/`:
       `python scripts/supervise.py --run campaign_gates --config configs/campaign_gates_full.yaml --count 12 --monitor 1`.
       It finds no trainer, treats it as DEAD, relaunches the twelve games, waits out the boot gate and resumes
       from the checkpoint with the most timesteps. Same run name, same weights, same `best.zip`, same archives.
    7. **Confirm within ten minutes**: `runs/campaign_gates_train.log` is **growing** (it is the regression test
       for the detached-spawn fix and had been dead for seven hours), twelve `runs/campaign_gates/env_*.log`
       files exist and are each getting `reset_start`/`reset_end` pairs, and `status.json`'s `timesteps` is
       climbing. Rollback is steps 1-2 and 4-6 with `git revert` of the merge.- **Mod v0.7.1 — released and installed 2026-09-17, during a 9-minute planned pause.** Two changes plus the
  Python plumbing for the first. The 12-game `campaign_gates` run went down at 20:24:43 and was stepping again
  by ~20:33, resuming from `ckpt_10153414_steps.zip` (the live run was at 10,163,902 when it was paused, so
  ~10.5k steps were lost — well inside the documented 50k).
  1. **`-aibridge-nosteam`: training is hidden from Steam, by default.** A Harmony prefix on Facepunch's
     `SteamClient.Init(uint, bool)` skips the call, so `SteamAPI.Init()` never runs and a running Steam client
     never learns the app is live. **Why one patch is the whole job** (audited against the shipped
     `Facepunch.Steamworks.Win64.dll` and `decompiled/`, written up in `docs/game-internals.md`): `Init` is the
     only thing that registers; `SteamClient.RestartAppIfNecessary` is **never called** anywhere in the game;
     there is **no `steam_appid.txt`** (Facepunch's own `Init` sets the appid env var, which is why launching
     the exe directly works at all); and every consumer — rich presence, user stats, UGC playtime tracking,
     `Shutdown`, the leaderboards — gates itself on `SteamClient.IsValid`, which *is* the private `initialized`
     flag `Init` sets. So skipping it switches all of them off by their own code. The two `SteamClient.SteamId`
     reads that are not directly gated sit behind `LeaderboardsSupported` and behind `SafetyPatches`.
     - **Verified in game.** Control, launched with `--steam`: `HKCU\...\Apps\1229490\Running` went 0 → **1**
       and `RunningAppID` → **1229490**, about 10 s after the copy passed ~1 GB working set. Hidden, the new
       default: the copy reached ~950 MB and `Running`/`RunningAppID` stayed **0 / 0 for the whole two minutes**
       it was polled; all twelve training games then came up hidden and Steam still read 0. In band, the
       handshake reported `mod 0.7.1 ... steam_hidden True`, and that instance loaded `Level 0-2`, teleported,
       stepped and activated a checkpoint normally — so the game runs fine without Steam.
     - **Not verified:** the Steam client's own window and tray. That is GUI state this session could not read;
       the registry pair and `steam_hidden` are what was measured.
  2. **The exit is re-resolved on every `Scan`** (`CampaignObserver.ResolveExit`) — the proper fix behind
     Python's `ExitGuard` stopgap, which stays. **Verified on `Level 0-2` against the RAW bridge** (deliberately
     not through `UltrakillEnv`, whose `ExitGuard` would have masked the regression): fresh load
     `exit (-199.0, -86.1, 277.0)`, teleport onto checkpoint `-55,-11,277`, activated in 1 decision, 20 further
     steps so a rescan lands — `exit (-199.0, -86.1, 277.0)`, **moved 0.0 m**. Before the fix that read
     `(9801, -86.1, 277)`.
  3. **0-1 regression, `campaign_check.py --level "Level 0-1"`:** 1 level load PASS | 2 arsenal SKIP |
     3 checkpoint PASS | 4 death respawn PASS | 5 exit FAIL | **6 gates PASS — 11 gates, hops 0..9 (10
     distinct), 0 unordered, unchanged across a respawn**. Check 5's FAIL is the long-standing teleport-onto-a-
     collider-in-a-switched-off-room artifact (see its gotcha), not a regression.
  - **A no-game test was writing live state, found by this change and fixed.**
    `test_freeze_recovery.py::test_a_single_laggard_port_is_repaired_instead_of_failing_the_whole_launch`
    drives the REAL `games.launch`, so once `launch` began recording `runs/instance_flags.json` that test's
    three fake ports landed in the file a live twelve-game run reads back during a recovery. Caught because a
    one-game launch produced a file listing 47800-47802. The test now stubs `record_instance_flags`. Worth
    remembering as a class: a test that calls a real entry point inherits every side effect that entry point
    later grows.
  - Evidence: mod builds clean, and the full no-game suite green twice (before and after the install) —
    **22 files, 554 named tests, 0 failures** (6 new: 5 in `test_games.py` for the flag, the flags file and the
    retired alias, 1 in `test_supervise.py` for the supervisor's launches).
- **Speed phase plan (user, 2026-09-17): Brutal + a time-scaled completion bonus.** When a level set completes
  reliably on Violent, the speed phase sets `difficulty: 4` (Brutal, the highest this build exposes; the mod applies
  it in memory for AI runs only) at the same time as the completion bonus that scales with official time against the
  human reference, so combat is re-learned once, on the enemy set the records will be set on. Not before: the
  curriculum/route work is the bottleneck now and Brutal would only lower completion rates. `times.md` rows carry the
  difficulty, so Violent bests stay as their own rows.

## 2026-09-18 — Learning-progress curriculum weighting, and mod v0.7.2 route targets

- **Learning-progress curriculum weighting (branch `curriculum-progress`, 2026-09-18, MERGED and live from the 0.7.2 pause).**
  The fresh-load weight was `max(floor, 1 - fresh_completion_rate)`, so **the level with the lowest completion
  rate got the most fresh starts**. Measured on this run: Level 0-3, blocked at 0 completions, took **41-45% of
  every fresh start for hours** while 0-1 and 0-2 — which were completing — were held at 33% and 22-24%, and
  their own rates slid (0-1 0.55 -> 0.16-0.27, 0-2 0.69 -> 0.26-0.52 over the same window). A blocked level
  starving the levels that are learning is the rule working as written, not a bug in the data.
  - **The rule now** (`curriculum_weighting: progress`, `configs/campaign_gates_full.yaml` only): every unlocked
    level keeps a retention floor (0.10 of the mass, shrunk so the floors never exceed half of it); a **blocked**
    level gets the floor and nothing more; the rest is split by **learning progress** `|fast - slow|` over each
    level's own fresh episodes; a 0.5 cap then spreads any one level's excess over the others that are still
    learning. Blocked = no completion in 100 fresh episodes AND not progressing (its slow average has fallen
    more than 0.02 below its own high-water mark, or does not move at all). Unlock rules are untouched. Full
    mechanics and the two rejected designs are in the curriculum bullet under Layout.
  - **The replay, `scripts/replay_curriculum.py` over the live `runs/campaign_gates/episodes.jsonl`** (1,854
    episodes with a level; mean share of the fresh draws per hour under each rule, floor 0.1, cap 0.5, blocked
    after 100 dry fresh episodes). `dry` is 0-3's fresh episodes since its last completion:

    | hour | 0-1 today -> progress | 0-2 today -> progress | 0-3 today -> progress | 0-3 dry | 0-3 blocked |
    |---|---|---|---|---|---|
    | -7h | 31% -> 43% | 28% -> 24% | 40% -> 33% | 6 | 7% |
    | -6h | 33% -> 30% | 29% -> 29% | 38% -> 41% | 19 | 0% |
    | -5h | 33% -> 43% | 30% -> 19% | 38% -> 38% | 38 | 0% |
    | -4h | 34% -> 33% | 26% -> 35% | 40% -> 32% | 64 | 0% |
    | -3h | 33% -> 38% | 23% -> 28% | 44% -> 34% | 99 | 0% |
    | **-2h** | 33% -> **46%** | 22% -> **40%** | 45% -> **14%** | 125 | 90% |
    | **-1h** | 33% -> **47%** | 23% -> **37%** | 44% -> **16%** | 138 | 85% |
    | **-0h** | 34% -> **50%** | 24% -> **40%** | 42% -> **10%** | 170 | 100% |

    The two rules agree until 0-3 crosses 100 dry fresh episodes, which is the point: the damping waits for the
    evidence instead of guessing. From there the draws move to the levels that are completing, and 0-3 keeps the
    retention floor rather than being dropped.
  - **Evidence:** full no-game suite green, **22 files, 572 named tests, 0 failures** (18 new: 12 in
    `test_campaign.py` for the rule, the score, the blocked test and a 400-table fuzz; 5 in `test_progress.py`
    for the published statistics, the restart carry, the end-to-end starvation case and the replay script;
    1 in `test_campaign_env.py` for the config guard).
  - **Activation** (not done here): merge, then bounce the trainer through the supervisor pause file — the
    trainer reads the code only at start. See the supervisor entry under Commands. The first trainer line to
    check is `curriculum: 30 levels, unlocked [...], weighting progress`.
  - **Risks, in the order they would show.** (1) **A warm-up window after every restart**: the statistics carry
    through `status.json`, but the windows do not, and a level needs 20 fresh episodes before its pair is
    trusted — until then the run weights by rate exactly as today, which is the intended fallback but also means
    the first ~30 minutes after a bounce look unchanged. (2) **`gates_total` is a ratchet**: a level whose
    ladder reports a longer route later rescales its own score once, which shows as one spurious blip of
    learning progress. (3) **Noise still drives the weights** — `|fast - slow|` on a swinging score moves the
    shares hour to hour (the table above shows 0-1 between 30% and 50%); the floor and the cap bound it, and the
    number to judge is still `campaign.levels[*].episodes`, not the weight. (4) **A level damped just before it
    would have broken through** keeps 10% of fresh draws plus every checkpoint respawn, and one completion
    clears the damping for the next 100 fresh episodes. Watch the per-level `w` and `prog` rows on the
    dashboard, and 0-1's and 0-2's fresh completion rates over windows of >= 50.
- **Mod v0.7.2 + the route-target fixes — released and installed 2026-09-18, during a planned pause.** Two
  independent target bugs, both found by a live probe on a private game (port 47812) while the 12-game run kept
  training. The measurements are under the transform-vs-ground gotcha; what follows is what shipped.
  1. **`exit.ground_pos`** (mod, `CampaignObserver.UpdateExitGround`): the NavMesh-snapped standable point near
     the pit, kept from the `SampleExit` call the path hint already made, reported on the same 4-obs cadence and
     computed **before** the player-end snap so it is reported even where `path` reads `none` (0-1's own spawn).
     `null` when NavMesh holds nothing within 95 m; absent on an older mod. Python's `GateProgress._exit`
     prefers it and falls back to `exit.pos`, which moves look mode 2, `gate_approach` and the target slots
     448-455 onto standable ground in one edit. **Observation slots 0-4 are deliberately unchanged.**
     `ExitGuard` drops a rejected report's `ground_pos` with its `pos`, so a banished twin's sample can never
     become the target. New per-episode `exit_ground_dist_min` through `CAMPAIGN_INFO_KEYS`, `status.json`,
     `poll_status.py` and the dashboard's "closest to exit" row; `exit_dist_min` keeps its old definition.
  2. **0-3's hallway rung moved, and a ground rule behind it.** `rung_overrides.json` (new) moves the rung to
     z 340 in a way `build_routes.py` re-applies on every regeneration, and `GateProgress._on_ground` refuses
     to credit any **room-trunk** rung from the air with no ground within `route_ground_m` (8.0 m). Gates are
     exempt by object identity, so all 18 gates levels and Cyber Grind are provably untouched.
  - **Judge the fixes on:** 0-3's bowl share (26 of 66 fresh route episodes ended there before), 0-2's
    fall deaths at full health (11 of 18 respawns in one episode), and `exit_ground_dist_min` falling where
    `exit_dist_min` had a floor it could never cross. `route_source` 2 and `gates_reached` on 0-3 are the
    readout for the rung move: a fresh 0-3 load should now stop crediting rung 9 from the main room.
  - **`metrics_log.csv` gained a column** (`exit_ground_dist_min`), and `poll_status.py` keeps an existing
    header, so the old file was moved aside at the pause.
  - **The in-game check caught a regression IN THE FIX, and it is the reason the check exists.** The first
    build reused `SampleExit` — nearest NavMesh in ANY direction, which is right for the path hint's endpoint
    — and on `Level 0-2` that returned **(-199, -133.5, 277)**, another **47.4 m DOWN the pit shaft**. That
    would have made the exit target 47 m worse than the bug being fixed, and every no-game test passed, because
    a fake bridge cannot know which way the real mesh lies. `UpdateExitGround` now searches **upward** from the
    pit (heights 0/20/40/60/80/100 at a 30 m radius) and accepts only a hit at or above it; `SampleExit` is
    left exactly as it was for the path. `campaign.exit_ground_point` repeats the same rule in Python, so an
    older or future mod cannot reintroduce it, and both callers — the target and `exit_ground_dist_min` —
    share that one rule. Nothing found means `null` and a fall back to `exit.pos`, so the worst case of the
    whole mechanism is a no-op.
  - **Measured on all three levels, and the number is a constant: the `FinalPit` transform sits exactly
    62.2 m below its own room's floor.** 0-2 pit y -86.1 -> ground y -23.9; 0-3 19.9 -> 82.1; 0-1 -17.1 ->
    45.1. Same prefab, same offset, which is why the "61-75 m" estimate was in the right band, and it is a
    cheap sanity check on any future reading: a `ground_pos` that is not ~62 m above its pit is suspect.
    0-2's is 1.1 m in y from where the level's two real completions triggered (y -25).
  - **The offline rescan of all 114 shipped rungs (2026-09-18), and the measure that actually works.** Four
    obvious measures do NOT separate the known-bad rung from its neighbours, and one of them was tried first
    and discarded: **raw "share of the reach cylinder over foreign floor"** scores the known-bad `0,10,331` at
    **29.0%** while four rungs on the same level that live play proves are fine score **higher** (44.6, 49.0,
    33.5, 31.9). R4's standable-cell view, floor-height-per-column, room-ownership share and geodesic
    separation all fail too. The measure that works has two extra ingredients: (1) count only foreign-floor
    air **within `route_ground_m` of drop**, i.e. what is still creditable after the ground rule — that alone
    takes the known-bad rung to 7.3% and its neighbours to ~0, because their foreign floor is 14-50 m down;
    and (2) **attribute** the remaining air to the room it belongs to, because the signature of the bug is
    that the foreign floor is the room the route comes FROM. On `0,10,331` 100% of it is `1 - Main Room -
    Floor 1`, the previous rung's room. The fixed `0,10,340` scores 0.0%, and an independent search for the
    nearest clean point on the hallway floor proposes z 339.2 against the 340 that was applied by hand.
    It needs a probe box reaching **46 m down**; the generator's own +/-14 m box cannot see the floor below.
  - **Four more rungs are flagged and NOT yet shipped**, because the analysis itself asks for a live probe
    first and each move re-aims a level. Ready to apply as `rung_overrides.json` entries at the next pause:

    | level | rung (`was`) | creditable foreign air | attributed to | proposed |
    |---|---|---|---|---|
    | 0-3 | `-87,-15,413` `6 - Path 1 - Boss Arena` | 31.6% | **100% `5 - Path 1 - First Encounter`, the previous rung's room** | `[-95.2,-15.0,413.2]` `-95,-15,413` (0.0%) |
    | 0-3 | `76,50,397` `9 - Windtunnel` | 25.1% | 42.7% `7 - Path 2 - Menacing Room`, two rungs earlier | `[64.8,50.0,377.2]` `65,50,377` (0.2%) |
    | 8-3 | `201,155,928` `15 - Space Mass` | 41.0% | unattributed | `[202.2,157.8,916.8]` `202,158,917` (0.1%) |
    | 8-3 | `513,254.2,941` `13 - Space Streets` | 30.0% | unattributed | `[512.8,256.8,930.8]` `513,257,931` (0.0%) |

    Left alone with a reason: **7-1 `217,1,467`** (43.8%, but the foreign room is the NEXT rung's, so the
    error is credit-early-by-one, not stranding, and the best point still scores 5.7%); **7-2 `0,28,340`**
    (contaminated by `1 - Empty Hall` with nothing better than 8.8% within 26 m — it needs a different
    mechanism, not a move); and the **5-2 / 7-3 / 7-4 rungs with a real lower deck inside the cylinder**,
    whose floors are parented outside any room `RoomTrunk` parses, so a benign second standing place cannot
    be told from a cross-room leak offline. 88 of 114 rungs are clean, and a further 12 are clean only
    **because of** the ground rule (8.3-66.6% raw air, 0.0-3.6% after it) — those would all have been
    suspects a day ago.


## 2026-09-18 — Per-level specialists: the driver, and specialist mode going live

- **Per-level SPECIALISTS: one policy per level, trained sequentially (branch `specialists`, 2026-09-18, BUILT,
  NOT YET ACTIVATED).** The lead's decision, on twelve hours of measurement from the live twelve-game run: a
  single shared policy on a multi-level mixture **thrashes**. Whichever level receives the fresh-start share
  improves while the others regress.
  - **The numbers.** Level 0-1's fresh completion rate went **0.65 -> 0.13 -> 0.36** as the draws moved around
    it, and Level 0-3 fell from **7.3 gates reached to 3.6** as soon as its share was damped. Learning-progress
    weighting (the 2026-09-18 change above) did what it was designed to do — it moved the draws to the levels
    that were moving — but it changed WHICH level is starved, not that one is. One set of weights cannot hold
    twelve levels' worth of route at once, and every hour spent on the mixture is an hour of one level being
    forgotten while another is learned.
  - **The plan.** Train ONE SPECIALIST PER LEVEL, one level at a time, all twelve games on the current level,
    each specialist initialised from the previous level's best (the first from the current shared policy), and
    chain the specialists afterwards for a full-game run — one policy per level. `scripts/campaign_driver.py`
    is the driver and `scripts/full_run.py` the chainer; see their Layout entries for the mechanism and the
    Commands entry for how to start, pause and watch it.
  - **Nothing about the environment changes.** No observation, action or reward semantics move: a stage is the
    existing single-level env mode (`level:`, no `levels:` list), which is what `configs/campaign_0-1.yaml` has
    always used. `tests/test_specialists_config.py` pins every setting, every reward weight and every
    hyperparameter of `configs/specialists.yaml`'s stage template equal to `configs/campaign_gates_full.yaml`,
    so a specialist cannot silently drift away from the policy it was initialised from.
  - **The stage rule, as implemented.** A stage ends when the level's fresh completion rate over its last 50
    fresh episodes reaches 0.5 with `fresh_window >= 30` — latched, so a later dip cannot deadlock it — AND 300k
    steps have passed since the later of that moment and the last time `keep_best.py` moved `best.zip`, so a run
    still setting new bests keeps training and the peak is what gets promoted. OR the stage has consumed 6M
    steps, in which case it moves on regardless and is recorded `"unfinished"` for a later revisit.
  - **Evidence:** full no-game suite green — **25 files, 612 named tests (plus `test_progress.py`'s 26 that
    print no count), 0 failures**, run in the `specialists` worktree with `PYTHONPATH` pointed at it. 42 of
    those tests are new (26 driver, 9 chainer, 7 config).
  - **Risks, in the order they would show.** (1) **30 stages x 6M steps is 180M steps at ~200 steps/s ≈ 250
    days** if every stage runs to its cap; the cap is a backstop, not a plan, and the early levels are expected
    to finish on the rate in well under 1M. Watch the first three stages' actual cost before trusting the
    ladder's total. (2) **Forward transfer is assumed, not measured** — each specialist starts from the
    previous level's best, and if that turns out to be worse than starting from the shared policy every time,
    the `--init` of a stage is the lever and the sidecars record exactly which file each one used. (3) **A
    specialist is only as good as its stage's last 50 fresh episodes**: `best.zip` is chosen by
    `keep_best.py --metric campaign`, which needs `fresh_window >= 20`, so a stage that ends on the step cap
    with no completions promotes a checkpoint nothing ever scored. The sidecar says `"unfinished"` and
    `source_kind: "newest"` when that happens. (4) **The driver and `supervise.py` must never run together**;
    the activation checklist under Commands stops the supervisor first for exactly this reason. (5) **30 stages
    means 30 model directories** of checkpoints — `ckpt_*` files are gitignored, but the disk is not infinite,
    and a stage's numbered checkpoints are worth pruning once its specialist is promoted (keep the one
    `best.json` names).
- **Specialist mode ACTIVE (2026-09-18 07:01).** The shared `campaign_gates` run was stopped at 17,002,318 steps (its
  last 12 hours thrashed between levels: 0-1 0.65 -> 0.13 -> 0.36 as its fresh-start share moved, 0-3 7.3 -> 3.6
  rungs once damped) and `scripts/campaign_driver.py` now trains one specialist per level in mission order, all 12
  games on the current level: stage 1/30 `spec_0-1`, initialised from `models/campaign_gates/ckpt_17002318_steps.zip`
  (the generalist that finishes 0-1, 0-2 and 0-3). Started through `runs/start_driver.cmd` (a bare
  `Start-Process cmd /c "..."` mangles the quoted `"Level 0-1"`). Stage rule: fresh rate >= 0.5 over 50 (window >= 30),
  latched, plus 300k settle steps after the last `best.zip` move, else the 6M cap. Do NOT run `supervise.py` while the
  driver runs. Status: `python scripts/specialists_status.py`; pause: `runs/specialists/DRIVER_PAUSE`; full-game
  chain: `python scripts/full_run.py`. The speed phase (Brutal + time-scaled completion bonus) will revisit each
  specialist in turn.
- **0-3's route trunk chained both branches of the fork, and the ladder paid more for the detour than for
  finishing (2026-09-18, branch `route-0-3`).** `Level 0-3` ("Double Down") forks into Path 1, the lower branch
  (y 5 -> -15), and Path 2, the upper one (y 50) -- you take one. `scripts/build_routes.py` shipped **both,
  chained in series**, as an 11-rung / 688 m trunk.
  - **Mechanism.** Guard T collapses a parallel branch only when the branch marker is a LEADING letter
    (`RX_G_BR = ^([A-Z])(\d{1,2})\s*-\s+`, e.g. `A1 - ...`). 0-3 spells its fork in the MIDDLE of the room name
    -- `5 - Path 1 - First Encounter` against `7 - Path 2 - Menacing Room` -- so `name_kind()` returns
    `('ord', N, '')` for all twelve rooms, no rung is kind `branch`, and guard T's `len(prefixes) <= 1` test is
    vacuous on exactly the level that needed it (`drop_trunk: []`, `drop_i6: []`).
    `tests/test_route_files.py::test_only_the_trunk_ships` passed on a route shipping both branches for the
    same reason, which is why `test_the_0_3_wing_stays_off_the_route` now sits beside it.
  - **Evidence (offline only; nothing was probed in game).** `gates_reached` is the count of DISTINCT rungs the
    player physically stood on -- `_is_reached` is a grounded 8 m x 6 m cylinder and `_note_reached` pays once
    per new lower `hops` -- so it reads back the route an episode took. In `runs/campaign_gates/episodes.jsonl`
    (895 0-3 episodes, 606 fresh) the level's **only four completions credited 3 or 4 rungs and never more**:
    gr 3 / 3 / **4** / 3, the gr=4 one being the 263.903 s row in `times.md`. Its stored trace
    (`best_runs/Level_0-3.json`, 4064 points) reaches exactly `1 - Main Room - Floor 1`, `2 - Side Hallway`,
    `10 - Main Room - Floor 2`, `11 - Boss Arena - Floor 2` and enters **none** of the seven wing rooms (closest
    approach 20.3 m to the Side Arena, 157.9 m to the Menacing Hallway); its climb to Floor 2 is 39.4 m in 3.0 s
    inside the main-room footprint (x -14..+5, z 303..330), not via the Side Stairway. In `runs/spec_0-3` (238
    fresh episodes, 2.83M steps, **0 fresh completions**) the specialist had learned the detour instead: 115 of
    238 fresh episodes (48%) walk the wing to `7 - Path 2 - Menacing Room`, wing endings rose 47% -> 68% over
    the stage, mean episode length 6185 -> 7419, and only 16 of 238 ever got past it. The arithmetic is the
    whole story: at `gate: 15.0` a 7-rung wing tour pays **105**, and `level_complete` pays **100**. The trunk
    was paying more for the dead end than for the exit, with certainty instead of at 1%.
  - **What the data did NOT say.** The 12 fresh episodes that credit exactly 2 rungs -- Main Room Floor 1 then
    Main Room Floor 2, skipping the wing -- are spread over the whole stage (ts 19.38M-21.13M) and are the only
    ones that get near the exit. That proves the main-room ascent is **available from level start and reachable
    by the current policy**, which is what the trim depends on; the ascent's MECHANISM (0-3 has 7 moving
    platforms) is still unidentified, and no in-game probe was run.
  - **The fix, data plus one generator hook.** `rung_overrides.json` gained a second kind of entry,
    `{"name": ..., "drop": true, "why": ...}`, honoured by `build_routes.apply_drops()` between I2 and R1 --
    so R1 counts what ships and `hops` stays gapless -- with three rules of its own: matched by room NAME, an
    unknown name REFUSED rather than ignored, and the whole list for a level refused if it would go under
    `MIN_RUNGS` (a sub-minimum file reads as NO file and would drop the level to its gate ladder in silence).
    A plain hand-edit could not be used: the next `build_routes.py` run would have restored the detour, and the
    override README forbids a move entry from removing a rung. `route_Level_0-3.json` was then REGENERATED, not
    hand-written: **4 rungs, hops 3..0** (`1 - Main Room - Floor 1`, `2 - Side Hallway - Floor 1`,
    `10 - Main Room - Floor 2`, `10B - Second Encounter + 11 - Boss Arena - Floor 2`), tour_ratio 0.944 ->
    **1.000**, polyline 688 -> **178.4 m**, max gap 117.6 -> **92.5 m**, last rung to pit 119.7 m unchanged, and
    the shipped file now records the seven dropped rooms in `trunk_dropped`. `checkpoints_within_60m` falls
    4/4 -> **1/4** and `legs_witnessed` 9/11 -> **2/4** because three of the level's four checkpoints are IN the
    wing -- both are diagnostics, and what they now say is true. The 2026-09-18 Side Hallway position override
    still applies cleanly (`override_refused` empty). **No config, reward weight, observation or mod change**;
    `prefer_route_when_collapsed: true` is untouched, so 0-3 still prefers its trunk and 4-3's is unaffected.
  - **Tests.** New `tests/test_route_drops.py` (10 tests) lifts `apply_drops`/`apply_overrides` out of
    `build_routes.py` by AST -- importing it would run `refuse_if_commit_high()` and fail whenever the box is
    busy -- and pins the three rules plus what actually ships. `test_route_files.py` 23 -> 25 tests
    (`test_the_0_3_wing_stays_off_the_route`, `test_a_drop_entry_names_a_room_that_was_really_there`;
    `EXPECTED_LADDERS`, `TOTAL_RUNGS` 114 -> 107 and `build_routes.EXPECTED_SHAPE` moved with the data, which is
    the pin working). Full no-game suite green, one file at a time, in the worktree: **29 files, 0 failures**.
    `build_routes.py --levels 0-3 --dry-run --validate` reports `0-3 unchanged` against the new shape row; its
    two FAIL lines ("the gates guard passes on 1 levels, not the measured 18" and the EXPECTED_SHAPE /
    EXPECTED_SHIPPED ordering) are **pre-existing on main for any single-level `--validate`** -- verified by
    running the identical command on the unmodified main tree, which prints the same two lines and exits 1.
  - **What to watch, and the honest limit.** Baseline, read at the bounce (`spec_0-3` ts **21,697,762**;
    the trainer was killed and the driver restarted it 2 minutes later from `ckpt_21701566_steps.zip`, so
    **the clock for +400k steps starts at 21,701,566**): `mean_100.exit_ground_dist_min` **131.83 m**,
    `mean_fresh_100.gates_reached` **5.00**, `mean_100.length` **6757**, `mean_100.targets_parked` **2.54**,
    `end_reasons_100` stuck 77 / max_steps 12 / bridge_reset 5, wing endings **68 of the last 100**, fresh
    episodes reaching `gate_hops_best <= 1` **12 of 238**, `part_gate` **43.26**, `part_gate_approach`
    **28.75**, `mean_100.reward` **-45.21**, fresh completions **0 of 238**. Judge at +400k steps
    on: wing endings (`end_pos` x < -40 and z > 370; should fall well under 20/100), `exit_ground_dist_min`
    (should fall under ~100 m -- the Boss Arena Floor 2 rung is ~97 m from the exit ground point, so simply
    getting into the arena drives it there), `length`, `max_steps` endings, `targets_parked`, and the share of
    fresh episodes reaching `gate_hops_best <= 1`, which is the real bottleneck rate. Do NOT judge on
    `gates_reached` or `part_gate`: both are capped by the rung count and fall by arithmetic (4 rungs instead of
    11) whatever the policy does. Do NOT judge on the completion rate before ~1.5M steps -- the stage produces
    ~84 fresh episodes per 1M steps, so 400k is ~34 of them and the best rate ever observed on 0-3 is 1.8%.
    **REVERT TRIGGER:** if wing endings fall but `exit_ground_dist_min` is not under ~120 m by 400k steps, the
    wing was at least carrying the agent upward and the trim was not the binding constraint -- revert and probe
    the Floor 1 -> Floor 2 transition in game.
  - **Not claimed.** That the trunk was the ONLY cause. The stall reproduces across layers: 0 completions in 292
    fresh episodes before any route file existed, 2 in 113 under the gate ladder, 2 in 201 under the trunk in
    the shared run, 0 in 238 under the trunk in the specialist -- and 2/113 vs 2/201 is not a significant
    difference. What is removed here is a measured, arithmetically certain perverse incentive (105 > 100) and a
    ladder that pointed at rooms no completion has ever entered; that is not the same as a fix for 0-3. The
    generator's blind spot itself (`RX_G_BR` sees only a LEADING branch letter) is NOT fixed -- it is a
    campaign-wide correctness change and is documented in the `OVERRIDES_NAME` comment for whoever takes it.
- **Speed stages built on branch `speed-stages` (2026-09-18, NOT merged, NOT live).** The interrupted WIP
  (`db384dc`) was rebased onto `origin/main` at `59ccf91` — one conflict, `scripts/full_run.py`, where the
  memory refactor's `cap_blas_threads()` before `import numpy` and the speed branch's `COMPLETE` import both
  had to survive. Nothing of the speed work touched `scripts/train.py`, so the light-worker split
  (`ultrakill_ai/training.py`, `ultrakill_ai/envfactory.py`, `EpisodeMonitor`) needed no re-application; the
  speed reporting lives in `ultrakill_ai/progress.py`, which `training.py` still owns, and `envfactory.py`'s
  import set is untouched (`tests/test_light_workers.py` green).
  The spec is `docs/superpowers/specs/2026-09-18-speed-stages.md`, §1-7 as approved plus a §8 for two
  additions the lead made afterwards:
  - **§8a, the hold line.** As specced, a speed stage that hit its 8M cap was recorded `"unfinished"` and the
    ladder walked on to 0-4 anyway — the exact opposite of "dont have it promote to 0-4 untell it gets better
    times on these levels". `configs/specialists.yaml` now carries `hold_before: "Level 0-4"` (nullable to lift
    it). No stage at or after that level starts while any stage in front of it is not `"done"`; while the line
    is up, `Driver.choose_stage` round-robins the not-done stages in front of it, fewest ended rounds first and
    ties in plan order, each round a whole fresh step budget. `Driver.round_init` resumes a round from the
    stage's OWN newest weights (`choose_resume` on its model dir, else `best.zip`, else that level's promoted
    specialist) — never from scratch, never from another level. A `(level, speed)` stage is only eligible once
    `(level, complete)` is `"done"` and `models/specialists/<level>.zip` exists; if everything in front of the
    line is blocked the driver logs it and stops (`"held"`, exit 1) rather than spinning. Rounds are recorded
    in `Stage.round`, in one history entry per round, and in the promoted sidecar, so
    `specialists_status.py` prints `holding before Level 0-4: waiting on <stages>, round N`.
  - **§8b, the target scale.** Measured: the 0-2 specialist's 139.5 s already scores rank S, so a bare S
    threshold would have promoted 0-2's speed stage on its first observation with zero improvement. New plan
    key `speed.target_scale` (default **0.75**): `target_seconds = scale x campaign.ranks.time[-1]`, applied in
    exactly one place — `UltrakillEnv._note_speed_target` — so the reward and the driver's rule keep agreeing
    through `info["target_seconds"] -> status.json -> driver_state.json -> the sidecar`. A per-level
    `speed.targets` override is NOT scaled. The raw threshold travels beside it as `s_rank_seconds` and is
    recorded in the sidecar, never compared against.
  **Verified against the live run without touching it.** The real `runs/specialists/driver_state.json` (stage 3
  `Level 0-3` running at 18,752,038 steps, 0-1 and 0-2 `"done"`, no `kind` field anywhere) was read, never
  written. `tests/test_campaign_driver.py` loads a COPY of that real file against the new 33-stage plan and
  pins that `reconcile` keeps it at index 2, that `tick` keeps driving it, and that nothing is killed,
  launched or spawned. A `campaign_driver.py --dry-run` was then run from an isolated scratch cwd holding only
  copies (plan, state, `runs/spec_0-3/status.json`): it reported `STAGE ... Level 0-3 (complete): ok,
  2,724,960 steps into the stage, rate 0.000 over 38 fresh`, printed the hold line as waiting on 0-3 complete
  plus the three speed stages, and exited 0 having spawned and killed nothing. Whole no-game suite green, one
  file at a time, 28 files.
- **Speed stages reviewed, fixed and MERGED (2026-09-18).** Two adversarial reviewers (an RL-exploit lens and
  a driver-logic lens) read `speed-stages` before it landed and independently returned **the same blocker**.
  Full write-up: `docs/superpowers/specs/2026-09-18-speed-stages.md` §9. The short version, because the
  mistake generalises: *the mechanism was wired correctly end to end and still could not have worked, because
  two of its three decision points read the wrong statistic.*
  - **The blocker.** The speed promotion clause and `keep_best --metric time` both keyed on `status.json`'s
    `campaign.best_time`, which is the run's **lifetime minimum** over every fresh completion by any of the
    twelve envs — `ProgressCallback` only ever lowers it and `_restore` carries it through every trainer
    restart and into every later round. One lucky load would have satisfied `best_time <= target` for the rest
    of the run, and the hold line — whose entire job is to refuse the ladder a slow policy — would have opened
    on it. Measured on the live runs the same day: `spec_0-1` best **243.4 s** against a median of **490.9 s**;
    `spec_0-2` best **139.5 s** against **236.5 s**. The run's single best is about half its typical
    completion in both. Both clauses now read `campaign.median_time_50`, which `ProgressCallback` already
    computed and `poll_status.py` already wrote into `metrics_log.csv`; `best_time` is reported everywhere it
    was and gated on nowhere. A second consequence of the same defect: because `-best_time` is monotone
    non-decreasing, every `--metric time` sample after the last record tied at the top score and the ranking
    fell through to the tie-break, so it silently ranked the completion RATE and its degradation warning was
    unreachable.
  - **The reward had no gradient where the policy is.** `max(target/official, 0.25)` is flat past 4x the
    target. Over the 68 fresh 0-1 completions in the live log (median 481.9 s, target 90 s), **58 of 68 (85%)**
    sat exactly on the clip: a flat 75% pay cut carrying no information about the clock. The floor is now an
    asymptote, `scale = 0.25 + 0.75 * target/official` above the target, so a slower completion always pays
    strictly less than a faster one and a completion still can never pay under 25. 482 s pays 39.0, 243 s
    52.8, 150 s 70.0, 90 s 100.0. Untrained: nothing has run against it.
  - **The stage paid 4x more for what it does not measure.** The bonus is scaled only on a fresh-start
    completion and the stage is scored only on fresh-start episodes, so at `fresh_start_prob: 0.2` a
    checkpoint respawn paid the full 100 — and 37% of live `spec_0-1` completions were respawns. `fresh_start`
    is not in the observation, so PPO would have fitted one baseline across both and given every fresh-start
    completion a negative advantage. A speed stage's generated config now carries `fresh_start_prob: 1.0`.
  - **A speed round can no longer regress a committed specialist.** `promote()` overwrote
    `models/specialists/<level>.zip` unconditionally, cap-ended rounds included, with no comparison against
    the file there; `models/specialists/` is the only copy in git. `refuse_promotion` now declines when a
    SPEED stage did not end `"done"` and the level already has a specialist. The round's weights stay in
    `models/spec_<level>_speed/`, which is where `round_init` resumes from anyway.
  - **Three smaller holes.** (1) A new round reset the latch but reused the run directory, so its first tick
    could re-latch on the previous round's `status.json` tail and record `"done"` a settle later on data the
    cap had just rejected — `Stage.stale_below` now discards samples at or below what the run had written when
    the round began. (2) `--start-at` bypassed the hold line entirely and would silently re-run a finished
    stage after a `"held"` exit left `current: null` (which `runs/start_driver.cmd`, carrying
    `--start-at "Level 0-1" --init <17.0M ckpt>`, would have done); `start_at_objection` refuses both, behind
    `--ignore-hold` / `--rerun-stage`. (3) `begin_stage` takes `start_steps` from what `ensure_trainer` will
    actually resume rather than from `--init`, so a stage's cap and its generated `timesteps` cannot be
    measured from an origin the trainer never visits. `speed.max_rounds: 3` gives the round robin an exit:
    `0.75 x S` on 0-1 is **90 s against a 183.6 s leaderboard best**, so "cannot reach the target" is the
    expected case, and the driver now stops with a loud `HELD` instead of looping.
  - **Rejected:** annealing `speed_target_scale` 1.0 -> 0.75 as a cure for the value-function shock (at 1.0
    the target is still 120 s against a 482 s median — the same ratio regime; the loop half of that finding is
    fixed by `max_rounds`), and clearing `campaign.best_time` per round (unnecessary once the gate is the
    median, and `times.md` wants the cumulative record).
  - **Verification.** Whole no-game suite green one file at a time with `PYTHONPATH` set to the worktree, 28
    files. New tests fail before each fix and pass after. The live driver state file was read and never
    written; a `--dry-run` from an isolated scratch cwd holding only copies reported `Level 0-3 (complete):
    ok` with the hold line waiting on 0-3 complete plus the three speed stages, exit 0, nothing spawned,
    killed or launched. **No in-game validation of any of it**: no speed stage has ever been trained, and
    whether `0.75 x S` is reachable on any level is unknown.

### 2026-09-18 — Never idle the machine: `max_rounds: 0` under both kinds, and `target_scale: 1.0`

The lead's two decisions on the plan file, hours after the speed-stage merge (e80651d) and before any speed
stage has run. Both are `configs/specialists.yaml` only: no code changed, and the driver already supported
each value (`Driver.stage_blocked` treats `max_rounds <= 0` as "no cap"; `target_scale` is applied in exactly
one place, the env).

- **`stage.max_rounds` and `speed.max_rounds` are both 0 — unbounded rounds.** `max_rounds: 3` came out of
  the adversarial review as an exit for a round robin that had none, and it works: after three rounds each the
  driver logs `HELD` and exits 1. But the driver exiting is the failure, not the cure. **Nobody watches this
  run** — no LLM monitors, by the user's standing instruction, and the token-free watchers (`post_times.py
  --watch`, `mem_guard.py`) do not restart a driver. A `HELD` exit is therefore hours or days of twelve idle
  ULTRAKILL instances and an idle trainer, against the alternative of continuing to train exactly the stages
  the hold line is waiting on. Training them is strictly better even when the target is out of reach: the
  weights keep improving, `keep_best` keeps the peak, and `times.md` keeps getting rows. The round robin now
  runs until the targets are met or a human lifts `hold_before`. A `HELD` exit is still possible and still
  means what it should — nothing is eligible for a reason training cannot fix, e.g. a speed stage whose
  complete stage is not `"done"` — and `test_with_no_round_cap_rounds_alone_never_hold_the_ladder` pins both
  halves.
- **`speed.target_scale` 0.75 -> 1.0 — the level's own S-rank time, unscaled.** 0.75 was chosen while the
  promotion gate was the run's LIFETIME BEST, which one lucky load sets for good: `spec_0-2`'s 139.5 s best
  already beat 0-2's S threshold, so a bare S target would have promoted that stage on its first observation
  with zero improvement. The same review replaced that gate with `median_time_50`, the median official time
  over the completions in the last 50 fresh episodes — and against a median, S is already a demanding,
  objective bar: `spec_0-1`'s median is **482 s against an S of 120 s**, `spec_0-2`'s is **236 s**. At scale
  1.0 the rule reads "the TYPICAL run S-ranks the clock", which is the same sentence the game uses. Pressure
  does not stop at the gate either: `completion_bonus` keeps scaling up to **2x** (`SPEED_BONUS_MAX`) for a
  completion at half the target, and the settle rule keeps a stage training while `keep_best` is still moving
  `best.zip`. Note 0.75 x 120 = 90 s was *harder* than the new 120 s target, so this loosens the gate — which
  is the point: it is now reachable in principle from a 482 s median, and it is the level's own number rather
  than a hand-picked fraction.
- **`runs/start_driver.cmd` lost its two flags.** It still carried `--start-at "Level 0-1" --init
  models\campaign_gates\ckpt_17002318_steps.zip`, and `start_at_objection` (same review) now refuses
  `--start-at` for a stage that has already run — so that file would have hard-failed (argparse exit 2) at any
  restart made while `current` was null, which is exactly the state a stage boundary or a `HELD` exit leaves.
  Flagless it resumes `current` (`main()` only reads the flags when `current is None`), and with `current`
  null it picks the next stage through `choose_stage` and resumes it from `round_init`. `runs/` is gitignored,
  so the file's exact content is recorded in `docs/commands.md`.
- **Verification.** `tests/test_campaign_driver.py` 53 tests and `tests/test_specialists_config.py` 11 tests
  green, then the whole no-game suite one file at a time. Memory before starting: `mem_guard.py --dry-run`
  reported 12 games, 20.8 GB total, fattest 1.9 GB, system commit **67%**. **Nothing here is validated in
  game** — no speed stage has been trained, so whether an S-rank median is reachable on any level is still
  unknown, and the first speed stage cannot start until 0-3's complete stage is `"done"`.

## 2026-09-18 — The 0-3 main-room ascent: route waypoints, and a third override kind

**Mechanism.** `Level 0-3`'s trunk had four rungs, and the leg between rungs 2 and 3 —
`2 - Side Hallway - Floor 1` (0,10,340) to `10 - Main Room - Floor 2` (0,50,330) — was **10.0 m of
horizontal against 40.0 m of vertical, 76.0 degrees**. The target vector the policy reads in
`campaign_block` slots 448-455 therefore said "up" and carried no usable heading, and the only physical
way up (a spiral ramp round the main room) *starts* by moving **18.7 m away** from that target, which
`gate_approach` pays nothing for. A room trunk can only aim at rooms, and on this level one room's
centroid to the next is not a direction the agent can walk.

**Evidence, measured on `runs/spec_0-3/episodes.jsonl`, 264 fresh episodes since timesteps 21,701,566
(4.3M stage steps).** Zero completions. `gate_hops_best` 2 — the hallway rung — in **219** of them, 3
in 32, 1 in only 13. `end_reason` stuck in 237, `bridge_reset` 20, `max_steps` 7. Median end height
**y 17.8**; share of episodes ending above y 20 **0.398**, above y 40 **0.057**, above y 48 **0.027**.
The four densest 10 m end cells are (10,20,340) 19, (0,20,340) 15, (10,20,330) 11, (10,20,350) 11 —
the policy climbs something by the hallway and wedges at y ≈ 20. Median episode 6,106 steps of a
12,000 cap, median reward −76.6.

**How the ascent is actually done**, from `runs/campaign_gates/best_runs/Level_0-3.json` — the AI's
**own** 263.904 s completion (4064 positions at 15.4 Hz, rank B, difficulty 3), not a human demo. Trace
indices 1000-1042 are **one continuous supported run** up a spiral ramp: per-sample vertical speed holds
+1.0 to +1.7 m with no gravity signature (free fall on this level measures −0.175 m per sample, over five
clean onsets at indices 58, 1774, 1790, 1791, 1813). It is not a jump chain and it has no airborne gap —
an earlier analysis called indices 1028-1035 airborne from a failed standability probe; the velocities
say otherwise, and `standability()` reports 276-714 standable cells at the points now shipped.

**Fix — a third `rung_overrides.json` entry kind, `insert_after` (`build_routes.apply_inserts`).** It
ADDS a rung behind the one it names; hand-editing a route file is still forbidden and the generator is
still the only writer. It runs immediately **after** the moves, so `anchor_was` is the anchor's *shipped*
position and every check runs on final geometry, and before `hops` is assigned from the row order, so the
ladder stays a gapless n−1..0. Rules 7-17 of the `OVERRIDES_NAME` comment, one test each in the new
`tests/test_route_inserts.py` (36 tests): reserved `WP<n> - ` namespace, duplicate, unknown/dropped
anchor, anchor drift > 1.0 m, I2 separation, **co-credit**, R4 standability, seed margin, budget cap, and
**all-or-nothing per level**.

**Co-credit is the rule this incident buys.** Two `_is_reached` cylinders share a point whenever their
horizontal gap is ≤ 2·`REACH_H` (16 m) **and** their vertical gap is ≤ 2·`REACH_V` (12 m). I2's sphere
cannot express that: a reviewed candidate pair sat 13.5 m apart horizontally and 10.8 m vertically —
**17.3 m in a straight line, so I2's 16 m passed it** — while (−4.0, 27.0, 328.3) lies inside both, and
one arrival there would pay two ladder instalments for one place. Applied to inserted rungs only; the
other 13 shipped files have **not** been measured against it.

**The 0-3 ladder now ships six rungs, hops 5..0:**

| hops | pos | rung |
|---|---|---|
| 5 | 0,−10,300 | `1 - Main Room - Floor 1` (absorbed by `_seed_start`, pays nothing) |
| 4 | 0,10,340 | `2 - Side Hallway - Floor 1` |
| 3 | −10.7,21.6,327.2 | `WP1 - Main Room Stack Foot` — insert, trace #1014 |
| 2 | 5.3,36.7,328.9 | `WP2 - Main Room Stack Top` — insert, trace #1024 |
| 1 | −7.1,48.5,315.2 | `10 - Main Room - Floor 2` — **moved** from the centroid 0,50,330 |
| 0 | −82,90,315 | `10B - Second Encounter + 11 - Boss Arena - Floor 2` |

Leg angles 26.6, 34.8, 43.2, 32.6, 29.0 degrees — **steepest 43.2, down from 76.0**. tour_ratio 0.972
(cap 1.35), med gap 22.1, max gap 85.6, legs_witnessed 2/6, checkpoints_within_60m 1/4,
last_rung_to_exit_m 119.7 unchanged. Minimum pairwise distance 20.3 m and no pair co-credits.

**The pay ceiling is unchanged.** The spawn (0, 0.5, 253) is 48.2 m from rung 1 and 77.9 m from the
nearest waypoint, so `_seed_start` still absorbs rung 1 and the ladder is still **five hops deep**: a
level load can earn `gate` 15.0 x 5 = **75.0** by walking the whole ladder without finishing, exactly as
the four-rung file could, against `level_complete` 100.0. The arithmetic that condemned the side wing
(7 x 15 = 105 > 100) is **not** re-created. The margin also survives play: it is 29.7 m at the spawn and
still 23.0 m 2.6 s in, where a candidate rejected in review went negative by trace index 54.

**Credit order is monotone on the one run that finished.** Walking all 4064 trace positions through
`_is_reached`'s own 8 m x 6 m cylinder, first credit per rung is **34, 651, 688, 1021, 1034, 1241** —
strictly increasing. A candidate rejected in review was first credited at index **130**, 8.4 s in and 521
samples *before* the hallway it was anchored behind, which would have paid 30 points for walking over a
low staircase and taken the hallway rung out of `_choose_target` for the rest of the load. The shipped
waypoints credit nothing from index 130, nothing from any of the five live stall cells, and nothing from
directly below (main floor, y −10) or directly above (Floor 2, y 48.5) themselves.

**Also fixed, pre-existing:** `build_routes.py --validate` compared `tuple(EXPECTED_SHAPE)` against
`EXPECTED_SHIPPED` by ORDER. `EXPECTED_SHAPE` is grouped by its own commentary (the two `COLLAPSED_SHIP`
rows lead it) while `EXPECTED_SHIPPED` is sorted, so `--validate` had been printing **FAIL** on every run
since that grouping was written — a permanently red gate is the same as no gate. It now compares the
level sets, which have always agreed.

**Live signal to judge this on, within 400k steps of the restart, on fresh episodes.** The trainer was
bounced at **timesteps 23,731,594** — the step count the driver's own log records resuming from — so the
+400k check falls at about **24,130,000**. The baseline is the 264 fresh episodes above (all of them
before the bounce):

1. **Primary, ladder-independent — read this one, not `gates_reached`:** share of fresh episodes with
   `end_pos` y > 40 rises from **0.057** to >= 0.12, and y > 48 from **0.027** to >= 0.06.
2. Share with y > 20 rises from **0.398** to >= 0.45, and the (0..10, 20, 330..350) end cells lose mass.
3. `gate_hops_best` in the NEW numbering: the hallway is hops 4, so success is the mode moving from 4 to
   <= 2. **Confound:** `gates_reached` rises mechanically because there are more rungs — on its own it is
   not evidence.
4. Ultimate: a first fresh completion (0 in 4.3M stage steps).

**Revert triggers.** At two consecutive checks >= 200k steps apart, the y > 40 share is not above 0.057;
or the y > 20 share has fallen below 0.30 (the waypoints have pulled the policy into the main room and
parked it); or `targets_parked` rises well above its baseline of **42 of 264 episodes** with >= 1. Revert
= delete the three `0-3` entries (the Floor-2 move and the two inserts) and regenerate; the file returns
to its four rungs, because the generator is the only writer. **Note that a revert does not restore the
diagnostics:** `progress.py` ratchets `gates_total = max(gates_total, gates_reached + gate_hops_best)`
and persists it, so 0-3's denominator moves 4 -> 6 at the first post-bounce episode and `progress_score`
for the same physical progress reads 2/6 instead of 2/4 for the rest of the run unless `gates_total` is
cleared by hand in `runs/spec_0-3/status.json` and `curriculum.json`. Nothing in `keep_best.py` or
`campaign_driver.py` reads it; it is a dashboard column.

**Not verified.** (1) Whether the Side Hallway's `ActivateNextWave` "Wave 1"/"Wave 2" must fire to open
the Floor-2 `Door (Large)` at (0,53,330.5) — the wiring was not traced. The hallway rung was **kept** for
that reason, though the live data argues it is not required: **5 of the 264 fresh episodes reached Floor 2
with `gates_reached` 2 and `gate_hops_best` 1**, i.e. `hops_reached` = {3,1}, skipping the hallway
entirely. (2) The second climb, Floor 2 y 48 -> y 72, gets **no waypoint in this pass** — its target
vector is already usable (the boss rung lies 74.9 m west and 41.5 m up, 29.0 degrees) and the live data
says the blocker is climb 1. The measured candidate if it becomes the wall is trace #1159,
(−13.5, 66.0, 313.6). One change at a time. (3) Whether the co-credit rule holds on the other 13 shipped
files. (4) Nothing was run in game: no socket, no `games.py launch/stop`, no `supervise.py`, no mod build.
(5) `route_Level_0-3.json`'s `start_room` still reads `3 - Side Arena - Floor 1`, a room the drop list
removes — a diagnostic field nothing in `campaign.py` reads, but it is wrong.

**The bounce, and the route confirmed live.** `runs/specialists/DRIVER_PAUSE` created, the trainer's two
processes and its twelve `spawn_main` workers stopped individually by PID (never a game, never the driver,
never `taskkill /T`), all twelve ports re-checked as listening so nothing needed `relaunch`, then the pause
file removed. The driver restarted the trainer 20 s later from `latest.zip` at **23,731,594 steps** — the
dying trainer's teardown wrote it, so against the 23,728,342 read just before the kill **no steps were
lost**. No `route file ... is unreadable` line in any `env_478*.log`.

The first ten post-bounce fresh episodes carry `gate_hops_best` 3 in nine of them with `gates_reached` 3,
which is the marker that the SIX-rung file is what the envs loaded: under the four-rung ladder
`_seed_start` absorbed hops 3 itself, so `gate_hops_best` 3 could only ever come with `gates_reached` 1.
Under the new one it means the hallway (hops 4) and then `WP1` (hops 3) — the first waypoint on the climb,
y 21.6. The tenth reached hops 1, Floor 2, and the ten end heights run to a maximum of y 47.9. Ten
episodes is not evidence of improvement and is not offered as any; it is proof the file is live. Watch
`targets_parked`, which is 5 of those 10 at >= 1 against a baseline of 42 of 264 — far too small a sample
to read, and the first thing to re-measure at the +400k check.

## 2026-09-18 — Level 0-3's main room, looked at in game for the first time

**Three offline fixes had shipped on this level in two days and nobody had opened the game.** This pass is a
live rollout of the current 0-3 policy on ONE private instance (port 47812, hand-started with
`games.start_instance` + `-aibridge-nosteam`, never `games.py launch`, killed by its own pid; 47800-47811 were
verified listening before and after, and `instance_flags.json` was not touched). New tool:
`python/scripts/probe_rollout.py`, one JSON line per decision. Evidence, all under `python/runs/probe_0-3/`
(gitignored, still on disk): `rollout_0..9.jsonl` (10 stochastic), `det_0..2.jsonl` (3 deterministic),
`rollout_scripted_0..1.jsonl` (a scripted driver), `geometry_dropscan*.jsonl` (teleport-and-fall probes).
Weights: a COPY of `models/spec_0-3/ckpt_24181522_steps.zip` (24.18M), CPU, config
`configs/generated/spec_0-3.yaml` unchanged.

**Mechanism: `WP1` and `WP2` are points in open air, so the rung they mark can only be credited from
somewhere that is not on the climb.**

- Drop probes (teleport to y 40/44, no input, let the player fall) at **WP1's own (x,z) = (-10.7, 327.2)**
  fall past **y -35 still at -80 m/s**: there is nothing under WP1 for at least 80 m. Under **WP2's (5.3,
  328.9)** the nearest surface is at y ≈ -0.6, **~37 m below it**.
- The policy agrees. Across 13 policy episodes the closest approach to WP1 was **0.73 m**, at
  (-10.3, 21.1, 327.6) with `ground_ray_center` **24.1**; to WP2 **2.72 m**, at (4.8, 35.7, 326.4) with
  **25.7**. Every single closest approach to either waypoint was airborne over 21-30 m of nothing.
- `route_ground_m` (8 m, the rule added the same day for the Side Hallway overhang) therefore refuses the
  credit. **WP2's 8 m x 6 m cylinder was occupied for 540 decisions across the 13 episodes and passed
  `_on_ground` on 11 of them (2%)**; in the ten stochastic episodes, 311 cylinder-steps, 6 on ground, and
  the one credit came at (1.53, 30.81, 329.34) — a block top 5.9 m BELOW WP2, not WP2.
- **WP1 is credited in 13 of 13 episodes and not once at WP1.** The first credit per episode sits at
  x -2.99..-8.62 (WP1's x is -10.7), y 15.7..21.6 (WP1's y is 21.6; nine of ten are 2-6 m below it),
  z 326.2..334.2 — three of them past the main room's north wall face (z 329.5), in the Side Hallway mouth,
  with 2-4 m of ground below. That is the *same* wrong-side credit the hallway rung was moved to stop, one
  rung later. Crediting WP1 advances the ladder to WP2, and the target vector then reads +15 m up with no
  usable heading.

**What the climb actually is.** The 1 m drop scan finds a staircase of narrow block tops against the north
wall — measured rest points: **(-4.30, 10.76, 327.27)**, **(0.93, 11.17, 328.70)**, **(4, 21.5, 330)**,
**(-11.92, 30.83, 326.99)**, **(-3.68, 31.64, 331.66)** and a broad y≈31.6 platform over z 331-334,
**(5.00, 39.98, 328.87)** and **(7.25, 39.84, 329.49)**, then Floor 2 at y 47-49. It is **not walkable**: the
scripted face-the-target-walk-and-jump driver (`rollout_scripted_*.jsonl`, 5,000 decisions) never rose above
**y 6.0** in the main room and entered no rung cylinder at all. The policy's own ascent is a jump/dash/slide
chain — in the 900 decisions after WP1 is credited it is grounded on **7%** of steps and presses jump on
**697** of 900 — and it does reach the height: per-episode peaks inside the main-room box are y 36.1-49.0.
**Three of thirteen episodes entered Floor 2's cylinder (217 steps, 96% grounded)** and `rollout_6` credited
WP2 and then Floor 2 at decisions 463 and 502, about 35 s in.

**Second, independent failure: a parked rung sends the agent backwards.** `_pick`'s fallback is "the nearest
active gate that is neither reached nor parked", with no hop constraint, so a rung the episode skipped stays
a magnet for the rest of the load. In `rollout_6` the boss rung was parked at decision **867** while the
player stood on Floor 2 at (-6.8, 48.9, 303.1); the target became **WP1, 34 m away and 27 m BELOW**
(observation slot 449 read -0.5463, i.e. -27.3 m), the policy went back down, and the level reloaded at
decision 1149 — which is why that episode's `gate_hops_best` reports 3 and not 1. Parks fired in 8 of the 13
episodes; in `rollout_5`, `rollout_7` and `rollout_9` they walked WP2 -> Floor 2 -> Boss and ended with the
**exit** as the target, 187 m away through walls. Parking is on here because `patience_mode: collapsed`
reads the verdict off the GATE ladder (`_layer_gates`), while `prefer_route_when_collapsed: true` means the
level is actually walking its ROOM TRUNK — which `GateProgress`'s own docstring calls a total order on which
parking is "nearly INERT" and a park "can only mislead".

**The exploration archives agree over the whole stage.** All twelve `models/spec_0-3/explore_Level_0-3_*.npz`
together hold 2,516 cells and 521,185 episode-entries. In the main-room column (x -24..16, z 296..336) the
band **y 32..48 is completely empty — 0 cells, 0 entries** — and y 20..32 holds 283. **WP2's own 4 m cell
(1,9,82) has 0 entries and so does its entire 3x3x3 block**; WP1's own cell has 0 and its block 229. The
archive is keyed on the ground point, so this says the player has never had ground under it at the height of
either waypoint, in 5.4M stage steps.

**Live numbers at the time of the probe** (`runs/spec_0-3/episodes.jsonl`, fresh rows since 23,731,594,
n=68): `gate_hops_best` 3 in 62, **2 in ZERO**, 1 in 4; 0 completions; end y median 19.4, share above 40
0.059. `check_run.py` at 24,723,490 steps reads the last 100 fresh as hops {1: 6, 3: 91, 4: 1, 5: 2} — still
**no episode credits WP2**. The stage reaches its 6M cap at ~24.75M.

**Nothing was changed.** No route file, no config, no code beyond adding `probe_rollout.py` and these notes;
the live run was not touched. The proposed fix and what would falsify it are in the handover, not here.

**Not verified.** (1) Why `grounded` reads true through parts of the ascent while `ground_ray_center` reads
the 30 m sentinel — documented for Floor 2's walkway, unexplained on the blocks. (2) Whether the block tops
listed above are wide enough to land on reliably; the 1 m scan settled on only 24 of 161 probes and the
player was soft-dead and sliding for most of it. (3) Whether the 0-3 completion trace's "supported spiral
ramp" reading is right — this pass found no ramp, but it did not re-derive that trace. (4) The probe wedged
the private game once (a `reset` after a death timed out at 300 s, `Responding` false, ~1 s of CPU in 20 s);
it was killed by pid and restarted, and no training game was involved.

## 2026-09-18 (late) — 0-3: parking is switched off on a collapsed-ladder trunk load

**The fourth pass on the 0-3 main-room climb, and the first that changed code.** The handover proposed moving
both inserted waypoints onto measured block tops. **That fix was refused on its own evidence**; a different
one shipped. Branch `route-0-3-climb`, merged `--no-ff`.

**Why the route move was refused.** Three independent reasons, each checked against the generator's own
constants and the 13 recorded rollouts in `runs/probe_0-3/` (89,458 decisions, re-streamed for this pass):

1. **The generator refuses it.** Proposed WP1 `(-3.5, 21.5, 329.5)` is **15.96 m** in 3-D from `2 - Side
   Hallway - Floor 1` `(0, 10, 340)`, under I2's `SEP` of 16.0, *and* trips `co_credit` (horizontal 11.07 m
   against 16, vertical 11.50 m against 12). `apply_inserts` is all-or-nothing (rule 17), so the whole 0-3
   insert list would be withheld and the regenerated trunk would ship **4 rungs**, restoring the 76-degree
   Side-Hallway-to-Floor-2 leg the waypoints exist to remove.
2. **It would make the wrong-side credit worse, not better.** Replaying the proposed cylinder over the
   recordings (inside 8x6 **and** passing `_on_ground`), the proposed WP1 is creditable in **13 of 13**
   episodes and its first creditable step comes **strictly earlier** than the shipped WP1's in **11 of 13** —
   `rollout_4` i=233 against i=2141, `rollout_0` i=645 against i=1506 — and in the same mid-air band
   **y 15.7-17.5**, four to six metres *below* the ledge it is meant to mark. **59 recorded steps** lie inside
   the hallway rung's cylinder *and* the proposed WP1's while passing `_on_ground`: one arrival, two
   instalments, which is exactly what `co_credit` exists to stop.
3. **The WP2 move buys nothing.** Proposed WP2 `(5.0, 39.8, 328.9)` is creditable in the same **2 of 13**
   episodes as the shipped WP2. WP2 is the rung that has never been credited at scale, and moving it there
   changes no episode's outcome.

**The handover's stated mechanism was also wrong.** It claimed `_on_ground` "refuses the credit on every frame
the policy is inside those cylinders". It does not: `_on_ground` **short-circuits on the game's `grounded`
flag before it looks at the ray** (`campaign.py`, and the docstring says so). Measured over the 13 rollouts:
**11,954 of 89,458 decisions (13.4%) read grounded**, and **2,613 of those (21.9%) have the centre ray more
than 8 m below, 681 at the 30 m "nothing hit" sentinel**. The shipped WP1's cylinder holds 1,600 steps of
which only 35 are grounded but **418 pass `_on_ground`** — the credits come from the *drop test* over the
y 10-11 hallway floor, not from a void. The "nothing supports the player under WP1 for 80 m" reading does not
survive that.

**What shipped instead: one condition in `GateProgress.patience_active`.** Parking, the fallback target and
the fallback payment are **off whenever `_prefers_route()` is true** — a load the collapse verdict handed its
room trunk. The two switches were designed against each other and nobody noticed when the second shipped:
`gate_patience_mode: collapsed` takes its verdict from the **gate** ladder, so on 0-3 it turns parking on,
while `prefer_route_when_collapsed` (flipped true 2026-09-17 21:58) makes the same verdict walk the **room
trunk** — and `_note_reached`'s docstring has said since the trunk shipped that parking is "nearly INERT"
there and "can only mislead".

**The harm, measured in game.** 23 parks across the 13 rollouts. `_pick`'s fallback is "the nearest active
gate that is neither reached nor parked" with **no hop constraint**, so it aims backwards and downwards:
recorded parked targets read dy **-2.6, -5.7, -8.4** and, in `rollout_6` at decision 867 — the one episode in
13 that ever stood on 0-3's Floor 2 — the boss rung parked while the player was at `(-6.8, 48.9, 303.1)` and
the target became WP1, **36.6 m away and 27.3 m below** (observation slot 449 read -0.5463). The agent went
back down and the load ended. **And the descent was paid**: 31 decisions collected `gate_approach` while
losing height under a fallback target, **+3.87 in total**. Live over the same window, **84 of 126** fresh
episodes (67%) parked at least once and none completed.

**The credit patience earned on 0-3 does not carry over.** "Patience is what made 0-3 move at all"
(`detect_collapsed_ladder`) was measured under **gates plus patience** — the configuration abandoned on
2026-09-17 21:58, when 0-3 sat at **0 of 75** fresh completions with 8 parks per episode and the flag was
flipped to walk the trunk (`configs/campaign_gates_full.yaml:119`). Nothing measured is being switched off.

**Scope.** Only the collapse-verdict trunk changes: 0-3 and 4-3, and only while `prefer_route_when_collapsed`
is true (every rollback config has it false, where the line cannot fire). A collapsed level with **no** trunk
(1-1, 1-2, 2-3, 8-1) keeps gates plus patience. A **route-fallback** trunk keeps parking, because the detector
grades the trunk itself there and a total order walked from its own top rung is healthy — that last one is a
property of the *data*, not of the rule, and a fallback level whose spawn sat off its top rung would lose
parking too. **No route file, config or reward weight was touched.**

**Tests.** `tests/test_campaign.py` +3, with the `rollout_6` geometry and the shipped 0-3 trunk copied in as
literals (`runs/` is gitignored): the switch itself against all four neighbouring configurations; the
decision-867 scenario, which without the change parks and retargets WP1 and with it keeps the boss rung; and
the fallback-trunk guard. The first two **fail without the one-line change**, verified by removing it. Whole
no-game suite run one file at a time, 30 files — all pass except a **pre-existing, unrelated** failure in
`test_campaign_driver.py::test_the_real_live_state_file_loads_into_the_new_plan_and_keeps_stage_three_running`,
which asserts the **live** `runs/specialists/driver_state.json` is still the pre-kinds Level 0-3 file; the
driver has moved on to `Level 0-1` speed, so its own precondition fires. It fails identically on `main`
without this change.

**Live signal, and the baseline to beat.** Read with `scripts/check_run.py` from
`runs/spec_0-3/episodes.jsonl`, `fresh_start == 1`, timesteps at or above the round-2 relaunch step.
Baselines, all from the round-1 stage: `targets_parked >= 1` in **67%** of fresh episodes (84/126) — this
should go to **0**, and it is the only prediction this change makes directly. Then: `gate_hops_best == 2`
(a WP2 credit) **0 of 126**; `gate_hops_best == 1` **8 of 126**; completions **0**; median end y **17.2**;
share of fresh episodes ending above y 40 **0.063** (0.057 before the waypoints, so that number has never
moved). **This change is not predicted to make WP2 reachable** — WP2 is credited in 2 of 13 recorded episodes
whether or not it fires. What it removes is a measured wrong gradient.

**Revert trigger.** If, 400k steps into the next 0-3 round, `targets_parked` is not ~0, the change did not
take effect — check `_prefers_route()` is true on that load before anything else. If `targets_parked` is 0 but
`gate_hops_best` 3 has fallen below round 1's 114/126 share, or median end y has dropped below 17.2, revert
the one condition and say so here. **Do not stack a route change on top while this is being measured.**

**Activation.** Nothing was bounced. The driver had already ended the 0-3 stage at its 6M cap (24,757,714
steps, recorded `"unfinished"`) and moved to `Level 0-1` speed, so the next 0-3 round picks up the new code
when it starts. No live process, game or port was touched by this pass.

**Not verified.** (1) No live confirmation — this ships on recorded rollouts plus code reading, and 13
episodes on one checkpoint (`ckpt_24181522`) on one game is the whole in-game sample. (2) `build_routes.py`
was **not** run; the SEP/`co_credit`/seed-margin arithmetic above was re-applied from the generator's own
constants, not by executing it. (3) Why `gc.onGround` stays true through parts of the ascent is still
unexplained — the infinite-jump reading is inference from vy resetting to 26.8 six times with the ray at the
sentinel, not from the mod's code path, and it means the "supported/airborne" counts in the previous entry
rest on the flag. (4) The 2026-09-18 "supported spiral ramp" reading of the completion trace was not
re-derived. (5) Whether removing the fallback leaves the agent aiming at an unreachable rung for the rest of
the load — it does, by construction; that is the status quo minus the backwards payments, and it is not
claimed to be better than a reachable rung. (6) The exploration archives were not re-read this pass.

## 2026-09-19 â€” 0-1 speed stage: the regression reversed itself; no rollback, no config change

**What was asked.** Land a one-line stabilisation change (`specialists.yaml` `ent_coef_max: 0.02 -> 0.008`)
and roll `spec_0-1_speed` back to its peak checkpoint. **Neither was done.** Both rest on a premise the data
no longer supports, and the live run was left untouched. No process, port, game or model file was modified.

**The premise.** The stage was reported as eroding monotonically from ~19.27M: fresh completion rate
0.68 -> 0.32, `median_time_50` 312 -> 450-470 s, and a "zero-kill" share (fresh episodes with `kills == 0`,
the wander that never clears the first arena) rising 0.053 -> 0.394. Two independent analyses agreed on that
much. **Both stopped reading at 20.40-20.45M.** The run was at 20.53M when this pass started.

**The measurement.** Fresh episodes from `runs/spec_0-1_speed/episodes.jsonl` (447 total, the whole stage),
100k-step buckets from the trough to now:

| steps | n | rate | zero-kill |
|---|---|---|---|
| 19.90M | 18 | 0.167 | 0.500 |
| 20.00M | 21 | 0.381 | 0.429 |
| 20.10M | 18 | 0.278 | 0.444 |
| 20.20M | 25 | 0.320 | 0.360 |
| 20.30M | 20 | 0.400 | 0.300 |
| 20.40M | 16 | 0.625 | 0.188 |
| 20.50M | 13 | 0.615 | 0.154 |

Trend across those seven buckets: `r(bucket, rate) = +0.889`, `r(bucket, zeroKill) = -0.975`. Pooled,
trough (19.90-20.20M, n=57) against recent (20.40M+, n=29): completion rate **0.281 -> 0.621**, two-proportion
**z=+3.05, p=0.0023**; zero-kill share **0.456 -> 0.172**, **z=-2.59, p=0.0096**. Against the peak region
(19.00-19.30M, n=53) the recovered policy is **not distinguishable**: rate 0.679 vs 0.621 (z=-0.53, p=0.59),
zero-kill 0.075 vs 0.172 (z=+1.34, p=0.18).

The trainer's own trailing-50 bookkeeping confirms it independently, over the last ~250k steps:
`fresh_completion_rate` **0.30 -> 0.58** (the speed rule's `target_rate` is 0.40, so that clause is now MET),
`median_time_50` **434 -> 390 s**, `enemy_visible_frac` **0.180 -> 0.223**, `kills_per_min` **5.00 -> 6.27**,
`look_free_frac` **0.670 -> 0.641**. Every metric cited as a symptom is moving back, in the same direction,
at the same time.

**Confounds checked, and none of them explains it away.** (1) *Not a restart*: the driver log shows trainer
pid 14476 continuous since 2026-09-18 23:31:01 with "0 restart(s) in the last hour" and no traceback;
`timesteps` in `metrics_log.csv` are monotone across the recovery. (2) *Not survivorship in the log tail*:
completions run **5,951-7,337** decisions against **2,090-2,508** for a zero-kill wander, so long completions
are the episodes most likely to still be in flight and unlogged â€” the censoring bias runs AGAINST detecting a
recovery, and the recovery is if anything understated. (3) *Not fewer game recycles*: mean `bridge_resets`
per episode **rose** 1.02 (trough) -> 1.50 (recent) while performance improved, which further undercuts the
relaunched-games theory that both prior analyses had already failed to support.

**Why the config change was refused.** The proposal capped the adaptive entropy floor at 0.008 because
`spec_0-1_speed` ran to 0.01381 (3.45x base) while the two stages that worked peaked at 0.00779 and 0.00586.
The historical objection is that 0.008 prunes into the operating band of a promoted stage (`spec_0-1` used
0.00779 = **97.4%** of the proposed cap; the controller's next raise step, x1.10, is 0.00857, one update
above it). The **live** objection is stronger and is new: at the moment of writing the controller is ramping
again and sits at exactly **0.00779**, and it is ramping *through* the recovery described above â€” rate
0.30 -> 0.58 while `ppo_ent_coef_live` went 0.00400 -> 0.00779. The cap would bind on the controller during
the behaviour that is currently repairing the policy. `ent_coef_max` also lives in `specialists.yaml`'s
`train:` block, which `campaign_driver.stage_config` copies verbatim into **both** stage kinds, so it would
land on every future stage as well. Refused.

**Why the rollback was refused.** The instruction's own justification â€” "the weights have measurably
degraded" â€” is what fails. They degraded and then recovered, to a level statistically indistinguishable from
the peak. The two candidate rollback points are also not better on the metric that broke: `best.zip`
(= `ckpt_19551910`, `at_timesteps` 19,594,390, `median_time_50` 308.18 s, penalty rate 0.5089) sits inside
window E, whose episode-level rate is **0.444** with completion-rate-given-engagement **0.471** â€” i.e. it is
mid-decline, and its attractive median time is conditioned on completing, so it partly reflects "if it does
not finish fast it does not finish at all". `ckpt_19251958` is in the peak region (bucket rate 0.690) but is
still no better than the current policy's trailing-20 (rate 0.700, zero-kill 0.050). A rollback would
discard ~1.3M steps, cost a trainer kill plus up to 50k uncommitted steps, and interrupt a significant
recovery, to land on weights that are not measurably better.

**Why waiting costs nothing.** Verified in code, not assumed: no script in `python/` deletes `ckpt_*_steps.zip`
(`keep_best.py` only globs and `shutil.copy2`s them; `train.py` has no deletion path), so every rollback
candidate stays on disk indefinitely and the rollback remains executable at any later moment. The stage
cannot promote a slow policy by accident â€” `campaign_driver.stage_verdict` requires `median_time_50 <=
target_seconds` (150.00 s) for a SPEED stage and the median is 390 s. And the cap is 8M steps from
`start_steps` 18,052,150, so with the run at ~20.60M there are **~5.5M steps** of headroom. The asymmetry is
one-sided: intervening destroys information that waiting preserves.

**Decision.** Change one thing at a time â€” and the thing that changed is that the policy recovered on its
own. The correct action is to observe, not to intervene on a stale window.

**Live signal, and the baseline at this pass.** Read from `runs/spec_0-1_speed/episodes.jsonl`
(`fresh_start == 1`) and `status.json`; both read-only, no socket on 47800-47811. Baseline recorded
2026-09-19 02:55 at **20,598,550** total steps (2,546,400 into the stage): trailing-50 rate **0.580**,
`median_time_50` **390.115 s**, best ever **156.984 s**, `enemy_visible_frac` **0.223**, `kills_per_min`
**6.27**, `ppo_ent_coef_live` **0.00779**, zero-kill share over the last 20 fresh episodes **0.050**.

PRIMARY metric, unchanged from the proposal because it is the right one: **share of fresh episodes with
`kills == 0`** (equivalently `end_reason == "stuck"` with `level_started == 0`; the two agree to within 0.02
in every 250k bucket). SECONDARY, and the one neither prior analysis tracked: **completion rate conditional
on `kills > 0`**, which is the component that actually turned first (0.684 in 18.95-19.30M, 0.471 in
19.45-19.60M, 0.684 over the last 50).

**Revert trigger â€” i.e. when to stop waiting and roll back after all.** If the zero-kill share exceeds
**0.15** sustained over any 250k-step window, or trailing-50 `fresh_completion_rate` falls below **0.40**
(the speed rule's own `target_rate`) for 250k steps, roll back to **`ckpt_19251958_steps.zip`** â€” the peak
region, not `best.zip`, for the reason given above. Operationally that means: create
`runs/specialists/DRIVER_PAUSE` FIRST, then move every `ckpt_*` newer than the target plus `latest.zip`
aside, because `supervise.choose_resume` picks the file with the MOST steps and would otherwise silently
undo the rollback. Note `models/spec_0-1_speed/latest.zip` is dated Sep 18 08:21, before this stage began at
23:29 â€” it is stale, but its internal step count was NOT opened and should be checked rather than assumed.

**Do NOT drop this guard rail into the watch list**: "`ppo_entropy_loss` must stay inside -6.5..-7.6" was
proposed as a health criterion and is wrong â€” `spec_0-3` spent 457 rows above 8.1 nats (max 9.27) and
`spec_0-2` peaked at 8.23, so it would flag two normal stages as failing. This stage's 8.1275-nat maximum at
19,918,750 is inside `spec_0-3`'s ordinary operating range, and it occurred ~270k steps AFTER the zero-kill
share had already reached 0.30 â€” it follows the failure rather than leading it.

**Still unexplained, and still true.** Nobody has identified what initiated the decline. The zero-kill share
went 0.04 -> 0.17 across 19.50-19.60M with `ent_coef` pinned at exactly base 0.004, and nothing in the PPO
telemetry moves at the turn (`approx_kl` 0.029-0.030, `clip_fraction` 0.17-0.31, `explained_variance`
0.89-0.98 in every window; `target_kl: 0.03` IS set and IS passed through to PPO). The verified
`novelty > 0` clause in `env.py:_campaign_progress`, which re-arms the stuck clock on any first-visit 4 m
cell and let one wander run 4,907 decisions, remains a real efficiency cost â€” zero-kill wanders burned ~20%
of all decisions during the trough â€” but it is an amplifier, not the cause, and it changes episode
termination and therefore the meaning of every windowed metric, so it must not be bundled with a rollback or
a reproduction run. Left for a separate, deliberate pass.

**Not verified.** (1) No in-game evaluation of any checkpoint â€” this is file reading plus arithmetic, and by
the project's own rule an offline probe cannot settle a policy question; the recovery claim rests on live
`status.json` and `episodes.jsonl`, which is the live signal, but no eval was run. (2) Whether the recovery
persists â€” the strongest window is n=29, and the trailing-20 claim is n=20. (3) `latest.zip`'s internal step
count. (4) The cause of the initial decline. (5) Whether the entropy controller's ramp is *causing* the
recovery or merely accompanying it â€” the lagged correlations in both prior analyses say the controller
FOLLOWS behaviour, so the ramp is most likely a response to the same dip, not the repair. No causal claim is
made either way; the point is only that the proposed cap would bind there.

## 2026-09-19 — Where Level 0-1 loses its time: a recorded time budget of the speed stage

**What was done.** `scripts/probe_rollout.py` was extended (the official clock and `timer_running`,
kills/style/restarts, hp/dead, the enemies on screen, arena/door state, and `beh` -- the per-decision diff of
the env's own behaviour counters, which is the only way to read the APPLIED look mode and whether a shot was
on target) and run for 20 episodes on a HAND-STARTED private game on port 47812, against a COPY of
`models/spec_0-1_speed/ckpt_21751558_steps.zip` and the stage's own
`configs/generated/spec_0-1_speed.yaml`: stochastic sampling, fresh starts, Violent, exploration counts read
from `explore_Level_0-1_47800.npz` and never written back. No trainer port was ever connected to. Recordings:
`python/runs/probe_0-1_speed/` (146 MB, 20 files plus `rollout_summary.json`).

**What the 13th game cost the live run.** `mem_guard` was already recycling a game every 7-8 minutes before
the probe started (03:57, 04:05, 04:13, 04:18, 04:25); during the probe the cadence was 4-6 minutes (04:32
47804, 04:36 47802, 04:42 47810, 04:48 47801, 04:52 47808, 05:04 47800) -- about **two extra trainer
recycles**, ~48 s of one worker each. At 04:57 the guard recycled the PRIVATE game, because `--ports 32`
covers 47812 and it was the fattest; that truncated one probe episode as `bridge_reset` and the probe
reconnected to the replacement by itself. Budget for both effects before adding a 13th game to this box.

**The probe reproduces the live run.** 20 episodes, **9 completions (45%)**, median official time **343.3 s**
against the live `status.json` median of 345.5 s (`mean_100.level_seconds` 377.3). Fresh completion rate came
out lower than the live 0.64, on n=20. The ladder is 11 rungs, hops 9..0: `40,1,408` -> `40,1,470` ->
`40,-9,490` -> `40,-9,552` -> `40,11,624` -> `66,21,640` -> `146,31,640` -> `192,31,594` -> `202,56,452` ->
`202,56,432`, exit ground `[202, ~42, 354]`.

**The headline: the path is twice as long, not twice as slow.** A median completion walks **13,362 m in
343.3 s** (38.9 m/s of 3-D path); the 156.98 s best run walks **6,929 m** (44.1 m/s). Median horizontal speed
16.2 m/s against 19.3; under 2 m/s for 20.4% of decisions against 11.5%. So **distance accounts for ~165 s of
the ~186 s gap and movement speed for ~20 s**. Slide is pressed on 55% of decisions, jump 42%, dash 13% --
the policy already chains movement tech, it chains it in the wrong direction.

**The time budget, split by rung credited** (leg h = between crediting rung h and rung h-1; median / best of
9 completions, official-clock seconds; `med_dec` is the median decision count, which is the clock-free
comparison against the best run's own leg lengths):

| leg | median s | best s | gap s | med_dec | best-run dec | what it is |
|---|---|---|---|---|---|---|
| 2 (`192,31,594` -> `202,56,452`) | **150.5** | 54.9 | **95.6** | 2272 | 426 | a 25 m climb in one shaft |
| 3 (`146,31,640` -> `192,31,594`) | 51.4 | 31.1 | 20.3 | 806 | 608 | arena leg; 33 s is a forced fight |
| 6 (`40,-9,552` -> `40,11,624`) | 37.3 | 18.0 | 19.3 | 562 | 171 | travel |
| 9 (`40,1,408` -> `40,1,470`) | 42.9 | 27.2 | 15.7 | 677 | 567 | the first arena; 8 s is fight |
| others (8,7,5,4,1,0,pre) | 61.2 | 35.5 | 25.7 | 952 | 1005 | leg 1 is FASTER than the best run's |
| **total** | **343.3** | **166.7** | **176.6** | 5419 | 2777 | |

The top three legs are **135 s of the 177 s**. Sum-of-leg-bests is 166.7 s, within 10 s of the run's lifetime
best, so the whole gap is inside the policy's existing repertoire -- it is consistency, not capability.

**Leg 2 is one failed climb, repeated.** 82% of its decisions are spent in 5 m cells the episode had already
visited during that same leg, and only **5.0%** set a new closest approach to the target. Height oscillates
between y~7 and y~100 (the rung is at y=56) with a median of **23 up-down cycles** against 8 in the fastest
leg; once the policy reaches y~65 near x~205 it covers the last 92 m and credits the rung in ~14 s, every
single time. Median horizontal distance to the target across the leg is 95 m; 2,983 m travelled for 142 m of
net displacement. No arena is alive for any of it -- this is navigation, not combat.

**Deaths, fights and stalls are NOT where the time goes.** A death costs a 39-80 m setback and a measured
**1.5-8.3 s** re-run (median 4.5 s) at ~0.8 deaths per completion: **~4 s per median run**. Door-gated arena
time is 67.5 s of 343.3 s and is irreducible. Hunting the last arena enemy ("arena alive, none on screen") is
5.8 s. Grounded under 2 m/s is 19.4 s. Splitting the arena legs: leg 9 is 9 s before the first arena enemy,
10 s of fight, then **23 s after the last one dies**; leg 3 is 33 s of fight then 14 s; leg 0 is 8 s then
13 s -- so ~50 s of the run is spent in rooms that are already clear. Aim is poor and cheap: the policy fires
on **74.1%** of decisions and is on target for **18.0%**.

**Why the reward lets it happen.** Over the nine completions' leg 2 (1186 s of recording) the reward parts per
second are `time` -0.300, `gate_approach` +0.162, `gate` +0.114, `checkpoint` +0.076, `kill` +0.043,
`damage_dealt` +0.043, `novelty` +0.035, `punch` -0.030, `damage_taken` -0.015, `death` -0.004 -- **net
+0.123/s**. The one-off payments fall whenever the rung falls, so the MARGINAL rate of one extra second in
the shaft is **-0.23/s**: 100 extra seconds cost 23 reward against a standard deviation of 21 across the
completions' episode rewards. **The single largest time loss in the level is worth about one sigma.**

**And the time-scaled completion bonus cannot reach it.** At `gamma: 0.998` the horizon is 500 decisions =
33 game seconds. Leg 2 begins ~2700 decisions from the end, where the discount is 4.5e-3 (2.7e-5 at the
spawn). Finishing 100 s sooner is worth +13.3 of bonus raw but **+0.06 discounted to the moment the time is
actually lost**, against +9.50 for the per-decision `time` term over the same 1500 decisions -- **159:1**.
The 2026-09-18 reviewer's undiscounted 13.8-vs-71.7 understates the imbalance by an order of magnitude. A
TERMINAL bonus cannot shape a mid-episode dawdle at this gamma; only a per-decision term can, so
`speed.target_scale` is not a lever.

**Three recurring dead ends account for every non-completion** (11 of 20; one of those was the
guard-induced `bridge_reset`): (a) **the level never starts** -- 4 episodes (rollout_0, 4, 14, 15, plus
rollout_5 whose clock ran 5 s) ended in the opening room at z 402-429 with `timer_running` false, no enemy
ever spawning and no door unlocking, burning 115-260 game seconds each; two of them parked on a ledge at
**y=10.5, z~428**. In runs that work the clock starts at decision ~165 at `[37.7, -0.5, 407.6]`, on the
floor. HYPOTHESIS, NOT VERIFIED IN GAME: entering the first hall along that ledge instead of across the floor
skips the volume that starts the level. This is the same zero-kill wander the 2026-09-18 entry tracks
(`kills == 0` in every one of them). (b) **the hops-8 -> hops-7 drop**: rollout_1 and rollout_11 ended at
`[43, -1.2, 487]`, standing 8 m ABOVE the rung at `[40, -9, 490.5]`. (c) **the leg-2 shaft**: rollout_7, 17
and 18 died there, one of them after 649 game seconds. The `novelty > 0` clause in `_campaign_progress`
re-arms the stuck clock on any first-visit 4 m cell, which is why a shaft full of new cells never truncates.

**Proposal (NOT implemented, NOT validated in game).** Raise the per-decision `time` weight for SPEED STAGES
ONLY, **0.02 -> 0.05**. It is the only lever that is (i) per-decision, so it is visible where the time is
lost; (ii) a pure cost, so it creates nothing to farm -- the most it can pay without finishing is 0, and a
policy cannot choose to end an episode (a truncation bootstraps V(s), so triggering `stuck` gains nothing);
(iii) strictly monotone in duration, so faster always beats slower, by 0.76 reward per game second instead of
0.30. Finishing beats not finishing by MORE than today: on this probe a completion is gross +500 ex-time and
the longest stall gross +333, so the gap widens from 260 to 398. The marginal cost of a second in the shaft
moves -0.23 -> -0.70/s, taking 100 s of shaft from 1.1 to 3.3 standard deviations of episode reward. It does
not bias climbing against running -- the term is uniform per second -- and the forced arena fights stay net
positive (`arena_clear` 10 + `door_unlock` 15 against 33 s of fight). Ranked alternatives, both comparable in
magnitude and both less safe: `gate` 15 -> 30 (the same ~+14 of "hurry" pressure, but it raises the value of
collecting rungs WITHOUT finishing, which is the direction this project has been burned by), and `novelty`
0.2 -> 0.0 (leg 2 only moves +0.123 -> +0.088/s; too small alone, though it is also what keeps the stuck
clock re-arming in the shaft). `speed:` in `configs/specialists.yaml` has no `rewards:` override today, so
this needs ~5 lines in `campaign_driver.stage_config` plus a plan value; `tests/test_specialists_config.py`
pins `env.rewards` against `campaign_gates_full.yaml`, so the override must live under `speed:`, never in
`env:`.

**Live signal and revert trigger.** Within 50k steps `status.json.reward_parts_mean_100.time` must move from
~-104 to ~-260 -- that is the check that the config took at all. Then `mean_100.level_seconds` and the
driver's `median official time` (05:45.518 now; S is 02:30.000) at 400k and at 800k steps. Expect <= 320 s at
400k and <= 290 s at 800k with `mean_100.completed` >= 0.5. **Revert to 0.02 if, at two consecutive checks at
least 400k steps apart, `completed` < 0.45 or `level_seconds` is not below 345 s.** One change at a time.

**Not verified.** No policy edit was made or trained; everything above is an offline read of a recording, and
by this project's own rule that cannot settle a policy question. The level-start trigger geometry is a
hypothesis from positions, not from the game's colliders. Per-leg medians are n=9 completions. The best run's
2777 recorded decisions do not reconcile with its 156.98 s official clock at the probe's measured 0.0656 s
per decision (the probe's own clock and decision count agree to 1.6%), so the best run's per-leg SECONDS are
approximate; its per-leg DECISIONS (49 / 567 / 27 / 62 / 171 / 83 / 173 / 608 / 426 / 365, 246 tail) are
exact. `speed.target_scale` and `gamma` were not tested.

## 2026-09-19 -- 0-1 speed: the `time` weight refused, and why the shaft is a geometry problem

**What was asked.** Land ONE reward change on the live `spec_0-1_speed` stage to make it faster: the
per-decision `time` weight `0.02 -> 0.05` (conservative variant 0.04), speed stages only, off the recorded
time budget in the entry above. **Nothing was changed.** No reward weight, config, plan, policy, process,
port or game was touched. The trainer was NOT bounced. This is the second refusal on this run today (see the
`ent_coef_max` entry above) and it has the same shape: the proposal is measured against a premise the run
has already overtaken.

**What reproduced.** Streaming all 20 recordings in `runs/probe_0-1_speed/` independently reproduces the
time budget to the decimal. Per-leg medians over the 9 completions, bucketed by the recording's own
`target_hops`: hops 1 (the `192,31,594 -> 202,56,452` shaft climb) **150.5 s median against a 54.9 s best,
a 95.6 s gap** on 2272 median decisions against 831; hops 2 51.4/31.1; hops 5 37.3/18.0; hops 0 3.9/1.5.
Shaft-leg reward rates per game second, pooled: `time` -0.3024, `gate_approach` +0.1628, `gate` +0.1147,
`checkpoint` +0.0765, `kill` +0.0438, `damage_dealt` +0.0435, `novelty` +0.0349, `punch` -0.0305,
`damage_taken` -0.0152, `death` -0.0042 -> **NET +0.124/s, MARGINAL (recurring only) -0.230/s**. The
discounting finding also holds and is the best thing in the budget: at `gamma` 0.998 the horizon is 500
decisions (~32 s), the shaft ends ~2700 decisions before the end, so a terminal bonus is discounted ~4.7e-3
where the time is actually lost. **`speed.target_scale` is a dead lever** and should not be tried.

**Correction 1 -- the run has not plateaued; it is at its best point of the stage.** The budget was written
against a reading of "350-375 s for the last 1.5M steps". `metrics_log.csv` (697 rows) in 0.25M buckets:
+0.00M 499.9, +0.50M 349.2, +1.25M 312.3, +2.00M **428.6**, +2.50M 357.9, +3.00M 326.4, +3.50M 369.9,
+3.75M 342.7, **+4.00M 294.3, +4.25M 289.1**. That is not a plateau, it is a +/-50 s oscillation on a
~0.5M-step scale, and the current point is the **lowest `median_time_50` the stage has ever recorded**.
Live at 22,355,422 steps (+4.30M into a 9M cap): `median_time_50` **289.12**, `fresh_completion_rate`
**0.82** against a 0.40 bar, `mean_100.completed` 0.71, `level_started` 0.96, `explained_variance` 0.941,
`reward_parts_mean_100.time` -95.0. The only unmet promotion clause is the clock, and the clock is the thing
that is moving.

**Correction 2 -- the proposed evaluation plan could not have judged the change.** Its success bar
("median `level_seconds` <= 320 s at 400k, <= 290 s at 800k") is **already met with no change**, and its
revert trigger ("`level_seconds` not below 345 s") is already satisfied by the status quo by 30 s, so it
could essentially never fire: a harmful change would have been kept. And the control is too noisy for the
window. From `episodes.jsonl` (817 episodes, 443 fresh completions, mean 369.4 s, median 348.0, sd 111.7):
completion throughput is **105 per 1M steps, so 400k steps = 42 completions = ONE 40-completion window**,
and the sd of non-overlapping 40-completion medians across the stage is **36.5 s**, with a median
absolute change of **44.4 s between ADJACENT windows with nothing changed**. A 25-55 s claimed effect read
at 400k is indistinguishable from the run's own oscillation. Roughly 1.5-2M steps per arm would be needed.

**Correction 3 -- two of the supporting numbers are wrong in the proposal's favour.** (a) The "1.1 -> 3.3
sigma" headline divides the new cost by the OLD spread; raising `time` also inflates the spread of episode
return, because episode length varies. Recomputed per episode with the time part rescaled: at 0.02 mean
+374.3 sd 27.7 (100 shaft seconds cost 23.0 = **0.83 sigma**), at 0.04 mean +263.3 sd 44.0 (**1.24 sigma**),
at 0.05 mean +207.8 sd 52.7 (**1.33 sigma**). The real gain is 1.6x, not 3x -- and 0.04 buys 1.24 of the
1.33 for half the return shift. (b) "Finishing beats not finishing by MORE, the gap widens 260 -> 398" is
backwards: it compares the best completion against the worst stall. Worst completion against best
non-completion gives **+117.0 at 0.02, +102.6 at 0.04, +95.4 at 0.05** -- the margin **narrows by 18%**,
because a slow completion is charged more than a short stall. The invariant still holds at every weight
(finishing still strictly beats not finishing, faster still strictly beats slower, the 25..200 band is
untouched), but it holds by less, not more.

**Correction 4, and the reason not to retry this lever later -- the shaft is a DIRECTION problem, and a
uniform cost has no direction.** Inside the shaft leg, the running-best 3-D distance to the rung by decile
(median over the 9 completions) is 103.6, 83.8, 83.6, 83.2, 82.8, 82.8, 82.8, 82.8, 82.3, 29.7 m, while
median player `y` runs 33.5, 34.2, 35.0, 36.4, 38.0, 38.3, 36.6, 50.1, 56.6, 57.2 against a rung at y=56.
So over deciles 2-9 -- 80% of the leg, ~125 s of the 150 s median -- **best-ever distance improves by 1.5 m
in total** while the agent mills at the bottom of a shaft whose rung is 20 m above it. `gate_approach` paid
by quintile: Q1 +91.3 (47.7%), **Q2 +0.6 (0.3%), Q3 +6.9 (3.6%), Q4 +12.1 (6.3%)**, Q5 +80.7 (42.1%). The
directional signal in the stretch that holds the loss is **+0.002/s against a -0.30/s uniform cost**. The
geometry is why: the rung is 142 m away in z and only 25 m up, so climbing the entire shaft at constant x,z
buys 151.5 -> 142.6 m = 8.9 m of "closeness" = 1.34 reward at 0.15/m. `gate_approach` is a bounded potential
that is flat over exactly the move the policy cannot make. Raising a uniform per-decision cost makes the
direction-to-cost ratio inside the shaft **2.5x worse**, not better. `rewards.py` lines 200-204 already say
this in the project's own words, about `punch`: a per-press charge is "a gradient it can actually act on,
unlike a flat per-step cost, which the value baseline absorbs."

**The change would also not have loaded or taken effect as specified.** Three facts, all verified by
reading: (1) `campaign_driver.load_plan` validates the `speed:` block's keys against `StageRule`'s fields
after popping only `targets` and `target_scale`, so a `speed.rewards` block **raises and takes the driver
down**; it needs a pop, a `Plan` field and a merge in `stage_config`, three sites. (2) `write_stage_config`
is called from exactly one place, `start_stage` (line 952), so editing `specialists.yaml` changes nothing
until a stage or round begins -- applying it to the live stage means hand-editing the "GENERATED ... do not
edit" `configs/generated/spec_0-1_speed.yaml` AND bouncing the trainer, which per CLAUDE.md costs up to 50k
steps. (3) `tests/test_specialists_config.py:145` (`test_a_speed_stage_only_adds_the_bonus_switch_to_the_env`)
pins `changed == {"fresh_start_prob"}` and `cfg.rewards == EnvConfig.from_dict(complete).rewards`, and
`stage_config`'s own docstring states the invariant: "A SPEED stage is the same config with `speed_bonus` on
-- no reward weight moves". Landing any reward change on speed stages means deliberately retiring that
invariant, which is a design decision and not a test fix.

**Alternative rejected outright: `gate` 15 -> 30.** `gate` already pays **150.0 per 0-1 completion against a
`level_complete` of ~57.9** -- the ladder pays 2.6x what finishing pays, and doubling it makes that 5.2x.
This project has already been burned by exactly this and the evidence is committed: `routes/rung_overrides.json`,
entry "3 - Side Arena - Floor 1", records 115 of 238 fresh 0-3 episodes (48%) walking a wing for 7 x 15.0 =
105 reward, MORE than the 100.0 a completion paid, with zero fresh completions; seven rungs were dropped to
stop it. Raising `gate` raises the value of collecting rungs WITHOUT finishing. Do not.

**What to do instead.** (1) **Change nothing while the clock is falling.** Re-read `status.json` at +5.0M
and +5.5M. If `median_time_50` is still descending, let the stage run to its cap -- it may promote on its
own; it needs 150 s and it has come 500 -> 289 unaided. (2) **Only if `median_time_50` flattens above ~250 s
for 1.5M steps** is a reward change worth its cost, and then it is `time: 0.02 -> 0.04` (not 0.05), on
branch `speed-stages`, with the three code sites and the retired test above, judged against a baseline
measured immediately before the edit and read at **+1.7M steps, not 400k**. The guard must be
`mean_100.level_started` (0.96 now; revert below 0.85 sustained over 250k steps) because the zero-kill
wander is this run's established failure signature and it leads `completed`; secondary guard
`mean_100.completed < 0.55`, not 0.45. (3) **The mechanism fix is not a reward weight.** The measured defect
is that `gate_approach` is flat across the shaft, so the fix is an intermediate waypoint up the shaft --
`rung_overrides.json`'s `insert_after`, which exists for precisely this ("a leg a room trunk cannot express
... gets waypoints on the real path"). It does not apply yet: `routes/` has 14 trunks and **no
`route_Level_0-1.json`**, 0-1 runs off the door-graph gate ladder, and `prefer_route_when_collapsed` only
fires on a detected-collapsed ladder. A rung also adds +15 of `gate` income to an already 2.6:1 imbalance,
so it needs the `gate`-vs-completion balance looked at in the same spec. That is spec-sized work, and it is
the only proposal on the table that targets the measured cause. The time budget ranked it 7th on the grounds
that it "does not touch the median completion time"; its own data contradicts that -- the shaft leg is 54%
of the median gap.

**Not verified.** No game was started, no socket opened, no test run, no A/B performed; this is file reading
and arithmetic over the same 20 recordings, and by this project's own rule that cannot settle a policy
question -- which is the argument for leaving the live run alone, not for acting on it. n=9 completions
underlies every per-leg median, the same sample as the budget's. The budget's "leg 9" (42.9 s) does not
match the `target_hops == 8` bucket (34.4 s median); its other legs match to 0.1 s and the discrepancy is
unexplained and small. I did not re-derive the per-episode distance figures (my shaft-leg path length
disagrees with the budget's, 5,294 m against 2,983 m, while the whole-episode 13,362 m matches exactly, so
the disagreement is in the leg split; it changes no conclusion). The "level never starts" and "hops-8 drop"
geometry remain hypotheses from positions -- no collider was inspected. Whether arena enemies re-spawn after
a checkpoint respawn (which would let `kill`/`damage_dealt` be re-collected in a stall) was not checked.
System commit was 73% throughout; only short streaming read-only Python processes were run.

## 2026-09-19 -- A 0.0 s "official time" poisoned the 0-2 leaderboard, and the one predicate that ends it

**The bug.** At 18,801,478 steps, env 7 of `spec_0-2_speed` finished a fresh-start Level 0-2 in 4,120
decisions and the completion frame reported `seconds` **0.0** with `restarts` **3**: the mod's campaign block
was read after the game's own level stats had already reset, so the episode's "official time" was the clock of
a level that had just started. Every reader of an official time took it at face value, and the value is the
minimum of everything: it became `campaign.best_time` in `status.json` (a lifetime minimum `ProgressCallback`
also restores across restarts, so no real completion could ever be the best again), it became
`runs/spec_0-2_speed/best_runs/Level_0-2.json` (`save_best_run` keeps the FASTEST, and nothing is faster than
zero), and from there the `post_times.py --watch` helper posted `| 0-2 | 00:00.000 | A |` to `times.md` and
pushed it to GitHub -- over a real `02:07.927 (S)` row, and unreplaceable, because a time is posted only when
it BEATS the row already there. It was also one sample inside `median_time_50`, which is the speed stage's
promotion gate (`median <= target_seconds`), so a few of these could have promoted a slow policy.

**Why the mod cannot say more.** The evidence of the cause is exactly two numbers in that frame -- `seconds`
0.0 beside `restarts` 3 -- which is the stats-reset signature and not a timer that failed to start. The mod
was NOT changed (its DLL is locked while the games run). The env now logs the pair as
`official_time_missing` (level, raw seconds, restarts, steps) in `env_<port>.log` whenever it discards one,
so the next occurrence carries its own evidence.

**The fix, one predicate.** `times.valid_official_seconds(seconds)` returns the time or None, rejecting
missing, non-numeric, non-finite and anything at or under **1.0 s** (`MIN_OFFICIAL_SECONDS`; the fastest human
IL record in the first act is 6.6 s, so the floor cannot touch a real run). Applied at the source in
`env._level_result`, so a completion with no usable clock reports `level_seconds` None and `rank` None, pays
the PLAIN `level_complete` weight (the completion itself is real, and `completion_bonus` already reads "no
time" as plain), and writes no best run at all; and as defence in depth in `progress.py` (the pooled and
per-level `best_time`, the fresh deques behind `median_time_50`, and `_restore`/`_restore_levels`, so a
poisoned `status.json` HEALS at the next trainer start), in `post_times.py` (an invalid best run is never
posted; an invalid leaderboard row reads as NO row, so a real time can replace it), in `times.record` (refuses
an invalid entry outright, and treats an invalid held row and an invalid history predecessor as absent), in
`keep_best.py` (`--metric time` ranks on the lowest `median_time_50` and `--metric campaign` ties on
`best_time`; both now pass the predicate, because `metrics_log.csv` is append-only and still holds rows
written before this fix), and in `campaign_driver.read_sample` (an invalid `median_time_50` reads as "not
measured yet", which the speed rule already refuses to latch on).

**Data repaired.** `times.md`: the leaderboard row restored to `02:07.927 | S | campaign_gates@10.18M |
2026-09-17` and the bogus `00:00.000` history row deleted. The stage's fastest REAL completion, **123.537 s at
19,004,158 steps (env 3, 65 kills, 2 deaths)**, would genuinely beat 127.927 s, but it was NOT posted: its
`best_runs` record had already been overwritten by the 0.0 run and `episodes.jsonl` carries no `style` or
`restarts`, so its rank cannot be established honestly. The fixed watcher will post the next real best by
itself. `best_runs/Level_0-2.json` was renamed to `Level_0-2.json.bogus-0s` BEFORE the bounce so the old
watcher could not re-post it; `status.json`'s 0.0 healed through the restore guard at the next trainer start.

**Swept for the same poison, read-only** (streamed `episodes.jsonl`, fresh-start completions with
`level_seconds <= 1.0`): `spec_0-2_speed` 1 of 50 -- the row above; `spec_0-1_speed` 0 of 1,101; `spec_0-1` 0
of 68; `spec_0-2` 0 of 85; `spec_0-3` 0 of 0; `campaign_gates` 0 of 264. No other `times.md` row is invalid,
and no `models/specialists/*.json` sidecar carries an invalid `best_time` (0-1 243.44, 0-2 139.53, 0-3 null).
`runs/spec_0-2_speed/metrics_log.csv` has **no** polluted `median_time_50` row (the zero never was the median)
but **87 rows with `best_time` 0.0**; `best_time` is only the tie-break of `--metric campaign` and this stage
runs `--metric time`, so the live `keep_best.py` was never misled and was left running.

**Tests.** Nine new assertions across six files, each verified to FAIL with its own guard neutered: env
completion with an official 0.0 (`level_seconds` None, plain bonus, no best-run file), progress ignoring it
for best/median and healing a poisoned `status.json`, `times.record` refusing it and replacing an invalid row,
`post_times` refusing it and replacing an invalid row, `keep_best --metric time` dropping a zero median (and
`--metric campaign` reading a zero `best_time` as no time), and the speed verdict refusing to latch on a zero
median. Two fixtures reported times no real level could produce and were corrected to the range a real run
lives in (`test_progress`'s fake campaign env, `test_campaign_check`'s fake game clock). Whole no-game suite,
one file at a time: 30 files, 0 failures.

## 2026-09-19 — Depth-first stage order (`hold_order: sequential`) and the `END_STAGE` control file

**The user, on the round robin:** *"shouldnt we just work on 0-1 untell its finished before working on the
other levels in that case?"* The hold line's §8a round robin was handing every not-done stage in front of
`hold_before: "Level 0-4"` a fresh 8M-step budget before any of them got a second, so twelve games were being
spread over 0-1 speed, 0-2 speed, 0-3 complete and 0-3 speed in turn — the `campaign_gates` thrash moved from
inside one policy to the schedule. Built on branch `hold-order` (worktree, live run untouched); design in
`docs/superpowers/specs/2026-09-18-speed-stages.md` §10.

**What changed.**
- `hold_order` in `configs/specialists.yaml`, two values. `round_robin` is §8a unchanged and is still the
  CODE default, so every plan and test written before the key means what it meant. `sequential` makes
  `Driver.choose_stage` return the first not-done stage in plan order that can start, round after round,
  until its own rule records it `"done"`. `load_plan` refuses an unknown value (a silent fallback to round
  robin is exactly the failure the key exists to stop). A stage that cannot start at all (`stage_blocked`) is
  passed over only when nothing in front of it can run, and the log names it and the reason.
- **The shipped plan is now level-major**: 0-1 complete, 0-1 speed, 0-2 complete, 0-2 speed, 0-3 complete,
  0-3 speed, then 0-4 and the rest — so "first in plan order" is "the earliest unfinished level".
  `Plan.order` (the distinct levels `full_run.py` plays) and the hold index (6) are unchanged. History is
  matched by `(level, kind)`, so `reconcile` renumbers the indices of the live state file's four entries
  (0-1 complete 0, 0-2 complete 2, 0-3 complete 4, 0-1 speed 1) and the running 0-2 speed stage (index 3)
  without losing, duplicating or renaming anything.
- **`runs/specialists/END_STAGE`**, a control file beside `DRIVER_PAUSE`: on the next poll the driver ends the
  CURRENT stage through `finish_stage` with status `"unfinished"` and `"reason": "ended by operator"`, so
  `refuse_promotion` still refuses to put a speed round's weights over the level's promoted specialist and
  nothing in `models/` is deleted; the file is deleted BEFORE the stage ends (one file, at most one stage) and
  a file that cannot be deleted ends nothing. Its text may name the stage (`Level 0-2 speed`); a name that
  does not match the running stage is refused, logged and deleted. `DRIVER_PAUSE` is checked first.
- `specialists_status.py` prints the order rule beside the waiting list
  (`order rule -- depth-first: finishing Level 0-1 before Level 0-2`), and the waiting list is already in the
  order the stages will be trained in.

**What the order does to the live run.** State at the time: 0-1 complete done, 0-2 complete done, 0-3 complete
unfinished (round 1, 0 completions), 0-1 speed unfinished (round 1: median 500 → 212 s, best 117.5 s, target
150 s), 0-2 speed running (round 1, ~3.9M of 8M, median 306 → ~190 s, target 120 s). Ending 0-2 speed with
`END_STAGE` loses nothing — its weights and optimiser state stay in `models/spec_0-2_speed/` and its next
round resumes from them — and the depth-first rule then runs **0-1 speed round 2** from
`models/spec_0-1_speed/`'s own newest checkpoint until 0-1 is done, then 0-2 speed, then 0-3 complete, then
0-3 speed, and only then 0-4.

**One real bug, found by the scratch dry-run** (the whole reason for running one): PowerShell's
`Set-Content -Encoding utf8` writes a **BOM**, so `END_STAGE` holding exactly what `docs/commands.md`
recommends parsed as the level `"﻿Level 0-2"` and the driver refused a request that was right —
`END_STAGE REFUSED: it names '﻿Level 0-2' and the running stage is Level 0-2 (speed)`. Control files are
now decoded by `decode_control_file` (`utf-8-sig`, then `utf-16` for PowerShell 5.1's `>`/`Out-File`, then
`latin-1`), and the parser strips a stray BOM and NULs. Two tests cover it, one of them end to end.

**Tests.** `tests/test_campaign_driver.py` +11 (the key loaded/defaulted/refused; sequential holding one stage
across rounds where round robin moves on; a blocked stage passed over with the reason logged; today's state
reconciled onto the reordered plan and the whole order walked; seven `END_STAGE` cases). Whole no-game suite
run one file at a time with `PYTHONPATH` on the worktree: **30 files, 0 failures**. Also dry-run from an
isolated scratch directory holding only copies of the plan, the real `driver_state.json` and
`runs/spec_0-2_speed/status.json` — it reported `HOLD LINE before Level 0-4 (depth-first: finishing Level 0-1
before Level 0-2) ... waiting on Level 0-1 (speed), Level 0-2 (speed), Level 0-3 (complete), Level 0-3
(speed)` and left the running stage alone (`Level 0-2 (speed): ok, 4,285,128 steps into the stage, rate 0.900
over 50 fresh, median 199.62 (best 97.65) vs target 120.00`).

### 2026-09-19 — review of the depth-first branch: one real bug, two log fixes, three doc corrections

An adversarial review of `hold-order` (branch at 98ab1a8) could not reproduce any of the four hazards it was
pointed at — index collisions, a history entry matched to the wrong stage, rounds counted wrong, or the ladder
passing the hold line — but found five other things. Fixed before the merge:

- **The real one (major).** `end_stage_now` read `status.json` with `read_sample` directly and never applied
  the `stale_below` filter `tick` applies, so a stage ended by hand in the first minutes of a round >= 2
  recorded the PREVIOUS round's rate, median and best as its own. The window is not the step gap alone:
  `runs/<run>/status.json` keeps the old round's contents until the new trainer's first write, and
  `start_grace_seconds` is 900, so it is ~15 minutes wide at every round boundary. On a COMPLETE stage
  `promote` wrote those numbers into the committed `models/specialists/<level>.json`. Nothing decides on them
  (the hold line reads `status`, `round_init` reads files), so no weights could move and no stage could be
  mis-chosen — the damage was a false record in git. Both readers now go through one `Driver.current_sample`,
  and a filtered sample prints "an unknown number of steps into the stage" instead of the run's whole step
  count as a negative.
- **`log_once` had one key for the whole driver**, so two messages written in the same tick alternated and
  both were re-logged every poll: an `END_STAGE` file the driver cannot unlink (an editor holding it, an ACL)
  meant two lines a minute forever and the "this is stuck" signal lost in them. It takes a `slot` now — one
  per concern, and the running commentary keeps the default.
- **A blocked leading stage was announced once per driver process.** Under `sequential` a stage that cannot
  start is passed over and a LATER level takes the machine for a whole 8M-step round, which is the one thing
  the user's instruction forbids; it now says so for every round the later level takes, and
  `specialists_status.py` prints `BLOCKED (the order passes over it): ...` beside the order rule from the
  driver's own `stage_blocked`, which moved to module level so the report cannot drift from it.
- **Three doc corrections**, all behaviour that was right and described wrongly: a COMPLETE stage ended by
  `END_STAGE` DOES promote its `best.zip` (`refuse_promotion` declines only for a speed stage); under
  `sequential`, ending the stage that is already first in the order starts the next ROUND of that same stage,
  not another level; and "the games are not touched" is too strong — the next stage's `ensure_games` relaunches
  all twelve if a port is not listening, which is why the recipe now says to check `games.py status` first.

**Tests.** +3 in `tests/test_campaign_driver.py` (an `END_STAGE` inside a round-2 stale window records `None`s
and its complete-stage sidecar does too; a control file the driver cannot unlink says so once across three
polls while the stage line is still logged once; a blocked leading stage is named every round and shown by
`specialists_status`). All three were run against the pre-fix code first and failed there. `test_campaign_driver.py`
69 passed, `test_specialists_config.py` 12 passed, whole suite one file at a time: 30 files, 0 failures.

**Activated on the live box, 2026-09-19 19:33-19:36.** Merged as 225a7e8 (`--no-ff`, branch commits 4cbf2e9,
e959f40, 98ab1a8, 1f3936b). `campaign_driver.py --dry-run` first reported the new plan against the live state
(33 stages, hold index 6, `unplanned == []`, `HOLD LINE before Level 0-4 (depth-first: finishing Level 0-1
before Level 0-2)`, current `Level 0-2 (speed)` / `spec_0-2_speed`). Then `DRIVER_PAUSE`; the old driver, its
`start_driver.cmd`, the `spec_0-2_speed` trainer, its twelve workers and the five helpers were stopped one pid
at a time, each re-verified by its command line first (no tree kill; all twelve `ULTRAKILL.exe` survived and
all twelve ports stayed listening); pause removed; `start_driver.cmd` restarted detached. It resumed the SAME
stage from `models/spec_0-2_speed/latest.zip` at 23,292,490 steps, exactly as `tick`'s short circuit on a
non-None `current` promises.

`Set-Content runs\specialists\END_STAGE "Level 0-2 speed"` at 19:34:38; the driver's 19:35:32 poll logged, in
order: `END_STAGE: ending Level 0-2 (speed, round 1) now, at the operator's request` → `STAGE END Level 0-2
(speed): unfinished -- ended by operator (rate 0.000 over 1 fresh, median -, best 97.65, target 120.00,
4,558,092 steps into the stage)` → `NOT promoting Level 0-2 (speed)` → `holding before Level 0-4 (depth-first:
finishing Level 0-1 before Level 0-2)` → `STAGE 2/33 Level 0-1 (speed, round 2): resuming from
models/spec_0-1_speed/latest.zip` at **26,063,110** steps, and the control file was gone. `Level_0-2.zip` is
byte-identical across the switch (SHA-256 `F3C3A29A…`) and `models/spec_0-2_speed/` still holds all 106 files.
**The rate 0.000 over 1 fresh in that history entry is the post-restart window, not round 1's 0.94 / 205.8 s**:
restarting a trainer resets `fresh_window`, and `END_STAGE` fired a minute later. Nothing reads those numbers
(the hold line reads `status`), but the round's real headline lives only here and in `times.md`.

Round 2 verified over the following ten minutes: `spec_0-1_speed` stepping from 26,063,110 (26,070,106 at
19:36:37, 217 steps/s), `stale_below` 26,063,110 so the new fix is doing real work at this very boundary,
helpers up with `keep_best --metric time --min-rate 0.3`, 12/12 ports listening, system commit 77-82%,
`check_run.py` `ALERTS none`, no traceback in the driver log.
69 passed, `test_specialists_config.py` 12 passed, then the whole suite one file at a time.

## 2026-09-20 -- 0-2 speed: the route-potential lever REFUSED, and the plateau is deaths, not navigation

**Outcome: nothing was changed on the live box.** No reward term, no config, no weight, no mod rebuild, no
trainer bounce. The run `spec_0-2_speed` (stage `Level 0-2` speed, round 2) was left stepping untouched on
ports 47800-47811 throughout. This entry is the measurement, and it redirects the stage.

### 1. The time budget of 0-2 (probe, 24 recorded episodes at checkpoint 25,959,958)

Recorded by `scripts/probe_rollout.py` on a private game (port 47812) in an earlier session; raw data in
`runs/probe_0-2_speed/`. 22 completions, official seconds p10 123.8 / median 214.0 / p90 345.0, min 117.0,
max 438.6 -- inside the +-45 s noise band of the live trailing-50 at the same moment, so the probe reproduces
the live run. Ladder: 8 rungs, hops 7..0, then the exit; all 22 completions monotone (9 target transitions, 0
parks), so "leg h = decisions aiming at rung h" is exact. Reference: the run's own best, 86.74 s
(`runs/spec_0-2_speed/best_runs/Level_0-2.json`, 1389 samples, 2915 m raw).

Per-leg mean seconds vs the best run's own leg, with the mean split into fight / hunt / post-clear / nav:

| leg | mean s | best s | gap | fight | hunt | post-clear | nav | deaths/run |
|----|----|----|----|----|----|----|----|----|
| 7 | 2.5 | 4.0 | -1.5 | 0 | 0 | 0 | 5.7 | 0 |
| 6 | 3.1 | 1.5 | +1.6 | 0 | 0 | 0 | 3.8 | 0 |
| 5 | 12.1 | 10.9 | +1.2 | 0 | 0 | 0 | 15.2 | 0 |
| 4 | 29.5 | 15.9 | +13.6 | 20.3 | 0.3 | 9.0 | 5.0 | 0 |
| 3 | 3.6 | 5.6 | -2.0 | 0 | 0 | 0 | 3.5 | 0.36 |
| 2 | 4.0 | 2.4 | +1.6 | 0.4 | 0 | 0.1 | 3.5 | 0.32 |
| 1 | 42.4 | 14.7 | +27.7 | 19.3 | 0.7 | 24.2 | 0.1 | 1.82 |
| 0 | 7.5 | 6.4 | +1.1 | 3.6 | 0.3 | 0.1 | 3.8 | 0 |
| exit | 120.5 | 25.4 | +95.0 | 63.8 | 17.5 | 42.8 | 0.0 | 2.55 |

The exit leg + leg 1 + leg 4 are 136.3 s of the 138.3 s mean gap (98.5%). The six pure-navigation legs
(7/6/5/3/2/0) total 32.8 s against the best run's 30.8 s -- **parity**. Raw movement speed is identical
between the fastest and slowest thirds (16.7 vs 16.5 m/s median; under 2 m/s on 16.6% vs 18.9% of decisions);
the path is 2.7x longer. Same work done either way: kills 51.1 vs 49.6, arena clears 5.0/5.0, doors 11.0/11.0.
**0-2 is not the 0-1 shaft problem.** On 0-1, 54% of the gap sat in one navigation leg where `gate_approach`
was flat; on 0-2 no such leg exists.

### 2. Signal tests, all offline on the 22 completions -- four candidates, all rejected

- **(a) ghost_max** (new maximum of arc along the policy's own best trace): discrimination 2.37 against
  `gate_approach`'s 3.02 -- worse. It is a high-water-mark term, so like `gate_approach` it is silent exactly
  where the dawdle happens, after the maximum is set (pays on 1% of dark decisions at 1.89 m/s).
- **(b) path-distance** instead of straight-line to the rung: median |r| against the ground truth 0.67 vs the
  straight line's 0.68 across the nine legs. No improvement, even though the straight line disagrees in SIGN
  with true route progress on 40% of the exit leg's decisions.
- **(c) inserted waypoints** on the exit leg: would repair the direction signal on the 33-40% of decisions
  where the straight line disagrees -- a navigation problem worth **0.0 s** of that leg's 120.5 s mean, which
  is 63.8 fight + 17.5 hunt + 42.8 post-clear. Also a per-level hand-placed patch.
- **(d) per-leg time budget** (truncate at k x the policy's own best leg time): no k below **15.5** spares the
  completions. k=3 cuts 94% of recorded time and kills 22/22. It breaks "finishing must always beat not
  finishing" outright.

### 3. Route potential ("ghost potential") -- designed, independently re-derived, and REFUSED

The one candidate that lit up offline: a per-decision `B * (u_t - u_{t-1})` on the arc fraction `u` of the
policy's own fastest recorded run, `B = SPEED_BONUS_MIN * level_complete = 25.0`, telescoping so a loop nets
zero. It pays on 74-78% of the decisions where `gate_approach` is dark (73% of all decisions, 3901 s of the
4877 s of recorded completion time) and it is the only tested term with any gradient in the cleared-room time.

**Its safety holds.** Re-derived here from the raw recordings (reference rebuilt from `best_runs`, tracker
re-implemented from the spec, run over all 88,799 decisions): episode total on every completion **24.984 to
24.998**, max running total at any point in any episode **24.9977**, truncated episodes -0.005. No loop,
oscillation, respawn or stall can exceed `B` -- the bound is structural, not empirical. Forward motion pays
`B/L` = 0.0098 reward/m against a time cost of 0.0182 reward/m at the policy's own 19.6 m/s, so no detour
along the reference pays for itself. A death's potential drop is charged and refunded only by re-walking, so
the die-and-re-earn inversion is closed. Finishing beats walking-the-route-and-stalling by ~294.

**Its mechanism does not.** Measured here on the same 88,799 decisions, at the design's own constants
(W = 40 m, D_max = 20 m, jump 15 m, loop 8 m):

- **12.26% of decisions move the arc faster than the player physically moved** (|d arc| > 2*step + 2 m), and
  those decisions carry **82.0% of gross |payment| and 80.8% of net**. On the exit leg -- 51.2% of all
  decisions, 69% of the mean time gap -- it is 86.8% of gross and 84.9% of net.
- **corr(d arc, the player's actual displacement along the route tangent) = 0.222** overall, **0.210** on the
  exit leg.
- **Only 50.3% of the positive payment lands on a decision that actually moved forward** along the route by
  more than 0.5 m. Half the reward pays for the projection switching folds, not for motion.
- The sign is right 90.6% of the time, but the 9.4% of wrong-signed decisions carry **35.2% of gross
  payment** (39.0% on the exit leg): the errors are ~5x larger than the average payment.
- Mean |payment| per decision is **0.0515 -- 2.6x the per-decision `time` cost of 0.020** -- while the net
  contribution is 0.0062. The term pushes **207.7 of gross reward through an episode to deliver 25 net**
  (churn 8.3x).
- The stated per-decision bound in the design (`B*W/L = 0.391`) is **wrong**: windowed decisions reach 0.425
  (the window is on segment-midpoint arc, so the reachable arc runs to W plus half a segment), and the 0.42%
  of decisions that fall back to a global re-sync pay up to **+8.09 / -20.09** -- a third of the whole episode
  budget on one decision, against gate 15 and door_unlock 15 as the largest existing single-step rewards.

**Root cause, geometric and not tunable:** 0-2's reference has **48,174 vertex pairs within 20 m of each other
but more than 40 m apart along the route**, maximum arc separation **847 m**. The exit chain is 904 m of arc
for 109 m of net displacement (path/net 8.3), a U-turn chain that passes within metres of itself. A 1-D arc
coordinate is genuinely ill-defined there; no window width fixes it.

**The obvious repair fails.** Clamping the arc update to the player's own displacement + 2 m removes every
impossible move and caps |payment| at 3.45, but sign accuracy *falls* to 72.6%, wrong-signed decisions still
carry 35.6% of gross, positive payment landing on forward motion stays at 49.1%, and `u_end` drops to a median
0.943 (completions pay 20.9-25.0 instead of a constant 25). It trades spikes for drag and buys no correlation.

**Refused**, therefore: a reward whose per-decision magnitude is 2.6x the clock, whose correlation with the
quantity it claims to measure is 0.22, and 80% of whose payment lands on projection artifacts, on 100% of
decisions -- on a run whose trailing-50 median swung 160 -> 175 -> 236 s inside one session with nothing
changed. It is safe. It is not judgeable, and it is more likely to add gradient variance than direction.
Reference length is also under-specified by the design's own prose: three implementations of "the LAST earlier
vertex within 8 m" gave 2514, 2539.2 and 2550.6 m.

### 4. What the plateau actually is: deaths, and only deaths

Live `episodes.jsonl`, streamed, **1,960 fresh completions** over 18.78M-26.93M steps -- not the 22-episode
probe:

| deaths in the episode | n | share | median s | p10 s |
|----|----|----|----|----|
| 0 | 163 | 8.3% | **123.6** | 105.3 |
| 1-2 | 503 | 25.7% | 149.7 | 118.5 |
| 3-4 | 453 | 23.1% | 192.2 | 148.5 |
| 5-8 | 517 | 26.4% | 253.3 | 181.7 |
| >= 9 | 324 | 16.5% | 378.3 | 265.9 |

`corr(deaths, level_seconds) = +0.863` per episode; OLS **+22.3 s per death with an intercept of 121.1 s**
against an S-rank target of **120 s**. Across the 17 x 500k buckets, `corr(mean deaths, median time) = +0.778`
and +28.6 s per +1 mean death. **The policy already finishes 0-2 at a 123.6 s median when it does not die**,
on 8.3% of its fresh completions. The remaining ~80-110 s of median time is deaths.

**The causation was separated, offline, and it runs death -> time, not slow -> death.** Two independent tests
on the probe recordings:

1. *Pre-treatment pace.* Legs 7/6/5/4 carry **zero deaths in all 22 completions**, and include a full gated
   arena fight, so the time an episode takes over that opening (median 43.2 s, 22% of the median run) is
   measured before any death. `corr(opening seconds, final seconds) = -0.109`; `corr(opening seconds, eventual
   death count) = -0.155`; `partial corr(deaths, total | opening) = +0.862`, unchanged from +0.863. Slow
   episodes are **not** slow before they die. The slowest opening (72.8 s) finished in 147.7 s with 0 deaths;
   a median opening (43.0 s) finished in 387.2 s with 16 deaths.
2. *Removing the recovery.* Taking the UNION of the intervals from each death until the episode regains the
   closest approach to the exit it held before dying: time **inside** recovery is a median 88.0 s (38% of the
   median run); time **outside** recovery is a median **127.3 s** (p10 101.4, p90 153.4) and
   `corr(outside-recovery time, total) = +0.189` -- flat across the entire 117-439 s range. Episode 0 (387.2 s,
   16 deaths) spends only 97.9 s outside recovery, *less* than episode 19 (147.7 s, 0 deaths). 18.6 s of
   recovery per death on the union basis; 85 m of net respawn-to-high-water displacement per death but 1234 m
   actually walked, because deaths cluster and the walk-back is itself a wander.

This **reinterprets the profile's own headline.** "76.0 s per run of post-clear wandering plus 18.5 s of
hunting" was classified by room state, which cannot tell aimless wandering in a cleared room apart from
walking back through a cleared room to where you died. On the two legs holding 86% of the deaths, they are the
same seconds.

**The deaths are three places, not a difficulty.** 111 deaths fall into 7 greedy 15 m clusters; the top three
hold 75%: **39 (35%) at [-135.6, -21.5, 278.6]** on the exit leg at the bottom of the descent, **23 (21%) at
[16.9, -5.7, 245.0]** and **21 (19%) at [16.5, -3.5, 217.5]** on legs 1-3. 93% of deaths occur at hp >= 70 and
**none at all below hp 20** -- the median hp on the last sample before the restart marker is **100**. That is
not combat attrition; it is consistent with instant kills (0-2 ships 44 `DeathZone`s and 5 moving platforms per
`docs/level-survey.md`) or with the damage never being sampled. Deaths y: p10 -24.5, median -10.1 -- both
clusters sit at the bottom of a drop.

### 5. What to do next, in order

1. **Classify the deaths before choosing any lever.** The probe does not sample `player.dead` at frameskip 2
   and reads deaths as `restarts` increments, so cause is unknown. A probe (or a mod field) that records the
   last pre-death hp trajectory and whether a `DeathZone` was the killer decides between two very different
   fixes: a fall/hazard problem at three coordinates, or a combat problem. **Do not pick a lever before this.**
2. **Do not raise the `death` weight blind.** The current marginal price of a death is already ~14 reward
   (5.0 `death` + ~5.6 of `time` over the 18.6 s of recovery + ~3.4 of lost completion bonus). Note the one
   thing the new data does change about the old objection: a death costs mostly *clock*, and standing still
   costs the same clock, so timidity is not free -- but this still needs its own farmability pass, and if the
   deaths are falls then a blanket death penalty teaches caution in fights that are not killing the policy.
3. **Judging, when a lever is finally chosen.** Use the **median of six consecutive 500k-step bucket medians**
   over the 3M steps after activation, against a baseline taken the same way from full buckets only. Measured
   here: the 17 bucket medians have mean 206.8 and **sd 28.6**, while the pooled sigma of completion times is
   ~88 s, so a ~120-completion bucket median has only ~10 s of sampling noise -- policy drift dominates, and
   more completions per window will not help, only longer windows. Never read a partial bucket and never read
   the trailing-50 point median: it read 160.4, 175.5 and 239.1 within one session with nothing changed. The
   3.0M pooled median is 186 s at 24.0M and 203 s at 27.0M; the 5.5M trend is 217 -> 203 s. That is the
   plateau, against a 120 s target.
4. Useful baselines recorded today at 26,864,698 steps, for whoever activates next: fresh completion rate 0.87
   over the last 200 fresh episodes, `level_started` 1.000, **zero-kill share 0.000 across all 17 buckets**
   (0.006 in one), `mean_100` level_seconds 228.6 / deaths 4.56 / kills 48.85, reward parts door_unlock 140.3,
   gate 116.1, gate_approach 75.1, time -68.5, level_complete 62.9, kill 46.8, arena_clear 46.6, death -22.8.

### Live-box safety of this session

Read-only throughout. No socket was opened, no game started or stopped, no trainer or driver process
signalled, no port 47800-47811 touched. All analysis streamed one 29 KB JSON, one 2.6 MB npz and one 1.1 MB
jsonl in short-lived numpy processes (no torch). `mem_guard.py --dry-run` at the start: 12 games, 24.0 GB
total, fattest 2.3 GB against a 2.6 GB limit, system commit 72%. Scratch scripts stayed in the session
scratchpad; the only repo change is this entry.

## 2026-09-20 — 0-2 speed: the deaths are crusher pistons, and a speed stage now prices a death at 12.0

The follow-up to 2026-09-20's "the plateau is deaths, and only deaths". Three questions were open: what kills
the policy, whether it could do anything about it, and what to change. Answered, then ONE change landed.

### 1. What kills it: crusher pistons, and they are pure instant-kill triggers

From `decompiled/Piston.cs` + `decompiled/DeathZone.cs`, read offline, and the 126 deaths recorded in
`python/runs/probe_0-2_speed/` (24 episodes on the spec_0-2_speed checkpoint; analysis in that run's
`analysis/hazards/`):

- 0-2 ships **44 `DeathZone`** objects (the offline parse reproduces `docs/level-survey.md`'s 44 exactly), one
  off-route `HurtZone` and 5 off-route `MovingPlatform`. **16 of the 44** are the head/base plates of 16
  `Crusher Symmetrical` prefabs driven by a `Piston`, all with `notInstakill == false`, so `DeathZone.GotHit`
  calls `pm.GetHurt(999999, …, instablack: true)` — **a kill from any HP**; the `damage = 50` field is dead
  code on these.
- `Piston.Update` runs `timer -= Time.deltaTime * 2`, so `attackTime: 4` / `returnTime: 1` is a **2.5 s cycle**,
  and travel is `MoveTowards(…, dt * 75f)` — 5–8 m in **0.067–0.107 s**. The head's DeathZone is enabled only
  during the outward slam, so the lethal duty cycle is **2.7–4.3%**. There is essentially no solid collider in
  the shaft: what kills is a trigger volume sweeping it in two or three frames.
- **>= 96% of the deaths are instant kills.** 123/126 lost ZERO hp in the preceding second; 117/126 were at
  hp >= 70 with no hp loss in three seconds; 84/126 were at exactly hp 100 on the last live decision. Exactly
  **1/126** is classifiable as accumulated damage.
- **100/126 (79%) died inside a piston's swept head volume** padded by 1 m, 119/126 (94%) within 3 m. Every
  cluster with >= 3 deaths sits on a `Crusher Symmetrical`. Rate check: 126 deaths over 479 s spent inside a
  padded footprint = **0.263 deaths/s**, against the **0.40 slams/s** ceiling a 2.5 s cycle allows. The
  observed rate inside a crusher footprint IS the crusher's own firing rate; nothing else need be invoked.
- The clusters: 34% on the opposed horizontal pair in `9 - Crushers Arena` (the exit leg), 21% on the vertical
  crusher in `6 - Crusher Arena`, 17% on `5 - Crusher Tutorial` (whose piston ships `off: true` but is armed
  25 m earlier by a `ScriptActivator` trigger, so it is live on every pass), 17% on room 9's two verticals,
  6% on `7 - Crusher Hallway`, and 5% on the room-7 wall `Fan` (an inference, see "not verified" below).

### 2. Why the policy cannot dodge, and what it CAN do

Every DeathZone on 0-2 is a trigger. `BuildHorizontalRays`, `BuildGroundRays` and `GroundRayCenter`
(`mod/UltrakillAIBridge/Obs/ObservationBuilder.cs`) all pass `QueryTriggerInteraction.Ignore`, there is no
hazard channel anywhere in the 479 floats, the bridge reports no piston position, phase or hurt-cause, and the
policy is PPO + `MlpPolicy` [512,512] with **no frame stack and no recurrence**. It could not tell a rising
crusher from a falling one if the crusher were visible. **A timed dodge is not learnable at any reward
weight.** What is learnable is fewer passes: expected deaths = **0.115 × crusher-footprint crossings**
(126 deaths / 1,095 crossings; predicted 4.1 for the median episode's 46 crossings, measured 4.6 in the probe
and 4.75 live). The **86.74 s best run** does not route around the crushers — it crosses the same footprints
**19 times for 8.26 s** where the median run crosses **46 times for 17.5 s**, at an *identical* exposure
fraction (8.9% vs 8.1% of play time). The lever is crossing count, and the way to cut it is to go through
once. Waiting cannot help: p(death | crossing) is memoryless w.r.t. anything the policy observes.

### 3. The defect in the reward, including one the design missed

Re-derived from `runs/spec_0-2_speed/episodes.jsonl` (2,392 fresh episodes, 2,068 completions, 18.78M–27.31M):
`seconds = 121.33 + 22.32 × deaths` (r +0.864), `length = 1959.0 + 349.6 × deaths`, 15.63 decisions per
official second, mean **4.75** deaths per completed run, zero-death median 123.2 s, best 86.74 s. Milestones
are NOT re-paid on a respawn — `gates_reached = 8.00 + 0.0000 × deaths`, `checkpoints_level = 2.00 + 0.0001 ×
deaths` — so `mark_paid` / `new_level_load(keep_paid=True)` hold.

**Objective cost of one death: 14.27** (6.99 of `time` over the 349.6 extra decisions, plus 7.28 of
`level_complete` given up by the slower clock; the MARGINAL first death costs 18.52, the fifth 10.37, because
the bonus is hyperbolic). **What the policy felt: about −6.8.** `death` 5.0 and `damage_taken` 1.0 land at the
decision; the time stream discounts to 5.04 at gamma 0.998; the completion bonus arrives ~1,484 decisions
later at **0.998^1484 = 5.1% of face value** (0.37) — the only channel that encodes "this is a speedrun".

**And a death RE-PAYS combat reward.** Streaming all 24 probe rollouts and summing per-step `reward_parts`:
`kill = 28.77 + 4.178 × deaths` (r +0.854), `damage_dealt = 30.25 + 1.994 × deaths` — **+6.17 per death**.
Mechanism, seen directly in `rollout_3.jsonl`: at each restart the game's kill counter rolls back to the
checkpoint value (24→16 at i=777, 18→16 at i=806, 24→16 at i=967), the arena resurrects, and
`new_kills = max(0, cs - ps)` reads 0 on the rollback step and then **pays again for every re-kill** — 133
rollbacks, **2,434 paid kill events against 1,126 final kills** (2.16x). This closes the ledger: −5.0 −1.0
−6.99 −7.28 **+6.17** = −14.10 against the measured episode slope of **−13.72**. At the old weight the
immediate channels of a death (5.0 + 1.0) were *smaller than the re-pay*: **a death paid for itself on
everything the policy feels promptly.** The live log contains the consequence — the three `max_steps`
episodes averaged **24.33 deaths** and a return of **295.4**, MORE than the 140 `stuck` episodes' **200.0**.

### 4. The change

**`speed.rewards.death: 5.0 → 12.0`, speed stages only.** Nothing else moves: no observation, no action space,
no other weight, no mod, no `env.py`, no `rewards.py`.

- `configs/specialists.yaml` gains a `rewards:` block under `speed:`; `campaign_driver.load_plan` pops and
  validates it against `RewardConfig` (a misspelt weight is refused, not defaulted); `stage_config` merges it
  over `env["rewards"]` **inside `if kind == SPEED` only**, by REBINDING — `env` there is a shallow copy, so
  `.update()` would leak the weight into every complete stage generated afterwards from the same `Plan`, and
  comparing the two generated configs would not catch it because both would be the one mutated object.
- **Complete stages are byte-identical**: `Plan.rule_for(COMPLETE)` is untouched and the COMPLETE branch never
  reads `speed_rewards`, so the `campaign_gates_full.yaml` pin still holds unchanged.
- **Sizing.** Felt cost at w: `−(w + 1.0 + 5.04 + 0.37 − 4.64)`. Setting that equal to the average objective
  cost (14.27) gives **12.7**; equal to the marginal (18.52) gives 17.0. **12.0** is the conservative end —
  it prices the death the policy HAS and under-prices the one it will have when it is fast — and buys most of
  the effect for half the added return variance (sd of completion reward 158.7 → ~170 at 12, ~181 at 17).
- **Farm bounds, all pinned in `python/tests/test_speed_death_weight.py` (15 tests, 7 of which fail at 5.0).**
  Never-moving is capped by the stuck rule at 675 decisions = **−13.5** with no other income, at any weight.
  Dying cannot end an episode (`env.py` respawns unconditionally in campaign mode; **0 of 2,392** episodes
  ended `end_reason: "death"`). Camping costs 0.313 per official second and cannot lower p(death | crossing).
  A suicide shortcut never advances the player and the clock runs through it. Faster always beats slower (the
  bonus is strictly decreasing, floor approached never reached). And the per-death slope is negative **even at
  w = 0** once the re-pay is counted in dying's favour. Re-scored at 12.0: mean completion **411.1** vs mean
  non-completion **212.3**; the death loop goes **295.4 → 125.1**, below `stuck`'s **177.2** — the ordering
  inverts at w ≈ 9.5, which is the thing this change actually buys.
- NOT claimed: improved credit assignment. p(death | crossing) = 0.115 means signal 0.115·C and noise
  0.319·√C — **SNR is unchanged at every weight**. This works by making the death channel large relative to
  the arena income that keeps the policy in the shafts, not by making the hazard easier to learn.
- It applies to **every** speed stage, not just 0-2 (`speed:` is per-kind). 0-1's is already done; 0-3's picks
  it up when it first starts. 12.0 was derived at 0-2's constants (T = 120 s, 22.32 s/death, t0 = 121.33 s)
  and the re-pay is level-specific too — it scales with enemies-killed-since-checkpoint.
- Three statements from the design were **corrected before landing** and must not be re-quoted: "dying buys
  almost no combat income (+0.067)" (it buys **+6.17**; the design tested the reported unique-kill field, not
  the `kill` reward); "w=12 is strictly decreasing in k, w=5 is not" (**neither is**, on 2,068 completions —
  both break at k=10→11 on small n; use the structural slope instead); and the 12.25 derivation, which omitted
  the re-pay.

### 5. Baseline, horizon and revert trigger

**Baseline, taken immediately before the switch** — last six FULL 500k buckets, 24.0M–26.5M, fresh episodes,
`level_seconds` filtered through `times.valid_official_seconds`, partial bucket excluded:

| bucket | 24.0M | 24.5M | 25.0M | 25.5M | 26.0M | 26.5M | six-bucket median |
|---|---|---|---|---|---|---|---|
| median official time (s) | 225.9 | 200.1 | 207.3 | 220.5 | 163.4 | 200.5 | **203.9** |
| deaths per completed run | 5.05 | 4.81 | 5.44 | 5.19 | 3.90 | 4.69 | **4.93** |
| fresh completion rate | 0.868 | 0.899 | 0.801 | 0.887 | 0.863 | 0.879 | min **0.801** |

`level_started` 1.0000 and **zero-kill share 0.0000 in all six** (the whole-stage 0.0004 is one early bucket).

**PRIMARY signal** the six-bucket median of official time; **MECHANISM signal** mean deaths per completed run
per bucket — if deaths do not move, the change did nothing and comes out even if time looks flat.
**HORIZON >= 3M steps** (six full buckets) after the contaminated one, ~6 h at 137 steps/s.

**The noise figure in the earlier plan was the wrong one.** Individual bucket medians have sd **29.9 s**, but
the judging statistic — the median of six — has a measured sd of **10.7 s** over 12 rolling windows, and it
**drifts with nothing changed**: 225, 214, 198, 191, 191, 191, 191, 196, 204, 207, 207, 204. A bare "worse
than 203.9" would revert on that drift alone. Restated on 10.7 s: deaths 4.93 → 3.0 predicts −43 s (4.0 sd,
clear); → 3.9 predicts −23 s (2.1 sd, visible); **below ~16 s treat as a null.**

**REVERT TRIGGERS — any one fires:**
1. fresh completion rate **< 0.40** (the stage's own `target_rate`) for two consecutive full 500k buckets;
2. `level_started` share **< 0.85** in any full bucket;
3. zero-kill share **> 0.15** in any full bucket — the "stopped fighting, doors never open" farm;
4. the six-bucket median is **worse than the switch-time baseline by more than 15 s (~1.4 sd)** at two
   consecutive checks;
5. **mechanism null**: after 3M steps, mean deaths per completed run is not below 4.93 AND time has not
   improved — revert rather than leave an inert change in the config.

Watch without a hard trigger: `reward_parts_mean_100/death` should read about −12 × deaths/ep (the one-line
confirmation the weight reached the env), and **`reward_parts_mean_100/kill`** — 43.06 today against 46.81
reported kills, the 2.16x re-pay signature. If the change works by cutting crossings, `kill` should FALL
toward ~27; if deaths fall but `kill` does not, the policy cut deaths by fighting less, which is the timid
failure — watch `arena_clear` (47.0) and `door_unlock` (141.3) with it.

**TO REVERT:** delete the `rewards:` block under `speed:` in `configs/specialists.yaml`; the next round
regenerates `configs/generated/spec_0-2_speed.yaml` without it. No weights are lost.

### 6. A SEPARATE, LARGER defect found in passing — NOT bundled, and next in line

`python/ultrakill_ai/rewards.py`, the `vanished` branch (the "killed in one hit" credit), adds
`e["health"] / max(enemy_max_health.get(e["id"], e["health"]), 1e-3)` with **no clamp on a negative health**,
while the other branch is guarded by `before > e["health"]`. When an overkilled enemy reports negative health
AND its id is missing from `enemy_max_health`, the fallback divisor is that same negative number, `max(...)`
returns **1e-3**, and the term becomes health × 1000. Measured: one step in
`runs/probe_0-2_speed/rollout_11.jsonl` (i=2053) paid `damage_dealt = −250.0` in a single decision, and
**71 of 2,068 completions (3.43%) have a negative TOTAL episode reward, worst −1,119.5**. That is 4–20x the
whole death channel, it is unbounded, and it fires during exactly the arena fighting 0-2 requires. The fix is
a one-line clamp (`max(0.0, e["health"])`) but it is on the SHARED reward path — it changes complete stages
and every level — so it needs its own six-bucket baseline and must not run concurrently with this change, or
neither is interpretable. **Land it next, alone.** Related and much smaller: `completion_bonus` returns the
full unscaled `level_complete` when the official time is missing, so a completion whose timer was lost pays
more than any genuine completion slower than target (one such episode in the live log) — the reward path does
not use `times.valid_official_seconds` the way the leaderboard does.

### 7. Not verified

- **The piston's runtime phase at each death.** The recordings carry no piston state, so "the head was
  mid-slam" is inferred from geometry, the decompiled code and the rate match (0.263 deaths/s inside against
  a 0.40 slams/s ceiling) — never observed. A live probe logging `Piston.transform.localPosition` would
  settle it. No in-game probe was run for any of this work; no socket was opened.
- **Cluster 5** (6 deaths, 5%) attributed to the room-7 wall `Fan` DeathZone at 2.21 m. Nearest hazard
  offline and the kinematics fit (52 m/s, airborne, 148° off heading), but it is an inference.
- **Whether the mod's `pos` is the capsule centre or the feet** — `ground_ray_center` was calibrated at
  ~1.50 m on grounded decisions, but `NewMovement`'s collider offset was not read, so the "inside the swept
  volume" distances carry up to ~1.75 m of vertical uncertainty. Volumes were padded 1 m in every axis; the
  79%/94% figures move if that pad is wrong.
- **`notInstakill == false` is read from the serialized scene only.** `ScriptActivator`, `ObjectActivator`,
  `PlayerActivator`, `Door`, `CheckPoint`, `ActivateArena` and `ActivateNextWave` were checked for writers;
  all ~1,200 game classes were not.
- **Everything behavioural is measured on the probe checkpoint** (recorded 04:27–05:16), not on the live
  weights at 27.3M steps. Live `mean_100` deaths 4.49 and level_seconds 244.6 against the probe's 4.6 and
  190–226 s says it is representative; deaths-per-crossing on the current weights is not directly measured.
- **The effect on the value function's fit is not bounded offline.** 12.0 widens the death term's spread from
  0…−120 to 0…−288 over the observed 0–24 death range. That is what the judging plan has to catch.

## 2026-09-20 — FOCUS on one level: a rung ladder toward the 0-1 record, and the two reward bugs fixed

The user, verbatim: *"can we focuse on one level tell we get it to a point that is close to the speed run
record"*. The lead chose **Level 0-1**: it is the first level, it has the most trained brain in the project
(28.39M cumulative steps across `spec_0-1` and `spec_0-1_speed`), and it is the only level whose speed stage
has ever passed. This entry records what was built, the numbers it was built from, and what is NOT verified.
The design is spec §11 of `docs/superpowers/specs/2026-09-18-speed-stages.md`. Branch `focus-mode`, built in
the worktree `F:\Github\ULTRAKILL-AI-focus` while `spec_0-2_speed` round 2 kept running on the main tree; no
port was opened, no game was touched, and no live file was written.

### The gap the ladder has to close

| | seconds | x the record |
|---|---|---|
| Human INBOUNDS IL record (`configs/il_records.yaml`, geshem8, 2026-05-07) | **19.798** | 1.00x |
| Human Any% IL record (leaves the level; NOT a target) | 4.915 | — |
| `spec_0-1_speed` best single fresh run (2026-09-19) | **81.464** | **4.11x** |
| `spec_0-1_speed` median_time_50 at promotion (2026-09-19) | **147.159** | **7.43x** |
| The S-rank rung it passed | 150.0 | 7.58x |
| Human casual reference playthrough | 146.58 | 7.40x |

The 2026-09-19 round promoted at a 0.92 fresh rate, so the level is being FINISHED reliably; what is left is
entirely the clock. The depth-first order of 2026-09-19 would have sent the machine to 0-2 at that point and
left 0-1 at 7.4x the record, which is what the user's instruction refuses.

### What a focus is

A nullable `focus:` block in `configs/specialists.yaml`:

```yaml
focus:
  level: "Level 0-1"
  targets: [120, 100, 85, 72, 60, 50, 42, 35, 30, 25]
```

While it is set the driver trains **only** that level's speed stage, one RUNG at a time. Rung *k* is an
ordinary speed stage whose `target_seconds` is `targets[k]` exactly — not the S-rank time, not scaled by
`speed.target_scale` — using the same run (`spec_0-1_speed`), the same model directory, and `round_init`'s
own rule for the weights (the stage's own newest checkpoint). It promotes by the unchanged speed rule (fresh
rate ≥ 0.4 **and** `median_time_50` ≤ the rung, latched, then the settle); a met rung is recorded `"done"`
with its target and **overwrites** the specialist, because its median is by construction faster than the one
that was promoted; a round that hits the 8M-step cap is `"unfinished"`, promotes nothing, and repeats the
same rung. Unbounded rounds, as today (`speed.max_rounds: 0` — never idle the machine).

Rung spacing: ~15–20% at the top, where the loss is route length rather than execution (2026-09-19's time
budget: the runs were 2x as long in PATH, not slower in speed, ~16–19 m/s, and the single biggest loss was a
25 m shaft climb), tightening toward the bottom. The last rung, 25 s, is 1.26x the record — "close to" without
pretending a memoryless MLP with no hazard channel and no frame stack will beat a human.

**Which rung is current is derived, never stored.** `focus_rung` walks the ladder and takes the first target
no `"done"` speed round has met; `rung_met` counts a rung met by a done round's own `target_seconds` being at
or under it, or by its recorded `median_time` being at or under it. Nothing migrates: today's history entry
(0-1 speed, round 2, done, target 150.0, median 147.1586145) makes every rung above 148 met and puts the
focus on **120**, which is pinned by a literal-state test. An `"unfinished"` round proves nothing however good
its median looked; an impossible time is rejected by `times.valid_official_seconds`.

**The plan is read ONCE, when the driver process starts.** Editing `focus:` does nothing to a driver that
is already running: it has to be paused, stopped, and started again (no flags -- the state file decides).
A running STAGE is still never interrupted; the focus takes the machine at the next stage boundary, which
`END_STAGE` can bring forward.

`load_plan` refuses a bad block loudly (unknown level, a level with no `kind: speed` entry, empty targets, a
non-numeric/zero/negative target, a ladder that is not strictly decreasing, an unknown key), and
`--start-at` is refused for any other stage while a focus with rungs left is set. When the last rung is done
the focus STOPS, says `FOCUS COMPLETE` once, and the ordinary plan takes over.

### The two reward bugs, fixed together and deliberately

Both were found on 2026-09-20 (previous entry, §6) and both are on the SHARED reward path, so they change
every stage. That was the reason not to land them beside the `death: 12.0` experiment — neither would have
been interpretable. It is acceptable now precisely because the focus starts a new regime: the first rung is a
new baseline, taken after the switch. The `death: 12.0` experiment of earlier today is superseded by this
direction and its judging plan does not apply.

1. **`damage_dealt` had no lower bound** (`rewards.py`, the "killed in one hit" credit). It divided by
   `max(enemy_max_health.get(id, health), 1e-3)`, so an overkilled enemy reporting NEGATIVE health whose id
   was missing from `enemy_max_health` produced `health × 1000`. Measured: `runs/probe_0-2_speed/rollout_11.jsonl`
   i=2053 paid `damage_dealt = −250.0` in one decision (health −0.5 at weight 0.5), and **71 of 2,068 live
   completions (3.43%) ended with a negative TOTAL episode reward, worst −1,119.5** — 4–20x the whole death
   channel. The new `rewards.damage_share` bounds every per-enemy contribution to `[0, 1] × damage_dealt` in
   BOTH credit paths, and falls back to the caller's old divisor only when the enemy's own bar is unknown or
   unusable, so the ordinary case is arithmetically identical to what it always was.
2. **`completion_bonus` paid the full weight when the official time was missing.** A completion with no clock
   paid 100 while every genuine completion slower than the target paid 39–99, so the best-paying completion
   available on a speed stage was one with no clock at all. It now pays the FLOOR (`SPEED_BONUS_MIN ×
   level_complete` = 25) whenever a target is set and the time does not pass `times.valid_official_seconds`.
   A run with no target — every complete stage, Cyber Grind, every older config — is untouched. A
   checkpoint-respawn completion on a speed stage is priced the same way; it cannot arise on a shipped speed
   stage, which forces `fresh_start_prob: 1.0`, and two env tests were updated to pin the new price.

Both keep the three invariants: finishing (25 at worst) still beats not finishing, faster still pays strictly
more than slower, and living still beats dying.

### Verification

All 33 no-game test files, one at a time, with `PYTHONPATH` on the worktree: **0 failures**.
`tests/test_campaign_driver.py` 69 → **83** tests (14 new, including the literal live-state pin);
`tests/test_campaign_rewards.py` 23 → **27** (4 new, all confirmed to FAIL against the pre-fix `rewards.py`);
`tests/test_specialists_config.py` 12 → **14**; `tests/test_campaign_env.py` 108, with two assertions changed
from 100.0 to 25.0 by fix 2; `tests/test_speed_death_weight.py`'s farmability table still green.

A `--dry-run` of the new driver was run from an isolated scratch directory holding copies of the plan, the
real `driver_state.json` and the live `status.json`: it kept `Level 0-2 (speed)` as the stage to resume,
changed nothing, and named the focus and its rung. Quoted in the pull request; no port was opened
(`_port_pids` is netstat only) and every destructive path in `supervise.Supervisor.tick` is `dry_run`-guarded.

### Not verified

- **Nothing here has run in game.** No rung has been trained, no reward change has been measured live, and
  the ladder's spacing is a judgement from the 2026-09-19 time budget, not a fitted curve. The project's own
  rule applies: never judge a policy edit from an offline probe alone.
- **Whether 25 s is reachable at all** by a memoryless MLP with no hazard channel. If a rung turns out to be
  unreachable the round repeats forever (`max_rounds: 0`), which is the deliberate "never idle the machine"
  choice — it must be watched, and the ladder retuned by hand if a rung stalls over several rounds.
- **The effect of fix 1 on the value function's fit** is not bounded offline: it removes a heavy negative
  tail from 3.43% of episodes, which should help, but that is an expectation and not a measurement.
- **`keep_best`'s carried `best.json`** across rungs is argued, not observed: the metric and `penalty_name`
  are unchanged so it will not refuse to start, and a carried `best_at` can only lengthen a settle. No rung
  boundary has actually happened yet.

## 2026-09-20 — FOCUS review: four defects fixed before the branch went anywhere near the live box

A review of the focus branch (b0a5721) found no blocker and six findings. Four were real and are fixed here,
each with a test confirmed to FAIL against the pre-fix code and pass after. All 31 no-game test files pass one
at a time with `PYTHONPATH` on the worktree; `tests/test_campaign_driver.py` 83 → **85**.

1. **The ladder advanced on the round's own target, not on what it measured.** `stage_verdict` latches on the
   FIRST 50-episode window whose rate and median clear the bar and never un-latches, so `"done"` can be
   written on one window and the 300k-step settle can end with the median drifted straight back above the
   target. `rung_met` accepted that entry's `target_seconds` as proof, so the whole ladder would advance on a
   single lucky window — while keep_best (which ranks a 9-sample smoothed median over the append-only
   `metrics_log.csv`) never moved `best.zip`, `promote` re-copied the identical file, and the next rung asked
   100 s of a policy typically at 136 s. With `max_rounds: 0` and no monitors that rung then repeats 8M-step
   rounds for good. 0-1's completions span 81 s to well past its 147 s median, so a transient dip is not
   exotic. **`rung_met` now reads the recorded `median_time` and nothing else**: a second, independent window
   has to agree 300k steps later before the ladder moves, and a round that drifted back simply runs its rung
   again. A `"done"` round with no median recorded meets nothing and repeats — conservative, in the safe
   direction. The reviewer's alternative (require `best_at > start_steps`, i.e. that keep_best moved the file
   during the rung) was **rejected**: a rung whose target the policy already meets would never move `best.zip`
   at all and would then never be recordable as done — a worse deadlock than the bug.
2. **`--start-at` began a focus rung with no rung.** `main()` called `begin_stage` without `rung=`, and
   `start_at_objection` deliberately permits the focus level's own speed stage, so an operator restarting by
   hand after a `held`/`no_checkpoint` exit got `target_seconds = None` → the env reported 0-1's S-rank 150 s
   live → the tick clause accepted it because `stage.rung` was None → the stage would latch (the run's measured
   state is rate 0.92, median 147.16) and promote on a bar the level passed on 2026-09-19. Fixed at the source
   rather than at the call site: **`begin_stage(rung=...)` defaults to `AUTO_RUNG` and asks `rung_for`**, so
   every path that begins a stage gets the same answer and a future call site cannot forget. Explicit
   `rung=None` still means "not a rung".
3. **A focus on an unfinished level exited with 12 games idle.** `focus.level` naming a level whose complete
   stage is not `"done"` loaded cleanly and then returned `(None, [spec])` from `choose_stage` → `"held"` →
   exit 1. The games are children of `mem_guard`, not of the tick loop, so twelve instances would keep burning
   a commit-bound box with no trainer until a human noticed — the one outcome `max_rounds: 0` exists to
   prevent — and `focus.level` is a single plan line the user's own direction invites changing. **The focus
   now runs that level's COMPLETE stage first** (never as a rung) and starts the ladder when it is done; both
   of the focus level's stages are allowed through `start_at_objection` for the same reason. A missing
   specialist FILE still stops the driver — that is a broken tree, not a plan — and the stopping line now
   leads with the focus instead of the irrelevant hold line and says out loud that the games are still running.
4. **The shipped `focus:` block carried no `record_seconds`**, so both driver log sites fell silent and the
   sample in `docs/commands.md` described a line the config could not produce. Added `record_seconds: 19.798`
   and pinned it in `tests/test_specialists_config.py`.

Two findings were accepted as constraints rather than code changes, and are now written down beside the focus
recipe in `docs/commands.md`:

- **A shared-path reward change must not straddle a round that will be judged.** The live trainer holds the
  old `rewards.py` in memory, but the driver re-execs `train.py` on a hang or a crash, so merging mid-round
  silently mixes two reward functions with nothing recording which. The reviewer's suggested order (END_STAGE,
  then merge) is wrong in this particular case: the OLD driver would pick the next stage from the old plan and
  start 0-3 complete. The correct order when the plan itself is changing is end the stage under the NEW driver
  — which is what was done here, on a round that was being cut short anyway and therefore has no verdict.
- **`speed.max_rounds` stays 0 while a focus is set.** A positive cap turns a stalled rung into the same held
  exit: driver gone, games running, nobody watching.

### Not verified (this entry)

- Still nothing in game. No rung has been trained and no reward change has been measured live, so every claim
  about what a rung does in practice — including how often a latched window drifts back — remains offline
  reasoning.
- The `test_speed_death_weight.py` farmability table (6.17 per respawn) was measured under the OLD damage
  clamp and was not recomputed; the file still passes.

## 2026-09-20 — THE FOCUS IS LIVE: the whole machine is on Level 0-1, rung 1 of 10 (120 s)

The user, verbatim: *"can we focuse on one level tell we get it to a point that is close to the speed run
record"*. The lead chose Level 0-1 — the first level, and the fastest brain. This entry records the switch.

### What was ended, and what it did NOT decide

`Level 0-2 (speed)` round 2 was ended by the operator at **29,231,794** steps (started 23,310,382, so
**5,921,412** steps of its 8M budget), recorded `"unfinished" — "ended by operator"`, `promoted: false`.
`models/specialists/Level_0-2.zip` is byte-identical across the switch (SHA-256
`F3C3A29A…7E12B4B8` before and after) and `models/spec_0-2_speed/` still holds all 224 files.

**The death-weight experiment (5.0 → 12.0 at 27,422,782 steps) has NO VERDICT.** It was cut short on the
user's new direction, not on its numbers. What it did get, measured by streaming
`runs/spec_0-2_speed/episodes.jsonl` (3,003 rows across both rounds):

| | episodes | deaths/episode | completions | median | best |
|---|---|---|---|---|---|
| before 27,422,782 (death 5.0) | 2,428 | **4.577** | 2,100 | 199.15 s | 86.74 s |
| after 27,422,782 (death 12.0) | 575 | **4.663** | 502 | 178.05 s | 77.09 s |

Deaths per episode did not fall — it rose slightly, over ~1.8M steps. The completion median fell 185.5 →
178.05 s, which is inside the swing the round already shows without any reward change (six equal step-buckets
across the run: 226.67, 186.31, 192.90, 212.40, 185.53, 178.05 s). Pricing a death at 12.0 is therefore
neither confirmed nor refuted; it stays in the shipped `speed.rewards.death` and the focus rung inherits it.
The last status.json before the switch read rate **0.94** over 50 fresh, median **176.42 s** against the 120 s
target, best **77.09 s**. 122 of the 3,003 rows (4.06%) had a negative total episode reward — the tail the
`damage_share` clamp removes.

**One cost of the ordering, recorded honestly:** the driver was restarted onto the new code ~1 minute before
`END_STAGE`, and a trainer restart deliberately does not carry the fresh-episode windows, so the history row
for that round has `fresh_completion_rate: null`, `fresh_window: 0`, `median_time: null`. The real numbers are
the ones above, from the pre-restart status.json and from episodes.jsonl. Nothing reads those nulls — the row
is `"unfinished"`, so `rung_met` ignores it and `refuse_promotion` had already declined. The alternative
orderings were worse: ending the stage under the OLD driver would have started 0-3 complete under the old
plan, and waiting ~40 minutes for the window to refill would have spent it on a round with no verdict either
way.

### What is live now

`Level 0-1 (speed)` **round 3 = FOCUS rung 1 of 10**, run `spec_0-1_speed`, started 2026-09-20 11:21:44 and
training since 11:23:22. Resumed from `models/spec_0-1_speed/latest.zip` at **28,393,030** steps (the driver
picked it over the highest checkpoint, `ckpt_28362742_steps.zip`). `stale_below` = 28,393,030, so the rung
cannot latch on the previous round's median. Generated config `configs/generated/spec_0-1_speed.yaml`:
`speed_target_seconds: 120.0` (explicit, never scaled — `speed_target_scale` is still 1.0 and unused),
`fresh_start_prob: 1.0`, `rewards.death: 12.0`, `rewards.time: 0.02`, `level_complete: 100`, difficulty 3
(Violent — **the Brutal switch has NOT been made**), `timesteps: 37,393,030` (init + 8M cap + 1M slack).

**The baseline this chase starts from**, all Violent: promoted median **147.16 s**, best single run **81.46 s**
(`times.md` 0-1 01:21.464), fresh rate 0.92. The last 100 fresh episodes of round 2 (steps 28,141,426–
28,392,274) were 93 completions, min 81.5 s, median **145.6 s**. The human INBOUNDS IL record is **19.798 s**
(`configs/il_records.yaml`; the 4.915 s Any% leaves the level and is not the target). So the median is **7.4x**
the record and the best is **4.1x** it. The ladder is 120, 100, 85, 72, 60, 50, 42, 35, 30, 25 — the last rung
is 1.26x the record.

### The restart itself

The old driver (pid 11156), its `start_driver.cmd` wrapper, the `spec_0-2_speed` trainer and its 12 workers,
and the five helpers — 33 processes — were stopped INDIVIDUALLY by pid after `DRIVER_PAUSE` was created; every
`ULTRAKILL.exe` survived (12 alive after each batch, all 12 ports still listening). The pause was then removed
and the driver started detached through `runs/start_driver.cmd`. It resumed 0-2 speed from
`ckpt_29222494_steps.zip` (latest.zip was stale after the kill, exactly as the gotcha says) and was stepping
before `END_STAGE` was written. At the stage boundary the driver found port 47805 not listening and relaunched
all 12 games itself — so the instances are NOT the ones that survived the restart, and the memory leak is
reset with them (system commit 79% → **66%**).

### Verification at the switch

`check_run.py`: **ALERTS none**; trainer running at 223 steps/s, 12/12 games listening, 17.7 GB total, fattest
1.5 GB. Five helpers up, `keep_best.py --run spec_0-1_speed --metric time --min-rate 0.3`.
`specialists_status.py` prints the focus first: the record 00:19.798, "rung 1 of 10: the median must reach
02:00.000 (6.06x the record)", the ladder, and the one rung already done (target 02:30.000, median 02:27.159).

### Not verified (this entry)

- **No rung has been trained.** Every claim about what the 120 s rung does to the policy is still to come.
- The first readings after the switch are noted, not judged: the stale guard means there is no rate and no
  median until the new trainer has produced its own 50 fresh episodes.
- The Brutal (difficulty 4) switch has not been made, and making it mid-ladder would invalidate the baseline
  above. That is a lead decision, not an operator one.
- The 0-2 death-weight experiment's 1.8M steps were NOT re-analysed per-leg; only the aggregate above.

## 2026-09-20 -- Campaign evaluation samples its actions: argmax was scoring 0/5 on the promoted 0-1

**The bug.** `scripts/eval.py` and `scripts/full_run.py` both defaulted to `deterministic=True` -- argmax --
in their `model.predict` calls. Campaign policies are trained, promoted and timed on SAMPLED actions (every
`times.md` row says `training episode (sampled actions)`) and their action heads are deliberately held near
**7 nats** of entropy, so the most likely action is a policy nobody has ever measured.

**Measured 2026-09-20 on a private game** (recordings in `python/runs/probe_0-1_record/`): the promoted
`models/specialists/Level_0-1.zip` completed **19 of 20** sampled episodes and **0 of 5** deterministic ones.
All five argmax episodes ended `stuck`, **two never left the spawn**, argmax picked **look mode 2 on 96%** of
its decisions and hit **0.1% on target**. So every `full_run.py` chain and every `eval.py --record-times` of a
campaign specialist was failing at the spawn, and had been since those scripts were written.

**The fix** (scripts + tests + docs only; the live driver was not touched). One rule, `resolve_deterministic`
in `scripts/eval.py`, imported by `full_run.py`: campaign mode samples unless `--deterministic` is given,
**Cyber Grind keeps its old argmax default** because nothing was measured to say otherwise, and `--stochastic`
still parses -- a no-op in campaign mode, still the way to sample in Cyber Grind -- so documented commands
keep running. Both flags in one command is an argparse error rather than a silent winner. `full_run.play_level`
now defaults to `deterministic=False`, and both scripts print the mode they are about to run in.

**The row has to say which mode produced the time.** `ultrakill_ai.times.actions_note` is the single wording
(`sampled actions` / `deterministic (argmax) actions`); `eval.py`'s note is now
`N/M eval runs completed (sampled actions)` and `full_run.py`'s `full run, one specialist per level (sampled
actions)`. A sampled time and an argmax time are not comparable, and a leaderboard row that does not name its
mode cannot be read later.

**Tests.** New `python/tests/test_eval.py` (7 tests: the `resolve_deterministic` table, the flags, that
`rollout` hands the chosen mode to every `predict` call, and the recorded note) plus four more in
`test_full_run.py` (`play_level`'s default, the flags, the posted note). They fail on the previous commit --
`resolve_deterministic`, `build_parser`, `rollout` and `actions_note` did not exist, and `play_level` defaulted
to True. Whole no-game suite, one file at a time from `python/`: **32 files, 823 named tests, all green**
(commit charge 70% throughout; `mem_guard.py --dry-run` checked before starting).

### Not verified (this entry)

- **No real eval was run**: all twelve games belong to the live driver, so the fix is proven against a stub
  env, `FakeLevel` and a temp copy of `times.md`, never against the game. The 19/20 vs 0/5 numbers above are
  the earlier probe's, not a re-measurement.
- The next chained `full_run.py` on a private game is what confirms the specialists actually complete their
  levels end to end; nothing here proves the chain's total.
- Cyber Grind's argmax default is untouched and unmeasured -- it may well be wrong there too, for the same
  reason, but no one has recorded it.

## 2026-09-20 — Speedrun technique as capability: the spec, and the three code facts that reshaped it

**The instruction**, verbatim: *"make sure the bot knows about the coin rocket jumping and all the other ways
people speedrun these games to make the time go by faster"*. Interpretation agreed by the lead and written
into the spec: **still no recorded human routes and no demonstrations**, but human **knowledge of techniques**
imported as **capability** — actions, observations, and a safe way to make discovery likely.

Spec: `docs/superpowers/specs/2026-09-20-speedrun-tech.md`. **Docs only — nothing live was touched**: no
socket to 47800-47811, no game or driver started or stopped, no mod build, nothing written to `python/models`
or `python/runs`. Research ran against `decompiled/`, the installed `Unity.InputSystem.dll` and
`BepInEx.Preloader.dll` (via the repo's own `.tools/ilspycmd`), and a streamed read of the live
`status.json` / `episodes.jsonl`.

**Three findings reshaped the design after a skeptic pass, and all three were re-verified against the code
before being written down.**

1. **`TrySSJ` OVERWRITES velocity, it does not add** — `rb.velocity = velocityAfterSlide + direction * num5`,
   and `velocityAfterSlide` is written only by `StopSlide`. `Jump()` calls `StopSlide()` *before* `TrySSJ`, so
   the jump variant is safe. **`WallJump` never calls `StopSlide`** and reaches `TrySSJ` through the
   `sliding ||` disjunct — so a one-frame wall-SSJ macro lands on a **stale** `velocityAfterSlide` from the
   previous slide, in that old direction: a wall SSJ at 80 u/s can come out at ~61 u/s pointing elsewhere.
   **A speed loss disguised as a technique.** The wall macro must be two frames, releasing slide on the first.
2. **The SSJ window is real wall-clock time, not game time.** `jumpTimestamp`/`slideTimestamp` come from
   `ctx.time`, which `InputRuntime` sources from the native clock; `Time.captureDeltaTime` does not touch it.
   This **kills the "run at `fixed_fps` 60 / `frameskip` 4 and land bucket 2 for free" fallback** — raising the
   capture rate changes game time per frame, not the real interval between two queued events. Explicit
   timestamps are the only route.
3. **The Input System silently DROPS a state event whose timestamp precedes the device's last**, and
   `Keyboard` has no state callbacks so the drop is unconditional. Queue a jump at `now + 0.012` and the *next*
   frame's ordinary event can carry a lower timestamp and vanish — **eating a whole frame of the agent's
   input**, fleet-wide, looking exactly like a policy regression. `ActionInjector` needs a monotonic timestamp
   cursor. This was missing from the first draft entirely.

**The break got about four times smaller.** Only SSJ and wall-SSJ have windows shorter than one decision;
`core_nuke`, `rocket_down` and `coin_rocket` are ordinary multi-decision sequences (five, three, ~seven) whose
real blockers are variant selection and projectile observations. Both surviving macros fit inside the existing
2-frame step, which deletes the whole frame-scripting apparatus — `obs["frames"]`, `env._frames`, a
`time_scale` in `compute_reward`, frame-denominated `stuck`/`max_steps` — and its test. The principle written
into the spec: **macro what is unreachable TIMING; never macro what is unreachable AIM**, because a mod that
steers the crosshair onto a moving core is auto-aim, not technique knowledge. (The game's own `autoAim` and
`majorAssist` prefs default false and the mod does not patch `GetBool`; that line has been kept cleanly so far.)

**Two corrections to the folklore, from the code.** A plain rocket jump is **damage-free and does not decay** —
`harmlessExplosion` has damage 0, so `rocketExplosion` is never set and the `100/((rocketJumps+3)/3)` ladder is
**dead code for player rockets**; it falls through to a flat 200 launch. And **dash i-frames do protect against
explosion self-damage** — `GetHurt` returns immediately when `invincible && layer == 15`, and `Dodge()` sets
layer 15 while `boostLeft > 0`. "Dash, then detonate" is damage-free by construction.

**The measurement that reordered the plan.** Live at 29.01M steps: median 158.6 s, best 81.5 s, `completed`
0.89, `gates_reached` 9.36/10, `wedged_steps` 0, episode 2,720 decisions. Undiscounted, the clock-sensitive
share of gross positive reward is `(72.68 + 54.41) / 473.39` = **26.8 %**. **Discounted at episode start it is
about 6 %**: at `gamma` 0.998 the horizon is 500 decisions (33 s) against a 2,720-decision episode, so
`0.998^2720 = 0.0043` and a 72.68 completion bonus is worth **0.31** at t=0. At 0.999 it is 4.78; at 0.9995,
18.64. **The agent is not ignoring the clock because the weights are wrong — the finish line is four horizons
away.** Gamma goes first, and in **two steps** rather than one, because `approx_kl` 0.0251 is already close to
`target_kl` 0.03 with `explained_variance` 0.57.

**The entropy alarm in the first draft was backwards.** `EntropyFloorCallback` is one-directional above its
band: over `floor + 1.0` it only decays `ent_coef` back towards `base`, **never below**, and `ent_coef_live` is
already at `base`. Not re-basing `ent_floor` would not "loosen every head" — decaying `ent_coef` makes a policy
*sharper*. It is hygiene, and the number must be **measured** on the migrated model, not guessed. The nats were
also wrong: the widening adds **1.3986**, not 1.502.

**The reward gate was perversely signed** and is now gated on the mod-reported SSJ bucket instead. Because
`TrySSJ` overwrites with `velocityAfterSlide + bonus` and `velocityAfterSlide` floors at 24, a
"speed rose by >= 12" gate **pays from a standstill (24 -> 48.75) and stops paying at 90 u/s** (clamped to
100, under a 102 bar) — it would have rewarded stopping and re-accelerating.

**Also recorded**, each verified: the mod already binds `hook` and `change_fist` and Python never sends them;
`heavy_fall`, `weapon_variation` and `slot_counts` are already on the wire and `pack_observation` throws all
three away; `Punch.BlastCheck()` requires `heldAction.IsPressed()` and `punch` is a TapButton, so **the
Knuckleblaster blast wave can never fire today**; `Railcannon` fires on `WasPerformedThisFrame` while `fire1`
is a HoldButton, so holding it across steps fires once; `add_look_mode.carry_optimizer` pads only along axis 0
and the new first layer grows along axis 1 (it **fails closed** into a silent fresh optimizer, so the caller
must log loudly); `probe_rollout.py`'s `TARGET_SLICE` uses negative indices and will read the wrong slots after
any append; and there is **no `VecNormalize`** in the project's own code, so the usual observation-widening
killer does not exist here.

**Stage list** (S0 measure, S1 gamma 0.999, S2 gamma 0.9995, S3 `level_complete` 300, S4 halve the milestone
pile, S5 sticky slot, S6 private-game verification, **S7 the one break**, S8 tech bonus, S9 variant, S10 own
projectiles + hook, S11 hazard rays, S12 Brutal). S1-S5 need no pause, no DLL and no migration. S4 moved ahead
of the mobility work because more mobility raises the *rate* at which `novelty` pays, and novelty is not
bounded by geometry the way `gate_approach` is.

### Open, needing the lead or the user

1. Two-step gamma ladder instead of the planned single move to 0.9995?
2. Is S4 (halving `gate`, `gate_approach`, `checkpoint`, `door_unlock`, `novelty`) approved as its own round?
   It is the biggest lever on the reward mix and weakens the ladder that reached 0.89 completion.
3. Is the private-game test (S6) approved beside the live fleet — one extra ~1.2 GB game on 47812 via an
   isolated `BepInEx-test` tree, under 30 minutes?
4. Is "macro timing, never macro aim" the right line? It defers `core_nuke` and `coin_rocket` from S7 to S10.
5. Is the ~15-25 minute full pause at S7 acceptable, and wanted at a rung boundary?
6. **`coin_rocket` — the technique the user named — is scheduled last and is honestly low value on 0-1** (little
   verticality; a one-weapon rocket jump does the same job). Kept for the vertical 4-x/5-x levels. Confirm that
   answers the instruction, or say it should be pulled forward.

### Not verified (this entry)

- **Nothing in this design has run in game.** No macro, no observation, no migration has been executed.
- `walkSpeed` and `Time.fixedDeltaTime` are **serialized**, so every u/s figure in the spec is derived
  (750 x 0.008 reproduces the wiki's 16.5 / 49.5 / 24.75 exactly, and the probe's max horizontal speed 99.64
  confirms the 100 clamp), never read.
- Whether an explicit `QueueStateEvent` timestamp reaches `ctx.time` unchanged on Unity 2022.3.29 Mono, and
  what `InputSystem.settings.updateMode` is. S6 exists to settle both; `TrySSJ`'s own `ssjIndicator` pref
  prints the bucket and the exact `+Nu/s`, which turns the question into a readout.
- **The 76 % equipped-slot press rate is from a probe of the PROMOTED `Level_0-1.zip`, not the live policy**,
  and `status.json` carries no slot histogram to corroborate it. S0 exists to re-measure it; the size of S5's
  claimed gain rests on that number.
- The self-damage of a plain (non-`ultrabooster`) beam detonating a core, and whether `SelectVariant1/2/3` have
  default keybindings (fallback: `GunControl.SwitchWeapon` is public).
- **No public text describes the 19.798 s inbounds 0-1 route** — all top-12 run comments are one word or
  moderator notes, and no route was imported from any of them. speedrun.com's prose pages are Cloudflare-403.
- The honest ceiling, stated in the spec rather than buried: the agent travels roughly an order of magnitude
  further than the 195 m straight-line distance to the exit. **These techniques multiply speed along that path;
  they do not shorten it.** Rungs at 120 s and maybe 100 s are in reach; 60/42/30/25 s need a shorter route,
  which a memoryless 512x512 MLP cannot represent, and nothing here addresses that.

## 2026-09-20 — mod v0.8.0 built and verified on a private game (S6/S7 mod half, branch `mod-0.8`)

Built the mod half of `docs/superpowers/specs/2026-09-20-speedrun-tech.md` in worktree
`F:\Github\ULTRAKILL-AI-mod8`, compiled with `-p:InstallPlugin=false` only (0 warnings), and verified it on
**one** hand-started private game on 47812 while the 12-game fleet kept running the installed v0.7.2 DLL.
**Nothing was installed and the live policy's behaviour was not touched.** The Python half (obs packing,
`add_tech_heads.py`, the `tech` reward, the config flips) is not in this branch.

**The isolated test tree.** `<game>\BepInEx-test\` — `core` and `config` **copied** (never moved) from
`BepInEx\`, plus empty `plugins\` and `patchers\`; 1.7 MB. The installed `winhttp.dll` is Unity Doorstop
**4.5.0** and its accepted arguments were read out of the DLL's UTF-16 strings: `--doorstop-enabled` and
`--doorstop-target-assembly` (not Doorstop 3's `--doorstop-target`). The launch line mirrors `games.py`'s
arguments plus `--doorstop-enabled true --doorstop-target-assembly "<game>\BepInEx-test\core\BepInEx.Preloader.dll"`.

- **The path must be quoted as ONE argument.** PowerShell 5.1's `Start-Process -ArgumentList <array>` joins
  without quoting, so `C:\Program Files (x86)\...` split into several arguments; the game started, loaded no
  BepInEx at all, opened no port and exited by itself. Pass a single pre-quoted argument string instead. The
  first attempt cost ~4 minutes and left the fleet untouched (12/12 games and 12/12 ports throughout).
- **The gate that makes the test mean anything:** `hello` reported `mod_version 0.8.0` and BepInEx created
  `BepInEx-test\cache\`, while `BepInEx\LogOutput.log` stayed at its 2026-09-17 mtime and
  `BepInEx\plugins\UltrakillAIBridge\UltrakillAIBridge.dll` stayed at its 2026-09-18 one. Had the override
  been ignored, 47812 would have loaded the **live** plugin and every number below would be a false negative.

**The four questions the spec wrote this stage for, answered.**

1. **Does an explicit `QueueStateEvent` timestamp reach `ctx.time` unchanged? YES, exactly.** A requested
   12 ms slide-to-jump gap measured `dt` **11.999–12.000 ms** over 50 macro trials, and the `slide_timestamp`
   the game recorded came back equal to the anchor the mod queued.
2. **`InputSystem.settings.updateMode` is `ProcessEventsInDynamicUpdate`** — the default dynamic mode, in
   which a future-dated event is processed in the current update. The macro design holds; had it been
   `ProcessEventsInFixedUpdate` the whole thing would have needed redesigning.
3. **The monotonic cursor is both necessary and sufficient.** Right after a macro, the mod's timestamp cursor
   was **ahead of the live clock in 11 of 12 samples** — i.e. without the lift, the next frame's ordinary
   event would have been older than the device's last update and silently dropped, costing a whole frame of
   the agent's input in nearly every case. With the cursor, a dash issued on the step after a macro
   registered **12 of 12**.
4. **The serialized values, read at last:** `walkSpeed` **750**, `jumpPower` **90**, `wallJumpPower` **150**,
   `ssjMaxFrames` **4**, `Time.fixedDeltaTime` **0.008**. These confirm every derived u/s figure in the spec
   exactly: `0.5 x 750 x 2.75 x 3 x 0.008 = 24.75` u/s for a ground SSJ, `37.125` for a wall one.

**Jump SSJ (M1): GO. 25 of 25 landed bucket 1; the control landed 0 of 25.**

| | macro | control (plain slide jump) |
|---|---|---|
| trials | 25 | 25 |
| buckets | `{1: 25}` | 19…522, i.e. all far past the accepted 1–3 |
| landed | **25/25** | **0/25** |
| `TrySSJ` h-gain | mean **+30.39** u/s, median **+24.75**, max +50.89 | **+0.00** u/s, every trial |
| gap `dt` | 12.00 ms | mean 1966 ms |
| whole-step dspeed | mean +7.51, median +4.73 | +0.00 |

The control is measured by the **same instrument**, a prefix/postfix around `TrySSJ` itself, so its +0.00 is
not noise: `TrySSJ` contributes *literally nothing* to an ordinary slide jump, because the slide is still
held and `SlideCancelled` never fires. The median gain of exactly **24.75** is the pure bucket-1 bonus.
**Caveat on the mean:** an SSJ from a standstill reads as +48.75 (`velocityAfterSlide` floors at 24, plus
24.75), so trials that start slow flatter the macro. A first run without a turn-to-open-space step had
`h_speed_before` median **0.0** and reported +45.6; adding the turn dropped it to +30.4 mean / +24.75 median.
**Quote the median.**

**Wall SSJ (M2): NO-GO as a demonstrated technique, and the test found a real defect.**

- The **first** run reported 11 of 25 attempts as `ran` — and **not one reached `TrySSJ`**. Cause: the
  implementation did not require an active slide. `WallJump` only reaches `TrySSJ` through
  `sliding || currentTime - slideTimestamp < 0.032`, and `SlideCancelled` records `slideTimestamp`
  **only `if (sliding)`**. Without a slide the macro fires an ordinary wall jump and reports a success it
  does not have. Fixed: `not_sliding` is now a refusal, so the failure is legible instead of silent.
- With the fix, **0 of 25 attempts ever reached a firing window** on 0-1's opening: only **3 steps** in the
  whole run had the player simultaneously airborne and sliding. Holding `slide` in the air does **not** slide
  — it calls `TryStartSlam` — so an airborne slide needs sliding off a ledge, and jumping out of a slide
  cannot produce one because `Jump()`'s own `if (sliding)` branch calls `StopSlide`.
- **This is a substantive correction to the spec's section 4.2.** M2's precondition is not "a wall plus a
  jump"; it is "a wall plus an **airborne slide**", which is a much rarer state. M2 should be expected to
  refuse most of the time, and gating the `tech` bonus on the reported bucket is what protects the reward
  from paying for the ordinary wall jumps it would otherwise have bought. Whether M2 is worth its action row
  at all is now an open question for the lead. Its row costs nothing if it stays refused.
- Separately, the grace is load-dependent and untested under load: the release is queued at the end of frame
  N and the jump is not read until frame N+2's `Update`, so the interval that must fit inside 32 ms is about
  **two real frames**. On the idle private game that was 4.4–5.6 ms, so the adaptive lead computed 0.0 and
  the grace was never the binding constraint. **On a loaded 12-game fleet a frame is tens of ms and it would
  be.** `macro_wall_lead_frames` exists for that and is unexercised.

**Time accounting: no hole.** Plain steps and macro steps both advanced `frame` by exactly **2** and
`Time.time` by **0.066667 s** at `frameskip` 2 / `fixed_fps` 30. A macro step costs exactly what any other
step costs, so `rewards.py`'s flat per-decision `time` charge and the decision-counted `stuck` / `max_steps`
budgets need no correction. (Measured on `Time.time`, not `StatsManager.seconds`: the level timer is not
running during 0-1's opening and a first attempt read a flat 0.0 that meant nothing.)

**Backward compatibility: exact.** A second connection behaving as a 0.7.2 client — no `macro`, no `variant`,
none of the `obs_*` flags — ran 60 steps and saw **zero unexpected top-level obs keys, zero unexpected
`player` keys and zero missing `player` keys**; step latency median 7.2 ms, p90 10.9 ms. **This only holds on
a game that has never been told about a 0.8 feature:** config is per game **process**, not per connection (as
`frameskip` always has been), so a first run of this check — made *after* the main pass had switched the
blocks on — inherited them and reported `move_tech`, `weapon_tech`, `projectiles` and `input` as unexpected.
The compatibility check now runs first, on a clean game. Worth remembering before anyone concludes a live env
is unaffected by a config that another client sent to the same game.

**Observation blocks: partially verified. Do not read the gaps as working.**

- **`projectiles` (block C) confirmed for coins.** Switching to the Marksman through the new `variant` action
  (`result: ran`) and throwing one gave `{"kind":"coin","dist":6.96,"age":0.20,"rel":[-1.12,5.27,4.39]}` —
  sensible relative coordinates in the player's yaw frame and a real age.
- **`weapon_tech` (block B) reads.** `slot_counts [3,3,3,3,3,0]` (all five standard variants per slot, the
  sixth empty), `variations_in_slot 3`, `gun_ready true`, `coin_charge 400`, `rai_charge 5`,
  `hook_equipped true`, and `rocket_frozen` did go true when Freezeframe was triggered.
- **NOT verified: rockets and grenades in block C.** No `rocket`, `grenade` or `cannonball` entry ever
  appeared, across all five populated slots. Coins prove the list plumbing works, so this is specific to
  `ObjectTracker.grenadeList` and is unexplained — most likely the probe never actually fired those weapons.
- **NOT verified: the slam fields in block A.** `heavy_fall`, `slam_force`, `bounce_window` and
  `pre_slide_speed` all stayed at their resting values because the probe never reached a qualifying fall:
  `TryStartSlam` needs `fallTime > 0.5` **and** nothing within 3 m below **and** `slamCooldown == 0`, which a
  jump on flat ground can never satisfy. The fields read plausibly at rest; nothing confirms they move.
  **Whoever wires block A into `spaces.py` should re-run this against a real slam first** — the slam family
  is, per the spec's section 8, one of the two cheapest real gains in the whole document.

**What this branch leaves for the break.** Every default is the 0.7.2 behaviour: macros only run when an
action asks for one, the three observation blocks are off, `variant` is refused (`variant_switching` false)
and macro values 3–5 are refused as reserved. So S7 is a config flip plus the Python-side migration, exactly
as the spec intends. `python/scripts/macro_check.py` is the test client (standalone, stdlib only, and it
**refuses** to connect to 47800–47811). The test tree stays at
`C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx-test\` for re-runs; it is outside the repo,
is not committed, and does not travel between machines — recreate it with a copy of `BepInEx\core` and
`BepInEx\config` on any machine that needs it.

## 2026-09-20 — GO on the SSJ macro, with M2 cut to reserved: the mod-0.8 review, and the S7 install plan

Review of branch `mod-0.8` at `15481e6` (built and privately verified earlier the same day, entry above),
then the fixes, then the merge. **Source only: the installed DLL was not touched at any point.** The live
plugin `BepInEx\plugins\UltrakillAIBridge\UltrakillAIBridge.dll` is still 73,728 bytes at 2026-09-18 01:34
(v0.7.2); every build in this session used `-p:InstallPlugin=false`, and the csproj's copy target is
`Condition="'$(InstallPlugin)' == 'true' AND Exists(...)"`, so the flag genuinely gates the install.
The record chase on `Level 0-1` (`spec_0-1_speed`, rung 1, control arm) ran untouched throughout: no
connection was ever made to 47800-47811, no `games.py launch` / `stop`, no `supervise.py`.

### VERDICT: **GO** on the macro approach, with one scope cut

**Ship M1 `ssj` at S7. Cut M2 `ssj_wall` to reserved-and-refused**, alongside macro values 3-5, until an
airborne slide is demonstrated on a level that has one. The action row still exists, so enabling it later is
a config flip and not a second break. Two independent reasons, one measured and one derived:

- **It never fired.** 0 of 25 attempts on 0-1 reached a firing window. Only 3 steps in the whole run had the
  player airborne *and* sliding at once — holding slide in the air calls `TryStartSlam`, not `StartSlide`,
  and jumping out of a slide cannot leave one because `Jump()`'s `if (sliding)` branch calls `StopSlide`.
- **Its lead mechanism perturbs the game outside the macro.** See finding 1 below.

### The numbers the GO rests on

Private game, port 47812, isolated `BepInEx-test` tree, `Level 0-1`, `frameskip 2` / `fixed_fps 30`.

| measurement | macro | control |
|---|---|---|
| jump SSJ bucket 1 | **25 / 25** | **0 / 25** (buckets 19..522) |
| `TrySSJ` horizontal gain, instantaneous | median **+24.75 u/s**, mean +30.39, max +50.89 | +0.00 on every trial |
| whole-step `dspeed` | mean +7.51, **median +4.73 u/s** | +0.00 |
| wall SSJ | **0 / 25** ever reached a firing window | — |
| requested vs measured event gap | 12 ms requested, **11.999-12.000 ms** measured over 50 trials | — |
| step length, plain vs macro | both advanced `frame` by exactly 2 and `Time.time` by 0.066667 s | — |
| monotonic cursor | ahead of the live clock in **11 / 12** samples after a macro; dash on the next step registered **12 / 12** | — |
| old-style client on the new DLL | 0 unexpected obs keys, 0 unexpected/missing player keys over 60 steps; step latency median 7.2 ms, p90 10.9 ms | — |

Serialized values read from a live player at last (they cannot be read from decompiled C#): `walkSpeed` 750,
`jumpPower` 90, `wallJumpPower` 150, `ssjMaxFrames` 4, `Time.fixedDeltaTime` 0.008,
`InputSystem.settings.updateMode` `ProcessEventsInDynamicUpdate`. Those reproduce the spec's derived
24.75 / 37.125 u/s exactly, which retires the spec's section 9 caveat that every u/s figure was derived.

**Quote +4.73 u/s, not +24.75, against the speed ceiling.** The +24.75 is the instantaneous delta across the
patched `TrySSJ` call; about five sixths of it does not survive the step. The spec's section 8 ceiling
("~18 -> 25-30 u/s sustained, median ~159 s -> ~105-120 s") was written against the larger number and is
**overstated**; it should be re-derived from the whole-step figure before any rung is judged on it.

### Findings fixed

1. **(major) The wall macro forward-dated `NewMovement.slideTimestamp` past the wall clock with no bound.**
   Confirmed from the game: `SlideCancelled` writes `ctx.time` straight into the public field, so a
   future-dated release event writes a future stamp, and **four** call sites read it — `WallJump`'s
   `currentTime - slideTimestamp < ssjMaxFrames * 0.008` (trivially true while the difference is negative, so
   every plain wall jump takes the SSJ / momentum-reflect branch), `HandleInputs`' enemy-step `windState`
   test at 0.1 s, `Jump`'s dash-jump branch (`jumpTimestamp - slideTimestamp > 0.008 * ssjMaxFrames`, which a
   negative difference fails), and `TrySSJ` itself, where an ordinary jump one or two decisions later lands in
   bucket 1-3 and fires an **unrequested** SSJ that *overwrites* `rb.velocity` with a one-step-old slide
   vector floored at 24 — a speed loss on a step the obs reports as `macro: none`.
   The arithmetic is worse than the reviewer's sweep scenario, because it bites at the **default**: the lead
   M2 needs is `~ 2 x frame_gap - 0.012` and the largest safe lead is `(step_frames + 1) x frame_gap - 0.032`.
   They are compatible only inside a window one frame gap wide, and the default leaves it whenever
   `frame_gap < 0.020` — at a 15 ms gap the needed lead is 18 ms against a 13 ms bound, and the next
   decision's plain jump reads bucket 3. Measured on this box: **4.4-5.6 ms idle, ~30 ms under the 12-game
   fleet**, so both sides of that boundary occur in practice.
   **Fixed twice over.** `ssj_wall` is now reserved and refused by default (`macro_ssj_wall`, the scope cut),
   *and* `WallLead()` computes the safe cap and `BeginMacro` **refuses** with `lead_unsafe` rather than
   queueing an unsafe lead. A private test can override only through `macro_wall_lead_unsafe`. The bound is
   conservative: it ignores Python think time, which only lengthens the real interval.

2. **(major) `move_tech` mixed wall-clock seconds into an otherwise game-time observation.**
   `slide_since` is `InputState.currentTime - nm.slideTimestamp` — real seconds, because the game's own grace
   is measured on the Input System event clock — while every other quantity in the observation is game time,
   which `Time.captureDeltaTime` pins per frame. The same game state would read `~0.08` on the loaded fleet
   and `~0.01` on a single eval game: an order of magnitude of train/eval shift on the feature the spec
   designates as the macro's precondition. `slide_timestamp` and `jump_timestamp` are worse — unbounded
   process-uptime doubles, non-stationary as policy inputs and losing millisecond resolution in float32 after
   a few hours of uptime.
   **Fixed:** new `move_tech.slide_grace`, the fraction of the SSJ window still open (1.0 at release, 0.0 once
   expired), bounded and normalised by the game's own window. That is the field to pack. The three raw fields
   stay as diagnostics and `docs/protocol.md` now says in terms that they must never be packed.

3. **(major) `macro.ssj_bucket` could report a landed SSJ for a REFUSED macro.**
   `BuildMacroReport` read the instrument unconditionally, and `TrySSJ` runs at the end of **every** `Jump()`
   (`decompiled/NewMovement.cs:1653`) and inside `WallJump`'s grace branch (1762). So a refused macro whose
   plain action happened to contain a jump came back as `result:"refused"` beside `ssj_bucket:2,
   ssj_landed:true`. The spec's S8 reward gates on the mod-reported bucket, so this would have **paid for the
   refusal** — training the policy to request macros whose preconditions it cannot meet, the exact failure
   section 4.5 exists to prevent.
   **Fixed:** `ssj` / `ssj_bucket` / `ssj_landed` are populated only when `result == "ran"`. The unconditional
   reading stays in `move_tech.ssj_last`, where it belongs, so a plain slide jump is still measured by the
   same instrument. Documented in `docs/protocol.md`.

4. **(minor) M1's precondition did not match the game's own jump gate; M2 ignored `fakeFallRequests`.**
   `HandleInputs` computes `flag = !falling` and `flag2 = !gc.onGround && (gc.canJump ||
   wcGroup.CheckForEnemyCols())` and calls `Jump()` iff `flag2 || flag` — not `onGround || canJump`, which is
   what the macro tested. And `decompiled/NewMovement.cs:1040` gates the **whole** wall-jump block on
   `!gc.onGround && fakeFallRequests <= 0`, which M2 never checked, so during a fake fall it would report
   `"ran"` while `WallJump` never executed. Both are the class of false success the builder's own M2 fix
   removed.
   **Fixed:** M1 refuses with `no_jump_path` when neither branch would run, and reports `"note":"enemy_step"`
   when the `flag2` branch will run — the macro does succeed there, but that branch also calls
   `EnemyStepResets()`, zeroing the wall-jump and rocket-jump budgets as an unreported side effect of the
   request. M2 refuses with `fake_fall`, on request and again on its second frame.

5. **(minor) Block A allocated a fresh SSJ counters object and a full `ssj_last` object on every step.**
   `BuildLast(-1)` can never return null once the instrument is available, because `Last.Frame` starts at 0
   and `0 <= -1` is false. About 25 extra JTokens per step, ~4k short-lived objects per second at the fleet's
   161 steps/s, on a box that is commit-bound beside games leaking ~790 MB/h. Not a leak; pure waste.
   **Fixed:** both are emitted only when they changed, and are `null` in between. The counters are cumulative,
   so nothing is lost.

6. **(minor) Block C's `rel` is in a different frame from the `enemies` block's `rel`.**
   `TechObserver` uses `nm.transform.InverseTransformPoint`, and `CameraController` sets the player
   transform's rotation from `rotationY` alone, so that is a **yaw** frame; `ObservationBuilder.cs:283` builds
   the enemy block's `rel` with `cam.InverseTransformPoint`, which **includes pitch**. Two frames now coexist
   in one observation. Whoever packs block C at S10 by copying the enemy packer gets rocket and coin
   directions wrong by exactly the current pitch — and the coin rocket jump is a look-up-then-detonate
   technique. **Fixed in `docs/protocol.md`,** stated next to both blocks. No code change: the yaw frame is
   the right one for block C.

7. **(minor) A dead SSJ instrument would silently pay zero across the whole fleet.**
   `Plugin.Awake` isolates `PatchAll(typeof(MovementPatches))` in its own try/catch, which is correct — a game
   update renaming the private `TrySSJ` must cost the instrument and not the bridge. But `BuildLast` then
   returns null forever, every macro reports `ssj_bucket: -1`, and an S8 reward keyed on the bucket turns off
   across 12 games while training continues and the median quietly stops improving. **Fixed as a documented
   client obligation:** `docs/protocol.md` now requires a client enabling such a reward to assert
   `ssj_instrument` and `macro.ssj` are in `hello.features` at connect and fail the worker loudly. The Python
   side of that assertion is S7/S8 work and is **not** on this branch.

8. **(minor) The `identical_to_0_7_2` claim was weaker than its name.** It is a key-set comparison over 60
   steady-state `step` replies against a hand-typed baseline: no value comparison, no run against an actual
   0.7.2 DLL, and no coverage of reset, `get_obs`, episode end or a scene change. **Fixed in
   `python/scripts/macro_check.py`:** renamed to `key_sets_identical_to_0_7_2`, now also checks for **missing**
   required top-level obs keys and counts distinct key sets, and carries a comment naming what it does not
   prove. The real check — a field-by-field diff of a fixed action script against the live 0.7.2 game and
   against the test tree — is now step 4 of the install plan below.

9. **(minor) The SSJ control arm measured a different input pattern from the one the policy produces.**
   The control jumped while still **holding** slide, so `SlideCancelled` never fires. Today's policy instead
   drops slide and adds jump in the *same* step, putting release and press in one `KeyboardState` event where
   `TrySSJ` returns at `if (!(num > 0.0)) return;`. Both fail, so the GO survives, but the A/B measured was
   "release slide + SSJ" versus "keep sliding + ordinary jump". **Fixed in `macro_check.py`:** three arms now
   — `macro`, `hold` and `release` — with `release` as the honest baseline, and the whole-step median reported
   beside the instantaneous delta. **The third arm has not been run in a game.**

10. **(minor) The legacy path's cursor behaviour was reasoned about, not measured.** Traced and agreed: the
    first event of each frame resets `lastQueuedTime` to `now`, so a lift is bounded at one 0.5 ms epsilon per
    frame and cannot accumulate. **Added to `macro_check.py`** as an explicit assertion — N legacy steps with
    `obs_input_clock` on, recording how often `cursor > now` and the maximum lift, expecting zero or
    <= 0.0005. **Not yet run.**

11. **(minor) `CLAUDE.md` spent its last headroom on per-branch history.** Collapsed the four-line `mod-0.8`
    bullet to one line linking here. **217 lines** (main was 215, the branch had taken it to 219).

### Findings flagged and NOT changed

12. **The three pushed commits carry `Co-Authored-By: Claude Opus 5 (1M context)`, not the
    `Claude Fable 5.1` the task specified.** Not rewritten: the branch is pushed, another engineer may have
    fetched it, and a force-push to fix a trailer is not worth the hazard. The merge commit uses the same
    trailer as the three it merges, so the history is at least internally consistent. **Lead decides** whether
    the convention changes going forward.

Also **not** changed: S4 (halving the route rewards) stays unapproved per the lead — an earlier measurement
said cutting gate pay lowers the speed gradient, and it needs its own evidence after S1-S3. Nothing in this
branch touches rewards.

### Backward compatibility — why merging this cannot change the live policy

The branch touches **no file under `python/ultrakill_ai/`**; the only Python file it adds is
`python/scripts/macro_check.py`, a standalone test client that hard-refuses ports 47800-47811.
`python/ultrakill_ai/spaces.py:44-57` emits only `move` / `buttons` / `slot` / `look`, and the strings
`"macro"` and `"variant"` appear nowhere in the package — so on every live step `ParseMacro(null)` returns
`None`, `ApplyVariant(null)` returns null, `BeginMacro` returns immediately, and `QueueAt(QueueOpts.None,
-1.0)` reduces to the old `Queue()` plus a timestamp decision that takes the pre-0.8 `time = -1` call whenever
the clock advanced. `BuildStepObs` adds `macro` only when one was requested, `variant` only when non-null,
`input` only behind `obs_input_clock`, and `BuildTech` returns immediately when all three flags are false —
all default false. **Merging the source cannot change behaviour. Only the S7 DLL install can.**

Lockstep safety re-checked: no macro path blocks, sleeps or loops; macro state is cleared at the head of every
`SetAction` and by `BeginReset`, so a macro cannot span two steps or survive a reset; `ReleaseControl ->
Detach()` makes `ApplyFrame` a no-op and the next `Attach()` calls `Clear()`, so a client disconnecting
mid-macro leaves nothing held.

### The S7 full-pause install, step by step

Downtime **15-25 minutes**, at a **rung boundary**, not mid-rung (lead-approved). The relaunch also resets the
games' memory leak.

1. `New-Item runs\specialists\DRIVER_PAUSE` **first**. A deliberate Ctrl+C is indistinguishable from a crash
   and the driver will relaunch the games and the trainer underneath you.
2. Write `END_STAGE` and let the round in flight record "unfinished — ended by operator", so no judged round
   straddles the change (the 2026-09-20 lesson).
3. Stop the driver, the trainer and the workers **by pid**; then `games.py stop` — safe only because no run is
   live by this point. The live DLL is locked while games run.
4. **Before installing anything**, settle finding 8: run one fixed action script against the live 0.7.2 game
   on 47812 and the same script against the `BepInEx-test` 0.8.0 tree, and diff the two obs streams field by
   field — including a `reset` and an episode end, not just steady-state steps. Exclude the known
   non-deterministic fields. This is the only check that would catch a changed *value* or a changed
   reset/settle cadence, which the key-set check cannot.
5. `dotnet build -c Release` — this one installs. ~30 s.
6. Run `add_tech_heads.py`; confirm the printed `mean_kl` is **0.0**, not merely small; read the measured
   entropy total and write it into the config as the new `ent_floor`; quarantine the 479-wide checkpoints
   under `pre_tech/` **and** add the `train.py` resume-path shape guard, because the driver's round init would
   otherwise happily pick a 479-wide `ckpt_*_steps.zip` and re-exec `train.py` in a loop across 12 games.
7. Edit `configs/specialists.yaml`: the new `ent_floor`, `obs_move_tech: true`, the `tech` block. Leave
   `macro_ssj_wall`, `allow_reserved_macros`, `variant_switching`, `obs_weapon_tech` and `obs_projectiles`
   **false** — that is what makes the migration behaviour-preserving.
8. Remove `DRIVER_PAUSE`; start the driver through `runs\start_driver.cmd`.
9. **Verify old-client behaviour on the fleet BEFORE any Python break lands**: `scripts/check_run.py`, expect
   ALERTS none, 12/12 listening, five helpers up including `mem_guard.py`, and step latency in line with the
   7.2 ms median / 10.9 ms p90 measured on the private game.

### What is NOT verified

- **Every in-game number above is the builder's, from the private 47812 game.** This review connected to no
  bridge, ran no game and ran no Python test file; it verified that the instrument reporting those numbers
  computes byte-for-byte what the game computes (`(int)((jumpTimestamp - slideTimestamp) / 0.00800000037997961)`
  on the game's own two public doubles, one line before the game decides), and nothing more.
- **None of this session's fixes has been run in a game.** They compile (`dotnet build -c Release
  -p:InstallPlugin=false -t:Rebuild`, **0 warnings 0 errors**) and `macro_check.py` byte-compiles, but the new
  refusal reasons (`no_jump_path`, `fake_fall`, `lead_unsafe`), the `note` field, `slide_grace`, the
  change-only SSJ blocks and the three-arm control have **not** been exercised against a running game. The
  next private-game run must re-measure the jump SSJ arm, because finding 4 changed M1's precondition and
  could move the 25/25.
- **The `lead_unsafe` bound is derived, never measured.** M2 is off by default, so it is untested code on a
  path that cannot run.
- **The macros have never been run under fleet load.** Every measurement is from one game on an idle-ish box
  at a 4.4-5.6 ms frame gap; the fleet's is ~30 ms.
- Not verified independently: `InputSystem.settings.updateMode`, `walkSpeed` 750, `fixedDeltaTime` 0.008, the
  11/12 cursor result, the time-accounting result, `key_sets_identical_to_0_7_2`, and the contents of
  `BepInEx-test\core`.
- **Not on this branch and not written:** the S7 checkpoint quarantine, the `train.py` shape guard, the
  `probe_rollout.py` absolute-index fix (its `TARGET_SLICE` uses negative indices from the end and will
  quietly read the wrong slots after any append) and `add_tech_heads.py`.
- **Block C rockets/grenades and block A's slam fields remain unconfirmed**, per the previous entry.

### Where the test tree lives

`C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx-test\` — `core` and `config` copied from
`BepInEx\`, empty `plugins\` and `patchers\`, ~1.7 MB, left in place. It is a **sibling** of `BepInEx\`, not
inside it, so neither the live chainloader nor a live launch can reach the test DLL; `doorstop_config.ini` is
untouched (mtime 2026-02-08 01:50, `target_assembly=BepInEx\core\BepInEx.Preloader.dll`). `winhttp.dll` is
Unity Doorstop 4.5.0 and the override argument is `--doorstop-target-assembly`. The tree is outside the repo,
is not committed and does not travel between machines — recreate it by copying `BepInEx\core` and
`BepInEx\config` on any machine that needs it. The 0.8.0 build currently there is 93,696 bytes, 2026-09-20,
and is now **stale** relative to this session's fixes: re-copy before the next private run.
