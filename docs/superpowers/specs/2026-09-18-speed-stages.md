# Speed stages: finish 0-1..0-3 FASTER before promoting to 0-4

Status: approved by the lead, 2026-09-18. Implemented on branch `speed-stages`.

**The instruction.** "dont have it promote to 0-4 untell it gets better times on these levels i know they can
be done much faster." Today a stage ends on a fresh-start *completion rate* alone, so a specialist that
finishes 0-1 half the time in four minutes promotes exactly like one that finishes it in ninety seconds. The
ladder then walks off to 0-4 carrying a slow policy on every level behind it. This adds a second KIND of stage
whose whole job is the clock, and puts three of them in front of any 0-4 stage.

## 1. Stage kinds

A plan entry is now a `(level, kind)` pair, `kind` in `{"complete", "speed"}`.

| | `complete` | `speed` |
|---|---|---|
| reward | today's, unchanged | today's, with a time-scaled `level_complete` (§2) |
| promotes on | fresh completion rate >= `target_rate` (0.5) | rate >= 0.4 **and** best official time <= `target_seconds` |
| step cap | 6M | 8M |
| run name | `spec_0-1` | `spec_0-1_speed` |
| initialised from | the previous stage's specialist | **this level's own** `models/specialists/<level>.zip` |
| on promotion | writes that specialist | **overwrites** that specialist |
| `keep_best` metric | `campaign` (rate) | `time` (§5) |

A speed stage gets its own run name because `best_time` becomes the thing `best.zip` is scored on: a shared
`runs/spec_0-1/` would mix two meanings of "best" in one `metrics_log.csv` and one `best.json`, which is the
same accounting break that forced `campaign_ppo_ground` to be a new run after the novelty fix.

**Plan order** (`configs/specialists.yaml`): 0-1c, 0-2c, 0-3c, **0-1s, 0-2s, 0-3s**, then 0-4c, 0-5c, ... as
today. `Plan.order` stays the 30 DISTINCT levels in mission order, so `full_run.py` still plays each level once.

## 2. The time-scaled completion bonus

In a speed stage `level_complete` is paid as

    level_complete * clip(target_seconds / official_seconds, 0.25, 2.0)

so with `level_complete: 100`: exactly the target pays 100, half the target pays the 200 ceiling, twice the
target pays 50, and anything slower bottoms out at 25 -- **finishing is always worth strictly more than not
finishing**, which is the property the void-farming post-mortem says never to give up. The per-decision `time`
cost is unchanged; this is the terminal half of the same pressure, and it is the half a `gamma` of 0.998 can
actually see from the spawn.

Paid only on a **fresh-start completion whose official time exists**. A checkpoint-respawn completion starts
partway through the level and inherits a running timer, so its "time" is meaningless; it pays the plain
`level_complete`, exactly as today. So does any completion frame that arrives with no official time (the scene
can unload before the campaign block is rebuilt).

`rewards.completion_bonus(cfg, target_seconds, official_seconds)` is the whole rule and is pure. With either
argument missing or non-positive it returns `cfg.level_complete` unchanged, so every non-speed run is
byte-identical.

## 3. Where `target_seconds` comes from

**The level's own S-rank time threshold**, never a hand-picked number. The mod's campaign block carries
`campaign.ranks = {"time": [t0, t1, t2, t3], "kills": [...], "style": [...]}`, and `campaign.grade` scores the
time category 4 (S) only when `seconds <= ranks["time"][3]`. So:

    target_seconds = obs["campaign"]["ranks"]["time"][3]        # campaign.s_rank_time()

Read **live at the first observation of the stage** (`UltrakillEnv._reset`, campaign branch, only while the
env has no target yet, so it is a constant per level load and heals a first load whose block was missing) and
then:

- the env reports it every episode as `info["target_seconds"]`;
- `ProgressCallback` keeps the last non-None one and writes it to `status.json` as `campaign.target_seconds`;
- `campaign_driver` reads it from there into `Stage.target_seconds` (saved in `driver_state.json`) and into the
  promoted specialist's sidecar.

One source of truth: the env. A per-level override is allowed in the plan (`speed.targets: {"Level 0-1": 95}`),
which the generated stage config passes as `EnvConfig.speed_target_seconds`; the env then reports THAT number
down the same path, so the driver's rule and the reward always agree.

`EnvConfig.speed_bonus: bool` is the switch. `speed_target_seconds: 0.0` (the default) means "read the level's
own S threshold". A speed stage that never sees a `ranks` block gets no target, pays the plain bonus and can
only end at its step cap -- a loud, safe failure, logged by the driver.

## 4. Promotion from a speed stage

`campaign_driver.stage_verdict(sample, start_steps, target_reached_at, rule, kind=..., target_seconds=...)`:

    complete:  rate >= rule.target_rate  and  fresh_window >= rule.min_fresh_window
    speed:     the same, with rule.target_rate 0.4,  AND  best_time is not None and best_time <= target_seconds

latched exactly as today (a dip cannot un-reach it, because `keep_best` is holding the peak in `best.zip` and
the peak is what gets promoted), then `settle_steps` (300k) from the LATER of the latch and the last `best.zip`
move, then `max_steps_per_stage` (8M for speed) as the unconditional out, recorded `"unfinished"`.

## 5. `keep_best --metric time`

A third metric beside `kills_per_min` and `campaign`. It **gates** on the fresh completion rate and then ranks
on the clock:

- a sample counts only when `fresh_window >= 20` **and** `fresh_completion_rate >= --min-rate` (0.3): a 40-second
  time set by the one lucky load out of a hundred is not a policy;
- score = `best_time` with `score_sign -1` (lower is better), tie-break = `fresh_completion_rate` with
  `penalty_sign -1` (higher rate wins);
- `best.json` records `penalty_name: "fresh_completion_rate"`, so the existing guard -- refuse to start when
  `best.json` was written with another tie-break -- keeps `--metric time` and `--metric campaign` from ever
  overwriting each other's `best.zip`. The `campaign` and `kills_per_min` paths are untouched: both signs
  default to +1 and `scored()`/`rank_key()` behave exactly as before.

`StageSupervisor.helper_specs` starts `keep_best.py --metric time --min-rate 0.3` for a speed stage and
`--metric campaign` for a complete one.

## 6. Plan reconciliation

The driver is LIVE on stage 3/30 and must restart into the new plan without losing it.

- `driver_state.json` is matched to the plan by **(level, kind)**. A history entry or a `current` stage written
  before this change has no `kind` and reads as `"complete"` -- which is what it was.
- On load, `DriverState.reconcile(plan)` re-numbers `current.index` and every history entry's `index` from the
  plan's stage list, so inserting three speed stages does not renumber the running stage out of existence. A
  `(level, kind)` the plan no longer lists keeps its stored index and is reported.
- The plan file is re-read on every driver start (`load_plan` in `main`), so editing
  `configs/specialists.yaml` and restarting the driver is the whole upgrade path.
- `next_stage()` is the first plan stage with no history entry for its `(level, kind)`.

## 7. Tests (all no-game)

- `tests/test_campaign_rewards.py`: the bonus at the target (100), at half (200 ceiling), at double (50), at 5x
  (25 floor); no target => byte-identical plain `level_complete`; a target with no official time => plain.
- `tests/test_campaign_env.py`: a speed env reads the S threshold off the fake level's `ranks` into
  `info["target_seconds"]`; a fresh-start completion pays the scaled bonus and reports it as
  `info["completion_bonus"]`; a checkpoint-respawn completion pays the plain 100; a config override wins over
  the live threshold; a non-speed env is unchanged.
- `tests/test_campaign_driver.py`: stage kinds in names and paths; the speed promotion rule (rate alone does not
  promote, rate + time does, an unread target never promotes); the sidecar's `mode`/`target_seconds`; a speed
  stage initialises from its own level's specialist and overwrites it; reconciliation of a pre-change state
  file against the new plan; `keep_best --metric time` in the speed stage's helper set.
- `tests/test_keep_best.py`: the min-rate gate, lower time wins, the rate tie-break, the cross-metric guard.
- `tests/test_specialists_config.py`: the new order (the three speed stages sit before any 0-4 stage, and
  `Plan.order` is still the 30 levels in mission order), and the speed rule's numbers.
- `tests/test_progress.py`: `target_seconds` / `completion_bonus` survive the numeric pipeline and reach
  `status.json`'s campaign block.
- `tests/test_full_run.py`: a specialist promoted by a speed stage is played with the speed run's env config.

## 8. Two additions the lead made after this spec was approved (2026-09-18)

### 8a. The hold line

§4 says a speed stage that reaches its 8M cap is recorded `"unfinished"`, and §6's `next_stage()` then walks
on. Together those two sentences promote the ladder to 0-4 with exactly the slow policies the instruction --
"dont have it promote to 0-4 untell it gets better times on these levels" -- forbids. The cap was written as a
guard against ONE blocked level stopping the other 29; here the blocked levels are the whole point.

A plan key, `hold_before: "Level 0-4"` (nullable -- `null` or absent lifts the line, and it is the whole
upgrade path back):

- **The line.** No stage at or after `hold_before`'s FIRST plan position starts while any stage in front of it
  has a latest status other than `"done"`. `"skipped"` (an operator's `--start-at`) counts as satisfied;
  `"unfinished"` and "never run" do not. `load_plan` refuses a `hold_before` that names no level in the plan,
  because a typo there would silently lift the line.
- **Round robin while held** (`Driver.choose_stage`): among the not-done stages in front of the line, the one
  with the FEWEST ended rounds, ties in plan order. Each round is a whole fresh step budget, because
  `begin_stage` measures `start_steps` from the file it resumes from and the rule measures the cap from
  `start_steps`.
- **A round resumes from that stage's OWN newest weights** (`Driver.round_init`): `supervise.choose_resume` on
  the stage's own model directory (what `ensure_trainer` will actually load, so `start_steps` matches the
  resume point), else `best.zip` via `promotion_source`, else the level's own promoted specialist. Never from
  scratch, and never from another level.
- **Eligibility.** A `(level, speed)` stage may only run once `(level, complete)` is `"done"` AND
  `models/specialists/<level>.zip` exists -- it resumes from that file, so without it there is nothing to
  resume. An ineligible stage is skipped over (logged once); if EVERY stage in front of the line is blocked the
  driver logs it and stops (`"held"`, exit 1) rather than spinning.
- **Rounds are recorded.** `Stage.round`, one history entry per round carrying `round`, and `round` in the
  promoted sidecar. `reconcile` numbers rounds for entries written before the key existed.
  `specialists_status.py` prints `holding before Level 0-4: waiting on Level 0-1 (speed, round 1, unfinished),
  ...` and the running stage's `[speed, round N]`.

The live `driver_state.json` -- stage 3 `Level 0-3` running, 0-1 and 0-2 `"done"`, no `kind` anywhere -- loads
into all of this unchanged and keeps running: `tick` never consults `choose_stage` while a stage is current.
`tests/test_campaign_driver.py` pins that against a COPY of the real file.

### 8b. The target scale

§3's bare S threshold is too weak to be a speed target. Measured 2026-09-18: the 0-2 specialist's best is
**139.5 s**, which already scores rank S, so a speed stage on 0-2 would have latched on its first observation
and promoted with zero improvement. An S-rank time is what a competent human run scores, not a fast one.

`speed.target_scale` (default **0.75**), so

    target_seconds = speed_target_scale * campaign.ranks.time[-1]

applied in exactly **one** place, `UltrakillEnv._note_speed_target`, so the reward (§2) and the driver's
promotion rule (§4) still read the same number through `info["target_seconds"] -> status.json ->
driver_state.json -> the sidecar`. A per-level `speed.targets` override is a decision already made and is
**not** scaled. The raw threshold travels beside the target as `s_rank_seconds` (env info -> `status.json` ->
`StageSample` -> `Stage` -> the sidecar) and is recorded, never compared against -- a sidecar that says
"target 82 s" without saying what S is cannot be read a month later. A zero or negative scale means "no
target", which §3's rule already handles as a loud, safe failure.

Tests: `tests/test_campaign_env.py` (the scale at 1.0/0.75/0.5, the raw threshold reported, an override not
scaled), `tests/test_progress.py` (`s_rank_seconds` through the numeric pipeline into the campaign block),
`tests/test_campaign_driver.py` (the generated config carries the scale; the sidecar carries both numbers),
`tests/test_specialists_config.py` (the plan's own 0.75 and its hold line).

## 9. What the 2026-09-18 adversarial review changed

Two reviewers (an RL-exploit lens and a driver-logic lens) read the branch before it was merged. Both returned
**the same blocker**, and it is worth writing down because it was a mistake of statistics, not of wiring: the
mechanism was correct end to end and still could not have worked.

### 9a. The clock clause reads the MEDIAN, not `best_time` (blocker)

`status.json`'s `campaign.best_time` is the run's **lifetime minimum** over every fresh completion by any of
the twelve envs. `ProgressCallback` only ever lowers it, and `_restore` carries it across every trainer
restart and into every later round. It is not a property of any checkpoint and it never decays, so one lucky
load satisfied `best_time <= target_seconds` for the rest of the run -- and the hold line, whose entire job is
to refuse the ladder a slow policy, would have opened on one.

The gap is measured, not hypothetical. Read from the live runs on 2026-09-18:

| run | `best_time` | `median_time_50` |
|---|---|---|
| `spec_0-1` | 243.4 s | 490.9 s |
| `spec_0-2` | 139.5 s | 236.5 s |

The run's single best is about **half** its typical completion in both cases. `stage_verdict`'s speed clause
is now `sample.median_time <= target_seconds`, reading `campaign.median_time_50`, which `ProgressCallback`
already computed and `poll_status.py` already carried into `metrics_log.csv`. `best_time` stays in the sample,
the log line and the sidecar as a headline and is never compared against anything.

`keep_best --metric time` had the same defect with a second consequence: because `-best_time` is monotone
non-decreasing, every sample after the last record tied at the top score and `rank_key` fell through to the
tie-break, so `--metric time` silently ranked the completion RATE -- and the "current is well below best"
warning could never fire, since `raw_new > raw_best` cannot hold for a running minimum. Its score column is
now `median_time_50` too.

### 9b. The floor is approached, never reached (major)

`max(target/official, 0.25)` is flat wherever the policy is more than 4x its target, which is where every
policy starts. Over the 68 fresh 0-1 completions in the live log (median 481.9 s, target 90 s) 58 of them --
**85%** -- sat exactly on the clip: a flat 75% pay cut with `d(bonus)/d(time) == 0`, paying less for the
behaviour the policy already had and saying nothing about how to improve it. Above the target the ratio is now
mapped into `(MIN, 1]` rather than clipped into `[MIN, 1]`:

    scale = SPEED_BONUS_MIN + (1 - SPEED_BONUS_MIN) * target/official        (official > target)
    scale = min(target/official, SPEED_BONUS_MAX)                            (official <= target)

1.0 at exactly the target (unchanged), strictly decreasing at every slower time, and strictly above the floor
at any finite time -- so the floor's safety property is stronger than the clip's, not weaker. At
`level_complete: 100` and a 90 s target: 482 s pays 39.0, 243 s pays 52.8, 150 s pays 70.0, 90 s pays 100.0.
**Not validated in game**: no speed stage has been trained with this, and the slope is still small next to the
per-step `time` term (a 482 s -> 243 s improvement is +13.8 here against +71.7 there).

### 9c. Every episode of a speed stage is a fresh level load (major)

The bonus is scaled only on a fresh-start completion, and the stage is scored only on fresh-start episodes
(`fresh_completion_rate`, `median_time_50` and `best_time` are all fresh-only). At the plan's
`fresh_start_prob: 0.2` a checkpoint-respawn completion therefore paid the full unscaled 100 while the
completions the stage actually measures paid 39-100 -- and 37% of the live `spec_0-1` completions were
respawns. `fresh_start` is not in the observation, so at the same packed state near the exit the terminal
reward depended on a hidden variable and PPO would have fitted one baseline across both, handing every
fresh-start completion a systematically negative advantage. `stage_config` now sets `fresh_start_prob: 1.0`
for a speed stage. Deaths still respawn at checkpoints INSIDE an episode; only the episode's own start moves.

### 9d. A speed round never overwrites a specialist it did not beat (major)

`promote()` wrote `models/specialists/<level>.zip` unconditionally, including for a stage recorded
`"unfinished"` at its cap, with no comparison against the file already there. Before speed stages each level
was promoted at most once so nothing could regress; now `refuse_promotion` declines when a SPEED stage did not
end `"done"` and the level already has a specialist. The round's weights are not lost -- they stay in
`models/spec_<level>_speed/`, which is exactly where `round_init` resumes the next round from -- and the
history entry records `promoted: false` with the reason.

### 9e. The round robin has an exit (minor, and the hold line needs one)

`speed.max_rounds` (3, and `stage.max_rounds` for a stage in front of the line) caps how many rounds one stage
may have before `choose_stage` returns nothing and the driver stops with a loud `HELD`. The line has no other
exit, and `0.75 x S` on Level 0-1 is **90 s against a 183.6 s leaderboard best**, so "the policy cannot reach
this target" is the expected case rather than a remote one. When the driver stops, retune `speed.target_scale`
or add a per-level `speed.targets` override.

### 9f. A new round ignores the previous round's `status.json` (major)

A round resets `target_reached_at` but REUSES the run directory, so the first tick of round N read round N-1's
final numbers -- the same rate, the same window, the same cumulative best -- and could re-latch immediately,
recording `"done"` a settle later on exactly the data the cap had just rejected. `Stage.stale_below` is
stamped at `begin_stage` from whatever the run's `status.json` says at that moment, and samples at or below it
are discarded until the new trainer passes them.

### 9g. `--start-at` is inside the hold line (major)

The line was enforced only in `choose_stage`, and `--start-at` bypasses it. `start_at_objection` (pure,
tested) now refuses a stage at or after `hold_before` while anything in front is not done (`--ignore-hold`
overrides it, loudly), and refuses a stage that has already run (`--rerun-stage` overrides it). The second
guard matters because `current` is null after a `"held"` exit as well as on a first launch, and
`runs/start_driver.cmd` carries `--start-at "Level 0-1" --init <a 17.0M checkpoint>` permanently: an operator
reading exit code 1 as a crash would otherwise have begun a second round of the finished 0-1 stage.
`begin_stage` also takes `start_steps` from what `ensure_trainer` will actually resume (`choose_resume` on the
stage's own model directory) rather than from `--init`, so the stage's cap and the generated `timesteps` total
can no longer be measured from an origin the trainer never visits.

### 9h. Rejected

- *"The value function is shocked by the ~75% terminal-reward change on resume."* Real in mechanism and
  partly mitigated by 9b (the median 0-1 completion now pays 39, not 25), but the proposed anneal of
  `speed_target_scale` from 1.0 to 0.75 does **not** address it: at scale 1.0 the target is still 120 s
  against a 482 s median, the same ratio regime. A real anneal would have to blend the plain weight with the
  scaled one over the first rollouts, which is new machinery on a mechanism nothing has trained against yet.
  The unbounded-loop half of the same finding is fixed by 9e.
- *"Clear `campaign.best_time` per round."* Not needed once the gate is the median (9a); `best_time` is a
  reported headline and a cumulative record of the run, which is what `times.md` wants it to be.

## 10. Depth-first by level, and ending a stage on purpose (2026-09-19)

The user, reading the round robin back: *"shouldnt we just work on 0-1 untell its finished before working on
the other levels in that case?"* §8a's round robin spread twelve games over three levels at once -- a fresh
budget for every not-done stage before any of them got a second -- which is the shared-run failure mode
(`campaign_gates`' thrash) moved from inside one policy to the schedule. Two changes, both in the driver and
its plan; no env, reward, observation or action semantics move.

### 10a. `hold_order`

A plan key beside `hold_before`, with two values:

- **`round_robin`** -- §8a exactly as it shipped, and still the CODE default, so every plan and test written
  before this key means what it meant.
- **`sequential`** -- what `configs/specialists.yaml` now sets. While the line is up, `Driver.choose_stage`
  returns the **first not-done stage in plan order that can start**, and returns it again round after round
  (each round a fresh step budget, resumed through `round_init` from that stage's own newest weights, exactly
  as before) until its own rule records it `"done"`. Only then does the next stage start.

`load_plan` refuses an unknown value rather than defaulting it: `hold_order: sequental` would quietly go back
to the round robin. A stage that cannot start at all (`stage_blocked` -- a speed stage whose complete stage is
not `"done"`, or with no specialist file) is passed over only because nothing in front of it can run, and the
driver logs which stage and why. Nothing else changes: the line itself, eligibility, `max_rounds: 0`, the
`"held"` exit when NOTHING is runnable, and `--start-at`'s two objections are all §8a's.

**The plan is reordered level-major** so that "first in plan order" IS "the earliest unfinished level":
0-1 complete, 0-1 speed, 0-2 complete, 0-2 speed, 0-3 complete, 0-3 speed, then 0-4 and the rest of the
ladder. `Plan.order` -- the DISTINCT levels, which is what `full_run.py` plays -- is unchanged, and so is the
hold index (6). History is matched by `(level, kind)`, so `DriverState.reconcile` renumbers every entry's
`index` onto the new positions and loses, duplicates and renames nothing;
`tests/test_campaign_driver.py::test_todays_state_reconciles_onto_the_depth_first_plan_and_finishes_0_1_first`
pins that against a literal copy of the live state file's shape and then walks the whole depth-first order.

### 10b. `END_STAGE`

Switching the order is worth nothing while a stage that the new order would not have chosen is holding the
machine. A second control file beside `DRIVER_PAUSE`, `runs/specialists/END_STAGE`, ends the CURRENT stage on
the driver's next poll through `finish_stage` -- the ordinary stage-end path:

- status `"unfinished"` with `"reason": "ended by operator"` in the history entry, so every other rule reads
  it exactly as it reads a stage that ran out of steps;
- `refuse_promotion` therefore applies unchanged: a SPEED stage's weights do NOT replace the level's promoted
  specialist, and nothing in `models/` is deleted -- the round's weights stay in `models/<run>/`, which is
  where its next round resumes from. A **COMPLETE** stage is the other case and DOES promote its `best.zip`
  to `models/specialists/<level>.zip` with a fresh sidecar, exactly as it does at the step cap; the source is
  `keep_best`'s lifetime peak for that model directory, so the weights cannot regress, but a committed file
  does change (2026-09-19 review);
- the sample is read through `Driver.current_sample`, the same reader (and the same `stale_below` filter)
  `tick` uses: an operator picks the moment, and a moment inside the first `start_grace_seconds` of a round
  >= 2 would otherwise record the PREVIOUS round's rate, median and best as this round's, and write them into
  that committed sidecar (2026-09-19 review);
- the trainer and its helpers are stopped exactly as at any stage end, and the games are not deliberately
  restarted -- but, as at ANY stage change, `ensure_games` relaunches all twelve if a port is not listening
  when the next stage starts, so check `games.py status` before writing the file;
- the file is deleted BEFORE the stage ends, so one file can only ever end one stage, and a file that cannot
  be deleted ends nothing at all (it says so once, in its own log slot);
- then the plan's own rule chooses and starts the next stage. Under `sequential` that is the first not-done
  stage in plan order -- which, when the stage just ended IS that stage, is the **next round of the same
  stage** with a fresh budget. `END_STAGE` moves the machine to another level only when the one it ended was
  not the leading one; to stop working on a level, change `hold_order`, set a positive `max_rounds`, or
  retune its target.

The file's text may name the stage it means (`Level 0-2 speed`, `Level 0-2/speed`, `Level 0-2`, or empty for
"whatever is running"). A name that does not match the running stage is refused, logged and deleted: a stale
file may not end the wrong stage. `DRIVER_PAUSE` is checked first, so a paused driver does nothing at all,
including deleting the file; `--dry-run` only reports what it would do.

The text is decoded by `decode_control_file`, not by `read_text("utf-8")`. The scratch dry-run against a copy
of the live state caught the reason: `Set-Content -Encoding utf8` writes a **byte order mark**, so the file
the recipe in `docs/commands.md` tells an operator to write parsed as the level `"﻿Level 0-2"` and was
REFUSED. PowerShell 5.1's `>` and `Out-File` can write UTF-16, which is not valid UTF-8 at all and raised
where the caller expects "no text". `utf-8-sig`, then `utf-16`, then `latin-1`, and the parser strips a stray
BOM and NULs as well.

### 10c. Tests (all no-game)

`tests/test_campaign_driver.py`: `hold_order` loaded, defaulted and refused; sequential runs one stage round
after round and only then the next, where round robin would have moved on; a blocked stage passed over with
the reason logged; today's state reconciled onto the reordered plan and the whole order walked (0-2 speed
keeps running, then 0-1 speed round 2 from its own weights, 0-2 speed round 2, 0-3 complete, 0-3 speed only
after it, 0-4 last); `END_STAGE` ends the right stage once, removes the file, records the reason, leaves the
specialist and the model files alone, is refused when stale or mismatched, loses to `DRIVER_PAUSE` and only
reports under `--dry-run`. `tests/test_specialists_config.py`: the shipped plan is `sequential` and
level-major, and `Plan.order` is untouched.

The 2026-09-19 review added three more: an `END_STAGE` inside a round-2 stale window records `None` for the
rate, median, best and step count rather than the previous round's numbers (and the complete-stage sidecar it
promotes carries the same `None`s); a control file the driver cannot unlink says so ONCE across three polls
instead of alternating with the running-stage line every poll; and a blocked leading stage is named again for
every round a later level takes, and is printed by `specialists_status.py` beside the order rule.

## 11. FOCUS: one level, a ladder of medians, chasing the record (2026-09-20)

The user, 2026-09-20: *"can we focuse on one level tell we get it to a point that is close to the speed run
record"*. §10's depth-first order still spreads the project across levels over time -- 0-1 speed passes its
S-rank rung and the machine leaves for 0-2 with 0-1 still **7.4x** the human record. A focus refuses that: it
holds the whole machine on ONE level until it is close to the record, and it replaces the target question
("what is S?") with a LADDER.

### 11a. The plan block

```yaml
focus:
  level: "Level 0-1"
  targets: [120, 100, 85, 72, 60, 50, 42, 35, 30, 25]   # median OFFICIAL seconds
```

Nullable. Absent or `focus: null` is today's behaviour exactly -- the hold line and `hold_order` decide
everything, and every plan and test written before this key means what it meant.

**Why Level 0-1.** It is the first level, it has the most trained brain in the project (28.39M cumulative
steps) and it is the only level whose speed stage has ever passed: 2026-09-19, median_time_50 **147.16 s**
against a 150 s target, fresh rate 0.92, best single run **81.46 s**, promoted to
`models/specialists/Level_0-1.zip`. The human INBOUNDS individual-level record is **19.798 s**
(`configs/il_records.yaml`, `docs/il-records.md`); the 4.915 s Any% leaves the level and is not a target a
campaign policy can reach. So the AI's median is 7.43x the record and its best 4.11x it. The last rung, 25 s,
is 1.26x the record -- "close to" without pretending a memoryless MLP with no hazard channel will beat a
human.

`load_plan` validates the block LOUDLY, for the reason `hold_before`'s typo check exists: a focus that
silently does nothing spends days of twelve games on the wrong thing with nobody watching. Refused: a level
not in the plan; a level with no `{level: ..., kind: speed}` entry under `order:` (every name, path and index
comes from that entry, and `begin_stage` would raise on `index_of` at the moment the stage started); an empty
`targets`; a non-numeric, zero or negative target; a ladder that is not STRICTLY decreasing; an unknown key.

### 11b. A rung is a speed stage with an explicit target

Rung *k* is an ordinary speed stage of the focus level whose `target_seconds` is `targets[k]` **exactly** --
never the S-rank time, never multiplied by `speed.target_scale`. It uses the level's own speed run
(`spec_0-1_speed`: the same run directory, the same model directory, the same `metrics_log.csv`), is resumed
through `round_init` from **that run's own newest weights**, and the target reaches the env down the one
existing path, `env.speed_target_seconds` -> `_speed_target` -> `info["target_seconds"]` -> `status.json`, so
the completion bonus and the promotion rule are measured against the same number. §4's promotion rule is
unchanged: fresh rate >= `speed.target_rate` (0.4) over >= 30 fresh episodes AND `median_time_50` at or under
the rung, latched together, then `settle_steps` from the later of the latch and the last `best.zip` move.

**Which rung is current is derived, never stored.** `focus_rung(plan, state)` walks the ladder and returns the
first target no `"done"` speed round has met. `rung_met` counts a rung met when a done round's own
`target_seconds` was at or under it (a harder rung already passed implies every easier one) or when its
recorded `median_time` was at or under it (achieved in fact, even if it was aimed higher). Both read the
history entries the driver has always written, so **nothing migrates**: today's `"done"` 0-1 speed round at
target 150 with median 147.16 makes every rung above 148 met and puts the focus on **120**. An `"unfinished"`
round proves nothing however good its median looked, and an impossible time is rejected by the shared
`times.valid_official_seconds`.

### 11c. What the driver does with it

- `choose_stage` short-circuits: while a focus is set and its ladder has a rung left, the only stage it ever
  returns is that level's speed stage -- whatever `order:`, `hold_before:` and `hold_order:` would say.
  Nothing on 0-2, 0-3 or 0-4 starts. `--start-at` is refused for any other stage for the same reason.
- A rung that is MET is recorded `"done"` with its `target_seconds` and its `rung` index, and it **promotes**:
  the specialist is overwritten because a met rung's median is by construction faster than the last promoted
  one. A round that runs out of steps is `"unfinished"`, `refuse_promotion` declines exactly as it does
  today, and the SAME rung runs again from the stage's own newest weights -- unbounded, because
  `speed.max_rounds` is 0. A round cap, if one is ever turned back on, counts the rounds of THIS rung
  (`DriverState.rung_rounds`), or the tenth rung would start already out of rounds.
- **THE LADDER ADVANCES ON THE MEASURED MEDIAN, NEVER ON THE ROUND'S OWN TARGET** (2026-09-20 review).
  `rung_met` reads `median_time` out of the `"done"` entry and nothing else. `stage_verdict` latches on the
  FIRST 50-episode window whose rate and median clear the bar and never un-latches, so `"done"` can be written
  on one lucky window and the 300k-step settle can end with the median drifted back above the target -- with
  keep_best never moving `best.zip`, `promote` re-copying the identical file, and the ladder nevertheless
  walking on to a rung the policy has never been near, then repeating 8M-step rounds of it for good.
  0-1's completion times run from 81 s to well past its 147 s median, so a transient dip is not exotic.
  Reading the recorded median makes a second, independent window agree 300k steps later before the ladder
  moves; a round whose median drifted back simply runs the same rung again, which is the whole correction.
  The cost is that a `"done"` round with no median recorded at all (a filtered final sample) meets nothing and
  repeats its rung -- conservative, and safe in the direction that matters.
- **The rung is a property of the stage, not of the code path that started it.** `begin_stage(rung=...)`
  defaults to `AUTO_RUNG` and asks `rung_for`, so `tick`, `finish_stage` and `main()`'s `--start-at` all get
  the same answer. Before that, `--start-at` was the one caller that forgot: `start_at_objection` deliberately
  permits the focus level's own speed stage, so an operator restarting it by hand after a `held` or
  `no_checkpoint` exit began it with no rung at all -- `target_seconds` fell back to the plan (0-1 has no
  `speed.targets:` override), the env reported the level's S-rank 150 s live, the tick clause accepted that
  number because `stage.rung` was None, and the stage would latch and promote on a bar the level passed on
  2026-09-19. Passing `rung=None` explicitly still means "not a rung".
- **A focus finishes its level first.** If the focus level's speed stage is blocked because its complete stage
  is not `"done"`, `choose_stage` returns that COMPLETE stage (never a rung) and says so, instead of returning
  nothing. "Focus on Level X" means get X done and then chase its clock. The old behaviour was a `"held"` exit,
  and the 12 games are not children of the driver: they would have kept burning a commit-bound box with no
  trainer until a human noticed -- the one outcome `max_rounds: 0` exists to avoid -- and `focus.level` is a
  single line of the plan that the user's own direction invites changing. Both of the focus level's stages are
  therefore also allowed through `start_at_objection`. A focus still stops when its level's specialist FILE is
  missing (a broken tree, not a plan), and the stopping line now leads with the focus rather than with the
  hold line, and says out loud that the games are still running.
- **`speed.max_rounds` must stay 0 while a focus is set.** A positive cap turns a stalled rung into exactly the
  held exit above -- driver gone, 12 games running, nobody watching. Retune the ladder instead.
- `Stage.stale_below` is what stops a new rung latching on the previous rung's `status.json`: a rung boundary
  IS a stage boundary, so `begin_stage` sets it from whatever the file says at that instant and
  `current_sample` returns `EMPTY_SAMPLE` until the new trainer passes it. Without it the 100 rung would
  latch on the 120 rung's final median a settle later and promote on data it never produced. The driver logs
  the number when the rung starts. A rung's target is also never overwritten by the live `target_seconds` the
  run reports (`stage.rung is None` guards that path): a rung is a decision already made.
- **`keep_best` carries over, and that is right.** A rung does not change the metric: `--metric time` ranks
  the LOWEST median over the run's whole `metrics_log.csv`, and `best.json`'s `penalty_name` therefore still
  matches, so keep_best does not refuse to start. `best.zip` at the start of a rung is the fastest policy the
  run has ever produced -- the one that passed the previous rung -- and the new rung has to beat it to move
  the file. If it never does, the rung ends `"unfinished"` and promotes nothing; if it does, `best.zip` is
  strictly faster than what is in `models/specialists/`, which is exactly the file a met rung should promote.
  The only thing the carried `best.json` affects is `best_at`, which can only ever make the settle LONGER.
- When the last rung is done the focus **stops** and says so loudly, once, and the ordinary plan takes over.
  It does not silently keep training the final target.

### 11d. Reporting

`specialists_status.py` prints the focus first: the level, the human inbounds record, the current rung
("rung 1 of 10: the median must reach 02:00.000 (6.06x the record)"), the whole ladder, the live best and
median each as a multiple of the record ("best 01:21.464 (4.11x the record) median 02:27.160 (7.43x the
record)"), and every rung already done with its median and its step count. The record is read from
`configs/il_records.yaml` for DISPLAY only -- no reward, gate or observation reads it, the inbounds time
rather than the Any% one, and a missing file is simply no record to print. The hold-line block still prints,
with a line saying the focus overrides it.

### 11e. Tests (all no-game)

`tests/test_campaign_driver.py`: the block loaded, defaulted and refused ten ways; `focus_rung` starting at
120 from a literal copy of the live state and skipping rungs a wider ladder has already met; `rung_met`'s
monotonicity, its refusal of an unfinished round, of an impossible time, of a round with no clock, and of a
round that latched at 120 but ended with its median drifted to 136; the generated config carrying the exact
target and nothing else changed; the focus overriding the depth-first rule that would have gone to 0-2;
a met rung promoting with `rung` in the sidecar and the next rung starting from the stage's own weights with
the stale guard set; an unfinished rung repeating itself without touching the specialist; the focus stopping
when the ladder is done; a focus whose level is not finished running that level's COMPLETE stage first and
then returning to the ladder; a stage started BY HAND getting the same rung `choose_stage` would have given
it, and no rung for any other stage; a focus with no specialist file stopping the driver rather than training
something else, with the stopping line naming the focus; `--start-at` refused for another level and allowed
for either of the focus level's own stages; a round cap counted per rung; and **the literal-state test** -- the live file of
2026-09-20 keeps 0-2 speed round 2 running on `tick()`, and once it is ended the next stage is 0-1 speed at
rung 120 resumed from `models/spec_0-1_speed`'s own newest checkpoint. `tests/test_specialists_config.py`
pins the shipped block. `tests/test_campaign_rewards.py` covers the two reward fixes that landed with it.

### 11f. The two reward fixes that landed with it

Both are on the SHARED reward path and change every stage. That is acceptable only because a new training
regime starts with them: the focus's first rung is a new baseline, taken after the switch, so nothing that
was being judged against an older baseline is disturbed. Neither is validated in game.

1. **`damage_dealt` had no lower bound.** The "killed in one hit" credit divided by
   `max(enemy_max_health.get(id, health), 1e-3)`, so an OVERKILLED enemy (negative health) whose id was
   missing from `enemy_max_health` produced `health x 1000`. Measured: one decision in
   `runs/probe_0-2_speed/rollout_11.jsonl` (i=2053) paid `damage_dealt = -250.0`, and 71 of 2,068 live
   completions (3.43%) ended with a NEGATIVE total episode reward, worst -1,119.5 -- 4-20x the whole death
   channel, unbounded, firing during exactly the arena fighting a gated level forces. `rewards.damage_share`
   now bounds EVERY per-enemy contribution to `[0, 1] x damage_dealt`, in both credit paths, and falls back
   to the caller's old divisor only when the enemy's own bar is unknown or unusable, so the ordinary case is
   arithmetically identical.
2. **`completion_bonus` paid the full weight when the official time was missing.** A completion with no clock
   paid 100 while every genuine completion slower than the target paid 39-99, which made a lost timer the
   best-paying completion on a speed stage. It now pays the FLOOR, `SPEED_BONUS_MIN x level_complete` = 25,
   whenever a target is set and the official time does not pass `times.valid_official_seconds` -- including a
   checkpoint-respawn completion, which cannot arise on a shipped speed stage because they force
   `fresh_start_prob: 1.0`. A run with NO target (every complete stage, Cyber Grind, every older config) is
   untouched and still pays `level_complete` exactly.
