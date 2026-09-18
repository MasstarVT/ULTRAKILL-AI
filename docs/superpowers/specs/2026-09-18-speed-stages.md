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
