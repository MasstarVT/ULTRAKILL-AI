"""Trains ONE SPECIALIST POLICY PER LEVEL, sequentially, without an LLM watching it.

**Why this exists.** Measured on the live `campaign_gates` run over the twelve hours to 2026-09-18, one shared
policy trained on a multi-level mixture thrashes: whichever level receives the fresh-start share improves while
the others regress. Level 0-1's fresh completion rate went 0.65 -> 0.13 -> 0.36 as the draws moved, and Level
0-3 fell from 7.3 gates reached to 3.6 as soon as its share was damped. Learning-progress weighting changed
WHICH level is starved, not that one is. The lead's decision is a specialist per level, trained one level at a
time on all twelve games, each initialised from the previous level's best, chained afterwards by
`scripts/full_run.py` into a full-game run -- one policy per level.

Nothing about the environment changes. Each stage is the existing single-level campaign mode (`level:` in the
config, no `levels:` list), so no observation, action or reward semantics move and a specialist is an ordinary
campaign checkpoint.

    python scripts/campaign_driver.py --start-at "Level 0-1" --init models/campaign_gates/best.zip --monitor 1

**This REPLACES supervise.py while it runs. Do not run both**: they would fight over the games and over which
trainer should exist. The driver embeds a `supervise.Supervisor` per stage and reuses its health checks, its
crash recovery, its boot gate and its resume-file rule verbatim -- see `StageSupervisor`.

**Pausing.** Exactly like the supervisor, and for the same reason (a planned stop looks like a crash to it):

    New-Item runs/specialists/DRIVER_PAUSE            # PowerShell, from python/
    Remove-Item runs/specialists/DRIVER_PAUSE         # when the pause is over

**Ending the current stage on purpose** (`END_STAGE`, 2026-09-19). A second control file, read only when the
driver is NOT paused:

    New-Item runs/specialists/END_STAGE                             # end whatever is running
    Set-Content runs/specialists/END_STAGE "Level 0-2 speed"        # ... only if THAT is what is running

On its next poll the driver ends the current stage through the ordinary stage-end path -- status
`"unfinished"`, reason `"ended by operator"`, the trainer and helpers stopped exactly as at any stage end, a
speed stage's weights NOT promoted over the level's specialist (`refuse_promotion`), nothing in `models/`
deleted -- deletes the file, and starts whatever the plan's rule chooses next. The round's weights stay in
`models/<run>/`, which is where `round_init` resumes its next round from, so nothing is lost. A file whose
text names a stage that is not the one running is refused, logged and deleted: a stale file may not end the
wrong stage.

**Stage kinds** (2026-09-18, `docs/superpowers/specs/2026-09-18-speed-stages.md`). A plan entry is a
`(level, kind)` pair. `"complete"` is the rule below, unchanged. `"speed"` is the same stage run again on a
level that is already being finished, with `level_complete` scaled by `target_seconds / official_seconds` and a
promotion rule that will not let the ladder move on until the level is actually being finished FAST: the
lead's instruction on 2026-09-18 was that nothing promotes to 0-4 until 0-1..0-3 have better times. A speed
stage keys on `spec_<level>_speed`, starts from the level's own promoted specialist and OVERWRITES it.

**The stage rule** (`stage_verdict`, pure, `configs/specialists.yaml` holds the numbers). A stage ends when
BOTH hold:

  - the level's fresh completion rate over its last 50 fresh episodes reaches `target_rate` (0.5) with
    `fresh_window >= min_fresh_window` (30). That is LATCHED: a later dip does not un-reach it, because
    `keep_best.py` is holding the peak in `best.zip` and the peak is what gets promoted. Requiring the rate to
    still be high at the moment of promotion would deadlock a stage that peaked and then collapsed -- which is
    this project's twice-observed failure mode (CLAUDE.md, "second peak and second collapse").
  - and `settle_steps` (300k) have passed since the LATER of that moment and the last time `keep_best.py` moved
    `best.zip` (`best.json`'s `at_timesteps`). A run still setting new bests keeps resetting its own clock and
    keeps training; the stage ends once the peak stops moving. Without the "later of" the settle could be
    already satisfied at the instant the target is reached, by a best saved long before it.

A SPEED stage adds one clause to the first: its `median_time_50` -- the MEDIAN official time over the
completions in the last 50 fresh episodes -- must also be at or under `target_seconds`, which is
`speed.target_scale` (0.75) times the level's own S-rank threshold (`campaign.ranks.time[-1]`), computed by the
env and carried here through `status.json`. It is the median and NOT `best_time` because `best_time` is a
run-lifetime minimum that one lucky load satisfies forever, across restarts and rounds (2026-09-18 review; on
the live runs the single best is about half the median). Its `target_rate` is lower (0.4), because a fast
policy that finishes four loads in ten is worth more to the ladder than a slow one that finishes six.

OR the stage has consumed `max_steps_per_stage` (6M, 8M for a speed stage) of its own, at which point it moves
on REGARDLESS and is recorded as `"unfinished"` so it can be revisited later. A blocked level must not stop
the other 29.

**The hold line** (`hold_before`, §8a of the spec) is the exception to that last sentence, and it exists because
the lead's instruction -- "dont have it promote to 0-4 untell it gets better times on these levels" -- is
precisely a refusal to walk past an unfinished stage. `hold_before: "Level 0-4"` means no stage at or after 0-4
starts while any stage before it is not `"done"`. While the line is up the driver keeps training the stages in
front of it, each round a fresh step budget resumed from that stage's OWN newest weights (`round_init`), never
from scratch and never from another level. Every round is one history entry carrying its `round` number, so
`specialists_status.py` can report "holding before Level 0-4: waiting on ..., round N". Setting
`hold_before: null` lifts the line.

`hold_order` decides WHICH held stage runs next (§10, 2026-09-19). `"round_robin"` is the rule the line
shipped with and is still the code default -- fewest ended rounds first, ties in plan order.
`"sequential"`, which `configs/specialists.yaml` now sets, is DEPTH-FIRST: the first not-done stage in plan
order runs round after round until it is `"done"`, and only then does the next one start. The user's
instruction on 2026-09-19 was "shouldnt we just work on 0-1 untell its finished before working on the other
levels", and the plan's stages are ordered level-major (0-1 complete, 0-1 speed, 0-2 complete, ...) so that
depth-first over the plan IS depth-first over the levels. A stage that cannot start at all (`stage_blocked`)
is passed over, and the log says which and why.

**Per stage**: run name `spec_<short level>` (e.g. `spec_0-1`), `models/spec_0-1/`, `runs/spec_0-1/`, the
generated config at `configs/generated/spec_0-1.yaml`, and the shared run's `explore_<level>_*.npz` copied in
when they exist (their floor counts carry the `1/sqrt(N)` decay that makes the agent push outward at all).

**On stage end** the trainer is killed and the checkpoint promoted to `models/specialists/<level>.zip` with a
JSON sidecar (rate, best time, steps, source checkpoint, difficulty). `best.zip` is preferred; with no
`best.zip` the newest checkpoint wins, chosen by `supervise.choose_resume`, which reads the step count out of
the zips rather than trusting `latest.zip`'s name -- `latest.zip` is only written on a graceful stop and a
killed trainer leaves it stale. The next stage then resumes from that promoted file.

**FOCUS / RECORD CHASE** (2026-09-20, `docs/superpowers/specs/2026-09-18-speed-stages.md` §11). The user:
"can we focuse on one level tell we get it to a point that is close to the speed run record". A `focus:` block
in the plan names ONE level and a LADDER of median times, and while it is set the driver trains nothing else:

    focus:
      level: "Level 0-1"
      targets: [120, 100, 85, 72, 60, 50, 42, 35, 30, 25]   # median official seconds

Each `targets` entry is a RUNG: a speed stage of that level whose `target_seconds` is that number exactly,
never the S-rank time and never scaled. It uses the level's own speed run (`spec_0-1_speed`), resumed through
`round_init` from that run's own newest weights, and it promotes by the ordinary speed rule -- fresh rate at
or above `speed.target_rate` AND `median_time_50` at or under the rung's target, latched, then the settle. A
rung that is met is recorded `"done"` WITH ITS TARGET and the specialist is overwritten (the rung's median is
by construction faster than the last promoted one); a round that runs out of steps is `"unfinished"`, does
NOT overwrite the specialist (`refuse_promotion`) and repeats the same rung. Which rung is current is derived
from the history, not stored: `focus_rung` walks the ladder and skips every target a `"done"` speed round has
already met, either by its own target or by its recorded median -- so today's `"done"` 0-1 speed round at
target 150 with median 147.16 puts the focus on the 120 rung. When the last rung is done the focus STOPS,
says so loudly, and the ordinary plan takes over. `focus: null` (or no block at all) is exactly today's
behaviour.

**State** lives in `runs/specialists/driver_state.json` (current stage, its start step count, the history), so
restarting the driver resumes the stage it was on instead of starting the ladder again. The plan file is
re-read on every start and the state is matched to it by `(level, kind)` (`DriverState.reconcile`), so stages
can be INSERTED into the plan -- which is how the three speed stages reached a driver that was live on stage
3 of 30 -- without the running stage losing its place. An entry written before kinds existed reads as
`"complete"`, which is what it was.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultrakill_ai.procmem import cap_blas_threads  # noqa: E402

cap_blas_threads()  # before ultrakill_ai.campaign; the driver only reads JSON and starts processes

import yaml  # noqa: E402

import supervise  # noqa: E402
from ultrakill_ai.campaign import CAMPAIGN_LEVELS_SHIPPED, safe_name  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402
from ultrakill_ai.times import short_level, valid_official_seconds  # noqa: E402

# `ultrakill_ai.progress.write_json_atomic` is the same nine lines, but importing that module pulls in
# stable_baselines3 and therefore torch. The driver polls beside twelve games and a trainer, and
# `specialists_status.py` imports this file only to print a report: neither may pay 300 MB for one helper.


def write_json_atomic(path: Path, data: dict) -> None:
    """Replaces `path` in one operation, so a reader sees the old whole file or the new one, never a torn one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader may hold the file open for a moment
            if attempt == 4:
                raise
            time.sleep(0.05)

STATE_VERSION = 1
DRIVER_RUN = "specialists"           # runs/specialists/: the driver's own state and log, not a training run
SPECIALIST_DIR = "specialists"       # models/specialists/: one promoted policy per level
GENERATED_DIR = "configs/generated"  # the real per-stage configs, written from configs/specialists.yaml
# Keys of a multi-level curriculum config that must NOT reach a single-level stage. `stage_config` drops them
# whatever the plan file holds, so a plan copied from campaign_gates_full.yaml cannot turn a stage back into a
# mixture -- which is the exact failure this whole driver exists to undo.
MULTI_LEVEL_KEYS = ("levels", "unlock_rate", "unlock_window", "unlock_after_fresh_episodes",
                    "level_weight_floor", "curriculum_weighting", "curriculum_weight_cap",
                    "curriculum_blocked_fresh_episodes", "curriculum_path", "max_steps_per_level")
# Slack on top of `max_steps_per_stage` in a stage's `timesteps` budget, so `train.py` never finishes the run
# out from under the stage rule: `timesteps` is the run TOTAL and learn() stops at it. The rule decides when a
# stage ends; this number only has to be bigger than the rule's own cap.
TIMESTEPS_SLACK = 1_000_000
COMPLETE = "complete"  # the stage kind that has always existed: promote on the fresh completion RATE
SPEED = "speed"        # ... and the one that promotes on the CLOCK as well (2026-09-18-speed-stages.md)
STAGE_KINDS = (COMPLETE, SPEED)
# HOW THE HOLD LINE SPENDS ITS ROUNDS (`hold_order` in the plan file, §10 of the spec).
ROUND_ROBIN = "round_robin"  # the original rule: fewest ended rounds first, ties in plan order
SEQUENTIAL = "sequential"    # depth-first: the FIRST not-done stage in plan order, round after round
HOLD_ORDERS = (ROUND_ROBIN, SEQUENTIAL)
# The operator's control file, beside DRIVER_PAUSE: end the CURRENT stage now, through the normal stage-end
# path. Its text may name the stage it means, and then it only ends THAT stage (see `end_stage_request`).
END_STAGE_FILE = "END_STAGE"
END_STAGE_REASON = "ended by operator"
# A speed stage's target is this times the level's own S-rank threshold (§8b). Duplicated as `EnvConfig`'s
# default so a stage config that predates the key still means the same thing; the env is where it is APPLIED.
DEFAULT_TARGET_SCALE = 0.75


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def stage_run_name(level: str, kind: str = COMPLETE) -> str:
    """`spec_0-1` for `Level 0-1`: the run name, the model directory and the runs directory all share it.

    A speed stage is `spec_0-1_speed`, a run of its own, because `best.zip` changes meaning inside it:
    `keep_best --metric time` scores the clock, the campaign metric scores the rate, and one `metrics_log.csv`
    and one `best.json` cannot hold both. This project has been here -- `campaign_ppo_ground` is a separate run
    for exactly this reason after the novelty measure changed under `part_novelty`.
    """
    return "spec_%s%s" % (short_level(level), "_speed" if kind == SPEED else "")


def stage_config_path(level: str, generated_dir: str = GENERATED_DIR, kind: str = COMPLETE) -> str:
    return "%s/%s.yaml" % (generated_dir.rstrip("/"), stage_run_name(level, kind))


def specialist_path(models_dir: Path, level: str) -> Path:
    return models_dir / SPECIALIST_DIR / ("%s.zip" % safe_name(level))


# ---------------------------------------------------------------------------
# The plan file, and the per-stage config generated from it
# ---------------------------------------------------------------------------


@dataclass
class StageRule:
    """When a stage ends. Every number comes from `configs/specialists.yaml`'s `stage:` block."""

    target_rate: float = 0.5
    min_fresh_window: int = 30
    settle_steps: int = 300_000
    max_steps_per_stage: int = 6_000_000
    # How many ROUNDS of one stage the hold line may spend before the driver stops and asks for a human
    # (§8a, 2026-09-18 review). Without it a target that the policy cannot reach -- and 0.75 x S is faster
    # than this project's own leaderboard best on 0-1 -- makes the round robin an unbounded loop that burns
    # `max_steps_per_stage` again and again with nobody told. 0 or less means no cap.
    max_rounds: int = 3
    count: int = 12
    monitor: int = 1
    seed_explore_from: str = "campaign_gates"


@dataclass(frozen=True)
class FocusPlan:
    """The `focus:` block: ONE level, and a ladder of median times to chase down it (§11, 2026-09-20).

    `targets` is strictly decreasing seconds. Each one is a RUNG -- a speed stage of `level` measured against
    that exact number. `record_seconds` is the human inbounds IL record, carried only so a report can print
    "x.xx x the record"; nothing in the rule reads it.
    """

    level: str
    targets: tuple[float, ...]
    record_seconds: float | None = None


class Rung(NamedTuple):
    """Which rung of the focus ladder is current: its 0-based position, its target, and how many there are."""

    index: int
    target: float
    total: int

    @property
    def number(self) -> int:
        """1-based, for a log line or a report -- "rung 1 of 10"."""
        return self.index + 1


def entry_target(entry: dict) -> float | None:
    """The `target_seconds` a history entry was measured against, or None. Never raises on an old entry."""
    return _num(entry.get("target_seconds"))


def rung_met(entries: Iterable[dict], target: float) -> bool:
    """Has a `"done"` round already achieved `target`?

    Two ways, and either is enough. A done round whose own target was AT OR UNDER this one has already proved
    the harder thing (today's 0-1 speed round at target 150 does NOT satisfy the 120 rung; a later one at 120
    satisfies the 120 rung and every easier one). A done round whose recorded MEDIAN was at or under this one
    has achieved it in fact even if it was aimed higher -- which is how a ladder skips rungs the level is
    already past instead of spending 8M steps re-proving them.
    """
    for entry in entries:
        if str(entry.get("status") or "") != "done":
            continue
        own = entry_target(entry)
        if own is not None and own <= target:
            return True
        median = valid_official_seconds(entry.get("median_time"))
        if median is not None and median <= target:
            return True
    return False


def focus_rung(plan: "Plan", state: "DriverState") -> Rung | None:
    """The rung the focus is on, or None when there is no focus or the whole ladder is done.

    Pure: the plan and the state, nothing on disk. `specialists_status.py` calls it directly, so the read-only
    report cannot disagree with the driver about which rung is being trained.
    """
    focus = plan.focus
    if focus is None:
        return None
    entries = state.entries_for((focus.level, SPEED))
    for index, target in enumerate(focus.targets):
        if not rung_met(entries, target):
            return Rung(index, target, len(focus.targets))
    return None


class StageSpec(NamedTuple):
    """One entry of the plan. `(level, kind)` is a stage's identity everywhere: names, state, history."""

    level: str
    kind: str = COMPLETE

    @property
    def run(self) -> str:
        return stage_run_name(self.level, self.kind)

    @property
    def key(self) -> tuple[str, str]:
        return (self.level, self.kind)


@dataclass
class Plan:
    stages: list[StageSpec]
    rule: StageRule
    env: dict
    train: dict
    speed: dict = field(default_factory=dict)     # per-kind overrides of the stage rule for a speed stage
    targets: dict[str, float] = field(default_factory=dict)  # per-level override of the S-rank target time
    target_scale: float = DEFAULT_TARGET_SCALE    # what a level's own S threshold is multiplied by (§8b)
    # A SPEED STAGE'S OWN REWARD WEIGHTS, merged over `env.rewards` for that kind and no other (2026-09-20).
    # Empty by default, so a plan written before this key -- and every test that loads one -- generates
    # exactly the config it always did, for both kinds.
    speed_rewards: dict[str, float] = field(default_factory=dict)
    # THE HOLD LINE (§8a): the level at which the ladder stops until everything before it is "done". `None`
    # lifts it. The lead's instruction on 2026-09-18 was "dont have it promote to 0-4 untell it gets better
    # times on these levels", and a stage that hits its step cap is recorded "unfinished" and walked past --
    # so without this key the ladder reaches 0-4 with exactly the slow policies the instruction forbids.
    hold_before: str | None = None
    # HOW THE HELD STAGES TAKE THEIR TURNS (§10). `round_robin` is the rule the line shipped with and stays
    # the code default, so a plan written before this key -- and every test that loads one -- means exactly
    # what it meant. `sequential` is DEPTH-FIRST: the first not-done stage in plan order runs round after
    # round until it is "done", and only then does the next one start. The user, 2026-09-19: "shouldnt we
    # just work on 0-1 untell its finished before working on the other levels".
    hold_order: str = ROUND_ROBIN
    # THE FOCUS / RECORD CHASE (§11, 2026-09-20). `None` -- the default, and what every plan written before
    # this key means -- is exactly today's behaviour: the hold line and `hold_order` decide everything. Set,
    # it overrides both: only that level's speed stage runs, one rung of its ladder at a time.
    focus: FocusPlan | None = None

    def focus_stage(self) -> StageSpec | None:
        """The stage the focus trains: its level's SPEED stage, or None when there is no focus."""
        return None if self.focus is None else StageSpec(self.focus.level, SPEED)

    def hold_index(self) -> int | None:
        """The plan position the hold line sits in front of, or None when there is no line.

        The line is named by a LEVEL, and it bites at that level's FIRST stage, so `hold_before: "Level 0-4"`
        blocks 0-4's complete stage and everything after it while leaving all six 0-1..0-3 stages runnable.
        """
        if not self.hold_before:
            return None
        return next((i for i, s in enumerate(self.stages) if s.level == self.hold_before), None)

    @property
    def order(self) -> list[str]:
        """The DISTINCT levels in plan order: what `full_run.py` plays, one episode per level."""
        seen: list[str] = []
        for stage in self.stages:
            if stage.level not in seen:
                seen.append(stage.level)
        return seen

    def rule_for(self, kind: str) -> StageRule:
        """The stage rule of one kind: the `stage:` block, with the `speed:` block laid over it for a speed stage."""
        if kind != SPEED or not self.speed:
            return self.rule
        return dataclasses.replace(self.rule, **self.speed)

    def target_for(self, level: str) -> float | None:
        """A per-level override of the target time, or None to read the level's own S-rank threshold live."""
        value = self.targets.get(level)
        return float(value) if value else None

    def index_of(self, level: str, kind: str = COMPLETE) -> int:
        try:
            return [s.key for s in self.stages].index((level, kind))
        except ValueError:
            raise ValueError("%r (%s) is not in the plan: %s"
                             % (level, kind, ", ".join("%s/%s" % s.key for s in self.stages))) from None


def _stage_spec(entry) -> StageSpec:
    """One plan entry: a bare level name (a complete stage) or `{level: ..., kind: speed}`."""
    if isinstance(entry, dict):
        level, kind = entry.get("level"), str(entry.get("kind", COMPLETE))
        if not level:
            raise ValueError("a plan entry has no `level`: %r" % (entry,))
        return StageSpec(str(level), kind)
    return StageSpec(str(entry), COMPLETE)


def load_plan(path: str | Path) -> Plan:
    """Reads `configs/specialists.yaml`. Every level is validated against the levels this build ships."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    stages = [_stage_spec(entry) for entry in data.get("order", [])]
    if not stages:
        raise ValueError("%s lists no levels under `order`" % path)
    unknown = [s.level for s in stages if s.level not in CAMPAIGN_LEVELS_SHIPPED]
    if unknown:
        raise ValueError("%s lists levels this build cannot load: %s" % (path, ", ".join(unknown)))
    bad_kinds = sorted({s.kind for s in stages} - set(STAGE_KINDS))
    if bad_kinds:
        raise ValueError("%s: unknown stage kinds %s (expected %s)" % (path, bad_kinds, list(STAGE_KINDS)))
    keys = [s.key for s in stages]
    if len(set(keys)) != len(keys):
        raise ValueError("%s lists the same (level, kind) twice" % path)
    known = {f.name for f in StageRule.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    speed = dict(data.get("speed", {}) or {})
    targets = {str(k): float(v) for k, v in (speed.pop("targets", {}) or {}).items()}
    scale = float(speed.pop("target_scale", DEFAULT_TARGET_SCALE))
    # A speed stage's own reward weights. Popped here, with `targets` and `target_scale`, because the block
    # below refuses every `speed:` key that is not a StageRule field -- and validated against RewardConfig for
    # the same reason that check exists: `EnvConfig.from_dict` drops a weight it does not know, so a misspelt
    # one would train at the shared weight while the plan file said otherwise.
    speed_rewards = {str(k): float(v) for k, v in (speed.pop("rewards", {}) or {}).items()}
    unknown_weights = sorted(set(speed_rewards) - {f.name for f in dataclasses.fields(RewardConfig)})
    if unknown_weights:
        raise ValueError("%s: unknown speed reward weights %s" % (path, unknown_weights))
    for name, block in (("stage", dict(data.get("stage", {}) or {})), ("speed", speed)):
        unexpected = sorted(set(block) - known)
        if unexpected:  # a misspelt knob would silently use its default, exactly as EnvConfig.from_dict would
            raise ValueError("%s: unknown %s settings %s" % (path, name, unexpected))
    hold = data.get("hold_before")
    hold = str(hold) if hold else None
    if hold is not None and hold not in {s.level for s in stages}:
        # A typo here would silently lift the hold line and let the ladder walk to 0-4 on slow policies, which
        # is the one thing this key exists to prevent. It is a hard error, not a warning.
        raise ValueError("%s: hold_before %r is not a level in the plan" % (path, hold))
    # An unknown `hold_order` is refused rather than defaulted, for the same reason a misspelt stage knob is:
    # `hold_order: sequental` would silently go back to round robin and quietly spread the machine across
    # three levels again, which is precisely what the key exists to stop.
    order_rule = str(data.get("hold_order") or ROUND_ROBIN)
    if order_rule not in HOLD_ORDERS:
        raise ValueError("%s: unknown hold_order %r (expected %s)" % (path, order_rule, list(HOLD_ORDERS)))
    plan = Plan(stages=stages, rule=StageRule(**dict(data.get("stage", {}) or {})), speed=speed, targets=targets,
                target_scale=scale, speed_rewards=speed_rewards, hold_before=hold, hold_order=order_rule,
                focus=_focus_plan(path, data.get("focus"), stages),
                env=dict(data.get("env", {}) or {}), train=dict(data.get("train", {}) or {}))
    return plan


FOCUS_KEYS = ("level", "targets", "record_seconds")


def _focus_plan(path, block, stages: list[StageSpec]) -> FocusPlan | None:
    """The `focus:` block, validated LOUDLY (§11). `None`/absent is today's behaviour and is not an error.

    Every check here is a hard error for the same reason `hold_before`'s typo check is: a focus that silently
    does nothing -- or that silently names the wrong level -- spends days of twelve games on the wrong thing
    with nobody watching, and there are no LLM monitors on this run by the user's own instruction.
    """
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError("%s: `focus:` must be a block with `level:` and `targets:`, not %r" % (path, block))
    unexpected = sorted(set(block) - set(FOCUS_KEYS))
    if unexpected:
        raise ValueError("%s: unknown focus settings %s (expected %s)" % (path, unexpected, list(FOCUS_KEYS)))
    level = str(block.get("level") or "")
    if not level:
        raise ValueError("%s: `focus:` has no `level:`" % path)
    if level not in {s.level for s in stages}:
        raise ValueError("%s: focus.level %r is not a level in the plan" % (path, level))
    if (level, SPEED) not in {s.key for s in stages}:
        # The focus trains that level's SPEED stage, and every name, path and index comes from the plan entry
        # for it. Without the entry `begin_stage` would raise on `index_of` at the moment the stage starts.
        raise ValueError("%s: focus.level %r has no `{level: %s, kind: speed}` entry under `order:`"
                         % (path, level, level))
    raw = block.get("targets") or []
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("%s: focus.targets must be a non-empty list of median seconds" % path)
    try:
        ladder = [float(value) for value in raw]
    except (TypeError, ValueError):
        raise ValueError("%s: focus.targets must all be numbers: %r" % (path, list(raw))) from None
    bad = [value for value in ladder if not (value > 0.0)]
    if bad:
        raise ValueError("%s: focus.targets must all be positive seconds: %s" % (path, bad))
    flat = [(a, b) for a, b in zip(ladder, ladder[1:]) if b >= a]
    if flat:
        # A ladder that does not get harder is either a typo or a rung that can never be reached by getting
        # faster, and `focus_rung` walks it assuming each target is strictly under the one before it.
        raise ValueError("%s: focus.targets must be strictly decreasing (a ladder toward the record); "
                         "these do not decrease: %s" % (path, flat))
    return FocusPlan(level=level, targets=tuple(ladder), record_seconds=_num(block.get("record_seconds")))


def order_rule_text(hold_order: str, levels: Iterable[str]) -> str:
    """One phrase for HOW the held stages take their turns. Shared with `specialists_status.py` verbatim.

    `levels` is the distinct levels still waiting, in plan order, so the depth-first phrase names the two the
    operator actually cares about: "depth-first: finishing Level 0-1 before Level 0-2".
    """
    names = list(dict.fromkeys(levels))
    if hold_order != SEQUENTIAL:
        return "round robin: fewest rounds first, ties in plan order"
    if len(names) >= 2:
        return "depth-first: finishing %s before %s" % (names[0], names[1])
    if names:
        return "depth-first: finishing %s" % names[0]
    return "depth-first: one stage at a time, in plan order"


def stage_config(plan: Plan, level: str, *, init_steps: int | None = None, kind: str = COMPLETE,
                 target_seconds: float | None = None) -> dict:
    """The real single-level training config for one stage: `{env: ..., train: ...}`, ready to write as YAML.

    `timesteps` is the run TOTAL in train.py, and every stage resumes from the previous specialist, so the
    count keeps climbing across the ladder: the budget has to be measured from where this stage STARTS. With
    no readable step count in the init file it falls back to the plan's own `timesteps`, or to the cap plus
    slack, either of which only has to exceed what the stage rule will allow.

    A SPEED stage is the same config with `speed_bonus` on, plus whatever the plan's `speed.rewards:` block
    overrides (2026-09-20: `death: 12.0`). No observation or action changes, and a reward weight is not a
    policy parameter, so the level's own specialist still loads into it unchanged. The override is REBOUND,
    never mutated in place -- see the comment at the merge. `speed_target_seconds` is set only when the
    plan overrides it for this level; 0 tells the env to read the level's own S-rank time live and scale it by
    `speed_target_scale`. The scale is written into every speed stage's config even when it is the default, so
    a generated file says what the run was actually measured against.
    """
    rule = plan.rule_for(kind)
    env = {key: value for key, value in plan.env.items() if key not in MULTI_LEVEL_KEYS}
    env["level"] = level
    if kind == SPEED:
        env["speed_bonus"] = True
        env["speed_target_scale"] = float(plan.target_scale)
        # EVERY EPISODE IS A FRESH LEVEL LOAD on a speed stage (2026-09-18 review). The bonus is scaled only on
        # a fresh-start completion -- a checkpoint respawn begins partway through the level with the clock
        # already running, so its "official time" says nothing -- which means that at `fresh_start_prob: 0.2` a
        # respawn completion pays the full 100 while the fresh completions the stage is actually SCORED on
        # (`fresh_completion_rate`, `median_time_50`, `best_time` are all fresh-only) pay the scaled 39-100.
        # `fresh_start` is not in the observation, so the policy cannot tell the two apart at the same state
        # near the exit: PPO would fit one baseline across both and hand every fresh completion -- the one
        # event the stage exists to reinforce -- a systematically negative advantage. A stage whose whole
        # subject is the WHOLE-LEVEL clock has no business training on episodes that start two thirds of the
        # way through the level anyway. Deaths still respawn at checkpoints INSIDE an episode; only the
        # episode's own start is forced.
        env["fresh_start_prob"] = 1.0
        if plan.speed_rewards:
            # REBIND, NEVER MUTATE. `env` above is a SHALLOW copy of `plan.env`, so `env["rewards"]` IS the
            # plan's own dict: an in-place `.update()` here would leak this stage's weights into every
            # complete stage generated afterwards from the same Plan object -- and the pin that a complete
            # stage's weights are `campaign_gates_full.yaml`'s would not catch it, because both dicts would
            # be the one mutated object and would still compare equal. Pinned by the order-independence test
            # in tests/test_speed_death_weight.py.
            env["rewards"] = {**env["rewards"], **plan.speed_rewards}
        # A FOCUS RUNG's target wins over the plan's per-level override, which wins over reading the level's
        # own S-rank time live (§11). All three arrive at the env down the ONE path, `speed_target_seconds`,
        # which the env reports back as `info["target_seconds"]` -- so the reward and the promotion rule are
        # measured against the same number whichever of the three set it.
        target = target_seconds if target_seconds else plan.target_for(level)
        if target:
            env["speed_target_seconds"] = float(target)
    train = dict(plan.train)
    train["run_name"] = stage_run_name(level, kind)
    budget = rule.max_steps_per_stage + TIMESTEPS_SLACK
    train["timesteps"] = int(init_steps) + budget if init_steps is not None else int(train.get("timesteps", budget))
    return {"env": env, "train": train}


def write_stage_config(plan: Plan, level: str, cwd: Path, *, init_steps: int | None = None,
                       generated_dir: str = GENERATED_DIR, kind: str = COMPLETE,
                       target_seconds: float | None = None) -> Path:
    path = cwd / stage_config_path(level, generated_dir, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# GENERATED by scripts/campaign_driver.py from configs/specialists.yaml -- do not edit.\n"
              "# Stage: %s (%s), run %s%s. Edit the plan file and let the driver rewrite this.\n"
              % (level, kind, stage_run_name(level, kind),
                 (", FOCUS rung at %.2f s" % target_seconds) if target_seconds else ""))
    path.write_text(header + yaml.safe_dump(stage_config(plan, level, init_steps=init_steps, kind=kind,
                                                         target_seconds=target_seconds),
                                            sort_keys=False),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The stage rule (pure)
# ---------------------------------------------------------------------------


class StageSample(NamedTuple):
    """Everything the stage rule reads, all of it from files two read-only helpers already maintain."""

    timesteps: float | None      # runs/<run>/status.json
    fresh_rate: float | None     # ... campaign.fresh_completion_rate, over the last 50 fresh episodes
    fresh_window: int            # ... campaign.fresh_window, how many of them there are
    # ... campaign.best_time, the fastest fresh-start completion EVER recorded by this run. It is a lifetime
    # minimum: `ProgressCallback` only ever lowers it and `_restore` carries it across every trainer restart
    # and every round. It is REPORTED and never gated on -- see `median_time`.
    best_time: float | None
    best_at: float | None        # models/<run>/best.json's at_timesteps: when keep_best last moved best.zip
    # ... campaign.target_seconds: the target the ENV computed live off the campaign block -- the level's own
    # S-rank threshold times `speed_target_scale`, or the plan's per-level override. None on a complete stage
    # and on any run older than 2026-09-18, which is why it carries a default.
    target_seconds: float | None = None
    # ... campaign.s_rank_seconds: the UNSCALED threshold behind it. Recorded in the sidecar, never compared
    # against: the promotion rule only ever reads `target_seconds` (§8b).
    s_rank_seconds: float | None = None
    # ... campaign.median_time_50: the MEDIAN official time over the completions in the last 50 fresh episodes.
    # THE CLOCK CLAUSE OF THE SPEED RULE READS THIS AND NOT `best_time` (2026-09-18 review). `best_time` is a
    # run-lifetime minimum: one lucky load satisfies it forever, across restarts and across rounds, and the
    # live runs show the gap is not theoretical -- spec_0-1 best 243.4 s against a median of 490.9 s, spec_0-2
    # best 139.5 s against 236.5 s, the run's single best about half its typical completion in both cases.
    # Promoting on that would open the hold line on a policy whose typical time is twice the target, which is
    # precisely what the line exists to prevent. The median is a property of the policy that is running now.
    median_time: float | None = None


EMPTY_SAMPLE = StageSample(None, None, 0, None, None)


def read_sample(status_path: Path, best_json: Path) -> StageSample:
    """One `StageSample` from disk. A missing or half-written file reads as "nothing known yet", never raises."""
    timesteps = rate = best_time = best_at = target = s_rank = median = None
    window = 0
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if isinstance(status, dict):
        timesteps = _num(status.get("timesteps"))
        campaign = status.get("campaign")
        if isinstance(campaign, dict):
            rate = _num(campaign.get("fresh_completion_rate"))
            window = int(_num(campaign.get("fresh_window")) or 0)
            # The two CLOCKS go through the shared predicate, never `_num`: a status.json written before the
            # 2026-09-19 fix (or by an older trainer still running) can carry an impossible time, and the
            # speed rule promotes when `median_time <= target_seconds` -- a zero median would latch instantly.
            # An invalid clock reads as "not measured yet", which the rule already handles: it cannot latch.
            best_time = valid_official_seconds(campaign.get("best_time"))
            target = _num(campaign.get("target_seconds"))
            s_rank = _num(campaign.get("s_rank_seconds"))
            median = valid_official_seconds(campaign.get("median_time_50"))
    try:
        best = json.loads(best_json.read_text(encoding="utf-8"))
        best_at = _num(best.get("at_timesteps")) if isinstance(best, dict) else None
    except (OSError, ValueError):
        best_at = None
    return StageSample(timesteps, rate, window, best_time, best_at, target, s_rank, median)


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def stage_verdict(sample: StageSample, start_steps: float, target_reached_at: float | None,
                  rule: StageRule, *, kind: str = COMPLETE,
                  target_seconds: float | None = None) -> tuple[str, float | None]:
    """`("running" | "done" | "unfinished", target_reached_at)` -- the whole stage rule, and nothing else.

    `target_reached_at` is carried in the driver state and returned updated, so the latch survives a restart of
    the driver. See the module docstring for why the target latches and why the settle counts from the later of
    the two moments.

    On a SPEED stage the latch also needs the clock: the MEDIAN official time over the completions in the last
    50 fresh episodes at or under `target_seconds`. It is deliberately not `best_time`, which is the run's
    lifetime minimum and is satisfied forever by a single lucky load (see `StageSample.median_time`): a stage
    is fast when the policy is typically fast, not when it once was. With no target read yet, or no median yet
    (a stage that has never completed the level), the latch can never fire, so the stage runs to its cap and is
    recorded "unfinished" -- loud and safe, and never a silent promotion on the rate alone, which is the exact
    thing the lead's 2026-09-18 instruction forbids.
    """
    steps, reached = sample.timesteps, target_reached_at
    fast_enough = (kind != SPEED or (target_seconds is not None and sample.median_time is not None
                                     and sample.median_time <= target_seconds))
    if (reached is None and steps is not None and sample.fresh_rate is not None
            and sample.fresh_window >= rule.min_fresh_window and sample.fresh_rate >= rule.target_rate
            and fast_enough):
        reached = steps
    if reached is not None and steps is not None:
        hold_from = max(reached, sample.best_at) if sample.best_at is not None else reached
        if steps - hold_from >= rule.settle_steps:
            return "done", reached
    if steps is not None and steps - start_steps >= rule.max_steps_per_stage:
        return "unfinished", reached
    return "running", reached


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promotion_source(model_dir: Path, zip_steps: Callable[[Path], int | None] = supervise.zip_timesteps
                     ) -> tuple[Path | None, str]:
    """`(checkpoint, source_kind)` to promote: `best.zip` when keep_best saved one, else the newest checkpoint.

    `source_kind` is "best"/"newest"/"none" -- WHICH FILE was chosen. It is not a STAGE kind; the two were
    briefly both called `kind` inside `promote`, and the stage kind lost.

    "Newest" is `supervise.choose_resume`'s rule, not the file's mtime and not `latest.zip` on trust: a killed
    trainer leaves `latest.zip` stale, so the step count is read out of each zip and the largest wins.
    """
    best = model_dir / "best.zip"
    if best.exists():
        return best, "best"
    resume, _ = supervise.choose_resume(model_dir, zip_steps)
    return (resume, "newest") if resume is not None else (None, "none")


def refuse_promotion(models_dir: Path, level: str, *, kind: str, status: str) -> str:
    """Why this stage must NOT overwrite the level's promoted specialist, or `""` when it may.

    A complete stage promotes whatever it produced, exactly as it always has: before speed stages every level
    was promoted at most once, so there was never a file to regress. A SPEED stage runs on a level that ALREADY
    HAS a specialist, and `models/specialists/` is the only copy of it that is committed to git -- so a speed
    round that burns its 8M-step cap without ever beating the clock must not copy `keep_best`'s pick over a
    policy that was promoted for finishing the level. Nothing anywhere compares the two, and the weights are
    not lost by declining: they stay in the stage's own model directory, which is where `round_init` resumes a
    later round from (2026-09-18 review).
    """
    if kind != SPEED or status == "done":
        return ""
    if not specialist_path(models_dir, level).exists():
        return ""  # nothing to regress: the level has no specialist at all, so any weights beat none
    return ("it ended %r and %s already holds a promoted specialist"
            % (status, specialist_path(models_dir, level).name))


def promote(model_dir: Path, models_dir: Path, level: str, *, sample: StageSample, status: str,
            start_steps: float, difficulty: int, run: str, kind: str = COMPLETE,
            target_seconds: float | None = None, s_rank_seconds: float | None = None, round: int = 1,
            rung: int | None = None,
            zip_steps: Callable[[Path], int | None] = supervise.zip_timesteps,
            copy: Callable[[Path, Path], object] = shutil.copy2) -> tuple[Path | None, dict]:
    """Copies the stage's checkpoint to `models/specialists/<level>.zip` and writes its JSON sidecar.

    A speed stage writes to the SAME path, so a SUCCESSFUL one overwrites the specialist its own level's
    complete stage promoted: one policy per level is the whole design, and the speed run is that policy made
    faster. One that ended `"unfinished"` declines instead -- see `refuse_promotion`. The sidecar's `mode` is
    how `full_run.py` and `specialists_status.py` tell which stage produced the file.
    """
    source, source_kind = promotion_source(model_dir, zip_steps)
    if source is None:
        return None, {}
    destination = specialist_path(models_dir, level)
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy(source, destination)
    sidecar = {
        "level": level,
        "run": run,
        "mode": kind,                           # "complete" (the rate rule) or "speed" (the rate AND the clock)
        "target_seconds": target_seconds,       # what a speed stage was actually measured against
        "s_rank_seconds": s_rank_seconds,       # ... and the raw S threshold it was scaled from (§8b)
        "round": round,                         # which attempt at this stage produced the file (§8a)
        "rung": rung,                           # 0-based position on the focus ladder, or None (§11)
        "status": status,                       # "done" (the target rate was reached) or "unfinished" (the cap)
        "fresh_completion_rate": sample.fresh_rate,
        "fresh_window": sample.fresh_window,
        "best_time": sample.best_time,            # the run's lifetime minimum: reported, never gated on
        "median_time": sample.median_time,        # ... and the statistic the speed rule actually promoted on
        "timesteps": sample.timesteps,
        "stage_steps": (sample.timesteps - start_steps) if sample.timesteps is not None else None,
        "source_checkpoint": source.name,
        "source_kind": source_kind,             # "best": keep_best's peak; "newest": no best.zip existed
        "difficulty": difficulty,
        "env_config": (model_dir / "env_config.yaml").as_posix(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    destination.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return destination, sidecar


# ---------------------------------------------------------------------------
# The END_STAGE control file (pure)
# ---------------------------------------------------------------------------


def decode_control_file(raw: bytes) -> str:
    """The text of a control file PowerShell wrote, whatever encoding it chose. Never raises.

    THIS IS NOT PEDANTRY (found by the 2026-09-19 scratch dry-run). `Set-Content -Encoding utf8` writes a
    BYTE ORDER MARK, and `read_text("utf-8")` keeps it, so `END_STAGE` holding "Level 0-2 speed" parsed as the
    level `"\\ufeffLevel 0-2"` and a perfectly good request was refused. Windows PowerShell 5.1's `>` and
    `Out-File` can write UTF-16 as well, which is not valid UTF-8 at all and would raise where the caller
    expects "no text".
    """
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return ""


def end_stage_request(text: str) -> tuple[str | None, str | None]:
    """What `runs/specialists/END_STAGE`'s text names: `(level, kind)`, either `None` for "do not care".

    An empty file means "the stage that is running, whatever it is" -- the common case, and the one the
    command in `docs/commands.md` writes. Naming the stage is the safe form:

        Level 0-2 speed        Level 0-2/speed        Level 0-2, speed        Level 0-2        speed

    Anything after the first non-comment line is ignored, so a file may carry a note about why it was made.
    """
    clean = text.replace("﻿", "").replace("\x00", "")  # a BOM decoded by hand, and UTF-16 read as bytes
    line = next((raw.strip() for raw in clean.splitlines()
                 if raw.strip() and not raw.strip().startswith("#")), "")
    tokens = line.replace("/", " ").replace(",", " ").split()
    kind = None
    if tokens and tokens[-1].lower() in STAGE_KINDS:
        kind = tokens.pop().lower()
    level = " ".join(tokens) or None
    return level, kind


def end_stage_objection(stage: Stage, text: str) -> str:
    """Why this END_STAGE file must NOT end `stage`, or `""` when it may.

    A file left behind from an earlier stage -- written, forgotten, or dropped while the driver was between
    stages -- would otherwise end whatever happens to be running when the driver next polls, which is exactly
    the accident the naming form exists to prevent.
    """
    level, kind = end_stage_request(text)
    if level is not None and level != stage.level:
        return ("it names %r and the running stage is %s (%s)" % (level, stage.level, stage.kind))
    if kind is not None and kind != stage.kind:
        return ("it names the %s stage of %s and the running stage is the %s one"
                % (kind, stage.level, stage.kind))
    return ""


# ---------------------------------------------------------------------------
# Driver state
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    level: str
    run: str
    index: int
    init: str                              # the checkpoint this stage resumed from
    start_steps: float = 0.0               # the trainer's step count when the stage began
    started_at: float = 0.0
    target_reached_at: float | None = None  # the latch of the stage rule
    # A stage written before kinds existed has none of these fields, and every default is what it was: the
    # first (and only) complete round of a stage with no clock to beat. That is the whole on-disk migration,
    # and `tests/test_campaign_driver.py` pins it against a copy of the REAL live state file.
    kind: str = COMPLETE
    target_seconds: float | None = None    # the target the run reported, once it has reported one
    s_rank_seconds: float | None = None    # ... and the raw S threshold behind it, for the record only
    # Which attempt at this (level, kind) this is. 1 for every stage the ladder walks through once; a second
    # and later round only happens while the HOLD LINE is up (§8a), when a stage that ended "unfinished" is
    # given another budget instead of the ladder moving on to a level it is not allowed to reach yet.
    round: int = 1
    # WHICH RUNG of the focus ladder this stage is (§11, 2026-09-20), 0-based, or None when the driver is not
    # focusing. `target_seconds` above carries the rung's actual target; this is only its position, so a log
    # line and a report can say "rung 3 of 10" and `stage_blocked` can count rounds of THIS rung rather than
    # of every rung the level has ever run. None on every stage written before the focus existed, which is
    # exactly right for them: they were measured against the S-rank time, not against a ladder.
    rung: int | None = None
    # Samples at or below this step count belong to an EARLIER ROUND of this stage and are ignored (2026-09-18
    # review). A round reuses the run directory, so `runs/<run>/status.json` still holds the previous round's
    # final numbers -- the same rate, the same window, the same lifetime `best_time` -- until the new trainer
    # overwrites it, and the latch was just reset. Without this the first tick of round N re-latches on round
    # N-1's tail and the stage is recorded "done" a settle later on the very data the cap had just rejected,
    # which makes the cap meaningless. Set by `begin_stage` from whatever status.json says at that moment;
    # None on every stage written before the review, which is exactly right for them.
    stale_below: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @property
    def key(self) -> tuple[str, str]:
        return (self.level, self.kind)

    @classmethod
    def from_dict(cls, data: dict) -> "Stage":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DriverState:
    current: Stage | None = None
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": STATE_VERSION,
                "current": self.current.to_dict() if self.current else None,
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, data: dict) -> "DriverState":
        current = data.get("current")
        return cls(current=Stage.from_dict(current) if isinstance(current, dict) else None,
                   history=[h for h in data.get("history", []) if isinstance(h, dict)])

    @classmethod
    def load(cls, path: Path) -> "DriverState":
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        write_json_atomic(path, self.to_dict())

    def finished_levels(self) -> set[str]:
        return {h["level"] for h in self.history if isinstance(h.get("level"), str)}

    def finished_stages(self) -> set[tuple[str, str]]:
        """The `(level, kind)` of every stage already handled. A history entry with no `kind` was a complete one."""
        return {(h["level"], str(h.get("kind") or COMPLETE))
                for h in self.history if isinstance(h.get("level"), str)}

    def entries_for(self, key: tuple[str, str]) -> list[dict]:
        """Every history entry for one `(level, kind)`, oldest first: one per ROUND of that stage."""
        return [h for h in self.history
                if isinstance(h.get("level"), str) and (h["level"], str(h.get("kind") or COMPLETE)) == key]

    def rounds(self, key: tuple[str, str]) -> int:
        """How many rounds of this stage have already ENDED. The next one is this plus one."""
        return len(self.entries_for(key))

    def rung_rounds(self, key: tuple[str, str], target: float | None) -> int:
        """How many rounds of this stage were measured against `target` (§11). `None` falls back to `rounds`.

        A focus ladder runs many rounds of ONE (level, speed) stage, each against a different target, so a
        round cap has to be per RUNG -- otherwise the tenth rung would start already out of rounds because of
        the nine before it. The shipped plan sets `max_rounds: 0` (unbounded), so this only bites if someone
        turns a cap back on; it must still be the right count when they do.
        """
        if target is None:
            return self.rounds(key)
        return len([h for h in self.entries_for(key) if entry_target(h) == target])

    def stage_status(self, key: tuple[str, str]) -> str | None:
        """The status of the LATEST round of this stage, or None when it has never run.

        The hold line reads this and nothing else: only `"done"` counts as satisfied. `"unfinished"` is a stage
        that ran out of steps, which is exactly the case the line exists to catch, and `"skipped"` is an
        operator's explicit `--start-at` decision, which the line respects rather than second-guessing.
        """
        entries = self.entries_for(key)
        return str(entries[-1].get("status") or "") if entries else None

    def reconcile(self, plan: "Plan") -> list[str]:
        """Matches this state to `plan` by `(level, kind)` and re-numbers the indices. Returns what it could not place.

        Inserting stages into the plan renumbers everything after them, and the driver was LIVE on stage 3 of 30
        when the three speed stages were added: without this the running stage's stored `index` would print the
        wrong position and `--start-at` would skip the wrong prefix. Nothing is dropped and nothing is renamed --
        a `(level, kind)` the plan no longer lists keeps the index it was saved with and is reported here.
        """
        positions = {s.key: i for i, s in enumerate(plan.stages)}
        missing: list[str] = []
        seen: dict[tuple, int] = {}
        for entry in self.history:
            key = (entry.get("level"), str(entry.get("kind") or COMPLETE))
            entry.setdefault("kind", key[1])
            # Round numbers are filled in for entries written before rounds existed, in history order, so
            # `specialists_status.py` can count them without knowing which driver version wrote which row.
            seen[key] = seen.get(key, 0) + 1
            entry.setdefault("round", seen[key])
            if key in positions:
                entry["index"] = positions[key]
            elif isinstance(key[0], str):
                missing.append("%s/%s" % key)
        if self.current is not None:
            if self.current.key in positions:
                self.current.index = positions[self.current.key]
            else:
                missing.append("%s/%s (current)" % self.current.key)
        return missing


def stage_blocked(plan: Plan, state: DriverState, spec: StageSpec, models_dir: Path) -> str:
    """Why this stage cannot start yet, or `""` when it can. Pure: the plan, the state and one directory.

    Only a SPEED stage is ever blocked, and only by its own level: it resumes from that level's promoted
    specialist, so a level whose complete stage has not finished has nothing for it to resume FROM. Starting it
    anyway would either train the clock on a policy that cannot reach the exit or, worse, silently pick up some
    other level's weights.

    `Driver.stage_blocked` is this function, and `specialists_status.py` calls it directly, so the read-only
    report cannot disagree with the driver about which waiting stage is actually able to run (2026-09-19
    review: under `sequential` a blocked stage hands the machine to a LATER level, which is the one thing the
    depth-first instruction forbids, and it must not be invisible to the daily check).
    """
    cap = plan.rule_for(spec.kind).max_rounds
    # Under a FOCUS the rounds are counted per RUNG: the ladder deliberately runs this one stage over and
    # over, and a cap meant for "this stage cannot reach its target" must not fire because of rungs already
    # passed. `focus_rung` is None once the ladder is done, and then this is the ordinary count again.
    rung = focus_rung(plan, state) if plan.focus is not None and spec.key == (plan.focus.level, SPEED) else None
    spent = state.rung_rounds(spec.key, rung.target if rung is not None else None)
    if cap > 0 and spent >= cap:
        # Only ever reachable under the hold line or a focus: the ladder itself runs each stage once.
        return "it has had its %d rounds and is still not done" % cap
    if spec.kind != SPEED:
        return ""
    if state.stage_status((spec.level, COMPLETE)) != "done":
        return "its complete stage is not done yet"
    if not specialist_path(models_dir, spec.level).exists():
        return "no specialist file for the level yet"
    return ""


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------


class StageSupervisor(supervise.Supervisor):
    """`supervise.Supervisor` for one stage, with `post_times.py` added to the helper set.

    Everything else -- the health test, the hung/dead/draining split, the sick report, the kill, the boot gate,
    the resume-file rule, the restart budget -- is inherited unchanged. This class exists so the driver reuses
    that machinery per stage rather than owning a second copy of it.
    """

    # Which `keep_best.py` metric protects this stage's best.zip. A complete stage scores the fresh completion
    # RATE; a speed stage scores the CLOCK, gated on a minimum rate so one lucky load cannot set the record.
    # Set by `Driver.supervisor_for` from the stage's kind.
    metric: str = "campaign"
    min_rate: float = 0.3

    def helper_specs(self) -> list[tuple[str, list[str], str]]:
        base = []
        for script, args, log in super().helper_specs():
            if Path(script).name == "keep_best.py" and self.metric != "campaign":
                args = ["--run", self.cfg.run, "--metric", self.metric, "--min-rate", str(self.min_rate)]
            base.append((script, args, log))
        return base + [
            # A specialist's fastest fresh-start completion is a real level time; posting it as it happens is
            # what keeps times.md current without an eval run and without a free bridge port. It commits
            # times.md ALONE, so a live trainer's checkpoints are never staged.
            ("scripts/post_times.py", ["--run", self.cfg.run, "--watch", "600", "--push"],
             "%s_post_times.log" % self.cfg.run),
            # The live dashboard follows the stage's run, so it is started (and restarted) per stage like the
            # other helpers instead of being left behind at a stage boundary (the user noticed it missing).
            ("scripts/dashboard.py", ["--run", self.cfg.run, "--monitor", str(self.cfg.monitor)],
             "%s_dashboard.log" % self.cfg.run)]


@dataclass
class DriverConfig:
    plan_path: str = "configs/specialists.yaml"
    count: int = 12
    monitor: int = 1
    poll_seconds: float = 60.0
    start_grace_seconds: float = 900.0
    stale_seconds: float = 900.0
    max_restarts_per_hour: int = 3
    base_port: int = 47800
    python: str = sys.executable
    cwd: Path = field(default_factory=Path.cwd)
    runs_dir: str = "runs"
    models_dir: str = "models"
    generated_dir: str = GENERATED_DIR
    dry_run: bool = False
    no_steam: bool = True


class Driver:
    """One `tick()` per poll: keep the current stage healthy, and end it when the stage rule says so.

    Every side effect is an injected callable, exactly as in `supervise.Supervisor`, so the tests drive the
    whole ladder -- stage start, promotion, the next stage -- against fakes and never launch a game, start a
    trainer or sleep.
    """

    def __init__(self, cfg: DriverConfig, plan: Plan, *,
                 processes: Callable[[], list[supervise.Proc]] = supervise.list_processes,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 probe: Callable[[Path, float], supervise.Status] = supervise.probe_status,
                 kill: Callable[[int], None] = supervise.kill_tree,
                 spawn: Callable[[str, Path], int] = supervise.spawn_detached,
                 stop_games: Callable[[], None] | None = None,
                 launch_games: Callable[[int, int], bool] | None = None,
                 zip_steps: Callable[[Path], int | None] = supervise.zip_timesteps,
                 working_sets: Callable[[], dict[int, int]] | None = None,
                 port_pids: Callable[[], dict[int, int]] | None = None,
                 relaunch_one: Callable[[int], bool] | None = None,
                 established: Callable[[Iterable[int]], dict[int, int]] | None = None,
                 cpu: Callable[[], dict[int, float]] | None = None,
                 copy: Callable[[Path, Path], object] = shutil.copy2,
                 pid: int | None = None):
        self.cfg, self.plan = cfg, plan
        self._injected = dict(processes=processes, now=now, sleep=sleep, probe=probe, kill=kill, spawn=spawn,
                              stop_games=stop_games, launch_games=launch_games, zip_steps=zip_steps,
                              working_sets=working_sets, port_pids=port_pids, relaunch_one=relaunch_one,
                              established=established, cpu=cpu, pid=pid)
        self.processes, self.now, self.sleep = processes, now, sleep
        self.kill, self.spawn, self.zip_steps, self.copy = kill, spawn, zip_steps, copy
        self.pid = pid

        self.run_dir = cfg.cwd / cfg.runs_dir / DRIVER_RUN
        self.state_path = self.run_dir / "driver_state.json"
        self.pause_path = self.run_dir / "DRIVER_PAUSE"
        # The operator's "end this stage now" file. DRIVER_PAUSE is checked FIRST in `tick`, so a paused
        # driver does nothing at all with this one -- including deleting it.
        self.end_stage_path = self.run_dir / END_STAGE_FILE
        self.log_path = cfg.cwd / cfg.runs_dir / "specialists_driver.log"
        self.state = DriverState.load(self.state_path)
        # The plan is re-read on every start, so stages can be inserted into it; the state is matched to the
        # new plan by (level, kind) before anything else runs.
        self.unplanned = self.state.reconcile(plan)
        self.sup: StageSupervisor | None = None
        # One "what is it doing now" key per SLOT. `log_once` logs when a slot's key CHANGES, so two lines
        # written in the same tick need two slots: sharing one made their keys alternate and both were
        # re-logged every single poll, forever (2026-09-19 review, the stuck-END_STAGE case).
        self._slots: dict[str, str] = {}

    # -- logging (the supervisor's own rule: stdout AND the file, needing only one to work) ---------

    def log(self, message: str) -> None:
        line = "%s [driver] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def log_once(self, key: str, message: str, *, slot: str = "state") -> None:
        """Logs `message` only when `key` differs from the last key logged in `slot`.

        The default slot is the driver's running commentary -- paused / holding / this stage's health -- where
        one line at a time is the whole point. A line that is written in the SAME tick as one of those needs
        its own slot, or the two keys alternate and `log_once` degrades into `log`.
        """
        if self._slots.get(slot) != key:
            self.log(message)
        self._slots[slot] = key

    # -- per-stage paths and the supervisor behind the current stage --------------------------------

    def model_dir(self, level: str, kind: str = COMPLETE) -> Path:
        return self.cfg.cwd / self.cfg.models_dir / stage_run_name(level, kind)

    def stage_run_dir(self, level: str, kind: str = COMPLETE) -> Path:
        return self.cfg.cwd / self.cfg.runs_dir / stage_run_name(level, kind)

    def supervisor_for(self, stage: Stage) -> StageSupervisor:
        if self.sup is not None and self.sup.cfg.run == stage.run:
            return self.sup
        config = supervise.Config(
            run=stage.run, config=stage_config_path(stage.level, self.cfg.generated_dir, stage.kind),
            count=self.cfg.count, monitor=self.cfg.monitor, stale_seconds=self.cfg.stale_seconds,
            poll_seconds=self.cfg.poll_seconds, max_restarts_per_hour=self.cfg.max_restarts_per_hour,
            start_grace_seconds=self.cfg.start_grace_seconds, base_port=self.cfg.base_port,
            python=self.cfg.python, cwd=self.cfg.cwd, runs_dir=self.cfg.runs_dir,
            models_dir=self.cfg.models_dir, dry_run=self.cfg.dry_run, no_steam=self.cfg.no_steam)
        self.sup = StageSupervisor(config, **self._injected)
        # A speed stage's best.zip is scored on the clock, not the rate, so its keep_best runs --metric time.
        self.sup.metric = "time" if stage.kind == SPEED else "campaign"
        return self.sup

    def current_sample(self, stage: Stage) -> StageSample:
        """This stage's numbers from disk, with an EARLIER ROUND's tail filtered out (`Stage.stale_below`).

        EVERY reader goes through here (2026-09-19 review). `tick` filtered and `end_stage_now` did not, so a
        stage ended by the operator inside the stale window -- up to the whole `start_grace_seconds` wide, at
        every round boundary -- recorded the PREVIOUS round's rate, median and best as its own, and on a
        COMPLETE stage `promote` wrote those numbers into the committed `models/specialists/<level>.json`.
        Nothing decides on them, so no weights moved; the damage was a false record in git.
        """
        sample = read_sample(self.stage_run_dir(stage.level, stage.kind) / "status.json",
                             self.model_dir(stage.level, stage.kind) / "best.json")
        if (stage.stale_below is not None and sample.timesteps is not None
                and sample.timesteps <= stage.stale_below):
            # An earlier round of this stage wrote that file and the new trainer has not caught up with it yet.
            # Reading it would let the round re-latch on numbers the previous round's cap had just rejected.
            self.log_once("stale:%s:%s" % (stage.level, stage.kind),
                          "%s (%s): ignoring round %d's own status.json until the trainer passes %s steps"
                          % (stage.level, stage.kind, stage.round - 1, "{:,.0f}".format(stage.stale_below)),
                          slot="sample")
            return EMPTY_SAMPLE
        return sample

    # -- starting a stage ---------------------------------------------------------------------------

    def next_stage(self) -> StageSpec | None:
        """The first plan stage with no history entry for its `(level, kind)`, or None when the ladder is done."""
        done = self.state.finished_stages()
        return next((stage for stage in self.plan.stages if stage.key not in done), None)

    def next_level(self) -> str | None:
        """The level of `next_stage`, or None. Kept for readers that only care which level comes next."""
        stage = self.next_stage()
        return stage.level if stage is not None else None

    def stage_blocked(self, spec: StageSpec) -> str:
        """Why this stage cannot start yet, or `""` when it can -- the module-level rule, on this driver."""
        return stage_blocked(self.plan, self.state, spec, self.cfg.cwd / self.cfg.models_dir)

    def focus_rung(self) -> Rung | None:
        """The rung of the focus ladder to train now, or None (no focus, or every rung is done)."""
        return focus_rung(self.plan, self.state)

    def rung_for(self, spec: StageSpec) -> Rung | None:
        """The focus rung `spec` would be started as, or None when it is an ordinary stage.

        `choose_stage` keeps its three-value shape (every caller and every test unpacks it), so the rung is
        asked for separately by the two places that actually begin a stage.
        """
        if self.plan.focus is None or spec.key != (self.plan.focus.level, SPEED):
            return None
        return self.focus_rung()

    def held_by(self) -> list[StageSpec]:
        """The stages in front of the hold line that are not `"done"`, in plan order. Empty = the line is open.

        THE HOLD LINE (§8a). `hold_before: "Level 0-4"` means no stage at or after 0-4 starts while any stage
        before it is still unfinished or unrun. Without it a stage that burns its 8M-step cap is recorded
        `"unfinished"` and the ladder walks on to 0-4 regardless -- which is exactly what the lead's
        "dont have it promote to 0-4 untell it gets better times on these levels" forbids.
        """
        hold = self.plan.hold_index()
        if hold is None:
            return []
        return [s for s in self.plan.stages[:hold]
                if self.state.stage_status(s.key) not in ("done", "skipped")]

    def order_rule_text(self) -> str:
        """How this driver's hold line spends its rounds, in one phrase, naming the levels it is on."""
        return order_rule_text(self.plan.hold_order, [s.level for s in self.held_by()])

    def choose_stage(self) -> tuple[StageSpec | None, list[StageSpec], str]:
        """`(the stage to start, the stages the hold line is waiting on, why nothing can start)`.

        With the line open (or absent) this is the ladder's own order, skipping a speed stage whose level is
        not ready for one. With the line up, `plan.hold_order` decides between the stages still in front of it:

          * `round_robin` (the default, and what the line shipped with): fewest ended rounds first, ties in
            plan order, so every not-done stage gets a fresh step budget before any of them gets a second.
          * `sequential` (§10, the user's 2026-09-19 instruction): DEPTH-FIRST -- the FIRST not-done stage in
            plan order that can run, and the same one again next time, until it is `"done"`. A blocked stage
            is passed over only because nothing in front of it can run; the reason is logged.

        A FOCUS (§11) overrides both: while one is set and its ladder still has a rung left, the ONLY stage
        this ever returns is that level's speed stage, whatever the hold line or the plan order would say.
        When the ladder is exhausted the focus stops -- loudly -- and the rules below take over again.

        Either way a stage picked for another round resumes from its OWN newest weights (`round_init`).
        """
        focus = self.plan.focus
        if focus is not None:
            spec = StageSpec(focus.level, SPEED)
            rung = self.focus_rung()
            if rung is None:
                self.log_once("focus_done:%s" % focus.level,
                              "FOCUS COMPLETE: every one of the %d rungs for %s is done (the last was %.2f s). "
                              "The focus is over; the ordinary plan decides from here. Clear `focus:` in %s, "
                              "or add faster rungs to keep chasing."
                              % (len(focus.targets), focus.level, focus.targets[-1], self.cfg.plan_path),
                              slot="focus")
            else:
                why = self.stage_blocked(spec)
                if why:
                    return None, [spec], "FOCUS %s (%s): %s" % (spec.level, spec.kind, why)
                self.log_once("focus:%s@%g" % (focus.level, rung.target),
                              "FOCUS on %s: rung %d of %d, median must reach %.2f s%s. Nothing else in the "
                              "plan starts while the focus is set."
                              % (focus.level, rung.number, rung.total, rung.target,
                                 (" (%.2fx the %.3f s record)" % (rung.target / focus.record_seconds,
                                                                  focus.record_seconds))
                                 if focus.record_seconds else ""),
                              slot="focus")
                return spec, [], ""
        waiting = self.held_by()
        if waiting:
            eligible = [s for s in waiting if not self.stage_blocked(s)]
            if not eligible:
                # Every stage in front of the line is blocked by another one that is also in front of it. That
                # cannot resolve itself, so say which and let the driver stop rather than spin.
                return None, waiting, "; ".join("%s (%s): %s" % (s.level, s.kind, self.stage_blocked(s))
                                                for s in waiting)
            if self.plan.hold_order == SEQUENTIAL:
                pick = eligible[0]  # `waiting` is in plan order, so this is the first runnable stage
                passed = waiting[:waiting.index(pick)]
                if passed:
                    # Say why the depth-first order is NOT taking the stage that comes first: the only reason
                    # is that it cannot start at all, and a silent skip here reads as the rule being ignored.
                    # The key carries the ROUND, so every 8M-step round spent on a LATER level says again what
                    # the earlier one is waiting for (2026-09-19 review) instead of once per driver process,
                    # and `slot="order"` keeps it out of the running-stage line's slot.
                    self.log_once("skip:%s>%s/%s@%d" % (";".join("%s/%s" % s.key for s in passed),
                                                        pick.level, pick.kind, self.state.rounds(pick.key)),
                                  "depth-first order: %s cannot start, so the first stage that can is %s (%s), "
                                  "round %d of it"
                                  % ("; ".join("%s (%s) -- %s" % (s.level, s.kind, self.stage_blocked(s))
                                               for s in passed), pick.level, pick.kind,
                                     self.state.rounds(pick.key) + 1), slot="order")
                return pick, waiting, ""
            pick = min(eligible, key=lambda s: (self.state.rounds(s.key), self.plan.index_of(s.level, s.kind)))
            return pick, waiting, ""
        done = self.state.finished_stages()
        for spec in self.plan.stages:
            if spec.key in done:
                continue
            why = self.stage_blocked(spec)
            if why:
                self.log_once("blocked:%s/%s" % spec.key,
                              "skipping %s (%s) for now: %s" % (spec.level, spec.kind, why))
                continue
            return spec, [], ""
        return None, [], ""

    def round_init(self, spec: StageSpec) -> Path | None:
        """The weights a stage resumes from, preferring its OWN: never from scratch, never another level's.

        A repeat round (§8a) must not restart the level: the order is what `ensure_trainer` will actually load
        (`choose_resume`: the highest-step checkpoint in the stage's own model directory), then what `promote`
        treats as authoritative (`best.zip`), then the level's own promoted specialist. Only when the stage has
        never run does this fall through to `initial_checkpoint`, the ladder's walk backwards.
        """
        model_dir = self.model_dir(spec.level, spec.kind)
        resume, _ = supervise.choose_resume(model_dir, self.zip_steps)
        if resume is not None:
            return resume  # exactly what the trainer will load, so `start_steps` matches the resume point
        source, _ = promotion_source(model_dir, self.zip_steps)
        if source is not None:
            return source  # best.zip, when the checkpoints were pruned but keep_best's peak was kept
        own = specialist_path(self.cfg.cwd / self.cfg.models_dir, spec.level)
        if self.state.rounds(spec.key) and own.exists():
            return own  # this stage ran before and promoted; that file is its own newest weights
        return self.initial_checkpoint(spec.level, spec.kind)

    def begin_stage(self, level: str, init: Path, kind: str = COMPLETE, *, rung: Rung | None = None) -> Stage:
        """Prepares a stage's files and records it as current. Starting the trainer is `ensure_trainer`.

        `rung` makes it a FOCUS RUNG (§11): the same speed stage of the same level, with an explicit
        `target_seconds` instead of the level's S-rank time, written into the generated config so the env
        scales the completion bonus against the very number the promotion rule will read back.
        """
        model_dir = self.model_dir(level, kind)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.stage_run_dir(level, kind).mkdir(parents=True, exist_ok=True)
        self.seed_archives(level, model_dir, kind)
        # The init checkpoint is seeded as this stage's latest.zip when the directory has nothing to resume
        # from, so `supervise.choose_resume` can answer from the first tick. Without it a crash inside the
        # first 50k steps -- before the first checkpoint rotation -- would meet "NO RESUME FILE" and stop the
        # supervisor dead.
        seeded = model_dir / "latest.zip"
        # What `ensure_trainer` will ACTUALLY resume from, read before the seeding below can change the answer.
        # `start_steps` has to come from that file and not from `init`, because the two disagree whenever the
        # stage's model directory already holds newer checkpoints than the `--init` an operator passed -- and
        # then the stage's step cap and the generated `timesteps` total are both measured from the wrong
        # origin, so the "6M-step" round is silently shorter and `learn()` stops early (2026-09-18 review).
        resume, resume_steps = supervise.choose_resume(model_dir, self.zip_steps)
        if resume is None and init.exists():
            self.log("seeding %s from %s" % (seeded, init))
            if not self.cfg.dry_run:
                self.copy(init, seeded)
        init_steps = self.zip_steps(init) if init.exists() else None
        if resume is not None and resume_steps is not None:
            if init_steps is not None and int(resume_steps) != int(init_steps):
                self.log("NOTE %s (%s) will resume from %s at %s steps, not from %s at %s: the budget is "
                         "measured from where the trainer actually starts"
                         % (level, kind, resume.name, "{:,}".format(int(resume_steps)), Path(init).name,
                            "{:,}".format(int(init_steps))))
            init_steps = resume_steps
        rung_target = rung.target if rung is not None else None
        if not self.cfg.dry_run:
            path = write_stage_config(self.plan, level, self.cfg.cwd, init_steps=init_steps,
                                      generated_dir=self.cfg.generated_dir, kind=kind,
                                      target_seconds=rung_target)
            self.log("stage config %s (init %s steps)" % (path, "{:,}".format(init_steps) if init_steps else "?"))
        round_n = self.state.rounds((level, kind)) + 1
        # Whatever the run's status.json says right now was written by an EARLIER round of this stage (the run
        # directory is reused), and the latch has just been reset: those samples are ignored until the new
        # trainer's own numbers pass them. See `Stage.stale_below`.
        stale = read_sample(self.stage_run_dir(level, kind) / "status.json", model_dir / "best.json").timesteps
        target = rung_target or (self.plan.target_for(level) if kind == SPEED else None)
        stage = Stage(level=level, run=stage_run_name(level, kind), index=self.plan.index_of(level, kind),
                      init=init.as_posix(), start_steps=float(init_steps or 0), started_at=self.now(),
                      kind=kind, target_seconds=target,
                      round=round_n, rung=rung.index if rung is not None else None, stale_below=stale)
        self.state.current = stage
        self.save_state()
        self.sup = None  # the next supervisor_for() builds one for this stage
        self.log("STAGE %d/%d %s (%s, round %d): run %s, resuming from %s"
                 % (stage.index + 1, len(self.plan.stages), level, kind, round_n, stage.run, init))
        if round_n > 1:
            # The budget is measured from `start_steps`, which is this init file's own step count, so a repeat
            # round gets a whole fresh cap rather than resuming into an already-spent one.
            self.log("round %d of %s (%s): a fresh %s-step budget from %s, the stage's own newest weights"
                     % (round_n, level, kind, "{:,}".format(self.plan.rule_for(kind).max_steps_per_stage), init))
        if kind == SPEED:
            self.log("%s is a SPEED stage: it promotes only once median_time_50 <= %s"
                     % (level, ("%.2f s (%s)" % (stage.target_seconds,
                                                 "FOCUS rung %d of %d" % (rung.number, rung.total)
                                                 if rung is not None else "the plan's override"))
                        if stage.target_seconds
                        else "%.2f x the level's own S-rank time, read live from the first observation"
                             % self.plan.target_scale))
        if rung is not None:
            # The stale window matters more here than anywhere else: a rung reuses the run directory of every
            # rung before it, so `runs/spec_<level>_speed/status.json` still holds the PREVIOUS rung's median
            # -- which passed its own, easier target -- until the new trainer overwrites it. Say the number.
            self.log("FOCUS rung %d of %d for %s: target %.2f s; ignoring %s's status.json until the trainer "
                     "passes %s steps, so this rung cannot latch on the previous rung's median"
                     % (rung.number, rung.total, level, rung.target, stage.run,
                        "{:,.0f}".format(stale) if stale is not None else "?"))
        return stage

    def seed_archives(self, level: str, model_dir: Path, kind: str = COMPLETE) -> list[str]:
        """Copies exploration archives for THIS level into the stage's model directory.

        Only when the stage has none of its own: a resumed stage must keep the counts it has been building.
        A SPEED stage prefers its own level's complete stage, which has been building those counts on this
        level for millions of steps; the shared run is the fallback. Their floor counts carry the `1/sqrt(N)`
        decay that makes the agent push outward at all, and a stage that starts without them restarts
        exploration from nothing -- which is exactly the trap CLAUDE.md's "renaming a run orphans its
        exploration archives" entry records, and a speed stage IS a new run name.
        """
        pattern = "explore_%s_*.npz" % safe_name(level)
        if any(model_dir.glob(pattern)):
            return []
        candidates = []
        if kind == SPEED:
            candidates.append(self.model_dir(level, COMPLETE))
        if self.plan.rule.seed_explore_from:
            candidates.append(self.cfg.cwd / self.cfg.models_dir / self.plan.rule.seed_explore_from)
        for source in candidates:
            if source == model_dir:
                continue
            copied = []
            for path in sorted(source.glob(pattern)):
                if not self.cfg.dry_run:
                    self.copy(path, model_dir / path.name)
                copied.append(path.name)
            if copied:
                self.log("seeded %d exploration archive(s) for %s from %s" % (len(copied), level, source))
                return copied
        return []

    def ensure_games(self, sup: StageSupervisor) -> bool:
        """Launches the games only when they are not already up: a stage change must not cost a relaunch.

        Twelve cold starts are about four minutes, and the games do not care which level is being trained --
        the env loads the scene on its next reset. Only a missing port justifies the round trip.
        """
        wanted = [self.cfg.base_port + i for i in range(self.cfg.count)]
        try:
            listening = set(sup._port_pids())
        except Exception as exc:  # noqa: BLE001 - never let a probe end the driver
            self.log("could not read the listening ports (%s: %s); launching" % (type(exc).__name__, exc))
            listening = set()
        missing = [port for port in wanted if port not in listening]
        if not missing:
            return True
        self.log("launching %d games on monitor %d (ports not listening: %s)"
                 % (self.cfg.count, self.cfg.monitor, missing))
        if self.cfg.dry_run:
            return True
        if not sup._launch_games(self.cfg.count, self.cfg.monitor):
            return False
        sup.await_boot()
        return True

    def ensure_trainer(self, stage: Stage, sup: StageSupervisor, procs: list[supervise.Proc]) -> str:
        """Starts this stage's trainer when no process is running it.

        `"running"` (one is already there, so the supervisor judges it), `"started"` (this call spawned it) or
        `"waiting"` (the games are not up yet; try again next poll rather than spending a restart on it).
        """
        mine = supervise.self_and_ancestors(procs, sup.pid)
        if any(p.pid not in mine and supervise.matches_script(p.cmdline, "train.py", stage.run, sup.cfg.config)
               for p in procs):
            return "running"
        if not self.ensure_games(sup):
            self.log("games did not come up; trying again at the next poll")
            return "waiting"
        resume, steps = supervise.choose_resume(self.model_dir(stage.level, stage.kind), self.zip_steps)
        if resume is None:
            resume, steps = Path(stage.init), self.zip_steps(Path(stage.init))
        command = supervise.shell_command(
            self.cfg.python, "scripts/train.py",
            ["--config", sup.cfg.config, "--resume", resume.as_posix()],
            "%s\\%s_train.log" % (self.cfg.runs_dir, stage.run))
        if self.cfg.dry_run:
            self.log("[dry-run] would start the trainer: %s" % command)
            return "started"
        pid = self.spawn(command, self.cfg.cwd)
        self.log("started trainer for %s (pid %s, resume %s at %s steps): %s"
                 % (stage.level, pid, resume.name, "{:,}".format(steps) if steps else "?", command))
        sup.grace_until = self.now() + self.cfg.start_grace_seconds
        sup.ensure_helpers(self.processes())
        return "started"

    # -- ending a stage -----------------------------------------------------------------------------

    def stop_stage(self, stage: Stage, procs: list[supervise.Proc], sup: StageSupervisor) -> list[int]:
        """Kills this stage's trainer and its helpers. The resume-file rule is what recovers the weights."""
        mine = supervise.self_and_ancestors(procs, sup.pid)
        roots = [p.pid for p in procs if p.pid not in mine
                 and (supervise.matches_script(p.cmdline, "train.py", stage.run, sup.cfg.config)
                      or any(supervise.matches_script(p.cmdline, Path(script).name, stage.run)
                             for script, _, _ in sup.helper_specs()))]
        killed = supervise.descendants(procs, roots) + roots
        for pid in killed:
            self.log("killing pid %d" % pid)
            if not self.cfg.dry_run:
                self.kill(pid)
        return killed

    def finish_stage(self, stage: Stage, status: str, sample: StageSample,
                     procs: list[supervise.Proc], sup: StageSupervisor, *, reason: str | None = None) -> str:
        """Ends the current stage: stop its processes, promote (or refuse to), record it, start the next one.

        `reason` is recorded in the history entry and says WHY a stage ended other than by its own rule --
        today only `END_STAGE_REASON`, the operator's control file. The status is still the ordinary one, so
        every other rule (`refuse_promotion`, the hold line, `round_init`) reads the entry exactly as it reads
        a stage that ran out of steps.
        """
        self.log("STAGE END %s (%s): %s%s (rate %s over %d fresh, median %s, best %s, target %s, %s steps into "
                 "the stage)"
                 % (stage.level, stage.kind, status, (" -- %s" % reason) if reason else "",
                    _fmt(sample.fresh_rate, 3), sample.fresh_window,
                    _fmt(sample.median_time, 2), _fmt(sample.best_time, 2), _fmt(stage.target_seconds, 2),
                    _steps_into(sample, stage.start_steps)))
        self.stop_stage(stage, procs, sup)
        self.sleep(supervise.STOP_WAIT_S)
        model_dir = self.model_dir(stage.level, stage.kind)
        models_dir = self.cfg.cwd / self.cfg.models_dir
        if promotion_source(model_dir, self.zip_steps)[0] is None:
            # Nothing to promote means nothing ever trained: promoting the init file would hide that, and the
            # next stage would silently start from the stage before it. Stop and let a human look.
            self.log("NO CHECKPOINT to promote in %s: stopping. The stage produced no weights." % model_dir)
            return "no_checkpoint"
        difficulty = int(self.plan.env.get("difficulty", 3))
        refused = refuse_promotion(models_dir, stage.level, kind=stage.kind, status=status)
        if refused:
            # The round is still OVER and still counts -- it just does not replace a policy that was promoted
            # for finishing the level with one that never beat the clock.
            self.log("NOT promoting %s (%s): %s. The round's weights stay in %s, which is where the next "
                     "round resumes from." % (stage.level, stage.kind, refused, model_dir))
            destination, sidecar = specialist_path(models_dir, stage.level), {}
        else:
            destination, sidecar = promote(
                model_dir, models_dir, stage.level,
                sample=sample, status=status, start_steps=stage.start_steps, difficulty=difficulty,
                run=stage.run, kind=stage.kind, target_seconds=stage.target_seconds,
                s_rank_seconds=stage.s_rank_seconds, round=stage.round, rung=stage.rung,
                zip_steps=self.zip_steps, copy=self.copy)
            self.log("promoted %s -> %s" % (sidecar.get("source_checkpoint", "?"), destination))
        entry = {"level": stage.level, "kind": stage.kind, "run": stage.run, "status": status,
                 "index": stage.index, "round": stage.round,
                 # WHICH RUNG of the focus ladder this round was, or None (§11). `target_seconds` below is what
                 # `rung_met` actually reads -- this is the position, for a report and for the round count.
                 "rung": stage.rung,
                 # Why it ended, when that was not its own rule. None for every stage the rule ended.
                 "reason": reason,
                 "start_steps": stage.start_steps, "end_steps": sample.timesteps,
                 "fresh_completion_rate": sample.fresh_rate, "fresh_window": sample.fresh_window,
                 "best_time": sample.best_time, "median_time": sample.median_time,
                 "target_seconds": stage.target_seconds,
                 "s_rank_seconds": stage.s_rank_seconds, "init": stage.init,
                 # `promoted` false means the round ended without replacing the level's specialist: its weights
                 # are in the stage's own model directory and nothing in models/specialists/ moved.
                 "promoted": not refused,
                 "not_promoted_because": refused or None,
                 "specialist": destination.as_posix() if (destination and not refused) else None,
                 "source_checkpoint": sidecar.get("source_checkpoint"),
                 "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.state.history.append(entry)
        self.state.current = None
        self.save_state()

        nxt, waiting, blocked = self.choose_stage()
        if nxt is None and not waiting:
            self.log("EVERY STAGE FINISHED: %d specialists in %s"
                     % (len(self.state.history), self.cfg.cwd / self.cfg.models_dir / SPECIALIST_DIR))
            return "finished"
        if nxt is None:
            self.log("HELD before %s and nothing can run: %s. Stopping so a human can look."
                     % (self.plan.hold_before, blocked))
            return "held"
        if waiting:
            self.log("holding before %s (%s): waiting on %s"
                     % (self.plan.hold_before, self.order_rule_text(),
                        ", ".join("%s (%s)" % (s.level, s.kind) for s in waiting)))
        # A speed stage starts from ITS OWN level's specialist and a repeat round from its own newest weights,
        # neither of which is the file this stage just promoted: that one belongs to the level that ended.
        # The freshly promoted checkpoint stays the fallback for the ordinary "next level up" case.
        init = self.round_init(nxt) or destination or Path(stage.init)
        started = self.begin_stage(nxt.level, init, nxt.kind, rung=self.rung_for(nxt))
        self.ensure_trainer(started, self.supervisor_for(started), self.processes())
        return "advanced"

    # -- ending a stage on purpose (the END_STAGE control file) -------------------------------------

    def clear_end_stage(self) -> bool:
        """Removes the control file. False means it is still there, and then NOTHING may be ended.

        A file that cannot be deleted would end the current stage, then the stage after it, then the stage
        after that, one per poll, for as long as it stays on disk. Refusing to act on it is the safe failure.
        """
        try:
            self.end_stage_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            # Its own slot: this line is written in the same tick as the running-stage line, and two keys in
            # one slot alternate, which turned a stuck file into two log lines a minute forever.
            self.log_once("stuck:%s" % type(exc).__name__,
                          "END_STAGE: could not remove %s (%s: %s), so NO stage is being ended -- a file that "
                          "cannot be removed would end every stage in turn. Delete it by hand."
                          % (self.end_stage_path, type(exc).__name__, exc), slot="end_stage_stuck")
            return False

    def end_stage_now(self) -> str | None:
        """Handles `runs/specialists/END_STAGE`. A string ends the tick; None means "carry on as usual".

        The stage ends through `finish_stage`, the ordinary stage-end path, with status `"unfinished"` and the
        reason recorded -- so a SPEED stage's weights are NOT promoted over the level's specialist
        (`refuse_promotion`), nothing in `models/` is deleted, the round counts, and the next stage is chosen
        by the plan's own rule. The round's weights stay in the stage's own model directory, which is where
        `round_init` resumes its next round from.
        """
        try:
            text = decode_control_file(self.end_stage_path.read_bytes())
        except OSError:
            text = ""  # unreadable is not a reason to refuse: an empty file means "whatever is running"
        stage = self.state.current
        if stage is None:
            self.log_once("idle", "END_STAGE: no stage is running, so there is nothing to end; removing the "
                                  "file", slot="end_stage")
            self.clear_end_stage()
            return None
        objection = end_stage_objection(stage, text)
        if objection:
            # `slot="end_stage"`, not the default: a file the driver cannot delete is re-read every poll, and
            # sharing the running-stage slot made both lines repeat every poll instead of once.
            self.log_once("refused:%s/%s:%s" % (stage.key + (objection,)),
                          "END_STAGE REFUSED: %s. Nothing was ended; removing the file and leaving %s (%s) "
                          "running." % (objection, stage.level, stage.kind), slot="end_stage")
            self.clear_end_stage()
            return None
        if self.cfg.dry_run:
            self.log("[dry-run] END_STAGE: would end %s (%s, round %d) now as 'unfinished' (%s), leave "
                     "models/specialists alone and choose the next stage; changing nothing"
                     % (stage.level, stage.kind, stage.round, END_STAGE_REASON))
            return "would_end_stage"
        # Removed BEFORE the stage ends: if anything below raises, the tick loop carries on and the file is
        # already gone, so at most one stage is ever ended per file.
        if not self.clear_end_stage():
            return None
        sup = self.supervisor_for(stage)
        procs = self.processes()
        # THE SAME reader `tick` uses, filter and all: an operator picks the moment, and the moment may well be
        # inside the stale window at the start of a round >= 2 (2026-09-19 review).
        sample = self.current_sample(stage)
        self.log("END_STAGE: ending %s (%s, round %d) now, at the operator's request"
                 % (stage.level, stage.kind, stage.round))
        return self.finish_stage(stage, "unfinished", sample, procs, sup, reason=END_STAGE_REASON)

    # -- one poll -----------------------------------------------------------------------------------

    def tick(self) -> str:
        if self.pause_path.exists():
            # FIRST, and before END_STAGE is even read: a paused driver does nothing at all, which is the
            # whole promise of the pause file. An END_STAGE left beside it is handled when the pause lifts.
            self.log_once("paused", "PAUSED: %s exists, doing nothing (delete it to resume the ladder)"
                          % self.pause_path)
            return "paused"

        if self.end_stage_path.exists():
            action = self.end_stage_now()
            if action is not None:
                return action

        stage = self.state.current
        if stage is None:
            spec, waiting, blocked = self.choose_stage()
            if spec is None and not waiting:
                self.log_once("finished", "every stage in the plan is finished; nothing to drive")
                return "finished"
            if spec is None:
                self.log("HELD before %s and nothing can run: %s. Stopping so a human can look."
                         % (self.plan.hold_before, blocked))
                return "held"
            if waiting:
                self.log_once("holding:%s" % spec.key[0] + spec.key[1],
                              "holding before %s (%s): waiting on %s"
                              % (self.plan.hold_before, self.order_rule_text(),
                                 ", ".join("%s (%s)" % (s.level, s.kind) for s in waiting)))
            init = self.round_init(spec)
            if init is None:
                self.log("NO INIT CHECKPOINT for %s (%s): pass --init on the first start. Stopping."
                         % (spec.level, spec.kind))
                return "no_init"
            stage = self.begin_stage(spec.level, init, spec.kind, rung=self.rung_for(spec))

        sup = self.supervisor_for(stage)
        procs = self.processes()
        trainer = self.ensure_trainer(stage, sup, procs)
        if trainer != "running":
            return trainer  # "started" (grace now runs) or "waiting" (no games yet)

        health = sup.tick()
        if health in ("budget", "no_resume"):
            self.log("the stage supervisor gave up (%s); the driver stops with it" % health)
            return health

        rule = self.plan.rule_for(stage.kind)
        sample = self.current_sample(stage)
        # The target time is read LIVE off the run, because only the env can see the level's `campaign.ranks`.
        # A plan override is already in the stage and wins; otherwise the first status.json carrying one sets it
        # for good, so the promotion rule and the reward are measured against the same number.
        # A FOCUS RUNG's target is a decision already made, exactly like a `speed.targets:` override, so it is
        # never replaced by whatever the run reports -- not even by the previous rung's number lingering in a
        # status.json the stale filter has stopped covering.
        if (stage.kind == SPEED and sample.target_seconds is not None and stage.rung is None
                and not self.plan.target_for(stage.level) and stage.target_seconds != sample.target_seconds):
            stage.target_seconds = sample.target_seconds
            self.save_state()
            self.log("%s target time: %.2f s (%.2f x the level's own S-rank time, read live by the env)"
                     % (stage.level, sample.target_seconds, self.plan.target_scale))
        if stage.kind == SPEED and sample.s_rank_seconds is not None and stage.s_rank_seconds != sample.s_rank_seconds:
            # Carried for the record only -- the sidecar says what the game itself calls S beside what the
            # stage was actually measured against. Nothing compares against this number.
            stage.s_rank_seconds = sample.s_rank_seconds
            self.save_state()
        verdict, reached = stage_verdict(sample, stage.start_steps, stage.target_reached_at, rule,
                                         kind=stage.kind, target_seconds=stage.target_seconds)
        if reached != stage.target_reached_at:
            stage.target_reached_at = reached
            self.save_state()
            self.log("%s reached the target at %s steps: %s more before promotion, or longer if "
                     "keep_best sets another best" % (stage.level, "{:,.0f}".format(reached or 0),
                                                      "{:,}".format(rule.settle_steps)))
        if verdict == "running":
            self.log_once("stage:%s:%s:%s" % (stage.level, stage.kind, health),
                          "%s (%s): %s, %s steps into the stage, rate %s over %d fresh, median %s (best %s) "
                          "vs target %s"
                          % (stage.level, stage.kind, health,
                             _steps_into(sample, stage.start_steps),
                             _fmt(sample.fresh_rate, 3), sample.fresh_window, _fmt(sample.median_time, 2),
                             _fmt(sample.best_time, 2), _fmt(stage.target_seconds, 2)))
            return health
        if self.cfg.dry_run:
            self.log("[dry-run] the stage rule says %s for %s (%s); changing nothing"
                     % (verdict, stage.level, stage.kind))
            return "would_%s" % verdict
        return self.finish_stage(stage, verdict, sample, procs, sup)

    def initial_checkpoint(self, level: str, kind: str = COMPLETE) -> Path | None:
        """The checkpoint a stage starts from: the previous stage's specialist, or None.

        A SPEED stage looks at its OWN level first -- it is the same level made faster, and starting it from
        some later level's policy would throw away everything the complete stage learned about this one. Only
        if that file is missing does it fall back to the walk backwards, which is the complete-stage rule and
        is unchanged.
        """
        models = self.cfg.cwd / self.cfg.models_dir
        if kind == SPEED:
            own = specialist_path(models, level)
            if own.exists():
                return own
        index = self.plan.index_of(level, kind)
        for earlier in reversed(self.plan.stages[:index]):
            candidate = specialist_path(models, earlier.level)
            if candidate.exists():
                return candidate
        return None

    def save_state(self) -> None:
        if self.cfg.dry_run:
            return
        try:
            self.state.save(self.state_path)
        except OSError as exc:
            self.log("could not write %s: %s" % (self.state_path, exc))

    # -- the loop -----------------------------------------------------------------------------------

    def run(self, max_ticks: int | None = None) -> int:
        speed = self.plan.rule_for(SPEED)
        self.log("driver up: plan=%s stages=%d levels=%d count=%d monitor=%d target=%.2f settle=%s cap=%s"
                 % (self.cfg.plan_path, len(self.plan.stages), len(self.plan.order), self.cfg.count,
                    self.cfg.monitor, self.plan.rule.target_rate, "{:,}".format(self.plan.rule.settle_steps),
                    "{:,}".format(self.plan.rule.max_steps_per_stage)))
        speed_stages = [s.level for s in self.plan.stages if s.kind == SPEED]
        if speed_stages:
            self.log("speed stages (%d): %s -- rate %.2f AND median_time_50 <= %.2f x the level's own S-rank "
                     "time, cap %s, %s"
                     % (len(speed_stages), ", ".join(speed_stages), speed.target_rate, self.plan.target_scale,
                        "{:,}".format(speed.max_steps_per_stage),
                        ("at most %d rounds each" % speed.max_rounds) if speed.max_rounds > 0
                        else "no round cap"))
        focus = self.plan.focus
        if focus is not None:
            rung = self.focus_rung()
            self.log("FOCUS on %s: the ONLY stage that runs is its speed stage, one rung at a time. "
                     "Ladder %s s%s. %s"
                     % (focus.level, ", ".join("%g" % t for t in focus.targets),
                        " -- the record is %.3f s" % focus.record_seconds if focus.record_seconds else "",
                        ("now on rung %d of %d, target %.2f s (%d of the ladder's rungs are already met)"
                         % (rung.number, rung.total, rung.target, rung.index)) if rung is not None
                        else "EVERY RUNG IS DONE: the focus is over and the ordinary plan decides."))
            current = self.state.current
            if rung is not None and current is not None and current.key != (focus.level, SPEED):
                # The focus never interrupts a running stage -- `tick` does not consult `choose_stage` while
                # one is current -- so say plainly that it takes over at the next stage boundary, and how to
                # bring that boundary forward.
                self.log("the stage running now is %s (%s), which the focus SUPERSEDES: it keeps running "
                         "until its own rule ends it or you write %s. The next stage after it will be "
                         "%s (speed), FOCUS rung %d of %d at %.2f s."
                         % (current.level, current.kind, self.end_stage_path, focus.level,
                            rung.number, rung.total, rung.target))
        if self.plan.hold_before:
            waiting = self.held_by()
            self.log("HOLD LINE before %s (%s): no stage at or after it starts until every stage before it is "
                     "done%s" % (self.plan.hold_before, self.order_rule_text(),
                                 "; waiting on " + ", ".join("%s (%s)" % (s.level, s.kind) for s in waiting)
                                 if waiting else " -- the line is open"))
        if self.unplanned:
            self.log("state entries the plan no longer lists (kept, not reordered): %s" % ", ".join(self.unplanned))
        self.log("pause with: New-Item %s" % self.pause_path)
        self.log("end the current stage early with: New-Item %s (its text may name the stage, e.g. \"%s\")"
                 % (self.end_stage_path,
                    "%s %s" % (self.state.current.level, self.state.current.kind) if self.state.current
                    else "Level 0-1 speed"))
        self.log("DO NOT run supervise.py at the same time: the driver supervises each stage itself")
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            try:
                action = self.tick()
            except Exception as exc:  # noqa: BLE001 - a driver that dies on a hiccup is worse than none
                self.log("tick failed (%s: %s); continuing" % (type(exc).__name__, exc))
                action = "error"
            if action in ("budget", "no_resume", "no_init", "no_checkpoint", "held"):
                return 1
            if action == "finished":
                return 0
            if self.cfg.dry_run:
                return 0
            self.sleep(self.cfg.poll_seconds)
        return 0


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else "%.*f" % (digits, value)


def _steps_into(sample: StageSample, start_steps: float) -> str:
    """"12,240" -- how far into the stage a sample is, for a log line, or "an unknown number of".

    A FILTERED sample (`Driver.current_sample`) has no step count at all, and `(None or 0) - start_steps`
    printed the run's whole step count as a negative number of steps into the stage.
    """
    return ("an unknown number of" if sample.timesteps is None
            else "{:,.0f}".format(sample.timesteps - start_steps))


def start_at_objection(plan: Plan, driver: Driver, level: str, kind: str, *, plan_path: str = "the plan",
                       ignore_hold: bool = False, rerun_stage: bool = False) -> str:
    """Why `--start-at level kind` must be refused, or `""`. Pure: it reads the plan and the state, nothing else.

    `--start-at` bypasses `choose_stage`, which is where every other rule lives, so the two holes it leaves are
    closed here (2026-09-18 review):

      * THE HOLD LINE. `choose_stage` will never pick a stage at or after `hold_before` while one in front of
        it is not done. This path would, silently -- and `load_plan` hard-errors on a `hold_before` typo
        precisely so the line cannot be lifted by accident.
      * A STAGE THAT HAS ALREADY RUN. `current` is null after a `"held"` exit as well as on a first launch,
        and `runs/start_driver.cmd` carries `--start-at "Level 0-1" --init <a 17.0M checkpoint>` for good, so
        an operator who reads exit code 1 as a crash and re-runs it would begin a second round of the finished
        0-1 stage -- measuring its budget from the wrong origin and promoting over its specialist.
    """
    focus = plan.focus
    if focus is not None and (level, kind) != (focus.level, SPEED) and focus_rung(plan, driver.state) is not None:
        # `--start-at` bypasses `choose_stage`, which is where the focus lives, so it could start 0-3 under a
        # focus on 0-1 and nothing would ever notice. Clearing the block is the way to change the subject.
        return ("a FOCUS on %s (speed) is set in %s and its ladder is not finished: %s (%s) cannot be started "
                "by hand. Clear `focus:` in the plan to train anything else."
                % (focus.level, plan_path, level, kind))
    if (level, kind) in driver.state.finished_stages() and not rerun_stage:
        return ("%s (%s) has already run (status %r): drop --start-at/--init and let the state file decide, "
                "or pass --rerun-stage to give it another round on purpose"
                % (level, kind, driver.state.stage_status((level, kind))))
    hold = plan.hold_index()
    if hold is None or ignore_hold:
        return ""
    waiting = driver.held_by()
    if not waiting or plan.index_of(level, kind) < hold:
        return ""
    return ("the hold line before %s is up: %s %s not done. Finish them, lift `hold_before` in %s, or pass "
            "--ignore-hold to start %s (%s) anyway"
            % (plan.hold_before, ", ".join("%s (%s)" % (s.level, s.kind) for s in waiting),
               "is" if len(waiting) == 1 else "are", plan_path, level, kind))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/specialists.yaml", help="the level order and the stage template")
    ap.add_argument("--start-at", help='first level, e.g. "Level 0-1" (first launch only; the state file wins after that)')
    ap.add_argument("--start-kind", choices=list(STAGE_KINDS), default=COMPLETE,
                    help="which KIND of stage --start-at names, when a level appears in the plan twice")
    ap.add_argument("--init", help="checkpoint the first stage resumes from, e.g. models/campaign_gates/best.zip")
    ap.add_argument("--count", type=int, help="game instances (default: the plan's)")
    ap.add_argument("--monitor", type=int, help="Windows display number for the games (default: the plan's)")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--stale-seconds", type=float, default=900.0, help="see supervise.Config.stale_seconds")
    ap.add_argument("--start-grace-seconds", type=float, default=900.0)
    ap.add_argument("--max-restarts-per-hour", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=47800)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--dry-run", action="store_true", help="report what it would do and exit, changing nothing")
    ap.add_argument("--ignore-hold", action="store_true",
                    help="let --start-at name a stage at or after `hold_before` while stages in front of the "
                         "line are not done (it lifts the line for that stage; say so out loud)")
    ap.add_argument("--rerun-stage", action="store_true",
                    help="let --start-at name a stage that has already run: a new round of it, from --init")
    import games  # noqa: PLC0415 - late, like every other games import here

    games.add_steam_flags(ap)
    a = ap.parse_args()

    plan = load_plan(a.plan)
    cfg = DriverConfig(plan_path=a.plan, count=a.count or plan.rule.count, monitor=a.monitor or plan.rule.monitor,
                       poll_seconds=a.poll_seconds, stale_seconds=a.stale_seconds,
                       start_grace_seconds=a.start_grace_seconds, max_restarts_per_hour=a.max_restarts_per_hour,
                       base_port=a.base_port, python=a.python, cwd=Path.cwd(), runs_dir=a.runs_dir,
                       models_dir=a.models_dir, dry_run=a.dry_run, no_steam=not a.steam)
    driver = Driver(cfg, plan)
    if driver.state.current is None and (a.start_at or a.init):
        level = a.start_at or plan.stages[0].level
        kind = a.start_kind
        index = plan.index_of(level, kind)  # raises with the plan's own stages when the name is wrong
        objection = start_at_objection(plan, driver, level, kind, plan_path=a.plan,
                                       ignore_hold=a.ignore_hold, rerun_stage=a.rerun_stage)
        if objection:
            ap.error(objection)
        hold = plan.hold_index()
        waiting = driver.held_by()
        if hold is not None and index >= hold and waiting:
            driver.log("--ignore-hold: STARTING %s (%s) PAST THE HOLD LINE before %s, with %s still not done"
                       % (level, kind, plan.hold_before,
                          ", ".join("%s (%s)" % (s.level, s.kind) for s in waiting)))
        # Every stage before the starting one counts as already handled, so `next_stage` does not walk back to
        # the top of the ladder on the next tick. `"skipped"` satisfies the hold line permanently, so it is
        # logged rather than written quietly.
        done = driver.state.finished_stages()
        skipped = [s for s in plan.stages[:index] if s.key not in done]
        if skipped:
            driver.log("--start-at %s (%s) marks %d earlier stage(s) SKIPPED, which satisfies the hold line "
                       "for them for good: %s"
                       % (level, kind, len(skipped), ", ".join("%s (%s)" % (s.level, s.kind) for s in skipped)))
        for earlier in skipped:
            driver.state.history.append({"level": earlier.level, "kind": earlier.kind, "status": "skipped",
                                         "run": earlier.run, "specialist": None})
        init = Path(a.init) if a.init else driver.initial_checkpoint(level, kind)
        if init is None or not init.exists():
            ap.error("--init is required on the first launch: pass the checkpoint the first stage resumes from")
        driver.begin_stage(level, init, kind)
    sys.exit(driver.run())


if __name__ == "__main__":
    main()
