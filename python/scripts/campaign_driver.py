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

A SPEED stage adds one clause to the first: its `best_time` must also be at or under the level's own S-rank
time (`target_seconds`, read live off `campaign.ranks.time[-1]` by the env and carried here through
`status.json`). Its `target_rate` is lower (0.4), because a fast policy that finishes four loads in ten is
worth more to the ladder than a slow one that finishes six.

OR the stage has consumed `max_steps_per_stage` (6M, 8M for a speed stage) of its own, at which point it moves
on REGARDLESS and is recorded as `"unfinished"` so it can be revisited later. A blocked level must not stop
the other 29.

**Per stage**: run name `spec_<short level>` (e.g. `spec_0-1`), `models/spec_0-1/`, `runs/spec_0-1/`, the
generated config at `configs/generated/spec_0-1.yaml`, and the shared run's `explore_<level>_*.npz` copied in
when they exist (their floor counts carry the `1/sqrt(N)` decay that makes the agent push outward at all).

**On stage end** the trainer is killed and the checkpoint promoted to `models/specialists/<level>.zip` with a
JSON sidecar (rate, best time, steps, source checkpoint, difficulty). `best.zip` is preferred; with no
`best.zip` the newest checkpoint wins, chosen by `supervise.choose_resume`, which reads the step count out of
the zips rather than trusting `latest.zip`'s name -- `latest.zip` is only written on a graceful stop and a
killed trainer leaves it stale. The next stage then resumes from that promoted file.

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
from ultrakill_ai.times import short_level  # noqa: E402

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
    count: int = 12
    monitor: int = 1
    seed_explore_from: str = "campaign_gates"


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
    for name, block in (("stage", dict(data.get("stage", {}) or {})), ("speed", speed)):
        unexpected = sorted(set(block) - known)
        if unexpected:  # a misspelt knob would silently use its default, exactly as EnvConfig.from_dict would
            raise ValueError("%s: unknown %s settings %s" % (path, name, unexpected))
    return Plan(stages=stages, rule=StageRule(**dict(data.get("stage", {}) or {})), speed=speed, targets=targets,
                env=dict(data.get("env", {}) or {}), train=dict(data.get("train", {}) or {}))


def stage_config(plan: Plan, level: str, *, init_steps: int | None = None, kind: str = COMPLETE) -> dict:
    """The real single-level training config for one stage: `{env: ..., train: ...}`, ready to write as YAML.

    `timesteps` is the run TOTAL in train.py, and every stage resumes from the previous specialist, so the
    count keeps climbing across the ladder: the budget has to be measured from where this stage STARTS. With
    no readable step count in the init file it falls back to the plan's own `timesteps`, or to the cap plus
    slack, either of which only has to exceed what the stage rule will allow.

    A SPEED stage is the same config with `speed_bonus` on -- no reward weight moves, no observation or action
    changes, so the level's own specialist loads into it unchanged. `speed_target_seconds` is set only when the
    plan overrides it for this level; 0 tells the env to read the level's own S-rank time live.
    """
    rule = plan.rule_for(kind)
    env = {key: value for key, value in plan.env.items() if key not in MULTI_LEVEL_KEYS}
    env["level"] = level
    if kind == SPEED:
        env["speed_bonus"] = True
        target = plan.target_for(level)
        if target:
            env["speed_target_seconds"] = float(target)
    train = dict(plan.train)
    train["run_name"] = stage_run_name(level, kind)
    budget = rule.max_steps_per_stage + TIMESTEPS_SLACK
    train["timesteps"] = int(init_steps) + budget if init_steps is not None else int(train.get("timesteps", budget))
    return {"env": env, "train": train}


def write_stage_config(plan: Plan, level: str, cwd: Path, *, init_steps: int | None = None,
                       generated_dir: str = GENERATED_DIR, kind: str = COMPLETE) -> Path:
    path = cwd / stage_config_path(level, generated_dir, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# GENERATED by scripts/campaign_driver.py from configs/specialists.yaml -- do not edit.\n"
              "# Stage: %s (%s), run %s. Edit the plan file and let the driver rewrite this.\n"
              % (level, kind, stage_run_name(level, kind)))
    path.write_text(header + yaml.safe_dump(stage_config(plan, level, init_steps=init_steps, kind=kind),
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
    best_time: float | None      # ... campaign.best_time, the fastest fresh-start completion
    best_at: float | None        # models/<run>/best.json's at_timesteps: when keep_best last moved best.zip
    # ... campaign.target_seconds: the level's own S-rank time, as the ENV read it live off the campaign block.
    # None on a complete stage and on any run older than 2026-09-18, which is why it carries a default.
    target_seconds: float | None = None


EMPTY_SAMPLE = StageSample(None, None, 0, None, None)


def read_sample(status_path: Path, best_json: Path) -> StageSample:
    """One `StageSample` from disk. A missing or half-written file reads as "nothing known yet", never raises."""
    timesteps = rate = best_time = best_at = target = None
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
            best_time = _num(campaign.get("best_time"))
            target = _num(campaign.get("target_seconds"))
    try:
        best = json.loads(best_json.read_text(encoding="utf-8"))
        best_at = _num(best.get("at_timesteps")) if isinstance(best, dict) else None
    except (OSError, ValueError):
        best_at = None
    return StageSample(timesteps, rate, window, best_time, best_at, target)


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

    On a SPEED stage the latch also needs the clock: `best_time` at or under `target_seconds`, the level's own
    S-rank time. With no target read yet the latch can never fire, so the stage runs to its cap and is recorded
    "unfinished" -- loud and safe, and never a silent promotion on the rate alone, which is the exact thing the
    lead's 2026-09-18 instruction forbids.
    """
    steps, reached = sample.timesteps, target_reached_at
    fast_enough = (kind != SPEED or (target_seconds is not None and sample.best_time is not None
                                     and sample.best_time <= target_seconds))
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


def promote(model_dir: Path, models_dir: Path, level: str, *, sample: StageSample, status: str,
            start_steps: float, difficulty: int, run: str, kind: str = COMPLETE,
            target_seconds: float | None = None,
            zip_steps: Callable[[Path], int | None] = supervise.zip_timesteps,
            copy: Callable[[Path, Path], object] = shutil.copy2) -> tuple[Path | None, dict]:
    """Copies the stage's checkpoint to `models/specialists/<level>.zip` and writes its JSON sidecar.

    A speed stage writes to the SAME path, so it overwrites the specialist its own level's complete stage
    promoted: one policy per level is the whole design, and the speed run is that policy made faster. The
    sidecar's `mode` is how `full_run.py` and `specialists_status.py` tell which stage produced the file.
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
        "target_seconds": target_seconds,       # the level's own S-rank time a speed stage was measured against
        "status": status,                       # "done" (the target rate was reached) or "unfinished" (the cap)
        "fresh_completion_rate": sample.fresh_rate,
        "fresh_window": sample.fresh_window,
        "best_time": sample.best_time,
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
    # A stage written before kinds existed has neither field, and both defaults are what it was: a complete
    # stage with no clock to beat. That is the whole of the on-disk migration.
    kind: str = COMPLETE
    target_seconds: float | None = None    # the level's own S-rank time, once the run has reported it

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

    def reconcile(self, plan: "Plan") -> list[str]:
        """Matches this state to `plan` by `(level, kind)` and re-numbers the indices. Returns what it could not place.

        Inserting stages into the plan renumbers everything after them, and the driver was LIVE on stage 3 of 30
        when the three speed stages were added: without this the running stage's stored `index` would print the
        wrong position and `--start-at` would skip the wrong prefix. Nothing is dropped and nothing is renamed --
        a `(level, kind)` the plan no longer lists keeps the index it was saved with and is reported here.
        """
        positions = {s.key: i for i, s in enumerate(plan.stages)}
        missing: list[str] = []
        for entry in self.history:
            key = (entry.get("level"), str(entry.get("kind") or COMPLETE))
            entry.setdefault("kind", key[1])
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
        self.log_path = cfg.cwd / cfg.runs_dir / "specialists_driver.log"
        self.state = DriverState.load(self.state_path)
        # The plan is re-read on every start, so stages can be inserted into it; the state is matched to the
        # new plan by (level, kind) before anything else runs.
        self.unplanned = self.state.reconcile(plan)
        self.sup: StageSupervisor | None = None
        self._last_state: str | None = None

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

    def log_once(self, key: str, message: str) -> None:
        if key != self._last_state:
            self.log(message)
        self._last_state = key

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

    # -- starting a stage ---------------------------------------------------------------------------

    def next_stage(self) -> StageSpec | None:
        """The first plan stage with no history entry for its `(level, kind)`, or None when the ladder is done."""
        done = self.state.finished_stages()
        return next((stage for stage in self.plan.stages if stage.key not in done), None)

    def next_level(self) -> str | None:
        """The level of `next_stage`, or None. Kept for readers that only care which level comes next."""
        stage = self.next_stage()
        return stage.level if stage is not None else None

    def begin_stage(self, level: str, init: Path, kind: str = COMPLETE) -> Stage:
        """Prepares a stage's files and records it as current. Starting the trainer is `ensure_trainer`."""
        model_dir = self.model_dir(level, kind)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.stage_run_dir(level, kind).mkdir(parents=True, exist_ok=True)
        self.seed_archives(level, model_dir, kind)
        # The init checkpoint is seeded as this stage's latest.zip when the directory has nothing to resume
        # from, so `supervise.choose_resume` can answer from the first tick. Without it a crash inside the
        # first 50k steps -- before the first checkpoint rotation -- would meet "NO RESUME FILE" and stop the
        # supervisor dead.
        seeded = model_dir / "latest.zip"
        if supervise.choose_resume(model_dir, self.zip_steps)[0] is None and init.exists():
            self.log("seeding %s from %s" % (seeded, init))
            if not self.cfg.dry_run:
                self.copy(init, seeded)
        init_steps = self.zip_steps(init) if init.exists() else None
        if not self.cfg.dry_run:
            path = write_stage_config(self.plan, level, self.cfg.cwd, init_steps=init_steps,
                                      generated_dir=self.cfg.generated_dir, kind=kind)
            self.log("stage config %s (init %s steps)" % (path, "{:,}".format(init_steps) if init_steps else "?"))
        stage = Stage(level=level, run=stage_run_name(level, kind), index=self.plan.index_of(level, kind),
                      init=init.as_posix(), start_steps=float(init_steps or 0), started_at=self.now(),
                      kind=kind, target_seconds=self.plan.target_for(level) if kind == SPEED else None)
        self.state.current = stage
        self.save_state()
        self.sup = None  # the next supervisor_for() builds one for this stage
        self.log("STAGE %d/%d %s (%s): run %s, resuming from %s"
                 % (stage.index + 1, len(self.plan.stages), level, kind, stage.run, init))
        if kind == SPEED:
            self.log("%s is a SPEED stage: it promotes only once best_time <= %s"
                     % (level, ("%.2f s (the plan's override)" % stage.target_seconds) if stage.target_seconds
                        else "the level's own S-rank time, read live from the first observation"))
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
                     procs: list[supervise.Proc], sup: StageSupervisor) -> str:
        self.log("STAGE END %s (%s): %s (rate %s over %d fresh, best %s, target %s, %s steps into the stage)"
                 % (stage.level, stage.kind, status, _fmt(sample.fresh_rate, 3), sample.fresh_window,
                    _fmt(sample.best_time, 2), _fmt(stage.target_seconds, 2),
                    "{:,.0f}".format((sample.timesteps or 0) - stage.start_steps)))
        self.stop_stage(stage, procs, sup)
        self.sleep(supervise.STOP_WAIT_S)
        difficulty = int(self.plan.env.get("difficulty", 3))
        destination, sidecar = promote(
            self.model_dir(stage.level, stage.kind), self.cfg.cwd / self.cfg.models_dir, stage.level,
            sample=sample, status=status, start_steps=stage.start_steps, difficulty=difficulty,
            run=stage.run, kind=stage.kind, target_seconds=stage.target_seconds,
            zip_steps=self.zip_steps, copy=self.copy)
        if destination is None:
            # Nothing to promote means nothing ever trained: promoting the init file would hide that, and the
            # next stage would silently start from the stage before it. Stop and let a human look.
            self.log("NO CHECKPOINT to promote in %s: stopping. The stage produced no weights."
                     % self.model_dir(stage.level, stage.kind))
            return "no_checkpoint"
        self.log("promoted %s -> %s" % (sidecar.get("source_checkpoint", "?"), destination))
        entry = {"level": stage.level, "kind": stage.kind, "run": stage.run, "status": status,
                 "index": stage.index,
                 "start_steps": stage.start_steps, "end_steps": sample.timesteps,
                 "fresh_completion_rate": sample.fresh_rate, "fresh_window": sample.fresh_window,
                 "best_time": sample.best_time, "target_seconds": stage.target_seconds, "init": stage.init,
                 "specialist": destination.as_posix() if destination else None,
                 "source_checkpoint": sidecar.get("source_checkpoint"),
                 "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.state.history.append(entry)
        self.state.current = None
        self.save_state()

        nxt = self.next_stage()
        if nxt is None:
            self.log("EVERY STAGE FINISHED: %d specialists in %s"
                     % (len(self.state.history), self.cfg.cwd / self.cfg.models_dir / SPECIALIST_DIR))
            return "finished"
        # A speed stage starts from ITS OWN level's specialist, which `initial_checkpoint` knows and this
        # stage's promotion does not: the file just written belongs to the level that ended, not to the next
        # one. The freshly promoted checkpoint stays the fallback for the ordinary "next level up" case.
        init = self.initial_checkpoint(nxt.level, nxt.kind) or destination or Path(stage.init)
        started = self.begin_stage(nxt.level, init, nxt.kind)
        self.ensure_trainer(started, self.supervisor_for(started), self.processes())
        return "advanced"

    # -- one poll -----------------------------------------------------------------------------------

    def tick(self) -> str:
        if self.pause_path.exists():
            self.log_once("paused", "PAUSED: %s exists, doing nothing (delete it to resume the ladder)"
                          % self.pause_path)
            return "paused"

        stage = self.state.current
        if stage is None:
            spec = self.next_stage()
            if spec is None:
                self.log_once("finished", "every stage in the plan is finished; nothing to drive")
                return "finished"
            init = self.initial_checkpoint(spec.level, spec.kind)
            if init is None:
                self.log("NO INIT CHECKPOINT for %s (%s): pass --init on the first start. Stopping."
                         % (spec.level, spec.kind))
                return "no_init"
            stage = self.begin_stage(spec.level, init, spec.kind)

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
        sample = read_sample(self.stage_run_dir(stage.level, stage.kind) / "status.json",
                             self.model_dir(stage.level, stage.kind) / "best.json")
        # The target time is read LIVE off the run, because only the env can see the level's `campaign.ranks`.
        # A plan override is already in the stage and wins; otherwise the first status.json carrying one sets it
        # for good, so the promotion rule and the reward are measured against the same number.
        if (stage.kind == SPEED and sample.target_seconds is not None
                and not self.plan.target_for(stage.level) and stage.target_seconds != sample.target_seconds):
            stage.target_seconds = sample.target_seconds
            self.save_state()
            self.log("%s target time: %.2f s (the level's own S-rank time, read live)" % (stage.level, sample.target_seconds))
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
                          "%s (%s): %s, %s steps into the stage, rate %s over %d fresh, best %s vs target %s"
                          % (stage.level, stage.kind, health,
                             "{:,.0f}".format((sample.timesteps or 0) - stage.start_steps),
                             _fmt(sample.fresh_rate, 3), sample.fresh_window,
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
            self.log("speed stages (%d): %s -- rate %.2f AND best_time <= the level's own S-rank time, cap %s"
                     % (len(speed_stages), ", ".join(speed_stages), speed.target_rate,
                        "{:,}".format(speed.max_steps_per_stage)))
        if self.unplanned:
            self.log("state entries the plan no longer lists (kept, not reordered): %s" % ", ".join(self.unplanned))
        self.log("pause with: New-Item %s" % self.pause_path)
        self.log("DO NOT run supervise.py at the same time: the driver supervises each stage itself")
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            try:
                action = self.tick()
            except Exception as exc:  # noqa: BLE001 - a driver that dies on a hiccup is worse than none
                self.log("tick failed (%s: %s); continuing" % (type(exc).__name__, exc))
                action = "error"
            if action in ("budget", "no_resume", "no_init", "no_checkpoint"):
                return 1
            if action == "finished":
                return 0
            if self.cfg.dry_run:
                return 0
            self.sleep(self.cfg.poll_seconds)
        return 0


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else "%.*f" % (digits, value)


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
        # Every stage before the starting one counts as already handled, so `next_stage` does not walk back to
        # the top of the ladder on the next tick.
        done = driver.state.finished_stages()
        for earlier in plan.stages[:index]:
            if earlier.key not in done:
                driver.state.history.append({"level": earlier.level, "kind": earlier.kind, "status": "skipped",
                                             "run": earlier.run, "specialist": None})
        init = Path(a.init) if a.init else driver.initial_checkpoint(level, kind)
        if init is None or not init.exists():
            ap.error("--init is required on the first launch: pass the checkpoint the first stage resumes from")
        driver.begin_stage(level, init, kind)
    sys.exit(driver.run())


if __name__ == "__main__":
    main()
