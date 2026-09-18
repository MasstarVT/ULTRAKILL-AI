"""Tests for scripts/campaign_driver.py, the per-level specialist driver.

No game, no real process, no real clock:  python tests/test_campaign_driver.py   (or pytest)

Every side effect the driver has -- listing processes, killing a pid, stopping and launching the games,
spawning a detached trainer, copying a checkpoint -- is injected, exactly as `supervise.Supervisor`'s are, so
these tests drive the whole ladder (stage start, the stage rule, promotion, the next stage) against fakes.
Nothing here starts a game, a trainer or a timer, and nothing touches the repo's models/ or runs/.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_driver as cd  # noqa: E402
import supervise  # noqa: E402
import yaml  # noqa: E402
from supervise import Proc, Status  # noqa: E402

LEVELS = ["Level 0-1", "Level 0-3", "Level 0-4"]
SELF_PID = 999
RULE = cd.StageRule(target_rate=0.5, min_fresh_window=30, settle_steps=300_000,
                    max_steps_per_stage=6_000_000, count=12, monitor=1, seed_explore_from="campaign_gates")


def sample(timesteps=None, rate=None, window=0, best_time=None, best_at=None) -> cd.StageSample:
    return cd.StageSample(timesteps, rate, window, best_time, best_at)


# ---------------------------------------------------------------------------
# The stage rule
# ---------------------------------------------------------------------------


def test_a_stage_keeps_running_below_the_target_rate():
    verdict, reached = cd.stage_verdict(sample(11_000_000, 0.31, 50), 10_000_000, None, RULE)
    assert (verdict, reached) == ("running", None)


def test_a_rate_over_the_target_on_too_few_fresh_episodes_does_not_count():
    """29 fresh episodes at 1.00 is three good loads in a row, not a level that is finished."""
    verdict, reached = cd.stage_verdict(sample(11_000_000, 1.0, 29), 10_000_000, None, RULE)
    assert (verdict, reached) == ("running", None)
    verdict, reached = cd.stage_verdict(sample(11_000_000, 1.0, 30), 10_000_000, None, RULE)
    assert (verdict, reached) == ("running", 11_000_000), "reached, but the settle has not been paid yet"


def test_the_stage_ends_once_the_settle_has_been_paid_after_the_target():
    reached = 11_000_000
    verdict, _ = cd.stage_verdict(sample(11_299_999, 0.6, 40), 10_000_000, reached, RULE)
    assert verdict == "running", "one step short of the settle"
    verdict, _ = cd.stage_verdict(sample(11_300_000, 0.6, 40), 10_000_000, reached, RULE)
    assert verdict == "done"


def test_a_new_best_restarts_the_settle_so_keep_best_can_catch_the_peak():
    """The settle counts from the LATER of the target and the last time best.zip moved."""
    reached = 11_000_000
    # best.zip moved at 11.2M, so the clock runs from there, not from the target.
    verdict, _ = cd.stage_verdict(sample(11_400_000, 0.6, 40, best_at=11_200_000), 10_000_000, reached, RULE)
    assert verdict == "running"
    verdict, _ = cd.stage_verdict(sample(11_500_000, 0.6, 40, best_at=11_200_000), 10_000_000, reached, RULE)
    assert verdict == "done"
    # And a best saved LONG BEFORE the target must not satisfy the settle the instant the target is reached.
    verdict, got = cd.stage_verdict(sample(11_000_000, 0.6, 40, best_at=10_100_000), 10_000_000, None, RULE)
    assert (verdict, got) == ("running", 11_000_000)


def test_the_target_latches_so_a_dip_cannot_un_reach_it():
    """keep_best.py holds the peak in best.zip, and the peak is what gets promoted: a collapse after the
    target must not deadlock the stage. This project has watched a policy peak and then degrade twice."""
    verdict, reached = cd.stage_verdict(sample(11_400_000, 0.04, 50), 10_000_000, 11_000_000, RULE)
    assert (verdict, reached) == ("done", 11_000_000)


def test_a_stage_that_runs_out_of_steps_is_recorded_unfinished():
    verdict, reached = cd.stage_verdict(sample(16_000_000, 0.0, 50), 10_000_000, None, RULE)
    assert (verdict, reached) == ("unfinished", None), "it moves on regardless, so one level cannot block 29"
    # A stage that reached the target AND ran out on the same poll is done, not unfinished.
    verdict, _ = cd.stage_verdict(sample(16_000_000, 0.9, 50), 10_000_000, 11_000_000, RULE)
    assert verdict == "done"


def test_a_stage_with_no_status_yet_is_simply_running():
    assert cd.stage_verdict(cd.EMPTY_SAMPLE, 10_000_000, None, RULE) == ("running", None)


def test_read_sample_survives_a_missing_or_half_written_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert cd.read_sample(root / "nope.json", root / "also_nope.json") == cd.EMPTY_SAMPLE
        (root / "status.json").write_text("{not json", encoding="utf-8")
        assert cd.read_sample(root / "status.json", root / "also_nope.json") == cd.EMPTY_SAMPLE
        (root / "status.json").write_text(json.dumps({
            "timesteps": 11_000_000,
            "campaign": {"fresh_completion_rate": 0.52, "fresh_window": 41, "best_time": 118.5}}), encoding="utf-8")
        (root / "best.json").write_text(json.dumps({"at_timesteps": 10_900_000}), encoding="utf-8")
        got = cd.read_sample(root / "status.json", root / "best.json")
        assert got == cd.StageSample(11_000_000.0, 0.52, 41, 118.5, 10_900_000.0)


# ---------------------------------------------------------------------------
# The plan and the generated per-stage config
# ---------------------------------------------------------------------------


def write_plan(tmp: Path, *, order=LEVELS) -> Path:
    path = tmp / "configs" / "specialists.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "order": list(order),
        "stage": {"target_rate": 0.5, "min_fresh_window": 30, "settle_steps": 300_000,
                  "max_steps_per_stage": 6_000_000, "count": 12, "monitor": 1,
                  "seed_explore_from": "campaign_gates"},
        # A curriculum key is included on purpose: `stage_config` must drop it whatever the plan holds.
        "env": {"mode": "campaign", "level": "Level 0-1", "levels": ["Level 0-1", "Level 0-3"],
                "difficulty": 3, "max_steps": 12000},
        "train": {"algo": "ppo", "num_envs": 12, "save_every": 50000,
                  "hyperparams": {"ent_coef": 0.004}},
    }), encoding="utf-8")
    return path


def test_names_follow_the_level():
    assert cd.stage_run_name("Level 0-1") == "spec_0-1"
    assert cd.stage_config_path("Level 2-3") == "configs/generated/spec_2-3.yaml"
    assert cd.specialist_path(Path("models"), "Level 0-1").as_posix() == "models/specialists/Level_0-1.zip"


def test_load_plan_refuses_a_level_the_build_cannot_load_and_an_unknown_knob():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bad = root / "bad.yaml"
        bad.write_text(yaml.safe_dump({"order": ["Level 9-1"]}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("a level with no scene bundle must be refused")
        except ValueError as exc:
            assert "Level 9-1" in str(exc)
        bad.write_text(yaml.safe_dump({"order": LEVELS, "stage": {"targit_rate": 0.5}}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("a misspelt stage knob must be refused, not silently defaulted")
        except ValueError as exc:
            assert "targit_rate" in str(exc)


def test_a_generated_stage_config_is_single_level_and_budgeted_from_where_it_starts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = cd.load_plan(write_plan(root))
        path = cd.write_stage_config(plan, "Level 0-3", root, init_steps=10_000_000)
        assert path == root / "configs" / "generated" / "spec_0-3.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["env"]["level"] == "Level 0-3"
        assert "levels" not in data["env"], "the plan listed one; a stage may never carry it"
        assert data["train"]["run_name"] == "spec_0-3"
        assert data["train"]["timesteps"] == 10_000_000 + 6_000_000 + cd.TIMESTEPS_SLACK
        assert "GENERATED" in path.read_text(encoding="utf-8").splitlines()[0]


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_best_zip_is_promoted_when_it_exists_and_the_newest_checkpoint_otherwise():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "models" / "spec_0-1"
        model_dir.mkdir(parents=True)
        assert cd.promotion_source(model_dir, lambda p: None) == (None, "none")
        (model_dir / "ckpt_11000000_steps.zip").write_bytes(b"a")
        (model_dir / "ckpt_11500000_steps.zip").write_bytes(b"b")
        # latest.zip is STALE after a kill, which is exactly why the step count is read out of the zip.
        (model_dir / "latest.zip").write_bytes(b"c")
        source, kind = cd.promotion_source(model_dir, lambda p: 10_000_000)
        assert (source.name, kind) == ("ckpt_11500000_steps.zip", "newest")
        (model_dir / "best.zip").write_bytes(b"d")
        source, kind = cd.promotion_source(model_dir, lambda p: 10_000_000)
        assert (source.name, kind) == ("best.zip", "best")


def test_promote_writes_the_specialist_and_its_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "models" / "spec_0-1"
        model_dir.mkdir(parents=True)
        (model_dir / "best.zip").write_bytes(b"weights")
        destination, sidecar = cd.promote(
            model_dir, root / "models", "Level 0-1",
            sample=sample(11_500_000, 0.62, 44, best_time=131.25, best_at=11_300_000),
            status="done", start_steps=10_000_000, difficulty=3, run="spec_0-1")
        assert destination == root / "models" / "specialists" / "Level_0-1.zip"
        assert destination.read_bytes() == b"weights"
        assert sidecar["source_checkpoint"] == "best.zip" and sidecar["source_kind"] == "best"
        assert sidecar["fresh_completion_rate"] == 0.62 and sidecar["best_time"] == 131.25
        assert sidecar["stage_steps"] == 1_500_000 and sidecar["difficulty"] == 3
        written = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
        assert written == sidecar


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_driver_state_round_trips_and_a_missing_file_is_an_empty_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "driver_state.json"
        assert cd.DriverState.load(path) == cd.DriverState()
        state = cd.DriverState(current=cd.Stage(level="Level 0-3", run="spec_0-3", index=1,
                                                init="models/specialists/Level_0-1.zip",
                                                start_steps=11_500_000.0, started_at=1.0,
                                                target_reached_at=12_000_000.0),
                               history=[{"level": "Level 0-1", "status": "done"}])
        state.save(path)
        again = cd.DriverState.load(path)
        assert again.current == state.current and again.history == state.history
        assert again.finished_levels() == {"Level 0-1"}
        path.write_text("{ broken", encoding="utf-8")
        assert cd.DriverState.load(path) == cd.DriverState()


# ---------------------------------------------------------------------------
# The driver, end to end against fakes
# ---------------------------------------------------------------------------


def trainer_procs(run: str) -> list[Proc]:
    cmd = ('"PY.EXE" -u scripts/train.py --config configs/generated/%s.yaml '
           "--resume models/%s/latest.zip" % (run, run))
    return [Proc(3000, 1, cmd), Proc(3001, 3000, '"python.exe" "-c" "from multiprocessing.spawn import spawn_main"')]


def helper_procs(run: str) -> list[Proc]:
    return [Proc(4000, 1, "PY.EXE -u scripts/poll_status.py --run %s" % run),
            Proc(4001, 1, "PY.EXE -u scripts/keep_best.py --run %s --metric campaign" % run),
            Proc(4002, 1, "PY.EXE -u scripts/post_times.py --run %s --watch 600 --push" % run)]


BYSTANDERS = [
    Proc(SELF_PID, 1, "PY.EXE -u scripts/campaign_driver.py --start-at 'Level 0-1'"),
    # The live shared run's trainer: same script, another run and another config. Never matched.
    Proc(777, 1, "PY.EXE -u scripts/train.py --config configs/campaign_gates_full.yaml "
                 "--resume models/campaign_gates/latest.zip"),
]


class Harness:
    """A driver wired to fakes, with every side effect recorded."""

    def __init__(self, tmp: Path, procs=(), status: Status | None = None, **cfg_kwargs):
        self.tmp = tmp
        self.procs = list(procs) or list(BYSTANDERS)
        self.status = status or Status(12.0, "running", 11_000_000)
        self.time = 1_000_000.0
        self.killed: list[int] = []
        self.spawned: list[str] = []
        self.launched: list[tuple[int, int]] = []
        self.copied: list[tuple[str, str]] = []
        self.stopped = 0
        self.slept: list[float] = []
        self.ports = {47800 + i: 23000 + i for i in range(12)}
        self.sets = {pid: 1000 * supervise.MB for pid in self.ports.values()}
        # `num_timesteps` per zip, keyed by PATH: two runs both have a best.zip, and the driver has to tell
        # them apart. A copy carries its source's count, as a real copied checkpoint does.
        self.zip_steps: dict[str, int] = {}

        self.plan = cd.load_plan(write_plan(tmp))
        self.init = tmp / "models" / "campaign_gates" / "best.zip"
        self.init.parent.mkdir(parents=True, exist_ok=True)
        self.init.write_bytes(b"shared")
        self.zip_steps[self.init.as_posix()] = 10_000_000
        kwargs = dict(plan_path=str(tmp / "configs" / "specialists.yaml"), cwd=tmp, python="PY.EXE",
                      start_grace_seconds=0.0)
        kwargs.update(cfg_kwargs)
        self.driver = cd.Driver(
            cd.DriverConfig(**kwargs), self.plan,
            processes=lambda: list(self.procs),
            now=lambda: self.time,
            sleep=self.slept.append,
            probe=lambda path, now: self.status,
            kill=self.killed.append,
            spawn=self._spawn,
            stop_games=self._stop,
            launch_games=self._launch,
            zip_steps=lambda path: self.zip_steps.get(Path(path).as_posix()),
            working_sets=lambda: dict(self.sets),
            port_pids=lambda: dict(self.ports),
            relaunch_one=lambda port: True,
            established=lambda ports: {p: 1 for p in ports},
            cpu=lambda: {},
            copy=self._copy,
            pid=SELF_PID,
        )

    def _spawn(self, command: str, cwd: Path) -> int:
        self.spawned.append(command)
        return 5000 + len(self.spawned)

    def _stop(self) -> None:
        self.stopped += 1

    def _launch(self, count: int, monitor: int) -> bool:
        self.launched.append((count, monitor))
        return True

    def _copy(self, src, dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(Path(src).read_bytes())
        steps = self.zip_steps.get(Path(src).as_posix())
        if steps is not None:
            self.zip_steps[Path(dst).as_posix()] = steps
        self.copied.append((Path(src).name, Path(dst).as_posix()))
        return dst

    # -- helpers the tests drive the fake run with -------------------------------------------------

    def write_status(self, run: str, timesteps: float, rate: float | None, window: int,
                     best_time: float | None = None) -> None:
        path = self.tmp / "runs" / run / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign = {"fresh_completion_rate": rate, "fresh_window": window, "best_time": best_time}
        path.write_text(json.dumps({"timesteps": timesteps, "campaign": campaign}), encoding="utf-8")
        self.status = Status(12.0, "running", timesteps)

    def write_best(self, run: str, at_timesteps: float, *, checkpoint: str = "ckpt_11000000_steps.zip") -> None:
        model_dir = self.tmp / "models" / run
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.json").write_text(json.dumps({"at_timesteps": at_timesteps,
                                                         "checkpoint": checkpoint}), encoding="utf-8")
        (model_dir / "best.zip").write_bytes(b"best-%s" % run.encode())
        self.zip_steps[(model_dir / "best.zip").as_posix()] = int(at_timesteps)

    def trainer_up(self, run: str) -> None:
        self.procs = list(BYSTANDERS) + trainer_procs(run) + helper_procs(run)

    @property
    def trainer_commands(self) -> list[str]:
        return [c for c in self.spawned if "train.py" in c]


def harness(tmp, **kwargs) -> Harness:
    return Harness(Path(tmp), **kwargs)


def test_the_first_tick_of_a_stage_prepares_it_and_starts_the_trainer():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        # The shared run's archives for THIS level, and one for another level that must not be copied.
        shared = h.tmp / "models" / "campaign_gates"
        for port in (47800, 47801):
            (shared / ("explore_Level_0-1_%d.npz" % port)).write_bytes(b"counts")
        (shared / "explore_Level_0-3_47800.npz").write_bytes(b"other")

        h.driver.begin_stage("Level 0-1", h.init)
        assert h.driver.tick() == "started"

        model_dir = h.tmp / "models" / "spec_0-1"
        assert sorted(p.name for p in model_dir.glob("explore_*.npz")) == \
            ["explore_Level_0-1_47800.npz", "explore_Level_0-1_47801.npz"], "only this level's archives"
        assert (model_dir / "latest.zip").read_bytes() == b"shared", "the init is seeded so a crash can resume"
        generated = h.tmp / "configs" / "generated" / "spec_0-1.yaml"
        assert yaml.safe_load(generated.read_text(encoding="utf-8"))["env"]["level"] == "Level 0-1"
        assert h.launched == [], "the games were already listening; a stage must not cost a relaunch"
        assert len(h.trainer_commands) == 1
        command = h.trainer_commands[0]
        assert "--config configs/generated/spec_0-1.yaml" in command
        assert "--resume" in command and "models/spec_0-1/latest.zip" in command.replace("\\", "/"), \
            "the stage resumes from its own seeded copy, so a crash inside the first 50k steps can too"
        assert "runs\\spec_0-1_train.log" in command
        started = sorted(Path(c.split(" -u ")[1].split()[0]).name for c in h.spawned)
        assert started == ["dashboard.py", "keep_best.py", "mem_guard.py", "poll_status.py", "post_times.py", "train.py"]
        state = json.loads((h.tmp / "runs" / "specialists" / "driver_state.json").read_text(encoding="utf-8"))
        assert state["current"]["level"] == "Level 0-1" and state["current"]["start_steps"] == 10_000_000


def test_a_healthy_stage_below_the_target_just_keeps_running():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")
        h.write_status("spec_0-1", 11_000_000, 0.22, 50)
        assert h.driver.tick() == "ok"
        assert len(h.trainer_commands) == 1, "no second trainer"
        assert h.killed == [] and h.stopped == 0


def test_the_stage_rule_promotes_the_best_and_starts_the_next_level():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")

        # The target is reached, but the settle has not been paid: nothing moves.
        h.write_status("spec_0-1", 11_000_000, 0.55, 40, best_time=131.25)
        h.write_best("spec_0-1", 10_950_000)
        assert h.driver.tick() == "ok"
        assert h.driver.state.current.target_reached_at == 11_000_000
        assert h.driver.state.history == []

        # ... and now it has been.
        h.write_status("spec_0-1", 11_400_000, 0.48, 50, best_time=131.25)
        assert h.driver.tick() == "advanced"

        specialist = h.tmp / "models" / "specialists" / "Level_0-1.zip"
        assert specialist.exists() and specialist.read_bytes() == b"best-spec_0-1"
        sidecar = json.loads(specialist.with_suffix(".json").read_text(encoding="utf-8"))
        assert sidecar["status"] == "done" and sidecar["best_time"] == 131.25
        assert sidecar["source_checkpoint"] == "best.zip" and sidecar["stage_steps"] == 1_400_000

        assert sorted(h.killed) == [3000, 3001, 4000, 4001, 4002], "the trainer, its workers and its helpers"
        assert h.stopped == 0, "the games keep running across a stage change"
        history = h.driver.state.history
        assert [e["level"] for e in history] == ["Level 0-1"] and history[0]["status"] == "done"
        stage = h.driver.state.current
        assert stage.level == "Level 0-3" and stage.run == "spec_0-3"
        assert stage.init == specialist.as_posix(), "the next level starts from the previous specialist"
        assert len(h.trainer_commands) == 2
        assert "--config configs/generated/spec_0-3.yaml" in h.trainer_commands[1]


def test_a_stage_that_runs_out_of_steps_is_promoted_as_unfinished_and_the_ladder_goes_on():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")
        # No best.zip at all: the newest checkpoint is promoted instead.
        model_dir = h.tmp / "models" / "spec_0-1"
        (model_dir / "ckpt_15900000_steps.zip").write_bytes(b"newest")
        h.write_status("spec_0-1", 16_000_000, 0.0, 50)
        assert h.driver.tick() == "advanced"
        sidecar = json.loads((h.tmp / "models" / "specialists" / "Level_0-1.json").read_text(encoding="utf-8"))
        assert sidecar["status"] == "unfinished" and sidecar["source_kind"] == "newest"
        assert sidecar["source_checkpoint"] == "ckpt_15900000_steps.zip"
        assert h.driver.state.current.level == "Level 0-3", "a blocked level does not block the ladder"


def test_a_restart_of_the_driver_resumes_the_stage_it_was_on():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")

        fresh = harness(tmp)  # a new process reading the same state file
        assert fresh.driver.state.current is not None
        assert fresh.driver.state.current.level == "Level 0-1"
        fresh.procs = list(h.procs)
        fresh.write_status("spec_0-1", 11_000_000, 0.1, 50)
        assert fresh.driver.tick() == "ok"
        assert fresh.trainer_commands == [], "the trainer is already running; it must not be started twice"


def test_the_ladder_finishes_when_every_level_has_a_specialist():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.state.history = [{"level": level, "status": "done"} for level in LEVELS]
        assert h.driver.next_level() is None
        assert h.driver.tick() == "finished"
        assert h.driver.run(max_ticks=1) == 0 and h.trainer_commands == []


def test_the_pause_file_stops_the_driver_doing_anything_at_all():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.pause_path.parent.mkdir(parents=True, exist_ok=True)
        h.driver.pause_path.write_text("", encoding="utf-8")
        assert h.driver.tick() == "paused"
        assert (h.spawned, h.killed, h.launched, h.stopped) == ([], [], [], 0)
        h.driver.pause_path.unlink()
        assert h.driver.tick() == "started"


def test_a_dry_run_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp, dry_run=True)
        h.driver.begin_stage("Level 0-1", h.init)
        assert h.driver.tick() == "started"
        assert (h.spawned, h.killed, h.launched, h.stopped, h.copied) == ([], [], [], 0, [])
        assert not (h.tmp / "configs" / "generated").exists()
        assert not (h.tmp / "runs" / "specialists" / "driver_state.json").exists()


def test_a_dry_run_reports_the_stage_rule_without_promoting():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")
        h.write_status("spec_0-1", 16_000_000, 0.0, 50)
        dry = harness(tmp, dry_run=True)
        dry.procs = list(h.procs)
        assert dry.driver.tick() == "would_unfinished"
        assert not (h.tmp / "models" / "specialists").exists()


def test_the_games_are_launched_only_when_a_port_is_not_listening():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.ports.pop(47805)
        h.driver.begin_stage("Level 0-1", h.init)
        assert h.driver.tick() == "started"
        assert h.launched == [(12, 1)]


def test_the_stage_supervisor_adds_post_times_to_the_helper_set():
    """A specialist's fastest fresh-start completion is a real level time; post_times.py posts it as it happens."""
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        sup = h.driver.supervisor_for(cd.Stage(level="Level 0-1", run="spec_0-1", index=0, init="x"))
        specs = sup.helper_specs()
        assert [Path(script).name for script, _, _ in specs] == ["poll_status.py", "keep_best.py", "mem_guard.py", "post_times.py", "dashboard.py"]
        args = dict((Path(script).name, arguments) for script, arguments, _ in specs)
        assert args["keep_best.py"] == ["--run", "spec_0-1", "--metric", "campaign"], \
            "the campaign metric, or best.zip would be scored on kills"
        assert args["post_times.py"] == ["--run", "spec_0-1", "--watch", "600", "--push"]
        # And the base class is unchanged for the shared run.
        plain = supervise.Supervisor(supervise.Config(run="campaign_gates", cwd=h.tmp), pid=SELF_PID)
        assert [Path(s).name for s, _, _ in plain.helper_specs()] == ["poll_status.py", "keep_best.py", "mem_guard.py"]


def test_a_stage_never_matches_another_runs_trainer():
    """The live shared run is a `train.py` on another config; killing it would be catastrophic."""
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.procs = list(BYSTANDERS)  # only the shared run's trainer, no spec_0-1 trainer
        stage = h.driver.state.current
        sup = h.driver.supervisor_for(stage)
        assert h.driver.stop_stage(stage, h.procs, sup) == [], "nothing of another run may be killed"
        assert h.driver.ensure_trainer(stage, sup, h.procs) == "started", "and ours is started"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
