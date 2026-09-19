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


SPEED_RULE = cd.StageRule(target_rate=0.4, min_fresh_window=30, settle_steps=300_000,
                          max_steps_per_stage=8_000_000, count=12, monitor=1,
                          seed_explore_from="campaign_gates")
# The plan the speed tests run against: the three complete stages, then their three speed stages, then 0-4.
SPEED_ORDER = ["Level 0-1", "Level 0-3",
               {"level": "Level 0-1", "kind": "speed"}, {"level": "Level 0-3", "kind": "speed"},
               "Level 0-4"]


def sample(timesteps=None, rate=None, window=0, best_time=None, best_at=None,
           target_seconds=None, s_rank_seconds=None, median_time=None) -> cd.StageSample:
    """`median_time` defaults to `best_time`: a run whose every completion took the same time.

    The speed rule gates on the MEDIAN, so the tests that are about the difference between the two pass both.
    """
    return cd.StageSample(timesteps, rate, window, best_time, best_at, target_seconds, s_rank_seconds,
                          best_time if median_time is None else median_time)


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


def write_plan(tmp: Path, *, order=LEVELS, speed: dict | None = None, hold_before=None) -> Path:
    path = tmp / "configs" / "specialists.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "order": list(order),
        "hold_before": hold_before,
        "speed": speed if speed is not None else {"target_rate": 0.4, "max_steps_per_stage": 8_000_000,
                                                  "targets": {}},
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
                     best_time: float | None = None, target_seconds: float | None = None,
                     s_rank_seconds: float | None = None, median_time: float | None = None) -> None:
        """`median_time` defaults to `best_time`: the speed rule gates on the median, the sidecar reports both."""
        path = self.tmp / "runs" / run / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign = {"fresh_completion_rate": rate, "fresh_window": window, "best_time": best_time,
                    "median_time_50": best_time if median_time is None else median_time,
                    "target_seconds": target_seconds, "s_rank_seconds": s_rank_seconds}
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


# ---------------------------------------------------------------------------
# Speed stages (docs/superpowers/specs/2026-09-18-speed-stages.md)
# ---------------------------------------------------------------------------


def test_a_speed_stage_has_its_own_run_name_and_paths():
    """A run of its own, because best.zip changes meaning inside it: keep_best scores the clock, not the rate."""
    assert cd.stage_run_name("Level 0-1") == "spec_0-1"
    assert cd.stage_run_name("Level 0-1", cd.COMPLETE) == "spec_0-1"
    assert cd.stage_run_name("Level 0-1", cd.SPEED) == "spec_0-1_speed"
    assert cd.stage_config_path("Level 0-1", kind=cd.SPEED) == "configs/generated/spec_0-1_speed.yaml"
    # ... and there is only ever ONE specialist file per level: the speed stage overwrites it.
    assert (cd.specialist_path(Path("models"), "Level 0-1")
            == cd.specialist_path(Path("models"), "Level 0-1"))


def test_the_plan_carries_stage_kinds_and_keeps_one_row_per_level_for_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        plan = cd.load_plan(write_plan(Path(tmp), order=SPEED_ORDER))
        assert [s.key for s in plan.stages] == [
            ("Level 0-1", "complete"), ("Level 0-3", "complete"),
            ("Level 0-1", "speed"), ("Level 0-3", "speed"), ("Level 0-4", "complete")]
        assert plan.order == ["Level 0-1", "Level 0-3", "Level 0-4"], "full_run plays each level once"
        assert plan.index_of("Level 0-1", cd.SPEED) == 2 and plan.index_of("Level 0-1") == 0
        assert plan.rule_for(cd.COMPLETE).target_rate == 0.5
        speed = plan.rule_for(cd.SPEED)
        assert (speed.target_rate, speed.max_steps_per_stage) == (0.4, 8_000_000)
        assert speed.settle_steps == 300_000 and speed.min_fresh_window == 30, "unset knobs fall back to `stage`"
        assert plan.target_for("Level 0-1") is None, "no override: the env reads the level's own S-rank time"


def test_load_plan_refuses_an_unknown_kind_and_a_repeated_stage():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.yaml"
        bad.write_text(yaml.safe_dump({"order": [{"level": "Level 0-1", "kind": "fast"}]}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("an unknown stage kind must be refused")
        except ValueError as exc:
            assert "fast" in str(exc)
        bad.write_text(yaml.safe_dump({"order": ["Level 0-1", "Level 0-1"]}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("the same (level, kind) twice must be refused")
        except ValueError as exc:
            assert "twice" in str(exc)
        # ... but the same level under two DIFFERENT kinds is the whole point and must be accepted.
        bad.write_text(yaml.safe_dump({"order": ["Level 0-1", {"level": "Level 0-1", "kind": "speed"}]}),
                       encoding="utf-8")
        assert len(cd.load_plan(bad).stages) == 2
        # A misspelt knob in the speed block is refused exactly as one in `stage:` is.
        bad.write_text(yaml.safe_dump({"order": ["Level 0-1"], "speed": {"targit_rate": 0.4}}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("a misspelt speed knob must be refused")
        except ValueError as exc:
            assert "targit_rate" in str(exc)


def test_a_speed_stage_config_turns_the_bonus_on_and_carries_an_override():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = cd.load_plan(write_plan(root, order=SPEED_ORDER,
                                       speed={"target_rate": 0.4, "max_steps_per_stage": 8_000_000,
                                              "targets": {"Level 0-3": 95.0}}))
        complete = cd.stage_config(plan, "Level 0-1", init_steps=10_000_000)
        assert "speed_bonus" not in complete["env"], "a complete stage is byte-identical to what it was"
        speed = cd.stage_config(plan, "Level 0-1", init_steps=10_000_000, kind=cd.SPEED)
        assert speed["env"]["speed_bonus"] is True
        assert "speed_target_seconds" not in speed["env"], "no override: the env reads the S-rank time live"
        assert speed["env"]["speed_target_scale"] == 0.75, "the default scale is written out explicitly (§8b)"
        assert speed["train"]["run_name"] == "spec_0-1_speed"
        assert speed["train"]["timesteps"] == 10_000_000 + 8_000_000 + cd.TIMESTEPS_SLACK, "the speed cap"
        # Every episode is a fresh level load: the bonus is only scaled on a fresh-start completion, and the
        # stage is only SCORED on fresh-start episodes, so a checkpoint respawn would pay the full unscaled
        # weight for the outcome the stage does not measure (2026-09-18 review).
        assert speed["env"]["fresh_start_prob"] == 1.0
        # Every reward weight and other env setting is the complete stage's: the same policy continues into it.
        assert {k: v for k, v in speed["env"].items()
                if k not in ("speed_bonus", "speed_target_scale", "fresh_start_prob")} == \
            {k: v for k, v in complete["env"].items() if k != "fresh_start_prob"}
        overridden = cd.stage_config(plan, "Level 0-3", kind=cd.SPEED)
        assert overridden["env"]["speed_target_seconds"] == 95.0
        path = cd.write_stage_config(plan, "Level 0-1", root, kind=cd.SPEED)
        assert path.name == "spec_0-1_speed.yaml"
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["speed_bonus"] is True


def test_a_speed_stage_promotes_on_the_median_and_never_on_one_lucky_load():
    """THE 2026-09-18 REVIEW'S BLOCKER. `campaign.best_time` is the run's LIFETIME MINIMUM.

    `ProgressCallback` only ever lowers it and `_restore` carries it across every trainer restart and into
    every later round, so a single lucky fresh load satisfies `best_time <= target` for the rest of the run --
    and the hold line would then open on a policy whose typical completion is twice the target. On the live
    runs the gap is about 2x in both directions that have ever completed a level: spec_0-1 best 243.4 s with a
    median of 490.9 s, spec_0-2 best 139.5 s with a median of 236.5 s.
    """
    target = 120.0
    # One 88 s load in the record, a policy whose median is 480 s, a rate comfortably over the speed bar.
    lucky = sample(11_000_000, 0.45, 50, best_time=88.0, median_time=480.0)
    assert cd.stage_verdict(lucky, 10_000_000, None, SPEED_RULE, kind=cd.SPEED,
                            target_seconds=target) == ("running", None), \
        "one lucky load is not a fast policy, however long it stays in best_time"
    # It still cannot latch 400k steps later, which is where the old rule recorded the stage \"done\".
    assert cd.stage_verdict(sample(11_400_000, 0.45, 50, best_time=88.0, median_time=480.0),
                            10_000_000, None, SPEED_RULE, kind=cd.SPEED, target_seconds=target)[0] == "running"
    # ... and the same run, once the MEDIAN is actually inside the target, latches at once.
    verdict, reached = cd.stage_verdict(sample(11_400_000, 0.45, 50, best_time=88.0, median_time=110.0),
                                        10_000_000, None, SPEED_RULE, kind=cd.SPEED, target_seconds=target)
    assert (verdict, reached) == ("running", 11_400_000)
    # A stage that has never completed the level has no median at all and can never latch.
    assert cd.stage_verdict(sample(11_000_000, 0.9, 50, best_time=None, median_time=None), 10_000_000, None,
                            SPEED_RULE, kind=cd.SPEED, target_seconds=target) == ("running", None)


def test_a_speed_stage_will_not_promote_on_the_rate_alone():
    """The lead's instruction: nothing moves on until the level is finished FAST, not just finished."""
    fast = sample(11_000_000, 0.9, 50, best_time=95.0)
    slow = sample(11_000_000, 0.9, 50, best_time=131.0)
    assert cd.stage_verdict(slow, 10_000_000, None, SPEED_RULE, kind=cd.SPEED,
                            target_seconds=120.0) == ("running", None), "a 131 s best misses a 120 s target"
    verdict, reached = cd.stage_verdict(fast, 10_000_000, None, SPEED_RULE, kind=cd.SPEED, target_seconds=120.0)
    assert (verdict, reached) == ("running", 11_000_000), "latched; the settle has still to be paid"
    verdict, _ = cd.stage_verdict(sample(11_300_000, 0.9, 50, best_time=95.0), 10_000_000, 11_000_000,
                                  SPEED_RULE, kind=cd.SPEED, target_seconds=120.0)
    assert verdict == "done"
    # The rate bar is the speed rule's own, lower one, and it still has to be met.
    assert cd.stage_verdict(sample(11_000_000, 0.35, 50, best_time=95.0), 10_000_000, None, SPEED_RULE,
                            kind=cd.SPEED, target_seconds=120.0) == ("running", None)
    assert cd.stage_verdict(sample(11_000_000, 0.45, 50, best_time=95.0), 10_000_000, None, SPEED_RULE,
                            kind=cd.SPEED, target_seconds=120.0)[1] == 11_000_000


def test_a_speed_stage_with_no_target_read_runs_to_its_cap_instead_of_promoting():
    """Loud and safe. A missing target may never be read as "promote on the rate", which is the failure the
    whole change exists to prevent."""
    good = sample(11_000_000, 0.9, 50, best_time=42.0)
    assert cd.stage_verdict(good, 10_000_000, None, SPEED_RULE, kind=cd.SPEED,
                            target_seconds=None) == ("running", None)
    assert cd.stage_verdict(sample(18_000_000, 0.9, 50, best_time=42.0), 10_000_000, None, SPEED_RULE,
                            kind=cd.SPEED, target_seconds=None) == ("unfinished", None)
    # And a stage with a target but NO completion at all cannot latch either.
    assert cd.stage_verdict(sample(11_000_000, 0.9, 50, best_time=None), 10_000_000, None, SPEED_RULE,
                            kind=cd.SPEED, target_seconds=120.0) == ("running", None)
    # A complete stage is untouched by any of it.
    assert cd.stage_verdict(good, 10_000_000, None, RULE)[1] == 11_000_000


def test_read_sample_carries_the_target_the_env_measured():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "status.json").write_text(json.dumps({
            "timesteps": 11_000_000,
            "campaign": {"fresh_completion_rate": 0.52, "fresh_window": 41, "best_time": 118.5,
                         "target_seconds": 120.0}}), encoding="utf-8")
        got = cd.read_sample(root / "status.json", root / "none.json")
        assert got.target_seconds == 120.0
        # ... and the median the rule actually gates on, beside the lifetime best it only reports.
        (root / "both.json").write_text(json.dumps({
            "timesteps": 11_000_000,
            "campaign": {"fresh_completion_rate": 0.52, "fresh_window": 41, "best_time": 118.5,
                         "median_time_50": 243.0, "target_seconds": 120.0}}), encoding="utf-8")
        both = cd.read_sample(root / "both.json", root / "none.json")
        assert (both.best_time, both.median_time) == (118.5, 243.0)
        # A status.json written before the speed stages simply has no target, and reads as None.
        (root / "old.json").write_text(json.dumps({
            "timesteps": 11_000_000,
            "campaign": {"fresh_completion_rate": 0.52, "fresh_window": 41, "best_time": 118.5}}),
            encoding="utf-8")
        assert cd.read_sample(root / "old.json", root / "none.json").target_seconds is None


def test_the_state_is_matched_to_a_plan_with_stages_inserted_into_it():
    """The driver was LIVE on stage 3/30 when the speed stages were added, and had to restart into them."""
    plan_stages = [cd.StageSpec("Level 0-1"), cd.StageSpec("Level 0-3"),
                   cd.StageSpec("Level 0-1", cd.SPEED), cd.StageSpec("Level 0-3", cd.SPEED),
                   cd.StageSpec("Level 0-4")]
    plan = cd.Plan(stages=plan_stages, rule=RULE, env={}, train={})
    # Written by the version before kinds existed: no `kind` anywhere, and the old indices.
    state = cd.DriverState.from_dict({
        "current": {"level": "Level 0-4", "run": "spec_0-4", "index": 2, "init": "x", "start_steps": 1.0},
        "history": [{"level": "Level 0-1", "status": "done", "index": 0},
                    {"level": "Level 0-3", "status": "done", "index": 1}]})
    assert state.current.kind == "complete", "a stage written before kinds WAS a complete one"
    missing = state.reconcile(plan)
    assert missing == []
    assert state.current.index == 4, "0-4 moved from 2 to 4 when three speed stages were inserted"
    assert [h["kind"] for h in state.history] == ["complete", "complete"]
    assert state.finished_stages() == {("Level 0-1", "complete"), ("Level 0-3", "complete")}
    # ... and the ladder's next stage is now the FIRST speed stage, not Level 0-4.
    assert [s.key for s in plan.stages if s.key not in state.finished_stages()][0] == ("Level 0-1", "speed")
    # A (level, kind) the plan no longer lists keeps its index and is reported rather than dropped.
    state.history.append({"level": "Level 2-1", "status": "done", "index": 9})
    assert state.reconcile(plan) == ["Level 2-1/complete"]
    assert state.history[-1]["index"] == 9


def test_a_restart_into_the_new_plan_keeps_the_running_stage():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)  # the old plan: three complete stages
        h.driver.begin_stage("Level 0-3", h.init)
        h.trainer_up("spec_0-3")
        assert h.driver.state.current.index == 1

        # The plan is edited and the driver restarted: the same state file, a plan with speed stages in it.
        write_plan(Path(tmp), order=SPEED_ORDER)
        fresh = harness(tmp)
        stage = fresh.driver.state.current
        assert stage is not None and stage.level == "Level 0-3" and stage.kind == "complete"
        assert stage.run == "spec_0-3", "the live run keeps its name, so its files and helpers still match"
        assert stage.index == 1, "renumbered against the new plan, and still where it was"
        fresh.procs = list(h.procs)
        fresh.write_status("spec_0-3", 11_000_000, 0.1, 50)
        assert fresh.driver.tick() == "ok"
        assert fresh.trainer_commands == [], "the trainer is already running; it must not be started twice"


def test_the_speed_stage_starts_from_its_own_levels_specialist_and_overwrites_it():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        # The harness writes the default plan; swap in one with the speed stages, as editing the file does.
        h.plan = cd.load_plan(write_plan(Path(tmp), order=SPEED_ORDER))
        h.driver.plan = h.plan
        # Both complete stages are done, and each promoted a specialist.
        for level in ("Level 0-1", "Level 0-3"):
            path = cd.specialist_path(h.tmp / "models", level)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"complete-%s" % level.encode())
            h.zip_steps[path.as_posix()] = 12_000_000
        h.driver.state.history = [{"level": "Level 0-1", "kind": "complete", "status": "done"},
                                  {"level": "Level 0-3", "kind": "complete", "status": "done"}]

        spec = h.driver.next_stage()
        assert spec.key == ("Level 0-1", "speed"), "the speed stages come before Level 0-4"
        init = h.driver.initial_checkpoint(spec.level, spec.kind)
        assert init == cd.specialist_path(h.tmp / "models", "Level 0-1"), \
            "its OWN level's specialist, not the last one promoted"
        assert h.driver.tick() == "started"
        stage = h.driver.state.current
        assert (stage.level, stage.kind, stage.run) == ("Level 0-1", "speed", "spec_0-1_speed")
        assert "--config configs/generated/spec_0-1_speed.yaml" in h.trainer_commands[-1]

        # It promotes over the complete stage's specialist, and the sidecar says which stage produced it.
        h.trainer_up("spec_0-1_speed")
        h.write_status("spec_0-1_speed", 13_000_000, 0.55, 50, best_time=95.0, target_seconds=120.0)
        h.write_best("spec_0-1_speed", 12_900_000)
        assert h.driver.tick() == "ok"
        assert h.driver.state.current.target_seconds == 120.0, "read live off the run's status.json"
        assert h.driver.state.current.target_reached_at == 13_000_000
        h.write_status("spec_0-1_speed", 13_400_000, 0.55, 50, best_time=95.0, target_seconds=120.0)
        assert h.driver.tick() == "advanced"
        specialist = h.tmp / "models" / "specialists" / "Level_0-1.zip"
        assert specialist.read_bytes() == b"best-spec_0-1_speed", "the speed stage overwrote it"
        sidecar = json.loads(specialist.with_suffix(".json").read_text(encoding="utf-8"))
        assert sidecar["mode"] == "speed" and sidecar["target_seconds"] == 120.0
        assert sidecar["best_time"] == 95.0 and sidecar["status"] == "done"
        assert h.driver.state.current.key == ("Level 0-3", "speed"), "then the next speed stage"


def test_a_speed_stages_keep_best_scores_the_clock():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        speed = h.driver.supervisor_for(cd.Stage(level="Level 0-1", run="spec_0-1_speed", index=2,
                                                 init="x", kind=cd.SPEED))
        args = dict((Path(script).name, arguments) for script, arguments, _ in speed.helper_specs())
        assert args["keep_best.py"] == ["--run", "spec_0-1_speed", "--metric", "time", "--min-rate", "0.3"]
        assert args["post_times.py"] == ["--run", "spec_0-1_speed", "--watch", "600", "--push"]
        # ... and a complete stage is unchanged: the rate metric, or best.zip would be scored on a lucky load.
        h.driver.sup = None
        plain = h.driver.supervisor_for(cd.Stage(level="Level 0-1", run="spec_0-1", index=0, init="x"))
        plain_args = dict((Path(script).name, arguments) for script, arguments, _ in plain.helper_specs())
        assert plain_args["keep_best.py"] == ["--run", "spec_0-1", "--metric", "campaign"]


def test_a_speed_stage_inherits_its_own_levels_exploration_archives():
    """Renaming a run orphans its archives, and a speed stage IS a new run name (CLAUDE.md's own warning)."""
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        complete = h.tmp / "models" / "spec_0-1"
        complete.mkdir(parents=True, exist_ok=True)
        for port in (47800, 47801):
            (complete / ("explore_Level_0-1_%d.npz" % port)).write_bytes(b"millions of steps of counts")
        shared = h.tmp / "models" / "campaign_gates"
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "explore_Level_0-1_47800.npz").write_bytes(b"the shared run's, far colder")

        model_dir = h.tmp / "models" / "spec_0-1_speed"
        model_dir.mkdir(parents=True, exist_ok=True)
        copied = h.driver.seed_archives("Level 0-1", model_dir, cd.SPEED)
        assert sorted(copied) == ["explore_Level_0-1_47800.npz", "explore_Level_0-1_47801.npz"]
        assert (model_dir / "explore_Level_0-1_47800.npz").read_bytes() == b"millions of steps of counts"
        assert h.driver.seed_archives("Level 0-1", model_dir, cd.SPEED) == [], "never over its own counts"


# ---------------------------------------------------------------------------
# The hold line and its rounds (spec §8a)
# ---------------------------------------------------------------------------


# The live driver's state file exactly as it stood on 2026-09-18, BEFORE stage kinds, rounds and the hold line
# existed (no "kind", no "round"). A literal, not a read of runs/: the live file moves on with the run, and a test
# that depends on it goes red the moment the driver changes stage (it did, at 24,757,714 steps).
PRE_KINDS_STATE = {
    "version": 1,
    "current": {"level": "Level 0-3", "run": "spec_0-3", "index": 2,
                "init": "F:/Github/ULTRAKILL-AI/python/models/specialists/Level_0-2.zip",
                "start_steps": 18752038.0, "started_at": 1789745506.3306668, "target_reached_at": None},
    "history": [
        {"level": "Level 0-1", "run": "spec_0-1", "status": "done", "index": 0, "start_steps": 17002318.0,
         "end_steps": 18362686.0, "fresh_completion_rate": 0.48, "fresh_window": 50, "best_time": 243.441513,
         "init": "models/campaign_gates/ckpt_17002318_steps.zip",
         "specialist": "F:/Github/ULTRAKILL-AI/python/models/specialists/Level_0-1.zip",
         "source_checkpoint": "best.zip", "finished_at": "2026-09-18 08:45:39"},
        {"level": "Level 0-2", "run": "spec_0-2", "status": "done", "index": 1, "start_steps": 18052150.0,
         "end_steps": 19089202.0, "fresh_completion_rate": 0.72, "fresh_window": 50, "best_time": 139.530548,
         "init": "F:/Github/ULTRAKILL-AI/python/models/specialists/Level_0-1.zip",
         "specialist": "F:/Github/ULTRAKILL-AI/python/models/specialists/Level_0-2.zip",
         "source_checkpoint": "best.zip", "finished_at": "2026-09-18 10:31:46"},
    ],
}


def held_harness(tmp, **kwargs):
    """A driver on the speed plan with `hold_before: Level 0-4`, both complete stages already done."""
    h = harness(tmp, **kwargs)
    h.plan = cd.load_plan(write_plan(Path(tmp), order=SPEED_ORDER, hold_before="Level 0-4"))
    h.driver.plan = h.plan
    for level in ("Level 0-1", "Level 0-3"):
        path = cd.specialist_path(h.tmp / "models", level)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete-%s" % level.encode())
        h.zip_steps[path.as_posix()] = 12_000_000
    h.driver.state.history = [{"level": "Level 0-1", "kind": "complete", "status": "done", "round": 1},
                              {"level": "Level 0-3", "kind": "complete", "status": "done", "round": 1}]
    return h


def test_the_plan_carries_a_hold_line_and_refuses_a_level_that_is_not_in_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = cd.load_plan(write_plan(root, order=SPEED_ORDER, hold_before="Level 0-4"))
        assert plan.hold_before == "Level 0-4"
        # The line bites at 0-4's FIRST stage, so all four stages before it stay runnable.
        assert plan.hold_index() == 4 and [s.key for s in plan.stages[:4]] == [
            ("Level 0-1", "complete"), ("Level 0-3", "complete"),
            ("Level 0-1", "speed"), ("Level 0-3", "speed")]
        # No key at all, and an explicit null, both mean "no line" -- that is how an operator lifts it.
        assert cd.load_plan(write_plan(root, order=SPEED_ORDER)).hold_index() is None
        assert cd.load_plan(write_plan(root, order=SPEED_ORDER, hold_before=None)).hold_before is None
        bad = root / "bad.yaml"
        bad.write_text(yaml.safe_dump({"order": LEVELS, "hold_before": "Level 0-5"}), encoding="utf-8")
        try:
            cd.load_plan(bad)
            raise AssertionError("a hold line naming a level outside the plan must be refused")
        except ValueError as exc:
            assert "hold_before" in str(exc), "a typo here would silently let the ladder walk to 0-4"


def test_nothing_at_or_after_the_hold_line_starts_while_a_stage_before_it_is_unfinished():
    """The lead's instruction: nothing promotes to 0-4 until these levels have better times."""
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        # 0-1's speed stage burned its cap: "unfinished" is exactly the case the line exists to catch.
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        assert [s.key for s in h.driver.held_by()] == [("Level 0-1", "speed"), ("Level 0-3", "speed")]
        pick, waiting, blocked = h.driver.choose_stage()
        assert blocked == "" and pick.key == ("Level 0-3", "speed"), \
            "round robin: 0-3's speed stage has had no round at all, 0-1's has had one"
        assert [s.key for s in waiting] == [("Level 0-1", "speed"), ("Level 0-3", "speed")]
        # ... and 0-4 is never the answer while anything is waiting, for every round the cap allows.
        h.driver.state.history.append({"level": "Level 0-3", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        for _ in range(4):
            pick, waiting, _ = h.driver.choose_stage()
            assert pick.level != "Level 0-4" and waiting, "the line holds for as long as it takes"
            h.driver.state.history.append({"level": pick.level, "kind": pick.kind, "status": "unfinished",
                                           "round": h.driver.state.rounds(pick.key) + 1})
        # Once both are done the line opens by itself and the ladder walks on.
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "done", "round": 9})
        h.driver.state.history.append({"level": "Level 0-3", "kind": "speed", "status": "done", "round": 9})
        pick, waiting, _ = h.driver.choose_stage()
        assert waiting == [] and pick.key == ("Level 0-4", "complete")


def test_the_round_robin_takes_the_fewest_rounds_first_and_ties_in_plan_order():
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        picks = []
        for _ in range(6):
            pick, waiting, _ = h.driver.choose_stage()
            assert waiting, "the line is up the whole time"
            picks.append((pick.level, h.driver.state.rounds(pick.key) + 1))
            h.driver.state.history.append({"level": pick.level, "kind": pick.kind, "status": "unfinished",
                                           "round": h.driver.state.rounds(pick.key) + 1})
        assert picks == [("Level 0-1", 1), ("Level 0-3", 1), ("Level 0-1", 2), ("Level 0-3", 2),
                         ("Level 0-1", 3), ("Level 0-3", 3)], "strict alternation, ties in plan order"


def test_the_round_robin_stops_after_max_rounds_instead_of_looping_forever():
    """The 2026-09-18 review: the hold line's round robin had NO exit.

    A speed target the policy cannot reach -- and 0.75 x S on Level 0-1 is 90 s against a 183.6 s leaderboard
    best -- would otherwise make the driver burn `max_steps_per_stage` on the same stages for ever, with
    nobody told. After `max_rounds` the driver stops with a loud "held" so a human can retune the target.

    This pins the MECHANISM, on a plan that carries a cap (3, `StageRule`'s default). The SHIPPED plan sets
    `max_rounds: 0` under both kinds -- see the test below, and `test_specialists_config.py` -- because a
    "held" exit idles twelve games that nobody is watching.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        assert h.driver.plan.rule_for(cd.SPEED).max_rounds == 3
        for _ in range(6):
            pick, _, _ = h.driver.choose_stage()
            assert pick is not None
            h.driver.state.history.append({"level": pick.level, "kind": pick.kind, "status": "unfinished",
                                           "round": h.driver.state.rounds(pick.key) + 1})
        pick, waiting, blocked = h.driver.choose_stage()
        assert pick is None, "three rounds each, and neither is done: nothing may start"
        assert [s.key for s in waiting] == [("Level 0-1", "speed"), ("Level 0-3", "speed")]
        assert "had its 3 rounds" in blocked and "Level 0-1" in blocked and "Level 0-3" in blocked
        assert "Level 0-4" not in blocked, "the line still holds; the driver stops rather than walking past it"
        # The driver reports it and stops, rather than spinning or advancing.
        h.driver.state.current = None
        assert h.driver.tick() == "held"
        assert h.driver.run(max_ticks=1) == 1, "a nonzero exit: a human has to look at the target"
        # Lifting the cap makes it runnable again without touching anything else.
        h.driver.plan.speed["max_rounds"] = 0
        pick, _, blocked = h.driver.choose_stage()
        assert blocked == "" and pick.key == ("Level 0-1", "speed")


def test_with_no_round_cap_rounds_alone_never_hold_the_ladder():
    """`max_rounds: 0` (the shipped plan, 2026-09-18): NEVER IDLE THE MACHINE.

    A "held" exit is code 1 and the driver STOPS. Nobody watches this run -- there are no LLM monitors, by the
    user's own instruction -- so a stage that cannot reach its target would leave twelve games idle for hours
    or days, which is strictly worse than carrying on training the stages in front of the line. With no cap the
    round robin keeps handing out fresh budgets until the targets are met or a human lifts `hold_before`.

    It must still hold when nothing is eligible for a REASON the driver cannot train its way out of -- a speed
    stage whose own level has no finished complete stage has nothing to resume from -- because spinning on that
    would be worse than stopping.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        h.driver.plan.rule.max_rounds = 0          # `stage:` in the plan file
        h.driver.plan.speed["max_rounds"] = 0      # ... and `speed:`
        assert h.driver.plan.rule_for(cd.SPEED).max_rounds == 0
        assert h.driver.plan.rule_for(cd.COMPLETE).max_rounds == 0
        # Far more rounds than any cap would have allowed, and every one of them keeps training.
        for n in range(20):
            pick, waiting, blocked = h.driver.choose_stage()
            assert pick is not None and blocked == "", "round %d: rounds alone may never hold the ladder" % n
            assert h.driver.stage_blocked(pick) == ""
            assert waiting and pick.level != "Level 0-4", "the hold line itself still holds"
            h.driver.state.history.append({"level": pick.level, "kind": pick.kind, "status": "unfinished",
                                           "round": h.driver.state.rounds(pick.key) + 1})
        assert h.driver.state.rounds(("Level 0-1", cd.SPEED)) == 10, "ten rounds each and still going"
        assert h.driver.state.rounds(("Level 0-3", cd.SPEED)) == 10
        # ... and the driver does not stop: the next tick starts round 11 instead of exiting 1.
        h.driver.state.current = None
        assert h.driver.tick() == "started"
        assert h.driver.state.current is not None and h.driver.state.current.round == 11

        # BUT a stage blocked for a reason still holds, with no cap and no rounds at all: `--start-at` marked
        # both complete stages "skipped", which satisfies the line, and a speed stage will not start without a
        # "done" complete stage to resume from. No amount of training resolves that, so the driver stops.
        h.driver.state.current = None
        h.driver.state.history = [{"level": "Level 0-1", "kind": "complete", "status": "skipped", "round": 1},
                                  {"level": "Level 0-3", "kind": "complete", "status": "skipped", "round": 1}]
        pick, waiting, blocked = h.driver.choose_stage()
        assert pick is None and [s.key for s in waiting] == [("Level 0-1", "speed"), ("Level 0-3", "speed")]
        assert "complete stage is not done yet" in blocked and "rounds" not in blocked
        assert h.driver.tick() == "held" and h.driver.run(max_ticks=1) == 1


def test_a_speed_stage_is_not_eligible_until_its_own_level_is_done_and_has_a_specialist():
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        # 0-3's complete stage ran out of steps, so its speed stage has nothing to resume from.
        h.driver.state.history = [{"level": "Level 0-1", "kind": "complete", "status": "done", "round": 1},
                                  {"level": "Level 0-3", "kind": "complete", "status": "unfinished",
                                   "round": 1}]
        assert h.driver.stage_blocked(cd.StageSpec("Level 0-3", cd.SPEED)) == \
            "its complete stage is not done yet"
        assert h.driver.stage_blocked(cd.StageSpec("Level 0-1", cd.SPEED)) == ""
        assert h.driver.stage_blocked(cd.StageSpec("Level 0-3")) == "", "a complete stage is never blocked"
        # 0-3's speed stage is skipped over entirely; the other three take their turns by round count.
        pick, waiting, blocked = h.driver.choose_stage()
        assert blocked == "" and [s.key for s in waiting] == [
            ("Level 0-3", "complete"), ("Level 0-1", "speed"), ("Level 0-3", "speed")]
        assert pick.key == ("Level 0-1", "speed"), "fewest rounds first: it has had none, 0-3 complete has one"
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        pick, _, _ = h.driver.choose_stage()
        assert pick.key == ("Level 0-3", "complete"), \
            "now the rounds are level, so plan order decides -- and never the blocked 0-3 speed stage"
        # A done stage whose specialist file is missing is still not eligible: nothing to resume from.
        cd.specialist_path(h.tmp / "models", "Level 0-1").unlink()
        assert h.driver.stage_blocked(cd.StageSpec("Level 0-1", cd.SPEED)) == \
            "no specialist file for the level yet"


def test_a_round_resumes_from_the_stages_own_newest_weights():
    """Never from scratch and never from another level: a second round continues the one that ran."""
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        spec = cd.StageSpec("Level 0-1", cd.SPEED)
        model_dir = h.tmp / "models" / "spec_0-1_speed"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Nothing left in the stage's own directory: its promoted specialist is the fallback, not 0-3's.
        assert h.driver.round_init(spec) == cd.specialist_path(h.tmp / "models", "Level 0-1")
        # keep_best's peak, when the checkpoints were pruned.
        h.write_best("spec_0-1_speed", 13_000_000)
        assert h.driver.round_init(spec) == model_dir / "best.zip"
        # ... and once a checkpoint exists, what `ensure_trainer` will load, so `start_steps` agrees with it.
        ckpt = model_dir / "ckpt_13500000_steps.zip"
        ckpt.write_bytes(b"newest")
        h.zip_steps[ckpt.as_posix()] = 13_500_000
        assert h.driver.round_init(spec) == ckpt

        h.driver.begin_stage(spec.level, ckpt, spec.kind)
        stage = h.driver.state.current
        assert stage.round == 2 and stage.start_steps == 13_500_000, \
            "the round gets a fresh budget measured from where it resumes"
        generated = yaml.safe_load(
            (h.tmp / "configs" / "generated" / "spec_0-1_speed.yaml").read_text(encoding="utf-8"))
        assert generated["train"]["timesteps"] == 13_500_000 + 8_000_000 + cd.TIMESTEPS_SLACK


def test_a_finished_round_is_one_history_entry_and_the_next_round_follows_it():
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        h.driver.begin_stage("Level 0-1", cd.specialist_path(h.tmp / "models", "Level 0-1"), cd.SPEED)
        h.trainer_up("spec_0-1_speed")
        assert h.driver.tick() == "ok"
        # It runs out of steps without ever beating the clock: "unfinished", and the line refuses to advance.
        h.write_status("spec_0-1_speed", 21_000_000, 0.9, 50, best_time=200.0, target_seconds=90.0,
                       s_rank_seconds=120.0)
        h.write_best("spec_0-1_speed", 20_000_000)
        assert h.driver.tick() == "advanced"
        entry = h.driver.state.history[-1]
        assert (entry["level"], entry["kind"], entry["status"], entry["round"]) == \
            ("Level 0-1", "speed", "unfinished", 1)
        assert entry["target_seconds"] == 90.0 and entry["s_rank_seconds"] == 120.0
        # It ended at its cap without ever beating the clock, so it did NOT replace the level's specialist.
        assert entry["promoted"] is False and entry["specialist"] is None
        assert cd.specialist_path(h.tmp / "models", "Level 0-1").read_bytes() == b"complete-Level 0-1"
        assert not (h.tmp / "models" / "specialists" / "Level_0-1.json").exists()
        # The ladder did NOT walk to 0-4: it went to the other stage in front of the line.
        assert h.driver.state.current.key == ("Level 0-3", "speed")


def test_an_unfinished_speed_round_never_overwrites_the_levels_specialist():
    """THE 2026-09-18 REVIEW. `models/specialists/` is the only copy of a specialist that is committed.

    Before speed stages each level was promoted exactly once, so `promote()` could not regress anything. A
    speed stage runs on a level that already HAS a specialist and writes to the same path, and `keep_best
    --metric time` only has to clear `--min-rate 0.3` -- so a round that burns its 8M-step cap without ever
    beating the clock would replace a 0.48-rate policy with whatever it happened to hold. Nothing anywhere
    compares the two. The round's weights are not lost: they stay in the stage's own model directory, which is
    exactly where `round_init` resumes the next round from.
    """
    models = Path("models")
    # A complete stage promotes whatever it produced, as it always has, whatever the verdict.
    assert cd.refuse_promotion(models, "Level 0-1", kind=cd.COMPLETE, status="unfinished") == ""
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        good = cd.specialist_path(h.tmp / "models", "Level 0-1")
        assert cd.refuse_promotion(h.tmp / "models", "Level 0-1", kind=cd.SPEED, status="done") == "", \
            "a speed stage that MET its target is the whole point: it promotes"
        why = cd.refuse_promotion(h.tmp / "models", "Level 0-1", kind=cd.SPEED, status="unfinished")
        assert "unfinished" in why and "Level_0-1.zip" in why
        # ... and with no specialist there at all, any weights beat none.
        good.unlink()
        assert cd.refuse_promotion(h.tmp / "models", "Level 0-1", kind=cd.SPEED, status="unfinished") == ""

        # End to end: the round's own best.zip stays put and the promoted file is untouched.
        good.write_bytes(b"complete-Level 0-1")
        h.driver.begin_stage("Level 0-1", good, cd.SPEED)
        h.trainer_up("spec_0-1_speed")
        h.write_status("spec_0-1_speed", 21_000_000, 0.42, 50, best_time=85.0, median_time=400.0,
                       target_seconds=90.0)
        h.write_best("spec_0-1_speed", 20_000_000)
        assert h.driver.tick() == "advanced"
        assert good.read_bytes() == b"complete-Level 0-1", "the committed specialist is still the good one"
        assert (h.tmp / "models" / "spec_0-1_speed" / "best.zip").exists(), "the round's weights are kept"


def test_a_new_round_cannot_latch_on_the_previous_rounds_status_file():
    """The 2026-09-18 review: a round resets the latch but REUSES the run directory.

    `runs/<run>/status.json` still holds the previous round's final numbers -- the same rate, the same window,
    the same cumulative best -- until the new trainer overwrites it. Reading them on the first tick re-latches
    immediately, and the stage is recorded "done" a settle later on exactly the data its cap had just
    rejected, which makes the cap meaningless.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        # Round 1 ended at its 8M cap with a rate over the bar and a median inside the target.
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        h.write_status("spec_0-1_speed", 21_000_000, 0.9, 50, best_time=85.0, median_time=85.0,
                       target_seconds=90.0)
        model_dir = h.tmp / "models" / "spec_0-1_speed"
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt = model_dir / "ckpt_20950000_steps.zip"
        ckpt.write_bytes(b"round one's newest")
        h.zip_steps[ckpt.as_posix()] = 20_950_000

        stage = h.driver.begin_stage("Level 0-1", ckpt, cd.SPEED)
        assert stage.round == 2 and stage.stale_below == 21_000_000, \
            "everything already in that status.json belongs to round 1"
        h.trainer_up("spec_0-1_speed")
        assert h.driver.tick() == "ok"
        assert h.driver.state.current.target_reached_at is None, \
            "round 2 may not latch on round 1's tail, which round 1's own cap had just rejected"
        # ... and once the new trainer's own numbers pass it, the rule reads them normally.
        h.write_status("spec_0-1_speed", 21_100_000, 0.9, 50, best_time=85.0, median_time=85.0,
                       target_seconds=90.0)
        assert h.driver.tick() == "ok"
        assert h.driver.state.current.target_reached_at == 21_100_000
        # A stage that has never run has nothing stale to ignore.
        assert h.driver.begin_stage("Level 0-3", ckpt, cd.SPEED).stale_below is None


def test_a_stage_budget_is_measured_from_what_the_trainer_will_actually_resume():
    """The 2026-09-18 review: `--init` and `choose_resume` disagree, and the budget followed the wrong one.

    `ensure_trainer` resumes from the highest-step checkpoint in the stage's OWN model directory, not from the
    `--init` it was handed. When the directory is already ahead of that file, `start_steps` taken from `--init`
    makes the stage's cap and the generated `timesteps` total both start from an origin the trainer never
    visits: the "6M-step" round is really 6M minus the gap, and `learn()` stops that much early.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        model_dir = h.tmp / "models" / "spec_0-1"
        model_dir.mkdir(parents=True, exist_ok=True)
        newest = model_dir / "ckpt_18362686_steps.zip"
        newest.write_bytes(b"where the trainer will really start")
        h.zip_steps[newest.as_posix()] = 18_362_686

        stage = h.driver.begin_stage("Level 0-1", h.init, cd.COMPLETE)  # h.init reads 10,000,000 steps
        assert stage.start_steps == 18_362_686, "the resume file's count, not the init's"
        generated = yaml.safe_load(
            (h.tmp / "configs" / "generated" / "spec_0-1.yaml").read_text(encoding="utf-8"))
        assert generated["train"]["timesteps"] == 18_362_686 + 6_000_000 + cd.TIMESTEPS_SLACK
        assert (model_dir / "latest.zip").exists() is False, "nothing to seed: it already has a resume file"
    # With an empty directory the init IS what the trainer resumes from, exactly as before.
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        assert h.driver.begin_stage("Level 0-1", h.init, cd.COMPLETE).start_steps == 10_000_000
        assert (h.tmp / "models" / "spec_0-1" / "latest.zip").exists(), "and the init is seeded as before"


def test_start_at_respects_the_hold_line_and_refuses_a_stage_that_already_ran():
    """The two holes `--start-at` left in the hold line (2026-09-18 review). It bypasses `choose_stage`."""
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        plan, driver = h.driver.plan, h.driver
        # 0-4 sits at the line, and two stages in front of it are not done.
        why = cd.start_at_objection(plan, driver, "Level 0-4", cd.COMPLETE, plan_path="configs/x.yaml")
        assert "hold line before Level 0-4" in why and "Level 0-1 (speed)" in why
        assert "--ignore-hold" in why and "configs/x.yaml" in why
        assert cd.start_at_objection(plan, driver, "Level 0-4", cd.COMPLETE, ignore_hold=True) == "", \
            "an operator may lift it on purpose; the driver logs that it did"
        # A stage in FRONT of the line is fine, as long as it has not already run.
        assert cd.start_at_objection(plan, driver, "Level 0-3", cd.SPEED) == ""
        # The finished-stage guard: this is what `runs/start_driver.cmd` would do after a "held" exit.
        again = cd.start_at_objection(plan, driver, "Level 0-1", cd.COMPLETE)
        assert "already run" in again and "'done'" in again and "--rerun-stage" in again
        assert cd.start_at_objection(plan, driver, "Level 0-1", cd.COMPLETE, rerun_stage=True) == ""
        # An unfinished stage has run too: a second round is a decision, not a restart's side effect.
        assert "already run" in cd.start_at_objection(plan, driver, "Level 0-1", cd.SPEED)
        # With no line in the plan at all, only the finished-stage guard is left.
        open_plan = cd.load_plan(write_plan(Path(tmp), order=SPEED_ORDER))
        assert cd.start_at_objection(open_plan, driver, "Level 0-4", cd.COMPLETE) == ""


def test_specialists_status_reports_the_hold_line_and_the_round():
    with tempfile.TemporaryDirectory() as tmp:
        h = held_harness(tmp)
        h.driver.state.history.append({"level": "Level 0-1", "kind": "speed", "status": "unfinished",
                                       "round": 1})
        h.driver.begin_stage("Level 0-3", cd.specialist_path(h.tmp / "models", "Level 0-3"), cd.SPEED)
        h.driver.save_state()
        import specialists_status  # noqa: PLC0415 - a script, imported only where it is tested

        data = specialists_status.collect(h.tmp, "configs/specialists.yaml", "runs", "models")
        assert data["hold_before"] == "Level 0-4"
        assert [(x["level"], x["kind"], x["rounds"], x["status"]) for x in data["held_by"]] == [
            ("Level 0-1", "speed", 1, "unfinished"), ("Level 0-3", "speed", 0, None)]
        # `rounds` counts ENDED rounds, so the stage that is running right now would otherwise read
        # "round 0, not started" in the same report that prints it as the live stage two lines above.
        assert [x["current"] for x in data["held_by"]] == [False, True]
        text = specialists_status.render(data)
        assert "holding before Level 0-4: waiting on Level 0-1 (speed, round 1, unfinished)" in text
        assert "Level 0-3 (speed, round 0, RUNNING NOW)" in text
        assert "round 1]" in text, "the running stage says which round it is"


def test_the_real_live_state_file_loads_into_the_new_plan_and_keeps_stage_three_running():
    """The state file the live driver was using when stage kinds shipped (a literal copy), against the new plan.

    Stage 3 is `Level 0-3` complete, started at 18,752,038 steps, with 0-1 and 0-2 done. The three speed
    stages and the hold line were added underneath it, and none of that may disturb it: `reconcile` renumbers
    it, `tick` keeps driving it, and `choose_stage` is not consulted at all while a stage is current.
    """
    raw = json.loads(json.dumps(PRE_KINDS_STATE))  # a deep copy: the driver may back-fill keys in place
    assert raw["current"]["level"] == "Level 0-3" and "kind" not in raw["current"], \
        "this test is about a state file written BEFORE stage kinds existed"
    with tempfile.TemporaryDirectory() as tmp:
        order = ["Level 0-1", "Level 0-2", "Level 0-3",
                 {"level": "Level 0-1", "kind": "speed"}, {"level": "Level 0-2", "kind": "speed"},
                 {"level": "Level 0-3", "kind": "speed"}, "Level 0-4", "Level 0-5"]
        h = harness(tmp)
        h.plan = cd.load_plan(write_plan(Path(tmp), order=order, hold_before="Level 0-4"))
        state_path = h.tmp / "runs" / "specialists" / "driver_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(raw), encoding="utf-8")
        driver = cd.Driver(cd.DriverConfig(plan_path=str(h.tmp / "configs" / "specialists.yaml"), cwd=h.tmp,
                                           python="PY.EXE", start_grace_seconds=0.0),
                           h.plan,
                           processes=lambda: list(h.procs), now=lambda: h.time, sleep=h.slept.append,
                           probe=lambda path, now: h.status, kill=h.killed.append, spawn=h._spawn,
                           stop_games=h._stop, launch_games=h._launch,
                           zip_steps=lambda path: h.zip_steps.get(Path(path).as_posix()),
                           working_sets=lambda: dict(h.sets), port_pids=lambda: dict(h.ports),
                           relaunch_one=lambda port: True, established=lambda ports: {p: 1 for p in ports},
                           cpu=lambda: {}, copy=h._copy, pid=SELF_PID)
        assert driver.unplanned == [], "every (level, kind) in the live file is still in the plan"
        stage = driver.state.current
        assert (stage.level, stage.kind, stage.run) == ("Level 0-3", "complete", "spec_0-3")
        assert stage.index == 2, "0-3's complete stage is still the third stage of the new plan"
        assert stage.start_steps == 18_752_038.0 and stage.round == 1
        assert stage.target_seconds is None and stage.target_reached_at is None
        assert [(e["level"], e["kind"], e["index"], e["round"]) for e in driver.state.history] == [
            ("Level 0-1", "complete", 0, 1), ("Level 0-2", "complete", 1, 1)]
        # The hold line is up -- the running stage counts as not-done too, which is what it is -- but it
        # changes NOTHING while a stage is current: `tick` never consults `choose_stage` until one ends.
        assert [s.key for s in driver.held_by()] == [
            ("Level 0-3", "complete"), ("Level 0-1", "speed"), ("Level 0-2", "speed"),
            ("Level 0-3", "speed")]
        h.trainer_up("spec_0-3")
        h.write_status("spec_0-3", 19_000_000, 0.2, 40)
        assert driver.tick() == "ok", "the live stage just keeps running"
        assert driver.state.current is not None and driver.state.current.key == ("Level 0-3", "complete")
        assert h.trainer_commands == [], "no trainer was started: one is already running that stage"
        assert h.killed == [] and h.launched == [] and h.stopped == 0
        # And the copy on disk still describes the same stage after a save.
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert saved["current"]["level"] == "Level 0-3" and saved["current"]["start_steps"] == 18_752_038.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
