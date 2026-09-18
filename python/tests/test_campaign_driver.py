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
           target_seconds=None) -> cd.StageSample:
    return cd.StageSample(timesteps, rate, window, best_time, best_at, target_seconds)


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


def write_plan(tmp: Path, *, order=LEVELS, speed: dict | None = None) -> Path:
    path = tmp / "configs" / "specialists.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "order": list(order),
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
                     best_time: float | None = None, target_seconds: float | None = None) -> None:
        path = self.tmp / "runs" / run / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign = {"fresh_completion_rate": rate, "fresh_window": window, "best_time": best_time,
                    "target_seconds": target_seconds}
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
        assert speed["train"]["run_name"] == "spec_0-1_speed"
        assert speed["train"]["timesteps"] == 10_000_000 + 8_000_000 + cd.TIMESTEPS_SLACK, "the speed cap"
        # Every reward weight and env setting is the complete stage's: the same policy continues into it.
        assert {k: v for k, v in speed["env"].items() if k != "speed_bonus"} == complete["env"]
        overridden = cd.stage_config(plan, "Level 0-3", kind=cd.SPEED)
        assert overridden["env"]["speed_target_seconds"] == 95.0
        path = cd.write_stage_config(plan, "Level 0-1", root, kind=cd.SPEED)
        assert path.name == "spec_0-1_speed.yaml"
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["speed_bonus"] is True


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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
