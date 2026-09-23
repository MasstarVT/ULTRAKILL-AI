"""The FLEET-level refusal of a mod that cannot serve the config: `runs/<run>/MOD_INCOMPATIBLE`.

A `tech_layout: v2` env refusing a DLL without the 0.8.0 features writes that file into its run directory before
it raises `BridgeIncompatible`. The supervisor's restart and the driver's trainer start both read it first and
then start NOTHING -- no game, no trainer, no helper -- and exit with code 4; nothing but the operator removes it;
`check_run.py` puts it in ALERTS. Before this, the driver respawned the dying trainer at every poll and
`supervise.py` relaunched all twelve games up to three times an hour, for a configuration error.

No game, no real process, no real clock:  python tests/test_mod_incompatible.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run  # noqa: E402
import supervise  # noqa: E402
import test_campaign_driver as tcd  # noqa: E402
import test_supervise as tsu  # noqa: E402
from supervise import Status  # noqa: E402
from test_campaign_env import make_env  # noqa: E402
from test_tech_env import tech_env  # noqa: E402

from ultrakill_ai.protocol import (  # noqa: E402
    MOD_INCOMPATIBLE_FILE,
    BridgeIncompatible,
    read_mod_incompatible,
    write_mod_incompatible,
)

RUN = "spec_0-1_speed"


def refuse(run_dir) -> str:
    """A v2 env against FakeLevel (mod 0.5.0, no `features`) with `run_dir` as its run directory; the refusal text."""
    env, fake = make_env(tech_layout="v2", env_log_dir=str(run_dir))
    try:
        env.reset()
    except BridgeIncompatible as exc:
        assert fake.configures == 0
        return str(exc)
    finally:
        env.close()
    raise AssertionError("a v2 env accepted a mod without the 0.8.0 features")


# ---------------------------------------------------------------------------------------------
# The env writes it
# ---------------------------------------------------------------------------------------------


def test_a_refused_mod_leaves_the_file_in_the_run_directory():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        message = refuse(run_dir)
        text = (run_dir / MOD_INCOMPATIBLE_FILE).read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == message, "the first line is the refusal itself: check_run.py prints exactly that"
        assert "mod_version: 0.5.0" in lines and "features_seen: none" in lines, lines
        assert "features_missing: macro.ssj, monotonic_input_clock, obs.move_tech" in lines, lines
        assert "tech_layout: v2" in lines
        assert "Remove %s" % (run_dir / MOD_INCOMPATIBLE_FILE) in text and "installing the mod" in text
        assert "code 4" in text
        assert not list(run_dir.glob("*.tmp")), "the atomic write leaves no temp file behind"
        assert "mod_incompatible_file" in (run_dir / "env_47800.log").read_text(encoding="utf-8")


def test_an_accepted_mod_writes_nothing_and_never_deletes_an_old_file():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        env, _ = tech_env(env_log_dir=str(run_dir))  # FakeTechLevel: mod 0.8.0, every feature
        try:
            env.reset()
        finally:
            env.close()
        assert not (run_dir / MOD_INCOMPATIBLE_FILE).exists()
        # The mod is installed but the operator has not removed the file yet: a good connect must not do it for
        # them -- only the operator may declare the fleet fixed.
        write_mod_incompatible(run_dir, "left by an earlier refusal\n")
        env, _ = tech_env(env_log_dir=str(run_dir))
        try:
            env.reset()
        finally:
            env.close()
        assert read_mod_incompatible(run_dir) == "left by an earlier refusal\n"


def test_todays_v1_env_on_an_old_mod_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        env, _ = make_env(env_log_dir=str(run_dir))  # v1 against FakeLevel's 0.5.0: nothing required
        try:
            env.reset()
        finally:
            env.close()
        assert not (run_dir / MOD_INCOMPATIBLE_FILE).exists()


def test_without_a_run_directory_the_env_only_raises():
    """eval.py, campaign_check.py and every test build an EnvConfig with no env_log_dir: no file, anywhere."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            refuse("")
        finally:
            os.chdir(cwd)
        assert list(Path(tmp).rglob(MOD_INCOMPATIBLE_FILE + "*")) == []


def test_the_file_helpers_replace_whole_and_read_presence():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        assert read_mod_incompatible(run_dir) is None
        assert write_mod_incompatible(run_dir, "first\n") == (run_dir / MOD_INCOMPATIBLE_FILE, None)
        assert write_mod_incompatible(run_dir, "second\n") == (run_dir / MOD_INCOMPATIBLE_FILE, None)
        assert read_mod_incompatible(run_dir) == "second\n", "a later refusal replaces the report whole"
        assert sorted(p.name for p in run_dir.iterdir()) == [MOD_INCOMPATIBLE_FILE]
        blocker = Path(tmp) / "a_file"
        blocker.write_text("", encoding="utf-8")
        written, error = write_mod_incompatible(blocker / "run", "x")  # a FILE where the run directory should be
        assert written is None and error and error.split(":")[0] in ("FileExistsError", "NotADirectoryError"), error


def test_a_file_that_cannot_be_written_is_said_in_the_env_log_and_the_exception():
    """Unwritten, the file restores the respawn loop it exists to stop: that must reach runs/<run>_train.log."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        (run_dir / MOD_INCOMPATIBLE_FILE).mkdir(parents=True)  # a DIRECTORY where the file should go
        message = refuse(run_dir)
        assert "could NOT be written" in message and "DRIVER_PAUSE" in message, message
        error = message.split("could NOT be written (")[1].split(")")[0]
        assert error.split(":")[0] in ("PermissionError", "IsADirectoryError"), error
        env_log = (run_dir / "env_47800.log").read_text(encoding="utf-8")
        line = next(ln for ln in env_log.splitlines() if "mod_incompatible_file" in ln)
        assert "written=False" in line and error.split(":")[0] in line, line
        assert not list(run_dir.glob("*.tmp")), "the temp file is cleaned up"


def test_a_written_file_leaves_the_exception_as_it_was():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / RUN
        message = refuse(run_dir)
        assert "could NOT be written" not in message and message.endswith("set tech_layout back to v1.")
        line = next(ln for ln in (run_dir / "env_47800.log").read_text(encoding="utf-8").splitlines()
                    if "mod_incompatible_file" in ln)
        assert "written=True" in line and "error=" not in line, line


# ---------------------------------------------------------------------------------------------
# The supervisor reads it
# ---------------------------------------------------------------------------------------------


def test_the_supervisor_relaunches_nothing_while_the_file_exists():
    h = tsu.harness(tsu.BYSTANDERS + tsu.HELPERS, Status(1800.0, "stopped", 6_901_546))  # a dead trainer
    refuse(h.sup.run_dir)
    assert h.sup.tick() == "mod_incompatible"
    assert (h.killed, h.spawned, h.launched, h.stopped) == ([], [], [], 0), "no game stopped or launched"
    assert h.sup.restarts == [], "no restart budget spent on a configuration error"
    log = h.sup.log_path.read_text(encoding="utf-8")
    assert "MOD INCOMPATIBLE" in log and "mod_version: 0.5.0" in log and "remove " in log, log
    assert h.sup.run(max_ticks=5) == supervise.EXIT_MOD_INCOMPATIBLE == 4
    assert (h.spawned, h.launched, h.stopped) == ([], [], 0)
    assert h.sup.mod_incompatible_path.exists(), "never deleted by the supervisor"
    h.sup.mod_incompatible_path.unlink()  # the operator, after installing the mod
    assert h.sup.tick() == "restarted", "the file was the only thing in the way"


def test_a_hung_trainer_is_left_alone_too():
    """Nothing is killed either: the refusing worker's trainer exits by itself, and the operator decides."""
    h = tsu.harness(tsu.BYSTANDERS + tsu.TRAINER_TREE + tsu.HELPERS, Status(1200.0, "running", 6_901_546))
    refuse(h.sup.run_dir)
    assert h.sup.tick() == "mod_incompatible"
    assert (h.killed, h.spawned, h.launched, h.stopped) == ([], [], [], 0)


def test_the_exit_code_is_not_one_any_other_stop_uses():
    assert supervise.EXIT_MOD_INCOMPATIBLE not in (0, 1, 2, 3), "0/1 run(), 2 argparse, 3 train.py's layout refusal"


# ---------------------------------------------------------------------------------------------
# The driver reads it
# ---------------------------------------------------------------------------------------------


def test_the_driver_starts_no_game_and_no_trainer_while_the_file_exists():
    with tempfile.TemporaryDirectory() as tmp:
        h = tcd.harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.ports = {}  # no game listening: without the file this very tick would launch all twelve
        run_dir = h.driver.stage_run_dir("Level 0-1")
        refuse(run_dir)
        state_path = h.tmp / "runs" / "specialists" / "driver_state.json"
        before = json.loads(state_path.read_text(encoding="utf-8"))
        assert h.driver.tick() == "mod_incompatible"
        assert (h.spawned, h.killed, h.launched, h.stopped) == ([], [], [], 0), "no game, trainer or helper"
        assert json.loads(state_path.read_text(encoding="utf-8")) == before, \
            "not a stage outcome: `current` and the history are untouched, so the fixed fleet resumes this round"
        assert h.driver.run(max_ticks=5) == supervise.EXIT_MOD_INCOMPATIBLE
        assert h.slept == [] and h.spawned == [], "it exits at once instead of polling"
        log = (h.tmp / "runs" / "specialists_driver.log").read_text(encoding="utf-8")
        assert "MOD INCOMPATIBLE" in log and "mod_version: 0.5.0" in log and "remove " in log
        assert (run_dir / MOD_INCOMPATIBLE_FILE).exists(), "never deleted by the driver"
        (run_dir / MOD_INCOMPATIBLE_FILE).unlink()  # the operator, after installing the mod
        assert h.driver.tick() == "started"
        assert h.launched == [(12, 1)] and len(h.trainer_commands) == 1


def test_the_stage_supervisors_restart_refuses_and_the_driver_stops_with_it():
    """A trainer that is up but stale goes through `sup.tick()` -> `restart`, the path that stops all twelve games."""
    with tempfile.TemporaryDirectory() as tmp:
        h = tcd.harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        assert h.driver.tick() == "started"
        h.trainer_up("spec_0-1")
        h.status = Status(1800.0, "running", 11_000_000)
        h.spawned.clear()
        refuse(h.driver.stage_run_dir("Level 0-1"))
        assert h.driver.tick() == "mod_incompatible"
        assert (h.spawned, h.killed, h.launched, h.stopped) == ([], [], [], 0)
        assert h.driver.run(max_ticks=5) == supervise.EXIT_MOD_INCOMPATIBLE


def test_another_runs_file_does_not_stop_this_stage():
    with tempfile.TemporaryDirectory() as tmp:
        h = tcd.harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        refuse(h.tmp / "runs" / "spec_0-3")
        assert h.driver.tick() == "started"


def test_a_stage_end_into_a_refused_run_stops_at_once_and_says_it_once():
    """`finish_stage` starts the NEXT stage's trainer; when that run holds the file the driver exits on that very
    tick, instead of returning "advanced" and logging the whole block again at the next poll."""
    with tempfile.TemporaryDirectory() as tmp:
        h = tcd.harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        h.driver.tick()
        h.trainer_up("spec_0-1")
        h.write_status("spec_0-1", 11_000_000, 0.55, 40, best_time=131.25)
        h.write_best("spec_0-1", 10_950_000)
        assert h.driver.tick() == "ok"  # the target latches; the settle is still owed
        refuse(h.driver.stage_run_dir("Level 0-3"))  # the next stage's run
        h.write_status("spec_0-1", 11_400_000, 0.48, 50, best_time=131.25)
        h.spawned.clear()
        assert h.driver.run(max_ticks=5) == supervise.EXIT_MOD_INCOMPATIBLE
        assert h.driver.state.current.level == "Level 0-3", "the next stage is begun and kept for after the fix"
        assert [e["level"] for e in h.driver.state.history] == ["Level 0-1"], "the finished stage is recorded"
        assert h.trainer_commands == [] and h.launched == []
        log = (h.tmp / "runs" / "specialists_driver.log").read_text(encoding="utf-8")
        assert log.count("Exiting with code 4") == 1 and log.count("MOD INCOMPATIBLE") == 1, log


# ---------------------------------------------------------------------------------------------
# check_run.py reports it
# ---------------------------------------------------------------------------------------------


def test_check_run_alerts_on_the_file_the_env_wrote():
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        assert check_run.mod_incompatible_alerts(runs) == []
        message = refuse(runs / RUN)
        assert check_run.mod_incompatible_alerts(runs) == ["MOD_INCOMPATIBLE exists for %s: %s" % (RUN, message)]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
