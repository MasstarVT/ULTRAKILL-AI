"""Tests for scripts/supervise.py, the crash supervisor (no game, no real process, no real clock).

Every side effect the supervisor has -- listing processes, reading status.json's mtime, killing a pid,
stopping and launching the games, spawning a detached trainer -- is injected as a function, so these
tests drive the whole decision machine against fakes. Nothing here starts a game, a trainer or a timer.

The command lines below are the real ones, copied from the live `campaign_gates` run: the trainer is a
chain of three processes (a cmd.exe wrapper, the venv python shim and the real python) plus twelve
SubprocVecEnv workers whose command lines mention neither the script nor the run.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import supervise  # noqa: E402
from supervise import Config, Proc, Status, Supervisor  # noqa: E402

RUN = "campaign_gates"
CONFIG = "configs/campaign_gates_main.yaml"

CMD_WRAPPER = ('"C:\\WINDOWS\\system32\\cmd.exe" /c "F:\\Github\\ULTRAKILL-AI\\python\\.venv\\Scripts\\python.exe" '
               "-u scripts/train.py --config configs/campaign_gates_main.yaml "
               "--resume models/campaign_gates/latest.zip >> runs\\campaign_gates_train.log 2>&1")
CMD_SHIM = ("F:\\Github\\ULTRAKILL-AI\\python\\.venv\\Scripts\\python.exe -u scripts/train.py "
            "--config configs/campaign_gates_main.yaml --resume models/campaign_gates/latest.zip")
CMD_TRAINER = ('"C:\\Users\\tyler\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -u scripts/train.py '
               "--config configs/campaign_gates_main.yaml --resume models/campaign_gates/latest.zip")
CMD_WORKER = ('"python.exe" "-c" "from multiprocessing.spawn import spawn_main; '
              'spawn_main(parent_pid=32324, pipe_handle=1536)" "--multiprocessing-fork"')
CMD_POLL = "F:\\...\\python.exe -u scripts/poll_status.py --run campaign_gates"
CMD_KEEP_BEST = "F:\\...\\python.exe -u scripts/keep_best.py --run campaign_gates --metric campaign"
# The supervisor itself, and the query it uses to list processes. Both MENTION the trainer; neither IS one.
CMD_SELF = ('"C:\\WINDOWS\\system32\\cmd.exe" /c "F:\\...\\python.exe" -u scripts/supervise.py --run campaign_gates '
            "--config configs/campaign_gates_main.yaml --count 12 --monitor 1 >> runs\\campaign_gates_supervisor.log")
CMD_QUERY = ('powershell -NoProfile -Command "@(Get-CimInstance -ClassName Win32_Process | '
             'Select-Object ProcessId,CommandLine) | ConvertTo-Json"  # campaign_gates train.py')

SELF_PID = 999
TRAINER_TREE = [
    Proc(23480, 10352, CMD_WRAPPER),
    Proc(7220, 23480, CMD_SHIM),
    Proc(32324, 7220, CMD_TRAINER),
    Proc(30024, 32324, CMD_WORKER),
    Proc(5016, 32324, CMD_WORKER),
]
HELPERS = [Proc(13504, 4728, CMD_POLL), Proc(21572, 20576, CMD_KEEP_BEST)]
BYSTANDERS = [
    Proc(SELF_PID, 111, CMD_SELF),
    Proc(1234, SELF_PID, CMD_QUERY),
    Proc(555, 1, "python.exe ugraph.py 0-1 8-1 1-3"),
    # Another run's trainer: same script, different run and config.
    Proc(666, 1, "python.exe -u scripts/train.py --config configs/cybergrind.yaml --resume "
                 "models/cybergrind_ppo_v2/best.zip"),
]


class Harness:
    """A supervisor wired to fakes, with every side effect recorded."""

    def __init__(self, tmp: Path, procs, status: Status, *, steps=(6_800_000, 6_850_000), **cfg_kwargs):
        self.tmp = tmp
        self.procs = list(procs)
        self.status = status
        self.time = 1_000_000.0
        self.killed: list[int] = []
        self.spawned: list[str] = []
        self.launched: list[tuple[int, int]] = []
        self.stopped = 0
        self.slept: list[float] = []
        # The boot health gate. By default every one of the twelve games is over the line from the first poll,
        # so a restart test does not have to care about it; a boot test overrides `sets`.
        self.ports = {47800 + i: 23000 + i for i in range(12)}
        self.sets = {pid: 1000 * supervise.MB for pid in self.ports.values()}
        self.relaunched: list[int] = []
        self.polls = 0
        # The restart report (established connections per port, CPU delta per pid). Injected like everything
        # else: without it a test would shell out to netstat and CIM on the machine running the suite, and
        # read the LIVE training run's ports.
        self.established = dict.fromkeys(self.ports, 0)
        self.cpu = dict.fromkeys(self.ports.values(), 0.0)

        (tmp / "runs" / RUN).mkdir(parents=True, exist_ok=True)
        model_dir = tmp / "models" / RUN
        model_dir.mkdir(parents=True, exist_ok=True)
        for n in steps:
            (model_dir / f"ckpt_{n}_steps.zip").write_bytes(b"")
        (model_dir / "latest.zip").write_bytes(b"")
        self.latest_steps = 6_771_574

        kwargs = dict(run=RUN, config=CONFIG, count=12, monitor=1, cwd=tmp,
                      start_grace_seconds=0.0, python="PY.EXE")
        kwargs.update(cfg_kwargs)
        self.sup = Supervisor(
            Config(**kwargs),
            processes=lambda: list(self.procs),
            now=lambda: self.time,
            sleep=self._sleep,
            probe=lambda path, now: self.status,
            kill=self.killed.append,
            spawn=self._spawn,
            stop_games=self._stop,
            launch_games=self._launch,
            zip_steps=lambda path: self.latest_steps,
            working_sets=self._working_sets,
            port_pids=lambda: dict(self.ports),
            relaunch_one=self._relaunch_one,
            established=lambda ports: {p: self.established.get(p, 0) for p in ports},
            cpu=lambda: dict(self.cpu),
            pid=SELF_PID,
        )

    def _sleep(self, seconds: float) -> None:
        """Records the wait AND advances the fake clock, so anything that waits on a deadline terminates."""
        self.slept.append(seconds)
        self.time += seconds

    def _working_sets(self) -> dict[int, int]:
        self.polls += 1
        return dict(self.sets)

    def _relaunch_one(self, port: int) -> bool:
        self.relaunched.append(port)
        self.sets[self.ports[port]] = 1000 * supervise.MB  # the restarted copy boots properly
        return True

    def _spawn(self, command: str, cwd: Path) -> int:
        self.spawned.append(command)
        return 4242

    def _stop(self) -> None:
        self.stopped += 1

    def _launch(self, count: int, monitor: int) -> bool:
        self.launched.append((count, monitor))
        return True

    @property
    def trainer_commands(self) -> list[str]:
        return [c for c in self.spawned if "train.py" in c]


def harness(procs, status, **kw):
    tmp = Path(tempfile.mkdtemp())
    return Harness(tmp, procs, status, **kw)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_the_supervisor_never_matches_its_own_command_line_or_its_own_query():
    # Both of these contain "train.py", the run name and the config name: a substring match alone
    # reports a trainer that does not exist, which is exactly how an earlier report script miscounted.
    assert supervise.matches_script(CMD_SELF, "train.py", RUN, CONFIG) is False
    assert supervise.matches_script(CMD_QUERY, "train.py", RUN, CONFIG) is False
    assert supervise.matches_script(CMD_TRAINER, "train.py", RUN, CONFIG) is True
    assert supervise.matches_script(CMD_WRAPPER, "train.py", RUN, CONFIG) is True
    assert supervise.matches_script(CMD_WORKER, "train.py", RUN, CONFIG) is False
    # Another run's trainer is not this run's trainer.
    assert supervise.matches_script(BYSTANDERS[3].cmdline, "train.py", RUN, CONFIG) is False
    # The helpers match on the run name alone (they take no config).
    assert supervise.matches_script(CMD_POLL, "poll_status.py", RUN) is True
    assert supervise.matches_script(CMD_KEEP_BEST, "keep_best.py", RUN) is True
    assert supervise.matches_script(CMD_POLL, "keep_best.py", RUN) is False


def test_the_supervisors_own_tree_is_excluded_by_pid_as_well():
    procs = BYSTANDERS + TRAINER_TREE
    assert supervise.self_and_ancestors(procs, SELF_PID) == {SELF_PID, 111}
    assert 1234 in supervise.self_and_ancestors(procs, 1234)  # the query process sits under the supervisor


def test_descendants_finds_the_subproc_workers():
    found = supervise.descendants(TRAINER_TREE, [23480])
    assert set(found) == {7220, 32324, 30024, 5016}
    assert found.index(30024) < found.index(32324)  # deepest first, so a parent cannot orphan its children


# ---------------------------------------------------------------------------
# Resume-file choice
# ---------------------------------------------------------------------------


def _model_dir(tmp: Path, ckpts: list[int], latest: bool = True) -> Path:
    d = tmp / "models" / RUN
    d.mkdir(parents=True, exist_ok=True)
    for n in ckpts:
        (d / f"ckpt_{n}_steps.zip").write_bytes(b"")
    if latest:
        (d / "latest.zip").write_bytes(b"")
    return d


def test_resume_picks_the_checkpoint_when_latest_is_behind():
    # The live case after a hard kill: latest.zip was last written at the previous graceful stop.
    with tempfile.TemporaryDirectory() as tmp:
        d = _model_dir(Path(tmp), [6_755_630, 6_821_566, 6_871_558])
        path, steps = supervise.choose_resume(d, zip_steps=lambda p: 6_771_574)
        assert path.name == "ckpt_6871558_steps.zip" and steps == 6_871_558


def test_resume_picks_latest_when_it_is_ahead():
    # The graceful case: Ctrl+C wrote latest.zip past the newest checkpoint.
    with tempfile.TemporaryDirectory() as tmp:
        d = _model_dir(Path(tmp), [6_755_630, 6_800_000])
        path, steps = supervise.choose_resume(d, zip_steps=lambda p: 6_812_345)
        assert path.name == "latest.zip" and steps == 6_812_345


def test_an_unreadable_latest_drops_out_and_the_newest_checkpoint_wins():
    with tempfile.TemporaryDirectory() as tmp:
        d = _model_dir(Path(tmp), [6_755_630, 6_800_000])
        path, steps = supervise.choose_resume(d, zip_steps=lambda p: None)  # truncated mid-save
        assert path.name == "ckpt_6800000_steps.zip" and steps == 6_800_000
        # And with no checkpoints at all there is nothing to resume from: never start from scratch.
        empty = _model_dir(Path(tmp) / "empty", [], latest=False)
        assert supervise.choose_resume(empty, zip_steps=lambda p: None) == (None, None)


def test_the_step_count_is_read_out_of_a_real_sb3_zip():
    # SB3 keeps the model's scalars as JSON in the archive's `data` member.
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "latest.zip"
        with zipfile.ZipFile(good, "w") as z:
            z.writestr("data", json.dumps({"num_timesteps": 6_771_574, "n_envs": 12}))
            z.writestr("policy.pth", b"not really a tensor")
        assert supervise.zip_timesteps(good) == 6_771_574
        bad = Path(tmp) / "bad.zip"
        bad.write_bytes(b"this is not a zip")
        assert supervise.zip_timesteps(bad) is None
        assert supervise.checkpoint_steps("ckpt_6871558_steps.zip") == 6_871_558
        assert supervise.checkpoint_steps("best.zip") is None


# ---------------------------------------------------------------------------
# The decisions
# ---------------------------------------------------------------------------


def test_a_healthy_run_is_left_completely_alone():
    h = harness(BYSTANDERS + TRAINER_TREE + HELPERS, Status(12.0, "running", 6_901_546))
    assert h.sup.tick() == "ok"
    assert (h.killed, h.spawned, h.launched, h.stopped) == ([], [], [], 0)


def test_a_dead_trainer_is_restarted_once_with_the_best_resume_file():
    h = harness(BYSTANDERS + HELPERS, Status(1800.0, "stopped", 6_901_546))
    assert h.sup.tick() == "restarted"
    assert h.killed == []                      # nothing left to kill
    assert h.stopped == 1 and h.launched == [(12, 1)]   # games stopped, then 12 games on monitor 1
    assert len(h.trainer_commands) == 1
    cmd = h.trainer_commands[0]
    assert "--config configs/campaign_gates_main.yaml" in cmd
    assert "ckpt_6850000_steps.zip" in cmd      # the most timesteps of latest.zip and the ckpt_* files
    assert cmd.startswith('cmd.exe /s /c "') and cmd.endswith('2>&1"')
    assert ">> runs\\campaign_gates_train.log" in cmd


def test_a_hung_trainer_is_killed_with_its_workers_before_the_restart():
    # The 2026-09-17 failure: the process is still there, but status.json stopped moving.
    h = harness(BYSTANDERS + TRAINER_TREE + HELPERS, Status(1200.0, "running", 6_901_546))
    assert h.sup.tick() == "restarted"
    assert set(h.killed) == {23480, 7220, 32324, 30024, 5016}
    assert h.killed.index(30024) < h.killed.index(23480)  # workers before the root
    assert h.stopped == 1 and h.launched == [(12, 1)] and len(h.trainer_commands) == 1


def test_a_trainer_saving_on_a_graceful_stop_is_not_killed():
    # state is already "stopped" but status.json is fresh: train.py's finally is writing latest.zip.
    h = harness(BYSTANDERS + TRAINER_TREE + HELPERS, Status(3.0, "stopped", 6_901_546))
    assert h.sup.tick() == "draining"
    assert (h.killed, h.spawned, h.launched) == ([], [], [])


def test_the_pause_file_stops_the_supervisor_doing_anything():
    h = harness(BYSTANDERS, Status(9999.0, "stopped", 6_901_546))  # as dead as it gets
    h.sup.pause_path.write_text("planned pause", encoding="utf-8")
    assert h.sup.tick() == "paused"
    assert (h.killed, h.spawned, h.launched, h.stopped) == ([], [], [], 0)
    h.sup.pause_path.unlink()
    assert h.sup.tick() == "restarted"  # and it picks straight back up when the file goes


def test_the_restart_budget_stops_the_supervisor():
    h = harness(BYSTANDERS + HELPERS, Status(1800.0, "stopped", 6_901_546), max_restarts_per_hour=1)
    code = h.sup.run(max_ticks=5)
    assert code == 1                               # exits non-zero
    assert len(h.trainer_commands) == 1            # one restart, then it gave up
    assert "GIVING UP" in h.sup.log_path.read_text(encoding="utf-8")


def test_the_grace_period_stops_a_second_trainer_being_started():
    h = harness(BYSTANDERS + HELPERS, Status(1800.0, "stopped", 6_901_546), start_grace_seconds=900.0)
    assert h.sup.tick() == "restarted"
    h.time += 60.0
    assert h.sup.tick() == "grace"                 # the games are still loading; status.json is still old
    assert len(h.trainer_commands) == 1


def test_missing_helpers_are_started_and_running_ones_are_not_duplicated():
    h = harness(BYSTANDERS + TRAINER_TREE, Status(12.0, "running", 6_901_546))
    assert h.sup.tick() == "ok"
    assert sorted(Path(c.split(" -u ")[1].split()[0]).name for c in h.spawned) == \
        ["keep_best.py", "poll_status.py"]
    h2 = harness(BYSTANDERS + TRAINER_TREE + HELPERS, Status(12.0, "running", 6_901_546))
    assert h2.sup.tick() == "ok" and h2.spawned == []


def test_a_dry_run_changes_nothing():
    h = harness(BYSTANDERS, Status(1800.0, "stopped", 6_901_546), dry_run=True)
    assert h.sup.tick() == "would_restart"
    assert (h.killed, h.spawned, h.launched, h.stopped) == ([], [], [], 0)


def test_a_finished_run_is_not_restarted():
    h = harness(BYSTANDERS + HELPERS, Status(30.0, "finished", 20_000_000))
    assert h.sup.tick() == "finished"
    assert h.sup.run(max_ticks=1) == 0 and h.trainer_commands == []


def test_logging_survives_the_shell_holding_the_log_file():
    # Measured on the live run: `cmd /c ... >> runs\<run>_supervisor.log` holds that file exclusively, so the
    # supervisor's own open(..., "a") raises PermissionError for its whole life. If the file were the only
    # path, every line would silently vanish -- which is exactly what happened on the first start. stdout is
    # redirected into the same file by the shell, so it has to carry the line too. A directory in the log's
    # place reproduces the failing open without needing the lock.
    h = harness(BYSTANDERS + TRAINER_TREE + HELPERS, Status(12.0, "running", 6_901_546))
    h.sup.log_path.parent.mkdir(parents=True, exist_ok=True)
    h.sup.log_path.mkdir()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        h.sup.log("the trainer is fine")
    assert "the trainer is fine" in buf.getvalue()


# ---------------------------------------------------------------------------
# The boot health gate
# ---------------------------------------------------------------------------


def test_a_game_counts_as_booted_only_after_two_polls_over_the_line():
    # A copy that is still loading crosses the threshold while it climbs, so one sample over the line proves
    # nothing. The stuck instance measured on 2026-09-17 sat at 56 MB while the others sat near 1 GB.
    gate = supervise.BootGate(600 * supervise.MB, consecutive=2)
    gate.observe({1: 900 * supervise.MB, 2: 56 * supervise.MB})
    assert gate.booted() == set() and gate.lagging() == {1, 2}
    gate.observe({1: 950 * supervise.MB, 2: 56 * supervise.MB})
    assert gate.booted() == {1} and gate.lagging() == {2}
    assert gate.ready(1) and not gate.ready(2)


def test_a_game_that_falls_back_under_the_line_loses_its_streak():
    gate = supervise.BootGate(600 * supervise.MB, consecutive=2)
    gate.observe({1: 900 * supervise.MB})
    gate.observe({1: 100 * supervise.MB})
    assert gate.booted() == set()
    gate.observe({1: 900 * supervise.MB})
    assert gate.booted() == set(), "the streak starts again, it does not resume"


def test_a_vanished_game_is_forgotten_rather_than_counted_forever():
    gate = supervise.BootGate(600 * supervise.MB, consecutive=1)
    gate.observe({1: 900 * supervise.MB, 2: 900 * supervise.MB})
    assert gate.ready(2)
    gate.observe({1: 900 * supervise.MB})  # pid 2 exited
    assert gate.booted() == {1} and not gate.ready(2)


def test_a_laggard_is_named_by_its_own_port():
    ports = {47800: 100, 47801: 200, 47802: 300}
    assert supervise.laggard_ports(ports, {200}, ports) == [47801]
    # A laggard that is not listening at all has no port to restart it by: games.relaunch_one identifies an
    # instance through the pid on its port, so there is nothing to address and it is only waited out.
    assert supervise.laggard_ports(ports, {999}, ports) == []


def test_the_trainer_does_not_start_until_every_game_has_booted():
    h = harness(BYSTANDERS, Status(1800.0, "stopped", 6_901_546))
    assert h.sup.tick() == "restarted"
    assert h.polls >= 2, "the gate needs two consecutive polls, so it cannot pass on one sample"
    assert h.trainer_commands, "a healthy set of games still starts the trainer"


def test_a_single_laggard_is_restarted_on_its_own_port_leaving_the_others_alone():
    h = harness(BYSTANDERS, Status(1800.0, "stopped", 6_901_546),
                boot_timeout_seconds=60.0, boot_poll_seconds=15.0)
    h.sets[h.ports[47808]] = 56 * supervise.MB  # the half-booted copy from the live recovery
    assert h.sup.tick() == "restarted"
    assert h.relaunched == [47808], "only the laggard is restarted"
    assert h.stopped == 1, "and stop_all ran once for the restart itself, not again for the laggard"
    assert h.trainer_commands


def test_a_game_that_never_boots_does_not_block_training_forever():
    # Starting anyway is the right call now: the env waits out `unknown scene` for minutes instead of dying on
    # it, so a cold game stalls one worker rather than killing the run. Refusing to train would be worse.
    h = harness(BYSTANDERS, Status(1800.0, "stopped", 6_901_546),
                boot_timeout_seconds=60.0, boot_poll_seconds=15.0, boot_relaunch_rounds=0)
    h.sets[h.ports[47803]] = 56 * supervise.MB
    assert h.sup.tick() == "restarted"
    assert h.relaunched == [], "rounds 0 means wait only"
    assert h.trainer_commands, "the trainer still starts"


def test_await_boot_reports_whether_it_succeeded():
    h = harness(BYSTANDERS, Status(30.0, "running", 6_901_546))
    assert h.sup.await_boot() is True
    h.sets[h.ports[47805]] = 10 * supervise.MB
    h.sup.cfg.boot_timeout_seconds = 30.0
    h.sup.cfg.boot_relaunch_rounds = 0
    assert h.sup.await_boot() is False


def test_the_command_line_matches_the_pattern_a_human_would_type():
    cmd = supervise.shell_command(r"F:\python\.venv\Scripts\python.exe", "scripts/train.py",
                                  ["--config", CONFIG], r"runs\campaign_gates_train.log")
    assert cmd == ('cmd.exe /s /c ""F:\\python\\.venv\\Scripts\\python.exe" -u scripts/train.py '
                   '--config configs/campaign_gates_main.yaml >> runs\\campaign_gates_train.log 2>&1"')


def test_the_supervisor_launches_its_games_hidden_from_steam_unless_steam_is_asked_for():
    """The 12 games come up through `_default_launch_games`, which the other tests replace wholesale. This is
    the only place the real one is exercised, and it is the whole of "training is hidden by default": the
    supervisor is what relaunches the games after every restart."""
    import games  # noqa: PLC0415 - the supervisor imports it the same way, inside the function

    seen: list = []
    original = games.launch
    games.launch = lambda *a, **kw: seen.append(kw.get("no_steam"))
    try:
        Supervisor(Config(run=RUN, config=CONFIG))._default_launch_games(12, 1)
        Supervisor(Config(run=RUN, config=CONFIG, no_steam=False))._default_launch_games(12, 1)
    finally:
        games.launch = original
    assert seen == [True, False], seen


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
