"""The 2026-09-17 17:52 freeze, and every bound that now stops it:  python tests/test_freeze_recovery.py

The incident: the `campaign_gates` run stopped advancing at 12,441,598 steps with `state: "running"`, and the
supervisor did not act for 629 s, then tore down and relaunched all twelve games -- about 19 minutes of a
twelve-game machine, for the sixth time that day. What was measurable during the freeze: one game (port 47800)
Not Responding with a 0.00-core CPU delta and the ONLY established connection of the twelve; the other eleven
Responding, burning ~1 core each, with no connection at all.

Nothing here needs a game, a socket, a process or a real clock. Every test pins one of the findings:

  - `close()` promised 5 s and delivered `self.timeout` (120 s live), because `recv()` re-armed the socket.
    Measured against a suspended game on 2026-09-17: 20.0 s against a client whose `timeout` was 20.
  - the `hello` handshake and `config` had no bound of their own either.
  - a socket timeout is STICKY, so a `reset`'s bound stayed on the socket for the next `send`.
  - `_resilient_reset` had per-attempt bounds but no TOTAL, so three attempts composed to ~70 minutes against
    a supervisor that gives up at 600 s -- every local recovery was killed mid-flight.
  - `SubprocVecEnv.close()` blocks forever on a wedged worker, so a crashed trainer became a hung one.
  - the eleven idle games lost their clients because the mod drops a client that sends nothing for 300 s
    (`EpisodeController.commandTimeoutMs`), which was SHORTER than the client's own 600 s reset timeout.
  - and none of it could be attributed, because the supervisor spawns the trainer with `DETACHED_PROCESS`,
    which silently discards a shell redirect's output: `campaign_gates_train.log` had not been written to in
    seven hours and the supervisor was printing its stale tail as evidence on every restart.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import games  # noqa: E402
import supervise  # noqa: E402
import train  # noqa: E402
import ultrakill_ai.env as env_mod  # noqa: E402
from test_bridge_recovery import FAST, wired  # noqa: E402
from test_campaign_env import forward, make_env  # noqa: E402
from ultrakill_ai.env import EnvConfig  # noqa: E402
from ultrakill_ai.envlog import EnvLog, env_log_path, tail  # noqa: E402
from ultrakill_ai.protocol import (  # noqa: E402
    DEFAULT_CLOSE_TIMEOUT,
    DEFAULT_HANDSHAKE_TIMEOUT,
    DEFAULT_RESET_TIMEOUT,
    DEFAULT_STEP_TIMEOUT,
    RECOVERABLE,
    BridgeClient,
    BridgeClosed,
    BridgeError,
    BridgeTimeout,
    _LineReader,
)

MOD_RESET_TIMEOUT_S = 120.0  # EpisodeController.resetTimeoutSeconds
MOD_COMMAND_TIMEOUT_S = 300.0  # EpisodeController.commandTimeoutMs, after which the mod DROPS the client


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# 1. Every blocking socket call is bounded, in every state
# ---------------------------------------------------------------------------


def test_close_is_bounded_by_the_close_timeout_and_not_by_the_step_timeout():
    # The regression, exactly as measured live: close() sets settimeout(5.0) and then calls request(), whose
    # recv() armed `self.timeout` straight back over it. Against a frozen game that is 120 s per env; twelve of
    # those is a 24-minute teardown while the supervisor is already counting.
    client, sock = wired(['{"type":"ok"}\n'], timeout=120.0)
    client.close()
    assert sock.timeouts, "close() must arm a bound before it waits for the release reply"
    assert set(sock.timeouts) == {DEFAULT_CLOSE_TIMEOUT}, sock.timeouts
    assert 120.0 not in sock.timeouts
    assert sock.closed


def test_a_close_whose_release_times_out_still_drops_the_socket_and_does_not_raise():
    client, sock = wired([TimeoutError("timed out")], timeout=120.0)
    client.close()  # must not raise: a teardown may not fail over a game that has stopped answering
    assert sock.closed
    assert client._sock is None


def test_a_broken_client_skips_the_release_entirely():
    client, sock = wired([TimeoutError("timed out")], timeout=5.0)
    try:
        client.step({})
    except BridgeTimeout:
        pass
    sock.timeouts.clear()
    sock.sent.clear()
    client.close()
    assert sock.sent == [], "there is nothing to hand control back to on a poisoned stream"
    assert sock.timeouts == [], "and nothing to wait for"


def test_the_handshake_has_a_bound_of_its_own():
    # `connect()`'s 5 s only ever covered the TCP connect. The hello that follows it was bounded by
    # `self.timeout`, so a game that accepts the socket and then answers nothing -- the state a recovery is
    # trying to escape -- held the reconnect for two minutes.
    client, sock = wired(['{"type":"config-ok"}\n'], timeout=120.0)
    client.configure(frameskip=2)
    assert set(sock.timeouts) == {DEFAULT_HANDSHAKE_TIMEOUT}
    assert DEFAULT_HANDSHAKE_TIMEOUT < DEFAULT_STEP_TIMEOUT


def test_a_reset_does_not_leave_its_long_bound_on_the_socket():
    # A socket timeout is sticky. `reset()` armed 600 s for its read and left it there, so the NEXT `send`
    # inherited 600 s -- a "120 s" step whose send could block for ten minutes.
    client, sock = wired(['{"type":"obs"}\n', '{"type":"obs"}\n'])
    client.reset("Level 0-1")
    sock.timeouts.clear()
    client.step({})
    assert set(sock.timeouts) == {DEFAULT_STEP_TIMEOUT}
    assert DEFAULT_RESET_TIMEOUT not in sock.timeouts


def test_the_line_reader_bounds_wall_clock_and_not_just_each_recv():
    # `socket.makefile().readline()` restarts the socket timeout on every underlying recv, so a peer dribbling
    # one byte just inside the bound keeps a "bounded" read alive indefinitely.
    class Dribble:
        def __init__(self):
            self.clock = [0.0]
            self.armed: list[float] = []

        def settimeout(self, value):
            self.armed.append(value)

        def recv(self, _n):
            self.clock[0] += 1.0  # each byte costs a second of wall clock
            return b"x"

    sock = Dribble()
    reader = _LineReader(sock)
    import ultrakill_ai.protocol as protocol

    real = protocol.time.monotonic
    protocol.time.monotonic = lambda: sock.clock[0]
    try:
        try:
            reader.readline(5.0)
            raise AssertionError("a dribbling peer must not outlast the bound")
        except TimeoutError:
            pass
    finally:
        protocol.time.monotonic = real
    assert sock.clock[0] <= 6.0, "it gave up at the deadline, not after an unbounded number of reads"
    assert max(sock.armed) <= 5.0, "each read is armed with what is LEFT of the deadline"


def test_a_reset_can_be_clamped_below_its_own_timeout():
    """The clamp the recovery ladder needs: a reset asked for LESS than `reset_timeout` must honour it.

    Without it the deadline was advisory. An attempt admitted with a moment of budget left still ran its own
    full 180 s bound on top -- twice, because `request` arms the bound for the send AND the read -- so a
    "540 s total" recovery measured 555.6 s in the best case and 780 s in the worst.
    """
    client, sock = wired([""], timeout=DEFAULT_STEP_TIMEOUT)
    try:
        client.reset("Level 0-1", timeout=7.0)
    except BridgeError:
        pass  # the empty line is a hang-up; the bound is what is under test
    assert sock.timeouts and max(sock.timeouts) == 7.0, (
        "the clamped bound is what reaches the socket, for the send as well as the read: %r" % (sock.timeouts,))


def test_against_a_real_silent_socket_every_bound_is_honoured():
    """The live shape, on a real loopback socket: a peer that accepts the connection and then says nothing.

    This is what game 47809 looked like at 19:13 on 2026-09-17 -- Not Responding, a 0.00-core CPU delta, and
    the only ESTABLISHED connection of the twelve -- with the client blocked in `recv` against it. Nothing
    here needs a game; it needs a socket that never answers, which is the same thing from the client's side.
    """
    import socket as _socket
    import threading

    server = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: list = []
    stop = threading.Event()

    def serve():
        server.settimeout(10.0)
        try:
            conn, _ = server.accept()
            accepted.append(conn)  # accepted, then dead silence: never a single byte back
            stop.wait(30.0)
        except OSError:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = BridgeClient("127.0.0.1", port, timeout=120.0, reset_timeout=180.0,
                          handshake_timeout=2.0, close_timeout=1.0)
    try:
        # 1. The handshake has a bound of its own and does not inherit `timeout` (120 s).
        started = time.monotonic()
        try:
            client.connect(retry_seconds=2.0)
            raise AssertionError("a silent peer cannot complete a handshake")
        except BridgeTimeout:
            pass
        handshake = time.monotonic() - started
        assert handshake < 10.0, "the hello waited %.1fs; it must use handshake_timeout, not timeout" % handshake

        # 2. A reset clamped by a recovery budget honours the clamp, not its own 180 s.
        client.broken = False
        started = time.monotonic()
        try:
            client.reset("Level 0-1", timeout=2.0)
            raise AssertionError("a silent peer cannot answer a reset")
        except BridgeTimeout:
            pass
        clamped = time.monotonic() - started
        assert clamped < 10.0, "the clamped reset waited %.1fs instead of ~2s" % clamped

        # 3. close() never waits on a sick game: it is bounded by close_timeout, not by `timeout`.
        client.broken = False
        started = time.monotonic()
        client.close()
        closed = time.monotonic() - started
        assert closed < 10.0, "close() waited %.1fs; twelve of those is a teardown of minutes" % closed
    finally:
        stop.set()
        client._drop()
        for conn in accepted:
            conn.close()
        server.close()
        thread.join(timeout=5.0)


def test_the_reset_bound_sits_between_the_mods_own_and_the_supervisors():
    # It has to outlast the mod's own give-up (or the mod's graceful "reset timed out" error reply is dead
    # code again) and fit inside one supervisor window with a whole recovery ladder around it. 600 s did not:
    # a single unanswered reset was already the supervisor's entire patience.
    assert DEFAULT_RESET_TIMEOUT > MOD_RESET_TIMEOUT_S
    assert DEFAULT_RESET_TIMEOUT < supervise.Config().stale_seconds / 2


def test_the_mod_is_told_to_outlast_a_whole_recovery_before_dropping_a_silent_client():
    # Why the freeze left ELEVEN games running free with no connection: they were not broken, they were idle
    # behind one blocked worker (vectorized envs step in lockstep), and the mod drops a client it has heard
    # nothing from for `commandTimeoutMs`. At 300 s that is shorter than one worker's recovery, so one sick
    # game cost twelve episodes. The key is a plain config value the shipped mod already reads -- verified
    # accepted by the installed v0.7.0 on 2026-09-17 -- so this needs no rebuild.
    cfg = EnvConfig()
    assert cfg.mod_command_timeout_s > cfg.bridge_recovery_budget_s + cfg.reset_timeout_s
    assert cfg.mod_command_timeout_s > MOD_COMMAND_TIMEOUT_S, "the mod's own default is what has to be raised"
    env, fake = make_env(**FAST)
    env.reset()
    assert fake.settings["command_timeout_s"] == cfg.mod_command_timeout_s
    assert isinstance(fake.settings["command_timeout_s"], int), "ApplyConfig reads it as Value<int>()"


def test_a_reset_never_goes_silent_for_longer_than_the_mod_tolerates():
    # The mod drops a client that sends nothing for 300 s and releases control. With the old 600 s reset bound
    # one worker's long reset guaranteed the mod dropped it -- and the other eleven games, idle behind the same
    # lockstep, were dropped too. That is the eleven-free-running-games state the freeze was observed in.
    assert DEFAULT_RESET_TIMEOUT < MOD_COMMAND_TIMEOUT_S
    assert DEFAULT_STEP_TIMEOUT < MOD_COMMAND_TIMEOUT_S


# ---------------------------------------------------------------------------
# 2. The recovery is bounded in TOTAL, and ends in a relaunch of its own game
# ---------------------------------------------------------------------------


def dead_env(**overrides):
    """An env whose every reset fails, with no sleeping and a budget measured in milliseconds."""
    env, fake = make_env(bridge_backoff_s=0.0, unknown_scene_wait_s=0.0, bridge_retries=3,
                         bridge_recovery_budget_s=0.2, **overrides)
    fake.fail_next_resets(BridgeTimeout("no reply"), times=500)
    return env, fake


def test_a_recovery_is_bounded_by_one_total_budget():
    env, _ = dead_env()
    started = time.monotonic()
    try:
        env.reset()
        raise AssertionError("expected the budget to run out")
    except BridgeError as exc:
        assert "did not come back within" in str(exc)
    assert time.monotonic() - started < 5.0, "the whole ladder ran inside its budget"


def test_the_budget_covers_the_outer_retry_too_and_is_not_paid_twice():
    # `reset()` catches what the inner ladder raises and calls it again. Without a shared deadline that is two
    # full budgets, which is how per-attempt bounds composed into a ~70-minute reset.
    env, _ = dead_env()
    opened: list[str] = []
    real = env._begin_recovery

    def spy(reason):
        opened.append(reason)
        return real(reason)

    env._begin_recovery = spy
    try:
        env.reset()
    except BridgeError:
        pass
    assert opened, "a recovery opened a budget"
    assert env._recovery_deadline is None, "and closed it on the way out"


def test_the_relaunch_rung_restarts_this_envs_own_port_and_no_other():
    env, fake = make_env(bridge_backoff_s=0.0, unknown_scene_wait_s=0.0, bridge_retries=2,
                         bridge_recovery_budget_s=60.0, bridge_relaunch=True,
                         bridge_relaunch_reserve_s=20.0)  # a reserve the small test budget can actually hold
    fake.fail_next_resets(BridgeTimeout("no reply"), times=2)  # rung 0 spent, then the relaunch fixes it
    relaunched: list[int] = []

    def fake_relaunch(port, wait_s):
        relaunched.append(port)
        return True

    env._relaunch_hook = fake_relaunch
    env.reset()
    assert relaunched == [env.cfg.port], "a worker may restart its own game and nothing else"
    assert env._relaunches == 1


def test_a_relaunch_is_not_attempted_when_the_budget_cannot_boot_a_game():
    env, fake = make_env(bridge_backoff_s=0.0, unknown_scene_wait_s=0.0, bridge_retries=1,
                         bridge_recovery_budget_s=1.0, bridge_relaunch=True)
    fake.fail_next_resets(BridgeTimeout("no reply"), times=50)
    env._relaunch_hook = lambda port, wait_s: (_ for _ in ()).throw(AssertionError("must not relaunch"))
    try:
        env.reset()
        raise AssertionError("expected a bounded failure")
    except BridgeError:
        pass
    assert env._relaunches == 0


def test_only_train_py_may_ever_turn_the_relaunch_on():
    # eval.py, bridge_test.py, campaign_check.py and every test build an EnvConfig of their own. None of them
    # may be able to kill a game process, so the default is off and exactly one caller turns it on.
    assert EnvConfig().bridge_relaunch is False
    filled = train.fill_run_dirs(EnvConfig(), Path("runs") / "campaign_gates")
    assert filled.bridge_relaunch is True
    assert filled.env_log_dir.endswith("runs/campaign_gates")


def test_a_recovery_that_fails_inside_a_step_still_ends_the_episode_rather_than_the_run():
    env, fake = make_env(**FAST)
    env.reset()
    env.step(forward())
    fake.fail_next_steps(BridgeTimeout("dead"))
    fake.fail_next_resets(BridgeTimeout("dead"), times=50)
    try:
        env.step(forward())
        raise AssertionError("a bridge that never comes back must finally raise")
    except BridgeError as exc:
        assert "port" in str(exc)


# ---------------------------------------------------------------------------
# 3. Attribution: every worker writes its own bounded event log
# ---------------------------------------------------------------------------


def test_the_env_log_records_the_reset_and_the_recovery_against_its_port():
    directory = tmpdir()
    env, fake = make_env(env_log_dir=str(directory), **FAST)
    env.reset()
    fake.fail_next_steps(BridgeTimeout("dead"))
    env.step(forward())
    lines = (directory / "env_47800.log").read_text(encoding="utf-8").splitlines()
    kinds = [line.split()[3] for line in lines]  # <date> <time> port=<n> <kind> ...
    assert "reset_start" in kinds and "reset_end" in kinds
    assert "bridge_reset" in kinds, "the line that names what broke is the whole point of the file"
    assert "recover_start" in kinds and "recover_end" in kinds
    assert all("port=47800" in line for line in lines)


def test_a_reset_that_works_first_time_is_not_logged_as_a_recovery():
    # Twelve workers reset every few minutes all night; a log that called each of those a recovery would bury
    # the one line that matters under thousands that do not.
    directory = tmpdir()
    env, _ = make_env(env_log_dir=str(directory), **FAST)
    env.reset()
    kinds = [line.split()[3] for line in (directory / "env_47800.log").read_text(encoding="utf-8").splitlines()]
    assert kinds == ["reset_start", "reset_end"]


def test_the_env_log_is_size_bounded():
    directory = tmpdir()
    log = EnvLog(directory / "env_47800.log", 47800, max_bytes=200)
    for _ in range(200):
        log.event("reset_start", level="Level 0-1")
    assert (directory / "env_47800.log").stat().st_size <= 400
    assert (directory / "env_47800.log.1").exists(), "the older half is rolled, not thrown away"


def test_an_env_with_no_log_directory_writes_nothing_and_still_runs():
    assert env_log_path("", 47800) is None
    log = EnvLog(None, 47800)
    # A value with a space is quoted, so one line is still one parseable record.
    assert log.event("reset_start", level="Level 0-1").endswith('reset_start level="Level 0-1"')
    env, _ = make_env(**FAST)  # env_log_dir defaults to "", which is eval.py and every test
    env.reset()
    env.step(forward())


def test_the_log_line_carries_a_timestamp_a_port_and_readable_fields():
    log = EnvLog(None, 47803, clock=lambda: 1_700_000_000.0)
    line = log.event("recover_end", outcome="ok", rung=1, dt_s=12.5)
    assert " port=47803 recover_end " in line
    assert "outcome=ok" in line and "rung=1" in line and "dt_s=12.50" in line
    assert line[:4].isdigit(), "it starts with a wall-clock stamp, not a monotonic one"


def test_tail_reads_the_end_of_a_log_and_survives_a_missing_file():
    directory = tmpdir()
    log = EnvLog(directory / "env_47800.log", 47800)
    for i in range(50):
        log.event("reset_start", level="Level 0-%d" % i)
    assert len(tail(directory / "env_47800.log", 4)) == 4
    assert "Level 0-49" in tail(directory / "env_47800.log", 1)[0]
    assert tail(directory / "nope.log") == []


# ---------------------------------------------------------------------------
# 4. A dead worker exits the trainer promptly instead of hanging it
# ---------------------------------------------------------------------------


class FakeProcess:
    """A worker that may refuse to join, so the teardown has to terminate it."""

    def __init__(self, wedged: bool = False):
        self.wedged = wedged
        self.alive = True
        self.joins: list[float] = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.joins.append(timeout)
        if not self.wedged:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


class FakeRemote:
    def __init__(self, broken: bool = False):
        self.sent: list[object] = []
        self.broken = broken
        self.recvs = 0

    def send(self, payload):
        if self.broken:
            raise BrokenPipeError("[WinError 109] The pipe has been ended")
        self.sent.append(payload)

    def recv(self):
        self.recvs += 1
        raise AssertionError("a teardown must never block reading from a worker that may be wedged")


class FakeVecEnv:
    def __init__(self, processes, remotes, waiting=True):
        self.processes = processes
        self.remotes = remotes
        self.waiting = waiting
        self.closed = False


def test_a_wedged_worker_is_terminated_so_the_trainer_exits_promptly():
    # SubprocVecEnv.close() does `process.join()` with no timeout, so ONE worker stuck in a bridge call keeps
    # the trainer alive for ever. The supervisor then sees a HUNG trainer at the stale-window mark instead of a
    # DEAD one at the next poll, and pays a full twelve-game restart for it.
    procs = [FakeProcess(), FakeProcess(wedged=True), FakeProcess()]
    venv = FakeVecEnv(procs, [FakeRemote() for _ in procs])
    assert train.close_vec_env(venv, timeout=0.0) == "terminated"
    assert procs[1].terminated, "the one that would not join was terminated"
    assert not any(p.alive for p in procs)
    assert venv.closed


def test_a_healthy_teardown_is_still_a_clean_one():
    procs = [FakeProcess(), FakeProcess()]
    venv = FakeVecEnv(procs, [FakeRemote() for _ in procs])
    assert train.close_vec_env(venv, timeout=5.0) == "clean"
    assert not any(p.terminated for p in procs)


def test_a_teardown_never_waits_for_a_reply_and_survives_a_dead_pipe():
    procs = [FakeProcess(), FakeProcess()]
    remotes = [FakeRemote(broken=True), FakeRemote()]
    venv = FakeVecEnv(procs, remotes, waiting=True)  # waiting=True is what makes SB3's close() call recv()
    train.close_vec_env(venv, timeout=1.0)
    assert all(r.recvs == 0 for r in remotes)
    assert venv.waiting is False


# ---------------------------------------------------------------------------
# 5. The supervisor: steps, not mtime; and a report that names the sick port
# ---------------------------------------------------------------------------


def supervisor_harness(**kw):
    from test_supervise import BYSTANDERS, TRAINER_TREE, harness

    return harness(BYSTANDERS + TRAINER_TREE, supervise.Status(1.0, "running", 12_441_598.0), **kw)


def test_a_frozen_run_with_a_perfectly_fresh_status_file_is_caught_by_the_step_count():
    # NOT what happened on 2026-09-17: there the writes stopped with the steps and mtime went stale on its own.
    # This is the case mtime cannot see -- a status file refreshed off the step loop, which already happens in
    # `_on_rollout_start` after a step-free PPO update and would happen for every poll if a heartbeat thread is
    # ever added. `timesteps` is the thing actually being asked about; mtime is a proxy for it.
    h = supervisor_harness(stale_seconds=600.0)
    assert h.sup.tick() == "ok"
    h.time += 300.0
    assert h.sup.tick() == "ok", "still inside the window"
    h.time += 400.0  # 700 s with the same step count and an always-1s-old file
    assert h.sup.tick() == "restarted"
    assert any("steps have not moved" in line for line in h.sup.log_path.read_text(encoding="utf-8").splitlines())


def test_a_step_count_that_moves_keeps_the_run_healthy_however_slowly():
    h = supervisor_harness(stale_seconds=600.0)
    for i in range(6):
        h.status = supervise.Status(1.0, "running", 12_441_598.0 + i)
        h.time += 500.0
        assert h.sup.tick() == "ok"


def test_a_resume_from_an_older_checkpoint_counts_as_progress_not_as_a_stall():
    # A restart resumes from a checkpoint, so `timesteps` goes BACKWARDS. A watermark that only accepted an
    # increase would call every restart a stall and restart it again.
    h = supervisor_harness(stale_seconds=600.0)
    h.sup.tick()
    h.time += 700.0
    h.status = supervise.Status(1.0, "running", 12_426_514.0)  # the checkpoint it resumed from
    assert h.sup.tick() == "ok"


def test_a_draining_trainer_is_not_judged_on_its_step_count():
    # A Ctrl+C in progress is SUPPOSED to have stopped stepping while it writes latest.zip.
    h = supervisor_harness(stale_seconds=600.0)
    h.sup.tick()
    h.time += 700.0
    h.status = supervise.Status(1.0, "stopped", 12_441_598.0)
    assert h.sup.tick() == "draining"


def test_the_stale_window_comfortably_exceeds_a_workers_whole_recovery_budget():
    """The arithmetic, pinned -- but derived from the MEASURED ladder, not from a sum of the knobs.

    The old version of this test added `budget + reset_timeout + poll` and called the result the worst case.
    That sum was wrong in both directions at once: the ladder overshot its budget by a whole in-flight call
    (so the true worst case was larger), and it counted a reset timeout that a clamped ladder never pays on
    top (so the sum was the wrong shape). `test_the_measured_ladder_fits_inside_the_supervisors_patience`
    below is the one that actually runs the ladder; this keeps the headline numbers honest.

    Worst case for a worker's silence: the step that faults pays its own bound (120 s), then the recovery
    budget (540 s) covers everything after it -- reconnects, resets, the relaunch, the lot. 660 s. The
    supervisor polls every 60 s, so it can notice up to one poll late: 720 s against a 900 s window.
    """
    cfg, env_cfg = supervise.Config(), EnvConfig()
    worst = env_cfg.step_timeout_s + env_cfg.bridge_recovery_budget_s + cfg.poll_seconds
    assert worst == 720.0, "the documented arithmetic changed; re-derive it in CLAUDE.md too"
    assert cfg.stale_seconds > worst, "the supervisor must outlast a whole local recovery"
    # And the mod must outlast it as well, or the eleven idle siblings are dropped while one worker recovers,
    # turning one sick game into twelve recoveries. That is what happened at 17:52 and again at 19:13.
    assert env_cfg.mod_command_timeout_s > env_cfg.step_timeout_s + env_cfg.bridge_recovery_budget_s


def _fake_clock_ladder(*, per_reset_s: float, port_down: bool = False, **overrides):
    """Runs a real `reset()` on a fake clock with every blocking call charged its REAL bound.

    This is the test the old suite lacked. Its fakes failed instantly, so three attempts cost ~0 s, the budget
    was never spent and the relaunch rung was always reached -- while production, where each attempt costs a
    full reset timeout, never reached it once. Returns `(elapsed_seconds, relaunched, outcome)`.
    """
    class Clock:
        def __init__(self):
            self.t = 1000.0

        monotonic = perf_counter = lambda self: self.t  # noqa: E731

        def sleep(self, s):
            self.t += max(0.0, s)

        def charge(self, s):
            self.t += max(0.0, s)

    clock = Clock()
    # A TRAINING worker's settings: the shipped defaults, with the relaunch rung on as `train.py` turns it on.
    settings = dict(bridge_recovery_budget_s=540.0, reset_timeout_s=180.0, connect_retry_s=180.0,
                    bridge_retries=3, bridge_backoff_s=5.0, unknown_scene_wait_s=0.0, bridge_relaunch=True)
    settings.update(overrides)
    env, fake = make_env(**settings)
    relaunched: list[int] = []

    def relaunch_hook(port, wait_s):
        relaunched.append(port)
        clock.charge(min(wait_s, 90.0))  # a cold Unity start
        return True

    def connect(retry_seconds=180.0):
        if port_down:  # connect() retries for its whole (clamped) window, then gives up
            clock.charge(retry_seconds)
            raise BridgeClosed("Could not connect to the mod on 127.0.0.1:47800.")
        clock.charge(0.2)
        return {"type": "hello", "protocol": 1, "scene": "Level 0-1"}

    def reset(scene=None, checkpoint=False, timeout=None):
        clock.charge(per_reset_s if timeout is None else min(per_reset_s, timeout))
        raise BridgeTimeout("no reply from 127.0.0.1:47800")

    env._relaunch_hook = relaunch_hook
    fake.connect, fake.reset, fake.close = connect, reset, lambda: None
    env_mod.time = clock
    started = clock.t
    outcome = "recovered"
    try:
        env.reset()
    except BaseException as exc:  # noqa: BLE001 - the outcome is the measurement
        outcome = type(exc).__name__
    finally:
        env_mod.time = time
    return clock.t - started, bool(relaunched), outcome


def test_the_measured_ladder_fits_inside_the_supervisors_patience():
    """Every fault shape, charged its real bounds, must end inside `stale_seconds`.

    Measured before the fix, with the shipped defaults: a game that answered `hello` and never answered a
    reset ran 555.6 s and NEVER reached the relaunch; the reset tail opened a second full budget on top
    (two budgets, 1000->1540 and 1183->1723, for one fault). Both are why a local recovery presented to the
    supervisor as a hung trainer and was paid for as a twelve-game restart.
    """
    window = supervise.Config().stale_seconds
    for label, kwargs in (("a game that never answers a reset", dict(per_reset_s=180.0)),
                          ("a port that never opens", dict(per_reset_s=180.0, port_down=True)),
                          ("with the relaunch rung off", dict(per_reset_s=180.0, bridge_relaunch=False))):
        elapsed, _, _ = _fake_clock_ladder(**kwargs)
        assert elapsed <= EnvConfig().bridge_recovery_budget_s + 1.0, (
            "%s: the ladder ran %.1fs against a %.0fs budget" % (label, elapsed, EnvConfig().bridge_recovery_budget_s))
        assert elapsed < window, "%s: %.1fs outlasts the supervisor's %.0fs" % (label, elapsed, window)


def test_the_relaunch_rung_is_reached_when_the_bridge_fails_SLOWLY():
    """The fault shape the rung exists for, which the instant-failure fake never exercised.

    Rung 0's attempts each cost a full reset timeout against a wedged game, so without a reserve the budget
    was always negative by the time the rung was tested and the loop broke instead. Measured before the fix:
    relaunch called on the fast-failure shape, NOT called on the slow one -- the only covered path was the
    one that does not occur.
    """
    _, slow_relaunched, _ = _fake_clock_ladder(per_reset_s=180.0)
    assert slow_relaunched, "a game that answers nothing slowly must still get its instance restarted"
    _, down_relaunched, _ = _fake_clock_ladder(per_reset_s=180.0, port_down=True)
    assert down_relaunched, "a port that never opens is exactly what the relaunch rung is for"
    _, fast_relaunched, _ = _fake_clock_ladder(per_reset_s=0.0)
    assert fast_relaunched, "and the fast-failure shape still reaches it"


def test_a_cold_port_at_startup_does_not_kill_the_worker():
    """`supervise.await_boot` deliberately starts the trainer against a copy that has not finished booting.

    A game that has not opened its PORT yet fails the FIRST `_ensure_connected`, above the ladder, and that
    used to raise a bare `BridgeError` -- which is not in RECOVERABLE, so `reset()` did not catch it and the
    worker died. Three rounds of that burns `--max-restarts-per-hour` and the supervisor gives up for the
    night: the 18:08 pathology.
    """
    from ultrakill_ai.protocol import BridgeClient
    assert issubclass(BridgeClosed, BridgeError)
    assert BridgeClosed in RECOVERABLE, "a port that is not listening must be recoverable, not fatal"
    client = BridgeClient(port=59999, connect_timeout=0.05)  # nothing listens here
    try:
        client.connect(retry_seconds=0.0)
        raise AssertionError("expected the connect to fail")
    except BridgeClosed:
        pass  # RECOVERABLE, so reset()'s handler catches it and the ladder waits the game out

    # And end to end: the env survives it and reaches the relaunch rung rather than raising out of reset().
    elapsed, relaunched, _ = _fake_clock_ladder(per_reset_s=180.0, port_down=True)
    assert relaunched and elapsed <= EnvConfig().bridge_recovery_budget_s + 1.0


def test_the_first_connect_is_patient_enough_for_the_boot_gate_to_give_up_on():
    # `await_boot` logs "giving up waiting; starting the trainer anyway" on the stated grounds that the env
    # waits a cold game out. The connect retry window is what backs that claim for a port that is not up yet.
    assert EnvConfig().connect_retry_s >= 180.0, "a cold Unity instance takes minutes to open its port"


def test_run_only_settings_never_travel_in_the_config_file():
    """`train.py` writes `models/<run>/env_config.yaml`; `eval.py` loads exactly that file.

    Verified before the fix: the round trip yielded `bridge_relaunch=True` and
    `env_log_dir='runs/campaign_gates'`, so the documented eval command ran against a live training port with
    the power to `taskkill` it on the first bridge hiccup -- and wrote into the live run's attribution log.
    """
    import yaml

    filled = train.fill_run_dirs(EnvConfig(), Path("runs") / "campaign_gates")
    assert filled.bridge_relaunch is True and filled.env_log_dir.endswith("runs/campaign_gates")
    on_disk = yaml.safe_dump(filled.to_dict())
    keys = yaml.safe_load(on_disk)
    # By key, not by substring: `bridge_relaunch_wait_s` and friends are tuning knobs and DO belong in the file.
    assert "bridge_relaunch" not in keys and "env_log_dir" not in keys
    far_side = EnvConfig.from_dict(yaml.safe_load(on_disk))
    assert far_side.bridge_relaunch is False, "an eval may never restart a game process"
    assert far_side.env_log_dir == "", "an eval may never write into a live run's log directory"


# ---------------------------------------------------------------------------
# 2b. The relaunch rung is serialized, staggered, and bounded at the OS edge
# ---------------------------------------------------------------------------


def test_only_a_few_workers_relaunch_at_once():
    """Twelve workers reach this rung on the same step, and twelve cold Unity starts boot none of them."""
    directory = str(tmpdir())
    held = [env_mod.acquire_relaunch_slot(directory, 2, 600.0, deadline=0.0) for _ in range(2)]
    assert all(h for h in held), "both permits are handed out"
    assert held[0] != held[1]
    clock = {"t": 0.0}
    denied = env_mod.acquire_relaunch_slot(directory, 2, 600.0, deadline=-1.0,
                                           now=lambda: clock["t"], sleep=lambda s: None)
    assert denied is None, "the third worker is turned away rather than piling on"
    env_mod.release_relaunch_slot(held[0])
    again = env_mod.acquire_relaunch_slot(directory, 2, 600.0, deadline=0.0)
    assert again == held[0], "a released permit comes back"


def test_a_permit_left_behind_by_a_dead_worker_is_taken_over():
    directory = str(tmpdir())
    stale = env_mod.acquire_relaunch_slot(directory, 1, 600.0, deadline=0.0)
    assert stale
    # The holder died without releasing it. Wall clock, not monotonic: it is compared against a file mtime.
    taken = env_mod.acquire_relaunch_slot(directory, 1, 600.0, deadline=0.0,
                                          wall=lambda: time.time() + 10_000.0)
    assert taken == stale, "a crash may not wedge the rung shut for the rest of the run"


def test_an_env_with_no_log_directory_never_blocks_on_a_permit():
    # eval.py, bridge_test.py and every test have no run directory to serialize against.
    assert env_mod.acquire_relaunch_slot("", 2, 600.0, deadline=0.0) is not None


def test_the_restart_report_names_the_sick_port_before_anything_is_killed():
    h = supervisor_harness()
    assert h.sup.tick() == "ok"  # the step watermark is set on the first poll
    h.time += 5000.0
    h.established = {p: (2 if p == 47800 else 0) for p in h.ports}
    (h.sup.run_dir / "env_47800.log").write_text(
        "2026-09-17 17:52:29 port=47800 reset_start level=Level 0-2\n", encoding="utf-8")
    assert h.sup.tick() == "restarted"
    log = h.sup.log_path.read_text(encoding="utf-8")
    assert "port 47800: pid 23000" in log and "conns 2" in log
    assert "port 47801:" in log and "conns 0" in log
    assert "reset_start level=Level 0-2" in log, "the worker's own last word is quoted into the report"
    assert log.index("port 47800: pid") < log.index("killing pid"), "reported before it is destroyed"


def test_established_connections_are_counted_per_port_from_netstat_alone():
    # Never by connecting: BridgeServer drops its current client when a new one connects, so a probe kicks the
    # trainer off that game. This has killed a run before.
    out = "\n".join([
        "  Proto  Local Address          Foreign Address        State           PID",
        "  TCP    127.0.0.1:47800        0.0.0.0:0              LISTENING       2588",
        "  TCP    127.0.0.1:47800        127.0.0.1:53121        ESTABLISHED     2588",
        "  TCP    127.0.0.1:53121        127.0.0.1:47800        ESTABLISHED     7777",
        "  TCP    127.0.0.1:47801        0.0.0.0:0              LISTENING       2589",
        "  TCP    127.0.0.1:443          93.184.216.34:443      ESTABLISHED     4242",
    ])
    counts = supervise.parse_established(out, [47800, 47801, 47802])
    assert counts == {47800: 2, 47801: 0, 47802: 0}


def test_the_sick_report_reads_a_lockstepped_game_apart_from_a_free_running_one():
    ports = [47800, 47801, 47802]
    owners = {47800: 100, 47801: 101}  # 47802 never came up
    sets = {100: 1397 * supervise.MB, 101: 973 * supervise.MB}
    lines = supervise.sick_report(ports, owners, sets, {47800: 2, 47801: 0},
                                  {100: 10.0, 101: 10.0}, {100: 10.0, 101: 15.0}, window=5.0)
    assert lines[0] == "port 47800: pid 100 ws 1397 MB conns 2 cores 0.00"  # in lockstep, waiting for a command
    assert lines[1] == "port 47801: pid 101 ws 973 MB conns 0 cores 1.00"   # no client, running free
    assert lines[2] == "port 47802: NOT LISTENING"


def test_the_trainer_is_not_spawned_detached_because_that_throws_its_log_away():
    # DETACHED_PROCESS (0x8) makes `cmd /s /c "... >> log 2>&1"` create the file and discard every line the
    # program prints: the grandchild starts with no console and no inherited handles, so its sys.stdout is
    # None. Measured 2026-09-17 with 0x8, 0x200, 0x08000000 and 0. campaign_gates_train.log had been dead for
    # seven hours and the supervisor quoted its stale tail as evidence on every restart.
    assert supervise.DETACHED & 0x00000008 == 0, "DETACHED_PROCESS silently blinds the trainer log"
    assert supervise.DETACHED & 0x00000200, "CREATE_NEW_PROCESS_GROUP is what keeps it clear of Ctrl+C"


# ---------------------------------------------------------------------------
# 6. One laggard port no longer fails a whole launch
# ---------------------------------------------------------------------------


def test_a_wedged_instance_with_no_port_is_found_and_a_copy_on_another_port_is_spared():
    # The live one sat at 242 MB burning a core with no bridge port while eleven healthy games waited on it.
    # A private copy on 47812 is listening, so it is not ours to touch and is left alone.
    pids = [2588, 24080, 13652]
    owners = {47801: 24080, 47812: 13652}
    assert games.unlistening_pids(pids, owners) == [2588]


def test_a_single_laggard_port_is_repaired_instead_of_failing_the_whole_launch():
    # 2026-09-17 18:03: eleven games up, 47800 stuck, `launch` exited, and the supervisor's answer to a failed
    # launch is another full stop-and-launch -- twelve torn down to fix one, three times, then it gives up.
    state = {"open": {47801, 47802}, "repairs": []}
    original = (games.listening_ports, games.running_pids, games.relaunch_one, games.kill_unlistening,
                games.stop_all, games.start_instance, games.backup_display, games.tile, games.game_dir,
                games.time.sleep, games.disk_logging_enabled)

    def repair(port, *a, **kw):
        state["repairs"].append(port)
        state["open"].add(port)
        return True

    games.listening_ports = lambda: set(state["open"])
    games.running_pids = lambda: [1, 2, 3]
    games.relaunch_one = repair
    games.kill_unlistening = lambda: []
    games.stop_all = lambda *a, **kw: None
    games.start_instance = lambda *a, **kw: 1
    games.backup_display = lambda: None
    games.tile = lambda *a, **kw: None
    games.game_dir = lambda: Path(__file__).resolve().parent
    games.disk_logging_enabled = lambda path: False
    games.time.sleep = lambda s: None
    try:
        (games.game_dir() / "ULTRAKILL.exe").write_bytes(b"")
        games.launch(3, 47800, 368, 207, 0.0, 0.0, None, 3)
    finally:
        (games.listening_ports, games.running_pids, games.relaunch_one, games.kill_unlistening,
         games.stop_all, games.start_instance, games.backup_display, games.tile, games.game_dir,
         games.time.sleep, games.disk_logging_enabled) = original
        (Path(__file__).resolve().parent / "ULTRAKILL.exe").unlink(missing_ok=True)
    assert state["repairs"] == [47800], "only the port that did not come up was restarted"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
