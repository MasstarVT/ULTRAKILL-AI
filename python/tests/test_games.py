"""Tests for scripts/games.py's instance-count guard (no game needed).

The five-copy limit was never the game's: BepInEx's DiskLogListener opens `LogOutput.log` plus `.1`-`.4`, so a
sixth copy fails to load. `launch --count N > 5` therefore refuses unless `[Logging.Disk] Enabled = false` is set
in BepInEx.cfg, and reading that flag has to look at the right section -- `[Logging.Console]` has an `Enabled`
key of its own a few lines above it, and it is `false` by default, which would read as "disk logging is off".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import games  # noqa: E402

CFG = """\
[Logging.Console]

## Enables showing a console for log output.
# Setting type: Boolean
# Default value: false
Enabled = {console}

[Logging.Disk]

## Appends to the log file instead of overwriting, on game startup.
AppendLog = false

## Enables writing log messages to disk.
# Setting type: Boolean
# Default value: true
Enabled = {disk}

[Preloader]
ApplyRuntimePatches = true
"""


def write(tmp: Path, console: str, disk: str) -> Path:
    path = tmp / "BepInEx.cfg"
    path.write_text(CFG.format(console=console, disk=disk), encoding="utf-8")
    return path


def test_disk_logging_reads_the_disk_section_not_the_console_one():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # The shipped shape: console off, disk on. Reading the first `Enabled` would wrongly say "off".
        assert games.disk_logging_enabled(write(tmp, "false", "true")) is True
        assert games.disk_logging_enabled(write(tmp, "false", "false")) is False
        assert games.disk_logging_enabled(write(tmp, "true", "false")) is False


def test_a_missing_file_or_key_means_the_bepinex_default():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert games.disk_logging_enabled(tmp / "nope.cfg") is True  # missing file: assume the default, on
        bare = tmp / "bare.cfg"
        bare.write_text("[Logging.Disk]\nAppendLog = false\n", encoding="utf-8")
        assert games.disk_logging_enabled(bare) is True  # missing key: the default is true


def test_the_value_is_read_case_and_space_insensitively():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "BepInEx.cfg"
        path.write_text("[Logging.Disk]\nEnabled =   FALSE  \n", encoding="utf-8")
        assert games.disk_logging_enabled(path) is False
        path.write_text("[Logging.Disk]\nEnabled=True\n", encoding="utf-8")
        assert games.disk_logging_enabled(path) is True


def test_five_is_the_documented_disk_log_limit():
    assert games.DISK_LOG_LIMIT == 5


# -- launch readiness: by port and process, never by the Popen handle --------------------------------

NETSTAT = """\
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1084
  TCP    127.0.0.1:47800        0.0.0.0:0              LISTENING       23180
  TCP    127.0.0.1:47801        0.0.0.0:0              LISTENING       9044
  TCP    127.0.0.1:47800        127.0.0.1:52001        ESTABLISHED     23180
  TCP    [::]:47802             [::]:0                 LISTENING       4120
"""

TASKLIST = (
    '"ULTRAKILL.exe","23180","Console","1","1,004,532 K"\n'
    '"ULTRAKILL.exe","9044","Console","1","57,344 K"\n'
    '"ULTRAKILL.exe","4120","Console","1","N/A"\n'
)


def test_listening_ports_are_read_with_their_owning_pid():
    found = games.parse_listening(NETSTAT)
    assert found == {135: 1084, 47800: 23180, 47801: 9044, 47802: 4120}
    # ESTABLISHED rows are not listeners and must not overwrite the listener's pid.
    assert found[47800] == 23180


def test_working_sets_survive_thousands_separators_and_na():
    sets = games.parse_working_sets(TASKLIST)
    assert sets[23180] == 1004532 * 1024  # a booted copy, about 1 GB
    assert sets[9044] == 57344 * 1024  # the half-booted one that answered every reset "unknown scene"
    assert 4120 not in sets, "N/A must drop out, not read as 0 and so as 'never booted'"


def test_a_re_exec_during_startup_is_not_reported_as_an_exit():
    # The false alarm: the games re-exec under new pids, so the Popen handles report dead processes while the
    # instances are coming up perfectly well. It printed "Instance(s) [0..8] exited during startup" twice
    # during the 2026-09-17 recovery while the ports came up two minutes later.
    wanted = [47800, 47801]
    assert games.startup_verdict(wanted, set(), game_count=9, elapsed=90.0, timeout=180.0) == "waiting"
    assert games.startup_verdict(wanted, {47800}, game_count=9, elapsed=90.0, timeout=180.0) == "waiting"
    assert games.startup_verdict(wanted, {47800, 47801}, game_count=9, elapsed=90.0, timeout=180.0) == "ready"


def test_a_launch_with_no_game_process_left_is_a_real_failure_but_only_after_the_grace():
    wanted = [47800]
    assert games.startup_verdict(wanted, set(), game_count=0, elapsed=5.0, timeout=180.0) == "waiting"
    assert games.startup_verdict(wanted, set(), game_count=0, elapsed=90.0, timeout=180.0) == "no_processes"
    # Ready wins even with nothing matched by name, so an odd tasklist cannot fail a launch that worked.
    assert games.startup_verdict(wanted, {47800}, game_count=0, elapsed=900.0, timeout=180.0) == "ready"


def test_a_launch_that_never_opens_its_ports_times_out():
    assert games.startup_verdict([47800], set(), game_count=3, elapsed=181.0, timeout=180.0) == "timeout"


# ---------------------------------------------------------------------------
# Every process query is bounded and cached (the 2026-09-17 freeze; see CLAUDE.md)
# ---------------------------------------------------------------------------


def _fake_subprocess(monkey, behaviour):
    """Replaces games.subprocess.run and returns the list of argv it was called with."""
    calls: list[list[str]] = []

    def run(argv, **kw):
        calls.append(list(argv))
        return behaviour(argv, kw)

    monkey.append((games.subprocess, "run", games.subprocess.run))
    games.subprocess.run = run
    return calls


def _restore(monkey):
    for obj, name, original in monkey:
        setattr(obj, name, original)
    games._invalidate_proc_cache()


def test_every_process_query_is_bounded():
    """`netstat` and `tasklist` block under WMI/Tcpip contention, which is normal on a box running twelve
    games plus a recovery storm -- and these are now called from inside a worker's recovery ladder and from
    the supervisor. An unbounded one there is a hang indistinguishable from the one being fixed."""
    monkey: list = []
    seen: list[dict] = []

    def behaviour(argv, kw):
        seen.append(kw)
        class R:  # noqa: E306
            stdout = ""
        return R()

    _fake_subprocess(monkey, behaviour)
    try:
        games._invalidate_proc_cache()
        games.listening_pids()
        games._invalidate_proc_cache()
        games.working_sets()
        games._invalidate_proc_cache()
        games.running_pids()
    finally:
        _restore(monkey)
    assert seen, "the queries ran"
    for kw in seen:
        assert kw.get("timeout"), "every process query must carry a timeout"


def test_a_process_query_that_times_out_returns_no_data_instead_of_raising():
    """Empty maps make the caller fall through to its next rung; an exception out of a diagnostic does not."""
    monkey: list = []

    def behaviour(argv, kw):
        raise games.subprocess.TimeoutExpired(argv[0], kw.get("timeout", 1))

    _fake_subprocess(monkey, behaviour)
    try:
        games._invalidate_proc_cache()
        assert games.listening_pids() == {}
        games._invalidate_proc_cache()
        assert games.working_sets() == {}
        games._invalidate_proc_cache()
        assert games.running_pids() == []
        games._invalidate_proc_cache()
        assert games.port_open(47800) is False
    finally:
        _restore(monkey)


def test_a_taskkill_that_hangs_does_not_hang_the_caller():
    monkey: list = []

    def behaviour(argv, kw):
        if argv[0] == "taskkill":
            raise games.subprocess.TimeoutExpired(argv[0], kw.get("timeout", 1))
        class R:  # noqa: E306
            stdout = ""
        return R()

    _fake_subprocess(monkey, behaviour)
    try:
        games._invalidate_proc_cache()
        assert games.kill_unlistening() == []  # nothing is running, so nothing to kill, and no exception
        assert games._run_bounded(["taskkill", "/F", "/PID", "1"]) is False
    finally:
        _restore(monkey)


def test_repeated_polls_share_one_snapshot():
    """Twelve workers polling once a second each is ~18 process spawns a second on a thrashing machine."""
    monkey: list = []

    def behaviour(argv, kw):
        class R:  # noqa: E306
            stdout = "  TCP    127.0.0.1:47800    0.0.0.0:0    LISTENING    4242\n"
        return R()

    calls = _fake_subprocess(monkey, behaviour)
    try:
        games._invalidate_proc_cache()
        for _ in range(10):
            assert games.port_open(47800)
        netstats = [c for c in calls if c[0] == "netstat"]
        assert len(netstats) == 1, "ten polls in one TTL window must cost one netstat, not ten"
    finally:
        _restore(monkey)


# ---------------------------------------------------------------------------
# Hiding training instances from Steam (mod v0.7.1)
# ---------------------------------------------------------------------------


def _fake_popen(monkey):
    """Replaces games.subprocess.Popen and returns the list of argv it was called with."""
    calls: list[list[str]] = []

    class Handle:
        pid = 4242

    def popen(argv, **kw):
        calls.append(list(argv))
        return Handle()

    monkey.append((games.subprocess, "Popen", games.subprocess.Popen))
    games.subprocess.Popen = popen
    return calls


def _flags_file(monkey, tmp: Path):
    monkey.append((games, "INSTANCE_FLAGS", games.INSTANCE_FLAGS))
    games.INSTANCE_FLAGS = tmp / "instance_flags.json"
    return games.INSTANCE_FLAGS


def test_a_training_launch_is_hidden_from_steam_by_default():
    """The user's standing instruction (2026-09-17): training must always be hidden from Steam."""
    monkey: list = []
    calls = _fake_popen(monkey)
    try:
        games.start_instance(47800, 368, 207, 3)
    finally:
        _restore(monkey)
    assert games.NO_STEAM_FLAG in calls[0], calls[0]


def test_steam_opts_one_launch_back_into_visibility():
    monkey: list = []
    calls = _fake_popen(monkey)
    try:
        games.start_instance(47800, 368, 207, 3, no_steam=False)
    finally:
        _restore(monkey)
    assert games.NO_STEAM_FLAG not in calls[0], calls[0]
    assert "-aibridge-port" in calls[0], "the rest of the command line is unchanged"


def test_a_relaunch_rebuilds_the_command_line_the_instance_was_started_with():
    """The env's own-game recovery runs in an SB3 worker that never called `launch`, so the flags have to
    come off disk. A recovery must not quietly flip whether Steam can see that game."""
    for recorded in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            monkey: list = []
            _flags_file(monkey, Path(tmp))
            calls = _fake_popen(monkey)
            _fake_subprocess(monkey, lambda argv, kw: type("R", (), {"stdout": ""})())
            try:
                games.record_instance_flags([47800, 47801], recorded)
                games._invalidate_proc_cache()
                games.relaunch_one(47801, timeout=0)  # timeout 0: start it, don't wait for the port
            finally:
                _restore(monkey)
            started = [c for c in calls if "-aibridge-port" in c]
            assert started, "the replacement was started"
            assert (games.NO_STEAM_FLAG in started[-1]) is recorded, (recorded, started[-1])


def test_an_unknown_port_or_unreadable_file_falls_back_to_hidden():
    """Guessing wrong costs bookkeeping; raising inside the recovery ladder costs a running game."""
    with tempfile.TemporaryDirectory() as tmp:
        monkey: list = []
        path = _flags_file(monkey, Path(tmp))
        try:
            assert games.instance_no_steam(47800) is True, "no file at all"
            games.record_instance_flags([47800], False)
            assert games.instance_no_steam(47800) is False
            assert games.instance_no_steam(47999) is True, "a port nobody recorded"
            path.write_text("{not json", encoding="utf-8")
            assert games.instance_no_steam(47800) is True, "a half-written file"
        finally:
            _restore(monkey)


def test_the_retired_no_steam_flag_still_parses_as_a_no_op():
    """`--no-steam` was the opt-IN for about an hour on 2026-09-17. A shortcut carrying it must still mean
    hidden, not fail to parse."""
    import argparse

    ap = argparse.ArgumentParser()
    games.add_steam_flags(ap)
    assert not ap.parse_args([]).steam, "hidden by default"
    assert not ap.parse_args(["--no-steam"]).steam, "the alias still means hidden"
    assert ap.parse_args(["--steam"]).steam, "--steam is the only way back to visible"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
