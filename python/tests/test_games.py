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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
