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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
