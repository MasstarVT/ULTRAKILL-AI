"""Prints test_tech_env.py's v1 pins (V1_FORWARD_LINE .. V1_PROGRESS_KEYS_SHA256), computed on a BASE commit.

    python tests/tools/tech_v1_pins.py [--commit 0be539f]

A pin is only legitimate when it is computed by the package the live fleet ran BEFORE the change it guards. This
extracts `python/ultrakill_ai` of `--commit` (default 0be539f, the base of S7 Task 3) with `git archive` into a
temporary directory, imports it BEFORE test_tech_env (whose own `sys.path` insert would otherwise pick up the
working tree's package), checks that every `ultrakill_ai` module in play came from that extraction, and prints the
constants ready to paste. The harness -- the fakes, the script, `v1_pins()` -- is the working tree's, so the base
commit must still import what test_tech_env imports from `ultrakill_ai` (true for 0be539f and later).

Never run it against the working tree to "update" a failing pin: the output would describe the code under test,
not the code it must match. A failing v1 pin means the v1 path moved -- find out what moved it.

No game, no socket: the env talks to test_campaign_env's FakeLevel, and ProgressCallback writes into a temp dir.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PYTHON = Path(__file__).resolve().parents[2]  # the repo's python/
REPO = PYTHON.parent
TESTS = PYTHON / "tests"
DEFAULT_BASE = "0be539f"


def extract_package(commit: str, dest: Path) -> Path:
    """`git archive <commit> python/ultrakill_ai`, unpacked under `dest`. Returns the directory to put on sys.path."""
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "python/ultrakill_ai"],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(dest, filter="data")
    return dest / "python"


def _outside(base: Path) -> list[str]:
    """Every imported `ultrakill_ai` module whose file is NOT under the extracted base package."""
    return sorted(name for name, module in list(sys.modules.items())
                  if name.split(".")[0] == "ultrakill_ai" and getattr(module, "__file__", None)
                  and not Path(module.__file__).resolve().is_relative_to(base.resolve()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", default=DEFAULT_BASE, help=f"the base commit (default {DEFAULT_BASE})")
    args = parser.parse_args(argv)
    if "ultrakill_ai" in sys.modules:
        raise SystemExit("ultrakill_ai is already imported; run this as a script, in a fresh interpreter")
    sys.dont_write_bytecode = True  # nothing written into the extraction, so the temp dir always cleans up
    with tempfile.TemporaryDirectory() as tmp:
        base = extract_package(args.commit, Path(tmp))
        sys.path.insert(0, str(base))
        import ultrakill_ai  # noqa: PLC0415 - must be the extracted package, cached before the tests import it

        sys.path.insert(0, str(TESTS))
        import test_tech_env  # noqa: PLC0415

        pins = test_tech_env.v1_pins()
        stray = _outside(base)
        if stray:
            raise SystemExit(f"not computed on {args.commit}: {stray} imported from outside {base}")
        print(f"# v1 pins computed by python/ultrakill_ai at {args.commit} ({Path(ultrakill_ai.__file__).parent})")
        for name in test_tech_env.V1_PIN_NAMES:
            print(f"{name} = {pins[name]!r}")
        now = test_tech_env.v1_pin_mismatches()
        print("# test_tech_env.py's constants " + ("match." if not now else "DIFFER:\n#   " + "\n#   ".join(now)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
