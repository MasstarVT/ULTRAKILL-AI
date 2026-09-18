"""`run_capped.py` must kill a runaway script and NOTHING else.  python tests/test_run_capped.py

Two offline analysis scripts have taken this machine down, at 55 GB and 70 GB against a 60 GB commit limit,
and both times they took the training run, the driver and the desktop session with them. These tests run a
child that really does try to allocate past its cap and assert that the child dies, the launcher reports it,
and this test process is untouched. Nothing here commits more than a few MB: the whole point of a job-object
cap is that the doomed allocation FAILS instead of succeeding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultrakill_ai.procmem import IGNORE_COMMIT_ENV, commit_refusal, private_bytes  # noqa: E402

CAPPED = ROOT / "scripts" / "run_capped.py"
import os  # noqa: E402


def _run(args: list[str], timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CAPPED), *args], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))


def test_a_command_that_stays_under_its_cap_runs_normally():
    done = _run(["--max-gb", "1", "--", sys.executable, "-c", "print('hello from inside the job')"])
    assert done.returncode == 0, done.stderr
    assert "hello from inside the job" in done.stdout
    assert "1.0 GB cap" in done.stdout


def test_a_child_that_allocates_past_the_cap_dies_and_the_parent_survives():
    """The headline property: 2 GB wanted against a 0.5 GB cap. The allocation must FAIL, not succeed."""
    before = private_bytes(os.getpid())
    # The marker is spelled in two halves so it does not appear in the launcher's own "running: ..." banner,
    # which echoes the command line and would otherwise match every check below.
    done = _run(["--max-gb", "0.5", "--", sys.executable, "-c",
                 "buf = bytearray(2 * 1024**3); print('ALLO' + 'CATED', len(buf))"])
    after = private_bytes(os.getpid())
    assert done.returncode != 0, f"the runaway child was allowed to finish: {done.stdout}"
    assert "MemoryError" in done.stderr, f"expected a failed allocation, got: {done.stderr[-200:]}"
    assert "ALLOCATED" not in done.stdout, "the capped child actually got its 2 GB"
    assert (after - before) < 200 * 1024 * 1024, "this test process paid for the child's allocation"
    print(f"[ok] capped child exited {done.returncode}; test process grew "
          f"{(after - before) / 1e6:.1f} MB")


def test_the_cap_binds_the_whole_tree_not_just_the_first_process():
    """A script that spawns a worker must not be able to multiply its way around the job limit."""
    inner = ("import subprocess, sys;"
             "sys.exit(subprocess.run([sys.executable, '-c',"
             " 'buf = bytearray(2 * 1024**3); print(\"GRAND\" + \"CHILD OK\")']).returncode)")
    done = _run(["--max-gb", "0.5", "--", sys.executable, "-c", inner])
    assert done.returncode != 0, "a grandchild escaped the job's memory limit"
    assert "GRANDCHILD OK" not in done.stdout, "a grandchild got memory the job should have refused"


def test_the_exit_code_is_the_commands_own():
    done = _run(["--max-gb", "1", "--", sys.executable, "-c", "raise SystemExit(7)"])
    assert done.returncode == 7, f"expected the command's own 7, got {done.returncode}"


def test_it_refuses_an_empty_or_nonsense_invocation():
    assert _run(["--max-gb", "1", "--"]).returncode != 0
    assert _run(["--max-gb", "0", "--", sys.executable, "-c", "pass"]).returncode != 0


# --------------------------------------------------------------- the offline-script commit refusal


def test_a_heavy_script_refuses_when_the_box_is_nearly_out_of_commit():
    message = commit_refusal("build_routes.py", limit=0.85, frac=0.91)
    assert "refuses to start" in message and "91%" in message
    assert "run_capped.py" in message, "the refusal must say how to run it anyway"
    assert IGNORE_COMMIT_ENV in message


def test_a_heavy_script_runs_when_there_is_room():
    assert commit_refusal("build_routes.py", limit=0.85, frac=0.60) == ""
    assert commit_refusal("build_routes.py", limit=0.85, frac=0.85) == "", "at the limit is not over it"


def test_build_routes_really_carries_the_refusal():
    """A guard nobody wired in is not a guard. This reads the file rather than running the 2-minute script."""
    source = (ROOT / "scripts" / "build_routes.py").read_text(encoding="utf-8")
    assert "refuse_if_commit_high(\"build_routes.py\")" in source
    assert source.index("cap_blas_threads()") < source.index("import numpy"), \
        "the BLAS cap must be set before numpy is imported or it does nothing"


def main() -> None:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok", name)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
