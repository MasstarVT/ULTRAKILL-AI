"""Runs a command inside a Windows Job Object with a hard memory cap, so it dies ALONE.

Twice now a one-off offline analysis script has taken this whole machine down: 2026-09-17 09:50 and
2026-09-18 01:40, a single `python.exe` at **55 GB and 70 GB of virtual memory** against a 60 GB commit
limit. Both times the training run, the driver and the desktop session died with it, and the script that
caused it was not the thing anyone was trying to protect.

A Job Object with `JOB_OBJECT_LIMIT_PROCESS_MEMORY` fixes that at the kernel: once the capped process has
committed more than the limit, every further allocation **fails** -- `VirtualAlloc` returns NULL, Python
raises `MemoryError`, numpy raises `numpy.core._exceptions._ArrayMemoryError` -- and the process dies on
its own while everything else on the box keeps running. `JOB_OBJECT_LIMIT_JOB_MEMORY` applies the same
ceiling to the whole process tree, so a script that spawns workers cannot multiply its way around it, and
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` kills the tree if this launcher is itself killed.

    python scripts/run_capped.py --max-gb 8 -- python scripts/build_routes.py --validate
    python scripts/run_capped.py --max-gb 4 -- python scripts/replay_curriculum.py --run campaign_gates

**Use it for every offline analysis script run beside a live training run.** It is the difference between
"my script died" and "the run, the driver and the desktop died".

The launcher assigns ITSELF to the job and then spawns the command normally, rather than the usual
CREATE_SUSPENDED / AssignProcessToJobObject / ResumeThread dance: a child inherits its parent's job, so
this needs no handle surgery, and the launcher's own ~15 MB is charged to a job whose limit is in
gigabytes. The exit code is the command's own, so it composes in a shell exactly as the command did.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.procmem import GB, cap_blas_threads, commit_fraction  # noqa: E402

cap_blas_threads()

JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)), ("PriorityClass", wt.DWORD),
                ("SchedulingClass", wt.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def capped_job(max_bytes: int):
    """Creates the job and puts THIS process in it, so every child inherits the cap. Returns the handle.

    Raises OSError if the kernel refuses, which the caller turns into a refusal to run: silently running
    uncapped would be the one outcome this script exists to prevent.
    """
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = wt.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                             | JOB_OBJECT_LIMIT_JOB_MEMORY
                                             | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.ProcessMemoryLimit = max_bytes
    info.JobMemoryLimit = max_bytes
    if not kernel32.SetInformationJobObject(wt.HANDLE(job), JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(kernel32.GetCurrentProcess())):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def run(command: list[str], max_gb: float) -> int:
    """Runs `command` under a `max_gb` cap and returns its exit code."""
    max_bytes = int(max_gb * GB)
    capped_job(max_bytes)  # kept alive by the process; KILL_ON_JOB_CLOSE reaps the tree if we are killed
    print("run_capped: %.1f GB cap, system commit %.0f%%, running: %s"
          % (max_gb, commit_fraction() * 100, " ".join(command)), flush=True)
    return subprocess.run(command).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gb", type=float, default=8.0, help="hard per-process and per-tree commit cap")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run, after a bare -- separator")
    args = parser.parse_args()

    command = [part for part in args.command if part != "--"]
    if not command:
        parser.error("nothing to run: put the command after a bare -- separator")
    if args.max_gb <= 0:
        parser.error("--max-gb must be positive; a cap of zero would kill the command instantly")
    sys.exit(run(command, args.max_gb))


if __name__ == "__main__":
    main()
