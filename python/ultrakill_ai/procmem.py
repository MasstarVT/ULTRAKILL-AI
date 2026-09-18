"""Process-memory hygiene for the light processes around a training run.

**Measured 2026-09-18 on this box** (24 logical CPUs, numpy 2.5.3): `import numpy` alone costs
**785 MB of COMMITTED private bytes** for 13 MB of working set, because OpenBLAS reserves a per-thread
buffer pool (~34 MB per thread, measured: 9 MB at 1 thread, 43 MB at 2) when the library loads.

Committed bytes are what this machine runs out of -- it hit its 60 GB commit limit twice and took the
desktop session down with it -- and that reservation is charged in full to every process that so much as
touches numpy. That is twelve `SubprocVecEnv` workers whose entire numeric workload is packing a
479-float observation, plus the dashboard, the driver and the times watcher, which mostly read JSON:

    import numpy                                785 MB private,  28 MB working set
    OPENBLAS_NUM_THREADS=1, import numpy          9 MB private,  26 MB working set

**Only `OPENBLAS_NUM_THREADS` is set, and that is deliberate.** `OMP_NUM_THREADS` is torch's knob as
well, and setting it to 1 costs the trainer 2.5x per update (the KMP_BLOCKTIME note in `training.py`).
Capping BLAS threads here leaves torch's own OpenMP pool alone, so a process may still import torch at
full speed afterwards -- it simply loses numpy's multithreaded BLAS, which no process in this project
uses for anything bigger than a 512x512 matrix.

**It must run BEFORE numpy is imported**: OpenBLAS reads the variable once, when the DLL loads. Every
caller therefore calls it as its first statement, above its `ultrakill_ai` imports. `cap_blas_threads`
returns whether it was in time, so a test can prove the ordering rather than assume it.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import sys

# Set only what numpy's BLAS reads. OMP_NUM_THREADS is left alone on purpose -- see the module docstring.
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS",)

GB = 1024 ** 3
MB = 1024 ** 2


def cap_blas_threads(threads: int = 1) -> bool:
    """Caps numpy's BLAS thread pool, which is where the committed memory goes.

    Uses `setdefault`, so an operator who exports `OPENBLAS_NUM_THREADS` by hand still wins. Returns
    True when this ran early enough to matter (numpy not yet imported) and False when numpy is already
    loaded and the variable can no longer change anything.
    """
    for name in BLAS_THREAD_VARS:
        os.environ.setdefault(name, str(max(1, int(threads))))
    return "numpy" not in sys.modules


def heavy_modules() -> list[str]:
    """The heavy top-level modules currently imported. The light processes assert this is empty-ish."""
    return sorted({"torch", "stable_baselines3", "sb3_contrib"} & {m.split(".")[0] for m in sys.modules})


# --------------------------------------------------------------------------- reading what a process costs
#
# COMMIT, not working set. A leaking game's working set is held down by the memory manager while its commit
# charge keeps climbing, and commit is what the machine ran out of. These live here rather than in
# `scripts/mem_guard.py` because `UltrakillEnv` needs the same reading to recycle its own game at an episode
# boundary, and the env may not import from `scripts/`.


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def private_bytes(pid: int) -> int | None:
    """Committed private memory of `pid`, or None when the process cannot be opened (gone, or protected)."""
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.OpenProcess.restype = wt.HANDLE
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return None
    try:
        counters = _MemCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(wt.HANDLE(handle), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PrivateUsage)
    finally:
        kernel32.CloseHandle(wt.HANDLE(handle))


def commit_fraction() -> float:
    """System commit charge as a share of the commit limit (RAM + page file). This is what actually runs out."""
    status = _MemStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total = float(status.ullTotalPageFile) or 1.0
    return 1.0 - float(status.ullAvailPageFile) / total


# A half-booted copy sits at ~56-250 MB, and taking THAT as the fresh baseline would put the limit under the
# boot size and recycle the whole fleet forever. Nothing below this counts as a baseline sample.
MIN_FRESH_GB = 0.8
# The guard must still fire when every copy has grown TOGETHER -- the shape that killed this box, three games
# at 5.8-6.1 GB with nothing fresh left to compare against.
MAX_LIMIT_GB = 3.0


def derive_game_limit(sizes, growth: int, min_fresh: int = int(MIN_FRESH_GB * GB),
                      ceiling: int = int(MAX_LIMIT_GB * GB)) -> int:
    """The per-game recycle limit, measured off the live fleet: freshest copy + allowed growth.

    The freshest copy running is the smallest one that is actually booted, and it is the only honest estimate
    of "what this build costs before it has leaked" that anything can get without restarting something. A
    constant cannot do this job: a fresh game was ~1.0 GB before mod 0.7.x and is 1.3-1.5 GB now, so a number
    written against one build silently becomes wrong against the other -- too low and healthy games are
    recycled in a loop, too high and the guard never fires at all.

    Shared by `scripts/mem_guard.py` (which watches the fleet from outside) and `UltrakillEnv` (which recycles
    its own game at an episode boundary), so the two cannot drift apart. Pure, so it is tested without a
    process; accepts a dict of port -> bytes or a bare sequence of sizes.
    """
    values = list(sizes.values()) if hasattr(sizes, "values") else list(sizes)
    booted = [size for size in values if size >= min_fresh]
    baseline = min(booted) if booted else min_fresh
    return min(ceiling, baseline + growth)


# The share of the commit limit above which a heavy OFFLINE script refuses to start. Not a guess: the live
# run alone sits near 0.80, so anything above this is either a second analysis script or a fleet that has
# already leaked, and in both cases starting a 50 GB job is what takes the machine down rather than the run.
OFFLINE_COMMIT_LIMIT = 0.85
IGNORE_COMMIT_ENV = "ULTRAKILL_AI_IGNORE_COMMIT"


def commit_refusal(what: str, limit: float = OFFLINE_COMMIT_LIMIT, frac: float | None = None) -> str:
    """The message a heavy offline script should die with, or "" when there is room. Pure, so it is tested."""
    frac = commit_fraction() if frac is None else frac
    if frac <= limit:
        return ""
    return ("%s refuses to start: system commit is at %.0f%% (limit %.0f%%). Offline analysis has twice taken "
            "this machine down at 55-70 GB while a run was live. Wait for the run to settle, or bound it:\n"
            "    python scripts/run_capped.py --max-gb 8 -- <your command>\n"
            "Set %s=1 to override." % (what, frac * 100, limit * 100, IGNORE_COMMIT_ENV))


def refuse_if_commit_high(what: str, limit: float = OFFLINE_COMMIT_LIMIT) -> None:
    """Exits rather than starting heavy offline work on a box that is nearly out of committed memory."""
    if os.environ.get(IGNORE_COMMIT_ENV):
        return
    message = commit_refusal(what, limit)
    if message:
        raise SystemExit(message)
