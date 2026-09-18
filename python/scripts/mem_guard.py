"""Memory guard for a live training run: recycles ONE game at a time before the machine runs out of memory.

ULTRAKILL leaks under training. A copy boots near 1 GB of committed memory and was measured at ~6 GB after 5.5
hours (2026-09-18 12:38: Windows' resource-exhaustion event named three copies at 5.8-6.1 GB each; twelve of
them is ~70 GB against a 60 GB commit limit, and the run, the driver and the desktop session all died). Before
the freeze fix the games were restarted every hour or two by crashes and pauses, which hid it.

Every poll this reads each listening game's PRIVATE bytes (commit, not working set: commit is what runs out)
and the system commit charge, and restarts at most one game per cooldown through `games.relaunch_one`, which
leaves the other copies running. The env that owns the port sees its socket drop, reconnects inside its
recovery budget and truncates that one episode as `bridge_reset` (measured: 47.7 s for a killed game).
Never opens a bridge port. Honours the supervisor's and the driver's pause files.

    python scripts/mem_guard.py --run spec_0-3 --game-limit-gb 2.5 --commit-limit-frac 0.93
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GB = 1024 ** 3


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


def choose_victim(private_by_port: dict[int, int], commit_frac: float, game_limit: int,
                  commit_limit_frac: float) -> tuple[int, str] | None:
    """The one port to recycle now, with the reason, or None. Pure, so it is tested without a process.

    A game over its own limit goes first (the fattest of them). Otherwise, when the SYSTEM is short, the fattest
    game goes whatever its size: something else may be eating the memory (an analysis script has twice taken
    50-70 GB), and the games are the only thing this guard can safely give back.
    """
    if not private_by_port:
        return None
    port, size = max(private_by_port.items(), key=lambda kv: kv[1])
    if size > game_limit:
        return port, "game at %.1f GB, over its %.1f GB limit" % (size / GB, game_limit / GB)
    if commit_frac > commit_limit_frac:
        return port, "system commit at %.0f%% (limit %.0f%%); fattest game is %.1f GB" % (
            commit_frac * 100, commit_limit_frac * 100, size / GB)
    return None


def bridge_games(listening: dict[int, int], game_pids: set[int], base_port: int, ports: int) -> dict[int, int]:
    """port -> pid for bridge ports only. `listening_pids` is every listening port on the machine (19 of them
    with no game running at all), and recycling "the fattest" of those would kill something that is not ours:
    a port counts only inside the bridge range AND when its owner is an ULTRAKILL process."""
    return {port: pid for port, pid in listening.items()
            if base_port <= port < base_port + ports and pid in game_pids}


def paused(run: str) -> bool:
    return any(p.exists() for p in (ROOT / "runs" / run / "SUPERVISOR_PAUSE",
                                    ROOT / "runs" / "specialists" / "DRIVER_PAUSE"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="", help="run whose SUPERVISOR_PAUSE file is honoured (the driver's always is)")
    parser.add_argument("--game-limit-gb", type=float, default=2.5, help="recycle a game above this much commit")
    parser.add_argument("--commit-limit-frac", type=float, default=0.93, help="recycle the fattest game above this system commit share")
    parser.add_argument("--base-port", type=int, default=47800)
    parser.add_argument("--ports", type=int, default=32, help="size of the bridge port range that is ours to touch")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--cooldown-seconds", type=float, default=240.0, help="at most one recycle per this long")
    parser.add_argument("--dry-run", action="store_true", help="print one reading and the decision, change nothing")
    args = parser.parse_args()

    import games  # noqa: E402  (netstat-based; never connects to a bridge port)

    last_recycle, last_heartbeat = 0.0, 0.0
    while True:
        try:
            owners = bridge_games(games.listening_pids(), set(games.working_sets()), args.base_port, args.ports)
            sizes = {port: private_bytes(pid) for port, pid in owners.items()}
            sizes = {port: size for port, size in sizes.items() if size}
            frac = commit_fraction()
            now = time.monotonic()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if args.dry_run or now - last_heartbeat > 1800:
                total = sum(sizes.values()) / GB
                top = max(sizes.values(), default=0) / GB
                print("%s mem: %d games, %.1f GB total, fattest %.1f GB, system commit %.0f%%"
                      % (stamp, len(sizes), total, top, frac * 100), flush=True)
                last_heartbeat = now
            victim = choose_victim(sizes, frac, int(args.game_limit_gb * GB), args.commit_limit_frac)
            if victim and args.dry_run:
                print("%s would recycle port %d: %s" % (stamp, victim[0], victim[1]), flush=True)
            elif victim and not paused(args.run) and now - last_recycle >= args.cooldown_seconds:
                print("%s recycling port %d: %s" % (stamp, victim[0], victim[1]), flush=True)
                ok = games.relaunch_one(victim[0])
                print("%s port %d %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), victim[0],
                                         "is listening again" if ok else "did NOT come back"), flush=True)
                last_recycle = time.monotonic()
        except Exception as error:  # the guard must outlive a netstat hiccup
            print("%s error: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), error), flush=True)
        if args.dry_run:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
