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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultrakill_ai.procmem import cap_blas_threads  # noqa: E402

cap_blas_threads()  # a guard that reserves 785 MB for numpy's BLAS threads is part of the problem

from ultrakill_ai.procmem import (  # noqa: E402,F401
    GB, MAX_LIMIT_GB, MIN_FRESH_GB, commit_fraction, derive_game_limit, private_bytes)

# How much a game may grow above the FRESHEST copy running before it is recycled; see `derive_game_limit`,
# which this guard shares with the env's own episode-boundary recycle so the two cannot disagree.
DEFAULT_GROWTH_GB = 1.0


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
    parser.add_argument("--game-limit-gb", type=float, default=0.0,
                        help="fixed per-game limit in GB; 0 (the default) derives it from the live fleet")
    parser.add_argument("--growth-gb", type=float, default=DEFAULT_GROWTH_GB,
                        help="how far above the FRESHEST running copy a game may grow before it is recycled")
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
            limit = (int(args.game_limit_gb * GB) if args.game_limit_gb > 0
                     else derive_game_limit(sizes, int(args.growth_gb * GB)))
            if args.dry_run or now - last_heartbeat > 1800:
                total = sum(sizes.values()) / GB
                top = max(sizes.values(), default=0) / GB
                print("%s mem: %d games, %.1f GB total, fattest %.1f GB, limit %.1f GB, system commit %.0f%%"
                      % (stamp, len(sizes), total, top, limit / GB, frac * 100), flush=True)
                last_heartbeat = now
            victim = choose_victim(sizes, frac, limit, args.commit_limit_frac)
            if victim and args.dry_run:
                print("%s would recycle port %d: %s" % (stamp, victim[0], victim[1]), flush=True)
            elif victim and not paused(args.run) and now - last_recycle >= args.cooldown_seconds:
                port, reason = victim
                before = sizes.get(port, 0)
                print("%s recycling port %d: %s" % (stamp, port, reason), flush=True)
                ok = games.relaunch_one(port)
                after = private_bytes(games.listening_pids().get(port) or 0) or 0
                # One line per recycle, with both sizes: this is the only record of what the leak costs, and
                # `after` is also the freshest baseline measurement the derived limit will ever see.
                print("%s port %d %s: %.2f GB -> %.2f GB (freed %.2f GB), system commit %.0f%%"
                      % (time.strftime("%Y-%m-%d %H:%M:%S"), port,
                         "is listening again" if ok else "did NOT come back",
                         before / GB, after / GB, (before - after) / GB, commit_fraction() * 100), flush=True)
                last_recycle = time.monotonic()
        except Exception as error:  # the guard must outlive a netstat hiccup
            print("%s error: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), error), flush=True)
        if args.dry_run:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
