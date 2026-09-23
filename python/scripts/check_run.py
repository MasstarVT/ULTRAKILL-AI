"""One cheap reading of the live specialist run, for a periodic check-in. No game, no socket, no torch.

Reads only files and netstat: the driver's state, the stage's status.json, the TAIL of its episodes.jsonl, the
memory guard's log and the system commit charge. Prints a dozen lines and an ALERTS list; exit code 1 when
something needs a look, 0 when it does not.

    python scripts/check_run.py            # the running stage
    python scripts/check_run.py --last 150 # judge on the last 150 fresh episodes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultrakill_ai.procmem import GB, commit_fraction, private_bytes  # noqa: E402

TAIL_BYTES = 600_000  # ~1,000 episode rows; the file itself is never read whole


def tail_rows(path: Path, max_bytes: int = TAIL_BYTES) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        chunk = handle.read().decode("utf-8", errors="replace")
    rows = []
    for line in chunk.splitlines()[1 if size > max_bytes else 0:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def age_of_last_stamp(path: Path) -> float | None:
    """Seconds since the last `YYYY-MM-DD HH:MM:SS`-stamped line of a log, or None."""
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]):
        try:
            return time.time() - time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
    return None


DRIVER_QUERY = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'campaign_driver' } | "
                "ForEach-Object { $_.ProcessId }")


def parse_pids(text: str) -> list[int]:
    return [int(tok) for tok in text.split() if tok.isdigit()]


def driver_pids(timeout: float = 40.0) -> list[int] | None:
    """Pids of every python process running campaign_driver.py (a read-only WMI query), None if the query failed."""
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", DRIVER_QUERY],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_pids(done.stdout) if done.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int, default=100, help="fresh episodes to judge on")
    parser.add_argument("--base-port", type=int, default=47800)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    alerts: list[str] = []

    state_path = ROOT / "runs" / "specialists" / "driver_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    current = state.get("current") or {}
    done = ["%s%s=%s" % (h.get("level", "?").replace("Level ", ""), "s" if h.get("kind") == "speed" else "",
                         h.get("status")) for h in state.get("history", [])]
    run = current.get("run")
    print("time      %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("stage     %s [%s, round %s] run %s | history: %s" % (
        current.get("level"), current.get("kind") or "complete", current.get("round") or 1, run, " ".join(done) or "-"))
    if not run:
        alerts.append("the driver has no current stage")

    status = {}
    if run:
        status_path = ROOT / "runs" / run / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            age = time.time() - status_path.stat().st_mtime
            steps = int(status.get("timesteps") or 0)
            into = steps - int(current.get("start_steps") or 0)
            camp = status.get("campaign") or {}
            print("trainer   %s, %s steps (%s into the stage), %.0f steps/s, status %.0f s old" % (
                status.get("state"), f"{steps:,}", f"{into:,}", status.get("steps_per_s") or 0, age))
            print("campaign  fresh rate %s over %s | median %s | best %s | target %s" % (
                camp.get("fresh_completion_rate"), camp.get("fresh_window"), camp.get("median_time_50"),
                camp.get("best_time"), camp.get("target_seconds")))
            if age > 900:
                alerts.append("status.json is %.0f min old: the trainer is not stepping" % (age / 60))
        else:
            alerts.append("no status.json for %s" % run)

        rows = [r for r in tail_rows(ROOT / "runs" / run / "episodes.jsonl") if r.get("fresh_start") == 1.0]
        rows = rows[-args.last:]
        if rows:
            hops = Counter(r.get("gate_hops_best") for r in rows)
            ends = Counter(r.get("end_reason") for r in rows)
            completed = [r for r in rows if r.get("completed") == 1.0]
            times = sorted(r["level_seconds"] for r in completed if r.get("level_seconds"))
            ys = sorted(r["end_pos"][1] for r in rows if r.get("end_pos"))
            print("last %d fresh episodes (steps %s..%s): %d completed%s" % (
                len(rows), f"{int(rows[0]['timesteps']):,}", f"{int(rows[-1]['timesteps']):,}", len(completed),
                (", times min %.1f median %.1f s" % (times[0], times[len(times) // 2])) if times else ""))
            print("  hops_best %s" % dict(sorted(hops.items(), key=lambda kv: (kv[0] is None, kv[0]))))
            print("  ends %s | end y median %.0f, max %.0f" % (dict(ends), ys[len(ys) // 2] if ys else 0, ys[-1] if ys else 0))
            if ends.get("bridge_reset", 0) > len(rows) * 0.25:
                alerts.append("%d of %d fresh episodes ended in bridge_reset" % (ends["bridge_reset"], len(rows)))

    import games  # noqa: E402  netstat only; never opens a bridge port

    listening = games.listening_pids()
    game_pids = set(games.working_sets())
    ours = {p: pid for p, pid in listening.items()
            if args.base_port <= p < args.base_port + args.count and pid in game_pids}
    sizes = [private_bytes(pid) or 0 for pid in ours.values()]
    frac = commit_fraction()
    print("games     %d/%d listening, %.1f GB total, fattest %.1f GB | system commit %.0f%%" % (
        len(ours), args.count, sum(sizes) / GB, max(sizes, default=0) / GB, frac * 100))
    if len(ours) < args.count:
        missing = [p for p in range(args.base_port, args.base_port + args.count) if p not in ours]
        alerts.append("ports not listening: %s" % missing)
    if frac > 0.90:
        alerts.append("system commit at %.0f%%" % (frac * 100))

    guard_log = ROOT / "runs" / "mem_guard.log"
    if guard_log.exists():
        lines = guard_log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
        hour_ago = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 3600))
        recent = [ln for ln in lines if ln[:19] >= hour_ago and ln[:2] == "20"]
        print("guard     %d recycles in the last hour, last line %.0f min ago" % (
            sum("recycling port" in ln for ln in recent), (age_of_last_stamp(guard_log) or 0) / 60))
        if any("did NOT come back" in ln for ln in recent):
            alerts.append("mem_guard: a recycled game did NOT come back")
        if (age_of_last_stamp(guard_log) or 0) > 2400:
            alerts.append("mem_guard has been silent for over 40 min: is it running?")
    driver_age = age_of_last_stamp(ROOT / "runs" / "specialists_driver.log")
    pids = driver_pids()
    alive = "process query failed" if pids is None else ("pid %s" % ",".join(map(str, pids)) if pids else "NOT RUNNING")
    print("driver    %s, last log line %.0f min ago" % (alive, (driver_age or 0) / 60))
    if pids is not None and not pids:
        alerts.append("the driver process is NOT running")
    elif driver_age is None or driver_age > 4500:
        alerts.append("the driver has logged nothing for over 75 min: is it running?")
    if (ROOT / "runs" / "specialists" / "DRIVER_PAUSE").exists():
        alerts.append("DRIVER_PAUSE exists (a bounce in progress, or one left behind)")

    print("ALERTS    %s" % ("; ".join(alerts) if alerts else "none"))
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
