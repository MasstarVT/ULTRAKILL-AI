"""Launches and stops several ULTRAKILL instances for parallel training.

    python scripts/games.py launch --count 5     # ports 47800..47804, small windows tiled on monitor 3
    python scripts/games.py tile --monitor 3     # move running instances to a monitor
    python scripts/games.py status
    python scripts/games.py stop                 # closes them and restores your display settings

Each instance runs as a "training instance": it opens your settings read-only and never writes
settings or save data, so parallel copies can't corrupt anything. Unity saves the window size on exit,
so launch snapshots your display settings and stop puts them back.

**How many copies can run.** The old "at most 5" limit was never the game's: BepInEx's DiskLogListener opens
`LogOutput.log` plus `.1`-`.4` and a sixth copy fails to load because it cannot open a log file. Setting
`[Logging.Disk] Enabled = false` in `<game>/BepInEx/config/BepInEx.cfg` removes the listener and with it the
limit (measured 2026-09-17: 8 copies load and serve the bridge). Nothing else here caps the count -- ports are
`base_port + i`, readiness is read from netstat and `tile` wraps onto a second row when a row is full. With disk
logging off the plugin still logs to the console listener, so a crash has no file to leave a trace in; turn
`Enabled` back on when diagnosing one.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import winreg
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.windows import monitor_work_area  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BACKUP = REPO / "python" / "runs" / "display_backup.json"
REG_PATH = r"Software\Hakita\ULTRAKILL"
EXE_NAME = "ULTRAKILL.exe"
STEAM_APP_ID = "1229490"


def game_dir() -> Path:
    if os.environ.get("ULTRAKILL_DIR"):
        return Path(os.environ["ULTRAKILL_DIR"])
    props = REPO / "mod" / "GamePaths.props"
    if props.exists():
        m = re.search(r"<UltrakillDir>(.*?)</UltrakillDir>", props.read_text(encoding="utf-8"))
        if m:
            return Path(m.group(1))
    return Path(r"C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL")


# How many copies BepInEx's DiskLogListener allows: LogOutput.log plus .1-.4.
DISK_LOG_LIMIT = 5


def disk_logging_enabled(cfg_path: Path) -> bool:
    """True when BepInEx writes a log file per copy, which is what caps the number of instances at five.

    Reads `Enabled` inside `[Logging.Disk]` only -- `[Logging.Console]` has a key of the same name above it.
    A missing file or a missing key means the BepInEx default, which is on.
    """
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return True
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section == "Logging.Disk" and line.lower().startswith("enabled"):
            return line.split("=", 1)[-1].strip().lower() != "false"
    return True


# ---------------------------------------------------------------------------
# Processes and windows
# ---------------------------------------------------------------------------


def running_pids() -> list[int]:
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {EXE_NAME}", "/FO", "CSV", "/NH"], capture_output=True, text=True
    ).stdout
    return [int(line.split(",")[1].strip('"')) for line in out.splitlines() if line.startswith(f'"{EXE_NAME}"')]


def stop_all(timeout: float = 30.0) -> None:
    pids = running_pids()
    if not pids:
        return
    print(f"Closing {len(pids)} running game(s)...")
    subprocess.run(["taskkill", "/IM", EXE_NAME], capture_output=True)  # polite close, lets the game save state
    deadline = time.monotonic() + timeout
    while running_pids() and time.monotonic() < deadline:
        time.sleep(1)
    if running_pids():
        subprocess.run(["taskkill", "/F", "/IM", EXE_NAME], capture_output=True)
        time.sleep(2)


def parse_listening(netstat_output: str) -> dict[int, int]:
    """Port -> owning pid, for every LISTENING TCP socket in `netstat -ano -p tcp` output.

    The pid matters as well as the port: a game that boots slowly still opens its bridge port (the plugin
    starts the server long before Addressables finish), so the only way to address that one instance -- to
    restart it without stopping the other eleven -- is through the pid listening on its port.
    """
    found: dict[int, int] = {}
    for line in netstat_output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
            try:
                found[int(parts[1].rsplit(":", 1)[1])] = int(parts[4])
            except ValueError:
                continue
    return found


def listening_pids() -> dict[int, int]:
    """Port -> pid for everything listening. Never probe the bridge by connecting: it is a single-client server
    that drops its current client when a new one connects, so a probe connection kicks a running trainer off the
    game (this killed a training run once)."""
    return parse_listening(subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout)


def listening_ports() -> set[int]:
    return set(listening_pids())


def port_open(port: int) -> bool:
    return port in listening_ports()


def parse_working_sets(tasklist_output: str) -> dict[int, int]:
    """pid -> working set in bytes, from `tasklist /FO CSV /NH`.

    The memory column is a display string with thousands separators and a unit (`"1,004,532 K"`), so it is
    parsed rather than trusted: separators are locale-dependent and anything that is not a digit is dropped.
    Rows whose memory reads as "N/A" (a process this session cannot query) are skipped rather than counted as 0,
    which would read as "not booted" forever.
    """
    sets: dict[int, int] = {}
    for row in csv.reader(io.StringIO(tasklist_output)):
        if len(row) < 5:
            continue
        digits = re.sub(r"[^0-9]", "", row[4])
        if not digits:
            continue
        try:
            sets[int(row[1])] = int(digits) * 1024  # the column is kilobytes
        except ValueError:
            continue
    return sets


def working_sets() -> dict[int, int]:
    """Working set in bytes for every running copy of the game: the boot-progress signal the health gate uses."""
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {EXE_NAME}", "/FO", "CSV", "/NH"], capture_output=True, text=True
    ).stdout
    return parse_working_sets(out)


def startup_verdict(wanted: list[int], open_ports: set[int], game_count: int, elapsed: float,
                    timeout: float, grace: float = 60.0) -> str:
    """"ready" / "waiting" / "no_processes" / "timeout" for a launch that is coming up.

    Readiness is judged by PORT and by whether ANY game process exists, never by the `Popen` handles `launch`
    created. The game re-execs itself under a new pid during startup, so the original handle reaps a dead
    process while the instance is coming up perfectly well -- which is what printed the false
    "Instance(s) [0..8] exited during startup" twice during the 2026-09-17 recovery while the ports came up two
    minutes later. Only "not one copy of the game is running, well after they were all started" is real.
    """
    if all(port in open_ports for port in wanted):
        return "ready"
    if game_count == 0 and elapsed >= grace:
        return "no_processes"
    if elapsed >= timeout:
        return "timeout"
    return "waiting"


def windows_for_pid(pid: int) -> list[int]:
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def tile(pids: list[int], width: int, height: int, monitor: int | None) -> None:
    """Arranges the game windows in a row along the top of the chosen monitor, leaving room below for the dashboard."""
    user32 = ctypes.windll.user32
    left, top, right, bottom = monitor_work_area(monitor)
    # Window borders add a little to the client size.
    cell_w, cell_h = width + 16, height + 39
    cols = max(1, (right - left) // cell_w)
    for i, pid in enumerate(pids):
        for hwnd in windows_for_pid(pid):
            x, y = left + (i % cols) * cell_w, top + (i // cols) * cell_h
            user32.SetWindowPos(hwnd, 0, x, y, cell_w, cell_h, 0x0010 | 0x0004)  # SWP_NOACTIVATE | SWP_NOZORDER


# ---------------------------------------------------------------------------
# Display settings backup (Unity saves window size to the registry on exit)
# ---------------------------------------------------------------------------


def backup_display() -> None:
    if BACKUP.exists():
        return  # Keep the original backup from before any training launch.
    values = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH) as key:
            i = 0
            while True:
                try:
                    name, value, kind = winreg.EnumValue(key, i)
                except OSError:
                    break
                if name.startswith("Screenmanager"):
                    values[name] = {"type": kind, "value": value.hex() if isinstance(value, bytes) else value}
                i += 1
    except FileNotFoundError:
        return
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_text(json.dumps(values, indent=1), encoding="utf-8")
    print(f"Saved your display settings to {BACKUP}")


def restore_display() -> None:
    if not BACKUP.exists():
        return
    values = json.loads(BACKUP.read_text(encoding="utf-8"))
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_SET_VALUE) as key:
        for name, entry in values.items():
            value = bytes.fromhex(entry["value"]) if entry["type"] == winreg.REG_BINARY else entry["value"]
            winreg.SetValueEx(key, name, 0, entry["type"], value)
    BACKUP.unlink()
    print("Restored your display settings")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def start_instance(port: int, width: int, height: int, job_workers: int) -> int:
    """Starts one copy on `port` and returns the pid it was started with.

    That pid is only a launch handle: the game re-execs under a new one, so nothing may treat it as the
    instance's identity. Use `listening_pids()[port]` for that once the port is up.
    """
    exe = game_dir() / EXE_NAME
    args = [
        str(exe), "-aibridge-port", str(port),
        "-screen-fullscreen", "0", "-screen-width", str(width), "-screen-height", str(height),
        # Fewer Unity worker threads per copy, so several games don't fight over the CPU.
        "-job-worker-count", str(job_workers),
    ]
    # Below-normal priority keeps your PC responsive while the games train.
    flags = subprocess.DETACHED_PROCESS | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    env = dict(os.environ, SteamAppId=STEAM_APP_ID, SteamGameId=STEAM_APP_ID)
    return subprocess.Popen(args, cwd=exe.parent, env=env, creationflags=flags).pid


def relaunch_one(port: int, width: int = 368, height: int = 207, job_workers: int = 3,
                 timeout: float = 240.0) -> bool:
    """Restarts the single instance on `port`, leaving every other game running.

    `launch` cannot do this: it calls `stop_all()` first, which would take down eleven healthy games to fix one
    laggard and throw away the rollouts in flight. The instance is found by port (netstat gives the owning pid),
    killed by that pid, and started again with the same `-aibridge-port`, so the worker that owns the port finds
    its game exactly where it expects it.

    Returns True once the port is listening again. An instance that never opened a port in the first place
    cannot be addressed this way -- there is nothing to look up -- and the caller has to fall back to a full
    stop and launch.
    """
    pid = listening_pids().get(port)
    if pid is not None:
        print(f"Killing the instance on port {port} (pid {pid})")
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        deadline = time.monotonic() + 30.0
        while port_open(port) and time.monotonic() < deadline:
            time.sleep(1)
    else:
        print(f"No process is listening on port {port}; starting one anyway")
    start_instance(port, width, height, job_workers)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            print(f"Port {port} is listening again")
            return True
        time.sleep(1)
    print(f"Port {port} did not come back within {timeout:.0f}s")
    return False


def launch(
    count: int, base_port: int, width: int, height: int, stagger: float, timeout: float, monitor: int | None, job_workers: int
) -> None:
    exe = game_dir() / EXE_NAME
    if not exe.exists():
        sys.exit(f"Game not found at {exe}. Set ULTRAKILL_DIR or mod/GamePaths.props.")
    cfg = game_dir() / "BepInEx" / "config" / "BepInEx.cfg"
    if count > DISK_LOG_LIMIT and disk_logging_enabled(cfg):
        sys.exit(
            f"--count {count} needs BepInEx's disk log off: copy {DISK_LOG_LIMIT + 1} onward cannot open a log "
            f"file and the plugin never loads. Set [Logging.Disk] Enabled = false in {cfg} (back it up first)."
        )

    stop_all()
    backup_display()

    for i in range(count):
        port = base_port + i
        print(f"Started instance {i} on port {port} (pid {start_instance(port, width, height, job_workers)})")
        time.sleep(stagger)

    ports = [base_port + i for i in range(count)]
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        verdict = startup_verdict(ports, listening_ports(), len(running_pids()), elapsed, timeout)
        if verdict == "ready":
            break
        if verdict == "no_processes":
            sys.exit(f"No {EXE_NAME} process is running {elapsed:.0f}s after launching {count}. "
                     "Check BepInEx/LogOutput.log.")
        if verdict == "timeout":
            sys.exit(f"Timed out waiting for ports {sorted(set(ports) - listening_ports())}")
        time.sleep(1)

    tile(running_pids(), width, height, monitor)
    print(f"All {count} instances ready on ports {ports[0]}-{ports[-1]}")


def status(base_port: int) -> None:
    pids = running_pids()
    sets = working_sets()
    print(f"{len(pids)} game process(es) running: {pids}")
    owners = listening_pids()
    for port in range(base_port, base_port + 16):
        pid = owners.get(port)
        if pid is None:
            continue
        mb = sets.get(pid)
        # A listening port is not a booted game: the plugin serves long before Addressables are ready, and a
        # copy stuck at ~56 MB answers every reset "unknown scene". Roughly 1 GB is what a booted copy sits at.
        note = "unknown size" if mb is None else f"{mb / (1024 * 1024):.0f} MB"
        print(f"  bridge listening on {port} (pid {pid}, {note})")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--count", type=int, default=5,
                          help="how many copies; more than 5 needs [Logging.Disk] Enabled = false in BepInEx.cfg")
    p_launch.add_argument("--base-port", type=int, default=47800)
    p_launch.add_argument("--width", type=int, default=368)
    p_launch.add_argument("--height", type=int, default=207)
    p_launch.add_argument("--stagger", type=float, default=4.0, help="seconds between launches")
    p_launch.add_argument("--timeout", type=float, default=180.0)
    p_launch.add_argument("--monitor", type=int, default=3, help="Windows display number to put the games on")
    p_launch.add_argument("--job-workers", type=int, default=3, help="Unity job worker threads per game")
    p_tile = sub.add_parser("tile")
    p_tile.add_argument("--monitor", type=int, default=3)
    p_tile.add_argument("--width", type=int, default=368)
    p_tile.add_argument("--height", type=int, default=207)
    p_status = sub.add_parser("status")
    p_status.add_argument("--base-port", type=int, default=47800)
    p_one = sub.add_parser("relaunch", help="restart ONE instance by port, leaving the others running")
    p_one.add_argument("--port", type=int, required=True)
    p_one.add_argument("--width", type=int, default=368)
    p_one.add_argument("--height", type=int, default=207)
    p_one.add_argument("--job-workers", type=int, default=3)
    sub.add_parser("stop")
    args = parser.parse_args()

    if args.cmd == "launch":
        launch(args.count, args.base_port, args.width, args.height, args.stagger, args.timeout, args.monitor, args.job_workers)
    elif args.cmd == "tile":
        tile(running_pids(), args.width, args.height, args.monitor)
    elif args.cmd == "relaunch":
        sys.exit(0 if relaunch_one(args.port, args.width, args.height, args.job_workers) else 1)
    elif args.cmd == "status":
        status(args.base_port)
    else:
        stop_all()
        restore_display()


if __name__ == "__main__":
    main()
