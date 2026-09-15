"""Launches and stops several ULTRAKILL instances for parallel training.

    python scripts/games.py launch --count 4     # ports 47800..47803, small windows tiled on monitor 3
    python scripts/games.py tile --monitor 3     # move running instances to a monitor
    python scripts/games.py status
    python scripts/games.py stop                 # closes them and restores your display settings

Each instance runs as a "training instance": it opens your settings read-only and never writes
settings or save data, so parallel copies can't corrupt anything. Unity saves the window size on exit,
so launch snapshots your display settings and stop puts them back.
"""

from __future__ import annotations

import argparse
import ctypes
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


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


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


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def monitor_work_area(number: int | None) -> tuple[int, int, int, int]:
    """Work area (left, top, right, bottom) of Windows display N (DISPLAYN); primary if not found."""
    user32 = ctypes.windll.user32
    monitors: dict[str, tuple[tuple[int, int, int, int], bool]] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    def callback(hmon, _hdc, _rect, _):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        w = info.rcWork
        monitors[info.szDevice] = ((w.left, w.top, w.right, w.bottom), bool(info.dwFlags & 1))
        return True

    user32.EnumDisplayMonitors(None, None, callback, 0)
    wanted = rf"\\.\DISPLAY{number}"
    if number is not None and wanted in monitors:
        return monitors[wanted][0]
    if number is not None:
        print(f"Monitor {number} not found, using the primary monitor")
    return next(area for area, primary in monitors.values() if primary)


def tile(pids: list[int], width: int, height: int, monitor: int | None) -> None:
    """Arranges the game windows in a grid on the chosen monitor so they don't overlap."""
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


def launch(
    count: int, base_port: int, width: int, height: int, stagger: float, timeout: float, monitor: int | None, job_workers: int
) -> None:
    exe = game_dir() / EXE_NAME
    if not exe.exists():
        sys.exit(f"Game not found at {exe}. Set ULTRAKILL_DIR or mod/GamePaths.props.")

    stop_all()
    backup_display()

    env = dict(os.environ, SteamAppId=STEAM_APP_ID, SteamGameId=STEAM_APP_ID)
    procs = []
    for i in range(count):
        port = base_port + i
        args = [
            str(exe), "-aibridge-port", str(port),
            "-screen-fullscreen", "0", "-screen-width", str(width), "-screen-height", str(height),
            # Fewer Unity worker threads per copy, so several games don't fight over the CPU.
            "-job-worker-count", str(job_workers),
        ]
        # Below-normal priority keeps your PC responsive while the games train.
        flags = subprocess.DETACHED_PROCESS | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        procs.append(subprocess.Popen(args, cwd=exe.parent, env=env, creationflags=flags))
        print(f"Started instance {i} on port {port} (pid {procs[-1].pid})")
        time.sleep(stagger)

    ports = [base_port + i for i in range(count)]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dead = [i for i, p in enumerate(procs) if p.poll() is not None]
        if dead:
            sys.exit(f"Instance(s) {dead} exited during startup. Check BepInEx/LogOutput.log.")
        if all(port_open(p) for p in ports):
            break
        time.sleep(1)
    else:
        sys.exit(f"Timed out waiting for ports {[p for p in ports if not port_open(p)]}")

    tile([p.pid for p in procs], width, height, monitor)
    print(f"All {count} instances ready on ports {ports[0]}-{ports[-1]}")


def status(base_port: int) -> None:
    pids = running_pids()
    print(f"{len(pids)} game process(es) running: {pids}")
    for port in range(base_port, base_port + 16):
        if port_open(port):
            print(f"  bridge listening on {port}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--count", type=int, default=4)
    p_launch.add_argument("--base-port", type=int, default=47800)
    p_launch.add_argument("--width", type=int, default=640)
    p_launch.add_argument("--height", type=int, default=360)
    p_launch.add_argument("--stagger", type=float, default=4.0, help="seconds between launches")
    p_launch.add_argument("--timeout", type=float, default=180.0)
    p_launch.add_argument("--monitor", type=int, default=3, help="Windows display number to put the games on")
    p_launch.add_argument("--job-workers", type=int, default=3, help="Unity job worker threads per game")
    p_tile = sub.add_parser("tile")
    p_tile.add_argument("--monitor", type=int, default=3)
    p_tile.add_argument("--width", type=int, default=640)
    p_tile.add_argument("--height", type=int, default=360)
    p_status = sub.add_parser("status")
    p_status.add_argument("--base-port", type=int, default=47800)
    sub.add_parser("stop")
    args = parser.parse_args()

    if args.cmd == "launch":
        launch(args.count, args.base_port, args.width, args.height, args.stagger, args.timeout, args.monitor, args.job_workers)
    elif args.cmd == "tile":
        tile(running_pids(), args.width, args.height, args.monitor)
    elif args.cmd == "status":
        status(args.base_port)
    else:
        stop_all()
        restore_display()


if __name__ == "__main__":
    main()
