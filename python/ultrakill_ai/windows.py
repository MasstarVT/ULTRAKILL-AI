"""Windows display helpers shared by scripts/games.py and scripts/dashboard.py."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def monitor_work_area(number: int | None, quiet: bool = False) -> tuple[int, int, int, int]:
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
    if number is not None and not quiet:
        print(f"Monitor {number} not found, using the primary monitor")
    return next(area for area, primary in monitors.values() if primary)
