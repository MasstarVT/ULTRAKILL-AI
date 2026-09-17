"""Live training dashboard (Tkinter, no extra dependencies).

    python scripts/dashboard.py                      # newest runs/*/status.json
    python scripts/dashboard.py --run cybergrind_ppo
    python scripts/dashboard.py --file path/to/status.json

Reads the status file that train.py writes through ultrakill_ai/progress.py. It never talks to the
training process or the games, so it is safe to open and close at any time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
REFRESH_MS = 2000
STALE_AFTER_S = 60
STALLED_GAME_S = 120

BG = "#14161b"
PANEL = "#1d2027"
PANEL_EDGE = "#2a2e37"
FG = "#e6e8ee"
MUTED = "#8a90a0"
GRID = "#2c3039"
GREEN = "#3ecf6e"
ORANGE = "#f0a03c"
BLUE = "#4c9dff"
RED = "#ff5d5d"
PURPLE = "#b58cff"
YELLOW = "#e8c547"

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 10)
FONT_TITLE = ("Segoe UI Semibold", 15)
FONT_BIG = ("Segoe UI Semibold", 16)
FONT_MONO = ("Consolas", 9)


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------

def num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def fmt_int(value) -> str:
    v = num(value)
    return "—" if v is None else f"{int(round(v)):,}"


def fmt_float(value, digits: int = 2) -> str:
    v = num(value)
    if v is None:
        return "—"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    return f"{v:,.{digits}f}"


def fmt_compact(value) -> str:
    v = num(value)
    if v is None:
        return "—"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:.1f}{suffix}"
    return f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"


def fmt_duration(seconds) -> str:
    s = num(seconds)
    if s is None or s < 0:
        return "—"
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def fmt_time(seconds) -> str:
    """Official level time as mm:ss.mmm, the format times.md uses."""
    s = num(seconds)
    if s is None:
        return "—"
    try:
        from ultrakill_ai.times import format_time
    except ImportError:  # a checkout from before times.py: the same whole-millisecond format
        ms = round(max(0.0, s) * 1000)
        return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"
    return format_time(s)


def fmt_pct(value) -> str:
    v = num(value)
    return "—" if v is None else f"{v * 100:.0f}%"


def short_level(scene) -> str:
    """"Level 0-1" -> "0-1", the form times.md and this panel use."""
    if not isinstance(scene, str):
        return "?"
    try:
        from ultrakill_ai.times import short_level as shorten
    except ImportError:  # a checkout from before times.py
        return scene.removeprefix("Level ")
    return shorten(scene)


def level_lines(campaign: dict) -> list[str]:
    """One row per unlocked level of a multi-level run, or nothing at all on a single-level run.

    R6 of the design spec: once two levels of different size run together the pooled `mean_100` numbers are
    cross-level means that say nothing, so this table is what a multi-level run is judged on. `cp` is
    checkpoints per level load and `w` the share of fresh draws the curriculum is giving the level.
    """
    levels = campaign.get("levels")
    if not isinstance(levels, dict):
        return []
    rows = ["  levels"]
    for name, row in levels.items():
        if not isinstance(row, dict) or not row.get("unlocked"):
            continue
        rows.append(f"    {short_level(name):<4} fresh {fmt_pct(row.get('fresh_completion_rate'))}"
                    f" ({fmt_int(row.get('fresh_window'))})  best {fmt_time(row.get('best_time'))}"
                    f"  cp {fmt_float(row.get('checkpoints_level'), 1)}  w {fmt_float(row.get('weight'))}")
    return rows if len(rows) > 1 else []


def campaign_lines(campaign: dict, mean: dict, parts: dict | None = None, best: dict | None = None,
                   fresh: dict | None = None, ppo: dict | None = None) -> list[str]:
    """Campaign panel rows: completion and official times, per-episode means over the last 100, then the largest reward parts.

    `gates/load` shows the all-episode mean AND the fresh-start mean, because a respawn episode inherits the
    count from its level load: reading the all-episode number alone is how the pre-fix run's "peak by depth"
    was misread. The look-mode shares and the per-dimension entropy sit together, since a look mode leaves the
    yaw and pitch heads causally inert and only the entropy bonus acts on them on those steps.

    On a multi-level run the headline is labelled a SCORE, not a percentage: `fresh_completion_rate` is then a
    shrunk sum over the unlocked levels and can exceed 1.0. The real rates are in the per-level rows below it.
    """
    best, fresh, ppo = best or {}, fresh or {}, ppo or {}
    exit_dist = fmt_float(mean.get("exit_dist_min"), 0)
    unlocked = [row for row in (campaign.get("levels") or {}).values() if isinstance(row, dict) and row.get("unlocked")]
    if unlocked:
        headline = ("fresh score     ", f"{fmt_float(campaign.get('fresh_completion_rate'))} / {len(unlocked)} levels")
    else:
        headline = ("fresh completed ", f"{fmt_pct(campaign.get('fresh_completion_rate'))} of last {fmt_int(campaign.get('fresh_window'))}")
    rows = (
        headline,
        ("all completed   ", fmt_pct(mean.get("completed"))),  # fresh starts and checkpoint respawns alike
        ("best time       ", fmt_time(campaign.get("best_time"))),
        ("median time     ", fmt_time(campaign.get("median_time_50"))),
        ("gates/load      ", f"{fmt_float(mean.get('gates_reached'), 1)} fresh {fmt_float(fresh.get('gates_reached'), 1)}"
                             f" best {fmt_int(best.get('best_gates_reached'))} hops {fmt_int(best.get('best_gate_hops'))}"),
        ("checkpoints/load", fmt_float(mean.get("checkpoints_level"), 1)),
        # The two mechanisms of the ladder-patience/exit-guard spec. `parked/ep` above ~0 says the ladder is
        # collapsed on the level being played (expected on 0-3, 1-1, 1-2, 2-3, 4-3, 8-1, not on 0-1);
        # `exit banished` above 0 says a CheckPoint clone moved the reported FinalPit and the guard caught it.
        ("parked/ep       ", f"{fmt_float(mean.get('targets_parked'), 2)}"
                             f"  exit banished {fmt_pct(mean.get('exit_banished'))}"),
        ("wedged/ep       ", fmt_float(mean.get("wedged_steps"), 0)),
        ("new cells/ep    ", fmt_float(mean.get("cells_new"), 0)),
        ("deaths/ep       ", fmt_float(mean.get("deaths"))),
        ("closest to exit ", exit_dist if exit_dist == "—" else f"{exit_dist}m"),
        ("look free/gate  ", f"{fmt_pct(mean.get('look_free_frac'))}/{fmt_pct(mean.get('look_gate_frac'))}"
                             f"  ent y/p/m {fmt_compact(ppo.get('entropy_yaw'))}/{fmt_compact(ppo.get('entropy_pitch'))}"
                             f"/{fmt_compact(ppo.get('entropy_look_mode'))}"),
    )
    lines = [f"  {label} {value}" for label, value in rows]
    lines.extend(level_lines(campaign))
    # The four largest reward parts by size, so a term that dominates the return shows up at a glance.
    values = [(str(name), v) for name, v in ((name, num(value)) for name, value in (parts or {}).items()) if v is not None]
    if values:
        lines.append("  reward parts/ep")
        lines.extend(f"    {name[:14]:<14} {v:+.1f}" for name, v in sorted(values, key=lambda nv: -abs(nv[1]))[:4])
    return lines


# ---------------------------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------------------------

def find_status_file(runs_dir: Path, run: str | None) -> Path | None:
    if run:
        return runs_dir / run / "status.json"
    candidates = []
    for p in runs_dir.glob("*/status.json"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            pass
    return max(candidates)[1] if candidates else None


def read_status(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, "No runs/*/status.json yet. Start train.py."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"Waiting for {path}"
    except (OSError, ValueError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not isinstance(data, dict):
        return None, f"Unexpected content in {path}"
    return data, None


# ---------------------------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------------------------

def panel(parent, **grid) -> tk.Frame:
    f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=PANEL_EDGE)
    f.grid(**grid)
    return f


class LineChart(tk.Canvas):
    """Minimal multi-series line chart. Series: list of (label, color, [(x, y), ...])."""

    def __init__(self, parent, title: str, x_label: str = "steps"):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=PANEL_EDGE, height=140, width=200)
        self.title = title
        self.x_label = x_label
        self.series: list[tuple[str, str, list[tuple[float, float]]]] = []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_series(self, series) -> None:
        self.series = series
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 50:
            return
        self.create_text(10, 8, text=self.title, anchor="nw", fill=FG, font=FONT_BOLD)
        lx = w - 10
        for label, color, pts in reversed(self.series):
            latest = f"{label} {fmt_float(pts[-1][1])}" if pts else label
            t = self.create_text(lx, 9, text=latest, anchor="ne", fill=color, font=FONT_SMALL)
            lx = self.bbox(t)[0] - 12

        left, right, top, bottom = 48, w - 12, 30, h - 22
        if right - left < 20 or bottom - top < 20:
            return
        pts_all = [p for _, _, pts in self.series for p in pts]
        if not pts_all:
            self.create_text((left + right) / 2, (top + bottom) / 2, text="no data yet", fill=MUTED, font=FONT_SMALL)
            return
        xs = [p[0] for p in pts_all]
        ys = [p[1] for p in pts_all]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 == x0:
            pad = abs(x0) * 0.05 or 1.0
            x0, x1 = x0 - pad, x1 + pad
        if y1 == y0:
            pad = abs(y0) * 0.1 or 1.0
            y0, y1 = y0 - pad, y1 + pad
        else:
            pad = (y1 - y0) * 0.05
            y0, y1 = y0 - pad, y1 + pad

        def sx(x):
            return left + (x - x0) / (x1 - x0) * (right - left)

        def sy(y):
            return bottom - (y - y0) / (y1 - y0) * (bottom - top)

        for frac in (0.0, 0.5, 1.0):
            yy = bottom - frac * (bottom - top)
            self.create_line(left, yy, right, yy, fill=GRID)
        self.create_text(left - 6, top, text=fmt_compact(y1), anchor="e", fill=MUTED, font=FONT_SMALL)
        self.create_text(left - 6, bottom, text=fmt_compact(y0), anchor="e", fill=MUTED, font=FONT_SMALL)
        self.create_text(left, bottom + 4, text=fmt_compact(x0), anchor="nw", fill=MUTED, font=FONT_SMALL)
        self.create_text(right, bottom + 4, text=f"{fmt_compact(x1)} {self.x_label}", anchor="ne", fill=MUTED, font=FONT_SMALL)

        for _label, color, pts in self.series:
            if len(pts) == 1:
                cx, cy = sx(pts[0][0]), sy(pts[0][1])
                self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline="")
            elif len(pts) > 1:
                coords = []
                for x, y in pts:
                    coords.extend((sx(x), sy(y)))
                self.create_line(*coords, fill=color, width=2)


class Dashboard:
    def __init__(self, root: tk.Tk, runs_dir: Path, run: str | None, file: Path | None):
        self.root = root
        self.runs_dir = runs_dir
        self.run = run
        self.file = file
        self.path: Path | None = None
        self.env_rows: list[list[tk.Label]] = []

        root.title("ULTRAKILL AI — Training")
        root.configure(bg=BG)
        root.geometry("920x680")
        root.minsize(760, 640)

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1)

        # Header
        header = tk.Frame(outer, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        self.run_label = tk.Label(header, text="—", bg=BG, fg=FG, font=FONT_TITLE)
        self.run_label.pack(side="left")
        self.badge = tk.Label(header, text="…", bg=MUTED, fg=BG, font=FONT_BOLD, padx=8, pady=1)
        self.badge.pack(side="left", padx=10)
        self.games_label = tk.Label(header, text="", bg=BG, fg=MUTED, font=FONT)
        self.games_label.pack(side="left")
        self.updated_label = tk.Label(header, text="", bg=BG, fg=MUTED, font=FONT_SMALL)
        self.updated_label.pack(side="right")

        # Progress
        prog = panel(outer, row=1, column=0, sticky="ew", pady=(8, 0))
        prog.columnconfigure(0, weight=1)
        self.progress_text = tk.Label(prog, text="", bg=PANEL, fg=FG, font=FONT_BIG, anchor="w")
        self.progress_text.grid(row=0, column=0, sticky="w", padx=10, pady=(6, 2))
        self.bar = tk.Canvas(prog, height=16, bg=PANEL, highlightthickness=0)
        self.bar.grid(row=1, column=0, sticky="ew", padx=10)
        self.bar.bind("<Configure>", lambda _e: self._draw_bar())
        self.bar_fraction = 0.0
        self.bar_color = GREEN
        self.progress_sub = tk.Label(prog, text="", bg=PANEL, fg=MUTED, font=FONT, anchor="w")
        self.progress_sub.grid(row=2, column=0, sticky="w", padx=10, pady=(3, 6))

        # Stat tiles
        tiles = tk.Frame(outer, bg=BG)
        tiles.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.tiles: dict[str, tk.Label] = {}
        for i, (key, title) in enumerate((
            ("episodes", "Episodes"), ("mean_reward", "Mean reward"), ("best_reward", "Best reward"),
            ("mean_kills", "Mean kills"), ("kills_per_min", "Kills/min"), ("mean_deaths", "Deaths/ep"), ("mean_wave", "Mean wave"), ("best_wave", "Best wave"),
        )):
            tiles.columnconfigure(i, weight=1, uniform="tile")
            t = panel(tiles, row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            tk.Label(t, text=title, bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(anchor="w", padx=8, pady=(5, 0))
            value = tk.Label(t, text="—", bg=PANEL, fg=FG, font=FONT_BIG)
            value.pack(anchor="w", padx=8, pady=(0, 5))
            self.tiles[key] = value
        tk.Label(outer, text="means over the last 100 episodes", bg=BG, fg=MUTED, font=FONT_SMALL).grid(row=3, column=0, sticky="e")

        # Charts + games table
        middle = tk.Frame(outer, bg=BG)
        middle.grid(row=4, column=0, sticky="nsew", pady=(2, 0))
        outer.rowconfigure(4, weight=1)
        middle.columnconfigure(0, weight=1, uniform="mid")
        middle.columnconfigure(1, weight=1, uniform="mid")
        middle.rowconfigure(0, weight=1, uniform="midrow")
        middle.rowconfigure(1, weight=1, uniform="midrow")
        self.chart_reward = LineChart(middle, "Mean reward (100 ep)")
        self.chart_reward.grid(row=0, column=0, sticky="nsew", padx=(0, 3), pady=(0, 3))
        self.chart_kw = LineChart(middle, "Kills/min & wave (100 ep)")
        self.chart_kw.grid(row=0, column=1, sticky="nsew", padx=(3, 0), pady=(0, 3))
        self.chart_speed = LineChart(middle, "Steps/s", x_label="elapsed")
        self.chart_speed.grid(row=1, column=0, sticky="nsew", padx=(0, 3), pady=(3, 0))

        games = panel(middle, row=1, column=1, sticky="nsew", padx=(3, 0), pady=(3, 0))
        tk.Label(games, text="Games", bg=PANEL, fg=FG, font=FONT_BOLD).pack(anchor="w", padx=10, pady=(6, 2))
        self.games_table = tk.Frame(games, bg=PANEL)
        self.games_table.pack(fill="both", expand=True, padx=10)
        self.games_heads: list[tk.Label] = []
        for c, head in enumerate(("#", "last reward", "kills", "wave", "last episode")):
            self.games_table.columnconfigure(c, weight=1 if c else 0)
            head_label = tk.Label(self.games_table, text=head, bg=PANEL, fg=MUTED, font=FONT_SMALL, anchor="w")
            head_label.grid(row=0, column=c, sticky="w", padx=(0, 8))
            self.games_heads.append(head_label)

        # Bottom text
        bottom = tk.Frame(outer, bg=BG)
        bottom.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1, minsize=380)
        bottom.columnconfigure(2, weight=1, minsize=260)
        self.end_reasons = tk.Label(bottom, text="", bg=BG, fg=MUTED, font=FONT_MONO, anchor="w", justify="left")
        self.end_reasons.grid(row=0, column=0, sticky="w")
        self.ppo = tk.Label(bottom, text="", bg=BG, fg=MUTED, font=FONT_MONO, anchor="w", justify="left")
        self.ppo.grid(row=0, column=1, sticky="w")
        self.behaviour = tk.Label(bottom, text="", bg=BG, fg=MUTED, font=FONT_MONO, anchor="w", justify="left")
        self.behaviour.grid(row=0, column=2, sticky="w")
        self.message = tk.Label(outer, text="", bg=BG, fg=ORANGE, font=FONT_SMALL, anchor="w")
        self.message.grid(row=6, column=0, sticky="ew")

    # -- refresh --------------------------------------------------------------------------------

    def refresh(self) -> None:
        self.path = self.file or find_status_file(self.runs_dir, self.run)
        data, error = read_status(self.path)
        if data is None:
            self.message.config(text=error or "")
            if self.run_label.cget("text") == "—":
                self.render({})
            return
        self.message.config(text="")
        self.render(data)

    def schedule(self) -> None:
        try:
            self.refresh()
        except Exception:  # keep the window alive on odd data
            traceback.print_exc()
            self.message.config(text="render error, see console")
        self.root.after(REFRESH_MS, self.schedule)

    def render(self, d: dict) -> None:
        now = time.time()
        get = d.get
        mean = get("mean_100") if isinstance(get("mean_100"), dict) else {}
        campaign = get("campaign") if isinstance(get("campaign"), dict) else None

        # Header
        self.run_label.config(text=str(get("run_name") or (self.path.parent.name if self.path else "—")))
        state = str(get("state") or "unknown").lower()
        updated = num(get("updated_at"))
        age = now - updated if updated else None
        if state == "running" and age is not None and age > STALE_AFTER_S:
            badge, color = "STALE", ORANGE
        elif state == "running":
            badge, color = "RUNNING", GREEN
        elif state == "finished":
            badge, color = "FINISHED", BLUE
        elif state == "stopped":
            badge, color = "STOPPED", ORANGE
        else:
            badge, color = state.upper(), MUTED
        self.badge.config(text=badge, bg=color)
        n_envs = get("num_envs")
        self.games_label.config(text=f"{fmt_int(n_envs)} game{'s' if num(n_envs) != 1 else ''}" if n_envs is not None else "")
        self.updated_label.config(text=f"updated {fmt_duration(age)} ago" if age is not None else "")

        # Progress
        steps, target = num(get("timesteps")), num(get("target_timesteps"))
        frac = steps / target if steps is not None and target else 0.0
        pct = f" ({frac * 100:.1f}%)" if target else ""
        self.progress_text.config(text=f"{fmt_int(steps)} / {fmt_int(target)} steps{pct}")
        self.bar_fraction = max(0.0, min(1.0, frac))
        self.bar_color = color if badge != "RUNNING" else GREEN
        self._draw_bar()
        elapsed = num(get("elapsed_s"))
        if state == "running" and elapsed is not None and age is not None and age <= STALE_AFTER_S:
            elapsed += age
        if badge == "RUNNING":
            speed_text = f"ETA {fmt_duration(get('eta_s'))}    {fmt_float(get('steps_per_s'), 1)} steps/s"
        else:
            done = (steps or 0) - (num(get("start_timesteps")) or 0)
            avg = done / elapsed if elapsed and elapsed > 1 and done > 0 else None
            speed_text = f"avg {fmt_float(avg, 1)} steps/s"
        self.progress_sub.config(text=f"elapsed {fmt_duration(elapsed)}    {speed_text}")

        # Tiles
        self.tiles["episodes"].config(text=fmt_int(get("episodes")))
        self.tiles["mean_reward"].config(text=fmt_float(mean.get("reward")))
        self.tiles["best_reward"].config(text=fmt_float(get("best_reward")))
        self.tiles["mean_kills"].config(text=fmt_float(mean.get("kills")))
        self.tiles["kills_per_min"].config(text=fmt_float(mean.get("kills_per_min")))
        self.tiles["mean_wave"].config(text=fmt_float(mean.get("wave")))
        self.tiles["best_wave"].config(text=fmt_int(get("best_wave")))
        self.tiles["mean_deaths"].config(text=fmt_float(mean.get("deaths")))

        # Charts
        history = [p for p in (get("history") or []) if isinstance(p, dict)]

        def series(key, x_key="timesteps", x_offset=0.0):
            # x must increase, or the line doubles back and the chart looks scrambled. A status.json written
            # before the ProgressCallback fix can still hold a rewind, from a training restart that resumed an
            # older checkpoint, so clean it at draw time too: this is a viewer, and it must render whatever is
            # already on disk. Dropping the superseded points keeps the run that continued.
            out = []
            for p in history:
                x, y = num(p.get(x_key)), num(p.get(key))
                if x is None or y is None:
                    continue
                while out and out[-1][0] >= x:
                    out.pop()
                out.append((x, y))
            return [(x - x_offset, y) for x, y in out]

        self.chart_reward.set_series([("reward", GREEN, series("mean_reward_100"))])
        if campaign is not None:
            self.chart_kw.title = "Fresh completion % & gates"
            fresh_pct = [(x, y * 100.0) for x, y in series("completion_rate_fresh_50")]
            self.chart_kw.set_series([("fresh %", PURPLE, fresh_pct), ("gates", YELLOW, series("mean_gates_reached_100"))])
        else:
            self.chart_kw.title = "Kills/min & wave (100 ep)"
            self.chart_kw.set_series([("kills/min", RED, series("mean_kills_per_min_100")), ("wave", YELLOW, series("mean_wave_100"))])
        started = num(get("started_at"))
        if started is None and history:
            started = num(history[0].get("wall_time")) or 0.0
        speed = [(x / 60.0, y) for x, y in series("steps_per_s", "wall_time", started or 0.0)]
        self.chart_speed.x_label = "min"
        self.chart_speed.set_series([("steps/s", BLUE, speed)])

        self.games_heads[3].config(text="checkpoints" if campaign is not None else "wave")
        self._render_games(get("envs") or [], age if state == "running" else None)

        # End reasons + PPO
        shooting = [
            f"  {label} {mean[key] * 100:.0f}%"
            for key, label in (("firing_frac", "firing          "), ("on_target_frac", "enemy in crosshair"), ("enemy_visible_frac", "enemy visible   "), ("enemy_close_frac", "enemy within 5m "))
            if isinstance(mean.get(key), (int, float))
        ]
        for key, label, unit in (("enemy_angle_mean", "angle off       ", "deg"), ("enemy_yaw_angle_mean", "yaw off         ", "deg"), ("enemy_pitch_err_mean", "pitch error     ", "deg"), ("pitch_abs_mean", "camera pitch    ", "deg"), ("enemy_elev_mean", "enemy elevation ", "deg"),
                                 ("enemy_dist_mean", "enemy distance  ", "m"), ("yaw_per_step_mean", "turn per step   ", "deg")):
            if isinstance(mean.get(key), (int, float)):
                shooting.append(f"  {label} {mean[key]:.0f}{unit}")
        for key, label in (("yaw_track", "yaw tracking    "), ("pitch_track", "pitch tracking  ")):
            if isinstance(mean.get(key), (int, float)):
                shooting.append(f"  {label} {mean[key]:+.2f}")
        if campaign is not None:
            parts = get("reward_parts_mean_100") if isinstance(get("reward_parts_mean_100"), dict) else {}
            fresh = get("mean_fresh_100") if isinstance(get("mean_fresh_100"), dict) else {}
            bests = {k: get(k) for k in ("best_gates_reached", "best_gate_hops")}
            ppo_now = get("ppo") if isinstance(get("ppo"), dict) else {}
            self.behaviour.config(text="Campaign (last 100)\n" + "\n".join(campaign_lines(campaign, mean, parts, bests, fresh, ppo_now)))
        else:
            self.behaviour.config(text="Shooting (last 100)\n" + "\n".join(shooting) if shooting else "")
        reasons = get("end_reasons_100") if isinstance(get("end_reasons_100"), dict) else {}
        total = sum(v for v in (num(x) for x in reasons.values()) if v) or 0
        items = []
        for name, count in sorted(reasons.items(), key=lambda kv: -(num(kv[1]) or 0))[:6]:
            c = num(count) or 0
            items.append(f"{str(name)[:16]} {c / total * 100 if total else 0:.0f}%")
        rows = [items[:3], items[3:]] if items else [["—"]]
        self.end_reasons.config(text="End reasons (last 100)\n" + "\n".join("  " + "   ".join(r) for r in rows if r))

        ppo = get("ppo") if isinstance(get("ppo"), dict) else {}
        cells = [f"{label} {fmt_compact(ppo.get(key))}" for key, label in (
            ("entropy_loss", "entropy"), ("approx_kl", "approx_kl"), ("value_loss", "value_loss"),
            ("explained_variance", "explained_var"), ("clip_fraction", "clip_frac"))]
        self.ppo.config(text="PPO (last update)\n  " + "   ".join(cells[:3]) + "\n  " + "   ".join(cells[3:]))

    def _draw_bar(self) -> None:
        c = self.bar
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        c.create_rectangle(0, 0, w, h, fill=GRID, outline="")
        if self.bar_fraction > 0:
            c.create_rectangle(0, 0, max(2, w * self.bar_fraction), h, fill=self.bar_color, outline="")

    def _render_games(self, envs: list, extra_age: float | None) -> None:
        envs = [e for e in envs if isinstance(e, dict)][:32]
        while len(self.env_rows) < len(envs):
            r = len(self.env_rows) + 1
            row = [tk.Label(self.games_table, bg=PANEL, fg=FG, font=FONT_SMALL, anchor="w") for _ in range(5)]
            for c, lbl in enumerate(row):
                lbl.grid(row=r, column=c, sticky="w", padx=(0, 8))
            self.env_rows.append(row)
        while len(self.env_rows) > len(envs):
            for lbl in self.env_rows.pop():
                lbl.destroy()
        for row, e in zip(self.env_rows, envs):
            age = num(e.get("age_s"))
            if age is not None and extra_age is not None and extra_age <= STALE_AFTER_S:
                age += extra_age  # time since the status was written
            stalled = age is not None and age > STALLED_GAME_S
            episodes = num(e.get("episodes")) or 0
            if episodes:
                last = f"{fmt_duration(age)} ago"
            else:
                last = f"none yet ({fmt_duration(age)})"
            # Campaign games report the checkpoints of the level load instead of a wave (their wave stays 0).
            progress = e.get("checkpoints_level") if e.get("checkpoints_level") is not None else e.get("wave")
            # On a curriculum run each game picks its own level at every fresh load, so the row says which.
            index = fmt_int(e.get("env"))
            if isinstance(e.get("level"), str):
                index = f"{index} {short_level(e['level'])}"
            values = (index, fmt_float(e.get("reward")), fmt_int(e.get("kills")), fmt_int(progress),
                      last + ("  ⚠ stalled?" if stalled else ""))
            for c, (lbl, text) in enumerate(zip(row, values)):
                lbl.config(text=text, fg=(ORANGE if stalled else FG) if c in (0, 4) else FG)


def place_window(root: tk.Tk, monitor: int | None, reserve_top: int) -> None:
    """Fills the chosen monitor below the game windows; falls back to the default size on any failure."""
    if monitor is None:
        return
    try:
        from ultrakill_ai.windows import monitor_work_area

        left, top, right, bottom = monitor_work_area(monitor, quiet=True)
    except Exception:
        return
    width = max(760, right - left)
    height = max(640, bottom - top - reserve_top)
    root.geometry(f"{width}x{height}+{left}+{top + reserve_top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--run", help="run name under runs/ (default: most recently updated)")
    parser.add_argument("--file", type=Path, help="explicit path to a status.json")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--monitor", type=int, default=3, help="Windows display to open on (default 3, where the games run)")
    parser.add_argument("--reserve-top", type=int, default=250, help="pixels left free at the top for the game windows")
    parser.add_argument("--smoke-test", action="store_true", help="render once, then exit (exit code 1 on errors)")
    args = parser.parse_args()

    root = tk.Tk()
    errors: list[str] = []

    def report(exc, val, tb):
        errors.append("".join(traceback.format_exception(exc, val, tb)))
        sys.stderr.write(errors[-1])
        if args.smoke_test:
            root.destroy()

    root.report_callback_exception = report
    dash = Dashboard(root, args.runs_dir, args.run, args.file)
    place_window(root, args.monitor, args.reserve_top)

    if args.smoke_test:
        root.update_idletasks()
        dash.refresh()  # raises straight into this frame on bad rendering
        root.update()
        root.after(1500, root.destroy)
        root.mainloop()
        if errors:
            sys.exit(1)
        print(f"smoke test ok ({dash.path})")
        return

    dash.schedule()
    root.mainloop()


if __name__ == "__main__":
    main()
