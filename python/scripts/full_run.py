"""Plays the whole campaign with ONE SPECIALIST POLICY PER LEVEL, on one game, and times it.

`scripts/campaign_driver.py` trains a specialist per level and promotes each one to
`models/specialists/<level>.zip`. This is the other half: it walks `configs/specialists.yaml`'s order, loads
that level's specialist, plays the level from a fresh load, records the official time, and prints the table
plus the total. A level with no specialist yet is SKIPPED and reported -- the run is still useful, it just has
a hole in it, and the table says exactly where.

    python scripts/games.py launch --count 1 --monitor 1
    python scripts/full_run.py                       # sampled actions, the whole order
    python scripts/full_run.py --levels "Level 0-1" "Level 0-2" --episodes 2
    python scripts/games.py stop

**Actions are SAMPLED, not argmax** (since 2026-09-20, `eval.resolve_deterministic` has the measurement):
every specialist here is a campaign policy, trained and promoted on sampled actions with its action heads
near 7 nats of entropy, so argmax is a policy nobody measured -- 0-1's promoted specialist completed 0 of 5
argmax episodes against 19 of 20 sampled. `--deterministic` opts back in; `--stochastic` is kept as a no-op
alias so documented commands still run.

**One game, one port.** Never run this against a port a trainer is using: the bridge drops its current client
when a new one connects, so connecting would kill that worker's run.

Like `eval.py` this is read-only on the training runs. It reads each specialist's exploration archive (the
policy was trained with those counts as inputs) and never writes it back, never writes a `best_runs` file and
never touches a curriculum file. With `--record-times` each completed level is posted to the repo-root
`times.md` through `ultrakill_ai.times`, under the generation `specialists@<date>`; the leaderboard row only
moves when the time is actually faster, which is that helper's own rule.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultrakill_ai.procmem import cap_blas_threads  # noqa: E402

cap_blas_threads()  # before numpy: OpenBLAS reserves ~785 MB of commit for thread buffers at load

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from campaign_driver import (  # noqa: E402
    COMPLETE, load_plan, specialist_path, stage_config_path, stage_run_name)
from eval import resolve_deterministic  # noqa: E402  -- one rule for what `predict` gets, shared with eval.py
from ultrakill_ai.campaign import ExplorationArchive, safe_name  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.times import TimeEntry, actions_note, format_time, record_file  # noqa: E402

TIMES_MD = ROOT.parent / "times.md"


@dataclass
class LevelResult:
    """One level of a full run. `skipped` carries the reason when nothing was played."""

    level: str
    specialist: str | None = None
    completed: bool = False
    seconds: float | None = None       # the game's own official time, the only time worth posting
    rank: str = ""
    kills: int = 0
    deaths: int = 0
    steps: int = 0
    difficulty: int = 3
    end_reason: str | None = None
    skipped: str | None = None

    @property
    def played(self) -> bool:
        return self.skipped is None


# ---------------------------------------------------------------------------
# Pure: which levels can be played, what the run adds up to, how it reads
# ---------------------------------------------------------------------------


def find_specialist(models_dir: Path, level: str) -> Path | None:
    path = specialist_path(models_dir, level)
    return path if path.exists() else None


def specialist_mode(models_dir: Path, level: str) -> str:
    """Which KIND of stage promoted this level's specialist: "complete" or "speed" (the sidecar's `mode`).

    It decides which run's `env_config.yaml` and exploration archives the level is played with, because a
    speed stage is a run of its own (`spec_0-1_speed`). A sidecar written before stage kinds existed, or none
    at all, reads as "complete" -- which is what it was.
    """
    sidecar = specialist_path(models_dir, level).with_suffix(".json")
    try:
        return str(json.loads(sidecar.read_text(encoding="utf-8")).get("mode") or COMPLETE)
    except (OSError, ValueError, AttributeError):
        return COMPLETE


def run_chain(levels: Iterable[str], models_dir: Path,
              play: Callable[[str, Path], LevelResult]) -> list[LevelResult]:
    """Plays each level that has a specialist, in order, and records a skip for each one that does not.

    `play` is injected so the chaining logic -- the order, the skips, the accumulation -- is testable without a
    game: the offline test drives it with a fake level.
    """
    results: list[LevelResult] = []
    for level in levels:
        model = find_specialist(models_dir, level)
        if model is None:
            results.append(LevelResult(level=level, skipped="no specialist in %s" % (models_dir / "specialists")))
            continue
        result = play(level, model)
        result.specialist = model.as_posix()
        results.append(result)
    return results


def total_seconds(results: Iterable[LevelResult]) -> float:
    """The sum of the official times of the levels that were COMPLETED. A skipped or failed level adds nothing,
    which is why the table prints the completed count next to it: the total is only a full-game time when every
    level completed."""
    return sum(r.seconds or 0.0 for r in results if r.completed and r.seconds is not None)


def format_table(results: list[LevelResult]) -> str:
    rows = ["%-12s %-9s %-5s %-6s %-7s %-7s %s" % ("level", "time", "rank", "kills", "deaths", "steps", "note")]
    for r in results:
        if not r.played:
            rows.append("%-12s %-9s %-5s %-6s %-7s %-7s %s" % (r.level, "-", "-", "-", "-", "-", "SKIPPED: %s" % r.skipped))
            continue
        rows.append("%-12s %-9s %-5s %-6d %-7d %-7d %s" % (
            r.level,
            format_time(r.seconds) if r.seconds is not None else "-",
            r.rank or "-", r.kills, r.deaths, r.steps,
            "completed" if r.completed else "did not finish (%s)" % (r.end_reason or "?")))
    played = [r for r in results if r.played]
    completed = [r for r in played if r.completed]
    rows.append("")
    rows.append("%d/%d levels completed (%d skipped, no specialist); total of the completed times %s"
                % (len(completed), len(played), len(results) - len(played), format_time(total_seconds(results))))
    return "\n".join(rows)


def generation(date: str | None = None) -> str:
    return "specialists@%s" % (date or time.strftime("%Y-%m-%d"))


def post_results(times_md: Path, results: list[LevelResult], *, gen: str | None = None,
                 deterministic: bool = False, record=record_file) -> list[str]:
    """Posts every completed level's official time through `ultrakill_ai.times`. Returns one line per post.

    The helper itself decides whether the leaderboard row moves (only a faster time replaces one); the
    generation history takes every row either way, which is the point of recording a whole chained run.
    `deterministic` is what the chain actually played with, and the note says so -- the two action modes are
    not the same policy, so a row that did not name its mode would not be comparable with the one above it.
    """
    gen = gen or generation()
    posted = []
    for r in results:
        if not (r.completed and r.seconds is not None):
            continue
        record(times_md, TimeEntry(
            level=r.level, seconds=r.seconds, rank=r.rank or "", generation=gen,
            difficulty=r.difficulty, date=time.strftime("%Y-%m-%d"), kills=r.kills, deaths=r.deaths,
            notes="full run, one specialist per level (%s)" % actions_note(deterministic)))
        posted.append("%s %s rank %s" % (r.level, format_time(r.seconds), r.rank or "-"))
    return posted


# ---------------------------------------------------------------------------
# Playing one level
# ---------------------------------------------------------------------------


def eval_config(level: str, model: Path, plan, cwd: Path, *, port: int, mode: str = COMPLETE) -> EnvConfig:
    """The env one level is played with: the specialist's own training config, made eval-safe.

    Preference order, so a specialist is always played under the settings it was TRAINED under when they are
    still on disk: the stage's `env_config.yaml` (written by train.py), then the generated stage config, then
    the plan's own template. `mode` is the sidecar's -- a specialist promoted by a speed stage was trained in
    `spec_<level>_speed`, so that run's config is looked at first. Every one of them is then forced into eval
    shape the same way `eval.py` does it; `speed_bonus` rides along harmlessly, since nothing here reads the
    reward.
    """
    runs = [stage_run_name(level, mode)] + ([stage_run_name(level, COMPLETE)] if mode != COMPLETE else [])
    sources = [cwd / "models" / run / "env_config.yaml" for run in runs]
    sources += [cwd / stage_config_path(level, kind=kind)
                for kind in ([mode, COMPLETE] if mode != COMPLETE else [COMPLETE])]
    data: dict | None = None
    for path in sources:
        if not path.exists():
            continue
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data = loaded.get("env", loaded)  # env_config.yaml is flat; a stage config has an `env:` section
        break
    if data is None:
        data = dict(plan.env)
    cfg = EnvConfig.from_dict(dict(data))
    cfg.level, cfg.levels, cfg.curriculum_path = level, [], ""
    cfg.port = port
    cfg.soft_death = False          # real deaths
    cfg.fresh_start_prob = 1.0      # every episode is a whole level, so it has an official time
    cfg.explore_dir = ""            # the archives belong to training; read below, never written
    cfg.best_runs_dir = ""
    cfg.bridge_relaunch = False     # an eval may never restart a game process
    cfg.env_log_dir = ""
    return cfg


def load_archive(env: UltrakillEnv, level: str, cwd: Path, port: int, mode: str = COMPLETE) -> int:
    """Gives the policy the visit counts it was trained with. Read-only: `explore_dir` is empty, so nothing
    is ever written back. A speed-stage specialist reads its own run's archives, falling back to the complete
    stage's when that run never saved one for this port."""
    name = "explore_%s_%d.npz" % (safe_name(level), port)
    runs = [stage_run_name(level, mode)] + ([stage_run_name(level, COMPLETE)] if mode != COMPLETE else [])
    path = next((p for p in (cwd / "models" / run / name for run in runs) if p.exists()),
                cwd / "models" / runs[0] / name)
    env.archive = ExplorationArchive.load(path, env.cfg.cell_size)
    return len(env.archive.counts)


def play_level(level: str, env, model, *, deterministic: bool = False, max_steps: int = 200_000) -> LevelResult:
    """One episode of one level with one specialist. The env and the model are passed in, so the offline test
    drives this same function against a fake level and a stub policy.

    `deterministic` defaults to False -- SAMPLED actions, the way every specialist was trained and promoted.
    See the module docstring and `eval.resolve_deterministic` for what argmax measured instead."""
    obs, _ = env.reset()
    state, start = None, np.ones((1,), dtype=bool)
    info: dict = {}
    steps = 0
    while steps < max_steps:
        action, state = model.predict(obs, state=state, episode_start=start, deterministic=deterministic)
        start = np.zeros((1,), dtype=bool)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    difficulty = (getattr(env, "_raw", {}) or {}).get("campaign", {}).get("difficulty", env.cfg.difficulty)
    return LevelResult(
        level=level,
        completed=bool(info.get("completed")),
        seconds=info.get("level_seconds"),
        rank=str(info.get("rank") or ""),
        kills=int(info.get("kills") or 0),
        deaths=int(info.get("deaths") or 0),
        steps=steps,
        difficulty=int(difficulty if difficulty is not None else 3),
        end_reason=info.get("end_reason"),
    )


def best_of(results: list[LevelResult]) -> LevelResult | None:
    completed = [r for r in results if r.completed and r.seconds is not None]
    return min(completed, key=lambda r: r.seconds) if completed else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/specialists.yaml")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--levels", nargs="+", help="only these levels, in this order (default: the plan's order)")
    ap.add_argument("--port", type=int, default=47800, help="the ONE game's bridge port; never a trainer's")
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--deterministic", action="store_true",
                         help="take the most likely action (argmax) instead of sampling; NOT how these "
                              "specialists were trained or promoted (0-1: 0/5 against 19/20, 2026-09-20)")
    actions.add_argument("--stochastic", action="store_true",
                         help="no-op alias: sampling is the default here (kept so older commands still run)")
    ap.add_argument("--episodes", type=int, default=1, help="attempts per level; the fastest completion is kept")
    ap.add_argument("--record-times", action="store_true", help="post each completed level to times.md")
    ap.add_argument("--times", default=str(TIMES_MD))
    ap.add_argument("--json", help="write the results to this file as JSON")
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()

    from stable_baselines3 import PPO

    # Everything chained here is a campaign specialist, so this is "sample" unless --deterministic says argmax.
    deterministic = resolve_deterministic("campaign", deterministic=a.deterministic, stochastic=a.stochastic)
    print("actions: %s" % actions_note(deterministic))
    cwd = Path.cwd()
    plan = load_plan(a.plan)
    levels = a.levels or plan.order
    unknown = [level for level in levels if level not in plan.order]
    if unknown:
        ap.error("not in the plan's order: %s" % ", ".join(unknown))

    def play(level: str, model_path: Path) -> LevelResult:
        mode = specialist_mode(Path(a.models_dir), level)
        cfg = eval_config(level, model_path, plan, cwd, port=a.port, mode=mode)
        model = PPO.load(str(model_path), device="cpu")
        env = UltrakillEnv(cfg)
        attempts: list[LevelResult] = []
        try:
            cells = load_archive(env, level, cwd, a.port, mode)
            print("%s: %s (%s stage, %d exploration cells)"
                  % (level, model_path.name, mode, cells), flush=True)
            for _ in range(max(1, a.episodes)):
                result = play_level(level, env, model, deterministic=deterministic)
                print("  %s time=%s kills=%d deaths=%d end=%s"
                      % ("completed" if result.completed else "did not finish",
                         format_time(result.seconds) if result.seconds is not None else "-",
                         result.kills, result.deaths, result.end_reason), flush=True)
                attempts.append(result)
        finally:
            env.close()
        return best_of(attempts) or attempts[-1]

    results = run_chain(levels, Path(a.models_dir), play)
    print()
    print(format_table(results))
    if a.json:
        Path(a.json).write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2), encoding="utf-8")
    if a.record_times:
        posted = post_results(Path(a.times), results, deterministic=deterministic)
        for line in posted:
            print("posted", line)
        if not posted:
            print("nothing to post: no level completed")


if __name__ == "__main__":
    main()
