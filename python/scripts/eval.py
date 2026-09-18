"""Runs a trained model so you can watch it or measure it.

    python scripts/eval.py models/cybergrind_ppo/latest.zip --episodes 3 --realtime
    python scripts/eval.py models/campaign_ppo/best.zip --level "Level 0-1" --episodes 10 --record-times

Campaign evaluation always starts from a fresh level load with real deaths. It reads the first training game's
exploration counts (the policy was trained with them as inputs) but never writes them back, and never writes the
training runs' best-run files. `--record-times` adds the fastest completion to the repo-root times.md
(generation history, plus the leaderboard when it is a record).

Point it at the run whose action space the model has: a campaign checkpoint from before the look modes has 11
action dimensions, and its actions still decode (look mode 0, free look), but scripts/add_look_mode.py is what
migrates a run properly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import CAMPAIGN_LEVELS, ExplorationArchive, safe_name  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.times import TimeEntry, format_time, record_file  # noqa: E402

TIMES_MD = Path(__file__).resolve().parents[2] / "times.md"


def report_campaign(args: argparse.Namespace, level: str, model, results: list[tuple[float, dict]]) -> None:
    """Completion summary for a campaign eval, and the times.md entry for its fastest completion.

    `level` is read from the env, never from the config: `--record-times` writes a row only a FASTER time can
    ever replace, so recording it against the wrong level would be permanent.
    """
    completions = sum(1 for _, info in results if info.get("completed"))
    print(f"completed {completions}/{len(results)} fresh runs of {level}")
    timed = [info for _, info in results if info.get("completed") and info.get("level_seconds") is not None]
    if not timed:
        if args.record_times:
            print("no completed run, times.md unchanged")
        return
    best = min(timed, key=lambda info: info["level_seconds"])
    print(f"fastest {format_time(best['level_seconds'])} rank={best.get('rank') or '-'} kills={best['kills']}"
          f" style={best.get('style', 0)} restarts={best.get('restarts', '-')} deaths={best['deaths']}")
    if not args.record_times:
        return
    entry = TimeEntry(
        level=level,
        seconds=best["level_seconds"],
        rank=best.get("rank") or "",
        generation=f"{Path(args.model).parent.name}@{model.num_timesteps / 1e6:.2f}M",
        difficulty=best["difficulty"],
        date=time.strftime("%Y-%m-%d"),
        kills=best["kills"],
        deaths=best["deaths"],
        notes=f"{completions}/{len(results)} eval runs completed",
    )
    record_file(TIMES_MD, entry)
    print(f"recorded {format_time(entry.seconds)} ({entry.generation}) in {TIMES_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to a saved .zip")
    parser.add_argument("--algo", choices=["ppo", "rppo"], default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--realtime", action="store_true", help="normal speed with sound")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    parser.add_argument("--level", help='campaign scene, e.g. "Level 0-1" (default: the level in env_config.yaml next to the model)')
    parser.add_argument("--record-times", action="store_true", help="campaign: add the fastest completion to times.md")
    args = parser.parse_args()

    cfg_path = Path(args.model).parent / "env_config.yaml"
    cfg = EnvConfig.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else EnvConfig()
    if (args.level or args.record_times) and cfg.mode != "campaign":
        parser.error(f"--level and --record-times need a campaign model; the env config next to it gives mode {cfg.mode!r}")
    if args.level and args.level not in CAMPAIGN_LEVELS:
        parser.error(f"--level {args.level!r} is not a campaign scene name (Level 0-1 .. Level 9-2)")
    if cfg.mode == "campaign":
        # A curriculum config trains on a list; an eval runs exactly one level, or --record-times writes a row
        # for whichever level the sampler happened to draw. Clearing `levels` is what pins it.
        if args.level:
            cfg.level, cfg.levels = args.level, []
        elif cfg.levels:
            cfg.level, cfg.levels = cfg.levels[0], []
            print(f"the env config lists several levels; evaluating {cfg.level} (pass --level to choose another)")
    if args.realtime:
        cfg.unlimited_fps = False
        cfg.mute = False
        cfg.windowed = False
        cfg.render = True
    cfg.soft_death = False  # evaluate with real deaths
    # Belt and braces with `EnvConfig.RUN_ONLY_FIELDS`, which keeps these out of the file in the first place:
    # an eval must never be able to kill a game process, and it must never write into a live run's
    # attribution log, whatever an `env_config.yaml` written by an older train.py happens to contain.
    cfg.bridge_relaunch = False
    cfg.env_log_dir = ""
    if cfg.mode == "campaign":
        # Every episode is a fresh level load, so each one is a whole run with an official time. The exploration
        # archive and best-run files belong to the training games, so eval never saves either of them.
        cfg.fresh_start_prob = 1.0
        cfg.explore_dir = ""
        cfg.best_runs_dir = ""
        cfg.curriculum_path = ""  # never read, let alone perturb, a live run's curriculum

    if args.algo == "rppo":
        from sb3_contrib import RecurrentPPO as cls
    else:
        from stable_baselines3 import PPO as cls
    model = cls.load(args.model, device="cpu")

    env = UltrakillEnv(cfg)
    if cfg.mode == "campaign":
        # The last 9 campaign inputs are the game's visit counts, which in training come from thousands of earlier
        # episodes. A fresh archive would show the policy an all-unexplored map it never trained on, so eval reads the
        # counts the first training game (the config's base port) saved next to the model. explore_dir stays empty,
        # so the env never writes them back. A missing file gives an empty archive, and the count below shows it.
        archive_path = Path(args.model).parent / f"explore_{safe_name(env.level)}_{cfg.port}.npz"
        env.archive = ExplorationArchive.load(archive_path, cfg.cell_size)
        print(f"exploration counts: {len(env.archive.counts)} cells from {archive_path}")
    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            state, start = None, np.ones((1,), dtype=bool)
            total, steps = 0.0, 0
            while True:
                action, state = model.predict(obs, state=state, episode_start=start, deterministic=not args.stochastic)
                start = np.zeros((1,), dtype=bool)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            if cfg.mode == "campaign":
                # The difficulty the game actually read this run: the campaign block reports the override, if any.
                info = dict(info, difficulty=(env._raw.get("campaign") or {}).get("difficulty", cfg.difficulty))
                seconds = info.get("level_seconds")
                extra = (f" completed={info.get('completed', 0)} time={format_time(seconds) if seconds is not None else '-'}"
                         f" rank={info.get('rank') or '-'} style={info.get('style', 0)}"
                         f" restarts={info.get('restarts', '-')} deaths={info['deaths']}"
                         f" gates={info.get('gates_reached', 0)} hops={info.get('gate_hops_best')}"
                         f" wedged={info.get('wedged_steps', 0)}")
            else:
                extra = f" wave={info['wave']}"
            results.append((total, info))
            print(f"episode {ep}: reward={total:.1f} steps={steps} kills={info['kills']}{extra} end={info.get('end_reason')}")
    finally:
        env.close()

    print(f"mean reward {np.mean([r for r, _ in results]):.1f} over {len(results)} episodes")
    if cfg.mode == "campaign":
        report_campaign(args, env.level, model, results)


if __name__ == "__main__":
    main()
