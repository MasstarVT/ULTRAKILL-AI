"""Watches a human play a campaign level and reports whether the bridge sees the level complete.

    python scripts/watch_completion.py          # then play the level yourself, to the end

Read-only: it polls the observation and never takes control, never configures the game and never writes a file,
so difficulty, resolution, audio and saves are all yours. The mod's AI-only behaviour (difficulty override,
blocked restarts, muting) applies only while the AI has control, which this never takes.

Why this exists. `campaign_check.py` check 5 FAILs on 0-1 and 1-1 because a fresh load reports the exit
`active: false`, and nothing had ever confirmed that real play switches that room on. Two automated attempts
could not answer it (2026-09-16): teleport hops walk the level's geometry while every room stays switched off, so
no trigger can fire; and scripted movement cannot leave 0-1's starting room, which a probe showed is sealed --
walls 3.5-6.6 m in all 16 ray directions, `level_started` false and no weapons. A human playthrough is the
shortest path to the answer.

It prints every change to the three signals that matter and exits when the level reports complete:

  level_started   the run has begun (campaign.level_started)
  exit.active     the room holding the FinalPit has been switched on -- the step check 5 could never reach
  level_over      stats.level_complete or campaign.level_over: what env.py grades a completion on

A PASS here means the completion path the campaign reward depends on works end to end, and training toward it is
sound. A FAIL means no reward change can ever move the completion rate, and the next work is mod-side.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.protocol import BridgeClient  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a human playthrough for the level-complete signal.")
    p.add_argument("--port", type=int, default=47800)
    p.add_argument("--interval", type=float, default=0.5, help="seconds between polls")
    p.add_argument("--minutes", type=float, default=30.0, help="give up after this long")
    return p.parse_args(argv)


def snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    camp = raw.get("campaign") or {}
    ext = camp.get("exit") or {}
    return {
        "scene": raw.get("scene"),
        "level_started": camp.get("level_started"),
        "exit_active": ext.get("active"),
        "level_over": bool(raw.get("stats", {}).get("level_complete") or camp.get("level_over")),
        "checkpoints": sum(1 for c in (camp.get("checkpoints") or []) if c.get("activated")),
        "cleared_arenas": len(camp.get("cleared_arenas") or []),
        "unlocked_doors": len(camp.get("unlocked_doors") or []),
        "seconds": camp.get("seconds"),
        "kills": raw.get("stats", {}).get("kills"),
    }


def main() -> None:
    args = parse_args()
    client = BridgeClient(port=args.port)
    print(f"waiting for a game on port {args.port} (launch ULTRAKILL now if it is not up)...", flush=True)
    client.connect(retry_seconds=600.0)  # long enough to start the game by hand
    print(f"connected, read-only. Play the level to the end; Ctrl+C to stop.\n", flush=True)
    deadline = time.time() + args.minutes * 60.0
    last: dict[str, Any] = {}
    seen_exit_active = False
    try:
        while time.time() < deadline:
            try:
                now = snapshot(client.get_obs())
            except Exception as e:  # the game may be between scenes, or closing
                print(f"  (poll failed: {e})", flush=True)
                time.sleep(args.interval)
                continue
            watched = {k: now[k] for k in ("scene", "level_started", "exit_active", "level_over")}
            if watched != {k: last.get(k) for k in watched}:
                print(f"[{time.strftime('%H:%M:%S')}] scene={now['scene']} level_started={now['level_started']} "
                      f"exit.active={now['exit_active']} level_over={now['level_over']} | "
                      f"checkpoints={now['checkpoints']} arenas={now['cleared_arenas']} doors={now['unlocked_doors']} "
                      f"kills={now['kills']} t={now['seconds']}", flush=True)
            seen_exit_active = seen_exit_active or bool(now["exit_active"])
            if now["level_over"]:
                print(f"\nPASS: the bridge reported the level complete. official time {now['seconds']}s, "
                      f"kills {now['kills']}, checkpoints {now['checkpoints']}, arenas {now['cleared_arenas']}, "
                      f"doors {now['unlocked_doors']}.")
                print("The completion path env.py grades on works end to end.")
                return
            last = now
            time.sleep(args.interval)
        print(f"\nFAIL: gave up after {args.minutes} minutes without a level_over. "
              f"exit.active was ever true: {seen_exit_active}.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nstopped by hand. exit.active was ever true: {seen_exit_active}.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
