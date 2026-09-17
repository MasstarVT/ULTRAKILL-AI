"""Plays a campaign level to its exit with scripted movement, to prove a completion registers through the bridge.

    python scripts/games.py launch --count 1 --monitor 1
    python scripts/walk_to_exit.py                       # Level 0-1, rendering on so you can watch
    python scripts/walk_to_exit.py --level "Level 1-1"
    python scripts/games.py stop

Why this exists. `campaign_check.py` check 5 teleports straight onto the `FinalPit` and FAILs on 0-1 and 1-1,
because a fresh load reports the exit `active: false`: the room holding it is switched off, so its trigger
collider cannot raise `OnTriggerEnter` however you arrive. That was assumed harmless on the theory that real play
switches the room on, but nothing ever confirmed it, and a run that cannot report a completion can never be
trained toward one.

**Teleporting cannot answer the question.** Measured on 0-1 (2026-09-16): teleport hops of 5 m walk the player
through the level's whole geometry while `enemies`, `locked_doors` and `cleared_arenas` all stay empty and the
exit stays inactive -- rooms ahead of the player are switched off, so their trigger volumes are off too, and a
teleport into dead space fires nothing. Dropping to 1.2 m hops instead jams the player against the first closed
door at z 347, which is what the 5 m hops had been tunnelling through. So this drives the player with real
movement input, the way training does: training reaches checkpoint 5 of 6 and is paid for `arena_clear` and
`door_unlock`, which proves walking does activate rooms and open doors.

Waypoints are the level's own checkpoints in nearest-neighbour order (its progression order on a linear level),
then the exit. Enemies in the way are shot. Deaths are disabled (`soft_death`) so Violent cannot end the test,
nothing is written, and every reset is a fresh level load.

The bridge is single-client: never run this against a port a trainer is using.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "campaign_0-1.yaml"
ARRIVE = 4.0          # metres from a waypoint that count as reached
MAX_TURN = 25.0       # degrees of yaw per decision, so the camera does not spin past the target
STALL_STEPS = 45      # decisions without closing on the waypoint before trying to get unstuck
STALL_GAIN = 1.0      # metres of progress that count as "not stuck"
UNSTICK_NUDGE = 2.5   # metres of teleport assist after repeated stalls, along the way we are already facing
FIRE_RANGE = 60.0     # metres within which a visible enemy is worth shooting
PIT_FALL_STEPS = 120  # decisions to keep falling once over the pit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play a campaign level to its exit and check the completion signal.")
    p.add_argument("--level", default="Level 0-1", help="campaign scene name")
    p.add_argument("--port", type=int, default=47800)
    p.add_argument("--budget", type=int, default=9000, help="decisions per waypoint before giving up on it")
    p.add_argument("--no-render", action="store_true", help="cameras off (faster, nothing to watch)")
    p.add_argument("--no-teleport-assist", action="store_true", help="never nudge past a stall; pure play")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> EnvConfig:
    env = dict((yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("env", {}))
    env.update(
        level=args.level, port=args.port, fresh_start_prob=1.0, explore_dir="", best_runs_dir="",
        render=not args.no_render,
        soft_death=True,     # the walk is about the exit, not about surviving Violent on the way
        max_steps=10_000_000,  # the env's own truncation must not end the walk
    )
    return EnvConfig.from_dict(env)


def block(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("campaign") or {}


def vec(p) -> str:
    return f"({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})"


def finished(raw: dict[str, Any]) -> bool:
    """The two independent ways the mod reports a finished level (ObservationBuilder: sm.infoSent || nm.levelOver)."""
    return bool(raw.get("stats", {}).get("level_complete") or block(raw).get("level_over"))


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def heading_to(src, dst) -> float:
    """Yaw in degrees that faces `dst` from `src`, in the game's convention (forward = (sin y, cos y))."""
    return math.degrees(math.atan2(dst[0] - src[0], dst[2] - src[2]))


def elevation_to(src, dst) -> float:
    flat = math.hypot(dst[0] - src[0], dst[2] - src[2])
    return math.degrees(math.atan2(dst[1] - src[1], max(flat, 1e-6)))


def route(start, checkpoints: list[dict[str, Any]], exit_pos) -> list[tuple[str, list[float]]]:
    """Checkpoints nearest-neighbour from the spawn, then the exit: a linear level's own progression order."""
    pending, here, legs = list(checkpoints), list(start), []
    while pending:
        nxt = min(pending, key=lambda c: math.dist(here, c["pos"]))
        pending.remove(nxt)
        legs.append((f"checkpoint {nxt['id']}", list(nxt["pos"])))
        here = list(nxt["pos"])
    legs.append(("exit", list(exit_pos)))
    return legs


def target_enemy(raw: dict[str, Any], pos) -> dict[str, Any] | None:
    best = None
    for e in raw.get("enemies") or []:
        if e.get("health", 0) <= 0 or not e.get("visible"):
            continue
        d = e.get("dist") or math.dist(pos, e["pos"])
        if d <= FIRE_RANGE and (best is None or d < best[0]):
            best = (d, e)
    return best[1] if best else None


def go_to(env: UltrakillEnv, name: str, target, budget: int, assist: bool) -> tuple[dict, int, bool]:
    """Walks toward `target`, shooting what gets in the way. Returns (raw, decisions_used, finished)."""
    raw = env._raw
    used, stalls, best_gap = 0, 0, math.inf
    since_gain = 0
    while used < budget:
        player = raw.get("player")
        if player is None:
            raw = env.client.step({})
            used += 1
            continue
        pos, yaw, pitch = player["pos"], player["yaw"], player["pitch"]
        gap = math.dist(pos, target)
        if gap <= ARRIVE:
            return raw, used, finished(raw)

        # Aim: an enemy if one is worth shooting, otherwise the waypoint.
        enemy = target_enemy(raw, pos)
        aim_at = enemy["pos"] if enemy else target
        yaw_err = wrap(heading_to(pos, aim_at) - yaw)
        pitch_err = elevation_to(pos, aim_at) - pitch
        look = [max(-MAX_TURN, min(MAX_TURN, yaw_err)), max(-MAX_TURN, min(MAX_TURN, pitch_err))]

        buttons = []
        if enemy and abs(yaw_err) < 12.0:
            buttons.append("fire1")
        # Move forward toward the waypoint whatever we are aiming at; strafe and jump when progress stops.
        move = [0, 1]
        if since_gain > STALL_STEPS // 2:
            buttons.append("jump")
            move = [1 if (used // 12) % 2 == 0 else -1, 1]
        command = {"move": move, "buttons": buttons, "slot": 1, "look": look}

        raw = env.client.step(command)
        used += 1
        if finished(raw):
            return raw, used, True

        new_pos = (raw.get("player") or {}).get("pos", pos)
        new_gap = math.dist(new_pos, target)
        if new_gap < best_gap - STALL_GAIN:
            best_gap, since_gain = new_gap, 0
        else:
            since_gain += 1
        if since_gain >= STALL_STEPS:
            stalls += 1
            since_gain = 0
            if assist:
                # Nudge along the direction we are already facing. Small, so it cannot tunnel a whole room the way
                # the teleport-only version did -- it only gets us off a lip or a doorframe.
                f = [math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))]
                raw = env.client.teleport([new_pos[0] + f[0] * UNSTICK_NUDGE,
                                           new_pos[1] + 1.0,
                                           new_pos[2] + f[2] * UNSTICK_NUDGE])
                used += 1
    return raw, used, finished(raw)


def main() -> None:
    args = parse_args()
    env = UltrakillEnv(build_config(args))
    seen_active = False
    try:
        env.reset()
        raw = env._raw
        camp = block(raw)
        if not camp:
            print("FAIL: no campaign block; is this a campaign scene?")
            sys.exit(1)
        ext, checkpoints = camp.get("exit"), camp.get("checkpoints") or []
        if not ext:
            print("FAIL: no exit (FinalPit) in the scene")
            sys.exit(1)
        spawn = (raw.get("player") or {}).get("pos", [0, 0, 0])
        print(f"{args.level}: spawn {vec(spawn)}, exit {vec(ext['pos'])} active={ext['active']}, "
              f"{len(checkpoints)} checkpoints, difficulty {camp.get('difficulty')}", flush=True)

        done = False
        for name, target in route(spawn, checkpoints, ext["pos"]):
            raw, used, done = go_to(env, name, target, args.budget, not args.no_teleport_assist)
            c = block(raw)
            e = c.get("exit") or {}
            seen_active = seen_active or bool(e.get("active"))
            pos = (raw.get("player") or {}).get("pos", [0, 0, 0])
            print(f"  -> {name:<16} {vec(target):<18} at {vec(pos)} after {used} decisions | "
                  f"exit active={e.get('active')} | kills={raw.get('stats', {}).get('kills')} | "
                  f"cleared={len(c.get('cleared_arenas') or [])} doors={len(c.get('unlocked_doors') or [])} | "
                  f"level_over={finished(raw)}", flush=True)
            if done:
                break

        if not done:
            print(f"  -> over the pit, falling for {PIT_FALL_STEPS} decisions", flush=True)
            for _ in range(PIT_FALL_STEPS):
                raw = env.client.step({})
                if finished(raw):
                    done = True
                    break

        camp = block(raw)
        if done:
            print(f"PASS: the level reported complete. official time {camp.get('seconds')}s, "
                  f"kills {raw.get('stats', {}).get('kills')}, restarts {camp.get('restarts')}, "
                  f"timer_running {camp.get('timer_running')}")
        else:
            e = camp.get("exit") or {}
            print(f"FAIL: never reported complete. exit active={e.get('active')} (was it ever active: {seen_active}) "
                  f"at {vec(e.get('pos', [0, 0, 0]))}, player {vec((raw.get('player') or {}).get('pos', [0, 0, 0]))}")
        sys.exit(0 if done else 1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
