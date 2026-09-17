"""In-game checks for campaign support, run on one game before any campaign training.

    python scripts/games.py launch --count 1 --monitor 1
    python scripts/campaign_check.py                                 # Level 0-1 at the training speed settings
    python scripts/campaign_check.py --level "Level 1-1"             # the full arsenal
    python scripts/campaign_check.py --fixed-fps 60 --frameskip 4    # when a trigger did not fire at 30 fps / frameskip 2
    python scripts/campaign_check.py --render                        # when it did not fire at 60 / 4 either
    python scripts/games.py stop

Uses the env settings of configs/campaign_0-1.yaml, except that every reset is a fresh level load and nothing is
written (no exploration archive, no best runs). Each check prints PASS, FAIL or SKIP, then a one-line summary; the
exit code is 1 when any check failed. The bridge is single-client, so never run this against a port a trainer uses.

  1 level load     a fresh reset has a campaign block with difficulty 3 (Violent); prints exit, checkpoints, path
  2 arsenal        player.slot_counts has a weapon in each of slots 1-5 (0-1 has none before the revolver pickup)
  3 checkpoint     teleporting onto the nearest pending checkpoint activates it within 30 decisions
  4 death respawn  a real (not healed) death respawns at that checkpoint inside the same episode
  5 exit           teleporting into the exit ends the episode with level_complete, and the official time stops
  6 gates          campaign.gates is present, ordered, sorted by hops, uniquely keyed, and unchanged after a
                   respawn -- hops must be stable for a level load, or the route silently renumbers mid-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.spaces import noop_action  # noqa: E402

CAMPAIGN_NOOP = noop_action(campaign=True)  # the campaign action space has the look mode as a 12th dimension

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "campaign_0-1.yaml"
VIOLENT = 3
CHECKPOINT_TRIES = 3  # pending checkpoints to try, nearest first: one may sit in a room that is still switched off
CHECKPOINT_STEPS = 30  # decisions to wait on each (2 game seconds at 15 decisions/s)
EXIT_STEPS = 60  # decisions to wait for the exit trigger (4 game seconds, room to drop into the pit)
TIMER_STEPS = 15  # decisions after the finish during which the official time must not move (1 game second)
RESPAWN_RADIUS = 10.0  # metres from the checkpoint a respawn must land within
NAMES = ("level load", "arsenal", "checkpoint", "death respawn", "exit", "gates")
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

Result = tuple[int, str, str, str]  # (number, name, status, detail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-game checks for campaign support (one game).")
    parser.add_argument("--level", default="Level 0-1", help="campaign scene name")
    parser.add_argument("--port", type=int, default=47800)
    parser.add_argument("--fixed-fps", type=float, help="override the config's fixed_fps (60 with --frameskip 4 keeps 15 decisions/s)")
    parser.add_argument("--frameskip", type=int, help="override the config's frameskip")
    parser.add_argument("--render", action="store_true", help="turn the cameras on, for triggers that may depend on rendering")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> EnvConfig:
    """configs/campaign_0-1.yaml's env settings, with fresh level loads only and no files written."""
    env = dict((yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("env", {}))
    env.update(level=args.level, port=args.port, fresh_start_prob=1.0, explore_dir="", best_runs_dir="")
    if args.fixed_fps:
        env["fixed_fps"] = args.fixed_fps
    if args.frameskip:
        env["frameskip"] = args.frameskip
    if args.render:
        env["render"] = True
    return EnvConfig.from_dict(env)


def block(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("campaign") or {}


def vec(v) -> str:
    return "(" + ", ".join(f"{c:.1f}" for c in v) + ")"


def secs(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def is_activated(raw: dict[str, Any], checkpoint_id: str) -> bool:
    return any(cp["id"] == checkpoint_id and (cp["activated"] or cp["current"]) for cp in block(raw).get("checkpoints", []))


def pending_checkpoints(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Checkpoints that are neither activated nor current, nearest to the player first."""
    pos = raw["player"]["pos"]
    pending = [cp for cp in block(raw).get("checkpoints", []) if not cp["activated"] and not cp["current"]]
    return sorted(pending, key=lambda cp: math.dist(cp["pos"], pos))


def run_noops(env: UltrakillEnv, steps: int, until: Callable[[dict[str, Any]], bool]) -> tuple[bool, dict[str, Any], bool]:
    """Steps no-op decisions until `until(raw)` holds or the episode ends: (condition met, last info, episode ended)."""
    info: dict[str, Any] = {}
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(CAMPAIGN_NOOP)
        if until(env._raw):  # the raw snapshot the env just received from the mod
            return True, info, terminated or truncated
        if terminated or truncated:
            return False, info, True
    return False, info, False


def check_level_load(raw: dict[str, Any]) -> tuple[str, str]:
    player, cb = raw.get("player"), raw.get("campaign")
    if not player or not cb:
        missing = "player" if not player else "campaign block"
        return FAIL, f"scene {raw.get('scene')!r} has no {missing} after a fresh reset (mod older than v0.5.0, or not a campaign scene)"
    ext, path = cb.get("exit"), cb.get("path") or {}
    exit_text = f"exit {vec(ext['pos'])} active={ext['active']}" if ext else "exit null"
    path_text = f"path {path.get('status')}" + (f" {path['length']:.1f} m" if path.get("length") is not None else "")
    checkpoints = sorted(cb.get("checkpoints", []), key=lambda cp: math.dist(cp["pos"], player["pos"]))
    detail = (f"mission {cb.get('mission')}, difficulty {cb.get('difficulty')}, {exit_text}, {len(checkpoints)} checkpoints, "
              f"{path_text}, {len(cb.get('locked_doors', []))} locked doors listed, input_locked={cb.get('input_locked')}")
    for cp in checkpoints:
        detail += (f"\n      checkpoint {cp['id']} {math.dist(cp['pos'], player['pos']):.1f} m away, "
                   f"activated={cp['activated']} current={cp['current']}")
    if cb.get("difficulty") != VIOLENT:
        return FAIL, detail + f"\n      difficulty reads {cb.get('difficulty')}, expected {VIOLENT} (Violent)"
    return PASS, detail


def check_arsenal(raw: dict[str, Any], level: str) -> tuple[str, str]:
    counts = (raw.get("player") or {}).get("slot_counts")
    if counts is None:
        return FAIL, "player.slot_counts is missing (mod older than v0.5.0)"
    if len(counts) >= 5 and all(c > 0 for c in counts[:5]):
        return PASS, f"slot_counts {counts}"
    if level == "Level 0-1":
        return SKIP, f"slot_counts {counts}: 0-1 has no weapons until the revolver pickup; run with --level \"Level 1-1\" to see the arsenal"
    return SKIP, f"slot_counts {counts}: not every weapon slot is filled, so unlock_all_gear did not reach GunSetter"


def check_checkpoint(env: UltrakillEnv) -> tuple[str, str, dict[str, Any] | None, bool]:
    """(status, detail, the checkpoint that activated, episode ended)."""
    raw = env._raw
    if not raw.get("player") or not raw.get("campaign"):
        return SKIP, "needs a player and the campaign block (check 1)", None, False
    candidates = pending_checkpoints(raw)[:CHECKPOINT_TRIES]
    if not candidates:
        return SKIP, "no pending checkpoint in this level", None, False
    notes = []
    for cp in candidates:
        x, y, z = cp["pos"]
        env.client.teleport([x, y + 1.0, z])
        reached, info, ended = run_noops(env, CHECKPOINT_STEPS, lambda r, cid=cp["id"]: is_activated(r, cid))
        if reached:
            notes.append(f"checkpoint {cp['id']} activated (checkpoints_level {info.get('checkpoints_level')})")
            return PASS, "; ".join(notes), cp, ended
        notes.append(f"checkpoint {cp['id']} did not activate in {CHECKPOINT_STEPS} decisions "
                     f"(arena_enemies_alive {block(env._raw).get('arena_enemies_alive')})")
        if ended:
            notes.append(f"episode ended: {info.get('end_reason')}")
            return FAIL, "; ".join(notes), None, True
    return FAIL, "; ".join(notes), None, False


def check_death(env: UltrakillEnv, checkpoint: dict[str, Any] | None) -> tuple[str, str, bool]:
    """(status, detail, episode ended)."""
    if checkpoint is None:
        return SKIP, "needs the checkpoint activated in check 3", False
    # Check 3 can already have cost a death (a checkpoint in a room that is still switched off has no floor under
    # it), so the kill must add exactly one to the episode's count rather than make it 1.
    deaths_before = env._deaths
    killed = env.client.kill()
    # A real death only: soft death heals the lethal hit, and the env would still count and respawn that one.
    killed_dead = (killed.get("player") or {}).get("dead")
    _, _, terminated, truncated, info = env.step(CAMPAIGN_NOOP)
    ended = terminated or truncated
    player = env._raw.get("player")
    alive = player is not None and not player["dead"]
    dist = math.dist(player["pos"], checkpoint["pos"]) if player else math.inf
    detail = (f"kill reply dead={killed_dead}, deaths {deaths_before} -> {info.get('deaths')}, "
              f"episode ended={ended}" + (f" ({info.get('end_reason')})" if ended else "")
              + f", player {'alive' if alive else 'missing or dead'} {dist:.1f} m from checkpoint {checkpoint['id']}")
    ok = (killed_dead is True and info.get("deaths") == deaths_before + 1 and not ended and alive
          and dist <= RESPAWN_RADIUS)
    return (PASS if ok else FAIL), detail, ended


def check_exit(env: UltrakillEnv) -> tuple[str, str]:
    raw = env._raw
    if not raw.get("player") or not raw.get("campaign"):
        return SKIP, "needs a player and the campaign block (check 1)"
    ext = block(raw).get("exit")
    if not ext:
        return FAIL, "exit null: no real FinalPit was found (decoy or template filter, or the pit is not in the scene)"
    env.client.teleport(ext["pos"])
    _, info, ended = run_noops(env, EXIT_STEPS, lambda r: False)
    if not ended:
        return FAIL, f"exit {vec(ext['pos'])} active={ext['active']}: the trigger did not fire in {EXIT_STEPS} decisions"
    reason, level_seconds = info.get("end_reason"), info.get("level_seconds")
    final = later = block(env._raw).get("seconds")
    for _ in range(TIMER_STEPS):
        later = block(env.client.step({})).get("seconds")
    detail = (f"end_reason {reason}, level_seconds {secs(level_seconds)}, block seconds {secs(final)} at the finish and "
              f"{secs(later)} one game second later, rank {info.get('rank')}")
    ok = (reason == "level_complete" and None not in (level_seconds, final, later)
          and abs(level_seconds - final) < 1e-3 and abs(later - final) < 1e-3)
    return (PASS if ok else FAIL), detail


def gate_summary(gates: list[dict]) -> str:
    hops = [g.get("hops") for g in gates]
    ordered_hops = sorted({h for h in hops if h is not None})
    return (f"{len(gates)} gates, hops {min(ordered_hops) if ordered_hops else '-'}..{max(ordered_hops) if ordered_hops else '-'} "
            f"({len(ordered_hops)} distinct), {sum(1 for h in hops if h is None)} unordered, "
            f"{sum(1 for g in gates if not g.get('controller_active', True))} with an inactive controller")


def check_gates(before: dict[str, Any], after: dict[str, Any] | None) -> tuple[str, str]:
    """(status, detail) for the gates block on a fresh load, and again after a respawn if there was one."""
    if "gates" not in before:
        return SKIP, "campaign.gates is missing (mod older than v0.6.0)"
    gates = before.get("gates") or []
    detail = gate_summary(gates)
    problems = []
    if not before.get("gates_ordered"):
        problems.append("gates_ordered is false: no room node is an ancestor of the exit pit")
    if before.get("gates_truncated"):
        problems.append("gates_truncated is true: the level has more gate candidates than the cap")
    keys = [str(g.get("key")) for g in gates]
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys: Python uses the key alone to detect a target change")
    if not gates:
        problems.append("the gates array is empty")
    elif not any(g.get("hops") == 0 for g in gates):
        problems.append("no gate reports hops 0, so nothing leads into the exit's room")
    # Sorted by hops ascending with null last, then by key: the contract the mod sends them under.
    sort_key = [(g.get("hops") is None, g.get("hops") if g.get("hops") is not None else 0, str(g.get("key"))) for g in gates]
    if sort_key != sorted(sort_key):
        problems.append("the array is not sorted by hops ascending with null last, then by key")
    if after is not None:
        was = {str(g.get("key")): g.get("hops") for g in gates}
        now = {str(g.get("key")): g.get("hops") for g in (after.get("gates") or [])}
        detail += f"; after a respawn: {gate_summary(after.get('gates') or [])}"
        if was != now:
            changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))[:5]
            problems.append(f"hops changed across a respawn for {changed}: the route renumbered mid-load")
    else:
        detail += "; no respawn happened, so stability across one was not checked"
    if problems:
        return FAIL, detail + "".join(f"\n      {p}" for p in problems)
    return PASS, detail


def run_checks(env: UltrakillEnv) -> list[Result]:
    results: list[Result] = []

    def record(number: int, status: str, detail: str) -> None:
        results.append((number, NAMES[number - 1], status, detail))
        print(f"[{status}] {number} {NAMES[number - 1]}: {detail}", flush=True)

    env.reset()
    fresh_gates = dict(block(env._raw))
    record(1, *check_level_load(env._raw))
    record(2, *check_arsenal(env._raw, env.cfg.level))
    status, detail, checkpoint, ended = check_checkpoint(env)
    record(3, status, detail)
    respawned = None
    if ended:
        record(4, SKIP, "the episode ended during check 3")
    else:
        status, detail, ended = check_death(env, checkpoint)
        record(4, status, detail)
        if checkpoint is not None:
            respawned = dict(block(env._raw))  # the obs after the death respawn
    if ended:
        env.reset()  # the exit check needs a running episode; this is another fresh level load
    record(5, *check_exit(env))
    record(6, *check_gates(fresh_gates, respawned))
    return results


def exit_code(results: list[Result]) -> int:
    return 1 if any(status == FAIL for _, _, status, _ in results) else 0


def main() -> int:
    cfg = build_config(parse_args())
    print(f"{cfg.level} on port {cfg.port}: fixed_fps {cfg.fixed_fps:g}, frameskip {cfg.frameskip}, render {cfg.render}, "
          f"difficulty {cfg.difficulty}, unlock_all_gear {cfg.unlock_all_gear}", flush=True)
    env = UltrakillEnv(cfg)
    try:
        results = run_checks(env)
    finally:
        env.close()
    print("summary: " + " | ".join(f"{number} {name} {status}" for number, name, status, _ in results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
