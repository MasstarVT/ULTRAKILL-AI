"""Records a full-fidelity per-step trace of a policy (or a scripted driver) on one private game.

    python scripts/probe_rollout.py --config configs/generated/spec_0-3.yaml \
        --model <a COPY of a checkpoint> --port 47812 --episodes 8 --out runs/probe_0-3

Why this exists: `episodes.jsonl` carries one row per episode (`end_pos`, `gate_hops_best`,
`targets_parked`), which is enough to say a level is stuck and never enough to say WHY. This writes one
JSON line per DECISION -- position, velocity, the ground reading, the route target, the exact eight
observation slots the target occupies, every rung credited so far, the whole patience/parking state,
the decoded 12-dimension action and the reward parts -- so a stall can be read back frame by frame.

It builds the SAME env the stage builds, from the stage's own generated config, and then forces the
read-only settings an off-run probe must have (see `probe_config`): a private port, one level, fresh
loads only, no bridge relaunch, no archive/best-run/curriculum writes, no env log. It reads the live
run's exploration archive exactly as `eval.py` does -- the policy was trained with those counts as
inputs -- and never writes it back.

**Never point `--port` at 47800-47811.** The bridge is single-client: connecting to a trainer's game
drops the trainer's socket and kills that worker. The default is 47812, the private-instance port, and
a port in the training range is refused below.

`--scripted` replaces the policy with a hand-written controller (face the target, walk into it, jump
when horizontal speed collapses). That is a probe of GEOMETRY -- is this climb walkable at all -- and
is never training data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.procmem import cap_blas_threads  # noqa: E402

cap_blas_threads()  # before numpy: OpenBLAS reserves ~785 MB of commit for thread buffers at load

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ultrakill_ai.campaign import ExplorationArchive, safe_name  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.spaces import BUTTONS, PITCH_BINS, YAW_BINS, LOOK_MODE_INDEX  # noqa: E402

TRAINING_PORTS = range(47800, 47812)
# The campaign block is the tail of the observation; the target occupies its slots 5..12.
CAMPAIGN_BLOCK = 36
TARGET_SLICE = slice(-CAMPAIGN_BLOCK + 5, -CAMPAIGN_BLOCK + 13)


def probe_config(cfg: EnvConfig, port: int, level: str | None) -> EnvConfig:
    """The stage's config with every write and every recovery lever turned off. Read-only by construction."""
    if port in TRAINING_PORTS:
        raise SystemExit(f"refusing port {port}: 47800-47811 are the live trainer's games and the bridge is "
                         "single-client -- connecting would kill that worker")
    cfg.port = port
    if level:
        cfg.level = level
    cfg.levels = []           # one level, never a curriculum draw
    cfg.curriculum_path = ""  # never read, let alone perturb, a live run's curriculum
    cfg.fresh_start_prob = 1.0  # every episode is a fresh level load, like the fresh rows we are explaining
    cfg.explore_dir = ""      # the archive is loaded by hand below and never written back
    cfg.best_runs_dir = ""
    cfg.env_log_dir = ""
    cfg.bridge_relaunch = False  # this probe never restarts a game
    cfg.soft_death = False
    return cfg


def rung_names(env: UltrakillEnv) -> dict[str, str]:
    route = getattr(env.gates, "_route", None) or {}
    return {str(r.get("key")): str(r.get("name")) for r in route.get("rungs", ())}


def gate_state(env: UltrakillEnv, pos, names: dict[str, str]) -> dict:
    """Everything `GateProgress` knows this step, flattened. Read-only attribute access."""
    g = env.gates
    t = g.target or {}
    key = str(t.get("key")) if t else None
    out = {
        "target_key": key,
        "target_name": names.get(key or ""),
        "target_pos": [round(float(v), 2) for v in t["pos"]] if t.get("pos") else None,
        "target_hops": t.get("hops"),
        "target_subgoal": t.get("subgoal"),
        "best_hops": g.best_hops,
        "reached": sorted(g.reached),
        "gates_reached": g.gates_reached,
        "parked": sorted(g.parked),
        "park_count": {k: v for k, v in g.park_count.items()},
        "parks": g.parks,
        "clock": g._clock,
        "clock_suspended": g._clock_suspended,
        "clock_best": None if math.isinf(g._clock_best) else round(g._clock_best, 2),
        "clock_key": g._clock_key,
        "fallback": bool(g._fallback),
        "route_source": g.route_source,
        "ladder_collapsed": g.ladder_collapsed,
    }
    if pos is not None and t.get("pos"):
        d = [t["pos"][i] - pos[i] for i in range(3)]
        flat = math.hypot(d[0], d[2])
        out["target_dist3"] = round(math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2), 2)
        out["target_dist_h"] = round(flat, 2)
        out["target_dy"] = round(d[1], 2)
        out["target_elev_deg"] = round(math.degrees(math.atan2(d[1], flat)), 1) if flat > 1e-6 else 90.0
    return out


def behaviour_delta(env: UltrakillEnv, prev: dict) -> tuple[dict, dict]:
    """What `_note_behaviour` counted for THIS decision, by diffing the per-episode counters.

    The env grades the look mode that actually drove the camera (modes 1 and 2 fall back to 0 when there is
    nothing to aim at), and whether the shot was on target, and keeps only episode sums. The diff is the only
    way to get either per step without duplicating the env's aim maths here.
    """
    cur = {k: float(v) for k, v in env._behaviour.items()}
    out = {}
    for k, v in cur.items():
        d = v - prev.get(k, 0.0)
        if d:
            out[k] = round(d, 4)
    return out, cur


def world_state(raw: dict, pos) -> dict:
    """The parts of the raw frame a time budget needs: the clock, the fight, the doors and the deaths."""
    camp = raw.get("campaign") or {}
    stats = raw.get("stats") or {}
    enemies = raw.get("enemies") or []
    visible = [e for e in enemies if e.get("visible")]
    locked = [d for d in (camp.get("locked_doors") or ()) if isinstance(d, dict)]
    out = {
        "kills": stats.get("kills"),
        "style": stats.get("style"),
        "restarts": stats.get("restarts"),
        "level_complete": stats.get("level_complete"),
        "timer_running": camp.get("timer_running"),
        "input_locked": camp.get("input_locked"),
        "level_over": camp.get("level_over"),
        "dead": p_dead(raw),
        "anti_hp": (raw.get("player") or {}).get("anti_hp"),
        # Recorded RAW, which is the 1-BASED slot KEY (1..6, or -1 before GunControl starts) -- see
        # docs/protocol.md and `env.held_slot_key`. Anything reading this column back must not treat it
        # as a 0-based index: that was the 2026-09-20 bug.
        "weapon_slot": (raw.get("player") or {}).get("weapon_slot"),
        "n_enemies": len(enemies),
        "n_visible": len(visible),
        "nearest_dist": round(float(enemies[0]["dist"]), 2) if enemies else None,
        "nearest_vis_dist": round(float(visible[0]["dist"]), 2) if visible else None,
        "cleared_arenas": len(camp.get("cleared_arenas") or ()),
        "unlocked_doors": len(camp.get("unlocked_doors") or ()),
        "n_locked_doors": len(locked),
    }
    if locked and pos:
        near = min(locked, key=lambda d: sum((d["pos"][k] - pos[k]) ** 2 for k in range(3))
                   if d.get("pos") else float("inf"))
        if near.get("pos"):
            out["locked_door_dist"] = round(math.dist(near["pos"], pos), 2)
            out["locked_door_id"] = near.get("id")
    return out


def p_dead(raw: dict):
    return (raw.get("player") or {}).get("dead")


def step_record(i: int, raw: dict, action, obs, reward: float, info: dict,
                env: UltrakillEnv, names: dict[str, str], beh: dict | None = None) -> dict:
    p = raw.get("player") or {}
    pos = p.get("pos")
    vel = p.get("vel") or (0.0, 0.0, 0.0)
    a = np.asarray(action, dtype=np.int64).tolist()
    bi = 2 + len(BUTTONS)
    rec = {
        "i": i,
        "hspeed": round(math.hypot(float(vel[0]), float(vel[2])), 2),
        "pos": [round(float(v), 2) for v in pos] if pos else None,
        "vel": [round(float(v), 2) for v in p.get("vel", ())] or None,
        "local_vel": [round(float(v), 2) for v in p.get("local_vel", ())] or None,
        "yaw": round(float(p["yaw"]), 1) if "yaw" in p else None,
        "pitch": round(float(p["pitch"]), 1) if "pitch" in p else None,
        "grounded": p.get("grounded"),
        "sliding": p.get("sliding"),
        "slow_mode": p.get("slow_mode"),
        "heavy_fall": p.get("heavy_fall"),
        "hp": p.get("hp"),
        "stamina": round(float(p.get("stamina", 0.0)), 1),
        "ground_ray_center": raw.get("ground_ray_center"),
        "ground_drop": env._ground_drop(raw),
        "level_seconds": (raw.get("campaign") or {}).get("seconds"),
        "arena_alive": (raw.get("campaign") or {}).get("arena_enemies_alive"),
        "act": {
            "forward": a[0] - 1, "side": a[1] - 1,
            "buttons": [n for n, b in zip(BUTTONS, a[2:bi]) if b],
            "slot": a[bi], "yaw_deg": YAW_BINS[a[bi + 1]], "pitch_deg": PITCH_BINS[a[bi + 2]],
            "look_mode": a[LOOK_MODE_INDEX] if len(a) > LOOK_MODE_INDEX else 0,
        },
        # The eight observation slots the target occupies, exactly as the policy read them this step:
        # rel x/right, y/up, z/forward and 3-D distance (all /50 and /100 for a gate, clipped to +-4),
        # then mask, open, locked, hops/20.
        "obs_target": [round(float(v), 4) for v in np.asarray(obs)[TARGET_SLICE]],
        "reward": round(float(reward), 4),
        "reward_parts": {k: round(float(v), 4) for k, v in (info.get("reward_parts") or {}).items() if v},
    }
    rec.update(world_state(raw, pos))
    if beh:
        rec["beh"] = beh
    rec.update(gate_state(env, pos, names))
    return rec


def scripted_action(raw: dict, env: UltrakillEnv, state: dict) -> np.ndarray:
    """Face the current route target, walk into it, and jump when forward motion collapses.

    A deliberately dumb driver: it tests whether the geometry between here and the target can be walked
    or hopped at all, which is a question about the level and not about the policy.
    """
    a = np.zeros(12, dtype=np.int64)
    a[0] = a[1] = 1
    a[2 + len(BUTTONS)] = 0
    a[3 + len(BUTTONS)] = YAW_BINS.index(0.0)
    a[4 + len(BUTTONS)] = PITCH_BINS.index(0.0)
    a[LOOK_MODE_INDEX] = 2  # let the env aim the camera at the target, as the policy may
    p = raw.get("player") or {}
    t = env.gates.target or {}
    if not p.get("pos") or not t.get("pos"):
        return a
    pos, tgt = p["pos"], t["pos"]
    want = math.degrees(math.atan2(tgt[0] - pos[0], tgt[2] - pos[2]))
    err = (want - float(p["yaw"]) + 180.0) % 360.0 - 180.0
    a[3 + len(BUTTONS)] = int(np.argmin([abs(b - max(-90.0, min(90.0, err))) for b in YAW_BINS]))
    a[0] = 2  # forward, always
    speed = math.hypot(p.get("vel", (0, 0, 0))[0], p.get("vel", (0, 0, 0))[2])
    state["slow"] = state.get("slow", 0) + 1 if speed < 2.0 else 0
    # Jump on a cadence when moving, and immediately when motion has collapsed against something.
    if p.get("grounded") and (state["slow"] >= 3 or state.get("n", 0) % 12 == 0):
        a[2 + BUTTONS.index("jump")] = 1
    state["n"] = state.get("n", 0) + 1
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the stage's generated config, e.g. configs/generated/spec_0-3.yaml")
    ap.add_argument("--model", help="a COPY of a checkpoint (never the trainer's own file); omit with --scripted")
    ap.add_argument("--port", type=int, default=47812)
    ap.add_argument("--level", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="runs/probe_0-3")
    ap.add_argument("--tag", default="rollout")
    ap.add_argument("--first", type=int, default=0, help="episode number to start naming files at")
    ap.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--scripted", action="store_true", help="drive by hand instead of by policy (geometry probe)")
    ap.add_argument("--max-steps", type=int, default=0, help="extra cap below the env's own max_steps (0 = none)")
    ap.add_argument("--archive-port", type=int, default=47800, help="whose exploration counts to read (read-only)")
    ap.add_argument("--archive-dir", default="models/spec_0-3",
                    help="the RUN's model folder, where the live exploration archives are (never written)")
    args = ap.parse_args()

    doc = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = probe_config(EnvConfig.from_dict(doc["env"]), args.port, args.level)
    model = None
    if not args.scripted:
        if not args.model:
            raise SystemExit("--model is required unless --scripted")
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = UltrakillEnv(cfg)
    # The policy was trained reading these visit counts; a fresh archive would show it a map it never saw.
    # `explore_dir` is "" above, so the env never writes them back.
    archive_path = Path(args.archive_dir) / f"explore_{safe_name(env.level)}_{args.archive_port}.npz"
    if archive_path.exists():
        env.archive = ExplorationArchive.load(archive_path, cfg.cell_size)
    print(f"exploration counts: {len(env.archive.counts)} cells from {archive_path}", flush=True)
    names = rung_names(env)
    print(f"route rungs: {names}", flush=True)

    summary = []
    try:
        for ep in range(args.first, args.first + args.episodes):
            path = out / f"{args.tag}_{ep}.jsonl"
            obs, _ = env.reset()
            state, i, total, t0 = {}, 0, 0.0, time.monotonic()
            beh_prev = {k: float(v) for k, v in env._behaviour.items()}
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "meta": True, "episode": ep, "level": env.level, "port": args.port,
                    "model": str(args.model), "deterministic": bool(args.deterministic),
                    "scripted": bool(args.scripted), "config": args.config,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                }) + "\n")
                while True:
                    if args.scripted:
                        action = scripted_action(env._raw, env, state)
                    else:
                        action, _ = model.predict(obs, deterministic=args.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    beh, beh_prev = behaviour_delta(env, beh_prev)
                    rec = step_record(i, env._raw, action, obs, reward, info, env, names, beh)
                    if terminated or truncated:
                        rec["end_reason"] = info.get("end_reason")
                        rec["info"] = {k: info.get(k) for k in (
                            "completed", "gates_reached", "gate_hops_best", "targets_parked", "end_pos",
                            "episode_seconds", "level_seconds", "cells_new", "oob_frac", "deaths",
                            "exit_ground_dist_min", "route_source_name", "ladder_collapsed", "wedged_steps")}
                    fh.write(json.dumps(rec) + "\n")
                    i += 1
                    if terminated or truncated:
                        break
                    if args.max_steps and i >= args.max_steps:
                        fh.write(json.dumps({"i": i, "end_reason": "probe_cap"}) + "\n")
                        break
            row = {"episode": ep, "steps": i, "reward": round(total, 1),
                   "end_reason": info.get("end_reason"), "hops": info.get("gate_hops_best"),
                   "parked": info.get("targets_parked"), "end_pos": info.get("end_pos"),
                   "completed": info.get("completed"), "seconds": round(time.monotonic() - t0, 1),
                   "level_seconds": info.get("level_seconds"), "deaths": info.get("deaths"),
                   "episode_seconds": info.get("episode_seconds"), "gates_reached": info.get("gates_reached"),
                   "file": str(path)}
            summary.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
        (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
