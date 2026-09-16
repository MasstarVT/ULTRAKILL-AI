"""Reward functions. All weights live in RewardConfig so they can be tuned from a YAML file."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardConfig:
    damage_dealt: float = 1.0  # per full enemy health bar removed
    kill: float = 2.0
    damage_taken: float = 0.02  # per HP lost
    death: float = 10.0
    style: float = 0.002  # per style point gained
    step_penalty: float = 0.0  # small per-step cost to discourage idling

    # Early-training shaping for aiming at the nearest visible enemy.
    # `aim` is paid on a slope from facing away (0) to facing straight at it (full), so turning the right
    # way always pays a little more. A cone-only reward gives no gradient when the agent is never on
    # target, which is exactly what happened at 1.9M steps (on-target 0.8% of steps).
    aim: float = 0.0  # slope on the 3-D angle off the enemy
    aim_locked: float = 0.0  # extra, inside the cone
    aim_cone_deg: float = 15.0
    # Separate slopes for the two look axes. A single 3-D angle gives pitch no gradient while yaw is still
    # random (any pitch scores the same on average), and the 2.3M-step weight audit found the camera pinned at
    # the +90° clamp for that reason. `aim_yaw` pays for the heading being right regardless of pitch, and
    # `aim_pitch` for the enemy elevation being on the camera horizon regardless of yaw.
    aim_yaw: float = 0.0
    aim_pitch: float = 0.0

    # Cyber Grind
    wave: float = 5.0

    # Campaign
    route_point: float = 0.1  # per route point reached (~1 m of progress)
    level_complete: float = 50.0
    stuck: float = 1.0


@dataclass
class RewardResult:
    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        if value:
            self.parts[name] = self.parts.get(name, 0.0) + value
            self.total += value


def aim_errors(player: dict[str, Any], enemy: dict[str, Any]) -> tuple[float, float, float] | None:
    """Degrees the crosshair is off an enemy: (3-D angle, horizontal/yaw angle, vertical/pitch angle).

    The yaw error compares the camera heading and the enemy direction projected onto the ground plane, so it
    ignores pitch. The pitch error is the enemy elevation in camera space, so it ignores yaw. Both only use
    vectors the mod reports (no assumptions about the game rotation sign conventions).
    """
    x, y, z = enemy["rel"]  # camera space: x right, y up, z forward
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-6:
        return None
    angle = math.degrees(math.acos(max(-1.0, min(1.0, z / length))))
    pitch_err = abs(math.degrees(math.asin(max(-1.0, min(1.0, y / length)))))

    if "pos" in enemy and "pos" in player and "forward" in player:
        fx, _, fz = player["forward"]
        if math.hypot(fx, fz) < 1e-3:  # looking straight up or down: take the heading from the yaw angle
            yaw = math.radians(player["yaw"])
            fx, fz = math.sin(yaw), math.cos(yaw)
        ex, ez = enemy["pos"][0] - player["pos"][0], enemy["pos"][2] - player["pos"][2]
        h = math.hypot(ex, ez)
        if h <= 1e-6:
            return angle, 0.0, pitch_err
        cos = (fx * ex + fz * ez) / (math.hypot(fx, fz) * h)
    else:  # older snapshots without world positions: horizontal angle in camera space (exact when level)
        h = math.hypot(x, z)
        cos = z / h if h > 1e-6 else 1.0
    yaw_err = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return angle, yaw_err, pitch_err


def horizon_elevation(player: dict[str, Any], enemy: dict[str, Any]) -> float | None:
    """Degrees the enemy sits above the horizontal through the camera, independent of where the camera points.

    Rebuilds the camera basis from the reported world `forward` (no roll) and lifts the camera-space `rel` back
    into world space, so it needs no assumption about the sign of the pitch angle. Diagnostics only.
    """
    fwd = player.get("forward")
    if not fwd:
        return None
    fx, fy, fz = fwd
    rx, rz = fz, -fx  # right = cross(world up, forward)
    n = math.hypot(rx, rz)
    if n < 1e-6:  # looking straight up or down: no horizontal heading to build the basis from
        return None
    rx, rz = rx / n, rz / n
    uy = fz * rx - fx * rz  # y of up = cross(forward, right)
    x, y, z = enemy["rel"]
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-6:
        return None
    world_y = y * uy + z * fy
    return math.degrees(math.asin(max(-1.0, min(1.0, world_y / length))))


def compute_reward(
    cfg: RewardConfig,
    prev: dict[str, Any],
    cur: dict[str, Any],
    enemy_max_health: dict[int, float],
    route_gain: int = 0,
    stuck: bool = False,
    died: bool | None = None,
) -> RewardResult:
    r = RewardResult()
    pp, cp = prev.get("player"), cur.get("player")
    if not pp or not cp:
        return r

    # Damage dealt, normalised per enemy so a Filth and a Maurice are worth the same when killed.
    prev_hp = {e["id"]: e["health"] for e in prev.get("enemies", [])}
    cur_ids = {e["id"] for e in cur.get("enemies", [])}
    dealt = 0.0
    for e in cur.get("enemies", []):
        before = prev_hp.get(e["id"])
        if before is not None and before > e["health"]:
            dealt += (before - e["health"]) / max(enemy_max_health.get(e["id"], before), 1e-3)

    ps, cs = prev.get("stats", {}), cur.get("stats", {})
    new_kills = max(0, cs.get("kills", 0) - ps.get("kills", 0))
    if new_kills:
        # Enemies killed in one hit vanish before a health drop is ever observed. Credit the health they
        # had left, nearest first, for as many enemies as the kill counter went up.
        vanished = [e for e in prev.get("enemies", []) if e["id"] not in cur_ids]
        for e in vanished[:new_kills]:
            dealt += e["health"] / max(enemy_max_health.get(e["id"], e["health"]), 1e-3)
    r.add("damage_dealt", cfg.damage_dealt * dealt)
    r.add("kill", cfg.kill * new_kills)
    r.add("style", cfg.style * max(0, cs.get("style", 0) - ps.get("style", 0)))

    if died is None:
        died = cp["dead"] and not pp["dead"]
    # A soft death heals the player, so the lethal hit is the HP they had left.
    hp_lost = pp["hp"] if died else max(0, pp["hp"] - cp["hp"])
    r.add("damage_taken", -cfg.damage_taken * hp_lost)
    if died:
        r.add("death", -cfg.death)

    r.add("step", -cfg.step_penalty)

    if cfg.aim or cfg.aim_locked or cfg.aim_yaw or cfg.aim_pitch:
        visible = [e for e in cur.get("enemies", []) if e["visible"]]
        if visible:
            errors = aim_errors(cp, visible[0])  # nearest first
            if errors is not None:
                angle, yaw_err, pitch_err = errors
                r.add("aim", cfg.aim * (1.0 - angle / 180.0))
                r.add("aim_yaw", cfg.aim_yaw * (1.0 - yaw_err / 180.0))
                r.add("aim_pitch", cfg.aim_pitch * (1.0 - pitch_err / 90.0))
                if angle <= cfg.aim_cone_deg:
                    r.add("aim_locked", cfg.aim_locked * (1.0 - angle / cfg.aim_cone_deg))

    pcg, ccg = prev.get("cybergrind"), cur.get("cybergrind")
    if pcg and ccg:
        r.add("wave", cfg.wave * max(0, ccg["wave"] - pcg["wave"]))

    r.add("route", cfg.route_point * route_gain)
    if cs.get("level_complete") and not ps.get("level_complete"):
        r.add("level_complete", cfg.level_complete)
    if stuck:
        r.add("stuck", -cfg.stuck)

    return r
