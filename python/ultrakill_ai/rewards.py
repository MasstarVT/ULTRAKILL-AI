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
    aim: float = 0.0
    aim_locked: float = 0.0  # extra, inside the cone
    aim_cone_deg: float = 15.0

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

    if cfg.aim or cfg.aim_locked:
        visible = [e for e in cur.get("enemies", []) if e["visible"]]
        if visible:
            x, y, z = visible[0]["rel"]  # camera space, nearest first
            length = math.sqrt(x * x + y * y + z * z)
            if length > 1e-6:
                angle = math.degrees(math.acos(max(-1.0, min(1.0, z / length))))
                r.add("aim", cfg.aim * (1.0 - angle / 180.0))
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
