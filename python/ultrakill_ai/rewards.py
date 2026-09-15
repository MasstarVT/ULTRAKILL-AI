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

    # Early-training shaping: bonus for keeping the nearest visible enemy near the crosshair.
    aim: float = 0.0
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
    dealt = 0.0
    for e in cur.get("enemies", []):
        before = prev_hp.get(e["id"])
        if before is not None and before > e["health"]:
            dealt += (before - e["health"]) / max(enemy_max_health.get(e["id"], before), 1e-3)
    r.add("damage_dealt", cfg.damage_dealt * dealt)

    ps, cs = prev.get("stats", {}), cur.get("stats", {})
    r.add("kill", cfg.kill * max(0, cs.get("kills", 0) - ps.get("kills", 0)))
    r.add("style", cfg.style * max(0, cs.get("style", 0) - ps.get("style", 0)))

    if died is None:
        died = cp["dead"] and not pp["dead"]
    # A soft death heals the player, so the lethal hit is the HP they had left.
    hp_lost = pp["hp"] if died else max(0, pp["hp"] - cp["hp"])
    r.add("damage_taken", -cfg.damage_taken * hp_lost)
    if died:
        r.add("death", -cfg.death)

    r.add("step", -cfg.step_penalty)

    if cfg.aim:
        visible = [e for e in cur.get("enemies", []) if e["visible"]]
        if visible:
            x, y, z = visible[0]["rel"]  # camera space, nearest first
            if z > 0:
                angle = math.degrees(math.atan2(math.hypot(x, y), z))
                r.add("aim", cfg.aim * max(0.0, 1.0 - angle / cfg.aim_cone_deg))

    pcg, ccg = prev.get("cybergrind"), cur.get("cybergrind")
    if pcg and ccg:
        r.add("wave", cfg.wave * max(0, ccg["wave"] - pcg["wave"]))

    r.add("route", cfg.route_point * route_gain)
    if cs.get("level_complete") and not ps.get("level_complete"):
        r.add("level_complete", cfg.level_complete)
    if stuck:
        r.add("stuck", -cfg.stuck)

    return r
