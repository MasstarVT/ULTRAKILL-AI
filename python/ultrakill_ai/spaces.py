"""Observation packing and action decoding.

The mod sends a raw structured snapshot (see docs/protocol.md). This module turns it into a
fixed-size float vector for the policy, and turns MultiDiscrete policy outputs into mod actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

YAW_BINS = (-90.0, -30.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 30.0, 90.0)  # degrees per step
PITCH_BINS = (-20.0, -6.0, -1.5, 0.0, 1.5, 6.0, 20.0)  # degrees per step, positive looks up
BUTTONS = ("jump", "dash", "slide", "fire1", "fire2", "punch")
NUM_WEAPON_CHOICES = 6  # 0 = keep current, 1..5 = select slot

# [move_forward, move_side, *buttons, weapon, yaw, pitch]
ACTION_NVEC = np.array([3, 3, *([2] * len(BUTTONS)), NUM_WEAPON_CHOICES, len(YAW_BINS), len(PITCH_BINS)], dtype=np.int64)


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_NVEC)


def decode_action(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.int64)
    forward = int(a[0]) - 1
    side = int(a[1]) - 1
    pressed = [name for name, bit in zip(BUTTONS, a[2 : 2 + len(BUTTONS)]) if bit]
    i = 2 + len(BUTTONS)
    return {
        "move": [side, forward],
        "buttons": pressed,
        "slot": int(a[i]),
        "look": [YAW_BINS[int(a[i + 1])], PITCH_BINS[int(a[i + 2])]],
    }


def noop_action() -> np.ndarray:
    a = np.zeros(len(ACTION_NVEC), dtype=np.int64)
    a[0] = a[1] = 1
    a[-2] = YAW_BINS.index(0.0)
    a[-1] = PITCH_BINS.index(0.0)
    return a


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

NUM_ENEMY_TYPES = 43  # EnemyType enum values 0..42 in the current game build
NUM_WEAPON_SLOTS = 6


@dataclass
class ObsLayout:
    max_enemies: int = 8
    horizontal_rays: int = 16
    ground_rays: int = 8
    ray_length: float = 50.0
    ground_ray_length: float = 30.0

    @property
    def player_size(self) -> int:
        # local_vel, hp/anti_hp/stamina, grounded/sliding/pitch, weapon slot one-hot, yaw sin/cos
        return 3 + 3 + 3 + NUM_WEAPON_SLOTS + 2

    @property
    def enemy_size(self) -> int:
        return 3 + 1 + 1 + 1 + 1 + NUM_ENEMY_TYPES  # rel, dist, health, visible, mask, type

    @property
    def mode_size(self) -> int:
        return 2 + 5  # cybergrind (wave, enemies_left) + campaign (waypoint rel xyz, dist, progress)

    @property
    def size(self) -> int:
        return (
            self.player_size
            + self.max_enemies * self.enemy_size
            + self.horizontal_rays
            + self.ground_rays
            + self.mode_size
        )

    def mod_config(self) -> dict[str, Any]:
        """Settings sent to the mod so it produces what this layout expects."""
        return {
            "max_enemies": self.max_enemies,
            "horizontal_rays": self.horizontal_rays,
            "ground_rays": self.ground_rays,
            "ray_length": self.ray_length,
            "ground_ray_length": self.ground_ray_length,
        }

    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.size,), dtype=np.float32)


def yaw_frame(vec_world: tuple[float, float, float], yaw_deg: float) -> tuple[float, float, float]:
    """Rotates a world-space vector into the player's yaw frame (x = right, y = up, z = forward)."""
    y = math.radians(yaw_deg)
    dx, dy, dz = vec_world
    return (dx * math.cos(y) - dz * math.sin(y), dy, dx * math.sin(y) + dz * math.cos(y))


def pack_observation(
    obs: dict[str, Any],
    layout: ObsLayout,
    enemy_max_health: dict[int, float],
    waypoint: tuple[float, float, float] | None = None,
    route_progress: float = 0.0,
) -> np.ndarray:
    out = np.zeros(layout.size, dtype=np.float32)
    p = obs.get("player")
    if not p:
        return out

    i = 0

    def put(values) -> None:
        nonlocal i
        n = len(values)
        out[i : i + n] = values
        i += n

    lv = p["local_vel"]
    put([lv[0] / 30.0, lv[1] / 30.0, lv[2] / 30.0])
    put([p["hp"] / 100.0, p["anti_hp"] / 100.0, p["stamina"] / 300.0])
    put([float(p["grounded"]), float(p["sliding"]), p["pitch"] / 90.0])
    slot = np.zeros(NUM_WEAPON_SLOTS, dtype=np.float32)
    if 0 <= p["weapon_slot"] < NUM_WEAPON_SLOTS:
        slot[p["weapon_slot"]] = 1.0
    put(slot)
    yaw = math.radians(p["yaw"])
    put([math.sin(yaw), math.cos(yaw)])

    enemies = obs.get("enemies", [])[: layout.max_enemies]
    for k in range(layout.max_enemies):
        block = np.zeros(layout.enemy_size, dtype=np.float32)
        if k < len(enemies):
            e = enemies[k]
            rel = e["rel"]
            max_hp = max(enemy_max_health.get(e["id"], e["health"]), 1e-3)
            block[0:3] = (rel[0] / 50.0, rel[1] / 50.0, rel[2] / 50.0)
            block[3] = e["dist"] / 100.0
            block[4] = e["health"] / max_hp
            block[5] = float(e["visible"])
            block[6] = 1.0
            if 0 <= e["type"] < NUM_ENEMY_TYPES:
                block[7 + e["type"]] = 1.0
        put(block)

    rays = obs.get("rays", [])[: layout.horizontal_rays]
    put(np.pad(np.asarray(rays, dtype=np.float32) / layout.ray_length, (0, layout.horizontal_rays - len(rays))))
    ground = obs.get("ground_rays", [])[: layout.ground_rays]
    put(np.pad(np.clip(np.asarray(ground, dtype=np.float32) / layout.ground_ray_length, 0.0, 1.0), (0, layout.ground_rays - len(ground))))

    cg = obs.get("cybergrind")
    put([cg["wave"] / 30.0, max(cg["enemies_left"], 0) / 30.0] if cg else [0.0, 0.0])

    if waypoint is not None:
        pos = p["pos"]
        delta = (waypoint[0] - pos[0], waypoint[1] - pos[1], waypoint[2] - pos[2])
        lx, ly, lz = yaw_frame(delta, p["yaw"])
        dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        put([lx / 50.0, ly / 50.0, lz / 50.0, dist / 100.0, route_progress])
    else:
        put([0.0] * 5)

    assert i == layout.size, f"packed {i} values, layout expects {layout.size}"
    return out
