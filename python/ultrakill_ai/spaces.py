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
LOOK_MODES = 3  # 0 free look, 1 nearest visible enemy, 2 the current route target (campaign only)

# [move_forward, move_side, *buttons, weapon, yaw, pitch]
BASE_NVEC = (3, 3, *([2] * len(BUTTONS)), NUM_WEAPON_CHOICES, len(YAW_BINS), len(PITCH_BINS))
ACTION_NVEC = np.array(BASE_NVEC, dtype=np.int64)  # 11 dims, 42 logits: Cyber Grind, unchanged
# Campaign only, with the look mode appended so every existing logit row keeps its index. Widening the Cyber
# Grind vector would make its checkpoints unloadable against their own env, and look mode 1 is an auto-aim:
# handing it to the run that has spent 6.5M steps learning to aim would short-circuit the thing it is learning.
ACTION_NVEC_CAMPAIGN = np.array((*BASE_NVEC, LOOK_MODES), dtype=np.int64)  # 12 dims, 45 logits
LOOK_MODE_INDEX = len(BASE_NVEC)


def action_space(campaign: bool = False) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC)


def decode_action(a: np.ndarray) -> dict[str, Any]:
    """The mod command for one action. The width tells the mode apart: 12 values carry a look mode, 11 do not."""
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
        # env.step resolves this and pops it: it is a Python-side look policy, never sent to the mod.
        "look_mode": int(a[LOOK_MODE_INDEX]) if len(a) > LOOK_MODE_INDEX else 0,
    }


def noop_action(campaign: bool = False) -> np.ndarray:
    """Stand still and look straight ahead. Explicit indices: negative ones address the wrong slots at width 12."""
    nvec = ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC
    a = np.zeros(len(nvec), dtype=np.int64)
    a[0] = a[1] = 1
    i = 2 + len(BUTTONS)
    a[i + 1] = YAW_BINS.index(0.0)
    a[i + 2] = PITCH_BINS.index(0.0)
    return a  # the look-mode slot stays 0 (free look)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

NUM_ENEMY_TYPES = 43  # EnemyType enum values 0..42 in the current game build
NUM_WEAPON_SLOTS = 6
# `GunControl.currentVariationIndex`, which the mod already sends as `player.weapon_variation` and nothing has
# ever read. Three per slot: `CampaignPatches.OverridePrefInt` rewrites every `weapon.*` pref to 1, which
# equips the STANDARD variant of all three weapons in each slot (spec §3.5), and pressing the slot already
# held cycles between them (`WeaponRedrawBehaviour` 0, spec §3.4). Bookkeeping only -- no observation packs it.
NUM_WEAPON_VARIATIONS = 3
CAMPAIGN_BLOCK = 36  # campaign values packed after the Cyber Grind two (see campaign_block)


@dataclass
class ObsLayout:
    max_enemies: int = 8
    horizontal_rays: int = 16
    ground_rays: int = 8
    ray_length: float = 50.0
    ground_ray_length: float = 30.0
    campaign: bool = False  # campaign levels: the level block replaces the 5 retired route values (448 -> 479)

    @property
    def player_size(self) -> int:
        # local_vel, hp/anti_hp/stamina, grounded/sliding/pitch, weapon slot one-hot, yaw sin/cos
        return 3 + 3 + 3 + NUM_WEAPON_SLOTS + 2

    @property
    def enemy_size(self) -> int:
        return 3 + 1 + 1 + 1 + 1 + NUM_ENEMY_TYPES  # rel, dist, health, visible, mask, type

    @property
    def mode_size(self) -> int:
        # Cyber Grind (wave, enemies_left), then the campaign block, or 5 zeros where the retired route waypoint
        # used to be, so 448-input Cyber Grind checkpoints still load.
        return 2 + (CAMPAIGN_BLOCK if self.campaign else 5)

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


def campaign_block(obs: dict[str, Any], explore: list[float] | None = None, target: dict | None = None) -> list[float]:
    """The 36 campaign values, all zero without a `campaign` block or a player.

    Targets are relative to the player in its yaw frame (x right, y up, z forward), so "ahead" is +z whichever way
    the player faces.

        0-4    exit: rel xyz / 100, distance / 200, mask
        5-8    route target (GateProgress.target): rel xyz and 3-D distance, gate scales 50/100, exit scales
               100/200 -- the exit is ~195 m from 0-1's spawn, which the gate scales would put near 4.0. All four
               are clipped to +-4.0 as a guard
        9      target mask: 1.0 when a target exists
        10-12  target gate `open`, `locked`, min(`hops`, 20) / 20 -- all 0.0 when the target is the exit or absent
        13-17  nearest checkpoint neither activated nor current: rel xyz / 100, distance / 200, mask
        18-22  first locked door (the mod sends them nearest first): rel xyz / 50, distance / 100, mask
        23-26  arena enemies alive / 20, timer running, input locked, level seconds / 600
        27-35  exploration map (ExplorationArchive.features: current cell, then 8 neighbours from straight ahead)

    Slots 5-12 carried the NavMesh path hint until the gates work. That hint never once read `complete` in
    2.9M steps, so it was replaced in place rather than appended: every other index keeps its meaning, the vector
    stays 479 long and the run continues from its own weights (scripts/add_look_mode.py zeroes those eight
    first-layer columns and folds their mean into the bias). `target` is threaded in from the env because
    GateProgress lives there; a target with no `hops` is the exit sentinel.
    """
    out = [0.0] * CAMPAIGN_BLOCK
    c, p = obs.get("campaign"), obs.get("player")
    if not c or not p:
        return out
    pos, yaw = p["pos"], p["yaw"]

    def relative(point, rel_scale: float, dist_scale: float) -> list[float]:
        x, y, z = yaw_frame((point[0] - pos[0], point[1] - pos[1], point[2] - pos[2]), yaw)
        return [x / rel_scale, y / rel_scale, z / rel_scale, math.sqrt(x * x + y * y + z * z) / dist_scale]

    exit_ = c.get("exit")
    if exit_:
        out[0:5] = relative(exit_["pos"], 100.0, 200.0) + [1.0]

    if target and target.get("pos"):
        is_exit = target.get("hops") is None
        scales = (100.0, 200.0) if is_exit else (50.0, 100.0)
        out[5:9] = [max(-4.0, min(4.0, v)) for v in relative(target["pos"], *scales)]
        out[9] = 1.0
        if not is_exit:
            out[10] = float(bool(target.get("open")))
            out[11] = float(bool(target.get("locked")))
            # Bounded, so slot 12 cannot read above 1.0 whatever ships in a route file: a 25-rung trunk would
            # otherwise pin a learned input column. A proven no-op on everything that exists -- the deepest
            # gate ladder in the campaign is 13 hops and the longest room trunk 14 (spec §9).
            out[12] = min(float(target["hops"]), 20.0) / 20.0

    pending = [cp for cp in (c.get("checkpoints") or []) if not cp["activated"] and not cp["current"]]
    if pending:
        nearest = min(pending, key=lambda cp: math.dist(cp["pos"], pos))
        out[13:18] = relative(nearest["pos"], 100.0, 200.0) + [1.0]

    doors = c.get("locked_doors") or []
    if doors:
        out[18:23] = relative(doors[0]["pos"], 50.0, 100.0) + [1.0]

    out[23:27] = [
        c.get("arena_enemies_alive", 0) / 20.0,
        float(bool(c.get("timer_running"))),
        float(bool(c.get("input_locked"))),
        c.get("seconds", 0.0) / 600.0,
    ]
    if explore is not None:
        values = [float(v) for v in explore[:9]]
        out[27 : 27 + len(values)] = values
    return out


def pack_observation(
    obs: dict[str, Any],
    layout: ObsLayout,
    enemy_max_health: dict[int, float],
    explore: list[float] | None = None,
    target: dict | None = None,
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
    # THE WEAPON-SLOT ONE-HOT IS INDEXED BY THE RAW FIELD, AND MUST STAY THAT WAY. `player.weapon_slot` is
    # `GunControl.currentSlotIndex`, which is 1-BASED (see `env.held_slot_key`), so slot KEY k lights index k
    # and index 0 is never set -- a permanently dead input, not an off-by-one. The mapping is still one-to-one
    # over every value the policy can cause (keys 1..5, plus -1 "GunControl has not started" -> all zeros), so
    # the packing loses nothing; the only collision is key 6, which reads as all-zeros, and nothing can select
    # it (`NUM_WEAPON_CHOICES` stops the action at key 5 and slot 6 ships empty).
    # Every policy ever trained learned THIS mapping. Re-basing it to `weapon_slot - 1` would move five input
    # features under the live weights and silently break them, which is why the 2026-09-20 fix to the 1-based
    # convention deliberately stopped at the env's bookkeeping and left these four lines alone.
    # `tests/test_spaces.py::test_the_weapon_slot_one_hot_is_indexed_by_the_raw_1_based_field` pins it.
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

    # Cyber Grind keeps 5 zeros where the route waypoint used to be, so its 448-input checkpoints still load.
    put(campaign_block(obs, explore, target) if layout.campaign else [0.0] * 5)

    assert i == layout.size, f"packed {i} values, layout expects {layout.size}"
    return out
