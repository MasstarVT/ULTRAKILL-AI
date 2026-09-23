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

from ultrakill_ai.rewards import landed_ssj_bucket

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

# ---------------------------------------------------------------------------
# The v2 TECH action layout -- stage S7 of docs/superpowers/specs/2026-09-20-speedrun-tech.md, §4.1.
# APPEND ONLY: every campaign logit row 0-44 keeps its index, which is what lets scripts/add_tech_heads.py copy
# rows 0-44 verbatim and makes the migration exact.
# ---------------------------------------------------------------------------
TECH_LAYOUTS = ("v1", "v2")  # EnvConfig.tech_layout; ultrakill_ai/ckpt_layout.py carries the same literal
MACROS = ("none", "ssj", "ssj_wall", "core_nuke", "rocket_down", "coin_rocket")  # the mod's own values 0..5
VARIANT_CHOICES = 4  # 0 keep, 1..3 = variation 0..2 of the held slot (the mod's `variant`)
HOOK_CHOICES = 2  # 0 off, 1 hold the whiplash (the mod's `hook` HoldButton)
ACTION_NVEC_TECH = np.array((*ACTION_NVEC_CAMPAIGN, len(MACROS), VARIANT_CHOICES, HOOK_CHOICES),
                            dtype=np.int64)  # 15 dims, 57 logits
MACRO_INDEX, VARIANT_INDEX, HOOK_INDEX = 12, 13, 14
_CAMPAIGN_LOGITS = int(ACTION_NVEC_CAMPAIGN.sum())  # 45
MACRO_ROWS = range(_CAMPAIGN_LOGITS, _CAMPAIGN_LOGITS + len(MACROS))  # 45-50
VARIANT_ROWS = range(MACRO_ROWS.stop, MACRO_ROWS.stop + VARIANT_CHOICES)  # 51-54
HOOK_ROWS = range(VARIANT_ROWS.stop, VARIANT_ROWS.stop + HOOK_CHOICES)  # 55-56


def action_space(campaign: bool = False, tech: bool = False) -> spaces.MultiDiscrete:
    if tech and not campaign:
        raise ValueError("the tech action layout is campaign-only: Cyber Grind's 11-dimension space never widens")
    return spaces.MultiDiscrete(ACTION_NVEC_TECH if tech else ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC)


def decode_action(a: np.ndarray) -> dict[str, Any]:
    """The mod command for one action. The width tells the mode apart: 15 values carry the tech heads, 12 a look
    mode, 11 neither. A 12- or 11-wide action decodes to exactly the dict it always did (tests/test_tech_layout.py)."""
    a = np.asarray(a, dtype=np.int64)
    # Any other width is a layout mismatch: a 13- or 14-wide action would otherwise decode as a 12-wide one and
    # silently drop the tech heads.
    assert len(a) in (len(ACTION_NVEC), len(ACTION_NVEC_CAMPAIGN), len(ACTION_NVEC_TECH)), (
        f"an action is 11, 12 or 15 values wide, got {len(a)}")
    forward = int(a[0]) - 1
    side = int(a[1]) - 1
    pressed = [name for name, bit in zip(BUTTONS, a[2 : 2 + len(BUTTONS)]) if bit]
    i = 2 + len(BUTTONS)
    out = {
        "move": [side, forward],
        "buttons": pressed,
        "slot": int(a[i]),
        "look": [YAW_BINS[int(a[i + 1])], PITCH_BINS[int(a[i + 2])]],
        # env.step resolves this and pops it: it is a Python-side look policy, never sent to the mod.
        "look_mode": int(a[LOOK_MODE_INDEX]) if len(a) > LOOK_MODE_INDEX else 0,
    }
    if len(a) > HOOK_INDEX:
        # v2 only. env.step pops all three and decides what reaches the wire (UltrakillEnv._apply_tech): a raw
        # policy value is never sent as is.
        out["macro"] = int(a[MACRO_INDEX])
        out["variant"] = int(a[VARIANT_INDEX])
        out["hook"] = bool(a[HOOK_INDEX])
    return out


def noop_action(campaign: bool = False, tech: bool = False) -> np.ndarray:
    """Stand still and look straight ahead. Explicit indices: negative ones address the wrong slots at width 12+."""
    nvec = ACTION_NVEC_TECH if tech else ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC
    a = np.zeros(len(nvec), dtype=np.int64)
    a[0] = a[1] = 1
    i = 2 + len(BUTTONS)
    a[i + 1] = YAW_BINS.index(0.0)
    a[i + 2] = PITCH_BINS.index(0.0)
    return a  # look mode 0 (free look); under v2 also no macro, keep the variant, no hook


def pinned_action_rows(live_macros, variant: bool, hook: bool) -> list[int]:
    """The v2 logit rows whose value can never reach the game under these gates, ascending.

    training.pin_action_rows freezes them (the plan's "Spec deviations" 2): a head that cannot act receives only
    the entropy bonus and would drift toward uniform, losing the §4.4 prior before its stage opens. A dimension
    with NO live non-default value is pinned whole; otherwise only its dead values are.

    `live_macros` holds macro VALUES 1-5 (`MACROS[1:]`); 0 "none" is never a gate. Anything else -- 6, -1, a
    name, a float, a bool -- raises, where it used to be dropped silently and pin the whole macro dimension.
    """
    live = set()
    for v in live_macros:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 < v < len(MACROS):
            raise ValueError(f"a live macro is an int 1-{len(MACROS) - 1} ({', '.join(MACROS[1:])}), got {v!r}")
        live.add(int(v))
    rows = list(MACRO_ROWS) if not live else [MACRO_ROWS[v] for v in range(1, len(MACROS)) if v not in live]
    if not variant:
        rows += list(VARIANT_ROWS)
    if not hook:
        rows += list(HOOK_ROWS)
    return rows


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
# The v2 TECH block (§4.3): 51 floats appended AFTER the campaign block, so indices 0-478 keep their meaning.
# Offsets inside the block; the absolute start is ObsLayout.tech_start (479 at the defaults).
TECH_BLOCK = 51
TECH_MOVE = slice(0, 12)  # block A, live at S7           -> 479-490
TECH_WEAPON = slice(12, 22)  # block B, reserved 0.0 (S9)   -> 491-500
TECH_PROJECTILES = slice(22, 40)  # block C, reserved (S10) -> 501-518
TECH_MACRO = slice(40, 43)  # block D, live at S7          -> 519-521
TECH_HAZARD = slice(43, 51)  # block E, reserved (S11)     -> 522-529
# Block A in packing order: (mod key, divisor, low, high). The scales come from the decompiled NewMovement and
# are DERIVED, not measured -- the S6 slam check reads the real ranges. `slide_grace`, NOT `slide_since`: the
# mod review's finding 2 (2026-09-20). Never pack slide_since, slide_timestamp or jump_timestamp (docs/protocol.md).
MOVE_TECH_FIELDS = (
    ("heavy_fall", 1.0, 0.0, 1.0),
    ("slam_force", 10.0, 0.0, 1.0),  # 1 at slam start, +5/s while heavyFall; >= 5.5 is the 12.5x bounce
    ("bounce_window", 1.0, 0.0, 1.0),
    ("coyote", 1.0, 0.0, 1.0),  # gc.sinceLastGrounded, game seconds (999 without a ground check)
    ("wall_jumps", 3.0, 0.0, 1.0),  # currentWallJumps; the budget is 3
    ("wall_available", 1.0, 0.0, 1.0),
    ("boost", 1.0, 0.0, 1.0),
    ("boost_left", 100.0, 0.0, 1.0),  # dash i-frames: 100 at Dodge(), -4 per fixed step
    ("pre_slide_speed", 3.0, 0.0, 2.0),  # |v|/24 or slamForce; StartSlide clamps it to 3
    ("jump_cooldown", 1.0, 0.0, 1.0),
    ("slide_grace", 1.0, 0.0, 1.0),  # the fraction of the SSJ window still open
    ("riding_rocket", 1.0, 0.0, 1.0),
)


@dataclass
class ObsLayout:
    max_enemies: int = 8
    horizontal_rays: int = 16
    ground_rays: int = 8
    ray_length: float = 50.0
    ground_ray_length: float = 30.0
    campaign: bool = False  # campaign levels: the level block replaces the 5 retired route values (448 -> 479)
    tech: bool = False  # tech_layout v2: the 51-float TECH block (479 -> 530)

    def __post_init__(self) -> None:
        # The TECH block follows the campaign block: without one, `tech_start` would say 479 while
        # pack_observation put the block at 448.
        if self.tech and not self.campaign:
            raise ValueError("the tech observation layout is campaign-only: it is appended after the campaign block")

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
    def campaign_start(self) -> int:
        """Absolute index of the campaign block's first value (443 at the defaults)."""
        return self.player_size + self.max_enemies * self.enemy_size + self.horizontal_rays + self.ground_rays + 2

    @property
    def tech_start(self) -> int:
        """Absolute index of the v2 TECH block (479 at the defaults). Meaningful only when `tech` is set."""
        return self.campaign_start + CAMPAIGN_BLOCK

    @property
    def size(self) -> int:
        return (
            self.player_size
            + self.max_enemies * self.enemy_size
            + self.horizontal_rays
            + self.ground_rays
            + self.mode_size
            + (TECH_BLOCK if self.tech else 0)
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


def _scalar(value: Any) -> float:
    """A mod field as a float: a bool is 0/1; missing, null, non-numeric or non-finite is 0.0 (the old-DLL value)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def tech_block(obs: dict[str, Any]) -> list[float]:
    """The 51 v2 values (index map: docs/superpowers/plans/2026-09-23-tech-break-python.md).

    Every read defaults to 0.0, so a 0.7.2 DLL, an `obs_move_tech` that is off, or a frame with no player all
    pack the vector scripts/add_tech_heads.py initialised the new input columns for. Blocks B, C and E are
    reserved and stay 0.0 even if the mod sends their source blocks: their packers are written at S9-S11.
    """
    out = [0.0] * TECH_BLOCK
    move = obs.get("move_tech") if obs.get("player") else None
    if isinstance(move, dict):
        for k, (name, divisor, low, high) in enumerate(MOVE_TECH_FIELDS):
            out[TECH_MOVE.start + k] = min(high, max(low, _scalar(move.get(name)) / divisor))
    report = obs.get("macro")
    if isinstance(report, dict):
        ran = report.get("result") == "ran"
        out[TECH_MACRO.start] = 1.0 if ran else 0.0
        out[TECH_MACRO.start + 1] = 0.0 if ran else 1.0
        out[TECH_MACRO.start + 2] = landed_ssj_bucket(report) / 3.0  # the S8 gate's own int bucket, 0 if none
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

    if layout.tech:
        put(tech_block(obs))

    assert i == layout.size, f"packed {i} values, layout expects {layout.size}"
    return out
