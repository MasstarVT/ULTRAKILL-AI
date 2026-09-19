"""Reward functions. All weights live in RewardConfig so they can be tuned from a YAML file."""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    punch: float = 0.0  # charged per decision that presses the punch button (see compute_reward)

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

    # Campaign. The new terms default to 0, so Cyber Grind runs are unchanged; campaign configs set them.
    level_complete: float = 100.0  # once, on the step the level ends (both modes)
    time: float = 0.0  # charged per decision, so finishing sooner always pays
    checkpoint: float = 0.0  # per checkpoint first reached in a level load
    arena_clear: float = 0.0  # per arena whose last wave died, once per level load
    door_unlock: float = 0.0  # per door unlocked by play, once per level load (respawn unlocks never pay)
    novelty: float = 0.0  # times CampaignStep.novelty, the summed 1/sqrt(N+1) of cells new this episode
    path: float = 0.0  # per metre of new best NavMesh distance to the exit
    # The door-graph route (campaign.gates): the signal the NavMesh never gave, since `path.status` was never
    # once `complete` in 2.9M logged steps. `gate` is the milestone, `gate_approach` the shaping between them.
    gate: float = 0.0  # per new lower `hops` value reached, once per level load
    gate_approach: float = 0.0  # per metre of new best 3-D closeness to the current target
    # Skull carry: the two rungs below a gate whose door is held shut by an unfilled altar. Both are once per
    # level load and keyed on the item TYPE (pickup) or the puzzle (placement), never on an object instance, so
    # neither a respawn's re-instantiated skull nor a level's duplicate props can pay twice. See MilestoneTracker.
    item_pickup: float = 0.0  # per accepted item type first picked up in a level load
    item_placed: float = 0.0  # per altar puzzle first solved in a level load


# The speed stage's time-scaled completion bonus (docs/superpowers/specs/2026-09-18-speed-stages.md §2).
# The ceiling stops a bogus near-zero official time paying unbounded reward; the FLOOR is the safety property
# that matters -- a completion must never be worth less than staying in the level, whatever the clock says,
# because the first campaign run spent 2.5M steps proving that a policy will refuse to terminate an episode
# whose continuation pays better. At `level_complete: 100` the band is 25..200.
SPEED_BONUS_MIN = 0.25
SPEED_BONUS_MAX = 2.0


def completion_bonus(cfg: "RewardConfig", target_seconds: float | None = None,
                     official_seconds: float | None = None) -> float:
    """What one level completion pays: the plain weight, or it scaled by the clock.

    With either argument missing or non-positive this is `cfg.level_complete` and nothing else, so every run
    that is not a speed stage -- and every completion inside one that has no official time (a frame that
    arrived after the campaign block was gone) -- is unchanged.

    THE FLOOR IS APPROACHED, NEVER REACHED (2026-09-18 review). A hard `max(target/official, 0.25)` clip is
    flat wherever the policy is more than 4x its target, and that is exactly where every policy starts: over
    the 68 fresh Level 0-1 completions in the live `spec_0-1` log the median is 481.9 s against a 90 s target,
    so 58 of 68 (85%) sat precisely ON the clip -- a flat 75% pay cut with d(bonus)/d(time) == 0, which pays
    less for the behaviour the policy already has and says nothing about how to improve it. Below the target
    the scale is still `target/official` (the spec's own number, so the meaning at and under the target is
    unchanged); above it the ratio is mapped into `(MIN, 1]` instead of being clipped into `[MIN, 1]`:

        scale = MIN + (1 - MIN) * target/official

    which is 1.0 at exactly the target, strictly decreasing for every slower completion, and strictly greater
    than MIN at any finite time. The floor's safety property is therefore stronger than it was -- a completion
    can never pay less than `MIN * level_complete` -- while a slower completion always pays strictly less than
    a faster one. At `level_complete: 100` and a 90 s target: 482 s pays 39.0, 243 s pays 52.8, 150 s pays
    70.0, 90 s pays 100.0. NOT VALIDATED IN GAME: no speed stage has been trained with this.
    """
    if not target_seconds or not official_seconds or target_seconds <= 0.0 or official_seconds <= 0.0:
        return cfg.level_complete
    ratio = target_seconds / official_seconds
    scale = min(ratio, SPEED_BONUS_MAX) if ratio >= 1.0 else SPEED_BONUS_MIN + (1.0 - SPEED_BONUS_MIN) * ratio
    return cfg.level_complete * scale


@dataclass
class CampaignStep:
    """What the level did this step, measured by the env (see campaign.py)."""

    checkpoints: int = 0
    arenas: int = 0
    doors: int = 0
    novelty: float = 0.0  # sum of 1/sqrt(N+1) over cells entered for the first time this episode
    path_gain: float = 0.0  # metres of new best NavMesh distance to the exit
    gates: int = 0  # new lower `hops` values reached this step (GateProgress.update)
    gate_approach: float = 0.0  # metres of new best closeness to the current gate/exit target
    item_pickups: int = 0  # accepted item types picked up for the first time this level load
    item_placements: int = 0  # altar puzzles solved for the first time this level load


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
    *,
    died: bool | None = None,
    campaign: CampaignStep | None = None,
    buttons: Sequence[str] = (),
    target_seconds: float | None = None,
    official_seconds: float | None = None,
) -> RewardResult:
    r = RewardResult()
    ps, cs = prev.get("stats", {}), cur.get("stats", {})
    if cs.get("level_complete") and not ps.get("level_complete"):
        # Paid before the player check, on the rising edge. The frame that ends a level arrives while the scene
        # is unloading and often has no player, and returning early there would pay nothing for the one outcome
        # the campaign run is for. It reads only the stats, so it is safe this early.
        # `target_seconds`/`official_seconds` are a SPEED STAGE's two numbers and are None everywhere else, in
        # which case `completion_bonus` is `cfg.level_complete` exactly.
        r.add("level_complete", completion_bonus(cfg, target_seconds, official_seconds))
    if "punch" in buttons:
        # The punch button is an independent coin flip in the action space with no cost and, outside a parry, no
        # effect, so nothing ever taught the policy to stop pressing it and it flails constantly. A small charge
        # per press is a gradient it can actually act on, unlike a flat per-step cost, which the value baseline
        # absorbs. Fights the game forces still pay far more through `arena_clear`.
        r.add("punch", -cfg.punch)
    if campaign is not None:
        # Paid before the player check: the env has already marked these milestones paid, so a step that
        # arrives without a player (a level load) must not drop them.
        r.add("time", -cfg.time)
        r.add("checkpoint", cfg.checkpoint * campaign.checkpoints)
        r.add("arena_clear", cfg.arena_clear * campaign.arenas)
        r.add("door_unlock", cfg.door_unlock * campaign.doors)
        r.add("novelty", cfg.novelty * campaign.novelty)
        r.add("path", cfg.path * campaign.path_gain)
        r.add("gate", cfg.gate * campaign.gates)
        r.add("gate_approach", cfg.gate_approach * campaign.gate_approach)
        r.add("item_pickup", cfg.item_pickup * campaign.item_pickups)
        r.add("item_placed", cfg.item_placed * campaign.item_placements)

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

    return r
