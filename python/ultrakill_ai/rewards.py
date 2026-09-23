"""Reward functions. All weights live in RewardConfig so they can be tuned from a YAML file."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ultrakill_ai.times import valid_official_seconds


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
    # THE PIT (2026-09-22). Charged per DECISION on which no ground lay within `layout.ground_ray_length`
    # (30 m) beneath the player -- the env's own `_oob_steps` condition, the same count `info["oob_frac"]`
    # reports, so the reward channel and the metric it is judged by cannot drift apart. 0.0 is every run
    # before this and every complete stage; only a speed stage's `speed.rewards:` block moves it.
    oob: float = 0.0
    # FALLS THAT COST HEALTH (2026-09-22). Charged per HP that a game-side rescue teleport removed: a
    # non-instakill `DeathZone` (the pit, the shaft) hurts the player by min(50, hp - 1) and puts them back on
    # the walkway, so two falls leave them on 1 HP and the next hit kills. The env measures it
    # (`CampaignStep.rescue_hp`, reported per episode as `rescue_hp`). On top of `damage_taken`, which already
    # charges the same HP at its own weight. 0.0 everywhere but a speed stage's `speed.rewards:` block.
    fall_hp: float = 0.0
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

    With NO TARGET -- every run that is not a speed stage -- this is `cfg.level_complete` and nothing else.

    WITH A TARGET AND NO USABLE OFFICIAL TIME it is the FLOOR, `SPEED_BONUS_MIN * cfg.level_complete`
    (2026-09-20). It used to be the full unscaled weight, and that was a hole in the whole mechanism: a
    completion whose timer the game never reported paid 100 while a genuine completion slower than the target
    paid 39-99, so the best-paying completion available to the policy was one with no clock at all. Such
    frames are real and not rare enough to ignore -- the 2026-09-19 incident that produced
    `times.valid_official_seconds` was a 4,120-decision episode whose completion frame arrived after the
    game's level stats had reset -- and nothing about them says the level was played fast. The floor is what
    the slowest conceivable genuine completion approaches, so a timer-less one can never out-earn a genuine
    one, and it still pays far more than not finishing, which is the safety property below. Both arguments go
    through `times.valid_official_seconds`, the same predicate the leaderboard and the driver's rule use, so a
    0.0 or a NaN cannot slip past as a "time".

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
    target = valid_official_seconds(target_seconds)
    if target is None:
        return cfg.level_complete  # not a speed stage: nothing about this call has changed
    official = valid_official_seconds(official_seconds)
    if official is None:
        return cfg.level_complete * SPEED_BONUS_MIN
    ratio = target / official
    scale = min(ratio, SPEED_BONUS_MAX) if ratio >= 1.0 else SPEED_BONUS_MIN + (1.0 - SPEED_BONUS_MIN) * ratio
    return cfg.level_complete * scale


def damage_share(removed: float, max_health: float | None, fallback: float | None) -> float:
    """One enemy's contribution to `damage_dealt` this step: the fraction of its health bar removed, in [0, 1].

    THE CLAMP IS THE POINT (2026-09-20). Both credit paths used to divide by
    `max(enemy_max_health.get(id, <a health value>), 1e-3)`, and when the id was missing from
    `enemy_max_health` AND the health value was NEGATIVE -- an overkilled enemy, which the game does report --
    `max()` returned the 1e-3 rather than the negative number and the term became health x 1000. Measured on
    the live 0-2 speed stage: one decision in `runs/probe_0-2_speed/rollout_11.jsonl` paid
    `damage_dealt = -250.0` (health -0.5, weight 0.5), and 71 of 2,068 completions (3.43%) had a NEGATIVE total
    episode reward, worst -1,119.5 -- 4-20x the whole death channel, unbounded, and firing during exactly the
    arena fighting a gated level forces. A per-enemy share can only ever be "none of its bar" to "all of its
    bar", so that is what this returns.

    `max_health` is the enemy's own bar when the env has seen one. `fallback` is what each caller used before
    this function existed -- the previous frame's health, or the vanished enemy's last health -- and it is used
    only when the bar is unknown, so the ordinary case is arithmetically identical to what it always was. With
    neither usable and real damage observed, the share is a whole bar: bounded, and never a credit for damage
    that did not happen (`removed <= 0` pays nothing).
    """
    try:
        taken = float(removed)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(taken) or taken <= 0.0:
        return 0.0
    for denominator in (max_health, fallback):
        try:
            bar = float(denominator)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(bar) and bar > 0.0:
            return min(1.0, taken / bar)
    return 1.0


def landed_ssj_bucket(report: Any) -> int:
    """The accepted SSJ bucket of a macro report as an int, 1-3, or 0 when the macro did not land.

    The ONE reading of the bucket: `macro_landed` (the S8 gate) and `spaces.tech_block` (obs index 521, which
    packs this value / 3) both call it, so the gate and the observation can never disagree. `int()` truncates a
    fractional value the way the mod's own `(int)` cast does; a missing, non-numeric, NaN or infinite bucket
    (`int(inf)` raises OverflowError) is 0, never an exception inside a training worker.
    """
    if not isinstance(report, dict) or report.get("result") != "ran" or not report.get("ssj_landed"):
        return 0
    try:
        bucket = int(report.get("ssj_bucket"))
    except (TypeError, ValueError, OverflowError):
        return 0
    return bucket if 1 <= bucket <= 3 else 0


def macro_landed(report: Any) -> bool:
    """THE S8 GATE (spec §4.5): the mod RAN a macro and its own instrument reports an accepted SSJ bucket.

    Accepted means 1-3: `TrySSJ` computes `(int)(dt / 0.008)` and rejects bucket 0 and anything at or past
    `ssjMaxFrames` (4). Never a speed delta -- that gate is perversely signed, because `TrySSJ` OVERWRITES the
    velocity with `velocityAfterSlide` (floored at 24) plus the bonus, so a speed bar pays most when the player
    is slow. `ssj_bucket` / `ssj_landed` are populated only when `result == "ran"` (docs/protocol.md, the mod
    review's finding 3); `result` is checked again here so a refused macro can never pay or read as landed.
    """
    return landed_ssj_bucket(report) > 0


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
    oob_steps: int = 0  # 1 when the ground ray ran its full length and found nothing: off the map, or falling
    rescue_hp: float = 0.0  # HP a rescue teleport removed this step (UltrakillEnv._note_rescue); 0 otherwise


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
        # Beside `time` on purpose: both are clock charges, and both must be paid on a frame that carries no
        # player -- a fall does not stop the game's timer. `oob_steps` is 0 or 1, so this term is non-positive
        # in every state and at every weight: there is no input that makes it pay.
        r.add("oob", -cfg.oob * campaign.oob_steps)
        # A rescue's HP, clamped at 0 so no input can make it pay. It is 0 on a death step and on a respawn (the
        # env never measures those), so a death cannot re-pay anything, and a fall at 1 HP costs nothing more.
        r.add("fall_hp", -cfg.fall_hp * max(0.0, campaign.rescue_hp))
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
            dealt += damage_share(before - e["health"], enemy_max_health.get(e["id"]), before)

    new_kills = max(0, cs.get("kills", 0) - ps.get("kills", 0))
    if new_kills:
        # Enemies killed in one hit vanish before a health drop is ever observed. Credit the health they
        # had left, nearest first, for as many enemies as the kill counter went up. An OVERKILLED enemy
        # reports negative health here, which is the -250 defect `damage_share` exists to bound.
        vanished = [e for e in prev.get("enemies", []) if e["id"] not in cur_ids]
        for e in vanished[:new_kills]:
            dealt += damage_share(e["health"], enemy_max_health.get(e["id"]), e["health"])
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
