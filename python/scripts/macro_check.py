"""In-game verification of mod 0.8.0's SSJ macros, on ONE private game. Never touches the fleet.

This is the S6 stage of `docs/superpowers/specs/2026-09-20-speedrun-tech.md`: the GO/NO-GO for widening
the action space. It answers, with numbers rather than inference:

1. Does an explicit `QueueStateEvent` timestamp reach `ctx.time` unchanged on Unity 2022.3.29 Mono?
2. What is `InputSystem.settings.updateMode` -- i.e. is a future-dated event processed in the current
   update, or time-sliced into a later one?
3. Does the monotonic timestamp cursor hold: after queueing at `now + 0.012`, is the NEXT frame's
   ordinary event still delivered rather than silently dropped?
4. What are `NewMovement.walkSpeed` and `Time.fixedDeltaTime`? Both are serialized, so every u/s figure
   in the design is derived rather than read.
5. Does the jump SSJ macro land a valid bucket, and what speed does it actually add against a plain
   slide jump measured by the same instrument?
6. The same for the wall SSJ.
7. Do macro steps consume the same game time as ordinary steps?
8. Do the new observation blocks read sensibly?
9. Does an OLD-style client -- one that sends none of the new fields -- see exactly the 0.7.2 message?

**Safety.** The 12 training games hold ports 47800-47811 and the bridge is single-client: connecting to
one of them drops the trainer's socket and kills that worker. This script therefore REFUSES to connect
to anything in that range, and defaults to 47812, the hand-started private port. It also never launches
or stops a game; `games.py launch` and `games.py stop` both call `stop_all()` and would take down all
twelve.

Usage (from `python/`, with a private game already listening on 47812):

    python scripts/macro_check.py --port 47812 --level "Level 0-1" --trials 25 --json out.json
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from collections import Counter
from typing import Any

FLEET_PORTS = range(47800, 47812)
PRIVATE_PORT = 47812

# The 0.7.2 top-level observation keys. Anything outside this set arriving at a client that asked for
# none of the 0.8 features is a backward-compatibility failure, not a bonus.
BASE_OBS_KEYS = {
    "type", "step", "frame", "time", "scene", "ready", "player", "enemies", "rays", "ground_rays",
    "ground_ray_center", "stats", "event", "campaign", "cybergrind", "mem",
}
# The subset of BASE_OBS_KEYS that must be present on EVERY step. The rest are conditional: `campaign`
# only in a campaign scene, `cybergrind` only in the Cyber Grind, `mem` only under report_memory, `event`
# only on a reset. Checking only for EXTRA keys would miss a regression that dropped one.
REQUIRED_OBS_KEYS = {
    "type", "step", "frame", "time", "scene", "ready", "player", "enemies", "rays", "ground_rays",
    "ground_ray_center", "stats",
}
BASE_PLAYER_KEYS = {
    "pos", "vel", "local_vel", "forward", "yaw", "pitch", "hp", "anti_hp", "stamina", "grounded",
    "sliding", "slow_mode", "heavy_fall", "crouching", "dead", "activated", "level_over",
    "weapon_slot", "weapon_variation", "slot_counts", "soft_deaths", "soft_death_instakill",
}


class BridgeError(RuntimeError):
    pass


class Bridge:
    """A minimal newline-delimited JSON client. Deliberately standalone: no ultrakill_ai import, so this
    can run beside a live trainer without touching the package another engineer is editing."""

    def __init__(self, port: int, timeout: float = 120.0):
        if port in FLEET_PORTS:
            raise SystemExit(
                f"refusing to connect to {port}: 47800-47811 are the live fleet and the bridge is "
                f"single-client, so this would kill a training worker. Use {PRIVATE_PORT}."
            )
        self.port = port
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        self.sock.settimeout(timeout)
        self.buf = b""
        self.step_times: list[float] = []

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _send(self, msg: dict) -> dict:
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise BridgeError("bridge closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        reply = json.loads(line)
        if reply.get("type") == "error":
            raise BridgeError(reply.get("message", "unknown bridge error"))
        return reply

    def hello(self) -> dict:
        return self._send({"type": "hello", "protocol": 1})

    def config(self, **kw: Any) -> dict:
        return self._send({"type": "config", **kw})

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict:
        msg: dict[str, Any] = {"type": "reset", "checkpoint": checkpoint}
        if scene:
            msg["scene"] = scene
        return self._send(msg)

    def step(self, move=(0.0, 0.0), look=(0.0, 0.0), buttons=(), slot=0,
             macro: int | None = None, variant: int | None = None) -> dict:
        action: dict[str, Any] = {
            "move": list(move), "look": list(look), "buttons": list(buttons), "slot": slot,
        }
        if macro:
            action["macro"] = macro
        if variant:
            action["variant"] = variant
        t0 = time.perf_counter()
        obs = self._send({"type": "step", "action": action})
        self.step_times.append(time.perf_counter() - t0)
        return obs

    def legacy_step(self, move=(0.0, 0.0), buttons=(), slot=0) -> dict:
        """Exactly the action a 0.7.2 client sends: no `macro`, no `variant`, no extra keys."""
        return self.step(move=move, buttons=buttons, slot=slot)


# ---------------------------------------------------------------------------------------------------
# helpers


def hspeed(obs: dict) -> float:
    p = obs.get("player") or {}
    v = p.get("vel") or [0.0, 0.0, 0.0]
    return (v[0] ** 2 + v[2] ** 2) ** 0.5


def alive(obs: dict) -> bool:
    p = obs.get("player") or {}
    return bool(p.get("activated")) and not p.get("dead")


def ssj_this_step(obs: dict, frameskip: int) -> dict | None:
    """The instrument's last SSJ attempt, if it happened inside the step this obs ends."""
    mt = obs.get("move_tech") or {}
    last = mt.get("ssj_last")
    if not last:
        return None
    if obs.get("frame", 0) - last.get("frame", -10 ** 9) >= frameskip:
        return None
    return last


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "n/a"


# ---------------------------------------------------------------------------------------------------
# trials


def face_open_space(b: Bridge) -> dict | None:
    """Turn toward the longest horizontal ray.

    Without this, repeated trials walk the player into a wall and every SSJ is then measured from a
    standstill -- which is exactly the case the spec warns pays most (velocityAfterSlide floors at 24),
    so it would flatter the macro rather than measure it.
    """
    obs = b.step(move=(0.0, 0.0))
    if not alive(obs):
        return None
    rays = obs.get("rays") or []
    if not rays:
        return obs
    n = len(rays)
    idx = max(range(n), key=lambda i: rays[i])
    yaw = 360.0 * idx / n
    if yaw > 180.0:
        yaw -= 360.0
    return b.step(look=(yaw, 0.0))


def get_sliding(b: Bridge, warm: int = 8, tries: int = 6, turn: bool = False) -> dict | None:
    """Walk forward, then hold slide, until the player is grounded AND sliding."""
    obs = None
    if turn and face_open_space(b) is None:
        return None
    for _ in range(warm):
        obs = b.step(move=(0.0, 1.0))
        if not alive(obs):
            return None
    for _ in range(tries):
        obs = b.step(move=(0.0, 1.0), buttons=["slide"])
        if not alive(obs):
            return None
        p = obs["player"]
        if p.get("sliding") and p.get("grounded"):
            return obs
    return obs if (obs and (obs.get("player") or {}).get("sliding")) else None


def settle(b: Bridge, steps: int = 12) -> dict | None:
    """Let the player land and the jump cooldown clear between trials."""
    obs = None
    for _ in range(steps):
        obs = b.step(move=(0.0, 0.0))
        if not alive(obs):
            return None
        mt = obs.get("move_tech") or {}
        if (obs["player"].get("grounded") and not mt.get("jump_cooldown")
                and not obs["player"].get("sliding")):
            return obs
    return obs


def jump_trials(b: Bridge, trials: int, frameskip: int, arm: str) -> dict:
    """`trials` attempts at a ground SSJ, measured by the same instrument on three input patterns.

    - ``macro``   -- the mod times the release and the press 12 ms apart, inside one frame.
    - ``hold``    -- jump while still HOLDING slide. SlideCancelled never fires, slideTimestamp stays
                     stale, and TrySSJ rejects the gap as far too large.
    - ``release`` -- **the pattern today's policy actually produces**: drop slide and add jump in the SAME
                     step, so both land in one KeyboardState event, `jumpTimestamp == slideTimestamp`,
                     and TrySSJ returns at its `if (!(num > 0.0)) return;`.

    ``release`` is the honest control for the macro: ``hold`` is a different action, and quoting the macro
    against it alone measures "release slide + SSJ" versus "keep sliding + ordinary jump".
    """
    out = {
        "attempted": 0, "macro_result": Counter(), "macro_reason": Counter(), "macro_note": Counter(),
        "buckets": Counter(), "landed": 0, "gains": [], "h_gains": [],
        "h_before": [], "h_after": [], "dt_ms": [], "no_event": 0,
        "obs_delta": [],
    }
    for _ in range(trials):
        if settle(b) is None:
            break
        start = get_sliding(b, turn=True)
        if start is None:
            continue
        before_obs = hspeed(start)
        out["attempted"] += 1
        if arm == "macro":
            obs = b.step(move=(0.0, 1.0), buttons=["slide"], macro=1)
            m = obs.get("macro") or {}
            out["macro_result"][m.get("result", "missing")] += 1
            if m.get("reason"):
                out["macro_reason"][m["reason"]] += 1
            if m.get("note"):
                out["macro_note"][m["note"]] += 1
        elif arm == "hold":
            obs = b.step(move=(0.0, 1.0), buttons=["slide", "jump"])
        elif arm == "release":
            obs = b.step(move=(0.0, 1.0), buttons=["jump"])
        else:
            raise ValueError(f"unknown arm {arm!r}")
        ev = ssj_this_step(obs, frameskip)
        if ev is None:
            out["no_event"] += 1
            continue
        out["buckets"][ev["bucket"]] += 1
        if ev["landed"]:
            out["landed"] += 1
        out["gains"].append(ev["gain"])
        out["h_gains"].append(ev["h_gain"])
        out["h_before"].append(ev["h_speed_before"])
        out["h_after"].append(ev["h_speed_after"])
        out["dt_ms"].append(ev["dt_ms"])
        out["obs_delta"].append(hspeed(obs) - before_obs)
    return out


def approach_wall(b: Bridge, steps: int = 40) -> dict | None:
    """Turn toward the nearest wall in the horizontal ray ring and close on it."""
    obs = b.step(move=(0.0, 0.0))
    if not alive(obs):
        return None
    rays = obs.get("rays") or []
    if not rays:
        return None
    n = len(rays)
    idx = min(range(n), key=lambda i: rays[i])
    # Ray 0 is straight ahead and the ring goes around the player.
    yaw = 360.0 * idx / n
    if yaw > 180.0:
        yaw -= 360.0
    obs = b.step(look=(yaw, 0.0))
    for _ in range(steps):
        obs = b.step(move=(0.0, 1.0))
        if not alive(obs):
            return None
        rays = obs.get("rays") or [99.0]
        if rays[0] < 3.0:
            return obs
    return None


def wall_trials(b: Bridge, trials: int, frameskip: int) -> dict:
    """Attempts at a wall SSJ: close on a wall, jump, then fire the macro while falling beside it."""
    out = {
        "attempted": 0, "fired": 0, "macro_result": Counter(), "macro_reason": Counter(),
        "buckets": Counter(), "landed": 0, "gains": [], "h_gains": [], "dt_ms": [],
        "no_event": 0, "no_wall_window": 0, "lead_s": [], "frame_gap": [],
    }
    for _ in range(trials):
        if settle(b) is None:
            break
        if approach_wall(b) is None:
            continue
        out["attempted"] += 1
        # The macro needs an AIRBORNE slide: WallJump only reaches TrySSJ through
        # `sliding || currentTime - slideTimestamp < 0.032`, and SlideCancelled only records that timestamp
        # while sliding. So start a GROUND slide and keep holding it off the edge, rather than jumping --
        # Jump()'s own `if (sliding)` branch calls StopSlide and would end the slide before the wall.
        if get_sliding(b) is None:
            out["no_slide"] = out.get("no_slide", 0) + 1
            continue
        fired = False
        for _ in range(12):
            obs = b.step(move=(0.0, 1.0), buttons=["slide"])
            if not alive(obs):
                break
            mt = obs.get("move_tech") or {}
            out["airborne_sliding"] = out.get("airborne_sliding", 0) + (
                1 if (not obs["player"].get("grounded") and obs["player"].get("sliding")) else 0)
            if (not obs["player"].get("grounded") and obs["player"].get("sliding")
                    and mt.get("falling") and mt.get("wall_available")
                    and not mt.get("jump_cooldown")):
                obs = b.step(move=(0.0, 1.0), buttons=["slide"], macro=2)
                m = obs.get("macro") or {}
                out["macro_result"][m.get("result", "missing")] += 1
                if m.get("reason"):
                    out["macro_reason"][m["reason"]] += 1
                if m.get("lead_s") is not None:
                    out["lead_s"].append(m["lead_s"])
                if m.get("frame_gap") is not None:
                    out["frame_gap"].append(m["frame_gap"])
                out["fired"] += 1
                fired = True
                ev = ssj_this_step(obs, frameskip)
                if ev is None:
                    out["no_event"] += 1
                else:
                    out["buckets"][ev["bucket"]] += 1
                    if ev["landed"]:
                        out["landed"] += 1
                    out["gains"].append(ev["gain"])
                    out["h_gains"].append(ev["h_gain"])
                    out["dt_ms"].append(ev["dt_ms"])
                break
        if not fired:
            out["no_wall_window"] += 1
    return out


def time_accounting(b: Bridge, level: str, frameskip: int, n: int = 15) -> dict:
    """Do macro steps consume the same frames and the same game seconds as ordinary steps?

    Measured on `Time.time`, not `StatsManager.seconds`: the level timer does not run during the level's
    opening and a flat zero there would prove nothing, while `Time.time` advances by `captureDeltaTime`
    every frame the game runs and so is exactly "game time consumed".
    """
    b.reset(level)
    plain_frames, plain_secs, macro_frames, macro_secs = [], [], [], []
    timer_running = None
    prev = b.step(move=(0.0, 1.0))
    for _ in range(n):
        obs = b.step(move=(0.0, 1.0))
        if timer_running is None:
            timer_running = ((obs.get("campaign") or {}).get("timer_running"))
        plain_frames.append(obs["frame"] - prev["frame"])
        plain_secs.append(obs["time"] - prev["time"])
        prev = obs
    for _ in range(n):
        start = get_sliding(b, warm=4, tries=4)
        if start is None:
            continue
        obs = b.step(move=(0.0, 1.0), buttons=["slide"], macro=1)
        macro_frames.append(obs["frame"] - start["frame"])
        macro_secs.append(obs["time"] - start["time"])
        settle(b, steps=8)
    return {
        "frameskip": frameskip,
        "timer_running": timer_running,
        "expected_seconds_per_step": frameskip / 30.0,
        "plain_frames": sorted(set(plain_frames)),
        "macro_frames": sorted(set(macro_frames)),
        "plain_seconds_mean": statistics.mean(plain_secs) if plain_secs else None,
        "macro_seconds_mean": statistics.mean(macro_secs) if macro_secs else None,
        "plain_seconds_set": sorted({round(s, 5) for s in plain_secs}),
        "macro_seconds_set": sorted({round(s, 5) for s in macro_secs}),
        "n_plain": len(plain_frames), "n_macro": len(macro_frames),
    }


def cursor_check(b: Bridge, frameskip: int, n: int = 12) -> dict:
    """After a macro dates an event forward, is the NEXT step's input still delivered?

    Two independent readings. The mod reports its timestamp cursor and the live clock, so `cursor >
    now` is the exact case in which an un-lifted event would have been dropped. The dash probe is the
    behavioural half: a dash that costs 100 stamina proves the tap arrived.
    """
    ahead = 0
    samples = 0
    dash_ok = 0
    dash_tried = 0
    for _ in range(n):
        if settle(b) is None:
            break
        if get_sliding(b) is None:
            continue
        obs = b.step(move=(0.0, 1.0), buttons=["slide"], macro=1)
        inp = obs.get("input") or {}
        if "cursor" in inp and "now" in inp:
            samples += 1
            if inp["cursor"] > inp["now"]:
                ahead += 1
        stamina_before = (obs.get("player") or {}).get("stamina", 0.0)
        if stamina_before >= 100.0:
            dash_tried += 1
            after = b.step(move=(0.0, 1.0), buttons=["dash"])
            if (after.get("player") or {}).get("stamina", stamina_before) < stamina_before - 50.0:
                dash_ok += 1
    return {
        "samples": samples, "cursor_ahead_of_clock": ahead,
        "dash_after_macro_tried": dash_tried, "dash_after_macro_registered": dash_ok,
    }


def obs_sanity(b: Bridge, frameskip: int) -> dict:
    """Slam, coin and rocket-freeze readouts. Slam is always available; the other two need an arsenal."""
    result: dict[str, Any] = {}

    # --- slam. TryStartSlam is much fussier than folklore suggests: it needs `Slide.WasPerformedThisFrame`
    # while airborne AND nothing within 3 m below AND `fallTime > 0.5` AND no slam cooldown. A jump on flat
    # ground can never satisfy it, so the probe has to find a real drop first -- a ground ray reading near
    # its maximum is a pit.
    heavy = False
    slam_force_seen = 0.0
    bounce_seen = False
    pre_slide_seen = 0.0
    fall_time_ok = False
    for _ in range(6):
        if settle(b) is None:
            break
        obs = b.step(move=(0.0, 0.0))
        if not alive(obs):
            break
        gr = obs.get("ground_rays") or []
        if not gr:
            break
        idx = max(range(len(gr)), key=lambda i: gr[i])
        if gr[idx] < 8.0:                     # no pit around: walk on and try again
            b.step(move=(0.0, 1.0))
            b.step(move=(0.0, 1.0))
            continue
        yaw = 360.0 * idx / len(gr)
        if yaw > 180.0:
            yaw -= 360.0
        b.step(look=(yaw, 0.0))
        airborne_steps = 0
        obs = None
        for i in range(30):                   # walk off the edge, then fall with slide RELEASED
            obs = b.step(move=(0.0, 1.0))
            if not alive(obs):
                break
            if not obs["player"].get("grounded"):
                airborne_steps += 1
            # > 0.5 s of falling at 1/30 s per frame and 2 frames per step is ~8 steps
            if airborne_steps >= 9:
                break
        if obs is None or not alive(obs) or airborne_steps < 9:
            continue
        fall_time_ok = True
        obs = b.step(move=(0.0, 0.0), buttons=["slide"])   # the fresh airborne press -> TryStartSlam
        for _ in range(30):
            if not alive(obs):
                break
            mt = obs.get("move_tech") or {}
            heavy = heavy or bool(mt.get("heavy_fall"))
            slam_force_seen = max(slam_force_seen, float(mt.get("slam_force") or 0.0))
            pre_slide_seen = max(pre_slide_seen, float(mt.get("pre_slide_speed") or 0.0))
            bounce_seen = bounce_seen or bool(mt.get("bounce_window"))
            if bounce_seen:
                break
            obs = b.step(move=(0.0, 0.0), buttons=["slide"])
        if heavy and bounce_seen:
            break
    result["slam"] = {
        "reached_a_long_fall": fall_time_ok,
        "heavy_fall_seen": heavy, "max_slam_force": slam_force_seen,
        "bounce_window_seen": bounce_seen, "max_pre_slide_speed": pre_slide_seen,
    }

    # --- what is actually in the arsenal right now
    settle(b)
    obs = b.step()
    slot_counts = (obs.get("player") or {}).get("slot_counts") or []
    result["slot_counts"] = slot_counts
    wt = obs.get("weapon_tech") or {}
    result["weapon_tech_sample"] = {k: wt.get(k) for k in
                                    ("slot", "variation", "variations_in_slot", "gun_ready",
                                     "coin_charge", "rai_charge", "rocket_frozen", "hook_equipped")}

    # --- coin in the air: Marksman is slot 1 variation 1, thrown with fire2.
    coin = {"attempted": False, "seen": False, "max_age": 0.0, "note": ""}
    if slot_counts and len(slot_counts) >= 1 and slot_counts[0] >= 2:
        coin["attempted"] = True
        b.step(slot=1)
        v = b.step(variant=2)  # variant 2 == variation index 1 == Marksman
        coin["variant_result"] = (v.get("variant") or {}).get("result")
        coin["variant_reason"] = (v.get("variant") or {}).get("reason")
        b.step(look=(0.0, 25.0))  # look up so the coin stays airborne longer
        for _ in range(14):
            obs = b.step(buttons=["fire2"] if not coin["seen"] else [])
            for pr in obs.get("projectiles") or []:
                if pr.get("kind") == "coin":
                    coin["seen"] = True
                    coin["max_age"] = max(coin["max_age"], float(pr.get("age") or 0.0))
                    coin["sample"] = {k: pr.get(k) for k in ("kind", "dist", "age", "rel")}
        b.step(look=(0.0, -25.0))
    else:
        coin["note"] = "slot 1 has fewer than 2 variations; Marksman not equipped"
    result["coin"] = coin

    # --- rocket + freeze. Try every populated slot: which slot holds the rocket launcher depends on the
    # level's arsenal, and guessing wrong reads as "the block is broken" when it is not.
    rocket = {"attempted": False, "seen": False, "frozen_flag_seen": False,
              "frozen_projectile_seen": False, "slots_tried": [], "note": ""}
    for slot_i, count in enumerate(slot_counts, start=1):
        if count < 1 or rocket["seen"]:
            continue
        rocket["attempted"] = True
        rocket["slots_tried"].append(slot_i)
        settle(b)
        b.step(slot=slot_i)
        b.step(look=(0.0, 20.0))          # up, so anything launched stays in the air
        for i in range(18):
            # fire1 is a HOLD button and several weapons fire on WasPerformedThisFrame, so it has to be
            # released between shots; fire2 later is the Freezeframe trigger.
            buttons = ["fire1"] if i % 3 == 0 and i < 9 else (["fire2"] if i in (10, 13) else [])
            obs = b.step(buttons=buttons)
            if not alive(obs):
                break
            wt = obs.get("weapon_tech") or {}
            if wt.get("rocket_frozen"):
                rocket["frozen_flag_seen"] = True
            for pr in obs.get("projectiles") or []:
                if pr.get("kind") in ("rocket", "grenade", "cannonball"):
                    rocket["seen"] = True
                    rocket["slot"] = slot_i
                    rocket["sample"] = {k: pr.get(k) for k in
                                        ("kind", "dist", "age", "frozen", "rideable", "vel")}
                    if pr.get("frozen"):
                        rocket["frozen_projectile_seen"] = True
        b.step(look=(0.0, -20.0))
    if not rocket["attempted"]:
        rocket["note"] = "no populated weapon slots"
    result["rocket"] = rocket
    return result


def legacy_check(port: int, level: str, steps: int, frameskip: int, fixed_fps: float) -> dict:
    """A separate connection that behaves exactly like a 0.7.2 client and checks what it gets back."""
    b = Bridge(port)
    try:
        hello = b.hello()
        b.config(frameskip=frameskip, fixed_fps=fixed_fps, unlimited_fps=True, mute=True,
                 windowed=True, difficulty=3, unlock_all_gear=True, soft_death=True)
        obs = b.reset(level)
        extra_obs: set[str] = set()
        extra_player: set[str] = set()
        missing_player: set[str] = set()
        missing_required: set[str] = set()
        key_sets: set[frozenset] = set()
        for i in range(steps):
            obs = b.legacy_step(move=(0.0, 1.0), buttons=["slide"] if i % 3 == 0 else [])
            keys = set(obs)
            extra_obs |= keys - BASE_OBS_KEYS
            missing_required |= REQUIRED_OBS_KEYS - keys
            key_sets.add(frozenset(keys))
            p = obs.get("player") or {}
            extra_player |= set(p) - BASE_PLAYER_KEYS
            missing_player |= BASE_PLAYER_KEYS - set(p)
        times = sorted(b.step_times[2:])

        # The monotonic cursor on the LEGACY path. Every legacy event asks for no timestamp, so the cursor
        # should only ever lift one when the native clock did not tick between two events of the same frame
        # -- and then by exactly one TimeEpsilon (0.5 ms), which the next frame resets. Expect zero lifts, or
        # lifts bounded at 0.0005. Anything larger means the cursor is running ahead of the clock on a path
        # that has no macro in it, which is the one way this change could degrade a 0.7.2 client.
        b.config(obs_input_clock=True)
        lifts, max_lift = 0, 0.0
        for i in range(steps):
            obs = b.legacy_step(move=(0.0, 1.0), buttons=["slide"] if i % 3 == 0 else [])
            clock = obs.get("input") or {}
            lift = float(clock.get("cursor", 0.0)) - float(clock.get("now", 0.0))
            if lift > 0.0:
                lifts += 1
                max_lift = max(max_lift, lift)
        b.config(obs_input_clock=False)

        return {
            "mod_version": hello.get("mod_version"),
            "protocol": hello.get("protocol"),
            "steps": steps,
            "unexpected_obs_keys": sorted(extra_obs),
            "unexpected_player_keys": sorted(extra_player),
            "missing_player_keys": sorted(missing_player),
            "missing_required_obs_keys": sorted(missing_required),
            "distinct_obs_key_sets": len(key_sets),
            "step_ms_median": round(1000 * statistics.median(times), 3) if times else None,
            "step_ms_p90": round(1000 * times[int(0.9 * (len(times) - 1))], 3) if times else None,
            "cursor_lift_steps": lifts,
            "cursor_max_lift_s": round(max_lift, 6),
            "cursor_bounded": max_lift <= 0.0005 + 1e-9,
            # NOTE the limits of this claim. It is a KEY-SET comparison over steady-state `step` replies
            # against a hand-typed 0.7.2 baseline. It does not compare VALUES, does not run against an actual
            # 0.7.2 DLL, and does not cover reset, get_obs, episode end or a scene change. Before installing
            # 0.8.0 on the fleet, run a fixed action script against the live 0.7.2 game and against the test
            # tree and diff the two obs streams field by field, including a reset and an episode end.
            "key_sets_identical_to_0_7_2": (not extra_obs and not extra_player and not missing_player
                                            and not missing_required),
        }
    finally:
        b.close()


# ---------------------------------------------------------------------------------------------------


def summarise(name: str, t: dict) -> str:
    lines = [f"  {name}: attempted {t['attempted']}"]
    if t.get("macro_result"):
        lines.append(f"    result      {dict(t['macro_result'])}")
    if t.get("macro_reason"):
        lines.append(f"    reason      {dict(t['macro_reason'])}")
    if t.get("macro_note"):
        lines.append(f"    note        {dict(t['macro_note'])}  (ran, but with a side effect)")
    lines.append(f"    buckets     {dict(sorted(t['buckets'].items()))}  (valid = 1,2,3)")
    total = sum(t["buckets"].values())
    lines.append(f"    landed      {t['landed']}/{total} {pct(t['landed'], total)}")
    if t["h_gains"]:
        lines.append(f"    h-gain u/s  mean {statistics.mean(t['h_gains']):+.2f}  "
                     f"median {statistics.median(t['h_gains']):+.2f}  "
                     f"min {min(t['h_gains']):+.2f}  max {max(t['h_gains']):+.2f}")
    if t["gains"]:
        lines.append(f"    gain u/s    mean {statistics.mean(t['gains']):+.2f}  "
                     f"median {statistics.median(t['gains']):+.2f}")
    if t["dt_ms"]:
        lines.append(f"    dt ms       mean {statistics.mean(t['dt_ms']):.2f}  "
                     f"median {statistics.median(t['dt_ms']):.2f}")
    if t.get("obs_delta"):
        # The WHOLE step's horizontal speed change, TrySSJ included: the SSJ gain above is only TrySSJ's
        # own contribution, measured across the patched call.
        lines.append(f"    step dspeed mean {statistics.mean(t['obs_delta']):+.2f}  "
                     f"median {statistics.median(t['obs_delta']):+.2f}")
    if t.get("no_event"):
        lines.append(f"    no TrySSJ   {t['no_event']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PRIVATE_PORT,
                    help=f"private game's bridge port (default {PRIVATE_PORT}; 47800-47811 are refused)")
    ap.add_argument("--level", default="Level 0-1")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--frameskip", type=int, default=2)
    ap.add_argument("--fixed-fps", type=float, default=30.0)
    ap.add_argument("--ssj-gap", type=float, default=0.012, help="seconds between the release and the jump")
    ap.add_argument("--wall-lead", type=float, default=-1.0, help="fixed wall-macro lead; <0 = adaptive")
    ap.add_argument("--wall-lead-frames", type=float, default=2.0)
    ap.add_argument("--skip-wall", action="store_true")
    ap.add_argument("--json", help="write the full result to this file")
    args = ap.parse_args()

    report: dict[str, Any] = {"port": args.port, "level": args.level, "trials": args.trials}

    # FIRST, on a game that has never been told about any 0.8 feature. Config is per game process, not per
    # connection (it always has been -- `frameskip` behaves the same way), so running this after the main
    # pass would inherit the main pass's flags and prove nothing.
    print("== old-style client on the new DLL (run first, on a clean game) ==")
    legacy = legacy_check(args.port, args.level, 60, args.frameskip, args.fixed_fps)
    report["legacy"] = legacy
    print(json.dumps(legacy, indent=2, sort_keys=True))
    print()

    b = Bridge(args.port)
    try:
        hello = b.hello()
        report["hello"] = hello
        print(f"mod {hello.get('mod_version')}  protocol {hello.get('protocol')}  "
              f"port {hello.get('port')}  training_instance {hello.get('training_instance')}  "
              f"steam_hidden {hello.get('steam_hidden')}")
        print(f"features: {hello.get('features')}")
        print(f"diag:     {json.dumps(hello.get('diag'), sort_keys=True)}")

        if hello.get("mod_version") != "0.8.0":
            print(f"\nFATAL: this game is running mod {hello.get('mod_version')}, not 0.8.0. The Doorstop "
                  f"override did not take effect and every result below would be a false negative.",
                  file=sys.stderr)
            return 2

        b.config(frameskip=args.frameskip, fixed_fps=args.fixed_fps, unlimited_fps=True, mute=True,
                 windowed=True, difficulty=3, unlock_all_gear=True, soft_death=True,
                 obs_move_tech=True, obs_weapon_tech=True, obs_projectiles=True,
                 obs_input_clock=True, ssj_indicator=True, variant_switching=True,
                 ssj_gap_s=args.ssj_gap, macro_wall_lead_s=args.wall_lead,
                 macro_wall_lead_frames=args.wall_lead_frames)
        b.reset(args.level)
        # walkSpeed and fixedDeltaTime only exist once a player does; re-read them in the level.
        report["diag_in_level"] = b.hello().get("diag")
        print(f"diag in level: {json.dumps(report['diag_in_level'], sort_keys=True)}\n")

        print("== jump SSJ ==")
        macro_t = jump_trials(b, args.trials, args.frameskip, arm="macro")
        hold_t = jump_trials(b, args.trials, args.frameskip, arm="hold")
        release_t = jump_trials(b, args.trials, args.frameskip, arm="release")
        report["jump_macro"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in macro_t.items()}
        report["jump_control_hold"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in hold_t.items()}
        report["jump_control_release"] = {k: (dict(v) if isinstance(v, Counter) else v)
                                          for k, v in release_t.items()}
        print(summarise("macro          ", macro_t))
        print(summarise("control hold   ", hold_t))
        print(summarise("control release", release_t))
        # `release` is the honest baseline: it is what the live policy produces today.
        if macro_t["h_gains"] and release_t["h_gains"]:
            delta = statistics.mean(macro_t["h_gains"]) - statistics.mean(release_t["h_gains"])
            report["jump_speed_advantage_u_s"] = delta
            print(f"  macro advantage over the policy's own release-and-jump: {delta:+.2f} u/s "
                  f"(instantaneous TrySSJ delta, horizontal, mean)")
        # The number that survives the whole step, which is the one a route time actually sees. It is
        # several times smaller than the instantaneous TrySSJ delta; quote THIS against a speed ceiling.
        if macro_t["obs_delta"] and release_t["obs_delta"]:
            step_delta = statistics.median(macro_t["obs_delta"]) - statistics.median(release_t["obs_delta"])
            report["jump_step_advantage_u_s_median"] = step_delta
            print(f"  whole-step advantage (median dspeed over the step): {step_delta:+.2f} u/s")

        if not args.skip_wall:
            print("\n== wall SSJ ==")
            wall_t = wall_trials(b, args.trials, args.frameskip)
            report["wall_macro"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in wall_t.items()}
            print(summarise("macro  ", wall_t))
            print(f"    fired       {wall_t['fired']} of {wall_t['attempted']} approaches "
                  f"({wall_t['no_wall_window']} never reached a wall window)")
            if wall_t["lead_s"]:
                print(f"    lead s      mean {statistics.mean(wall_t['lead_s']):.4f}  "
                      f"frame_gap mean {statistics.mean(wall_t['frame_gap']):.4f}")

        print("\n== time accounting ==")
        ta = time_accounting(b, args.level, args.frameskip)
        report["time_accounting"] = ta
        print(f"  frames per step: plain {ta['plain_frames']} macro {ta['macro_frames']} "
              f"(frameskip {ta['frameskip']})")
        print(f"  game seconds per step: plain {ta['plain_seconds_mean']} macro {ta['macro_seconds_mean']} "
              f"(expected {ta['expected_seconds_per_step']:.5f}, timer_running {ta['timer_running']})")

        print("\n== monotonic cursor ==")
        cc = cursor_check(b, args.frameskip)
        report["cursor"] = cc
        print(f"  cursor ahead of the live clock right after a macro: "
              f"{cc['cursor_ahead_of_clock']}/{cc['samples']}")
        print(f"  dash registered on the step after a macro: "
              f"{cc['dash_after_macro_registered']}/{cc['dash_after_macro_tried']}")

        print("\n== observation blocks ==")
        sanity = obs_sanity(b, args.frameskip)
        report["obs_sanity"] = sanity
        print(json.dumps(sanity, indent=2, sort_keys=True, default=str))
    finally:
        b.close()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
