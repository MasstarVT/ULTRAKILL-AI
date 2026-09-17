"""In-game checks for the skull carry, run on one game before `item_pickup` / `item_placed` are raised off 0.0.

    python scripts/games.py launch --count 1 --monitor 1
    python scripts/skull_check.py                       # Level 1-1 at the training settings, rendering OFF
    python scripts/skull_check.py --render              # the control run: the same thing with the cameras on
    python scripts/skull_check.py --via 46 0 330        # route through a waypoint when the direct line fails
    python scripts/skull_check.py --from-checkpoint 81 -6 231 --approach 81 0 266   # start next to the pedestal
    python scripts/games.py stop

These are checks 1-3 of section 8 of docs/superpowers/specs/2026-09-17-multi-level-and-skull-gates-design.md,
the three that gate raising the two item weights. Each prints PASS, FAIL or SKIP, then a one-line summary; the
exit code is 1 when any check failed. The bridge is single-client, so never run this against a port a trainer uses.

  1 pickup     walk to the red pedestal (81.0, -2.2, 275.0), face it and punch: that items[] entry must read
               held: true -- with every camera disabled, which is the whole point. `Punch.ActiveFrame` only runs
               from `ActiveStart`, a Unity AnimationEvent with no C# caller, so if the fist Animator is culled
               under render: false the mechanism is inert and nothing else in S4 matters.
  2 placement  carry it to (0.0, -6.76, 381.0) and punch: that altar must read filled: true and gate 20,-10,381
               must drop its needs_item (the dead twin 0,-7,381#2 must not keep it set), and both must survive
               100 further decisions of punch spam pressed THROUGH the env, so `_protect_carry` is what is tested.
  3 death      take the skull back out, die with it held, and report what items[] says after the respawn: how
               many entries of that type exist, which is held, where it is. The one thing offline cannot settle.

**It walks; it never teleports.** The pedestal's room (`3 - Skull Field`) starts switched off, and this project
has measured that a bare teleport into a switched-off room activates nothing -- rooms ahead of the player are off,
so their trigger volumes are off too. Check 1 reads `items[].active` before punching and says so plainly when the
item never becomes active, because that, not the punch, is then the thing that failed. `--teleport-assist` allows
walk_to_exit.py's small 2.5 m nudge off a lip; it is off by default so a PASS can never be an artifact of it.

**`--from-checkpoint` is how you switch that room on**, and it is what a bare walk from spawn cannot do: 1-1's
route to the pedestal runs through the skull-locked door `81,-6,240` itself, so on foot from spawn the pedestal
is unreachable without solving the blue leg first. Teleporting onto checkpoint `81,-6,231` and letting it
activate is NOT enough on its own -- measured 2026-09-17, the pedestal still reads `active: false` with the
checkpoint `current: true`. The **respawn** that follows is what brings the room up (`active: true,
inactive_ancestors: 0`). `--approach` then teleports into that now-live room, and the last metres to the item
are still walked, so the punch is never reached by teleport. Both are off by default and both are named in the
check's own output, so a PASS can say how it was set up.

The probe drives through `env.client` directly (like walk_to_exit.py) and keeps `env._raw` in step with it, so the
punch-spam and the death run through `env.step` and exercise the real reward and carry-protection paths.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # campaign_check / walk_to_exit are plain scripts

import campaign_check  # noqa: E402
import walk_to_exit  # noqa: E402

from ultrakill_ai.campaign import altar_aim_point  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.spaces import BUTTONS, noop_action  # noqa: E402

block, vec, exit_code = campaign_check.block, campaign_check.vec, campaign_check.exit_code
go_to, wrap, heading_to, elevation_to = walk_to_exit.go_to, walk_to_exit.wrap, walk_to_exit.heading_to, walk_to_exit.elevation_to

CONFIG = ROOT / "configs" / "campaign_1-1.yaml"
PEDESTAL = (81.0, -2.2, 275.0)   # Level 1-1's red skull, section 8 check 1's target
# The altar it opens, section 8 check 2's target: the zone's own transform position, as `altars[].pos` reports
# it. The punch is NOT aimed here -- `altar_aim_point` moves it to the zone's collider centre a metre below,
# which is why this no longer needs the hand-tuned `--altar 0 -8 381` the first in-game run used.
ALTAR = (0.0, -6.76, 381.0)
GATE_KEY = "20,-10,381"          # the door that altar unlocks
PUNCH_RANGE = 4.0                # Punch.ActiveFrame's own reach, and EnvConfig.subgoal_punch_range_m's default
PITCH_LIMIT = 85.0               # env.MODE1_PITCH_LIMIT: the game's own clamp, which section 6.7 rule 2 uses here
AIM_TOL = 8.0                    # degrees of aim error a 4 m ray still lands a cube-sized target inside
# Metres from `player.pos` (`NewMovement.transform.position`) up to where the punch ray actually starts.
# `Punch.ActiveFrame` rays from `cc.GetDefaultPos()` -- the CAMERA -- not from the player transform, so aiming
# at a floor-level skull from `player.pos` points the ray straight over it. Measured in game on 1-1's red
# pedestal, 2026-09-17: at 1.57 m of flat range a command pitch of -3.0 picks it up and +27.0 (what aiming from
# `player.pos` asks for) does not; at 2.71 m, +8.0 works and +11.3 does not. Both solve to 0.88-0.90 m, and the
# band is only a couple of degrees wide at 2.7 m, so this is not a detail that can be left approximate.
# Taken from the env, which aims look mode 2 from the same point: check 2's punch spam runs through env.step,
# so the two must agree or the script and the thing it is checking are not measuring the same geometry.
CAMERA_HEIGHT = EnvConfig.camera_height_m
CREEP_RANGE = 2.0                # stop walking this close and only aim, so the last step cannot overshoot
CHECKPOINT_STEPS = 40            # decisions to wait on a checkpoint we teleported onto (under 3 game seconds)
CHECKPOINT_RADIUS = 3.0          # metres within which a checkpoints[] entry is the one --from-checkpoint means
SETTLE_STEPS = 5                 # decisions after a staging teleport, so the player lands before the walk starts
MATCH_RADIUS = 8.0               # metres within which an altars[]/items[] entry counts as "the one we mean"
PUNCH_STEPS = 60                 # decisions to spend facing and punching a target (4 game seconds)
SPAM_STEPS = 100                 # section 8 check 2's punch spam, through env.step so the gating applies
NAMES = ("pickup", "placement", "held skull on death")
PASS, FAIL, SKIP = campaign_check.PASS, campaign_check.FAIL, campaign_check.SKIP

Result = tuple[int, str, str, str]  # (number, name, status, detail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="In-game checks for the skull carry (one game).")
    p.add_argument("--level", default="Level 1-1", help="campaign scene name")
    p.add_argument("--port", type=int, default=47800)
    p.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), default=list(PEDESTAL),
                   help="the item to pick up (default: 1-1's red pedestal)")
    p.add_argument("--altar", type=float, nargs=3, metavar=("X", "Y", "Z"), default=list(ALTAR),
                   help="the altar to place it in (default: the altar gate 20,-10,381 waits on)")
    p.add_argument("--via", type=float, nargs=3, metavar=("X", "Y", "Z"), action="append",
                   help="a waypoint to walk through on the way to --target; repeatable")
    p.add_argument("--from-checkpoint", type=float, nargs=3, metavar=("X", "Y", "Z"), action="append",
                   help="teleport onto this checkpoint, let it activate, then respawn there before check 1; "
                        "repeatable, and the respawn happens at the last one")
    p.add_argument("--approach", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="staging teleport before the walk to --target: a point in a room --from-checkpoint has "
                        "already switched ON. The last metres are still walked")
    p.add_argument("--altar-approach", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="the same staging teleport for check 2's walk to --altar, carrying the item")
    p.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT,
                   help="metres from player.pos up to the punch ray's origin; 0 aims from the player transform")
    p.add_argument("--gate", default=GATE_KEY, help="the gate key whose needs_item must clear (empty to skip it)")
    p.add_argument("--budget", type=int, default=4000, help="decisions per leg before giving up on it")
    p.add_argument("--render", action="store_true", help="cameras ON: the control run, which check 1 does not count")
    p.add_argument("--teleport-assist", action="store_true", help="allow the 2.5 m nudge off a lip after a stall")
    p.add_argument("--skip-kill", action="store_true", help="stop after check 2 and leave the level solved")
    p.add_argument("--fixed-fps", type=float, help="override the config's fixed_fps")
    p.add_argument("--frameskip", type=int, help="override the config's frameskip")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> EnvConfig:
    """configs/campaign_1-1.yaml's env settings, with fresh level loads only and nothing written.

    `soft_death` is forced off: check 3 needs a real death, and soft death would heal the kill command instead.
    `max_steps` is raised so the env's own truncation cannot end a leg half way.
    """
    env = dict((yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("env", {}))
    env.update(level=args.level, port=args.port, fresh_start_prob=1.0, explore_dir="", best_runs_dir="",
               render=bool(args.render), soft_death=False, max_steps=10_000_000,
               camera_height_m=args.camera_height)
    if args.fixed_fps:
        env["fixed_fps"] = args.fixed_fps
    if args.frameskip:
        env["frameskip"] = args.frameskip
    return EnvConfig.from_dict(env)


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def eye(player: dict[str, Any], height: float) -> list[float]:
    """Where the punch ray starts: `cc.GetDefaultPos()`, which is `height` metres above `player.pos`.

    Everything about a punch -- its 4 m reach and the direction it is cast in -- is measured from here, so this
    is what an aim and a range test have to use. See CAMERA_HEIGHT.
    """
    x, y, z = player["pos"]
    return [x, y + height, z]


def punch_action():
    """The campaign action vector for 'press punch and nothing else', so the spam runs through the real env."""
    a = noop_action(campaign=True)
    a[2 + BUTTONS.index("punch")] = 1
    return a


def entries(raw: dict[str, Any], field: str) -> list[dict]:
    return [e for e in (block(raw).get(field) or ()) if isinstance(e, dict)]


def nearest_entry(raw: dict[str, Any], field: str, pos) -> tuple[dict | None, float]:
    """The `altars[]` / `items[]` entry closest to `pos`, and its distance, or (None, inf) beyond MATCH_RADIUS."""
    best, best_d = None, math.inf
    for e in entries(raw, field):
        if not e.get("pos"):
            continue
        d = math.dist(e["pos"], pos)
        if d < best_d:
            best, best_d = e, d
    return (best, best_d) if best is not None and best_d <= MATCH_RADIUS else (None, best_d)


def by_key(raw: dict[str, Any], field: str, key: str | None) -> dict | None:
    return next((e for e in entries(raw, field) if key is not None and e.get("key") == key), None)


def gate_by_key(raw: dict[str, Any], key: str) -> dict | None:
    return next((g for g in (block(raw).get("gates") or ()) if isinstance(g, dict) and g.get("key") == key), None)


def describe_item(item: dict | None, player_pos=None) -> str:
    if item is None:
        return "no items[] entry"
    where = "held" if item.get("held") else (f"in {item.get('placed_in')}" if item.get("placed") else "loose")
    text = (f"{item.get('item')} {item.get('key')} at {vec(item.get('pos') or [0, 0, 0])} {where}, "
            f"active={item.get('active')} active_self={item.get('active_self')} "
            f"inactive_ancestors={item.get('inactive_ancestors')}")
    if player_pos is not None and item.get("pos"):
        text += f", {math.dist(item['pos'], player_pos):.1f} m from the player"
    return text


def drive(env: UltrakillEnv, name: str, target, args: argparse.Namespace) -> tuple[dict[str, Any], int, bool]:
    """Walks to `target` with walk_to_exit's mover, keeping env._raw in step. (raw, decisions, level finished)."""
    raw, used, done = go_to(env, name, list(target), args.budget, bool(args.teleport_assist))
    env._raw = raw
    return raw, used, done


def activate_checkpoints(env: UltrakillEnv, args: argparse.Namespace) -> str:
    """Teleports onto each `--from-checkpoint`, waits for it to activate, then respawns at the last one.

    The respawn is the point. Measured on 1-1 (2026-09-17): teleporting onto `81,-6,231` makes it
    `activated: true, current: true` within a handful of decisions, and the red pedestal 44 m away still reads
    `active: false, inactive_ancestors: 1` -- the room is switched off, so nothing there can be punched. The
    `reset(checkpoint=True)` that follows re-creates the checkpoint's rooms and the pedestal comes up
    `active: true, inactive_ancestors: 0`. `env._respawn` is used rather than a bare client call so the env's
    milestone and gate bookkeeping stays consistent, exactly as it would after a death in training.
    """
    notes = []
    for cp in args.from_checkpoint or ():
        x, y, z = cp
        env._raw = env.client.teleport([x, y + 1.0, z])
        reached = False
        for _ in range(CHECKPOINT_STEPS):
            env.step(campaign_check.CAMPAIGN_NOOP)
            reached = any(math.dist(e["pos"], cp) <= CHECKPOINT_RADIUS and (e.get("activated") or e.get("current"))
                          for e in (block(env._raw).get("checkpoints") or ()) if e.get("pos"))
            if reached:
                break
        notes.append(f"checkpoint {vec(cp)} activated={reached}")
    if args.from_checkpoint:
        env._raw = env._respawn()
        player = (env._raw.get("player") or {}).get("pos")
        notes.append(f"respawned at {vec(player) if player else 'no player'}")
    return "; ".join(notes)


def stage(env: UltrakillEnv, point) -> str:
    """Staging teleport into a room `--from-checkpoint` has already switched on, then a few decisions to land."""
    env._raw = env.client.teleport(list(point))
    for _ in range(SETTLE_STEPS):
        env._raw = env.client.step({})
    player = (env._raw.get("player") or {}).get("pos")
    return f"staged by teleport to {vec(point)}, landed at {vec(player) if player else 'no player'}"


def face_and_punch(env: UltrakillEnv, target, done: Callable[[dict[str, Any]], bool], *,
                   steps: int = PUNCH_STEPS, height: float = CAMERA_HEIGHT) -> tuple[dict[str, Any], int, bool]:
    """Creeps the last metres, faces `target` inside the game's own pitch clamp and punches. (raw, presses, met).

    The pitch clamp rather than the campaign band is section 6.7 rule 2: the 4 m ray runs along camera forward
    from the camera's default position, so a floor-level cube a metre away sits under a 45 degree band entirely.

    The aim and the range test are both taken from `eye(player, height)`, never from `player.pos`: the ray starts
    at the camera, and at skull range the difference is the whole answer (see CAMERA_HEIGHT).
    """
    raw, fired = env._raw, 0
    for _ in range(steps):
        player = raw.get("player")
        if player is None or block(raw).get("input_locked"):
            raw = env.client.step({})
            continue
        pos = eye(player, height)
        yaw_err = wrap(heading_to(pos, target) - player["yaw"])
        pitch_err = max(-PITCH_LIMIT, min(PITCH_LIMIT, elevation_to(pos, target))) - player["pitch"]
        gap = math.dist(pos, target)
        aimed = abs(yaw_err) <= AIM_TOL and abs(pitch_err) <= AIM_TOL and gap <= PUNCH_RANGE
        fired += aimed
        raw = env.client.step({
            "move": [0, 1 if gap > CREEP_RANGE else 0],
            "buttons": ["punch"] if aimed else [],
            "look": [clamp(yaw_err, walk_to_exit.MAX_TURN), clamp(pitch_err, walk_to_exit.MAX_TURN)],
        })
        env._raw = raw
        if done(raw):
            return raw, fired, True
    return raw, fired, done(raw)


# -- the three checks ---------------------------------------------------------------------------


def check_pickup(env: UltrakillEnv, args: argparse.Namespace) -> tuple[str, str, str | None]:
    """(status, detail, the key of the item that ended up held)."""
    target = list(args.target)
    for i, via in enumerate(args.via or ()):
        _, _, done = drive(env, f"via {i + 1}", via, args)
        if done:
            return FAIL, f"the level ended while walking to the waypoint {vec(via)}", None
    staged = stage(env, args.approach) if args.approach else ""
    raw, used, done = drive(env, "pedestal", target, args)
    player = raw.get("player")
    if player is None:
        return FAIL, f"no player after {used} decisions toward {vec(target)}", None
    gap = math.dist(player["pos"], target)
    where = (f"walked {used} decisions to {vec(player['pos'])}, {gap:.1f} m from {vec(target)}; "
             f"render={env.cfg.render}")
    if staged:
        where += f"\n      {staged} -- the last metres to the item were walked from there"
    if "items" not in block(raw):
        return FAIL, where + "; campaign.items is missing (mod older than v0.7.0)", None
    item, dist = nearest_entry(raw, "items", target)
    if item is None:
        return FAIL, (where + f"; no items[] entry within {MATCH_RADIUS:.0f} m of the target "
                              f"(nearest is {dist:.1f} m away, of {len(entries(raw, 'items'))} in the level): "
                              "either the point is wrong or the room holding it is still switched off"), None
    where += f"\n      before the punch: {describe_item(item, player['pos'])}"
    if not item.get("active"):
        where += ("\n      the item is NOT active: its room is still switched off, so nothing here can pick it "
                  "up. This is the walk failing, not the punch -- route the approach with --via.")
    key = item.get("key")
    raw, fired, held = face_and_punch(env, target, lambda r: bool((by_key(r, "items", key) or {}).get("held")),
                                      height=args.camera_height)
    after = by_key(raw, "items", key) or nearest_entry(raw, "items", target)[0]
    detail = where + f"\n      after {fired} aimed punches: {describe_item(after, (raw.get('player') or {}).get('pos'))}"
    if not held:
        return FAIL, detail + ("\n      held never went true"
                               + (" -- and with rendering off that is exactly the ActiveStart/AnimationEvent "
                                  "failure S4 is gated on; rerun with --render as the control"
                                  if not env.cfg.render else " even with the cameras on")), None
    if env.cfg.render:
        detail += "\n      NOTE: this is the control run (--render). Check 1 only counts with rendering OFF."
    return PASS, detail, key


def check_placement(env: UltrakillEnv, args: argparse.Namespace, key: str | None) -> tuple[str, str]:
    if key is None:
        return SKIP, "needs the skull picked up in check 1"
    target = list(args.altar)
    # Walk to and aim at the zone's COLLIDER CENTRE, not the transform position `--altar` names: a placement
    # punch has to hit the zone's own collider, and that position sits 0.4 m under its lid (altar_aim_point).
    # Looked up before the drive, because the altars array is sent every step whatever the distance.
    found, _ = nearest_entry(env._raw, "altars", target)
    aim = altar_aim_point(found) if found else list(target)
    args.altar_aim = aim  # check 3 takes the skull back out at the same point
    staged = stage(env, args.altar_approach) if args.altar_approach else ""
    raw, used, done = drive(env, "altar", aim, args)
    if done:
        return FAIL, f"the level ended while carrying the skull to {vec(aim)}"
    altar, dist = nearest_entry(raw, "altars", target)
    if altar is None:
        return FAIL, (f"walked {used} decisions; no altars[] entry within {MATCH_RADIUS:.0f} m of {vec(target)} "
                      f"(nearest is {dist:.1f} m away, of {len(entries(raw, 'altars'))} in the level)"
                      + (f"\n      {staged}" if staged else ""))
    altar_key = altar.get("key")
    aim = altar_aim_point(altar)
    args.altar_aim = aim
    raw, fired, filled = face_and_punch(env, aim, lambda r: bool((by_key(r, "altars", altar_key) or {}).get("filled")),
                                        height=args.camera_height)

    def state(r: dict[str, Any]) -> str:
        a = by_key(r, "altars", altar_key) or {}
        text = f"altar {altar_key} filled={a.get('filled')} (wants {a.get('item')}, active={a.get('active')})"
        if args.gate:
            g = gate_by_key(r, args.gate)
            text += (f", gate {args.gate} " + ("absent" if g is None else
                     f"needs_item={g.get('needs_item')!r} open={g.get('open')} locked={g.get('locked')} "
                     f"controller_active={g.get('controller_active')}"))
        text += f", item: {describe_item(by_key(r, 'items', key))}"
        return text

    at_place = state(raw) + f"\n      aimed at the collider centre {vec(aim)}, reported pos {vec(target)}"
    gate = gate_by_key(raw, args.gate) if args.gate else None
    ok = filled and (gate is None or not gate.get("needs_item"))
    detail = f"walked {used} decisions, {fired} aimed punches\n      at the placement: {at_place}"
    if staged:
        detail += f"\n      {staged} -- the last metres to the altar were walked from there"
    if not ok:
        missing = "the altar never read filled" if not filled else f"gate {args.gate} still reads needs_item"
        return FAIL, detail + f"\n      {missing}"

    # The spam runs through env.step, so section 6.7's carry protection is what is being tested: standing next to
    # a filled altar with nothing held, every one of these presses should be dropped before it reaches the game.
    # Every decision is inspected, not just the last: an unprotected punch pulls the skull out and the next one
    # puts it back, so sampling only the end state reports whatever the parity of SPAM_STEPS happens to be --
    # while the door that altar opened has been slammed and reopened fifty times, and `Close()` is real.
    def solved(r: dict[str, Any]) -> bool:
        g = gate_by_key(r, args.gate) if args.gate else None
        return bool((by_key(r, "altars", altar_key) or {}).get("filled")) and not (g or {}).get("needs_item")

    undone_at, spammed = None, 0
    for i in range(SPAM_STEPS):
        _, _, terminated, truncated, info = env.step(punch_action())
        spammed = i + 1
        if undone_at is None and not solved(env._raw):
            undone_at = spammed
        if terminated or truncated:
            detail += f"\n      the episode ended after the placement: {info.get('end_reason')}"
            break
    raw = env._raw
    detail += f"\n      after {spammed} decisions of punch spam: {state(raw)}"
    if undone_at is not None:
        return FAIL, detail + (f"\n      the placement was undone by the spam, first at decision {undone_at} of "
                               f"{spammed}: the carry protection did not hold")
    if not solved(raw):
        return FAIL, detail + "\n      the placement did not survive the spam"
    return PASS, detail


def check_death(env: UltrakillEnv, args: argparse.Namespace, key: str | None) -> tuple[str, str]:
    """Takes the skull back out, dies holding it, and reports what items[] says after the respawn."""
    if key is None:
        return SKIP, "needs the skull picked up in check 1"
    if args.skip_kill:
        return SKIP, "--skip-kill"
    raw = env._raw
    item = by_key(raw, "items", key)
    kind = (item or {}).get("item")
    # Re-acquire by punching the filled altar: AltHit's not-holding branch ForceHolds whatever it hits, including
    # a skull resting in a zone. Driven through env.client, so the env's own gating does not block the probe.
    raw, fired, held = face_and_punch(env, list(getattr(args, "altar_aim", None) or args.altar),
                                      lambda r: bool((by_key(r, "items", key) or {}).get("held")),
                                      height=args.camera_height)
    if not held:
        return SKIP, (f"could not take the skull back out of the altar in {fired} aimed punches, so there was "
                      f"nothing held to die with: {describe_item(by_key(raw, 'items', key))}")
    before = describe_item(by_key(raw, "items", key), (raw.get("player") or {}).get("pos"))
    deaths_before = env._deaths
    killed = env.client.kill()
    env._raw = killed
    _, _, terminated, truncated, info = env.step(campaign_check.CAMPAIGN_NOOP)
    raw = env._raw
    player = raw.get("player")
    same_kind = [e for e in entries(raw, "items") if e.get("item") == kind]
    detail = (f"holding before the kill: {before}\n      kill reply dead={(killed.get('player') or {}).get('dead')}, "
              f"deaths {deaths_before} -> {info.get('deaths')}, episode ended={terminated or truncated}"
              f", player {'alive at ' + vec(player['pos']) if player and not player['dead'] else 'missing or dead'}")
    detail += f"\n      {len(same_kind)} {kind} entries after the respawn:"
    for e in same_kind[:4]:
        detail += f"\n        {describe_item(e, player['pos'] if player else None)}"
    if not same_kind:
        return FAIL, detail + ("\n      the skull is GONE from items[]: a death mid-carry would make the puzzle "
                               "unsolvable for the rest of the level load")
    if any(e.get("held") for e in same_kind):
        return PASS, detail + "\n      the skull survived the death still held"
    return PASS, detail + "\n      the skull is back in the level, not held (re-fetchable)"


def run_checks(env: UltrakillEnv, args: argparse.Namespace) -> list[Result]:
    results: list[Result] = []

    def record(number: int, status: str, detail: str) -> None:
        results.append((number, NAMES[number - 1], status, detail))
        print(f"[{status}] {number} {NAMES[number - 1]}: {detail}", flush=True)

    env.reset()
    raw = env._raw
    camp = block(raw)
    if not camp:
        record(1, FAIL, f"scene {raw.get('scene')!r} has no campaign block; is this a campaign scene?")
        record(2, SKIP, "needs the campaign block")
        record(3, SKIP, "needs the campaign block")
        return results
    print(f"  spawn {vec((raw.get('player') or {}).get('pos', [0, 0, 0]))}, "
          f"{len(entries(raw, 'altars'))} altars, {len(entries(raw, 'items'))} items, "
          f"{len(camp.get('gates') or [])} gates, difficulty {camp.get('difficulty')}", flush=True)
    if args.from_checkpoint:
        print("  " + activate_checkpoints(env, args), flush=True)

    status, detail, key = check_pickup(env, args)
    record(1, status, detail)
    record(2, *check_placement(env, args, key))
    record(3, *check_death(env, args, key))
    return results


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    print(f"{cfg.level} on port {cfg.port}: fixed_fps {cfg.fixed_fps:g}, frameskip {cfg.frameskip}, "
          f"render {cfg.render}, difficulty {cfg.difficulty}, punch range {cfg.subgoal_punch_range_m:g} m, "
          f"teleport assist {bool(args.teleport_assist)}", flush=True)
    env = UltrakillEnv(cfg)
    try:
        results = run_checks(env, args)
    finally:
        env.close()
    print("summary: " + " | ".join(f"{number} {name} {status}" for number, name, status, _ in results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
