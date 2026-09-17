"""Checks the mod bridge end to end.

    python scripts/bridge_test.py              # print what the mod sees (you keep control)
    python scripts/bridge_test.py --campaign   # also print the campaign block (you keep control)
    python scripts/bridge_test.py --drive      # also drive the player through a scripted sequence

Start a level or Cyber Grind first, then run this. The bridge serves one client at a time, so never point
it at a game a trainer is using: connecting drops the trainer.

--campaign only reads (`get_obs`), so the mod's difficulty and gear overrides, which apply only while the AI
has control, are not in effect: `difficulty` shows the game's own setting here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.protocol import BridgeClient  # noqa: E402


def summarize(obs: dict) -> str:
    p = obs.get("player")
    if not p:
        return f"scene={obs.get('scene')} (no player)"
    enemies = obs.get("enemies", [])
    nearest = ", ".join(f"{e['type_name']}@{e['dist']:.1f}m{'*' if e['visible'] else ''}" for e in enemies[:4])
    pos = ", ".join(f"{v:.1f}" for v in p["pos"])
    cg = obs.get("cybergrind")
    wave = f" wave={cg['wave']} left={cg['enemies_left']}" if cg else ""
    return (
        f"step={obs['step']} scene={obs['scene']} hp={p['hp']} stamina={p['stamina']:.0f} "
        f"pos=({pos}) yaw={p['yaw']:.0f} pitch={p['pitch']:.0f} grounded={p['grounded']} "
        f"kills={obs['stats'].get('kills')}{wave} enemies={len(enemies)} [{nearest}]"
    )


def vec(v) -> str:
    return ", ".join(f"{x:.1f}" for x in v)


def summarize_campaign(obs: dict) -> str:
    c = obs.get("campaign")
    if not c:
        return "no campaign block (not in a campaign level, or the mod is older than 0.5.0)"
    lines = [
        f"mission={c['mission']} difficulty={c['difficulty']} seconds={c['seconds']:.2f} "
        f"timer_running={c['timer_running']} level_started={c['level_started']} level_over={c['level_over']} "
        f"restarts={c['restarts']} input_locked={c['input_locked']}"
    ]
    exit_ = c.get("exit")
    lines.append(f"exit: ({vec(exit_['pos'])}) active={exit_['active']}" if exit_ else "exit: null (its room may not be loaded yet)")
    checkpoints = c.get("checkpoints", [])
    lines.append(f"checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        flags = (" activated" if cp["activated"] else "") + (" current" if cp["current"] else "")
        lines.append(f"  {cp['id']} at ({vec(cp['pos'])}){flags}")
    path = c.get("path") or {"status": "none"}
    if path["status"] == "none":
        lines.append("path: none")
    else:
        lines.append(f"path: {path['status']} length={path['length']:.1f}m next_corner=({vec(path['next_corner'])})")
    gates = c.get("gates")
    if gates is None:
        lines.append("gates: - (a mod older than 0.6.0)")
    else:
        ordered, truncated = c.get("gates_ordered"), c.get("gates_truncated")
        with_hops = sum(1 for g in gates if g.get("hops") is not None)
        # The ratio is what GateProgress's usability guard tests: below 0.5 the ladder is ignored entirely.
        ratio = f"{with_hops / len(gates):.3f}" if gates else "-"
        lines.append(f"gates: {len(gates)} ordered={ordered} truncated={truncated} with hops={with_hops} ({ratio})")
        for g in gates:
            flags = "".join(f" {name}" for name in ("open", "locked", "active", "controller_active") if g.get(name))
            flags += " altar_only" if g.get("altar_only") else ""
            needs = f" needs={g['needs_item']}" if g.get("needs_item") else ""
            lines.append(f"  {g['key']} at ({vec(g['pos'])}) hops={g.get('hops')}{flags}{needs}")
    # Skull carry (mod 0.7.0). `-` is a mod that does not send them at all; an empty list is a level with none.
    altars = c.get("altars")
    if altars is None:
        lines.append("altars: - (a mod older than 0.7.0)")
    else:
        lines.append(f"altars: {len(altars)}")
        for a in altars:
            doors = ", ".join(str(d.get("key")) for d in a.get("doors") or ()) or "none"
            reverse = ", ".join(str(d.get("key")) for d in a.get("reverse_doors") or ()) or "none"
            # `aim_pos` is the collider centre a placement punch must hit (mod 0.7.1). `-` means an older mod,
            # where `campaign.altar_aim_point` falls back to `pos` minus a metre.
            aim = f"({vec(a['aim_pos'])})" if a.get("aim_pos") else "-"
            lines.append(f"  {a['key']} at ({vec(a['pos'])}) aim={aim} {a.get('item')} filled={a.get('filled')}"
                         f" active={a.get('active')} inactive_ancestors={a.get('inactive_ancestors')}"
                         f" opens=[{doors}] closes=[{reverse}]")
    items = c.get("items")
    if items is None:
        lines.append("items: - (a mod older than 0.7.0)")
    else:
        lines.append(f"items: {len(items)}")
        for i in items:
            lines.append(f"  {i['key']} at ({vec(i['pos'])}) {i.get('item')} held={i.get('held')}"
                         f" placed={i.get('placed')} in={i.get('placed_in')} active={i.get('active')}"
                         f" active_self={i.get('active_self')} inactive_ancestors={i.get('inactive_ancestors')}")
    doors = c.get("locked_doors", [])
    lines.append("locked doors: " + (", ".join(f"({vec(d['pos'])}) {d['dist']:.1f}m" for d in doors) or "none"))
    lines.append(f"arena enemies alive: {c['arena_enemies_alive']}")
    lines.append(f"cleared arenas: {c.get('cleared_arenas') or 'none'}  unlocked doors: {c.get('unlocked_doors') or 'none'}")
    ranks = c.get("ranks") or {}
    lines.append(f"ranks: time={ranks.get('time')} kills={ranks.get('kills')} style={ranks.get('style')}")
    p = obs.get("player")
    if p and "slot_counts" in p:
        lines.append(f"weapons per slot: {p['slot_counts']}")
    return "\n".join(lines)


SEQUENCE = [
    ("walk forward", 30, {"move": [0, 1]}),
    ("turn right 90 degrees", 6, {"look": [15, 0]}),
    ("strafe left", 15, {"move": [-1, 0]}),
    ("jump", 1, {"buttons": ["jump"]}),
    ("wait", 15, {}),
    ("dash forward", 1, {"move": [0, 1], "buttons": ["dash"]}),
    ("look up then down", 10, {"look": [0, 4]}),
    ("", 10, {"look": [0, -4]}),
    ("fire primary", 15, {"buttons": ["fire1"]}),
    ("switch to slot 2 and fire", 1, {"slot": 2}),
    ("", 15, {"buttons": ["fire1"]}),
    ("punch", 1, {"buttons": ["punch"]}),
    ("slide", 15, {"move": [0, 1], "buttons": ["slide"]}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=47800)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drive", action="store_true", help="take control and run a scripted input sequence")
    mode.add_argument("--campaign", action="store_true", help="also print the campaign block (read-only, never takes control)")
    parser.add_argument("--realtime", action="store_true", help="run at normal speed with sound so you can watch")
    args = parser.parse_args()

    with BridgeClient(port=args.port) as client:
        hello = client.connect()
        print(f"Connected: mod {hello['mod_version']}, protocol {hello['protocol']}, scene {hello['scene']}")
        obs = client.get_obs()
        print(summarize(obs))
        if args.campaign:
            print(summarize_campaign(obs))

        if not args.drive:
            return

        client.configure(
            frameskip=4,
            unlimited_fps=not args.realtime,
            mute=not args.realtime,
            windowed=not args.realtime,
            render=True,
            block_human_input=True,
        )
        for label, steps, action in SEQUENCE:
            if label:
                print(f"-- {label}")
            for _ in range(steps):
                obs = client.step(action)
            print(summarize(obs))
        client.release()
        print("Released control.")


if __name__ == "__main__":
    main()
