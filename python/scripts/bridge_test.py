"""Checks the mod bridge end to end.

    python scripts/bridge_test.py            # print what the mod sees (you keep control)
    python scripts/bridge_test.py --drive    # also drive the player through a scripted sequence

Start a level or Cyber Grind first, then run this.
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
    parser.add_argument("--drive", action="store_true", help="take control and run a scripted input sequence")
    parser.add_argument("--realtime", action="store_true", help="run at normal speed with sound so you can watch")
    args = parser.parse_args()

    with BridgeClient(port=args.port) as client:
        hello = client.connect()
        print(f"Connected: mod {hello['mod_version']}, protocol {hello['protocol']}, scene {hello['scene']}")
        print(summarize(client.get_obs()))

        if not args.drive:
            return

        client.configure(
            frameskip=4,
            unlimited_fps=not args.realtime,
            mute=not args.realtime,
            windowed=not args.realtime,
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
