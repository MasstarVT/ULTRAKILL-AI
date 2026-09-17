"""skull_check.py's three checks, run against the fake skull room. No game needed:  python tests/test_skull_check.py  (or pytest).

The point of this file is that the probe is proven before a game is ever launched: the pause in the integration
plan only has to run it. `FakeLevel` (tests/test_campaign_env.py) is the same corridor the env tests use, with the
skull room switched on -- a pedestal source at z 20, a destination altar at z 40 and the gate they open at z 50 --
so the script drives it exactly as it would drive 1-1's red leg, only shorter.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultrakill_ai.env import UltrakillEnv  # noqa: E402

import test_campaign_env as corridor  # noqa: E402  the FakeLevel the campaign env tests are written against

# `--target` names the ITEM to pick up (items[].pos), `--altar` names the ZONE (altars[].pos); the script
# derives the point to aim a placement at from the zone's own aim_pos. Exactly as the real run passes them.
PEDESTAL = [f"{v:g}" for v in corridor.PEDESTAL_ITEM_POS]
ALTAR = [f"{v:g}" for v in corridor.ALTAR_POS]
CHECKPOINT = ["0", "1", "20"]  # corridor.CHECKPOINT_ID's position, which is also the pedestal's


class CheckpointLevel(corridor.FakeLevel):
    """`FakeLevel` with Level 1-1's shape: the skull's room stays switched OFF until a checkpoint RESPAWN.

    Measured on 1-1 (2026-09-17): teleporting onto checkpoint `81,-6,231` makes it `activated`/`current` within a
    few decisions while the pedestal 44 m away still reads `active: false` -- so a probe that only teleports onto
    the checkpoint sees a dead item and blames the punch. Only the `reset(checkpoint=True)` that follows brings the
    room up. That two-step shape is what `--from-checkpoint` exists for, so it is what the fake reproduces.
    """

    def __init__(self):
        super().__init__()
        self.teleports: list[list[float]] = []
        self.item_active = False

    def teleport(self, pos):
        self.teleports.append(list(pos))
        self.y, self.z = pos[1], pos[2]
        return self._obs("teleport")

    def reset(self, scene=None, checkpoint=False):
        super().reset(scene, checkpoint)
        # `_load` clears `self.checkpoint`, so it still reading True means the respawn branch ran.
        self.item_active = bool(checkpoint and self.checkpoint)
        self.y = 1.0  # a respawn puts the player back on the corridor floor, undoing the teleport's metre of lift
        return self._obs("reset")


def load_script():
    spec = importlib.util.spec_from_file_location("skull_check", ROOT / "scripts" / "skull_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*extra: str, skulls: dict | None = None, cfg_overrides: dict | None = None,
        client_cls=corridor.FakeLevel, **attrs):
    """Runs the three checks against a fresh skull room. `attrs` are set on the FakeLevel after enable_skulls."""
    script = load_script()
    # The real 0.9 m eye offset, not 0: the fake corridor's altars and skulls now stand at the heights the
    # game's do (a zone transform 1.9 m up, its collider centre a metre below that), so a punch aimed from the
    # camera is a level shot and one aimed from `player.pos` -- or at `altars[].pos` -- is 25-27 degrees out
    # and misses. That is the whole point: the fake has to be able to fail the way the game did.
    args = script.parse_args(["--level", corridor.LEVEL, "--target", *PEDESTAL, "--altar", *ALTAR,
                              "--gate", corridor.SKULL_GATE_KEY, *extra])
    cfg = script.build_config(args)
    if cfg_overrides:
        cfg = replace(cfg, **cfg_overrides)
    env = UltrakillEnv(cfg)
    game = client_cls()
    game.enable_skulls(**(skulls or {}))
    for name, value in attrs.items():
        setattr(game, name, value)
    env.client = game
    try:
        results = script.run_checks(env, args)
    finally:
        env.close()
    return script, game, results


def statuses(results):
    return [status for _, _, status, _ in results]


def test_the_whole_carry_passes_on_the_fake_skull_room():
    """Walk to the pedestal, punch, carry, punch, spam, take it back out, die: all three checks green."""
    script, game, results = run()
    assert statuses(results) == ["PASS", "PASS", "PASS"], results
    assert game.skull_held is False and game.thrown == 0, "nothing was ever thrown away"
    # Exactly two pickups: the pedestal in check 1 and back out of the altar in check 3. If the punch spam of
    # check 2 had reached the game this would be in the dozens, so it doubles as proof the gating held.
    assert game.picked_up == 2, f"picked up {game.picked_up} times"
    assert "held" in results[0][3] and "active=True" in results[0][3]
    assert "needs_item=None" in results[1][3] and "punch spam" in results[1][3]
    assert "not held (re-fetchable)" in results[2][3] and "skull#1" in results[2][3]
    assert script.exit_code(results) == 0


def test_an_item_whose_room_is_switched_off_is_named_as_the_reason():
    """The known hazard: the pedestal's room starts off, and then it is the walk that failed, not the punch."""
    script, game, results = run(item_active=False)
    assert statuses(results) == ["FAIL", "SKIP", "SKIP"], results
    assert "the item is NOT active" in results[0][3] and "still switched off" in results[0][3]
    assert "held never went true" in results[0][3]
    assert "ActiveStart/AnimationEvent" in results[0][3], "and the render: false reading is offered, not assumed"
    assert results[1][3].startswith("needs the skull") and results[2][3].startswith("needs the skull")
    assert script.exit_code(results) == 1


def test_a_placement_undone_by_the_punch_spam_fails_check_2():
    """With the carry protection off the spam really does pull the skull back out, and the check catches it.

    This is the failure section 8 check 2 exists to detect, and the reason `_protect_carry` is not optional:
    `AltHit`'s not-holding branch ForceHolds whatever it hits, including a skull resting in a filled altar.
    """
    script, game, results = run(cfg_overrides={"subgoal_punch_range_m": 0.0})
    assert statuses(results)[1] == "FAIL", results
    assert "undone by the spam" in results[1][3]
    assert game.picked_up > 2, "the spam really did keep pulling it out"
    assert script.exit_code(results) == 1


def test_a_mod_without_items_fails_the_pickup_early():
    """A 0.6.x mod sends no altars/items, so there is nothing to punch at and nothing to report."""
    script, game, results = run(skulls={"fields": False})
    assert statuses(results) == ["FAIL", "SKIP", "SKIP"], results
    assert "older than v0.7.0" in results[0][3]
    assert script.exit_code(results) == 1


def test_skip_kill_stops_after_the_placement():
    script, game, results = run("--skip-kill")
    assert statuses(results) == ["PASS", "PASS", "SKIP"], results
    assert results[2][3] == "--skip-kill"
    assert game.picked_up == 1, "the skull was never taken back out"
    assert script.exit_code(results) == 0


def test_the_control_run_marks_itself_as_not_counting():
    """--render is the control: it proves the punch works at all, never that it works under training settings."""
    script, game, results = run("--render")
    assert statuses(results) == ["PASS", "PASS", "PASS"], results
    assert "control run" in results[0][3] and "rendering OFF" in results[0][3]


def test_from_checkpoint_switches_the_room_on_and_the_whole_carry_then_passes():
    """1-1's shape end to end: without the respawn the item is dead, with it the carry works."""
    script, game, results = run("--from-checkpoint", *CHECKPOINT, client_cls=CheckpointLevel)
    assert statuses(results) == ["PASS", "PASS", "PASS"], results
    assert game.teleports == [[0.0, 2.0, 20.0]], "teleported onto the checkpoint, one metre up, and nowhere else"
    assert game.item_active is True, "the respawn is what switched the room on"
    assert "active=True" in results[0][3] and "held" in results[0][3]
    assert script.exit_code(results) == 0


def test_the_checkpoint_teleport_alone_is_not_enough():
    """The control for the option: teleporting onto the checkpoint without the respawn leaves the item dead.

    This is the trap the last in-game attempt fell into, so it is pinned rather than described in a comment.
    """
    script, game, results = run(client_cls=CheckpointLevel)
    assert statuses(results) == ["FAIL", "SKIP", "SKIP"], results
    assert game.teleports == [], "nothing was teleported at all without --from-checkpoint"
    assert "the item is NOT active" in results[0][3] and "still switched off" in results[0][3]


def test_approach_stages_by_teleport_but_the_last_metres_are_still_walked():
    script, game, results = run("--from-checkpoint", *CHECKPOINT, "--approach", "0", "1", "8",
                                client_cls=CheckpointLevel)
    assert statuses(results) == ["PASS", "PASS", "PASS"], results
    assert game.teleports == [[0.0, 2.0, 20.0], [0.0, 1.0, 8.0]], "the checkpoint, then the staging point"
    assert "staged by teleport to (0.0, 1.0, 8.0)" in results[0][3]
    assert "the last metres to the item were walked from there" in results[0][3]
    walked = int(results[0][3].split("walked ", 1)[1].split(" ", 1)[0])
    assert walked >= 4, f"only {walked} decisions of walking from 12 m out: it did not walk in"


def test_the_punch_is_aimed_from_the_camera_not_from_the_player_transform():
    """`Punch.ActiveFrame` rays from `cc.GetDefaultPos()`. Aiming from `player.pos` is what made the first two
    in-game attempts FAIL at a live pedestal 1.8 m away, with rendering both off AND on."""
    script = load_script()
    assert script.CAMERA_HEIGHT == 0.9, "measured in game on 1-1's red pedestal, 2026-09-17"
    assert script.eye({"pos": [10.0, -3.0, 20.0]}, script.CAMERA_HEIGHT) == [10.0, -2.1, 20.0]
    assert script.eye({"pos": [10.0, -3.0, 20.0]}, 0.0) == [10.0, -3.0, 20.0], "0 is the old, wrong behaviour"
    # The error this corrects, at the range it was measured at: a skull 0.8 m up and 1.57 m away reads +27 deg
    # of elevation from the player transform and -3 deg from the camera -- 30 degrees apart at punch range.
    from_player = script.elevation_to([81.2, -3.0, 273.4], [81.0, -2.2, 275.0])
    from_camera = script.elevation_to(script.eye({"pos": [81.2, -3.0, 273.4]}, 0.9), [81.0, -2.2, 275.0])
    assert 25 < from_player < 28 and -5 < from_camera < -2, (from_player, from_camera)
    assert from_player - from_camera > 25, "the correction is worth ~30 degrees at punch range"


def test_command_line_defaults_are_the_specs_coordinates():
    script = load_script()
    args = script.parse_args([])
    assert args.level == "Level 1-1" and args.port == 47800
    assert args.target == [81.0, -2.2, 275.0], "section 8 check 1's red pedestal"
    assert args.altar == [0.0, -6.76, 381.0] and args.gate == "20,-10,381", "check 2's altar and its gate"
    assert args.via is None and args.teleport_assist is False, "it walks; the nudge is opt-in"
    assert args.from_checkpoint is None and args.approach is None, "both teleporting options are opt-in too"
    assert args.altar_approach is None and args.camera_height == 0.9, "and the eye offset defaults to the game's"

    cfg = script.build_config(args)
    assert (cfg.mode, cfg.level, cfg.fixed_fps, cfg.frameskip) == ("campaign", "Level 1-1", 30, 2)
    assert cfg.render is False, "training settings: check 1 is about the cameras being off"
    assert cfg.soft_death is False, "check 3 needs a real death, not a healed one"
    assert cfg.subgoal_punch_range_m == 4.0 and cfg.difficulty == 3 and cfg.unlock_all_gear is True
    assert (cfg.fresh_start_prob, cfg.explore_dir, cfg.best_runs_dir) == (1.0, "", "")

    overridden = script.build_config(script.parse_args(["--port", "47801", "--fixed-fps", "60", "--frameskip", "4",
                                                        "--render", "--level", "Level 0-2"]))
    assert (overridden.port, overridden.fixed_fps, overridden.frameskip) == (47801, 60.0, 4)
    assert overridden.render is True and overridden.level == "Level 0-2"


def test_the_probe_walks_and_never_teleports():
    """A bare teleport into a switched-off room activates nothing (measured on 0-1), so the default must walk.

    `FakeLevel` implements no `teleport` at all, so any call would raise AttributeError and take the run with
    it: a green run above is proof the probe drove the whole way on movement input.
    """
    assert not hasattr(corridor.FakeLevel, "teleport")
    script, game, results = run()
    assert statuses(results) == ["PASS", "PASS", "PASS"], results


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
