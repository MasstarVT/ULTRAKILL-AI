"""Observation layouts and the campaign block. No game needed:  python tests/test_spaces.py  (or pytest)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC,
    ACTION_NVEC_CAMPAIGN,
    BUTTONS,
    CAMPAIGN_BLOCK,
    PITCH_BINS,
    YAW_BINS,
    ObsLayout,
    action_space,
    campaign_block,
    decode_action,
    noop_action,
    pack_observation,
    yaw_frame,
)

EXPLORE = [0.05 * (k + 1) for k in range(9)]


def target(pos=(0.0, 0.0, 50.0), hops=3, open=False, locked=False) -> dict:
    """A GateProgress target, as the env threads it into the packer."""
    return {"key": "0,0,50", "pos": list(pos), "hops": hops, "open": open, "locked": locked, "active": True}


def exit_target(pos=(0.0, 0.0, 195.0)) -> dict:
    return {"key": "exit", "pos": list(pos), "hops": None, "open": False, "locked": False, "active": True}


def player(pos=(0.0, 0.0, 0.0), yaw: float = 0.0) -> dict:
    return {
        "pos": list(pos), "yaw": yaw, "pitch": 0.0, "local_vel": [0.0, 0.0, 0.0], "hp": 100, "anti_hp": 0,
        "stamina": 300.0, "grounded": True, "sliding": False, "weapon_slot": 1, "dead": False,
    }


def level(**values) -> dict:
    """A campaign block the way the mod reports it, empty apart from the given values."""
    block = {
        "mission": 1, "difficulty": 3, "seconds": 0.0, "timer_running": False, "level_started": True,
        "level_over": False, "restarts": 0, "input_locked": False, "exit": None, "checkpoints": [],
        "path": {"status": "none"}, "locked_doors": [], "arena_enemies_alive": 0, "cleared_arenas": [],
        "unlocked_doors": [], "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]},
    }
    block.update(values)
    return block


def snapshot(campaign: dict | None = None, pos=(0.0, 0.0, 0.0), yaw: float = 0.0, cybergrind: dict | None = None) -> dict:
    obs = {"player": player(pos, yaw), "enemies": [], "rays": [10.0] * 16, "ground_rays": [1.0] * 8, "stats": {}}
    if campaign is not None:
        obs["campaign"] = campaign
    if cybergrind is not None:
        obs["cybergrind"] = cybergrind
    return obs


def exit_at(x: float, y: float, z: float) -> dict:
    return {"pos": [x, y, z], "active": True}


def close(actual, expected, tol: float = 1e-5) -> bool:
    return len(actual) == len(expected) and all(abs(a - e) <= tol for a, e in zip(actual, expected))


def test_layout_sizes():
    assert CAMPAIGN_BLOCK == 36
    assert ObsLayout().size == 448 and ObsLayout().mode_size == 7
    assert ObsLayout(campaign=True).size == 479 and ObsLayout(campaign=True).mode_size == 38
    assert ObsLayout(campaign=True).space().shape == (479,)
    assert "campaign" not in ObsLayout(campaign=True).mod_config()  # the mod sends the block whatever the layout


def test_cybergrind_packing_leaves_the_last_5_values_zero():
    obs = snapshot(level(exit=exit_at(0.0, 0.0, 50.0), timer_running=True, seconds=60.0), cybergrind={"wave": 3, "enemies_left": 6})
    out = pack_observation(obs, ObsLayout(), {}, explore=EXPLORE)
    assert out.shape == (448,)
    assert close(out[441:443], [0.1, 0.2])
    assert not out[443:].any()


def test_campaign_block_is_packed_after_the_unchanged_prefix():
    obs = snapshot(level(exit=exit_at(0.0, 0.0, 50.0), timer_running=True, seconds=60.0), pos=(3.0, 1.0, -2.0), yaw=20.0)
    grind = pack_observation(obs, ObsLayout(), {})
    out = pack_observation(obs, ObsLayout(campaign=True), {}, explore=EXPLORE, target=target())
    assert out.shape == (479,)
    assert np.array_equal(out[:443], grind[:443])  # player 0-16, enemies 17-416, rays 417-432, ground rays 433-440, Cyber Grind 441-442
    assert close(out[443:], campaign_block(obs, EXPLORE, target()))
    assert out[443 + 4] == 1.0 and out[443 + 24] == 1.0  # exit mask, timer running
    assert out[443 + 9] == 1.0  # target mask, at absolute index 452


def test_only_the_target_slots_changed_meaning():
    """Every index outside 448-455 packs exactly what it did before the route gates existed."""
    obs = snapshot(level(
        exit=exit_at(10.0, 2.0, 45.0), timer_running=True, seconds=60.0, arena_enemies_alive=4,
        checkpoints=[{"id": "0,0,90", "pos": [0.0, 0.0, 90.0], "activated": False, "current": False}],
        locked_doors=[{"pos": [5.0, 0.0, 0.0], "dist": 5.0}],
    ), pos=(3.0, 1.0, -2.0), yaw=20.0)
    without = pack_observation(obs, ObsLayout(campaign=True), {}, explore=EXPLORE)
    with_target = pack_observation(obs, ObsLayout(campaign=True), {}, explore=EXPLORE, target=target())
    assert np.array_equal(without[:448], with_target[:448])
    assert np.array_equal(without[456:], with_target[456:])
    assert not without[448:456].any() and with_target[448:456].any()


def test_exit_straight_ahead_at_yaw_0():
    block = campaign_block(snapshot(level(exit=exit_at(10.0, 2.0, 45.0)), pos=(10.0, 2.0, -5.0)))
    assert close(block[0:5], [0.0, 0.0, 0.5, 0.25, 1.0])


def test_exit_follows_the_yaw_frame():
    ahead = campaign_block(snapshot(level(exit=exit_at(50.0, 0.0, 0.0)), yaw=90.0))
    assert close(ahead[0:5], [0.0, 0.0, 0.5, 0.25, 1.0])  # yaw 90 faces +x
    right = campaign_block(snapshot(level(exit=exit_at(0.0, 0.0, -30.0)), yaw=90.0))
    assert close(right[0:5], [0.3, 0.0, 0.0, 0.15, 1.0])  # facing +x, -z is on the right
    assert close(campaign_block(snapshot(level(exit=exit_at(30.0, 0.0, 0.0))))[0:3], [0.3, 0.0, 0.0])  # yaw 0: +x is right
    above = campaign_block(snapshot(level(exit=exit_at(0.0, 20.0, 0.0)), yaw=-135.0))
    assert close(above[0:4], [0.0, 0.2, 0.0, 0.1])
    delta, yaw = (12.0, -4.0, 30.0), 37.0
    exact = campaign_block(snapshot(level(exit=exit_at(*delta)), yaw=yaw))
    assert close(exact[0:4], [v / 100.0 for v in yaw_frame(delta, yaw)] + [math.sqrt(sum(d * d for d in delta)) / 200.0])


def test_every_target_uses_the_yaw_frame():
    pos, yaw, point = (4.0, 1.0, -3.0), 90.0, (4.0, 3.0, 17.0)  # 20 m along +z and 2 m up; facing +x, +z is on the left
    rel = yaw_frame(tuple(t - p for t, p in zip(point, pos)), yaw)
    assert close(rel, [-20.0, 2.0, 0.0])
    block = campaign_block(snapshot(level(
        exit=exit_at(*point),
        checkpoints=[{"id": "4,3,17", "pos": list(point), "activated": False, "current": False}],
        locked_doors=[{"pos": list(point), "dist": 20.1}],
    ), pos=pos, yaw=yaw), target=target(point))
    for start, scale in ((0, 100.0), (5, 50.0), (13, 100.0), (18, 50.0)):  # exit, route target, checkpoint, door
        assert close(block[start : start + 3], [v / scale for v in rel])


def test_missing_exit_gives_mask_0():
    assert close(campaign_block(snapshot(level(exit=None)))[0:5], [0.0] * 5)
    no_key = level()
    del no_key["exit"]
    assert close(campaign_block(snapshot(no_key))[0:5], [0.0] * 5)


def test_target_slots_carry_the_route_gate():
    block = campaign_block(snapshot(level()), target=target((0.0, 0.0, 40.0), hops=6, open=True, locked=True))
    assert close(block[5:13], [0.0, 0.0, 40.0 / 50.0, 40.0 / 100.0, 1.0, 1.0, 1.0, 6.0 / 20.0])
    closed = campaign_block(snapshot(level()), target=target((-15.0, 3.0, 20.0), hops=0))
    assert close(closed[5:9], [-15.0 / 50.0, 3.0 / 50.0, 20.0 / 50.0, math.sqrt(15**2 + 3**2 + 20**2) / 100.0])
    assert close(closed[9:13], [1.0, 0.0, 0.0, 0.0])


def test_target_slots_use_the_exit_scales_for_the_exit():
    """0-1's exit is ~195 m from spawn: at the gate scales that would put the slots near 4.0."""
    block = campaign_block(snapshot(level()), target=exit_target((0.0, 0.0, 195.0)))
    assert close(block[5:13], [0.0, 0.0, 1.95, 0.975, 1.0, 0.0, 0.0, 0.0])
    far = campaign_block(snapshot(level()), target=exit_target((0.0, 0.0, 900.0)))
    assert close(far[5:9], [0.0, 0.0, 4.0, 4.0])  # clipped to +-4.0 as a guard


def test_no_target_leaves_the_slots_zero():
    assert close(campaign_block(snapshot(level()))[5:13], [0.0] * 8)
    assert close(campaign_block(snapshot(level()), target=None)[5:13], [0.0] * 8)
    # The NavMesh path is no longer packed at all: its slots belong to the target now.
    with_path = campaign_block(snapshot(level(path={"status": "complete", "length": 84.2, "next_corner": [0.0, 0.0, 10.0]})))
    assert close(with_path[5:13], [0.0] * 8)


def test_checkpoint_slot_picks_the_nearest_pending_checkpoint():
    checkpoints = [
        {"id": "0,0,5", "pos": [0.0, 0.0, 5.0], "activated": True, "current": False},
        {"id": "0,0,-8", "pos": [0.0, 0.0, -8.0], "activated": False, "current": True},
        {"id": "0,0,90", "pos": [0.0, 0.0, 90.0], "activated": False, "current": False},
        {"id": "0,0,-40", "pos": [0.0, 0.0, -40.0], "activated": False, "current": False},
    ]
    block = campaign_block(snapshot(level(checkpoints=checkpoints)))
    assert close(block[13:18], [0.0, 0.0, -0.4, 0.2, 1.0])
    reached = campaign_block(snapshot(level(checkpoints=checkpoints[:2])))
    assert close(reached[13:18], [0.0] * 5)


def test_locked_door_slot_uses_the_first_entry():
    doors = [{"pos": [5.0, 0.0, 0.0], "dist": 5.0}, {"pos": [0.0, 0.0, 2.0], "dist": 2.0}]
    block = campaign_block(snapshot(level(locked_doors=doors)))
    assert close(block[18:23], [0.1, 0.0, 0.0, 0.05, 1.0])  # the mod sorts them nearest first; the block keeps its order
    assert close(campaign_block(snapshot(level(locked_doors=[])))[18:23], [0.0] * 5)


def test_arena_timer_input_and_seconds():
    block = campaign_block(snapshot(level(arena_enemies_alive=5, timer_running=True, input_locked=True, seconds=120.0)))
    assert close(block[23:27], [0.25, 1.0, 1.0, 0.2])
    assert close(campaign_block(snapshot(level()))[23:27], [0.0, 0.0, 0.0, 0.0])


def test_explore_features_fill_the_last_9():
    assert close(campaign_block(snapshot(level()), EXPLORE)[27:36], EXPLORE)
    assert close(campaign_block(snapshot(level()))[27:36], [0.0] * 9)
    longer = campaign_block(snapshot(level()), EXPLORE + [1.0])
    assert len(longer) == CAMPAIGN_BLOCK and close(longer[27:36], EXPLORE)


def test_no_campaign_block_gives_36_zeros():
    assert campaign_block(snapshot(), EXPLORE) == [0.0] * 36
    no_player = snapshot(level(exit=exit_at(0.0, 0.0, 50.0)))
    del no_player["player"]
    assert campaign_block(no_player, EXPLORE, target()) == [0.0] * 36
    out = pack_observation(snapshot(), ObsLayout(campaign=True), {}, explore=EXPLORE, target=target())
    assert out.shape == (479,) and not out[443:].any()


def test_cybergrind_action_space_is_unchanged():
    """Widening the module constant would make every Cyber Grind checkpoint unloadable against its own env."""
    assert len(ACTION_NVEC) == 11 and int(ACTION_NVEC.sum()) == 42
    assert list(ACTION_NVEC) == [3, 3, 2, 2, 2, 2, 2, 2, 6, len(YAW_BINS), len(PITCH_BINS)]
    assert np.array_equal(action_space().nvec, ACTION_NVEC)
    assert np.array_equal(action_space(campaign=False).nvec, ACTION_NVEC)


def test_campaign_action_space_appends_the_look_mode():
    assert len(ACTION_NVEC_CAMPAIGN) == 12 and int(ACTION_NVEC_CAMPAIGN.sum()) == 45
    assert np.array_equal(ACTION_NVEC_CAMPAIGN[:11], ACTION_NVEC)  # every existing row keeps its index
    assert ACTION_NVEC_CAMPAIGN[11] == 3
    assert np.array_equal(action_space(campaign=True).nvec, ACTION_NVEC_CAMPAIGN)


def test_noop_action_is_neutral_at_both_widths():
    for campaign, width in ((False, 11), (True, 12)):
        a = noop_action(campaign)
        assert len(a) == width
        decoded = decode_action(a)
        assert decoded["move"] == [0, 0] and decoded["buttons"] == []
        assert decoded["look"] == [0.0, 0.0]  # not the negative-index slots, which are wrong at width 12
        assert decoded["look_mode"] == 0


def test_decode_action_infers_the_width():
    campaign = noop_action(campaign=True)
    campaign[2 + len(BUTTONS) + 3] = 2
    assert decode_action(campaign)["look_mode"] == 2
    assert decode_action(noop_action())["look_mode"] == 0  # an 11-wide vector has no look mode
    turning = noop_action(campaign=True)
    turning[2 + len(BUTTONS) + 1] = YAW_BINS.index(30.0)
    turning[2 + len(BUTTONS) + 2] = PITCH_BINS.index(-6.0)
    assert decode_action(turning)["look"] == [30.0, -6.0]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
