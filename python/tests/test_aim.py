"""Aim-reward geometry and the pitch clamp. No game needed:  python tests/test_aim.py  (or pytest)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import clamp_pitch_command  # noqa: E402
from ultrakill_ai.rewards import RewardConfig, aim_errors, compute_reward, horizon_elevation  # noqa: E402


def camera(yaw_deg: float, pitch_up_deg: float):
    """Camera basis (right, up, forward) in Unity axes for a camera at the given yaw and pitch (positive up)."""
    psi, phi = math.radians(yaw_deg), math.radians(pitch_up_deg)
    forward = (math.cos(phi) * math.sin(psi), math.sin(phi), math.cos(phi) * math.cos(psi))
    up = (-math.sin(phi) * math.sin(psi), math.cos(phi), -math.sin(phi) * math.cos(psi))
    right = (math.cos(psi), 0.0, -math.sin(psi))
    return right, up, forward


def snapshot(yaw_deg: float, pitch_up_deg: float, enemy_world: tuple[float, float, float]):
    """Player and enemy dicts the way the mod reports them, for an enemy at a world offset from the eye."""
    right, up, forward = camera(yaw_deg, pitch_up_deg)

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    rel = (dot(enemy_world, right), dot(enemy_world, up), dot(enemy_world, forward))
    player = {"pos": [0.0, 0.0, 0.0], "forward": list(forward), "yaw": yaw_deg, "pitch": pitch_up_deg, "hp": 100, "dead": False}
    enemy = {"id": 1, "health": 10.0, "visible": True, "pos": list(enemy_world), "rel": list(rel), "dist": math.sqrt(dot(enemy_world, enemy_world))}
    return player, enemy


def ground_enemy(bearing_deg: float, dist: float = 10.0, height: float = 0.0):
    b = math.radians(bearing_deg)
    return (dist * math.sin(b), height, dist * math.cos(b))


def test_level_camera_enemy_to_the_side():
    angle, yaw_err, pitch_err = aim_errors(*snapshot(0.0, 0.0, ground_enemy(30.0)))
    assert abs(angle - 30.0) < 1e-6 and abs(yaw_err - 30.0) < 1e-6 and abs(pitch_err) < 1e-6


def test_camera_pitched_up_enemy_ahead_is_a_pitch_error_only():
    angle, yaw_err, pitch_err = aim_errors(*snapshot(45.0, 80.0, ground_enemy(45.0)))
    assert abs(angle - 80.0) < 1e-6 and abs(yaw_err) < 1e-6 and abs(pitch_err - 80.0) < 1e-6


def test_straight_up_falls_back_to_the_yaw_angle():
    angle, yaw_err, pitch_err = aim_errors(*snapshot(120.0, 90.0, ground_enemy(120.0)))
    assert abs(angle - 90.0) < 1e-6 and abs(yaw_err) < 1e-4 and abs(pitch_err - 90.0) < 1e-6


def test_yaw_and_pitch_errors_are_independent():
    el, az = math.radians(20), math.radians(45)
    enemy = (-10.0 * math.cos(el) * math.sin(az), 10.0 * math.sin(el), 10.0 * math.cos(el) * math.cos(az))
    _, yaw_err, pitch_err = aim_errors(*snapshot(0.0, 0.0, enemy))
    assert abs(yaw_err - 45.0) < 1e-6 and abs(pitch_err - 20.0) < 1e-6


def test_yaw_error_ignores_pitch():
    _, yaw_err, pitch_err = aim_errors(*snapshot(10.0, 60.0, ground_enemy(40.0)))
    assert abs(yaw_err - 30.0) < 1e-6
    assert 40.0 < pitch_err < 90.0  # the enemy is below the camera horizon; the exact value depends on the yaw offset


def test_camera_space_fallback_without_world_positions():
    player, enemy = snapshot(0.0, 0.0, ground_enemy(-25.0))
    del enemy["pos"]
    _, yaw_err, pitch_err = aim_errors(player, enemy)
    assert abs(yaw_err - 25.0) < 1e-6 and abs(pitch_err) < 1e-6


def test_reward_parts_split_by_axis():
    cfg = RewardConfig(aim=0.0, aim_yaw=1.0, aim_pitch=1.0, aim_locked=1.0, aim_cone_deg=20.0)

    def parts(yaw, pitch, enemy):
        player, e = snapshot(yaw, pitch, enemy)
        cur = {"player": player, "enemies": [e], "stats": {}}
        return compute_reward(cfg, cur, cur, {1: 10.0}).parts

    sky = parts(0.0, 80.0, ground_enemy(0.0))
    assert abs(sky["aim_yaw"] - 1.0) < 1e-6 and abs(sky["aim_pitch"] - (1 - 80 / 90)) < 1e-6 and "aim_locked" not in sky
    side = parts(0.0, 0.0, ground_enemy(30.0))
    assert abs(side["aim_yaw"] - (1 - 30 / 180)) < 1e-6 and abs(side["aim_pitch"] - 1.0) < 1e-6 and "aim_locked" not in side
    locked = parts(0.0, 0.0, ground_enemy(5.0))
    assert abs(locked["aim_locked"] - 0.75) < 1e-6
    assert "aim" not in locked  # weight 0 adds nothing


def test_horizon_elevation_ignores_camera_pitch_and_yaw():
    for yaw, pitch in ((0.0, 0.0), (70.0, 30.0), (-120.0, -12.0), (33.0, 60.0)):
        assert abs(horizon_elevation(*snapshot(yaw, pitch, ground_enemy(20.0)))) < 1e-6
        above = ground_enemy(-50.0, dist=10.0 * math.cos(math.radians(25)), height=10.0 * math.sin(math.radians(25)))
        assert abs(horizon_elevation(*snapshot(yaw, pitch, above)) - 25.0) < 1e-6
        below = ground_enemy(10.0, dist=8.0, height=-8.0 * math.tan(math.radians(15)))
        assert abs(horizon_elevation(*snapshot(yaw, pitch, below)) + 15.0) < 1e-6
    assert horizon_elevation(*snapshot(0.0, 90.0, ground_enemy(0.0))) is None


def test_pitch_clamp():
    assert clamp_pitch_command(0.0, 20.0, 40.0) == 20.0
    assert clamp_pitch_command(30.0, 20.0, 40.0) == 10.0
    assert clamp_pitch_command(40.0, 6.0, 40.0) == 0.0
    assert clamp_pitch_command(89.0, 6.0, 40.0) == -49.0  # outside the band: snap back in one step
    assert clamp_pitch_command(-89.0, -20.0, 40.0) == 49.0
    assert clamp_pitch_command(-10.0, -1.5, 40.0) == -1.5


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
