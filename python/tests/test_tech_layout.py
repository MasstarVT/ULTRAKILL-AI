"""The S7 tech layout: v1 pinned byte for byte, and v2's index maps. No game:  python tests/test_tech_layout.py

The v1 half is a PIN, not a behaviour test: it was written against the unchanged spaces.py and must pass both
before and after the tech layout lands. The live trainer re-imports this package at every round boundary, so a
v1 packing or action table that moved by one float would reach twelve games unannounced. The hashes below were
taken on 2026-09-23 from exactly this fixture; if one fails on a clean main, do not "update the hash" -- find
what changed the v1 vector.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC,
    ACTION_NVEC_CAMPAIGN,
    ObsLayout,
    action_space,
    decode_action,
    noop_action,
    pack_observation,
)

PIN_RAW = {
    "player": {
        "pos": [10.0, 2.0, 30.0], "vel": [3.0, -1.0, 12.0], "local_vel": [1.5, -1.0, 12.5],
        "forward": [0.0, 0.0, 1.0], "yaw": 37.5, "pitch": -12.0, "hp": 83, "anti_hp": 12.0,
        "stamina": 212.5, "grounded": False, "sliding": True, "weapon_slot": 3, "dead": False,
        "heavy_fall": True, "weapon_variation": 1, "slot_counts": [3, 3, 3, 3, 3, 0],
    },
    "enemies": [
        {"id": 101, "type": 0, "health": 0.75, "visible": True, "rel": [4.0, 1.0, 9.0], "dist": 10.0,
         "pos": [14.0, 3.0, 39.0]},
        {"id": 102, "type": 7, "health": 2.5, "visible": False, "rel": [-20.0, 0.5, 30.0], "dist": 36.1,
         "pos": [-10.0, 2.5, 60.0]},
        {"id": 103, "type": 42, "health": 10.0, "visible": True, "rel": [0.0, 8.0, 45.0], "dist": 45.7,
         "pos": [10.0, 10.0, 75.0]},
    ],
    "rays": [float(3 + 2 * k) for k in range(16)],
    "ground_rays": [1.0, 1.5, 30.0, 2.0, 45.0, -1.0, 0.5, 12.0],
    "ground_ray_center": 1.2,
    "stats": {"kills": 4, "style": 120, "seconds": 41.5, "restarts": 0, "level_complete": False},
    "campaign": {
        "exit": {"pos": [60.0, -5.0, 220.0], "active": True},
        "checkpoints": [
            {"id": "a", "pos": [12.0, 2.0, 80.0], "activated": False, "current": False},
            {"id": "b", "pos": [0.0, 0.0, 5.0], "activated": True, "current": True},
        ],
        "locked_doors": [{"pos": [20.0, 2.0, 50.0]}],
        "arena_enemies_alive": 3, "timer_running": True, "input_locked": False, "seconds": 41.5,
    },
}
PIN_EXPLORE = [0.1, 0.0, 0.25, 0.5, 1.0, 0.0, 0.75, 0.2, 0.05]
PIN_TARGET = {"key": "40,1,408", "pos": [40.0, 1.0, 108.5], "hops": 9, "open": True, "locked": False,
              "active": True}
PIN_ENEMY_MAX = {101: 1.0, 102: 5.0}
V1_CAMPAIGN_SHA256 = "ff3c2c8040590275974a586fe0351d6324521cecdb28c9119b861ab791265644"
V1_GRIND_SHA256 = "3fee7c7fcf166e5bd92e9faa658965c9fbfe76455ed7f46f0c9dd140c74093c6"
V1_CAMPAIGN_NVEC = (3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3)
V1_DECODE_CASES = [
    ([1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0],
     {"move": [0, 0], "buttons": [], "slot": 0, "look": [0.0, 0.0], "look_mode": 0}),
    ([2, 0, 1, 0, 1, 1, 0, 1, 5, 10, 0, 2],
     {"move": [-1, 1], "buttons": ["jump", "slide", "fire1", "punch"], "slot": 5, "look": [90.0, -20.0],
      "look_mode": 2}),
    ([0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 6, 1],
     {"move": [1, -1], "buttons": ["dash", "fire2"], "slot": 0, "look": [-90.0, 20.0], "look_mode": 1}),
    ([1, 2, 0, 0, 0, 0, 0, 1, 2, 4, 5],  # Cyber Grind, 11 wide
     {"move": [1, 0], "buttons": ["punch"], "slot": 2, "look": [-1.0, 6.0], "look_mode": 0}),
]


def sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def pinned_campaign_vector() -> np.ndarray:
    return pack_observation(PIN_RAW, ObsLayout(campaign=True), PIN_ENEMY_MAX, PIN_EXPLORE, PIN_TARGET)


def test_v1_campaign_packing_is_pinned_byte_for_byte():
    out = pinned_campaign_vector()
    assert out.shape == (479,) and out.dtype == np.float32
    assert sha(out) == V1_CAMPAIGN_SHA256, sha(out)


def test_v1_cyber_grind_packing_is_pinned_byte_for_byte():
    out = pack_observation(PIN_RAW, ObsLayout(), PIN_ENEMY_MAX)
    assert out.shape == (448,)
    assert sha(out) == V1_GRIND_SHA256, sha(out)


def test_v1_action_tables_are_pinned():
    assert tuple(int(v) for v in ACTION_NVEC) == V1_CAMPAIGN_NVEC[:11]
    assert tuple(int(v) for v in ACTION_NVEC_CAMPAIGN) == V1_CAMPAIGN_NVEC
    assert tuple(int(v) for v in action_space(campaign=True).nvec) == V1_CAMPAIGN_NVEC
    assert tuple(int(v) for v in action_space().nvec) == V1_CAMPAIGN_NVEC[:11]
    assert ObsLayout(campaign=True).size == 479 and ObsLayout().size == 448


def test_v1_decoding_is_pinned():
    for vector, expected in V1_DECODE_CASES:
        assert decode_action(np.array(vector)) == expected, vector
    assert list(noop_action(campaign=True)) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0]
    assert list(noop_action()) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3]


from ultrakill_ai.rewards import macro_landed  # noqa: E402
from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC_TECH,
    HOOK_INDEX,
    HOOK_ROWS,
    MACRO_INDEX,
    MACRO_ROWS,
    MACROS,
    MOVE_TECH_FIELDS,
    TECH_BLOCK,
    TECH_HAZARD,
    TECH_LAYOUTS,
    TECH_MACRO,
    TECH_MOVE,
    TECH_PROJECTILES,
    TECH_WEAPON,
    VARIANT_INDEX,
    VARIANT_ROWS,
    pinned_action_rows,
    tech_block,
)

MOVE_TECH = {
    "heavy_fall": True, "slam_force": 6.5, "bounce_window": True, "coyote": 0.03, "wall_jumps": 2,
    "wall_available": True, "boost": False, "boost_left": 36.0, "pre_slide_speed": 4.5, "jump_cooldown": True,
    "slide_grace": 0.25, "riding_rocket": False,
    # diagnostics the packer must never read (docs/protocol.md): wall-clock seconds and absolute stamps
    "slide_since": 0.08, "slide_timestamp": 5321.25, "jump_timestamp": 5321.3,
    "slam_storage": False, "can_jump": True, "ssj": None, "ssj_last": None,
}
EXPECTED_A = [1.0, 0.65, 1.0, 0.03, 2.0 / 3.0, 1.0, 0.0, 0.36, 1.5, 1.0, 0.25, 0.0]
PLAYER = {"hp": 100}


def test_v2_layout_is_530_with_the_tech_block_at_479():
    v2 = ObsLayout(campaign=True, tech=True)
    assert v2.size == 530 and v2.campaign_start == 443 and v2.tech_start == 479
    assert TECH_BLOCK == 51 and TECH_LAYOUTS == ("v1", "v2")
    assert (TECH_MOVE, TECH_WEAPON, TECH_PROJECTILES, TECH_MACRO, TECH_HAZARD) == (
        slice(0, 12), slice(12, 22), slice(22, 40), slice(40, 43), slice(43, 51))
    assert [name for name, *_ in MOVE_TECH_FIELDS] == [
        "heavy_fall", "slam_force", "bounce_window", "coyote", "wall_jumps", "wall_available", "boost",
        "boost_left", "pre_slide_speed", "jump_cooldown", "slide_grace", "riding_rocket"]


def test_v2_action_space_is_15_dims_57_logits_appended():
    assert tuple(int(v) for v in ACTION_NVEC_TECH) == V1_CAMPAIGN_NVEC + (6, 4, 2)
    assert int(ACTION_NVEC_TECH.sum()) == 57
    assert (MACRO_INDEX, VARIANT_INDEX, HOOK_INDEX) == (12, 13, 14)
    assert (list(MACRO_ROWS), list(VARIANT_ROWS), list(HOOK_ROWS)) == (
        list(range(45, 51)), list(range(51, 55)), [55, 56])
    assert MACROS == ("none", "ssj", "ssj_wall", "core_nuke", "rocket_down", "coin_rocket")
    assert tuple(int(v) for v in action_space(campaign=True, tech=True).nvec) == V1_CAMPAIGN_NVEC + (6, 4, 2)


def test_the_tech_action_space_is_campaign_only():
    try:
        action_space(campaign=False, tech=True)
    except ValueError as exc:
        assert "campaign" in str(exc)
    else:
        raise AssertionError("a Cyber Grind tech action space must be refused")


def test_v2_decoding_keeps_every_v1_key_and_adds_three():
    vector, expected = V1_DECODE_CASES[1]
    out = decode_action(np.array(vector + [1, 3, 1]))
    assert {k: out[k] for k in expected} == expected
    assert (out["macro"], out["variant"], out["hook"]) == (1, 3, True)
    assert set(out) == set(expected) | {"macro", "variant", "hook"}
    assert list(noop_action(campaign=True, tech=True)) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0]


def test_v2_packing_keeps_indices_0_to_478_exactly():
    raw = {**PIN_RAW, "move_tech": MOVE_TECH,
           "macro": {"requested": "ssj", "result": "ran", "ssj_bucket": 1, "ssj_landed": True}}
    v2 = pack_observation(raw, ObsLayout(campaign=True, tech=True), PIN_ENEMY_MAX, PIN_EXPLORE, PIN_TARGET)
    assert v2.shape == (530,)
    assert sha(v2[:479]) == V1_CAMPAIGN_SHA256, "the tech keys must not move one v1 float"
    assert np.allclose(v2[479:491], EXPECTED_A, atol=1e-6)
    assert np.allclose(v2[519:522], [1.0, 0.0, 1.0 / 3.0], atol=1e-6)
    assert not v2[491:519].any() and not v2[522:530].any()


def test_block_a_packs_the_twelve_fields_in_order():
    block = tech_block({"player": PLAYER, "move_tech": MOVE_TECH})
    assert len(block) == TECH_BLOCK
    assert np.allclose(block[0:12], EXPECTED_A, atol=1e-6)


def test_block_a_clips_every_scale():
    extreme = {**MOVE_TECH, "slam_force": 50.0, "coyote": 999.0, "wall_jumps": 7, "boost_left": 250.0,
               "pre_slide_speed": 30.0, "slide_grace": 3.0}
    block = tech_block({"player": PLAYER, "move_tech": extreme})
    assert block[1] == 1.0 and block[3] == 1.0 and block[4] == 1.0 and block[7] == 1.0
    assert block[8] == 2.0 and block[10] == 1.0
    negative = {**MOVE_TECH, "slam_force": -3.0, "coyote": -1.0, "pre_slide_speed": -2.0}
    low = tech_block({"player": PLAYER, "move_tech": negative})
    assert low[1] == 0.0 and low[3] == 0.0 and low[8] == 0.0


def test_the_wall_clock_fields_are_never_packed():
    base = tech_block({"player": PLAYER, "move_tech": MOVE_TECH})
    moved = {**MOVE_TECH, "slide_since": 9.0e9, "slide_timestamp": -1.0, "jump_timestamp": 1.0e12}
    assert tech_block({"player": PLAYER, "move_tech": moved}) == base


def test_block_d_reports_ran_not_run_and_the_landed_bucket():
    cases = [
        ({"result": "ran", "ssj_bucket": 1, "ssj_landed": True}, [1.0, 0.0, 1.0 / 3.0]),
        ({"result": "ran", "ssj_bucket": 3, "ssj_landed": True}, [1.0, 0.0, 1.0]),
        ({"result": "ran", "ssj_bucket": 0, "ssj_landed": False}, [1.0, 0.0, 0.0]),
        ({"result": "ran", "ssj_bucket": 7, "ssj_landed": True}, [1.0, 0.0, 0.0]),  # impossible: never paid
        ({"result": "refused", "reason": "not_sliding", "ssj_bucket": -1, "ssj_landed": False}, [0.0, 1.0, 0.0]),
        ({"result": "degraded", "ssj_bucket": 2, "ssj_landed": True}, [0.0, 1.0, 0.0]),
        ({"result": "disabled", "reason": "reserved", "ssj_bucket": -1}, [0.0, 1.0, 0.0]),
    ]
    for report, expected in cases:
        block = tech_block({"player": PLAYER, "macro": report})
        assert np.allclose(block[40:43], expected, atol=1e-6), report
        assert macro_landed(report) == (block[42] > 0.0), report
    assert tech_block({"player": PLAYER})[40:43] == [0.0, 0.0, 0.0]


def test_reserved_blocks_stay_zero_even_when_the_mod_sends_them():
    raw = {"player": PLAYER, "move_tech": MOVE_TECH,
           "weapon_tech": {"gun_ready": True, "coin_charge": 400, "variation": 1},
           "projectiles": [{"kind": "coin", "rel": [1.0, 2.0, 3.0], "dist": 3.7, "age": 0.2}]}
    block = tech_block(raw)
    assert not any(block[12:40]) and not any(block[43:51])


def test_an_old_dll_frame_and_a_frame_without_a_player_pack_zeros():
    assert tech_block(PIN_RAW) == [0.0] * TECH_BLOCK, "0.7.2: no move_tech, no macro"
    assert tech_block({"move_tech": MOVE_TECH}) == [0.0] * TECH_BLOCK, "no player: nothing is read"


def test_the_pinned_rows_at_each_gate_setting():
    assert pinned_action_rows({1}, False, False) == [47, 48, 49, 50, 51, 52, 53, 54, 55, 56], "the S7 gates"
    assert pinned_action_rows(set(), False, False) == list(range(45, 57)), "no live macro: the whole dim"
    assert pinned_action_rows({1, 2}, True, True) == [48, 49, 50]
    assert pinned_action_rows({1}, True, False) == [47, 48, 49, 50, 55, 56]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
