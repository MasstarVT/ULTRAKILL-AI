"""scripts/add_look_mode.py: the weight surgery that carries the campaign run across the action-space and
observation-slot change. No game needed:  python tests/test_look_mode_transfer.py  (or pytest).

The contract is a tensor-level one plus a bound on how far the policy moved, NOT distribution identity: inputs
448-455 carried real, varying values, so zeroing those columns displaces the policy. The mean-fold has to halve
that displacement and the result has to stay inside one PPO update (target_kl 0.03).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from add_look_mode import (  # noqa: E402
    FIRST_BIASES,
    FIRST_LAYERS,
    KL_BOUND,
    MU_MEASURED,
    TARGET_SLOTS,
    add_look_mode,
    displacement,
    legacy_target_slots,
    recorded_observations,
    widen_state_dict,
)
from transfer_weights import ACTION_HEAD, SpacesEnv  # noqa: E402
from ultrakill_ai.spaces import ACTION_NVEC, ObsLayout, action_space  # noqa: E402

NET_ARCH = [16, 16]
GRIND_LOGITS = int(ACTION_NVEC.sum())  # 42


def small_model(seed: int, campaign_actions: bool) -> PPO:
    torch.manual_seed(seed)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), action_space(campaign=campaign_actions)),
               policy_kwargs={"net_arch": NET_ARCH}, device="cpu")


def states(rows: int = 64, seed: int = 0) -> np.ndarray:
    """Random campaign observations with non-zero values in the slots being taken away."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, (rows, ObsLayout(campaign=True).size)).astype(np.float32)
    obs[:, TARGET_SLOTS] += np.asarray(MU_MEASURED, dtype=np.float32)  # around the measured live mean
    return obs


def widened(mean_fold: bool = True):
    src, dst = small_model(1, campaign_actions=False), small_model(2, campaign_actions=True)
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), MU_MEASURED, mean_fold))
    return src, dst


def test_action_head_grows_by_three_zero_rows():
    src, dst = widened()
    src_state, dst_state = src.policy.state_dict(), dst.policy.state_dict()
    for name in ACTION_HEAD:
        assert dst_state[name].shape[0] == GRIND_LOGITS + 3
        assert torch.equal(dst_state[name][:GRIND_LOGITS], src_state[name])  # bit-identical, no scaling here
        assert not dst_state[name][GRIND_LOGITS:].any()


def test_hidden_stacks_are_identical_apart_from_the_eight_columns():
    src, dst = widened()
    src_state, dst_state = src.policy.state_dict(), dst.policy.state_dict()
    for name, value in dst_state.items():
        if name in ACTION_HEAD or name in FIRST_LAYERS or name in FIRST_BIASES:
            continue
        assert torch.equal(value, src_state[name]), name  # including the critic's own output layer
    fold = torch.as_tensor(MU_MEASURED, dtype=torch.float32)
    for weight_name, bias_name in zip(FIRST_LAYERS, FIRST_BIASES):
        assert not dst_state[weight_name][:, TARGET_SLOTS].any()
        untouched = slice(0, TARGET_SLOTS.start)
        assert torch.equal(dst_state[weight_name][:, untouched], src_state[weight_name][:, untouched])
        assert torch.equal(dst_state[weight_name][:, TARGET_SLOTS.stop:], src_state[weight_name][:, TARGET_SLOTS.stop:])
        expected = src_state[bias_name] + src_state[weight_name][:, TARGET_SLOTS] @ fold
        assert torch.allclose(dst_state[bias_name], expected, atol=1e-6)


def test_without_the_mean_fold_the_bias_is_untouched():
    src, dst = widened(mean_fold=False)
    for name in FIRST_BIASES:
        assert torch.equal(dst.policy.state_dict()[name], src.policy.state_dict()[name])


def test_the_three_new_logits_are_equal_on_random_inputs():
    _, dst = widened()
    with torch.no_grad():
        latent, _ = dst.policy.mlp_extractor(torch.as_tensor(states()))
        logits = dst.policy.action_net(latent)
    new = logits[:, GRIND_LOGITS:]
    assert torch.allclose(new, torch.zeros_like(new))  # uniform over free / enemy / gate


def test_the_mean_fold_keeps_the_displacement_inside_one_update():
    obs = states(rows=256, seed=3)
    src, folded = widened(mean_fold=True)
    _, zeroed = widened(mean_fold=False)
    with_fold = displacement(src, folded, obs)
    without = displacement(src, zeroed, obs)
    assert with_fold["mean_kl"] <= KL_BOUND, with_fold
    assert with_fold["mean_kl"] < without["mean_kl"], (with_fold, without)
    assert with_fold["mean_abs_dv"] < without["mean_abs_dv"]
    assert with_fold["greedy_changed"] <= without["greedy_changed"]


def test_legacy_target_slots_reproduce_the_retired_path_packing():
    raw = {
        "player": {"pos": [0.0, 0.0, 0.0], "yaw": 0.0},
        "campaign": {"path": {"status": "partial", "length": 30.0, "next_corner": [-5.0, 0.0, 0.0]}},
    }
    assert legacy_target_slots(raw) == [-0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 1.0, 0.0]
    none = {"player": {"pos": [0.0, 0.0, 0.0], "yaw": 0.0}, "campaign": {"path": {"status": "none"}}}
    assert legacy_target_slots(none) == [0.0] * 5 + [0.0, 0.0, 1.0]
    assert legacy_target_slots({"player": {"pos": [0, 0, 0], "yaw": 0.0}}) == [0.0] * 8


def test_recorded_observations_put_the_path_back_in_the_target_slots():
    raw = {
        "player": {"pos": [0.0, 0.0, 0.0], "vel": [0, 0, 0], "local_vel": [0, 0, 0], "yaw": 0.0, "pitch": 0.0,
                   "hp": 100, "anti_hp": 0.0, "stamina": 300.0, "grounded": True, "sliding": False,
                   "weapon_slot": 1, "dead": False},
        "enemies": [], "rays": [10.0] * 16, "ground_rays": [1.0] * 8, "stats": {},
        "campaign": {"exit": {"pos": [0.0, 0.0, 50.0], "active": True}, "checkpoints": [], "locked_doors": [],
                     "path": {"status": "partial", "length": 30.0, "next_corner": [-5.0, 0.0, 0.0]}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_raw.jsonl"
        path.write_text("\n".join(json.dumps({"raw": raw}) for _ in range(3)) + "\n", encoding="utf-8")
        obs = recorded_observations(path)
    assert obs.shape == (3, 479)
    assert list(obs[0, TARGET_SLOTS]) == [-0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 1.0, 0.0]
    assert obs[0, 443 + 4] == 1.0  # the rest of the vector is packed as it is today


def test_add_look_mode_saves_a_loadable_model_that_keeps_the_step_count():
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp) / "ground.zip", Path(tmp) / "gates" / "init.zip"
        src = small_model(1, campaign_actions=False)
        src.num_timesteps = 2_914_785
        src._n_updates = 7103
        src.save(source)

        _, dst, carried = add_look_mode(source, dest, MU_MEASURED)
        assert carried is False or carried is True  # a fresh optimizer is an acceptable fallback
        loaded = PPO.load(dest, device="cpu")
        assert loaded.observation_space.shape == (479,)
        assert list(loaded.action_space.nvec) == [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3]
        assert loaded.num_timesteps == 2_914_785 and loaded._n_updates == 7103
        assert loaded.policy_kwargs == {"net_arch": NET_ARCH}
        state = loaded.policy.state_dict()
        assert state["action_net.weight"].shape == (GRIND_LOGITS + 3, NET_ARCH[-1])
        assert not state["action_net.weight"][GRIND_LOGITS:].any()
        for name in FIRST_LAYERS:
            assert not state[name][:, TARGET_SLOTS].any()

        # A model that has already been through the surgery goes through again unchanged in shape.
        again = Path(tmp) / "again.zip"
        add_look_mode(dest, again, MU_MEASURED)
        assert list(PPO.load(again, device="cpu").action_space.nvec)[-1] == 3


def test_a_cyber_grind_checkpoint_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "grind.zip"
        torch.manual_seed(5)
        PPO("MlpPolicy", SpacesEnv(ObsLayout().space(), action_space()), policy_kwargs={"net_arch": NET_ARCH},
            device="cpu").save(source)
        try:
            add_look_mode(source, Path(tmp) / "out.zip")
        except ValueError as exc:
            assert "campaign layout" in str(exc)
        else:
            raise AssertionError("a 448-input Cyber Grind checkpoint must be refused")


def test_the_optimizer_is_carried_with_zero_rows_and_zero_columns():
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp) / "ground.zip", Path(tmp) / "init.zip"
        src = small_model(1, campaign_actions=False)
        # One real optimizer step, so Adam has moments to carry rather than an empty state.
        loss = src.policy.action_net(src.policy.mlp_extractor(torch.as_tensor(states(8)))[0]).square().mean()
        src.policy.optimizer.zero_grad()
        loss.backward()
        src.policy.optimizer.step()
        src.save(source)

        _, dst, carried = add_look_mode(source, dest, MU_MEASURED)
        assert carried, "Adam's moments should carry over; a fresh optimizer is only the fallback"
        names = [name for name, _ in dst.policy.named_parameters()]
        state = dst.policy.optimizer.state_dict()["state"]
        # Only the parameters that took a gradient have Adam state; the toy loss above skips the critic stack.
        checked = 0
        for index, name in enumerate(names):
            if index not in state:
                continue
            checked += 1
            if name in ACTION_HEAD:
                assert state[index]["exp_avg"].shape[0] == GRIND_LOGITS + 3
                assert not state[index]["exp_avg"][GRIND_LOGITS:].any()
            if name in FIRST_LAYERS:
                assert not state[index]["exp_avg"][:, TARGET_SLOTS].any()
                assert not state[index]["exp_avg_sq"][:, TARGET_SLOTS].any()
        assert checked >= 4, names
        assert float(state[0]["step"]) == 1.0  # the step count carries, so bias correction does not restart


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
