"""Weight transfer from the Cyber Grind policy to the campaign layout. No game needed:  python tests/test_transfer.py  (or pytest)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transfer_weights import SHARED_INPUTS, SpacesEnv, transfer, widen_state_dict  # noqa: E402
from ultrakill_ai.spaces import CAMPAIGN_BLOCK, ObsLayout  # noqa: E402

NET_ARCH = [16, 16]


def small_model(campaign: bool, seed: int) -> PPO:
    torch.manual_seed(seed)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=campaign).space()), policy_kwargs={"net_arch": NET_ARCH}, device="cpu")


def inputs(rows: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random Cyber Grind inputs (retired route values zero, as in live play) and the same inputs in the campaign layout."""
    rng = np.random.default_rng(seed)
    grind = rng.uniform(-1.0, 1.0, (rows, ObsLayout().size)).astype(np.float32)
    grind[:, SHARED_INPUTS:] = 0.0
    campaign = np.concatenate([grind[:, :SHARED_INPUTS], np.zeros((rows, CAMPAIGN_BLOCK), dtype=np.float32)], axis=1)
    return torch.from_numpy(grind), torch.from_numpy(campaign)


def outputs(policy, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(policy latent, value latent, action logits)."""
    with torch.no_grad():
        latent_pi, latent_vf = policy.mlp_extractor(x)
        return latent_pi, latent_vf, policy.action_net(latent_pi)


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, rtol=0.0, atol=1e-6)


def widened(action_scale: float):
    """(source model, destination model with the widened weights loaded)."""
    src, dst = small_model(campaign=False, seed=1), small_model(campaign=True, seed=2)
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), SHARED_INPUTS, action_scale))
    return src, dst


def test_layouts_share_the_first_443_inputs():
    assert ObsLayout().size == 448
    assert ObsLayout(campaign=True).size == SHARED_INPUTS + CAMPAIGN_BLOCK == 479


def test_widened_latents_match_the_source():
    src, dst = widened(1.0)
    grind, campaign = inputs()
    src_pi, src_vf, src_logits = outputs(src.policy, grind)
    dst_pi, dst_vf, dst_logits = outputs(dst.policy, campaign)
    assert close(src_pi, dst_pi) and close(src_vf, dst_vf)
    assert close(src_logits, dst_logits)


def test_action_scale_halves_the_logits():
    src, dst = widened(0.5)
    grind, campaign = inputs(seed=1)
    src_pi, src_vf, src_logits = outputs(src.policy, grind)
    dst_pi, dst_vf, dst_logits = outputs(dst.policy, campaign)
    assert close(src_pi, dst_pi) and close(src_vf, dst_vf)
    assert close(dst_logits, 0.5 * src_logits)
    assert (src_logits - dst_logits).abs().max() > 1e-4  # the scale really changed something


def test_widen_keeps_the_destinations_value_head_and_zeroes_new_inputs():
    src, dst = small_model(campaign=False, seed=1), small_model(campaign=True, seed=2)
    src_state, dst_state = src.policy.state_dict(), dst.policy.state_dict()
    dst_before = {name: value.clone() for name, value in dst_state.items()}
    out = widen_state_dict(src_state, dst_state, SHARED_INPUTS, 0.5)
    assert set(out) == set(dst_state)
    for name in ("value_net.weight", "value_net.bias"):
        assert torch.equal(out[name], dst_before[name])
    assert not torch.equal(out["value_net.weight"], src_state["value_net.weight"])
    for net in ("policy_net", "value_net"):
        first = out[f"mlp_extractor.{net}.0.weight"]
        assert first.shape == (NET_ARCH[0], 479)
        assert torch.equal(first[:, :SHARED_INPUTS], src_state[f"mlp_extractor.{net}.0.weight"][:, :SHARED_INPUTS])
        assert not first[:, SHARED_INPUTS:].any()
        assert torch.equal(out[f"mlp_extractor.{net}.2.weight"], src_state[f"mlp_extractor.{net}.2.weight"])
    for name, value in dst_before.items():  # the destination dict itself was not written to
        assert torch.equal(dst_state[name], value)


def test_transfer_saves_a_loadable_campaign_model():
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp) / "grind.zip", Path(tmp) / "campaign" / "transfer_init.zip"
        small_model(campaign=False, seed=1).save(source)
        transfer(source, dest, action_scale=0.5, seed=3)
        loaded = PPO.load(dest, device="cpu")
        assert loaded.observation_space.shape == (479,)
        assert loaded.policy_kwargs == {"net_arch": NET_ARCH}
        assert loaded.seed is None
        grind, campaign = inputs(seed=2)
        _, _, src_logits = outputs(PPO.load(source, device="cpu").policy, grind)
        _, _, dst_logits = outputs(loaded.policy, campaign)
        assert close(dst_logits, 0.5 * src_logits)

        again = Path(tmp) / "again.zip"
        transfer(source, again, action_scale=0.5, seed=3)
        assert torch.equal(PPO.load(again, device="cpu").policy.state_dict()["value_net.weight"], loaded.policy.state_dict()["value_net.weight"])

        try:
            transfer(dest, Path(tmp) / "twice.zip")
        except ValueError as e:
            assert "Cyber Grind layout" in str(e)
        else:
            raise AssertionError("a 479-input source must be rejected")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
