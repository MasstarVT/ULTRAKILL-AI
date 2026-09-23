"""scripts/add_tech_heads.py: the S7 weight migration, on tiny SB3 models. No game:  python tests/test_add_tech_heads.py

The contract is EXACTNESS, a strictly stronger one than add_look_mode's `mean_kl <= 0.03`: nothing is repurposed,
so on the twelve shared action dims the widened policy must compute what the source computed (spec §4.4). Anything
non-zero is a copy bug. The three new heads must sit exactly on their priors, independent of the state.
"""

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

import add_tech_heads as ath  # noqa: E402
from add_tech_heads import (  # noqa: E402
    FIRST_LAYERS,
    KL_TOL,
    PRIOR_BIAS,
    PRIOR_ENTROPY,
    V1_INPUTS,
    V1_LOGITS,
    V2_INPUTS,
    V2_LOGITS,
    MigrationRefused,
    add_tech_heads,
    build_tech_model,
    checkpoint_observations,
    displacement,
    verdict,
)
from transfer_weights import SpacesEnv  # noqa: E402
from ultrakill_ai.spaces import ObsLayout, action_space  # noqa: E402

NET_ARCH = [16, 16]
V2_NVEC = [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3, 6, 4, 2]


def v1_model(seed: int = 1) -> PPO:
    torch.manual_seed(seed)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), action_space(campaign=True)),
               policy_kwargs={"net_arch": NET_ARCH}, device="cpu")


def trained_v1(seed: int = 1) -> PPO:
    """A v1 model after one real Adam step on BOTH stacks, so every parameter has moments to carry."""
    model = v1_model(seed)
    obs = torch.as_tensor(np.random.default_rng(seed).normal(size=(8, V1_INPUTS)).astype(np.float32))
    latent_pi, latent_vf = model.policy.mlp_extractor(obs)
    loss = model.policy.action_net(latent_pi).square().mean() + model.policy.value_net(latent_vf).square().mean()
    model.policy.optimizer.zero_grad()
    loss.backward()
    model.policy.optimizer.step()
    return model


def saved(tmp, model: PPO, name: str = "src.zip") -> Path:
    path = Path(tmp) / name
    model.save(path)
    return path


def states(rows: int = 128, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, V1_INPUTS)).astype(np.float32)


def migrated(tmp):
    return build_tech_model(saved(tmp, trained_v1()))


def test_first_layers_copy_479_columns_and_zero_the_51_new_ones():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        for name in FIRST_LAYERS:
            assert d[name].shape == (NET_ARCH[0], V2_INPUTS)
            assert torch.equal(d[name][:, :V1_INPUTS], s[name])
            assert not d[name][:, V1_INPUTS:].any()
            bias = name.replace(".weight", ".bias")
            assert torch.equal(d[bias], s[bias]), "no mean-fold: nothing is repurposed"


def test_everything_else_is_copied_bit_for_bit_including_the_critic():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        for name, value in d.items():
            if name in FIRST_LAYERS or name.startswith("action_net"):
                continue
            assert torch.equal(value, s[name]), name  # value_net too: never reset (§4.4 step 2)


def test_the_action_head_gets_zero_rows_and_the_prior_biases():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        assert d["action_net.weight"].shape == (V2_LOGITS, NET_ARCH[-1])
        assert torch.equal(d["action_net.weight"][:V1_LOGITS], s["action_net.weight"])
        assert not d["action_net.weight"][V1_LOGITS:].any()
        assert torch.equal(d["action_net.bias"][:V1_LOGITS], s["action_net.bias"])
        expected = torch.zeros(V2_LOGITS - V1_LOGITS)
        for row, bias in PRIOR_BIAS.items():
            expected[row - V1_LOGITS] = bias
        assert torch.allclose(d["action_net.bias"][V1_LOGITS:], expected)


def test_the_migration_is_exact_on_the_twelve_shared_dims():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        d = displacement(src, dst, states())
        assert d["mean_kl"] <= 1e-9 and d["max_kl"] <= 1e-9, d
        assert d["greedy_changed"] == 0.0 and d["mean_abs_dv"] <= 1e-5, d
        assert verdict(d) == [], verdict(d)


def test_the_new_heads_sit_on_their_priors_whatever_the_state():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        d = displacement(src, dst, states())
        for name in ("macro", "variant", "hook"):
            assert d[f"{name}_prior_err"] <= 1e-6, (name, d)
            assert d[f"{name}_state_spread"] == 0.0, (name, d)
        assert abs(d["entropy_added"] - PRIOR_ENTROPY) <= 1e-5, d


def test_the_verdict_names_a_broken_copy():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        with torch.no_grad():
            dst.policy.action_net.weight[0, 0] += 1.0
        problems = verdict(displacement(src, dst, states()))
        assert any("mean_kl" in p for p in problems), problems


def test_the_optimizer_is_carried_along_both_axes():
    """add_look_mode.carry_optimizer padded axis 0 only; the first layer grows along axis 1 here (§4.4 step 4)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, carried, why = migrated(tmp)
        assert carried, why
        names = [name for name, _ in dst.policy.named_parameters()]
        old = src.policy.optimizer.state_dict()["state"]
        new = dst.policy.optimizer.state_dict()["state"]
        assert set(old) == set(new) and len(old) == len(names)
        for index, name in enumerate(names):
            for key in ("exp_avg", "exp_avg_sq"):
                a, b = old[index][key], new[index][key]
                if name in FIRST_LAYERS:
                    assert b.shape == (NET_ARCH[0], V2_INPUTS)
                    assert torch.equal(b[:, :V1_INPUTS], a) and not b[:, V1_INPUTS:].any()
                elif name.startswith("action_net"):
                    assert b.shape[0] == V2_LOGITS
                    assert torch.equal(b[:V1_LOGITS], a) and not b[V1_LOGITS:].any()
                else:
                    assert torch.equal(b, a), name
            assert float(new[index]["step"]) == float(old[index]["step"]) == 1.0, "bias correction continues"


def test_add_tech_heads_writes_a_loadable_v2_model_and_keeps_the_step_axis():
    with tempfile.TemporaryDirectory() as tmp:
        src = trained_v1()
        src.num_timesteps = 66_524_386
        src._n_updates = 123_932
        dest = Path(tmp) / "out" / "tech_init.zip"
        _, _, measured, carried = add_tech_heads(saved(tmp, src), dest, observations=states())
        assert carried and measured["mean_kl"] <= KL_TOL
        loaded = PPO.load(dest, device="cpu")
        assert loaded.observation_space.shape == (V2_INPUTS,)
        assert [int(v) for v in loaded.action_space.nvec] == V2_NVEC
        assert loaded.num_timesteps == 66_524_386 and loaded._n_updates == 123_932
        assert loaded.policy_kwargs == {"net_arch": NET_ARCH}


def test_it_never_overwrites_anything():
    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, v1_model())
        dest = Path(tmp) / "exists.zip"
        dest.write_bytes(b"keep me")
        for target in (dest, source):
            try:
                add_tech_heads(source, target, observations=states(8))
            except FileExistsError:
                pass
            else:
                raise AssertionError(f"overwrote {target}")
        assert dest.read_bytes() == b"keep me"


def test_it_refuses_anything_but_a_v1_campaign_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        v2_path = Path(tmp) / "v2.zip"
        add_tech_heads(saved(tmp, v1_model()), v2_path, observations=states(8))
        torch.manual_seed(5)
        grind = PPO("MlpPolicy", SpacesEnv(ObsLayout().space(), action_space()),
                    policy_kwargs={"net_arch": NET_ARCH}, device="cpu")
        for bad in (v2_path, saved(tmp, grind, "grind.zip")):
            out = Path(tmp) / f"from_{bad.stem}.zip"
            try:
                add_tech_heads(bad, out, observations=states(8))
            except ValueError as exc:
                assert "v1 campaign layout" in str(exc)
            else:
                raise AssertionError(f"migrated {bad.name}")
            assert not out.exists()


def test_a_refused_migration_writes_nothing():
    original = ath.displacement
    ath.displacement = lambda *a, **k: {**original(*a, **k), "mean_kl": 0.5}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "never.zip"
            try:
                ath.add_tech_heads(saved(tmp, v1_model()), dest, observations=states(8))
            except MigrationRefused as exc:
                assert "mean_kl" in str(exc)
            else:
                raise AssertionError("a copy bug was written to disk")
            assert not dest.exists()
    finally:
        ath.displacement = original


def test_a_fresh_optimizer_is_refused_unless_allowed():
    original = ath.carry_optimizer
    ath.carry_optimizer = lambda src, dst: (False, "forced by the test")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source = saved(tmp, v1_model())
            try:
                ath.add_tech_heads(source, Path(tmp) / "a.zip", observations=states(8))
            except MigrationRefused as exc:
                assert "forced by the test" in str(exc)
            else:
                raise AssertionError("losing Adam's state at 60M steps must be an explicit decision")
            ath.add_tech_heads(source, Path(tmp) / "b.zip", observations=states(8), allow_fresh_optimizer=True)
            assert (Path(tmp) / "b.zip").exists()
    finally:
        ath.carry_optimizer = original


def test_real_observations_are_read_from_the_checkpoints_last_obs():
    with tempfile.TemporaryDirectory() as tmp:
        for steps, value in ((100, 0.25), (200, 0.5)):
            model = v1_model()
            model._last_obs = np.full((3, V1_INPUTS), value, dtype=np.float32)
            model.save(Path(tmp) / f"ckpt_{steps}_steps.zip")
        obs = checkpoint_observations(Path(tmp), 4)
        assert obs.shape == (4, V1_INPUTS)
        assert np.all(obs[:3] == 0.5) and np.all(obs[3] == 0.25), "newest checkpoint first"
        assert checkpoint_observations(Path(tmp) / "empty", 4) is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
