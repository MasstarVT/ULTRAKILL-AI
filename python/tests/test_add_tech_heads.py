"""scripts/add_tech_heads.py: the S7 weight migration, on tiny SB3 models. No game:  python tests/test_add_tech_heads.py

The contract is EXACTNESS, a strictly stronger one than add_look_mode's `mean_kl <= 0.03`: nothing is repurposed,
so on the twelve shared action dims the widened policy must compute what the source computed (spec §4.4). Anything
non-zero is a copy bug. The three new heads must sit exactly on their priors, independent of the state.
"""

from __future__ import annotations

import contextlib
import io
import re
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
    RANDOM_TECH,
    V1_INPUTS,
    V1_LOGITS,
    V2_INPUTS,
    V2_LOGITS,
    MigrationRefused,
    add_tech_heads,
    build_tech_model,
    check_destination,
    checkpoint_observations,
    displacement,
    random_tech_block,
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


def migrated(tmp, **kwargs):
    return build_tech_model(saved(tmp, trained_v1()), **kwargs)


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
    """add_look_mode.carry_optimizer padded axis 0 only; the first layer grows along axis 1 here (§4.4 step 4).

    Run with zero_new_moments: the plan's surgery, zero padding in BOTH moments, is what this test pins; the
    default fills the new exp_avg_sq entries (test_new_second_moments_take_the_old_means).
    """
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, carried, why = migrated(tmp, zero_new_moments=True)
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


# --- the 2026-09-23 code-quality review's fixes -----------------------------------------------------------------

def run_main(argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ath.main([str(arg) for arg in argv])
    return code, out.getvalue(), err.getvalue()


def refused_in_one_line(err: str) -> bool:
    lines = [line for line in err.splitlines() if line.startswith("MIGRATION REFUSED: ")]
    return len(lines) == 1 and "Traceback" not in err


@contextlib.contextmanager
def planted(mutate):
    """Every migration inside the block carries `mutate`'s damage to the widened state dict."""
    original = ath.widen_state_dict

    def damaged(src, dst):
        out = original(src, dst)
        with torch.no_grad():
            mutate(out)
        return out

    ath.widen_state_dict = damaged
    try:
        yield
    finally:
        ath.widen_state_dict = original


@contextlib.contextmanager
def forbidden(*names):
    """Any call to these module functions inside the block fails the test."""
    originals = {name: getattr(ath, name) for name in names}

    def boom(*a, **k):
        raise AssertionError("observations were read before the destination was checked")

    for name in names:
        setattr(ath, name, boom)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(ath, name, fn)


def test_a_nan_in_a_new_column_is_refused_with_exit_2_and_nothing_written():
    """Item 1: every check was `x > tol`, so a NaN passed; now a NaN (or inf) is refused, never compared."""
    def nan_in_a_new_value_column(sd):
        sd["mlp_extractor.value_net.0.weight"][0, V1_INPUTS] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        source, dest = saved(tmp, v1_model()), Path(tmp) / "nan.zip"
        with planted(nan_in_a_new_value_column):
            code, _, err = run_main([source, dest, "--synthetic", "8"])
        assert code == 2 and refused_in_one_line(err) and "mean_abs_dv" in err, err
        assert not dest.exists()


def test_the_verdict_refuses_any_non_finite_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        good = displacement(src, dst, states(8))
        assert verdict(good) == []
        for key in ("mean_kl", "greedy_changed", "mean_abs_dv", "mean_abs_v", "macro_prior_err",
                    "hook_state_spread", "entropy_added"):
            for bad in (float("nan"), float("inf")):
                problems = verdict({**good, key: bad})
                assert any(key in p and "non-finite" in p for p in problems), (key, bad, problems)


def test_a_destination_without_the_zip_suffix_is_refused_and_its_zip_twin_kept():
    """Item 2: SB3 writes `<dest>.zip` for a suffix-less dest, which the existence check never looked at."""
    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, v1_model())
        twin = Path(tmp) / "nosuffix.zip"
        twin.write_bytes(b"keep me")
        try:
            add_tech_heads(source, Path(tmp) / "nosuffix", observations=states(8))
        except ValueError as exc:
            assert "does not end in .zip" in str(exc)
        else:
            raise AssertionError("a suffix-less destination was accepted")
        code, _, err = run_main([source, Path(tmp) / "nosuffix", "--synthetic", "8"])
        assert code == 2 and refused_in_one_line(err), err
        assert twin.read_bytes() == b"keep me"
        assert check_destination(Path(tmp) / "free.zip") == Path(tmp) / "free.zip"
        for bad in (twin, Path(tmp) / "free.ZIP", Path(tmp) / "free.zip.bak"):
            try:
                check_destination(bad)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError(f"accepted {bad}")


def test_main_checks_the_destination_before_reading_any_observation():
    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, v1_model())
        dest = Path(tmp) / "exists.zip"
        dest.write_bytes(b"keep me")
        with forbidden("checkpoint_observations", "synthetic_observations", "build_tech_model"):
            for extra in ([], ["--synthetic", "8"]):
                code, _, err = run_main([source, dest, *extra])
                assert code == 2 and refused_in_one_line(err) and "exists" in err, err
        assert dest.read_bytes() == b"keep me"


def test_the_correct_model_reads_exactly_zero_with_a_random_tech_block_too():
    """Item 3: the new DLL sends a non-zero TECH block from the first step, so it is measured both ways."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        obs = states()
        d = displacement(src, dst, obs, tech=random_tech_block(len(obs), 0))
        assert d["mean_kl"] == 0.0 and d["mean_abs_dv"] == 0.0 and d["greedy_changed"] == 0.0, d
        assert d["bitwise"] == 1.0 and verdict(d) == [], d
        _, _, measured, _ = add_tech_heads(saved(tmp, v1_model(), "s.zip"), Path(tmp) / "ok.zip", observations=obs)
        for prefix in ("", RANDOM_TECH):
            assert measured[prefix + "mean_kl"] == 0.0 and measured[prefix + "bitwise"] == 1.0, measured


def test_a_bad_new_column_is_seen_only_with_a_nonzero_tech_block_and_refused():
    def half_in_the_new_columns(sd):
        sd["mlp_extractor.policy_net.0.weight"][:, V1_INPUTS:] = 0.5

    with tempfile.TemporaryDirectory() as tmp:
        obs = states(32)
        with planted(half_in_the_new_columns):
            src, dst, _, _ = migrated(tmp)
            assert verdict(displacement(src, dst, obs)) == [], "zero inputs cannot see a new column"
            noisy = verdict(displacement(src, dst, obs, tech=random_tech_block(len(obs), 0)))
            assert any("mean_kl" in p for p in noisy), noisy
            dest = Path(tmp) / "bad.zip"
            try:
                ath.add_tech_heads(saved(tmp, v1_model(), "s.zip"), dest, observations=obs)
            except MigrationRefused as exc:
                assert "with a random TECH block, mean_kl" in str(exc), exc
            else:
                raise AssertionError("a bad new column was written to disk")
            assert not dest.exists()


def test_main_turns_every_refusal_into_one_line_and_exit_2():
    """Item 4: an existing destination, a non-v1 source and --max-states 0 used to exit 1 with a traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, v1_model())
        existing = Path(tmp) / "exists.zip"
        existing.write_bytes(b"keep me")
        v2_path = Path(tmp) / "v2.zip"
        add_tech_heads(source, v2_path, observations=states(8))
        (Path(tmp) / "obs").mkdir()
        v1_model().save(Path(tmp) / "obs" / "ckpt_100_steps.zip")
        cases = {
            "exists": [source, existing, "--synthetic", "8"],
            "not the v1 campaign layout": [v2_path, Path(tmp) / "from_v2.zip", "--synthetic", "8"],
            "--max-states 0": [source, Path(tmp) / "zero.zip", "--obs-from", Path(tmp) / "obs", "--max-states", "0"],
            "no ckpt_*_steps.zip": [source, Path(tmp) / "none.zip", "--obs-from", Path(tmp) / "nothing"],
        }
        for needle, argv in cases.items():
            code, _, err = run_main(argv)
            assert code == 2 and refused_in_one_line(err) and needle in err, (needle, err)
            assert not Path(argv[1]).exists() or Path(argv[1]) == existing, needle
        assert existing.read_bytes() == b"keep me"


def test_the_report_prints_repr_and_a_computed_bitwise_line():
    """Item 5: repr() for the exactness numbers, and `bitwise: yes/no` from torch.equal, not an unconditional EXACT."""
    def a_hair_on_an_old_logit_row(sd):
        sd["action_net.weight"][0, 0] += 1e-5

    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, trained_v1())
        code, out, err = run_main([source, Path(tmp) / "a.zip", "--synthetic", "16", "--ent-floor", "6.5"])
        assert code == 0, err
        assert re.search(r"^\s+mean_kl\s+0\.0$", out, re.M) and re.search(r"^\s+mean_abs_dv\s+0\.0$", out, re.M), out
        assert "=> bitwise: yes" in out and "EXACT" not in out, out
        assert "seeded uniform" in out and "= 0.0 (an old DLL" in out, out
        assert "ent_floor 6.5 -> 7.8986" in out, out
        with planted(a_hair_on_an_old_logit_row):
            code, out, err = run_main([source, Path(tmp) / "b.zip", "--synthetic", "16"])
        assert code == 0, err  # inside KL_TOL: written, but it must not claim bit-for-bit
        assert "=> bitwise: no" in out, out


def test_the_value_tolerance_is_relative_to_the_value_scale():
    """Item 6: DV_TOL scales with mean |V|: 1e-6 x max(1, mean |V|)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        good = displacement(src, dst, states(8))
        assert verdict({**good, "mean_abs_v": 1000.0, "mean_abs_dv": 5e-4}) == []
        assert any("mean_abs_dv" in p for p in verdict({**good, "mean_abs_v": 1000.0, "mean_abs_dv": 2e-3}))
        assert any("mean_abs_dv" in p for p in verdict({**good, "mean_abs_v": 0.5, "mean_abs_dv": 5e-6}))
        assert verdict({**good, "mean_abs_v": 0.5, "mean_abs_dv": 5e-7}) == []


def test_the_sources_hyperparameters_and_adam_settings_are_carried():
    """Item 7: the new file carries the run's values, not SB3's defaults."""
    torch.manual_seed(2)
    src = PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), action_space(campaign=True)),
              policy_kwargs={"net_arch": NET_ARCH}, device="cpu", learning_rate=2e-4, gamma=0.995,
              ent_coef=0.004, target_kl=0.03, n_steps=256, batch_size=128, clip_range=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = saved(tmp, src), Path(tmp) / "hp.zip"
        _, dst, _, _ = add_tech_heads(source, dest, observations=states(8))
        assert dst.policy.optimizer.param_groups[0]["lr"] == 2e-4, "SB3's default 3e-4 was left in the group"
        loaded = PPO.load(dest, device="cpu")
        for name in ("learning_rate", "gamma", "ent_coef", "target_kl", "n_steps", "batch_size"):
            assert getattr(loaded, name) == getattr(src, name), (name, getattr(loaded, name))
        assert loaded.clip_range(1.0) == 0.1
        assert loaded.policy.optimizer.param_groups[0]["lr"] == 2e-4


def test_new_second_moments_take_the_old_means_and_the_flag_restores_zeros():
    """Item 8: each new exp_avg_sq entry = the mean of the old entries along the axis that grew; exp_avg stays 0."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, carried, why = migrated(tmp)
        assert carried, why
        names = [name for name, _ in dst.policy.named_parameters()]
        old = src.policy.optimizer.state_dict()["state"]
        new = dst.policy.optimizer.state_dict()["state"]
        for index, name in enumerate(names):
            a, b = old[index]["exp_avg_sq"], new[index]["exp_avg_sq"]
            m = new[index]["exp_avg"]
            if name in FIRST_LAYERS:  # a new input column: its row's old mean
                assert torch.equal(b[:, :V1_INPUTS], a)
                fill = a.mean(dim=1, keepdim=True).expand(-1, V2_INPUTS - V1_INPUTS)
                assert torch.equal(b[:, V1_INPUTS:], fill) and fill.any()
                assert not m[:, V1_INPUTS:].any()
            elif name == "action_net.weight":  # a new logit row: its column's old mean
                assert torch.equal(b[:V1_LOGITS], a)
                fill = a.mean(dim=0, keepdim=True).expand(V2_LOGITS - V1_LOGITS, -1)
                assert torch.equal(b[V1_LOGITS:], fill) and fill.any()
                assert not m[V1_LOGITS:].any()
            elif name == "action_net.bias":  # 1-D: the old tensor's mean
                assert torch.equal(b[:V1_LOGITS], a)
                assert torch.equal(b[V1_LOGITS:], a.mean().expand(V2_LOGITS - V1_LOGITS)) and a.mean() > 0
                assert not m[V1_LOGITS:].any()
            else:
                assert torch.equal(b, a), name
        # the flag, through the CLI and a save/load round trip: the plan's zeros
        code, out, err = run_main([saved(tmp, trained_v1(), "s.zip"), Path(tmp) / "z.zip", "--synthetic", "8",
                                   "--zero-new-moments"])
        assert code == 0 and "second moments zero" in out, err
        state = PPO.load(Path(tmp) / "z.zip", device="cpu").policy.optimizer.state_dict()["state"]
        for index, name in enumerate(names):
            if name in FIRST_LAYERS:
                assert not state[index]["exp_avg_sq"][:, V1_INPUTS:].any()
            elif name.startswith("action_net"):
                assert not state[index]["exp_avg_sq"][V1_LOGITS:].any()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
