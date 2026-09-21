"""A RESUMED PPO model trains at the CONFIG's gamma, not the zip's -- in the model and in the rollout buffer.

No game needed:  python tests/test_resume_hyperparams.py   (or pytest)

WHY THIS FILE EXISTS. A gamma change on a run that NEVER STARTS FRESH: the specialist driver re-execs
`train.py --resume <newest ckpt>` every round, so every step of such a stage is a resume. A saved SB3 zip
carries the hyperparameters it was trained with, and `RolloutBuffer` keeps its OWN copies of `gamma` and
`gae_lambda`, taken when the buffer is constructed and used by `compute_returns_and_advantage`. If either of
those won, the stage would train at one gamma while the config, the generated YAML and the project log all
said another -- an unattributable and very expensive wrong conclusion.

THIS CUTS BOTH WAYS, AND THAT IS WHY THE FILE OUTLIVED THE STAGE IT WAS WRITTEN FOR. Stage S1 (0.998 ->
0.999 with `gae_lambda` 0.98) went live on 2026-09-21 and was REVERTED the same day: it made the policy
slower at every quantile over 2.66M steps (docs/project-log.md, 2026-09-21), S2 is cancelled, and no
`speed.train:` block ships any more. Reverting is the same mechanism run backwards -- a resume that must
train at the CONFIG's 0.998 even when the zip it loads was written at 0.999 -- so this test guards the way
back as well as the way out. The values below are S1's, kept because they are a real pair that differs in
both keys; nothing here reads the shipped plan.

WHAT WAS MEASURED, 2026-09-20, against the installed stable_baselines3 2.9.0: the config ALREADY wins.
`BaseAlgorithm.load` does `model.__dict__.update(data)` and then `model.__dict__.update(kwargs)` BEFORE
calling `_setup_model()`, and `OnPolicyAlgorithm._setup_model` builds the buffer from `self.gamma` /
`self.gae_lambda`. So `training.apply_resume_hyperparams` normally moves nothing. It is kept, and this file
is written, because that is an undocumented ordering inside a third-party library: one line of it moving at
the next upgrade would break a gamma stage silently, and here it breaks a test instead.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.training import (  # noqa: E402
    BUFFER_HYPERPARAMS,
    DEFAULT_HYPERPARAMS,
    apply_resume_hyperparams,
    hyperparams_in_force,
)

SAVED = {"gamma": 0.998, "gae_lambda": 0.95}
RESUMED = {"gamma": 0.999, "gae_lambda": 0.98}  # what S1 was, kept as a pair that differs in both keys


class TinyEnv(gym.Env):
    """Four floats in, two discrete actions out: the smallest thing PPO will build a policy for."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def _obs(self):
        return np.zeros(4, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


def tiny(**hyper) -> PPO:
    return PPO("MlpPolicy", DummyVecEnv([TinyEnv]), n_steps=16, batch_size=8, n_epochs=1,
               policy_kwargs={"net_arch": [8]}, device="cpu", verbose=0, **hyper)


def saved_model(tmp: Path) -> Path:
    path = tmp / "saved.zip"
    tiny(**SAVED).save(path)
    return path


def test_the_defaults_were_lifted_out_of_main_unchanged():
    """`main()` now does `{**DEFAULT_HYPERPARAMS, **config}`; these are the values it always merged over."""
    assert DEFAULT_HYPERPARAMS == {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 512, "n_epochs": 5,
                                   "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01}
    assert BUFFER_HYPERPARAMS == ("gamma", "gae_lambda"), "the two a RolloutBuffer keeps its own copies of"


def test_a_resume_trains_at_the_configs_gamma_in_the_model_and_in_the_buffer():
    with tempfile.TemporaryDirectory() as tmp:
        path = saved_model(Path(tmp))
        stale = PPO.load(path, device="cpu")
        assert (stale.gamma, stale.gae_lambda) == (0.998, 0.95), "the zip really does carry the old values"
        assert (stale.rollout_buffer.gamma, stale.rollout_buffer.gae_lambda) == (0.998, 0.95)

        hyper = {"n_steps": 16, "batch_size": 8, "n_epochs": 1, **RESUMED}
        model = PPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu", **hyper)
        # THE MEASUREMENT, recorded rather than assumed: on stable_baselines3 2.9.0 `load` alone already
        # applies the kwargs to the model AND (through `_setup_model`) to the buffer, so the helper below has
        # nothing to move. This assertion is what will fail first if a future SB3 changes that ordering.
        assert apply_resume_hyperparams(model, hyper) == {}, \
            "SB3 no longer applies load() kwargs before building the rollout buffer -- the helper is now load-bearing"
        assert (model.gamma, model.gae_lambda) == (0.999, 0.98)
        # The buffer is the one that matters: GAE is computed from ITS copies, not from the model's.
        assert (model.rollout_buffer.gamma, model.rollout_buffer.gae_lambda) == (0.999, 0.98)


def test_the_helper_repairs_a_model_whose_buffer_kept_the_old_values():
    """The failure this guards against, staged by hand: a load that left the buffer on the saved gamma."""
    with tempfile.TemporaryDirectory() as tmp:
        path = saved_model(Path(tmp))
        model = PPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu", n_steps=16, batch_size=8, n_epochs=1)
        assert (model.gamma, model.rollout_buffer.gamma) == (0.998, 0.998)

        moved = apply_resume_hyperparams(model, {"n_steps": 16, **RESUMED})
        assert moved == {"gamma": (0.998, 0.999), "gae_lambda": (0.95, 0.98)}
        assert (model.gamma, model.gae_lambda) == (0.999, 0.98)
        assert (model.rollout_buffer.gamma, model.rollout_buffer.gae_lambda) == (0.999, 0.98)
        assert model.n_steps == 16, "only the two buffer hyperparameters are ever written"


def test_the_helper_is_a_no_op_when_the_values_already_agree():
    with tempfile.TemporaryDirectory() as tmp:
        path = saved_model(Path(tmp))
        model = PPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu", n_steps=16, batch_size=8, n_epochs=1)
        assert apply_resume_hyperparams(model, {"n_steps": 16, **SAVED}) == {}
        assert (model.gamma, model.rollout_buffer.gamma) == (0.998, 0.998)
        # ... and a config that names neither leaves everything alone.
        assert apply_resume_hyperparams(model, {"learning_rate": 1e-4}) == {}
        assert (model.gamma, model.gae_lambda) == (0.998, 0.95)


def test_the_log_line_reports_what_is_in_force_including_the_buffers_own_copies():
    """A gamma stage is judged on a number that has to be verifiable in the train log of its own round."""
    with tempfile.TemporaryDirectory() as tmp:
        path = saved_model(Path(tmp))
        hyper = {"n_steps": 16, "batch_size": 8, "n_epochs": 1, **RESUMED}
        model = PPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu", **hyper)
        apply_resume_hyperparams(model, hyper)
        line = hyperparams_in_force(model, hyper)
        assert "gamma=0.999" in line and "gae_lambda=0.98" in line
        assert "rollout buffer: gamma=0.999, gae_lambda=0.98" in line


def test_the_gamma_ladders_two_steps_both_survive_a_resume():
    """0.999 then 0.9995, the lead's two-step ruling, each resumed from the round before it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = saved_model(tmp)
        for gamma in (0.999, 0.9995):
            hyper = {"n_steps": 16, "batch_size": 8, "n_epochs": 1, "gamma": gamma, "gae_lambda": 0.98}
            model = PPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu", **hyper)
            apply_resume_hyperparams(model, hyper)
            assert model.rollout_buffer.gamma == gamma and model.rollout_buffer.gae_lambda == 0.98
            path = tmp / f"round_{gamma}.zip"
            model.save(path)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
