"""Widens a v1 campaign checkpoint to the v2 TECH layout -- the weight half of stage S7, "the one break", of
docs/superpowers/specs/2026-09-20-speedrun-tech.md (§4.4). Plan: docs/superpowers/plans/2026-09-23-tech-break-python.md.

    python scripts/add_tech_heads.py models/spec_0-1_speed/ckpt_66500000_steps.zip models/spec_0-1_speed/tech_init.zip --ent-floor 6.5
    python scripts/add_tech_heads.py <src> <dest> --synthetic 512          # no checkpoints to read states from

v1 is 479 inputs and 12 action dims / 45 logits; v2 is 530 inputs and 15 dims / 57 logits (ultrakill_ai/spaces.py).
Unlike scripts/add_look_mode.py, which REPURPOSED eight inputs that carried real values (mean KL 0.0262, 29.6% of
greedy actions changed), nothing is repurposed here, so the migration is EXACT on the twelve shared dims:

  1. first layers (512, 479) -> (512, 530): columns 0-478 copied bit for bit, 479-529 ZERO, biases unchanged,
     NO mean-fold (the new inputs read 0.0 against an old DLL and carry no mean to fold);
  2. the second hidden layers and the value head copied verbatim, never reset: the reward scale does not move
     at the break, and discarding a 60M-step critic is strictly destructive;
  3. action head (45, 512) -> (57, 512): rows 0-44 copied, rows 45-56 ZERO weights, biases ln 45 / ln 17 / ln 9
     on the macro "none", variant "keep" and hook "off" rows, so P(none) = 0.90 (0.02 per macro),
     P(keep) = 0.85, P(no hook) = 0.90 -- state-independent at init;
  4. Adam's moments carried, zero-padded along EVERY axis (add_look_mode.carry_optimizer pads axis 0 only, and its
     broad except would silently fall back to a fresh optimizer here), with `step` kept so bias correction
     continues; a carry that fails is REFUSED unless --allow-fresh-optimizer.
     A DELIBERATE DEVIATION from the plan, which spec §4.4 did not consider: each NEW `exp_avg_sq` entry is then
     filled with the mean of the old entries along the axis that grew -- a new input column takes its row's old
     mean, a new logit row its column's old mean, a new bias entry the old bias's mean -- while `exp_avg` stays
     zero. A zero second moment under a carried `step` of ~270k (bias correction ~1) would run the first updates
     of the 51 new columns and of the live new rows at ~3-6.5x lr. No weight is touched, so the forward pass and
     the measured KL do not move. --zero-new-moments leaves them at 0.0, the plan's original surgery;
  5. `num_timesteps` and `_n_updates` carried, so the driver's step axis stays continuous, and the source's PPO
     hyperparameters (learning_rate, gamma, ent_coef, target_kl, n_steps, ...) and Adam's param_group settings
     (lr, betas, eps) too, so a tool that loads the file without kwargs sees the run's values rather than SB3's
     defaults (train.py --resume forces the config's on every resume anyway).

Before writing anything it MEASURES, twice over the same states: the source fed the v1 vector against the result
fed that vector plus a TECH block (479-529) of 0.0 -- what an old DLL, or a flag that is off, sends -- and of
seeded uniform [0, 1) values -- what the new DLL sends from the first step, and the only way a bad NEW column
shows. Each reports mean KL over the 12 shared dims, greedy changes, mean |dV|, and whether the shared logits and
the value are bitwise equal (torch.equal). The states are REAL observations -- the `_last_obs` every SB3 checkpoint
carries, twelve per ckpt_*_steps.zip, read ~120 KB at a time -- or a synthetic batch. Tolerances: mean KL <= 1e-6,
no greedy change, mean |dV| <= 1e-6 x max(1, mean |V|) (RELATIVE: float reordering scales with the value), the new
heads on their priors and state-independent, and every measurement finite (a NaN or inf is refused, never
compared). Anything outside is a copy bug: the script prints one "MIGRATION REFUSED: ..." line, exits 2 and
writes nothing -- as it does for an existing destination, a source that is not v1, or a bad argument. It also
prints the entropy the new heads add (1.3986 nats at these priors): the new `ent_floor` is the old one plus that.

Never overwrites: `dest` must end in .zip (SB3 appends .zip to a path without a suffix, which would dodge the
check) and must not exist, and that is checked before anything is read. The source is only read. After the
break, the 479-wide files go to `pre_tech/` with scripts/quarantine_pre_tech.py (a MOVE, never a delete).
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import ACTION_NVEC_CAMPAIGN, ObsLayout, action_space  # noqa: E402

from transfer_weights import SpacesEnv  # noqa: E402

V1_INPUTS, V2_INPUTS = 479, 530
V1_LOGITS, V2_LOGITS = 45, 57
TECH_WIDTH = V2_INPUTS - V1_INPUTS  # the 51-float TECH block
FIRST_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
# §4.4 step 3: with zero weights the bias alone sets each new head's marginal, identically in every state.
PRIOR_BIAS = {45: math.log(45.0), 51: math.log(17.0), 55: math.log(9.0)}  # macro none / variant keep / hook off
PRIORS = {  # new head -> (action dim, the marginal those biases give)
    "macro": (12, (0.90, 0.02, 0.02, 0.02, 0.02, 0.02)),
    "variant": (13, (0.85, 0.05, 0.05, 0.05)),
    "hook": (14, (0.90, 0.10)),
}
PRIOR_ENTROPY = 1.3986106691325138  # nats the three heads add: 0.4860 + 0.5875 + 0.3251
KL_TOL = 1e-6  # mean KL over the 12 shared dims: above this is a copy bug, not a tolerable displacement
DV_TOL = 1e-6  # RELATIVE: mean |dV| may reach DV_TOL * max(1, mean |V|) -- float reordering, nothing more
PRIOR_TOL = 1e-3
SPREAD_TOL = 1e-6
DEFAULT_STATES = 2400  # 200 checkpoints x 12 games
RANDOM_TECH = "random_tech."  # key prefix of the second measurement (a seeded random TECH block)
MODE_LABELS = {
    "": "TECH block 479-529 = 0.0 (an old DLL, or a flag that is off)",
    RANDOM_TECH: "TECH block 479-529 = seeded uniform [0, 1) (the new DLL, from the first step)",
}
REPR_KEYS = ("mean_kl", "max_kl", "mean_abs_dv")  # printed with repr(): 0.0 means 0.0, not "rounds to 0"
# PPO attributes the saved file carries as the source's rather than SB3's defaults (§4.4 step 5).
HYPERPARAMETERS = ("learning_rate", "lr_schedule", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                   "clip_range", "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm",
                   "target_kl", "use_sde", "sde_sample_freq", "rollout_buffer_class", "rollout_buffer_kwargs")
_CKPT = re.compile(r"ckpt_(\d+)_steps\.zip")


class MigrationRefused(RuntimeError):
    """The measurement found a difference it must not. Nothing was written."""


def check_destination(dest) -> Path:
    """`dest`, or a refusal. SB3 writes `<dest>.zip` when `dest` has no suffix, so a path is accepted only when it
    already ends in `.zip` -- the file checked is then the file written -- and neither form of it may exist."""
    dest = Path(dest)
    if dest.suffix != ".zip":
        raise ValueError(f"{dest} does not end in .zip: SB3 would write a different file from the one checked here")
    for path in (dest, dest.with_suffix(".zip")):
        if path.exists():
            raise FileExistsError(f"{path} exists: this script never overwrites (write beside it, then copy)")
    return dest


def check_observations(observations) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] < 1 or obs.shape[1] != V1_INPUTS:
        raise ValueError(f"the measurement needs at least one {V1_INPUTS}-float state, got an array of {obs.shape}")
    return obs


def widen_state_dict(src: dict, dst: dict) -> dict:
    """`dst`'s state dict filled from `src` per steps 1-3 of the module docstring. Neither input is modified."""
    out = {}
    for name, value in dst.items():
        source = src[name]
        if name in FIRST_LAYERS:
            widened = torch.zeros_like(value)
            widened[:, :V1_INPUTS] = source
            out[name] = widened
        elif name in ("action_net.weight", "action_net.bias"):
            widened = torch.zeros_like(value)
            widened[:V1_LOGITS] = source
            if name == "action_net.bias":
                for row, bias in PRIOR_BIAS.items():
                    widened[row] = bias
            out[name] = widened
        else:
            if source.shape != value.shape:
                raise ValueError(f"{name}: {tuple(source.shape)} cannot be copied into {tuple(value.shape)}")
            out[name] = source.clone()
    return out


def carry_optimizer(src_model: PPO, dst_model: PPO) -> tuple[bool, str]:
    """Carries Adam's moments into the widened policy, zero-padding along EVERY axis. (carried, why)."""
    try:
        names = [name for name, _ in src_model.policy.named_parameters()]
        if names != [name for name, _ in dst_model.policy.named_parameters()]:
            return False, "the two policies do not have the same parameters"
        params = dict(dst_model.policy.named_parameters())
        state = {"param_groups": copy.deepcopy(dst_model.policy.optimizer.state_dict()["param_groups"]), "state": {}}
        for index, entry in src_model.policy.optimizer.state_dict()["state"].items():
            name = names[int(index)]
            shape = params[name].shape
            moments = {}
            for key, value in entry.items():
                if not torch.is_tensor(value) or value.dim() == 0 or value.shape == shape:
                    # Adam's `step` is a 0-dim tensor in torch 2.x: carried as is, so bias correction continues.
                    moments[key] = value.clone() if torch.is_tensor(value) else value
                elif value.dim() == len(shape) and all(a <= b for a, b in zip(value.shape, shape)):
                    padded = torch.zeros(shape, dtype=value.dtype)
                    padded[tuple(slice(0, s) for s in value.shape)] = value
                    moments[key] = padded
                else:
                    return False, f"{name}.{key}: {tuple(value.shape)} does not fit {tuple(shape)}"
            state["state"][int(index)] = moments
        dst_model.policy.optimizer.load_state_dict(state)
        return True, "carried"
    except (KeyError, IndexError, ValueError, RuntimeError, TypeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def fill_new_second_moments(src_model: PPO, dst_model: PPO) -> None:
    """After `carry_optimizer`: every NEW `exp_avg_sq` entry of the widened policy gets the mean of the source's old
    entries along the axis that grew (docstring step 4); old entries and `exp_avg` are left alone. In place."""
    old_state = src_model.policy.optimizer.state_dict()["state"]
    for index, param in enumerate(dst_model.policy.parameters()):
        entry = dst_model.policy.optimizer.state.get(param)
        old = old_state.get(index, {}).get("exp_avg_sq")
        if not entry or old is None or entry["exp_avg_sq"].shape == old.shape:
            continue
        padded = entry["exp_avg_sq"]
        new = torch.ones(padded.shape, dtype=torch.bool)
        new[tuple(slice(0, s) for s in old.shape)] = False
        padded[new] = old.mean()  # 1-D (a new bias entry), and the corner if a 2-D moment ever grew both ways
        if old.dim() == 2:
            rows, cols = old.shape
            padded[:rows, cols:] = old.mean(dim=1, keepdim=True)  # a new input column: its row's old mean
            padded[rows:, :cols] = old.mean(dim=0, keepdim=True)  # a new logit row: its column's old mean


def copy_hyperparameters(src_model: PPO, dst_model: PPO) -> None:
    """The source's PPO hyperparameters and Adam's param_group settings onto the widened model (step 5)."""
    for name in HYPERPARAMETERS:
        if hasattr(src_model, name):
            setattr(dst_model, name, copy.deepcopy(getattr(src_model, name)))
    for old, new in zip(src_model.policy.optimizer.param_groups, dst_model.policy.optimizer.param_groups):
        new.update({key: copy.deepcopy(value) for key, value in old.items() if key != "params"})


def build_tech_model(source, seed: int = 0, *, zero_new_moments: bool = False) -> tuple[PPO, PPO, bool, str]:
    """(the v1 source, the widened v2 model in memory, optimizer carried, why). Writes nothing."""
    src = PPO.load(Path(source), device="cpu")
    shape = tuple(src.observation_space.shape)
    nvec = tuple(int(v) for v in getattr(src.action_space, "nvec", ()))
    if shape != (V1_INPUTS,) or nvec != tuple(int(v) for v in ACTION_NVEC_CAMPAIGN):
        raise ValueError(f"{source} has {shape} inputs and {len(nvec)} action dims, not the v1 campaign layout "
                         f"({V1_INPUTS} / 12): there is nothing to migrate")
    torch.manual_seed(seed)  # nothing in the result depends on it; kept so a rerun builds the same object
    dst = PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True, tech=True).space(),
                                     action_space(campaign=True, tech=True)),
              policy_kwargs=src.policy_kwargs, device="cpu")
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict()))
    carried, why = carry_optimizer(src, dst)
    if carried and not zero_new_moments:
        fill_new_second_moments(src, dst)
    copy_hyperparameters(src, dst)
    dst.num_timesteps = src.num_timesteps
    dst._n_updates = getattr(src, "_n_updates", 0)
    return src, dst, carried, why


def synthetic_observations(rows: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, V1_INPUTS)).astype(np.float32)


def random_tech_block(rows: int, seed: int = 0) -> np.ndarray:
    """A seeded stand-in for the TECH block the new DLL sends: every field packs into [0, 1] (or [0, 2])."""
    return np.random.default_rng([seed, V2_INPUTS]).uniform(0.0, 1.0, size=(rows, TECH_WIDTH)).astype(np.float32)


def _steps(path: Path) -> int:
    match = _CKPT.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def checkpoint_observations(model_dir: Path, max_states: int) -> np.ndarray | None:
    """Real v1 observations: `_last_obs` out of a run's checkpoints, newest first, twelve per file.

    SB3 saves the vector env's last observation into every zip's `data` member (base64 cloudpickle), so each
    `ckpt_*_steps.zip` of a live run carries twelve genuine 479-float states. Streamed one small `data` member at
    a time, never the weights; a half-written or unreadable zip (a trainer mid-save) is skipped.
    """
    rows: list[np.ndarray] = []
    total = 0
    for path in sorted(Path(model_dir).glob("ckpt_*_steps.zip"), key=_steps, reverse=True):
        try:
            with zipfile.ZipFile(path) as z:
                data = json.loads(z.read("data").decode("utf-8"))
            blob = (data.get("_last_obs") or {}).get(":serialized:")
            if not blob:
                continue
            obs = np.asarray(pickle.loads(base64.b64decode(blob)), dtype=np.float32)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError):
            continue
        if obs.ndim != 2 or obs.shape[1] != V1_INPUTS:
            continue
        rows.append(obs)
        total += obs.shape[0]
        if total >= max_states:
            break
    return np.concatenate(rows)[:max_states] if rows else None


def _raw_heads(model: PPO, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(action logits, values), computed exactly as SB3's get_distribution / predict_values compute them."""
    policy = model.policy
    pi = BaseModel.extract_features(policy, obs, policy.pi_features_extractor)
    vf = BaseModel.extract_features(policy, obs, policy.vf_features_extractor)
    logits = policy.action_net(policy.mlp_extractor.forward_actor(pi))
    return logits, policy.value_net(policy.mlp_extractor.forward_critic(vf)).flatten()


def displacement(src: PPO, dst: PPO, observations: np.ndarray, tech: np.ndarray | None = None) -> dict[str, float]:
    """How far the widened policy moved, and where its new heads sit. `observations` are v1 vectors; the result is
    fed them plus `tech` as its TECH block (0.0 when None)."""
    observations = np.asarray(observations, dtype=np.float32)
    block = np.zeros((observations.shape[0], TECH_WIDTH), dtype=np.float32) if tech is None else tech
    old_obs = torch.as_tensor(observations, dtype=torch.float32)
    new_obs = torch.as_tensor(np.concatenate([observations, block], axis=1), dtype=torch.float32)
    with torch.no_grad():
        old = src.policy.get_distribution(old_obs).distribution
        new = dst.policy.get_distribution(new_obs).distribution
        old_v = src.policy.predict_values(old_obs).flatten()
        new_v = dst.policy.predict_values(new_obs).flatten()
        old_logits, old_raw_v = _raw_heads(src, old_obs)
        new_logits, new_raw_v = _raw_heads(dst, new_obs)
        bitwise = torch.equal(old_logits, new_logits[:, :V1_LOGITS]) and torch.equal(old_raw_v, new_raw_v)
        kl = torch.zeros(old_obs.shape[0], dtype=torch.float64)
        changed = torch.zeros(old_obs.shape[0], dtype=torch.bool)
        for k in range(len(old)):  # the 12 shared dims
            p, q = old[k].probs.double(), new[k].probs.double()
            kl += (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(dim=-1)
            changed |= old[k].probs.argmax(dim=-1) != new[k].probs.argmax(dim=-1)
        ent_old = sum(d.entropy().double() for d in old)
        ent_new = sum(d.entropy().double() for d in new)
        out = {
            "states": float(old_obs.shape[0]),
            "mean_kl": float(kl.mean()),
            "max_kl": float(kl.max()),
            "greedy_changed": float(changed.float().mean()),
            "mean_abs_dv": float((old_v - new_v).abs().mean()),
            "mean_abs_v": float(old_v.abs().mean()),
            "bitwise": 1.0 if bitwise else 0.0,
            "entropy_old": float(ent_old.mean()),
            "entropy_new": float(ent_new.mean()),
            "entropy_added": float((ent_new - ent_old).mean()),
        }
        for name, (dim, prior) in PRIORS.items():
            probs = new[dim].probs.double()
            out[f"{name}_prior_err"] = float((probs - torch.tensor(prior, dtype=torch.float64)).abs().max())
            out[f"{name}_state_spread"] = float((probs.max(dim=0).values - probs.min(dim=0).values).max())
    return out


def verdict(measured: dict[str, float]) -> list[str]:
    """Every tolerance the measurement breaks, as text. Empty = exact. Each check is `not (x <= tol)`, so a NaN
    fails it, and any non-finite value is named on its own as well."""
    problems = [f"{name} is {value}: a non-finite measurement is refused"
                for name, value in measured.items() if not math.isfinite(value)]
    if not measured["mean_kl"] <= KL_TOL:
        problems.append(f"mean_kl {measured['mean_kl']!r} > {KL_TOL:g}")
    if not measured["greedy_changed"] <= 0.0:
        problems.append(f"greedy_changed {measured['greedy_changed']:.3g} > 0")
    dv_tol = DV_TOL * max(1.0, measured["mean_abs_v"])
    if not measured["mean_abs_dv"] <= dv_tol:
        problems.append(f"mean_abs_dv {measured['mean_abs_dv']!r} > {dv_tol:.3g} "
                        f"({DV_TOL:g} x max(1, mean |V| {measured['mean_abs_v']:.4g}))")
    for name in PRIORS:
        if not measured[f"{name}_prior_err"] <= PRIOR_TOL:
            problems.append(f"{name} head off its prior by {measured[f'{name}_prior_err']:.3g}")
        if not measured[f"{name}_state_spread"] <= SPREAD_TOL:
            problems.append(f"{name} head depends on the state ({measured[f'{name}_state_spread']:.3g})")
    if not abs(measured["entropy_added"] - PRIOR_ENTROPY) <= PRIOR_TOL:
        problems.append(f"entropy added {measured['entropy_added']:.4f}, the priors say {PRIOR_ENTROPY:.4f}")
    return problems


def split_modes(measured: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    """[(label, values)] for the zero-TECH and the random-TECH measurement in a flat `add_tech_heads` result."""
    noisy = {k[len(RANDOM_TECH):]: v for k, v in measured.items() if k.startswith(RANDOM_TECH)}
    zero = {k: v for k, v in measured.items() if not k.startswith(RANDOM_TECH)}
    return [(MODE_LABELS[""], zero)] + ([(MODE_LABELS[RANDOM_TECH], noisy)] if noisy else [])


def add_tech_heads(source, dest, *, observations: np.ndarray | None = None, seed: int = 0,
                   allow_fresh_optimizer: bool = False,
                   zero_new_moments: bool = False) -> tuple[PPO, PPO, dict[str, float], bool]:
    """Builds, MEASURES both ways, and only then writes `dest`. (source, widened, measurement, optimizer carried).

    The measurement is flat: the zero-TECH one under its plain keys, the random-TECH one under `RANDOM_TECH` + key.
    """
    source, dest = Path(source), check_destination(dest)
    observations = check_observations(observations if observations is not None
                                      else synthetic_observations(64, seed))
    src, dst, carried, why = build_tech_model(source, seed, zero_new_moments=zero_new_moments)
    zero = displacement(src, dst, observations)
    noisy = displacement(src, dst, observations, tech=random_tech_block(observations.shape[0], seed))
    problems = verdict(zero) + [f"with a random TECH block, {p}" for p in verdict(noisy)]
    if not carried and not allow_fresh_optimizer:
        problems.append(f"Adam's state was NOT carried ({why}); pass --allow-fresh-optimizer to accept a fresh one")
    if problems:
        raise MigrationRefused("; ".join(problems))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dst.save(dest)
    return src, dst, {**zero, **{RANDOM_TECH + k: v for k, v in noisy.items()}}, carried


def describe(src: PPO, dst: PPO) -> None:
    s = src.policy.state_dict()
    for name, value in dst.policy.state_dict().items():
        if name in FIRST_LAYERS:
            note = f"columns 0-{V1_INPUTS - 1} copied, {V1_INPUTS}-{V2_INPUTS - 1} zero"
        elif name == "action_net.weight":
            note = f"rows 0-{V1_LOGITS - 1} copied, {V1_LOGITS}-{V2_LOGITS - 1} zero"
        elif name == "action_net.bias":
            note = f"rows 0-{V1_LOGITS - 1} copied; 45 = ln 45, 51 = ln 17, 55 = ln 9, the rest 0"
        else:
            note = "copied"
        print(f"{name:36s} {str(tuple(s[name].shape)):>12s} -> {str(tuple(value.shape)):12s} {note}")


def _value_text(name: str, value: float) -> str:
    if name == "bitwise":
        return "yes" if value else "no"
    return repr(value) if name in REPR_KEYS else f"{value:.9f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="a v1 campaign checkpoint (479 inputs, 12 action dims)")
    ap.add_argument("dest", type=Path, help="where to write the v2 checkpoint; must end in .zip and not exist")
    ap.add_argument("--obs-from", type=Path,
                    help="model dir whose ckpt_*_steps.zip hold real observations (default: the source's own dir)")
    ap.add_argument("--max-states", type=int, default=DEFAULT_STATES)
    ap.add_argument("--synthetic", type=int, default=0, help="measure on N synthetic states instead")
    ap.add_argument("--ent-floor", type=float, help="the stage's current ent_floor; the re-based one is printed")
    ap.add_argument("--allow-fresh-optimizer", action="store_true", help="accept losing Adam's state (not for S7)")
    ap.add_argument("--zero-new-moments", action="store_true",
                    help="leave the new exp_avg_sq entries at 0.0 (the plan's original surgery; see step 4)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    try:
        check_destination(a.dest)  # before a single observation is read
        if a.max_states < 1:
            raise ValueError(f"--max-states {a.max_states}: the measurement needs at least one state")
        if a.synthetic < 0:
            raise ValueError(f"--synthetic {a.synthetic}: pass a positive count, or leave it out")
        if a.synthetic > 0:
            observations, where = synthetic_observations(a.synthetic, a.seed), f"{a.synthetic} synthetic states"
        else:
            obs_dir = a.obs_from or a.source.parent
            observations = checkpoint_observations(obs_dir, a.max_states)
            if observations is None:
                raise ValueError(f"no ckpt_*_steps.zip with a readable _last_obs in {obs_dir}: pass --synthetic N")
            where = f"{len(observations)} real states (_last_obs) from {obs_dir}"
        src, dst, measured, carried = add_tech_heads(a.source, a.dest, observations=observations, seed=a.seed,
                                                     allow_fresh_optimizer=a.allow_fresh_optimizer,
                                                     zero_new_moments=a.zero_new_moments)
    except (FileExistsError, ValueError, MigrationRefused) as exc:
        reason = " ".join(str(exc).split())
        print(f"MIGRATION REFUSED: {reason} -- nothing was written", file=sys.stderr)
        return 2
    describe(src, dst)
    modes = split_modes(measured)
    print(f"measured on {where}, both ways:")
    for label, values in modes:
        print(f"  {label}:")
        for name, value in values.items():
            print(f"    {name:22s} {_value_text(name, value)}")
    bitwise = all(values["bitwise"] for _, values in modes)
    if bitwise:
        print("  => bitwise: yes -- the 12 shared dims' logits and the value equal the source's bit for bit, "
              "on every state, both ways")
    else:
        print("  => bitwise: no -- inside every tolerance, but not bit for bit (float reordering)")
    print("  => mean_kl " + " / ".join(repr(values["mean_kl"]) for _, values in modes)
          + f" (tolerance {KL_TOL:g}); mean_abs_dv " + " / ".join(repr(values["mean_abs_dv"]) for _, values in modes))
    if a.ent_floor is not None:
        print(f"  => ent_floor {a.ent_floor:g} -> {a.ent_floor + measured['entropy_added']:.4f} "
              "(speed.train.ent_floor at the install)")
    print(f"Saved {a.dest} ({dst.num_timesteps:,} steps and {dst._n_updates:,} updates carried, optimizer "
          f"{'carried' if carried else 'FRESH (--allow-fresh-optimizer)'}, new second moments "
          f"{'zero (--zero-new-moments)' if a.zero_new_moments else 'filled with the old means'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
