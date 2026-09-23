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
     continues; a carry that fails is REFUSED unless --allow-fresh-optimizer;
  5. `num_timesteps` and `_n_updates` carried, so the driver's step axis stays continuous.

Before writing anything it MEASURES: mean KL over the 12 shared dims, greedy changes and mean |dV| between the
source (fed the v1 vector) and the result (fed the same vector with 479-529 = 0.0), over REAL observations -- the
`_last_obs` every SB3 checkpoint carries, twelve states per ckpt_*_steps.zip, read ~120 KB at a time -- or a
synthetic batch. Anything above tolerance is a copy bug: the script exits 2 and writes nothing. It also prints the
entropy the new heads add (1.3986 nats at these priors): the new `ent_floor` is the old one plus that.

Never overwrites: `dest` must not exist, and the source is only read. After the break, the 479-wide files go to
`pre_tech/` with scripts/quarantine_pre_tech.py (a MOVE, never a delete).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import ACTION_NVEC_CAMPAIGN, ObsLayout, action_space  # noqa: E402

from transfer_weights import SpacesEnv  # noqa: E402

V1_INPUTS, V2_INPUTS = 479, 530
V1_LOGITS, V2_LOGITS = 45, 57
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
DV_TOL = 1e-3  # mean |dV|: float reordering of a 530- vs a 479-long dot product, nothing more
PRIOR_TOL = 1e-3
DEFAULT_STATES = 2400  # 200 checkpoints x 12 games
_CKPT = re.compile(r"ckpt_(\d+)_steps\.zip")


class MigrationRefused(RuntimeError):
    """The measurement found a difference it must not. Nothing was written."""


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


def build_tech_model(source, seed: int = 0) -> tuple[PPO, PPO, bool, str]:
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
    dst.num_timesteps = src.num_timesteps
    dst._n_updates = getattr(src, "_n_updates", 0)
    return src, dst, carried, why


def synthetic_observations(rows: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, V1_INPUTS)).astype(np.float32)


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


def displacement(src: PPO, dst: PPO, observations: np.ndarray) -> dict[str, float]:
    """How far the widened policy moved, and where its new heads sit. `observations` are v1 vectors."""
    old_obs = torch.as_tensor(observations, dtype=torch.float32)
    new_obs = torch.as_tensor(np.pad(observations, ((0, 0), (0, V2_INPUTS - V1_INPUTS))), dtype=torch.float32)
    with torch.no_grad():
        old = src.policy.get_distribution(old_obs).distribution
        new = dst.policy.get_distribution(new_obs).distribution
        old_v = src.policy.predict_values(old_obs).flatten()
        new_v = dst.policy.predict_values(new_obs).flatten()
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
    """Every tolerance the measurement breaks, as text. Empty = exact."""
    problems = []
    if measured["mean_kl"] > KL_TOL:
        problems.append(f"mean_kl {measured['mean_kl']:.3g} > {KL_TOL:g}")
    if measured["greedy_changed"] > 0.0:
        problems.append(f"greedy_changed {measured['greedy_changed']:.3g} > 0")
    if measured["mean_abs_dv"] > DV_TOL:
        problems.append(f"mean_abs_dv {measured['mean_abs_dv']:.3g} > {DV_TOL:g}")
    for name in PRIORS:
        if measured[f"{name}_prior_err"] > PRIOR_TOL:
            problems.append(f"{name} head off its prior by {measured[f'{name}_prior_err']:.3g}")
        if measured[f"{name}_state_spread"] > 1e-6:
            problems.append(f"{name} head depends on the state ({measured[f'{name}_state_spread']:.3g})")
    if abs(measured["entropy_added"] - PRIOR_ENTROPY) > PRIOR_TOL:
        problems.append(f"entropy added {measured['entropy_added']:.4f}, the priors say {PRIOR_ENTROPY:.4f}")
    return problems


def add_tech_heads(source, dest, *, observations: np.ndarray | None = None, seed: int = 0,
                   allow_fresh_optimizer: bool = False) -> tuple[PPO, PPO, dict[str, float], bool]:
    """Builds, MEASURES, and only then writes `dest`. (source, widened, measurement, optimizer carried)."""
    source, dest = Path(source), Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists: this script never overwrites (write beside it, then copy)")
    src, dst, carried, why = build_tech_model(source, seed)
    measured = displacement(src, dst, observations if observations is not None
                            else synthetic_observations(64, seed))
    problems = verdict(measured)
    if not carried and not allow_fresh_optimizer:
        problems.append(f"Adam's state was NOT carried ({why}); pass --allow-fresh-optimizer to accept a fresh one")
    if problems:
        raise MigrationRefused("; ".join(problems))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dst.save(dest)
    return src, dst, measured, carried


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="a v1 campaign checkpoint (479 inputs, 12 action dims)")
    ap.add_argument("dest", type=Path, help="where to write the v2 checkpoint; must not exist")
    ap.add_argument("--obs-from", type=Path,
                    help="model dir whose ckpt_*_steps.zip hold real observations (default: the source's own dir)")
    ap.add_argument("--max-states", type=int, default=DEFAULT_STATES)
    ap.add_argument("--synthetic", type=int, default=0, help="measure on N synthetic states instead")
    ap.add_argument("--ent-floor", type=float, help="the stage's current ent_floor; the re-based one is printed")
    ap.add_argument("--allow-fresh-optimizer", action="store_true", help="accept losing Adam's state (not for S7)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.synthetic > 0:
        observations, where = synthetic_observations(a.synthetic, a.seed), f"{a.synthetic} synthetic states"
    else:
        obs_dir = a.obs_from or a.source.parent
        observations = checkpoint_observations(obs_dir, a.max_states)
        if observations is None:
            print(f"no ckpt_*_steps.zip with a readable _last_obs in {obs_dir}: pass --synthetic N", file=sys.stderr)
            return 2
        where = f"{len(observations)} real states (_last_obs) from {obs_dir}"
    try:
        src, dst, measured, carried = add_tech_heads(a.source, a.dest, observations=observations, seed=a.seed,
                                                     allow_fresh_optimizer=a.allow_fresh_optimizer)
    except MigrationRefused as exc:
        print(f"MIGRATION REFUSED -- nothing was written: {exc}", file=sys.stderr)
        return 2
    describe(src, dst)
    print(f"measured on {where}:")
    for name, value in measured.items():
        print(f"  {name:22s} {value:.9f}")
    print(f"  => mean_kl {measured['mean_kl']:.9f} (tolerance {KL_TOL:g}): EXACT on the 12 shared dims")
    if a.ent_floor is not None:
        print(f"  => ent_floor {a.ent_floor:g} -> {a.ent_floor + measured['entropy_added']:.4f} "
              "(speed.train.ent_floor at the install)")
    print(f"Saved {a.dest} ({dst.num_timesteps:,} steps and {dst._n_updates:,} updates carried, optimizer "
          f"{'carried' if carried else 'FRESH (--allow-fresh-optimizer)'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
