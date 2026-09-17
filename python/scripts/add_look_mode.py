"""Widens a campaign checkpoint for the look-mode action and the re-used target observation slots.

    python scripts/add_look_mode.py models/campaign_ppo_ground/latest.zip models/campaign_gates/look_init.zip
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_gates/look_init.zip

Two things change about the campaign policy and this script migrates a run across both without restarting it:

  - The action space gains a 12th dimension (free / enemy / gate look). Three logit rows are appended at the
    end, both zero, so all three modes start equally likely and every existing row keeps its index.
  - Observation inputs 448-455 stop carrying the NavMesh path hint and start carrying the route target. Those
    eight first-layer columns are therefore ZEROED in both hidden stacks, and -- this part is required, not
    optional -- the mean the removed inputs used to contribute is folded into the first-layer bias:

        b += W[:, 448:456] @ MU ; W[:, 448:456] = 0

Do not read this as behaviour-preserving. Those inputs carried real, varying values (`path.status` was
`partial` on 82.19% of 32,022 logged decisions), so the new policy is displaced from the old one; the fold
halves the displacement rather than removing it. Measured on models/campaign_ppo_ground/latest.zip over 436
recorded Level 0-1 states: zero only 0.0449 mean KL and 45.6% of greedy actions changed, zero + mean-fold
0.0262 and 29.6%, against a target_kl of 0.03. Pass --stats to re-measure both on a real observation dump; the
numbers print at the end.

The action-space change makes every earlier campaign checkpoint unloadable against the new env: this script is
the only migration path. Cyber Grind is untouched (its action space does not change), and
scripts/transfer_weights.py does the same widening on the Cyber Grind -> campaign path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import ExplorationArchive, safe_name  # noqa: E402
from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC,
    ACTION_NVEC_CAMPAIGN,
    ObsLayout,
    action_space,
    pack_observation,
    yaw_frame,
)

from transfer_weights import ACTION_HEAD, SpacesEnv  # noqa: E402

TARGET_SLOTS = slice(448, 456)  # the campaign block's indices 5-12, absolute
FIRST_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
FIRST_BIASES = ("mlp_extractor.policy_net.0.bias", "mlp_extractor.value_net.0.bias")
# Per-column mean of inputs 448-455, measured over 436 real Level 0-1 observations. --stats recomputes it.
MU_MEASURED = (-0.054, -0.098, -0.034, 0.324, 0.114, 0.000, 0.821, 0.179)
KL_BOUND = 0.03  # the run's target_kl: the displacement must stay inside one PPO update


# ---------------------------------------------------------------------------
# The observations the source policy was trained on
# ---------------------------------------------------------------------------

def legacy_target_slots(raw: dict) -> list[float]:
    """What inputs 448-455 carried before this change: the NavMesh path hint, packed exactly as spaces.py did.

    Reproduced here rather than imported, because `campaign_block` no longer packs it at all. This is the only
    way to feed the old policy the inputs it actually saw, which is what makes the measurement below honest.
    """
    out = [0.0] * 8
    c, p = raw.get("campaign"), raw.get("player")
    if not c or not p:
        return out
    path = c.get("path") or {}
    status = path.get("status", "none")
    if status in ("complete", "partial") and path.get("next_corner"):
        pos = p["pos"]
        corner = path["next_corner"]
        x, y, z = yaw_frame((corner[0] - pos[0], corner[1] - pos[1], corner[2] - pos[2]), p["yaw"])
        out[0:5] = [x / 50.0, y / 50.0, z / 50.0, math.sqrt(x * x + y * y + z * z) / 50.0, path.get("length", 0.0) / 300.0]
    out[5:8] = [float(status == "complete"), float(status == "partial"), float(status not in ("complete", "partial"))]
    return out


def ground_point(raw: dict, ground_ray_length: float):
    """env._ground_point for a recorded observation (0.5.x dumps have no centre ray, so the ring minimum it is)."""
    player, rays = raw.get("player"), raw.get("ground_rays") or ()
    if not player or not rays:
        return None
    drop = min(rays)
    if drop >= ground_ray_length - 0.5:
        return None
    x, y, z = player["pos"]
    return (x, y - drop, z)


def recorded_observations(path: Path, archive: ExplorationArchive | None = None) -> np.ndarray:
    """Every raw observation in a `{"raw": {...}}` JSONL dump, packed the way the SOURCE policy saw it."""
    layout = ObsLayout(campaign=True)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line).get("raw")
        if not raw or not raw.get("player"):
            continue
        explore = None
        if archive is not None:
            explore = archive.features(ground_point(raw, layout.ground_ray_length) or raw["player"]["pos"], raw["player"]["yaw"])
        vector = pack_observation(raw, layout, {}, explore)
        vector[TARGET_SLOTS] = legacy_target_slots(raw)
        rows.append(vector)
    if not rows:
        raise ValueError(f"{path} holds no observations with a player")
    return np.stack(rows)


def find_archive(model: Path, level: str = "Level 0-1", port: int = 47800) -> ExplorationArchive | None:
    """The exploration counts saved next to the model, so the 9 map inputs are not all zero in the measurement."""
    path = model.parent / f"explore_{safe_name(level)}_{port}.npz"
    if not path.exists():
        return None
    archive = ExplorationArchive.load(path, 4.0)
    return archive if archive.counts else None


# ---------------------------------------------------------------------------
# The surgery
# ---------------------------------------------------------------------------

def widen_state_dict(src: dict, dst: dict, mu, mean_fold: bool = True) -> dict:
    """`dst`'s state dict with everything taken from `src`, the eight columns zeroed and the action head widened."""
    out = {}
    for name, value in dst.items():
        source = src.get(name)
        out[name] = source.clone() if source is not None and source.shape == value.shape else value.clone()
    fold = torch.as_tensor(mu, dtype=torch.float32)
    for weight_name, bias_name in zip(FIRST_LAYERS, FIRST_BIASES):
        weight = out[weight_name]
        if mean_fold:  # before zeroing: the removed columns' mean contribution becomes a constant in the bias
            out[bias_name] = out[bias_name] + weight[:, TARGET_SLOTS] @ fold.to(weight.dtype)
        weight[:, TARGET_SLOTS] = 0.0
    for name in ACTION_HEAD:
        widened = torch.zeros_like(dst[name])  # the three new look-mode rows stay zero: all modes equally likely
        widened[: src[name].shape[0]] = src[name]
        out[name] = widened
    return out


def carry_optimizer(src_model: PPO, dst_model: PPO) -> bool:
    """Carries Adam's moments over: zero rows for the new action logits, zeroed moments for the changed columns.

    A fresh optimizer is an acceptable fallback (it is what transfer_weights.py does), but it throws away the
    per-parameter scaling of a 2.9M-step run, so it is worth the attempt. Returns whether it worked.
    """
    try:
        names = [name for name, _ in src_model.policy.named_parameters()]
        if names != [name for name, _ in dst_model.policy.named_parameters()]:
            return False
        params = dict(dst_model.policy.named_parameters())
        state = {"param_groups": copy.deepcopy(dst_model.policy.optimizer.state_dict()["param_groups"]), "state": {}}
        for index, entry in src_model.policy.optimizer.state_dict()["state"].items():
            name = names[int(index)]
            shape = params[name].shape
            moments = {}
            for key, value in entry.items():
                if not torch.is_tensor(value) or value.dim() == 0 or value.shape == shape:
                    # Adam's `step` is a 0-dim tensor in torch 2.x: carried as is, so the bias correction
                    # continues from where the run left it rather than restarting at step 1.
                    moments[key] = value.clone() if torch.is_tensor(value) else value
                elif value.dim() == len(shape):
                    padded = torch.zeros(shape, dtype=value.dtype)
                    padded[: value.shape[0]] = value
                    moments[key] = padded
                else:
                    return False
            if name in FIRST_LAYERS:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in moments:
                        moments[key][:, TARGET_SLOTS] = 0.0
            state["state"][int(index)] = moments
        dst_model.policy.optimizer.load_state_dict(state)
        return True
    except (KeyError, IndexError, ValueError, RuntimeError, TypeError):
        return False


def add_look_mode(source: Path, dest: Path, mu=MU_MEASURED, mean_fold: bool = True, seed: int = 0) -> tuple[PPO, PPO, bool]:
    """(source model, widened model, optimizer carried) -- the widened one is saved to `dest`."""
    src = PPO.load(source, device="cpu")
    expected = ObsLayout(campaign=True).space().shape
    if tuple(src.observation_space.shape) != expected:
        raise ValueError(f"{source} takes inputs of shape {src.observation_space.shape}, not the campaign layout's {expected}")
    widths = tuple(getattr(src.action_space, "nvec", ()))
    if widths not in (tuple(ACTION_NVEC), tuple(ACTION_NVEC_CAMPAIGN)):
        raise ValueError(f"{source} has action space {widths}, neither the 11- nor the 12-dimension one")
    torch.manual_seed(seed)  # only the layers that are replaced wholesale depend on it
    dst = PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), action_space(campaign=True)),
              policy_kwargs=src.policy_kwargs, device="cpu")
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), mu, mean_fold))
    carried = carry_optimizer(src, dst)
    # The step axis has to stay continuous: this is the same run, not a new one.
    dst.num_timesteps = src.num_timesteps
    dst._n_updates = getattr(src, "_n_updates", 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dst.save(dest)
    return src, dst, carried


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def displacement(src: PPO, dst: PPO, observations: np.ndarray) -> dict[str, float]:
    """How far the widened policy moved: mean KL over the shared action dimensions, greedy changes, mean |dV|.

    The old policy is fed the inputs it was trained on and the new one the same vector; its first-layer columns
    there are zero, so what those slots hold no longer reaches it either way.
    """
    obs = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        old = src.policy.get_distribution(obs).distribution
        new = dst.policy.get_distribution(obs).distribution
        old_v = src.policy.predict_values(obs).flatten()
        new_v = dst.policy.predict_values(obs).flatten()
    shared = min(len(old), len(new))
    kl = torch.zeros(obs.shape[0])
    changed = torch.zeros(obs.shape[0], dtype=torch.bool)
    for k in range(shared):
        p, q = old[k].probs, new[k].probs
        kl += (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(dim=-1)
        changed |= old[k].probs.argmax(dim=-1) != new[k].probs.argmax(dim=-1)
    return {
        "states": float(obs.shape[0]),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "greedy_changed": float(changed.float().mean()),
        "mean_abs_dv": float((old_v - new_v).abs().mean()),
        "value_min": float(old_v.min()),
        "value_max": float(old_v.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="campaign PPO checkpoint, e.g. models/campaign_ppo_ground/latest.zip")
    ap.add_argument("dest", type=Path, help="where to save the widened model")
    ap.add_argument("--stats", type=Path, help="JSONL dump of recorded observations: recomputes MU and measures the displacement")
    ap.add_argument("--level", default="Level 0-1", help="level whose exploration archive to read beside the source")
    ap.add_argument("--no-mean-fold", action="store_true", help="zero the columns without folding their mean into the bias (worse; for comparison)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mu, observations = MU_MEASURED, None
    if a.stats:
        archive = find_archive(a.source, a.level)
        observations = recorded_observations(a.stats, archive)
        mu = observations[:, TARGET_SLOTS].mean(axis=0).tolist()
        print(f"MU over {len(observations)} states from {a.stats}"
              f"{' with the exploration archive beside the model' if archive else ' (no exploration archive found)'}:")
    else:
        print("MU measured over 436 recorded Level 0-1 states (pass --stats <run_raw.jsonl> to recompute):")
    print("  " + ", ".join(f"{v:+.3f}" for v in mu))

    src, dst, carried = add_look_mode(a.source, a.dest, mu, mean_fold=not a.no_mean_fold, seed=a.seed)
    src_state = src.policy.state_dict()
    for name, value in dst.policy.state_dict().items():
        if name in FIRST_LAYERS:
            note = "copied, columns 448-455 zeroed"
        elif name in FIRST_BIASES:
            note = "copied" if a.no_mean_fold else "copied + mean-fold"
        elif name in ACTION_HEAD:
            note = f"rows 0-{src_state[name].shape[0] - 1} copied, the rest zero (look mode)"
        else:
            note = "copied"
        print(f"{name:34s} {str(tuple(src_state[name].shape)):>12s} -> {str(tuple(value.shape)):12s} {note}")

    if observations is not None:
        for name, value in displacement(src, dst, observations).items():
            print(f"  {name:14s} {value:.4f}")
        measured = displacement(src, dst, observations)["mean_kl"]
        print(f"  mean KL {measured:.4f} against the run's target_kl {KL_BOUND}: "
              f"{'inside one update' if measured <= KL_BOUND else 'OUTSIDE THE BOUND, do not start the run'}")
    print(f"Saved {a.dest} ({dst.num_timesteps:,} steps carried, optimizer {'carried' if carried else 'FRESH'})")


if __name__ == "__main__":
    main()
