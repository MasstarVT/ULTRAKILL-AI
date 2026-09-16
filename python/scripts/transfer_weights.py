"""Builds the campaign's starting weights from a Cyber Grind checkpoint.

The campaign observation layout is the Cyber Grind one with its 5 retired route values (inputs 443-447,
always zero in Cyber Grind) replaced by the 36-value campaign block, so inputs 0-442 mean the same thing
in both. The widened policy:
  - copies both hidden stacks (policy and value) from the source;
  - takes first-layer columns 0-442 from the source and leaves columns 443-478 at zero, so on any Cyber
    Grind input it computes exactly what the source did and the campaign block starts disconnected;
  - scales the action head (weights and bias) by --action-scale, which softens every logit and raises
    entropy, so the fighting reflexes carry over without locking in "fire always, walk backwards";
  - keeps its own freshly initialised final value layer, because the campaign's reward scale has nothing
    to do with Cyber Grind's;
  - starts with a fresh optimizer.

    python scripts/transfer_weights.py models/cybergrind_ppo_v2/best.zip models/campaign_ppo/transfer_init.zip
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip

If the first live rollout's entropy is outside 6-10 nats, rerun this once with another --action-scale and
restart training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.spaces import ObsLayout, action_space  # noqa: E402

SHARED_INPUTS = 443  # player 0-16, enemies 17-416, rays 417-432, ground rays 433-440, Cyber Grind 441-442
HIDDEN_LAYERS = tuple(
    f"mlp_extractor.{net}.{layer}.{kind}" for net in ("policy_net", "value_net") for layer in (0, 2) for kind in ("weight", "bias")
)
ACTION_HEAD = ("action_net.weight", "action_net.bias")


class SpacesEnv(gym.Env):
    """Holds an observation and action space so PPO can build a policy without a game. Never reset or stepped."""

    def __init__(self, observation_space: gym.spaces.Space, actions: gym.spaces.Space | None = None):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = actions if actions is not None else action_space()


def widen_state_dict(src: dict, dst: dict, shared_inputs: int, action_scale: float) -> dict:
    """`dst` (a policy state dict) with the source's weights widened into it; neither input is modified.

    First-layer columns from `shared_inputs` on are zero; `value_net.*` stays the destination's own.
    """
    out = {name: value.clone() for name, value in dst.items()}
    for name in HIDDEN_LAYERS:
        if name.endswith(".0.weight"):
            widened = torch.zeros_like(dst[name])
            widened[:, :shared_inputs] = src[name][:, :shared_inputs]
            out[name] = widened
        else:
            out[name] = src[name].clone()
    for name in ACTION_HEAD:
        out[name] = src[name] * action_scale
    return out


def transfer(source: Path, dest: Path, action_scale: float = 0.5, seed: int = 0) -> tuple[PPO, PPO]:
    """Loads a Cyber Grind PPO checkpoint, widens it to the campaign layout and saves it to `dest`."""
    src = PPO.load(source, device="cpu")
    expected = ObsLayout().space().shape
    if src.observation_space.shape != expected:
        raise ValueError(f"{source} takes inputs of shape {src.observation_space.shape}, not the Cyber Grind layout's {expected}")
    # Seed torch here instead of passing seed= to PPO: a seed stored in the model would reseed every
    # later resume of the campaign run to the same random sequence.
    torch.manual_seed(seed)
    dst = PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), src.action_space), policy_kwargs=src.policy_kwargs, device="cpu")
    dst.policy.load_state_dict(widen_state_dict(src.policy.state_dict(), dst.policy.state_dict(), SHARED_INPUTS, action_scale))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dst.save(dest)
    return src, dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="Cyber Grind PPO checkpoint, e.g. models/cybergrind_ppo_v2/best.zip")
    ap.add_argument("dest", type=Path, help="where to save the campaign starting weights")
    ap.add_argument("--action-scale", type=float, default=0.5, help="multiplies the action logits; below 1 raises entropy")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the fresh value head")
    a = ap.parse_args()

    src, dst = transfer(a.source, a.dest, a.action_scale, a.seed)
    src_state = src.policy.state_dict()
    for name, value in dst.policy.state_dict().items():
        if name.endswith(".0.weight"):
            note = f"columns 0-{SHARED_INPUTS - 1} copied, {SHARED_INPUTS}+ zero"
        elif name in ACTION_HEAD:
            note = f"copied x{a.action_scale}"
        elif name in HIDDEN_LAYERS:
            note = "copied"
        else:
            note = "fresh"
        print(f"{name:34s} {str(tuple(src_state[name].shape)):>10s} -> {str(tuple(value.shape)):10s} {note}")
    print(f"Saved {a.dest} ({src.num_timesteps:,} source steps, optimizer fresh)")


if __name__ == "__main__":
    main()
