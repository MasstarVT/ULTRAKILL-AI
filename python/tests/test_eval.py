"""Tests for scripts/eval.py's action mode: what `model.predict` is actually asked for.

No game needed:  python tests/test_eval.py   (or pytest)

The bug this file pins (measured 2026-09-20, recordings in `runs/probe_0-1_record/`): campaign policies are
trained, promoted and timed on SAMPLED actions, and their action heads are deliberately held near 7 nats of
entropy. `eval.py` and `full_run.py` nevertheless asked for argmax, and the promoted `Level_0-1.zip`
completed 0 of 5 argmax episodes against 19 of 20 sampled -- all five ended "stuck", two never left the spawn.
So campaign evaluation now samples by default, Cyber Grind keeps its old argmax default, and the times.md row
has to say which of the two produced the time.

Nothing here opens a port, loads a policy or writes into the repo: the rollout runs against a stub env and a
policy that only records the flag it was handed, and the recorded row goes into a temp copy of times.md.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval as eval_script  # noqa: E402  -- scripts/eval.py, not the builtin
import numpy as np  # noqa: E402
from ultrakill_ai.times import actions_note  # noqa: E402

TIMES_MD = ROOT.parent / "times.md"
MODEL = "models/specialists/Level_0-1.zip"


class RecordingPolicy:
    """`PPO.predict`'s signature, remembering only the action mode it was asked for."""

    def __init__(self):
        self.modes: list[bool] = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.modes.append(deterministic)
        return np.zeros(4, dtype=np.int64), None


class StubEnv:
    """Just enough env for `rollout`: `limit` decisions, then terminated."""

    def __init__(self, limit: int = 3):
        self.limit, self.steps = limit, 0

    def reset(self):
        self.steps = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.limit
        return np.zeros(8, dtype=np.float32), 1.0, done, False, {"end_reason": "level_complete"}


# ---------------------------------------------------------------------------
# Which mode an eval runs in
# ---------------------------------------------------------------------------


def test_campaign_evaluation_samples_actions_by_default():
    """The fix: a campaign policy is evaluated the way it was trained and promoted."""
    assert eval_script.resolve_deterministic("campaign") is False
    assert eval_script.resolve_deterministic("campaign", stochastic=True) is False, "already the default"
    assert eval_script.resolve_deterministic("campaign", deterministic=True) is True, "the explicit opt-in"


def test_cyber_grind_keeps_the_old_argmax_default():
    """Only campaign mode was measured, so only campaign mode changes."""
    assert eval_script.resolve_deterministic("cybergrind") is True
    assert eval_script.resolve_deterministic("cybergrind", stochastic=True) is False, "--stochastic still means this"
    assert eval_script.resolve_deterministic("cybergrind", deterministic=True) is True


def test_the_action_flags_parse_and_cannot_both_be_given():
    parser = eval_script.build_parser()
    args = parser.parse_args([MODEL, "--level", "Level 0-1"])
    assert args.deterministic is False and args.stochastic is False
    assert parser.parse_args([MODEL, "--deterministic"]).deterministic is True
    assert parser.parse_args([MODEL, "--stochastic"]).stochastic is True, "the old flag still parses"
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args([MODEL, "--deterministic", "--stochastic"])
        except SystemExit:
            pass
        else:
            raise AssertionError("asking for both modes at once must be refused, not silently resolved")


def test_the_difficulty_override_parses_and_defaults_to_the_configs():
    """`models/specialists/<level>.zip` has NO env_config.yaml beside it, so without this flag an eval of a
    promoted specialist runs at EnvConfig's default difficulty of -1: whatever the game itself happens to be
    set to. The flag is how a recorded time is pinned to a difficulty on purpose."""
    parser = eval_script.build_parser()
    assert parser.parse_args([MODEL]).difficulty is None, "default: the env config's own value"
    assert parser.parse_args([MODEL, "--difficulty", "4"]).difficulty == 4
    assert parser.parse_args([MODEL, "--difficulty", "-1"]).difficulty == -1, "-1 is the game's own setting"
    with contextlib.redirect_stderr(io.StringIO()):
        for refused in ("5", "-2"):
            try:
                parser.parse_args([MODEL, "--difficulty", refused])
            except SystemExit:
                continue
            raise AssertionError(f"--difficulty {refused} is not a difficulty this build can run")


# ---------------------------------------------------------------------------
# What predict is actually handed
# ---------------------------------------------------------------------------


def test_the_rollout_passes_the_chosen_action_mode_to_every_decision():
    for deterministic in (False, True):
        policy, env = RecordingPolicy(), StubEnv(3)
        total, steps, info = eval_script.rollout(env, policy, deterministic=deterministic)
        assert (steps, total) == (3, 3.0) and info["end_reason"] == "level_complete"
        assert policy.modes == [deterministic] * 3


def test_a_campaign_eval_with_no_flags_samples_every_decision():
    """Flags to predict, end to end -- the path that failed at 0-1's spawn before 2026-09-20."""
    args = eval_script.build_parser().parse_args([MODEL, "--level", "Level 0-1", "--record-times"])
    deterministic = eval_script.resolve_deterministic(
        "campaign", deterministic=args.deterministic, stochastic=args.stochastic)
    policy = RecordingPolicy()
    eval_script.rollout(StubEnv(4), policy, deterministic=deterministic)
    assert policy.modes == [False, False, False, False]


# ---------------------------------------------------------------------------
# What the recorded row says
# ---------------------------------------------------------------------------


def test_the_recorded_row_says_which_action_mode_produced_the_time():
    assert actions_note(False) == "sampled actions"
    assert actions_note(True) == "deterministic (argmax) actions"
    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.md"
        shutil.copy2(TIMES_MD, times)
        model = types.SimpleNamespace(num_timesteps=1_000_000)
        args = argparse.Namespace(record_times=True, model=MODEL)

        def report(seconds: float, deterministic: bool) -> None:
            info = {"completed": True, "level_seconds": seconds, "rank": "A", "kills": 3, "deaths": 0,
                    "style": 0, "difficulty": 3, "restarts": 0}
            eval_script.report_campaign(args, "Level 0-1", model, [(0.0, info)],
                                        deterministic=deterministic, times_md=times)

        report(80.0, False)
        assert "1/1 eval runs completed (sampled actions)" in times.read_text(encoding="utf-8")
        report(79.0, True)
        text = times.read_text(encoding="utf-8")
        assert "1/1 eval runs completed (deterministic (argmax) actions)" in text
        assert "specialists@1.00M" in text, "the generation still names the model's folder and its steps"


def test_nothing_is_recorded_without_record_times():
    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.md"
        shutil.copy2(TIMES_MD, times)
        before = times.read_text(encoding="utf-8")
        info = {"completed": True, "level_seconds": 12.5, "rank": "A", "kills": 3, "deaths": 0,
                "style": 0, "difficulty": 3, "restarts": 0}
        eval_script.report_campaign(argparse.Namespace(record_times=False, model=MODEL), "Level 0-1",
                                    types.SimpleNamespace(num_timesteps=1), [(0.0, info)],
                                    deterministic=False, times_md=times)
        assert times.read_text(encoding="utf-8") == before


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
