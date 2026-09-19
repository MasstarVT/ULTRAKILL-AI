"""Tests for scripts/full_run.py, the specialist chaining tool.

No game needed:  python tests/test_full_run.py   (or pytest)

The chaining logic -- the order, the skipped levels, the total, the times.md posting -- is tested against an
injected `play`, and `play_level` itself is driven against `test_campaign_env.FakeLevel`, the same fake bridge
the campaign env tests use. Nothing here opens a port or writes into the repo.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import full_run  # noqa: E402
import yaml  # noqa: E402
from full_run import LevelResult  # noqa: E402
from test_campaign_env import LEVEL, FakeLevel, forward  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.times import parse_time, short_level  # noqa: E402

TIMES_MD = ROOT.parent / "times.md"
PLAN = types.SimpleNamespace(env={"mode": "campaign", "level": LEVEL, "fixed_fps": 30, "frameskip": 2,
                                 "rewards": {"level_complete": 100.0, "time": 0.02}})


def make_specialists(models: Path, levels) -> None:
    (models / "specialists").mkdir(parents=True, exist_ok=True)
    for level in levels:
        full_run.specialist_path(models, level).write_bytes(b"weights")


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def test_the_chain_plays_every_level_that_has_a_specialist_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp) / "models"
        make_specialists(models, ["Level 0-1", "Level 0-4"])
        played = []

        def play(level, model_path):
            played.append((level, model_path.name))
            return LevelResult(level=level, completed=True, seconds=100.0 + len(played), rank="C", kills=7)

        results = full_run.run_chain(["Level 0-1", "Level 0-3", "Level 0-4"], models, play)
        assert [r.level for r in results] == ["Level 0-1", "Level 0-3", "Level 0-4"]
        assert played == [("Level 0-1", "Level_0-1.zip"), ("Level 0-4", "Level_0-4.zip")]
        assert results[1].skipped and not results[1].played, "a level with no specialist is skipped, not failed"
        assert results[0].specialist.endswith("models/specialists/Level_0-1.zip")
        assert full_run.total_seconds(results) == 101.0 + 102.0


def test_only_completed_levels_count_toward_the_total_and_the_table_says_so():
    results = [LevelResult(level="Level 0-1", specialist="x", completed=True, seconds=90.0, rank="B"),
               LevelResult(level="Level 0-3", specialist="x", completed=False, end_reason="stuck", steps=12000),
               LevelResult(level="Level 0-4", skipped="no specialist")]
    assert full_run.total_seconds(results) == 90.0
    table = full_run.format_table(results)
    assert "SKIPPED" in table and "did not finish (stuck)" in table
    assert "1/2 levels completed (1 skipped" in table
    assert "01:30.000" in table


def test_the_generation_names_the_chain_and_the_date():
    assert full_run.generation("2026-09-18") == "specialists@2026-09-18"
    assert full_run.generation().startswith("specialists@")


def test_posting_writes_one_row_per_completed_level_through_the_times_helpers():
    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.md"
        shutil.copy2(TIMES_MD, times)
        results = [LevelResult(level="Level 0-1", completed=True, seconds=1.25, rank="P", kills=40, deaths=0),
                   LevelResult(level="Level 0-3", completed=False, end_reason="stuck"),
                   LevelResult(level="Level 0-4", skipped="no specialist")]
        posted = full_run.post_results(times, results, gen="specialists@2026-09-18")
        assert posted == ["Level 0-1 00:01.250 rank P"], "only a completed level has a time to post"
        text = times.read_text(encoding="utf-8")
        assert "specialists@2026-09-18" in text and "00:01.250" in text
        assert "one specialist per level" in text
        # The leaderboard row for 0-1 is now this (absurdly fast) time, because the helper replaces a slower one.
        row = next(line for line in text.splitlines()
                   if line.startswith("| %s " % short_level("Level 0-1")))
        assert parse_time(row.split("|")[2].strip()) == 1.25
        assert full_run.post_results(times, [], gen="specialists@2026-09-18") == []


# ---------------------------------------------------------------------------
# The env one level is played with
# ---------------------------------------------------------------------------


def test_the_eval_config_is_forced_into_eval_shape():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        cfg = full_run.eval_config(LEVEL, cwd / "model.zip", PLAN, cwd, port=47800)
        assert cfg.level == LEVEL and cfg.levels == [] and cfg.curriculum_path == ""
        assert cfg.fresh_start_prob == 1.0, "every episode is a whole level, so it has an official time"
        assert cfg.soft_death is False and cfg.port == 47800
        assert cfg.explore_dir == "" and cfg.best_runs_dir == "", "the archives belong to training"
        assert cfg.bridge_relaunch is False and cfg.env_log_dir == "", "an eval may never restart a game"


def test_the_eval_config_prefers_the_specialists_own_training_settings():
    """A specialist is played under the settings it was TRAINED under whenever they are still on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        model_dir = cwd / "models" / "spec_0-1"
        model_dir.mkdir(parents=True)
        trained = EnvConfig(mode="campaign", level=LEVEL, max_steps=12345, pitch_limit_deg=45.0,
                            levels=["Level 0-1", "Level 0-3"], explore_dir="models/spec_0-1")
        (model_dir / "env_config.yaml").write_text(yaml.safe_dump(trained.to_dict()), encoding="utf-8")
        cfg = full_run.eval_config(LEVEL, model_dir / "best.zip", PLAN, cwd, port=47801)
        assert cfg.max_steps == 12345 and cfg.pitch_limit_deg == 45.0
        assert cfg.levels == [] and cfg.explore_dir == "", "and then forced into eval shape anyway"

        # With no env_config.yaml the generated stage config is next, and the plan template last.
        (model_dir / "env_config.yaml").unlink()
        generated = cwd / "configs" / "generated" / "spec_0-1.yaml"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(yaml.safe_dump({"env": {"mode": "campaign", "level": LEVEL, "max_steps": 777},
                                             "train": {"run_name": "spec_0-1"}}), encoding="utf-8")
        assert full_run.eval_config(LEVEL, model_dir / "best.zip", PLAN, cwd, port=47800).max_steps == 777
        generated.unlink()
        assert full_run.eval_config(LEVEL, model_dir / "best.zip", PLAN, cwd, port=47800).max_steps == \
            EnvConfig().max_steps


def test_a_speed_stage_specialist_is_played_with_its_own_runs_settings():
    """`mode` in the sidecar says which stage promoted the file, and a speed stage is a run of its own."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        models = cwd / "models"
        specialist = models / "specialists" / "Level_0-1.zip"
        specialist.parent.mkdir(parents=True)
        specialist.write_bytes(b"weights")
        assert full_run.specialist_mode(models, LEVEL) == "complete", "no sidecar reads as the old behaviour"
        specialist.with_suffix(".json").write_text(json.dumps({"level": LEVEL, "status": "done"}),
                                                   encoding="utf-8")
        assert full_run.specialist_mode(models, LEVEL) == "complete", "a sidecar written before kinds existed"
        specialist.with_suffix(".json").write_text(
            json.dumps({"level": LEVEL, "mode": "speed", "target_seconds": 120.0}), encoding="utf-8")
        assert full_run.specialist_mode(models, LEVEL) == "speed"

        for run, steps in (("spec_0-1", 111), ("spec_0-1_speed", 222)):
            d = models / run
            d.mkdir(parents=True, exist_ok=True)
            (d / "env_config.yaml").write_text(
                yaml.safe_dump(EnvConfig(mode="campaign", level=LEVEL, max_steps=steps).to_dict()),
                encoding="utf-8")
        assert full_run.eval_config(LEVEL, specialist, PLAN, cwd, port=47800, mode="speed").max_steps == 222
        assert full_run.eval_config(LEVEL, specialist, PLAN, cwd, port=47800).max_steps == 111
        # ... and a speed run that never wrote one falls back to the complete stage's, not to the defaults.
        (models / "spec_0-1_speed" / "env_config.yaml").unlink()
        assert full_run.eval_config(LEVEL, specialist, PLAN, cwd, port=47800, mode="speed").max_steps == 111


# ---------------------------------------------------------------------------
# Playing one level, against the fake bridge
# ---------------------------------------------------------------------------


class WalkForward:
    """A policy that always walks forward, with `PPO.predict`'s signature."""

    def __init__(self):
        self.calls = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls += 1
        return forward(), None


def fake_env(cwd: Path):
    cfg = full_run.eval_config(LEVEL, cwd / "model.zip", PLAN, cwd, port=47800)
    env = UltrakillEnv(cfg)
    env.client = FakeLevel()
    return env


def test_play_level_reports_the_official_time_of_a_completed_level():
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_env(Path(tmp))
        try:
            result = full_run.play_level(LEVEL, env, WalkForward())
        finally:
            env.close()
        assert result.completed and result.level == LEVEL
        assert result.seconds is not None and result.seconds > 0
        assert result.end_reason == "level_complete" and result.steps > 0
        assert isinstance(result.rank, str)
        assert result.difficulty == 3, "the difficulty the game actually read, from the campaign block"


def test_play_level_reports_a_level_that_did_not_finish_without_inventing_a_time():
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_env(Path(tmp))
        env.cfg.max_steps = 5  # truncate long before the exit
        try:
            result = full_run.play_level(LEVEL, env, WalkForward())
        finally:
            env.close()
        assert not result.completed and result.seconds is None
        assert result.end_reason == "max_steps"


def test_the_chain_end_to_end_over_the_fake_level():
    """run_chain + play_level together: one level played, one skipped, and the JSON shape holds."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        models = cwd / "models"
        make_specialists(models, [LEVEL])
        envs = []

        def play(level, model_path):
            env = fake_env(cwd)
            envs.append(env)
            try:
                return full_run.play_level(level, env, WalkForward())
            finally:
                env.close()

        results = full_run.run_chain([LEVEL, "Level 0-3"], models, play)
        assert results[0].completed and results[1].skipped
        assert full_run.total_seconds(results) == results[0].seconds
        assert full_run.best_of(results) is results[0]
        assert json.loads(json.dumps([r.__dict__ for r in results]))[0]["level"] == LEVEL


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
