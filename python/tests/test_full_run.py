"""Tests for scripts/full_run.py, the specialist chaining tool.

No game needed:  python tests/test_full_run.py   (or pytest)

The chaining logic -- the order, the skipped levels, the total, the times.md posting -- is tested against an
injected `play`, and `play_level` itself is driven against `test_campaign_env.FakeLevel`, the same fake bridge
the campaign env tests use. Nothing here opens a port, or reads or writes anything in the repo -- the posting
tests build their own times.md from `test_times.TIMES_MD` and say which leaderboard row they are posting
against (`fixture_times`), so the suite never depends on what the live run last wrote to the real file.
"""

from __future__ import annotations

import contextlib
import io
import json
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
from test_times import TIMES_MD as EMPTY_TIMES_MD  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.times import TimeEntry, format_time, parse_time, record, short_level  # noqa: E402

PLAN = types.SimpleNamespace(env={"mode": "campaign", "level": LEVEL, "fixed_fps": 30, "frameskip": 2,
                                 # Brutal, as configs/specialists.yaml has been since 2026-09-20. The fake
                                 # level echoes back whatever difficulty the env asked for, exactly as the
                                 # mod does, so the chain's result carries it end to end.
                                 "difficulty": 4,
                                 "rewards": {"level_complete": 100.0, "time": 0.02}})


def make_specialists(models: Path, levels) -> None:
    (models / "specialists").mkdir(parents=True, exist_ok=True)
    for level in levels:
        full_run.specialist_path(models, level).write_bytes(b"weights")


def held_row(level=LEVEL, seconds=183.628, difficulty=3, rank="A") -> TimeEntry:
    """The leaderboard row a posting test is posting AGAINST, stated by the test itself."""
    return TimeEntry(level=level, seconds=seconds, rank=rank, generation="fixture@1.00M", difficulty=difficulty,
                     date="2026-09-20", kills=1, deaths=0, notes="fixture row")


def fixture_times(tmp: Path, *held: TimeEntry) -> Path:
    """A times.md of OUR OWN under `tmp`, holding exactly the rows `held` names.

    Never the repo's real times.md. That file is whatever the live training run last posted -- its 0-1 row
    went Brutal on 2026-09-20 -- so a test that copied it was asserting against a moving target, and the
    difficulty rule (`times.beats_record`, hardest difficulty first) then refused the Violent post these tests
    make. The seed rows go in through `times.record` itself, so the fixture can only ever hold rows the
    helper under test would have written.
    """
    text = EMPTY_TIMES_MD
    for entry in held:
        text = record(text, entry)
    path = tmp / "times.md"
    path.write_text(text, encoding="utf-8")
    return path


def leaderboard_row(text: str, level: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("| %s " % short_level(level)))


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
        # The fixture holds a Violent 0-1 row, and the chain posts Violent: a run has to be on at least the
        # row's own difficulty to replace it, whatever its time (`times.beats_record`).
        times = fixture_times(Path(tmp), held_row(level="Level 0-1", seconds=183.628, difficulty=3))
        results = [LevelResult(level="Level 0-1", completed=True, seconds=1.25, rank="P", kills=40, deaths=0,
                               difficulty=3),
                   LevelResult(level="Level 0-3", completed=False, end_reason="stuck"),
                   LevelResult(level="Level 0-4", skipped="no specialist")]
        posted = full_run.post_results(times, results, gen="specialists@2026-09-18")
        assert posted == ["Level 0-1 00:01.250 rank P"], "only a completed level has a time to post"
        text = times.read_text(encoding="utf-8")
        assert "specialists@2026-09-18" in text and "00:01.250" in text
        assert "one specialist per level" in text
        # The leaderboard row for 0-1 is now this (absurdly fast) time, because the helper replaces a slower one.
        assert parse_time(leaderboard_row(text, "Level 0-1").split("|")[2].strip()) == 1.25
        assert format_time(183.628) not in leaderboard_row(text, "Level 0-1"), "the seeded row is gone"
        assert len([line for line in text.splitlines()
                    if line.startswith("| %s " % short_level("Level 0-1"))]) == 1, "one row per level"
        assert not [line for line in text.splitlines()
                    if line.startswith(("| %s " % short_level("Level 0-3"),
                                        "| %s " % short_level("Level 0-4")))], \
            "a level that did not complete, or was skipped, gets no row at all"
        assert full_run.post_results(times, [], gen="specialists@2026-09-18") == []


def test_a_chained_run_that_never_learned_its_difficulty_does_not_take_a_labelled_row():
    """`LevelResult.difficulty` defaults to UNKNOWN, not to 3: a row must never say Violent by default.

    An UNKNOWN run ranks below every named difficulty, so it cannot overwrite a row that does know what it
    was set on -- however fast it is. It is still recorded in the generation history.
    """
    with tempfile.TemporaryDirectory() as tmp:
        times = fixture_times(Path(tmp), held_row(level="Level 0-1", seconds=183.628, difficulty=3))
        before = leaderboard_row(times.read_text(encoding="utf-8"), "Level 0-1")
        assert LevelResult(level="Level 0-1").difficulty == -1
        full_run.post_results(times, [LevelResult(level="Level 0-1", completed=True, seconds=1.25, rank="P")],
                              gen="specialists@2026-09-20")
        text = times.read_text(encoding="utf-8")
        assert leaderboard_row(text, "Level 0-1") == before, \
            "an unlabelled run may not replace a Violent leaderboard row"
        assert "specialists@2026-09-20" in text, "but the history still records that it happened"


def test_a_violent_chain_cannot_take_a_brutal_row_but_a_slower_brutal_chain_can():
    """The 2026-09-20 rule, through `post_results`: hardest difficulty first, then fastest.

    This is the case that used to be hidden by copying the repo's real times.md -- once its 0-1 row went
    Brutal, the Violent post above was correctly refused and the test failed for a reason that had nothing to
    do with `full_run`. Now the row is stated here, and both directions are asserted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        times = fixture_times(Path(tmp), held_row(level="Level 0-1", seconds=143.5, difficulty=4))
        before = leaderboard_row(times.read_text(encoding="utf-8"), "Level 0-1")
        assert "Brutal" in before

        # Violent, and far faster: the history takes it, the leaderboard does not.
        assert full_run.post_results(times, [LevelResult(level="Level 0-1", completed=True, seconds=1.25,
                                                         rank="P", difficulty=3)],
                                     gen="specialists@violent") == ["Level 0-1 00:01.250 rank P"]
        text = times.read_text(encoding="utf-8")
        assert leaderboard_row(text, "Level 0-1") == before, "a Violent time is not a claim about Brutal"
        assert "specialists@violent" in text, "the generation history still records the run"

        # Brutal, and SLOWER than the held Brutal row: still refused, because within a difficulty time wins.
        full_run.post_results(times, [LevelResult(level="Level 0-1", completed=True, seconds=200.0, rank="C",
                                                  difficulty=4)], gen="specialists@slow-brutal")
        assert leaderboard_row(times.read_text(encoding="utf-8"), "Level 0-1") == before

        # Brutal and faster: the row moves.
        full_run.post_results(times, [LevelResult(level="Level 0-1", completed=True, seconds=130.25, rank="A",
                                                  difficulty=4)], gen="specialists@fast-brutal")
        row = leaderboard_row(times.read_text(encoding="utf-8"), "Level 0-1")
        assert parse_time(row.split("|")[2].strip()) == 130.25 and "Brutal" in row


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


def test_a_chain_plays_each_specialist_on_the_difficulty_it_is_currently_trained_on():
    """The difficulty comes from the stage's own env_config.yaml, which train.py rewrites every round.

    That is what makes a chained run after the 2026-09-20 switch honest without any extra step: the level
    whose Brutal round has written that file plays Brutal, and one whose stage has not been re-run yet still
    plays the Violent settings its policy was actually trained under. `--difficulty` overrides both.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        model_dir = cwd / "models" / "spec_0-1"
        model_dir.mkdir(parents=True)
        trained = EnvConfig(mode="campaign", level=LEVEL, difficulty=4)
        (model_dir / "env_config.yaml").write_text(yaml.safe_dump(trained.to_dict()), encoding="utf-8")
        cfg = full_run.eval_config(LEVEL, model_dir / "best.zip", PLAN, cwd, port=47801)
        assert cfg.difficulty == 4, "the difficulty the policy is currently trained on"
        # The explicit override wins over all of it, in both directions.
        forced = full_run.eval_config(LEVEL, model_dir / "best.zip", PLAN, cwd, port=47801, difficulty=3)
        assert forced.difficulty == 3
        assert full_run.build_parser().parse_args([]).difficulty is None, "default: the config's"
        assert full_run.build_parser().parse_args(["--difficulty", "4"]).difficulty == 4

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
    """A policy that always walks forward, with `PPO.predict`'s signature.

    It also records the action mode it was asked for, which is what pins the 2026-09-20 fix in place."""

    def __init__(self):
        self.calls = 0
        self.modes: list[bool] = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls += 1
        self.modes.append(deterministic)
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
        assert result.difficulty == 4, "the difficulty the game actually read, from the campaign block"


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


def test_play_level_samples_actions_the_way_the_specialists_were_trained():
    """Measured 2026-09-20: 0-1's promoted specialist completes 19/20 sampled and 0/5 with argmax, because it
    is trained and promoted on sampled actions with ~7 nats of entropy in its heads. So the chain samples."""
    with tempfile.TemporaryDirectory() as tmp:
        for asked, expected in ((None, False), (False, False), (True, True)):
            env = fake_env(Path(tmp))
            policy = WalkForward()
            try:
                kwargs = {} if asked is None else {"deterministic": asked}
                full_run.play_level(LEVEL, env, policy, **kwargs)
            finally:
                env.close()
            assert policy.modes and set(policy.modes) == {expected}, \
                "every decision of the episode is made in the mode the chain chose"


def test_the_action_flags_parse_and_cannot_both_be_given():
    parser = full_run.build_parser()
    a = parser.parse_args([])
    assert a.deterministic is False and a.stochastic is False
    assert full_run.resolve_deterministic("campaign", deterministic=a.deterministic,
                                          stochastic=a.stochastic) is False, "sampling is the default"
    opted_in = parser.parse_args(["--deterministic"])
    assert full_run.resolve_deterministic("campaign", deterministic=opted_in.deterministic,
                                          stochastic=opted_in.stochastic) is True
    # --stochastic is kept as a no-op alias so the documented command still runs.
    alias = parser.parse_args(["--levels", "Level 0-1", "--stochastic", "--episodes", "2"])
    assert full_run.resolve_deterministic("campaign", deterministic=alias.deterministic,
                                          stochastic=alias.stochastic) is False
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--deterministic", "--stochastic"])
        except SystemExit:
            pass
        else:
            raise AssertionError("asking for both modes at once must be refused, not silently resolved")


def test_the_posted_row_says_which_action_mode_played_the_chain():
    with tempfile.TemporaryDirectory() as tmp:
        times = fixture_times(Path(tmp), held_row(level="Level 0-1", seconds=183.628, difficulty=3))
        results = [LevelResult(level="Level 0-1", completed=True, seconds=1.25, rank="P", kills=40, deaths=0,
                               difficulty=3)]
        assert full_run.post_results(times, results, gen="specialists@2026-09-20") == \
            ["Level 0-1 00:01.250 rank P"]
        assert "one specialist per level (sampled actions)" in times.read_text(encoding="utf-8")
        full_run.post_results(times, [LevelResult(level="Level 0-1", completed=True, seconds=1.2, rank="P",
                                                  difficulty=3)],
                              gen="specialists@2026-09-20", deterministic=True)
        assert "one specialist per level (deterministic (argmax) actions)" in times.read_text(encoding="utf-8")


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
