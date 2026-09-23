"""The plan's three SPEED-ONLY override blocks: `speed.rewards:`, `speed.train:` and `speed.env:`.

No game needed:  python tests/test_speed_overrides.py   (or pytest)

Stages S1/S2 (gamma) and S3 (`level_complete`) of docs/superpowers/specs/2026-09-20-speedrun-tech.md, plus
the path stage S5's sticky slot reaches a config by. THE SHIPPED PLAN SETS NONE OF THEM except the `death:
12.0` that landed on 2026-09-20, and the first test in this file is the pin that says so: every lever here is
DORMANT, and turning one on is a line in `configs/specialists.yaml`, never a code change.

The three blocks share one rule, and it is the rule these tests exist to hold:

    ABSENT MEANS THE GENERATED CONFIG IS BYTE FOR BYTE WHAT IT WAS, for BOTH stage kinds.

and one hazard, which has bitten this file's subject once already (see the comment at the merge in
`campaign_driver.stage_config`): the plan's own nested dicts are SHARED with the shallow copy a stage config
is built from, so an in-place update would leak a speed stage's values into every complete stage generated
afterwards from the same Plan object -- and would not be caught by comparing the two, because both names
would point at the one mutated dict. Every merge rebinds; `test_..._order_independence` is the pin.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_driver  # noqa: E402
import yaml  # noqa: E402

from ultrakill_ai.env import EnvConfig  # noqa: E402
from ultrakill_ai.rewards import SPEED_BONUS_MAX, SPEED_BONUS_MIN, RewardConfig, completion_bonus  # noqa: E402

PLAN = ROOT / "configs" / "specialists.yaml"
SPEED = campaign_driver.SPEED
COMPLETE = campaign_driver.COMPLETE


def plan():
    return campaign_driver.load_plan(PLAN)


def write_plan(tmp_path: Path, **extra_speed) -> Path:
    """The shipped plan with extra keys under `speed:`, so every other number stays the real one."""
    data = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    data.setdefault("speed", {}).update(extra_speed)
    out = tmp_path / "plan.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------------------
# Dormancy
# ---------------------------------------------------------------------------------------------------------


def test_the_shipped_plan_ships_no_train_or_env_override():
    """NO `speed.train:` BLOCK SHIPS. S1 was tried 2026-09-21 and REVERTED the same day; S2 is cancelled.

    S1 (gamma 0.999 / gae_lambda 0.98) ran live on spec_0-1_speed round 6 for 2.66M steps and made the policy
    slower at every quantile with a healthy critic -- bucket median 117 s -> 170 s (docs/project-log.md,
    2026-09-21). The weights were rolled back to the switch point and the block was removed, so a SPEED stage
    trains at exactly the same hyperparameters as a complete one again.

    The failure this guards against is now the OPPOSITE of what it guarded before: a `speed.train:` block
    reappearing in the shipped plan -- by a bad merge, a revert of the revert, or someone switching S2 on
    without the fresh argument the log says it needs -- and silently costing another round of twelve games.
    """
    p = plan()
    assert p.speed_train == {}, \
        "no speed.train block ships: S1 was reverted 2026-09-21 and S2 (gamma 0.9995) is CANCELLED"
    assert p.speed_env == {}, "S5 is not switched on: `speed.env:` is absent from the shipped plan"
    assert set(p.speed_rewards) == {"death", "oob"}, \
        "the 2026-09-20 death weight and the 2026-09-22 oob weight, and no other reward lever"


def test_a_complete_stage_and_a_speed_stage_now_train_at_the_same_hyperparameters():
    """`env` still gains only what the speed rule has always added, and `train` gains nothing at all."""
    p = plan()
    complete = campaign_driver.stage_config(p, "Level 0-1")["train"]
    speed = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]
    assert complete["hyperparams"] == p.train["hyperparams"], COMPLETE
    assert complete["hyperparams"]["gamma"] == 0.998 and complete["hyperparams"]["gae_lambda"] == 0.95
    assert speed["hyperparams"]["gamma"] == 0.998 and speed["hyperparams"]["gae_lambda"] == 0.95
    moved = {k for k in speed["hyperparams"] if speed["hyperparams"][k] != complete["hyperparams"].get(k)}
    assert moved == set(), moved
    complete_env = campaign_driver.stage_config(p, "Level 0-1")["env"]
    speed_env = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["env"]
    assert set(speed_env) - set(complete_env) == {"speed_bonus", "speed_target_scale"}


# ---------------------------------------------------------------------------------------------------------
# S1 / S2: `speed.train:`
# ---------------------------------------------------------------------------------------------------------


def test_the_train_block_overrides_the_hyperparameters_of_a_speed_stage_only(tmp_path=None):
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    path = write_plan(tmp_path, train={"gamma": 0.999, "gae_lambda": 0.98})
    p = campaign_driver.load_plan(path)
    assert p.speed_train == {"gamma": 0.999, "gae_lambda": 0.98}

    speed = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]
    complete = campaign_driver.stage_config(p, "Level 0-1")["train"]
    assert speed["hyperparams"]["gamma"] == 0.999 and speed["hyperparams"]["gae_lambda"] == 0.98
    assert complete["hyperparams"] == plan().train["hyperparams"], "a complete stage keeps 0.998 / 0.95"
    # ... and only those two moved: every other hyperparameter is the plan's.
    moved = {k for k in speed["hyperparams"] if speed["hyperparams"][k] != complete["hyperparams"].get(k)}
    assert moved == {"gamma", "gae_lambda"}
    # The two-step ladder the lead approved: 0.999 first, then 0.9995, each with gae_lambda 0.98.
    second = campaign_driver.load_plan(write_plan(tmp_path, train={"gamma": 0.9995, "gae_lambda": 0.98}))
    assert campaign_driver.stage_config(second, "Level 0-1", kind=SPEED)["train"]["hyperparams"]["gamma"] == 0.9995


def test_the_train_block_does_not_leak_between_stage_kinds_in_either_order(tmp_path=None):
    """The shallow-copy hazard, pinned in both orders: neither generation may change the Plan object."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    p = campaign_driver.load_plan(write_plan(tmp_path, train={"gamma": 0.999}))
    before = dict(p.train["hyperparams"])

    speed_first = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]["hyperparams"]
    then_complete = campaign_driver.stage_config(p, "Level 0-2")["train"]["hyperparams"]
    assert speed_first["gamma"] == 0.999 and then_complete["gamma"] == before["gamma"] == 0.998
    assert p.train["hyperparams"] == before, "generating a stage config mutated the plan"

    q = campaign_driver.load_plan(write_plan(tmp_path, train={"gamma": 0.999}))
    complete_first = campaign_driver.stage_config(q, "Level 0-2")["train"]["hyperparams"]
    then_speed = campaign_driver.stage_config(q, "Level 0-1", kind=SPEED)["train"]["hyperparams"]
    assert complete_first["gamma"] == 0.998 and then_speed["gamma"] == 0.999


def test_a_misspelt_or_non_numeric_train_key_is_refused_not_defaulted(tmp_path=None):
    """A plan that silently trains at the old gamma while the file says 0.999 costs a round of twelve games."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    for block, needle in (({"gama": 0.999}, "unknown speed.train"),
                          ({"discount": 0.999}, "unknown speed.train"),
                          ({"gamma": "0.999"}, "must be a number"),
                          ({"gamma": True}, "must be a number")):
        try:
            campaign_driver.load_plan(write_plan(tmp_path, train=block))
        except ValueError as exc:
            assert needle in str(exc), (block, exc)
        else:
            raise AssertionError("accepted %r" % (block,))


def test_the_train_allowlist_is_ppos_own_numeric_hyperparameters():
    assert set(campaign_driver.SPEED_TRAIN_KEYS) >= {"gamma", "gae_lambda", "learning_rate", "ent_coef",
                                                     "target_kl", "n_steps", "batch_size", "n_epochs"}
    # Every key the shipped plan already sets must be overridable, or S1 could not be written for this run.
    assert set(plan().train["hyperparams"]) <= set(campaign_driver.SPEED_TRAIN_KEYS)


def test_an_int_hyperparameter_stays_an_int(tmp_path=None):
    """`n_epochs: 5.0` would reach PPO as a float and `range(5.0)` raises; the block must not coerce."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    p = campaign_driver.load_plan(write_plan(tmp_path, train={"n_epochs": 4, "n_steps": 4096}))
    hyper = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]["hyperparams"]
    assert hyper["n_epochs"] == 4 and isinstance(hyper["n_epochs"], int)
    assert hyper["n_steps"] == 4096 and isinstance(hyper["n_steps"], int)


# ---------------------------------------------------------------------------------------------------------
# S3: `speed.rewards:` already carries anything RewardConfig has
# ---------------------------------------------------------------------------------------------------------


def test_the_reward_block_can_already_carry_level_complete_novelty_and_punch(tmp_path=None):
    """S3. Nothing was missing: the block is validated against `RewardConfig`'s fields, and those are fields.

    So `level_complete: 300` (S3), and the `novelty`/`gate` halving S4 would need, are one line in the plan
    each -- for the SPEED stage only, which is what those stages ask for. NOT SET HERE: S3 is not approved to
    run in this workflow and S4 is not approved at all (an earlier measurement says cutting gate pay lowers
    the speed gradient, and the lead's ruling is that it needs its own evidence after S1-S3).
    """
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    weights = {"death": 12.0, "level_complete": 300.0, "novelty": 0.1, "punch": 0.0}
    p = campaign_driver.load_plan(write_plan(tmp_path, rewards=weights))
    assert p.speed_rewards == weights

    speed = EnvConfig.from_dict(campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["env"]).rewards
    complete = EnvConfig.from_dict(campaign_driver.stage_config(p, "Level 0-1")["env"]).rewards
    assert (speed.level_complete, speed.novelty, speed.punch, speed.death) == (300.0, 0.1, 0.0, 12.0)
    assert (complete.level_complete, complete.novelty, complete.punch, complete.death) == (100.0, 0.2, 0.01, 5.0)
    # And nothing else moved.
    assert dataclasses.replace(speed, level_complete=complete.level_complete, novelty=complete.novelty,
                               punch=complete.punch, death=complete.death) == complete


def test_the_completion_bonus_floor_and_ceiling_scale_with_level_complete():
    """S3's other half: the speed band is a MULTIPLE of `level_complete`, so raising it raises the whole band.

    `completion_bonus` returns `cfg.level_complete * scale`, so at 300 the band is 75..600 where at 100 it is
    25..200 -- which is why §4.4 requires S3 to land BEFORE the observation break: it is the one change that
    moves the value scale, and the value head has to re-fit before it is copied.
    """
    for weight in (100.0, 300.0):
        cfg = RewardConfig(level_complete=weight)
        target = 120.0
        assert completion_bonus(cfg) == weight, "no target: not a speed stage, nothing scales"
        assert completion_bonus(cfg, target, None) == weight * SPEED_BONUS_MIN, "no usable clock: the floor"
        assert completion_bonus(cfg, target, target) == weight, "exactly on target pays the plain weight"
        # Half the target is already at the ceiling (`min(ratio, 2.0)`); 1.0 s or less is not a time at all
        # and reads as MISSING through `times.valid_official_seconds`, so the ceiling is probed at 20 s.
        assert completion_bonus(cfg, target, target / 2) == weight * SPEED_BONUS_MAX, "the ceiling"
        assert completion_bonus(cfg, target, 20.0) == weight * SPEED_BONUS_MAX, "and it is a clip, not a slope"
        # Strictly decreasing above the target, and strictly above the floor at any finite time.
        slow = [completion_bonus(cfg, target, t) for t in (150.0, 300.0, 600.0, 10_000.0)]
        assert slow == sorted(slow, reverse=True)
        assert all(v > weight * SPEED_BONUS_MIN for v in slow)
    # The floor at 300 is 75.0, which is the number §4.5's farmability table quotes against the 10.0 tech cap.
    assert completion_bonus(RewardConfig(level_complete=300.0), 120.0, None) == 75.0


# ---------------------------------------------------------------------------------------------------------
# S5: `speed.env:`
# ---------------------------------------------------------------------------------------------------------


def test_the_env_block_overrides_env_settings_of_a_speed_stage_only(tmp_path=None):
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    p = campaign_driver.load_plan(write_plan(tmp_path, env={"sticky_weapon_slot": True,
                                                            "sticky_slot_switch_every": 4}))
    speed = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["env"]
    complete = campaign_driver.stage_config(p, "Level 0-1")["env"]
    assert EnvConfig.from_dict(speed).sticky_weapon_slot is True
    assert "sticky_weapon_slot" not in complete
    assert complete == campaign_driver.stage_config(plan(), "Level 0-1")["env"]


def test_the_env_block_may_not_overrule_what_the_speed_rule_owns(tmp_path=None):
    """`fresh_start_prob: 1.0` and the bonus switch are the speed rule, not a preference (2026-09-18 review)."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    for block in ({"fresh_start_prob": 0.2}, {"speed_bonus": False}, {"level": "Level 0-2"},
                  {"speed_target_seconds": 42.0}, {"rewards": {"death": 1.0}}, {"levels": ["Level 0-1"]}):
        try:
            campaign_driver.load_plan(write_plan(tmp_path, env=block))
        except ValueError as exc:
            assert "may not set" in str(exc), (block, exc)
        else:
            raise AssertionError("accepted %r" % (block,))


def test_a_misspelt_env_key_is_refused_not_defaulted(tmp_path=None):
    """`EnvConfig.from_dict` DROPS a key it does not know, so a typo here would train at the default."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    try:
        campaign_driver.load_plan(write_plan(tmp_path, env={"sticky_weapon_slott": True}))
    except ValueError as exc:
        assert "unknown speed.env" in str(exc)
    else:
        raise AssertionError("a misspelt env key was accepted")


def test_an_env_value_of_the_wrong_TYPE_is_refused_not_passed_through(tmp_path=None):
    """The name check is not enough: `EnvConfig.from_dict` does not coerce, so a quoted YAML boolean is truthy.

    `sticky_weapon_slot: "false"` would turn the lever ON for the next round while the plan file read off --
    the silent-wrong-value failure `speed.train`'s number check already refuses -- and
    `sticky_slot_switch_every: "every 3"` would not fail here at all: it raises inside a SubprocVecEnv worker
    on the first honoured switch, mid-episode, and the driver restarts the trainer into the same config.
    """
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    bad = [({"sticky_weapon_slot": "false"}, "boolean"), ({"sticky_weapon_slot": 1}, "boolean"),
           ({"sticky_slot_switch_every": "every 3"}, "whole number"),
           ({"sticky_slot_switch_every": True}, "whole number"),
           ({"sticky_slot_switch_every": 2.5}, "whole number"),
           ({"stuck_seconds": "lots"}, "number"), ({"mode": 4}, "string")]
    for block, wanted in bad:
        try:
            campaign_driver.load_plan(write_plan(tmp_path, env=block))
        except ValueError as exc:
            assert wanted in str(exc), (block, exc)
        else:
            raise AssertionError("accepted %r" % (block,))
    # ... and the right types still load, including an int where a float is declared.
    ok = campaign_driver.load_plan(write_plan(tmp_path, env={"sticky_weapon_slot": True,
                                                             "sticky_slot_switch_every": 4,
                                                             "stuck_seconds": 12}))
    assert ok.speed_env == {"sticky_weapon_slot": True, "sticky_slot_switch_every": 4, "stuck_seconds": 12}


def test_a_plan_with_speed_rewards_and_no_shared_rewards_block_still_generates(tmp_path=None):
    """It used to raise KeyError: `env["rewards"]` was indexed, not `.get`. A legitimate file, so it must work."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    data = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    data["env"].pop("rewards")
    data["speed"]["rewards"] = {"death": 12.0}
    path = tmp_path / "no_shared_rewards.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    p = campaign_driver.load_plan(path)
    assert campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["env"]["rewards"] == {"death": 12.0}
    assert "rewards" not in campaign_driver.stage_config(p, "Level 0-1")["env"]


if __name__ == "__main__":
    import tempfile

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                fn(Path(tmp))
            else:
                fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
