"""The shape guards of the S7 break: a checkpoint's layout read without torch, train-time refusal, the driver.

No game:  python tests/test_ckpt_layout.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_campaign_driver import harness  # noqa: E402
from transfer_weights import SpacesEnv  # noqa: E402

from ultrakill_ai import ckpt_layout, spaces  # noqa: E402
from ultrakill_ai.ckpt_layout import (  # noqa: E402
    CAMPAIGN_NVEC_V1,
    CAMPAIGN_NVEC_V2,
    LAYOUT_SHAPES,
    checkpoint_layout,
    checkpoint_shapes,
    resume_problem,
)
from ultrakill_ai.env import EnvConfig  # noqa: E402


def fake_ckpt(path: Path, width: int, nvec, steps: int = 10_000_000) -> Path:
    """Just the fields ckpt_layout reads, in SB3's own JSON shapes (nvec as numpy prints it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"observation_space": {":type:": "<class 'gymnasium.spaces.box.Box'>", "_shape": [width]},
            "action_space": {":type:": "<class 'gymnasium.spaces.multi_discrete.MultiDiscrete'>",
                             "nvec": "[" + " ".join("%2d" % v for v in nvec) + "]", "_shape": [len(nvec)]},
            "num_timesteps": steps}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps(data))
    return path


def tiny(obs_space, actions) -> PPO:
    torch.manual_seed(0)
    return PPO("MlpPolicy", SpacesEnv(obs_space, actions), policy_kwargs={"net_arch": [8]}, device="cpu")


def test_the_layout_table_matches_spaces():
    assert ckpt_layout.TECH_LAYOUTS == spaces.TECH_LAYOUTS
    assert LAYOUT_SHAPES["v1"] == (spaces.ObsLayout(campaign=True).size, tuple(int(v) for v in spaces.ACTION_NVEC_CAMPAIGN))
    assert LAYOUT_SHAPES["v2"] == (spaces.ObsLayout(campaign=True, tech=True).size,
                                   tuple(int(v) for v in spaces.ACTION_NVEC_TECH))


def test_ckpt_layout_imports_nothing_heavy():
    """The driver imports it and polls beside twelve games: numpy alone is ~785 MB of commit on this box."""
    code = ("import sys; sys.path.insert(0, %r); import ultrakill_ai.ckpt_layout; "
            "print(sorted(m for m in ('numpy', 'torch', 'gymnasium', 'stable_baselines3') if m in sys.modules))"
            % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_a_real_sb3_zip_reads_back_its_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        cases = (("v1", spaces.ObsLayout(campaign=True), spaces.action_space(campaign=True)),
                 ("v2", spaces.ObsLayout(campaign=True, tech=True), spaces.action_space(campaign=True, tech=True)),
                 (None, spaces.ObsLayout(), spaces.action_space()))
        for want, layout, actions in cases:
            path = Path(tmp) / f"{want}.zip"
            tiny(layout.space(), actions).save(path)
            assert checkpoint_shapes(path) == (layout.size, tuple(int(v) for v in actions.nvec))
            assert checkpoint_layout(path) == want


def test_resume_problem_names_the_migration_only_for_v1_under_v2():
    with tempfile.TemporaryDirectory() as tmp:
        v1 = fake_ckpt(Path(tmp) / "ckpt_1_steps.zip", 479, CAMPAIGN_NVEC_V1)
        v2 = fake_ckpt(Path(tmp) / "tech_init.zip", 530, CAMPAIGN_NVEC_V2)
        assert resume_problem(v1, "v1") is None and resume_problem(v2, "v2") is None
        text = resume_problem(v1, "v2")
        assert "add_tech_heads.py" in text and "479" in text and "530" in text
        back = resume_problem(v2, "v1")
        assert back and "add_tech_heads" not in back and "not a checkpoint for this config" in back
        assert "unknown tech_layout" in resume_problem(v1, "v9")


def test_an_unreadable_zip_is_left_to_sb3():
    with tempfile.TemporaryDirectory() as tmp:
        junk = Path(tmp) / "latest.zip"
        junk.write_bytes(b"shared")
        assert checkpoint_shapes(junk) is None and resume_problem(junk, "v2") is None
        assert resume_problem(Path(tmp) / "missing.zip", "v2") is None


def test_resume_refusal_covers_only_campaign_resumes():
    from ultrakill_ai.training import resume_refusal  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        v1 = fake_ckpt(Path(tmp) / "old.zip", 479, CAMPAIGN_NVEC_V1)
        v2cfg = EnvConfig(mode="campaign", tech_layout="v2")
        assert "add_tech_heads.py" in resume_refusal(str(v1), v2cfg)
        assert resume_refusal(None, v2cfg) is None, "a fresh start has nothing to check"
        assert resume_refusal(str(v1), EnvConfig(mode="cybergrind")) is None
        assert resume_refusal(str(v1), EnvConfig(mode="campaign")) is None


def test_training_main_refuses_before_touching_a_game_or_writing_a_file():
    from ultrakill_ai import training  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "cfg.yaml").write_text(yaml.safe_dump({"env": {"mode": "campaign", "tech_layout": "v2"},
                                                      "train": {"run_name": "t"}}), encoding="utf-8")
        fake_ckpt(tmp / "old.zip", 479, CAMPAIGN_NVEC_V1)
        argv, cwd = sys.argv, os.getcwd()
        sys.argv = ["train.py", "--config", str(tmp / "cfg.yaml"), "--resume", str(tmp / "old.zip")]
        os.chdir(tmp)
        try:
            training.main()
        except SystemExit as exc:
            assert exc.code == 3
        else:
            raise AssertionError("a 479-wide checkpoint was resumed under tech_layout v2")
        finally:
            sys.argv = argv
            os.chdir(cwd)
        assert not (tmp / "models").exists() and not (tmp / "runs").exists(), "refused before writing anything"


def test_the_driver_will_not_start_a_trainer_on_a_mismatched_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        generated = h.tmp / "configs" / "generated" / "spec_0-1.yaml"
        data = yaml.safe_load(generated.read_text(encoding="utf-8"))
        data["env"]["tech_layout"] = "v2"
        generated.write_text(yaml.safe_dump(data), encoding="utf-8")
        latest = h.tmp / "models" / "spec_0-1" / "latest.zip"
        fake_ckpt(latest, 479, CAMPAIGN_NVEC_V1)
        assert h.driver.tick() == "layout_mismatch"
        assert h.trainer_commands == [], "a trainer that would refuse the file is never spawned"
        fake_ckpt(latest, 530, CAMPAIGN_NVEC_V2)  # the migration, done by hand
        assert h.driver.tick() == "started"
        assert len(h.trainer_commands) == 1


def test_a_v1_stage_on_todays_files_still_starts():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        h.driver.begin_stage("Level 0-1", h.init)
        fake_ckpt(h.tmp / "models" / "spec_0-1" / "latest.zip", 479, CAMPAIGN_NVEC_V1)
        assert h.driver.tick() == "started"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
