"""The shape guards of the S7 break: a checkpoint's layout read without torch, train-time refusal, the driver.

No game:  python tests/test_ckpt_layout.py
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
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


# -- review follow-up: unreadable zips of every kind, the log-once key, no games launched for a mismatch ------


def _patch_entry(path: Path, *, flags: int | None = None, method: int | None = None, data_byte: int | None = None):
    """Rewrites the one member's header fields in place: `flags` / `method` in BOTH headers (the local file header
    at offset 0 and the central directory entry), or the first byte of its compressed data."""
    raw = bytearray(path.read_bytes())
    central = raw.index(b"PK\x01\x02")
    for header, flag_at, method_at in ((0, 6, 8), (central, 8, 10)):
        if flags is not None:
            struct.pack_into("<H", raw, header + flag_at, flags)
        if method is not None:
            struct.pack_into("<H", raw, header + method_at, method)
    if data_byte is not None:
        name_len, extra_len = struct.unpack_from("<HH", raw, 26)
        raw[30 + name_len + extra_len] = data_byte
    path.write_bytes(bytes(raw))
    return path


def _unreadable_zips(tmp: Path) -> dict[str, tuple[Path, type]]:
    """One zip per failure `checkpoint_shapes` once let escape, with the exception zipfile/int really raise."""
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    out["encrypted"] = (_patch_entry(fake_ckpt(tmp / "encrypted.zip", 479, CAMPAIGN_NVEC_V1), flags=0x1),
                        RuntimeError)
    out["method"] = (_patch_entry(fake_ckpt(tmp / "method.zip", 479, CAMPAIGN_NVEC_V1), method=99),
                     NotImplementedError)
    deflated = tmp / "deflate.zip"
    with zipfile.ZipFile(deflated, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("data", json.dumps({"observation_space": {"_shape": [479]}}) * 20)
    out["deflate"] = (_patch_entry(deflated, data_byte=0xFF), zlib.error)  # BTYPE 11: an invalid block type
    infinite = tmp / "infinite.zip"
    with zipfile.ZipFile(infinite, "w") as z:
        z.writestr("data", '{"observation_space": {"_shape": [Infinity]}, "action_space": {"nvec": "[3 3]"}}')
    out["infinity"] = (infinite, OverflowError)
    return out


def test_every_kind_of_unreadable_zip_is_left_to_sb3():
    with tempfile.TemporaryDirectory() as tmp:
        for kind, (path, raised) in _unreadable_zips(Path(tmp)).items():
            try:  # the fixture really is the failure it is named for
                with zipfile.ZipFile(path) as z:
                    data = json.loads(z.read("data").decode("utf-8"))
                int(data["observation_space"]["_shape"][0])
            except raised:
                pass
            else:
                raise AssertionError("%s: expected %s" % (kind, raised.__name__))
            assert checkpoint_shapes(path) is None, kind
            assert checkpoint_layout(path) is None and resume_problem(path, "v2") is None, kind


def test_an_unreadable_resume_file_does_not_stall_the_driver():
    """Before, the exception escaped `ensure_trainer` into "tick failed; continuing" every poll, no trainer started."""
    for kind in ("encrypted", "method", "deflate", "infinity"):
        with tempfile.TemporaryDirectory() as tmp:
            h = harness(tmp)
            h.driver.begin_stage("Level 0-1", h.init)
            path, _ = _unreadable_zips(h.tmp / "fixtures")[kind]
            latest = h.tmp / "models" / "spec_0-1" / "latest.zip"
            latest.write_bytes(path.read_bytes())
            assert h.driver.tick() == "started", kind
            assert len(h.trainer_commands) == 1, kind


def _v2_stage(h) -> Path:
    h.driver.begin_stage("Level 0-1", h.init)
    generated = h.tmp / "configs" / "generated" / "spec_0-1.yaml"
    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    data["env"]["tech_layout"] = "v2"
    generated.write_text(yaml.safe_dump(data), encoding="utf-8")
    return h.tmp / "models" / "spec_0-1"


def test_a_mismatch_is_logged_once_per_file_and_again_for_a_new_wrong_file():
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        model_dir = _v2_stage(h)
        fake_ckpt(model_dir / "latest.zip", 479, CAMPAIGN_NVEC_V1)
        log = h.tmp / "runs" / "specialists_driver.log"
        for _ in range(3):
            assert h.driver.tick() == "layout_mismatch"
        assert log.read_text(encoding="utf-8").count("LAYOUT MISMATCH") == 1, "once per (run, file), not per poll"
        # A botched migration: a newer file, still not v2 (530 inputs, the v1 action head).
        fake_ckpt(model_dir / "ckpt_20000000_steps.zip", 530, CAMPAIGN_NVEC_V1)
        for _ in range(2):
            assert h.driver.tick() == "layout_mismatch"
        text = log.read_text(encoding="utf-8")
        assert text.count("LAYOUT MISMATCH") == 2 and "ckpt_20000000_steps.zip takes 530 inputs" in text
        assert h.trainer_commands == []


def test_a_mismatch_launches_no_games():
    """The resume file and its layout are read BEFORE `ensure_games`: twelve games launched to idle is waste."""
    with tempfile.TemporaryDirectory() as tmp:
        h = harness(tmp)
        model_dir = _v2_stage(h)
        fake_ckpt(model_dir / "latest.zip", 479, CAMPAIGN_NVEC_V1)
        h.ports = {}  # nothing listening: a start would launch all twelve
        assert h.driver.tick() == "layout_mismatch"
        assert h.launched == [] and h.spawned == []
        fake_ckpt(model_dir / "latest.zip", 530, CAMPAIGN_NVEC_V2)
        assert h.driver.tick() == "started" and h.launched == [(12, 1)]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
