"""Which observation / action LAYOUT a saved SB3 checkpoint was built for -- read without torch or numpy.

The S7 tech break (docs/superpowers/specs/2026-09-20-speedrun-tech.md §4.4, "Integration hazard") makes two
campaign layouts coexist: v1 = 479 inputs and 12 action dims / 45 logits, v2 = 530 inputs and 15 / 57. The
driver's round init picks a stage's newest checkpoint and would happily hand a 479-wide `ckpt_*_steps.zip` to a v2
config, then re-exec train.py in a loop across twelve games on a commit-bound box. `training.main` and
`campaign_driver.Driver.ensure_trainer` both ask `resume_problem` first.

STDLIB ONLY: the driver imports this and polls beside twelve games; numpy alone is ~785 MB of commit on this box.
SB3 keeps the spaces as JSON in the zip's `data` member (`observation_space._shape`, and `action_space.nvec` as
the string numpy prints), so ~120 KB is read per call and never the weights.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

TECH_LAYOUTS = ("v1", "v2")  # the same literal as ultrakill_ai.spaces.TECH_LAYOUTS (tests/test_ckpt_layout.py)
CAMPAIGN_NVEC_V1 = (3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3)
CAMPAIGN_NVEC_V2 = CAMPAIGN_NVEC_V1 + (6, 4, 2)
LAYOUT_SHAPES = {"v1": (479, CAMPAIGN_NVEC_V1), "v2": (530, CAMPAIGN_NVEC_V2)}
MIGRATION_HINT = ("migrate it with `python scripts/add_tech_heads.py <this zip> <dest zip>` and move the 479-wide "
                  "files aside with `python scripts/quarantine_pre_tech.py <model dir> --apply` "
                  "(docs/commands.md, 'S7 -- the one break')")


def checkpoint_shapes(path) -> tuple[int, tuple[int, ...]] | None:
    """(observation width, action nvec) out of an SB3 zip, or None when it cannot be read.

    EVERY failure is None, not a named list of them: zipfile alone raises RuntimeError (an encrypted member),
    NotImplementedError (an unknown compression method) and zlib.error (a corrupt deflate stream), and
    `int(float("inf"))` is an OverflowError. Anything that escaped here would escape the driver's `ensure_trainer`
    into "tick failed; continuing" every poll with no trainer started -- a silent stall on the v1 path. None hands
    the file to SB3's own load, exactly as before this guard existed.
    """
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data").decode("utf-8"))
        width = int(data["observation_space"]["_shape"][0])
        nvec = tuple(int(v) for v in re.findall(r"-?\d+", str(data["action_space"]["nvec"])))
    except Exception:  # noqa: BLE001 - see the docstring: an unreadable checkpoint is SB3's to judge
        return None
    return (width, nvec) if nvec else None


def layout_of(shapes) -> str | None:
    for name, expected in LAYOUT_SHAPES.items():
        if shapes == expected:
            return name
    return None


def checkpoint_layout(path) -> str | None:
    """"v1", "v2", or None (unreadable, or not a campaign layout -- Cyber Grind's 448 / 11 is None)."""
    shapes = checkpoint_shapes(path)
    return layout_of(shapes) if shapes is not None else None


def resume_problem(path, tech_layout: str) -> str | None:
    """Why the checkpoint at `path` cannot be trained under `tech_layout`, or None when it can.

    An unreadable zip is None, deliberately: SB3's own load decides then, exactly as before this guard existed,
    and a half-written file is already `supervise.choose_resume`'s concern.
    """
    shapes = checkpoint_shapes(path)
    if shapes is None:
        return None
    want = LAYOUT_SHAPES.get(tech_layout)
    if want is None:
        return f"unknown tech_layout {tech_layout!r} (expected one of {list(TECH_LAYOUTS)})"
    if shapes == want:
        return None
    text = (f"{Path(path).name} takes {shapes[0]} inputs and {len(shapes[1])} action dims ({sum(shapes[1])} "
            f"logits), but tech_layout {tech_layout!r} needs {want[0]} and {len(want[1])} ({sum(want[1])})")
    if layout_of(shapes) == "v1" and tech_layout == "v2":
        return text + ": " + MIGRATION_HINT
    return text + ": this is not a checkpoint for this config"
