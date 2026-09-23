# S7 "The One Break" — Python Side — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test — with no game, and without disturbing the live `spec_0-1_speed` run — everything
Python needs for stage S7 of `docs/superpowers/specs/2026-09-20-speedrun-tech.md` (observation 479 → 530,
action 12 dims / 45 logits → 15 / 57, the jump-SSJ macro on the wire, `add_tech_heads.py`, the shape guards),
the dormant S8 pieces (`tech` bonus, `ent_floor` re-base), and the S6 re-measure and S7 install runbooks.

**Architecture:** One env switch, `tech_layout: "v1" | "v2"`. `v1` is the default and is today's env byte for
byte (pinned by hashes before any code moves). `v2` appends a 51-float TECH block and three action dims; every
head the game must not execute yet is masked in Python, refused by the mod through config the env sends on every
connect, and its logits are pinned (gradient-masked) in training. The weight migration is a pure surgery (zero
columns, zero rows + prior biases) proven exact at the tensor level and by a measured KL of 0.0. Shape guards in
`training.main` and in the driver make a 479-wide checkpoint impossible to resume under `v2`.

**Tech Stack:** Python 3 (the repo's `python/.venv`), stable-baselines3 2.9.0, torch 2.14 (CPU),
gymnasium 1.3, numpy, PyYAML. Anything the driver imports stays stdlib-only. Tests are standalone files
(`python tests/test_x.py` prints `N tests passed`), no pytest required.

---

## Ground rules (read before Task 0)

1. **The live run is untouchable.** Twelve games on 47800-47811 train `spec_0-1_speed` under
   `campaign_driver.py`. Never open a socket to 47800-47811; never run `games.py launch` or `games.py stop`;
   never edit `python/configs/specialists.yaml` or `python/configs/generated/*` in the MAIN tree; never move,
   copy over or delete anything under `python/models/` (until the S7 runbook's pause). Reading files is fine.
2. **Why v1 must be byte-identical, and why that is pinned FIRST.** The driver re-execs `train.py` at every
   round boundary and after every crash, and each new trainer imports whatever `python/ultrakill_ai/` holds at
   that moment in `F:\Github\ULTRAKILL-AI`. Every commit this plan lands on `main` therefore reaches the live
   fleet at its next restart. Every task must leave the v1 path exactly as it is today; Task 1's hash pins and
   Tasks 2-3's key-set pins are what prove it.
3. **Work in a worktree, land on main only green.** Editing `env.py` in the main tree while a trainer might
   restart would import a half-edited file into twelve workers. Task 0 creates `F:\Github\ULTRAKILL-AI-tech` on
   branch `tech-break-python`; every edit happens there, and each finished, green task is fast-forwarded into
   `main` (Task 0, step 4). CLAUDE.md's worktree gotcha applies: **set `PYTHONPATH` to the worktree's
   `python/`**, or the editable install silently tests the main tree.
4. **One thing at a time.** Each task is one reviewable unit. S7 (the break) and S8 (the `tech` bonus) are two
   different live changes, never switched on together; S8 waits >= 400k steps after S7 (spec §6).
5. **Every task ends green on the full no-game suite** (41 files today, ~3 min), from the worktree's `python/`:
   ```powershell
   $env:PYTHONPATH = "F:\Github\ULTRAKILL-AI-tech\python"
   Get-ChildItem tests\test_*.py | ForEach-Object { F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
   ```
   Below, `PY` means `F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python` run from
   `F:\Github\ULTRAKILL-AI-tech\python` with that `PYTHONPATH`.
6. **Git:** `git add <explicit paths>` only (never `-A`: the trainer rewrites `python/models/` continuously).
   Every commit message ends with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
   Push after every task. Never commit `*.dll`, `*.pdb` or `decompiled/`.
7. **Memory:** the box is commit-bound (~66-75% of a 60 GB commit limit with the fleet up). Every script here
   streams its inputs and stays well under ~4 GB. `add_tech_heads.py` reads ~120 KB per checkpoint (the `data`
   member only) plus two PPO models of ~12 MB each.
8. **"Test" in this plan always means a no-game test.** The in-game checks are the S6 runbook (Task 11), run by
   the lead when memory allows, on port 47812 only.

---

## The settled index maps

### Observation under `tech_layout: v2` = 530 floats

Indices 0-478 are the v1 vector, unchanged: player 0-16, 8 enemies x 50 = 17-416, horizontal rays 417-432,
ground rays 433-440, Cyber Grind pair 441-442, campaign block 443-478 (route target 448-455). The TECH block is
appended at `ObsLayout.tech_start` = 479.

| abs | block | offset | field (mod key) | packed as |
|---|---|---|---|---|
| 479 | A move | 0 | `heavy_fall` | 0/1 |
| 480 | A | 1 | `slam_force` | /10, clip [0, 1] (1 at slam start, +5/s; >= 5.5 is the 12.5x bounce) |
| 481 | A | 2 | `bounce_window` | 0/1 |
| 482 | A | 3 | `coyote` (`gc.sinceLastGrounded`, game s) | clip [0, 1] (no ground check sends 999 -> 1.0) |
| 483 | A | 4 | `wall_jumps` | /3, clip [0, 1] |
| 484 | A | 5 | `wall_available` | 0/1 |
| 485 | A | 6 | `boost` | 0/1 |
| 486 | A | 7 | `boost_left` (i-frames, 100 -> 0) | /100, clip [0, 1] |
| 487 | A | 8 | `pre_slide_speed` | /3, clip [0, 2] |
| 488 | A | 9 | `jump_cooldown` | 0/1 |
| 489 | A | 10 | **`slide_grace`** (NOT the spec's `slide_since`) | clip [0, 1] |
| 490 | A | 11 | `riding_rocket` | 0/1 |
| 491-500 | B weapon | 12-21 | reserved (S9) | 0.0 always |
| 501-518 | C own projectiles | 22-39 | reserved (S10) | 0.0 always |
| 519 | D macro | 40 | macro ran (`result == "ran"`) | 0/1 |
| 520 | D | 41 | macro not run (`refused` / `degraded` / `disabled`) | 0/1 |
| 521 | D | 42 | SSJ bucket | `bucket / 3` when ran AND `ssj_landed` AND 1 <= bucket <= 3, else 0.0 |
| 522-529 | E hazard rays | 43-50 | reserved (S11) | 0.0 always |

Every field is read with a 0.0 default, so an old DLL, a flag that is off, or a frame with no player packs the
exact vector the migration initialised the new columns for. Block D is all zeros on a step that sent no macro
(the mod reports a macro only when one was requested).

### Action under `tech_layout: v2` = 15 dims / 57 logits (append only)

| dim | name | n | logit rows | at S7 |
|---|---|---|---|---|
| 0 | move forward | 3 | 0-2 | live, unchanged |
| 1 | move side | 3 | 3-5 | live |
| 2-7 | jump, dash, slide, fire1, fire2, punch | 2 each | 6-17 | live |
| 8 | weapon slot | 6 | 18-23 | live |
| 9 | yaw | 11 | 24-34 | live |
| 10 | pitch | 7 | 35-41 | live |
| 11 | look mode | 3 | 42-44 | live |
| 12 | **macro** | 6 | 45 none (bias ln 45 = 3.8067), **46 ssj (LIVE)**, 47 ssj_wall, 48 core_nuke, 49 rocket_down, 50 coin_rocket | 1 live; 2-5 masked + pinned |
| 13 | **variant** | 4 | 51 keep (bias ln 17 = 2.8332), 52-54 variation 0/1/2 | whole dim masked + pinned |
| 14 | **hook** | 2 | 55 off (bias ln 9 = 2.1972), 56 on | whole dim masked + pinned |

New rows have zero weights, so at init P(macro none) = 0.90 (0.02 per macro), P(variant keep) = 0.85,
P(no hook) = 0.90, state-independent, and the new heads add exactly **1.3986 nats** (macro 0.4860, variant
0.5875, hook 0.3251). Pinned rows at the S7 gates: **47, 48, 49, 50, 51, 52, 53, 54, 55, 56**.

On the wire (v2): `"macro": 1` only when the policy chose 1 and `macro_ssj` is on (the default); `"variant"`
only when `variant_switching` is on; `"hook"` appended to `buttons` only when `hook_action` is on. Otherwise the
message is the v1 action message, key for key.

---

## Spec deviations and decisions, with the reason for each

1. **`slide_grace` at 489, not `slide_since`.** The 2026-09-20 mod review (finding 2) showed `slide_since` is
   real wall-clock seconds -- ~0.08 on the loaded fleet vs ~0.01 on one eval game. `docs/protocol.md` already
   says never pack it.
2. **Refused heads are masked in Python AND refused by the mod AND pinned in training.** The spec has the mod
   refuse macros 2-5 and `variant`. Masking in Python too keeps block D's "not run" bit meaning "an SSJ
   precondition failed" rather than "you sampled a reserved value". Pinning the logits (Task 7) stops the entropy
   bonus from diffusing heads that get no advantage signal: Adam moves a consistently-signed gradient ~`lr` per
   minibatch step whatever its size (~0.05 logit per rollout at 240 minibatch steps), so an unpinned `variant`
   head would drift toward uniform within a few million steps, the §4.4 prior would be gone by the S9 flip, and
   its free entropy would satisfy the entropy floor while the live dims collapsed.
3. **`hook` is masked at S7.** The spec lists `hook` (dim 14) at the break but says nothing about refusing it,
   and the mod cannot refuse it -- it is a plain HoldButton. Unmasked, the break would hold the whiplash on 10%
   of steps from the first rollout. It opens at S10 (`hook_action: true`).
4. **The KL measurement's observations come from the run's own checkpoints.** The spec names
   `runs/probe_0-1_record/stoch_*.jsonl`, but those are `probe_rollout.py` step records with no observation
   vector in them. Every SB3 checkpoint carries `_last_obs` (twelve real 479-float states; verified on
   `models/spec_0-1_speed/ckpt_19951846_steps.zip`), so `add_tech_heads.py` reads those, streamed, with a
   synthetic fallback. The tensor-level tests are the exact proof; the KL is the float-noise confirmation.
5. **`ent_floor` rides in `speed.train:`.** It is a top-level `train:` key, not a PPO hyperparameter, so
   `campaign_driver` gains a routing for `ent_floor` / `ent_coef_max` (Task 8). New floor = 6.5 + the entropy
   the heads add = **7.8986** (deterministic: the new heads are state-independent at init, so the measurement
   only confirms it).
6. **v2 is switched on through `speed.env:` only.** The focus runs 0-1's speed stage alone. 0-2 / 0-3
   checkpoints stay v1 until migrated; the driver refuses to start a mismatched stage (Task 5) rather than
   crash-looping twelve games.
7. **`best.zip` is migrated too**, so `keep_best.py`'s held peak and the next promotion are v2 files.
8. **The S8 decay is driven by the trainer.** `TechDecayCallback` pushes the scale into every env through
   `VecEnv.env_method` at each rollout start, from an absolute `rewards.tech_decay_start` step the operator writes
   at switch-on; per-env step counters reset at every trainer restart and cannot carry a 3M-step schedule.
9. **The review's finding-8 field diff (0.7.2 vs 0.8.0 obs streams) moves from the pause into S6**, so the
   15-25 min full pause does not also carry two private-game runs.
10. **A v1 env talking to a 0.8 DLL sends every 0.8 switch OFF explicitly.** Mod config is per game process, so
    a rollback to v1 (or an eval after a private test) would otherwise inherit whatever the last client turned
    on. Against a 0.7.2 DLL (no `features` in `hello`) nothing extra is sent: today's call, exactly.

---

## File structure

**Create**
- `python/ultrakill_ai/ckpt_layout.py` -- stdlib-only: the layout a checkpoint zip was built for, the resume
  refusal text, layout adoption for eval tools. The driver imports it, so no numpy / torch.
- `python/scripts/add_tech_heads.py` -- the §4.4 weight migration, measure-before-write.
- `python/scripts/quarantine_pre_tech.py` -- moves a model dir's v1 zips into `pre_tech/` (move, never delete).
- `python/tests/test_tech_layout.py` -- v1 pins; v2 packing, decoding and index maps.
- `python/tests/test_tech_env.py` -- handshake, config, wire encoding, counters, info, progress.
- `python/tests/test_add_tech_heads.py` -- the migration on tiny SB3 models.
- `python/tests/test_ckpt_layout.py` -- shape reading, train-time refusal, the driver guard, layout adoption.
- `python/tests/test_quarantine_pre_tech.py` -- the quarantine mover.
- `python/tests/test_tech_training.py` -- pinned rows, absolute entropy dims.
- `python/tests/test_tech_bonus.py` -- the S8 reward, gate, cap, decay schedule and callback.
- `python/tests/test_probe_tech.py` -- `probe_rollout.py`'s absolute indices and v2 records.
- `python/tests/test_macro_check.py` -- `macro_check.py`'s new pure helpers.

**Modify**
- `python/ultrakill_ai/spaces.py` -- `ObsLayout.tech`, TECH constants, `tech_block`, the v2 action space,
  decode, noop, `pinned_action_rows`, `TECH_LAYOUTS`.
- `python/ultrakill_ai/rewards.py` -- `macro_landed` (Task 1); the `tech` weights, `tech_scale`, the paid term
  (Task 8).
- `python/ultrakill_ai/protocol.py` -- `BridgeIncompatible`, `mod_features`, `TECH_LAYOUT_FEATURES`.
- `python/ultrakill_ai/env.py` -- config fields, init validation, the handshake check, the 0.8 config, the wire
  gates, macro report capture, counters, v2 info keys, the S8 pay-out.
- `python/ultrakill_ai/progress.py` -- `TECH_METRICS`, `TECH_LOG_RAW`, present-only in `status.json`.
- `python/ultrakill_ai/training.py` -- resume refusal, `pin_action_rows`, absolute `ENTROPY_DIMS`,
  `TechDecayCallback`.
- `python/scripts/campaign_driver.py` -- `speed.env.tech_layout` validation, the layout guard in
  `ensure_trainer`, the `speed.train` routing for `ent_floor` / `ent_coef_max`.
- `python/scripts/probe_rollout.py` -- absolute target slice, v2 action / obs / macro in records, `--tech-layout`.
- `python/scripts/macro_check.py` -- `--refusals`, `--slam-trials`, `--record`, `--diff`.
- `python/scripts/eval.py`, `python/scripts/full_run.py` -- play a checkpoint under its own layout.
- `python/scripts/add_look_mode.py` -- docstring note (spec §4.3).
- `python/tests/test_speed_overrides.py` -- the `ent_floor` route.
- `.gitignore` -- `python/models/spec_*/pre_tech/`.
- `docs/commands.md`, `docs/protocol.md`, `docs/layout.md`, `docs/project-log.md`, `CLAUDE.md`, `AGENTS.md`.

---

### Task 0: Worktree, baseline, and the landing procedure

**Files:** none changed.

- [ ] **Step 1: Create the worktree from the pushed main**

```powershell
git -C F:\Github\ULTRAKILL-AI fetch origin
git -C F:\Github\ULTRAKILL-AI worktree add F:\Github\ULTRAKILL-AI-tech -b tech-break-python origin/main
```

- [ ] **Step 2: Run the whole suite in the worktree, against the worktree's code**

```powershell
cd F:\Github\ULTRAKILL-AI-tech\python
$env:PYTHONPATH = "F:\Github\ULTRAKILL-AI-tech\python"
F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python -c "import ultrakill_ai; print(ultrakill_ai.__file__)"
Get-ChildItem tests\test_*.py | ForEach-Object { F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python $_.FullName; if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
```

Expected: the first command prints a path under `F:\Github\ULTRAKILL-AI-tech\python\ultrakill_ai\`; all 41 files
print `N tests passed`. If the path is the main tree, fix `PYTHONPATH` before going on.

- [ ] **Step 3: Record the baseline** (file count and the sum of the `N tests passed` lines) in the task notes;
  Task 12 updates CLAUDE.md's test count from it.

- [ ] **Step 4: The landing procedure every later task ends with** (do not run it now)

```powershell
# in the worktree, after the task's commit:
git -C F:\Github\ULTRAKILL-AI-tech fetch origin
git -C F:\Github\ULTRAKILL-AI-tech rebase origin/main     # post_times.py pushes times.md to main every 10 min
# re-run the FULL suite (rule 5) after the rebase, then:
git -C F:\Github\ULTRAKILL-AI-tech push origin HEAD:main
git -C F:\Github\ULTRAKILL-AI-tech push -f origin tech-break-python
# bring the MAIN tree (the live one) level with origin/main, so post_times.py can keep pushing:
git -C F:\Github\ULTRAKILL-AI fetch origin
git -C F:\Github\ULTRAKILL-AI merge --ff-only origin/main
```

No commit in this plan touches `python/models/`, so `merge --ff-only` never conflicts with the trainer's
uncommitted model files; if it refuses because main moved, fetch and repeat. If `post_times.py` holds
`index.lock` for a moment, retry.

---

### Task 1: Layout constants, the v1 pins, v2 packing and decoding (`spaces.py`)

**Files:**
- Modify: `python/ultrakill_ai/spaces.py`; `python/ultrakill_ai/rewards.py` (add `macro_landed` only)
- Create: `python/tests/test_tech_layout.py`

- [ ] **Step 1: Write the v1 pins FIRST -- they must pass on the unchanged code**

Create `python/tests/test_tech_layout.py`. The two hashes were computed on 2026-09-23 from the unchanged
`spaces.py` with exactly this fixture.

```python
"""The S7 tech layout: v1 pinned byte for byte, and v2's index maps. No game:  python tests/test_tech_layout.py

The v1 half is a PIN, not a behaviour test: it was written against the unchanged spaces.py and must pass both
before and after the tech layout lands. The live trainer re-imports this package at every round boundary, so a
v1 packing or action table that moved by one float would reach twelve games unannounced. The hashes below were
taken on 2026-09-23 from exactly this fixture; if one fails on a clean main, do not "update the hash" -- find
what changed the v1 vector.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC,
    ACTION_NVEC_CAMPAIGN,
    ObsLayout,
    action_space,
    decode_action,
    noop_action,
    pack_observation,
)

PIN_RAW = {
    "player": {
        "pos": [10.0, 2.0, 30.0], "vel": [3.0, -1.0, 12.0], "local_vel": [1.5, -1.0, 12.5],
        "forward": [0.0, 0.0, 1.0], "yaw": 37.5, "pitch": -12.0, "hp": 83, "anti_hp": 12.0,
        "stamina": 212.5, "grounded": False, "sliding": True, "weapon_slot": 3, "dead": False,
        "heavy_fall": True, "weapon_variation": 1, "slot_counts": [3, 3, 3, 3, 3, 0],
    },
    "enemies": [
        {"id": 101, "type": 0, "health": 0.75, "visible": True, "rel": [4.0, 1.0, 9.0], "dist": 10.0,
         "pos": [14.0, 3.0, 39.0]},
        {"id": 102, "type": 7, "health": 2.5, "visible": False, "rel": [-20.0, 0.5, 30.0], "dist": 36.1,
         "pos": [-10.0, 2.5, 60.0]},
        {"id": 103, "type": 42, "health": 10.0, "visible": True, "rel": [0.0, 8.0, 45.0], "dist": 45.7,
         "pos": [10.0, 10.0, 75.0]},
    ],
    "rays": [float(3 + 2 * k) for k in range(16)],
    "ground_rays": [1.0, 1.5, 30.0, 2.0, 45.0, -1.0, 0.5, 12.0],
    "ground_ray_center": 1.2,
    "stats": {"kills": 4, "style": 120, "seconds": 41.5, "restarts": 0, "level_complete": False},
    "campaign": {
        "exit": {"pos": [60.0, -5.0, 220.0], "active": True},
        "checkpoints": [
            {"id": "a", "pos": [12.0, 2.0, 80.0], "activated": False, "current": False},
            {"id": "b", "pos": [0.0, 0.0, 5.0], "activated": True, "current": True},
        ],
        "locked_doors": [{"pos": [20.0, 2.0, 50.0]}],
        "arena_enemies_alive": 3, "timer_running": True, "input_locked": False, "seconds": 41.5,
    },
}
PIN_EXPLORE = [0.1, 0.0, 0.25, 0.5, 1.0, 0.0, 0.75, 0.2, 0.05]
PIN_TARGET = {"key": "40,1,408", "pos": [40.0, 1.0, 108.5], "hops": 9, "open": True, "locked": False,
              "active": True}
PIN_ENEMY_MAX = {101: 1.0, 102: 5.0}
V1_CAMPAIGN_SHA256 = "ff3c2c8040590275974a586fe0351d6324521cecdb28c9119b861ab791265644"
V1_GRIND_SHA256 = "3fee7c7fcf166e5bd92e9faa658965c9fbfe76455ed7f46f0c9dd140c74093c6"
V1_CAMPAIGN_NVEC = (3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3)
V1_DECODE_CASES = [
    ([1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0],
     {"move": [0, 0], "buttons": [], "slot": 0, "look": [0.0, 0.0], "look_mode": 0}),
    ([2, 0, 1, 0, 1, 1, 0, 1, 5, 10, 0, 2],
     {"move": [-1, 1], "buttons": ["jump", "slide", "fire1", "punch"], "slot": 5, "look": [90.0, -20.0],
      "look_mode": 2}),
    ([0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 6, 1],
     {"move": [1, -1], "buttons": ["dash", "fire2"], "slot": 0, "look": [-90.0, 20.0], "look_mode": 1}),
    ([1, 2, 0, 0, 0, 0, 0, 1, 2, 4, 5],  # Cyber Grind, 11 wide
     {"move": [1, 0], "buttons": ["punch"], "slot": 2, "look": [-1.0, 6.0], "look_mode": 0}),
]


def sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def pinned_campaign_vector() -> np.ndarray:
    return pack_observation(PIN_RAW, ObsLayout(campaign=True), PIN_ENEMY_MAX, PIN_EXPLORE, PIN_TARGET)


def test_v1_campaign_packing_is_pinned_byte_for_byte():
    out = pinned_campaign_vector()
    assert out.shape == (479,) and out.dtype == np.float32
    assert sha(out) == V1_CAMPAIGN_SHA256, sha(out)


def test_v1_cyber_grind_packing_is_pinned_byte_for_byte():
    out = pack_observation(PIN_RAW, ObsLayout(), PIN_ENEMY_MAX)
    assert out.shape == (448,)
    assert sha(out) == V1_GRIND_SHA256, sha(out)


def test_v1_action_tables_are_pinned():
    assert tuple(int(v) for v in ACTION_NVEC) == V1_CAMPAIGN_NVEC[:11]
    assert tuple(int(v) for v in ACTION_NVEC_CAMPAIGN) == V1_CAMPAIGN_NVEC
    assert tuple(int(v) for v in action_space(campaign=True).nvec) == V1_CAMPAIGN_NVEC
    assert tuple(int(v) for v in action_space().nvec) == V1_CAMPAIGN_NVEC[:11]
    assert ObsLayout(campaign=True).size == 479 and ObsLayout().size == 448


def test_v1_decoding_is_pinned():
    for vector, expected in V1_DECODE_CASES:
        assert decode_action(np.array(vector)) == expected, vector
    assert list(noop_action(campaign=True)) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0]
    assert list(noop_action()) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run the pins on the unchanged code**

Run: `PY tests\test_tech_layout.py`
Expected: `4 tests passed`. A hash mismatch here, before any edit, is a platform float difference: stop and
report it; do not regenerate the pin.

- [ ] **Step 3: Add the failing v2 tests** (insert above the `__main__` block)

```python
from ultrakill_ai.rewards import macro_landed  # noqa: E402
from ultrakill_ai.spaces import (  # noqa: E402
    ACTION_NVEC_TECH,
    HOOK_INDEX,
    HOOK_ROWS,
    MACRO_INDEX,
    MACRO_ROWS,
    MACROS,
    MOVE_TECH_FIELDS,
    TECH_BLOCK,
    TECH_HAZARD,
    TECH_LAYOUTS,
    TECH_MACRO,
    TECH_MOVE,
    TECH_PROJECTILES,
    TECH_WEAPON,
    VARIANT_INDEX,
    VARIANT_ROWS,
    pinned_action_rows,
    tech_block,
)

MOVE_TECH = {
    "heavy_fall": True, "slam_force": 6.5, "bounce_window": True, "coyote": 0.03, "wall_jumps": 2,
    "wall_available": True, "boost": False, "boost_left": 36.0, "pre_slide_speed": 4.5, "jump_cooldown": True,
    "slide_grace": 0.25, "riding_rocket": False,
    # diagnostics the packer must never read (docs/protocol.md): wall-clock seconds and absolute stamps
    "slide_since": 0.08, "slide_timestamp": 5321.25, "jump_timestamp": 5321.3,
    "slam_storage": False, "can_jump": True, "ssj": None, "ssj_last": None,
}
EXPECTED_A = [1.0, 0.65, 1.0, 0.03, 2.0 / 3.0, 1.0, 0.0, 0.36, 1.5, 1.0, 0.25, 0.0]
PLAYER = {"hp": 100}


def test_v2_layout_is_530_with_the_tech_block_at_479():
    v2 = ObsLayout(campaign=True, tech=True)
    assert v2.size == 530 and v2.campaign_start == 443 and v2.tech_start == 479
    assert TECH_BLOCK == 51 and TECH_LAYOUTS == ("v1", "v2")
    assert (TECH_MOVE, TECH_WEAPON, TECH_PROJECTILES, TECH_MACRO, TECH_HAZARD) == (
        slice(0, 12), slice(12, 22), slice(22, 40), slice(40, 43), slice(43, 51))
    assert [name for name, *_ in MOVE_TECH_FIELDS] == [
        "heavy_fall", "slam_force", "bounce_window", "coyote", "wall_jumps", "wall_available", "boost",
        "boost_left", "pre_slide_speed", "jump_cooldown", "slide_grace", "riding_rocket"]


def test_v2_action_space_is_15_dims_57_logits_appended():
    assert tuple(int(v) for v in ACTION_NVEC_TECH) == V1_CAMPAIGN_NVEC + (6, 4, 2)
    assert int(ACTION_NVEC_TECH.sum()) == 57
    assert (MACRO_INDEX, VARIANT_INDEX, HOOK_INDEX) == (12, 13, 14)
    assert (list(MACRO_ROWS), list(VARIANT_ROWS), list(HOOK_ROWS)) == (
        list(range(45, 51)), list(range(51, 55)), [55, 56])
    assert MACROS == ("none", "ssj", "ssj_wall", "core_nuke", "rocket_down", "coin_rocket")
    assert tuple(int(v) for v in action_space(campaign=True, tech=True).nvec) == V1_CAMPAIGN_NVEC + (6, 4, 2)


def test_the_tech_action_space_is_campaign_only():
    try:
        action_space(campaign=False, tech=True)
    except ValueError as exc:
        assert "campaign" in str(exc)
    else:
        raise AssertionError("a Cyber Grind tech action space must be refused")


def test_v2_decoding_keeps_every_v1_key_and_adds_three():
    vector, expected = V1_DECODE_CASES[1]
    out = decode_action(np.array(vector + [1, 3, 1]))
    assert {k: out[k] for k in expected} == expected
    assert (out["macro"], out["variant"], out["hook"]) == (1, 3, True)
    assert set(out) == set(expected) | {"macro", "variant", "hook"}
    assert list(noop_action(campaign=True, tech=True)) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0]


def test_v2_packing_keeps_indices_0_to_478_exactly():
    raw = {**PIN_RAW, "move_tech": MOVE_TECH,
           "macro": {"requested": "ssj", "result": "ran", "ssj_bucket": 1, "ssj_landed": True}}
    v2 = pack_observation(raw, ObsLayout(campaign=True, tech=True), PIN_ENEMY_MAX, PIN_EXPLORE, PIN_TARGET)
    assert v2.shape == (530,)
    assert sha(v2[:479]) == V1_CAMPAIGN_SHA256, "the tech keys must not move one v1 float"
    assert np.allclose(v2[479:491], EXPECTED_A, atol=1e-6)
    assert np.allclose(v2[519:522], [1.0, 0.0, 1.0 / 3.0], atol=1e-6)
    assert not v2[491:519].any() and not v2[522:530].any()


def test_block_a_packs_the_twelve_fields_in_order():
    block = tech_block({"player": PLAYER, "move_tech": MOVE_TECH})
    assert len(block) == TECH_BLOCK
    assert np.allclose(block[0:12], EXPECTED_A, atol=1e-6)


def test_block_a_clips_every_scale():
    extreme = {**MOVE_TECH, "slam_force": 50.0, "coyote": 999.0, "wall_jumps": 7, "boost_left": 250.0,
               "pre_slide_speed": 30.0, "slide_grace": 3.0}
    block = tech_block({"player": PLAYER, "move_tech": extreme})
    assert block[1] == 1.0 and block[3] == 1.0 and block[4] == 1.0 and block[7] == 1.0
    assert block[8] == 2.0 and block[10] == 1.0
    negative = {**MOVE_TECH, "slam_force": -3.0, "coyote": -1.0, "pre_slide_speed": -2.0}
    low = tech_block({"player": PLAYER, "move_tech": negative})
    assert low[1] == 0.0 and low[3] == 0.0 and low[8] == 0.0


def test_the_wall_clock_fields_are_never_packed():
    base = tech_block({"player": PLAYER, "move_tech": MOVE_TECH})
    moved = {**MOVE_TECH, "slide_since": 9.0e9, "slide_timestamp": -1.0, "jump_timestamp": 1.0e12}
    assert tech_block({"player": PLAYER, "move_tech": moved}) == base


def test_block_d_reports_ran_not_run_and_the_landed_bucket():
    cases = [
        ({"result": "ran", "ssj_bucket": 1, "ssj_landed": True}, [1.0, 0.0, 1.0 / 3.0]),
        ({"result": "ran", "ssj_bucket": 3, "ssj_landed": True}, [1.0, 0.0, 1.0]),
        ({"result": "ran", "ssj_bucket": 0, "ssj_landed": False}, [1.0, 0.0, 0.0]),
        ({"result": "ran", "ssj_bucket": 7, "ssj_landed": True}, [1.0, 0.0, 0.0]),  # impossible: never paid
        ({"result": "refused", "reason": "not_sliding", "ssj_bucket": -1, "ssj_landed": False}, [0.0, 1.0, 0.0]),
        ({"result": "degraded", "ssj_bucket": 2, "ssj_landed": True}, [0.0, 1.0, 0.0]),
        ({"result": "disabled", "reason": "reserved", "ssj_bucket": -1}, [0.0, 1.0, 0.0]),
    ]
    for report, expected in cases:
        block = tech_block({"player": PLAYER, "macro": report})
        assert np.allclose(block[40:43], expected, atol=1e-6), report
        assert macro_landed(report) == (block[42] > 0.0), report
    assert tech_block({"player": PLAYER})[40:43] == [0.0, 0.0, 0.0]


def test_reserved_blocks_stay_zero_even_when_the_mod_sends_them():
    raw = {"player": PLAYER, "move_tech": MOVE_TECH,
           "weapon_tech": {"gun_ready": True, "coin_charge": 400, "variation": 1},
           "projectiles": [{"kind": "coin", "rel": [1.0, 2.0, 3.0], "dist": 3.7, "age": 0.2}]}
    block = tech_block(raw)
    assert not any(block[12:40]) and not any(block[43:51])


def test_an_old_dll_frame_and_a_frame_without_a_player_pack_zeros():
    assert tech_block(PIN_RAW) == [0.0] * TECH_BLOCK, "0.7.2: no move_tech, no macro"
    assert tech_block({"move_tech": MOVE_TECH}) == [0.0] * TECH_BLOCK, "no player: nothing is read"


def test_the_pinned_rows_at_each_gate_setting():
    assert pinned_action_rows({1}, False, False) == [47, 48, 49, 50, 51, 52, 53, 54, 55, 56], "the S7 gates"
    assert pinned_action_rows(set(), False, False) == list(range(45, 57)), "no live macro: the whole dim"
    assert pinned_action_rows({1, 2}, True, True) == [48, 49, 50]
    assert pinned_action_rows({1}, True, False) == [47, 48, 49, 50, 55, 56]
```

- [ ] **Step 4: Run to verify the v2 half fails**

Run: `PY tests\test_tech_layout.py`
Expected: FAIL with `ImportError: cannot import name 'macro_landed'`.

- [ ] **Step 5: Add `macro_landed` to `rewards.py`** (after `damage_share`, before `class CampaignStep`)

```python
def macro_landed(report: Any) -> bool:
    """THE S8 GATE (spec §4.5): the mod RAN a macro and its own instrument reports an accepted SSJ bucket.

    Accepted means 1-3: `TrySSJ` computes `(int)(dt / 0.008)` and rejects bucket 0 and anything at or past
    `ssjMaxFrames` (4). Never a speed delta -- that gate is perversely signed, because `TrySSJ` OVERWRITES the
    velocity with `velocityAfterSlide` (floored at 24) plus the bonus, so a speed bar pays most when the player
    is slow. `ssj_bucket` / `ssj_landed` are populated only when `result == "ran"` (docs/protocol.md, the mod
    review's finding 3); `result` is checked again here so a refused macro can never pay or read as landed.
    """
    if not isinstance(report, dict) or report.get("result") != "ran" or not report.get("ssj_landed"):
        return False
    try:
        bucket = int(report.get("ssj_bucket"))
    except (TypeError, ValueError):
        return False
    return 1 <= bucket <= 3
```

- [ ] **Step 6: Implement the layout in `spaces.py`**

(a) After `from gymnasium import spaces`, add `from ultrakill_ai.rewards import macro_landed` (rewards imports
only `ultrakill_ai.times`, so there is no cycle).

(b) After `LOOK_MODE_INDEX = len(BASE_NVEC)`, add:

```python
# ---------------------------------------------------------------------------
# The v2 TECH action layout -- stage S7 of docs/superpowers/specs/2026-09-20-speedrun-tech.md, §4.1.
# APPEND ONLY: every campaign logit row 0-44 keeps its index, which is what lets scripts/add_tech_heads.py copy
# rows 0-44 verbatim and makes the migration exact.
# ---------------------------------------------------------------------------
TECH_LAYOUTS = ("v1", "v2")  # EnvConfig.tech_layout; ultrakill_ai/ckpt_layout.py carries the same literal
MACROS = ("none", "ssj", "ssj_wall", "core_nuke", "rocket_down", "coin_rocket")  # the mod's own values 0..5
VARIANT_CHOICES = 4  # 0 keep, 1..3 = variation 0..2 of the held slot (the mod's `variant`)
HOOK_CHOICES = 2  # 0 off, 1 hold the whiplash (the mod's `hook` HoldButton)
ACTION_NVEC_TECH = np.array((*ACTION_NVEC_CAMPAIGN, len(MACROS), VARIANT_CHOICES, HOOK_CHOICES),
                            dtype=np.int64)  # 15 dims, 57 logits
MACRO_INDEX, VARIANT_INDEX, HOOK_INDEX = 12, 13, 14
_CAMPAIGN_LOGITS = int(ACTION_NVEC_CAMPAIGN.sum())  # 45
MACRO_ROWS = range(_CAMPAIGN_LOGITS, _CAMPAIGN_LOGITS + len(MACROS))  # 45-50
VARIANT_ROWS = range(MACRO_ROWS.stop, MACRO_ROWS.stop + VARIANT_CHOICES)  # 51-54
HOOK_ROWS = range(VARIANT_ROWS.stop, VARIANT_ROWS.stop + HOOK_CHOICES)  # 55-56
```

(c) Replace `action_space`, `decode_action` and `noop_action` with the following, and add `pinned_action_rows`:

```python
def action_space(campaign: bool = False, tech: bool = False) -> spaces.MultiDiscrete:
    if tech and not campaign:
        raise ValueError("the tech action layout is campaign-only: Cyber Grind's 11-dimension space never widens")
    return spaces.MultiDiscrete(ACTION_NVEC_TECH if tech else ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC)


def decode_action(a: np.ndarray) -> dict[str, Any]:
    """The mod command for one action. The width tells the mode apart: 15 values carry the tech heads, 12 a look
    mode, 11 neither. A 12- or 11-wide action decodes to exactly the dict it always did (tests/test_tech_layout.py)."""
    a = np.asarray(a, dtype=np.int64)
    forward = int(a[0]) - 1
    side = int(a[1]) - 1
    pressed = [name for name, bit in zip(BUTTONS, a[2 : 2 + len(BUTTONS)]) if bit]
    i = 2 + len(BUTTONS)
    out = {
        "move": [side, forward],
        "buttons": pressed,
        "slot": int(a[i]),
        "look": [YAW_BINS[int(a[i + 1])], PITCH_BINS[int(a[i + 2])]],
        # env.step resolves this and pops it: it is a Python-side look policy, never sent to the mod.
        "look_mode": int(a[LOOK_MODE_INDEX]) if len(a) > LOOK_MODE_INDEX else 0,
    }
    if len(a) > HOOK_INDEX:
        # v2 only. env.step pops all three and decides what reaches the wire (UltrakillEnv._apply_tech): a raw
        # policy value is never sent as is.
        out["macro"] = int(a[MACRO_INDEX])
        out["variant"] = int(a[VARIANT_INDEX])
        out["hook"] = bool(a[HOOK_INDEX])
    return out


def noop_action(campaign: bool = False, tech: bool = False) -> np.ndarray:
    """Stand still and look straight ahead. Explicit indices: negative ones address the wrong slots at width 12+."""
    nvec = ACTION_NVEC_TECH if tech else ACTION_NVEC_CAMPAIGN if campaign else ACTION_NVEC
    a = np.zeros(len(nvec), dtype=np.int64)
    a[0] = a[1] = 1
    i = 2 + len(BUTTONS)
    a[i + 1] = YAW_BINS.index(0.0)
    a[i + 2] = PITCH_BINS.index(0.0)
    return a  # look mode 0 (free look); under v2 also no macro, keep the variant, no hook


def pinned_action_rows(live_macros, variant: bool, hook: bool) -> list[int]:
    """The v2 logit rows whose value can never reach the game under these gates, ascending.

    training.pin_action_rows freezes them (the plan's "Spec deviations" 2): a head that cannot act receives only
    the entropy bonus and would drift toward uniform, losing the §4.4 prior before its stage opens. A dimension
    with NO live non-default value is pinned whole; otherwise only its dead values are.
    """
    live = {int(v) for v in live_macros if 0 < int(v) < len(MACROS)}
    rows = list(MACRO_ROWS) if not live else [MACRO_ROWS[v] for v in range(1, len(MACROS)) if v not in live]
    if not variant:
        rows += list(VARIANT_ROWS)
    if not hook:
        rows += list(HOOK_ROWS)
    return rows
```

(d) After `CAMPAIGN_BLOCK = 36 ...`, add:

```python
# The v2 TECH block (§4.3): 51 floats appended AFTER the campaign block, so indices 0-478 keep their meaning.
# Offsets inside the block; the absolute start is ObsLayout.tech_start (479 at the defaults).
TECH_BLOCK = 51
TECH_MOVE = slice(0, 12)  # block A, live at S7           -> 479-490
TECH_WEAPON = slice(12, 22)  # block B, reserved 0.0 (S9)   -> 491-500
TECH_PROJECTILES = slice(22, 40)  # block C, reserved (S10) -> 501-518
TECH_MACRO = slice(40, 43)  # block D, live at S7          -> 519-521
TECH_HAZARD = slice(43, 51)  # block E, reserved (S11)     -> 522-529
# Block A in packing order: (mod key, divisor, low, high). The scales come from the decompiled NewMovement and
# are DERIVED, not measured -- the S6 slam check reads the real ranges. `slide_grace`, NOT `slide_since`: the
# mod review's finding 2 (2026-09-20). Never pack slide_since, slide_timestamp or jump_timestamp (docs/protocol.md).
MOVE_TECH_FIELDS = (
    ("heavy_fall", 1.0, 0.0, 1.0),
    ("slam_force", 10.0, 0.0, 1.0),  # 1 at slam start, +5/s while heavyFall; >= 5.5 is the 12.5x bounce
    ("bounce_window", 1.0, 0.0, 1.0),
    ("coyote", 1.0, 0.0, 1.0),  # gc.sinceLastGrounded, game seconds (999 without a ground check)
    ("wall_jumps", 3.0, 0.0, 1.0),  # currentWallJumps; the budget is 3
    ("wall_available", 1.0, 0.0, 1.0),
    ("boost", 1.0, 0.0, 1.0),
    ("boost_left", 100.0, 0.0, 1.0),  # dash i-frames: 100 at Dodge(), -4 per fixed step
    ("pre_slide_speed", 3.0, 0.0, 2.0),  # |v|/24 or slamForce; StartSlide clamps it to 3
    ("jump_cooldown", 1.0, 0.0, 1.0),
    ("slide_grace", 1.0, 0.0, 1.0),  # the fraction of the SSJ window still open
    ("riding_rocket", 1.0, 0.0, 1.0),
)
```

(e) In `ObsLayout`, add the field `tech: bool = False  # tech_layout v2: the 51-float TECH block (479 -> 530)`
after `campaign`, add two properties, and count the block in `size`:

```python
    @property
    def campaign_start(self) -> int:
        """Absolute index of the campaign block's first value (443 at the defaults)."""
        return self.player_size + self.max_enemies * self.enemy_size + self.horizontal_rays + self.ground_rays + 2

    @property
    def tech_start(self) -> int:
        """Absolute index of the v2 TECH block (479 at the defaults). Meaningful only when `tech` is set."""
        return self.campaign_start + CAMPAIGN_BLOCK

    @property
    def size(self) -> int:
        return (
            self.player_size
            + self.max_enemies * self.enemy_size
            + self.horizontal_rays
            + self.ground_rays
            + self.mode_size
            + (TECH_BLOCK if self.tech else 0)
        )
```

(f) After `campaign_block`, add:

```python
def _scalar(value: Any) -> float:
    """A mod field as a float: a bool is 0/1; missing, null, non-numeric or non-finite is 0.0 (the old-DLL value)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def tech_block(obs: dict[str, Any]) -> list[float]:
    """The 51 v2 values (index map: docs/superpowers/plans/2026-09-23-tech-break-python.md).

    Every read defaults to 0.0, so a 0.7.2 DLL, an `obs_move_tech` that is off, or a frame with no player all
    pack the vector scripts/add_tech_heads.py initialised the new input columns for. Blocks B, C and E are
    reserved and stay 0.0 even if the mod sends their source blocks: their packers are written at S9-S11.
    """
    out = [0.0] * TECH_BLOCK
    move = obs.get("move_tech") if obs.get("player") else None
    if isinstance(move, dict):
        for k, (name, divisor, low, high) in enumerate(MOVE_TECH_FIELDS):
            out[TECH_MOVE.start + k] = min(high, max(low, _scalar(move.get(name)) / divisor))
    report = obs.get("macro")
    if isinstance(report, dict):
        ran = report.get("result") == "ran"
        out[TECH_MACRO.start] = 1.0 if ran else 0.0
        out[TECH_MACRO.start + 1] = 0.0 if ran else 1.0
        if macro_landed(report):
            out[TECH_MACRO.start + 2] = _scalar(report.get("ssj_bucket")) / 3.0
    return out
```

(g) In `pack_observation`, immediately before the final `assert i == layout.size`, add:

```python
    if layout.tech:
        put(tech_block(obs))
```

- [ ] **Step 7: Run the new tests and the existing spaces tests**

Run: `PY tests\test_tech_layout.py` → `16 tests passed`
Run: `PY tests\test_spaces.py` → passes unchanged.

- [ ] **Step 8: Full suite (rule 5) -> all green.**

- [ ] **Step 9: Commit and land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/spaces.py python/ultrakill_ai/rewards.py python/tests/test_tech_layout.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 layout: v1 pinned by hash; v2 obs 530 / action 15-57 in spaces.py (dormant)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4 (land).

---

### Task 2: The protocol half -- `features`, the loud refusal, the 0.8 config sent on every connect

**Files:**
- Modify: `python/ultrakill_ai/protocol.py`, `python/ultrakill_ai/env.py`, `python/scripts/campaign_driver.py`
  (`_speed_env` only)
- Create: `python/tests/test_tech_env.py`

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_tech_env.py`:

```python
"""S7 env wiring: the mod handshake, the 0.8 config, the action on the wire, the macro counters, the info keys.

No game:  python tests/test_tech_env.py

`FakeTechLevel` is test_campaign_env's FakeLevel speaking mod 0.8.0: `hello` carries `features`, every frame with
a player carries `move_tech` once `obs_move_tech` has been configured, and a step that asked for a macro gets the
mod's `macro` report back. The v1 tests here are pins: a v1 env's configure call and wire action are today's,
key for key (the key sets below were read off FakeLevel on 2026-09-23, before this change).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_campaign_env import LEVEL, FakeLevel, action, forward, make_env  # noqa: E402

import campaign_driver  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv, tech_gates  # noqa: E402
from ultrakill_ai.protocol import (  # noqa: E402
    RECOVERABLE,
    TECH_LAYOUT_FEATURES,
    BridgeError,
    BridgeIncompatible,
    mod_features,
)
from ultrakill_ai.spaces import HOOK_INDEX, MACRO_INDEX, MACROS, VARIANT_INDEX  # noqa: E402

V1_CONFIG_KEYS = {
    "block_human_input", "command_timeout_s", "difficulty", "fixed_fps", "frameskip", "ground_ray_length",
    "ground_rays", "horizontal_rays", "max_enemies", "mute", "ray_length", "render", "reset_settle_frames",
    "soft_death", "unlimited_fps", "unlock_all_gear", "window_height", "window_width", "windowed",
}
TECH_CONFIG_KEYS = {
    "obs_move_tech", "obs_weapon_tech", "obs_projectiles", "obs_input_clock", "macros", "macro_ssj_wall",
    "allow_reserved_macros", "macro_wall_lead_unsafe", "variant_switching", "ssj_gap_s", "ssj_indicator",
}
V1_WIRE_KEYS = {"move", "buttons", "slot", "look"}
V08_FEATURES = ("monotonic_input_clock", "macro.ssj", "macro.ssj_wall", "obs.move_tech", "obs.weapon_tech",
                "obs.projectiles", "action.variant", "ssj_instrument")
MOVE_TECH = {
    "heavy_fall": False, "slam_force": 1.0, "bounce_window": False, "coyote": 0.0, "wall_jumps": 0,
    "wall_available": False, "boost": False, "boost_left": 0.0, "pre_slide_speed": 0.5, "jump_cooldown": False,
    "slide_grace": 0.0, "riding_rocket": False, "slide_since": 3.2, "slide_timestamp": 101.5,
    "jump_timestamp": 99.0,
}
EXPECTED_A = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 / 3.0, 0.0, 0.0, 0.0]
LANDED = {"result": "ran", "reason": None, "note": None, "ssj_bucket": 1, "ssj_landed": True}
REFUSED = {"result": "refused", "reason": "not_sliding", "note": None, "ssj_bucket": -1, "ssj_landed": False}
OFF_08 = {"obs_move_tech": False, "obs_weapon_tech": False, "obs_projectiles": False, "obs_input_clock": False,
          "macros": True, "macro_ssj_wall": False, "allow_reserved_macros": False, "macro_wall_lead_unsafe": False,
          "variant_switching": False, "ssj_gap_s": 0.012, "ssj_indicator": False}


class FakeTechLevel(FakeLevel):
    """FakeLevel speaking mod 0.8.0 (see the module docstring)."""

    def __init__(self, features=V08_FEATURES, mod_version: str = "0.8.0", emit_move_tech: bool = True):
        super().__init__()
        self.features = list(features)
        self.mod_version = mod_version
        self.emit_move_tech = emit_move_tech
        self.move_tech = dict(MOVE_TECH)
        self.macro_outcome = dict(LANDED)

    def connect(self, retry_seconds: float = 60.0) -> dict:
        self.connects += 1
        return {"type": "hello", "protocol": 1, "mod_version": self.mod_version, "scene": LEVEL,
                "features": list(self.features)}

    def step(self, action: dict) -> dict:
        obs = super().step(action)
        macro = action.get("macro")
        if macro:
            obs["macro"] = {"requested": MACROS[macro], **self.macro_outcome}
        return obs

    def _obs(self, event=None) -> dict:
        obs = super()._obs(event)
        if self.emit_move_tech and obs.get("player") and self.settings.get("obs_move_tech"):
            obs["move_tech"] = dict(self.move_tech)
        return obs


def tech_env(level_cls=FakeTechLevel, **overrides):
    env, _ = make_env(**{"tech_layout": "v2", **overrides})
    env.client = level_cls()
    return env, env.client


def tech_action(*, macro: int = 0, variant: int = 0, hook: int = 0, move: bool = True, buttons=()):
    a = np.zeros(15, dtype=np.int64)
    a[:12] = action(move=move, buttons=tuple(buttons))
    a[MACRO_INDEX], a[VARIANT_INDEX], a[HOOK_INDEX] = macro, variant, hook
    return a


# ---------------------------------------------------------------------------------------------
# Task 2: the handshake and the config
# ---------------------------------------------------------------------------------------------


def test_a_v1_env_sends_todays_configure_call_exactly():
    env, fake = make_env()
    try:
        env.reset()
        assert set(fake.settings) == V1_CONFIG_KEYS, sorted(set(fake.settings) ^ V1_CONFIG_KEYS)
    finally:
        env.close()


def test_a_v1_env_on_a_0_8_dll_switches_every_0_8_feature_off():
    env, _ = make_env()
    env.client = fake = FakeTechLevel()
    try:
        obs, _ = env.reset()
        assert set(fake.settings) == V1_CONFIG_KEYS | TECH_CONFIG_KEYS
        assert {k: fake.settings[k] for k in TECH_CONFIG_KEYS} == OFF_08
        assert obs.shape == (479,) and "move_tech" not in env._raw
    finally:
        env.close()


def test_a_v2_env_on_a_0_7_2_mod_fails_loudly_at_connect():
    env, fake = make_env(tech_layout="v2")  # FakeLevel: mod 0.5.0, no `features` at all
    try:
        env.reset()
    except BridgeIncompatible as exc:
        text = str(exc)
        for feature in TECH_LAYOUT_FEATURES:
            assert feature in text, text
        assert "0.5.0" in text and "tech_layout" in text
    else:
        raise AssertionError("a v2 env must refuse a mod without the 0.8.0 features")
    finally:
        env.close()
    assert fake.configures == 0, "nothing may be configured on a mod that cannot serve the layout"


def test_bridge_incompatible_escapes_every_recovery_handler():
    assert not issubclass(BridgeIncompatible, BridgeError), "the reconnect ladder retries every BridgeError"
    assert not issubclass(BridgeIncompatible, OSError)
    assert not any(issubclass(BridgeIncompatible, kind) for kind in RECOVERABLE)
    env, _ = make_env(tech_layout="v2")
    try:
        env._try_reconnect_until(time.monotonic() + 30.0)
    except BridgeIncompatible:
        pass
    else:
        raise AssertionError("the reconnect ladder swallowed an incompatible mod instead of ending the worker")
    finally:
        env.close()


def test_a_v2_env_on_a_0_8_mod_sends_the_tech_config():
    env, fake = tech_env()
    try:
        obs, _ = env.reset()
        assert obs.shape == (530,) and len(env.action_space.nvec) == 15
        assert {k: fake.settings[k] for k in TECH_CONFIG_KEYS} == {**OFF_08, "obs_move_tech": True}
    finally:
        env.close()


def test_the_gates_reach_the_mod_config():
    env, fake = tech_env(macro_ssj_wall=True, variant_switching=True)
    try:
        env.reset()
        assert fake.settings["macro_ssj_wall"] is True and fake.settings["variant_switching"] is True
        assert fake.settings["allow_reserved_macros"] is False, "3-5 have no switch: they are not built"
    finally:
        env.close()


def test_mod_features_reads_the_hello_array():
    assert mod_features({"features": ["a", 3, "b"]}) == frozenset({"a", "b"})
    assert mod_features({"protocol": 1}) == frozenset()
    assert mod_features(None) == frozenset()
    assert mod_features({"features": "obs.move_tech"}) == frozenset(), "a string is not an array"


def test_the_tech_gates_follow_the_config():
    assert tech_gates(EnvConfig()) == (frozenset({1}), False, False), "S7: M1 alone"
    assert tech_gates(EnvConfig(macro_ssj=False)).macros == frozenset()
    assert tech_gates(EnvConfig(macro_ssj_wall=True, variant_switching=True, hook_action=True)) == (
        frozenset({1, 2}), True, True)


def test_tech_layout_is_validated_at_init():
    for cfg in (EnvConfig(mode="campaign", tech_layout="v3"), EnvConfig(mode="cybergrind", tech_layout="v2")):
        try:
            UltrakillEnv(cfg).close()
        except ValueError as exc:
            assert "tech_layout" in str(exc)
        else:
            raise AssertionError(f"accepted {cfg.mode} / {cfg.tech_layout}")


def test_the_tech_fields_round_trip_through_a_config_file():
    cfg = EnvConfig(mode="campaign", tech_layout="v2", macro_ssj=True, macro_ssj_wall=False,
                    variant_switching=False, hook_action=True)
    back = EnvConfig.from_dict(cfg.to_dict())
    assert (back.tech_layout, back.macro_ssj, back.macro_ssj_wall, back.variant_switching, back.hook_action) == (
        "v2", True, False, False, True)


def test_the_plan_validator_accepts_the_tech_fields_and_refuses_a_bad_layout():
    assert campaign_driver._speed_env("plan.yaml", {"tech_layout": "v2", "macro_ssj": True}) == {
        "tech_layout": "v2", "macro_ssj": True}
    for block, needle in (({"tech_layout": "v3"}, "tech_layout"), ({"hook_action": "false"}, "boolean")):
        try:
            campaign_driver._speed_env("plan.yaml", block)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"accepted {block}")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run to verify it fails**

Run: `PY tests\test_tech_env.py`
Expected: FAIL with `ImportError: cannot import name 'tech_gates'`.

- [ ] **Step 3: `protocol.py` -- add after the `RECOVERABLE = (...)` line**

```python
class BridgeIncompatible(RuntimeError):
    """The mod on the other end cannot serve what this client is configured for -- e.g. `tech_layout: v2` against
    a 0.7.2 DLL, which would otherwise train 530 inputs of which 51 are always zero.

    DELIBERATELY NOT a BridgeError and NOT in RECOVERABLE. `UltrakillEnv._try_reconnect_until` retries every
    BridgeError, and its last rung relaunches the game: a wrong DLL would then cost a relaunch loop across twelve
    games. This escapes every handler instead, so the worker -- and with it the trainer -- dies with this message in
    runs/<run>_train.log. That is the loud failure spec §4.7 asks for.
    """


# What a `tech_layout: v2` client needs in `hello.features` (docs/protocol.md). A 0.7.x DLL sends no such array.
TECH_LAYOUT_FEATURES = ("monotonic_input_clock", "macro.ssj", "obs.move_tech")


def mod_features(hello: dict[str, Any] | None) -> frozenset[str]:
    """`hello.features` as a set of strings; missing (a 0.7.x DLL) or malformed reads as the empty set."""
    raw = (hello or {}).get("features")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(f for f in raw if isinstance(f, str))
```

- [ ] **Step 4: `env.py` -- imports, constants, config fields, gates**

(a) Imports: add `NamedTuple` to `from typing import Any`; add `BridgeIncompatible`, `TECH_LAYOUT_FEATURES` and
`mod_features` to the `ultrakill_ai.protocol` import; add `MOVE_TECH_FIELDS` and `TECH_LAYOUTS` to the
`ultrakill_ai.spaces` import.

(b) After `RESCUE_JUMP_M = 12.0`, add:

```python
SSJ_GAP_S = 0.012  # the mod's own `ssj_gap_s` default: the middle of SSJ bucket 1 (docs/protocol.md), sent explicitly
MOVE_TECH_GRACE = 30  # v2: frames with a player and no usable `move_tech` before the one-time warning
```

(c) In `EnvConfig`, directly after `sticky_slot_switch_every: int = 3`, add:

```python
    # THE S7 TECH LAYOUT (docs/superpowers/specs/2026-09-20-speedrun-tech.md §4.1-§4.3; the plan is
    # docs/superpowers/plans/2026-09-23-tech-break-python.md). "v1" -- the default, and every run before the break
    # -- is the 479-input, 12-dimension policy BYTE FOR BYTE (tests/test_tech_layout.py pins it). "v2" is 530
    # inputs and 15 dimensions / 57 logits; it needs mod 0.8.0 and refuses to start against anything older.
    tech_layout: str = "v1"
    # The v2 gates: which new heads may reach the game. A value that is off here is masked in Python (it never
    # reaches the wire), refused by the mod as well where the mod has a switch for it, and its logits are pinned
    # in training (training.pin_action_rows). The defaults are the S7 scope: M1 `ssj` alone.
    macro_ssj: bool = True
    macro_ssj_wall: bool = False  # also the mod's `macro_ssj_wall`; M2 was cut to reserved (2026-09-20 review)
    variant_switching: bool = False  # also the mod's `variant_switching`; opens at S9
    hook_action: bool = False  # the whiplash: a plain HoldButton the mod cannot refuse; opens at S10
```

(d) After the `EnvConfig` class (before `BEHAVIOUR_KEYS`), add:

```python
class TechGates(NamedTuple):
    """Which v2 heads may reach the game: the macro VALUES forwarded, and whether variant / hook are."""

    macros: frozenset
    variant: bool
    hook: bool


def tech_gates(cfg: EnvConfig) -> TechGates:
    """The gates of a config. Macro values 3-5 have no switch: they are reserved and not built in the mod."""
    macros = frozenset(v for v, on in ((1, cfg.macro_ssj), (2, cfg.macro_ssj_wall)) if on)
    return TechGates(macros, bool(cfg.variant_switching), bool(cfg.hook_action))
```

- [ ] **Step 5: `env.py` -- `__init__`**

Replace this block of `UltrakillEnv.__init__`:

```python
        if self.cfg.layout.campaign != campaign:
            # The layout follows the mode (Cyber Grind 448 inputs, campaign 479). Replaced, not edited in place:
            # train.py builds each game's config with dataclasses.replace, so they all share one layout object.
            self.cfg = replace(self.cfg, layout=replace(self.cfg.layout, campaign=campaign))

        self.observation_space = self.cfg.layout.space()
        # Campaign only: the look mode is appended as a 12th dimension, so Cyber Grind checkpoints stay loadable.
        self.action_space = action_space(campaign)
```

with:

```python
        if self.cfg.tech_layout not in TECH_LAYOUTS:
            # Caught here, not in a worker's first step: a typo would otherwise build v1 shapes under a config
            # everyone reads as v2.
            raise ValueError(f"tech_layout must be one of {TECH_LAYOUTS}, got {self.cfg.tech_layout!r}")
        tech = self.cfg.tech_layout == "v2"
        if tech and not campaign:
            raise ValueError("tech_layout v2 is campaign-only: Cyber Grind's 448 / 11 layout never widens")
        if self.cfg.layout.campaign != campaign or self.cfg.layout.tech != tech:
            # The layout follows the mode and the tech switch (Cyber Grind 448 inputs, campaign 479, tech 530).
            # Replaced, not edited in place: train.py builds each game's config with dataclasses.replace, so they
            # all share one layout object.
            self.cfg = replace(self.cfg, layout=replace(self.cfg.layout, campaign=campaign, tech=tech))

        self.observation_space = self.cfg.layout.space()
        # Campaign only: the look mode is appended as a 12th dimension, so Cyber Grind checkpoints stay loadable;
        # tech_layout v2 appends macro / variant / hook after it (15 dims, 57 logits).
        self.action_space = action_space(campaign, tech=tech)
        self._tech = tech
        self._gates = tech_gates(self.cfg)
        self._mod_features: frozenset = frozenset()  # hello.features at the last connect
        self._macro_reasons: dict[str, int] = {}  # per episode: why each sent macro did not run
        self._move_tech_missing = 0  # consecutive v2 frames with a player and no usable move_tech
        self._move_tech_warned = False
```

- [ ] **Step 6: `env.py` -- `_ensure_connected` checks the mod and sends the 0.8 config**

In `_ensure_connected`, replace `self.client.connect(retry_seconds=retry)` with:

```python
        hello = self.client.connect(retry_seconds=retry)
        self._check_mod(hello)
```

and add `**self._tech_mod_config(),` as the last argument of the `self.client.configure(...)` call (after
`**mod_layout,`). Then add these three methods right after `_ensure_connected`:

```python
    def _required_features(self) -> set[str]:
        """The `hello.features` this config cannot run without. Empty under v1: a 0.7.2 DLL serves v1 as today."""
        return set(TECH_LAYOUT_FEATURES) if self._tech else set()

    def _check_mod(self, hello: dict[str, Any] | None) -> None:
        """Refuses, loudly and unrecoverably, a mod that cannot serve this config (see BridgeIncompatible)."""
        self._mod_features = mod_features(hello)
        need = self._required_features()
        if not need:
            return
        version = (hello or {}).get("mod_version")
        missing = sorted(need - self._mod_features)
        if missing:
            self.envlog.event("mod_incompatible", mod=version, missing=",".join(missing))
            raise BridgeIncompatible(
                f"port {self.cfg.port}: tech_layout {self.cfg.tech_layout!r} needs mod features {missing}; the game "
                f"runs mod {version!r} with features {sorted(self._mod_features) or 'none'}. Install mod 0.8.0 "
                "(docs/commands.md, 'S7 -- the one break') or set tech_layout back to v1.")
        self.envlog.event("mod_features_ok", mod=version, layout=self.cfg.tech_layout)

    def _tech_mod_config(self) -> dict[str, Any]:
        """The mod 0.8.0 settings, sent EXPLICITLY on every connect -- or nothing at all to a 0.7.x DLL.

        The mod's config is per GAME PROCESS, not per connection (docs/protocol.md): a private test that switched
        `variant_switching` on, or a v2 client before a rollback, would otherwise be inherited by whoever connects
        next. So every switch is sent with the value THIS config means. Against a DLL that advertises no
        `features` (0.7.x) the dict is empty and the configure call is today's, key for key.
        """
        if not self._tech and "obs.move_tech" not in self._mod_features:
            return {}
        return {
            "obs_move_tech": self._tech,  # block A
            "obs_weapon_tech": False,  # block B: S9
            "obs_projectiles": False,  # block C: S10
            "obs_input_clock": False,  # a diagnostic, never for training
            "macros": True,  # the master switch; a macro still runs only when an action asks for one
            "macro_ssj_wall": bool(self._tech and 2 in self._gates.macros),
            "allow_reserved_macros": False,
            "macro_wall_lead_unsafe": False,
            "variant_switching": bool(self._tech and self._gates.variant),
            "ssj_gap_s": SSJ_GAP_S,
            "ssj_indicator": False,
        }
```

- [ ] **Step 7: `campaign_driver._speed_env` -- refuse an unknown `tech_layout` at plan load**

In `_speed_env`, directly after the `for key, value in block.items():` type-check loop and before
`refused = sorted(...)`, add:

```python
    layout = block.get("tech_layout")
    if layout is not None:
        from ultrakill_ai.spaces import TECH_LAYOUTS  # noqa: PLC0415 - lazy for the same reason as EnvConfig above

        if layout not in TECH_LAYOUTS:
            # At plan load, not in twelve workers: an unknown value raises in UltrakillEnv.__init__, i.e. a crash
            # loop, and a typo that happened to be accepted would train a layout nobody asked for.
            raise ValueError("%s: speed.env.tech_layout must be one of %s, not %r"
                             % (path, list(TECH_LAYOUTS), layout))
```

- [ ] **Step 8: Run the tests**

Run: `PY tests\test_tech_env.py` → `11 tests passed`
Run: `PY tests\test_campaign_env.py`, `PY tests\test_freeze_recovery.py`, `PY tests\test_bridge_recovery.py`,
`PY tests\test_speed_overrides.py` → all pass unchanged.

- [ ] **Step 9: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/protocol.py python/ultrakill_ai/env.py python/scripts/campaign_driver.py python/tests/test_tech_env.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 protocol: hello.features check (BridgeIncompatible), explicit 0.8 config, tech_layout switch (default v1)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 3: The action on the wire, the macro report, the counters, `info` and `status.json`

**Files:**
- Modify: `python/ultrakill_ai/env.py`, `python/ultrakill_ai/progress.py`
- Modify: `python/tests/test_tech_env.py` (append)

- [ ] **Step 1: Append the failing tests** to `test_tech_env.py` (above the `__main__` block)

```python
# ---------------------------------------------------------------------------------------------
# Task 3: the wire, block A / D through the env, the counters, info, status.json
# ---------------------------------------------------------------------------------------------


def test_a_v1_step_puts_todays_keys_on_the_wire():
    env, fake = make_env()
    try:
        env.reset()
        env.step(forward())
        assert set(fake.last_action) == V1_WIRE_KEYS
    finally:
        env.close()


def test_v2_forwards_the_ssj_macro_and_masks_every_reserved_value():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(macro=1))
        assert fake.last_action["macro"] == 1
        for value in (0, 2, 3, 4, 5):
            env.step(tech_action(macro=value))
            assert set(fake.last_action) == V1_WIRE_KEYS, value
    finally:
        env.close()


def test_v2_masks_variant_and_hook_until_their_stages_open():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(variant=2, hook=1))
        assert set(fake.last_action) == V1_WIRE_KEYS and "hook" not in fake.last_action["buttons"]
    finally:
        env.close()
    env, fake = tech_env(variant_switching=True, hook_action=True, macro_ssj_wall=True)
    try:
        env.reset()
        env.step(tech_action(macro=2, variant=2, hook=1))
        assert fake.last_action["macro"] == 2 and fake.last_action["variant"] == 2
        assert "hook" in fake.last_action["buttons"]
    finally:
        env.close()


def test_block_a_and_block_d_are_packed_from_the_mod_reports():
    env, fake = tech_env()
    try:
        obs, _ = env.reset()
        assert np.allclose(obs[479:491], EXPECTED_A, atol=1e-6)
        obs, *_ = env.step(tech_action(macro=1))
        assert np.allclose(obs[519:522], [1.0, 0.0, 1.0 / 3.0], atol=1e-6)
        obs, *_ = env.step(tech_action())
        assert not obs[519:522].any(), "no macro sent, no report: zeros"
        fake.macro_outcome = dict(REFUSED)
        obs, *_ = env.step(tech_action(macro=1))
        assert np.allclose(obs[519:522], [0.0, 1.0, 0.0])
        assert not obs[491:519].any() and not obs[522:530].any()
    finally:
        env.close()


def test_the_macro_report_survives_an_input_lock_after_the_step():
    env, fake = tech_env()
    try:
        env.reset()
        fake.lock_steps = 3
        before = fake.steps
        obs, *_ = env.step(tech_action(macro=1))
        assert fake.steps - before > 1, "the env stepped through the lock"
        assert obs[519] == 1.0, "block D kept the macro step's own report"
        assert env._behaviour["macro_ran"] == 1
    finally:
        env.close()


def test_the_counters_and_the_v2_info_keys():
    env, fake = tech_env()
    try:
        env.reset()
        env.step(tech_action(macro=1))  # sent, ran, landed
        fake.macro_outcome = dict(REFUSED)
        env.step(tech_action(macro=1))  # sent, refused: not_sliding
        env.step(tech_action(macro=4))  # requested, masked
        _, _, _, _, info = env.step(tech_action(variant=1, hook=1))
        assert info["macro_request_frac"] == 3 / 4 and info["macro_sent_frac"] == 2 / 4
        assert info["macro_ran_frac"] == 1 / 4 and info["macro_refused_frac"] == 1 / 4
        assert info["macro_landed_frac"] == 1 / 4 and info["macro_ran_share"] == 1 / 2
        assert info["variant_request_frac"] == 1 / 4 and info["hook_request_frac"] == 1 / 4
        assert info["macro_refusal_reasons"] == {"not_sliding": 1}
    finally:
        env.close()


def test_a_v1_info_carries_no_tech_keys():
    env, _ = make_env()
    try:
        env.reset()
        _, _, _, _, info = env.step(forward())
        assert not [k for k in info if k.startswith(("macro_", "variant_request", "hook_request"))]
    finally:
        env.close()


def test_a_missing_move_tech_is_warned_once_and_packs_zeros():
    env, _ = tech_env(level_cls=lambda: FakeTechLevel(emit_move_tech=False))
    try:
        obs, _ = env.reset()
        for _ in range(40):
            obs, *_ = env.step(tech_action(move=False))
        assert env._move_tech_warned and not obs[479:491].any()
    finally:
        env.close()


def test_progress_carries_the_macro_channel_only_when_the_env_reports_it():
    from test_progress import episode_info  # noqa: PLC0415

    from ultrakill_ai.progress import ProgressCallback  # noqa: PLC0415

    tech = {"macro_request_frac": 0.1, "macro_sent_frac": 0.02, "macro_ran_frac": 0.005,
            "macro_refused_frac": 0.015, "macro_landed_frac": 0.005, "macro_ran_share": 0.25,
            "variant_request_frac": 0.15, "hook_request_frac": 0.1,
            "macro_refusal_reasons": {"not_sliding": 3}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra in (("v1", {}), ("v2", tech)):
            run = Path(tmp) / name
            cb = ProgressCallback(run / "status.json", 1000, name, 1, update_every_s=0.0)
            cb._on_training_start()
            cb._record_episode(0, {**episode_info("Level 0-1", completed=1, seconds=90.0), **extra})
            cb._write(time.time())
            mean = json.loads((run / "status.json").read_text(encoding="utf-8"))["mean_100"]
            line = json.loads((run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            if extra:
                assert mean["macro_sent_frac"] == 0.02 and line["macro_refusal_reasons"] == {"not_sliding": 3}
            else:
                assert "macro_sent_frac" not in mean and "macro_refusal_reasons" not in line
```

- [ ] **Step 2: Run to verify they fail**

Run: `PY tests\test_tech_env.py`
Expected: FAIL at `test_a_missing_move_tech...` / `test_block_a_and_block_d...` (the env has no `_note_tech_obs`,
sends the raw `macro`/`variant`/`hook` keys from `decode_action`, and `info` has no macro keys).

- [ ] **Step 3: `env.py` -- the counters**

Replace the last two lines of `BEHAVIOUR_KEYS`:

```python
                  *("held_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)),     # decisions each slot was held
                  *("kills_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)))    # kills while each slot was held
```

with:

```python
                  *("held_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)),     # decisions each slot was held
                  *("kills_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)),    # kills while each slot was held
                  # STAGE S7, the macro channel. All zero under v1, and `_info` emits them only under v2.
                  #   macro_request  decisions whose policy macro value was not 0 (raw use of the head)
                  #   macro_sent     ... and the gates forwarded it to the mod
                  #   macro_ran / macro_refused  the mod's `result` for a sent macro (refused = refused, degraded
                  #                  or disabled); macro_landed = ran with an accepted SSJ bucket
                  #   variant_request / hook_request  raw use of the other two heads, masked or not
                  "macro_request", "macro_sent", "macro_ran", "macro_refused", "macro_landed",
                  "variant_request", "hook_request")
```

and in `_reset`, directly after `self._behaviour = dict.fromkeys(BEHAVIOUR_KEYS, 0)`, add
`self._macro_reasons = {}`.

- [ ] **Step 4: `env.py` -- `_step`**

(a) Directly after `look_mode = int(command.pop("look_mode", 0))`, add:

```python
        # v2: the three tech heads come off the command HERE, so no raw policy value can reach the wire;
        # `_apply_tech` below puts back exactly what the gates allow. None under v1, and then nothing changes.
        tech = self._pop_tech(command)
```

(b) Directly after `self._sticky_slot(prev, command)`, add `self._apply_tech(command, tech)`.

(c) Replace:

```python
        cur = self.client.step(command)
        if campaign:
            cur = self._guard_exit(self._skip_locked(cur))
```

with:

```python
        cur = self.client.step(command)
        # The mod reports a macro on THIS reply only. `_skip_locked` can replace `cur` with a later frame (an input
        # lock right after the step), so the report is taken first and put back for block D's packer.
        macro_report = cur.get("macro") if tech is not None else None
        if campaign:
            cur = self._guard_exit(self._skip_locked(cur))
        if macro_report is not None and "macro" not in cur:
            cur["macro"] = macro_report
        self._note_macro_result(macro_report)
```

(d) Add these methods directly after `_sticky_slot`:

```python
    def _pop_tech(self, command: dict[str, Any]) -> dict[str, Any] | None:
        """Takes the three v2 heads off the decoded command. None under v1 (`decode_action` adds them at width 15)."""
        if "macro" not in command:
            return None
        return {"macro": int(command.pop("macro")), "variant": int(command.pop("variant")),
                "hook": bool(command.pop("hook"))}

    def _apply_tech(self, command: dict[str, Any], tech: dict[str, Any] | None) -> None:
        """Puts on the wire exactly what the gates allow, and counts every request whether or not it was sent.

        A masked value leaves the message as the v1 action message; it is never charged (spec §4.2: a charge
        teaches the policy to avoid the head rather than to learn its preconditions).
        """
        if tech is None:
            return
        b = self._behaviour
        if tech["macro"]:
            b["macro_request"] += 1
            if tech["macro"] in self._gates.macros:
                command["macro"] = tech["macro"]
                b["macro_sent"] += 1
        if tech["variant"]:
            b["variant_request"] += 1
            if self._gates.variant:
                command["variant"] = tech["variant"]
        if tech["hook"]:
            b["hook_request"] += 1
            if self._gates.hook:
                command["buttons"] = [*command["buttons"], "hook"]

    def _note_macro_result(self, report: dict[str, Any] | None) -> None:
        """The mod's verdict on a sent macro. Refusal reasons are kept per episode: a macro refused ~always is a
        design bug (§4.2), and the reason says which precondition."""
        if not isinstance(report, dict):
            return
        b = self._behaviour
        if report.get("result") == "ran":
            b["macro_ran"] += 1
            if report.get("ssj_landed"):
                b["macro_landed"] += 1
        else:
            b["macro_refused"] += 1
            reason = str(report.get("reason") or report.get("result") or "unknown")
            self._macro_reasons[reason] = self._macro_reasons.get(reason, 0) + 1

    def _note_tech_obs(self, raw: dict[str, Any]) -> None:
        """Warns ONCE per env when a v2 env keeps receiving frames with a player and no usable `move_tech`.

        Block A then packs 0.0 -- the old-DLL vector, so nothing breaks -- but the policy is blind to exactly what
        the break was for. The handshake already refused a DLL without `obs.move_tech`; this catches a flag lost
        inside the game (a per-process config another client overwrote) or a renamed field.
        """
        if self._move_tech_warned or not raw.get("player"):
            return
        move = raw.get("move_tech")
        names = [name for name, *_ in MOVE_TECH_FIELDS]
        missing = [n for n in names if n not in move] if isinstance(move, dict) else names
        if not missing:
            self._move_tech_missing = 0
            return
        self._move_tech_missing += 1
        if self._move_tech_missing >= MOVE_TECH_GRACE:
            self._move_tech_warned = True
            what = "no move_tech block" if not isinstance(move, dict) else "move_tech without " + ",".join(missing)
            self.envlog.event("move_tech_missing", frames=self._move_tech_missing, what=what)
            print(f"WARNING port {self.cfg.port}: {self._move_tech_missing} frames with a player and {what}; "
                  "block A packs 0.0 (warned once per env)", flush=True)
```

(e) At the top of `_pack`, add:

```python
        if self._tech:
            self._note_tech_obs(raw)
```

(f) In `_info`, directly after `info["variation_known_frac"] = b["variation_known"] / steps`, add:

```python
        if self._tech:
            # STAGE S7, the macro channel -- v2 only, so a v1 info dict is byte for byte what it was. Per decision,
            # like the slot counters. `macro_request_frac` is the policy's raw use of the head (0.10 at the 5 x 2%
            # prior); `macro_sent_frac` what the gates let through (M1 alone at S7); `macro_ran_share` the share of
            # sent macros the mod RAN -- near 0 is a design bug (§4.2), and `macro_refusal_reasons` says why.
            info["macro_request_frac"] = b["macro_request"] / steps
            info["macro_sent_frac"] = b["macro_sent"] / steps
            info["macro_ran_frac"] = b["macro_ran"] / steps
            info["macro_refused_frac"] = b["macro_refused"] / steps
            info["macro_landed_frac"] = b["macro_landed"] / steps
            info["macro_ran_share"] = b["macro_ran"] / max(1, b["macro_sent"])
            info["variant_request_frac"] = b["variant_request"] / steps
            info["hook_request_frac"] = b["hook_request"] / steps
            info["macro_refusal_reasons"] = dict(self._macro_reasons)
```

- [ ] **Step 5: `progress.py` -- present-only metrics**

(a) After `SLOT_METRICS = (...)`, add:

```python
# STAGE S7's macro channel (UltrakillEnv._info, tech_layout v2 only). A mean each in status.json's `mean_100` ONLY
# when the window carries them, and raw in episodes.jsonl only when the env reported them -- so a v1 run's two
# files are byte for byte what they were. Not in metrics_log.csv: adding a column mid-run would misalign that file.
TECH_METRICS = ("macro_request_frac", "macro_sent_frac", "macro_ran_frac", "macro_refused_frac",
                "macro_landed_frac", "macro_ran_share", "variant_request_frac", "hook_request_frac")
TECH_LOG_RAW = ("macro_refusal_reasons",)
```

(b) In `_record_episode`'s `stats = {...}`, after `**{name: field(name) for name in SLOT_METRICS},` add
`**{name: field(name) for name in TECH_METRICS},`.

(c) In `_log_episode`, directly after `line.update({name: info.get(name) for name in EPISODE_LOG_RAW})`, add:

```python
        line.update({name: info[name] for name in (*TECH_METRICS, *TECH_LOG_RAW) if name in info})
```

(d) In `_snapshot`, directly after the `recent = {key: self._recent_mean(key) for key in (...)}` statement, add:

```python
        # S7's macro channel: present only when the window carries it (see TECH_METRICS).
        for key in TECH_METRICS:
            value = _mean(ep.get(key) for ep in self.episodes_recent)
            if value is not None:
                recent[key] = value
```

(`.get`, not `self._recent_mean`: an episode restored from an older status file has no such key.)

- [ ] **Step 6: Run the tests**

Run: `PY tests\test_tech_env.py` → `20 tests passed`
Run: `PY tests\test_slot_counters.py`, `PY tests\test_progress.py`, `PY tests\test_campaign_env.py` → pass unchanged.

- [ ] **Step 7: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/env.py python/ultrakill_ai/progress.py python/tests/test_tech_env.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 env: gated macro/variant/hook on the wire, macro report kept across input locks, v2-only counters" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 4: `scripts/add_tech_heads.py` -- the weight migration (§4.4)

**Files:**
- Create: `python/scripts/add_tech_heads.py`, `python/tests/test_add_tech_heads.py`
- Modify: `python/scripts/add_look_mode.py` (docstring only)

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_add_tech_heads.py`:

```python
"""scripts/add_tech_heads.py: the S7 weight migration, on tiny SB3 models. No game:  python tests/test_add_tech_heads.py

The contract is EXACTNESS, a strictly stronger one than add_look_mode's `mean_kl <= 0.03`: nothing is repurposed,
so on the twelve shared action dims the widened policy must compute what the source computed (spec §4.4). Anything
non-zero is a copy bug. The three new heads must sit exactly on their priors, independent of the state.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import add_tech_heads as ath  # noqa: E402
from add_tech_heads import (  # noqa: E402
    FIRST_LAYERS,
    KL_TOL,
    PRIOR_BIAS,
    PRIOR_ENTROPY,
    V1_INPUTS,
    V1_LOGITS,
    V2_INPUTS,
    V2_LOGITS,
    MigrationRefused,
    add_tech_heads,
    build_tech_model,
    checkpoint_observations,
    displacement,
    verdict,
)
from transfer_weights import SpacesEnv  # noqa: E402
from ultrakill_ai.spaces import ObsLayout, action_space  # noqa: E402

NET_ARCH = [16, 16]
V2_NVEC = [3, 3, 2, 2, 2, 2, 2, 2, 6, 11, 7, 3, 6, 4, 2]


def v1_model(seed: int = 1) -> PPO:
    torch.manual_seed(seed)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True).space(), action_space(campaign=True)),
               policy_kwargs={"net_arch": NET_ARCH}, device="cpu")


def trained_v1(seed: int = 1) -> PPO:
    """A v1 model after one real Adam step on BOTH stacks, so every parameter has moments to carry."""
    model = v1_model(seed)
    obs = torch.as_tensor(np.random.default_rng(seed).normal(size=(8, V1_INPUTS)).astype(np.float32))
    latent_pi, latent_vf = model.policy.mlp_extractor(obs)
    loss = model.policy.action_net(latent_pi).square().mean() + model.policy.value_net(latent_vf).square().mean()
    model.policy.optimizer.zero_grad()
    loss.backward()
    model.policy.optimizer.step()
    return model


def saved(tmp, model: PPO, name: str = "src.zip") -> Path:
    path = Path(tmp) / name
    model.save(path)
    return path


def states(rows: int = 128, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, V1_INPUTS)).astype(np.float32)


def migrated(tmp):
    return build_tech_model(saved(tmp, trained_v1()))


def test_first_layers_copy_479_columns_and_zero_the_51_new_ones():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        for name in FIRST_LAYERS:
            assert d[name].shape == (NET_ARCH[0], V2_INPUTS)
            assert torch.equal(d[name][:, :V1_INPUTS], s[name])
            assert not d[name][:, V1_INPUTS:].any()
            bias = name.replace(".weight", ".bias")
            assert torch.equal(d[bias], s[bias]), "no mean-fold: nothing is repurposed"


def test_everything_else_is_copied_bit_for_bit_including_the_critic():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        for name, value in d.items():
            if name in FIRST_LAYERS or name.startswith("action_net"):
                continue
            assert torch.equal(value, s[name]), name  # value_net too: never reset (§4.4 step 2)


def test_the_action_head_gets_zero_rows_and_the_prior_biases():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        s, d = src.policy.state_dict(), dst.policy.state_dict()
        assert d["action_net.weight"].shape == (V2_LOGITS, NET_ARCH[-1])
        assert torch.equal(d["action_net.weight"][:V1_LOGITS], s["action_net.weight"])
        assert not d["action_net.weight"][V1_LOGITS:].any()
        assert torch.equal(d["action_net.bias"][:V1_LOGITS], s["action_net.bias"])
        expected = torch.zeros(V2_LOGITS - V1_LOGITS)
        for row, bias in PRIOR_BIAS.items():
            expected[row - V1_LOGITS] = bias
        assert torch.allclose(d["action_net.bias"][V1_LOGITS:], expected)


def test_the_migration_is_exact_on_the_twelve_shared_dims():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        d = displacement(src, dst, states())
        assert d["mean_kl"] <= 1e-9 and d["max_kl"] <= 1e-9, d
        assert d["greedy_changed"] == 0.0 and d["mean_abs_dv"] <= 1e-5, d
        assert verdict(d) == [], verdict(d)


def test_the_new_heads_sit_on_their_priors_whatever_the_state():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        d = displacement(src, dst, states())
        for name in ("macro", "variant", "hook"):
            assert d[f"{name}_prior_err"] <= 1e-6, (name, d)
            assert d[f"{name}_state_spread"] == 0.0, (name, d)
        assert abs(d["entropy_added"] - PRIOR_ENTROPY) <= 1e-5, d


def test_the_verdict_names_a_broken_copy():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, _, _ = migrated(tmp)
        with torch.no_grad():
            dst.policy.action_net.weight[0, 0] += 1.0
        problems = verdict(displacement(src, dst, states()))
        assert any("mean_kl" in p for p in problems), problems


def test_the_optimizer_is_carried_along_both_axes():
    """add_look_mode.carry_optimizer padded axis 0 only; the first layer grows along axis 1 here (§4.4 step 4)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, carried, why = migrated(tmp)
        assert carried, why
        names = [name for name, _ in dst.policy.named_parameters()]
        old = src.policy.optimizer.state_dict()["state"]
        new = dst.policy.optimizer.state_dict()["state"]
        assert set(old) == set(new) and len(old) == len(names)
        for index, name in enumerate(names):
            for key in ("exp_avg", "exp_avg_sq"):
                a, b = old[index][key], new[index][key]
                if name in FIRST_LAYERS:
                    assert b.shape == (NET_ARCH[0], V2_INPUTS)
                    assert torch.equal(b[:, :V1_INPUTS], a) and not b[:, V1_INPUTS:].any()
                elif name.startswith("action_net"):
                    assert b.shape[0] == V2_LOGITS
                    assert torch.equal(b[:V1_LOGITS], a) and not b[V1_LOGITS:].any()
                else:
                    assert torch.equal(b, a), name
            assert float(new[index]["step"]) == float(old[index]["step"]) == 1.0, "bias correction continues"


def test_add_tech_heads_writes_a_loadable_v2_model_and_keeps_the_step_axis():
    with tempfile.TemporaryDirectory() as tmp:
        src = trained_v1()
        src.num_timesteps = 66_524_386
        src._n_updates = 123_932
        dest = Path(tmp) / "out" / "tech_init.zip"
        _, _, measured, carried = add_tech_heads(saved(tmp, src), dest, observations=states())
        assert carried and measured["mean_kl"] <= KL_TOL
        loaded = PPO.load(dest, device="cpu")
        assert loaded.observation_space.shape == (V2_INPUTS,)
        assert [int(v) for v in loaded.action_space.nvec] == V2_NVEC
        assert loaded.num_timesteps == 66_524_386 and loaded._n_updates == 123_932
        assert loaded.policy_kwargs == {"net_arch": NET_ARCH}


def test_it_never_overwrites_anything():
    with tempfile.TemporaryDirectory() as tmp:
        source = saved(tmp, v1_model())
        dest = Path(tmp) / "exists.zip"
        dest.write_bytes(b"keep me")
        for target in (dest, source):
            try:
                add_tech_heads(source, target, observations=states(8))
            except FileExistsError:
                pass
            else:
                raise AssertionError(f"overwrote {target}")
        assert dest.read_bytes() == b"keep me"


def test_it_refuses_anything_but_a_v1_campaign_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        v2_path = Path(tmp) / "v2.zip"
        add_tech_heads(saved(tmp, v1_model()), v2_path, observations=states(8))
        torch.manual_seed(5)
        grind = PPO("MlpPolicy", SpacesEnv(ObsLayout().space(), action_space()),
                    policy_kwargs={"net_arch": NET_ARCH}, device="cpu")
        for bad in (v2_path, saved(tmp, grind, "grind.zip")):
            out = Path(tmp) / f"from_{bad.stem}.zip"
            try:
                add_tech_heads(bad, out, observations=states(8))
            except ValueError as exc:
                assert "v1 campaign layout" in str(exc)
            else:
                raise AssertionError(f"migrated {bad.name}")
            assert not out.exists()


def test_a_refused_migration_writes_nothing():
    original = ath.displacement
    ath.displacement = lambda *a, **k: {**original(*a, **k), "mean_kl": 0.5}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "never.zip"
            try:
                ath.add_tech_heads(saved(tmp, v1_model()), dest, observations=states(8))
            except MigrationRefused as exc:
                assert "mean_kl" in str(exc)
            else:
                raise AssertionError("a copy bug was written to disk")
            assert not dest.exists()
    finally:
        ath.displacement = original


def test_a_fresh_optimizer_is_refused_unless_allowed():
    original = ath.carry_optimizer
    ath.carry_optimizer = lambda src, dst: (False, "forced by the test")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source = saved(tmp, v1_model())
            try:
                ath.add_tech_heads(source, Path(tmp) / "a.zip", observations=states(8))
            except MigrationRefused as exc:
                assert "forced by the test" in str(exc)
            else:
                raise AssertionError("losing Adam's state at 60M steps must be an explicit decision")
            ath.add_tech_heads(source, Path(tmp) / "b.zip", observations=states(8), allow_fresh_optimizer=True)
            assert (Path(tmp) / "b.zip").exists()
    finally:
        ath.carry_optimizer = original


def test_real_observations_are_read_from_the_checkpoints_last_obs():
    with tempfile.TemporaryDirectory() as tmp:
        for steps, value in ((100, 0.25), (200, 0.5)):
            model = v1_model()
            model._last_obs = np.full((3, V1_INPUTS), value, dtype=np.float32)
            model.save(Path(tmp) / f"ckpt_{steps}_steps.zip")
        obs = checkpoint_observations(Path(tmp), 4)
        assert obs.shape == (4, V1_INPUTS)
        assert np.all(obs[:3] == 0.5) and np.all(obs[3] == 0.25), "newest checkpoint first"
        assert checkpoint_observations(Path(tmp) / "empty", 4) is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run to verify it fails**

Run: `PY tests\test_add_tech_heads.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'add_tech_heads'`.

- [ ] **Step 3: Write `python/scripts/add_tech_heads.py`**

```python
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
```

- [ ] **Step 4: Add the spec §4.3 note to `add_look_mode.py`'s docstring**, as its last paragraph:

```text
After the S7 tech break (docs/superpowers/specs/2026-09-20-speedrun-tech.md) this script's shape guard refuses
every v2 checkpoint (530 inputs, 15 action dims). That is correct, not a regression: its job ended at the
look-mode migration, and the only path across the tech break is scripts/add_tech_heads.py.
```

- [ ] **Step 5: Run the tests**

Run: `PY tests\test_add_tech_heads.py` → `13 tests passed`
Run: `PY tests\test_look_mode_transfer.py` → passes unchanged.

- [ ] **Step 6: A dry run on a scratch COPY of the live newest checkpoint** (reads the live dir, writes only to
  `%TEMP%`; the live run is not touched):

```powershell
$tmp = "$env:TEMP\tech_dry"; New-Item -ItemType Directory $tmp -Force | Out-Null
$ck = F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python -c "import sys; sys.path.insert(0,'scripts'); import supervise; from pathlib import Path; print(supervise.choose_resume(Path(r'F:\Github\ULTRAKILL-AI\python\models\spec_0-1_speed'))[0])"
Copy-Item $ck "$tmp\src.zip"
PY scripts\add_tech_heads.py "$tmp\src.zip" "$tmp\tech_init.zip" --obs-from F:\Github\ULTRAKILL-AI\python\models\spec_0-1_speed --ent-floor 6.5
```

Expected: the tensor table, `states` near 2400, `mean_kl` printing `0.000000000` (<= 1e-6), `greedy_changed`
`0.000000000`, `entropy_added` `1.398610669` (±0.001), `ent_floor 6.5 -> 7.8986`, `optimizer carried`. Record
the printed block in the task notes; Task 12's log entry quotes it. Delete `$tmp` afterwards (it is scratch).

- [ ] **Step 7: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/scripts/add_tech_heads.py python/scripts/add_look_mode.py python/tests/test_add_tech_heads.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 migration: add_tech_heads.py (479->530, 45->57, exact on shared dims, measure-before-write)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 5: The shape guards -- `ckpt_layout.py`, `training.main`, the driver

**Files:**
- Create: `python/ultrakill_ai/ckpt_layout.py`, `python/tests/test_ckpt_layout.py`
- Modify: `python/ultrakill_ai/training.py`, `python/scripts/campaign_driver.py`

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_ckpt_layout.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `PY tests\test_ckpt_layout.py`
Expected: FAIL with `ImportError: cannot import name 'ckpt_layout' from 'ultrakill_ai'`.

- [ ] **Step 3: Write `python/ultrakill_ai/ckpt_layout.py`**

```python
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
    """(observation width, action nvec) out of an SB3 zip, or None when it cannot be read."""
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data").decode("utf-8"))
        width = int(data["observation_space"]["_shape"][0])
        nvec = tuple(int(v) for v in re.findall(r"-?\d+", str(data["action_space"]["nvec"])))
    except (OSError, KeyError, IndexError, TypeError, ValueError, zipfile.BadZipFile, UnicodeDecodeError):
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
```

- [ ] **Step 4: `training.py` -- refuse before any game is touched**

(a) Imports: add `from ultrakill_ai.ckpt_layout import TECH_LAYOUTS, resume_problem  # noqa: E402`.

(b) Add, after `hyperparams_in_force`:

```python
def resume_refusal(resume: str | None, env_cfg: EnvConfig) -> str | None:
    """Why `--resume` cannot be trained under this config's layout, or None. Checked BEFORE any worker or game.

    SB3's own `load(env=...)` would also refuse a space mismatch, but only after twelve SubprocVecEnv workers exist
    and with a message that names no migration. Under the driver that is a restart loop; here it is one line
    naming scripts/add_tech_heads.py and exit code 3.
    """
    if not resume or env_cfg.mode != "campaign":
        return None
    if env_cfg.tech_layout not in TECH_LAYOUTS:
        return f"unknown tech_layout {env_cfg.tech_layout!r} (expected one of {list(TECH_LAYOUTS)})"
    return resume_problem(resume, env_cfg.tech_layout)
```

(c) In `main()`, directly after `env_cfg = EnvConfig.from_dict(env_cfg_dict)`, add:

```python
    refusal = resume_refusal(args.resume, env_cfg)
    if refusal:
        print(f"REFUSING TO RESUME {args.resume}: {refusal}", flush=True)
        raise SystemExit(3)
```

- [ ] **Step 5: `campaign_driver.py` -- the guard in `ensure_trainer`**

(a) Imports: after `from ultrakill_ai.campaign import CAMPAIGN_LEVELS_SHIPPED, safe_name`, add
`from ultrakill_ai import ckpt_layout  # noqa: E402  (stdlib only)`.

(b) In `Driver.ensure_trainer`, directly after the two lines that resolve `resume, steps` (the
`choose_resume` call and its `if resume is None:` fallback), add:

```python
        problem = self.layout_problem(resume, sup.cfg.config)
        if problem:
            # NOT a crash loop: training.main would refuse the same file at every restart, twelve games up and
            # nothing learning. Say it once and keep polling; check_run.py's ALERTS shows the missing trainer. The
            # fix is the S7 runbook's migration -- after it, the next poll starts the trainer by itself.
            self.log_once("layout:%s" % stage.run, "LAYOUT MISMATCH -- trainer NOT started for %s (%s): %s"
                          % (stage.level, stage.kind, problem))
            return "layout_mismatch"
```

(c) Add the method to `Driver` (next to `ensure_trainer`):

```python
    def layout_problem(self, resume: Path, config: str) -> str | None:
        """None, or why `resume` must not be trained under the generated config `config` (plan Task 5).

        The generated YAML and the zip's JSON only -- no numpy, no torch -- and only on the path that is about to
        spawn a trainer. An unreadable file of either kind is None: train.py and SB3 decide then.
        """
        try:
            data = yaml.safe_load((self.cfg.cwd / config).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None
        env = data.get("env") or {}
        if env.get("mode") != "campaign":
            return None
        path = Path(resume)
        path = path if path.is_absolute() else self.cfg.cwd / path
        return ckpt_layout.resume_problem(path, str(env.get("tech_layout", "v1")))
```

Also update `ensure_trainer`'s docstring return list: `"layout_mismatch"` (the resume file's layout does not
match the stage config; the trainer is not started).

- [ ] **Step 6: Run the tests**

Run: `PY tests\test_ckpt_layout.py` → `9 tests passed`
Run: `PY tests\test_campaign_driver.py`, `PY tests\test_supervise.py`, `PY tests\test_resume_hyperparams.py` →
pass unchanged.

- [ ] **Step 7: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/ckpt_layout.py python/ultrakill_ai/training.py python/scripts/campaign_driver.py python/tests/test_ckpt_layout.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 guards: refuse a checkpoint whose layout is not the config's (train.py exit 3, driver layout_mismatch)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 6: `scripts/quarantine_pre_tech.py` -- move the 479-wide files aside, never delete

**Files:**
- Create: `python/scripts/quarantine_pre_tech.py`, `python/tests/test_quarantine_pre_tech.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_quarantine_pre_tech.py`:

```python
"""scripts/quarantine_pre_tech.py: v1 checkpoints MOVE to pre_tech/, nothing is deleted, nothing else moves.

No game:  python tests/test_quarantine_pre_tech.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_ckpt_layout import fake_ckpt  # noqa: E402

from quarantine_pre_tech import QUARANTINE, apply_moves, plan_moves  # noqa: E402
from ultrakill_ai.ckpt_layout import CAMPAIGN_NVEC_V1, CAMPAIGN_NVEC_V2  # noqa: E402


def model_dir(tmp) -> Path:
    d = Path(tmp) / "spec_0-1_speed"
    fake_ckpt(d / "ckpt_66450000_steps.zip", 479, CAMPAIGN_NVEC_V1)
    fake_ckpt(d / "ckpt_66500000_steps.zip", 479, CAMPAIGN_NVEC_V1)
    fake_ckpt(d / "latest.zip", 479, CAMPAIGN_NVEC_V1)
    fake_ckpt(d / "best.zip", 479, CAMPAIGN_NVEC_V1)
    fake_ckpt(d / "tech_init.zip", 530, CAMPAIGN_NVEC_V2)
    (d / "junk.zip").write_bytes(b"not a zip")
    (d / "best.json").write_text("{}", encoding="utf-8")
    (d / "explore_Level_0-1_47800.npz").write_bytes(b"counts")
    return d


def test_the_plan_moves_every_v1_zip_and_nothing_else():
    with tempfile.TemporaryDirectory() as tmp:
        d = model_dir(tmp)
        moves, other = plan_moves(d)
        assert sorted(src.name for src, _ in moves) == [
            "best.zip", "ckpt_66450000_steps.zip", "ckpt_66500000_steps.zip", "latest.zip"]
        assert all(dst == d / QUARANTINE / src.name for src, dst in moves)
        assert [p.name for p in other] == ["junk.zip"]


def test_apply_moves_and_deletes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        d = model_dir(tmp)
        before = sorted(p.name for p in d.rglob("*") if p.is_file())
        moves, _ = plan_moves(d)
        apply_moves(moves)
        after = sorted(p.name for p in d.rglob("*") if p.is_file())
        assert before == after, "a move, never a delete"
        assert sorted(p.name for p in (d / QUARANTINE).iterdir()) == [
            "best.zip", "ckpt_66450000_steps.zip", "ckpt_66500000_steps.zip", "latest.zip"]
        assert (d / "tech_init.zip").exists() and (d / "best.json").exists()
        assert plan_moves(d)[0] == [], "a second run finds nothing left to move"


def test_a_clash_in_pre_tech_moves_nothing_at_all():
    with tempfile.TemporaryDirectory() as tmp:
        d = model_dir(tmp)
        fake_ckpt(d / QUARANTINE / "latest.zip", 479, CAMPAIGN_NVEC_V1)
        moves, _ = plan_moves(d)
        try:
            apply_moves(moves)
        except FileExistsError as exc:
            assert "latest.zip" in str(exc)
        else:
            raise AssertionError("overwrote a quarantined file")
        assert (d / "ckpt_66500000_steps.zip").exists(), "all or nothing"


def test_the_quarantine_is_gitignored():
    ignore = (ROOT.parent / ".gitignore").read_text(encoding="utf-8")
    assert "python/models/spec_*/pre_tech/" in ignore.splitlines()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run to verify it fails**

Run: `PY tests\test_quarantine_pre_tech.py` → FAIL, `ModuleNotFoundError: No module named 'quarantine_pre_tech'`.

- [ ] **Step 3: Write `python/scripts/quarantine_pre_tech.py`**

```python
"""Moves every v1 (479-input / 12-dimension) checkpoint of ONE model directory into `<dir>/pre_tech/`.

Stage S7 of docs/superpowers/specs/2026-09-20-speedrun-tech.md, §4.4 "Integration hazard": after the break the
driver's round init picks the stage's newest checkpoint, and a 479-wide `ckpt_*_steps.zip` must never be handed to
a 530-wide config. train.py and the driver both refuse that already (ultrakill_ai/ckpt_layout.py); this takes the
files out of the running. MOVE, never delete: `pre_tech/` is the rollback.

    python scripts/quarantine_pre_tech.py models/spec_0-1_speed            # dry run: lists what would move
    python scripts/quarantine_pre_tech.py models/spec_0-1_speed --apply    # moves them

Only `*.zip` files whose `data` member reads as the v1 layout move. v2 files, zips of any other layout or none,
`best.json`, `env_config.yaml` and the `explore_*.npz` archives stay. Nothing moves if any destination already
exists. Run it ONLY with the stage's trainer stopped (the S7 runbook in docs/commands.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.ckpt_layout import checkpoint_layout  # noqa: E402

QUARANTINE = "pre_tech"


def plan_moves(model_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(moves, other): every v1 zip -> pre_tech/<name>, and the zips that are neither v1 nor v2."""
    moves: list[tuple[Path, Path]] = []
    other: list[Path] = []
    for path in sorted(Path(model_dir).glob("*.zip")):
        layout = checkpoint_layout(path)
        if layout == "v1":
            moves.append((path, Path(model_dir) / QUARANTINE / path.name))
        elif layout is None:
            other.append(path)
    return moves, other


def apply_moves(moves: list[tuple[Path, Path]]) -> None:
    """All or nothing: every destination is checked before the first file moves."""
    clashes = [dst for _, dst in moves if dst.exists()]
    if clashes:
        raise FileExistsError("already in the quarantine, nothing moved: " + ", ".join(str(c) for c in clashes))
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--apply", action="store_true", help="move them (without it: list only)")
    a = ap.parse_args()
    moves, other = plan_moves(a.model_dir)
    for src, dst in moves:
        print(f"{'MOVE' if a.apply else 'would move'} {src.name} -> {dst.parent.name}/{dst.name}")
    for path in other:
        print(f"left in place (not a v1/v2 campaign checkpoint, or unreadable): {path.name}")
    if not moves:
        print("nothing to move")
        return 0
    if a.apply:
        apply_moves(moves)
        print(f"moved {len(moves)} file(s) into {a.model_dir / QUARANTINE}")
    else:
        print(f"{len(moves)} file(s) would move; pass --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: `.gitignore`** -- directly after the line `python/models/spec_*/ckpt_*.zip`, add:

```text
# S7 (2026-09-23): the 479-wide checkpoints scripts/quarantine_pre_tech.py MOVES aside at the tech break. Kept on
# disk as the rollback, never committed (up to hundreds of ~12 MB files).
python/models/spec_*/pre_tech/
```

- [ ] **Step 5: Run** `PY tests\test_quarantine_pre_tech.py` → `4 tests passed`.

- [ ] **Step 6: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add .gitignore python/scripts/quarantine_pre_tech.py python/tests/test_quarantine_pre_tech.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7: quarantine_pre_tech.py moves 479-wide checkpoints to pre_tech/ (never deletes)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 7: Pinned action rows at train time, and absolute entropy dims

**Files:**
- Modify: `python/ultrakill_ai/training.py`
- Create: `python/tests/test_tech_training.py`

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_tech_training.py`:

```python
"""Training-side S7 pieces: pinned (gradient-masked) action rows, and entropy dims named by ABSOLUTE index.

No game:  python tests/test_tech_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transfer_weights import SpacesEnv  # noqa: E402

from ultrakill_ai.env import EnvConfig  # noqa: E402
from ultrakill_ai.spaces import ObsLayout, action_space, pinned_action_rows  # noqa: E402
from ultrakill_ai.training import ENTROPY_DIMS, apply_tech_training, dim_entropies, pin_action_rows  # noqa: E402

S7_ROWS = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]


def tiny(tech: bool) -> PPO:
    torch.manual_seed(0)
    return PPO("MlpPolicy", SpacesEnv(ObsLayout(campaign=True, tech=tech).space(),
                                      action_space(campaign=True, tech=tech)),
               policy_kwargs={"net_arch": [16, 16]}, device="cpu")


def obs(width: int) -> torch.Tensor:
    return torch.as_tensor(np.random.default_rng(0).normal(size=(32, width)).astype(np.float32))


def test_pinned_rows_do_not_move_under_adam_and_the_others_do():
    model = tiny(tech=True)
    rows = pinned_action_rows({1}, False, False)
    assert rows == S7_ROWS and len(pin_action_rows(model.policy, rows)) == 2
    w0 = model.policy.action_net.weight.detach().clone()
    b0 = model.policy.action_net.bias.detach().clone()
    batch = obs(530)
    for _ in range(3):  # a loss that touches every logit, entropy bonus included
        latent, _ = model.policy.mlp_extractor(batch)
        loss = (model.policy.action_net(latent).square().mean()
                - model.policy.get_distribution(batch).entropy().mean())
        model.policy.optimizer.zero_grad()
        loss.backward()
        model.policy.optimizer.step()
    w1, b1 = model.policy.action_net.weight.detach(), model.policy.action_net.bias.detach()
    assert torch.equal(w1[rows], w0[rows]) and torch.equal(b1[rows], b0[rows]), "pinned rows moved"
    live = [r for r in range(57) if r not in rows]
    assert not torch.equal(w1[live], w0[live]) and not torch.equal(b1[live], b0[live])


def test_apply_tech_training_pins_the_s7_rows_and_nothing_under_v1():
    assert apply_tech_training(tiny(tech=True), EnvConfig(mode="campaign", tech_layout="v2")) == S7_ROWS
    assert apply_tech_training(tiny(tech=False), EnvConfig(mode="campaign")) == []
    opened = EnvConfig(mode="campaign", tech_layout="v2", macro_ssj_wall=True, variant_switching=True,
                       hook_action=True)
    assert apply_tech_training(tiny(tech=True), opened) == [48, 49, 50]


def test_the_entropy_dims_are_absolute_and_name_v1_exactly_as_before():
    assert ENTROPY_DIMS == {9: "yaw", 10: "pitch", 11: "look_mode", 12: "macro"}
    before = {-3: "yaw", -2: "pitch", -1: "look_mode"}  # the pre-S7 names, from the END of a 12-dim action
    assert {12 + k: v for k, v in before.items()} == {k: v for k, v in ENTROPY_DIMS.items() if k < 12}
    v1 = dim_entropies(tiny(tech=False).policy.get_distribution(obs(479)).distribution)
    v2 = dim_entropies(tiny(tech=True).policy.get_distribution(obs(530)).distribution)
    assert set(v1) == {"yaw", "pitch", "look_mode"} and set(v2) == {"yaw", "pitch", "look_mode", "macro"}


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Run to verify it fails**

Run: `PY tests\test_tech_training.py` → FAIL, `ImportError: cannot import name 'apply_tech_training'`.

- [ ] **Step 3: `training.py`**

(a) Imports: `from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, tech_gates` and
`from ultrakill_ai.spaces import pinned_action_rows`.

(b) Replace `ENTROPY_DIMS = {-1: "look_mode", -2: "pitch", -3: "yaw"}  # ...` with:

```python
# The action dimensions worth naming, by ABSOLUTE index. They were negative indices from the END until the S7 tech
# break appended three dims after the look mode, which would have labelled `hook` as "look_mode". For a 12-dim
# campaign action 9 / 10 / 11 are exactly the old -3 / -2 / -1 (tests/test_tech_training.py pins it).
ENTROPY_DIMS = {9: "yaw", 10: "pitch", 11: "look_mode", 12: "macro"}
```

(c) Add, after `ENTROPY_SAMPLE`:

```python
def dim_entropies(dists) -> dict[str, float]:
    """Mean entropy of each named action dimension present in a MultiCategorical's list of Categoricals."""
    return {name: float(dists[index].entropy().mean()) for index, name in ENTROPY_DIMS.items() if len(dists) > index}
```

and in `ActionEntropyCallback._on_rollout_end` replace the dict comprehension that sets `self.entropies` with
`self.entropies = dim_entropies(dists)`.

(d) Add, after `resume_refusal`:

```python
def pin_action_rows(policy, rows) -> list:
    """Freezes the listed rows of `policy.action_net` (weights and bias) by zeroing their gradient.

    WHY (the plan's "Spec deviations" 2): a head that can never reach the game -- a reserved macro value, the whole
    `variant` dim before S9, `hook` before S10 -- gets no advantage signal, only the entropy bonus, and Adam moves a
    consistently-signed gradient ~lr per minibatch step whatever its size. Unpinned, those heads diffuse toward
    uniform within a few million steps: the §4.4 priors are gone by the time the stage that opens them arrives, and
    their free entropy satisfies the entropy floor while the live dims collapse. A zero gradient on rows whose Adam
    moments are zero (add_tech_heads.py pads them so) makes Adam's update for those rows exactly 0.0.

    Tensor hooks do not survive save/load: `main` re-applies them on every start from the config.
    """
    rows = sorted({int(r) for r in rows})
    if not rows:
        return []
    net = policy.action_net
    keep = torch.ones(net.out_features, dtype=net.weight.dtype, device=net.weight.device)
    keep[rows] = 0.0
    return [net.weight.register_hook(lambda grad, k=keep.unsqueeze(1): grad * k),
            net.bias.register_hook(lambda grad, k=keep: grad * k)]


def apply_tech_training(model, env_cfg: EnvConfig) -> list[int]:
    """tech_layout v2 only: pins every action row its gates keep off the wire. Returns the rows, for the log."""
    if env_cfg.tech_layout != "v2":
        return []
    gates = tech_gates(env_cfg)
    rows = pinned_action_rows(gates.macros, gates.variant, gates.hook)
    pin_action_rows(model.policy, rows)
    return rows
```

(e) In `main()`, directly after the `print(hyperparams_in_force(model, hyper), flush=True)` line, add:

```python
    pinned = apply_tech_training(model, env_cfg)
    if env_cfg.tech_layout == "v2":
        print(f"tech layout v2: {model.observation_space.shape[0]} inputs, {len(model.action_space.nvec)} action "
              f"dims ({int(model.action_space.nvec.sum())} logits); pinned action rows {pinned}", flush=True)
```

- [ ] **Step 4: Run** `PY tests\test_tech_training.py` → `3 tests passed`; `PY tests\test_resume_hyperparams.py` →
  passes unchanged.

- [ ] **Step 5: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/training.py python/tests/test_tech_training.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 training: pin the gated action rows (gradient mask), entropy dims by absolute index" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 8: `ent_floor` re-basing and the S8 `tech` bonus -- built now, OFF

**Files:**
- Modify: `python/scripts/campaign_driver.py`, `python/ultrakill_ai/rewards.py`, `python/ultrakill_ai/env.py`,
  `python/ultrakill_ai/training.py`
- Modify: `python/tests/test_speed_overrides.py` (append), `python/tests/test_campaign_rewards.py` (one pinned list)
- Create: `python/tests/test_tech_bonus.py`

- [ ] **Step 1: Append the failing `ent_floor` route tests** to `test_speed_overrides.py` (above `__main__`)

```python
# ---------------------------------------------------------------------------------------------------------
# S7: `ent_floor` / `ent_coef_max` through `speed.train:` (plan 2026-09-23, Task 8)
# ---------------------------------------------------------------------------------------------------------


def test_the_train_block_can_rebase_the_entropy_floor_of_a_speed_stage_only(tmp_path=None):
    """`ent_floor` is a TOP-LEVEL train key read by training.main, never a PPO hyperparameter."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    p = campaign_driver.load_plan(write_plan(tmp_path, train={"ent_floor": 7.8986}))
    speed = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]
    complete = campaign_driver.stage_config(p, "Level 0-1")["train"]
    assert speed["ent_floor"] == 7.8986 and complete["ent_floor"] == 6.5
    assert "ent_floor" not in speed["hyperparams"], "PPO(**hyperparams) would reject it"
    assert speed["hyperparams"] == complete["hyperparams"], "an ent_floor-only block moves no hyperparameter"
    assert complete == campaign_driver.stage_config(plan(), "Level 0-1")["train"]


def test_the_entropy_floor_does_not_leak_between_stage_kinds_in_either_order(tmp_path=None):
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    p = campaign_driver.load_plan(write_plan(tmp_path, train={"ent_floor": 7.8986, "gamma": 0.999}))
    first = campaign_driver.stage_config(p, "Level 0-1")["train"]
    campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)
    again = campaign_driver.stage_config(p, "Level 0-1")["train"]
    assert first == again and again["ent_floor"] == 6.5 and again["hyperparams"]["gamma"] == 0.998
    speed = campaign_driver.stage_config(p, "Level 0-1", kind=SPEED)["train"]
    assert speed["ent_floor"] == 7.8986 and speed["hyperparams"]["gamma"] == 0.999


def test_a_bad_entropy_floor_is_refused_not_defaulted(tmp_path=None):
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    for block in ({"ent_floor": "7.9"}, {"ent_floor": True}, {"ent_flor": 7.9}):
        try:
            campaign_driver.load_plan(write_plan(tmp_path, train=block))
        except ValueError as exc:
            assert "speed.train" in str(exc), exc
        else:
            raise AssertionError("accepted %r" % (block,))
```

- [ ] **Step 2: Create the failing `python/tests/test_tech_bonus.py`**

```python
"""S8 -- the `tech` bonus of spec §4.5, built DORMANT. No game:  python tests/test_tech_bonus.py

Paid only when a sent macro RAN and the mod reported an accepted SSJ bucket, the player lost no HP and did not die,
the per-episode cap is unexhausted, and the trainer's decay scale is above zero. Inert at the default weight 0.0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_tech_env import REFUSED, V08_FEATURES, FakeTechLevel, tech_action, tech_env  # noqa: E402

from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.protocol import BridgeIncompatible  # noqa: E402
from ultrakill_ai.rewards import SPEED_BONUS_MIN, CampaignStep, RewardConfig, compute_reward, tech_scale  # noqa: E402
from ultrakill_ai.training import TechDecayCallback, tech_decay_problem  # noqa: E402


def frames(hp_before: int = 100, hp_after: int = 100):
    return {"player": {"hp": hp_before, "dead": False}}, {"player": {"hp": hp_after, "dead": False}}


def paid(cfg, prev, cur, *, died=False, landed=1, scale=1.0, room=10.0):
    return compute_reward(cfg, prev, cur, {}, died=died, campaign=CampaignStep(tech_landed=landed),
                          tech_scale=scale, tech_room=room).parts.get("tech")


def test_the_bonus_is_inert_by_default():
    cfg = RewardConfig()
    assert (cfg.tech, cfg.tech_cap, cfg.tech_decay_start, cfg.tech_decay_steps) == (0.0, 10.0, 0.0, 3_000_000.0)
    assert paid(cfg, *frames()) is None


def test_a_landed_ssj_pays_weight_times_scale():
    assert paid(RewardConfig(tech=0.5), *frames(), scale=0.4) == 0.5 * 0.4


def test_no_pay_on_hp_loss_a_death_no_landing_or_a_spent_decay():
    cfg = RewardConfig(tech=0.5)
    assert paid(cfg, *frames(100, 90)) is None, "the step cost HP"
    assert paid(cfg, *frames(), died=True) is None
    assert paid(cfg, *frames(), landed=0) is None
    assert paid(cfg, *frames(), scale=0.0) is None


def test_the_cap_bounds_what_is_left_of_the_episode():
    cfg = RewardConfig(tech=4.0)
    assert paid(cfg, *frames(), room=10.0) == 4.0 and paid(cfg, *frames(), room=2.5) == 2.5
    assert paid(cfg, *frames(), room=0.0) is None and paid(cfg, *frames(), room=-1.0) is None


def test_the_decay_schedule():
    assert tech_scale(0, 1_000_000, 3_000_000) == 1.0
    assert tech_scale(1_000_000, 1_000_000, 3_000_000) == 1.0
    assert tech_scale(2_500_000, 1_000_000, 3_000_000) == 0.5
    assert tech_scale(4_000_000, 1_000_000, 3_000_000) == 0.0 and tech_scale(9e9, 1_000_000, 3_000_000) == 0.0
    assert tech_scale(5e7, 0, 0) == 1.0, "no decay window: the full weight"


def test_the_three_invariants_hold_at_the_s8_numbers():
    """§4.5's farmability bound, at the values S8 will switch on (tech 0.5, cap 10) against the live weights."""
    cap, level_complete, death = RewardConfig().tech_cap, 100.0, 12.0
    assert cap < SPEED_BONUS_MIN * level_complete, "finishing > not finishing: the completion floor beats the cap"
    assert cap < death, "living > dying: one death costs more than a whole episode of tech"


def test_the_env_pays_on_landed_ssjs_and_stops_at_the_cap():
    env, fake = tech_env(rewards=RewardConfig(tech=4.0, tech_cap=10.0, time=0.01))
    try:
        env.reset()
        assert env.set_tech_scale(1.0) == 1.0
        got = []
        for _ in range(4):
            _, _, _, _, info = env.step(tech_action(macro=1))
            got.append(info["reward_parts"].get("tech", 0.0))
        assert got == [4.0, 4.0, 2.0, 0.0]
        fake.macro_outcome = dict(REFUSED)
        env.reset()
        _, _, _, _, info = env.step(tech_action(macro=1))
        assert "tech" not in info["reward_parts"], "a refused macro never pays"
    finally:
        env.close()


def test_a_fresh_env_pays_nothing_until_the_trainer_sets_the_scale():
    env, _ = tech_env(rewards=RewardConfig(tech=4.0))
    try:
        env.reset()
        _, _, _, _, info = env.step(tech_action(macro=1))
        assert "tech" not in info["reward_parts"]
    finally:
        env.close()


def test_a_tech_reward_needs_the_ssj_instrument_and_the_v2_layout():
    without = [f for f in V08_FEATURES if f != "ssj_instrument"]
    env, _ = tech_env(level_cls=lambda: FakeTechLevel(features=without), rewards=RewardConfig(tech=0.5))
    try:
        env.reset()
    except BridgeIncompatible as exc:
        assert "ssj_instrument" in str(exc)
    else:
        raise AssertionError("a bucket-gated reward against a dead instrument pays 0 fleet-wide, silently")
    finally:
        env.close()
    try:
        UltrakillEnv(EnvConfig(mode="campaign", rewards=RewardConfig(tech=0.5))).close()
    except ValueError as exc:
        assert "tech_layout" in str(exc)
    else:
        raise AssertionError("a tech reward on the v1 layout can never pay")


def test_the_decay_callback_pushes_the_scale_into_every_env():
    calls, records = [], {}

    class FakeVecEnv:
        def env_method(self, name, *args):
            calls.append((name, args))
            return [None] * 12

    class FakeLogger:
        def record(self, key, value):
            records[key] = value

    vec = FakeVecEnv()
    cb = TechDecayCallback(1_000_000, 3_000_000)
    cb.model = SimpleNamespace(num_timesteps=2_500_000, get_env=lambda: vec, logger=FakeLogger())
    cb.on_rollout_start()
    assert calls == [("set_tech_scale", (0.5,))] and records["train/tech_scale"] == 0.5


def test_a_tech_bonus_without_a_decay_start_is_refused():
    assert tech_decay_problem(EnvConfig(mode="campaign")) is None
    on = EnvConfig(mode="campaign", tech_layout="v2", rewards=RewardConfig(tech=0.5))
    assert "tech_decay_start" in tech_decay_problem(on)
    ok = EnvConfig(mode="campaign", tech_layout="v2", rewards=RewardConfig(tech=0.5, tech_decay_start=66_000_000))
    assert tech_decay_problem(ok) is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 3: Run both to verify they fail**

Run: `PY tests\test_speed_overrides.py` → FAIL (`unknown speed.train settings ['ent_floor']`).
Run: `PY tests\test_tech_bonus.py` → FAIL (`ImportError: cannot import name 'tech_scale'`).

- [ ] **Step 4: `campaign_driver.py` -- route `ent_floor` / `ent_coef_max`**

(a) After `SPEED_TRAIN_KEYS = (...)`, add:

```python
# ... and the two TOP-LEVEL `train:` keys a speed stage may re-base (2026-09-23, the S7 tech break): the entropy
# floor is read by training.main from the top of `train:` (EntropyFloorCallback), not passed to PPO, so it cannot
# ride in `hyperparams`. The new heads add 1.3986 nats; the floor moves with them (spec §4.4).
SPEED_TRAIN_TOP_KEYS = ("ent_floor", "ent_coef_max")
```

(b) In `_speed_train`, replace the `unexpected = ...` check with:

```python
    allowed = set(SPEED_TRAIN_KEYS) | set(SPEED_TRAIN_TOP_KEYS)
    unexpected = sorted(set(map(str, block)) - allowed)
    if unexpected:
        raise ValueError("%s: unknown speed.train settings %s (expected %s)"
                         % (path, unexpected, list(SPEED_TRAIN_KEYS) + list(SPEED_TRAIN_TOP_KEYS)))
```

(c) In `stage_config`, replace:

```python
    if kind == SPEED and plan.speed_train:
        # REBIND, NEVER MUTATE, ...
        train["hyperparams"] = {**(train.get("hyperparams") or {}), **plan.speed_train}
```

(keep its comment) with:

```python
    if kind == SPEED and plan.speed_train:
        # REBIND, NEVER MUTATE, exactly as `rewards` above: `train` is a SHALLOW copy of `plan.train`, so
        # `train["hyperparams"]` IS the plan's own dict and an in-place update would leak this stage's gamma
        # into every complete stage generated afterwards from the same Plan object. Pinned by
        # `test_the_train_block_does_not_leak_between_stage_kinds_in_either_order` in tests/test_speed_overrides.py.
        hyper = {k: v for k, v in plan.speed_train.items() if k not in SPEED_TRAIN_TOP_KEYS}
        if hyper:
            train["hyperparams"] = {**(train.get("hyperparams") or {}), **hyper}
        # `ent_floor` / `ent_coef_max` sit at the TOP of `train:`. Assigning a key of the fresh `train` dict
        # rebinds nothing the plan shares.
        for key in SPEED_TRAIN_TOP_KEYS:
            if key in plan.speed_train:
                train[key] = plan.speed_train[key]
```

- [ ] **Step 5: `rewards.py` -- the weights, the schedule, the paid term**

(a) In `RewardConfig`, after `item_placed`, add:

```python
    # THE TECH BONUS -- stage S8 of docs/superpowers/specs/2026-09-20-speedrun-tech.md §4.5. 0.0 (the default,
    # and every run until S8 is switched on) is inert. Paid per decision on which a sent macro RAN and the mod
    # reported an accepted SSJ bucket (`macro_landed`), the player lost no HP and did not die, until `tech_cap` has
    # been paid this episode -- times the trainer's decay scale: 1.0 until `tech_decay_start` (CUMULATIVE steps,
    # the run's own count at switch-on), then linearly to 0.0 over `tech_decay_steps`. Needs tech_layout v2 and
    # the mod's `ssj_instrument` feature; a refusal is never charged (§4.2).
    tech: float = 0.0
    tech_cap: float = 10.0
    tech_decay_start: float = 0.0
    tech_decay_steps: float = 3_000_000.0
```

(b) In `CampaignStep`, add the field
`tech_landed: int = 0  # 1 when this step's macro landed an accepted SSJ bucket (macro_landed); set by the env`.

(c) Add after `macro_landed`:

```python
def tech_scale(num_timesteps: float, start: float, steps: float) -> float:
    """The S8 decay: 1.0 up to `start` (cumulative steps), then linearly to 0.0 over `steps`; 1.0 if `steps` <= 0."""
    if steps <= 0 or num_timesteps <= start:
        return 1.0
    return max(0.0, 1.0 - (num_timesteps - start) / steps)
```

(d) `compute_reward`: add the keyword arguments `tech_scale: float = 0.0, tech_room: float = 0.0` after
`official_seconds`, and directly after the lines

```python
    r.add("damage_taken", -cfg.damage_taken * hp_lost)
    if died:
        r.add("death", -cfg.death)
```

add:

```python
    if (campaign is not None and campaign.tech_landed and cfg.tech > 0 and tech_scale > 0
            and hp_lost <= 0 and not died):
        # S8. Never on a step that cost HP or a life (§4.5's third invariant), never past the per-episode cap
        # (`tech_room` is what the env has left of `tech_cap`), and nothing once the trainer's decay has run out.
        r.add("tech", min(cfg.tech * tech_scale * campaign.tech_landed, max(0.0, tech_room)))
```

(Inside `compute_reward` the argument `tech_scale` shadows the module function of the same name; nothing in the
function body calls the function, so that is harmless -- say so in a comment on the parameter.)

(e) `tests/test_campaign_rewards.py::test_route_terms_are_retired` pins `compute_reward`'s parameter list; extend
its expected list (and add this file to Step 9's `git add`):

```python
        "target_seconds", "official_seconds",
        # S8's two (plan 2026-09-23, Task 8): the trainer's decay scale and what is left of the episode's cap.
        "tech_scale", "tech_room"]
```

- [ ] **Step 6: `env.py` -- pay it**

(a) Import `macro_landed` from `ultrakill_ai.rewards` (next to `compute_reward`).

(b) In `__init__`, right after the S7 lines added in Task 2 step 5, add:

```python
        if self.cfg.rewards.tech > 0 and not tech:
            raise ValueError("rewards.tech pays for macros, which only tech_layout v2 sends")
        self._tech_scale = 0.0  # S8: set by training.TechDecayCallback through env_method; 0.0 pays nothing
        self._tech_paid = 0.0  # this episode's tech reward so far, against `tech_cap`
```

(c) `_required_features` becomes:

```python
    def _required_features(self) -> set[str]:
        """The `hello.features` this config cannot run without. Empty under v1: a 0.7.2 DLL serves v1 as today."""
        need = set(TECH_LAYOUT_FEATURES) if self._tech else set()
        if self.cfg.rewards.tech > 0:
            # docs/protocol.md: a reward keyed on `macro.ssj_bucket` must assert the instrument at connect. If the
            # TrySSJ patch failed to apply, every bucket reads -1 and the bonus would pay 0 fleet-wide, silently.
            need |= {"ssj_instrument", "macro.ssj"}
        return need
```

(d) Add the method (next to `_note_macro_result`):

```python
    def set_tech_scale(self, scale: float) -> float:
        """The S8 decay scale, pushed by training.TechDecayCallback at every rollout start. Returns it, clamped."""
        self._tech_scale = max(0.0, min(1.0, float(scale)))
        return self._tech_scale
```

(e) In `_reset`, next to `self._macro_reasons = {}`, add `self._tech_paid = 0.0`.

(f) In `_step`, directly after `campaign_step = self._campaign_progress(prev, cur, died=died) if campaign else None`:

```python
        if campaign_step is not None and macro_landed(macro_report):
            campaign_step.tech_landed = 1
```

and give the `compute_reward(...)` call two more arguments,
`tech_scale=self._tech_scale, tech_room=self.cfg.rewards.tech_cap - self._tech_paid`, then directly after it:

```python
        self._tech_paid += reward.parts.get("tech", 0.0)
```

- [ ] **Step 7: `training.py` -- the decay callback and its refusal**

(a) Import `from ultrakill_ai.rewards import tech_scale`.

(b) Add after `apply_tech_training`:

```python
class TechDecayCallback(BaseCallback):
    """Pushes the S8 `tech` bonus's decay scale into every env at each rollout start (spec §4.5).

    Keyed on the model's CUMULATIVE step count and the config's absolute `tech_decay_start`, so a driver round
    boundary or a crash restart continues the schedule instead of restarting it -- which a per-env counter could
    not do, since every env is rebuilt with the trainer. Recorded as `train/tech_scale`.
    """

    def __init__(self, start: float, steps: float):
        super().__init__()
        self.start = float(start)
        self.steps = float(steps)
        self.scale: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.scale = tech_scale(self.model.num_timesteps, self.start, self.steps)
        self.training_env.env_method("set_tech_scale", self.scale)
        self.logger.record("train/tech_scale", self.scale)


def tech_decay_problem(env_cfg: EnvConfig) -> str | None:
    """A `tech` bonus with a decay window but no start would be fully decayed at a 60M-step resume. Refused."""
    r = env_cfg.rewards
    if r.tech > 0 and r.tech_decay_steps > 0 and r.tech_decay_start <= 0:
        return ("rewards.tech is on but rewards.tech_decay_start is 0: set it to the run's cumulative step count "
                "at switch-on (status.json `timesteps`), or the bonus is already fully decayed")
    return None
```

(c) In `main()`, right after the `resume_refusal` block:

```python
    problem = tech_decay_problem(env_cfg)
    if problem:
        print(f"REFUSING TO START: {problem}", flush=True)
        raise SystemExit(3)
```

and in the `CallbackList([...])`, insert before `progress,`:

```python
        *([TechDecayCallback(env_cfg.rewards.tech_decay_start, env_cfg.rewards.tech_decay_steps)]
          if env_cfg.rewards.tech > 0 else []),
```

with a print beside the entropy-floor print:

```python
    if env_cfg.rewards.tech > 0:
        r = env_cfg.rewards
        print(f"tech bonus: {r.tech} per landed SSJ, cap {r.tech_cap} per episode, full until "
              f"{r.tech_decay_start:,.0f} steps then to 0 over {r.tech_decay_steps:,.0f}", flush=True)
```

- [ ] **Step 8: Run the tests**

Run: `PY tests\test_tech_bonus.py` → `11 tests passed`
Run: `PY tests\test_speed_overrides.py` → all pass, 3 more than before.
Run: `PY tests\test_campaign_rewards.py` → passes with the extended list (it FAILS without step 5(e), which is the
pin doing its job). `PY tests\test_speed_death_weight.py`, `PY tests\test_speed_oob_weight.py`,
`PY tests\test_speed_fall_hp_weight.py`, `PY tests\test_tech_env.py` → pass unchanged.

- [ ] **Step 9: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/scripts/campaign_driver.py python/ultrakill_ai/rewards.py python/ultrakill_ai/env.py python/ultrakill_ai/training.py python/tests/test_speed_overrides.py python/tests/test_campaign_rewards.py python/tests/test_tech_bonus.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S8 dormant: tech bonus (bucket-gated, capped, trainer-decayed, off at 0.0); speed.train may re-base ent_floor" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 9: `probe_rollout.py` absolute indices and v2 records; `macro_check.py` for v2

**Files:**
- Modify: `python/scripts/probe_rollout.py`, `python/scripts/macro_check.py`
- Create: `python/tests/test_probe_tech.py`, `python/tests/test_macro_check.py`

- [ ] **Step 1: Write the failing probe tests**

Create `python/tests/test_probe_tech.py`:

```python
"""scripts/probe_rollout.py after the S7 break: ABSOLUTE target indices, v2 action / obs / macro in each record,
and a checkpoint probed under its own layout. No game:  python tests/test_probe_tech.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_rollout  # noqa: E402
from test_campaign_env import forward, make_env  # noqa: E402
from test_ckpt_layout import fake_ckpt  # noqa: E402
from test_tech_env import tech_action, tech_env  # noqa: E402
from test_tech_layout import PIN_ENEMY_MAX, PIN_EXPLORE, PIN_RAW, PIN_TARGET  # noqa: E402

from ultrakill_ai.ckpt_layout import CAMPAIGN_NVEC_V1, CAMPAIGN_NVEC_V2  # noqa: E402
from ultrakill_ai.spaces import LOOK_MODE_INDEX, ObsLayout, action_space, campaign_block, pack_observation  # noqa: E402


def test_the_target_slice_is_absolute_and_reads_the_same_slots_in_both_layouts():
    assert probe_rollout.TARGET_SLICE == slice(448, 456)
    assert (probe_rollout.TECH_SLICE_A, probe_rollout.TECH_SLICE_D) == (slice(479, 491), slice(519, 522))
    expected = campaign_block(PIN_RAW, PIN_EXPLORE, PIN_TARGET)[5:13]
    for tech in (False, True):
        vector = pack_observation(PIN_RAW, ObsLayout(campaign=True, tech=tech), PIN_ENEMY_MAX, PIN_EXPLORE,
                                  PIN_TARGET)
        assert np.allclose(vector[probe_rollout.TARGET_SLICE], expected), tech
    v2 = pack_observation(PIN_RAW, ObsLayout(campaign=True, tech=True), PIN_ENEMY_MAX, PIN_EXPLORE, PIN_TARGET)
    assert not np.allclose(v2[slice(-36 + 5, -36 + 13)], expected), "the old negative slice reads the TECH block"


def test_the_scripted_driver_fills_the_whole_action_width():
    for tech, width in ((False, 12), (True, 15)):
        stub = SimpleNamespace(action_space=action_space(campaign=True, tech=tech),
                               gates=SimpleNamespace(target=None))
        a = probe_rollout.scripted_action({}, stub, {})
        assert len(a) == width and a[LOOK_MODE_INDEX] == 2


def test_a_v2_step_record_carries_the_heads_blocks_a_and_d_and_the_macro_report():
    env, _ = tech_env()
    try:
        env.reset()
        a = tech_action(macro=1, variant=2, hook=1)
        obs, reward, _, _, info = env.step(a)
        rec = probe_rollout.step_record(0, env._raw, a, obs, reward, info, env, {}, None)
        assert (rec["act"]["macro"], rec["act"]["variant"], rec["act"]["hook"]) == ("ssj", 2, True)
        assert rec["obs_macro"] == [1.0, 0.0, 0.3333] and len(rec["obs_move_tech"]) == 12
        assert rec["macro"]["result"] == "ran" and rec["macro"]["ssj_bucket"] == 1
        assert rec["obs_target"] == [round(float(v), 4) for v in obs[448:456]]
        json.dumps(rec)
    finally:
        env.close()


def test_a_v1_step_record_keeps_its_shape():
    env, _ = make_env()
    try:
        env.reset()
        a = forward()
        obs, reward, _, _, info = env.step(a)
        rec = probe_rollout.step_record(0, env._raw, a, obs, reward, info, env, {}, None)
        assert set(rec["act"]) == {"forward", "side", "buttons", "slot", "yaw_deg", "pitch_deg", "look_mode"}
        assert not {"obs_move_tech", "obs_macro", "macro"} & set(rec)
    finally:
        env.close()


def test_the_probe_runs_a_checkpoint_under_its_own_layout():
    with tempfile.TemporaryDirectory() as tmp:
        v1 = fake_ckpt(Path(tmp) / "a.zip", 479, CAMPAIGN_NVEC_V1)
        v2 = fake_ckpt(Path(tmp) / "b.zip", 530, CAMPAIGN_NVEC_V2)
        assert probe_rollout.resolve_tech_layout("v1", None, str(v2)) == "v2"
        assert probe_rollout.resolve_tech_layout("v1", "v2", None) == "v2"
        assert probe_rollout.resolve_tech_layout("v2", None, None) == "v2"
        try:
            probe_rollout.resolve_tech_layout("v1", "v2", str(v1))
        except SystemExit as exc:
            assert "v1 checkpoint" in str(exc)
        else:
            raise AssertionError("a v1 checkpoint was probed under v2")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 2: Write the failing macro_check tests**

Create `python/tests/test_macro_check.py`:

```python
"""scripts/macro_check.py's new pure helpers (S6 re-measure before the S7 install). No game, no socket:
python tests/test_macro_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import macro_check  # noqa: E402

RESERVED = {"result": "disabled", "reason": "reserved"}


def test_the_refusal_verdict():
    ok = macro_check.refusal_verdict([dict(RESERVED)] * 4, [dict(RESERVED)] * 3)
    assert ok["ok"] and ok["requested"] == 7 and ok["bad"] == []
    for bad in ({"result": "ran", "reason": None}, None, {"result": "refused", "reason": "not_sliding"}):
        verdict = macro_check.refusal_verdict([dict(RESERVED)] * 3 + [bad], [dict(RESERVED)] * 3)
        assert not verdict["ok"] and len(verdict["bad"]) == 1, bad


def test_the_slam_verdict():
    rest = [{"heavy_fall": False, "slam_force": 0.0, "bounce_window": False, "pre_slide_speed": 0.5}] * 5
    assert not macro_check.slam_verdict(rest)["ok"], "a fall that never slammed proves nothing"
    slam = rest + [{"heavy_fall": True, "slam_force": 3.0}, {"heavy_fall": True, "slam_force": 6.1},
                   {"heavy_fall": False, "bounce_window": True, "pre_slide_speed": 6.1}]
    v = macro_check.slam_verdict(slam)
    assert v["ok"] and v["slam_force_max"] == 6.1 and v["pre_slide_speed_max"] == 6.1


def test_the_field_by_field_diff():
    a = [{"frame": 10, "time": 1.0, "player": {"pos": [1.0, 2.0, 3.0], "hp": 100}, "rays": [5.0, 6.0]}]
    same = [{"frame": 99, "time": 7.0, "player": {"pos": [1.0, 2.0, 3.0004], "hp": 100}, "rays": [5.0, 6.0]}]
    assert macro_check.diff_records(a, same) == [], "frame / time ignored, 4e-4 inside the tolerance"
    moved = [{"frame": 10, "time": 1.0, "player": {"pos": [1.0, 2.0, 3.5], "hp": 100}, "rays": [5.0, 6.0]}]
    assert any("player.pos.2" in d for d in macro_check.diff_records(a, moved))
    extra = [{**a[0], "move_tech": {"heavy_fall": False}}]
    assert any("move_tech.heavy_fall" in d for d in macro_check.diff_records(a, extra)), "a new key is a diff"
    assert any("length" in d for d in macro_check.diff_records(a, a + a))


def test_the_legacy_script_is_fixed_and_input_only():
    script = macro_check.legacy_script(60)
    assert len(script) == 60 and script == macro_check.legacy_script(60)
    assert all(set(step) == {"move", "buttons"} for step in script), "no macro, no variant: a 0.7.2 client"


def test_the_bridge_still_refuses_the_fleet_ports():
    try:
        macro_check.Bridge(47805)
    except SystemExit as exc:
        assert "47812" in str(exc)
    else:
        raise AssertionError("macro_check connected to a training port")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
```

- [ ] **Step 3: Run both to verify they fail**

Run: `PY tests\test_probe_tech.py` → FAIL (`AttributeError: ... has no attribute 'TECH_SLICE_A'`).
Run: `PY tests\test_macro_check.py` → FAIL (`AttributeError: ... 'refusal_verdict'`).

- [ ] **Step 4: `probe_rollout.py`**

(a) Replace the spaces import with
`from ultrakill_ai.spaces import BUTTONS, HOOK_INDEX, LOOK_MODE_INDEX, MACRO_INDEX, MACROS, PITCH_BINS, VARIANT_INDEX, YAW_BINS, ObsLayout  # noqa: E402`
and replace

```python
# The campaign block is the tail of the observation; the target occupies its slots 5..12.
CAMPAIGN_BLOCK = 36
TARGET_SLICE = slice(-CAMPAIGN_BLOCK + 5, -CAMPAIGN_BLOCK + 13)
```

with

```python
# The eight observation slots the route target occupies -- the campaign block's 5..12 -- as ABSOLUTE indices
# (448-455). They were negative indices from the end until 2026-09-23, which read the wrong slots the moment the
# S7 TECH block was appended after the campaign block (spec §4.3). Blocks A and D are v2 only.
_V1 = ObsLayout(campaign=True)
TARGET_SLICE = slice(_V1.campaign_start + 5, _V1.campaign_start + 13)
TECH_SLICE_A = slice(_V1.tech_start, _V1.tech_start + 12)
TECH_SLICE_D = slice(_V1.tech_start + 40, _V1.tech_start + 43)
```

(b) In `step_record`, directly after the `rec = {...}` literal and before `rec.update(world_state(raw, pos))`:

```python
    if len(a) > HOOK_INDEX:
        # tech_layout v2: the three new heads as the POLICY chose them. What the gates actually sent is in `beh`
        # (macro_request / macro_sent / variant_request / hook_request for this decision).
        rec["act"].update(macro=MACROS[a[MACRO_INDEX]], variant=a[VARIANT_INDEX], hook=bool(a[HOOK_INDEX]))
    vector = np.asarray(obs)
    if vector.shape[0] >= TECH_SLICE_D.stop:
        rec["obs_move_tech"] = [round(float(v), 4) for v in vector[TECH_SLICE_A]]
        rec["obs_macro"] = [round(float(v), 4) for v in vector[TECH_SLICE_D]]
    report = raw.get("macro")
    if isinstance(report, dict):
        rec["macro"] = {k: report.get(k) for k in ("requested", "result", "reason", "note", "ssj_bucket",
                                                   "ssj_landed")}
```

(c) In `scripted_action`, replace `a = np.zeros(12, dtype=np.int64)` with
`a = np.zeros(len(env.action_space.nvec), dtype=np.int64)  # 12 under v1, 15 under v2; the tech heads stay 0`.

(d) Add after `probe_config`:

```python
def resolve_tech_layout(config_layout: str, requested: str | None, model: str | None) -> str:
    """The layout a probe runs under: a checkpoint's OWN layout when --model is given (whatever the stage config
    says -- the S6 check probes a migrated scratch copy under a still-v1 stage config), else --tech-layout, else the
    config's. A --tech-layout that contradicts the checkpoint is refused."""
    from ultrakill_ai.ckpt_layout import checkpoint_layout  # noqa: PLC0415 - stdlib only

    own = checkpoint_layout(model) if model else None
    if requested and own and requested != own:
        raise SystemExit(f"--tech-layout {requested}, but {model} is a {own} checkpoint")
    return own or requested or config_layout
```

(e) In `main()`: add the argument

```python
    ap.add_argument("--tech-layout", choices=("v1", "v2"),
                    help="probe under this layout (e.g. S6's v2 check before the install); a --model's own wins")
```

and directly after `cfg = probe_config(...)`, add
`cfg.tech_layout = resolve_tech_layout(cfg.tech_layout, args.tech_layout, None if args.scripted else args.model)`.

- [ ] **Step 5: `macro_check.py`** (stays standalone: stdlib only, no `ultrakill_ai` import)

(a) Add two methods to `Bridge`, after `reset`:

```python
    def teleport(self, pos) -> dict:
        """Takes control, moves the player there and zeroes its velocity (docs/protocol.md)."""
        return self._send({"type": "teleport", "pos": [float(v) for v in pos]})

    def kill(self) -> dict:
        """The protocol's debug kill: a lethal GetHurt, so a recording covers a death and the reset after it."""
        return self._send({"type": "kill"})
```

(b) Add a new section before `def summarise`:

```python
# ---------------------------------------------------------------------------------------------------
# S6 before the S7 install (docs/superpowers/plans/2026-09-23-tech-break-python.md, Task 11)

RESERVED_MACROS = (2, 3, 4, 5)
VARIANT_VALUES = (1, 2, 3)
DIFF_IGNORED = frozenset({"frame", "time", "mem"})  # differ between two game processes by construction
DIFF_TOL = 1e-3


def refusal_verdict(macro_reports: list, variant_reports: list) -> dict:
    """Pure. Under the training defaults every reserved request must come back NOT run, reason `reserved`."""
    bad = [{"kind": kind, "report": report}
           for kind, reports in (("macro", macro_reports), ("variant", variant_reports))
           for report in reports
           if not isinstance(report, dict) or report.get("result") == "ran" or report.get("reason") != "reserved"]
    return {"requested": len(macro_reports) + len(variant_reports), "bad": bad, "ok": not bad}


def refusal_check(b: Bridge) -> dict:
    """The mod half of S7's behaviour-preservation argument: with the defaults the Python env sends on every
    connect, macros 2-5 and every variant are refused as `reserved` even if one ever reached the wire."""
    b.config(macro_ssj_wall=False, allow_reserved_macros=False, variant_switching=False)
    try:
        settle(b)
        macro_reports = [b.step(macro=m).get("macro") for m in RESERVED_MACROS]
        variant_reports = [b.step(variant=v).get("variant") for v in VARIANT_VALUES]
    finally:
        b.config(variant_switching=True)  # what main() set for the blocks after this one
    return refusal_verdict(macro_reports, variant_reports)


def slam_verdict(frames: list[dict]) -> dict:
    """Pure. Over the `move_tech` blocks of one drop-and-slam trial: did the slam fields block A packs MOVE?"""
    heavy = any(bool(f.get("heavy_fall")) for f in frames)
    force = max((float(f.get("slam_force") or 0.0) for f in frames), default=0.0)
    pre = max((float(f.get("pre_slide_speed") or 0.0) for f in frames), default=0.0)
    bounce = any(bool(f.get("bounce_window")) for f in frames)
    return {"heavy_fall_seen": heavy, "slam_force_max": force, "pre_slide_speed_max": pre,
            "bounce_window_seen": bounce, "ok": heavy and force > 1.0 and bounce}


def slam_trials(b: Bridge, trials: int, height: float = 20.0, steps: int = 45) -> dict:
    """Block A's slam fields against a REAL slam, which the 2026-09-20 run never produced: teleport `height` m up,
    hold slide through the fall (TryStartSlam needs fallTime > 0.5 s and nothing within 3 m below), land, and keep
    stepping ~3 s so the bounce window and the slam slide's preSlideSpeed show. The maxima are the numbers the
    packing scales in spaces.MOVE_TECH_FIELDS were derived for (slam_force /10, pre_slide_speed /3)."""
    results = []
    for _ in range(trials):
        obs = settle(b)
        if not obs or not alive(obs):
            break
        x, y, z = obs["player"]["pos"]
        b.teleport([x, y + height, z])
        frames = []
        for _ in range(steps):
            obs = b.step(buttons=["slide"])
            frames.append(obs.get("move_tech") or {})
            if not alive(obs):
                break
        results.append(slam_verdict(frames))
    return {"trials": len(results), "ok": sum(1 for r in results if r["ok"]), "results": results}


def legacy_script(steps: int) -> list[dict]:
    """A fixed, input-only action script -- walk, slide, jump, strafe, stand -- exactly what a 0.7.2 client sends."""
    pattern = ([{"move": (0.0, 1.0), "buttons": ()}] * 10 + [{"move": (0.0, 1.0), "buttons": ("slide",)}] * 6
               + [{"move": (0.0, 1.0), "buttons": ("jump",)}] + [{"move": (1.0, 0.0), "buttons": ()}] * 5
               + [{"move": (0.0, 0.0), "buttons": ()}] * 8)
    return [dict(pattern[i % len(pattern)]) for i in range(steps)]


def record_script(port: int, level: str, frameskip: int, fixed_fps: float, steps: int) -> list[dict]:
    """The mod review's finding 8: ONE fixed script -- reset, `steps` legacy steps, a kill, a second reset --
    recorded reply by reply as an OLD-STYLE client, so the same call against the live 0.7.2 plugin and the 0.8.0
    test tree can be diffed field by field (`--diff`). Run it FIRST on a freshly started game: config is per
    game process, so anything a previous client switched on would be in the recording."""
    b = Bridge(port)
    try:
        b.hello()
        b.config(frameskip=frameskip, fixed_fps=fixed_fps, unlimited_fps=True, mute=True, windowed=True,
                 difficulty=4, unlock_all_gear=True, soft_death=False)
        rows = [b.reset(level)]
        rows += [b.legacy_step(move=s["move"], buttons=s["buttons"]) for s in legacy_script(steps)]
        rows.append(b.kill())
        rows.append(b.reset(level))
        return rows
    finally:
        b.close()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}{i}."))
        return out
    return {prefix[:-1]: value}


def diff_records(a: list[dict], b: list[dict], ignored=DIFF_IGNORED, tol: float = DIFF_TOL) -> list[str]:
    """Every field that differs between two recordings of the SAME script. Top-level keys in `ignored` are skipped;
    numbers within `tol` are equal. A key present on one side only is a difference (that is the 0.8 leak test)."""
    problems = []
    if len(a) != len(b):
        problems.append(f"length {len(a)} != {len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        fx = {k: v for k, v in _flatten(x).items() if k.split(".")[0] not in ignored}
        fy = {k: v for k, v in _flatten(y).items() if k.split(".")[0] not in ignored}
        for key in sorted(set(fx) | set(fy)):
            if key not in fx or key not in fy:
                problems.append(f"#{i} {key}: only in {'b' if key not in fx else 'a'}")
                continue
            u, v = fx[key], fy[key]
            numbers = all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (u, v))
            if (abs(float(u) - float(v)) > tol) if numbers else (u != v):
                problems.append(f"#{i} {key}: {u!r} != {v!r}")
    return problems
```

(c) In `main()`: add the arguments

```python
    ap.add_argument("--refusals", action="store_true", help="check that macros 2-5 and variant are refused")
    ap.add_argument("--slam-trials", type=int, default=0, help="drop-and-slam trials for block A's slam fields")
    ap.add_argument("--record", help="record the fixed legacy script to this JSONL (works on a 0.7.2 game) and exit")
    ap.add_argument("--record-steps", type=int, default=60)
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="diff two --record files field by field and exit")
```

and, directly after `args = ap.parse_args()`, before the `legacy_check` block:

```python
    if args.diff:  # no game at all
        rows = []
        for path in args.diff:
            with open(path, encoding="utf-8") as fh:
                rows.append([json.loads(line) for line in fh if line.strip()])
        problems = diff_records(*rows)
        print("\n".join(problems[:200]) if problems else "IDENTICAL outside frame / time / mem")
        print(f"{len(problems)} difference(s)")
        return 1 if problems else 0
    if args.record:  # before any 0.8 config; the version gate below does not apply
        rows = record_script(args.port, args.level, args.frameskip, args.fixed_fps, args.record_steps)
        with open(args.record, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in rows)
        print(f"recorded {len(rows)} replies to {args.record}")
        return 0
```

and in the main pass, directly after the jump-SSJ block (before `if not args.skip_wall:`):

```python
        if args.refusals:
            print("\n== reserved macros and variant, under the training defaults ==")
            report["refusals"] = refusal_check(b)
            print(json.dumps(report["refusals"], indent=2, default=str))
        if args.slam_trials:
            print("\n== block A against a real slam ==")
            report["slam"] = slam_trials(b, args.slam_trials)
            print(f"  {report['slam']['ok']}/{report['slam']['trials']} trials moved every slam field; maxima "
                  + ", ".join(f"{r['slam_force_max']:.2f}/{r['pre_slide_speed_max']:.2f}"
                              for r in report["slam"]["results"]))
```

- [ ] **Step 6: Run the tests**

Run: `PY tests\test_probe_tech.py` → `5 tests passed`; `PY tests\test_macro_check.py` → `5 tests passed`.

- [ ] **Step 7: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/scripts/probe_rollout.py python/scripts/macro_check.py python/tests/test_probe_tech.py python/tests/test_macro_check.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7 tools: probe_rollout absolute target slice + v2 records; macro_check --refusals/--slam-trials/--record/--diff" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 10: Play a checkpoint under its own layout (`eval.py`, `full_run.py`)

After the break, `models/specialists/Level_0-1.zip` becomes a v2 file at the next promotion while other levels'
specialists stay v1, and `full_run.py` reads each level's config from whichever `env_config.yaml` it finds first.
Without this task, a chained run or an eval of the wrong pairing crashes on the space check.

**Files:**
- Modify: `python/ultrakill_ai/ckpt_layout.py`, `python/scripts/eval.py`, `python/scripts/full_run.py`
- Modify: `python/tests/test_ckpt_layout.py` (append)

- [ ] **Step 1: Append the failing tests** to `test_ckpt_layout.py` (above `__main__`)

```python
def test_a_checkpoint_is_played_under_its_own_layout():
    from types import SimpleNamespace  # noqa: PLC0415

    from ultrakill_ai.ckpt_layout import adopt_layout  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        v2 = fake_ckpt(Path(tmp) / "Level_0-1.zip", 530, CAMPAIGN_NVEC_V2)
        cfg = SimpleNamespace(mode="campaign", tech_layout="v1")
        note = adopt_layout(cfg, v2)
        assert cfg.tech_layout == "v2" and "v1 -> v2" in note
        assert adopt_layout(cfg, v2) is None, "already right: nothing to say"


def test_adoption_leaves_cyber_grind_and_unreadable_files_alone():
    from types import SimpleNamespace  # noqa: PLC0415

    from ultrakill_ai.ckpt_layout import adopt_layout  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        v2 = fake_ckpt(Path(tmp) / "b.zip", 530, CAMPAIGN_NVEC_V2)
        grind = SimpleNamespace(mode="cybergrind", tech_layout="v1")
        assert adopt_layout(grind, v2) is None and grind.tech_layout == "v1"
        junk = Path(tmp) / "junk.zip"
        junk.write_bytes(b"x")
        cfg = SimpleNamespace(mode="campaign", tech_layout="v1")
        assert adopt_layout(cfg, junk) is None and cfg.tech_layout == "v1"


def test_eval_and_full_run_adopt_the_checkpoint_layout():
    """The two call sites are one line each; this pins that they exist, next to the loads they guard."""
    for script in ("eval.py", "full_run.py"):
        text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "adopt_layout(cfg," in text, script
```

- [ ] **Step 2: Run** `PY tests\test_ckpt_layout.py` → FAIL (`cannot import name 'adopt_layout'`).

- [ ] **Step 3: Add to `ckpt_layout.py`**

```python
def adopt_layout(cfg, path) -> str | None:
    """Sets `cfg.tech_layout` to the layout the checkpoint at `path` was built for; returns what changed, or None.

    For the tools that PLAY a policy (eval.py, full_run.py; probe_rollout.py has its own resolve_tech_layout): a
    checkpoint is played under its OWN layout whatever config sits beside it, because a promoted specialist and
    the stage config it is read with can straddle the break. Only a campaign config is touched; an unreadable zip
    changes nothing (SB3's load then reports whatever is wrong with it).
    """
    if getattr(cfg, "mode", "campaign") != "campaign":
        return None
    layout = checkpoint_layout(path)
    before = getattr(cfg, "tech_layout", "v1")
    if layout is None or layout == before:
        return None
    cfg.tech_layout = layout
    return f"{Path(path).name} is a {layout} checkpoint: tech_layout {before} -> {layout} for this run"
```

- [ ] **Step 4: The two call sites**

`eval.py`: import `from ultrakill_ai.ckpt_layout import adopt_layout  # noqa: E402`, and directly before
`model = cls.load(args.model, device="cpu")`:

```python
    note = adopt_layout(cfg, args.model)  # a checkpoint is played under its own layout (tech_layout v1 / v2)
    if note:
        print(note)
```

`full_run.py`: same import; in `play()`, directly after `cfg = eval_config(...)`:

```python
        note = adopt_layout(cfg, model_path)  # a specialist is played under its own layout (tech_layout v1 / v2)
        if note:
            print("%s: %s" % (level, note), flush=True)
```

- [ ] **Step 5: Run** `PY tests\test_ckpt_layout.py` → `12 tests passed`; `PY tests\test_eval.py`,
  `PY tests\test_full_run.py` → pass unchanged.

- [ ] **Step 6: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add python/ultrakill_ai/ckpt_layout.py python/scripts/eval.py python/scripts/full_run.py python/tests/test_ckpt_layout.py
git -C F:\Github\ULTRAKILL-AI-tech commit -m "S7: eval.py and full_run.py play each checkpoint under its own tech_layout" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 11: The S6 re-measure runbook, the S7 install runbook, the S8 switch -- in `docs/commands.md`

**Files:**
- Modify: `docs/commands.md` (a new section before `## Starting the driver so it survives a desktop-app restart`)

No code and no test of its own; its commands exercise Tasks 1-10. Add the section below verbatim (fill the
`<...>` placeholders only when running it). The S6 half is run by the lead, beside the fleet, when memory allows;
the S7 half at the lead-approved pause.

- [ ] **Step 1: Add the section**

````markdown
## S6 / S7 / S8 -- the tech break, step by step (plan: docs/superpowers/plans/2026-09-23-tech-break-python.md)

### S6 -- re-measure mod 0.8.0 on ONE private game, beside the fleet

The 2026-09-20 numbers predate the review's fixes (M1's `no_jump_path` precondition, `slide_grace`, the three-arm
control, the legacy cursor assertion), block A's slam fields have never moved in a game, and the review's
finding 8 (a field-by-field diff of 0.7.2 against 0.8.0) was never run. Two private games, one after the other,
each under 30 minutes, on **47812 only**. Never `games.py launch` / `stop`.

Gate first (from `python/`):

```powershell
python scripts\check_run.py      # ALERTS none, and "system commit" <= 85% (one game is ~1.2 GB, ~2% of 60 GB)
python scripts\mem_guard.py --run spec_0-1_speed --game-limit-gb 2.5 --commit-limit-frac 0.93 --dry-run
```

1. **Build without installing, copy by hand** (no code path can write the locked live DLL):

```powershell
cd F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge
dotnet build -c Release -p:InstallPlugin=false          # 0 warnings, 0 errors
$game = 'C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL'
Copy-Item bin\Release\UltrakillAIBridge.dll "$game\BepInEx-test\plugins\UltrakillAIBridge.dll" -Force
Remove-Item "$game\BepInEx-test\LogOutput.log" -ErrorAction SilentlyContinue
Remove-Item "$game\BepInEx-test\cache" -Recurse -Force -ErrorAction SilentlyContinue
Get-Item "$game\BepInEx\plugins\UltrakillAIBridge\UltrakillAIBridge.dll" | Select-Object Length, LastWriteTime
# the LIVE DLL must still read 73728 bytes, 2026-09-18 01:34
```

2. **The 0.7.2 reference recording**, on a normally started private game (it loads the LIVE plugin):

```powershell
cd F:\Github\ULTRAKILL-AI\python
$s6 = "runs\s6_$(Get-Date -Format yyyy-MM-dd)"; New-Item -ItemType Directory $s6 -Force | Out-Null
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'scripts'); import games; print(games.start_instance(47812, 368, 207, 3))"
netstat -ano | findstr ":47812"   # repeat until LISTENING; that pid is the game (the launch pid is only a handle)
# ~90 s after the port opens (Addressables finish booting), then:
.venv\Scripts\python scripts\macro_check.py --port 47812 --level "Level 0-1" --record "$s6\legacy_072.jsonl"
Stop-Process -Id <the LISTENING pid>
python scripts\games.py status    # 12/12 fleet ports listening, 47812 gone
```

3. **The 0.8.0 test-tree game** (Doorstop 4.5.0 override; ONE pre-quoted argument string, because PowerShell 5.1
   does not quote the spaces of `Program Files (x86)` inside an argument array):

```powershell
$pre = "$game\BepInEx-test\core\BepInEx.Preloader.dll"
$argline = "-aibridge-port 47812 -screen-fullscreen 0 -screen-width 368 -screen-height 207 -job-worker-count 3 -aibridge-nosteam --doorstop-enabled true --doorstop-target-assembly `"$pre`""
$env:SteamAppId = '1229490'; $env:SteamGameId = '1229490'
Start-Process -FilePath "$game\ULTRAKILL.exe" -ArgumentList $argline -WorkingDirectory $game
netstat -ano | findstr ":47812"   # note the LISTENING pid
```

4. **Gate everything on the tree**: `Select-String -Path "$game\BepInEx-test\LogOutput.log" -Pattern UltrakillAIBridge`
   names 0.8.0, and `(Get-Item "$game\BepInEx\LogOutput.log").LastWriteTime` did not move. (macro_check also exits 2
   unless `hello` says 0.8.0.) If the override was ignored, 47812 runs the live plugin and every result is a false
   negative.

5. **Record first (a clean process), diff, then the full pass:**

```powershell
.venv\Scripts\python scripts\macro_check.py --port 47812 --level "Level 0-1" --record "$s6\legacy_080.jsonl"
.venv\Scripts\python scripts\macro_check.py --diff "$s6\legacy_072.jsonl" "$s6\legacy_080.jsonl"
.venv\Scripts\python scripts\macro_check.py --port 47812 --level "Level 0-1" --trials 25 --skip-wall --refusals --slam-trials 10 --json "$s6\macro_check.json"
```

   Watch the private window during the jump arm: macro_check turns `ssj_indicator` on, so every landed SSJ shows
   the game's own bucket bar and `+25u/s` subtitle -- the human-visible cross-check of `macro.ssj`.

6. **The Python half, end to end, on the same game** (a scratch COPY of the newest checkpoint, migrated in %TEMP%):

```powershell
$tmp = "$env:TEMP\s6_tech"; New-Item -ItemType Directory $tmp -Force | Out-Null
$ck = .venv\Scripts\python -c "import sys; sys.path.insert(0, 'scripts'); import supervise; from pathlib import Path; print(supervise.choose_resume(Path('models/spec_0-1_speed'))[0])"
Copy-Item $ck "$tmp\src.zip"
.venv\Scripts\python scripts\add_tech_heads.py "$tmp\src.zip" "$tmp\tech_init.zip" --obs-from models\spec_0-1_speed --ent-floor 6.5
.venv\Scripts\python scripts\probe_rollout.py --config configs\generated\spec_0-1_speed.yaml --model "$tmp\tech_init.zip" --port 47812 --episodes 2 --max-steps 3000 --out "$s6\probe_v2" --tag v2 --archive-dir models\spec_0-1_speed
```

7. **Stop** by the LISTENING pid; `python scripts\games.py status` shows 12/12 and 47812 closed;
   `python scripts\check_run.py` shows ALERTS none; delete `$tmp`.

**GO criteria** -- every row, recorded in `docs/project-log.md` (dated) before S7 is scheduled:

| what | where | GO |
|---|---|---|
| Q1: an explicit timestamp reaches `ctx.time` | `macro_check.json` `jump_macro.dt_ms` | 11.999-12.000 ms |
| Q2: `updateMode` | `diag_in_level.update_mode` | `ProcessEventsInDynamicUpdate` |
| Q3: the monotonic cursor | `cursor.dash_after_macro_registered` / `..._tried` | 12 / 12 |
| Q4: the serialized values | `diag_in_level.walk_speed` / `fixed_delta_time` | 750 / 0.008 |
| M1 after review finding 4 | `jump_macro.landed`; `jump_control_release.landed` | >= 23 / 25; 0 / 25 |
| the whole-step gain | `jump_step_advantage_u_s_median` | about +4.7 u/s -- quote THIS, not +24.75 |
| reserved heads refused | `refusals.ok` | true |
| block A's slam fields move | `slam.ok`, the maxima | >= 5 / 10; `slam_force` max <= 10 and `pre_slide_speed` max <= 6 (else re-scale `spaces.MOVE_TECH_FIELDS` and its tests BEFORE S7) |
| finding 8 | the `--diff` line | 0 differences outside frame / time / mem, or each one explained in the log |
| the migration | the `add_tech_heads.py` block | `mean_kl` 0.000000000, `greedy_changed` 0, `entropy_added` 1.3986, optimizer carried |
| the v2 env | `$s6\probe_v2\v2_*.jsonl` | some `obs_move_tech` non-zero (slide_grace, heavy_fall); `beh.macro_request` on ~10% of records, `beh.macro_sent` ~2%, a `macro` block on exactly the sent ones; `beh.variant_request` / `beh.hook_request` > 0 with nothing sent |

### S7 -- the one break (a full pause, 15-25 minutes)

**Prepare, with the fleet running:**

- P1. S6 is GO in the log.
- P2. The **install commit** exists on branch `s7-install` in the worktree (NEVER in the main tree) and the full
  suite is green on it. It holds exactly:
  - `python/configs/specialists.yaml`, inside `speed:` (the `rewards:` lines unchanged; rewrite the "NO `train:`
    BLOCK" comment to say the only `speed.train` key is the re-based floor):

    ```yaml
      # S7, THE ONE BREAK (<date>): obs 479 -> 530, action 12 dims / 45 logits -> 15 / 57. M1 `ssj` live; macros 2-5,
      # `variant` and `hook` masked in Python, refused by the mod and pinned in training (the EnvConfig gate
      # defaults). Plan: docs/superpowers/plans/2026-09-23-tech-break-python.md.
      env:
        tech_layout: v2
      # The entropy floor moves with the new heads: 6.5 + the 1.3986 nats they add at their priors, as
      # add_tech_heads.py MEASURED it (spec §4.4). Complete stages keep train.ent_floor 6.5.
      train:
        ent_floor: 7.8986
    ```

  - `tests/test_speed_overrides.py::test_the_shipped_plan_ships_no_train_or_env_override` -- re-pin the two asserts:
    `assert p.speed_train == {"ent_floor": 7.8986}` and `assert p.speed_env == {"tech_layout": "v2"}` (docstring:
    S7 landed; S1 stays reverted, S2 cancelled);
  - `tests/test_speed_overrides.py::test_a_complete_stage_and_a_speed_stage_now_train_at_the_same_hyperparameters`
    -- its last line becomes `== {"speed_bonus", "speed_target_scale", "tech_layout"}`;
  - `tests/test_specialists_config.py::test_a_speed_stage_only_adds_the_bonus_switch_the_death_weight_and_oob` --
    `added = {"speed_bonus", "speed_target_scale", "tech_layout"}`, the final shape assert becomes
    `env.observation_space.shape == (530,) and len(env.action_space.nvec) == 15`, and, beside the 6.5 pin in
    `test_the_train_template_is_the_full_configs_optimiser`, add
    `assert campaign_driver.stage_config(p, "Level 0-2", kind=campaign_driver.SPEED)["train"]["ent_floor"] == 7.8986`;
  - `CLAUDE.md` (Task 12's phase-2 text) and the regenerated `AGENTS.md`.
- P3. The migration is rehearsed (S6 step 6).
- P4. Timing: minutes after a round boundary (a `STAGE END` in `runs\specialists_driver.log`), so the round
  END_STAGE ends is young, and just after the hourly watch (:47), so it does not fire mid-pause.

**The pause** (from `F:\Github\ULTRAKILL-AI\python`):

1. `New-Item runs\specialists\DRIVER_PAUSE` -- FIRST. Within a minute the driver log says `PAUSED`.
2. `Set-Content runs\specialists\END_STAGE "Level 0-1 speed"` -- a paused driver ignores it; the RESTARTED
   driver's first tick ends the round through the normal path (`unfinished -- ended by operator`, nothing
   promoted), so no judged round straddles the break.
3. **Stop by pid, never `taskkill /T`** (the games are children of the driver's tree):

   ```powershell
   Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'campaign_driver|train\.py|poll_status|keep_best|post_times|dashboard|mem_guard|spawn_main' } | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
   Stop-Process -Id <the campaign_driver.py pid>
   Stop-Process -Id <each helper pid: poll_status, keep_best, post_times, dashboard, mem_guard>
   Stop-Process -Id <the train.py pid>
   Stop-Process -Id <each spawn_main worker pid whose parent is that train.py>
   ```

4. `python scripts\games.py stop` (safe now: nothing is live), then `python scripts\games.py status` -> 0/12.
5. **Install:** `cd ..\mod\UltrakillAIBridge; dotnet build -c Release; cd ..\..\python`, then check that
   `...\BepInEx\plugins\UltrakillAIBridge\UltrakillAIBridge.dll` is no longer 73,728 bytes and is dated now.
6. **Migrate** the newest checkpoint and `best.zip`:

   ```powershell
   $ck = .venv\Scripts\python -c "import sys; sys.path.insert(0, 'scripts'); import supervise; from pathlib import Path; print(supervise.choose_resume(Path('models/spec_0-1_speed'))[0])"
   .venv\Scripts\python scripts\add_tech_heads.py $ck models\spec_0-1_speed\tech_init.zip --ent-floor 6.5
   .venv\Scripts\python scripts\add_tech_heads.py models\spec_0-1_speed\best.zip models\spec_0-1_speed\best_tech.zip --ent-floor 6.5
   ```

   Both must print `mean_kl 0.000000000`, `greedy_changed 0.000000000`, `entropy_added` 1.3986, `ent_floor 6.5 ->
   7.8986` and `optimizer carried`. **Exit code 2 = STOP** (nothing was written): go to "Abort".
7. **Quarantine**, then put the v2 files where `choose_resume` and `promote` look:

   ```powershell
   .venv\Scripts\python scripts\quarantine_pre_tech.py models\spec_0-1_speed            # read the list
   .venv\Scripts\python scripts\quarantine_pre_tech.py models\spec_0-1_speed --apply
   Copy-Item models\spec_0-1_speed\tech_init.zip models\spec_0-1_speed\latest.zip
   Copy-Item models\spec_0-1_speed\best_tech.zip models\spec_0-1_speed\best.zip
   .venv\Scripts\python -c "from ultrakill_ai.ckpt_layout import checkpoint_layout as c; print(c('models/spec_0-1_speed/latest.zip'), c('models/spec_0-1_speed/best.zip'))"   # v2 v2
   ```

8. **Land the install commit**, so the main tree's plan is v2:

   ```powershell
   git -C F:\Github\ULTRAKILL-AI-tech fetch origin
   git -C F:\Github\ULTRAKILL-AI-tech rebase origin/main s7-install
   # the full suite once more in the worktree, then:
   git -C F:\Github\ULTRAKILL-AI-tech push origin s7-install:main
   git -C F:\Github\ULTRAKILL-AI fetch origin
   git -C F:\Github\ULTRAKILL-AI merge --ff-only origin/main
   ```

9. **The config gate diff** -- the only differences allowed between the live generated config and the next one:

   ```powershell
   .venv\Scripts\python -c "import sys, yaml; sys.path.insert(0, 'scripts'); import campaign_driver as cd; p = cd.load_plan('configs/specialists.yaml'); print(yaml.safe_dump(cd.stage_config(p, 'Level 0-1', kind=cd.SPEED, target_seconds=<current rung, e.g. 85.0>, init_steps=<latest.zip steps>), sort_keys=False))" | Out-File -Encoding ascii $env:TEMP\next_speed.yaml
   git diff --no-index -- configs\generated\spec_0-1_speed.yaml $env:TEMP\next_speed.yaml
   ```

   Expected, and nothing else: the two generated header comment lines, `tech_layout: v2` added under `env:`,
   `ent_floor: 6.5` -> `7.8986`, and `timesteps`. Any other line: stop and read.
10. **Dry runs:** `.venv\Scripts\python scripts\campaign_driver.py --dry-run` (paused: validates the plan only,
    exit 0, `PAUSED`); `Remove-Item runs\specialists\DRIVER_PAUSE`; the same command again must print
    `[dry-run] END_STAGE: would end Level 0-1 (speed, round N) now as 'unfinished'`.
11. **Start:** `schtasks /Run /TN "ULTRAKILL-AI driver"` -- never a bare `Start-Process` (the section below).

**Verify from files only, never a socket** (the driver relaunches the games and the trainer by itself):

- `runs\specialists_driver.log`: `END_STAGE: ending Level 0-1 (speed, round N)`, `STAGE END ... unfinished --
  ended by operator`, `NOT promoting`, `stage config configs\generated\spec_0-1_speed.yaml (init N steps)`,
  `started trainer for Level 0-1 (pid ..., resume latest.zip at N steps)`, and NO `LAYOUT MISMATCH`.
- `configs\generated\spec_0-1_speed.yaml`: `tech_layout: v2`, `ent_floor: 7.8986`.
- `runs\spec_0-1_speed_train.log`: `hyperparameters in force: ...`, `tech layout v2: 530 inputs, 15 action dims
  (57 logits); pinned action rows [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]`, `entropy floor: 7.8986 nats`, no
  `REFUSING`.
- `runs\spec_0-1_speed\env_478xx.log` (all twelve): `mod_features_ok` with `mod=0.8.0 layout=v2` at each connect;
  no `mod_incompatible`, no `move_tech_missing`.
- `python scripts\check_run.py`: ALERTS none, 12/12 listening, five helpers up including `mem_guard.py`.
- After >= 20 episodes, `runs\spec_0-1_speed\status.json`: `mean_100.macro_request_frac` ~0.10,
  `macro_sent_frac` ~0.02, `variant_request_frac` ~0.15, `hook_request_frac` ~0.10, and `macro_ran_share` > 0
  (0 over 100 episodes is a design bug: read `macro_refusal_reasons` in `episodes.jsonl`); the first updates'
  `approx_kl` <= 0.03; `train/entropy_loss` about -(the pre-break entropy + 1.40).
- Commit, explicit paths: `python/models/spec_0-1_speed/tech_init.zip`, `latest.zip`, `best.zip`, `best.json`,
  and the dated `docs/project-log.md` entry; push.

**Judged** by the standing rule: one thing, >= 400k steps, confirmed against live `status.json`; the fresh
`median_time_50` against the pre-break 500k buckets. Revert trigger: a bucket median more than 15 s worse than
the pre-break baseline for two consecutive 500k buckets, or the fresh completion rate down more than 0.1. §8's
"105-120 s" ceiling was derived from the instantaneous +24.75 u/s and is overstated; the step survives +4.73.

**Abort** (anything off before step 8): keep the plan un-merged; if step 7 ran, move `pre_tech\*` back into
`models\spec_0-1_speed\` and delete nothing; the 0.8.0 DLL may stay (a v1 client is backward compatible, and the
v1 env sends every 0.8 switch OFF); leave END_STAGE (the next round runs v1), remove DRIVER_PAUSE, dry run,
`schtasks /Run`.

**Rollback** (after the break is live): DRIVER_PAUSE; stop the driver, helpers, trainer and workers by pid (the
games may keep running); `git revert` the install commit in the worktree, full suite, push to main, ff the main
tree; in `models\spec_0-1_speed\` move every v2 zip (`checkpoint_layout` says so) into `tech\` and move
`pre_tech\*` back; `Set-Content runs\specialists\END_STAGE "Level 0-1 speed"`; remove DRIVER_PAUSE; dry run;
`schtasks /Run`.

### S8 -- the `tech` bonus (>= 400k steps after S7; one thing at a time)

"The procedure every lever shares" (above), with this plan edit, and END_STAGE after the plan lands (a reward
change must not straddle a judged round):

```yaml
speed:
  rewards:
    death: 12.0
    oob: 0.035
    tech: 0.5
    tech_cap: 10.0
    tech_decay_start: <status.json `timesteps` at the edit>   # CUMULATIVE steps; 0 is refused by train.py
    tech_decay_steps: 3000000
```

Re-pin `test_the_shipped_plan_ships_no_train_or_env_override`'s `speed_rewards` in the same commit. Verify: the
train log's `tech bonus: 0.5 per landed SSJ, cap 10.0 per episode, ...`; `train/tech_scale` 1.0 then falling;
`reward_parts_mean_100.tech` > 0. The env refuses to start unless `hello.features` carries `ssj_instrument`.
Judged on the macro-use rate moving off its 2% prior (`macro_request_frac`), `macro_landed_frac`, the median,
and §4.5's three invariants (`tests/test_tech_bonus.py`).
````

- [ ] **Step 2: Commit and land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add docs/commands.md
git -C F:\Github\ULTRAKILL-AI-tech commit -m "docs/commands.md: S6 re-measure, S7 install and S8 switch runbooks for the tech break" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

### Task 12: Docs -- `protocol.md`, `layout.md`, `project-log.md`, `CLAUDE.md` (under 220 lines), `AGENTS.md`

**Files:** `docs/protocol.md`, `docs/layout.md`, `docs/project-log.md`, `CLAUDE.md`, `AGENTS.md`

- [ ] **Step 1: `docs/protocol.md`**

(a) At the end of the `### hello: features and diag (mod 0.8.0)` paragraph on `features`, add:

```markdown
**The Python client checks them at every connect** (`UltrakillEnv._check_mod`): `tech_layout: v2` needs
`monotonic_input_clock`, `macro.ssj` and `obs.move_tech`, and a `rewards.tech` above 0 also needs `ssj_instrument`
(the obligation stated under `macro` below). A missing one raises `BridgeIncompatible`, which is deliberately NOT
a `BridgeError`: no recovery rung retries it and no relaunch starts, so the worker ends with the reason in the
train log.
```

(b) Directly after the paragraph that begins `Added in mod 0.8.0. **Every one defaults to the 0.7.2 behaviour**`
(just above the 0.8.0 config table), add:

```markdown
**The Python env sends every key below on EVERY connect to a DLL that advertises `features`** -- the values its
config means, `false` for everything it does not use -- because config is per game process: a private test that
turned `variant_switching` on would otherwise be inherited. Against a 0.7.x DLL it sends none of them.
```

(c) In the `macro` bullet, after "...fail the worker loudly, rather than train against a dead channel.", add
` Done in Python: UltrakillEnv._required_features.`

(d) Append a subsection at the end of `## Technique blocks (mod 0.8.0)`:

```markdown
### How Python packs them (`tech_layout: v2`, 530 floats)

Indices 0-478 are unchanged. 479-490 block A in this order: `heavy_fall`, `slam_force`/10, `bounce_window`,
`coyote` (clip 1 s), `wall_jumps`/3, `wall_available`, `boost`, `boost_left`/100, `pre_slide_speed`/3 (clip 2),
`jump_cooldown`, **`slide_grace`**, `riding_rocket`. 491-518 blocks B and C, 522-529 block E: reserved, always 0.0.
519-521 block D from this step's `macro` report: ran, not run (`refused` / `degraded` / `disabled`), and
`ssj_bucket`/3 when it ran and landed a bucket 1-3. Every read defaults to 0.0. The action gains three dims after
the look mode: `macro` (6), `variant` (4), `hook` (2); only `macro` 1 reaches the wire at S7. The full map is in
docs/superpowers/plans/2026-09-23-tech-break-python.md.
```

- [ ] **Step 2: `docs/layout.md`**

(a) Append to the `spaces.py` bullet:

```markdown
Under `tech_layout: v2` (`ObsLayout(tech=True)`) a 51-float TECH block is appended at 479 (block A movement,
block D macro feedback; B, C and E reserved zeros) and the campaign action gains macro / variant / hook (15 dims,
57 logits); `tests/test_tech_layout.py` pins the v1 vector by SHA-256.
```

(b) Add, next to the other `python/ultrakill_ai/` bullets:

```markdown
- `python/ultrakill_ai/ckpt_layout.py`: stdlib only. Which layout (v1 479 / 12, v2 530 / 15) a checkpoint zip was
  built for, read from its `data` JSON; `resume_problem` is the refusal text train.py (exit 3) and the driver
  (`layout_mismatch`) share; `adopt_layout` makes eval.py / full_run.py play a checkpoint under its own layout.
```

(c) Add, next to `add_look_mode.py`:

```markdown
- `python/scripts/add_tech_heads.py`: the S7 migration, v1 -> v2. First-layer columns 479-529 zero, action rows
  45-56 zero with the prior biases (ln 45 / ln 17 / ln 9), everything else copied bit for bit (critic included),
  Adam carried along every axis, the step axis carried. Measures before it writes -- mean KL over the 12 shared
  dims on real `_last_obs` states from the run's checkpoints must be 0.0 -- and never overwrites.
- `python/scripts/quarantine_pre_tech.py`: moves a model dir's v1 zips into `pre_tech/` (gitignored); never deletes.
```

(d) Add to the tests list:

```markdown
- `python/tests/test_tech_layout.py`: the v1 packing (SHA-256) and action tables pinned; v2's index maps, block A
  scales and clips, block D, reserved zeros, pinned rows (no game needed).
- `python/tests/test_tech_env.py`: the `hello.features` refusal, the explicit 0.8 config, the wire gates, the macro
  report across input locks, the v2-only counters, info and status keys, on `FakeTechLevel` (no game needed).
- `python/tests/test_add_tech_heads.py`: the migration's exactness, priors, optimizer carry, never-overwrite and
  measure-before-write, on tiny SB3 models (no game needed).
- `python/tests/test_ckpt_layout.py`: shapes read from real SB3 zips, the resume refusal, the driver's guard,
  layout adoption, and that the module imports no numpy / torch (no game needed).
- `python/tests/test_quarantine_pre_tech.py`, `test_tech_training.py` (pinned rows, absolute entropy dims),
  `test_tech_bonus.py` (the dormant S8 reward), `test_probe_tech.py`, `test_macro_check.py` (no game needed).
```

(e) Append to the `probe_rollout.py` bullet:

```markdown
Target slots are read at absolute 448-455 (they were negative indices from the end until the S7 block was
appended); under v2 each record also carries the three new heads, blocks A and D and the mod's macro report, and
`--tech-layout` probes a layout the stage config does not use yet.
```

- [ ] **Step 3: `docs/project-log.md`** -- append a dated entry, `## 2026-09-2x -- the S7 Python half, built
  dormant`, stating: what was built (the task list), the commit hashes, the suite's file and test counts, the
  Task 4 step 6 dry-run block (mean_kl, entropy_added, optimizer carried) quoted verbatim, the decisions listed in
  this plan's "Spec deviations" section (one line each), and **what is NOT verified**: nothing here ran in a game;
  block A's scales are derived from decompiled code; the S6 checklist and the S7 install are still to run.

- [ ] **Step 4: `CLAUDE.md`, phase 1 (with the code, before the install) -- line-neutral, stays at 219 lines**

(a) Line 58 becomes (the baseline from Task 0 step 3; this plan adds 92 named tests):

```markdown
- `python/tests/` — 50 no-game test files, ~<baseline + 92> named tests, about 3 minutes for the lot.
```

(b) Lines 192-193 (the `**Mod v0.8.0 source is merged but NOT installed**` bullet, 2 lines) become exactly 2 lines:

```markdown
- **Mod v0.8.0 and the S7 Python half are built, tested and DORMANT** (`tech_layout: v1` everywhere; plan
  `docs/superpowers/plans/2026-09-23-tech-break-python.md`); the install is a 15-25 min pause, `docs/commands.md` S7.
```

Then `wc -l CLAUDE.md` must print 219 (under 220), then:

```powershell
cd F:\Github\ULTRAKILL-AI-tech\python
PY scripts\sync_agents_md.py
PY tests\test_agents_md.py
```

- [ ] **Step 5: `CLAUDE.md`, phase 2 -- this text goes into the S7 install commit (Task 11, P2), not now.** Lines
  192-196 (phase 1's 2-line bullet plus the 3-line "Numbers a newcomer needs" bullet) become exactly 5 lines:

```markdown
- **S7 INSTALLED <date>** (mod v0.8.0; `speed.env.tech_layout: v2`; `ent_floor` 7.8986): 0-1 speed trains obs
  **530** / action **15 dims, 57 logits** (M1 `ssj` live; macros 2-5, `variant`, `hook` masked + pinned); every
  other stage keeps v1 (479 / 12 / 45). Cyber Grind 11 / 42; look mode 1 is campaign-only; **33 of 35** levels ship;
  Brutal, all weapons unlocked in memory for AI runs only. A 479-wide checkpoint is refused under v2: migrate with
  `scripts/add_tech_heads.py`, move the old ones with `quarantine_pre_tech.py` (`docs/commands.md` S7).
```

- [ ] **Step 6: Full suite (rule 5), commit, land**

```powershell
git -C F:\Github\ULTRAKILL-AI-tech add docs/protocol.md docs/layout.md docs/project-log.md CLAUDE.md AGENTS.md
git -C F:\Github\ULTRAKILL-AI-tech commit -m "docs: the S7 Python half (protocol packing, layout, log entry, CLAUDE.md state line-neutral)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Then Task 0 step 4.

---

## Self-review (run against the spec, 2026-09-23)

**Spec coverage.**
- §4.1 action layout -> Task 1 (tables, decode), Task 3 (wire gates), Task 7 (pinned rows). §4.2 refusal as a
  first-class, uncharged signal -> Task 3 (`macro_refused`, `macro_refusal_reasons`, block D), no charge anywhere;
  equal-length steps need no time correction (measured 2026-09-20). §4.3 observations -> Task 1 (blocks A + D,
  B/C/E zero), Task 9 (`probe_rollout` absolute indices), Task 4 step 4 (`add_look_mode` docstring -- the spec says
  its guard rejects "every pre-break checkpoint"; what it actually rejects is every POST-break one, and the note
  says that). §4.4 migration -> Task 4 (all five steps, the axis-1 carry fix, the loud refusal of a fresh
  optimizer, KL == 0, marginals, entropy +1.3986), Task 5 (the train-time guard plus the driver guard), Task 6
  (`pre_tech/`), Task 11 (the `ent_floor` pin beside 6.5 in `test_specialists_config.py`). §4.5 discovery -> Task 8
  (bucket gate, cap, no pay on HP loss, 3M decay, the three invariants), Task 7 (priors preserved by pinning).
  §4.6 -> Task 11 S6 (the four questions, `ssjIndicator`, the test tree, 47812, memory gate). §4.7 -> Task 11 S7
  (DRIVER_PAUSE first, END_STAGE, stop by pid, `games.py stop` only when nothing is live, build, migrate +
  quarantine, plan edit, `schtasks /Run`, verification from files). §5 S6-S8 rows and §6 -> Task 11. §8 -> the
  runbook quotes +4.73 u/s, not the overstated ceiling.
- The mod review's open items for the Python side: finding 7's client obligation (`ssj_instrument`) -> Task 8;
  finding 8's field diff -> Task 9 (`--record`/`--diff`) + Task 11 S6; finding 9's three-arm control and finding
  10's legacy cursor assertion -> re-run in S6 step 5; "re-run block A against a real slam first" -> `--slam-trials`.

**Placeholder scan.** The only `<...>` left are runtime values the operator reads off the live run at the moment
(pids, the current rung, the step count, the date, the baseline test count); every code step carries its code.

**Type / name consistency.** `tech_gates -> TechGates(macros, variant, hook)`; `pinned_action_rows(live_macros,
variant, hook)`; `pin_action_rows(policy, rows)`; `apply_tech_training(model, env_cfg)`; `resume_refusal(resume,
env_cfg)`; `resume_problem(path, tech_layout)`; `checkpoint_layout` / `checkpoint_shapes` / `adopt_layout`;
`macro_landed(report)`; `tech_scale(num_timesteps, start, steps)` and `compute_reward(..., tech_scale=,
tech_room=)`; `TechDecayCallback(start, steps)` -> `env_method("set_tech_scale", s)` -> `UltrakillEnv.set_tech_scale`;
`TECH_METRICS` / `TECH_LOG_RAW`; `dim_entropies`; `resolve_tech_layout(config_layout, requested, model)` -- each
defined once, used with the same signature everywhere.

**Test totals.** 9 new files, 92 new named tests (16 + 20 + 13 + 12 + 4 + 3 + 11 + 5 + 5, plus 3 in
`test_speed_overrides.py`); the suite goes from 41 to 50 files.

**Already validated while writing this plan (2026-09-23), without touching the repo or the run.** Every code
block of Tasks 1-10 was applied mechanically, by script, to a scratch copy of `python/` in the session
scratchpad, and the full no-game suite ran there: **50 files green, 1,025 counted named tests** (plus
`test_progress.py`, which prints "all tests passed"). The one gap it found -- `test_campaign_rewards.py` pins
`compute_reward`'s parameter list -- is now Task 8 step 5(e). The v1 pins were also run against the UNCHANGED
package and pass, so the hashes are today's. And the real migration ran on a scratch COPY of the live newest
checkpoint, `ckpt_57974314_steps.zip`, with its 2,400 real `_last_obs` states: `mean_kl` 0.000000000, `max_kl`
0.000000000, `greedy_changed` 0, `mean_abs_dv` 0.000000000, `entropy_added` 1.398610592 (7.1796 -> 8.5782 nats),
all three heads on their priors to < 3e-7 and state-independent, optimizer carried, 57,974,314 steps and 141,126
updates carried. Nothing was written outside the scratchpad.

## Open questions (with the default this plan takes)

1. Observations for the KL check: the spec's `stoch_*.jsonl` hold no observation vectors -> **`_last_obs` from the
   run's own checkpoints** (twelve real states each), synthetic as the fallback.
2. `hook` at S7: the spec neither refuses it nor can the mod -> **masked in Python and pinned until S10**.
3. Reserved heads -> **pinned** (gradient-masked). Leaving them free would let the entropy bonus diffuse them, lose
   the §4.4 priors before S9/S10, and hand the entropy floor ~2.5 nats of free entropy.
4. Masking in Python as well as the mod's refusal -> **both** (block D "not run" then means a real precondition
   failure; the mod's refusal is the second lock).
5. Block A scales (`slam_force` /10, `coyote` clip 1 s, `pre_slide_speed` /3 clip 2) are derived from decompiled
   code -> **as stated**, re-checked by S6's `--slam-trials` before the install.
6. Block D encoding -> **ran, not-run, bucket/3 when landed**.
7. Which stages go v2 -> **`speed.env` only** (the focus runs 0-1 speed alone); 0-2 / 0-3 are migrated when the
   focus ends, and the driver refuses them loudly until then.
8. `ent_floor` -> **7.8986 in `speed.train:`** via a new top-level routing; complete stages keep 6.5.
9. The S8 decay start -> **an absolute `rewards.tech_decay_start` the operator writes**, pushed through
   `env_method`; 0 with a window is refused.
10. The finding-8 diff -> **moved into S6**, off the pause.
11. `best.zip` -> **migrated too**, so keep_best's peak and the next promotion are v2.
12. Workflow -> **a worktree, fast-forwarded into main per green task**, because the live trainer imports main's
    code at every restart.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-23-tech-break-python.md`. Two execution options:

1. **Subagent-Driven (recommended)** -- a fresh subagent per task, review between tasks, fast iteration
   (superpowers:subagent-driven-development).
2. **Inline Execution** -- tasks in one session with checkpoints (superpowers:executing-plans).

Tasks 0-10 and 12 need no game and no pause. Task 11's S6 half needs one private game on 47812 and the lead's
memory check; its S7 half is the scheduled full pause.
