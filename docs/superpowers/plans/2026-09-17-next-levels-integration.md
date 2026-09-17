# Integrating branch `next-levels` — the checklist for the next training pause

**What this is.** Branch `next-levels` (worktree `F:\Github\ULTRAKILL-AI-next`, branched from `7da1204`) carries
the multi-level curriculum (S1), the gates usability guard (S2), the 6-2 exit tie-break (S3) and the skull-carry
gates (S4) of `docs/superpowers/specs/2026-09-17-multi-level-and-skull-gates-design.md`. Mod version 0.6.0 → 0.7.0.

**What has been verified, and what has not.** Everything offline: `dotnet build -c Release -p:InstallPlugin=false`
is 0 warnings / 0 errors, and all thirteen no-game test files pass (258 assertions). **Nothing on this branch has
been run against the game.** No game was launched, no port opened and the live run's tree was never touched while
the branch was written. So every claim below about in-game behaviour is a prediction from scene files and
decompiled source, and this checklist is how it gets tested.

**Read this first: the branch splits cleanly into a shippable half and a blocked half.**

| | what it is | state | gated on |
|---|---|---|---|
| **S1** multi-level curriculum | `env.levels`, `curriculum.json`, per-level stats | ready to run | §4.1, §6 |
| **S2** gates guard | Python-only, ignores a ladder below 50% hops coverage | ready to run | §4.6 |
| **S3** 6-2 exit tie-break | mod, `Level P-` dropped / `Intermission*` kept | ready to run | §4.5 |
| **S4** skull-carry gates | `altars[]`, `items[]`, `needs_item`, two new milestones | **wired but dormant** | §5 — `skull_check.py`, offline-green, never run in the game |

S4 ships with `item_pickup` and `item_placed` at `0.0` in `configs/campaign_1-1.yaml` and absent (so `0.0`) in
`configs/campaign_prelude.yaml`, and the prelude's three levels have no altars at all. **So S1+S2+S3 can merge,
verify and start training while S4 stays inert.** Do not raise those two weights until §5 passes. Nothing in §6
or §7 depends on §5.

Time estimate for the whole list: **35-50 minutes of pause** for §1-§4 and §6, plus the 20k smoke run (~2 min at
177 steps/s) and however long §7's restart takes. §0 costs nothing — do it while training is still running.

---

## 0. Before the pause (training keeps running)

Nothing here touches the main tree, the live run or the game.

1. **Re-run the offline suite in the worktree**, so the pause does not start with a surprise. From
   `F:\Github\ULTRAKILL-AI-next\python` in PowerShell, with the main venv and `PYTHONPATH` pointed at the
   worktree (the package is an editable install pointing at the main tree; without `PYTHONPATH` this silently
   tests the *live* code):

   ```powershell
   $env:PYTHONPATH = "F:\Github\ULTRAKILL-AI-next\python"
   Get-ChildItem tests\test_*.py | ForEach-Object {
     F:\Github\ULTRAKILL-AI\python\.venv\Scripts\python.exe $_.FullName
     if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" } }
   ```

   Expected: thirteen files, every one ending in `N tests passed` / `all tests passed`. `test_campaign_check.py`
   and `test_skull_check.py` print `[FAIL]` lines from their *fake* levels on purpose — those files are asserting
   that a broken level is reported as broken. Judge them on their last line only.

2. **Rebuild the mod without installing**, to confirm the toolchain is there before the games are down. `dotnet`
   is not on `PATH` in this shell; the full path works:

   ```powershell
   Set-Location F:\Github\ULTRAKILL-AI-next\mod\UltrakillAIBridge
   & "C:\Program Files\dotnet\dotnet.exe" build -c Release -p:InstallPlugin=false
   ```

   Expected: `Build succeeded. 0 Warning(s) 0 Error(s)`. `-p:InstallPlugin=false` is new on this branch and is
   what keeps the copy step from touching the DLL the five running games hold open. **It defaults to `true`**, so
   the ordinary `dotnet build -c Release` in §3 still installs exactly as it always has.

3. **Decide the run name now.** §7 starts a new run called `campaign_prelude`. Its exploration archives must be
   copied from the live run before it starts, and that copy is the one step of this list that cannot be undone by
   a rollback if it is done wrong (it writes into a new directory, so in practice it is safe — but do it from a
   *stopped* trainer, in §7, not now).

---

## 1. Pause the live run

1. `Ctrl+C` in the training console and **wait for `train.py` to print `Saved models\campaign_gates\latest.zip`
   and exit**. `runs/campaign_gates_train.log` is appended across restarts, so an older such line proves nothing —
   watch the console.
2. `python scripts/games.py stop` from `F:\Github\ULTRAKILL-AI\python`.
3. Stop `poll_status.py`, `keep_best.py` and `dashboard.py`.
4. Confirm nothing is listening: `python scripts/games.py status` should report no instances. The mod DLL is
   locked by any running game and §3 will fail loudly if one is still up.

**Commit the live run's outputs to `main` before merging.** At the time this branch was written the main tree
had `python/models/campaign_gates/best.zip`, `best.json`, `env_config.yaml`, `first_completion_3214785.zip` and
five modified `explore_*.npz` uncommitted. Committing them first keeps the merge a clean one and keeps the
rollback in §8 honest.

---

## 2. Merge

From the **main tree** `F:\Github\ULTRAKILL-AI`:

```powershell
git rev-parse main                 # WRITE THIS HASH DOWN — it is the rollback target
git merge next-levels
```

`next-levels` is checked out in the other worktree; merging a branch checked out elsewhere is allowed (only
*checking it out* twice is not). If `main` has not moved past `7da1204`, this fast-forwards. If it has (the model
commit above), it is an ordinary merge and should not conflict: the branch touches no file under
`python/models/` or `python/runs/`.

Sanity check after the merge:

```powershell
git log --oneline -3
Select-String -Path mod\UltrakillAIBridge\Plugin.cs -Pattern '0\.7\.0'
Test-Path python\configs\campaign_prelude.yaml, python\configs\campaign_1-1.yaml
```

---

## 3. Build and install the mod

Games must be closed (§1.4), or the DLL is locked and the copy fails.

```powershell
Set-Location F:\Github\ULTRAKILL-AI\mod\UltrakillAIBridge
& "C:\Program Files\dotnet\dotnet.exe" build -c Release
```

Expected: `0 Warning(s) 0 Error(s)` **and** a `Copied UltrakillAIBridge.dll to ...` line. Verify the install
landed, since the whole point of `InstallPlugin` is that the copy can now be switched off:

```powershell
Get-Item "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\plugins\UltrakillAIBridge\UltrakillAIBridge.dll" |
  Select-Object Length, LastWriteTime
```

The timestamp must be from this build, and the file must be ~68-70 KB (0.6.0 was 58,880 B).

---

## 4. In-game verification

One game, port 47800, nothing else connected — **the bridge is single-client, so a second connection silently
kicks the first**. Launch with `python scripts/games.py launch --count 1 --monitor 1` from
`F:\Github\ULTRAKILL-AI\python` with the venv active, and `python scripts/games.py stop` at the end of §4.

Two tools do the work and they are used in a fixed pairing:

- **`campaign_check.py --level X`** takes control, **loads** the level, configures it (Violent, all gear,
  30 fps / frameskip 2, rendering off) and runs six checks. This is the only way to get the game into a chosen
  level from a script.
- **`bridge_test.py --campaign`** is read-only, never takes control and **cannot load a level** — it dumps the
  `campaign` block of whatever is loaded right now. So it always runs *straight after* a `campaign_check.py` on
  the same level, while the scene is still up.

Because `campaign_check.py` releases control on exit, the `difficulty` line `bridge_test.py` then prints is the
game's own setting, not the override. That is expected and is not a failure.

### 4.1 `Level 0-1` — the regression that matters most

```powershell
python scripts\campaign_check.py --level "Level 0-1"
```

**Pass:** the `summary:` line must be no worse than the branch point — checks 1, 3, 4 and 6 PASS, 2 SKIP,
5 FAIL (the exit room is switched off at load; a known, documented artifact of the check teleporting onto a
collider in an inactive room, cleared for real by the human playthrough on 2026-09-16).

**This is the load-bearing check of the whole merge.** 0-1 is the level the live policy has 3.2M steps on, it has
no altars and no items, and S4 must be completely invisible there. Then:

```powershell
python scripts\bridge_test.py --campaign
```

**Pass:** `altars: 0` and `items: 0` (the lists are present and empty — *not* the `- (a mod older than 0.7.0)`
line), every gate shows `needs_item` absent or null, `gates: 11 ordered=True ... (1.000)`. If `altars`/`items`
print the "older than 0.7.0" line, the DLL in the game folder is stale — go back to §3.

### 4.2 `Level 1-1` — altars, items and `needs_item` (S4's structural half)

```powershell
python scripts\campaign_check.py --level "Level 1-1"
python scripts\bridge_test.py --campaign
```

Check §6.2 of the spec for the full expected block. The six things that must be true, all of them predictions
from the scene bundle that have never been read out of the running game:

1. `altars: 7` — six functional zones plus the coincident duplicates, with keys `-15,27,427`, `-15,27,427#2`,
   `0,-7,381`, `0,-7,381#2`, `81,-2,275`, `81,-4,251`, `81,-4,251#2`.
2. `items: 5`, every one `placed=true` at load (there is no loose skull on 1-1 — this is why the sub-goal rule is
   "not in the altar we are trying to fill" rather than "not placed").
3. The `#2` entries carry `inactive_ancestors=2`. That is the dead-twin marker; if they read `1`, `needs_item`
   will never clear after a placement (M14) and S4 must not be enabled.
4. Gate `81,-6,240` is present with **`altar_only`**, `hops=1` and `needs=SkullBlue`. This is the gate the mod's
   new phase-2 pass adds; without it the blue leg is invisible to the route.
5. Gate `20,-10,381` shows `hops=2` and `needs=SkullRed`.
6. Gate `16,20,427` shows **no** `needs` — it is a *reverse* door (a blue skull *closes* it), and `needs_item`
   is built from `doors` only.

Also confirm `campaign_check.py`'s check 6 reports 14 gates (13 phase-1 plus the one `altar_only`) and that the
key-collision and rank-tie warnings, if any, are in
`"C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\LogOutput.log"`.

### 4.3 `Level 0-2` — altars with nothing wired to a gate

```powershell
python scripts\campaign_check.py --level "Level 0-2"
python scripts\bridge_test.py --campaign
```

**Pass:** `altars: 2` (plus any duplicate twins), both `SkullBlue`; `items: 2`, both `SkullBlue`; and — the point
of this level — **every gate reports no `needs_item`**, because 0-2's two altars are not wired to any two-room
gate door. 0-2 exercises "altars exist, route unaffected", which is the shape eleven other levels have.

### 4.4 `Level 0-4` — items with no altars, and the punch side effect

```powershell
python scripts\campaign_check.py --level "Level 0-4"
python scripts\bridge_test.py --campaign
```

**Pass:** `altars: 0`, `items: 1` (`CustomKey1`), no `needs_item` anywhere.

**0-4 is in the prelude curriculum of §6, and this is where its punch bug was found and fixed (2026-09-17).**
`UltrakillEnv._protect_carry` used to drop the `punch` button on any step where *anything* read `held: true` and
the current sub-goal was not an altar. 0-4 has a carryable key and no altar, so **the moment the agent picked the
key up it lost the punch button for the rest of the carry, and could not throw the key either** (throwing is
itself a punch). It now gates on a *carry* — a held item some live unfilled altar accepts, through
`campaign.wanting_altars`, the same filter `GateProgress._subgoal` picks a destination with — so on 0-4 the button
is untouched. The forced look mode 2 beside it was checked and was already narrow: it keys on the target carrying
a `subgoal` field, which only exists for a gate with `needs_item`.

**Pass, therefore:** nothing about 0-4 changes the `bridge_test.py` readout above, and in §6's smoke run
`part_punch` on 0-4 must look like it does on 0-1. (0-4 is locked at 20k steps, so that is for the real run.)

### 4.5 `Level 6-2` — the exit tie-break (S3)

The bug is a `FindObjectsOfType`-order tie, so **one load proves nothing**. Run three times:

```powershell
python scripts\campaign_check.py --level "Level 6-2"   # x3, separately
```

**Pass:** check 1's detail line reports the **same** `exit (x, y, z)` all three times, and that pit is the
`Intermission2` one — not either of the two `Level P-` Prime Sanctum pits, which `Scan` now discards outright.
6-2 has no gate candidates at all, so check 6 reporting no gates is expected and is not a failure. Then:

```powershell
Select-String -Path "C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\BepInEx\LogOutput.log" `
  -Pattern 'ChooseExit|tie'
```

**Pass:** **no** rank-tie warning for 6-2 (the Prime Sanctum pits are gone before ranking, leaving one candidate
at the best rank). The same warning *is* expected on 3-2 (two `Intermission1` pits at an identical position),
2-4 (two `Level 3-1` pits 1.4 m apart) and 8-4 (`EarlyAccessEnd`, which parses as neither a level nor an
intermission and is picked by uniqueness alone). Those three are benign and are logged rather than silent on
purpose; do not chase them.

### 4.6 `Level 7-2` — the usability guard (S2)

```powershell
python scripts\campaign_check.py --level "Level 7-2"
python scripts\bridge_test.py --campaign
```

**Pass:** the `gates:` line reads `ordered=True` with a hops ratio **of about 0.250** — e.g.
`gates: 4 ordered=True truncated=False with hops=1 (0.250)`. That parenthesised ratio is precisely what
`GateProgress._gates` tests, and 0.250 is below the 0.5 threshold, so the ladder is discarded and the env falls
back to the straight-line exit vector. The fallback itself is unit-tested offline
(`test_unordered_level_falls_back_to_the_exit_vector`); what this in-game step proves is the *input* — that the
mod really does report a `gates_ordered: true` ladder with one hop value out of four, which is the measurement
the threshold was chosen against. If the ratio comes back at or above 0.5, the guard will not fire on 7-2 and the
threshold needs re-measuring before any run that includes 7-2. (No run in this plan includes it.)

`8-3` is the other degenerate level, at 1 of 32 (0.031). Checking it is optional; 7-2 is the tighter case.

### 4.7 Stop the game

```powershell
python scripts\games.py stop
```

---

## 5. The skull-carry checks — `scripts/skull_check.py`

§8 checks 1-3 of the spec gate raising `item_pickup` / `item_placed` off `0.0`. **The tool that runs them now
exists** (2026-09-17, on this branch): `scripts/skull_check.py`, in the shape of `campaign_check.py`, green
against `FakeLevel`'s skull room in `tests/test_skull_check.py`. It has never been pointed at the game. The pause
only has to run it.

| check | what it settles | how the script does it |
|---|---|---|
| 1. `ActiveFrame` survives `render: false` | whether S4 is inert under training settings at all | walks to `(81.0, -2.2, 275.0)` on 1-1 with the cameras off, faces it inside the game's own 85° clamp and punches until `items[].held`. A human playthrough **cannot** test this: `render: false` only applies while the AI has control. |
| 2. placement survives punch spam | whether `filled` and `needs_item: null` stick | carries to `(0.0, -6.76, 381.0)`, punches, then presses punch for 100 decisions **through `env.step`**, so §6.7's gating is what is under test. Every decision is inspected, not just the last. |
| 3. held skull on death | the one thing that cannot be settled offline | takes the skull back out of the altar, runs the protocol's `kill`, and prints every `items[]` entry of that type after the respawn. |

```powershell
python scripts\games.py launch --count 1 --monitor 1
python scripts\skull_check.py                 # Level 1-1, training settings, rendering OFF
python scripts\skull_check.py --render        # the control run, if check 1 fails
python scripts\games.py stop
```

**Pass:** `summary: 1 pickup PASS | 2 placement PASS | 3 held skull on death PASS`, exit code 0. Check 2's detail
must show `altar 81,-2,275`-side pickup, then `filled=True` and `gate 20,-10,381 needs_item=None` **both at the
placement and after the spam** — the dead twin `0,-7,381#2` keeping `needs_item` set is M14 and is a FAIL.

**It walks; it never teleports.** A bare teleport into a switched-off room activates nothing (measured on 0-1) and
1-1's Skull Field starts off. 1-1 spawns at `(0, 105, 253)`, 136 m away, and the Skull Field is adjacent to the
spawn room. The script prints the item's `active` flag **before** it punches, so the two failures are never
confused: if check 1 FAILs with `the item is NOT active`, the **walk** failed and the fix is `--via X Y Z`
(repeatable) to route the approach — it says nothing about `ActiveStart`. Only a FAIL with `active=True` before
the punch is evidence about the AnimationEvent, and then `--render` is the control that separates "the punch does
not work" from "the punch does not work with the cameras off".

Until all three pass, leave `item_pickup` and `item_placed` at `0.0` and do not start a 1-1 run. Check 1 is the
kill switch: if the fist Animator is culled under `render: false` *with* `TrainingSpeed.ForcePlayerAnimators`
applied, S4 is inert under training settings and the rest of S4 does not matter.

---

## 6. The 20k-step multi-level smoke run

This is the first time the curriculum, `curriculum.json`, the per-level tables and the level switch inside a
worker run against real games. Throwaway outputs — a separate run name, deleted afterwards.

```powershell
python scripts\games.py launch --count 5 --monitor 1
python scripts\train.py --config configs\campaign_prelude.yaml --timesteps 20000 --run-name campaign_prelude_smoke
```

Start it from **fresh weights** (no `--resume`). The point is the plumbing, not the policy, and a fresh start
keeps the live checkpoint out of a directory that is about to be deleted.

**Pass, in the order it appears:**

1. `train.py` prints `curriculum: 3 levels, unlocked ['Level 0-1']` before learning starts. It is printed there
   because SB3 resets the envs inside `_setup_learn`, *before* `_on_training_start` — if this line is missing or
   lists the wrong set, the workers are reading a stale file.
2. `runs\campaign_prelude_smoke\curriculum.json` exists, names all three levels in order and marks only
   `Level 0-1` unlocked. 20k steps will not unlock a second level (that needs 20 fresh completions of 0-1), so
   **every episode should run on 0-1** — that is the correct result, not a bug.
3. It finishes cleanly at roughly **150-180 steps/s** (0-1 alone ran 177 with five games).
4. `runs\campaign_prelude_smoke\episodes.jsonl` — every row carries `"level": "Level 0-1"`.
5. `models\campaign_prelude_smoke\explore_Level_0-1_478*.npz` — five files, one per game, level-keyed.
6. `python scripts\dashboard.py --run campaign_prelude_smoke --smoke-test` renders once and exits. With one
   unlocked level the Campaign panel shows the `fresh score  N / 1 levels` headline and a `levels` block with one
   row. (The headline is a *score*, not a percentage — see §7.)
7. `python scripts\poll_status.py --run campaign_prelude_smoke` for one tick, then check
   `runs\campaign_prelude_smoke\metrics_log.csv` has the `levels_unlocked` column and the
   `part_item_pickup` / `part_item_placed` columns.

**Watch list while it runs** — none of these fails the smoke run, but note them:

- `oob_frac` and the reward parts should look like the live 0-1 run's. If `part_gate_approach` collapses to ~0,
  the S2 guard is firing on 0-1, which it must not (0-1's ratio is 1.000).
- Any `UltrakillEnv: ignoring ...curriculum.json` line in the log means a worker rejected the file (wrong run
  name or level order) and is training on `levels[0]` only. Harmless here, fatal to the curriculum later.
- The 0-4 punch behaviour of §4.4 cannot show up yet — 0-4 is locked at 20k steps.

Then `python scripts\games.py stop` and delete `runs\campaign_prelude_smoke\` and
`models\campaign_prelude_smoke\`.

---

## 7. Resuming the live 0-1 policy into the multi-level run

**The claim: `models/campaign_gates/best.zip` drops into `campaign_prelude` without being invalidated.** Checked
field by field against the branch:

| | why it carries |
|---|---|
| observation | 479 inputs, unchanged. S4's sub-goal **reuses slots 448-455** (the gate-target block); no level id is added anywhere, by design. |
| action space | unchanged. Look mode 2 shipped in 0.6.0; S4 reuses it and adds no button — `Punch.AltHit` does pick-up and place off the existing `punch`, so **no weight surgery** and no `add_look_mode.py`-style migration. |
| reward weights | `configs/campaign_prelude.yaml`'s `rewards:` is character-for-character `configs/campaign_0-1.yaml`'s. `item_pickup`/`item_placed` are absent, so `0.0`. |
| hyperparameters | identical, including `gamma 0.998` and `ent_coef 0.01`. |
| the curriculum's first rung | only `Level 0-1` is unlocked at the start, and 0-3 unlocks only at 50% fresh completion over 20+ fresh episodes. **So for its whole early life this run is the 0-1 run continuing.** |

The one thing that genuinely does not carry is the *bookkeeping*, and it is why a new `run_name` is mandatory
rather than a preference:

- **Copy the exploration archives first.** `explore_dir` follows `run_name`, so a renamed run orphans them and
  restarts exploration from nothing — and their floor counts carry the `1/sqrt(N)` decay that is the only thing
  pushing the agent outward. **Do this with the trainer stopped:**

  ```powershell
  New-Item -ItemType Directory -Force F:\Github\ULTRAKILL-AI\python\models\campaign_prelude
  Copy-Item F:\Github\ULTRAKILL-AI\python\models\campaign_gates\explore_*.npz `
            F:\Github\ULTRAKILL-AI\python\models\campaign_prelude\
  ```

  They are already level-keyed (`explore_Level_0-1_47800.npz`), so 0-3 and 0-4 start empty as they should.
- **`campaign.fresh_completion_rate` changes meaning.** On a multi-level run it is a **shrunk sum over the
  unlocked levels, not a rate**, and it can exceed 1.0. With one unlocked level at `fresh_window >= 20` it equals
  the number it has always been, so nothing jumps at the start — but once 0-3 unlocks, the pooled figure is a
  score. The dashboard relabels it `fresh score N / K levels` for exactly this reason. **Judge this run on the
  per-level rows**, never on the pooled `mean_100` numbers: `max_steps` is one value for every level, so a longer
  level is truncated by construction.
- **Why a sum and not the mean the scope asked for.** A plain mean over unlocked levels *drops* at every unlock
  (~0.52 → ~0.26 the moment the second level joins), and `keep_best.py` only replaces `best.zip` on a strict
  improvement and is deliberately not edited here. A mean would freeze `best.zip` on the single-level policy and
  print the "15% below best" warning forever. A shrunk sum cannot drop — a newly unlocked level contributes 0 and
  grows from there. This is a deliberate deviation from the scope wording, taken to keep `keep_best.py`
  untouched; the per-level means the scope wanted are all in `status.json`'s `campaign.levels` table and on the
  dashboard.
- **Move any old `metrics_log.csv` aside.** `poll_status.py` keeps an existing header and silently drops columns
  it lacks, so an inherited file would swallow `levels_unlocked`. A new run name gives a new file anyway.
- **`keep_best.py` needs no new flag and no migration.** `--metric campaign` reads `fresh_completion_rate` from
  the CSV, which is now the curriculum score; the new run directory gets a fresh `best.json`, so the
  "written by the other metric" refusal cannot trigger.

Start it:

```powershell
python scripts\games.py launch --count 5 --monitor 1
python scripts\train.py --config configs\campaign_prelude.yaml --resume models\campaign_gates\best.zip
python scripts\poll_status.py --run campaign_prelude
python scripts\keep_best.py --run campaign_prelude --metric campaign
python scripts\dashboard.py --run campaign_prelude --monitor 1
```

**First-hour checks:**

1. `curriculum: 3 levels, unlocked ['Level 0-1']` at startup. If it prints a *different* unlocked set, `_restore`
   found a `status.json` from another run name — stop and look, because a rename silently re-locks every level.
2. Entropy, `approx_kl` and the reward parts should sit where the live run left them. They are the same policy on
   the same level under the same weights; a jump means something in the carry is wrong.
3. `part_item_pickup` and `part_item_placed` must stay exactly `0` for the entire run. They are dormant by weight
   and the prelude levels have no altars — a non-zero value means a weight got raised by accident.
4. The first unlock is the event to watch for. When 0-3 unlocks, `levels_unlocked` goes 1 → 2, the dashboard
   grows a second `levels` row, and the pooled headline stops being readable as a percentage.

---

## 8. Rollback

The merge is one commit and the install is one file, so both reverse cleanly.

**Code:** from `F:\Github\ULTRAKILL-AI`, with the games stopped,

```powershell
git reset --hard <the hash written down in §2>
```

Nothing under `python/models/` or `python/runs/` is touched by the merge, so no checkpoint or log is at risk —
provided §1's commit of the live run's outputs happened first. If it did not, use `git revert -m 1 <merge>`
instead, which leaves the working tree's untracked files alone.

**Mod:** rebuild and reinstall from the rolled-back tree — `& "C:\Program Files\dotnet\dotnet.exe" build -c
Release` — and confirm the installed DLL is back to ~58,880 bytes. **The mod must match the Python.** The two
halves degrade gracefully in both directions (new Python reads a missing `altars`/`items`/`needs_item` as "no
altars"; an old Python ignores the new fields), and there is an offline test for each, so a mismatch is survivable
rather than fatal — but it is not a state to run a training session in.

**The live run:** resume the pre-merge world with
`python scripts\train.py --config configs\campaign_0-1.yaml --resume models\campaign_gates\latest.zip` after a
graceful stop, or from the newest `ckpt_*_steps.zip` after a hard one. `models/campaign_prelude/` can be deleted
outright; it shares nothing with `campaign_gates` except the archive files that were *copied* into it.

**Partial rollback is available and is the likely shape of any trouble.** S4 is dormant by weight, so if only the
skull work looks wrong, nothing needs reverting — just leave the two weights at `0.0`. If only the curriculum
looks wrong, go back to `configs/campaign_0-1.yaml`: with no `levels` key, no curriculum file is opened, the
level never changes and the campaign block is byte-identical to what it has always been.

---

## 9. What this list does not cover

- **§8 checks 1-3 of the spec** — see §5. The probe script exists and is offline-green, but nothing about the
  skull carry has been observed in the running game, so the two item weights stay at `0.0` either way.
- **Levels beyond the prelude three.** `campaign_prelude.yaml` lists 0-1, 0-3 and 0-4 only. 0-2 is held back for
  the skull carry and 0-5 has no goal room in its door graph at all (the Cerberus kill opens the exit), so it has
  no route signal to train against.
- **The checkpoint-chain route fallback for Tier D levels, boss end conditions, more than 5 game instances,
  trams and water** — all out of scope by decision.
- **Gate phase 2 gives a usable `hops` on only three levels** (1-1, 5-3, 8-1). Elsewhere the widened altar door's
  rooms are outside the room graph, so `needs_item` exists but nothing on the route carries it. That is the
  out-of-scope checkpoint-chain fallback, not a defect.
