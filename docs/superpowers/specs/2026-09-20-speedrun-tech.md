# Speedrun technique as capability — approved design spec

**Date:** 2026-09-20. **Status:** approved design, nothing built. **Scope:** `Level 0-1` record chase
(run `spec_0-1_speed`), but every part of it is level-agnostic.

## The instruction

The user, 2026-09-20, verbatim:

> make sure the bot knows about the coin rocket jumping and all the other ways people speedrun these games to
> make the time go by faster

**Interpretation agreed by the lead, and the line this spec holds to.** Still **no recorded human routes and
no demonstrations** — that rule (CLAUDE.md, "It learns alone") is untouched. What is imported is human
**knowledge of techniques**, delivered as **capability**: actions the agent can express, observations the
techniques need, and a safe way to make discovery likely. The agent still has to decide when to use them.

One principle falls out of that line and settles several arguments below:

> **Macro what is unreachable TIMING. Never macro what is unreachable AIM.**

A mod-executed input pair that lands inside an 8 ms window is technique knowledge — a human's fingers do the
same thing. A mod that steers the crosshair onto a moving core so the shot connects is **auto-aim**, which is
assistance, not knowledge. Note that the game's own `autoAim` and `majorAssist` prefs default false and the
mod does not patch `GetBool`, so the project has kept that line cleanly so far. This spec keeps it.

---

## 1. The three facts that shape everything else

### 1.1 The tightest windows are shorter than one decision

The agent decides every **66.7 ms** (`fixed_fps: 30`, `frameskip: 2`). Against that, from `NewMovement`:

| Mechanic | Window | vs one decision |
|---|---|---|
| Super Slide Jump (`TrySSJ`) | **8–32 ms**, quantised into three 8 ms buckets | **half a decision — unreachable** |
| Coyote time (`gc.sinceLastGrounded < 0.06f`) | 60 ms | unreachable as a deliberate act |
| Slide-start grace after a jump (`lastJump < 0.06f`) | 60 ms | unreachable |
| Wall-SSJ slide grace (`ssjMaxFrames * 0.008f`) | 32 ms | unreachable |
| Slide-jump grace (`currentTime - slideTimestamp < 0.1`) | 100 ms | **reachable**, 1.5 decisions |
| Dash-jump gap (`> 0.008f * ssjMaxFrames`) | > 32 ms | **reachable** on consecutive steps |
| Jump cooldown (`Invoke("JumpReady", 0.2f)`) | 200 ms | 3 decisions, fine |
| Slam-bounce window (`superJumpChance` 0.1 + `extraJumpChance` 0.306) | ~406 ms | **6 decisions, fine** |

`TrySSJ` computes `(int)((jumpTimestamp - slideTimestamp) / 0.008)` and rejects bucket 0 and bucket
`>= ssjMaxFrames` (4). `ActionInjector.Queue()` builds **one** `KeyboardState` and calls
`InputSystem.QueueStateEvent` once per frame, so a step that drops `slide` and adds `jump` puts release and
press in the **same event**: `num = 0`, `if (!(num > 0.0)) return;`. Same-step SSJ is structurally impossible,
not merely unlikely.

### 1.2 That window is **real wall-clock time, not game time**

`jumpTimestamp` and `slideTimestamp` are `ctx.time` — the Input System event timestamp, which
`InputRuntime.currentTime` sources from the native wall clock. `Time.captureDeltaTime` does not touch it.

Two consequences, and the second **kills a fallback the first draft of this design relied on**:

- The across-step gap is not 66.7 ms of game time, it is whatever the machine takes. The live run reports
  `steps_per_s` 161.3 over 12 envs = ~13.4 decisions/s per env = **~74 ms real** per decision. Still rejected,
  but for a load-dependent reason, and it drifts with fleet load. Whatever SSJ the agent gets today is luck
  whose rate changes when the machine gets busier.
- **Raising `fixed_fps` does not help.** Changing `captureDeltaTime` changes game time per frame; it does not
  change the real interval between two queued events. The "run at 60/4 and land bucket 2 for free" fallback is
  **invalid** and is not in this design. Explicit timestamps are the only route, not merely the best one.

### 1.3 Path length, not top speed, is 0-1's dominant cost

Live at 29.01M steps: median fresh time **158.6 s**, best **81.5 s**, `completed` 0.89, `gates_reached` 9.36
of 10, `wedged_steps` 0, `deaths` 0.50. The agent is not dying, not wedged, and not over-fighting (40 kills of
106). At a measured mean horizontal speed of ~18.4 u/s over ~169 s of level time, it travels on the order of
**3,000 m** in a level whose exit sits **~195 m** from spawn — roughly an order of magnitude further than it
needs to.

**Every technique in this document is a multiplier on that path. None of them shortens it.** The spec says so
again, louder, in section 8.

---

## 2. Technique catalogue

Status codes: **EXPR** = expressible with today's action space; **OBS** = expressible but the agent is blind
to the state it needs; **MACRO** = window shorter than one decision, needs mod-side timing; **INPUT** = needs a
new action dimension; **OUT** = out of reach or not worth it here.

| Technique | What it buys | Real window | Needs | Status |
|---|---|---|---|---|
| Slide | the spine of all ground movement; `preSlideSpeed` clamped to 3 | hold | — | **EXPR** |
| Slide jump | preserves momentum; `StopSlide` floors it at `max(24, preDashSpeed)` | 100 ms grace | — | **EXPR** (measured *below* chance) |
| **Super Slide Jump** | `+0.5 × walkSpeed × 2.75 × 3 × fixedDeltaTime` along `dodgeDirection`, clamped to 100 u/s | **8–32 ms** | timing | **MACRO** |
| **Super Wall Jump** | same at `speedMultiplier` 0.75 — stronger, and it *redirects* momentum round a corner | **8–32 ms** | timing + `wall_available` | **MACRO + OBS** |
| Wall jump | `currentWallJumps < 3`, resets on ground/enemy step | tap | `wall_available` | **OBS** |
| Dash | `boostCharge >= 100` of 300, regen 70/s, paused while sliding | tap | — | **EXPR** |
| Dash jump | `jumpPower × 1.5` + dash speed; free after an enemy step | gap > 32 ms | — | **EXPR** (measured 1.37× chance) |
| **Slam bounce** | `jumpPower × (3 + slamForce − 1)`, and **× 12.5** once `slamForce >= 5.5`, against × 2.6 | **~406 ms** | `heavy_fall`, `slam_force`, `bounce_window` | **OBS** |
| **Slam slide** | `FixedUpdate` sets `preSlideSpeed = slamForce` on a heavy-fall landing → up to **3× slide speed** | 0.2 s | `pre_slide_speed` | **OBS** |
| Slam storage | keeps `heavyFall` alive through a wall jump or a whiplash pull | — | `slam_storage` | **OBS** |
| Enemy step | `EnemyStepResets()` zeroes wall jumps, rocket jumps, rocket rides; next dash jump is free | tap | — | **EXPR**, unmeasured |
| **Rocket jump** | flat **200** launch, **damage-free**, `safeExplosionLaunchCooldown` 0.5 s | many decisions | own-rocket obs | **OBS** |
| Freezeframe | freezes rockets up to 5 s — makes a rocket jump *placeable* rather than reflex-timed | toggle | variant + obs | **INPUT + OBS** |
| Rocket riding | mount within 2.25 m while falling; steerable | many decisions | own-rocket obs | **OBS** |
| Core eject / core nuke | 24-unit-radius blast, the largest a player can make; one-action arena clear | charge ≤ 1 s | own-core obs | **OBS** |
| **Coin rocket jump** | the coin **aims the detonation for you** — its real value for a coarse-aim policy | many decisions | variant + coin/rocket obs | **INPUT + OBS** |
| Coin shot / split shot / ricoshot | auto-aimed multi-target damage; +1 damage and +1 target per coin | many decisions | variant + coin obs | **INPUT + OBS** |
| Swap cancel | cancels reload recovery; near-double DPS | sequencing | sticky slot | **EXPR once slot is fixed** |
| Whiplash | 60 u/s reel, 0.5 s cooldown; the mechanism for slam storage and rocket boarding | tap | `hook` dim + hook-point obs | **INPUT + OBS** |
| Projectile boost | punch your own projectile: `LaunchFromPointAtSpeed(max(60, |v|))` — clears *and* moves | **1 physics frame** | timing | **MACRO**, deferred |
| Knuckleblaster blast | `Punch.BlastCheck()` requires `heldAction.IsPressed()` | hold | `punch` is a TapButton | **OUT** today — see 3.3 |
| Parry | `NewMovement.Parry` → `FullStamina()`: a free full stamina refill | reactive, ~6 physics frames | incoming-projectile obs | **OUT** for now |
| Railcannon | shared ~16–20 s recharge; one arena clear per level | tap | `fire1` released between shots | **EXPR but broken** — see 3.3 |
| Pipe clip (0-1) | the 4.915 s route | — | — | **OUT — out of bounds, not the target** |

### 2.1 Two corrections to the folklore, from the code

- **A plain rocket jump is damage-free and does not decay.** `Grenade.Explode` sets `rocketExplosion` only
  when `rocket && explosion.damage != 0`. A wall or floor hit instantiates `harmlessExplosion` (damage 0), so
  `rocketExplosion` is never set, and `Explosion.Collide` skips both the `100/((rocketJumps+3)/3)` diminishing
  ladder and the wind-state grant, falling through to a flat `LaunchFromPoint(pos, 200 × pushForceMultiplier)`
  with no `GetHurt`. The diminishing-returns formula appears to be **dead code for player rockets** in this
  build. Rocket jumps are therefore *stronger and cleaner* than assumed, not weaker.
- **Dash i-frames do protect against explosion self-damage.** `NewMovement.GetHurt` returns immediately when
  `invincible && layer == 15 && !ignoreInvincibility`, and `Dodge()` sets layer 15 while `boostLeft > 0`.
  `Explosion.Collide` calls `GetHurt(..., invincible: true, ...)` with `ignoreInvincibility` defaulting false.
  So "dash, then detonate" is damage-free **by construction** — a better guard than any HP threshold.
  (The 35/50 self-damage figures quoted in community guides are the `ultrabooster` path only, which is set by
  `RevolverBeam`'s Railgun/MaliciousFace branch, not by a plain revolver beam.)

---

## 3. Agent audit — what exists, what does not

### 3.1 Inputs the mod already binds that Python never sends

`ActionInjector.ResolveBindings()` binds **`hook`** (a HoldButton) and **`change_fist`** (a TapButton), and
resolves slot 6. `spaces.BUTTONS` is `("jump", "dash", "slide", "fire1", "fire2", "punch")` and
`NUM_WEAPON_CHOICES` is 6. The whiplash is **one list entry away** — no mod rebuild needed for the binding,
only a widened action head. `CampaignPatches` already equips it (`FistControl.CheckFist("arm2")` sets
`HookArm.equipped`).

### 3.2 Observations already on the wire and thrown away

`ObservationBuilder.BuildPlayer` sends **`heavy_fall`**, **`weapon_variation`** and **`slot_counts`**.
`spaces.pack_observation`'s 17-float player block packs none of them. The agent is blind to being mid-slam —
the state every slam technique is built on — and blind to which variant it is holding, which every coin, core
and pump technique depends on. This is free ground.

### 3.3 Three hold-vs-tap defects worth recording

`HoldButtons` are queued every frame of the step; `TapButtons` and the slot key only on frame 1.

- **`Railcannon.Update` fires on `Fire1.WasPerformedThisFrame`**, and `fire1` is a hold button, so holding it
  across steps fires **once**. A second shot needs a step with `fire1` off.
- **Marksman coin toss fires on `Fire2.WasPerformedThisFrame`** — holding `fire2` throws one coin, not a stream.
- **`Punch.BlastCheck()` requires `heldAction.IsPressed()`** at an animation event, and `punch` is a tap held
  for one frame. **The Knuckleblaster blast wave can never fire today.** Fixing it means moving `punch` to
  `HoldButtons`, which changes an existing channel (currently charged `punch: 0.01` per press) for no 0-1 gain.
  Recorded, out of scope.

### 3.4 The slot channel is probably suppressing the agent's own fire

`GunControl.SwitchWeapon` with `targetSlotIndex == currentSlotIndex` reads `PrefsManager`'s
`WeaponRedrawBehaviour`, whose default is **0 = cycle to the next variation**. So pressing the equipped slot
re-draws the weapon. `Revolver.OnEnable` sets `shootCharge = 100f` (an accidental swap-cancel) **but also
`gunReady = false`**, which only the `ReadyGun()` animation event clears — and `Revolver.Update` gates firing
on `gunReady`.

A probe of the **promoted** `Level_0-1.zip` measured the policy pressing its already-equipped slot on
**76.0 %** of steps (39,371 of 51,772). If that holds for the live policy, the agent is keeping its weapons
permanently in the draw animation and suppressing its own primary fire most of the time.

**That 76 % is from a different checkpoint and `status.json` carries no slot histogram to corroborate it.**
Re-measuring it is stage S0 below, and the size of S5's claimed gain rests on it.

### 3.5 What is unlocked

`CampaignPatches.OverridePrefInt` rewrites `weapon.*` pref reads from 0 to 1, and `GunSetter.CheckWeapon`
treats 1 as "add `prefabs[0]`" — the **standard** variant. Value 2, which adds the alternate, is deliberately
left alone. So **all 15 standard variants** are equipped (Piercer / Marksman / Sharpshooter, Core Eject /
Pump Charge / Sawed-On, Freezeframe / S.R.S. / Firestarter, …) and **no alternates**. All four arms are
equipped, with the Feedbacker active at level start.

Usefully: **Core Eject, Piercer, Electric Railcannon and Freezeframe are all variation 0** — exactly what a
sticky slot leaves you holding. Only the Marksman needs a deliberate variant switch, which is why the
`variant` dimension can be deferred without blocking the core and rocket work.

---

## 4. Final design

### 4.1 Action layout

Current: `ACTION_NVEC_CAMPAIGN = (3,3,2,2,2,2,2,2,6,11,7,3)` — 12 dims, 45 logits.

**Append only**, so every existing logit row keeps its index — the invariant both `transfer_weights.py` and
`add_look_mode.py` rely on:

| dim | name | values |
|---|---|---|
| 12 | `macro` | 6 — `{0 none, 1 ssj, 2 ssj_wall, 3 core_nuke, 4 rocket_down, 5 coin_rocket}` |
| 13 | `variant` | 4 — `{0 keep, 1 variant0, 2 variant1, 3 variant2}` |
| 14 | `hook` | 2 |

New: **15 dims, 57 logits** (+12 rows). At the break, macro values 3–5 and the **whole `variant` dim** are
**refused by the mod**, so the migration is exactly behaviour-preserving (see 4.4). They are reserved headroom:
enabling them later is a config flip, not a second break.

`variant`'s fallback, if `SelectVariant1/2/3` turn out to have no default keybinding: call
`GunControl.SwitchWeapon(slot, variation, ...)` directly — it is public. Weapon selection has no
Harmony-inlining hazard, so this is the one place the mod may legitimately bypass the virtual-device pipeline.
Document it there.

**Deliberately excluded:** `change_fist` (the Feedbacker is the arm you want on 0-1, and the Knuckleblaster's
blast cannot fire anyway — 3.3), `punch_hold`, `next_variation`, and parry.

### 4.2 Macros — two, both inside the existing 2-frame step

Only SSJ and wall-SSJ have windows shorter than a decision. `core_nuke`, `rocket_down` and `coin_rocket` are
**ordinary multi-decision sequences** (five, three and about seven decisions respectively) whose real blockers
are variant selection and projectile observations — a macro fixes neither, and for the two that must land a
hitscan on a *moving* object a macro would have to aim, which section 0's principle forbids. They keep their
reserved `macro` values and are delivered as **capability** at S10 instead.

With those cut, **no step ever changes length**, and that deletes the whole frame-scripting apparatus:
`obs["frames"]`, `env._frames`, a `time_scale` in `compute_reward`, and frame-denominated `stuck`/`max_steps`.

**The one new primitive.** Replace `InputSystem.QueueStateEvent(keyboard, state)` with the timestamped
overload `QueueStateEvent<TState>(InputDevice, TState, double time = -1.0)`, which exists in the installed
`Unity.InputSystem.dll`. Base timestamps on `InputState.currentTime` — the same clock `NewMovement` compares
against.

**M1 `ssj` — one frame, two events.**
Precondition: `sliding`, `!jumpCooldown`, and `(onGround || canJump)`.
Frame 1 queues release-slide at `T` and press-jump at `T + 0.012` (bucket 1). Frame 2 is the ordinary action
with jump released so the next press re-triggers. The agent keeps move, look and every other button.

This works *because* of the `Update` ordering: `HandleInputs()` runs at line 686, `HandleSlideState()` at 740,
and `SlideCancelled` **only records the timestamp** — it does not stop the slide. So `Jump()` is reached with
`sliding` still true, takes the `if (sliding)` branch, calls `StopSlide()` (which refreshes
`velocityAfterSlide`), and *then* calls `TrySSJ`.

**M2 `ssj_wall` — two frames, and it must be two.**
Precondition: `!onGround`, `currentWallJumps < 3`, an active `WallCheck`, `!jumpCooldown`, `falling` true
(otherwise `HandleInputs`' first Jump block fires, sets `jumpCooldown`, and `WallJump` is skipped), and no
enemy columns under the feet (which would make it an enemy step instead — report `degraded`, do not claim a
wall SSJ).
Frame 1 queues the slide **release** alone, so *that frame's* `HandleSlideState` runs `StopSlide()`.
Frame 2 queues the jump at the release timestamp + 0.012.

**Why two frames is not optional.** `TrySSJ` does not add to current velocity — it **overwrites**:
`rb.velocity = velocityAfterSlide + direction * num5`. `velocityAfterSlide` is written in exactly one place,
`StopSlide`, as `dodgeDirection.normalized * Mathf.Max(24f, preDashSpeed.magnitude)`. `WallJump` never calls
`StopSlide`, and it reaches `TrySSJ` through the `sliding ||` disjunct — so a one-frame wall SSJ lands on a
**stale** `velocityAfterSlide` from the *previous* slide, in that old direction. A wall SSJ at 80 u/s can come
out at ~61 u/s pointing somewhere else: **a speed loss disguised as a technique.** Releasing on frame 1
refreshes it and the two-frame script still fits inside `frameskip: 2` exactly.

**The monotonic timestamp cursor — non-negotiable.** `InputManager.OnUpdate` **silently discards** a state
event whose timestamp precedes the device's `m_LastUpdateTimeInternal`, and `Keyboard` has no state callbacks,
so the drop applies unconditionally. After queueing a jump at `now + 0.012`, the *next* frame's ordinary queue
uses `currentTime` — and the middle frame of a step is only a few ms of real work, so the next event can
easily carry a **lower** timestamp and be dropped, eating a whole frame of the agent's input. `ActionInjector`
must therefore queue every event at `max(InputState.currentTime, lastQueued + epsilon)`, across keyboard and
mouse both. Without this the change degrades input reliability fleet-wide and would read as a policy regression.

**Refusal is a first-class signal.** A refused macro runs the plain action, costs nothing extra, and is
reported: `obs["macro"] = {requested, result: ran|refused|degraded|disabled, reason, ssj_bucket}`. **Do not
charge for a refusal** — a charge teaches the policy to avoid macros rather than to learn their preconditions.
Refusal rates per macro go into `env._behaviour` so a macro refused 99 % of the time shows up as a design bug.

The mod reports the **actual SSJ bucket**, which matters for the reward gate in 4.5.

**One implementation note the first draft missed:** `ApplyFrame()` queues input for the *next* frame
(`EpisodeController`'s own comment says so), so any script longer than `frameskip` would need `frameskip + 1`
frames for its last entry to be consumed and observed. Both macros here are `frameskip`-length, so the
off-by-one never arises — but it is a trap for whoever adds a longer one.

**On the time-accounting hole,** which the first draft called the most farmable thing in the plan: it is real
but narrower. `rewards.py` charges `time` flat per decision and `stuck`/`max_steps` count decisions, so a long
macro would distort the shaping term and silently extend the episode budget. But the two things that decide
promotion — `completion_bonus` and the driver's rung median — both read the **game's own clock**
(`campaign.seconds`, `valid_official_seconds`), so the objective was never farmable this way. With equal-length
macros the question is moot; if a longer macro is ever added, fix it then, and fix it for the **episode budget**.

### 4.3 Observations: 479 → 530

Fifty-one floats appended **after** the campaign block as a new `TECH_BLOCK`, so indices 0–478 keep their
meaning. All are `.get()`-defaulted to 0.0, so an old DLL — or a new one with the feature flags off — produces
exactly the vector the migration initialised. **That is what makes S9–S11 config flips rather than
re-migrations.**

- **A. Movement (12), 479–490** — `heavy_fall`, `slam_force`, `bounce_window` (any of `superJumpChance`,
  `bounceChance`, `extraJumpChance` positive), `coyote`, `wall_jumps`, **`wall_available`**, `boost`,
  `boost_left`, `pre_slide_speed`, `jump_cooldown`, `slide_since`, `riding_rocket`.
- **B. Weapon (10), 491–500** — variant one-hot, `gun_ready`, `pierce_charge`, `coin_charge`, `rai_charge`,
  `rocket_freeze`, `punch_stamina`, `core_ready`.
- **C. Own projectiles (18), 501–518** — nearest three of the player's own live objects, 6 floats each
  (relative xyz in the player's yaw frame, distance, age, kind), from `CoinTracker.revolverCoinsList` and
  `ObjectTracker.grenadeList` / `cannonballList`. **Without this block every coin, core and rocket technique
  is invisible to the policy at the exact moment it must act.**
- **D. Macro feedback (3), 519–521** — ran, refused, bucket.
- **E. Hazard-trigger rays (8), 522–529** — the existing rays pass `QueryTriggerInteraction.Ignore`, so lava,
  pits and crushers are invisible. Honest sizing: 0-1 shows 0.50 deaths per episode, so this buys **almost
  nothing on 0-1**. It is taken now only because the break is the free moment to reserve the slots.

**`wall_available` is the one field the 16 horizontal rays cannot express**, because a `WallCheck` is a
component, not a distance. It is M2's precondition.

Only block A and D are emitted at the break; B, C and E are reserved and wired to 0.0, and their mod-side
emitters are written at their own stages. Six of the source fields are private (`jumpCooldown`, `slamStorage`,
`boostLeft`, `dashStorage`, `wcGroup`, `velocityAfterSlide`, `Shotgun.sinceLastCore`) and need
`AccessTools.FieldRefAccess`, which `ObservationBuilder` already uses — about a line each, not zero.

**Two call sites break silently and must be fixed in the same commit:**
`scripts/probe_rollout.py` line 52 uses `TARGET_SLICE = slice(-CAMPAIGN_BLOCK + 5, -CAMPAIGN_BLOCK + 13)` —
**negative indices from the end** — and will quietly read the wrong slots after any append; change to absolute.
`scripts/add_look_mode.py`'s shape guard will now reject every pre-break checkpoint, which is correct
behaviour; say so in its docstring so nobody reads it as a regression.

### 4.4 Weight migration

`scripts/add_tech_heads.py`, modelled on `add_look_mode.py`, reusing `transfer_weights.py`'s `SpacesEnv`.

1. **First layers** `(512, 479) → (512, 530)`: copy columns 0–478 verbatim, **zero** 479–529, biases unchanged
   with **no mean-fold**. This is the key difference from `add_look_mode`, which *repurposed* eight inputs that
   carried real values and so measured 0.0262 mean KL and 29.6 % greedy change. Nothing is repurposed here.
2. **Second hidden layers and the value head:** copied verbatim. Do **not** reset the value head the way
   `transfer_weights.py` does — that script resets because Cyber Grind's reward scale is unrelated to the
   campaign's. Here the scale is unchanged at the break, and discarding a 29M-step value function is strictly
   destructive. This is precisely why `level_complete: 300` must land **before** the break: it is the one
   change that moves the value scale, and the head should have re-fitted by then.
3. **Action head** `(45, 512) → (57, 512)`: rows 0–44 copied exactly; new rows get **zero weights** and biases
   `ln 45 = 3.807` / `ln 17 = 2.833` / `ln 9 = 2.197`, giving `P(none) = 0.90`, `P(keep) = 0.85`,
   `P(no hook) = 0.90`. Zero weights make the new heads state-independent at init, so the bias alone sets the
   marginal. At 2 % per macro that is ~54 attempts of each per episode per env.
4. **Optimizer:** carry Adam's moments, keeping the 0-dim `step` tensor so bias correction continues.
   **Fix a latent bug while copying:** `add_look_mode.carry_optimizer` pads only along axis 0
   (`padded[: value.shape[0]] = value`), but the new first layer grows along **axis 1**. Use
   `padded[tuple(slice(0, s) for s in value.shape)] = value`. The existing code **fails closed** — the bad
   assignment raises, the broad `except` catches it, and it silently falls back to a **fresh optimizer** — so
   it would not corrupt moments, but losing Adam state at 29M steps would be invisible. **Make the caller log
   loudly when the carry fails.**
5. `num_timesteps` and `_n_updates` carried, so the step axis stays continuous for the driver's rung logic.

**Offline verification** (`python/tests/test_add_tech_heads.py`, no game), over every raw state in
`runs/probe_0-1_record/stoch_*.jsonl` packed under the new layout with 479–529 forced to 0.0:

- `mean_kl` over the **12 shared dims == 0.0** to 1e-6, `greedy_changed == 0.0`, `mean_abs_dv == 0.0`. Anything
  non-zero is a copy bug, **not** a tolerable displacement — a strictly stronger assertion than
  `add_look_mode`'s `mean_kl <= 0.03`. It holds because SB3's `MultiCategoricalDistribution` splits logits in
  order, so appended dims append rows; because zero columns meet zero-valued inputs; and because there is **no
  `VecNormalize`** anywhere in the project's own code, so there is no `obs_rms` of shape (479,) to migrate —
  the usual killer in an observation widening simply does not exist here.
- New-head marginals within 1e-3 and **state-independent**.
- Total entropy == source + **1.3986** nats (macro 0.4860 + variant 0.5875 + hook 0.3251). **Print the measured
  total and write it into the config** as the new `ent_floor` rather than guessing; pin it in
  `test_specialists_config.py` beside today's 6.5.

**Behaviour preservation is conditional on `variant` being refused at the break.** The test checks the shared
dims' distributions, but a sampled action now has 15 entries, and a live `variant` at `P(!= keep) = 0.15`
would change the held variant on 15 % of steps from the first rollout — an environment change the test cannot
see. Refuse it at S7; it costs nothing, because the variants worth holding are all variation 0 (3.5).

**`ent_floor` is hygiene, not an emergency.** `EntropyFloorCallback` is one-directional above its band: over
`floor + 1.0` it only decays `ent_coef` **back towards `base`, never below**, and `ent_coef_live` is already
at `base` (0.004). Leaving the floor at 6.5 would not "loosen every head" — decaying `ent_coef` makes a policy
*sharper*. What it would do is stop the floor protecting anything until the policy had already sharpened by
~1.4 nats. Re-base it.

**Integration hazard, and the guard that matters.** `campaign_driver`'s round init picks the stage's newest
checkpoint and would happily pick a 479-wide `ckpt_*_steps.zip`, then re-exec `train.py` in a loop with 12
games burning a commit-bound box. Two guards: quarantine the old checkpoints under `pre_tech/`, **and** make
`train.py`'s resume path refuse a shape mismatch with a message naming the migration script. The second one
survives the next break too.

### 4.5 Discovery

**Primary mechanism: initial-logit management** (4.4 step 3). Exploration volume is not the bottleneck;
credit assignment across ~2,700 decisions is — which is why the gamma stages come first.

**Secondary: a capped, decaying, benefit-gated bonus.** Reward key `tech`, paid only when a macro **ran**, the
next observation shows a **measured** benefit, the player did **not lose HP** on that step, and the per-episode
cap is unexhausted. Weight `tech: 0.5` per event, `tech_cap: 10.0` per episode, decayed to zero over 3M steps
— well inside one rung, so no rung is judged on a stage where the bonus is still large.

**Gate on the mod-reported SSJ bucket, not on a speed delta.** The obvious gate — "horizontal speed rose by
≥ 12 u/s" — is **perversely signed**, because `TrySSJ` overwrites with `velocityAfterSlide + bonus` and
`velocityAfterSlide` floors at 24:

| speed before | after SSJ | a "+12 over max(prev, 24)" bar | pays? |
|---|---|---|---|
| standstill | 24 + 24.75 = 48.75 | 36 | **yes** |
| 80 u/s | clamped to 100 | 92 | yes |
| 90 u/s | clamped to 100 | 102 | **no** |

It pays most reliably when the agent is **slow** and stops paying exactly when it is fastest — it rewards
stopping and re-accelerating. Since the mod knows the bucket, gate on that: the mechanism itself, which cannot
be gamed by a proxy.

**Farmability bound**, against the live numbers (episode 2,720 decisions, `time` 0.02, `death` 12.0 effective,
`SPEED_BONUS_MIN` 0.25):

| invariant | margin |
|---|---|
| finishing > not finishing | completion floor `0.25 × 100 = 25.0` (→ **75.0** at `level_complete: 300`) vs a 10.0 cap; and a farming episode still pays `0.02 × 2720 = 54.4` of clock |
| faster > slower | 158.6 s → 120 s is worth ~24 of bonus + ~11.6 of avoided clock = **~36** vs a 10.0 maximum tech advantage |
| living > dying | `death` 12.0 **per death** vs 10.0 **per episode**, plus the no-pay-on-HP-loss rule |
| a refused macro is free | no charge, so preconditions are learned rather than macros avoided |

All three strengthen at `level_complete: 300`. The cap never needs to move. Suicide-to-respawn is not an
exploit on 0-1 (all six checkpoints are behind the player).

**One second-order effect that changes the stage order:** more mobility raises the *rate* at which `novelty`
and `cells_new` pay. `gate_approach` is a ratchet on closest approach and is bounded by geometry, but novelty
is not bounded the same way. **That makes S4 (halving the milestone pile) a prerequisite of the mobility work,
not an independent experiment to run after it.**

**Drills are rejected** for now: they need reset/teleport plumbing and a curriculum the driver has no concept
of, and a drill placed at the wall the designer knows is there is a recorded human route wearing a different
hat. Reconsider only if, 1M steps after the break, the macro heads have not moved off their 2 % priors.

**What the bonus cannot do:** teach *when* a technique is strategically worth using. An SSJ into a wall still
scores as a valid bucket. It only makes the technique frequent enough for the value function to judge it —
which is the stated goal: import technique **knowledge** as capability, not routes as demonstrations.

### 4.6 Testing without stopping training

**A second BepInEx tree in the same game folder, selected per process by a Doorstop argument.** No copy of the
game, no elevation, no touching the live DLL.

The installed `winhttp.dll` is Unity Doorstop 4.5.0 and accepts `--doorstop-target-assembly`. BepInEx derives
its whole root from that path (`PreloaderRunner` takes the preloader's parent directory two levels up), so a
preloader at `<game>\BepInEx-test\core\BepInEx.Preloader.dll` yields a fully isolated root with its own
`plugins\`, `config\`, `cache\` and `LogOutput.log`. The tree is ~1.8 MB. The game folder grants
`BUILTIN\Users: FullControl`, so no elevation.

1. Copy `BepInEx` → `BepInEx-test`; clear its log and cache.
2. `dotnet build -c Release -p:InstallPlugin=false`, then copy the DLL into the test tree **by hand**. Prefer
   this over `-p:PluginDir=...` so there is **no code path** by which the build can write the locked live DLL.
3. Launch **one** game by hand on **47812** with `-aibridge-nosteam` and the Doorstop override. Never
   `games.py launch` or `stop` — both call `stop_all()` and would take down all twelve.
4. **Gate everything on one check first:** read `BepInEx-test\LogOutput.log` and confirm the plugin loaded
   from the **test** tree, and that the hello reply carries the new `features` array. If the Doorstop argument
   were ignored the game would load the **live** plugin on 47812 and every result would be a false negative.
5. Validate with `probe_rollout.py --port 47812` plus a new `scripts/macro_check.py`.

**The four questions this test exists to settle:**

1. Does `QueueStateEvent`'s explicit timestamp reach `ctx.time` unchanged on Unity 2022.3.29 Mono?
2. What is `InputSystem.settings.updateMode`? ULTRAKILL reads `WasPerformedThisFrame` from `Update`, i.e. the
   default dynamic mode, in which a **future-dated event is processed in the current update** — good news for
   forward offsets. If it were ever `ProcessEventsInFixedUpdate`, events would be time-sliced and the whole
   macro design changes.
3. Does the monotonic cursor hold — queue at `now + 0.012` and confirm the **next** frame's ordinary event is
   not dropped?
4. Log `NewMovement.walkSpeed` and `Time.fixedDeltaTime`. **Both are serialized and cannot be read from the
   decompiled C#, so every u/s figure anywhere in this spec is derived, not read.**

**The cheapest instrument needs no macro at all:** `TrySSJ` reads a `ssjIndicator` pref and, when it is on,
prints the bucket and the exact `+{num5:f0}u/s` gained as a subtitle. Turning that on in the test instance
converts the entire question from an inference into a readout.

Memory: one extra game is ~1.2 GB on a commit-bound box. `mem_guard.py --dry-run` first, keep the window under
30 minutes, stop it by pid.

### 4.7 The single full-pause install

1. `New-Item runs\specialists\DRIVER_PAUSE` — **before** anything else. A deliberate Ctrl+C is
   indistinguishable from a crash and the driver will relaunch everything underneath you.
2. Per the 2026-09-20 lesson, a shared-reward-path change must not straddle a round that will be judged: land
   the new plan first, write `END_STAGE`, let that round record "unfinished — ended by operator".
3. Stop driver, trainer and workers by pid; then `games.py stop` (safe — no run is live). The live DLL is
   locked while games run.
4. `dotnet build -c Release` — installs. ~30 s.
5. Run `add_tech_heads.py`; confirm the printed `mean_kl` is **0.0**; read the measured entropy total;
   quarantine the 479-wide checkpoints under `pre_tech/`.
6. Edit `configs/specialists.yaml`: the new `ent_floor`, the feature flags, the `tech` block.
7. Remove `DRIVER_PAUSE`; start the driver through `runs\start_driver.cmd`.

**Downtime ~15–25 minutes**, in line with the 35–50 min the 2026-09-17 integration budgeted for a larger list.
The relaunch also resets the memory leak. Afterwards: `scripts/check_run.py`, expect ALERTS none, 12/12
listening, five helpers up including `mem_guard.py`.

---

## 5. Staged plan

Nothing before S6 needs a pause, a DLL or a migration. **S7 is the only break**, and with the frame-scripting
engine and the three non-timing macros removed it is roughly a quarter of the change first proposed.

| # | Stage | Expected on 0-1 | Judged on |
|---|---|---|---|
| **S0** | **Measure first.** Re-measure the equipped-slot press rate on the **live** policy; add slot/variant counters to `env._behaviour` and the dashboard. | none — it is a measurement | a number exists; S5's premise is confirmed or dropped |
| **S1** | `gamma` 0.998 → **0.999**, `gae_lambda` 0.95 → 0.98 | **largest single move of the median available, and it is free** | median over 50 fresh; `explained_variance` must not collapse; `approx_kl` vs `target_kl` 0.03 |
| **S2** | `gamma` → **0.9995**, only if S1's explained variance held | further, smaller | same |
| **S3** | `level_complete` 100 → **300** | modest alone, large with S1–S2 | median; must precede the break so the value head re-fits |
| **S4** | Halve the clock-independent pile: `gate` 15→7.5, `door_unlock` 15→7.5, `checkpoint` 10→5, `gate_approach` 0.15→0.075, `novelty` 0.2→0.1 | **biggest lever on the reward mix in the list** | median **and** `completed` must not fall; own round, must not straddle a judged one |
| **S5** | **Sticky slot** (Python only): drop the slot press when it equals the equipped slot | if the 76 % holds, plausibly the **largest single win**, and no break | `kills_per_min`, `firing_on_target_frac`, median; explicit revert criterion |
| **S6** | **Private-game verification** on 47812 via the `BepInEx-test` tree, `ssjIndicator` on | none — GO/NO-GO for S7 | the four questions in 4.6 answered |
| **S7** | **THE ONE BREAK.** Mod v0.8.0 (monotonic cursor, M1 + M2, FieldRef readers); obs 479→530 (blocks A+D live); action 12/45→15/57 with macro 3–5 and `variant` **refused**; `add_tech_heads.py`; `ent_floor` re-based; `train.py` shape check; `probe_rollout.py` absolute indices | the SSJ pair is the only thing in the game unreachable at 15 Hz | `mean_kl == 0.0` offline; then median at ≥ 400k steps |
| **S8** | `tech` bonus on, gated on the reported bucket, cap 10.0, decayed over 3M steps | makes the macros frequent enough to be judged | macro-use rate off the 2 % prior; median; the three invariants |
| **S9** | Observation block **B** + the `variant` dim (Marksman) | enables the coin family | median; variant histogram |
| **S10** | Observation block **C** (own coins / cores / rockets) + `hook` | **core nuke, rocket jump and coin rocket jump become reachable here** — as ordinary multi-decision sequences, no macro, no auto-aim | median; kills/min; deaths must not rise |
| **S11** | Hazard rays (block **E**) | ~nothing on 0-1 | deferred to a level whose deaths are hazards |
| **S12** | Brutal (difficulty 4) and the remaining rungs | per CLAUDE.md | after completion is reliable under the new action space |

**Interleaving with what was already planned.** S1–S4 *are* the compatible steps already on the books
(gamma + lambda, `level_complete` 300) plus the milestone halving, reordered so that S4 precedes the mobility
work for the novelty-rate reason in 4.5. `ent_floor` is no longer an independent stage: it is re-based **inside**
S7 from a **measured** number.

### 5.1 The arithmetic behind putting gamma first

At `gamma` 0.998 the horizon is `1/(1-γ)` = 500 decisions = **33 s**, while the episode is ~2,720 decisions.
Measured live (`level_complete` 72.68, `time` −54.41 over the window):

| γ | horizon | γ^2720 | completion bonus at t=0 | discounted time stream |
|---|---|---|---|---|
| 0.998 | 33 s | 0.0043 | **0.31** | −9.96 |
| 0.999 | 67 s | 0.0658 | **4.78** | −18.68 |
| 0.9995 | 133 s | 0.2565 | **18.64** | −29.74 |

Undiscounted, the clock-sensitive share is `(72.68 + 54.41) / 473.39` = **26.8 %**. **Discounted at episode
start it is about 6 %.** The agent is not ignoring the clock because the weights are wrong; it is ignoring the
clock because at γ = 0.998 the finish line is **four horizons away** and worth 0.31.

**Go in two steps, not one.** At the time of writing `approx_kl` is 0.0251 against `target_kl` 0.03,
`clip_fraction` 0.26 and `explained_variance` 0.57 — PPO is already close to early-stopping its epochs.
Quadrupling the effective horizon in one move on a run in that state is how you get an unattributable
regression. 0.999 alone is a 15× improvement in the bonus's visibility.

---

## 6. What each stage must not do

The project's standing rule applies to all of it: **never judge a policy edit from an offline probe alone**,
confirm against live `status.json`, revert if the headline metric is worse at the next two checks, change one
thing at a time, and give it **≥ 400k steps**. S1, S4, S5 and S7 are each one thing.

---

## 7. Open questions for the lead or the user

1. **Does the two-step gamma ladder (S1 then S2) replace the planned single move to 0.9995?** This spec says
   yes, on the PPO health numbers. It costs one extra stage.
2. **Is S4 (halving `gate`, `gate_approach`, `checkpoint`, `door_unlock`, `novelty`) approved as its own
   round?** It is the biggest lever on the reward mix and the prerequisite for the mobility work, but it
   weakens the ladder that got 0-1 to 0.89 completion in the first place.
3. **Is the private-game test (S6) approved to run beside the live fleet?** One extra ~1.2 GB game on 47812 on
   a commit-bound box, under 30 minutes, via the isolated `BepInEx-test` tree.
4. **Is "macro timing, never macro aim" the right line?** It is what defers `core_nuke` and `coin_rocket` from
   S7 to S10. A looser line would let the mod aim the final shot and make both work sooner — and would, in this
   spec's reading, cross from technique knowledge into assistance.
5. **Is the ~15–25 minute full pause at S7 acceptable**, and is it wanted at a rung boundary rather than mid-rung?
6. **`coin_rocket` specifically — the technique the user named — is scheduled last and is honestly low value on
   0-1** (little verticality; a one-weapon rocket jump does the same job). It is kept for the vertical 4-x/5-x
   levels. Confirm that answers the instruction, or say it should be pulled forward.

---

## 8. What stays out of reach, and the revised floor

**The honest ceiling, stated plainly rather than buried.**

- **SSJ + wall-SSJ, if they become habitual, plausibly move sustained speed from ~18 u/s toward 25–30 u/s and
  the median from ~159 s to roughly 105–120 s.** That clears rung 1 (120 s) and puts rung 2 (100 s) in reach.
- **Sticky slot and the slam family are the cheapest real gains in this document** and neither is a macro. The
  slam bounce is *already* fully expressible — a 406 ms window, six decisions wide — and was measured at 0.747
  against a 0.921 chance rate, i.e. **anti-learned**. Four observation floats buy the bounce, the slam slide
  and the dash-jump window, with **no new action, no macro and no mod input change**.
- **Rungs at 60, 42, 30 and 25 s are not reachable by technique at all.** They need a **shorter route**, and a
  memoryless 512×512 MLP cannot represent one. The agent travels roughly an order of magnitude further than the
  195 m straight-line distance to the exit; every technique here multiplies speed along that path without
  shortening it. The honest fix is **memory** (a frame stack or a recurrent policy) or an explicit route
  representation, and **neither is in this design**.
- **The 19.798 s human inbounds record is out of reach.** It is ~106 kills across ~10 arenas in ~20 s — about
  2 s per arena with zero search. That needs a perfect route *and* one-action arena clears.

**Revised floor for the agent as built, with the techniques in: ~105–120 s median, against the ~35–50 s
previously estimated for the agent as built.** The earlier figure was a *speed* floor; it assumed a path
length this agent does not achieve. Best-case single episodes will go well below the median — the current best
is already 81.5 s — but the rung gate reads the **median over 50 fresh episodes**, and that is the number this
plan moves.

---

## 9. Sources

**Code in this build** (`decompiled/`, gitignored — class and method names only, never pasted):
`NewMovement` (`TrySSJ`, `Jump`, `WallJump`, `StopSlide`, `SlideCancelled`, `JumpPerformed`, `HandleInputs`,
`HandleSlideState`, `TryDash`, `TryStartSlam`, `Dodge`, `GetHurt`, `Parry`, `EnemyStepResets`),
`GroundCheck`, `GunControl.SwitchWeapon`, `GunSetter.CheckWeapon`, `PrefsManager` (`WeaponRedrawBehaviour`,
`autoAim`, `majorAssist`, `ssjIndicator`), `Revolver`, `Shotgun`, `Coin.ReflectRevolver`, `Grenade`,
`Explosion.Collide`, `RevolverBeam`, `Railcannon`, `Punch` (`BlastCheck`, `ActiveFrame`, `TryParryProjectile`),
`FistControl`, `HookArm`, `WeaponCharges`, `CoinTracker`, `ObjectTracker`, `PlayerInput`, `AssistController`.
Installed assemblies decompiled with the repo's `.tools/ilspycmd`: `Unity.InputSystem.dll`
(`InputSystem.QueueStateEvent`, `InputState.currentTime`, `InputManager.OnUpdate`'s stale-event drop and
time-slicing branch), `BepInEx.Preloader.dll` (`PreloaderRunner`, `Paths.SetExecutablePath`).

**This repo:** `mod/UltrakillAIBridge/Act/ActionInjector.cs`, `Env/EpisodeController.cs`,
`Env/CampaignPatches.cs`, `Obs/ObservationBuilder.cs`, `Obs/CampaignObserver.cs`;
`python/ultrakill_ai/{spaces,env,rewards,training,times}.py`;
`python/scripts/{transfer_weights,add_look_mode,probe_rollout,campaign_driver,train}.py`;
`python/configs/specialists.yaml`; `docs/level-survey.md` (0-1: 33 rooms, 11 gates, 106/108 enemies, exit
`FinalPit` → `Level 0-2`); `docs/il-records.md`; `docs/project-log.md`.

**Live measurements:** `python/runs/spec_0-1_speed/status.json` at 29,011,018 steps (median 158.6 s, best
81.5 s, completed 0.89, episode 2,720 decisions, the reward-part means and the PPO block quoted in §5.1);
`python/runs/probe_0-1_record/` (the 2026-09-20 probe of the promoted `Level_0-1.zip`, source of the 76 %
slot figure, the 18.41 u/s mean and the slam/slide co-occurrence rates).

**Web** (untrusted data, used only to map code behaviour to community names, and overruled by the code
wherever they disagreed):
`https://ultrakill.wiki.gg/wiki/Movement`, `.../Marksman`, `.../Rocket_Launcher`, `.../Shotgun`, `.../Arms`,
`.../Railcannon`, `.../Parry`, `.../0-1:_INTO_THE_FIRE`;
`https://steamcommunity.com/sharedfiles/filedetails/?id=3287803504`;
speedrun.com API for the leaderboard (`https://www.speedrun.com/api/v1/leaderboards/369p3p81/level/29vl6lqw/5dw44w5k`),
the record `https://www.speedrun.com/ultrakill/runs/zqn8rdrz` (19.798 s, geshem8, 2026-05-07, Inbounds) and
`https://www.speedrun.com/ultrakill/runs/z5433ngm` (4.915 s, Extra Subcategory = None, i.e. out of bounds —
**not** the target).

**Not verified, and named as such:** `walkSpeed` and `Time.fixedDeltaTime` are serialized, so every u/s figure
here is derived (750 × 0.008 reproduces the wiki's 16.5 / 49.5 / 24.75 exactly, and the probe's max horizontal
speed of 99.64 confirms the 100 clamp); whether an explicit `QueueStateEvent` timestamp reaches `ctx.time`
unchanged; `InputSystem.settings.updateMode`; the self-damage of a plain (non-`ultrabooster`) beam detonating a
core; whether `SelectVariant1/2/3` have default keybindings; the 76 % slot figure on the **live** policy; and
the fact that **no public text describes the 19.798 s route** — all top-12 run comments on that leaderboard are
one word or moderator notes, and no route was imported from any of them.
