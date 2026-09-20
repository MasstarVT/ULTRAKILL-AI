# Bridge protocol (v1)

The mod listens on `127.0.0.1:47800` (configurable in `BepInEx/config/masstarvt.ultrakill.aibridge.cfg`).
Launching the game with `-aibridge-port N` makes it a **training instance** on port N: it opens preferences read-only and never writes preferences or save data. A normally launched game moves to the next free port if 47800 is taken.
Adding `-aibridge-nosteam` (mod v0.7.1) additionally hides the instance from Steam: the plugin skips Facepunch's `SteamClient.Init`, so the copy never registers with a running Steam client — no playtime is credited and Steam does not show the game as running. `scripts/games.py` adds it to every training launch by default; `--steam` there leaves an instance visible. The handshake reports whether the skip actually took effect, as `steam_hidden`.
Messages are single-line JSON objects terminated by `\n`, in both directions. Every request gets exactly one reply.

## Control model

- **Idle:** the human plays. The mod answers `hello`, `config` and `get_obs` without blocking the game.
- **AI control:** begins with the first `reset` or `step`.
  - Human keyboard, mouse and gamepad are disabled (unless `block_human_input` is false).
  - Each frame covers a fixed amount of game time (`Time.captureDeltaTime = 1/fixed_fps`).
  - After each step the game **blocks** until the next command arrives.
- **Timing:** the lockstep runs at the end of each frame. A step received at the end of frame N applies from frame N+1, and its obs is built at the end of frame N+frameskip. Hitstop and parry freezes count frames instead of wall-clock time while in control.
- **Leaving AI control:** control goes back to the human on `release`, on disconnect, on the panic key (F8), or when no command arrives for `command_timeout_s`.

Because of the fixed frame time, how long Python takes to decide never changes what happens in the game. With `unlimited_fps`, training runs faster than real time.

## Requests

| type | fields | reply |
|---|---|---|
| `hello` | `protocol` | `{"type":"hello","protocol":1,"mod_version":..,"port":..,"training_instance":..,"steam_hidden":..,"scene":..,"features":[..],"diag":{..}}` |
| `config` | any of the settings below | `{"type":"ok"}` |
| `get_obs` | | an `obs` message (doesn't take control) |
| `reset` | `scene` (e.g. `"Endless"`, `"Level 0-1"`; omit = current), `checkpoint` (bool) | an `obs` with `"event":"reset"` once the player is spawned and has been ready for `reset_settle_frames` frames |
| `step` | `action` | an `obs` after `frameskip` frames |
| `teleport` | `pos` `[x,y,z]` | an `obs` with `"event":"teleport"` (takes control; moves the player and zeroes velocity) |
| `kill` | | an `obs` with `"event":"kill"` (takes control; debug: a lethal `NewMovement.GetHurt(999)` for the in-game death check. With `soft_death` on it is healed like any lethal hit; ignored once the level is over; an error when there is no living player). While in control a dead player stays dead until a `reset`: the game's own restarts (Fire1 or R while dead, the pause menu) are blocked |
| `release` | | `{"type":"ok"}` |

Errors come back as `{"type":"error","message":..}`.

### `hello`: `features` and `diag` (mod 0.8.0)

**The protocol number stays 1.** Every 0.8 addition is optional on both sides: a client that sends no `macro`, no `variant` and none of the `obs_*` flags gets byte-identical messages to 0.7.2 (verified in game on 2026-09-20, 60 steps, zero unexpected keys). `features` is how a client tells the builds apart without parsing a version string:

`monotonic_input_clock`, `macro.ssj`, `macro.ssj_wall`, `obs.move_tech`, `obs.weapon_tech`, `obs.projectiles`, `action.variant`, and `ssj_instrument` when the `TrySSJ` patch applied.

`diag` carries the values the macro design depends on but which **cannot be read from decompiled C#**, because they are serialized in the scene rather than assigned in code: `walk_speed`, `jump_power`, `wall_jump_power`, `ssj_max_frames`, `fixed_delta_time`, `capture_delta_time`, `update_mode` (`InputSystem.settings.updateMode`), `input_now` (`InputState.currentTime`), `unity_time`, and the derived `ssj_bonus_jump` / `ssj_bonus_wall`. The player fields are absent when no player exists, so read `diag` again **inside a level**. Measured on this build: `walk_speed` 750, `jump_power` 90, `wall_jump_power` 150, `ssj_max_frames` 4, `fixed_delta_time` 0.008, `update_mode` `ProcessEventsInDynamicUpdate`.

### config settings

| key | default | meaning |
|---|---|---|
| `frameskip` | 4 | frames per step |
| `fixed_fps` | 60 | game time per frame is `1/fixed_fps`; 0 = real time |
| `unlimited_fps` | true | disable vsync and the frame cap while in control |
| `mute` | true | mute audio while in control |
| `block_human_input` | true | disable real input devices while in control |
| `windowed`, `window_width`, `window_height` | true, 640, 360 | run in a small window while in control (restored on release) |
| `max_enemies` | 16 | enemies included in obs (nearest first) |
| `horizontal_rays`, `ray_length` | 16, 50 | wall distance ring |
| `ground_rays`, `ground_ray_radius`, `ground_ray_length` | 8, 4, 30 | pit detection ring |
| `reset_timeout_s` | 120 | give up on a reset after this many seconds (wall clock) |
| `reset_settle_frames` | 10 | frames the player must be ready before a reset completes |
| `command_timeout_s` | 300 | drop the client if no command arrives for this long |
| `difficulty` | -1 | difficulty the game reads while in control: 0 Harmless, 1 Lenient, 2 Standard, 3 Violent, 4 Brutal; -1 keeps the game's own setting. In memory only. Send it before the level loads: enemies and the player read it at scene start |
| `unwedge` | true | break the absorbing airborne `slowMode` state while in control (see `player.slow_mode` below). A kill switch: send `false` to reproduce the state, never for training |
| `unwedge_frames` | 10 | consecutive **frames** (not decisions) the airborne `slowMode` signature must hold before `unwedge` breaks it; clamped to at least 1, where 1 acts on the first frame. A ground slam started out of a slide wears the same signature while it falls, so the hold is what keeps a real slam from being cancelled. 10 frames is 0.33 s of game time at `fixed_fps` 30 and 0.17 s at 60; the shortest wedge ever measured is 672 decisions |
| `unlock_all_gear` | false | while in control, every weapon, variant and arm reads as owned and switched on (`GameProgressSaver.CheckGear` 0 and `weapon.<name>` prefs of 0 read as 1). In memory only. Send it before the level loads: `GunSetter` builds the arsenal at scene start |

Added in mod 0.8.0. **Every one defaults to the 0.7.2 behaviour**, so a client that sends none of them is unaffected. Note that config is **per game process, not per connection** (as `frameskip` always has been): a later client inherits whatever the previous one set, so a client that cares must send its own value rather than rely on the default.

| key | default | meaning |
|---|---|---|
| `macros` | true | master switch. `false` makes every requested macro report `disabled`. Macros only ever run when an action asks for one, so `true` changes nothing by itself |
| `allow_reserved_macros` | false | macro values 3–5 (`core_nuke`, `rocket_down`, `coin_rocket`) are reserved headroom and are refused. Setting this true only changes the refusal `reason` to `not_implemented`; none of them is built |
| `macro_ssj_wall` | false | whether macro value 2 (`ssj_wall`) may run. **Reserved and refused by default**, beside values 3–5. It is built and it is enforced, but it fired 0 of 25 times on Level 0-1 and its lead mechanism writes a future `slideTimestamp` that four game call sites read — see the wall macro below. Leave it off until an airborne slide is demonstrated on a level that has one |
| `macro_wall_lead_unsafe` | false | lets a private test drive `macro_wall_lead_s` past the safety bound below. **Must stay false anywhere a policy is learning** |
| `ssj_gap_s` | 0.012 | real seconds queued between a macro's slide release and its jump press. `TrySSJ` buckets this as `(int)(gap / 0.008)` and accepts 1–3, so 0.012 is the middle of **bucket 1**, which carries the strongest multiplier at both call sites. Sweepable by a test |
| `macro_wall_lead_s` | -1 | fixed forward-dating of the wall macro's slide release, in seconds. Negative selects the adaptive path below. Both are subject to the safety bound below unless `macro_wall_lead_unsafe` is on |
| `macro_wall_lead_frames` | 2.0 | adaptive lead = `max(0, frames × measured_frame_gap − ssj_gap_s)`. See the wall macro's grace problem below |
| `variant_switching` | false | whether the `variant` action may switch the held weapon's variation. **Refused by default**, which is what makes widening the action head behaviour-preserving |
| `ssj_indicator` | false | forces the game's own `ssjIndicator` preference on while in control, so `TrySSJ` prints its bucket bar and `+{n}u/s` as a subtitle. A human-visible readout for a test instance; the machine-readable answer is `macro.ssj`. The stored preference is never written |
| `obs_move_tech` | false | adds the `move_tech` block (block A) |
| `obs_weapon_tech` | false | adds the `weapon_tech` block (block B) |
| `obs_projectiles` | false | adds the `projectiles` block (block C) |
| `max_projectiles` | 3 | how many of the player's own projectiles `projectiles` keeps, nearest first |
| `obs_input_clock` | false | adds the diagnostic `input` block (timestamp cursor vs the live clock) to step replies |

### action

```json
{"move": [x, y], "look": [yaw_deg, pitch_deg], "buttons": ["fire1", "jump"], "slot": 0}
```

- **`move`:** x = strafe right, y = forward. Each value is in -1..1; anything beyond ±0.33 presses the bound key.
- **`look`:** total degrees of turn over the step, spread evenly across its frames. Positive pitch looks up.
- **`buttons`:** two kinds.
  - Held for the whole step: `fire1`, `fire2`, `slide`, `hook`.
  - Pressed on the first frame only: `jump`, `dash`, `punch`, `change_fist`.
- **`slot`:** 0 keeps the current weapon. 1–6 presses that weapon slot key on the first frame (pressing the current slot again switches variation).
- **`macro`** (mod 0.8.0, optional, default 0): `0` none, `1` `ssj`, `2` `ssj_wall`, `3` `core_nuke`, `4` `rocket_down`, `5` `coin_rocket`. A string name is accepted too. **Only `ssj` runs by default**: values 3–5 are reserved, and `ssj_wall` is reserved too unless `macro_ssj_wall` is on. See **macros** below.
- **`variant`** (mod 0.8.0, optional, default 0): `0` keeps the held variation, `1`/`2`/`3` select variation 0/1/2 of the **current** slot. Refused unless `variant_switching` is on.

Keys are resolved from the player's own bindings, so rebinding in game options is fine.

## Macros (mod 0.8.0)

A macro is a short input script the **mod** plays across the frames of one step, so the client can express a timing a ~15 Hz decision loop cannot reach. It exists for exactly one thing: `NewMovement.TrySSJ` accepts a slide-release-to-jump-press gap of **8–32 ms**, quantised into three 8 ms buckets, and that gap is measured on the Input System's event clock — **real wall-clock time, which `Time.captureDeltaTime` does not touch**, so no frame-rate setting can reach it. Building one `KeyboardState` per frame also put the release and the press in the *same* event, where `TrySSJ`'s `num > 0.0` test rejects them. Both are structural.

**A macro never aims and never moves the camera.** The client keeps `look`, `move` and every other button for the whole step. Macro timing, never macro aim.

**A macro step consumes exactly `frameskip` game frames — the same as any other step.** Verified in game 2026-09-20: plain steps and macro steps both advanced `frame` by 2 and `Time.time` by 0.06667 s at `frameskip` 2 / `fixed_fps` 30. **Nothing about time accounting changes and the env needs no correction.** `macro.frames` and `macro.step_frames` are reported separately anyway, so a future longer macro cannot introduce the hole silently.

- **`ssj`** — one frame, two events. Event A releases the slide (and every tap), event B presses jump `ssj_gap_s` later. Both land in one `Update`, so `HandleInputs` runs with `sliding` still true — `SlideCancelled` only records a timestamp, and `HandleSlideState` comes later in `Update`. `Jump()` therefore takes its `if (sliding)` branch, calls `StopSlide()` (which refreshes `velocityAfterSlide`) and only then calls `TrySSJ`. Preconditions: a player, `sliding`, `!jumpCooldown`, and a jump path that actually runs.
  - **That last one is `HandleInputs`' own gate, not `onGround || canJump`.** The game computes `flag = !falling` and `flag2 = !gc.onGround && (gc.canJump || wcGroup.CheckForEnemyCols())`, and calls `Jump()` iff `flag2 || flag`. Anything looser lets the macro report `ran` on a step where no jump executes at all. When the `flag2` branch is the one that will run, the macro still succeeds — `Jump()` is reached either way — but it reports `"note":"enemy_step"`, because that branch also sets `enemyStepping` and calls `EnemyStepResets()`, zeroing the wall-jump and rocket-jump budgets as a side effect of the request.
- **`ssj_wall`** — **reserved and refused by default** (`macro_ssj_wall`); **two frames, and it must be two.** `TrySSJ` does not add to velocity, it **overwrites**: `rb.velocity = velocityAfterSlide + direction × bonus`. `velocityAfterSlide` is written in exactly one place, `StopSlide`, and `WallJump` never calls it — it reaches `TrySSJ` through the `sliding ||` disjunct. A one-frame wall SSJ would therefore land on the **previous** slide's vector, in that old direction: a speed loss disguised as a technique. Frame 1 queues the slide release alone so that frame's `HandleSlideState` runs `StopSlide()`; frame 2 queues the jump at the release timestamp + `ssj_gap_s`. Needs `frameskip >= 2`.
  - **`ssj_wall` requires an AIRBORNE slide**, which is much rarer than it sounds. `WallJump` only reaches `TrySSJ` through `sliding || currentTime - slideTimestamp < 0.032`, and `SlideCancelled` only records `slideTimestamp` `if (sliding)`. Holding `slide` while airborne does **not** start a slide — it calls `TryStartSlam`. The precondition is enforced, so the macro refuses with `not_sliding` rather than firing a plain wall jump and reporting a success it did not have.
  - **The wall macro's grace is real time and load-dependent.** The release is queued at the end of frame N and the jump is not read until frame N+2's `Update`, so the interval that must fit inside 32 ms is about **two real frames** — a few ms on an idle machine, tens of ms on a loaded 12-game fleet. `macro_wall_lead_frames` dates the release forward to compensate. This is the least certain part of the design, and the reason the macro is off by default.
  - **The lead perturbs the game outside the macro, and is bounded for it.** Frame 1 dates the slide release into the future, and `SlideCancelled` writes that future value straight into the public `NewMovement.slideTimestamp`. Four call sites read that field, and while the stamp is ahead of the clock all four are perturbed on steps the obs reports as `macro: none` — `WallJump`'s `currentTime - slideTimestamp < ssjMaxFrames × 0.008` (trivially true while the difference is negative, so every plain wall jump takes the SSJ / momentum-reflect branch), `HandleInputs`' enemy-step `windState` test at 0.1 s, `Jump`'s dash-jump branch (`jumpTimestamp - slideTimestamp > 0.008 × ssjMaxFrames`, which a negative difference fails), and `TrySSJ` itself, where an ordinary jump one or two decisions later can land in bucket 1–3 and fire an **unrequested** SSJ that overwrites `rb.velocity` with a one-step-old slide vector floored at 24.
  - So the lead is capped at `(step_frames + 1) × frame_gap − ssjMaxFrames × 0.008` — the largest lead that still leaves the next decision's earliest jump outside the SSJ window — and a macro whose needed lead exceeds it is **refused** with `lead_unsafe` rather than queued. The bound is conservative: it ignores Python's think time, which only lengthens the real interval. The needed lead (`≈ 2 × gap − ssj_gap_s`) and the cap are compatible only inside a window one frame gap wide, and the default leaves it whenever `frame_gap < 0.020` — at a 15 ms gap the needed lead is 18 ms against a 13 ms cap. Measured on this box: 4.4–5.6 ms idle, ~30 ms under a 12-game fleet, so both sides of that boundary occur.

**Refusal is a first-class signal, and is never charged.** A refused macro simply runs the plain action. `result` is `ran`, `refused` (preconditions failed when the step arrived), `degraded` (the script started and the world changed under it) or `disabled`. `reason` is one of a fixed vocabulary — `no_player`, `no_ground_check`, `jump_cooldown`, `not_sliding`, `no_jump_path`, `frameskip_lt_2`, `grounded`, `fake_fall`, `not_falling`, `coyote`, `wall_jumps_spent`, `no_wall_check_group`, `enemy_step`, `no_wall`, `slide_grace_expired`, `lead_unsafe`, `macros_disabled`, `reserved` — so a macro refused 99 % of the time shows up as a design bug, and the reason says which one. `note` is separate: a side effect on a macro that **did** run (today only `enemy_step`).

**The monotonic timestamp cursor.** Every event is queued with an explicit timestamp on `InputState.currentTime`, forced strictly increasing across keyboard and mouse. `InputManager.OnUpdate` silently discards a state event older than the device's last update time and `Keyboard` has no state callbacks, so after a macro dates an event forward the *next* frame's ordinary event can carry a lower timestamp and be dropped outright — a whole frame of the agent's input lost, fleet-wide, reading as a policy regression. Measured 2026-09-20: right after a macro the cursor was ahead of the live clock in **11 of 12** samples, i.e. the drop would have happened in almost every case, and with the cursor a dash issued on the following step registered **12 of 12**. When nothing asks for a timestamp the event still goes out with `time = -1`, the pre-0.8 call.

## obs

```json
{
  "type": "obs", "step": 12, "frame": 3456, "time": 57.6, "scene": "Endless", "ready": true,
  "player": {"pos": [x,y,z], "vel": [..], "local_vel": [..], "forward": [..], "yaw": 90.0, "pitch": -5.0,
             "hp": 100, "anti_hp": 0.0, "stamina": 300.0, "grounded": true, "sliding": false,
             "slow_mode": false, "heavy_fall": false, "crouching": false, "dead": false,
             "activated": true, "level_over": false, "weapon_slot": 0, "weapon_variation": 1,
             "slot_counts": [3, 3, 3, 3, 3, 0]},
  "enemies": [{"id": 12345, "type": 3, "type_name": "Filth", "health": 0.5, "pos": [..],
               "rel": [x,y,z], "dist": 12.3, "visible": true}],
  "rays": [..], "ground_rays": [..], "ground_ray_center": 1.62,
  "stats": {"kills": 3, "style": 450, "seconds": 57.6, "restarts": 0, "level_complete": false},
  "cybergrind": {"wave": 2, "enemies_left": 4, "start_trigger": {"center": [..], "size": [..]}}
}
```

- **`enemies[].rel`:** position in camera space (x right, y up, z forward).
- **`stamina`:** 100 per dash charge (300 = 3 dashes).
- **`rays`:** start straight ahead and go around the player.
- **`ground_rays`:** distance from the player's height down to the ground at points on a ring. A large value means a pit. Exactly `ground_ray_length` when a ray hits nothing, so "off the map" is unambiguous; negative when the ground is above the player's feet.
- **`ground_ray_center`:** the same measurement straight down from the player, outside the ring and outside the array (adding a 9th ring value would change the packed observation size). This is the one to use for "what is the player standing on": the ring's minimum can be a ledge 4 m away rather than the floor underfoot (measured spread among hit rays: p50 0.5 m, p90 10.8 m).
- **`slow_mode`, `heavy_fall`, `crouching`:** `NewMovement.slowMode`, `NewMovement.gc.heavyFall` and the private `NewMovement.crouching`. `slow_mode` blocks dash and stamina regen and cuts walk speed to 1.25x `walkSpeed`; while **grounded** it is an ordinary crouch under a low ceiling. Airborne with `slow_mode` (and no movement) is the wedge the `unwedge` config setting breaks: a slide that ends in the air leaves the player in a state the ground check can never end by itself. `crouching` is `false` with one logged warning if a game update renames the private field.
- **`cybergrind`:** only present in the Cyber Grind scene. `start_trigger` is the volume that starts wave 1 when entered, and is only present before waves start.
- **`player`:** `null` when no player exists (e.g. the main menu).
- **`player.slot_counts`:** weapons in each of the six slots, slot 1 first. Empty until `GunControl` has started; 0-1 has none until the revolver pickup.

## Technique blocks (mod 0.8.0)

Four optional top-level blocks. `move_tech`, `weapon_tech` and `projectiles` appear only when their config flag is on; `macro` and `variant` appear only on a step whose action asked for one; `input` only under `obs_input_clock`. **All values are additive** — indices 0–478 of the packed observation keep their meaning, and a consumer that `.get()`s with a 0.0 default sees the same vector against an old DLL.

- **`macro`** — `{"requested":"ssj","result":"ran","reason":null,"note":null,"frames":1,"step_frames":2,"gap_s":0.012,"lead_s":0.0,"anchor":123.456,"frame_gap":0.0045,"ssj_bucket":1,"ssj_landed":true,"ssj":{...}}`. `ssj` is `{"frame","dt","dt_ms","bucket","landed","wall","speed_before","speed_after","gain","h_speed_before","h_speed_after","h_gain"}`. `bucket` is `(int)(dt / 0.008)`, `-1` when no slide release preceded the jump; only 1–3 are accepted by the game. **Gate a reward on the bucket, not on a speed delta** — `TrySSJ` overwrites velocity with `velocityAfterSlide + bonus` and `velocityAfterSlide` floors at 24, so a speed bar pays most when the player is slow and stops paying when it is fastest.
  - **`ssj` / `ssj_bucket` / `ssj_landed` are populated only when `result` is `"ran"`**, and are `null` / `-1` / `false` otherwise. They have to be: `TrySSJ` runs at the end of **every** `Jump()` and inside `WallJump`'s grace branch, so the instrument's last reading may belong to the plain action rather than to the macro. Without the gate a **refused** macro could come back as `result:"refused"` beside `ssj_bucket:2, ssj_landed:true`, and a reward keyed on the bucket alone would pay for the refusal — teaching the policy to request macros whose preconditions it cannot meet. A reward may key on `ssj_bucket` directly, but anything reading `move_tech.ssj_last` instead must require `result == "ran"` itself.
  - **If the `TrySSJ` Harmony patch fails to apply, every macro reports `ssj_bucket: -1` forever and the bridge keeps working.** That is deliberate isolation, but it means a reward gated on the bucket would silently pay **zero across the whole fleet** after a game update renamed the private method. A client that enables such a reward must assert `ssj_instrument` and `macro.ssj` are in `hello.features` at connect and fail the worker loudly, rather than train against a dead channel. `hello.diag.ssj_instrument` carries the same flag.
- **`variant`** — `{"requested":2,"result":"ran","variation":1,"variations":3}`, or a `refused`/`disabled` with a `reason` of `reserved`, `no_gun_control`, `no_such_variation` or `already_held`. Implemented by calling `GunControl.SwitchWeapon(slot, variation)` directly rather than pressing a key: weapon selection has **no** Harmony-inlining hazard (that hazard is about `InputActionState`'s getters), and `SelectVariant1/2/3` are not reliably bound. This is the one place the mod bypasses the virtual-device pipeline, and it is deliberate.
- **`move_tech`** (block A) — `heavy_fall`, `slam_force`, `slam_storage`, `slam_cooldown`, `bounce_window` (any of `superJumpChance` / `bounceChance` / `extraJumpChance` positive) with the three chances themselves, `coyote` (`gc.sinceLastGrounded`), `can_jump`, `wall_jumps`, **`wall_available`**, `wall_touching`, `enemy_cols`, `boost`, `boost_left` (dash i-frames left; `> 0` also means layer 15), `dash_storage`, `invincible`, `pre_slide_speed`, `jump_cooldown`, `jumping`, `falling`, `fake_fall`, **`slide_grace`**, `slide_since`, `slide_timestamp`, `jump_timestamp`, `velocity_after_slide`, `riding_rocket`, `ssj` (cumulative `{attempts, landed, histogram}`, histogram index 0–4 = bucket and index 5 = `dt <= 0`), `ssj_last` (the last attempt whether or not a macro asked for it, so a **plain** jump is measured by the same instrument).
  - **`wall_available` is the one field the 16 horizontal rays cannot express**, because a `WallCheck` is a component, not a distance. It is the wall macro's precondition.
  - **Pack `slide_grace`. Never pack `slide_since`, `slide_timestamp` or `jump_timestamp`.** `slide_grace` is the fraction of the SSJ window still open, 1.0 at the instant of release down to 0.0 once expired — bounded, dimensionless, normalised by the game's own window. The other three are **real wall-clock seconds**, because the game's two timestamps are Input System event times and its grace is measured against the same clock, while every other quantity in this observation is **game** time, which `Time.captureDeltaTime` pins at a fixed step per frame. The same game state therefore reads `slide_since ≈ 0.08` on a loaded 12-game fleet and `≈ 0.01` on a single eval game (measured: ~30 ms vs 4.4–5.6 ms real per frame) — an order-of-magnitude train/eval shift on the macro's own precondition. The two absolute stamps are worse: unbounded process-uptime doubles that are non-stationary as policy inputs and lose millisecond resolution in float32 after a few hours. They are diagnostics.
  - **`ssj` and `ssj_last` are emitted only when they changed**, and are `null` on the steps in between; the counters are cumulative, so nothing is lost. Building both unconditionally cost ~25 extra JTokens on every step, on a box that is commit-bound beside games leaking ~790 MB/h.
- **`weapon_tech`** (block B) — `slot`, `variation`, `variations_in_slot`, `gun_ready`, and from `WeaponCharges`: `pierce_charge` (of 100), `coin_charge` (of 400), `sharp_charge` (of 300), `core_charge` (the Core Eject grenade, of 1), `saw_charge`, `rai_charge` (of 5), `rocket_charge`, `rocket_frozen`, `rocket_freeze_time`; plus `fist_cooldown`, `holding_item`, `hook_equipped`, `hook_state`, `hook_state_name`, `being_pulled`.
  - **`gun_ready` is `Revolver.gunReady`**, and it is the field that shows a policy suppressing its own fire: pressing the already-equipped slot re-draws the weapon (`WeaponRedrawBehaviour` defaults to "cycle variation"), `Revolver.OnEnable` clears `gunReady`, only the `ReadyGun` animation event sets it back, and `Revolver.Update` gates firing on it. Reported `true` for anything that is not a Revolver.
- **`projectiles`** (block C) — the player's **own** live objects, nearest first, at most `max_projectiles`: `{"id","kind","pos","rel","vel","dist","age","riding","frozen","rideable"}` with `kind` one of `coin`, `rocket`, `grenade`, `cannonball`. Enemy-owned objects are excluded — an enemy's coin or rocket is a hazard, not a tool, and already shows up through `enemies`. `age` is measured by the mod from the first step the object was seen on.
  - **`projectiles[].rel` and `enemies[].rel` are in DIFFERENT frames.** This block uses `NewMovement.transform.InverseTransformPoint`, and the player transform's rotation is set from `rotationY` alone, so it is a **yaw** frame. `enemies[].rel` uses `cam.InverseTransformPoint`, which **includes pitch**. Whoever packs block C must not copy the enemy packer: the coin rocket jump is a look-up-then-detonate technique, so every direction would be wrong by exactly the angle the agent is looking up at.
- **`input`** — `{"cursor": <last queued event timestamp>, "now": <InputState.currentTime>, "frame_gap": <measured real seconds per game frame>}`. `cursor > now` is exactly the case in which an un-lifted event would have been dropped.

## campaign

Present only in the 35 main levels (`StatsManager.levelNumber` 1 to 35, in a scene whose name starts with `Level`) when a player exists, and only when the block itself built successfully: an exception building it is caught, logged once (not every step, to stay readable at 150-250 steps/s), and the block is simply omitted from that step's obs rather than failing the whole reply.

```json
"campaign": {
  "mission": 1, "difficulty": 3, "seconds": 12.3, "timer_running": true, "level_started": true,
  "level_over": false, "restarts": 0, "input_locked": false,
  "exit": {"pos": [x, y, z], "active": true, "ground_pos": [x, y, z]},
  "checkpoints": [{"id": "12,3,-40", "pos": [x, y, z], "activated": false, "current": false}],
  "path": {"status": "complete", "length": 84.2, "next_corner": [x, y, z]},
  "locked_doors": [{"pos": [x, y, z], "dist": 9.5}],
  "arena_enemies_alive": 0,
  "cleared_arenas": ["30,1,5"],
  "unlocked_doors": ["22,0,17"],
  "gates_ordered": true,
  "gates_truncated": false,
  "gates": [{"key": "202,56,432", "pos": [202.0, 56.0, 431.5], "hops": 0,
             "open": false, "locked": false, "active": true, "controller_active": true,
             "needs_item": null},
            {"key": "81,-6,240", "pos": [81.0, -6.0, 239.5], "hops": 1,
             "open": false, "locked": false, "active": true, "controller_active": false,
             "needs_item": "SkullBlue", "altar_only": true}],
  "altars": [{"key": "81,-4,251", "pos": [81.0, -3.76, 251.0], "aim_pos": [81.0, -4.76, 251.0],
              "item": "SkullBlue", "filled": false, "active": false, "inactive_ancestors": 1,
              "doors": [{"key": "81,-6,240", "pos": [81.0, -6.0, 239.5]}], "reverse_doors": []}],
  "items": [{"key": "-15,27,427", "pos": [-15.0, 26.64, 427.0], "item": "SkullBlue",
             "held": false, "placed": true, "placed_in": "-15,27,427",
             "active": false, "active_self": true, "inactive_ancestors": 1}],
  "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}
}
```

- **Position keys** (`checkpoints[].id`, `cleared_arenas`, `unlocked_doors`): `"x,y,z"`, each coordinate rounded to whole metres. Objects that round to the same metre share a key (for example two waves' `ActivateNextWave` components on one GameObject, or wave containers placed at the same point), so only the first of them pays.
- **`mission`:** `StatsManager.levelNumber` (1 = 0-1 through 35 = 9-2).
- **`difficulty`:** the difficulty the game reads right now, so it shows the `difficulty` config override while in control.
- **`seconds`, `timer_running`, `level_started`, `restarts`:** from `StatsManager`. The timer keeps running through checkpoint respawns and cutscenes, and stops when the player enters the exit.
- **`level_over`:** `NewMovement.levelOver`, set on entering the real exit.
- **`input_locked`:** `GameStateManager.PlayerInputLocked` (any registered game state that locks player input: the pause, cheat and spawn menus, the console, and scene objects with `AutoRegisterState`) or the player not activated (the level-start drop, after entering the exit, and while dead).
- **`exit`:** the real `FinalPit`, or `null`. Decoys (`fakeEnd`, `secondPit`, `rankless`), room templates and pits whose `targetLevelName` is empty, ends with `-S` (a secret level) or starts with `Level P-` (a Prime Sanctum) are skipped; then a pit leading **onward** is preferred — the next mission (`Level a-b` to `Level a-(b+1)` or `Level (a+1)-1`, parsed, no level table) **or an `Intermission*` target**, since an act finale's real exit drops into an intermission — then an active pit over one whose room has not loaded. The choice is frozen for the level load, because "active" changes as the level plays and every `gates[].hops` depends on it; it is only re-run if the pit is destroyed. `active` is false while its room has not loaded. (Measured: 0-2 carries pits to `Level 0-3` and to `Level 0-S`, 1-1 carries one to `Level 1-2` and two to `Level 1-S`, and all of them are inactive at load, so without this the choice was arbitrary.)
  - **`ground_pos`** (mod 0.7.2): the nearest **standable** point to the pit — the NavMesh sample `path` has always snapped its exit end to, kept and reported so a consumer can aim at it. `pos` is the `FinalPit`'s own transform, which sits **inside the drop it triggers**, far below anything walkable: measured 61–75 m below the floor on `Level 0-2` and 70 m on `Level 0-3`, and both of 0-2's real completions triggered at y −25 to −27.5 while `pos` read y −86.1. So a target or a distance built from `pos` points down a killing fall and can never reach zero; `ground_pos` is what to aim at. Refreshed on the same 4-obs cadence as `path`, by the same widening 20 / 55 / 95 m search, and independent of the player-end snap — so it is reported even where `path` reads `none` (0-1's own spawn). `null` when NavMesh holds nothing within 95 m of the pit, and absent on a mod older than 0.7.2; both cases mean "fall back to `pos`".
  - The `Level P-` and `Intermission` rules exist for **6-2**, which ships two Prime Sanctum pits alongside its real `Intermission2` pit: all three used to rank equally and the exit was decided by `FindObjectsOfType` order. Campaign-wide only 3-1 and 6-2 ship a `Level P-` pit and only 3-2 and 6-2 ship an `Intermission*` one, and no main level's legitimate successor starts with `Level P-`.
  - A **tie at the best rank is logged as a warning**, once per level load. Two levels stay legitimately tied and are benign: 3-2's two `Intermission1` pits sit at the identical position, and 2-4's two `Level 3-1` pits are 1.4 m apart in the same room. 8-4's `EarlyAccessEnd` pit parses as neither a level nor an intermission and is picked by **uniqueness alone**, which is why the warning exists.
- **`checkpoints`:** every checkpoint except room templates. `current` marks `StatsManager.currentCheckPoint`.
- **`path`:** NavMesh path from the player to the exit, each end snapped onto the mesh, recalculated every 4 obs. The player end snaps within 25 m (ULTRAKILL is played in the air, and at 6 m an airborne player missed the mesh and dropped the whole path to `none`). The exit end snaps within 20 m of the `FinalPit`'s own position first, then within 55 m, then 95 m: a `FinalPit`'s transform sits inside the drop it triggers, not on walkable ground (measured in game, the nearest NavMesh point was 47.5 m away straight down in one level and 84.6 m away up and to the side in another), so the widening search finds whichever direction actually holds mesh instead of assuming "above." `length`/`next_corner` are always measured to that snapped point, i.e. distance still to walk to the nearest standable point back on the mesh, never a straight line into the pit. `length` runs from the player through every corner; `next_corner` is the first corner more than 1.5 m away horizontally, else the last. `partial` is common: the mesh does not link jumps or gaps, doors carry obstacles, and a `FinalPit`'s snapped exit point is itself typically off the main walkable graph, connecting back only partially. `{"status": "none"}` when there is no exit or no path.
- **`locked_doors`:** the nearest 4 active doors that are locked.
- **`arena_enemies_alive`:** live enemies under an `ActivateNextWave` whose wave has not been cleared.
- **`cleared_arenas`, `unlocked_doors`:** keys of arenas whose last wave was cleared (`ActivateNextWave.EndWaves`) and of doors that went from locked to unlocked (`Door.Unlock`), recorded while in control and emptied on every scene load. A checkpoint respawn re-creates rooms at the same positions, so an arena cleared again after a death gives the same key; a respawn also unlocks the checkpoint's doors, which adds keys. Pay each key once per level load.
- **`gates`:** the level's route, derived from the door graph rather than from the NavMesh. A **gate** is a `Door` whose `activatedRooms` holds at least two distinct GameObjects; those rooms (and only those) are the graph's nodes, two rooms are adjacent when one gate lists both, and a breadth-first search from the room holding the chosen exit gives every room a hop count. A second pass then appends the doors an altar **opens** that the two-room test rejected (`altar_only`, below). Sorted by `hops` ascending with `null` last, then by `key`. At most 64 entries.
  - **`hops`:** rooms still to traverse **after** passing this door, the lowest over the door's own rooms. `0` is the door into the exit's room. `null` means the door is not in the exit's connected component. Several gates can share a hop count: that is a fork in the level, not a duplicate (0-1 has two gates at `hops` 2).
  - **`key`, `pos`:** the door's **closed** world position, frozen for the level load. A Normal door moves its own transform by `openPos` while it opens (measured `(0, 5.75, 0)` on every gate door of 0-1 to 0-5), so reading its live position would change its key halfway through. `key` is the same `"x,y,z"` whole-metre convention as the other position keys, with `#2`, `#3`, ... appended when two doors round to the same metre (logged once per level load). **Treat `key` as opaque**: `Mathf.RoundToInt` rounds halves to even and 0-1's doors sit on exact `.5` z values, so recomputing it from `pos` gives different strings.
    - **One key table covers every door in the level**, not only the gate candidates, and `altars[].doors[].key` reads out of the same table, so a gate key and an altar's door key **string-match by construction**. Suffixing is deterministic: doors are keyed in ascending closed position, falling back to the instance id for two doors authored at the same point. (Suffixes used to follow `FindObjectsOfType` order, which is not stable across rescans while the hop counts are keyed on the suffixed string, so a respawn could hand a gate the other gate's hops. Measured over all 33 shipped levels, no gate key collides with a non-gate door key, so widening the table changed no existing gate key; 1-2's two doors at `0,20,380` are the only pair the new ordering reorders.)
  - **`needs_item`:** the `item` type an **unfilled** altar wants before this door will open, or `null`. It comes from `altars[].doors` only — never `reverse_doors`, which an altar *closes* when it is filled. Dead twins (see `altars`) are ignored, because they can never fill and the lock would never clear. With several such altars the lowest type name wins, so the answer is deterministic. A skull-locked door is otherwise indistinguishable from a walk-up one: `ItemPlaceZone.ColorDoors` calls `Door.AltarControlled()`, which deactivates the door's `DoorController`s, so it reports `open: false, locked: false, controller_active: false` — the same signature as a door an arena holds shut.
  - **`altar_only`:** present and `true` only on a door that is a gate **because** an altar opens it; absent otherwise. Such a door adds no node and no edge to the room graph, so no other gate's `hops` can change, and it reads its own `hops` off the rooms it does list — which is `null` everywhere those rooms are outside the graph. Measured: of 28 distinct altar-driven forward doors campaign-wide, 6 are already ordinary gates and this pass gives a usable `hops` to 3 more (1-1's `81,-6,240` at hops 1, 5-3, 8-1); the rest stay `null`.
  - **`open`:** `Door.open`, i.e. "currently open or opening", **never** "has been passed". A `DoorController` closes the door as soon as the player and every enemy leave it, enemies open doors too, and opening one door force-closes the others.
  - **`locked`, `active`:** `Door.locked` and `activeInHierarchy`. A destroyed door stays in the array with all four flags false rather than disappearing.
  - **`controller_active`:** at least one of the door's `DoorController`s is `activeInHierarchy`. This is what separates "walk up to it" from "a fight gates it": 0-1's gun-room gate reports `open: false, locked: false` at load and still cannot be opened, because its controller is switched off until `ActivateArena.Activate` runs.
  - **`gates_ordered`:** at least one gate has a hop count. `false` when no room on the exit's ancestor chain is a graph node (measured: 0-5), in which case every `hops` is `null` and the straight-line `exit` vector is the only route signal. A hop count a gate has been given survives a rescan that transiently loses the exit's room (a checkpoint respawn re-creates it), so the route never renumbers mid-load.
  - **`gates_truncated`:** the level had more gate candidates than fit; the 64 kept are the ones nearest the player. Measured candidate counts: 11 (0-1), 17 (0-2), 12 (0-3), 7 (0-4), 3 (0-5), 13 (1-1).
- **`altars`:** one entry per `ItemPlaceZone` whose `acceptedItemType` is not `None`, room templates excluded, sorted by `key` ascending, at most 64 (nearest the player kept). Empty on the 12 shipped levels with no altar, 0-1 among them. These are the skull pedestals and their destinations — both ends of a carry are `ItemPlaceZone`s.
  - **`key`:** the zone's own rounded position, `#2`/`#3` on collision in ascending position order (several levels ship coincident duplicate zones ~0.2 m apart; 1-1 has three such pairs). Its own namespace: an altar key never has to differ from a gate or item key. **Opaque.**
  - **`pos`:** the zone's own world transform position. **Do not aim a punch at it** — see `aim_pos`.
  - **`aim_pos`:** where a placement punch must be pointed: the centre of the zone's **own collider**. `Punch.AltHit` calls `GetComponents<ItemPlaceZone>()` on the very transform the raycast returned, and `ItemPlaceZone.Start` reads its own `GetComponent<Collider>()`, so that collider is the thing to hit — and it is not centred on the transform. Every one of the campaign's 104 zones is the same prefab: a trigger `BoxCollider` of local size `(2.2, 3.5, 2.2)` centred at `(0, -1.25, 0)` on a transform scaled `(0.9, 0.8, 0.8)`, so the box sits **1 m below `pos`** (103 of 104; the one exception is 0.625 m) and spans `pos.y - 2.4 … pos.y + 0.4`. `pos` is therefore only 0.4 m under the lid with no margin for aim error, while the centre has 1.4 m of it. Measured in game on 1-1: aiming at `pos` does not place and aiming ~1.25 m below it does. Python falls back to `pos` minus 1 m when this field is absent (`campaign.altar_aim_point`).
  - **`item`:** the `ItemType` name it accepts: `SkullBlue`, `SkullRed`, `SkullGreen`, `Readable`, `Torch`, `Soap`, `CustomKey1`..`CustomKey3`.
  - **`filled`:** computed exactly as `ItemPlaceZone.CheckItem` does — `GetComponentInChildren<ItemIdentifier>()` **without** `includeInactive`, then the type test. With `includeInactive` every `Altar (… Skull) Variant` would read as filled at load, because it carries a disabled decoration skull.
  - **`doors`, `reverse_doors`:** `[{"key", "pos"}]` for the doors the altar **opens** and the ones it **closes** when filled, keys from the shared door-key table above.
  - **`active`, `inactive_ancestors`:** `activeInHierarchy`, and the number of inactive GameObjects on the zone's own chain including itself. **The dead-twin test is RELATIVE, not a constant:** a zone is dead when another zone with the same `item`, the same `doors` set and the same rounded position reports **strictly fewer** `inactive_ancestors`. Such a zone never activates, so its `CheckItem` never runs and it reads `filled: false` forever. Measured, 20 of the campaign's 104 functional zones are dead twins of a live one, on 11 levels, and no door anywhere is driven only by dead zones.
    - It used to be the absolute `inactive_ancestors > 1`, and that is wrong because **the count is not a property of the zone**: it is the zone's own chain plus however much of the room above it happens to be switched off, so it **shifts when the room lights**. Measured in game on 1-1, the live altar `81,-4,251` and its twin read 1 and 2 on a fresh load and **0 and 1** after the checkpoint respawn switched that room on, so the twin stopped being filtered exactly when the player arrived and the gate kept `needs_item` set however often the puzzle was solved. A shared room contributes equally to both halves of a pair, so the relative test cannot be shifted; and the minimum of each group always survives, so a group is never filtered away entirely and a real lock can never be lost.
    - Position is part of the identity because twins are co-located (~0.2 m apart, which is why the keys need `#N` at all). Without it the rule would also drop five zones that are not twins but two separate altars of one item type driving no door (4-2 ×2, 4-3, 5-3, 7-1).
    - Re-validated offline against all 21 altar levels: the relative rule drops the same 20 of 104 zones, loses no lock on any level, and takes the doors left stuck after a solved puzzle from 11 to 0 once a room lights.
- **`items`:** one entry per `ItemIdentifier` with a real `itemType` and `infiniteSource == false`, room templates excluded, sorted by `key` ascending, at most 64 (nearest the player kept). `infiniteSource` items are skipped because `Punch.AltHit` mints a fresh `ItemIdentifier` per punch for one, which would be an unbounded key stream (measured: no shipped level has one).
  - **`key`:** the item's rounded position at the scan that first saw it, `#N` on collision, then memoized by instance id for the rest of the level load so a carried item keeps its key while it moves. **Opaque, and guaranteed unique only within a step** — a checkpoint respawn re-instantiates rooms, so do not key anything durable on it.
  - **`pos`, `held`:** live world position every step, and `ItemIdentifier.pickedUp`.
  - **`placed`, `placed_in`:** whether the item sits in an `ItemPlaceZone` (`ipz`, falling back to a parent search, because `ItemPlaceZone.Start` has not run while its room is off), and that zone's `altars[].key`, or `null` when the zone is not in `altars`. **Every `ItemIdentifier` on 1-1 starts `placed: true`**, source pedestals included, so "not held and not placed" finds no source at all.
  - **`active`, `active_self`, `inactive_ancestors`:** as for altars. **`active_self && inactive_ancestors <= 1` is the carryable test**: it separates a real source from the decoration skulls a destination altar carries and from phantom copies under a permanently disabled node. Measured on 1-1, three of five items have `active_self: true` and only two of those pass.
  - While a room is switched off, its altars read `filled: false` and everything in it reads `active: false`. That is conservative and self-correcting for routing — `needs_item` says "yes" before the room loads, which is the right answer — but never read `filled: false` as evidence that an altar is *reachable*.
- **`ranks`:** the 4 thresholds per category from `StatsManager`: `time` in seconds (lower is better), `kills` and `style` (higher is better).
- **Scene objects** are cached and searched again every 30 obs, on a new scene, after a checkpoint respawn, and when a checkpoint activates, becomes current or takes over rooms. The same rescan rebuilds the door-key table, the gate list, its room graph and each gate's `key`, `pos` and `hops`, and the `altars` and `items` lists with their keys, types and door wiring; per step only `open`, `locked`, `active` and `controller_active` (gates), `filled`, `active` and `inactive_ancestors` (altars) and `pos`, `held`, `placed`, `placed_in`, `active`, `active_self` and `inactive_ancestors` (items) are read. The whole array is sent every step (11 gates is 1.5 KB of a 3.6 KB obs line, measured on 0-1; 1-1's 7 altars and 5 items add ~1.2 KB), so there is no cache to keep in sync on the Python side.
