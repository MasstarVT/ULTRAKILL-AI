# Bridge protocol (v1)

The mod listens on `127.0.0.1:47800` (configurable in `BepInEx/config/masstarvt.ultrakill.aibridge.cfg`).
Launching the game with `-aibridge-port N` makes it a **training instance** on port N: it opens preferences read-only and never writes preferences or save data. A normally launched game moves to the next free port if 47800 is taken.
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
| `hello` | `protocol` | `{"type":"hello","protocol":1,"mod_version":..,"port":..,"training_instance":..,"scene":..}` |
| `config` | any of the settings below | `{"type":"ok"}` |
| `get_obs` | | an `obs` message (doesn't take control) |
| `reset` | `scene` (e.g. `"Endless"`, `"Level 0-1"`; omit = current), `checkpoint` (bool) | an `obs` with `"event":"reset"` once the player is spawned and has been ready for `reset_settle_frames` frames |
| `step` | `action` | an `obs` after `frameskip` frames |
| `teleport` | `pos` `[x,y,z]` | an `obs` with `"event":"teleport"` (takes control; moves the player and zeroes velocity) |
| `kill` | | an `obs` with `"event":"kill"` (takes control; debug: a lethal `NewMovement.GetHurt(999)` for the in-game death check. With `soft_death` on it is healed like any lethal hit; ignored once the level is over; an error when there is no living player). While in control a dead player stays dead until a `reset`: the game's own restarts (Fire1 or R while dead, the pause menu) are blocked |
| `release` | | `{"type":"ok"}` |

Errors come back as `{"type":"error","message":..}`.

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
| `unlock_all_gear` | false | while in control, every weapon, variant and arm reads as owned and switched on (`GameProgressSaver.CheckGear` 0 and `weapon.<name>` prefs of 0 read as 1). In memory only. Send it before the level loads: `GunSetter` builds the arsenal at scene start |

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

Keys are resolved from the player's own bindings, so rebinding in game options is fine.

## obs

```json
{
  "type": "obs", "step": 12, "frame": 3456, "time": 57.6, "scene": "Endless", "ready": true,
  "player": {"pos": [x,y,z], "vel": [..], "local_vel": [..], "forward": [..], "yaw": 90.0, "pitch": -5.0,
             "hp": 100, "anti_hp": 0.0, "stamina": 300.0, "grounded": true, "sliding": false, "dead": false,
             "activated": true, "level_over": false, "weapon_slot": 0, "weapon_variation": 1,
             "slot_counts": [3, 3, 3, 3, 3, 0]},
  "enemies": [{"id": 12345, "type": 3, "type_name": "Filth", "health": 0.5, "pos": [..],
               "rel": [x,y,z], "dist": 12.3, "visible": true}],
  "rays": [..], "ground_rays": [..],
  "stats": {"kills": 3, "style": 450, "seconds": 57.6, "restarts": 0, "level_complete": false},
  "cybergrind": {"wave": 2, "enemies_left": 4, "start_trigger": {"center": [..], "size": [..]}}
}
```

- **`enemies[].rel`:** position in camera space (x right, y up, z forward).
- **`stamina`:** 100 per dash charge (300 = 3 dashes).
- **`rays`:** start straight ahead and go around the player.
- **`ground_rays`:** distance from the player's height down to the ground at points on a ring. A large value means a pit.
- **`cybergrind`:** only present in the Cyber Grind scene. `start_trigger` is the volume that starts wave 1 when entered, and is only present before waves start.
- **`player`:** `null` when no player exists (e.g. the main menu).
- **`player.slot_counts`:** weapons in each of the six slots, slot 1 first. Empty until `GunControl` has started; 0-1 has none until the revolver pickup.

## campaign

Present only in the 35 main levels (`StatsManager.levelNumber` 1 to 35, in a scene whose name starts with `Level`) when a player exists, and only when the block itself built successfully: an exception building it is caught, logged once (not every step, to stay readable at 150-250 steps/s), and the block is simply omitted from that step's obs rather than failing the whole reply.

```json
"campaign": {
  "mission": 1, "difficulty": 3, "seconds": 12.3, "timer_running": true, "level_started": true,
  "level_over": false, "restarts": 0, "input_locked": false,
  "exit": {"pos": [x, y, z], "active": true},
  "checkpoints": [{"id": "12,3,-40", "pos": [x, y, z], "activated": false, "current": false}],
  "path": {"status": "complete", "length": 84.2, "next_corner": [x, y, z]},
  "locked_doors": [{"pos": [x, y, z], "dist": 9.5}],
  "arena_enemies_alive": 0,
  "cleared_arenas": ["30,1,5"],
  "unlocked_doors": ["22,0,17"],
  "ranks": {"time": [300, 240, 180, 120], "kills": [10, 20, 30, 40], "style": [1000, 2000, 3000, 4000]}
}
```

- **Position keys** (`checkpoints[].id`, `cleared_arenas`, `unlocked_doors`): `"x,y,z"`, each coordinate rounded to whole metres. Objects that round to the same metre share a key (for example two waves' `ActivateNextWave` components on one GameObject, or wave containers placed at the same point), so only the first of them pays.
- **`mission`:** `StatsManager.levelNumber` (1 = 0-1 through 35 = 9-2).
- **`difficulty`:** the difficulty the game reads right now, so it shows the `difficulty` config override while in control.
- **`seconds`, `timer_running`, `level_started`, `restarts`:** from `StatsManager`. The timer keeps running through checkpoint respawns and cutscenes, and stops when the player enters the exit.
- **`level_over`:** `NewMovement.levelOver`, set on entering the real exit.
- **`input_locked`:** `GameStateManager.PlayerInputLocked` (any registered game state that locks player input: the pause, cheat and spawn menus, the console, and scene objects with `AutoRegisterState`) or the player not activated (the level-start drop, after entering the exit, and while dead).
- **`exit`:** the real `FinalPit`, or `null`. Decoys (`fakeEnd`, `secondPit`, `rankless`) and room templates are skipped, and an active pit is preferred; `active` is false while its room has not loaded.
- **`checkpoints`:** every checkpoint except room templates. `current` marks `StatsManager.currentCheckPoint`.
- **`path`:** NavMesh path from the player to the exit, each end snapped onto the mesh, recalculated every 4 obs. The player end snaps within 6 m. The exit end snaps within 20 m of the `FinalPit`'s own position first, then within 55 m, then 95 m: a `FinalPit`'s transform sits inside the drop it triggers, not on walkable ground (measured in game, the nearest NavMesh point was 47.5 m away straight down in one level and 84.6 m away up and to the side in another), so the widening search finds whichever direction actually holds mesh instead of assuming "above." `length`/`next_corner` are always measured to that snapped point, i.e. distance still to walk to the nearest standable point back on the mesh, never a straight line into the pit. `length` runs from the player through every corner; `next_corner` is the first corner more than 1.5 m away horizontally, else the last. `partial` is common: the mesh does not link jumps or gaps, doors carry obstacles, and a `FinalPit`'s snapped exit point is itself typically off the main walkable graph, connecting back only partially. `{"status": "none"}` when there is no exit or no path.
- **`locked_doors`:** the nearest 4 active doors that are locked.
- **`arena_enemies_alive`:** live enemies under an `ActivateNextWave` whose wave has not been cleared.
- **`cleared_arenas`, `unlocked_doors`:** keys of arenas whose last wave was cleared (`ActivateNextWave.EndWaves`) and of doors that went from locked to unlocked (`Door.Unlock`), recorded while in control and emptied on every scene load. A checkpoint respawn re-creates rooms at the same positions, so an arena cleared again after a death gives the same key; a respawn also unlocks the checkpoint's doors, which adds keys. Pay each key once per level load.
- **`ranks`:** the 4 thresholds per category from `StatsManager`: `time` in seconds (lower is better), `kills` and `style` (higher is better).
- **Scene objects** are cached and searched again every 30 obs, on a new scene, after a checkpoint respawn, and when a checkpoint activates, becomes current or takes over rooms.
