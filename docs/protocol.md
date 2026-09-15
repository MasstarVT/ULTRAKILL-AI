# Bridge protocol (v1)

The mod listens on `127.0.0.1:47800` (configurable in `BepInEx/config/masstarvt.ultrakill.aibridge.cfg`).
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
| `hello` | `protocol` | `{"type":"hello","protocol":1,"mod_version":..,"scene":..}` |
| `config` | any of the settings below | `{"type":"ok"}` |
| `get_obs` | | an `obs` message (doesn't take control) |
| `reset` | `scene` (e.g. `"Endless"`, `"Level 0-1"`; omit = current), `checkpoint` (bool) | an `obs` with `"event":"reset"` once the player is spawned and has been ready for `reset_settle_frames` frames |
| `step` | `action` | an `obs` after `frameskip` frames |
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
| `reset_settle_frames` | 30 | frames the player must be ready before a reset completes |
| `command_timeout_s` | 300 | drop the client if no command arrives for this long |

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
             "activated": true, "level_over": false, "weapon_slot": 0, "weapon_variation": 1},
  "enemies": [{"id": 12345, "type": 3, "type_name": "Filth", "health": 0.5, "pos": [..],
               "rel": [x,y,z], "dist": 12.3, "visible": true}],
  "rays": [..], "ground_rays": [..],
  "stats": {"kills": 3, "style": 450, "seconds": 57.6, "restarts": 0, "level_complete": false},
  "cybergrind": {"wave": 2, "enemies_left": 4}
}
```

- **`enemies[].rel`:** position in camera space (x right, y up, z forward).
- **`stamina`:** 100 per dash charge (300 = 3 dashes).
- **`rays`:** start straight ahead and go around the player.
- **`ground_rays`:** distance from the player's height down to the ground at points on a ring. A large value means a pit.
- **`cybergrind`:** only present in the Cyber Grind scene.
- **`player`:** `null` when no player exists (e.g. the main menu).
