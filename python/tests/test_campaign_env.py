"""Campaign episodes in UltrakillEnv, against a fake level instead of the game:  python tests/test_campaign_env.py  (or pytest)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.campaign import (  # noqa: E402
    ROUTE_SOURCE_GATES,
    ROUTE_SOURCE_ROOMS,
    ROUTE_VERSION,
    _read_route,
    load_route,
    route_path,
)
from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.protocol import BridgeClient  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402
from ultrakill_ai.spaces import BUTTONS, PITCH_BINS, YAW_BINS, noop_action  # noqa: E402

LEVEL = "Level 0-1"
EXIT_Z = 60.0
GROUND_RAY_LENGTH = 30.0  # what the mod writes when a ground ray hits nothing (ObsLayout.ground_ray_length)
CHECKPOINT_ID = "0,1,20"
ARENA_KEY = "0,1,30"
RESPAWN_DOOR_KEY = "0,0,25"
RANKS = {"time": [120, 90, 60, 30], "kills": [0, 1, 2, 3], "style": [0, 100, 200, 300]}
ENEMY_ID = 7
ENEMY_MAX_HP = 50.0
BANISH_X = 10000.0  # CheckPoint.Start / ResetRoom: defaultRooms[i].transform.position.x + 10000f
# Three doors along the corridor, the door graph the mod reports as campaign.gates: hops counts the rooms left
# after passing one, so 0 is the door into the exit's room.
GATES = (("0,1,15", (0.0, 1.0, 15.0), 2), ("0,1,35", (0.0, 1.0, 35.0), 1), ("0,1,55", (0.0, 1.0, 55.0), 0))
TIER_GATES = (("0,1,15", (0.0, 1.0, 15.0), 2), ("60,1,15", (60.0, 1.0, 15.0), 2), ("0,1,55", (0.0, 1.0, 55.0), 1))
GATE_BLOCK = 443 + 5  # absolute index of the campaign block's target slots (448-455)
WEDGE_HOLD = 45  # wedge_seconds 3.0 at 30 fps / frameskip 2 = 15 decisions/s
# The skull room (FakeLevel.enable_skulls): one altar-locked gate at z 50, its source on a pedestal at z 20 and
# its destination altar at z 40, so the whole fetch-carry-place leg is walkable straight down the corridor. The
# shape is Level 1-1's red leg: a live destination altar, a dead twin of it that can never fill, and a source
# skull that starts INSIDE an ItemPlaceZone (every ItemIdentifier on 1-1 does).
SKULL_GATE_KEY, SKULL_GATE_POS = "0,1,50", (0.0, 1.0, 50.0)
SKULL_GATES = ((SKULL_GATE_KEY, SKULL_GATE_POS, 0),)
# The altars stand like the game's. `pos` is the zone's transform, 1.9 m above this corridor's floor (y 1);
# `aim_pos` is its collider centre, the metre below that a placement punch has to hit, which for a player on
# the floor is exactly eye height (1 + camera_height_m). A skull on a pedestal, or resting in an altar, sits at
# that same centre -- so every one of these legs is a level shot, as it is in the real game.
PEDESTAL_KEY, PEDESTAL_POS = "0,1,20", (0.0, 2.9, 20.0)
ALTAR_KEY, ALTAR_POS = "0,1,40", (0.0, 2.9, 40.0)
ALTAR_AIM_POS = (0.0, 1.9, 40.0)
PEDESTAL_ITEM_POS = (0.0, 1.9, 20.0)  # where the source skull actually is, which is what a pickup aims at
ALTAR_ITEM_POS = (0.0, 1.9, 40.0)     # and where it sits once placed
DEAD_TWIN_KEY = "0,1,40#2"
ALTAR_ITEM = "SkullRed"  # what every ItemPlaceZone in the skull room accepts; FakeLevel.item_type is what exists
PUNCH_RANGE = 4.0  # Punch.ActiveFrame's own reach, and EnvConfig.subgoal_punch_range_m's default
# The two-level curriculum tests. Both names must be in CAMPAIGN_LEVELS_SHIPPED or UltrakillEnv refuses them.
LEVELS = ["Level 0-1", "Level 0-3"]
# Layer 2, the offline room trunk (docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md).
# Four rungs down the same corridor, 16 m apart so they clear invariant I2's own 16 m separation, ending 6 m
# short of the pit. The first rung is 6 m from the spawn, which is what the seeding rule absorbs.
ROUTE_RUNGS = ((0.0, 1.0, 6.0, 3), (0.0, 1.0, 22.0, 2), (0.0, 1.0, 38.0, 1), (0.0, 1.0, 54.0, 0))
# The 0-1 spawn shape the seeding rule exists for: the NEAREST rung of a total order is 14 m BEHIND the player
# while the route's next rung is 20 m ahead. Without seeding, the trunk aims at the room already left.
BEHIND_RUNGS = ((0.0, 1.0, -14.0, 2), (0.0, 1.0, 20.0, 1), (0.0, 1.0, 50.0, 0))


class FakeLevel:
    """Stands in for BridgeClient: a straight corridor along +z.

    Walking forward covers 2 m a step. The checkpoint at z 20 activates (and becomes current) on arrival, the
    arena at z 30 clears on arrival, gates sit at z 15, 35 and 55, and the exit is at z 60. A checkpoint respawn
    puts the player back at z 20 and unlocks a door, the way `StatsManager.Restart` unlocks `doorsToUnlock`: that
    unlock must never pay.

    The room holds one enemy, alive from the moment the level is entered or re-entered (`reset`, fresh or
    checkpoint), so a respawn re-creates it exactly like the real game while `kills` (the game's own counter)
    keeps whatever value it already had -- the pairing `test_kill_reward_can_be_farmed_across_a_checkpoint_respawn`
    documents.

    The switches stand in for states the real game gets into and for older mods:
      `falling`        off the map, ground rays missing, still moving (a void fall)
      `wedged`         the absorbing slowMode state: airborne, not sliding, not moving, z frozen
      `mod_flags`      False drops slow_mode/heavy_fall/crouching, as a mod older than 0.6.0 does
      `gates_present`  False drops the whole gates block, same reason
      `gates_ordered`  False is a level whose door graph has no goal room (0-5), i.e. the ROUTE MODE: with a
                       route file for the scene the env falls to layer 2 and walks the offline room trunk, and
                       `enable_route()` is the same switch under the name the route spec uses
      `ground_center`  overrides the centre ground ray, which the ring minimum can disagree with
      `skulls`         the skull room: an altar-locked gate, a pedestal source, a destination altar and its
                       dead twin, with `punch` picking up, placing and throwing exactly as `Punch.AltHit` does
      `skull_fields`   False keeps the skull room but drops `altars`, `items` and `needs_item`, as a mod older
                       than 0.7.0 does
      `altars_present` False is Level 0-4's shape: a carryable and not one `ItemPlaceZone` in the level, so the
                       gate is an ordinary door and the only thing a punch can do with the item is throw it
      `item_type`      what the carryable is. Anything but ALTAR_ITEM is an item no zone here accepts, which is
                       0-4's `CustomKey1` standing in a level that also has skull altars
      `item_active`    False is the carryable's room still switched off: it is reported but has no collider, so
                       no punch can reach it -- what a teleport into an unactivated room leaves you with

    Bridge faults are injected with `fail_resets` / `fail_steps`: each is a queue of exceptions, one popped and
    raised per call, so a test can say "time out the next two resets, then work". That is what the live failures
    look like from Python -- a `BridgeTimeout` on a reset, a `BridgeClosed` when the socket drops, a
    `BridgeSceneUnknown` from a game that has not finished booting -- and `connects` counts how often the env
    rebuilt the connection in response.
    """

    def __init__(self):
        self.settings: dict = {}  # what `configure` was last given; `_obs` echoes its difficulty back
        self.resets: list[bool] = []  # the checkpoint flag of every reset
        self.reset_scenes: list[str | None] = []  # the scene name of every reset, so a level switch is visible
        self.scene_name = LEVEL
        self.steps = 0
        self.kill_next = False  # the next step returns a dead player
        self.kill_enemy_next = False  # the next step removes the room's enemy, if alive, and pays a kill
        self.lock_steps = 0  # the next N steps report input_locked
        self.drop_campaign_steps = 0  # the next N obs omit "campaign", as the mod does when building it throws
        self.falling = False  # off the map: the ground rays miss and the player sinks, as in a real void fall
        self.wedged = False
        # Airborne but still MOVING, which `falling` and `wedged` are not: a jump, a dash, or the player
        # hanging against a wall on the far side of a room they have not entered. Set `ground_center` with it
        # to say how far the floor is; that pair is what `GateProgress._on_ground` reads.
        self.airborne = False
        self.mod_flags = True
        self.gates_present = True
        self.gates_ordered = True
        self.gates = GATES
        self.ground_center: float | None = None
        # `CheckPoint.Start` / `ResetRoom` move the ORIGINAL room by x + 10000 and leave a clone behind; the
        # mod's frozen FinalPit reference follows the original. This is that displacement, in banish steps:
        # set it mid-load to reproduce Level 0-2's jump, and the level load clears it like a real reload.
        self.exit_banish_steps = 0
        # Mod 0.7.2's `exit.ground_pos`: the NavMesh-snapped standable point near the pit. None is "NavMesh
        # found nothing", which is the same fall-back-to-`pos` case as a mod that does not send the field at
        # all -- so the default emits the key with a null and every test that does not care is unaffected.
        # `mod_ground_pos = False` drops the key, which is the pre-0.7.2 degradation case.
        self.exit_ground_pos: list[float] | None = None
        self.mod_ground_pos = True
        self.yaw = 0.0
        # `GunControl.currentSlotIndex`, 1-BASED (1..6, the game's own convention -- see `env.held_slot_key`)
        # and -1 before GunControl starts (0-1 has no weapon at all until the revolver pickup). 1 is the
        # revolver, the slot every level opens on. A plain attribute, NOT driven by the action: the corridor
        # models no weapons, and a slot that moved with the action would change the packed one-hot under every
        # existing test. tests/test_slot_counters.py and tests/test_sticky_slot.py set it directly, always to
        # a value the mod could really send.
        self.weapon_slot = 1
        # `GunControl.currentVariationIndex`, and `player.slot_counts` -- weapons per slot, slot 1 first, and
        # EMPTY until GunControl starts. Plain attributes for the same reason `weapon_slot` is one. The
        # default owns every slot the action space can press (keys 1..5), which is the mid-level state; a
        # test that wants 0-1's opening, where only the revolver has been picked up, sets it to [1,0,0,0,0].
        self.weapon_variation = 0
        self.slot_counts = [1, 1, 1, 1, 1]
        self.enemy_rel = [0.0, 0.0, 5.0]
        self.last_action: dict | None = None  # the command dict the env last sent
        self.skulls = False
        self.skull_fields = True
        self.altars_present = True
        self.altar_aim_pos = True  # False is a mod that sends no aim_pos: the Python fallback drop is used
        self.skull_gate_offladder = False  # 0-2's shape: the altar gate is off the exit chain, so hops is null
        self.item_type = ALTAR_ITEM
        self.item_active = True
        self.picked_up = 0  # times a punch picked the skull up, and times one threw it: the physical events
        self.thrown = 0
        # Injected bridge faults, popped one per call (see the class docstring).
        self.fail_resets: list[BaseException] = []
        self.fail_steps: list[BaseException] = []
        self.connects = 0
        self.configures = 0
        self.closes = 0
        self.drops = 0
        self._load()

    def fail_next_resets(self, exc: BaseException, times: int = 1) -> None:
        self.fail_resets.extend(type(exc)(*exc.args) for _ in range(times))

    def fail_next_steps(self, exc: BaseException, times: int = 1) -> None:
        self.fail_steps.extend(type(exc)(*exc.args) for _ in range(times))

    def enable_route(self) -> None:
        """Route mode: a level whose door graph has no goal room, so `_gates()` gives up and layer 2 fires.

        This is 0-5's measured shape -- `gates_ordered` false, every `hops` null -- and it is all the mod side
        of the route fallback needs, because a room rung carries no live state at all (spec §3). The trunk
        itself comes from the env's `route_dir`, which is what `route_env` writes.
        """
        self.gates_ordered = False

    def enable_skulls(self, *, fields: bool = True, altars: bool = True, item_type: str = ALTAR_ITEM) -> None:
        """Turns the level into the skull room. Call before reset(); `fields=False` is a mod older than 0.7.0.

        `altars=False` and `item_type` are the two ways a level can hold a carryable that nothing accepts; see
        the class docstring.
        """
        self.skulls = True
        self.skull_fields = fields
        self.altars_present = altars
        self.item_type = item_type
        self.gates = SKULL_GATES
        self._load()

    def _zone_accepts(self) -> bool:
        """Whether this level has an ItemPlaceZone that takes the item it ships (`ItemPlaceZone.CheckItem`)."""
        return self.altars_present and self.item_type == ALTAR_ITEM

    def _load(self) -> None:
        self.z = 0.0
        self.y = 1.0
        self.exit_banish_steps = 0  # a real level load re-instantiates the rooms, so nothing is banished yet
        # A level load puts the skull back on its pedestal. A checkpoint respawn does too, but under a fresh key:
        # CheckPoint.ResetRoom destroys and re-instantiates the room, which is why nothing may key on an instance.
        self.skull_key = "skull"
        self.skull_held = False
        self.skull_in: str | None = PEDESTAL_KEY if self._zone_accepts() else None
        self.skull_pos = list(PEDESTAL_ITEM_POS)
        self.seconds = 0.0
        self.kills = 0
        self.restarts = 0
        self.dead = False
        self.locked = False
        self.checkpoint = False
        self.arenas: list[str] = []
        self.doors: list[str] = []
        self.enemy_alive = False  # reset() below turns this on: every level (re-)entry re-creates the room
        self.enemy_health = ENEMY_MAX_HP

    # The BridgeClient methods UltrakillEnv uses ----------------------------

    def connect(self, retry_seconds: float = 60.0) -> dict:
        self.connects += 1
        return {"type": "hello", "protocol": 1, "mod_version": "0.5.0", "scene": LEVEL}

    def configure(self, **settings) -> None:
        self.configures += 1
        self.settings = settings

    def close(self) -> None:
        self.closes += 1

    def _drop(self) -> None:
        """The hang-up without `release` that UltrakillEnv uses on a mod it refuses (BridgeIncompatible)."""
        self.drops += 1

    def get_obs(self) -> dict:
        return self._obs()

    def kill(self) -> dict:
        """The protocol's debug kill. soft_death is not modelled here, so this is always a real death."""
        self.dead = True
        return self._obs("kill")

    def reset(self, scene: str | None = None, checkpoint: bool = False,
              timeout: float | None = None) -> dict:
        # `timeout` is accepted and ignored: the real client clamps a reset to what is left of the recovery
        # budget, and a fake that rejected the argument would pass while production raised TypeError.
        if self.fail_resets:
            raise self.fail_resets.pop(0)
        self.resets.append(checkpoint)
        self.reset_scenes.append(scene)
        switched = scene is not None and scene != self.scene_name
        if scene is not None:
            self.scene_name = scene
        if checkpoint and self.checkpoint and not switched:
            self.z = 20.0
            self.dead = False
            self.restarts += 1
            if RESPAWN_DOOR_KEY not in self.doors:
                self.doors.append(RESPAWN_DOOR_KEY)
            # The respawn re-instantiates the room, so its skull comes back under a new key.
            self.skull_key = f"skull#{self.restarts}"
            self.skull_held = False
            self.skull_in = PEDESTAL_KEY if self._zone_accepts() else None
            self.skull_pos = list(PEDESTAL_ITEM_POS)
        else:
            self._load()
        self.enemy_alive = True  # a fresh load or a checkpoint respawn both re-create the room's enemy
        self.enemy_health = ENEMY_MAX_HP
        return self._obs("reset")

    def _punch(self) -> None:
        """What `Punch.AltHit` does: place if holding and in reach of a zone, else throw; pick up if not holding."""
        if self.skull_held:
            if self._zone_accepts() and abs(self.z - ALTAR_POS[2]) <= PUNCH_RANGE:
                self.skull_held, self.skull_in, self.skull_pos = False, ALTAR_KEY, list(ALTAR_ITEM_POS)
            else:  # ActiveStart throws whatever it is holding when the active frame did not place it
                self.skull_held, self.skull_in, self.skull_pos = False, None, [0.0, self.y, self.z]
                self.thrown += 1
        elif self.item_active and abs(self.z - self.skull_pos[2]) <= PUNCH_RANGE:
            # Including out of a filled altar: AltHit's !holding branch is ForceHold whatever the skull sits in.
            self.skull_held, self.skull_in = True, None
            self.picked_up += 1

    def step(self, action: dict) -> dict:
        if self.fail_steps:
            raise self.fail_steps.pop(0)
        self.steps += 1
        self.last_action = dict(action)
        if self.falling:
            self.y -= 20.0  # terminal velocity through the void, a brand-new 4 m cell every step
        self.locked = self.lock_steps > 0
        if self.locked:
            self.lock_steps -= 1
        if not self.dead and self.z < EXIT_Z and not self.wedged:
            if self.kill_next:
                self.dead, self.kill_next = True, False
            elif not self.locked and action.get("move", [0, 0])[1] > 0:
                self.z = min(EXIT_Z, self.z + 2.0)
            self.seconds += 2.0 / 15.0
        if self.skulls and not self.dead and not self.locked and "punch" in (action.get("buttons") or ()):
            self._punch()
        if self.skull_held:
            self.skull_pos = [0.0, self.y, self.z]  # a carried skull moves with the player
        if self.enemy_alive and self.kill_enemy_next:
            # A one-shot kill: the enemy vanishes without a health drop first, same as the mod reports it.
            self.enemy_alive, self.kill_enemy_next = False, False
            self.kills += 1
        if self.z >= 20.0:
            self.checkpoint = True
        if self.z >= 30.0 and ARENA_KEY not in self.arenas:
            self.arenas.append(ARENA_KEY)
        return self._obs()

    def _gates_block(self) -> dict:
        """The gates the mod reports, or nothing at all when `gates_present` stands in for an older mod."""
        if not self.gates_present:
            return {}
        gates = [{
            "key": key, "pos": list(pos),
            "hops": None if (not self.gates_ordered or (self.skull_gate_offladder and key == SKULL_GATE_KEY))
            else hops,
            "open": abs(self.z - pos[2]) <= 8.0,  # the DoorController proximity trigger
            "locked": False, "active": True, "controller_active": True,
        } for key, pos, hops in self.gates]
        if self.skulls and self.skull_fields and self.altars_present:
            for g in gates:
                if g["key"] == SKULL_GATE_KEY:
                    # A skull-locked door reports exactly like a walk-up door; needs_item is the only difference.
                    # It clears from the LIVE altar only -- the dead twin can never fill (M14/X1).
                    g["needs_item"] = None if self.skull_in == ALTAR_KEY else ALTAR_ITEM
                    g["altar_only"] = True
        return {"gates_ordered": self.gates_ordered, "gates_truncated": False, "gates": gates}

    def _skull_block(self) -> dict:
        """`campaign.altars` and `campaign.items`, or nothing at all against a mod older than 0.7.0."""
        if not self.skulls or not self.skull_fields:
            return {}
        def zone(key, pos, filled, doors, ancestors=1, aim=None):
            z = {"key": key, "pos": list(pos), "item": ALTAR_ITEM, "filled": filled, "active": True,
                 "inactive_ancestors": ancestors, "reverse_doors": [],
                 "doors": [{"key": k, "pos": list(SKULL_GATE_POS)} for k in doors]}
            if aim is not None and self.altar_aim_pos:
                z["aim_pos"] = list(aim)
            return z
        altars = [
            # The pedestal drives no door and reads filled the moment its room switches on (31 zones
            # campaign-wide do); the live destination altar opens the gate; the dead twin never can.
            zone(PEDESTAL_KEY, PEDESTAL_POS, self.skull_in == PEDESTAL_KEY, ()),
            zone(ALTAR_KEY, ALTAR_POS, self.skull_in == ALTAR_KEY, (SKULL_GATE_KEY,), aim=ALTAR_AIM_POS),
            zone(DEAD_TWIN_KEY, ALTAR_POS, False, (SKULL_GATE_KEY,), ancestors=2, aim=ALTAR_AIM_POS),
        ]
        return {
            "altars": altars if self.altars_present else [],
            "items": [{
                "key": self.skull_key, "pos": list(self.skull_pos), "item": self.item_type, "held": self.skull_held,
                "placed": self.skull_in is not None, "placed_in": self.skull_in, "active": self.item_active,
                "active_self": True, "inactive_ancestors": 1,
            }],
        }

    def _exit_block(self) -> dict:
        block = {"pos": [BANISH_X * self.exit_banish_steps, 1.0, EXIT_Z], "active": True}
        if self.mod_ground_pos:
            block["ground_pos"] = (list(self.exit_ground_pos) if self.exit_ground_pos is not None else None)
        return block

    def _ground_rays(self) -> list[float]:
        # The mod measures down from the player to the floor (y = 1 in this corridor) and writes
        # ground_ray_length exactly when the ray hits nothing, which is what being off the map looks like.
        if self.falling:
            return [GROUND_RAY_LENGTH] * 8
        return [min(GROUND_RAY_LENGTH, max(0.0, self.y - 1.0))] * 8

    def _obs(self, event: str | None = None) -> dict:
        over = self.z >= EXIT_Z
        enemies = []
        if self.enemy_alive:
            enemies = [{
                "id": ENEMY_ID, "type": 0, "health": self.enemy_health, "visible": True,
                "rel": list(self.enemy_rel), "dist": 5.0, "pos": [0.0, 1.0, self.z + 5.0],
            }]
        airborne = self.falling or self.wedged or self.airborne
        player = {
            "pos": [0.0, self.y, self.z], "vel": [0.0, -20.0 if self.falling else 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, 1.0], "yaw": self.yaw, "pitch": 0.0, "hp": 0 if self.dead else 100,
            "anti_hp": 0.0, "stamina": 300.0, "grounded": not airborne, "sliding": False, "dead": self.dead,
            "activated": True, "level_over": over, "weapon_slot": self.weapon_slot,
            "weapon_variation": self.weapon_variation,
            "soft_deaths": 0, "soft_death_instakill": False, "slot_counts": list(self.slot_counts),
        }
        if self.mod_flags:  # mod 0.6.0 and later
            player.update(slow_mode=self.wedged, heavy_fall=self.falling, crouching=False)
        obs = {
            "type": "obs",
            "step": self.steps,
            "scene": self.scene_name,
            "ready": not self.dead,
            "player": player,
            "enemies": enemies,
            "rays": [50.0] * 16,
            "ground_rays": self._ground_rays(),
            "ground_ray_center": self.ground_center if self.ground_center is not None else self._ground_rays()[0],
            "stats": {"kills": self.kills, "style": 0, "seconds": self.seconds, "restarts": self.restarts, "level_complete": over},
            "campaign": {
                # The difficulty the GAME read, which the mod reports back each step. The fake echoes whatever
                # the env asked for at reset, exactly as `CampaignPatches.DifficultyOverride` does, so a test
                # can follow a difficulty all the way from the config to the episode info.
                "mission": 1, "difficulty": self.settings.get("difficulty", 3),
                "seconds": self.seconds, "timer_running": not over,
                "level_started": True, "level_over": over, "restarts": self.restarts, "input_locked": self.locked,
                "exit": self._exit_block(),
                "checkpoints": [{"id": CHECKPOINT_ID, "pos": [0.0, 1.0, 20.0], "activated": self.checkpoint, "current": self.checkpoint}],
                "path": {"status": "complete", "length": EXIT_Z - self.z, "next_corner": [0.0, 1.0, EXIT_Z]},
                "locked_doors": [],
                "arena_enemies_alive": 0,
                "cleared_arenas": list(self.arenas),
                "unlocked_doors": list(self.doors),
                "ranks": RANKS,
                **self._gates_block(),
                **self._skull_block(),
            },
        }
        if event:
            obs["event"] = event
        if self.drop_campaign_steps > 0:
            # docs/protocol.md: the mod logs and omits the whole campaign block for a step if building it throws.
            self.drop_campaign_steps -= 1
            del obs["campaign"]
        return obs


def make_env(rewards: RewardConfig | None = None, **overrides) -> tuple[UltrakillEnv, FakeLevel]:
    rewards = rewards or RewardConfig(time=0.01, checkpoint=10.0, arena_clear=10.0, door_unlock=3.0, novelty=0.5,
                                      path=0.1, gate=15.0, gate_approach=0.15)
    cfg = EnvConfig(mode="campaign", level=LEVEL, fixed_fps=30, frameskip=2, rewards=rewards, **overrides)
    env = UltrakillEnv(cfg)
    env.client = FakeLevel()
    return env, env.client


def write_route(directory, *, level: str = LEVEL, rungs=ROUTE_RUNGS, exit_pos=(0.0, 1.0, EXIT_Z),
                version: int = ROUTE_VERSION, **overrides) -> Path:
    """Writes one `route_<scene>.json` in the spec's §4.1 schema and returns its directory.

    Every field the schema lists is written, diagnostics included, so a test fixture cannot accidentally pass
    against a loader that ignores half the file. The real files come from `build_routes.py`; until they exist
    these hand-written ones are what the loader is tested on, and the wrap-up stage runs it over the real 12.

    The loader caches parsed documents per (scene, dir), so the cache is cleared here: a test that rewrites one
    file in one temp directory must not read the previous test's document back.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # `rungs` is normally (x, y, z, hops) tuples; a test that wants a malformed file passes the entries itself.
    entries = list(rungs) if any(isinstance(r, dict) for r in rungs) else [
        {"key": f"{round(x)},{round(y)},{round(z)}", "pos": [float(x), float(y), float(z)],
         "hops": int(hops), "name": f"{len(rungs) - int(hops)} - Room", "gated_by": [],
         "open": False, "locked": False, "active": True}
        for x, y, z, hops in rungs]
    doc = {
        "level": level, "version": version, "source": "room-trunk offline v2 (test fixture)",
        "exit": {"pos": [float(v) for v in exit_pos], "target": "Level 0-2"},
        "start_room": "1 - Opening Hallway", "trunk_collapsed": [],
        "tour_ratio": 1.0, "checkpoints_within_60m": "1/1", "legs_witnessed": f"{len(entries)}/{len(entries) + 1}",
        "last_rung_to_exit_m": round(abs(exit_pos[2] - entries[-1]["pos"][2]), 1) if entries else 0.0,
        "rungs": entries,
    }
    doc.update(overrides)
    route_path(level, str(directory)).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    _read_route.cache_clear()
    return directory


def route_env(tmp, *, rungs=ROUTE_RUNGS, exit_pos=(0.0, 1.0, EXIT_Z), ordered: bool = False,
              **overrides) -> tuple[UltrakillEnv, FakeLevel, Path]:
    """An env on a level with a route file, and (by default) no usable gate ladder: layer 2.

    `ordered=True` keeps the gate ladder working, which is how the "the trunk is never read when the gates are
    good" case is set up: the file exists, the level ships gates, and nothing must consult the trunk.
    """
    route_dir = write_route(Path(tmp) / "routes", rungs=rungs, exit_pos=exit_pos)
    env, fake = make_env(route_dir=str(route_dir), **overrides)
    if not ordered:
        fake.enable_route()
    return env, fake, route_dir


def idle():
    return noop_action(campaign=True)


def forward():
    a = idle()
    a[0] = 2  # move forward
    return a


def action(*, look_mode: int = 0, yaw: float = 0.0, pitch: float = 0.0, buttons: tuple[str, ...] = (), move: bool = False):
    a = idle()
    if move:
        a[0] = 2
    for name in buttons:
        a[2 + BUTTONS.index(name)] = 1
    i = 2 + len(BUTTONS)
    a[i + 1] = YAW_BINS.index(yaw)
    a[i + 2] = PITCH_BINS.index(pitch)
    a[i + 3] = look_mode
    return a


def add_parts(total: dict[str, float], info: dict) -> None:
    for name, value in info["reward_parts"].items():
        total[name] = total.get(name, 0.0) + value


def run_until_end(env: UltrakillEnv, limit: int = 500) -> dict:
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(idle())
        if terminated or truncated:
            return info
    raise AssertionError(f"episode did not end within {limit} steps")


def test_fresh_start_walked_to_the_exit_completes_the_level():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(best_runs_dir=str(Path(tmp) / "best_runs"), difficulty=3, unlock_all_gear=True)
        _, info = env.reset(seed=0)
        assert fake.resets == [False]
        assert fake.settings["difficulty"] == 3 and fake.settings["unlock_all_gear"] is True
        assert fake.settings["soft_death"] is False  # real deaths drive respawns in the campaign
        assert info["fresh_start"] == 1 and info["completed"] == 0 and info["level_seconds"] is None
        parts: dict[str, float] = {}
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["end_reason"] == "level_complete"
        assert {"checkpoint", "arena_clear", "level_complete", "time"} <= set(parts)
        assert parts["checkpoint"] == 10.0 and parts["arena_clear"] == 10.0  # each paid exactly once
        assert all(key in info for key in CAMPAIGN_INFO_KEYS)
        assert info["completed"] == 1 and info["fresh_start"] == 1
        assert info["level_seconds"] == fake.seconds
        assert isinstance(info["rank"], str)
        assert info["checkpoints_level"] == 1 and info["cells_new"] > 0 and info["exit_dist_min"] == 0.0
        env.close()

        run = json.loads((Path(tmp) / "best_runs" / "Level_0-1.json").read_text(encoding="utf-8"))
        assert run["level"] == LEVEL and run["seconds"] == fake.seconds and run["deaths"] == 0
        assert run["positions"][0] == [0.0, 1.0, 0.0] and run["positions"][-1] == [0.0, 1.0, EXIT_Z]


def test_the_episode_info_carries_the_difficulty_the_game_actually_read():
    """Not `cfg.difficulty`: the config ASKS, the mod's campaign block ANSWERS, and only the answer is a fact.

    It is what `ProgressCallback` folds into `best_time`, the fresh-episode windows and status.json, so a
    Brutal round's numbers can never be silently compared with a Violent one's (2026-09-20).
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(best_runs_dir=str(Path(tmp) / "best_runs"), difficulty=4)
        env.reset(seed=0)
        assert fake.settings["difficulty"] == 4
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete"
        assert info["difficulty"] == 4, "the episode says which difficulty it was played on"
        env.close()
        run = json.loads((Path(tmp) / "best_runs" / "Level_0-1.json").read_text(encoding="utf-8"))
        assert run["difficulty"] == 4, "and so does the best-run file post_times.py reads"


def test_an_episode_that_does_not_complete_still_reports_its_difficulty():
    with tempfile.TemporaryDirectory() as tmp:
        env, _ = make_env(fresh_start_prob=0.0, max_steps=5, best_runs_dir=tmp, difficulty=4)
        env.reset(seed=0)
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert not info.get("completed"), "the level is not finished in five decisions"
        assert info["difficulty"] == 4
        env.close()


def test_completion_after_a_checkpoint_respawn_has_no_official_time():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(fresh_start_prob=0.0, max_steps=25, best_runs_dir=tmp)
        env.reset(seed=0)
        for _ in range(25):  # z 50: past the checkpoint, short of the exit, then the 25-decision cap
            _, _, terminated, truncated, info = env.step(forward())
        assert truncated and info["end_reason"] == "max_steps"
        _, info = env.reset()
        assert fake.resets == [False, True] and info["fresh_start"] == 0 and fake.z == 20.0
        for _ in range(25):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["completed"] == 1
        assert info["level_seconds"] is None and "rank" not in info  # the timer ran on from the earlier episode
        env.close()
        assert not (Path(tmp) / "Level_0-1.json").exists()


# ---------------------------------------------------------------------------
# Speed stages (docs/superpowers/specs/2026-09-18-speed-stages.md)
# ---------------------------------------------------------------------------


def speed_env(**overrides):
    """A speed stage's env: the same rewards as a complete stage, with the scaled completion bonus on."""
    rewards = RewardConfig(time=0.01, checkpoint=10.0, arena_clear=10.0, level_complete=100.0)
    return make_env(rewards=rewards, speed_bonus=True, **overrides)


def test_a_speed_stage_reads_the_levels_own_s_rank_time_as_its_target():
    """Never a hand-picked number: the target is `campaign.ranks.time[-1]` scaled, the game's own idea of S."""
    env, fake = speed_env()
    _, info = env.reset(seed=0)
    assert RANKS["time"][-1] == 30
    # The default scale is 0.75 (spec §8b): an S-rank time is a competent run, not a fast one.
    assert env._speed_target == 22.5 and info["target_seconds"] == 22.5
    assert env._s_rank_seconds == 30.0 and info["s_rank_seconds"] == 30.0, "the raw threshold travels too"
    assert info["completion_bonus"] == 0.0, "nothing has been completed yet"
    env.close()

    # ... and a run that is not a speed stage carries no target at all, so it pays the plain weight.
    plain, _ = make_env(rewards=RewardConfig(level_complete=100.0))
    _, info = plain.reset(seed=0)
    assert plain._speed_target is None and info["target_seconds"] is None
    assert info["s_rank_seconds"] is None
    plain.close()


def test_the_target_scale_multiplies_the_s_threshold_and_is_applied_only_here():
    """§8b: one place, the env, so `info["target_seconds"]` is what the reward AND the driver both read."""
    for scale, expected in ((1.0, 30.0), (0.75, 22.5), (0.5, 15.0)):
        env, _ = speed_env(speed_target_scale=scale)
        _, info = env.reset(seed=0)
        assert info["target_seconds"] == expected and info["s_rank_seconds"] == 30.0
        assert env.cfg.rewards.level_complete == 100.0, "the scale never touches a reward weight"
        env.close()


def test_a_config_override_wins_over_the_live_threshold_and_is_never_scaled():
    """A per-level `speed.targets` entry is a decision already made: 8 s means 8 s, not 8 x 0.75 (§8b)."""
    env, fake = speed_env(speed_target_seconds=8.0, speed_target_scale=0.5)
    _, info = env.reset(seed=0)
    assert env._speed_target == 8.0 and info["target_seconds"] == 8.0, "the plan's per-level override"
    assert info["s_rank_seconds"] == 30.0, "the raw threshold is still recorded beside it"
    env.close()


def test_a_level_load_with_no_ranks_leaves_the_target_unread():
    """An older mod, or a block the mod could not build: no target, the plain bonus, and nothing raises."""
    env, fake = speed_env()
    fake.drop_campaign_steps = 1  # the reset observation has no campaign block at all
    _, info = env.reset(seed=0)
    assert env._speed_target is None and info["target_seconds"] is None
    assert env._s_rank_seconds is None and info["s_rank_seconds"] is None
    info = run_forward_until_end(env)
    assert info["end_reason"] == "level_complete"
    assert info["completion_bonus"] == 100.0, "the plain weight: no target means today's rule exactly"
    env.close()


def run_forward_until_end(env: UltrakillEnv, limit: int = 200) -> dict:
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            return info
    raise AssertionError(f"episode did not end within {limit} steps")


def test_a_fast_fresh_start_completion_pays_the_scaled_bonus():
    """The fake corridor is finished in about four seconds against a 22.5 s target, so it hits the 2.0 ceiling."""
    env, fake = speed_env()
    _, info = env.reset(seed=0)
    info = run_forward_until_end(env)
    assert info["end_reason"] == "level_complete" and info["fresh_start"] == 1
    assert info["level_seconds"] == fake.seconds and fake.seconds < 22.5
    assert info["completion_bonus"] == 200.0 == info["reward_parts"]["level_complete"]
    assert info["target_seconds"] == 22.5 and info["s_rank_seconds"] == 30.0
    env.close()


def test_the_bonus_tracks_the_official_time_between_the_two_clips():
    """With the target set to twice the corridor's own time the bonus is exactly half the ceiling."""
    env, fake = speed_env(speed_target_seconds=4.0)
    env.reset(seed=0)
    info = run_forward_until_end(env)
    seconds = info["level_seconds"]
    assert seconds is not None and seconds > 0
    assert abs(info["completion_bonus"] - 100.0 * min(2.0, 4.0 / seconds)) < 1e-9
    assert 25.0 <= info["completion_bonus"] <= 200.0
    env.close()


def test_a_completion_the_game_gave_no_official_time_pays_the_floor():
    """2026-09-19, live on `spec_0-2_speed`: the completion frame arrived AFTER the level stats had reset.

    The episode ran 4,120 decisions and the frame said `seconds` 0.0 with `restarts` 3. The completion is
    real and still ends the episode; only its clock is missing, so it reports no time and no rank and writes
    no best run -- a 0.0 in `best_runs/` is a record nothing real can ever beat.

    IT PAYS THE FLOOR, NOT THE PLAIN WEIGHT (2026-09-20). Paying the full 100 made a completion with NO clock
    the best-paying completion on the whole stage: every genuine one slower than the target pays 39-99. The
    floor is what the slowest genuine completion approaches and never reaches, so a lost timer can never be
    worth more than a real run -- and it still pays four times what not finishing pays, which is the other
    safety property. NOT VALIDATED IN GAME.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = speed_env(best_runs_dir=tmp)
        env.reset(seed=0)
        info = None
        for _ in range(200):
            if fake.z >= EXIT_Z - 2.0:
                fake.seconds, fake.restarts = 0.0, 3  # StatsManager reset under us, one frame early
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["completed"] == 1 and info["fresh_start"] == 1
        assert fake.seconds <= 1.0, "the fake really did report a time no level run can have taken"
        assert info["level_seconds"] is None and info["rank"] is None
        assert info["completion_bonus"] == 25.0 == info["reward_parts"]["level_complete"], "the floor"
        assert not list(Path(tmp).glob("*.json")), "no official time, no best run"
        env.close()


def test_a_checkpoint_respawn_completion_pays_the_floor_on_a_speed_stage():
    """Its official timer carries over from an earlier episode, so its "time" says nothing about the run.

    On a SPEED STAGE that makes it the same case as a lost timer, and since 2026-09-20 it is priced the same
    way: the floor, never the full weight. It cannot arise on a shipped speed stage at all -- they force
    `fresh_start_prob: 1.0` precisely so that the stage never trains on episodes it does not score -- and this
    test builds one by hand to pin the price if a future stage lowers that. A complete stage has no target and
    is untouched: it pays the plain weight for every completion, respawn or not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = speed_env(fresh_start_prob=0.0, max_steps=25, best_runs_dir=tmp)
        env.reset(seed=0)
        for _ in range(25):
            _, _, terminated, truncated, info = env.step(forward())
        assert truncated and info["end_reason"] == "max_steps"
        _, info = env.reset()
        assert info["fresh_start"] == 0
        info = run_forward_until_end(env)
        assert info["end_reason"] == "level_complete" and info["level_seconds"] is None
        assert info["completion_bonus"] == 25.0, "the floor: no usable clock, so it cannot out-earn a real run"
        env.close()


def test_death_after_the_checkpoint_respawns_inside_the_episode():
    env, fake = make_env()
    env.reset(seed=0)
    parts: dict[str, float] = {}
    for _ in range(11):  # z 22, just past the checkpoint
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
    fake.kill_next = True
    _, _, terminated, truncated, info = env.step(idle())
    add_parts(parts, info)
    assert not terminated and not truncated
    assert fake.resets == [False, True]
    assert info["deaths"] == 1 and "death" in info["reward_parts"]
    assert RESPAWN_DOOR_KEY in env._raw["campaign"]["unlocked_doors"]
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert "door_unlock" not in parts  # the respawn's own unlock pays nothing
    assert parts["checkpoint"] == 10.0  # and the respawn checkpoint is not paid again
    assert info["end_reason"] == "level_complete" and info["deaths"] == 1 and info["checkpoints_level"] == 1
    env.close()


def test_death_before_any_checkpoint_reloads_the_level_inside_the_episode():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake = make_env(best_runs_dir=tmp)
        env.reset(seed=0)
        for _ in range(5):  # z 10, short of the checkpoint
            env.step(forward())
        fake.kills = 2
        fake.kill_next = True
        _, _, terminated, truncated, info = env.step(idle())
        assert not terminated and not truncated
        assert fake.resets == [False, True] and fake.z == 0.0  # no checkpoint yet, so the level reloaded
        assert info["deaths"] == 1 and info["kills"] == 2  # kills from before the reload still count
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["level_seconds"] == fake.seconds and info["kills"] == 2
        env.close()
        run = json.loads((Path(tmp) / "Level_0-1.json").read_text(encoding="utf-8"))
        assert len(run["positions"]) == 31 and run["positions"][0] == [0.0, 1.0, 0.0]  # only the attempt that finished


def test_kill_reward_can_be_farmed_across_a_checkpoint_respawn():
    """Documents current, intended-by-omission behaviour -- not an endorsement of it. A checkpoint respawn
    re-creates the room's enemies (FakeLevel.enemy_alive goes back to True on every reset, matching the mod
    re-instantiating the room), but the game's own kill counter does not reset with it, so re-killing the same
    enemy after a death pays `kill` and `damage_dealt` again. This is a known reward-farming risk to watch
    during training, not something this change fixes.
    """
    env, fake = make_env()
    env.reset(seed=0)
    for _ in range(11):  # z 22, just past the checkpoint so a death respawns instead of reloading the level
        env.step(forward())

    fake.kill_enemy_next = True
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated
    first_kill, first_damage = info["reward_parts"]["kill"], info["reward_parts"]["damage_dealt"]
    assert info["kills"] == 1 and first_kill > 0 and first_damage > 0

    fake.kill_next = True  # the player also dies now, which triggers a checkpoint respawn
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated  # a death inside a campaign episode respawns it, not ends it
    assert info["deaths"] == 1
    assert fake.enemy_alive and fake.kills == 1  # the room's enemy is back; the kill counter did not reset

    fake.kill_enemy_next = True
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated
    assert info["kills"] == 2 and fake.kills == 2
    assert info["reward_parts"]["kill"] == first_kill  # paid again for the re-created enemy
    assert info["reward_parts"]["damage_dealt"] == first_damage
    env.close()


def test_missing_campaign_block_mid_episode_does_not_break_the_episode():
    """docs/protocol.md: the mod logs and omits the whole `campaign` block for a step if building it throws.
    The env must survive that -- no exception, the stuck clock keeps ticking instead of freezing or getting
    confused, milestones are neither re-paid nor lost once the block returns, and combat rewards (which never
    touch the campaign block) are still paid during the gap.
    """
    env, fake = make_env(fresh_start_prob=0.0)
    env.reset(seed=0)
    parts: dict[str, float] = {}
    for _ in range(10):  # z 20: the checkpoint activates, so there is a milestone to protect across the gap
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
    assert parts["checkpoint"] == 10.0 and env._steps_since_progress == 0

    fake.drop_campaign_steps = 2
    fake.kill_enemy_next = True
    _, _, terminated, truncated, info = env.step(idle())  # campaign block missing, but the kill still lands
    add_parts(parts, info)
    assert not terminated and not truncated
    assert info["kills"] == 1
    assert info["reward_parts"]["kill"] > 0 and info["reward_parts"]["damage_dealt"] > 0
    assert env._steps_since_progress == 0  # the kill counts as progress; it does not need the campaign block

    _, _, terminated, truncated, info = env.step(idle())  # still missing (2nd of the 2 dropped steps)
    add_parts(parts, info)
    assert not terminated and not truncated
    assert fake.drop_campaign_steps == 0
    # No checkpoint/arena/door/novelty/path/gate signal while the block is gone, and no kill this step either.
    assert env._steps_since_progress == 1

    for _ in range(40):  # the block is back; finish the level to prove the gap left nothing corrupted
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert info["end_reason"] == "level_complete"
    assert parts["checkpoint"] == 10.0  # still paid exactly once: neither re-paid nor lost across the gap
    env.close()


def test_three_stuck_episodes_at_one_checkpoint_force_a_fresh_load():
    env, fake = make_env(fresh_start_prob=0.0, stuck_seconds=1.0)  # 15 decisions without progress
    env.reset(seed=0)
    for _ in range(10):  # z 20: the checkpoint activates
        env.step(forward())
    assert run_until_end(env)["end_reason"] == "stuck"
    for _ in range(2):
        _, info = env.reset()
        assert info["fresh_start"] == 0
        assert run_until_end(env)["end_reason"] == "stuck"
    _, info = env.reset()
    assert fake.resets == [False, True, True, False]
    assert info["fresh_start"] == 1
    env.close()


def test_input_locked_frames_are_skipped():
    env, fake = make_env()
    env.reset(seed=0)
    fake_steps, env_steps = fake.steps, env._steps
    fake.lock_steps = 3
    env.step(idle())
    assert fake.steps - fake_steps == 3 + 1  # the policy's step, then empty steps until the lock ends
    assert env._steps - env_steps == 1
    assert env._raw["campaign"]["input_locked"] is False
    env.close()


def test_campaign_observation_is_479_values():
    env, _ = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (479,) and env.observation_space.shape == (479,)
    assert env.observation_space.contains(obs)
    assert obs[443 + 4] == 1.0  # exit mask: the campaign block starts after the 2 Cyber Grind values
    assert obs[443 + 27] > 0.0  # exploration map: the spawn cell has been entered once
    obs, *_ = env.step(forward())
    assert env.observation_space.contains(obs)
    env.close()


def test_cybergrind_default_stays_448_without_connecting():
    assert EnvConfig().layout.size == 448
    env = UltrakillEnv(EnvConfig())
    assert env.observation_space.shape == (448,)
    assert not env._connected and env.client._sock is None
    cfg = EnvConfig(mode="campaign")
    campaign = UltrakillEnv(cfg)
    assert campaign.observation_space.shape == (479,) and not campaign._connected
    assert campaign.cfg.layout.campaign and not cfg.layout.campaign  # the caller's config is left alone


def test_from_dict_ignores_retired_keys():
    cfg = EnvConfig.from_dict({
        "mode": "campaign",
        "fixed_fps": 30,
        "checkpoint_resets": True,
        "stuck_steps": 450,
        "route_points_file": "routes/human_0-1.json",
        "layout": {"max_enemies": 8, "waypoint_offset": 3},
        "rewards": {"kill": 0.5, "route_point": 0.1, "stuck": 1.0},
    })
    assert cfg.mode == "campaign" and cfg.fixed_fps == 30 and cfg.layout.max_enemies == 8 and cfg.rewards.kill == 0.5
    for retired in ("checkpoint_resets", "stuck_steps", "route_points_file"):
        assert not hasattr(cfg, retired)
    assert not hasattr(cfg.rewards, "route_point") and not hasattr(cfg.rewards, "stuck")
    assert EnvConfig.from_dict(cfg.to_dict()) == cfg


def test_exploration_archive_is_saved_on_close_and_loaded_again():
    with tempfile.TemporaryDirectory() as tmp:
        env, _ = make_env(explore_dir=tmp)
        env.reset(seed=0)
        for _ in range(5):
            env.step(forward())
        counts = dict(env.archive.counts)
        assert counts
        env.close()
        assert (Path(tmp) / f"explore_Level_0-1_{env.cfg.port}.npz").exists()
        again = UltrakillEnv(env.cfg)
        assert again.archive.counts == counts


def test_bridge_client_kill_sends_the_kill_command():
    client = BridgeClient()
    sent = []
    client.request = lambda msg: sent.append(msg) or {"type": "obs", "event": "kill"}
    assert client.kill() == {"type": "obs", "event": "kill"}
    assert sent == [{"type": "kill"}]


def test_falling_off_the_map_pays_no_novelty():
    """The exploit that broke the first campaign run, pinned so it cannot come back.

    Novelty used to be keyed on the raw player position, so a fall through the void entered a brand-new 4 m cell
    every single step and paid the full 1/sqrt(0+1) = 1.0 for each one. It never decayed (those cells are never
    revisited), the fall never ended, and any novelty > 0 reset the stuck clock, so diving off the level was an
    unbounded income stream that strictly beat playing it. The real run banked 47% of all its novelty below
    y = -60 m, reaching 57 km under the map, and completed the level zero times in 397 episodes.
    """
    env, level = make_env()
    env.reset()
    level.falling = True
    parts: dict[str, float] = {}
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    env.close()
    assert level.y <= -300.0, "the fake level should have fallen well below the map"
    assert "novelty" not in parts, f"a fall must pay no novelty, got {parts.get('novelty')}"
    assert info["cells_new"] == 0
    assert info["oob_frac"] > 0.9  # nearly every step of this episode had no ground beneath


def test_novelty_pays_for_new_ground_not_for_height():
    """Jumping on the spot pays once; running forward keeps paying.

    Keying on the ground point is what separates the two: the real run's playable footprint was 1,397 columns
    but 15,251 cells, ~11 stacked per column, so 90% of what it earned inside the level was for being airborne
    over ground it had already been paid for.
    """
    env, level = make_env()
    env.reset()
    # Hovering 12 m up over the same spot: the ground point never moves, so only the first step pays.
    paid = []
    for _ in range(4):
        level.y += 3.0
        _, _, _, _, info = env.step(idle())
        paid.append(info["reward_parts"].get("novelty", 0.0))
    assert paid[0] == 0.0 and sum(paid) == 0.0, f"height alone must not pay, got {paid}"
    # Moving forward over new floor still pays, airborne or not -- fast movement in this game is airborne.
    before = env._cells_new
    for _ in range(4):
        env.step(forward())
    env.close()
    assert env._cells_new > before, "covering new ground must still pay while airborne"


def test_walking_the_corridor_pays_each_gate_once():
    """The gate ladder: one payment per new lower hops value, and nothing at all on a second pass."""
    env, fake = make_env(fresh_start_prob=0.0, max_steps=27)  # stop at z 54, short of the exit
    env.reset(seed=0)
    parts: dict[str, float] = {}
    for _ in range(11):  # z 22: past the gate at 15 (hops 2) and the checkpoint, short of the one at 35
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
    assert parts["gate"] == 15.0 and info["gates_reached"] == 1 and info["gate_hops_best"] == 2
    assert parts["gate_approach"] > 0.0
    for _ in range(16):  # on past the hops 1 and hops 0 gates
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
    assert truncated and info["end_reason"] == "max_steps"
    assert parts["gate"] == 45.0 and info["gates_reached"] == 3 and info["gate_hops_best"] == 0

    # A respawn absorbs what the player already stands at; re-walking the same doors pays nothing.
    _, info = env.reset()
    assert fake.resets == [False, True] and info["fresh_start"] == 0
    again: dict[str, float] = {}
    for _ in range(27):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(again, info)
    assert "gate" not in again, f"the gate ladder is per level load, got {again.get('gate')}"
    assert info["gates_reached"] == 3  # still counted, just not paid again
    env.close()


def test_gate_slots_track_the_target():
    env, fake = make_env()
    obs, info = env.reset(seed=0)
    # retarget runs before reset's observation is packed, so the target is live on the very first decision.
    assert obs[GATE_BLOCK + 4] == 1.0  # target mask
    assert "reward_parts" not in info  # retarget picks a target without paying for it
    assert abs(obs[GATE_BLOCK + 2] - 15.0 / 50.0) < 1e-6  # the hops 2 gate, 15 m straight ahead, gate scale
    assert abs(obs[GATE_BLOCK + 3] - 15.0 / 100.0) < 1e-6
    assert obs[GATE_BLOCK + 7] == 2.0 / 20.0  # hops / 20
    for _ in range(9):  # z 18: the hops 2 gate is reached, so the target is the hops 1 gate at z 35
        obs, _, _, _, _ = env.step(forward())
    assert abs(obs[GATE_BLOCK + 2] - (35.0 - 18.0) / 50.0) < 1e-6
    assert obs[GATE_BLOCK + 7] == 1.0 / 20.0
    for _ in range(15):  # z 48: past the hops 0 gate, so the target becomes the exit, at the exit's own scales
        obs, _, _, _, _ = env.step(forward())
    assert obs[GATE_BLOCK + 4] == 1.0 and obs[GATE_BLOCK + 5] == 0.0 and obs[GATE_BLOCK + 7] == 0.0
    assert abs(obs[GATE_BLOCK + 2] - (EXIT_Z - 48.0) / 100.0) < 1e-6
    env.close()


# -- the banished FinalPit (docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md) -----------

def test_a_banished_exit_is_ignored_for_the_target_the_observation_and_exit_dist():
    """Level 0-2's bug: `CheckPoint.Start` moves the ORIGINAL room x + 10000 and the mod's exit follows it."""
    env, fake = make_env()
    obs, _ = env.reset(seed=0)
    assert obs[443 + 4] == 1.0 and abs(obs[443 + 2] - EXIT_Z / 100.0) < 1e-6
    for _ in range(24):  # z 48: past the hops 0 gate, so the target is the exit itself
        obs, _, _, _, info = env.step(forward())
    assert not info["exit_banished"]
    before = env.gates.target["pos"]
    fake.exit_banish_steps = 1  # the checkpoint's room clone lands, and the reported pit jumps +10,000 on x
    obs, _, _, _, info = env.step(forward())
    assert info["exit_banished"] == 1
    assert env.gates.target["pos"] == before, "the target stays on the real pit"
    assert abs(obs[443 + 2] - (EXIT_Z - 50.0) / 100.0) < 1e-6, "and so do observation slots 0-4"
    assert obs[443] == 0.0, "rel x would read ~100 (10,000 m / the 100 m scale, clipped) against the twin"
    assert abs(obs[443 + 3] - (EXIT_Z - 50.0) / 200.0) < 1e-6
    assert info["exit_dist_min"] < 100.0, "exit_dist_min never sees the 9,801 m twin"
    fake.exit_banish_steps = 2  # ResetRoom runs again on the next respawn: the offset is k * 10000
    obs, _, terminated, truncated, info = env.step(forward())
    assert env.gates.target["pos"] == before and info["exit_banished"] == 1
    env.close()


def test_a_banished_exit_does_not_stop_the_level_completing():
    env, fake = make_env()
    env.reset(seed=0)
    fake.exit_banish_steps = 1
    parts: dict[str, float] = {}
    for _ in range(35):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert info["end_reason"] == "level_complete" and info["exit_banished"] == 1
    assert parts.get("gate", 0) > 0
    env.close()


def test_a_new_level_load_accepts_whatever_exit_it_reports():
    env, fake = make_env()
    env.reset(seed=0)
    fake.exit_banish_steps = 1
    env.step(forward())
    assert env.exit_guard.banished
    env.reset(seed=1)  # a fresh load: FakeLevel._load puts the rooms back, as the real game does
    assert not env.exit_guard.banished and env.exit_guard.frozen == [0.0, 1.0, EXIT_Z]
    _, _, _, _, info = env.step(forward())
    assert info["exit_banished"] == 0
    env.close()


# -- the exit's STANDABLE point (mod 0.7.2's exit.ground_pos) ---------------------------------------------
#
# A `FinalPit`'s transform sits inside the drop it triggers, not on anything you can walk on: 61-75 m below
# the floor on `Level 0-2` and 70 m on `Level 0-3`. Every consumer of `GateProgress._exit` is a place the
# agent is being TOLD TO GO -- look mode 2, `gate_approach`, the target slots -- so aiming them at `pos`
# aims them off the ledge above the pit. Measured live on 0-2: eleven of one episode's eighteen respawns
# were falls taken at full health while following that vector, and both of that level's real completions
# triggered at y -25 to -27.5 while `exit.pos` read y -86.1.
#
# FakeLevel's corridor is flat, so `ground_pos` is put 8 m short of the pit along +z: near enough to be the
# same place, far enough that every assertion below can tell which one is being used.

EXIT_GROUND = [0.0, 1.0, EXIT_Z - 8.0]


def test_the_exit_target_prefers_the_standable_point():
    env, fake = make_env()
    fake.exit_ground_pos = list(EXIT_GROUND)
    env.reset(seed=0)
    for _ in range(24):  # z 48: past the hops 0 gate, so the target IS the exit
        _, _, _, _, info = env.step(forward())
    assert env.gates.target["key"] == "exit"
    assert env.gates.target["pos"] == EXIT_GROUND, \
        f"the exit target is {env.gates.target['pos']}, not the standable point"
    env.close()


def test_the_raw_exit_vector_the_policy_reads_is_unchanged():
    """Observation slots 0-4 are a learned input column the policy has read since the run began, so they stay
    on `exit.pos` whatever `ground_pos` says. Only the TARGET slots (448-455) follow the new point."""
    env, fake = make_env()
    fake.exit_ground_pos = list(EXIT_GROUND)
    obs, _ = env.reset(seed=0)
    plain, _ = make_env()
    plain.client = FakeLevel()  # the same level with no ground_pos at all
    plain.client.mod_ground_pos = False
    obs_plain, _ = plain.reset(seed=0)
    assert list(obs[443:448]) == list(obs_plain[443:448]), "slots 0-4 moved with ground_pos"
    assert abs(obs[443 + 2] - EXIT_Z / 100.0) < 1e-6, "slot 2 is still the raw pit's z"
    env.close()
    plain.close()


def test_the_target_slots_follow_the_standable_point():
    """The other half of the pair above: the slots that DO change, and by exactly the 8 m offset."""
    env, fake = make_env()
    fake.exit_ground_pos = list(EXIT_GROUND)
    env.reset(seed=0)
    for _ in range(24):
        obs, _, _, _, _ = env.step(forward())
    # Slot 5-8 of the campaign block: rel xyz over 100, then the 3-D distance over 200 (the exit scales).
    z_rel = obs[GATE_BLOCK + 2] * 100.0
    assert abs(z_rel - (EXIT_GROUND[2] - 48.0)) < 0.5, f"target slot z is {z_rel}"
    env.close()


def test_exit_ground_dist_min_is_reported_beside_exit_dist_min():
    env, fake = make_env()
    fake.exit_ground_pos = list(EXIT_GROUND)
    env.reset(seed=0)
    for _ in range(24):
        _, _, _, _, info = env.step(forward())
    assert info["exit_ground_dist_min"] is not None
    assert abs(info["exit_ground_dist_min"] - info["exit_dist_min"]) > 7.0, \
        "the two distances must be measured to different points"
    assert info["exit_ground_dist_min"] < info["exit_dist_min"], "the standable point is the nearer one here"
    env.close()


def test_an_old_mod_without_ground_pos_behaves_exactly_as_today():
    """The degradation case, and the one that has to be airtight: a game still running mod 0.7.1 sends no
    `ground_pos`, so every consumer must fall back to `pos` and the episode must be identical."""
    new, fake_new = make_env()
    fake_new.mod_ground_pos = False          # mod 0.7.1: the key is absent
    old, fake_old = make_env()
    fake_old.mod_ground_pos = True
    fake_old.exit_ground_pos = None          # mod 0.7.2 where NavMesh found nothing: the key is null
    a = b = None
    for env in (new, old):
        env.reset(seed=0)
        for _ in range(24):
            _, _, _, _, info = env.step(forward())
        if a is None:
            a = (list(env.gates.target["pos"]), info["exit_dist_min"], info["exit_ground_dist_min"])
        else:
            b = (list(env.gates.target["pos"]), info["exit_dist_min"], info["exit_ground_dist_min"])
    assert a == b, f"a null ground_pos and an absent one differ: {a} vs {b}"
    assert a[0] == [0.0, 1.0, EXIT_Z], "both fall back to the pit's own position"
    assert a[2] is None, "and neither reports a standable distance"
    new.close()
    old.close()


def test_a_ground_pos_below_the_pit_is_refused():
    """Found by the first in-game run of this code, on the level it was built for. A nearest-mesh-in-any-
    direction snap at `Level 0-2`'s pit returns (-199, -133.5, 277) -- **47.4 m further DOWN the shaft** than
    the pit's own (-199, -86.1, 277), while the ground the level is actually completed from is 60 m the other
    way, at y -25 to -27.5. Aiming at that would have been strictly worse than the bug being fixed.

    Below the pit is the shaft, never the ledge over it. The mod now searches upward and refuses such a hit;
    this is the same rule repeated in Python, so an older or differently-tuned mod cannot reintroduce it. The
    fall-back is `pos`, which is exactly the behaviour before `ground_pos` existed.
    """
    env, fake = make_env()
    fake.exit_ground_pos = [0.0, -50.0, EXIT_Z]  # a point 51 m down the shaft, as 0-2 really reported
    env.reset(seed=0)
    for _ in range(24):
        _, _, _, _, info = env.step(forward())
    assert env.gates.target["pos"] == [0.0, 1.0, EXIT_Z], \
        f"the target went down the shaft: {env.gates.target['pos']}"
    assert info["exit_ground_dist_min"] is None, "a refused ground_pos must not be measured to either"
    env.close()


def test_a_ground_pos_level_with_the_pit_is_still_accepted():
    """The boundary, so the guard above cannot quietly become 'only strictly above'. A pit that already sits
    on walkable ground reports a `ground_pos` at its own height, and that is the normal case."""
    env, fake = make_env()
    fake.exit_ground_pos = [0.0, 1.0, EXIT_Z - 8.0]  # same y as the pit, 8 m nearer
    env.reset(seed=0)
    for _ in range(24):
        env.step(forward())
    assert env.gates.target["pos"] == [0.0, 1.0, EXIT_Z - 8.0]
    env.close()


def test_a_banished_exit_takes_its_standable_point_with_it():
    """`ground_pos` is sampled AROUND the position being reported, so a banished twin's sample belongs to the
    twin. `ExitGuard` restores `pos` and must drop `ground_pos`, or the target would follow a standable point
    10 km away -- the very failure the guard exists to stop, through the new field."""
    env, fake = make_env()
    fake.exit_ground_pos = list(EXIT_GROUND)
    env.reset(seed=0)
    for _ in range(24):
        env.step(forward())
    assert env.gates.target["pos"] == EXIT_GROUND
    fake.exit_banish_steps = 1
    fake.exit_ground_pos = [BANISH_X, 1.0, EXIT_Z - 8.0]  # the sample the mod would take at the twin
    _, _, _, _, info = env.step(forward())
    assert info["exit_banished"] == 1
    assert env.gates.target["pos"] == [0.0, 1.0, EXIT_Z], \
        f"the target followed the banished twin's standable point: {env.gates.target['pos']}"
    env.close()


def test_targets_parked_is_zero_on_a_monotone_level():
    """The corridor is 0-1's shape: every rung is walked in order, so nothing is ever parked."""
    env, _ = make_env()
    env.reset(seed=0)
    for _ in range(35):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            break
    assert info["end_reason"] == "level_complete"
    assert info["targets_parked"] == 0 and info["exit_banished"] == 0
    env.close()


def test_patience_is_measured_in_game_seconds_whatever_the_frameskip():
    for fps, skip in ((30.0, 2), (60.0, 4), (60.0, 1)):
        cfg = EnvConfig(mode="campaign", level=LEVEL, fixed_fps=fps, frameskip=skip, gate_target_patience_s=20.0)
        env = UltrakillEnv(cfg)
        env.client = FakeLevel()
        assert env.gates.patience_steps == int(round(20.0 * fps / skip))
        env.close()
    env, _ = make_env(gate_target_patience_s=0.0)
    assert env.gates.patience_steps == 0, "0 turns parking off and restores the pre-patience tracker"
    env.close()


# A COLLAPSED ladder in the corridor, with 0-3's shape and 0-3's ratio: the gate nearest the spawn already
# carries hops 1 of 3, the rung below it is 200 m straight up, and the walkable route runs forward through the
# hops 3 door. `detect_collapsed_ladder` reads 1 <= 3/2, so parking is allowed here under the shipped default.
COLLAPSED_GATES = (("0,1,15", (0.0, 1.0, 15.0), 1), ("0,201,15", (0.0, 201.0, 15.0), 0),
                   ("0,1,55", (0.0, 1.0, 55.0), 3))
# 0-1's shape, with the one thing the plain corridor lacks: a door deeper in the level that happens to be
# physically NEAR (0-1 has several -- `202,56,534` at hops 2 sits 82 m from `202,56,452` at hops 1). That is
# what satisfies the "a park is only ever a switch" guard on a healthy ladder, so it is the shape where
# unconditional patience really does take the agent off the correct door.
MONOTONE_WITH_SIDE_DOOR = (("0,1,15", (0.0, 1.0, 15.0), 2), ("0,1,35", (0.0, 1.0, 35.0), 1),
                           ("0,1,55", (0.0, 1.0, 55.0), 0), ("10,1,30", (10.0, 1.0, 30.0), 0))


def test_a_target_the_agent_cannot_reach_is_parked_in_a_real_episode():
    """The 0-3 shape through the env: the ladder's next rung is 200 m up, and standing still must not wedge.

    Run at the SHIPPED settings, with no `gate_patience_mode` override: on a level whose ladder collapses at the
    spawn the detector switches the whole mechanism on by itself, which is the point of the 2026-09-17 revision.
    """
    env, fake = make_env(max_steps=1200, stuck_seconds=1000.0)
    fake.gates = COLLAPSED_GATES
    fake._load()
    env.reset(seed=0)
    assert env.gates.ladder_collapsed is True and env.gates.patience_active
    for _ in range(9):  # reach the hops 1 gate, which locks best_hops at 1
        env.step(forward())
    assert env.gates.target["key"] == "0,201,15", "the ladder points at the unreachable rung"
    for _ in range(env.gates.patience_steps + 2):
        _, _, terminated, truncated, info = env.step(idle())
        if terminated or truncated:
            break
    assert env.gates.parked == {"0,201,15"}
    assert env.gates.target["key"] == "0,1,55", "the nearest unreached, unparked gate at any hop count"
    for _ in range(25):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            break
    assert info["targets_parked"] == 1 and info["ladder_collapsed"] == 1
    env.close()


def test_a_monotone_level_parks_nothing_however_long_the_agent_stalls():
    """The 0-1 regression, as an env test: the same stall, on a healthy ladder, with and without the detector.

    Live, the unconditional mechanism parked a target in 34% of 0-1's fresh episodes and its fresh completion
    rate went 0.55 (n=43) to 0.30 (n=56). The corridor here is 0-1's shape -- hops 2, 1, 0 in walking order --
    so the honest behaviour is the one on the left: nothing to switch to, so nothing is parked.
    """
    for mode, parks in (("collapsed", 0), ("always", 1)):
        env, fake = make_env(max_steps=2000, stuck_seconds=1000.0, gate_patience_mode=mode)
        fake.gates = MONOTONE_WITH_SIDE_DOOR
        fake._load()
        env.reset(seed=0)
        assert env.gates.ladder_collapsed is False, "the corridor's nearest gate is its top rung"
        for _ in range(9):  # through the first door, so the target is the correct next one
            env.step(forward())
        assert env.gates.target["key"] == "0,1,35"
        for _ in range(env.gates.patience_steps + 2):
            _, _, terminated, truncated, info = env.step(idle())
            if terminated or truncated:
                break
        assert env.gates.parks == parks, mode
        assert (env.gates.target["key"] == "0,1,35") is (mode == "collapsed"), \
            f"{mode}: the correct door is kept only when the detector says the ladder is right"
        env.close()


def test_the_verdict_reaches_the_episode_info_for_every_level():
    """`ladder_collapsed` is per episode and per level, so the dashboard can show `parked/ep` against it."""
    for gates_array, want in ((GATES, 0), (COLLAPSED_GATES, 1)):
        env, fake = make_env(max_steps=40)
        fake.gates = gates_array
        fake._load()
        env.reset(seed=0)
        for _ in range(35):
            _, _, terminated, truncated, info = env.step(forward())
            if terminated or truncated:
                break
        assert info["ladder_collapsed"] == want, gates_array
        env.close()


def test_unordered_level_falls_back_to_the_exit_vector():
    env, fake = make_env()
    fake.gates_ordered = False  # 0-5: the door graph has no goal room, so every hops is null
    obs, _ = env.reset(seed=0)
    assert not obs[GATE_BLOCK : GATE_BLOCK + 8].any()
    assert obs[443 + 4] == 1.0  # the straight-line exit vector is still there, as it was before gates existed
    parts: dict[str, float] = {}
    for _ in range(35):
        obs, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert info["end_reason"] == "level_complete" and info["gates_reached"] == 0
    assert "gate" not in parts and "gate_approach" not in parts
    assert not obs[GATE_BLOCK : GATE_BLOCK + 8].any()
    env.close()


def test_old_mod_without_gates_still_runs():
    """Graceful degradation: the Python side has to run against a mod that sends no gates block at all."""
    env, fake = make_env()
    fake.gates_present = False
    obs, _ = env.reset(seed=0)
    assert not obs[GATE_BLOCK : GATE_BLOCK + 8].any()
    parts: dict[str, float] = {}
    for _ in range(35):
        _, _, terminated, truncated, info = env.step(action(look_mode=2, move=True))  # mode 2 with no target: free look
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert info["end_reason"] == "level_complete"
    assert "gate" not in parts and info["gates_reached"] == 0 and info["gate_hops_best"] is None
    env.close()


def test_a_wedged_player_ends_the_episode():
    env, fake = make_env()
    env.reset(seed=0)
    fake.wedged = True
    for step in range(WEDGE_HOLD):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info["end_reason"] == "wedged" and step == WEDGE_HOLD - 1
    assert info["wedged_steps"] == WEDGE_HOLD  # the whole run, credited when it tripped
    env.close()


def test_a_short_wedge_is_not_counted():
    env, fake = make_env()
    env.reset(seed=0)
    fake.wedged = True
    for _ in range(WEDGE_HOLD - 1):
        _, _, terminated, truncated, info = env.step(forward())
    assert not terminated and not truncated and info["wedged_steps"] == 0
    fake.wedged = False
    _, _, terminated, truncated, info = env.step(forward())
    assert info["wedged_steps"] == 0, "a run that never reached the hold is ordinary airborne time"
    env.close()


def test_falling_is_not_wedged():
    """Ordinary airborne time flags the state flags but keeps moving; without the movement term it would be
    26.9% of all decisions."""
    env, fake = make_env()
    env.reset(seed=0)
    fake.falling = True
    for _ in range(WEDGE_HOLD + 5):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            break
    assert info["wedged_steps"] == 0 and info.get("end_reason") != "wedged"
    env.close()


def test_wedge_fallback_without_mod_flags():
    """A mod older than 0.6.0 sends no slow_mode/heavy_fall: the same predicate without the flag test."""
    env, fake = make_env()
    fake.mod_flags = False
    env.reset(seed=0)
    assert "slow_mode" not in env._raw["player"]
    fake.wedged = True
    for _ in range(WEDGE_HOLD):
        _, _, terminated, truncated, info = env.step(forward())
        if terminated or truncated:
            break
    assert truncated and info["end_reason"] == "wedged" and info["wedged_steps"] == WEDGE_HOLD
    env.close()


def test_three_wedges_force_a_fresh_start():
    """`wedged` has to feed the same escape hatch as `stuck`, or a wedge-prone checkpoint is inescapable."""
    env, fake = make_env(fresh_start_prob=0.0)
    env.reset(seed=0)
    for _ in range(10):  # z 20: the checkpoint activates
        env.step(forward())
    fake.wedged = True
    assert run_until_end(env)["end_reason"] == "wedged"
    for _ in range(2):
        _, info = env.reset()
        assert info["fresh_start"] == 0
        assert run_until_end(env)["end_reason"] == "wedged"
    _, info = env.reset()
    assert fake.resets == [False, True, True, False]
    assert info["fresh_start"] == 1
    env.close()


def test_kills_reset_the_stuck_clock():
    """Four of 0-1's gate doors are behind kill gates in rooms where novelty runs out: fighting is progress."""
    env, fake = make_env(stuck_seconds=1.0)  # 15 decisions without progress
    env.reset(seed=0)
    for _ in range(14):
        _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated and env._steps_since_progress == 14
    fake.kills += 1  # the game's own kill counter, which only the player moves
    env.step(idle())
    assert env._steps_since_progress == 0
    for _ in range(14):
        _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated, "29 decisions of standing still, and the kill kept it alive"
    env.close()


def test_enemy_crossfire_does_not_reset_the_stuck_clock():
    """Damage dealt is unattributed (enemies damage each other), so it must not hold an episode open."""
    env, fake = make_env(stuck_seconds=1.0)
    env.reset(seed=0)
    for _ in range(14):
        _, _, terminated, truncated, info = env.step(idle())
    fake.enemy_health -= 10.0  # something hurt the enemy; no kill and no style
    _, _, terminated, truncated, info = env.step(idle())
    assert info["reward_parts"].get("damage_dealt", 0.0) > 0.0
    assert truncated and info["end_reason"] == "stuck"
    env.close()


def test_respawn_resets_the_stuck_clock():
    env, fake = make_env(stuck_seconds=1.0)
    env.reset(seed=0)
    for _ in range(10):  # z 20, the checkpoint
        env.step(forward())
    for _ in range(5):
        env.step(idle())
    assert env._steps_since_progress == 5
    fake.kill_next = True
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated and info["deaths"] == 1
    assert env._steps_since_progress == 0, "a respawn moves the player, so the pre-death clock is meaningless"
    env.close()


def test_slide_min_hold_holds_slide():
    env, fake = make_env(slide_min_hold=3)
    env.reset(seed=0)
    env.step(action(buttons=("slide",)))
    assert "slide" in fake.last_action["buttons"]
    for _ in range(2):  # the next N-1 decisions get it added
        env.step(idle())
        assert "slide" in fake.last_action["buttons"]
    env.step(idle())
    assert "slide" not in fake.last_action["buttons"]
    assert env._info(env._raw)["slide_forced_frac"] > 0.0
    env.step(action(buttons=("slide",)))
    env.reset()
    env.step(idle())
    assert "slide" not in fake.last_action["buttons"], "the latch is cleared on reset"
    env.close()


def settle(env: UltrakillEnv) -> None:
    """One idle decision, so the next one is resolved against a fresh observation.

    A look mode aims from `prev`, the observation the policy acted on, so a change made to the fake level only
    reaches the geometry after the next step brings it back.
    """
    env.step(idle())


def test_look_mode_enemy_turns_toward_the_enemy():
    env, fake = make_env()
    env.reset(seed=0)
    fake.enemy_rel = [5.0, 0.0, 0.0]  # 90 degrees to the right, in camera space
    settle(env)
    env.step(action(look_mode=1))
    assert fake.last_action["look"] == [90.0, 0.0]
    fake.enemy_rel = [0.0, 5.0, 0.0]  # straight overhead: positive pitch, capped at one decision's worth
    settle(env)
    env.step(action(look_mode=1))
    assert fake.last_action["look"][1] == 20.0
    fake.enemy_alive = False  # nothing visible: exactly mode 0, the sampled bins
    settle(env)
    env.step(action(look_mode=1, yaw=30.0, pitch=-6.0))
    assert fake.last_action["look"] == [30.0, -6.0]
    env.close()


def test_look_mode_gate_turns_toward_the_target():
    env, fake = make_env()
    env.reset(seed=0)
    env.step(action(look_mode=2))
    # The hops-2 gate is straight ahead; the pitch is the camera's own 0.9 m over the sill 15 m away, which is
    # 3.4 degrees down. See test_look_mode_2_aims_from_the_camera_not_from_the_feet.
    assert fake.last_action["look"][0] == 0.0 and abs(fake.last_action["look"][1] + 3.43) < 0.01
    fake.yaw = 90.0  # facing +x, so the gate along +z is on the left
    settle(env)
    env.step(action(look_mode=2))
    assert fake.last_action["look"][0] < 0.0
    fake.yaw, fake.y = 0.0, -100.0  # far below the gate: clamped to the band, then to one decision's worth
    settle(env)
    env.step(action(look_mode=2))
    assert fake.last_action["look"][1] == 20.0  # pitch_limit_deg 0 means "off": the 85 degree fallback, not 0
    env.close()

    banded, fake = make_env(pitch_limit_deg=5.0)
    banded.reset(seed=0)
    fake.y = -100.0
    settle(banded)
    banded.step(action(look_mode=2))
    assert abs(fake.last_action["look"][1] - 5.0) < 1e-6
    banded.close()


def test_look_mode_2_aims_from_the_camera_not_from_the_feet():
    """F1. `Punch.ActiveFrame` rays from `cc.GetDefaultPos()` -- the camera, 0.9 m above `player.pos` -- so an
    aim computed from the player transform points the ray straight over a target at its own feet's height.

    At punch range that is the whole answer: atan(0.9/1.5) is 31 degrees, and it is why the skull leg could
    never place. In the far field it is small and in the RIGHT direction: a camera 0.9 m up really does have to
    look slightly down at a door sill on its own floor -- atan(0.9/10) = 5.1 degrees, atan(0.9/30) = 1.7.
    """
    env, fake = make_env()
    env.reset(seed=0)
    env.step(action(look_mode=2))  # the hops-2 gate: same y, 15 m down the corridor
    far = fake.last_action["look"]
    assert far[0] == 0.0, "yaw is unaffected: only the height of the ray's origin changed"
    assert abs(far[1] - -math.degrees(math.atan2(0.9, 15.0))) < 1e-9 and abs(far[1] + 3.43) < 0.01
    env.close()

    # The same geometry at punch range, and the same aim taken from the feet, for the size of what was wrong.
    # gate_reach_m is shrunk only so standing 3 m from the gate is not "reaching" it and retargeting.
    near, fake = make_env(gate_reach_m=0.5, gate_reach_v_m=0.5)
    near.reset(seed=0)
    fake.z = 12.0  # 3 m short of the gate, close enough to punch and inside one decision's 20 degrees
    settle(near)
    near.step(action(look_mode=2))
    assert abs(fake.last_action["look"][1] + math.degrees(math.atan2(0.9, 3.0))) < 1e-9
    assert abs(fake.last_action["look"][1] + 16.70) < 0.01, "16.7 degrees at 3 m, and 31 at 1.5 m (PITCH_CAP 20)"
    near.close()

    feet, fake = make_env(camera_height_m=0.0, gate_reach_m=0.5, gate_reach_v_m=0.5)  # the override
    feet.reset(seed=0)
    fake.z = 12.0
    settle(feet)
    feet.step(action(look_mode=2))
    assert fake.last_action["look"] == [0.0, 0.0], "aimed from the feet the gate is exactly level: the old bug"
    feet.close()


def test_look_mode_1_is_already_camera_relative_and_is_left_alone():
    """F1, the other half: verified rather than assumed. `enemies[].rel` is `cam.InverseTransformPoint(centre)`
    (ObservationBuilder.cs:246), i.e. already measured FROM the camera, so mode 1 needs no eye correction and
    must not be given one -- adding 0.9 m there would introduce the very error mode 2 had.
    """
    for height in (0.0, 0.9, 5.0):
        env, fake = make_env(camera_height_m=height)
        env.reset(seed=0)
        fake.enemy_rel = [0.0, 0.0, 5.0]  # dead ahead in camera space, 5 m out
        settle(env)
        env.step(action(look_mode=1))
        assert fake.last_action["look"] == [0.0, 0.0], f"camera_height_m {height} must not move mode 1"
        env.close()


def test_look_mode_is_popped_from_the_command():
    env, fake = make_env()
    env.reset(seed=0)
    for mode in (0, 1, 2):
        env.step(action(look_mode=mode))
        assert "look_mode" not in fake.last_action, "the mod never sees it; the wire message is unchanged"
    env.close()


def test_look_mode_counters_record_the_mode_that_applied():
    """A mode that finds nothing to aim at IS mode 0 that step, and must be counted and graded as one.

    Counting the requested mode instead reported `look_free_frac` 0% for a run whose look heads were driving the
    camera on every single step -- the share R6 says to read the per-dimension entropy against -- and suppressed
    every yaw/pitch tracking sample on those steps. Both fall-backs are common: mode 2 has no target at all on an
    unordered level or against a 0.5.x mod, and mode 1 falls back whenever nothing is visible.
    """
    env, fake = make_env(max_steps=20)
    fake.gates_present = False  # mode 2 with no target: the sampled bins drive the camera, as in mode 0
    env.reset(seed=0)
    fake.enemy_rel = [5.0, 0.0, 0.0]  # 90 degrees right, so the sampled yaw bin can be graded against it
    settle(env)
    for _ in range(10):
        _, _, _, _, info = env.step(action(look_mode=2, yaw=30.0))
    assert fake.last_action["look"] == [30.0, 0.0], "the look heads, not the geometry, aimed this step"
    assert info["look_free_frac"] == 1.0 and info["look_gate_frac"] == 0.0
    assert info["yaw_track"] == 1.0, "a fall-back step is a look-head sample and has to be graded"
    env.close()

    env, fake = make_env(max_steps=20)
    env.reset(seed=0)
    fake.enemy_alive = False  # mode 1 with nothing visible: the same fall-back
    settle(env)
    for _ in range(10):
        _, _, _, _, info = env.step(action(look_mode=1, yaw=30.0))
    assert fake.last_action["look"] == [30.0, 0.0]
    assert info["look_free_frac"] == 1.0 and info["look_enemy_frac"] == 0.0
    env.close()

    env, fake = make_env(max_steps=20)  # the control: a mode that really does aim is counted and left ungraded
    env.reset(seed=0)
    fake.enemy_rel = [5.0, 0.0, 0.0]
    settle(env)
    for _ in range(10):
        _, _, _, _, info = env.step(action(look_mode=1, yaw=30.0))
    assert fake.last_action["look"] == [90.0, 0.0], "the geometry aimed this step"
    assert abs(info["look_enemy_frac"] - 10 / 11) < 1e-9 and abs(info["look_free_frac"] - 1 / 11) < 1e-9
    assert info["yaw_track"] == 0.0, "an aimed step would score ~1.0 by construction, so it is not a sample"
    env.close()


def test_max_steps_can_be_set_per_level():
    """F5. A `levels` ladder's rungs are not the same size, so one cap for all of them is either too short for
    the long level or too generous for the short one. Empty (the default) is the unchanged single cap.
    """
    env, _ = make_env(max_steps=4)
    env.reset(seed=0)
    assert env._max_steps() == 4, "no override: max_steps, exactly as before"
    for _ in range(4):
        _, _, _, truncated, info = env.step(forward())
    assert truncated and info["end_reason"] == "max_steps"
    env.close()

    env, _ = make_env(max_steps=4, max_steps_per_level={LEVEL: 8})
    env.reset(seed=0)
    assert env._max_steps() == 8
    for _ in range(4):
        _, _, _, truncated, _ = env.step(forward())
    assert not truncated, "this level's own cap is what applies"
    env.close()

    env, _ = make_env(max_steps=4, max_steps_per_level={"Level 9-9": 8})
    env.reset(seed=0)
    assert env._max_steps() == 4, "a level the table does not name falls back to max_steps"
    env.close()


def test_ground_ray_center_is_preferred():
    """The ring minimum can be a ledge 4 m away (p90 spread 10.8 m); the centre ray is the floor underfoot."""
    env, fake = make_env()
    env.reset(seed=0)
    fake.ground_center = 8.0  # the ring still reads 0 m to the floor at y 1
    env.step(idle())
    assert (0, -2, 0) in env.archive.counts, sorted(env.archive.counts)
    assert (0, 0, 0) in env.archive.counts  # the spawn cell, from the ring, before the override
    fake.ground_center = GROUND_RAY_LENGTH  # the centre misses while every ring ray hits: no ground under us
    _, _, _, _, info = env.step(idle())
    assert info["oob_frac"] > 0.0
    env.close()


def test_archive_saved_on_a_step_schedule():
    """Episodes are thousands of decisions long, so an episode-counted save alone saves nothing in practice."""
    with tempfile.TemporaryDirectory() as tmp:
        env, _ = make_env(explore_dir=tmp, archive_save_steps=10)
        env.reset(seed=0)
        path = Path(tmp) / f"explore_Level_0-1_{env.cfg.port}.npz"
        for _ in range(500):  # the schedule is checked every 500 steps
            _, _, terminated, truncated, _ = env.step(idle())
            assert not terminated and not truncated
        assert path.exists(), "the archive should be saved mid-episode, without close() or 20 episodes"
        env.close()


def test_episode_info_has_the_new_keys():
    env, fake = make_env(fresh_start_prob=0.0, max_steps=25)
    env.reset(seed=0)
    for _ in range(25):  # z 50: past the checkpoint and two gates, short of the exit, then the cap
        _, _, terminated, truncated, info = env.step(forward())
    assert truncated and info["end_reason"] == "max_steps"
    for key in CAMPAIGN_INFO_KEYS:
        assert key in info, key
    for key in ("gate_hops_best", "look_free_frac", "look_enemy_frac", "start_checkpoint", "end_pos"):
        assert key in info, key
    assert info["level_started"] == 1 and info["start_checkpoint"] is None  # a fresh load starts at no checkpoint
    assert info["end_pos"] == [0.0, 1.0, 50.0] and len(info["end_pos"]) == 3
    assert info["look_free_frac"] == 1.0 and info["look_gate_frac"] == 0.0
    # z 50 is within reach of the hops 0 gate at z 55 as well, so all three are behind us.
    assert info["gates_reached"] == 3 and info["gate_hops_best"] == 0 and info["wedged_steps"] == 0

    _, info = env.reset()  # a checkpoint respawn: the episode records where it began
    assert info["fresh_start"] == 0 and info["start_checkpoint"] == CHECKPOINT_ID
    _, _, _, _, info = env.step(forward())
    assert info["start_checkpoint"] == CHECKPOINT_ID
    env.close()


# ---------------------------------------------------------------------------------------------
# S2: layer 2, the offline room trunk (2026-09-17-route-fallback-and-boss-levels-design.md §7.2)
# ---------------------------------------------------------------------------------------------

def test_route_fallback_pays_the_ladder():
    """The whole point: a level with no usable gate ladder walks its shipped trunk and pays for it.

    Three of the four rungs pay, not four, and the missing one is the seeding rule doing its job: the spawn is
    6 m from rung `0,1,6`, so that rung is ABSORBED rather than earned (see
    `test_route_seeding_absorbs_the_nearest_rung`). A second walk in the same level load pays nothing at all,
    exactly as a gate ladder does -- `gate` is per level load, and it is the same rule, not a parallel one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=29)
        _, info = env.reset(seed=0)
        assert env.gates.route_source == 2 and info["route_source"] == 2
        assert info["route_source_name"] == "rooms"
        parts: dict[str, float] = {}
        for _ in range(29):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert truncated and info["end_reason"] == "max_steps" and fake.z == 58.0
        assert parts["gate"] == 45.0, f"one instalment per rung descended, got {parts['gate']}"
        assert parts["gate_approach"] > 0.0
        assert info["gates_reached"] == 4 and info["gate_hops_best"] == 0
        assert info["route_source"] == 2 and env.gates.route_reads > 0

        _, info = env.reset()  # a checkpoint respawn: the trunk is per level load, like the gate ladder
        assert fake.resets == [False, True] and info["fresh_start"] == 0
        again: dict[str, float] = {}
        for _ in range(29):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(again, info)
        assert "gate" not in again, f"the trunk is per level load, got {again.get('gate')}"
        assert info["gates_reached"] == 4 and info["route_source"] == 2
        env.close()


def test_a_room_rung_is_not_credited_from_the_air_over_nothing():
    """`GateProgress._on_ground`, end to end through the env, on the geometry it was measured on.

    The same corridor walk, flown 12 m above the floor instead of walked along it, pays no rung at all. That
    is Level 0-3's bug: its `2 - Side Hallway` rung sat 1.5 m past the main room's far wall, so 680 of the
    rung's 801 live credit steps were the player hanging against the wall 10-20 m above the floor of the room
    BEFORE it, and the ladder then advanced the target to the next rung through the wall.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=29)
        env.reset(seed=0)
        fake.airborne, fake.ground_center = True, 12.0  # airborne, floor 12 m down: the wall-face signature
        parts: dict[str, float] = {}
        for _ in range(29):
            _, _, _, _, info = env.step(forward())
            add_parts(parts, info)
        assert fake.z == 58.0, "the walk itself still happened"
        assert "gate" not in parts, f"a rung was paid from the air: {parts.get('gate')}"
        assert info["gates_reached"] == 1, "only the rung the load absorbed at the spawn"
        env.close()


def test_a_room_rung_is_credited_from_a_legal_hop_and_from_the_floor():
    """The control the test above needs: the rule is about how far the ground is, not about being airborne.
    A hop 3 m over the floor pays every rung, exactly as walking does."""
    with tempfile.TemporaryDirectory() as tmp:
        walked: dict[str, float] = {}
        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=29)
        env.reset(seed=0)
        for _ in range(29):
            _, _, _, _, info = env.step(forward())
            add_parts(walked, info)
        env.close()

        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=29)
        env.reset(seed=0)
        fake.airborne, fake.ground_center = True, 3.0
        hopped: dict[str, float] = {}
        for _ in range(29):
            _, _, _, _, info = env.step(forward())
            add_parts(hopped, info)
        assert hopped["gate"] == walked["gate"] == 45.0, f"{hopped.get('gate')} vs {walked.get('gate')}"
        assert info["gates_reached"] == 4
        env.close()


def test_a_gate_is_credited_from_the_air_over_nothing():
    """The other control, and the one that keeps this change off the 18 levels layer 1 routes: the ground rule
    is for ROOM CENTROIDS only. The identical flight over the corridor's GATE ladder pays every gate."""
    env, fake = make_env()
    env.reset(seed=0)
    fake.airborne, fake.ground_center = True, 12.0
    parts: dict[str, float] = {}
    for _ in range(29):
        _, _, terminated, truncated, info = env.step(forward())
        add_parts(parts, info)
        if terminated or truncated:
            break
    assert parts["gate"] == 45.0, f"the gate ladder was affected by the ground rule: {parts.get('gate')}"
    env.close()


def test_route_ground_m_zero_restores_the_pre_rule_behaviour_through_the_env():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=29, route_ground_m=0.0)
        env.reset(seed=0)
        fake.airborne, fake.ground_center = True, 12.0
        parts: dict[str, float] = {}
        for _ in range(29):
            _, _, _, _, info = env.step(forward())
            add_parts(parts, info)
        assert parts["gate"] == 45.0 and info["gates_reached"] == 4
        env.close()


def test_route_fallback_is_not_used_when_gates_are_good():
    """Layer 1 wins, and the trunk is not consulted at all -- the byte-for-byte claim, through the env.

    `route_reads` counts every call that hands the trunk out as the ladder, so 0 means `_gates()` never left
    its success path. (The FILE is opened once, when the env is built: knowing whether a level has one is the
    only way to know which layer applies. What must never happen is the trunk being READ as a route while a
    gate ladder exists, which is what this pins.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, ordered=True, fresh_start_prob=0.0, max_steps=27)
        assert env.gates._route is not None, "the level does ship a file, so the guard is being tested"
        _, info = env.reset(seed=0)
        assert info["route_source"] == 1 and info["route_source_name"] == "gates"
        parts: dict[str, float] = {}
        targets = set()
        for _ in range(27):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            targets.add(env.gates.target["key"])
        assert targets <= {key for key, _, _ in GATES} | {"exit"}, f"a rung became the target: {targets}"
        assert parts["gate"] == 45.0 and info["gates_reached"] == 3  # 0-1's own numbers, unchanged
        assert env.gates.route_reads == 0, "the trunk was read while the gate ladder was working"
        assert env.gates._hops_source == "gates" and info["route_source"] == 1
        env.close()


def test_route_seeding_absorbs_the_nearest_rung():
    """§6's seeding rule: the rung the player STARTS at is absorbed, and the target is the one below it.

    The trunk is a total order over the level, so its nearest rung can be behind the player -- measured at
    0-1's own spawn, where the nearest room rung is 31.6 m back and the route's next rung 34.4 m ahead. Here
    the nearest rung is 14 m BEHIND: without the rule the observation would point the agent at the room it has
    already left, which is the "a wrong route is worse than none" case. Absorbing pays nothing, so the rule
    cannot be farmed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, rungs=BEHIND_RUNGS, fresh_start_prob=0.0, max_steps=30)
        _, info = env.reset(seed=0)
        assert env.gates.best_hops == 2 and env.gates.paid_hops == 2, "the rung behind is absorbed"
        assert env.gates.target["key"] == "0,1,20", "and the target is the rung below it, 20 m AHEAD"
        # Absorbing is `mark_paid` semantics in ALL FIVE places, not just the two hop floors: the rung counts as
        # reached, its hop value counts, and it can never pay a fallback instalment. Marking only the floors left
        # it looking unreached, and it is by construction the NEAREST rung, so it satisfied `_nearer_unreached`
        # from the first decision and made the rung AHEAD parkable -- see the test below.
        assert env.gates.reached == {"0,1,-14"} and env.gates.hops_reached == {2}
        assert env.gates.paid_fallback == {"0,1,-14"}
        parts: dict[str, float] = {}
        for _ in range(30):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert parts["gate"] == 30.0, "the two rungs ahead pay; the absorbed one never does"
        assert info["gates_reached"] == 3, \
            "the absorbed rung counts, exactly as it would on a spawn that happened to fall inside its cylinder"
        env.close()

    # Out of range: the rule is a tolerance, not a licence. Below the distance to the nearest rung nothing is
    # absorbed and the level gets today's behaviour -- which on this trunk is the bug itself, the target sitting
    # on the room the player has already left. Note what the rule does and does not buy: the same instalments
    # either way (the rung behind is never reachable, so it never pays), but the TARGET, and with it
    # observation slots 448-455 and look mode 2, points forward from the first decision instead of backward.
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, rungs=BEHIND_RUNGS, fresh_start_prob=0.0, max_steps=30, route_seed_m=2.0)
        env.reset(seed=0)
        assert env.gates.best_hops is None
        assert env.gates.target["key"] == "0,1,-14", "unseeded, the trunk aims at the rung 14 m behind"
        parts = {}
        for _ in range(30):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert parts["gate"] == 30.0 and info["gates_reached"] == 2
        env.close()


def test_a_seeded_load_that_stalls_at_the_spawn_keeps_aiming_forward():
    """A fresh policy on an unseen level does not close on the first rung, and one patience window is short.

    The seeding rule's whole job is to stop the trunk aiming at the room the player has already left. Absorbing
    only `best_hops`/`paid_hops` left the absorbed rung out of `reached`, and it is by construction the NEAREST
    rung -- so `_nearer_unreached` found it strictly nearer than the forward target on the very first decision,
    which is exactly the "a park is only ever a SWITCH" guard being satisfied. The rung AHEAD then became
    parkable, `_pick`'s fallback chose the absorbed rung BEHIND, and `_pay_fallback` paid an instalment for
    walking back to it: the failure the rule exists to prevent, plus a payment `_seed_start`'s docstring says is
    impossible. Reproduced at the shipped `gate_target_patience_s` 20.0 (300 decisions): parked at decision 299.

    Held here at 1.0 s so the window is 15 decisions, and the agent does nothing at all for longer than that.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, rungs=BEHIND_RUNGS, fresh_start_prob=0.0, max_steps=200,
                                 gate_target_patience_s=1.0)
        env.reset(seed=0)
        assert env.gates.patience_steps == 15 and env.gates.target["key"] == "0,1,20"
        for _ in range(17):  # the whole window and two decisions more, standing still at the spawn
            _, _, terminated, truncated, info = env.step(idle())
            assert not terminated and not truncated
        assert env.gates.parked == set() and info["targets_parked"] == 0, \
            "nothing to switch to: the rung behind is absorbed, so it is not an unreached candidate"
        assert env.gates.target["key"] == "0,1,20" and env.gates._fallback is False, \
            "the target is still the rung AHEAD, chosen by the ladder"
        parts: dict[str, float] = {}
        for _ in range(60):  # and the forward walk still pays the trunk's depth, no more and no less
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert parts["gate"] == 30.0, f"depth-1 over a 3-rung trunk: {parts.get('gate')}"
        env.close()


def test_route_respawn_mid_ladder_targets_forward():
    """A checkpoint respawn keeps the ladder and keeps aiming forward, and does not re-seed.

    `best_hops` is level-load scoped and `mark_paid` never clears it, so the seeding rule fires once per LOAD,
    not per respawn (spec §6 corrects revision 1 here). The respawn point is 6 m from the rung behind it and
    16 m from the rung ahead, so a rule that re-ran on every respawn would absorb the rung behind and could
    keep re-absorbing its way down the trunk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, fresh_start_prob=0.0, max_steps=13)
        env.reset(seed=0)
        parts: dict[str, float] = {}
        for _ in range(13):  # z 26: past the checkpoint at 20 and the rung at 22
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert truncated and env.gates.best_hops == 2 and parts["gate"] == 15.0
        _, info = env.reset()  # respawn at z 20
        assert info["fresh_start"] == 0 and fake.z == 20.0
        assert env.gates.best_hops == 2 and env.gates.paid_hops == 2, "the ladder survives the respawn"
        assert env.gates.target["key"] == "0,1,38", "the rung below, not the 2 m-away rung behind"
        _, info = env.reset()  # and a second respawn cannot walk the absorption down the trunk either
        assert env.gates.best_hops == 2 and env.gates.target["key"] == "0,1,38"
        env.close()


def test_route_reload_without_checkpoint_reseeds():
    """The `_respawn` bug of spec §6: a death with NO checkpoint reloads the level, so the ladder must restart.

    `StatsManager.Restart` reloads the whole level when there is no checkpoint yet, and `_respawn` was calling
    `mark_paid` on that branch too -- leaving `best_hops` at whatever the dead attempt reached, pointing the
    target at a rung far ahead of a player standing at the spawn, and making `gate` unpayable for the rest of
    the load. It exists on every gate ladder today; a 13-rung trunk makes it bite much harder.

    The other half of the fix, and the reason for `keep_paid`: the LADDER restarts, the PAYMENTS do not. This
    episode is still running, and `gate` is "once per new lower rung", so the prefix already walked must not
    become re-earnable -- see `test_a_reload_inside_one_episode_cannot_re_earn_the_prefix`. `paid_hops` stays 2
    while `best_hops` goes back to 3, which is what lets the re-walk pay for the rungs BELOW 2 and nothing else.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, max_steps=60)
        env.reset(seed=0)
        assert env.gates.best_hops == 3, "the spawn stands inside the first rung's own cylinder"
        parts: dict[str, float] = {}
        for _ in range(7):  # z 14: the rung at 22 is reached (8 m), and the checkpoint at 20 is not
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert env.gates.best_hops == 2 and parts["gate"] == 15.0 and fake.checkpoint is False
        approach_before = parts["gate_approach"]
        fake.kill_next = True
        _, _, terminated, truncated, info = env.step(idle())
        assert not terminated and not truncated and fake.z == 0.0, "no checkpoint, so the level reloaded"
        assert env.gates.best_hops == 3, \
            "the ladder restarted at the spawn; carrying 2 would leave `gate` unpayable for the rest of the load"
        assert env.gates.paid_hops == 2, "but the payment record did not: the episode is still running"
        assert env.gates.reached == {"0,1,6"} and env.gates.gates_reached == 1
        assert env.gates.target["key"] == "0,1,22", "and the target is the rung ahead of the spawn again"
        for _ in range(7):  # back over the identical 14 m, toward the identical target
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert parts["gate_approach"] == approach_before, \
            "re-walking to the pre-death best pays no approach either (ruling R1): `best_dist` stood"
        assert parts["gate"] == 15.0, "and reaching the rung at 22 a second time pays no instalment"
        for _ in range(24):  # on past it: ground the dead attempt never covered, which does pay
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete"
        assert parts["gate"] == 45.0, \
            f"the trunk's own depth over the whole load, no more: {parts['gate']}"
        assert parts["gate_approach"] > approach_before, "the two rungs below the pre-death best are new ground"
        env.close()


def test_a_reload_inside_one_episode_cannot_re_earn_the_prefix():
    """A death with no checkpoint restarts the ladder; it must not restart the LADDER'S INCOME.

    `_respawn`'s reload branch calls `new_level_load`, which is the honest call for the targeting bug above --
    the level really did load again. But `new_level_load` also clears `paid_hops`, `paid_fallback` and
    `best_dist`, and `_respawn` runs INSIDE a continuing episode. Unguarded, that makes `gate` and
    `gate_approach` re-earnable once per death for ground already covered, which is exactly what `_respawn`
    refuses for `MilestoneTracker` two lines above ("re-paying them would make 'die with no checkpoint' a way to
    earn `checkpoint` and `arena_clear` again for ground already covered -- a farm"). The re-walk costing the
    same walk is no defence: a checkpoint re-walk costs the same walk too.

    Three rungs all placed BEFORE FakeLevel's checkpoint at z 20, so no checkpoint ever activates and every
    death takes the reload branch. Measured before the fix: 30 / 30 / 30 per lap, an unbounded stream inside one
    12 000-step episode against a `death` of 5 -- and worse, reaching the level's first checkpoint would move
    every later death onto the `mark_paid` branch and end the stream for good, so the gradient pointed away from
    the first checkpoint. The same three laps run on a pure GATES level (`route=None`, layer 1), because this
    branch is not gated on the route layer and the live run trains on 0-1/0-3/0-4 today.
    """
    early = ((0.0, 1.0, 2.0, 2), (0.0, 1.0, 12.0, 1), (0.0, 1.0, 18.0, 0))

    def three_laps(env, fake) -> list[float]:
        per_lap = []
        for _ in range(3):
            parts: dict[str, float] = {}
            while env._raw["player"]["pos"][2] < 18.0:
                _, _, _, _, info = env.step(forward())
                add_parts(parts, info)
            fake.kill_next = True
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            assert not terminated and not truncated and fake.z == 0.0, "the level reloaded, the episode stands"
            per_lap.append(round(parts.get("gate", 0.0), 1))
        return per_lap

    with tempfile.TemporaryDirectory() as tmp:  # layer 2, the room trunk
        env, fake, _ = route_env(tmp, rungs=early, fresh_start_prob=0.0, max_steps=4000)
        env.reset(seed=0)
        assert env.gates.route_source == ROUTE_SOURCE_ROOMS
        assert three_laps(env, fake) == [30.0, 0.0, 0.0], "the trunk's depth once, not once per death"
        env.close()

    env, fake = make_env(fresh_start_prob=0.0, max_steps=4000)  # layer 1, the live gate ladder
    fake.gates = tuple((f"0,1,{round(z)}", (x, y, z), h) for x, y, z, h in early)
    fake._load()
    env.reset(seed=0)
    assert env.gates._route is None and env.gates.route_source == ROUTE_SOURCE_GATES
    assert three_laps(env, fake) == [30.0, 0.0, 0.0], "and a gates level behaves identically: A3 is unaffected"
    env.close()


def test_a_respawn_frame_with_no_campaign_block_does_not_wipe_the_ladder():
    """The `_respawn` reload branch is gated on the block being PRESENT, not just on "no checkpoint".

    The mod omits the whole `campaign` block for any step whose build throws (docs/protocol.md), and an absent
    block has no `checkpoints` either -- which reads as "no checkpoint yet", i.e. as a level reload. Treating
    that as a reload would clear a ladder that was fine. Reading it as an ordinary respawn is the safe
    direction and self-corrects at the next episode boundary, where `choose_fresh_start` sees no checkpoint and
    forces a fresh load anyway.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, max_steps=60)
        env.reset(seed=0)
        for _ in range(11):  # z 22: past the checkpoint at 20 and the rung at 22
            env.step(forward())
        assert env.gates.best_hops == 2 and fake.checkpoint
        fake.kill_next = True
        fake.drop_campaign_steps = 2  # the death frame and the respawn frame both arrive without a block
        _, _, terminated, truncated, info = env.step(idle())
        assert not terminated and not truncated and fake.z == 20.0, "a real checkpoint respawn happened"
        assert env.gates.best_hops == 2 and env.gates.paid_hops == 2, "the ladder survived the blind frame"
        assert env.gates.reached == {"0,1,6", "0,1,22"}
        env.close()


def test_route_rejected_when_exit_moved():
    """Guard I5: the trunk is offline data, so a FinalPit that has moved disables the file for that load.

    And it really disables it. The first frame here carries no `campaign` block at all -- a mid-load frame,
    which the mod does send -- so the trunk is used unvalidated and a rung becomes the target; the next frame
    reports an exit 20 m from the file's, and the stale target has to be dropped rather than held while
    `gate_approach` keeps paying toward it (the defect the reviewer found in revision 1).
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, exit_pos=(0.0, 1.0, EXIT_Z + 20.0), max_steps=40)
        fake.drop_campaign_steps = 1  # the reset frame: no block, so the check cannot be answered yet
        env.reset(seed=0)
        assert env.gates.target is not None and env.gates._route_ok is None
        parts: dict[str, float] = {}
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert env.gates._route_ok is False and env.gates.target is None
        assert "gate" not in parts and "gate_approach" not in parts, parts
        assert info["gates_reached"] == 0 and info["route_source"] == 0
        assert info["route_source_name"] == "none", "the level fell all the way to the exit vector"
        assert info["end_reason"] == "level_complete", "and the level is still perfectly playable"
        env.close()


def test_route_source_flip_clears_the_ladder():
    """A mid-load flip between the two layers must not pay the hop difference between their scales.

    The two ladders are different measures of the same level -- 0-5's trunk is 5 rungs deep where its door
    graph reports nothing at all -- so carrying `paid_hops` from one to the other would pay
    `paid_hops - best_hops` instalments for a change of units. Here the trunk has paid down to hops 2 when the
    gate graph comes good with a hops 0 door within reach: the guard forgets the ladder and the new scale
    starts from its own first instalment.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, max_steps=40)
        fake.gates = (("0,1,15", (0.0, 1.0, 15.0), 0), ("0,1,35", (0.0, 1.0, 35.0), 0))
        env.reset(seed=0)
        parts: dict[str, float] = {}
        for _ in range(12):  # z 24: rung 3 absorbed at the spawn, rung 2 reached and paid
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
        assert env.gates._hops_source == "rooms" and env.gates.best_hops == 2 and parts["gate"] == 15.0
        reads = env.gates.route_reads
        fake.gates_ordered = True  # the mod's Scan() finds the goal room as rooms light up
        after: dict[str, float] = {}
        for _ in range(5):  # the flip, and on to the hops 0 gate at z 35
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(after, info)
        assert env.gates._hops_source == "gates" and info["route_source"] == 1
        assert env.gates.reached == {"0,1,35"}, "the trunk's reached set went with its hop scale"
        assert after["gate"] == 15.0, \
            "one instalment -- the new ladder's own first rung. Carrying paid_hops 2 into a hops 0 gate " \
            f"would have paid two, for a change of units: {after['gate']}"
        assert env.gates.route_reads == reads, "and the trunk is not read again once the gates work"
        env.close()


def test_route_absent_is_todays_behaviour():
    """The 21 levels with no file, and the `route_fallback: false` switch: no target, no payment, no change."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "no-routes"
        empty.mkdir()
        _read_route.cache_clear()
        assert load_route(LEVEL, str(empty)) is None and load_route(LEVEL, str(empty / "missing")) is None
        env, fake = make_env(route_dir=str(empty))
        fake.enable_route()
        obs, info = env.reset(seed=0)
        assert env.gates._route is None and info["route_source"] == 0
        parts: dict[str, float] = {}
        for _ in range(35):
            obs, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert info["end_reason"] == "level_complete" and info["gates_reached"] == 0
        assert "gate" not in parts and "gate_approach" not in parts
        assert not obs[GATE_BLOCK : GATE_BLOCK + 8].any(), "the target slots stay empty, as they are today"
        assert env.gates.route_reads == 0 and env.gates.route_source == 0
        env.close()

    # The same level WITH a file, but the fallback switched off in the config: the file is never even read.
    with tempfile.TemporaryDirectory() as tmp:
        route_dir = write_route(Path(tmp) / "routes")
        env, fake = make_env(route_dir=str(route_dir), route_fallback=False)
        fake.enable_route()
        env.reset(seed=0)
        assert env.gates._route is None
        parts = {}
        for _ in range(35):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(parts, info)
            if terminated or truncated:
                break
        assert "gate" not in parts and info["route_source"] == 0
        env.close()


def test_a_route_rung_goes_through_the_same_patience_and_parking_machinery():
    """A rung is shaped like a gate, so the ladder-patience spec applies to it unchanged -- and must.

    A trunk is offline data with no live check that a leg is walkable (spec §2.8), so the mechanism that
    rescues a collapsed gate ladder is the same one that rescues a bad rung: the target is parked after
    `patience_steps` without getting closer, and the fallback hands over to the nearest unreached, unparked
    rung at any hop count.

    And the two payment rules stay mutually exclusive on a trunk, which is the "no double payment" claim. The
    fallback rung here is BELOW the floor (`hops` 0 against `best_hops` 2), so `_pay_fallback` declines it and
    the LADDER pays the two hop values the skip crossed -- the same telescoping a gate ladder does for a
    shortcut. `_pay_fallback` only ever pays a rung at or above the floor, which the ladder by construction
    cannot pay for, so no rung is paid twice and the load's whole `gate` income is still the trunk's depth.

    `gate_patience_mode: "always"`, because a trunk is a total order and the collapse detector therefore calls
    it healthy -- correctly: a trunk's nearest rung at the spawn IS its top rung. That is the same structural
    fact as `test_the_patience_rule_is_inert_on_a_walkable_trunk` below, and it is why the route spec's detector
    for a bad rung is `route_source` 2 with `gates_reached` stuck rather than `targets_parked`. What this test
    pins is that a rung goes through the machinery unchanged WHEN the machinery is switched on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Rung hops 1 is 200 m straight up -- 0-3's shape, on a trunk. The rung below it is on the floor.
        rungs = ((0.0, 1.0, 6.0, 3), (0.0, 1.0, 22.0, 2), (0.0, 201.0, 30.0, 1), (0.0, 1.0, 54.0, 0))
        env, fake, _ = route_env(tmp, rungs=rungs, max_steps=1200, stuck_seconds=1000.0,
                                 gate_patience_mode="always")
        env.reset(seed=0)
        for _ in range(12):  # z 24: rung 2 reached, so the ladder points at the unreachable rung 1
            env.step(forward())
        assert env.gates.target["key"] == "0,201,30" and env.gates.best_hops == 2
        paid = 0.0
        for _ in range(env.gates.patience_steps + 2):
            _, _, terminated, truncated, info = env.step(idle())
            paid += info["reward_parts"].get("gate", 0.0)
            if terminated or truncated:
                break
        assert env.gates.parked == {"0,201,30"} and info["targets_parked"] == 1
        assert env.gates.target["key"] == "0,1,54", "the nearest unreached, unparked rung at any hop count"
        assert paid == 0.0, "parking is a switch, not a payment"
        for _ in range(20):  # walk to the fallback rung, which is two hop values below the floor
            _, _, terminated, truncated, info = env.step(forward())
            paid += info["reward_parts"].get("gate", 0.0)
            if terminated or truncated:
                break
        assert paid == 30.0, f"the ladder pays the two hop values the skip crossed, and only it: {paid}"
        assert env.gates.best_hops == 0 and env.gates.paid_hops == 0
        assert "0,1,54" not in env.gates.paid_fallback, \
            "the fallback rule declined it (hops below the floor), so the ladder's payment is the only one"
        # The whole level load earned the trunk's depth and not one instalment more: rung 3 absorbed where the
        # player spawned inside its own cylinder, rung 2 on the way, then the two hop values the skip crossed.
        assert env.gates.paid_fallback == {"0,1,6"}, "absorbed by mark_paid at the load, never paid"
        env.close()


def test_the_patience_rule_is_inert_on_a_walkable_trunk():
    """And where the trunk IS walkable it changes nothing, which is the other half of the same claim.

    Worth pinning because a monotone trunk is the shape where the patience rule's known residual bites: its
    rung-below is usually also the nearest unreached rung, so `_nearer_unreached` finds nothing nearer and a
    ladder pick is never parked. That makes parking nearly inert on a healthy trunk -- and means the detector
    for a bad rung is `route_source` 2 with `gates_reached` stuck and `targets_parked` 0, exactly as the
    ladder-patience spec's deviation 2 records. The cure is the data (`"rungs": []`), never a knob.

    Forced on with "always", so the inertness is the RULE's and not the detector's: under the shipped default a
    trunk reads healthy and parking never runs here at all, which is a second reason for the same answer.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, _ = route_env(tmp, max_steps=40, gate_patience_mode="always")
        env.reset(seed=0)
        assert env.gates.ladder_collapsed is False and env.gates.patience_active, "forced on over a False verdict"
        on: dict[str, float] = {}
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(on, info)
            if terminated or truncated:
                break
        assert info["targets_parked"] == 0 and info["end_reason"] == "level_complete"
        env.close()
    with tempfile.TemporaryDirectory() as tmp:  # the same walk with parking turned off entirely
        env, fake, _ = route_env(tmp, max_steps=40, gate_target_patience_s=0.0)
        env.reset(seed=0)
        off: dict[str, float] = {}
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(forward())
            add_parts(off, info)
            if terminated or truncated:
                break
        assert env.gates.patience_steps == 0
        env.close()
    assert on["gate"] == off["gate"] == 45.0
    assert abs(on["gate_approach"] - off["gate_approach"]) < 1e-9, (on["gate_approach"], off["gate_approach"])


def test_a_route_file_the_loader_refuses_is_the_same_as_no_file():
    """Every rejection falls through to layer 3, because "no file" is the normal case on 21 of 33 levels."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "routes"
        cases = {
            "version": {"version": 3},                       # a newer emitter than this loader
            "level": {"level": "Level 0-3"},                 # a file renamed onto the wrong level
            "rungs": {"rungs": []},                          # §12.4's recovery path: disable one level by hand
            "two rungs": {"rungs": [{"key": "0,1,6", "pos": [0.0, 1.0, 6.0], "hops": 1},
                                    {"key": "0,1,22", "pos": [0.0, 1.0, 22.0], "hops": 0}]},  # R1
            "no exit": {"exit": {"target": "Level 0-2"}},     # guard I5 could never be evaluated
            "bad pos": {"rungs": [{"key": "0,1,6", "pos": [0.0, 1.0], "hops": 2},
                                  {"key": "0,1,22", "pos": [0.0, None, 22.0], "hops": 1},
                                  {"key": "0,1,38", "pos": [0.0, 1.0, float("nan")], "hops": 0}]},
        }
        for name, override in cases.items():
            write_route(base, **override)
            assert load_route(LEVEL, str(base)) is None, name
        # Not JSON at all, and a file that is JSON but not an object.
        for text in ("{not json", "[]", "null"):
            route_path(LEVEL, str(base)).write_text(text, encoding="utf-8")
            _read_route.cache_clear()
            assert load_route(LEVEL, str(base)) is None, text
        # And the TYPE-level breakage, which the cases above never reached: each of these is valid JSON that
        # used to raise straight out of the loader. `load_route` runs in `UltrakillEnv.__init__` and in
        # `_switch_level` -> `_campaign_reset` -> `reset()`, neither of which catches anything, and SB3's
        # SubprocVecEnv worker does not either, so on a 30-level curriculum one bad file would kill a worker
        # the first time any env sampled that level -- hours into a run. Hand-editing a route file is the
        # documented recovery path for a bad rung (spec §12.4), so a typo lands exactly here.
        write_route(base)  # `rungs` is a parameter of the fixture, so a scalar one is written by hand
        good = json.loads(route_path(LEVEL, str(base)).read_text(encoding="utf-8"))
        for scalar in (5, True, 3.5, "three"):  # each is truthy and not iterable: `or ()` never saw it
            good["rungs"] = scalar
            route_path(LEVEL, str(base)).write_text(json.dumps(good), encoding="utf-8")
            _read_route.cache_clear()
            assert load_route(LEVEL, str(base)) is None, f"rungs is {scalar!r}"
        # `1e400` parses to inf (json.dumps writes it as `Infinity`, json.loads reads it back), and int()
        # refuses inf with OverflowError -- not the ValueError the clause caught.
        write_route(base, rungs=[{"key": f"0,1,{z}", "pos": [0.0, 1.0, float(z)], "hops": float("inf"),
                                  "name": "r", "gated_by": [], "open": False, "locked": False, "active": True}
                                 for z in (6, 22, 38, 54)])
        assert load_route(LEVEL, str(base)) is None, "every rung unreadable leaves fewer than ROUTE_MIN_RUNGS"


def test_the_loader_never_raises_out_of_env_step():
    """The backstop: any shape `_parse_route` did not anticipate is a layer-3 refusal, never an exception.

    Every case above is a rejection the loader names explicitly; this pins the property they add up to, which
    is the one `env.step` depends on. `_parse_route` is stubbed to raise because, with the type checks in
    place, no file can reach the backstop any more -- which is the point: the guard is for the shape nobody
    thought of. It prints one line (the only loud rejection there is) and `lru_cache` holds the None, so a
    training run sees it once per scene per process rather than once per level load.
    """
    import ultrakill_ai.campaign as campaign_mod
    with tempfile.TemporaryDirectory() as tmp:
        base = write_route(Path(tmp) / "routes")
        original = campaign_mod._parse_route
        campaign_mod._parse_route = lambda scene, doc: (_ for _ in ()).throw(RuntimeError("unanticipated shape"))
        try:
            _read_route.cache_clear()
            assert load_route(LEVEL, str(base)) is None, "the loader swallowed it and fell through to layer 3"
        finally:
            campaign_mod._parse_route = original
            _read_route.cache_clear()
        assert load_route(LEVEL, str(base)) is not None, "and the good file still loads once the stub is gone"


def test_the_loader_normalises_the_fields_the_reach_test_reads():
    """`open`, `locked`, `active` and `needs_item` are schema constants on a rung, not data.

    `_is_reached` DOUBLES its cylinder for an open gate (16 m x 12 m, which R4 never measured) and refuses one
    that carries `needs_item` however close the player stands, so a hand-edited file must not be able to widen
    the reach test or wedge a rung. The loader guarantees what the reach test assumes; `test_route_files.py`
    is what fails when a SHIPPED file disagrees.
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = write_route(Path(tmp) / "routes", rungs=ROUTE_RUNGS)
        doc = json.loads(route_path(LEVEL, str(base)).read_text(encoding="utf-8"))
        for rung in doc["rungs"]:
            rung.update(open=True, locked=True, active=False, needs_item="SkullBlue")
        route_path(LEVEL, str(base)).write_text(json.dumps(doc), encoding="utf-8")
        _read_route.cache_clear()
        route = load_route(LEVEL, str(base))
        assert route is not None
        for rung in route["rungs"]:
            assert rung["open"] is False and rung["locked"] is False and rung["active"] is True
            assert "needs_item" not in rung, "a static needs_item is a permanent wedge; S3 stamps it live"
        assert route["rungs"][0]["name"] == "1 - Room", "and everything else the file carries is kept"
        assert route["rungs"][0]["gated_by"] == []


def test_each_env_gets_its_own_copy_of_the_trunk():
    """`load_route` deep-copies, because a rung is handed straight out as `GateProgress.target`."""
    with tempfile.TemporaryDirectory() as tmp:
        base = write_route(Path(tmp) / "routes")
        one, two = load_route(LEVEL, str(base)), load_route(LEVEL, str(base))
        assert one == two and one is not two
        assert one["rungs"][0] is not two["rungs"][0] and one["exit_pos"] is not two["exit_pos"]
        one["rungs"][0]["hops"] = 99
        assert load_route(LEVEL, str(base))["rungs"][0]["hops"] == 3, "the cached document was not mutated"


# ---------------------------------------------------------------------------------------------
# S4: the skull-carry leg
# ---------------------------------------------------------------------------------------------

SKULL_REWARDS = RewardConfig(time=0.01, checkpoint=10.0, arena_clear=10.0, door_unlock=3.0, novelty=0.5,
                             gate=15.0, gate_approach=0.15, item_pickup=15.0, item_placed=15.0, punch=0.01)


def skull_env(*, skulls: dict | None = None, **overrides) -> tuple[UltrakillEnv, FakeLevel]:
    env, fake = make_env(rewards=SKULL_REWARDS, **overrides)
    fake.enable_skulls(**(skulls or {}))
    return env, fake


def walk(env, steps: int, *, punch: bool = False, parts: dict | None = None):
    """`steps` forward decisions, optionally pressing punch on every one of them."""
    info = None
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action(move=True, buttons=("punch",) if punch else ()))
        if parts is not None:
            add_parts(parts, info)
        if terminated or truncated:
            break
    return info


def test_the_skull_leg_pays_pickup_and_placement_once_each():
    """Fetch, carry, place: the whole leg, with punch held down the entire way as the live policy does.

    The gate at z 50 is a dead end without this -- `needs_item` is the only field that separates "kill eleven
    enemies" from "fetch a skull", and an agent that cannot tell them apart pays gate_approach to reach a door
    it can never pass.
    """
    env, fake = skull_env(max_steps=40)
    env.reset(seed=0)
    assert env.gates.target["subgoal"] == "item", "the locked gate sends us to its source first"
    parts: dict[str, float] = {}
    walk(env, 40, punch=True, parts=parts)
    env.close()
    assert fake.picked_up == 1 and fake.thrown == 0, "the carry survived a punch on every decision"
    assert fake.skull_in == ALTAR_KEY
    assert parts["item_pickup"] == 15.0 and parts["item_placed"] == 15.0
    assert parts["gate_approach"] > 0.0
    assert parts["gate"] == 15.0, "and the rung pays once, after the placement cleared the lock"


def test_walking_up_to_a_locked_gate_without_the_skull_pays_no_rung():
    """A door that cannot be passed is not 'reached', however close the agent stands.

    Paying the rung for arriving at a sealed door was worse than the stray +15: `best_hops` moved past the
    lock, so `_choose_target` dropped to the rung below -- which carries no `needs_item`, so `_subgoal` handed
    it straight back -- and the fetch/carry machine never fired again for the rest of the level load.
    """
    env, fake = skull_env(max_steps=40)
    env.reset(seed=0)
    parts: dict[str, float] = {}
    info = walk(env, 26, parts=parts)  # no punch, so the skull is never picked up and the lock never clears
    assert fake.skull_in == PEDESTAL_KEY and not fake.skull_held
    assert fake.z >= SKULL_GATE_POS[2], "the agent really did walk through where the gate stands"
    assert info["gates_reached"] == 0 and info["gate_hops_best"] is None
    assert "gate" not in parts, "the locked rung paid nothing"
    assert env.gates.target["subgoal"] == "item", "and the sub-goal still points at the skull"
    env.close()


def test_punch_at_a_solved_altar_is_dropped_so_the_gate_stays_open():
    """`AltHit`'s not-holding branch is ForceHold on whatever it hits, INCLUDING a skull resting in an altar,
    and `ForceHold` re-runs `CheckItem`, whose empty branch calls Close() on the doors that altar opened.

    Without this the puzzle is undone on the very next decision and the gate re-locks, which is the §8 check 2
    guarantee. The env drops only presses that can do nothing but harm; the policy still chooses to press.
    """
    env, fake = skull_env(max_steps=80)
    env.reset(seed=0)
    walk(env, 19, punch=True)  # picked up at z 16, carried, placed at z 38
    assert fake.skull_in == ALTAR_KEY and fake.picked_up == 1
    for _ in range(40):  # stand on the altar and hammer punch
        fake.z = ALTAR_POS[2]
        env.step(action(buttons=("punch",)))
        assert "punch" not in fake.last_action["buttons"]
    assert fake.skull_in == ALTAR_KEY and fake.picked_up == 1
    assert env._raw["campaign"]["gates"][0]["needs_item"] is None, "the gate stayed open"
    env.close()


def test_a_placement_pays_once_even_when_the_skull_is_pulled_back_out():
    """With the protection off the puzzle really can be undone and redone; the milestone keys stay spent."""
    env, fake = skull_env(max_steps=80, subgoal_punch_range_m=0.0)
    env.reset(seed=0)
    parts: dict[str, float] = {}
    walk(env, 10, punch=True, parts=parts)  # z 20: picked up
    assert fake.skull_held
    for z in (40.0,) * 12:  # teleport to the altar and hammer punch: in, out, in, out ...
        fake.z = z
        _, _, _, _, info = env.step(action(buttons=("punch",)))
        add_parts(parts, info)
    env.close()
    assert fake.picked_up > 1, "the fake level really did let it be pulled back out"
    assert parts["item_pickup"] == 15.0 and parts["item_placed"] == 15.0, "both keys are spent for this level load"


def test_punch_is_dropped_while_carrying_outside_range_and_kept_inside():
    """Punch held down the whole way: exactly the two useful presses survive, and nothing is ever thrown."""
    env, fake = skull_env(max_steps=40)
    env.reset(seed=0)
    picked = placed = carried_and_dropped = 0
    for _ in range(20):
        was_held = fake.skull_held
        env.step(action(move=True, buttons=("punch",)))
        kept = "punch" in fake.last_action["buttons"]
        if not was_held and fake.skull_held:
            picked += 1
            assert kept, "the pickup press is the one press that helps: it is never dropped"
        elif was_held and fake.skull_in == ALTAR_KEY:
            placed += 1
            assert kept, "nor is the placing press"
        elif was_held:
            assert not kept, f"a press while carrying, {abs(fake.z - ALTAR_POS[2]):.0f} m from the altar"
            carried_and_dropped += 1
    env.close()
    assert (picked, placed) == (1, 1) and carried_and_dropped >= 5
    assert fake.thrown == 0 and fake.picked_up == 1


def test_punch_passes_through_while_carrying_an_item_no_altar_wants():
    """Level 0-4's shape: a `CustomKey1` carryable and not one `ItemPlaceZone` in the level.

    The protection exists to stop a punch throwing an item that has somewhere to go. Where nothing accepts what
    is held there is nothing to protect, and the button is not free: dropping it costs the parry, the melee and
    the throw itself, for the whole carry, on a level 0-4's key is carried across. The first version tested
    `any(items[].held)`, so picking the key up silenced punch until the key was delivered -- and it could not
    even be thrown away to get the button back, because throwing is a punch.
    """
    env, fake = skull_env(max_steps=40, skulls=dict(altars=False))
    env.reset(seed=0)
    assert env._raw["campaign"]["altars"] == [] and env._raw["campaign"]["items"], "items, and no altars at all"
    assert "needs_item" not in env._raw["campaign"]["gates"][0], "so the gate is an ordinary walk-up door"
    assert "subgoal" not in env.gates.target, "and the forced look mode is inert: it needs a sub-goal target"
    kept = 0
    for _ in range(12):  # walks from z 0 to z 24, over the key at z 20, pressing punch every decision
        env.step(action(move=True, buttons=("punch",)))
        kept += "punch" in fake.last_action["buttons"]
    env.close()
    assert kept == 12, f"every press must reach the game; {12 - kept} were dropped"
    assert fake.picked_up >= 1 and fake.thrown >= 1, "it was picked up by punching and thrown by punching"


def test_punch_passes_through_while_carrying_an_item_the_altars_do_not_accept():
    """The test is the held item's TYPE, not "something is held": 1-1 carries two skull colours at once."""
    env, fake = skull_env(max_steps=60, skulls=dict(item_type="CustomKey1"))
    env.reset(seed=0)
    assert env._raw["campaign"]["gates"][0]["needs_item"] == ALTAR_ITEM, "the gate still wants a red skull"
    assert env.gates.target["key"] == SKULL_GATE_KEY and "subgoal" not in env.gates.target, \
        "no source of the wanted type exists, so the fetch machine never starts"
    walk(env, 8)  # z 16, within punch range of the loose key at z 20
    env.step(action(buttons=("punch",)))
    assert fake.skull_held and "punch" in fake.last_action["buttons"], "the pickup press reached the game"
    kept = 0
    for _ in range(6):
        env.step(action(move=True, buttons=("punch",)))
        kept += "punch" in fake.last_action["buttons"]
    env.close()
    assert kept == 6, "an unwanted carry never gates the button, however far from an altar it goes"


def test_a_carry_that_stalls_past_the_patience_window_keeps_its_punch_protection():
    """The interaction the patience mechanism must not break, end to end through the env.

    `_protect_carry` drops the punch button only while `gates.target` is the ALTAR sub-goal of a held item --
    one source of truth, so the button can never be taken away and left that way. Parking the GATE that leg
    serves deletes the leg, `_subgoal` never rebuilds it, and the protection silently lapses with the skull
    still held; the live policy presses punch on ~35% of decisions and `Punch.ActiveStart` turns one into a
    throw. On 1-1 that gate is the only route forward, so the level load is lost. `_tick_patience` therefore
    never parks a carry, nor any gate still reporting `needs_item`.

    Stalled here for twenty patience windows, 16 m short of the altar and punching on every decision.
    """
    env, fake = skull_env(max_steps=700, gate_target_patience_s=2.0)  # 30 decisions per window at 30 fps / 2
    assert env.gates.patience_steps == 30
    env.reset(seed=0)
    walk(env, 9, punch=True)  # z 18: the skull was picked up at z 16 and is being carried
    assert fake.skull_held and fake.picked_up == 1 and fake.thrown == 0
    for _ in range(600):  # stand still, punching, for twenty windows
        env.step(action(buttons=("punch",)))
    env.close()
    assert fake.thrown == 0 and fake.skull_held, "the skull is still in the player's hands"
    assert env.gates.parks == 0 and not env.gates.parked, "and the gate its altar opens was never parked"
    assert env.gates.target["subgoal"] == "altar", "so the carry leg is still the target"


def test_carry_protection_can_be_turned_off_and_then_the_punch_throws_it():
    env, fake = skull_env(max_steps=40, subgoal_punch_range_m=0.0)
    env.reset(seed=0)
    walk(env, 10)  # z 20, standing on the pedestal, nothing punched yet
    assert not fake.skull_held
    env.step(action(buttons=("punch",)))
    assert fake.skull_held and fake.picked_up == 1
    env.step(action(move=True, buttons=("punch",)))  # z 22, nowhere near the altar
    assert "punch" in fake.last_action["buttons"] and fake.thrown == 1 and not fake.skull_held
    env.close()


def test_throwing_the_skull_retargets_and_pays_no_second_approach():
    """Pick up, throw, pick up again: each leg's approach budget is seeded once an episode and never re-seeded."""
    env, fake = skull_env(max_steps=60, subgoal_punch_range_m=0.0)
    env.reset(seed=0)
    parts: dict[str, float] = {}
    walk(env, 10, parts=parts)  # z 20, on the pedestal
    _, _, _, _, info = env.step(action(buttons=("punch",)))
    add_parts(parts, info)
    assert fake.skull_held and env.gates.target["key"].startswith("altar:")
    walk(env, 6, parts=parts)  # z 32, closing on the altar
    earned = parts["gate_approach"]
    _, _, _, _, info = env.step(action(buttons=("punch",)))  # throw it at our feet
    add_parts(parts, info)
    assert fake.thrown == 1 and env.gates.target["key"] == "item:SkullRed", "the machine drops back to fetch"
    for _ in range(4):  # stand on it, pick it up again, drop it again
        _, _, _, _, info = env.step(action(buttons=("punch",)))
        add_parts(parts, info)
    env.close()
    assert fake.picked_up >= 2 and parts["gate_approach"] == earned, "shuttling earned nothing"
    assert parts["item_pickup"] == 15.0


def test_dying_while_holding_the_skull_cannot_pay_the_pickup_twice():
    """A checkpoint respawn re-instantiates the room, so the skull comes back under a brand-new key."""
    env, fake = skull_env(max_steps=80)
    env.reset(seed=0)
    parts: dict[str, float] = {}
    walk(env, 11, punch=True, parts=parts)  # z 22: checkpoint reached and the skull picked up
    assert fake.skull_held and parts["item_pickup"] == 15.0
    fake.kill_next = True
    _, _, terminated, truncated, info = env.step(idle())
    add_parts(parts, info)
    assert not terminated and not truncated and info["deaths"] == 1
    assert fake.skull_key == "skull#1" and not fake.skull_held, "a fresh instance back on its pedestal"
    walk(env, 4, punch=True, parts=parts)
    env.close()
    assert fake.picked_up == 2, "it really was picked up a second time"
    assert parts["item_pickup"] == 15.0, "type-keyed, so a re-instantiated skull cannot re-pay"


def test_look_mode_2_is_forced_within_punch_range_of_a_subgoal():
    env, fake = skull_env(max_steps=40)
    env.reset(seed=0)
    fake.yaw = 90.0  # facing +x, so the sub-goal down the corridor is a left turn
    for _ in range(8):  # decisions taken from z 0..14, all more than 4 m from the pedestal skull at z 20
        env.step(action(move=True, yaw=30.0))
    assert fake.last_action["look"] == [30.0, 0.0], "look mode 0: the sampled bins drove the camera"
    _, _, _, _, info = env.step(action(move=True, yaw=30.0))  # taken from z 16, inside 4 m
    assert fake.last_action["look"][0] < 0.0, "the env took the camera regardless of the sampled look mode"
    assert info["look_gate_frac"] > 0.0
    env.close()


def test_the_carry_leg_aims_at_the_altars_collider_centre():
    """F2. A placement punch has to hit the zone's own collider, and the position the block reports sits 0.4 m
    under its lid with no margin. The mod's `aim_pos` is the collider centre; without it the measured 1 m drop
    is used. Both the aim and the range test have to use it, or the agent stands at a spot the punch cannot
    reach and points over the top of the altar from there.
    """
    env, fake = skull_env(max_steps=40)
    env.reset(seed=0)
    walk(env, 10, punch=True)  # picked up on the way past the pedestal
    assert fake.skull_held
    target = env.gates.target
    assert target["subgoal"] == "altar" and target["pos"] == list(ALTAR_AIM_POS), "the collider centre, not pos"
    env.close()

    old_mod, fake = skull_env(max_steps=40, skulls=dict())
    fake.altar_aim_pos = False  # a mod that sends no aim_pos at all
    old_mod.reset(seed=0)
    walk(old_mod, 10, punch=True)
    assert old_mod.gates.target["pos"] == [ALTAR_POS[0], ALTAR_POS[1] - 1.0, ALTAR_POS[2]], "the fallback drop"
    old_mod.close()


def test_an_off_ladder_altar_never_takes_the_punch_button_away():
    """F4, Level 0-2's trap. Its blue skull sits on the main route; the only zone that accepts it is behind the
    `altar_only` gate `-60,-6,236`, a SECRET ARENA whose `hops` is null because it is off the exit chain (its
    one activated room `-60,-11,236` appears nowhere in 0-2's pit chain). `_choose_target` skips `hops is None`
    gates, so no sub-goal can ever exist for it.

    The old key was `campaign.wanting_altars`, which does not look at gates at all: picking the skull up dropped
    the punch button, and `_near_subgoal` could never give it back, because releasing needs a sub-goal target
    that 0-2 cannot produce. Punch is parry, melee and the throw that would have got rid of the skull, so it was
    gone for the rest of the level. Keying both halves on the sub-goal makes that unrepresentable.
    """
    env, fake = skull_env(max_steps=60)
    fake.skull_gate_offladder = True
    env.reset(seed=0)
    assert env.gates.target is None, "no targetable gate: 0-2 has nothing to build a sub-goal from"
    assert env._raw["campaign"]["altars"], "and yet the level does ship an altar that accepts the skull"

    kept = 0
    for _ in range(14):  # walks z 0 -> 28, over the skull at z 20, pressing punch on every decision
        env.step(action(move=True, buttons=("punch",)))
        kept += "punch" in fake.last_action["buttons"]
    env.close()
    assert kept == 14, "every press reached the game: nothing was ever protected, so nothing can be stuck"
    assert fake.picked_up >= 1 and fake.thrown >= 1, "picked up by punching and thrown away by punching"


def test_a_subgoal_target_packs_through_the_unchanged_479_observation():
    env, fake = skull_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (479,) and env.observation_space.contains(obs)
    assert env.gates.target["subgoal"] == "item"
    assert obs[GATE_BLOCK + 4] == 1.0  # target mask
    assert abs(obs[GATE_BLOCK + 2] - PEDESTAL_POS[2] / 50.0) < 1e-6, "the gate scales, via the inherited hops"
    assert obs[GATE_BLOCK + 5] == 0.0 and obs[GATE_BLOCK + 6] == 0.0, "open/locked stay 0 with target_kind_slots off"
    assert obs[GATE_BLOCK + 7] == 0.0  # the gate's own hops (0), inherited
    env.close()

    kinds, _ = make_env(rewards=SKULL_REWARDS, target_kind_slots=True)
    kinds.client.enable_skulls()
    obs, _ = kinds.reset(seed=0)
    assert obs[GATE_BLOCK + 5] == 1.0 and obs[GATE_BLOCK + 6] == 0.0, "the escape hatch: 'the target is an item'"
    kinds.close()


def test_old_mod_without_altars_still_runs():
    """New Python against a 0.6.x mod: no altars, no items, no needs_item. Byte-identical to today."""
    env, fake = skull_env(max_steps=40)
    fake.skull_fields = False
    obs, _ = env.reset(seed=0)
    assert "altars" not in env._raw["campaign"] and "needs_item" not in env._raw["campaign"]["gates"][0]
    assert env.gates.target["key"] == SKULL_GATE_KEY and "subgoal" not in env.gates.target
    parts: dict[str, float] = {}
    info = walk(env, 40, punch=True, parts=parts)
    env.close()
    assert info["end_reason"] == "level_complete"
    assert "item_pickup" not in parts and "item_placed" not in parts
    assert parts["punch"] < 0.0, "with nothing held, no press is ever dropped, so every one is charged"


def test_new_mod_fields_ignored_by_old_rules():
    """New mod against pre-S4 Python: `_choose_target` is the old rule, and the new fields do not disturb it."""
    env, fake = skull_env()
    env.reset(seed=0)
    campaign = env._raw["campaign"]
    assert campaign["gates"][0]["needs_item"] == "SkullRed" and campaign["altars"] and campaign["items"]
    old_target = env.gates._choose_target(campaign, [0.0, 1.0, 0.0])
    assert old_target["key"] == SKULL_GATE_KEY and "subgoal" not in old_target
    assert env.gates._gates(campaign) == campaign["gates"], "an altar_only gate is just another gate"
    env.close()


# ---------------------------------------------------------------------------------------------
# S1: the multi-level curriculum
# ---------------------------------------------------------------------------------------------

def write_curriculum(path: Path, *, order=LEVELS, run_name=None, favour=None, text=None) -> None:
    """A curriculum file whose weights make `favour` the only level a fresh load can draw (with floor 0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        path.write_text(text, encoding="utf-8")
        return
    levels = {level: {"unlocked": True, "fresh_window": 50, "best_time": None, "episodes": 0,
                      "fresh_completion_rate": 0.0 if level == favour else 1.0} for level in order}
    path.write_text(json.dumps({"version": 1, "updated_at": 1.0, "run_name": run_name or path.parent.name,
                                "order": list(order), "levels": levels}), encoding="utf-8")


def curriculum_env(tmp: str, **overrides) -> tuple[UltrakillEnv, FakeLevel, Path]:
    run = Path(tmp) / "runs" / "campaign_multi"
    path = run / "curriculum.json"
    env, fake = make_env(levels=LEVELS, curriculum_path=str(path), level_weight_floor=0.0,
                         explore_dir=str(Path(tmp) / "models"), best_runs_dir=str(run / "best_runs"),
                         **overrides)
    return env, fake, path


def test_a_fresh_start_switches_level_and_a_respawn_never_does():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, path = curriculum_env(tmp, fresh_start_prob=0.0, max_steps=40)
        write_curriculum(path, favour=LEVELS[0])
        _, info = env.reset(seed=0)
        assert env.level == LEVELS[0] and info["level"] == LEVELS[0]
        assert fake.reset_scenes == [LEVELS[0]]
        info = walk(env, 40)  # to the exit: a completion forces the next episode to be a fresh load
        assert info["end_reason"] == "level_complete" and info["level"] == LEVELS[0]

        write_curriculum(path, favour=LEVELS[1])  # re-read at the next fresh start, not cached from startup
        _, info = env.reset()
        assert env.level == LEVELS[1] and info["level"] == LEVELS[1] and info["fresh_start"] == 1
        assert fake.reset_scenes[-1] == LEVELS[1] and fake.z == 0.0

        write_curriculum(path, favour=LEVELS[0])  # the curriculum now wants 0-1 again ...
        walk(env, 11)  # ... but this episode ends at a checkpoint, so the next reset is a respawn
        run_until_end(env)
        _, info = env.reset()
        assert info["fresh_start"] == 0 and fake.resets[-1] is True
        assert env.level == LEVELS[1] and info["level"] == LEVELS[1], "a respawn never re-samples"
        env.close()


def test_a_no_checkpoint_death_reloads_the_level_it_is_on():
    """`_respawn` reloads the whole level when there is no checkpoint, and that must NOT be a re-sample.

    Sampling there would change the scene mid-episode and leave info["level"], the best run's positions and the
    exploration archive all disagreeing with each other.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, path = curriculum_env(tmp, max_steps=60)
        write_curriculum(path, favour=LEVELS[1])
        env.reset(seed=0)
        assert env.level == LEVELS[1]
        walk(env, 5)  # z 10, short of the checkpoint
        write_curriculum(path, favour=LEVELS[0])
        fake.kill_next = True
        _, _, terminated, truncated, info = env.step(idle())
        assert not terminated and not truncated and info["deaths"] == 1
        assert fake.z == 0.0 and fake.resets[-1] is True  # StatsManager.Restart reloaded the level
        assert env.level == LEVELS[1] and fake.reset_scenes[-1] == LEVELS[1]
        env.close()


def test_each_level_keeps_its_own_archive_and_best_run():
    with tempfile.TemporaryDirectory() as tmp:
        env, fake, path = curriculum_env(tmp, max_steps=40)
        models = Path(tmp) / "models"
        write_curriculum(path, favour=LEVELS[0])
        env.reset(seed=0)
        walk(env, 40)  # a completion on 0-1
        first = dict(env.archive.counts)
        assert first

        write_curriculum(path, favour=LEVELS[1])
        env.reset()
        assert env.level == LEVELS[1]
        assert env.archive.counts != first and len(env.archive.counts) <= 1, "0-3 starts from its own empty archive"
        assert (models / f"explore_Level_0-1_{env.cfg.port}.npz").exists(), "the outgoing archive was saved first"
        walk(env, 40)  # a completion on 0-3 too

        write_curriculum(path, favour=LEVELS[0])
        env.reset()
        assert env.level == LEVELS[0]
        assert set(first) <= set(env.archive.counts), "switching back restores 0-1's own counts"
        env.close()

        best = Path(tmp) / "runs" / "campaign_multi" / "best_runs"
        assert sorted(p.name for p in best.glob("*.json")) == ["Level_0-1.json", "Level_0-3.json"]
        for name, level in (("Level_0-1.json", LEVELS[0]), ("Level_0-3.json", LEVELS[1])):
            assert json.loads((best / name).read_text(encoding="utf-8"))["level"] == level
        assert sorted(p.name for p in models.glob("*.npz")) == [
            f"explore_Level_0-1_{env.cfg.port}.npz", f"explore_Level_0-3_{env.cfg.port}.npz"]


def test_an_unusable_curriculum_file_leaves_the_worker_on_the_first_level():
    """Missing, torn, another run's name, another run's order: all four mean "the first level only"."""
    broken = {
        "missing": None,
        "torn": '{"version": 1, "order": ["Level 0-1", "Level 0-3"], "lev',
        "empty": "{}",
        "wrong_run": json.dumps({"version": 1, "run_name": "someone_else", "order": LEVELS,
                                 "levels": {LEVELS[0]: {"unlocked": True, "fresh_window": 50, "fresh_completion_rate": 1.0},
                                            LEVELS[1]: {"unlocked": True, "fresh_window": 50, "fresh_completion_rate": 0.0}}}),
        "wrong_order": json.dumps({"version": 1, "run_name": "campaign_multi", "order": ["Level 0-3", "Level 0-1"],
                                   "levels": {LEVELS[1]: {"unlocked": True, "fresh_window": 50, "fresh_completion_rate": 0.0}}}),
    }
    for name, text in broken.items():
        with tempfile.TemporaryDirectory() as tmp:
            env, fake, path = curriculum_env(tmp, max_steps=6)
            if text is not None:
                write_curriculum(path, text=text)
            for _ in range(3):
                _, info = env.reset(seed=0)
                assert env.level == LEVELS[0] and info["level"] == LEVELS[0], name
                run_until_end(env, limit=10)
            env.close()


def test_a_single_level_config_never_opens_a_curriculum_file():
    env, fake = make_env(curriculum_path="F:/nowhere/curriculum.json")  # no `levels`, so it is never read
    assert env.cfg.levels == [] and env.level == LEVEL
    _, info = env.reset(seed=0)
    assert info["level"] == LEVEL and fake.reset_scenes == [LEVEL]
    env.close()


def test_levels_must_be_scenes_this_build_ships():
    for bad in (["Level 9-1"], ["Level 0-1", "Level 9-2"], ["Level 0-0"]):
        try:
            UltrakillEnv(EnvConfig(mode="campaign", levels=bad))
        except ValueError as exc:
            assert "cannot load" in str(exc), exc
        else:
            raise AssertionError(f"{bad} should have been refused")
    ok = UltrakillEnv(EnvConfig(mode="campaign", levels=LEVELS, level="Level 4-1"))
    assert ok.level == LEVELS[0], "`levels` wins over `level` and levels[0] is the start"
    ok.close()


def test_the_weighting_rule_must_be_one_the_code_implements():
    """A typo in `curriculum_weighting` is refused at construction, not swallowed.

    Swallowing it would silently hand the run back to `inverse_rate` -- the rule that let one blocked level take
    half of every fresh draw -- with nothing on the dashboard to say so.
    """
    for bad in ("progres", "Progress", "learning_progress", ""):
        try:
            UltrakillEnv(EnvConfig(mode="campaign", levels=LEVELS, curriculum_weighting=bad))
        except ValueError as exc:
            assert "curriculum_weighting" in str(exc), exc
        else:
            raise AssertionError(f"{bad!r} should have been refused")
    for good in ("inverse_rate", "progress"):
        env = UltrakillEnv(EnvConfig(mode="campaign", levels=LEVELS, curriculum_weighting=good))
        assert env.cfg.curriculum_weighting == good
        env.close()
    assert EnvConfig().curriculum_weighting == "inverse_rate", "every run before 2026-09-18 keeps its rule"


def test_human_routes_are_retired():
    """No human demo and no recorded route, which layer 2 does not change: its rungs are level DATA.

    `route_dir` is a live config field again, but it names the offline room trunks `build_routes.py` derives
    from the shipped scene bundles -- room transforms and their authored numbering, nothing a person played.
    """
    root = Path(__file__).resolve().parents[1]
    assert not (root / "ultrakill_ai" / "routes.py").exists()
    assert not (root / "scripts" / "record_route.py").exists()
    assert not hasattr(RewardConfig(), "route_point"), "the human-route reward terms stay gone"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
