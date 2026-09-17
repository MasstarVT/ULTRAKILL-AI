"""Campaign episodes in UltrakillEnv, against a fake level instead of the game:  python tests/test_campaign_env.py  (or pytest)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
      `gates_ordered`  False is a level whose door graph has no goal room (0-5)
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
        self.mod_flags = True
        self.gates_present = True
        self.gates_ordered = True
        self.gates = GATES
        self.ground_center: float | None = None
        self.yaw = 0.0
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
        self._load()

    def fail_next_resets(self, exc: BaseException, times: int = 1) -> None:
        self.fail_resets.extend(type(exc)(*exc.args) for _ in range(times))

    def fail_next_steps(self, exc: BaseException, times: int = 1) -> None:
        self.fail_steps.extend(type(exc)(*exc.args) for _ in range(times))

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

    def get_obs(self) -> dict:
        return self._obs()

    def kill(self) -> dict:
        """The protocol's debug kill. soft_death is not modelled here, so this is always a real death."""
        self.dead = True
        return self._obs("kill")

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict:
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
        airborne = self.falling or self.wedged
        player = {
            "pos": [0.0, self.y, self.z], "vel": [0.0, -20.0 if self.falling else 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, 1.0], "yaw": self.yaw, "pitch": 0.0, "hp": 0 if self.dead else 100,
            "anti_hp": 0.0, "stamina": 300.0, "grounded": not airborne, "sliding": False, "dead": self.dead,
            "activated": True, "level_over": over, "weapon_slot": 0, "weapon_variation": 0,
            "soft_deaths": 0, "soft_death_instakill": False, "slot_counts": [1, 0, 0, 0, 0],
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
                "mission": 1, "difficulty": 3, "seconds": self.seconds, "timer_running": not over,
                "level_started": True, "level_over": over, "restarts": self.restarts, "input_locked": self.locked,
                "exit": {"pos": [0.0, 1.0, EXIT_Z], "active": True},
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
        "route_dir": "routes",
        "layout": {"max_enemies": 8, "waypoint_offset": 3},
        "rewards": {"kill": 0.5, "route_point": 0.1, "stuck": 1.0},
    })
    assert cfg.mode == "campaign" and cfg.fixed_fps == 30 and cfg.layout.max_enemies == 8 and cfg.rewards.kill == 0.5
    for retired in ("checkpoint_resets", "stuck_steps", "route_dir"):
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


def test_human_routes_are_retired():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "ultrakill_ai" / "routes.py").exists()
    assert not (root / "scripts" / "record_route.py").exists()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
