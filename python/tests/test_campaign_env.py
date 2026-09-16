"""Campaign episodes in UltrakillEnv, against a fake level instead of the game:  python tests/test_campaign_env.py  (or pytest)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrakill_ai.env import CAMPAIGN_INFO_KEYS, EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.protocol import BridgeClient  # noqa: E402
from ultrakill_ai.rewards import RewardConfig  # noqa: E402
from ultrakill_ai.spaces import noop_action  # noqa: E402

LEVEL = "Level 0-1"
EXIT_Z = 60.0
CHECKPOINT_ID = "0,1,20"
ARENA_KEY = "0,1,30"
RESPAWN_DOOR_KEY = "0,0,25"
RANKS = {"time": [120, 90, 60, 30], "kills": [0, 1, 2, 3], "style": [0, 100, 200, 300]}
ENEMY_ID = 7
ENEMY_MAX_HP = 50.0


class FakeLevel:
    """Stands in for BridgeClient: a straight corridor along +z.

    Walking forward covers 2 m a step. The checkpoint at z 20 activates (and becomes current) on arrival, the
    arena at z 30 clears on arrival, and the exit is at z 60. A checkpoint respawn puts the player back at z 20
    and unlocks a door, the way `StatsManager.Restart` unlocks `doorsToUnlock`: that unlock must never pay.

    The room holds one enemy, alive from the moment the level is entered or re-entered (`reset`, fresh or
    checkpoint), so a respawn re-creates it exactly like the real game while `kills` (the game's own counter)
    keeps whatever value it already had -- the pairing `test_kill_reward_can_be_farmed_across_a_checkpoint_respawn`
    documents.
    """

    def __init__(self):
        self.resets: list[bool] = []  # the checkpoint flag of every reset
        self.steps = 0
        self.kill_next = False  # the next step returns a dead player
        self.kill_enemy_next = False  # the next step removes the room's enemy, if alive, and pays a kill
        self.lock_steps = 0  # the next N steps report input_locked
        self.drop_campaign_steps = 0  # the next N obs omit "campaign", as the mod does when building it throws
        self._load()

    def _load(self) -> None:
        self.z = 0.0
        self.seconds = 0.0
        self.kills = 0
        self.restarts = 0
        self.dead = False
        self.locked = False
        self.checkpoint = False
        self.arenas: list[str] = []
        self.doors: list[str] = []
        self.enemy_alive = False  # reset() below turns this on: every level (re-)entry re-creates the room

    # The BridgeClient methods UltrakillEnv uses ----------------------------

    def connect(self, retry_seconds: float = 60.0) -> dict:
        return {"type": "hello", "protocol": 1, "mod_version": "0.5.0", "scene": LEVEL}

    def configure(self, **settings) -> None:
        self.settings = settings

    def close(self) -> None:
        pass

    def get_obs(self) -> dict:
        return self._obs()

    def reset(self, scene: str | None = None, checkpoint: bool = False) -> dict:
        self.resets.append(checkpoint)
        if checkpoint and self.checkpoint:
            self.z = 20.0
            self.dead = False
            self.restarts += 1
            if RESPAWN_DOOR_KEY not in self.doors:
                self.doors.append(RESPAWN_DOOR_KEY)
        else:
            self._load()
        self.enemy_alive = True  # a fresh load or a checkpoint respawn both re-create the room's enemy
        return self._obs("reset")

    def step(self, action: dict) -> dict:
        self.steps += 1
        self.locked = self.lock_steps > 0
        if self.locked:
            self.lock_steps -= 1
        if not self.dead and self.z < EXIT_Z:
            if self.kill_next:
                self.dead, self.kill_next = True, False
            elif not self.locked and action.get("move", [0, 0])[1] > 0:
                self.z = min(EXIT_Z, self.z + 2.0)
            self.seconds += 2.0 / 15.0
        if self.enemy_alive and self.kill_enemy_next:
            # A one-shot kill: the enemy vanishes without a health drop first, same as the mod reports it.
            self.enemy_alive, self.kill_enemy_next = False, False
            self.kills += 1
        if self.z >= 20.0:
            self.checkpoint = True
        if self.z >= 30.0 and ARENA_KEY not in self.arenas:
            self.arenas.append(ARENA_KEY)
        return self._obs()

    def _obs(self, event: str | None = None) -> dict:
        over = self.z >= EXIT_Z
        enemies = []
        if self.enemy_alive:
            enemies = [{
                "id": ENEMY_ID, "type": 0, "health": ENEMY_MAX_HP, "visible": True,
                "rel": [0.0, 0.0, 5.0], "dist": 5.0, "pos": [0.0, 1.0, self.z + 5.0],
            }]
        obs = {
            "type": "obs",
            "step": self.steps,
            "scene": LEVEL,
            "ready": not self.dead,
            "player": {
                "pos": [0.0, 1.0, self.z], "vel": [0.0, 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, 1.0], "yaw": 0.0, "pitch": 0.0, "hp": 0 if self.dead else 100,
                "anti_hp": 0.0, "stamina": 300.0, "grounded": True, "sliding": False, "dead": self.dead,
                "activated": True, "level_over": over, "weapon_slot": 0, "weapon_variation": 0,
                "soft_deaths": 0, "soft_death_instakill": False, "slot_counts": [1, 0, 0, 0, 0],
            },
            "enemies": enemies,
            "rays": [50.0] * 16,
            "ground_rays": [0.0] * 8,
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
            },
        }
        if event:
            obs["event"] = event
        if self.drop_campaign_steps > 0:
            # docs/protocol.md: the mod logs and omits the whole campaign block for a step if building it throws.
            self.drop_campaign_steps -= 1
            del obs["campaign"]
        return obs


def make_env(**overrides) -> tuple[UltrakillEnv, FakeLevel]:
    rewards = RewardConfig(time=0.01, checkpoint=10.0, arena_clear=10.0, door_unlock=3.0, novelty=0.5, path=0.1)
    cfg = EnvConfig(mode="campaign", level=LEVEL, fixed_fps=30, frameskip=2, rewards=rewards, **overrides)
    env = UltrakillEnv(cfg)
    env.client = FakeLevel()
    return env, env.client


def forward():
    a = noop_action()
    a[0] = 2  # move forward
    return a


def add_parts(total: dict[str, float], info: dict) -> None:
    for name, value in info["reward_parts"].items():
        total[name] = total.get(name, 0.0) + value


def run_until_end(env: UltrakillEnv, limit: int = 500) -> dict:
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(noop_action())
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
    _, _, terminated, truncated, info = env.step(noop_action())
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
        _, _, terminated, truncated, info = env.step(noop_action())
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
    _, _, terminated, truncated, info = env.step(noop_action())
    assert not terminated and not truncated
    first_kill, first_damage = info["reward_parts"]["kill"], info["reward_parts"]["damage_dealt"]
    assert info["kills"] == 1 and first_kill > 0 and first_damage > 0

    fake.kill_next = True  # the player also dies now, which triggers a checkpoint respawn
    _, _, terminated, truncated, info = env.step(noop_action())
    assert not terminated and not truncated  # a death inside a campaign episode respawns it, not ends it
    assert info["deaths"] == 1
    assert fake.enemy_alive and fake.kills == 1  # the room's enemy is back; the kill counter did not reset

    fake.kill_enemy_next = True
    _, _, terminated, truncated, info = env.step(noop_action())
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
    _, _, terminated, truncated, info = env.step(noop_action())  # campaign block missing, but the kill still lands
    add_parts(parts, info)
    assert not terminated and not truncated
    assert info["kills"] == 1
    assert info["reward_parts"]["kill"] > 0 and info["reward_parts"]["damage_dealt"] > 0
    assert env._steps_since_progress == 1  # no checkpoint/arena/door/novelty/path signal while the block is gone

    _, _, terminated, truncated, info = env.step(noop_action())  # still missing (2nd of the 2 dropped steps)
    add_parts(parts, info)
    assert not terminated and not truncated
    assert fake.drop_campaign_steps == 0
    assert env._steps_since_progress == 2

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
    env.step(noop_action())
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
