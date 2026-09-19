"""campaign_check.py's five in-game checks, run against a fake campaign level. No game needed:  python tests/test_campaign_check.py  (or pytest)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.env import UltrakillEnv  # noqa: E402

SPAWN = (0.0, 1.0, 0.0)
CHECKPOINTS = ((0.0, 0.0, 20.0), (0.0, 0.0, 40.0))
EXIT = (0.0, 0.0, 80.0)
TRIGGER_RADIUS = 2.0
# The door graph, sorted by hops ascending as the mod sends it.
GATES = (("0,0,60", (0.0, 0.0, 60.0), 0), ("0,0,30", (0.0, 0.0, 30.0), 1), ("0,0,10", (0.0, 0.0, 10.0), 2))


def load_script():
    spec = importlib.util.spec_from_file_location("campaign_check", ROOT / "scripts" / "campaign_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGame:
    """A campaign level as the mod reports it: checkpoints at z 20 and z 40 along +z, and an exit pit at z 80.

    Stands in for BridgeClient. Triggers fire on the step after the player comes within 2 m, as Unity trigger
    callbacks do on the next physics update. The timer stops when the exit fires. `apply_difficulty`, `triggers`
    and `exit` switch single features off for the failure cases. `void_checkpoint` is a checkpoint whose room is
    still switched off: arriving on it is a fall to death, not an activation. `heal_lethal` is soft death healing
    the kill (a living player in the reply, counted in `soft_deaths`).
    """

    def __init__(self, *, scene="Level 0-1", slot_counts=(0, 0, 0, 0, 0, 0), apply_difficulty=True, triggers=True, exit=EXIT,
                 void_checkpoint=None, heal_lethal=False, gates=GATES, gates_ordered=True, renumber_on_respawn=False):
        self.gates = gates  # None stands in for a mod older than 0.6.0
        self.gates_ordered = gates_ordered
        self.renumber_on_respawn = renumber_on_respawn
        self.scene = scene
        self.slot_counts = list(slot_counts)
        self.apply_difficulty = apply_difficulty
        self.triggers = triggers
        self.exit = exit
        self.void_checkpoint = void_checkpoint
        self.heal_lethal = heal_lethal
        self.soft_deaths = 0  # the mod's counter lives across level loads
        self.settings: dict = {}
        self.resets: list[bool] = []
        self._load()

    def _load(self):
        self.pos = list(SPAWN)
        self.dead = False
        self.activated: set[int] = set()
        self.current: int | None = None
        self.seconds = 0.0
        self.level_over = False
        self.restarts = 0
        self.steps = 0

    def connect(self, retry_seconds: float = 60.0):
        # Same signature as BridgeClient.connect: the env passes `retry_seconds` so a relaunched game has time
        # to start listening.
        return {"type": "hello", "protocol": 1, "mod_version": "0.5.0", "scene": "Main Menu"}

    def configure(self, **settings):
        self.settings.update(settings)

    def close(self):
        pass

    def reset(self, scene=None, checkpoint=False, timeout=None):  # timeout: the ladder's budget clamp
        self.resets.append(checkpoint)
        if checkpoint and self.current is not None:
            x, y, z = CHECKPOINTS[self.current]
            self.pos = [x, y + 1.25, z]  # CheckPoint respawns the player 1.25 m above its origin
            self.dead = False
            self.restarts += 1
        else:
            self._load()
        return self._obs("reset")

    def step(self, action):
        self.steps += 1
        if not self.level_over:
            # One game SECOND per step: this corridor is crossed in a handful of steps, and a level time under
            # a second is exactly what the env now discards as "the game never reported one"
            # (times.valid_official_seconds), so a miniature clocked in frames would hand check 5 a time no
            # real run can have. The check compares the env's time with the block's, whatever the rate is.
            self.seconds += 1.0
        if self.triggers and not self.dead:
            for i, cp in enumerate(CHECKPOINTS):
                if i not in self.activated and math.dist(self.pos, cp) <= TRIGGER_RADIUS:
                    if i == self.void_checkpoint:
                        self.dead = True
                        break
                    self.activated.add(i)
                    self.current = i
            if self.exit is not None and math.dist(self.pos, self.exit) <= TRIGGER_RADIUS:
                self.level_over = True
        return self._obs()

    def teleport(self, pos):
        self.pos = [float(v) for v in pos]
        return self._obs("teleport")

    def kill(self):
        if self.heal_lethal:
            self.soft_deaths += 1
        else:
            self.dead = True
        return self._obs("kill")

    def get_obs(self):
        return self._obs()

    def _obs(self, event=None):
        difficulty = self.settings.get("difficulty", -1) if self.apply_difficulty else -1
        player = {
            "pos": list(self.pos), "vel": [0.0, 0.0, 0.0], "local_vel": [0.0, 0.0, 0.0], "forward": [0.0, 0.0, 1.0],
            "yaw": 0.0, "pitch": 0.0, "hp": 0 if self.dead else 100, "anti_hp": 0.0, "stamina": 300.0,
            "grounded": True, "sliding": False, "dead": self.dead, "activated": not self.level_over,
            "level_over": self.level_over, "weapon_slot": -1, "weapon_variation": -1, "soft_deaths": self.soft_deaths,
            "soft_death_instakill": False, "slot_counts": list(self.slot_counts),
        }
        campaign = {
            "mission": 1, "difficulty": difficulty if difficulty >= 0 else 2, "seconds": self.seconds,
            "timer_running": not self.level_over, "level_started": True, "level_over": self.level_over,
            "restarts": self.restarts, "input_locked": self.level_over,
            "exit": {"pos": list(self.exit), "active": True} if self.exit is not None else None,
            "checkpoints": [
                {"id": f"{x:.0f},{y:.0f},{z:.0f}", "pos": [x, y, z], "activated": i in self.activated, "current": i == self.current}
                for i, (x, y, z) in enumerate(CHECKPOINTS)
            ],
            "path": {"status": "complete", "length": 80.0 - self.pos[2], "next_corner": list(self.exit)} if self.exit else {"status": "none"},
            "locked_doors": [], "arena_enemies_alive": 0, "cleared_arenas": [], "unlocked_doors": [],
            "ranks": {"time": [120, 90, 60, 30], "kills": [0, 1, 2, 3], "style": [0, 100, 200, 300]},
        }
        if self.gates is not None:
            shift = 1 if (self.renumber_on_respawn and self.restarts) else 0  # a Scan() that renumbered the route
            campaign.update(
                gates_ordered=self.gates_ordered,
                gates_truncated=False,
                gates=[{"key": key, "pos": list(pos), "hops": (hops + shift) if self.gates_ordered else None,
                        "open": False, "locked": False, "active": True, "controller_active": True}
                       for key, pos, hops in self.gates],
            )
        obs = {
            "type": "obs", "step": self.steps, "scene": self.scene, "ready": not self.dead, "player": player,
            "enemies": [], "rays": [50.0] * 16, "ground_rays": [0.0] * 8,
            "stats": {"kills": 0, "style": 0, "seconds": self.seconds, "restarts": self.restarts, "level_complete": self.level_over},
            "campaign": campaign,
        }
        if event:
            obs["event"] = event
        return obs


def run(game: FakeGame, *args: str):
    script = load_script()
    env = UltrakillEnv(script.build_config(script.parse_args(["--level", game.scene, *args])))
    env.client = game
    try:
        results = script.run_checks(env)
    finally:
        env.close()
    return script, results


def statuses(results):
    return [status for _, _, status, _ in results]


def test_every_check_passes_on_a_working_level():
    game = FakeGame()
    script, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "PASS", "PASS"], results
    assert game.settings["difficulty"] == 3 and game.settings["unlock_all_gear"] is True
    assert game.resets == [False, True]  # one fresh load, then the respawn after the kill
    assert "revolver pickup" in results[1][3]
    assert "3 gates, hops 0..2 (3 distinct)" in results[5][3]
    assert "after a respawn" in results[5][3]
    assert script.exit_code(results) == 0


def test_the_arsenal_passes_when_every_weapon_slot_is_filled():
    _, results = run(FakeGame(scene="Level 1-1", slot_counts=(4, 3, 3, 3, 3, 0)))
    assert statuses(results) == ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS"], results


def test_a_difficulty_the_game_ignores_fails_the_level_load():
    script, results = run(FakeGame(apply_difficulty=False))
    assert statuses(results)[0] == "FAIL"
    assert "expected 3" in results[0][3]
    assert script.exit_code(results) == 1


def test_triggers_that_never_fire_fail_the_checkpoint_and_exit_and_skip_the_death():
    game = FakeGame(triggers=False)
    script, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "FAIL", "SKIP", "FAIL", "PASS"], results
    assert results[2][3].count("did not activate") == len(CHECKPOINTS)  # every pending checkpoint was tried
    assert "did not fire" in results[4][3]
    assert "no respawn happened" in results[5][3]  # nothing to compare the gates against
    assert script.exit_code(results) == 1


def test_a_null_exit_fails_the_exit_check():
    _, results = run(FakeGame(exit=None))
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "FAIL", "PASS"], results
    assert results[4][3].startswith("exit null")


def test_a_fall_during_the_checkpoint_check_does_not_fail_the_death_check():
    game = FakeGame(void_checkpoint=0)
    _, results = run(game)
    assert statuses(results) == ["PASS", "SKIP", "PASS", "PASS", "PASS", "PASS"], results
    assert game.resets == [False, True, True]  # the fresh load, the reload after the fall, the respawn after the kill
    assert "0,0,20 did not activate" in results[2][3] and "0,0,40 activated" in results[2][3]
    assert "deaths 1 -> 2" in results[3][3]


def test_a_kill_that_soft_death_heals_fails_the_death_check():
    script, results = run(FakeGame(heal_lethal=True))
    assert statuses(results) == ["PASS", "SKIP", "PASS", "FAIL", "PASS", "PASS"], results
    assert "kill reply dead=False, deaths 0 -> 1" in results[3][3]  # the env still counted and respawned it
    assert script.exit_code(results) == 1


def test_a_mod_without_gates_skips_the_gates_check():
    """Graceful degradation: Python may be merged before the mod, and then there is nothing to check."""
    script, results = run(FakeGame(gates=None))
    assert statuses(results)[5] == "SKIP", results
    assert "older than v0.6.0" in results[5][3]
    assert script.exit_code(results) == 0


def test_an_unordered_level_fails_the_gates_check():
    script, results = run(FakeGame(gates_ordered=False))
    assert statuses(results)[5] == "FAIL", results
    assert "gates_ordered is false" in results[5][3] and "no gate reports hops 0" in results[5][3]
    assert script.exit_code(results) == 1


def test_hops_that_renumber_across_a_respawn_fail_the_gates_check():
    """A Scan() landing mid-room-recreation must not blank or renumber the route: hops is stable per level load."""
    script, results = run(FakeGame(renumber_on_respawn=True))
    assert statuses(results)[5] == "FAIL", results
    assert "renumbered mid-load" in results[5][3]
    assert script.exit_code(results) == 1


def test_gates_out_of_order_or_duplicated_fail_the_gates_check():
    unsorted = (("0,0,10", (0.0, 0.0, 10.0), 2), ("0,0,60", (0.0, 0.0, 60.0), 0))
    _, results = run(FakeGame(gates=unsorted))
    assert statuses(results)[5] == "FAIL" and "not sorted" in results[5][3], results
    duplicated = (("0,0,60", (0.0, 0.0, 60.0), 0), ("0,0,60", (0.0, 0.0, 60.5), 1))
    _, results = run(FakeGame(gates=duplicated))
    assert statuses(results)[5] == "FAIL" and "duplicate keys" in results[5][3], results


def test_command_line_overrides_and_config_defaults():
    script = load_script()
    cfg = script.build_config(script.parse_args(["--level", "Level 1-1", "--port", "47801", "--fixed-fps", "60", "--frameskip", "4", "--render"]))
    assert (cfg.mode, cfg.level, cfg.port, cfg.fixed_fps, cfg.frameskip, cfg.render) == ("campaign", "Level 1-1", 47801, 60.0, 4, True)
    assert (cfg.fresh_start_prob, cfg.explore_dir, cfg.best_runs_dir) == (1.0, "", "")
    assert cfg.difficulty == 3 and cfg.unlock_all_gear is True

    env = yaml.safe_load((ROOT / "configs" / "campaign_0-1.yaml").read_text(encoding="utf-8"))["env"]
    default = script.build_config(script.parse_args([]))
    assert (default.level, default.port) == ("Level 0-1", 47800)
    assert (default.fixed_fps, default.frameskip, default.render) == (env["fixed_fps"], env["frameskip"], env["render"])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
