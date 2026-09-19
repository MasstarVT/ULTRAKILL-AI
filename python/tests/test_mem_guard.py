"""mem_guard.py: which game is recycled, and why. No game, no process:  python tests/test_mem_guard.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mem_guard  # noqa: E402
from mem_guard import GB, choose_victim  # noqa: E402
from ultrakill_ai.env import EnvConfig, UltrakillEnv  # noqa: E402
from ultrakill_ai.procmem import derive_game_limit  # noqa: E402

LIMIT = 3 * GB


def test_nothing_is_recycled_while_every_game_is_small_and_the_system_has_room():
    sizes = {47800: int(1.1 * GB), 47801: int(1.4 * GB)}
    assert choose_victim(sizes, 0.45, LIMIT, 0.80) is None


def test_the_fattest_game_over_its_limit_goes_first():
    sizes = {47800: int(3.2 * GB), 47801: int(5.9 * GB), 47802: int(1.0 * GB)}
    port, reason = choose_victim(sizes, 0.50, LIMIT, 0.80)
    assert port == 47801 and "over its" in reason


def test_a_short_system_recycles_the_fattest_game_even_under_its_own_limit():
    """An analysis script has twice eaten 50-70 GB: the games are what the guard can give back."""
    sizes = {47800: int(1.2 * GB), 47801: int(2.4 * GB)}
    port, reason = choose_victim(sizes, 0.91, LIMIT, 0.80)
    assert port == 47801 and "system commit" in reason


def test_exactly_at_the_limit_is_not_over_it():
    assert choose_victim({47800: LIMIT}, 0.80, LIMIT, 0.80) is None


def test_no_games_means_no_victim_however_short_the_system_is():
    assert choose_victim({}, 0.99, LIMIT, 0.80) is None


def test_a_fleet_that_ages_together_is_thinned_before_every_copy_reaches_its_limit():
    """2026-09-19 01:28: twelve copies launched together reached 32 GB and 89% commit with none over its limit."""
    sizes = {47800 + i: int((2.1 + 0.01 * i) * GB) for i in range(12)}  # 25.9 GB, every copy under 3 GB
    assert choose_victim(sizes, 0.80, LIMIT, 0.93) is None, "no budget given: the old behaviour"
    port, reason = choose_victim(sizes, 0.80, LIMIT, 0.93, total_limit=24 * GB)
    assert port == 47811 and "budget" in reason


def test_a_staggered_fleet_under_the_budget_is_left_alone():
    sizes = {47800 + i: int((0.4 + 0.2 * i) * GB) for i in range(12)}  # 18 GB, fattest 2.6 GB
    assert choose_victim(sizes, 0.70, LIMIT, 0.93, total_limit=24 * GB) is None


def test_only_bridge_ports_owned_by_a_game_are_ever_candidates():
    """The first dry run counted 19 "games" with none running: every listening port on the machine."""
    listening = {135: 1000, 445: 4, 47800: 5001, 47801: 5002, 47812: 5003, 50000: 5004, 47802: 9999}
    games_running = {5001, 5002, 5003, 5004}
    assert mem_guard.bridge_games(listening, games_running, 47800, 32) == {47800: 5001, 47801: 5002, 47812: 5003}
    assert mem_guard.bridge_games(listening, set(), 47800, 32) == {}


def test_the_readings_work_on_this_process():
    assert (mem_guard.private_bytes(os.getpid()) or 0) > 1024 * 1024
    assert 0.0 < mem_guard.commit_fraction() < 1.0
    assert mem_guard.private_bytes(0x7FFFFFF0) is None  # a pid that cannot exist


# --------------------------------------------------------------- the limit is measured, not a constant


def test_the_limit_is_the_freshest_game_plus_the_allowed_growth():
    """A constant cannot track a build whose fresh size moved 1.0 -> 1.4 GB; the freshest copy running can."""
    fleet = {47800: int(1.4 * GB), 47801: int(1.9 * GB), 47802: int(2.2 * GB)}
    assert derive_game_limit(fleet, GB) == int(1.4 * GB) + GB


def test_a_half_booted_copy_is_never_taken_for_the_fresh_baseline():
    """The 2026-09-17 laggard sat at 56 MB. Believing that would put the limit under the boot size."""
    fleet = {47800: int(0.06 * GB), 47801: int(1.5 * GB), 47802: int(2.4 * GB)}
    assert derive_game_limit(fleet, GB) == int(1.5 * GB) + GB


def test_a_fleet_that_has_all_grown_together_is_still_capped():
    """The shape that killed the box: three copies at 5.8-6.1 GB and nothing fresh left to compare against."""
    fleet = {47800: int(5.8 * GB), 47801: int(6.1 * GB)}
    assert derive_game_limit(fleet, GB) == int(mem_guard.MAX_LIMIT_GB * GB)


def test_an_empty_fleet_gives_a_usable_limit_rather_than_nonsense():
    assert derive_game_limit({}, GB) == int(mem_guard.MIN_FRESH_GB * GB) + GB


def test_the_guard_and_the_env_derive_the_same_limit():
    """They recycle the same games for the same reason, so they must not drift apart."""
    fleet = {47800: int(1.35 * GB), 47801: int(3.0 * GB)}
    assert derive_game_limit(fleet, GB) == mem_guard.derive_game_limit(fleet, GB)


# --------------------------------------------------------------- the env recycles its own game at a boundary


class _Recorder(UltrakillEnv):
    """An env that never touches a game: the fleet reading, the relaunch and the reconnect are all stubbed."""

    def __init__(self, cfg, fleet, boots_to=None, relaunch_ok=True):
        super().__init__(cfg)
        self._fleet, self._boots_to, self._relaunch_ok = fleet, boots_to, relaunch_ok
        self.relaunched, self.reconnected = 0, 0

    def _fleet_memory_hook(self):
        return self._fleet

    def _relaunch_own_game(self, deadline):
        self.relaunched += 1
        if self._relaunch_ok and self._boots_to is not None:
            self._fleet = {**self._fleet, self.cfg.port: self._boots_to}
        return self._relaunch_ok

    def _reconnect(self):
        self.reconnected += 1


def _env(fleet, growth=1.0, relaunch=True, **kwargs):
    cfg = EnvConfig(port=47800, game_memory_growth_gb=growth, bridge_relaunch=relaunch)
    return _Recorder(cfg, fleet, **kwargs)


def test_a_fat_game_is_replaced_between_episodes():
    env = _env({47800: int(3.0 * GB), 47801: int(1.4 * GB)}, boots_to=int(1.4 * GB))
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 1 and env.reconnected == 1 and env._mem_recycles == 1


def test_a_healthy_game_is_left_alone():
    env = _env({47800: int(1.9 * GB), 47801: int(1.4 * GB)})
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 0 and env._mem_recycles == 0


def test_only_this_envs_own_port_is_ever_judged():
    """Another worker's game being fat is that worker's business; this one must not touch it."""
    env = _env({47800: int(1.4 * GB), 47801: int(5.9 * GB)})
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 0


def test_the_recycle_is_off_unless_the_run_turned_relaunching_on():
    """`bridge_relaunch` is train.py's alone, so an eval can no more recycle a game than relaunch one."""
    env = _env({47800: int(5.9 * GB)}, relaunch=False, boots_to=int(1.4 * GB))
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 0


def test_growth_zero_is_off():
    env = _env({47800: int(5.9 * GB)}, growth=0.0, boots_to=int(1.4 * GB))
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 0


def test_a_fresh_game_still_over_the_limit_latches_the_mechanism_off():
    """Otherwise a limit set below the boot size relaunches forever and the run never trains again."""
    env = _env({47800: int(3.0 * GB), 47801: int(1.4 * GB)}, boots_to=int(3.0 * GB))
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 1 and env._mem_recycle_off is True
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 1, "a latched-off env must not try again"


def test_a_failed_relaunch_does_not_count_as_a_recycle_or_drop_the_connection():
    env = _env({47800: int(3.0 * GB), 47801: int(1.4 * GB)}, relaunch=True, boots_to=None)

    def refuse(deadline):
        env.relaunched += 1
        return False

    env._relaunch_own_game = refuse
    env._recycle_own_game_if_fat(deadline=1e9)
    assert env.relaunched == 1 and env.reconnected == 0 and env._mem_recycles == 0


def test_a_broken_memory_reading_never_breaks_a_reset():
    """A memory optimisation may not become a new way for an episode to die."""
    env = _env({47800: int(3.0 * GB), 47801: int(1.4 * GB)}, boots_to=int(1.4 * GB))

    def explode():
        raise OSError("netstat said no")

    env._fleet_memory_hook = explode
    env._recycle_own_game_if_fat(deadline=1e9)  # must not raise
    assert env.relaunched == 0 and env._mem_recycle_off is True


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
