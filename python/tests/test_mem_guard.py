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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
