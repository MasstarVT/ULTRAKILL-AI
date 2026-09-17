"""One sick game must not kill a twelve-game run:  python tests/test_bridge_recovery.py  (or pytest).

The `campaign_gates` run died three times on 2026-09-17 the same way: a worker's socket raised `TimeoutError`
on a level reset, the exception escaped into SubprocVecEnv's `_worker` (which does not catch it), the worker
process exited, the parent's pipe read EOF, and all twelve games' rollouts went with it. A fourth death was the
same blast radius from a different cause -- a half-booted game answering every reset `unknown scene`.

These tests inject exactly those three faults (a timed-out reply, a dropped socket, a game still booting) into
the fake client and assert that the episode dies instead of the run, with the campaign trackers left consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_campaign_env import CHECKPOINT_ID, LEVEL, forward, idle, make_env  # noqa: E402
from ultrakill_ai.protocol import (  # noqa: E402
    DEFAULT_RESET_TIMEOUT,
    DEFAULT_STEP_TIMEOUT,
    RECOVERABLE,
    BridgeClient,
    BridgeClosed,
    BridgeError,
    BridgeSceneUnknown,
    BridgeTimeout,
)

# Fast settings for every recovery test: no real backoff sleeping, but plenty of patience on the clock.
FAST = dict(bridge_backoff_s=0.0, unknown_scene_wait_s=60.0, bridge_retries=3)


class FakeSocket:
    """The two socket methods BridgeClient uses, plus a record of every timeout it asked for."""

    def __init__(self, sends_raise: BaseException | None = None):
        self.timeouts: list[float] = []
        self.sent: list[bytes] = []
        self.closed = False
        self.sends_raise = sends_raise

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendall(self, payload):
        if self.sends_raise is not None:
            raise self.sends_raise
        self.sent.append(payload)

    def close(self):
        self.closed = True


class FakeReader:
    """Newline-JSON lines, or an exception raised on the read that would have returned them."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def readline(self):
        if not self.lines:
            return ""
        item = self.lines.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def wired(lines, sends_raise=None, **kwargs) -> tuple[BridgeClient, FakeSocket]:
    client = BridgeClient("127.0.0.1", 47800, **kwargs)
    sock = FakeSocket(sends_raise)
    client._sock, client._reader = sock, FakeReader(lines)
    return client, sock


# -- the protocol ------------------------------------------------------------------------------------


def test_a_reset_waits_far_longer_than_a_step():
    # The whole root cause in one assertion: a reset blocks on a Unity scene load and a step on one frame, so
    # they cannot share a deadline. They used to, at 120 s -- the same number as the mod's own reset timeout.
    client, sock = wired(['{"type":"obs"}\n', '{"type":"obs"}\n'])
    client.step({})
    client.reset(LEVEL)
    assert sock.timeouts == [DEFAULT_STEP_TIMEOUT, DEFAULT_RESET_TIMEOUT]
    assert DEFAULT_RESET_TIMEOUT > DEFAULT_STEP_TIMEOUT


def test_a_timed_out_request_poisons_the_connection():
    # The reply the game was still writing is in the socket. Reusing the stream would return it as the answer
    # to the NEXT request, and every observation after it would be one request stale.
    client, _ = wired([TimeoutError("timed out"), '{"type":"obs","late":true}\n'])
    try:
        client.step({})
        raise AssertionError("expected BridgeTimeout")
    except BridgeTimeout:
        pass
    assert client.broken
    for call in (lambda: client.step({}), lambda: client.get_obs()):
        try:
            call()
            raise AssertionError("expected BridgeClosed on a broken client")
        except BridgeClosed:
            pass


def test_a_dropped_socket_reads_as_closed_not_as_a_generic_error():
    client, _ = wired([""])  # server hung up
    try:
        client.get_obs()
        raise AssertionError("expected BridgeClosed")
    except BridgeClosed:
        pass
    assert client.broken
    dead, _ = wired([], sends_raise=ConnectionResetError("reset by peer"))
    try:
        dead.step({})
        raise AssertionError("expected BridgeClosed")
    except BridgeClosed:
        pass
    assert dead.broken


def test_unknown_scene_is_its_own_error_and_leaves_the_connection_usable():
    # A game that is still booting: the reply arrived in step, so the stream is fine and the caller should ask
    # again rather than rebuild anything.
    client, _ = wired(['{"type":"error","message":"unknown scene \'Level 0-1\'"}\n', '{"type":"obs"}\n'])
    try:
        client.reset(LEVEL)
        raise AssertionError("expected BridgeSceneUnknown")
    except BridgeSceneUnknown as exc:
        assert "unknown scene" in str(exc)
    assert not client.broken
    assert client.reset(LEVEL)["type"] == "obs"


def test_every_recoverable_error_is_still_a_bridge_error():
    # eval.py, bridge_test.py and campaign_check.py all catch BridgeError; none of them may start leaking.
    for cls in RECOVERABLE:
        assert issubclass(cls, BridgeError)


def test_an_ordinary_mod_error_is_untouched():
    client, _ = wired(['{"type":"error","message":"teleport needs a player"}\n'])
    try:
        client.teleport([0, 0, 0])
        raise AssertionError("expected BridgeError")
    except BridgeSceneUnknown:
        raise AssertionError("a plain mod error must not read as a booting game")
    except BridgeError as exc:
        assert "teleport" in str(exc)
    assert not client.broken


# -- the env: a fault ends the episode, not the run --------------------------------------------------


def run_forward(env, steps):
    for _ in range(steps):
        env.step(forward())


def test_a_step_timeout_truncates_the_episode_and_rebuilds_the_connection():
    env, fake = make_env(**FAST)
    env.reset()
    run_forward(env, 3)
    fake.fail_next_steps(BridgeTimeout("no reply within 60s"))

    obs, reward, terminated, truncated, info = env.step(forward())

    assert (terminated, truncated) == (False, True), "a bridge fault truncates, it never terminates"
    assert info["end_reason"] == "bridge_reset"
    assert reward == 0.0, "the lost step has no observation to grade, so it pays nothing"
    assert obs.shape == env.observation_space.shape
    assert fake.connects == 2, "the env reconnected to its own port"
    assert fake.configures == 2, "and re-sent its config"
    assert info["bridge_resets"] == 1
    # The env is usable afterwards: the next episode runs normally.
    env.reset()
    env.step(forward())


def test_a_dropped_socket_mid_rollout_is_survived_the_same_way():
    env, fake = make_env(**FAST)
    env.reset()
    run_forward(env, 2)
    fake.fail_next_steps(BridgeClosed("Connection closed by the game"))
    _, _, terminated, truncated, info = env.step(forward())
    assert (terminated, truncated, info["end_reason"]) == (False, True, "bridge_reset")


def test_a_reset_timeout_is_retried_against_a_rebuilt_connection():
    # Two of the three live crashes were exactly this: TimeoutError inside _campaign_reset at an episode
    # boundary. The episode now starts, one reconnect later, instead of the worker dying.
    env, fake = make_env(**FAST)
    fake.fail_next_resets(BridgeTimeout("no reply within 600s"), times=2)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert fake.connects == 3, "one connect, then one per failed reset"
    assert env._bridge_resets >= 1
    assert info["fresh_start"] == 1, "a rebuilt bridge always reloads the level from the top"


def test_a_booting_game_is_waited_out_rather_than_raised():
    # `unknown scene` is a game whose Addressables locators are still empty, not a bad level name. It used to
    # reach the trainer as a BridgeError and kill the whole vec env at env.reset().
    env, fake = make_env(**FAST)
    fake.fail_next_resets(BridgeSceneUnknown("unknown scene 'Level 0-1'"), times=5)
    env.reset()
    assert fake.connects == 1, "waiting for a boot needs no reconnect: the connection is fine"
    assert fake.resets, "the reset eventually went through"


def test_a_game_that_never_boots_finally_raises_instead_of_hanging():
    env, fake = make_env(bridge_backoff_s=0.0, unknown_scene_wait_s=0.0, bridge_retries=2)
    fake.fail_next_resets(BridgeSceneUnknown("unknown scene 'Level 0-1'"), times=50)
    try:
        env.reset()
        raise AssertionError("expected the retries to be bounded")
    except BridgeError:
        pass


def test_a_permanently_dead_bridge_raises_after_its_retries():
    env, fake = make_env(**FAST)
    env.reset()
    fake.fail_next_steps(BridgeTimeout("dead"))
    fake.fail_next_resets(BridgeTimeout("dead"), times=20)
    try:
        env.step(forward())
        raise AssertionError("expected BridgeError once the retries ran out")
    except BridgeError:
        pass


def test_a_death_respawn_that_loses_the_bridge_ends_the_episode():
    # _respawn is a reset request issued from inside step(); the third live crash was a timeout there. Carrying
    # the episode on across a reload would leave its start stats, best-run positions and gate approach all
    # describing a level load that no longer exists.
    env, fake = make_env(**FAST)
    env.reset()
    run_forward(env, 2)
    fake.kill_next = True
    fake.fail_next_resets(BridgeTimeout("no reply within 600s"))
    _, reward, terminated, truncated, info = env.step(forward())
    assert (terminated, truncated) == (False, True)
    assert info["end_reason"] == "bridge_reset"
    assert reward == 0.0


# -- the campaign trackers survive a recovery ---------------------------------------------------------


def test_a_recovery_reload_is_treated_as_a_fresh_level_load():
    # The reload really did put the level back at the top, so the trackers have to agree with that: nothing it
    # reveals may pay, and nothing already paid may be forgotten into a state where the level can never pay it
    # again.
    env, fake = make_env(fresh_start_prob=0.0, **FAST)
    env.reset()
    run_forward(env, 12)  # past the gate at z 15 and onto the checkpoint at z 20
    assert env.milestones.checkpoints_reached >= 1
    assert env.gates.gates_reached >= 1

    fake.fail_next_steps(BridgeTimeout("dead"))
    _, _, _, truncated, info = env.step(forward())
    assert truncated and info["end_reason"] == "bridge_reset"

    # Re-baselined on the reload, which is standing at spawn with nothing reached.
    assert env.milestones.checkpoints_reached == 0
    assert env.gates.gates_reached == 0
    assert env._fresh_start is True
    assert env._stuck_streak == 0


def test_the_milestones_of_a_recovered_level_pay_again_and_only_once():
    env, fake = make_env(fresh_start_prob=0.0, **FAST)
    env.reset()
    run_forward(env, 12)
    fake.fail_next_steps(BridgeTimeout("dead"))
    env.step(forward())

    env.reset()
    paid = {"checkpoint": 0.0, "gate": 0.0}
    for _ in range(12):
        _, _, _, _, info = env.step(forward())
        for name in paid:
            paid[name] += info["reward_parts"].get(name, 0.0)
    # The reloaded level pays its milestones again -- it is a new attempt at them -- and each exactly once.
    assert paid["checkpoint"] > 0.0, "a reloaded checkpoint must still be reachable and payable"
    assert paid["gate"] > 0.0
    again = {"checkpoint": 0.0, "gate": 0.0}
    for _ in range(3):
        _, _, _, _, info = env.step(forward())
        for name in again:
            again[name] += info["reward_parts"].get(name, 0.0)
    assert again["checkpoint"] == 0.0, "standing past a checkpoint must not keep paying for it"


def test_the_truncated_episodes_stats_describe_the_episode_that_was_lost():
    # `info` is built from the last good frame, not from the reload: grading it against a level that had just
    # started again would subtract this episode's start stats from a counter that went back to zero.
    env, fake = make_env(fresh_start_prob=0.0, **FAST)
    env.reset()
    run_forward(env, 12)
    gates_before = env.gates.gates_reached
    fake.fail_next_steps(BridgeTimeout("dead"))
    _, _, _, _, info = env.step(forward())
    assert info["gates_reached"] == gates_before
    assert info["checkpoints_level"] >= 1
    assert info["kills"] >= 0, "not a negative count from a reloaded stats counter"
    assert info["level"] == LEVEL


def test_the_exploration_archive_keeps_its_counts_across_a_recovery():
    # The per-game visit counts span episodes and carry the 1/sqrt(N) decay that makes the agent push outward;
    # only the episode's own first-entry set starts again.
    env, fake = make_env(fresh_start_prob=0.0, **FAST)
    env.reset()
    run_forward(env, 8)
    counted = len(env.archive.counts)
    assert counted > 0
    fake.fail_next_steps(BridgeTimeout("dead"))
    env.step(forward())
    assert len(env.archive.counts) >= counted, "a recovery must not wipe the visit counts"


def test_a_bridge_reset_is_not_a_completion_and_not_a_death():
    env, fake = make_env(fresh_start_prob=0.0, **FAST)
    env.reset()
    run_forward(env, 3)
    fake.fail_next_steps(BridgeClosed("gone"))
    _, _, _, _, info = env.step(forward())
    assert info["completed"] == 0
    assert info["level_seconds"] is None
    assert info["reward_parts"] == {}


def test_the_default_timeouts_are_not_the_mods_reset_timeout():
    # EpisodeController.resetTimeoutSeconds is 120 s. A client that also waits 120 s always gives up first,
    # because it started its clock before the mod started its own -- so the mod's graceful "reset timed out"
    # error reply could never arrive and was dead code.
    from ultrakill_ai.env import EnvConfig

    cfg = EnvConfig()
    assert cfg.reset_timeout_s > 120.0
    assert cfg.step_timeout_s < cfg.reset_timeout_s
    assert CHECKPOINT_ID  # keeps the shared fake level's constants imported and in step with this file


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
