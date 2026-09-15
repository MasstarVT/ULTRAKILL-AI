"""Gymnasium environment for ULTRAKILL (Cyber Grind and campaign levels)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ultrakill_ai.protocol import DEFAULT_PORT, BridgeClient
from ultrakill_ai.rewards import RewardConfig, compute_reward
from ultrakill_ai.routes import Route, RouteTracker, route_path
from ultrakill_ai.spaces import ObsLayout, action_space, decode_action, pack_observation

CYBERGRIND_SCENE = "Endless"


@dataclass
class EnvConfig:
    mode: str = "cybergrind"  # "cybergrind" or "campaign"
    level: str = "Level 0-1"  # campaign scene name
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    # Game speed / stepping
    frameskip: int = 4  # frames per decision (4 frames at 60 fps = 15 decisions per game second)
    fixed_fps: float = 60.0
    unlimited_fps: bool = True  # render as fast as possible so training runs faster than real time
    mute: bool = True
    block_human_input: bool = True
    windowed: bool = True  # run in a small window while training so it doesn't hold your mouse or screen
    window_width: int = 640
    window_height: int = 360

    # Episodes
    max_steps: int = 4500  # 5 minutes of game time at 15 decisions/s
    max_wave: int = 0  # Cyber Grind curriculum: end the episode once this many waves are cleared (0 = off)
    auto_enter_arena: bool = True  # Cyber Grind: put the player into the arena on reset so wave 1 starts
    reset_settle_frames: int = 10  # frames the player must be spawned before a reset completes
    checkpoint_resets: bool = False  # campaign: after a death, respawn at the checkpoint instead of reloading
    stuck_steps: int = 450  # campaign: end the episode after this many steps without route progress
    route_dir: str = "routes"

    layout: ObsLayout = field(default_factory=ObsLayout)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        layout = ObsLayout(**d.pop("layout", {}))
        rewards = RewardConfig(**d.pop("rewards", {}))
        return cls(**d, layout=layout, rewards=rewards)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UltrakillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        if self.cfg.mode not in ("cybergrind", "campaign"):
            raise ValueError(f"Unknown mode {self.cfg.mode!r}")

        self.observation_space = self.cfg.layout.space()
        self.action_space = action_space()

        self.client = BridgeClient(self.cfg.host, self.cfg.port)
        self._connected = False

        self.route_tracker: RouteTracker | None = None
        if self.cfg.mode == "campaign":
            path = route_path(self.cfg.route_dir, self.cfg.level)
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"No route for {self.cfg.level} at {path}. Record one with scripts/record_route.py first."
                )
            self.route_tracker = RouteTracker(Route.load(path))

        self._raw: dict[str, Any] = {}
        self._enemy_max_health: dict[int, float] = {}
        self._steps = 0
        self._steps_since_progress = 0
        self._last_end_reason = ""
        self._reset_seconds = 0.0

    @property
    def scene(self) -> str:
        return CYBERGRIND_SCENE if self.cfg.mode == "cybergrind" else self.cfg.level

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self.client.connect()
        mod_layout = self.cfg.layout.mod_config()
        # Ask the mod for more enemies than the policy sees so damage rewards aren't missed.
        mod_layout["max_enemies"] = max(32, self.cfg.layout.max_enemies)
        self.client.configure(
            frameskip=self.cfg.frameskip,
            fixed_fps=self.cfg.fixed_fps,
            unlimited_fps=self.cfg.unlimited_fps,
            mute=self.cfg.mute,
            block_human_input=self.cfg.block_human_input,
            reset_settle_frames=self.cfg.reset_settle_frames,
            windowed=self.cfg.windowed,
            window_width=self.cfg.window_width,
            window_height=self.cfg.window_height,
            **mod_layout,
        )
        self._connected = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._ensure_connected()

        checkpoint = (
            self.cfg.mode == "campaign" and self.cfg.checkpoint_resets and self._last_end_reason == "death"
        )
        start = time.perf_counter()
        self._raw = self.client.reset(self.scene, checkpoint=checkpoint)
        if self.cfg.mode == "cybergrind" and self.cfg.auto_enter_arena:
            self._raw = self._enter_arena(self._raw)
        self._reset_seconds = time.perf_counter() - start
        self._enemy_max_health = {}
        self._track_enemies(self._raw)
        self._steps = 0
        self._steps_since_progress = 0
        self._last_end_reason = ""

        if self.route_tracker is not None and self._raw.get("player"):
            self.route_tracker.start(self._raw["player"]["pos"])

        return self._pack(self._raw), self._info(self._raw)

    def step(self, action):
        prev = self._raw
        cur = self.client.step(decode_action(action))
        self._raw = cur
        self._steps += 1
        self._track_enemies(cur)

        route_gain = 0
        player = cur.get("player")
        if self.route_tracker is not None and player:
            route_gain = self.route_tracker.update(player["pos"])
            self._steps_since_progress = 0 if route_gain else self._steps_since_progress + 1

        stuck = self.cfg.mode == "campaign" and self._steps_since_progress >= self.cfg.stuck_steps
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, route_gain, stuck)

        terminated, truncated, reason = False, False, ""
        stats = cur.get("stats", {})
        if player is None or player["dead"]:
            terminated, reason = True, "death"
        elif cur.get("scene") != self.scene:
            terminated, reason = True, "scene_changed"
        elif self.cfg.mode == "campaign" and stats.get("level_complete"):
            terminated, reason = True, "level_complete"
        elif self.cfg.mode == "cybergrind" and self.cfg.max_wave and (cur.get("cybergrind") or {}).get("wave", 0) > self.cfg.max_wave:
            terminated, reason = True, "max_wave"
        elif stuck:
            truncated, reason = True, "stuck"
        elif self._steps >= self.cfg.max_steps:
            truncated, reason = True, "max_steps"
        self._last_end_reason = reason

        info = self._info(cur)
        info["reward_parts"] = reward.parts
        if reason:
            info["end_reason"] = reason
            info["reset_seconds"] = self._reset_seconds
        return self._pack(cur), float(reward.total), terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self.client.close()
            self._connected = False

    # ------------------------------------------------------------------

    def _enter_arena(self, raw: dict[str, Any]) -> dict[str, Any]:
        """The Cyber Grind spawn is a ledge above the arena; waves only start once the player enters the grid's trigger.

        Teleports straight into the trigger when the mod reports it (fast, so parallel games aren't held
        up by resets), otherwise walks off the ledge. Times are in game seconds so this doesn't depend on
        frameskip.
        """
        steps_per_second = self.cfg.fixed_fps / self.cfg.frameskip
        trigger = (raw.get("cybergrind") or {}).get("start_trigger")
        if trigger:
            raw = self.client.teleport(trigger["center"])
            for _ in range(int(2 * steps_per_second)):
                raw = self.client.step({})
                if (raw.get("cybergrind") or {}).get("wave", 0) >= 1:
                    return raw
        forward = {"move": [0, 1]}
        for _ in range(int(2.7 * steps_per_second)):
            raw = self.client.step(forward)
        raw = self.client.step({"move": [0, 1], "buttons": ["jump"]})
        for _ in range(int(10 * steps_per_second)):
            if (raw.get("cybergrind") or {}).get("wave", 0) >= 1:
                return raw
            raw = self.client.step(forward)
        raise RuntimeError("Failed to enter the Cyber Grind arena after reset")

    def _track_enemies(self, raw: dict[str, Any]) -> None:
        for e in raw.get("enemies", []):
            if e["health"] > self._enemy_max_health.get(e["id"], 0.0):
                self._enemy_max_health[e["id"]] = e["health"]

    def _pack(self, raw: dict[str, Any]) -> np.ndarray:
        waypoint = self.route_tracker.waypoint if self.route_tracker else None
        progress = self.route_tracker.progress if self.route_tracker else 0.0
        return pack_observation(raw, self.cfg.layout, self._enemy_max_health, waypoint, progress)

    def _info(self, raw: dict[str, Any]) -> dict[str, Any]:
        stats = raw.get("stats", {})
        player = raw.get("player") or {}
        info = {
            "kills": stats.get("kills", 0),
            "style": stats.get("style", 0),
            "hp": player.get("hp", 0),
            "wave": (raw.get("cybergrind") or {}).get("wave", 0),
        }
        if self.route_tracker is not None:
            info["route_progress"] = self.route_tracker.progress
        return info
