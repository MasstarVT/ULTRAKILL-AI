"""Gymnasium environment for ULTRAKILL (Cyber Grind and campaign levels)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ultrakill_ai.campaign import (
    CAMPAIGN_LEVELS_SHIPPED,
    SUBGOAL_ALTAR,
    SUBGOAL_ITEM,
    ExplorationArchive,
    GateProgress,
    MilestoneTracker,
    PathProgress,
    altar_aim_point,
    choose_fresh_start,
    choose_level,
    compute_rank,
    read_curriculum,
    safe_name,
    save_best_run,
)
from ultrakill_ai.protocol import DEFAULT_PORT, BridgeClient
from ultrakill_ai.rewards import CampaignStep, RewardConfig, aim_errors, compute_reward, horizon_elevation
from ultrakill_ai.spaces import PITCH_BINS, YAW_BINS, ObsLayout, action_space, decode_action, pack_observation, yaw_frame

CYBERGRIND_SCENE = "Endless"
# Per-episode info the campaign Monitor records (scripts/train.py); every key is in every campaign info.
CAMPAIGN_INFO_KEYS = ("kills", "style", "deaths", "completed", "fresh_start", "level_seconds",
                      "checkpoints_level", "cells_new", "exit_dist_min", "oob_frac",
                      "gates_reached", "wedged_steps", "level_started", "look_gate_frac", "slide_forced_frac")

YAW_CAP = max(abs(b) for b in YAW_BINS)  # 90 degrees per decision, the widest look bin
PITCH_CAP = max(abs(b) for b in PITCH_BINS)  # 20 degrees per decision
MODE1_PITCH_LIMIT = 85.0  # inside the game's own +-90 clamp (ActionInjector.ApplyLook)
ARCHIVE_CHECK_EVERY = 500  # steps between exploration-archive schedule checks (the save itself costs ~16 ms)
DEFAULT_WEDGE_HOLD_S = 3.0  # wedge_seconds 0 turns the episode end off; counting still uses this hold


def _known_fields(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Keeps the keys that are fields of dataclass `cls`, so configs saved with retired settings still load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


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
    # Per-level override of `max_steps`, keyed by scene name, for a `levels` ladder whose rungs are not the same
    # size (1-1 is ~4x 0-1). Empty -- the default -- means every level uses `max_steps`, unchanged. A level that
    # is not named here falls back to `max_steps`, so only the exceptions have to be listed.
    max_steps_per_level: dict[str, int] = field(default_factory=dict)
    max_wave: int = 0  # Cyber Grind curriculum: end the episode once this many waves are cleared (0 = off)
    hard_reset_above_wave: int = 5  # reload the arena once waves get this far, so training keeps seeing early waves
    auto_enter_arena: bool = True  # Cyber Grind: put the player into the arena on reset so wave 1 starts
    reset_settle_frames: int = 10  # frames the player must be spawned before a reset completes
    end_episode_on_death: bool = True  # False (with soft_death) keeps the run going after a death, so episodes are a
    # fixed slice of time: total reward then measures kills per minute instead of rewarding hiding
    soft_death: bool = True  # Cyber Grind: lethal hits heal instead of killing; the episode still ends with the death
    # penalty, but the next one continues in the same arena without a scene reload
    render: bool = False  # the agent never sees pixels; turning cameras off saves CPU/GPU
    pitch_limit_deg: float = 0.0  # keep the camera within ±this many degrees of level (0 = off). Without it the
    # policy drifted to the +90° clamp and stared at the sky, so no yaw ever put an enemy near the crosshair

    # Campaign (docs/superpowers/specs/2026-09-16-campaign-foundation-design.md)
    # Multi-level curriculum. Empty `levels` is the single-level setting above and today's behaviour exactly:
    # nothing below is read, no curriculum file is opened and the level never changes.
    levels: list[str] = field(default_factory=list)  # ordered ladder; levels[0] is always unlocked and the start
    curriculum_path: str = ""  # runs/<run>/curriculum.json, written by ProgressCallback ("" = no curriculum)
    unlock_rate: float = 0.5  # fresh completion rate a level needs before the next one unlocks
    unlock_window: int = 20  # ... over at least this many of its own fresh episodes
    # Safety valve (0 = off): a level also opens its successor after this many of its own CUMULATIVE fresh
    # episodes, whatever its rate, so one level the policy cannot crack does not block the whole campaign.
    unlock_after_fresh_episodes: int = 0
    level_weight_floor: float = 0.1  # a mastered level keeps this much of the sampling weight, so it is not forgotten
    difficulty: int = -1  # difficulty the game reads while the AI has control (3 = Violent, -1 = leave the game's own)
    unlock_all_gear: bool = False  # every weapon and variant while the AI has control, in memory only
    fresh_start_prob: float = 0.2  # chance of a fresh level load when a checkpoint respawn would also do
    stuck_seconds: float = 45.0  # game seconds without progress (milestone, new cell, shorter path) before truncating
    stuck_repeats: int = 3  # episodes in a row stuck at the same checkpoint before a fresh load is forced
    cell_size: float = 4.0  # metres per exploration cell
    max_locked_skip_s: float = 120.0  # longest input lock (landing, cutscene) stepped through without the policy
    explore_dir: str = ""  # folder for the exploration archive, so a resumed run keeps its visit counts ("" = memory only)
    best_runs_dir: str = ""  # folder for the fastest fresh-start completion of each level ("" = off)
    # Route gates (campaign.gates): the door-graph ladder GateProgress walks.
    gate_reach_m: float = 8.0  # horizontal radius at which a gate counts as reached (2x while it is open)
    gate_reach_v_m: float = 6.0  # vertical half-height of the same test, so a roof over a door is not "reached"
    gate_min_gain_m: float = 0.5  # metres of new best closeness below which gate_approach pays nothing
    gate_hops_min_frac: float = 0.5  # share of gates that must carry `hops` before the ladder is trusted at all
    # Skull carry. 0 disables both carry-protection rules; they are inert on any level with no ItemPlaceZone.
    subgoal_punch_range_m: float = 4.0  # Punch.ActiveFrame's own 4 m reach: inside it a punch can pick up or place
    camera_height_m: float = 0.9  # metres from player.pos up to the camera, where every ray starts (see _eye)
    target_kind_slots: bool = False  # repurpose the target's `open`/`locked` slots as "is an item"/"is an altar"
    # The absorbing slowMode/heavyFall movement state: airborne forever, stamina frozen, a 0.477 m/s creep.
    wedge_seconds: float = 3.0  # game seconds wedged before the episode ends (0 = no end, the steps are still counted)
    wedge_creep_mps: float = 1.5  # movement below this counts as not moving, in the wedge test only
    slide_min_hold: int = 0  # decisions slide stays held once pressed (0 = off; an experiment, see the design spec)
    archive_save_steps: int = 20000  # env lifetime steps between exploration-archive saves
    archive_save_seconds: float = 600.0  # ... or this much wall time, whichever comes first

    layout: ObsLayout = field(default_factory=ObsLayout)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        layout = ObsLayout(**_known_fields(ObsLayout, d.pop("layout", None) or {}))
        rewards = RewardConfig(**_known_fields(RewardConfig, d.pop("rewards", None) or {}))
        return cls(**_known_fields(cls, d), layout=layout, rewards=rewards)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BEHAVIOUR_KEYS = ("steps", "firing", "on_target", "firing_on_target", "enemy_visible", "close", "angle_sum", "yaw_err_sum", "dist_sum", "yaw_sum",
                  "pitch_steps", "pitch_sum", "pitch_signed_sum", "look_up_sum", "elev_steps", "elev_sum", "elev_abs_sum", "elev_over15", "pitch_err_sum",
                  "yaw_track", "yaw_track_n", "pitch_track", "pitch_track_n",
                  "look_free", "look_enemy", "look_gate", "slide_forced")


def clamp_pitch_command(current_pitch: float, pitch_cmd: float, limit: float) -> float:
    """Trims a pitch command so the camera ends within ±limit degrees of level (snapping back if it is outside)."""
    target = max(-limit, min(limit, current_pitch + pitch_cmd))
    return target - current_pitch


class UltrakillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        if self.cfg.mode not in ("cybergrind", "campaign"):
            raise ValueError(f"Unknown mode {self.cfg.mode!r}")
        campaign = self.cfg.mode == "campaign"
        unknown = [lv for lv in self.cfg.levels if lv not in CAMPAIGN_LEVELS_SHIPPED]
        if unknown:
            # Caught here rather than at the first reset, where it would strand a worker waiting for a scene the
            # game cannot load. `Level 9-1` and `Level 9-2` ship no scene bundle in this build.
            raise ValueError(f"levels contains scenes this game build cannot load: {unknown}")
        # The level is mutable from here on: a curriculum run changes it at a fresh load and nowhere else.
        self.level = (self.cfg.levels[0] if self.cfg.levels else self.cfg.level) if campaign else self.cfg.level
        if self.cfg.layout.campaign != campaign:
            # The layout follows the mode (Cyber Grind 448 inputs, campaign 479). Replaced, not edited in place:
            # train.py builds each game's config with dataclasses.replace, so they all share one layout object.
            self.cfg = replace(self.cfg, layout=replace(self.cfg.layout, campaign=campaign))

        self.observation_space = self.cfg.layout.space()
        # Campaign only: the look mode is appended as a 12th dimension, so Cyber Grind checkpoints stay loadable.
        self.action_space = action_space(campaign)

        self.client = BridgeClient(self.cfg.host, self.cfg.port)
        self._connected = False

        self._raw: dict[str, Any] = {}
        self._enemy_max_health: dict[int, float] = {}
        self._steps = 0
        self._steps_since_progress = 0
        self._last_end_reason = ""
        self._reset_seconds = 0.0
        self._episode_start_seconds: float | None = None  # campaign: the mod's own clock at this episode's reset
        self._deaths = 0
        self._behaviour = dict.fromkeys(BEHAVIOUR_KEYS, 0)
        self._arena_spawn: list[float] | None = None
        self._episode_start_stats: dict[str, Any] = {}

        # Campaign state. The exploration archive counts, per game and per level, how many earlier episodes
        # entered each cell, so the novelty reward fades where this game has already been. A curriculum run keeps
        # one archive per level it has visited: `archive` is the current level's, `_archives` holds them all so a
        # switch back does not re-read from disk, and every one of them is saved.
        if campaign and self.cfg.explore_dir:
            self.archive = ExplorationArchive.load(self._archive_path(self.level), self.cfg.cell_size)
        else:
            self.archive = ExplorationArchive(self.cfg.cell_size)
        self._archives: dict[str, ExplorationArchive] = {self.level: self.archive}
        self._curriculum: dict = {}  # last curriculum table read successfully, kept as the fallback
        self._curriculum_warned = False
        self.milestones = MilestoneTracker()
        self.gates = GateProgress(self.cfg.gate_reach_m, self.cfg.gate_reach_v_m, self.cfg.gate_min_gain_m,
                                  self.cfg.gate_hops_min_frac, self.cfg.target_kind_slots)
        self.path_progress = PathProgress()
        self._rng = random.Random()
        self._stuck_streak = 0  # episodes in a row that ended stuck at the same current checkpoint
        self._stuck_checkpoint: str | None = None
        self._fresh_start = False  # this episode began with a fresh level load
        self._positions: list[list[float]] = []  # fresh-start episodes only, for best runs
        self._cells_new = 0
        self._oob_steps = 0  # steps with no ground under the player: off the map, or in a fall
        self._exit_dist_min = math.inf
        self._episodes = 0
        self._start_checkpoint: str | None = None  # the checkpoint this episode began at (None on a fresh load)
        self._level_started = False  # campaign.level_started was true at some point this episode
        self._last_pos: list[float] | None = None  # so end_pos survives a final frame without a player
        # The wedge detector (see _wedge_now). CREEP is per decision, so it is frameskip-independent.
        steps_per_second = self.cfg.fixed_fps / max(1, self.cfg.frameskip)
        self._creep_m = self.cfg.wedge_creep_mps / steps_per_second
        self._wedge_hold = max(1, int(round((self.cfg.wedge_seconds or DEFAULT_WEDGE_HOLD_S) * steps_per_second)))
        self._wedge_ends = self.cfg.wedge_seconds > 0
        self._wedge_run = 0  # consecutive wedged decisions right now
        self._wedged_steps = 0  # steps inside runs that reached the hold, credited retroactively
        self._slide_latch = 0  # decisions slide is still held for (slide_min_hold)
        # Lifetime steps never reset, so the archive schedule does not depend on episode boundaries: the whole
        # ground run saved nothing because episodes are ~3900 decisions and no worker reached 20 of them.
        self._lifetime_steps = 0
        self._last_save_steps = 0
        self._last_save_time = time.monotonic()

    @property
    def scene(self) -> str:
        return CYBERGRIND_SCENE if self.cfg.mode == "cybergrind" else self.level

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
            soft_death=self.cfg.soft_death and self.cfg.mode == "cybergrind",
            render=self.cfg.render,
            windowed=self.cfg.windowed,
            window_width=self.cfg.window_width,
            window_height=self.cfg.window_height,
            # Sent before the first reset: enemies and GunSetter read both when the level loads.
            difficulty=self.cfg.difficulty,
            unlock_all_gear=self.cfg.unlock_all_gear,
            **mod_layout,
        )
        self._connected = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._ensure_connected()

        start = time.perf_counter()
        if self.cfg.mode == "campaign":
            self._raw = self._campaign_reset()
        elif self._can_soft_reset():
            self._raw = self._soft_reset()
        else:
            self._raw = self.client.reset(self.scene, checkpoint=False)
            if self.cfg.auto_enter_arena:
                self._raw = self._enter_arena(self._raw)
            if self._raw.get("player"):
                self._arena_spawn = list(self._raw["player"]["pos"])
        self._reset_seconds = time.perf_counter() - start
        self._episode_start_stats = dict(self._raw.get("stats", {}))
        # Campaign: baseline for episode_seconds, the mod's own clock rather than a step count (see step()).
        self._episode_start_seconds = (self._raw.get("campaign") or {}).get("seconds")
        self._enemy_max_health = {}
        self._track_enemies(self._raw)
        self._steps = 0
        self._steps_since_progress = 0
        self._deaths = 0
        self._behaviour = dict.fromkeys(BEHAVIOUR_KEYS, 0)
        self._last_end_reason = ""
        self._wedge_run = 0
        self._wedged_steps = 0
        self._slide_latch = 0

        if self.cfg.mode == "campaign":
            self.archive.start_episode()
            self.path_progress.reset()
            self._positions = []
            self._cells_new = 0
            self._oob_steps = 0
            self._exit_dist_min = math.inf
            self._start_checkpoint = self._current_checkpoint(self._raw)
            self._level_started = bool((self._raw.get("campaign") or {}).get("level_started"))
            player = self._raw.get("player")
            if player:
                self._last_pos = list(player["pos"])
                ground = self._ground_point(self._raw)
                if ground is not None:
                    self.archive.visit(ground)  # the spawn cell is entered, but pays nothing
                if self._fresh_start:
                    self._positions.append(self._rounded(player["pos"]))
            # _campaign_reset has already run new_level_load or mark_paid; the episode's own approach baseline is
            # cleared here, and retarget runs before the observation is packed so slot 452 reads 1.0 on the first
            # decision of every episode and look mode 2 is never dead on step 1.
            self.gates.reset_episode()
            self.gates.retarget(self._raw.get("campaign"), player["pos"] if player else None)

        return self._pack(self._raw), self._info(self._raw)

    def step(self, action):
        prev = self._raw
        command = decode_action(action)
        # The look mode is resolved here, against the observation the policy acted on, and popped: the wire
        # `action` message is unchanged and the mod never sees it.
        look_mode = int(command.pop("look_mode", 0))
        raw_pitch_cmd = command["look"][1]
        # Skull carry (§6.7 of the multi-level/skull spec): within punch range of a sub-goal the env takes the
        # camera, because the punch that picks up or places is a 4 m raycast along camera forward and the policy
        # volunteers a useful aim on ~3% of steps. Inert without a sub-goal, so 0-1 is untouched.
        decisive = self._near_subgoal(prev, (SUBGOAL_ITEM, SUBGOAL_ALTAR))
        if decisive:
            look_mode = 2
        self._protect_carry(prev, command)
        resolved = (self._look_at_enemy(prev) if look_mode == 1
                    else self._look_at_target(prev, wide=decisive) if look_mode == 2 else None)
        # A mode that found nothing to aim at behaves exactly as mode 0, so mode 0 is what applied this step. The
        # diagnostics record the applied mode, never the requested one: mode 1 falls back whenever nothing is
        # visible (~30% of steps at the measured enemy_visible_frac) and mode 2 falls back on every step of an
        # unordered level or an older mod, and counting those as aimed would report the yaw and pitch heads as
        # causally inert on steps where they alone drove the camera -- the share R6 says to read the
        # per-dimension entropy against -- and would throw away their tracking samples.
        applied_mode = look_mode if resolved is not None else 0
        if resolved is not None:
            command["look"] = resolved
        elif self.cfg.pitch_limit_deg and prev.get("player"):
            command["look"][1] = clamp_pitch_command(prev["player"]["pitch"], command["look"][1], self.cfg.pitch_limit_deg)
        self._hold_slide(prev, command)
        self._note_behaviour(prev, command, raw_pitch_cmd, applied_mode)
        campaign = self.cfg.mode == "campaign"
        cur = self.client.step(command)
        if campaign:
            cur = self._skip_locked(cur)
        self._raw = cur
        self._steps += 1
        self._lifetime_steps += 1
        if self._lifetime_steps % ARCHIVE_CHECK_EVERY == 0:
            self._save_archive_on_schedule()
        self._track_enemies(cur)

        player = cur.get("player")
        prev_player = prev.get("player") or {}
        died = player is None or player["dead"] or (
            player.get("soft_deaths", 0) > prev_player.get("soft_deaths", player.get("soft_deaths", 0))
        )
        # Milestones, novelty, route gates and path progress are measured before the reward, from the frame the
        # policy caused; `prev` is needed as well, because kills and style reset the stuck clock.
        campaign_step = self._campaign_progress(prev, cur) if campaign else None
        wedged = campaign and self._note_wedge(prev, cur)
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, died=died,
                                campaign=campaign_step, buttons=command["buttons"])

        completed = bool(campaign and (cur.get("stats", {}).get("level_complete") or (cur.get("campaign") or {}).get("level_over")))
        if campaign:
            if died and player is not None and not completed:
                # A death does not end a campaign episode: the penalty is paid above, then the player respawns at
                # the checkpoint and the level clock keeps running, as in real play.
                cur = self._respawn()
                self._raw = cur
                player = cur.get("player")
                self._deaths += 1
                died = False
        elif died and not self.cfg.end_episode_on_death and player is not None and not player["dead"]:
            # Soft death inside a timed episode: stay in the run, but get out of the pit that killed us.
            if player.get("soft_death_instakill") and self._arena_spawn is not None:
                x, y, z = self._arena_spawn
                cur = self.client.teleport([x, y + 10.0, z])
                self._raw = cur
                player = cur.get("player")
                self._track_enemies(cur)
            self._deaths += 1
            died = False

        stuck = campaign and self._steps_since_progress >= int(self.cfg.stuck_seconds * self.cfg.fixed_fps / self.cfg.frameskip)
        terminated, truncated, reason = False, False, ""
        stats = cur.get("stats", {})
        if completed:
            # Graded before death and scene change, because a completion arrives as the level unloads: the same
            # frame can have no player (which reads as a death) and a new scene name. Grading either of those
            # first would pay 0 instead of `level_complete`, leave info["completed"] at 0 and never save the best
            # run -- silently losing the one event this whole run exists to produce.
            terminated, reason = True, "level_complete"
        elif died:
            terminated, reason = True, "death"
        elif cur.get("scene") != self.scene:
            terminated, reason = True, "scene_changed"
        elif self.cfg.mode == "cybergrind" and self.cfg.max_wave and (cur.get("cybergrind") or {}).get("wave", 0) > self.cfg.max_wave:
            terminated, reason = True, "max_wave"
        elif wedged and self._wedge_ends:
            # A distinct reason, never folded into "stuck", so the dashboard can tell "cannot move" from "moving
            # but making no progress". It is a safety net: once the mod's un-wedge lands this should almost never
            # fire, and if it keeps firing no reward change will help.
            truncated, reason = True, "wedged"
        elif stuck:
            truncated, reason = True, "stuck"
        elif self._steps >= self._max_steps():
            truncated, reason = True, "max_steps"
        self._last_end_reason = reason

        info = self._info(cur)
        info["reward_parts"] = reward.parts
        if reason:
            info["end_reason"] = reason
            info["reset_seconds"] = self._reset_seconds
            seconds = self._steps * self.cfg.frameskip / self.cfg.fixed_fps
            if campaign:
                # The step count excludes every frame _skip_locked steps through (the opening drop, cutscenes,
                # each respawn), which can understate elapsed time by minutes across a death-heavy episode. The
                # mod's own clock counts all of it, so prefer it here; fall back to the step count only when the
                # campaign block (and so this episode's start reading of it) was missing.
                camp_seconds = (cur.get("campaign") or {}).get("seconds")
                if camp_seconds is not None and self._episode_start_seconds is not None:
                    seconds = camp_seconds - self._episode_start_seconds
            info["episode_seconds"] = seconds
            info["kills_per_min"] = info["kills"] / seconds * 60.0 if seconds > 0 else 0.0
            if campaign:
                self._end_campaign_episode(cur, reason, info)
        return self._pack(cur), float(reward.total), terminated, truncated, info

    def close(self) -> None:
        try:
            self._save_archive()
        finally:
            # Release the game even if the archive could not be written, so it never stays in lockstep.
            if self._connected:
                self.client.close()
                self._connected = False

    # ------------------------------------------------------------------

    def _can_soft_reset(self) -> bool:
        """After a soft death or a timeout, Cyber Grind can continue in place instead of reloading the scene."""
        player = self._raw.get("player") if self._raw else None
        return (
            self.cfg.mode == "cybergrind"
            and self.cfg.soft_death
            and self._last_end_reason in ("death", "max_steps")
            and self._raw.get("scene") == self.scene
            and player is not None
            and not player["dead"]
            and 1 <= (self._raw.get("cybergrind") or {}).get("wave", 0) <= (self.cfg.hard_reset_above_wave or 10 ** 9)
        )

    def _soft_reset(self) -> dict[str, Any]:
        player = self._raw["player"]
        if player.get("soft_death_instakill") and self._arena_spawn is not None:
            # Died in a pit: put the player back above the arena so they don't keep falling.
            x, y, z = self._arena_spawn
            return self.client.teleport([x, y + 10.0, z])
        return self.client.get_obs()

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

    def _look_at_enemy(self, prev: dict[str, Any]) -> list[float] | None:
        """Look mode 1: turn toward the nearest visible enemy the policy can actually see.

        Only the first `max_enemies` entries count: the env asks the mod for 32 enemies so damage rewards are not
        missed, but the observation carries 8, and this must not aim at something that is not in it. `rel` is the
        enemy centre in camera space, so un-pitching by the current pitch gives the player's yaw frame. `visible`
        is line of sight with no frustum test, so the target can be behind the player and this turns 180 degrees
        toward it -- intended (turn and fight), not a bug to filter out.

        Returns None (behave exactly as mode 0) with no player, no visible enemy in the first 8, or a degenerate
        `rel`. Mode 1 deliberately ignores `pitch_limit_deg` -- an enemy overhead in an arena is exactly the case
        the band blocks -- but stays inside the game's own +-90 clamp.
        """
        player = prev.get("player")
        if not player:
            return None
        for enemy in (prev.get("enemies") or ())[: self.cfg.layout.max_enemies]:
            if not enemy.get("visible"):
                continue
            x, y, z = enemy["rel"]
            if math.sqrt(x * x + y * y + z * z) <= 1e-6:
                return None
            p = player["pitch"]
            cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
            xf, yf, zf = x, y * cp + z * sp, -y * sp + z * cp  # yaw frame: +x right, +y up, +z forward
            yaw_cmd = max(-YAW_CAP, min(YAW_CAP, math.degrees(math.atan2(xf, zf))))
            elevation = math.degrees(math.atan2(yf, math.hypot(xf, zf)))
            pitch_cmd = max(-PITCH_CAP, min(PITCH_CAP, elevation - p))
            pitch_cmd = max(-MODE1_PITCH_LIMIT, min(MODE1_PITCH_LIMIT, p + pitch_cmd)) - p
            return [yaw_cmd, pitch_cmd]
        return None

    def _max_steps(self) -> int:
        """`max_steps`, or this level's own cap when `max_steps_per_level` names it. Read per step, not cached,
        because a `levels` ladder switches `self.level` between episodes."""
        return int(self.cfg.max_steps_per_level.get(self.level, self.cfg.max_steps))

    def _eye(self, player: dict[str, Any]) -> tuple[float, float, float]:
        """Where the camera is: `cc.GetDefaultPos()`, `camera_height_m` above `player.pos`.

        Everything a punch or a shot does starts here, not at the player transform. `Punch.ActiveFrame`
        (`decompiled/Punch.cs:562`) rays from `cc.GetDefaultPos()` along camera forward for 4 m, so both the aim
        and the reach test have to be measured from this point. `player.pos` is `NewMovement.transform.position`,
        which sits at the feet; the mod reports no camera position of its own, so the offset is a constant here.
        Measured in game on 1-1's red pedestal, 2026-09-17: at 1.57 m a command pitch of -3.0 picks the skull up
        and +27.0 -- what aiming from `player.pos` asks for -- does not; at 2.71 m, +8.0 works and +11.3 does
        not. Both solve to 0.88-0.90 m.

        The error this removes is entirely a short-range one, and it is the whole answer at punch range:
        atan(0.9/r) is 29.8 degrees at 1.57 m, 5.1 at 10 m, 1.7 at 30 m. So a gate 10-30 m off moves by a couple
        of degrees, well inside the aim tolerance, and moves in the correct direction -- a camera 0.9 m above the
        feet really does have to look slightly DOWN at a door sill at its own feet's height.

        A future mod field (`player.cam_pos`) would make this exact for a crouched or sliding player, whose
        camera is lower; until then `camera_height_m` overrides it and 0 restores the old feet-relative aim.
        """
        x, y, z = player["pos"]
        return (x, y + self.cfg.camera_height_m, z)

    def _look_at_target(self, prev: dict[str, Any], wide: bool = False) -> list[float] | None:
        """Look mode 2: turn toward the route target, the same object packed into the observation's target slots.

        Movement stays camera-relative, as the game computes it, so turning toward a gate also turns "forward"
        toward it -- which is the point. Returns None (mode 0) without a player or a target. `pitch_limit_deg` 0
        means "off" everywhere else in this codebase, so the band falls back to the game's own clamp rather than
        welding the camera level.

        The aim is taken from the camera (`_eye`), not from `player.pos`: at punch range the 0.9 m between them
        is a ~30 degree pitch error and the ray passes clean over the target, which is why the skull leg could
        never place before this. Yaw is unaffected -- only the height differs.

        `wide` takes the same exemption look mode 1 already takes for an enemy overhead: a sub-goal a metre away
        at floor level needs a steeper look than the campaign band's 45 degrees, so inside punch range the band
        is the game's own clamp instead and the ray can actually reach the cube.
        """
        player, target = prev.get("player"), self.gates.target
        if not player or not target or not target.get("pos"):
            return None
        pos, tp = self._eye(player), target["pos"]
        gx, gy, gz = yaw_frame((tp[0] - pos[0], tp[1] - pos[1], tp[2] - pos[2]), player["yaw"])
        if math.sqrt(gx * gx + gy * gy + gz * gz) <= 1e-6:
            return None
        p = player["pitch"]
        band = MODE1_PITCH_LIMIT if wide else (self.cfg.pitch_limit_deg or MODE1_PITCH_LIMIT)
        yaw_cmd = max(-YAW_CAP, min(YAW_CAP, math.degrees(math.atan2(gx, gz))))
        elevation = math.degrees(math.atan2(gy, math.hypot(gx, gz)))
        target_pitch = max(-band, min(band, elevation))
        return [yaw_cmd, max(-PITCH_CAP, min(PITCH_CAP, target_pitch - p))]

    def _near_subgoal(self, prev: dict[str, Any], kinds: tuple[str, ...]) -> bool:
        """Whether the current target is a sub-goal of one of `kinds` and the player is within punch range of it.

        Measured from the camera (`_eye`), because the 4 m the punch reaches is 4 m from `cc.GetDefaultPos()`.
        """
        if self.cfg.mode != "campaign" or self.cfg.subgoal_punch_range_m <= 0:
            return False
        target, player = self.gates.target, prev.get("player")
        if not target or not player or target.get("subgoal") not in kinds or not target.get("pos"):
            return False
        return math.dist(self._eye(player), target["pos"]) <= self.cfg.subgoal_punch_range_m

    def _near_filled_altar(self, prev: dict[str, Any]) -> bool:
        """Whether a solved altar is within punch range: punching one takes the skull back out."""
        player = prev.get("player")
        if not player:
            return False
        eye = self._eye(player)
        for altar in ((prev.get("campaign") or {}).get("altars") or ()):
            if (isinstance(altar, dict) and altar.get("filled") and altar.get("pos")
                    and math.dist(eye, altar_aim_point(altar)) <= self.cfg.subgoal_punch_range_m):
                return True
        return False

    def _protect_carry(self, prev: dict[str, Any], command: dict[str, Any]) -> None:
        """Drops a punch that can only undo a skull puzzle. Never adds one; the policy still chooses to press.

        Two ways a punch destroys progress, both from `Punch`:
          - `ActiveStart` THROWS whatever is held when the active frame did not place it. The live policy
            presses punch on ~35% of decisions, about five a second, so an unprotected carry across 1-1's 134 m
            red leg survives a fraction of a second;
          - `AltHit`'s not-holding branch is `ForceHold` on whatever `ItemIdentifier` it hits, including one
            resting in a FILLED altar, and `ForceHold` re-runs `ItemPlaceZone.CheckItem`, whose empty branch
            calls `Close()` on the doors that altar had opened. So a punch next to a solved altar re-locks the
            gate it just opened.

        So exactly two presses are kept: the one that places (carrying, within range of the altar we are heading
        for) and the one that picks up (not carrying, within range of the item we are heading for -- which may
        itself be sitting in some OTHER puzzle's filled altar, which is why that case is checked first). Every
        other press is dropped while a real carry is in progress or while a solved altar is in reach.

        **A carry is a held item THE CURRENT SUB-GOAL is carrying to an altar**, not merely `any(items[].held)`
        and not "some live unfilled altar somewhere accepts it" either. Protection and release must come from
        one source or the button can be taken away and never given back, and `GateProgress.target` is that
        source: a press is kept when the player is within punch range of the altar sub-goal, so a press is only
        ever DROPPED while such a sub-goal exists to walk to.

        The two rejected keys, and why:
          - `any(items[].held)`: Level 0-4 ships a `CustomKey1` carryable and zero `ItemPlaceZone`s, so picking
            the key up cost the agent its punch -- parry, melee and the throw that would have given the button
            back -- for the rest of the carry, on a level in the prelude curriculum;
          - `campaign.wanting_altars` (gate-blind): Level 0-2's blue skull sits on the main route, and the only
            zone that accepts it is behind the `altar_only` gate `-60,-6,236`, a **secret arena** whose `hops`
            is null because it is off the exit chain entirely (checked against 0-2's door graph: its one
            activated room `-60,-11,236` is nowhere on the pit chain). `_choose_target` skips `hops is None`
            gates, so no sub-goal can ever exist for it -- and yet picking the skull up dropped the punch button
            for the rest of the level, with `_near_subgoal` structurally unable to release it. Under the rule
            here 0-2 simply never protects that carry: the agent keeps punch, may carry the skull or throw it
            away freely, and loses nothing the route needs.

        The release is the spec's: within `subgoal_punch_range_m` of the sub-goal altar. That is deliberately
        narrower than "any altar that accepts this", because the SOURCE pedestal is itself an unfilled zone that
        accepts the item, and releasing next to it would let the very first press after the pickup throw the
        skull straight back down.

        Inert with no altar sub-goal and no filled altar nearby, so a level with no `ItemPlaceZone` -- and any
        mod that sends no `altars`/`items` -- behaves exactly as before, and punch stays available as an attack
        and a parry everywhere else. A removed press is not charged `RewardConfig.punch` either: the command the
        game receives is what the reward is computed from, here as for the slide latch.
        """
        if self.cfg.mode != "campaign" or self.cfg.subgoal_punch_range_m <= 0:
            return
        if "punch" not in command["buttons"]:
            return
        campaign = prev.get("campaign") or {}
        held = {i.get("item") for i in (campaign.get("items") or ())
                if isinstance(i, dict) and i.get("held")}
        target = self.gates.target or {}
        # The one source of truth: an altar sub-goal for something actually held. `_near_subgoal` releases on
        # the same target, so protection can never outlive the thing that would lift it.
        carrying = bool(held) and target.get("subgoal") == SUBGOAL_ALTAR and target.get("item") in held
        keep = self._near_subgoal(prev, (SUBGOAL_ALTAR,)) if carrying else self._near_subgoal(prev, (SUBGOAL_ITEM,))
        if keep or not (carrying or self._near_filled_altar(prev)):
            return
        command["buttons"] = [b for b in command["buttons"] if b != "punch"]

    def _hold_slide(self, prev: dict[str, Any], command: dict[str, Any]) -> None:
        """Keeps slide held for `slide_min_hold` decisions once pressed. Never removes a slide, only adds one.

        0-1's vent needs slide held over a run of 3-18 consecutive decisions and the policy presses it on ~90% of
        them, which is close to the observed pass rate -- but the latch also holds slide while the policy presses
        jump, and a jump out of a grounded slide is one of the two ways into the wedge. Off by default.
        """
        if self.cfg.slide_min_hold <= 0:
            return
        if (prev.get("campaign") or {}).get("input_locked"):
            self._slide_latch = 0
            return
        if "slide" in command["buttons"]:
            self._slide_latch = self.cfg.slide_min_hold - 1
        elif self._slide_latch > 0:
            command["buttons"] = [*command["buttons"], "slide"]
            self._slide_latch -= 1
            self._behaviour["slide_forced"] += 1

    def _note_behaviour(self, raw: dict[str, Any], command: dict[str, Any], raw_pitch_cmd: float, applied_mode: int = 0) -> None:
        """Per-episode diagnostics: is the agent shooting, is it shooting at anything, and does it turn toward enemies?

        `applied_mode` is the look mode that actually drove the camera (step() passes 0 whenever the requested
        mode found nothing to aim at), not the one the policy sampled.
        """
        self._behaviour["steps"] += 1
        self._behaviour[("look_free", "look_enemy", "look_gate")[applied_mode if 0 <= applied_mode <= 2 else 0]] += 1
        firing = "fire1" in command["buttons"] or "fire2" in command["buttons"]
        self._behaviour["firing"] += firing
        self._behaviour["yaw_sum"] += abs(command["look"][0])
        player = raw.get("player")
        if not player:
            return
        self._behaviour["pitch_steps"] += 1
        self._behaviour["pitch_sum"] += abs(player["pitch"])
        self._behaviour["pitch_signed_sum"] += player["pitch"]
        self._behaviour["look_up_sum"] += player.get("forward", (0.0, 0.0, 0.0))[1]
        visible = [e for e in raw.get("enemies", []) if e["visible"]]
        if not visible:
            return
        self._behaviour["enemy_visible"] += 1
        errors = aim_errors(player, visible[0])
        if errors is None:
            return
        angle, yaw_err, pitch_err = errors
        self._behaviour["angle_sum"] += angle
        self._behaviour["yaw_err_sum"] += yaw_err
        self._behaviour["pitch_err_sum"] += pitch_err
        # Tracking scores: +1 when the look command turns toward the nearest visible enemy, -1 when away
        # (random play scores 0). Uses the unclamped pitch command so the band does not bias it.
        #
        # Two corrections after the 2.07M-step audit, which found the raw score reproduced almost entirely
        # by measurement artefacts rather than by policy behaviour:
        #  - Grade only when the nearest enemy overall is the visible one. `pack_observation` feeds the
        #    policy the nearest 8 by distance REGARDLESS of visibility, so when the nearest is occluded the
        #    old score graded the look command against a farther enemy at an unrelated azimuth. Simulating
        #    that alone reproduced the live 0.121-0.150 at the observed visibility, from a true ~0.24.
        #  - Use an ANGULAR deadzone. `abs(x) > 0.5` was 0.5 metres, i.e. 1.4 degrees off at 20 m but
        #    4.8 degrees at 6 m, so the threshold silently tightened with range.
        # Note these scores are still computed on the SAMPLED action, so with a soft policy they sit well
        # below the mean-action agreement (86% at 2.0M): the gap between them is the exploration noise.
        x, y, z = visible[0]["rel"]
        yaw_cmd = command["look"][0]
        # Look modes 1 and 2 aim from geometry, not from the look heads, so grading those steps would score ~1.0
        # by construction and stop measuring the thing these numbers exist to measure. A step where the requested
        # mode fell back to the sampled bins is graded, because there the look heads did drive the camera.
        if applied_mode == 0 and visible[0] is raw["enemies"][0]:  # gates only the two tracking scores, not the metrics below
            if yaw_cmd and abs(math.degrees(math.atan2(x, z))) > 5.0:
                self._behaviour["yaw_track_n"] += 1
                self._behaviour["yaw_track"] += 1 if (yaw_cmd > 0) == (x > 0) else -1
            if raw_pitch_cmd and abs(math.degrees(math.atan2(y, math.hypot(x, z)))) > 5.0:
                self._behaviour["pitch_track_n"] += 1
                self._behaviour["pitch_track"] += 1 if (raw_pitch_cmd > 0) == (y > 0) else -1
        elev = horizon_elevation(player, visible[0])
        if elev is not None:
            self._behaviour["elev_steps"] += 1
            self._behaviour["elev_sum"] += elev
            self._behaviour["elev_abs_sum"] += abs(elev)
            self._behaviour["elev_over15"] += abs(elev) > 15.0
        self._behaviour["dist_sum"] += visible[0]["dist"]
        self._behaviour["close"] += visible[0]["dist"] <= 5.0
        on_target = angle <= 15.0
        self._behaviour["on_target"] += on_target
        self._behaviour["firing_on_target"] += on_target and firing

    def _track_enemies(self, raw: dict[str, Any]) -> None:
        for e in raw.get("enemies", []):
            if e["health"] > self._enemy_max_health.get(e["id"], 0.0):
                self._enemy_max_health[e["id"]] = e["health"]

    # Campaign ----------------------------------------------------------

    def _archive_path(self, level: str | None = None) -> Path:
        return Path(self.cfg.explore_dir) / f"explore_{safe_name(level or self.level)}_{self.cfg.port}.npz"

    def _save_archive(self) -> None:
        """Saves every level's archive this env has touched, not only the current one."""
        if self.cfg.mode != "campaign" or not self.cfg.explore_dir:
            return
        self._archives[self.level] = self.archive  # eval.py assigns env.archive directly; keep the map honest
        for level, archive in self._archives.items():
            path = self._archive_path(level)
            path.parent.mkdir(parents=True, exist_ok=True)
            archive.save(path)

    def _switch_level(self, previous_level: str, new_level: str) -> None:
        """Moves this env to `new_level`, saving the outgoing archive under the OUTGOING level's path first.

        Both names are arguments on purpose: an implementation that assigns `self.level` and then builds the save
        path from it writes the outgoing counts under the incoming level's filename and corrupts both archives.
        `_stuck_streak` / `_stuck_checkpoint` are level-scoped and are cleared, so a streak on the level being
        left cannot force a reload on the level being entered.
        """
        self._archives[previous_level] = self.archive
        if self.cfg.explore_dir:
            try:
                path = self._archive_path(previous_level)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.archive.save(path)
            except OSError as exc:
                print(f"UltrakillEnv: could not save the exploration archive for {previous_level!r}: {exc}")
        self.level = new_level
        archive = self._archives.get(new_level)
        if archive is None:
            archive = (ExplorationArchive.load(self._archive_path(new_level), self.cfg.cell_size)
                       if self.cfg.explore_dir else ExplorationArchive(self.cfg.cell_size))
            self._archives[new_level] = archive
        self.archive = archive
        self._stuck_streak, self._stuck_checkpoint = 0, None

    def _sample_level(self) -> str:
        """The level the next fresh load uses, from the curriculum the trainer writes.

        A file that is missing, torn, from another run or listing another order leaves `stats` empty, and
        `choose_level` then returns `levels[0]` -- the first level only, which is the safe fallback. The warning
        fires once per worker so a silently disabled curriculum is visible in the training log.
        """
        stats = self._curriculum
        if self.cfg.curriculum_path:
            table = read_curriculum(self.cfg.curriculum_path, order=list(self.cfg.levels),
                                    run_name=Path(self.cfg.curriculum_path).parent.name)
            if table is None:
                if not self._curriculum_warned and Path(self.cfg.curriculum_path).exists():
                    self._curriculum_warned = True
                    print(f"UltrakillEnv: ignoring {self.cfg.curriculum_path} (wrong run or level order); "
                          f"this game will train on {self.cfg.levels[0]!r} only")
            else:
                self._curriculum = stats = table
        return choose_level(self._rng, list(self.cfg.levels), stats, floor=self.cfg.level_weight_floor)

    def _save_archive_on_schedule(self) -> None:
        """Saves the exploration archive every `archive_save_steps` or `archive_save_seconds`, whichever is first.

        The episode-counted save is not enough on its own: campaign episodes are thousands of decisions, no
        worker of the ground run ever reached 20 of them before a restart, and SubprocVecEnv workers never call
        close() on Ctrl+C -- so that run saved nothing at all and threw away every visit count it earned.
        """
        if self.cfg.mode != "campaign" or not self.cfg.explore_dir:
            return
        now = time.monotonic()
        if (self._lifetime_steps - self._last_save_steps < self.cfg.archive_save_steps
                and now - self._last_save_time < self.cfg.archive_save_seconds):
            return
        self._last_save_steps, self._last_save_time = self._lifetime_steps, now
        try:
            self._save_archive()
        except OSError as exc:
            print(f"UltrakillEnv: could not save the exploration archive: {exc}")

    @staticmethod
    def _current_checkpoint(raw: dict[str, Any]) -> str | None:
        """Id of the checkpoint a respawn would return to, or None when no checkpoint is active in this level load."""
        for cp in (raw.get("campaign") or {}).get("checkpoints", []):
            if cp.get("current"):
                return cp["id"]
        return None

    @staticmethod
    def _rounded(pos) -> list[float]:
        return [round(float(v), 2) for v in pos]

    def _campaign_reset(self) -> dict[str, Any]:
        """Fresh level load or checkpoint respawn, so each game drifts toward the part of the level it cannot do yet."""
        prev = self._raw or {}
        camp = prev.get("campaign") or {}
        current = self._current_checkpoint(prev)
        if self._last_end_reason in ("stuck", "wedged"):
            # A respawn can leave a door locked behind the player; three stuck episodes in a row at the same
            # checkpoint force a fresh load. "wedged" counts too: a checkpoint whose respawn point is wedge-prone
            # could otherwise never escalate to a reload, since it never ends an episode "stuck".
            self._stuck_streak = self._stuck_streak + 1 if current == self._stuck_checkpoint else 1
            self._stuck_checkpoint = current
        else:
            self._stuck_streak, self._stuck_checkpoint = 0, None
        fresh = choose_fresh_start(
            self._rng,
            # Judged against the OUTGOING level: this decides whether a respawn is even possible, and a respawn
            # is only possible in the level the game is already in.
            in_level=prev.get("scene") == self.level and prev.get("player") is not None,
            level_over=bool(camp.get("level_over")),
            has_checkpoint=current is not None,
            stuck_streak=self._stuck_streak,
            stuck_limit=self.cfg.stuck_repeats,
            fresh_prob=self.cfg.fresh_start_prob,
        )
        if fresh and self.cfg.levels:
            # The only place a level is ever sampled. A fresh load is the only moment the scene can change
            # without corrupting the episode's info, its best-run positions and its archive.
            new_level = self._sample_level()
            if new_level != self.level:
                self._switch_level(self.level, new_level)
        raw = self._skip_locked(self.client.reset(self.level, checkpoint=not fresh))
        pos = (raw.get("player") or {}).get("pos")
        if fresh:
            self.milestones.new_level_load(raw.get("campaign"))
            self.gates.new_level_load(raw.get("campaign"), pos)
            self._stuck_streak, self._stuck_checkpoint = 0, None
        else:
            # Whatever the respawn itself changes (doors it unlocks, rooms it resets, gates it already stands at)
            # pays nothing.
            self.milestones.mark_paid(raw.get("campaign"))
            self.gates.mark_paid(raw.get("campaign"), pos)
        self._fresh_start = fresh
        return raw

    def _respawn(self) -> dict[str, Any]:
        """Respawns after a death inside the same episode. Milestones the respawn itself changes pay nothing.

        This never re-samples the level, even though `StatsManager.Restart` reloads the whole level when there is
        no checkpoint yet: a switch here would change the scene mid-episode and make `info["level"]`, the best
        run's positions and the exploration archive all disagree with each other. Sampling lives in
        `_campaign_reset` and nowhere else.
        """
        before = self._raw.get("stats", {})
        raw = self._skip_locked(self.client.reset(self.level, checkpoint=True))
        self.milestones.mark_paid(raw.get("campaign"))
        player = raw.get("player")
        # No reset_episode here: the episode continues, so the gate approach it has already earned stands.
        self.gates.mark_paid(raw.get("campaign"), (player or {}).get("pos"))
        self.gates.retarget(raw.get("campaign"), (player or {}).get("pos"))
        # A respawn moves the player, so the pre-death stuck clock says nothing about where they are now.
        self._steps_since_progress = 0
        self._wedge_run = 0
        self._slide_latch = 0
        self._track_enemies(raw)
        if player and self._current_checkpoint(raw) is None:
            # No checkpoint yet, so StatsManager.Restart reloaded the level and its counters started again. Keep
            # the kills and style from before the death in this episode's info, and start a best run's positions
            # again from the spawn: the official time is the reloaded attempt's.
            after = raw.get("stats", {})
            for key in ("kills", "style"):
                self._episode_start_stats[key] = self._episode_start_stats.get(key, 0) - before.get(key, 0) + after.get(key, 0)
            self._positions = [self._rounded(player["pos"])] if self._fresh_start else []
        return raw

    def _skip_locked(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Steps with an empty action while input is locked (landing, cutscenes), so the policy never sees those frames."""
        for _ in range(int(self.cfg.max_locked_skip_s * self.cfg.fixed_fps / self.cfg.frameskip)):
            camp = raw.get("campaign") or {}
            player = raw.get("player")
            if not camp.get("input_locked") or camp.get("level_over") or not player or player["dead"]:
                break
            raw = self.client.step({})
            self._track_enemies(raw)
        return raw

    def _ground_point(self, raw: dict[str, Any]) -> tuple[float, float, float] | None:
        """The point on the ground under the player, or None when there is no ground within reach.

        Exploration is meant to pay for ground covered, not for volume occupied, and the cheapest volume in a
        3-D game is vertical. Keying the archive on the raw player position paid the full 1/sqrt(N+1) = 1.0 for
        every cell of a jump and, far worse, for every cell of a fall: the first 2.5M-step run banked 47% of all
        its novelty below y = -60 m, in cells 99.98% of which were entered exactly once, reaching 57 km under the
        map. Free fall is an unbounded supply of never-visited cells, it never dies, and it kept resetting the
        stuck clock, so diving off the level strictly beat playing it. Projecting onto the ground fixes both: a
        fall pays nothing (no ground within range), a jump on the spot pays once instead of once per cell of
        height, and running forward still pays for every new patch of floor -- including while airborne, which
        matters because fast movement in this game is mostly airborne.

        The centre ray is the measure, when the mod reports one: the 8-ray ring's minimum can be a ledge 4 m away
        rather than the floor, and the measured spread among rays that hit is p50 0.5 m but p90 10.8 m. That
        re-keys about 23% of the cells a carried archive holds, which is accepted -- the counts are not
        comparable across this change anyway, and `oob_frac` takes a fresh baseline. A 0.5.x mod sends no centre
        ray, so the ring minimum stays as the fallback.
        """
        player = raw.get("player")
        if not player:
            return None
        centre = raw.get("ground_ray_center")
        rays = raw.get("ground_rays") or ()
        if centre is None and not rays:
            return None
        drop = float(centre) if centre is not None else min(rays)
        if drop >= self.cfg.layout.ground_ray_length - 0.5:
            return None  # the mod writes ground_ray_length exactly when the ray hits nothing: off the map
        x, y, z = player["pos"]
        return (x, y - drop, z)

    def _note_wedge(self, prev: dict[str, Any], cur: dict[str, Any]) -> bool:
        """Tracks the absorbing slowMode/heavyFall state; returns whether the current run has reached the hold.

        The signature is airborne, not sliding, not moving, with the game's own `slowMode` or `heavyFall` set.
        Both halves are load-bearing, measured over 32,022 live decisions: without the movement term ordinary
        falls and slams flag 26.9% of all decisions (346 an episode), and at a 0.05 m threshold one of the five
        real wedge runs -- the 694-decision one after checkpoint 3, whose per-decision movement has p90 0.055 m
        -- fragments below the hold and is lost. 1.5 m/s (0.10 m per decision at both speed settings) keeps all
        five. A mod older than 0.6.0 sends neither flag; the fallback is the same predicate without the flag test.

        `wedged_steps` counts only steps inside a run that reached the hold, crediting the whole run
        retroactively when it trips, so ordinary airborne time never enters it.
        """
        player = cur.get("player")
        wedged = False
        if player:
            vel = player.get("vel") or (0.0, 0.0, 0.0)
            prev_pos = (prev.get("player") or {}).get("pos")
            moved = math.dist(player["pos"], prev_pos) if prev_pos else 0.0
            still = math.hypot(vel[0], vel[2]) < 1.0 and moved < self._creep_m
            wedged = still and not player.get("grounded") and not player.get("sliding")
            if wedged and ("slow_mode" in player or "heavy_fall" in player):
                wedged = bool(player.get("slow_mode") or player.get("heavy_fall"))
        self._wedge_run = self._wedge_run + 1 if wedged else 0
        if self._wedge_run == self._wedge_hold:
            self._wedged_steps += self._wedge_hold  # the whole run, credited the moment it counts as one
        elif self._wedge_run > self._wedge_hold:
            self._wedged_steps += 1
        return self._wedge_run >= self._wedge_hold

    def _campaign_progress(self, prev: dict[str, Any], raw: dict[str, Any]) -> CampaignStep:
        """What the level did this step. Any progress restarts the stuck clock."""
        camp = raw.get("campaign") or {}
        checkpoints, arenas, doors, pickups, placements = self.milestones.update(raw.get("campaign"))
        novelty = 0.0
        player = raw.get("player")
        pos = player["pos"] if player else None
        gates_new, approach = self.gates.update(raw.get("campaign"), pos)
        if player:
            self._last_pos = list(pos)
            ground = self._ground_point(raw)
            if ground is None:
                self._oob_steps += 1
            else:
                novelty = self.archive.visit(ground)
            if novelty > 0:
                self._cells_new += 1
            if self._fresh_start:
                self._positions.append(self._rounded(pos))
            if camp.get("exit"):
                self._exit_dist_min = min(self._exit_dist_min, math.dist(pos, camp["exit"]["pos"]))
        if camp.get("level_started"):
            self._level_started = True
        path_gain = self.path_progress.update(camp.get("path"))
        # Kills and style restart the clock too: four of 0-1's gate doors sit behind kill gates in locked rooms
        # where novelty runs out, and the live probe recorded 193 s of fighting after checkpoint 3 that the clock
        # truncated. Damage dealt deliberately does NOT count -- rewards.py attributes none of it to the player
        # and ULTRAKILL enemies damage each other, so a crossfire tail would simply never truncate.
        stats, before = raw.get("stats", {}), prev.get("stats", {})
        fought = stats.get("kills", 0) > before.get("kills", 0) or stats.get("style", 0) > before.get("style", 0)
        if (checkpoints or arenas or doors or pickups or placements
                or novelty > 0 or path_gain > 0 or gates_new or approach > 0 or fought):
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        return CampaignStep(checkpoints=checkpoints, arenas=arenas, doors=doors, novelty=novelty, path_gain=path_gain,
                            gates=gates_new, gate_approach=approach, item_pickups=pickups, item_placements=placements)

    def _level_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Official time, kills, style, restarts and rank as the game's results screen would count them."""
        camp = raw.get("campaign") or {}
        stats = raw.get("stats", {})
        seconds = camp.get("seconds", stats.get("seconds", 0.0))
        restarts = camp.get("restarts", stats.get("restarts", 0))
        kills, style = stats.get("kills", 0), stats.get("style", 0)
        ranks = camp.get("ranks")
        rank = compute_rank(seconds, kills, style, restarts, ranks) if ranks else None
        return {"seconds": seconds, "kills": kills, "style": style, "restarts": restarts, "rank": rank}

    def _end_campaign_episode(self, raw: dict[str, Any], reason: str, info: dict[str, Any]) -> None:
        if reason == "level_complete":
            info["completed"] = 1
            if self._fresh_start:
                # Only a fresh load has a meaningful official time: the timer carries across respawn episodes.
                result = self._level_result(raw)
                info["level_seconds"] = result["seconds"]
                info["restarts"] = result["restarts"]
                info["rank"] = result["rank"]
                # A save failure here (this project has hit Windows PermissionError on these paths) must not
                # end a training worker mid-run, the same way close() releases the game even when its own
                # archive write fails, and progress.py warns rather than raises on a status-file write failure.
                try:
                    self._save_best_run(raw)
                except OSError as exc:
                    print(f"UltrakillEnv: could not save the best run for {self.level!r}: {exc}")
        self._episodes += 1
        if self._episodes % 20 == 0:
            try:
                self._save_archive()
            except OSError as exc:
                print(f"UltrakillEnv: could not save the exploration archive: {exc}")

    def _save_best_run(self, raw: dict[str, Any]) -> None:
        if not self.cfg.best_runs_dir:
            return
        result = self._level_result(raw)
        run = {
            "level": self.level,
            "seconds": result["seconds"],
            "kills": result["kills"],
            "style": result["style"],
            "restarts": result["restarts"],
            "deaths": self._deaths,
            "rank": result["rank"],
            "difficulty": (raw.get("campaign") or {}).get("difficulty"),
            "positions": self._positions,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = Path(self.cfg.best_runs_dir) / f"{safe_name(self.level)}.json"  # one file per level, always
        path.parent.mkdir(parents=True, exist_ok=True)
        save_best_run(path, run)

    def _pack(self, raw: dict[str, Any]) -> np.ndarray:
        player = raw.get("player")
        explore = None
        if self.cfg.mode == "campaign" and player:
            # The same ground projection the archive is keyed on, so the map the policy reads addresses the cells
            # novelty is actually paid for. Falling back to the raw position off the map keeps the map defined.
            explore = self.archive.features(self._ground_point(raw) or player["pos"], player["yaw"])
        # The same target object look mode 2 aims at, so what the policy sees and what it can point at agree.
        target = self.gates.target if self.cfg.mode == "campaign" else None
        return pack_observation(raw, self.cfg.layout, self._enemy_max_health, explore, target)

    def _info(self, raw: dict[str, Any]) -> dict[str, Any]:
        stats = raw.get("stats", {})
        player = raw.get("player") or {}
        start = self._episode_start_stats
        info = {
            "kills": stats.get("kills", 0) - start.get("kills", 0),
            "style": stats.get("style", 0) - start.get("style", 0),
            "hp": player.get("hp", 0),
            "wave": (raw.get("cybergrind") or {}).get("wave", 0),
            "deaths": self._deaths,
        }
        b = self._behaviour
        steps = max(1, b["steps"])
        seen = max(1, b["enemy_visible"])
        info["firing_frac"] = b["firing"] / steps
        info["on_target_frac"] = b["on_target"] / steps
        info["firing_on_target_frac"] = b["firing_on_target"] / steps
        info["enemy_visible_frac"] = b["enemy_visible"] / steps
        info["enemy_angle_mean"] = b["angle_sum"] / seen  # degrees off the crosshair
        info["enemy_yaw_angle_mean"] = b["yaw_err_sum"] / seen  # heading error only, ignoring pitch
        info["enemy_pitch_err_mean"] = b["pitch_err_sum"] / seen  # vertical miss: enemy elevation in camera space
        info["yaw_track"] = b["yaw_track"] / max(1, b["yaw_track_n"])  # -1..1, turns toward the enemy horizontally
        info["pitch_track"] = b["pitch_track"] / max(1, b["pitch_track_n"])  # -1..1, vertically
        info["pitch_abs_mean"] = b["pitch_sum"] / max(1, b["pitch_steps"])  # camera pitch away from level
        info["pitch_mean"] = b["pitch_signed_sum"] / max(1, b["pitch_steps"])  # signed rotationX
        info["look_up_mean"] = b["look_up_sum"] / max(1, b["pitch_steps"])  # mean camera forward.y (>0 = looking up)
        elev_steps = max(1, b["elev_steps"])
        info["enemy_elev_mean"] = b["elev_sum"] / elev_steps  # nearest visible enemy, degrees above the horizon
        info["enemy_elev_abs_mean"] = b["elev_abs_sum"] / elev_steps
        info["enemy_elev_over15_frac"] = b["elev_over15"] / elev_steps  # share of steps the pitch band cannot reach
        info["enemy_dist_mean"] = b["dist_sum"] / seen
        info["enemy_close_frac"] = b["close"] / steps  # nearest visible enemy within 5 m
        info["yaw_per_step_mean"] = b["yaw_sum"] / steps  # degrees turned per decision
        if self.cfg.mode == "campaign":
            # The level this episode RAN on: `info` is built in step() before SubprocVecEnv calls reset(), so a
            # switch episode's row still carries the level it played. Deliberately not in CAMPAIGN_INFO_KEYS,
            # which go through Monitor(info_keywords=...) and ProgressCallback._num and must all be numeric.
            info["level"] = self.level
            info["fresh_start"] = int(self._fresh_start)
            info["checkpoints_level"] = self.milestones.checkpoints_reached  # distinct checkpoints this level load
            info["cells_new"] = self._cells_new  # cells entered for the first time this episode
            # Steps with no ground beneath: falling, or off the map entirely. The first campaign run banked 47%
            # of its novelty in the void below the level, so this is the number that says whether that is over.
            info["oob_frac"] = self._oob_steps / steps
            info["exit_dist_min"] = None if math.isinf(self._exit_dist_min) else self._exit_dist_min
            info["completed"] = 0  # set to 1 on the step that ends with level_complete
            info["level_seconds"] = None  # official time, only for a fresh-start completion
            # Route gates. `gates_reached` is numeric and always present so it can be charted; `gate_hops_best`
            # is None until a gate is reached, so it only ever reaches episodes.jsonl. Both are inherited by a
            # respawn episode from its level load, so judge them on fresh starts.
            info["gates_reached"] = self.gates.gates_reached
            info["gate_hops_best"] = self.gates.best_hops
            info["wedged_steps"] = self._wedged_steps
            info["level_started"] = int(self._level_started)
            info["look_free_frac"] = b["look_free"] / steps
            info["look_enemy_frac"] = b["look_enemy"] / steps
            info["look_gate_frac"] = b["look_gate"] / steps
            info["slide_forced_frac"] = b["slide_forced"] / steps
            info["start_checkpoint"] = self._start_checkpoint
            pos = player.get("pos") or self._last_pos
            info["end_pos"] = self._rounded(pos) if pos else None  # where the episode actually ended up
        return info
