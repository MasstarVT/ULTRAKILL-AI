"""Gymnasium environment for ULTRAKILL (Cyber Grind and campaign levels)."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np

from ultrakill_ai.campaign import (
    CAMPAIGN_LEVELS_SHIPPED,
    CURRICULUM_BLOCKED_FRESH_EPISODES,
    CURRICULUM_WEIGHT_CAP,
    CURRICULUM_WEIGHTINGS,
    ROUTE_SOURCE_NAMES,
    SUBGOAL_ALTAR,
    SUBGOAL_ITEM,
    ExitGuard,
    ExplorationArchive,
    GateProgress,
    MilestoneTracker,
    PathProgress,
    altar_aim_point,
    choose_fresh_start,
    choose_level,
    compute_rank,
    exit_ground_point,
    load_route,
    read_curriculum,
    s_rank_time,
    safe_name,
    save_best_run,
)
from ultrakill_ai.protocol import (
    DEFAULT_PORT,
    DEFAULT_RESET_TIMEOUT,
    DEFAULT_STEP_TIMEOUT,
    MOD_INCOMPATIBLE_FILE,
    RECOVERABLE,
    TECH_LAYOUT_FEATURES,
    BridgeClient,
    BridgeClosed,
    BridgeError,
    BridgeIncompatible,
    BridgeSceneUnknown,
    BridgeTimeout,
    mod_features,
    write_mod_incompatible,
)
from ultrakill_ai.envlog import EnvLog, env_log_path
from ultrakill_ai.procmem import GB as PROC_GB
from ultrakill_ai.procmem import MB as PROC_MB
from ultrakill_ai.procmem import derive_game_limit, private_bytes
from ultrakill_ai.rewards import (
    CampaignStep,
    RewardConfig,
    aim_errors,
    compute_reward,
    horizon_elevation,
    macro_landed,
)
from ultrakill_ai.spaces import (
    MOVE_TECH_FIELDS,
    NUM_WEAPON_SLOTS,
    NUM_WEAPON_VARIATIONS,
    PITCH_BINS,
    TECH_LAYOUTS,
    YAW_BINS,
    ObsLayout,
    action_space,
    decode_action,
    pack_observation,
    yaw_frame,
)
from ultrakill_ai.times import valid_official_seconds

CYBERGRIND_SCENE = "Endless"
# Per-episode info the campaign Monitor records (scripts/train.py); every key is in every campaign info.
CAMPAIGN_INFO_KEYS = ("kills", "style", "deaths", "completed", "fresh_start", "level_seconds",
                      "checkpoints_level", "cells_new", "exit_dist_min", "exit_ground_dist_min", "oob_frac",
                      "gates_reached", "wedged_steps", "level_started", "look_gate_frac", "slide_forced_frac",
                      "targets_parked", "exit_banished", "route_source", "ladder_collapsed",
                      # The speed stage (docs/superpowers/specs/2026-09-18-speed-stages.md): the level's target
                      # time, the raw S-rank threshold it was scaled from, and what the completion edge paid.
                      # `target_seconds` is how the driver learns the target at all -- it travels
                      # env -> status.json -> driver_state.json -> the sidecar, and `s_rank_seconds` rides with
                      # it so the sidecar can say what the game itself calls S.
                      "target_seconds", "s_rank_seconds", "completion_bonus")

YAW_CAP = max(abs(b) for b in YAW_BINS)  # 90 degrees per decision, the widest look bin
PITCH_CAP = max(abs(b) for b in PITCH_BINS)  # 20 degrees per decision
MODE1_PITCH_LIMIT = 85.0  # inside the game's own +-90 clamp (ActionInjector.ApplyLook)
ARCHIVE_CHECK_EVERY = 500  # steps between exploration-archive schedule checks (the save itself costs ~16 ms)
DEFAULT_WEDGE_HOLD_S = 3.0  # wedge_seconds 0 turns the episode end off; counting still uses this hold
# A RESCUE is a one-decision move of at least this many metres on a step that is not a death: the game's
# non-instakill DeathZone teleporting the player back onto the walkway. Measured over 114,877 recorded decision
# pairs on Level 0-1 (runs/probe_0-1_rung85 + runs/probe_0-1_brutal): ordinary movement tops out at 11.99 m per
# decision, all 146 such moves that cost HP land on the level's known rescue points, and none is anything else.
# Three real rescues moved 8.9-10.3 m and are missed, which errs toward charging less. See `_note_rescue`.
RESCUE_JUMP_M = 12.0
SSJ_GAP_S = 0.012  # the mod's own `ssj_gap_s` default: the middle of SSJ bucket 1 (docs/protocol.md), sent explicitly
MOVE_TECH_GRACE = 30  # v2: frames with a player and no usable `move_tech` before the one-time warning


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
    # How the fresh-load weights are computed. "inverse_rate" is what every run before 2026-09-18 used: weight
    # `max(floor, 1 - fresh completion rate)`, so the WORST level gets the most attention -- which starved the
    # levels that were learning as soon as one level blocked. "progress" weights by learning progress instead,
    # with a retention floor, a cap and a damping for blocked levels (see campaign.level_weights).
    curriculum_weighting: str = "inverse_rate"
    curriculum_weight_cap: float = CURRICULUM_WEIGHT_CAP  # `progress` only: most of the mass one level may take
    # `progress` only: fresh episodes a level may go without a completion before it counts as blocked.
    curriculum_blocked_fresh_episodes: int = CURRICULUM_BLOCKED_FRESH_EPISODES
    difficulty: int = -1  # difficulty the game reads while the AI has control (3 = Violent, -1 = leave the game's own)
    unlock_all_gear: bool = False  # every weapon and variant while the AI has control, in memory only
    fresh_start_prob: float = 0.2  # chance of a fresh level load when a checkpoint respawn would also do
    stuck_seconds: float = 45.0  # game seconds without progress (milestone, new cell, shorter path) before truncating
    stuck_repeats: int = 3  # episodes in a row stuck at the same checkpoint before a fresh load is forced
    cell_size: float = 4.0  # metres per exploration cell
    max_locked_skip_s: float = 120.0  # longest input lock (landing, cutscene) stepped through without the policy
    explore_dir: str = ""  # folder for the exploration archive, so a resumed run keeps its visit counts ("" = memory only)
    best_runs_dir: str = ""  # folder for the fastest fresh-start completion of each level ("" = off)
    # A SPEED STAGE (docs/superpowers/specs/2026-09-18-speed-stages.md). Off everywhere else, and off is today's
    # behaviour exactly: `rewards.completion_bonus` returns the plain weight when it is handed no target.
    speed_bonus: bool = False  # scale `level_complete` by target_seconds / official_seconds on a fresh completion
    # 0.0 -- the default -- means the level's OWN S-rank time threshold, read live off `campaign.ranks.time[-1]`
    # at the first observation of the run. A positive number is a per-level override from the plan, and the env
    # reports whichever it is as `info["target_seconds"]`, so the reward and the driver's rule agree by
    # construction. Never a hand-picked number chosen away from the level.
    speed_target_seconds: float = 0.0
    # What the level's own S-rank threshold is MULTIPLIED by to get the target (§8). Measured 2026-09-18: the
    # 0-2 specialist's 139.5 s is already inside 0-2's S window, so a bare S threshold would have promoted that
    # stage with zero improvement -- an S-rank time is what a competent human run scores, not a fast one. The
    # scale is applied HERE and nowhere else, so `info["target_seconds"]` is the single number the reward and
    # the driver's promotion rule both read. An explicit `speed_target_seconds` is a decision already made and
    # is NEVER scaled.
    speed_target_scale: float = 0.75
    # Route gates (campaign.gates): the door-graph ladder GateProgress walks.
    gate_reach_m: float = 8.0  # horizontal radius at which a gate counts as reached (2x while it is open)
    gate_reach_v_m: float = 6.0  # vertical half-height of the same test, so a roof over a door is not "reached"
    gate_min_gain_m: float = 0.5  # metres of new best closeness below which gate_approach pays nothing
    gate_hops_min_frac: float = 0.5  # share of gates that must carry `hops` before the ladder is trusted at all
    # Target patience, the answer to a ladder collapsed by multi-room doors (0-3, 1-1, 1-2, 2-3, 4-3, 8-1); see
    # docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md. 0 restores the pre-patience tracker.
    gate_target_patience_s: float = 20.0  # game seconds a target may go without getting closer before it is parked
    gate_unpark_m: float = 2.0  # metres nearer than it was parked at that bring a parked gate back
    gate_fallback_hysteresis_m: float = 10.0  # metres a rival must beat the sticky fallback target by
    # WHERE parking may act. "collapsed" (the default) is only on a level whose gate ladder the detector calls
    # collapsed at the spawn -- measured on 0-3, 1-1, 1-2, 2-3, 4-3 and 8-1 and on none of the twelve healthy
    # ladders. "always" is the unconditional mechanism as first shipped, which cost 0-1 half its fresh
    # completion rate; "off" is `gate_target_patience_s: 0` by another name. See `detect_collapsed_ladder`.
    gate_patience_mode: str = "collapsed"
    # Layer choice on a collapsed level that also ships a room trunk (0-3, 4-3): False -- the default, and the
    # lead's ruling -- keeps them on gates plus patience, which is what is moving 0-3 today. True hands them
    # their trunk instead. A level with a HEALTHY ladder never reads its route file either way.
    prefer_route_when_collapsed: bool = False
    exit_max_shift_m: float = 100.0  # metres the exit may legitimately move inside one level load (ExitGuard)
    # The route fallback: layer 2, the offline room trunk shipped per level in ultrakill_ai/routes/ (see
    # docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md). On by default and INERT on
    # every level with a usable gate ladder and on every level with no file, which is all 21 the live configs
    # name; `route_fallback: false` refuses every file without deleting one.
    route_fallback: bool = True
    route_dir: str = ""  # folder holding route_<scene>.json ("" = the packaged ultrakill_ai/routes/)
    route_exit_tol_m: float = 5.0  # guard I5: metres the live FinalPit may differ before the file is refused
    route_seed_m: float = 150.0  # how near a rung the player may start and still have it absorbed, not paid
    # Metres of ground that must lie under an airborne player before a ROOM-TRUNK rung counts as
    # reached. A room centroid's 8 m x 6 m reach cylinder can overhang the wall of the room it names,
    # and on Level 0-3 one did: 680 of that rung's 801 live credit steps were the player hanging
    # against the wall from the room BEFORE it. Gates are never subject to this -- a door is a place
    # you pass through, a room is a place you land in. See `GateProgress._on_ground` for the
    # measurement behind 8.0; 0 disables the rule.
    route_ground_m: float = 8.0
    # Skull carry. 0 disables both carry-protection rules; they are inert on any level with no ItemPlaceZone.
    subgoal_punch_range_m: float = 4.0  # Punch.ActiveFrame's own 4 m reach: inside it a punch can pick up or place
    camera_height_m: float = 0.9  # metres from player.pos up to the camera, where every ray starts (see _eye)
    target_kind_slots: bool = False  # repurpose the target's `open`/`locked` slots as "is an item"/"is an altar"
    # The absorbing slowMode/heavyFall movement state: airborne forever, stamina frozen, a 0.477 m/s creep.
    wedge_seconds: float = 3.0  # game seconds wedged before the episode ends (0 = no end, the steps are still counted)
    wedge_creep_mps: float = 1.5  # movement below this counts as not moving, in the wedge test only
    slide_min_hold: int = 0  # decisions slide stays held once pressed (0 = off; an experiment, see the design spec)
    # THE STICKY WEAPON SLOT -- stage S5 of docs/superpowers/specs/2026-09-20-speedrun-tech.md, and DORMANT:
    # False is today's action stream byte for byte, and nothing below is read while it is False.
    #
    # WHY IT EXISTS. `GunControl.SwitchWeapon` with `targetSlotIndex == currentSlotIndex` reads
    # `PrefsManager`'s `WeaponRedrawBehaviour`, whose default is 0 = cycle to the next variation, so pressing
    # the slot already held RE-DRAWS the weapon: `Revolver.OnEnable` sets `gunReady = false` and only the
    # `ReadyGun()` animation event clears it, while `Revolver.Update` gates firing on `gunReady`. A probe of
    # the PROMOTED `Level_0-1.zip` measured that press on 76.0% of steps (39,371 of 51,772), which would mean
    # the agent keeps its weapon permanently in the draw animation and suppresses its own primary fire. That
    # 76% is from a DIFFERENT CHECKPOINT and is being re-measured on the live policy by the S0 counters below
    # (`slot_same_frac`); this switch must not be turned on before that number exists.
    sticky_weapon_slot: bool = False
    # ... and, while it is on, how many decisions a switch to a DIFFERENT slot must wait for. A switch also
    # plays a draw animation, so honouring one every decision reproduces the same defect with two slots instead
    # of one. THE SPEC NAMES NO NUMBER FOR THIS and the draw animation's real length is NOT MEASURED -- the
    # game's animation clips are serialized and cannot be read from the decompiled C#. 3 is the decisions
    # covered by the nearest documented window in the spec's own table (the 200 ms `JumpReady` cooldown) at
    # `fixed_fps: 30` / `frameskip: 2`, i.e. 66.7 ms of game time per decision. It is a starting value to be
    # replaced by a measurement, not a finding. 0 or 1 honours every switch, which is the drop rule alone.
    # NOTE THE TENSION the spec records: "swap cancel" (§2) is a technique that WANTS rapid switching, so a
    # large number here trades one gain for another. Judge them together.
    sticky_slot_switch_every: int = 3
    # THE S7 TECH LAYOUT (docs/superpowers/specs/2026-09-20-speedrun-tech.md §4.1-§4.3; the plan is
    # docs/superpowers/plans/2026-09-23-tech-break-python.md). "v1" -- the default, and every run before the break
    # -- is the 479-input, 12-dimension policy BYTE FOR BYTE (tests/test_tech_layout.py pins it). "v2" is 530
    # inputs and 15 dimensions / 57 logits; it needs mod 0.8.0 and refuses to start against anything older.
    tech_layout: str = "v1"
    # The v2 gates: which new heads may reach the game. A value that is off here is masked in Python (it never
    # reaches the wire), refused by the mod as well where the mod has a switch for it, and its logits are pinned
    # in training (training.pin_action_rows). The defaults are the S7 scope: M1 `ssj` alone.
    macro_ssj: bool = True
    macro_ssj_wall: bool = False  # also the mod's `macro_ssj_wall`; M2 was cut to reserved (2026-09-20 review)
    variant_switching: bool = False  # also the mod's `variant_switching`; opens at S9
    hook_action: bool = False  # the whiplash: a plain HoldButton the mod cannot refuse; opens at S10
    archive_save_steps: int = 20000  # env lifetime steps between exploration-archive saves
    archive_save_seconds: float = 600.0  # ... or this much wall time, whichever comes first

    # Bridge resilience. One sick game used to take the whole vectorized run down with it: a TimeoutError out of
    # a worker's socket is unhandled inside SubprocVecEnv's `_worker`, so the worker process exits, the parent's
    # pipe reads EOF and every other game's rollout dies with it. See the timeout gotcha in CLAUDE.md.
    step_timeout_s: float = DEFAULT_STEP_TIMEOUT  # one frame; generous, but a step is not a scene load
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT  # a scene load, with 12 games loading at once
    bridge_retries: int = 3  # reconnect attempts before the relaunch rung is tried
    bridge_backoff_s: float = 5.0  # first backoff; the nth attempt waits n times this
    # How long `unknown scene` is tolerated on a reset before it counts as a failure. It means the game is still
    # booting (its Addressables locators are empty while the bridge port is already up), so the cure is waiting.
    unknown_scene_wait_s: float = 300.0
    # How long connect() keeps retrying while a relaunched or still-booting game starts. This is a RETRY
    # WINDOW, not a blocking bound -- each attempt is capped by `DEFAULT_CONNECT_TIMEOUT` (10 s) and the
    # ladder clamps the window to whatever is left of the recovery budget -- so it is set by how long a cold
    # Unity instance takes to open its port, not by how long a worker may block. It was briefly 60 s, which
    # is less patience than `supervise.await_boot` assumes when it gives up on a cold copy and starts the
    # trainer anyway ("the env retries for 300s"): that start then lost the race and burned a restart.
    connect_retry_s: float = 180.0
    # THE TOTAL a recovery may take, wall clock, every attempt and the relaunch included. Everything above is a
    # per-step bound; without a total, three retries of (close + connect + handshake + reset) composed to about
    # 70 minutes, and `reset()` retried the whole ladder on top of that. The supervisor's patience is 900 s, so
    # a recovery that outlasts this budget is not a recovery -- it is a hang, and it should end the worker
    # promptly instead. See the freeze gotcha in CLAUDE.md for the arithmetic.
    bridge_recovery_budget_s: float = 540.0
    # What the MOD is told to tolerate before it drops a silent client (`EpisodeController.commandTimeoutMs`,
    # 300 s by default). This is why the 2026-09-17 freeze left eleven games running free with no connection:
    # they were not broken, they were idle behind one blocked worker -- vectorized envs step in lockstep -- and
    # the mod dropped each of them after five minutes of hearing nothing. Their workers then had to recover too,
    # so ONE sick game cost twelve episodes. It must outlast one worker's whole recovery: the step that faults
    # 120 + the recovery budget 540 = 660 s, so 900 leaves four minutes of margin. The mod reads the key at
    # `EpisodeController.cs:266`, so this is live -- but a DLL built before that line keeps its own 300 s and
    # the siblings are dropped whatever is set here. Confirmed present in the installed mod on 2026-09-17.
    mod_command_timeout_s: int = 900
    # Last rung of the ladder: relaunch THIS env's own game and wait for it to boot. Never touches another port.
    # OFF by default and turned on by `train.py` alone, so nothing else that builds an env -- eval.py,
    # bridge_test.py, campaign_check.py, every test -- can ever restart a game process. A relaunch is a real
    # side effect on a real machine; only the process whose job is to keep twelve games running gets it.
    bridge_relaunch: bool = False
    bridge_relaunch_boot_mb: int = 600  # working set an instance must pass before it counts as booted (~1 GB when up)
    bridge_relaunch_wait_s: float = 240.0  # longest wait for the relaunched instance to open its port and boot
    # Budget RESERVED for the relaunch rung, so the reconnect rung above it cannot spend the lot. Without a
    # reserve the relaunch was unreachable in the one fault shape it exists for: rung 0's three attempts each
    # cost a full reset timeout against a game that answers nothing, so the budget was always negative by the
    # time the rung was tested and the loop broke instead. Measured on a fake clock: a game that answered
    # `hello` and never answered a reset reached the relaunch on 0 of 1 runs before this, 1 of 1 after.
    bridge_relaunch_reserve_s: float = 240.0
    # At most this many workers may be inside a relaunch at once, across processes (a lock file in
    # `env_log_dir`). Twelve workers step in lockstep, so a fault that drops the games hits all twelve within
    # one step, and twelve simultaneous cold Unity starts on an already-thrashing box boot none of them in time.
    bridge_relaunch_slots: int = 2
    bridge_relaunch_stagger_s: float = 4.0  # per-port stagger, the same one `games.launch` uses for cold starts
    env_log_dir: str = ""  # runs/<run>/, where env_<port>.log is written ("" = no attribution log)
    # Recycle THIS env's own game at an EPISODE BOUNDARY once it has grown this many GB above the freshest
    # copy running (0 = off). ULTRAKILL leaks under training -- measured 2026-09-18 at 1.2-1.4 GB per game
    # per hour from a 1.2 GB boot, which took a 60 GB commit limit down -- and `scripts/mem_guard.py` already
    # recycles a fat game from outside. The difference is WHEN: the guard kills the game at an arbitrary
    # moment and the env truncates that episode as `bridge_reset`, while this fires inside `reset()`, where
    # the episode has already ended and the level is about to be loaded anyway. Same boot cost, no episode
    # lost. It rides the existing relaunch rung, so it inherits the cross-process permits and the per-port
    # stagger, and it is inert without `bridge_relaunch` -- which only `train.py` ever sets, so an eval or a
    # test can no more recycle a game this way than it can relaunch one.
    game_memory_growth_gb: float = 0.0

    layout: ObsLayout = field(default_factory=ObsLayout)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        layout = ObsLayout(**_known_fields(ObsLayout, d.pop("layout", None) or {}))
        rewards = RewardConfig(**_known_fields(RewardConfig, d.pop("rewards", None) or {}))
        return cls(**_known_fields(cls, d), layout=layout, rewards=rewards)

    # Settings that belong to ONE running trainer and must never travel in a file. `train.py` writes
    # `models/<run>/env_config.yaml` from `to_dict`, and `eval.py` loads exactly that file -- so serializing
    # these handed every eval the power to kill a game process (`bridge_relaunch`) and a path into the live
    # run's attribution log (`env_log_dir`). An eval against port 47800 during a run would then `taskkill` a
    # TRAINING game on the first bridge hiccup. `fill_run_dirs` puts them back for the trainer, in memory,
    # every time it starts.
    RUN_ONLY_FIELDS = ("bridge_relaunch", "env_log_dir")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in self.RUN_ONLY_FIELDS:
            out.pop(key, None)
        return out


class TechGates(NamedTuple):
    """Which v2 heads may reach the game: the macro VALUES forwarded, and whether variant / hook are."""

    macros: frozenset
    variant: bool
    hook: bool


def tech_gates(cfg: EnvConfig) -> TechGates:
    """The gates of a config. Macro values 3-5 have no switch: they are reserved and not built in the mod."""
    macros = frozenset(v for v, on in ((1, cfg.macro_ssj), (2, cfg.macro_ssj_wall)) if on)
    return TechGates(macros, bool(cfg.variant_switching), bool(cfg.hook_action))


BEHAVIOUR_KEYS = ("steps", "firing", "on_target", "firing_on_target", "enemy_visible", "close", "angle_sum", "yaw_err_sum", "dist_sum", "yaw_sum",
                  "pitch_steps", "pitch_sum", "pitch_signed_sum", "look_up_sum", "elev_steps", "elev_sum", "elev_abs_sum", "elev_over15", "pitch_err_sum",
                  "yaw_track", "yaw_track_n", "pitch_track", "pitch_track_n",
                  "look_free", "look_enemy", "look_gate", "slide_forced",
                  # STAGE S0 of docs/superpowers/specs/2026-09-20-speedrun-tech.md: the weapon channel, which
                  # `status.json` has never carried. PURE BOOKKEEPING -- nothing here is read by a reward, an
                  # observation or an action, and `tests/test_slot_counters.py` pins that the step outputs are
                  # byte for byte what they are with the counters removed.
                  #   slot_press     decisions that pressed ANY slot key (the action's `slot` is not 0)
                  #   slot_same      ... and the key pressed was the slot already held: the redraw (§3.4)
                  #   slot_switch    ... and it was a different slot
                  #   slot_known     decisions whose held slot the mod actually reported (the denominator for
                  #                  the per-slot shares; `weapon_slot` is -1 before `GunControl` starts, and
                  #                  is otherwise the 1-BASED slot KEY -- see `held_slot_key`)
                  #   slot_unowned   ... and the key pressed named a slot `player.slot_counts` says is EMPTY,
                  #                  so `SwitchWeapon` cannot move `currentSlotIndex` and no switch happens.
                  #                  0-1 acquires its weapons one pickup at a time, so this is most of the
                  #                  level at the start; it is the number that says how much of the sticky
                  #                  lever's switch budget would go on switches the game never performed.
                  "slot_press", "slot_same", "slot_switch", "slot_known", "slot_unowned",
                  # What the STICKY SLOT lever suppressed, both 0 while `sticky_weapon_slot` is False.
                  "slot_dropped", "slot_blocked",
                  # The weapon VARIATION held (`GunControl.currentVariationIndex`). The redraw §3.4 describes
                  # cycles it, so the sticky slot FREEZES whichever variation the episode happens to hold --
                  # and variation 0 is Piercer / Core Eject / Freezeframe (§3.5), the set every technique in
                  # the spec wants. Without this, a sticky-slot round that froze on the Marksman reads as
                  # "sticky slot is worse" when what was measured is "variation 1 is worse".
                  "variation_known",
                  *("held_variation_%d" % i for i in range(NUM_WEAPON_VARIATIONS)),
                  # Press counts for the three buttons the technique work cares about. `firing` above is
                  # fire1-OR-fire2 and cannot separate them, and nothing counted `punch` at all even though
                  # `rewards.punch` charges for it.
                  "press_fire1", "press_fire2", "press_punch",
                  # Both indexed by slot KEY - 1: entry 0 is key 1 (the revolver) and entry 5 is key 6, which
                  # the game has but no action can select. NOT by `weapon_slot` raw, which is the key itself.
                  *("held_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)),     # decisions each slot was held
                  *("kills_slot_%d" % i for i in range(NUM_WEAPON_SLOTS)),    # kills while each slot was held
                  # STAGE S7, the macro channel. All zero under v1, and `_info` emits them only under v2.
                  #   macro_request  decisions whose policy macro value was not 0 (raw use of the head)
                  #   macro_sent     ... and the gates forwarded it to the mod
                  #   macro_ran / macro_refused  the mod's `result` for a sent macro (refused = refused, degraded
                  #                  or disabled); macro_landed = ran with an accepted SSJ bucket
                  #   variant_request / hook_request  raw use of the other two heads, masked or not
                  "macro_request", "macro_sent", "macro_ran", "macro_refused", "macro_landed",
                  "variant_request", "hook_request")


def _slot_owned(player: dict[str, Any], slot: int) -> bool | None:
    """Does `player.slot_counts` say slot KEY `slot` (1..5) holds a weapon? None when it cannot answer.

    `ObservationBuilder.SlotCounts` sends one count per `GunControl.slots` entry, slot 1 first, and an EMPTY
    array until `GunControl` has started -- 0-1 has no weapon at all until the revolver pickup and acquires
    the rest one pickup at a time, so "empty" is the normal state for much of the level, not an error.

    Three outcomes, never two: True owned, False known-empty, None unknown (array absent, too short, or not
    numbers -- a mod older than v0.5.0, or a slot beyond what this build reports). Callers must not treat
    None as False; guessing "empty" would make the sticky lever pass every press through on an old mod.
    """
    counts = player.get("slot_counts")
    if not counts or not 1 <= slot <= len(counts):
        return None
    try:
        return int(counts[slot - 1]) > 0
    except (TypeError, ValueError):
        return None


def held_slot_key(player: dict[str, Any] | None) -> int | None:
    """The slot KEY (1..`NUM_WEAPON_SLOTS`) that re-selects the weapon in hand, or None when unknown.

    THE ONE PLACE `player.weapon_slot` IS INTERPRETED. It is `GunControl.currentSlotIndex` sent RAW by
    `ObservationBuilder.BuildPlayer`, and the game keeps that field **1-BASED**: it is seeded
    `PlayerPrefs.GetInt("CurSlo", 1)`, reset to `1` (never 0) whenever it runs past `slots.Count`, indexed
    everywhere as `slots[currentSlotIndex - 1]`, and compared against the `Slot1`..`Slot6` bindings as
    `currentSlotIndex != 1` .. `!= 6`. The mod's own `TechObserver.VariationsInSlot` and
    `EpisodeController`'s variation macro both subtract 1 before indexing, for the same reason. The only
    other value on the wire is **-1**, "GunControl has not started" -- on 0-1 there is no weapon at all
    until the revolver pickup, so that is the normal state for the opening of the level, not an error.

    So the key returned is the number the policy would press, and `weapon_slot` IS that number already: no
    +1 anywhere. Until 2026-09-20 the callers below read it as 0-based and compared `slot == weapon_slot + 1`,
    which made every counter name the wrong set -- see `docs/project-log.md` 2026-09-20.

    Anything outside 1..`NUM_WEAPON_SLOTS` (-1, a missing key, a non-number from a future mod) is None:
    unknown is a third outcome and callers must never guess a slot from it. Note that a key of 6 is a real
    game slot the policy can never press -- `NUM_WEAPON_CHOICES` stops the action at key 5 -- so it is
    reported honestly here and the callers handle it as "held, but no press can name it".
    """
    if not player:
        return None
    try:
        held = int(player.get("weapon_slot", -1))
    except (TypeError, ValueError):
        return None
    return held if 1 <= held <= NUM_WEAPON_SLOTS else None


_NO_SLOT = "<none>"  # the sentinel for "nothing to serialize against", never a real path


def acquire_relaunch_slot(directory: str, slots: int, stale_age_s: float,
                          deadline: float, *, now=time.monotonic, sleep=time.sleep,
                          wall=time.time) -> str | None:
    """Takes one of `slots` cross-process relaunch permits, or None if none came free in time.

    Twelve workers step in lockstep, so a fault that drops the games hits all twelve within one step and every
    one of them reaches the relaunch rung at the same moment. Twelve simultaneous cold Unity starts on a box
    that is already thrashing boot none of them inside the per-worker wait, so each `relaunch_one` returns
    False, each worker dies anyway -- and twelve games have been killed and restarted behind the supervisor's
    back for nothing. `games.launch` staggers its twelve cold starts by 4 s for the same reason.

    A permit is a file created with `O_EXCL` (atomic on Windows as on POSIX) holding the pid and the wall
    clock. A permit older than `stale_age_s` is assumed to belong to a worker that died holding it and is
    taken over, so a crash cannot wedge the rung shut for the rest of the run.

    Returns the permit path to pass to `release_relaunch_slot`, or None when the caller should give up the
    rung cleanly. With no `directory` (eval, tests, any env with no run log) there is nothing to serialize
    against and the permit is free: `_NO_SLOT`.
    """
    if not directory:
        return _NO_SLOT
    base = Path(directory)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _NO_SLOT  # an unwritable run directory may not be the reason a game is not restarted
    while True:
        freed = False
        for index in range(max(1, slots)):
            path = base / ("relaunch.%d.lock" % index)
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:  # a permit whose holder died is taken over rather than waited on forever
                    if wall() - path.stat().st_mtime > stale_age_s:
                        path.unlink()
                        freed = True
                except OSError:
                    pass
                continue
            except OSError:
                return _NO_SLOT
            try:
                os.write(fd, ("pid=%d t=%.0f\n" % (os.getpid(), wall())).encode("utf-8"))
            finally:
                os.close(fd)
            return str(path)
        if freed:
            continue  # a permit was just reclaimed; take it before sleeping or giving up
        if deadline - now() <= 0:
            return None
        sleep(min(2.0, max(0.1, deadline - now())))


def release_relaunch_slot(permit: str | None) -> None:
    if not permit or permit == _NO_SLOT:
        return
    try:
        os.unlink(permit)
    except OSError:
        pass


class BridgeRecovered(Exception):
    """Internal signal: the bridge was rebuilt in the middle of an episode.

    Carries the observation of the fresh level load that replaced the lost one. `step()` turns it into a
    truncated episode with `end_reason` "bridge_reset"; it never escapes the env.
    """

    def __init__(self, raw: dict[str, Any]):
        super().__init__("bridge rebuilt mid-episode")
        self.raw = raw


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
        if self.cfg.curriculum_weighting not in CURRICULUM_WEIGHTINGS:
            # Caught here, not swallowed: a typo would silently hand the run back to the rule that starved the
            # levels that were learning, and the dashboard would show no sign of it.
            raise ValueError(f"curriculum_weighting must be one of {CURRICULUM_WEIGHTINGS}, "
                             f"got {self.cfg.curriculum_weighting!r}")
        # The level is mutable from here on: a curriculum run changes it at a fresh load and nowhere else.
        self.level = (self.cfg.levels[0] if self.cfg.levels else self.cfg.level) if campaign else self.cfg.level
        if self.cfg.tech_layout not in TECH_LAYOUTS:
            # Caught here, not in a worker's first step: a typo would otherwise build v1 shapes under a config
            # everyone reads as v2.
            raise ValueError(f"tech_layout must be one of {TECH_LAYOUTS}, got {self.cfg.tech_layout!r}")
        tech = self.cfg.tech_layout == "v2"
        if tech and not campaign:
            raise ValueError("tech_layout v2 is campaign-only: Cyber Grind's 448 / 11 layout never widens")
        if self.cfg.layout.campaign != campaign or self.cfg.layout.tech != tech:
            # The layout follows the mode and the tech switch (Cyber Grind 448 inputs, campaign 479, tech 530).
            # Replaced, not edited in place: train.py builds each game's config with dataclasses.replace, so they
            # all share one layout object.
            self.cfg = replace(self.cfg, layout=replace(self.cfg.layout, campaign=campaign, tech=tech))

        self.observation_space = self.cfg.layout.space()
        # Campaign only: the look mode is appended as a 12th dimension, so Cyber Grind checkpoints stay loadable;
        # tech_layout v2 appends macro / variant / hook after it (15 dims, 57 logits).
        self.action_space = action_space(campaign, tech=tech)
        self._tech = tech
        self._gates = tech_gates(self.cfg)
        self._mod_features: frozenset = frozenset()  # hello.features at the last connect
        self._macro_reasons: dict[str, int] = {}  # per episode: why each sent macro did not run
        self._move_tech_missing = 0  # consecutive v2 frames with a player and no usable move_tech
        self._move_tech_warned = False

        self.client = BridgeClient(self.cfg.host, self.cfg.port,
                                   timeout=self.cfg.step_timeout_s, reset_timeout=self.cfg.reset_timeout_s)
        self._connected = False
        self._bridge_resets = 0  # times this env rebuilt its connection (episodes.jsonl end_reason bridge_reset)
        # Attribution: one line per reset, recovery attempt, relaunch and timeout, in runs/<run>/env_<port>.log.
        # `print` is not enough -- the supervisor's detached spawn discards the trainer's stdout entirely.
        self.envlog = EnvLog(env_log_path(self.cfg.env_log_dir, self.cfg.port), self.cfg.port)
        self._recovery_deadline: float | None = None  # set while ONE recovery is in flight, shared by its layers
        self._recovery_logged = False  # ... and whether anything has actually failed inside it yet
        self._recovery_reason = ""
        self._relaunches = 0  # times this env relaunched its OWN game instance
        self._mem_recycles = 0  # times it replaced its own game between episodes for leaking
        self._mem_recycle_off = False  # latched off if a freshly booted game is still over the limit

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
        # A speed stage's target time, in game seconds. The override wins; 0 leaves it None until the first
        # observation carrying `campaign.ranks` sets it (see `_note_speed_target`). None on every other run,
        # which is what makes `compute_reward` pay the plain `level_complete`.
        self._speed_target: float | None = (float(self.cfg.speed_target_seconds)
                                            if campaign and self.cfg.speed_bonus and self.cfg.speed_target_seconds > 0
                                            else None)
        # The level's RAW S-rank threshold, before `speed_target_scale`. Reported and recorded beside the target
        # so a sidecar says both what was demanded and what the game calls S; nothing is measured against it.
        self._s_rank_seconds: float | None = None

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
        # Patience is configured in game SECONDS and kept here in decisions, so the same config behaves the same
        # at any fixed_fps/frameskip. 20 s at 30/2 is 300 decisions, well under `stuck_seconds` 45.
        patience_steps = max(0, int(round(self.cfg.gate_target_patience_s * self.cfg.fixed_fps
                                          / max(1, self.cfg.frameskip))))
        self.gates = GateProgress(self.cfg.gate_reach_m, self.cfg.gate_reach_v_m, self.cfg.gate_min_gain_m,
                                  self.cfg.gate_hops_min_frac, self.cfg.target_kind_slots,
                                  patience_steps=patience_steps, unpark_m=self.cfg.gate_unpark_m,
                                  fallback_hysteresis_m=self.cfg.gate_fallback_hysteresis_m,
                                  route=self._route_for(self.level),
                                  route_exit_tol_m=self.cfg.route_exit_tol_m,
                                  route_seed_m=self.cfg.route_seed_m,
                                  patience_mode=self.cfg.gate_patience_mode,
                                  prefer_route_when_collapsed=self.cfg.prefer_route_when_collapsed,
                                  route_ground_m=self.cfg.route_ground_m)
        self.exit_guard = ExitGuard(self.cfg.exit_max_shift_m)
        self.path_progress = PathProgress()
        self._parks_at_start = 0  # GateProgress.parks when this episode began, so info reports the difference
        self._exit_banished = False  # the guard rejected at least one exit report during this episode
        self._rng = random.Random()
        self._stuck_streak = 0  # episodes in a row that ended stuck at the same current checkpoint
        self._stuck_checkpoint: str | None = None
        self._fresh_start = False  # this episode began with a fresh level load
        self._positions: list[list[float]] = []  # fresh-start episodes only, for best runs
        self._cells_new = 0
        self._oob_steps = 0  # steps with no ground under the player: off the map, or in a fall
        self._rescues = 0  # rescue teleports this episode, free ones (at 1 HP) included (_note_rescue)
        self._rescue_hp = 0.0  # HP those rescues removed: the quantity `RewardConfig.fall_hp` charges
        self._rescue_floored = 0  # rescues that took the player from above 1 HP down to 1
        self._hp_lost_other = 0.0  # HP lost on every other non-death step: enemies, mostly
        self._exit_dist_min = math.inf
        self._exit_ground_dist_min = math.inf
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
        # Decisions still to wait before the sticky-slot lever honours another switch. Always 0 while
        # `sticky_weapon_slot` is False, because nothing ever writes it: see `_sticky_slot`.
        self._slot_cooldown = 0
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
        # Clamped to the recovery budget when one is live: a connect that retried past the deadline is how the
        # "total" budget was overspent by a whole retry window.
        retry = self.cfg.connect_retry_s
        if self._recovery_deadline is not None:
            retry = self._clamp(retry, self._recovery_deadline)
        hello = self.client.connect(retry_seconds=retry)
        try:
            self._check_mod(hello)
        except BridgeIncompatible:
            # Hang up before raising. `_connected` stays False, so `close()` would skip this socket and the next
            # connect would overwrite it, open. No `release`: this client never took control of the game.
            self.client._drop()
            raise
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
            # So one worker's recovery does not get the other eleven games' clients dropped (see the field).
            command_timeout_s=int(self.cfg.mod_command_timeout_s),
            soft_death=self.cfg.soft_death and self.cfg.mode == "cybergrind",
            render=self.cfg.render,
            windowed=self.cfg.windowed,
            window_width=self.cfg.window_width,
            window_height=self.cfg.window_height,
            # Sent before the first reset: enemies and GunSetter read both when the level loads.
            difficulty=self.cfg.difficulty,
            unlock_all_gear=self.cfg.unlock_all_gear,
            **mod_layout,
            **self._tech_mod_config(),
        )
        self._connected = True

    def _required_features(self) -> set[str]:
        """The `hello.features` this config cannot run without. Empty under v1: a 0.7.2 DLL serves v1 as today."""
        return set(TECH_LAYOUT_FEATURES) if self._tech else set()

    def _check_mod(self, hello: dict[str, Any] | None) -> None:
        """Refuses, loudly and unrecoverably, a mod that cannot serve this config (see BridgeIncompatible)."""
        self._mod_features = mod_features(hello)
        need = self._required_features()
        if not need:
            return
        version = (hello or {}).get("mod_version")
        missing = sorted(need - self._mod_features)
        if missing:
            self.envlog.event("mod_incompatible", mod=version, missing=",".join(missing))
            message = (
                f"port {self.cfg.port}: tech_layout {self.cfg.tech_layout!r} needs mod features {missing}; the game "
                f"runs mod {version!r} with features {sorted(self._mod_features) or 'none'}. Install mod 0.8.0 "
                "(docs/commands.md, 'S7 -- the one break') or set tech_layout back to v1.")
            self._write_mod_incompatible(message, version, missing)
            raise BridgeIncompatible(message)
        self.envlog.event("mod_features_ok", mod=version, layout=self.cfg.tech_layout)

    def _write_mod_incompatible(self, message: str, version: Any, missing: list[str]) -> None:
        """`runs/<run>/MOD_INCOMPATIBLE`, so the refusal stops the FLEET and not just this worker (see
        BridgeIncompatible). Only a training run has a run directory -- `env_log_dir`, which train.py alone fills
        (`fill_run_dirs`) -- so an eval, a check script or a test only raises. Written, never deleted: a successful
        connect leaves an old file alone, because only the operator may declare the mod fixed.
        """
        if not self.cfg.env_log_dir:
            return
        path = Path(self.cfg.env_log_dir) / MOD_INCOMPATIBLE_FILE
        text = "\n".join([
            message,
            "mod_version: %s" % version,
            "features_seen: %s" % (", ".join(sorted(self._mod_features)) or "none"),
            "features_missing: %s" % ", ".join(missing),
            "tech_layout: %s" % self.cfg.tech_layout,
            "port: %d" % self.cfg.port,
            "written_at: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
            "While this file exists campaign_driver.py and supervise.py start no game and no trainer for this run "
            "(they exit with code 4). Remove %s after installing the mod (or after setting tech_layout back to "
            "v1); nothing removes it for you." % path,
        ]) + "\n"
        written = write_mod_incompatible(self.cfg.env_log_dir, text)
        self.envlog.event("mod_incompatible_file", path=written.as_posix() if written else None,
                          written=bool(written))

    def _tech_mod_config(self) -> dict[str, Any]:
        """The mod 0.8.0 settings, sent EXPLICITLY on every connect -- or nothing at all to a 0.7.x DLL.

        The mod's config is per GAME PROCESS, not per connection (docs/protocol.md): a private test that switched
        `variant_switching` on, or a v2 client before a rollback, would otherwise be inherited by whoever connects
        next. So every switch is sent with the value THIS config means. Against a DLL that advertises no
        `features` (0.7.x) the dict is empty and the configure call is today's, key for key.
        """
        if not self._tech and "obs.move_tech" not in self._mod_features:
            return {}
        return {
            "obs_move_tech": self._tech,  # block A
            "obs_weapon_tech": False,  # block B: S9
            "obs_projectiles": False,  # block C: S10
            "obs_input_clock": False,  # a diagnostic, never for training
            "macros": True,  # the master switch; a macro still runs only when an action asks for one
            "macro_ssj_wall": bool(self._tech and 2 in self._gates.macros),
            "allow_reserved_macros": False,
            "macro_wall_lead_unsafe": False,
            "variant_switching": bool(self._tech and self._gates.variant),
            "ssj_gap_s": SSJ_GAP_S,
            "ssj_indicator": False,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """One bounded retry against a rebuilt connection, then the failure is real and is raised.

        `_resilient_reset` already reconnects around the reset request itself; this outer layer catches the
        rest of a reset -- `_skip_locked`'s steps, `_enter_arena`'s -- so no part of starting an episode can
        kill a worker while the other eleven games are mid-rollout.

        The retry runs inside ONE recovery budget with the inner ladder, not on top of it: the budget opens
        HERE, at the outermost frame, before `_reset` runs, so `_resilient_reset` and this tail handler both
        JOIN it rather than each opening their own. A `reset()` therefore costs at most
        `bridge_recovery_budget_s` in total, however many layers of it end up retrying. Two nested ladders,
        each paying a full budget, are how a single bad reset became a 19-minute freeze.

        The first connect is inside the budget too. `_ensure_connected` runs at the top of `_reset`, so a game
        whose port is not listening yet -- a cold copy the supervisor's boot gate gave up on -- now gets the
        whole budget's patience through this handler instead of raising out of the worker.
        """
        started = time.monotonic()
        self.envlog.event("reset_start", level=self.level)
        deadline, owned = self._begin_recovery("reset")
        try:
            self._recycle_own_game_if_fat(deadline)
            try:
                out = self._reset(seed=seed, options=options)
            except BridgeRecovered as rebuilt:
                self._adopt_fresh_load(rebuilt.raw)
                out = self._reset(seed=seed, options=options)
            except RECOVERABLE as exc:
                self._note_bridge_reset("%s while resetting: %s" % (type(exc).__name__, exc))
                self._reconnect_within(deadline)
                out = self._reset(seed=seed, options=options)
        finally:
            self._end_recovery(owned, "reset_done")
        self.envlog.event("reset_end", level=self.level, dt_s=time.monotonic() - started)
        return out

    def _reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
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
            self._raw, _ = self._resilient_reset(checkpoint=False)
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
        self._macro_reasons = {}
        self._last_end_reason = ""
        self._wedge_run = 0
        self._wedged_steps = 0
        self._slide_latch = 0
        self._slot_cooldown = 0

        if self.cfg.mode == "campaign":
            self.archive.start_episode()
            self.path_progress.reset()
            self._positions = []
            self._cells_new = 0
            self._oob_steps = 0
            self._rescues = 0
            self._rescue_hp = 0.0
            self._rescue_floored = 0
            self._hp_lost_other = 0.0
            self._exit_dist_min = math.inf
            self._exit_ground_dist_min = math.inf
            self._parks_at_start = self.gates.parks
            self._exit_banished = self.exit_guard.banished  # a load that is already banished stays flagged
            self._start_checkpoint = self._current_checkpoint(self._raw)
            self._level_started = bool((self._raw.get("campaign") or {}).get("level_started"))
            self._note_speed_target(self._raw)
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
        """A bridge failure ends the episode instead of the run.

        Anything the bridge can throw -- a timed-out reply, a dropped socket, a game that is still booting --
        is turned into a truncated episode whose `end_reason` is "bridge_reset", after the connection has been
        rebuilt and the level reloaded. Before this, the exception escaped into SubprocVecEnv's `_worker`,
        which does not catch it, so the worker process died and took all twelve games' rollouts with it.
        """
        try:
            return self._step(action)
        except BridgeRecovered as rebuilt:
            return self._end_on_bridge_reset(rebuilt.raw)
        except RECOVERABLE as exc:
            # The budget opens HERE, before the first log line, so the event log says the fault began in a
            # step rather than inheriting the reason of the reset that started this episode, and so the whole
            # recovery -- ladder, relaunch and all -- is closed by this frame and not by one inside it.
            _, owned = self._begin_recovery("step")
            try:
                self._note_bridge_reset("%s while stepping: %s" % (type(exc).__name__, exc))
                raw, _ = self._resilient_reset(checkpoint=False, reconnect_first=True)
                return self._end_on_bridge_reset(raw)
            finally:
                self._end_recovery(owned, "step_done")

    def _step(self, action):
        prev = self._raw
        command = decode_action(action)
        # The look mode is resolved here, against the observation the policy acted on, and popped: the wire
        # `action` message is unchanged and the mod never sees it.
        look_mode = int(command.pop("look_mode", 0))
        # v2: the three tech heads come off the command HERE, so no raw policy value can reach the wire;
        # `_apply_tech` below puts back exactly what the gates allow. None under v1, and then nothing changes.
        tech = self._pop_tech(command)
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
        # THE S0 COUNTERS READ THE POLICY'S OWN SLOT INTENT, so they are taken BEFORE `_sticky_slot` rewrites
        # it. That ordering is the point: `slot_same_frac` is a fact about the POLICY (§3.4's 76% re-measured
        # on the live one), and it has to stay comparable before and after the sticky lever is switched on --
        # measured after the rewrite it would read ~0 by construction and say nothing. What the lever
        # suppressed is counted separately, inside `_sticky_slot`, as `slot_dropped` / `slot_blocked`.
        self._note_behaviour(prev, command, raw_pitch_cmd, applied_mode)
        self._sticky_slot(prev, command)
        self._apply_tech(command, tech)
        campaign = self.cfg.mode == "campaign"
        cur = self.client.step(command)
        # The mod reports a macro on THIS reply only. `_skip_locked` can replace `cur` with a later frame (an input
        # lock right after the step), so the report is taken first and put back for block D's packer.
        macro_report = cur.get("macro") if tech is not None else None
        # Counted BEFORE the lock is stepped through: a bridge fault inside `_skip_locked` must not leave a sent
        # macro counted with neither a ran nor a refused verdict.
        self._note_macro_result(macro_report)
        if campaign:
            cur = self._guard_exit(self._skip_locked(cur))
        if macro_report is not None and "macro" not in cur:
            cur["macro"] = macro_report
        self._raw = cur
        self._steps += 1
        self._lifetime_steps += 1
        if self._lifetime_steps % ARCHIVE_CHECK_EVERY == 0:
            self._save_archive_on_schedule()
        self._track_enemies(cur)
        self._note_slot_kills(prev, cur)

        player = cur.get("player")
        prev_player = prev.get("player") or {}
        died = player is None or player["dead"] or (
            player.get("soft_deaths", 0) > prev_player.get("soft_deaths", player.get("soft_deaths", 0))
        )
        # Milestones, novelty, route gates and path progress are measured before the reward, from the frame the
        # policy caused; `prev` is needed as well, because kills and style reset the stuck clock.
        campaign_step = self._campaign_progress(prev, cur, died=died) if campaign else None
        wedged = campaign and self._note_wedge(prev, cur)
        # Graded BEFORE the reward, because a speed stage's completion bonus needs the official time of this
        # very frame. It reads only `cur`, so nothing below it can change the answer.
        completed = bool(campaign and (cur.get("stats", {}).get("level_complete") or (cur.get("campaign") or {}).get("level_over")))
        # A speed stage scales `level_complete` by the clock, but only on a FRESH-START completion: a checkpoint
        # respawn begins partway through the level with the timer already running, so its "official time" says
        # nothing about how fast the level was played and it pays the plain weight, exactly as every other run
        # does. `_level_result` returns None for a time the game never reported (a frame from after the stats
        # reset, or a missing block), which `completion_bonus` reads as "no time" and also pays plain.
        official = (self._level_result(cur)["seconds"]
                    if (campaign and completed and self._fresh_start and self._speed_target) else None)
        reward = compute_reward(self.cfg.rewards, prev, cur, self._enemy_max_health, died=died,
                                campaign=campaign_step, buttons=command["buttons"],
                                target_seconds=self._speed_target, official_seconds=official)

        if campaign:
            if died and player is not None and not completed:
                # A death does not end a campaign episode: the penalty is paid above, then the player respawns at
                # the checkpoint and the level clock keeps running, as in real play.
                # Deliberately, a macro report is NOT carried onto the respawn frame (unlike the input-lock case):
                # that frame is a new life at the checkpoint, so block D reads "no macro" there, while the
                # counters above have already recorded the verdict.
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
        # What this step actually paid for finishing: 0 on every step but the completion edge, `level_complete`
        # on a plain completion, and the time-scaled figure on a speed stage's fresh-start one. The one number
        # that says the speed mechanism is live, next to `target_seconds`.
        info["completion_bonus"] = reward.parts.get("level_complete", 0.0)
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
            # STAGE S0: slot presses per GAME second, which is what the stage asks for and what is comparable
            # between runs at different `fixed_fps`/`frameskip`. Set beside `kills_per_min` because this is the
            # only place the episode's game-clock duration exists.
            info["slot_press_per_s"] = self._behaviour["slot_press"] / seconds if seconds > 0 else 0.0
            if campaign:
                self._end_campaign_episode(cur, reason, info)
        return self._pack(cur), float(reward.total), terminated, truncated, info

    def close(self) -> None:
        """Bounded teardown: this runs while eleven other workers wait to exit, and one sick game may not
        hold the trainer open. `BridgeClient.close` waits at most `close_timeout` for the release reply."""
        try:
            self._save_archive()
        finally:
            # Release the game even if the archive could not be written, so it never stays in lockstep.
            if self._connected:
                try:
                    self.client.close()
                except (OSError, BridgeError, ValueError) as exc:
                    self.envlog.event("close_failed", error="%s: %s" % (type(exc).__name__, exc))
                self._connected = False
            self.envlog.event("closed", bridge_resets=self._bridge_resets or None,
                              relaunches=self._relaunches or None)

    # Bridge recovery ---------------------------------------------------

    def _note_bridge_reset(self, message: str) -> None:
        """Counts a rebuilt bridge and says so, tagged with the port that broke.

        Written to `runs/<run>/env_<port>.log` as well as printed, because the supervisor spawns the trainer
        detached and a detached shell redirect silently discards stdout: for seven hours on 2026-09-17 every
        one of these lines went nowhere and the freeze could not be attributed. See the freeze gotcha.
        """
        self._bridge_resets += 1
        self._recovery_noticed()
        self.envlog.event("bridge_reset", n=self._bridge_resets, why=message)
        print("UltrakillEnv[%d]: %s (bridge reset #%d)" % (self.cfg.port, message, self._bridge_resets), flush=True)

    # -- the recovery budget ----------------------------------------------

    def _begin_recovery(self, reason: str) -> tuple[float, bool]:
        """Opens (or joins) the ONE wall-clock budget a recovery gets. Returns `(deadline, opened_here)`.

        Joining matters: `reset()` catches what `_resilient_reset` finally raises and calls it again, so
        without a shared deadline the budget is paid twice -- and the pair of them then outlast the supervisor,
        which is the whole failure.

        `opened_here` is what makes the deadline authoritative instead of advisory. Only the frame that OPENED
        a budget may close it. `_resilient_reset` used to close every budget it touched, including one opened
        above it, so `reset()`'s tail handler -- which fires when `_skip_locked` hits a frozen game after a
        reset that DID answer -- found a cleared deadline and opened a SECOND full budget. Measured on a fake
        clock before this change: two budgets, 1000->1540 and 1183->1723, for one fault.

        Silent on the happy path: every reset passes through here, and a reset that works first time is not a
        recovery. `_recovery_noticed` is what puts a line in the log, on the first thing that actually fails.
        """
        if self._recovery_deadline is None:
            self._recovery_deadline = time.monotonic() + max(1.0, self.cfg.bridge_recovery_budget_s)
            self._recovery_reason = reason  # the reason the budget OPENED, not the layer that joined it
            return self._recovery_deadline, True
        return self._recovery_deadline, False

    def _recovery_noticed(self) -> None:
        """Logs `recover_start` once per recovery, at the first failure rather than at every reset."""
        if not self._recovery_logged:
            self._recovery_logged = True
            self.envlog.event("recover_start", reason=self._recovery_reason,
                              budget_s=self.cfg.bridge_recovery_budget_s)

    def _end_recovery(self, owned: bool, outcome: str, **fields: Any) -> None:
        """Closes the budget, but only from the frame that opened it (see `_begin_recovery`)."""
        if not owned:
            return
        if self._recovery_logged:
            self.envlog.event("recover_end", outcome=outcome, **fields)
        self._recovery_deadline = None
        self._recovery_logged = False
        self._recovery_reason = ""

    def _budget_left(self, deadline: float) -> float:
        return deadline - time.monotonic()

    def _clamp(self, want: float, deadline: float) -> float:
        """`want`, cut down to what is left of the budget (never below 1 s, so a call is always really made).

        This is what turns the deadline from advisory into authoritative. A call admitted with a moment of
        budget left used to run its own full bound on top of it -- a reset timeout of 180 s past a spent
        budget, twice over if a connect went first -- so "bounded by 540 s" meant 780 s in the worst case.
        """
        return max(1.0, min(want, self._budget_left(deadline)))

    def _relaunch_own_game(self, deadline: float) -> bool:
        """Restarts THIS env's instance and waits for it to boot. Never touches another port.

        The rung below a reconnect: a game that answers nothing after three rebuilt connections is not going to
        start answering, and `games.relaunch_one` restarts exactly the one process listening on this env's port,
        leaving the other eleven games and their rollouts alone. A whole-set relaunch is the supervisor's job and
        costs four minutes; this costs one game.
        """
        if not self.cfg.bridge_relaunch:
            return False
        wait = min(self.cfg.bridge_relaunch_wait_s, max(0.0, self._budget_left(deadline)))
        if wait < 30.0:  # not enough budget left to boot a game; let the caller give up cleanly instead
            self.envlog.event("relaunch_skipped", budget_left_s=max(0.0, self._budget_left(deadline)))
            return False
        # De-synchronise the twelve workers that all reached this rung on the same step, then take one of the
        # few cross-process permits. Both exist so a fault that drops every game does not become twelve
        # simultaneous cold Unity starts; see `acquire_relaunch_slot`.
        stagger = (self.cfg.port % 4) * max(0.0, self.cfg.bridge_relaunch_stagger_s)
        if stagger:
            self._sleep_within(stagger, deadline)
        permit = acquire_relaunch_slot(self.cfg.env_log_dir, self.cfg.bridge_relaunch_slots,
                                       self.cfg.bridge_relaunch_wait_s + 120.0, deadline)
        if permit is None:
            self.envlog.event("relaunch_no_slot", budget_left_s=max(0.0, self._budget_left(deadline)))
            return False
        wait = min(self.cfg.bridge_relaunch_wait_s, max(0.0, self._budget_left(deadline)))
        if wait < 30.0:  # the queue ate the budget: give up the rung cleanly rather than half-boot a game
            release_relaunch_slot(permit)
            self.envlog.event("relaunch_skipped", budget_left_s=max(0.0, self._budget_left(deadline)))
            return False
        self._relaunches += 1
        self._recovery_noticed()
        self.envlog.event("relaunch_start", n=self._relaunches, wait_s=wait)
        try:
            ok = self._relaunch_hook(self.cfg.port, wait)
        except Exception as exc:  # noqa: BLE001 - a failed relaunch must not be a new kind of crash
            self.envlog.event("relaunch_failed", error="%s: %s" % (type(exc).__name__, exc))
            return False
        finally:
            release_relaunch_slot(permit)
        self.envlog.event("relaunch_end", ok=bool(ok))
        return bool(ok)

    def _recycle_own_game_if_fat(self, deadline: float) -> None:
        """Restarts this env's own game between episodes when it has leaked past the fleet-derived limit.

        Called from `reset()` and nowhere else, so the game is replaced at the one moment when nothing is
        lost: the episode has ended and a level load is about to happen regardless. `scripts/mem_guard.py`
        does the same job from outside for the cases this cannot reach (a worker wedged mid-episode, a
        run with `bridge_relaunch` off, a game nobody is stepping), and both derive the limit from the same
        `procmem.derive_game_limit`, so they cannot disagree about what "too fat" means.

        Every failure here is swallowed: a memory optimisation may not be a new way for a reset to die.
        """
        if self.cfg.game_memory_growth_gb <= 0 or not self.cfg.bridge_relaunch or self._mem_recycle_off:
            return
        try:
            sizes = self._fleet_memory_hook()
            mine = sizes.get(self.cfg.port)
            if not mine:
                return
            limit = derive_game_limit(sizes, int(self.cfg.game_memory_growth_gb * PROC_GB))
            if mine <= limit:
                return
            self.envlog.event("mem_recycle_start", private_mb=mine // PROC_MB, limit_mb=limit // PROC_MB)
            if not self._relaunch_own_game(deadline):
                return
            self._reconnect()  # the process it was talking to is gone; never reuse that socket
            after = self._fleet_memory_hook().get(self.cfg.port) or 0
            self._mem_recycles += 1
            self.envlog.event("mem_recycle_end", n=self._mem_recycles, before_mb=mine // PROC_MB,
                              after_mb=after // PROC_MB, freed_mb=(mine - after) // PROC_MB)
            print("UltrakillEnv[%d]: recycled a leaking game between episodes, %.2f -> %.2f GB (freed %.2f GB)"
                  % (self.cfg.port, mine / PROC_GB, after / PROC_GB, (mine - after) / PROC_GB), flush=True)
            if after > limit:
                # A fresh game that is already over the limit means the limit is wrong, not that the game is
                # fat. Relaunching again would be an infinite loop that never trains, so this env stops.
                self._mem_recycle_off = True
                self.envlog.event("mem_recycle_disabled", after_mb=after // PROC_MB, limit_mb=limit // PROC_MB)
        except BridgeIncompatible:
            # Not a memory problem: the relaunched game runs a mod this config cannot use. Loud here, rather
            # than logged as a mem_recycle_error and raised again by the reset's own connect.
            raise
        except Exception as exc:  # noqa: BLE001 - never turn a memory check into a failed reset
            self._mem_recycle_off = True
            self.envlog.event("mem_recycle_error", error="%s: %s" % (type(exc).__name__, exc))

    def _fleet_memory_hook(self) -> dict[int, int]:
        """port -> committed private bytes for every listening game. Replaced in tests; never opens a port."""
        import sys as _sys

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in _sys.path:
            _sys.path.insert(0, str(scripts))
        import games  # noqa: PLC0415 - lazy on purpose, exactly as `_relaunch_hook` does

        # Only ULTRAKILL processes. `listening_pids` is every listening port on the machine -- 19 of them with
        # no game running at all -- and a stray 1 GB service answering on some port would otherwise be taken
        # for the "freshest game" and drag the derived limit down onto healthy copies.
        game_pids = set(games.working_sets())
        sizes = {}
        for port, pid in games.listening_pids().items():
            if pid not in game_pids:
                continue
            size = private_bytes(pid)
            if size:
                sizes[port] = size
        return sizes

    def _relaunch_hook(self, port: int, wait_s: float) -> bool:
        """The live relaunch, then the boot gate. Split out so the tests replace it and never start a game.

        Imported lazily and by port: `games.launch`/`games.stop` both call `stop_all()`, which would take down
        every other game in the run, so only `relaunch_one` may ever be called from here.

        A listening port is NOT a booted game -- the plugin opens the bridge server long before Addressables
        finish, and a reset against a half-booted instance answers `unknown scene`. The working set is the
        signal the supervisor's own boot gate uses (a booted copy sits near 1 GB), so the same one is used
        here before the reset is attempted.
        """
        import sys as _sys

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in _sys.path:
            _sys.path.insert(0, str(scripts))
        import games  # noqa: PLC0415 - lazy on purpose: nothing else in the env needs it

        deadline = time.monotonic() + wait_s
        if not games.relaunch_one(port, timeout=max(1.0, deadline - time.monotonic())):
            return False
        want = self.cfg.bridge_relaunch_boot_mb * 1024 * 1024
        while time.monotonic() < deadline:
            pid = games.listening_pids().get(port)
            if pid is not None and games.working_sets().get(pid, 0) >= want:
                self.envlog.event("boot_gate", pid=pid, mb=self.cfg.bridge_relaunch_boot_mb)
                return True
            time.sleep(2.0)
        # Out of budget rather than out of hope: `_reset_while_booting` waits out `unknown scene` anyway.
        self.envlog.event("boot_gate_timeout", mb=self.cfg.bridge_relaunch_boot_mb)
        return True

    def _reconnect(self) -> None:
        """Rebuilds this env's connection to its OWN port and re-sends its config.

        `BridgeServer` accepts a new client and drops the old one, so a worker can reconnect to the game it
        already owns without touching any other worker's game -- each worker owns exactly one port. The dead
        socket is never reused: the reply the game was still writing when the client gave up would otherwise be
        read as the answer to the next request.
        """
        try:
            self.client.close()
        except (OSError, BridgeError, ValueError):
            pass
        self._connected = False
        self._ensure_connected()

    def _reconnect_within(self, deadline: float) -> None:
        """Rebuilds the connection, retrying while budget remains, and raises a RECOVERABLE error if it cannot.

        `reset()`'s tail handler used a bare `_reconnect()`, whose `connect` failure is a `BridgeClosed` that
        the handler is already inside the `except` of -- so a game that had not opened its port simply killed
        the worker. Here the retry is the budget's, and what finally comes out is still RECOVERABLE, so the
        caller's own contract (a bounded failure, not a surprise exception class) holds.

        The retry stops short of the relaunch reserve, because a port that never opens is precisely what the
        relaunch rung is for: spending the whole budget asking a dead port to answer, and never restarting the
        instance behind it, is the same unreachable-rung bug one layer up.
        """
        reserve = self.cfg.bridge_relaunch_reserve_s if self.cfg.bridge_relaunch else 0.0
        last = self._try_reconnect_until(deadline - max(0.0, reserve))
        if last is None:
            return
        if self._relaunch_own_game(deadline):
            last = self._try_reconnect_until(deadline)
            if last is None:
                return
        raise BridgeClosed("could not rebuild the bridge on port %d inside the recovery budget: %s"
                           % (self.cfg.port, last))

    def _try_reconnect_until(self, deadline: float) -> Exception | None:
        """Reconnects with a growing backoff until `deadline`. None on success, else the last error."""
        attempt = 0
        last: Exception | None = BridgeClosed("no time left to reconnect")
        while self._budget_left(deadline) > 0:
            attempt += 1
            try:
                self._reconnect()
                return None
            except (BridgeError, OSError) as exc:
                last = exc
                self._recovery_noticed()
                self.envlog.event("reconnect_failed", rung="tail", attempt=attempt,
                                  error="%s: %s" % (type(exc).__name__, exc))
                self._sleep_within(self.cfg.bridge_backoff_s * attempt, deadline)
        return last

    def _resilient_reset(self, *, checkpoint: bool, reconnect_first: bool = False) -> tuple[dict[str, Any], bool]:
        """One reset request that survives a booting game, a dead socket and a game that has to be relaunched.

        Returns `(observation, recovered)`. `recovered` is True when the connection had to be rebuilt, and then
        the observation is a FRESH level load whatever `checkpoint` asked for: after a reconnect the game's
        state is unknown, and a checkpoint respawn into an unknown state would leave the milestone and gate
        trackers describing a level the player is no longer in. Callers must treat a recovered observation as a
        fresh level load (see `_adopt_fresh_load`).

        **Bounded, in total.** The waiting is deliberate -- a blocked worker stalls every other game, and a
        stall of minutes costs one rollout where the crash it replaces cost the whole run -- but it now runs
        against one wall-clock deadline (`bridge_recovery_budget_s`) that covers every attempt, every backoff
        and the relaunch. Before that, three attempts of (close + connect + handshake + reset) at the old
        bounds composed to ~70 minutes for a single reset, seven times the supervisor's patience, so every
        recovery this was built for was killed mid-flight and paid for as a twelve-game restart.

        The ladder, in order, each rung only while budget remains:
          1. ask again on the connection there is (a booting game answers `unknown scene`; the cure is time);
          2. rebuild the connection and ask again, `bridge_retries` times with a growing backoff;
          3. relaunch THIS env's own game, wait for it to boot, rebuild and ask again.

        Rung 2 gets its own RESERVE (`bridge_relaunch_reserve_s`) rather than rung 1's leftovers. Rung 1's
        attempts each cost a full reset timeout against a game that answers nothing, so they always spent the
        whole budget and the relaunch -- the rung that exists for exactly that fault -- was never reached. The
        only shape that did reach it was an instantly-refused connection, which is the shape the test used and
        not the shape production sees.
        """
        deadline, owned = self._begin_recovery("reset" if reconnect_first else "episode_boundary")
        boot_poll = min(5.0, max(0.0, self.cfg.bridge_backoff_s))
        attempts = max(1, self.cfg.bridge_retries)
        recovered = False
        relaunched = False
        last: Exception | None = None
        # What rung 1 may spend, so rung 2 still has enough left to boot a game and ask it once.
        reserve = self.cfg.bridge_relaunch_reserve_s if self.cfg.bridge_relaunch else 0.0
        rung_deadline = deadline - max(0.0, reserve)

        for rung in (0, 1):
            if rung == 1:
                if not self._relaunch_own_game(deadline):
                    break
                relaunched, recovered, reconnect_first = True, True, True
                rung_deadline = deadline  # the last rung may use everything that is left
            boot_deadline = min(time.monotonic() + self.cfg.unknown_scene_wait_s, rung_deadline)
            for attempt in range(attempts):
                if self._budget_left(rung_deadline) <= 0:
                    self._recovery_noticed()
                    self.envlog.event("budget_spent", rung=rung, attempt=attempt + 1,
                                      reserved_s=reserve if rung == 0 else None)
                    last = last or BridgeTimeout("recovery budget spent before the bridge answered")
                    break
                if reconnect_first or recovered:
                    try:
                        self._reconnect()
                        recovered = True
                    except (BridgeError, OSError) as exc:
                        last = exc
                        self._recovery_noticed()
                        self.envlog.event("reconnect_failed", rung=rung, attempt=attempt + 1,
                                          error="%s: %s" % (type(exc).__name__, exc))
                        self._sleep_within(self.cfg.bridge_backoff_s * (attempt + 1), rung_deadline)
                        # `recovered` too, not just `reconnect_first`: a reconnect that FAILED leaves the
                        # game's state as unknown as one that succeeded, and `recovered` is what stops a
                        # checkpoint respawn into a level the player may no longer be in.
                        recovered, reconnect_first = True, True
                        continue
                reconnect_first = False
                try:
                    raw = self._reset_while_booting(checkpoint and not recovered, boot_deadline, boot_poll,
                                                    rung_deadline)
                    self._end_recovery(owned, "ok", rung=rung, relaunched=relaunched or None)
                    return raw, recovered
                except RECOVERABLE as exc:
                    last = exc
                    wait = self.cfg.bridge_backoff_s * (attempt + 1)
                    self._note_bridge_reset("%s on reset: %s; reconnecting in %.0fs (attempt %d/%d%s)"
                                            % (type(exc).__name__, exc, wait, attempt + 1, attempts,
                                               ", after a relaunch" if relaunched else ""))
                    self._sleep_within(wait, rung_deadline)
                    recovered, reconnect_first = True, True
            if relaunched or not self.cfg.bridge_relaunch or self._budget_left(deadline) <= 0:
                break

        self._end_recovery(owned, "failed", relaunched=relaunched or None,
                           budget_left_s=max(0.0, self._budget_left(deadline)))
        raise BridgeError("bridge on port %d did not come back within %.0fs (%d attempts%s): %s"
                          % (self.cfg.port, self.cfg.bridge_recovery_budget_s, attempts,
                             ", one relaunch" if relaunched else "", last))

    def _sleep_within(self, seconds: float, deadline: float) -> None:
        """A backoff never eats the budget it is backing off inside."""
        time.sleep(max(0.0, min(seconds, self._budget_left(deadline))))

    def _reset_while_booting(self, checkpoint: bool, deadline: float, poll: float,
                             budget_deadline: float | None = None) -> dict[str, Any]:
        """Sends the reset, waiting out a game that has not finished booting.

        A half-booted instance answers every reset `unknown scene '<level>'`, because
        `EpisodeController.SceneExists` searches Addressables locators that stay empty until boot completes
        while the bridge port is already listening. That is not a bad level name and not a broken connection:
        it is a game that needs another minute, so it is asked again rather than raised at the trainer.

        The request is CLAMPED to what is left of `budget_deadline`, so the reset in flight when the recovery
        budget expires ends with the budget instead of adding a full reset timeout on top of it.
        """
        while True:
            try:
                bound = (self.cfg.reset_timeout_s if budget_deadline is None
                         else self._clamp(self.cfg.reset_timeout_s, budget_deadline))
                return self.client.reset(self.scene, checkpoint=checkpoint, timeout=bound)
            except BridgeSceneUnknown as exc:
                if time.monotonic() >= deadline:
                    raise
                print("UltrakillEnv[%d]: %s -- the game is still booting, retrying in %.0fs"
                      % (self.cfg.port, exc, poll), flush=True)
                time.sleep(poll)

    def _adopt_fresh_load(self, raw: dict[str, Any]) -> None:
        """Makes `raw`, the observation of a recovery reload, this env's state, as a fresh level load.

        A recovery reload IS a fresh level load: the scene came back from the top, so every checkpoint, arena,
        door and gate the level ships with is in its starting state. `new_level_load` re-baselines the trackers
        on exactly that, so nothing the reload reveals can be paid a second time and nothing already paid is
        forgotten -- the same guarantee `_campaign_reset` gives a normal fresh start. The exploration archive
        keeps its per-game visit counts (they span episodes and carry the 1/sqrt(N) decay) and only starts a new
        episode's first-entry set; `reset()` starts another, which is idempotent.
        """
        self._raw = raw
        self._last_end_reason = "bridge_reset"
        if self.cfg.mode != "campaign":
            return
        self.exit_guard.new_level_load()  # a recovery reload IS a fresh level load, for the exit as for the gates
        self._guard_exit(raw)
        pos = (raw.get("player") or {}).get("pos")
        self.milestones.new_level_load(raw.get("campaign"))
        self.gates.new_level_load(raw.get("campaign"), pos)
        self.gates.reset_episode()
        self.gates.retarget(raw.get("campaign"), pos)
        self.path_progress.reset()
        self.archive.start_episode()
        self._fresh_start = True
        self._stuck_streak, self._stuck_checkpoint = 0, None
        self._positions = []
        self._episode_start_stats = dict(raw.get("stats", {}))
        self._episode_start_seconds = (raw.get("campaign") or {}).get("seconds")
        self._steps_since_progress = 0
        self._wedge_run = 0
        self._slide_latch = 0
        self._slot_cooldown = 0

    def _end_on_bridge_reset(self, raw: dict[str, Any]):
        """Truncates the episode the bridge failure destroyed, and hands back the fresh level load.

        Pays exactly 0. The step that lost the connection has no observation to grade -- there is no `cur` to
        compare against `prev` -- and paying nothing is also what keeps the milestone accounting honest: the
        reload is adopted as a fresh level load, so no gate, checkpoint, arena or door is paid twice or lost.

        `info` is built from the LAST GOOD frame, not from the reload, so the episode's own counters (kills,
        gates reached, cells, deaths) still describe the episode that was lost. Grading them against the
        reloaded scene would subtract this episode's start stats from a level that had just started again.
        """
        last_good = self._raw
        info = self._info(last_good)
        info["reward_parts"] = {}
        info["end_reason"] = "bridge_reset"
        info["reset_seconds"] = self._reset_seconds
        seconds = self._steps * self.cfg.frameskip / self.cfg.fixed_fps
        info["episode_seconds"] = seconds
        info["kills_per_min"] = info["kills"] / seconds * 60.0 if seconds > 0 else 0.0
        info["slot_press_per_s"] = self._behaviour["slot_press"] / seconds if seconds > 0 else 0.0
        self._last_end_reason = "bridge_reset"
        if self.cfg.mode == "campaign":
            self._end_campaign_episode(last_good, "bridge_reset", info)
        self._adopt_fresh_load(raw)
        return self._pack(raw), 0.0, False, True, info

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

    def _sticky_slot(self, prev: dict[str, Any], command: dict[str, Any]) -> None:
        """STAGE S5, and DORMANT: `sticky_weapon_slot` is False, so this returns before reading anything else.

        Two rules, both of which only ever turn a slot press into "keep" -- this can add no press and can never
        change which slot is selected, only whether a selection happens at all:

          1. A press of the slot ALREADY HELD becomes `slot: 0` (keep). In game that press re-draws the weapon
             (`WeaponRedrawBehaviour` 0 = cycle variation), which sets `gunReady = false` until the `ReadyGun()`
             animation event, so a policy that presses it constantly suppresses its own primary fire (§3.4).
          2. A switch to a DIFFERENT slot is honoured at most once per `sticky_slot_switch_every` decisions,
             because a switch draws too. The counter runs over DECISIONS, not game seconds, and is reset by a
             level load and by a respawn along with `_slide_latch`. A press naming a slot `slot_counts` says
             is EMPTY is not a switch at all -- the game cannot honour it -- so it is passed through and
             costs nothing; see the `_slot_owned` branch below.

        Rule 1 cannot fire while slot KEY 6 is held, because the action space stops at slot KEY 5
        (`NUM_WEAPON_CHOICES` 6 = keep plus keys 1..5) while the game has six slots. That is correct rather
        than a gap: the policy has no way to press key 6, so it can never re-draw slot 6, and every key it
        CAN press while holding slot 6 is a genuine switch to a different slot.

        The action that was refused is NOT replaced by anything: the step goes out with `slot: 0`, which is the
        action the policy could already have chosen, so the wire message stays inside the existing protocol and
        the mod needs no change. Nothing here is charged or paid for -- the same rule the macro design states
        for a refusal (§4.2): charging teaches the policy to avoid the channel rather than to use it well.

        The held slot comes from `held_slot_key`, which is already the KEY that re-selects it (1-based, like
        `GunControl.currentSlotIndex` itself), and the action's `slot` is 0 for keep and 1..5 for a slot KEY,
        so "the slot already held" is simply `slot == key`. With no player, or a `weapon_slot` of -1 (before
        `GunControl` has started -- 0-1 has no weapon at all until the revolver pickup), nothing is known
        about what is held and the action is passed through untouched.
        """
        if not self.cfg.sticky_weapon_slot:
            return
        # The cooldown is read as it stood at the START of this decision and spent at the end of it, so
        # `sticky_slot_switch_every: 3` means "honour, refuse, refuse, honour" -- three decisions between two
        # honoured switches. Decrementing before the test would spend this decision's own tick on itself and
        # give every OTHER decision a switch at any setting, which is what the first draft of this did.
        was_cooling = self._slot_cooldown > 0
        if was_cooling:
            self._slot_cooldown -= 1
        slot = int(command.get("slot", 0))
        if not slot:
            return
        key = held_slot_key(prev.get("player"))
        if key is None:
            return  # nothing known about what is held: never guess, pass the press through
        if slot == key:
            command["slot"] = 0
            self._behaviour["slot_dropped"] += 1
        elif was_cooling:
            command["slot"] = 0
            self._behaviour["slot_blocked"] += 1
        elif _slot_owned(prev.get("player") or {}, slot) is False:
            # THE COOLDOWN IS CHARGED FOR A SWITCH, AND A PRESS OF AN EMPTY SLOT IS NOT ONE.
            # `GunControl.SwitchWeapon` cannot move `currentSlotIndex` into a slot with no weapon in it, so
            # nothing is drawn and nothing is suppressed -- there is no cost to ration. Charging here would
            # spend the whole switch budget on switches the game never performed and cut real switching by up
            # to `sticky_slot_switch_every` times, while `slot_blocked_frac` read as though the cooldown were
            # doing its job. That would corrupt the S5 round rather than break the run: the judged metrics
            # could not separate "sticky slot did not help" from "the cooldown was spent on non-events".
            # How often this happens is measured PASSIVELY and right now, as `slot_unowned_frac` -- S5 may
            # not be switched on until that number has been read off a live status.json.
            # `None` (an old mod, or a `slot_counts` too short to answer) is NOT treated as empty: unknown
            # falls through and is charged, which is the conservative half of the rule.
            pass
        else:
            self._slot_cooldown = max(0, int(self.cfg.sticky_slot_switch_every) - 1)

    def _pop_tech(self, command: dict[str, Any]) -> dict[str, Any] | None:
        """Takes the three v2 heads off the decoded command. None under v1 (`decode_action` adds them at width 15)."""
        tech = None if "macro" not in command else {
            "macro": int(command.pop("macro")), "variant": int(command.pop("variant")),
            "hook": bool(command.pop("hook"))}
        # A 12-wide action on a v2 env, or a 15-wide one on v1, is a layout mismatch: fail loudly, never silently
        # drop (or silently act on) the tech heads.
        assert (tech is not None) == self._tech, (
            f"a tech_layout {self.cfg.tech_layout} env got an action {'with' if tech else 'without'} the tech heads")
        return tech

    def _apply_tech(self, command: dict[str, Any], tech: dict[str, Any] | None) -> None:
        """Puts on the wire exactly what the gates allow, and counts every request whether or not it was sent.

        A masked value leaves the message as the v1 action message; it is never charged (spec §4.2: a charge
        teaches the policy to avoid the head rather than to learn its preconditions).
        """
        if tech is None:
            return
        b = self._behaviour
        if tech["macro"]:
            b["macro_request"] += 1
            if tech["macro"] in self._gates.macros:
                command["macro"] = tech["macro"]
                b["macro_sent"] += 1
        if tech["variant"]:
            b["variant_request"] += 1
            if self._gates.variant:
                command["variant"] = tech["variant"]
        if tech["hook"]:
            b["hook_request"] += 1
            if self._gates.hook:
                command["buttons"] = [*command["buttons"], "hook"]

    def _note_macro_result(self, report: dict[str, Any] | None) -> None:
        """The mod's verdict on a sent macro. Refusal reasons are kept per episode: a macro refused ~always is a
        design bug (§4.2), and the reason says which precondition.

        `macro_landed` is counted through `rewards.macro_landed`, the ONE landed-bucket reader: the counter, obs
        index 521 and the S8 gate can never disagree about what landed.
        """
        if not isinstance(report, dict):
            return
        b = self._behaviour
        if report.get("result") == "ran":
            b["macro_ran"] += 1
            if macro_landed(report):
                b["macro_landed"] += 1
        else:
            b["macro_refused"] += 1
            reason = str(report.get("reason") or report.get("result") or "unknown")
            self._macro_reasons[reason] = self._macro_reasons.get(reason, 0) + 1

    def _note_tech_obs(self, raw: dict[str, Any]) -> None:
        """Warns ONCE per env when a v2 env keeps receiving frames with a player and no usable `move_tech`.

        Block A then packs 0.0 -- the old-DLL vector, so nothing breaks -- but the policy is blind to exactly what
        the break was for. The handshake already refused a DLL without `obs.move_tech`; this catches a flag lost
        inside the game (a per-process config another client overwrote) or a renamed field.
        """
        if self._move_tech_warned or not raw.get("player"):
            return
        move = raw.get("move_tech")
        names = [name for name, *_ in MOVE_TECH_FIELDS]
        # A field sent as null is missing too: the packer reads it as 0.0, exactly like an absent one.
        missing = [n for n in names if move.get(n) is None] if isinstance(move, dict) else names
        if not missing:
            self._move_tech_missing = 0
            return
        self._move_tech_missing += 1
        if self._move_tech_missing >= MOVE_TECH_GRACE:
            self._move_tech_warned = True
            what = "no move_tech block" if not isinstance(move, dict) else "move_tech without " + ",".join(missing)
            self.envlog.event("move_tech_missing", frames=self._move_tech_missing, what=what)
            print(f"WARNING port {self.cfg.port}: {self._move_tech_missing} frames with a player and {what}; "
                  "block A packs 0.0 (warned once per env)", flush=True)

    def _note_slot_kills(self, prev: dict[str, Any], cur: dict[str, Any]) -> None:
        """STAGE S0: kills attributed to the weapon slot that was held when the shot went out.

        Bookkeeping only. `kills` is the game's own cumulative counter, which ROLLS BACK on a checkpoint
        respawn (measured: 133 rollbacks over 126 deaths in runs/probe_0-2_speed), so the same `max(0, delta)`
        `compute_reward` uses is used here -- a rollback contributes nothing rather than a negative count. The
        slot read is the PREVIOUS frame's, because that is the weapon that fired; a kill landing on the frame
        of a switch is credited to the weapon that fired it, not to the one being drawn.

        `kills_slot_<i>` is indexed by KEY - 1, so entry 0 is slot key 1 (the revolver) and entry 5 is key 6:
        see `held_slot_key` for why the raw field is not the index.
        """
        new_kills = max(0, cur.get("stats", {}).get("kills", 0) - prev.get("stats", {}).get("kills", 0))
        if not new_kills:
            return
        key = held_slot_key(prev.get("player"))
        if key is not None:
            self._behaviour["kills_slot_%d" % (key - 1)] += new_kills

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
        # STAGE S0. Separate press counts for the three buttons the technique work reads (`firing` above is the
        # OR of the two fire buttons and cannot tell them apart), and the slot channel §3.4 says is probably
        # suppressing the agent's own fire. Counting only; nothing below is read by a reward or an observation.
        buttons = command["buttons"]
        for name in ("fire1", "fire2", "punch"):
            if name in buttons:
                self._behaviour["press_" + name] += 1
        slot = int(command.get("slot", 0))
        if slot:
            self._behaviour["slot_press"] += 1
        player = raw.get("player")
        if not player:
            return
        key = held_slot_key(player)
        if key is not None:
            # `held_slot_key` returns the 1-based slot KEY (`GunControl.currentSlotIndex` is 1-based) or None
            # before `GunControl` starts, so `slot_known` is the honest denominator for every per-slot share
            # below, and `held_slot_<i>` is indexed by KEY - 1: entry 0 is slot key 1, entry 5 is key 6.
            self._behaviour["slot_known"] += 1
            self._behaviour["held_slot_%d" % (key - 1)] += 1
            if slot:
                # The action's `slot` is already a KEY, so the redraw press is `slot == key` -- no offset.
                self._behaviour["slot_same" if slot == key else "slot_switch"] += 1
        if slot and _slot_owned(player, slot) is False:
            # A press the game CANNOT honour. Counted here, passively and whether or not the sticky lever is
            # on, because it is the measurement that decides whether the lever's ownership gate matters: it is
            # `slot_unowned_frac` that says how much of the policy's slot channel is aimed at empty slots.
            self._behaviour["slot_unowned"] += 1
        variation = player.get("weapon_variation", -1)
        if 0 <= variation < NUM_WEAPON_VARIATIONS:
            self._behaviour["variation_known"] += 1
            self._behaviour["held_variation_%d" % variation] += 1
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

    def _route_for(self, level: str) -> dict | None:
        """This level's offline room trunk, or None -- which is layer 1 or layer 3, i.e. today's behaviour.

        None for every Cyber Grind env, for `route_fallback: false`, and for the 21 campaign levels that ship
        no file (the 18 the gate ladder already routes plus 1-3, 5-4 and 6-2). `GateProgress` built with
        `route=None` is byte for byte the tracker without this spec, which is what makes acceptance check A3
        on Level 0-1 a fair test.
        """
        if self.cfg.mode != "campaign" or not self.cfg.route_fallback:
            return None
        return load_route(level, self.cfg.route_dir)

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
        left cannot force a reload on the level being entered. The route trunk is level-scoped too and is
        swapped here: it is read from disk once per scene (`_read_route` is cached), so a curriculum that
        revisits a level does not re-parse its file.
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
        self.gates.set_route(self._route_for(new_level))
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
        return choose_level(self._rng, list(self.cfg.levels), stats, floor=self.cfg.level_weight_floor,
                            rule=self.cfg.curriculum_weighting, cap=self.cfg.curriculum_weight_cap,
                            blocked_fresh_episodes=self.cfg.curriculum_blocked_fresh_episodes)

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
        raw, recovered = self._resilient_reset(checkpoint=not fresh)
        if recovered:
            # A rebuilt connection always reloads the level from the top, so this episode is a fresh start
            # whatever `choose_fresh_start` asked for; anything else would mark milestones paid against a level
            # load that no longer exists.
            fresh = True
        raw = self._skip_locked(raw)
        if fresh:
            # Before the guard sees the block: a real level load puts the FinalPit back, so whatever this fresh
            # scene reports is the truth however far it is from the exit the last load ended on.
            self.exit_guard.new_level_load()
        raw = self._guard_exit(raw)
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

        `MilestoneTracker` is deliberately left on `mark_paid` even on the reload branch. Its keys are rounded
        positions, identical across loads, so re-paying them would make "die with no checkpoint" a way to earn
        `checkpoint` and `arena_clear` again for ground already covered -- a farm. `GateProgress` is held to the
        same rule by `new_level_load(keep_paid=True)`: the ladder restarts, the payments do not.
        """
        before = self._raw.get("stats", {})
        raw, recovered = self._resilient_reset(checkpoint=True)
        if recovered:
            # The socket was rebuilt, so the level reloaded from the top and this episode's context -- its start
            # stats, its best-run positions, its gate approach -- describes a level load that is gone. Ending
            # the episode is the only honest move; step()'s wrapper turns this into end_reason "bridge_reset".
            raise BridgeRecovered(raw)
        raw = self._guard_exit(self._skip_locked(raw))
        self.milestones.mark_paid(raw.get("campaign"))
        player = raw.get("player")
        pos = (player or {}).get("pos")
        # Was there a checkpoint to respawn at? With none, `StatsManager.Restart` reloaded the WHOLE level and
        # the player is back at the spawn, so the ladder has to start again: `mark_paid` would leave `best_hops`
        # at whatever the pre-death attempt reached, `_choose_target` would point at a rung far ahead of a
        # player standing at the start, and `gate` could never pay again for this load. The bug predates the
        # route fallback -- it exists on every gates ladder -- and a 13-rung trunk makes it bite much harder
        # (spec §6). `new_level_load` is the honest call: the level really did load again.
        # Gated on the block being PRESENT: the mod omits the whole `campaign` block for a step whose build
        # throws, and an absent block has no `checkpoints` either, which would read as "no checkpoint" and wipe
        # a ladder that was fine. Reading it as an ordinary respawn is the safe direction and self-corrects at
        # the next episode boundary, where `choose_fresh_start` sees no checkpoint and forces a fresh load.
        reloaded = raw.get("campaign") is not None and self._current_checkpoint(raw) is None
        if reloaded:
            # `keep_paid` is what stops the honest call from becoming a farm, and it is the SAME rule the
            # paragraph above applies to `MilestoneTracker`: the episode is continuing, so `paid_hops`,
            # `paid_fallback` and `best_dist` survive while the ladder itself starts again. Without it a death
            # here re-arms `gate` and `gate_approach` for the whole prefix already walked -- measured at 30 / 30
            # / 30 over three laps of a 3-rung FakeLevel trunk, against a `death` of 5 and with no bound but
            # `max_steps`. Worse, reaching the level's first checkpoint moves every later death onto the branch
            # below and ends the stream for good, so the gradient would point AWAY from the first checkpoint.
            # The re-walk still pays for genuinely new ground: `paid_hops` is a floor, not a lock.
            self.gates.new_level_load(raw.get("campaign"), pos, keep_paid=True)
        else:
            # No reset_episode here: the episode continues, so the gate approach it has already earned stands.
            self.gates.mark_paid(raw.get("campaign"), pos)
        self.gates.retarget(raw.get("campaign"), pos)
        # A respawn moves the player, so the pre-death stuck clock says nothing about where they are now.
        self._steps_since_progress = 0
        self._wedge_run = 0
        self._slide_latch = 0
        self._slot_cooldown = 0
        self._track_enemies(raw)
        if player and reloaded:
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

    def _guard_exit(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Runs the banished-exit guard over an observation the env is about to adopt, in place.

        Called on every campaign observation that becomes `self._raw` -- a reset, a step, a respawn, a recovery
        reload -- because the exit is read in three places (the gate target, observation slots 0-4 and
        `exit_dist_min`) and rewriting the block once covers all of them. `ExitGuard.apply` is a no-op without a
        `campaign` block or an exit, so a Cyber Grind frame or a dropped block costs one dict lookup.
        """
        if self.cfg.mode == "campaign" and self.exit_guard.apply(raw.get("campaign")):
            self._exit_banished = True
        return raw

    @staticmethod
    def _ground_drop(raw: dict[str, Any]) -> float | None:
        """Metres of ground below the player, or None when the observation does not say at all.

        Split out of `_ground_point` because its two consumers need different things from the same
        reading. Exploration wants a POINT and treats "no ground within the ray" as "no point", so
        `_ground_point` folds the off-the-map case into the same None. `GateProgress._on_ground` has
        to tell the two apart -- an observation with no ray at all is unanswerable and stays
        permissive, while a ray that ran its full `ground_ray_length` without hitting anything is a
        definite "nothing under you" and must fail the test. So the raw drop is what is returned here
        and each caller applies its own rule.
        """
        centre = raw.get("ground_ray_center")
        rays = raw.get("ground_rays") or ()
        if centre is None and not rays:
            return None
        return float(centre) if centre is not None else min(rays)

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
        drop = self._ground_drop(raw)
        if drop is None or drop >= self.cfg.layout.ground_ray_length - 0.5:
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

    def _note_rescue(self, prev: dict[str, Any], raw: dict[str, Any], died: bool) -> float:
        """Counts a rescue teleport on this step and returns the HP it removed (0.0 on any other step).

        THE GAME'S RULE (decompiled/DeathZone.cs). A zone with `notInstakill` does not kill: it hurts the player
        by `damage` (50) while hp > damage, by hp - 1 while hp > 1, and only `FakeHurt`s at 1 HP, then puts
        them back on the walkway. So two falls from full health leave 1 HP, and every fall after that is free.
        On Level 0-1 on Brutal 58% of deaths (18 of 31 recorded) were one-hit kills set up that way, a
        median ~300 decisions after the fall that caused them.

        THE SIGNATURE, measured (see RESCUE_JUMP_M): a move of 12 m or more in one decision on a step that is
        not a death. Exclusions, each load-bearing:
          - `died` is the env's own verdict (a dead or missing player, or a soft-death increment), so the lethal
            hit is never counted here -- `damage_taken` and `death` already price it.
          - A checkpoint respawn is applied AFTER the reward and becomes the next step's `prev`, so its
            teleport is never compared against the frame before it; and a respawn heals, so it could not charge.
          - An HP RISE across a jump is counted as a rescue but charges nothing.
        Every other HP drop on a non-death step lands in `_hp_lost_other`, the reading that would show a policy
        trading rescue HP for enemy HP (the per-HP charge's one known asymmetry).
        """
        player = raw.get("player")
        before = prev.get("player")
        if died or not player or not before or player.get("dead"):
            return 0.0
        hp, hp_before = player.get("hp"), before.get("hp")
        pos, pos_before = player.get("pos"), before.get("pos")
        if hp is None or hp_before is None or pos is None or pos_before is None:
            return 0.0
        lost = float(hp_before) - float(hp)
        if math.dist(pos, pos_before) >= RESCUE_JUMP_M:
            self._rescues += 1
            if lost > 0.0:
                self._rescue_hp += lost
                if float(hp) <= 1.0 < float(hp_before):
                    self._rescue_floored += 1
                return lost
            return 0.0
        if lost > 0.0:
            self._hp_lost_other += lost
        return 0.0

    def _campaign_progress(self, prev: dict[str, Any], raw: dict[str, Any], died: bool = False) -> CampaignStep:
        """What the level did this step. Any progress restarts the stuck clock."""
        camp = raw.get("campaign") or {}
        checkpoints, arenas, doors, pickups, placements = self.milestones.update(raw.get("campaign"))
        novelty = 0.0
        oob = 0  # the same condition `_oob_steps` counts, carried into the reward (rewards.RewardConfig.oob)
        player = raw.get("player")
        pos = player["pos"] if player else None
        # Kills and style restart the stuck clock (see below) and, for the same reason, suspend the gate
        # tracker's patience clock: a door an ActivateArena wave is holding shut cannot be approached at all.
        stats, before = raw.get("stats", {}), prev.get("stats", {})
        fought = stats.get("kills", 0) > before.get("kills", 0) or stats.get("style", 0) > before.get("style", 0)
        # The ground reading rides along so `GateProgress` can refuse to credit a ROOM-TRUNK rung the
        # player is only touching from the air -- a room centroid's reach cylinder can overhang the
        # wall of the room it names (Level 0-3). It is inert on every gate and on every level with no
        # trunk; see `GateProgress._on_ground`.
        gates_new, approach = self.gates.update(
            raw.get("campaign"), pos, fought=fought,
            ground=((player or {}).get("grounded"), self._ground_drop(raw)) if player else None)
        if player:
            self._last_pos = list(pos)
            ground = self._ground_point(raw)
            if ground is None:
                oob = 1
                self._oob_steps += 1
            else:
                novelty = self.archive.visit(ground)
            if novelty > 0:
                self._cells_new += 1
            if self._fresh_start:
                self._positions.append(self._rounded(pos))
            if camp.get("exit"):
                self._exit_dist_min = min(self._exit_dist_min, math.dist(pos, camp["exit"]["pos"]))
                # ... and the same measure to the STANDABLE point near the pit (mod 0.7.2). `exit_dist_min`
                # keeps its definition, so its history stays comparable across this change, but it is a
                # distance to a transform 61-75 m below the floor and therefore has a floor of its own that
                # it can never go under. This one really does approach zero as the agent reaches the exit.
                ground_exit = exit_ground_point(camp["exit"])
                if ground_exit:
                    self._exit_ground_dist_min = min(self._exit_ground_dist_min,
                                                     math.dist(pos, ground_exit))
        # Measured, never fed to the stuck clock below: like `oob`, a fall must keep ticking toward `stuck`.
        rescue_hp = self._note_rescue(prev, raw, died)
        if camp.get("level_started"):
            self._level_started = True
        path_gain = self.path_progress.update(camp.get("path"))
        # Kills and style restart the clock too: four of 0-1's gate doors sit behind kill gates in locked rooms
        # where novelty runs out, and the live probe recorded 193 s of fighting after checkpoint 3 that the clock
        # truncated. Damage dealt deliberately does NOT count -- rewards.py attributes none of it to the player
        # and ULTRAKILL enemies damage each other, so a crossfire tail would simply never truncate.
        if (checkpoints or arenas or doors or pickups or placements
                or novelty > 0 or path_gain > 0 or gates_new or approach > 0 or fought):
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        return CampaignStep(checkpoints=checkpoints, arenas=arenas, doors=doors, novelty=novelty, path_gain=path_gain,
                            gates=gates_new, gate_approach=approach, item_pickups=pickups, item_placements=placements,
                            oob_steps=oob, rescue_hp=rescue_hp)

    def _note_speed_target(self, raw: dict[str, Any]) -> None:
        """Reads the level's own S-rank time off the first observation that carries one, once.

        Only on a speed stage, and only while the raw threshold has not been read yet, so the numbers are
        constants for the run and a first load whose campaign block was missing still gets them at the next
        reset.

        The TARGET is `s_rank_seconds * speed_target_scale` (§8): an S-rank time is a competent human run, not a
        fast one, and the 0-2 specialist was already inside it before any speed stage existed. An override from
        the config is a decision already made -- it is set in `__init__`, is never overwritten here, and is never
        scaled -- but the raw threshold is still recorded beside it, because a sidecar that says "target 82 s"
        without saying what S is cannot be read a month later.
        """
        if not self.cfg.speed_bonus or self._s_rank_seconds is not None:
            return
        s_rank = s_rank_time(raw.get("campaign"))
        if s_rank is None:
            return
        self._s_rank_seconds = s_rank
        if self._speed_target is None:
            scaled = s_rank * float(self.cfg.speed_target_scale)
            self._speed_target = scaled if scaled > 0.0 else None  # a zero or negative scale means "no target"
        self.envlog.event("speed_target", level=self.level, s_rank_seconds=s_rank,
                          scale=self.cfg.speed_target_scale, target_seconds=self._speed_target)

    def _level_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Official time, kills, style, restarts and rank as the game's results screen would count them.

        `seconds` is None -- and with it `rank`, which is computed FROM the clock -- whenever the frame carries
        no usable official time (`times.valid_official_seconds`). That is not hypothetical: on 2026-09-19 a
        completion frame on `spec_0-2_speed` arrived after the game's own level stats had reset and reported
        0.0 s with 3 restarts behind a 4,120-decision episode. A missing time must stay missing all the way
        out: it pays the plain completion bonus, writes no best run, and reaches `info` as None so that
        nothing downstream can mistake it for a record. `seconds_raw` keeps what the frame actually said, for
        the log line only.
        """
        camp = raw.get("campaign") or {}
        stats = raw.get("stats", {})
        raw_seconds = camp.get("seconds", stats.get("seconds", 0.0))
        seconds = valid_official_seconds(raw_seconds)
        restarts = camp.get("restarts", stats.get("restarts", 0))
        kills, style = stats.get("kills", 0), stats.get("style", 0)
        ranks = camp.get("ranks")
        rank = compute_rank(seconds, kills, style, restarts, ranks) if (ranks and seconds is not None) else None
        return {"seconds": seconds, "seconds_raw": raw_seconds, "kills": kills, "style": style,
                "restarts": restarts, "rank": rank}

    def _end_campaign_episode(self, raw: dict[str, Any], reason: str, info: dict[str, Any]) -> None:
        # The difficulty the game ACTUALLY read this episode, straight from the mod's campaign block rather
        # than from `cfg.difficulty`: the config asks, `CampaignPatches.DifficultyOverride` answers, and only
        # the answer is worth recording. On EVERY campaign episode, not just completions, so the episode log
        # says what a stalled or dead episode was playing on too, and so `ProgressCallback` can tell that the
        # difficulty changed under it without waiting for the first completion (2026-09-20, the Brutal switch).
        info["difficulty"] = (raw.get("campaign") or {}).get("difficulty")
        if reason == "level_complete":
            info["completed"] = 1
            if self._fresh_start:
                # Only a fresh load has a meaningful official time: the timer carries across respawn episodes.
                result = self._level_result(raw)
                info["level_seconds"] = result["seconds"]
                info["restarts"] = result["restarts"]
                info["rank"] = result["rank"]
                if result["seconds"] is None:
                    # The completion was real; only its clock is not. Logged with what the frame did say and
                    # how many restarts it counted, because that pair is the only evidence of WHY -- the
                    # 2026-09-19 row read seconds 0.0 with restarts 3, i.e. the stats had already reset.
                    self.envlog.event("official_time_missing", level=self.level,
                                      seconds=result["seconds_raw"], restarts=result["restarts"],
                                      steps=self._steps)
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
        if result["seconds"] is None:
            return  # no official time, no best run: `save_best_run` ranks on `seconds`, and a 0.0 wins forever
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
        if self._tech:
            self._note_tech_obs(raw)
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
        # STAGE S0, the weapon channel (docs/superpowers/specs/2026-09-20-speedrun-tech.md §3.4 and stage S0).
        # BOOKKEEPING ONLY, and in BOTH modes: a weapon slot is not a campaign idea. The denominator for the
        # first three is every decision of the episode, which is the denominator §3.4's 76% was measured with
        # (39,371 of 51,772 steps), so `slot_same_frac` is directly comparable to it.
        info["slot_press_frac"] = b["slot_press"] / steps   # pressed any slot key
        info["slot_same_frac"] = b["slot_same"] / steps     # ... the slot ALREADY HELD: the redraw
        info["slot_switch_frac"] = b["slot_switch"] / steps  # ... a different slot
        # ... and the share of all decisions that pressed a slot `slot_counts` reported EMPTY. The game cannot
        # switch into an empty slot, so these presses do nothing at all -- and the sticky lever must not spend
        # its switch budget on them. High here means the ownership gate in `_sticky_slot` is load-bearing.
        info["slot_unowned_frac"] = b["slot_unowned"] / steps
        info["fire1_frac"] = b["press_fire1"] / steps
        info["fire2_frac"] = b["press_fire2"] / steps
        info["punch_frac"] = b["press_punch"] / steps
        # What the sticky-slot lever suppressed. Both 0.0 on every run with `sticky_weapon_slot` False, which
        # is every run today, so these two columns are how "the lever is live" is read off status.json.
        info["slot_dropped_frac"] = b["slot_dropped"] / steps
        info["slot_blocked_frac"] = b["slot_blocked"] / steps
        # Time share per held slot, and kills per held slot. LISTS, so they travel to episodes.jsonl through
        # EPISODE_LOG_RAW rather than through `_num` -- six columns each in status.json would be six columns
        # nobody reads, while the single scalar below (how concentrated the held weapon was) is chartable.
        # ENTRY i IS SLOT KEY i + 1: entry 0 is the revolver. Readings taken before 2026-09-20 are shifted one
        # place right of this (entry 1 was the revolver) -- see `held_slot_key` and the project log.
        known = max(1, b["slot_known"])
        held_slots = [b["held_slot_%d" % i] for i in range(NUM_WEAPON_SLOTS)]
        info["slot_held_frac"] = [n / known for n in held_slots]
        info["slot_held_top_frac"] = max(held_slots) / known
        info["slot_kills"] = [b["kills_slot_%d" % i] for i in range(NUM_WEAPON_SLOTS)]
        info["slot_known_frac"] = b["slot_known"] / steps  # ... and how much of the episode had a known slot
        # WHICH VARIATION the episode held, on the same pattern: the list per episode, one scalar in
        # status.json. `variation0_frac` is the share of known-variation decisions holding variation 0 --
        # Piercer, Core Eject, Electric Railcannon, Freezeframe (§3.5), the set the technique work is built
        # on. The sticky slot freezes the variation for the episode, so this is the control reading that
        # separates "sticky slot is worse" from "the round happened to sit on the Marksman".
        var_known = max(1, b["variation_known"])
        info["held_variation_frac"] = [b["held_variation_%d" % i] / var_known for i in range(NUM_WEAPON_VARIATIONS)]
        info["variation0_frac"] = b["held_variation_0"] / var_known
        info["variation_known_frac"] = b["variation_known"] / steps
        if self._tech:
            # STAGE S7, the macro channel -- v2 only, so a v1 info dict is byte for byte what it was. Per decision,
            # like the slot counters. `macro_request_frac` is the policy's raw use of the head (0.10 at the 5 x 2%
            # prior); `macro_sent_frac` what the gates let through (M1 alone at S7); `macro_ran_share` the share of
            # sent macros the mod RAN -- near 0 is a design bug (§4.2), and `macro_refusal_reasons` says why.
            info["macro_request_frac"] = b["macro_request"] / steps
            info["macro_sent_frac"] = b["macro_sent"] / steps
            info["macro_ran_frac"] = b["macro_ran"] / steps
            info["macro_refused_frac"] = b["macro_refused"] / steps
            info["macro_landed_frac"] = b["macro_landed"] / steps
            # None, not 0.0, for an episode that sent nothing: status.json averages this ratio per episode, and a
            # zero from every no-send episode would drag the one "near 0 = design bug" metric toward 0.
            info["macro_ran_share"] = b["macro_ran"] / b["macro_sent"] if b["macro_sent"] else None
            info["variant_request_frac"] = b["variant_request"] / steps
            info["hook_request_frac"] = b["hook_request"] / steps
            info["macro_refusal_reasons"] = dict(self._macro_reasons)
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
            # The four rescue readings (`_note_rescue`): the mechanism metrics of the speed stage's `fall_hp`
            # weight. Episode totals, never fractions, and carried to episodes.jsonl through EPISODE_LOG_RAW --
            # deliberately NOT in CAMPAIGN_INFO_KEYS, so the Monitor's columns are exactly what they were.
            info["rescues"] = self._rescues
            info["rescue_hp"] = self._rescue_hp
            info["rescue_floored"] = self._rescue_floored
            info["hp_lost_other"] = self._hp_lost_other
            info["exit_dist_min"] = None if math.isinf(self._exit_dist_min) else self._exit_dist_min
            info["exit_ground_dist_min"] = (None if math.isinf(self._exit_ground_dist_min)
                                            else self._exit_ground_dist_min)
            info["completed"] = 0  # set to 1 on the step that ends with level_complete
            info["level_seconds"] = None  # official time, only for a fresh-start completion
            # The speed stage's two numbers. `target_seconds` is the level's own S-rank time (or the config's
            # override) and is None on every run that is not a speed stage; `completion_bonus` is overwritten in
            # step() with what the completion edge actually paid. Both numeric, so both survive `_num`.
            info["target_seconds"] = self._speed_target
            info["s_rank_seconds"] = self._s_rank_seconds
            info["completion_bonus"] = 0.0
            # Route gates. `gates_reached` is numeric and always present so it can be charted; `gate_hops_best`
            # is None until a gate is reached, so it only ever reaches episodes.jsonl. Both are inherited by a
            # respawn episode from its level load, so judge them on fresh starts.
            info["gates_reached"] = self.gates.gates_reached
            info["gate_hops_best"] = self.gates.best_hops
            # The two mechanisms of the 2026-09-17 patience/exit-guard spec, so the monitor can see them fire.
            # `targets_parked` counts parks during THIS episode (the tracker's counter is env-lifetime); it is
            # expected to be 0 on a monotone level such as 0-1 and to rise on a collapsed one such as 0-3.
            info["targets_parked"] = self.gates.parks - self._parks_at_start
            info["exit_banished"] = int(self._exit_banished)
            # The detector's verdict for the level load this episode ran on: 1 when the gate ladder collapses at
            # the spawn, which is what switches parking on at all under `gate_patience_mode: collapsed`. A
            # constant per level, so a level whose row reads anything but 0 or 1 means workers disagree about it.
            # Undecided (a load whose block never carried a ladder) reports 0, which is how it behaves.
            info["ladder_collapsed"] = int(bool(self.gates.ladder_collapsed))
            # Which of the three route layers drove this level load: 0 exit vector, 1 the mod's gate ladder,
            # 2 the offline room trunk. An INTEGER here because everything in CAMPAIGN_INFO_KEYS goes through
            # Monitor(info_keywords=...) and ProgressCallback._num, which writes None for anything float()
            # rejects -- the string belongs in EPISODE_LOG_RAW, next to `level` and `end_reason`. Level-load
            # scoped, so a respawn episode inherits it, which is correct: the layer is a fact about the load.
            info["route_source"] = self.gates.route_source
            info["route_source_name"] = ROUTE_SOURCE_NAMES.get(self.gates.route_source, "none")
            info["wedged_steps"] = self._wedged_steps
            info["level_started"] = int(self._level_started)
            info["look_free_frac"] = b["look_free"] / steps
            info["look_enemy_frac"] = b["look_enemy"] / steps
            info["look_gate_frac"] = b["look_gate"] / steps
            info["slide_forced_frac"] = b["slide_forced"] / steps
            info["start_checkpoint"] = self._start_checkpoint
            # Lifetime count for this worker, so a game whose bridge keeps breaking is visible in episodes.jsonl
            # next to end_reason "bridge_reset" rather than only in the training log.
            info["bridge_resets"] = self._bridge_resets
            pos = player.get("pos") or self._last_pos
            info["end_pos"] = self._rounded(pos) if pos else None  # where the episode actually ended up
        return info
