"""Campaign helpers: level names, rank maths, and the measures behind the campaign rewards.

Plain Python plus numpy (for the archive file), so all of it is tested without the game in
tests/test_campaign.py. The inputs come from the mod's `campaign` observation block (docs/protocol.md).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import zipfile
from pathlib import Path

import numpy as np


def _replace_file(tmp: Path, path: Path) -> None:
    """os.replace(tmp, path), retried briefly as progress.write_json_atomic does.

    On Windows the replace fails with PermissionError while another process has `path` open for a moment (another
    training game reading the same best-run file, an editor, a virus scan).
    """
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def _acquire_lock(lock_path: Path, timeout: float, poll: float) -> bool:
    """Creates `lock_path` exclusively as a mutex; returns whether it was acquired within `timeout` seconds.

    `os.open` with O_CREAT | O_EXCL is atomic across processes (unlike a read-then-write), so only one caller can
    hold the lock at a time; the rest retry until the holder removes it. A stale lock (left behind by a process
    that was killed mid-write) must never wedge every future save for a level, so the wait is bounded: giving up
    just means this one call skips its write, same as losing the race it exists to prevent.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)


# ---------------------------------------------------------------------------
# Levels and ranks
# ---------------------------------------------------------------------------

# The 35 main levels in mission order (StatsManager.levelNumber 1..35, see GetMissionName). Prime Sanctums and
# encores are out of scope.
CAMPAIGN_LEVELS: tuple[str, ...] = (
    "Level 0-1", "Level 0-2", "Level 0-3", "Level 0-4", "Level 0-5",
    "Level 1-1", "Level 1-2", "Level 1-3", "Level 1-4",
    "Level 2-1", "Level 2-2", "Level 2-3", "Level 2-4",
    "Level 3-1", "Level 3-2",
    "Level 4-1", "Level 4-2", "Level 4-3", "Level 4-4",
    "Level 5-1", "Level 5-2", "Level 5-3", "Level 5-4",
    "Level 6-1", "Level 6-2",
    "Level 7-1", "Level 7-2", "Level 7-3", "Level 7-4",
    "Level 8-1", "Level 8-2", "Level 8-3", "Level 8-4",
    "Level 9-1", "Level 9-2",
)

RANK_LETTERS = ("D", "C", "B", "A", "S")


def safe_name(scene: str) -> str:
    """A scene name usable in file names ("Level 0-1" -> "Level_0-1")."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scene)


def grade(thresholds: list[int], value: float, reverse: bool) -> int:
    """One rank category scored the way StatsManager.GetRanks does it: 0 (D) to 4 (S).

    Thresholds are checked in order and counting stops at the first one missed; meeting all of them scores 4.
    `reverse` is for time, where lower is better (`value <= t`); kills and style need `value >= t`.
    """
    for i, t in enumerate(thresholds):
        met = value <= t if reverse else value >= t
        if not met:
            return i
    return 4


def compute_rank(seconds: float, kills: int, style: int, restarts: int, ranks: dict) -> str:
    """The level rank StatsManager.GetFinalRank gives without cheats or major assists: "D".."S", or "P".

    `ranks` is the campaign block's {"time": [4], "kills": [4], "style": [4]}. The three category scores are
    summed and restarts subtracted (floored at 0); 12 is P, anything else rounds total / 3 to a letter. Thirds
    never land on .5, so round-half-up matches Unity's RoundToInt.
    """
    total = (
        grade(ranks["time"], seconds, reverse=True)
        + grade(ranks["kills"], kills, reverse=False)
        + grade(ranks["style"], style, reverse=False)
    )
    total = max(0, total - int(restarts))
    if total == 12:
        return "P"
    return RANK_LETTERS[math.floor(total / 3 + 0.5)]


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

EXPLORE_LOG_SCALE = math.log(1001.0)  # a cell entered in 1000 episodes reads 1.0 on the exploration map


class ExplorationArchive:
    """Visit counts over a grid of cubic cells in one level, kept per game.

    `counts[cell]` is the number of episodes that entered the cell (the current one included, once it has).
    `visit` pays novelty 1/sqrt(N + 1) on a cell's first entry in an episode, so a cell pays 1.0 the first time
    any episode reaches it and less each time after. `features` shows the policy the same counts around the
    player as a 9-value map.
    """

    def __init__(self, cell_size: float = 4.0):
        self.cell_size = float(cell_size)
        self.counts: dict[tuple[int, int, int], int] = {}
        self._episode: set[tuple[int, int, int]] = set()

    def cell(self, pos) -> tuple[int, int, int] | None:
        """The integer cell containing `pos`, or None when a coordinate is NaN or infinite.

        `math.floor` raises ValueError on NaN and OverflowError on infinity. This runs every step on the live
        player position (via visit and features), so a single glitched physics frame (a NaN from a bad raycast, an
        infinite velocity spike) must read as "no cell" instead of crashing the whole training run.
        """
        s = self.cell_size
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        return (math.floor(x / s), math.floor(y / s), math.floor(z / s))

    def start_episode(self) -> None:
        self._episode.clear()

    def visit(self, pos) -> float:
        """Novelty for being at `pos`: 1/sqrt(N + 1) on the first entry of its cell this episode, else 0.

        A non-finite `pos` has no cell (see `cell`) and pays no novelty. It must not be recorded either: letting a
        `None` cell into `_episode`/`counts` would mark every future non-finite step as "already visited" and, on
        the next save(), a `None` key would break the array reshape that expects 3-tuples.
        """
        cell = self.cell(pos)
        if cell is None:
            return 0.0
        if cell in self._episode:
            return 0.0
        self._episode.add(cell)
        n = self.counts.get(cell, 0)
        self.counts[cell] = n + 1
        return 1.0 / math.sqrt(n + 1)

    @property
    def episode_cells(self) -> int:
        """Cells entered so far this episode."""
        return len(self._episode)

    def features(self, pos, yaw_deg: float) -> list[float]:
        """The exploration map: the player's cell, then the 8 horizontal neighbours one cell_size away.

        Neighbour k sits at yaw + 45k degrees (k = 0 straight ahead, then clockwise to the right), the same yaw
        convention as spaces.yaw_frame, so the map turns with the player. Each value is min(1, log1p(N) / log(1001)).

        A non-finite `pos` or `yaw_deg` reads as 0 for every point it touches instead of raising: `cell()` returns
        None for a non-finite point, and `counts.get(None, 0)` is just a dict miss, so no special-casing is needed
        here beyond `cell()`'s own guard.
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        points = [(x, y, z)]
        for k in range(8):
            a = math.radians(yaw_deg + 45.0 * k)
            points.append((x + self.cell_size * math.sin(a), y, z + self.cell_size * math.cos(a)))
        return [min(1.0, math.log1p(self.counts.get(self.cell(p), 0)) / EXPLORE_LOG_SCALE) for p in points]

    def save(self, path) -> None:
        """Writes the counts to an .npz file atomically (a temp file, then a rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = np.array(list(self.counts.keys()), dtype=np.int32).reshape(-1, 3)
        counts = np.array(list(self.counts.values()), dtype=np.int64)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:  # a file object, so numpy does not append .npz to the temp name
            np.savez(f, cells=cells, counts=counts, cell_size=np.float64(self.cell_size))
        _replace_file(tmp, path)

    @classmethod
    def load(cls, path, cell_size: float) -> "ExplorationArchive":
        """The archive saved at `path`, or an empty one when the file is missing, unreadable or used another cell size."""
        archive = cls(cell_size)
        try:
            with np.load(Path(path)) as data:
                if float(data["cell_size"]) != archive.cell_size:
                    return archive
                cells, counts = data["cells"], data["counts"]
            loaded = {(int(c[0]), int(c[1]), int(c[2])): int(n) for c, n in zip(cells, counts)}
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            return archive
        archive.counts = loaded
        return archive


# ---------------------------------------------------------------------------
# Milestones, route gates, path progress and episode starts
# ---------------------------------------------------------------------------


def _dist3(a, b) -> float:
    """3-D distance between two [x, y, z] sequences (tolerant of lists, tuples and numpy rows)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class MilestoneTracker:
    """Pays each checkpoint, arena clear and door unlock once per level load.

    The mod reports arenas and doors as rounded-position keys. A checkpoint respawn re-instantiates rooms at the
    same position, so an arena cleared again after a death gives the same key and cannot pay twice. Keys that
    show up during a reset or respawn (Restart() unlocks the checkpoint's doors) are absorbed with `mark_paid`
    instead of paid. A checkpoint counts once it is activated or current.
    """

    def __init__(self):
        self._checkpoints: set[str] = set()
        self._arenas: set[str] = set()
        self._doors: set[str] = set()

    @staticmethod
    def _keys(campaign: dict | None) -> tuple[set[str], set[str], set[str]]:
        if not campaign:
            return set(), set(), set()
        checkpoints = {str(cp["id"]) for cp in campaign.get("checkpoints") or () if cp.get("activated") or cp.get("current")}
        return checkpoints, set(campaign.get("cleared_arenas") or ()), set(campaign.get("unlocked_doors") or ())

    def new_level_load(self, campaign: dict | None) -> None:
        """Starts a level load: forgets what was paid, then absorbs whatever the fresh level already reports."""
        self._checkpoints.clear()
        self._arenas.clear()
        self._doors.clear()
        self.mark_paid(campaign)

    def mark_paid(self, campaign: dict | None) -> None:
        """Absorbs the block's current keys without paying for them."""
        self.update(campaign)

    def update(self, campaign: dict | None) -> tuple[int, int, int]:
        """(new checkpoints, new arena clears, new door unlocks): keys not yet paid or absorbed this level load."""
        checkpoints, arenas, doors = self._keys(campaign)
        new = (len(checkpoints - self._checkpoints), len(arenas - self._arenas), len(doors - self._doors))
        self._checkpoints |= checkpoints
        self._arenas |= arenas
        self._doors |= doors
        return new

    @property
    def checkpoints_reached(self) -> int:
        """Distinct checkpoints activated in this level load (paid or absorbed)."""
        return len(self._checkpoints)


GATE_EXIT_KEY = "exit"  # the sentinel target key for campaign.exit; a real Door key is always "x,y,z"


class GateProgress:
    """Route progress along the door graph the mod reports as `campaign.gates` (docs/protocol.md).

    Each gate carries `hops`: the rooms still to traverse after passing it, 0 being the door into the exit's
    room. That ladder is the route signal the NavMesh never gave us (`path.status` was never once `complete` in
    2.9M steps), so the reward and the observation both hang off it:

      - `gate` pays once per level load for every new lower `hops` value reached, so a shortcut from 9 to 6 pays
        three instalments and re-walking the same doors after a death pays nothing;
      - `gate_approach` pays metres of new best closeness to the *current target*, which is the nearest active
        gate one rung down the ladder (or the exit once rung 0 is reached).

    `best_dist` is keyed per gate, seeded the first time a key becomes the target and never re-seeded, so
    walking A -> B -> A across a multi-gate tier pays A's approach once and B's once and nothing after. The
    earlier design's single scalar, re-seeded on every target change, was a farming loop worth more than
    finishing the level.

    Scope, and the one place this differs from the design spec (lead ruling R1): the `gate` ladder
    (`best_hops` / `paid_hops` / `reached`) is LEVEL-LOAD scoped exactly like `MilestoneTracker`, but
    `best_dist` is EPISODE scoped -- cleared by `reset_episode`, not by a death respawn inside an episode.
    PPO returns are computed within an episode and a truncation bootstraps the same state, so ending an episode
    early is never profitable, and most episodes are checkpoint respawns that need the dense signal on ground
    earlier episodes already covered. It is still not farmable: an episode's total approach is bounded by the
    gate-to-gate polyline either way (723-736 m on 0-1).

    `gates[].controller_active` is reported by the mod (a door whose `ActivateArena` has not run is neither
    locked nor approachable) but no rule here reads it yet; every field is read with `.get`, so a 0.5.x mod
    without it -- or without `gates` at all -- simply produces no target and pays nothing.
    """

    def __init__(self, reach_m: float = 8.0, reach_v_m: float = 6.0, min_gain_m: float = 0.5):
        self.reach_h = float(reach_m)
        self.reach_v = float(reach_v_m)
        self.min_gain = float(min_gain_m)
        # Per level load
        self.best_hops: int | None = None  # lowest hops reached
        self.paid_hops: int | None = None  # lowest hops already paid or absorbed
        self.reached: set[str] = set()  # gate keys ever reached this level load
        self.hops_reached: set[int] = set()  # their hop values, which is what `gates_reached` counts
        # Per episode
        self.best_dist: dict[str, float] = {}  # gate key (or "exit") -> closest approach made this episode
        self.target: dict | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def new_level_load(self, campaign: dict | None, player_pos=None) -> None:
        """Starts a level load: forgets the ladder, then absorbs whatever the fresh level already reports."""
        self.best_hops = self.paid_hops = None
        self.reached.clear()
        self.hops_reached.clear()
        self.best_dist.clear()
        self.target = None
        self.mark_paid(campaign, player_pos)

    def mark_paid(self, campaign: dict | None, player_pos=None) -> None:
        """Absorbs the gates the player already stands at (a checkpoint respawn, or any reset) without paying."""
        self._note_reached(campaign, player_pos)
        self.paid_hops = self.best_hops
        # best_dist is deliberately untouched: a respawn inside an episode must not re-earn the approach.

    def reset_episode(self) -> None:
        """A new episode: pick the target again from scratch, and let it re-earn its approach (ruling R1)."""
        self.target = None
        self.best_dist.clear()

    def retarget(self, campaign: dict | None, player_pos=None) -> None:
        """Chooses the current target. Pays nothing; call it before packing any observation."""
        self._note_reached(campaign, player_pos)
        self.target = self._choose_target(campaign, player_pos)
        key = self._key_of(self.target)
        if key is not None and player_pos is not None and key not in self.best_dist:
            self.best_dist[key] = _dist3(player_pos, self.target["pos"])  # seeded once, never re-seeded

    # -- per step ----------------------------------------------------------------------------

    def update(self, campaign: dict | None, player_pos=None) -> tuple[int, float]:
        """(new hop values crossed, metres of new best closeness to the target) for this step."""
        prev_key = self._key_of(self.target)
        prev_best = self.best_dist.get(prev_key, math.inf) if prev_key is not None else math.inf
        self.retarget(campaign, player_pos)

        paid = 0
        if self.best_hops is not None:
            if self.paid_hops is None:
                paid = 1
            elif self.best_hops < self.paid_hops:
                paid = self.paid_hops - self.best_hops
            self.paid_hops = self.best_hops if self.paid_hops is None else min(self.paid_hops, self.best_hops)

        approach = 0.0
        key = self._key_of(self.target)
        # Only a target that did not change this step can pay: the step a target changes has no comparable
        # baseline, and `prev_best` is infinite until a target has been seeded with the player's position.
        if key is not None and key == prev_key and player_pos is not None and math.isfinite(prev_best):
            distance = _dist3(player_pos, self.target["pos"])
            if distance < prev_best - self.min_gain:
                approach = prev_best - distance
                self.best_dist[key] = distance
        return paid, approach

    @property
    def gates_reached(self) -> int:
        """Distinct hop values reached in this level load (0 when none)."""
        return len(self.hops_reached)

    # -- helpers -----------------------------------------------------------------------------

    @staticmethod
    def _gates(campaign: dict | None) -> list[dict]:
        """The ordered gate array, or nothing at all when the mod did not order this level (0-5, or a 0.5.x mod)."""
        if not campaign or not campaign.get("gates_ordered"):
            return []
        return [g for g in (campaign.get("gates") or ()) if isinstance(g, dict) and g.get("pos")]

    @staticmethod
    def _key_of(target: dict | None) -> str | None:
        return None if target is None else str(target.get("key"))

    @staticmethod
    def _exit(campaign: dict | None) -> dict | None:
        """The exit as a target, or None when the mod reports no `FinalPit`. `hops` None marks it as the exit."""
        exit_ = (campaign or {}).get("exit")
        if not exit_ or not exit_.get("pos"):
            return None
        return {"key": GATE_EXIT_KEY, "pos": list(exit_["pos"]), "hops": None,
                "open": False, "locked": False, "active": True}

    def _is_reached(self, gate: dict, pos) -> bool:
        """A cylinder, not a sphere: 8 m horizontal (the DoorController trigger plus the door's height over the
        floor), 6 m vertical so standing on the roof above a door is not "passing" it. An open door doubles
        both; `open` alone never counts, because enemies open doors too."""
        hops, gate_pos = gate.get("hops"), gate.get("pos")
        if hops is None or not gate.get("active") or not gate_pos:
            return False
        margin = 2.0 if gate.get("open") else 1.0
        return (math.hypot(pos[0] - gate_pos[0], pos[2] - gate_pos[2]) <= margin * self.reach_h
                and abs(pos[1] - gate_pos[1]) <= margin * self.reach_v)

    def _note_reached(self, campaign: dict | None, pos) -> None:
        if pos is None:
            return
        for gate in self._gates(campaign):
            if self._is_reached(gate, pos):
                self.reached.add(str(gate.get("key")))
                hops = int(gate["hops"])
                self.hops_reached.add(hops)
                if self.best_hops is None or hops < self.best_hops:
                    self.best_hops = hops

    @staticmethod
    def _nearest(gates: list[dict], pos) -> dict | None:
        return min(gates, key=lambda g: _dist3(pos, g["pos"])) if gates else None

    def _choose_target(self, campaign: dict | None, pos) -> dict | None:
        if pos is None:
            return self.target  # no player this frame: keep pointing where we were
        active = [g for g in self._gates(campaign) if g.get("hops") is not None and g.get("active")]
        if not active:
            return self.target  # keep the previous target rather than flapping while the graph is rebuilt
        if self.best_hops is None:
            return self._nearest(active, pos)
        if self.best_hops == 0:
            return self._exit(campaign) or self._nearest([g for g in active if int(g["hops"]) == 0], pos) or self.target
        lower = {int(g["hops"]) for g in active if int(g["hops"]) < self.best_hops}
        if not lower:
            return self._exit(campaign) or self.target
        return self._nearest([g for g in active if int(g["hops"]) == max(lower)], pos)


class PathProgress:
    """Pays metres of new best NavMesh path length to the exit within an episode.

    Only complete paths count: a partial path stops wherever the NavMesh does, so its length says nothing about
    the distance left. The first complete path of an episode is the baseline and pays nothing. A new best must
    beat the old one by more than MIN_GAIN, and smaller improvements leave the best where it was, so they add up
    instead of paying jitter from the snapped path ends.
    """

    MIN_GAIN = 1.0

    def __init__(self):
        self.best = math.inf

    def reset(self) -> None:
        self.best = math.inf

    def update(self, path: dict | None) -> float:
        if not path or path.get("status") != "complete" or path.get("length") is None:
            return 0.0
        length = float(path["length"])
        if math.isinf(self.best):
            self.best = length
            return 0.0
        if length < self.best - self.MIN_GAIN:
            gain = self.best - length
            self.best = length
            return gain
        return 0.0


def choose_fresh_start(
    rng: random.Random,
    *,
    in_level: bool,
    level_over: bool,
    has_checkpoint: bool,
    stuck_streak: int,
    stuck_limit: int,
    fresh_prob: float,
) -> bool:
    """Whether the next campaign episode reloads the level (True) or respawns at the current checkpoint.

    A reload is forced when the game is not in the level with a player, the level is over, there is no current
    checkpoint to respawn at, or `stuck_limit` episodes in a row ended stuck at the same checkpoint (a respawn can
    leave a door locked behind the player). Otherwise it reloads with probability `fresh_prob`, so fresh-start
    completions keep being measured while most episodes land on each game's frontier. A forced reload does not
    draw from `rng`.
    """
    if not in_level or level_over or not has_checkpoint or stuck_streak >= stuck_limit:
        return True
    return rng.random() < fresh_prob


def save_best_run(path, run: dict, lock_timeout: float = 2.0, lock_poll: float = 0.02) -> bool:
    """Stores `run` as the level's best run if it is faster than the stored one. Returns whether it wrote.

    Guaranteed: a missing or unreadable file, or one without a time, counts as no stored run; only a run
    strictly faster than the stored one is written; the file is replaced atomically (retrying briefly while
    another training game has it open), so a crash mid-write never leaves a broken best run behind; and the
    read-compare-write is serialised across processes by an exclusive lock file, so two of the five training
    games finishing at nearly the same time cannot both read the same stale "stored" value and have the slower
    one win the write race (the atomic replace alone only protects against a torn file, not this race). If the
    lock is still held after `lock_timeout` seconds (a stale lock left by a killed process), this gives up and
    returns False rather than blocking the run forever.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # the lock file needs the directory to exist too
    lock_path = path.with_name(f"{path.name}.lock")
    if not _acquire_lock(lock_path, lock_timeout, lock_poll):
        return False
    try:
        try:
            stored = float(json.loads(path.read_text(encoding="utf-8"))["seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            stored = math.inf
        # `not (a < b)` (rather than `a >= b`) also refuses a NaN `run["seconds"]`, since every comparison with
        # NaN is False either way -- do not "simplify" this into `>=`, which would let a NaN time through.
        if not float(run["seconds"]) < stored:
            return False
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
        _replace_file(tmp, path)
        return True
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass
