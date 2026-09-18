"""Campaign helpers: level names, rank maths, and the measures behind the campaign rewards.

Plain Python plus numpy (for the archive file), so all of it is tested without the game in
tests/test_campaign.py. The inputs come from the mod's `campaign` observation block (docs/protocol.md).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import time
import zipfile
from functools import lru_cache
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

# The 33 of those whose scene bundle ships in this game build. `Level 9-1` and `Level 9-2` have no bundle, so
# the game cannot load them at all: a curriculum that lists one would stall a worker on a scene that never
# arrives. `EnvConfig.levels` validates against this set; `CAMPAIGN_LEVELS` still carries all 35, because
# times.md orders its rows by it and eval's --level check has always accepted every name.
CAMPAIGN_LEVELS_SHIPPED: frozenset[str] = frozenset(CAMPAIGN_LEVELS) - {"Level 9-1", "Level 9-2"}

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
# The multi-level curriculum
# ---------------------------------------------------------------------------
#
# One trainer process writes runs/<run>/curriculum.json; the SubprocVecEnv workers read it at every fresh level
# load and nowhere else. The two functions below are the whole policy, kept pure so they are tested offline.

CURRICULUM_VERSION = 1


def _level_stat(stats: dict | None, level: str, first: str) -> dict:
    """One level's record, or the default one for a level that has never finished an episode.

    `ProgressCallback.per_level` only gains an entry once a level produces an episode, and a hand-written or
    truncated curriculum.json can be missing any level, so every read goes through this: it is what keeps
    `choose_level` and `unlock_next` from raising on a sparse table. `first` (order[0]) is unlocked by default,
    so the unlock chain always has a starting point.
    """
    record = (stats or {}).get(level)
    if not isinstance(record, dict):
        return {"unlocked": level == first, "fresh_window": 0, "fresh_completion_rate": None,
                "best_time": None, "episodes": 0, "fresh_episodes": 0}
    return record


def level_weights(order, stats: dict | None, *, floor: float = 0.1) -> list[tuple[str, float]]:
    """(level, weight) for every unlocked level, in `order`. Weight is max(floor, 1 - fresh completion rate).

    A level with no data has rate None and so weight 1.0 (maximum attention); a mastered level falls to `floor`,
    which is what keeps a completed level in the mix so the policy does not forget it. Shared by `choose_level`
    (which samples from these) and the dashboard (which shows the normalised share).
    """
    order = list(order)
    if not order:
        return []
    first = order[0]
    out = []
    for level in order:
        record = _level_stat(stats, level, first)
        if level != first and not record.get("unlocked"):
            continue
        out.append((level, max(floor, 1.0 - (record.get("fresh_completion_rate") or 0.0))))
    return out


def choose_level(rng: random.Random, order, stats: dict | None, *, floor: float = 0.1) -> str:
    """The level the next fresh load uses, sampled from the unlocked set by `level_weights`.

    The weight controls fresh *draws*, not episodes: a worker only re-draws at a fresh start, and
    `choose_fresh_start` keeps it on the level it reached a checkpoint in for about 1/fresh_start_prob
    episodes. The number to read is the per-level `episodes` count, not the weight.
    """
    weighted = level_weights(order, stats, floor=floor)
    if not weighted:
        raise ValueError("choose_level needs a non-empty level order")
    levels = [level for level, _ in weighted]
    return rng.choices(levels, weights=[w for _, w in weighted], k=1)[0]


def unlock_next(order, stats: dict | None, *, unlock_rate: float = 0.5, unlock_window: int = 20,
                unlock_after_fresh_episodes: int = 0) -> str | None:
    """The first still-locked level whose predecessor has earned it, or None. Called once per finished episode.

    Unlocking is chained and stops at the first locked level, so 0-3 cannot unlock before 0-2 has: the order in
    the config is a ladder, not a menu. An empty `stats` returns None rather than raising, because `order[0]` is
    unlocked by the default record and every later level then fails the predecessor test.

    **This function never locks anything.** It only names a level for the caller to latch open, so a level whose
    rate later collapses stays unlocked and the learning in progress on it is not thrown away. That also makes
    inserting a level into `order` safe: with the new level locked the walk stops there and never revisits the
    already-unlocked levels behind it, which keep their own `unlocked` flag and their sampling weight.

    Two ways a predecessor earns its successor:
      - **the rate bar** (the fast path): `unlock_rate` over at least `unlock_window` of its own fresh episodes;
      - **the safety valve** (`unlock_after_fresh_episodes`, 0 = off): that many CUMULATIVE fresh episodes,
        whatever the rate. A campaign is a ladder, so without it one level the policy cannot crack blocks every
        level behind it forever -- and the run has no way to tell "needs 2M more steps" from "needs a mechanic
        nobody has built yet". Cumulative rather than windowed because `fresh_window` saturates at
        `FRESH_WINDOW` (50) and so cannot express "600 tries". A table written before this field existed reads
        0 and the valve simply never fires on it.
    """
    order = list(order)
    for i in range(1, len(order)):
        if _level_stat(stats, order[i], order[0]).get("unlocked"):
            continue
        previous = _level_stat(stats, order[i - 1], order[0])
        if not previous.get("unlocked"):
            return None
        earned = (int(previous.get("fresh_window") or 0) >= unlock_window
                  and (previous.get("fresh_completion_rate") or 0.0) >= unlock_rate)
        exhausted = (unlock_after_fresh_episodes > 0
                     and int(previous.get("fresh_episodes") or 0) >= unlock_after_fresh_episodes)
        return order[i] if earned or exhausted else None
    return None


def read_curriculum(path, *, order=None, run_name: str | None = None) -> dict | None:
    """The per-level table from `path`, or None when the file cannot be trusted.

    None means "use the first level only". That covers a missing file (the trainer has not written one yet), a
    torn or truncated read, a table that is not a dict, an `order` that disagrees with this worker's config, and
    a `run_name` that disagrees with the run directory the file sits in (a leftover from a renamed run).
    The write itself is atomic, so a reader only ever sees a whole file; the checks here are about *which* file.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, dict):
        return None
    if order is not None and [str(v) for v in (data.get("order") or ())] != list(order):
        return None
    if run_name is not None and data.get("run_name") != run_name:
        return None
    return {name: record for name, record in levels.items() if isinstance(record, dict)}


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


def altar_placement_key(altar: dict) -> str:
    """The key `item_placed` is paid on: the altar's item type plus the sorted keys of the doors it opens.

    Several levels ship coincident duplicate `ItemPlaceZone`s ~0.2 m apart wired to the same door (1-1 has
    three such pairs). Keying on the puzzle -- the type and the doors it unlocks -- instead of on the zone means
    punching the skull back out of one and placing it in its twin cannot earn a second payment. An altar that
    drives no door falls back to its own key, so the two never collide.
    """
    doors = sorted(str(d.get("key")) for d in (altar.get("doors") or ()) if isinstance(d, dict) and d.get("key"))
    item = str(altar.get("item"))
    return f"{item}|{'|'.join(doors)}" if doors else f"{item}@{altar.get('key')}"


class MilestoneTracker:
    """Pays each checkpoint, arena clear, door unlock, item pickup and item placement once per level load.

    The mod reports arenas and doors as rounded-position keys. A checkpoint respawn re-instantiates rooms at the
    same position, so an arena cleared again after a death gives the same key and cannot pay twice. Keys that
    show up during a reset or respawn (Restart() unlocks the checkpoint's doors) are absorbed with `mark_paid`
    instead of paid. A checkpoint counts once it is activated or current.

    The two item milestones are keyed so that neither the game's duplicate objects nor a respawn can pay twice:

      - `item_pickup` is keyed on the item TYPE, and only for types some altar in this level accepts. A respawn
        re-instantiates skulls under fresh mod-side keys, so an instance key would re-pay after every death;
        and without the accepted-type filter, 8-2's 22 `CustomKey1` props and 6-1's 11 would be a side quest
        with no route value.
      - `item_placed` is keyed on `altar_placement_key`, and a `filled` false -> true transition only pays when
        the item now sitting in that altar is the one the agent was carrying on the previous step. Thirty-one
        source pedestals campaign-wide read `filled: true` the moment their room switches on, with no agent
        action (ItemPlaceZone.CheckItem runs on room activation), and `mark_paid` cannot catch those: it only
        runs at a reset or a respawn, never when a room activates mid-episode. Matching the *instance* --
        `items[].placed_in` against the altar's own key, and that item's key against the previous step's held
        set -- is what separates a real placement from a pedestal streaming in. Type-matching alone is not
        enough: "something of this type was held a step ago" is true for the whole duration of any legitimate
        carry, so on 1-4 (four blue sources, rooms switching on one at a time) every source pedestal the agent
        walks past while carrying would pay for nothing.
    """

    def __init__(self):
        self._checkpoints: set[str] = set()
        self._arenas: set[str] = set()
        self._doors: set[str] = set()
        self._item_pickups: set[str] = set()
        self._item_placements: set[str] = set()
        self._held_keys: set[str] = set()  # item keys held in the block seen on the previous update

    @staticmethod
    def _keys(campaign: dict | None) -> tuple[set[str], set[str], set[str], set[str], dict[str, set[str]]]:
        """(checkpoints, arenas, doors, pickup types, {placement key: the filled altars' own keys}) here.

        The placements come back as a dict rather than a set because paying one needs to know which zone
        actually holds an item, which is what `items[].placed_in` is matched against. Coincident duplicate zones
        share one placement key, so the value is a set.
        """
        if not campaign:
            return set(), set(), set(), set(), {}
        checkpoints = {str(cp["id"]) for cp in campaign.get("checkpoints") or () if cp.get("activated") or cp.get("current")}
        altars = [a for a in (campaign.get("altars") or ()) if isinstance(a, dict) and a.get("item")]
        items = [i for i in (campaign.get("items") or ()) if isinstance(i, dict) and i.get("item")]
        accepted = {str(a["item"]) for a in altars}
        pickups = {str(i["item"]) for i in items if i.get("held") and str(i["item"]) in accepted}
        placements: dict[str, set[str]] = {}
        for a in altars:
            if a.get("filled"):
                placements.setdefault(altar_placement_key(a), set()).add(str(a.get("key")))
        return checkpoints, set(campaign.get("cleared_arenas") or ()), set(campaign.get("unlocked_doors") or ()), pickups, placements

    @staticmethod
    def _held_now(campaign: dict | None) -> set[str]:
        """The keys of the items reading `held: true` in this block (rule A4: unique within a step)."""
        return {str(i["key"]) for i in ((campaign or {}).get("items") or ())
                if isinstance(i, dict) and i.get("item") and i.get("held") and i.get("key") is not None}

    def _just_placed(self, campaign: dict | None) -> set[str]:
        """The altar keys that now hold an item the agent was carrying on the previous step.

        `Punch.PlaceHeldObject` is synchronous -- it reparents the item to the zone, clears `pickedUp` and calls
        `CheckItem()` in one call -- so the step that reports the altar `filled` is the same step that reports
        the item `placed_in` it, and the step before it reported the item `held`. A one-step lookback is
        therefore exact, and (unlike an "ever held this load" set) it cannot be fooled by rule A4 handing a
        re-instantiated skull the key a destroyed one used to have.
        """
        return {str(i["placed_in"]) for i in ((campaign or {}).get("items") or ())
                if isinstance(i, dict) and i.get("placed_in") and str(i.get("key")) in self._held_keys}

    def new_level_load(self, campaign: dict | None) -> None:
        """Starts a level load: forgets what was paid, then absorbs whatever the fresh level already reports."""
        self._checkpoints.clear()
        self._arenas.clear()
        self._doors.clear()
        self._item_pickups.clear()
        self._item_placements.clear()
        self._held_keys.clear()
        self.mark_paid(campaign)

    def mark_paid(self, campaign: dict | None) -> None:
        """Absorbs the block's current keys without paying for them."""
        self.update(campaign)

    def update(self, campaign: dict | None) -> tuple[int, int, int, int, int]:
        """New checkpoints, arena clears, door unlocks, item pickups and item placements for this step."""
        checkpoints, arenas, doors, pickups, placements = self._keys(campaign)
        placed_in = self._just_placed(campaign)
        new_placements = sum(1 for key, altars in placements.items()
                             if key not in self._item_placements and altars & placed_in)
        new = (len(checkpoints - self._checkpoints), len(arenas - self._arenas), len(doors - self._doors),
               len(pickups - self._item_pickups), new_placements)
        self._checkpoints |= checkpoints
        self._arenas |= arenas
        self._doors |= doors
        self._item_pickups |= pickups
        self._item_placements |= set(placements)  # a transition that paid nothing is still absorbed
        self._held_keys = self._held_now(campaign)
        return new

    @property
    def checkpoints_reached(self) -> int:
        """Distinct checkpoints activated in this level load (paid or absorbed)."""
        return len(self._checkpoints)


GATE_EXIT_KEY = "exit"  # the sentinel target key for campaign.exit; a real Door key is always "x,y,z"
SUBGOAL_ITEM = "item"  # the sub-goal kinds, and the prefixes of their best_dist keys ("item:SkullRed")
SUBGOAL_ALTAR = "altar"


def _live(entry: dict, *, need_active_self: bool = True) -> bool:
    """Whether an `items[]` entry is a real carryable rather than a decoration or a dead branch.

    Measured on the scene files: a carryable source has `active_self` true and at most ONE inactive ancestor
    (its own room switch); two means it sits under a permanently disabled node. 1-1 ships three `ItemIdentifier`s
    with `active_self` true, one of which is such a phantom.

    Altars are NOT tested with this any more -- see `dead_twin`, which replaced the absolute threshold after it
    was measured shifting under the mod's feet. An older mod sends neither field, so both default to "usable"
    and nothing here changes its behaviour.
    """
    if need_active_self and not entry.get("active_self", True):
        return False
    return int(entry.get("inactive_ancestors", 0) or 0) <= 1


def _altar_identity(altar: dict) -> tuple:
    """What makes two `altars[]` entries the same puzzle slot: item type, door set and rounded position.

    The mod's own key is the rounded world position, with `#2`, `#3` ... appended when several zones round to
    it, so stripping the suffix recovers the shared position key without re-rounding anything here.
    """
    doors = frozenset(d.get("key") for d in (altar.get("doors") or ()) if isinstance(d, dict))
    return (altar.get("item"), doors, str(altar.get("key", "")).split("#")[0])


def dead_twin(altar: dict, altars) -> bool:
    """Whether `altar` is the dead half of a duplicated `ItemPlaceZone` pair, judged RELATIVE to its twins.

    Every duplicate zone pair in the campaign is one live zone plus one dead twin -- 20 of the 104 zones. A dead
    zone can never activate, so its `CheckItem` never runs and it reads `filled: false` forever: counting one
    would leave a gate's lock set after the puzzle was solved and send the agent to punch the skull back out of
    the altar it had just filled, closing the gate again.

    The test used to be the absolute `inactive_ancestors > 1`, and that is the M14 bug: the number is not a
    property of the zone, it is the zone's chain plus however much of the room above it happens to be switched
    off, so it SHIFTS when the room lights. Measured in game on 1-1: on a fresh load the live altar `81,-4,251`
    read 1 and its twin read 2, and after the checkpoint respawn switched that room on they read 0 and 1 -- so
    the twin stopped being filtered exactly when the player arrived, and the gate kept `needs_item` forever
    however many times the puzzle was solved.

    The relative rule is immune to that shift, because a shared room contributes the same count to both halves
    of a pair: a zone is dead when another zone with the same item type, the same door set and the same rounded
    position reports STRICTLY FEWER inactive ancestors. The minimum of each group always survives, so a group
    can never be filtered away entirely and a real lock can never be lost. Re-validated offline against all 21
    altar levels: it drops the same 20 of 104 zones the old rule did, loses no lock on any level, and takes the
    doors left stuck after a solved puzzle from 11 to 0 once a room lights (see CLAUDE.md).

    Position is part of the identity because twins are co-located (~0.2 m apart, which is why the mod suffixes
    the key at all). Without it the rule also drops five zones that are not twins at all but merely two separate
    altars of one item type that drive no door -- 4-2 ×2, 4-3, 5-3 and 7-1.

    An older mod sends no `inactive_ancestors`, so every entry reads 0, no zone is strictly fewer than another,
    and nothing is filtered -- the unchanged behaviour.
    """
    ident = _altar_identity(altar)
    mine = int(altar.get("inactive_ancestors", 0) or 0)
    return any(other is not altar and _altar_identity(other) == ident
               and int(other.get("inactive_ancestors", 0) or 0) < mine
               for other in altars if isinstance(other, dict))


ALTAR_AIM_DROP_M = 1.0
"""Fallback metres BELOW `altars[].pos` to aim a punch, when the mod sends no `aim_pos`.

`Punch.AltHit` only places when the ray hits the very GameObject that carries the `ItemPlaceZone`, and every
one of the campaign's 104 zones is the same prefab: a trigger BoxCollider of local size (2.2, 3.5, 2.2) whose
local centre is (0, -1.25, 0), on a transform scaled (0.9, 0.8, 0.8). So the collider's world centre sits
1.25 * 0.8 = 1.0 m below `zone.transform.position` (103 of 104 zones; the one exception is 0.625 m), and the
box spans `pos.y - 2.4 .. pos.y + 0.4` -- the reported position is only 0.4 m under the lid, with no margin,
while the centre has 1.4 m of it. Measured in game on 1-1: aiming at the reported position does not place and
aiming ~1.25 m below it does.
"""


def altar_aim_point(altar: dict) -> list[float]:
    """Where to point a punch to fill `altar`: the zone's own collider centre.

    Prefers the mod's `aim_pos` (computed from the real collider, so it is right even for the odd zone whose
    scale differs) and falls back to ALTAR_AIM_DROP_M below `pos` against a mod that does not send it.
    """
    aim = altar.get("aim_pos")
    if aim and len(aim) == 3:
        return [float(v) for v in aim]
    x, y, z = altar["pos"]
    return [float(x), float(y) - ALTAR_AIM_DROP_M, float(z)]


def wanting_altars(campaign: dict | None, items) -> list[dict]:
    """The live, unfilled `altars[]` entries that accept one of `items` (a type name or a set of them).

    This is the filter a carry destination is chosen with, and it applies the same dead-twin rule the mod's
    `CampaignObserver.NeedsItem` applies, so the two sides agree about which zone a gate is waiting on.
    `GateProgress._subgoal` narrows it further to the altars wired to the gate it is trying to open. Empty
    means the held item has no destination at all (Level 0-4's `CustomKey1`, delivered by walking it into an
    `ItemTrigger`).

    An `altars` list the mod never sent reads as empty, so a 0.6.x mod, and every level with no `ItemPlaceZone`,
    gets the empty list and with it the unprotected, unchanged behaviour.
    """
    wanted = {items} if isinstance(items, str) else set(items)
    altars = [a for a in ((campaign or {}).get("altars") or ()) if isinstance(a, dict)]
    return [a for a in altars
            if a.get("pos") and a.get("item") in wanted
            and not a.get("filled") and not dead_twin(a, altars)]


# ---------------------------------------------------------------------------
# Layer 2: the offline room trunk (docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md)
# ---------------------------------------------------------------------------
#
# 15 of the 33 shipped levels have no usable gate ladder at all (`gates_ordered` false, or too few gates
# carrying `hops`), so `GateProgress._gates` returns [] and those levels have never had a route signal. For 12
# of them `build_routes.py` ships an offline ROOM TRUNK: the numbered rooms every playthrough must pass, in
# order, as a ladder shaped exactly like a gate array. Python reads it -- the mod is not involved, because a
# room rung needs no live state (spec §3) -- and it is consulted ONLY on the three `return []` arms of
# `_gates()`, so a level with a working gate ladder never touches any of this.
#
# Precedence, decided per level load, top to bottom: gates ladder, then room trunk, then the exit vector.

ROUTE_VERSION = 2  # schema version this loader understands; anything else falls through to the exit vector
ROUTE_MIN_RUNGS = 3  # guard R1, re-checked here so a hand-edited file cannot ship a 1-rung "route"
ROUTE_EXIT_TOL_M = 5.0  # guard I5: metres the live FinalPit may differ from the one the trunk was built against
ROUTE_SEED_M = 150.0  # how near a rung the player may start and still have it absorbed rather than paid (§6)
# Metres of ground that must lie under an AIRBORNE player before a TRUNK rung counts as reached. See
# `_on_ground`; a gate is never subject to it. Measured on Level 0-3, 2026-09-18, over three fresh
# episodes (25,204 decisions) and three scripted runs:
#     legitimate airborne credits, per rung  max 5.8  5.9  6.0  6.0  6.0  7.9   (six rungs)
#     against the wall face, first credit of each episode          9.5 10.5 10.5
#     scripted runs that never crossed the wall, first credit      8.8  9.1
#     standing over the main room's void ("10 - Main Room - Floor 2")     30.0 = ground_ray_length
# 8.0 is the only round number above every legitimate measurement and below every illegitimate one.
ROUTE_GROUND_M = 8.0
ROUTE_DIR = Path(__file__).resolve().parent / "routes"  # the packaged files; `route_dir` "" means this one

# `route_source`: which of the three layers is driving the target. An INTEGER, because it travels through
# `CAMPAIGN_INFO_KEYS` -> `Monitor(info_keywords=...)` -> `ProgressCallback._num`, which turns anything
# `float()` rejects into None (spec §7.4). The string goes to `episodes.jsonl` as `route_source_name`.
ROUTE_SOURCE_NONE = 0  # layer 3: no ladder at all, the exit vector and exploration carry the level
ROUTE_SOURCE_GATES = 1  # layer 1: the mod's live `campaign.gates`
ROUTE_SOURCE_ROOMS = 2  # layer 2: the shipped room trunk
ROUTE_SOURCE_NAMES = {ROUTE_SOURCE_NONE: "none", ROUTE_SOURCE_GATES: "gates", ROUTE_SOURCE_ROOMS: "rooms"}


# Patience is level-conditional (the 2026-09-17 ladder-patience spec, revised after the live 0-1 regression).
PATIENCE_MODES = ("collapsed", "always", "off")
COLLAPSE_RATIO = 0.5  # a ladder is collapsed when the gate nearest the spawn sits at or below HALF of max hops


def detect_collapsed_ladder(rungs, spawn) -> bool | None:
    """Is this level's ladder COLLAPSED at the spawn? None when the frame cannot answer it yet.

    The whole of the patience machinery hangs off this one bit, so it is a pure function of the two things a
    fresh load reports: the ladder (`campaign.gates`, or the room trunk on a level that has no usable one) and
    the player's first position. No history, no timers, nothing the policy can influence -- so a level's verdict
    is a property of the LEVEL, identical for every worker and every load of it.

    **The rule, and the measurement behind it.** `hops` counts the rooms still to traverse, so on a healthy
    ladder the door nearest the spawn is the one at the TOP of it: `near_hops == max_hops`. Bug A (spec §1) is
    exactly the opposite -- a multi-room door makes the BFS treat a door near the exit as adjacent to the start
    room, the nearest gate already carries a low `hops`, `best_hops` locks there and every rung above it is
    unreachable to `_choose_target` forever. Measured over all 18 shipped levels whose gate ladder
    `_gates()` accepts (`scratchpad/collapse_detect_scan.py`, from the offline level survey, with the two live
    spawns read off the running game for 0-1 and 0-3 and the level's first authored room elsewhere):

        COLLAPSED  0-3 2/6   1-1 1/5   1-2 2/4   2-3 0/2   4-3 1/2   8-1 0/4        (ratios 0.00-0.50)
        healthy    0-1 9/9   0-2 7/7   0-4 5/5   2-1 3/3   2-2 2/2   3-1 7/7
                   3-2 3/3   4-1 8/8   5-1 2/2   6-1 2/2   5-3 12/13  8-2 5/8       (ratios 0.62-1.00)

    `2 * near <= max` separates them with a clear gap -- the worst collapsed level is exactly 0.50 and the
    nearest healthy one 0.625 (8-2, whose spawn is the least certain of the eighteen: at each of the three
    plausible spawn points it scores 5, 5 or 7 of 8, so it stays healthy whichever is right). Nothing else
    tried separated them at all: the absolute deficit `max - near` puts 8-2 (3) above three collapsed levels
    (4-3 1, 1-2 2, 2-3 2), so it has no threshold that works.

    The three levels this matters for are 0-1 (healthy; patience regressed it live), 0-3 and 4-3 (collapsed;
    patience is what made 0-3 move at all).

    Read defensively, like every other field in this module: `hops` may be null (an `altar_only` door), a
    frame may carry no ladder at all, and a mid-load frame may have no player. Any of those is None, "ask
    again on the next frame", never False.
    """
    if spawn is None:
        return None
    graded = [g for g in (rungs or ())
              if isinstance(g, dict) and g.get("hops") is not None and g.get("pos")]
    if not graded:
        return None
    max_hops = max(int(g["hops"]) for g in graded)
    if max_hops < 1:
        return False  # one tier of doors: there is no ladder above the spawn to lose, so nothing to rescue
    nearest = min(graded, key=lambda g: _dist3(spawn, g["pos"]))
    return int(nearest["hops"]) <= COLLAPSE_RATIO * max_hops


def _point(value) -> list[float] | None:
    """`value` as an [x, y, z] of finite floats, or None when it is not one.

    Every position that reaches the reach test or `_dist3` comes through here: a file with a 2-element `pos`,
    a null coordinate or a NaN must read as "no rung" rather than raise out of `env.step`, which catches only
    `BridgeRecovered` and would otherwise take all twelve games down with one bad line of JSON.
    """
    try:
        if len(value) < 3:
            return None
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def route_path(scene: str, route_dir: str = "") -> Path:
    """Where this scene's route file lives: `<route_dir or the packaged routes/>/route_<safe scene>.json`."""
    return (Path(route_dir) if route_dir else ROUTE_DIR) / f"route_{safe_name(scene)}.json"


@lru_cache(maxsize=64)
def _read_route(scene: str, route_dir: str) -> dict | None:
    """The parsed, validated route document for `scene`, or None -- which means today's behaviour.

    Cached because twelve workers in one process would otherwise re-read and re-parse the same file at every
    level load. The cache holds the PARSED document and `load_route` hands out a deep copy, so nothing a
    consumer does to its rungs can reach another env's. Call `_read_route.cache_clear()` after rewriting a
    file in the same process (the tests do; a training run never rewrites one).

    Every rejection is silent and falls through to layer 3, because "there is no file" is the normal case on
    21 of the 33 levels:

      - the file is missing, unreadable or not JSON;
      - `version` is not ROUTE_VERSION (a future emitter must not be read by an older loader);
      - `level` is not this scene (the one check that catches a file renamed onto the wrong level);
      - `exit.pos` is missing, so guard I5 could never be evaluated and a level whose pit moved could never be
        detected -- exactly the case the guard exists for, so a file without it is not usable;
      - fewer than ROUTE_MIN_RUNGS usable rungs (R1).

    `open`, `locked` and `active` are NORMALISED rather than trusted. They are schema constants, not data: a
    room rung is always standable ground with no door state, `_is_reached` DOUBLES its cylinder when `open` is
    true (16 m x 12 m, which R4 never measured and which would collide adjacent rungs), and `active` false or
    `needs_item` set would make a rung permanently unreachable. So the loader guarantees the invariants the
    reach test assumes; `tests/test_route_files.py` is what fails when a shipped file disagrees.

    **The validation is TOTAL**, not best-effort, and `_parse_route` below carries the backstop that makes it
    so. A file that is syntactically valid JSON with the wrong TYPES used to raise straight out of here:
    `rungs: 5` is truthy and not iterable (TypeError), and `hops: 1e400` parses to inf, which `int()` rejects
    with OverflowError rather than the ValueError the clause caught. `load_route` runs in `UltrakillEnv.__init__`
    and in `_switch_level` -> `_campaign_reset` -> `reset()`, neither of which catches anything, and SB3's
    SubprocVecEnv worker does not either -- so on the 30-level curriculum one bad file would kill a worker the
    first time any env sampled that level, hours into a run. And hand-editing a route file is the DOCUMENTED
    recovery path for a bad rung (spec §12.4: set its `"rungs": []`), so a typo lands exactly here.
    """
    try:
        doc = json.loads(route_path(scene, route_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return _parse_route(scene, doc)
    except Exception as exc:  # noqa: BLE001 -- see the docstring: the blast radius is twelve training games
        # The backstop, and the only loud rejection: every case the checks below name is a normal file-level
        # refusal and stays silent, so anything reaching here is a shape `_parse_route` did not anticipate and
        # is worth one line. `lru_cache` caches the None, so it prints once per scene per process, not per load.
        print("GateProgress: the route file for %s is unreadable (%s: %s); falling back to the exit vector"
              % (scene, type(exc).__name__, exc), flush=True)
        return None


def _parse_route(scene: str, doc) -> dict | None:
    """`_read_route`'s validation, split out so its every failure is one `except` away from layer 3."""
    if not isinstance(doc, dict) or doc.get("version") != ROUTE_VERSION or doc.get("level") != scene:
        return None
    exit_ = doc.get("exit")
    exit_pos = _point(exit_.get("pos")) if isinstance(exit_, dict) else None
    if exit_pos is None:
        return None
    entries = doc.get("rungs")
    if not isinstance(entries, list):
        return None  # a scalar `rungs` is truthy and not iterable; `"rungs": []` is §12.4's recovery path
    rungs = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("hops") is None:
            continue
        pos = _point(entry.get("pos"))
        if pos is None:
            continue
        try:
            hops = int(entry["hops"])
        except (TypeError, ValueError, OverflowError):
            continue  # OverflowError: `1e400` parses to inf, which int() refuses
        rung = dict(entry)
        rung.update(key=str(entry.get("key") or ",".join(str(round(v)) for v in pos)), pos=pos, hops=hops,
                    open=False, locked=False, active=True)
        rung.pop("needs_item", None)  # stage S3 stamps this LIVE from campaign.altars; a static one wedges
        rungs.append(rung)
    if len(rungs) < ROUTE_MIN_RUNGS:
        return None
    return {"level": scene, "exit_pos": exit_pos, "rungs": rungs}


def load_route(scene: str, route_dir: str = "") -> dict | None:
    """One env's private copy of `scene`'s room trunk, or None when there is no usable file.

    A DEEP copy, and that matters: `_choose_target` hands a rung straight out as `self.target`, `_subgoal` can
    build a sub-goal from it, and the twelve SubprocVecEnv workers of a training run are separate processes but
    `eval.py`, the tests and any single-process tool build several envs side by side. An lru_cache'd document
    would be aliased by every one of them.
    """
    doc = _read_route(str(scene), str(route_dir or ""))
    return None if doc is None else copy.deepcopy(doc)


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

    One load boundary falls INSIDE an episode: a death with no checkpoint yet, where `StatsManager.Restart`
    reloads the whole level under a running episode. There the ladder restarts and the payments do not --
    `new_level_load(keep_paid=True)`, which `env._respawn` is the only caller of. Both halves are load scope
    honestly applied: the player really is back at the spawn, so the TARGET must be, but "once per level load"
    cannot be allowed to mean "once per death" while the episode that collects it is still running.

    `gates[].controller_active` is reported by the mod (a door whose `ActivateArena` has not run is neither
    locked nor approachable) but no rule here reads it yet; every field is read with `.get`, so a 0.5.x mod
    without it -- or without `gates` at all -- simply produces no target and pays nothing.

    **`hops` is a lower bound, not a route.** It is the shortest path in a room graph that multi-room doors
    over-connect, so on six of the shipped levels (0-3, 1-1, 1-2, 2-3, 4-3, 8-1) the gate nearest the spawn
    already sits near `hops` 0 and the monotone rule above locks onto a door that cannot be walked to -- on 0-3,
    one 66 m straight up through a ceiling, for 175 of 177 training episodes. `patience_steps` is the answer
    (docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md): a target that goes that many
    decisions without getting closer is PARKED for this level load, the target becomes the nearest unreached,
    unparked gate at any hop count, and reaching one of those pays a `gate` instalment for each NEW RUNG, so the
    forward legs of a non-monotone route earn something. `patience_steps` 0 disables all of it and restores this
    class exactly as it was before that spec, which is how the 0-1 inertness proof is written.

    **And it only runs where the ladder is collapsed** (`patience_mode`, `detect_collapsed_ladder`,
    `patience_active`). Shipped unconditionally, the mechanism did what it was built for on 0-3 -- fresh
    `gates_reached` 1.0 -> 4.5 -- and REGRESSED 0-1, whose ladder is monotone and correct, from a 0.55 fresh
    completion rate (n=43) to 0.30 (n=56), parking a target in 34% of its fresh episodes. On a correct ladder
    there is nothing better to switch to, so a park can only mislead; the verdict is taken once per level load
    from the gates block and the spawn, and "collapsed" is the shipped default.

    The four rules that keep the park from becoming a second wedge, each of them a reproduced failure rather
    than a precaution -- see `_tick_patience`, `_unpark`, `_pay_fallback` and `mark_paid`:

      - a park is only ever a SWITCH for a LADDER pick, but a FALLBACK pick parks unconditionally, because the
        fallback is by construction the nearest candidate and "is something nearer?" can never be true for it;
      - `park_best` keeps falling while a gate is parked and the un-park bar doubles per park, so bobbing about
        under an unreachable door cannot un-park it;
      - the clock runs on its own baseline, not on `gate_approach`, so a death respawn does not park the one
        door the route needs;
      - the arena suspension is bounded at `2 * patience_steps`, and a carry leg is never parked at all.

    **The route fallback** (`route`, the 2026-09-17 route-fallback spec). On the three arms where `_gates()`
    gives up -- no block, `gates_ordered` false, too few `hops` -- a level that ships an offline room trunk
    uses that instead, and everything above applies to it unchanged: the same ladder rule, the same
    `gate_approach`, the same patience, parking, un-parking and fallback payment, reading the same fields.
    Three rules keep the two layers apart:

      - **the gates success path is not touched.** `return gates` is byte for byte what it was, so a level with
        a working ladder -- all 18 of them, 0-1 included -- gets the identical list it gets today and
        `GateProgress(route=None)` cannot behave differently from the class before this spec;
      - **the rooms-only rules are guarded by object IDENTITY against the loaded document** (`_is_room_ladder`),
        never by a field on the entries, so nothing in the data can turn the seeding rule on for a gate;
      - **a mid-load flip between the two layers clears the ladder** (`_hops_source`), because the two carry
        different hop scales and crossing from one to the other would otherwise pay the difference.

    How the trunk and the patience machinery interact, since a rung reuses both payment rules:

      - the ladder rule pays per hop value DESCENDED and `_pay_fallback` only pays a rung at or above the
        current floor, which the ladder by construction can never pay for. The two are mutually exclusive, so
        no rung is ever paid twice and a level load's `gate` income stays bounded by the trunk's depth --
        tested straight through the env (`test_a_route_rung_goes_through_the_same_patience_and_parking_machinery`)
        and as arithmetic (`test_a_fallback_rung_behind_the_floor_pays_once_and_only_once`);
      - parking is nearly INERT on a healthy trunk, and that is structural rather than a bug: a trunk is a
        total order, so its rung-below is usually also the nearest unreached rung, and `_nearer_unreached`
        keeps a ladder pick whenever nothing is strictly nearer. It fires where it is needed -- a rung the
        agent cannot reach with another rung nearer -- and not otherwise. The detector for a rung that is
        simply wrong is therefore `route_source` 2 with `gates_reached` stuck and `targets_parked` 0, and the
        cure is that level's data (`"rungs": []` falls it back to the exit vector), never a knob here.
    """

    UNPARK_ESCALATION = 2.0  # each further park of one key doubles the improvement needed to undo it

    def __init__(self, reach_m: float = 8.0, reach_v_m: float = 6.0, min_gain_m: float = 0.5,
                 hops_min_frac: float = 0.5, target_kind_slots: bool = False,
                 patience_steps: int = 0, unpark_m: float = 2.0, fallback_hysteresis_m: float = 10.0,
                 route: dict | None = None, route_exit_tol_m: float = ROUTE_EXIT_TOL_M,
                 route_seed_m: float = ROUTE_SEED_M, patience_mode: str = "collapsed",
                 prefer_route_when_collapsed: bool = False,
                 route_ground_m: float = ROUTE_GROUND_M):
        self.reach_h = float(reach_m)
        self.reach_v = float(reach_v_m)
        self.min_gain = float(min_gain_m)
        self.hops_min_frac = float(hops_min_frac)
        self.target_kind_slots = bool(target_kind_slots)
        # Target patience (docs/superpowers/specs/2026-09-17-ladder-patience-and-exit-guard.md). 0 turns parking
        # and the fallback off completely, which is byte for byte the tracker before that spec.
        self.patience_steps = int(patience_steps)
        self.unpark_m = float(unpark_m)
        self.fallback_hysteresis = float(fallback_hysteresis_m)
        # ... and WHERE it is allowed to act: "collapsed" (the shipped default) only on a level whose ladder
        # `detect_collapsed_ladder` reports collapsed at the spawn, "always" unconditionally (what shipped on
        # 2026-09-17 and regressed 0-1), "off" never. See `patience_active`.
        self.patience_mode = str(patience_mode) if patience_mode in PATIENCE_MODES else "collapsed"
        self.prefer_route_when_collapsed = bool(prefer_route_when_collapsed)
        # The route fallback. `route` is `load_route(level)`, or None on the 21 levels with no file and for any
        # env built with `route_fallback: false` -- and None makes every branch below inert.
        self._route = route
        self.route_exit_tol = float(route_exit_tol_m)
        self.route_seed_m = float(route_seed_m)
        self.route_ground_m = float(route_ground_m)
        # This step's ground reading, as (grounded, drop) -- set for the duration of `update()` and
        # None everywhere else, which is what keeps every ABSORBING call permissive. See `_on_ground`.
        self._ground: tuple[bool | None, float | None] | None = None
        self._route_warned = False  # the stale-file message, printed at most once per level per worker
        self.route_reads = 0  # env lifetime: times the trunk was used as the ladder, so a test can prove it wasn't
        # Per level load
        self._route_ok: bool | None = None  # guard I5's verdict, decided once per load and re-decided per load
        self.ladder_collapsed: bool | None = None  # the detector's verdict, decided once per load (None = not yet)
        self._collapse_spawn: list[float] | None = None  # the first player position this load reported
        self._hops_source: str | None = None  # "gates" | "rooms": which ladder set `best_hops`
        self.route_source = ROUTE_SOURCE_NONE  # the layer driving the target, reported per episode as an int
        self._seed_pending = False  # §6's seeding rule fires once per level load, on the rooms path only
        self.best_hops: int | None = None  # lowest hops reached
        self.paid_hops: int | None = None  # lowest hops already paid or absorbed
        self.reached: set[str] = set()  # gate keys ever reached this level load
        self.hops_reached: set[int] = set()  # their hop values, which is what `gates_reached` counts
        self.parked: set[str] = set()  # gate keys the patience rule gave up on this level load
        self.park_best: dict[str, float] = {}  # ... and the closest the player has been since it was parked
        self.park_count: dict[str, int] = {}  # times each key has been parked, which escalates its un-park bar
        self.paid_fallback: set[str] = set()  # gate keys that have already paid a fallback `gate` instalment
        # Per env lifetime, so the env can report parks per episode by difference
        self.parks = 0
        # Per episode
        self.best_dist: dict[str, float] = {}  # gate key (or "exit") -> closest approach made this episode
        self.target: dict | None = None
        self._fallback = False  # whether `target` came from the fallback rather than the ladder rung
        self._fallback_key: str | None = None  # the sticky fallback choice, kept until it is reached or parked
        self._clock_key: str | None = None  # the target the patience clock is timing
        self._clock = 0  # decisions it has gone without a new best approach
        self._clock_suspended = 0  # ... plus decisions the arena rule refused to count, bounded in _tick_patience
        self._clock_best = math.inf  # closest approach to it in THIS attempt; never `best_dist`, see _tick_patience

    # -- lifecycle ---------------------------------------------------------------------------

    def set_route(self, route: dict | None) -> None:
        """Swaps in another level's room trunk (a curriculum run switching level). None is layer 1 or 3 only.

        `new_level_load` runs immediately afterwards in `env._campaign_reset` and clears the per-load verdicts,
        but they are cleared here too so that a caller who only switches the route cannot be left holding the
        previous level's staleness verdict.
        """
        self._route = route
        self._route_ok = None
        self._route_warned = False
        self._hops_source = None
        self.route_source = ROUTE_SOURCE_NONE

    @staticmethod
    def _floor(a: int | None, b: int | None) -> int | None:
        """The lower of two hop floors, either of which may be None ("nothing paid or absorbed yet")."""
        return b if a is None else a if b is None else min(a, b)

    def new_level_load(self, campaign: dict | None, player_pos=None, *, keep_paid: bool = False) -> None:
        """Starts a level load: forgets the ladder, then absorbs whatever the fresh level already reports.

        `keep_paid` is for the one caller that is NOT an episode boundary: `env._respawn` on the branch where a
        death with no checkpoint made `StatsManager.Restart` reload the whole level (spec §6). The LADDER has to
        start again there, because the player really is back at the spawn and `_choose_target` would otherwise
        aim at a rung far ahead of them -- but the PAYMENTS must not, because the episode is continuing.

        `gate` is "once per level load for every new lower `hops` value reached, so re-walking the same doors
        after a death pays nothing" (this class's own docstring), and `env._respawn` refuses the same re-payment
        for `MilestoneTracker` in the same breath. Clearing `paid_hops` / `paid_fallback` / `best_dist` inside a
        continuing episode would make "die with no checkpoint" re-arm the whole prefix of the ladder for ground
        already covered: measured on FakeLevel, three laps of a 3-rung trunk paid 30 / 30 / 30 where the ladder's
        own depth is 30 once -- a stream bounded only by `max_steps`, against a `death` of 5, and one that
        activating the first checkpoint would permanently end. So the three payment records survive the call and
        everything else starts again. What is kept is a FLOOR, never a raise: a rung the reload reveals under the
        player can still lower `paid_hops`, and genuinely new progress past the dead attempt's best still pays.
        """
        prev_paid = self.paid_hops if keep_paid else None
        prev_fallback = set(self.paid_fallback) if keep_paid else set()
        prev_dist = dict(self.best_dist) if keep_paid else {}
        # Per load, and before mark_paid: the staleness check is re-run on every load (a reload puts a banished
        # FinalPit back), the hop scale starts unknown, and the seeding rule gets its one chance.
        self._route_ok = None
        self._hops_source = None
        self.route_source = ROUTE_SOURCE_NONE
        # The collapse verdict is per LEVEL LOAD and is taken from the player's first position of that load, so
        # it is cleared here and nowhere else: `mark_paid` (every checkpoint respawn, and every reset that is
        # not a fresh load) deliberately keeps it, because a respawn starts the player half-way through a level
        # where the nearest gate says nothing about the level's shape.
        self.ladder_collapsed = None
        self._collapse_spawn = None
        self._seed_pending = self._route is not None
        self.best_hops = self.paid_hops = None
        self.reached.clear()
        self.hops_reached.clear()
        self.parked.clear()
        self.park_best.clear()
        self.park_count.clear()
        self.paid_fallback.clear()  # before mark_paid, which seeds it from `reached`
        self.best_dist.clear()
        self.target = None
        self.reset_episode()
        self.mark_paid(campaign, player_pos)
        # After mark_paid, which sets `paid_hops = best_hops` from whatever the player is standing in and seeds
        # `paid_fallback` from it. `_seed_start` takes the same floor, so a re-seed cannot raise it either.
        self.paid_hops = self._floor(self.paid_hops, prev_paid)
        self.paid_fallback |= prev_fallback
        self.best_dist.update(prev_dist)

    def mark_paid(self, campaign: dict | None, player_pos=None) -> None:
        """Absorbs the gates the player already stands at (a checkpoint respawn, or any reset) without paying."""
        self._note_reached(campaign, player_pos)
        self.paid_hops = self.best_hops
        # The fallback instalment is absorbed the same way the ladder's is: a respawn that lands the player on
        # the gate the fallback was pointing at reveals it, it does not earn it. A gate already in `reached` can
        # never become a fallback target again, so this only ever blocks the one step the respawn creates.
        self.paid_fallback |= {str(key) for key in self.reached}
        # best_dist is deliberately untouched: a respawn inside an episode must not re-earn the approach.
        # The patience CLOCK is restarted, which is the opposite call and the right one for it. A respawn puts
        # the player back at a checkpoint, so every metre between there and the target has to be re-walked -- and
        # by ruling R1 that re-walk pays no `gate_approach`, because `best_dist` still holds the pre-death
        # minimum. Timing the clock off the reward would therefore run it out during a flawless approach and park
        # the one door the route needs (measured: parked at decision 299 of 389 with 101 m still to go). The
        # clock measures the CURRENT attempt; the reward measures the episode. They are different questions.
        self._clock = self._clock_suspended = 0
        self._clock_best = math.inf

    def reset_episode(self) -> None:
        """A new episode: pick the target again from scratch, and let it re-earn its approach (ruling R1).

        `parked` / `park_best` / `paid_fallback` deliberately survive: a park is a fact about this level load's
        geometry (that door is not reachable from here), while the patience clock measures the current attempt,
        whose `best_dist` baseline has just been cleared. Carrying a stale clock across the boundary would park
        a gate the new episode never had a chance to approach.
        """
        self.target = None
        self.best_dist.clear()
        self._fallback = False
        self._fallback_key = None
        self._clock_key = None
        self._clock = self._clock_suspended = 0
        self._clock_best = math.inf

    def retarget(self, campaign: dict | None, player_pos=None) -> None:
        """Chooses the current target. Pays nothing; call it before packing any observation."""
        self._note_reached(campaign, player_pos)
        self.target = self._subgoal(campaign, self._choose_target(campaign, player_pos), player_pos)
        key = self._key_of(self.target)
        if key is not None and player_pos is not None and key not in self.best_dist:
            self.best_dist[key] = _dist3(player_pos, self.target["pos"])  # seeded once, never re-seeded

    # -- per step ----------------------------------------------------------------------------

    def update(self, campaign: dict | None, player_pos=None, *, fought: bool = False,
               ground: tuple[bool | None, float | None] | None = None) -> tuple[int, float]:
        """(gate instalments, metres of new best closeness to the target) for this step.

        `fought` is "a kill or a style event landed this step", which the env already computes for the stuck
        clock. It suspends the patience clock, because a door held shut by an `ActivateArena` wave cannot be
        approached until the wave is dead (§3 of the patience spec).

        `ground` is this step's `(player.grounded, metres of ground below the player)`, which `_on_ground`
        applies to ROOM-TRUNK rungs and to nothing else. It is held for the duration of this call and cleared
        afterwards, so the two PAYING uses of `_is_reached` -- `_note_reached` by way of `retarget`, and
        `_pay_fallback` -- see it while every ABSORBING call, all of which are outside `update()`, does not.
        `ground=None` is the default and leaves the tracker exactly as it was; it is what every caller that is
        not the campaign env passes.
        """
        self._ground = ground
        try:
            return self._update(campaign, player_pos, fought=fought)
        finally:
            self._ground = None

    def _update(self, campaign: dict | None, player_pos=None, *, fought: bool = False) -> tuple[int, float]:
        prev_target, prev_fallback = self.target, self._fallback
        prev_key = self._key_of(prev_target)
        prev_best = self.best_dist.get(prev_key, math.inf) if prev_key is not None else math.inf
        prev_floor = self.best_hops
        # Judged before `retarget`, which is what adds this step's gates to `reached`.
        paid = self._pay_fallback(campaign, prev_target, prev_fallback, prev_floor, player_pos)
        self._unpark(campaign, player_pos)
        self.retarget(campaign, player_pos)

        if self.best_hops is not None:
            if self.paid_hops is None:
                paid += 1
            elif self.best_hops < self.paid_hops:
                paid += self.paid_hops - self.best_hops
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

        self._tick_patience(campaign, player_pos, fought)
        return paid, approach

    @property
    def gates_reached(self) -> int:
        """Distinct hop values reached in this level load (0 when none)."""
        return len(self.hops_reached)

    @property
    def patience_active(self) -> bool:
        """Whether parking, the fallback target and the fallback payment may act on this level load.

        `patience_steps` 0 is the global off switch it has always been. `patience_mode` is the level-conditional
        one, and the reason it exists is measured rather than argued: shipped unconditionally on 2026-09-17, the
        mechanism took Level 0-1's fresh completion rate from 0.55 (n=43) to 0.30 (n=56) while it was taking
        0-3's gates from 1.0 to 4.5. 0-1's ladder is monotone and CORRECT, so parking its target can only ever
        move the agent off the right door -- and it did: 34% of its fresh episodes parked at least once, and a
        third of their stuck endings piled up at one spot two gates in. On a level whose ladder is right there
        is nothing for the fallback to find, so the honest setting is not to run it there at all.

        While the verdict is still None -- a mid-load frame with no ladder or no player -- "collapsed" reads as
        OFF. That is the conservative direction: a level whose block never arrives behaves exactly as it did
        before the patience spec.
        """
        if not self.patience_steps or self.patience_mode == "off":
            return False
        return True if self.patience_mode == "always" else bool(self.ladder_collapsed)

    def _prefers_route(self) -> bool:
        """Whether this load should walk the shipped room trunk INSTEAD of its own collapsed gate ladder.

        Lead ruling (2026-09-17): the flag is off by default, so 0-3 and 4-3 keep running on gates plus
        patience, which is what is currently making 0-3 progress. Turning it on is a per-run decision to be made
        on evidence, and it can never reach a level whose ladder is healthy: the verdict gates it.
        """
        return bool(self.prefer_route_when_collapsed and self._route is not None and self.ladder_collapsed)

    # -- patience, parking and the fallback ---------------------------------------------------

    @staticmethod
    def _park_key(target: dict | None) -> str | None:
        """The GATE a target belongs to: itself, or, for a fetch/carry leg, the door that leg opens.

        Parking a stalled sub-goal by its own key would achieve nothing -- `_choose_target` would pick the same
        gate and `_subgoal` would rebuild the same leg -- and sub-goal keys are not stable anyway, since
        `CheckPoint.ResetRoom` re-instantiates skulls. Parking the gate moves the whole carry machine on.
        """
        if not target:
            return None
        return str(target.get("gate_key") or target.get("key"))

    def _gate_by_key(self, campaign: dict | None, key: str | None) -> dict | None:
        return next((g for g in self._gates(campaign) if str(g.get("key")) == key), None) if key else None

    def _pay_fallback(self, campaign: dict | None, target: dict | None, was_fallback: bool,
                      floor: int | None, pos) -> int:
        """A4: reaching a gate the FALLBACK chose pays one `gate` instalment, once per NEW RUNG per level load.

        Without this a forward leg of a non-monotone route earns nothing, because its `hops` is at or above
        `best_hops` and the ladder rule only pays new lower rungs. `hops >= floor` keeps the two rules from ever
        paying for the same reach: a fallback gate that does lower `best_hops` is paid by the ladder instead.

        **The instalment is bounded by the ladder's own unit, not by the gate key.** A4 asks for "no incentive to
        tour doors", and per-key payment does not deliver it: `_pick` re-selects the nearest unreached, unparked
        gate every step, so door after door becomes the fallback target in turn and pays. Measured on eight side
        doors sharing one rung, a tour collected 6 instalments where the ladder's own depth was 2 -- 90 points
        against `level_complete` 100, re-earned on every fresh load, and on 8-1's 52 phase-1 gates the ceiling is
        780. `hops in hops_reached` closes it exactly: every forward leg of 0-3's real route (hops 2 -> 3 -> 4 ->
        5 -> 6) is a new rung and still pays, while a second door on a rung already reached pays nothing, so the
        total stays bounded by the ladder depth just as the monotone rule is. `hops_reached` is already
        level-load scoped and already absorbed by `mark_paid` through `_note_reached`, so the respawn semantics
        come for free.

        `hops` is read defensively, as it is everywhere else in this class: `_gate_by_key` searches the FULL
        array, which deliberately carries `altar_only` doors with `hops: null`, and an unguarded `int(...)` here
        would raise out of `update` -- past `env.step`, which catches only `BridgeRecovered`, past SubprocVecEnv's
        worker and into all twelve games.
        """
        gate_key = self._park_key(target)
        if not (was_fallback and pos is not None and gate_key and gate_key != GATE_EXIT_KEY):
            return 0
        if gate_key in self.paid_fallback or floor is None:
            return 0  # floor None: nothing is reached yet, so the ladder's own first instalment covers this
        gate = self._gate_by_key(campaign, gate_key)
        hops = gate.get("hops") if gate else None
        if hops is None or int(hops) < floor or not self._is_reached(gate, pos):
            return 0
        self.paid_fallback.add(gate_key)
        if int(hops) in self.hops_reached:
            return 0  # this rung has already been paid for; touring its other doors is worth nothing
        return 1

    def _unpark(self, campaign: dict | None, pos) -> None:
        """A3: a parked gate comes back when the player passively gets genuinely nearer than it has ever been.

        Never on a timer: the situation has to have changed -- the agent has climbed toward the high door some
        other way. What makes "genuinely" hold up is that the bar DOUBLES with each park of that key
        (`UNPARK_ESCALATION`), so a door that has already proved unreachable twice needs 4 m, then 8 m, then
        16 m. A flat 2 m is less than one ULTRAKILL jump, and on the recorded 0-3 probe that let the collapsed
        ladder re-form on a roughly 50% duty cycle: episode 0 parked and un-parked three times (40.87 -> 38.16
        -> 35.98 -> 29.16 -> 26.13 -> 23.25 m), ended with the door fully back and still spent 40.4% of its
        decisions pointed at it. With the escalation it un-parks once, on a genuine 24 m climb, and parks again
        when that stalls too. `park_count` is level-load scoped like the rest of the ladder.

        `park_best` is the closest the player has been to that gate at any park of it this level load: it is
        fixed while the gate stays parked, and a later park can only lower it (`_tick_patience` takes the `min`).
        Both halves matter, and the second is where the reviewer's proposal went wrong.

        It must not follow the player's running minimum every step, because such a baseline is beaten by at most
        one decision's travel, so no gradual approach can clear a bar wider than that: walking 1 m per decision
        straight at a door with a 2 m bar gives `d < (d + 1) - 2`, false at every distance. Measured, that kept
        the door parked while the agent climbed 34 m to stand on it, which breaks A3 outright.

        It must equally not be re-read from wherever the player happens to stand when the clock next runs out.
        On the probe the second park was taken at 77.5 m, having wandered away from the 40.9 m the first one
        recorded, and walking back into the pit then cleared the bar for free. Keeping the minimum makes each
        un-park cost strictly more than the last, and with the doubling bar the sequence converges: the probe's
        episode 0 goes 40.9 -> 38.4 -> 29.2 -> 17.6 m, each one a real climb, and the fourth park would need the
        agent to stand inside 2 m of the door.
        """
        if pos is None or not self.parked:
            return
        for gate in self._gates(campaign):
            key = str(gate.get("key"))
            if key not in self.parked or not gate.get("pos"):
                continue
            bar = self.unpark_m * self.UNPARK_ESCALATION ** max(0, self.park_count.get(key, 1) - 1)
            if _dist3(pos, gate["pos"]) < self.park_best.get(key, math.inf) - bar:
                self.parked.discard(key)  # park_best is KEPT: the next park of this key measures from it

    def _nearer_unreached(self, campaign: dict | None, pos, target: dict | None) -> dict | None:
        """An unreached, unparked, active gate strictly nearer than `target`, or None.

        Parking is only ever a SWITCH, never a loss: with no better candidate the ladder's own answer is still
        the best guess available and the target is kept. Measured on the 32,022 recorded Level 0-1 decisions,
        this test is what takes the parks per episode from 0-13 to 0-1 and leaves 6 of 7 episodes byte-identical.
        """
        if pos is None or not target or not target.get("pos"):
            return None
        here = _dist3(pos, target["pos"])
        skip = self._park_key(target)
        candidates = [g for g in self._gates(campaign)
                      if g.get("hops") is not None and g.get("active") and g.get("pos")
                      and str(g.get("key")) not in self.reached and str(g.get("key")) not in self.parked
                      and str(g.get("key")) != skip and _dist3(pos, g["pos"]) < here]
        return self._nearest(candidates, pos)

    def _tick_patience(self, campaign: dict | None, pos, fought: bool) -> None:
        """A1: park the current target once it has gone `patience_steps` decisions without getting closer.

        The clock has its OWN baseline, `_clock_best`, rather than reading `gate_approach`. They look like the
        same question and are not: `best_dist` is episode scoped by ruling R1 and deliberately survives an
        in-episode death, so after a respawn the whole re-walk of ground already covered pays no approach at all
        -- and a clock reading the reward would run out on a target the agent is walking straight at. See
        `mark_paid`, which restarts the clock for exactly that reason. Within one uninterrupted attempt the two
        agree by construction, because both are "a new best by more than `min_gain_m`".

        The arena suspension is BOUNDED. `arena_enemies_alive` counts every enemy whose `ActivateNextWave` has
        not run (CampaignObserver.cs), scoped neither to the target's own arena nor to the player's room, so an
        un-cleared wave anywhere on the level used to freeze the clock for the rest of the level load: 20,000
        decisions under an unreachable door produced zero parks, and the telemetry read `targets_parked = 0`,
        which on the dashboard is indistinguishable from "monotone level, correctly inert". Suspended decisions
        are counted instead, and the park is forced once clock + suspended reaches `2 * patience_steps` -- 600
        decisions at the shipped settings, still inside the 675 of `stuck_seconds` 45, so the episode is still
        the outer bound. Below that bound the suspension does its job: on the recorded 0-1 run four stalls of up
        to 2,543 decisions ran with an arena alive on 96-100% of their steps, where no approach is possible
        however well the agent plays. On 0-3's unreachable door the arena count is 0 on 100% of the locked steps.

        Two targets are never parked:

          - the exit, which is terminal: parking it would leave nothing to aim at;
          - a fetch or carry leg, and any gate still reporting `needs_item`. That one is a safety rule, not a
            tuning choice. `env._protect_carry` keys the punch-drop on `gates.target` being the ALTAR sub-goal of
            a held item, because protection and release have to come from one source or the button can be taken
            away and never given back. Parking the gate deletes the leg, `_subgoal` never rebuilds it, and the
            protection silently switches off while the skull is still in the player's hands -- and the policy
            presses punch on ~35% of decisions, which `Punch.ActiveStart` turns into a throw. Reproduced end to
            end. A skull-locked door is held shut by its altar, not by geometry, which is the same reason the
            arena case is exempt; the difference is that here the cost of being wrong is the level load.
        """
        key = self._key_of(self.target)
        if key != self._clock_key:
            # Not a `return`, and the baseline is seeded HERE rather than on the first tick: the step a target is
            # taken is its first decision, so `patience_steps` decisions holding one target is exactly
            # `patience_steps` calls, with no off-by-one against the config.
            self._clock_key, self._clock, self._clock_suspended = key, 0, 0
            self._clock_best = _dist3(pos, self.target["pos"]) \
                if pos is not None and (self.target or {}).get("pos") else math.inf
        gate_key = self._park_key(self.target)
        if not self.patience_active or pos is None or key is None or gate_key in (None, GATE_EXIT_KEY):
            return  # the exit is never parked: it is the terminal target, and parking it leaves nothing
        if (self.target or {}).get("subgoal") in (SUBGOAL_ITEM, SUBGOAL_ALTAR):
            return  # a carry in progress: see the docstring. Never park the machine holding the skull.
        gate = self._gate_by_key(campaign, gate_key)
        if (gate or {}).get("needs_item"):
            return  # blocked by an altar rather than by geometry, exactly like an arena-held door
        distance = _dist3(pos, self.target["pos"]) if self.target.get("pos") else math.inf
        if distance < self._clock_best - self.min_gain:
            self._clock_best = distance
            self._clock = self._clock_suspended = 0
            return
        held = ((campaign or {}).get("arena_enemies_alive") or 0) > 0 or fought
        if held:
            self._clock_suspended += 1
        else:
            self._clock += 1
        forced = self._clock + self._clock_suspended >= 2 * self.patience_steps
        if (self._clock < self.patience_steps and not forced) or gate_key in self.parked:
            return
        self._clock = self._clock_suspended = 0  # park or not, the next one is another full window away
        # A2 + A1: the ladder's own pick is only ever SWITCHED away from, never abandoned -- with nothing better
        # to aim at, its answer remains the best guess available. A FALLBACK pick gets no such protection, and
        # must not: `_pick` chose it as the nearest unreached, unparked, active gate and `_nearer_unreached`
        # filters that identical set, so nothing can ever be strictly nearer and the test would make every
        # fallback target permanent. That turned the wedge into a wedge one door over -- reproduced as 4,000
        # consecutive decisions aimed through a ceiling with `targets_parked` reading 1, the mechanism showing
        # as fired. A parked fallback simply hands over to the next-nearest, which is the A2 rule already.
        if not self._fallback and self._nearer_unreached(campaign, pos, self.target) is None:
            return
        here = _dist3(pos, gate["pos"]) if gate and gate.get("pos") else math.inf
        if key == gate_key:  # a plain gate target: the clock's own baseline is this attempt's closest approach
            here = min(here, self._clock_best)
        self.parked.add(gate_key)
        # Monotone across parks: a re-park can lower the baseline the un-park bar is measured from, never raise
        # it, so wandering away and stalling somewhere farther off cannot buy a cheap un-park. See `_unpark`.
        self.park_best[gate_key] = min(here, self.park_best.get(gate_key, math.inf))
        self.park_count[gate_key] = self.park_count.get(gate_key, 0) + 1
        self.parks += 1
        self.retarget(campaign, pos)
        self._clock_key = self._key_of(self.target)
        # The replacement target starts its own full window against its own baseline, taken here so its first
        # decision counts the same as any other target's.
        self._clock_best = _dist3(pos, self.target["pos"]) \
            if self.target is not None and self.target.get("pos") else math.inf

    # -- helpers -----------------------------------------------------------------------------

    def _rooms(self, campaign: dict | None) -> list[dict]:
        """Layer 2: the shipped room trunk, or [] -- today's behaviour -- when there is none or it is stale.

        Guard I5, staleness. A trunk is offline data built against one particular `FinalPit`; a game update
        that moves the pit invalidates the whole file, and a wrong route is worse than none. So the first frame
        of a load that actually reports an exit decides it, once: more than `route_exit_tol_m` from the
        `exit.pos` the file was built against and the level falls to layer 3 for the rest of the load.

        On refusal the TARGET and `best_dist` are cleared as well, which is the difference between falling back
        and pretending to. `_choose_target` opens with `if not active: return self.target`, so without the
        clear it would keep handing out the last stale rung while `gate_approach` went on paying toward it --
        the failure the reviewer found in revision 1 of the spec.

        A frame with no exit at all (mid-load, or a mod that dropped the block) uses the trunk UNVALIDATED and
        leaves the verdict for a later frame: the alternative is having no route on the frames that a fresh
        load reports before the pit is found, and the check is re-run until it can be answered.
        """
        if not self._route:
            return []
        if self._route_ok is None:
            exit_ = (campaign or {}).get("exit")
            pos = _point(exit_.get("pos")) if isinstance(exit_, dict) else None
            if pos is None:
                self.route_reads += 1
                return self._route["rungs"]  # mid-load frame: use it, decide the check when there is an exit
            self._route_ok = _dist3(pos, self._route["exit_pos"]) <= self.route_exit_tol
            if not self._route_ok:
                if not self._route_warned:
                    self._route_warned = True
                    print("GateProgress: the route file for %s is stale (its exit is %.1f m from the one the "
                          "game reports); falling back to the exit vector"
                          % (self._route.get("level"), _dist3(pos, self._route["exit_pos"])), flush=True)
                self.target = None
                self.best_dist.clear()
                self.route_source = ROUTE_SOURCE_NONE
        if not self._route_ok:
            return []
        self.route_reads += 1
        return self._route["rungs"]

    def _is_room_ladder(self, rungs: list[dict]) -> bool:
        """Whether `rungs` IS the loaded trunk -- identity against the document, never a field on the entries.

        Object identity is what guarantees nothing in the data can turn the rooms-only rules on for a gate: the
        gates success path builds a fresh list every call, so it can never be this object. The
        `self._route is not None` half short-circuits, so a gates level never subscripts None.
        """
        return self._route is not None and rungs is self._route["rungs"]

    def _gates(self, campaign: dict | None) -> list[dict]:
        """The ladder in use: `_layer_gates()`, except where a collapsed level prefers its shipped trunk.

        The preference is the ONLY thing between the caller and the layer precedence this class has always had,
        and it is off unless three things hold at once: the run asked for it (`prefer_route_when_collapsed`),
        this level ships a trunk, and the detector called its gate ladder collapsed at the spawn. A healthy
        level therefore cannot reach its route file however the flag is set -- which is what makes the flag safe
        to flip per run -- and with the flag off this method IS `_layer_gates`.

        A stale trunk (guard I5) returns [], and the level falls back to its own gate ladder rather than to
        nothing: a collapsed ladder is a bad route, an absent one is no route at all.
        """
        if self._prefers_route():
            rooms = self._rooms(campaign)
            if rooms:
                return rooms
        return self._layer_gates(campaign)

    def _layer_gates(self, campaign: dict | None) -> list[dict]:
        """The usable gate array, the room trunk behind it, or nothing at all.

        Three guards on the gate ladder, all needed:

          - no `campaign` block at all. `env._campaign_progress` passes `raw.get("campaign")`, which is absent
            on every frame of a scene load and whenever the mod's own build of the block throws;
          - `gates_ordered` false: the door graph has no goal room (0-5 and six others), so every `hops` is null;
          - fewer than `hops_min_frac` of the LADDER gates carry a `hops` value. `gates_ordered` is true on 7-2
            (1 of 4) and 8-3 (1 of 32), where `_choose_target` locks onto that single ordered door -- 892 m from
            the start on 8-3 -- and never retargets, so the ladder is worse than no ladder. Measured across the
            campaign the ratio is 1.000 on fourteen levels, then 0.960 (8-2), 0.600 (4-3), 0.538 (8-1),
            0.500 (6-1), 0.250 (7-2), 0.031 (8-3): a `<` test at 0.5 keeps 6-1 exactly on the threshold and drops
            the two degenerate levels. 6-1 is kept by the threshold, not by evidence that its reachable half is
            the half on the route, so this is the first knob if 6-1 ever wedges.

        The ratio is taken over the **phase-1** gates only, i.e. those without `altar_only`. Those measured
        ratios were taken before the mod began appending altar-driven one-room doors as extra gates, and such a
        door carries `hops: null` on all but three levels (1-1, 5-3, 8-1) because its room is outside the door
        graph. Counting them would move 6-1 from 6/12 = 0.500 to 6/13 = 0.462 and silently drop the ladder the
        threshold was chosen to keep, and 5-1 from 1.000 to 3/6 = 0.500 -- one more altar door from the same
        fate. A phase-2 gate adds no node and no edge to the graph, so it cannot make the ladder it was appended
        to any less trustworthy; it is additive route information and is returned with the rest on success.
        A level whose gates are ALL `altar_only` has no phase-1 ladder to dilute, so the ratio falls back to the
        whole array and the rule is the unchanged one. The only such level that ships is 7-1, whose four altar
        doors all carry `hops: null`; it scores 0.000 and is dropped either way, exactly as it was before the
        widening, when it had no gates at all. Phase-1-only and whole-array are indistinguishable on all 33
        shipped levels; the fallback exists so that a level whose route genuinely runs through altar doors is
        judged by the same ratio as any other rather than discarded unread.

        Evaluated on every call rather than cached: a mid-load `Scan()` can change the ratio as rooms activate,
        and filtering at most 64 dicts is nothing next to a step. The ladder is taken from the array as sent, so
        on a level with more than 64 gates the mod's nearest-first truncation would make the ratio depend on
        where the player stands; the largest array that ships is 8-1's 52 phase-1 gates plus one phase-2 door,
        so that is latent rather than handled.

        Each of those three arms is where LAYER 2 gets its chance, and the only place it does: `_rooms()`
        returns [] whenever there is no trunk, so all three stay `return []` on every level with no file. The
        success path below is unchanged, byte for byte, which is what makes the fallback unreachable on the 18
        levels the gate ladder already routes.
        """
        if not campaign or not campaign.get("gates_ordered"):
            # The latch (see `_note_reached`): a load already walking its gate ladder does not fall to a trunk
            # on a frame that simply carries no block.
            return [] if self._hops_source == "gates" else self._rooms(campaign)
        gates = [g for g in (campaign.get("gates") or ()) if isinstance(g, dict) and g.get("pos")]
        ladder = [g for g in gates if not g.get("altar_only")] or gates
        if not ladder:
            return self._rooms(campaign)
        with_hops = sum(1 for g in ladder if g.get("hops") is not None)
        if with_hops / len(ladder) < self.hops_min_frac:
            return self._rooms(campaign)
        return gates

    @staticmethod
    def _key_of(target: dict | None) -> str | None:
        return None if target is None else str(target.get("key"))

    @staticmethod
    def _exit(campaign: dict | None) -> dict | None:
        """The exit as a target, or None when the mod reports no `FinalPit`. `hops` None marks it as the exit.

        **Aim at `ground_pos`, not at `pos`.** A `FinalPit`'s transform sits inside the drop it triggers, far
        under anything standable: 61-75 m below the floor on `Level 0-2` and 70 m on `Level 0-3`. Every
        consumer of this method is a place the agent is being told to GO -- look mode 2, `gate_approach`, and
        the target slots of the observation -- so pointing them into the pit points the agent off the ledge
        above it. Measured on 0-2: eleven of one episode's eighteen respawns were falls taken at full health
        while following that vector, and both of that level's real completions triggered at y -25 to -27.5
        while `exit.pos` read y -86.1.

        `ground_pos` is mod 0.7.2's NavMesh-snapped point, the nearest standable ground to the pit, and the
        same point the path hint has always measured its length to. It is absent on an older mod and null
        wherever NavMesh has nothing near the pit, and both fall back to `pos`, which is what every level did
        until now. `spaces.campaign_block`'s slots 0-4 read `exit["pos"]` directly and are deliberately NOT
        changed: that raw vector is a learned input column the policy has read since the run began.
        """
        exit_ = (campaign or {}).get("exit")
        if not exit_ or not exit_.get("pos"):
            return None
        pos = _point(exit_.get("ground_pos")) or list(exit_["pos"])
        return {"key": GATE_EXIT_KEY, "pos": pos, "hops": None,
                "open": False, "locked": False, "active": True}

    def _is_reached(self, gate: dict, pos) -> bool:
        """A cylinder, not a sphere: 8 m horizontal (the DoorController trigger plus the door's height over the
        floor), 6 m vertical so standing on the roof above a door is not "passing" it. An open door doubles
        both; `open` alone never counts, because enemies open doors too.

        A gate that still reports `needs_item` is never reached, however close the player stands. Reaching is
        a proxy for *passing*, and a skull-locked door cannot be passed until its altar is filled -- at which
        point the mod drops `needs_item` and the gate becomes reachable like any other. Without this test the
        ladder rewards walking up to a sealed door and then, because `best_hops` has moved past it,
        `_choose_target` picks the rung BELOW the lock (which carries no `needs_item`, so `_subgoal` returns it
        unchanged) and the fetch/carry machine never fires again. `best_hops` and `reached` are level-load
        scoped, so that state persists through every checkpoint-respawn episode of the load. An old mod sends no
        `needs_item`, so `.get` is None and every 0.5.x level -- 0-1 included -- behaves byte for byte as before.
        """
        hops, gate_pos = gate.get("hops"), gate.get("pos")
        if hops is None or not gate.get("active") or not gate_pos or gate.get("needs_item"):
            return False
        margin = 2.0 if gate.get("open") else 1.0
        if not (math.hypot(pos[0] - gate_pos[0], pos[2] - gate_pos[2]) <= margin * self.reach_h
                and abs(pos[1] - gate_pos[1]) <= margin * self.reach_v):
            return False
        return not self._is_route_rung(gate) or self._on_ground()

    def _is_route_rung(self, gate: dict) -> bool:
        """Whether `gate` IS one of the loaded trunk's own rungs -- object identity, never a field.

        The same rule as `_is_room_ladder` and for the same reason: nothing in a JSON document can
        turn a rooms-only test on for a gate, because the mod's gates are built fresh from the
        `campaign` block every frame and can never be these objects. `load_route` hands each env a
        deep copy and `_choose_target` passes a rung out by reference, so identity survives targeting.
        `self._route is None` (Cyber Grind, `route_fallback: false`, and the 19 levels with no file)
        short-circuits to False, which is what makes the ground rule provably absent there.
        """
        return self._route is not None and any(gate is rung for rung in self._route["rungs"])

    def _on_ground(self) -> bool:
        """Is there ground under the player -- within `route_ground_m` when airborne? Rooms only.

        A gate is a DOOR: you pass through its frame, and the mod's own `hops` graph says what that
        means. A trunk rung is a ROOM CENTROID picked offline, and a centroid carries no promise that
        its 8 m x 6 m cylinder stays inside the room. On `Level 0-3` one did not: the `2 - Side
        Hallway` rung sat 1.5 m past the main room's far wall, so its cylinder covered the wall FACE,
        and 680 of the rung's 801 live credit steps were the player hanging against that wall on the
        wrong side, 10-20 m above the main room's floor. The ladder then advanced the target to the
        next rung THROUGH the wall and the agent dropped into the bowl. The rung has been moved (see
        `rung_overrides.json`), but a room centroid can always develop the same overhang, so the
        general defence is here: a room you have not landed in is a room you have not reached.

        "Ground under you" is read exactly as `env._ground_point` reads it -- the mod's centre ground
        ray, falling back to the 8-ray ring's minimum -- so the off-the-map sentinel (the ray length,
        30 m, written when nothing is hit) fails the test by arithmetic and needs no special case.
        `grounded` is the game's own controller flag and short-circuits: it is authoritative, and on
        0-3's second-floor walkway the centre ray reads the full 30 m while the player is provably
        standing on it.

        Permissive whenever it cannot answer, which is what keeps this from being a behaviour change
        anywhere it was not measured: `self._ground` is None outside `update()`, so every ABSORBING
        call (`mark_paid`, `new_level_load`, `retarget`) still absorbs -- refusing to absorb would
        re-arm a rung a respawn revealed, which is a farm, the opposite of what this rule is for.
        A mod that reports neither the flag nor a ray is the same "cannot answer" case.
        """
        if self._ground is None or self.route_ground_m <= 0:
            return True  # nothing to judge from, or the rule is switched off (`route_ground_m: 0`)
        grounded, drop = self._ground
        if grounded is None and drop is None:
            return True
        if grounded:
            return True
        return drop is not None and drop <= self.route_ground_m

    def _note_reached(self, campaign: dict | None, pos) -> None:
        """Adds the rungs the player is standing in to `reached`, and lowers `best_hops` to the lowest of them.

        It also owns the SOURCE GUARD, because this is the only method that turns a hop value into state. The
        gate ladder and the room trunk carry different hop scales -- 0-5's trunk is 5 rungs deep where its
        (unusable) gate graph reports nulls, and a level whose gates come good mid-load would otherwise hand
        `best_hops` from one scale to the other and pay `paid_hops - best_hops` instalments for the difference.
        So a flip forgets the ladder and starts again on the new scale. On a level with no route file the
        source is "gates" on every frame, including the frames where `_gates()` is empty, so the guard can
        never fire there and the class behaves exactly as it did before the route fallback existed.

        `route_source` latches the last layer that produced a usable ladder rather than following every frame,
        because `_gates()` is legitimately empty mid-load and on the frame a level completes (no player, no
        block), and the episode's `info` is built on exactly those frames. It is cleared by `new_level_load`
        and by I5's refusal, so it can only ever name a layer that really did drive this load.

        It is also where the COLLAPSE VERDICT is taken, for the same reason: this is the first thing every frame
        runs, so the verdict is in place before any rule reads it, and it is taken from `_layer_gates` -- the
        layer precedence WITHOUT the collapsed-level route preference -- so that deciding it cannot depend on
        itself. `_collapse_spawn` is the first position of the load, kept even if the ladder only arrives a
        frame or two later (a mid-load block, a game still switching scenes), because the answer is about where
        the level STARTS you, not about where you happen to be when its doors finally report.

        The interaction that used to be documented as impossible is now real, and handled: 0-3 and 4-3 ship a
        route file AND have a usable gate ladder. A frame the mod sent with no `campaign` block would take the
        `not campaign` arm, hand out the trunk unvalidated, flip the source and clear a ladder that was fine --
        losing that load's `gate` income on the one level the patience fix is currently rescuing. So the layer
        is LATCHED per load: once a load's ladder has come from the gates, a block-less frame returns [] rather
        than the trunk. A level whose ladder never came from gates (all twelve trunk levels) is untouched,
        because its source is "rooms" from its first frame.
        """
        if self.ladder_collapsed is None:
            if self._collapse_spawn is None and pos is not None:
                self._collapse_spawn = [float(v) for v in pos]
            verdict = detect_collapsed_ladder(self._layer_gates(campaign), self._collapse_spawn)
            if verdict is not None:
                self.ladder_collapsed = verdict
        rungs = self._gates(campaign)
        source = "rooms" if self._is_room_ladder(rungs) else "gates"
        if self._hops_source is not None and source != self._hops_source:
            self.best_hops = self.paid_hops = None  # a different ladder is a different hop scale
            self.reached.clear()
            self.hops_reached.clear()
        self._hops_source = source
        if rungs:
            self.route_source = ROUTE_SOURCE_ROOMS if source == "rooms" else ROUTE_SOURCE_GATES
        if pos is None:
            return
        for gate in rungs:
            if self._is_reached(gate, pos):
                self.reached.add(str(gate.get("key")))
                hops = int(gate["hops"])
                self.hops_reached.add(hops)
                if self.best_hops is None or hops < self.best_hops:
                    self.best_hops = hops

    @staticmethod
    def _nearest(gates: list[dict], pos) -> dict | None:
        return min(gates, key=lambda g: _dist3(pos, g["pos"])) if gates else None

    def _ladder_pick(self, target: dict | None) -> dict | None:
        """Marks `target` as the ladder's own choice, so reaching it cannot pay a fallback instalment."""
        self._fallback, self._fallback_key = False, None
        return target

    def _pick(self, campaign: dict | None, active: list[dict], rung: list[dict], pos) -> dict | None:
        """A2: the nearest unparked gate on `rung`, else the fallback. None when neither exists.

        The fallback is STICKY within `fallback_hysteresis` metres rather than re-picking the nearest every
        step: pure "nearest" flaps between two near-equidistant doors 23-59 times per 0-3 episode, which is
        noise in observation slots 448-455 and in look mode 2. At 10 m the flapping halves to 9-20 with no
        instalment lost on a walk of 0-3's real route (8 either way, against full stickiness's 5).
        """
        free = [g for g in rung if str(g.get("key")) not in self.parked]
        if free:
            return self._ladder_pick(self._nearest(free, pos))
        if not self.patience_active:
            return None  # parking off: no fallback exists, so the choice is byte for byte the pre-patience one
        candidates = [g for g in active
                      if str(g.get("key")) not in self.reached and str(g.get("key")) not in self.parked]
        if not candidates:
            return None
        nearest = self._nearest(candidates, pos)
        sticky = next((g for g in candidates if str(g.get("key")) == self._fallback_key), None)
        pick = nearest
        if sticky is not None and _dist3(pos, nearest["pos"]) > _dist3(pos, sticky["pos"]) - self.fallback_hysteresis:
            pick = sticky
        self._fallback, self._fallback_key = True, str(pick.get("key"))
        return pick

    def _seed_start(self, rungs: list[dict], active: list[dict], pos) -> None:
        """§6's seeding rule: ABSORB the rung the player starts at, so the trunk aims forward. Rooms only.

        `_choose_target` with `best_hops is None` returns the NEAREST candidate, which on a gate array is the
        right answer -- the mod only reports doors, and the nearest door is where the player is. A room trunk
        is a total order over the level, and the nearest rung of a total order can be BEHIND the player: at
        0-1's real spawn (39.7, -0.5, 343.7) the nearest room rung is `2B - Hallway`, 31.6 m back, while the
        route's next rung is `3 - Gun Room`, 34.4 m ahead. Aiming at the room the player just left is the
        "wrong route is worse than none" case this whole spec exists to avoid.

        So the rung the player starts at is absorbed -- `best_hops`, `paid_hops`, `reached`, `hops_reached` and
        `paid_fallback` all set exactly as `mark_paid` sets them for a gate the player is standing at -- and
        `_choose_target` then targets the rung BELOW it. Absorbing pays nothing by construction (the ladder only
        pays `paid_hops - best_hops`, and `paid_fallback` bars the other payment rule), which is correct: the
        player did not travel to the rung it spawned at, and it means the rule cannot be farmed. All five, not
        just the two hop floors: see the comment on the assignments for what the missing three cost.

        Fires ONCE PER LEVEL LOAD, on the rooms path only:

          - `_seed_pending` is set by `new_level_load` and nowhere else. `best_hops` is level-load scoped and
            `mark_paid` never clears it (spec §6 corrects revision 1 here), so a respawn does not re-seed --
            and the flag also stops a later retry, which could otherwise absorb a rung the agent had walked
            150 m to earn. A spawn farther than `route_seed_m` from every rung simply gets today's
            nearest-candidate behaviour, which is the honest answer when the trunk does not reach the spawn;
          - `_is_room_ladder` is object identity against the loaded document, so a gates level never reaches
            the rule at all -- which is half of why `route=None` provably changes nothing.

        `route_seed_m` is 150 m, calibrated against 0-1's real spawn. The offline parse has no per-level spawn
        point (every level's `spawn` field reads the player prefab's authored transform), so it is UNVALIDATED
        on the other 11 levels -- spec §2.8 and risk 7.

        The cost of being wrong is bounded but real: absorbing a rung forgoes the `gate` instalments of every
        rung above it, so a trunk that loops back to within 150 m of the spawn would seed low and skip most of
        its own ladder. `route_source` plus a `gates_reached` that starts high on a fresh load is the detector;
        the recovery is the data (spec §12.4), not a knob here.
        """
        if not (self._seed_pending and self.best_hops is None and self._is_room_ladder(rungs)):
            return
        self._seed_pending = False  # one chance per load, taken on the first frame with a player and a ladder
        near = self._nearest(active, pos)
        if near is None or _dist3(pos, near["pos"]) > self.route_seed_m:
            return
        key, hops = str(near.get("key")), int(near["hops"])  # `.get`, as everywhere else that keys a gate
        self.best_hops = hops
        # A FLOOR, not an assignment: `new_level_load(keep_paid=True)` (env._respawn's reload branch) has just
        # restored the continuing episode's payment record, and a re-seed must not raise it back up and re-arm
        # the rungs below. On every other load `paid_hops` is None here and this is plain `= hops`.
        self.paid_hops = self._floor(self.paid_hops, hops)
        # Absorbing is `mark_paid` semantics, and `mark_paid` records a gate the player stands at in ALL FOUR
        # places, not two. Setting only the hop floor leaves the rung looking unreached, and it is by
        # construction the NEAREST rung, so `_nearer_unreached` finds it strictly nearer than the forward target
        # from the first decision of the load -- which satisfies the patience rule's "a park is only ever a
        # switch" guard and makes the rung AHEAD parkable. `_pick`'s fallback then picks the absorbed rung
        # BEHIND the player and `_pay_fallback` pays it an instalment, so the one load both aims at the room it
        # has already left and pays for walking back to it: the exact failure this rule exists to prevent, plus
        # a payment the docstring above says is impossible. Reproduced on the §6 geometry with the shipped
        # `gate_target_patience_s` 20.0: parked at decision 299, total 3 instalments over a 3-rung trunk.
        self.reached.add(key)
        self.hops_reached.add(hops)
        self.paid_fallback.add(key)

    def _choose_target(self, campaign: dict | None, pos) -> dict | None:
        if pos is None:
            return self.target  # no player this frame: keep pointing where we were
        rungs = self._gates(campaign)
        active = [g for g in rungs if g.get("hops") is not None and g.get("active")]
        if not active:
            return self.target  # keep the previous target rather than flapping while the graph is rebuilt
        self._seed_start(rungs, active, pos)
        if self.best_hops is None:
            return self._pick(campaign, active, active, pos) or self._exhausted(campaign)
        if self.best_hops == 0:
            exit_ = self._exit(campaign)
            if exit_ is not None:
                return self._ladder_pick(exit_)
            zero = [g for g in active if int(g["hops"]) == 0]
            return self._pick(campaign, active, zero, pos) or self._exhausted(campaign)
        lower = {int(g["hops"]) for g in active if int(g["hops"]) < self.best_hops}
        if not lower:
            exit_ = self._exit(campaign)
            if exit_ is not None:
                return self._ladder_pick(exit_)
            return self._pick(campaign, active, [], pos) or self._exhausted(campaign)
        rung = [g for g in active if int(g["hops"]) == max(lower)]
        return self._pick(campaign, active, rung, pos) or self._exhausted(campaign)

    def _exhausted(self, campaign: dict | None) -> dict | None:
        """Nothing left to pick: A2's "then the exit when none remain", else the target we already had.

        Only reachable with patience ON, and only once every unreached gate has been parked -- which Finding 1's
        relaxation makes possible, since a parked fallback now hands over to the next candidate and can walk the
        list to the end. Keeping a parked target there would aim at a door already proved unreachable, whereas
        the exit is at worst a straight line the novelty and path terms already point along. With patience off
        `_pick` only returns None in the branches that have already tested `_exit` and found nothing, so this is
        byte for byte the old `_ladder_pick(self.target)`.
        """
        return self._ladder_pick(self._exit(campaign) or self.target)

    def _subgoal(self, campaign: dict | None, gate: dict | None, pos) -> dict | None:
        """The chosen gate, or the rung below it when a skull has to be fetched to open it.

        A skull-locked door reports exactly like a walk-up door -- `ItemPlaceZone.ColorDoors` deactivates its
        `DoorController`s, so it reads `open: false, locked: false, controller_active: false`, the same signature
        an arena-held door has. `needs_item` is the only field that separates "kill 11 enemies" from "fetch a
        skull", and without it the gate is a dead end the agent pays `gate_approach` to reach and can never pass.

        Three states, all derived from this step's block alone, so there is nothing to keep in sync and a respawn
        re-derives them for free:

          1. fetch  -- the gate needs type T and nothing of type T is held: target the nearest live free T item;
          2. carry  -- something of type T is held: target the nearest unfilled wired T altar;
          3. open   -- every wired altar is filled: target the gate again.

        The sub-goal's `best_dist` key is the item TYPE and the gate, never the instance: the anti-farm rule is
        "seed a key once per episode and never re-seed it", which only holds while the key string is stable, and
        `CheckPoint.ResetRoom` re-instantiates skulls under new mod-side keys. Two sources of one type therefore
        share one approach budget within an episode, which is strictly anti-farm. `hops` is inherited from the
        gate so the sub-goal packs through `campaign_block`'s existing gate branch with no layout change -- which
        does mean slot 455 reports the gate's hops while the target is a fetch or carry leg, and is not a route
        distance then.

        Against a mod that sends no `needs_item`/`altars`/`items` this returns the gate untouched.
        """
        campaign = campaign or {}
        if not gate or gate.get("subgoal"):
            return gate  # already a sub-goal: a target kept from a frame with no player
        need = gate.get("needs_item")
        if not need or pos is None:
            return gate
        altars = [a for a in wanting_altars(campaign, need)
                  if any(isinstance(d, dict) and d.get("key") == gate.get("key") for d in (a.get("doors") or ()))]
        if not altars:
            return gate  # 3. every wired altar is filled (or only dead twins are left): the lock is open
        items = [i for i in (campaign.get("items") or ())
                 if isinstance(i, dict) and i.get("pos") and i.get("item") == need]
        if any(i.get("held") for i in items):
            target, kind = self._nearest(altars, pos), SUBGOAL_ALTAR  # 2. carrying
        else:
            altar_keys = {a.get("key") for a in altars}
            # "Not held and not placed" finds nothing on 1-1, where every ItemIdentifier starts inside an
            # ItemPlaceZone -- a source skull sits on a pedestal, which is a zone. The test is "not already in
            # one of the altars we are trying to fill".
            free = [i for i in items if not i.get("held") and _live(i) and i.get("placed_in") not in altar_keys]
            if not free:
                return gate  # no reachable source: fall back to the gate rather than pointing at nothing
            target, kind = self._nearest(free, pos), SUBGOAL_ITEM  # 1. fetch
        key = f"{kind}:{need}" if kind == SUBGOAL_ITEM else f"{kind}:{need}:{gate.get('key')}"
        # An altar is aimed at, approached and range-tested at its COLLIDER centre, never at the transform
        # position the mod reports as `pos`: `Punch.AltHit` has to hit the zone's own collider, and that
        # position sits 0.4 m under its lid. See `altar_aim_point`. An item is aimed at where it is.
        pos = altar_aim_point(target) if kind == SUBGOAL_ALTAR else list(target["pos"])
        # `open` and `locked` are 0.0 for every non-gate target, so with target_kind_slots on they carry "this is
        # an item" / "this is an altar" instead -- the width-preserving escape hatch, off by default because
        # turning it on changes what two learned input columns mean.
        kinds = self.target_kind_slots
        return {"key": key, "pos": pos, "hops": gate.get("hops"),
                "open": bool(kinds and kind == SUBGOAL_ITEM), "locked": bool(kinds and kind == SUBGOAL_ALTAR),
                "active": True, "subgoal": kind, "gate_key": gate.get("key"), "item": need}


class ExitGuard:
    """Freezes `campaign.exit.pos` per level load, so a banished `FinalPit` cannot move the goal.

    `CheckPoint.Start` (decompiled/CheckPoint.cs:132) and `CheckPoint.ResetRoom` (:681) clone every room the
    checkpoint owns and then move the ORIGINAL by `transform.position.x + 10000f`. The clone -- and the real
    `FinalPit` trigger -- stay where they were, but the mod's frozen exit reference follows the original, so on
    `Level 0-2` the reported exit jumps (-199.0, -86.1, 277.0) -> (9801.0, -86.1, 277.0) the moment checkpoint
    `-55,-11,277` activates, which is exactly when the gate ladder hands the target over to the exit.
    `ResetRoom` runs again on every respawn, so the offset is k * 10000 for k >= 1.

    The guard rewrites the block IN PLACE, which is what lets one call cover all three consumers with no
    signature change: `GateProgress._exit` (targeting), `spaces.campaign_block` slots 0-4 (the observation) and
    `env._exit_dist_min`. The proper fix is mod-side, at the next rebuild.

    Recovery matters as much as rejection: the banish only ever ADDS to x, so a later report a whole multiple of
    10000 BELOW the frozen one is the true pit coming back (the mod re-scanned and found the live clone), and
    that is accepted rather than ignored -- otherwise a load whose first report was already banished would stay
    wrong for its whole life.
    """

    BANISH_X = 10000.0  # CheckPoint.Start / ResetRoom, read from the decompiled game
    AXIS_EPS = 1.0  # how far y and z may differ and still read as "the same pit, moved along x"

    def __init__(self, max_shift_m: float = 100.0, banish_x: float = BANISH_X, axis_eps: float = AXIS_EPS):
        self.max_shift = float(max_shift_m)
        self.banish_x = float(banish_x)
        self.axis_eps = float(axis_eps)
        self.frozen: list[float] | None = None
        self.banished = False  # whether any report has been rejected since the level loaded
        self.rejections = 0

    def new_level_load(self) -> None:
        """A real level load: whatever the fresh scene reports is the truth, however far it is from the last."""
        self.frozen = None
        self.banished = False

    def apply(self, campaign: dict | None) -> bool:
        """Rewrites `campaign["exit"]["pos"]` to the frozen position. Returns whether this report was rejected."""
        exit_ = (campaign or {}).get("exit")
        pos = exit_.get("pos") if isinstance(exit_, dict) else None
        if not pos or len(pos) < 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos[:3]):
            return False
        if self.frozen is None:
            self.frozen = [float(v) for v in pos[:3]]
            return False
        dx, dy, dz = (float(pos[i]) - self.frozen[i] for i in range(3))
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6:
            return False
        multiples = self._banish_multiples(dx, dy, dz)
        if multiples < 0:  # a whole multiple back toward x = 0: the live pit, not a banished twin
            self.frozen = [float(v) for v in pos[:3]]
            return False
        if multiples > 0 or math.sqrt(dx * dx + dy * dy + dz * dz) > self.max_shift:
            exit_["pos"] = list(self.frozen)
            # `ground_pos` was sampled around the position being rejected, so it belongs to the banished
            # twin, not to the frozen pit. Dropping it sends `_exit` back to `pos`, which has just been
            # restored, rather than leaving a standable point 10 km away as the target.
            if "ground_pos" in exit_:
                exit_["ground_pos"] = None
            self.banished = True
            self.rejections += 1
            return True
        self.frozen = [float(v) for v in pos[:3]]  # a small, plausible move: the mod found a better reference
        return False

    def _banish_multiples(self, dx: float, dy: float, dz: float) -> int:
        """How many 10,000 m banish steps this displacement is, signed; 0 when it is not one at all."""
        if abs(dy) > self.axis_eps or abs(dz) > self.axis_eps:
            return 0
        k = round(dx / self.banish_x)
        if k == 0 or abs(dx - k * self.banish_x) > self.axis_eps:
            return 0
        return int(k)


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
