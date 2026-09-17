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

    def __init__(self, reach_m: float = 8.0, reach_v_m: float = 6.0, min_gain_m: float = 0.5,
                 hops_min_frac: float = 0.5, target_kind_slots: bool = False):
        self.reach_h = float(reach_m)
        self.reach_v = float(reach_v_m)
        self.min_gain = float(min_gain_m)
        self.hops_min_frac = float(hops_min_frac)
        self.target_kind_slots = bool(target_kind_slots)
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
        self.target = self._subgoal(campaign, self._choose_target(campaign, player_pos), player_pos)
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

    def _gates(self, campaign: dict | None) -> list[dict]:
        """The usable gate array, or nothing at all when this level's ladder cannot be trusted.

        Three guards, all needed:

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
        """
        if not campaign or not campaign.get("gates_ordered"):
            return []
        gates = [g for g in (campaign.get("gates") or ()) if isinstance(g, dict) and g.get("pos")]
        ladder = [g for g in gates if not g.get("altar_only")] or gates
        if not ladder:
            return []
        with_hops = sum(1 for g in ladder if g.get("hops") is not None)
        if with_hops / len(ladder) < self.hops_min_frac:
            return []
        return gates

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
