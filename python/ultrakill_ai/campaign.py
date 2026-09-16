"""Campaign helpers: level names, rank maths, and the measures behind the campaign rewards.

Plain Python plus numpy (for the archive file), so all of it is tested without the game in
tests/test_campaign.py. The inputs come from the mod's `campaign` observation block (docs/protocol.md).
"""

from __future__ import annotations

import math
import os
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

    def cell(self, pos) -> tuple[int, int, int]:
        s = self.cell_size
        return (math.floor(pos[0] / s), math.floor(pos[1] / s), math.floor(pos[2] / s))

    def start_episode(self) -> None:
        self._episode.clear()

    def visit(self, pos) -> float:
        """Novelty for being at `pos`: 1/sqrt(N + 1) on the first entry of its cell this episode, else 0."""
        cell = self.cell(pos)
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
