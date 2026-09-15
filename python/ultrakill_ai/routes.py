"""Campaign routes: a human-recorded path through a level, used for navigation rewards and observations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def route_path(route_dir: str | Path, scene: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene)
    return Path(route_dir) / f"{safe}.json"


@dataclass
class Route:
    scene: str
    points: np.ndarray  # (N, 3) world positions, roughly evenly spaced

    @classmethod
    def load(cls, path: str | Path) -> "Route":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(scene=data["scene"], points=np.asarray(data["points"], dtype=np.float32))

    def save(self, path: str | Path, spacing: float) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scene": self.scene, "spacing": spacing, "points": np.round(self.points, 3).tolist()}
        path.write_text(json.dumps(payload), encoding="utf-8")


class RouteTracker:
    """Tracks how far along a route the player has got.

    The player "reaches" point j when within reach_radius of it, and may only skip ahead by up to
    lookahead points at a time, so shortcuts through walls or falling onto a later section of the
    level don't count as progress.
    """

    def __init__(self, route: Route, reach_radius: float = 4.0, lookahead: int = 20, waypoint_offset: int = 3):
        self.route = route
        self.reach_radius = reach_radius
        self.lookahead = lookahead
        self.waypoint_offset = waypoint_offset
        self.index = 0
        self.best = 0

    def start(self, pos) -> None:
        """Snap to the nearest route point at the start of an episode (e.g. after a checkpoint respawn)."""
        d = np.linalg.norm(self.route.points - np.asarray(pos, dtype=np.float32), axis=1)
        self.index = int(np.argmin(d))
        self.best = self.index

    def update(self, pos) -> int:
        """Returns how many new route points were reached this step (0 if none)."""
        pts = self.route.points
        hi = min(len(pts), self.index + self.lookahead + 1)
        window = pts[self.index : hi]
        d = np.linalg.norm(window - np.asarray(pos, dtype=np.float32), axis=1)
        reached = np.nonzero(d < self.reach_radius)[0]
        if len(reached):
            self.index = self.index + int(reached.max())
        gained = max(0, self.index - self.best)
        self.best = max(self.best, self.index)
        return gained

    @property
    def progress(self) -> float:
        return self.index / max(1, len(self.route.points) - 1)

    @property
    def waypoint(self) -> tuple[float, float, float]:
        p = self.route.points[min(self.index + self.waypoint_offset, len(self.route.points) - 1)]
        return float(p[0]), float(p[1]), float(p[2])

    @property
    def finished(self) -> bool:
        return self.index >= len(self.route.points) - 1
