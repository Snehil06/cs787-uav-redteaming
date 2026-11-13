"""Adapter around the open-source PythonRobotics RRT / RRT* (vendored, unmodified).

The planners themselves are the system under test. This adapter only:
- seeds their global `random` module from our generator, for reproducible runs,
- silences RRT*'s per-iteration printing,
- shrinks obstacles that already contain the UAV, so the planner can try to escape,
- returns the start-to-goal waypoints, shortcut-smoothed as a standard post-processing step.
"""
import contextlib
import io
import os
import random
import sys

import numpy as np

from scenarios.spec import MAP_SIZE

os.environ.setdefault("MPLBACKEND", "Agg")   # the library imports pyplot; never open windows
_VENDOR = os.path.join(os.path.dirname(__file__), "pythonrobotics")
if _VENDOR not in sys.path:
    sys.path.append(_VENDOR)

from RRT.rrt import RRT                 # noqa: E402  (vendored PythonRobotics)
from RRTStar.rrt_star import RRTStar    # noqa: E402

# Planner settings, fixed for every experiment.
RRT_PARAMS = dict(expand_dis=3.0, path_resolution=0.5, goal_sample_rate=10, max_iter=500)
RRT_STAR_PARAMS = dict(expand_dis=3.0, path_resolution=0.5, goal_sample_rate=10, max_iter=250,
                       connect_circle_dist=20.0, search_until_max_iter=True)


def seg_clear(p: np.ndarray, q: np.ndarray, circles: np.ndarray) -> bool:
    """True if segment p-q stays outside every circle (rows: x, y, r)."""
    if len(circles) == 0:
        return True
    d = q - p
    l2 = d @ d
    c = circles[:, :2]
    t = np.zeros(len(c)) if l2 < 1e-12 else np.clip(((c - p) @ d) / l2, 0.0, 1.0)
    closest = p + t[:, None] * d
    return bool(np.all(((c - closest) ** 2).sum(1) > circles[:, 2] ** 2))


def _relax(start: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """Shrink circles that already contain the start, so the planner can try to escape them."""
    if len(circles) == 0:
        return circles
    circles = circles.copy()
    dist = np.hypot(*(circles[:, :2] - start).T)
    inside = dist < circles[:, 2]
    circles[inside, 2] = np.maximum(dist[inside] - 0.05, 0.0)
    return circles


def shortcut(path: np.ndarray, circles: np.ndarray) -> np.ndarray:
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not seg_clear(path[i], path[j], circles):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


class RRTPlanner:
    def __init__(self, star: bool = False, rng: np.random.Generator | None = None):
        self.star = star
        self.rng = rng or np.random.default_rng()

    def plan(self, start: np.ndarray, goal: np.ndarray, circles: np.ndarray) -> np.ndarray | None:
        """Return waypoints from start to goal, or None if no path was found."""
        circles = _relax(start, circles)
        if seg_clear(start, goal, circles):
            return np.array([start, goal])
        if len(circles) and np.any(np.hypot(*(circles[:, :2] - goal).T) < circles[:, 2]):
            return None                       # goal currently covered by an obstacle

        random.seed(int(self.rng.integers(2**31)))
        obstacles = [tuple(c) for c in circles]
        common = dict(start=list(start), goal=list(goal), obstacle_list=obstacles,
                      rand_area=[0.0, MAP_SIZE], robot_radius=0.0)
        if self.star:
            planner = RRTStar(**common, **RRT_STAR_PARAMS)
            with contextlib.redirect_stdout(io.StringIO()):
                raw = planner.planning(animation=False)
        else:
            planner = RRT(**common, play_area=[0.0, MAP_SIZE, 0.0, MAP_SIZE], **RRT_PARAMS)
            raw = planner.planning(animation=False)
        if raw is None:
            return None
        path = np.array(raw[::-1], dtype=float)  # library returns goal -> start
        path[0] = start
        return shortcut(path, circles)
