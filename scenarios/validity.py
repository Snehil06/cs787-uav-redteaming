"""Rule-based checks every scenario must pass before it is simulated."""
import numpy as np

from . import spec
from .spec import (BOUNDS, EDGE_MARGIN, MAP_SIZE, MIN_START_GOAL_DIST, UAV_RADIUS,
                   Scenario)

START_CLEARANCE = 1.0   # extra free space around the start at t = 0, metres
A_MAX, SENSE_DELAY = 5.0, 0.2   # must match sim.simulator.SimConfig defaults
REACT_MARGIN = 1.0      # metres of slack on top of the reaction distance


def _appears_in_reach(s: Scenario, x, y, vx, vy, r, ta) -> bool:
    """True if the obstacle appears closer to the UAV's nominal position than the UAV can react to.

    The UAV is assumed to fly the straight route, accelerating at A_MAX up to its cruise speed. The obstacle must
    appear at least one reaction distance away: sensing delay plus braking distance, plus how far the obstacle
    itself closes in during that time. Otherwise no planner could avoid it, and the crash says nothing about the
    planner."""
    dist = float(np.linalg.norm(s.goal - s.start))
    t_acc = s.speed / A_MAX
    travel = 0.5 * A_MAX * ta ** 2 if ta < t_acc else 0.5 * s.speed * t_acc + s.speed * (ta - t_acc)
    pos = s.start + s.travel_dir * min(travel, dist)
    v_now = min(s.speed, A_MAX * ta)
    reach = (v_now * SENSE_DELAY + v_now ** 2 / (2 * A_MAX)
             + float(np.hypot(vx, vy)) * (SENSE_DELAY + v_now / A_MAX) + REACT_MARGIN)
    return float(np.hypot(x - pos[0], y - pos[1])) - r - UAV_RADIUS < reach


def _within(x, key) -> bool:
    lo, hi = BOUNDS[key]
    return lo - 1e-9 <= x <= hi + 1e-9


def check(s: Scenario) -> list[str]:
    """Return the list of violated rules; an empty list means the scenario is valid."""
    errs = []
    if not _within(s.speed, "speed"):
        errs.append("uav_speed")
    for name, p in (("start", s.start), ("goal", s.goal)):
        if not (EDGE_MARGIN <= p[0] <= MAP_SIZE - EDGE_MARGIN and EDGE_MARGIN <= p[1] <= MAP_SIZE - EDGE_MARGIN):
            errs.append(f"{name}_outside_map")
    if np.linalg.norm(s.goal - s.start) < MIN_START_GOAL_DIST:
        errs.append("start_goal_too_close")

    if not _within(float(np.hypot(*s.wind)), "wind"):
        errs.append("wind")
    if not _within(s.gust_amp, "gust_amp") or not _within(s.gust_t, "gust_t"):
        errs.append("gust")
    if not _within(s.sense_range, "sense_range"):
        errs.append("sense_range")
    if not _within(s.noise_std, "noise_std"):
        errs.append("noise_std")

    obs = s.obstacles[s.active]
    for k, (x, y, vx, vy, r, ta, _) in enumerate(obs):
        if not _within(r, "radius"):
            errs.append(f"obs{k}_radius")
        if not _within(float(np.hypot(vx, vy)), "obs_speed"):
            errs.append(f"obs{k}_speed")
        if not _within(ta, "t_appear"):
            errs.append(f"obs{k}_t_appear")
        if not (_within(x, "obs_pos") and _within(y, "obs_pos")):
            errs.append(f"obs{k}_position")
        if ta <= 1e-9 and np.hypot(x - s.start[0], y - s.start[1]) < r + UAV_RADIUS + START_CLEARANCE:
            errs.append(f"obs{k}_on_start")
        if np.hypot(vx, vy) < 0.1 and np.hypot(x - s.goal[0], y - s.goal[1]) < r + UAV_RADIUS + START_CLEARANCE:
            errs.append(f"obs{k}_on_goal")
        if spec.FAIR_POPUP_RULE and _appears_in_reach(s, x, y, vx, vy, r, ta):
            errs.append(f"obs{k}_appears_in_reach")

    # static obstacles present from the start must not overlap each other
    static0 = obs[(np.hypot(obs[:, 2], obs[:, 3]) < 0.1) & (obs[:, 5] <= 1e-9)]
    for i in range(len(static0)):
        for j in range(i + 1, len(static0)):
            if np.hypot(*(static0[i, :2] - static0[j, :2])) < static0[i, 4] + static0[j, 4]:
                errs.append("static_overlap")
    return errs


def is_valid(s: Scenario) -> bool:
    return not check(s)
