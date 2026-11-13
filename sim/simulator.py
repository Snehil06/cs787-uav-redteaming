"""2D UAV simulator. The rest of the framework reaches it only through `simulate`.

Fidelity choices (so failures come from physical limits, not simulator quirks):
- the UAV tracks a desired velocity with bounded acceleration and turn rate,
- wind acts as a force the controller has to fight,
- the planner sees only obstacles within sensing range, after a delay, with noise.
"""
import time
from dataclasses import dataclass

import numpy as np

from planners.rrt import RRTPlanner, seg_clear
from scenarios.spec import GOAL_TOL, GUST_DURATION, K_OBS, MAP_SIZE, UAV_RADIUS, Scenario


@dataclass
class SimConfig:
    dt: float = 0.05
    a_max: float = 5.0            # m/s^2, commanded acceleration limit
    turn_rate: float = 1.5        # rad/s, heading-change limit of the commanded motion
    v_max: float = 12.0           # m/s, airframe speed cap
    kp: float = 2.0               # velocity-tracking gain
    wind_coeff: float = 0.4       # wind velocity -> acceleration; strong wind + gust nears a_max
    geofence: float = 5.0         # m beyond the map edge before the flight counts as out of bounds
    sense_delay: float = 0.2      # s between an obstacle's true state and what the planner sees
    replan_period: float = 3.0    # s between routine replans (immediate if blocked or new obstacle)
    retry_period: float = 0.25    # s between retries while no path exists
    margin: float = 0.6           # m safety margin added around obstacles when planning
    dyn_horizon: float = 1.0      # s of obstacle motion swept into its planning footprint
    lookahead: float = 2.0        # m, pure-pursuit lookahead
    t_max_min: float = 20.0
    t_max_factor: float = 3.0     # timeout = factor * nominal travel time
    # ---- hardening options (stage 6). All off by default; RRT/RRT* code itself is never changed.
    wind_ff: bool = False         # feed-forward: cancel the (estimated) wind force in the controller
    speed_governor: bool = False  # cap speed so the UAV can stop within sensing range / before sensed obstacles
    plan_attempts: int = 1        # run the planner k times per replan and keep the shortest path


# settings of the hardened system: each targets a top measured cause from the counterfactual analysis
HARDENED = {"wind_ff": True, "speed_governor": True, "plan_attempts": 3, "dyn_horizon": 2.0}


@dataclass
class Trace:
    metrics: dict
    series: dict | None = None


def obstacle_positions(obs: np.ndarray, t: float):
    """Positions (K, 2) at time t and a mask of obstacles that exist at t."""
    elapsed = np.maximum(t - obs[:, 5], 0.0)
    pos = obs[:, :2] + obs[:, 2:4] * elapsed[:, None]
    present = (obs[:, 6] > 0.5) & (t >= obs[:, 5])
    return pos, present


def wind_at(s: Scenario, t: float) -> np.ndarray:
    w = s.wind.copy()
    if s.gust_amp > 0 and s.gust_t <= t < s.gust_t + GUST_DURATION:
        mag = np.hypot(*w)
        if mag > 0.1:
            direction = w / mag
        else:                                   # no mean wind: gust blows across the route
            direction = np.array([-s.travel_dir[1], s.travel_dir[0]])
        w = w + s.gust_amp * direction
    return w


def _clip_norm(v: np.ndarray, m: float) -> np.ndarray:
    n = np.hypot(*v)
    return v if n <= m else v * (m / n)


def _limit_turn(v_old: np.ndarray, v_new: np.ndarray, max_ang: float) -> np.ndarray:
    s_old, s_new = np.hypot(*v_old), np.hypot(*v_new)
    if s_old < 0.5 or s_new < 0.5:
        return v_new
    ang = np.arctan2(v_old[0] * v_new[1] - v_old[1] * v_new[0], v_old @ v_new)
    if abs(ang) <= max_ang:
        return v_new
    a = np.sign(ang) * max_ang
    c, s = np.cos(a), np.sin(a)
    d = v_old / s_old
    return s_new * np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _planning_circles(centers, radii, vels, cfg: SimConfig) -> np.ndarray:
    """Inflate each sensed obstacle by the UAV size, a margin and its motion over dyn_horizon."""
    if len(centers) == 0:
        return np.zeros((0, 3))
    speed = np.hypot(vels[:, 0], vels[:, 1])
    c = centers + vels * (cfg.dyn_horizon * 0.5)
    r = radii + UAV_RADIUS + cfg.margin + speed * (cfg.dyn_horizon * 0.5)
    return np.column_stack([c, r])


def _path_clear(pos, path, idx, circles) -> bool:
    pts = np.vstack([pos, path[idx:]])
    return all(seg_clear(pts[i], pts[i + 1], circles) for i in range(len(pts) - 1))


def simulate(s: Scenario, planner: str = "rrt", seed: int = 0, cfg: SimConfig | None = None,
             record: bool = False) -> Trace:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    pl = RRTPlanner(star=(planner == "rrt_star"), rng=rng)

    obs = s.obstacles
    radii, obs_vel = obs[:, 4], obs[:, 2:4]
    pos, vel, goal = s.start.astype(float).copy(), np.zeros(2), s.goal
    straight = float(np.linalg.norm(goal - pos))
    t_max = max(cfg.t_max_min, cfg.t_max_factor * straight / s.speed)
    n_steps = int(t_max / cfg.dt)

    path, idx, last_plan = None, 0, -1e9
    seen = np.zeros(K_OBS, bool)
    first_seen_t = np.full(K_OBS, np.nan)
    first_seen_gap = np.full(K_OBS, np.nan)
    n_plans = plan_fail = 0
    plan_time = plan_time_max = 0.0
    min_clear, t_min_clear, closest = np.inf, np.nan, -1
    path_len = effort = 0.0
    max_wind = 0.0
    status = "timeout"
    ser = {"t": [], "pos": [], "vel": [], "obs": [], "wind": [], "plans": []} if record else None

    step = 0
    for step in range(n_steps):
        t = step * cfg.dt

        # perception: delayed, noisy, range-limited
        p_del, pres_del = obstacle_positions(obs, t - cfg.sense_delay)
        gap = np.hypot(*(p_del - pos).T) - radii
        visible = pres_del & (gap <= s.sense_range)
        new = visible & ~seen
        if new.any():
            first_seen_t[new], first_seen_gap[new] = t, gap[new]
            seen |= new
        meas = p_del + rng.normal(0.0, s.noise_std, p_del.shape) if s.noise_std > 0 else p_del
        circles = _planning_circles(meas[visible], radii[visible], obs_vel[visible], cfg)

        # replanning
        blocked = path is not None and not _path_clear(pos, path, idx, circles)
        due = t - last_plan >= (cfg.retry_period if path is None else cfg.replan_period)
        if new.any() or blocked or due:
            t0 = time.perf_counter()
            new_path = None
            for _ in range(cfg.plan_attempts):
                cand = pl.plan(pos, goal, circles)
                if cand is not None and (new_path is None or
                                         np.sum(np.hypot(*np.diff(cand, axis=0).T)) <
                                         np.sum(np.hypot(*np.diff(new_path, axis=0).T))):
                    new_path = cand
            el = time.perf_counter() - t0
            n_plans += 1
            plan_time += el
            plan_time_max = max(plan_time_max, el)
            last_plan = t
            if new_path is None:
                plan_fail += 1
                if blocked:
                    path = None
            else:
                path, idx = new_path, 1
            if record and new_path is not None:
                ser["plans"].append((t, new_path))

        # guidance: pure pursuit along the current path, brake if there is none
        if path is not None:
            while idx < len(path) - 1 and np.linalg.norm(path[idx] - pos) < cfg.lookahead:
                idx += 1
            d = path[idx] - pos
            slow = min(1.0, np.linalg.norm(goal - pos) / 3.0 + 0.2)
            speed = s.speed
            if cfg.speed_governor:
                brake = 0.6 * cfg.a_max          # keep some authority in reserve for wind and turning
                speed = min(speed, np.sqrt(2 * brake * max(s.sense_range - 2.0, 0.5)))
                if visible.any():
                    g = float(np.min(gap[visible])) - UAV_RADIUS - cfg.margin
                    speed = min(speed, max(1.5, np.sqrt(2 * brake * max(g, 0.0))))
            v_des = d / max(np.hypot(*d), 1e-9) * speed * slow
        else:
            v_des = np.zeros(2)

        # dynamics: bounded acceleration and turn rate, wind as an external force
        w = wind_at(s, t)
        max_wind = max(max_wind, float(np.hypot(*w)))
        ff = -cfg.wind_coeff * w if cfg.wind_ff else 0.0
        a_cmd = _clip_norm(cfg.kp * (v_des - vel) + ff, cfg.a_max)
        v_cmd = _limit_turn(vel, vel + a_cmd * cfg.dt, cfg.turn_rate * cfg.dt)
        new_vel = _clip_norm(v_cmd + cfg.wind_coeff * w * cfg.dt, cfg.v_max)
        new_pos = pos + new_vel * cfg.dt
        path_len += float(np.linalg.norm(new_pos - pos))
        effort += float(np.hypot(*a_cmd)) * cfg.dt
        pos, vel = new_pos, new_vel

        # ground truth
        p_true, pres_true = obstacle_positions(obs, t + cfg.dt)
        if record:
            ser["t"].append(t + cfg.dt)
            ser["pos"].append(pos.copy())
            ser["vel"].append(vel.copy())
            ser["obs"].append(np.where(pres_true[:, None], p_true, np.nan))
            ser["wind"].append(w)
        if pres_true.any():
            cl = np.where(pres_true, np.hypot(*(p_true - pos).T) - radii - UAV_RADIUS, np.inf)
            k = int(np.argmin(cl))
            if cl[k] < min_clear:
                min_clear, t_min_clear, closest = float(cl[k]), t + cfg.dt, k
            if cl[k] < 0:
                status = "collision"
                break
        if not (-cfg.geofence <= pos[0] <= MAP_SIZE + cfg.geofence
                and -cfg.geofence <= pos[1] <= MAP_SIZE + cfg.geofence):
            status = "out_of_bounds"
            break
        if np.linalg.norm(goal - pos) <= GOAL_TOL:
            status = "goal"
            break

    t_end = (step + 1) * cfg.dt
    has_close = closest >= 0
    metrics = {
        "status": status,
        "collision": status == "collision",
        "out_of_bounds": status == "out_of_bounds",
        "goal_reached": status == "goal",
        "timeout": status == "timeout",
        "t_end": t_end,
        "nominal_time": straight / s.speed,
        "min_clearance": min_clear if np.isfinite(min_clear) else 99.0,
        "t_min_clearance": t_min_clear,
        "closest_obstacle": closest,
        "closest_detect_time": float(first_seen_t[closest]) if has_close else np.nan,
        "closest_detect_gap": float(first_seen_gap[closest]) if has_close else np.nan,
        "detect_lead_time": float(t_min_clear - first_seen_t[closest]) if has_close else np.nan,
        "n_detected": int(seen.sum()),
        "n_plans": n_plans,
        "n_replans": max(n_plans - 1, 0),
        "plan_failures": plan_fail,
        "plan_time_total": plan_time,
        "plan_time_max": plan_time_max,
        "path_length": path_len,
        "straight_dist": straight,
        "deviation_ratio": path_len / straight if status == "goal" else np.nan,
        "control_effort": effort / t_end,
        "max_wind": max_wind,
        "final_dist_to_goal": float(np.linalg.norm(goal - pos)),
    }
    if record:
        for key in ("t", "pos", "vel", "obs", "wind"):
            ser[key] = np.array(ser[key])
    return Trace(metrics=metrics, series=ser)
