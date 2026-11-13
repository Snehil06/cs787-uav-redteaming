"""Procedural scenario generator: the seed dataset and the random baseline.

Obstacles are placed near the start-goal corridor (static) or on crossing courses
timed to meet the UAV (dynamic), so a useful share of random scenarios is non-trivial.
"""
import numpy as np

from . import real_data, spec
from .spec import BOUNDS, EDGE_MARGIN, K_OBS, MAP_SIZE, Scenario
from .validity import is_valid

_REAL = {}


def _real() -> dict:
    if not _REAL:
        _REAL.update(real_data.load())
    return _REAL

P_DYNAMIC = 0.35
P_POPUP = 0.15           # static obstacle that appears late
P_LATE_DYNAMIC = 0.3     # dynamic obstacle that appears late
MAX_RUNUP = 25.0         # metres a dynamic obstacle travels before its crossing point
N_OBS_PROBS = [0.2, 0.3, 0.25, 0.15, 0.07, 0.03]   # ordinary scenarios have few obstacles
CROSS_SPREAD = 8.0       # m, lateral spread of a dynamic obstacle's crossing point
STATIC_SPREAD = 10.0     # m, lateral spread of static obstacles around the route


def _u(rng, key):
    return rng.uniform(*BOUNDS[key])


def _draw(rng: np.random.Generator) -> Scenario:
    m = EDGE_MARGIN + 1.0
    start = np.array([rng.uniform(m, 10.0), rng.uniform(m, MAP_SIZE - m)])
    goal = np.array([rng.uniform(MAP_SIZE - 10.0, MAP_SIZE - m), rng.uniform(m, MAP_SIZE - m)])
    if rng.random() < 0.5:                       # also fly right-to-left
        start, goal = goal, start
    speed = _u(rng, "speed")
    d = goal - start
    dist = np.linalg.norm(d)
    u = d / dist
    normal = np.array([-u[1], u[0]])

    obstacles = np.zeros((K_OBS, 7))
    n_obs = rng.choice(K_OBS + 1, p=N_OBS_PROBS)
    for k in range(n_obs):
        s_frac = rng.uniform(0.15, 0.9)
        t_pass = s_frac * dist / speed          # when the UAV nominally passes this point
        r = _u(rng, "radius")
        if rng.random() < P_DYNAMIC:
            cross = start + s_frac * d + normal * rng.normal(0.0, CROSS_SPREAD)
            t_cross = max(0.5, t_pass + rng.normal(0.0, 2.5))
            ang = np.deg2rad(rng.uniform(30, 150)) * rng.choice([-1, 1])
            heading = np.array([u[0] * np.cos(ang) - u[1] * np.sin(ang),
                                u[0] * np.sin(ang) + u[1] * np.cos(ang)])
            vmag = (real_data.sample(_real()["obstacle_speed_quantiles"], rng, 0.3, BOUNDS["obs_speed"][1])
                    if spec.REAL_STATS else rng.uniform(1.0, BOUNDS["obs_speed"][1]))
            v = heading * vmag
            t_app = rng.uniform(0.0, max(0.0, t_cross - 1.0)) if rng.random() < P_LATE_DYNAMIC else 0.0
            # start at most MAX_RUNUP metres back along its course, so it begins near the map
            t_app = float(np.clip(max(t_app, t_cross - MAX_RUNUP / vmag), *BOUNDS["t_appear"]))
            p0 = cross - v * min(t_cross - t_app, MAX_RUNUP / vmag)
        else:
            v = np.zeros(2)
            p0 = start + s_frac * d + normal * rng.normal(0.0, STATIC_SPREAD)
            t_app = rng.uniform(0.0, min(BOUNDS["t_appear"][1], t_pass)) if rng.random() < P_POPUP else 0.0
        obstacles[k] = [p0[0], p0[1], v[0], v[1], r, t_app, 1.0]

    # draw order (wind speed, direction, gust) is kept identical to the original generator, so that with
    # REAL_STATS off the same seed reproduces the original scenarios exactly
    st = _real() if spec.REAL_STATS else None
    wind_mag = real_data.sample(st["wind_quantiles"], rng, *BOUNDS["wind"]) if st else _u(rng, "wind")
    wind_dir = rng.uniform(-np.pi, np.pi)
    if st:                                       # real wind / gust statistics (SFO, 2023)
        gust = (real_data.sample(st["gust_excess_quantiles"], rng, *BOUNDS["gust_amp"])
                if rng.random() < st["gust_prob"] else 0.0)
    else:
        gust = _u(rng, "gust_amp") if rng.random() < 0.5 else 0.0
    return Scenario(
        start=start, goal=goal, speed=speed, obstacles=obstacles,
        wind=wind_mag * np.array([np.cos(wind_dir), np.sin(wind_dir)]),
        gust_amp=gust,
        gust_t=_u(rng, "gust_t"),
        sense_range=_u(rng, "sense_range"),
        noise_std=_u(rng, "noise_std"),
    )


def sample_scenario(rng: np.random.Generator, max_tries: int = 1000) -> Scenario:
    """Draw scenarios until one passes the validity checks."""
    for _ in range(max_tries):
        s = _draw(rng)
        if is_valid(s):
            return s
    raise RuntimeError("procedural generator failed to produce a valid scenario")
