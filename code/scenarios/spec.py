"""Scenario format: a fixed-size parameter vector with K obstacle slots and an active mask.

The generative models produce these vectors; the simulator plays them out.
"""
from dataclasses import dataclass

import numpy as np

MAP_SIZE = 60.0          # square world, metres
K_OBS = 5                # obstacle slots per scenario
UAV_RADIUS = 0.5         # metres
GOAL_TOL = 1.5           # metres from goal counts as reached
GUST_DURATION = 3.0      # seconds a gust lasts

UAV_FIELDS = ("start_x", "start_y", "goal_x", "goal_y", "speed")
OBS_FIELDS = ("x0", "y0", "vx", "vy", "radius", "t_appear", "active")
WIND_FIELDS = ("wind_x", "wind_y", "gust_amp", "gust_t")
SENSE_FIELDS = ("sense_range", "noise_std")

VECTOR_NAMES = (
    list(UAV_FIELDS)
    + [f"o{k}_{f}" for k in range(K_OBS) for f in OBS_FIELDS]
    + list(WIND_FIELDS)
    + list(SENSE_FIELDS)
)
VECTOR_DIM = len(VECTOR_NAMES)

# Physical limits shared by the procedural generator and the validity checks.
BOUNDS = {
    "speed": (2.0, 10.0),
    "radius": (1.0, 4.0),
    "obs_speed": (0.0, 6.0),
    "t_appear": (0.0, 10.0),
    "obs_pos": (-20.0, MAP_SIZE + 20.0),
    "wind": (0.0, 8.0),
    "gust_amp": (0.0, 4.0),
    "gust_t": (0.0, 20.0),
    "sense_range": (5.0, 25.0),
    "noise_std": (0.0, 0.5),
}
MIN_START_GOAL_DIST = 30.0
EDGE_MARGIN = 2.0

# ---- switches for the re-run (see popup-loophole fix). Both off = the original experiments.
# REAL_STATS: draw obstacle speeds and wind/gusts from real data (scenarios/real_stats.json) in the
#             procedural generator, instead of the original uniform ranges.
# FAIR_POPUP_RULE: reject scenarios where an obstacle appears inside the UAV's reaction distance
#             along its route (closes the loophole behind unavoidable pop-up crashes).
REAL_STATS = True
FAIR_POPUP_RULE = True

COND_NAMES = ("speed", "n_obstacles", "dynamic_frac", "wind_mag",
              "wind_rel_cos", "wind_rel_sin", "sense_range")


@dataclass
class Scenario:
    start: np.ndarray        # (2,)
    goal: np.ndarray         # (2,)
    speed: float             # UAV cruise speed, m/s
    obstacles: np.ndarray    # (K_OBS, 7), columns = OBS_FIELDS; position is at t_appear
    wind: np.ndarray         # (2,) mean wind velocity, m/s
    gust_amp: float          # extra wind during the gust, m/s
    gust_t: float            # gust start time, s
    sense_range: float       # metres from UAV to obstacle surface
    noise_std: float         # obstacle position noise, metres

    @property
    def active(self) -> np.ndarray:
        return self.obstacles[:, 6] > 0.5

    @property
    def travel_dir(self) -> np.ndarray:
        d = self.goal - self.start
        return d / max(np.linalg.norm(d), 1e-9)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.start, self.goal, [self.speed],
            self.obstacles.ravel(),
            self.wind, [self.gust_amp, self.gust_t, self.sense_range, self.noise_std],
        ]).astype(np.float64)

    @classmethod
    def from_vector(cls, v) -> "Scenario":
        v = np.asarray(v, dtype=np.float64)
        obs = v[5:5 + K_OBS * 7].reshape(K_OBS, 7).copy()
        obs[:, 6] = (obs[:, 6] > 0.5).astype(np.float64)
        obs[obs[:, 6] == 0] = 0.0          # inactive slots carry no data
        i = 5 + K_OBS * 7
        return cls(
            start=v[0:2].copy(), goal=v[2:4].copy(), speed=float(v[4]),
            obstacles=obs,
            wind=v[i:i + 2].copy(), gust_amp=float(v[i + 2]), gust_t=float(v[i + 3]),
            sense_range=float(v[i + 4]), noise_std=float(v[i + 5]),
        )

    def copy(self) -> "Scenario":
        return Scenario.from_vector(self.to_vector())


def condition_vector(s: Scenario) -> np.ndarray:
    """Coarse description C of a scenario, used to condition the generators."""
    act = s.active
    n = int(act.sum())
    obs_speed = np.hypot(s.obstacles[act, 2], s.obstacles[act, 3])
    dyn_frac = float((obs_speed > 0.1).mean()) if n else 0.0
    wind_mag = float(np.hypot(*s.wind))
    # wind direction relative to the direction of travel (cos = headwind/tailwind, sin = crosswind)
    rel = np.arctan2(s.wind[1], s.wind[0]) - np.arctan2(s.travel_dir[1], s.travel_dir[0])
    cos_r, sin_r = (np.cos(rel), np.sin(rel)) if wind_mag > 1e-6 else (0.0, 0.0)
    return np.array([s.speed, n, dyn_frac, wind_mag, cos_r, sin_r, s.sense_range])
