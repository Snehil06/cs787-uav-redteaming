"""Counterfactual reruns: replay a failed scenario with one factor changed at a time.

A change that turns the failure into a success is a measured cause of that failure.
These results are the evidence the LLM analyst is given (it never guesses causes).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from scenarios.spec import BOUNDS, UAV_RADIUS, VECTOR_NAMES, Scenario
from sim.simulator import SimConfig, simulate

# intervention -> human-readable cause it tests
CAUSES = {
    "no_wind": "wind / gusts",
    "full_sensing": "late detection (short sensing range)",
    "no_noise": "sensor noise",
    "no_popups": "pop-up / late-appearing obstacles",
    "slower": "UAV too fast to react",
    "reseed": "unlucky planner randomness",
}
DESCRIPTIONS = {
    "no_wind": "wind and gusts set to zero",
    "full_sensing": f"sensing range raised to {BOUNDS['sense_range'][1]:.0f} m",
    "no_noise": "sensor noise set to zero",
    "no_popups": "late-appearing obstacles present from t = 0 on the same paths",
    "slower": "UAV cruise speed reduced by 30%",
    "reseed": "same scenario re-planned with 3 other RRT random seeds (fixed if at least 2 succeed)",
}
N_RESEEDS = 3


def intervene(s: Scenario, name: str) -> Scenario | None:
    """Return the modified scenario, or None if the intervention does not apply."""
    c = s.copy()
    if name == "no_wind":
        if np.hypot(*s.wind) < 0.5 and s.gust_amp < 0.5:
            return None
        c.wind[:] = 0.0
        c.gust_amp = 0.0
    elif name == "full_sensing":
        if s.sense_range >= BOUNDS["sense_range"][1] - 1e-6:
            return None
        c.sense_range = BOUNDS["sense_range"][1]
    elif name == "no_noise":
        if s.noise_std < 0.02:
            return None
        c.noise_std = 0.0
    elif name == "no_popups":
        late = s.active & (s.obstacles[:, 5] > 0)
        if not late.any():
            return None
        o = c.obstacles
        o[late, 0:2] -= o[late, 2:4] * o[late, 5:6]      # same straight-line path, present from t = 0
        o[late, 5] = 0.0
        on_start = np.hypot(*(o[late, 0:2] - s.start).T) < o[late, 4] + UAV_RADIUS + 0.5
        if on_start.any():
            return None
    elif name == "slower":
        c.speed = max(BOUNDS["speed"][0], 0.7 * s.speed)
    else:
        raise ValueError(name)
    return c


def _ok(m: dict) -> bool:
    return bool(m["goal_reached"])


def analyse(vec, planner: str = "rrt", seed: int = 0, sim_cfg: dict | None = None) -> dict:
    """Run every intervention on one scenario. Values: True (fixes it), False, or None (n/a)."""
    s = Scenario.from_vector(vec)
    cfg = SimConfig(**(sim_cfg or {}))
    out = {}
    for name in CAUSES:
        if name == "reseed":
            wins = sum(_ok(simulate(s, planner, seed=seed + 7919 * (k + 1), cfg=cfg).metrics) for k in range(N_RESEEDS))
            out[name] = wins >= 2
            out["reseed_wins"] = wins
            continue
        c = intervene(s, name)
        out[name] = None if c is None else _ok(simulate(c, planner, seed=seed, cfg=cfg).metrics)
    fixes = [k for k in CAUSES if out[k]]
    out["n_fixes"] = len(fixes)
    out["fixed_by"] = ",".join(fixes)
    return out


def _job(args):
    idx, vec, planner, seed, sim_cfg = args
    r = analyse(vec, planner, seed, sim_cfg)
    r["row"] = idx
    return r


def analyse_failures(df: pd.DataFrame, workers=None, sim_cfg: dict | None = None) -> pd.DataFrame:
    """Counterfactual table for every row of df (expects VECTOR_NAMES, planner, seed columns)."""
    jobs = [(i, row[VECTOR_NAMES].to_numpy(float), row["planner"], int(row["seed"]), sim_cfg)
            for i, row in df.iterrows()]
    from experiments.common import MAX_WORKERS
    workers = workers or MAX_WORKERS
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    return pd.DataFrame(rows).set_index("row")
