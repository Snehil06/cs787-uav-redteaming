"""Shared helpers: run scenarios in parallel and log one flat row per run."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from loop.scoring import failure_score, outcome
from scenarios.spec import COND_NAMES, VECTOR_NAMES, Scenario, condition_vector
from sim.simulator import SimConfig, simulate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
# capped: each Windows worker re-imports numpy/scipy, and too many at once exhaust the page file
MAX_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 1))


def run_row(vec: np.ndarray, planner: str = "rrt", seed: int = 0, sim_cfg: dict | None = None, **extra) -> dict:
    """Simulate one scenario vector and return its parameters, condition and metrics as one row.
    sim_cfg: SimConfig field overrides, e.g. {"dt": 0.025} or the hardened settings."""
    s = Scenario.from_vector(vec)
    m = simulate(s, planner=planner, seed=seed, cfg=SimConfig(**(sim_cfg or {}))).metrics
    row = dict(zip(VECTOR_NAMES, s.to_vector()))
    row.update({f"c_{k}": v for k, v in zip(COND_NAMES, condition_vector(s))})
    row.update(m)
    row.update({"F": failure_score(m), "outcome": outcome(m), "planner": planner, "seed": seed})
    row.update(extra)
    return row


def _run_star(args):
    vec, planner, seed, extra = args[:4]
    sim_cfg = args[4] if len(args) > 4 else None
    return run_row(vec, planner, seed, sim_cfg, **extra)


def run_many(vectors, planner="rrt", seeds=None, extras=None, workers=None, chunksize=8,
             sim_cfg: dict | None = None) -> pd.DataFrame:
    """Run many scenario vectors across CPU cores; rows come back in input order."""
    n = len(vectors)
    seeds = list(range(n)) if seeds is None else list(seeds)
    extras = [{}] * n if extras is None else extras
    jobs = [(np.asarray(v), planner, int(sd), ex, sim_cfg) for v, sd, ex in zip(vectors, seeds, extras)]
    workers = workers or MAX_WORKERS
    with Pool(workers) as pool:
        rows = pool.map(_run_star, jobs, chunksize=chunksize)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str = "") -> str:
    oc = df["outcome"].value_counts(normalize=True)
    return (f"{label} n={len(df)}  failure={oc.get('failure', 0):.1%}  "
            f"near_failure={oc.get('near_failure', 0):.1%}  collision={df['collision'].mean():.1%}  "
            f"F mean={df['F'].mean():.3f}")
