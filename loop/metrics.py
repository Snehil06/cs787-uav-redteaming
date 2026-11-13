"""Red-teaming metrics for one method run (rows in the order they were flown)."""
import numpy as np
import pandas as pd

from loop.redteam import novelty_filter
from scenarios.spec import VECTOR_NAMES

# model-space distance below which two failures count as the same case; two independent random
# scenarios sit ~1.2 apart (median nearest-neighbour distance, stage 2), so 1.0 means "clearly a variant"
MIN_DIST = 1.0
TYPES_AT = 200     # failure types are also counted among each run's first 200 failures (fair across methods)
GRID = {           # condition-space grid for coverage: a cell counts once it holds a failure
    "c_speed": [2, 4, 6, 8, 10.01],
    "c_n_obstacles": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    "c_dynamic_frac": [-0.01, 0.01, 1.01],
    "c_wind_mag": [0, 2, 4, 6, 8.01],
    "c_sense_range": [5, 15, 25.01],
}
N_CELLS = int(np.prod([len(v) - 1 for v in GRID.values()]))


def unavoidable_mask(df: pd.DataFrame, a_max: float = 5.0, delay: float = 0.2) -> np.ndarray:
    """Failures caused by an obstacle that appeared less than one braking time (plus sensing delay) before the
    closest approach: no planner could react. These exploit a gap in the validity rules (see FINAL_REPORT)."""
    k = df.closest_obstacle.astype(int).clip(lower=0).to_numpy()
    t_app = np.array([df[f"o{j}_t_appear"].iloc[i] for i, j in enumerate(k)])
    react = df.t_min_clearance.to_numpy() - t_app
    return (df.outcome == "failure").to_numpy() & (t_app > 0) & (react < df.c_speed.to_numpy() / a_max + delay)


def distinct_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean per row: a failure that is not a near-duplicate of an earlier failure."""
    from generators.codec import encode
    fail = np.flatnonzero((df.outcome == "failure").to_numpy())
    mask = np.zeros(len(df), bool)
    if len(fail):
        z = encode(df.iloc[fail][VECTOR_NAMES].to_numpy(float))
        keep = novelty_filter(z, np.zeros((0, z.shape[1])), MIN_DIST)
        mask[fail[keep]] = True
    return mask


def cells(df: pd.DataFrame) -> pd.Series:
    parts = [pd.cut(df[c], edges, labels=False).astype("Int64").astype(str) for c, edges in GRID.items()]
    return parts[0].str.cat(parts[1:], sep="|")


def summarize(df: pd.DataFrame, cf: pd.DataFrame | None = None) -> dict:
    fail = (df.outcome == "failure").to_numpy()
    dist = distinct_mask(df)
    order = np.cumsum(dist)

    def first(k):
        hit = np.flatnonzero(order >= k)
        return int(hit[0]) + 1 if len(hit) else np.nan

    out = {
        "flights": len(df),
        "failures": int(fail.sum()),
        "failure_rate": float(fail.mean()),
        "near_failures": int((df.outcome == "near_failure").sum()),
        "distinct_failures": int(dist.sum()),
        "coverage_cells": int(cells(df[fail]).nunique()) if fail.any() else 0,
        "flights_to_1st": first(1), "flights_to_10th": first(10), "flights_to_50th": first(50),
        "mean_F": float(df.F.mean()),
    }
    if "valid_frac" in df:
        out["validity"] = float(df.valid_frac.mean())
    if cf is not None and len(cf):
        sig = cf["fixed_by"].fillna("").replace("", "compound")
        sig = sig.loc[[i for i in df.index if i in sig.index]]          # order of discovery
        out["failure_types"] = int(sig.nunique())
        out[f"types_in_first_{TYPES_AT}"] = int(sig.iloc[:TYPES_AT].nunique()) if len(sig) >= TYPES_AT else np.nan
        out["compound_share"] = float((sig == "compound").mean())
    return out
