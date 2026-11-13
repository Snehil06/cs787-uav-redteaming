"""Evidence record for one flight: the only facts the LLM analyst may use.

Sections: scenario (inputs), trace (what the simulator measured), counterfactuals (measured
causes) and similar_past_cases (background from the edge-case library, never a cause).
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from analyst.counterfactuals import CAUSES, DESCRIPTIONS
from scenarios.spec import K_OBS

# vocabulary the analyst must use for factors; maps to the counterfactual that can confirm it
FACTORS = {
    "wind": "no_wind", "uav_speed": "slower", "late_detection": "full_sensing", "sensor_noise": "no_noise",
    "popup_obstacle": "no_popups", "planner_randomness": "reseed",
    "moving_obstacle": None, "obstacle_density": None, "other": None,
}


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _wind_words(row) -> str:
    if row.c_wind_mag < 0.5:
        return "calm"
    cos, sin = row.c_wind_rel_cos, row.c_wind_rel_sin
    if abs(cos) >= abs(sin):
        kind = "tailwind" if cos > 0 else "headwind"
    else:
        kind = "crosswind from the right" if sin > 0 else "crosswind from the left"
    return f"{kind}, {row.c_wind_mag:.1f} m/s"


def failure_type(cf_row) -> str:
    """Measured failure type = the set of single changes that prevent it."""
    if cf_row is None or not isinstance(cf_row.get("fixed_by"), str) or not cf_row["fixed_by"]:
        return "compound (no single fix)"
    return " + ".join(CAUSES[k] for k in cf_row["fixed_by"].split(","))


def build(row: pd.Series, cf_row: dict | None = None, similar: list | None = None,
          include_counterfactuals: bool = True) -> dict:
    k = int(row.closest_obstacle) if row.closest_obstacle >= 0 else None
    obs = None
    if k is not None:
        v = (row[f"o{k}_vx"], row[f"o{k}_vy"])
        obs = {"id": k, "radius_m": _r(row[f"o{k}_radius"], 1), "speed_m_s": _r(np.hypot(*v), 1),
               "moving": bool(np.hypot(*v) > 0.1), "appears_at_s": _r(row[f"o{k}_t_appear"], 1)}
    n_pop = int(sum(row[f"o{j}_active"] > 0.5 and row[f"o{j}_t_appear"] > 0 for j in range(K_OBS)))
    ev = {
        "outcome": row.outcome,
        "end_status": row.status,
        "scenario": {
            "uav_cruise_speed_m_s": _r(row.c_speed, 1),
            "n_obstacles": int(row.c_n_obstacles),
            "share_moving_obstacles": _r(row.c_dynamic_frac),
            "n_late_appearing_obstacles": n_pop,
            "wind": _wind_words(row),
            "gust": {"amplitude_m_s": _r(row.gust_amp, 1), "starts_at_s": _r(row.gust_t, 1)} if row.gust_amp > 0 else None,
            "sensing_range_m": _r(row.c_sense_range, 1),
            "sensor_noise_std_m": _r(row.noise_std),
        },
        "trace": {
            "flight_time_s": _r(row.t_end, 1),
            "min_clearance_m": _r(row.min_clearance),
            "time_of_min_clearance_s": _r(row.t_min_clearance, 1),
            "closest_obstacle": obs,
            "closest_obstacle_detected_at_s": _r(row.closest_detect_time, 1),
            "gap_when_detected_m": _r(row.closest_detect_gap, 1),
            "seconds_between_detection_and_closest_approach": _r(row.detect_lead_time, 1),
            "replans": int(row.n_replans),
            "planner_failures": int(row.plan_failures),
            "max_wind_felt_m_s": _r(row.max_wind, 1),
            "final_distance_to_goal_m": _r(row.final_dist_to_goal, 1),
        },
    }
    if include_counterfactuals and cf_row is not None:
        ev["counterfactuals"] = {
            k_: {"change": DESCRIPTIONS[k_],
                 "result": "not applicable" if cf_row.get(k_) is None or (isinstance(cf_row.get(k_), float) and np.isnan(cf_row.get(k_)))
                 else ("failure prevented" if cf_row.get(k_) else "still fails")}
            for k_ in CAUSES
        }
    if similar:
        ev["similar_past_cases"] = similar
    return ev


class Library:
    """Edge-case library: past failures with their measured type, searchable by similarity."""
    FEATS = ["c_speed", "c_n_obstacles", "c_dynamic_frac", "c_wind_mag", "c_wind_rel_cos", "c_wind_rel_sin",
             "c_sense_range", "noise_std", "min_clearance", "detect_lead_time", "n_replans"]

    def __init__(self, failures: pd.DataFrame, cf: pd.DataFrame):
        self.df = failures.copy()
        self.df["type"] = [failure_type(cf.loc[i].to_dict() if i in cf.index else None) for i in failures.index]
        x = self.df[self.FEATS].fillna(0).to_numpy(float)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        self.nn = NearestNeighbors(n_neighbors=6).fit((x - self.mu) / self.sd)

    def similar(self, row: pd.Series, k: int = 3) -> list:
        q = (row[self.FEATS].fillna(0).to_numpy(float) - self.mu) / self.sd
        _, idx = self.nn.kneighbors(q[None])
        out = []
        for i in idx[0]:
            r = self.df.iloc[i]
            if r.name == row.name:
                continue
            out.append({"failure_type": r["type"], "uav_speed_m_s": _r(r.c_speed, 1),
                        "n_obstacles": int(r.c_n_obstacles), "wind_m_s": _r(r.c_wind_mag, 1),
                        "sensing_range_m": _r(r.c_sense_range, 1), "min_clearance_m": _r(r.min_clearance)})
            if len(out) == k:
                break
        counts = self.df["type"].value_counts()
        for o in out:
            o["cases_of_this_type_in_library"] = int(counts.get(o["failure_type"], 0))
        return out
