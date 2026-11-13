"""Failure score F in [0, 1] and outcome labels. Weights are fixed before any experiment."""
import numpy as np

WEIGHTS = {"collision": 0.4, "no_goal": 0.2, "clearance": 0.2, "replans": 0.1, "deviation": 0.1}
D_SAFE = 2.0          # metres of clearance considered comfortable
TAU = 0.5             # softness of the clearance term
REPLAN_SCALE = 20     # replans at which the replan term saturates
NEAR_MISS = 0.5       # metres: a success closer than this is a near-failure


def failure_score(m: dict) -> float:
    crashed = bool(m["collision"] or m["out_of_bounds"])
    clearance = 1.0 / (1.0 + np.exp(-(D_SAFE - m["min_clearance"]) / TAU))
    replans = min(m["n_replans"] / REPLAN_SCALE, 1.0)
    dev = m["deviation_ratio"]
    deviation = 0.0 if not np.isfinite(dev) else float(np.clip(dev - 1.0, 0.0, 1.0))
    return float(
        WEIGHTS["collision"] * crashed
        + WEIGHTS["no_goal"] * (not m["goal_reached"])
        + WEIGHTS["clearance"] * clearance
        + WEIGHTS["replans"] * replans
        + WEIGHTS["deviation"] * deviation
    )


def outcome(m: dict) -> str:
    if not m["goal_reached"]:
        return "failure"
    if m["min_clearance"] < NEAR_MISS:
        return "near_failure"
    return "success"
