"""Plot one simulated run: UAV track, obstacle tracks, plans, closest approach.

    python -m experiments.plot_run data/random_rrt.parquet --row 12
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scenarios.spec import MAP_SIZE, VECTOR_NAMES, Scenario
from sim.simulator import simulate


def plot_trace(s: Scenario, tr, title: str = "", ax=None):
    ser, m = tr.series, tr.metrics
    ax = ax or plt.gca()
    ax.set_xlim(-5, MAP_SIZE + 5)
    ax.set_ylim(-5, MAP_SIZE + 5)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fill=False, ls=":", color="grey"))
    for t_plan, path in ser["plans"]:
        ax.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.15, lw=1)
    ax.plot(ser["pos"][:, 0], ser["pos"][:, 1], color="tab:blue", lw=2, label="UAV")
    ax.plot(*s.start, "go", ms=8, label="start")
    ax.plot(*s.goal, "r*", ms=12, label="goal")
    i_min = int(np.nanargmin(np.abs(ser["t"] - m["t_min_clearance"]))) if np.isfinite(m["t_min_clearance"]) else -1
    for k in np.flatnonzero(s.active):
        track = ser["obs"][:, k]
        ok = ~np.isnan(track[:, 0])
        if not ok.any():
            continue
        r = s.obstacles[k, 4]
        moving = np.hypot(*s.obstacles[k, 2:4]) > 0.1
        if moving:
            ax.plot(track[ok, 0], track[ok, 1], "--", color="tab:orange", lw=1)
        p = track[i_min] if i_min >= 0 and ok[i_min] else track[ok][-1]
        hit = k == m["closest_obstacle"]
        ax.add_patch(plt.Circle(p, r, color="tab:red" if hit else "tab:orange", alpha=0.5))
        ax.text(*p, f"{k}", ha="center", va="center", fontsize=8)
    if i_min >= 0:
        ax.plot(*ser["pos"][i_min], "kx", ms=10)
    w = s.wind
    ax.arrow(2, MAP_SIZE + 2, w[0], w[1], head_width=0.8, color="k")
    ax.set_title(title or f"{m['status']}  clear={m['min_clearance']:.2f}m  "
                          f"replans={m['n_replans']}  t={m['t_end']:.1f}s", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--row", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    os.makedirs(args.out, exist_ok=True)
    for r in args.row:
        row = df.iloc[r]
        s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float))
        tr = simulate(s, planner=row["planner"], seed=int(row["seed"]), record=True)
        plt.figure(figsize=(6, 6))
        plot_trace(s, tr)
        f = os.path.join(args.out, f"run_{r}.png")
        plt.savefig(f, dpi=110, bbox_inches="tight")
        plt.close()
        print("saved", f, tr.metrics["status"])


if __name__ == "__main__":
    main()
