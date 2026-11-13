"""Real-world statistics that ground the scenario distribution.

- Moving obstacles: speeds of pedestrians, cyclists, skaters, carts and cars filmed top-down by a drone over the
  Stanford campus (Stanford Drone Dataset, Robicquet et al., ECCV 2016; annotations via the OpenTraj repo).
- Wind: hourly mean wind and gusts at San Francisco International Airport, 2023 (NOAA Integrated Surface
  Database, global-hourly, station 72494023234), near the same campus.

`extract()` reads the raw downloads in data/external/ once and writes scenarios/real_stats.json (quantiles only,
small enough to commit). The generator only needs that JSON.

    python -m scenarios.real_data
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(__file__)
STATS = os.path.join(HERE, "real_stats.json")
RAW = os.path.join(os.path.dirname(HERE), "data", "external")
SDD = os.path.join(RAW, "OpenTraj", "datasets", "SDD")
NOAA = os.path.join(RAW, "noaa", "72494023234_2023.csv")
FPS = 29.97                      # SDD video frame rate
WINDOW = 30                      # frames: speeds measured over ~1 s
MOVING = 0.3                     # m/s: slower windows count as standing still
Q = np.linspace(0, 1, 101)


def _scales() -> dict:
    """{(scene, video): metres per pixel} from estimated_scales.yaml (simple two-level YAML)."""
    out, scene, video = {}, None, None
    for line in open(os.path.join(SDD, "estimated_scales.yaml"), encoding="utf-8"):
        if re.match(r"^\S.*:\s*$", line):
            scene = line.strip().rstrip(":")
        elif re.match(r"^\s{3}\S.*:\s*$", line):
            video = line.strip().rstrip(":")
        elif "scale:" in line:
            out[(scene, video)] = float(line.split(":")[1])
    return out


def sdd_speeds() -> dict:
    """Per-class arrays of 1-second speeds (m/s) while moving, pooled over all videos."""
    import pandas as pd
    scales = _scales()
    speeds = {}
    for path in sorted(glob.glob(os.path.join(SDD, "*", "video*", "annotations.txt"))):
        scene, video = path.split(os.sep)[-3], path.split(os.sep)[-2]
        if (scene, video) not in scales:
            continue
        a = pd.read_csv(path, sep=" ", header=None, quotechar='"',
                        names=["id", "x0", "y0", "x1", "y1", "frame", "lost", "occluded", "generated", "label"])
        a = a[a.lost == 0].sort_values(["id", "frame"])
        a["cx"] = (a.x0 + a.x1) / 2 * scales[(scene, video)]
        a["cy"] = (a.y0 + a.y1) / 2 * scales[(scene, video)]
        for (tid, label), g in a.groupby(["id", "label"]):
            if len(g) <= WINDOW:
                continue
            f, x, y = g.frame.to_numpy(), g.cx.to_numpy(), g.cy.to_numpy()
            ok = (f[WINDOW:] - f[:-WINDOW]) == WINDOW            # contiguous 1 s windows only
            v = np.hypot(x[WINDOW:] - x[:-WINDOW], y[WINDOW:] - y[:-WINDOW])[ok] * FPS / WINDOW
            v = v[::WINDOW]                                      # non-overlapping samples
            speeds.setdefault(label, []).append(v[v > MOVING])
    return {k: np.concatenate(v) for k, v in speeds.items() if v}


def noaa_wind() -> dict:
    """Hourly mean wind (m/s) and gust excess over the mean (m/s) from routine METAR reports."""
    import pandas as pd
    d = pd.read_csv(NOAA, low_memory=False)
    d = d[d.REPORT_TYPE.str.strip() == "FM-15"]
    w = d.WND.str.split(",", expand=True)
    speed = pd.to_numeric(w[3], errors="coerce")
    good = w[4].isin(["1", "5"]) & (speed < 9999)
    wind = speed[good] / 10.0
    g = d.loc[good, "OC1"].dropna().str.split(",", expand=True)
    gust = pd.to_numeric(g[0], errors="coerce")
    gust = gust[g[1].isin(["1", "5"]) & (gust < 9999)] / 10.0
    excess = (gust - wind.loc[gust.index]).clip(lower=0)
    return {"wind": wind.to_numpy(), "gust_prob": len(gust) / len(wind), "gust_excess": excess.to_numpy()}


def extract() -> dict:
    sp = sdd_speeds()
    wn = noaa_wind()
    counts = {k: int(len(v)) for k, v in sp.items()}
    pooled = np.concatenate(list(sp.values()))
    stats = {
        "sources": {
            "obstacles": "Stanford Drone Dataset (Robicquet et al., ECCV 2016), annotations via github.com/crowdbotp/OpenTraj",
            "wind": "NOAA ISD global-hourly 2023, San Francisco Intl Airport (72494023234), FM-15 reports",
        },
        "obstacle_speed_quantiles": np.quantile(pooled, Q).round(3).tolist(),
        "obstacle_speed_by_class": {k: {"samples": counts[k], "median": round(float(np.median(v)), 2),
                                        "p90": round(float(np.quantile(v, 0.9)), 2)} for k, v in sp.items()},
        "wind_quantiles": np.quantile(wn["wind"], Q).round(2).tolist(),
        "gust_prob": round(float(wn["gust_prob"]), 4),
        "gust_excess_quantiles": np.quantile(wn["gust_excess"], Q).round(2).tolist(),
        "hours": int(len(wn["wind"])),
    }
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    return stats


def load() -> dict:
    with open(STATS, encoding="utf-8") as f:
        return json.load(f)


def sample(quantiles: list, rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw from an empirical distribution given by its 101 quantiles, clipped to [lo, hi]."""
    return float(np.clip(np.interp(rng.random(), Q, quantiles), lo, hi))


if __name__ == "__main__":
    s = extract()
    print("obstacle speeds by class:", json.dumps(s["obstacle_speed_by_class"], indent=1))
    print("pooled obstacle speed median / p90:", s["obstacle_speed_quantiles"][50], s["obstacle_speed_quantiles"][90])
    print(f"wind ({s['hours']} hours): median {s['wind_quantiles'][50]}, p90 {s['wind_quantiles'][90]}, "
          f"max {s['wind_quantiles'][100]} m/s; gust reported in {s['gust_prob']:.1%} of hours, "
          f"median gust excess {s['gust_excess_quantiles'][50]} m/s")
