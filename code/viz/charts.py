"""Result charts for a batch of runs. Each function returns a figure whose title states the finding."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import CAUSES
from viz import style as st

FACTORS = [  # column, label, bins (None = integer categories)
    ("c_speed", "UAV cruise speed (m/s)", [2, 4, 6, 8, 10]),
    ("c_n_obstacles", "number of obstacles", None),
    ("c_dynamic_frac", "share of moving obstacles", ([-0.01, 0.01, 0.34, 0.67, 1.0],
                                                     ["none", "up to 1/3", "1/3-2/3", "over 2/3"])),
    ("c_wind_mag", "mean wind (m/s)", [0, 2, 4, 6, 8]),
    ("crosswind", "crosswind component (m/s)", [0, 1.5, 3, 4.5, 8]),
    ("c_sense_range", "sensing range (m)", [5, 10, 15, 20, 25]),
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["crosswind"] = (df["c_wind_mag"] * df["c_wind_rel_sin"]).abs()
    df["failed"] = df["outcome"] == "failure"
    return df


def _bin_labels(edges):
    return [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]


def outcomes(df: pd.DataFrame):
    n = len(df)
    rows = [
        ("Reached goal safely", ((df.status == "goal") & (df.outcome == "success")).sum(), st.GOOD),
        ("Reached goal, near-miss (< 0.5 m)", (df.outcome == "near_failure").sum(), st.WARNING),
        ("Collision", (df.status == "collision").sum(), st.CRITICAL),
        ("Left the geofence", (df.status == "out_of_bounds").sum(), st.SERIOUS),
        ("Timed out", (df.status == "timeout").sum(), st.SERIOUS),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, c, col) in zip(y, rows):
        ax.barh(yi, c, color=col, height=0.62)
        ax.text(c + n * 0.01, yi, f"{c:,}  ({c / n:.1%})", va="center", fontsize=9, color=st.INK)
    ax.set_yticks(y, [r[0] for r in rows], color=st.INK)
    ax.set_xlim(0, n * 1.12)
    ax.set_xlabel("runs")
    ax.grid(axis="y", visible=False)
    fail = (df.outcome == "failure").mean()
    st.finding(fig, f"{fail:.1%} of {n:,} random flights failed; collisions dominate",
               "Outcome of every run in the random baseline (RRT planner)")
    return fig


def scenario_mix(df: pd.DataFrame):
    cols = [("c_speed", "UAV cruise speed (m/s)"), ("c_n_obstacles", "number of obstacles"),
            ("c_dynamic_frac", "share of moving obstacles"), ("c_wind_mag", "mean wind (m/s)"),
            ("crosswind", "crosswind component (m/s)"), ("c_sense_range", "sensing range (m)"),
            ("noise_std", "sensor noise std (m)"), ("straight_dist", "start-goal distance (m)")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        if c == "c_n_obstacles":
            vc = df[c].value_counts().sort_index()
            ax.bar(vc.index, vc.values, color=st.BLUE, width=0.7)
        else:
            ax.hist(df[c], bins=20, color=st.BLUE, rwidth=0.88)
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("runs")
    fig.tight_layout(h_pad=2)
    st.finding(fig, "The random generator covers each factor broadly, with few-obstacle scenes most common",
               "Distribution of scenario parameters in the seed / random-baseline set")
    return fig


def failure_by_factor(df: pd.DataFrame):
    base = df["failed"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True)
    spreads, peak = {}, 0.0
    for ax, (c, lab, edges) in zip(axes.ravel(), FACTORS):
        names = None
        if isinstance(edges, tuple):
            edges, names = edges
        if edges is None:
            g = df.groupby(c)["failed"].agg(["mean", "size"])
            xs = [f"{int(v)}" for v in g.index]
        else:
            g = df.groupby(pd.cut(df[c], edges), observed=True)["failed"].agg(["mean", "size"])
            xs = (names or _bin_labels(edges))[: len(g)]
        peak = max(peak, g["mean"].max())
        spreads[lab] = g["mean"].max() - g["mean"].min()
        x = np.arange(len(g))
        ax.bar(x, g["mean"] * 100, color=st.BLUE, width=0.68)
        for xi, (r, sz) in enumerate(zip(g["mean"], g["size"])):
            ax.text(xi, r * 100 + 0.6, f"{r:.0%}", ha="center", fontsize=8.5, color=st.INK)
            ax.text(xi, -2.6, f"n={sz}", ha="center", fontsize=7, color=st.MUTED)
        ax.axhline(base * 100, color=st.INK_2, lw=0.8)
        ax.set_xticks(x, xs)
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylim(-4, peak * 100 * 1.15)
    for ax in axes[:, 0]:
        ax.set_ylabel("failure rate (%)")
    axes[0, 0].text(-0.4, base * 100 + 0.6, f"overall {base:.0%}", fontsize=7.5, color=st.INK_2)
    top = max(spreads, key=spreads.get)
    fig.tight_layout(h_pad=2.5)
    st.finding(fig, f"Failure rate depends most on {top}",
               "Share of runs that failed, by scenario factor (grey line = overall rate; n = runs per bin)")
    return fig


def failure_map(df: pd.DataFrame):
    sp = pd.cut(df["c_speed"], [2, 4, 6, 8, 10], labels=_bin_labels([2, 4, 6, 8, 10]))
    tab = df.groupby([df["c_n_obstacles"].astype(int), sp], observed=False)["failed"].agg(["mean", "size"])
    rate = tab["mean"].unstack()
    size = tab["size"].unstack()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", st.BLUE_RAMP)
    im = ax.imshow(rate.values * 100, cmap=cmap, aspect="auto", origin="lower", vmin=0)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v, sz = rate.values[i, j], size.values[i, j]
            if sz > 0:
                dark = v * 100 > im.norm.vmax * 0.55
                ax.text(j, i, f"{v:.0%}\nn={sz}", ha="center", va="center", fontsize=7.5,
                        color=st.SURFACE if dark else st.INK)
    ax.set_xticks(range(rate.shape[1]), rate.columns)
    ax.set_yticks(range(rate.shape[0]), rate.index)
    ax.set_xlabel("UAV cruise speed (m/s)")
    ax.set_ylabel("number of obstacles")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("failure rate (%)")
    cb.outline.set_visible(False)
    worst = rate.stack().idxmax()
    st.finding(fig, f"Hardest region: {worst[0]} obstacles at {worst[1]} m/s ({rate.stack().max():.0%} fail)",
               "Failure rate by obstacle count and UAV speed")
    return fig


def crash_causes(cf: pd.DataFrame):
    rows = []
    for k, lab in CAUSES.items():
        applicable = cf[k].notna()
        fixed = (cf[k] == True).sum()  # noqa: E712
        rows.append((lab, fixed / len(cf), fixed, applicable.sum()))
    rows.append(("no single change prevents it (compound)", (cf["n_fixes"] == 0).mean(),
                 (cf["n_fixes"] == 0).sum(), len(cf)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(rows))
    for yi, (lab, share, fixed, app) in zip(y, rows):
        col = st.MUTED if lab.startswith("no single") else st.BLUE
        ax.barh(yi, share * 100, color=col, height=0.62)
        ax.text(share * 100 + 0.8, yi, f"{share:.0%}  ({fixed} of {len(cf)})", va="center", fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows], color=st.INK)
    ax.set_xlim(0, max(r[1] for r in rows) * 100 * 1.25)
    ax.set_xlabel("% of failures this single change prevents")
    ax.grid(axis="y", visible=False)
    top = max((r for r in rows if not r[0].startswith("no single")), key=lambda r: r[1])
    st.finding(fig, f"Most common measured cause: {top[0]} ({top[1]:.0%} of failures)",
               "Each failure re-run with one factor changed; a bar = share of failures that change prevents "
               "(a failure can have several causes)")
    return fig
