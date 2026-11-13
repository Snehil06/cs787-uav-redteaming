"""Charts for the method comparison (stage 5) and the red-team loop dynamics (stage 3).

Six methods are too many for six hues, so charts use emphasis: the proposed method in blue,
its no-feedback ablation in orange, every baseline in grey with a direct label.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz import style as st

ORDER = ["random", "cmaes", "cvae", "cvae+feedback", "diffusion", "diffusion+feedback"]
NAMES = {"random": "Random", "cmaes": "CMA-ES search", "cvae": "cVAE", "cvae+feedback": "cVAE + feedback",
         "diffusion": "Diffusion", "diffusion+feedback": "Diffusion + feedback (ours)"}
COLOR = {m: st.MUTED for m in ORDER} | {"diffusion": st.ORANGE, "diffusion+feedback": st.BLUE}
SHORT = {"no_wind": "wind", "full_sensing": "late detection", "no_noise": "noise", "no_popups": "pop-up",
         "slower": "speed", "reseed": "RRT luck"}


def headline(summ: pd.DataFrame):
    """Mean ± std over seeds for the four discovery metrics."""
    metrics = [("failures", "failures found"), ("distinct_failures", "distinct failures\n(near-duplicates removed)"),
               ("coverage_cells", "regions of scenario space\nwith a failure"),
               ("types_in_first_200", "failure types among\nthe first 200 failures")]
    g = summ.groupby("method")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, (m, lab) in zip(axes, metrics):
        mean, sd = g[m].mean().reindex(ORDER), g[m].std().reindex(ORDER).fillna(0)
        x = np.arange(len(ORDER))
        ax.bar(x, mean, yerr=sd, color=[COLOR[o] for o in ORDER], width=0.7, error_kw=dict(lw=1, ecolor=st.INK_2))
        for xi, v in zip(x, mean):
            ax.text(xi, v + sd.max() * 0.6 + mean.max() * 0.02, f"{v:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, [NAMES[o] for o in ORDER], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    ours, best_base = g["distinct_failures"].mean()["diffusion+feedback"], g["distinct_failures"].mean().drop(
        ["diffusion+feedback", "diffusion"]).max()
    st.finding(fig, f"Diffusion + feedback finds {ours / best_base:.1f}x the distinct failures of the best baseline",
               "Same budget of flights per method; bars = mean over seeds, whiskers = std")
    return fig


def discovery_curves(runs: dict):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for m in ORDER:
        curves = [np.cumsum(mask) for mask in runs[m]]
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], axis=0)
        lw = 2.4 if m.startswith("diffusion") else 1.4
        ax.plot(np.arange(1, n + 1), mean, color=COLOR[m], lw=lw)
        ax.text(n * 1.01, mean[-1], NAMES[m], va="center", fontsize=8.5,
                color=COLOR[m] if COLOR[m] != st.MUTED else st.INK_2)
    ax.set_xlim(0, n * 1.3)
    ax.set_xlabel("flights spent")
    ax.set_ylabel("distinct failures found")
    st.finding(fig, "The generative methods keep finding new failures; search plateaus",
               "Cumulative distinct failures as the flight budget is spent (mean over seeds)")
    return fig


def efficiency(summ: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    g = summ.groupby("method")
    for ax, (k, lab) in zip(axes, (("flights_to_10th", "10th"), ("flights_to_50th", "50th"))):
        mean = g[k].mean().reindex(ORDER)
        y = np.arange(len(ORDER))
        ax.barh(y, mean, color=[COLOR[o] for o in ORDER], height=0.65)
        for yi, v in zip(y, mean):
            ax.text((0 if np.isnan(v) else v) + 15, yi, "not reached" if np.isnan(v) else f"{v:.0f}", va="center",
                    fontsize=8.5)
        ax.set_yticks(y, [NAMES[o] for o in ORDER])
        ax.invert_yaxis()
        ax.set_title(f"flights needed to find the {lab} distinct failure", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=2)
    st.finding(fig, "Fewer flights to the same number of distinct failures = more efficient red-teaming",
               "Mean over seeds; lower is better")
    return fig


def failure_maps(dfs: dict):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes.ravel(), ORDER):
        d = dfs[m]
        f = d[d.outcome == "failure"]
        ax.scatter(d.c_wind_mag, d.c_speed, s=3, color=st.GRID, lw=0)
        ax.scatter(f.c_wind_mag + rng.normal(0, 0.05, len(f)), f.c_speed, s=7, color=COLOR[m] if COLOR[m] != st.MUTED
                   else st.INK_2, lw=0, alpha=0.7)
        ax.set_title(f"{NAMES[m]}: {len(f)} failures", fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("mean wind (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("UAV speed (m/s)")
    fig.tight_layout(h_pad=2)
    st.finding(fig, "Where each method's failures sit in wind x speed space",
               "Grey = every scenario the method flew; dots = its failures (seed 0)")
    return fig


def cause_heatmap(cfs: dict):
    rows = {}
    for m in ORDER:
        sig = cfs[m]["fixed_by"].fillna("").replace("", "compound")
        rows[m] = sig.value_counts(normalize=True)
    tab = pd.DataFrame(rows).fillna(0)
    tab = tab.loc[tab.max(1).sort_values(ascending=False).index[:12]]
    labels = [" + ".join(SHORT[k] for k in s.split(",")) if s != "compound" else "compound (no single fix)"
              for s in tab.index]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", st.BLUE_RAMP)
    im = ax.imshow(tab.values * 100, cmap=cmap, aspect="auto", vmin=0)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color=st.SURFACE if v > im.norm.vmax * 0.55 else st.INK)
    ax.set_xticks(range(len(ORDER)), [NAMES[m] for m in ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("% of that method's failures")
    cb.outline.set_visible(False)
    st.finding(fig, "CMA-ES concentrates on one failure type; the generative methods spread across many",
               "Failure type = the set of single changes that prevent it (counterfactual reruns), all seeds pooled. "
               "Labels: wind, speed, late detection, noise, pop-up obstacle, RRT luck (re-seeding fixes it)")
    return fig


def loop_dynamics(dfs_all: dict):
    """Per-round failure rate and new distinct failures, feedback vs no feedback."""
    pairs = [("diffusion+feedback", "diffusion"), ("cvae+feedback", "cvae")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for ax, (fb, nofb) in zip(axes, pairs):
        for m, col in ((fb, st.BLUE), (nofb, st.ORANGE)):
            per = pd.concat(dfs_all[m]).groupby(["seed_run", "round"]).apply(
                lambda d: (d.outcome == "failure").mean(), include_groups=False).groupby("round")
            mean, sd = per.mean() * 100, per.std().fillna(0) * 100
            ax.plot(mean.index + 1, mean, color=col, marker="o")
            ax.fill_between(mean.index + 1, mean - sd, mean + sd, color=col, alpha=0.12, lw=0)
            ax.text(mean.index[-1] + 1.1, mean.iloc[-1], NAMES[m], va="center", fontsize=8.5, color=col)
        ax.set_xlim(0.7, 6.6)
        ax.set_xlabel("round (each = 1/5 of the flight budget)")
        ax.set_title(NAMES[fb].replace(" (ours)", "") + " vs no feedback", fontsize=9.5)
    axes[0].set_ylabel("failure rate in that round (%)")
    fig.tight_layout(w_pad=3)
    st.finding(fig, "Fine-tuning on found failures raises the failure rate round after round",
               "Round 1 is identical (no feedback yet); band = std over seeds")
    return fig
