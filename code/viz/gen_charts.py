"""Charts for the generator evaluation (reports/02_generators)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import CAUSES
from scenarios.spec import Scenario
from viz import style as st
from viz.runs import thumbnail

# diffusion guidance levels are one model at ordered settings -> ordinal blue steps; cVAE = slot 2
SERIES = {"diffusion w=1": st.BLUE_RAMP[2], "diffusion w=2": st.BLUE_RAMP[4], "diffusion w=4": st.BLUE_RAMP[6],
          "cvae": st.ORANGE}


def steering(steer: pd.DataFrame, random_rate: float):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for name, col in SERIES.items():
        g = steer[steer.config == name].sort_values("difficulty")
        for ax, y in zip(axes, ("failure_rate", "validity")):
            ax.plot(g.difficulty, g[y] * 100, color=col, marker="o", ms=5)
            ax.text(g.difficulty.iloc[-1] + 0.02, g[y].iloc[-1] * 100, name, va="center", fontsize=8.5, color=col)
    axes[0].axhline(random_rate * 100, color=st.INK_2, lw=0.8)
    axes[0].text(0.08, random_rate * 100 + 0.8, f"random baseline {random_rate:.0%}", fontsize=8, color=st.INK_2)
    axes[0].set_title("Failure rate of the generated scenarios", fontsize=9.5)
    axes[0].set_ylabel("failure rate (%)")
    axes[1].set_title("Share of generated scenarios that pass the validity checks", fontsize=9.5)
    axes[1].set_ylabel("valid (%)")
    for ax in axes:
        ax.set_xlabel("requested difficulty")
        ax.set_xlim(0, 1.25)
    best = steer[steer.config == "diffusion w=4"].failure_rate.max()
    cv = steer[steer.config == "cvae"].failure_rate.max()
    fig.tight_layout(w_pad=3)
    st.finding(fig, f"Diffusion can be steered to {best:.0%} failures ({best / random_rate:.1f}x random); "
                    f"the cVAE only reaches {cv:.0%}",
               "Each point: 600 scenarios sampled at that difficulty, valid ones flown by RRT. "
               "w = classifier-free guidance strength (higher = obeys the condition harder)")
    return fig


def distributions(real: pd.DataFrame, gens: dict):
    cols = [("c_speed", "UAV cruise speed (m/s)"), ("c_n_obstacles", "number of obstacles"),
            ("c_dynamic_frac", "share of moving obstacles"), ("c_wind_mag", "mean wind (m/s)"),
            ("c_sense_range", "sensing range (m)"), ("noise_std", "sensor noise std (m)"),
            ("radius_mean", "mean obstacle radius (m)"), ("straight_dist", "start-goal distance (m)")]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        lo, hi = np.nanpercentile(np.concatenate([real[c]] + [g[c] for g in gens.values()]), [0.5, 99.5])
        bins = np.arange(-0.5, 6.5) if c == "c_n_obstacles" else np.linspace(lo, hi, 25)
        ax.hist(real[c], bins=bins, density=True, color=st.GRID, label="real (seed set)")
        for (name, g), col in zip(gens.items(), (st.BLUE, st.ORANGE)):
            ax.hist(g[c], bins=bins, density=True, histtype="step", lw=1.8, color=col, label=name)
        ax.set_title(lab, fontsize=9.5)
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(h_pad=2)
    st.finding(fig, "Both generators reproduce the scenario distribution; the cVAE smooths away its edges",
               "Grey = real seed scenarios; lines = valid unconditional samples from each generator (density)")
    return fig


def adherence(adh: pd.DataFrame):
    keys = [("speed", "UAV speed (m/s)"), ("n_obstacles", "number of obstacles"), ("wind", "mean wind (m/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (k, lab) in zip(axes, keys):
        sub = adh[adh.key == k]
        lo, hi = sub.requested.min(), sub.requested.max()
        ax.plot([lo, hi], [lo, hi], color=st.AXIS, lw=1)
        for (name, col), off in zip((("diffusion", st.BLUE), ("cvae", st.ORANGE)), (-0.08, 0.08)):
            g = sub[sub.model == name]
            span = (hi - lo) * off
            ax.errorbar(g.requested + span, g.achieved_mean, yerr=g.achieved_std, fmt="o", color=col, ms=6,
                        capsize=0, lw=1.2, label=name)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("requested")
    axes[0].set_ylabel("achieved (mean ± std)")
    axes[0].legend(loc="upper left")
    err = adh.assign(e=(adh.achieved_mean - adh.requested).abs() / adh.groupby("key").requested.transform("max"))
    by = err.groupby("model").e.mean()
    fig.tight_layout(w_pad=3)
    st.finding(fig, f"Both models follow explicit requests (mean error {by['diffusion']:.0%} diffusion, {by['cvae']:.0%} cVAE of range)",
               "Ask for a specific value, measure what the generated scenarios actually contain; grey diagonal = perfect")
    return fig


def novelty(dists: dict):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bins = np.linspace(0, np.percentile(np.concatenate(list(dists.values())), 99), 40)
    cols = {"fresh random scenarios": st.GRID, "diffusion": st.BLUE, "cvae": st.ORANGE}
    for name, d in dists.items():
        if name.startswith("fresh"):
            ax.hist(d, bins=bins, density=True, color=cols[name], label=f"{name} (reference)")
        else:
            ax.hist(d, bins=bins, density=True, histtype="step", lw=1.8, color=cols[name], label=name)
    ax.set_xlabel("distance to the nearest training scenario (normalised units)")
    ax.set_yticks([])
    ax.legend()
    ax.grid(False)
    med = {k: np.median(v) for k, v in dists.items()}
    st.finding(fig, f"Generated scenarios are new, not copies: median distance {med['diffusion']:.2f} (diffusion) "
                    f"vs {med['fresh random scenarios']:.2f} for brand-new random ones",
               "A copied training scenario would sit at distance 0")
    return fig


def cvae_tradeoff(sweep: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cv = sweep[sweep.model == "cvae"].sort_values("beta")
    ax.plot(cv.validity * 100, cv.w_distance, color=st.ORANGE, marker="o", ms=7)
    for _, r in cv.iterrows():
        ax.text(r.validity * 100 + 1, r.w_distance + 0.008, f"cVAE β={r.beta:g}", fontsize=8.5, color=st.ORANGE)
    d = sweep[sweep.model == "diffusion"].iloc[0]
    ax.plot(d.validity * 100, d.w_distance, "o", color=st.BLUE, ms=9)
    ax.text(d.validity * 100 - 1, d.w_distance + 0.012, "diffusion", fontsize=9, color=st.BLUE, ha="right")
    ax.set_xlabel("valid scenarios (%)  →  better")
    ax.set_ylabel("distance from real distribution\n(lower = more realistic)")
    ax.set_ylim(0, None)
    ax.set_xlim(30, 100)
    st.finding(fig, "No cVAE setting is both valid and realistic; diffusion is both",
               "cVAE trained at four KL weights β (same data, steps, conditioning). Best corner = bottom right")
    return fig


def gallery(vecs, outcomes, title, subtitle, ncol=6):
    n = len(vecs)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.45 * nrow))
    colors = {"success": st.GOOD, "near_failure": "#b07a00", "failure": st.CRITICAL}
    for ax, v, o in zip(axes.ravel(), vecs, outcomes):
        thumbnail(ax, Scenario.from_vector(v), o.replace("_", "-"), colors[o])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    st.finding(fig, title, subtitle)
    return fig


def cause_comparison(cfs: dict, title: str = "Failures found by each method have different causes"):
    labels = list(CAUSES.values()) + ["no single change (compound)"]
    keys = list(CAUSES.keys())
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(labels))
    h = 0.8 / len(cfs)
    for j, ((name, cf), col) in enumerate(zip(cfs.items(), (st.MUTED, st.BLUE, st.ORANGE))):
        vals = [(cf[k] == True).mean() for k in keys] + [(cf.n_fixes == 0).mean()]  # noqa: E712
        ax.barh(y + (j - (len(cfs) - 1) / 2) * h, np.array(vals) * 100, height=h * 0.9, color=col,
                label=f"{name} ({len(cf)} failures)")
    ax.set_yticks(y, labels, color=st.INK)
    ax.invert_yaxis()
    ax.set_xlabel("% of failures this single change prevents")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    st.finding(fig, title,
               "Each failure re-run with one factor changed (same as stage 1); a failure can have several causes")
    return fig
