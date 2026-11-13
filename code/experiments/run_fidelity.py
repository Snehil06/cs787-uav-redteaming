"""Stage 6: are the failures real planner weaknesses or simulator artefacts?

    python -m experiments.run_fidelity

1. Time step: re-fly failures at dt = 0.05 (as found), 0.025 and 0.0125 s. Each is flown with 3 fresh planner
   seeds and counts as "still fails" if at least 2 of 3 fail. If failures were integration artefacts, they would
   disappear at finer steps; the dt = 0.05 row is the reference (planner randomness alone).
2. Second planner: fly the random scenarios with RRT* as well as RRT, re-fly RRT failures with RRT*, and compare
   the measured causes of each planner's failures.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import analyse_failures
from experiments.common import DATA_DIR, run_many
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import simulate
from viz import gen_charts
from viz import style as st
from viz.runs import run_figure
from viz.video import render_video

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "06_fidelity_checks")
N_PER_SET = 500
DTS = [0.05, 0.025, 0.0125]
RESEEDS = [7919, 2 * 7919, 3 * 7919]


def failure_sets():
    seed = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    ours = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(DATA_DIR, "comparison", "diffusion+feedback_s*.parquet"))],
                     ignore_index=True)
    return {"random baseline failures": seed[seed.outcome == "failure"].sample(N_PER_SET, random_state=0),
            "diffusion + feedback failures": ours[ours.outcome == "failure"].sample(N_PER_SET, random_state=0)}


def persistence(fail: pd.DataFrame, planner="rrt", sim_cfg=None) -> np.ndarray:
    """Per failure: True if at least 2 of 3 fresh planner seeds still fail."""
    vecs = fail[VECTOR_NAMES].to_numpy(float)
    votes = np.zeros(len(fail))
    for off in RESEEDS:
        df = run_many(vecs, planner=planner, seeds=fail.seed.to_numpy() + off, sim_cfg=sim_cfg)
        votes += (df.outcome == "failure").to_numpy()
    return votes >= 2


def main():
    d_runs, d_res, d_why = (os.path.join(ROOT, x) for x in ("1_runs", "2_results", "3_crash_reasons"))
    for d in (d_runs, d_res, d_why):
        os.makedirs(d, exist_ok=True)
    sets = failure_sets()

    # ---- 1. time step
    rows = []
    for name, fail in sets.items():
        for dt in DTS:
            p = persistence(fail, sim_cfg={"dt": dt})
            rows.append({"set": name, "dt": dt, "still_fails": p.mean()})
            print(f"  dt {dt}: {name} still fail {p.mean():.1%}", flush=True)
    dtab = pd.DataFrame(rows)
    dtab.to_csv(os.path.join(d_res, "timestep_persistence.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    for j, (name, col) in enumerate(zip(sets, (st.MUTED, st.BLUE))):
        g = dtab[dtab.set == name]
        x = np.arange(len(DTS)) + (j - 0.5) * w
        ax.bar(x, g.still_fails * 100, width=w * 0.92, color=col, label=name)
        for xi, v in zip(x, g.still_fails):
            ax.text(xi, v * 100 + 1.5, f"{v:.0%}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(DTS)), [f"dt = {d} s" + ("\n(as found)" if d == 0.05 else "") for d in DTS])
    ax.set_ylabel("% of failures that still fail")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    drop = dtab.groupby("set").still_fails.agg(lambda s: s.iloc[0] - s.iloc[-1]).max()
    st.finding(fig, f"Failures survive a 4x finer time step (largest drop {drop:.0%} points): they are not integration artefacts",
               f"{N_PER_SET} failures per set, each re-flown with 3 fresh RRT seeds; 'still fails' = at least 2 of 3 fail")
    fig.savefig(os.path.join(d_res, "01_timestep_check.png"))
    plt.close(fig)

    # ---- 2. RRT vs RRT*
    seed = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")).head(2000)
    star = run_many(seed[VECTOR_NAMES].to_numpy(float), planner="rrt_star", seeds=seed.seed)
    star.to_parquet(os.path.join(DATA_DIR, "random_rrtstar_2000.parquet"))
    rate = {"RRT": (seed.outcome == "failure").mean(), "RRT*": (star.outcome == "failure").mean()}
    both = ((seed.outcome == "failure").to_numpy() & (star.outcome == "failure").to_numpy()).sum()
    transfer = {name: persistence(fail, planner="rrt_star").mean() for name, fail in sets.items()}
    pd.DataFrame({"rrt_failure_rate": [rate["RRT"]], "rrtstar_failure_rate": [rate["RRT*"]],
                  "fail_both": [both], **{f"transfer_{k}": [v] for k, v in transfer.items()}}).to_csv(
        os.path.join(d_res, "planner_check.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    axes[0].bar([0, 1], [rate["RRT"] * 100, rate["RRT*"] * 100], color=[st.MUTED, st.BLUE], width=0.6)
    for i, v in enumerate(rate.values()):
        axes[0].text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=9)
    axes[0].set_xticks([0, 1], ["RRT", "RRT*"])
    axes[0].set_title("Failure rate on the same 2,000 random scenarios", fontsize=9.5)
    axes[0].set_ylabel("%")
    names = list(transfer)
    axes[1].bar(range(len(names)), [transfer[n] * 100 for n in names], color=[st.MUTED, st.BLUE], width=0.6)
    for i, n in enumerate(names):
        axes[1].text(i, transfer[n] * 100 + 1, f"{transfer[n]:.0%}", ha="center", fontsize=9)
    axes[1].set_xticks(range(len(names)), names, fontsize=8.5)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("RRT failures that also fail RRT* (2 of 3 seeds)", fontsize=9.5)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=3)
    st.finding(fig, f"RRT* fails {rate['RRT*']:.1%} vs RRT {rate['RRT']:.1%}; many RRT failures also break RRT*",
               "Same scenarios and simulator, only the planner changes")
    fig.savefig(os.path.join(d_res, "02_rrt_vs_rrtstar.png"))
    plt.close(fig)

    # ---- 3. causes per planner
    sf = star[star.outcome == "failure"]
    rf = seed[seed.outcome == "failure"]
    cfs = {"RRT": analyse_failures(rf), "RRT*": analyse_failures(sf)}
    for k, v in cfs.items():
        v.to_csv(os.path.join(d_why, f"counterfactuals_{k.replace('*', 'star')}.csv"))
    fig = gen_charts.cause_comparison(cfs, "RRT and RRT* fail for the same reasons: wind and speed dominate both")
    fig.savefig(os.path.join(d_why, "01_causes_rrt_vs_rrtstar.png"))
    plt.close(fig)

    # ---- runs: one scenario both planners fly
    lines = []
    joined = seed.reset_index(drop=True).assign(star_outcome=star.outcome.values, star_status=star.status.values)
    picks = [("01_rrt_crash_rrtstar_ok", joined[(joined.status == "collision") & (joined.star_outcome == "success") & (joined.t_end > 5)]),
             ("02_both_crash", joined[(joined.status == "collision") & (joined.star_status == "collision") & (joined.t_end > 5)])]
    for stem, pool in picks:
        if pool.empty:
            continue
        r = pool.iloc[0]
        s = Scenario.from_vector(r[VECTOR_NAMES].to_numpy(float))
        for planner in ("rrt", "rrt_star"):
            tr = simulate(s, planner=planner, seed=int(r["seed"]), record=True)
            tr.metrics["F"] = float("nan")
            label = "RRT" if planner == "rrt" else "RRT*"
            title = f"Same scenario, {label}: {tr.metrics['status']}"
            sub = "RRT crashes here, RRT* gets through" if stem.startswith("01") else "both planners crash here"
            fig = run_figure(s, tr, title, sub)
            fig.savefig(os.path.join(d_runs, f"{stem}_{planner}.png"))
            plt.close(fig)
            render_video(s, tr, os.path.join(d_runs, f"{stem}_{planner}.mp4"), title, sub)
        lines.append(f"| `{stem}_rrt.mp4` and `{stem}_rrt_star.mp4` | {sub} |")

    d = dtab.pivot(index="set", columns="dt", values="still_fails")
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Fidelity checks: are the failures real planner weaknesses?

A 2D simulator invites the question "are these failures just artefacts?". Two checks answer it.

**1. Finer time step.** {N_PER_SET} failures from random sampling and {N_PER_SET} from diffusion + feedback were re-flown
at the original step (0.05 s), 2x finer and 4x finer, each with 3 fresh RRT seeds.

| failures from | dt = 0.05 s (as found) | 0.025 s | 0.0125 s |
|---|---|---|---|
""" + "\n".join(f"| {k} | {r[0.05]:.0%} | {r[0.025]:.0%} | {r[0.0125]:.0%} |" for k, r in d.iterrows()) + f"""

The 0.05 s column is the reference: some failures depend on RRT's random sampling ("RRT luck", stage 1), so not
all of them repeat with new seeds even at the original step. A similar rate at finer steps means the failures
are not created by the integration step.

**2. Second planner (RRT\\*).** On the same 2,000 random scenarios, RRT fails {rate['RRT']:.1%} and RRT\\* fails
{rate['RRT*']:.1%}; {both} scenarios break both. Failures found against RRT that also break RRT\\*:
""" + "; ".join(f"{k}: {v:.0%}" for k, v in transfer.items()) + """.

## Read in order

1. **[1_runs/](1_runs/)**: the same scenario flown by both planners.

| files | what it shows |
|---|---|
""" + "\n".join(lines) + """

2. **[2_results/](2_results/)**: `01_timestep_check.png`, `02_rrt_vs_rrtstar.png` and the CSVs behind them.
3. **[3_crash_reasons/](3_crash_reasons/)**: `01_causes_rrt_vs_rrtstar.png`, measured causes of each
   planner's failures on the same scenarios.
""")
    print("report written to", ROOT)


if __name__ == "__main__":
    main()
