"""Stage 7: close the loop. Harden the system around RRT, check the fixes, then attack it again.

    python -m experiments.run_hardening

1. Regression: replay every failure found so far (all stages, all methods) against the hardened system,
   same scenario and planner seed. Report how many are fixed, by measured failure type.
2. Side effects: fly the same random scenarios on both systems (failure rate, flight time, near-misses).
3. Re-attack: run random sampling and diffusion + feedback against the hardened system, then measure the
   causes of the failures that remain.
The hardening (sim.simulator.HARDENED) changes only the code around RRT; the planner is the same.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import CAUSES, analyse_failures
from analyst.evidence import failure_type
from experiments.common import DATA_DIR, run_many
from loop.metrics import summarize
from scenarios.procedural import sample_scenario
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import HARDENED, SimConfig, simulate
from viz import gen_charts
from viz import style as st
from viz.runs import run_figure
from viz.video import render_video

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "07_harden_and_reattack")
OUT = os.path.join(DATA_DIR, "hardening")
BUDGET, SEEDS = 2000, [0, 1]
FIXES = {
    "wind feed-forward": "controller cancels the estimated wind force (targets: wind / gusts)",
    "speed governor": "never faster than the UAV can stop within its sensing range, and slower near sensed "
                      "obstacles (targets: speed, late detection)",
    "best of 3 plans": "RRT runs 3 times per replan and the shortest path is kept (targets: unlucky randomness)",
    "wider moving-obstacle footprint": "2 s of obstacle motion swept into its planning footprint instead of 1 s",
}


def library() -> pd.DataFrame:
    """Every failure found so far, with its measured type."""
    parts = []
    seed = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    cf = pd.read_csv(os.path.join(os.path.dirname(ROOT), "01_random_baseline", "3_crash_reasons", "counterfactuals.csv"),
                     index_col=0)
    parts.append(seed.loc[cf.index].assign(source="random baseline (stage 1)", fixed_by=cf.fixed_by))
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "comparison", "*_s*.parquet"))):
        df = pd.read_parquet(p)
        cfp = pd.read_csv(p.replace(".parquet", "_cf.csv"), index_col=0)
        parts.append(df.loc[cfp.index].assign(source=df.method.iloc[0], fixed_by=cfp.fixed_by))
    lib = pd.concat(parts, ignore_index=True)
    lib["type"] = [failure_type({"fixed_by": f}) for f in lib.fixed_by]
    return lib


def run_method(method: str, seed: int) -> pd.DataFrame:
    if method == "random":
        rng = np.random.default_rng(5000 + seed)
        vecs = [sample_scenario(rng).to_vector() for _ in range(BUDGET)]
        return run_many(vecs, seeds=seed * 1_000_000 + np.arange(BUDGET), extras=[{"method": "random"}] * BUDGET,
                        sim_cfg=HARDENED)
    from experiments.train_generators import MODEL_DIR
    from generators.diffusion import DiffusionGenerator
    from loop.redteam import red_team
    gen = DiffusionGenerator(seed=seed).load(os.path.join(MODEL_DIR, "diffusion.pt"))
    seed_df = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    return red_team(gen, seed_df[VECTOR_NAMES].to_numpy(float), seed_df.F.to_numpy(float), budget=BUDGET, rounds=5,
                    feedback=True, sim_seed=seed, label="diffusion+feedback", sim_cfg=HARDENED)


def main():
    d_runs, d_res, d_why = (os.path.join(ROOT, x) for x in ("1_runs", "2_results", "3_crash_reasons"))
    for d in (d_runs, d_res, d_why, OUT):
        os.makedirs(d, exist_ok=True)

    # ---- 1. regression on the edge-case library
    lib = library()
    replay = run_many(lib[VECTOR_NAMES].to_numpy(float), seeds=lib.seed.to_numpy(), sim_cfg=HARDENED)
    lib["fixed"] = (replay.outcome != "failure").to_numpy()
    lib.to_parquet(os.path.join(OUT, "library_replay.parquet"))
    by_type = lib.groupby("type").agg(cases=("fixed", "size"), fixed=("fixed", "mean")).sort_values("cases", ascending=False)
    by_type.to_csv(os.path.join(d_res, "regression_by_type.csv"))
    by_src = lib.groupby("source").fixed.mean()
    print(f"  regression: {lib.fixed.mean():.1%} of {len(lib)} past failures fixed", flush=True)

    top = by_type.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(top))
    ax.barh(y, top.fixed * 100, color=st.BLUE, height=0.65)
    for yi, (n, f_) in enumerate(zip(top.cases, top.fixed)):
        ax.text(f_ * 100 + 1, yi, f"{f_:.0%}  (n={n})", va="center", fontsize=8.5)
    ax.set_yticks(y, [t if len(t) < 60 else t[:57] + "..." for t in top.index], fontsize=8.5, color=st.INK)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of these past failures that the hardened system now survives")
    ax.grid(axis="y", visible=False)
    st.finding(fig, f"The hardened system survives {lib.fixed.mean():.0%} of the {len(lib):,} failures found so far",
               "Every past failure replayed with the same scenario and planner seed; 12 most common measured types")
    fig.savefig(os.path.join(d_res, "01_regression_by_failure_type.png"))
    plt.close(fig)

    # ---- 2. side effects on ordinary flights
    seed = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")).head(3000)
    hard = run_many(seed[VECTOR_NAMES].to_numpy(float), seeds=seed.seed, sim_cfg=HARDENED)
    side = pd.DataFrame({
        "original": [(seed.outcome == "failure").mean(), (seed.outcome == "near_failure").mean(),
                     seed[seed.status == "goal"].t_end.mean(), seed.plan_time_total.mean()],
        "hardened": [(hard.outcome == "failure").mean(), (hard.outcome == "near_failure").mean(),
                     hard[hard.status == "goal"].t_end.mean(), hard.plan_time_total.mean()]},
        index=["failure rate", "near-failure rate", "mean flight time to goal (s)", "planning time per flight (s)"])
    side.to_csv(os.path.join(d_res, "side_effects.csv"))
    broke = int(((seed.outcome != "failure").to_numpy() & (hard.outcome == "failure").to_numpy()).sum())

    # ---- 3. re-attack
    runs = {}
    for m in ("random", "diffusion+feedback"):
        for s in SEEDS:
            p = os.path.join(OUT, f"{m}_s{s}.parquet")
            if not os.path.exists(p):
                df = run_method(m, s)
                df["seed_run"] = s
                df.to_parquet(p)
            runs.setdefault(m, []).append(pd.read_parquet(p))
    before = {}
    for m in runs:
        before[m] = [pd.read_parquet(os.path.join(DATA_DIR, "comparison", f"{m}_s{s}.parquet")) for s in SEEDS]
    rows = []
    for label, src in (("original system", before), ("hardened system", runs)):
        for m, dfs in src.items():
            for s, df in zip(SEEDS, dfs):
                rows.append({"system": label, "method": m, "seed": s, **summarize(df)})
    att = pd.DataFrame(rows)
    att.to_csv(os.path.join(d_res, "reattack_per_seed.csv"), index=False)
    g = att.groupby(["system", "method"])[["failure_rate", "distinct_failures"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, k, lab in zip(axes, ("failure_rate", "distinct_failures"), ("failure rate (%)", "distinct failures")):
        for j, (m, col) in enumerate((("random", st.MUTED), ("diffusion+feedback", st.BLUE))):
            vals = [g.loc[(sy, m), k] * (100 if k == "failure_rate" else 1) for sy in ("original system", "hardened system")]
            x = np.arange(2) + (j - 0.5) * 0.38
            ax.bar(x, vals, width=0.35, color=col, label="random" if m == "random" else "diffusion + feedback")
            for xi, v in zip(x, vals):
                ax.text(xi, v * 1.02 + 0.3, f"{v:.0f}" if k != "failure_rate" else f"{v:.1f}%", ha="center", fontsize=8.5)
        ax.set_xticks([0, 1], ["original system", "hardened system"])
        ax.set_title(lab + f" per {BUDGET:,} flights", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    fig.tight_layout(w_pad=3)
    ratio = g.loc[("hardened system", "diffusion+feedback"), "distinct_failures"] / max(1, g.loc[("hardened system", "random"), "distinct_failures"])
    st.finding(fig, f"After hardening, random sampling barely finds failures; the red team still finds {ratio:.1f}x more",
               f"Same flight budget, mean over seeds {SEEDS}. The red team starts from the same pre-trained diffusion model")
    fig.savefig(os.path.join(d_res, "02_reattack.png"))
    plt.close(fig)

    # remaining causes: measured on the hardened system
    new_fail = pd.concat(runs["diffusion+feedback"], ignore_index=True)
    new_fail = new_fail[new_fail.outcome == "failure"]
    cf_after = analyse_failures(new_fail, sim_cfg=HARDENED)
    cf_after.to_csv(os.path.join(d_why, "counterfactuals_hardened.csv"))
    cf_before = pd.concat([pd.read_csv(os.path.join(DATA_DIR, "comparison", f"diffusion+feedback_s{s}_cf.csv"), index_col=0)
                           for s in SEEDS])
    fig = gen_charts.cause_comparison({"before hardening": cf_before, "after hardening": cf_after},
                                      "What still breaks the planner after hardening")
    fig.savefig(os.path.join(d_why, "01_causes_before_after.png"))
    plt.close(fig)

    # ---- runs: a fixed failure (before / after) and a new failure on the hardened system
    lines = []
    fixed = lib[lib.fixed & (lib.status == "collision") & (lib.t_end > 5)]
    if len(fixed):
        r = fixed.iloc[0]
        s = Scenario.from_vector(r[VECTOR_NAMES].to_numpy(float))
        for tag, cfg in (("before", None), ("after", HARDENED)):
            tr = simulate(s, planner="rrt", seed=int(r["seed"]), cfg=SimConfig(**(cfg or {})), record=True)
            tr.metrics["F"] = float("nan")
            title = f"{'Original' if tag == 'before' else 'Hardened'} system: {tr.metrics['status']}"
            sub = f"past failure ({r['type']}) replayed with the same scenario and planner seed"
            fig = run_figure(s, tr, title, sub)
            fig.savefig(os.path.join(d_runs, f"01_fixed_{tag}.png"))
            plt.close(fig)
            render_video(s, tr, os.path.join(d_runs, f"01_fixed_{tag}.mp4"), title, sub)
        lines.append(f"| `01_fixed_before.mp4` / `01_fixed_after.mp4` | the same past failure on the original and the hardened system |")
    nf = new_fail[(new_fail.status == "collision") & (new_fail.t_end > 4)]
    if len(nf):
        r = nf.iloc[0]
        s = Scenario.from_vector(r[VECTOR_NAMES].to_numpy(float))
        tr = simulate(s, planner="rrt", seed=int(r["seed"]), cfg=SimConfig(**HARDENED), record=True)
        tr.metrics["F"] = r["F"]
        t = failure_type(cf_after.loc[r.name].to_dict()) if r.name in cf_after.index else "?"
        fig = run_figure(s, tr, "New failure found against the hardened system", f"measured type: {t}")
        fig.savefig(os.path.join(d_runs, "02_new_failure_hardened.png"))
        plt.close(fig)
        render_video(s, tr, os.path.join(d_runs, "02_new_failure_hardened.mp4"),
                     "New failure found against the hardened system", f"measured type: {t}")
        lines.append(f"| `02_new_failure_hardened.mp4` | a weakness the red team found after hardening ({t}) |")

    rem = {CAUSES[k]: float((cf_after[k] == True).mean()) for k in CAUSES}  # noqa: E712
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Harden and re-attack: closing the loop on the planner

The red team's measured crash causes (stages 1 and 5) pointed at four fixes. All are changes around RRT;
the PythonRobotics RRT code is unchanged.

| fix | what it does |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in FIXES.items()) + f"""

## Results

**1. Regression.** Replaying all {len(lib):,} failures found so far (same scenario, same planner seed), the hardened
system survives **{lib.fixed.mean():.0%}** of them. By where they were found:
""" + "; ".join(f"{k}: {v:.0%}" for k, v in by_src.items()) + f""".

**2. Side effects** on {len(seed):,} ordinary random scenarios (original → hardened):

| | original | hardened |
|---|---|---|
""" + "\n".join(f"| {i} | {r.original:.3g} | {r.hardened:.3g} |" for i, r in side.iterrows()) + f"""

{broke} scenarios that the original system flew safely now fail on the hardened one (a regression cost).

**3. Re-attack** ({BUDGET:,} flights per run, seeds {SEEDS}):

| | random: failure rate | random: distinct failures | diffusion + feedback: failure rate | diffusion + feedback: distinct failures |
|---|---|---|---|---|
""" + "\n".join(f"| {sy} | {g.loc[(sy, 'random'), 'failure_rate']:.1%} | {g.loc[(sy, 'random'), 'distinct_failures']:.0f} | "
                f"{g.loc[(sy, 'diffusion+feedback'), 'failure_rate']:.1%} | {g.loc[(sy, 'diffusion+feedback'), 'distinct_failures']:.0f} |"
                for sy in ("original system", "hardened system")) + """

Measured causes of the failures that remain after hardening (share a single change prevents):
""" + "; ".join(f"{k}: {v:.0%}" for k, v in sorted(rem.items(), key=lambda kv: -kv[1])) + """.

## Read in order

1. **[1_runs/](1_runs/)**

| files | what it shows |
|---|---|
""" + "\n".join(lines) + """

2. **[2_results/](2_results/)**: `01_regression_by_failure_type.png`, `02_reattack.png`, and the CSVs
   (`regression_by_type.csv`, `side_effects.csv`, `reattack_per_seed.csv`).
3. **[3_crash_reasons/](3_crash_reasons/)**: `01_causes_before_after.png`, what still breaks the planner.
""")
    print("report written to", ROOT)


if __name__ == "__main__":
    main()
