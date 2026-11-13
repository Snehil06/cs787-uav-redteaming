"""Build reports/03_red_team_loop/ and reports/05_comparison/ from data/comparison/.

    python -m experiments.report_comparison
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.evidence import failure_type
from experiments.common import DATA_DIR
from loop.metrics import N_CELLS, distinct_mask, summarize, unavoidable_mask
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import simulate
from viz import compare_charts as cc
from viz import gen_charts
from viz.runs import run_figure
from viz.video import render_video

REPORTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
SRC = os.path.join(DATA_DIR, "comparison")


def load():
    dfs, cfs = {}, {}
    for p in sorted(glob.glob(os.path.join(SRC, "*_s*.parquet"))):
        m, s = re.match(r"(.+)_s(\d+)\.parquet", os.path.basename(p)).groups()
        df = pd.read_parquet(p)
        cf = pd.read_csv(p.replace(".parquet", "_cf.csv"), index_col=0)
        dfs.setdefault(m, {})[int(s)] = df
        cfs.setdefault(m, {})[int(s)] = cf
    return dfs, cfs


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def render(row, stem, title, sub, folder, video=True):
    s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float))
    tr = simulate(s, planner="rrt", seed=int(row["seed"]), record=True)
    tr.metrics["F"] = row["F"]
    save(run_figure(s, tr, title, sub), os.path.join(folder, stem + ".png"))
    if video:
        render_video(s, tr, os.path.join(folder, stem + ".mp4"), title, sub)
    return f"| `{stem}.mp4` / `.png` | {title} | {sub} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    dfs, cfs = load()
    methods = [m for m in cc.ORDER if m in dfs]
    seeds = sorted(set.intersection(*[set(dfs[m]) for m in methods]))

    rows, masks = [], {m: [] for m in methods}
    for m in methods:
        for s in seeds:
            r = summarize(dfs[m][s], cfs[m][s])
            rows.append({"method": m, "seed": s, **r})
            masks[m].append(distinct_mask(dfs[m][s]))
    summ = pd.DataFrame(rows)
    agg = summ.groupby("method").agg(["mean", "std"]).reindex(methods)

    # ================= stage 5: comparison
    base = os.path.join(REPORTS, "05_comparison")
    d_runs, d_res, d_why = (os.path.join(base, x) for x in ("1_runs", "2_results", "3_crash_reasons"))
    for d in (d_runs, d_res, d_why):
        os.makedirs(d, exist_ok=True)
    summ.to_csv(os.path.join(d_res, "per_seed.csv"), index=False)
    agg.round(3).to_csv(os.path.join(d_res, "summary_mean_std.csv"))
    save(cc.headline(summ), os.path.join(d_res, "01_headline.png"))
    save(cc.discovery_curves(masks), os.path.join(d_res, "02_discovery_curves.png"))
    save(cc.efficiency(summ), os.path.join(d_res, "03_efficiency.png"))
    save(cc.failure_maps({m: dfs[m][seeds[0]] for m in methods}), os.path.join(d_res, "04_failure_maps.png"))
    pooled_cf = {m: pd.concat(cfs[m].values()) for m in methods}
    save(cc.cause_heatmap(pooled_cf), os.path.join(d_why, "01_failure_types_by_method.png"))

    # failure types only our method found (pooled over seeds), with a video each
    sig = {m: set(pooled_cf[m]["fixed_by"].fillna("").replace("", "compound")) for m in methods}
    others = set().union(*[sig[m] for m in methods if m != "diffusion+feedback"])
    unique = sorted(sig.get("diffusion+feedback", set()) - others)
    pd.Series(unique, name="failure types found only by diffusion+feedback").to_csv(
        os.path.join(d_why, "types_only_ours.csv"), index=False)

    lines = []
    ours = dfs["diffusion+feedback"][seeds[0]]
    dm = masks["diffusion+feedback"][0]
    gal = ours[dm].head(18)
    save(gen_charts.gallery(gal[VECTOR_NAMES].to_numpy(float), gal.outcome.tolist(),
                            "First 18 distinct failures found by diffusion + feedback (seed 0)",
                            "Near-duplicates removed. Route = faint blue line; orange = obstacles (faded = appears "
                            "later; arrows = velocity); grey arrow = wind"), os.path.join(d_runs, "00_gallery_ours.png"))
    cf0 = cfs["diffusion+feedback"][seeds[0]]
    picked = 0
    for i, r in ours[dm & (ours.status == "collision")].iterrows():
        t = failure_type(cf0.loc[i].to_dict()) if i in cf0.index else "?"
        key = cf0.loc[i, "fixed_by"] if i in cf0.index and isinstance(cf0.loc[i, "fixed_by"], str) else "compound"
        if key in unique or (picked == 0 and key == "compound"):
            picked += 1
            lines.append(render(r, f"0{picked}_ours_{'unique' if key in unique else 'compound'}",
                                "Found by diffusion + feedback" + (" only" if key in unique else ""),
                                f"round {int(r['round']) + 1}; measured type: {t}", d_runs, not args.no_video))
        if picked >= 3:
            break

    def cell(m, k, fmt="{:.0f}"):
        mu, sd = agg.loc[m, (k, "mean")], agg.loc[m, (k, "std")]
        return "-" if pd.isna(mu) else (fmt.format(mu) + (f" ± {fmt.format(sd)}" if pd.notna(sd) else ""))

    table = "\n".join(
        f"| {cc.NAMES[m]} | {cell(m, 'failures')} | {cell(m, 'failure_rate', '{:.1%}')} | {cell(m, 'distinct_failures')} | "
        f"{cell(m, 'coverage_cells')} | {cell(m, 'types_in_first_200')} | {cell(m, 'failure_types')} | "
        f"{cell(m, 'flights_to_50th')} | "
        f"{cell(m, 'validity', '{:.0%}') if 'validity' in summ else '-'} |" for m in methods)
    budget = int(summ.flights.iloc[0])
    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Final comparison: which method red-teams RRT best?

Every method flew **{budget:,} scenarios per run**, repeated over **{len(seeds)} seeds** (mean ± std below).

- **distinct failures**: failures left after removing near-duplicates (model-space distance < 1.0; two
  independent random scenarios sit about 1.2 apart).
- **regions covered**: cells of a {N_CELLS}-cell grid over speed x obstacles x moving x wind x sensing that
  contain at least one failure.
- **failure types**: groups of failures with the same measured causes (counterfactual reruns). Counted among
  the first 200 failures (fair: does not reward simply finding more) and in total.

| method | failures | failure rate | distinct failures | regions covered | types in first 200 | types total | flights to 50th distinct | valid samples |
|---|---|---|---|---|---|---|---|---|
{table}

Failure types found **only** by diffusion + feedback (not by any other method, all seeds): **{len(unique)}**
(listed in `3_crash_reasons/types_only_ours.csv`).

## Read in order

1. **[1_runs/](1_runs/)**: `00_gallery_ours.png` (the first distinct failures our method found) and videos of
   failures it found (types unique to it if any exist, otherwise compound failures no single change prevents).

| file | what it shows | detail |
|---|---|---|
""" + "\n".join(lines) + """

2. **[2_results/](2_results/)**

| file | question it answers |
|---|---|
| `01_headline.png` | How many failures, distinct failures, regions and failure types does each method find? |
| `02_discovery_curves.png` | How does discovery grow as the flight budget is spent? |
| `03_efficiency.png` | How many flights does each method need to find 10 / 50 distinct failures? |
| `04_failure_maps.png` | Where in scenario space does each method's failures sit? |
| `per_seed.csv`, `summary_mean_std.csv` | All numbers. |

3. **[3_crash_reasons/](3_crash_reasons/)**: `01_failure_types_by_method.png`, the mix of measured failure types
   each method finds.
""")

    # ================= stage 3: red-team loop dynamics
    base3 = os.path.join(REPORTS, "03_red_team_loop")
    r_runs, r_res = os.path.join(base3, "1_runs"), os.path.join(base3, "2_results")
    for d in (r_runs, r_res):
        os.makedirs(d, exist_ok=True)
    loop_m = [m for m in ("diffusion+feedback", "diffusion", "cvae+feedback", "cvae") if m in dfs]
    save(cc.loop_dynamics({m: list(dfs[m].values()) for m in loop_m}), os.path.join(r_res, "01_failure_rate_by_round.png"))
    per_round = pd.concat([pd.concat(dfs[m].values()) for m in loop_m]).groupby(["method", "round"]).agg(
        failure_rate=("outcome", lambda o: (o == "failure").mean()), validity=("valid_frac", "mean"), flights=("F", "size"))
    per_round.round(3).to_csv(os.path.join(r_res, "per_round.csv"))
    late = ours[(ours["round"] == ours["round"].max()) & (ours.status == "collision") & dm
                & ~unavoidable_mask(ours) & (ours.t_end > 5)]
    l3 = [render(late.iloc[0], "01_last_round_crash", "Crash found in the final round of the loop",
                 "after 4 rounds of fine-tuning on earlier failures", r_runs, not args.no_video)] if len(late) else []
    pr = per_round["failure_rate"].unstack(0)
    with open(os.path.join(base3, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Red-team loop: does learning from found failures help?

Each run spends its budget over 5 rounds. With feedback, the generator is fine-tuned after every round on
the seed data plus the (up-weighted, de-duplicated) failures found so far; without feedback it keeps sampling
from the original model. Failure rate per round (mean over {len(seeds)} seeds):

| round | """ + " | ".join(cc.NAMES[m] for m in pr.columns) + " |\n|---|" + "---|" * len(pr.columns) + "\n" +
                "\n".join(f"| {int(r) + 1} | " + " | ".join(f"{v:.1%}" for v in pr.loc[r]) + " |" for r in pr.index) + """

## Read in order

1. **[1_runs/](1_runs/)**

| file | what it shows | detail |
|---|---|---|
""" + "\n".join(l3) + """

2. **[2_results/](2_results/)**: `01_failure_rate_by_round.png` and `per_round.csv` (failure rate, validity and
   flights per round).

Why the failures happen, and how the methods compare overall: see [../05_comparison/](../05_comparison/).
""")
    print(agg[[("failures", "mean"), ("distinct_failures", "mean"), ("coverage_cells", "mean"),
               ("failure_types", "mean"), ("flights_to_50th", "mean")]].round(1))
    print("reports written")


if __name__ == "__main__":
    main()
