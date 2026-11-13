"""Build reports/01_random_baseline/: 1_runs (videos), 2_results (charts), 3_crash_reasons.

    python -m experiments.report_baseline --data seed_rrt.parquet
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import CAUSES, DESCRIPTIONS, analyse_failures
from experiments.common import DATA_DIR
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import simulate
from viz import charts
from viz.runs import run_figure
from viz.video import render_video

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _replay(row):
    s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float))
    tr = simulate(s, planner=row["planner"], seed=int(row["seed"]), record=True)
    tr.metrics["F"] = row["F"]
    return s, tr


def pick_examples(df: pd.DataFrame, cf: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """(file stem, title, subtitle, row) for the runs worth watching."""
    long = df["t_end"] >= 6
    out = []
    ok = df[(df.outcome == "success") & (df.c_n_obstacles >= 3) & (df.c_dynamic_frac > 0) & long]
    r = ok.sort_values("n_replans", ascending=False).iloc[0]
    out.append(("01_success", "Safe flight through moving obstacles",
                f"{int(r.c_n_obstacles)} obstacles, {r.n_replans} replans, closest pass {r.min_clearance:.1f} m", r))
    nf = df[(df.outcome == "near_failure") & long]
    r = nf.sort_values("min_clearance").iloc[0]
    out.append(("02_near_miss", "Near-miss: goal reached, but only just",
                f"closest pass {r.min_clearance:.2f} m (near-miss zone is < 0.5 m)", r))

    crashes = df.loc[cf.index]
    counts = {k: int((cf[k] == True).sum()) for k in CAUSES}  # noqa: E712
    n = 3
    for k in sorted(counts, key=counts.get, reverse=True):
        only = cf[(cf[k] == True) & (cf.n_fixes == 1)].index  # noqa: E712
        pool = crashes.loc[only] if len(only) else crashes.loc[cf[cf[k] == True].index]  # noqa: E712
        pool = pool[(pool.status == "collision") & (pool.t_end >= 4)]
        if pool.empty:
            continue
        r = pool.iloc[0]
        sole = "the only single change that does" if r.name in only else "other changes also help"
        out.append((f"{n:02d}_crash_{k}", f"Crash cause: {CAUSES[k]}",
                    f"Re-run with {DESCRIPTIONS[k]} → no crash ({sole})", r))
        n += 1
        if n > 6:
            break
    comp = crashes.loc[cf[cf.n_fixes == 0].index]
    comp = comp[(comp.status == "collision") & (comp.t_end >= 4)]
    if len(comp):
        out.append((f"{n:02d}_crash_compound", "Compound crash: no single change prevents it",
                    "several factors combine; removing any one alone still crashes", comp.iloc[0]))
        n += 1
    for status, title in (("out_of_bounds", "Blown out of the geofence"), ("timeout", "Timed out: never reached goal")):
        p = df[df.status == status]
        if len(p):
            r = p.iloc[0]
            out.append((f"{n:02d}_{status}", title, f"wind {r.c_wind_mag:.1f} m/s, {int(r.c_n_obstacles)} obstacles", r))
            n += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seed_rrt.parquet")
    ap.add_argument("--out", default="01_random_baseline")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    base = os.path.join(ROOT, args.out)
    d_runs, d_res, d_why = (os.path.join(base, x) for x in ("1_runs", "2_results", "3_crash_reasons"))
    for d in (d_runs, d_res, d_why):
        os.makedirs(d, exist_ok=True)

    df = charts.add_derived(pd.read_parquet(os.path.join(DATA_DIR, args.data)))
    n, fail = len(df), df["failed"].mean()

    # ---- 2. results
    figs = [("01_outcomes.png", charts.outcomes(df)), ("02_scenario_mix.png", charts.scenario_mix(df)),
            ("03_failure_rate_by_factor.png", charts.failure_by_factor(df)),
            ("04_failure_map_speed_vs_obstacles.png", charts.failure_map(df))]
    for name, fig in figs:
        _save(fig, os.path.join(d_res, name))
    summary = pd.DataFrame({
        "runs": [n], "failure_rate": [fail], "near_failure_rate": [(df.outcome == "near_failure").mean()],
        "collision_rate": [(df.status == "collision").mean()], "out_of_bounds_rate": [(df.status == "out_of_bounds").mean()],
        "timeout_rate": [(df.status == "timeout").mean()], "mean_F": [df.F.mean()],
    })
    summary.to_csv(os.path.join(d_res, "summary.csv"), index=False)

    # ---- 3. crash reasons (counterfactual reruns)
    cf_path = os.path.join(d_why, "counterfactuals.csv")
    failures = df[df.failed]
    cf = analyse_failures(failures)
    cf.to_csv(cf_path)
    _save(charts.crash_causes(cf), os.path.join(d_why, "01_crash_causes.png"))

    # ---- 1. runs
    examples = pick_examples(df, cf)
    run_lines = []
    for stem, title, sub, row in examples:
        s, tr = _replay(row)
        _save(run_figure(s, tr, title, sub), os.path.join(d_runs, stem + ".png"))
        if not args.no_video:
            render_video(s, tr, os.path.join(d_runs, stem + ".mp4"), title, sub)
        run_lines.append(f"| `{stem}.mp4` / `.png` | {title} | {sub} |")
        print("rendered", stem)

    # ---- READMEs
    cause_rows = []
    for k, lab in sorted(CAUSES.items(), key=lambda kv: -(cf[kv[0]] == True).sum()):  # noqa: E712
        fixed = int((cf[k] == True).sum())  # noqa: E712
        cause_rows.append(f"| {lab} | {DESCRIPTIONS[k]} | {fixed} ({fixed / len(cf):.0%}) |")
    compound = int((cf.n_fixes == 0).sum())

    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Random baseline: {n:,} random scenarios flown by RRT

**Headline:** {fail:.1%} of random flights failed ({(df.status == 'collision').mean():.1%} collisions);
{(df.outcome == 'near_failure').mean():.1%} more were near-misses. This set is also the training data for the
generative models, and the baseline they must beat.

Read the folders in order:

1. **[1_runs/](1_runs/)**: watch individual flights. Each has an `.mp4` video and a `.png` summary
   (flight path, clearance over time, speed and wind).
2. **[2_results/](2_results/)**: what happened across all {n:,} runs and which factors drive failure.
3. **[3_crash_reasons/](3_crash_reasons/)**: why the failures happened, measured by re-running each one
   with a single factor changed.

## 1. Runs

| file | what it shows | detail |
|---|---|---|
""" + "\n".join(run_lines) + f"""

**How to read a video:** the drone (black body, blue rotors, yellow nose = heading) follows the blue
plan line toward the star. The pale blue disc is its sensing range. Orange circles are obstacles; dashed
"unseen" ones have not been sensed yet. Arrows on obstacles show their velocity. Grey streaks show the wind.
The panel on the right lists live speed, wind (GUST when a gust is blowing), clearance and replans; the
chart below it traces clearance over time, with the yellow band marking the near-miss zone and the red line
marking contact.

## 2. Results

| file | question it answers |
|---|---|
| `01_outcomes.png` | How did the {n:,} flights end? |
| `02_scenario_mix.png` | What kinds of scenarios did the random generator produce? |
| `03_failure_rate_by_factor.png` | Which scenario factors make failure more likely? |
| `04_failure_map_speed_vs_obstacles.png` | Where in speed × obstacle-count space do failures concentrate? |
| `summary.csv` | The headline numbers. |

## 3. Crash reasons

Each of the {len(cf):,} failures was re-run with one factor changed at a time, using the same planner seed.
If the change turns the failure into a success, that factor is a measured cause. One failure can have several
causes. {compound} failures ({compound / len(cf):.0%}) are not prevented by any single change: they come from a
combination of factors.

| cause | change tested | failures it prevents |
|---|---|---|
""" + "\n".join(cause_rows) + f"""

Files: `01_crash_causes.png` (the chart) and `counterfactuals.csv` (one row per failure; `True` = that change
prevents it, empty = the change does not apply to that scenario).
""")
    print(f"report written to {base}")


if __name__ == "__main__":
    main()
