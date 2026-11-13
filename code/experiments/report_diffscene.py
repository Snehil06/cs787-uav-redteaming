"""Compare the DiffScene-style baseline with every method of the main comparison (same metrics as Table 1).

    python -m experiments.report_diffscene

Writes reports/09_diffscene_baseline/ (README.md, summary.csv, per_seed.csv, chart).
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.common import DATA_DIR
from loop.metrics import N_CELLS, distinct_mask, summarize, unavoidable_mask
from viz.charts import add_derived

ROOT = os.path.join(os.path.dirname(DATA_DIR), "reports", "09_diffscene_baseline")
ORDER = ["random", "cmaes", "cvae+feedback", "diffusion", "diffusion+feedback", "diffscene", "diffscene+feedback", "diffscene-tuned"]
LABEL = {"random": "Random", "cmaes": "CMA-ES", "cvae+feedback": "cVAE + feedback", "diffusion": "Diffusion (CFG)",
         "diffusion+feedback": "Diffusion + feedback (ours)", "diffscene": "DiffScene-style",
         "diffscene+feedback": "DiffScene-style + critic feedback",
         "diffscene-tuned": "DiffScene-style, tuned (CFG prior + late guidance)"}
REAL_FEATS = ["c_speed", "c_n_obstacles", "c_dynamic_frac", "c_wind_mag", "c_sense_range", "noise_std", "gust_amp"]


def realism(df, seed):
    """Mean over features of the 1-D Wasserstein distance to the seed data, in units of the seed's std."""
    return float(np.mean([wasserstein_distance(df[f], seed[f]) / (seed[f].std() + 1e-9) for f in REAL_FEATS]))


def main():
    os.makedirs(ROOT, exist_ok=True)
    seed = add_derived(pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")))
    rows = []
    for m in ORDER:
        for p in sorted(glob.glob(os.path.join(DATA_DIR, "comparison", f"{m}_s[0-9].parquet"))):
            s = int(p[-9])
            df = add_derived(pd.read_parquet(p))
            cf = pd.read_csv(p.replace(".parquet", "_cf.csv"), index_col=0)
            r = summarize(df, cf)
            fail = df[df.outcome == "failure"]
            sig = cf["fixed_by"].fillna("").replace("", "compound")
            r["top_type_share"] = float(sig.value_counts(normalize=True).iloc[0]) if len(sig) else np.nan
            r["top_type"] = sig.value_counts().index[0] if len(sig) else ""
            r["unavoidable_share"] = float(unavoidable_mask(fail).mean()) if len(fail) else np.nan
            r["realism_w"] = realism(df, seed)
            r["mean_wind"] = float(df.c_wind_mag.mean())
            rows.append({"method": m, "seed": s, **r})
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(ROOT, "per_seed.csv"), index=False)
    num = per.drop(columns=["top_type"]).groupby("method").agg(["mean", "std"]).reindex([m for m in ORDER if m in set(per.method)])
    num.round(3).to_csv(os.path.join(ROOT, "summary.csv"))
    tops = per.groupby("method").top_type.agg(lambda x: x.value_counts().index[0])

    cols = [("failures", "{:.0f}"), ("distinct_failures", "{:.0f}"), ("coverage_cells", "{:.0f}"),
            ("types_in_first_200", "{:.0f}"), ("failure_types", "{:.0f}"), ("top_type_share", "{:.0%}"),
            ("unavoidable_share", "{:.1%}"), ("validity", "{:.0%}"), ("realism_w", "{:.2f}"), ("mean_wind", "{:.1f}")]

    def cell(m, k, f):
        if (k, "mean") not in num.columns or pd.isna(num.loc[m, (k, "mean")]):
            return "-"
        mu, sd = num.loc[m, (k, "mean")], num.loc[m, (k, "std")]
        return f.format(mu) + (f" ± {f.format(sd)}" if f == "{:.0f}" and not pd.isna(sd) else "")

    lines = ["| method | " + " | ".join(k for k, _ in cols) + " | most common type |",
             "|---|" + "---|" * (len(cols) + 1)]
    for m in num.index:
        lines.append(f"| {LABEL[m]} | " + " | ".join(cell(m, k, f) for k, f in cols) + f" | {tops[m]} |")

    # cumulative distinct failures
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in num.index:
        curves = [np.cumsum(distinct_mask(pd.read_parquet(p))) for p in
                  sorted(glob.glob(os.path.join(DATA_DIR, "comparison", f"{m}_s[0-9].parquet")))]
        n = min(len(c) for c in curves)
        ax.plot(np.mean([c[:n] for c in curves], 0), label=LABEL[m],
                lw=2.2 if m.startswith("diffscene") or m == "diffusion+feedback" else 1.2,
                ls="--" if m.startswith("diffscene") else "-")
    ax.set_xlabel("simulated flights"), ax.set_ylabel("distinct failures found"), ax.legend(fontsize=8)
    ax.set_title("Distinct failures vs. flight budget (mean of seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "01_cumulative_distinct.png"), dpi=150)
    plt.close(fig)

    import shutil
    shutil.copy(os.path.join(DATA_DIR, "diffscene_tuning.csv"), os.path.join(ROOT, "tuning_sweep.csv"))
    info = json.load(open(os.path.join(DATA_DIR, "diffscene_info.json")))
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# DiffScene-style baseline: gradient guidance through a learned critic

Report Sec. 4.6 and 5.7, Table 5. Code: `generators/diffscene.py`, `experiments/run_diffscene.py`,
`experiments/tune_diffscene.py`, `experiments/run_diffscene_tuned.py`.

**Method.** Our trained diffusion model, used unconditionally as the prior. At every denoising step the
predicted clean scenario takes 3 gradient-ascent steps on a learned critic's predicted failure score, with a
proximal penalty that keeps it near the prior's prediction (DiffScene's constraint objective), then is
re-noised. The critic is an MLP trained on the same 10,000 seed flights (held-out AUC for failure
{info['critic']['held_out_auc_failure']:.2f}, R² for F {info['critic']['held_out_r2']:.2f}). Step size α = 3 was chosen by a pilot
sweep on a separate seed (`data/diffscene_info.json`). "+ critic feedback" refits the critic after each round
on everything flown so far. Same budget (2,000 flights, 5 rounds), validity and fairness filter and metrics as
the main comparison.

**Tuned version.** A sweep of 14 variants on a separate pilot seed (400 flights each; `tuning_sweep.csv`)
tried a failure classifier as critic, critic ensembles with an uncertainty penalty, normalised gradients, a
learned validity term, late guidance, and a classifier-free-guided prior. The selection rule, fixed in
advance, was the most distinct failures among variants with at least 70% of our coverage and at most 10%
unavoidable failures. The winner, "DiffScene-style, tuned", uses our CFG prior (F = 1, w = 2) and applies
the critic gradient (alpha = 2) only in the last half of the denoising steps.

{chr(10).join(lines)}

Coverage is out of {N_CELLS} cells. `top_type_share` is the share of failures with the most common measured type;
`realism_w` is the mean 1-D Wasserstein distance of flown scenarios to the seed data (in seed standard deviations,
lower is more realistic); `unavoidable_share` counts failures where an obstacle appeared within braking time.

![cumulative](01_cumulative_distinct.png)

## Strengths and weaknesses

- **DiffScene-style: targeting.** The critic gradient optimises every sample toward a high predicted failure
  score: about 60% of flights fail (ours 28%), and it finds 2.3x more distinct failures. They are fair (share of
  unavoidable failures no higher than ours) and not collapsed onto one type (CMA-ES: 88% one type).
- **DiffScene-style: narrow and less realistic.** The gradient points most samples the same way (strong wind,
  fast flight), so it covers half as many scenario regions and finds fewer failure types per 200 failures. It
  also pushes samples off the data manifold: only 17-23% pass the validity rules (ours 80%) and the realism gap
  is larger. Refitting the critic on its own flights makes it worse (the guidance exploits the critic).
- **Ours: diverse and realistic, but less targeted.** Classifier-free guidance learns "hard" jointly with the
  scenario distribution and needs no surrogate of the planner, so samples stay valid and spread out; most of
  its flights succeed, though.
- **Together.** The best DiffScene variant uses our CFG prior plus a late critic gradient: the mechanisms are
  complementary.
""")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
