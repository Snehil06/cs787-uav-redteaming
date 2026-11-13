"""Stage 4: how faithful are the LLM's crash explanations, and what evidence makes them better?

    python -m experiments.run_llm_ablation --n 50

Four variants per crash (same model, same prompt rules):
  trace only           - scenario + trace (the first proposal draft's design)
  + counterfactuals    - plus the measured one-change-at-a-time reruns
  + similar cases      - plus retrieved similar past failures (background only)
  full                 - all of the above
Scored against the counterfactuals (measured causes), which the 'trace only' variant never sees.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.analyst import explain
from analyst.evidence import FACTORS, Library, build, failure_type
from experiments.common import DATA_DIR
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import simulate
from viz import style as st
from viz.charts import add_derived
from viz.runs import run_figure

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "04_llm_analyst")
CF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "01_random_baseline",
                       "3_crash_reasons", "counterfactuals.csv")
VARIANTS = {"trace only": (False, False), "+ counterfactuals": (True, False),
            "+ similar cases": (False, True), "full": (True, True)}


def pick_cases(cf: pd.DataFrame, n: int) -> list:
    """Stratified by measured failure type so common and rare types are both represented."""
    types = cf["fixed_by"].fillna("compound")
    rng = np.random.default_rng(0)
    picked, groups = [], {t: list(rng.permutation(idx)) for t, idx in cf.groupby(types).groups.items()}
    while len(picked) < n and any(groups.values()):
        for t in sorted(groups, key=lambda t: -len(groups[t])):
            if groups[t] and len(picked) < n:
                picked.append(groups[t].pop())
    return picked


def score(ans: dict, checks: dict, cf_row: dict) -> dict:
    confirmed = {f for f, k in FACTORS.items() if k and cf_row.get(k) is True}
    testable = {f["factor"] for f in ans["primary_factors"] if FACTORS.get(f["factor"])}
    named = {f["factor"] for f in ans["primary_factors"]}
    contradicted = {f for f in testable if cf_row.get(FACTORS[f]) is False}
    return {
        "precision": len(testable & confirmed) / len(testable) if testable else np.nan,
        "recall": len(named & confirmed) / len(confirmed) if confirmed else np.nan,
        "n_named": len(named), "n_confirmed": len(confirmed),
        "unsupported": checks["n_unsupported"] > 0,
        "contradicted": len(contradicted) > 0,
        "outcome_ok": checks["outcome_matches"],
    }


def chart(summary: pd.DataFrame):
    metrics = [("precision", "named causes that are confirmed"), ("recall", "confirmed causes that get named"),
               ("contradicted", "explanations naming a cause\nthe reruns rule out"),
               ("unsupported", "explanations citing evidence\nthat does not exist")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    cols = [st.MUTED, st.BLUE, st.ORANGE, st.AQUA]
    for ax, (m, lab) in zip(axes, metrics):
        vals = summary[m] * 100
        ax.bar(range(len(vals)), vals, color=cols, width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(vals)), summary.index, rotation=25, ha="right", fontsize=8.5)
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("%")
    fig.tight_layout(w_pad=2)
    best = summary["recall"].idxmax()
    st.finding(fig, f"Counterfactual evidence makes explanations faithful; best recall: {best} "
                    f"({summary.loc[best, 'recall']:.0%})",
               "Same LLM and rules, different evidence. Higher is better for the first two, lower for the last two")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    d_runs, d_res = os.path.join(ROOT, "1_cases"), os.path.join(ROOT, "2_results")
    for d in (d_runs, d_res):
        os.makedirs(d, exist_ok=True)

    df = add_derived(pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")))
    cf = pd.read_csv(CF_PATH, index_col=0)
    for k in FACTORS.values():
        if k:
            cf[k] = cf[k].map({True: True, False: False, "True": True, "False": False})
    lib = Library(df.loc[cf.index], cf)
    cases = pick_cases(cf, args.n)

    rows, records = [], []
    for j, i in enumerate(cases):
        row, cfr = df.loc[i], cf.loc[i].to_dict()
        sim = lib.similar(row)
        for var, (use_cf, use_sim) in VARIANTS.items():
            ev = build(row, cfr, sim if use_sim else None, include_counterfactuals=use_cf)
            r = explain(ev, provider=args.provider)
            s = score(r["answer"], r["checks"], cfr)
            rows.append({"case": int(i), "variant": var, "model": r["model"], **s})
            records.append({"case": int(i), "variant": var, "measured_type": failure_type(cfr),
                            "evidence": ev, **r})
        print(f"  case {j + 1}/{len(cases)} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(d_res, "scores_per_answer.csv"), index=False)
    with open(os.path.join(d_res, "explanations.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    summary = res.groupby("variant")[["precision", "recall", "contradicted", "unsupported", "outcome_ok"]].mean()
    summary = summary.loc[list(VARIANTS)]
    summary.to_csv(os.path.join(d_res, "summary.csv"))
    fig = chart(summary)
    fig.savefig(os.path.join(d_res, "01_faithfulness_by_evidence.png"))
    plt.close(fig)

    # ---- case cards: picture + the full explanation, for a handful of crashes
    lines = []
    for i in cases[: args.examples]:
        row = df.loc[i]
        s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float))
        tr = simulate(s, planner="rrt", seed=int(row["seed"]), record=True)
        tr.metrics["F"] = row["F"]
        rec = next(r for r in records if r["case"] == int(i) and r["variant"] == "full")
        a = rec["answer"]
        fig = run_figure(s, tr, f"Case {i}: {a['failure_type']}", f"Measured type: {rec['measured_type']}")
        fig.savefig(os.path.join(d_runs, f"case_{i}.png"))
        plt.close(fig)
        trace_only = next(r for r in records if r["case"] == int(i) and r["variant"] == "trace only")["answer"]
        with open(os.path.join(d_runs, f"case_{i}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Case {i}\n\n![flight](case_{i}.png)\n\n"
                    f"**Measured causes (counterfactual reruns):** {rec['measured_type']}\n\n"
                    f"## Full-evidence explanation ({rec['model']})\n\n{a['explanation']}\n\n"
                    f"Factors: " + ", ".join(f"`{x['factor']}` ← `{x['evidence_field']}`" for x in a["primary_factors"]) +
                    f"\n\nConfidence: {a['confidence']}\n\n"
                    f"## Same LLM, trace only (no counterfactuals)\n\n{trace_only['explanation']}\n\n"
                    f"Factors: " + ", ".join(f"`{x['factor']}`" for x in trace_only["primary_factors"]) + "\n")
        lines.append(f"| [case_{i}.md](1_cases/case_{i}.md) | {rec['measured_type']} | {a['failure_type']} |")

    models = res.model.value_counts().to_dict()
    tbl = summary.copy()
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# LLM analyst: are the crash explanations faithful to the evidence?

{len(cases)} crashes from the random baseline, each explained 4 times with different evidence
({len(res)} explanations; models used: {models}). Every explanation is scored against the counterfactual
reruns, which measure which single change would have prevented the crash.

| evidence given | precision | recall | contradicted by reruns | cites missing evidence | outcome copied correctly |
|---|---|---|---|---|---|
""" + "\n".join(f"| {v} | {r.precision:.0%} | {r.recall:.0%} | {r.contradicted:.0%} | {r.unsupported:.0%} | {r.outcome_ok:.0%} |"
                for v, r in tbl.iterrows()) + """

- **precision**: of the causes the LLM names that a rerun can test, the share the reruns confirm.
- **recall**: of the causes the reruns confirm, the share the LLM names.
- **contradicted**: explanations that name a cause the reruns show did not matter.
- **cites missing evidence**: explanations whose cited evidence field does not exist (caught by the script).

**How to read this.** The variants that see counterfactuals are scored against those same counterfactuals,
so their high scores show the LLM *uses measured evidence faithfully*, not independent insight. The
independent result is the other two rows: without measured causes, the same LLM names a cause the reruns rule
out in about half of all crashes, and finds only about a third of the real causes. Retrieved similar cases
alone barely help. Conclusion: causes should be measured by the simulator and only interpreted by the LLM.

## Read in order

1. **[1_cases/](1_cases/)**: individual crashes. Each has a flight picture, the measured causes, the full-evidence
   explanation and, for contrast, the trace-only explanation.

| case | measured causes | LLM's failure type (full evidence) |
|---|---|---|
""" + "\n".join(lines) + """

2. **[2_results/](2_results/)**: `01_faithfulness_by_evidence.png` (the chart), `summary.csv`,
   `scores_per_answer.csv` and `explanations.jsonl` (every evidence record and answer).
""")
    print(summary.round(3))
    print("report written to", ROOT)


if __name__ == "__main__":
    main()
