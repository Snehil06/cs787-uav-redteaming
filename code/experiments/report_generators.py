"""Evaluate the trained generators and build reports/02_generators/.

    python -m experiments.report_generators

PyTorch is imported only inside functions: the parallel simulation workers re-import this
module on Windows, and loading CUDA in every worker exhausts the page file.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.counterfactuals import CAUSES, DESCRIPTIONS, analyse_failures
from experiments.common import DATA_DIR, run_many
from scenarios.procedural import sample_scenario
from scenarios.spec import COND_NAMES, K_OBS, VECTOR_NAMES, Scenario, condition_vector
from scenarios.validity import check
from sim.simulator import simulate
from viz import charts, gen_charts
from viz.runs import run_figure
from viz.video import render_video

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "02_generators")
CACHE = os.path.join(DATA_DIR, "gen_eval")
DIFFS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
CONFIGS = [("diffusion w=1", "diffusion", 1.0), ("diffusion w=2", "diffusion", 2.0),
           ("diffusion w=4", "diffusion", 4.0), ("cvae", "cvae", None)]


def load_models():
    from generators.cvae import CVAEGenerator
    from generators.diffusion import DiffusionGenerator
    from experiments.train_generators import MODEL_DIR
    return {"diffusion": DiffusionGenerator().load(os.path.join(MODEL_DIR, "diffusion.pt")),
            "cvae": CVAEGenerator().load(os.path.join(MODEL_DIR, "cvae.pt"))}


def sample(models, model, n, spec=None, w=None):
    kw = {"guidance": w} if (model == "diffusion" and w is not None) else {}
    return models[model].sample(n, spec, **kw)


def features(vecs) -> pd.DataFrame:
    rows = []
    for v in vecs:
        s = Scenario.from_vector(v)
        r = dict(zip(COND_NAMES, condition_vector(s)))
        r = {f"c_{k}": x for k, x in r.items()}
        r["noise_std"] = s.noise_std
        r["straight_dist"] = float(np.hypot(*(s.goal - s.start)))
        r["radius_mean"] = float(s.obstacles[s.active, 4].mean()) if s.active.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def nn_distance(query, ref) -> np.ndarray:
    import torch
    from generators.codec import encode
    q = torch.tensor(encode(query), device="cuda" if torch.cuda.is_available() else "cpu")
    r = torch.tensor(encode(ref), device=q.device)
    return torch.cat([torch.cdist(qb, r).min(1).values for qb in q.split(1000)]).cpu().numpy()


def evaluate(models, seed_df):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    # unconditional samples: validity, distributions, novelty
    unc = {}
    for m in ("diffusion", "cvae"):
        vecs = sample(models, m, 3000, None, 2.0)
        errs = [check(Scenario.from_vector(v)) for v in vecs]
        unc[m] = (vecs, errs)
    out["unc"] = unc

    # difficulty steering: sample, fly the valid ones
    rows, flown = [], []
    for name, m, w in CONFIGS:
        for d in DIFFS:
            vecs = sample(models, m, 600, {"difficulty": d}, w)
            ok = [v for v in vecs if not check(Scenario.from_vector(v))][:500]
            df = run_many(ok, seeds=range(len(ok)), extras=[{"method": name, "difficulty_req": d}] * len(ok))
            rows.append({"config": name, "difficulty": d, "validity": len([1 for v in vecs if not check(Scenario.from_vector(v))]) / len(vecs),
                         "failure_rate": (df.outcome == "failure").mean(), "n": len(df)})
            flown.append(df)
            print(f"  steering {name} d={d}: {rows[-1]['failure_rate']:.1%} fail, {rows[-1]['validity']:.0%} valid")
    out["steer"] = pd.DataFrame(rows)
    out["flown"] = pd.concat(flown, ignore_index=True)

    # condition adherence: request one value, measure what the scenarios contain
    adh = []
    asks = {"speed": [3, 5, 7, 9], "n_obstacles": [0, 1, 2, 3, 4, 5], "wind": [1, 3, 5, 7]}
    col = {"speed": "c_speed", "n_obstacles": "c_n_obstacles", "wind": "c_wind_mag"}
    for m in ("diffusion", "cvae"):
        for k, vals in asks.items():
            for v in vals:
                vecs = [x for x in sample(models, m, 300, {k: v}, 2.0) if not check(Scenario.from_vector(x))]
                f = features(vecs)[col[k]]
                adh.append({"model": m, "key": k, "requested": v, "achieved_mean": f.mean(), "achieved_std": f.std()})
    out["adh"] = pd.DataFrame(adh)

    # novelty vs a fresh procedural reference set
    rng = np.random.default_rng(12345)
    fresh = np.stack([sample_scenario(rng).to_vector() for _ in range(3000)])
    train = seed_df[VECTOR_NAMES].to_numpy(float)
    valid = {m: np.stack([v for v, e in zip(*unc[m]) if not e]) for m in unc}
    out["novelty"] = {"fresh random scenarios": nn_distance(fresh, train),
                      "diffusion": nn_distance(valid["diffusion"], train), "cvae": nn_distance(valid["cvae"], train)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-data", default="seed_rrt.parquet")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    d_runs, d_res, d_why = (os.path.join(ROOT, x) for x in ("1_runs", "2_results", "3_crash_reasons"))
    for d in (d_runs, d_res, d_why):
        os.makedirs(d, exist_ok=True)

    seed_df = charts.add_derived(pd.read_parquet(os.path.join(DATA_DIR, args.seed_data)))
    random_rate = seed_df.failed.mean()
    models = load_models()
    ev = evaluate(models, seed_df)
    ev["steer"].to_csv(os.path.join(d_res, "steering.csv"), index=False)
    ev["adh"].to_csv(os.path.join(d_res, "condition_adherence.csv"), index=False)
    ev["flown"].to_parquet(os.path.join(CACHE, "steering_runs.parquet"))

    def save(fig, name, d=d_res):
        fig.savefig(os.path.join(d, name))
        plt.close(fig)

    # ---- 2. results
    real = seed_df.assign(radius_mean=seed_df[[f"o{k}_radius" for k in range(K_OBS)]].replace(0, np.nan).mean(1))
    gens = {m: features([v for v, e in zip(*ev["unc"][m]) if not e]) for m in ("diffusion", "cvae")}
    save(gen_charts.steering(ev["steer"], random_rate), "01_difficulty_steering.png")
    save(gen_charts.distributions(real, gens), "02_realism_distributions.png")
    save(gen_charts.adherence(ev["adh"]), "03_condition_adherence.png")
    save(gen_charts.novelty(ev["novelty"]), "04_novelty.png")
    hard = ev["flown"][(ev["flown"].method == "diffusion w=2") & (ev["flown"].difficulty_req == 1.0)].head(18)
    save(gen_charts.gallery(hard[VECTOR_NAMES].to_numpy(float), hard.outcome.tolist(),
                            f"Scenarios diffusion produced when asked for 'difficulty 1.0': "
                            f"{(hard.outcome == 'failure').mean():.0%} of these crash or fail",
                            "Route = faint blue line; orange = obstacles (faded = appears later; arrows = velocity); "
                            "grey arrow top-right = wind"), "05_gallery_hard_requests.png")
    sweep_path = os.path.join(d_res, "cvae_beta_sweep.csv")
    if os.path.exists(sweep_path):
        save(gen_charts.cvae_tradeoff(pd.read_csv(sweep_path)), "06_cvae_beta_tradeoff.png")
    viol = {}
    for m in ("diffusion", "cvae"):
        for e in ev["unc"][m][1]:
            for r in {x.split("_", 1)[1] if x.startswith("obs") else x for x in e}:
                viol[(m, r)] = viol.get((m, r), 0) + 1
    vt = pd.Series(viol).unstack(0).fillna(0).astype(int).sort_values("diffusion", ascending=False)
    vt.to_csv(os.path.join(d_res, "validity_violations.csv"))

    # ---- 3. crash reasons: random vs diffusion-hard vs cVAE-hard failures
    fl = ev["flown"]
    sets = {"random baseline": seed_df[seed_df.failed].sample(300, random_state=0),
            "diffusion (hard requests)": fl[(fl.method == "diffusion w=2") & (fl.difficulty_req >= 0.9) & (fl.outcome == "failure")],
            "cVAE (hard requests)": fl[(fl.method == "cvae") & (fl.difficulty_req >= 0.9) & (fl.outcome == "failure")]}
    cfs = {k: analyse_failures(v) for k, v in sets.items() if len(v)}
    for k, cf in cfs.items():
        cf.to_csv(os.path.join(d_why, f"counterfactuals_{k.split()[0].lower()}.csv"))
    save(gen_charts.cause_comparison(cfs), "01_causes_by_method.png", d_why)

    # ---- 1. runs: generated hard scenarios worth watching
    dh = sets["diffusion (hard requests)"]
    dh = dh[(dh.status == "collision") & (dh.t_end >= 4)]
    cfd = cfs.get("diffusion (hard requests)")
    picks = []
    for k in sorted(CAUSES, key=lambda k: -(cfd[k] == True).sum())[:3]:  # noqa: E712
        cand = dh[dh.index.isin(cfd[(cfd[k] == True)].index)]  # noqa: E712
        cand = cand[~cand.index.isin([p[1].name for p in picks])]
        if len(cand):
            picks.append((f"0{len(picks) + 1}_diffusion_crash_{k}", cand.iloc[0],
                          f"Generated crash: {CAUSES[k]}", f"Diffusion asked for difficulty {cand.iloc[0].difficulty_req:.1f}; "
                          f"re-run with {DESCRIPTIONS[k]} -> no crash"))
    ch = sets["cVAE (hard requests)"]
    ch = ch[ch.status == "collision"]
    if len(ch):
        picks.append((f"0{len(picks) + 1}_cvae_crash", ch.iloc[0], "cVAE-generated crash",
                      "for comparison: one of the few failures the cVAE produced"))
    lines = []
    for stem, row, title, sub in picks:
        s = Scenario.from_vector(row[VECTOR_NAMES].to_numpy(float))
        tr = simulate(s, planner="rrt", seed=int(row["seed"]), record=True)
        tr.metrics["F"] = row["F"]
        fig = run_figure(s, tr, title, sub)
        save(fig, stem + ".png", d_runs)
        if not args.no_video:
            render_video(s, tr, os.path.join(d_runs, stem + ".mp4"), title, sub)
        lines.append(f"| `{stem}.mp4` / `.png` | {title} | {sub} |")
        print("rendered", stem)

    # ---- README
    st_ = ev["steer"].set_index(["config", "difficulty"])
    vfree = {m: np.mean([not e for e in ev["unc"][m][1]]) for m in ("diffusion", "cvae")}
    med = {k: float(np.median(v)) for k, v in ev["novelty"].items()}
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Generators: can diffusion / cVAE produce valid, novel, controllable scenarios?

Both models were trained on the {len(seed_df):,} seed scenarios (stage 1) with identical data, conditioning and
training steps. "Difficulty" is the failure score F the scenario should reach (0 = easy, 1 = hardest).

**Headlines**

| | diffusion | cVAE |
|---|---|---|
| valid scenarios (no conditions) | {vfree['diffusion']:.1%} | {vfree['cvae']:.1%} |
| failure rate when asked for difficulty 1.0 | {st_.loc[('diffusion w=2', 1.0), 'failure_rate']:.1%} (guidance 2), {st_.loc[('diffusion w=4', 1.0), 'failure_rate']:.1%} (guidance 4) | {st_.loc[('cvae', 1.0), 'failure_rate']:.1%} |
| failure rate when asked for difficulty 0.1 | {st_.loc[('diffusion w=2', 0.1), 'failure_rate']:.1%} | {st_.loc[('cvae', 0.1), 'failure_rate']:.1%} |
| median distance to nearest training scenario | {med['diffusion']:.2f} | {med['cvae']:.2f} |

Random baseline failure rate: {random_rate:.1%}. Fresh random scenarios sit at median distance
{med['fresh random scenarios']:.2f} from the training set, so generated scenarios are about as new as brand-new ones.
This is before any failure-feedback loop (that is stage 3).

## 1. Runs: [1_runs/](1_runs/)

| file | what it shows | detail |
|---|---|---|
""" + "\n".join(lines) + """

## 2. Results: [2_results/](2_results/)

| file | question it answers |
|---|---|
| `01_difficulty_steering.png` | If we ask for harder scenarios, do we get more failures? At what validity cost? |
| `02_realism_distributions.png` | Do generated scenarios look like real ones? |
| `03_condition_adherence.png` | If we ask for a specific speed / obstacle count / wind, do we get it? |
| `04_novelty.png` | Are generated scenarios new, or copies of training data? |
| `05_gallery_hard_requests.png` | What do "hard" generated scenarios look like? |
| `06_cvae_beta_tradeoff.png` | Was the cVAE baseline tuned fairly? (sweep of its KL weight β) |
| `steering.csv`, `condition_adherence.csv`, `validity_violations.csv`, `cvae_beta_sweep.csv` | The numbers behind the charts. |

The cVAE uses β = 0.3, the best validity / realism balance in the sweep. A first version with β = 0.5 suffered
posterior collapse (it ignored its latent input and produced near-identical average scenarios), so it was
re-tuned with KL warm-up and free bits before this comparison.

## 3. Crash reasons: [3_crash_reasons/](3_crash_reasons/)

`01_causes_by_method.png` compares the measured causes of failures found by random sampling, by diffusion asked
for hard scenarios, and by the cVAE asked for hard scenarios. Per-failure tables: `counterfactuals_*.csv`.
""")
    print("report written to", ROOT)


if __name__ == "__main__":
    main()
