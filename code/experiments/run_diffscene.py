"""DiffScene-style baseline (gradient guidance through a learned critic), run like every other method.

    python -m experiments.run_diffscene critic          # train the critic, report its accuracy
    python -m experiments.run_diffscene pilot           # small alpha sweep on a separate seed
    python -m experiments.run_diffscene run --alpha A   # 2,000 flights x 3 seeds, with and without feedback

Outputs go to data/comparison/diffscene*_s<seed>.parquet (+ _cf.csv) next to the other methods' runs.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from analyst.counterfactuals import analyse_failures
from experiments.common import DATA_DIR, run_many
from loop.metrics import summarize
from scenarios.spec import VECTOR_NAMES, Scenario
from scenarios.validity import is_valid

OUT = os.path.join(DATA_DIR, "comparison")
CRITIC = os.path.join(os.path.dirname(DATA_DIR), "models", "diffscene_critic.pt")
INFO = os.path.join(DATA_DIR, "diffscene_info.json")


def seed_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    return df[VECTOR_NAMES].to_numpy(float), df.F.to_numpy(float), df


def load_prior(seed=0):
    from experiments.train_generators import MODEL_DIR
    from generators.diffusion import DiffusionGenerator
    return DiffusionGenerator(seed=seed).load(os.path.join(MODEL_DIR, "diffusion.pt"))


def cmd_critic():
    import torch
    from sklearn.metrics import r2_score, roc_auc_score
    from generators.diffscene import predict, train_critic
    X, F, df = seed_data()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    te, tr = idx[:2000], idx[2000:]
    c = train_critic(X[tr], F[tr])
    p = predict(c, X[te])
    fail = (df.outcome.to_numpy()[te] == "failure")
    info = {"held_out_r2": float(r2_score(F[te], p)), "held_out_auc_failure": float(roc_auc_score(fail, p)),
            "corr": float(np.corrcoef(F[te], p)[0, 1]), "n_train": int(len(tr)), "n_test": int(len(te))}
    print(json.dumps(info, indent=1))
    c = train_critic(X, F)                     # final critic on all seed data
    os.makedirs(os.path.dirname(CRITIC), exist_ok=True)
    torch.save(c.state_dict(), CRITIC)
    json.dump({"critic": info}, open(INFO, "w"), indent=1)


def make_gen(alpha, seed, inner=3, lam=1.0):
    import torch
    from generators.base import DEVICE
    from generators.diffscene import Critic, DiffSceneGenerator
    c = Critic()
    c.load_state_dict(torch.load(CRITIC, map_location=DEVICE))
    return DiffSceneGenerator(load_prior(seed), c.to(DEVICE).eval(), alpha=alpha, inner_steps=inner, lam=lam,
                              seed=seed)


def diffscene_red_team(gen, budget, rounds, feedback, sim_seed, label, seed_X, seed_F):
    """Same budget, rounds and validity filter as loop.redteam.red_team; feedback refits the critic."""
    per_round = [budget // rounds + (1 if r < budget % rounds else 0) for r in range(rounds)]
    frames, ctr = [], sim_seed * 1_000_000
    all_X, all_F = seed_X, seed_F
    for r, n in enumerate(per_round):
        vecs, drawn = [], 0
        while len(vecs) < n:
            m = int((n - len(vecs)) * 1.4) + 8
            cand = gen.sample(m)
            drawn += m
            vecs += [v for v in cand if is_valid(Scenario.from_vector(v))]
        vf = len(vecs) / drawn
        vecs = np.array(vecs[:n])
        df = run_many(vecs, seeds=range(ctr, ctr + n), extras=[{"method": label, "round": r, "valid_frac": vf}] * n)
        ctr += n
        frames.append(df)
        print(f"  [{label}] round {r + 1}/{rounds}: failure {(df.outcome == 'failure').mean():.1%}, valid {vf:.0%}",
              flush=True)
        if feedback and r < rounds - 1:
            all_X = np.vstack([all_X, df[VECTOR_NAMES].to_numpy(float)])
            all_F = np.concatenate([all_F, df.F.to_numpy(float)])
            gen.refit_critic(all_X, all_F, seed=sim_seed * 10 + r)
    return pd.concat(frames, ignore_index=True)


def cmd_pilot(alphas, budget=400, seed=9):
    X, F, _ = seed_data()
    res = []
    for a in alphas:
        t0 = time.time()
        g = make_gen(a, seed)
        df = diffscene_red_team(g, budget, 1, False, 900 + seed, f"pilot_a{a}", X, F)
        s = summarize(df)
        res.append({"alpha": a, **{k: s[k] for k in ("failure_rate", "distinct_failures", "coverage_cells",
                                                     "validity")}, "secs": round(time.time() - t0)})
        print(res[-1], flush=True)
    info = json.load(open(INFO)) if os.path.exists(INFO) else {}
    info["pilot"] = res
    json.dump(info, open(INFO, "w"), indent=1)


def cmd_run(alpha, seeds, budget=2000):
    X, F, _ = seed_data()
    for seed in seeds:
        for label, fb in (("diffscene", False), ("diffscene+feedback", True)):
            path = os.path.join(OUT, f"{label}_s{seed}.parquet")
            if os.path.exists(path):
                continue
            t0 = time.time()
            df = diffscene_red_team(make_gen(alpha, seed), budget, 5, fb, seed, label, X, F)
            df["seed_run"] = seed
            df.to_parquet(path)
            cf = analyse_failures(df[df.outcome == "failure"])
            cf.to_csv(path.replace(".parquet", "_cf.csv"))
            print(f"{label} seed {seed}: {(df.outcome == 'failure').mean():.1%} failures, {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["critic", "pilot", "run"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"critic": cmd_critic, "pilot": lambda: cmd_pilot(a.alphas), "run": lambda: cmd_run(a.alpha, a.seeds)}[a.cmd]()
