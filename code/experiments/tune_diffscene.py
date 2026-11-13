"""Tuning sweep for the DiffScene-style baseline, on a separate pilot seed (never the evaluation seeds 0-2).

    python -m experiments.tune_diffscene stage1     # one change at a time vs. the plain baseline
    python -m experiments.tune_diffscene stage2     # combinations of the best stage-1 changes

Each config: 400 flights, one round, pilot seed 9. Our method is run on the same pilot for reference.
Selection rule (fixed before looking at results): most distinct failures among configs whose coverage is at least
70% of ours on the pilot and whose unavoidable-failure share is at most 10%.
Results: data/diffscene_tuning.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from experiments.common import DATA_DIR
from experiments.run_diffscene import diffscene_red_team, load_prior, seed_data
from loop.metrics import summarize, unavoidable_mask
from scenarios.spec import Scenario
from scenarios.validity import is_valid

OUT = os.path.join(DATA_DIR, "diffscene_tuning.csv")
PILOT_SEED, BUDGET = 9, 400

STAGE1 = {
    "plain a=3": {},
    "fail-classifier a=1": {"target": "fail", "alpha": 1.0},
    "fail-classifier a=3": {"target": "fail", "alpha": 3.0},
    "ensemble5 k=0.5": {"n_critics": 5, "k_unc": 0.5},
    "ensemble5 k=1": {"n_critics": 5, "k_unc": 1.0},
    "normgrad a=0.05": {"norm_grad": True, "alpha": 0.05},
    "normgrad a=0.1": {"norm_grad": True, "alpha": 0.1},
    "normgrad a=0.2": {"norm_grad": True, "alpha": 0.2},
    "late guidance 50%": {"t_frac": 0.5},
    "late guidance 30%": {"t_frac": 0.3},
    "validity w=0.2": {"w_val": 0.2},
    "validity w=0.5": {"w_val": 0.5},
    "hybrid CFG + a=1": {"cfg": True, "alpha": 1.0},
    "hybrid CFG + a=3": {"cfg": True, "alpha": 3.0},
}

_cache = {}


def critics(n, target):
    key = (n, target)
    if key not in _cache:
        from generators.diffscene import train_ensemble
        X, F, df = seed_data()
        y, loss = ((df.outcome.to_numpy() == "failure").astype(float), "bce") if target == "fail" else (F, "mse")
        _cache[key] = train_ensemble(X, y, n, loss)
    return _cache[key]


def validity_critic():
    """Classifier of P(valid), trained on labelled samples of the plain prior and the plain guided sampler."""
    if "val" not in _cache:
        from generators.diffscene import DiffSceneGenerator, train_critic
        prior = load_prior(PILOT_SEED)
        vecs = []
        for a in (0.0, 1.0, 3.0):
            g = DiffSceneGenerator(prior, critics(1, "F"), alpha=a, seed=PILOT_SEED)
            vecs.append(g.sample(3000))
        V = np.vstack(vecs)
        y = np.array([is_valid(Scenario.from_vector(v)) for v in V], dtype=float)
        print(f"  validity classifier trained on {len(V)} samples, {y.mean():.0%} valid", flush=True)
        _cache["val"] = train_critic(V, y, loss="bce", steps=3000, seed=1)
    return _cache["val"]


def build(cfg: dict, seed: int):
    from generators.diffscene import DiffSceneGenerator
    cfg = dict(cfg)
    n, target = cfg.pop("n_critics", 1), cfg.get("target", "F")
    kw = {"alpha": 3.0, **cfg}
    if kw.pop("cfg", False):
        kw.update(cfg_spec={"difficulty": 1.0}, cfg_w=2.0)
    if kw.get("w_val"):
        kw["validity"] = validity_critic()
    return DiffSceneGenerator(load_prior(seed), critics(n, target), seed=seed, **kw)


def metrics(df):
    s = summarize(df)
    fail = df[df.outcome == "failure"]
    return {"failure_rate": s["failure_rate"], "distinct": s["distinct_failures"], "coverage": s["coverage_cells"],
            "validity": s.get("validity", np.nan),
            "unavoidable": float(unavoidable_mask(fail).mean()) if len(fail) else np.nan,
            "mean_wind": float(np.hypot(df.wind_x, df.wind_y).mean())}


def done():
    return pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["config"])


def record(name, df, secs):
    row = {"config": name, **metrics(df), "secs": round(secs)}
    print(row, flush=True)
    out = pd.concat([done(), pd.DataFrame([row])], ignore_index=True)
    out.to_csv(OUT, index=False)


def run_ours():
    from generators.diffusion import DiffusionGenerator  # noqa: F401
    from loop.redteam import red_team
    X, F, _ = seed_data()
    t0 = time.time()
    df = red_team(load_prior(PILOT_SEED), X, F, budget=BUDGET, rounds=1, feedback=False,
                  sim_seed=900 + PILOT_SEED, label="ours pilot")
    record("OURS (diffusion, CFG w=2)", df, time.time() - t0)


def run_configs(configs: dict):
    X, F, _ = seed_data()
    have = set(done().config)
    if "OURS (diffusion, CFG w=2)" not in have:
        run_ours()
    for name, cfg in configs.items():
        if name in have:
            continue
        t0 = time.time()
        g = build(cfg, PILOT_SEED)
        df = diffscene_red_team(g, BUDGET, 1, False, 900 + PILOT_SEED, name, X, F)
        record(name, df, time.time() - t0)


def ranked():
    d = done()
    ours = d[d.config.str.startswith("OURS")].iloc[0]
    d["eligible"] = (d.coverage >= 0.7 * ours.coverage) & (d.unavoidable <= 0.10)
    return d.sort_values(["eligible", "distinct"], ascending=False), ours


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "show"])
    ap.add_argument("--combos", nargs="*", default=[])
    a = ap.parse_args()
    if a.stage == "stage1":
        run_configs(STAGE1)
    elif a.stage == "stage2":
        import json
        run_configs({c: json.loads(c.split("::", 1)[1]) for c in a.combos})
    d, ours = ranked()
    print(d.to_string(index=False))
