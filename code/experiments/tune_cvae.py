"""Sweep the cVAE's KL weight and record validity vs. realism (reports/02_generators/2_results/cvae_beta_sweep.csv).

    python -m experiments.tune_cvae

Realism = mean 1-D Wasserstein distance between generated and real condition features, scaled by the real
standard deviation (lower = closer to the seed data). The diffusion model is scored the same way for reference.
"""
import os

import numpy as np
import pandas as pd

from experiments.common import DATA_DIR
from scenarios.spec import VECTOR_NAMES, Scenario, condition_vector
from scenarios.validity import is_valid

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "02_generators", "2_results",
                   "cvae_beta_sweep.csv")
BETAS = [0.02, 0.05, 0.1, 0.3]


def score(gen, real: np.ndarray, n: int = 2000):
    out = gen.sample(n)
    ok = [v for v in out if is_valid(Scenario.from_vector(v))]
    feats = np.array([condition_vector(Scenario.from_vector(v)) for v in ok])
    rng = np.random.default_rng(0)
    w = [np.mean(np.abs(np.sort(rng.choice(feats[:, j], 1500)) - np.sort(rng.choice(real[:, j], 1500))))
         / real[:, j].std() for j in range(real.shape[1])]
    return len(ok) / len(out), float(np.mean(w))


def main():
    from experiments.train_generators import MODEL_DIR
    from generators.cvae import CVAEGenerator
    from generators.diffusion import DiffusionGenerator
    df = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    vecs, F = df[VECTOR_NAMES].to_numpy(float), df.F.to_numpy(float)
    real = np.array([condition_vector(Scenario.from_vector(v)) for v in vecs[:3000]])
    rows = []
    for beta in BETAS:
        g = CVAEGenerator(beta=beta)
        z, c = g.prepare(vecs, F)
        g.fit(z, c, steps=6000, log_every=0)
        v, w = score(g, real)
        rows.append({"model": "cvae", "beta": beta, "validity": round(v, 3), "w_distance": round(w, 3)})
        print(rows[-1], flush=True)
    v, w = score(DiffusionGenerator().load(os.path.join(MODEL_DIR, "diffusion.pt")), real)
    rows.append({"model": "diffusion", "beta": None, "validity": round(v, 3), "w_distance": round(w, 3)})
    print(rows[-1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
