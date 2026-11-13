"""Train the conditional diffusion model and the cVAE on the seed dataset.

    python -m experiments.train_generators --data seed_rrt.parquet --steps 6000
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from experiments.common import DATA_DIR
from generators.cvae import CVAEGenerator
from generators.diffusion import DiffusionGenerator
from scenarios.spec import VECTOR_NAMES, Scenario
from scenarios.validity import is_valid

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def validity_rate(vecs) -> float:
    return float(np.mean([is_valid(Scenario.from_vector(v)) for v in vecs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seed_rrt.parquet")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["diffusion", "cvae"], default=None)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(DATA_DIR, args.data))
    vecs, F = df[VECTOR_NAMES].to_numpy(float), df["F"].to_numpy(float)
    log_path = os.path.join(MODEL_DIR, "train_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for G in (DiffusionGenerator, CVAEGenerator):
        if args.only and G.name != args.only:
            continue
        g = G(seed=args.seed)
        z, cvals = g.prepare(vecs, F)
        t0 = time.time()
        losses = g.fit(z, cvals, steps=args.steps)
        secs = time.time() - t0
        g.save(os.path.join(MODEL_DIR, f"{g.name}.pt"))
        v_free = validity_rate(g.sample(2000))
        v_hard = validity_rate(g.sample(2000, {"difficulty": 0.8}))
        log[g.name] = {"train_seconds": secs, "final_loss": float(np.mean(losses[-200:])),
                       "validity_unconditional": v_free, "validity_difficulty_0.8": v_hard,
                       "params": sum(p.numel() for p in g.net.parameters()), "losses": losses[::20]}
        print(f"{g.name}: {secs:.0f}s, params {log[g.name]['params']:,}, validity {v_free:.1%} "
              f"(hard-conditioned {v_hard:.1%})")
    with open(log_path, "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()
