"""Day-4 comparison: every method gets the same simulator budget, repeated over seeds.

    python -m experiments.run_comparison --budget 2000 --seeds 0 1 2

Writes data/comparison/<method>_s<seed>.parquet (every flight) and
data/comparison/<method>_s<seed>_cf.csv (counterfactual causes of its failures).
Already-finished runs are skipped, so the script can be resumed after an interruption.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from analyst.counterfactuals import analyse_failures
from experiments.common import DATA_DIR, run_many
from generators.cmaes_search import cmaes_search
from scenarios.procedural import sample_scenario
from scenarios.spec import VECTOR_NAMES

OUT = os.path.join(DATA_DIR, "comparison")
METHODS = ["random", "cmaes", "cvae", "cvae+feedback", "diffusion", "diffusion+feedback"]


def run_method(method: str, budget: int, seed: int, seed_df: pd.DataFrame) -> pd.DataFrame:
    if method == "random":
        rng = np.random.default_rng(1000 + seed)
        vecs = [sample_scenario(rng).to_vector() for _ in range(budget)]
        return run_many(vecs, seeds=seed * 1_000_000 + np.arange(budget), extras=[{"method": "random"}] * budget)
    if method == "cmaes":
        return cmaes_search(budget, seed=seed)
    # learned generators: torch imported here only (keeps simulation workers light on Windows)
    from experiments.train_generators import MODEL_DIR
    from generators.cvae import CVAEGenerator
    from generators.diffusion import DiffusionGenerator
    from loop.redteam import red_team
    name = method.split("+")[0]
    G = DiffusionGenerator if name == "diffusion" else CVAEGenerator
    gen = G(seed=seed).load(os.path.join(MODEL_DIR, f"{name}.pt"))
    return red_team(gen, seed_df[VECTOR_NAMES].to_numpy(float), seed_df.F.to_numpy(float), budget=budget,
                    rounds=5, feedback=method.endswith("+feedback"), sim_seed=seed, label=method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seed_df = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet"))
    for seed in args.seeds:
        for m in args.methods:
            path = os.path.join(OUT, f"{m}_s{seed}.parquet")
            if os.path.exists(path):
                continue
            t0 = time.time()
            df = run_method(m, args.budget, seed, seed_df)
            df["seed_run"] = seed
            df.to_parquet(path)
            cf = analyse_failures(df[df.outcome == "failure"])
            cf.to_csv(path.replace(".parquet", "_cf.csv"))
            print(f"{m:20s} seed {seed}: {(df.outcome == 'failure').mean():6.1%} failures, "
                  f"{len(cf)} analysed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
