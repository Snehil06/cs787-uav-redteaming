"""Random/procedural baseline and seed dataset.

    python -m experiments.run_random --n 10000 --out seed_rrt.parquet
"""
import argparse
import os
import time

import numpy as np

from experiments.common import DATA_DIR, run_many, summarize
from scenarios.procedural import sample_scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--planner", default="rrt", choices=["rrt", "rrt_star"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    vectors = [sample_scenario(rng).to_vector() for _ in range(args.n)]
    seeds = args.seed * 1_000_000 + np.arange(args.n)

    t0 = time.time()
    df = run_many(vectors, planner=args.planner, seeds=seeds,
                  extras=[{"method": "random"}] * args.n, workers=args.workers)
    print(summarize(df, f"random/{args.planner}"), f"  ({time.time() - t0:.0f}s)")
    print(df["status"].value_counts().to_string())

    out = args.out or f"random_{args.planner}_n{args.n}_s{args.seed}.parquet"
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(os.path.join(DATA_DIR, out))
    print("saved", os.path.join(DATA_DIR, out))


if __name__ == "__main__":
    main()
