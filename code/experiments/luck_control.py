"""Null control for the hardening regression: how much of '79% of past failures survived' is planner luck?

    python -m experiments.luck_control

The hardened system runs RRT three times per replan, which also re-rolls RRT's randomness. So we replay a sample
of 2,000 past failures five ways: the original system with the same seed (must reproduce: 0% survive), the
original system with two fresh seeds (luck alone), and the hardened system with the same and a fresh seed.
Writes reports/demo_clips/luck_control.csv (quoted on the demo video's closing card) (rows: all, and per source method).
"""
import os

import pandas as pd

from experiments.common import DATA_DIR, run_many
from scenarios.spec import VECTOR_NAMES
from sim.simulator import HARDENED

OUT = os.path.join(os.path.dirname(DATA_DIR), "reports", "demo_clips")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lib = pd.read_parquet(os.path.join(DATA_DIR, "hardening", "library_replay.parquet"))
    smp = lib.sample(2000, random_state=1)
    V = smp[VECTOR_NAMES].to_numpy(float)
    cols = {}
    for tag, cfg, off in [("original_same_seed", None, 0), ("original_new_seed", None, 777),
                          ("original_new_seed_2", None, 991), ("hardened_same_seed", HARDENED, 0),
                          ("hardened_new_seed", HARDENED, 777)]:
        r = run_many(V, seeds=(smp.seed.to_numpy() + off).astype(int), sim_cfg=cfg)
        cols[tag] = (r.outcome != "failure").to_numpy()
        print(f"{tag:22s} survives {cols[tag].mean():.1%}", flush=True)
    smp = smp.assign(**cols)
    out = pd.concat([smp[list(cols)].mean().rename("all").to_frame().T,
                     smp.groupby("source")[list(cols)].mean()])
    out.insert(0, "n", [len(smp)] + smp.groupby("source").size().tolist())
    out.round(3).to_csv(os.path.join(OUT, "luck_control.csv"))
    print(out.round(3).to_string())
