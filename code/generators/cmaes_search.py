"""Search baseline: CMA-ES maximises the failure score F directly over scenario parameters.

No generative model; it answers "why not just optimise F?". Several independent restarts
are used so that it can find more than one failure region.
"""
from multiprocessing import Pool

import cma
import numpy as np
import pandas as pd

from experiments.common import _run_star
from generators.codec import decode
from scenarios.spec import VECTOR_DIM, Scenario
from scenarios.validity import is_valid


def cmaes_search(budget: int, seed: int = 0, restarts: int = 4, popsize: int = 16, planner: str = "rrt",
                 workers: int | None = None, max_asks_factor: int = 20) -> pd.DataFrame:
    """Run CMA-ES until `budget` valid scenarios have been simulated. Invalid ones are not simulated
    (they cost no simulator time) and get the worst fitness."""
    rng = np.random.default_rng(seed)
    per = [budget // restarts + (1 if r < budget % restarts else 0) for r in range(restarts)]
    rows, sim_seed = [], seed * 1_000_000
    from experiments.common import MAX_WORKERS
    workers = workers or MAX_WORKERS
    with Pool(workers) as pool:
        for r, b in enumerate(per):
            es = cma.CMAEvolutionStrategy(rng.uniform(-0.5, 0.5, VECTOR_DIM), 0.3,
                                          {"bounds": [-1, 1], "popsize": popsize, "seed": int(rng.integers(1, 2**31)),
                                           "verbose": -9})
            done, asks, gen = 0, 0, 0
            while done < b and asks < b * max_asks_factor:
                X = es.ask()
                asks += len(X)
                vecs = decode(np.array(X))
                valid = [is_valid(Scenario.from_vector(v)) for v in vecs]
                todo = [i for i, ok in enumerate(valid) if ok][: b - done]
                jobs = [(vecs[i], planner, sim_seed + k, {"method": "cmaes", "restart": r, "generation": gen})
                        for k, i in enumerate(todo)]
                sim_seed += len(jobs)
                res = pool.map(_run_star, jobs) if jobs else []
                fit = np.zeros(len(X))                 # cma minimises; invalid -> 0 (worst)
                for i, row in zip(todo, res):
                    fit[i] = -row["F"]
                es.tell(X, fit.tolist())
                rows += res
                done += len(res)
                gen += 1
    return pd.DataFrame(rows)
