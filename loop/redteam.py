"""Failure-guided generation: generate -> fly -> score -> fine-tune on failures -> generate again.

Each round:
  1. sample scenarios asking for maximum difficulty (classifier-free guidance),
  2. fly the valid ones and compute F,
  3. add high-F scenarios to a failure buffer, skipping near-duplicates of what is already there,
  4. fine-tune the generator on the seed data plus the (up-weighted) buffer, labelled with the F each
     scenario actually reached. Keeping the seed data stops the model forgetting ordinary scenarios.
"""
import numpy as np
import pandas as pd

from experiments.common import run_many
from scenarios.spec import VECTOR_NAMES, Scenario
from scenarios.validity import is_valid


def novelty_filter(z_new: np.ndarray, z_buf: np.ndarray, min_dist: float) -> np.ndarray:
    """Indices of rows in z_new at least min_dist from the buffer and from each other (greedy)."""
    keep, ref = [], [z_buf] if len(z_buf) else []
    for i, z in enumerate(z_new):
        pool = np.vstack(ref + [z_new[keep]]) if (ref or keep) else np.zeros((0, z_new.shape[1]))
        if len(pool) == 0 or np.min(np.linalg.norm(pool - z, axis=1)) >= min_dist:
            keep.append(i)
    return np.array(keep, dtype=int)


def red_team(gen, seed_vecs: np.ndarray, seed_F: np.ndarray, budget: int = 2000, rounds: int = 5,
             difficulty: float = 1.0, guidance: float = 2.0, f_threshold: float = 0.5, min_dist: float = 0.5,
             upweight: int = 5, ft_steps: int = 800, ft_lr: float = 3e-4, feedback: bool = True,
             sim_seed: int = 0, label: str = "diffusion+feedback", log=print,
             sim_cfg: dict | None = None) -> pd.DataFrame:
    """Run the loop with a total simulator budget; returns one row per flown scenario."""
    from generators.codec import encode, scenario_conditions

    z_seed, c_seed = gen.prepare(seed_vecs, seed_F)
    buf_vecs, buf_F = np.zeros((0, seed_vecs.shape[1])), np.zeros(0)
    per_round = [budget // rounds + (1 if r < budget % rounds else 0) for r in range(rounds)]
    frames, seed_ctr = [], sim_seed * 1_000_000
    kw = {"guidance": guidance} if gen.name == "diffusion" else {}
    for r, n in enumerate(per_round):
        vecs, drawn = [], 0
        while len(vecs) < n:
            m = int((n - len(vecs)) * 1.4) + 8
            cand = gen.sample(m, {"difficulty": difficulty}, **kw)
            drawn += m
            vecs += [v for v in cand if is_valid(Scenario.from_vector(v))]
        valid_frac = len(vecs) / drawn
        vecs = np.array(vecs[:n])
        df = run_many(vecs, seeds=range(seed_ctr, seed_ctr + n),
                      extras=[{"method": label, "round": r, "valid_frac": valid_frac}] * n, sim_cfg=sim_cfg)
        seed_ctr += n
        frames.append(df)

        hard = df[df.F >= f_threshold]
        if len(hard):
            hv = hard[VECTOR_NAMES].to_numpy(float)
            keep = novelty_filter(encode(hv), encode(buf_vecs) if len(buf_vecs) else np.zeros((0, hv.shape[1])), min_dist)
            buf_vecs = np.vstack([buf_vecs, hv[keep]])
            buf_F = np.concatenate([buf_F, hard.F.to_numpy()[keep]])
        log(f"  [{label}] round {r + 1}/{rounds}: {n} flown, failure {(df.outcome == 'failure').mean():.1%}, "
            f"buffer {len(buf_vecs)}")

        if feedback and r < rounds - 1 and len(buf_vecs):
            zb = encode(buf_vecs)
            cb = scenario_conditions(buf_vecs, buf_F)
            z = np.vstack([z_seed] + [zb] * upweight)
            c = np.vstack([c_seed] + [cb] * upweight)
            gen.fit(z, c, steps=ft_steps, lr=ft_lr, log_every=0)
    return pd.concat(frames, ignore_index=True)
