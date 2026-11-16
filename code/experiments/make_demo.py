"""Assemble reports/demo.mp4: title cards + the key flights; every flight shown is a robust, typical example.

    python -m experiments.make_demo

Why the robustness rules: an earlier version showed the first scenario that matched a loose filter. An audit re-flew each clip with
20 fresh RRT seeds and found that several were luck: e.g. the "before hardening" crash happened with only 15% of
seeds, and the "new failure against the hardened system" with none. Here every clip must pass explicit
robustness rules, fixed before looking at the candidates:

  - a crash clip must crash with at least 90% of 20 fresh planner seeds (80% for RRT* and the re-attack);
  - the safe-flight clip must succeed with at least 95% of seeds;
  - before/after hardening: the original system crashes with >= 90% of seeds AND the hardened one survives >= 90%;
  - typical crashes only: not unavoidable, the obstacle hit is no faster than the UAV's cruise speed and has a
    radius of at most 3 m (obstacles that outrun and ram the UAV are rare: about 2% of our failures).

Writes reports/demo.mp4 and reports/demo_clips/ (clips, selection.csv).
"""
import os
import subprocess
import tempfile

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst.evidence import failure_type
from experiments.common import DATA_DIR, run_many
from loop.metrics import distinct_mask, unavoidable_mask
from scenarios.spec import VECTOR_NAMES, Scenario
from sim.simulator import HARDENED, SimConfig, simulate
from viz import style as st
from viz.charts import add_derived
from viz.runs import run_figure
from viz.video import render_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "demo_clips")
CLIPS = os.path.join(OUT, "clips")
FF = imageio_ffmpeg.get_ffmpeg_exe()
N_SEEDS, POOL = 20, 40
SEED_BASE = 50_000


def typical(df: pd.DataFrame) -> np.ndarray:
    """Closest obstacle no faster than the UAV's cruise speed, radius <= 3 m, failure not unavoidable."""
    k = df.closest_obstacle.astype(int).clip(lower=0).to_numpy()
    vx = np.array([df[f"o{j}_vx"].iloc[i] for i, j in enumerate(k)])
    vy = np.array([df[f"o{j}_vy"].iloc[i] for i, j in enumerate(k)])
    rad = np.array([df[f"o{j}_radius"].iloc[i] for i, j in enumerate(k)])
    ok = (np.hypot(vx, vy) <= df.speed.to_numpy()) & (rad <= 3.0)
    if "outcome" in df and (df.outcome == "failure").any():
        ok &= ~unavoidable_mask(df)
    return ok


def fail_rates(df: pd.DataFrame, planner="rrt", cfg=None) -> np.ndarray:
    """Share of N_SEEDS fresh planner seeds with which each scenario fails."""
    if df.empty:
        return np.array([])
    V = np.repeat(df[VECTOR_NAMES].to_numpy(float), N_SEEDS, axis=0)
    seeds = np.tile(SEED_BASE + np.arange(N_SEEDS), len(df))
    r = run_many(V, planner=planner, seeds=seeds, sim_cfg=cfg)
    return (r.outcome.to_numpy() == "failure").reshape(len(df), N_SEEDS).mean(1)


def first_robust(pool: pd.DataFrame, rule, planner="rrt", cfg=None, extra=None):
    """First row of the pool (in its own order) whose fail rate satisfies rule; returns (row, rate, extra rate)."""
    pool = pool.head(POOL)
    rates = fail_rates(pool, planner, cfg)
    rates2 = fail_rates(pool, *extra) if extra else [None] * len(pool)
    for (_, r), a, b in zip(pool.iterrows(), rates, rates2):
        if rule(a, b):
            return r, a, b
    raise RuntimeError("no candidate passed the rule; widen POOL")


def cause_text(cf: pd.DataFrame, idx) -> str:
    return failure_type(cf.loc[idx].to_dict()) if idx in cf.index else "?"


def select():
    """Each entry: stem, card title, card text, row, planner, sim cfg, clip title, clip subtitle, stats."""
    picks = []
    seed = add_derived(pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")))
    cf_seed = pd.read_csv(os.path.join(REP, "01_random_baseline", "3_crash_reasons", "counterfactuals.csv"), index_col=0)

    # 1. a normal flight
    pool = seed[(seed.outcome == "success") & (seed.c_n_obstacles >= 3) & (seed.c_dynamic_frac > 0) & (seed.t_end >= 6)]
    r, a, _ = first_robust(pool.sort_values("n_replans", ascending=False), lambda a, b: a <= 0.05)
    picks.append(("01_normal", "1. What a normal flight looks like",
                  f"A safe flight past moving obstacles; this scenario succeeds with {1 - a:.0%} of planner seeds.",
                  r, "rrt", None, "Safe flight past moving obstacles", f"succeeds with {1 - a:.0%} of 20 planner seeds",
                  {"fail_rate": a}))

    # 2. a random-scenario crash with one measured cause (not planner luck)
    single = cf_seed[(cf_seed.n_fixes == 1) & (cf_seed.fixed_by != "reseed")].index
    fails = seed.loc[seed.index.isin(single)]
    pool = fails[(fails.status == "collision") & (fails.t_end >= 5)]
    pool = pool[typical(pool)]
    r, a, _ = first_robust(pool, lambda a, b: a >= 0.9)
    t = cause_text(cf_seed, r.name)
    picks.append(("02_random_crash", "2. A crash in a random scenario",
                  f"Measured cause: {t}. Removing it alone prevents the crash.\n"
                  f"It is not luck: the planner crashes here with {a:.0%} of 20 random seeds.",
                  r, "rrt", None, f"Random-scenario crash: {t}", f"crashes with {a:.0%} of 20 planner seeds", {"fail_rate": a}))

    # 3. diffusion asked for a hard scenario
    fl = pd.read_parquet(os.path.join(DATA_DIR, "gen_eval", "steering_runs.parquet"))
    cfd = pd.read_csv(os.path.join(REP, "02_generators", "3_crash_reasons", "counterfactuals_diffusion.csv"), index_col=0)
    dh = fl[(fl.method == "diffusion w=2") & (fl.difficulty_req >= 0.9) & (fl.outcome == "failure")
            & (fl.status == "collision") & (fl.t_end >= 5) & fl.index.isin(cfd.index)]
    dh = dh[typical(dh)]
    r, a, _ = first_robust(dh, lambda a, b: a >= 0.9)
    t = cause_text(cfd, r.name)
    picks.append(("03_diffusion_hard", "3. Diffusion, asked for a hard scenario",
                  f"The model generates a scenario on request and RRT crashes in it ({a:.0%} of 20 planner seeds).\n"
                  f"Measured cause: {t}.",
                  r, "rrt", None, f"Generated on request: {t}", f"crashes with {a:.0%} of 20 planner seeds", {"fail_rate": a}))

    # 4-5. ours, seed 0: a final-round crash and a compound crash
    ours = pd.read_parquet(os.path.join(DATA_DIR, "comparison", "diffusion+feedback_s0.parquet"))
    cf0 = pd.read_csv(os.path.join(DATA_DIR, "comparison", "diffusion+feedback_s0_cf.csv"), index_col=0)
    dm = distinct_mask(ours)
    base = ours[dm & (ours.status == "collision") & (ours.t_end >= 5) & ours.index.isin(cf0.index)]
    base = base[typical(base)]
    no_luck = base[~base.index.isin(cf0[cf0.reseed == True].index)]  # noqa: E712
    late = no_luck[no_luck["round"] == ours["round"].max()]
    r, a, _ = first_robust(late, lambda a, b: a >= 0.9)
    t = cause_text(cf0, r.name)
    picks.append(("04_final_round", "4. After learning from its own finds",
                  f"A crash found in the final round of the loop, after four rounds of fine-tuning.\n"
                  f"Measured cause: {t}; crashes with {a:.0%} of 20 planner seeds.",
                  r, "rrt", None, "Found in the final round of the loop", f"{t}; crashes with {a:.0%} of 20 seeds",
                  {"fail_rate": a}))
    comp = base[base.index.isin(cf0[cf0.n_fixes == 0].index)]
    r, a, _ = first_robust(comp, lambda a, b: a >= 0.9)
    picks.append(("05_compound", "5. A compound failure",
                  f"Found by diffusion + feedback. No single change prevents it: several factors combine.\n"
                  f"Crashes with {a:.0%} of 20 planner seeds.",
                  r, "rrt", None, "Compound failure: no single change prevents it",
                  f"crashes with {a:.0%} of 20 planner seeds", {"fail_rate": a}))

    # 6. the same scenario, RRT*
    s2k = pd.read_parquet(os.path.join(DATA_DIR, "seed_rrt.parquet")).head(2000)
    star = pd.read_parquet(os.path.join(DATA_DIR, "random_rrtstar_2000.parquet"))
    j = s2k.reset_index(drop=True).assign(star_status=star.status.values)
    pool = j[(j.status == "collision") & (j.star_status == "collision") & (j.t_end >= 5)]
    pool = pool[typical(pool)]
    r, a, b = first_robust(pool, lambda a, b: a >= 0.8 and b >= 0.8, extra=("rrt_star", None))
    picks.append(("06_rrt_star", "6. Same scenario, RRT*",
                  f"RRT* fails for the same reasons: RRT crashes with {a:.0%} of seeds, RRT* with {b:.0%}.",
                  r, "rrt_star", None, "Same scenario, RRT*", f"RRT* crashes with {b:.0%} of 20 seeds (RRT: {a:.0%})",
                  {"fail_rate": a, "fail_rate_rrt_star": b}))

    # 7. before / after hardening: robust on both sides
    lib = pd.read_parquet(os.path.join(DATA_DIR, "hardening", "library_replay.parquet"))
    pool = lib[lib.fixed & (lib.status == "collision") & (lib.t_end >= 5) & (lib.source == "diffusion+feedback")]
    pool = pool[typical(pool)]
    r, a, b = first_robust(pool, lambda a, b: a >= 0.9 and b <= 0.1, extra=("rrt", HARDENED))
    picks.append(("07_before", "7. Closing the loop: before hardening",
                  f"A past failure on the original system. Measured type: {r['type']}.\n"
                  f"The original system crashes here with {a:.0%} of 20 planner seeds.",
                  r, "rrt", None, "Original system", f"{r['type']}; crashes with {a:.0%} of 20 seeds", {"fail_rate": a}))
    picks.append(("07_after", "   ...and after hardening",
                  f"Same scenario, same planner seed, hardened system.\n"
                  f"It survives with {1 - b:.0%} of 20 planner seeds.",
                  r, "rrt", HARDENED, "Hardened system", f"survives with {1 - b:.0%} of 20 seeds",
                  {"fail_rate_hardened": b}))

    # 8. a new failure against the hardened system, robust on the hardened system
    nf = pd.concat([pd.read_parquet(os.path.join(DATA_DIR, "hardening", f"diffusion+feedback_s{s}.parquet"))
                    for s in (0, 1)], ignore_index=True)
    nf = nf[(nf.outcome == "failure") & (nf.status == "collision") & (nf.t_end >= 5)]
    nf = nf[typical(nf)]
    r, a, _ = first_robust(nf, lambda a, b: a >= 0.8, cfg=HARDENED)
    picks.append(("08_new_failure", "8. The red team attacks again",
                  f"A new weakness found against the hardened system.\n"
                  f"The hardened system crashes here with {a:.0%} of 20 planner seeds.",
                  r, "rrt", HARDENED, "New failure against the hardened system",
                  f"hardened system crashes with {a:.0%} of 20 seeds", {"fail_rate_hardened": a}))
    return picks


def card(path, title, text):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(0.07, 0.58, title, fontsize=30, weight="semibold", color=st.INK, va="bottom")
    fig.text(0.07, 0.52, text, fontsize=15, color=st.INK_2, va="top", linespacing=1.6)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=100, facecolor=st.SURFACE)
    plt.close(fig)


INTRO = ("Red-teaming a UAV path planner with generative AI",
         "RRT / RRT* (open-source PythonRobotics) flies a 2D simulator with wind, moving and pop-up obstacles,\n"
         "and limited, noisy, delayed sensing. Generative models search for the scenarios that break it.\n"
         "Every flight in this video is a robust example: re-flown with 20 fresh planner seeds.")


def outro(luck):
    txt = ("Diffusion + feedback: 552 distinct failures per 2,000 flights (CMA-ES 510, random 214),\n"
           "spread over 109 regions of scenario space (CMA-ES 31); 88% of CMA-ES failures are one type.\n"
           "Grounded LLM explanations name 96% of measured causes (34% from the trace alone).\n"
           "Hardening: 79% of past failures survived")
    if luck is not None:
        txt += f" (re-rolling the planner's seed alone: {luck:.0%})"
    return ("Results", txt + ";\nthe red team still finds 2x more failures than random afterwards.")


def main():
    os.makedirs(CLIPS, exist_ok=True)
    picks = select()
    rows = []
    for stem, ctitle, ctext, r, planner, cfg, title, sub, stats in picks:
        s = Scenario.from_vector(r[VECTOR_NAMES].to_numpy(float))
        tr = simulate(s, planner=planner, seed=int(r["seed"]), cfg=SimConfig(**(cfg or {})), record=True)
        tr.metrics["F"] = float("nan")
        title = f"{title}: {tr.metrics['status']}"
        fig = run_figure(s, tr, title, sub)
        fig.savefig(os.path.join(CLIPS, stem + ".png"))
        plt.close(fig)
        render_video(s, tr, os.path.join(CLIPS, stem + ".mp4"), title, sub)
        rows.append({"clip": stem, "source_row": r.name, "planner_seed": int(r["seed"]), "status_shown": tr.metrics["status"],
                     **{k: round(float(v), 3) for k, v in stats.items()}})
        print("rendered", stem, rows[-1], flush=True)
    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(OUT, "selection.csv"), index=False)

    luck_p = os.path.join(OUT, "luck_control.csv")
    luck = pd.read_csv(luck_p, index_col=0).loc["all", "original_new_seed"] if os.path.exists(luck_p) else None

    tmp = tempfile.mkdtemp(prefix="demo_")
    story = [(*INTRO, None)] + [(p[1], p[2], os.path.join(CLIPS, p[0] + ".mp4")) for p in picks] + [(*outro(luck), None)]
    inputs, filters = [], []
    for i, (title, text, video) in enumerate(story):
        png = os.path.join(tmp, f"card{i}.png")
        card(png, title, text)
        inputs += ["-loop", "1", "-t", "4" if video else "6", "-i", png]
        filters.append(f"[{len(filters)}:v]scale=1280:720,setsar=1,fps=30,format=yuv420p[v{len(filters)}]")
        if video:
            inputs += ["-i", video]
            filters.append(f"[{len(filters)}:v]scale=1280:720,setsar=1,fps=30,format=yuv420p[v{len(filters)}]")
    n = len(filters)
    graph = ";".join(filters) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    out = os.path.join(REP, "demo.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
