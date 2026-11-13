import os, time
from analyst.counterfactuals import analyse_failures
from experiments.run_diffscene import OUT, diffscene_red_team, seed_data
from experiments.tune_diffscene import build

if __name__ == "__main__":
    X, F, _ = seed_data()
    cfg = {"cfg": True, "alpha": 2.0, "t_frac": 0.5}
    for seed in (0, 1, 2):
        path = os.path.join(OUT, f"diffscene-tuned_s{seed}.parquet")
        if os.path.exists(path):
            continue
        t0 = time.time()
        df = diffscene_red_team(build(cfg, seed), 2000, 5, False, seed, "diffscene-tuned", X, F)
        df["seed_run"] = seed
        df.to_parquet(path)
        analyse_failures(df[df.outcome == "failure"]).to_csv(path.replace(".parquet", "_cf.csv"))
        print(f"diffscene-tuned seed {seed}: {(df.outcome == 'failure').mean():.1%} failures, {time.time() - t0:.0f}s", flush=True)
