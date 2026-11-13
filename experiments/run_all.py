"""Run the whole pipeline end to end, one stage at a time, resumable.

    python -m experiments.run_all            # skips stages already marked done in data/run_all_done.txt
    python -m experiments.run_all --fresh    # forget progress and start again

Each stage runs as its own process (so a crash in one does not take down the rest) and its output is
appended to data/run_all.log.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONE = os.path.join(ROOT, "data", "run_all_done.txt")
LOG = os.path.join(ROOT, "data", "run_all.log")

STAGES = [
    ("seed data", ["-m", "experiments.run_random", "--n", "10000", "--seed", "0", "--out", "seed_rrt.parquet"]),
    ("stage 1 report", ["-m", "experiments.report_baseline"]),
    ("train generators", ["-m", "experiments.train_generators"]),
    ("cVAE beta sweep", ["-m", "experiments.tune_cvae"]),
    ("stage 2 report", ["-m", "experiments.report_generators"]),
    ("comparison runs", ["-m", "experiments.run_comparison"]),
    ("stage 3+5 reports", ["-m", "experiments.report_comparison"]),
    ("LLM ablation", ["-m", "experiments.run_llm_ablation", "--n", "50"]),
    ("fidelity checks", ["-m", "experiments.run_fidelity"]),
    ("harden and re-attack", ["-m", "experiments.run_hardening"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(DONE), exist_ok=True)
    if args.fresh and os.path.exists(DONE):
        os.remove(DONE)
    done = set(open(DONE).read().splitlines()) if os.path.exists(DONE) else set()
    for name, cmd in STAGES:
        if name in done:
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        print(f"[run ] {name} ...", flush=True)
        with open(LOG, "a", encoding="utf-8") as log:
            log.write(f"\n===== {name} ({time.strftime('%H:%M:%S')}) =====\n")
            log.flush()
            code = subprocess.call([sys.executable, *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if code != 0:
            print(f"[FAIL] {name} (exit {code}, {mins:.1f} min) - see data/run_all.log", flush=True)
            sys.exit(code)
        with open(DONE, "a") as f:
            f.write(name + "\n")
        print(f"[ ok ] {name} ({mins:.1f} min)", flush=True)
    print("all stages done", flush=True)


if __name__ == "__main__":
    main()
