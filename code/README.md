# Code

Complete source for the project. Run every command from inside this `code/` folder.

## Layout (follows the pipeline in Figure 1 of the report)

```
scenarios/     scenario format (46-number vector), validity + fairness rules, procedural generator,
               real-data statistics (Stanford Drone Dataset speeds, NOAA wind) in real_stats.json
sim/           2D UAV simulator: dynamics, wind and gusts, delayed noisy sensing, replanning, hardening options
planners/      adapter around the vendored, unmodified PythonRobotics RRT / RRT* (MIT licence, see LICENSE)
generators/    conditional diffusion model (DDPM + classifier-free guidance), conditional VAE, CMA-ES,
               scenario <-> model-space codec
loop/          failure score F, closed-loop red-teaming (Algorithm 1), metrics (distinct, coverage, types)
analyst/       counterfactual re-runs (measured causes), evidence records, Gemini LLM analyst + grounding check
experiments/   one script per stage; run_all.py runs everything in order
viz/           flight pictures, MP4 videos and charts
tools/         paper figures and PDF builder for the report
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt        # for GPU training, install the CUDA build of torch from pytorch.org
set GEMINI_API_KEY=...                 # free key from https://aistudio.google.com; only the LLM stage needs it
```

## Reproduce everything

```bash
python -m experiments.run_all --fresh
```

About 2.5 hours on an 8-core CPU with a GPU. It writes simulation data to `data/`, trained models to
`models/` and all outputs to `reports/` (the same content as `../results/`). Progress is logged to
`data/run_all.log`; running `python -m experiments.run_all` again resumes after the last finished stage.

| stage | command | output |
|---|---|---|
| seed data (10,000 random flights) | `python -m experiments.run_random --n 10000 --seed 0 --out seed_rrt.parquet` | `data/` |
| 1. random baseline | `python -m experiments.report_baseline` | `reports/01_random_baseline/` |
| train generators | `python -m experiments.train_generators` | `models/` |
| cVAE β sweep | `python -m experiments.tune_cvae` | `reports/02_generators/2_results/` |
| 2. generators | `python -m experiments.report_generators` | `reports/02_generators/` |
| method comparison | `python -m experiments.run_comparison` | `data/comparison/` |
| 3 + 5. loop and comparison | `python -m experiments.report_comparison` | `reports/03_*`, `reports/05_*` |
| 4. LLM analyst | `python -m experiments.run_llm_ablation --n 50` | `reports/04_llm_analyst/` |
| 6. fidelity checks | `python -m experiments.run_fidelity` | `reports/06_fidelity_checks/` |
| 7. harden and re-attack | `python -m experiments.run_hardening` | `reports/07_harden_and_reattack/` |
| demo video | `python -m experiments.make_demo` | `reports/demo.mp4` |
| end-to-end case study | `python -m experiments.case_study` | `reports/08_case_study/` |
| report | `python -m tools.paper_figures` then `python -m tools.build_paper` | `paper/report.pdf` |

A single flight can be inspected with `python -m experiments.plot_run`.

## Real-world data

The generator only needs `scenarios/real_stats.json` (included). To rebuild it from the raw sources, download
them into `data/external/` and run `python -m scenarios.real_data`:

- Stanford Drone Dataset annotations via OpenTraj (<https://github.com/crowdbotp/OpenTraj>), cloned to
  `data/external/OpenTraj/`.
- NOAA Integrated Surface Database, station 72494023234 (San Francisco Intl.), 2023:
  <https://noaa-global-hourly-pds.s3.amazonaws.com/2023/72494023234.csv>, saved as
  `data/external/noaa/72494023234_2023.csv`.
