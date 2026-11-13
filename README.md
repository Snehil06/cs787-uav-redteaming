# Closed-Loop Diffusion Red-Teaming for UAV Path Planners

**CS787 course project, Snehil Chatterjee.** A conditional diffusion model generates scenarios that make an
unmodified open-source RRT / RRT\* UAV planner fail. A simulator decides what happens, counterfactual re-runs
measure *why* each failure happened, an LLM (Gemini) explains it under a citation check, and the measured
causes drive targeted fixes to the planner stack, which is then attacked again. Base paper: DiffScene
(Xu et al., ICML 2023 workshop; AAAI 2025).

| start here | what it is |
|---|---|
| [`paper/report.pdf`](paper/report.pdf) | **The report**: a 6-page research paper (method, experiments, limitations, references) |
| [`reports/demo.mp4`](reports/demo.mp4) | 2.5-minute demo video of flights and results |
| [`reports/README.md`](reports/README.md) | All results, stage by stage, with flight videos, charts and crash causes |
| [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) | The same results as a plain-language write-up |

## Headline results (2,000 simulated flights per run, mean of 3 seeds)

| method | distinct failures | scenario regions covered (of 384) | failure types in first 200 failures |
|---|---|---|---|
| Random sampling | 214 | 112 | 28 |
| CMA-ES search | 510 | 31 | 8 |
| Conditional VAE + feedback | 209 | 80 | 35 |
| **Diffusion + feedback (ours)** | **552** | 109 | 35 |

- Diffusion is 87% valid vs. 50% for the best cVAE, and 2.1x closer to the real scenario distribution.
- Measured counterfactuals raise the LLM's recall of true crash causes from 34% (flight log only) to 96%;
  retrieving similar past cases does not help.
- Failures persist at 4x finer simulation steps, and 77% also break RRT\*.
- Fixes derived from the measured causes survive 79% of past failures and cut the failure rate from 12.0% to
  4.7%; re-attacking, the red team still finds 2.1x more failures than random sampling.

## Repository layout

The code follows the pipeline in the paper's Figure 1:

```
scenarios/     scenario format (46-number vector), validity + fairness rules, procedural generator,
               real-data statistics (Stanford Drone Dataset speeds, NOAA wind) -> real_stats.json
sim/           2D UAV simulator: dynamics, wind and gusts, delayed noisy sensing, replanning, hardening options
planners/      adapter around the vendored, unmodified PythonRobotics RRT / RRT* (MIT licence, see LICENSE)
generators/    conditional diffusion model (DDPM + classifier-free guidance), conditional VAE, CMA-ES,
               scenario <-> model-space codec
loop/          failure score F, closed-loop red-teaming (Algorithm 1), metrics (distinct, coverage, types)
analyst/       counterfactual re-runs (measured causes), evidence records, Gemini LLM analyst + grounding check
experiments/   one script per stage; run_all.py runs everything in order
viz/           flight pictures, MP4 videos and charts used in reports/
tools/         paper figures and PDF builder (paper/report.pdf)
models/        trained diffusion model and cVAE (v2 run)
data/          processed results of the reported run (see data/README.md)
reports/       outputs, one folder per stage: 1_runs (videos) -> 2_results (charts, CSVs) -> 3_crash_reasons
paper/         report.pdf and its figures
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Rebuild the outputs from the included data and models (minutes, no API key needed):

```bash
python -m experiments.report_comparison     # reports/03 and 05
python -m tools.paper_figures               # paper/figures
python -m tools.build_paper                 # paper/report.pdf
python -m experiments.make_demo             # reports/demo.mp4
```

Re-run the full pipeline from scratch (about 2.5 hours on an 8-core CPU + GPU; resumable):

```bash
set GEMINI_API_KEY=...                      # free key from https://aistudio.google.com; only the LLM stage needs it
python -m experiments.run_all --fresh
```

Stages, in order: seed data → stage 1 report → train generators → cVAE β sweep → stage 2 report → method
comparison → stage 3/5 reports → LLM ablation → fidelity checks → harden and re-attack. Progress is logged to
`data/run_all.log`; running `python -m experiments.run_all` again resumes after the last finished stage.
A single flight can be inspected with `python -m experiments.plot_run`.

## Acknowledgements and data sources

- Planner: [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (Sakai et al., MIT licence).
- Obstacle speeds: Stanford Drone Dataset (Robicquet et al., ECCV 2016), via
  [OpenTraj](https://github.com/crowdbotp/OpenTraj).
- Wind: NOAA Integrated Surface Database, station 72494023234 (San Francisco Intl.), 2023.
- LLM: Google Gemini API, free tier (`gemini-3.1-flash-lite`).
