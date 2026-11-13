# Generative AI-Based Red-Teaming for Autonomous UAV Navigation

CS787 Course Project. Snehil Chatterjee (241026), snehil24@iitk.ac.in. Base paper: DiffScene (Xu et al., ICML 2023 AdvML-Frontiers workshop;
AAAI 2025).

| folder | contents |
|---|---|
| [`report/`](report/) | **[report.pdf](report/report.pdf)**: the project report, written as a research paper |
| [`results/`](results/) | all experimental outputs, one folder per experiment, plus the [demo video](results/demo.mp4) and a plain-language [FINAL_REPORT.md](results/FINAL_REPORT.md) |
| [`code/`](code/) | the complete source code that produces everything in `results/` and `report/` ([code/README.md](code/README.md)) |

## How to go through this repository

**In 5 minutes**

1. Read the headline results below.
2. Open [`results/08_case_study/case_study.pdf`](results/08_case_study/case_study.pdf): **one real case followed
   through the whole pipeline**: a crash the red team created, its measured causes, Gemini's checked explanation,
   the fix, the same scenario flown safely, and the next crash the red team finds. Same story as a 75-second video:
   [`case_study.mp4`](results/08_case_study/case_study.mp4).
3. Optionally watch [`results/demo.mp4`](results/demo.mp4) (2.5 minutes): more example flights and the main charts.

**In 30 minutes**

4. Read [`report/report.pdf`](report/report.pdf). Appendix A of the report lists, with links, where each
   section's results and code live.
5. For any section you want to check, open its results folder (table below). Every results folder has the
   same layout:
   - `README.md`: what the experiment asks, how it was run, and what it found, in plain language
   - `1_runs/`: pictures and videos of individual flights
   - `2_results/`: the charts and the CSV tables behind them
   - `3_crash_reasons/`: measured causes and explanations of individual failures (where relevant)

**To look at or run the code**

6. Read [`code/README.md`](code/README.md): folder layout, setup, and the one command per experiment.
   The code folders follow the pipeline in Figure 1 of the report:
   `scenarios/` → `generators/` → `sim/` + `planners/` → `loop/` → `analyst/`, with `experiments/`
   holding one script per experiment.
7. To reproduce everything: `python -m experiments.run_all --fresh` from inside `code/` (about 5 hours;
   only the LLM stage needs a Gemini API key).

## Headline results (2,000 simulated flights per run, mean of 3 seeds)

| method | distinct failures | scenario regions covered (of 384) | failure types in first 200 failures |
|---|---|---|---|
| Random sampling | 214 | 112 | 28 |
| CMA-ES search | 510 | 31 | 8 |
| Conditional VAE + feedback | 209 | 80 | 35 |
| **Diffusion + feedback (ours)** | 552 | 109 | **35** |
| DiffScene-style gradient guidance (tuned) | **1,245** | 57 | 22 |

Two generators, two strengths. DiffScene-style guidance (the base paper's own steering, re-implemented with a
learned failure critic) finds 2.3x more distinct failures; ours covers twice as much of the scenario space,
finds more failure types early, and keeps 80% of samples valid against 23% (report Sec. 5.7, Table 5).

- Diffusion is 87% valid vs. 50% for the best cVAE, and 2.1x closer to the real scenario distribution.
- Measured counterfactuals raise the LLM's recall of true crash causes from 34% (flight log only) to 96%.
- Failures persist at 4x finer simulation steps, and 77% also break RRT\*.
- Fixes derived from the measured causes survive 79% of past failures and cut the failure rate from 12.0% to
  4.7%; re-attacking, the red team still finds 2.1x more failures than random sampling.

## Where each part of the report lives

| report section | results | main code |
|---|---|---|
| 4.1 Simulator and real-data prior; Fig. 2 | [`results/01_random_baseline/`](results/01_random_baseline/) | [`code/sim/`](code/sim/), [`code/scenarios/`](code/scenarios/) |
| 4.2, 5.2 Diffusion vs. VAE (Table 2, Fig. 4a) | [`results/02_generators/`](results/02_generators/) | [`code/generators/`](code/generators/) |
| 4.3, 5.3 Closed loop, does feedback help? (Fig. 4b) | [`results/03_red_team_loop/`](results/03_red_team_loop/) | [`code/loop/`](code/loop/) |
| 4.4–4.5, 5.4 Measured causes, LLM faithfulness (Table 3, Fig. 5) | [`results/04_llm_analyst/`](results/04_llm_analyst/) | [`code/analyst/`](code/analyst/) |
| 5.1 Main comparison (Table 1, Fig. 3) | [`results/05_comparison/`](results/05_comparison/) | [`code/experiments/run_comparison.py`](code/experiments/run_comparison.py) |
| 5.5 Are the failures real? | [`results/06_fidelity_checks/`](results/06_fidelity_checks/) | [`code/experiments/run_fidelity.py`](code/experiments/run_fidelity.py) |
| 5.6 Harden and re-attack (Table 4) | [`results/07_harden_and_reattack/`](results/07_harden_and_reattack/) | [`code/experiments/run_hardening.py`](code/experiments/run_hardening.py) |
| 4.6 + 5.7 DiffScene-style baseline (Table 5) | [`results/09_diffscene_baseline/`](results/09_diffscene_baseline/) | [`code/generators/diffscene.py`](code/generators/diffscene.py), [`code/experiments/run_diffscene.py`](code/experiments/run_diffscene.py) |
| Appendix A: one case end to end | [`results/08_case_study/`](results/08_case_study/) | [`code/experiments/case_study.py`](code/experiments/case_study.py) |

## Sources

- Planner: [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (Sakai et al., MIT licence), vendored unmodified.
- Obstacle speeds: Stanford Drone Dataset (Robicquet et al., ECCV 2016), via [OpenTraj](https://github.com/crowdbotp/OpenTraj).
- Wind: [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database), station 72494023234 (San Francisco Intl.), 2023.
- LLM: [Google Gemini API](https://ai.google.dev), free tier (`gemini-3.1-flash-lite`).
