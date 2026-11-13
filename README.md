# CS787 Project Submission: Closed-Loop Diffusion Red-Teaming for UAV Path Planners

Snehil Chatterjee. Base paper: DiffScene (Xu et al., ICML 2023 AdvML-Frontiers workshop; AAAI 2025).

| folder | contents |
|---|---|
| [`report/`](report/) | **[report.pdf](report/report.pdf)**: the project report, written as a 6-page research paper |
| [`results/`](results/) | all experimental outputs, one folder per stage, plus the [demo video](results/demo.mp4) and a plain-language [FINAL_REPORT.md](results/FINAL_REPORT.md). Start at [results/README.md](results/README.md) |
| [`code/`](code/) | the complete source code that produces everything in `results/` and `report/`. See [code/README.md](code/README.md) |

## What the project does

A conditional diffusion model generates scenarios (obstacles, wind, sensing limits) that make an unmodified
open-source RRT / RRT\* UAV planner fail. A simulator decides what happens. Each failure is re-flown with one
factor changed at a time to measure *why* it happened, an LLM (Gemini) explains it under a citation check,
and the measured causes drive targeted fixes to the planner stack, which is then attacked again.
Scenario statistics follow real data (Stanford Drone Dataset obstacle speeds, NOAA wind records).

## Headline results (2,000 simulated flights per run, mean of 3 seeds)

| method | distinct failures | scenario regions covered (of 384) | failure types in first 200 failures |
|---|---|---|---|
| Random sampling | 214 | 112 | 28 |
| CMA-ES search | 510 | 31 | 8 |
| Conditional VAE + feedback | 209 | 80 | 35 |
| **Diffusion + feedback (ours)** | **552** | 109 | 35 |

- Diffusion is 87% valid vs. 50% for the best cVAE, and 2.1x closer to the real scenario distribution.
- Measured counterfactuals raise the LLM's recall of true crash causes from 34% (flight log only) to 96%.
- Failures persist at 4x finer simulation steps, and 77% also break RRT\*.
- Fixes derived from the measured causes survive 79% of past failures and cut the failure rate from 12.0% to
  4.7%; re-attacking, the red team still finds 2.1x more failures than random sampling.

## How the results map to the report

| report section | results folder |
|---|---|
| 4.1 Simulator, real-data prior; Fig. 2 | [`results/01_random_baseline/`](results/01_random_baseline/) |
| 5.2 Why diffusion rather than a VAE (Table 2, Fig. 4a) | [`results/02_generators/`](results/02_generators/) |
| 5.3 Does feedback help? (Fig. 4b) | [`results/03_red_team_loop/`](results/03_red_team_loop/) |
| 5.4 LLM faithfulness (Table 3, Fig. 5) | [`results/04_llm_analyst/`](results/04_llm_analyst/) |
| 5.1 Main comparison (Table 1, Fig. 3) | [`results/05_comparison/`](results/05_comparison/) |
| 5.5 Are the failures real? | [`results/06_fidelity_checks/`](results/06_fidelity_checks/) |
| 5.6 Harden and re-attack (Table 4) | [`results/07_harden_and_reattack/`](results/07_harden_and_reattack/) |
