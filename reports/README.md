# Reports: start here

Every stage of the project writes its own folder here. Inside each stage folder, read the
subfolders in order: **1_runs** (watch individual flights) → **2_results** (numbers and charts
across all runs) → **3_crash_reasons** (why the failures happened).

| stage | folder | status | what it answers |
|---|---|---|---|
| 1 | [01_random_baseline/](01_random_baseline/) | done | How often does RRT fail on random scenarios, and why? |
| 2 | [02_generators/](02_generators/) | done | Can diffusion / cVAE learn to produce valid, novel scenarios on request? |
| 3 | [03_red_team_loop/](03_red_team_loop/) | done | Does learning from found failures make the generator find more? |
| 4 | [04_llm_analyst/](04_llm_analyst/) | done | Are the LLM's crash explanations faithful to the measured evidence? |
| 5 | [05_comparison/](05_comparison/) | done | Final comparison of all methods (random, CMA-ES, cVAE, diffusion). |
| 6 | [06_fidelity_checks/](06_fidelity_checks/) | done | Are the failures real (finer time step, RRT\*)? |
| 7 | [07_harden_and_reattack/](07_harden_and_reattack/) | done | Can the findings improve the planner, and what breaks it next? |

**The write-up is [FINAL_REPORT.md](FINAL_REPORT.md); the 2.5-minute demo is [demo.mp4](demo.mp4). Base paper presentation deck is [diffscene_deck.html](diffscene_deck.html) (script in [diffscene_deck.md](diffscene_deck.md)).**

Scenarios use real-world statistics: obstacle speeds from the Stanford Drone Dataset (pedestrians, cyclists,
carts filmed from a drone) and wind / gusts from NOAA hourly records (San Francisco airport, 2023). A fairness
rule rejects scenarios in which an obstacle appears inside the UAV's reaction distance, so failures are
avoidable in principle.

## Headline results

- **Diffusion + feedback finds the most distinct failures, across the widest space:** 552 per 2,000 flights
  vs 510 for CMA-ES search and 214 for random sampling, covering 109 regions of scenario space (CMA-ES 31).
- **CMA-ES finds many failures but mostly one kind:** 88% of them are wind-only failures in one corner of the
  space; in its first 200 failures it finds 8 failure types, ours finds 35.
- **The feedback loop works for diffusion** (21% to 36% failure rate from round 1 to 5) but barely for the cVAE.
- **Diffusion is the better generator** (as DiffScene argued for driving): 87% valid vs 50% for the best cVAE,
  and 2.1x closer to the real scenario distribution.
- **Grounded explanations:** with measured counterfactual evidence the LLM names 96% of the real causes;
  from the trace alone it names 34%, and 44% of its explanations blame a cause the reruns rule out.
- **The failures are real:** they persist at 4x finer time steps, and 77% also break RRT\*.
- **Closing the loop:** four targeted fixes survive 79% of past failures and cut the ordinary failure rate from
  12.0% to 4.7%; re-attacking the hardened system, the red team still finds 2.1x as many distinct failures as
  random sampling.

Regenerate a stage with its script, e.g. `python -m experiments.report_baseline`, or everything with
`python -m experiments.run_all`. Videos are `.mp4` (H.264) and play in any browser or media player.
