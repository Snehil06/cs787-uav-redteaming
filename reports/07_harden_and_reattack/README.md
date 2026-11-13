# Harden and re-attack: closing the loop on the planner

The red team's measured crash causes (stages 1 and 5) pointed at four fixes. All are changes around RRT;
the PythonRobotics RRT code is unchanged.

| fix | what it does |
|---|---|
| wind feed-forward | controller cancels the estimated wind force (targets: wind / gusts) |
| speed governor | never faster than the UAV can stop within its sensing range, and slower near sensed obstacles (targets: speed, late detection) |
| best of 3 plans | RRT runs 3 times per replan and the shortest path is kept (targets: unlucky randomness) |
| wider moving-obstacle footprint | 2 s of obstacle motion swept into its planning footprint instead of 1 s |

## Results

**1. Regression.** Replaying all 8,650 failures found so far (same scenario, same planner seed), the hardened
system survives **79%** of them. By where they were found:
cmaes: 65%; cvae: 96%; cvae+feedback: 97%; diffusion: 80%; diffusion+feedback: 82%; random: 81%; random baseline (stage 1): 77%.

**2. Side effects** on 3,000 ordinary random scenarios (original → hardened):

| | original | hardened |
|---|---|---|
| failure rate | 0.12 | 0.0473 |
| near-failure rate | 0.04 | 0.00933 |
| mean flight time to goal (s) | 12 | 12.7 |
| planning time per flight (s) | 0.00729 | 0.0422 |

61 scenarios that the original system flew safely now fail on the hardened one (a regression cost).

**3. Re-attack** (2,000 flights per run, seeds [0, 1]):

| | random: failure rate | random: distinct failures | diffusion + feedback: failure rate | diffusion + feedback: distinct failures |
|---|---|---|---|---|
| original system | 11.0% | 220 | 27.6% | 552 |
| hardened system | 4.2% | 84 | 8.8% | 176 |

Measured causes of the failures that remain after hardening (share a single change prevents):
wind / gusts: 60%; late detection (short sensing range): 50%; UAV too fast to react: 45%; unlucky planner randomness: 33%; sensor noise: 29%; pop-up / late-appearing obstacles: 10%.

## Read in order

1. **[1_runs/](1_runs/)**

| files | what it shows |
|---|---|
| `01_fixed_before.mp4` / `01_fixed_after.mp4` | the same past failure on the original and the hardened system |
| `02_new_failure_hardened.mp4` | a weakness the red team found after hardening (wind / gusts + late detection (short sensing range) + sensor noise + UAV too fast to react + unlucky planner randomness) |

2. **[2_results/](2_results/)**: `01_regression_by_failure_type.png`, `02_reattack.png`, and the CSVs
   (`regression_by_type.csv`, `side_effects.csv`, `reattack_per_seed.csv`).
3. **[3_crash_reasons/](3_crash_reasons/)**: `01_causes_before_after.png`, what still breaks the planner.
