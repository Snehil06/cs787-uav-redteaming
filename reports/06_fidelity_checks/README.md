# Fidelity checks: are the failures real planner weaknesses?

A 2D simulator invites the question "are these failures just artefacts?". Two checks answer it.

**1. Finer time step.** 500 failures from random sampling and 500 from diffusion + feedback were re-flown
at the original step (0.05 s), 2x finer and 4x finer, each with 3 fresh RRT seeds.

| failures from | dt = 0.05 s (as found) | 0.025 s | 0.0125 s |
|---|---|---|---|
| diffusion + feedback failures | 78% | 78% | 76% |
| random baseline failures | 78% | 77% | 76% |

The 0.05 s column is the reference: some failures depend on RRT's random sampling ("RRT luck", stage 1), so not
all of them repeat with new seeds even at the original step. A similar rate at finer steps means the failures
are not created by the integration step.

**2. Second planner (RRT\*).** On the same 2,000 random scenarios, RRT fails 12.2% and RRT\* fails
12.4%; 222 scenarios break both. Failures found against RRT that also break RRT\*:
random baseline failures: 77%; diffusion + feedback failures: 77%.

## Read in order

1. **[1_runs/](1_runs/)**: the same scenario flown by both planners.

| files | what it shows |
|---|---|
| `01_rrt_crash_rrtstar_ok_rrt.mp4` and `01_rrt_crash_rrtstar_ok_rrt_star.mp4` | RRT crashes here, RRT* gets through |
| `02_both_crash_rrt.mp4` and `02_both_crash_rrt_star.mp4` | both planners crash here |

2. **[2_results/](2_results/)**: `01_timestep_check.png`, `02_rrt_vs_rrtstar.png` and the CSVs behind them.
3. **[3_crash_reasons/](3_crash_reasons/)**: `01_causes_rrt_vs_rrtstar.png`, measured causes of each
   planner's failures on the same scenarios.
