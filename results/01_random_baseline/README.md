# Random baseline: 10,000 random scenarios flown by RRT

**Headline:** 11.1% of random flights failed (7.3% collisions);
3.9% more were near-misses. This set is also the training data for the
generative models, and the baseline they must beat.

Read the folders in order:

1. **[1_runs/](1_runs/)**: watch individual flights. Each has an `.mp4` video and a `.png` summary
   (flight path, clearance over time, speed and wind).
2. **[2_results/](2_results/)**: what happened across all 10,000 runs and which factors drive failure.
3. **[3_crash_reasons/](3_crash_reasons/)**: why the failures happened, measured by re-running each one
   with a single factor changed.

## 1. Runs

| file | what it shows | detail |
|---|---|---|
| `01_success.mp4` / `.png` | Safe flight through moving obstacles | 4 obstacles, 139 replans, closest pass 0.8 m |
| `02_near_miss.mp4` / `.png` | Near-miss: goal reached, but only just | closest pass 0.01 m (near-miss zone is < 0.5 m) |
| `03_crash_no_wind.mp4` / `.png` | Crash cause: wind / gusts | Re-run with wind and gusts set to zero → no crash (the only single change that does) |
| `04_crash_slower.mp4` / `.png` | Crash cause: UAV too fast to react | Re-run with UAV cruise speed reduced by 30% → no crash (the only single change that does) |
| `05_crash_full_sensing.mp4` / `.png` | Crash cause: late detection (short sensing range) | Re-run with sensing range raised to 25 m → no crash (the only single change that does) |
| `06_crash_reseed.mp4` / `.png` | Crash cause: unlucky planner randomness | Re-run with same scenario re-planned with 3 other RRT random seeds (fixed if at least 2 succeed) → no crash (other changes also help) |
| `07_crash_compound.mp4` / `.png` | Compound crash: no single change prevents it | several factors combine; removing any one alone still crashes |
| `08_out_of_bounds.mp4` / `.png` | Blown out of the geofence | wind 6.1 m/s, 2 obstacles |
| `09_timeout.mp4` / `.png` | Timed out: never reached goal | wind 7.7 m/s, 2 obstacles |

**How to read a video:** the drone (black body, blue rotors, yellow nose = heading) follows the blue
plan line toward the star. The pale blue disc is its sensing range. Orange circles are obstacles; dashed
"unseen" ones have not been sensed yet. Arrows on obstacles show their velocity. Grey streaks show the wind.
The panel on the right lists live speed, wind (GUST when a gust is blowing), clearance and replans; the
chart below it traces clearance over time, with the yellow band marking the near-miss zone and the red line
marking contact.

## 2. Results

| file | question it answers |
|---|---|
| `01_outcomes.png` | How did the 10,000 flights end? |
| `02_scenario_mix.png` | What kinds of scenarios did the random generator produce? |
| `03_failure_rate_by_factor.png` | Which scenario factors make failure more likely? |
| `04_failure_map_speed_vs_obstacles.png` | Where in speed × obstacle-count space do failures concentrate? |
| `summary.csv` | The headline numbers. |

## 3. Crash reasons

Each of the 1,112 failures was re-run with one factor changed at a time, using the same planner seed.
If the change turns the failure into a success, that factor is a measured cause. One failure can have several
causes. 60 failures (5%) are not prevented by any single change: they come from a
combination of factors.

| cause | change tested | failures it prevents |
|---|---|---|
| wind / gusts | wind and gusts set to zero | 873 (79%) |
| UAV too fast to react | UAV cruise speed reduced by 30% | 506 (46%) |
| late detection (short sensing range) | sensing range raised to 25 m | 329 (30%) |
| unlucky planner randomness | same scenario re-planned with 3 other RRT random seeds (fixed if at least 2 succeed) | 274 (25%) |
| sensor noise | sensor noise set to zero | 265 (24%) |
| pop-up / late-appearing obstacles | late-appearing obstacles present from t = 0 on the same paths | 38 (3%) |

Files: `01_crash_causes.png` (the chart) and `counterfactuals.csv` (one row per failure; `True` = that change
prevents it, empty = the change does not apply to that scenario).
