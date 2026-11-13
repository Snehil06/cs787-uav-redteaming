# Final comparison: which method red-teams RRT best?

Every method flew **2,000 scenarios per run**, repeated over **3 seeds** (mean ± std below).

- **distinct failures**: failures left after removing near-duplicates (model-space distance < 1.0; two
  independent random scenarios sit about 1.2 apart).
- **regions covered**: cells of a 384-cell grid over speed x obstacles x moving x wind x sensing that
  contain at least one failure.
- **failure types**: groups of failures with the same measured causes (counterfactual reruns). Counted among
  the first 200 failures (fair: does not reward simply finding more) and in total.

| method | failures | failure rate | distinct failures | regions covered | types in first 200 | types total | flights to 50th distinct | valid samples |
|---|---|---|---|---|---|---|---|---|
| Random | 215 ± 15 | 10.7% ± 0.7% | 214 ± 15 | 112 ± 2 | 28 ± 2 | 28 ± 2 | 461 ± 51 | - |
| CMA-ES search | 743 ± 109 | 37.2% ± 5.4% | 510 ± 71 | 31 ± 2 | 8 ± 3 | 18 ± 1 | 234 ± 51 | - |
| cVAE | 262 ± 4 | 13.1% ± 0.2% | 216 ± 2 | 84 ± 5 | 36 ± 4 | 39 ± 3 | 374 ± 48 | 53% ± 1% |
| cVAE + feedback | 295 ± 9 | 14.8% ± 0.4% | 209 ± 8 | 80 ± 3 | 35 ± 4 | 39 ± 1 | 375 ± 51 | 68% ± 3% |
| Diffusion | 445 ± 19 | 22.3% ± 1.0% | 445 ± 20 | 116 ± 1 | 36 ± 3 | 47 ± 3 | 230 ± 22 | 76% ± 0% |
| Diffusion + feedback (ours) | 553 ± 22 | 27.6% ± 1.1% | 552 ± 21 | 109 ± 2 | 35 ± 4 | 46 ± 3 | 230 ± 22 | 80% ± 1% |

Failure types found **only** by diffusion + feedback (not by any other method, all seeds): **2**
(listed in `3_crash_reasons/types_only_ours.csv`).

## Read in order

1. **[1_runs/](1_runs/)**: `00_gallery_ours.png` (the first distinct failures our method found) and videos of
   failures it found (types unique to it if any exist, otherwise compound failures no single change prevents).

| file | what it shows | detail |
|---|---|---|
| `01_ours_compound.mp4` / `.png` | Found by diffusion + feedback | round 1; measured type: compound (no single fix) |
| `02_ours_unique.mp4` / `.png` | Found by diffusion + feedback only | round 5; measured type: late detection (short sensing range) + pop-up / late-appearing obstacles |

2. **[2_results/](2_results/)**

| file | question it answers |
|---|---|
| `01_headline.png` | How many failures, distinct failures, regions and failure types does each method find? |
| `02_discovery_curves.png` | How does discovery grow as the flight budget is spent? |
| `03_efficiency.png` | How many flights does each method need to find 10 / 50 distinct failures? |
| `04_failure_maps.png` | Where in scenario space does each method's failures sit? |
| `per_seed.csv`, `summary_mean_std.csv` | All numbers. |

3. **[3_crash_reasons/](3_crash_reasons/)**: `01_failure_types_by_method.png`, the mix of measured failure types
   each method finds.
