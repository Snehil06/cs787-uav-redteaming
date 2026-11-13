# Red-team loop: does learning from found failures help?

Each run spends its budget over 5 rounds. With feedback, the generator is fine-tuned after every round on
the seed data plus the (up-weighted, de-duplicated) failures found so far; without feedback it keeps sampling
from the original model. Failure rate per round (mean over 3 seeds):

| round | cVAE | cVAE + feedback | Diffusion | Diffusion + feedback (ours) |
|---|---|---|---|---|
| 1 | 13.4% | 13.4% | 21.3% | 21.3% |
| 2 | 13.2% | 13.8% | 21.2% | 22.8% |
| 3 | 12.2% | 15.3% | 24.8% | 29.0% |
| 4 | 13.2% | 15.9% | 21.8% | 29.4% |
| 5 | 13.2% | 15.2% | 22.2% | 35.7% |

## Read in order

1. **[1_runs/](1_runs/)**

| file | what it shows | detail |
|---|---|---|
| `01_last_round_crash.mp4` / `.png` | Crash found in the final round of the loop | after 4 rounds of fine-tuning on earlier failures |

2. **[2_results/](2_results/)**: `01_failure_rate_by_round.png` and `per_round.csv` (failure rate, validity and
   flights per round).

Why the failures happen, and how the methods compare overall: see [../05_comparison/](../05_comparison/).
