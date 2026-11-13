# Generators: can diffusion / cVAE produce valid, novel, controllable scenarios?

Both models were trained on the 10,000 seed scenarios (stage 1) with identical data, conditioning and
training steps. "Difficulty" is the failure score F the scenario should reach (0 = easy, 1 = hardest).

**Headlines**

| | diffusion | cVAE |
|---|---|---|
| valid scenarios (no conditions) | 86.8% | 49.6% |
| failure rate when asked for difficulty 1.0 | 22.9% (guidance 2), 35.1% (guidance 4) | 13.1% |
| failure rate when asked for difficulty 0.1 | 6.6% | 6.1% |
| median distance to nearest training scenario | 1.04 | 0.90 |

Random baseline failure rate: 11.1%. Fresh random scenarios sit at median distance
1.10 from the training set, so generated scenarios are about as new as brand-new ones.
This is before any failure-feedback loop (that is stage 3).

## 1. Runs: [1_runs/](1_runs/)

| file | what it shows | detail |
|---|---|---|
| `01_diffusion_crash_no_wind.mp4` / `.png` | Generated crash: wind / gusts | Diffusion asked for difficulty 0.9; re-run with wind and gusts set to zero -> no crash |
| `02_diffusion_crash_full_sensing.mp4` / `.png` | Generated crash: late detection (short sensing range) | Diffusion asked for difficulty 0.9; re-run with sensing range raised to 25 m -> no crash |
| `03_diffusion_crash_slower.mp4` / `.png` | Generated crash: UAV too fast to react | Diffusion asked for difficulty 0.9; re-run with UAV cruise speed reduced by 30% -> no crash |
| `04_cvae_crash.mp4` / `.png` | cVAE-generated crash | for comparison: one of the few failures the cVAE produced |

## 2. Results: [2_results/](2_results/)

| file | question it answers |
|---|---|
| `01_difficulty_steering.png` | If we ask for harder scenarios, do we get more failures? At what validity cost? |
| `02_realism_distributions.png` | Do generated scenarios look like real ones? |
| `03_condition_adherence.png` | If we ask for a specific speed / obstacle count / wind, do we get it? |
| `04_novelty.png` | Are generated scenarios new, or copies of training data? |
| `05_gallery_hard_requests.png` | What do "hard" generated scenarios look like? |
| `06_cvae_beta_tradeoff.png` | Was the cVAE baseline tuned fairly? (sweep of its KL weight β) |
| `steering.csv`, `condition_adherence.csv`, `validity_violations.csv`, `cvae_beta_sweep.csv` | The numbers behind the charts. |

The cVAE uses β = 0.3, the best validity / realism balance in the sweep. A first version with β = 0.5 suffered
posterior collapse (it ignored its latent input and produced near-identical average scenarios), so it was
re-tuned with KL warm-up and free bits before this comparison.

## 3. Crash reasons: [3_crash_reasons/](3_crash_reasons/)

`01_causes_by_method.png` compares the measured causes of failures found by random sampling, by diffusion asked
for hard scenarios, and by the cVAE asked for hard scenarios. Per-failure tables: `counterfactuals_*.csv`.
