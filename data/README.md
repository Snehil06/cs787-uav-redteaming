# data/

Processed results of the reported (v2) run, so that every report, figure and the paper can be rebuilt
without re-running the simulations.

| path | contents | produced by |
|---|---|---|
| `seed_rrt.parquet` | 10,000 random scenarios flown by RRT (stage 1, also the generators' training data) | `experiments.run_random` |
| `random_rrtstar_2000.parquet` | 2,000 random scenarios flown by RRT\* | `experiments.run_fidelity` |
| `comparison/` | one parquet per method and seed (2,000 flights each) plus counterfactual CSVs (`*_cf.csv`) | `experiments.run_comparison` |
| `gen_eval/` | flights of generated scenarios at each requested difficulty and guidance weight | `experiments.report_generators` |
| `hardening/` | regression, side-effect and re-attack runs of the hardened system | `experiments.run_hardening` |
| `llm_cache/` | cached Gemini responses; lets `experiments.run_llm_ablation` re-run without an API key | `analyst/llm.py` |

Each parquet row is one flight: the 46 scenario parameters (`VECTOR_NAMES` in `scenarios/spec.py`), the
condition features (`c_*`), the outcome (`success` / `near_failure` / `failure`), the failure score `F`
and trace measurements (minimum clearance, detection times, replans, ...).

## Raw real-world data (not included)

The scenario generator only needs the small quantile file `scenarios/real_stats.json`, which is included.
To rebuild it from the raw sources, download them into `data/external/` and run
`python -m scenarios.real_data`:

- **Stanford Drone Dataset** annotations, via the OpenTraj repository
  (<https://github.com/crowdbotp/OpenTraj>): clone it to `data/external/OpenTraj/`
  (uses `datasets/SDD/*/video*/annotations.txt` and `estimated_scales.yaml`).
- **NOAA Integrated Surface Database**, station 72494023234 (San Francisco Intl.), 2023:
  <https://noaa-global-hourly-pds.s3.amazonaws.com/2023/72494023234.csv>, saved as
  `data/external/noaa/72494023234_2023.csv`.
