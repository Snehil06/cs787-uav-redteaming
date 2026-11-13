# One real case, end to end

The whole pipeline on a single case, as one document and one video:

- **[case_study.pdf](case_study.pdf)**: 6-page visual storyboard
- **[case_study.mp4](case_study.mp4)**: the same story with the flights animated

| step | what happens in this case |
|---|---|
| 1. crash created | diffusion + feedback (round 4) generates a scenario; RRT ends with `collision` |
| 2. measured causes | one-factor re-runs: **UAV too fast to react + unlucky planner randomness** |
| 3. LLM explanation | Gemini names uav_speed, planner_randomness; all citations verified |
| 4. fix and re-fly | **speed governor** switched on; same scenario, same seed: `goal` |
| 5. attacked again | the red team finds a new crash on the hardened system: **wind / gusts + late detection (short sensing range) + sensor noise** (Gemini names wind, late_detection, sensor_noise; all citations verified) |

Regenerate with `python -m experiments.case_study` (needs `GEMINI_API_KEY` unless the answers are cached).
