# LLM analyst: are the crash explanations faithful to the evidence?

50 crashes from the random baseline, each explained 4 times with different evidence
(200 explanations; models used: {'gemini:gemini-3.1-flash-lite': 198, 'gemini:gemini-3.5-flash-lite': 2}). Every explanation is scored against the counterfactual
reruns, which measure which single change would have prevented the crash.

| evidence given | precision | recall | contradicted by reruns | cites missing evidence | outcome copied correctly |
|---|---|---|---|---|---|
| trace only | 63% | 34% | 44% | 0% | 100% |
| + counterfactuals | 100% | 92% | 0% | 0% | 100% |
| + similar cases | 59% | 36% | 60% | 0% | 100% |
| full | 98% | 96% | 2% | 0% | 100% |

- **precision**: of the causes the LLM names that a rerun can test, the share the reruns confirm.
- **recall**: of the causes the reruns confirm, the share the LLM names.
- **contradicted**: explanations that name a cause the reruns show did not matter.
- **cites missing evidence**: explanations whose cited evidence field does not exist (caught by the script).

**How to read this.** The variants that see counterfactuals are scored against those same counterfactuals,
so their high scores show the LLM *uses measured evidence faithfully*, not independent insight. The
independent result is the other two rows: without measured causes, the same LLM names a cause the reruns rule
out in about half of all crashes, and finds only about a third of the real causes. Retrieved similar cases
alone barely help. Conclusion: causes should be measured by the simulator and only interpreted by the LLM.

## Read in order

1. **[1_cases/](1_cases/)**: individual crashes. Each has a flight picture, the measured causes, the full-evidence
   explanation and, for contrast, the trace-only explanation.

| case | measured causes | LLM's failure type (full evidence) |
|---|---|---|
| [case_4230.md](1_cases/case_4230.md) | wind / gusts | wind / gusts |
| [case_2709.md](1_cases/case_2709.md) | wind / gusts + UAV too fast to react | wind / gusts + UAV too fast to react |
| [case_966.md](1_cases/case_966.md) | wind / gusts + late detection (short sensing range) + sensor noise + UAV too fast to react + unlucky planner randomness | wind / gusts + sensor noise + UAV too fast to react + unlucky planner randomness |
| [case_5434.md](1_cases/case_5434.md) | compound (no single fix) | wind / gusts + UAV too fast to react |
| [case_2487.md](1_cases/case_2487.md) | wind / gusts + late detection (short sensing range) + UAV too fast to react | wind / gusts + late detection (short sensing range) + UAV too fast to react |
| [case_1938.md](1_cases/case_1938.md) | UAV too fast to react | UAV too fast to react to moving obstacle |

2. **[2_results/](2_results/)**: `01_faithfulness_by_evidence.png` (the chart), `summary.csv`,
   `scores_per_answer.csv` and `explanations.jsonl` (every evidence record and answer).
