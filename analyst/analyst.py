"""Grounded LLM test analyst: explains a flight using only its evidence record, then a script checks it.

The simulator decides the outcome; the LLM only interprets. Every factor the LLM names must cite
an evidence field that exists in this case's record (not the similar-past-cases background),
otherwise the explanation is flagged as unsupported.
"""
import json

from analyst.evidence import FACTORS
from analyst.llm import ask_json

SYSTEM = """You are a test analyst for an autonomous UAV path planner (RRT/RRT*) flying in a 2D simulator.
You are given an evidence record for ONE flight. Explain why the flight ended the way it did.

Rules:
- Use only facts in the evidence record. Do not invent numbers or events.
- The outcome is decided by the simulator; copy it exactly.
- Every primary factor must cite the evidence field that supports it, as a dotted path into the record
  (for example "trace.seconds_between_detection_and_closest_approach" or "counterfactuals.no_wind").
- If counterfactuals are present, a factor is confirmed only when its counterfactual says "failure prevented".
  Prefer confirmed factors; name an unconfirmed one only if the trace strongly supports it.
- "similar_past_cases" is background only. Never cite it as the cause of this flight; you may use it to name
  the failure type and to compare.
- factor must be one of: """ + ", ".join(FACTORS) + """.
- Keep the explanation to 2-4 sentences, plain English, with the key numbers."""

SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["success", "near_failure", "failure"]},
        "primary_factors": {"type": "array", "items": {
            "type": "object",
            "properties": {"factor": {"type": "string", "enum": list(FACTORS)},
                           "evidence_field": {"type": "string"}},
            "required": ["factor", "evidence_field"]}},
        "failure_type": {"type": "string"},
        "explanation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["outcome", "primary_factors", "failure_type", "explanation", "confidence"],
}


def _has_path(ev: dict, path: str) -> bool:
    cur = ev
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def check(ev: dict, out: dict) -> dict:
    """Script-side grounding checks on one LLM answer."""
    unsupported = [f for f in out["primary_factors"]
                   if not _has_path(ev, f["evidence_field"]) or f["evidence_field"].startswith("similar_past_cases")]
    return {"outcome_matches": out["outcome"] == ev["outcome"],
            "n_factors": len(out["primary_factors"]),
            "n_unsupported": len(unsupported),
            "unsupported": unsupported}


def explain(ev: dict, provider: str = "auto") -> dict:
    prompt = "Evidence record:\n" + json.dumps(ev, indent=1)
    out, model = ask_json(SYSTEM, prompt, SCHEMA, provider=provider)
    return {"answer": out, "model": model, "checks": check(ev, out)}
