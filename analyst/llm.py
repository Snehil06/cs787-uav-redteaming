"""LLM client: Gemini free tier first, local Ollama (Qwen 2.5 7B) as fallback, every response cached on disk.

The cache key is a hash of (model, system prompt, user prompt, schema), so re-running an
experiment never spends quota twice and results are reproducible.
"""
import hashlib
import json
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_cache")
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
OLLAMA_MODEL = "qwen2.5:7b"
MIN_INTERVAL = 4.5          # seconds between Gemini calls (free-tier friendly)
_last_call = [0.0]


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def _gemini(model, system, prompt, schema):
    from google import genai
    from google.genai import types
    client = genai.Client()
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    r = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                           response_mime_type="application/json", response_schema=schema))
    return json.loads(r.text)


def _ollama(system, prompt, schema):
    import ollama
    r = ollama.chat(model=OLLAMA_MODEL, format=schema, options={"temperature": 0},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return json.loads(r["message"]["content"])


def ask_json(system: str, prompt: str, schema: dict, provider: str = "auto", retries: int = 3) -> tuple[dict, str]:
    """Return (parsed JSON, model name that produced it). provider: auto | gemini | ollama."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    order = ([f"gemini:{m}" for m in GEMINI_MODELS] if provider in ("auto", "gemini") else []) + \
            ([f"ollama:{OLLAMA_MODEL}"] if provider in ("auto", "ollama") else [])
    for name in order:
        path = os.path.join(CACHE_DIR, _key(name, system, prompt, schema) + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f), name
    errors = []
    for name in order:
        kind, model = name.split(":", 1)
        for attempt in range(retries):
            try:
                out = _gemini(model, system, prompt, schema) if kind == "gemini" else _ollama(system, prompt, schema)
                with open(os.path.join(CACHE_DIR, _key(name, system, prompt, schema) + ".json"), "w", encoding="utf-8") as f:
                    json.dump(out, f)
                return out, name
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types; we fall through to the next
                msg = str(e)
                errors.append(f"{name}: {msg[:120]}")
                transient = any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timed out"))
                if not transient:
                    break
                time.sleep(5 * (attempt + 1))
    raise RuntimeError("all LLM providers failed:\n" + "\n".join(errors))
