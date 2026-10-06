"""
llm_comparison.py

Shared multi-LLM comparison harness used by:
  agents/fundamental_agent_llm.py
  agents/market_sentiment_agent_llm.py
  agents/macro_agent_llm_compare.py

Each of those is a thin adapter that supplies agent-specific context
loading + prompts; this module owns the provider/client plumbing, retry
logic, and the fixed judge-panel scoring shared across all three.

Four genuinely free, open-weight models across the two keys already in
.env (GROQ_API_KEY, OPENROUTER_API_KEY) — confirmed live against both
providers' real model-list endpoints. Free-tier rosters rotate (this repo
has hit 404s on "gone" free models before), so every entry carries
fallback_models tried in order if the primary 404s.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from agents.orchestrator.critic_agent import _safe_float

MODELS: dict[str, dict[str, Any]] = {
    "groq_gptoss120b": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "fallback_models": ["openai/gpt-oss-20b"],
    },
    "groq_gptoss20b": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
        "fallback_models": ["openai/gpt-oss-120b"],
    },
    "openrouter_qwen27b": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.8-27b:free",
        "fallback_models": ["nvidia/nemotron-3.5-lightning:free"],
        "default_headers": {
            "HTTP-Referer": "https://github.com/Ramreddy2748/Data298A--Masters-Project",
            "X-Title": "Data298A Financial Risk Multi-Agent",
        },
    },
    "openrouter_nemotron": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "fallback_models": ["nvidia/nemotron-3.5-lightning:free", "google/gemma-4-31b-it:free"],
        "default_headers": {
            "HTTP-Referer": "https://github.com/Ramreddy2748/Data298A--Masters-Project",
            "X-Title": "Data298A Financial Risk Multi-Agent",
        },
    },
}

# The two judges are the largest model on each provider — never both from
# the same provider, so no single provider's judging style dominates.
JUDGE_PANEL = ["groq_gptoss120b", "openrouter_nemotron"]


def get_client(model_key: str):
    import os

    from openai import OpenAI

    cfg = MODELS[model_key]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise ValueError(f"{cfg['env_key']} not set — required for model '{model_key}'")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"], default_headers=cfg.get("default_headers"))


def parse_response(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    cleaned = raw.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break
    try:
        import json

        return json.loads(cleaned)
    except Exception:
        return None


def _clamp(value: Any, default: float = 5.0) -> float:
    number = _safe_float(value, default) or default
    return round(max(1.0, min(10.0, number)), 2)


def _create_completion(
    client, model_key: str, model_name: str, system_prompt: str, user_prompt: str,
    max_tokens: int, max_retries: int = 5,
):
    from openai import NotFoundError, RateLimitError

    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except NotFoundError:
            raise
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"  {model_key} ({model_name}) rate-limited — retrying in {wait}s...")
            time.sleep(wait)


def run_candidate(model_key: str, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> dict[str, Any]:
    """Calls model_key, trying fallback_models in order if the primary 404s."""
    from openai import NotFoundError

    cfg = MODELS[model_key]
    client = get_client(model_key)
    candidates = [cfg["model"]] + list(cfg.get("fallback_models", []))

    last_exc: Optional[Exception] = None
    used_model = None
    response = None
    for model_name in candidates:
        try:
            response = _create_completion(client, model_key, model_name, system_prompt, user_prompt, max_tokens)
            used_model = model_name
            break
        except NotFoundError as exc:
            print(f"  {model_key}: model '{model_name}' unavailable, trying next fallback...")
            last_exc = exc
            continue

    if response is None:
        raise RuntimeError(
            f"All models for '{model_key}' are unavailable ({candidates}). "
            f"Free-tier rosters rotate — check the current list and update "
            f"agents/llm_comparison.py::MODELS['{model_key}']."
        ) from last_exc

    raw = response.choices[0].message.content
    result = parse_response(raw) or {
        "risk_score": 5.0,
        "risk_label": "MODERATE",
        "confidence": "LOW",
        "evidence": [f"{model_key} ({used_model}) response could not be parsed as JSON."],
    }
    result["risk_score"] = _clamp(result.get("risk_score"))
    score = result["risk_score"]
    result["risk_label"] = "HIGH" if score > 6.0 else "MODERATE" if score > 3.5 else "LOW"
    result["model_key"] = model_key
    result["model"] = used_model
    return result


def run_judge_panel(
    candidate: dict[str, Any],
    judge_prompt_builder: Callable[[dict[str, Any]], str],
    judge_system_prompt: str,
    judge_panel: list[str] = JUDGE_PANEL,
    max_tokens: int = 2500,
    delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """Every judge_panel member except the candidate's own model judges it; scores are averaged."""
    judges = [j for j in judge_panel if j != candidate.get("model_key")]
    breakdown = []

    for judge_key in judges:
        prompt = judge_prompt_builder(candidate)
        try:
            client = get_client(judge_key)
            cfg = MODELS[judge_key]
            candidates_models = [cfg["model"]] + list(cfg.get("fallback_models", []))
            response = None
            for model_name in candidates_models:
                try:
                    from openai import NotFoundError

                    response = _create_completion(client, judge_key, model_name, judge_system_prompt, prompt, max_tokens)
                    break
                except NotFoundError:
                    continue
            if response is None:
                raise RuntimeError(f"No available model for judge '{judge_key}'")
            raw = response.choices[0].message.content
            judged = parse_response(raw) or {"groundedness": 5.0, "consistency": 5.0, "overall": 5.0,
                                               "rationale": f"{judge_key} judge response could not be parsed."}
        except Exception as exc:
            judged = {"groundedness": None, "consistency": None, "overall": None, "rationale": f"judge failed: {exc}"}

        for key in ("groundedness", "consistency", "overall"):
            if judged.get(key) is not None:
                judged[key] = _clamp(judged.get(key))
        judged["judged_by"] = judge_key
        breakdown.append(judged)
        time.sleep(delay_seconds)

    def _avg(key: str) -> Optional[float]:
        values = [j[key] for j in breakdown if j.get(key) is not None]
        return round(sum(values) / len(values), 2) if values else None

    return {
        "judge_groundedness": _avg("groundedness"),
        "judge_consistency": _avg("consistency"),
        "judge_overall": _avg("overall"),
        "judged_by": [j["judged_by"] for j in breakdown],
        "judge_breakdown": breakdown,
    }


def compare_models(
    agent_label: str,
    tickers: list[str],
    context_loader: Callable[[str], dict[str, Any]],
    prompt_builder: Callable[[str, dict[str, Any]], str],
    judge_prompt_builder_factory: Callable[[str, dict[str, Any]], Callable[[dict[str, Any]], str]],
    system_prompt: str,
    judge_system_prompt: str,
    models: Optional[list[str]] = None,
    judge_panel: Optional[list[str]] = None,
    out_path: Optional[str] = None,
    delay_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    models = models or list(MODELS.keys())
    judge_panel = judge_panel or JUDGE_PANEL
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n[{ticker}] running {len(models)} models for {agent_label}...")
        context = context_loader(ticker)
        judge_prompt_builder = judge_prompt_builder_factory(ticker, context)

        outputs = []
        for model_key in models:
            try:
                output = run_candidate(model_key, system_prompt, prompt_builder(ticker, context))
                output["ticker"] = ticker
                outputs.append(output)
                print(f"  {model_key} ({output['model']}): score={output['risk_score']}/10 {output['risk_label']}")
            except Exception as exc:
                print(f"  {model_key} failed: {exc}")
            time.sleep(delay_seconds)

        for output in outputs:
            judge = run_judge_panel(output, judge_prompt_builder, judge_system_prompt, judge_panel,
                                     delay_seconds=delay_seconds)
            rows.append({
                "ticker": ticker,
                "agent": agent_label,
                "model_key": output["model_key"],
                "model": output["model"],
                "risk_score": output["risk_score"],
                "risk_label": output["risk_label"],
                "confidence": output.get("confidence"),
                "evidence": output.get("evidence", []),
                "judge_groundedness": judge["judge_groundedness"],
                "judge_consistency": judge["judge_consistency"],
                "judge_overall": judge["judge_overall"],
                "judged_by": judge["judged_by"],
            })
            print(f"    judged by {judge['judged_by']}: overall={judge['judge_overall']}/10")

    if out_path:
        import json
        from pathlib import Path

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved {len(rows)} rows to {path}")

    return rows
