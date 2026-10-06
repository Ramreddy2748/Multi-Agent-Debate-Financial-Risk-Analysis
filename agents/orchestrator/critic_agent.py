"""
critic_agent.py

Critic / Orchestrator Agent
---------------------------
Runs the three project agents for one ticker, prints every agent output,
and produces one final risk decision.

The critic can optionally use an OpenAI-compatible LLM endpoint for the final
reasoning step. If no critic API key is configured, it uses a transparent
weighted rule-based decision so the pipeline still runs locally.

Example:
    python agents/critic_agent.py A
    python agents/critic_agent.py A --company "Agilent Technologies" --sector Healthcare
    python agents/critic_agent.py A --use-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
except Exception:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")
MARKET_LORA_ADAPTER_DIR = Path(os.getenv("MARKET_LORA_ADAPTER_DIR", PROJECT_ROOT / "models" / "market_sentiment_lora"))
MARKET_LORA_BASE_MODEL = os.getenv("MARKET_LORA_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MARKET_LORA_ENABLED = os.getenv("MARKET_LORA_ENABLED", "0").lower() in {"1", "true", "yes"}
MARKET_LORA_MAX_NEW_TOKENS = int(os.getenv("MARKET_LORA_MAX_NEW_TOKENS", "48"))
MARKET_LORA_REMOTE_URL = os.getenv("MARKET_LORA_REMOTE_URL", "").strip()
MARKET_LORA_REMOTE_TIMEOUT = int(os.getenv("MARKET_LORA_REMOTE_TIMEOUT", "600"))
FUNDAMENTAL_FINBERT_ENABLED = os.getenv("FUNDAMENTAL_FINBERT_ENABLED", "0").lower() in {"1", "true", "yes"}
FUNDAMENTAL_FINBERT_MODEL_DIR = Path(os.getenv("FUNDAMENTAL_FINBERT_MODEL_DIR", PROJECT_ROOT / "models" / "fundamental_agent_finbert"))
sys.path.insert(0, str(PROJECT_ROOT))

from agents.fundamental.finbert import predict_fundamental_finbert
from agents.orchestrator.policy_evidence import build_policy_evidence_trail
from agents.sentiment.finbert import run_sentiment_agent

_MARKET_LORA_RUNTIME: dict[str, Any] | None = None
_MARKET_LORA_LAST_ERROR: str | None = None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _label_from_score(score: float) -> str:
    if score <= 3.5:
        return "LOW"
    if score <= 6.0:
        return "MODERATE"
    return "HIGH"


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(10.0, float(score))), 2)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _latest_row(rows: list[dict[str, str]], date_columns: tuple[str, ...]) -> dict[str, str]:
    if not rows:
        return {}
    for col in date_columns:
        dated = [r for r in rows if r.get(col)]
        if dated:
            return sorted(dated, key=lambda r: str(r.get(col, "")))[-1]
    return rows[-1]


def _usd_m(value: Any) -> float:
    """Normalize values that may be raw USD or already USD millions."""
    number = _safe_float(value, 0.0) or 0.0
    if abs(number) > 1_000_000:
        return number / 1_000_000.0
    return number


def _load_gold_context(ticker: str) -> dict[str, str]:
    rows = _read_csv(GOLD_DIR / "gold_risk_scores_ALL.csv")
    ticker = ticker.upper()
    for row in rows:
        if str(row.get("ticker", "")).upper() == ticker:
            return row
    return {}


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the final JSON object from chatty causal-LM output."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except json.JSONDecodeError:
            continue
    for parsed in reversed(candidates):
        if any(key in parsed for key in ("risk_score", "market_sentiment_risk_score", "combined_risk", "score")):
            return parsed
    if candidates:
        return candidates[-1]
    raise ValueError("LoRA output did not contain valid JSON")


def _load_market_lora_runtime() -> dict[str, Any] | None:
    global _MARKET_LORA_RUNTIME, _MARKET_LORA_LAST_ERROR
    _MARKET_LORA_LAST_ERROR = None
    if _MARKET_LORA_RUNTIME is not None:
        return _MARKET_LORA_RUNTIME

    if not MARKET_LORA_ENABLED:
        _MARKET_LORA_LAST_ERROR = "MARKET_LORA_ENABLED is false."
        return None
    if not (MARKET_LORA_ADAPTER_DIR / "adapter_config.json").exists():
        _MARKET_LORA_LAST_ERROR = f"Missing adapter_config.json at {MARKET_LORA_ADAPTER_DIR}."
        return None

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        _MARKET_LORA_LAST_ERROR = f"Missing LoRA inference dependency: {exc}"
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MARKET_LORA_BASE_MODEL, trust_remote_code=True)
        use_cuda = torch.cuda.is_available()
        use_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        dtype = torch.float16 if use_cuda or use_mps else None
        model = AutoModelForCausalLM.from_pretrained(
            MARKET_LORA_BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(MARKET_LORA_ADAPTER_DIR))
        if not use_cuda and use_mps:
            model = model.to("mps")
        model.eval()
        _MARKET_LORA_RUNTIME = {"tokenizer": tokenizer, "model": model}
        return _MARKET_LORA_RUNTIME
    except Exception as exc:
        _MARKET_LORA_LAST_ERROR = f"Failed to load market LoRA runtime: {type(exc).__name__}: {exc}"
        return None


def _run_market_lora_remote(features: dict[str, Any]) -> dict[str, Any] | None:
    global _MARKET_LORA_LAST_ERROR
    if not MARKET_LORA_REMOTE_URL:
        return None

    payload = {
        "ticker": features["ticker"],
        "features": {
            "annualized_volatility": features["annualized_volatility"],
            "beta": features["beta"],
            "max_drawdown": features["max_drawdown"],
            "sentiment_mean": features["sentiment_mean"],
            "article_count": features["article_count"],
        },
        "max_new_tokens": MARKET_LORA_MAX_NEW_TOKENS,
    }
    request = urllib.request.Request(
        MARKET_LORA_REMOTE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=MARKET_LORA_REMOTE_TIMEOUT) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("prediction"), dict):
            return parsed["prediction"]
        if isinstance(parsed, dict) and isinstance(parsed.get("raw_output"), dict):
            return parsed["raw_output"]
        if isinstance(parsed, dict):
            return parsed
        _MARKET_LORA_LAST_ERROR = "Remote LoRA response was not a JSON object."
        return None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        _MARKET_LORA_LAST_ERROR = f"Remote LoRA HTTP {exc.code}: {detail[:300]}"
        return None
    except Exception as exc:
        _MARKET_LORA_LAST_ERROR = f"Remote LoRA failed: {type(exc).__name__}: {exc}"
        return None


def _run_market_lora(features: dict[str, Any]) -> dict[str, Any] | None:
    global _MARKET_LORA_LAST_ERROR
    remote_result = _run_market_lora_remote(features)
    if remote_result is not None:
        return remote_result
    if MARKET_LORA_REMOTE_URL and _MARKET_LORA_LAST_ERROR:
        return None

    runtime = _load_market_lora_runtime()
    if runtime is None:
        return None

    try:
        import torch

        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        prompt = json.dumps({
            "ticker": features["ticker"],
            "task": (
                "Assess market volatility and financial news sentiment risk. "
                "Return only strict JSON with risk_score, risk_label, confidence, and explanation."
            ),
            "features": {
                "annualized_volatility": features["annualized_volatility"],
                "beta": features["beta"],
                "max_drawdown": features["max_drawdown"],
                "sentiment_mean": features["sentiment_mean"],
                "article_count": features["article_count"],
            },
        })
        messages = [
            {"role": "system", "content": "You are a financial risk agent. Return strict JSON."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"SYSTEM: {messages[0]['content']}\n\nUSER: {messages[1]['content']}\n\nASSISTANT:"
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MARKET_LORA_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _extract_json_object(decoded)
    except Exception as exc:
        _MARKET_LORA_LAST_ERROR = f"LoRA inference failed: {type(exc).__name__}: {exc}"
        return None


def predict_market_lora_payload(features: dict[str, Any]) -> dict[str, Any]:
    """Direct API helper for VSCode/Colab tests of the market LoRA adapter."""
    normalized_features = {
        "ticker": str(features.get("ticker") or "UNKNOWN").upper(),
        "annualized_volatility": _safe_float(features.get("annualized_volatility"), 0.0) or 0.0,
        "beta": _safe_float(features.get("beta"), 1.0) or 1.0,
        "max_drawdown": _safe_float(features.get("max_drawdown"), 0.0) or 0.0,
        "sentiment_mean": _safe_float(features.get("sentiment_mean"), 0.0) or 0.0,
        "article_count": int(_safe_float(features.get("article_count"), 0) or 0),
    }
    fallback_score = _clamp_score(
        1.0
        + min((normalized_features["annualized_volatility"] * 100.0) / 50.0, 1.0) * 3.0
        + min(max(normalized_features["beta"] - 0.8, 0.0) / 1.2, 1.0) * 2.0
        + min(abs(min(normalized_features["max_drawdown"], 0.0)) / 0.40, 1.0) * 2.0
        + min(max(-normalized_features["sentiment_mean"], 0.0) / 0.6, 1.0) * 2.0
    )
    raw = _run_market_lora(normalized_features)
    if raw is None:
        return {
            "ok": False,
            "claim_type": "MARKET_RULE_BASED",
            "fallback_reason": _MARKET_LORA_LAST_ERROR,
            "prediction": {
                "risk_score": fallback_score,
                "risk_label": _label_from_score(fallback_score),
                "confidence": 0.5,
                "explanation": "Market LoRA was unavailable, so this is a rule fallback estimate.",
            },
            "features": normalized_features,
        }
    normalized = _normalize_market_lora_output(raw, normalized_features, fallback_score=fallback_score)
    return {
        "ok": True,
        "claim_type": "LORA_MODEL_INFERENCE",
        "model_name": MARKET_LORA_BASE_MODEL,
        "adapter_dir": str(MARKET_LORA_ADAPTER_DIR),
        "inference_backend": "remote_colab" if MARKET_LORA_REMOTE_URL else "local",
        "prediction": {
            "risk_score": normalized["risk_score"],
            "risk_label": normalized["risk_label"],
            "confidence": normalized["confidence"],
            "explanation": normalized["explanation"],
        },
        "raw_output": normalized["raw_output"],
        "features": normalized_features,
    }


def _normalize_market_lora_output(raw: dict[str, Any], features: dict[str, Any], fallback_score: float) -> dict[str, Any]:
    score = _safe_float(
        raw.get("risk_score")
        or raw.get("market_sentiment_risk_score")
        or raw.get("combined_risk")
        or raw.get("score"),
        fallback_score,
    )
    score = _clamp_score(score or fallback_score)
    label = str(raw.get("risk_label") or _label_from_score(score)).upper()
    if label not in {"LOW", "MODERATE", "HIGH"}:
        label = _label_from_score(score)
    confidence = _safe_float(raw.get("confidence"), 0.65)
    confidence = round(max(0.0, min(1.0, confidence or 0.65)), 2)
    explanation = str(raw.get("explanation") or raw.get("justification") or "LoRA market/news model generated the risk estimate.")
    return {
        "risk_score": score,
        "risk_label": label,
        "confidence": confidence,
        "explanation": explanation,
        "raw_output": raw,
        "evidence": [
            f"LoRA adapter {MARKET_LORA_ADAPTER_DIR.name} produced market/news risk score {score:.2f}.",
            f"Annualized volatility is {features['annualized_volatility']:.2%}; beta is {features['beta']:.2f}.",
            f"Max drawdown is {features['max_drawdown']:.2%}.",
            f"News sentiment mean is {features['sentiment_mean']:.3f} across {features['article_count']} recent articles.",
        ],
    }


def run_fundamental_agent(ticker: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """Rule implementation matching the fundamental fine-tune target logic."""
    ticker = ticker.upper()
    rows = _read_csv(SILVER_DIR / f"silver_edgar_{ticker}.csv")
    row = _latest_row(rows, ("end_date", "filed"))

    if not row:
        return {
            "agent": "Fundamental Agent",
            "ticker": ticker,
            "risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.15,
            "evidence": [f"No EDGAR silver file found for {ticker}."],
            "positive_signals": [],
            "negative_signals": ["Fundamental data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    model_row = {
        **row,
        "ticker": ticker,
        "company": company_name or row.get("company") or ticker,
    }
    if FUNDAMENTAL_FINBERT_ENABLED:
        finbert_output = predict_fundamental_finbert(model_row, FUNDAMENTAL_FINBERT_MODEL_DIR)
        if finbert_output is not None:
            finbert_output["as_of_date"] = row.get("end_date") or row.get("filed")
            return finbert_output

    cash = _usd_m(row.get("cash_usd_m"))
    debt = _usd_m(row.get("long_term_debt"))
    net_income = _usd_m(row.get("net_income_usd_m"))
    operating_income = _usd_m(row.get("operating_income_usd_m"))
    revenue = _usd_m(row.get("revenue") or row.get("revenue_alt"))
    equity = _usd_m(row.get("stockholders_equity_usd_m") or row.get("stockholders_equity"))

    debt_to_cash = debt / (cash + 1.0)
    debt_to_equity = debt / (equity + 1.0) if equity > 0 else 3.0
    profit_margin = net_income / (revenue + 1.0) if revenue > 0 else 0.0
    operating_margin = operating_income / (revenue + 1.0) if revenue > 0 else 0.0

    score = (
        min(debt_to_cash / 5.0, 1.0) * 2.5
        + min(debt_to_equity / 2.0, 1.0) * 2.5
        + (1.0 - max(min(profit_margin / 0.20, 1.0), -1.0)) * 2.5
        + (1.0 - max(min(operating_margin / 0.25, 1.0), -1.0)) * 2.5
    )
    score = _clamp_score(score)

    positive, negative = [], []
    if debt_to_cash <= 2:
        positive.append("Cash reserves are reasonable relative to long-term debt.")
    else:
        negative.append("Long-term debt is elevated relative to cash reserves.")

    if debt_to_equity <= 1:
        positive.append("Debt-to-equity does not indicate severe leverage pressure.")
    elif debt_to_equity <= 2:
        negative.append("Debt-to-equity indicates moderate leverage exposure.")
    else:
        negative.append("Debt-to-equity indicates high leverage pressure.")

    if profit_margin >= 0.08:
        positive.append("Profit margin indicates positive earnings strength.")
    elif profit_margin >= 0:
        negative.append("Profit margin is low, limiting financial flexibility.")
    else:
        negative.append("Negative profit margin indicates profitability stress.")

    if operating_margin >= 0.10:
        positive.append("Operating margin shows stable core business performance.")
    elif operating_margin >= 0:
        negative.append("Operating margin is thin, suggesting limited operational cushion.")
    else:
        negative.append("Negative operating margin indicates weak core operations.")

    return {
        "agent": "Fundamental Agent",
        "ticker": ticker,
        "company": company_name or row.get("company") or ticker,
        "as_of_date": row.get("end_date") or row.get("filed"),
        "risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.78 if rows else 0.15,
        "claim_type": "FUNDAMENTAL_RULE_BASED",
        "metrics": {
            "cash_usd_m": round(cash, 2),
            "long_term_debt_usd_m": round(debt, 2),
            "revenue_usd_m": round(revenue, 2),
            "profit_margin": round(profit_margin, 4),
            "operating_margin": round(operating_margin, 4),
            "debt_to_cash": round(debt_to_cash, 4),
            "debt_to_equity": round(debt_to_equity, 4),
        },
        "evidence": [
            f"Cash {cash:.2f} USDm versus long-term debt {debt:.2f} USDm.",
            f"Profit margin {profit_margin:.2%}; operating margin {operating_margin:.2%}.",
            f"Debt-to-equity {debt_to_equity:.2f}.",
        ],
        "positive_signals": positive,
        "negative_signals": negative,
        "overall_assessment": (
            f"Fundamental risk is {_label_from_score(score)} based on leverage, "
            "profitability, and operating margin."
        ),
    }


def run_market_sentiment_agent(ticker: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """Market + volatility + news sentiment agent."""
    ticker = ticker.upper()
    price_rows = _read_csv(SILVER_DIR / f"silver_prices_{ticker}.csv")
    price_row = _latest_row(price_rows, ("date",))
    news_rows = [
        r for r in _read_csv(SILVER_DIR / "silver_news_sentiment.csv")
        if str(r.get("ticker", "")).upper() == ticker
    ]

    if not price_row:
        return {
            "agent": "Market + Volatility + News Sentiment Agent",
            "ticker": ticker,
            "risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.15,
            "evidence": [f"No price silver file found for {ticker}."],
            "positive_signals": [],
            "negative_signals": ["Market data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    daily_returns = [
        x for x in (_safe_float(r.get("daily_return")) for r in price_rows)
        if x is not None
    ]
    last_30_returns = daily_returns[-30:]
    realized_vol_30d = statistics.pstdev(last_30_returns) if len(last_30_returns) >= 2 else 0.0
    close_values = [
        x for x in (_safe_float(r.get("close")) for r in price_rows)
        if x is not None and x > 0
    ]
    max_drawdown = 0.0
    if close_values:
        peak = close_values[0]
        drawdowns = []
        for value in close_values:
            peak = max(peak, value)
            drawdowns.append((value - peak) / peak)
        max_drawdown = min(drawdowns)

    rolling_vol_30d = _safe_float(price_row.get("rolling_vol_30d"), realized_vol_30d) or 0.0
    rolling_vol_60d = _safe_float(price_row.get("rolling_vol_60d"), rolling_vol_30d) or rolling_vol_30d
    beta = _safe_float(price_row.get("beta"), 1.0) or 1.0
    price_vs_high = _safe_float(price_row.get("price_vs_52w_high_pct"), 0.0) or 0.0
    ytd_return = _safe_float(_load_gold_context(ticker).get("ytd_return_pct"), 0.0) or 0.0

    sentiment_values = [
        x for x in (_safe_float(r.get("sentiment_mean")) for r in news_rows)
        if x is not None
    ]
    avg_sentiment = statistics.mean(sentiment_values[-10:]) if sentiment_values else 0.0
    article_count = sum(int(_safe_float(r.get("article_count"), 0) or 0) for r in news_rows[-10:])

    # In this project some silver files store rolling volatility as an
    # annualized decimal (for example 0.26 = 26%), while raw daily-return
    # volatility is usually much smaller. Handle both forms.
    annual_vol_pct = (
        rolling_vol_30d * 100
        if rolling_vol_30d > 0.10
        else rolling_vol_30d * math.sqrt(252) * 100
    )
    vol_component = min(annual_vol_pct / 50.0, 1.0) * 3.0
    beta_component = min(max(beta - 0.8, 0.0) / 1.2, 1.0) * 2.0
    drawdown_component = min(abs(min(price_vs_high, 0.0)) / 40.0, 1.0) * 2.0
    sentiment_component = min(max(-avg_sentiment, 0.0) / 0.6, 1.0) * 2.0
    ytd_component = min(abs(min(ytd_return, 0.0)) / 35.0, 1.0) * 1.0
    score = _clamp_score(1.0 + vol_component + beta_component + drawdown_component + sentiment_component + ytd_component)

    lora_features = {
        "ticker": ticker,
        "annualized_volatility": annual_vol_pct / 100.0,
        "beta": beta,
        "max_drawdown": max_drawdown,
        "sentiment_mean": avg_sentiment,
        "article_count": article_count,
    }
    lora_result = _run_market_lora(lora_features)
    if lora_result is not None:
        normalized_lora = _normalize_market_lora_output(lora_result, lora_features, fallback_score=score)
        return {
            "agent": "Market + Volatility + News Sentiment Agent",
            "ticker": ticker,
            "company": company_name or price_row.get("long_name") or ticker,
            "as_of_date": price_row.get("date"),
            "risk_score": normalized_lora["risk_score"],
            "risk_label": normalized_lora["risk_label"],
            "confidence": normalized_lora["confidence"],
            "claim_type": "LORA_MODEL_INFERENCE",
            "model_name": MARKET_LORA_BASE_MODEL,
            "inference_backend": "remote_colab" if MARKET_LORA_REMOTE_URL else "local",
            "adapter_dir": str(MARKET_LORA_ADAPTER_DIR.relative_to(PROJECT_ROOT) if MARKET_LORA_ADAPTER_DIR.is_relative_to(PROJECT_ROOT) else MARKET_LORA_ADAPTER_DIR),
            "metrics": {
                "rolling_vol_30d": round(rolling_vol_30d, 6),
                "rolling_vol_60d": round(rolling_vol_60d, 6),
                "annualized_volatility_pct": round(annual_vol_pct, 2),
                "beta": round(beta, 4),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "price_vs_52w_high_pct": round(price_vs_high, 2),
                "ytd_return_pct": round(ytd_return, 2),
                "avg_news_sentiment": round(avg_sentiment, 4),
                "recent_article_count": article_count,
                "rule_fallback_score": score,
            },
            "evidence": normalized_lora["evidence"],
            "positive_signals": [
                "Fine-tuned market/news LoRA adapter was available and used for this specialist verdict."
            ],
            "negative_signals": [],
            "overall_assessment": normalized_lora["explanation"],
            "raw_model_output": normalized_lora["raw_output"],
        }

    positive, negative = [], []
    if rolling_vol_30d <= rolling_vol_60d:
        positive.append("Short-term volatility is not above medium-term volatility.")
    else:
        negative.append("Short-term volatility is above medium-term volatility.")

    if beta <= 1.0:
        positive.append("Beta is at or below market sensitivity.")
    elif beta > 1.5:
        negative.append("Beta indicates high market sensitivity.")

    if price_vs_high > -15:
        positive.append("Price remains relatively close to its 52-week high.")
    elif price_vs_high < -30:
        negative.append("Price is far below its 52-week high.")

    if avg_sentiment >= 0.05:
        positive.append("Recent news sentiment is positive.")
    elif avg_sentiment <= -0.05:
        negative.append("Recent news sentiment is negative.")
    if MARKET_LORA_ENABLED and _MARKET_LORA_LAST_ERROR:
        negative.append(f"Market LoRA fallback reason: {_MARKET_LORA_LAST_ERROR}")

    return {
        "agent": "Market + Volatility + News Sentiment Agent",
        "ticker": ticker,
        "company": company_name or price_row.get("long_name") or ticker,
        "as_of_date": price_row.get("date"),
        "risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.80 if price_rows else 0.15,
        "claim_type": "MARKET_RULE_BASED",
        "metrics": {
            "rolling_vol_30d": round(rolling_vol_30d, 6),
            "rolling_vol_60d": round(rolling_vol_60d, 6),
            "annualized_volatility_pct": round(annual_vol_pct, 2),
            "beta": round(beta, 4),
            "price_vs_52w_high_pct": round(price_vs_high, 2),
            "ytd_return_pct": round(ytd_return, 2),
            "avg_news_sentiment": round(avg_sentiment, 4),
            "recent_article_count": article_count,
                "market_lora_enabled": MARKET_LORA_ENABLED,
                "market_lora_remote_url_configured": bool(MARKET_LORA_REMOTE_URL),
                "market_lora_fallback_reason": _MARKET_LORA_LAST_ERROR,
            },
        "evidence": [
            f"Annualized 30-day volatility is {annual_vol_pct:.2f}%.",
            f"Beta is {beta:.2f}; price is {price_vs_high:.2f}% versus 52-week high.",
            f"Average recent news sentiment is {avg_sentiment:.3f} across {article_count} articles.",
        ],
        "positive_signals": positive,
        "negative_signals": negative,
        "overall_assessment": (
            f"Market/news risk is {_label_from_score(score)} based on volatility, "
            "beta, price drawdown, YTD return, and news sentiment."
        ),
    }


def run_macro_agent_safe(ticker: str, sector: Optional[str] = None, company_name: Optional[str] = None) -> dict[str, Any]:
    """Use the macro agent if available, otherwise fall back to local rules."""
    ticker = ticker.upper()
    sector = sector or _load_gold_context(ticker).get("sector") or "General"

    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from agents.macro.agent import run_macro_agent

            result = run_macro_agent(ticker=ticker, sector=sector, company_name=company_name or ticker)
            if "macro_risk_score" in result and "risk_score" not in result:
                result["risk_score"] = result["macro_risk_score"]
            result.setdefault("agent", "Macro-Economic Agent")
            result.setdefault("risk_label", _label_from_score(float(result.get("risk_score", 5.0))))
            return result
        except Exception as exc:
            return _local_macro_agent(ticker, sector, company_name, error=str(exc))

    return _local_macro_agent(ticker, sector, company_name)


def _local_macro_agent(
    ticker: str,
    sector: str,
    company_name: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    rows = _read_csv(SILVER_DIR / "silver_macro.csv")
    row = _latest_row(rows, ("date",))

    if not row:
        return {
            "agent": "Macro-Economic Agent",
            "ticker": ticker,
            "company": company_name or ticker,
            "sector": sector,
            "risk_score": 5.0,
            "macro_risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.10,
            "evidence": ["Macro data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    fed = _safe_float(row.get("fed_funds_rate"), 0.0) or 0.0
    cpi = _safe_float(row.get("cpi"), 0.0) or 0.0
    treasury = _safe_float(row.get("treasury_10y"), 0.0) or 0.0
    unemployment = _safe_float(row.get("unemployment"), 0.0) or 0.0
    gdp_growth = _safe_float(row.get("gdp_growth"), 0.0) or 0.0
    yield_curve = _safe_float(row.get("yield_curve_10y2y"))
    vix = _safe_float(row.get("vix"))
    oil = _safe_float(row.get("wti_oil"))

    cpi_prev = None
    if len(rows) > 250:
        cpi_prev = _safe_float(rows[-252].get("cpi"))
    cpi_yoy = ((cpi - cpi_prev) / abs(cpi_prev) * 100.0) if cpi and cpi_prev else 0.0

    score = 3.0
    score += min(max(fed - 2.5, 0.0) / 3.0, 1.0) * 2.0
    score += min(max(cpi_yoy - 2.0, 0.0) / 4.0, 1.0) * 1.5
    score += min(max(unemployment - 4.5, 0.0) / 3.0, 1.0) * 1.5
    score += 1.0 if gdp_growth < 0 else 0.0
    score += 0.5 if treasury > 4.5 else 0.0
    score += 0.75 if yield_curve is not None and yield_curve < 0 else 0.0
    score += min(max(vix - 20.0, 0.0) / 20.0, 1.0) * 1.0 if vix is not None else 0.0
    score += min(max(oil - 85.0, 0.0) / 40.0, 1.0) * 0.5 if oil is not None else 0.0
    score = _clamp_score(score)

    sensitivity_note = "General macro sensitivity."
    if sector in {"Real Estate", "Utilities"}:
        score = _clamp_score(score + 0.75)
        sensitivity_note = "Rate-sensitive sector; higher rates raise refinancing and valuation pressure."
    elif sector in {"Consumer Staples", "Health Care", "Healthcare"}:
        score = _clamp_score(score - 0.35)
        sensitivity_note = "Defensive sector; macro pressure is partially cushioned by stable demand."
    elif sector in {"Information Technology", "Technology", "Consumer Discretionary"}:
        score = _clamp_score(score + 0.35)
        sensitivity_note = "Growth-sensitive sector; higher rates can pressure valuations."

    expanded_macro = []
    if yield_curve is not None:
        expanded_macro.append(f"10Y-2Y yield spread is {yield_curve:.2f}%")
    if vix is not None:
        expanded_macro.append(f"VIX is {vix:.2f}")
    if oil is not None:
        expanded_macro.append(f"WTI oil is {oil:.2f}")

    evidence = [
        f"Fed funds rate is {fed:.2f}%; 10Y Treasury is {treasury:.2f}%.",
        f"Unemployment is {unemployment:.2f}%; GDP growth is {gdp_growth:.2f}%.",
        f"CPI YoY estimate is {cpi_yoy:.2f}%.",
    ]
    if expanded_macro:
        evidence.append("; ".join(expanded_macro) + ".")
    else:
        evidence.append("Expanded macro fields (10Y-2Y spread, VIX, WTI oil) are unavailable; rerun ingestion and cleaning to populate them.")
    if error:
        evidence.append(f"LLM macro agent unavailable, local macro rules used: {error}")

    return {
        "agent": "Macro-Economic Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "sector": sector,
        "as_of_date": row.get("date"),
        "risk_score": score,
        "macro_risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.70,
        "claim_type": "RULE_BASED_ESTIMATE",
        "metrics": {
            "fed_funds_rate": round(fed, 2),
            "cpi": round(cpi, 3),
            "cpi_yoy_pct": round(cpi_yoy, 2),
            "treasury_10y": round(treasury, 2),
            "unemployment": round(unemployment, 2),
            "gdp_growth": round(gdp_growth, 2),
            "yield_curve_10y2y": round(yield_curve, 2) if yield_curve is not None else None,
            "vix": round(vix, 2) if vix is not None else None,
            "wti_oil": round(oil, 2) if oil is not None else None,
        },
        "evidence": evidence,
        "positive_signals": [],
        "negative_signals": [sensitivity_note] if score > 6 else [],
        "sector_sensitivity": sensitivity_note,
        "overall_assessment": f"Macro risk is {_label_from_score(score)} for {sector}.",
    }


def _weighted_critic(agent_outputs: list[dict[str, Any]], query: str = "") -> dict[str, Any]:
    weights = {
        "Fundamental Agent": 0.35,
        "Market + Volatility + News Sentiment Agent": 0.25,
        "Sentiment Agent": 0.15,
        "Macro-Economic Agent": 0.25,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for output in agent_outputs:
        agent = output.get("agent", "")
        score = _safe_float(output.get("risk_score") or output.get("macro_risk_score"))
        if score is None:
            continue
        weight = weights.get(agent, 1.0 / len(agent_outputs))
        weighted_sum += score * weight
        total_weight += weight

    final_score = _clamp_score(weighted_sum / total_weight) if total_weight else 5.0
    final_label = _label_from_score(final_score)
    labels = [o.get("risk_label") for o in agent_outputs if o.get("risk_label")]
    disagreement = len(set(labels)) > 1

    evidence = []
    for output in agent_outputs:
        evidence.append(
            f"{output.get('agent')}: {output.get('risk_label')} "
            f"({output.get('risk_score', output.get('macro_risk_score'))}/10)"
        )

    main_risks = []
    risk_offsets = []
    for output in sorted(
        agent_outputs,
        key=lambda o: _safe_float(o.get("risk_score") or o.get("macro_risk_score"), 0.0) or 0.0,
        reverse=True,
    ):
        for item in output.get("negative_signals", [])[:2]:
            if item not in main_risks:
                main_risks.append(item)
        for item in output.get("positive_signals", [])[:2]:
            if item not in risk_offsets:
                risk_offsets.append(item)

    agent_by_name = {output.get("agent"): output for output in agent_outputs}
    fundamental_label = agent_by_name.get("Fundamental Agent", {}).get("risk_label")
    market_label = agent_by_name.get("Market + Volatility + News Sentiment Agent", {}).get("risk_label")
    sentiment_label = agent_by_name.get("Sentiment Agent", {}).get("risk_label")
    macro_label = agent_by_name.get("Macro-Economic Agent", {}).get("risk_label")

    fundamental_phrase = {
        "LOW": "fundamental strength",
        "MODERATE": "mixed fundamentals",
        "HIGH": "fundamental weakness",
    }.get(fundamental_label, "fundamental evidence")
    market_phrase = {
        "LOW": "low market/news pressure",
        "MODERATE": "moderate market/news pressure",
        "HIGH": "high market/news pressure",
    }.get(market_label, "market/news evidence")
    sentiment_phrase = {
        "LOW": "supportive headline sentiment",
        "MODERATE": "mixed headline sentiment",
        "HIGH": "negative headline sentiment",
    }.get(sentiment_label, "headline sentiment evidence")
    macro_phrase = {
        "LOW": "supportive macro conditions",
        "MODERATE": "moderate macro conditions",
        "HIGH": "macro pressure",
    }.get(macro_label, "macro conditions")

    return {
        "agent": "LLM-Based Critic Agent",
        "critic_type": "llm_based_critic_with_local_fallback",
        "query": query,
        "final_risk_score": final_score,
        "final_risk_label": final_label,
        "confidence": 0.72 if not disagreement else 0.60,
        "disagreement_detected": disagreement,
        "agent_scorecard": evidence,
        "main_risk_drivers": main_risks[:5],
        "risk_offsets": risk_offsets[:5],
        "final_decision": (
            f"Final risk is {final_label} at {final_score}/10. "
            f"The critic combined {fundamental_phrase}, {market_phrase}, "
            f"{sentiment_phrase}, and {macro_phrase}."
        ),
    }


def _llm_critic(agent_outputs: list[dict[str, Any]], query: str) -> Optional[dict[str, Any]]:
    """Optional OpenAI-compatible critic. Returns None when not configured."""
    api_key = os.getenv("CRITIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("CRITIC_BASE_URL")
    model = os.getenv("CRITIC_MODEL", "gpt-4o-mini")
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("CRITIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = base_url or "https://api.deepseek.com"
        model = os.getenv("CRITIC_MODEL", "deepseek-chat")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        prompt = {
            "query": query,
            "agent_outputs": agent_outputs,
            "required_output": {
                "agent": "LLM-Based Critic Agent",
                "final_risk_score": "number 1-10",
                "final_risk_label": "LOW/MODERATE/HIGH",
                "confidence": "0-1",
                "disagreement_detected": "boolean",
                "agent_scorecard": ["short summary per agent"],
                "main_risk_drivers": ["top reasons"],
                "risk_offsets": ["positive reasons"],
                "final_decision": "clear final answer",
            },
        }
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial critic agent. Compare the specialist agent outputs, "
                        "resolve disagreements, and return strict JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            result["critic_type"] = f"llm:{model}"
            return result
    except Exception as exc:
        return {
            "agent": "LLM-Based Critic Agent",
            "critic_type": "llm_failed_fallback_needed",
            "error": str(exc),
        }

    return None


def run_critic_agent(
    ticker: str,
    query: str = "",
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    ticker = ticker.upper()
    gold = _load_gold_context(ticker)
    sector = sector or gold.get("sector") or "General"
    company_name = company_name or ticker

    outputs = [
        run_fundamental_agent(ticker, company_name=company_name),
        run_market_sentiment_agent(ticker, company_name=company_name),
        run_sentiment_agent(ticker, company_name=company_name),
        run_macro_agent_safe(ticker, sector=sector, company_name=company_name),
    ]

    llm_result = _llm_critic(outputs, query) if use_llm else None
    if llm_result and llm_result.get("critic_type") != "llm_failed_fallback_needed":
        final = llm_result
    else:
        final = _weighted_critic(outputs, query=query)
        if llm_result and llm_result.get("error"):
            final["llm_error"] = llm_result["error"]

    report = {
        "ticker": ticker,
        "query": query,
        "agent_outputs": outputs,
        "final_output": final,
    }
    final["policy_evidence_trail"] = build_policy_evidence_trail(report)
    return report


def print_critic_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"CRITIC AGENT REPORT - {report['ticker']}")
    print("=" * 72)

    for output in report["agent_outputs"]:
        print(f"\n[{output.get('agent')}]")
        print(json.dumps(output, indent=2))

    print("\n[FINAL LLM-BASED CRITIC OUTPUT]")
    print(json.dumps(report["final_output"], indent=2))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three agents and a critic agent for one ticker.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL or A")
    parser.add_argument("--query", default="", help="Optional natural-language query for the critic.")
    parser.add_argument("--company", default=None, help="Optional company name.")
    parser.add_argument("--sector", default=None, help="Optional sector.")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM critic instead of weighted fallback.")
    parser.add_argument("--save", default=None, help="Optional JSON output path.")
    parser.add_argument("--pdf", default=None, help="Optional PDF verdict path.")
    args = parser.parse_args()

    report = run_critic_agent(
        ticker=args.ticker,
        query=args.query,
        company_name=args.company,
        sector=args.sector,
        use_llm=args.use_llm,
    )
    print_critic_report(report)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {save_path}")

    if args.pdf:
        from agents.orchestrator.verdict_report import save_pdf_verdict

        pdf_path = save_pdf_verdict(report, args.pdf)
        print(f"Saved PDF verdict to {pdf_path}")


if __name__ == "__main__":
    main()
