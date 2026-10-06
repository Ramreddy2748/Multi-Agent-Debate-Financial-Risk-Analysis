"""
FastAPI backend for the stock-risk application.

Run:
    uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8766
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agents.orchestrator.critic_agent import predict_market_lora_payload, run_critic_agent
from agents.orchestrator.debate_state_machine import run_debate_state_machine
from agents.orchestrator.monitoring import log_verdict_run
from agents.orchestrator.verdict_report import save_json_verdict, save_pdf_verdict


FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
GOLD_PATH = DATA_DIR / "gold" / "gold_risk_scores_ALL.csv"
MONITORING_PATH = DATA_DIR / "gold" / "monitoring_runs.jsonl"


class VerdictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    company: str | None = None
    sector: str | None = None
    query: str = ""
    mode: Literal["critic", "debate"] = "critic"
    use_llm: bool = False


class CompareRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=5)


class MarketLoraRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    annualized_volatility: float = 0.0
    beta: float = 1.0
    max_drawdown: float = 0.0
    sentiment_mean: float = 0.0
    article_count: int = 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _gold_rows() -> list[dict[str, str]]:
    return _read_csv(GOLD_PATH)


def _find_company(ticker: str) -> dict[str, str]:
    ticker = ticker.upper()
    for row in _gold_rows():
        if row.get("ticker", "").upper() == ticker:
            return row
    raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in Gold table")


def _verdict_path(ticker: str) -> Path:
    return DATA_DIR / "verdicts" / f"{ticker.upper()}_risk_verdict.json"


def _debate_path(ticker: str) -> Path:
    return DATA_DIR / "verdicts" / f"{ticker.upper()}_debate_verdict.json"


def _pdf_path(ticker: str) -> Path:
    return DATA_DIR / "verdicts" / f"{ticker.upper()}_risk_verdict.pdf"


def _price_path(ticker: str) -> Path:
    return DATA_DIR / "silver" / f"silver_prices_{ticker.upper()}.csv"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed
    except (TypeError, ValueError):
        return None


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _latest_trainer_state(model_dir: Path) -> dict[str, Any] | None:
    states = sorted(model_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        direct = model_dir / "trainer_state.json"
        states = [direct] if direct.exists() else []
    best_state = None
    best_step = -1
    for path in states:
        data = _read_json(path)
        if not data:
            continue
        step = int(data.get("global_step") or 0)
        if step >= best_step:
            best_step = step
            best_state = data
    return best_state


def _training_series(model_dir: Path) -> dict[str, Any]:
    state = _latest_trainer_state(model_dir)
    if not state:
        return {"available": False, "train_loss": [], "eval_loss": [], "summary": {}}

    train_loss = []
    eval_loss = []
    for item in state.get("log_history", []):
        step = item.get("step")
        epoch = item.get("epoch")
        if item.get("loss") is not None:
            train_loss.append({
                "step": step,
                "epoch": epoch,
                "loss": round(float(item["loss"]), 6),
            })
        if item.get("eval_loss") is not None:
            eval_loss.append({
                "step": step,
                "epoch": epoch,
                "loss": round(float(item["eval_loss"]), 6),
            })

    summary = {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "max_steps": state.get("max_steps"),
        "train_batch_size": state.get("train_batch_size"),
        "best_metric": state.get("best_metric"),
    }
    return {
        "available": True,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "summary": summary,
    }


def _model_registry() -> list[dict[str, Any]]:
    finbert_dir = PROJECT_ROOT / "models" / "fundamental_agent_finbert"
    lora_dir = PROJECT_ROOT / "models" / "market_sentiment_lora"
    mistral_lora_dir = PROJECT_ROOT / "models" / "market_mistral_lora"
    configured_lora_dir = Path(os.getenv("MARKET_LORA_ADAPTER_DIR", lora_dir))
    finbert_metrics = _read_json(finbert_dir / "training_metrics.json") or {}

    return [
        {
            "agent": "Fundamental Agent",
            "model": "ProsusAI/finbert fine-tuned classifier",
            "backend": "local_transformers",
            "status": "active" if finbert_dir.exists() else "fallback_rules",
            "claim_type": "FINBERT_MODEL_INFERENCE",
            "data": "SEC EDGAR Silver fundamentals + Gold risk labels",
            "train_examples": _count_jsonl(DATA_DIR / "gold" / "fundamental_finetune_data.jsonl"),
            "model_dir": "models/fundamental_agent_finbert",
            "metrics": finbert_metrics,
            "training": _training_series(finbert_dir),
        },
        {
            "agent": "Market + Volatility + News Sentiment Agent",
            "model": os.getenv("MARKET_LORA_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct") + " + LoRA",
            "backend": "remote_colab" if os.getenv("MARKET_LORA_REMOTE_URL") else "local_transformers",
            "status": "active" if os.getenv("MARKET_LORA_ENABLED", "0").lower() in {"1", "true", "yes"} else "fallback_rules",
            "claim_type": "LORA_MODEL_INFERENCE",
            "data": "Yahoo price features + NewsAPI sentiment + Gold risk labels",
            "train_examples": _count_jsonl(DATA_DIR / "gold" / "market_sentiment_finetune_data.jsonl"),
            "model_dir": str(configured_lora_dir.relative_to(PROJECT_ROOT) if configured_lora_dir.is_relative_to(PROJECT_ROOT) else configured_lora_dir),
            "metrics": {},
            "training": _training_series(configured_lora_dir),
        },
        {
            "agent": "Market + Volatility + News Sentiment Agent",
            "model": "mistralai/Mistral-7B-Instruct-v0.3 + LoRA",
            "backend": "local_transformers_or_remote_colab",
            "status": "available" if mistral_lora_dir.exists() else "not_trained_yet",
            "claim_type": "LORA_MODEL_INFERENCE",
            "data": "Yahoo price features + NewsAPI sentiment + Gold risk labels",
            "train_examples": _count_jsonl(DATA_DIR / "gold" / "market_sentiment_finetune_data.jsonl"),
            "model_dir": "models/market_mistral_lora",
            "metrics": {},
            "training": _training_series(mistral_lora_dir),
        },
        {
            "agent": "Sentiment Agent",
            "model": os.getenv("SENTIMENT_FINBERT_MODEL", "ProsusAI/finbert"),
            "backend": "local_transformers",
            "status": "active_if_news_available",
            "claim_type": "FINBERT_SENTIMENT_INFERENCE",
            "data": "NewsAPI headline title/description text per ticker",
            "train_examples": 0,
            "model_dir": "downloaded from Hugging Face cache",
            "metrics": {},
            "training": {"available": False, "train_loss": [], "eval_loss": [], "summary": {}},
        },
        {
            "agent": "Macro-Economic Agent",
            "model": "DeepSeek macro reasoning when DEEPSEEK_API_KEY is configured",
            "backend": "api_or_rules",
            "status": "llm_configured" if os.getenv("DEEPSEEK_API_KEY") else "rule_based_active",
            "claim_type": "RULE_BASED_ESTIMATE / MACRO_LLM_INFERENCE",
            "data": "FRED macro series: rates, CPI, unemployment, GDP, yield curve, VIX, WTI oil",
            "train_examples": 0,
            "model_dir": "",
            "metrics": {},
            "training": {"available": False, "train_loss": [], "eval_loss": [], "summary": {}},
        },
        {
            "agent": "Critic / Orchestrator",
            "model": os.getenv("CLAUDE_MODEL") or os.getenv("CRITIC_MODEL", "weighted critic; optional Claude/OpenAI-compatible LLM"),
            "backend": "weighted_rules_or_api",
            "status": "llm_configured" if (
                os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("CLAUDE_API_KEY")
                or os.getenv("CRITIC_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY")
            ) else "weighted_rules_active",
            "claim_type": "llm_based_critic_with_local_fallback",
            "data": "Specialist agent outputs, confidence, disagreement flags, and evidence trail",
            "train_examples": 0,
            "model_dir": "",
            "metrics": {},
            "training": {"available": False, "train_loss": [], "eval_loss": [], "summary": {}},
        },
    ]


def _price_metadata(ticker: str) -> dict[str, str]:
    """Company name/sector/industry live in the Silver prices file, not the
    Gold table (which is missing `company` entirely and has a blank `sector`
    for a few tickers ingested before the sector-labeled batch run)."""
    rows = _read_csv(_price_path(ticker))
    if not rows:
        return {}
    latest = rows[-1]
    return {
        "company": latest.get("long_name") or "",
        "sector": latest.get("sector") or "",
        "industry": latest.get("industry") or "",
    }


def _company_payload(row: dict[str, str]) -> dict[str, Any]:
    ticker = row.get("ticker", "")
    score = _safe_float(row.get("composite_risk"))
    meta = _price_metadata(ticker)
    return {
        **row,
        "company": row.get("company") or meta.get("company") or ticker,
        "sector": row.get("sector") or meta.get("sector") or "Unknown sector",
        "industry": row.get("industry") or meta.get("industry") or "",
        "composite_risk_num": score,
        "ann_volatility_pct_num": _safe_float(row.get("ann_volatility_pct")),
        "max_drawdown_pct_num": _safe_float(row.get("max_drawdown_pct")),
        "has_verdict": _verdict_path(ticker).exists(),
        "has_debate": _debate_path(ticker).exists(),
        "has_price_history": _price_path(ticker).exists(),
    }


def _gold_fallback_verdict(row: dict[str, str]) -> dict[str, Any]:
    ticker = row.get("ticker", "UNKNOWN")
    name = row.get("company") or ticker
    score = _safe_float(row.get("composite_risk")) or 5.0
    label = row.get("risk_label") or "MODERATE"
    def component_driver(label_text: str, key: str) -> str:
        value = _safe_float(row.get(key))
        return f"{label_text} {value:.2f}/10" if value is not None else f"{label_text} -/10"

    decision = {
        "LOW": f"{name} ({ticker}) is a lower-risk candidate for further research based on the Gold feature table.",
        "MODERATE": f"{name} ({ticker}) should stay on the watchlist pending deeper review of volatility, drawdown, and sector exposure.",
        "HIGH": f"{name} ({ticker}) is high risk in the Gold feature table and should require review before investment consideration.",
    }.get(label, f"{name} ({ticker}) should be reviewed before investment consideration.")
    return {
        "agent": "Gold Table Fallback",
        "critic_type": "gold_table_fallback",
        "final_risk_score": round(score, 2),
        "final_risk_label": label,
        "confidence": None,
        "disagreement_detected": False,
        "requires_human_review": label == "HIGH",
        "final_decision": decision,
        "main_risk_drivers": [
            component_driver("Fundamental risk", "fundamental_risk"),
            component_driver("Volatility risk", "volatility_risk"),
            component_driver("Sentiment risk", "sentiment_risk"),
            component_driver("Macro risk", "macro_risk"),
        ],
        "risk_offsets": [],
        "policy_evidence_trail": [
            {
                "policy_id": "A",
                "policy_name": "Data provenance",
                "status": "PASS",
                "evidence": ["Fallback verdict generated from data/gold/gold_risk_scores_ALL.csv."],
            },
            {
                "policy_id": "H",
                "policy_name": "Confidence and fallback disclosure",
                "status": "WARN",
                "evidence": ["No generated per-ticker agent verdict JSON was found."],
            },
        ],
    }


def _compare_companies(tickers: list[str]) -> dict[str, Any]:
    unique_tickers = []
    for ticker in tickers:
        normalized = ticker.upper().strip()
        if normalized and normalized not in unique_tickers:
            unique_tickers.append(normalized)

    if len(unique_tickers) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two different tickers to compare")

    companies_payload = []
    for ticker in unique_tickers:
        row = _company_payload(_find_company(ticker))
        final = _read_json(_debate_path(ticker)) or _read_json(_verdict_path(ticker))
        final_output = (final or {}).get("final_output") or _gold_fallback_verdict(row)
        companies_payload.append({
            "ticker": ticker,
            "company": row,
            "final_output": final_output,
            "score": _safe_float(final_output.get("final_risk_score")) or row.get("composite_risk_num"),
            "label": final_output.get("final_risk_label") or row.get("risk_label"),
            "confidence": _safe_float(final_output.get("confidence")),
            "has_generated_verdict": final is not None,
        })

    ranked = sorted(companies_payload, key=lambda item: item.get("score") if item.get("score") is not None else 99.0)
    lowest = ranked[0]
    highest = ranked[-1]
    score_delta = None
    if lowest.get("score") is not None and highest.get("score") is not None:
        score_delta = round(float(highest["score"]) - float(lowest["score"]), 2)

    return {
        "tickers": unique_tickers,
        "count": len(companies_payload),
        "companies": companies_payload,
        "summary": {
            "lower_risk_ticker": lowest["ticker"],
            "higher_risk_ticker": highest["ticker"],
            "score_delta": score_delta,
            "decision": (
                f"{lowest['ticker']} has the lower current project risk score"
                + (f" by {score_delta:.2f} points." if score_delta is not None else ".")
            ),
        },
    }


app = FastAPI(
    title="Financial Risk Multi-Agent API",
    version="1.0.0",
    description="Production-style API for company risk scores, verdicts, debate, and monitoring lineage.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8766", "http://localhost:8766"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/frontend/index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "gold_table": GOLD_PATH.exists(), "companies": len(_gold_rows())}


@app.get("/api/models/training")
def model_training() -> dict[str, Any]:
    models = _model_registry()
    return {
        "count": len(models),
        "models": models,
        "notes": [
            "Training loss is available only for locally saved fine-tuned models with trainer_state.json.",
            "API-only or rule-based agents do not have local training loss curves.",
        ],
    }


@app.post("/api/agents/market-lora/predict")
def market_lora_predict(request: MarketLoraRequest) -> dict[str, Any]:
    return predict_market_lora_payload(request.model_dump())


@app.get("/api/companies")
def companies() -> dict[str, Any]:
    rows = [_company_payload(row) for row in _gold_rows()]
    rows.sort(key=lambda row: row.get("composite_risk_num") or 0.0, reverse=True)
    return {"count": len(rows), "companies": rows}


@app.get("/api/compare")
def compare_get(tickers: str) -> dict[str, Any]:
    return _compare_companies(tickers.split(","))


@app.post("/api/compare")
def compare_post(request: CompareRequest) -> dict[str, Any]:
    return _compare_companies(request.tickers)


@app.get("/api/companies/{ticker}")
def company(ticker: str) -> dict[str, Any]:
    row = _find_company(ticker)
    return {"company": _company_payload(row)}


@app.get("/api/companies/{ticker}/prices")
def company_prices(ticker: str) -> dict[str, Any]:
    path = _price_path(ticker)
    if not path.exists():
        return {"ticker": ticker.upper(), "count": 0, "prices": []}
    rows = _read_csv(path)
    prices = [
        {
            "date": row.get("date"),
            "close": _safe_float(row.get("close")),
            "daily_return": _safe_float(row.get("daily_return")),
            "rolling_vol_30d": _safe_float(row.get("rolling_vol_30d")),
        }
        for row in rows
    ]
    return {"ticker": ticker.upper(), "count": len(prices), "prices": prices}


@app.get("/api/verdicts/{ticker}")
def get_verdict(ticker: str) -> dict[str, Any]:
    row = _find_company(ticker)
    company_payload = _company_payload(row)
    verdict_path = _verdict_path(ticker)
    debate_path = _debate_path(ticker)
    pdf_path = _pdf_path(ticker)
    verdict = _read_json(verdict_path)
    debate = _read_json(debate_path)
    debate_final = (debate or {}).get("final_output")
    critic_final = (verdict or {}).get("final_output")
    final = debate_final or critic_final or _gold_fallback_verdict(company_payload)
    active_type = "debate" if debate_final else "critic" if critic_final else "fallback"
    return {
        "ticker": ticker.upper(),
        "company": company_payload,
        "verdict": verdict,
        "debate": debate,
        "final_output": final,
        "active_verdict_type": active_type,
        "fallback": verdict is None and debate is None,
        "json_url": f"/data/verdicts/{ticker.upper()}_risk_verdict.json" if verdict_path.exists() else None,
        "debate_url": f"/data/verdicts/{ticker.upper()}_debate_verdict.json" if debate_path.exists() else None,
        "pdf_url": f"/data/verdicts/{ticker.upper()}_risk_verdict.pdf" if pdf_path.exists() else None,
    }


@app.post("/api/verdict")
def generate_verdict(request: VerdictRequest) -> dict[str, Any]:
    ticker = request.ticker.upper().strip()
    row = _find_company(ticker)
    sector = request.sector or row.get("sector") or "General"
    company_name = request.company or row.get("company") or ticker

    if request.mode == "debate":
        report = run_debate_state_machine(
            ticker=ticker,
            query=request.query,
            company_name=company_name,
            sector=sector,
        )
        path = _debate_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        artifact_paths = [str(path)]
        pdf_path = None
        run_type = "debate_state_machine"
    else:
        report = run_critic_agent(
            ticker=ticker,
            query=request.query,
            company_name=company_name,
            sector=sector,
            use_llm=request.use_llm or os.getenv("CRITIC_LLM_ENABLED", "0").lower() in {"1", "true", "yes"},
        )
        json_path = Path(save_json_verdict(report))
        pdf_path = Path(save_pdf_verdict(report))
        artifact_paths = [str(json_path), str(pdf_path)]
        run_type = "critic_verdict"

    tracking = log_verdict_run(report, run_type=run_type, artifact_paths=artifact_paths)
    return {
        "ok": True,
        "ticker": ticker,
        "mode": request.mode,
        "report": report,
        "tracking": tracking,
        "json_url": f"/data/verdicts/{ticker}_risk_verdict.json",
        "pdf_url": f"/data/verdicts/{ticker}_risk_verdict.pdf" if pdf_path else None,
    }


@app.get("/api/monitoring")
def monitoring(ticker: str | None = None) -> dict[str, Any]:
    rows = _read_jsonl(MONITORING_PATH)
    if ticker:
        rows = [row for row in rows if row.get("ticker") == ticker.upper()]
    return {"count": len(rows), "records": rows}


@app.get("/api/artifacts/{ticker}/pdf")
def verdict_pdf(ticker: str) -> FileResponse:
    path = _pdf_path(ticker)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No PDF verdict found for {ticker.upper()}")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
