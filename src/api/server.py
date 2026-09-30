"""
FastAPI backend for the stock-risk application.

Run:
    uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8766
"""

from __future__ import annotations

import csv
import json
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
from agents.critic_agent import run_critic_agent
from agents.debate_state_machine import run_debate_state_machine
from agents.monitoring import log_verdict_run
from agents.verdict_report import save_json_verdict, save_pdf_verdict


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


def _company_payload(row: dict[str, str]) -> dict[str, Any]:
    score = _safe_float(row.get("composite_risk"))
    return {
        **row,
        "composite_risk_num": score,
        "ann_volatility_pct_num": _safe_float(row.get("ann_volatility_pct")),
        "max_drawdown_pct_num": _safe_float(row.get("max_drawdown_pct")),
        "has_verdict": _verdict_path(row.get("ticker", "")).exists(),
        "has_debate": _debate_path(row.get("ticker", "")).exists(),
        "has_price_history": _price_path(row.get("ticker", "")).exists(),
    }


def _gold_fallback_verdict(row: dict[str, str]) -> dict[str, Any]:
    ticker = row.get("ticker", "UNKNOWN")
    score = _safe_float(row.get("composite_risk")) or 5.0
    label = row.get("risk_label") or "MODERATE"
    decision = {
        "LOW": f"{ticker} is a lower-risk candidate for further research based on the Gold feature table.",
        "MODERATE": f"{ticker} should stay on the watchlist pending deeper review of volatility, drawdown, and sector exposure.",
        "HIGH": f"{ticker} is high risk in the Gold feature table and should require review before investment consideration.",
    }.get(label, f"{ticker} should be reviewed before investment consideration.")
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
            f"Fundamental risk {row.get('fundamental_risk', '-')}/10",
            f"Volatility risk {row.get('volatility_risk', '-')}/10",
            f"Sentiment risk {row.get('sentiment_risk', '-')}/10",
            f"Macro risk {row.get('macro_risk', '-')}/10",
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


@app.get("/api/companies")
def companies() -> dict[str, Any]:
    rows = [_company_payload(row) for row in _gold_rows()]
    rows.sort(key=lambda row: row.get("composite_risk_num") or 0.0, reverse=True)
    return {"count": len(rows), "companies": rows}


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
    verdict_path = _verdict_path(ticker)
    debate_path = _debate_path(ticker)
    pdf_path = _pdf_path(ticker)
    verdict = _read_json(verdict_path)
    debate = _read_json(debate_path)
    debate_final = (debate or {}).get("final_output")
    critic_final = (verdict or {}).get("final_output")
    final = debate_final or critic_final or _gold_fallback_verdict(row)
    active_type = "debate" if debate_final else "critic" if critic_final else "fallback"
    return {
        "ticker": ticker.upper(),
        "company": _company_payload(row),
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
            use_llm=request.use_llm,
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
