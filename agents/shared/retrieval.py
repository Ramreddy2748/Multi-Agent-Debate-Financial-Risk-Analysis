"""
retrieval.py

Phase 1 of the Hybrid RAG architecture: chunk Silver-layer evidence per
ticker (fundamentals, price snapshot/trend, news sentiment), embed it with
chromadb's default sentence embedding model, and expose retrieve() so other
agents (and later the Claude critic) can pull the top-k most relevant
evidence chunks for a ticker + natural-language query.

Build the index once, and again after every Silver refresh:
    python agents/retrieval.py --build-all

Then query it:
    python agents/retrieval.py AAPL --query "leverage and debt risk"
"""

from __future__ import annotations

import argparse
import glob
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "gold" / "vector_store"
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator.critic_agent import _read_csv, _safe_float, _usd_m

COLLECTION_NAME = "financial_documents"

_collection = None


def get_collection():
    """Lazily create the persistent chromadb collection used for all tickers."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils import embedding_functions

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    _collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )
    return _collection


def ticker_universe() -> list[str]:
    """Every ticker that has a Silver price file, derived from filenames on disk."""
    tickers = []
    for path in sorted(glob.glob(str(SILVER_DIR / "silver_prices_*.csv"))):
        tickers.append(Path(path).stem.replace("silver_prices_", ""))
    return tickers


def _fmt(value: Optional[float], suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:,.2f}{suffix}"


def _edgar_chunks(ticker: str) -> list[dict[str, Any]]:
    """One chunk per SEC filing row: cash, debt, income, assets, liabilities, equity."""
    rows = _read_csv(SILVER_DIR / f"silver_edgar_{ticker}.csv")
    chunks = []
    for row in rows:
        end_date = row.get("end_date") or "unknown date"
        company = row.get("company") or ticker
        form = row.get("form") or "filing"
        filed = row.get("filed") or "unknown"
        # long_term_debt (and, defensively, the other _usd_m fields) are stored
        # inconsistently as raw USD or already-USD-millions across filings —
        # reuse critic_agent's normalizer rather than trusting the column name.
        cash = _usd_m(row.get("cash_usd_m"))
        debt = _usd_m(row.get("long_term_debt"))
        net_income = _usd_m(row.get("net_income_usd_m"))
        operating_income = _usd_m(row.get("operating_income_usd_m"))
        assets = _safe_float(row.get("total_assets_usd_m"), 0.0) or 0.0
        liabilities = _safe_float(row.get("total_liabilities_usd_m"), 0.0) or 0.0
        equity = _usd_m(row.get("stockholders_equity_usd_m"))

        text = (
            f"{company} ({ticker}) fundamentals as of {end_date} "
            f"({form} filed {filed}): cash ${cash:,.0f}M, long-term debt "
            f"${debt:,.0f}M, net income ${net_income:,.0f}M, operating "
            f"income ${operating_income:,.0f}M, total assets ${assets:,.0f}M, "
            f"total liabilities ${liabilities:,.0f}M, stockholders equity "
            f"${equity:,.0f}M."
        )
        chunks.append({
            "id": f"{ticker}:fundamentals:{end_date}",
            "text": text,
            "metadata": {"ticker": ticker, "source": "fundamentals", "end_date": str(end_date)},
        })
    return chunks


def _price_chunks(ticker: str) -> list[dict[str, Any]]:
    """A latest-snapshot chunk plus a trailing-30-day trend chunk."""
    rows = _read_csv(SILVER_DIR / f"silver_prices_{ticker}.csv")
    rows = [r for r in rows if r.get("date")]
    rows.sort(key=lambda r: r["date"])
    if not rows:
        return []

    latest = rows[-1]
    company = latest.get("long_name") or ticker
    sector = latest.get("sector") or "unknown sector"
    industry = latest.get("industry") or "unknown industry"
    date = latest.get("date")

    snapshot_text = (
        f"{company} ({ticker}, {sector}/{industry}) latest snapshot as of "
        f"{date}: close ${_fmt(_safe_float(latest.get('close')))}, "
        f"PE {_fmt(_safe_float(latest.get('pe_ratio')))}, "
        f"beta {_fmt(_safe_float(latest.get('beta')))}, "
        f"52-week high/low ${_fmt(_safe_float(latest.get('52w_high')))}/"
        f"${_fmt(_safe_float(latest.get('52w_low')))}, "
        f"price vs 52-week high {_fmt(_safe_float(latest.get('price_vs_52w_high_pct')), '%')}, "
        f"rolling volatility 30d/60d "
        f"{_fmt(_safe_float(latest.get('rolling_vol_30d')))}/"
        f"{_fmt(_safe_float(latest.get('rolling_vol_60d')))}, "
        f"market cap ${_fmt(_safe_float(latest.get('market_cap_usd_m')))}M, "
        f"debt/equity {_fmt(_safe_float(latest.get('debt_to_equity')))}."
    )
    chunks = [{
        "id": f"{ticker}:price_snapshot",
        "text": snapshot_text,
        "metadata": {"ticker": ticker, "source": "price_snapshot", "date": str(date)},
    }]

    window = rows[-30:]
    returns = [x for x in (_safe_float(r.get("daily_return")) for r in window) if x is not None]
    closes = [x for x in (_safe_float(r.get("close")) for r in window) if x is not None]
    if returns and len(closes) >= 2:
        mean_return = statistics.mean(returns)
        vol = statistics.pstdev(returns) if len(returns) >= 2 else 0.0
        pct_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0.0
        trend_text = (
            f"{ticker} 30-day trend ending {date}: mean daily return "
            f"{mean_return:.3%}, realized volatility {vol:.3%}, price "
            f"change over the window {pct_change:+.2f}%."
        )
        chunks.append({
            "id": f"{ticker}:price_trend:{date}",
            "text": trend_text,
            "metadata": {"ticker": ticker, "source": "price_trend", "date": str(date)},
        })
    return chunks


def _news_chunks(ticker: str) -> list[dict[str, Any]]:
    """One chunk per (ticker, date) row in the shared news sentiment file, if any."""
    rows = _read_csv(SILVER_DIR / "silver_news_sentiment.csv")
    rows = [r for r in rows if str(r.get("ticker", "")).upper() == ticker.upper()]
    chunks = []
    for row in rows:
        date = row.get("date") or "unknown date"
        sentiment_mean = _safe_float(row.get("sentiment_mean"), 0.0) or 0.0
        sentiment_std = _safe_float(row.get("sentiment_std"), 0.0) or 0.0
        article_count = int(_safe_float(row.get("article_count"), 0) or 0)
        text = (
            f"{ticker} news sentiment on {date}: mean sentiment "
            f"{sentiment_mean:.3f} (std {sentiment_std:.3f}) across "
            f"{article_count} articles."
        )
        chunks.append({
            "id": f"{ticker}:news:{date}",
            "text": text,
            "metadata": {"ticker": ticker, "source": "news", "date": str(date)},
        })
    return chunks


def build_index(tickers: Optional[list[str]] = None) -> int:
    """(Re)index the given tickers (default: every ticker found in data/silver/)."""
    tickers = [t.upper() for t in tickers] if tickers else ticker_universe()
    collection = get_collection()

    total = 0
    for ticker in tickers:
        chunks = _edgar_chunks(ticker) + _price_chunks(ticker) + _news_chunks(ticker)
        if not chunks:
            continue
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        total += len(chunks)
    return total


def retrieve(ticker: str, query: str, k: int = 5) -> list[dict[str, Any]]:
    """Top-k evidence chunks for a ticker, ranked by embedding similarity to query."""
    collection = get_collection()
    result = collection.query(
        query_texts=[query],
        n_results=k,
        where={"ticker": ticker.upper()},
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    return [
        {"text": text, "distance": distance, **meta}
        for text, meta, distance in zip(documents, metadatas, distances)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the Silver-layer retrieval index.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol to query, for example AAPL")
    parser.add_argument("--query", default=None, help="Natural-language query to retrieve evidence for.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--build-all", action="store_true", help="(Re)build the index for every ticker in data/silver/.")
    args = parser.parse_args()

    if args.build_all:
        total = build_index()
        print(f"Indexed {total} chunks across {len(ticker_universe())} tickers.")
        return

    if not args.ticker or not args.query:
        parser.error("Provide a ticker and --query, or use --build-all.")

    results = retrieve(args.ticker, args.query, k=args.k)
    if not results:
        print(f"No chunks found for {args.ticker.upper()}. Run --build-all first.")
        return

    print(f"Top {len(results)} chunks for {args.ticker.upper()} — query: {args.query!r}\n")
    for i, item in enumerate(results, 1):
        when = item.get("end_date") or item.get("date") or "-"
        print(f"[{i}] distance={item['distance']:.4f} source={item.get('source')} date={when}")
        print(f"    {item['text']}\n")


if __name__ == "__main__":
    main()
