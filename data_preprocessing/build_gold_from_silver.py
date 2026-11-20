"""Rebuild the Gold risk table from all persisted Silver price files.

This is useful after a partial/test ingestion run. It does not call Yahoo,
SEC, FRED, or NewsAPI; it recomputes analytics from the Silver layer already
stored locally.

Usage:
    python data_preprocessing/build_gold_from_silver.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data_preprocessing import eda


log = logging.getLogger("build_gold_from_silver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_prices(silver_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict[str, str]]]:
    price_data: dict[str, pd.DataFrame] = {}
    sectors: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}
    for path in sorted(silver_dir.glob("silver_prices_*.csv")):
        ticker = path.stem.removeprefix("silver_prices_").upper()
        if not ticker:
            continue
        frame = pd.read_csv(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date")
        if frame.empty or "close" not in frame.columns:
            log.warning("Skipping %s: no usable close data", path)
            continue
        price_data[ticker] = frame
        raw_sector = frame.iloc[-1].get("sector")
        sector = str(raw_sector).strip() if pd.notna(raw_sector) else "Unknown"
        sectors[ticker] = sector or "Unknown"
        metadata[ticker] = {
            "company": str(frame.iloc[-1].get("long_name") or ticker),
            "sub_industry": str(frame.iloc[-1].get("industry") or ""),
        }
    return price_data, sectors, metadata


def rebuild_gold(silver_dir: Path, output_dir: Path, news_path: Path) -> pd.DataFrame:
    price_data, sectors, metadata = _load_prices(silver_dir)
    if not price_data:
        raise RuntimeError(f"No Silver price files found in {silver_dir}")

    stats = eda.compute_statistics(price_data)
    news = None
    if news_path.exists():
        news = pd.read_csv(news_path)
        log.info("Loaded Silver news rows: %d", len(news))

    gold = eda.compute_risk_scores(stats, news)
    gold = gold.reset_index().rename(columns={"index": "ticker"})
    gold["ticker"] = gold["ticker"].astype(str).str.upper().str.strip()
    gold["sector"] = gold["ticker"].map(sectors).fillna("Unknown")
    gold["as_of_date"] = pd.Timestamp.today().date().isoformat()
    gold["batch"] = "silver_rebuild"
    gold = gold.drop_duplicates(subset=["ticker"], keep="last").sort_values("ticker")

    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "gold_risk_scores_ALL.csv"
    gold.to_csv(master_path, index=False)

    universe = pd.DataFrame({"ticker": sorted(price_data)})
    universe["company"] = universe["ticker"].map(lambda ticker: metadata[ticker]["company"])
    universe["sector"] = universe["ticker"].map(sectors)
    universe["sub_industry"] = universe["ticker"].map(lambda ticker: metadata[ticker]["sub_industry"])
    universe["index"] = "SILVER_REBUILD"
    universe.to_csv(output_dir / "ticker_universe.csv", index=False)

    for sector, sector_frame in gold.groupby("sector", dropna=False):
        safe_sector = str(sector).replace(" ", "_").replace("/", "_")
        sector_path = output_dir / f"gold_risk_full_{safe_sector}.csv"
        sector_frame.to_csv(sector_path, index=False)

    log.info("Rebuilt Gold master: %s (%d companies)", master_path, len(gold))
    log.info("Created %d sector Gold files", gold["sector"].nunique())
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Gold risk scores from all Silver price files.")
    parser.add_argument("--silver-dir", default=config.LOCAL_SILVER)
    parser.add_argument("--output-dir", default=config.LOCAL_GOLD)
    parser.add_argument(
        "--news-file",
        default=str(Path(config.LOCAL_SILVER) / "silver_news_sentiment.csv"),
    )
    args = parser.parse_args()
    rebuild_gold(Path(args.silver_dir), Path(args.output_dir), Path(args.news_file))


if __name__ == "__main__":
    main()
