"""
pipeline_large_scale.py
─────────────────────────────────────────────────────────────────────────────
LARGE-SCALE PIPELINE ORCHESTRATOR
Runs the full Bronze → Silver → Gold pipeline for S&P 500 + NASDAQ-100
(~600 companies) using sector-by-sector batching.

Key features vs pipeline.py:
  - Fetches live S&P 500 + NASDAQ-100 ticker lists automatically
  - Processes companies sector by sector (11 sectors)
  - Each sector split into batches of 50 for rate-limit safety
  - Progress tracking with resume capability (skips already-done tickers)
  - Per-sector GCS folders and BigQuery partitions
  - Detailed run report saved at the end

Usage:
  # Full run — all 600 companies, all sectors
  python pipeline_large_scale.py

  # Run specific sectors only
  python pipeline_large_scale.py --sectors "Information Technology" "Health Care"

  # Run with GCS upload
  python pipeline_large_scale.py --gcs

  # Resume interrupted run (skips tickers already in Bronze)
  python pipeline_large_scale.py --resume --gcs

  # Test run — first 10 companies per sector only
  python pipeline_large_scale.py --test
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import ingest
import clean
import eda
from tickers import get_all_tickers, get_sector_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_large_scale.log", mode="a"),
    ]
)
log = logging.getLogger("pipeline_large")

# ─── Progress tracking file ───────────────────────────────────────────────────
PROGRESS_FILE = "pipeline_progress.json"


def _load_progress() -> dict:
    """Load progress from previous run (for resume capability)."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_tickers": [], "failed_tickers": [], "completed_sectors": []}

def _save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def _already_fetched(ticker: str) -> bool:
    """Check if Bronze CSV already exists for this ticker."""
    import glob
    pattern = os.path.join(config.LOCAL_BRONZE, f"yahoo_{ticker}_*.csv")
    return len(glob.glob(pattern)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sector_batch(
    sector:     str,
    batch_num:  int,
    total_batches: int,
    tickers:    list,
    upload_gcs: bool,
    resume:     bool,
    progress:   dict,
) -> dict:
    """
    Run full pipeline (ingest → clean → EDA) for one batch of tickers.
    Returns summary of what succeeded/failed.
    """
    log.info(f"\n{'─' * 60}")
    log.info(f"  Sector : {sector}")
    log.info(f"  Batch  : {batch_num}/{total_batches}")
    log.info(f"  Tickers: {tickers}")
    log.info(f"{'─' * 60}")

    # Filter out already-completed tickers if resuming
    if resume:
        skip     = [t for t in tickers if t in progress["completed_tickers"]]
        tickers  = [t for t in tickers if t not in progress["completed_tickers"]]
        if skip:
            log.info(f"  Skipping {len(skip)} already-completed: {skip}")
        if not tickers:
            log.info("  All tickers in this batch already done — skipping")
            return {"succeeded": skip, "failed": []}

    succeeded = []
    failed    = []

    # ── INGESTION ─────────────────────────────────────────────────────────
    try:
        raw_data = ingest.run_ingestion(tickers=tickers, upload_gcs=upload_gcs)
    except Exception as e:
        log.error(f"  Ingestion failed for batch: {e}")
        return {"succeeded": [], "failed": tickers}

    # Track which tickers actually got price data
    fetched_tickers = list(raw_data.get("prices", {}).keys())
    failed.extend([t for t in tickers if t not in fetched_tickers])

    if not fetched_tickers:
        log.warning("  No price data fetched for this batch")
        return {"succeeded": [], "failed": tickers}

    # ── CLEANING ──────────────────────────────────────────────────────────
    try:
        clean_data = clean.run_cleaning(raw_data, upload_gcs=upload_gcs)
    except Exception as e:
        log.error(f"  Cleaning failed: {e}")
        clean_data = {"prices": raw_data.get("prices", {}),
                      "edgar": {}, "macro": pd.DataFrame(), "news": pd.DataFrame()}

    # ── EDA (statistics + risk scores only, skip heavy plots for scale) ───
    try:
        price_data = clean_data.get("prices", {})
        news_df    = clean_data.get("news", pd.DataFrame())

        if price_data:
            stats_df = eda.compute_statistics(price_data)
            risk_df  = eda.compute_risk_scores(
                stats_df,
                news_df if not news_df.empty else None
            )

            # Tag with sector
            risk_df["sector"]     = sector
            risk_df["as_of_date"] = pd.Timestamp.today().date()
            risk_df["batch"]      = f"{sector}_batch{batch_num}"

            # Save Gold CSV per sector batch
            Path(config.LOCAL_GOLD).mkdir(parents=True, exist_ok=True)
            safe_sector = sector.replace(" ", "_").replace("/", "_")
            gold_file   = os.path.join(
                config.LOCAL_GOLD,
                f"gold_risk_{safe_sector}_batch{batch_num}.csv"
            )
            risk_df.to_csv(gold_file)
            log.info(f"  ✔ Gold saved: {gold_file}")

            # Upload Gold to GCS
            if upload_gcs:
                eda._upload_gold_to_gcs = True
                try:
                    from google.cloud import storage
                    client  = storage.Client(project=config.GCP_PROJECT_ID)
                    bucket  = client.bucket(config.GCS_BUCKET)
                    gcs_key = f"gold/sectors/{safe_sector}/batch{batch_num}.csv"
                    bucket.blob(gcs_key).upload_from_filename(gold_file)
                    log.info(f"  ☁  GCS: gs://{config.GCS_BUCKET}/{gcs_key}")
                except Exception as e:
                    log.warning(f"  GCS upload skipped: {e}")

            succeeded.extend(list(risk_df.index))

    except Exception as e:
        log.error(f"  EDA failed: {e}")
        succeeded.extend(fetched_tickers)

    return {"succeeded": succeeded, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# MERGE ALL GOLD FILES INTO BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_load_bigquery():
    """
    Combine all sector Gold CSVs into one master table and load to BigQuery.
    """
    import glob
    gold_files = glob.glob(os.path.join(config.LOCAL_GOLD, "gold_risk_*.csv"))

    if not gold_files:
        log.warning("No Gold files found to merge")
        return

    log.info(f"\n[Merge] Combining {len(gold_files)} Gold files...")
    dfs = []
    for f in gold_files:
        try:
            dfs.append(pd.read_csv(f, index_col=0))
        except Exception as e:
            log.error(f"  Failed to read {f}: {e}")

    if not dfs:
        return

    master = pd.concat(dfs, ignore_index=False)
    master = master[~master.index.duplicated(keep="last")]  # deduplicate tickers

    master_path = os.path.join(config.LOCAL_GOLD, "gold_risk_scores_ALL.csv")
    master.to_csv(master_path)
    log.info(f"  ✔ Master Gold table: {master_path}  ({len(master)} companies)")

    # Load to BigQuery
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=config.GCP_PROJECT_ID)
        job = client.load_table_from_dataframe(
            master.reset_index(),
            config.BQ_TABLE_RISK,
            job_config=bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                autodetect=True,
            )
        )
        job.result()
        log.info(f"  ☁  BigQuery loaded: {config.BQ_TABLE_RISK} ({len(master)} rows)")
    except ImportError:
        log.warning("  google-cloud-bigquery not installed — skipping BQ load")
    except Exception as e:
        log.error(f"  BigQuery load failed: {e}")

    return master


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────

def run(
    sectors:    list = None,
    upload_gcs: bool = False,
    resume:     bool = False,
    test_mode:  bool = False,
    batch_size: int  = 50,
    universe_csv: str = None,
):
    start_time = time.time()
    run_date   = datetime.today().strftime("%Y-%m-%d %H:%M")

    log.info("╔" + "═" * 68 + "╗")
    log.info("║  LARGE-SCALE FINANCIAL RISK PIPELINE                             ║")
    log.info("║  S&P 500 + NASDAQ-100 | Sector Batching | TEAM-14, DATA 266      ║")
    log.info("╚" + "═" * 68 + "╝")
    log.info(f"Run date   : {run_date}")
    log.info(f"GCS upload : {upload_gcs}")
    log.info(f"Resume     : {resume}")
    log.info(f"Test mode  : {test_mode} {'(10 per sector)' if test_mode else ''}")
    log.info(f"Batch size : {batch_size}")
    log.info(f"Universe CSV: {universe_csv or 'live Wikipedia fetch'}")

    # ── Load ticker universe ───────────────────────────────────────────────
    log.info("\n[Setup] Loading ticker universe...")
    if universe_csv:
        universe_df = pd.read_csv(universe_csv)
        required = {"ticker", "company", "sector"}
        if not required.issubset(universe_df.columns):
            missing = ", ".join(sorted(required - set(universe_df.columns)))
            raise ValueError(f"Universe CSV missing required column(s): {missing}")
        if "sub_industry" not in universe_df.columns:
            universe_df["sub_industry"] = ""
        if "index" not in universe_df.columns:
            universe_df["index"] = "CUSTOM"
        universe_df["ticker"] = universe_df["ticker"].astype(str).str.strip()
        universe_df = universe_df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    else:
        universe_df = get_all_tickers()
    if universe_df.empty:
        log.error("Could not load tickers — aborting")
        sys.exit(1)

    # Filter to requested sectors
    if sectors:
        universe_df = universe_df[universe_df["sector"].isin(sectors)]
        log.info(f"  Filtered to {len(sectors)} sectors: {sectors}")

    log.info(f"  Universe: {len(universe_df)} companies across {universe_df['sector'].nunique()} sectors")

    # Test mode: cap at 10 per sector
    if test_mode:
        universe_df = universe_df.groupby("sector").head(10).reset_index(drop=True)
        log.info(f"  Test mode: reduced to {len(universe_df)} companies")

    # Save universe to file
    Path(config.LOCAL_GOLD).mkdir(parents=True, exist_ok=True)
    universe_df.to_csv(os.path.join(config.LOCAL_GOLD, "ticker_universe.csv"), index=False)

    # ── Create batches ─────────────────────────────────────────────────────
    batches  = get_sector_batches(universe_df, batch_size=batch_size)
    progress = _load_progress() if resume else {
        "completed_tickers": [], "failed_tickers": [], "completed_sectors": []
    }

    log.info(f"\n[Plan] {len(batches)} total batches to process")
    log.info(f"       Estimated time: {len(batches) * 2:.0f}–{len(batches) * 4:.0f} minutes\n")

    # ── Run sector by sector ───────────────────────────────────────────────
    run_summary = {
        "total_companies":   len(universe_df),
        "total_batches":     len(batches),
        "succeeded":         [],
        "failed":            [],
        "sectors_completed": [],
        "start_time":        run_date,
    }

    current_sector    = None
    sector_start_time = time.time()

    for i, batch in enumerate(batches):
        sector      = batch["sector"]
        batch_num   = batch["batch_num"]
        total_b     = batch["total_batches"]
        tickers     = batch["tickers"]

        # Log sector header when sector changes
        if sector != current_sector:
            if current_sector:
                elapsed = time.time() - sector_start_time
                log.info(f"\n  ✅ Sector complete: {current_sector} ({elapsed:.0f}s)")
                run_summary["sectors_completed"].append(current_sector)
                progress["completed_sectors"].append(current_sector)
                _save_progress(progress)

            log.info(f"\n{'═' * 60}")
            log.info(f"  SECTOR [{universe_df[universe_df['sector']==sector].shape[0]} companies]: {sector}")
            log.info(f"{'═' * 60}")
            current_sector    = sector
            sector_start_time = time.time()

        log.info(f"\n  Batch {i+1}/{len(batches)} — {sector} ({batch_num}/{total_b})")

        result = run_sector_batch(
            sector=sector,
            batch_num=batch_num,
            total_batches=total_b,
            tickers=tickers,
            upload_gcs=upload_gcs,
            resume=resume,
            progress=progress,
        )

        run_summary["succeeded"].extend(result["succeeded"])
        run_summary["failed"].extend(result["failed"])
        progress["completed_tickers"].extend(result["succeeded"])
        progress["failed_tickers"].extend(result["failed"])
        _save_progress(progress)

        # Polite delay between batches to avoid rate limiting
        if i < len(batches) - 1:
            log.info("  Pausing 3s between batches...")
            time.sleep(3)

    # Last sector
    if current_sector:
        run_summary["sectors_completed"].append(current_sector)

    # ── Merge all Gold files → BigQuery ───────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  MERGING ALL SECTOR GOLD FILES → BIGQUERY")
    log.info("═" * 60)
    master_df = merge_and_load_bigquery()

    # ── Final run report ──────────────────────────────────────────────────
    elapsed  = time.time() - start_time
    run_summary["end_time"]      = datetime.today().strftime("%Y-%m-%d %H:%M")
    run_summary["elapsed_sec"]   = round(elapsed)
    run_summary["success_count"] = len(set(run_summary["succeeded"]))
    run_summary["fail_count"]    = len(set(run_summary["failed"]))

    report_path = os.path.join(config.LOCAL_GOLD, "run_report.json")
    with open(report_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    log.info("\n" + "╔" + "═" * 68 + "╗")
    log.info("║  PIPELINE RUN COMPLETE                                            ║")
    log.info("╚" + "═" * 68 + "╝")
    log.info(f"  Total companies targeted : {run_summary['total_companies']}")
    log.info(f"  Successfully processed   : {run_summary['success_count']}")
    log.info(f"  Failed                   : {run_summary['fail_count']}")
    log.info(f"  Sectors completed        : {len(run_summary['sectors_completed'])}")
    log.info(f"  Total time               : {elapsed/60:.1f} minutes")
    log.info(f"  Run report               : {report_path}")
    if master_df is not None:
        log.info(f"  Master Gold table        : {len(master_df)} companies in BigQuery")
    log.info("")

    if run_summary["fail_count"] > 0:
        log.warning(f"  Failed tickers: {set(run_summary['failed'])}")
        log.warning(f"  Re-run with --resume to retry failed tickers")

    return run_summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large-Scale Financial Risk Pipeline — S&P 500 + NASDAQ-100"
    )
    parser.add_argument(
        "--sectors", nargs="*", default=None,
        help='Specific sectors to run e.g. --sectors "Information Technology" "Health Care"'
    )
    parser.add_argument(
        "--gcs", action="store_true",
        help="Upload all outputs to GCS and load Gold to BigQuery"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last run — skip already-completed tickers"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode — only process 10 companies per sector"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Number of tickers per batch (default: 50)"
    )
    parser.add_argument(
        "--universe-csv", default=None,
        help="CSV with ticker, company, sector columns; avoids live Wikipedia ticker fetch"
    )
    parser.add_argument(
        "--list-sectors", action="store_true",
        help="Just print available sectors and exit"
    )

    args = parser.parse_args()

    # ── Handle --list-sectors FIRST before any pipeline imports run ───────
    if args.list_sectors:
        logging.basicConfig(level=logging.WARNING)  # silence noisy logs
        print("\nFetching S&P 500 + NASDAQ-100 sector breakdown...\n")
        df = get_all_tickers()
        if df.empty:
            print("Could not fetch ticker data. Check your internet connection.")
            sys.exit(1)
        print(f"{'Sector':<42} {'Companies':>10}  {'Tickers (sample)'}")
        print("─" * 80)
        for sector, group in df.groupby("sector"):
            sample = ", ".join(group["ticker"].head(5).tolist()) + "..."
            print(f"  {sector:<40} {len(group):>5} companies   {sample}")
        print("─" * 80)
        print(f"  {'TOTAL':<40} {len(df):>5} companies\n")
        sys.exit(0)

    run(
        sectors=args.sectors,
        upload_gcs=args.gcs,
        resume=args.resume,
        test_mode=args.test,
        batch_size=args.batch_size,
        universe_csv=args.universe_csv,
    )
