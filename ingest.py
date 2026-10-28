"""Compatibility wrapper for data_preprocessing.ingest."""

from data_preprocessing.ingest import *  # noqa: F401,F403


if __name__ == "__main__":
    import sys
    import config

    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else config.DEFAULT_TICKERS
    run_ingestion(tickers=tickers)
