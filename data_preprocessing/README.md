# Data Preprocessing

This folder contains the project files that build the Bronze, Silver, and Gold
data layers.

## Main Files

- `ingest.py` — pulls Yahoo Finance, SEC EDGAR, FRED, and NewsAPI data into Bronze.
- `clean.py` — cleans Bronze files into Silver CSV/parquet outputs.
- `eda.py` — computes risk scores, insights, Gold outputs, and reports 01-07.
- `split.py` — creates train/validation/test splits and reports 08-09.
- `stats_summary.py` — creates report 10 and the data statistics CSV.
- `tickers.py` — builds the company universe.
- `newsapi_only.py` — fetches NewsAPI headlines without rerunning the full pipeline.
- `pipeline_large_scale.py` — runs the larger multi-sector pipeline.
- `macro_agent_plot.py` — generates macro agent plots.

Root-level wrapper files are kept for compatibility, so existing commands like
`python3 ingest.py AAPL` and imports like `import clean` still work.
