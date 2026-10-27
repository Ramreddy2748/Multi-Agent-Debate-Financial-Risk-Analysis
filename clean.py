"""Compatibility wrapper for data_preprocessing.clean."""

from data_preprocessing.clean import *  # noqa: F401,F403


if __name__ == "__main__":
    import glob
    import pandas as pd
    import config

    price_dfs = {}
    for f in glob.glob(f"{config.LOCAL_BRONZE}/yahoo_*.csv"):
        ticker = f.split("yahoo_")[1].split("_")[0]
        price_dfs[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)
    run_cleaning({"prices": price_dfs, "edgar": {}, "macro": pd.DataFrame(), "news": pd.DataFrame()})
