"""Compatibility wrapper for data_preprocessing.eda."""

from data_preprocessing.eda import *  # noqa: F401,F403


if __name__ == "__main__":
    import glob
    import pandas as pd
    import config

    price_dfs = {}
    for f in glob.glob(f"{config.LOCAL_SILVER}/silver_prices_*.csv"):
        ticker = f.split("silver_prices_")[1].replace(".csv", "")
        price_dfs[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)
    run_eda({"prices": price_dfs, "macro": pd.DataFrame(), "news": pd.DataFrame()})
