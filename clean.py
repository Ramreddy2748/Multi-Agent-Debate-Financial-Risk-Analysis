"""Compatibility wrapper for data_preprocessing.clean."""

from data_preprocessing.clean import *  # noqa: F401,F403


if __name__ == "__main__":
    import pandas as pd
    import config

    price_dfs = load_latest_bronze_prices()
    run_cleaning({"prices": price_dfs, "edgar": {}, "macro": pd.DataFrame(), "news": pd.DataFrame()})
