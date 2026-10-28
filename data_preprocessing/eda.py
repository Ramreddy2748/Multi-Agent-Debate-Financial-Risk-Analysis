"""
eda.py
─────────────────────────────────────────────────────────────────────────────
EXPLORATORY DATA ANALYSIS MODULE  (Silver → Gold Layer)
Produces:
  1. Descriptive statistics  (per ticker: returns, volatility, P/E, etc.)
  2. Visualizations          (price trends, volatility, correlation, sentiment,
                              macro overlay, risk score heatmap)
  3. Key insights            (auto-generated text insights flagged by thresholds)
  4. Risk score computation  (composite 1–10 score per ticker per agent)
  5. Gold-layer export       (analytics-ready CSV → BigQuery-ready)

─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional

import config

warnings.filterwarnings("ignore")
log = logging.getLogger("eda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)

try:
    plt.style.use(config.PLOT_STYLE)
except OSError:
    plt.style.use("ggplot")

# ─── Color palette ────────────────────────────────────────────────────────────
PALETTE = {
    "navy":   "#0D1B3E",
    "teal":   "#0D9488",
    "gold":   "#F59E0B",
    "red":    "#EF4444",
    "green":  "#22C55E",
    "purple": "#A855F7",
    "pink":   "#EC4899",
    "gray":   "#64748B",
}
TICKER_COLORS = [
    "#0D9488", "#F59E0B", "#EF4444", "#A855F7", "#0D1B3E",
    "#2563EB", "#DB2777", "#16A34A", "#EA580C", "#0891B2",
]
REPORT_LINE_TICKERS = 8
REPORT_DISTRIBUTION_TICKERS = 6
REPORT_CORRELATION_TICKERS = 10
REPORT_RISK_TICKERS = 12
REPORT_SENTIMENT_TICKERS = 8


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [config.LOCAL_GOLD, config.LOCAL_REPORTS]:
        Path(d).mkdir(parents=True, exist_ok=True)

def _save_fig(fig: plt.Figure, name: str) -> str:
    _ensure_dirs()
    path = os.path.join(config.LOCAL_REPORTS, name)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info(f"  📊 Saved: {path}")
    return path

def _risk_color(score: float) -> str:
    if score <= config.RISK_SCORE_LOW:
        return PALETTE["green"]
    elif score <= config.RISK_SCORE_HIGH:
        return PALETTE["gold"]
    return PALETTE["red"]

def _risk_label(score: float) -> str:
    if score <= config.RISK_SCORE_LOW:
        return "LOW"
    elif score <= config.RISK_SCORE_HIGH:
        return "MODERATE"
    return "HIGH"

def _market_cap_for_plot(df: pd.DataFrame) -> float:
    if "market_cap_usd_m" not in df.columns:
        return 0.0
    values = pd.to_numeric(df["market_cap_usd_m"], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else 0.0

def _select_price_report_data(
    price_data: dict[str, pd.DataFrame],
    max_tickers: int,
) -> dict[str, pd.DataFrame]:
    """Choose a small, readable subset for presentation charts."""
    duplicate_share_classes = {"GOOG": "GOOGL", "BRK-B": "BRK-A", "FOX": "FOXA", "NWS": "NWSA"}
    ranked = sorted(
        price_data.items(),
        key=lambda item: (_market_cap_for_plot(item[1]), item[0]),
        reverse=True,
    )
    selected = []
    selected_tickers = set()
    for ticker, df in ranked:
        preferred = duplicate_share_classes.get(ticker)
        if preferred and preferred in price_data:
            continue
        if ticker in selected_tickers:
            continue
        selected.append((ticker, df))
        selected_tickers.add(ticker)
        if len(selected) >= max_tickers:
            break
    return dict(selected)

def _subtitle(ax, text: str) -> None:
    ax.text(
        0,
        1.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gray"],
        va="bottom",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(
    price_data: dict[str, pd.DataFrame],
    macro_df:   Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute per-ticker summary statistics used for risk scoring.

    Returns:
        DataFrame indexed by ticker with columns:
        current_price, ytd_return_pct, ann_volatility_pct,
        sharpe_ratio, max_drawdown_pct, beta, pe_ratio,
        market_cap_usd_m, mean_daily_return, skewness, kurtosis
    """
    log.info("[EDA] Computing descriptive statistics...")
    rows = []

    for ticker, df in price_data.items():
        df = df.sort_index()

        # Identify close price column
        close_col = "close" if "close" in df.columns else "Close"
        ret_col   = "daily_return" if "daily_return" in df.columns else None

        if close_col not in df.columns:
            log.warning(f"  No close column for {ticker} — skipping")
            continue

        close   = df[close_col].dropna()
        returns = df[ret_col].dropna() if ret_col else close.pct_change().dropna()

        # ── Core metrics ──────────────────────────────────────────────────
        current_price    = close.iloc[-1]
        start_price      = close.iloc[0]
        ytd_return       = (current_price - start_price) / start_price * 100
        ann_vol          = returns.std() * np.sqrt(252) * 100
        mean_daily_ret   = returns.mean()
        sharpe           = (mean_daily_ret * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        skewness         = returns.skew()
        kurtosis         = returns.kurtosis()

        # Max Drawdown
        cumulative       = (1 + returns).cumprod()
        rolling_max      = cumulative.cummax()
        drawdown         = (cumulative - rolling_max) / rolling_max
        max_drawdown     = drawdown.min() * 100

        # Fundamentals (latest available)
        pe_ratio  = df["pe_ratio"].dropna().iloc[-1]   if "pe_ratio"  in df.columns and not df["pe_ratio"].dropna().empty  else None
        beta      = df["beta"].dropna().iloc[-1]        if "beta"      in df.columns and not df["beta"].dropna().empty      else None
        mkt_cap   = df["market_cap_usd_m"].dropna().iloc[-1] if "market_cap_usd_m" in df.columns and not df["market_cap_usd_m"].dropna().empty else None

        rows.append({
            "ticker":            ticker,
            "current_price":     round(current_price, 2),
            "ytd_return_pct":    round(ytd_return, 2),
            "ann_volatility_pct":round(ann_vol, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown_pct":  round(max_drawdown, 2),
            "mean_daily_return": round(mean_daily_ret * 100, 4),
            "skewness":          round(skewness, 3),
            "kurtosis":          round(kurtosis, 3),
            "pe_ratio":          round(pe_ratio, 1) if pe_ratio is not None else None,
            "beta":              round(beta, 2)     if beta     is not None else None,
            "market_cap_usd_m":  round(mkt_cap, 0) if mkt_cap  is not None else None,
        })

    stats_df = pd.DataFrame(rows).set_index("ticker")
    log.info(f"  ✔ Statistics computed for {len(stats_df)} tickers")
    return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# 2 — RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_scores(stats_df: pd.DataFrame, news_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute multi-dimensional risk scores (1–10 scale) per ticker.
    Dimensions mirror the 4 specialized agents:
      - fundamental_risk  : P/E x beta x drawdown
      - volatility_risk   : annualized volatility, kurtosis
      - sentiment_risk    : VADER compound (from news_df)
      - macro_risk        : derived from FRED silver data
      - composite_risk    : weighted average
    """
    log.info("[EDA] Computing risk scores...")

    def _scale(series: pd.Series, invert: bool = False) -> pd.Series:
        """Scale series to 1-10."""
        s_min, s_max = series.min(), series.max()
        if s_min == s_max:
            return pd.Series([5.0] * len(series), index=series.index)
        scaled = 1 + 9 * (series - s_min) / (s_max - s_min)
        return 10 - scaled + 1 if invert else scaled

    df = stats_df.copy()

    # ── Volatility risk ───────────────────────────────────────────────────
    df["volatility_risk"] = _scale(df["ann_volatility_pct"])

    # ── Fundamental risk ──────────────────────────────────────────────────
    pe_score   = _scale(df["pe_ratio"].fillna(df["pe_ratio"].median()))
    beta_score = _scale(df["beta"].fillna(1.0))
    dd_score   = _scale(df["max_drawdown_pct"].abs())
    df["fundamental_risk"] = (
        pe_score * 0.4 + beta_score * 0.3 + dd_score * 0.3
    ).clip(1, 10)

    # ── Sentiment risk ────────────────────────────────────────────────────
    if news_df is not None and not news_df.empty and "sentiment_mean" in news_df.columns:
        sentiment_by_ticker = news_df.groupby("ticker")["sentiment_mean"].mean()
        df["sentiment_risk"] = _scale(
            df.index.map(sentiment_by_ticker).fillna(0), invert=True
        )
    else:
        df["sentiment_risk"] = 5.0

    # ── Macro risk (derived from FRED silver data) ────────────────────────
    try:
        macro_path = os.path.join(config.LOCAL_SILVER, "silver_macro.csv")
        if os.path.exists(macro_path):
            macro_silver = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            latest = macro_silver.dropna(how="all").iloc[-1]

            fed = float(latest.get("fed_funds_rate", 3.0))
            fed_score = float(np.clip((fed / 6.0) * 10, 1, 10))

            if "cpi" in macro_silver.columns:
                cpi_series = macro_silver["cpi"].dropna()
                cpi_z = abs(
                    (cpi_series.iloc[-1] - cpi_series.mean()) /
                    (cpi_series.std() + 1e-9)
                )
                cpi_score = float(np.clip(1 + cpi_z * 3, 1, 10))
            else:
                cpi_score = 5.0

            macro_score = round((fed_score + cpi_score) / 2, 2)
            df["macro_risk"] = macro_score
            log.info(
                f"  [Macro Risk] fed_funds={fed:.2f}% → {fed_score:.1f}, "
                f"cpi_z={cpi_z:.2f} → {cpi_score:.1f}, "
                f"macro_risk={macro_score}"
            )
        else:
            log.warning("  silver_macro.csv not found — defaulting to 5.0")
            df["macro_risk"] = 5.0
    except Exception as e:
        log.warning(f"  Macro risk computation failed ({e}) — defaulting to 5.0")
        df["macro_risk"] = 5.0

    # ── Composite risk (weighted) ─────────────────────────────────────────
    df["composite_risk"] = (
        df["fundamental_risk"]  * 0.30 +
        df["volatility_risk"]   * 0.30 +
        df["sentiment_risk"]    * 0.20 +
        df["macro_risk"]        * 0.20
    ).round(2)

    df["risk_label"] = df["composite_risk"].apply(_risk_label)

    risk_cols = [
        "fundamental_risk", "volatility_risk",
        "sentiment_risk", "macro_risk",
        "composite_risk", "risk_label"
    ]
    log.info(f"  Risk scores computed:\n{df[risk_cols].to_string()}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_trends(price_data: dict[str, pd.DataFrame]) -> str:
    """Normalized price performance (rebased to 100) for a readable subset."""
    selected = _select_price_report_data(price_data, REPORT_LINE_TICKERS)
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")

    for i, (ticker, df) in enumerate(selected.items()):
        df = df.sort_index()
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns:
            continue
        close    = df[close_col].dropna()
        if close.empty:
            continue
        rebased  = close / close.iloc[0] * 100
        color    = TICKER_COLORS[i % len(TICKER_COLORS)]
        ax.plot(rebased.index, rebased.values, label=ticker, color=color, linewidth=2.1)

    ax.axhline(100, color=PALETTE["gray"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Normalized Price Performance (Base = 100)", fontsize=15, fontweight="bold", color=PALETTE["navy"], pad=18)
    _subtitle(ax, f"Showing top {len(selected)} companies by available market cap for readability; full universe remains in Gold data.")
    ax.set_ylabel("Rebased Price", fontsize=11)
    ax.set_xlabel("")
    ax.legend(fontsize=9, framealpha=0.92, ncol=4, loc="upper left")
    ax.grid(alpha=0.22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    plt.tight_layout()
    return _save_fig(fig, "01_price_trends.png")

def plot_volatility(price_data: dict[str, pd.DataFrame]) -> str:
    """30-day rolling annualized volatility for a readable subset."""
    selected = _select_price_report_data(price_data, REPORT_LINE_TICKERS)
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (ticker, df) in enumerate(selected.items()):
        df = df.sort_index()
        if "rolling_vol_30d" not in df.columns:
            continue
        color = TICKER_COLORS[i % len(TICKER_COLORS)]
        ax.plot(df.index, df["rolling_vol_30d"] * 100, label=ticker, color=color, linewidth=2.0)

    ax.set_title("30-Day Rolling Annualized Volatility (%)", fontsize=15, fontweight="bold", color=PALETTE["navy"], pad=18)
    _subtitle(ax, f"Showing top {len(selected)} companies by available market cap to avoid overplotting.")
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.92, ncol=4, loc="upper left")
    ax.grid(alpha=0.22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    return _save_fig(fig, "02_volatility.png")

def plot_return_distribution(price_data: dict[str, pd.DataFrame]) -> str:
    """Histogram of daily returns with KDE overlay for selected tickers."""
    selected = _select_price_report_data(price_data, REPORT_DISTRIBUTION_TICKERS)
    n = len(selected)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.2 * rows), squeeze=False)

    for idx, (ticker, df) in enumerate(selected.items()):
        ax = axes[idx // cols][idx % cols]
        ret_col = "daily_return" if "daily_return" in df.columns else None
        if ret_col is None:
            close_col = "close" if "close" in df.columns else "Close"
            returns = df[close_col].pct_change().dropna() * 100
        else:
            returns = df[ret_col].dropna() * 100

        color = TICKER_COLORS[idx % len(TICKER_COLORS)]
        ax.hist(returns, bins=36, color=color, alpha=0.62, edgecolor="white", linewidth=0.3, density=True)
        returns.plot.kde(ax=ax, color=color, linewidth=2)
        ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=0.9)
        ax.set_title(f"{ticker} — Daily Returns", fontsize=12, fontweight="bold", color=PALETTE["navy"])
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"Daily Return Distributions — Top {len(selected)} Companies",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    plt.tight_layout()
    return _save_fig(fig, "03_return_distributions.png")

def plot_correlation_matrix(price_data: dict[str, pd.DataFrame]) -> str:
    """Pearson correlation heatmap of daily returns across tickers."""
    selected = _select_price_report_data(price_data, REPORT_CORRELATION_TICKERS)
    # Build returns matrix
    ret_dict = {}
    for ticker, df in selected.items():
        ret_col   = "daily_return" if "daily_return" in df.columns else None
        close_col = "close" if "close" in df.columns else "Close"
        if ret_col:
            ret_dict[ticker] = df[ret_col]
        elif close_col in df.columns:
            ret_dict[ticker] = df[close_col].pct_change()

    if not ret_dict:
        return ""

    returns_df = pd.DataFrame(ret_dict).dropna()
    corr       = returns_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    import matplotlib.cm as cm
    cmap = cm.RdYlGn
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    tickers = corr.columns.tolist()
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, fontsize=10, fontweight="bold", rotation=35, ha="right")
    ax.set_yticklabels(tickers, fontsize=10, fontweight="bold")

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 else PALETTE["navy"],
                    fontweight="bold")

    ax.set_title("Daily Return Correlation Matrix", fontsize=15, fontweight="bold", color=PALETTE["navy"], pad=18)
    _subtitle(ax, f"Showing top {len(tickers)} companies by available market cap.")
    plt.tight_layout()
    return _save_fig(fig, "04_correlation_matrix.png")

def plot_risk_scores(risk_df: pd.DataFrame) -> str:
    """Grouped bar chart of multi-dimensional risk scores."""
    if "composite_risk" in risk_df.columns and len(risk_df) > REPORT_RISK_TICKERS:
        risk_df = risk_df.sort_values("composite_risk", ascending=False).head(REPORT_RISK_TICKERS)
    dims = ["fundamental_risk", "volatility_risk", "sentiment_risk", "macro_risk"]
    dim_labels = ["Fundamental", "Volatility", "Sentiment", "Macro"]
    tickers = risk_df.index.tolist()
    x = np.arange(len(tickers))
    width = 0.18
    colors = [PALETTE["teal"], PALETTE["gold"], PALETTE["purple"], PALETTE["navy"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Grouped bar chart ─────────────────────────────────────────────────
    for i, (dim, label, color) in enumerate(zip(dims, dim_labels, colors)):
        if dim in risk_df.columns:
            vals = risk_df[dim].values
            bars = ax1.bar(x + i * width, vals, width, label=label, color=color, alpha=0.88)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(tickers, fontsize=9, fontweight="bold", rotation=35, ha="right")
    ax1.set_ylim(0, 10.5)
    ax1.axhline(config.RISK_SCORE_LOW,  color=PALETTE["green"], linestyle="--", linewidth=1, alpha=0.7, label=f"Low threshold ({config.RISK_SCORE_LOW})")
    ax1.axhline(config.RISK_SCORE_HIGH, color=PALETTE["red"],   linestyle="--", linewidth=1, alpha=0.7, label=f"High threshold ({config.RISK_SCORE_HIGH})")
    ax1.set_ylabel("Risk Score (1–10)", fontsize=11)
    ax1.set_title(f"Multi-Dimensional Risk Scores — Top {len(tickers)} by Composite Risk", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.2)

    # ── Composite risk bar ────────────────────────────────────────────────
    comp_scores = risk_df["composite_risk"].values if "composite_risk" in risk_df.columns else np.zeros(len(tickers))
    bar_colors  = [_risk_color(s) for s in comp_scores]
    bars2 = ax2.barh(tickers, comp_scores, color=bar_colors, alpha=0.88, height=0.5)

    for bar, score, label in zip(bars2, comp_scores, risk_df.get("risk_label", [""] * len(tickers))):
        ax2.text(score + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}  {label}", va="center", fontsize=10, fontweight="bold",
                 color=_risk_color(score))

    ax2.set_xlim(0, 12)
    ax2.axvline(config.RISK_SCORE_LOW,  color=PALETTE["green"], linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(config.RISK_SCORE_HIGH, color=PALETTE["red"],   linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_title("Composite Risk Score", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax2.set_xlabel("Score (1–10)")
    ax2.grid(axis="x", alpha=0.18)

    plt.suptitle("Financial Risk Assessment Dashboard", fontsize=15, fontweight="bold",
                 color=PALETTE["navy"], y=1.02)
    plt.tight_layout()
    return _save_fig(fig, "05_risk_scores.png")

def plot_macro_overlay(
    price_data: dict[str, pd.DataFrame],
    macro_df:   Optional[pd.DataFrame],
    ticker:     Optional[str] = None,
) -> str:
    """Price trend for one ticker overlaid with macro indicators."""
    if macro_df is None or macro_df.empty:
        log.info("  [Macro Overlay] No macro data — skipping.")
        return ""

    # Pick first ticker if not specified
    ticker = ticker or next(iter(price_data))
    df     = price_data[ticker].sort_index()
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{ticker} — Price vs. Macro Environment", fontsize=14,
                 fontweight="bold", color=PALETTE["navy"])

    # Stock price
    axes[0].plot(df.index, df[close_col], color=PALETTE["teal"], linewidth=1.8)
    axes[0].set_ylabel("Price (USD)", fontsize=10)
    axes[0].set_title("Stock Price", fontsize=11, color=PALETTE["navy"])

    # Fed Funds Rate + 10Y Treasury
    if "fed_funds_rate" in macro_df.columns:
        axes[1].plot(macro_df.index, macro_df["fed_funds_rate"], color=PALETTE["red"],  linewidth=1.5, label="Fed Funds")
    if "treasury_10y" in macro_df.columns:
        axes[1].plot(macro_df.index, macro_df["treasury_10y"],   color=PALETTE["gold"], linewidth=1.5, label="10Y Treasury")
    axes[1].set_ylabel("Rate (%)", fontsize=10)
    axes[1].set_title("Interest Rates", fontsize=11, color=PALETTE["navy"])
    axes[1].legend(fontsize=9)

    # CPI
    if "cpi" in macro_df.columns:
        axes[2].plot(macro_df.index, macro_df["cpi"], color=PALETTE["purple"], linewidth=1.5)
        axes[2].set_ylabel("CPI Index", fontsize=10)
        axes[2].set_title("Consumer Price Index (CPI)", fontsize=11, color=PALETTE["navy"])

    plt.tight_layout()
    return _save_fig(fig, f"06_macro_overlay_{ticker}.png")

def plot_sentiment_trend(news_df: Optional[pd.DataFrame]) -> str:
    """Daily sentiment trend per ticker."""
    if news_df is None or news_df.empty or "sentiment_mean" not in news_df.columns:
        log.info("  [Sentiment] No sentiment data — skipping.")
        return ""

    fig, ax = plt.subplots(figsize=(13, 5))
    if "article_count" in news_df.columns:
        ticker_order = (
            news_df.groupby("ticker")["article_count"]
            .sum()
            .sort_values(ascending=False)
            .head(REPORT_SENTIMENT_TICKERS)
            .index
            .tolist()
        )
    else:
        ticker_order = list(news_df["ticker"].dropna().unique()[:REPORT_SENTIMENT_TICKERS])

    for i, ticker in enumerate(ticker_order):
        sub = news_df[news_df["ticker"] == ticker].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date")
        color = TICKER_COLORS[i % len(TICKER_COLORS)]
        ax.plot(sub["date"], sub["sentiment_mean"], marker="o", markersize=3,
                linewidth=1.5, color=color, label=ticker, alpha=0.8)

    ax.axhline(0,     color=PALETTE["gray"], linestyle="--", linewidth=0.8)
    ax.axhline(0.05,  color=PALETTE["green"], linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(-0.05, color=PALETTE["red"],   linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title("Daily News Sentiment Trend (VADER Compound Score)", fontsize=14,
                 fontweight="bold", color=PALETTE["navy"], pad=14)
    ax.set_ylabel("Sentiment Score (−1 to +1)", fontsize=11)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9, framealpha=0.92, ncol=4, loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return _save_fig(fig, "07_sentiment_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4 — KEY INSIGHTS (Auto-Generated)
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(stats_df: pd.DataFrame, risk_df: pd.DataFrame) -> list[dict]:
    """
    Auto-generate text insights based on statistical thresholds.
    Returns a list of insight dicts with keys: category, ticker, insight, severity.
    """
    insights = []

    def _add(category, ticker, text, severity="INFO"):
        insights.append({"category": category, "ticker": ticker, "insight": text, "severity": severity})

    for ticker in stats_df.index:
        row  = stats_df.loc[ticker]
        risk = risk_df.loc[ticker] if ticker in risk_df.index else None

        # ── Volatility ─────────────────────────────────────────────────
        vol = row.get("ann_volatility_pct")
        if vol is not None:
            if vol > 50:
                _add("Volatility", ticker, f"Extreme annualized volatility of {vol:.1f}% — significantly above market average (~20%). High uncertainty for near-term positions.", "HIGH")
            elif vol > 30:
                _add("Volatility", ticker, f"Elevated annualized volatility of {vol:.1f}%. Monitor closely around earnings and macro events.", "MEDIUM")
            else:
                _add("Volatility", ticker, f"Relatively stable — annualized volatility of {vol:.1f}% is within normal range.", "LOW")

        # ── Returns ────────────────────────────────────────────────────
        ytd = row.get("ytd_return_pct")
        if ytd is not None:
            if ytd < -20:
                _add("Returns", ticker, f"Significant YTD decline of {ytd:.1f}%. May signal fundamental deterioration or sector-wide headwinds.", "HIGH")
            elif ytd > 50:
                _add("Returns", ticker, f"Exceptional YTD return of +{ytd:.1f}%. Elevated valuation risk — growth expectations are priced in.", "MEDIUM")

        # ── Drawdown ───────────────────────────────────────────────────
        dd = row.get("max_drawdown_pct")
        if dd is not None and dd < -30:
            _add("Drawdown", ticker, f"Max drawdown of {dd:.1f}% over the period. Indicates significant peak-to-trough risk exposure.", "HIGH")

        # ── P/E Valuation ──────────────────────────────────────────────
        pe = row.get("pe_ratio")
        if pe is not None:
            if pe > 60:
                _add("Valuation", ticker, f"P/E of {pe:.1f}× reflects extremely high growth expectations. Vulnerable to earnings misses.", "HIGH")
            elif pe < 10:
                _add("Valuation", ticker, f"P/E of {pe:.1f}× is very low — may indicate value opportunity or market skepticism.", "MEDIUM")

        # ── Sharpe ─────────────────────────────────────────────────────
        sharpe = row.get("sharpe_ratio")
        if sharpe is not None and sharpe < 0:
            _add("Risk-Adjusted", ticker, f"Negative Sharpe ratio ({sharpe:.2f}) — returns don't compensate for volatility taken.", "HIGH")

        # ── Skewness ───────────────────────────────────────────────────
        skew = row.get("skewness")
        if skew is not None and skew < -1:
            _add("Distribution", ticker, f"Negatively skewed returns (skew={skew:.2f}) — tail risk tilted to downside.", "MEDIUM")

        # ── Composite Risk ─────────────────────────────────────────────
        if risk is not None:
            comp = risk.get("composite_risk")
            if comp is not None and comp > config.RISK_SCORE_HIGH:
                _add("Composite Risk", ticker, f"Composite risk score {comp:.1f}/10 — classified as HIGH RISK. Multi-agent debate recommended before any position.", "HIGH")

    log.info(f"[EDA] Generated {len(insights)} insights ({sum(1 for i in insights if i['severity']=='HIGH')} HIGH severity)")
    return insights

def print_insights(insights: list[dict]):
    """Pretty-print insights to console."""
    severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️ "}
    print("\n" + "═" * 70)
    print("  KEY INSIGHTS FROM EDA")
    print("═" * 70)
    for ins in sorted(insights, key=lambda x: ["HIGH", "MEDIUM", "LOW", "INFO"].index(x["severity"])):
        icon = severity_icon.get(ins["severity"], "  ")
        print(f"\n{icon} [{ins['ticker']}] {ins['category']}")
        print(f"   {ins['insight']}")
    print("\n" + "═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5 — GOLD LAYER EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_gold(
    stats_df:   pd.DataFrame,
    risk_df:    pd.DataFrame,
    insights:   list[dict],
    upload_gcs: bool = False,
) -> dict[str, str]:
    """
    Save analytics-ready tables to Gold layer.
    These CSVs are directly BigQuery-importable.
    """
    _ensure_dirs()
    paths = {}

    # Risk scores gold table
    gold_risk = pd.concat([stats_df, risk_df[["fundamental_risk", "volatility_risk",
                                               "sentiment_risk", "macro_risk",
                                               "composite_risk", "risk_label"]]], axis=1)
    gold_risk["as_of_date"] = pd.Timestamp.today().date()
    p = os.path.join(config.LOCAL_GOLD, "gold_risk_scores.csv")
    gold_risk.to_csv(p)
    paths["risk_scores"] = p
    log.info(f"  ✔ Gold — risk scores: {p}")

    # Insights table
    insights_df = pd.DataFrame(insights)
    insights_df["as_of_date"] = pd.Timestamp.today().date()
    p = os.path.join(config.LOCAL_GOLD, "gold_insights.csv")
    insights_df.to_csv(p, index=False)
    paths["insights"] = p
    log.info(f"  ✔ Gold — insights:    {p}")

    if upload_gcs:
        for key, local_path in paths.items():
            filename = os.path.basename(local_path)
            try:
                from google.cloud import storage
                client = storage.Client(project=config.GCP_PROJECT_ID)
                client.bucket(config.GCS_BUCKET).blob(f"gold/{filename}").upload_from_filename(local_path)
                log.info(f"  ☁  Uploaded: gs://{config.GCS_BUCKET}/gold/{filename}")
            except Exception as e:
                log.error(f"GCS upload failed: {e}")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(clean_data: dict, upload_gcs: bool = False) -> dict:
    """
    Run full EDA on cleaned Silver-layer data.

    Args:
        clean_data: Output dict from clean.run_cleaning()
        upload_gcs: Upload Gold outputs to GCS if True

    Returns:
        dict with keys: stats, risk_scores, insights, plot_paths, gold_paths
    """
    log.info("=" * 70)
    log.info("FINANCIAL RISK PIPELINE — EDA (Gold Layer)")
    log.info("=" * 70 + "\n")

    price_data = clean_data.get("prices", {})
    macro_df   = clean_data.get("macro",  pd.DataFrame())
    news_df    = clean_data.get("news",   pd.DataFrame())

    if not price_data:
        log.error("No price data available for EDA.")
        return {}

    # 1 — Statistics
    stats_df  = compute_statistics(price_data, macro_df)
    print("\n── DESCRIPTIVE STATISTICS ──────────────────────────────────────────")
    print(stats_df.to_string())

    # 2 — Risk Scores
    risk_df   = compute_risk_scores(stats_df, news_df if not news_df.empty else None)

    # 3 — Visualizations
    log.info("\n[EDA] Generating visualizations...")
    plot_paths = {}
    plot_paths["price_trends"]          = plot_price_trends(price_data)
    plot_paths["volatility"]            = plot_volatility(price_data)
    plot_paths["return_distributions"]  = plot_return_distribution(price_data)
    plot_paths["correlation_matrix"]    = plot_correlation_matrix(price_data)
    plot_paths["risk_scores"]           = plot_risk_scores(risk_df)
    plot_paths["macro_overlay"]         = plot_macro_overlay(price_data, macro_df if not macro_df.empty else None)
    plot_paths["sentiment_trend"]       = plot_sentiment_trend(news_df if not news_df.empty else None)

    # 4 — Insights
    insights  = generate_insights(stats_df, risk_df)
    print_insights(insights)

    # 5 — Gold Export
    gold_paths = export_gold(stats_df, risk_df, insights, upload_gcs=upload_gcs)

    log.info("\n[EDA] ✅ Complete.\n")
    return {
        "stats":      stats_df,
        "risk_scores": risk_df,
        "insights":   insights,
        "plot_paths": plot_paths,
        "gold_paths": gold_paths,
    }


if __name__ == "__main__":
    # Standalone test using Silver CSVs
    import glob
    price_dfs = {}
    for f in glob.glob(f"{config.LOCAL_SILVER}/silver_prices_*.csv"):
        ticker = f.split("silver_prices_")[1].replace(".csv", "")
        price_dfs[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)
    run_eda({"prices": price_dfs, "macro": pd.DataFrame(), "news": pd.DataFrame()})
