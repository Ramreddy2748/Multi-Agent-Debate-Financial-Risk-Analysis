"""
llm_comparison_plots.py

Generates comparison plots for any of the multi-LLM agent comparisons
(fundamental, market_sentiment, macro — see agents/fundamental_agent_llm.py,
agents/market_sentiment_agent_llm.py, agents/macro_agent_llm_compare.py).
Reads data/gold/{agent}_llm_comparison.json, saves to data/reports/.

Run: python3 agents/llm_comparison_plots.py --agent fundamental
     python3 agents/llm_comparison_plots.py --agent market_sentiment
     python3 agents/llm_comparison_plots.py --agent macro
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENT_TITLES = {
    "fundamental": "Fundamental Agent",
    "market_sentiment": "Market/Volatility/Sentiment Agent",
    "macro": "Macro-Economic Agent",
}

NAVY = "#1E2761"
TEAL = "#0D9488"
MUTED = "#94A3B8"
MODEL_PALETTE = ["#0D9488", "#7C3AED", "#F59E0B", "#EF4444"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a multi-LLM agent comparison.")
    parser.add_argument("--agent", required=True, choices=list(AGENT_TITLES.keys()))
    args = parser.parse_args()

    source_path = Path(f"data/gold/{args.agent}_llm_comparison.json")
    with open(source_path) as f:
        rows = json.load(f)

    if not rows:
        raise SystemExit(f"No rows found in {source_path} — run the {args.agent} LLM comparison script first.")

    Path("data/reports").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    models = sorted(df["model_key"].dropna().unique())
    colors = {model: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, model in enumerate(models)}
    title = AGENT_TITLES[args.agent]

    # ═══════════════════════════════════════════════════════════
    # PLOT 1 — Risk Score per Ticker, by Model
    # ═══════════════════════════════════════════════════════════
    tickers = sorted(df["ticker"].dropna().unique())
    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.6), 5.5))

    bar_width = 0.8 / max(len(models), 1)
    x = np.arange(len(tickers))

    for i, model in enumerate(models):
        sub = df[df["model_key"] == model].set_index("ticker")
        scores = [sub["risk_score"].get(t, np.nan) for t in tickers]
        offset = (i - (len(models) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, scores, width=bar_width, label=model,
                       color=colors[model], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, score in zip(bars, scores):
            if not np.isnan(score):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{score:.1f}", ha="center", fontsize=7, fontweight="bold")

    ax.axhline(3.5, color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(6.0, color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylim(0, 10.5)
    ax.set_title(f"{title} — Risk Score per Ticker, by Model", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Risk Score (1–10)", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"data/reports/{args.agent}_llm_01_scores_by_ticker.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {args.agent}_llm_01_scores_by_ticker.png")

    # ═══════════════════════════════════════════════════════════
    # PLOT 2 — Average Judge "Overall" Score per Model
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7.5, 5))

    judge_avg = df.groupby("model_key")["judge_overall"].mean().reindex(models)
    bars = ax.bar(judge_avg.index, judge_avg.values,
                   color=[colors[m] for m in judge_avg.index],
                   alpha=0.88, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, value in zip(bars, judge_avg.values):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{value:.2f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_title(f"{title} — Average Judge-Panel Score per Model\n"
                 "(fixed 2-judge panel, grounded in retrieved evidence)",
                 fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("Average Judge Overall Score (1–10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"data/reports/{args.agent}_llm_02_judge_overall.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {args.agent}_llm_02_judge_overall.png")

    # ═══════════════════════════════════════════════════════════
    # PLOT 3 — Judge Groundedness vs Consistency, by Model
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5.5))

    metrics = ["judge_groundedness", "judge_consistency"]
    metric_labels = ["Groundedness", "Consistency"]
    x = np.arange(len(metrics))
    bar_width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        sub = df[df["model_key"] == model]
        values = [sub[m].mean() for m in metrics]
        offset = (i - (len(models) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width, label=model,
                       color=colors[model], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{value:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 10.5)
    ax.set_title(f"{title} — Why One Model Scores Higher: Groundedness vs Consistency",
                 fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("Average Judge Sub-Score (1–10)", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"data/reports/{args.agent}_llm_03_judge_subscores.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {args.agent}_llm_03_judge_subscores.png")

    # ═══════════════════════════════════════════════════════════
    print(f"\nSummary — average judge overall score per model ({title}):")
    for model, value in judge_avg.items():
        print(f"  {model}: {value:.2f}/10" if not np.isnan(value) else f"  {model}: n/a")
    if not judge_avg.dropna().empty:
        best = judge_avg.idxmax()
        print(f"\nBest by cross-judge score: {best} ({judge_avg[best]:.2f}/10)")


if __name__ == "__main__":
    main()
