"""
split.py
─────────────────────────────────────────────────────────────────────
DATA SPLITTING MODULE
Single-snapshot stratified split (123 companies, one date).
Temporal split deferred until multi-date collection is complete.

Split: 70% train / 15% val / 15% test (stratified by risk_label)
─────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import config

log = logging.getLogger("split")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


def load_gold() -> pd.DataFrame:
    path = os.path.join(config.LOCAL_GOLD, "gold_risk_scores_ALL.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold file not found at {path}. Run pipeline first.")
    df = pd.read_csv(path, index_col=0)

    # Fix sector naming inconsistency
    df["sector"] = df["sector"].replace("Technology", "Information Technology")
    log.info(f"Loaded Gold table: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def stratified_split(df: pd.DataFrame):
    """
    Stratified split by risk_label to preserve class balance.
    Since all data is from one date, temporal split is not applicable.
    """
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=0.30, random_state=42,
        stratify=df["risk_label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42,
        stratify=temp["risk_label"]
    )

    log.info(f"Stratified split complete:")
    log.info(f"  Train: {len(train)} rows (70%)")
    log.info(f"  Val:   {len(val)} rows (15%)")
    log.info(f"  Test:  {len(test)} rows (15%)")
    return train, val, test


def print_split_summary(train, val, test):
    print("\n" + "═" * 60)
    print("  SPLIT SUMMARY  —  Stratified by Risk Label")
    print("═" * 60)

    for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n── {name} ({len(split)} companies) ──────────────────")
        counts = split["risk_label"].value_counts()
        for label, count in counts.items():
            pct = count / len(split) * 100
            bar = "█" * int(pct / 5)
            print(f"  {label:<12} {count:>3} companies  ({pct:.0f}%)  {bar}")

        sample_cols = ["composite_risk", "risk_label", 
                       "ann_volatility_pct", "sector"]
        available   = [c for c in sample_cols if c in split.columns]
        print(f"\n  Sample rows:")
        print(split[available].head(3).to_string())

    print("\n" + "═" * 60)
    print("  NOTE: Single-snapshot data (2026-03-04).")
    print("  Temporal split will be applied once multi-date")
    print("  collection is complete. Stratified split used")
    print("  here to preserve HIGH/MODERATE/LOW class balance.")
    print("═" * 60 + "\n")


def plot_class_balance(train, val, test):
    labels  = ["HIGH", "MODERATE", "LOW"]
    colors  = {"HIGH": "#EF4444", "MODERATE": "#F59E0B", "LOW": "#22C55E"}
    splits  = {"Train (70%)": train, "Val (15%)": val, "Test (15%)": test}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        "Risk Label Class Balance Across Splits\n(Stratified — 123 S&P 500 + NASDAQ-100 Companies)",
        fontsize=13, fontweight="bold", color="#0D1B3E"
    )

    for ax, (split_name, split_df) in zip(axes, splits.items()):
        counts = split_df["risk_label"].value_counts().reindex(labels, fill_value=0)
        bars   = ax.bar(
            counts.index,
            counts.values,
            color=[colors[l] for l in counts.index],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            width=0.5,
        )
        ax.set_title(f"{split_name}\n(n={len(split_df)})",
                     fontsize=11, fontweight="bold", color="#0D1B3E")
        ax.set_ylabel("Number of Companies", fontsize=10)
        ax.set_ylim(0, max(counts.values) * 1.35 + 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val_count in zip(bars, counts.values):
            pct = val_count / len(split_df) * 100 if len(split_df) > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val_count}\n({pct:.0f}%)",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    Path(config.LOCAL_REPORTS).mkdir(parents=True, exist_ok=True)
    path = os.path.join(config.LOCAL_REPORTS, "08_class_balance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    print(f"  Chart saved → {path}")
    return path


def plot_sector_distribution(df: pd.DataFrame):
    """Bonus chart — sector breakdown of all 123 companies."""
    sector_counts = df["sector"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#EF4444" if v >= 10 else "#F59E0B" 
               if v >= 8 else "#22C55E" for v in sector_counts.values]
    bars = ax.barh(sector_counts.index, sector_counts.values,
                   color=colors, alpha=0.85, edgecolor="white")

    for bar, count in zip(bars, sector_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold")

    ax.set_title("Companies per Sector — 123 Total\n(S&P 500 + NASDAQ-100)",
                 fontsize=13, fontweight="bold", color="#0D1B3E")
    ax.set_xlabel("Number of Companies", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(sector_counts.values) * 1.15)

    plt.tight_layout()
    path = os.path.join(config.LOCAL_REPORTS, "09_sector_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    print(f"  Chart saved → {path}")
    return path


def save_splits(train, val, test):
    Path(config.LOCAL_GOLD).mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(config.LOCAL_GOLD, f"split_{name}.csv")
        df.to_csv(path)
        log.info(f"  Saved: {path}  ({len(df)} rows)")


def run_split():
    log.info("=" * 60)
    log.info("FINANCIAL RISK PIPELINE — DATA SPLITTING")
    log.info("=" * 60 + "\n")

    df= load_gold()
    train, val, test = stratified_split(df)

    print_split_summary(train, val, test)
    save_splits(train, val, test)
    plot_class_balance(train, val, test)
    plot_sector_distribution(df)

    log.info("\n✅ Splitting complete.")
    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    run_split()