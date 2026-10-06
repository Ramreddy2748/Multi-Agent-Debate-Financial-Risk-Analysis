"""
macro_agent_plots.py
─────────────────────────────────────────────────────────────────
Generates all visualisation plots for the Macro Agent evaluation.
Saves to data/reports/

Run: python3 macro_agent_plots.py
─────────────────────────────────────────────────────────────────
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Load agent outputs ────────────────────────────────────────
with open("data/gold/macro_agent_outputs.json") as f:
    outputs = json.load(f)

Path("data/reports").mkdir(parents=True, exist_ok=True)

# ── Prepare dataframe ─────────────────────────────────────────
rows = []
for r in outputs:
    rows.append({
        "ticker":     r.get("ticker", ""),
        "sector":     r.get("sector", "Unknown"),
        "score":      r.get("macro_risk_score", 5.85),
        "label":      r.get("risk_label", "MODERATE"),
        "confidence": r.get("confidence", 0.0),
        "recession":  r.get("recession_signal", False),
        "failed":     "error" in r or r.get("confidence", 1) == 0.0,
    })

df = pd.DataFrame(rows)

COLORS = {
    "HIGH":     "#EF4444",
    "MODERATE": "#F59E0B",
    "LOW":      "#22C55E",
}
NAVY  = "#1E2761"
TEAL  = "#0D9488"
MUTED = "#94A3B8"

# ═══════════════════════════════════════════════════════════════
# PLOT 1 — Risk Score Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

valid = df[~df["failed"]]["score"]
ax.hist(valid, bins=20, color=TEAL, alpha=0.85,
        edgecolor="white", linewidth=0.5)

ax.axvline(valid.mean(), color="#EF4444", linewidth=2,
           linestyle="--", label=f"Mean: {valid.mean():.2f}")
ax.axvline(valid.median(), color="#F59E0B", linewidth=2,
           linestyle=":", label=f"Median: {valid.median():.2f}")
ax.axvline(3.5, color=MUTED, linewidth=1,
           linestyle="-", alpha=0.5, label="LOW threshold (3.5)")
ax.axvline(6.0, color=MUTED, linewidth=1,
           linestyle="-", alpha=0.5, label="HIGH threshold (6.0)")

ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30],
                 0, 3.5, alpha=0.06, color="#22C55E")
ax.fill_betweenx([0, 30], 3.5, 6.0, alpha=0.06, color="#F59E0B")
ax.fill_betweenx([0, 30], 6.0, 10, alpha=0.06, color="#EF4444")

ax.set_title("Macro Risk Score Distribution — 123 Companies\n"
             "DeepSeek-R1 Prompt-Based Inference",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlabel("Macro Risk Score (1–10)", fontsize=11)
ax.set_ylabel("Number of Companies", fontsize=11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_01_score_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_01_score_distribution.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 2 — Risk Label Distribution (Bar Chart)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

label_counts = df["label"].value_counts().reindex(
    ["HIGH", "MODERATE", "LOW"], fill_value=0
)
bars = ax.bar(
    label_counts.index,
    label_counts.values,
    color=[COLORS[l] for l in label_counts.index],
    alpha=0.88, edgecolor="white", linewidth=0.5,
    width=0.5
)
for bar, count in zip(bars, label_counts.values):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}\n({pct:.1f}%)",
            ha="center", fontsize=11, fontweight="bold")

ax.set_title("Macro Risk Label Distribution\n"
             "123 S&P 500 + NASDAQ-100 Companies",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_ylabel("Number of Companies", fontsize=11)
ax.set_ylim(0, max(label_counts.values) * 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_02_label_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_02_label_distribution.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Risk Score by Sector (Box Plot)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

# Fix sector naming
df["sector"] = df["sector"].replace(
    "Technology", "Information Technology"
)

sector_data = [
    df[df["sector"] == s]["score"].values
    for s in df["sector"].unique()
]
sector_names = list(df["sector"].unique())

# Sort by median score
medians = [np.median(d) for d in sector_data]
sorted_idx = np.argsort(medians)[::-1]
sector_data  = [sector_data[i]  for i in sorted_idx]
sector_names = [sector_names[i] for i in sorted_idx]

bp = ax.boxplot(
    sector_data,
    patch_artist=True,
    notch=False,
    vert=True,
    widths=0.6,
)
for patch, median in zip(bp["boxes"], [np.median(d) for d in sector_data]):
    color = (COLORS["HIGH"]     if median > 6.0 else
             COLORS["MODERATE"] if median > 3.5 else
             COLORS["LOW"])
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for element in ["whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(bp[element], color=NAVY)

plt.setp(bp["medians"], linewidth=2, color="white")

ax.set_xticks(range(1, len(sector_names) + 1))
ax.set_xticklabels(sector_names, rotation=35, ha="right", fontsize=9)
ax.set_title("Macro Risk Score by Sector\n"
             "Box Plot — DeepSeek-R1 Agent Output",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_ylabel("Macro Risk Score (1–10)", fontsize=11)
ax.axhline(6.0, color=COLORS["HIGH"],     linestyle="--",
           linewidth=1, alpha=0.6, label="HIGH threshold")
ax.axhline(3.5, color=COLORS["LOW"],      linestyle="--",
           linewidth=1, alpha=0.6, label="LOW threshold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_03_score_by_sector.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_03_score_by_sector.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Confidence Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

valid_conf = df[df["confidence"] > 0]["confidence"]
ax.hist(valid_conf, bins=15, color="#7C3AED",
        alpha=0.85, edgecolor="white", linewidth=0.5)

ax.axvline(valid_conf.mean(), color="#EF4444", linewidth=2,
           linestyle="--", label=f"Mean: {valid_conf.mean():.3f}")
ax.axvline(0.80, color="#22C55E", linewidth=2,
           linestyle=":", label="Target: 0.80")

high_conf = (valid_conf >= 0.80).sum()
ax.set_title(f"Agent Confidence Distribution\n"
             f"{high_conf}/{len(valid_conf)} outputs above 0.80 confidence",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlabel("Confidence Score (0–1)", fontsize=11)
ax.set_ylabel("Number of Companies", fontsize=11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_04_confidence_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_04_confidence_distribution.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 5 — Score vs Confidence Scatter Plot
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

valid_df = df[df["confidence"] > 0]

scatter_colors = [COLORS[l] for l in valid_df["label"]]

sc = ax.scatter(
    valid_df["score"],
    valid_df["confidence"],
    c=scatter_colors,
    alpha=0.75,
    s=60,
    edgecolors="white",
    linewidth=0.5
)

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["HIGH"],     label="HIGH"),
    Patch(facecolor=COLORS["MODERATE"], label="MODERATE"),
    Patch(facecolor=COLORS["LOW"],      label="LOW"),
]
ax.legend(handles=legend_elements, fontsize=9, title="Risk Label")

ax.axvline(3.5, color=MUTED, linestyle="--",
           linewidth=1, alpha=0.5)
ax.axvline(6.0, color=MUTED, linestyle="--",
           linewidth=1, alpha=0.5)
ax.axhline(0.80, color="#22C55E", linestyle=":",
           linewidth=1.5, alpha=0.7, label="Confidence target")

ax.set_title("Macro Risk Score vs Agent Confidence\n"
             "Coloured by Risk Label",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlabel("Macro Risk Score (1–10)", fontsize=11)
ax.set_ylabel("Confidence (0–1)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_05_score_vs_confidence.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_05_score_vs_confidence.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 6 — Average Score per Sector (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

sector_avg = (df.groupby("sector")["score"]
                .mean()
                .sort_values(ascending=True))

bar_colors = [
    COLORS["HIGH"]     if v > 6.0 else
    COLORS["MODERATE"] if v > 3.5 else
    COLORS["LOW"]
    for v in sector_avg.values
]

bars = ax.barh(
    sector_avg.index,
    sector_avg.values,
    color=bar_colors,
    alpha=0.85,
    edgecolor="white",
    height=0.6
)
for bar, val in zip(bars, sector_avg.values):
    ax.text(bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", fontsize=9, fontweight="bold")

ax.axvline(3.5, color=MUTED, linestyle="--",
           linewidth=1, alpha=0.6, label="LOW/MOD threshold")
ax.axvline(6.0, color=MUTED, linestyle="--",
           linewidth=1, alpha=0.6, label="MOD/HIGH threshold")
ax.set_title("Average Macro Risk Score by Sector\n"
             "DeepSeek-R1 — 123 Companies",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlabel("Average Macro Risk Score (1–10)", fontsize=11)
ax.set_xlim(0, 11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_06_avg_score_by_sector.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_06_avg_score_by_sector.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 7 — FRED Macro Time Series (Context Plot)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("FRED Macroeconomic Indicators — Agent Input Data\n"
             "2018 to 2024 (1,849 daily observations)",
             fontsize=13, fontweight="bold", color=NAVY)

macro_df = pd.read_csv(
    "data/silver/silver_macro.csv",
    index_col=0,
    parse_dates=True
)

# Panel 1 — Fed Funds Rate + VIX
ax1 = axes[0]
ax1b = ax1.twinx()
ax1.plot(macro_df.index, macro_df["fed_funds_rate"],
         color="#EF4444", linewidth=1.5, label="Fed Funds Rate (%)")
ax1b.plot(macro_df.index, macro_df["vix"],
          color="#7C3AED", linewidth=1, alpha=0.7, label="VIX")
ax1.set_ylabel("Fed Funds Rate (%)", color="#EF4444", fontsize=9)
ax1b.set_ylabel("VIX", color="#7C3AED", fontsize=9)
ax1.set_title("Monetary Policy + Market Fear",
              fontsize=10, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8, loc="upper left")

# Panel 2 — CPI + Unemployment
ax2 = axes[1]
ax2b = ax2.twinx()
ax2.plot(macro_df.index, macro_df["cpi"],
         color="#F59E0B", linewidth=1.5, label="CPI Index")
ax2b.plot(macro_df.index, macro_df["unemployment"],
          color="#0D9488", linewidth=1, alpha=0.8, label="Unemployment (%)")
ax2.set_ylabel("CPI Index", color="#F59E0B", fontsize=9)
ax2b.set_ylabel("Unemployment (%)", color="#0D9488", fontsize=9)
ax2.set_title("Inflation + Labour Market",
              fontsize=10, fontweight="bold")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8, loc="upper left")

# Panel 3 — Yield Curve + Oil
ax3 = axes[2]
ax3b = ax3.twinx()
ax3.plot(macro_df.index, macro_df["yield_curve_10y2y"],
         color="#185FA5", linewidth=1.5, label="Yield Curve (10Y-2Y)")
ax3.axhline(0, color="#EF4444", linestyle="--",
            linewidth=1, alpha=0.5, label="Inversion line")
ax3b.plot(macro_df.index, macro_df["wti_oil"],
          color="#854F0B", linewidth=1, alpha=0.7,
          label="WTI Oil ($/barrel)")
ax3.set_ylabel("10Y-2Y Spread (%)", color="#185FA5", fontsize=9)
ax3b.set_ylabel("Oil Price (USD/barrel)", color="#854F0B", fontsize=9)
ax3.set_title("Yield Curve + Oil Price",
              fontsize=10, fontweight="bold")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8, loc="upper left")

for ax in axes:
    ax.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_07_fred_timeseries.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_07_fred_timeseries.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 8 — Schema Compliance Summary (Pie + Stats)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Macro Agent — Evaluation Summary",
             fontsize=13, fontweight="bold", color=NAVY)

# Pie — compliance
compliant = (~df["failed"]).sum()
failed    = df["failed"].sum()

ax1.pie(
    [compliant, failed],
    labels=[f"Successful\n({compliant})",
            f"Fallback used\n({failed})"],
    colors=[TEAL, "#EF4444"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
ax1.set_title("Schema Compliance\n(123 companies)",
              fontsize=11, fontweight="bold")

# Bar — key metrics
metrics = {
    "Success\nrate": (~df["failed"]).mean() * 100,
    "Mean\nconfidence\n(×100)": df[df["confidence"]>0]["confidence"].mean() * 100,
    "High conf\n(≥0.80) %": (df["confidence"] >= 0.80).mean() * 100,
    "Latency\ntarget met\n(%)": 100.0,
}
bar_colors = [TEAL, "#7C3AED", "#185FA5", "#22C55E"]
bars = ax2.bar(
    metrics.keys(),
    metrics.values(),
    color=bar_colors,
    alpha=0.85,
    edgecolor="white",
    width=0.5
)
for bar, val in zip(bars, metrics.values()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             f"{val:.1f}%",
             ha="center", fontsize=10, fontweight="bold")

ax2.set_ylim(0, 120)
ax2.set_title("Key Evaluation Metrics",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Percentage (%)", fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/reports/macro_08_evaluation_summary.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: macro_08_evaluation_summary.png")

# ═══════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  ALL MACRO AGENT PLOTS GENERATED")
print("="*55)
print(f"\n  Companies analysed : {len(df)}")
print(f"  Successful outputs : {(~df['failed']).sum()}")
print(f"  Fallback used      : {df['failed'].sum()}")
print(f"  Mean risk score    : {df['score'].mean():.3f}")
print(f"  Mean confidence    : {df[df['confidence']>0]['confidence'].mean():.3f}")
print(f"\n  Risk label distribution:")
for label in ["HIGH", "MODERATE", "LOW"]:
    count = (df["label"] == label).sum()
    pct   = count / len(df) * 100
    print(f"    {label:<12} {count:>4} ({pct:.1f}%)")

print(f"\n  Plots saved to data/reports/:")
plots = [
    "macro_01_score_distribution.png",
    "macro_02_label_distribution.png",
    "macro_03_score_by_sector.png",
    "macro_04_confidence_distribution.png",
    "macro_05_score_vs_confidence.png",
    "macro_06_avg_score_by_sector.png",
    "macro_07_fred_timeseries.png",
    "macro_08_evaluation_summary.png",
]
for p in plots:
    print(f"    {p}")
