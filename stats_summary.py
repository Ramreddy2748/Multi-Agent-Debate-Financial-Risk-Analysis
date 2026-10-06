"""
stats_summary.py
Generates data statistics summary table for ISA outcome 3.6.
Run once to produce data/reports/data_statistics_summary.csv
"""

import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("data/reports").mkdir(parents=True, exist_ok=True)

bronze_edgar  = glob.glob("data/bronze/edgar_*_raw_long.csv")
bronze_yahoo  = glob.glob("data/bronze/yahoo_*.csv")
bronze_fred   = glob.glob("data/bronze/fred_*.csv")
silver_edgar  = glob.glob("data/silver/silver_edgar_*.csv")
silver_prices = glob.glob("data/silver/silver_prices_*.csv")

b_edgar_rows  = sum(len(pd.read_csv(f)) for f in bronze_edgar)
b_yahoo_rows  = sum(len(pd.read_csv(f)) for f in bronze_yahoo)
b_fred_rows   = sum(len(pd.read_csv(f)) for f in bronze_fred)
s_edgar_rows  = sum(len(pd.read_csv(f)) for f in silver_edgar)
s_price_rows  = sum(len(pd.read_csv(f)) for f in silver_prices)
g_rows        = len(pd.read_csv("data/gold/gold_risk_scores_ALL.csv"))
t_rows        = len(pd.read_csv("data/gold/split_train.csv"))
v_rows        = len(pd.read_csv("data/gold/split_val.csv"))
te_rows       = len(pd.read_csv("data/gold/split_test.csv"))

rows = [
    ["Bronze", "EDGAR — raw long format",    len(bronze_edgar), f"{b_edgar_rows:,}",  "Raw SEC filings, long format, no cleaning"],
    ["Bronze", "Yahoo Finance — price data", len(bronze_yahoo), f"{b_yahoo_rows:,}",  "Raw OHLCV + fundamentals per ticker"],
    ["Bronze", "FRED — macro indicators",    len(bronze_fred),  f"{b_fred_rows:,}",   "Fed funds rate, CPI, treasury, unemployment"],
    ["Silver", "EDGAR — cleaned financials", len(silver_edgar), f"{s_edgar_rows:,}",  "Pivoted wide, USD millions, deduped"],
    ["Silver", "Prices — feature engineered",len(silver_prices),f"{s_price_rows:,}", "Returns, volatility, SMA, normalised"],
    ["Gold",   "All risk scores",            1,                  f"{g_rows:,}",        "4 risk dimensions + composite score per company"],
    ["Split",  "Train set (70%)",            1,                  f"{t_rows:,}",        "Stratified by HIGH/MODERATE/LOW label"],
    ["Split",  "Validation set (15%)",       1,                  f"{v_rows:,}",        "Stratified by HIGH/MODERATE/LOW label"],
    ["Split",  "Test set (15%)",             1,                  f"{te_rows:,}",       "Stratified by HIGH/MODERATE/LOW label"],
]

df = pd.DataFrame(rows, columns=["Layer", "Source", "Files", "Rows", "Description"])
csv_path = "data/reports/data_statistics_summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print(df.to_string(index=False))

# ── Plot as table image ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

colors_layer = {
    "Bronze": "#FEF3C7",
    "Silver": "#DBEAFE",
    "Gold":   "#D1FAE5",
    "Split":  "#EDE9FE",
}

row_colors = [[colors_layer[r[0]]] * 5 for r in rows]

table = ax.table(
    cellText=[[r[0], r[1], str(r[2]), r[3], r[4]] for r in rows],
    colLabels=["Layer", "Source", "Files", "Rows", "Description"],
    cellLoc="left",
    loc="center",
    cellColours=row_colors,
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row == 0:
        cell.set_facecolor("#1F3864")
        cell.set_text_props(color="white", fontweight="bold")

col_widths = [0.08, 0.22, 0.06, 0.08, 0.35]
for col_idx, width in enumerate(col_widths):
    for row_idx in range(len(rows) + 1):
        table[row_idx, col_idx].set_width(width)

plt.title(
    "Data Statistics Summary — Raw to Prepared\n"
    f"Team 3 | DATA 298A | {g_rows:,} Companies | S&P 500 + NASDAQ-100",
    fontsize=11, fontweight="bold", color="#1F3864", pad=12
)
plt.tight_layout()

img_path = "data/reports/10_data_statistics_summary.png"
fig.savefig(img_path, dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {img_path}")