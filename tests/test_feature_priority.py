import csv
import tempfile
import unittest
from pathlib import Path

from agents.evaluation.feature_priority import build_priority_rows


class FeaturePriorityTests(unittest.TestCase):
    def test_builds_priority_and_freshness_rows_for_all_agents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            silver = root / "silver"
            silver.mkdir()
            (silver / "silver_macro.csv").write_text(
                "date,fed_funds_rate,cpi,treasury_10y,unemployment,gdp_growth,yield_curve_10y2y,vix,wti_oil\n"
                "2026-10-10,4.0,300,4.2,4.1,2.0,0.4,16,75\n",
                encoding="utf-8",
            )
            (silver / "silver_prices_TEST.csv").write_text(
                "date,beta\n2026-10-10,1.1\n",
                encoding="utf-8",
            )
            (silver / "silver_edgar_TEST.csv").write_text(
                "filed,cash_usd_m,long_term_debt,total_assets_usd_m,net_income_usd_m,operating_income_usd_m,revenue\n"
                "2026-08-01,100,200,1000,50,70,500\n",
                encoding="utf-8",
            )
            (silver / "silver_news_sentiment.csv").write_text(
                "ticker,date,sentiment_mean,sentiment_std,article_count\n"
                "TEST,2026-10-10,-0.2,0.4,4\n",
                encoding="utf-8",
            )
            gold = root / "gold.csv"
            gold.write_text(
                "ticker,ann_volatility_pct,max_drawdown_pct,ytd_return_pct,sharpe_ratio,beta,as_of_date\n"
                "TEST,30,-20,5,0.4,1.1,2026-10-10\n",
                encoding="utf-8",
            )

            rows = build_priority_rows(gold, silver)

        agents = {row["agent"] for row in rows}
        self.assertEqual(len(rows), 19)
        self.assertEqual(agents, {
            "Fundamental Agent",
            "Market + Volatility Agent",
            "Sentiment Agent",
            "Macro-Economic Agent",
        })
        self.assertTrue(all("priority_score" in row for row in rows))
        self.assertTrue(all("freshness_status" in row for row in rows))
        self.assertTrue(any(row["is_high_priority"] == "Y" for row in rows))


if __name__ == "__main__":
    unittest.main()
