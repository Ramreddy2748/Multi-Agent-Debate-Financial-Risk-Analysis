import csv
import json
import tempfile
import unittest
from pathlib import Path

from agents.market.build_finetune_data import _target_from_gold, build_example
from agents.market.rule_based import MarketAgent


class MarketAgentSeparationTests(unittest.TestCase):
    def _write_prices(self, directory: Path) -> None:
        rows = []
        for index in range(1, 131):
            rows.append({
                "date": f"2025-{(index - 1) // 30 + 1:02d}-{(index - 1) % 28 + 1:02d}",
                "close": str(100 + index * 0.25),
                "daily_return": "0.002",
                "rolling_vol_30d": "0.20",
                "rolling_vol_60d": "0.18",
                "beta": "1.10",
                "price_vs_52w_high_pct": "-5.0",
                "sma_50": "120.0",
            })
        with (directory / "silver_prices_TEST.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_market_agent_ignores_news_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self._write_prices(directory)
            # This deliberately extreme file proves the market agent does not read it.
            (directory / "silver_news_sentiment.csv").write_text(
                "ticker,sentiment_mean,article_count\nTEST,-1.0,9999\n",
                encoding="utf-8",
            )

            result = MarketAgent(silver_dir=str(directory)).analyze("TEST")

            self.assertEqual(result.agent, "market_agent")
            self.assertNotIn("sentiment_mean", result.features)
            self.assertNotIn("article_count", result.features)
            self.assertTrue(all("news" not in item.lower() for item in result.evidence))

    def test_market_training_target_uses_volatility_only(self):
        score, label = _target_from_gold({
            "volatility_risk": "2.0",
            "sentiment_risk": "10.0",
            "composite_risk": "8.0",
        })
        self.assertEqual(score, 2.0)
        self.assertEqual(label, "LOW")

    def test_training_example_has_market_only_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self._write_prices(directory)
            example = build_example(
                {
                    "ticker": "TEST",
                    "sector": "Industrials",
                    "volatility_risk": "4.0",
                    "sentiment_risk": "9.0",
                    "composite_risk": "6.0",
                    "risk_label": "MODERATE",
                },
                MarketAgent(silver_dir=str(directory)),
            )

            self.assertIsNotNone(example)
            user = json.loads(example["messages"][1]["content"])
            assistant = json.loads(example["messages"][2]["content"])
            self.assertEqual(user["feature_schema"], "market_only_v1")
            self.assertNotIn("sentiment_mean", user["features"])
            self.assertNotIn("article_count", user["features"])
            self.assertEqual(assistant["market_risk_label"], "MODERATE")


if __name__ == "__main__":
    unittest.main()
