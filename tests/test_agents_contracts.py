import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.fundamental.rule_based import FundamentalAnalysisAgent
from agents.orchestrator.critic_agent import _weighted_critic
from agents.sentiment.finbert import run_sentiment_agent


class AgentContractTests(unittest.TestCase):
    def test_fundamental_agent_reads_sec_fields_and_returns_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "silver_edgar_TEST.csv"
            fields = [
                "end_date", "total_assets_usd_m", "total_liabilities_usd_m",
                "stockholders_equity_usd_m", "cash_usd_m", "net_income_usd_m",
                "operating_income_usd_m", "long_term_debt",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "end_date": "2025-12-31",
                    "total_assets_usd_m": "1000",
                    "total_liabilities_usd_m": "400",
                    "stockholders_equity_usd_m": "600",
                    "cash_usd_m": "150",
                    "net_income_usd_m": "100",
                    "operating_income_usd_m": "120",
                    "long_term_debt": "200",
                })

            result = FundamentalAnalysisAgent(silver_dir=temp_dir).analyze("TEST")

            self.assertEqual(result.agent, "fundamental_analysis_agent")
            self.assertIn("debt_to_assets", result.features)
            self.assertIn("Cash/debt", " ".join(result.evidence))
            self.assertIn(result.risk_label, {"LOW", "MODERATE", "HIGH"})

    def test_sentiment_agent_has_explicit_no_news_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("agents.sentiment.finbert.BRONZE_DIR", Path(temp_dir)), \
                 patch("agents.sentiment.finbert.SILVER_DIR", Path(temp_dir)):
                result = run_sentiment_agent("TEST")

            self.assertEqual(result["agent"], "Sentiment Agent")
            self.assertEqual(result["claim_type"], "NO_NEWS_FALLBACK")
            self.assertEqual(result["metrics"]["article_count"], 0)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is required for Macro Agent tests")
    def test_macro_agent_returns_fallback_contract_without_artifacts(self):
        from agents.macro.agent import run_macro_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {
                "MACRO_V2_DATA_ROOT": temp_dir,
                "MACRO_V2_USE_LORA_FORECAST": "0",
            }, clear=False):
                result = run_macro_agent("TEST", "Industrials")

            self.assertEqual(result["agent"], "Macro-Economic Agent")
            self.assertEqual(result["claim_type"], "MACRO_V2_UNAVAILABLE")
            self.assertEqual(result["risk_label"], "MODERATE")

    def test_critic_combines_all_four_specialist_outputs(self):
        outputs = [
            {"agent": "Fundamental Agent", "risk_score": 2.0, "risk_label": "LOW"},
            {"agent": "Market + Volatility Agent", "risk_score": 4.0, "risk_label": "MODERATE"},
            {"agent": "Sentiment Agent", "risk_score": 7.0, "risk_label": "HIGH"},
            {"agent": "Macro-Economic Agent", "risk_score": 5.0, "risk_label": "MODERATE"},
        ]

        result = _weighted_critic(outputs, query="Can I invest?")

        self.assertIn(result["final_risk_label"], {"LOW", "MODERATE", "HIGH"})
        self.assertTrue(result["disagreement_detected"])
        self.assertEqual(len(result["agent_scorecard"]), 4)


if __name__ == "__main__":
    unittest.main()
