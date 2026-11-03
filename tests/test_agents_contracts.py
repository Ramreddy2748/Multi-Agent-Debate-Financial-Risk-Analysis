import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.fundamental.rule_based import FundamentalAnalysisAgent
from agents.orchestrator.critic_agent import (
    _enforce_review_guardrails,
    _low_confidence_reason,
    _normalize_market_lora_output,
    _weighted_critic,
    run_specialist_agents,
)
from agents.orchestrator.debate_state_machine import run_debate_state_machine
from agents.sentiment.finbert import run_sentiment_agent
from agents.fundamental.llm_compare import prompt_builder as fundamental_prompt_builder
from agents.market.llm_compare import prompt_builder as market_prompt_builder
from agents.market.infer_lora import extract_prediction


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

    def test_self_healing_retries_invalid_specialist_output(self):
        valid = {
            "agent": "Fundamental Agent",
            "risk_score": 4.0,
            "risk_label": "MODERATE",
            "confidence": 0.8,
            "evidence": [],
        }
        with patch.dict(os.environ, {"SYSTEM_MODE": "self_healing"}, clear=False), \
             patch("agents.orchestrator.critic_agent.run_fundamental_agent", side_effect=[{}, valid]), \
             patch("agents.orchestrator.critic_agent.run_market_sentiment_agent", return_value={**valid, "agent": "Market + Volatility Agent"}), \
             patch("agents.orchestrator.critic_agent.run_sentiment_agent", return_value={**valid, "agent": "Sentiment Agent"}), \
             patch("agents.orchestrator.critic_agent.run_macro_agent_safe", return_value={**valid, "agent": "Macro-Economic Agent"}):
            outputs = run_specialist_agents("TEST", "Test Co", "Industrials")

        self.assertEqual(outputs[0]["attempts"], 2)
        self.assertTrue(all(output["execution_mode"] == "self_healing" for output in outputs))

    def test_self_healing_retries_internal_parse_error_signal(self):
        valid = {
            "agent": "Market + Volatility Agent",
            "risk_score": 4.0,
            "risk_label": "MODERATE",
            "confidence": 0.8,
            "evidence": [],
        }
        bad = {**valid, "overall_assessment": "JSON parse error."}
        with patch.dict(os.environ, {"SYSTEM_MODE": "self_healing"}, clear=False), \
             patch("agents.orchestrator.critic_agent.run_fundamental_agent", return_value={**valid, "agent": "Fundamental Agent"}), \
             patch("agents.orchestrator.critic_agent.run_market_sentiment_agent", side_effect=[bad, valid]), \
             patch("agents.orchestrator.critic_agent.run_sentiment_agent", return_value={**valid, "agent": "Sentiment Agent"}), \
             patch("agents.orchestrator.critic_agent.run_macro_agent_safe", return_value={**valid, "agent": "Macro-Economic Agent"}):
            outputs = run_specialist_agents("TEST", "Test Co", "Industrials")

        self.assertEqual(outputs[1]["attempts"], 2)

    def test_self_healing_retries_low_confidence_output(self):
        low = {
            "agent": "Fundamental Agent",
            "risk_score": 4.0,
            "risk_label": "MODERATE",
            "confidence": 0.30,
            "evidence": [],
        }
        recovered = {**low, "confidence": 0.80}
        valid_market = {**recovered, "agent": "Market + Volatility Agent"}
        valid_sentiment = {**recovered, "agent": "Sentiment Agent"}
        valid_macro = {**recovered, "agent": "Macro-Economic Agent"}
        with patch.dict(os.environ, {"SYSTEM_MODE": "self_healing", "SELF_HEALING_MIN_CONFIDENCE": "0.50"}, clear=False), \
             patch("agents.orchestrator.critic_agent.run_fundamental_agent", side_effect=[low, recovered]), \
             patch("agents.orchestrator.critic_agent.run_market_sentiment_agent", return_value=valid_market), \
             patch("agents.orchestrator.critic_agent.run_sentiment_agent", return_value=valid_sentiment), \
             patch("agents.orchestrator.critic_agent.run_macro_agent_safe", return_value=valid_macro):
            outputs = run_specialist_agents("TEST", "Test Co", "Industrials")

        self.assertEqual(outputs[0]["attempts"], 2)

    def test_low_confidence_reason_uses_probability_margin(self):
        reason = _low_confidence_reason(
            {
                "agent": "Fundamental Agent",
                "confidence": 0.4864,
                "raw_probabilities": {
                    "LOW": 0.4864,
                    "MODERATE": 0.4720,
                    "HIGH": 0.0416,
                },
                "evidence": ["SEC-derived fields were available."],
            },
            0.50,
        )

        self.assertIn("below the 0.50 threshold", reason)
        self.assertIn("top two class probabilities are close", reason)
        self.assertIn("margin 0.0144", reason)

    def test_debate_revises_contradictory_agents_and_keeps_review_flag(self):
        contradictory = [
            {"agent": "Fundamental Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.40, "evidence": ["Debt is elevated."]},
            {"agent": "Market + Volatility Agent", "risk_score": 2.0, "risk_label": "LOW", "confidence": 0.80, "evidence": ["Beta is 2.2.", "Volatility is 39%."]},
            {"agent": "Sentiment Agent", "risk_score": 8.0, "risk_label": "HIGH", "confidence": 0.90, "evidence": ["One negative headline."]},
            {"agent": "Macro-Economic Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.70, "evidence": ["Macro conditions are mixed."]},
        ]
        with patch("agents.orchestrator.debate_state_machine.run_specialist_agents", return_value=contradictory):
            report = run_debate_state_machine("TEST", max_revision_rounds=2)

        self.assertEqual(report["final_output"]["revision_rounds"], 2)
        self.assertTrue(report["final_output"]["requires_human_review"])
        self.assertTrue(any("reassess" in output.get("revision_request", "") for output in report["agent_outputs"]))
        self.assertTrue(all(output.get("revision_history") for output in report["agent_outputs"]))

    def test_market_lora_conflict_is_calibrated(self):
        result = _normalize_market_lora_output(
            {"risk_score": 2.58, "risk_label": "LOW", "confidence": 1.0},
            {"annualized_volatility": 0.39, "beta": 2.22, "max_drawdown": -0.20, "price_vs_52w_high_pct": -1.65},
            fallback_score=6.40,
        )

        self.assertEqual(result["risk_score"], 4.49)
        self.assertEqual(result["risk_label"], "MODERATE")
        self.assertEqual(result["confidence"], 0.55)
        self.assertTrue(result["calibration_applied"])

    def test_freshness_mismatch_forces_human_review(self):
        outputs = [
            {"agent": "Fundamental Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.7, "as_of_date": "2025-09-27"},
            {"agent": "Market + Volatility Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.7, "as_of_date": "2026-10-30"},
            {"agent": "Sentiment Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.7},
            {"agent": "Macro-Economic Agent", "risk_score": 5.0, "risk_label": "MODERATE", "confidence": 0.7, "as_of_date": "2026-09-30"},
        ]
        result = _enforce_review_guardrails(
            {"final_risk_score": 5.0, "final_risk_label": "MODERATE", "confidence": 0.8, "final_decision": "Review."},
            outputs,
        )

        self.assertTrue(result["requires_human_review"])
        self.assertEqual(result["confidence"], 0.55)
        self.assertTrue(result["freshness_warnings"])

    def test_llm_comparison_prompts_include_curated_financial_rag(self):
        fundamental_context = {
            "company": "Test Co",
            "edgar": {"cash_usd_m": 100, "long_term_debt_usd_m": 250},
            "chunks": [{"text": "Test Co debt evidence."}],
            "financial_terms_source": "data/rag/financial_terms.jsonl",
            "financial_terms": "- Debt-to-equity ratio | Thresholds: HIGH: above 1.5",
        }
        market_context = {
            "company": "Test Co",
            "market_features": {"beta": 1.4},
            "sentiment_features": {"article_count": 5},
            "sentiment_confidence": 0.8,
            "chunks": [{"text": "Test Co volatility evidence."}],
            "financial_terms_source": "data/rag/financial_terms.jsonl",
            "financial_terms": "- Beta | Thresholds: HIGH: above 1.3",
        }

        self.assertIn("Debt-to-equity ratio", fundamental_prompt_builder("TEST", fundamental_context))
        self.assertIn("Beta | Thresholds", market_prompt_builder("TEST", market_context))

    def test_lora_inference_extracts_last_json_and_normalizes_label(self):
        raw = 'prompt text {"market_risk_score": 4.5, "market_risk_label": "moderate"} '
        raw += '{"market_risk_score": 6.2, "market_risk_label": "high", "explanation": "volatile"}'
        result = extract_prediction(raw)

        self.assertEqual(result["market_risk_score"], 6.2)
        self.assertEqual(result["market_risk_label"], "HIGH")


if __name__ == "__main__":
    unittest.main()
