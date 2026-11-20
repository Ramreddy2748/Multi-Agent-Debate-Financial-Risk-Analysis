import unittest

from agents.orchestrator.critic_agent import _sanitize_llm_verdict


class SanitizeLLMVerdictTests(unittest.TestCase):
    def test_out_of_range_score_and_wrong_label_are_fixed(self):
        result = _sanitize_llm_verdict(
            {
                "final_risk_score": 11,
                "final_risk_label": "LOW",
                "confidence": "0.9",
                "disagreement_detected": "false",
            }
        )
        self.assertEqual(result["final_risk_score"], 10.0)
        self.assertEqual(result["final_risk_label"], "HIGH")
        self.assertEqual(result["confidence"], 0.9)
        self.assertIs(result["disagreement_detected"], False)

    def test_unparseable_score_is_rejected(self):
        self.assertIsNone(_sanitize_llm_verdict({"final_risk_score": "high"}))

    def test_label_boundaries(self):
        cases = {3.5: "LOW", 3.51: "MODERATE", 6.0: "MODERATE", 6.01: "HIGH"}
        for score, label in cases.items():
            with self.subTest(score=score):
                self.assertEqual(
                    _sanitize_llm_verdict({"final_risk_score": score})["final_risk_label"],
                    label,
                )


if __name__ == "__main__":
    unittest.main()
