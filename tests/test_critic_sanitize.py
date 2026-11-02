from agents.orchestrator.critic_agent import _sanitize_llm_verdict


def test_out_of_range_score_and_wrong_label_are_fixed():
    result = _sanitize_llm_verdict(
        {
            "final_risk_score": 11,
            "final_risk_label": "LOW",
            "confidence": "0.9",
            "disagreement_detected": "false",
        }
    )
    assert result["final_risk_score"] == 10.0
    assert result["final_risk_label"] == "HIGH"
    assert result["confidence"] == 0.9
    assert result["disagreement_detected"] is False


def test_unparseable_score_is_rejected():
    assert _sanitize_llm_verdict({"final_risk_score": "high"}) is None


def test_label_boundaries():
    assert _sanitize_llm_verdict({"final_risk_score": 3.5})["final_risk_label"] == "LOW"
    assert _sanitize_llm_verdict({"final_risk_score": 3.51})["final_risk_label"] == "MODERATE"
    assert _sanitize_llm_verdict({"final_risk_score": 6.0})["final_risk_label"] == "MODERATE"
    assert _sanitize_llm_verdict({"final_risk_score": 6.01})["final_risk_label"] == "HIGH"
