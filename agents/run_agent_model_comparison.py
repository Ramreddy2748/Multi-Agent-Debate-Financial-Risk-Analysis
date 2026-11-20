"""Run task-specific LLM comparisons for the financial risk agents.

The production agents remain the trained/local models already configured in
the project. This runner compares different remote LLM rosters by task:

  Fundamental: larger reasoning models for leverage, liquidity, and margins
  Market:      faster models for numeric volatility and price-risk analysis

Examples:
  python3 agents/run_agent_model_comparison.py fundamental AAPL MSFT NVDA
  python3 agents/run_agent_model_comparison.py market AAPL MSFT NVDA
  python3 agents/run_agent_model_comparison.py all AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.fundamental import llm_compare as fundamental
from agents.market import llm_compare as market
from agents.shared import llm_comparison


# Different rosters reflect the evidence and reasoning needs of each agent.
MODEL_ROSTERS = {
    "fundamental": [
        "groq_gptoss120b",
        "openrouter_nemotron",
        "openrouter_qwen27b",
    ],
    "market": [
        "groq_gptoss20b",
        "openrouter_qwen27b",
        "groq_gptoss120b",
    ],
}

AGENT_CONFIG = {
    "fundamental": {
        "label": fundamental.AGENT_LABEL,
        "context_loader": fundamental.context_loader,
        "prompt_builder": fundamental.prompt_builder,
        "judge_prompt_builder_factory": fundamental.judge_prompt_builder_factory,
        "system_prompt": fundamental.SYSTEM_PROMPT,
        "judge_system_prompt": fundamental.JUDGE_SYSTEM_PROMPT,
        "output": PROJECT_ROOT / "data" / "gold" / "fundamental_llm_comparison.json",
        "production_models": [
            "Fine-tuned FinBERT",
            "Rule-based fundamental fallback",
        ],
    },
    "market": {
        "label": market.AGENT_LABEL,
        "context_loader": market.context_loader,
        "prompt_builder": market.prompt_builder,
        "judge_prompt_builder_factory": market.judge_prompt_builder_factory,
        "system_prompt": market.SYSTEM_PROMPT,
        "judge_system_prompt": market.JUDGE_SYSTEM_PROMPT,
        "output": PROJECT_ROOT / "data" / "gold" / "market_sentiment_llm_comparison.json",
        "production_models": [
            "Qwen 1.5B market LoRA",
            "Mistral 7B market LoRA candidate",
            "Rule-based volatility fallback",
        ],
    },
}


def run(agent_name: str, tickers: list[str], models: list[str] | None, output: Path) -> None:
    config = AGENT_CONFIG[agent_name]
    selected_models = models or MODEL_ROSTERS[agent_name]
    unknown = sorted(set(selected_models) - set(llm_comparison.MODELS))
    if unknown:
        raise SystemExit(
            f"Unknown model key(s): {', '.join(unknown)}. "
            f"Available keys: {', '.join(llm_comparison.MODELS)}"
        )

    print(f"Agent: {agent_name}")
    print(f"Models: {', '.join(selected_models)}")
    print(f"Production candidates: {', '.join(config['production_models'])}")

    llm_comparison.compare_models(
        config["label"],
        [ticker.upper() for ticker in tickers],
        config["context_loader"],
        config["prompt_builder"],
        config["judge_prompt_builder_factory"],
        config["system_prompt"],
        config["judge_system_prompt"],
        models=selected_models,
        out_path=str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task-specific LLMs for the risk agents.")
    parser.add_argument("agent", choices=["fundamental", "market", "all"])
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example AAPL MSFT NVDA")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(llm_comparison.MODELS),
        help="Override the task-specific model roster.",
    )
    parser.add_argument("--out", type=Path, help="Output JSON path for a single-agent run.")
    args = parser.parse_args()

    agents = list(AGENT_CONFIG) if args.agent == "all" else [args.agent]
    for name in agents:
        output = args.out if args.out and len(agents) == 1 else AGENT_CONFIG[name]["output"]
        run(name, args.tickers, args.models, output)


if __name__ == "__main__":
    main()
