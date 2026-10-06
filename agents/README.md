# Agents Layout

The agent code is grouped by responsibility:

- `fundamental/` - SEC/fundamental risk, FinBERT runtime, and FinBERT training data.
- `market/` - Yahoo price risk, news/market LoRA training and inference.
- `sentiment/` - NewsAPI headline sentiment with ProsusAI/finbert.
- `macro/` - FRED macro-economic risk and macro LLM comparison.
- `orchestrator/` - critic synthesis, debate state machine, verdict reports, monitoring, and Policy A-H evidence.
- `shared/` - reusable comparison and retrieval helpers.

Top-level files such as `agents/verdict_report.py` are compatibility wrappers.
They keep existing commands working while the real implementation lives in the
organized package folders.
