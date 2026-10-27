# RAG Data Layer

This folder contains the source-of-truth configuration for the project RAG layer.
It is intentionally small and Git-friendly. Generated vector indexes stay outside
this folder and can be rebuilt from the project data.

## Files

- `financial_terms.jsonl`  
  Curated financial definitions, thresholds, risk-priority rules, and policy
  guidance. Each line is one JSON object so the file can be appended to over time.

- `agent_retrieval_config.json`  
  Maps each production agent to the data sources and financial terms it should
  retrieve before producing or reviewing a risk claim.

## Source Data Used By RAG

The RAG layer should retrieve from these project datasets:

- `data/silver/silver_edgar_*.csv`  
  SEC-derived fundamentals for the Fundamental Agent.

- `data/silver/silver_prices_*.csv`  
  Yahoo Finance price, beta, volatility, drawdown, and market features for the
  Market Agent.

- `data/silver/silver_news_sentiment.csv`  
  Cleaned and filtered headline sentiment aggregates for the Sentiment Agent.

- `data/bronze/newsapi_headlines_raw.csv`  
  Raw headline-level news records retained for audit and re-cleaning.

- `data/gold/gold_risk_scores_ALL.csv`  
  Gold-layer feature table and project risk labels.

## Generated Index

Runtime retrieval currently uses the Chroma vector store at:

```text
data/gold/vector_store/
```

This is a generated artifact, not source data. Rebuild it after refreshing Silver
or news data:

```bash
python3 agents/retrieval.py --build-all
```

The larger prototype RAG builder lives at:

```text
RAG/rag.py
```

It can build a separate Chroma database from the same source data:

```bash
python3 RAG/rag.py
```

## Query Curated Financial Rules

Use the lightweight runtime helper when you want only the curated financial
rules without loading a vector database:

```bash
python3 -m agents.shared.financial_rag \
  --agent fundamental_agent \
  --query "cash debt leverage profit margin"
```

Combine Silver evidence retrieval with curated RAG rules:

```bash
python3 agents/retrieval.py AAPL \
  --query "leverage and profitability risk" \
  --agent fundamental_agent
```

## Agent Priority Rule

Agents should prioritize evidence in this order:

1. Project numeric data from Silver/Gold tables.
2. Curated financial rules from `financial_terms.jsonl`.
3. Model output from FinBERT, LoRA, Claude, or fallback logic.
4. General narrative context from retrieved documents.

This prevents the LLM from overriding audited project data with generic finance
language.
