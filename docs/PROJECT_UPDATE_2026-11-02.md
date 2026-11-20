# Financial Risk Multi-Agent Project
## Consolidated Project Update

**Updated:** 2026-11-02  
**Project:** DATA 298A / 298B Financial Risk Multi-Agent System

## 1. Current System

The project is an auditable stock-risk application that converts raw market,
company, macroeconomic, and news data into specialist risk assessments and a
final debate-based verdict.

The system contains four specialist agents and one critic/orchestrator:

| Agent | Primary data | Current inference |
|---|---|---|
| Fundamental Agent | SEC EDGAR financial statements | Fine-tuned FinBERT, with rule fallback |
| Market and Volatility Agent | Yahoo Finance price history | Market LoRA candidate, with calibrated rule fallback |
| Sentiment Agent | NewsAPI headlines | FinBERT sentiment, with VADER/no-news fallback |
| Macro-Economic Agent | FRED observations and sector ETF context | Macro v2 Phi-4-mini LoRA, with FRED rule fallback |
| Critic / Orchestrator | All specialist outputs and evidence | Claude when configured, otherwise local weighted critic |

The application is exposed through FastAPI and an interactive frontend. Each
verdict includes scores, labels, confidence, evidence, data freshness, policy
checks, contradictions, and execution metadata.

## 2. End-to-End Data Flow

### Bronze ingestion

The ingestion layer preserves source responses as raw artifacts:

- Yahoo Finance: OHLCV prices and derived market inputs.
- SEC EDGAR: company facts and annual financial records.
- FRED: interest rates, CPI, unemployment, GDP, yield spread, VIX, and WTI.
- NewsAPI: ticker-specific financial headlines.

Raw files are retained so the system can answer what was received from each
source at the time of ingestion.

### Silver cleaning and validation

Cleaning standardizes dates, numeric fields, schemas, duplicates, and missing
values. News dates accept mixed ISO formats and are normalized in UTC. News
headlines receive VADER baseline sentiment and are also available for FinBERT
inference. Clean CSV and Parquet artifacts are generated for downstream use.

Invalid or incomplete data is surfaced as a warning or fallback condition; it
is not silently treated as reliable evidence.

### Gold features and risk tables

Gold outputs contain query-ready company features and risk scores, including:

- Leverage and liquidity features.
- Profit and operating margins.
- Annualized volatility, beta, drawdown, and price-to-high measures.
- News article count and sentiment aggregates.
- Macro conditions and sector context.
- Feature priority and freshness reports.
- Fine-tuning JSONL datasets and evaluation artifacts.

The complete ticker universe and consolidated Gold tables support the expanded
company coverage. News coverage remains thinner than price and SEC coverage,
so zero-article sentiment is reported as unavailable rather than neutral.

## 3. Specialist Agents

### Fundamental Agent

The agent reads the latest Silver SEC record for a ticker and uses cash,
long-term debt, total liabilities, equity, revenue, net income, and operating
income. It derives leverage, liquidity, profit margin, and operating margin
signals.

When enabled, the fine-tuned FinBERT classifier returns a risk label,
probability-based confidence, score, raw probabilities, and evidence. If the
model or SEC data is unavailable, the rule-based fundamental path remains
available and marks the result appropriately.

### Market and Volatility Agent

The agent reads Yahoo-derived Silver price history and calculates or consumes:

- Annualized volatility.
- Beta.
- Maximum drawdown.
- Price versus 52-week high.
- YTD return.
- Rolling 30-day and 60-day volatility.

The Qwen market LoRA adapter and Mistral 7B market LoRA are experimental model
paths. A deterministic feature-based score remains available. If a generated
model score conflicts strongly with the underlying volatility or beta data,
the system calibrates the score toward the independent feature score and caps
confidence.

### Sentiment Agent

The sentiment agent reads NewsAPI headlines by ticker and reports:

- Positive, neutral, and negative headline counts.
- Average FinBERT sentiment value.
- Model confidence.
- Article count.
- Top headline evidence.

The project uses ProsusAI/FinBERT for financial sentiment inference. VADER is a
baseline fallback. If there are no usable headlines, the result is explicitly
`NO_NEWS_FALLBACK` and the critic can require review.

### Macro-Economic Agent

The macro agent uses FRED series such as fed funds, CPI, 10-year Treasury,
unemployment, GDP growth, 10Y-2Y spread, VIX, and WTI oil. Macro v2 uses a
Phi-4-mini LoRA forecast with sector ETF mapping. The rule-based FRED path is
retained when the LoRA artifacts or required data are unavailable.

Macro is currently a supporting agent. The primary project focus remains
fundamental risk, market volatility, and news sentiment.

## 4. RAG and Evidence Grounding

The project uses two retrieval layers.

### Ticker-specific retrieval

`agents/shared/retrieval.py` and `data/gold/vector_store/` retrieve evidence
from company-specific Silver and Gold content. Retrieval is filtered by ticker
to prevent evidence from one company being assigned to another.

### Curated financial-terms retrieval

The following files define the curated financial knowledge base:

- `data/rag/financial_terms.jsonl`
- `data/rag/agent_retrieval_config.json`
- `data/rag/README.md`
- `agents/shared/financial_rag.py`

The terms file contains definitions, formulas, thresholds, priority rules, and
required evidence fields for leverage, cash coverage, margins, volatility,
beta, drawdown, sentiment, article count, confidence, and disagreement.

Fundamental and market LLM comparison prompts now receive both company-specific
evidence and the relevant curated financial rules. This allows the model to
interpret a metric instead of scoring a number without context.

## 5. Critic and Debate Workflow

The debate protocol is implemented in:

- `agents/orchestrator/debate_state_machine.py`
- `agents/orchestrator/critic_agent.py`

The workflow is:

1. Collect specialist outputs.
2. Detect label disagreement, score dispersion, low confidence, fallback use,
   missing news, and freshness problems.
3. Reassess affected agents using their supplied evidence and the contradiction
   context.
4. Re-run contradiction detection.
5. Synthesize the final verdict with Claude when configured, or use the local
   weighted critic.
6. Apply review guardrails and policy evidence checks.

The explicit revision loop is bounded to a maximum of two rounds. Each revised
agent output records:

- `revision_target`
- `revision_request`
- `revision_history`
- `pre_revision_score`
- `revision_note`

If disagreement remains after revision, the final result keeps
`requires_human_review: true`. The system does not repeatedly retry until a
preferred score or label appears.

## 6. Self-Healing Reliability

Execution modes are selected with:

```bash
SYSTEM_MODE=baseline
SYSTEM_MODE=self_healing
```

Self-healing retries an affected specialist once when it detects:

- Invalid or incomplete JSON.
- Timeout or model/API failure.
- Internal parse-error signals.
- Confidence below `SELF_HEALING_MIN_CONFIDENCE`.
- Missing or fallback evidence.

The result records attempt count, data-quality warnings, and healing actions.
If the retry also fails, the system returns a conservative structured fallback
and flags the result for review. Low confidence is explained using available
model probabilities and evidence; the confidence value is never fabricated.

Example configuration:

```env
SELF_HEALING_MIN_CONFIDENCE=0.50
```

## 7. Model Training and Evaluation

### Fundamental FinBERT

The fine-tuned fundamental model is stored outside the source repository under
the local or Drive model directory. The saved evaluation run reported:

- Accuracy: approximately `0.573`.
- Macro F1: approximately `0.405`.
- Evaluation loss: approximately `0.968`.

### Market LoRA models

Qwen 1.5B and Mistral 7B market adapters were trained as generative
JSON-output models. Their evaluation must parse generated JSON and compare the
predicted risk label against a held-out validation set.

An important data-quality correction was completed: the original market JSONL
contained `gold_*` target fields inside the input features. That was target
leakage because the target label was derived from those fields. The clean data
builder now removes all `gold_*` fields, and the evaluator rejects datasets
that contain them.

The earlier perfect Mistral result must not be reported as valid model quality.
The clean adapter must be retrained and evaluated on a clean validation split.
The clean dataset currently has class imbalance, so balanced training and
validation results should be reported before connecting Mistral to production.

### Model comparison

The comparison tools support local and remote models, including GPT-OSS,
OpenRouter fallback models, Qwen, Mistral, FinBERT, and the rule baseline.
Remote model scores are estimates generated from real local evidence; they are
not ground-truth labels and must be distinguished from supervised metrics.

Relevant scripts:

- `agents/fundamental/llm_compare.py`
- `agents/market/llm_compare.py`
- `agents/market/evaluate_lora.py`
- `agents/evaluation/compare_agent_evaluations.py`
- `agents/evaluation/evaluate_market_rule.py`

## 8. API and Frontend

FastAPI serves the application and interactive documentation at `/docs`. The
API supports company lookup, prices, saved verdicts, monitoring, feature
priority, comparison, debate review, and model-training dashboard data.

The frontend provides:

- Natural-language investment questions.
- Direct Ask, full-agent verdict, and debate review actions.
- Five agent scorecards.
- Contradiction and revision details.
- Confidence and review state.
- Price and risk charts.
- Company comparison with fresh debate runs.
- RAG, data freshness, and policy evidence visibility.
- Training curve and model-status display.

The UI should treat a missing-news result, fallback result, or unresolved
contradiction as a visible limitation rather than presenting it as a normal
high-confidence prediction.

## 9. Verification Completed

The automated contract suite currently passes:

```text
Ran 17 tests
OK
```

The tests cover:

- SEC fundamental contract.
- No-news sentiment fallback.
- Macro fallback contract.
- Four-agent critic composition.
- Self-healing retries for invalid output, parse errors, and low confidence.
- Low-confidence explanation generation.
- Market LoRA calibration.
- RAG prompt inclusion.
- Inference JSON extraction.
- Freshness review guardrails.
- Contradictory-agent self-revision with two bounded rounds.

Additional verification artifacts include:

- `data/reports/rag_self_healing_verification.json`
- `data/reports/multi_agent_model_evaluation.json`
- `data/reports/multi_agent_model_evaluation.md`
- `models/market_rule/evaluation_metrics.json`

## 10. Remaining Work

The following items remain before calling the system production-ready:

1. Retrain Mistral on the clean, leakage-free market dataset.
2. Balance minority classes during training without changing the validation
   distribution.
3. Re-evaluate Mistral, Qwen, and the rule baseline on exactly the same clean
   holdout.
4. Calibrate confidence using a separate calibration set.
5. Increase and continuously refresh NewsAPI coverage.
6. Add stronger authentication, rate limiting, secrets management, and cloud
   deployment controls for production hosting.
7. Monitor latency, parse success, fallback rate, freshness, disagreement, and
   human-review rate over time.

## 11. Reproducibility Commands

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

Verify RAG and self-healing:

```bash
.venv/bin/python agents/evaluation/verify_rag_self_healing.py
```

Run the debate state machine with two revision rounds:

```bash
SYSTEM_MODE=self_healing \
.venv/bin/python agents/debate_state_machine.py AAPL \
  --query "Can I invest in Apple?" \
  --max-revision-rounds 2 \
  --out data/verdicts/AAPL_debate_state_machine.json
```

Run the API:

```bash
SYSTEM_MODE=self_healing \
.venv/bin/uvicorn src.api.server:app \
  --host 127.0.0.1 --port 8766
```

Open:

```text
http://127.0.0.1:8766/frontend/index.html
http://127.0.0.1:8766/docs
```

## 12. Interpretation and Governance

The application provides research guidance, not personal financial advice.
Every final verdict should be interpreted with its evidence, confidence,
freshness, fallback state, contradictions, and human-review flag. A lower
confidence score is a valid result when the data or models do not support a
stronger conclusion.
