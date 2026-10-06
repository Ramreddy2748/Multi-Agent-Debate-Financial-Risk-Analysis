const api = {
  companies: "/api/companies",
  monitoring: (ticker = "") => `/api/monitoring${ticker ? `?ticker=${encodeURIComponent(ticker)}` : ""}`,
  verdict: (ticker) => `/api/verdicts/${ticker}`,
  prices: (ticker) => `/api/companies/${ticker}/prices`,
  generateVerdict: "/api/verdict",
  compare: (tickers) => `/api/compare?tickers=${encodeURIComponent(tickers.join(","))}`,
  pdf: (ticker) => `/api/artifacts/${ticker}/pdf`,
  modelTraining: "/api/models/training",
};

const state = {
  rows: [],
  filteredRows: [],
  selectedTicker: "AAPL",
  activeFilter: "ALL",
  current: null,
  monitoring: [],
  comparePair: null,
  comparePayload: null,
  modelTraining: null,
  generating: false,
  reportCategory: "pipeline",
  renderedCount: 60,
};

const REPORT_CATEGORIES = [
  {
    id: "pipeline",
    label: "Data & Pipeline",
    files: [
      "01_price_trends.png", "02_volatility.png", "03_return_distributions.png",
      "04_correlation_matrix.png", "05_risk_scores.png", "08_class_balance.png",
      "09_sector_distribution.png", "10_data_statistics_summary.png",
    ],
  },
  {
    id: "macro",
    label: "Macro Agent",
    files: [
      "macro_01_score_distribution.png", "macro_02_label_distribution.png",
      "macro_03_score_by_sector.png", "macro_04_confidence_distribution.png",
      "macro_05_score_vs_confidence.png", "macro_06_avg_score_by_sector.png",
      "macro_07_fred_timeseries.png", "macro_08_evaluation_summary.png",
      "06_macro_overlay_AAPL.png", "06_macro_overlay_EXE.png",
    ],
  },
  {
    id: "llm",
    label: "Multi-LLM Comparison",
    files: [
      "fundamental_llm_01_scores_by_ticker.png", "fundamental_llm_02_judge_overall.png",
      "fundamental_llm_03_judge_subscores.png", "market_sentiment_llm_01_scores_by_ticker.png",
      "market_sentiment_llm_02_judge_overall.png", "market_sentiment_llm_03_judge_subscores.png",
      "macro_llm_01_scores_by_ticker.png", "macro_llm_02_judge_overall.png",
      "macro_llm_03_judge_subscores.png",
    ],
  },
];

const AGENT_STYLES = {
  "Fundamental Agent": { tag: "FD", color: "#2563eb" },
  "fundamental_analysis_agent": { tag: "FD", color: "#2563eb" },
  "Market/Volatility/Sentiment Agent": { tag: "MV", color: "#7c5cff" },
  "market_sentiment_agent": { tag: "MV", color: "#7c5cff" },
  "Sentiment Agent": { tag: "SA", color: "#db2777" },
  "Macro-Economic Agent": { tag: "MC", color: "#0ea5a0" },
  "Critic / Orchestrator Agent": { tag: "CR", color: "#111827" },
  "LLM-Based Critic Agent": { tag: "CR", color: "#111827" },
};

function agentStyle(name) {
  return AGENT_STYLES[name] || { tag: String(name || "A").slice(0, 2).toUpperCase(), color: "#6b7686" };
}

function riskColor(label) {
  if (label === "HIGH") return "#e5484d";
  if (label === "LOW") return "#18a862";
  return "#e2910f";
}

let priceChartInstance = null;
const trainingChartInstances = {};

const $ = (id) => document.getElementById(id);

function number(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function fmt(value, digits = 2) {
  const parsed = number(value);
  return parsed === null ? "-" : parsed.toFixed(digits);
}

function pct(value, digits = 1) {
  const parsed = number(value);
  return parsed === null ? "-" : `${parsed.toFixed(digits)}%`;
}

function riskClass(label) {
  return `risk-${String(label || "").toLowerCase()}`;
}

function normalizeText(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function editDistance(a, b) {
  const left = String(a || "");
  const right = String(b || "");
  if (!left) return right.length;
  if (!right) return left.length;
  const dp = Array.from({ length: left.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= right.length; j += 1) dp[0][j] = j;
  for (let i = 1; i <= left.length; i += 1) {
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost,
      );
    }
  }
  return dp[left.length][right.length];
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || `${path} returned ${response.status}`);
  return data;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || `${path} returned ${response.status}`);
  return data;
}

function renderCounts() {
  $("companyCount").textContent = state.rows.length;
  $("highCount").textContent = state.rows.filter((row) => row.risk_label === "HIGH").length;
}

const TICKER_PAGE_SIZE = 60;
let tickerObserver = null;

function applyFilters() {
  const query = $("searchInput").value.trim().toLowerCase();
  state.filteredRows = state.rows
    .filter((row) => state.activeFilter === "ALL" || row.risk_label === state.activeFilter)
    .filter((row) => {
      if (!query) return true;
      return [row.ticker, row.company, row.sector, row.risk_label].some((value) => String(value || "").toLowerCase().includes(query));
    })
    .sort((a, b) => number(b.composite_risk, 0) - number(a.composite_risk, 0));
  state.renderedCount = TICKER_PAGE_SIZE;
  renderTickerList();
}

function buildTickerItem(row) {
  const template = $("tickerItemTemplate");
  const item = template.content.firstElementChild.cloneNode(true);
  item.dataset.ticker = row.ticker;
  item.dataset.risk = row.risk_label || "";
  item.classList.toggle("active", row.ticker === state.selectedTicker);
  item.querySelector(".ticker-symbol").textContent = row.ticker;
  item.querySelector(".ticker-sector").textContent = row.company && row.company !== row.ticker
    ? `${row.company} · ${row.sector || "Unknown sector"}`
    : row.sector || "Unknown sector";
  item.title = row.company && row.company !== row.ticker ? `${row.company} (${row.ticker})` : row.ticker;
  item.querySelector(".ticker-score").textContent = fmt(row.composite_risk, 1);
  item.addEventListener("click", () => selectTicker(row.ticker));
  return item;
}

function renderTickerList() {
  const list = $("tickerList");
  list.textContent = "";

  if (tickerObserver) {
    tickerObserver.disconnect();
    tickerObserver = null;
  }

  if (!state.filteredRows.length) {
    list.innerHTML = `<div class="empty">No tickers match the current filter.</div>`;
    return;
  }

  const visible = state.filteredRows.slice(0, state.renderedCount || TICKER_PAGE_SIZE);
  const fragment = document.createDocumentFragment();
  visible.forEach((row) => fragment.appendChild(buildTickerItem(row)));
  list.appendChild(fragment);

  if (visible.length < state.filteredRows.length) {
    const sentinel = document.createElement("div");
    sentinel.className = "ticker-sentinel";
    sentinel.textContent = `Loading more (${visible.length}/${state.filteredRows.length})...`;
    list.appendChild(sentinel);

    tickerObserver = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        state.renderedCount = (state.renderedCount || TICKER_PAGE_SIZE) + TICKER_PAGE_SIZE;
        renderTickerList();
      }
    }, { root: list, rootMargin: "120px" });
    tickerObserver.observe(sentinel);
  }
}

function findCompanyFromQuestion(question) {
  const normalized = normalizeText(question);
  if (!normalized) return null;
  const words = normalized.split(" ").filter(Boolean);
  const directTicker = state.rows.find((row) => words.includes(String(row.ticker).toLowerCase()));
  if (directTicker) return directTicker;

  const aliases = {
    adm: "ADM",
    apple: "AAPL",
    microsoft: "MSFT",
    nvidia: "NVDA",
    tesla: "TSLA",
    google: "GOOGL",
    alphabet: "GOOGL",
    amazon: "AMZN",
    jpmorgan: "JPM",
    "jp morgan": "JPM",
  };

  for (const [name, ticker] of Object.entries(aliases)) {
    if (normalized.includes(name)) {
      return state.rows.find((row) => row.ticker === ticker) || null;
    }
  }

  const companyMatch = state.rows.find((row) => {
    const company = normalizeText(row.company || "");
    return company && (normalized.includes(company) || company.includes(normalized));
  });
  if (companyMatch) return companyMatch;

  const tickerWords = words.filter((word) => word.length >= 2 && /[a-z]/.test(word));
  let best = null;
  tickerWords.forEach((word) => {
    state.rows.forEach((row) => {
      const ticker = String(row.ticker || "").toLowerCase();
      const distance = editDistance(word, ticker);
      if (distance <= 2 && (!best || distance < best.distance)) {
        best = { row, distance };
      }
    });
  });
  return best?.row || null;
}

function recommendationFrom(finalOutput, fallback) {
  const score = number(finalOutput.final_risk_score, 5);
  const label = finalOutput.final_risk_label || "MODERATE";
  const confidence = number(finalOutput.confidence, null);
  const disagreement = Boolean(finalOutput.disagreement_detected);
  const humanReview = Boolean(finalOutput.requires_human_review);

  if (confidence !== null && confidence < 0.55) {
    return {
      stance: "Insufficient confidence",
      className: "risk-moderate",
      action: "Do not make an investment decision from this run alone.",
    };
  }

  if (label === "HIGH" || score > 6 || humanReview) {
    return {
      stance: "Require review",
      className: "risk-high",
      action: fallback ? "Generate a full agent verdict before any investment consideration." : "Require senior review before investment consideration.",
    };
  }

  if (label === "LOW" && score <= 3.5 && !disagreement) {
    return {
      stance: "Research candidate",
      className: "risk-low",
      action: "Suitable for deeper investment research under this project model.",
    };
  }

  return {
    stance: "Watchlist",
    className: "risk-moderate",
    action: "Continue research and compare against alternatives before investment consideration.",
  };
}

function buildInvestmentAnswer(row, finalOutput, question, fallback) {
  const rec = recommendationFrom(finalOutput, fallback);
  const drivers = finalOutput.main_risk_drivers || [];
  const offsets = finalOutput.risk_offsets || [];
  return {
    ...rec,
    body: `For "${question || `Can I invest in ${row.ticker}?`}", ${row.ticker} is classified as ${finalOutput.final_risk_label} risk with score ${fmt(finalOutput.final_risk_score, 2)}/10. ${rec.action}`,
    bullets: [
      finalOutput.confidence === null || finalOutput.confidence === undefined ? "Confidence unavailable in fallback mode." : `Confidence: ${fmt(finalOutput.confidence, 2)}.`,
      fallback ? "Fallback mode: using Gold risk table until a generated verdict exists." : "Full mode: using generated agent verdict JSON.",
      finalOutput.disagreement_detected ? "Agents disagree; inspect the contradiction panel." : "No material disagreement flag in the final verdict.",
      ...(drivers.slice(0, 2).map((item) => `Risk driver: ${item}`)),
      ...(offsets.slice(0, 1).map((item) => `Offset: ${item}`)),
    ],
    drivers,
    offsets,
  };
}

function renderAdvisorAnswer(answer) {
  const box = $("advisorAnswer");
  box.className = `advisor-answer ${answer.className || ""}`;
  box.innerHTML = `
    <div class="answer-title">
      <strong>${answer.stance}</strong>
      <span>Research guidance, not personal financial advice</span>
    </div>
    <p>${answer.body}</p>
    <ul class="answer-bullets"></ul>
  `;
  const list = box.querySelector(".answer-bullets");
  answer.bullets.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderRunError(error) {
  renderAdvisorAnswer({
    stance: "Run failed",
    className: "risk-high",
    body: "The backend could not generate the verdict. Check the message below, then retry after fixing the server issue.",
    bullets: [error?.message || String(error || "Unknown error")],
    drivers: [],
    offsets: [],
  });
}

function setGenerateBusy(isBusy, mode = "critic") {
  state.generating = isBusy;
  const generateButton = $("generateVerdictButton");
  const debateButton = $("runDebateButton");
  const askButton = $("askButton");
  if (askButton) {
    askButton.disabled = isBusy;
    askButton.textContent = isBusy ? "Running..." : "Ask";
  }
  if (generateButton) {
    generateButton.disabled = isBusy;
    generateButton.textContent = isBusy && mode !== "debate" ? "Generating..." : "Generate full agent verdict";
  }
  if (debateButton) {
    debateButton.disabled = isBusy;
    debateButton.textContent = isBusy && mode === "debate" ? "Running debate..." : "Run debate review";
  }
}

function setTextList(id, items, emptyText) {
  const list = $(id);
  list.textContent = "";
  const visibleItems = (items || []).filter(Boolean).slice(0, 3);
  if (!visibleItems.length) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    list.appendChild(li);
    return;
  }
  visibleItems.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderDecisionHero(row, finalOutput, verdictPayload, answer) {
  const score = finalOutput.final_risk_score ?? row.composite_risk;
  const label = finalOutput.final_risk_label ?? row.risk_label ?? "-";
  const mode = verdictPayload.active_verdict_type === "debate" ? "debate verdict" : verdictPayload.active_verdict_type === "critic" ? "critic verdict" : "gold fallback";
  const confidence = finalOutput.confidence === null || finalOutput.confidence === undefined ? "-" : fmt(finalOutput.confidence, 2);
  const reviewState = finalOutput.requires_human_review ? "Human review" : finalOutput.disagreement_detected ? "Review disagreement" : "Auto review";

  $("recommendationBadge").textContent = answer.stance;
  $("recommendationBadge").className = `recommendation-badge ${answer.className || ""}`;
  $("heroSummary").textContent = `${row.ticker} is ${label} risk under the current ${mode}.`;
  $("heroRationale").textContent = answer.body;
  $("heroScore").textContent = fmt(score, 2);
  $("heroConfidence").textContent = confidence;
  $("heroReview").textContent = reviewState;
  setTextList("heroDrivers", finalOutput.main_risk_drivers, "No generated risk drivers yet.");
  setTextList("heroOffsets", finalOutput.risk_offsets, "No material offsets captured.");
  $("artifactStatus").textContent = verdictPayload.fallback
    ? "Gold fallback active - generate verdict for full audit package"
    : `${mode} loaded with ${finalOutput.policy_evidence_trail?.length || 0} policy checks`;
  drawRiskGauge(score, label);
}

function drawPriceChart(rows, ticker) {
  const wrap = $("priceChart").closest(".chart-wrap");
  const points = rows
    .map((row) => ({ x: row.date, y: number(row.close) }))
    .filter((point) => point.y !== null);

  if (priceChartInstance) {
    priceChartInstance.destroy();
    priceChartInstance = null;
  }

  let canvas = $("priceChart");
  if (points.length < 2) {
    wrap.innerHTML = `<div class="chart-empty">No price data found for ${ticker}</div>`;
    return;
  }
  if (!canvas) {
    wrap.innerHTML = `<canvas id="priceChart" role="img" aria-label="Price chart"></canvas>`;
    canvas = $("priceChart");
  }

  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, "rgba(59, 111, 224, 0.28)");
  gradient.addColorStop(1, "rgba(59, 111, 224, 0.0)");

  priceChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: points.map((point) => point.x),
      datasets: [{
        data: points.map((point) => point.y),
        borderColor: "#3b6fe0",
        backgroundColor: gradient,
        borderWidth: 2.5,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: "#3b6fe0",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#141b2b",
          padding: 10,
          titleFont: { size: 11 },
          bodyFont: { size: 12, weight: "700" },
          callbacks: { label: (item) => `$${Number(item.parsed.y).toFixed(2)}` },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: 6, color: "#97a1b0", font: { size: 10 } },
        },
        y: {
          grid: { color: "#eef1f6" },
          ticks: { callback: (value) => `$${value}`, color: "#97a1b0", font: { size: 10 } },
        },
      },
    },
  });
}

function drawRiskGauge(score, label) {
  const host = $("riskGauge");
  if (!host) return;
  const clamped = Math.max(0, Math.min(10, number(score, 0)));
  const radius = 64;
  const circumference = Math.PI * radius; // half circle
  const fraction = clamped / 10;
  const color = riskColor(label);

  host.innerHTML = `
    <svg width="160" height="104" viewBox="0 0 160 104">
      <path class="gauge-track" d="M 16 96 A ${radius} ${radius} 0 0 1 144 96" />
      <path class="gauge-value" stroke="${color}"
            stroke-dasharray="${(fraction * circumference).toFixed(1)} ${circumference.toFixed(1)}"
            d="M 16 96 A ${radius} ${radius} 0 0 1 144 96" />
      <text x="80" y="78" class="gauge-center gauge-score" fill="${color}">${fmt(clamped, 1)}</text>
      <text x="80" y="94" class="gauge-center gauge-max">/ 10</text>
    </svg>
    <span class="gauge-caption">Composite risk</span>
  `;
}

function criticCardFromFinal(finalOutput, activeType) {
  if (!finalOutput || finalOutput.final_risk_score === undefined) return null;
  const drivers = finalOutput.main_risk_drivers || [];
  const offsets = finalOutput.risk_offsets || [];
  const evidence = [
    finalOutput.final_decision,
    finalOutput.disagreement_detected ? "Critic detected a material disagreement across agents." : "Critic did not detect a material disagreement across agents.",
    finalOutput.requires_human_review ? "Human review is required before using this verdict." : "No human-review flag was raised by the critic.",
    ...drivers.map((item) => `Risk driver: ${item}`),
    ...offsets.map((item) => `Offset: ${item}`),
  ].filter(Boolean);

  return {
    agent: "Critic / Orchestrator Agent",
    claim_type: activeType === "debate" ? "DEBATE_SYNTHESIS" : finalOutput.critic_type || "CRITIC_SYNTHESIS",
    risk_score: finalOutput.final_risk_score,
    risk_label: finalOutput.final_risk_label,
    confidence: finalOutput.confidence,
    evidence,
    negative_signals: [],
  };
}

function renderAgentCards(outputs = [], finalOutput = null, activeType = "") {
  const grid = $("agentGrid");
  grid.textContent = "";
  const cards = [...(outputs || [])];
  const criticCard = criticCardFromFinal(finalOutput, activeType);
  if (criticCard) cards.push(criticCard);

  if (!cards.length) {
    grid.innerHTML = `<div class="empty">Generate a full agent verdict to view specialist evidence for this ticker.</div>`;
    return;
  }

  cards.forEach((agent) => {
    const score = number(agent.risk_score ?? agent.macro_risk_score, 0);
    const style = agentStyle(agent.agent);
    const article = document.createElement("article");
    article.className = "agent-card";
    article.innerHTML = `
      <header>
        <div>
          <h4><span class="agent-icon" style="background:${style.color}">${style.tag}</span>${agent.agent || "Agent"}</h4>
          <p>${agent.claim_type || "Claim"}</p>
        </div>
        <span class="label-pill ${riskClass(agent.risk_label)}">${agent.risk_label || "-"}</span>
      </header>
      <div class="agent-score">${score === null ? "-" : fmt(score, 2)}</div>
      <div class="agent-meter"><span style="width:${Math.max(0, Math.min(100, (score || 0) * 10))}%;background:${riskColor(agent.risk_label)}"></span></div>
      <p>Confidence: ${agent.confidence ?? "-"}</p>
      <ul class="evidence-list"></ul>
    `;
    const list = article.querySelector(".evidence-list");
    const evidenceItems = [
      ...(agent.evidence || []),
      ...(agent.negative_signals || []).map((item) => `Signal: ${item}`),
    ];
    evidenceItems.slice(0, 5).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
    grid.appendChild(article);
  });
}

function modelStatusClass(status = "") {
  if (String(status).includes("active") || String(status).includes("configured")) return "risk-low";
  if (String(status).includes("fallback") || String(status).includes("rule")) return "risk-moderate";
  return "";
}

function renderModelRegistry(payload) {
  const registry = $("modelRegistry");
  const charts = $("trainingCharts");
  if (!registry || !charts) return;

  Object.values(trainingChartInstances).forEach((chart) => chart.destroy());
  Object.keys(trainingChartInstances).forEach((key) => delete trainingChartInstances[key]);

  registry.textContent = "";
  charts.textContent = "";

  const models = payload?.models || [];
  if (!models.length) {
    registry.innerHTML = `<div class="empty">No model registry data available.</div>`;
    return;
  }

  models.forEach((model) => {
    const card = document.createElement("article");
    card.className = "model-card";
    card.innerHTML = `
      <div>
        <strong>${model.agent}</strong>
        <p>${model.model}</p>
      </div>
      <span class="label-pill ${modelStatusClass(model.status)}">${model.status}</span>
      <dl>
        <div><dt>Backend</dt><dd>${model.backend || "-"}</dd></div>
        <div><dt>Claim</dt><dd>${model.claim_type || "-"}</dd></div>
        <div><dt>Training rows</dt><dd>${model.train_examples || 0}</dd></div>
      </dl>
      <p class="model-data">${model.data || ""}</p>
    `;
    registry.appendChild(card);
  });

  models
    .filter((model) => model.training?.available && (model.training.train_loss || []).length)
    .forEach((model, index) => {
      const chartCard = document.createElement("article");
      chartCard.className = "training-card";
      const canvasId = `trainingChart${index}`;
      const latest = model.training.train_loss.at(-1);
      const metrics = model.metrics || {};
      chartCard.innerHTML = `
        <div class="training-head">
          <div>
            <strong>${model.agent}</strong>
            <p>${model.model}</p>
          </div>
          <span class="status-pill">loss ${latest ? fmt(latest.loss, 3) : "-"}</span>
        </div>
        <div class="training-metrics">
          <span>Steps ${model.training.summary?.global_step || "-"}</span>
          <span>Epoch ${fmt(model.training.summary?.epoch, 2)}</span>
          <span>Accuracy ${metrics.eval_accuracy === undefined ? "-" : pct(Number(metrics.eval_accuracy) * 100, 1)}</span>
          <span>Macro F1 ${metrics.eval_macro_f1 === undefined ? "-" : fmt(metrics.eval_macro_f1, 3)}</span>
        </div>
        <div class="training-chart-wrap"><canvas id="${canvasId}" aria-label="${model.agent} training loss"></canvas></div>
      `;
      charts.appendChild(chartCard);

      const train = model.training.train_loss || [];
      const evalLoss = model.training.eval_loss || [];
      const evalByStep = new Map(evalLoss.map((point) => [point.step, point.loss]));
      const ctx = $(canvasId).getContext("2d");
      trainingChartInstances[canvasId] = new Chart(ctx, {
        type: "line",
        data: {
          labels: train.map((point) => point.step),
          datasets: [
            {
              label: "train loss",
              data: train.map((point) => point.loss),
              borderColor: index === 0 ? "#2563eb" : "#7c5cff",
              backgroundColor: "transparent",
              borderWidth: 2.5,
              tension: 0.25,
              pointRadius: 2,
            },
            {
              label: "eval loss",
              data: train.map((point) => evalByStep.get(point.step) ?? null),
              borderColor: "#e2910f",
              backgroundColor: "transparent",
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0.25,
              pointRadius: 3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { boxWidth: 10, color: "#6b7686", font: { size: 11 } } },
            tooltip: { backgroundColor: "#141b2b", padding: 10 },
          },
          scales: {
            x: { title: { display: true, text: "step" }, grid: { display: false }, ticks: { color: "#97a1b0", font: { size: 10 } } },
            y: { title: { display: true, text: "loss" }, grid: { color: "#eef1f6" }, ticks: { color: "#97a1b0", font: { size: 10 } } },
          },
        },
      });
    });

  if (!charts.children.length) {
    charts.innerHTML = `<div class="empty">No local training loss history found for the currently configured models.</div>`;
  }
}

function renderPolicyTrail(finalOutput = {}) {
  const list = $("policyTrail");
  list.textContent = "";
  const policies = finalOutput.policy_evidence_trail || [];
  if (!policies.length) {
    list.innerHTML = `<div class="empty">No Policy A-H evidence available.</div>`;
    return;
  }

  policies.forEach((policy) => {
    const row = document.createElement("div");
    row.className = "policy-row";
    row.innerHTML = `
      <div class="policy-id">Policy ${policy.policy_id}</div>
      <div>
        <strong>${policy.policy_name}</strong>
        <p>${(policy.evidence || [])[0] || "No evidence captured."}</p>
      </div>
      <span class="label-pill">${policy.status}</span>
    `;
    list.appendChild(row);
  });
}

function renderContradictions(verdictPayload) {
  const list = $("contradictions");
  const debate = verdictPayload.debate || {};
  const final = verdictPayload.final_output || {};
  const contradictions = debate.contradictions || final.contradictions || [];
  const revisionRounds = debate.final_output?.revision_rounds ?? final.revision_rounds;
  list.textContent = "";

  if (!contradictions.length) {
    list.innerHTML = `<div class="empty">No contradiction records found for this ticker.</div>`;
  } else {
    contradictions.forEach((item) => {
      const row = document.createElement("div");
      row.className = "contradiction-row";
      row.innerHTML = `<strong>${item.type || "Contradiction"}</strong><p>${item.detail || ""}</p><p>Severity: ${item.severity || "-"}</p>`;
      list.appendChild(row);
    });
  }

  if (revisionRounds !== undefined) {
    const row = document.createElement("div");
    row.className = "monitor-row";
    row.innerHTML = `<strong>Revision rounds</strong><p>${revisionRounds}</p>`;
    list.appendChild(row);
  }
}

function renderMonitoring(records = []) {
  const list = $("monitoring");
  list.textContent = "";
  const latest = records.slice(-4).reverse();
  if (!latest.length) {
    list.innerHTML = `<div class="empty">No monitoring records found for this ticker.</div>`;
    return;
  }

  latest.forEach((record) => {
    const row = document.createElement("div");
    row.className = "monitor-row";
    row.innerHTML = `
      <strong>${record.run_type || "run"} - ${record.final_risk_label || "-"}</strong>
      <p>${record.timestamp || ""}</p>
      <p>score ${fmt(record.final_risk_score, 2)} | confidence ${fmt(record.confidence, 2)} | contradictions ${record.contradiction_count ?? 0}</p>
    `;
    list.appendChild(row);
  });
}

function renderDrivers(finalOutput) {
  const drivers = $("riskDrivers");
  drivers.textContent = "";
  (finalOutput.main_risk_drivers || []).slice(0, 4).forEach((item) => {
    const driver = document.createElement("div");
    driver.className = "driver";
    driver.textContent = item;
    drivers.appendChild(driver);
  });
}

function populateCompareOptions() {
  const datalist = $("compareOptions");
  if (!datalist) return;
  datalist.textContent = "";
  state.rows
    .slice()
    .sort((a, b) => String(a.ticker).localeCompare(String(b.ticker)))
    .forEach((row) => {
      const option = document.createElement("option");
      option.value = row.company && row.company !== row.ticker ? `${row.company} (${row.ticker})` : row.ticker;
      datalist.appendChild(option);
    });
}

function resolveCompareTicker(query) {
  const trimmed = String(query || "").trim();
  if (!trimmed) return null;

  const upper = trimmed.toUpperCase();
  const exactTicker = state.rows.find((row) => row.ticker === upper);
  if (exactTicker) return exactTicker.ticker;

  // "Company Name (TICKER)" picked from the datalist
  const parenMatch = trimmed.match(/\(([A-Za-z.\-]+)\)\s*$/);
  if (parenMatch) {
    const row = state.rows.find((r) => r.ticker === parenMatch[1].toUpperCase());
    if (row) return row.ticker;
  }

  const lower = trimmed.toLowerCase();
  const exactCompany = state.rows.find((row) => (row.company || "").toLowerCase() === lower);
  if (exactCompany) return exactCompany.ticker;

  const partial = state.rows.find((row) => (row.company || "").toLowerCase().includes(lower) || row.ticker.toLowerCase().includes(lower));
  return partial ? partial.ticker : null;
}

function compareInputLabel(row) {
  return row.company && row.company !== row.ticker ? `${row.company} (${row.ticker})` : row.ticker;
}

function renderCompare() {
  const container = $("compareGrid");
  if (!container) return;
  container.textContent = "";

  if (!state.comparePair || !state.comparePayload) {
    container.innerHTML = `<div class="empty">Type a second company and click "Compare".</div>`;
    return;
  }

  const summary = state.comparePayload.summary;
  if (summary) {
    const summaryCard = document.createElement("div");
    summaryCard.className = "compare-summary";
    summaryCard.innerHTML = `
      <strong>${summary.lower_risk_ticker} is lower risk</strong>
      <p>${summary.decision}</p>
    `;
    container.appendChild(summaryCard);
  }

  (state.comparePayload.companies || []).forEach((companyResult) => {
      const row = companyResult.company || {};
      const card = document.createElement("div");
      card.className = "compare-card";
      card.innerHTML = `
        <strong>${row.ticker}</strong>
        <span class="label-pill ${riskClass(companyResult.label)}">${companyResult.label || "-"}</span>
        <p>${row.company && row.company !== row.ticker ? row.company : ""}</p>
        <p>Verdict score ${fmt(companyResult.score, 2)} | Confidence ${companyResult.confidence === null || companyResult.confidence === undefined ? "-" : fmt(companyResult.confidence, 2)}</p>
        <p>Gold composite ${fmt(row.composite_risk, 2)} | Vol ${pct(row.ann_volatility_pct, 1)} | Drawdown ${pct(row.max_drawdown_pct, 1)}</p>
        <p>${companyResult.has_generated_verdict ? "Generated verdict available" : "Gold fallback mode"}</p>
      `;
      container.appendChild(card);
    });
}

async function generateVerdict(row, mode = "critic", question = "") {
  if (state.generating) return;
  setGenerateBusy(true, mode);
  renderAdvisorAnswer({
    stance: mode === "debate" ? `Running debate for ${row.ticker}` : `Generating full verdict for ${row.ticker}`,
    className: "risk-moderate",
    body: "The backend is running the project agents, saving artifacts, and logging monitoring metadata.",
    bullets: ["The dashboard refreshes when the run completes."],
    drivers: [],
    offsets: [],
  });
  try {
    await postJson(api.generateVerdict, {
      ticker: row.ticker,
      company: row.company || row.ticker,
      sector: row.sector || "General",
      query: question,
      mode,
    });
    await loadCompanies();
    await selectTicker(row.ticker, question);
  } catch (error) {
    renderRunError(error);
  } finally {
    setGenerateBusy(false, mode);
  }
}

async function selectTicker(ticker, question = "") {
  state.selectedTicker = ticker;
  document.querySelectorAll(".ticker-item").forEach((item) => item.classList.toggle("active", item.dataset.ticker === ticker));

  const [verdictPayload, pricePayload, monitoringPayload] = await Promise.all([
    fetchJson(api.verdict(ticker)),
    fetchJson(api.prices(ticker)),
    fetchJson(api.monitoring(ticker)),
  ]);

  state.current = verdictPayload;
  state.monitoring = monitoringPayload.records || [];
  const row = verdictPayload.company;
  const final = verdictPayload.final_output;
  const outputs = verdictPayload.active_verdict_type === "debate"
    ? verdictPayload.debate?.agent_outputs || verdictPayload.verdict?.agent_outputs || []
    : verdictPayload.verdict?.agent_outputs || verdictPayload.debate?.agent_outputs || [];
  const activeJsonUrl = verdictPayload.active_verdict_type === "debate" && verdictPayload.debate_url ? verdictPayload.debate_url : verdictPayload.json_url;
  const rec = buildInvestmentAnswer(row, final, question, verdictPayload.fallback);

  $("selectedTitle").textContent = row.company && row.company !== row.ticker
    ? `${row.company} (${row.ticker}) - ${row.sector || "Unknown sector"}`
    : `${row.ticker} - ${row.sector || "Unknown sector"}`;
  $("compositeRisk").textContent = fmt(final.final_risk_score ?? row.composite_risk, 2);
  $("riskLabel").textContent = final.final_risk_label ?? row.risk_label ?? "-";
  $("riskLabel").className = riskClass(final.final_risk_label ?? row.risk_label);
  $("volatility").textContent = pct(row.ann_volatility_pct, 1);
  $("drawdown").textContent = pct(row.max_drawdown_pct, 1);
  $("confidence").textContent = final.confidence === null || final.confidence === undefined ? "-" : fmt(final.confidence, 2);
  $("reviewFlag").textContent = verdictPayload.fallback ? "Gold fallback" : final.requires_human_review ? "Human review required" : "Auto verdict";
  $("asOfDate").textContent = row.as_of_date || "-";
  $("criticType").textContent = verdictPayload.active_verdict_type === "debate" ? "Debate review output" : final.critic_type || "critic";
  $("disagreementBadge").textContent = verdictPayload.fallback ? "Fallback" : final.disagreement_detected ? "Disagreement" : "Aligned";
  $("disagreementBadge").className = `status-pill ${verdictPayload.fallback || final.disagreement_detected ? "risk-moderate" : "risk-low"}`;
  $("finalDecision").textContent = final.final_decision || rec.body;
  $("jsonLink").hidden = !activeJsonUrl;
  $("pdfLink").hidden = !verdictPayload.pdf_url;
  $("jsonLink").href = activeJsonUrl || "#";
  $("pdfLink").href = verdictPayload.pdf_url || "#";
  $("priceSubtitle").textContent = `${pricePayload.count || 0} price rows loaded from API`;
  $("generateVerdictButton").hidden = !verdictPayload.fallback;
  $("runDebateButton").hidden = !verdictPayload.verdict;

  renderDecisionHero(row, final, verdictPayload, rec);
  renderAdvisorAnswer(rec);
  renderDrivers(final);
  drawPriceChart(pricePayload.prices || [], ticker);
  renderAgentCards(outputs, final, verdictPayload.active_verdict_type);
  renderPolicyTrail(final);
  renderContradictions(verdictPayload);
  renderMonitoring(state.monitoring);

  const compareInputA = $("compareInputA");
  if (compareInputA && document.activeElement !== compareInputA) {
    compareInputA.value = compareInputLabel(row);
  }
  renderCompare();
}

function reportTitle(filename) {
  return filename
    .replace(/\.png$/, "")
    .replace(/^\d+_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderReportTabs() {
  const tabs = $("reportTabs");
  tabs.textContent = "";
  REPORT_CATEGORIES.forEach((category) => {
    const button = document.createElement("button");
    button.className = `report-tab ${category.id === state.reportCategory ? "active" : ""}`;
    button.type = "button";
    button.textContent = category.label;
    button.addEventListener("click", () => {
      state.reportCategory = category.id;
      renderReportTabs();
      renderReportGrid();
    });
    tabs.appendChild(button);
  });
}

function renderReportGrid() {
  const grid = $("reportGrid");
  grid.textContent = "";
  const category = REPORT_CATEGORIES.find((item) => item.id === state.reportCategory) || REPORT_CATEGORIES[0];
  category.files.forEach((filename) => {
    const figure = document.createElement("figure");
    figure.className = "report-card";
    const src = `/data/reports/${filename}`;
    figure.innerHTML = `<img src="${src}" alt="${reportTitle(filename)}" loading="lazy"><figcaption>${reportTitle(filename)}</figcaption>`;
    figure.addEventListener("click", () => openLightbox(src, reportTitle(filename)));
    grid.appendChild(figure);
  });
}

function openLightbox(src, alt) {
  $("lightboxImage").src = src;
  $("lightboxImage").alt = alt;
  $("lightbox").hidden = false;
}

function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightboxImage").src = "";
}

async function loadCompanies() {
  const payload = await fetchJson(api.companies);
  state.rows = payload.companies || [];
  renderCounts();
  applyFilters();
  populateCompareOptions();
}

async function boot() {
  try {
    renderReportTabs();
    renderReportGrid();
    const [modelPayload] = await Promise.all([
      fetchJson(api.modelTraining).catch((error) => ({ models: [], notes: [error.message] })),
      loadCompanies(),
    ]);
    state.modelTraining = modelPayload;
    renderModelRegistry(modelPayload);
    const first = state.filteredRows[0] || state.rows[0];
    await selectTicker(first.ticker);
  } catch (error) {
    document.body.innerHTML = `<main class="workspace"><div class="panel"><h2>Unable to load app data</h2><p>${error.message}</p></div></main>`;
  }
}

$("lightboxClose").addEventListener("click", closeLightbox);
$("lightbox").addEventListener("click", (event) => {
  if (event.target.id === "lightbox") closeLightbox();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});

document.addEventListener("click", (event) => {
  const button = event.target.closest(".segmented button");
  if (!button) return;
  document.querySelectorAll(".segmented button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  state.activeFilter = button.dataset.filter;
  applyFilters();
});

$("searchInput").addEventListener("input", applyFilters);

$("askForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("askInput").value.trim();
  const row = findCompanyFromQuestion(question) || (!question ? state.rows.find((item) => item.ticker === state.selectedTicker) : null);
  if (!row) {
    renderAdvisorAnswer({
      stance: "Company not found",
      className: "risk-moderate",
      body: `I could not match "${question}" to a ticker in the current 519-company universe. Try a ticker such as GOOGL, GOOG, AAPL, or MSFT.`,
      bullets: ["No verdict was generated because using the currently selected ticker would answer the wrong company."],
      drivers: [],
      offsets: [],
    });
    return;
  }
  try {
    if (state.selectedTicker !== row.ticker) {
      await selectTicker(row.ticker, question);
    }
    if (state.current?.fallback) {
      await generateVerdict(row, "critic", question);
    } else {
      renderAdvisorAnswer(buildInvestmentAnswer(state.current.company, state.current.final_output, question, false));
    }
  } catch (error) {
    renderRunError(error);
  }
});

$("generateVerdictButton").addEventListener("click", async () => {
  const row = state.current?.company;
  if (row) await generateVerdict(row, "critic", $("askInput").value.trim());
});

$("runDebateButton").addEventListener("click", async () => {
  const row = state.current?.company;
  if (row) await generateVerdict(row, "debate", $("askInput").value.trim());
});

$("compareForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const inputA = $("compareInputA");
  const inputB = $("compareInputB");
  const tickerA = resolveCompareTicker(inputA.value);
  const tickerB = resolveCompareTicker(inputB.value);

  let hasError = false;
  [[inputA, tickerA], [inputB, tickerB]].forEach(([input, ticker]) => {
    if (!ticker) {
      input.classList.add("input-error");
      setTimeout(() => input.classList.remove("input-error"), 1200);
      hasError = true;
    }
  });
  if (hasError) return;

  if (tickerA === tickerB) {
    inputB.classList.add("input-error");
    setTimeout(() => inputB.classList.remove("input-error"), 1200);
    return;
  }

  state.comparePair = [tickerA, tickerB];
  $("compareGrid").innerHTML = `<div class="empty">Comparing ${tickerA} and ${tickerB} through the API...</div>`;
  fetchJson(api.compare(state.comparePair))
    .then((payload) => {
      state.comparePayload = payload;
      renderCompare();
    })
    .catch((error) => {
      state.comparePayload = null;
      $("compareGrid").innerHTML = `<div class="empty">Compare failed: ${error.message}</div>`;
    });
});

boot();
