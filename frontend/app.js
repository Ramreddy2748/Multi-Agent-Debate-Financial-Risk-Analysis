const api = {
  companies: "/api/companies",
  monitoring: (ticker = "") => `/api/monitoring${ticker ? `?ticker=${encodeURIComponent(ticker)}` : ""}`,
  verdict: (ticker) => `/api/verdicts/${ticker}`,
  prices: (ticker) => `/api/companies/${ticker}/prices`,
  generateVerdict: "/api/verdict",
  compare: (tickers) => `/api/compare?tickers=${encodeURIComponent(tickers.join(","))}`,
  pdf: (ticker) => `/api/artifacts/${ticker}/pdf`,
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
  generating: false,
};

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

function applyFilters() {
  const query = $("searchInput").value.trim().toLowerCase();
  state.filteredRows = state.rows
    .filter((row) => state.activeFilter === "ALL" || row.risk_label === state.activeFilter)
    .filter((row) => {
      if (!query) return true;
      return [row.ticker, row.company, row.sector, row.risk_label].some((value) => String(value || "").toLowerCase().includes(query));
    })
    .sort((a, b) => number(b.composite_risk, 0) - number(a.composite_risk, 0));
  renderTickerList();
}

function renderTickerList() {
  const list = $("tickerList");
  const template = $("tickerItemTemplate");
  list.textContent = "";

  state.filteredRows.forEach((row) => {
    const item = template.content.firstElementChild.cloneNode(true);
    item.dataset.ticker = row.ticker;
    item.classList.toggle("active", row.ticker === state.selectedTicker);
    item.querySelector(".ticker-symbol").textContent = row.ticker;
    item.querySelector(".ticker-sector").textContent = row.company && row.company !== row.ticker
      ? `${row.company} · ${row.sector || "Unknown sector"}`
      : row.sector || "Unknown sector";
    item.title = row.company && row.company !== row.ticker ? `${row.company} (${row.ticker})` : row.ticker;
    item.querySelector(".ticker-score").textContent = fmt(row.composite_risk, 1);
    item.addEventListener("click", () => selectTicker(row.ticker));
    list.appendChild(item);
  });

  if (!state.filteredRows.length) {
    list.innerHTML = `<div class="empty">No tickers match the current filter.</div>`;
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

  return null;
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
}

function drawPriceChart(rows, ticker) {
  const svg = $("priceChart");
  svg.textContent = "";
  const width = 720;
  const height = 220;
  const pad = { top: 20, right: 28, bottom: 32, left: 48 };
  const values = rows.map((row) => number(row.close)).filter((value) => value !== null);

  if (values.length < 2) {
    svg.innerHTML = `<text x="30" y="110" class="chart-label">No price data found for ${ticker}</text>`;
    return;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xFor = (index) => pad.left + (index / (values.length - 1)) * (width - pad.left - pad.right);
  const yFor = (value) => pad.top + (1 - (value - min) / span) * (height - pad.top - pad.bottom);
  const points = values.map((value, index) => `${xFor(index).toFixed(2)},${yFor(value).toFixed(2)}`);
  const area = `${pad.left},${height - pad.bottom} ${points.join(" ")} ${width - pad.right},${height - pad.bottom}`;

  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("class", "axis-line");
  axis.setAttribute("x1", pad.left);
  axis.setAttribute("x2", width - pad.right);
  axis.setAttribute("y1", height - pad.bottom);
  axis.setAttribute("y2", height - pad.bottom);

  const areaShape = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  areaShape.setAttribute("class", "area");
  areaShape.setAttribute("points", area);

  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("class", "sparkline");
  line.setAttribute("points", points.join(" "));

  const minLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  minLabel.setAttribute("class", "chart-label");
  minLabel.setAttribute("x", 8);
  minLabel.setAttribute("y", height - pad.bottom);
  minLabel.textContent = `$${min.toFixed(0)}`;

  const maxLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  maxLabel.setAttribute("class", "chart-label");
  maxLabel.setAttribute("x", 8);
  maxLabel.setAttribute("y", pad.top + 8);
  maxLabel.textContent = `$${max.toFixed(0)}`;

  svg.append(areaShape, axis, line, minLabel, maxLabel);
}

function renderAgentCards(outputs = []) {
  const grid = $("agentGrid");
  grid.textContent = "";
  if (!outputs.length) {
    grid.innerHTML = `<div class="empty">Generate a full agent verdict to view specialist evidence for this ticker.</div>`;
    return;
  }

  outputs.forEach((agent) => {
    const score = agent.risk_score ?? agent.macro_risk_score ?? "-";
    const article = document.createElement("article");
    article.className = "agent-card";
    article.innerHTML = `
      <header>
        <div>
          <h4>${agent.agent || "Agent"}</h4>
          <p>${agent.claim_type || "Claim"}</p>
        </div>
        <span class="label-pill ${riskClass(agent.risk_label)}">${agent.risk_label || "-"}</span>
      </header>
      <div class="agent-score">${fmt(score, 2)}</div>
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
  renderAgentCards(outputs);
  renderPolicyTrail(final);
  renderContradictions(verdictPayload);
  renderMonitoring(state.monitoring);

  const compareInputA = $("compareInputA");
  if (compareInputA && document.activeElement !== compareInputA) {
    compareInputA.value = compareInputLabel(row);
  }
  renderCompare();
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
    await loadCompanies();
    const first = state.rows.find((row) => row.ticker === "AAPL") || state.rows[0];
    await selectTicker(first.ticker);
  } catch (error) {
    document.body.innerHTML = `<main class="workspace"><div class="panel"><h2>Unable to load app data</h2><p>${error.message}</p></div></main>`;
  }
}

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
  const row = findCompanyFromQuestion(question) || state.rows.find((item) => item.ticker === state.selectedTicker);
  if (!row) return;
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
