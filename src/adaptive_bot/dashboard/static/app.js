const PROFILE = "musca-v5-binance";
const $ = (id) => document.getElementById(id);

const money = (value) => {
  const parsed = Number(value);
  return value != null && value !== "" && Number.isFinite(parsed)
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(parsed)
    : "--";
};
const number = (value, digits = 2) => {
  const parsed = Number(value);
  return value != null && value !== "" && Number.isFinite(parsed) ? parsed.toFixed(digits) : "--";
};
const integer = (value) => number(value, 0);
const percent = (value) => value != null && value !== "" && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "--";
const bps = (value) => value != null && value !== "" && Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} bps` : "--";
const signedBps = (value) => value != null && value !== "" && Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)} bps` : "--";
const dateTime = (value) => value ? new Date(value).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "medium" }) : "--";
const timeOnly = (value) => value ? new Date(value).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "--";
const age = (value) => {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
};
const setText = (id, value) => { const element = $(id); if (element) element.textContent = value; };
const show = (id, visible) => $(id)?.classList.toggle("hidden", !visible);
const paintValue = (id, value) => {
  const element = $(id);
  if (!element) return;
  const parsed = Number(value);
  element.classList.toggle("positive", Number.isFinite(parsed) && parsed > 0);
  element.classList.toggle("negative", Number.isFinite(parsed) && parsed < 0);
};

const decisionReasons = {
  FLAT_NO_POSITIVE_AUTO_MOE_ACTION: "No active expert currently has positive calibrated net EV after Binance costs.",
  MODEL_INPUT_FAIL_CLOSED: "The model input contract is incomplete. Trading is blocked until every causal feature is available.",
  BINANCE_DATA_FAIL_CLOSED: "A required Binance feed is stale or invalid. No order can be created.",
  RISK_REJECTED: "A valid Alpha candidate was rejected by the independent risk engine.",
  TRADE: "A positive-EV expert passed market data, execution and risk checks.",
};

let chartState = { candles: [], target: null, stop: null, breakEven: null };
let refreshBusy = false;

function paperAccount(audit) {
  const diagnostics = audit.one_position_diagnostics || {};
  return diagnostics.paper_accounts?.BINANCE || diagnostics.paper_account || {};
}

function currentAssessment(audit) {
  return audit.current_market_assessments?.BINANCE || audit.current_market_assessment || {};
}

function badge(id, label, state) {
  const element = $(id);
  if (!element) return;
  element.textContent = label;
  element.className = `badge ${state}`;
}

async function refresh() {
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    const response = await fetch(`/api/dashboard?profile=${PROFILE}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    const data = await response.json();
    if (!data.available) throw new Error(data.error || "MUSCA BTC report is unavailable");
    render(data);
    $("error-state").classList.add("hidden");
    $("dashboard").classList.remove("hidden");
  } catch (error) {
    $("dashboard").classList.add("hidden");
    $("error-state").classList.remove("hidden");
    setText("error-message", error.message);
    badge("system-status", "DISCONNECTED", "bad");
  } finally {
    refreshBusy = false;
  }
}

function render(data) {
  const summary = data.summary || {};
  const audit = summary.forward_audit || {};
  const paper = paperAccount(audit);
  const assessment = currentAssessment(audit);
  const market = assessment.market_inputs?.binance || {};
  const sources = assessment.sources || {};
  const readiness = audit.live_readiness || {};
  const coverage = audit.data_coverage || {};
  const allFeedsFresh = [sources.alpha, sources.execution, sources.funding].every((source) => source?.valid);

  setText("last-updated", `Updated ${timeOnly(data.generated_at)}`);
  badge("system-status", allFeedsFresh ? "BINANCE CONNECTED" : "FAIL CLOSED", allFeedsFresh ? "good" : "bad");
  renderAccount(paper, audit, summary);
  renderMarket(market, assessment, audit.market_chart || [], allFeedsFresh);
  renderDecision(assessment);
  renderOrder(paper, assessment);
  renderFeeds(sources, readiness, coverage);
  renderRisk(paper, data.safety || {}, summary);
  renderLedger(paper);
  renderStrategy(audit);
  renderValidation(audit, paper);
}

function renderAccount(paper, audit, summary) {
  const open = paper.open_position;
  setText("equity", money(paper.final_equity ?? summary.final_equity));
  setText("available-balance", `${money(paper.available_balance ?? paper.final_equity)} available`);
  setText("net-pnl", money(paper.net_pnl ?? 0));
  paintValue("net-pnl", paper.net_pnl ?? 0);
  setText("position-status", open ? `${open.side} OPEN` : paper.pending_order ? "ORDER PENDING" : "FLAT");
  setText("position-summary", open ? `${number(open.quantity_btc, 6)} BTC at ${money(open.entry_execution_price)}` : "No market exposure");
  setText("trade-count", integer(paper.trades?.length || 0));
  setText("signal-count", `${paper.trade_signal_count || 0} approved / ${paper.complete_candidate_count || 0} complete candidates`);
  setText("costs-paid", money(paper.modeled_costs ?? 0));
  setText("fee-model", `Maker ${number(paper.maker_fees_per_side_bps, 2)} / taker ${number(paper.fees_per_side_bps, 2)} bps per side`);
  setText("drawdown", percent(paper.max_drawdown ?? 0));
  paintValue("drawdown", -(Number(paper.max_drawdown) || 0));
}

function renderMarket(market, assessment, candles, fresh) {
  badge("market-status", fresh && market.book_synced ? "BINANCE LIVE" : "WAITING", fresh && market.book_synced ? "good" : "bad");
  setText("mark-price", money(market.mark_price ?? market.price));
  setText("market-time", `Decision candle ${dateTime(assessment.observed_at)}`);
  setText("best-bid", money(market.best_bid));
  setText("best-ask", money(market.best_ask));
  setText("spread", bps(market.spread_bps));
  setText("index-price", money(market.index_price));
  setText("funding-rate", percent(market.funding_rate));
  setText("next-funding", dateTime(market.next_funding_timestamp));
  setText("return-1m", signedBps(market.return_1m_bps));
  setText("return-5m", signedBps(market.return_5m_bps));
  setText("return-15m", signedBps(market.return_15m_bps));
  setText("return-30m", signedBps(market.return_30m_bps));
  setText("rolling-vwap", money(market.rolling_vwap));
  setText("vwap-distance", signedBps(market.vwap_distance_bps));
  setText("taker-imbalance", number(market.taker_imbalance_60s, 3));
  setText("oi-change", percent(market.oi_change_1h));
  ["return-1m", "return-5m", "return-15m", "return-30m", "vwap-distance", "oi-change"].forEach((id) => {
    const key = {
      "return-1m": market.return_1m_bps,
      "return-5m": market.return_5m_bps,
      "return-15m": market.return_15m_bps,
      "return-30m": market.return_30m_bps,
      "vwap-distance": market.vwap_distance_bps,
      "oi-change": market.oi_change_1h,
    }[id];
    paintValue(id, key);
  });

  chartState = {
    candles,
    target: assessment.target_price,
    stop: assessment.stop_price,
    breakEven: assessment.break_even_price,
  };
  show("target-legend", assessment.target_price != null);
  show("stop-legend", assessment.stop_price != null);
  show("breakeven-legend", assessment.break_even_price != null);
  drawMarketChart();
}

function renderDecision(assessment) {
  const decision = String(assessment.decision || "WAIT").toUpperCase();
  const model = assessment.model_evaluation || {};
  const best = model.best_action || {};
  const statusClass = decision === "TRADE" ? "good" : decision === "FLAT" ? "flat" : "bad";
  badge("decision-badge", decision, statusClass);
  setText("decision-title", decision === "TRADE" ? `${assessment.direction} candidate approved` : decision === "FLAT" ? "No trade right now" : "Trading is blocked");
  setText("decision-reason", decisionReasons[assessment.reason] || String(assessment.reason || "No decision reason reported").replaceAll("_", " "));
  setText("model-state", `${model.status || assessment.model_context || "--"}${model.reason ? ` / ${String(model.reason).replaceAll("_", " ")}` : ""}`);
  setText("expert-count", `${model.active_expert_count ?? 0} active / ${model.frozen_expert_count ?? 0} frozen`);
  setText("best-action", best.direction ? `${best.direction} / ${best.expert_id}` : "No active action");
  setText("expected-ev", signedBps(best.calibrated_ev_bps ?? assessment.expected_net_ev_bps));
  setText("target-probability", percent(best.probability_net_positive ?? assessment.target_probability));
  setText("forecast-horizon", best.horizon_minutes ? `${best.horizon_minutes} minutes` : `${(assessment.outcome_horizons_minutes || [60, 360]).join(" / ")} minutes available`);
  setText("risk-check", assessment.risk_approved === true ? "APPROVED" : assessment.risk_approved === false ? `REJECTED / ${assessment.risk_reason}` : "Not evaluated without a candidate");
  setText("execution-check", assessment.execution_status || "Not evaluated without a candidate");
  paintValue("expected-ev", best.calibrated_ev_bps ?? assessment.expected_net_ev_bps);

  if (decision === "TRADE") {
    setText("next-event", "The paper order will use the first valid Binance book snapshot after this signal.");
  } else if (decision === "FLAT") {
    setText("next-event", "Re-evaluate after the next closed one-minute candle. Entry requires at least one active expert with positive calibrated net EV.");
  } else {
    setText("next-event", "Wait for every required Binance feed and model input to become fresh and synchronized.");
  }
}

function renderOrder(paper, assessment) {
  const open = paper.open_position;
  const pending = paper.pending_order;
  const plan = open || pending || (assessment.candidate_complete ? assessment : null);
  const state = open ? "OPEN" : pending ? "PENDING" : "FLAT";
  badge("order-status", state, open ? "good" : pending ? "paper" : "flat");
  setText("order-summary", open
    ? `${open.side} position filled at ${dateTime(open.entry_at)}. Exit controls are active.`
    : pending
      ? `${pending.side} order is waiting for the first valid Binance book after the signal.`
      : "No position or pending order. Scenario prices are not shown as orders until a candidate passes every gate.");
  setText("order-side", plan?.side || plan?.direction || "--");
  setText("order-quantity", plan?.quantity_btc == null ? "--" : `${number(plan.quantity_btc, 6)} BTC`);
  setText("entry-price", money(open?.entry_execution_price ?? pending?.entry_price ?? assessment.entry_execution_vwap));
  setText("order-notional", money(plan?.notional));
  setText("target-one", money(open?.target_price ?? assessment.target_price));
  setText("target-two", money(open?.target_2_price ?? assessment.target_2_price));
  setText("stop-price", money(open?.current_stop_price ?? assessment.stop_price));
  setText("break-even", money(open?.break_even_price ?? assessment.break_even_price));
  setText("unrealized-pnl", money(open?.unrealized_pnl));
  paintValue("unrealized-pnl", open?.unrealized_pnl);
  setText("max-hold", `${plan?.maximum_hold_minutes ?? assessment.maximum_hold_minutes ?? 360} minutes`);
}

function renderFeeds(sources, readiness, coverage) {
  const definitions = [
    ["Alpha", sources.alpha],
    ["Execution", sources.execution],
    ["Funding", sources.funding],
  ];
  const list = $("feed-list");
  list.replaceChildren(...definitions.map(([label, source = {}]) => {
    const card = document.createElement("section");
    card.className = "feed-card";
    const header = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = label;
    const state = document.createElement("span");
    state.className = `badge ${source.valid ? "good" : "bad"}`;
    state.textContent = source.valid ? "FRESH" : "STALE";
    header.append(name, state);
    const purpose = document.createElement("p");
    purpose.textContent = source.purpose || "Required Binance source";
    const timing = document.createElement("small");
    timing.textContent = `Observed ${dateTime(source.observed_at)} / age ${age(source.age_seconds)} / limit ${age(source.max_age_seconds)}`;
    card.append(header, purpose, timing);
    return card;
  }));
  const allFresh = definitions.every(([, source]) => source?.valid);
  badge("feed-status", allFresh ? "ALL FRESH" : "FAIL CLOSED", allFresh ? "good" : "bad");
  setText("loaded-minutes", integer(readiness.official_closed_minutes_loaded));
  setText("model-rows", integer(readiness.model_ready_rows));
  setText("l2-rows", new Intl.NumberFormat("en-US").format(Number(coverage.binance_l2_rows) || 0));
  setText("l2-days", integer(coverage.binance_l2_utc_days));
}

function renderRisk(paper, safety, summary) {
  const drawdown = Number(paper.max_drawdown ?? summary.max_drawdown ?? 0);
  const limit = Number(safety.max_drawdown_limit ?? paper.max_strategy_drawdown ?? 0.08);
  const usage = limit > 0 ? Math.min(100, drawdown / limit * 100) : 0;
  $("risk-meter-fill").style.width = `${usage}%`;
  const blocked = Boolean(paper.risk_block_reason || Number(summary.kill_switches) > 0);
  badge("risk-status", blocked ? "BLOCKED" : "CLEAR", blocked ? "bad" : "good");
  setText("risk-per-trade", percent(paper.risk_per_trade ?? safety.risk_per_trade));
  setText("max-leverage", `${integer(paper.max_leverage ?? 10)}x`);
  setText("max-margin", percent(paper.margin_fraction ?? 0.1));
  setText("daily-loss", percent(paper.max_daily_loss ?? safety.max_daily_loss));
  setText("drawdown-limit", percent(paper.max_strategy_drawdown ?? safety.max_drawdown_limit));
}

function renderLedger(paper) {
  const rows = [];
  if (paper.pending_order) rows.push({ kind: "pending", value: paper.pending_order });
  if (paper.open_position) rows.push({ kind: "open", value: paper.open_position });
  (paper.trades || []).slice().reverse().forEach((trade) => rows.push({ kind: "closed", value: trade }));
  setText("ledger-count", `${rows.length} record${rows.length === 1 ? "" : "s"}`);
  const body = $("trade-ledger");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="10" class="empty-cell">No paper orders or completed trades yet. FLAT decisions do not create ledger entries.</td></tr>';
    return;
  }
  body.replaceChildren(...rows.map(({ kind, value }) => {
    const row = document.createElement("tr");
    const isOpen = kind === "open";
    const values = [
      dateTime(value.signal_at),
      kind.toUpperCase(),
      value.side || "--",
      money(value.entry_execution_price ?? value.entry_price),
      isOpen ? money(value.mark_price) : money(value.exit_execution_price),
      value.quantity_btc == null ? "--" : `${number(value.quantity_btc, 6)} BTC`,
      money(value.modeled_costs ?? value.fees),
      money(isOpen ? value.estimated_net_if_closed : value.net_pnl),
      money(value.balance ?? paper.final_equity),
      value.exit_reason ? String(value.exit_reason).replaceAll("_", " ") : isOpen ? "Position active" : "Awaiting execution",
    ];
    values.forEach((valueText, index) => {
      const cell = document.createElement("td");
      cell.textContent = valueText;
      if (index === 2) cell.className = value.side === "LONG" ? "positive" : value.side === "SHORT" ? "negative" : "";
      row.append(cell);
    });
    return row;
  }));
}

function renderStrategy(audit) {
  setText("protocol-hash", `Protocol ${String(audit.protocol_hash || "--").slice(0, 12)}`);
}

function renderValidation(audit, paper) {
  const alpha = audit.alpha || {};
  const historical = alpha.historical_audit || {};
  const metrics = historical.metrics || {};
  const selector = audit.selector || {};
  const coverage = audit.data_coverage || {};
  badge("validation-status", alpha.status || "RESEARCH PAPER", "paper");
  setText("audit-trades", integer(metrics.trades));
  setText("audit-frequency", `${number(metrics.trades_per_day, 2)} / day`);
  setText("audit-expectancy", signedBps(metrics.expectancy_bps));
  setText("audit-pf", number(metrics.profit_factor, 3));
  setText("audit-win-rate", percent(metrics.win_rate));
  setText("audit-drawdown", percent(metrics.max_drawdown));
  paintValue("audit-expectancy", metrics.expectancy_bps);
  setText("paper-trade-progress", `${paper.trades?.length || 0} / ${selector.minimum_holdout_trades || 100}`);
  setText("paper-day-progress", `${coverage.binance_l2_utc_days || 0} / ${selector.minimum_holdout_days || 10}`);
  setText("paper-minimum", `${selector.minimum_holdout_days || 10} days and ${selector.minimum_holdout_trades || 100} trades`);
  setText("holdout-opened", alpha.holdout_opened ? "YES" : "NO");

  const gates = historical.live_gates || {};
  const gateLabels = {
    minimum_historical_trades_300: "At least 300 historical OOS trades",
    frequency_3_per_day: "At least 3 trades per day",
    expectancy: "Positive net expectancy",
    profit_factor: "Profit factor gate",
    drawdown: "Drawdown gate",
    risk_budget: "No risk-budget violations",
    positive_active_days: "Majority of active days positive",
    stress_1_5x: "Positive under 1.5x execution-cost stress",
    bootstrap_lcb: "Positive 95% bootstrap lower bound",
    spa: "SPA significance gate",
  };
  const list = $("gate-list");
  list.replaceChildren(...Object.entries(gates).map(([key, passed]) => {
    const item = document.createElement("div");
    item.className = passed ? "gate-pass" : "gate-fail";
    const icon = document.createElement("span");
    icon.textContent = passed ? "PASS" : "FAIL";
    const label = document.createElement("strong");
    label.textContent = gateLabels[key] || key.replaceAll("_", " ");
    item.append(icon, label);
    return item;
  }));
}

function drawMarketChart() {
  const canvas = $("market-chart");
  if (!canvas) return;
  const candles = (chartState.candles || []).slice(-100);
  show("chart-empty", candles.length === 0);
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  if (!candles.length) return;

  const overlayValues = [chartState.target, chartState.stop, chartState.breakEven]
    .filter((value) => value != null && value !== "")
    .map(Number)
    .filter(Number.isFinite);
  const values = candles.flatMap((candle) => [Number(candle.low), Number(candle.high), Number(candle.center)]).filter(Number.isFinite).concat(overlayValues);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const pricePad = (rawMax - rawMin || rawMax * 0.001 || 1) * 0.08;
  const min = rawMin - pricePad;
  const max = rawMax + pricePad;
  const pad = { left: 12, right: 76, top: 15, bottom: 26 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const y = (price) => pad.top + (max - price) / (max - min) * plotHeight;
  const slot = plotWidth / candles.length;

  ctx.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "left";
  for (let index = 0; index <= 5; index += 1) {
    const price = max - (max - min) * index / 5;
    const py = y(price);
    ctx.strokeStyle = "rgba(116, 137, 164, .14)";
    ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    ctx.fillStyle = "#7f91a8";
    ctx.fillText(price.toLocaleString("en-US", { maximumFractionDigits: 1 }), width - pad.right + 9, py + 3);
  }

  candles.forEach((candle, index) => {
    const open = Number(candle.open); const high = Number(candle.high); const low = Number(candle.low); const close = Number(candle.close);
    if (![open, high, low, close].every(Number.isFinite)) return;
    const x = pad.left + slot * (index + 0.5);
    const color = close >= open ? "#22c99a" : "#f25f6d";
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y(high)); ctx.lineTo(x, y(low)); ctx.stroke();
    const bodyWidth = Math.max(1.5, Math.min(8, slot * 0.62));
    const bodyTop = Math.min(y(open), y(close));
    ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, Math.max(1, Math.abs(y(close) - y(open))));
  });

  const drawLine = (extract, color, dash = []) => {
    ctx.beginPath();
    let started = false;
    candles.forEach((candle, index) => {
      const value = Number(extract(candle));
      if (!Number.isFinite(value)) return;
      const x = pad.left + slot * (index + 0.5);
      started ? ctx.lineTo(x, y(value)) : ctx.moveTo(x, y(value));
      started = true;
    });
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  };
  drawLine((candle) => candle.center, "#54d8ce");

  const horizontal = (value, color, label) => {
    if (value == null || value === "") return;
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    const py = y(parsed);
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.fillText(label, pad.left + 5, py - 5);
  };
  horizontal(chartState.target, "#22c99a", "TARGET");
  horizontal(chartState.stop, "#f25f6d", "STOP");
  horizontal(chartState.breakEven, "#e7b14b", "BREAK-EVEN");

  const first = new Date(candles[0].timestamp);
  const last = new Date(candles.at(-1).timestamp);
  ctx.fillStyle = "#7f91a8";
  ctx.textAlign = "left"; ctx.fillText(first.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), pad.left, height - 7);
  ctx.textAlign = "right"; ctx.fillText(last.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), width - pad.right, height - 7);
}

window.addEventListener("resize", drawMarketChart);
refresh();
setInterval(refresh, 5000);
