const $ = (id) => document.getElementById(id);

const money = (value) => {
  if (value == null || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(number)
    : "—";
};
const number = (value, digits = 2) => value != null && value !== "" && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const percent = (value) => value != null && value !== "" && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
const dateTime = (value) => value ? new Date(value).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" }) : "—";
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const ageText = (value) => {
  if (value == null || value === "") return "—";
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.floor(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.floor(seconds % 3600 / 60)} min`;
};
const gateNames = {
  direction: "direzione trend Binance (15m/30m/VWAP)",
  trend_15m: "trend Binance 15m concorde",
  trend_30m: "trend Binance 30m concorde",
  vwap_zone: "prezzo nella zona VWAP prevista",
  taker_flow_restart: "taker flow concorde con la ripartenza",
  price_restart: "price action 1m concorde",
  vwap_extension: "distanza dal VWAP di almeno 5 bps",
  taker_flow_reversal: "taker flow di inversione",
  price_reversal: "price action di ritorno al VWAP",
  vwap_cross: "attraversamento causale del rolling VWAP",
  decision_cadence: "prossima decisione su candela 5m completata",
  episode_already_evaluated: "episodio gia valutato",
  anchor_available: "anchored VWAP Binance valido (60-1.800 s)",
  accepted_side: "prezzo sul lato accettato dell'anchor",
  binance_direction: "direzione Binance concorde con l'anchor",
  order_flow_direction: "order flow concorde con l'anchor",
  failed_anchor: "fallimento causale dell'anchor",
  binance_reversal: "inversione Binance rispetto all'anchor",
  waiting_new_impulse_pullback_restart: "nuovo impulso, pullback VWAP e ripartenza confermata",
  trend_1h_4h: "trend Binance 1h e 4h",
  spot_confirmation: "conferma Binance spot/perpetual",
  impulse_breakout: "breakout dell'impulso",
  relative_volume: "volume relativo",
  taker_flow_confirmation: "taker flow concorde",
  vwap_pullback: "pullback nella zona VWAP",
  causal_restart: "ripartenza causale confermata",
};
const featureNames = {
  anchor_age_seconds: "eta dell'anchor",
  anchored_vwap_distance_bps: "distanza dall'anchored VWAP",
  atr_15m: "ATR 15 minuti",
  atr_30m: "ATR 30 minuti",
  avwap_slope_bps_60s: "pendenza anchored VWAP",
  avwap_slope_change_bps: "variazione pendenza anchored VWAP",
  return_since_anchor_bps: "rendimento dall'anchor",
  rolling_vwap_avwap_convergence_bps: "convergenza rolling/anchored VWAP",
  rolling_vwap_vs_avwap_distance_bps: "distanza rolling/anchored VWAP",
};
const humanGate = (value) => gateNames[value] || String(value || "nessuno").replaceAll("_", " ");
let liveCandles = [];
let liveTicks = [];
let socket;
let pingTimer;
const savedProfile = (() => { try { return localStorage.getItem("strategy-profile"); } catch { return null; } })();
const allowedProfiles = ["musca-v5-binance", "musca-v2"];
let selectedProfile = new URLSearchParams(location.search).get("profile") || savedProfile || "musca-v5-binance";
if (selectedProfile === "musca-v5") selectedProfile = "musca-v5-binance";
if (!allowedProfiles.includes(selectedProfile)) selectedProfile = "musca-v5-binance";
let profileIds = "";
let refreshInFlight = false;
let researchData = null;

function setText(id, value) { const element = $(id); if (element) element.textContent = value; }

function setMuscaView(v2, v4) {
  const active = v2 || v4;
  document.querySelectorAll(".generic-profile-only").forEach((element) => element.classList.toggle("hidden", active));
  document.querySelectorAll(".musca-v2-only").forEach((element) => element.classList.toggle("hidden", !v2));
  document.querySelectorAll(".musca-v4-only").forEach((element) => element.classList.toggle("hidden", !v4));
  document.querySelectorAll(".musca-v2-runtime-only").forEach((element) => element.classList.toggle("hidden", !v2));
  document.querySelectorAll(".musca-runtime-only").forEach((element) => element.classList.toggle("hidden", !active));
  setText("center-label", active ? "Daily VWAP" : "VWAP center");
  setText("strategy-chart-title", v4 ? "Price, daily and anchored VWAP" : active ? "Price and daily VWAP" : "Price, center and adaptive range");
  setText("costs-label", v2 ? "Historical OOS replay" : "Modeled costs");
  setText("costs-detail", v2 ? "Separate from current shadow account" : "Fees + slippage");
}

function selectedV5Profile(audit) {
  return audit.selected_fee_profile || "VIP0";
}

function selectedV5Paper(audit) {
  const diagnostics = audit.one_position_diagnostics || {};
  return diagnostics.paper_accounts?.[selectedV5Profile(audit)] || diagnostics.paper_account || {};
}

function selectedV5Assessment(audit) {
  return audit.current_market_assessments?.[selectedV5Profile(audit)] || audit.current_market_assessment || {};
}

function paperVenue(audit = {}) {
  return selectedV5Paper(audit).execution_venue || (selectedProfile === "musca-v5-binance" ? "BINANCE" : "BITUNIX");
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const muscaOnly = selectedProfile.startsWith("musca-v5-") || ["musca-v2", "musca-v4", "musca-vwap-liquidity"].includes(selectedProfile);
    const [response, liveResponse, researchResponse, mlResponse] = await Promise.all([
      fetch(`/api/dashboard?profile=${encodeURIComponent(selectedProfile)}`, { cache: "no-store" }),
      fetch("/api/live", { cache: "no-store" }),
      muscaOnly ? null : fetch("/api/research", { cache: "no-store" }),
      muscaOnly ? null : fetch("/api/ml", { cache: "no-store" }),
    ]);
    if (!response.ok || !liveResponse.ok || researchResponse && !researchResponse.ok || mlResponse && !mlResponse.ok) throw new Error("Dashboard API is unavailable");
    render(await response.json(), await liveResponse.json(), researchResponse ? await researchResponse.json() : {}, mlResponse ? await mlResponse.json() : {});
  } catch (error) {
    $("empty-state").classList.remove("hidden");
    $("dashboard").classList.add("hidden");
    setText("empty-message", `Dashboard connection failed: ${error.message}`);
  } finally {
    refreshInFlight = false;
  }
}

function render(data, live, research, ml) {
  setText("last-updated", `Updated ${new Date(data.generated_at).toLocaleTimeString("en-US")}`);
  if (!data.available) {
    $("empty-state").classList.remove("hidden");
    $("dashboard").classList.add("hidden");
    setText("empty-message", data.error);
    return;
  }
  $("empty-state").classList.add("hidden");
  $("dashboard").classList.remove("hidden");
  const summary = data.summary;
  const latest = data.latest || {};
  const musca = summary.profile_id === "musca";
  const muscaV2 = summary.profile_id === "musca-v2";
  const muscaV4 = summary.profile_id === "musca-v4";
  const muscaV5 = summary.profile_id.startsWith("musca-v5-");
  const muscaLiquidity = summary.profile_id === "musca-vwap-liquidity";
  const v14Shadow = summary.profile_id === "v14-vwap";
  setMuscaView(muscaV2, muscaV4 || muscaV5 || muscaLiquidity);
  document.querySelectorAll(".musca-v5-only").forEach((element) => element.classList.toggle("hidden", !muscaV5));
  document.querySelectorAll(".legacy-musca-v4-only").forEach((element) => element.classList.toggle("hidden", !muscaV4));
  if (muscaV5) {
    setText("center-label", "Live rolling VWAP (diagnostic)");
    setText("anchor-label", "Live anchored VWAP (diagnostic)");
    setText("price-legend", "Binance Alpha price");
    setText("center-legend", "Rolling VWAP shown for live context");
    setText("anchor-legend", "Anchored VWAP shown for live context");
    setText("strategy-chart-title", "Binance Alpha · Binance execution · order plan");
  } else if (muscaV4) {
    setText("center-label", "Bitunix rolling VWAP (5m)");
    setText("anchor-label", "Execution anchored VWAP (Bitunix)");
    setText("price-legend", "Bitunix price");
    setText("center-legend", "Bitunix rolling VWAP 5m");
    setText("anchor-legend", "Bitunix anchored VWAP");
    setText("strategy-chart-title", "Bitunix price · rolling VWAP · anchored VWAP · piano ordine");
  } else {
    setText("price-legend", "Price");
    setText("center-legend", "VWAP");
    setText("anchor-legend", "Anchored VWAP");
  }
  setText("bot-title", muscaLiquidity ? "MUSCA VWAP LIQUIDITY" : muscaV5 ? "MUSCA · BTC VWAP ALPHA" : muscaV4 ? "MUSCA V4" : muscaV2 ? "MUSCA V2" : v14Shadow ? "V14 VWAP SHADOW" : musca ? "MUSCA BOT" : "Adaptive Range Bot");
  setText("entry-score-label", muscaV5 ? "Target probability" : muscaLiquidity ? "Flow/depth vote" : muscaV2 || muscaV4 ? "Stress EV (bps)" : musca ? "Momentum score" : "MR score");
  setText("atr-label", muscaV5 ? "Planned stop" : "ATR (14)");
  document.title = muscaLiquidity ? "MUSCA VWAP — Liquidity Shadow" : muscaV5 ? "MUSCA BTC Auto-MoE — Binance Paper Simulation" : muscaV4 ? "MUSCA V4 — Multi-Anchor VWAP Shadow" : muscaV2 ? "MUSCA V2 — Adaptive VWAP Shadow" : v14Shadow ? "V14 VWAP — BTC Shadow" : musca ? "MUSCA BOT — BTC Shadow" : "Adaptive Range Bot — Operations";
  if (muscaV5) setText("bot-title", `MUSCA BTC AUTO-MoE · ${summary.fee_profile || "BINANCE"}`);
  setText("mode", muscaV5 ? `${summary.fee_profile || "BINANCE"} PAPER` : String(summary.mode).toUpperCase());
  setText("instrument", summary.instrument);
  setText("timeframe", `${summary.timeframe_minutes} minutes`);
  const model15m = summary.model_input_readiness?.timeframes?.["15m"];
  const modelProgress = v14Shadow && model15m
    ? ` · MODEL ${summary.policy_action} · INPUT 15M ${model15m.consecutive_complete_bars}/${model15m.required_bars}`
    : "";
  setText(
    "active-profile",
    `${summary.profile_label} · ${summary.instrument} FUTURES · ${summary.timeframe_minutes}M · ${String(summary.mode).toUpperCase()}${modelProgress}`,
  );
  if (muscaV5) {
    setText(
      "active-profile",
      `MUSCA BTC AUTO-MoE · ${summary.fee_profile || "BINANCE"} · ${summary.instrument} FUTURES · ${summary.timeframe_minutes}M · BINANCE PAPER SIMULATION`,
    );
  }
  setText("latest-bar", dateTime(latest.timestamp));
  renderProfiles(data.variants || [], summary.profile_id);
  setText("equity-label", "Account equity");
  setText("net-pnl-label", "Net P&L");
  setText("drawdown-label", "Max drawdown");
  setText("signals-label", "Signals");
  setText("position-label", "Current position");
  setText("equity", money(summary.final_equity));
  setText("equity-change", `Started at ${money(summary.initial_equity)}`);
  setText("net-pnl", money(summary.net_pnl));
  $("net-pnl").className = Number(summary.net_pnl) >= 0 ? "safe" : "negative";
  setText("drawdown", percent(summary.max_drawdown));
  setText("signals", summary.signals);
  setText("rejected", `${summary.rejected_signals} rejected by risk controls`);
  const scenarios = summary.cost_scenarios || {};
  const historical = summary.historical_replay || {};
  setText(
    "costs",
    muscaV2 && scenarios["8bps_expectancy"] != null
      ? `OOS ${historical.trades || 0} trade · ${money(historical.final_equity)} · EV ${number(scenarios["8bps_expectancy"])} bps · stress ${number(scenarios["16bps_expectancy"])} bps`
      : v14Shadow && scenarios.maker_fee_only_4bps
      ? `${money(Number(summary.fees) + Number(summary.slippage))} conservative · maker net ${money(scenarios.maker_fee_only_4bps.net_pnl)} · taker net ${money(scenarios.taker_fee_only_12bps.net_pnl)}`
      : money(Number(summary.fees) + Number(summary.slippage)),
  );
  setText("fill-count", `${summary.operations} operation${summary.operations === 1 ? "" : "s"}`);
  setText("position-status", data.current_position.status);
  setText("position-quantity", `Quantity ${number(data.current_position.quantity, 6)}`);

  const paperAssessment = selectedV5Assessment(summary.forward_audit || {});
  const paperMarket = paperAssessment.market_inputs?.binance || {};
  const profileLive = muscaV5 && summary.profile_id === "musca-v5-binance"
    ? {
      available: true,
      status: paperMarket.book_synced ? "live" : "waiting",
      activity: `${paperAssessment.decision || "WAIT"} · ${paperAssessment.reason || "attesa dati"}`,
      latest: {
        close: paperMarket.price,
        open: paperMarket.price,
        high: paperMarket.price,
        low: paperMarket.price,
        volume: null,
        timestamp: paperAssessment.observed_at,
      },
      quote: {
        best_bid: paperMarket.best_bid,
        best_ask: paperMarket.best_ask,
        spread_bps: paperMarket.spread_bps,
      },
      bars: summary.forward_audit?.market_chart?.length || 0,
      warmup_bars: 288,
      age_seconds: paperAssessment.sources?.execution?.age_seconds,
      candles: [],
    }
    : live;
  renderLive(profileLive);
  const botLatest = summary.instrument === "BTCUSDT"
    ? latest
    : live.available ? { ...live.latest, activity: live.activity } : latest;
  renderLatest(botLatest);
  renderRisk(summary, data.safety);
  renderForwardTrades(summary.forward_audit || {});
  if (muscaV5) {
    renderForwardSummary(summary.forward_audit || {});
    renderForwardAssessment(summary.forward_audit || {});
    renderEconomicAlpha(summary.economic_alpha || {}, summary.forward_audit || {});
  }
  renderOperations(data.operations, data.no_trade_reason);
  renderTimeline(data.telemetry);
  renderMilestones(data.milestones);
  researchData = research;
  renderResearch();
  renderML(ml);
  const paper = selectedV5Paper(summary.forward_audit || {});
  const paperCurve = [
    { equity: paper.initial_equity },
    ...(paper.trades || []).map((trade) => ({ equity: trade.balance })),
  ];
  drawEquity(muscaV5 ? paperCurve : data.equity_curve);
  const selectedAssessment = selectedV5Assessment(summary.forward_audit || {});
  const selectedPaper = selectedV5Paper(summary.forward_audit || {});
  const chartPosition = selectedPaper.open_position || {};
  const chartSource = muscaV5 && summary.forward_audit?.market_chart?.length
    ? summary.forward_audit.market_chart.map((point) => ({
      ...point,
      target: chartPosition.target_price ?? selectedAssessment.target_price,
      stop: chartPosition.current_stop_price ?? selectedAssessment.stop_price,
      break_even: chartPosition.break_even_price ?? selectedAssessment.break_even_price,
    }))
    : muscaLiquidity ? data.telemetry : live.available ? live.candles : data.telemetry;
  drawRange(
    muscaV5
      ? chartSource
      : chartPoints(chartSource, latest, muscaV2, muscaV4 || muscaLiquidity),
  );
}

function renderEconomicAlpha(alpha, audit = {}) {
  const frozen = audit.alpha?.status === "RESEARCH_BASE_ALPHA_READY";
  if (frozen) {
    const paperProfiles = audit.alpha.paper_profiles || {};
    alpha = {
      available: true,
      status: audit.alpha.status,
      base_viable_profiles: Object.entries(paperProfiles)
        .filter(([, profile]) => profile.paper_eligible)
        .map(([profile]) => profile),
      profiles: Object.entries(paperProfiles).map(([profile, values]) => ({
        profile,
        trades: values.oos_2026?.trades,
        expectancy_bps: values.oos_2026?.expectancy_bps,
        lcb_95_bps: null,
        profit_factor: values.oos_2026?.profit_factor,
        max_drawdown: values.oos_2026?.max_drawdown,
        stress_2x_expectancy_bps: values.oos_2026_stress_2x?.expectancy_bps,
        base_financial_gates_passed: values.paper_eligible,
        stress_gate_passed: Number(values.oos_2026_stress_2x?.expectancy_bps) >= 0,
        trade_count_gate_passed: Number(values.oos_2026?.trades) >= 50,
      })),
    };
  }
  const status = alpha.status || "NOT_TRAINED";
  setText("v5-economic-status", status);
  $("v5-economic-status").className = status.includes("DEPLOYABLE") && !status.includes("NO_")
    ? "pill safe-pill"
    : "pill";
  const viable = alpha.base_viable_profiles || [];
  setText(
    "v5-economic-note",
    alpha.available
      ? frozen
        ? `Base deterministica congelata su 2024-2025; conferma 2026 pre-holdout. Profili paper con EV positivo: ${viable.length ? viable.join(" / ") : "nessuno"}. Lo stress esecuzione 2x non e leva. Gate 50 trade e holdout futuro vietano ancora denaro reale.`
        : `Ridge resta il champion formale. XGBoost e solo challenger shadow. Profili positivi ai gate base: ${viable.length ? viable.join(" / ") : "nessuno"}. Stress esecuzione 2x, 300 trade e holdout futuro restano obbligatori.`
      : "Nessun audit Alpha economico disponibile.",
  );
  const body = $("v5-vip-profile-body");
  if (!body) return;
  const profiles = alpha.profiles || [];
  if (!profiles.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty-cell">Training non disponibile.</td></tr>';
    return;
  }
  body.innerHTML = profiles.map((profile) => {
    const base = profile.base_financial_gates_passed;
    const complete = base && profile.stress_gate_passed && profile.trade_count_gate_passed;
    const outcome = complete ? "PASS COMPLETO" : base ? "BASE OK · RESEARCH" : "FLAT";
    const outcomeClass = complete ? "safe" : base ? "research-status-provisional" : "negative";
    return `<tr>
      <td data-label="Profilo"><strong>${escapeHtml(profile.profile)}</strong></td>
      <td data-label="Trade OOS">${number(profile.trades, 0)} / ${frozen ? 50 : 300}</td>
      <td data-label="EV netto" class="${Number(profile.expectancy_bps) > 0 ? "safe" : "negative"}">${number(profile.expectancy_bps)} bps</td>
      <td data-label="LCB 95%">${number(profile.lcb_95_bps)} bps</td>
      <td data-label="Profit factor">${number(profile.profit_factor, 3)}</td>
      <td data-label="Drawdown">${percent(profile.max_drawdown)}</td>
      <td data-label="Stress esecuzione 2× (non leva)" class="${Number(profile.stress_2x_expectancy_bps) >= 0 ? "safe" : "negative"}">${number(profile.stress_2x_expectancy_bps)} bps</td>
      <td data-label="Esito" class="${outcomeClass}">${outcome}</td>
    </tr>`;
  }).join("");
}

function renderForwardSummary(audit) {
  const paper = selectedV5Paper(audit);
  const venue = paperVenue(audit);
  const cost = audit.cost_interpretation || {};
  const pending = paper.pending_order ? 1 : 0;
  setText("equity-label", "V5 virtual balance");
  setText("equity", money(paper.final_equity ?? 10000));
  setText("equity-change", `Partito da ${money(paper.initial_equity ?? 10000)} · margine max ${percent(paper.margin_fraction ?? .10)} · leva ${number(paper.max_leverage ?? 10, 0)}x`);
  setText("net-pnl-label", "Net P&L · stress costs");
  setText("net-pnl", money(paper.net_pnl ?? 0));
  $("net-pnl").className = Number(paper.net_pnl || 0) >= 0 ? "safe" : "negative";
  setText("drawdown-label", "Virtual drawdown");
  setText("drawdown", percent(paper.max_drawdown ?? 0));
  setText("signals-label", "Paper trades");
  setText("signals", number(paper.trades?.length || 0, 0));
  setText("rejected", `${paper.trade_signal_count || 0}/${paper.complete_candidate_count || 0} candidati live autorizzati`);
  const open = paper.open_position || null;
  setText("position-label", "Paper position now");
  setText(
    "position-status",
    open
      ? `${open.side} OPEN · P&L ${money(open.unrealized_pnl)} · netto se chiusa ${money(open.estimated_net_if_closed)}`
      : paper.pending_order ? "ORDER PENDING" : "FLAT",
  );
  setText(
    "position-quantity",
    open
      ? `${number(open.quantity_btc, 6)} BTC · entry ${money(open.entry_execution_price)} · mark ${money(open.mark_price)} · target ${money(open.target_price)} · stop ${money(open.current_stop_price)}`
      : paper.pending_order
        ? `${paper.pending_order.side} ${number(paper.pending_order.quantity_btc, 6)} BTC · fill sul primo book ${venue} successivo`
        : `${pending} ordini in attesa · nessuna posizione aperta`,
  );
  setText("costs-label", "Virtual costs paid");
  setText("costs", money(paper.modeled_costs ?? 0));
  setText("costs-detail", `${cost.fee_venue || venue} · maker ${number(cost.maker_bps_per_side, 2)} / taker ${number(cost.taker_bps_per_side, 2)} bps per side · reserve/funding included`);
  const feeProfile = selectedV5Profile(audit);
  const venueProfile = feeProfile === venue ? venue : `${venue} ${feeProfile}`;
  setText(
    "rejected",
    `${paper.trade_signal_count || 0} TRADE live / ${paper.complete_candidate_count || 0} setup completi registrati dal ${dateTime(paper.decision_tracking_start || paper.paper_start)} · storico separato nel report Alpha`,
  );
  setText(
    "costs-detail",
    `${venueProfile} · maker ${number(paper.maker_fees_per_side_bps, 2)} / taker ${number(paper.fees_per_side_bps, 2)} bps per lato · spread/slippage nei fill a book · funding soltanto al settlement osservato`,
  );
}

function renderForwardAssessment(audit) {
  const state = selectedV5Assessment(audit);
  const venue = paperVenue(audit);
  const alpha = audit.alpha || {};
  const frozenBase = String(state.policy_source || "").startsWith("MUSCA_V8_FROZEN_BASE");
  const sourceAt = state.evaluated_at ? new Date(state.evaluated_at).getTime() : null;
  const elapsed = sourceAt == null ? 0 : Math.max(0, (Date.now() - sourceAt) / 1000);
  const sourceView = (source = {}) => {
    const reportedAge = Number(source.age_seconds);
    const age = Number.isFinite(reportedAge) ? reportedAge + elapsed : null;
    const fresh = Boolean(source.valid) && age != null && age <= Number(source.max_age_seconds);
    const currentState = fresh
      ? "FRESH"
      : source.valid && age > Number(source.max_age_seconds) ? "STALE" : source.state || "MISSING";
    return { ...source, age, fresh, state: currentState };
  };
  const alphaSource = sourceView(state.sources?.alpha);
  const executionSource = sourceView(state.sources?.execution);
  const dataFresh = alphaSource.fresh && executionSource.fresh;
  const reportedDecision = state.decision || "WAIT";
  const decision = dataFresh ? reportedDecision : "WAIT";
  const rawReason = String(state.reason || "In attesa di dati sincronizzati.");
  const blockedPrefix = "Closest setup blocked at: ";
  const readableReason = rawReason.startsWith(blockedPrefix)
    ? `Setup piu vicino bloccato da: ${humanGate(rawReason.slice(blockedPrefix.length))}`
    : rawReason;
  const missingFeatures = (state.model_feature_coverage?.missing || []).map(
    (feature) => featureNames[feature] || feature.replaceAll("_", " "),
  );
  setText("v5-decision", decision);
  $("v5-decision").className = decision === "TRADE" ? "pill safe-pill" : "pill";
  setText(
    "v5-decision-reason",
    dataFresh
      ? readableReason
      : `WAIT: osservazione non piu corrente. Ultima decisione registrata: ${reportedDecision} · ${readableReason}.`,
  );
  setText(
    "v5-model-context",
    frozenBase && !state.candidate_complete
      ? "Base congelata attiva: attende impulso, pullback VWAP e ripartenza confermata. Il challenger ML generico non puo autorizzare ordini."
      : `${state.candidate_complete
      ? "Candidato completo valutato"
      : "Ipotesi sul setup piu vicino · non e un ordine"} · feature modello ${state.model_feature_coverage?.complete ? "complete" : `incomplete (${missingFeatures.length}): ${missingFeatures.join(", ") || "non note"}`}`,
  );
  setText(
    "v5-evaluated-at",
    state.evaluated_at ? `${dateTime(state.evaluated_at)} · ${ageText(elapsed)} fa` : "—",
  );
  setText("v5-setup", String(state.setup || "-").replaceAll("_", " "));
  setText("v5-direction", state.direction || "-");
  setText("v5-probability", state.target_probability == null
    ? (state.probability_status || "NOT_TRAINED")
    : `target ${percent(state.target_probability)} · stop ${percent(state.stop_probability)} · timeout ${percent(state.timeout_probability)}`);
  setText("v5-ev", state.expected_net_ev_bps == null ? (frozenBase ? "Calcolato sul prossimo evento completo" : "Non disponibile prima del training") : `${number(state.expected_net_ev_bps)} bps`);
  setText("v5-excursion", state.expected_mfe_60m_bps == null ? "-" : `MFE ${number(state.expected_mfe_60m_bps)} · MAE ${number(state.expected_mae_60m_bps)} bps`);
  const vip = state.vip_ev_bps || {};
  setText("v5-vip-ev", Object.keys(vip).length ? Object.entries(vip).map(([level, value]) => `${level} ${number(value)} bps`).join(" · ") : "-");
  setText("v5-alpha-status", `${alpha.status || "NOT_TRAINED"} · profilo simulato ${selectedV5Profile(audit)}`);
  setText("v5-target", state.target_price == null ? "Nessun ordine proposto" : `${money(state.target_price)} · ${number(state.target_bps)} bps · tempo atteso ${number(state.expected_time_to_target_minutes)} min${decision === "TRADE" ? "" : " · scenario, non ordine"}`);
  setText("v5-stop", state.stop_price == null ? `${number(state.stop_bps)} bps se il setup completa` : `${money(state.stop_price)} · ${number(state.stop_bps)} bps`);
  setText("v5-break-even", money(state.break_even_price));
  setText("v5-cost", `${number(state.expected_cost_bps)} bps stimati`);
  setText(
    "v5-execution",
    state.entry_execution_vwap == null
      ? executionSource.fresh
        ? `Book ${venue} valido · fill non valutato senza candidato completo`
        : `Book ${venue} non valido o non corrente`
      : `${state.execution_type || "TAKER"} · entry ${money(state.entry_execution_vwap)} · exit stimata ${money(state.estimated_exit_execution_vwap)} · ${state.execution_levels_entry || 0} livelli`,
  );
  setText("v5-execution-status", state.execution_status || "NOT_EVALUATED");
  setText("v5-size", state.quantity_btc == null ? "Calcolata solo su candidato completo" : `${number(state.quantity_btc, 6)} BTC · ${money(state.notional)} notional`);
  const riskState = state.risk_status === "NOT_EVALUATED" || state.risk_approved == null
    ? "NON VALUTATO"
    : state.risk_approved ? "APPROVATO" : "RIFIUTATO";
  setText("v5-risk", `${riskState} · ${state.risk_reason || "-"}${state.risk_budget == null ? "" : ` · budget ${money(state.risk_budget)}`}`);
  const anchor = state.anchor || {};
  setText(
    "v5-anchor-state",
    `${anchor.state || "NONE"} · ${money(anchor.price)} · eta ${ageText(anchor.age_seconds)} · ${anchor.direction || "nessuna direzione"}`,
  );
  const setup = (state.setups || []).find((item) => item.setup === state.setup) || {};
  setText("v5-gates", `${setup.passed_checks || 0}/${setup.total_checks || 0} superati${setup.first_failed_check ? ` · manca: ${humanGate(setup.first_failed_check)}` : ""}`);
  setText("v5-horizons", frozenBase ? "Breakout 3 / 6 / 12 / 24 / 48 barre da 5m · gestione fino a 360 min" : `${(state.outcome_horizons_minutes || [5, 15, 30, 60]).join(" / ")} min · non sono frequenze di entrata`);
  setText("activity", `${decision} · ${dataFresh ? readableReason : "fonti non correnti"}`);
  setText("regime", state.probability_status || "NOT_TRAINED");
  setText("calc-close", money(state.price));
  setText("calc-center", money(state.rolling_vwap));
  setText("calc-anchor", money(state.anchored_vwap));
  setText("calc-atr", state.stop_bps == null ? "-" : `${number(state.stop_bps)} bps stop`);
  setText("calc-entry-score", state.target_probability == null ? "Non addestrata" : percent(state.target_probability));
  setText("calc-spread", `${number(state.spread_bps, 3)} bps`);
  setText("v5-alpha-status", `${alpha.status || "NOT_TRAINED"} · ${selectedV5Profile(audit)}`);

  const applySource = (prefix, source) => {
    setText(`${prefix}-state`, source.state || "MISSING");
    $(`${prefix}-state`).className = source.fresh ? "pill safe-pill" : "pill danger";
    setText(`${prefix}-purpose`, source.purpose || "Dato non disponibile");
    setText(`${prefix}-time`, dateTime(source.observed_at));
    setText(`${prefix}-age`, `${ageText(source.age)} · limite ${ageText(source.max_age_seconds)}`);
  };
  applySource("v5-alpha-source", alphaSource);
  applySource("v5-execution-source", executionSource);
  setText("v5-data-status", dataFresh ? "FRESH" : "FAIL CLOSED");
  $("v5-data-status").className = dataFresh ? "pill safe-pill" : "pill danger";
  setText(
    "v5-data-note",
    frozenBase
      ? `Binance perpetual e spot generano trend 1h/4h, impulso, pullback su daily/impulse/swing VWAP e ripartenza con volume e taker flow. Binance USD-M fornisce esecuzione, profondita, funding, costi e fill paper. Valutazione ${dateTime(state.evaluated_at)}.`
      : `Binance genera Alpha, bid/ask, profondita, funding, costi e fill paper. Valutazione ${dateTime(state.evaluated_at)}.`,
  );

  const binance = state.market_inputs?.binance || {};
  const execution = state.market_inputs?.[venue.toLowerCase()] || {};
  let inputs = [
    ["Prezzo Alpha", "Binance", money(binance.price), "Direzione e setup"],
    ["Rendimento 1 minuto", "Binance", `${number(binance.return_1m_bps)} bps`, "Price action causale"],
    ["Rendimento 5 minuti", "Binance", `${number(binance.return_5m_bps)} bps`, "Direzione del setup"],
    ["Rendimento 15 minuti", "Binance", `${number(binance.return_15m_bps)} bps`, "Trend Alpha intermedio"],
    ["Rendimento 30 minuti", "Binance", `${number(binance.return_30m_bps)} bps`, "Trend Alpha lento"],
    ["Rolling VWAP 60m", "Binance", money(binance.rolling_vwap), "Centro/benchmark Alpha"],
    ["Distanza dal VWAP", "Binance", `${number(binance.vwap_distance_bps)} bps`, "Zona pullback o estensione"],
    ["Slope VWAP", "Binance", `${number(binance.vwap_slope_bps)} bps`, "Terzo voto della direzione"],
    ["Voto trend", "Binance", number(binance.trend_score, 0), "sign(15m) + sign(30m) + sign(slope VWAP)"],
    ["Range ultimi 60 secondi", "Binance", `${number(binance.range_60s_bps)} bps`, "Stop e target dinamici"],
    ["Taker imbalance Alpha 60s", "Binance", number(binance.taker_imbalance_60s, 3), "Gate order flow causale"],
    ["Aggressive imbalance L2 60s", "Binance", number(binance.l2_aggressive_imbalance_60s, 3), "Diagnostica live; esclusa dal gate Alpha"],
    ["Depth imbalance", "Binance", number(binance.depth_imbalance_5, 3), "Diagnostica live; esclusa dall'Alpha storico"],
    ["Microprice distance", "Binance", `${number(binance.microprice_distance_bps, 3)} bps`, "Diagnostica live; esclusa dall'Alpha storico"],
    ["Flow vote L2", "Binance", number(binance.l2_flow_vote, 0), "Diagnostica live; esclusa dal gate Alpha"],
    ["Best bid / ask", venue, `${money(execution.best_bid)} / ${money(execution.best_ask)}`, "Quote eseguibile paper"],
    ["Spread", venue, `${number(execution.spread_bps, 3)} bps`, "Costo reale osservato"],
    ["Mark / index", venue, `${money(execution.mark_price)} / ${money(execution.index_price)}`, "Controllo prezzo perpetual"],
    ["Funding", venue, percent(execution.funding_rate), "Costo se attraversa il settlement"],
    ["Book sincronizzato", venue, execution.book_synced ? "SI" : "NO", "Gate di esecuzione"],
  ];
  if (frozenBase) {
    inputs = [
      ["Prezzo Alpha", "Binance", money(binance.price), "Prezzo causale perpetual"],
      ...inputs.slice(-5),
    ];
  }
  const inputsBody = $("v5-inputs-body");
  inputsBody.innerHTML = inputs.map(([name, venue, value, use]) => `<tr>
    <td data-label="Input"><strong>${escapeHtml(name)}</strong></td>
    <td data-label="Venue">${escapeHtml(venue)}</td>
    <td data-label="Valore">${escapeHtml(value)}</td>
    <td data-label="Uso">${escapeHtml(use)}</td>
  </tr>`).join("");

  const allSetups = state.setups || [];
  const setups = frozenBase
    ? allSetups.filter((item) => String(item.policy_source || "").startsWith("MUSCA_V8_FROZEN_BASE"))
    : allSetups;
  const completed = setups.filter((item) => item.candidate).length;
  setText("v5-setup-status", `${completed} / ${setups.length || 3} completi`);
  $("v5-setup-status").className = completed ? "pill safe-pill" : "pill";
  const setupBody = $("v5-setups-body");
  setupBody.innerHTML = setups.length ? setups.map((item) => {
    const checks = (item.checks || []).map((check) => `<div class="check-item ${check.passed ? "pass" : ""}">
      <i>${check.passed ? "✓" : "×"}</i><b>${escapeHtml(humanGate(check.name))}</b>
      <span>${escapeHtml(number(check.actual, 3))} · richiesto: ${escapeHtml(check.requirement)}</span>
    </div>`).join("");
    return `<tr>
      <td data-label="Setup"><strong>${escapeHtml(String(item.setup).replaceAll("_", " "))}</strong></td>
      <td data-label="Lato">${escapeHtml(item.direction || "-")}</td>
      <td data-label="Gate">${item.passed_checks || 0}/${item.total_checks || 0}</td>
      <td data-label="Stato" class="${item.candidate ? "safe" : "negative"}">${item.candidate ? "CANDIDATO" : "BLOCCATO"}</td>
      <td data-label="Primo blocco">${escapeHtml(humanGate(item.first_failed_check))}</td>
      <td data-label="Controlli"><div class="check-list">${checks}</div></td>
    </tr>`;
  }).join("") : '<tr><td colspan="6" class="empty-cell">In attesa della prima valutazione.</td></tr>';
}

function chartPoints(points, latest, muscaV2, muscaV4) {
  if (!muscaV2 && !muscaV4) return points;
  const anchorAt = muscaV4 && latest.anchor_at ? new Date(latest.anchor_at).getTime() : null;
  let weighted = 0;
  let volume = 0;
  return points.map((point) => {
    const result = { ...point, center: point.daily_vwap ?? point.center };
    if (anchorAt != null && new Date(point.timestamp).getTime() >= anchorAt) {
      const candleVolume = Number(point.volume);
      const close = Number(point.close);
      if (Number.isFinite(candleVolume) && candleVolume > 0 && Number.isFinite(close)) {
        weighted += Number.isFinite(Number(point.quote_volume)) ? Number(point.quote_volume) : close * candleVolume;
        volume += candleVolume;
        result.anchored_vwap = weighted / volume;
      }
    }
    return result;
  });
}

function renderML(data) {
  const status = data?.status || {};
  const phase = String(status.phase || (data?.available ? "complete" : "waiting")).replaceAll("_", " ");
  setText("ml-status", phase.toUpperCase());
  const progress = Number(status.percent || 0);
  const counter = status.total ? ` | ${status.completed || 0}/${status.total}` : "";
  const eta = status.eta_seconds != null ? ` | ETA ${formatDuration(status.eta_seconds)}` : "";
  setText("ml-progress", `${number(progress, 1)}%${counter}${eta}`);
  setText("ml-compute", `${String(status.backend || data?.compute?.backend || "waiting").toUpperCase()}${status.current_candidate ? ` | ${status.current_candidate}` : ""}`);
  if (!data?.available) {
    $("expert-policy-details").classList.add("hidden");
    setText("ml-summary", status.detail || data?.error || "No ML run loaded.");
    setText("ml-data", "Observed archive required");
    setText("ml-result", "No model trained");
    return;
  }
  if (data.protocol === "adaptive_range_multi_expert_v5") {
    const metrics = data.oos?.metrics || {};
    const experts = data.selected_experts || [];
    const enabled = Object.entries(data.oos?.side_enabled || {}).filter(([, value]) => value).map(([side]) => side.toUpperCase());
    const failures = data.oos?.failures || [];
    setText("ml-summary", `${data.run_id} | ${data.verdict} | ${experts.length} frozen experts | sides ${enabled.join(" + ") || "NONE"}`);
    setText("ml-data", `${dateTime(data.data?.development_end)} | ${number(data.counterfactual?.rows, 0)} causal rows`);
    setText("ml-result", `${data.holdout?.status || "sealed"} | PF ${number(metrics.profit_factor)} | trades ${number(metrics.trades, 0)} | drawdown ${percent(metrics.max_drawdown)}`);
    $("expert-policy-details").classList.remove("hidden");
    $("expert-policy-body").innerHTML = experts.slice(0, 24).map((expert) => `<tr><td>${escapeHtml(expert.expert_id)}</td><td>${escapeHtml(expert.side)}</td><td>${number(expert.timeframe_minutes, 0)}m</td><td>${number(expert.entry_z, 1)}</td><td>${number(expert.stop_atr, 1)}</td><td>${number(expert.exit_z, 1)}</td></tr>`).join("") || '<tr><td colspan="6" class="empty-cell">No expert passed train-only selection.</td></tr>';
    setText("expert-policy-gates", failures.length ? `FLAT / not deployable: ${failures.join(", ")}` : "All development gates passed; final holdout remains manual and sealed.");
    return;
  }
  $("expert-policy-details").classList.add("hidden");
  if (data.protocol === "scientific_v2") {
    const metrics = data.holdout?.opened ? data.holdout.metrics : data.development_metrics || {};
    const state = data.holdout?.opened
      ? (data.accepted ? "ACCEPTED CANDIDATE" : "REJECTED")
      : "SEALED";
    setText("ml-summary", `${status.detail || data.mode} | ${data.search.strategy_candidates} strategies | ${data.search.meta_candidates} GPU meta-candidates`);
    setText("ml-data", `${dateTime(data.data.development_start)} to ${dateTime(data.data.development_end)} | 5m / 15m / 30m`);
    setText("ml-result", `${state} | PF ${number(metrics.profit_factor)} | trades ${number(metrics.trades, 0)} | drawdown ${percent(metrics.max_drawdown)}`);
    return;
  }
  if (data.methodology_status === "baseline_invalid_for_selection") {
    setText("ml-summary", status.detail || "Legacy baseline invalidated.");
    setText("ml-data", `${data.data.start} to ${data.data.end} | legacy one-year archive`);
    setText("ml-result", "INVALID FOR SELECTION | waiting for scientific_v2");
    return;
  }
  const base = data.economic_scenarios?.base || {};
  setText("ml-summary", `${data.mode} | ${data.split.final_test_rows} untouched test rows | ${data.optimization.trials} Optuna trials`);
  setText("ml-data", `${data.data.start} → ${data.data.end} | spread ${data.data.historical_spread}`);
  setText("ml-result", `${data.accepted ? "ACCEPTED" : "REJECTED"} | PF ${number(base.profit_factor)} | trades ${number(base.trades, 0)} | drawdown ${percent(base.max_drawdown)}`);
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function renderResearch() {
  const data = researchData;
  const body = $("research-body");
  if (!data?.available) {
    const status = data?.status || {};
    const phase = String(status.phase || "waiting").replaceAll("_", " ");
    setText("research-status", phase.toUpperCase());
    const counter = ["evaluating", "shadow"].includes(status.phase) ? ` (${status.completed}/${status.total})` : "";
    const progress = status.total ? ` ${number(status.percent, 1)}%${counter}` : "";
    setText("research-summary", status.detail ? `${phase}${progress} | ${status.detail} | ${dateTime(status.updated_at)}` : data?.error || "No research run loaded.");
    body.innerHTML = '<tr><td colspan="11" class="empty-cell">No research results.</td></tr>';
    return;
  }
  const family = $("research-family").value;
  const state = $("research-state").value;
  const horizon = $("research-horizon").value;
  const rows = data.evaluations.filter((item) =>
    (family === "all" || item.family === family) && (state === "all" || item.status === state)
  ).slice(0, 100);
  const phase = String(data.status?.phase || "ready").replaceAll("_", " ");
  const running = data.status?.total
    ? `V1.1 progress ${number(data.status.percent, 1)}% (${data.status.completed}/${data.status.total}) | `
    : "";
  setText("research-status", phase.toUpperCase());
  const pbo = data.selection_bias?.pbo;
  const audit = pbo == null ? "PBO waiting for 4+ blocks" : `PBO ${percent(pbo)} across ${data.selection_bias.splits} CSCV splits`;
  setText("research-summary", `${running}${data.counts.validated} validated | ${data.counts.provisional} provisional | ${data.counts.insufficient} insufficient. ${audit}. Fixed risk ${percent(data.fixed.risk_per_trade)}, leverage ${data.fixed.leverage}x. Shadow updated ${dateTime(data.shadow_updated_at)}.`);
  body.replaceChildren(...rows.map((item) => {
    const metrics = item.metrics.horizons?.[horizon] || item.metrics;
    const row = document.createElement("tr");
    const values = [item.rank, item.candidate_id, item.family, item.status, JSON.stringify(item.parameters), metrics.trades, money(metrics.expectancy), number(metrics.profit_factor_cost_2x), percent(metrics.max_drawdown), percent(metrics.positive_window_rate), percent(item.metrics.deflated_sharpe_probability)];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 3) cell.className = `research-status-${item.status}`;
      row.append(cell);
    });
    return row;
  }));
  if (!rows.length) body.innerHTML = '<tr><td colspan="11" class="empty-cell">No candidates match these filters.</td></tr>';
}

function renderProfiles(variants, activeProfile) {
  variants = variants.filter((variant) => allowedProfiles.includes(variant.profile_id));
  const selector = $("adx-profile");
  selector.disabled = false;
  const ids = variants.map((variant) => variant.profile_id).join(",");
  if (variants.length && ids !== profileIds) {
    selector.replaceChildren(...variants.map((variant) => {
      const option = document.createElement("option");
      option.value = variant.profile_id;
      option.textContent = `${variant.profile_label} · ${money(variant.net_pnl)} · ${variant.operations} ops`;
      return option;
    }));
    profileIds = ids;
  }
  variants.forEach((variant) => {
    const option = [...selector.options].find((item) => item.value === variant.profile_id);
    if (option) option.textContent = `${variant.profile_label} · ${money(variant.net_pnl)} · ${variant.operations} ops`;
  });
  if (![...selector.options].some((option) => option.value === selectedProfile)) {
    selectedProfile = activeProfile;
  }
  selector.value = selectedProfile;
}

function renderLive(live) {
  const badge = $("live-status");
  badge.textContent = live.available ? String(live.status).toUpperCase() : "WAITING";
  badge.className = live.available && live.status === "live" ? "pill safe-pill" : "pill danger";
  setText("live-activity", live.activity || live.error);
  if (!live.available) return;
  const latest = live.latest;
  setText("live-close", money(latest.close));
  setText("live-time", dateTime(latest.timestamp));
  setText("live-age", `${live.age_seconds} seconds`);
  setText("live-warmup", `${live.bars}/${live.warmup_bars} bars`);
  setText("live-open", money(latest.open));
  setText("live-high", money(latest.high));
  setText("live-low", money(latest.low));
  setText("live-volume", number(latest.volume, 4));
  setText("live-bid", money(live.quote?.best_bid));
  setText("live-ask", money(live.quote?.best_ask));
  setText("live-spread", `${number(live.quote?.spread_bps, 3)} bps`);
  liveCandles = live.candles;
  drawCandlesticks(liveCandles);
}

function renderLatest(latest) {
  const regime = String(latest.regime || "unknown").toUpperCase().replaceAll("_", " ");
  const badge = $("regime");
  badge.textContent = regime;
  badge.dataset.regime = latest.regime || "unknown";
  setText("activity", latest.decision_reason || latest.activity || "No decision is available.");
  setText("calc-close", money(latest.close));
  setText("calc-center", money(latest.center));
  setText("calc-atr", number(latest.atr, 4));
  setText("calc-adx", number(latest.adx, 2));
  setText("calc-z", number(latest.z_score, 3));
  setText("calc-entry-score", number(latest.entry_score, 3));
  setText("calc-anchor", money(latest.anchored_vwap));
  setText("calc-atr-pct", latest.atr_percentile == null ? "—" : `${number(latest.atr_percentile, 1)}%`);
  setText("calc-slope", number(latest.ema_slope, 4));
  setText("calc-spread", `${number(latest.spread_bps, 2)} bps`);
  setText("calc-lower", money(latest.lower_band));
  setText("calc-upper", money(latest.upper_band));
}

function renderRisk(summary, safety) {
  const usage = Math.min(100, Number(summary.max_drawdown) / Number(safety.max_drawdown_limit) * 100);
  setText("risk-usage", `${number(usage, 1)}% of limit`);
  $("risk-meter-fill").style.width = `${usage}%`;
  setText("risk-per-trade", percent(safety.risk_per_trade));
  setText("daily-loss-limit", percent(safety.max_daily_loss));
  setText("weekly-loss-limit", percent(safety.max_weekly_loss));
  const audit = summary.forward_audit || {};
  const paper = selectedV5Paper(audit);
  const alpha = audit.alpha || {};
  const protocolHash = String(audit.protocol_hash || "").slice(0, 8);
  setText("forward-audit-status", audit.available
    ? `${audit.validation_status} · Alpha ${alpha.status || "NOT_TRAINED"} · protocollo ${protocolHash || "—"} · ${paper.complete_candidate_count || 0} setup live · ${paper.trade_signal_count || 0} TRADE`
    : "Waiting for forward rows");
  const active = Number(summary.kill_switches) > 0;
  const badge = $("kill-status");
  badge.textContent = active ? "TRIGGERED" : "CLEAR";
  badge.className = active ? "pill danger" : "pill safe-pill";
}

function renderForwardTrades(audit) {
  const body = $("forward-trades-body");
  if (!body) return;
  const feeProfile = selectedV5Profile(audit);
  const paper = selectedV5Paper(audit);
  const venue = paperVenue(audit);
  setText("forward-protocol-name", audit.protocol?.name || "BTC VWAP Alpha candidates");
  setText(
    "forward-portfolio-status",
    `Conto paper persistente: ${paper.trades?.length || 0} trade chiusi dal ${dateTime(paper.paper_start)} · ${paper.trade_signal_count || 0} TRADE autorizzati / ${paper.complete_candidate_count || 0} setup completi da quando il registro live è attivo · equity ${money(paper.final_equity ?? 10000)} · ${paper.open_position ? "1 posizione aperta" : paper.pending_order ? "1 ordine pending" : "FLAT"} · ultimo evento ${paper.last_event || "WAIT"}${paper.last_assessment ? ` · ultima decisione ${paper.last_assessment.decision}: ${paper.last_assessment.reason}` : ""}${paper.risk_block_reason ? ` · BLOCCO ${paper.risk_block_reason}` : ""}`,
  );
  const selector = audit.selector || {};
  const alpha = audit.alpha || {};
  const trackingStart = new Date(paper.decision_tracking_start || paper.paper_start || 0);
  const paperTrades = paper.trades || [];
  const trades = [...paperTrades].reverse();
  const coverage = audit.data_coverage || {};
  setText(
    "forward-selector-status",
    `Alpha ${alpha.status || "NOT_TRAINED"} · ${feeProfile} · paper da ${dateTime(paper.decision_tracking_start || paper.paper_start)}: ${paper.trade_signal_count || 0} TRADE / ${paper.complete_candidate_count || 0} setup completi · ${venue} L2 ${coverage.binance_l2_utc_days || 0} giorni / ${coverage.binance_l2_rows || 0} snapshot · audit dopo ≥${selector.minimum_holdout_days || 10} giorni e ≥${selector.minimum_holdout_trades || 100} trade`,
  );
  const protocolHash = (audit.protocol_hash || "").slice(0, 8);
  setText("forward-trade-count", `${paperTrades.length} paper trades · ${paper.trade_signal_count || 0} TRADE live · ${paper.complete_candidate_count || 0} setup completi live${protocolHash ? ` · ${protocolHash}` : ""}`);
  const labels = ["Segnale", "Stato", "Ingresso / ordine", "Entrata → uscita", "Size", "Costi", "P&L netto", "Saldo", "Motivo uscita"];
  const makeRow = (values, side) => {
    const row = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      cell.dataset.label = labels[index];
      if (index === 2) cell.className = side === "LONG" ? "side-buy" : "side-sell";
      row.appendChild(cell);
    });
    return row;
  };
  const active = [];
  if (paper.pending_order) {
    const order = paper.pending_order;
    active.push(makeRow([
      dateTime(order.signal_at),
      "ACKNOWLEDGED · ATTESA NEXT BOOK",
      `${String(order.setup || "setup").replaceAll("_", " ")} · ${order.side}`,
      "Il fill non usa il book che ha generato il segnale",
      `${number(order.quantity_btc, 6)} BTC`,
      `${number(order.expected_cost_bps)} bps attesi`,
      "—",
      money(paper.final_equity),
      `Market fill simulato sul prossimo snapshot ${venue} valido`,
    ], order.side));
  }
  if (paper.open_position) {
    const open = paper.open_position;
    active.push(makeRow([
      dateTime(open.signal_at),
      `${open.side} OPEN · ${open.trailing_status || "NO TRAIL"}`,
      `${String(open.setup || "setup").replaceAll("_", " ")} · fill ${dateTime(open.entry_at)}`,
      `${money(open.entry_execution_price)} → mark ${money(open.mark_price)}`,
      `${money(open.notional)} · ${number(open.quantity_btc, 6)} BTC · margine ${money(open.margin_used)}`,
      `fee ingresso ${money(open.entry_fee)} · uscita ancora stimata`,
      `${money(open.unrealized_pnl)} lordo · ${money(open.estimated_net_if_closed)} netto se chiusa ora`,
      money(paper.final_equity),
      `Target ${money(open.target_price)} · stop ${money(open.current_stop_price)} · timeout ${number(open.maximum_hold_minutes ?? 60, 0)}m`,
    ], open.side));
  }
  if (!trades.length && !active.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty-cell">Nessun ordine paper eseguito. Il conto registra il prossimo TRADE autorizzato.</td></tr>';
    return;
  }
  body.replaceChildren(...active, ...trades.map((trade) => {
    const trainingEligible = new Date(trade.signal_at) >= trackingStart;
    const family = trade.expert || trade.family;
    const entryRule = {
      pullback_continuation_dynamic: "Continuazione dopo pullback sul VWAP",
      vwap_reversion_dynamic: "Rientro verso il rolling VWAP",
      rolling_vwap_reentry_dynamic: "Ripartenza dopo attraversamento del rolling VWAP",
      anchor_continuation_dynamic: "Continuazione dall'anchored VWAP",
      anchor_failure_dynamic: "Fallimento anchored VWAP / inversione",
    }[family] || String(family || "Causal VWAP setup").replaceAll("_", " ");
    const reason = {
      FLOW_INVALIDATION: "Chiuso: order flow contrario due volte",
      AVWAP_FAILURE: "Chiuso: anchored VWAP fallito",
      DYNAMIC_STOP: "Chiuso: stop dinamico",
      TRAIL: "Chiuso: trailing profit",
      TIME: "Chiuso: limite temporale della policy",
    }[trade.exit_reason] || `Chiuso: ${trade.exit_reason}`;
    const values = [
      dateTime(trade.signal_at),
      trainingEligible ? "PAPER · OSSERVAZIONE EXECUTION" : "PAPER · PRELIMINARE ESCLUSO",
      `${entryRule} · ${trade.side === "LONG" ? "BUY → SELL" : "SELL → BUY"} · target ${number(trade.alpha_target_bps)} bps · P ${percent(trade.alpha_target_probability)}`,
      `${money(trade.entry_execution_price)} → ${money(trade.exit_execution_price)}`,
      `${money(trade.notional)} · ${number(trade.quantity_btc, 5)} BTC`,
      `${money(trade.modeled_costs)} · ${trade.fee_profile || feeProfile}`,
      money(trade.net_pnl ?? trade.stress_pnl),
      money(trade.balance),
      reason,
    ];
    return makeRow(values, trade.side);
  }));
}

function renderOperations(operations, noTradeReason) {
  const body = $("fills-body");
  if (!operations.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-cell"></td></tr>';
    body.querySelector("td").textContent = noTradeReason;
    return;
  }
  body.replaceChildren(...operations.map((operation) => {
    const row = document.createElement("tr");
    const values = [
      dateTime(operation.timestamp),
      operation.event,
      String(operation.side).toUpperCase(),
      number(operation.quantity, 6),
      money(operation.price),
      number(operation.position_after, 6),
      operation.details,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 2) cell.className = operation.side === "buy" ? "side-buy" : "side-sell";
      row.appendChild(cell);
    });
    return row;
  }));
}

function renderTimeline(points) {
  const meaningful = [...points].reverse().filter((point, index, all) =>
    index === 0 || point.activity !== all[index - 1].activity || point.regime !== all[index - 1].regime
  ).slice(0, 30);
  $("timeline").replaceChildren(...meaningful.map((point) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    const copy = document.createElement("p");
    time.dateTime = point.timestamp;
    time.textContent = `${dateTime(point.timestamp)} · ${String(point.regime).replaceAll("_", " ")}`;
    copy.textContent = point.activity;
    item.append(time, copy);
    return item;
  }));
}

function renderMilestones(milestones) {
  $("milestones").replaceChildren(...milestones.map((milestone, index) => {
    const item = document.createElement("li");
    item.className = milestone.status;
    const marker = document.createElement("span");
    marker.className = "milestone-index";
    marker.textContent = milestone.status === "complete" ? "✓" : index + 1;
    const copy = document.createElement("div");
    const name = document.createElement("strong");
    const label = document.createElement("small");
    name.textContent = milestone.name;
    label.textContent = milestone.label;
    copy.append(name, label);
    const status = document.createElement("span");
    status.className = "milestone-status";
    status.textContent = milestone.status;
    item.append(marker, copy, status);
    return item;
  }));
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, rect.width * ratio);
  canvas.height = Math.max(1, rect.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  return { context, width: rect.width, height: rect.height };
}

function drawSeries(context, values, bounds, width, height, color, lineWidth = 1.6) {
  if (values.length < 2) return;
  const pad = 18;
  const x = (index) => pad + index / (values.length - 1) * (width - pad * 2);
  const y = (value) => pad + (bounds.max - value) / (bounds.max - bounds.min || 1) * (height - pad * 2);
  context.beginPath();
  let started = false;
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) return;
    if (!started) context.moveTo(x(index), y(value)); else context.lineTo(x(index), y(value));
    started = true;
  });
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.stroke();
}

function chartGrid(context, width, height) {
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(145,161,181,.12)";
  context.lineWidth = 1;
  for (let row = 1; row < 5; row += 1) {
    const y = row * height / 5;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
}

function boundsOf(series) {
  const values = series.flat().filter(Number.isFinite);
  if (!values.length) return { min: 0, max: 1 };
  const min = Math.min(...values); const max = Math.max(...values); const pad = (max - min || 1) * .08;
  return { min: min - pad, max: max + pad };
}

function drawEquity(points) {
  const canvas = $("equity-chart");
  const { context, width, height } = setupCanvas(canvas);
  chartGrid(context, width, height);
  const values = points.map((point) => Number(point.equity));
  drawSeries(context, values, boundsOf([values]), width, height, "#4fd1c5", 2);
}

function drawRange(points) {
  const canvas = $("range-chart");
  const { context, width, height } = setupCanvas(canvas);
  chartGrid(context, width, height);
  const close = points.map((point) => Number(point.close));
  const execution = points.map((point) => point.execution_price == null ? null : Number(point.execution_price));
  const center = points.map((point) => point.center == null ? null : Number(point.center));
  const lower = points.map((point) => point.lower_band == null ? null : Number(point.lower_band));
  const upper = points.map((point) => point.upper_band == null ? null : Number(point.upper_band));
  const anchored = points.map((point) => point.anchored_vwap == null ? null : Number(point.anchored_vwap));
  const target = points.map((point) => point.target == null ? null : Number(point.target));
  const stop = points.map((point) => point.stop == null ? null : Number(point.stop));
  const breakEven = points.map((point) => point.break_even == null ? null : Number(point.break_even));
  const bounds = boundsOf([close, execution, center, lower, upper, anchored, target, stop, breakEven]);
  drawSeries(context, lower, bounds, width, height, "rgba(243,183,79,.6)", 1);
  drawSeries(context, upper, bounds, width, height, "rgba(243,183,79,.6)", 1);
  drawSeries(context, center, bounds, width, height, "#4fd1c5", 1.5);
  drawSeries(context, anchored, bounds, width, height, "#d58cff", 1.8);
  drawSeries(context, target, bounds, width, height, "#69d391", 1.4);
  drawSeries(context, stop, bounds, width, height, "#ff6b78", 1.4);
  drawSeries(context, breakEven, bounds, width, height, "#f3b74f", 1.2);
  drawSeries(context, execution, bounds, width, height, "rgba(244,247,251,.72)", 1.2);
  drawSeries(context, close, bounds, width, height, "#5aa7ff", 2);
}

function drawCandlesticks(points) {
  const canvas = $("live-candle-chart");
  const { context, width, height } = setupCanvas(canvas);
  chartGrid(context, width, height);
  const candles = points.slice(-60);
  if (!candles.length) return;
  const bounds = boundsOf([candles.flatMap((point) => [Number(point.low), Number(point.high)])]);
  const pad = 18;
  const slot = (width - pad * 2) / candles.length;
  const y = (value) => pad + (bounds.max - value) / (bounds.max - bounds.min || 1) * (height - pad * 2);
  candles.forEach((candle, index) => {
    const open = Number(candle.open); const high = Number(candle.high);
    const low = Number(candle.low); const close = Number(candle.close);
    if (![open, high, low, close].every(Number.isFinite)) return;
    const x = pad + slot * (index + .5);
    const color = close >= open ? "#69d391" : "#ff6b78";
    context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 1;
    context.beginPath(); context.moveTo(x, y(high)); context.lineTo(x, y(low)); context.stroke();
    context.fillRect(x - Math.max(1, slot * .28), Math.min(y(open), y(close)), Math.max(2, slot * .56), Math.max(1, Math.abs(y(close) - y(open))));
  });
}

function drawRealtime() {
  const canvas = $("realtime-chart");
  const { context, width, height } = setupCanvas(canvas);
  chartGrid(context, width, height);
  drawSeries(context, liveTicks, boundsOf([liveTicks]), width, height, "#4fd1c5", 2);
}

function connectBinance() {
  const badge = $("ws-status");
  badge.textContent = "CONNECTING";
  badge.className = "pill";
  socket = new WebSocket("wss://fstream.binance.com/stream?streams=btcusdt@markPrice@1s/btcusdt@kline_5m");
  socket.addEventListener("message", ({ data }) => {
    let message;
    try { message = JSON.parse(data); } catch { return; }
    const payload = message.data;
    if (!payload) return;
    badge.textContent = "STREAMING";
    badge.className = "pill safe-pill";
    if (payload.e === "kline" && payload.k) {
      const candle = { timestamp: new Date(payload.k.t).toISOString(), open: payload.k.o, high: payload.k.h, low: payload.k.l, close: payload.k.c };
      liveCandles = liveCandles.at(-1)?.timestamp === candle.timestamp
        ? [...liveCandles.slice(0, -1), candle]
        : [...liveCandles.slice(-59), candle];
      drawCandlesticks(liveCandles);
    }
    if (payload.e === "markPriceUpdate") {
      const mark = Number(payload.p);
      if (!Number.isFinite(mark)) return;
      liveTicks = [...liveTicks.slice(-199), mark];
      setText("realtime-price", money(mark));
      setText("realtime-time", `Updated ${new Date(payload.E).toLocaleTimeString("en-US")}`);
      setText("index-price", money(payload.i));
      setText("funding-rate", percent(payload.r));
      drawRealtime();
    }
  });
  socket.addEventListener("close", () => {
    clearInterval(pingTimer);
    badge.textContent = "RECONNECTING";
    badge.className = "pill danger";
    setTimeout(connectBinance, 3000);
  });
  socket.addEventListener("error", () => socket.close());
}

window.addEventListener("resize", () => refresh());
$("adx-profile").addEventListener("change", (event) => {
  selectedProfile = event.target.value;
  try { localStorage.setItem("strategy-profile", selectedProfile); } catch {}
  refresh();
});
["research-family", "research-state", "research-horizon"].forEach((id) => $(id).addEventListener("change", renderResearch));
refresh();
setInterval(refresh, 5000);
connectBinance();
