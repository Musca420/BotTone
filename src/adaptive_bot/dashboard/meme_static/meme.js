const $ = id => document.getElementById(id);
let payload = null;
const money = value => value == null ? "-" : `${Number(value).toFixed(2)} USDT`;
const number = (value, digits = 2) => value == null ? "-" : Number(value).toFixed(digits);

function canvas(id, series, color) {
  const element = $(id), ratio = devicePixelRatio || 1, width = element.clientWidth, height = element.clientHeight;
  element.width = width * ratio; element.height = height * ratio;
  const context = element.getContext("2d"); context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
  if (series.length < 2) return;
  const low = Math.min(...series), high = Math.max(...series), pad = 18;
  const y = value => pad + (high - value) / (high - low || 1) * (height - pad * 2);
  context.strokeStyle = "#263047";
  for (let index = 1; index < 5; index++) { context.beginPath(); context.moveTo(0, height * index / 5); context.lineTo(width, height * index / 5); context.stroke(); }
  context.strokeStyle = color; context.lineWidth = 2; context.beginPath();
  series.forEach((value, index) => { const x = pad + index / (series.length - 1) * (width - pad * 2); index ? context.lineTo(x, y(value)) : context.moveTo(x, y(value)); });
  context.stroke();
}

function events(id, items) {
  $(id).replaceChildren(...items.slice(-80).reverse().map(item => {
    const element = document.createElement("div"), heading = document.createElement("strong"), copy = document.createElement("p");
    element.className = "event"; heading.textContent = `${item.symbol || "SYSTEM"} - ${item.event || item.action}`;
    copy.textContent = `${new Date(item.timestamp).toLocaleString("en-US")} - ${item.reason || ""}`;
    element.append(heading, copy); return element;
  }));
}

function scanner(rows) {
  $("scanner").replaceChildren(...rows.map(item => {
    const row = document.createElement("tr");
    [item.rank ?? "-", item.symbol, item.status || (item.eligible ? "ELIGIBLE" : "BLOCKED"), number(item.momentum_atr), number(item.volume_zscore), `${number(item.spread_bps)} bps`, item.funding_8h == null ? "UNKNOWN" : `${number(Number(item.funding_8h) * 100, 4)}%`, money(item.depth), number(item.liquidity_score), number(item.manipulation_risk), (item.reasons || []).join(", ") || "Ready"].forEach((value, index) => {
      const cell = document.createElement("td"); cell.textContent = value;
      if (index === 2) cell.className = item.eligible ? "ok" : "blocked";
      row.append(cell);
    });
    return row;
  }));
}

function market() {
  if (!payload) return;
  const state = (payload.stream.symbols || {})[$("symbol").value] || {}, candle = state.latest_candle;
  const rate = state.funding_rate == null ? null : Number(state.funding_rate), interval = Number(state.funding_interval_hours);
  const funding = rate == null || !Number.isFinite(rate) || !Number.isFinite(interval) || interval <= 0 ? "UNKNOWN" : `${number(rate * 100, 4)}% / ${number(interval, 0)}h (${number(rate * 8 / interval * 100, 4)}% / 8h)`;
  $("market-detail").textContent = `Mark ${money(state.mark_price)} - Index ${money(state.index_price)} - Spread ${number(state.spread_bps)} bps - Funding ${funding}`;
  canvas("price-chart", candle ? [Number(candle.open), Number(candle.low), Number(candle.high), Number(candle.close)] : [], "#4fd1c5");
}

function readiness(stream, report) {
  const policy = report.luna?.policy || {}, eligible = (report.scanner || []).filter(item => item.eligible).length, open = (report.positions || []).length, maximum = Number(report.risk?.max_open_positions || 2);
  const gates = [
    ["Live market feed", Boolean(stream.connected), stream.connected ? "Receiving Bitunix public data" : "Feed is disconnected or stale"],
    ["Luna Max market policy", policy.action === "ALLOW_EVALUATION", policy.action ? `Current action: ${policy.action.replaceAll("_", " ")}` : "No validated policy"],
    ["Tradable contract", eligible > 0, eligible ? `${eligible} contract(s) passed liquidity, funding and spread checks` : "No contract currently passes the scanner"],
    ["Portfolio capacity", open < maximum, `${open}/${maximum} simulated position(s) open`],
    ["Probabilistic mode", Boolean(report.probabilistic?.can_trade), report.probabilistic?.can_trade ? "Paper bootstrap active; estimates remain shadow-only" : "Validated model gate is not ready"],
  ];
  $("entry-gates").replaceChildren(...gates.map(([label, pass, detail]) => {
    const row = document.createElement("div"), dot = document.createElement("i"), heading = document.createElement("strong"), copy = document.createElement("span");
    row.className = `gate ${pass ? "pass" : ""}`; dot.className = "dot"; heading.textContent = label; copy.textContent = detail;
    row.append(dot, heading, copy); return row;
  }));
  $("entry-summary").textContent = `${gates.filter(gate => gate[1]).length}/${gates.length} required gates ready`;
  $("policy-thesis").textContent = policy.thesis || "Waiting for Luna Max.";
  $("policy-expiry").textContent = policy.expires_at ? `Reassess by ${new Date(policy.expires_at).toLocaleString("en-US")}` : "No active policy";
  $("policy-conditions").replaceChildren(...(policy.invalidation_conditions || []).map(value => { const item = document.createElement("li"); item.textContent = value; return item; }));
  $("policy-sources").replaceChildren(...(policy.sources || []).map(source => { const link = document.createElement("a"); link.href = source.url; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = source.title; return link; }));
}

function render(data) {
  payload = data; $("updated").textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString("en-US")}`;
  if (!data.available) { $("empty-copy").textContent = data.error; return; }
  $("empty").classList.add("hidden"); $("dashboard").classList.remove("hidden");
  const stream = data.stream, report = data.report, badge = $("stream"), risk = report.risk || {}, model = report.probabilistic || {}, luna = report.luna || {};
  badge.textContent = stream.connected ? "STREAMING" : "STALE / STOPPED"; badge.className = `pill ${stream.connected ? "safe" : "danger"}`;
  $("equity").textContent = money(report.final_equity); $("pnl").textContent = `Net PnL ${money(report.net_pnl)}`;
  const positions = report.positions || (report.position ? [report.position] : []);
  $("position").textContent = positions.length ? `${positions.length} OPEN` : "FLAT";
  $("position-detail").textContent = positions.length ? positions.map(item => `${item.side.toUpperCase()} ${item.symbol} - ${item.leverage}x`).join(" | ") : "No exposure";
  $("risk").textContent = `${number(Number(risk.risk_per_trade || .0025) * 100)}% per trade`;
  $("limits").textContent = `${number(Number(risk.max_daily_loss || .015) * 100, 1)}% day - ${number(Number(risk.max_weekly_loss || .04) * 100, 0)}% week`;
  $("model").textContent = String(model.status || "collecting_data").replaceAll("_", " ").toUpperCase();
  $("model-note").textContent = model.can_trade ? "Bootstrap active; models shadow-only" : "Validated models required";
  $("luna").textContent = luna.ready ? "READY" : "FAIL CLOSED";
  $("luna-note").textContent = luna.policy ? `${luna.policy.regime} - ${luna.policy.action}` : (luna.reason || "Waiting for policy");
  readiness(stream, report); scanner(report.scanner || []); events("operations", report.operations || []); events("audit", report.audit || []);
  canvas("equity-chart", (report.equity_curve || []).map(point => Number(point.equity)), "#f3b74f");
  const select = $("symbol"), current = select.value, keys = Object.keys(stream.symbols || {});
  select.replaceChildren(...keys.map(key => { const option = document.createElement("option"); option.value = key; option.textContent = key; return option; }));
  if (keys.includes(current)) select.value = current; market();
}

async function refresh() {
  try { const response = await fetch("/api/meme", {cache: "no-store"}); if (!response.ok) throw Error("API unavailable"); render(await response.json()); }
  catch (error) { $("empty-copy").textContent = error.message; }
}

$("symbol").addEventListener("change", market);
window.addEventListener("resize", () => payload && render(payload));
refresh(); setInterval(refresh, 2000);
