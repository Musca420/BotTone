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
let liveCandles = [];
let liveTicks = [];
let socket;
let pingTimer;

function setText(id, value) { $(id).textContent = value; }

async function refresh() {
  try {
    const [response, liveResponse] = await Promise.all([
      fetch("/api/dashboard", { cache: "no-store" }),
      fetch("/api/live", { cache: "no-store" }),
    ]);
    if (!response.ok || !liveResponse.ok) throw new Error("Dashboard API is unavailable");
    render(await response.json(), await liveResponse.json());
  } catch (error) {
    $("empty-state").classList.remove("hidden");
    $("dashboard").classList.add("hidden");
    setText("empty-message", `Dashboard connection failed: ${error.message}`);
  }
}

function render(data, live) {
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
  setText("mode", String(summary.mode).toUpperCase());
  setText("instrument", summary.instrument);
  setText("timeframe", `${summary.timeframe_minutes} minutes`);
  setText("latest-bar", dateTime(latest.timestamp));
  setText("equity", money(summary.final_equity));
  setText("equity-change", `Started at ${money(summary.initial_equity)}`);
  setText("net-pnl", money(summary.net_pnl));
  $("net-pnl").className = Number(summary.net_pnl) >= 0 ? "safe" : "negative";
  setText("drawdown", percent(summary.max_drawdown));
  setText("signals", summary.signals);
  setText("rejected", `${summary.rejected_signals} rejected by risk controls`);
  setText("costs", money(Number(summary.fees) + Number(summary.slippage)));
  setText("fill-count", `${summary.operations} operation${summary.operations === 1 ? "" : "s"}`);
  setText("position-status", data.current_position.status);
  setText("position-quantity", `Quantity ${number(data.current_position.quantity, 6)}`);

  renderLive(live);
  const botLatest = summary.instrument === "BTCUSDT"
    ? latest
    : live.available ? { ...live.latest, activity: live.activity } : latest;
  renderLatest(botLatest);
  renderRisk(summary, data.safety);
  renderOperations(data.operations, data.no_trade_reason);
  renderTimeline(data.telemetry);
  renderMilestones(data.milestones);
  drawEquity(data.equity_curve);
  drawRange(live.available ? live.candles : data.telemetry);
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
  setText("activity", latest.activity || "No decision is available.");
  setText("calc-close", money(latest.close));
  setText("calc-center", money(latest.center));
  setText("calc-atr", number(latest.atr, 4));
  setText("calc-adx", number(latest.adx, 2));
  setText("calc-z", number(latest.z_score, 3));
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
  const active = Number(summary.kill_switches) > 0;
  const badge = $("kill-status");
  badge.textContent = active ? "TRIGGERED" : "CLEAR";
  badge.className = active ? "pill danger" : "pill safe-pill";
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
  const center = points.map((point) => point.center == null ? null : Number(point.center));
  const lower = points.map((point) => point.lower_band == null ? null : Number(point.lower_band));
  const upper = points.map((point) => point.upper_band == null ? null : Number(point.upper_band));
  const bounds = boundsOf([close, center, lower, upper]);
  drawSeries(context, lower, bounds, width, height, "rgba(243,183,79,.6)", 1);
  drawSeries(context, upper, bounds, width, height, "rgba(243,183,79,.6)", 1);
  drawSeries(context, center, bounds, width, height, "#4fd1c5", 1.5);
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

function connectBitunix() {
  const badge = $("ws-status");
  badge.textContent = "CONNECTING";
  badge.className = "pill";
  socket = new WebSocket("wss://fapi.bitunix.com/public/");
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ op: "subscribe", args: [
      { symbol: "BTCUSDT", ch: "mark_kline_5min" },
      { symbol: "BTCUSDT", ch: "price" },
    ] }));
    pingTimer = setInterval(() => socket.send(JSON.stringify({ op: "ping", ping: Math.floor(Date.now() / 1000) })), 15000);
  });
  socket.addEventListener("message", ({ data }) => {
    let message;
    try { message = JSON.parse(data); } catch { return; }
    if (message.symbol !== "BTCUSDT" || !message.data) return;
    badge.textContent = "STREAMING";
    badge.className = "pill safe-pill";
    if (message.ch === "mark_kline_5min") {
      const candleTime = Math.floor(Number(message.ts) / 300000) * 300000;
      const candle = { timestamp: new Date(candleTime).toISOString(), open: message.data.o, high: message.data.h, low: message.data.l, close: message.data.c };
      liveCandles = liveCandles.at(-1)?.timestamp === candle.timestamp
        ? [...liveCandles.slice(0, -1), candle]
        : [...liveCandles.slice(-59), candle];
      drawCandlesticks(liveCandles);
    }
    if (message.ch === "price") {
      const mark = Number(message.data.mp);
      if (!Number.isFinite(mark)) return;
      liveTicks = [...liveTicks.slice(-199), mark];
      setText("realtime-price", money(mark));
      setText("realtime-time", `Updated ${new Date(message.ts).toLocaleTimeString("en-US")}`);
      setText("index-price", money(message.data.ip));
      setText("funding-rate", percent(message.data.fr));
      drawRealtime();
    }
  });
  socket.addEventListener("close", () => {
    clearInterval(pingTimer);
    badge.textContent = "RECONNECTING";
    badge.className = "pill danger";
    setTimeout(connectBitunix, 3000);
  });
  socket.addEventListener("error", () => socket.close());
}

window.addEventListener("resize", () => refresh());
refresh();
setInterval(refresh, 2000);
connectBitunix();
