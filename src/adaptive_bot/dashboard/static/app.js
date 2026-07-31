const $ = (id) => document.getElementById(id);

const money = (value) => {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(number)
    : "—";
};
const number = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
const dateTime = (value) => value ? new Date(value).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" }) : "—";

function setText(id, value) { $(id).textContent = value; }

async function refresh() {
  try {
    const response = await fetch("/api/dashboard", { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("empty-state").classList.remove("hidden");
    $("dashboard").classList.add("hidden");
    setText("empty-message", `Dashboard connection failed: ${error.message}`);
  }
}

function render(data) {
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
  setText("fill-count", `${summary.fills} fill${summary.fills === 1 ? "" : "s"}`);

  renderLatest(latest);
  renderRisk(summary, data.safety);
  renderFills(data.fills);
  renderTimeline(data.telemetry);
  renderMilestones(data.milestones);
  drawEquity(data.equity_curve);
  drawRange(data.telemetry);
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
  setText("risk-per-trade", percent(summary.risk_per_trade));
  const active = Number(summary.kill_switches) > 0;
  const badge = $("kill-status");
  badge.textContent = active ? "TRIGGERED" : "CLEAR";
  badge.className = active ? "pill danger" : "pill safe-pill";
}

function renderFills(fills) {
  const body = $("fills-body");
  if (!fills.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-cell">No fills recorded.</td></tr>';
    return;
  }
  body.replaceChildren(...fills.map((fill) => {
    const row = document.createElement("tr");
    const values = [
      dateTime(fill.exchange_timestamp),
      String(fill.side).toUpperCase(),
      number(fill.quantity, 2),
      money(fill.price),
      money(fill.commission),
      money(fill.slippage),
      fill.client_order_id,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1) cell.className = fill.side === "buy" ? "side-buy" : "side-sell";
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
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) return;
    if (index === 0) context.moveTo(x(index), y(value)); else context.lineTo(x(index), y(value));
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

window.addEventListener("resize", () => refresh());
refresh();
setInterval(refresh, 2000);
