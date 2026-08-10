from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

ADVERSE_HORIZONS_SECONDS = (1, 5, 30)
HISTORY_ORDERS_ENDPOINT = "/api/v1/futures/trade/get_history_orders"
HISTORY_TRADES_ENDPOINT = "/api/v1/futures/trade/get_history_trades"
PRIVATE_ORDER_CHANNEL = "order"


def observed_execution_labels(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    midpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Build labels only from final private Bitunix order/trade observations."""
    required_orders = {
        "orderId",
        "clientId",
        "symbol",
        "qty",
        "tradeQty",
        "effect",
        "status",
        "ctime",
        "mtime",
    }
    required_trades = {"orderId", "qty", "price", "fee", "roleType", "ctime"}
    required_midpoints = {"exchange_timestamp", "midpoint"}
    if missing := required_orders - set(orders):
        raise ValueError(f"Bitunix observed orders missing columns: {sorted(missing)}")
    if missing := required_trades - set(trades):
        raise ValueError(f"Bitunix observed trades missing columns: {sorted(missing)}")
    if missing := required_midpoints - set(midpoints):
        raise ValueError(f"Bitunix observed midpoints missing columns: {sorted(missing)}")
    market = midpoints.copy()
    market["exchange_timestamp"] = pd.to_datetime(
        market["exchange_timestamp"], format="mixed", utc=True
    )
    market = market.sort_values("exchange_timestamp")
    rows: list[dict[str, Any]] = []
    for _, order in orders.loc[orders["effect"].astype(str).str.upper().eq("POST_ONLY")].iterrows():
        quantity = _decimal(order["qty"])
        filled = _decimal(order["tradeQty"])
        if quantity is None or filled is None or quantity <= 0 or filled < 0 or filled > quantity:
            continue
        created = _utc_milliseconds(order["ctime"])
        updated = _utc_milliseconds(order["mtime"])
        fills = trades.loc[trades["orderId"].astype(str).eq(str(order["orderId"]))].copy()
        fills["fill_timestamp"] = fills["ctime"].map(_utc_milliseconds)
        first_fill = fills["fill_timestamp"].min() if not fills.empty else pd.NaT
        weighted_price = _weighted_price(fills)
        roles = sorted(set(fills["roleType"].astype(str).str.upper())) if not fills.empty else []
        adverse = {
            f"adverse_selection_bps_{seconds}s": _observed_adverse(
                market, first_fill, weighted_price, str(order.get("side", "")), seconds
            )
            for seconds in ADVERSE_HORIZONS_SECONDS
        }
        fully_observed = bool(
            pd.notna(updated)
            and str(order["status"]).upper()
            in {"FILLED", "CANCELED", "PART_FILLED_CANCELED", "EXPIRED"}
        )
        rows.append(
            {
                "order_id": str(order["orderId"]),
                "client_order_id": str(order["clientId"]),
                "symbol": str(order["symbol"]),
                "created_at": created,
                "available_at": updated,
                "fill_probability_target": float(filled > 0),
                "fill_fraction_target": float(filled / quantity),
                "fill_latency_ms": (
                    (first_fill - created).total_seconds() * 1000
                    if pd.notna(first_fill) and pd.notna(created)
                    else None
                ),
                "fee": str(sum((_decimal(value) or Decimal(0)) for value in fills["fee"])),
                "liquidity_roles": roles,
                "maker_only": bool(roles) and set(roles) == {"MAKER"},
                "observation_complete": fully_observed,
                **adverse,
            }
        )
    return pd.DataFrame(rows)


def execution_observation_sources() -> dict[str, Any]:
    return {
        "private_websocket": PRIVATE_ORDER_CHANNEL,
        "rest_reconciliation": [HISTORY_ORDERS_ENDPOINT, HISTORY_TRADES_ENDPOINT],
        "adverse_horizons_seconds": list(ADVERSE_HORIZONS_SECONDS),
        "simulation_fallback": False,
    }


def _observed_adverse(
    midpoints: pd.DataFrame,
    filled_at: Any,
    fill_price: Decimal | None,
    side: str,
    seconds: int,
) -> float | None:
    if pd.isna(filled_at) or fill_price is None or fill_price <= 0:
        return None
    target = pd.Timestamp(filled_at) + pd.Timedelta(seconds=seconds)
    future = midpoints.loc[midpoints["exchange_timestamp"].ge(target)]
    if future.empty:
        return None
    midpoint = _decimal(future.iloc[0]["midpoint"])
    if midpoint is None:
        return None
    adverse = fill_price - midpoint if side.upper() == "BUY" else midpoint - fill_price
    return float(adverse / fill_price * Decimal(10_000))


def _weighted_price(fills: pd.DataFrame) -> Decimal | None:
    if fills.empty:
        return None
    pairs = [(_decimal(row["price"]), _decimal(row["qty"])) for _, row in fills.iterrows()]
    valid = [
        (price, qty) for price, qty in pairs if price is not None and qty is not None and qty > 0
    ]
    total = sum((qty for _, qty in valid), Decimal(0))
    return sum((price * qty for price, qty in valid), Decimal(0)) / total if total else None


def _decimal(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _utc_milliseconds(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(datetime.fromtimestamp(int(str(value)) / 1000, UTC))
    return timestamp
