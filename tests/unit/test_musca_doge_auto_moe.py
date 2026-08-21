from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_bot import musca_btc_auto_moe, musca_btc_moe
from adaptive_bot import musca_doge_data as doge_data
from adaptive_bot import musca_v5_microstructure as microstructure


def test_btc_frozen_protocol_is_unchanged_by_asset_generalization() -> None:
    assert (
        musca_btc_moe.PROTOCOL_HASH
        == "73563d1aed16e4f796d18d446dc9033946473429e48f18fe15c1a52ffc193ddd"
    )
    assert (
        musca_btc_auto_moe.PROTOCOL_HASH
        == "a195b75e41bf7bfd40dd2ca23103720cee1ea5dfbc7333cd34dcca4c7171a360"
    )


def test_doge_protocol_uses_own_paths_costs_and_btc_as_information() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "MUSCA_SYMBOL": "DOGEUSDT",
            "MUSCA_EXECUTION_RESERVE_ROUND_TRIP_BPS": "3.5",
        }
    )
    command = (
        "import json; "
        "from adaptive_bot import musca_btc_moe as b, musca_btc_auto_moe as a; "
        "print(json.dumps({'symbol':b.SYMBOL,'cost':b.ROUND_TRIP_COST_BPS,"
        "'features':list(b.FEATURES),'regime':list(b.REGIME_VIEW),"
        "'gating':list(b.GATING_CONTEXT),'gate_features':list(a.GATE_FEATURES),"
        "'root':str(a.ROOT),'protocol':a.PROTOCOL}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(completed.stdout)

    assert payload["symbol"] == "DOGEUSDT"
    assert payload["cost"] == 11.5
    assert payload["root"].endswith("musca_doge_auto_moe")
    assert "btc_return_5m_bps" in payload["features"]
    assert "oi_change_1h" not in payload["features"]
    assert "oi_change_1h" not in payload["regime"]
    assert "return_oi_interaction_raw" not in payload["regime"]
    assert "oi_change_1h" not in payload["gating"]
    assert "oi_change_1h" not in payload["gate_features"]
    assert payload["protocol"]["symbol"] == "DOGEUSDT"
    assert payload["protocol"]["real_capital_allowed"] is False
    assert "ETH" not in json.dumps(payload)


def test_doge_btc_context_is_causal_and_not_a_direction_veto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.date_range("2025-01-01", periods=300, freq="1min", tz="UTC")
    btc = pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_at": timestamps + pd.Timedelta(minutes=1),
            "perp_close": 100_000 + np.arange(len(timestamps), dtype=float),
        }
    )
    source = tmp_path / "btc.parquet"
    btc.to_parquet(source, index=False)
    monkeypatch.setattr(doge_data, "BTC_SOURCE", source)
    doge = pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_at": timestamps + pd.Timedelta(minutes=1),
            "perp_close": 0.2 + np.arange(len(timestamps), dtype=float) / 100_000,
        }
    )

    result = doge_data._btc_context(doge)

    assert result["btc_context_coverage"].all()
    assert result["btc_context_available_at"].le(result["available_at"]).all()
    assert result.loc[260:, "btc_beta_4h"].notna().all()
    assert set(result["direction_agreement_5m"].dropna().unique()).issubset(
        {-1.0, 0.0, 1.0}
    )


def test_official_timestamp_precisions_are_normalized_to_utc_nanoseconds() -> None:
    milliseconds = pd.Series(pd.to_datetime([1735689600000], unit="ms", utc=True))
    microseconds = pd.Series(
        pd.to_datetime(["2025-01-01T00:00:00.000001Z"], utc=True)
    )

    normalized_ms = doge_data._utc_ns(milliseconds)
    normalized_us = doge_data._utc_ns(microseconds)

    assert normalized_ms.dtype == normalized_us.dtype
    assert str(normalized_ms.dtype) == "datetime64[ns, UTC]"


def test_microstructure_aggregation_is_asset_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(microstructure, "ROOT", tmp_path)
    archive = tmp_path / "DOGEUSDT-aggTrades-2025-01.zip"
    csv = "1,0.2000,100,1,1,1735689600000,false\n2,0.2001,50,2,2,1735689601000,true\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("DOGEUSDT-aggTrades-2025-01.csv", csv)

    rows = microstructure._aggregate(
        archive,
        "2025-01",
        seconds=5,
        symbol="DOGEUSDT",
        status_path=tmp_path / "status.json",
    )

    assert len(rows) == 1
    assert rows.iloc[0]["signed_quote_volume"] > 0
    output = tmp_path / "DOGEUSDT-aggTrades-5s-2025-01.parquet"
    assert output.exists()
    normalized = musca_btc_moe._regularize_micro_buckets(pd.read_parquet(output))
    assert str(normalized["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert str(normalized["available_at"].dtype) == "datetime64[ns, UTC]"
