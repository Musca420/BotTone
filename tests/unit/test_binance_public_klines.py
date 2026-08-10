import io
import zipfile

from adaptive_bot.binance_public_klines import _months, _parse


def test_months_and_official_kline_columns() -> None:
    csv = b"1711929600000,1,2,0.5,1.5,10,1711929659999,15,7,6,9,0\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1m.csv", csv)
    frame = _parse(buffer.getvalue())
    assert _months("2024-04", "2024-06") == ["2024-04", "2024-05", "2024-06"]
    assert frame.loc[0, "trade_count"] == 7
    assert frame.loc[0, "taker_sell_volume"] == 4
    assert frame.loc[0, "taker_imbalance"] == 0.2
