from adaptive_bot.cli import main


def test_live_command_fails_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert main(["live"]) == 2

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "I_ACKNOWLEDGE_THE_RISK")
    assert main(["live"]) == 2
