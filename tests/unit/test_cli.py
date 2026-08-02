from adaptive_bot.cli import _parser, main


def test_meme_dashboard_accepts_container_bind_override() -> None:
    arguments = _parser().parse_args(
        [
            "meme-dashboard",
            "--config",
            "config.yaml",
            "--host",
            "0.0.0.0",
            "--allow-non-loopback",
        ]
    )
    assert arguments.host == "0.0.0.0"
    assert arguments.allow_non_loopback


def test_luna_max_can_be_refreshed_explicitly() -> None:
    arguments = _parser().parse_args(
        [
            "meme-luna-sidecar",
            "--config",
            "config.yaml",
            "--once",
            "--refresh-max",
        ]
    )
    assert arguments.once and arguments.refresh_max


def test_live_command_fails_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert main(["live"]) == 2

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "I_ACKNOWLEDGE_THE_RISK")
    assert main(["live"]) == 2
