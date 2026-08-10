import json

import pytest

from adaptive_bot.cli import _meme_actionable_symbols, _parser, main


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


def test_scientific_ml_commands_expose_resume_watch_and_holdout_lock() -> None:
    research = _parser().parse_args(["ml-research", "--config", "config.yaml", "--resume"])
    policy = _parser().parse_args(["ml-policy-research", "--config", "config.yaml"])
    status = _parser().parse_args(
        ["ml-status", "--config", "config.yaml", "--watch", "--interval", "2"]
    )
    finalize = _parser().parse_args(
        [
            "ml-finalize",
            "--config",
            "config.yaml",
            "--run-id",
            "run-1",
            "--open-holdout",
        ]
    )
    assert research.resume
    assert policy.command == "ml-policy-research"
    assert status.watch and status.interval == 2
    assert finalize.open_holdout and finalize.run_id == "run-1"

    expert = _parser().parse_args(["ml-expert-train", "--config", "config.yaml", "--resume"])
    expert_status = _parser().parse_args(["ml-expert-status", "--config", "config.yaml", "--watch"])
    expert_finalize = _parser().parse_args(
        [
            "ml-expert-finalize",
            "--config",
            "config.yaml",
            "--run-id",
            "expert-1",
            "--open-holdout",
        ]
    )
    assert expert.resume and expert_status.watch
    assert expert_finalize.open_holdout and expert_finalize.run_id == "expert-1"

    v6 = _parser().parse_args(["ml-expert-train", "--config", "config.yaml", "--protocol", "v6"])
    microstructure = _parser().parse_args(
        [
            "collect-bitunix-microstructure",
            "--config",
            "config.yaml",
            "--symbol",
            "ETHUSDT",
        ]
    )
    assert v6.protocol == "v6"
    assert microstructure.symbol == "ETHUSDT"

    v14 = _parser().parse_args(
        ["ml-hybrid-v14-forward-status", "--config", "config.yaml", "--watch"]
    )
    assert v14.command == "ml-hybrid-v14-forward-status"
    assert v14.watch and v14.interval == 3600.0

    one_minute = _parser().parse_args(
        [
            "collect-bitunix",
            "--config",
            "config.yaml",
            "--timeframe-minutes",
            "1",
            "--price-type",
            "MARK_PRICE",
        ]
    )
    assert one_minute.timeframe_minutes == 1
    assert one_minute.price_type == "MARK_PRICE"

    hybrid = _parser().parse_args(
        [
            "ml-hybrid-init",
            "--config",
            "config.yaml",
            "--start",
            "2024-01-01T00:00:00Z",
            "--end",
            "2026-01-01T00:00:00Z",
        ]
    )
    assert hybrid.start.tzinfo is not None
    assert _parser().parse_args(["ml-hybrid-train", "--config", "config.yaml"]).command == (
        "ml-hybrid-train"
    )


def test_closed_ml_output_pipe_does_not_mark_completed_training_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_pipe(_arguments: object) -> int:
        raise BrokenPipeError

    monkeypatch.setattr("adaptive_bot.cli._ml_research", broken_pipe)
    assert main(["ml-research", "--config", "config.yaml"]) == 0


def test_reduced_and_full_eligibility_trigger_luna_max_refresh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    report = tmp_path / "paper.json"
    report.write_text(
        json.dumps(
            {
                "scanner": [
                    {"symbol": "DOGEUSDT", "status": "ELIGIBLE_REDUCED"},
                    {"symbol": "PEPEUSDT", "status": "ELIGIBLE"},
                    {"symbol": "WIFUSDT", "status": "BLOCKED"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _meme_actionable_symbols(report) == {"DOGEUSDT", "PEPEUSDT"}


def test_live_command_fails_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert main(["live"]) == 2

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "I_ACKNOWLEDGE_THE_RISK")
    assert main(["live"]) == 2
