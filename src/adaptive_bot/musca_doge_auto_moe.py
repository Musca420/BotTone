from __future__ import annotations

import argparse
import json
import os
from typing import Any


def train(*, force: bool = False) -> dict[str, Any]:
    os.environ["MUSCA_SYMBOL"] = "DOGEUSDT"
    os.environ["MUSCA_EXECUTION_RESERVE_ROUND_TRIP_BPS"] = "3.5"
    from adaptive_bot.musca_btc_auto_moe import train as train_asset

    return train_asset(force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOGEUSDT automatic two-stage expert training"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(train(force=arguments.force), indent=2, default=str))


if __name__ == "__main__":
    main()
