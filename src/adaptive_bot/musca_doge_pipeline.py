from __future__ import annotations

import argparse
import json

from adaptive_bot.musca_doge_data import _status, build


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOGEUSDT official data preparation and Auto-MoE training"
    )
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    arguments = parser.parse_args()
    try:
        data_audit = build(force=arguments.force_data)

        # Import after preparation so the DOGE asset contract is installed first.
        from adaptive_bot.musca_doge_auto_moe import train

        report = train(force=arguments.force_training)
    except Exception as error:
        _status("failed", str(error), 0)
        raise
    print(json.dumps({"data": data_audit, "training": report}, indent=2, default=str))


if __name__ == "__main__":
    main()
