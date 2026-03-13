from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.workflows.process_energyplus_outputs import (
    run_process_energyplus_outputs_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process one or more EnergyPlus output CSV files into "
            "standardized 5-minute and 1-hour training datasets."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the processing YAML config file.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_process_energyplus_outputs_workflow(config_path=args.config)


if __name__ == "__main__":
    main()