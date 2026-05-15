from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.workflows.run_full_pipeline import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: edit IDFs, simulate, and process outputs."
    )

    parser.add_argument(
        "--edit-config",
        required=True,
        type=Path,
        help="Path to the IDF edit YAML config file.",
    )

    parser.add_argument(
        "--simulation-config",
        required=True,
        type=Path,
        help="Path to the EnergyPlus simulation YAML config file.",
    )

    parser.add_argument(
        "--processing-config",
        required=True,
        type=Path,
        help="Path to the processing YAML config file.",
    )

    parser.add_argument(
        "--idd",
        required=True,
        type=Path,
        help="Path to the EnergyPlus Energy+.idd file.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_full_pipeline(
        edit_config_path=args.edit_config,
        simulation_config_path=args.simulation_config,
        processing_config_path=args.processing_config,
        idd_path=args.idd,
    )


if __name__ == "__main__":
    main()
