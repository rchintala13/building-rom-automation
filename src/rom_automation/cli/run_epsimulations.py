from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.workflows.run_epsimulations import run_simulations_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EnergyPlus simulations for one or more edited IDF files."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the simulation YAML config file.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_simulations_workflow(config_path=args.config)


if __name__ == "__main__":
    main()