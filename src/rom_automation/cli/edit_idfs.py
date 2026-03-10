from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.workflows.edit_idfs import run_edit_idfs_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit one or more EnergyPlus IDF files from a YAML config."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the IDF edit YAML config file.",
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

    run_edit_idfs_workflow(
        config_path=args.config,
        idd_path=args.idd,
    )


if __name__ == "__main__":
    main()