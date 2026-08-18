from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.sysid.config_loader import load_sysid_config
from rom_automation.workflows.run_sysid_ekf import run_sysid_ekf_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EKF-based 4R2C system identification from a YAML config."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to EKF sysid YAML config.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_sysid_config(args.config)
    run_sysid_ekf_workflow(config=config, config_path=args.config)


if __name__ == "__main__":
    main()
