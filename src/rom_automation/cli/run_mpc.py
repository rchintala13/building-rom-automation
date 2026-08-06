from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.mpc.closed_loop import run_mpc_closed_loop
from rom_automation.mpc.config_loader import load_mpc_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a closed-loop MPC of the identified 4R2C model acting on the "
            "live EnergyPlus building, from a YAML config."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the run_mpc YAML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_mpc_config(args.config)
    run_mpc_closed_loop(cfg=cfg)


if __name__ == "__main__":
    main()
