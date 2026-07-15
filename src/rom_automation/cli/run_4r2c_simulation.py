from __future__ import annotations

import argparse
from pathlib import Path

from rom_automation.rom_sim.config_loader import load_simulation_config
from rom_automation.workflows.run_4r2c_simulation import (
    run_4r2c_simulation_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 4R2C simulation, one-step-ahead, or n-step-ahead prediction "
            "from a YAML config."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the run_4r2c_simulation YAML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_simulation_config(args.config)
    run_4r2c_simulation_workflow(cfg=cfg)


if __name__ == "__main__":
    main()
