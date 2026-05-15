from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rom_automation.workflows.run_sysid_ekf import run_sysid_ekf_workflow

_EDITED_SUFFIX = "__edited"
_PROCESSED_FILENAME = "processed_5min.csv"


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


def _resolve_and_validate_paths(cfg: dict) -> tuple[Path, Path]:
    raw_idf_root = Path(cfg["paths"]["raw_idf_root"])
    processed_root = Path(cfg["paths"]["processed_root"])
    output_root = Path(cfg["paths"]["output_root"])
    city = cfg["selection"]["city"]
    house_name = cfg["selection"]["house_name"]

    city_dir = raw_idf_root / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City not found in raw IDF root: {city_dir}")

    house_idf = city_dir / f"{house_name}.idf"
    if not house_idf.exists():
        raise FileNotFoundError(f"House IDF not found: {house_idf}")

    processed_csv = processed_root / city / f"{house_name}{_EDITED_SUFFIX}" / _PROCESSED_FILENAME
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {processed_csv}\n"
            "Run the processing workflow first."
        )

    output_dir = output_root / city / house_name
    return processed_csv, output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_csv_path, output_dir = _resolve_and_validate_paths(cfg)

    run_sysid_ekf_workflow(
        processed_csv_path=processed_csv_path,
        output_dir=output_dir,
        history_hours=float(cfg["dataset"]["history_hours"]),
        parameter_grid_dict=cfg["parameter_grid"],
        q_diag=cfg["ekf"]["q_diag"],
        r_value=float(cfg["ekf"]["r_value"]),
        p0_diag=cfg["ekf"]["p0_diag"],
        n_steps_ahead=int(cfg["ekf"]["n_steps_ahead"]),
        objective_weights=(
            float(cfg["ekf"]["objective_weights"]["one_step"]),
            float(cfg["ekf"]["objective_weights"]["n_step"]),
        ),
    )


if __name__ == "__main__":
    main()