from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.types import ProcessingConfig


def select_simulation_csv_files(config: ProcessingConfig) -> list[Path]:
    """
    Select raw EnergyPlus simulation CSV files for post-processing.

    Returns
    -------
    list[Path]
        A list of CSV file paths. Even in single mode, this returns
        a one-element list so downstream workflows can stay uniform.
    """
    simulation_root = config.paths.simulation_output_root
    mode = config.selection.mode

    if not simulation_root.exists():
        raise FileNotFoundError(
            f"Simulation output root does not exist: {simulation_root}"
        )

    if not simulation_root.is_dir():
        raise NotADirectoryError(
            f"Simulation output root is not a directory: {simulation_root}"
        )

    if mode == "single":
        return _select_single_file(config=config)

    if mode == "batch":
        return _select_batch_files(config=config)

    raise ValueError(f"Unsupported selection mode: {mode!r}")


def _select_single_file(config: ProcessingConfig) -> list[Path]:
    simulation_root = config.paths.simulation_output_root
    single_file = config.selection.single_file

    if not single_file:
        raise ValueError(
            "selection.single_file must be provided when selection.mode='single'"
        )

    input_path = simulation_root / single_file

    if not input_path.exists():
        raise FileNotFoundError(f"Single simulation CSV not found: {input_path}")

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Single simulation CSV path is not a file: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Selected file is not a CSV: {input_path}")

    return [input_path]


def _select_batch_files(config: ProcessingConfig) -> list[Path]:
    simulation_root = config.paths.simulation_output_root
    pattern = config.selection.pattern
    recursive = config.selection.recursive

    if recursive:
        files = sorted(
            path for path in simulation_root.rglob(pattern) if path.is_file()
        )
    else:
        files = sorted(
            path for path in simulation_root.glob(pattern) if path.is_file()
        )

    csv_files = [path for path in files if path.suffix.lower() == ".csv"]

    if not csv_files:
        raise FileNotFoundError(
            f"No simulation CSV files found under {simulation_root} "
            f"with pattern {pattern!r}"
        )

    return csv_files