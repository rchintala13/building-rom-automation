from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.types import SimulationConfig


def select_edited_idf_files(config: SimulationConfig) -> list[Path]:
    """
    Select edited IDF files for EnergyPlus simulation.

    Returns
    -------
    list[Path]
        A list of edited IDF file paths. Even in single mode, this returns
        a one-element list so downstream workflows can stay uniform.
    """
    edited_root = config.paths.edited_idf_root
    mode = config.selection.mode

    if not edited_root.exists():
        raise FileNotFoundError(f"Edited IDF root does not exist: {edited_root}")

    if not edited_root.is_dir():
        raise NotADirectoryError(f"Edited IDF root is not a directory: {edited_root}")

    if mode == "single":
        return _select_single_file(config=config)

    if mode == "batch":
        return _select_batch_files(config=config)

    raise ValueError(f"Unsupported selection mode: {mode!r}")


def _select_single_file(config: SimulationConfig) -> list[Path]:
    edited_root = config.paths.edited_idf_root
    single_file = config.selection.single_file

    if not single_file:
        raise ValueError(
            "selection.single_file must be provided when selection.mode='single'"
        )

    input_path = edited_root / single_file

    if not input_path.exists():
        raise FileNotFoundError(f"Single edited IDF file not found: {input_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Single edited IDF path is not a file: {input_path}")

    if input_path.suffix.lower() != ".idf":
        raise ValueError(f"Selected file does not look like an IDF: {input_path}")

    return [input_path]


def _select_batch_files(config: SimulationConfig) -> list[Path]:
    edited_root = config.paths.edited_idf_root
    pattern = config.selection.pattern
    recursive = config.selection.recursive

    if recursive:
        files = sorted(path for path in edited_root.rglob(pattern) if path.is_file())
    else:
        files = sorted(path for path in edited_root.glob(pattern) if path.is_file())

    idf_files = [path for path in files if path.suffix.lower() == ".idf"]

    if not idf_files:
        raise FileNotFoundError(
            f"No edited IDF files found under {edited_root} with pattern {pattern!r}"
        )

    return idf_files