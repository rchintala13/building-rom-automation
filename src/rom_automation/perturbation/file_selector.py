from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.types import IDFEditConfig


def select_idf_files(config: IDFEditConfig) -> list[Path]:
    """
    Select IDF files based on the config selection mode.

    Returns
    -------
    list[Path]
        A list of input IDF file paths. Even in single mode, this returns
        a one-element list so downstream workflows can stay uniform.
    """
    raw_root = config.paths.raw_idf_root
    mode = config.selection.mode

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw IDF root does not exist: {raw_root}")

    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw IDF root is not a directory: {raw_root}")

    if mode == "single":
        return _select_single_file(config=config)

    if mode == "batch":
        return _select_batch_files(config=config)

    raise ValueError(f"Unsupported selection mode: {mode!r}")


def _select_single_file(config: IDFEditConfig) -> list[Path]:
    raw_root = config.paths.raw_idf_root
    single_file = config.selection.single_file

    if not single_file:
        raise ValueError(
            "selection.single_file must be provided when selection.mode='single'"
        )

    input_path = raw_root / single_file

    if not input_path.exists():
        raise FileNotFoundError(f"Single IDF file not found: {input_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Single IDF path is not a file: {input_path}")

    if input_path.suffix.lower() != ".idf":
        raise ValueError(f"Selected file does not look like an IDF: {input_path}")

    return [input_path]


def _select_batch_files(config: IDFEditConfig) -> list[Path]:
    raw_root = config.paths.raw_idf_root
    pattern = config.selection.pattern
    recursive = config.selection.recursive
    city = config.selection.city

    search_root = raw_root / city if city else raw_root

    if not search_root.exists():
        raise FileNotFoundError(f"City directory not found: {search_root}")

    if recursive:
        files = sorted(path for path in search_root.rglob(pattern) if path.is_file())
    else:
        files = sorted(path for path in search_root.glob(pattern) if path.is_file())

    idf_files = [path for path in files if path.suffix.lower() == ".idf"]

    if not idf_files:
        raise FileNotFoundError(
            f"No IDF files found under {search_root} with pattern {pattern!r}"
        )

    return idf_files