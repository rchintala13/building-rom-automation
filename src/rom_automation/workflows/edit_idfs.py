from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.loader import load_idf_edit_config
from rom_automation.perturbation.file_selector import select_idf_files
from rom_automation.perturbation.idf_editor import IDFEditor


def run_edit_idfs_workflow(
    config_path: str | Path,
    idd_path: str | Path,
) -> None:
    """
    Run the IDF editing workflow.

    Steps
    -----
    1. Load YAML config
    2. Select one or more input IDF files
    3. Build output paths for edited IDFs
    4. Apply edits and save outputs
    """
    config = load_idf_edit_config(config_path)
    input_files = select_idf_files(config)

    editor = IDFEditor(idd_path=idd_path)

    print(f"Found {len(input_files)} IDF file(s) to edit.")

    for input_path in input_files:
        output_path = build_output_path(
            input_path=input_path,
            raw_root=config.paths.raw_idf_root,
            output_root=config.paths.output_root,
            suffix=config.output.suffix,
        )

        if output_path.exists() and not config.output.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        print(f"Editing: {input_path}")
        editor.edit_idf(
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        print(f"Saved: {output_path}")


def build_output_path(
    input_path: Path,
    raw_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    """
    Build the output path for an edited IDF while preserving relative
    directory structure under the raw root.

    Example
    -------
    input_path:
        data/raw/resstock_idf/group1/home_a.idf

    output_path:
        data/intermediate/perturbed_idf/group1/home_a__edited.idf
    """
    relative_path = input_path.relative_to(raw_root)
    output_filename = f"{relative_path.stem}{suffix}{relative_path.suffix}"
    return output_root / relative_path.parent / output_filename