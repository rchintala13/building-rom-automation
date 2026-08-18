from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.types import SimulationConfig
from rom_automation.simulation.eplus_runner import EnergyPlusRunner
from rom_automation.simulation.file_selector import select_edited_idf_files


def run_simulations_workflow(config: SimulationConfig) -> None:
    """
    Run EnergyPlus simulations for one or more edited IDF files.

    Steps
    -----
    1. Select one or more edited IDF files
    2. Build output directory for each IDF
    3. Run EnergyPlus
    """
    input_files = select_edited_idf_files(config)
    runner = EnergyPlusRunner(eplus_exe=config.simulation.eplus_exe)

    print(f"Found {len(input_files)} edited IDF file(s) to simulate.")

    for input_path in input_files:
        output_dir = build_simulation_output_dir(
            input_path=input_path,
            edited_root=config.paths.edited_idf_root,
            output_root=config.paths.output_root,
            preserve_relative_structure=config.run_options.preserve_relative_structure,
        )

        if _should_skip_existing_output(
            output_dir=output_dir,
            overwrite=config.run_options.overwrite,
        ):
            print(f"Skipping existing simulation output: {output_dir}")
            continue

        print(f"Running EnergyPlus for: {input_path}")
        runner.run_idf(
            idf_path=input_path,
            weather_path=config.simulation.weather_file,
            output_dir=output_dir,
        )
        print(f"Simulation complete: {output_dir}")


def build_simulation_output_dir(
    input_path: Path,
    edited_root: Path,
    output_root: Path,
    preserve_relative_structure: bool,
) -> Path:
    """
    Build the output directory for a simulation run.

    If preserve_relative_structure is True:
        edited_root/cz5b/home_001__edited.idf
    becomes:
        output_root/cz5b/home_001__edited/

    If preserve_relative_structure is False:
        edited_root/cz5b/home_001__edited.idf
    becomes:
        output_root/home_001__edited/
    """
    if preserve_relative_structure:
        relative_path = input_path.relative_to(edited_root)
        return output_root / relative_path.parent / relative_path.stem

    return output_root / input_path.stem


def _should_skip_existing_output(
    output_dir: Path,
    overwrite: bool,
) -> bool:
    """
    Decide whether to skip a simulation because output already exists.
    """
    if overwrite:
        return False

    if not output_dir.exists():
        return False

    if not output_dir.is_dir():
        raise NotADirectoryError(f"Simulation output path is not a directory: {output_dir}")

    return any(output_dir.iterdir())