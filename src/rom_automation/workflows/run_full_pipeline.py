from __future__ import annotations

from pathlib import Path

from rom_automation.epconfig.loader import (
    load_idf_edit_config,
    load_processing_config,
    load_simulation_config,
)
from rom_automation.workflows.edit_idfs import run_edit_idfs_workflow
from rom_automation.workflows.process_energyplus_outputs import (
    run_process_energyplus_outputs_workflow,
)
from rom_automation.workflows.run_epsimulations import run_simulations_workflow


def run_full_pipeline(
    edit_config_path: str | Path,
    simulation_config_path: str | Path,
    processing_config_path: str | Path,
    idd_path: str | Path,
) -> None:
    """
    Run the full pipeline: edit IDFs → simulate → process outputs.

    Validates that selection.mode and selection.city are consistent
    across all three configs before running any step.
    """
    edit_cfg = load_idf_edit_config(edit_config_path)
    sim_cfg = load_simulation_config(simulation_config_path)
    proc_cfg = load_processing_config(processing_config_path)

    _validate_selection_consistency(edit_cfg.selection, sim_cfg.selection, proc_cfg.selection)

    print("--- Step 1/3: Editing IDFs ---")
    run_edit_idfs_workflow(config_path=edit_config_path, idd_path=idd_path)

    print("--- Step 2/3: Running EnergyPlus simulations ---")
    run_simulations_workflow(config_path=simulation_config_path)

    print("--- Step 3/3: Processing simulation outputs ---")
    run_process_energyplus_outputs_workflow(config_path=processing_config_path)

    print("--- Pipeline complete ---")


def _validate_selection_consistency(edit_sel, sim_sel, proc_sel) -> None:
    modes = {
        "edit_idfs": edit_sel.mode,
        "run_epsimulations": sim_sel.mode,
        "process_energyplus_outputs": proc_sel.mode,
    }
    if len(set(modes.values())) > 1:
        raise ValueError(
            f"selection.mode must be the same across all three configs. Got: {modes}"
        )

    cities = {
        "edit_idfs": edit_sel.city,
        "run_epsimulations": sim_sel.city,
        "process_energyplus_outputs": proc_sel.city,
    }
    if len(set(cities.values())) > 1:
        raise ValueError(
            f"selection.city must be the same across all three configs. Got: {cities}"
        )
