## Workflows
# Edit idfs
python -m rom_automation.cli.edit_idfs --config configs/idf_edit/edit_idfs.yaml --idd "C:/EnergyPlusV24-2-0/Energy+.idd"

# Run EnergyPlus Simulation
python -m rom_automation.cli.run_epsimulations --config configs/simulation/run_single.yaml