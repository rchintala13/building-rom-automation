## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate building_rom_automation
```

The source code lives under `src/` (src-layout), so `rom_automation` is not on Python's path by default. Install the package in editable mode once to make it importable:

```bash
pip install -e .
```

This only needs to be done once per environment. After that, all CLI commands are run as Python modules from the project root:

```bash
python -m rom_automation.cli.<script_name> --config <path_to_config>
```

## Workflows

### Full Pipeline From Raw IDFs to Processed Training Data for EKF

Runs all three steps in sequence: edit IDFs → simulate → process outputs.
Before starting, it validates that `selection.mode` and `selection.city` are identical across all three config files, and raises an error if they differ.

```bash
python -m rom_automation.cli.run_full_pipeline \
    --edit-config configs/idf_edit/edit_idfs.yaml \
    --simulation-config configs/simulation/run_batch.yaml \
    --processing-config configs/processing/process_energyplus_outputs_batch.yaml \
    --idd "C:/EnergyPlusV24-2-0/Energy+.idd"
```

### Edit IDFs

Instead of using the default parameters of the IDF, automate the editing process.
The following objects are edited using a config file:

- Run Period
- Timestep
- Setpoints (only `Schedule:Day:Interval` objects are currently supported)
- Schedule file paths
- Output variables

Edited IDFs are written to `data/intermediate/perturbed_idfs/`, mirroring the input directory structure from `data/raw/resstock_idfs/` with an `__edited` suffix (e.g. `home_a.idf` → `home_a__edited.idf`).

Two modes are supported — use the corresponding config file:

| Mode | Config |
|------|--------|
| Single IDF | `configs/idf_edit/edit_single_idf.yaml` |
| Batch (all IDFs) | `configs/idf_edit/edit_idfs.yaml` |

```bash
python -m rom_automation.cli.edit_idfs \
    --config configs/idf_edit/edit_single_idf.yaml \
    --idd "C:/EnergyPlusV24-2-0/Energy+.idd"
```

### Run EnergyPlus Simulation

Runs EnergyPlus on edited IDFs. Simulation outputs are written to `data/intermediate/simulation_outputs/`, preserving the input directory structure from `data/intermediate/perturbed_idfs/`.

Two modes are supported — use the corresponding config file:

| Mode | Config |
|------|--------|
| Single IDF | `configs/simulation/run_single.yaml` |
| Batch (all IDFs) | `configs/simulation/run_batch.yaml` |

Set `city` in the config to restrict a batch run to one city subfolder (e.g. `Denver`), or leave it `null` to run all cities.

```bash
python -m rom_automation.cli.run_epsimulations \
    --config configs/simulation/run_single.yaml
```

### Process EnergyPlus Simulation Data for ROM

Processes raw EnergyPlus CSV outputs into resampled training data. Processed files are written to `data/processed/training_data/`, preserving the input directory structure from `data/intermediate/simulation_outputs/`.

Two modes are supported — use the corresponding config file:

| Mode | Config |
|------|--------|
| Single file | `configs/processing/process_energyplus_outputs_single_file.yaml` |
| Batch (all files) | `configs/processing/process_energyplus_outputs_batch.yaml` |

Set `city` in the config to restrict a batch run to one city subfolder (e.g. `Denver`), or leave it `null` to process all cities.

```bash
python -m rom_automation.cli.process_energyplus_outputs \
    --config configs/processing/process_energyplus_outputs_batch.yaml
```

### Run EKF System Identification

Runs EKF-based 4R2C system identification on a processed training CSV. The EKF is run over a parameter grid; the best-fit thermal parameters (R, C, solar gain coefficients) are saved to `output_dir` along with evaluation metrics.

The config file specifies the input CSV path, output directory, EKF tuning parameters (`q_diag`, `r_value`, `p0_diag`), prediction horizon (`n_steps_ahead`), and the parameter grid to search over.

```bash
python -m rom_automation.cli.run_sysid_ekf \
    --config configs/sysid/run_ekf_sysid.yaml
```
