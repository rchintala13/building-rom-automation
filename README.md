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

Runs EKF-based 4R2C system identification on a processed training CSV. For each candidate in the parameter grid, the workflow:

1. Splits the dataset into train / (optional) val / test segments (`dataset.splits`).
2. Runs a two-pass augmented-state EKF on training (the second pass re-warm-starts the wall temperatures — steady-state seed + anchored burn-in — using the parameters identified by the first pass; see `refine_initial_state`).
3. Freezes parameters at their end-of-train values and continues the filter on val and test for held-out evaluation.
4. Scores the candidate by RMSE on the validation segment (or test, if no val).

The config file specifies dataset splits, per-candidate P0 construction (`ekf.p0`, from temperature stds + per-parameter ratios), process-noise Q construction (`ekf.process_noise`, from heat-gain stds), per-parameter clip bounds (`parameter_bounds`), prediction horizon (`ekf.n_steps_ahead`), and the parameter grid.

Outputs are written to `data/processed/sysid_results/<city>/<house>/`:

| File | Contents |
|------|----------|
| `sysid_summary.json` | Best candidate, per-segment metrics, identified parameters |
| `model.json` | Identified 4R2C parameters plus the state-observer block (Q/R/dt/innovation std); consumed by the simulation and MPC workflows |
| `ekf_filtered_augmented_states.csv` | Filtered augmented state trajectory across all segments |
| `one_step_predictions.csv` | One-step predictions vs. measurements across all segments |
| `p_diagonals_last_candidate.csv` | EKF covariance diagonals per timestep for the last candidate |
| `q_diagonal_last_candidate.csv` | Process-noise covariance diagonal for the last candidate |

```bash
python -m rom_automation.cli.run_sysid_ekf \
    --config configs/sysid/run_ekf_sysid.yaml
```

### Experiment Tracking (MLflow)

When `tracking.enabled: true` in the sysid config, each `run_sysid_ekf` run is logged to MLflow so you can track methodological improvements over time. By default the store is local (metadata in a SQLite `mlflow.db`, artifacts under `mlruns/`); both are gitignored and should be backed up together.

```yaml
tracking:
  enabled: true
  experiment_name: "rom_sysid"
  tracking_uri: null            # null => local SQLite + ./mlruns; "file:./mlruns" for a plain-file store
```

Browse runs with `mlflow ui` (from the repo root) or query them via `mlflow.search_runs(experiment_names=["rom_sysid"])`. Each run logs:

**Artifacts** (plain files under `mlruns/<exp>/<run>/artifacts/`)

| Artifact | Contents |
|----------|----------|
| `model.json` | Identified 4R2C parameters + observer block (Q/R/dt/innovation std) |
| `sysid_summary.json` | Full run summary (per-segment metrics, warm-start init, best candidate) |
| `<config>.yaml` | The exact config file that produced the run (complete inputs, git-independent) |

**Params — model attributes** (`attr.*`; the methodology descriptors, so improvements are traceable)

| Param | Meaning |
|-------|---------|
| `attr.validation_evaluation` | Model evaluated over the entire validation dataset |
| `attr.wall_init_method` | Wall temps via steady-state seed + anchored burn-in |
| `attr.param_identification` | Augmented-state EKF |
| `attr.candidate_selection_metric` | Validation objective |
| `attr.observer_gain` | Innovation-consistent steady-state Kalman |
| `attr.n_step_horizon` | n-step prediction horizon |

**Params — config inputs**

| Param | Meaning |
|-------|---------|
| `city`, `house_name` | Building selection |
| `history_hours` | Warm-start / burn-in window |
| `split.train`, `split.val`, `split.test` | Dataset split fractions |
| `ekf.r_value`, `ekf.n_steps_ahead` | Measurement noise, n-step horizon |
| `ekf.w_one_step`, `ekf.w_n_step` | Selection-objective weights |
| `process_noise.q_in_std_kw`, `…q_iw_std_kw`, `…q_ow_std_kw` | Process-noise stds |
| `n_candidates_evaluated`, `selection_segment` | Grid size, segment used for selection |

**Metrics**

| Metric | Meaning |
|--------|---------|
| `rmse_one_step_{train,val,test}` | One-step prediction RMSE per segment |
| `rmse_n_step_{train,val,test}` | n-step prediction RMSE per segment |
| `selection_objective` | Weighted validation objective used to pick the candidate |
| `param.<name>` (8) | Identified parameter values (r_in_iw … alpha_ghi_inner_wall) |
| `observer.innovation_std_c`, `observer.r` | Observer ingredients |

**Tags**

| Tag | Meaning |
|-----|---------|
| `run_type` | `sysid` |
| `city`, `house_name` | Building selection (filterable) |
| `wall_init_method`, `validation_evaluation` | Mirror the model attributes for easy filtering |
| `git_commit`, `git_dirty` | Code version, and whether the working tree had uncommitted changes |

### Run 4R2C Simulation / Prediction

Runs the identified 4R2C model against real or custom input drivers, in one of three modes. Consumes the `model.json` produced by the sysid workflow.

| Mode (`mode.kind`) | Output columns |
|--------------------|---------------|
| `simulation` | `timestamp, t_in_pred_c, t_iw_pred_c, t_ow_pred_c` — open-loop rollout |
| `one_step` | `timestamp, t_in_measured_c, t_in_pred_one_step_c` — reset T_in each step, predict k+1 |
| `n_step` | `timestamp, t_in_measured_c, t_in_pred_n_step_<N>_c` — reset T_in each step, predict k+N |

The config's `inputs.source` selects between `processed` (reads `processed_5min.csv` for the selected city/house) or `custom` (any CSV with the same schema — must include `T_zone_C` for the one-step and n-step modes). The `window` section slices the input CSV into `[start, start + duration_hours)`, and `history_hours` of prior data is used to warm-start the wall temperatures (steady-state seed + anchored burn-in) using the identified parameters — no alphas or `wall_init` section are needed.

Outputs are written to `data/processed/sim_results/<city>/<house>/<mode>_<YYYYMMDDTHHMM>.csv` alongside a `simulation_summary.json` capturing the run metadata, the warm-start wall initialization (steady-state seed), and model parameters used.

```bash
python -m rom_automation.cli.run_4r2c_simulation \
    --config configs/simulation/run_4r2c_simulation.yaml
```
