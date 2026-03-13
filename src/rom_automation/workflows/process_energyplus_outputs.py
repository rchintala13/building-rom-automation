from __future__ import annotations

from pathlib import Path

import pandas as pd

from rom_automation.epconfig.loader import load_processing_config
from rom_automation.logging_utils import get_logger
from rom_automation.processing.column_mapper import rename_energyplus_columns
from rom_automation.processing.feature_builder import build_features
from rom_automation.processing.file_selector import select_simulation_csv_files
from rom_automation.processing.resampler import (
    detect_timestep_seconds,
    prepare_datetime_index,
    resample_processed_data,
    validate_timestep_seconds,
    write_processed_csv,
)


def run_process_energyplus_outputs_workflow(config_path: str | Path) -> None:
    """
    Process one or more EnergyPlus output CSV files into standardized
    5-minute and 1-hour training datasets using a YAML config.
    """
    config = load_processing_config(config_path)

    logger = get_logger(
        "rom_automation.process_energyplus_outputs",
        log_file=Path("logs") / "process_energyplus_outputs.log",
    )

    input_files = select_simulation_csv_files(config)
    logger.info("Found %d simulation CSV file(s) to process.", len(input_files))

    for raw_csv_path in input_files:
        output_dir = build_processed_output_dir(
            input_path=raw_csv_path,
            simulation_root=config.paths.simulation_output_root,
            processed_root=config.paths.processed_output_root,
            preserve_relative_structure=config.run_options.preserve_relative_structure,
        )

        output_5min = output_dir / config.output_files.file_5min
        output_1h = output_dir / config.output_files.file_1h

        if _should_skip_existing_outputs(
            output_5min=output_5min,
            output_1h=output_1h,
            overwrite=config.run_options.overwrite,
        ):
            logger.info("Skipping existing processed outputs for: %s", raw_csv_path)
            continue

        try:
            logger.info("Reading raw EnergyPlus CSV: %s", raw_csv_path)
            raw_df = pd.read_csv(raw_csv_path)

            logger.info("Renaming EnergyPlus columns.")
            mapped_df = rename_energyplus_columns(raw_df, strict=True)

            logger.info(
                "Preparing datetime index using calendar year=%s.",
                config.processing.calendar_year,
            )
            timed_df = prepare_datetime_index(
                mapped_df,
                year=config.processing.calendar_year,
                datetime_column=config.processing.datetime_column,
            )

            timestep_seconds = detect_timestep_seconds(timed_df)
            logger.info(
                "Detected timestep for %s: %s seconds",
                raw_csv_path,
                timestep_seconds,
            )

            if config.processing.expected_timestep_seconds is not None:
                validate_timestep_seconds(
                    timed_df,
                    expected_seconds=config.processing.expected_timestep_seconds,
                )
                logger.info(
                    "Validated timestep matches expected value: %s seconds",
                    config.processing.expected_timestep_seconds,
                )

            logger.info("Building processed features.")
            feature_df = build_features(
                timed_df,
                timestep_seconds=timestep_seconds,
            )

            logger.info("Creating 5-minute processed dataset.")
            df_5min = resample_processed_data(feature_df, freq="5min")

            logger.info("Creating 1-hour processed dataset.")
            df_1h = resample_processed_data(feature_df, freq="1h")

            logger.info("Writing processed 5-minute CSV: %s", output_5min)
            write_processed_csv(df_5min, output_5min)

            logger.info("Writing processed 1-hour CSV: %s", output_1h)
            write_processed_csv(df_1h, output_1h)

            logger.info("Finished processing: %s", raw_csv_path)

        except Exception as exc:
            logger.exception("Failed processing %s: %s", raw_csv_path, exc)


def build_processed_output_dir(
    input_path: Path,
    simulation_root: Path,
    processed_root: Path,
    preserve_relative_structure: bool,
) -> Path:
    """
    Build the output directory for processed CSVs.

    Example
    -------
    input_path:
        data/intermediate/simulation_outputs/cz5b/home_001__edited/eplusout.csv

    output_dir with preserve_relative_structure=True:
        data/processed/training_data/cz5b/home_001__edited
    """
    if preserve_relative_structure:
        relative_path = input_path.relative_to(simulation_root)
        return processed_root / relative_path.parent

    return processed_root / input_path.stem


def _should_skip_existing_outputs(
    output_5min: Path,
    output_1h: Path,
    overwrite: bool,
) -> bool:
    """
    Skip processing when both output files already exist and overwrite is False.
    """
    if overwrite:
        return False

    return output_5min.exists() and output_1h.exists()