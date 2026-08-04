from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from rom_automation.mpc.config_types import MpcConfig
from rom_automation.mpc.idf_preparer import IdfPreparer
from rom_automation.mpc.plant import EnergyPlusPlant, PlantObservation


_CACHE_FILENAME = "hvac_capacity.json"
# How far below the comfort floor to pin the setpoint so the DX runs full-tilt.
_PIN_BELOW_C = 15.0


def calibrate_max_sensible_cooling_kw(
    cfg: MpcConfig,
    logger,
    use_cache: bool = True,
) -> float:
    """
    Measure the building's actual max sensible cooling capacity by running one
    full-tilt EnergyPlus pass: the principal HVAC's cooling setpoint is pinned far
    below comfort so the DX coil runs at its top speed, and the peak delivered
    `Zone Air System Sensible Cooling Rate` over the window is returned (kW).

    The result is cached per (window, timestep) under the output dir so repeat
    runs reuse it. Delete `hvac_capacity.json` to force a recompute.
    """
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / _CACHE_FILENAME

    cache_key = {
        "window_start": cfg.window.start.isoformat(),
        "window_end": cfg.window.end.isoformat(),
        "dt_minutes": cfg.control.dt_minutes,
    }

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("key") == cache_key:
            value = float(cached["max_sensible_cooling_kw"])
            logger.info(
                "Using cached HVAC max sensible cooling capacity: %.3f kW (%s)",
                value,
                cache_path,
            )
            return value

    logger.info("Calibrating HVAC max sensible cooling (full-tilt EnergyPlus pass)...")

    preparer = IdfPreparer(idd_path=cfg.energyplus.idd_path)
    cal_idf_path = output_dir / "calibration.idf"
    prepared = preparer.prepare_calibration_idf(cfg, run_idf_path=cal_idf_path)

    pinned_setpoint_c = cfg.comfort.lower_c - _PIN_BELOW_C
    window_start = pd.Timestamp(cfg.window.start)
    window_end = pd.Timestamp(cfg.window.end)
    peak = {"cooling_kw": 0.0}

    def step_fn(ts: datetime, obs: PlantObservation):
        # Track peak delivered cooling (hvac_power_kw is negative for cooling).
        cooling_kw = -obs.hvac_power_kw
        if cooling_kw > peak["cooling_kw"]:
            peak["cooling_kw"] = cooling_kw

        t = pd.Timestamp(ts)
        if window_start <= t < window_end:
            return pinned_setpoint_c  # force full-tilt cooling
        return None

    plant = EnergyPlusPlant(
        install_dir=cfg.energyplus.install_dir,
        weather_path=cfg.paths.weather,
        zone_name=prepared.zone_name,
        dt_seconds=cfg.control.dt_seconds,
        calendar_year=cfg.window.start.year,
        actuation_mode="supervisory_setpoint",
        setpoint_deadband_c=cfg.actuation.setpoint_deadband_c,
    )
    plant.run(
        run_idf_path=prepared.run_idf_path,
        output_dir=output_dir / "eplus_calibration",
        step_fn=step_fn,
    )

    value = float(peak["cooling_kw"])
    if value <= 0.0:
        raise RuntimeError(
            "Calibration measured no cooling. The principal HVAC may be disabled "
            "or the setpoint actuator did not resolve. Inspect the calibration IDF."
        )

    cache_path.write_text(
        json.dumps({"key": cache_key, "max_sensible_cooling_kw": value}, indent=2),
        encoding="utf-8",
    )
    logger.info("Calibrated HVAC max sensible cooling capacity: %.3f kW", value)
    return value
