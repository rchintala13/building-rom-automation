from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rom_automation.logging_utils import get_logger
from rom_automation.mpc.calibration import calibrate_max_sensible_cooling_kw
from rom_automation.mpc.config_types import MpcConfig
from rom_automation.mpc.controller import MpcController
from rom_automation.mpc.disturbance import DisturbanceProvider
from rom_automation.mpc.idf_preparer import IdfPreparer, PreparedIdf
from rom_automation.mpc.plant import EnergyPlusPlant, PlantObservation
from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.state_estimator import (
    innovation_consistent_gain,
    steady_state_kalman_gain,
)
from rom_automation.models.types import FourR2CInput, FourR2CParameters, FourR2CState
from rom_automation.models.warm_start import warm_start_walls
from rom_automation.rom_sim.runner import FourR2CRunner
from rom_automation.sysid.dataset_adapter import REQUIRED_COLUMNS, build_input_sequence


_PROCESSED_FILENAME = "processed_5min.csv"
_EDITED_SUFFIX = "__edited"

_MODEL_PARAM_KEYS = (
    "r_in_iw",
    "r_iw_ow",
    "r_ow_oa",
    "r_in_oa",
    "c_in",
    "c_w",
    "alpha_ghi_outer_wall",
    "alpha_ghi_inner_wall",
)


def run_mpc_closed_loop(cfg: MpcConfig) -> None:
    """
    Run a closed-loop MPC simulation: the LP MPC (using the identified 4R2C model)
    drives HVAC thermal power injected into the live EnergyPlus building, over the
    configured window. Writes a per-step results CSV and a summary JSON with the
    key ROM KPI (plant-vs-model one-step RMSE).
    """
    logger = get_logger(
        "mpc.run_mpc_closed_loop",
        log_file=Path("logs") / "run_mpc.log",
    )

    params, observer_block = _load_model(cfg.resolved_model_path())
    logger.info("Loaded identified 4R2C model from %s", cfg.resolved_model_path())
    kalman_gain = _resolve_observer_gain(
        params=params, cfg=cfg, observer_block=observer_block, logger=logger
    )

    df = _load_processed_csv(cfg)
    logger.info("Loaded disturbance CSV with %d rows.", len(df))

    # Optionally set the MPC power bound to the real HVAC max sensible cooling
    # (measured by a full-tilt E+ pass). This bound also normalizes the power
    # zones used by the supervisory setpoint mapping.
    if cfg.control.auto_capacity:
        capacity_kw = calibrate_max_sensible_cooling_kw(cfg, logger)
        cfg = dataclasses.replace(
            cfg,
            control=dataclasses.replace(cfg.control, p_hvac_max_kw=capacity_kw),
        )
        logger.info("Using calibrated p_hvac_max_kw = %.3f kW", capacity_kw)
    else:
        logger.info("Using configured p_hvac_max_kw = %.3f kW", cfg.control.p_hvac_max_kw)

    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    eplus_out_dir = output_dir / "eplus"
    stamp = cfg.window.start.strftime("%Y%m%dT%H%M")
    run_idf_path = output_dir / f"run_{stamp}.idf"

    preparer = IdfPreparer(idd_path=cfg.energyplus.idd_path)
    prepared = preparer.prepare_run_idf(cfg, run_idf_path=run_idf_path)
    logger.info(
        "Prepared run IDF %s (zone=%s, equipment=%s).",
        prepared.run_idf_path,
        prepared.zone_name,
        prepared.equipment_name,
    )

    controller = MpcController(
        params=params,
        control=cfg.control,
        objective=cfg.objective,
    )
    runner = FourR2CRunner(params=params, dt_seconds=cfg.control.dt_seconds)

    orchestrator = _ClosedLoopOrchestrator(
        cfg=cfg,
        controller=controller,
        runner=runner,
        disturbance=DisturbanceProvider(
            df=df,
            dt_seconds=cfg.control.dt_seconds,
            tou=cfg.tou,
            comfort=cfg.comfort,
        ),
        init_walls=_initial_walls(cfg=cfg, params=params, df=df),
        kalman_gain=kalman_gain,
        logger=logger,
    )

    plant = EnergyPlusPlant(
        install_dir=cfg.energyplus.install_dir,
        weather_path=cfg.paths.weather,
        zone_name=prepared.zone_name,
        dt_seconds=cfg.control.dt_seconds,
        calendar_year=cfg.window.start.year,
        actuation_mode=cfg.actuation.mode,
        equipment_name=prepared.equipment_name if cfg.is_direct_power else None,
        setpoint_deadband_c=cfg.actuation.setpoint_deadband_c,
    )

    logger.info("Actuation mode: %s", cfg.actuation.mode)

    logger.info("Starting EnergyPlus closed-loop run...")
    plant.run(
        run_idf_path=prepared.run_idf_path,
        output_dir=eplus_out_dir,
        step_fn=orchestrator.step,
    )
    logger.info("EnergyPlus run complete. Acting steps: %d.", len(orchestrator.records))

    results_df = orchestrator.build_results()
    results_path = output_dir / f"mpc_closed_loop_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Wrote closed-loop results: %s", results_path)

    summary = _build_summary(
        cfg=cfg,
        prepared=prepared,
        results_df=results_df,
        results_path=results_path,
    )
    summary_path = output_dir / "mpc_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote MPC summary: %s", summary_path)
    logger.info(
        "Plant-vs-model one-step RMSE: %.4f degC | total cost: %.4f | "
        "comfort violations: %d steps",
        summary["kpis"]["plant_vs_model_one_step_rmse_c"],
        summary["kpis"]["total_tou_cost"],
        summary["kpis"]["comfort_violation_steps"],
    )


class _ClosedLoopOrchestrator:
    """
    Per-step MPC decision logic, driven by the plant's callback.

    Runs a fixed-gain (steady-state Kalman) state observer to estimate the full
    3-state vector [T_in, T_iw, T_ow] that seeds each MPC solve. Each step:
    predicts the state from the previous posterior and the previously-applied
    input, then corrects ALL three states from the T_in measurement via the
    stored gain K -- so the unmeasured walls are updated by the innovation, not
    just carried open-loop. With K=[1,0,0] this reduces to the previous behavior
    (T_in trusted, walls uncorrected).
    """

    def __init__(
        self,
        cfg: MpcConfig,
        controller: MpcController,
        runner: FourR2CRunner,
        disturbance: DisturbanceProvider,
        init_walls: tuple[float, float],
        kalman_gain: np.ndarray,
        logger,
    ) -> None:
        self.cfg = cfg
        self.controller = controller
        self.runner = runner
        self.disturbance = disturbance
        self.logger = logger

        self._kalman_gain = np.asarray(kalman_gain, dtype=float).reshape(-1)
        self._init_walls = init_walls
        # Posterior full-state estimate and the input applied over the previous
        # interval (needed for the predict step). Both None until the first step.
        self._x_hat: np.ndarray | None = None
        self._prev_input: FourR2CInput | None = None

        self.records: list[dict] = []

    def step(
        self,
        interval_start: datetime,
        obs: PlantObservation,
    ) -> float | None:
        ts = pd.Timestamp(interval_start)
        window_start = pd.Timestamp(self.cfg.window.start)
        window_end = pd.Timestamp(self.cfg.window.end)

        if ts < window_start or ts >= window_end:
            return None  # outside the acting window -> plant applies nothing

        t_in_measured_c = obs.t_in_c
        horizon = self.disturbance.horizon(interval_start, self.cfg.control.horizon_steps)

        # State observer: predict from the previous posterior + previously-applied
        # input, then correct all three states from the T_in measurement.
        x0_vec, t_in_apriori = self._estimate_state(t_in_measured_c)

        sol = self.controller.solve(x0_vec, horizon)
        if not sol.success:
            self.logger.warning("MPC solve failed at %s: %s", ts, sol.status)

        # Command sent to the plant, and the power that actually conditions the
        # zone (used for both cost accounting and as the 4R2C observer input):
        #   direct_power:        command IS the injected power; conditioning power
        #                        is that same power.
        #   supervisory_setpoint: command is the target temperature; the real
        #                        conditioning is the heat pump's delivered sensible
        #                        power, read back from the plant (obs.hvac_power_kw,
        #                        a one-step-lagged causal estimate).
        if self.cfg.is_direct_power:
            command: float = sol.p_hvac_applied_kw
            command_kind = "power_kw"
            conditioning_power_kw = sol.p_hvac_applied_kw
        else:
            command = self._supervisory_setpoint(sol, horizon)
            command_kind = "setpoint_c"
            conditioning_power_kw = obs.hvac_power_kw

        # Remember the input actually applied over this interval so the next
        # step's observer can predict forward from it.
        d0 = horizon.disturbance_matrix()[0]
        self._prev_input = FourR2CInput(
            t_oa_c=float(d0[0]),
            g_ghi_kw_m2=float(d0[1]),
            p_int_kw=float(d0[2]),
            p_sol_win_kw=0.0,
            p_hvac_kw=float(conditioning_power_kw),
        )

        self.records.append(
            {
                "timestamp": ts,
                "t_in_measured_c": float(t_in_measured_c),
                # a-priori predicted T_in (before the measurement update) = the
                # observer's one-step prediction; NaN on the first step.
                "t_in_model_pred_c": t_in_apriori,
                "mpc_command": float(command),
                "mpc_command_kind": command_kind,
                "conditioning_power_kw": float(conditioning_power_kw),
                "tou_rate": float(horizon.tou_rate[0]),
                "comfort_lower_c": float(horizon.comfort_lower_c[0]),
                "comfort_upper_c": float(horizon.comfort_upper_c[0]),
                "solver_ok": bool(sol.success),
            }
        )

        return command

    def _estimate_state(self, t_in_measured_c: float) -> tuple[np.ndarray, float]:
        """
        Fixed-gain state observer. On the first call, seed the estimate from the
        measured T_in and the history-based wall init. Thereafter: predict from
        the previous posterior + previously-applied input, then correct all three
        states from the T_in innovation via the stored gain.

        Returns the posterior state vector (x0 for the MPC) and the a-priori
        predicted T_in (NaN on the first step).
        """
        y = float(t_in_measured_c)

        if self._x_hat is None or self._prev_input is None:
            self._x_hat = np.array(
                [y, self._init_walls[0], self._init_walls[1]], dtype=float
            )
            return self._x_hat.copy(), float("nan")

        # Predict from the previous posterior using the previously-applied input.
        pred_state = self.runner.model.step(
            state=FourR2CState.from_vector(self._x_hat),
            inp=self._prev_input,
            dt_seconds=self.cfg.control.dt_seconds,
        )
        x_pred = pred_state.as_vector()
        t_in_apriori = float(x_pred[0])

        # Correct all three states from the T_in innovation.
        innovation = y - t_in_apriori
        self._x_hat = x_pred + self._kalman_gain * innovation
        return self._x_hat.copy(), t_in_apriori

    def _supervisory_setpoint(self, sol, horizon) -> float:
        """
        Map the MPC's planned power trajectory to a single setpoint via power-zone
        binning. Each planned step is classified into a power zone (a signed
        fraction of p_hvac_max). Starting at step 0, find how long that same zone
        persists (the current power regime), then command the planned temperature
        at the END of that regime. This clears the thermostat deadband (delta-T
        accumulates over the regime) while preserving near-term intent (the regime
        starts now), so the heat pump actually pre-cools / coasts as planned.
        """
        p_plan = sol.p_hvac_plan_kw
        t_plan = sol.t_in_plan_c
        h = len(p_plan)

        p_max = self.cfg.control.p_hvac_max_kw
        n_zones = self.cfg.actuation.num_power_zones

        z0 = _power_zone(p_plan[0], p_max, n_zones)
        run_len = 1
        while run_len < h and _power_zone(p_plan[run_len], p_max, n_zones) == z0:
            run_len += 1

        # Floor the lookahead so a very brief regime still clears the deadband.
        lookahead = min(max(run_len, self.cfg.actuation.min_lookahead_steps), h)
        idx = lookahead - 1

        target = float(t_plan[idx])
        # Clamp into the comfort band shrunk by half the deadband, so the deadband
        # slop around the setpoint still lands inside comfort.
        half_db = 0.5 * self.cfg.actuation.setpoint_deadband_c
        lo = horizon.comfort_lower_c[idx] + half_db
        hi = horizon.comfort_upper_c[idx] - half_db
        if lo > hi:  # deadband wider than the comfort band -> aim at its center
            lo = hi = 0.5 * (horizon.comfort_lower_c[idx] + horizon.comfort_upper_c[idx])
        return float(np.clip(target, lo, hi))

    _COLUMNS = [
        "timestamp",
        "t_in_measured_c",
        "t_in_model_pred_c",
        "mpc_command",
        "mpc_command_kind",
        "conditioning_power_kw",
        "tou_rate",
        "comfort_lower_c",
        "comfort_upper_c",
        "comfort_violation_c",
        "energy_cost_step",
        "solver_ok",
    ]

    def build_results(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=self._COLUMNS)

        df = pd.DataFrame(self.records)
        dt_hours = self.cfg.control.dt_seconds / 3600.0

        under = df["comfort_lower_c"] - df["t_in_measured_c"]
        over = df["t_in_measured_c"] - df["comfort_upper_c"]
        df["comfort_violation_c"] = np.maximum(0.0, np.maximum(under, over))
        # Cost is on the power that actually conditioned the zone (injected in
        # direct mode, delivered by the heat pump in supervisory mode).
        df["energy_cost_step"] = (
            df["tou_rate"] * df["conditioning_power_kw"].abs() * dt_hours
        )

        return df[self._COLUMNS]


def _power_zone(power_kw: float, p_max_kw: float, n_zones: int) -> tuple[int, int]:
    """
    Classify a planned power into a signed power zone: (direction, magnitude_bin).

    `magnitude_bin` is `floor(|power|/p_max * n_zones)` clamped to [0, n_zones-1].
    The lowest bin (near-zero power) is treated as direction-agnostic so tiny
    heating/cooling jitter doesn't break a coasting regime.
    """
    if p_max_kw <= 0:
        return (0, 0)
    frac = min(abs(power_kw) / p_max_kw, 1.0)
    bin_idx = min(int(frac * n_zones), n_zones - 1)
    if bin_idx == 0:
        return (0, 0)
    direction = 1 if power_kw > 0 else -1
    return (direction, bin_idx)


# ---------------------------------------------------------------------- #
# Loading helpers
# ---------------------------------------------------------------------- #


def _load_model(model_path: Path) -> tuple[FourR2CParameters, dict | None]:
    """
    Load identified 4R2C parameters and the state-observer ingredients
    (Q/R/dt/innovation std) from model.json. Raises a clear error if the
    identified model is not available. Wall initial conditions are not read here
    -- they are reconstructed from each run's own history via warm_start_walls.
    The observer block may be None (older model.json), in which case the MPC
    falls back to a K=[1,0,0] observer (T_in trusted, walls uncorrected).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No identified 4R2C model available at {model_path}. Run the EKF "
            "sysid workflow for this city/house first (it writes model.json), or "
            "set model.path in the config."
        )

    with model_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    missing = [k for k in _MODEL_PARAM_KEYS if k not in raw]
    if missing:
        raise KeyError(f"model file {model_path} is missing keys: {sorted(missing)}")

    params = FourR2CParameters(
        r_in_iw=float(raw["r_in_iw"]),
        r_iw_ow=float(raw["r_iw_ow"]),
        r_ow_oa=float(raw["r_ow_oa"]),
        r_in_oa=float(raw["r_in_oa"]),
        c_in=float(raw["c_in"]),
        c_w=float(raw["c_w"]),
        alpha_ghi_outer_wall=float(raw["alpha_ghi_outer_wall"]),
        alpha_ghi_inner_wall=float(raw["alpha_ghi_inner_wall"]),
    )

    observer = raw.get("observer")
    return params, observer


def _resolve_observer_gain(
    params: FourR2CParameters,
    cfg: MpcConfig,
    observer_block: dict | None,
    logger,
) -> np.ndarray:
    """
    Build the state-observer gain from the sysid-stored Q/R ingredients and the
    MPC observer config. Falls back to [1, 0, 0] (T_in trusted, walls uncorrected)
    if the observer block is missing or the gain cannot be computed.
    """
    fallback = np.array([1.0, 0.0, 0.0], dtype=float)

    if observer_block is None:
        logger.warning(
            "model.json has no observer block; using K=[1,0,0] (walls "
            "uncorrected). Re-run the sysid workflow to enable wall correction."
        )
        return fallback

    q_temp = np.asarray(observer_block["q_temp"], dtype=float)
    r = float(observer_block["r"])
    innovation_std = float(observer_block["innovation_std_c"])
    obs_dt = float(observer_block.get("dt_seconds", cfg.control.dt_seconds))

    dt = cfg.control.dt_seconds
    if abs(dt - obs_dt) > 1e-6:
        logger.warning(
            "Observer Q was built at dt=%.1fs but control dt=%.1fs; the stored Q "
            "assumes the sysid timestep. Match control.dt_minutes to the sysid "
            "timestep for a consistent gain.",
            obs_dt, dt,
        )

    disc = FourR2CModel(params=params).discretize(dt)

    try:
        if cfg.observer.mode == "fixed":
            f = cfg.observer.inflation_factor
            k = steady_state_kalman_gain(disc.a_d, disc.c, f * q_temp, r)
        else:  # innovation_consistency
            k, f = innovation_consistent_gain(
                disc.a_d, disc.c, q_temp, r, target_innovation_var=innovation_std**2
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning(
            "Could not build observer gain (%s); using K=[1,0,0].", exc
        )
        return fallback

    gain = np.asarray(k, dtype=float).reshape(-1)
    logger.info(
        "State observer: mode=%s, Q-inflation=%.4g, innovation_std=%.3f degC, "
        "gain [T_in, T_iw, T_ow]=[%.4f, %.4f, %.4f]",
        cfg.observer.mode, f, innovation_std, gain[0], gain[1], gain[2],
    )
    return gain


def _load_processed_csv(cfg: MpcConfig) -> pd.DataFrame:
    csv_path = (
        cfg.paths.processed_root
        / cfg.selection.city
        / f"{cfg.selection.house_name}{_EDITED_SUFFIX}"
        / _PROCESSED_FILENAME
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed disturbance CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV {csv_path} is missing columns: {missing}")
    return df


def _initial_walls(
    cfg: MpcConfig,
    params: FourR2CParameters,
    df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Warm-start wall temperatures from the history window preceding the control
    window: steady-state seed + anchored burn-in (warm_start_walls), reconstructed
    from this run's own recent data using the identified parameters.
    """
    start = pd.Timestamp(cfg.window.start)
    history_start = start - pd.Timedelta(hours=cfg.history_hours)
    history_df = df.loc[(df.index >= history_start) & (df.index < start)]

    if len(history_df) == 0:
        raise ValueError(
            f"History window [{history_start}, {start}) is empty in the processed "
            "CSV. Reduce history_hours or move the window start forward."
        )

    t_in_0 = (
        float(df.loc[start, "T_zone_C"])
        if start in df.index
        else float(history_df["T_zone_C"].iloc[-1])
    )

    result = warm_start_walls(
        params=params,
        history_inputs=build_input_sequence(history_df),
        history_t_in_c=history_df["T_zone_C"].to_numpy(dtype=float),
        t_in_0_c=t_in_0,
        dt_seconds=cfg.control.dt_seconds,
    )
    return result.initial_condition.t_iw_0_c, result.initial_condition.t_ow_0_c


# ---------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------- #


def _build_summary(
    cfg: MpcConfig,
    prepared: PreparedIdf,
    results_df: pd.DataFrame,
    results_path: Path,
) -> dict:
    dt_hours = cfg.control.dt_seconds / 3600.0

    setpoint_tracking_rmse = float("nan")
    if len(results_df) > 0:
        paired = results_df.dropna(subset=["t_in_model_pred_c"])
        if len(paired) > 0:
            err = paired["t_in_measured_c"] - paired["t_in_model_pred_c"]
            rmse = float(np.sqrt(np.mean(np.square(err))))
        else:
            rmse = float("nan")

        total_energy = float((results_df["conditioning_power_kw"].abs() * dt_hours).sum())
        total_cost = float(results_df["energy_cost_step"].sum())
        viol = results_df["comfort_violation_c"]
        viol_steps = int((viol > 1e-9).sum())
        viol_degc_hours = float((viol * dt_hours).sum())
        max_viol = float(viol.max())

        # Supervisory mode: how well did the real HVAC track the commanded setpoint?
        if cfg.is_supervisory:
            track_err = results_df["t_in_measured_c"] - results_df["mpc_command"]
            setpoint_tracking_rmse = float(np.sqrt(np.mean(np.square(track_err))))
    else:
        rmse = float("nan")
        total_energy = 0.0
        total_cost = 0.0
        viol_steps = 0
        viol_degc_hours = 0.0
        max_viol = 0.0

    return {
        "selection": {
            "city": cfg.selection.city,
            "house_name": cfg.selection.house_name,
            "zone_name": prepared.zone_name,
        },
        "window": {
            "start": cfg.window.start.isoformat(),
            "end": cfg.window.end.isoformat(),
            "duration_hours": cfg.window.duration_hours,
        },
        "control": {
            "dt_minutes": cfg.control.dt_minutes,
            "horizon_steps": cfg.control.horizon_steps,
            "p_hvac_max_kw": cfg.control.p_hvac_max_kw,
        },
        "comfort": {"lower_c": cfg.comfort.lower_c, "upper_c": cfg.comfort.upper_c},
        "tou": {
            "default_rate": cfg.tou.default_rate,
            "windows": [
                {"start_hour": w.start_hour, "end_hour": w.end_hour, "rate": w.rate}
                for w in cfg.tou.windows
            ],
        },
        "objective": {
            "w_energy": cfg.objective.w_energy,
            "w_comfort": cfg.objective.w_comfort,
        },
        "actuation": {
            "mode": cfg.actuation.mode,
            "setpoint_deadband_c": cfg.actuation.setpoint_deadband_c,
            "disable_native_hvac": cfg.disable_native_hvac if cfg.is_direct_power else False,
        },
        "model_path": str(cfg.resolved_model_path()),
        "run_idf_path": str(prepared.run_idf_path),
        "results_csv_path": str(results_path),
        "kpis": {
            "n_acting_steps": int(len(results_df)),
            "plant_vs_model_one_step_rmse_c": rmse,
            "supervisory_setpoint_tracking_rmse_c": setpoint_tracking_rmse,
            "total_energy_kwh": total_energy,
            "total_tou_cost": total_cost,
            "comfort_violation_steps": viol_steps,
            "comfort_violation_degc_hours": viol_degc_hours,
            "max_comfort_violation_c": max_viol,
        },
    }
