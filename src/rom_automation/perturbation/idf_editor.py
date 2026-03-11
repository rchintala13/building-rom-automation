from __future__ import annotations
from pathlib import Path
from eppy.modeleditor import IDF
from rom_automation.epconfig.types import IDFEditConfig, SetpointValueConfig


class IDFEditor:
    def __init__(self, idd_path: str | Path) -> None:
        self.idd_path = Path(idd_path)

        if not self.idd_path.exists():
            raise FileNotFoundError(f"IDD file not found: {self.idd_path}")

        IDF.setiddname(str(self.idd_path))

    def edit_idf(
        self,
        input_path: str | Path,
        output_path: str | Path,
        config: IDFEditConfig,
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input IDF not found: {input_path}")

        idf = IDF(str(input_path))

        self._edit_run_period(idf=idf, config=config)
        self._edit_timestep(idf=idf, config=config)
        self._edit_setpoints(idf=idf, config=config)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        idf.saveas(str(output_path))

    def _edit_run_period(self, idf: IDF, config: IDFEditConfig) -> None:
        run_periods = idf.idfobjects["RUNPERIOD"]
        if not run_periods:
            raise ValueError("No RUNPERIOD object found in IDF")

        rp = run_periods[0]
        run_period_cfg = config.edits.simulation_control.run_period

        rp.Begin_Month = run_period_cfg.begin_month
        rp.Begin_Day_of_Month = run_period_cfg.begin_day_of_month
        rp.End_Month = run_period_cfg.end_month
        rp.End_Day_of_Month = run_period_cfg.end_day_of_month

    def _edit_timestep(self, idf: IDF, config: IDFEditConfig) -> None:
        timesteps = idf.idfobjects["TIMESTEP"]
        if not timesteps:
            raise ValueError("No TIMESTEP object found in IDF")

        timesteps[0].Number_of_Timesteps_per_Hour = (
            config.edits.simulation_control.timestep_per_hour
        )

    def _edit_setpoints(self, idf: IDF, config: IDFEditConfig) -> None:
        if config.edits.setpoints.heating.enabled:
            self._edit_schedule_day_interval(
                idf=idf,
                setpoint_cfg=config.edits.setpoints.heating,
                offset_c=(
                    config.edits.perturbation.heating_offset_c
                    if config.edits.perturbation.enabled
                    else 0.0
                ),
            )

        if config.edits.setpoints.cooling.enabled:
            self._edit_schedule_day_interval(
                idf=idf,
                setpoint_cfg=config.edits.setpoints.cooling,
                offset_c=(
                    config.edits.perturbation.cooling_offset_c
                    if config.edits.perturbation.enabled
                    else 0.0
                ),
            )

    def _edit_schedule_day_interval(
        self,
        idf: IDF,
        setpoint_cfg: SetpointValueConfig,
        offset_c: float,
    ) -> None:
        if setpoint_cfg.schedule_type != "day_interval":
            raise ValueError(
                f"Unsupported schedule_type: {setpoint_cfg.schedule_type!r}"
            )

        schedule = self._find_schedule_day_interval_by_name(
            idf=idf,
            schedule_name=setpoint_cfg.schedule_name,
        )

        self._clear_day_interval_fields(schedule)

        for i, interval in enumerate(setpoint_cfg.intervals, start=1):
            setattr(schedule, f"Time_{i}", interval.until)
            setattr(schedule, f"Value_Until_Time_{i}", interval.value_c + offset_c)

    def _find_schedule_day_interval_by_name(self, idf: IDF, schedule_name: str):
        schedules = idf.idfobjects["SCHEDULE:DAY:INTERVAL"]

        for schedule in schedules:
            if str(schedule.Name).strip() == schedule_name:
                return schedule

        raise ValueError(
            f"Could not find Schedule:Day:Interval with name {schedule_name!r}"
        )

    @staticmethod
    def _clear_day_interval_fields(schedule) -> None:
        for i in range(1, 25):
            time_field = f"Time_{i}"
            value_field = f"Value_Until_Time_{i}"

            if hasattr(schedule, time_field):
                setattr(schedule, time_field, "")
            if hasattr(schedule, value_field):
                setattr(schedule, value_field, "")