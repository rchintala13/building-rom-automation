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
        self._rewrite_schedule_file_paths(idf=idf, config=config)
        self._replace_output_variables(idf=idf, config=config)
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

        old_schedule = self._find_schedule_day_interval_by_name(
            idf=idf,
            schedule_name=setpoint_cfg.schedule_name,
        )

        schedule_name = str(old_schedule.Name)
        schedule_type_limits_name = str(old_schedule.Schedule_Type_Limits_Name)
        interpolate_to_timestep = str(old_schedule.Interpolate_to_Timestep)

        # Remove the old schedule
        idf.removeidfobject(old_schedule)

        fields = {
            "Name": schedule_name,
            "Schedule_Type_Limits_Name": schedule_type_limits_name,
            "Interpolate_to_Timestep": interpolate_to_timestep,
        }

        for i, interval in enumerate(setpoint_cfg.intervals, start=1):

            self._validate_until_string(interval.until)

            fields[f"Time_{i}"] = interval.until
            fields[f"Value_Until_Time_{i}"] = interval.value_c + offset_c

        idf.newidfobject(
            "SCHEDULE:DAY:INTERVAL",
            **fields,
        )

    def _find_schedule_day_interval_by_name(self, idf: IDF, schedule_name: str):
        schedules = idf.idfobjects["SCHEDULE:DAY:INTERVAL"]

        for schedule in schedules:
            if str(schedule.Name).strip() == schedule_name:
                return schedule

        raise ValueError(
            f"Could not find Schedule:Day:Interval with name {schedule_name!r}"
        )
    
    def _rewrite_schedule_file_paths(self, idf: IDF, config: IDFEditConfig) -> None:
        """
        Rewrite Schedule:File paths to point to a local schedule directory.
        """
        schedule_cfg = config.schedule_files

        if schedule_cfg is None:
            return

        if not schedule_cfg.rewrite_paths:
            return

        schedules = idf.idfobjects["SCHEDULE:FILE"]

        for schedule in schedules:

            original_path = Path(schedule.File_Name)

            # Keep only filename
            filename = original_path.name

            new_path = schedule_cfg.root_dir / filename

            schedule.File_Name = str(new_path)

    def _replace_output_variables(self, idf: IDF, config: IDFEditConfig) -> None:
        """
        Remove all existing Output:Variable objects and replace them with the
        exact set requested in the YAML config.
        """
        zone_name = config.simoutputs.zone_name
        reporting_frequency = config.simoutputs.reporting_frequency

        self._validate_zone_exists(idf=idf, zone_name=zone_name)

        existing_output_vars = list(idf.idfobjects["OUTPUT:VARIABLE"])
        for obj in existing_output_vars:
            idf.removeidfobject(obj)

        for variable_name in config.simoutputs.variables.site:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="Environment",
                Variable_Name=variable_name,
                Reporting_Frequency=reporting_frequency,
            )

        for variable_name in config.simoutputs.variables.zone:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value=zone_name,
                Variable_Name=variable_name,
                Reporting_Frequency=reporting_frequency,
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

    @staticmethod
    def _validate_until_string(until: str) -> None:
        import re

        if not re.fullmatch(r"(?:[01]\d|2[0-4]):[0-5]\d", until):
            raise ValueError(
                f"Invalid schedule interval time {until!r}. Expected hh:mm, e.g. '06:00'."
            )
    
    def _validate_zone_exists(self, idf: IDF, zone_name: str) -> None:
        zone_names = {
            str(zone.Name).strip().lower()
            for zone in idf.idfobjects["ZONE"]
        }

        if zone_name.strip().lower() not in zone_names:
            available = sorted(zone_names)
            raise ValueError(
                f"Zone name {zone_name!r} not found in IDF. "
                f"Available zones: {available}"
            )