from __future__ import annotations

import subprocess
from pathlib import Path


class EnergyPlusRunner:
    """
    Run a single EnergyPlus simulation for one edited IDF file.
    """

    def __init__(self, eplus_exe: str | Path) -> None:
        self.eplus_exe = Path(eplus_exe)

        if not self.eplus_exe.exists():
            raise FileNotFoundError(
                f"EnergyPlus executable not found: {self.eplus_exe}"
            )

        if not self.eplus_exe.is_file():
            raise FileNotFoundError(
                f"EnergyPlus executable path is not a file: {self.eplus_exe}"
            )

    def run_idf(
        self,
        idf_path: str | Path,
        weather_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        """
        Run EnergyPlus on a single IDF.

        Parameters
        ----------
        idf_path
            Path to the edited IDF file.
        weather_path
            Path to the EPW weather file.
        output_dir
            Directory where EnergyPlus outputs should be written.
        """
        idf_path = Path(idf_path)
        weather_path = Path(weather_path)
        output_dir = Path(output_dir)

        self._validate_inputs(
            idf_path=idf_path,
            weather_path=weather_path,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(
            idf_path=idf_path,
            weather_path=weather_path,
            output_dir=output_dir,
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "EnergyPlus simulation failed.\n"
                f"IDF: {idf_path}\n"
                f"Weather: {weather_path}\n"
                f"Output directory: {output_dir}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

    def _build_command(
        self,
        idf_path: Path,
        weather_path: Path,
        output_dir: Path,
    ) -> list[str]:
        return [
            str(self.eplus_exe),
            "-w",
            str(weather_path),
            "-d",
            str(output_dir),
            str(idf_path),
        ]

    @staticmethod
    def _validate_inputs(
        idf_path: Path,
        weather_path: Path,
    ) -> None:
        if not idf_path.exists():
            raise FileNotFoundError(f"IDF file not found: {idf_path}")

        if not idf_path.is_file():
            raise FileNotFoundError(f"IDF path is not a file: {idf_path}")

        if idf_path.suffix.lower() != ".idf":
            raise ValueError(f"Input file is not an IDF: {idf_path}")

        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        if not weather_path.is_file():
            raise FileNotFoundError(f"Weather path is not a file: {weather_path}")

        if weather_path.suffix.lower() != ".epw":
            raise ValueError(f"Weather file is not an EPW: {weather_path}")