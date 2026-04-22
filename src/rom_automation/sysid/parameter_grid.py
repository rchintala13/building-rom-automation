from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rom_automation.models.types import FourR2CParameters


@dataclass(frozen=True)
class ParameterCandidateGrid:
    """
    Candidate values for each independent 4R2C parameter.
    """

    r_in_iw: list[float]
    r_iw_ow: list[float]
    r_ow_oa: list[float]
    r_in_oa: list[float]
    c_in: list[float]
    c_w: list[float]
    alpha_ghi_outer_wall: list[float]
    alpha_ghi_inner_wall: list[float]


def generate_parameter_candidates(
    grid: ParameterCandidateGrid,
) -> list[FourR2CParameters]:
    """
    Generate all parameter combinations as FourR2CParameters objects.
    """
    _validate_grid(grid)

    candidates: list[FourR2CParameters] = []

    for values in product(
        grid.r_in_iw,
        grid.r_iw_ow,
        grid.r_ow_oa,
        grid.r_in_oa,
        grid.c_in,
        grid.c_w,
        grid.alpha_ghi_outer_wall,
        grid.alpha_ghi_inner_wall,
    ):
        candidate = FourR2CParameters(
            r_in_iw=float(values[0]),
            r_iw_ow=float(values[1]),
            r_ow_oa=float(values[2]),
            r_in_oa=float(values[3]),
            c_in=float(values[4]),
            c_w=float(values[5]),
            alpha_ghi_outer_wall=float(values[6]),
            alpha_ghi_inner_wall=float(values[7]),
        )
        candidates.append(candidate)

    return candidates


def count_parameter_candidates(grid: ParameterCandidateGrid) -> int:
    """
    Count the number of Cartesian-product parameter combinations.
    """
    _validate_grid(grid)

    return (
        len(grid.r_in_iw)
        * len(grid.r_iw_ow)
        * len(grid.r_ow_oa)
        * len(grid.r_in_oa)
        * len(grid.c_in)
        * len(grid.c_w)
        * len(grid.alpha_ghi_outer_wall)
        * len(grid.alpha_ghi_inner_wall)
    )


def _validate_grid(grid: ParameterCandidateGrid) -> None:
    grid_dict = {
        "r_in_iw": grid.r_in_iw,
        "r_iw_ow": grid.r_iw_ow,
        "r_ow_oa": grid.r_ow_oa,
        "r_in_oa": grid.r_in_oa,
        "c_in": grid.c_in,
        "c_w": grid.c_w,
        "alpha_ghi_outer_wall": grid.alpha_ghi_outer_wall,
        "alpha_ghi_inner_wall": grid.alpha_ghi_inner_wall,
    }

    for name, values in grid_dict.items():
        if len(values) == 0:
            raise ValueError(f"Parameter grid '{name}' is empty.")

        for value in values:
            _validate_value(name=name, value=float(value))


def _validate_value(name: str, value: float) -> None:
    """
    Validate one parameter candidate value.
    """
    if name.startswith("r_") or name.startswith("c_"):
        if value <= 0:
            raise ValueError(f"{name} values must be positive. Got {value}.")

    if name.startswith("alpha_"):
        # Keep this permissive for now. If you want to enforce [0, 1],
        # uncomment the stricter check below.
        #
        # if not (0.0 <= value <= 1.0):
        #     raise ValueError(
        #         f"{name} values must lie in [0, 1]. Got {value}."
        #     )
        pass