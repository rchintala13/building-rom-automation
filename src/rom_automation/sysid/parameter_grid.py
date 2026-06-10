from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rom_automation.models.types import FourR2CParameters


_PARAMETER_NAMES: tuple[str, ...] = (
    "r_in_iw",
    "r_iw_ow",
    "r_ow_oa",
    "r_in_oa",
    "c_in",
    "c_w",
    "alpha_ghi_outer_wall",
    "alpha_ghi_inner_wall",
)


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


@dataclass(frozen=True)
class _Fractions:
    frac: float
    multiple: float


@dataclass(frozen=True)
class ParameterBoundFractions:
    """
    Per-parameter bound fractions used during EKF updates.

    For each parameter the EKF clips the augmented-state component to:
        [frac * initial_guess, multiple * initial_guess]

    where `initial_guess` is the candidate value used to initialize the EKF.
    """

    r_in_iw: _Fractions
    r_iw_ow: _Fractions
    r_ow_oa: _Fractions
    r_in_oa: _Fractions
    c_in: _Fractions
    c_w: _Fractions
    alpha_ghi_outer_wall: _Fractions
    alpha_ghi_inner_wall: _Fractions


@dataclass(frozen=True)
class ParameterBounds:
    """
    Absolute lower/upper bounds for each parameter, computed for a given
    candidate initial guess.
    """

    r_in_iw: tuple[float, float]
    r_iw_ow: tuple[float, float]
    r_ow_oa: tuple[float, float]
    r_in_oa: tuple[float, float]
    c_in: tuple[float, float]
    c_w: tuple[float, float]
    alpha_ghi_outer_wall: tuple[float, float]
    alpha_ghi_inner_wall: tuple[float, float]


def parameter_bound_fractions_from_dict(
    fractions_dict: dict,
) -> ParameterBoundFractions:
    """
    Build a ParameterBoundFractions from a nested dict, e.g.:
        {
            "r_in_iw": {"frac": 0.5, "multiple": 2.0},
            ...
        }
    """
    missing = [name for name in _PARAMETER_NAMES if name not in fractions_dict]
    if missing:
        raise ValueError(
            f"parameter_bounds is missing entries for: {missing}"
        )

    kwargs = {}
    for name in _PARAMETER_NAMES:
        entry = fractions_dict[name]
        if "frac" not in entry or "multiple" not in entry:
            raise ValueError(
                f"parameter_bounds['{name}'] must define both 'frac' and 'multiple'."
            )

        frac = float(entry["frac"])
        multiple = float(entry["multiple"])

        if frac > multiple:
            raise ValueError(
                f"parameter_bounds['{name}']: frac ({frac}) must be <= multiple ({multiple})."
            )

        kwargs[name] = _Fractions(frac=frac, multiple=multiple)

    return ParameterBoundFractions(**kwargs)


def compute_parameter_bounds(
    fractions: ParameterBoundFractions,
    initial_guess: FourR2CParameters,
) -> ParameterBounds:
    """
    Resolve per-parameter [frac * initial, multiple * initial] bounds for one
    candidate initial guess.
    """
    kwargs = {}
    for name in _PARAMETER_NAMES:
        f: _Fractions = getattr(fractions, name)
        initial = float(getattr(initial_guess, name))
        lower = f.frac * initial
        upper = f.multiple * initial

        if lower > upper:
            lower, upper = upper, lower

        kwargs[name] = (lower, upper)

    return ParameterBounds(**kwargs)


def parameter_names() -> tuple[str, ...]:
    return _PARAMETER_NAMES


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