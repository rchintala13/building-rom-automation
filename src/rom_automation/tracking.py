from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from rom_automation.logging_utils import get_logger
from rom_automation.sysid.config_types import SysIDConfig


def sysid_model_attributes(config: SysIDConfig) -> dict[str, Any]:
    """
    Descriptors of *how* the model was produced and evaluated -- the "model
    attributes" that let you track methodological improvements across runs. As
    the pipeline evolves, these values change, so an MLflow comparison shows
    exactly which method version each run used.
    """
    return {
        # The current identification / evaluation methodology.
        "validation_evaluation": "full_validation_dataset",
        "wall_init_method": "steady_state_seed+anchored_burn_in",
        "param_identification": "augmented_state_ekf",
        "candidate_selection_metric": "validation_objective",
        "observer_gain": "innovation_consistent_steady_state_kalman",
        "n_step_horizon": config.ekf.n_steps_ahead,
    }


def log_sysid_run(
    output_dir: str | Path,
    config: SysIDConfig,
    config_path: str | Path | None = None,
) -> None:
    """
    Log a completed sysid run to MLflow: model attributes, config params,
    identified-parameter and metric values, and the artifacts (model.json,
    sysid_summary.json, and the exact config YAML if `config_path` is given). No-op
    if tracking is disabled; a soft warning (not a failure) if MLflow is missing
    so the sysid workflow is never blocked by tracking.

    The raw config file is logged as an artifact (not just parsed params) so a run
    records its complete, byte-for-byte inputs independent of git state; a
    `git_dirty` tag flags whether the working tree had uncommitted changes, so a
    run is never silently misattributed to a clean commit.
    """
    logger = get_logger("tracking.sysid")
    if not config.tracking.enabled:
        return

    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tracking.enabled but MLflow is not importable (%s); skipping. "
            "Install mlflow to enable tracking.",
            exc,
        )
        return

    output_dir = Path(output_dir)
    summary = _read_json(output_dir / "sysid_summary.json")
    model = _read_json(output_dir / "model.json")
    if summary is None or model is None:
        logger.warning(
            "Missing sysid_summary.json / model.json in %s; skipping tracking.",
            output_dir,
        )
        return

    if config.tracking.tracking_uri:
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    run_name = f"{config.selection.city}/{config.selection.house_name}"
    with mlflow.start_run(run_name=run_name):
        attrs = sysid_model_attributes(config)
        mlflow.log_params({f"attr.{k}": v for k, v in attrs.items()})
        # Also expose the key method descriptors as tags for easy filtering.
        mlflow.set_tags(
            {
                "run_type": "sysid",
                "city": config.selection.city,
                "house_name": config.selection.house_name,
                "wall_init_method": attrs["wall_init_method"],
                "validation_evaluation": attrs["validation_evaluation"],
            }
        )
        git = _git_commit()
        if git:
            mlflow.set_tag("git_commit", git)
        mlflow.set_tag("git_dirty", str(_git_dirty()).lower())

        # Environment provenance (the axis git/config don't capture): Python +
        # platform, and the EnergyPlus system-dependency version.
        for key, value in _environment_tags().items():
            mlflow.set_tag(key, value)

        mlflow.log_params(_config_params(config, summary))
        mlflow.log_metrics(_sysid_metrics(summary, model))

        for name in ("sysid_summary.json", "model.json"):
            path = output_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))

        # The exact config YAML that produced this run (complete inputs,
        # git-independent), plus the pinned environment spec.
        if config_path is not None and Path(config_path).exists():
            mlflow.log_artifact(str(config_path))
        env_yml = Path("environment.yml")
        if env_yml.exists():
            mlflow.log_artifact(str(env_yml))

    logger.info(
        "Logged sysid run to MLflow experiment '%s' (%s).",
        config.tracking.experiment_name,
        run_name,
    )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _config_params(config: SysIDConfig, summary: dict) -> dict[str, Any]:
    ow = config.ekf.objective_weights
    pn = config.ekf.process_noise
    return {
        "city": config.selection.city,
        "house_name": config.selection.house_name,
        "history_hours": config.dataset.history_hours,
        "split.train": config.dataset.splits.train,
        "split.val": config.dataset.splits.val,
        "split.test": config.dataset.splits.test,
        "ekf.r_value": config.ekf.r_value,
        "ekf.n_steps_ahead": config.ekf.n_steps_ahead,
        "ekf.w_one_step": ow.one_step,
        "ekf.w_n_step": ow.n_step,
        "process_noise.q_in_std_kw": pn.q_in_std_kw,
        "process_noise.q_iw_std_kw": pn.q_iw_std_kw,
        "process_noise.q_ow_std_kw": pn.q_ow_std_kw,
        "n_candidates_evaluated": summary.get("n_candidates_evaluated"),
        "selection_segment": summary.get("selection_segment"),
    }


def _sysid_metrics(summary: dict, model: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}

    best = summary.get("best_metrics", {})
    for segment in ("train", "val", "test"):
        seg = best.get(segment)
        if isinstance(seg, dict):
            _put(metrics, f"rmse_one_step_{segment}", seg.get("rmse_one_step_c"))
            _put(metrics, f"rmse_n_step_{segment}", seg.get("rmse_n_step_c"))

    _put(metrics, "selection_objective", summary.get("best_selection_objective"))

    # Identified parameters (as metrics, so they're comparable across runs).
    for key in (
        "r_in_iw", "r_iw_ow", "r_ow_oa", "r_in_oa",
        "c_in", "c_w", "alpha_ghi_outer_wall", "alpha_ghi_inner_wall",
    ):
        _put(metrics, f"param.{key}", model.get(key))

    observer = model.get("observer", {})
    if isinstance(observer, dict):
        _put(metrics, "observer.innovation_std_c", observer.get("innovation_std_c"))
        _put(metrics, "observer.r", observer.get("r"))

    return metrics


def _put(d: dict[str, float], key: str, value: Any) -> None:
    if value is not None:
        try:
            d[key] = float(value)
        except (TypeError, ValueError):
            pass


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _environment_tags() -> dict[str, str]:
    import platform

    return {
        "env.python_version": platform.python_version(),
        "env.platform": platform.platform(),
        "env.energyplus_version": _energyplus_version(),
    }


def _energyplus_version() -> str:
    """
    Best-effort EnergyPlus version (a system dependency not in conda/pip). Prefers
    the ENERGYPLUS_VERSION env var (the reliable override, especially with
    multiple installs); otherwise returns the HIGHEST version parsed from a
    discovered install dir (e.g. 'EnergyPlusV24-2-0' -> '24.2.0'); 'unknown' if
    none found.
    """
    import glob
    import os
    import re

    explicit = os.environ.get("ENERGYPLUS_VERSION")
    if explicit:
        return explicit

    roots = [
        "C:/EnergyPlusV*",
        "C:/EnergyPlus*",
        "/usr/local/EnergyPlus-*",
        "/Applications/EnergyPlus*",
    ]
    versions: list[tuple[int, int, int]] = []
    for pattern in roots:
        for path in glob.glob(pattern):
            match = re.search(r"(\d+)[.\-](\d+)[.\-](\d+)", Path(path).name)
            if match:
                versions.append(
                    (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                )
    if not versions:
        return "unknown"
    hi = max(versions)
    return f"{hi[0]}.{hi[1]}.{hi[2]}"


def _git_dirty() -> bool:
    """
    True if the working tree has uncommitted changes at run time (so the run
    can't be faithfully reproduced from `git_commit` alone).
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False
