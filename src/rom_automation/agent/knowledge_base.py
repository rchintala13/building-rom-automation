from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Features used to fingerprint a building for similarity lookup.
# All are positive scalars so log-distance is meaningful.
_FINGERPRINT_KEYS = [
    "floor_area_m2",
    "zone_volume_m3",
    "wall_r_value_m2k_per_w",
    "wwr",
    "wall_thermal_mass_j_per_k_m2",
]
# Maximum log-RMS distance before we consider buildings too dissimilar to reuse the prior.
_MAX_DISTANCE = 1.5


def load_knowledge_base(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def save_knowledge_base(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(records, f, indent=2)


def find_closest_prior(
    records: list[dict[str, Any]],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Return the closest knowledge-base record or an empty dict if none is close enough."""
    best: dict[str, Any] = {}
    best_dist = float("inf")

    for record in records:
        dist = _fingerprint_distance(record.get("features", {}), features)
        if dist < best_dist:
            best_dist = dist
            best = record

    return best if best_dist <= _MAX_DISTANCE else {}


def append_record(
    records: list[dict[str, Any]],
    features: dict[str, Any],
    fitted_params: dict[str, float],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """Add a new distilled record; deduplicate by replacing the nearest if very close."""
    new_record = {
        "features": {k: features.get(k) for k in _FINGERPRINT_KEYS},
        "fitted_params": fitted_params,
        "metrics": metrics,
    }

    # Replace an existing near-identical record rather than appending a duplicate
    if records:
        dists = [
            _fingerprint_distance(r.get("features", {}), features) for r in records
        ]
        min_dist = min(dists)
        if min_dist < 0.3:
            idx = dists.index(min_dist)
            records[idx] = new_record
            return records

    records.append(new_record)
    return records


def _fingerprint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    squared = 0.0
    count = 0
    for key in _FINGERPRINT_KEYS:
        va = a.get(key)
        vb = b.get(key)
        if va and vb and va > 0 and vb > 0:
            squared += (math.log10(va) - math.log10(vb)) ** 2
            count += 1
    if count == 0:
        return float("inf")
    return math.sqrt(squared / count)
