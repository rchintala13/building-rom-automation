from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eppy.modeleditor import IDF

from rom_automation.agent.knowledge_base import find_closest_prior, load_knowledge_base
from rom_automation.agent.state import AgentState

# Standard air properties
_RHO_AIR = 1.2        # kg/m³
_CP_AIR = 1005.0      # J/kgK
_DEFAULT_CEILING_H = 2.7  # m, fallback if zone volume is missing


def load_idf(state: AgentState) -> dict[str, Any]:
    idf_path = state["idf_path"]
    idd_path = state.get("idd_path") or os.environ.get("ENERGYPLUS_IDD_PATH", "")
    kb_path = state.get("knowledge_base_path", "data/agent_knowledge_base.json")

    if not idd_path:
        raise ValueError(
            "IDD path not set. Provide idd_path in state or set ENERGYPLUS_IDD_PATH."
        )

    IDF.setiddname(str(idd_path))
    idf = IDF(str(idf_path))

    features: dict[str, Any] = {}

    _extract_zone_geometry(idf, features)
    _extract_temperature_capacity_multiplier(idf, features)
    _extract_surfaces(idf, features)
    _extract_windows(idf, features)
    _extract_wall_construction(idf, features)
    _extract_window_properties(idf, features)
    _extract_infiltration(idf, features)
    _fill_missing_geometry(features)

    records = load_knowledge_base(kb_path)
    distilled_prior = find_closest_prior(records, features)

    return {
        "building_features": features,
        "distilled_prior": distilled_prior,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_zone_geometry(idf: IDF, features: dict) -> None:
    zones = idf.idfobjects["ZONE"]
    # Use first zone as the primary conditioned zone
    for zone in zones:
        name = str(zone.Name).strip().lower()
        if "attic" in name or "garage" in name or "crawl" in name:
            continue
        try:
            vol = float(zone.Volume) if zone.Volume else None
        except (AttributeError, ValueError, TypeError):
            vol = None
        try:
            area = float(zone.Floor_Area) if zone.Floor_Area else None
        except (AttributeError, ValueError, TypeError):
            area = None
        features["zone_volume_m3"] = vol
        features["zone_floor_area_m2"] = area
        features["conditioned_zone_name"] = str(zone.Name).strip()
        return

    # Fallback: first zone
    if zones:
        zone = zones[0]
        features["zone_volume_m3"] = _safe_float(zone, "Volume")
        features["zone_floor_area_m2"] = _safe_float(zone, "Floor_Area")
        features["conditioned_zone_name"] = str(zone.Name).strip()


def _extract_temperature_capacity_multiplier(idf: IDF, features: dict) -> None:
    objs = idf.idfobjects["ZONECAPACITANCEMULTIPLIER:RESEARCHSPECIAL"]
    features["temp_capacity_multiplier"] = 1.0
    if objs:
        val = _safe_float(objs[0], "Temperature_Capacity_Multiplier")
        if val is not None:
            features["temp_capacity_multiplier"] = val


def _extract_surfaces(idf: IDF, features: dict) -> None:
    surfaces = idf.idfobjects["BUILDINGSURFACE:DETAILED"]
    conditioned = features.get("conditioned_zone_name", "").lower()

    ext_wall_area = 0.0
    floor_area = 0.0
    int_surface_area = 0.0
    ext_wall_constructions: list[str] = []

    for surf in surfaces:
        surf_type = str(surf.Surface_Type).strip().upper()
        bc = str(surf.Outside_Boundary_Condition).strip().upper()
        zone = str(surf.Zone_Name).strip().lower()

        try:
            area = float(surf.area)
        except (AttributeError, ValueError, TypeError):
            area = 0.0

        is_conditioned_zone = (not conditioned) or (conditioned in zone)

        if surf_type == "WALL":
            if bc == "OUTDOORS" and is_conditioned_zone:
                ext_wall_area += area
                ext_wall_constructions.append(str(surf.Construction_Name).strip())
            elif bc not in ("OUTDOORS", "GROUND"):
                int_surface_area += area
        elif surf_type == "FLOOR" and is_conditioned_zone:
            floor_area += area

    features["ext_wall_area_m2"] = ext_wall_area
    features["int_surface_area_m2"] = int_surface_area
    features["ext_wall_constructions"] = ext_wall_constructions

    if floor_area > 0 and not features.get("zone_floor_area_m2"):
        features["zone_floor_area_m2"] = floor_area


def _extract_windows(idf: IDF, features: dict) -> None:
    windows = idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]
    window_area = 0.0
    window_constructions: list[str] = []

    for win in windows:
        try:
            area = float(win.area)
        except (AttributeError, ValueError, TypeError):
            area = 0.0
        window_area += area
        window_constructions.append(str(win.Construction_Name).strip())

    features["window_area_m2"] = window_area
    features["window_constructions"] = window_constructions
    ext_wall_area = features.get("ext_wall_area_m2", 0.0)
    features["wwr"] = window_area / ext_wall_area if ext_wall_area > 0 else 0.0


def _extract_wall_construction(idf: IDF, features: dict) -> None:
    """Compute wall R-value (m²K/W) and thermal mass (J/K/m²) from material layers."""
    constructions = idf.idfobjects["CONSTRUCTION"]
    materials = {str(m.Name).strip(): m for m in idf.idfobjects["MATERIAL"]}

    construction_names: list[str] = features.get("ext_wall_constructions", [])
    target_name = construction_names[0] if construction_names else None

    const_obj = None
    for c in constructions:
        if target_name and str(c.Name).strip() == target_name:
            const_obj = c
            break

    if const_obj is None and constructions:
        # Fall back to a construction whose name looks like a wall
        for c in constructions:
            n = str(c.Name).strip().lower()
            if "wall" in n and "adiabatic" not in n and "partition" not in n:
                const_obj = c
                break

    r_total = 0.0
    mass_total = 0.0
    solar_abs_outer = 0.7  # default

    if const_obj is not None:
        layer_names = _get_construction_layers(const_obj)
        for i, layer_name in enumerate(layer_names):
            mat = materials.get(layer_name)
            if mat is None:
                continue
            thickness = _safe_float(mat, "Thickness") or 0.0
            conductivity = _safe_float(mat, "Conductivity") or 0.0
            density = _safe_float(mat, "Density") or 0.0
            specific_heat = _safe_float(mat, "Specific_Heat") or 0.0

            if conductivity > 0:
                r_total += thickness / conductivity
            mass_total += density * specific_heat * thickness

            if i == 0:  # outermost layer
                solar_abs_outer = _safe_float(mat, "Solar_Absorptance") or 0.7

    # Add standard air-film resistances (interior + exterior)
    r_total += 0.13 + 0.04  # m²K/W

    features["wall_r_value_m2k_per_w"] = r_total if r_total > 0.17 else 1.0
    features["wall_thermal_mass_j_per_k_m2"] = mass_total if mass_total > 0 else 5000.0
    features["solar_absorptance_exterior"] = solar_abs_outer


def _extract_window_properties(idf: IDF, features: dict) -> None:
    """Extract U-factor (W/m²K) and SHGC from WindowMaterial:SimpleGlazingSystem."""
    simple = idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]
    if simple:
        win = simple[0]
        features["window_u_w_m2k"] = _safe_float(win, "UFactor") or 3.0
        features["window_shgc"] = _safe_float(win, "Solar_Heat_Gain_Coefficient") or 0.4
    else:
        features["window_u_w_m2k"] = 3.0
        features["window_shgc"] = 0.4


def _extract_infiltration(idf: IDF, features: dict) -> None:
    """Estimate ACH from ZoneInfiltration:DesignFlowRate objects."""
    infil_objects = idf.idfobjects["ZONEINFILTRATION:DESIGNFLOWRATE"]
    vol = features.get("zone_volume_m3")
    ach = 0.0

    for obj in infil_objects:
        method = str(obj.Design_Flow_Rate_Calculation_Method).strip().upper()
        try:
            if method == "FLOW/ZONE":
                flow_m3s = float(obj.Design_Flow_Rate) if obj.Design_Flow_Rate else 0.0
                if flow_m3s > 0 and vol and vol > 0:
                    ach += flow_m3s * 3600.0 / vol
            elif method == "AIRCHANGES/HOUR":
                ach += float(obj.Air_Changes_per_Hour) if obj.Air_Changes_per_Hour else 0.0
            elif method in ("FLOW/EXTERIORAREA", "FLOW/EXTERIORWALLAREA"):
                flow_per_m2 = (
                    float(obj.Flow_Rate_per_Exterior_Surface_Area)
                    if obj.Flow_Rate_per_Exterior_Surface_Area
                    else 0.0
                )
                ext_area = features.get("ext_wall_area_m2", 0.0)
                if flow_per_m2 > 0 and ext_area > 0 and vol and vol > 0:
                    ach += flow_per_m2 * ext_area * 3600.0 / vol
            elif method == "FLOW/AREA":
                flow_per_m2 = (
                    float(obj.Flow_Rate_per_Floor_Area)
                    if obj.Flow_Rate_per_Floor_Area
                    else 0.0
                )
                floor_area = features.get("zone_floor_area_m2", 0.0)
                if flow_per_m2 > 0 and floor_area > 0 and vol and vol > 0:
                    ach += flow_per_m2 * floor_area * 3600.0 / vol
        except (AttributeError, ValueError, TypeError):
            continue

    # ResStock buildings commonly use EffectiveLeakageArea; fall back to 0.35 ACH if zero
    features["infiltration_ach"] = ach if ach > 0 else 0.35


def _fill_missing_geometry(features: dict) -> None:
    vol = features.get("zone_volume_m3")
    area = features.get("zone_floor_area_m2")

    if not vol and area and area > 0:
        features["zone_volume_m3"] = area * _DEFAULT_CEILING_H
    elif not area and vol and vol > 0:
        features["zone_floor_area_m2"] = vol / _DEFAULT_CEILING_H


def _get_construction_layers(const_obj) -> list[str]:
    layers = []
    for field in ["Outside_Layer", *[f"Layer_{i}" for i in range(2, 20)]]:
        try:
            val = getattr(const_obj, field, None)
            if val:
                layers.append(str(val).strip())
        except Exception:
            break
    return layers


def _safe_float(obj: Any, field: str) -> float | None:
    try:
        val = getattr(obj, field, None)
        return float(val) if val not in (None, "", " ") else None
    except (ValueError, TypeError):
        return None
