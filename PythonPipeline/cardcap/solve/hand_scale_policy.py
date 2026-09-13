"""Choose one source-unit scale for the hand solve and its matching bind mesh.

This resolver does not mutate geometry. Its returned ``applied`` payload contract
may be exported only after the caller applies the factor to both local MANO
geometry and initial camera-origin translations. Unknown metric scale keeps model
coordinates unchanged; UE display units do not establish physical centimetres.
No UE actor transform or already-metric card PnP output is scaled here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .scale import CardPnPScaleAnchor, estimate_scene_scale, measure_neutral_mano_hand


CONFIG_DIRECTORY = Path(__file__).resolve().parents[3] / "Config"
EVIDENCE_CATEGORIES = {"low", "medium", "high", "unknown"}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _read_object(path):
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _configuration(config):
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping or None")
        data = dict(config)
        return data, {"source": "caller_config", "supplied_keys": sorted(data)}
    path = CONFIG_DIRECTORY / "HandSolve.json"
    if path.is_file():
        data, provenance = _read_object(path)
        return data, dict(provenance, source="HandSolve.json")
    return {}, {"source": "no_optional_config", "missing_optional_config": str(path)}


def resolve_hand_scale(arrays, shared_shape, source_observations_sha256, model_sha256, camera,
                       *, card_anchors=(), anthropometry_path=None, config=None) -> dict:
    """Supported card anchors -> optional subject measurement -> unknown metric scale.

    ``arrays`` are the original canonical-right MANO arrays (model source
    units, not this subject's measured metres); both hands share ``shared_shape`` and the same factor. Cards must be
    existing CardPnPScaleAnchor objects with caller-verified correspondence to
    THIS hand scale group; a plain pose or mask is not an anchor. A sole supplied
    anchor group is used as its explicit caller declaration. Mixed groups require
    ``config['scale_group']``; otherwise the default group binds observations SHA.

    The optional anthropometry JSON accepts ``hand_length_mm`` (null/missing =
    unknown) and records all supplied definition/evidence fields. A provided
    length is a caller measurement assertion, not independent error validation.
    No physical square size, hand-length argument or new user input is required.
    """
    if not isinstance(camera, dict):
        raise ValueError("camera must be an object")
    if "calibrated" in camera and not isinstance(camera["calibrated"], bool):
        raise ValueError("camera.calibrated must be boolean when supplied")
    camera_calibrated = camera.get("calibrated") is True
    cfg, config_provenance = _configuration(config)
    anchors = tuple(card_anchors)
    if any(not isinstance(anchor, CardPnPScaleAnchor) for anchor in anchors):
        raise ValueError("card_anchors must contain CardPnPScaleAnchor objects, not plain PnP poses")
    groups = {anchor.scale_group for anchor in anchors}
    group = cfg.get("scale_group")
    if group is None:
        if len(groups) > 1:
            raise ValueError("Mixed card scale groups require explicit config.scale_group")
        group = next(iter(groups)) if groups else "mano_shared_shape:" + str(source_observations_sha256)
    if not isinstance(group, str) or not group.strip():
        raise ValueError("scale_group must be nonempty text")
    neutral = measure_neutral_mano_hand(arrays, shared_shape, scale_group=group,
        model_sha256=model_sha256, shape_source_sha256=source_observations_sha256, side="right")
    # Card eligibility is independent of any assumed personal hand length.
    estimate = estimate_scene_scale(scale_group=group, card_anchors=anchors,
        neutral_hand_measurements=[neutral])
    warnings = []
    target = None
    anthropometry = {"status": "not_needed_card_anchor_selected"}
    if estimate.anchor_method == "card_pnp":
        anchor_source = "supported_card_pnp"
    else:
        path = Path(anthropometry_path).resolve() if anthropometry_path is not None else CONFIG_DIRECTORY / "UserAnthropometry.json"
        if path.is_file():
            values, provenance = _read_object(path)
            measured = values.get("hand_length_mm")
            anthropometry = dict(provenance, supplied_fields=values,
                status="unknown" if measured is None else "caller_measurement_assertion")
        elif anthropometry_path is not None:
            raise FileNotFoundError(f"Explicit anthropometry file does not exist: {path}")
        else:
            measured = None
            anthropometry = {"path": str(path), "status": "optional_file_absent"}
        if measured is not None:
            target = _positive(measured, "hand_length_mm")
            definition = values.get("measurement_definition")
            if not isinstance(definition, str) or not definition.strip():
                raise ValueError("Measured hand length requires an explicit measurement_definition")
            supplied_uncertainty = values.get("measurement_uncertainty_mm")
            if supplied_uncertainty is not None:
                if isinstance(supplied_uncertainty, bool) or not isinstance(supplied_uncertainty, (int, float)) or not np.isfinite(supplied_uncertainty) or supplied_uncertainty < 0:
                    raise ValueError("measurement_uncertainty_mm must be finite nonnegative or null")
            measurement_evidence = dict(provenance, kind="user_measured", description=definition,
                uncertainty=None if supplied_uncertainty is None else {"value": supplied_uncertainty, "units": "mm", "source": "caller_measurement_assertion"},
                supplied_fields=values)
            estimate = estimate_scene_scale(scale_group=group, card_anchors=anchors,
                neutral_hand_measurements=[neutral], measured_hand_length_m=target / 1000.,
                measurement_provenance=measurement_evidence)
            anchor_source = "user_anthropometry"
        else:
            anchor_source = None
    category = estimate.scale_confidence
    if category not in EVIDENCE_CATEGORIES:
        raise ValueError(f"Unsupported scale evidence category: {category}")
    metric_known = estimate.meters_per_source_unit is not None
    cap_applied = not camera_calibrated and category == "high"
    if cap_applied:
        category = "medium"
    warnings.extend(estimate.warnings)
    if not camera_calibrated:
        warnings.append("Camera is uncalibrated or unknown; a subject-length mapping cannot establish correct metric depth or calibrate the camera.")
    if metric_known:
        warnings.append("Apply the evidenced factor to hand local geometry, camera-origin translations and the matching bind mesh exactly once. Already metric card PnP is excluded.")
    else:
        warnings.append("Factor 1 is identity on MANO model-source coordinates only. Multiplying those coordinates by 100 for UE display does not measure centimetres or one metre per unit.")
    return {
        "global_scale_factor_applied": float(estimate.global_scale_factor),
        "meters_per_unit": estimate.meters_per_source_unit,
        "nominal_meters_per_source_unit_before_application": None,
        "anchor_method": estimate.anchor_method, "anchor_source": anchor_source,
        "scale_confidence": category, "confidence": None,
        "confidence_semantics": "No calibrated numerical confidence is available; scale_confidence is an evidence category, not a probability or measured error",
        "anchor_frames": list(estimate.anchor_frames), "source_scale_group": group,
        "applied": metric_known,
        "application_kind": "evidenced_metric_mapping" if metric_known else "identity_on_model_coordinates",
        "payload_position_units": "cm_already_scaled" if metric_known else "relative_model_units_ue100",
        "consumer_must_multiply_scale_again": False,
        "application_owner": "Caller applies the returned transform consistently to model geometry and its bind mesh; resolver never mutates arrays",
        "meters_per_unit_semantics": "Original model-source unit to metre mapping when supported; null otherwise. A display conversion does not authorize physical-unit claims",
        "independent_metric_accuracy_measured": False, "relative_metric_error": None, "ten_percent_accuracy_verified": False,
        "source_observations_sha256": source_observations_sha256, "model_sha256": model_sha256,
        "target_hand_length_mm": target, "neutral_hand_measurement": neutral,
        "camera_calibrated": camera_calibrated,
        "camera_intrinsics_source": camera.get("intrinsics_source", "unspecified"),
        "camera_confidence_cap": "medium" if metric_known and not camera_calibrated else None,
        "camera_confidence_cap_applied": cap_applied,
        "config_provenance": config_provenance, "anthropometry_provenance": anthropometry,
        "provenance": {**estimate.provenance,
            "geometry_units": "metric_mapped_ue_centimeters" if metric_known else "relative_model_units_ue100",
            "source_observations_sha256": source_observations_sha256, "model_sha256": model_sha256,
            "identity_factor_is_metric_measurement": False},
        "m3_estimate": estimate.to_dict(),
        "m3_estimate_interpretation": "Independent card or explicit subject measurement only. Neutral model dimensions remain inferred and never constitute a personal measurement by themselves.",
        "warnings": warnings,
    }
