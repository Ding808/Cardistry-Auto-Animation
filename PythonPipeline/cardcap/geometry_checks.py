"""Read-only, scale-free geometry and preview review gates.

These bounds screen cooperative cardistry photographed within arm's reach.
They are explicit review criteria, not
observations, a calibration, a physical scale anchor, or solver residuals.
No function in this module adjusts geometry to satisfy a bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GeometryReviewCriteria:
    wrist_distance_over_hand_length_max: float = 1.5  # Strict upper bound.
    depth_over_hand_length_min: float = 2.0
    depth_over_hand_length_max: float = 10.0
    depth_difference_over_hand_length_max: float = 2.0  # Strict upper bound.
    display_hand_long_edge_min_px: float = 24.0
    display_hand_foreground_min_px: float = 64.0


DEFAULT_CRITERIA = GeometryReviewCriteria()


def _unknown(reason: str) -> dict:
    return {"status": "unassessable", "values": None, "reason": reason,
            "minimum": None, "median": None, "maximum": None,
            "failed_frame_indices": [], "unassessable_frame_indices": []}


def _numeric_gate(values, valid, *, lower=None, upper=None, strict_upper=False):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    failed = valid & ((values < lower) if lower is not None else False)
    if upper is not None:
        failed |= valid & ((values >= upper) if strict_upper else (values > upper))
    finite = values[valid]
    return {"status": "failed" if failed.any() else ("unassessable" if not valid.all() or not valid.any() else "passed"),
            "values": [float(x) if known else None for x, known in zip(values, valid)],
            "minimum": float(finite.min()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None,
            "failed_frame_indices": np.flatnonzero(failed).tolist(),
            "unassessable_frame_indices": np.flatnonzero(~valid).tolist(),
            "criterion": {"lower_inclusive": lower, "upper": upper, "upper_is_strict": strict_upper}}


def assess_geometry(wrists_camera, neutral_hand_length, *,
                    global_geometry_known: bool,
                    numeric_hypothesis_available: bool = False,
                    frame_validity=None,
                    criteria: GeometryReviewCriteria = DEFAULT_CRITERIA) -> dict:
    """Check N x 2 x 3 wrists ordered left/right in one camera frame.

    Z is signed optical-axis depth, not distance to the camera. All numbers,
    including the neutral hand length, MUST use the same coordinate units.
    The denominator is extended neutral MANO wrist joint0 to middle fingertip
    vertex443; never substitute a posed-frame length or a population length.

    Unknown geometry returns unassessable, not pass. An explicit historical
    numeric hypothesis can be audited even when not physically established:
    violations fail that hypothesis; satisfying its bounds remains unassessable
    for overall geometry acceptance. Every evaluable frame must meet each bound.
    """
    names = ("wrist_distance_over_hand_length", "left_depth_over_hand_length",
             "right_depth_over_hand_length", "depth_difference_over_hand_length")
    result = {"criteria": asdict(criteria), "criterion_scope": "Review-only cooperative-cardistry / arm-reach hypothesis; not a solver prior or measurement",
              "units": "dimensionless; same units in numerator and neutral-model denominator",
              "global_geometry_known": bool(global_geometry_known),
              "numeric_hypothesis_available": bool(numeric_hypothesis_available),
              "physical_geometry_validated": False}
    if not global_geometry_known and not numeric_hypothesis_available:
        result.update(status="unassessable", numeric_hypothesis_status="unassessable",
                      checks={name: _unknown("Global camera / relative hand placement is unknown") for name in names})
        return result
    if neutral_hand_length is None or not np.isfinite(neutral_hand_length) or neutral_hand_length <= 0:
        result.update(status="unassessable", numeric_hypothesis_status="unassessable",
                      checks={name: _unknown("A positive neutral-model length in matching units is unavailable") for name in names})
        return result
    if wrists_camera is None:
        result.update(status="unassessable", numeric_hypothesis_status="unassessable",
                      checks={name: _unknown("Global wrist coordinates are unavailable") for name in names})
        return result
    points = np.asarray(wrists_camera, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (2, 3):
        raise ValueError("wrists_camera must be an N x 2 x 3 array, ordered left then right")
    valid = np.isfinite(points).all(axis=2)
    if frame_validity is not None:
        masks = np.asarray(frame_validity, dtype=bool)
        if masks.shape != points.shape[:2]:
            raise ValueError("frame_validity must be N x 2")
        valid &= masks
    both = valid.all(axis=1)
    difference = points[:, 0] - points[:, 1]
    c = criteria
    checks = {
        names[0]: _numeric_gate(np.linalg.norm(difference, axis=1) / neutral_hand_length, both,
                               upper=c.wrist_distance_over_hand_length_max, strict_upper=True),
        names[1]: _numeric_gate(points[:, 0, 2] / neutral_hand_length, valid[:, 0],
                               lower=c.depth_over_hand_length_min, upper=c.depth_over_hand_length_max),
        names[2]: _numeric_gate(points[:, 1, 2] / neutral_hand_length, valid[:, 1],
                               lower=c.depth_over_hand_length_min, upper=c.depth_over_hand_length_max),
        names[3]: _numeric_gate(np.abs(difference[:, 2]) / neutral_hand_length, both,
                               upper=c.depth_difference_over_hand_length_max, strict_upper=True),
    }
    statuses = [x["status"] for x in checks.values()]
    numeric = "failed" if "failed" in statuses else ("unassessable" if "unassessable" in statuses else "passed")
    result.update(checks=checks, numeric_hypothesis_status=numeric,
                  status=numeric if global_geometry_known or numeric == "failed" else "unassessable")
    return result


def assess_display_samples(samples: Mapping[str, list[dict]] | None, *,
                           criteria: GeometryReviewCriteria = DEFAULT_CRITERIA) -> dict:
    """Gate per-hand visible pixels at final review-pane resolution.

    Each sample has frame, long_edge_px, foreground_pixels, and assessable.
    Callers must supply measured/rasterized hand support with a documented
    association method. Unknown/overlapping segmentation must not become pass.
    The 24 px span + 64 px area are declared display usability criteria chosen
    to reject a handful of pixels; they do not validate pose detail or geometry.
    """
    result = {"status": "unassessable", "scope": "display_only_review_criterion",
              "criterion": {"long_edge_min_px": criteria.display_hand_long_edge_min_px,
                            "foreground_min_px": criteria.display_hand_foreground_min_px},
              "per_side": {}, "physical_geometry_validated": False}
    if not samples:
        result["reason"] = "No independently identifiable rendered hand support available"
        return result
    for side in ("left", "right"):
        records = samples.get(side, [])
        valid = [bool(r.get("assessable", False)) for r in records]
        edges = [r.get("long_edge_px") if r.get("long_edge_px") is not None else np.nan for r in records]
        areas = [r.get("foreground_pixels") if r.get("foreground_pixels") is not None else np.nan for r in records]
        length = _numeric_gate(edges, valid, lower=criteria.display_hand_long_edge_min_px)
        area = _numeric_gate(areas, valid, lower=criteria.display_hand_foreground_min_px)
        statuses = (length["status"], area["status"])
        result["per_side"][side] = {"long_edge": length, "foreground_pixels": area,
            "frames": [r.get("frame", i) for i, r in enumerate(records)],
            "status": "failed" if "failed" in statuses else ("unassessable" if "unassessable" in statuses else "passed")}
    statuses = [s["status"] for s in result["per_side"].values()]
    result["status"] = "failed" if "failed" in statuses else ("unassessable" if "unassessable" in statuses else "passed")
    return result
