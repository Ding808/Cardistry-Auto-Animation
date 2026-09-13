"""Independent global scale estimates with explicit metric-reference evidence.

No exporter or asset is changed here. A fitted card's metric pose alone does
not identify the scale of a separately reconstructed hand. The card branch
requires caller-verified correspondence to independent source geometry.
Absent an independently linked reference or an explicit subject measurement,
metric scale is unknown. Neutral MANO geometry alone is a model prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

MIDDLE_TIP_VERTEX = 443  # Installed official smplx.vertex_ids['mano']['middle'].


def _positive(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: positive finite scalar required")
    return float(value)


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: finite array {shape} required")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name}: nonempty text required")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name}: SHA256 required")
    return value


def _evidence(value, name):
    if not isinstance(value, dict): raise ValueError(f"{name}: evidence object required")
    _text(value.get("description"), name + ".description")
    _hash(value.get("sha256"), name + ".sha256")
    return dict(value)


@dataclass(frozen=True)
class CardPnPScaleAnchor:
    """A card-local reference matched to independent source-unit geometry.

    same_segment uses two corresponding endpoints and their length ratio.
    same_camera_point uses one identical physical point, in the same known
    camera-origin basis, and checks scalar alignment. Proximity is insufficient.
    Evidence hashes identify caller-reviewed artifacts; this numerical module
    cannot itself prove their physical assertions. PnP fit residuals remain fit
    diagnostics, not independent metric validation.
    """
    frame: int
    view_id: str
    source_video_sha256: str
    scale_group: str
    pose: dict
    candidate_id: str
    source_points_units: np.ndarray
    card_points_m: np.ndarray
    correspondence_kind: str
    source_geometry_sha256: str
    source_geometry_method: str
    source_geometry_is_independent: bool
    source_camera_frame: str
    metric_camera_frame: str
    correspondence_evidence: dict
    full_face_evidence: dict
    camera_evidence: dict
    card_dimensions_evidence: dict


def evaluate_card_anchor(anchor: CardPnPScaleAnchor, *, max_point_relative_residual=.02) -> dict:
    """Return an accepted/rejected ratio; malformed API inputs raise ValueError."""
    if not isinstance(anchor, CardPnPScaleAnchor): raise ValueError("CardPnPScaleAnchor required")
    if isinstance(anchor.frame, bool) or not isinstance(anchor.frame, int) or anchor.frame < 0: raise ValueError("Nonnegative integer frame required")
    for name in ("view_id", "scale_group", "candidate_id", "source_geometry_method", "source_camera_frame", "metric_camera_frame"):
        _text(getattr(anchor, name), name)
    _hash(anchor.source_video_sha256, "source_video_sha256")
    _hash(anchor.source_geometry_sha256, "source_geometry_sha256")
    for name in ("correspondence_evidence", "full_face_evidence", "camera_evidence", "card_dimensions_evidence"):
        _evidence(getattr(anchor, name), name)
    if not isinstance(anchor.source_geometry_is_independent, bool): raise ValueError("Explicit source independence bool required")
    tolerance = _positive(max_point_relative_residual, "max_point_relative_residual")
    if anchor.correspondence_kind not in ("same_segment", "same_camera_point"):
        raise ValueError("correspondence_kind must be same_segment or same_camera_point")
    size = 2 if anchor.correspondence_kind == "same_segment" else 1
    points = _array(anchor.source_points_units, (size, 3), "source_points_units")
    card_points = _array(anchor.card_points_m, (size, 3), "card_points_m")
    reasons = []
    pose = anchor.pose
    if not isinstance(pose, dict): raise ValueError("pose must be CardPoseResult.to_dict()")
    for flag in ("accepted", "calibrated", "full_face_verified", "scale_anchor_eligible"):
        if pose.get(flag) is not True: reasons.append("card_pose_" + flag + "_not_verified")
    allowed = {"independent_scene_reconstruction", "mano_verified_contact_correspondence", "independently_measured_reference_geometry"}
    if not anchor.source_geometry_is_independent or anchor.source_geometry_method not in allowed:
        reasons.append("circular_or_unverified_source_geometry")
    if anchor.camera_evidence.get("calibrated") is not True:
        reasons.append("missing_independently_calibrated_camera_evidence")
    if anchor.correspondence_evidence.get("physical_correspondence_verified") is not True:
        reasons.append("physical_correspondence_not_verified")
    if anchor.full_face_evidence.get("physical_complete_face_verified") is not True:
        reasons.append("physical_full_face_not_verified")
    if anchor.card_dimensions_evidence.get("dimensions_apply_to_this_card_verified") is not True:
        reasons.append("nominal_dimensions_not_verified_for_this_card")
    matches = [c for c in pose.get("candidates", []) if c.get("candidate_id") == anchor.candidate_id]
    if len(matches) != 1 or matches[0].get("valid") is not True:
        reasons.append("missing_or_invalid_selected_pnp_candidate")
    # pose.scale_anchor_eligible describes the near-best candidates' depth
    # spread, not every solver output. Do not reuse that qualification for a
    # distant valid solution, even if its own point-scale fit looks perfect.
    near_best = pose.get("ambiguities", {}).get("near_best_candidate_ids", [])
    if not isinstance(near_best, list) or anchor.candidate_id not in near_best:
        reasons.append("selected_pnp_candidate_outside_scale_qualified_near_best_set")
    result = {"frame": anchor.frame, "view_id": anchor.view_id, "source_video_sha256": anchor.source_video_sha256,
        "scale_group": anchor.scale_group, "candidate_id": anchor.candidate_id,
        "correspondence_kind": anchor.correspondence_kind, "source_geometry_sha256": anchor.source_geometry_sha256,
        "source_geometry_method": anchor.source_geometry_method,
        "source_geometry_is_independent": anchor.source_geometry_is_independent,
        "scale_qualified_near_best_candidate_ids": near_best,
        "source_camera_frame": anchor.source_camera_frame, "metric_camera_frame": anchor.metric_camera_frame,
        "source_points_units": points.tolist(), "card_points_m": card_points.tolist(),
        "evidence": {name: getattr(anchor, name) for name in ("correspondence_evidence", "full_face_evidence", "camera_evidence", "card_dimensions_evidence")},
        "accepted": False, "meters_per_source_unit": None, "rejection_reasons": reasons,
        "independent_metric_accuracy_measured": False}
    if reasons: return result
    chosen = matches[0]
    dimensions = _array(pose.get("dimensions_m"), (2,), "card dimensions")
    if np.any(dimensions <= 0) or np.any(np.abs(card_points[:, :2]) > dimensions / 2 + 1e-9) or np.any(np.abs(card_points[:, 2]) > 1e-9):
        raise ValueError("Reference points must lie on the verified rectangular card face")
    rotation = _array(chosen["rotation_matrix"], (3, 3), "PnP rotation")
    translation = _array(chosen["tvec_m"], (3,), "PnP translation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError("PnP rotation must be proper orthogonal")
    metric = card_points @ rotation.T + translation
    result["metric_points_m"] = metric.tolist()
    result["input_corner_reprojection_rmse_px"] = chosen.get("input_corner_reprojection_rmse_px")
    if np.any(metric[:, 2] <= 0): reasons.append("reference_behind_camera")
    if anchor.correspondence_kind == "same_segment":
        source_length = float(np.linalg.norm(points[1] - points[0]))
        metric_length = float(np.linalg.norm(metric[1] - metric[0]))
        result.update(source_reference_length_units=source_length, metric_reference_length_m=metric_length)
        if min(source_length, metric_length) <= 1e-9: reasons.append("degenerate_reference_segment")
        ratio = metric_length / source_length if source_length > 1e-9 else None
        result["relative_vector_fit_residual"] = None
    else:
        if anchor.source_camera_frame != anchor.metric_camera_frame:
            reasons.append("same_point_requires_same_camera_origin_and_basis")
        denominator = float(np.sum(points * points))
        ratio = float(np.sum(points * metric) / denominator) if denominator > 1e-18 else None
        residual = float(np.linalg.norm(ratio * points - metric) / max(np.linalg.norm(metric), 1e-12)) if ratio is not None else None
        result["relative_vector_fit_residual"] = residual
        if residual is None: reasons.append("degenerate_camera_point")
        elif residual > tolerance: reasons.append("point_not_related_by_uniform_origin_scale")
    if ratio is None or not np.isfinite(ratio) or ratio <= 0: reasons.append("nonpositive_or_nonfinite_ratio")
    if not reasons: result.update(accepted=True, meters_per_source_unit=ratio)
    return result


def measure_neutral_mano_hand(arrays: dict, betas, *, scale_group: str, model_sha256: str,
                              shape_source_sha256: str, side="right") -> dict:
    """Measure the actual identity-rotation MANO rest, not a posed-frame chord.

    v=v_template+shapedirs*betas, J=J_regressor*v. With identity rotation matrices
    MANOLayer adds no hand mean, pose blend offsets are zero and LBS is identity.
    Endpoint is MANO wrist joint center to official middle fingertip vertex443.
    This anatomical convention is not a measured wrist-crease calibration.
    """
    _text(scale_group, "scale_group"); _hash(model_sha256, "model_sha256"); _hash(shape_source_sha256, "shape_source_sha256")
    if side not in ("left", "right"): raise ValueError("side must be left/right")
    required = ("v_template", "shapedirs", "J_regressor", "weights", "kintree_table")
    if any(k not in arrays for k in required): raise ValueError("Actual neutral MANO geometry unavailable; required model arrays missing")
    template = _array(arrays["v_template"], (778, 3), "v_template")
    shapedirs = _array(arrays["shapedirs"], (778, 3, 10), "shapedirs")
    regressor = _array(arrays["J_regressor"], (16, 778), "J_regressor")
    weights = _array(arrays["weights"], (778, 16), "weights")
    shape = _array(betas, (10,), "shared_betas")
    parents = np.asarray(arrays["kintree_table"])
    if parents.shape != (2, 16) or parents[0, 4] != 0 or parents[0, 5] != 4 or parents[0, 6] != 5:
        raise ValueError("MANO middle-finger hierarchy differs from reviewed model")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1., atol=1e-7, rtol=0):
        raise ValueError("Invalid MANO skinning weights; identity-rest equivalence is not established")
    vertices = template + np.tensordot(shapedirs, shape, axes=(2, 0))
    joints = regressor @ vertices
    reflection = np.diag([-1., 1., 1.]) if side == "left" else np.eye(3)
    chain = np.vstack((joints[[0, 4, 5, 6]], vertices[[MIDDLE_TIP_VERTEX]])) @ reflection.T
    edges = np.diff(chain, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    chord = float(np.linalg.norm(chain[-1] - chain[0]))
    if min(float(lengths.min()), chord) <= 1e-9: raise ValueError("Degenerate neutral hand geometry")
    return {"scale_group": scale_group, "side": side, "length_source_units": chord,
        "definition": "Euclidean MANO wrist joint0 center to middle fingertip vertex443 in shared-shape identity-rotation extended neutral rest; not a bent video frame or palm-only length",
        "model_sha256": model_sha256, "shape_source_sha256": shape_source_sha256, "shared_betas": shape.tolist(),
        "neutral_rotation_matrices": "all 16 identity; MANOLayer pose2rot=False; hand_mean not added",
        "geometry_source": "actual neutral shaped MANO arrays", "is_video_frame_measurement": False,
        "provenance": {"kind": "inferred", "geometry_units": "mano_model_source_units",
                       "basis": "MANO model and image-inferred shared shape; not this person's measured hand length",
                       "uncertainty": None},
        "subject_hand_length_m": None,
        "wrist_point_units": chain[0].tolist(), "middle_tip_point_units": chain[-1].tolist(),
        "middle_chain_points_units": chain.tolist(), "wrist_to_tip_chain_length_units": float(lengths.sum()),
        "middle_mcp_to_tip_chord_over_chain": float(np.linalg.norm(chain[-1] - chain[1]) / lengths[1:].sum()),
        "neutral_is_perfectly_collinear": False,
        "left_convention": "canonical right shaped geometry reflected in X" if side == "left" else None,
        "anthropometric_wrist_crease_equivalence_validated": False}


@dataclass
class ScaleEstimate:
    anchor_method: str | None
    scale_group: str
    meters_per_source_unit: float | None
    nominal_meters_per_source_unit: float | None
    scale_confidence: str
    anchor_frames: list[int] = field(default_factory=list)
    card_anchor_evaluations: list[dict] = field(default_factory=list)
    measurements: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def global_scale_factor(self):
        # One is an identity on model coordinates, never a measured metre.
        if self.meters_per_source_unit is None:
            return 1.0
        return self.meters_per_source_unit / (self.nominal_meters_per_source_unit or 1.0)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return dict(asdict(self), global_scale_factor=self.global_scale_factor,
            numeric_confidence=None, confidence_semantics="Evidence category, not calibrated probability or measured error",
            independent_metric_accuracy_measured=False, relative_metric_error=None, ten_percent_accuracy_verified=False,
            application=("Identity on model-source geometry only; metric scale is unknown and no metre conversion is authorized."
                         if self.meters_per_source_unit is None else
                         "One factor for the entire named source scale group about its camera origin, including root translations and local bone/mesh geometry. Already metric PnP outputs must not be multiplied again."))


def estimate_scene_scale(*, scale_group: str, card_anchors=(), neutral_hand_measurements=(),
                          measured_hand_length_m=None, measurement_provenance=None,
                          nominal_meters_per_source_unit=None) -> ScaleEstimate:
    """Card ratios -> explicitly measured subject length -> unknown scale.

    Duplicated cards/references in one frame first contribute one frame median.
    No fit-error weighting or adaptive clipping claims statistical accuracy.
    Median absolute deviation and >3 scaled-MAD/5% deviations are diagnostic.
    All inputs must belong to one globally shared scale group.
    """
    _text(scale_group, "scale_group")
    nominal = None if nominal_meters_per_source_unit is None else _positive(nominal_meters_per_source_unit, "nominal_meters_per_source_unit")
    target = None if measured_hand_length_m is None else _positive(measured_hand_length_m, "measured_hand_length_m")
    if target is not None:
        measurement_provenance = _evidence(measurement_provenance, "measurement_provenance")
        if measurement_provenance.get("kind") != "user_measured":
            raise ValueError("Explicit subject measurement requires provenance.kind=user_measured")
    elif measurement_provenance is not None:
        raise ValueError("Measurement provenance supplied without a measured hand length")
    evaluations = []
    grouped = {}
    for anchor in card_anchors:
        value = evaluate_card_anchor(anchor)
        if anchor.scale_group != scale_group:
            value["accepted"] = False; value["meters_per_source_unit"] = None
            value["rejection_reasons"].append("different_source_scale_group")
        evaluations.append(value)
        if value["accepted"]: grouped.setdefault((anchor.source_video_sha256, anchor.view_id, anchor.frame), []).append(value["meters_per_source_unit"])
    if grouped:
        by_frame = [{"source_video_sha256": k[0], "view_id": k[1], "frame": k[2], "meters_per_source_unit": float(np.median(v)), "reference_count": len(v)} for k, v in sorted(grouped.items())]
        values = np.array([r["meters_per_source_unit"] for r in by_frame])
        factor = float(np.median(values)); mad = float(np.median(np.abs(values - factor)))
        threshold = max(.05 * factor, 3 * 1.4826 * mad)
        diagnostics = {"aggregation": "median of per-frame medians; all accepted ratios retained",
            "per_frame_ratios": by_frame, "median_absolute_deviation": mad,
            "outlier_diagnostic_absolute_threshold": threshold,
            "outlier_diagnostic_frames": [r for r in by_frame if abs(r["meters_per_source_unit"] - factor) > threshold],
            "outlier_diagnostics_are_accuracy_bounds": False}
        warnings = ["Scale is conditional on caller-verified card size, complete face, calibrated camera and independent physical correspondence; PnP's own fit does not validate metric accuracy"]
        if len(by_frame) < 3: warnings.append("Fewer than three anchor frames; median has limited outlier resistance")
        return ScaleEstimate("card_pnp", scale_group, factor, nominal, "medium", sorted({r["frame"] for r in by_frame}), evaluations, [], diagnostics, warnings,
            {"kind": "inferred", "basis": "Independently linked card geometry with supplied measurement/calibration evidence", "uncertainty": None})
    measures = list(neutral_hand_measurements)
    if target is None:
        return ScaleEstimate(None, scale_group, None, nominal, "unknown", [], evaluations, measures,
            {"reason": "No accepted independently linked metric reference or explicit subject length",
             "identity_factor_is_metric_measurement": False},
            ["Metric scale is unobservable from the supplied evidence; retained neutral geometry is model-inferred, not a measured personal length."],
            {"kind": "unobservable", "geometry_units": "mano_model_source_units", "uncertainty": None,
             "basis": "No supplied metric reference ties this model scale group to the subject"})
    if not measures: raise ValueError("Measured subject length requires actual neutral MANO geometry for a defined correspondence")
    for item in measures:
        if item.get("scale_group") != scale_group: raise ValueError("Neutral hand geometry belongs to another scale group")
        if item.get("geometry_source") != "actual neutral shaped MANO arrays" or item.get("is_video_frame_measurement") is not False:
            raise ValueError("Subject-length mapping requires actual neutral model geometry, not a bent-frame distance or guessed length")
        _positive(item["length_source_units"], "neutral length")
        _hash(item["model_sha256"], "neutral model_sha256"); _hash(item["shape_source_sha256"], "shape_source_sha256")
    ratios = [target / item["length_source_units"] for item in measures]
    factor = float(np.median(ratios))
    return ScaleEstimate("user_measured", scale_group, factor, nominal, "medium", [], evaluations, measures,
        {"measured_hand_length_m": target, "measurement_provenance": measurement_provenance,
         "aggregation": "median of explicit subject length / neutral model length ratios", "per_hand_ratios": ratios,
         "mirrored_shared_shape_hands_are_independent_measurements": False},
        ["No usable independently linked full-card metric anchor",
         "Subject length is a caller measurement assertion; acquisition, model correspondence and metric accuracy are not independently verified",
         "MANO wrist-joint center differs from an anthropometric wrist crease; no subject-specific correction is known",
         "Scaling all 3D geometry preserves projection for the chosen camera but cannot calibrate a wrong camera model"],
        dict(measurement_provenance, uncertainty=measurement_provenance.get("uncertainty")))


def scale_reconstruction_geometry(estimate: ScaleEstimate, *, scale_group: str, hand_roots_units,
                                  hand_local_joints_units, hand_local_vertices_units=None,
                                  packet_positions_units=None, packet_local_vertices_units=None,
                                  packet_space=None) -> dict[str, np.ndarray]:
    """Return new metre arrays; rotations, normals and camera intrinsics stay fixed.

    Roots are absolute positions about the common camera origin. Joint/mesh
    coordinates are root-relative; both must be scaled together. Packet inputs
    require explicit confirmation of the same *unscaled* group. Already metric
    PnP positions/dimensions belong outside this conversion.
    This helper does not update export JSON, bind meshes, inverse binds or UE.
    """
    if not isinstance(estimate, ScaleEstimate) or scale_group != estimate.scale_group: raise ValueError("Scale group differs")
    if estimate.meters_per_source_unit is None:
        raise ValueError("Metric scale unavailable; preserve relative model arrays instead of exporting metre arrays")
    factor = _positive(estimate.meters_per_source_unit, "meters_per_source_unit")
    roots, joints = np.asarray(hand_roots_units, dtype=float), np.asarray(hand_local_joints_units, dtype=float)
    if roots.ndim < 1 or roots.shape[-1] != 3 or joints.ndim < 2 or joints.shape[-1] != 3 or joints.shape[:-2] != roots.shape[:-1]:
        raise ValueError("Roots (...,3) and local joints (...,J,3) must have matching leading shapes")
    if not np.isfinite(roots).all() or not np.isfinite(joints).all(): raise ValueError("Finite hand geometry required")
    result = {"hand_roots_m": roots * factor, "hand_local_joints_m": joints * factor,
              "hand_joints_camera_m": (roots[..., None, :] + joints) * factor}
    if hand_local_vertices_units is not None:
        vertices = np.asarray(hand_local_vertices_units, dtype=float)
        if vertices.ndim < 2 or vertices.shape[:-2] != roots.shape[:-1] or vertices.shape[-1] != 3 or not np.isfinite(vertices).all(): raise ValueError("Finite local vertices with matching hand leading shapes required")
        result["hand_local_vertices_m"] = vertices * factor
    if packet_positions_units is not None or packet_local_vertices_units is not None:
        if packet_space != "same_unscaled_camera_scale_group": raise ValueError("Packet geometry requires explicit same unscaled space; do not scale metric PnP again")
        for name, value in (("packet_positions", packet_positions_units), ("packet_local_vertices", packet_local_vertices_units)):
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim < 1 or array.shape[-1] != 3 or not np.isfinite(array).all(): raise ValueError("Finite packet (...,3) geometry required")
                result[name + "_m"] = array * factor
    return result
