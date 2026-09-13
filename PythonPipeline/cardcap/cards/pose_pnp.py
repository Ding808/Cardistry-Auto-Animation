"""Rectangular card IPPE candidates in OpenCV camera coordinates, metres.

Uses SOLVEPNP_IPPE, never the square-only SOLVEPNP_IPPE_SQUARE. Reprojection
residuals are fit residuals against input corners, not independent accuracy.
Unlabelled masks retain cyclic, face and planar-pose ambiguities for tracking.
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .quad_fit import QuadFitResult, fit_quad, validate_quad


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    calibrated: bool
    provenance: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera.matrix: expected finite (3, 3)")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0):
            raise ValueError("camera.matrix: positive focal lengths and bottom row [0, 0, 1] required")
        if abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12:
            raise ValueError("camera.matrix: OpenCV pinhole model requires zero skew/off-axis terms")
        if distortion.ndim > 2 or (distortion.ndim == 2 and min(distortion.shape) != 1) or distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("camera.distortion: finite vector with 0/4/5/8/12/14 coefficients required")
        if not isinstance(self.calibrated, bool):
            raise ValueError("camera.calibrated: explicit bool required")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("camera.provenance: nonempty description required")
        object.__setattr__(self, "matrix", np.ascontiguousarray(matrix))
        object.__setattr__(self, "distortion", np.ascontiguousarray(distortion.reshape(-1)))

    def to_dict(self) -> dict[str, Any]:
        return {"matrix": self.matrix.tolist(), "distortion": self.distortion.tolist(),
                "calibrated": self.calibrated, "provenance": self.provenance}


@dataclass
class PoseCandidate:
    candidate_id: str
    corner_indices: list[int]
    ippe_solution_index: int
    rvec: list[float] | None
    tvec_m: list[float] | None
    rotation_matrix: list[list[float]] | None
    depths_m: list[float] | None
    input_corner_reprojection_rmse_px: float | None
    input_corner_reprojection_max_px: float | None
    opencv_coordinate_rmse_px: float | None
    positive_depth: bool
    valid: bool
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class CardPoseResult:
    candidates: list[PoseCandidate]
    best_candidate_id: str | None
    accepted: bool
    calibrated: bool
    full_face_verified: bool
    scale_anchor_eligible: bool
    ambiguities: dict[str, Any]
    warnings: list[str]
    dimensions_m: list[float]
    dimensions_provenance: dict = field(default_factory=dict)
    solver: str = "cv2.solvePnPGeneric/SOLVEPNP_IPPE"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(coordinates="OpenCV camera: X right, Y down, Z forward; translation metres",
                      transform="camera_point = rotation_matrix @ card_point + tvec_m",
                      object_corner_order="[-w/2,-h/2,0],[w/2,-h/2,0],[w/2,h/2,0],[-w/2,h/2,0]",
                      reprojection_metric="RMS of per-corner Euclidean pixel distances to fitted input corners; not independently annotated accuracy",
                      independent_corner_validation_error_px=None,
                      physical_scale_ground_truth_validated=False,
                      dimensions_source="explicit caller dimensions and recorded provenance; no standard-card default")
        return result


def card_object_points(width_m: float, height_m: float) -> np.ndarray:
    if isinstance(width_m, bool) or isinstance(height_m, bool) or not np.isfinite([width_m, height_m]).all() or min(width_m, height_m) <= 0:
        raise ValueError("card dimensions: positive finite metres required")
    return np.array([[-width_m / 2, -height_m / 2, 0],
                     [width_m / 2, -height_m / 2, 0],
                     [width_m / 2, height_m / 2, 0],
                     [-width_m / 2, height_m / 2, 0]], dtype=np.float64)


def estimate_card_pose(quad: QuadFitResult, camera: CameraModel, *,
                       width_m: float, height_m: float, dimensions_provenance: dict,
                       known_corner_order: bool = False,
                       max_reprojection_rmse_px: float = 3.0,
                       ambiguity_tolerance_px: float = 0.5,
                       max_anchor_relative_depth_spread: float = 0.02) -> CardPoseResult:
    """Return all IPPE candidates; best fit is a proposal, not a temporal identity.

    With known_corner_order=True the caller asserts quad corners correspond
    exactly to card_object_points. Otherwise all four cyclic starts and both
    windings are retained. The 180-degree in-plane and front/back semantic
    identities cannot be established from an unlabelled rectangle alone.
    """
    if not isinstance(quad, QuadFitResult) or not isinstance(camera, CameraModel):
        raise ValueError("quad and camera must be QuadFitResult and CameraModel")
    if not isinstance(known_corner_order, bool):
        raise ValueError("known_corner_order: explicit bool required")
    for name, value in (("max_reprojection_rmse_px", max_reprojection_rmse_px),
                        ("ambiguity_tolerance_px", ambiguity_tolerance_px),
                        ("max_anchor_relative_depth_spread", max_anchor_relative_depth_spread)):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name}: finite nonnegative value required")
    objects = card_object_points(width_m, height_m)
    provenance = _validate_dimensions_provenance(dimensions_provenance)
    warnings = ["input-corner reprojection residual is a fit diagnostic, not independent image or metric ground truth"]
    if not camera.calibrated:
        warnings.append("camera is uncalibrated; metric distance depends on assumed intrinsics and is not a scale anchor")
    if not quad.full_face_verified:
        warnings.append("complete physical card face is not verified; not a scale anchor")
    if provenance["kind"] == "inferred":
        warnings.append("Card dimensions are an explicit inferred condition, not measured dimensions; no metric scale-anchor eligibility")
    ambiguity: dict[str, Any] = {"corner_correspondence_known": known_corner_order,
                                "in_plane_180_degree_semantic_ambiguity": not known_corner_order,
                                "front_back_semantic_ambiguity": not known_corner_order,
                                "near_best_candidate_ids": [],
                                "near_best_relative_depth_spread": None,
                                "planar_pose_ambiguity": False}
    result = CardPoseResult([], None, False, camera.calibrated, quad.full_face_verified,
                            False, ambiguity, warnings, [width_m, height_m], provenance)
    if not quad.accepted or quad.corners_px is None:
        result.warnings.append("quad rejected: " + ", ".join(quad.rejection_reasons))
        return result
    corners = validate_quad(quad.corners_px)
    orders = [list(range(4))] if known_corner_order else [
        [(start + direction * index) % 4 for index in range(4)]
        for direction in (1, -1) for start in range(4)]
    for order_index, order in enumerate(orders):
        pixels = np.ascontiguousarray(corners[order], dtype=np.float64)
        try:
            _, rotations, translations, cv_errors = cv2.solvePnPGeneric(
                objects, pixels, camera.matrix, camera.distortion,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error as exc:
            result.warnings.append(f"IPPE correspondence {order_index} failed: {exc}")
            continue
        for solution, (rvec, tvec) in enumerate(zip(rotations, translations)):
            candidate = PoseCandidate(f"order_{order_index}_ippe_{solution}", order, solution,
                                      None, None, None, None, None, None, None, False, False)
            result.candidates.append(candidate)
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                candidate.rejection_reasons.append("nonfinite_solver_result")
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            translation = tvec.reshape(3)
            camera_points = objects @ rotation.T + translation
            if not np.isfinite(rotation).all() or not np.isfinite(camera_points).all():
                candidate.rejection_reasons.append("nonfinite_derived_transform")
                continue
            candidate.rvec = rvec.reshape(3).tolist()
            candidate.tvec_m = translation.tolist()
            candidate.rotation_matrix = rotation.tolist()
            candidate.depths_m = camera_points[:, 2].tolist()
            candidate.positive_depth = bool(np.all(camera_points[:, 2] > 1e-8))
            if not candidate.positive_depth:
                candidate.rejection_reasons.append("one_or_more_corners_not_in_front_of_camera")
            projected, _ = cv2.projectPoints(objects, rvec, tvec, camera.matrix, camera.distortion)
            errors = np.linalg.norm(projected.reshape(4, 2) - pixels, axis=1)
            if not np.isfinite(errors).all():
                candidate.rejection_reasons.append("nonfinite_reprojection")
                continue
            candidate.input_corner_reprojection_rmse_px = float(np.sqrt(np.mean(errors ** 2)))
            candidate.input_corner_reprojection_max_px = float(errors.max())
            if cv_errors is not None and solution < len(cv_errors):
                cv_error = float(np.asarray(cv_errors[solution]).reshape(-1)[0])
                candidate.opencv_coordinate_rmse_px = cv_error if np.isfinite(cv_error) else None
            if candidate.input_corner_reprojection_rmse_px > max_reprojection_rmse_px:
                candidate.rejection_reasons.append("input_corner_fit_residual_above_limit")
            candidate.valid = not candidate.rejection_reasons
    result.candidates.sort(key=lambda c: (not c.valid,
                                          float("inf") if c.input_corner_reprojection_rmse_px is None else c.input_corner_reprojection_rmse_px,
                                          c.candidate_id))
    valid = [candidate for candidate in result.candidates if candidate.valid]
    if not valid:
        result.warnings.append("no positive-depth finite candidate within input-corner residual limit")
        return result
    best = valid[0]
    result.best_candidate_id = best.candidate_id
    result.accepted = True
    near = [candidate for candidate in valid if candidate.input_corner_reprojection_rmse_px <=
            best.input_corner_reprojection_rmse_px + ambiguity_tolerance_px]
    depths = np.array([candidate.tvec_m[2] for candidate in near])
    spread = float(np.ptp(depths) / best.tvec_m[2])
    ambiguity.update(near_best_candidate_ids=[candidate.candidate_id for candidate in near],
                     near_best_relative_depth_spread=spread,
                     candidate_count=len(result.candidates), valid_candidate_count=len(valid))
    # Compare plane normals modulo sign to exclude rectangle front/back identity.
    normal = np.asarray(best.rotation_matrix)[:, 2]
    angles = [float(np.arccos(np.clip(abs(np.dot(normal, np.asarray(c.rotation_matrix)[:, 2])), 0, 1))) for c in near]
    ambiguity["near_best_max_plane_normal_angle_rad"] = max(angles)
    ambiguity["planar_pose_ambiguity"] = bool(max(angles) > np.deg2rad(1.0) or spread > 0.001)
    result.scale_anchor_eligible = bool(camera.calibrated and quad.full_face_verified and provenance["kind"] != "inferred" and
                                        spread <= max_anchor_relative_depth_spread)
    if not known_corner_order:
        result.warnings.append("unlabelled rectangle retains corner/180-degree/front-back identities; tracker must select consistently")
    if ambiguity["planar_pose_ambiguity"]:
        result.warnings.append("near-equal residual planar pose alternatives retained; do not treat best residual as unique pose")
    if result.scale_anchor_eligible:
        result.warnings.append("anchor eligibility is conditional on supplied calibration, visibility evidence and card dimensions; it is not independent scale validation")
    return result


def _validate_dimensions_provenance(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("kind") not in ("user_measured", "calibrated", "inferred"):
        raise ValueError("dimensions_provenance.kind must identify user_measured/calibrated/inferred dimensions")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ValueError("dimensions_provenance requires a measurement/inference description")
    digest = value.get("source_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("dimensions_provenance.source_sha256 is required")
    # Preserve explicit uncertainty and any acquisition/conditional assumptions.
    return dict(value, uncertainty=value.get("uncertainty"))


def _dimensions_from_json(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    width, height = data["width_m"], data["height_m"]
    card_object_points(width, height)
    provenance = _validate_dimensions_provenance(data["provenance"])
    provenance.update(input_file=str(path.resolve()), input_file_sha256=hashlib.sha256(raw).hexdigest())
    return {"width_m": width, "height_m": height, "dimensions_provenance": provenance}


def _camera_from_json(path: Path) -> CameraModel:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    camera = data.get("camera", data)
    if "matrix" in camera:
        matrix = camera["matrix"]
    else:
        intrinsics = camera["intrinsics"]
        matrix = [[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]]
    return CameraModel(matrix, camera["distortion"], camera["calibrated"],
                       camera.get("provenance") or f"explicit input {path.resolve()} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility", choices=("unknown", "partial", "full"), default="unknown")
    parser.add_argument("--visibility-evidence")
    parser.add_argument("--known-corner-order", action="store_true")
    parser.add_argument("--dimensions-json", type=Path, required=True,
                        help="Explicit width_m, height_m and provenance {kind, description, source_sha256, uncertainty}")
    parser.add_argument("--image", type=Path, help="optional original frame for diagnostic overlay")
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        parser.error(f"could not read mask: {args.mask}")
    try:
        camera = _camera_from_json(args.camera_json)
        quad = fit_quad(mask, visibility=args.visibility, visibility_evidence=args.visibility_evidence)
        pose = estimate_card_pose(quad, camera, **_dimensions_from_json(args.dimensions_json),
                                  known_corner_order=args.known_corner_order)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    report = {"diagnostic_version": "m3-card-geometry-1", "opencv_version": cv2.__version__,
              "mask_path": str(args.mask.resolve()), "mask_sha256": hashlib.sha256(args.mask.read_bytes()).hexdigest(),
              "camera": camera.to_dict(), "quad": quad.to_dict(), "pose": pose.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if args.overlay:
        canvas = cv2.imread(str(args.image)) if args.image else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if canvas is None or canvas.shape[:2] != mask.shape:
            parser.error("overlay image must be readable and have the same resolution as the mask")
        if quad.corners_px is not None:
            points = np.rint(quad.corners_px).astype(np.int32)
            cv2.polylines(canvas, [points], True, (0, 220, 255), 2)
            for index, point in enumerate(points):
                cv2.circle(canvas, tuple(point), 4, (0, 220, 255), -1)
                cv2.putText(canvas, str(index), tuple(point + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
        text = f"quad={quad.accepted} visibility={quad.visibility} calibrated={camera.calibrated} anchor={pose.scale_anchor_eligible}"
        cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 220, 255), 1)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.overlay), canvas):
            parser.error(f"could not write overlay: {args.overlay}")
    print(json.dumps({"accepted": pose.accepted, "candidate_count": len(pose.candidates),
                      "scale_anchor_eligible": pose.scale_anchor_eligible, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
