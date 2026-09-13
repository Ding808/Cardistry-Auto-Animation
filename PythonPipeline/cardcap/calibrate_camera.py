"""Explicit chessboard camera calibration; no videos or downloads are searched.

User images must match the capture's camera, lens settings and pixel geometry.
Training fit residuals, held-out corner errors and independent length checks
are distinct. Synthetic fixtures never export an ingest calibration file.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import traceback

import cv2
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Chessboard:
    columns: int
    rows: int
    square_mm: float

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 3 for x in (self.columns, self.rows)):
            raise ValueError("Chessboard columns/rows are inner-corner counts, each >=3")
        if isinstance(self.square_mm, bool) or not np.isfinite(self.square_mm) or self.square_mm <= 0:
            raise ValueError("Physical square edge length must be positive finite millimetres")

    def object_points(self):
        points = np.zeros((self.columns * self.rows, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        return points * (self.square_mm / 1000.)


def _points(value, board):
    array = np.asarray(value, np.float32)
    if array.shape != (board.columns * board.rows, 2) or not np.isfinite(array).all():
        raise ValueError("Complete finite inner-corner array required")
    return np.ascontiguousarray(array)


def _view_distance(a, b, board, resolution):
    first = _points(a, board).reshape(board.rows, board.columns, 2)
    second = _points(b, board).reshape(first.shape)
    # The unmarked board may have an equivalent reversed enumeration.
    return min(float(np.sqrt(np.mean(np.sum((first - candidate) ** 2, axis=2))))
               for candidate in (second, second[::-1], second[:, ::-1], second[::-1, ::-1])) / math.hypot(*resolution)


def _image_paths(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir(): raise ValueError(f"Image directory does not exist: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
    if not paths: raise ValueError(f"No supplied still images in {folder}")
    return paths


def collect_chessboards(paths, board: Chessboard, *, subset, expected_resolution=None, debug_dir=None):
    """Detect full boards on unchanged stored pixels; failures stay in inventory."""
    records, resolution = [], expected_resolution
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for index, path in enumerate(paths):
        path = Path(path).resolve(); data = path.read_bytes()
        # Ignore EXIF rotation; no crop/resize/undistortion before calibration.
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if gray is None: raise ValueError(f"Cannot decode supplied image: {path}")
        size = [gray.shape[1], gray.shape[0]]
        if resolution is None: resolution = size
        if size != list(resolution): raise ValueError("All calibration and held-out images must have exactly one resolution")
        found, corners = cv2.findChessboardCornersSB(gray, (board.columns, board.rows), flags=flags)
        row = {"image_path": str(path), "image_sha256": hashlib.sha256(data).hexdigest(), "decoded_gray_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
            "subset": subset, "resolution": size, "detected": bool(found), "detector": "cv2.findChessboardCornersSB", "detector_flags": flags,
            "corner_confidence": None, "corners_px": _points(corners.reshape(-1, 2), board).tolist() if found else None,
            "rejection_reasons": [] if found else ["complete_requested_chessboard_not_detected"]}
        if debug_dir is not None:
            debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if found: cv2.drawChessboardCorners(debug, (board.columns, board.rows), corners, found)
            target = Path(debug_dir) / f"{index:04d}.png"; target.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", debug)
            if not ok: raise RuntimeError("Cannot encode corner diagnostic")
            target.write_bytes(encoded.tobytes()); row.update(debug_image=str(target.resolve()), debug_image_sha256=sha(target))
        records.append(row)
    return records, list(resolution)


def prepare_view_split(training, heldout, board, *, min_view_distance=.01):
    """Fix partitions before fitting; exclude redundant training views explicitly."""
    selected = []
    for row in training:
        if not row["detected"]: continue
        duplicate = next((p for p in selected if p["image_sha256"] == row["image_sha256"] or p["decoded_gray_sha256"] == row["decoded_gray_sha256"] or
                          _view_distance(p["corners_px"], row["corners_px"], board, row["resolution"]) < min_view_distance), None)
        if duplicate:
            row["rejection_reasons"].append("duplicate_or_near_duplicate_training_view")
            row["duplicate_of_sha256"] = duplicate["image_sha256"]
        else: selected.append(row)
    checked = []
    for row in heldout:
        # Leakage is an input error, not a reason to quietly delete bad validation.
        if any(row["image_sha256"] == other["image_sha256"] or row["decoded_gray_sha256"] == other["decoded_gray_sha256"] for other in training + checked):
            raise ValueError("Held-out images overlap training or duplicate another held-out image")
        if row["detected"] and any(other["detected"] and _view_distance(row["corners_px"], other["corners_px"], board, row["resolution"]) < min_view_distance for other in selected + checked):
            raise ValueError("Held-out view is nearly duplicated; supply an independently captured view")
        checked.append(row)
    return selected


def _residuals(objects, pixels, rvec, tvec, matrix, distortion):
    projection = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    error = np.linalg.norm(projection - np.asarray(pixels), axis=1)
    return {"corner_errors_px": error.tolist(), "rms_px": float(np.sqrt(np.mean(error ** 2))), "max_px": float(error.max())}


def fit_camera(views, board, *, min_views=10, min_normal_span_degrees=15., min_center_span_diagonal=.10, max_training_rms_px=2.):
    """Fit standard 5-coefficient pinhole calibration and report quality gates.

    Inputs are already partitioned; no residual-based view removal or automatic
    model search occurs. Gates are explicit heuristics, not accuracy guarantees.
    """
    if isinstance(min_views, bool) or not isinstance(min_views, int) or min_views < 10: raise ValueError("At least 10 distinct training views are required")
    if len(views) < min_views: raise ValueError(f"Need >= {min_views} detected, non-redundant training views; got {len(views)}")
    for value in (min_normal_span_degrees, min_center_span_diagonal, max_training_rms_px):
        if not np.isfinite(value) or value <= 0: raise ValueError("Positive finite quality gates required")
    resolution = views[0]["resolution"]
    if any(row["resolution"] != resolution for row in views): raise ValueError("Mixed calibration resolution")
    objects = board.object_points(); pixels = [_points(row["corners_px"], board) for row in views]
    values = cv2.calibrateCameraExtended([objects.copy() for _ in views], pixels, tuple(resolution), None, None,
        flags=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9))
    rms, matrix, distortion, rvecs, tvecs, std_i, std_e, per_view = values
    if any(not np.isfinite(x).all() for x in (matrix, distortion, std_i, std_e, per_view)) or not np.isfinite(rms):
        raise ValueError("Nonfinite calibration solution")
    normals, centers, measurements, reasons = [], [], [], []
    for row, pixel, rvec, tvec, reported in zip(views, pixels, rvecs, tvecs, per_view.reshape(-1)):
        rotation = cv2.Rodrigues(rvec)[0]
        if np.min((objects @ rotation.T + tvec.reshape(3))[:, 2]) <= 0: reasons.append("training_board_behind_camera")
        normals.append(rotation[:, 2]); centers.append(pixel.mean(axis=0))
        error = _residuals(objects, pixel, rvec, tvec, matrix, distortion)
        measurements.append(dict(image_sha256=row["image_sha256"], **error, opencv_rms_px=float(reported),
                                 rvec=rvec.reshape(3).tolist(), tvec_m=tvec.reshape(3).tolist()))
    angles = [math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1)))) for i, a in enumerate(normals) for b in normals[i + 1:]]
    center_span = max(float(np.linalg.norm(a - b)) for a in centers for b in centers) / math.hypot(*resolution)
    normal_span = max(angles)
    if normal_span < min_normal_span_degrees: reasons.append("insufficient_board_normal_diversity")
    if center_span < min_center_span_diagonal: reasons.append("insufficient_board_center_coverage")
    if rms > max_training_rms_px: reasons.append("training_rms_exceeds_declared_gate")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not (0 <= matrix[0, 2] < resolution[0] and 0 <= matrix[1, 2] < resolution[1]):
        reasons.append("nonpositive_focal_or_principal_point_outside_image")
    # Sample radial mapping derivative across the image corner ray radius.
    # This catches folding; it is not a formal bijectivity proof for the lens.
    k1, k2, p1, p2, k3 = distortion.reshape(5)
    rays = np.array([[0, 0], [resolution[0]-1, 0], [0, resolution[1]-1], np.array(resolution)-1], float)
    ray_r = np.linalg.norm((rays - matrix[:2, 2]) / [matrix[0, 0], matrix[1, 1]], axis=1)
    radius = np.linspace(0, max(ray_r), 256); q = radius ** 2
    derivative = 1 + 3*k1*q + 5*k2*q*q + 7*k3*q*q*q
    if derivative.min() <= 0: reasons.append("sampled_radial_mapping_folds_in_image_domain")
    all_pixels = np.concatenate(pixels)
    return {"accepted": not reasons, "rejection_reasons": sorted(set(reasons)), "resolution": resolution,
        "matrix": matrix.tolist(), "distortion": distortion.reshape(-1).tolist(), "training_rms_px": float(rms), "training_views": measurements,
        "std_intrinsics_opencv_order": std_i.reshape(-1).tolist(), "std_extrinsics_opencv_order": std_e.reshape(-1).tolist(),
        "uncertainty_semantics": "Local solver standard deviations, not independent physical accuracy",
        "coverage": {"normal_span_degrees": normal_span, "center_span_image_diagonal": center_span,
                     "detected_corner_bounds_px": [all_pixels.min(axis=0).tolist(), all_pixels.max(axis=0).tolist()],
                     "radial_mapping_min_sampled_derivative": float(derivative.min())},
        "gates": dict(min_views=min_views, min_normal_span_degrees=min_normal_span_degrees,
                      min_center_span_diagonal=min_center_span_diagonal, max_training_rms_px=max_training_rms_px),
        "training_residual_is_independent_validation": False, "relative_metric_error": None, "ten_percent_scale_accuracy_verified": False}


def validate_heldout_views(views, board, fit):
    """Freeze K/D; preassigned corner subset estimates only each board's R/t."""
    if not fit["accepted"]: raise ValueError("Rejected calibration cannot validate held-out views")
    training_hashes = {row["image_sha256"] for row in fit["training_views"]}
    if any(row["image_sha256"] in training_hashes for row in views):
        raise ValueError("Held-out view was used to fit this calibration")
    matrix, distortion = np.array(fit["matrix"]), np.array(fit["distortion"])
    objects = board.object_points(); rr, cc = np.indices((board.rows, board.columns))
    check = ((rr + cc) % 3 == 0).ravel(); pose_indices = np.flatnonzero(~check); check_indices = np.flatnonzero(check)
    result = {"status": "not_supplied" if not views else "evaluated", "intrinsics_refitted": False,
        "partition_rule": "(inner_row+inner_column)%3==0 withheld from per-view pose fit", "pose_corner_indices": pose_indices.tolist(),
        "validation_corner_indices": check_indices.tolist(), "views": [], "independent_metric_accuracy_measured": False}
    for row in views:
        entry = {"image_sha256": row["image_sha256"], "detected": row["detected"]}
        if row["detected"]:
            pixel = _points(row["corners_px"], board)
            ok, rvec, tvec = cv2.solvePnP(objects[~check], pixel[~check], matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: raise ValueError("Held-out per-view pose could not be solved")
            if np.min((objects @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3))[:, 2]) <= 0: raise ValueError("Held-out board pose behind camera")
            entry.update(pose_fit=_residuals(objects[~check], pixel[~check], rvec, tvec, matrix, distortion),
                withheld_corners=_residuals(objects[check], pixel[check], rvec, tvec, matrix, distortion),
                pose_fit_corners_px=pixel[~check].tolist(), rvec=rvec.reshape(3).tolist(), tvec_m=tvec.reshape(3).tolist())
        else: entry["rejection_reasons"] = row["rejection_reasons"]
        result["views"].append(entry)
    return result


def validate_planar_length(fit, heldout_view, *, endpoints_px, measured_length_mm, measurement_uncertainty_mm,
                           measurement_evidence: dict):
    """Independent measured segment, on the held-out board plane only.

    Requires caller verification that neither this length nor endpoints were
    used as square size, K/D constraints, or this view's R/t fitting points.
    Does not validate 3D hand scale, points off the plane, or the capture video.
    """
    if not fit["accepted"]: raise ValueError("Rejected calibration cannot validate a length")
    if heldout_view["image_sha256"] in {row["image_sha256"] for row in fit["training_views"]}:
        raise ValueError("Length validation view was used to fit this calibration")
    evidence = measurement_evidence
    for flag in ("independent_of_all_calibration_constraints", "endpoints_on_board_plane_verified"):
        if evidence.get(flag) is not True: raise ValueError("Independent measurement and verified common plane required")
    if not isinstance(evidence.get("description"), str) or not evidence["description"].strip(): raise ValueError("Measurement description required")
    digest = evidence.get("sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest): raise ValueError("Measurement evidence SHA256 required")
    if not np.isfinite(measured_length_mm) or measured_length_mm <= 0 or not np.isfinite(measurement_uncertainty_mm) or measurement_uncertainty_mm < 0:
        raise ValueError("Positive measured length and nonnegative uncertainty required")
    points = np.asarray(endpoints_px, float)
    if points.shape != (2, 2) or not np.isfinite(points).all(): raise ValueError("Two finite original-image endpoints required")
    fit_pixels = np.asarray(heldout_view["pose_fit_corners_px"])
    if np.min(np.linalg.norm(points[:, None] - fit_pixels[None], axis=2)) <= 2:
        raise ValueError("Length endpoint overlaps pose-fitting corner; use excluded independent points")
    normalized = cv2.undistortPoints(points.reshape(-1, 1, 2), np.array(fit["matrix"]), np.array(fit["distortion"])).reshape(2, 2)
    rays = np.c_[normalized, np.ones(2)]
    normal = cv2.Rodrigues(np.array(heldout_view["rvec"]))[0][:, 2]; translation = np.array(heldout_view["tvec_m"])
    denominator = rays @ normal
    if np.min(np.abs(denominator)) < 1e-8: raise ValueError("Endpoint ray is parallel to assumed board plane")
    depths = float(normal @ translation) / denominator
    if np.min(depths) <= 0: raise ValueError("Endpoint ray-plane intersection behind camera")
    measured = float(np.linalg.norm(rays[0] * depths[0] - rays[1] * depths[1]) * 1000)
    return {"heldout_image_sha256": heldout_view["image_sha256"], "endpoints_px": points.tolist(), "reconstructed_length_mm": measured,
        "independent_measured_length_mm": measured_length_mm, "measurement_uncertainty_mm": measurement_uncertainty_mm,
        "relative_error": abs(measured - measured_length_mm) / measured_length_mm, "measurement_evidence": dict(evidence),
        "scope": "This independent segment on the assumed held-out board plane; not hand/video/global scale accuracy",
        "ten_percent_capture_scale_accuracy_verified": False}


def calibration_payload(fit):
    """Exact existing ingest contract. Additional evidence stays in sidecars."""
    if not fit["accepted"]: raise ValueError("Rejected calibration cannot be exported")
    matrix = np.array(fit["matrix"])
    return {"resolution": fit["resolution"], "intrinsics": dict(zip(("fx", "fy", "cx", "cy"),
        [float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])])),
        "distortion": fit["distortion"], "world_to_camera": None}


def run_calibration(images_dir, output_dir, board, *, heldout_dir=None, camera_identity, capture_settings, test_fixtures=False, min_views=10):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a new output directory; prior calibration evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    report = {"format_version": "cardcap.chessboard_calibration/1.0", "status": "running", "code_sha256": sha(__file__),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__, "board": asdict(board),
        "board_definition": "columns/rows count inner corners; square_mm is physical edge length, not board width",
        "coordinate_system": "OpenCV camera X-right Y-down Z-forward; board XY plane; translations metres",
        "camera_identity": camera_identity, "capture_settings": capture_settings, "input_kind": "synthetic_test_fixtures" if test_fixtures else "user_supplied_chessboard_images",
        "source_is_real_camera_verified": False, "ten_percent_capture_scale_accuracy_verified": False,
        "image_processing": "Stored pixels, ignore EXIF orientation; no resize/crop/undistort; SB returns subpixel corners",
        "solver": "cv2.calibrateCameraExtended, flags=0, pinhole 5-coefficient model; no automatic view rejection by residual",
        "official_api": "https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html", "ingest_calibration_written": False}
    try:
        if not camera_identity.strip() or not capture_settings.strip(): raise ValueError("Camera identity and same-lens/zoom/focus/pixel-geometry capture settings description required")
        training_paths = _image_paths(images_dir); heldout_paths = _image_paths(heldout_dir) if heldout_dir else []
        # File lists and their roles fixed before detection/estimation.
        report["input_partition"] = {kind: [{"path": str(p), "sha256": sha(p)} for p in paths] for kind, paths in (("training", training_paths), ("heldout", heldout_paths))}
        training, resolution = collect_chessboards(training_paths, board, subset="training", debug_dir=output / "corners/training")
        heldout, _ = collect_chessboards(heldout_paths, board, subset="heldout", expected_resolution=resolution, debug_dir=output / "corners/heldout")
        report.update(training_inventory=training, heldout_inventory=heldout, resolution=resolution)
        for kind, inventory in (("training", training), ("heldout", heldout)):
            if [(r["image_path"], r["image_sha256"]) for r in inventory] != [(r["path"], r["sha256"]) for r in report["input_partition"][kind]]:
                raise ValueError("Source image changed after the training/held-out split was fixed")
        selected = prepare_view_split(training, heldout, board)
        report["selected_training_image_sha256"] = [row["image_sha256"] for row in selected]
        fit = fit_camera(selected, board, min_views=min_views); report["fit"] = fit
        if not fit["accepted"]: raise ValueError("Calibration failed explicit quality gates: " + ", ".join(fit["rejection_reasons"]))
        report["heldout_validation"] = validate_heldout_views(heldout, board, fit)
        report["independent_length_validation"] = {"status": "not_supplied", "interface": "validate_planar_length", "capture_scale_accuracy_verified": False}
        if not test_fixtures:
            payload = calibration_payload(fit); write_json(output / "camera.json", payload)
            provenance = f"Explicit chessboard estimate from {len(selected)} source-hashed views; {output / 'report.json'}; independent metric accuracy unverified"
            for space, distortion in (("original", fit["distortion"]), ("undistorted", [0.] * 5)):
                write_json(output / f"pose_camera_{space}.json", {"camera": {"matrix": fit["matrix"], "distortion": distortion,
                    "calibrated": True, "provenance": provenance, "image_space": "original_distorted" if space == "original" else "undistorted_same_intrinsics"}})
            report.update(ingest_calibration_written=True, camera_json_sha256=sha(output / "camera.json"))
        report["status"] = "completed_fixture_only" if test_fixtures else "completed_calibration_estimate"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--heldout-images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, required=True, help="Inner corner count horizontally")
    parser.add_argument("--rows", type=int, required=True, help="Inner corner count vertically")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--camera-identity", required=True)
    parser.add_argument("--capture-settings", required=True)
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--test-fixtures", action="store_true", help="Synthetic validation only; never write ingest/pose camera files")
    args = parser.parse_args()
    try:
        result = run_calibration(args.images, args.output, Chessboard(args.columns, args.rows, args.square_mm), heldout_dir=args.heldout_images,
            camera_identity=args.camera_identity, capture_settings=args.capture_settings, min_views=args.min_views, test_fixtures=args.test_fixtures)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({key: result.get(key) for key in ("status", "ingest_calibration_written", "error")}, indent=2))
    return 0 if result["status"].startswith("completed") else 3


if __name__ == "__main__": raise SystemExit(main())
