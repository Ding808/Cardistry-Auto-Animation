"""Local image-motion evidence from existing surface masks; never merge events.

Features are reseeded on each observed source frame. Shared/independent planar
image models are compared on the same held-out correspondences per track.
Neither co-motion nor different homographies determines physical membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .draft_export import _bound, _inside, read_json, read_jsonl, sha, write_json


@dataclass(frozen=True)
class MotionParameters:
    max_corners: int = 100
    quality_level: float = .01
    min_distance_px: float = 5.
    erosion_radius_px: int = 3
    lk_window: int = 21
    lk_max_level: int = 3
    max_fb_error_px: float = 1.5
    minimum_points: int = 12
    minimum_hull_over_mask: float = .15
    minimum_occupied_3x3_cells: int = 4
    minimum_minor_spread_px: float = 2.
    ransac_threshold_px: float = 2.
    ransac_max_iters: int = 2000
    ransac_confidence: float = .995
    min_train_inlier_fraction: float = .6
    compatible_heldout_p95_px: float = 2.
    incompatible_heldout_p95_px: float = 3.
    min_rms_improvement_px: float = 1.


def _finite_point(point): return np.asarray(point, float).tolist() if np.isfinite(point).all() else None


def _coverage(points, mask):
    area = int(np.count_nonzero(mask))
    if len(points) < 3 or area == 0: return {"hull_over_mask": 0., "occupied_3x3_cells": 0, "minor_spread_px": 0.}
    hull_area = float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32))))
    y, x = np.nonzero(mask); low = np.array([x.min(), y.min()]); size = np.array([x.max()-x.min()+1, y.max()-y.min()+1])
    cells = np.clip(((points-low) / size * 3).astype(int), 0, 2)
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False) / np.sqrt(len(points))
    return {"hull_over_mask": hull_area/area, "occupied_3x3_cells": len(set(map(tuple, cells))), "minor_spread_px": float(singular[-1])}


def track_mask_points(gray0, gray1, mask0, mask1, parameters=MotionParameters()):
    if gray0.dtype != np.uint8 or gray1.dtype != np.uint8 or gray0.ndim != 2 or gray0.shape != gray1.shape or mask0.shape != gray0.shape or mask1.shape != gray0.shape:
        raise ValueError("Same-resolution uint8 grayscale frames and matching masks required")
    p = parameters; m0, m1 = mask0 > 0, mask1 > 0
    report = {"source_mask_area_px": int(m0.sum()), "target_mask_area_px": int(m1.sum()), "features": [],
        "detected_features": 0, "accepted_correspondences": 0, "eligible_for_model_comparison": False,
        "rejection_reasons": [], "tracking_quality_is_pose_or_identity_confidence": False}
    if not m0.any() or not m1.any():
        report["rejection_reasons"] = ["empty_source_mask" if not m0.any() else "empty_target_mask"]
        return report
    kernel = np.ones((p.erosion_radius_px*2+1,)*2, np.uint8)
    interior0 = cv2.erode(m0.astype(np.uint8), kernel); interior1 = cv2.erode(m1.astype(np.uint8), kernel)
    points = cv2.goodFeaturesToTrack(gray0, maxCorners=p.max_corners, qualityLevel=p.quality_level,
        minDistance=p.min_distance_px, mask=interior0, blockSize=5, useHarrisDetector=False)
    if points is None:
        report["rejection_reasons"] = ["no_internal_image_corners"]
        return report
    start = points.reshape(-1, 2); n = len(start); report["detected_features"] = n
    lk = dict(winSize=(p.lk_window, p.lk_window), maxLevel=p.lk_max_level,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01), flags=0, minEigThreshold=1e-4)
    forward, status, patch_error = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None, **lk)
    end, back = np.full((n, 2), np.nan, np.float32), np.full((n, 2), np.nan, np.float32)
    good_forward = np.zeros(n, bool); good_backward = np.zeros(n, bool)
    if forward is not None:
        end = forward.reshape(-1, 2)
        good_forward = (status.reshape(-1) == 1) & np.isfinite(end).all(axis=1)
        indices = np.flatnonzero(good_forward)
        if len(indices):
            reverse, rs, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, np.ascontiguousarray(end[indices, None, :]), None, **lk)
            if reverse is not None:
                back[indices] = reverse.reshape(-1, 2)
                good_backward[indices] = (rs.reshape(-1) == 1) & np.isfinite(back[indices]).all(axis=1)
    fb = np.linalg.norm(back-start, axis=1)
    inside = np.zeros(n, bool)
    bounded = good_forward & (end[:, 0] >= 0) & (end[:, 0] < gray0.shape[1]-.5) & (end[:, 1] >= 0) & (end[:, 1] < gray0.shape[0]-.5)
    indices = np.flatnonzero(bounded)
    pixel = np.rint(end[indices]).astype(int)
    inside[indices] = interior1[pixel[:, 1], pixel[:, 0]] > 0
    accepted = good_forward & good_backward & (fb <= p.max_fb_error_px) & inside
    for i in range(n):
        reasons = []
        if not good_forward[i]: reasons.append("forward_lk_invalid")
        if not good_backward[i]: reasons.append("backward_lk_invalid")
        if not np.isfinite(fb[i]) or fb[i] > p.max_fb_error_px: reasons.append("forward_backward_inconsistent")
        if not inside[i]: reasons.append("outside_next_eroded_surface_mask")
        report["features"].append({"index": i, "source_px": start[i].tolist(), "target_px": _finite_point(end[i]), "backward_px": _finite_point(back[i]),
            "fb_error_px": float(fb[i]) if np.isfinite(fb[i]) else None,
            "forward_patch_l1_error": float(patch_error.reshape(-1)[i]) if good_forward[i] and patch_error is not None else None,
            "accepted": bool(accepted[i]), "rejection_reasons": reasons})
    good0, good1 = start[accepted], end[accepted]
    report["accepted_correspondences"] = len(good0)
    source_coverage, target_coverage = _coverage(good0, m0), _coverage(good1, m1)
    report.update(source_coverage=source_coverage, target_coverage=target_coverage,
        feature_rejection_counts=dict(Counter(reason for feature in report["features"] for reason in feature["rejection_reasons"])))
    if len(good0):
        report.update(median_displacement_px=np.median(good1-good0, axis=0).tolist(),
            accepted_fb_median_px=float(np.median(fb[accepted])), accepted_fb_p95_px=float(np.quantile(fb[accepted], .95)))
    if len(good0) < p.minimum_points: report["rejection_reasons"].append("insufficient_consistent_correspondences")
    for name, coverage in (("source", source_coverage), ("target", target_coverage)):
        if coverage["hull_over_mask"] < p.minimum_hull_over_mask: report["rejection_reasons"].append(name+"_insufficient_spatial_hull_coverage")
        if coverage["occupied_3x3_cells"] < p.minimum_occupied_3x3_cells: report["rejection_reasons"].append(name+"_insufficient_grid_coverage")
        if coverage["minor_spread_px"] < p.minimum_minor_spread_px: report["rejection_reasons"].append(name+"_nearly_collinear_points")
    report["eligible_for_model_comparison"] = not report["rejection_reasons"]
    return report


def fit_homography(source, target, parameters=MotionParameters()):
    a, b = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matching Nx2 correspondence arrays required")
    if len(a) < 4 or np.linalg.matrix_rank(a-a.mean(axis=0)) < 2 or np.linalg.matrix_rank(b-b.mean(axis=0)) < 2:
        return {"available": False, "reason": "insufficient_or_collinear_correspondences"}
    cv2.setRNGSeed(76190)
    matrix, inliers = cv2.findHomography(a, b, method=cv2.RANSAC, ransacReprojThreshold=parameters.ransac_threshold_px,
        maxIters=parameters.ransac_max_iters, confidence=parameters.ransac_confidence)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return {"available": False, "reason": "homography_estimation_failed"}
    return {"available": True, "matrix": matrix.tolist(), "train_count": len(a), "train_inlier_mask": inliers.reshape(-1).astype(bool).tolist(),
            "train_inlier_fraction": float(np.mean(inliers)), "train_errors": homography_errors(matrix, a, b)}


def homography_errors(matrix, source, target):
    a, b = np.asarray(source, float), np.asarray(target, float)
    homogeneous = np.c_[a, np.ones(len(a))] @ np.asarray(matrix).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9): return {"available": False, "reason": "projection_at_infinity"}
    residual = np.linalg.norm(homogeneous[:, :2]/homogeneous[:, [2]]-b, axis=1)
    return {"available": True, "count": len(residual), "rms_px": float(np.sqrt(np.mean(residual**2))),
            "median_px": float(np.median(residual)), "p95_px": float(np.quantile(residual, .95)), "per_point_px": residual.tolist()}


def compare_image_models(tracks, parameters=MotionParameters()):
    """Compare only fixed held-out points, retaining a separate score per track."""
    if len(tracks) != 2: raise ValueError("Pairwise comparison requires exactly two surface tracks")
    p = parameters
    result = {"surface_track_ids": list(tracks), "image_motion_evidence": "unknown", "physical_group_relation": "unknown",
        "is_lifecycle_event": False, "physical_identity_verified": False, "reasons": [], "independent_models": {},
        "common_model": None, "heldout": {}, "split_rule": "sort accepted source points by x then y; every third point held out; no held-out refit",
        "semantics": "Whether one planar image-motion model explains both tracks; not physical packet membership"}
    if any(not track["eligible_for_model_comparison"] for track in tracks.values()):
        result["reasons"] = [identity+": "+", ".join(track["rejection_reasons"]) for identity, track in tracks.items() if not track["eligible_for_model_comparison"]]
        return result
    split = {}
    for identity, track in tracks.items():
        good = [feature for feature in track["features"] if feature["accepted"]]
        source = np.array([f["source_px"] for f in good]); target = np.array([f["target_px"] for f in good])
        ordering = np.lexsort((source[:, 1], source[:, 0])); source, target = source[ordering], target[ordering]
        holdout = np.arange(len(source)) % 3 == 0
        split[identity] = (source[~holdout], target[~holdout], source[holdout], target[holdout])
        model = fit_homography(source[~holdout], target[~holdout], p)
        result["independent_models"][identity] = model
        if not model["available"] or model["train_inlier_fraction"] < p.min_train_inlier_fraction:
            result["reasons"].append(identity+": independent homography training support insufficient")
    if result["reasons"]: return result
    common = fit_homography(np.concatenate([s[0] for s in split.values()]), np.concatenate([s[1] for s in split.values()]), p)
    result["common_model"] = common
    if not common["available"]:
        result["reasons"].append("common homography unavailable")
        return result
    offset = 0
    for identity, (train0, train1, test0, test1) in split.items():
        result["heldout"][identity] = {"source_px": test0.tolist(), "target_px": test1.tolist(),
            "independent_model_errors": homography_errors(result["independent_models"][identity]["matrix"], test0, test1),
            "common_model_errors": homography_errors(common["matrix"], test0, test1),
            "common_train_inlier_fraction_this_track": float(np.mean(common["train_inlier_mask"][offset:offset+len(train0)]))}
        offset += len(train0)
    if any(not score[k]["available"] for score in result["heldout"].values() for k in ("independent_model_errors", "common_model_errors")):
        result["reasons"].append("held-out projection unavailable")
        return result
    per_track = list(result["heldout"].values())
    independent_good = all(s["independent_model_errors"]["p95_px"] <= p.compatible_heldout_p95_px for s in per_track)
    common_good = all(s["common_model_errors"]["p95_px"] <= p.compatible_heldout_p95_px for s in per_track)
    if independent_good and common_good:
        result["image_motion_evidence"] = "shared_planar_image_motion_compatible"
    elif independent_good and any(s["common_model_errors"]["p95_px"] > p.incompatible_heldout_p95_px and
        s["common_model_errors"]["rms_px"]-s["independent_model_errors"]["rms_px"] >= p.min_rms_improvement_px for s in per_track):
        result["image_motion_evidence"] = "separate_planar_image_models_better_supported"
    else: result["reasons"].append("held-out residuals do not meet either fixed evidence criterion")
    return result


def _png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok: raise RuntimeError("Diagnostic PNG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data.tobytes())


def run_motion_relation(bundle_dir, output_dir, start_frame, end_frame, parameters=MotionParameters()):
    output, bundle = Path(output_dir).resolve(), Path(bundle_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a new output directory; prior evidence is preserved")
    manifest = read_json(bundle / "manifest.json")
    count = manifest["source"]["frame_count"]
    if any(isinstance(f, bool) or not isinstance(f, int) for f in (start_frame, end_frame)) or not (0 <= start_frame < end_frame < count):
        raise ValueError("Inclusive motion window requires 0 <= start < end < source frame count")
    _bound(bundle / manifest["observations"]["file"], manifest["observations"]["sha256"], "draft observations")
    src_ref = manifest["inputs"]["source_manifest"]
    source_path = _bound(src_ref["path"], src_ref["sha256"], "source image manifest")
    source = read_json(source_path)
    if source["source_video_sha256"] != manifest["source"]["sha256"]: raise ValueError("Motion inputs source mismatch")
    _bound(source["source_video"], source["source_video_sha256"], "original source video")
    rows = read_jsonl(bundle / manifest["observations"]["file"])
    ids = [seed["object_id"] for seed in manifest["surface_track_seeds"]["objects"]]
    if len(ids) != 2: raise ValueError("This bounded pairwise entry requires exactly two existing surface tracks")
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter(); images, masks, inputs = {}, {}, []
    for frame in range(start_frame, end_frame+1):
        row, source_row = rows[frame], source["frames"][frame]
        if row["frame"] != frame or row["source_frame"] != frame or source_row["frame"] != frame: raise ValueError("Motion frame index mismatch")
        if row["source_jpeg_sha256"] != source_row["jpeg_sha256"]: raise ValueError("Motion source JPEG mismatch")
        path = _bound(_inside(source["frame_directory"], source_row["jpeg_file"]), source_row["jpeg_sha256"], "source JPEG")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2][::-1]) != source["resolution"]: raise ValueError("Source JPEG dimensions differ")
        images[frame] = image; masks[frame] = {}
        references = []
        for obj in row["objects"]:
            path = _bound(obj["mask_reference"]["path"], obj["mask_reference"]["sha256"], "surface mask")
            mask = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != image.shape[:2] or not np.isin(mask, [0, 255]).all(): raise ValueError("Invalid source mask")
            masks[frame][obj["surface_track_id"]] = mask
            references.append(dict(obj["mask_reference"], surface_track_id=obj["surface_track_id"],
                inherited_visibility=obj["track_visibility"], physical_identity_verified=False))
        if set(masks[frame]) != set(ids): raise ValueError("Source frame has missing track records; no mask is fabricated")
        _png(output / "native_frames" / f"frame_{frame:06d}.png", image)
        inputs.append({"frame": frame, "jpeg_sha256": source_row["jpeg_sha256"], "decoded_bgr_sha256": __import__('hashlib').sha256(image.tobytes()).hexdigest(),
                       "native_png_sha256": sha(output / "native_frames" / f"frame_{frame:06d}.png"), "masks": references})
    records = []
    for frame in range(start_frame, end_frame):
        gray0, gray1 = cv2.cvtColor(images[frame], cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[frame+1], cv2.COLOR_BGR2GRAY)
        tracks = {identity: track_mask_points(gray0, gray1, masks[frame][identity], masks[frame+1][identity], parameters) for identity in ids}
        comparison = compare_image_models(tracks, parameters)
        record = {"source_frame": frame, "target_frame": frame+1, "time_seconds": frame/source["fps"], "tracks": tracks,
            "pair": comparison, "physical_events": [], "source_video_sha256": source["source_video_sha256"]}
        records.append(record)
    relations = output / "motion_relations.jsonl"
    relations.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    height, width = images[start_frame].shape[:2]; banner_height = 192
    video_path = output / "motion_diagnostic.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), source["fps"], (width, height+banner_height))
    if not writer.isOpened(): raise RuntimeError("Diagnostic video writer could not open")
    colors = [(255, 190, 40), (70, 220, 100)]
    try:
        for frame in range(start_frame, end_frame+1):
            body = images[frame].copy(); record = records[frame-start_frame] if frame < end_frame else None
            header = np.full((banner_height, width, 3), 24, np.uint8)
            cv2.putText(header, f"Image-motion evidence | frame {frame} | native {width}x{height}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (245,245,245), 1, cv2.LINE_AA)
            label = record["pair"]["image_motion_evidence"] if record else "end frame; no outgoing interval evaluated"
            cv2.putText(header, label, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, .60, (230,230,230), 1, cv2.LINE_AA)
            for index, identity in enumerate(ids):
                contours, _ = cv2.findContours(masks[frame][identity], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(body, contours, -1, colors[index], 1, cv2.LINE_AA)
                if record:
                    track = record["tracks"][identity]
                    for feature in track["features"]:
                        start = tuple(np.rint(feature["source_px"]).astype(int))
                        if feature["accepted"]:
                            target = tuple(np.rint(feature["target_px"]).astype(int))
                            cv2.arrowedLine(body, start, target, colors[index], 1, cv2.LINE_AA, tipLength=.2)
                        else: cv2.circle(body, start, 2, (80, 80, 235), 1, cv2.LINE_AA)
                    info = f"track {index+1}: {track['accepted_correspondences']}/{track['detected_features']} points; eligible={track['eligible_for_model_comparison']}"
                    if record["pair"]["heldout"].get(identity):
                        score = record["pair"]["heldout"][identity]
                        info += f" | heldout RMS shared={score['common_model_errors'].get('rms_px', float('nan')):.2f} / separate={score['independent_model_errors'].get('rms_px', float('nan')):.2f}px"
                else: info = f"track {index+1}: mask area {np.count_nonzero(masks[frame][identity])} px"
                cv2.putText(header, info, (18, 88+index*30), cv2.FONT_HERSHEY_SIMPLEX, .48, colors[index], 1, cv2.LINE_AA)
            cv2.putText(header, "Physical membership UNKNOWN | no merge/split inferred | red marks: rejected flow", (18, 162), cv2.FONT_HERSHEY_SIMPLEX, .52, (215,215,215), 1, cv2.LINE_AA)
            rendered = np.vstack((header, body))
            _png(output / "diagnostic_frames" / f"frame_{frame:06d}.png", rendered); writer.write(rendered)
    finally: writer.release()
    capture = cv2.VideoCapture(str(video_path)); decoded = 0
    while True:
        ok, image = capture.read()
        if not ok: break
        if image.shape != (height+banner_height, width, 3): raise RuntimeError("Diagnostic video shape mismatch")
        decoded += 1
    capture.release()
    if decoded != end_frame-start_frame+1: raise RuntimeError("Incomplete diagnostic video")
    counts = dict(Counter(r["pair"]["image_motion_evidence"] for r in records))
    report = {"format_version": "cardcap.surface_motion_relations/1.0", "status": "completed", "bundle_manifest": str(bundle / "manifest.json"),
        "bundle_manifest_sha256": sha(bundle / "manifest.json"), "source_video_sha256": source["source_video_sha256"],
        "source_manifest_sha256": sha(source_path), "observations_sha256": manifest["observations"]["sha256"],
        "window_inclusive": [start_frame, end_frame], "intervals": len(records), "source_resolution": source["resolution"],
        "parameters": asdict(parameters), "code_sha256": sha(__file__), "opencv_version": cv2.__version__,
        "point_seeding": "Fresh Shi-Tomasi points on each source frame's eroded supplied mask; never re-seed from inferred missing masks",
        "lk_error_semantics": "Forward patch L1 error and forward-backward distance are image tracking diagnostics, not pose/identity confidence",
        "image_relation_counts": counts, "eligible_pair_intervals": sum(all(t["eligible_for_model_comparison"] for t in r["tracks"].values()) for r in records),
        "physical_group_relation": "unknown", "physical_events": [], "no_model_inference": True, "source_frames_or_masks_modified": False,
        "inputs": inputs, "motion_relations_sha256": sha(relations), "diagnostic_video": str(video_path), "diagnostic_video_sha256": sha(video_path),
        "diagnostic_video_redecoded_frames": decoded, "elapsed_seconds": time.perf_counter()-start_time,
        "limitations": ["Masks may span multiple physical faces or drift; inherited track identity remains unverified",
                        "Shared image motion does not prove one packet; one rigid group can require different homographies through parallax",
                        "Empty masks and insufficient feature support yield unknown, not a physical merge",
                        "Separate model has more parameters; comparison uses fixed held-out points per track, not training fit improvement alone"]}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True); parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()
    result = run_motion_relation(args.bundle, args.output, args.start_frame, args.end_frame)
    print(json.dumps({k: result[k] for k in ("status", "window_inclusive", "eligible_pair_intervals", "image_relation_counts", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
