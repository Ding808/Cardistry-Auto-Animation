"""Export explicitly reviewed, isolated real geometry observations to cardcap 1.1.

No segmentation, pose fitting, interpolation, physical identity association or
events are inferred. Local card XY is preserved while camera coordinates change.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cardcap.packet_contract import (read_json, loads_strict, validate_capture,
    _field, _field_key, _ue_ascii_key)

B = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
L = np.diag([1., 1., -1.])
MAX_ENGINEERING_CONFIDENCE = .35


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def field(value, name):
    return _field(value, name, "capture")


def set_field(value, name, item):
    value[_field_key(value, name, "capture")] = item


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), name + " has invalid shape/nonfinite values")
    return result


def rotation(value, name):
    result = array(value, (3, 3), name)
    require(np.max(np.abs(result.T @ result - np.eye(3))) < 1e-7 and abs(np.linalg.det(result)-1.) < 1e-7,
            name + " must be orthogonal with determinant +1")
    return result


def matrix_to_quaternion(matrix):
    axis = cv2.Rodrigues(rotation(matrix, "quaternion input rotation"))[0].reshape(3)
    angle = np.linalg.norm(axis)
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    result = np.r_[axis / angle * np.sin(angle/2), np.cos(angle/2)]
    return result / np.linalg.norm(result)


def quaternion_to_matrix(value):
    q = array(value, (4,), "stored xyzw quaternion")
    require(abs(np.linalg.norm(q)-1.) < 1e-7, "stored quaternion must have unit length")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def corners(dimensions_m):
    width, height = array(dimensions_m, (2,), "card dimensions_m")
    require(min(width, height) > 0, "card dimensions must be positive")
    return np.array([[-width/2, -height/2, 0.], [width/2, -height/2, 0.],
                     [width/2, height/2, 0.], [-width/2, height/2, 0.]])


def camera_parameters(camera):
    intr = field(camera, "intrinsics")
    k = np.array([[field(intr, "fx"), 0., field(intr, "cx")],
                  [0., field(intr, "fy"), field(intr, "cy")], [0., 0., 1.]])
    return k, np.asarray(field(camera, "distortion"), np.float64)


def project_stored_packet(position_cm, quaternion, dimensions_m, k, distortion):
    local_cm = corners(dimensions_m) * 100.
    world_cm = local_cm @ quaternion_to_matrix(quaternion).T + array(position_cm, (3,), "position_cm")
    camera_m = (world_cm / 100.) @ B
    require(np.all(camera_m[:, 2] > 1e-8), "stored packet corners must remain in front of camera")
    pixels = cv2.projectPoints(camera_m, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(4, 2)
    return camera_m, pixels


def convert_candidate(candidate, dimensions_m, fitted_corners, k, distortion):
    """Check and convert a selected valid candidate; no new PnP solve occurs."""
    require(candidate["valid"] is True and candidate["positive_depth"] is True and not candidate["rejection_reasons"],
            "selected candidate must be valid with positive depth and no rejection")
    r = rotation(candidate["rotation_matrix"], "candidate camera rotation")
    t = array(candidate["tvec_m"], (3,), "candidate translation metres")
    rvec = array(candidate["rvec"], (3,), "candidate rvec")
    require(np.max(np.abs(cv2.Rodrigues(rvec)[0]-r)) < 1e-7, "candidate rvec and rotation matrix differ")
    local = corners(dimensions_m)
    camera_points = local @ r.T + t
    require(np.all(camera_points[:, 2] > 1e-8), "candidate has a nonpositive actual corner depth")
    require(np.allclose(array(candidate["depths_m"], (4,), "candidate corner depths"), camera_points[:, 2], atol=1e-8, rtol=0), "stored candidate depths differ")
    source_pixels = cv2.projectPoints(local, rvec, t, k, distortion)[0].reshape(4, 2)
    order = candidate["corner_indices"]
    require(isinstance(order, list) and len(order) == 4 and all(type(i) is int for i in order) and sorted(order) == [0, 1, 2, 3], "candidate corner_indices must be a permutation")
    inputs = array(fitted_corners, (4, 2), "fitted quad corners")[order]
    errors = np.linalg.norm(source_pixels-inputs, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    require(abs(rms-float(candidate["input_corner_reprojection_rmse_px"])) < 1e-6 and
            abs(float(errors.max())-float(candidate["input_corner_reprojection_max_px"])) < 1e-6,
            "candidate fitted-input residual does not match camera/corners")
    # Camera change B is a reflection. Reflect only local normal Z with L so
    # local XY width/height stay unchanged and the stored rotation is proper.
    r_ue = rotation(B @ r @ L, "packet UE rotation")
    position = B @ t * 100.
    quaternion = matrix_to_quaternion(r_ue)
    restored_points, restored_pixels = project_stored_packet(position, quaternion, dimensions_m, k, distortion)
    pixel_error = float(np.max(np.linalg.norm(restored_pixels-source_pixels, axis=1)))
    require(pixel_error < 1e-6 and np.max(np.abs(restored_points-camera_points)) < 1e-8,
            "stored quaternion/XY-template conversion changed a source corner")
    return {"position_cm": position.tolist(), "rotation_quat": quaternion.tolist(),
            "rotation_matrix_ue": r_ue.tolist(), "source_camera_corner_points_m": camera_points.tolist(),
            "source_candidate_projected_corners_px": source_pixels.tolist(),
            "stored_packet_projected_corners_px": restored_pixels.tolist(),
            "fitted_input_corners_in_candidate_order_px": inputs.tolist(), "corner_indices": order,
            "input_corner_fit_rmse_px": rms, "input_corner_fit_errors_px": errors.tolist(),
            "conversion_max_corner_projection_difference_px": pixel_error,
            "rotation_determinant": float(np.linalg.det(r_ue)),
            "normal_and_front_back_identity": "unknown", "hand_prior_factor_applied_to_packet": False}


def load_selection(capture_path, selection_path):
    """Read only the requested source runs and selected masks/original frames."""
    capture_path, selection_path = Path(capture_path).resolve(), Path(selection_path).resolve()
    capture, selection = read_json(capture_path), read_json(selection_path)
    validate_capture(capture)
    require(field(capture, "format_version") in ("1.0", "1.1"),
            "This legacy reviewed-packet exporter supports 1.0/1.1 metric captures only; conditional 1.2 packet export is not implemented")
    require(selection["schema"] == "cardcap.reviewed_packet_selection/1.0", "unsupported selection schema")
    require(not field(capture, "packets") and all(not field(field(capture, "events"), key) for key in ("splits", "merges", "releases")), "input must contain hands only and no packet events")
    require("packet_draft" not in capture, "input already contains a packet draft")
    require(sha(capture_path) == selection["capture_sha256"], "capture hash differs from explicit selection")
    meta, cam = field(capture, "meta"), field(capture, "camera")
    source_hash = field(meta, "source_video_sha256")
    require(source_hash == selection["source_video_sha256"], "selection source video differs from capture")
    text(selection["source_observations"], "selection.source_observations")
    protected = {str(capture_path): sha(capture_path), str(selection_path): sha(selection_path)}

    def bound(path, expected):
        path = Path(path)
        path = (selection_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        require(sha(path) == expected, "source hash mismatch: " + str(path))
        protected[str(path)] = expected
        return path

    k, distortion = camera_parameters(cam)
    runs = {}
    require(isinstance(selection["geometry_runs"], list) and selection["geometry_runs"], "geometry_runs must be nonempty")
    for item in selection["geometry_runs"]:
        run_id = text(item["id"], "geometry id")
        require(run_id not in runs, "duplicate geometry id")
        report_path = bound(item["report_file"], item["report_sha256"])
        report = read_json(report_path)
        require(report["format_version"] == "cardcap.mask_geometry_report/1.0" and report["status"] == "completed", "geometry report must be completed")
        path = bound(item["frames_file"], item["frames_sha256"])
        require(path == (report_path.parent / report["geometry_frames_file"]).resolve() and item["frames_sha256"] == report["geometry_frames_sha256"], "geometry frames not bound by report")
        require(report["source_video_sha256"] == source_hash and report["frame_count"] == report["completed_frames"] == field(meta, "frame_count")
                and report["fps"] == field(meta, "fps") and report["resolution"] == field(meta, "resolution"), "geometry timeline/video differs from capture")
        require(np.array_equal(k, report["camera"]["matrix"]) and np.array_equal(distortion, report["camera"]["distortion"])
                and field(cam, "calibrated") is report["camera"]["calibrated"], "geometry camera differs from capture")
        run = Path(report["segmentation_run"]).resolve()
        source = read_json(bound(run / "source_manifest.json", report["input_hashes"]["source_manifest.json"]))
        bound(run / "report.json", report["input_hashes"]["report.json"])
        seg_path = bound(run / "frames.jsonl", report["input_hashes"]["frames.jsonl"])
        bound(source["source_video"], source_hash)
        require(source["source_video_sha256"] == source_hash and source["image_space"] == "original_distorted"
                and source["frame_count"] == report["frame_count"] and source["fps"] == report["fps"] and source["resolution"] == report["resolution"], "segmentation source/camera image space differs")
        frames = [loads_strict(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
        segments = [loads_strict(line) for line in seg_path.read_text(encoding="utf-8-sig").splitlines()]
        require(len(frames) == len(segments) == len(source["frames"]) == report["frame_count"], "source record counts differ")
        require(all(row["frame"] == row["source_frame"] == i for i, row in enumerate(frames)), "geometry frames are reordered or duplicated")
        runs[run_id] = {"report": report, "run": run, "source": source, "frames": frames, "segments": segments}
    require(isinstance(selection["selections"], list) and selection["selections"], "at least one explicit review selection is required")
    selected, packet_ids, selected_keys = [], set(), set()
    for choice in selection["selections"]:
        packet_id = text(choice["packet_id"], "packet_id")
        require(_ue_ascii_key(packet_id) not in packet_ids, "packet IDs must be distinct isolated observations")
        packet_ids.add(_ue_ascii_key(packet_id))
        frame = choice["frame"]
        require(type(frame) is int and 0 <= frame < field(meta, "frame_count"), "selected frame is outside source")
        confidence = choice["confidence"]
        require(type(confidence) in (int, float) and np.isfinite(confidence) and 0 <= confidence <= MAX_ENGINEERING_CONFIDENCE, "review confidence must be a low engineering score in [0,.35]")
        require(choice["geometry_id"] in runs, "unknown geometry id")
        key = (choice["geometry_id"], frame, choice["object_id"])
        require(key not in selected_keys, "same surface observation selected more than once")
        selected_keys.add(key)
        run = runs[choice["geometry_id"]]
        row, seg_row = run["frames"][frame], run["segments"][frame]
        require(seg_row["frame"] == seg_row["source_frame"] == frame and row["source_video_sha256"] == seg_row["source_video_sha256"] == source_hash
                and row["time_seconds"] == seg_row["time_seconds"] == frame/field(meta, "fps"), "selected source frame/time differs")
        objects = [obj for obj in row["objects"] if obj["object_id"] == choice["object_id"]]
        require(len(objects) == 1, "selected geometry object missing or duplicated")
        obj = objects[0]
        require(obj["quad"]["accepted"] is True and obj["pose"]["accepted"] is True, "selected quad and pose must be accepted")
        segs = [entry for entry in seg_row["objects"] if entry["object_id"] == choice["object_id"]]
        require(len(segs) == 1 and canonical(obj["segmentation"]) == canonical(segs[0]), "selected geometry segmentation differs from source")
        require(obj["mask_sha256"] == obj["segmentation"]["mask_sha256"] and obj["sam2_object_id"] == obj["segmentation"]["sam2_object_id"], "selected mask/ID source differs")
        mask_path = (run["run"] / obj["mask_file"]).resolve()
        require(mask_path.is_relative_to(run["run"]), "selected mask escapes run")
        bound(mask_path, obj["mask_sha256"])
        candidates = [candidate for candidate in obj["pose"]["candidates"] if candidate["candidate_id"] == choice["candidate_id"]]
        require(len(candidates) == 1, "selected pose candidate missing or duplicated")
        review = choice["review"]
        require(review["kind"] == "manual_original_frame_review", "explicit original-frame review required")
        text(review["notes"], "review.notes"); text(review["physical_packet_observation_basis"], "review.physical_packet_observation_basis")
        raw = bound(review["original_frame_file"], review["original_frame_sha256"])
        pixels = cv2.imdecode(np.frombuffer(raw.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        width, height = field(meta, "resolution")
        require(pixels is not None and pixels.dtype == np.uint8 and pixels.shape == (height, width, 3)
                and hashlib.sha256(pixels.tobytes()).hexdigest() == run["source"]["frames"][frame]["decoded_bgr_sha256"], "review original pixels differ from actual source frame")
        transform = convert_candidate(candidates[0], obj["pose"]["dimensions_m"], obj["quad"]["corners_px"], k, distortion)
        selected.append({"selection": choice, "geometry": obj, "candidate": candidates[0], "transform": transform})
    return capture, selection, selected, protected


def construct_capture(source, selection, selected):
    require(field(source, "format_version") in ("1.0", "1.1"),
            "This legacy reviewed-packet exporter supports 1.0/1.1 metric captures only; conditional 1.2 packet export is not implemented")
    output = deepcopy(source)
    set_field(output, "format_version", "1.1")
    packets = []
    for item in selected:
        choice, geometry, transform = item["selection"], item["geometry"], item["transform"]
        frame = choice["frame"]
        packets.append({"id": choice["packet_id"], "birth_frame": frame, "death_frame": frame,
            "card_count_estimate": None, "card_count_confidence": None,
            "dimensions_cm": [value*100. for value in geometry["pose"]["dimensions_m"]] + [None],
            "frames": [{"frame": frame, "confidence": choice["confidence"],
                "position_cm": transform["position_cm"], "rotation_quat": transform["rotation_quat"],
                "linear_velocity_cm_s": None, "angular_velocity_rad_s": None,
                "reprojection_error_px": item["candidate"]["input_corner_reprojection_rmse_px"],
                "contact_state": "UNKNOWN", "contact_bones": None,
                "measurement_label": "input_corner_fit", "independent_corner_validation_error_px": None,
                "confidence_semantics": "low engineering review score; not learned probability or measured accuracy",
                "source_observation": choice, "source_pose_candidate": item["candidate"],
                "source_pose_ambiguities": geometry["pose"]["ambiguities"],
                "face_visibility": geometry["quad"]["visibility"],
                "full_face_verified": geometry["pose"]["full_face_verified"]}],
            "observation_interval_only": True, "physical_existence_outside_observation": "unknown",
            "cross_tracklet_physical_identity": "unknown", "normal_and_front_back_identity": "unknown",
            "dimensions_source": geometry["pose"].get("dimensions_source", "legacy source IPPE nominal card dimensions; thickness unknown"),
            "dimensions_provenance": deepcopy(geometry["pose"].get("dimensions_provenance")),
            "physical_packet_hypothesis_basis": choice["review"]["physical_packet_observation_basis"]})
    set_field(output, "packets", packets)
    quality = field(output, "quality")
    quality["source_hand_low_confidence_ranges"] = deepcopy(field(quality, "low_confidence_ranges"))
    set_field(quality, "low_confidence_ranges", [[0, field(field(output, "meta"), "frame_count")-1]])
    warnings = field(quality, "warnings")
    warnings.append("Packet draft needs correction throughout the full clip: isolated manually reviewed observations are not a complete lifecycle; unobserved frames and cross-ID identity remain unknown.")
    warnings.append("Packet reprojection_error_px is fitted-input-corner residual, not independent accuracy. The source hand scale and source-declared card dimensions/IPPE depth are not a jointly validated metric scene or contact solution.")
    quality["packet_review_required_ranges"] = [[0, field(field(output, "meta"), "frame_count")-1]]
    quality["packet_lifecycle_complete"] = False
    output["packet_draft"] = {"schema": "cardcap.reviewed_packet_tracklets/1.0", "source_observations": selection["source_observations"],
        "events_inferred": False, "sparse_single_frame_observations": True,
        "birth_death_semantics": "observed interval only; not physical creation/destruction",
        "hand_payload_and_scale_unchanged": True,
        "packet_scale_domain": "source-declared card IPPE metres to UE centimetres once; existing hand scale factor not applied",
        "hand_card_metric_alignment_independently_verified": False, "contact_evaluated": False,
        "full_clip_packet_motion_requires_correction": True}
    validate_capture(output)
    return output


def export_reviewed_packet_draft(capture_json, selection_json, output_dir):
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "use a new output directory; existing artifacts are preserved")
    source, selection, selected, protected = load_selection(capture_json, selection_json)
    output = construct_capture(source, selection, selected)
    preserved = {key: hashlib.sha256(canonical(field(source, key))).hexdigest() for key in ("hands", "camera", "scale", "meta", "events")}
    for key in preserved:
        require(canonical(field(source, key)) == canonical(field(output, key)), "protected capture values changed: " + key)
    require(all(sha(path) == digest for path, digest in protected.items()), "input changed during export preparation")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "capture.cardcap.json"
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    stored = read_json(output_path)
    require(canonical(stored) == canonical(output), "output readback differs from constructed values")
    validation = validate_capture(stored)
    k, distortion = camera_parameters(field(stored, "camera"))
    checks = []
    for packet, item in zip(field(stored, "packets"), selected):
        sample = packet["frames"][0]
        _, pixels = project_stored_packet(sample["position_cm"], sample["rotation_quat"], item["geometry"]["pose"]["dimensions_m"], k, distortion)
        difference = float(np.max(np.linalg.norm(pixels-np.asarray(item["transform"]["source_candidate_projected_corners_px"]), axis=1)))
        require(difference < 1e-6, "stored packet changed source candidate projection")
        checks.append({"packet_id": packet["id"], "frame": sample["frame"], "candidate_id": item["candidate"]["candidate_id"],
                       "conversion": item["transform"], "actual_output_readback_max_corner_difference_px": difference})
    manifest = {"schema": "cardcap.reviewed_packet_draft_export/1.0", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "capture_input": str(Path(capture_json).resolve()),
        "selection_input": str(Path(selection_json).resolve()), "inputs_sha256": protected,
        "output": {"path": str(output_path), "sha256": sha(output_path)},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "preserved_top_level_canonical_sha256": preserved, "readback_validation": validation,
        "coordinate_convention": {"camera_to_ue_B": B.tolist(), "local_normal_reflection_L": L.tolist(),
            "rotation": "R_UE = B @ R_camera @ L; XY card template unchanged", "position": "position_cm = B @ tvec_m * 100",
            "local_card_template": "[-w/2,-h/2,0],[w/2,-h/2,0],[w/2,h/2,0],[-w/2,h/2,0]",
            "quaternion": "xyzw", "front_back_and_normal_identity": "unknown"},
        "selected_observation_count": len(selected), "packet_observation_intervals": [[item["selection"]["frame"]]*2 for item in selected],
        "projection_readback": checks, "full_clip_packet_lifecycle_complete": False,
        "events_inferred": False, "missing_packet_poses_filled": False,
        "hand_prior_factor_applied_to_packets": False, "joint_hand_card_metric_or_contact_validation": False,
        "measurement_label": "input_corner_fit", "independent_accuracy_measured": False}
    with (output_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_reviewed_packet_draft(args.capture_json, args.selection_json, args.output_dir)
    print(json.dumps({"status": result["status"], "observations": result["selected_observation_count"], "output": result["output"]}, indent=2))


if __name__ == "__main__":
    main()
