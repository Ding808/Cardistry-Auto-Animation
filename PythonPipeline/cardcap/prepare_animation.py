"""Prepare fixed-shape research animation from real per-frame observations.

No detector/model inference occurs here. Missing animation samples are explicit
SLERP/linear interpolation or endpoint holds, never new source observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from smplx import MANOLayer
from smplx.utils import Struct
from smplx.vertex_ids import vertex_ids

from . import __version__
from .display_space import DisplayInputError, build_display_space, source_crop_diagnostics
from .hand.mano_assets import load_mano_arrays
from .hand.research_hands_glb import export_research_hands

CAMERA_TO_UE = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
LEFT_REFLECTION = np.diag([-1., 1., 1.])
JOINT_MAP_21 = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
TIP_INDICES = np.asarray(list(vertex_ids["mano"].values()))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_to_quaternion(matrix):
    axis_angle = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))[0].reshape(3)
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    result = np.r_[axis_angle / angle * np.sin(angle / 2), np.cos(angle / 2)]
    return result / np.linalg.norm(result)


def quaternion_to_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def slerp(first, second, alpha):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        result = (1 - alpha) * a + alpha * b
    else:
        angle = np.arccos(dot)
        result = (np.sin((1 - alpha) * angle) * a + np.sin(alpha * angle) * b) / np.sin(angle)
    return result / np.linalg.norm(result)


def fill_samples(observed, frame_count):
    """Return rotations/translations plus provenance; source mapping untouched."""
    if not observed:
        raise ValueError("A hand with no real observations cannot be synthesized")
    keys = sorted(observed)
    result = []
    for frame in range(frame_count):
        if frame in observed:
            item = dict(observed[frame], sample_kind="observed", source_frames=[frame], alpha=None)
        else:
            before = [key for key in keys if key < frame]
            after = [key for key in keys if key > frame]
            if not before or not after:
                endpoint = after[0] if after else before[-1]
                source = observed[endpoint]
                item = {"rotations": source["rotations"].copy(), "translation": source["translation"].copy(),
                        "sample_kind": "endpoint_hold", "source_frames": [endpoint], "alpha": None}
            else:
                left, right = before[-1], after[0]
                alpha = (frame - left) / (right - left)
                item = {"rotations": np.asarray([slerp(a, b, alpha) for a, b in
                        zip(observed[left]["rotations"], observed[right]["rotations"])]),
                        "translation": (1 - alpha) * observed[left]["translation"] + alpha * observed[right]["translation"],
                        "sample_kind": "interpolated", "source_frames": [left, right], "alpha": alpha}
        # q and -q are equivalent rotations; choose continuous signs per bone.
        rotations = np.asarray(item["rotations"]).copy()
        if result:
            flip = np.sum(rotations * result[-1]["rotations"], axis=1) < 0
            rotations[flip] *= -1
        else:
            rotations[rotations[:, 3] < 0] *= -1
        item["rotations"] = rotations
        item["frame"] = frame
        result.append(item)
    return result


def frame_ranges(frames):
    result = []
    for frame in sorted(set(frames)):
        if result and frame == result[-1][1] + 1:
            result[-1][1] = frame
        else:
            result.append([frame, frame])
    return result


def projection_distortion(camera):
    """An ideal pinhole is an explicit rendering assumption when D is unknown."""
    value = camera.get("distortion")
    return np.asarray(value, dtype=np.float64) if value is not None else np.zeros(5)


def camera_configuration(view, observations=()):
    width, height = view["resolution"]
    camera = view.get("camera") or {}
    if not camera:
        from .ingest import resolve_intrinsics
        camera = resolve_intrinsics(view["source_video"], resolution=(width, height))
    if not camera.get("calibrated"):
        metadata = camera.get("metadata_report")
        legacy_evidence = camera.get("intrinsics_source") not in (None, "unobservable")
        if metadata is None and view.get("source_video"):
            from .video_metadata import read_video_metadata
            metadata = read_video_metadata(view["source_video"])
        # A checkpoint's crop focal is a normalization convention, not camera
        # evidence. It must never supply a shared full-image K or depth.
        return {"intrinsics": None,
                "distortion": None, "calibrated": False, "image_space": "original_distorted",
                "source_lens_distortion": None, "intrinsics_source": "unobservable",
                "provenance": {"kind": "unobservable", "physical_intrinsics": None,
                    "source_camera_evidence": camera,
                    "source_camera_evidence_is_legacy": legacy_evidence,
                    "current_container_metadata": metadata,
                    "assumptions": ["Source MANO shape and pose are wrist-local; relative hand placement and camera depth are unresolved.",
                        "Any assumed common-space or local viewing cameras are display_only and must not be reused for reconstruction or written into camera.intrinsics."],
                    "independent_accuracy_verified": False},
                "warnings": ["Camera K/D and inter-hand placement are unresolved. Conditional common-space and local previews are display-only hypotheses.",
                    "Container metadata status: " + (metadata["status"] if metadata else "unreadable") +
                    "; absence is limited to supported parsed fields. Historical camera warnings remain in source_camera_evidence only."],
                "translation_method": "Not solved: no full-image camera evidence. Global translations and inter-hand placement are unknown."}
    if camera.get("intrinsics"):
        intrinsics = camera.get("intrinsics")
        if not isinstance(intrinsics, dict) or any(key not in intrinsics for key in ("fx", "fy", "cx", "cy")):
            raise ValueError("Resolved input must supply fx/fy/cx/cy")
        values = np.asarray([intrinsics[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
        if not np.isfinite(values).all() or np.any(values[:2] <= 0):
            raise ValueError("Invalid calibrated camera intrinsics")
        distortion = camera.get("distortion")
        if distortion is None:
            raise ValueError("Calibrated input must explicitly supply distortion coefficients")
        if len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("Unsupported or nonfinite calibrated distortion")
        image_space = camera.get("image_space")
        if image_space not in ("original_distorted", "undistorted_same_intrinsics", "undistorted_rectified_intrinsics"):
            raise ValueError("Calibrated observations must explicitly identify their original or undistorted image_space")
        # VideoView already undistorts calibrated frames, with a recorded
        # source-to-effective K remap for nonsquare calibration pixels.
        # The exported projection model must describe the observed pixels; keep
        # lens coefficients separately as provenance, never apply them twice.
        projection_distortion = ([0.] * len(distortion)
                                 if image_space.startswith("undistorted_") else list(distortion))
        return {"intrinsics": dict(zip(("fx", "fy", "cx", "cy"), values.tolist())),
                "distortion": projection_distortion, "calibrated": bool(camera.get("calibrated")),
                "image_space": image_space, "source_lens_distortion": list(camera.get("source_lens_distortion", distortion)),
                **({"source_intrinsics": dict(camera["source_intrinsics"])} if "source_intrinsics" in camera else {}),
                "intrinsics_source": camera.get("intrinsics_source", "explicit_calibration" if camera.get("calibrated") else "prior-based"),
                "provenance": camera.get("provenance", {}), "warnings": camera.get("warnings", []),
                "translation_method": "Resolved full-image camera; joint two-hand perspective translation solve; not ground truth"}
    raise ValueError("Intrinsics resolution did not return a camera")


def calibrated_translation(joints, pixels, camera):
    intrinsics = camera["intrinsics"]
    matrix = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                       [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1.]])
    rays = cv2.undistortPoints(np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
                               matrix, projection_distortion(camera)).reshape(-1, 2)
    rows, target = [], []
    for point, ray in zip(joints, rays):
        rows.extend(([1., 0., -ray[0]], [0., 1., -ray[1]]))
        target.extend((ray[0] * point[2] - point[0], ray[1] * point[2] - point[1]))
    translation, _, rank, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(target), rcond=None)
    if rank < 3 or not np.isfinite(translation).all() or np.any(joints[:, 2] + translation[2] <= 0):
        raise ValueError("Calibrated translation fit is degenerate or behind camera")
    return translation


@lru_cache(maxsize=1)
def _legacy_crop_convention():
    """Read the authenticated model convention only to decode old cached crop cameras."""
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"models/model_config.yaml").read_text(encoding="utf-8"))
    return float(cfg["EXTRA"]["FOCAL_LENGTH"]), int(cfg["MODEL"]["IMAGE_SIZE"])


def estimated_full_translation(hand, camera, side):
    if camera.get("intrinsics") is None:
        raise ValueError("Full-image translation requires camera intrinsics; crop conventions cannot provide them")
    diagnostics = hand["diagnostics"]
    weak = diagnostics.get("crop_weak_perspective_camera")
    if weak is not None:
        scale, tx, ty = np.asarray(weak, dtype=float)
    else:
        tx, ty, tz = diagnostics["crop_camera_translation_m"]
        if tz <= 0:
            raise ValueError("WiLoR crop camera depth must be positive")
        model_focal, model_size = _legacy_crop_convention()
        crop_focal = diagnostics.get("crop_focal_length_px", model_focal)
        crop_size = diagnostics.get("crop_image_size_px", model_size)
        scale = 2 * crop_focal / (crop_size * tz)
    if not np.isfinite([scale,tx,ty]).all() or scale <= 0:
        raise ValueError("Invalid crop weak-perspective camera")
    box = np.asarray(diagnostics["box_xyxy_px"], dtype=float)
    center = (box[:2] + box[2:]) / 2
    bs = diagnostics["box_size_px"] * scale + 1e-9
    if side == "left":
        tx = -tx
    intrinsics = camera["intrinsics"]
    return np.array([2 * (center[0] - intrinsics["cx"]) / bs + tx,
                     2 * (center[1] - intrinsics["cy"]) / bs + ty,
                     2 * intrinsics["fx"] / bs])


def availability_score(hand, view_record):
    identity = hand.get("diagnostics", {}).get("hand_identity", {})
    score, factors = .6, {"observed_base": .6}
    for condition, name, factor in (
        (identity.get("corrected"), "side_corrected", .7),
        (identity.get("ambiguous"), "identity_ambiguous", .5),
        (identity.get("reacquired"), "identity_reacquired", .75),
        (view_record.get("blur_flag"), "image_blur_flag", .75),
        (bool(hand.get("outside_image_joints")), "outside_image_joints", .8),
    ):
        if condition:
            score *= factor
            factors[name] = factor
    return score, factors


def select_animation_observations(frames, view_id):
    """Keep a unique observation per side, without guessing unresolved identity.

    The temporal tracker deliberately retains native crop handedness when an
    association is ambiguous. Those observations are valid source evidence but
    cannot both drive the same animation hand. Exclude only explicitly ambiguous
    members of a conflicting group; never relabel already reconstructed MANO.
    """
    source_by_side = {"left": {}, "right": {}}
    excluded = []
    for frame in frames:
        selected = frame["views"][view_id]
        grouped = {"left": [], "right": []}
        for index, hand in enumerate(selected.get("hands", [])):
            side = hand["side"]
            if side not in grouped:
                raise ValueError("Input has unknown hand side; resolve identity before animation export")
            if hand.get("mano") is None:
                raise ValueError("Real MANO parameters required; MediaPipe-only poses cannot populate this export")
            shape = np.asarray(hand["mano"]["shape"], dtype=float)
            if shape.shape != (10,) or not np.isfinite(shape).all():
                raise ValueError("Each real observation must supply ten finite MANO shape values")
            grouped[side].append((index, hand))
        for side, candidates in grouped.items():
            if len(candidates) > 1:
                uncertain = [(index, hand) for index, hand in candidates
                             if hand.get("diagnostics", {}).get("hand_identity", {}).get("ambiguous") is True]
                retained = [(index, hand) for index, hand in candidates
                            if hand.get("diagnostics", {}).get("hand_identity", {}).get("ambiguous") is not True]
                if not uncertain or len(retained) > 1:
                    raise ValueError("Input has duplicate hand side without resolved identity; resolve identity before animation export")
                for index, hand in uncertain:
                    excluded.append({"frame": frame["frame"], "side": side,
                                     "source_hand_index": index,
                                     "reason": "unresolved_same_side_identity",
                                     "hand_identity": hand["diagnostics"]["hand_identity"]})
                candidates = retained
            if candidates:
                source_by_side[side][frame["frame"]] = (candidates[0][1], selected)
    if not all(source_by_side.values()):
        raise ValueError("Both hands need at least one usable real observation after identity conflict filtering")
    return source_by_side, excluded


class FixedShapeGeometry:
    def __init__(self, mano_path, betas, scale_factor=1.0):
        arrays = load_mano_arrays(mano_path)
        self.arrays = arrays
        self.scale_factor = float(scale_factor)
        self.model = MANOLayer(str(mano_path), data_struct=Struct(**arrays), is_rhand=True,
                               use_pca=False, dtype=torch.float64).eval()
        self.betas = np.asarray(betas, dtype=np.float64)
        self.parents = arrays["kintree_table"][0].astype(int)
        self.parents[0] = -1

    def evaluate(self, side_rotations, side, *, return_vertices=False):
        matrices = np.asarray(side_rotations, dtype=np.float64)
        if side == "left":
            matrices = LEFT_REFLECTION @ matrices @ LEFT_REFLECTION
        with torch.inference_mode():
            result = self.model(global_orient=torch.from_numpy(matrices[:, :1]),
                hand_pose=torch.from_numpy(matrices[:, 1:]),
                betas=torch.from_numpy(np.tile(self.betas, (len(matrices), 1))))
        joints16 = result.joints.numpy() * self.scale_factor
        vertices = result.vertices.numpy() * self.scale_factor
        joints21 = np.concatenate((joints16, vertices[:, TIP_INDICES]), axis=1)[:, JOINT_MAP_21]
        if side == "left":
            joints16 = joints16 @ LEFT_REFLECTION.T
            joints21 = joints21 @ LEFT_REFLECTION.T
            vertices = vertices @ LEFT_REFLECTION.T
        return (joints16, joints21, vertices) if return_vertices else (joints16, joints21)


def export_cardcap(observations_path, output_dir, *, view_id=None, bone_mapping_path=None, research_output_dir=None,
                   card_anchors=(), anthropometry_path=None, solve_config=None):
    pipeline = Path(__file__).resolve().parents[1]
    plugin = pipeline.parent
    observations_path, output_dir = Path(observations_path), Path(output_dir)
    source_hash = sha256(observations_path)
    data = json.loads(observations_path.read_text(encoding="utf-8"))
    if data.get("format_version") != "cardcap.hand_observations/1.0":
        raise ValueError("Expected real cardcap.hand_observations/1.0 input")
    views = data["meta"]["views"]
    view_id = view_id or data["meta"]["primary_view"]
    view = next((item for item in views if item["id"] == view_id), None)
    if view is None:
        raise ValueError("Selected view is absent")
    frames = data["frames"]
    frame_count = len(frames)
    if [item["frame"] for item in frames] != list(range(frame_count)):
        raise ValueError("Animation preparation requires consecutive source timeline samples")
    mapping_path = Path(bone_mapping_path or plugin / "Config/BoneMapping_UE5Mannequin.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    source_by_side, excluded = select_animation_observations(frames, view_id)
    original_count = sum(len(frame["views"][view_id].get("hands", [])) for frame in frames)
    # Preserve frame/detection order for unchanged inputs and shape aggregation.
    all_betas = [hand["mano"]["shape"] for frame in frames
                 for hand in frame["views"][view_id].get("hands", [])
                 if frame["frame"] in source_by_side[hand["side"]]
                 and source_by_side[hand["side"]][frame["frame"]][0] is hand]
    selection = {"policy": "Exclude explicitly ambiguous duplicate-side observations; do not relabel source hands",
                 "original_observation_count": original_count, "retained_observation_count": len(all_betas),
                 "excluded_observation_count": len(excluded),
                 "excluded_frame_ranges": frame_ranges(item["frame"] for item in excluded),
                 "excluded_observations": excluded}
    shared_shape = np.median(np.asarray(all_betas), axis=0)
    geometry = FixedShapeGeometry(pipeline / "models/MANO_RIGHT.pkl", shared_shape)
    if not np.array_equal(geometry.parents, mapping["mano_parents"]):
        raise ValueError("Configured bone hierarchy differs from actual MANO")
    camera = camera_configuration(view, [hand for side in source_by_side.values() for hand, _ in side.values()])
    shared_camera = camera["intrinsics"] is not None
    from .solve.hand_scale_policy import resolve_hand_scale
    from .solve.joint_translation import solve_joint_translations
    config = solve_config or json.loads((plugin / "Config/HandSolve.json").read_text(encoding="utf-8"))
    scale = resolve_hand_scale(geometry.arrays, shared_shape, source_hash, sha256(pipeline/"models/MANO_RIGHT.pkl"),
                              camera, card_anchors=card_anchors, anthropometry_path=anthropometry_path, config=config)
    geometry.scale_factor = scale["global_scale_factor_applied"]
    metric_valid = scale["meters_per_unit"] is not None
    units = "ue_centimeters" if metric_valid else "conditional_ue_units"
    local_geometries, local_meshes, source_wrists = {}, {}, {}
    hands, per_side_validation, all_confidences, low_frames = [], {}, [], [item["frame"] for item in excluded]
    for side, samples in source_by_side.items():
        observed = {}
        ordered_frames = sorted(samples)
        side_matrices = []
        for frame in ordered_frames:
            mano = samples[frame][0]["mano"]
            angles = np.r_[np.asarray(mano["global_orient_axis_angle"]).reshape(1, 3),
                           np.asarray(mano["hand_pose_axis_angle"]).reshape(15, 3)]
            if not np.isfinite(angles).all():
                raise ValueError("Nonfinite observed MANO rotations")
            side_matrices.append(np.asarray([cv2.Rodrigues(angle.astype(np.float64))[0] for angle in angles]))
        side_matrices = np.asarray(side_matrices)
        source_joints16, source_joints21 = geometry.evaluate(side_matrices, side)
        source_wrists[side] = {frame: source_joints16[index, 0].copy()
                               for index, frame in enumerate(ordered_frames)}
        for index, frame in enumerate(ordered_frames):
            hand, view_record = samples[frame]
            camera_translation = estimated_full_translation(hand, camera, side) * geometry.scale_factor if shared_camera else None
            global_wrist = source_joints16[index, 0] + camera_translation if shared_camera else np.zeros(3)
            rotations_ue = CAMERA_TO_UE @ side_matrices[index] @ CAMERA_TO_UE.T
            score, factors = availability_score(hand, view_record)
            observed[frame] = {"rotations": np.asarray([matrix_to_quaternion(matrix) for matrix in rotations_ue]),
                "translation": CAMERA_TO_UE @ global_wrist * 100,
                "confidence": score, "score_factors": factors, "source_hand": hand,
                "source_camera_translation_m": camera_translation.tolist() if shared_camera else None,
                "fixed_shape_original_wrist_m": source_joints16[index, 0].tolist()}
        filled = fill_samples(observed, frame_count)
        all_side_rotations = np.asarray([[CAMERA_TO_UE.T @ quaternion_to_matrix(q) @ CAMERA_TO_UE
                                          for q in item["rotations"]] for item in filled])
        joints16, joints21, vertices = geometry.evaluate(all_side_rotations, side, return_vertices=True)
        local_geometries[side] = np.ascontiguousarray(joints21 - joints16[:, [0]])
        local_meshes[side] = np.ascontiguousarray(vertices - joints16[:, [0]])
        bone_lengths = np.linalg.norm(joints16[:, 1:] - joints16[:, geometry.parents[1:]], axis=2)
        lengths_deviation_mm = np.max(np.abs(bone_lengths - bone_lengths[[0]]), axis=0) * 1000
        output_frames = []
        for index, item in enumerate(filled):
            measured = item["sample_kind"] == "observed"
            score = item["confidence"] if measured else 0.0
            all_confidences.append(score)
            if score < .5:
                low_frames.append(index)
            points_ue = (joints21[index] - joints16[index, [0]]) @ CAMERA_TO_UE.T * 100 + item["translation"]
            source_hand = item.get("source_hand")
            diagnostics = {"sample_kind": item["sample_kind"], "source_observation_frames": item["source_frames"],
                "interpolation_alpha": item["alpha"], "is_new_source_observation": False,
                "original_observation_present": measured,
                "excluded_identity_observations": [entry for entry in excluded if entry["frame"] == index and entry["side"] == side],
                "confidence_semantics": "Noncalibrated animation usability heuristic, not pose probability or measured accuracy",
                "confidence_factors": item.get("score_factors") if measured else {"missing_observation": 0},
                "native_handedness_confidence": source_hand["handedness_confidence"] if measured else None,
                "hand_identity": source_hand["diagnostics"].get("hand_identity") if measured else None,
                "source_camera_translation_m": item.get("source_camera_translation_m") if metric_valid else None,
                "fixed_shape_original_wrist_m": item.get("fixed_shape_original_wrist_m") if metric_valid else None,
                "source_camera_translation_model_units": item.get("source_camera_translation_m") if not metric_valid else None,
                "fixed_shape_original_wrist_model_units": item.get("fixed_shape_original_wrist_m") if not metric_valid else None,
                "position_units": "meters" if metric_valid else "relative_model_units",
                **source_crop_diagnostics(source_hand if measured else None)}
            output_frames.append({"frame": index, "confidence": score,
                "sample_kind": "detected_model_observation" if measured else item["sample_kind"],
                "validity": {"pose": True, "global_translation": shared_camera},
                "global_trans_cm": item["translation"].tolist() if shared_camera else None,
                "global_rot_quat": item["rotations"][0].tolist(),
                "joint_positions_cm": points_ue.tolist(),
                "bone_rotations": {name: item["rotations"][bone].tolist()
                                   for bone, name in enumerate(mapping["hands"][side]["bone_names"][1:], 1)},
                "occluded_joints": sorted(set(source_hand.get("outside_image_joints", []))) if measured else list(range(21)),
                "diagnostics": diagnostics})
        quaternions = np.asarray([item["rotations"] for item in filled])
        unit_error = float(np.max(np.abs(np.linalg.norm(quaternions, axis=2) - 1)))
        minimum_dot = float(np.min(np.sum(quaternions[1:] * quaternions[:-1], axis=2))) if frame_count > 1 else 1.
        assert unit_error < 1e-10 and minimum_dot >= -1e-12
        assert np.max(lengths_deviation_mm) < 1e-6
        hands.append({"side": side, "mano_shape": shared_shape.tolist(), "frames": output_frames})
        per_side_validation[side] = {
            "original_observations": len(observed), "animation_samples": frame_count,
            "excluded_identity_observation_count": sum(entry["side"] == side for entry in excluded),
            "missing_source_frames": [item["frame"] for item in filled if item["sample_kind"] != "observed"],
            "interpolated_frames": [item["frame"] for item in filled if item["sample_kind"] == "interpolated"],
            "endpoint_hold_frames": [item["frame"] for item in filled if item["sample_kind"] == "endpoint_hold"],
            "source_16_bones_max_length_deviation_mm": float(np.max(lengths_deviation_mm)),
            "per_bone_max_length_deviation_mm": dict(zip(mapping["hands"][side]["bone_names"][1:], lengths_deviation_mm.tolist())),
            "rest_bone_lengths_mm": dict(zip(mapping["hands"][side]["bone_names"][1:], (bone_lengths[0] * 1000).tolist())),
            "quaternion_unit_max_abs_error": unit_error, "minimum_adjacent_same_bone_quaternion_dot": minimum_dot}
        if not metric_valid:
            for name in ("source_16_bones_max_length_deviation_mm", "per_bone_max_length_deviation_mm", "rest_bone_lengths_mm"):
                per_side_validation[side][name.replace("_mm", "_model_units_x1000")] = per_side_validation[side][name]
                per_side_validation[side][name] = None
            per_side_validation[side]["length_semantics"] = "Model-source lengths times 1000 for numerical comparison; not physical millimeters"
    if shared_camera:
        ordered_sides = ("left", "right")
        target_pixels = np.full((frame_count,2,21,2), np.nan)
        model_projection_pixels = np.full_like(target_pixels,np.nan)
        observed_mask = np.zeros((frame_count,2), dtype=bool)
        target_sources = {"detector_image_landmarks":0,"legacy_model_projected_landmarks":0}
        for si, side in enumerate(ordered_sides):
            for frame, (hand, _) in source_by_side[side].items():
                detector_pixels = hand["diagnostics"].get("detector_image_landmarks_px")
                target_pixels[frame,si] = detector_pixels if detector_pixels is not None else hand["image_landmarks_px"]
                if detector_pixels is not None:
                    model_projection_pixels[frame,si] = hand["image_landmarks_px"]
                observed_mask[frame,si] = True
                target_sources["detector_image_landmarks" if detector_pixels is not None else "legacy_model_projected_landmarks"] += 1
        initial_wrists = np.stack([np.asarray([f["global_trans_cm"] for f in h["frames"]]) @ CAMERA_TO_UE / 100 for h in hands],axis=1)
        solved_wrists, joint_report = solve_joint_translations(
            np.stack([local_geometries[s] for s in ordered_sides],axis=1), initial_wrists,target_pixels,observed_mask,camera,
            vertices_m=np.stack([local_meshes[s] for s in ordered_sides],axis=1),
            faces=[geometry.arrays["f"][:,[0,2,1]],geometry.arrays["f"]], config=config,
            model_projection_pixels=model_projection_pixels)
        joint_report["target_sources"] = target_sources
        c = camera["intrinsics"]
        k = np.array([[c["fx"],0,c["cx"]],[0,c["fy"],c["cy"]],[0,0,1.]])
        reprojection_errors = []
        for si, hand in enumerate(hands):
            for fi, sample in enumerate(hand["frames"]):
                change = CAMERA_TO_UE @ (solved_wrists[fi,si]-initial_wrists[fi,si])*100
                sample["global_trans_cm"] = (np.asarray(sample["global_trans_cm"])+change).tolist()
                sample["joint_positions_cm"] = (np.asarray(sample["joint_positions_cm"])+change).tolist()
                sample["diagnostics"]["initial_wrist_translation_cm"] = (CAMERA_TO_UE@initial_wrists[fi,si]*100).tolist()
                sample["diagnostics"]["translation_solver"] = "joint_two_hand_trf_huber_then_sliding_windows"
                sample["diagnostics"]["reprojection_target"] = "image_detector" if fi in source_by_side[hand["side"]] and source_by_side[hand["side"]][fi][0]["diagnostics"].get("detector_image_landmarks_px") is not None else "legacy_model_projection" if observed_mask[fi,si] else None
                if observed_mask[fi,si]:
                    projected = cv2.projectPoints(local_geometries[hand["side"]][fi],np.zeros(3),solved_wrists[fi,si],k,projection_distortion(camera))[0].reshape(-1,2)
                    reprojection_errors.extend(np.linalg.norm(projected-target_pixels[fi,si],axis=1).tolist())
        joint_report["final_reprojection_rms_px"] = float(np.sqrt(np.mean(np.square(reprojection_errors))))
        joint_report["final_reprojection_mean_px"] = float(np.mean(reprojection_errors))
        conditional_distance = float(np.median(np.linalg.norm(solved_wrists[:,0]-solved_wrists[:,1],axis=1))*100)
        joint_report["coordinate_units"] = units
        joint_report["per_frame_pass_units"] = "meters" if metric_valid else "relative_model_units"
        if not metric_valid:
            for name in ("initial_intersection_volume_cm3", "final_intersection_volume_cm3"):
                joint_report[name.replace("_cm3", "_model_units_cubed_x1e6")] = joint_report[name]
                joint_report[name] = None
        joint_report["final_wrist_distance_median_cm"] = conditional_distance if metric_valid else None
        joint_report["final_wrist_distance_median_display_units"] = conditional_distance
        mean_reprojection = joint_report["final_reprojection_mean_px"]
    else:
        mean_reprojection = None
        joint_report = {"status": "not_solved_unknown_camera", "steps": [],
            "coordinate_frame": "per_hand_wrist_local", "coordinate_units": units,
            "inter_hand_transform_known": False, "per_frame_pass": None,
            "final_reprojection_rms_px": None, "final_reprojection_mean_px": None,
            "final_wrist_distance_median_cm": None, "final_wrist_distance_median_display_units": None,
            "initial_intersection_volume_cm3": None, "final_intersection_volume_cm3": None,
            "reason": "No full-image K evidence. Crop camera normalization cannot determine full-image depths or relative hand placement.",
            "independent_metric_accuracy_verified": False}
        for hand in hands:
            for sample in hand["frames"]:
                sample["diagnostics"]["translation_solver"] = "not_solved_unknown_camera"
                sample["diagnostics"]["reprojection_target"] = None
                sample["diagnostics"]["coordinate_frame"] = "per_hand_wrist_local"
    crop_residuals = []
    for side in source_by_side.values():
        for hand, _ in side.values():
            d = hand["diagnostics"]
            image_target = d.get("detector_image_landmarks_px")
            crop_size = d.get("crop_image_size_px")
            if image_target is not None and crop_size is not None:
                # The same crop affine resize acts on both 2D point sets; its
                # center and possible horizontal reflection cancel in norms.
                error = np.linalg.norm(np.asarray(hand["image_landmarks_px"]) - np.asarray(image_target),axis=1)
                crop_residuals.extend((error * float(crop_size) / float(d["box_size_px"])).tolist())
    crop_report = {"mean_px": float(np.mean(crop_residuals)) if crop_residuals else None,
        "rms_px": float(np.sqrt(np.mean(np.square(crop_residuals)))) if crop_residuals else None,
        "point_count": len(crop_residuals),
        "space": "original per-observation model crop pixels",
        "meaning": "Original model crop-projection points versus image detector landmarks; neither is ground truth. Not full-image or final local-animation reprojection; no camera K or depth is inferred."}
    joint_report["source_crop_model_detector_residual"] = crop_report
    from .geometry_checks import assess_geometry
    geometry_review = assess_geometry(solved_wrists if shared_camera else None,
        scale["neutral_hand_measurement"]["length_source_units"] * geometry.scale_factor,
        global_geometry_known=shared_camera)
    geometry_review["denominator_definition"] = scale["neutral_hand_measurement"]["definition"]
    geometry_review["scope"] = "User-reviewed cooperative-cardistry plausibility screen, not ground truth or a solver constraint"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry_review.json").write_text(json.dumps(geometry_review,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if shared_camera and geometry_review["status"] != "passed":
        raise ValueError("Shared-space geometry review failed; inspect geometry_review.json. Geometry is not changed to satisfy the criteria.")
    warnings = [
        "Local MANO/WiLoR/SMPL-X research assets and outputs retain applicable noncommercial/restricted terms; no redistribution clearance.",
        "This single-view reconstruction is not calibrated world motion or verified physical scale; no card-PnP or ruler scale anchor was performed.",
        "No population hand length is applied. Without a measured anchor, model geometry remains relative and legacy *_cm names denote conditional UE display units, not physical centimeters.",
        "Per-frame confidence is a noncalibrated usability heuristic; native handedness classification is diagnostic only, not pose accuracy.",
        "Missing rotations use shortest-path SLERP or endpoint holds. With supported K, missing translations interpolate or hold optimized observed anchors; without K all global translations remain null. Missing detector observations stay absent, confidence zero and all joints marked occluded.",
        "Left hand uses canonical-right shape reflected in camera X and F R F rotations; native MANO_LEFT with the same betas is not equivalent.",
        "Constant-length validation covers the 15 edges of the true 16-bone MANO skeleton, not pose-corrected skinned fingertip distances.",
        "Reported reprojection errors compare observed image detector landmarks, or explicitly marked legacy model projections; they are not ground-truth pose accuracy.",
        "GLB UE top4 preview approximates skin weights and omits pose-corrective blend shapes; full-weight GLB/source arrays retained.",
    ]
    if not camera["calibrated"]:
        warnings.append("Camera intrinsics and relative hand placement are unknown. Source geometry remains wrist-local. The separate display configuration offers an assumed common space and optional local views; neither establishes source camera parameters or physical scale.")
    if excluded:
        warnings.append(f"Excluded {len(excluded)} unresolved same-side identity observations in {len(set(item['frame'] for item in excluded))} frames from animation; source detections unchanged. Missing animation samples remain zero-confidence fills, not recovered observations.")
    warnings.extend(camera.get("warnings", []))
    source_video_hash = sha256(view["source_video"])
    if source_video_hash != view["source_sha256"]:
        raise ValueError("Source video hash differs from reconstruction provenance")
    capture = {"format_version": "1.3",
        "meta": {"source_video": str(Path(view["source_video"])).replace("\\", "/"), "fps": data["meta"]["fps"],
            "frame_count": frame_count, "resolution": view["resolution"],
            "processed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "pipeline_version": f"CardistryCapture {__version__}",
            "source_video_sha256": source_video_hash, "source_observations_sha256": source_hash,
            "selected_view": view_id, "original_hand_observation_count": original_count,
            "animation_source_hand_observation_count": len(all_betas),
            "shape_estimation": "Componentwise robust median of retained real ten-beta estimates; one person, shared both-hand shape"},
        "camera": camera,
        "scale": scale,
        "provenance": {"camera_intrinsics": "calibrated" if camera["calibrated"] else "unobservable",
            "camera_distortion": "calibrated" if camera["calibrated"] else "unobservable",
            "metric_scale": scale.get("provenance", {}).get("kind", "unobservable"),
            "hand_geometry": "inferred", "coordinate_units": units,
            "coordinate_frame": "shared_camera" if shared_camera else "per_hand_wrist_local",
            "assumptions": camera["provenance"].get("assumptions", []) + [
                "One person's shared median MANO shape; model rotations are conditional predictions. Translations are solved only when camera K is supported.",
                "Interpolation and endpoint holding provide animation continuity, not additional observations.",
                "Legacy *_cm geometry fields and GLB use the same declared coordinate units; actor and component transforms stay identity."]},
        "validity": {"camera_intrinsics": camera["intrinsics"] is not None,
            "camera_distortion": camera["distortion"] is not None, "metric_scale": metric_valid,
            "inter_hand_transform": shared_camera},
        "hands": hands, "packets": [], "events": {"splits": [], "merges": [], "releases": []},
        "quality": {"mean_hand_confidence": float(np.mean(all_confidences)),
                    "low_confidence_ranges": frame_ranges(low_frames), "mean_reprojection_error_px": mean_reprojection,
                    "warnings": warnings, "source_observation_selection": selection,
                    "source_crop_model_detector_residual": crop_report,
                    "geometry_review_status": geometry_review["status"]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joint_translation_report.json").write_text(json.dumps(joint_report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    capture_path = output_dir / "capture.cardcap.json"
    capture_path.write_text(json.dumps(capture, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    display_path = None
    display_status = "not_required_calibrated_camera"
    display_unavailable_reason = None
    if not shared_camera:
        try:
            needs_legacy_crop_convention = any(
                hand["diagnostics"].get("crop_weak_perspective_camera") is None and
                hand["diagnostics"].get("crop_camera_translation_m") is not None and
                (hand["diagnostics"].get("crop_focal_length_px") is None or
                 hand["diagnostics"].get("crop_image_size_px") is None)
                for samples in source_by_side.values() for hand, _ in samples.values())
            legacy_convention = None
            if needs_legacy_crop_convention:
                try:
                    legacy_convention = _legacy_crop_convention()
                except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
                    raise DisplayInputError("The recorded model convention needed to decode cached crop cameras is unavailable or invalid") from error
            display = build_display_space(source_by_side, source_wrists, frame_count, view["resolution"],
                data["meta"]["fps"], scale["neutral_hand_measurement"]["length_source_units"] * geometry.scale_factor,
                model_scale=geometry.scale_factor, legacy_crop_convention=legacy_convention)
        except DisplayInputError as error:
            display_status = "unavailable"
            display_unavailable_reason = str(error)
            unavailable = {"display_only": True, "status": display_status, "reason": display_unavailable_reason,
                "capture_sha256": sha256(capture_path), "source_observations_sha256": source_hash,
                "local_pose_export_available": True}
            (output_dir / "display_space_unavailable.json").write_text(
                json.dumps(unavailable, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        else:
            display_status = "available"
            display["capture_sha256"] = sha256(capture_path)
            display["source_observations_sha256"] = source_hash
            display["neutral_hand_length_definition"] = scale["neutral_hand_measurement"]["definition"]
            display_path = output_dir / "display_space.json"
            display_path.write_text(json.dumps(display, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    # Repeated take/animation leaf names must not collide across source runs.
    research_dir = Path(research_output_dir or output_dir / "hand_mesh")
    glb = export_research_hands(bone_mapping_path=mapping_path, output_dir=research_dir, betas=shared_shape,
                                meters_per_mano_unit=geometry.scale_factor, coordinate_units=units)
    if sha256(observations_path) != source_hash:
        raise RuntimeError("Source observation file changed during export")
    validation = {"status": "passed", "gpu_inference_performed": False, "source_observations_unchanged": True,
        "capture_path": str(capture_path.resolve()), "capture_sha256": sha256(capture_path),
        "source_observations_path": str(observations_path.resolve()), "source_observations_sha256": source_hash,
        "source_video_sha256": source_video_hash, "original_observation_count": original_count,
        "retained_animation_observation_count": len(all_betas), "source_observation_selection": selection,
        "animation_sample_count": sum(len(hand["frames"]) for hand in hands), "source_timeline_frames": frame_count,
        "shared_mano_shape": shared_shape.tolist(), "shape_statistic": "componentwise median across retained actual observations",
        "per_side": per_side_validation, "camera_calibrated": camera["calibrated"],
        "coordinate_frame": "shared_camera" if shared_camera else "per_hand_wrist_local",
        "inter_hand_transform_known": shared_camera, "source_crop_model_detector_residual": crop_report,
        "geometry_review": str((output_dir / "geometry_review.json").resolve()),
        "camera_to_ue_axis": CAMERA_TO_UE.tolist(), "meters_to_cm": 100 if metric_valid else None,
        "coordinate_units": units, "model_units_to_ue_display_units": 100,
        "low_confidence_ranges": capture["quality"]["low_confidence_ranges"],
        "mean_hand_usability_heuristic": capture["quality"]["mean_hand_confidence"],
        "bone_mapping_sha256": sha256(mapping_path), "research_hands_manifest": glb["manifest_path"],
        "research_glb_outputs": glb["outputs"], "mean_reprojection_error_px": mean_reprojection,
        "joint_translation_report": str((output_dir/"joint_translation_report.json").resolve()),
        "joint_wrist_distance_median_cm": joint_report["final_wrist_distance_median_cm"], "scale": scale,
        "display_space_config": str(display_path.resolve()) if display_path else None,
        "display_space_sha256": sha256(display_path) if display_path else None,
        "display_space_status": display_status, "display_space_unavailable_reason": display_unavailable_reason}
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return validation
