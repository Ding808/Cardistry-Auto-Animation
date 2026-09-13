"""Conditional common-space viewing without changing reconstructed source data.

The focal value here is a per-clip display assumption. Geometry criteria select
a useful view; they cannot establish camera calibration or physical accuracy.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from .geometry_checks import DEFAULT_CRITERIA, assess_geometry

CAMERA_TO_UE = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
SIDES = ("left", "right")
CHECKS = ("wrist_distance_over_hand_length", "left_depth_over_hand_length",
          "right_depth_over_hand_length", "depth_difference_over_hand_length")


class DisplayInputError(ValueError):
    """Known missing or invalid inputs prevent only conditional common viewing."""


def _numeric(value, name):
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise DisplayInputError(f"{name} must contain numeric values") from error


def source_crop_diagnostics(hand):
    """Copy original crop predictions, including native tz, without unit claims."""
    diagnostics = hand.get("diagnostics", {}) if hand else {}
    return {"source_" + name: deepcopy(diagnostics.get(key)) for name, key in (
        ("crop_camera_translation_model_units", "crop_camera_translation_m"),
        ("crop_weak_perspective_camera", "crop_weak_perspective_camera"),
        ("crop_focal_length_px", "crop_focal_length_px"),
        ("crop_image_size_px", "crop_image_size_px"),
        ("crop_box_xyxy_px", "box_xyxy_px"),
        ("crop_box_size_px", "box_size_px"))}


def crop_display_affine(hand, side, resolution, original_wrist, *,
                        model_scale=1., legacy_crop_convention=None):
    """Return camera-basis wrist = constant + focal_px * slope in model units.

    Crop XY offsets and the original shaped wrist are retained. Only the optical
    depth term depends on the assumed full-image focal. Left crop tx is reflected
    once, matching the canonical-right reconstruction's output convention.
    """
    if side not in SIDES:
        raise DisplayInputError("Display hand side must be left or right")
    dimensions = _numeric(resolution, "Display resolution")
    if dimensions.shape != (2,):
        raise DisplayInputError("Display resolution requires width and height")
    width, height = dimensions
    wrist = _numeric(original_wrist, "Original shaped wrist")
    if (not np.isfinite([width, height, model_scale]).all() or
            min(width, height, model_scale) <= 0 or wrist.shape != (3,) or not np.isfinite(wrist).all()):
        raise DisplayInputError("Display geometry requires finite positive dimensions and a finite wrist")
    diagnostics = hand.get("diagnostics") if isinstance(hand, Mapping) else None
    if not isinstance(diagnostics, Mapping):
        raise DisplayInputError("Source crop camera diagnostics are unavailable")
    weak = diagnostics.get("crop_weak_perspective_camera")
    if weak is not None:
        weak = _numeric(weak, "Crop weak camera")
        if weak.shape != (3,):
            raise DisplayInputError("Crop weak camera must have three components")
        scale, tx, ty = weak
    else:
        native = _numeric(diagnostics.get("crop_camera_translation_m"), "Native crop camera")
        if native.shape != (3,) or not np.isfinite(native).all() or native[2] <= 0:
            raise DisplayInputError("A valid native crop camera is required for a display hypothesis")
        focal = diagnostics.get("crop_focal_length_px")
        size = diagnostics.get("crop_image_size_px")
        if focal is None or size is None:
            if legacy_crop_convention is None:
                raise DisplayInputError("Legacy crop camera decoding needs its recorded model convention")
            fallback_focal, fallback_size = legacy_crop_convention
            focal = fallback_focal if focal is None else focal
            size = fallback_size if size is None else size
        focal, size = _numeric([focal, size], "Crop focal and image size")
        if not np.isfinite([focal, size]).all() or min(focal, size) <= 0:
            raise DisplayInputError("Invalid crop focal or image size")
        tx, ty, tz = native
        scale = 2. * focal / (size * tz)
    box = _numeric(diagnostics.get("box_xyxy_px"), "Crop box")
    box_size = _numeric(diagnostics.get("box_size_px"), "Crop box size")
    if box_size.shape != ():
        raise DisplayInputError("Crop box size must be a scalar")
    box_size = float(box_size)
    if (box.shape != (4,) or not np.isfinite(box).all() or
            np.any(box[2:] <= box[:2]) or not np.isfinite([scale, tx, ty, box_size]).all() or
            min(scale, box_size) <= 0):
        raise DisplayInputError("Invalid crop box or weak camera")
    center = (box[:2] + box[2:]) / 2.
    denominator = box_size * scale
    tx = -tx if side == "left" else tx
    constant = wrist + model_scale * np.array([
        2. * (center[0] - width / 2.) / denominator + tx,
        2. * (center[1] - height / 2.) / denominator + ty, 0.])
    slope = np.array([0., 0., model_scale * 2. / denominator])
    return constant, slope


def _arrays(constants, slopes, hand_length):
    constants, slopes = _numeric(constants, "Display constants"), _numeric(slopes, "Display depth slopes")
    if (constants.ndim != 3 or constants.shape[1:] != (2, 3) or not len(constants) or
            constants.shape != slopes.shape or not np.isfinite(constants).all() or not np.isfinite(slopes).all()):
        raise DisplayInputError("Display affine wrists must be finite N x 2 x 3 arrays")
    if not np.isfinite(hand_length) or hand_length <= 0:
        raise DisplayInputError("A positive neutral hand length is required")
    if np.any(slopes[:, :, :2] != 0.) or np.any(slopes[:, :, 2] <= 0):
        raise DisplayInputError("Display focal may change only positive optical depth slopes")
    return constants, slopes


def _failure_matrix(points, length, criteria):
    difference = points[:, 0] - points[:, 1]
    depth = points[:, :, 2] / length
    return np.column_stack((
        np.linalg.norm(difference, axis=1) / length >= criteria.wrist_distance_over_hand_length_max,
        (depth[:, 0] < criteria.depth_over_hand_length_min) | (depth[:, 0] > criteria.depth_over_hand_length_max),
        (depth[:, 1] < criteria.depth_over_hand_length_min) | (depth[:, 1] > criteria.depth_over_hand_length_max),
        np.abs(difference[:, 2]) / length >= criteria.depth_difference_over_hand_length_max))


def select_display_focal(constants, slopes, hand_length, *, criteria=DEFAULT_CRITERIA):
    """Minimize per-frame gate failures over all positive focal values.

    Gates are piecewise constant in focal. Enumerating their analytic transition
    points and every intervening interval avoids an orientation-dependent focal
    range or a sampled grid that could miss a narrow feasible interval.
    """
    constants, slopes = _arrays(constants, slopes, hand_length)
    if (criteria.depth_over_hand_length_min <= 0 or
            criteria.depth_over_hand_length_max <= criteria.depth_over_hand_length_min or
            min(criteria.wrist_distance_over_hand_length_max, criteria.depth_difference_over_hand_length_max) <= 0):
        raise DisplayInputError("Display selection requires ordered positive geometry criteria")
    boundaries = []
    for threshold in (criteria.depth_over_hand_length_min, criteria.depth_over_hand_length_max):
        boundaries.extend(((threshold * hand_length - constants[:, :, 2]) / slopes[:, :, 2]).ravel())
    difference = constants[:, 0] - constants[:, 1]
    difference_slope = slopes[:, 0, 2] - slopes[:, 1, 2]
    for difference_limit in (criteria.depth_difference_over_hand_length_max * hand_length,):
        nonzero = difference_slope != 0
        for sign in (-1., 1.):
            boundaries.extend((sign * difference_limit - difference[nonzero, 2]) / difference_slope[nonzero])
    remaining = (criteria.wrist_distance_over_hand_length_max * hand_length) ** 2 - np.sum(difference[:, :2] ** 2, axis=1)
    solvable = (remaining >= 0) & (difference_slope != 0)
    for sign in (-1., 1.):
        boundaries.extend((sign * np.sqrt(remaining[solvable]) - difference[solvable, 2]) / difference_slope[solvable])
    boundaries = np.unique([float(x) for x in boundaries if np.isfinite(x) and x > 0])
    # Tie preference comes from the middle of the declared depth interval for
    # this clip's observations, never image width, a lens claim or another clip.
    depth_middle = (criteria.depth_over_hand_length_min + criteria.depth_over_hand_length_max) / 2.
    preferred = ((depth_middle * hand_length - constants[:, :, 2]) / slopes[:, :, 2]).ravel()
    preferred = preferred[np.isfinite(preferred) & (preferred > 0)]
    if preferred.size:
        preferred_focal = float(np.median(preferred))
        reference_source = "positive_clip_depth_interval_centers"
    else:
        # A tie reference never constrains the search. Wrists can already lie
        # beyond the depth interval's middle at f=0 while still admitting a
        # positive feasible focal, or the hypothesis can be infeasible for all f.
        preferred_focal = float(np.median(boundaries)) if boundaries.size else float(hand_length / np.median(slopes[:, :, 2]))
        reference_source = "positive_gate_boundaries" if boundaries.size else "one_hand_length_depth_increment"
    candidates = []
    if boundaries.size:
        # An exact inclusive boundary can be the only feasible point, but prefer
        # a stable interior whenever it has the same violation counts. Geometric
        # interval centers maximize distance from either boundary in log focal.
        candidates.extend((float(x), True, 0.) for x in boundaries)
        candidates.append((float(boundaries[0] / 2.), False, float(np.log(2.))))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            center = float(np.exp((np.log(left) + np.log(right)) / 2.))
            candidates.append((center, False, float(min(np.log(center / left), np.log(right / center)))))
        candidates.append((float(boundaries[-1] * 2.), False, float(np.log(2.))))
    else:
        candidates.append((preferred_focal, False, 0.))
    candidates = [(f, boundary, clearance) for f, boundary, clearance in candidates if np.isfinite(f) and f > 0]
    scores = []
    for focal, boundary, clearance in candidates:
        failed = _failure_matrix(constants + focal * slopes, hand_length, criteria)
        scores.append((int(failed.any(axis=1).sum()), int(failed.sum()),
                       boundary, -clearance, abs(float(np.log(focal / preferred_focal))), float(focal)))
    score = min(scores)
    focal = score[-1]
    failed = _failure_matrix(constants + focal * slopes, hand_length, criteria)
    return focal, {
        "method": "analytic_positive_focal_gate_intervals",
        "objective": ["fewest_frames_with_any_geometry_failure", "fewest_total_frame_gate_failures",
                      "interior_interval_before_exact_boundary", "largest_log_clearance_from_gate_boundaries",
                      "nearest_log_focal_to_clip_depth_interval_center", "smaller_focal"],
        "orientation_independent": True, "frame_count": len(constants),
        "candidate_count": len(candidates), "positive_gate_boundary_count": len(boundaries),
        "preferred_depth_center_focal_px": preferred_focal,
        "tie_reference_source": reference_source,
        "selected_at_gate_boundary": bool(score[2]), "selected_log_boundary_clearance": float(-score[3]),
        "failed_frame_count": score[0], "frame_gate_failure_count": score[1],
        "failed_frame_indices": np.flatnonzero(failed.any(axis=1)).tolist(),
        "per_gate_failed_frame_count": dict(zip(CHECKS, failed.sum(axis=0).tolist())),
        "criteria": asdict(criteria),
        "scope": "Display hypothesis selection only; passing a criterion does not validate source geometry or camera calibration.",
    }


def build_display_space(source_by_side, source_wrists, frame_count, resolution, fps,
                        neutral_hand_length, *, model_scale=1., legacy_crop_convention=None,
                        criteria=DEFAULT_CRITERIA):
    """Prepare an independent sidecar from retained observations and local pose.

    Missing display translations linearly interpolate observed affine coefficients
    or hold the nearest endpoint, with their original source frames retained.
    No source dictionary, camera, pose, shape or scale is modified.
    """
    if not isinstance(frame_count, int) or frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise DisplayInputError("Display timeline must have a positive frame count and rate")
    constants = np.empty((frame_count, 2, 3))
    slopes = np.empty_like(constants)
    provenance = {side: [] for side in SIDES}
    for si, side in enumerate(SIDES):
        observed = source_by_side.get(side, {})
        if not observed:
            raise DisplayInputError("Both hands need retained real observations for common-space display")
        keys = sorted(observed)
        if any(not isinstance(key, int) or key < 0 or key >= frame_count for key in keys):
            raise DisplayInputError("Display source frame is outside the clip timeline")
        affine = {}
        for index in keys:
            hand = observed[index][0]
            affine[index] = crop_display_affine(hand, side, resolution, source_wrists[side][index],
                model_scale=model_scale, legacy_crop_convention=legacy_crop_convention)
        for frame in range(frame_count):
            if frame in affine:
                constants[frame, si], slopes[frame, si] = affine[frame]
                details = {"sample_kind": "detected_model_observation", "source_observation_frames": [frame],
                    "interpolation_alpha": None, **source_crop_diagnostics(observed[frame][0])}
            else:
                insertion = int(np.searchsorted(keys, frame))
                if insertion == 0 or insertion == len(keys):
                    endpoint = keys[0] if insertion == 0 else keys[-1]
                    constants[frame, si], slopes[frame, si] = affine[endpoint]
                    details = {"sample_kind": "endpoint_hold", "source_observation_frames": [endpoint], "interpolation_alpha": None}
                else:
                    before, after = keys[insertion - 1], keys[insertion]
                    alpha = (frame - before) / (after - before)
                    constants[frame, si] = (1. - alpha) * affine[before][0] + alpha * affine[after][0]
                    slopes[frame, si] = (1. - alpha) * affine[before][1] + alpha * affine[after][1]
                    details = {"sample_kind": "interpolated", "source_observation_frames": [before, after], "interpolation_alpha": alpha}
                details.update(source_crop_diagnostics(None))
            details["is_new_source_observation"] = False
            provenance[side].append(details)
    focal, selection = select_display_focal(constants, slopes, neutral_hand_length, criteria=criteria)
    wrists = constants + focal * slopes
    geometry_review = assess_geometry(wrists, neutral_hand_length, global_geometry_known=False,
        numeric_hypothesis_available=True, criteria=criteria)
    failures = _failure_matrix(wrists, neutral_hand_length, criteria)
    rows = []
    for frame in range(frame_count):
        hands = {}
        for si, side in enumerate(SIDES):
            hands[side] = {**provenance[side][frame],
                "wrist_translation_display_units": (CAMERA_TO_UE @ wrists[frame, si] * 100.).tolist(),
                "wrist_translation_constant_display_units": (CAMERA_TO_UE @ constants[frame, si] * 100.).tolist(),
                "wrist_translation_per_focal_px_display_units": (CAMERA_TO_UE @ slopes[frame, si] * 100.).tolist()}
        rows.append({"frame": frame, "hands": hands,
            "wrist_distance_over_hand_length": float(np.linalg.norm(wrists[frame, 0] - wrists[frame, 1]) / neutral_hand_length),
            "geometry_failed_checks": [name for name, failed in zip(CHECKS, failures[frame]) if failed]})
    width, height = map(int, resolution)
    return {"format_version": "cardcap.display_space/1.0", "display_only": True,
        "mode": "assumed_common_camera", "frame_count": frame_count, "fps": float(fps),
        "resolution": [width, height], "display_assumed_focal_px": focal,
        "principal_point_px": [width / 2., height / 2.],
        "projection_assumption": "ideal_pinhole_square_pixels_centered_principal_point",
        "distortion_assumption": "ignored_for_display_only; source distortion remains unknown",
        "coordinate_basis": "ue_x_forward_y_right_z_up", "coordinate_units": "conditional_ue_display_units",
        "neutral_hand_length_display_units": float(neutral_hand_length * 100.),
        "display_label": "Common space uses an assumed focal length; uncalibrated",
        "source_camera_intrinsics": None, "source_camera_distortion": None,
        "source_inter_hand_transform_known": False, "metric_scale_validated": False,
        "translation_equation": "wrist_translation_constant_display_units + display_assumed_focal_px * wrist_translation_per_focal_px_display_units",
        "selection": selection, "geometry_review": geometry_review, "frames": rows,
        "assumptions": [
            "This clip's display focal is selected by per-frame plausibility gates, not camera estimation or a translation solver.",
            "The source capture retains null camera intrinsics, distortion and global translations; physical scale remains unknown.",
            "Missing display translations interpolate or hold observed affine anchors; they are not recovered observations.",
            "Geometry gate failures remain visible. Focal editing changes only this conditional view and never source data.",
            "Do not compare display depths, separations or focal assumptions between clips as physical measurements."]}
