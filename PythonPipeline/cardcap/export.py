"""Internal observations and atomic progress files; separate from UE interchange."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cardcap.hand.base import HandEstimate
from cardcap.atomic_file import replace_with_windows_sharing_retry

OBSERVATIONS_VERSION = "cardcap.hand_observations/1.0"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.write_text(text, encoding="utf-8")
    replace_with_windows_sharing_retry(temporary, path)


def serialize_hand(hand: HandEstimate, width: int, height: int) -> dict:
    hand.validate()
    xy_px = hand.image_landmarks[:, :2] * [width, height]
    wrist_relative = hand.world_landmarks_m - hand.world_landmarks_m[0]
    outside = np.flatnonzero(((hand.image_landmarks[:, :2] < 0) | (hand.image_landmarks[:, :2] > 1)).any(axis=1))
    mano = None
    if any(x is not None for x in (hand.mano_global_orient, hand.mano_hand_pose, hand.mano_shape)):
        mano = {
            "global_orient_axis_angle": hand.mano_global_orient.tolist() if hand.mano_global_orient is not None else None,
            "hand_pose_axis_angle": hand.mano_hand_pose.tolist() if hand.mano_hand_pose is not None else None,
            "shape": hand.mano_shape.tolist() if hand.mano_shape is not None else None,
        }
    return {
        "side": hand.side,
        "handedness_confidence": hand.handedness_confidence,
        "pose_confidence": None,
        "image_landmarks_normalized": hand.image_landmarks.tolist(),
        "image_landmarks_px": xy_px.tolist(),
        "world_landmarks_m": hand.world_landmarks_m.tolist(),
        "wrist_relative_landmarks_m": wrist_relative.tolist(),
        "world_coordinate_system": hand.world_coordinate_system,
        "joint_confidences": hand.joint_confidences.tolist() if hand.joint_confidences is not None else None,
        "mano": mano, "outside_image_joints": outside.tolist(),
        "diagnostics": hand.diagnostics,
    }


def frame_ranges(indices: list[int]) -> list[list[int]]:
    ranges = []
    for index in sorted(set(indices)):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ranges


def summarize_frames(frames: list[dict], view_ids: list[str], low_score: float) -> dict:
    summaries = {}
    for view_id in view_ids:
        records = [(row["frame"], row["views"][view_id]) for row in frames]
        available = [(index, record) for index, record in records if record["available"]]
        valid = [(index, record) for index, record in available if record["hands"]]
        both = [(index, record) for index, record in valid if {hand["side"] for hand in record["hands"]} == {"left", "right"}]
        hands = [hand for _, record in valid for hand in record["hands"]]
        ambiguous = [index for index, record in valid if len({hand["side"] for hand in record["hands"]}) < len(record["hands"])]
        low = [index for index, record in available if len(record["hands"]) < 2
               or any(hand["handedness_confidence"] < low_score or hand["outside_image_joints"] for hand in record["hands"])
               or index in ambiguous]
        blur = [index for index, record in available if record["blur_flag"]]
        scores = [hand["handedness_confidence"] for hand in hands]
        summaries[view_id] = {
            "timeline_frames": len(records), "available_frames": len(available),
            "valid_frames_at_least_one_hand": len(valid),
            "valid_frame_rate": len(valid) / len(available) if available else None,
            "both_distinct_sides_frames": len(both),
            "frames_with_two_observations": sum(len(record["hands"]) == 2 for _, record in available),
            "total_hand_observations": len(hands),
            "mean_handedness_confidence": float(np.mean(scores)) if scores else None,
            "mean_pose_confidence": None,
            "frames_with_mano": sum(any(hand["mano"] is not None for hand in record["hands"]) for _, record in available),
            "side_observations": {side: sum(hand["side"] == side for hand in hands) for side in ("left", "right")},
            "side_unique_frames": {side: sum(any(hand["side"] == side for hand in record["hands"]) for _, record in available) for side in ("left", "right")},
            "review_frame_ranges": frame_ranges(low),
            "ambiguous_handedness_ranges": frame_ranges(ambiguous),
            "blur_flag_ranges": frame_ranges(blur),
            "confidence_note": "Handedness is the label probability, not 3D joint accuracy. Automated review ranges flag missing/ambiguous sides, low label score or off-image landmarks; they do not detect all inaccurate poses. side_observations counts detections, not unique frames.",
            "pa_mpjpe_mm": None, "acceleration_error_mm_per_frame_squared": None,
            "ground_truth_available": False,
        }
    return summaries
