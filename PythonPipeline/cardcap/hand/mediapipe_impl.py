"""MediaPipe Tasks observations, preserving native coordinates and confidence semantics."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from .base import EstimatorUnavailable, HandEstimate, IHandEstimator


class MediaPipeHandEstimator(IHandEstimator):
    """Stateful CPU/video hand tracking; create one instance per video view.

    `side` is MediaPipe's unmodified native handedness category. The legacy
    Hands documentation describes a mirrored/selfie input assumption, while
    the current Tasks guide does not specify a mirror correction. The caller
    must record source mirroring and validate anatomical labels before retargeting.
    This adapter never flips either the input pixels or the returned labels.
    """

    def __init__(
        self,
        model_path: Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._closed = True
        self._landmarker = None
        self._last_timestamp_ms: int | None = None
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.is_file():
            raise EstimatorUnavailable(
                f"MediaPipe model not found: {self._model_path}. Run setup_models.py."
            )
        if isinstance(num_hands, bool) or not isinstance(num_hands, int) or num_hands < 1:
            raise ValueError("num_hands must be a positive integer")
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_presence_confidence", min_presence_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        try:
            import mediapipe as mp
        except (ImportError, OSError) as exc:
            raise EstimatorUnavailable(
                "MediaPipe unavailable. Install requirements-mediapipe.txt into the pipeline venv."
            ) from exc
        self._mp = mp
        self._options = {
            "num_hands": num_hands,
            "min_hand_detection_confidence": float(min_detection_confidence),
            "min_hand_presence_confidence": float(min_presence_confidence),
            "min_tracking_confidence": float(min_tracking_confidence),
        }
        model_bytes = self._model_path.read_bytes()
        self._model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_buffer=model_bytes,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            **self._options,
        )
        try:
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except (RuntimeError, ValueError, OSError) as exc:
            raise EstimatorUnavailable(f"MediaPipe failed to initialize: {exc}") from exc
        self._closed = False

    @property
    def backend_info(self) -> dict[str, Any]:
        return {
            "name": "mediapipe_tasks_hand_landmarker",
            "version": importlib.metadata.version("mediapipe"),
            "model_path": str(self._model_path),
            "model_sha256": self._model_sha256,
            "running_mode": "VIDEO",
            "execution_device": "CPU",
            "world_coordinate_system": "mediapipe_hand_centered_m",
            "world_coordinate_description": (
                "MediaPipe native world landmarks in meters, origin at the hand's geometric center; "
                "independently centered per hand, without camera-global translation or calibrated metric scale."
            ),
            "image_coordinate_description": (
                "x/y normalized by source image width/height; z is wrist-relative depth "
                "in approximately the normalized-x scale, smaller z is closer to camera."
            ),
            "confidence_semantics": "Handedness classification score only; not landmark accuracy or visibility",
            "joint_confidences_available": False,
            "mano_parameters_available": False,
            "global_camera_pose_available": False,
            "handedness_convention": "Native MediaPipe Tasks category; no input mirroring or label swap applied",
            "handedness_caveat": (
                "Legacy MediaPipe Hands documents a mirrored/selfie assumption. Current Tasks guide "
                "does not restate it. Validate source mirroring/anatomical side before retargeting."
            ),
            "options": dict(self._options),
            "documentation": "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python",
        }

    def estimate(self, image_rgb: np.ndarray, timestamp_ms: int) -> list[HandEstimate]:
        if self._closed or self._landmarker is None:
            raise RuntimeError("MediaPipe estimator is closed")
        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be a NumPy uint8 RGB array")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or min(image_rgb.shape[:2]) < 1:
            raise ValueError("image_rgb must have nonempty shape [height,width,3]")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, np.integer)):
            raise ValueError("timestamp_ms must be an integer")
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms < 0 or timestamp_ms > (2**63 - 1) // 1000:
            raise ValueError("timestamp_ms is outside the supported nonnegative range")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("MediaPipe video timestamps must strictly increase per estimator instance")
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        count = len(result.hand_landmarks)
        if len(result.hand_world_landmarks) != count or len(result.handedness) != count:
            raise RuntimeError("MediaPipe returned inconsistent hand result counts")
        estimates = []
        for index in range(count):
            categories = result.handedness[index]
            if not categories:
                raise RuntimeError("MediaPipe returned a hand without a handedness classification")
            category = max(categories, key=lambda item: item.score)
            raw_label = category.category_name
            side = raw_label.lower() if isinstance(raw_label, str) else ""
            if side not in {"left", "right"}:
                raise RuntimeError(f"Unexpected MediaPipe handedness category: {raw_label!r}")
            estimate = HandEstimate(
                side=side,
                handedness_confidence=float(category.score),
                image_landmarks=np.asarray(
                    [[point.x, point.y, point.z] for point in result.hand_landmarks[index]],
                    dtype=np.float64,
                ),
                world_landmarks_m=np.asarray(
                    [[point.x, point.y, point.z] for point in result.hand_world_landmarks[index]],
                    dtype=np.float64,
                ),
                world_coordinate_system="mediapipe_hand_centered_m",
                joint_confidences=None,
                diagnostics={
                    "raw_handedness_category": raw_label,
                    "handedness_label_swapped": False,
                    "confidence_semantics": "handedness_classification_only",
                    "timestamp_ms": timestamp_ms,
                    "native_result_index": index,
                    "world_coordinates_are_camera_global": False,
                },
            )
            estimate.validate()
            estimates.append(estimate)
        return estimates

    def close(self) -> None:
        if self._landmarker is not None and not self._closed:
            self._landmarker.close()
            self._closed = True
