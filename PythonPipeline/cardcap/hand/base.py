"""Provider-neutral observations; unavailable MANO values are never fabricated."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HandEstimate:
    side: str
    handedness_confidence: float
    image_landmarks: np.ndarray
    world_landmarks_m: np.ndarray
    world_coordinate_system: str
    joint_confidences: np.ndarray | None = None
    mano_global_orient: np.ndarray | None = None
    mano_hand_pose: np.ndarray | None = None
    mano_shape: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError(f"Invalid hand side: {self.side}")
        if not np.isfinite(self.handedness_confidence) or not 0 <= self.handedness_confidence <= 1:
            raise ValueError("Handedness confidence must be finite in [0,1]")
        for name in ("image_landmarks", "world_landmarks_m"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (21, 3) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain 21 finite xyz triples")
            setattr(self, name, array)
        if not self.world_coordinate_system:
            raise ValueError("Provider world coordinate convention must be explicit")
        for name, size in (("joint_confidences", 21), ("mano_global_orient", 3),
                           ("mano_hand_pose", 45), ("mano_shape", 10)):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain {size} finite values when available")
                if name == "joint_confidences" and ((array < 0).any() or (array > 1).any()):
                    raise ValueError("Joint confidences must be in [0,1]")
                setattr(self, name, array)


class EstimatorUnavailable(RuntimeError):
    """A backend, model asset, or permitted dependency is unavailable."""


class IHandEstimator(ABC):
    @property
    @abstractmethod
    def backend_info(self) -> dict[str, Any]:
        """Backend/model provenance and output limitations."""

    @abstractmethod
    def estimate(self, image_rgb: np.ndarray, timestamp_ms: int) -> list[HandEstimate]:
        """RGB uint8 input; timestamps must increase per estimator instance."""

    @abstractmethod
    def close(self) -> None:
        """Release native/GPU resources."""

    def __enter__(self) -> IHandEstimator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
