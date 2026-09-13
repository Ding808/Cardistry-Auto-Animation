"""S1: synchronized multi-view decoding with explicit calibration provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import warnings as python_warnings

import cv2
import numpy as np

from .video_metadata import read_video_metadata


@dataclass(frozen=True)
class ViewSpec:
    id: str
    video: Path
    offset_seconds: float = 0.0
    calibration: Path | None = None


def load_views(video: Path | None, manifest: Path | None) -> tuple[list[ViewSpec], str]:
    if (video is None) == (manifest is None):
        raise ValueError("Provide exactly one of --video or --views-config")
    if video is not None:
        return [ViewSpec("primary", video.resolve())], "primary"
    manifest = manifest.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) != {"primary_view", "views"}:
        raise ValueError("Views config requires exactly primary_view and views")
    if not isinstance(data["views"], list) or not data["views"]:
        raise ValueError("views must be a nonempty list")
    views = []
    for index, item in enumerate(data["views"]):
        if not isinstance(item, dict) or not {"id", "video"} <= item.keys():
            raise ValueError(f"views[{index}] requires id and video")
        if set(item) - {"id", "video", "offset_seconds", "calibration"}:
            raise ValueError(f"Unknown fields in views[{index}]")
        if not isinstance(item["id"], str) or not item["id"] or not isinstance(item["video"], str):
            raise ValueError("View id and video path must be nonempty strings")
        offset = item.get("offset_seconds", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise ValueError("offset_seconds must be finite")
        path = Path(item["video"])
        calib = Path(item["calibration"]) if item.get("calibration") else None
        views.append(ViewSpec(
            item["id"], (manifest.parent / path).resolve() if not path.is_absolute() else path,
            float(offset), ((manifest.parent / calib).resolve() if calib and not calib.is_absolute() else calib),
        ))
    ids = [view.id for view in views]
    if len(ids) != len(set(ids)) or data["primary_view"] not in ids:
        raise ValueError("View ids must be unique and primary_view must identify one")
    return views, data["primary_view"]


def source_frame_for_time(time_seconds: float, fps: float, offset_seconds: float) -> int:
    """Video time = capture timeline time + supplied synchronization offset."""
    return math.floor((time_seconds + offset_seconds) * fps + 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_camera(path: Path, resolution: tuple[int, int]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not {"resolution", "intrinsics", "distortion"} <= data.keys():
        raise ValueError("Calibration requires resolution, intrinsics and distortion")
    if set(data) - {"resolution", "intrinsics", "distortion", "world_to_camera"}:
        raise ValueError("Unknown calibration fields")
    if data["resolution"] != list(resolution):
        raise ValueError("Calibration resolution must match video resolution")
    intrinsics = data["intrinsics"]
    if not isinstance(intrinsics, dict) or set(intrinsics) != {"fx", "fy", "cx", "cy"}:
        raise ValueError("Calibration intrinsics requires fx, fy, cx, cy")
    try:
        values = np.array([intrinsics[k] for k in ("fx", "fy", "cx", "cy")], dtype=float)
        distortion = np.asarray(data["distortion"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("All camera intrinsics must be finite")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if distortion.ndim != 1 or distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise ValueError("Invalid OpenCV distortion vector")
    transform = data.get("world_to_camera")
    if transform is not None:
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("world_to_camera must be finite 4x4 matrix with meter translation")
        rotation = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("world_to_camera must be a rigid right-handed transform")
    return {
        "calibrated": True, "intrinsics": dict(zip(("fx", "fy", "cx", "cy"), values.tolist())),
        "distortion": distortion.tolist(),
        "world_to_camera": transform.tolist() if transform is not None else None,
        "image_space": "undistorted_same_intrinsics", "calibration_source": str(path),
        "intrinsics_source": "explicit_calibration", "prior_based": False,
        "provenance": {"path": str(path), "sha256": _sha256(path),
                       "scope": "User-supplied calibration; independent physical accuracy not verified here"},
    }


def _detect_unit_grid(gray: np.ndarray) -> tuple[tuple[int, int], np.ndarray] | None:
    # CALIB_CB_LARGER reports the recovered grid dimensions in meta. No assumed
    # printed grid count or physical square size is needed for a K/D estimate.
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_LARGER | cv2.CALIB_CB_ACCURACY
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(gray, (3, 3), flags)
    if not found or corners is None or meta is None or meta.ndim != 2:
        return None
    rows, columns = meta.shape
    if min(rows, columns) < 3 or corners.size != rows * columns * 2 or not np.isfinite(corners).all():
        return None
    grid = corners.reshape(rows, columns, 2)
    if rows > columns:
        grid = grid.transpose(1, 0, 2)
        rows, columns = columns, rows
    return (columns, rows), np.ascontiguousarray(grid, dtype=np.float32)


def _fit_unit_grids(rows: list[dict], pattern: tuple[int, int], resolution: tuple[int, int]) -> dict:
    """Unit-grid calibration; extrinsics are temporary grid units, never metres."""
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 distinct grid views; got {len(rows)}")
    columns, height = pattern
    objects = np.zeros((columns * height, 3), np.float32)
    objects[:, :2] = np.mgrid[:columns, :height].T.reshape(-1, 2)
    pixels = [row["grid"].reshape(-1, 2) for row in rows]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        [objects.copy() for _ in rows], pixels, resolution, None, None,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9))
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Nonfinite chessboard calibration solution")
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rotations]
    normal_span = max(math.degrees(math.acos(float(np.clip(abs(a @ b), 0, 1))))
                      for i, a in enumerate(normals) for b in normals[i + 1:])
    centers = [p.mean(axis=0) for p in pixels]
    center_span = max(float(np.linalg.norm(a - b)) for a in centers for b in centers) / math.hypot(*resolution)
    if rms > 2.0 or normal_span < 15.0 or center_span < .10:
        raise ValueError(f"Chessboard quality gates failed: RMS={rms:.3g}px, normal span={normal_span:.3g}deg, center span={center_span:.3g}")
    if not (0 <= matrix[0, 2] < resolution[0] and 0 <= matrix[1, 2] < resolution[1]):
        raise ValueError("Chessboard principal point is outside image")
    if any(np.min((objects @ cv2.Rodrigues(r)[0].T + t.reshape(3))[:, 2]) <= 0
           for r, t in zip(rotations, translations)):
        raise ValueError("Chessboard solution places a grid behind camera")
    return {"calibrated": True,
        "intrinsics": {"fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
                       "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2])},
        "distortion": distortion.reshape(-1).tolist(), "world_to_camera": None,
        "image_space": "undistorted_same_intrinsics", "intrinsics_source": "adjacent_chessboard", "prior_based": False,
        "provenance": {"grid_inner_corners": list(pattern), "grid_spacing": "1 arbitrary grid unit; physical square size unknown",
                       "selected_frames": [{k: v for k, v in row.items() if k != "grid"} for row in rows],
                       "training_rms_px": float(rms), "normal_span_degrees": normal_span,
                       "center_span_image_diagonal": center_span, "independent_accuracy_verified": False,
                       "gates": {"min_views": 10, "min_normal_span_degrees": 15.,
                                 "min_center_span_image_diagonal": .10, "max_training_rms_px": 2.},
                       "extrinsics_exported": False}}


def _adjacent_chessboard(video_path: Path, resolution: tuple[int, int], warnings: list[str]) -> dict | None:
    # Never search outside the source video's immediate directory. Each clip is
    # fitted independently; do not pool potentially different lenses/settings.
    candidates = sorted((p for p in video_path.parent.glob("calib_*.mp4") if p.is_file() and p.resolve() != video_path), key=lambda p: p.name.lower())
    if not candidates:
        warnings.append("No adjacent calib_*.mp4 video; automatic chessboard calibration unavailable.")
    for path in candidates:
        capture = cv2.VideoCapture(str(path))
        groups: dict[tuple[int, int], list[dict]] = {}
        sampled = []
        try:
            if not capture.isOpened():
                raise ValueError("cannot open calibration video")
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if size != resolution or count <= 0:
                raise ValueError("calibration video resolution differs or frame count is invalid")
            for index in np.unique(np.linspace(0, count - 1, min(60, count), dtype=int)):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    continue
                ok, image = capture.read()
                if not ok or image.shape[:2] != (resolution[1], resolution[0]):
                    continue
                sampled.append(int(index))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected = _detect_unit_grid(gray)
                if detected is None:
                    continue
                pattern, grid = detected
                group = groups.setdefault(pattern, [])
                # Grid origins/orientation are not physical world identities.
                # Compare every flip so duplicate images cannot count as views.
                variants = (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1])
                if any(min(float(np.sqrt(np.mean(np.sum((g - p["grid"]) ** 2, axis=2))))
                           for g in variants) / math.hypot(*resolution) < .01 for p in group):
                    continue
                group.append({"frame_index": int(index), "decoded_gray_sha256": hashlib.sha256(gray.tobytes()).hexdigest(), "grid": grid})
            if not groups:
                raise ValueError("no usable chessboard grid detected")
            # Select the most supported grid count before fitting, not whichever
            # shape yields the lowest residual after trying many camera fits.
            pattern = max(groups, key=lambda p: (len(groups[p]), p[0] * p[1]))
            camera = _fit_unit_grids(groups[pattern], pattern, resolution)
            camera["calibration_source"] = str(path)
            camera["provenance"].update(path=str(path), sha256=_sha256(path), sampled_frames=sampled,
                                        detected_grid_groups={f"{p[0]}x{p[1]}": len(g) for p, g in groups.items()},
                                        selection="most distinct observations, then largest grid; first acceptable adjacent clip by filename")
            warnings.append("Adjacent chessboard K/D assumes the same lens, zoom, focus and pixel geometry as the source video; this relationship and physical scale are not independently verified.")
            return camera
        except (ValueError, cv2.error, OSError) as exc:
            warnings.append(f"Automatic chessboard calibration skipped {path.name}: {exc}")
        finally:
            capture.release()
    return None


def resolve_intrinsics(video_path: Path, calib_path: Path | None = None, *, resolution=None) -> dict:
    """Explicit JSON -> adjacent chessboard -> raw container metadata -> unknown.

    Metadata tags do not by themselves establish a complete applicable K/D.
    Unknown intrinsics and distortion stay None; original pixels are preserved.
    Accepted calibration has no width-based focal prior or range gate. Unequal
    calibrated focal lengths retain the established pixel rectification for UE;
    source K/D and the exact display pixel mapping are recorded separately.
    """
    video_path = Path(video_path).resolve()
    if resolution is None:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open video: {video_path}")
            resolution = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            capture.release()
    if len(resolution) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 2 for x in resolution):
        raise ValueError("Decoded resolution must contain two integer dimensions >= 2")
    resolution = tuple(int(x) for x in resolution)
    warnings: list[str] = []
    camera = _explicit_camera(Path(calib_path).resolve(), resolution) if calib_path is not None else _adjacent_chessboard(video_path, resolution, warnings)
    if camera is None:
        metadata = read_video_metadata(video_path)
        status = metadata["status"]
        if status == "unreadable":
            reason = "Container metadata could not be read; this does not establish that camera fields are absent."
        elif status == "fields_present_insufficient":
            reason = "Camera-related metadata fields are present but do not establish complete K, distortion or applicability to the decoded video."
        else:
            reason = "No recognized camera fields in the parsed container metadata scopes; unparsed vendor/codec/media regions may still contain camera information."
        warnings.append(reason)
        warnings.append("Camera intrinsics and distortion are unknown; source pixels are unchanged and no camera or metric-scale measurement is claimed.")
        camera = {"calibrated": False, "intrinsics": None, "distortion": None,
                  "world_to_camera": None, "image_space": "original_distorted",
                  "intrinsics_source": "unobservable", "prior_based": False,
                  "metadata_report": metadata,
                  "provenance": {"source": "container_metadata_reader",
                                 "metadata_status": status,
                                 "source_video_sha256": metadata["source"]["sha256"],
                                 "metadata_reader_sha256": metadata["reader"]["sha256"],
                                 "reason": reason, "independent_accuracy_verified": False}}
    if camera["calibrated"] and camera["intrinsics"]["fx"] != camera["intrinsics"]["fy"]:
        source_intrinsics = dict(camera["intrinsics"])
        source_distortion = list(camera["distortion"])
        focal = math.sqrt(source_intrinsics["fx"] * source_intrinsics["fy"])
        effective = dict(source_intrinsics, fx=focal, fy=focal)
        camera.update(source_intrinsics=source_intrinsics, source_lens_distortion=source_distortion,
                      intrinsics=effective, distortion=[0.] * len(source_distortion),
                      image_space="undistorted_rectified_intrinsics",
                      effective_intrinsics_source="geometric_mean_rectification_of_calibrated_source")
        camera["provenance"]["pixel_remapping"] = {
            "method": "cv2.undistort(original_pixels, source_K, source_D, newCameraMatrix=effective_K)",
            "source_intrinsics": source_intrinsics, "source_lens_distortion": source_distortion,
            "effective_intrinsics": effective,
            "rule": "f = sqrt(source_fx * source_fy); original cx/cy and output dimensions preserved",
            "output_resolution": list(resolution), "source_image_space": "original_distorted",
            "output_image_space": "undistorted_rectified_intrinsics", "pixel_resampling_required": True,
            "camera_axes_or_geometry_changed": False,
            "border_behavior": "OpenCV undistort constant-zero border; invalid edge pixels are not new image observations"}
        warnings.append("Calibrated unequal focal lengths are rectified by resampling pixels to f=sqrt(fx*fy) for the UE camera; source K/D are preserved and output pixels are not the original image.")
    camera.update(warnings=warnings, resolution=list(resolution))
    for warning in warnings:
        python_warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return camera


class VideoView:
    def __init__(self, spec: ViewSpec):
        self.spec = spec
        if not spec.video.is_file():
            raise FileNotFoundError(f"Video does not exist: {spec.video}")
        self.capture = cv2.VideoCapture(str(spec.video))
        if not self.capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {spec.video}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (0 < self.fps <= 1000) or self.frame_count <= 0 or min(self.width, self.height) < 2:
            self.close()
            raise ValueError(f"Unsupported video metadata: {spec.video}")
        self._last_index = -1
        self._last_image = None
        try:
            self.camera = self._load_camera()
        except Exception:
            self.close()
            raise

    def _load_camera(self) -> dict:
        return resolve_intrinsics(self.spec.video, self.spec.calibration, resolution=(self.width, self.height))

    @property
    def metadata(self) -> dict:
        return {
            "id": self.spec.id, "source_video": str(self.spec.video),
            "fps": self.fps, "frame_count_metadata": self.frame_count,
            "resolution": [self.width, self.height],
            "duration_seconds_metadata": self.frame_count / self.fps,
            "offset_seconds": self.spec.offset_seconds,
            "camera": self.camera, "decoder": self.capture.getBackendName(),
        }

    def read(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self.frame_count:
            return None
        if index == self._last_index:
            return self._last_image.copy()
        if index != self._last_index + 1:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"Video seek failed: {self.spec.id} frame {index}")
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"Decode failed before metadata end: {self.spec.id} frame {index}")
        if self.camera["image_space"] == "undistorted_same_intrinsics":
            c = self.camera["intrinsics"]
            matrix = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
            image = cv2.undistort(image, matrix, np.array(self.camera["distortion"]), None, matrix)
        elif self.camera["image_space"] == "undistorted_rectified_intrinsics":
            source = self.camera["source_intrinsics"]
            effective = self.camera["intrinsics"]
            source_matrix = np.array([[source["fx"], 0, source["cx"]], [0, source["fy"], source["cy"]], [0, 0, 1.]])
            effective_matrix = np.array([[effective["fx"], 0, effective["cx"]], [0, effective["fy"], effective["cy"]], [0, 0, 1.]])
            image = cv2.undistort(image, source_matrix, np.array(self.camera["source_lens_distortion"]), None, effective_matrix)
        self._last_index, self._last_image = index, image
        return image.copy()

    def close(self) -> None:
        self.capture.release()
