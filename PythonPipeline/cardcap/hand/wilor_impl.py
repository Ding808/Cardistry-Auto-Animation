"""WiLoR reconstruction using MediaPipe crops and reviewed local research assets."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import time
from typing import Any
import weakref

import numpy as np

from .base import EstimatorUnavailable, HandEstimate, IHandEstimator

UPSTREAM_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"
UPSTREAM_URL = f"https://github.com/rolpotamias/WiLoR/tree/{UPSTREAM_COMMIT}"
_MODEL_CACHE = weakref.WeakValueDictionary()


def _resolve_paths(models_dir=None, *, source_dir=None, checkpoint_path=None,
                   config_path=None, mano_path=None, mano_mean_path=None):
    root = Path(models_dir or Path(__file__).resolve().parents[2] / "models").expanduser().resolve()
    source = Path(source_dir or root / "wilor_source").expanduser().resolve()
    default_mean = root / "mano_mean_params.npz"
    if not default_mean.is_file():
        default_mean = source / "mano_data/mano_mean_params.npz"
    right = Path(mano_path or root / "MANO_RIGHT.pkl").expanduser().resolve()
    return root, {
        "source_dir": source,
        "checkpoint": Path(checkpoint_path or root / "wilor_final.ckpt").expanduser().resolve(),
        "config": Path(config_path or root / "model_config.yaml").expanduser().resolve(),
        "mano_right": right, "mano_left": right.with_name("MANO_LEFT.pkl"),
        "mano_mean": Path(mano_mean_path or default_mean).expanduser().resolve(),
        "detector_model": root / "hand_landmarker.task",
    }


def inspect_wilor_availability(models_dir=None, **asset_paths) -> dict[str, Any]:
    root, paths = _resolve_paths(models_dir, **asset_paths)
    assets = {name: {"configured": True, "path": str(path),
              "exists": path.is_dir() if name == "source_dir" else path.is_file()}
              for name, path in paths.items()}
    blockers = []
    missing = [name for name, status in assets.items() if not status["exists"]]
    if missing:
        blockers.append({"code": "missing_assets", "assets": missing})
    dependencies = {}
    for package in ("torch", "numpy", "opencv-python", "mediapipe", "PyYAML", "smplx"):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = None
    if any(value is None for value in dependencies.values()):
        blockers.append({"code": "missing_dependencies", "packages":
                         [key for key, value in dependencies.items() if value is None]})
    try:
        ultralytics = metadata.version("ultralytics")
        blockers.append({"code": "prohibited_dependency_present", "package": "ultralytics"})
    except metadata.PackageNotFoundError:
        ultralytics = None
    return {"backend": "wilor", "name": "wilor_mediapipe_crops", "status": "assets_available" if not blockers else "unavailable",
            "ready": not blockers, "inference_implemented": True, "inference_attempted": False,
            "availability_is_loaded_model_verification": False,
            "models_dir": str(root), "assets": assets, "blockers": blockers,
            "dependencies": dependencies, "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT, "prohibited_ultralytics_installed": ultralytics,
            "license_scope": "Local noncommercial research under separately accepted WiLoR/MANO/SMPL-X terms; no commercial clearance"}


class WiLoRUnavailable(EstimatorUnavailable):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("WiLoR unavailable: " + json.dumps(diagnostics.get("blockers", []), ensure_ascii=False))


def _crop(image, landmarks, side, size, rescale, mean, std, bbox_shape=(192, 256)):
    import cv2
    height, width = image.shape[:2]
    pixels = landmarks[:, :2] * np.array([width, height])
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    center = (minimum + maximum) * 0.5
    scaled_width, scaled_height = (maximum - minimum) * rescale
    if bbox_shape is not None:
        target_width, target_height = bbox_shape
        # Match load_wilor's ViT [192,256] setting and the author's
        # expand_to_aspect_ratio before selecting the square crop size.
        expanded_width = max(scaled_width, scaled_height * target_width / target_height)
        expanded_height = max(scaled_height, scaled_width * target_height / target_width)
        box_size = float(max(expanded_width, expanded_height))
    else:
        box_size = float(max(scaled_width, scaled_height))
    if not np.isfinite(box_size) or box_size < 2:
        raise ValueError("MediaPipe hand box is too small for WiLoR")
    crop_image = image
    factor = box_size / size / 2.0
    antialias_sigma = (factor - 1) / 2 if factor > 1.1 else 0.0
    if antialias_sigma > 0:
        radius = int(4 * antialias_sigma + 0.5)
        crop_image = cv2.GaussianBlur(image.astype(np.float64), (2 * radius + 1,) * 2,
                                     antialias_sigma, borderType=cv2.BORDER_REPLICATE)
    flip = side == "left"
    crop_center = center.copy()
    if flip:
        crop_image = crop_image[:, ::-1]
        crop_center[0] = width - crop_center[0] - 1
    half = box_size * 0.5
    source = np.asarray([crop_center, crop_center + [0, half], crop_center + [half, 0]], dtype=np.float32)
    target = np.asarray([[size / 2, size / 2], [size / 2, size], [size, size / 2]], dtype=np.float32)
    transform = cv2.getAffineTransform(source, target)
    patch = cv2.warpAffine(crop_image, transform, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    normalized = (patch.astype(np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)), {
        "box_xyxy_px": np.concatenate((minimum, maximum)).tolist(),
        "box_center_px": center.tolist(), "box_size_px": box_size,
        "flipped": flip, "inverse_transform": cv2.invertAffineTransform(transform),
        "antialias_sigma": antialias_sigma,
    }


def _compare_mano(state, right_path, left_path):
    from .mano_assets import load_mano_arrays
    right, left = load_mano_arrays(right_path), load_mano_arrays(left_path)
    parents = right["kintree_table"][0].astype(np.int64).copy()
    parents[0] = -1
    expected = {
        "v_template": right["v_template"], "shapedirs": right["shapedirs"][:, :, :10],
        "posedirs": right["posedirs"].reshape(-1, right["posedirs"].shape[-1]).T,
        "J_regressor": right["J_regressor"], "lbs_weights": right["weights"],
        "faces_tensor": right["f"],
        "parents": parents, "hand_components": right["hands_components"][:6],
        "hand_mean": right["hands_mean"],
    }
    comparisons = {}
    for name, array in expected.items():
        tensor = state["mano." + name].cpu().numpy()
        if tensor.shape != array.shape:
            raise ValueError(f"Downloaded MANO_RIGHT shape differs from checkpoint for {name}")
        difference = float(np.max(np.abs(tensor.astype(np.float64) - array.astype(np.float64))))
        comparisons[name] = {"shape": list(tensor.shape), "max_abs_difference": difference,
                             "agrees_at_float32_tolerance": difference <= 1e-6}
        if difference > 1e-6:
            raise ValueError(f"Downloaded MANO_RIGHT differs from checkpoint: {name}, max abs {difference}")
    reflection = np.array([-1, 1, 1])
    left_difference = float(np.max(np.abs(left["v_template"] - right["v_template"] * reflection)))
    left_shape_difference = float(np.max(np.abs(left["shapedirs"] - right["shapedirs"] * reflection[None, :, None])))
    return {"right_checkpoint_geometry": comparisons,
            "left_vs_mirrored_right_template_max_abs_m": left_difference,
            "left_vs_mirrored_right_shape_basis_max_abs": left_shape_difference,
            "same_betas_on_native_left_reproduce_mirrored_right": left_shape_difference <= 1e-6,
            "left_template_used_in_network": False,
            "left_method": "Official WiLoR canonical-right inference; reflected joints and F R F rotation matrices for left crops"}


class WiLoREstimator(IHandEstimator):
    def __init__(self, models_dir=None, *, source_dir=None, checkpoint_path=None,
                 config_path=None, mano_path=None, mano_mean_path=None,
                 num_hands=2, min_detection_confidence=0.5,
                 min_presence_confidence=0.5, min_tracking_confidence=0.5,
                 device="cuda", crop_rescale_factor=2.0,
                 handedness_policy="native", identity_max_gap_ms=1200):
        self._closed = True
        self._detector = None
        self._model = None
        if handedness_policy not in {"native", "temporal"}:
            raise ValueError("handedness_policy must be 'native' or 'temporal'")
        self._handedness_policy = handedness_policy
        self._identity_tracker = None
        if handedness_policy == "temporal":
            if num_hands > 2:
                raise ValueError("Temporal identity supports at most two hands")
            from .identity import TemporalHandIdentityTracker
            self._identity_tracker = TemporalHandIdentityTracker(max_gap_ms=identity_max_gap_ms)
        asset_args = dict(source_dir=source_dir, checkpoint_path=checkpoint_path,
                          config_path=config_path, mano_path=mano_path, mano_mean_path=mano_mean_path)
        self._info = inspect_wilor_availability(models_dir, **asset_args)
        if not self._info["ready"]:
            raise WiLoRUnavailable(self._info)
        if not np.isfinite(crop_rescale_factor) or crop_rescale_factor <= 0:
            raise ValueError("crop_rescale_factor must be finite and positive")
        self._root, self._paths = _resolve_paths(models_dir, **asset_args)
        from ._wilor_runtime import Config, Reconstruction, CHECKPOINT_SHA256, file_sha256, load_source_modules
        import torch
        import yaml
        self._torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise EstimatorUnavailable("WiLoR CUDA requested but CUDA is unavailable")
        self._device = torch.device(device)
        actual_hash = file_sha256(self._paths["checkpoint"])
        if actual_hash != CHECKPOINT_SHA256:
            raise EstimatorUnavailable("WiLoR checkpoint hash differs from the reviewed official checkpoint")
        for asset, expected in (
            ("config", "f69cb52704df88ef29a7cfe03f35a677a92c1ce08169b729ca7f5862f05d4297"),
            ("mano_mean", "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"),
        ):
            if file_sha256(self._paths[asset]) != expected:
                raise EstimatorUnavailable(f"WiLoR {asset} hash differs from reviewed official asset")
        cfg = Config.convert(yaml.safe_load(self._paths["config"].read_text(encoding="utf-8")))
        cfg.MANO["MEAN_PARAMS"] = str(self._paths["mano_mean"])
        if cfg.MODEL.IMAGE_SIZE != 256 or cfg.MODEL.BACKBONE.TYPE != "vit":
            raise EstimatorUnavailable("Only the reviewed WiLoR 256px ViT checkpoint/config is supported")
        cfg.MODEL.setdefault("BBOX_SHAPE", [192, 256])
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        self._cfg = cfg
        self._mean = np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        self._std = np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)
        self._rescale = float(crop_rescale_factor)
        cache_key = tuple((str(self._paths[key]), self._paths[key].stat().st_mtime_ns)
                          for key in ("checkpoint", "source_dir", "config", "mano_right", "mano_left", "mano_mean")) + (str(self._device),)
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            checkpoint = torch.load(self._paths["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint["state_dict"]
            comparison = _compare_mano(state, self._paths["mano_right"], self._paths["mano_left"])
            modules = load_source_modules(self._paths["source_dir"], self._root.parent)
            model = Reconstruction(cfg, modules, state).eval().requires_grad_(False).to(self._device)
            model.asset_comparison = comparison
            _MODEL_CACHE[cache_key] = model
        self._model = model
        from .mediapipe_impl import MediaPipeHandEstimator
        self._detector = MediaPipeHandEstimator(self._paths["detector_model"], num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._closed = False
        self._info.update({
            "status": "loaded", "ready": True, "availability_is_loaded_model_verification": True,
            "model_sha256": actual_hash, "execution_device": str(self._device),
            "precision": "float32", "checkpoint_loading": "weights_only=True; mmap=True; strict inference module keys",
            "loaded_state_keys": dict(model.loaded_keys), "ignored_training_state_keys": model.ignored_training_keys,
            "mano_asset_comparison": model.asset_comparison,
            "mano_right_sha256": file_sha256(self._paths["mano_right"]),
            "mano_left_sha256": file_sha256(self._paths["mano_left"]),
            "detector": self._detector.backend_info,
            "detection_configuration": "Hybrid: MediaPipe landmarks bounding box, WiLoR ViT/RefineNet reconstruction; no Ultralytics",
            "crop_rescale_factor": self._rescale,
            "bbox_shape": list(cfg.MODEL.BBOX_SHAPE),
            "preprocessing_version": 2,
            "crop_defaults_source": "Official load_wilor injects [192,256]; official demo overrides dataset rescale default with 2.0",
            "image_coordinate_description": "x/y exact inverse affine of network crop-camera projection, normalized by original image width/height; z is wrist-relative model depth in meters. This is distinct from the author's estimated full-camera mesh-render projection.",
            "world_coordinate_system": "wilor_wrist_centered_camera_axes_m",
            "world_coordinate_description": "MANO joints in meters, subtract wrist, x right/y down/z away in crop-camera axes; left x reflected. Monocular model scale, no calibrated camera-global translation.",
            "handedness_convention": "Unmodified MediaPipe category controls crop mirroring; source anatomical labels require validation",
            "confidence_semantics": "MediaPipe handedness classification only; WiLoR supplies no joint confidence or accuracy score",
            "joint_confidences_available": False, "mano_parameters_available": True,
            "global_camera_pose_available": False,
            "crop_camera_convention": {"focal_length_px": float(cfg.EXTRA.FOCAL_LENGTH),
                "image_size_px": int(cfg.MODEL.IMAGE_SIZE),
                "scope": "Checkpoint internal crop projection convention; never full-image physical intrinsics"},
            "mano_parameter_convention": "global_orient is root rotation in model camera axes; hand_pose is 15 parent-relative joint rotations. Both converted from matrices with OpenCV Rodrigues to radians; left rotations F R F with F=diag(-1,1,1), betas unchanged. No MANO mean added. Canonical-right learned shape mirrored for left; native MANO_LEFT with these same betas does not reproduce this geometry because its shape basis differs.",
            "preprocessing_difference": "OpenCV GaussianBlur with nearest-border equivalent replaces skimage/scipy antialiasing when large crops require blur; floating-point implementation may differ",
            "handedness_policy": handedness_policy,
            "identity_max_gap_ms": identity_max_gap_ms if handedness_policy == "temporal" else None,
            "identity_score_semantics": "Geometry and temporal association only; not pose confidence or a calibrated probability",
            "identity_ambiguity_policy": "Retain native side for ambiguous geometry matches. A clearly separate second hand may use one-person opposite-hand exclusivity after the first track is established, explicitly marked as an uncertain inference.",
        })
        if handedness_policy == "temporal":
            self._info["handedness_convention"] = (
                "Per-view temporal geometry association may correct the native category before crop/mirror and actual reconstruction. "
                "Ambiguous geometry matches retain native side; separate second-hand bootstrap may use explicitly recorded opposite-hand inference. No missing detections are fabricated."
            )
            self._info["confidence_semantics"] = (
                "handedness_confidence remains the score of the recorded native MediaPipe category, even if temporal identity changes side; "
                "it is not the corrected side probability. Identity association score is separate; neither is pose accuracy."
            )

    @property
    def backend_info(self):
        return dict(self._info)

    def estimate(self, image_rgb, timestamp_ms):
        if self._closed:
            raise RuntimeError("WiLoR estimator is closed")
        import cv2
        detections = self._detector.estimate(image_rgb, timestamp_ms)
        identities = self._identity_tracker.update(detections, timestamp_ms) if self._identity_tracker else [
            {"side": item.side, "native_side": item.side,
             "native_handedness_confidence": item.handedness_confidence,
             "track_id": None, "track_side": None, "corrected": False,
             "correction_suppressed": False, "source": "native_policy",
             "association_score": None, "ambiguous": False, "reacquired": False}
            for item in detections
        ]
        if not detections:
            return []
        height, width = image_rgb.shape[:2]
        estimates = []
        # Per-hand forward bounds activation memory; eval weights can be shared
        # while every video retains a distinct MediaPipe tracker.
        for detected, identity in zip(detections, identities):
            side_for_crop = identity["side"]
            patch, crop = _crop(image_rgb, detected.image_landmarks, side_for_crop,
                                self._cfg.MODEL.IMAGE_SIZE, self._rescale, self._mean, self._std,
                                self._cfg.MODEL.BBOX_SHAPE)
            tensor = self._torch.from_numpy(patch).unsqueeze(0).to(self._device)
            with self._torch.inference_mode():
                output = self._model(tensor)
            self._info["inference_attempted"] = True
            points = output["joints"][0].cpu().numpy().astype(np.float64)
            projected = output["projected"][0].cpu().numpy().astype(np.float64)
            crop_pixels = (projected + 0.5) * self._cfg.MODEL.IMAGE_SIZE
            full_pixels = np.column_stack((crop_pixels, np.ones(21))) @ crop["inverse_transform"].T
            if crop["flipped"]:
                full_pixels[:, 0] = width - full_pixels[:, 0] - 1
                points[:, 0] *= -1
            world = points - points[[0]]
            image_points = np.column_stack((full_pixels / np.array([width, height]), world[:, 2]))
            global_matrix = output["params"]["global_orient"][0].cpu().numpy().reshape(1, 3, 3)
            hand_matrices = output["params"]["hand_pose"][0].cpu().numpy().reshape(15, 3, 3)
            matrices = np.concatenate((global_matrix, hand_matrices)).astype(np.float64)
            if crop["flipped"]:
                reflection = np.diag([-1.0, 1.0, 1.0])
                matrices = reflection @ matrices @ reflection
            axis_angles = np.asarray([cv2.Rodrigues(matrix)[0].reshape(3) for matrix in matrices])
            estimate = HandEstimate(
                side=side_for_crop, handedness_confidence=detected.handedness_confidence,
                image_landmarks=image_points, world_landmarks_m=world,
                world_coordinate_system="wilor_wrist_centered_camera_axes_m", joint_confidences=None,
                mano_global_orient=axis_angles[0], mano_hand_pose=axis_angles[1:].reshape(45),
                mano_shape=output["params"]["betas"][0].cpu().numpy().reshape(10),
                diagnostics={
                    "timestamp_ms": int(timestamp_ms), "detector_backend": "mediapipe_tasks_hand_landmarker",
                    "confidence_semantics": "mediapipe_handedness_classification_only",
                    "box_xyxy_px": crop["box_xyxy_px"], "box_size_px": crop["box_size_px"],
                    "crop_mirrored": crop["flipped"], "left_rotations_reflected_F_R_F": crop["flipped"],
                    "mano_template": "checkpoint_right_canonical_then_left_reflection",
                    "crop_camera_translation_m": output["camera_translation"][0].cpu().numpy().tolist(),
                    "crop_weak_perspective_camera": output["camera"][0].cpu().numpy().tolist(),
                    "crop_focal_length_px": float(self._cfg.EXTRA.FOCAL_LENGTH),
                    "crop_image_size_px": int(self._cfg.MODEL.IMAGE_SIZE),
                    "detector_image_landmarks_px": (detected.image_landmarks[:, :2] * np.array([width, height])).tolist(),
                    "detector_landmark_source": "MediaPipe image landmarks on actual input pixels, before WiLoR crop reconstruction",
                    "camera_translation_is_calibrated_full_frame": False,
                    "world_coordinates_are_camera_global": False,
                    "antialias_sigma": crop["antialias_sigma"],
                    "hand_identity": identity,
                })
            estimate.validate()
            estimates.append(estimate)
        return estimates

    def close(self):
        if self._detector is not None:
            self._detector.close()
        self._detector = None
        self._model = None
        self._closed = True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect/load real hybrid WiLoR reconstruction")
    for flag, dest in (("models-dir", "models_dir"), ("source-dir", "source_dir"),
                       ("checkpoint", "checkpoint_path"), ("config", "config_path"),
                       ("mano", "mano_path"), ("mano-mean", "mano_mean_path")):
        parser.add_argument("--" + flag, dest=dest, type=Path)
    parser.add_argument("--attempt", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args(argv))
    attempt, image_path, output_path = args.pop("attempt"), args.pop("image"), args.pop("output")
    started = time.perf_counter()
    if attempt or image_path:
        estimator = None
        try:
            estimator = WiLoREstimator(**args)
            result = dict(estimator.backend_info, construction_attempted=True)
            if image_path:
                import cv2
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Image could not be read: {image_path}")
                observations = estimator.estimate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)
                result = dict(estimator.backend_info, construction_attempted=True,
                    hand_count=len(observations), observations=[{
                        "side": hand.side, "handedness_confidence": hand.handedness_confidence,
                        "image_landmarks": hand.image_landmarks.tolist(),
                        "world_landmarks_m": hand.world_landmarks_m.tolist(),
                        "mano_global_orient": hand.mano_global_orient.tolist(),
                        "mano_hand_pose": hand.mano_hand_pose.tolist(),
                        "mano_shape": hand.mano_shape.tolist(), "diagnostics": hand.diagnostics,
                    } for hand in observations])
            exit_code = 0
        except Exception as exc:
            result = dict(getattr(exc, "diagnostics", {}), status="unavailable",
                          construction_attempted=True, error_type=type(exc).__name__, error=str(exc))
            exit_code = 3
        finally:
            if estimator is not None:
                estimator.close()
    else:
        result = inspect_wilor_availability(**args)
        exit_code = 0 if result["ready"] else 3
    result["elapsed_seconds"] = time.perf_counter() - started
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
