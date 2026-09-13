"""Offline hand observations; optionally export WiLoR animation data in the same command."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from cardcap import __version__
from cardcap.export import OBSERVATIONS_VERSION, atomic_json, serialize_hand, summarize_frames
from cardcap.hand.base import EstimatorUnavailable
from cardcap.ingest import VideoView, load_views, source_frame_for_time
from cardcap.viz import OverlayWriter, contact_sheet, overlay_frame, verify_video


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_blur(image: np.ndarray, hands: list[dict], threshold: float) -> tuple[float, bool]:
    """Heuristic, not a ground-truth blur/pose probability; fixed analysis scale."""
    height, width = image.shape[:2]
    if hands:
        points = np.concatenate([np.asarray(hand["image_landmarks_px"]) for hand in hands])
        low, high = points.min(axis=0), points.max(axis=0)
        margin = np.maximum((high - low) * 0.1, 8)
        x0, y0 = np.maximum(low - margin, [0, 0]).astype(int)
        x1, y1 = np.minimum(high + margin, [width, height]).astype(int)
        roi = image[y0:y1, x0:x1]
        if roi.size:
            image = roi
    ratio = 256 / max(image.shape[:2])
    gray = cv2.cvtColor(cv2.resize(image, (max(2, round(image.shape[1] * ratio)), max(2, round(image.shape[0] * ratio)))), cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score, score < threshold


def create_estimator(args):
    if args.backend == "wilor":
        from cardcap.hand.wilor_impl import WiLoREstimator
        return WiLoREstimator(
            models_dir=args.models_dir, source_dir=args.wilor_source,
            checkpoint_path=args.wilor_checkpoint, config_path=args.wilor_config,
            mano_path=args.mano, mano_mean_path=args.mano_mean,
            min_detection_confidence=args.detection_threshold,
            min_presence_confidence=args.presence_threshold,
            min_tracking_confidence=args.tracking_threshold,
            handedness_policy=args.handedness_policy,
            identity_max_gap_ms=args.identity_max_gap_ms,
        )
    from cardcap.hand.mediapipe_impl import MediaPipeHandEstimator
    model_path = args.model or args.models_dir / "hand_landmarker.task"
    return MediaPipeHandEstimator(
        model_path=model_path, num_hands=2,
        min_detection_confidence=args.detection_threshold,
        min_presence_confidence=args.presence_threshold,
        min_tracking_confidence=args.tracking_threshold,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--views-config", type=Path)
    result.add_argument("--output", type=Path, required=True, help="New or empty output folder")
    result.add_argument("--backend", choices=["mediapipe", "wilor"], default="mediapipe")
    result.add_argument("--export-animation", action="store_true",
                        help="WiLoR only: export joint-camera/scale-solved capture into output/animation after observations complete")
    result.add_argument("--bone-mapping", type=Path,
                        help="Optional shared mapping for exported research mesh and animation capture")
    result.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    result.add_argument("--model", type=Path, help="Official MediaPipe hand_landmarker.task")
    result.add_argument("--wilor-source", type=Path)
    result.add_argument("--wilor-checkpoint", type=Path)
    result.add_argument("--wilor-config", type=Path)
    result.add_argument("--mano", type=Path)
    result.add_argument("--mano-mean", type=Path)
    result.add_argument("--handedness-policy", choices=["native", "temporal"], default="native",
                        help="WiLoR: associate hand identity before mirroring crops; no missing detections fabricated")
    result.add_argument("--identity-max-gap-ms", type=int, default=1200)
    result.add_argument("--detection-threshold", type=float, default=0.5)
    result.add_argument("--presence-threshold", type=float, default=0.5)
    result.add_argument("--tracking-threshold", type=float, default=0.5)
    result.add_argument("--review-label-threshold", type=float, default=0.8)
    result.add_argument("--blur-threshold", type=float, default=25.0)
    result.add_argument("--allow-blurry", action="store_true", help="Allow flagged footage as a documented debug run")
    result.add_argument("--cancel-file", type=Path, help="Existence requests clean cancellation between frames")
    result.add_argument("--max-frames", type=int, help="Explicit diagnostic subset, not full-video validation")
    return result


def run(args) -> int:
    export_animation = getattr(args, "export_animation", False)
    if export_animation and args.backend != "wilor":
        raise ValueError("--export-animation requires --backend wilor with MANO parameters; MediaPipe-only observations cannot export this animation")
    for field in ("detection_threshold", "presence_threshold", "tracking_threshold", "review_label_threshold"):
        value = getattr(args, field)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{field} must be in (0,1]")
    if not math.isfinite(args.blur_threshold) or args.blur_threshold < 0:
        raise ValueError("blur_threshold must be nonnegative and finite")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    if args.identity_max_gap_ms < 1:
        raise ValueError("identity_max_gap_ms must be positive")
    if args.backend != "wilor" and args.handedness_policy != "native":
        raise ValueError("Temporal hand identity is currently available for WiLoR crop reconstruction")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status_path = output / "progress.json"
    progress = {"stage": "initializing", "status": "running", "frames_completed": 0, "fraction": 0.0}
    atomic_json(status_path, progress)
    frames = []
    metadata = {}
    error = None
    writers = {}
    try:
        specs, primary_id = load_views(args.video, args.views_config)
        with ExitStack() as stack:
            views = {}
            estimators = {}
            for spec in specs:
                view = VideoView(spec)
                stack.callback(view.close)
                views[spec.id] = view
                estimator = create_estimator(args)
                stack.callback(estimator.close)
                estimators[spec.id] = estimator
            primary = views[primary_id]
            total = min(primary.frame_count, args.max_frames or primary.frame_count)
            metadata = {
                "primary_view": primary_id, "fps": primary.fps,
                "frame_count_requested": total,
                "full_source_video_requested": total == primary.frame_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": f"CardistryCapture {__version__}",
                "backend": args.backend,
                "views": [dict(view.metadata, source_sha256=sha256_file(view.spec.video),
                               backend_info=estimators[view_id].backend_info)
                          for view_id, view in views.items()],
                "settings": {
                    "detection_threshold": args.detection_threshold,
                    "presence_threshold": args.presence_threshold,
                    "tracking_threshold": args.tracking_threshold,
                    "review_label_threshold": args.review_label_threshold,
                    "blur_threshold": args.blur_threshold,
                    "allow_blurry": args.allow_blurry,
                    "handedness_policy": args.handedness_policy,
                    "identity_max_gap_ms": args.identity_max_gap_ms,
                },
                "output_scope": ("Hand observations plus requested joint-camera/scale-solved animation exchange; UE baking is separate"
                                 if export_animation else "Internal hand observations, not .cardcap.json UE exchange or baked animation"),
                "synchronization": "Primary video frame timeline; source_time = primary_time + view_offset - primary_offset. Offsets are user supplied, not estimated.",
                "multiview_fusion_performed": False,
                "warnings": [],
            }
            if primary.fps < 120:
                metadata["warnings"].append("Input below 120 fps shooting recommendation; fast motion may be temporally unresolved.")
            if primary.height < 1080:
                metadata["warnings"].append("Input below 1080p shooting recommendation.")
            if not all(view.camera["calibrated"] for view in views.values()):
                metadata["warnings"].append("Missing camera calibration; metric/global pose and absolute accuracy are unverified.")
            atomic_json(output / "metadata.json", metadata)
            # Filename ordinals prevent view ids from becoming filesystem paths.
            for ordinal, (view_id, view) in enumerate(views.items()):
                writer = OverlayWriter(output / f"overlay_{ordinal:02d}.mp4", primary.fps, view.width, view.height)
                stack.callback(writer.close)
                writers[view_id] = writer
            cached = {}
            previous_ms = -1
            with (output / "frames.checkpoint.jsonl").open("w", encoding="utf-8") as checkpoint:
                for index in range(total):
                    if args.cancel_file and args.cancel_file.exists():
                        progress.update(status="cancelled", stage="cancelled")
                        break
                    timestamp = round(index / primary.fps * 1000)
                    if timestamp <= previous_ms:
                        raise ValueError("Frame rate cannot be represented by strictly increasing millisecond timestamps")
                    previous_ms = timestamp
                    row = {"frame": index, "timestamp_ms": timestamp, "time_seconds": index / primary.fps, "views": {}}
                    for view_id, view in views.items():
                        source_index = source_frame_for_time(index / primary.fps, view.fps, view.spec.offset_seconds - primary.spec.offset_seconds)
                        image = view.read(source_index)
                        if image is None:
                            record = {"available": False, "source_frame": source_index, "reason": "outside_view_timeline", "hands": []}
                            missing = np.zeros((view.height, view.width, 3), np.uint8)
                            cv2.putText(missing, "VIEW OUTSIDE TIMELINE - no observations", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                            writers[view_id].write(missing)
                        else:
                            inference_start = time.perf_counter()
                            cache_hit = view_id in cached and cached[view_id][0] == source_index
                            if cache_hit:
                                hands = cached[view_id][1]
                            else:
                                predictions = estimators[view_id].estimate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), timestamp)
                                hands = [serialize_hand(prediction, view.width, view.height) for prediction in predictions]
                                cached[view_id] = (source_index, hands)
                            latency = (time.perf_counter() - inference_start) * 1000
                            blur_score, blur_flag = measure_blur(image, hands, args.blur_threshold)
                            record = {
                                "backend": args.backend,
                                "available": True, "source_frame": source_index, "source_time_seconds": source_index / view.fps,
                                "reused_source_frame": cache_hit, "inference_ms": latency,
                                "blur_variance_256": blur_score, "blur_flag": blur_flag, "hands": hands,
                            }
                            writers[view_id].write(overlay_frame(image, record, index, primary.fps))
                        row["views"][view_id] = record
                    frames.append(row)
                    checkpoint.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    checkpoint.flush()
                    progress.update(stage="hand_estimation", frames_completed=len(frames), frames_total=total,
                                    fraction=len(frames) / total, elapsed_seconds=time.perf_counter() - start)
                    atomic_json(status_path, progress)
                    if index == 0 or (index + 1) % 20 == 0 or index + 1 == total:
                        print(json.dumps(progress), flush=True)
            if progress["status"] != "cancelled":
                progress.update(status="completed", stage="validating")
            for view_metadata in metadata["views"]:
                view_metadata["backend_info"] = estimators[view_metadata["id"]].backend_info
            atomic_json(output / "metadata.json", metadata)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        progress.update(status="failed", stage="failed", error=error)
        if hasattr(exc, "diagnostics"):
            atomic_json(output / "backend-unavailable.json", exc.diagnostics)
    finally:
        progress["elapsed_seconds"] = time.perf_counter() - start

    view_ids = [view["id"] for view in metadata.get("views", [])]
    summaries = summarize_frames(frames, view_ids, args.review_label_threshold)
    observations = {"format_version": OBSERVATIONS_VERSION, "meta": metadata, "frames": frames, "summary": summaries}
    atomic_json(output / "hand_observations.json", observations)
    overlays = {}
    animation_export = None
    if progress["status"] == "completed":
        try:
            for view_id, writer in writers.items():
                overlays[view_id] = verify_video(writer.path, len(frames), metadata["fps"])
                indices = sorted(set(round(x) for x in np.linspace(0, len(frames) - 1, min(12, len(frames)))))
                contact_sheet(writer.path, output / (writer.path.stem + "_review.jpg"), indices)
            primary_summary = summaries[metadata["primary_view"]]
            if primary_summary["valid_frames_at_least_one_hand"] == 0:
                raise RuntimeError("No hand detected in the real video; observations and overlay retained, reconstruction not successful")
            flags = sum(row["views"][metadata["primary_view"]]["blur_flag"] for row in frames)
            if flags / len(frames) > 0.8 and not args.allow_blurry:
                raise RuntimeError("Over 80% of primary frames flagged by blur heuristic; reshoot or explicitly use --allow-blurry for diagnostic processing")
            if export_animation:
                progress.update(status="running", stage="animation_export")
                atomic_json(status_path, progress)
                from cardcap.prepare_animation import export_cardcap
                mapping = getattr(args, "bone_mapping", None)
                export_options = {"bone_mapping_path": mapping} if mapping is not None else {}
                animation_export = export_cardcap(output / "hand_observations.json", output / "animation", **export_options)
            progress.update(status="completed", stage="complete")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            progress.update(status="failed", stage="failed", error=error)
    report = {
        "status": progress["status"], "error": error, "backend": args.backend,
        "elapsed_seconds": time.perf_counter() - start, "summary": summaries, "overlays": overlays,
        "observation_file": str(output / "hand_observations.json"),
        "animation_export": animation_export,
        "visual_review_completed": False,
        "visual_review_note": "Overlay/contact sheets require visual inspection; numerical scores alone do not establish visual correctness.",
        "milestone_dod": {
            "real_video_at_least_3_seconds": len(frames) / metadata["fps"] >= 3 if metadata else False,
            "json_exported": True,
            "overlay_fully_decoded": bool(overlays) and len(overlays) == len(view_ids),
            "wilor_vs_mediapipe_real_comparison": False,
            "mano_parameters_recovered": any(hand["mano"] is not None for row in frames for record in row["views"].values() for hand in record["hands"]),
        },
    }
    atomic_json(output / "report.json", report)
    progress["elapsed_seconds"] = report["elapsed_seconds"]
    atomic_json(status_path, progress)
    print(json.dumps({"status": progress["status"], "output": str(output), "elapsed_seconds": report["elapsed_seconds"], "error": error}), flush=True)
    return 0 if progress["status"] == "completed" else 2 if progress["status"] == "cancelled" else 1


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except (ValueError, OSError, EstimatorUnavailable) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
