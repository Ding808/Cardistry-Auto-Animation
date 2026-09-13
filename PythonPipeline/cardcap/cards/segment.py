"""Auditable SAM 2.1 video-mask baseline, with explicit prompt provenance.

The baseline propagates frame-zero object prompts. A tracked object is not an
inferred count of physical packets. Split/merge lifecycle inference is not
implemented here; nonempty event plans fail rather than manufacture events.
No SAM 2 imports, downloads, installations, or model execution occur on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Callable

PIPELINE = Path(__file__).resolve().parents[2]
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_URL = "https://github.com/facebookresearch/sam2/tree/" + SAM2_COMMIT
MODEL_SPECS = {
    "tiny": {"filename": "sam2.1_hiera_tiny.pt", "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
             "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"},
    "small": {"filename": "sam2.1_hiera_small.pt", "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
              "sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"},
}
VIDEO_DEPENDENCIES = ("torch", "numpy", "tqdm", "hydra-core", "omegaconf", "iopath", "pillow", "portalocker", "antlr4-python3-runtime")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_status(source: Path) -> dict:
    result = {"path": str(source), "exists": source.is_dir(), "commit": None, "tracked_files_clean": None}
    if not source.is_dir():
        return result
    try:
        result["commit"] = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        result["tracked_files_clean"] = not changes.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        result["inspection_error"] = str(error)
    return result


def inspect_sam2_availability(models_dir: Path | None = None, *, variant: str = "tiny") -> dict:
    if variant not in MODEL_SPECS:
        raise ValueError("Only the reviewed SAM 2.1 tiny/small variants are configured")
    models = Path(models_dir or PIPELINE / "models").resolve()
    spec = MODEL_SPECS[variant]
    source = _source_status(models / "sam2_source")
    checkpoint = models / spec["filename"]
    actual_hash = file_sha256(checkpoint) if checkpoint.is_file() else None
    dependencies = {}
    for name in (*VIDEO_DEPENDENCIES, "torchvision"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    blockers = []
    if sys.version_info < (3, 10): blockers.append("python_below_3_10")
    if source["commit"] != SAM2_COMMIT or source["tracked_files_clean"] is not True:
        blockers.append("missing_or_modified_pinned_source")
    if actual_hash != spec["sha256"]: blockers.append("missing_or_unverified_checkpoint")
    missing = [name for name in VIDEO_DEPENDENCIES if dependencies[name] is None]
    if missing: blockers.append("missing_video_dependencies: " + ", ".join(missing))
    return {"backend": "sam2_video", "variant": variant, "upstream_url": SAM2_URL,
            "source": source, "checkpoint_path": str(checkpoint), "checkpoint_sha256": actual_hash,
            "expected_checkpoint_sha256": spec["sha256"], "config": spec["config"],
            "dependencies": dependencies, "blockers": blockers, "structural_ready": not blockers,
            "availability_is_import_or_inference_validation": False, "inference_attempted": False,
            "license_audit_is_separate_required_installation_evidence": True,
            "runtime_scope": "JPEG video predictor; torchvision image transforms/automatic mask generator and decord MP4 loader are unused",
            "original_sam2_distribution_requires_torchvision": True,
            "cuda_extension_required_for_this_configuration": False,
            "compile_enabled": False, "fill_hole_area": 0,
            "object_count_is_physical_packet_count": False,
            "automatic_split_merge_inference_implemented": False}


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def validate_seed_config(config: dict, *, baseline_only: bool = False) -> dict:
    if config.get("format_version") != "cardcap.sam2_seeds/1.0":
        raise ValueError("Unsupported seed format_version")
    source = config["source"]
    count = _integer(source["frame_count"], "source.frame_count", 1)
    resolution = source["resolution"]
    if len(resolution) != 2: raise ValueError("resolution must be [width,height]")
    width, height = [_integer(value, "resolution", 1) for value in resolution]
    if not math.isfinite(source["fps"]) or source["fps"] <= 0: raise ValueError("fps must be finite and positive")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", source["sha256"]): raise ValueError("source.sha256 must be SHA256")
    if not isinstance(source.get("video_path"), str): raise ValueError("source.video_path is required")
    objects = config["objects"]
    if not objects: raise ValueError("At least one explicitly prompted object is required")
    ids, sam_ids = set(), set()
    for obj in objects:
        object_id = obj["id"]
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", object_id):
            raise ValueError("Object IDs must be safe nonempty filename tokens")
        sam_id = _integer(obj["sam2_id"], "sam2_id", 1)
        if object_id in ids or sam_id in sam_ids: raise ValueError("Object and SAM2 IDs must be unique")
        ids.add(object_id); sam_ids.add(sam_id)
        if baseline_only and (obj.get("start_frame", 0) != 0 or obj.get("end_frame_exclusive", count) != count):
            raise NotImplementedError("The baseline has no object lifecycle engine; all prompted tracks cover the requested timeline")
    prompts = config["prompts"]
    prompted = set()
    prompt_slots = set()
    for prompt in prompts:
        frame = _integer(prompt["frame"], "prompt.frame")
        if frame >= count or prompt["object_id"] not in ids: raise ValueError("Prompt frame/object is invalid")
        if baseline_only and frame != 0:
            raise NotImplementedError("This first-frame baseline does not execute later correction prompts")
        slot = (frame, prompt["object_id"])
        if slot in prompt_slots:
            raise ValueError("Combine same-frame points/box into one prompt; a later clear_old_points call would replace the earlier prompt")
        prompt_slots.add(slot)
        points, labels, box = prompt.get("points_xy_px", []), prompt.get("point_labels", []), prompt.get("box_xyxy_px")
        if not points and box is None: raise ValueError("Prompt must contain points or a box")
        if len(points) != len(labels): raise ValueError("Every point needs a foreground/background label")
        for point, label in zip(points, labels):
            if len(point) != 2 or not all(math.isfinite(v) for v in point): raise ValueError("Invalid prompt point")
            if not (0 <= point[0] < width and 0 <= point[1] < height): raise ValueError("Prompt point is outside original image")
            if label not in (0, 1) or isinstance(label, bool): raise ValueError("Point labels must be 0/background or 1/foreground")
        if box is not None:
            if len(box) != 4 or not all(math.isfinite(v) for v in box): raise ValueError("Invalid box")
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height): raise ValueError("Invalid box extent")
        provenance = prompt.get("provenance", {})
        if not provenance.get("kind") or not provenance.get("description"):
            raise ValueError("Every prompt requires explicit provenance and description")
        if provenance.get("is_ground_truth") is not False:
            raise ValueError("This research seed format requires explicit is_ground_truth=false")
        prompted.add(prompt["object_id"])
    if prompted != ids: raise ValueError("Every object requires a prompt")
    if baseline_only and config.get("events", []):
        raise NotImplementedError("Automatic/manual split-merge event execution is outside this first-mask baseline")
    return config


class SegmentationCancelled(RuntimeError):
    pass


def prepare_jpeg_frames(source: dict, output_dir: Path, *, cancel_check: Callable[[], bool] | None = None,
                        progress_callback: Callable[[dict], None] | None = None) -> dict:
    import cv2
    video = Path(source["video_path"]).expanduser().resolve()
    actual_hash = file_sha256(video)
    if actual_hash.lower() != source["sha256"].lower(): raise ValueError("Source video hash differs from seed provenance")
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError("JPEG directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError("Could not decode source video with audited OpenCV")
    frames = []
    decoder = capture.getBackendName()
    try:
        decoder_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(decoder_fps) or decoder_fps <= 0:
            raise ValueError("Decoder did not provide a finite positive frame rate")
        if not math.isclose(decoder_fps, source["fps"], rel_tol=1.e-6, abs_tol=1.e-4):
            raise ValueError(f"Decoder fps {decoder_fps} differs from declared fps {source['fps']}")
        while True:
            if cancel_check and cancel_check(): raise SegmentationCancelled("Cancelled during frame preparation")
            ok, bgr = capture.read()
            if not ok: break
            height, width = bgr.shape[:2]
            if [width, height] != source["resolution"]: raise ValueError("Decoded image size differs from source declaration")
            frame = len(frames)
            path = output_dir / f"{frame:06d}.jpg"
            encoded_ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encoded_ok: raise OSError("Could not encode JPEG frame")
            path.write_bytes(encoded.tobytes())
            frames.append({"frame": frame, "source_frame": frame, "time_seconds": frame / decoder_fps,
                "jpeg_file": path.name, "jpeg_sha256": file_sha256(path),
                "decoded_bgr_sha256": hashlib.sha256(bgr.tobytes()).hexdigest()})
            if progress_callback: progress_callback({"phase": "prepare_frames", "decoded_frames": len(frames)})
    finally:
        capture.release()
    if len(frames) != source["frame_count"]:
        raise ValueError(f"Decoded {len(frames)} frames, expected {source['frame_count']}")
    return {"source_video": str(video), "source_video_sha256": actual_hash, "decoder": decoder,
            "opencv_version": cv2.__version__, "frame_count": len(frames), "fps": decoder_fps, "declared_fps": source["fps"],
            "resolution": source["resolution"], "frame_directory": str(output_dir.resolve()),
            "image_space": "original_distorted", "derivation": "Original decoded pixels re-encoded as JPEG quality 95; lossy derived model input, original video unchanged",
            "sam2_loader": "Official numeric-stem JPEG loader, RGB conversion and 1024-square resize",
            "frames": frames}


def run_sam2_segmentation(seed_config: Path, output_dir: Path, *, models_dir: Path | None = None,
                          variant: str = "tiny", cancel_file: Path | None = None,
                          progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Execute reviewed real model code; external dependency/license audit precedes use.

    Outputs are incremental and cancellation is cooperative between model calls.
    This function does not install/download dependencies and does not infer events.
    """
    seed_config, output_dir = Path(seed_config).resolve(), Path(output_dir).resolve()
    config = validate_seed_config(json.loads(seed_config.read_text(encoding="utf-8")), baseline_only=True)
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError("Use a fresh segmentation output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"format_version": "cardcap.sam2_segmentation_report/1.0", "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(), "seed_config": str(seed_config),
              "seed_config_sha256": file_sha256(seed_config), "completed_frames": 0,
              "frame_count_requested": config["source"]["frame_count"],
              "prompted_object_count": len(config["objects"]), "physical_packet_count": None,
              "automatic_split_merge_inferred": False, "full_face_verification_performed": False,
              "mask_confidence_is_ground_truth": False, "resume_supported": False}
    started = time.perf_counter()
    predictor = state = torch = None
    frame_records = []

    def cancelled(): return cancel_file is not None and Path(cancel_file).exists()

    def progress(update):
        payload = {"status": report["status"], "completed_frames": report["completed_frames"],
                   "total_frames": config["source"]["frame_count"], "elapsed_seconds": time.perf_counter() - started, **update}
        atomic_json(output_dir / "progress.json", payload)
        if progress_callback: progress_callback(payload)

    try:
        atomic_json(output_dir / "seeds_used.json", config)
        if cancelled(): raise SegmentationCancelled("Cancelled before model loading")
        availability = inspect_sam2_availability(models_dir, variant=variant)
        report["backend"] = availability
        if not availability["structural_ready"]: raise RuntimeError("SAM 2 unavailable: " + "; ".join(availability["blockers"]))
        progress({"phase": "prepare_frames"})
        source_manifest = prepare_jpeg_frames(config["source"], output_dir / "input_jpeg",
                                              cancel_check=cancelled, progress_callback=progress)
        atomic_json(output_dir / "source_manifest.json", source_manifest)
        source_root = Path(availability["source"]["path"])
        sys.path.insert(0, str(source_root))
        import torch as imported_torch
        import numpy as np
        import cv2
        import sam2
        from sam2.build_sam import build_sam2_video_predictor
        torch = imported_torch
        if Path(sam2.__file__).resolve().parent != (source_root / "sam2").resolve():
            raise RuntimeError("A different sam2 package shadowed the reviewed source")
        if not torch.cuda.is_available(): raise RuntimeError("This baseline requires the verified CUDA device")
        torch.cuda.reset_peak_memory_stats()
        report["device"] = {"name": torch.cuda.get_device_name(), "torch": torch.__version__,
                            "cuda_runtime": torch.version.cuda, "free_total_bytes_before_load": list(torch.cuda.mem_get_info())}
        if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 is not supported by this CUDA device")
        # apply_postprocessing=True appends fill_hole_area=8 AFTER extra overrides.
        # Keep its other defaults explicitly and omit the optional CUDA _C kernel.
        overrides = ["++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true", "++model.fill_hole_area=0",
            "++model.compile_image_encoder=false", "++model.non_overlap_masks=false"]
        report["configuration"] = {"variant": variant, "precision": "autocast_bfloat16",
            "offload_video_to_cpu": True, "offload_state_to_cpu": True, "async_loading_frames": False,
            "vos_optimized": False, "apply_postprocessing": False, "hydra_overrides": overrides,
            "mask_threshold_logit": 0.0, "non_overlap_masks": False,
            "postprocessing_difference": "Retain dynamic multimask and binarized conditioning memory; disable optional CUDA connected-component hole filling"}
        progress({"phase": "load_model"})
        report["model_loading_attempted"] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor = build_sam2_video_predictor(availability["config"], availability["checkpoint_path"],
                device="cuda", mode="eval", hydra_overrides_extra=overrides,
                apply_postprocessing=False, vos_optimized=False)
            if cancelled(): raise SegmentationCancelled("Cancelled after model loading")
            report["backend"]["inference_attempted"] = True
            state = predictor.init_state(video_path=str(output_dir / "input_jpeg"), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
            objects = {obj["id"]: obj for obj in config["objects"]}
            by_sam_id = {obj["sam2_id"]: obj for obj in config["objects"]}
            for prompt in config["prompts"]:
                if cancelled(): raise SegmentationCancelled("Cancelled before prompt evaluation")
                predictor.add_new_points_or_box(state, frame_idx=prompt["frame"], obj_id=objects[prompt["object_id"]]["sam2_id"],
                    points=prompt.get("points_xy_px") or None, labels=prompt.get("point_labels") or None,
                    box=prompt.get("box_xyxy_px"), clear_old_points=True, normalize_coords=True)
            # The upstream end index is start+max, inclusive: count-1 yields count.
            generator = predictor.propagate_in_video(state, start_frame_idx=0,
                max_frame_num_to_track=config["source"]["frame_count"] - 1, reverse=False)
            with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as records_file:
                while True:
                    if cancelled(): raise SegmentationCancelled("Cancelled between propagation frames")
                    frame_started = time.perf_counter()
                    try:
                        frame, sam_ids, mask_logits = next(generator)
                    except StopIteration:
                        break
                    torch.cuda.synchronize()
                    inference_ms = (time.perf_counter() - frame_started) * 1000
                    masks = mask_logits.detach().float().cpu().numpy()
                    if frame != report["completed_frames"]: raise RuntimeError("Unexpected propagated frame order")
                    rows = []
                    for index, sam_id in enumerate(sam_ids):
                        obj = by_sam_id[sam_id]
                        logits = masks[index, 0]
                        width, height = config["source"]["resolution"]
                        if logits.shape != (height, width) or not np.isfinite(logits).all():
                            raise RuntimeError("Invalid original-resolution SAM2 logits")
                        visible = np.ascontiguousarray((logits > 0).astype(np.uint8) * 255)
                        relative = Path("masks") / obj["id"] / f"frame_{frame:06d}.png"
                        path = output_dir / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        encoded_ok, encoded = cv2.imencode(".png", visible)
                        if not encoded_ok: raise OSError("Could not encode predicted mask")
                        path.write_bytes(encoded.tobytes())
                        nlabels, _, stats, _ = cv2.connectedComponentsWithStats(visible, connectivity=8)
                        components = [{"area_px": int(item[cv2.CC_STAT_AREA]),
                            "bbox_xyxy_px": [int(item[0]), int(item[1]), int(item[0] + item[2]), int(item[1] + item[3])]} for item in stats[1:]]
                        components.sort(key=lambda item: item["area_px"], reverse=True)
                        # Raw object-presence logit is not in the public yield tuple.
                        # Read this exact pinned-version state field; never invent a
                        # substitute score if it is absent in a future implementation.
                        obj_index = state["obj_id_to_idx"][sam_id]
                        outputs = state["output_dict_per_obj"][obj_index]
                        current = outputs["cond_frame_outputs"].get(frame)
                        if current is None: current = outputs["non_cond_frame_outputs"].get(frame)
                        raw_score = current.get("object_score_logits") if current is not None else None
                        raw_score = None if raw_score is None else float(raw_score.detach().float().cpu().reshape(-1)[0])
                        area = int(np.count_nonzero(visible))
                        rows.append({"object_id": obj["id"], "sam2_object_id": sam_id,
                            "mask_file": relative.as_posix(), "mask_sha256": file_sha256(path),
                            "area_px": area, "empty": area == 0, "component_count": nlabels - 1,
                            "components": components, "mask_logit_min": float(logits.min()), "mask_logit_max": float(logits.max()),
                            "raw_object_presence_logit": raw_score,
                            "raw_score_source": "pinned predictor output_dict_per_obj compact_current_out.object_score_logits",
                            "score_semantics": "Uncalibrated model object-presence logit; not mask IoU, complete-face visibility, corner accuracy, or physical-packet confidence",
                            "mask_quality_score": None, "full_face_verified": False,
                            "representation": "predicted_visible_mask_not_verified_full_card_face",
                            "seed_indices": [i for i, p in enumerate(config["prompts"]) if p["object_id"] == obj["id"]]})
                    record = {"frame": frame, "source_frame": frame, "timestamp_ms": round(frame / source_manifest["fps"] * 1000),
                        "time_seconds": frame / source_manifest["fps"], "source_video_sha256": source_manifest["source_video_sha256"],
                        "source_jpeg_sha256": source_manifest["frames"][frame]["jpeg_sha256"],
                        "inference_ms": inference_ms, "prompted_object_count": len(rows), "physical_packet_count": None,
                        "objects": rows}
                    records_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    records_file.flush()
                    frame_records.append(record)
                    report["completed_frames"] += 1
                    progress({"phase": "propagate", "last_frame": frame, "inference_ms": inference_ms})
            if report["completed_frames"] != config["source"]["frame_count"]:
                raise RuntimeError("SAM2 propagation ended before all requested frames")
        report["status"] = "completed"
        report["backend"]["inference_attempted"] = True
        report["backend"]["actual_inference_completed"] = True
    except SegmentationCancelled as error:
        report.update(status="cancelled", error=str(error))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if torch is not None and torch.cuda.is_available():
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if predictor is not None and state is not None:
            try:
                predictor.reset_state(state)
            except Exception as cleanup_error:
                report["cleanup_error"] = str(cleanup_error)
        state = predictor = None
        report["elapsed_seconds"] = time.perf_counter() - started
        report["empty_mask_samples"] = sum(obj["empty"] for row in frame_records for obj in row["objects"])
        report["multi_component_samples"] = sum(obj["component_count"] > 1 for row in frame_records for obj in row["objects"])
        report["frames_jsonl_sha256"] = file_sha256(output_dir / "frames.jsonl") if (output_dir / "frames.jsonl").is_file() else None
        atomic_json(output_dir / "report.json", report)
        progress({"phase": "finished"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--models-dir", type=Path)
    inspect.add_argument("--variant", choices=MODEL_SPECS, default="tiny")
    inspect.add_argument("--output", type=Path)
    validate = commands.add_parser("validate-seeds")
    validate.add_argument("--seeds", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--seeds", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models-dir", type=Path)
    run.add_argument("--variant", choices=MODEL_SPECS, default="tiny")
    run.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_sam2_availability(args.models_dir, variant=args.variant)
        if args.output: atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["structural_ready"] else 2
    if args.command == "validate-seeds":
        config = validate_seed_config(json.loads(args.seeds.read_text(encoding="utf-8")), baseline_only=True)
        print(json.dumps({"valid_for_first_frame_baseline": True, "objects": len(config["objects"]), "prompts": len(config["prompts"]), "model_inference": False}))
        return 0
    result = run_sam2_segmentation(args.seeds, args.output, models_dir=args.models_dir,
                                   variant=args.variant, cancel_file=args.cancel_file)
    print(json.dumps({key: result.get(key) for key in ("status", "completed_frames", "elapsed_seconds", "error")}, indent=2))
    return 0 if result["status"] == "completed" else (130 if result["status"] == "cancelled" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
