"""Bundle existing card surface observations without promoting physical claims.

Standalone draft format, not cardcap 1.0 or an Unreal import asset. No model
inference, geometry fitting, missing-pose filling, or scale application occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

VERSION = "cardcap.card_draft_observations/1.0"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def read_jsonl(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def write_json(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _bound(path, digest, label):
    path = Path(path).resolve()
    if not isinstance(digest, str) or sha(path) != digest: raise ValueError(f"{label}: source hash mismatch")
    return path


def _inside(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("Referenced relative path escapes its source run")
    return path


def _objects(items):
    result = {item["object_id"]: item for item in items}
    if len(result) != len(items): raise ValueError("Duplicate surface object ID in a source frame")
    return result


def _validate_scale_estimate(estimate):
    """Accept legacy evidence or an explicit unknown estimate, never apply it."""
    if not isinstance(estimate, dict):
        raise ValueError("Scale estimate must be an object")
    metric = estimate["meters_per_source_unit"]
    nominal = estimate["nominal_meters_per_source_unit"]
    factor = estimate["global_scale_factor"]
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    if metric is None:
        if (estimate["anchor_method"] is not None or estimate["scale_confidence"] != "unknown"
                or estimate.get("provenance", {}).get("kind") != "unobservable"
                or type(factor) not in (int, float) or factor != 1
                or estimate.get("numeric_confidence") is not None):
            raise ValueError("Unknown scale requires null anchor/confidence, unobservable provenance and an identity factor")
        if nominal is not None and not positive(nominal):
            raise ValueError("Scale estimate contains invalid nominal unit normalization")
    elif not positive(metric) or not positive(factor) or (nominal is not None and not positive(nominal)):
        raise ValueError("Scale estimate contains invalid positive factor")
    if estimate["anchor_method"] == "hand_prior" and estimate["scale_confidence"] != "low":
        raise ValueError("Legacy hand prior must remain low confidence")


def validate_sources(geometry_run, visibility_run, scale_estimate, *, visualization_video=None, visualization_report=None):
    """Verify actual source bytes and all frame/object bindings before export."""
    import cv2
    import numpy as np
    geometry_run, visibility_run = Path(geometry_run).resolve(), Path(visibility_run).resolve()
    scale_path = Path(scale_estimate).resolve()
    files = {"geometry_report": geometry_run / "report.json", "visibility_report": visibility_run / "report.json", "scale_estimate": scale_path}
    geometry, visibility, scale = (read_json(files[key]) for key in ("geometry_report", "visibility_report", "scale_estimate"))
    if any(item.get("status") != "completed" for item in (geometry, visibility, scale)): raise ValueError("Only completed input reports can be bundled")
    segmentation_run = Path(geometry["segmentation_run"]).resolve()
    if Path(visibility["source_run"]).resolve() != segmentation_run: raise ValueError("Visibility and geometry refer to different segmentation runs")
    files["geometry_frames"] = _bound(_inside(geometry_run, geometry["geometry_frames_file"]), geometry["geometry_frames_sha256"], "geometry records")
    files["visibility_frames"] = _bound(visibility_run / "visibility_frames.jsonl", visibility["visibility_frames_sha256"], "visibility records")
    for key, name in (("segmentation_report", "report.json"), ("source_manifest", "source_manifest.json"), ("segmentation_frames", "frames.jsonl")):
        files[key] = _bound(segmentation_run / name, geometry["input_hashes"][name], name)
    _bound(files["segmentation_report"], visibility["source_report_sha256"], "visibility source report")
    _bound(files["segmentation_frames"], visibility["source_frames_sha256"], "visibility source frames")
    segmentation, source = read_json(files["segmentation_report"]), read_json(files["source_manifest"])
    if segmentation.get("status") != "completed": raise ValueError("Segmentation is not complete")
    _bound(files["segmentation_frames"], segmentation["frames_jsonl_sha256"], "segmentation output inventory")
    source_hash, count = source["source_video_sha256"], source["frame_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0: raise ValueError("Positive source frame count required")
    if not math.isfinite(source["fps"]) or source["fps"] <= 0: raise ValueError("Positive source FPS required")
    if any(item.get("source_video_sha256") != source_hash for item in (geometry, visibility, scale)):
        raise ValueError("Geometry, visibility and scale must bind to the same source video SHA256")
    if any(item.get("frame_count", item.get("completed_frames")) != count for item in (geometry, visibility, segmentation)):
        raise ValueError("Source frame counts differ")
    if geometry["resolution"] != source["resolution"] or geometry["fps"] != source["fps"]:
        raise ValueError("Geometry and segmentation pixel/time spaces differ")
    source_video = _bound(source["source_video"], source_hash, "original video")
    camera = geometry["camera_provenance"]
    files["camera_source"] = _bound(camera["path"], camera["sha256"], "geometry camera")
    _bound(files["camera_source"], geometry["input_hashes"]["camera_json"], "camera report binding")
    files["seed_configuration"] = _bound(segmentation["seed_config"], segmentation["seed_config_sha256"], "segmentation seeds")
    seeds = read_json(files["seed_configuration"])
    if seeds["source_video_sha256"] != source_hash: raise ValueError("Seed configuration belongs to another video")
    seed_by_id = _objects(seeds["objects"])
    for seed in seeds["objects"]:
        _bound(_inside(files["seed_configuration"].parent, seed["mask_file"]), seed["mask_sha256"], "seed mask")
        if seed.get("candidate_file"):
            _bound(seed["candidate_file"], seed["candidate_file_sha256"], "seed candidate provenance")
    estimate = scale["scale_estimate"]
    _validate_scale_estimate(estimate)
    # Scale report retains historical executed-code hashes; do not demand that
    # today's source code still equals those earlier execution versions.
    if scale.get("inputs_sha256", {}).get(str(files["camera_source"])) != camera["sha256"]:
        raise ValueError("Scale estimate and geometry do not reference the same source capture/scale group")
    g_rows, v_rows, s_rows = (read_jsonl(files[key]) for key in ("geometry_frames", "visibility_frames", "segmentation_frames"))
    if len(g_rows) != count or len(v_rows) != count or len(s_rows) != count or len(source["frames"]) != count:
        raise ValueError("Incomplete frame inventories")
    external_masks, output = [], []
    for frame, (g, v, s, image_record) in enumerate(zip(g_rows, v_rows, s_rows, source["frames"])):
        if any(row["frame"] != frame for row in (g, v, s, image_record)): raise ValueError("Source timeline is missing, duplicated or out of order")
        if g["source_frame"] != frame or s["source_frame"] != frame: raise ValueError("Source frame index differs from timeline frame")
        if g["source_video_sha256"] != source_hash or s["source_video_sha256"] != source_hash: raise ValueError("Per-frame source video mismatch")
        if g["time_seconds"] != s["time_seconds"] or v["time_seconds"] != s["time_seconds"] or abs(s["time_seconds"]-frame/source["fps"]) > 1e-9:
            raise ValueError("Per-frame time mismatch")
        if g["source_jpeg_sha256"] != image_record["jpeg_sha256"] or s["source_jpeg_sha256"] != image_record["jpeg_sha256"]:
            raise ValueError("Per-frame source JPEG mismatch")
        _bound(_inside(source["frame_directory"], image_record["jpeg_file"]), image_record["jpeg_sha256"], "source frame JPEG")
        g_objects, s_objects, v_masks = _objects(g["objects"]), _objects(s["objects"]), _objects(v["mask_provenance"])
        if not (set(g_objects) == set(s_objects) == set(v_masks) == set(v["visibility"]["objects"])):
            raise ValueError("Frame surface IDs differ across source records")
        if set(g_objects) - set(seed_by_id): raise ValueError("Surface track has no recorded seed provenance")
        row = deepcopy(g); row["format_version"] = VERSION
        row.update(physical_packet_count=None, physical_events=[], track_visibility=deepcopy(v["visibility"]),
                   missing_surface_track_ids=sorted(set(seed_by_id)-set(g_objects)))
        row["objects"] = []
        for identity, obj in g_objects.items():
            segment, vm = s_objects[identity], v_masks[identity]
            seed = seed_by_id[identity]
            if obj["sam2_object_id"] != seed["sam2_object_id"] or segment["sam2_object_id"] != seed["sam2_object_id"]:
                raise ValueError("SAM2 object ID differs from the supplied seed mapping")
            if segment.get("seed_candidate_mask_sha256") != seed["mask_sha256"] or segment.get("seed_candidate_file_sha256") != seed.get("candidate_file_sha256"):
                raise ValueError("Per-observation candidate provenance differs from its seed")
            if obj["segmentation"] != segment: raise ValueError("Geometry changed the source segmentation observation")
            for key in ("mask_file", "mask_sha256"):
                if obj[key] != segment[key] or vm[key] != segment[key]: raise ValueError("Geometry/visibility mask provenance mismatch")
            mask_path = _bound(_inside(segmentation_run, segment["mask_file"]), segment["mask_sha256"], "visible mask")
            mask = cv2.imdecode(np.frombuffer(mask_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (source["resolution"][1], source["resolution"][0]) or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
                raise ValueError("Mask must be source-resolution binary PNG")
            area = int(np.count_nonzero(mask))
            if area != segment["area_px"] or (area == 0) != segment["empty"] or area != v["visibility"]["objects"][identity]["area_px"]:
                raise ValueError("Recorded mask visibility/area differs from actual pixels")
            if segment.get("mask_source") == "supplied_mask_conditioning_reprojection" and segment["raw_object_presence_logit"] is not None:
                raise ValueError("Conditioning mask must not expose input-derived presence as learned confidence")
            external = {"path": str(mask_path), "sha256": segment["mask_sha256"]}
            external_masks.append(dict(external, frame=frame, surface_track_id=identity))
            result = deepcopy(obj)
            result.update(surface_track_id=identity, surface_identity_kind="prompted_video_surface_track_candidate",
                physical_packet_id=None, card_count=None, thickness_m=None, contact_state=None, physical_identity_verified=False,
                mask_reference=external, track_visibility=deepcopy(v["visibility"]["objects"][identity]),
                missing_observation=area == 0, missing_observation_reason="empty_source_model_mask" if area == 0 else None)
            row["objects"].append(result)
        output.append(row)
    visualization = None
    if (visualization_video is None) != (visualization_report is None): raise ValueError("Visualization video requires its explicit source-bound report")
    if visualization_report:
        vr_path = Path(visualization_report).resolve(); vr = read_json(vr_path)
        if vr.get("status") != "completed" or vr.get("source_video_sha256") != source_hash or vr.get("frame_count") != count:
            raise ValueError("Visualization source/timeline mismatch")
        _bound(files["geometry_frames"], vr["geometry_records_sha256"], "visualization geometry")
        _bound(files["geometry_report"], vr["geometry_report_sha256"], "visualization geometry report")
        video = _bound(visualization_video, vr["output_video_sha256"], "visualization video")
        if video != Path(vr["output_video"]).resolve(): raise ValueError("Visualization report describes another video path")
        files["visualization_report"] = vr_path
        visualization = {"path": str(video), "sha256": vr["output_video_sha256"], "frame_count": count,
                         "full_output_redecode_frame_count_recorded": vr.get("full_output_redecode_frame_count"), "kind": "existing_geometry_candidate_review_video"}
    return dict(files=files, geometry=geometry, visibility=visibility, source=source, seeds=seeds, scale=scale,
                observations=output, masks=external_masks, visualization=visualization, source_video=str(source_video))


def _derived_manifest_fields(data):
    records = data["observations"]
    objects = [o for row in records for o in row["objects"]]
    counts = {"frames": len(records), "surface_observation_samples": len(objects), "surface_track_ids": len(data["seeds"]["objects"]),
        "empty_mask_samples": sum(o["missing_observation"] for o in objects), "quad_accepted": sum(o["quad"]["accepted"] for o in objects),
        "pose_accepted": sum(o["pose"]["accepted"] for o in objects), "pose_candidates": sum(len(o["pose"]["candidates"]) for o in objects),
        "full_face_verified_samples": sum(o["quad"]["full_face_verified"] for o in objects),
        "physical_packet_count": None, "physical_events": 0}
    return {"format_version": VERSION, "status": "completed", "ue_ingest_supported": False, "final_cardcap_1_0": False,
        "description": "Prompted video surface tracking and geometry candidates; physical packet identities remain unverified",
        "automatic_instance_discovery": False, "prompts_are_supported_input": True, "surface_tracks_are_physical_packets": False,
        "source": {"video": data["source_video"], "sha256": data["source"]["source_video_sha256"],
                   "fps": data["source"]["fps"], "frame_count": data["source"]["frame_count"], "resolution": data["source"]["resolution"]},
        "camera": data["geometry"]["camera"], "surface_track_seeds": data["seeds"], "counts": counts,
        "physical_packet_count": None, "card_count": None, "thickness_m": None, "contact_state": None, "physical_events": [],
        "scale": {"applied_to_observations": False, "applied_to_hands_or_assets": False, "unified_metric_scene_established": False,
                  "estimate": data["scale"]["scale_estimate"], "scope": "Unapplied estimate for its explicitly named source scale group; do not rescale already-metric PnP outputs"},
        "external_mask_references": data["masks"], "visualization": data["visualization"],
        "new_model_inference": False, "geometry_refitted": False, "missing_poses_filled": False,
        "external_files_required": True, "independent_metric_or_corner_accuracy_claimed": False}


def _readme_text(manifest):
    counts, video = manifest["counts"], manifest["visualization"]
    video_text = f"打开已有诊断视频：[牌面候选叠加视频]({Path(video['path']).as_posix()})。视频仅引用原文件，不复制到本包。" if video else "本包未指定诊断视频；可查看逐帧观测和来源记录。"
    estimate = manifest["scale"]["estimate"]
    if estimate["anchor_method"] == "hand_prior":
        scale_text = "尺度只保存尚未施加的估计。低置信度手长先验没有缩放手、牌、相机或 M2 资产，也不表示已得到统一米制场景。"
    elif estimate["meters_per_source_unit"] is None:
        scale_text = "尺度仍未知/null。恒等因子只保留模型相对坐标，没有施加米制缩放，也不表示已得到统一米制场景。"
    else:
        scale_text = "尺度只保存有来源但尚未施加的估计，没有缩放手、牌、相机或已有资产，也不表示手与牌已得到统一米制场景。"
    return f"""# 牌面观测草稿

本包整理 {counts['frames']} 帧、{counts['surface_observation_samples']} 个表面观测。可查看提示引导的可见牌面区域、四边形与姿态候选，以及遮挡、歧义和缺测记录。原结果中有 {counts['quad_accepted']} 个四边形拟合、{counts['pose_accepted']} 个姿态拟合通过原门槛，另有 {counts['empty_mask_samples']} 个空掩膜；通过拟合不代表完整牌面或物理身份已验证。

{video_text}

`manifest.json` 是总目录，`observations.jsonl` 每行对应一帧，`provenance` 保存来源报告。表面轨迹 ID 不是物理牌块 ID；提示是支持的输入方式。物理牌块、牌张数、厚度和接触仍为未知/null，物理事件为空。

{scale_text}本包没有补造缺测姿态。

这是独立草稿格式 `{VERSION}`，当前 UE 不支持导入，也不是最终 cardcap 1.0。PNG、原视频和诊断视频保持外部路径与 SHA256 引用，需保留原文件；本包不复制权重或整套图像。

技术回读：在 PythonPipeline 环境运行 `python -m cardcap.cards.draft_export --verify 草稿包目录`。
"""


def export_draft(geometry_run, visibility_run, scale_estimate, output_dir, **kwargs):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a new nonempty-free output directory; existing artifacts are preserved")
    data = validate_sources(geometry_run, visibility_run, scale_estimate, **kwargs)
    output.mkdir(parents=True, exist_ok=True); (output / "provenance").mkdir(exist_ok=True)
    inputs = {}
    for key, source in data["files"].items():
        # Geometry/segmentation/visibility JSONL content is already represented
        # in observations. Copy compact reports and provenance, not image trees.
        copy = source.suffix.lower() == ".json"
        target = output / "provenance" / (key + ".json") if copy else None
        payload = source.read_bytes()
        if copy: target.write_bytes(payload)
        inputs[key] = {"path": str(source), "sha256": hashlib.sha256(payload).hexdigest(),
                       "bundled_file": target.relative_to(output).as_posix() if copy else None}
    records = data["observations"]
    observations = output / "observations.jsonl"
    with observations.open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = dict(_derived_manifest_fields(data), created_at_utc=datetime.now(timezone.utc).isoformat(),
        exporter_sha256=sha(__file__), inputs=inputs, observations={"file": "observations.jsonl", "sha256": sha(observations)})
    (output / "README.md").write_text(_readme_text(manifest), encoding="utf-8")
    manifest["readme_sha256"] = sha(output / "README.md")
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_bundle(output_dir):
    """Read the whole bundle and rebind its original input chain on CPU."""
    output = Path(output_dir).resolve(); manifest = read_json(output / "manifest.json")
    if manifest["format_version"] != VERSION or manifest["status"] != "completed": raise ValueError("Not a completed card draft bundle")
    _bound(output / "README.md", manifest["readme_sha256"], "bundle README")
    _bound(_inside(output, manifest["observations"]["file"]), manifest["observations"]["sha256"], "bundle observations")
    for item in manifest["inputs"].values():
        _bound(item["path"], item["sha256"], "original bundle input")
        if item["bundled_file"]: _bound(_inside(output, item["bundled_file"]), item["sha256"], "copied provenance")
    visualization = manifest["visualization"]
    source = validate_sources(Path(manifest["inputs"]["geometry_report"]["path"]).parent, Path(manifest["inputs"]["visibility_report"]["path"]).parent,
        manifest["inputs"]["scale_estimate"]["path"], visualization_video=visualization["path"] if visualization else None,
        visualization_report=manifest["inputs"]["visualization_report"]["path"] if visualization else None)
    if read_jsonl(output / manifest["observations"]["file"]) != source["observations"]: raise ValueError("Bundle observations differ from the verified original records")
    for key, expected in _derived_manifest_fields(source).items():
        if manifest.get(key) != expected: raise ValueError(f"Bundle manifest summary differs from verified sources: {key}")
    return {"status": "passed", "format_version": VERSION, "manifest_sha256": sha(output / "manifest.json"),
            "observations_sha256": manifest["observations"]["sha256"], "frames_read_back": len(source["observations"]),
            "mask_samples_revalidated": len(source["masks"]), "new_model_inference": False, "ue_ingest_supported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path); parser.add_argument("--visibility-run", type=Path)
    parser.add_argument("--scale-estimate", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--visualization-video", type=Path); parser.add_argument("--visualization-report", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_bundle(args.verify)
    else:
        if not all((args.geometry_run, args.visibility_run, args.scale_estimate, args.output)): parser.error("Provide geometry, visibility, scale and output, or --verify")
        export_draft(args.geometry_run, args.visibility_run, args.scale_estimate, args.output,
            visualization_video=args.visualization_video, visualization_report=args.visualization_report)
        result = verify_bundle(args.output)
        write_json(args.output / "verification.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
