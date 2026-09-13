"""Strict cardcap 1.0/1.1/1.2/1.3 validation and value-preserving 1.0 -> 1.1 upgrade.

This is the final interchange contract, not the surface-observation draft format.
No inference, pose filling, coordinate conversion, scale application or event
discovery occurs here. Extra diagnostic fields are preserved, as in the UE reader.
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MAPPING = Path(__file__).resolve().parents[2] / "Config/BoneMapping_UE5Mannequin.json"
MAX_INT32 = 2**31 - 1
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
PROVENANCE_SOURCES = frozenset(("observed", "calibrated", "user_measured", "inferred", "unobservable"))
SAMPLE_KINDS = frozenset(("detected_model_observation", "tracked_roi_model_observation",
                          "interpolated", "endpoint_hold", "missing"))


def _ue_ascii_key(value: str) -> str:
    """UE's historical case-insensitive contract, restricted to ASCII folding.

    Preserve non-ASCII code points: Python Unicode casefold is not UE Stricmp.
    No claim is made of identical comparison for arbitrary Unicode packet names.
    """
    return value.translate(_ASCII_LOWER)


class ContractError(ValueError):
    """Invalid interchange data; the message identifies its JSON path."""


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ContractError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict:
    _require(type(value) is dict, path, "expected object")
    _require(all(type(key) is str for key in value), path, "expected string object keys")
    _require(len({_ue_ascii_key(key) for key in value}) == len(value), path, "duplicate object key (ASCII case insensitive)")
    return value


def _field_key(value: dict, key: str, path: str) -> str:
    matches = [original for original in value if _ue_ascii_key(original) == _ue_ascii_key(key)]
    _require(len(matches) > 0, f"{path}.{key}", "required field is missing")
    _require(len(matches) == 1, f"{path}.{key}", "duplicate object key (ASCII case insensitive)")
    return matches[0]


def _field(value: dict, key: str, path: str) -> Any:
    return value[_field_key(value, key, path)]


def _array(value: Any, path: str, count: int | None = None) -> list:
    _require(type(value) is list, path, "expected array")
    _require(count is None or len(value) == count, path, f"expected exactly {count} entries")
    return value


def _number(value: Any, path: str, minimum: float | None = None,
            maximum: float | None = None, positive: bool = False) -> int | float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, path, "expected finite JSON number (not boolean or string)")
    _require(minimum is None or value >= minimum, path, f"must be >= {minimum}")
    _require(maximum is None or value <= maximum, path, f"must be <= {maximum}")
    _require(not positive or value > 0, path, "must be positive")
    return value


def _integer(value: Any, path: str, minimum: int = 0, maximum: int = MAX_INT32) -> int:
    _number(value, path, minimum, maximum)
    _require(value == math.floor(value), path, "expected integer")
    return int(value)


def _string(value: Any, path: str) -> str:
    _require(type(value) is str and len(value) > 0, path, "expected nonempty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    _require(type(value) is bool, path, "expected boolean")
    return value


def _numbers(value: Any, path: str, count: int | None = None) -> list:
    return [_number(v, f"{path}[{i}]") for i, v in enumerate(_array(value, path, count))]


def _strings(value: Any, path: str, unique: bool = False,
             count: int | None = None) -> list[str]:
    values = [_string(v, f"{path}[{i}]") for i, v in enumerate(_array(value, path, count))]
    _require(not unique or len({_ue_ascii_key(v) for v in values}) == len(values), path, "duplicate string")
    return values


def _integers(value: Any, path: str, minimum: int, maximum: int,
              unique: bool = True, count: int | None = None) -> list[int]:
    values = [_integer(v, f"{path}[{i}]", minimum, maximum)
              for i, v in enumerate(_array(value, path, count))]
    _require(not unique or len(set(values)) == len(values), path, "duplicate index")
    return values


def _quaternion(value: Any, path: str) -> None:
    values = _numbers(value, path, 4)
    norm2 = sum(v * v for v in values)
    _require(math.isfinite(norm2) and abs(norm2 - 1.) <= 1e-4, path,
             "expected unit quaternion (squared-norm tolerance 1e-4)")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    seen = set()
    for key, value in pairs:
        folded = _ue_ascii_key(key)
        _require(folded not in seen, "$", f"duplicate object key {key!r} (ASCII case insensitive)")
        seen.add(folded)
        result[key] = value
    return result


def loads_strict(text: str) -> dict:
    def invalid_constant(value: str) -> None:
        raise ContractError(f"$: invalid JSON numeric constant {value}")
    def finite_float(value: str) -> float:
        number = float(value)
        _require(math.isfinite(number), "$", "nonfinite JSON number")
        return number
    try:
        result = json.loads(text, object_pairs_hook=_unique_pairs,
                            parse_constant=invalid_constant, parse_float=finite_float)
    except json.JSONDecodeError as error:
        raise ContractError(f"$: invalid JSON: {error}") from error
    return _object(result, "$")


def read_json(path: Path) -> dict:
    return loads_strict(Path(path).read_text(encoding="utf-8-sig"))


def validate_mapping(mapping: dict) -> dict[str, list[str]]:
    """Validate the same joint/name constraints used by the UE mapping reader."""
    p = "mapping"
    _object(mapping, p)
    _integer(_field(mapping, "schema_version", p), p + ".schema_version", 1, 1)
    root = _string(_field(mapping, "root_bone", p), p + ".root_bone")
    _require(_ue_ascii_key(root) != "none", p + ".root_bone", "invalid bone name")
    _strings(_field(mapping, "mano_joint_order", p), p + ".mano_joint_order", True, 16)
    parents = _integers(_field(mapping, "mano_parents", p), p + ".mano_parents", -1, 15, False, 16)
    _require(parents[0] == -1, p + ".mano_parents[0]", "wrist parent must be -1")
    for i in range(1, 16):
        _require(0 <= parents[i] < i, f"{p}.mano_parents[{i}]", "parent must precede child")
    _integers(_field(mapping, "landmark_indices", p), p + ".landmark_indices", 0, 20, True, 16)
    hands = _object(_field(mapping, "hands", p), p + ".hands")
    all_names = {_ue_ascii_key(root)}
    names = {}
    for side in ("left", "right"):
        q = p + ".hands." + side
        hand = _object(_field(hands, side, p + ".hands"), q)
        names[side] = _strings(_field(hand, "bone_names", q), q + ".bone_names", True, 16)
        _numbers(_field(hand, "rest_offset_camera_m", q), q + ".rest_offset_camera_m", 3)
        for i, name in enumerate(names[side]):
            folded = _ue_ascii_key(name)
            _require(folded != "none" and folded not in all_names,
                     f"{q}.bone_names[{i}]", "invalid or duplicate bone name")
            all_names.add(folded)
    return names


def validate_capture(document: dict, mapping: dict | None = None) -> dict:
    """Return validation counts without changing or sorting any input value.

    Contract compatibility follows CardCapJsonParser.cpp, including required keys,
    sparse frames, ignored extra diagnostics and existing event-reference checks.
    It does not establish physical identity, contact, metric accuracy or whether
    caller-supplied measurements/provenance are true.
    """
    root = _object(document, "$")
    version = _string(_field(root, "format_version", "$"), "$.format_version")
    _require(version in ("1.0", "1.1", "1.2", "1.3"), "$.format_version", "unsupported format; expected 1.0, 1.1, 1.2 or 1.3")
    local13 = version == "1.3"
    provenance12 = version in ("1.2", "1.3")
    nullable = version in ("1.1", "1.2", "1.3")
    per_hand_local = False
    names = validate_mapping(read_json(DEFAULT_MAPPING) if mapping is None else mapping)
    contact_names = {_ue_ascii_key(name) for name in names["left"] + names["right"]}
    meta = _object(_field(root, "meta", "$"), "$.meta")
    for key in ("source_video", "processed_at", "pipeline_version"):
        _string(_field(meta, key, "$.meta"), "$.meta." + key)
    _number(_field(meta, "fps", "$.meta"), "$.meta.fps", positive=True)
    count = _integer(_field(meta, "frame_count", "$.meta"), "$.meta.frame_count", 1)
    _integers(_field(meta, "resolution", "$.meta"), "$.meta.resolution", 1, MAX_INT32, False, 2)
    last = count - 1
    camera = _object(_field(root, "camera", "$"), "$.camera")
    intrinsics = _field(camera, "intrinsics", "$.camera")
    if not (provenance12 and intrinsics is None):
        _object(intrinsics, "$.camera.intrinsics")
        for key in ("fx", "fy", "cx", "cy"):
            _number(_field(intrinsics, key, "$.camera.intrinsics"), "$.camera.intrinsics." + key,
                    positive=key in ("fx", "fy"))
    distortion = _field(camera, "distortion", "$.camera")
    if not (provenance12 and distortion is None):
        _numbers(distortion, "$.camera.distortion")
        _require(len(distortion) in (4, 5, 8, 12, 14), "$.camera.distortion", "expected OpenCV count 4, 5, 8, 12 or 14")
    calibrated = _boolean(_field(camera, "calibrated", "$.camera"), "$.camera.calibrated")
    scale = _object(_field(root, "scale", "$"), "$.scale")
    metric_scale = _field(scale, "meters_per_unit", "$.scale")
    if not (provenance12 and metric_scale is None):
        _number(metric_scale, "$.scale.meters_per_unit", positive=True)
    method = _field(scale, "anchor_method", "$.scale")
    if not (provenance12 and method is None):
        _string(method, "$.scale.anchor_method")
        _require(_ue_ascii_key(method) in ("card_pnp", "hand_prior") or (provenance12 and method == "user_measured"),
                 "$.scale.anchor_method", "expected card_pnp, hand_prior, or (1.2+) user_measured")
    scale_confidence = _field(scale, "confidence", "$.scale")
    if not (provenance12 and scale_confidence is None):
        _number(scale_confidence, "$.scale.confidence", 0, 1)
    anchor_frames = _integers(_field(scale, "anchor_frames", "$.scale"), "$.scale.anchor_frames", 0, last)
    if provenance12:
        provenance = _object(_field(root, "provenance", "$"), "$.provenance")
        sources = {}
        for key in ("camera_intrinsics", "camera_distortion", "metric_scale", "hand_geometry"):
            source = _string(_field(provenance, key, "$.provenance"), "$.provenance." + key)
            _require(source in PROVENANCE_SOURCES, "$.provenance." + key,
                     "expected observed, calibrated, user_measured, inferred or unobservable")
            sources[key] = source
        units = _string(_field(provenance, "coordinate_units", "$.provenance"), "$.provenance.coordinate_units")
        assumptions = _strings(_field(provenance, "assumptions", "$.provenance"), "$.provenance.assumptions")
        validity = _object(_field(root, "validity", "$"), "$.validity")
        supplied = {"camera_intrinsics": intrinsics is not None, "camera_distortion": distortion is not None,
                    "metric_scale": metric_scale is not None}
        for key, has_value in supplied.items():
            mask = _boolean(_field(validity, key, "$.validity"), "$.validity." + key)
            _require(mask == has_value, "$.validity." + key, "mask must match null/non-null field")
        for key, has_value in supplied.items():
            _require(has_value == (sources[key] != "unobservable"), "$.provenance." + key,
                     "unobservable requires null; a supplied value requires a source")
        _require(not calibrated or (supplied["camera_intrinsics"] and sources["camera_intrinsics"] == "calibrated"),
                 "$.camera.calibrated", "calibrated flag requires calibrated intrinsics provenance")
        _require(method is not None if supplied["metric_scale"] else
                 (method is None and scale_confidence is None and not anchor_frames), "$.scale",
                 "unknown metric scale requires null anchor/confidence and empty anchor frames; known scale requires an anchor")
        _require(units == ("ue_centimeters" if supplied["metric_scale"] else "conditional_ue_units"),
                 "$.provenance.coordinate_units", "units must agree with metric scale validity")
        _require(supplied["metric_scale"] or bool(assumptions), "$.provenance.assumptions",
                 "conditional geometry requires explicit assumptions")
        if local13:
            coordinate_frame = _string(_field(provenance,"coordinate_frame","$.provenance"),"$.provenance.coordinate_frame")
            _require(coordinate_frame in ("shared_camera","per_hand_wrist_local"),"$.provenance.coordinate_frame","unsupported coordinate frame")
            per_hand_local = coordinate_frame == "per_hand_wrist_local"
            inter_hand = _boolean(_field(validity,"inter_hand_transform","$.validity"),"$.validity.inter_hand_transform")
            _require(inter_hand == (not per_hand_local),"$.validity.inter_hand_transform","must match coordinate frame")
            _require(not per_hand_local or (intrinsics is None and distortion is None and not calibrated),
                     "$.camera","per-hand local output must not publish a shared camera")
            _require(per_hand_local or intrinsics is not None,"$.camera.intrinsics","shared-camera output needs intrinsics")
    hands = _array(_field(root, "hands", "$"), "$.hands")
    _require(1 <= len(hands) <= 2, "$.hands", "expected one or two unique hand sides")
    sides, hand_sample_counts = set(), {}
    for h, hand in enumerate(hands):
        p = f"$.hands[{h}]"
        _object(hand, p)
        side = _string(_field(hand, "side", p), p + ".side")
        side_key = _ue_ascii_key(side)
        _require(side_key in names and side_key not in sides, p + ".side", "invalid or duplicate hand side")
        sides.add(side_key)
        _numbers(_field(hand, "mano_shape", p), p + ".mano_shape", 10)
        frames = _array(_field(hand, "frames", p), p + ".frames")
        seen = set()
        for i, frame in enumerate(frames):
            q = f"{p}.frames[{i}]"
            _object(frame, q)
            index = _integer(_field(frame, "frame", q), q + ".frame", 0, last)
            _require(index not in seen, q + ".frame", "duplicate hand frame")
            seen.add(index)
            _number(_field(frame, "confidence", q), q + ".confidence", 0, 1)
            translation = _field(frame, "global_trans_cm", q)
            if per_hand_local:
                _require(translation is None,q+".global_trans_cm","unknown shared translation must be null, not a local zero origin")
            else:
                _numbers(translation, q + ".global_trans_cm", 3)
            _quaternion(_field(frame, "global_rot_quat", q), q + ".global_rot_quat")
            joints = _array(_field(frame, "joint_positions_cm", q), q + ".joint_positions_cm", 21)
            for j, joint in enumerate(joints):
                _numbers(joint, f"{q}.joint_positions_cm[{j}]", 3)
            if per_hand_local:
                _require(all(abs(x) <= 1e-8 for x in joints[0]),q+".joint_positions_cm[0]","wrist-local frame must have a zero local wrist origin")
            rotations = _object(_field(frame, "bone_rotations", q), q + ".bone_rotations")
            _require({_ue_ascii_key(key) for key in rotations} == {_ue_ascii_key(key) for key in names[side_key][1:]},
                     q + ".bone_rotations", "expected exactly the 15 configured finger bones (legacy ASCII case insensitive)")
            for bone, rotation in rotations.items():
                _quaternion(rotation, q + ".bone_rotations." + bone)
            _integers(_field(frame, "occluded_joints", q), q + ".occluded_joints", 0, 20)
            if provenance12:
                kind = _string(_field(frame, "sample_kind", q), q + ".sample_kind")
                _require(kind in SAMPLE_KINDS, q + ".sample_kind", "unknown sample kind")
                validity = _object(_field(frame, "validity", q), q + ".validity")
                pose = _boolean(_field(validity, "pose", q + ".validity"), q + ".validity.pose")
                _require(pose == (kind != "missing"), q + ".validity.pose",
                         "missing must be invalid; supplied conditional poses must be valid")
                if local13:
                    has_translation = _boolean(_field(validity,"global_translation",q+".validity"),q+".validity.global_translation")
                    _require(has_translation == (translation is not None),q+".validity.global_translation","must match null/non-null translation")
        hand_sample_counts[side] = len(frames)
    packets = _array(_field(root, "packets", "$"), "$.packets")
    _require(not per_hand_local or not packets,"$.packets","per-hand local output cannot place packets in an unknown shared space")
    ids, packet_sample_counts = set(), {}
    for i, packet in enumerate(packets):
        p = f"$.packets[{i}]"
        _object(packet, p)
        packet_id = _string(_field(packet, "id", p), p + ".id")
        id_key = _ue_ascii_key(packet_id)
        _require(id_key not in ids, p + ".id", "duplicate packet id")
        ids.add(id_key)
        birth = _integer(_field(packet, "birth_frame", p), p + ".birth_frame", 0, last)
        death = _integer(_field(packet, "death_frame", p), p + ".death_frame", birth, last)
        card_count = _field(packet, "card_count_estimate", p)
        card_confidence = _field(packet, "card_count_confidence", p)
        if not (nullable and card_count is None):
            _integer(card_count, p + ".card_count_estimate", 1)
        if not (nullable and card_confidence is None):
            _number(card_confidence, p + ".card_count_confidence", 0, 1)
        _require((card_count is None) == (card_confidence is None), p + ".card_count_confidence", "count and confidence must both be numbers or both be null")
        dimensions = _field(packet, "dimensions_cm", p)
        if not (provenance12 and dimensions is None):
            _array(dimensions, p + ".dimensions_cm", 3)
            for d, value in enumerate(dimensions):
                if not (value is None and (provenance12 or (nullable and d == 2))):
                    _number(value, f"{p}.dimensions_cm[{d}]", positive=True)
        frames = _array(_field(packet, "frames", p), p + ".frames")
        seen = set()
        for j, frame in enumerate(frames):
            q = f"{p}.frames[{j}]"
            _object(frame, q)
            index = _integer(_field(frame, "frame", q), q + ".frame", birth, death)
            _require(index not in seen, q + ".frame", "duplicate packet frame")
            seen.add(index)
            _number(_field(frame, "confidence", q), q + ".confidence", 0, 1)
            _numbers(_field(frame, "position_cm", q), q + ".position_cm", 3)
            _quaternion(_field(frame, "rotation_quat", q), q + ".rotation_quat")
            for key in ("linear_velocity_cm_s", "angular_velocity_rad_s"):
                value = _field(frame, key, q)
                if not (nullable and value is None):
                    _numbers(value, q + "." + key, 3)
            error = _field(frame, "reprojection_error_px", q)
            if not (nullable and error is None):
                _number(error, q + ".reprojection_error_px", 0)
            state = _string(_field(frame, "contact_state", q), q + ".contact_state")
            # UE's original FString comparisons accepted case variants of the
            # four 1.0 states. Preserve that behavior; the new UNKNOWN is exact.
            valid_state = _ue_ascii_key(state) in {"gripped", "free", "resting", "sliding"} or (nullable and state == "UNKNOWN")
            _require(valid_state, q + ".contact_state", "unknown contact state")
            bones = _field(frame, "contact_bones", q)
            if not (nullable and bones is None):
                for k, bone in enumerate(_strings(bones, q + ".contact_bones", True)):
                    _require(_ue_ascii_key(bone) in contact_names, f"{q}.contact_bones[{k}]", "bone absent from mapping")
        packet_sample_counts[packet_id] = len(frames)
    events = _object(_field(root, "events", "$"), "$.events")
    event_counts = {}
    seen_events = set()
    for kind, actor, members in (("splits", "source", "results"), ("merges", "result", "sources"), ("releases", "packet", None)):
        rows = _array(_field(events, kind, "$.events"), "$.events." + kind)
        event_counts[kind] = len(rows)
        for i, event in enumerate(rows):
            p = f"$.events.{kind}[{i}]"
            _object(event, p)
            index = _integer(_field(event, "frame", p), p + ".frame", 0, last)
            subject = _string(_field(event, actor, p), p + "." + actor)
            subject_key = _ue_ascii_key(subject)
            _require(subject_key in ids, p + "." + actor, "unknown packet id")
            if members is not None:
                references = _strings(_field(event, members, p), p + "." + members, True)
                _require(len(references) >= 2, p + "." + members, "expected at least two unique packet ids")
                for j, reference in enumerate(references):
                    reference_key = _ue_ascii_key(reference)
                    _require(reference_key in ids and reference_key != subject_key, f"{p}.{members}[{j}]", "unknown packet id or self reference")
            else:
                release_velocity = _field(event, "release_velocity_cm_s", p)
                if not (provenance12 and release_velocity is None):
                    _numbers(release_velocity, p + ".release_velocity_cm_s", 3)
            key = (kind, index, subject_key)
            _require(key not in seen_events, p, "duplicate event type/frame/subject")
            seen_events.add(key)
    quality = _object(_field(root, "quality", "$"), "$.quality")
    _number(_field(quality, "mean_hand_confidence", "$.quality"), "$.quality.mean_hand_confidence", 0, 1)
    _strings(_field(quality, "warnings", "$.quality"), "$.quality.warnings")
    mean_error = _field(quality, "mean_reprojection_error_px", "$.quality")
    _require(not per_hand_local or mean_error is None,"$.quality.mean_reprojection_error_px",
             "local-only geometry has no shared-camera reprojection error; crop diagnostics must be separate")
    if mean_error is not None:
        _number(mean_error, "$.quality.mean_reprojection_error_px", 0)
    ranges = _array(_field(quality, "low_confidence_ranges", "$.quality"), "$.quality.low_confidence_ranges")
    previous = -1
    for i, bounds in enumerate(ranges):
        p = f"$.quality.low_confidence_ranges[{i}]"
        _array(bounds, p, 2)
        start = _integer(bounds[0], p + "[0]", 0, last)
        end = _integer(bounds[1], p + "[1]", start, last)
        _require(start > previous, p, "ranges must be ordered and nonoverlapping")
        previous = end
    return {"status": "passed", "format_version": version, "frame_count": count,
            "hand_count": len(hands), "hand_sample_counts": hand_sample_counts,
            "packet_count": len(packets), "packet_sample_counts": packet_sample_counts,
            "event_counts": event_counts, "sparse_frames_preserved": True,
            "measurement_truth_or_physical_identity_validated": False}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _preserved_fields(source: dict, upgraded: dict) -> dict:
    _require(_field(source, "format_version", "$") == "1.0" and _field(upgraded, "format_version", "$") == "1.1", "$", "expected 1.0 input and 1.1 output")
    a = {key: value for key, value in source.items() if _ue_ascii_key(key) != "format_version"}
    b = {key: value for key, value in upgraded.items() if _ue_ascii_key(key) != "format_version"}
    _require(_field_key(source, "format_version", "$") == _field_key(upgraded, "format_version", "$"), "$", "version field spelling changed")
    # JSON canonical bytes also distinguish booleans/numbers and 1 from 1.0.
    _require(_canonical(a) == _canonical(b), "$", "upgrade changed values outside format_version")
    return {key: hashlib.sha256(_canonical(value)).hexdigest() for key, value in a.items()}


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_upgrade(output_dir: Path) -> dict:
    """Read both files afresh, bind their hashes and compare every preserved value."""
    output_dir = Path(output_dir).resolve()
    report = read_json(output_dir / "upgrade_report.json")
    _require(report.get("schema") == "cardcap.format_upgrade/1.0", "report.schema", "unknown upgrade report")
    source_path = Path(report["input"]["path"])
    mapping_path = Path(report["bone_mapping"]["path"])
    output_path = output_dir / "capture.cardcap.json"
    for label, path in (("input", source_path), ("bone_mapping", mapping_path), ("output", output_path)):
        _require(str(path.resolve()) == report[label]["path"], "report." + label, "path mismatch")
        _require(sha256(path) == report[label]["sha256"], "report." + label, "file hash mismatch")
    source, upgraded, mapping = read_json(source_path), read_json(output_path), read_json(mapping_path)
    input_checks = validate_capture(source, mapping)
    output_checks = validate_capture(upgraded, mapping)
    preserved = _preserved_fields(source, upgraded)
    expected = {"changed_json_paths": ["$.format_version"],
                "preserved_top_level_canonical_sha256": preserved,
                "input_validation": input_checks, "output_validation": output_checks,
                "new_packet_data_constructed": False, "events_added": False,
                "scale_or_coordinate_conversion_applied": False,
                "poses_added_removed_or_filled": False}
    for key, value in expected.items():
        _require(report.get(key) == value, "report." + key, "summary differs from full readback")
    return {"status": "passed", "schema": "cardcap.format_upgrade_readback/1.0",
            "input_sha256": sha256(source_path), "output_sha256": sha256(output_path),
            "upgrade_report_sha256": sha256(output_dir / "upgrade_report.json"),
            **expected,
            "scope": "Every input/output JSON value compared except format_version; no physical packet, camera or scale accuracy claim."}


def upgrade_to_1_1(input_path: Path, output_dir: Path,
                    mapping_path: Path = DEFAULT_MAPPING) -> dict:
    """Write a new directory containing a strictly version-only upgraded capture.

    An existing empty directory is allowed; existing contents are never replaced.
    The source file and all original scalar/list/object values remain unchanged.
    A 1.1 source is rejected: this entry point is specifically a 1.0 upgrade.
    """
    input_path, output_dir, mapping_path = (Path(p).resolve() for p in (input_path, output_dir, mapping_path))
    if output_dir.exists():
        _require(output_dir.is_dir() and not any(output_dir.iterdir()), "output", "directory must be new or empty; existing artifacts are preserved")
    source_hash = sha256(input_path)
    mapping_hash = sha256(mapping_path)
    source, mapping = read_json(input_path), read_json(mapping_path)
    input_checks = validate_capture(source, mapping)
    _require(_field(source, "format_version", "$") == "1.0", "$.format_version", "upgrade input must be 1.0")
    upgraded = copy.deepcopy(source)
    upgraded[_field_key(upgraded, "format_version", "$")] = "1.1"
    output_checks = validate_capture(upgraded, mapping)
    preserved = _preserved_fields(source, upgraded)
    _require(sha256(input_path) == source_hash and sha256(mapping_path) == mapping_hash, "$", "source changed during validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capture.cardcap.json"
    _write_new_json(output_path, upgraded)
    report = {"schema": "cardcap.format_upgrade/1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": {"path": str(input_path), "sha256": source_hash},
              "output": {"path": str(output_path), "sha256": sha256(output_path)},
              "bone_mapping": {"path": str(mapping_path), "sha256": mapping_hash},
              "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
              "changed_json_paths": ["$.format_version"],
              "preserved_top_level_canonical_sha256": preserved,
              "input_validation": input_checks, "output_validation": output_checks,
              "new_packet_data_constructed": False, "events_added": False,
              "scale_or_coordinate_conversion_applied": False,
              "poses_added_removed_or_filled": False}
    _write_new_json(output_dir / "upgrade_report.json", report)
    readback = verify_upgrade(output_dir)
    _write_new_json(output_dir / "readback.json", readback)
    readme = ("# cardcap 1.1 兼容文件\n\n"
              "本目录将输入的 cardcap 1.0 升级为 1.1，仅改变 `format_version`。"
              "手、相机、尺度、packet、事件及诊断字段均逐值保留，未补帧、重建牌块或施加缩放。\n\n"
              f"输入包含 {len(_field(source, 'packets', '$'))} 个 packet；升级不会把表面跟踪草稿变成物理牌块。"
              "原文件保留在 `upgrade_report.json` 所记输入路径。"
              "`capture.cardcap.json` 供支持 1.1 的读取端使用，`readback.json` 记录完整回读。"
              "这是格式兼容结果，不是新的推理、相机标定、尺度验证或真实 packet 观测。\n\n"
              "1.1 的部分 packet 属性可显式为 `null`，字段本身仍必需；没有 pose 的 packet 帧不添加占位姿态。"
              "当前读取端构建及 UE 实际回读的结果由集成报告另行记录。\n\n"
              "重复校验：在 PythonPipeline 使用 `python -m cardcap.packet_contract verify <本目录>`。\n")
    with (output_dir / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(readme)
    return readback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validation = sub.add_parser("validate", help="Validate a 1.0, 1.1, 1.2 or 1.3 final capture without editing it")
    validation.add_argument("capture", type=Path)
    validation.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    upgrade = sub.add_parser("upgrade", help="Create a new 1.1 directory from a 1.0 capture")
    upgrade.add_argument("capture", type=Path)
    upgrade.add_argument("--output", type=Path, required=True)
    upgrade.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    verify = sub.add_parser("verify", help="Fully read back a version-only upgrade")
    verify.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_capture(read_json(args.capture), read_json(args.mapping))
            result["capture_sha256"] = sha256(args.capture)
        elif args.command == "upgrade":
            result = upgrade_to_1_1(args.capture, args.output, args.mapping)
        else:
            result = verify_upgrade(args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"cardcap contract: {error}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
