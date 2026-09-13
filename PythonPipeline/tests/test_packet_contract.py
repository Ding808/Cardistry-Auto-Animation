"""Synthetic interchange contract checks; no fixture is a real packet estimate."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardcap.packet_contract import (ContractError, DEFAULT_MAPPING, loads_strict, read_json,
                                     sha256, upgrade_to_1_1, validate_capture, verify_upgrade)


MAPPING = read_json(DEFAULT_MAPPING)


def fixture(version="1.0"):
    frame = {"frame": 0, "confidence": .5, "global_trans_cm": [0, 0, 1],
             "global_rot_quat": [0, 0, 0, 1], "joint_positions_cm": [[0, 0, 1]] * 21,
             "bone_rotations": {name: [0, 0, 0, 1] for name in MAPPING["hands"]["left"]["bone_names"][1:]},
             "occluded_joints": []}
    return {"format_version": version,
            "meta": {"source_video": "SYNTHETIC_TEST_ONLY", "fps": 30, "frame_count": 3,
                     "resolution": [640, 480], "processed_at": "synthetic", "pipeline_version": "unit"},
            "camera": {"intrinsics": {"fx": 600, "fy": 600, "cx": 320, "cy": 240},
                       "distortion": [0, 0, 0, 0, 0], "calibrated": False},
            "scale": {"meters_per_unit": 1, "anchor_method": "hand_prior", "confidence": .1, "anchor_frames": []},
            "hands": [{"side": "left", "mano_shape": [0] * 10, "frames": [frame]}],
            "packets": [], "events": {"splits": [], "merges": [], "releases": []},
            "quality": {"mean_hand_confidence": .5, "low_confidence_ranges": [[1, 2]],
                        "mean_reprojection_error_px": None, "warnings": ["SYNTHETIC TEST ONLY"]}}


def packet(name="unit", unknown=False):
    return {"id": name, "birth_frame": 0, "death_frame": 2,
            "card_count_estimate": None if unknown else 1,
            "card_count_confidence": None if unknown else .1,
            "dimensions_cm": [6.35, 8.89, None if unknown else .03],
            "frames": [{"frame": 0, "confidence": .1, "position_cm": [0, 0, 0],
                        "rotation_quat": [0, 0, 0, 1],
                        "linear_velocity_cm_s": None if unknown else [0, 0, 0],
                        "angular_velocity_rad_s": None if unknown else [0, 0, 0],
                        "contact_state": "UNKNOWN" if unknown else "FREE",
                        "contact_bones": None if unknown else [],
                        "reprojection_error_px": None if unknown else 1.2}]}


def fixture12():
    data = fixture("1.2")
    data["camera"]["distortion"] = None
    data["scale"] = {"meters_per_unit": None, "anchor_method": None, "confidence": None, "anchor_frames": []}
    data["provenance"] = {"camera_intrinsics": "inferred", "camera_distortion": "unobservable",
                          "metric_scale": "unobservable", "hand_geometry": "inferred",
                          "coordinate_units": "conditional_ue_units",
                          "assumptions": ["Synthetic contract test: arbitrary display units and conditional pinhole K; not footage"]}
    data["validity"] = {"camera_intrinsics": True, "camera_distortion": False, "metric_scale": False}
    frame = data["hands"][0]["frames"][0]
    frame["sample_kind"] = "detected_model_observation"
    frame["validity"] = {"pose": True}
    return data


class PacketContractTests(unittest.TestCase):
    def rejected(self, data, path=None):
        with self.assertRaises(ContractError) as caught:
            validate_capture(data, MAPPING)
        if path:
            self.assertIn(path, str(caught.exception))

    def test_numeric_1_0_remains_valid_in_both_versions_and_sparse_unchanged(self):
        for version in ("1.0", "1.1"):
            data = fixture(version)
            data["packets"] = [packet()]
            data["packets"][0]["frames"].append(dict(packet()["frames"][0], frame=2))
            original = deepcopy(data)
            result = validate_capture(data, MAPPING)
            self.assertEqual(result["packet_sample_counts"], {"unit": 2})
            self.assertEqual(data, original)  # missing packet frame 1 remains missing
            for state in ("free", "FrEe", "gripped", "Resting", "sliding"):
                data["packets"][0]["frames"][0]["contact_state"] = state
                validate_capture(data, MAPPING)
            data["packets"][0]["frames"] = []
            self.assertEqual(validate_capture(data, MAPPING)["packet_sample_counts"], {"unit": 0})

    def test_each_new_nullable_option_only_in_1_1(self):
        updates = [("packet", "card_count_estimate", None),
                   ("packet", "dimensions_cm", [6.35, 8.89, None]),
                   ("frame", "linear_velocity_cm_s", None), ("frame", "angular_velocity_rad_s", None),
                   ("frame", "reprojection_error_px", None), ("frame", "contact_bones", None),
                   ("frame", "contact_state", "UNKNOWN")]
        for location, key, value in updates:
            with self.subTest(key=key):
                data = fixture("1.1"); data["packets"] = [packet()]
                target = data["packets"][0] if location == "packet" else data["packets"][0]["frames"][0]
                target[key] = value
                if key == "card_count_estimate": target["card_count_confidence"] = None
                validate_capture(data, MAPPING)
                data["format_version"] = "1.0"
                self.rejected(data)
        data = fixture("1.1"); data["packets"] = [packet(unknown=True)]
        validate_capture(data, MAPPING)
        # State and bone evidence are independent; the parser infers neither.
        data["packets"][0]["frames"][0]["contact_bones"] = ["hand_l"]
        validate_capture(data, MAPPING)

    def test_every_packet_and_frame_key_is_required_even_if_nullable(self):
        for version in ("1.0", "1.1"):
            base = fixture(version); base["packets"] = [packet(unknown=version == "1.1")]
            for key in base["packets"][0]:
                with self.subTest(version=version, packet_key=key):
                    data = deepcopy(base); del data["packets"][0][key]
                    self.rejected(data, f".{key}")
            for key in base["packets"][0]["frames"][0]:
                with self.subTest(version=version, frame_key=key):
                    data = deepcopy(base); del data["packets"][0]["frames"][0][key]
                    self.rejected(data, f".{key}")

    def test_invalid_count_dimensions_and_pose_numbers_are_rejected(self):
        packet_cases = [("card_count_estimate", None), ("card_count_confidence", None),
                        ("card_count_estimate", 0), ("card_count_estimate", 1.5),
                        ("card_count_estimate", True), ("card_count_confidence", 1.1),
                        ("dimensions_cm", [6.35, 8.89]), ("dimensions_cm", None),
                        ("dimensions_cm", [None, 8.89, None]), ("dimensions_cm", [6.35, 0, None]),
                        ("dimensions_cm", [6.35, 8.89, 0]), ("dimensions_cm", [6.35, 8.89, float("inf")]),
                        ("birth_frame", -1), ("death_frame", 3)]
        frame_cases = [("linear_velocity_cm_s", [0, None, 0]), ("angular_velocity_rad_s", [0, 0]),
                       ("linear_velocity_cm_s", [0, True, 0]), ("position_cm", None),
                       ("rotation_quat", None), ("rotation_quat", [0, 0, 0, .999]),
                       ("position_cm", [0, float("nan"), 0]), ("frame", 3), ("confidence", None),
                       ("confidence", "0.5"), ("confidence", -1), ("reprojection_error_px", -1),
                       ("contact_state", "unknown"), ("contact_bones", ["root"]),
                       ("contact_bones", ["hand_l", "hand_l"]), ("contact_bones", ["hand_l", "Hand_L"])]
        for location, cases in (("packet", packet_cases), ("frame", frame_cases)):
            for key, value in cases:
                with self.subTest(location=location, key=key, value=value):
                    data = fixture("1.1"); data["packets"] = [packet()]
                    target = data["packets"][0] if location == "packet" else data["packets"][0]["frames"][0]
                    target[key] = value
                    self.rejected(data)

    def test_event_references_duplicates_and_release_velocity_stay_strict(self):
        for version in ("1.0", "1.1"):
            base = fixture(version); base["packets"] = [packet(n) for n in ("a", "b", "c")]
            base["events"] = {"splits": [{"frame": 1, "source": "a", "results": ["b", "c"]}],
                              "merges": [{"frame": 2, "sources": ["a", "b"], "result": "c"}],
                              "releases": [{"frame": 1, "packet": "a", "release_velocity_cm_s": [1, 2, 3]}]}
            validate_capture(base, MAPPING)
            for kind, key, values in (("splits", "results", [["b"], ["b", "b"], ["a", "b"], ["b", "missing"]]),
                                       ("merges", "sources", [["a"], ["a", "a"], ["a", "c"], ["a", "missing"]]),
                                       ("releases", "release_velocity_cm_s", [None, [None, 0, 0]])):
                for value in values:
                    data = deepcopy(base); data["events"][kind][0][key] = value
                    self.rejected(data)
            for kind in base["events"]:
                data = deepcopy(base); data["events"][kind].append(deepcopy(data["events"][kind][0]))
                self.rejected(data, "duplicate event")
                data = deepcopy(base); data["events"][kind][0]["frame"] = 3
                self.rejected(data)
            data = deepcopy(base); data["packets"].append(deepcopy(data["packets"][0]))
            self.rejected(data, "duplicate packet id")
            data = deepcopy(base); data["packets"][0]["frames"] *= 2
            self.rejected(data, "duplicate packet frame")

    def test_root_hand_camera_scale_quality_and_json_strictness(self):
        for version in ("2.0", "1", 1.1, None):
            data = fixture(); data["format_version"] = version
            self.rejected(data, "format_version")
        base = fixture()
        for key in base:
            data = deepcopy(base); del data[key]
            self.rejected(data, key)
        cases = [(["camera", "calibrated"], 1), (["camera", "intrinsics", "fx"], 0),
                 (["camera", "distortion"], [0, 0]), (["scale", "anchor_frames"], [0, 0]),
                 (["scale", "meters_per_unit"], 0), (["quality", "mean_reprojection_error_px"], -1),
                 (["quality", "low_confidence_ranges"], [[0, 1], [1, 2]]),
                 (["hands", 0, "mano_shape"], [0] * 9),
                 (["hands", 0, "frames", 0, "occluded_joints"], [21]),
                 (["hands", 0, "frames", 0, "bone_rotations"], {})]
        for path, value in cases:
            data = deepcopy(base); target = data
            for part in path[:-1]: target = target[part]
            target[path[-1]] = value
            self.rejected(data)
        for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"a":NaN}', '{"a":1e999}', '[]'):
            with self.assertRaises(ContractError): loads_strict(text)

    def test_legacy_ascii_case_matching_without_renaming_and_new_unknown_exact(self):
        for version in ("1.0", "1.1"):
            data = fixture(version)
            data["hands"][0]["side"] = "LEFT"
            data["hands"][0]["frames"][0]["bone_rotations"] = {
                name.upper(): value for name, value in data["hands"][0]["frames"][0]["bone_rotations"].items()}
            data["scale"]["anchor_method"] = "HAND_PRIOR"
            data["packets"] = [packet("Alpha"), packet("Beta"), packet("Gamma")]
            data["packets"][0]["frames"][0]["contact_bones"] = ["Hand_L", "INDEX_01_L"]
            data["events"]["splits"] = [{"frame": 1, "source": "ALPHA", "results": ["beta", "GAMMA"]}]
            data["META"] = data.pop("meta"); data["META"]["FPS"] = data["META"].pop("fps")
            data["FORMAT_VERSION"] = data.pop("format_version")
            original = deepcopy(data)
            self.assertEqual(validate_capture(data, MAPPING)["status"], "passed")
            self.assertEqual(data, original)
            duplicates = []
            bad = deepcopy(data); bad["packets"].append(packet("ALPHA")); duplicates.append(bad)
            bad = deepcopy(data); bad["hands"].append(deepcopy(bad["hands"][0])); bad["hands"][1]["side"] = "left"; duplicates.append(bad)
            bad = deepcopy(data); bad["events"]["splits"][0]["results"] = ["Beta", "BETA"]; duplicates.append(bad)
            bad = deepcopy(data); bad["events"]["splits"][0]["results"] = ["alpha", "Gamma"]; duplicates.append(bad)
            bad = deepcopy(data); bad["events"]["splits"].append({"frame": 1, "source": "alpha", "results": ["Beta", "Gamma"]}); duplicates.append(bad)
            bad = deepcopy(data); bad["meta"] = deepcopy(bad["META"]); duplicates.append(bad)
            for bad in duplicates:
                self.rejected(bad)
        for text in ('{"meta":{},"META":{}}', '{"a":{"fps":30,"FPS":30}}'):
            with self.assertRaisesRegex(ContractError, "duplicate object key"):
                loads_strict(text)
        # No Python Unicode upper/casefold expansion (long s -> S) is applied.
        data = fixture("1.1"); data["packets"] = [packet()]
        for state in ("unknown", "Unknown", "reſting"):
            data["packets"][0]["frames"][0]["contact_state"] = state
            self.rejected(data, "contact_state")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy_case.json"
            data = fixture(); data["FORMAT_VERSION"] = data.pop("format_version")
            data["PACKETS"] = data.pop("packets")
            source.write_text(json.dumps(data), encoding="utf-8")
            upgrade_to_1_1(source, Path(directory) / "out")
            upgraded = read_json(Path(directory) / "out/capture.cardcap.json")
            self.assertNotIn("format_version", upgraded)
            self.assertEqual(upgraded["FORMAT_VERSION"], "1.1")
            self.assertIn("PACKETS", upgraded)

    def test_upgrade_readback_preserves_all_values_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory); source = parent / "input.json"
            data = fixture(); data["meta"]["unrecognized_diagnostic"] = {"nested": [True, None, 1., -0., "中文"]}
            source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            source_hash = sha256(source); output = parent / "upgrade"
            result = upgrade_to_1_1(source, output)
            self.assertEqual(result["changed_json_paths"], ["$.format_version"])
            self.assertEqual(sha256(source), source_hash)
            upgraded = read_json(output / "capture.cardcap.json")
            upgraded["format_version"] = "1.0"
            self.assertEqual(upgraded, data)
            self.assertEqual(verify_upgrade(output)["status"], "passed")
            report_hash = sha256(output / "upgrade_report.json")
            with self.assertRaisesRegex(ContractError, "new or empty"):
                upgrade_to_1_1(source, output)
            self.assertEqual(sha256(output / "upgrade_report.json"), report_hash)
            # A plausible edited summary must not pass full readback.
            report = read_json(output / "upgrade_report.json")
            report["output_validation"]["packet_count"] = 1
            (output / "upgrade_report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "summary differs"):
                verify_upgrade(output)
            # Invalid input leaves no output artifacts, and 1.1 is not upgraded again.
            data["format_version"] = "1.1"
            source.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "input must be 1.0"):
                upgrade_to_1_1(source, parent / "invalid")
            self.assertFalse((parent / "invalid").exists())


class PacketContract12Tests(unittest.TestCase):
    def rejected(self, data, path):
        original = deepcopy(data)
        with self.assertRaises(ContractError) as caught:
            validate_capture(data, MAPPING)
        self.assertIn(path, str(caught.exception))
        self.assertEqual(data, original)

    def test_conditional_numeric_geometry_and_null_metric_claims_preserve_values(self):
        data = fixture12()
        original = deepcopy(data)
        result = validate_capture(data, MAPPING)
        self.assertEqual(result["format_version"], "1.2")
        self.assertEqual(result["hand_sample_counts"], {"left": 1})
        self.assertFalse(result["measurement_truth_or_physical_identity_validated"])
        self.assertEqual(data, original)
        self.assertIsNone(data["scale"]["meters_per_unit"])
        self.assertEqual(data["hands"][0]["frames"][0]["global_trans_cm"], [0, 0, 1])

    def test_known_scale_allows_unknown_confidence_without_fabricating_probability(self):
        for method, source in (("user_measured", "user_measured"), ("card_pnp", "calibrated")):
            data = fixture12()
            data["scale"].update(meters_per_unit=1., anchor_method=method, anchor_frames=[1])
            data["validity"]["metric_scale"] = True
            data["provenance"].update(metric_scale=source, coordinate_units="ue_centimeters")
            for confidence in (None, 0, .5, 1):
                data["scale"]["confidence"] = confidence
                original = deepcopy(data)
                validate_capture(data, MAPPING)
                self.assertEqual(data, original)
            for value in (True, -.1, 1.1, "unknown"):
                data["scale"]["confidence"] = value
                self.rejected(data, "$.scale.confidence")
        data = fixture12()
        data["scale"].update(meters_per_unit=1., anchor_method=None)
        data["validity"]["metric_scale"] = True
        data["provenance"].update(metric_scale="user_measured", coordinate_units="ue_centimeters")
        self.rejected(data, "$.scale")

    def test_null_intrinsics_and_distortion_are_read_without_inventing_camera(self):
        data = fixture12()
        data["camera"]["intrinsics"] = None
        data["validity"]["camera_intrinsics"] = False
        data["provenance"]["camera_intrinsics"] = "unobservable"
        original = deepcopy(data)
        validate_capture(data, MAPPING)
        self.assertEqual(data, original)
        data["camera"]["calibrated"] = True
        self.rejected(data, "$.camera.calibrated")
        data = fixture12()
        data["camera"]["distortion"] = [0] * 5
        data["validity"]["camera_distortion"] = True
        data["provenance"]["camera_distortion"] = "calibrated"
        validate_capture(data, MAPPING)
        self.assertEqual(data["camera"]["distortion"], [0] * 5)
        for key in ("fx", "fy", "cx", "cy"):
            bad = deepcopy(data); bad["camera"]["intrinsics"][key] = None
            self.rejected(bad, "$.camera.intrinsics." + key)

    def test_every_sample_source_is_distinct_and_missing_mask_is_not_a_pose(self):
        kinds = ("detected_model_observation", "tracked_roi_model_observation", "interpolated", "endpoint_hold", "missing")
        for kind in kinds:
            data = fixture12(); frame = data["hands"][0]["frames"][0]
            frame.update(sample_kind=kind, validity={"pose": kind != "missing"})
            original = deepcopy(data)
            validate_capture(data, MAPPING)
            self.assertEqual(data, original)
            frame["validity"]["pose"] = kind == "missing"
            self.rejected(data, ".validity.pose")
        # This minimal extension does not create a null/partial bone pose schema.
        # A missing slot may carry ignored numeric storage, but the UE baker rejects its false mask.
        for key in ("global_trans_cm", "global_rot_quat", "joint_positions_cm", "bone_rotations"):
            data = fixture12(); frame = data["hands"][0]["frames"][0]
            frame.update(sample_kind="missing", validity={"pose": False})
            frame[key] = None
            self.rejected(data, "." + key)
        for kind in ("detected", "Inferred", "DETECTED_MODEL_OBSERVATION", None, 0):
            data = fixture12(); data["hands"][0]["frames"][0]["sample_kind"] = kind
            self.rejected(data, ".sample_kind")

    def test_packet_dimensions_and_release_velocity_nulls_remain_distinct_from_zeros(self):
        for dimensions in (None, [None, None, None], [None, 8.89, None], [6.35, None, .03], [6.35, 8.89, .03]):
            data = fixture12(); data["packets"] = [packet(unknown=True)]
            data["packets"][0]["dimensions_cm"] = dimensions
            for velocity in (None, [0, 0, 0], [1, 2, 3]):
                data["events"]["releases"] = [{"frame": 1, "packet": "unit", "release_velocity_cm_s": velocity}]
                original = deepcopy(data)
                validate_capture(data, MAPPING)
                self.assertEqual(data, original)
        for bad_dimensions in ([None, None], [None, 0, None], [False, None, None], "unknown"):
            data = fixture12(); data["packets"] = [packet(unknown=True)]
            data["packets"][0]["dimensions_cm"] = bad_dimensions
            self.rejected(data, ".dimensions_cm")
        for bad_velocity in ([None, 0, 0], [0, 0], [True, 0, 0], "unknown"):
            data = fixture12(); data["packets"] = [packet(unknown=True)]
            data["events"]["releases"] = [{"frame": 1, "packet": "unit", "release_velocity_cm_s": bad_velocity}]
            self.rejected(data, ".release_velocity_cm_s")

    def test_claims_masks_units_and_numeric_booleans_cannot_contradict(self):
        for key in ("camera_intrinsics", "camera_distortion", "metric_scale"):
            for value in (not fixture12()["validity"][key], 0, None):
                data = fixture12(); data["validity"][key] = value
                self.rejected(data, "$.validity." + key)
            data = fixture12()
            data["provenance"][key] = "unobservable" if key == "camera_intrinsics" else "inferred"
            self.rejected(data, "$.provenance." + key)
        for key in ("camera_intrinsics", "camera_distortion", "metric_scale", "hand_geometry"):
            for value in ("Inferred", "measured", 0, None):
                data = fixture12(); data["provenance"][key] = value
                self.rejected(data, "$.provenance." + key)
        for value in ("ue_centimeters", "centimeters", "CONDITIONAL_UE_UNITS"):
            data = fixture12(); data["provenance"]["coordinate_units"] = value
            self.rejected(data, "$.provenance.coordinate_units")
        for value in ([], [""], [None], "model"):
            data = fixture12(); data["provenance"]["assumptions"] = value
            self.rejected(data, "$.provenance.assumptions")
        data = fixture12(); data["camera"]["calibrated"] = True
        self.rejected(data, "$.camera.calibrated")
        for key, value in (("confidence", 0), ("anchor_method", "hand_prior"), ("anchor_frames", [0])):
            data = fixture12(); data["scale"][key] = value
            self.rejected(data, "$.scale")

    def test_nullable_fields_provenance_and_masks_are_still_required(self):
        base = fixture12()
        for obj in ("camera", "scale", "provenance", "validity"):
            for key in base[obj]:
                data = deepcopy(base); del data[obj][key]
                self.rejected(data, f"$.{obj}.{key}")
        for key in ("provenance", "validity"):
            data = deepcopy(base); del data[key]
            self.rejected(data, "$." + key)
        for key in ("sample_kind", "validity"):
            data = deepcopy(base); del data["hands"][0]["frames"][0][key]
            self.rejected(data, "." + key)
        data = deepcopy(base); del data["hands"][0]["frames"][0]["validity"]["pose"]
        self.rejected(data, ".validity.pose")

    def test_old_versions_retain_every_new_null_boundary_and_require_no_new_fields(self):
        for version in ("1.0", "1.1"):
            validate_capture(fixture(version), MAPPING)
            for obj, keys in (("camera", ("intrinsics", "distortion")), ("scale", ("meters_per_unit", "anchor_method", "confidence"))):
                for key in keys:
                    data = fixture(version); data[obj][key] = None
                    self.rejected(data, f"$.{obj}.{key}")
            for dimensions in (None, [None, 8.89, .03], [6.35, None, .03]):
                data = fixture(version); data["packets"] = [packet()]
                data["packets"][0]["dimensions_cm"] = dimensions
                self.rejected(data, ".dimensions_cm")
            data = fixture(version); data["packets"] = [packet()]
            data["events"]["releases"] = [{"frame": 1, "packet": "unit", "release_velocity_cm_s": None}]
            self.rejected(data, ".release_velocity_cm_s")
            data = fixture(version); data["scale"]["anchor_method"] = "user_measured"
            self.rejected(data, "$.scale.anchor_method")


if __name__ == "__main__":
    unittest.main()
