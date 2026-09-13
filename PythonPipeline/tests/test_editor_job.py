"""Controlled job fixtures, not real video inference or Unreal validation.

The final test encodes/decodes synthetic color frames with the actual OpenCV
writer; all reconstructed geometry and commandlet reports remain test fixtures.
"""
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from cardcap import editor_job as job


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Process:
    pid = 424242  # Never sent to a real OS process API by these tests.

    def __init__(self, returncode=0, running=False):
        self.returncode = returncode
        self.running = running

    def poll(self):
        return None if self.running else self.returncode


class Fixture:
    def __init__(self, directory, *, mode=None, render=True):
        self.base = directory
        self.root = directory / "作业 空格" / "Take_Fixture"
        self.plugin = directory / "插件 空格"
        self.root.mkdir(parents=True)
        self.plugin.mkdir()
        self.mode = mode
        self.calls, self.terminated, self.sleeps = [], [], []
        self.mapping = {"schema_version": 1, "root_bone": "root", "mano_joint_order": [f"joint{i}" for i in range(16)],
                        "mano_parents": [-1] + [0] * 15, "landmark_indices": list(range(16)),
                        "hands": {side: {"bone_names": [f"{side}_bone{i}" for i in range(16)],
                                         "rest_offset_camera_m": [0, 0, 0]} for side in ("left", "right")}}
        mapping = directory / "自定义 骨映射.json"
        write_json(mapping, self.mapping)
        self.video = directory / "原片 空格.mp4"
        self.video.write_bytes(b"test fixture, not a real video")
        project = directory / "测试 工程.uproject"
        project.write_text("{}")
        editor = directory / "UnrealEditor-Cmd.exe"
        editor.write_bytes(b"fake: never executable")
        self.request = {"schema_version": 1, "job_id": "Take_Fixture", "video_path": str(self.video),
                        "plugin_dir": str(self.plugin), "project_file": str(project), "editor_executable": str(editor),
                        "output_dir": str(self.root), "bone_mapping_path": str(mapping), "render_preview": render}
        write_json(self.root / "request.json", self.request)

    def capture(self):
        hands = []
        for side in ("left", "right"):
            frames = [{"frame": index, "confidence": 0.2, "global_trans_cm": [100, 0, 0],
                       "global_rot_quat": [0, 0, 0, 1], "joint_positions_cm": [[100, 0, 0]] * 21,
                       "bone_rotations": {name: [0, 0, 0, 1] for name in self.mapping["hands"][side]["bone_names"][1:]},
                       "occluded_joints": [], "diagnostics": {"sample_kind": "observed"}} for index in range(3)]
            hands.append({"side": side, "mano_shape": [0] * 10, "frames": frames})
        return {"format_version": "1.0", "meta": {"source_video": str(self.video), "source_video_sha256": job.sha256(self.video),
                "frame_count": 3, "fps": 30.0, "resolution": [64, 48], "processed_at": "synthetic-fixture",
                "pipeline_version": "synthetic-fixture-not-inference"},
                "camera": {"intrinsics": {"fx": 64, "fy": 64, "cx": 32, "cy": 24}, "distortion": [0] * 5,
                           "calibrated": False, "intrinsics_source": "fixture-only"},
                "scale": {"meters_per_unit": 1.0, "anchor_method": "hand_prior", "confidence": 0.2,
                          "anchor_frames": [], "scale_confidence": "low"}, "hands": hands, "packets": [],
                "events": {"splits": [], "merges": [], "releases": []},
                "quality": {"mean_hand_confidence": 0.2, "low_confidence_ranges": [[0, 2]], "mean_reprojection_error_px": None,
                            "warnings": ["Synthetic orchestration fixture; not inference"]}}

    def reconstruct(self):
        output = self.root / "pipeline"
        animation = output / "animation"
        write_json(output / "report.json", {"status": "failed" if self.mode == "pipeline_false_success" else "completed"})
        write_json(output / "hand_observations.json", {"test_fixture": True})
        write_json(animation / "capture.cardcap.json", self.capture())
        research = job.expected_research_dir(self.plugin, animation)
        research.mkdir(parents=True)
        mesh = research / "MANO_ResearchHands_UE_Top4Preview.glb"
        mesh.write_bytes(b"fixture GLB marker, no geometry or UE import")
        record = {"weight_mode": "top4_renormalized_UE_preview", "path": str(mesh), "sha256": job.sha256(mesh), "bone_count": 33}
        mapping_hash = job.sha256(self.request["bone_mapping_path"])
        manifest_path = research / "MANO_ResearchHands_Manifest.json"
        manifest = {"outputs": [record], "shared_betas": [0] * 10, "bone_mapping_path": self.request["bone_mapping_path"],
                    "bone_mapping_sha256": "wrong" if self.mode == "wrong_mapping" else mapping_hash,
                    "coordinates": {"meters_per_mano_unit": 1.0}}
        write_json(manifest_path, manifest)
        validation = {"status": "passed", "capture_path": str(animation / "capture.cardcap.json"),
                      "capture_sha256": job.sha256(animation / "capture.cardcap.json"),
                      "source_observations_path": str(output / "hand_observations.json"),
                      "source_observations_sha256": job.sha256(output / "hand_observations.json"),
                      "source_video_sha256": job.sha256(self.video), "bone_mapping_sha256": mapping_hash,
                      "research_hands_manifest": str(manifest_path), "research_glb_outputs": [record], "shared_mano_shape": [0] * 10}
        if self.mode == "old_mesh":
            old = self.plugin / "Saved/Validation/old_take/manifest.json"
            write_json(old, manifest)
            validation["research_hands_manifest"] = str(old)
        write_json(animation / "validation.json", validation)

    def rendered_frames(self, real_images=False):
        render = self.root / "UE_Frames"
        frames = []
        for index in range(3):
            filename = f"frame_{index:06d}.png"
            for relative in (filename, "overview/" + filename):
                path = render / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if real_images:
                    import cv2
                    import numpy as np
                    image = np.full((48, 64, 3), [40 + index * 10, 100, 180], np.uint8)
                    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
                else:
                    path.write_bytes(b"fixture PNG marker")
            frames.append({"frame": index, "file": filename, "overview_file": "overview/" + filename,
                           "nonblack_pixels_above_8": 100, "overview_nonblack_pixels_above_8": 100})
        base = "/Game/CardistryCapture/Generated/Take_Fixture/Review/HandsReview"
        write_json(render / "sequence_render.json", {"success": self.mode != "render_false_success", "captured_frames": 3,
                   "requested_frames": 3, "frames": frames, "width": 64, "height": 48, "fps": 30,
                   "map": base + "_Map.HandsReview_Map", "sequence": base + "_Sequence.HandsReview_Sequence"})

    def preview_result(self):
        output = self.root / "Preview"
        output.mkdir()
        video = output / "review.mp4"
        video.write_bytes(b"fixture MP4 marker, not a video")
        write_json(output / "preview.json", {"status": "passed", "decoded_frames": 2 if self.mode == "preview_short" else 3,
                   "fps": 30, "path": str(video), "sha256": job.sha256(video),
                   "capture_sha256": job.sha256(self.root / "pipeline/animation/capture.cardcap.json"),
                   "source_video_sha256": job.sha256(self.video),
                   "sequence_render_sha256": job.sha256(self.root / "UE_Frames/sequence_render.json")})

    def popen(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.mode == "exit_failure":
            return Process(9)
        if self.mode in ("cancel_running", "cancel_exit_zero"):
            (self.root / "cancel.request").touch()
            return Process(0, running=self.mode == "cancel_running")
        values = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in args if "=" in arg}
        if "--export-animation" in args:
            self.reconstruct()
        elif values.get("-run") == "CardCapImportHands":
            base = values["-Destination"] + "/MANO_ResearchHands_UE_Top4Preview"
            report = {"status": "passed", "source": values["-MeshSource"],
                      "objects": [{"saved_on_disk": True}], "skeletal_meshes": [{"bone_count": 33,
                          "mesh_path": base + ".MANO_ResearchHands_UE_Top4Preview", "skeleton_path": base + "_Skeleton.Skeleton"}]}
            self.mesh_path = report["skeletal_meshes"][0]["mesh_path"]
            write_json(Path(values["-Evidence"]), report)
        elif values.get("-run") == "CardCapBake":
            reload = "-ValidateOnly" in args
            animation = values["-Animation"] if reload else values["-Animation"] + ".Hands"
            write_json(Path(values["-Evidence"]), {"status": "passed", "capture_file": values["-Capture"],
                       "mesh_path": self.mesh_path, "animation_path": animation, "source_frames": 3, "animation_keys": 4,
                       "fps": 30.0, "compressed_codec_and_structure_present": True, "force_raw_console_value": 0,
                       "reloaded_in_fresh_process": False if self.mode == "reload_not_fresh" else reload})
            if not reload:
                self.rendered_frames()
        elif "--preview-request" in args:
            self.preview_result()
        else:
            raise AssertionError(args)
        return Process()

    def worker(self):
        return job.EditorJob(self.root / "request.json", popen=self.popen, sleep=self.sleeps.append,
                             environment_check=lambda request: {"test_fixture": True, "no_real_model_or_UE_run": True},
                             terminate=self.terminated.append)


class EditorJobTests(unittest.TestCase):
    def test_generated_assets_belong_to_project_and_job_not_plugin_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            worker = fixture.worker()
            self.assertEqual(worker.asset_root, "/Game/CardistryCapture/Generated/Take_Fixture")
            self.assertEqual(worker.asset_directory, Path(fixture.request["project_file"]).parent /
                             "Content/CardistryCapture/Generated/Take_Fixture")
            self.assertEqual(worker.research_dir, worker.animation_dir / "hand_mesh")
            self.assertTrue(worker.research_dir.is_relative_to(fixture.root))
            self.assertFalse(worker.asset_directory.is_relative_to(fixture.plugin))
            self.assertFalse(worker.research_dir.is_relative_to(fixture.plugin))

    def test_success_unicode_paths_mapping_and_hidden_no_shell_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            self.assertEqual(fixture.worker().run(), 0)
            status = job.read_json(fixture.root / "status.json")
            self.assertEqual((status["status"], status["stage"], status["frames_completed"]), ("succeeded", "complete", 3))
            self.assertEqual(status["result"]["low_confidence_ranges"], [[0, 2]])
            self.assertEqual(status["result"]["scale_confidence"], "low")
            self.assertIsNotNone(status["result"]["preview_video"])
            self.assertEqual(len(fixture.calls), 5)
            reconstruct = fixture.calls[0][0]
            self.assertEqual(reconstruct[reconstruct.index("--output") + 1], str(fixture.root / "pipeline"))
            self.assertEqual(reconstruct[reconstruct.index("--bone-mapping") + 1], fixture.request["bone_mapping_path"])
            for args, kwargs in fixture.calls:
                self.assertFalse(kwargs["shell"])
                self.assertNotIn("-ReplaceExisting", args)
                if job.os.name == "nt":
                    self.assertTrue(kwargs["creationflags"] & job.subprocess.CREATE_NO_WINDOW)
            self.assertIn("-Mapping=" + fixture.request["bone_mapping_path"], fixture.calls[2][0])
            self.assertIn("-Mapping=" + fixture.request["bone_mapping_path"], fixture.calls[3][0])
            before = (fixture.root / "status.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "任务目录已使用"):
                fixture.worker().run()
            self.assertEqual(before, (fixture.root / "status.json").read_bytes())

    def test_failure_and_false_success_never_launch_following_stage(self):
        for mode, calls in (("exit_failure", 1), ("pipeline_false_success", 1), ("old_mesh", 1), ("wrong_mapping", 1),
                            ("render_false_success", 3), ("reload_not_fresh", 4), ("preview_short", 5)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), mode=mode)
                self.assertEqual(fixture.worker().run(), 1)
                status = job.read_json(fixture.root / "status.json")
                self.assertEqual(status["status"], "failed")
                self.assertIsNotNone(status["error"]["details"])
                self.assertEqual(len(fixture.calls), calls)
                self.assertFalse((fixture.root / "job_result.json").exists())

    def test_reconstruction_failure_preserves_specific_reason_and_log_path(self):
        raw_identity_error = ("ValueError: Input has unknown or duplicate hand side in a frame; "
                              "resolve identity before animation export")
        unknown_error = "RuntimeError: fixture model stopped; --status succeeded is diagnostic text"
        for raw_error, expected in ((raw_identity_error, "部分帧的左右手身份不明确或重复，无法生成动画。"),
                                    ("ValueError: Input has unknown hand side; resolve identity before animation export",
                                     "输入中有无法识别的左右手标签，无法生成动画。"),
                                    ("ValueError: Input has duplicate hand side without resolved identity; resolve identity before animation export",
                                     "部分帧的左右手身份仍有冲突，无法可靠生成动画。"),
                                    ("ValueError: Both hands need at least one usable real observation after identity conflict filtering",
                                     "未找到两只手各自可用的观测，无法生成双手动画。"),
                                    (unknown_error, unknown_error)):
            with self.subTest(error=raw_error), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                worker = fixture.worker()

                def fail_with_report(args, **kwargs):
                    fixture.calls.append((args, kwargs))
                    write_json(worker.output / "report.json", {"status": "failed", "error": raw_error,
                               "command": "ignored report data", "output_dir": "ignored report data"})
                    return Process(1)

                worker.popen = fail_with_report
                self.assertEqual(worker.run(), 1)
                status = job.read_json(fixture.root / "status.json")
                self.assertEqual(status["status"], "failed")
                self.assertIn(expected, status["error"]["message"])
                self.assertIn(str(fixture.root / "reconstruct.log"), status["error"]["message"])
                self.assertEqual(status["error"]["log_path"], str(fixture.root / "job.log"))
                self.assertIn(raw_error, (fixture.root / "job.log").read_text(encoding="utf-8"))
                self.assertEqual(len(fixture.calls), 1)
                self.assertFalse((fixture.root / "job_result.json").exists())

    def test_reconstruction_failure_fallback_and_cancel_do_not_become_success(self):
        cases = (("{partial", None, False, "退出码 7"),
                 ({"status": "failed", "error": {"command": "not error text"}}, None, False, "退出码 7"),
                 ({"status": "completed", "error": "ignore completed report"},
                  {"status": "failed", "error": "OSError: retained progress failure"}, False,
                  "OSError: retained progress failure"),
                 ({"status": "failed", "error": "must not override cancellation"}, None, True, None))
        for report, progress, cancel, expected in cases:
            with self.subTest(report=report, cancel=cancel), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                worker = fixture.worker()

                def fail_with_report(args, **kwargs):
                    fixture.calls.append((args, kwargs))
                    worker.output.mkdir()
                    if isinstance(report, str):
                        (worker.output / "report.json").write_text(report, encoding="utf-8")
                    else:
                        write_json(worker.output / "report.json", report)
                    if progress is not None:
                        write_json(worker.output / "progress.json", progress)
                    if cancel:
                        worker.cancel_file.touch()
                    return Process(7)

                worker.popen = fail_with_report
                self.assertEqual(worker.run(), 2 if cancel else 1)
                status = job.read_json(fixture.root / "status.json")
                self.assertEqual(status["status"], "cancelled" if cancel else "failed")
                if cancel:
                    self.assertIsNone(status["error"])
                else:
                    self.assertIn(expected, status["error"]["message"])
                    self.assertNotIn("ignore completed report", status["error"]["message"])
                self.assertEqual(len(fixture.calls), 1)
                self.assertFalse((fixture.root / "job_result.json").exists())

    def test_cancel_preflight_running_and_even_exit_zero(self):
        for mode in ("pre_cancel", "cancel_running", "cancel_exit_zero"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), mode=mode)
                if mode == "pre_cancel":
                    (fixture.root / "cancel.request").touch()
                self.assertEqual(fixture.worker().run(), 2)
                self.assertEqual(job.read_json(fixture.root / "status.json")["status"], "cancelled")
                self.assertEqual(len(fixture.calls), 0 if mode == "pre_cancel" else 1)
                self.assertEqual(len(fixture.terminated), 0 if mode == "pre_cancel" else 1)

    def test_existing_ue_or_research_assets_rejected_and_optional_mp4_only(self):
        for existing in ("ue", "research"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                worker = fixture.worker()
                directory = worker.asset_directory if existing == "ue" else worker.research_dir
                directory.mkdir(parents=True)
                marker = directory / "old.bin"
                marker.write_bytes(b"keep")
                if existing == "research":
                    with self.assertRaisesRegex(ValueError, "任务目录已使用"):
                        worker.run()
                else:
                    self.assertEqual(worker.run(), 1)
                self.assertEqual(marker.read_bytes(), b"keep")
                self.assertFalse(fixture.calls)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), render=False)
            self.assertEqual(fixture.worker().run(), 0)
            result = job.read_json(fixture.root / "status.json")["result"]
            self.assertIsNone(result["preview_video"])
            self.assertIsNotNone(result["map_asset"])
            self.assertEqual(len(fixture.calls), 4)

    def test_actual_codec_synthetic_preview_unicode_path_complete_readback(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            source = cv2.VideoWriter(str(fixture.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(source.isOpened())
            for index in range(3):
                source.write(np.full((48, 64, 3), [60, 100 + index * 10, 140], np.uint8))
            source.release()
            fixture.reconstruct()
            fixture.rendered_frames(real_images=True)
            job.build_preview(fixture.root / "request.json")
            report = job.read_json(fixture.root / "Preview/preview.json")
            self.assertEqual(report["decoded_frames"], 3)
            self.assertEqual(report["resolution"], [3840, 720])
            self.assertFalse(report["geometry_changed"])


if __name__ == "__main__":
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EditorJobTests))
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "seconds": time.perf_counter() - started,
              "scope": "Controlled fake subprocess/report fixtures; actual OpenCV roundtrip uses synthetic color frames only. No model or Unreal process run.",
              "implementation_sha256": job.sha256(Path(job.__file__)), "tests_sha256": job.sha256(Path(__file__))}
    target = Path(__file__).resolve().parents[1] / "evidence/editor-job"
    target.mkdir(exist_ok=True)
    write_json(target / "tests.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
