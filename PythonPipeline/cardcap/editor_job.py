"""Run an isolated editor job using existing reconstruction and UE commandlets.

No installation, model downloads, packet inference, or geometry adjustment occurs
here. A successful job means saved/reloaded hand assets and actual rendered frames;
it does not mean accurate motion capture or verified physical scale. Request and
status schema_version=1. render_preview=False skips MP4 only; the Review sequence,
map and both rendered PNG views are still produced by the current UE commandlet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from cardcap.atomic_file import replace_with_windows_sharing_retry

PIPELINE = Path(__file__).resolve().parents[1]
MESSAGES = {"preflight": "Checking the local environment and input", "reconstruct": "Reconstructing hand motion",
            "import": "Importing hand models for this job", "bake": "Generating animation and both hand views",
            "verify": "Verifying animation in a fresh process", "preview": "Creating the source and animation comparison video",
            "complete": "Processing complete. Please review low-confidence sections."}
BASE_PROGRESS = {"preflight": 0.0, "reconstruct": 0.05, "import": 0.52,
                 "bake": 0.60, "verify": 0.88, "preview": 0.93, "complete": 1.0}


class Cancelled(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_with_windows_sharing_retry(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def check_record(path, digest):
    require(Path(path).is_file() and sha256(path) == digest, f"File missing or checksum mismatch: {path}")


def expected_research_dir(plugin, animation_dir):
    """Keep generated mesh inputs with this animation, never in the installed plugin."""
    return Path(animation_dir) / "hand_mesh"


def local_display_review(root, capture, render, *, check_cancel=None):
    """Measure raw local PNGs independently of optional MP4 creation.

    The same display-only check guards saved Review assets and encoded previews.
    No joint, camera or mesh value is changed to satisfy its thresholds.
    """
    if capture.get("provenance", {}).get("coordinate_frame") != "per_hand_wrist_local":
        return None
    import cv2
    import numpy as np
    from cardcap.geometry_checks import assess_display_samples
    require(capture["camera"]["intrinsics"] is None and render.get("view_mode") == "per_hand_local",
            "Unknown camera parameters require separate local hand previews; a shared-space view cannot be reused")
    root = Path(root)
    samples = {"left": [], "right": []}
    png_hashes = {}
    for index in range(capture["meta"]["frame_count"]):
        if check_cancel is not None:
            check_cancel()
        for side, relative in (("left", f"frame_{index:06d}.png"),
                               ("right", f"overview/frame_{index:06d}.png")):
            payload = (root / "UE_Frames" / relative).read_bytes()
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            require(image is not None and list(image.shape[1::-1]) == [render["width"], render["height"]],
                    "An Unreal PNG could not be decoded or has unexpected dimensions")
            # Raw single-hand images have black backgrounds and no captions.
            ys, xs = np.nonzero(np.max(image, axis=2) > 32)
            ratio = min(1280 / image.shape[1], 576 / image.shape[0])
            edge = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) * ratio if len(xs) else 0.
            samples[side].append({"frame": index, "long_edge_px": edge,
                                  "foreground_pixels": float(len(xs)) * ratio * ratio, "assessable": True})
            png_hashes[relative] = hashlib.sha256(payload).hexdigest()
    report = assess_display_samples(samples)
    report["pixel_measurement"] = "max BGR channel >32 on raw single-hand PNG, scaled to final 1280x576 image area; black background, no captions"
    report["capture_sha256"] = sha256(root / "pipeline/animation/capture.cardcap.json")
    report["sequence_render_sha256"] = sha256(root / "UE_Frames/sequence_render.json")
    report["raw_png_sha256"] = png_hashes
    return report


def read_request(path):
    path = Path(path).resolve()
    request = read_json(path)
    require(type(request.get("schema_version")) is int and request["schema_version"] == 1, "Unsupported job request version")
    require(isinstance(request.get("job_id"), str) and re.fullmatch(r"[A-Za-z0-9_]{1,100}", request["job_id"]),
            "job_id must contain only letters, digits and underscores, with at most 100 characters")
    for key in ("video_path", "plugin_dir", "project_file", "editor_executable", "output_dir"):
        require(isinstance(request.get(key), str) and request[key].strip(), f"Missing path: {key}")
        require(Path(request[key]).is_absolute(), f"An absolute path is required: {key}")
        request[key] = str(Path(request[key]).resolve())
    require(same_path(request["output_dir"], path.parent) and path.name == "request.json", "request.json must be in this job's output_dir root")
    for key, default in (("allow_blurry", False), ("render_preview", True)):
        require(type(request.get(key, default)) is bool, f"{key} must be a boolean")
        request.setdefault(key, default)
    mapping = request.get("bone_mapping_path") or str(Path(request["plugin_dir"]) / "Config/BoneMapping_UE5Mannequin.json")
    require(isinstance(mapping, str) and Path(mapping).is_absolute(), "The bone mapping must be an absolute file path")
    request["bone_mapping_path"] = str(Path(mapping).resolve())
    return request


def preflight_environment(request):
    """Read-only availability; does not initialize a model or access the network."""
    plugin = Path(request["plugin_dir"])
    require(same_path(plugin / "PythonPipeline", PIPELINE), "The requested plugin directory does not match the running worker")
    require(sys.version_info[:2] == (3, 10) and same_path(sys.prefix, PIPELINE / ".venv"),
            "Use Python 3.10 from the plugin's PythonPipeline/.venv environment")
    for key in ("video_path", "project_file", "editor_executable", "bone_mapping_path"):
        require(Path(request[key]).is_file(), f"Local file not found: {request[key]}")
    require(Path(request["editor_executable"]).name.lower() == "unrealeditor-cmd.exe", "The worker requires UnrealEditor-Cmd.exe")
    require(Path(request["project_file"]).suffix.lower() == ".uproject", "The project file must be a .uproject file")
    from cardcap.hand.wilor_impl import inspect_wilor_availability
    from setup_solve import EXPECTED, installed_record
    availability = inspect_wilor_availability(PIPELINE / "models")
    require(availability["ready"], "Local models or dependencies are unavailable (no automatic download): " + json.dumps(availability, ensure_ascii=False))
    for name in EXPECTED:
        require(installed_record(name) is not None, f"Missing local dependency {name}. Run Scripts/Setup.cmd in the plugin; the worker does not install dependencies")
    return {"models": availability, "network_or_installation_performed": False}


def hidden_options():
    options = {"shell": False}
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    return options


def terminate_tree(process):
    """Only a process created by this worker is ever passed here."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run([str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"),
                        "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, **hidden_options())
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class EditorJob:
    def __init__(self, request_path, *, popen=None, sleep=None, environment_check=None, terminate=None):
        self.request_path = Path(request_path).resolve()
        self.request = read_request(self.request_path)
        self.root = Path(self.request["output_dir"])
        self.plugin = Path(self.request["plugin_dir"])
        self.pipeline = self.plugin / "PythonPipeline"
        self.python = self.pipeline / ".venv/Scripts/python.exe"
        self.output = self.root / "pipeline"
        self.animation_dir = self.output / "animation"
        self.research_dir = expected_research_dir(self.plugin, self.animation_dir)
        self.asset_root = "/Game/CardistryCapture/Generated/" + self.request["job_id"]
        self.asset_directory = Path(self.request["project_file"]).parent / "Content/CardistryCapture/Generated" / self.request["job_id"]
        self.cancel_file = self.root / "cancel.request"
        self.render_dir = self.root / "UE_Frames"
        self.popen = popen or subprocess.Popen
        self.sleep = sleep or time.sleep
        self.environment_check = environment_check or preflight_environment
        self.terminate = terminate or terminate_tree
        self.started = time.monotonic()
        self.state = {"schema_version": 1, "job_id": self.request["job_id"], "status": "running",
                      "message_language": "en",
                      "stage": "preflight", "progress": 0.0, "message": MESSAGES["preflight"],
                      "elapsed_seconds": 0.0, "frames_completed": 0, "frames_total": 0,
                      "result": {"map_asset": None, "sequence_asset": None, "animation_asset": None,
                                 "capture_file": None, "preview_video": None, "output_dir": str(self.root),
                                 "low_confidence_ranges": [], "frame_count": 0, "fps": None,
                                 "scale_confidence": None, "intrinsics_source": None}, "error": None}
        self.protected = {}

    def check_cancel(self):
        if self.cancel_file.exists():
            raise Cancelled("Processing cancelled. Partial results have been kept.")

    def log(self, message):
        with (self.root / "job.log").open("a", encoding="utf-8") as stream:
            stream.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")

    def process_event(self, event, stage, process):
        record = {"event": event, "stage": stage, "pid": process.pid, "returncode": process.poll(),
                  "alive": process.poll() is None, "elapsed_seconds": round(time.monotonic() - self.started, 3)}
        with (self.root / "child_processes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.log(json.dumps(record))

    def update(self, stage=None, **values):
        if stage is not None:
            self.state.update(stage=stage, progress=BASE_PROGRESS[stage], message=MESSAGES[stage], frames_completed=0)
            self.log(MESSAGES[stage])
        self.state.update(values)
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        atomic_json(self.root / "status.json", self.state)

    def protect(self, path):
        path = Path(path).resolve()
        self.protected[str(path)] = sha256(path)

    def check_protected(self):
        for path, digest in self.protected.items():
            check_record(path, digest)

    def claim(self):
        # The bridge creates only request.json. Never rewrite an old terminal job.
        allowed = {"request.json", "cancel.request"}
        require(all(p.name in allowed for p in self.root.iterdir()), "The job directory is already in use. Start a new job; existing output will not be overwritten")
        with (self.root / "job_started.json").open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "job_id": self.request["job_id"], "worker_pid": os.getpid(),
                       "request_sha256": sha256(self.request_path)}, stream)

    def monitor_progress(self, stage):
        completed = self.state["frames_completed"]
        total = self.state["frames_total"]
        fraction = 0.0
        if stage == "reconstruct":
            try:
                progress = read_json(self.output / "progress.json")
                completed = int(progress.get("frames_completed", 0))
                fraction = min(1.0, max(0.0, float(progress.get("fraction", 0))))
                total = int(progress.get("frames_total", total))
            except (OSError, ValueError, TypeError):
                pass
        elif stage == "bake" and self.render_dir.exists():
            completed = len(list((self.render_dir / "overview").glob("frame_*.png")))
            fraction = min(1.0, completed / total) if total else 0.0
        span = {"reconstruct": 0.44, "bake": 0.26}.get(stage, 0.0)
        self.update(frames_completed=completed, frames_total=total, progress=BASE_PROGRESS[stage] + span * fraction)

    def child_failure_message(self, stage, returncode, log_path):
        message = f"{MESSAGES[stage]} failed (exit code {returncode})"
        # Only read the current reconstruction's fixed report paths. Their
        # contents are plain diagnostic text, never commands or output paths.
        if stage == "reconstruct":
            for report_path in (self.output / "report.json", self.output / "progress.json"):
                try:
                    with report_path.open("rb") as stream:
                        payload = stream.read(1024 * 1024 + 1)
                    if len(payload) > 1024 * 1024:
                        continue
                    report = json.loads(payload.decode("utf-8-sig"))
                except (OSError, UnicodeError, ValueError):
                    continue
                if not isinstance(report, dict) or report.get("status") != "failed":
                    continue
                raw_error = report.get("error")
                if not isinstance(raw_error, str) or not raw_error.strip():
                    continue
                self.log(json.dumps({"stage": stage, "failure_report": str(report_path),
                                     "original_error": raw_error}, ensure_ascii=False))
                known_errors = {
                    "ValueError: Input has unknown or duplicate hand side in a frame; resolve identity before animation export":
                        "Some frames contain unknown or duplicate hand identities. Animation could not be generated.",
                    "ValueError: Input has unknown hand side; resolve identity before animation export":
                        "The input contains unrecognized left/right hand labels. Animation could not be generated.",
                    "ValueError: Input has duplicate hand side without resolved identity; resolve identity before animation export":
                        "Some frames still contain conflicting hand identities. Animation could not be generated reliably.",
                    "ValueError: Both hands need at least one usable real observation after identity conflict filtering":
                        "Each hand needs a usable observation. Both-hand animation could not be generated.",
                }
                reason = known_errors.get(raw_error.strip(), raw_error.strip())
                reason = " ".join(reason.split())
                if len(reason) > 1200:
                    reason = reason[:1200] + "... (see the log for the full reason)"
                return f"{message}: {reason} Details: {log_path}"
        return f"{message}. See {log_path}"

    def child(self, stage, args):
        self.check_cancel()
        self.check_protected()
        self.update(stage)
        log_path = self.root / (stage + ".log")
        require(not log_path.exists(), f"The stage log already exists and will not be overwritten: {log_path}")
        self.log(json.dumps([str(arg) for arg in args], ensure_ascii=False))
        with log_path.open("w", encoding="utf-8") as log:
            process = self.popen([str(arg) for arg in args], cwd=str(self.pipeline), stdout=log,
                                 stderr=subprocess.STDOUT, **hidden_options())
            self.process_event("started", stage, process)
            try:
                while process.poll() is None:
                    self.check_cancel()
                    self.monitor_progress(stage)
                    self.sleep(0.2)
                self.check_cancel()  # Cancellation wins even when UE termination returned zero.
            except BaseException:
                self.terminate(process)
                self.process_event("termination_requested_and_waited", stage, process)
                raise
            self.process_event("exited", stage, process)
            if process.returncode != 0:
                raise RuntimeError(self.child_failure_message(stage, process.returncode, log_path))

    def ue_args(self, command, evidence):
        return [self.request["editor_executable"], self.request["project_file"], "-run=" + command,
                "-Evidence=" + str(evidence), "-abslog=" + str(evidence.with_suffix(".ue.log")),
                "-unattended", "-nop4", "-nosplash"]

    def reconstruction(self):
        args = [self.python, self.pipeline / "run_pipeline.py", "--video", self.request["video_path"],
                "--output", self.output, "--backend", "wilor", "--handedness-policy", "temporal",
                "--export-animation", "--bone-mapping", self.request["bone_mapping_path"],
                "--cancel-file", self.cancel_file]
        if self.request["allow_blurry"]:
            args.append("--allow-blurry")
        self.child("reconstruct", args)
        require(read_json(self.output / "report.json")["status"] == "completed", "The reconstruction report is incomplete")
        validation_path = self.animation_dir / "validation.json"
        validation = read_json(validation_path)
        capture_path = self.animation_dir / "capture.cardcap.json"
        observations = self.output / "hand_observations.json"
        require(validation.get("status") == "passed" and same_path(validation["capture_path"], capture_path)
                and same_path(validation["source_observations_path"], observations), "The reconstruction validation does not reference this job's output")
        check_record(capture_path, validation["capture_sha256"])
        check_record(observations, validation["source_observations_sha256"])
        require(validation["source_video_sha256"] == self.protected[self.request["video_path"]], "Source video checksum mismatch")
        capture = read_json(capture_path)
        from cardcap.packet_contract import validate_capture
        validate_capture(capture, read_json(self.request["bone_mapping_path"]))
        require(capture["format_version"] in ("1.0", "1.2", "1.3") and not capture["packets"] and not any(capture["events"].values()),
                "This workflow generates hand drafts only; existing card packets or events are not accepted")
        require(capture["meta"]["source_video_sha256"] == validation["source_video_sha256"]
                and same_path(capture["meta"]["source_video"], self.request["video_path"]), "The capture references a different source video")
        manifest_path = self.research_dir / "MANO_ResearchHands_Manifest.json"
        require(same_path(validation["research_hands_manifest"], manifest_path), "The hand model was not generated by this job; previous models cannot be reused")
        manifest = read_json(manifest_path)
        mapping_hash = self.protected[self.request["bone_mapping_path"]]
        require(validation["bone_mapping_sha256"] == mapping_hash == manifest["bone_mapping_sha256"]
                and same_path(manifest["bone_mapping_path"], self.request["bone_mapping_path"]), "The bone mapping does not match the hand model")
        require(manifest["shared_betas"] == validation["shared_mano_shape"], "The hand model shape does not match the animation")
        if capture["format_version"] in ("1.2", "1.3"):
            require(math.isclose(manifest["coordinates"]["geometry_scale_factor"], capture["scale"]["global_scale_factor_applied"], rel_tol=1e-10)
                    and manifest["coordinates"]["coordinate_units"] == capture["provenance"]["coordinate_units"],
                    "The hand model and animation have mismatched geometry multipliers or units")
        else:
            require(math.isclose(manifest["coordinates"]["meters_per_mano_unit"], capture["scale"]["meters_per_unit"], rel_tol=1e-10),
                    "The legacy hand model and animation scales do not match")
        outputs = [item for item in manifest["outputs"] if item["weight_mode"] == "top4_renormalized_UE_preview"]
        require(len(outputs) == 1 and outputs[0] in validation["research_glb_outputs"], "The matching Top4 hand model record is missing")
        record = outputs[0]
        mesh = self.research_dir / "MANO_ResearchHands_UE_Top4Preview.glb"
        require(same_path(record["path"], mesh) and record["bone_count"] == 33, "The matching GLB path or bone count is incorrect")
        check_record(mesh, record["sha256"])
        for path in (validation_path, capture_path, observations, manifest_path, mesh):
            self.protect(path)
        self.capture = capture
        self.capture_path = capture_path
        self.mesh_source = mesh
        meta = capture["meta"]
        self.state["frames_total"] = meta["frame_count"]
        self.state["result"].update(capture_file=str(capture_path), low_confidence_ranges=capture["quality"]["low_confidence_ranges"],
                                    frame_count=meta["frame_count"], fps=meta["fps"],
                                    scale_confidence=capture["scale"].get("scale_confidence", "unknown"),
                                    intrinsics_source=capture["camera"].get("intrinsics_source", "unknown"),
                                    coordinate_frame=capture.get("provenance",{}).get("coordinate_frame","shared_camera"))

    def import_mesh(self):
        require(not self.asset_directory.exists(), "The Unreal asset directory already exists and will not be overwritten")
        path = self.root / "UE_Import.json"
        args = self.ue_args("CardCapImportHands", path) + ["-MeshSource=" + str(self.mesh_source),
                 "-Destination=" + self.asset_root + "/Hands", "-NullRHI"]
        self.child("import", args)
        report = read_json(path)
        require(report.get("status") == "passed" and same_path(report["source"], self.mesh_source), "Import validation failed or the source hand model does not match")
        require(report["objects"] and all(item.get("saved_on_disk") is True for item in report["objects"]), "Some imported assets were not saved")
        require(len(report["skeletal_meshes"]) == 1, "This job must import one skeletal mesh containing both hands")
        mesh = report["skeletal_meshes"][0]
        require(mesh["bone_count"] == 33 and mesh["mesh_path"].startswith(self.asset_root + "/Hands/")
                and mesh["skeleton_path"].startswith(self.asset_root + "/Hands/"), "The imported mesh is not this job's matching 33-bone asset")
        self.mesh_asset = mesh["mesh_path"]
        self.protect(path)

    def check_bake(self, path, animation, reload):
        report = read_json(path)
        require(report.get("status") == "passed" and report.get("reloaded_in_fresh_process") is reload
                and same_path(report["capture_file"], self.capture_path)
                and report["mesh_path"] == self.mesh_asset and report["animation_path"] == animation,
                "Animation validation failed or does not reference this job's capture, hand model and animation")
        require(report["source_frames"] == self.capture["meta"]["frame_count"]
                and report["animation_keys"] == self.capture["meta"]["frame_count"] + 1
                and abs(report["fps"] - self.capture["meta"]["fps"]) < 0.001
                and report.get("compressed_codec_and_structure_present") is True
                and report.get("force_raw_console_value") == 0, "Animation frame count, timing or compression validation failed")
        self.protect(path)

    def bake(self):
        package = self.asset_root + "/Animation/Hands"
        self.animation_asset = package + ".Hands"
        self.bake_common = ["-Capture=" + str(self.capture_path), "-Mesh=" + self.mesh_asset,
                            "-Mapping=" + self.request["bone_mapping_path"]]
        path = self.root / "UE_Bake.json"
        args = self.ue_args("CardCapBake", path) + self.bake_common + ["-Animation=" + package,
                "-SequenceDirectory=" + self.asset_root + "/Review", "-SequenceName=HandsReview",
                "-RenderDirectory=" + str(self.render_dir), "-AllowCommandletRendering", "-RenderOffscreen"]
        self.child("bake", args)
        self.check_bake(path, self.animation_asset, False)
        render_path = self.render_dir / "sequence_render.json"
        report = read_json(render_path)
        count = self.capture["meta"]["frame_count"]
        require(report.get("success") is True and report["captured_frames"] == report["requested_frames"] == count
                and len(report["frames"]) == count, "The Sequencer render did not complete both hand views")
        require(report["map"] == self.asset_root + "/Review/HandsReview_Map.HandsReview_Map"
                and report["sequence"] == self.asset_root + "/Review/HandsReview_Sequence.HandsReview_Sequence", "The review scene does not belong to this job")
        for index, item in enumerate(report["frames"]):
            require(item["frame"] == index, "Rendered frame indices are not consecutive")
            for key, relative in (("file", f"frame_{index:06d}.png"),
                                  ("overview_file", f"overview/frame_{index:06d}.png")):
                require(item[key].replace("\\", "/") == relative and (self.render_dir / relative).is_file(), "Some rendered PNG files are missing")
            require(item["nonblack_pixels_above_8"] > 0 and item["overview_nonblack_pixels_above_8"] > 0, "Unreal returned completely black frames")
        display_gate = local_display_review(self.root, self.capture, report, check_cancel=self.check_cancel)
        if display_gate is not None:
            display_path = self.render_dir / "display_geometry_review.json"
            atomic_json(display_path, display_gate)
            require(display_gate["status"] == "passed", "The local hand preview is too small or invisible and failed the display check")
            self.protect(display_path)
        self.protect(render_path)
        self.state["result"].update(map_asset=report["map"], sequence_asset=report["sequence"], animation_asset=self.animation_asset)

    def verify(self):
        path = self.root / "UE_Reload.json"
        self.child("verify", self.ue_args("CardCapBake", path) + self.bake_common
                   + ["-Animation=" + self.animation_asset, "-ValidateOnly", "-NullRHI"])
        self.check_bake(path, self.animation_asset, True)

    def preview(self):
        if not self.request["render_preview"]:
            self.log("render_preview=false: MP4 creation skipped; the review scene and both PNG views have been kept")
            return
        self.child("preview", [self.python, self.pipeline / "editor_job.py", "--preview-request", self.request_path])
        path = self.root / "Preview/preview.json"
        report = read_json(path)
        video = self.root / "Preview/review.mp4"
        require(report.get("status") == "passed" and report["decoded_frames"] == self.capture["meta"]["frame_count"]
                and abs(report["fps"] - self.capture["meta"]["fps"]) < 0.01
                and same_path(report["path"], video) and report["capture_sha256"] == self.protected[str(self.capture_path)]
                and report["source_video_sha256"] == self.protected[self.request["video_path"]]
                and report["sequence_render_sha256"] == self.protected[str(self.render_dir / "sequence_render.json")],
                "The preview could not be fully decoded or does not match this job's input")
        check_record(video, report["sha256"])
        self.protect(path)
        self.state["result"]["preview_video"] = str(video)

    def run(self):
        self.claim()  # Failure here deliberately leaves any existing job untouched.
        try:
            self.update("preflight")
            self.check_cancel()
            require(not self.output.exists() and not self.asset_directory.exists() and not self.research_dir.exists(),
                    "Generated output or the Unreal asset directory already exists. Start a new job")
            environment = self.environment_check(self.request)
            atomic_json(self.root / "preflight.json", environment)
            for path in (self.request_path, self.request["video_path"], self.request["bone_mapping_path"]):
                self.protect(path)
            self.reconstruction()
            self.import_mesh()
            self.bake()
            self.verify()
            self.preview()
            self.check_cancel()
            self.check_protected()
            atomic_json(self.root / "job_result.json", {"schema_version": 1, "job_id": self.request["job_id"],
                        "protected_inputs_and_reports": self.protected, "result": self.state["result"],
                        "render_preview_false_semantics": "Skip MP4 only; saved Review map/sequence and both PNG views remain",
                        "scope": "Hand-only research draft. Pose, identity and metric accuracy require review; no card reconstruction or physical events."})
            self.update("complete", status="succeeded", frames_completed=self.state["frames_total"])
            return 0
        except (Cancelled, KeyboardInterrupt) as error:
            self.log(str(error))
            self.update(status="cancelled", message="Processing cancelled. Partial results have been kept.", error=None)
            return 2
        except Exception as error:
            details = traceback.format_exc()
            self.log(details)
            # A request can arrive while reports are being read after a child
            # exits. Its absence/partial output must not change cancel to fail.
            if self.cancel_file.exists():
                self.update(status="cancelled", message="Processing cancelled. Partial results have been kept.", error=None)
                return 2
            self.update(status="failed", message="Processing failed. Please view the log.",
                        error={"message": str(error), "details": details, "log_path": str(self.root / "job.log")})
            return 1


def build_preview(request_path):
    """CPU-only, real decoded source + already rendered frames; no pose changes."""
    import cv2
    import numpy as np
    request = read_request(request_path)
    root = Path(request["output_dir"])
    cancel = root / "cancel.request"

    def check_cancel():
        if cancel.exists():
            raise Cancelled("Preview creation cancelled")

    check_cancel()
    output = root / "Preview"
    require(not output.exists(), "The preview directory already exists and will not be overwritten")
    output.mkdir()
    capture_path = root / "pipeline/animation/capture.cardcap.json"
    capture = read_json(capture_path)
    render_path = root / "UE_Frames/sequence_render.json"
    render = read_json(render_path)
    local_only = capture.get("provenance",{}).get("coordinate_frame") == "per_hand_wrist_local"
    display_gate = local_display_review(root, capture, render, check_cancel=check_cancel)
    if display_gate is not None:
        atomic_json(output / "display_geometry_review.json", display_gate)
        require(display_gate["status"] == "passed", "The local hand preview is too small or invisible and failed the display check")
    video = Path(request["video_path"])
    check_record(video, capture["meta"]["source_video_sha256"])
    count, fps = capture["meta"]["frame_count"], capture["meta"]["fps"]
    video_path = output / "review.mp4"
    source = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (3840, 720))
    require(source.isOpened() and writer.isOpened(), "Could not read the source video or create the MP4 preview")
    require(abs(source.get(cv2.CAP_PROP_FPS) - fps) < 0.01, "The source video and animation frame rates do not match")

    def pane(image, title, label):
        canvas = np.full((720, 1280, 3), 18, np.uint8)
        ratio = min(1280 / image.shape[1], 576 / image.shape[0])
        w, h = round(image.shape[1] * ratio), round(image.shape[0] * ratio)
        x, y = (1280 - w) // 2, 72 + (576 - h) // 2
        canvas[y:y+h, x:x+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 245), 1, cv2.LINE_AA)
        return canvas

    source_styles = {
        "detected_model_observation": ("detected", (90, 205, 110)),
        "tracked_roi_model_observation": ("tracked_roi", (235, 180, 65)),
        "interpolated": ("interpolated", (65, 200, 245)),
        "endpoint_hold": ("endpoint_hold", (205, 125, 205)),
        "missing": ("missing", (120, 120, 120)),
    }
    # Display colors are categorical source labels, not confidence thresholds.
    def source_bands(canvas, index):
        for row, side in enumerate(("left", "right")):
            hand = next((h for h in capture["hands"] if h["side"] == side), None)
            sample = next((f for f in hand["frames"] if f["frame"] == index), None) if hand else None
            kind = sample.get("sample_kind", sample.get("diagnostics", {}).get("sample_kind", "missing")) if sample else "missing"
            kind = "detected_model_observation" if kind == "observed" else kind
            require(kind in source_styles, "The preview contains an unknown motion source")
            text, color = source_styles[kind]
            cv2.rectangle(canvas, (18, 652 + row * 30), (31, 672 + row * 30), color, -1)
            cv2.putText(canvas, side + ": " + text, (42, 669 + row * 30), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        for column, (text, color) in enumerate(source_styles.values()):
            x = 320 + column * 185
            cv2.rectangle(canvas, (x, 655), (x + 10, 665), color, -1)
            cv2.putText(canvas, text, (x + 16, 665), cv2.FONT_HERSHEY_SIMPLEX, .37, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Source labels describe each animation hand; fills are not observations.",
                    (320, 697), cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 210), 1, cv2.LINE_AA)
        return canvas

    try:
        for index in range(count):
            check_cancel()
            ok, original = source.read()
            require(ok and original is not None and list(original.shape[1::-1]) == capture["meta"]["resolution"], "The source video has missing frames or an unexpected resolution")
            images = []
            for relative in (f"frame_{index:06d}.png", f"overview/frame_{index:06d}.png"):
                image = cv2.imdecode(np.frombuffer((root / "UE_Frames" / relative).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                require(image is not None and list(image.shape[1::-1]) == [render["width"], render["height"]], "An Unreal PNG could not be decoded or has unexpected dimensions")
                images.append(image)
            review = any(start <= index <= end for start, end in capture["quality"]["low_confidence_ranges"])
            label = f"Frame {index}/{count-1}" + (" | LOW CONFIDENCE - REVIEW" if review else " | Research draft")
            hand_labels = []
            for hand in capture["hands"]:
                sample = next((item for item in hand["frames"] if item["frame"] == index), None)
                if sample is None:
                    hand_labels.append(hand["side"] + ": missing")
                else:
                    kind = sample.get("sample_kind", sample.get("diagnostics", {}).get("sample_kind", "unknown"))
                    hand_labels.append(f"{hand['side']}: {sample['confidence']:.2f} {kind}")
            cameras = "Intrinsics: " + capture["camera"].get("intrinsics_source", "unknown")
            scale = "Metric scale: " + str(capture["scale"].get("scale_confidence", "unknown")) + " | units: " + capture.get("provenance", {}).get("coordinate_units", "legacy cm")
            if local_only:
                spatial_note = "Relative hand placement UNKNOWN | display_only local camera | no scene depth"
                pose_panes = [pane(images[0], "LEFT hand | independent wrist-local pose",spatial_note),
                              pane(images[1], "RIGHT hand | independent wrist-local pose",spatial_note)]
            else:
                pose_panes = [pane(images[0], "UE capture camera | " + cameras, scale),
                              pane(images[1], "UE independent overview | conditional geometry",
                                   "Shared MANO model | lens distortion unknown unless calibrated | no measured scale claim")]
            composed = np.concatenate([source_bands(pane(original, "Source video", label), index)] +
                                      [source_bands(p,index) for p in pose_panes], axis=1)
            writer.write(composed)
            if index in {0, count // 2, count - 1}:
                ok, encoded = cv2.imencode(".jpg", composed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                require(ok, "A preview key frame could not be encoded")
                (output / f"frame_{index:06d}.jpg").write_bytes(encoded.tobytes())
        require(not source.read()[0], "The source video contains additional frames that were not exported")
    finally:
        source.release()
        writer.release()
    check_cancel()
    reader = cv2.VideoCapture(str(video_path))
    decoded = 0
    try:
        require(reader.isOpened() and abs(reader.get(cv2.CAP_PROP_FPS) - fps) < 0.01, "The preview could not be reopened or has an unexpected frame rate")
        while True:
            check_cancel()
            ok, image = reader.read()
            if not ok:
                break
            require(image is not None and image.shape[:2] == (720, 3840), "The preview has unexpected dimensions")
            decoded += 1
    finally:
        reader.release()
    require(decoded == count, "The fully decoded preview has an unexpected frame count")
    report = {"status": "passed", "path": str(video_path), "sha256": sha256(video_path), "decoded_frames": decoded,
              "fps": fps, "resolution": [3840, 720], "capture_sha256": sha256(capture_path),
              "source_video_sha256": sha256(video), "sequence_render_sha256": sha256(render_path),
              "layout": "Three 1280x720 panes; source/left-local/right-local" if local_only else "Three 1280x720 panes; source/capture-camera/overview",
              "view_mode": "per_hand_local" if local_only else "shared_camera",
              "display_only": local_only, "inter_hand_transform_known": not local_only,
              "display_review": "display_geometry_review.json" if local_only else None,
              "source_color_legend": {kind: {"label": label, "bgr": color} for kind, (label, color) in source_styles.items()},
              "coordinate_units": capture.get("provenance", {}).get("coordinate_units"),
              "geometry_changed": False, "worker_source_sha256": sha256(Path(__file__))}
    atomic_json(output / "preview.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument("--preview-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.preview_request:
            build_preview(args.preview_request)
            return 0
        return EditorJob(args.request).run()
    except Cancelled:
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
