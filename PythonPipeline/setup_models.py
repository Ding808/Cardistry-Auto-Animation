"""Download and verify the official versioned MediaPipe hand model for local use."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main() -> int:
    pipeline_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=pipeline_dir / "models")
    parser.add_argument("--report", type=Path, default=pipeline_dir / ".runtime-cache" / "hand-model-provenance.json")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "hand_landmarker.task"
    downloaded = not model_path.is_file()
    if downloaded:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != MODEL_SHA256:
            raise SystemExit(f"Downloaded model SHA-256 mismatch: {actual_hash}")
        temporary = model_path.with_suffix(".task.partial")
        temporary.write_bytes(content)
        temporary.replace(model_path)
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != MODEL_SHA256:
        raise SystemExit(f"Existing model SHA-256 mismatch; refusing to overwrite: {model_path}")
    with ZipFile(model_path) as bundle:
        members = [
            {"name": name, "bytes": bundle.getinfo(name).file_size,
             "sha256": hashlib.sha256(bundle.read(name)).hexdigest()}
            for name in bundle.namelist() if not name.endswith("/")
        ]
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "MediaPipe HandLandmarker full, float16, version 1",
        "source_url": MODEL_URL,
        "source_documentation": "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker",
        "license": "Apache-2.0",
        "license_notice": "See ThirdParty/MediaPipe and the official model card",
        "local_path": str(model_path.resolve()),
        "sha256": actual_hash,
        "bytes": model_path.stat().st_size,
        "downloaded_this_run": downloaded,
        "bundle_members": members,
        "scope": "Local inference model asset; no MANO/SMPL-X asset included",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
