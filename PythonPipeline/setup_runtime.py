"""Install and verify the Windows runtime and separately licensed hand assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".runtime-cache"
CHECKPOINT_SHA256 = "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"
CONFIG_SHA256 = "f69cb52704df88ef29a7cfe03f35a677a92c1ce08169b729ca7f5862f05d4297"
MODEL_TERMS_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def manifest() -> dict:
    result = json.loads((ROOT / "runtime-artifacts.json").read_text(encoding="utf-8-sig"))
    checksum = result.get("archive_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("The runtime release manifest has no valid archive checksum.")
    return result


def runtime_archive(path: Path | None, records: dict, *, extract: bool = True) -> list[Path]:
    if path is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / "CardistryCapture-RuntimeDeps-v0.0.1.zip"
        if not path.is_file():
            temporary = path.with_suffix(".zip.partial")
            print("Downloading the versioned runtime dependency archive...", flush=True)
            with urllib.request.urlopen(records["archive_url"], timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if digest(temporary) != records["archive_sha256"]:
                raise ValueError("Downloaded runtime archive checksum mismatch; it was not installed.")
            temporary.replace(path)
    path = path.resolve(strict=True)
    if digest(path) != records["archive_sha256"]:
        raise ValueError("Runtime archive checksum mismatch; it was not installed.")
    wheels = []
    with ZipFile(path) as archive:
        # Never extract arbitrary paths from a downloaded archive.
        for record in records["wheels"]:
            members = [name for name in archive.namelist()
                       if Path(name.replace("\\", "/")).name == record["file"]]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one runtime wheel: {record['file']}")
            content = archive.read(members[0])
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Wheel checksum mismatch: {record['file']}")
            target = CACHE / "wheels" / record["file"]
            if extract:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and digest(target) != record["sha256"]:
                    raise ValueError(f"Existing cached wheel differs: {record['file']}")
                if not target.exists():
                    target.write_bytes(content)
            wheels.append(target)
    return wheels


def pinned_versions(path: Path) -> dict:
    return dict(line.strip().split("==", 1) for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def installed_wheel(record: dict) -> bool:
    try:
        distribution = metadata.distribution(record["package"])
    except metadata.PackageNotFoundError:
        return False
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_hash = source.get("archive_info", {}).get("hashes", {}).get("sha256")
    if distribution.version != record["version"] or source_hash != record["sha256"]:
        raise ValueError(f"{record['package']} has a different version or source. Use a new isolated .venv; it will not be automatically replaced.")
    return True


def run_pip(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "--isolated", *arguments], cwd=ROOT, check=True)


def copy_model(source: Path | None, name: str, expected: str) -> None:
    destination = ROOT / "models" / name
    if source is None:
        source = destination
    source = source.resolve(strict=True)
    if digest(source) != expected:
        raise ValueError(f"Unsupported model file or checksum mismatch: {name}")
    if destination.exists():
        if digest(destination) != expected:
            raise FileExistsError(f"Refusing to overwrite a different model: {name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected:
        raise ValueError(f"Model copy checksum mismatch: {name}")
    temporary.replace(destination)


def verify_runtime(records: dict, *, include_models: bool) -> dict:
    versions = pinned_versions(ROOT / "requirements-runtime.txt")
    versions["torch"] = "2.10.0+cu128"
    if include_models:
        versions.update(pinned_versions(ROOT / "requirements-models.txt"))
    for record in records["wheels"]:
        if not installed_wheel(record):
            raise RuntimeError(f"Missing runtime wheel: {record['package']}")
        versions[record["package"]] = record["version"]
    for name, expected in versions.items():
        if metadata.version(name) != expected:
            raise ValueError(f"Unexpected {name} version; run Scripts/Setup.cmd.")
    for forbidden in ("ultralytics",):
        try:
            metadata.version(forbidden)
        except metadata.PackageNotFoundError:
            continue
        raise ValueError(f"Unsupported dependency is installed: {forbidden}. Use a new isolated .venv.")
    import cv2
    import mediapipe
    import manifold3d
    import numpy
    import scipy
    import sounddevice
    import torch
    from scipy.optimize import least_squares
    trial = least_squares(lambda x: x - 1., [0.], method="trf", loss="huber")
    if not trial.success or abs(trial.x[0] - 1.) > 1e-7 or not hasattr(manifold3d, "Mesh64"):
        raise RuntimeError("Solver runtime verification failed.")
    cuda_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda_available else None
    if cuda_available:
        value = (torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")).sum().item()
        if value != 4096.:
            raise RuntimeError("CUDA arithmetic verification failed.")
    model_ready = False
    if include_models:
        import smplx
        import setup_wilor_sources
        from cardcap.hand.mano_assets import load_mano_arrays
        setup_wilor_sources.prepare(ROOT / "models/wilor_source", accepted_model_licenses=False, check_only=True)
        copy_model(None, "wilor_final.ckpt", CHECKPOINT_SHA256)
        copy_model(None, "model_config.yaml", CONFIG_SHA256)
        # The restricted pickle decoder validates the registered model files.
        for side in ("LEFT", "RIGHT"):
            load_mano_arrays(ROOT / "models" / f"MANO_{side}.pkl")
        from cardcap.hand._wilor_runtime import load_source_modules
        load_source_modules(ROOT / "models/wilor_source", ROOT)
        model_ready = True
    from setup_models import MODEL_SHA256
    if digest(ROOT / "models/hand_landmarker.task") != MODEL_SHA256:
        raise ValueError("MediaPipe model checksum mismatch.")
    return {"runtime_ready": True, "model_assets_ready": model_ready,
            "cuda_available": cuda_available, "cuda_device": device,
            "one_click_ready": model_ready and cuda_available,
            "versions": versions,
            "verification": "Dependency versions and source hashes, imports, solver and CUDA arithmetic; no video inference."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--verify-runtime-archive", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--accepted-runtime-licenses", action="store_true")
    parser.add_argument("--accepted-model-licenses", action="store_true")
    parser.add_argument("--mano-archive", type=Path)
    parser.add_argument("--wilor-checkpoint", type=Path)
    parser.add_argument("--wilor-config", type=Path)
    args = parser.parse_args()
    records = manifest()
    if args.verify_runtime_archive:
        if args.runtime_archive is None:
            parser.error("--verify-runtime-archive requires --runtime-archive")
        runtime_archive(args.runtime_archive, records, extract=False)
        print("Runtime archive and all five wheel checksums verified.")
        return 0
    if sys.version_info[:2] != (3, 10) or platform.system() != "Windows" or platform.architecture()[0] != "64bit":
        raise RuntimeError("Windows x64 and Python 3.10 x64 are required.")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Run Scripts/Setup.cmd to create and use this plugin's isolated Python environment.")
    if not args.check_only:
        acceptance_path = CACHE / "license-acceptance.json"
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig")) if acceptance_path.is_file() else {}
        runtime_notice_hash = digest(ROOT.parent / "ThirdParty/MSVC/README.md")
        prior_runtime_acceptance = (acceptance.get("release") == records["release"] and
                                    acceptance.get("runtime_license_notice_sha256") == runtime_notice_hash)
        if not args.accepted_runtime_licenses and not prior_runtime_acceptance:
            raise PermissionError("Read the runtime distribution terms in ThirdParty/MSVC/README.md, then accept them through Scripts/Setup.cmd.")
        prior_model_acceptance = (acceptance.get("accepted_model_licenses") is True and
                                  acceptance.get("model_terms_commit") == MODEL_TERMS_COMMIT)
        args.accepted_model_licenses = args.accepted_model_licenses or prior_model_acceptance
        if not args.runtime_only and not args.accepted_model_licenses:
            raise PermissionError("Model setup requires separately accepted licenses. Use Scripts/Setup.cmd, or choose --runtime-only.")
        CACHE.mkdir(parents=True, exist_ok=True)
        acceptance.update({"release": records["release"], "runtime_license_notice_sha256": runtime_notice_hash,
                           "runtime_accepted_at_utc": acceptance.get("runtime_accepted_at_utc", datetime.now(timezone.utc).isoformat())})
        if args.accepted_model_licenses:
            acceptance.update({"accepted_model_licenses": True, "model_terms_commit": MODEL_TERMS_COMMIT,
                               "model_accepted_at_utc": acceptance.get("model_accepted_at_utc", datetime.now(timezone.utc).isoformat())})
        acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
        wheels = runtime_archive(args.runtime_archive, records)
        missing = [str(path) for path, record in zip(wheels, records["wheels"]) if not installed_wheel(record)]
        run_pip("install", "--index-url", "https://pypi.org/simple", "--no-deps", "pip==25.3")
        if missing:
            run_pip("install", "--no-index", "--no-deps", *missing)
        run_pip("install", "--index-url", "https://pypi.org/simple", "--only-binary=:all:", "--no-deps", "-r", str(ROOT / "requirements-runtime.txt"))
        run_pip("install", "--index-url", "https://download.pytorch.org/whl/cu128", "--only-binary=:all:", "--no-deps", "torch==2.10.0+cu128")
        subprocess.run([sys.executable, str(ROOT / "setup_models.py")], cwd=ROOT, check=True)
        if not args.runtime_only:
            run_pip("install", "--index-url", "https://pypi.org/simple", "--only-binary=:all:", "--no-deps", "-r", str(ROOT / "requirements-models.txt"))
            import setup_mano
            import setup_wilor_sources
            if args.mano_archive is not None:
                setup_mano.install(args.mano_archive, ROOT / "models")
            copy_model(args.wilor_checkpoint, "wilor_final.ckpt", CHECKPOINT_SHA256)
            copy_model(args.wilor_config, "model_config.yaml", CONFIG_SHA256)
            setup_wilor_sources.prepare(ROOT / "models/wilor_source", accepted_model_licenses=True)
        run_pip("check")
    result = verify_runtime(records, include_models=not args.runtime_only)
    if not args.check_only:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "setup-status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.runtime_only:
        print("Runtime installed. Run Setup again with your licensed hand assets before processing videos.")
        return 0
    if not result["one_click_ready"]:
        print("A supported NVIDIA GPU and CUDA-compatible driver are required for one-click processing.", file=sys.stderr)
        return 2
    print("Setup complete. Enable CardistryCapture in Unreal Editor and open the capture panel.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
