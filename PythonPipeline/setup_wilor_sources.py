"""Prepare unchanged WiLoR sources after accepting their applicable terms."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from cardcap import __version__
COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"
FILES = {
    "wilor/models/backbones/vit.py": "5424ff143d2a8db9f0a722da34857bff2559c4867d9295fb3648e0293f5de013",
    "wilor/models/heads/refinement_net.py": "5878350b52af6c4cd27541ee02579ea7fe30549ef8e101a72fe2cf6a82eef59d",
    "wilor/utils/geometry.py": "9eb08e438622a36f3bc40d3fd48b776022b80ee26406f0d826d7b7132821e778",
    "mano_data/mano_mean_params.npz": "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988",
    "license.txt": "0bb835f5ed7272947a23d92d0e74bbdadcbf0beb9235c0735c32d1310a4bceda",
}


def prepare(destination: Path, *, accepted_model_licenses: bool, check_only: bool = False) -> dict:
    if not check_only and not accepted_model_licenses:
        raise PermissionError("Read and accept the applicable WiLoR, MANO and SMPL-X terms before preparing model sources.")
    records = []
    for relative, expected in FILES.items():
        path = destination / relative
        url = f"https://raw.githubusercontent.com/rolpotamias/WiLoR/{COMMIT}/{relative}"
        if path.is_file():
            content = path.read_bytes()
        elif check_only:
            raise FileNotFoundError(f"Missing model source: {relative}. Run Scripts/Setup.cmd.")
        else:
            request = urllib.request.Request(url, headers={"User-Agent": f"CardistryCapture-Setup/{__version__}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read(1024 * 1024)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Model source checksum mismatch: {relative}; existing files will not be replaced.")
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
        records.append({"relative_path": relative, "source_url": url, "sha256": expected, "modified": False})
    result = {"upstream_commit": COMMIT, "assets": records,
              "runtime_files": list(FILES)[:3], "redistribution_clearance": False,
              "usage": "Subject to the user's separately accepted WiLoR, MANO and SMPL-X terms."}
    if not check_only:
        (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-model-licenses", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent / "models/wilor_source")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, accepted_model_licenses=args.accepted_model_licenses,
                             check_only=args.check_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
