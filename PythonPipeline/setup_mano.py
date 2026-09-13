"""Install an already-authorized official MANO archive for local research only.

This copies only the two hand models and their accompanying license. It never
downloads restricted assets, executes archive code, or unpickles a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install(archive: Path, destination: Path) -> dict:
    archive = archive.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    entries = {
        "MANO_LEFT.pkl": "mano_v1_2/models/MANO_LEFT.pkl",
        "MANO_RIGHT.pkl": "mano_v1_2/models/MANO_RIGHT.pkl",
        "MANO_LICENSE.txt": "mano_v1_2/models/LICENSE.txt",
        "MANO_MODEL_INFO.txt": "mano_v1_2/models/info.txt",
    }
    payloads = {}
    with ZipFile(archive) as package:
        bad_entry = package.testzip()
        if bad_entry:
            raise ValueError(f"Archive CRC failed: {bad_entry}")
        for target, member in entries.items():
            data = package.read(member)
            if not data:
                raise ValueError(f"Empty archive member: {member}")
            payloads[target] = data
    files = []
    for name, data in payloads.items():
        target = destination / name
        digest = sha256(data)
        if target.exists() and sha256(target.read_bytes()) != digest:
            raise FileExistsError(f"Refusing to replace a different model: {target}")
        entry = {"file": name, "archive_member": entries[name], "bytes": len(data),
                 "sha256": digest}
        if name.endswith(".pkl"):
            operations = list(pickletools.genops(data))
            if operations[-1][0].name != "STOP":
                raise ValueError(f"Incomplete pickle stream: {name}")
            entry["pickle_global_references"] = sorted({arg for op, arg, _ in operations
                                                        if op.name == "GLOBAL"})
            entry["pickle_validation"] = "Static opcode parsing only; no unpickling or execution"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        files.append(entry)
    result = {
        "asset": "MANO v1.2 official left and right hand models",
        "source_page": "https://mano.is.tue.mpg.de/download.php",
        "official_links": [
            "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip",
            "https://mano.is.tue.mpg.de/download/dl.php?domain=mano&resume=1&sfile=mano_v1_2.zip",
        ],
        "retrieval": "User-provided official MANO archive",
        "usage_scope": "Subject to the user's separately acquired MANO license",
        "redistribution": "Excluded from plugin distribution; preserve accompanying license",
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive), "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive.read_bytes()),
        "archive_crc_all_members": "passed", "files": files,
    }
    (destination / "MANO_ASSET_MANIFEST.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
    arguments = parser.parse_args()
    print(json.dumps(install(arguments.archive, arguments.models_dir), ensure_ascii=False, indent=2))
