"""Read hash-registered MANO v1.2 assets without importing SciPy or Chumpy.

Only seven globals observed in the official assets are accepted. Legacy
Chumpy and CSC objects become inert state records; numerical conversion is
implemented explicitly. This is an asset-specific reader, not a general
pickle sandbox. It requires the adjacent audited MANO_ASSET_MANIFEST.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import sys
from typing import Any

import numpy as np


class _StateRecord:
    def __setstate__(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise pickle.UnpicklingError("Unexpected legacy MANO object state")
        self.state = state


class _ChRecord(_StateRecord):
    pass


class _SelectRecord(_StateRecord):
    pass


class _CSCRecord(_StateRecord):
    pass


_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("__builtin__", "set"): set,
    ("chumpy.ch", "Ch"): _ChRecord,
    ("chumpy.reordering", "Select"): _SelectRecord,
    ("scipy.sparse.csc", "csc_matrix"): _CSCRecord,
}


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"MANO pickle global is not permitted: {module}.{name}") from exc

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError("Persistent pickle references are not permitted")


def _numeric_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric NumPy array")
    if value.size > 10_000_000 or not np.isfinite(value).all():
        raise ValueError(f"{name} has an invalid size or nonfinite values")
    return value


def _convert_chumpy(value: Any, depth: int = 0) -> np.ndarray:
    if depth > 4:
        raise ValueError("Unexpected nested Chumpy graph")
    if isinstance(value, np.ndarray):
        return _numeric_array(value, "Chumpy array")
    if isinstance(value, _ChRecord):
        return _numeric_array(value.state["x"], "Ch.x")
    if isinstance(value, _SelectRecord):
        source = _convert_chumpy(value.state["a"], depth + 1)
        indices = _numeric_array(value.state["idxs"], "Select.idxs")
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError("Select indices must be a one-dimensional integer array")
        if indices.size and (indices.min() < 0 or indices.max() >= source.size):
            raise ValueError("Select index out of bounds")
        shape = value.state.get("preferred_shape", (indices.size,))
        if not isinstance(shape, tuple) or not all(isinstance(size, int) and size >= 0 for size in shape):
            raise ValueError("Select preferred_shape is invalid")
        if np.prod(shape, dtype=np.int64) != indices.size:
            raise ValueError("Select preferred_shape does not match its index count")
        # Chumpy Select.compute_r uses ravel()[idxs], then preferred_shape.
        return source.reshape(-1)[indices].copy().reshape(shape)
    raise ValueError(f"Unsupported Chumpy state: {type(value).__name__}")


def _dense_csc(value: Any) -> np.ndarray:
    if not isinstance(value, _CSCRecord):
        return _numeric_array(value, "J_regressor")
    state = value.state
    if state.get("_shape") != (16, 778):
        raise ValueError("Unexpected MANO joint regressor shape")
    data = _numeric_array(state["data"], "CSC.data")
    indices = _numeric_array(state["indices"], "CSC.indices")
    indptr = _numeric_array(state["indptr"], "CSC.indptr")
    if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (779,):
        raise ValueError("Invalid MANO CSC array sizes")
    if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
        raise ValueError("CSC indices and offsets must be integer arrays")
    if indptr[0] != 0 or indptr[-1] != data.size or (np.diff(indptr) < 0).any():
        raise ValueError("Invalid CSC column offsets")
    if indices.size and (indices.min() < 0 or indices.max() >= 16):
        raise ValueError("CSC row index out of bounds")
    columns = np.repeat(np.arange(778), np.diff(indptr))
    result = np.zeros((16, 778), dtype=data.dtype)
    # add.at preserves the sum of duplicate entries, as CSC.toarray does.
    np.add.at(result, (indices, columns), data)
    return result


def _verify_asset(path: Path) -> tuple[bytes, str]:
    manifest_path = path.parent / "MANO_ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = [entry for entry in manifest["files"] if entry["file"] == path.name]
    if len(entries) != 1 or path.name not in {"MANO_LEFT.pkl", "MANO_RIGHT.pkl"}:
        raise ValueError("MANO asset is not uniquely registered in its audited manifest")
    if path.stat().st_size != entries[0]["bytes"] or path.stat().st_size > 10_000_000:
        raise ValueError("MANO asset size does not match the audited manifest")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entries[0]["sha256"]:
        raise ValueError("MANO asset SHA-256 does not match the audited manifest")
    for opcode, argument, _ in pickletools.genops(data):
        if opcode.name == "GLOBAL":
            module, name = argument.split(" ", 1)
            if (module, name) not in _ALLOWED_GLOBALS:
                raise pickle.UnpicklingError(f"Unapproved MANO global: {module}.{name}")
        if opcode.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "STACK_GLOBAL"}:
            raise pickle.UnpicklingError(f"Unsupported MANO pickle opcode: {opcode.name}")
    return data, digest


_SHAPES = {
    "v_template": (778, 3),
    "shapedirs": (778, 3, 10),
    "posedirs": (778, 3, 135),
    "J_regressor": (16, 778),
    "kintree_table": (2, 16),
    "weights": (778, 16),
    "hands_components": (45, 45),
    "hands_mean": (45,),
    "f": (1538, 3),
    "J": (16, 3),
}


def validate_mano_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate fixed official MANO v1.2 topology and numerical structure."""
    for name, shape in _SHAPES.items():
        if name not in arrays:
            raise ValueError(f"Missing MANO array: {name}")
        array = _numeric_array(arrays[name], name)
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    faces, tree = arrays["f"], arrays["kintree_table"]
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= 778:
        raise ValueError("MANO face vertex indices are invalid")
    if tree.dtype.kind not in "iu" or not np.array_equal(tree[1], np.arange(16)):
        raise ValueError("MANO joint identifiers are unexpected")
    if any(not 0 <= int(tree[0, index]) < index for index in range(1, 16)):
        raise ValueError("MANO joint hierarchy is not a parent-before-child tree")
    weights = arrays["weights"]
    regressor = arrays["J_regressor"]
    weight_error = float(np.max(np.abs(weights.sum(axis=1) - 1)))
    regressor_error = float(np.max(np.abs(regressor.sum(axis=1) - 1)))
    if weights.min() < -1e-8 or weight_error > 1e-5 or regressor_error > 1e-5:
        raise ValueError("MANO skinning/regression weights fail normalization checks")
    return {
        "vertices": 778,
        "faces": 1538,
        "skeletal_joints": 16,
        "shape_parameters": 10,
        "hand_pose_axis_angle_parameters": 45,
        "pose_corrective_dimensions": 135,
        "maximum_skinning_weight_sum_error": weight_error,
        "maximum_joint_regressor_row_sum_error": regressor_error,
        "template_bounds_m": [arrays["v_template"].min(axis=0).tolist(), arrays["v_template"].max(axis=0).tolist()],
        "regressed_template_vs_stored_J_max_abs_m": float(np.max(np.abs(regressor @ arrays["v_template"] - arrays["J"]))),
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
    }


def load_mano_arrays(path: Path) -> dict[str, np.ndarray]:
    """Return source-keyed pure arrays from a hash-registered official MANO PKL.

    No cache is implicitly trusted and no unknown pickle global is resolved.
    Chumpy Select is evaluated as indexed array selection; CSC is expanded to
    a dense 16x778 regressor. Source values and units are preserved.
    """
    path = Path(path).expanduser().resolve()
    data, _ = _verify_asset(path)
    source = _ManoUnpickler(io.BytesIO(data), encoding="latin1").load()
    if not isinstance(source, dict) or source.get("bs_style") != "lbs" or source.get("bs_type") != "lrotmin":
        raise ValueError("Unexpected MANO model type")
    arrays = {}
    for name in _SHAPES:
        value = source[name]
        if name == "shapedirs":
            value = _convert_chumpy(value)
        elif name == "J_regressor":
            value = _dense_csc(value)
        arrays[name] = np.ascontiguousarray(_numeric_array(value, name)).copy()
    if "hands_coeffs" in source:
        arrays["hands_coeffs"] = np.ascontiguousarray(_numeric_array(source["hands_coeffs"], "hands_coeffs")).copy()
    validate_mano_arrays(arrays)
    return arrays


def write_mano_cache(path: Path, cache_path: Path | None = None) -> dict[str, Any]:
    """Create a numerical-only NPZ and a provenance/shape report alongside it."""
    path = Path(path).resolve()
    arrays = load_mano_arrays(path)
    cache_path = Path(cache_path) if cache_path else path.with_suffix(".arrays.npz")
    if cache_path.resolve() == path or cache_path.suffix.lower() != ".npz":
        raise ValueError("Cache must be a separate .npz file; source assets cannot be overwritten")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as output:
        np.savez_compressed(output, **arrays)
    with np.load(cache_path, allow_pickle=False) as reloaded:
        if set(reloaded.files) != set(arrays) or any(not np.array_equal(arrays[key], reloaded[key]) for key in arrays):
            raise ValueError("MANO numerical cache roundtrip failed")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "cache_allow_pickle": False,
        "cache_exact_array_roundtrip": True,
        "approved_globals": [f"{module}.{name}" for module, name in _ALLOWED_GLOBALS],
        "scipy_imported": any(name == "scipy" or name.startswith("scipy.") for name in sys.modules),
        "chumpy_imported": any(name == "chumpy" or name.startswith("chumpy.") for name in sys.modules),
        "license_scope": "MANO research assets and derivative caches remain local and excluded from redistribution",
        **validate_mano_arrays(arrays),
    }
    cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = [write_mano_cache(path) for path in args.models]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
