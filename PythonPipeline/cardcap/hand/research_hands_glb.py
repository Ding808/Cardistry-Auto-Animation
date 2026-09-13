"""Build local MANO research hand skins as dependency-free glTF 2.0 GLB.

Uses actual licensed canonical-right MANO arrays and reflects them for left.
No pose inference or animation is performed. UE bone names are external data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .mano_assets import load_mano_arrays

PIPELINE = Path(__file__).resolve().parents[2]
PLUGIN = PIPELINE.parent
DEFAULT_CAMERA_TO_GLTF = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
UE_GLTF_TO_UE = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_betas(betas):
    result = np.zeros(10) if betas is None else np.asarray(betas, dtype=np.float64)
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError("Shared MANO betas must be ten finite values")
    return result


def _normals(vertices, faces):
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-15):
        raise ValueError("Actual MANO mesh has a vertex without a defined normal")
    return normals / lengths[:, None]


def build_research_hand_arrays(mano_path: Path, bone_mapping_path: Path, *, betas=None,
                              camera_to_gltf_axis=None, meters_per_mano_unit=1.0):
    """Return shape-specific real mesh/bind data using one shared ten-beta vector.

    Each hand is centered at its regressed wrist, then given the configured
    display offset in the same numerical coordinate space. The legacy
    meters_per_mano_unit argument is a geometry multiplier; this low-level
    function cannot establish metric evidence. Native LEFT bases are not used.
    Local rotations are identity; translations are parent-relative regressed J.
    """
    mano_path, bone_mapping_path = Path(mano_path), Path(bone_mapping_path)
    arrays = load_mano_arrays(mano_path)
    if mano_path.name != "MANO_RIGHT.pkl":
        raise ValueError("This canonical-right export requires official MANO_RIGHT.pkl")
    mapping = json.loads(bone_mapping_path.read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") != 1:
        raise ValueError("Unsupported bone mapping schema")
    names = [mapping["root_bone"]] + [name for side in ("left", "right") for name in mapping["hands"][side]["bone_names"]]
    if len(names) != 33 or len(set(names)) != 33 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Mapping must provide 33 unique nonempty bone names")
    if len(mapping["mano_joint_order"]) != 16:
        raise ValueError("MANO mapping must have 16 entries, excluding observation tips")
    parents = arrays["kintree_table"][0].astype(np.int64).copy()
    parents[0] = -1
    if not np.array_equal(parents, mapping["mano_parents"]):
        raise ValueError("Bone mapping parents do not match authenticated MANO")
    axis = DEFAULT_CAMERA_TO_GLTF.copy() if camera_to_gltf_axis is None else np.asarray(camera_to_gltf_axis, dtype=np.float64)
    if axis.shape != (3, 3) or not np.isfinite(axis).all() or not np.allclose(axis @ axis.T, np.eye(3), atol=1e-12):
        raise ValueError("camera_to_gltf_axis must be an orthogonal 3x3 matrix")
    unit = float(meters_per_mano_unit)
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("meters_per_mano_unit must be finite and positive")
    shape = _checked_betas(betas)
    vertices = arrays["v_template"] + np.tensordot(arrays["shapedirs"], shape, axes=(2, 0))
    joints = arrays["J_regressor"] @ vertices
    hands = {}
    for side in ("left", "right"):
        reflection = np.diag([-1., 1., 1.]) if side == "left" else np.eye(3)
        offset = np.asarray(mapping["hands"][side]["rest_offset_camera_m"], dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Hand rest display offset must contain three finite coordinate values")
        camera_vertices = (vertices - joints[[0]]) @ reflection.T * unit + offset
        camera_joints = (joints - joints[[0]]) @ reflection.T * unit + offset
        gltf_vertices, gltf_joints = camera_vertices @ axis.T, camera_joints @ axis.T
        faces = arrays["f"].astype(np.uint16).copy()
        if np.linalg.det(axis @ reflection) < 0:
            faces = faces[:, [0, 2, 1]]
        local = gltf_joints.copy()
        local[1:] -= gltf_joints[parents[1:]]
        hands[side] = {"positions": gltf_vertices, "joints_global": gltf_joints,
                       "joints_local": local, "normals": _normals(gltf_vertices, faces),
                       "faces": faces, "weights": arrays["weights"].copy(),
                       "bone_names": mapping["hands"][side]["bone_names"],
                       "parents": parents.copy(), "rest_offset_camera_m": offset}
    return {"hands": hands, "mapping": mapping, "betas": shape, "axis": axis, "unit": unit,
            "mano_sha256": _sha(mano_path), "mapping_sha256": _sha(bone_mapping_path)}


class _Glb:
    def __init__(self):
        self.binary = bytearray()
        self.document = {"asset": {"version": "2.0", "generator": "CardistryCapture actual MANO research skin exporter",
                                  "copyright": "MANO licensed research asset; local use only; not cleared for redistribution"},
                         "buffers": [], "bufferViews": [], "accessors": []}

    def accessor(self, array, kind, component, *, target=None, bounds=False):
        data = np.ascontiguousarray(array)
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(data.tobytes())
        view = {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes}
        if target is not None:
            view["target"] = target
        view_id = len(self.document["bufferViews"])
        self.document["bufferViews"].append(view)
        record = {"bufferView": view_id, "componentType": component, "count": len(data), "type": kind}
        if bounds:
            record.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
        index = len(self.document["accessors"])
        self.document["accessors"].append(record)
        return index

    def write(self, path):
        self.document["buffers"] = [{"byteLength": len(self.binary)}]
        text = json.dumps(self.document, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        text += b" " * (-len(text) % 4)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        total = 12 + 8 + len(text) + 8 + len(self.binary)
        path.write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(self.binary), 0x004E4942) + self.binary)


def _write_skin(data, path, *, top4):
    glb = _Glb()
    nodes = [{"name": data["mapping"]["root_bone"], "children": [1, 17]}]
    global_positions = [np.zeros(3)]
    for side in ("left", "right"):
        hand = data["hands"][side]
        start = len(nodes)
        for index in range(16):
            record = {"name": hand["bone_names"][index], "translation": hand["joints_local"][index].tolist(),
                      "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}
            children = [start + child for child in range(1, 16) if hand["parents"][child] == index]
            if children:
                record["children"] = children
            nodes.append(record)
            global_positions.append(hand["joints_global"][index])
    inverse_bind = np.tile(np.eye(4), (33, 1, 1))
    inverse_bind[:, :3, 3] = -np.asarray(global_positions)
    # glTF matrix values are column-major. Each item below is one 16-float MAT4.
    inverse_accessor = glb.accessor(inverse_bind.transpose(0, 2, 1).reshape(33, 16).astype("<f4"), "MAT4", 5126)
    primitives, reports = [], {}
    for material, side in enumerate(("left", "right")):
        hand = data["hands"][side]
        attributes = {
            "POSITION": glb.accessor(hand["positions"].astype("<f4"), "VEC3", 5126, target=34962, bounds=True),
            "NORMAL": glb.accessor(hand["normals"].astype("<f4"), "VEC3", 5126, target=34962),
        }
        source_weights = hand["weights"]
        selected = np.argsort(-source_weights, axis=1, kind="stable")[:, :4 if top4 else 16]
        weights = np.take_along_axis(source_weights, selected, axis=1)
        retained = weights.sum(axis=1)
        if top4:
            weights /= retained[:, None]
        exported = weights.astype("<f4")
        joint_indices = (selected + 1 + material * 16).astype("<u2")
        joint_indices[exported == 0] = 0
        for group in range(exported.shape[1] // 4):
            segment = slice(group * 4, group * 4 + 4)
            attributes[f"JOINTS_{group}"] = glb.accessor(joint_indices[:, segment], "VEC4", 5123, target=34962)
            attributes[f"WEIGHTS_{group}"] = glb.accessor(exported[:, segment], "VEC4", 5126, target=34962)
        indices = glb.accessor(hand["faces"].reshape(-1).astype("<u2"), "SCALAR", 5123, target=34963)
        primitives.append({"attributes": attributes, "indices": indices, "material": material, "mode": 4})
        reports[side] = {"vertices": 778, "triangles": 1538, "joints": 16, "weight_sets": exported.shape[1] // 4,
                         "source_mass_retained_min": float(retained.min()), "source_mass_retained_mean": float(retained.mean()),
                         "discarded_mass_max": float((1 - retained).max()),
                         "exported_weight_sum_max_error": float(np.max(np.abs(exported.sum(axis=1, dtype=np.float64) - 1))),
                         "bounds_gltf_units": [hand["positions"].min(axis=0).tolist(), hand["positions"].max(axis=0).tolist()],
                         "bounds_gltf_m": [hand["positions"].min(axis=0).tolist(), hand["positions"].max(axis=0).tolist()] if data["coordinate_units"] == "ue_centimeters" else None}
    nodes.append({"name": "MANO_ResearchHands_Mesh", "mesh": 0, "skin": 0})
    glb.document.update({"nodes": nodes, "skins": [{"name": "MANO_ResearchHands_Skin", "joints": list(range(33)), "skeleton": 0, "inverseBindMatrices": inverse_accessor}],
                         "meshes": [{"name": "MANO_ResearchHands", "primitives": primitives}],
                         "materials": [{"name": side + "_research", "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": color, "metallicFactor": 0, "roughnessFactor": 0.75}} for side, color in (("left", [0.35, 0.65, 0.9, 1]), ("right", [0.92, 0.62, 0.35, 1]))],
                         "scenes": [{"nodes": [0, 33]}], "scene": 0,
                         "extras": {"researchOnly": True, "isVideoInference": False, "sharedBetas": data["betas"].tolist(), "weightMode": "top4_renormalized_UE_preview" if top4 else "complete_MANO_weights", "canonicalLeft": "reflect_right_after_shape",
                                    "coordinateUnits": data["coordinate_units"], "geometryScaleFactor": data["unit"],
                                    "metricScaleDeclared": data["coordinate_units"] == "ue_centimeters", "independentMetricAccuracyVerified": False}})
    glb.write(path)
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size, "bone_count": 33, "mesh_node_count": 1,
            "animation_count": 0, "weight_mode": glb.document["extras"]["weightMode"], "hands": reports}


def export_research_hands(mano_path: Path = PIPELINE / "models/MANO_RIGHT.pkl",
                          bone_mapping_path: Path = PLUGIN / "Config/BoneMapping_UE5Mannequin.json",
                          output_dir: Path | None = None, *, betas=None,
                          camera_to_gltf_axis=None, meters_per_mano_unit=1.0,
                          coordinate_units="conditional_ue_units") -> dict[str, Any]:
    """Write full-weight and UE-top4 GLBs plus exact source arrays and manifest.

    ``betas`` is one finite length-10 vector shared by both mirrored hands.
    ``coordinate_units`` defaults to model-conditioned display geometry.
    The legacy ``meters_per_mano_unit`` argument remains a numeric multiplier
    for compatibility. Only an explicit ``ue_centimeters`` declaration permits
    a metric mapping in the manifest; the caller must bind its measurement
    provenance through the matching capture/export report. No metric accuracy
    is independently established by this serializer.
    The caller supplies a new output directory outside the installed plugin.
    The editor worker places this directory in its project's Saved job folder.
    """
    if coordinate_units not in ("conditional_ue_units", "ue_centimeters"):
        raise ValueError("coordinate_units must be conditional_ue_units or ue_centimeters")
    if output_dir is None:
        raise ValueError("An explicit new output_dir outside the installed plugin is required")
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(PLUGIN.resolve()):
        raise ValueError("Generated MANO assets must be written outside the installed plugin")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing hand mesh directory: {output_dir}")
    data = build_research_hand_arrays(mano_path, bone_mapping_path, betas=betas,
                                     camera_to_gltf_axis=camera_to_gltf_axis,
                                     meters_per_mano_unit=meters_per_mano_unit)
    data["coordinate_units"] = coordinate_units
    metric = coordinate_units == "ue_centimeters"
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = [_write_skin(data, output_dir / "MANO_ResearchHands_FullWeights.glb", top4=False),
               _write_skin(data, output_dir / "MANO_ResearchHands_UE_Top4Preview.glb", top4=True)]
    cache = {f"{side}_{key}": value for side, hand in data["hands"].items() for key, value in hand.items() if isinstance(value, np.ndarray)}
    cache.update(shared_betas=data["betas"], camera_to_gltf_axis=data["axis"])
    cache_path = output_dir / "MANO_ResearchHands_Arrays.npz"
    np.savez_compressed(cache_path, **cache)
    manifest = {"schema_version": 1, "purpose": "Actual MANO shape-specific rest research assets; no animation or video inference", "research_only": True,
                "license_scope": "Locally accepted MANO research terms; no redistribution or commercial clearance",
                "mano_right_sha256": data["mano_sha256"], "bone_mapping_path": str(Path(bone_mapping_path).resolve()), "bone_mapping_sha256": data["mapping_sha256"],
                "shared_betas": data["betas"].tolist(), "left_geometry": "Canonical MANO_RIGHT shaped mesh reflected in camera X; native LEFT shape bases not used",
                "bind_convention": "Identity local rotations/scales; inferred shape-specific model joints; subtract wrist then configured display offset; parent-relative translations. The legacy mapping/cache name rest_offset_camera_m does not make that display offset a subject measurement.",
                "coordinates": {"input": "MANO/WiLoR axes x right, y down, z away; model-source numbers alone do not measure personal metres",
                                "coordinate_units": coordinate_units, "geometry_scale_factor": data["unit"],
                                "camera_to_gltf_axis_column_vector": data["axis"].tolist(), "meters_per_mano_unit": data["unit"] if metric else None,
                                "gltf_standard": "Right-handed, +Y up, front +Z; glTF uses linear metres by convention. Conditional assets use those numbers for display only, not a known physical scene scale.",
                                "ue54_gltf_to_ue_axis_column_vector": UE_GLTF_TO_UE.tolist(), "ue54_meters_to_centimeters": 100 if metric else None,
                                "ue54_gltf_units_to_ue_units": 100,
                                "composed_camera_to_ue_axis_column_vector": (UE_GLTF_TO_UE @ data["axis"]).tolist(),
                                "expected_ue_position_formula": "p_UE = 100 * U * A * p_camera_after_geometry_factor_and_display_offset; physical cm only when coordinate_units=ue_centimeters",
                                "rotations": "R_gltf=A R_camera A^-1; R_UE=(U A) R_camera (U A)^-1; translation units only are scaled"},
                "provenance": {"kind": "inferred", "basis": "Licensed MANO and supplied shared shape; neutral rest model geometry, not a personal length measurement",
                               "metric_mapping": "caller-declared; bind to matching capture measurement provenance" if metric else "unobservable",
                               "independent_metric_accuracy_verified": False, "uncertainty": None},
                "ue54_weight_limit": "Local UE5.4 GLTF parser consumes JOINTS_0/WEIGHTS_0 only. Import the named top4-renormalized preview; full GLB/cache retain complete weights for exact reconstruction or custom importer.",
                "model_limit": "Skeletal LBS asset only; posedirs corrective blend shapes are not encoded. Exact MANO posed mesh additionally requires pose-corrective deformation before skinning.",
                "source_arrays": {"path": str(cache_path), "sha256": _sha(cache_path)}, "outputs": outputs,
                "gltf_specification": "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html"}
    manifest_path = output_dir / "MANO_ResearchHands_Manifest.json"
    importer_evidence = PIPELINE / "evidence/ue54-gltf-importer-evidence.json"
    if importer_evidence.is_file():
        manifest["ue54_importer_source_evidence"] = json.loads(importer_evidence.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(manifest, manifest_path=str(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory outside the installed plugin")
    parser.add_argument("--bone-mapping", type=Path, default=PLUGIN / "Config/BoneMapping_UE5Mannequin.json")
    parser.add_argument("--betas", type=float, nargs=10)
    parser.add_argument("--coordinate-config", type=Path, help="JSON with camera_to_gltf_axis, optional geometry_scale_factor (legacy meters_per_mano_unit) and explicit coordinate_units; default conditional_ue_units")
    args = parser.parse_args()
    coordinate = json.loads(args.coordinate_config.read_text(encoding="utf-8")) if args.coordinate_config else {}
    result = export_research_hands(bone_mapping_path=args.bone_mapping, output_dir=args.output_dir,
                                   betas=args.betas, camera_to_gltf_axis=coordinate.get("camera_to_gltf_axis"),
                                   meters_per_mano_unit=coordinate.get("geometry_scale_factor", coordinate.get("meters_per_mano_unit", 1.0)),
                                   coordinate_units=coordinate.get("coordinate_units", "conditional_ue_units"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
