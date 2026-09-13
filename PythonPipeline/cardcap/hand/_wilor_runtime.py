"""Original inference orchestration for unchanged, local WiLoR source assets.

The author source remains in ignored models/wilor_source, with its own terms.
Only geometry, ViT and RefineNet are imported. No training/detector/renderer
package initializer is executed and no third-party package is impersonated.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn


CHECKPOINT_SHA256 = "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"
SOURCE_SHA256 = {
    "wilor/models/backbones/vit.py": "5424ff143d2a8db9f0a722da34857bff2559c4867d9295fb3648e0293f5de013",
    "wilor/models/heads/refinement_net.py": "5878350b52af6c4cd27541ee02579ea7fe30549ef8e101a72fe2cf6a82eef59d",
    "wilor/utils/geometry.py": "9eb08e438622a36f3bc40d3fd48b776022b80ee26406f0d826d7b7132821e778",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @classmethod
    def convert(cls, value):
        if isinstance(value, dict):
            return cls({key: cls.convert(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.convert(item) for item in value]
        return value


def _module(name, path, import_override=None):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    if import_override is not None:
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=import_override)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_source_modules(source_root: Path, pipeline_root: Path):
    """Resolve relative imports in an isolated package, without upstream init."""
    for relative, expected_hash in SOURCE_SHA256.items():
        if file_sha256(source_root / relative) != expected_hash:
            raise ValueError(f"Reviewed WiLoR source hash changed: {relative}")
    helper_package = pipeline_root.parent / "ThirdParty" / "timm"
    helper_records = json.loads((helper_package / "MANIFEST.json").read_text(encoding="utf-8-sig"))["files"]
    helper_root = helper_package / "timm" / "models" / "layers"
    expected = {Path(record["relative_path"]).name: record["sha256"] for record in helper_records}
    prefix = "_cardcap_local_wilor_" + hashlib.sha256(str(source_root).encode()).hexdigest()[:12]
    modules = {}
    for name in ("drop", "helpers", "weight_init"):
        path = helper_root / (name + ".py")
        if file_sha256(path) != expected[path.name]:
            raise ValueError(f"Audited timm helper hash changed: {path}")
        modules[name] = _module(prefix + "_timm_" + name, path)
    layers = types.ModuleType(prefix + "_timm_layers")
    layers.drop_path = modules["drop"].drop_path
    layers.to_2tuple = modules["helpers"].to_2tuple
    layers.trunc_normal_ = modules["weight_init"].trunc_normal_

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "timm.models.layers":
            if set(fromlist) - {"drop_path", "to_2tuple", "trunc_normal_"}:
                raise ImportError("Unreviewed timm helper requested")
            return layers
        return builtins.__import__(name, globals, locals, fromlist, level)

    for suffix in ("", ".utils", ".models", ".models.backbones", ".models.heads"):
        package = types.ModuleType(prefix + suffix)
        package.__path__ = []
        sys.modules[prefix + suffix] = package
    geometry = _module(prefix + ".utils.geometry", source_root / "wilor/utils/geometry.py")
    vit = _module(prefix + ".models.backbones.vit", source_root / "wilor/models/backbones/vit.py", local_import)
    refinement = _module(prefix + ".models.heads.refinement_net", source_root / "wilor/models/heads/refinement_net.py")
    return vit, refinement, geometry


class CheckpointMANO(nn.Module):
    """Run official SMPL-X LBS on the actual WiLoR checkpoint MANO buffers.

    The checkpoint's right-hand geometry is compared against independently
    downloaded MANO_RIGHT before inference; this class does not parse pickle.
    """
    def __init__(self, state):
        super().__init__()
        from smplx.lbs import lbs
        self._lbs = lbs
        for full_name, value in state.items():
            if not full_name.startswith("mano."):
                continue
            pieces = full_name.removeprefix("mano.").split(".")
            parent = self
            for component in pieces[:-1]:
                if not hasattr(parent, component):
                    parent.add_module(component, nn.Module())
                parent = getattr(parent, component)
            parent.register_buffer(pieces[-1], value)

    def forward(self, global_orient, hand_pose, betas):
        # MANOLayer accepts rotation matrices and does not add the axis-angle
        # hand mean in this path. Do not use MANO.forward's PCA/AA processing.
        full_pose = torch.cat((global_orient, hand_pose), dim=1)
        vertices, joints = self._lbs(
            betas, full_pose, self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.lbs_weights, pose2rot=False,
        )
        tips = vertices.index_select(1, self.extra_joints_idxs)
        joints = torch.cat((joints, tips), dim=1).index_select(1, self.joint_map)
        return vertices, joints


class Reconstruction(nn.Module):
    def __init__(self, cfg, modules, state):
        super().__init__()
        vit, refinement, geometry = modules
        self.cfg = cfg
        self._projection = geometry.perspective_projection
        self.backbone = vit.vit(cfg)
        self.refine_net = refinement.RefineNet(cfg, feat_dim=1280, upscale=3)
        self.mano = CheckpointMANO(state)
        self.loaded_keys = {}
        for name in ("backbone", "refine_net", "mano"):
            prefix = name + "."
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            getattr(self, name).load_state_dict(selected, strict=True, assign=True)
            self.loaded_keys[name] = len(selected)
        ignored = [key for key in state if not key.startswith(("backbone.", "refine_net.", "mano."))]
        if any(key != "initialized" and not key.startswith("discriminator.") for key in ignored):
            raise ValueError("Checkpoint contains unreviewed inference state")
        self.ignored_training_keys = len(ignored)

    def forward(self, crops):
        # Reproduce the author's forward_step ordering with unchanged modules.
        params, camera, features, image_features = self.backbone(crops[:, :, :, 32:-32])
        focal = crops.new_full((crops.shape[0], 2), float(self.cfg.EXTRA.FOCAL_LENGTH))
        vertices, _ = self.mano(**params)
        params, camera = self.refine_net(image_features, vertices, camera, features, focal)
        vertices, joints = self.mano(**params)
        translation = torch.stack((camera[:, 1], camera[:, 2],
            2 * focal[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * camera[:, 0] + 1e-9)), dim=-1)
        projected = self._projection(joints, translation=translation,
                                     focal_length=focal / self.cfg.MODEL.IMAGE_SIZE)
        return {"params": params, "joints": joints, "vertices": vertices,
                "projected": projected, "camera_translation": translation, "camera": camera}
