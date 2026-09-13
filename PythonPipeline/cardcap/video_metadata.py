"""Read bounded MP4/QuickTime container metadata using only the standard library.

This reader never estimates a camera. Track presentation dimensions, transforms,
timescales and even explicit focal-length tags are evidence, not complete K.
Atom/data layouts follow Apple's QuickTime File Format documentation:
https://developer.apple.com/documentation/quicktime-file-format
No external executable, codec, EXIF library, or vendor code is invoked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import math
import os
from pathlib import Path
import re
import struct
from typing import BinaryIO


@dataclass(frozen=True)
class MetadataReadLimits:
    """Resource protections; these do not choose or modify output geometry."""
    max_depth: int = 32
    max_atoms: int = 100_000
    max_payload_bytes: int = 4 * 1024 * 1024
    max_metadata_read_bytes: int = 32 * 1024 * 1024
    max_records: int = 100_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class _Malformed(ValueError):
    pass


@dataclass(frozen=True)
class _Atom:
    offset: int
    size: int
    type: bytes
    header_size: int
    path: str

    @property
    def payload_offset(self):
        return self.offset + self.header_size

    @property
    def end(self):
        return self.offset + self.size


def _fourcc(value: bytes) -> str:
    return value.decode("latin-1") if all(x >= 32 for x in value) else "0x" + value.hex()


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _camera_category(key: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if "focal" in normalized and any(x in normalized for x in ("35mm", "35milli", "35film")):
        return "focal_length_35mm_equivalent"
    if "focal" in normalized:
        return "focal_length_or_focal_metadata"
    if "sensor" in normalized and any(x in normalized for x in ("width", "height", "size", "dimension", "crop")):
        return "sensor_geometry"
    if any(x in normalized for x in ("intrinsic", "cameramatrix", "principalpoint", "lensdistortion")):
        return "camera_geometry"
    return None


class _Reader:
    _CONTAINERS = {b"moov", b"mdia", b"minf", b"stbl", b"dinf", b"edts", b"udta", b"tapt"}

    def __init__(self, stream, size, limits, report):
        self.stream, self.size, self.limits, self.report = stream, size, limits, report
        self.bytes_read = 0
        self.saw_moov = False

    def read(self, offset, count):
        if offset < 0 or count < 0 or offset + count > self.size:
            raise _Malformed(f"Read outside source at {offset}+{count}/{self.size}")
        if count > self.limits.max_payload_bytes or self.bytes_read + count > self.limits.max_metadata_read_bytes:
            raise _Malformed(f"Metadata resource limit exceeded at byte {offset}")
        self.stream.seek(offset)
        result = self.stream.read(count)
        self.bytes_read += len(result)
        if len(result) != count:
            raise _Malformed(f"Truncated read at byte {offset}: wanted {count}, got {len(result)}")
        return result

    def atoms(self, start, end, path, depth):
        if depth > self.limits.max_depth:
            raise _Malformed(f"Atom nesting exceeds resource limit at {path}")
        offset = start
        while offset < end:
            if end - offset < 8:
                raise _Malformed(f"Truncated atom header at {path}, byte {offset}")
            if len(self.report["atoms"]) >= self.limits.max_atoms:
                raise _Malformed("Atom count exceeds resource limit")
            size, kind = struct.unpack(">I4s", self.read(offset, 8))
            header = 8
            if size == 1:
                if end - offset < 16:
                    raise _Malformed(f"Truncated extended atom header at byte {offset}")
                size = struct.unpack(">Q", self.read(offset + 8, 8))[0]
                header = 16
            elif size == 0:
                size = end - offset
            if kind == b"uuid":
                header += 16
            if size < header or size > end - offset:
                raise _Malformed(f"Invalid atom size {size} at byte {offset}; parent ends at {end}")
            atom = _Atom(offset, size, kind, header, f"{path}/{_fourcc(kind)}@{offset}")
            self.report["atoms"].append({"path": atom.path, "type_hex": kind.hex(), "offset": offset,
                                         "size": size, "header_size": header})
            yield atom
            offset += size

    def payload(self, atom):
        return self.read(atom.payload_offset, atom.end - atom.payload_offset)

    def prefix(self, atom, count):
        if atom.end - atom.payload_offset < count:
            raise _Malformed(f"Truncated {atom.path}: need {count} payload bytes")
        return self.read(atom.payload_offset, count)

    def unparsed(self, atom, reason, metadata_possible=True):
        item = {"path": atom.path, "offset": atom.payload_offset, "size": atom.end - atom.payload_offset,
                "reason": reason, "may_contain_camera_metadata": metadata_possible}
        if atom.type == b"uuid":
            item["uuid_hex"] = self.read(atom.payload_offset - 16, 16).hex()
        self.report["unparsed_regions"].append(item)

    def record(self, atom, key, value, raw, encoding, **extra):
        if len(self.report["metadata_records"]) >= self.limits.max_records:
            raise _Malformed("Metadata record count exceeds resource limit")
        entry = {"key": key, "value": value, "encoding": encoding, "atom_path": atom.path,
                 "atom_offset": atom.offset, "raw_value_base64": base64.b64encode(raw).decode("ascii"),
                 "raw_value_sha256": hashlib.sha256(raw).hexdigest(), **extra}
        index = len(self.report["metadata_records"])
        self.report["metadata_records"].append(entry)
        category = _camera_category(key)
        if category:
            self.report["camera_fields"].append({"record_index": index, "category": category,
                "key": key, "value": value, "atom_path": atom.path, "atom_offset": atom.offset,
                "sufficient_for_intrinsics": False,
                "reason": "Raw metadata only; units, video crop, principal point and camera applicability are not established"})

    def track_header(self, atom, track):
        version = self.prefix(atom, 4)[0]
        if version not in (0, 1):
            self.unparsed(atom, f"Unsupported tkhd version {version}")
            return
        data = self.prefix(atom, 84 if version == 0 else 96)
        track_id_offset, matrix_offset = (12, 40) if version == 0 else (20, 52)
        matrix = list(struct.unpack_from(">9i", data, matrix_offset))
        matrix_float = [v / (2 ** (30 if i in (2, 5, 8) else 16)) for i, v in enumerate(matrix)]
        a, b, u, c, d, v, x, y, w = matrix
        # Exact integer tests preserve non-rotation/shear information instead of rounding it away.
        rotation = None
        if u == v == 0 and w == 2 ** 30 and a * c + b * d == 0 and a * a + b * b == c * c + d * d > 0 and a * d - b * c > 0:
            rotation = math.degrees(math.atan2(b, a))
        result = {"atom_path": atom.path, "version": version,
                  "track_id": struct.unpack_from(">I", data, track_id_offset)[0],
                  "presentation_width": struct.unpack_from(">I", data, matrix_offset + 36)[0] / 2 ** 16,
                  "presentation_height": struct.unpack_from(">I", data, matrix_offset + 40)[0] / 2 ** 16,
                  "matrix_raw_signed": matrix, "matrix_row_major": matrix_float,
                  "rotation_degrees": rotation,
                  "rotation_semantics": "atan2(b,a) of the track presentation matrix; not camera orientation or gravity",
                  "dimension_semantics": "tkhd presentation dimensions; not sensor dimensions or necessarily decoded raster dimensions"}
        track.setdefault("track_headers", []).append(result)

    def media_header(self, atom, track):
        version = self.prefix(atom, 4)[0]
        if version not in (0, 1):
            self.unparsed(atom, f"Unsupported mdhd version {version}")
            return
        data = self.prefix(atom, 24 if version == 0 else 36)
        offset = 12 if version == 0 else 20
        timescale = struct.unpack_from(">I", data, offset)[0]
        duration = struct.unpack_from(">I" if version == 0 else ">Q", data, offset + 4)[0]
        if timescale == 0:
            raise _Malformed(f"Zero mdhd timescale at {atom.path}")
        unknown_duration = duration == 2 ** (32 if version == 0 else 64) - 1
        track.setdefault("media_headers", []).append({"atom_path": atom.path, "version": version,
            "timescale": timescale, "duration_ticks_raw": duration,
            "duration_seconds": None if unknown_duration else duration / timescale,
            "semantics": "Media timescale/duration; does not establish original capture speed, edit timing, frame rate or shutter exposure"})

    def keys(self, atom):
        data = self.payload(atom)
        if len(data) < 8 or data[:4] != b"\0\0\0\0":
            raise _Malformed(f"Unsupported or truncated metadata keys header at {atom.path}")
        count = struct.unpack_from(">I", data, 4)[0]
        if count > self.limits.max_records or count > (len(data) - 8) // 8:
            raise _Malformed(f"Metadata keys count exceeds payload at {atom.path}")
        result, offset = {}, 8
        for index in range(1, count + 1):
            size = struct.unpack_from(">I", data, offset)[0]
            if size < 8 or offset + size > len(data):
                raise _Malformed(f"Invalid metadata key size at {atom.path}+{offset}")
            namespace = _fourcc(data[offset + 4:offset + 8])
            raw = data[offset + 8:offset + size]
            try:
                key = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise _Malformed(f"Invalid UTF-8 metadata key at {atom.path}+{offset}")
            result[index] = (namespace, key)
            self.report["metadata_keys"].append({"index": index, "namespace": namespace, "key": key,
                "atom_path": atom.path, "offset": atom.payload_offset + offset,
                "raw_key_base64": base64.b64encode(raw).decode("ascii")})
            if _camera_category(key):
                self.report["camera_fields"].append({"record_index": None, "category": _camera_category(key),
                    "key": key, "value": None, "atom_path": atom.path,
                    "atom_offset": atom.offset, "evidence_kind": "key_declared_value_not_established",
                    "sufficient_for_intrinsics": False,
                    "reason": "Declared metadata key; a declaration alone does not supply a camera value"})
            offset += size
        if offset != len(data):
            raise _Malformed(f"Trailing bytes after metadata keys at {atom.path}")
        return result

    def data_value(self, atom, key, namespace):
        data = self.payload(atom)
        if len(data) < 8:
            raise _Malformed(f"Truncated metadata data atom at {atom.path}")
        kind, locale = struct.unpack_from(">II", data)
        raw = data[8:]
        value, encoding = None, "opaque"
        try:
            if kind == 1:
                value, encoding = raw.decode("utf-8"), "UTF-8"
            elif kind == 2:
                value, encoding = raw.decode("utf-16-be"), "UTF-16-BE"
            elif kind in (21, 22) and len(raw) in (1, 2, 3, 4, 8):
                value, encoding = int.from_bytes(raw, "big", signed=kind == 21), "signed integer" if kind == 21 else "unsigned integer"
            elif kind in (23, 24) and len(raw) == (4 if kind == 23 else 8):
                candidate = struct.unpack(">f" if kind == 23 else ">d", raw)[0]
                value, encoding = (candidate if math.isfinite(candidate) else None), "IEEE float" if kind == 23 else "IEEE double"
        except UnicodeDecodeError:
            encoding = "invalid declared text encoding; raw preserved"
        if value is None:
            self.unparsed(atom, f"Metadata value type {kind} not decoded or nonfinite/invalid value")
        self.record(atom, key, value, raw, encoding, namespace=namespace, data_type=kind, locale=locale)

    def ilst(self, atom, keys, depth):
        for item in self.atoms(atom.payload_offset, atom.end, atom.path, depth + 1):
            index = int.from_bytes(item.type, "big")
            namespace, key = keys.get(index, ("fourcc", _fourcc(item.type)))
            children = list(self.atoms(item.payload_offset, item.end, item.path, depth + 2))
            if item.type == b"----":
                names = {}
                for child in children:
                    if child.type in (b"mean", b"name"):
                        data = self.payload(child)
                        if len(data) < 4:
                            raise _Malformed(f"Truncated freeform name at {child.path}")
                        names[child.type] = data[4:].decode("utf-8", errors="replace")
                        self.record(child, _fourcc(child.type), names[child.type], data[4:], "UTF-8 freeform name")
                namespace, key = names.get(b"mean", "freeform"), names.get(b"name", "unnamed freeform")
            elif namespace == "fourcc" and item.type[0] == 0:
                self.report["warnings"].append(f"Unresolved metadata key index {index} at {item.path}")
                self.unparsed(item, "Metadata key index not mapped to a keys entry")
            found = False
            for child in children:
                if child.type == b"data":
                    self.data_value(child, key, namespace)
                    found = True
                elif child.type not in (b"mean", b"name"):
                    self.unparsed(child, "Unsupported metadata item child")
            if not found:
                self.unparsed(item, "No supported data atom in metadata item")

    def meta(self, atom, depth):
        first = self.prefix(atom, 4)
        # ISO meta is a FullBox; classic QuickTime meta starts directly with hdlr.
        if first == b"\0\0\0\0":
            start, layout = atom.payload_offset + 4, "ISO FullBox version 0 flags 0"
        else:
            start, layout = atom.payload_offset, "QuickTime non-FullBox"
        children = list(self.atoms(start, atom.end, atom.path, depth + 1))
        key_atoms = [x for x in children if x.type == b"keys"]
        if len(key_atoms) > 1:
            raise _Malformed(f"Ambiguous duplicate metadata keys atoms at {atom.path}")
        keys = self.keys(key_atoms[0]) if key_atoms else {}
        self.report["parsed_metadata_scopes"].append({"path": atom.path, "layout": layout,
            "offset": start, "size": atom.end - start, "key_count": len(keys)})
        for child in children:
            if child.type == b"ilst":
                self.ilst(child, keys, depth + 1)
            elif child.type == b"hdlr":
                self.handler(child, None)
            elif child.type == b"keys":
                pass
            else:
                self.unparsed(child, "Unsupported meta child", child.type != b"free")

    def handler(self, atom, track):
        data = self.prefix(atom, 12)
        result = {"atom_path": atom.path, "handler_type": _fourcc(data[8:12]),
                  "component_type": _fourcc(data[4:8])}
        self.report["handlers"].append(result)
        if track is not None:
            track.setdefault("handlers", []).append(result)
        if data[8:12] in (b"meta", b"mdta") and track is not None:
            self.report["warnings"].append("Timed metadata track exists; its media samples are not interpreted")

    def vendor(self, atom):
        count = atom.end - atom.payload_offset
        if count > self.limits.max_payload_bytes:
            self.unparsed(atom, "Vendor payload exceeds per-atom read limit; payload not loaded")
            return
        raw = self.payload(atom)
        if atom.type == b"cprt" and len(raw) >= 6 and raw[:4] == b"\0\0\0\0":
            language = struct.unpack_from(">H", raw, 4)[0]
            text_bytes = raw[6:]
            if text_bytes.endswith(b"\0"):
                text_bytes = text_bytes[:-1]
            try:
                text = text_bytes.decode("utf-8")
                encoding = "ISO copyright UTF-8 notice"
            except UnicodeDecodeError:
                text, encoding = None, "Copyright text encoding not decoded; raw preserved"
                self.unparsed(atom, "Undecoded copyright notice text")
            self.record(atom, "cprt", text, raw, encoding, language_raw=language)
            return
        if atom.type.startswith(b"\xa9") and len(raw) >= 4:
            length, language = struct.unpack_from(">HH", raw)
            if length <= len(raw) - 4:
                text = raw[4:4 + length].decode("utf-8", errors="replace")
                self.record(atom, _fourcc(atom.type), text, raw, "QuickTime legacy user data; text encoding not verified", language=language)
                if length != len(raw) - 4:
                    self.unparsed(atom, "Additional legacy user data records not interpreted")
                return
        try:
            text = raw.decode("utf-8")
            if any(ord(c) < 32 and c not in "\r\n\t\0" for c in text):
                text = None
        except UnicodeDecodeError:
            text = None
        self.record(atom, _fourcc(atom.type), text, raw, "UTF-8-looking vendor payload" if text is not None else "opaque vendor payload")
        self.unparsed(atom, "Vendor schema not interpreted; retained raw bytes are not a complete EXIF/JSON/XML/vendor-tag parse")

    def walk(self, start, end, path="", depth=0, track=None, in_udta=False):
        for atom in self.atoms(start, end, path, depth):
            kind = atom.type
            if kind == b"mdat":
                self.unparsed(atom, "Media payload not read by metadata parser; only included in whole-file SHA256")
            elif kind == b"trak":
                next_track = {"atom_path": atom.path}
                self.report["tracks"].append(next_track)
                self.walk(atom.payload_offset, atom.end, atom.path, depth + 1, next_track)
            elif kind in self._CONTAINERS:
                self.saw_moov |= kind == b"moov"
                if kind == b"udta":
                    self.report["parsed_metadata_scopes"].append({"path": atom.path, "layout": "udta child headers; vendor raw values retained",
                        "offset": atom.payload_offset, "size": atom.end - atom.payload_offset})
                self.walk(atom.payload_offset, atom.end, atom.path, depth + 1, track, kind == b"udta")
            elif kind == b"meta":
                self.meta(atom, depth)
            elif kind == b"tkhd" and track is not None:
                self.track_header(atom, track)
            elif kind == b"mdhd" and track is not None:
                self.media_header(atom, track)
            elif kind == b"hdlr":
                self.handler(atom, track)
            elif kind == b"ftyp":
                data = self.payload(atom)
                if len(data) < 8 or (len(data) - 8) % 4:
                    raise _Malformed(f"Malformed ftyp at {atom.path}")
                self.report["file_types"].append({"atom_path": atom.path, "major_brand": _fourcc(data[:4]),
                    "minor_version": struct.unpack_from(">I", data, 4)[0],
                    "compatible_brands": [_fourcc(data[i:i + 4]) for i in range(8, len(data), 4)]})
            elif in_udta or kind == b"uuid":
                self.vendor(atom)
            else:
                self.unparsed(atom, "Atom payload outside supported container-metadata layouts", kind not in (b"free", b"skip", b"wide"))


def read_video_metadata(video_path: str | Path, *, limits: MetadataReadLimits | None = None) -> dict:
    """Return JSON-safe raw metadata and one of three camera-evidence statuses.

    ``fields_absent`` only says no recognized camera field was found in supported,
    successfully parsed metadata scopes. It does NOT assert absence in opaque
    vendor data, compressed sample descriptions, timed samples, or media payloads.
    Whole-source SHA256 streams all bytes separately; the metadata traversal seeks
    over mdat without reading/searching its payload. All file access is read-only.
    """
    limits = limits or MetadataReadLimits()
    path = Path(video_path).expanduser()
    report = {"schema_version": 1, "status": "unreadable", "intrinsics": None,
        "reader": {"name": "cardcap.video_metadata", "version": "1.0", "dependencies": "Python standard library only",
                   "path": str(Path(__file__).resolve()), "sha256": None, "limits": asdict(limits)},
        "source": {"path": str(path), "resolved_path": None, "size_bytes": None, "sha256": None, "unchanged_during_read": None},
        "file_types": [], "tracks": [], "handlers": [], "atoms": [], "metadata_keys": [],
        "metadata_records": [], "camera_fields": [], "parsed_metadata_scopes": [], "unparsed_regions": [],
        "warnings": [], "errors": [], "metadata_bytes_read": 0,
        "fields_absent_scope": "Recognized focal/sensor/intrinsics keys in supported parsed meta/keys/ilst and udta records only",
        "limitations": ["Not a complete decoder or validator of MP4/QuickTime media samples",
                        "Unknown/vendor payloads, codec metadata and timed metadata samples may contain unrecognized camera information",
                        "No sensor dimensions, focal length, crop, distortion, principal point, calibration or K are inferred from presentation dimensions"]}
    reader = None
    try:
        with Path(__file__).open("rb") as own:
            report["reader"]["sha256"] = _hash_stream(own)
        report["source"]["resolved_path"] = str(path.resolve())
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            report["source"]["size_bytes"] = before.st_size
            report["source"]["sha256"] = _hash_stream(stream)
            reader = _Reader(stream, before.st_size, limits, report)
            reader.walk(0, before.st_size)
            if not reader.saw_moov:
                raise _Malformed("No readable moov container; cannot establish metadata scope")
            after = os.fstat(stream.fileno())
            unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            report["source"]["unchanged_during_read"] = unchanged
            if not unchanged:
                raise _Malformed("Source changed during metadata inspection")
            report["status"] = "fields_present_insufficient" if report["camera_fields"] else "fields_absent"
    except (OSError, ValueError, struct.error, OverflowError, RecursionError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["status"] = "unreadable"
    finally:
        if reader is not None:
            report["metadata_bytes_read"] = reader.bytes_read
    report["scope_complete"] = report["status"] != "unreadable"
    report["all_possible_camera_metadata_locations_exhausted"] = False
    return report
