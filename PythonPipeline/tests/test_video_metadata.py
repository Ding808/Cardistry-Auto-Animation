"""Container evidence and hostile-boundary tests; no inferred camera ground truth."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardcap.video_metadata import MetadataReadLimits, read_video_metadata


def atom(kind, payload=b"", *, extended=False, to_end=False):
    if isinstance(kind, str):
        kind = kind.encode("latin-1")
    if extended:
        return struct.pack(">I4sQ", 1, kind, 16 + len(payload)) + payload
    return struct.pack(">I4s", 0 if to_end else len(payload) + 8, kind) + payload


def track(version=0, rotation=False, shear=False):
    data = bytearray(84 if version == 0 else 96)
    data[0] = version
    struct.pack_into(">I", data, 12 if version == 0 else 20, 7)
    offset = 40 if version == 0 else 52
    matrix = [0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30] if rotation else [65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30]
    if shear:
        matrix[3] = 32768
    struct.pack_into(">9iII", data, offset, *matrix, 720 << 16, 1280 << 16)
    mdhd = bytearray(24 if version == 0 else 36)
    mdhd[0] = version
    struct.pack_into(">II" if version == 0 else ">IQ", mdhd, 12 if version == 0 else 20, 600, 3700)
    hdlr = atom("hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    return atom("trak", atom("tkhd", bytes(data)) + atom("mdia", atom("mdhd", bytes(mdhd)) + hdlr))


def data_atom(raw, datatype=1):
    return atom("data", struct.pack(">II", datatype, 0) + raw)


def keyed_metadata(fields, *, fullbox=True, reverse=False, values=True):
    keys = atom("keys", b"\0" * 4 + struct.pack(">I", len(fields)) + b"".join(
        struct.pack(">I", len(key.encode("utf-8")) + 8) + b"mdta" + key.encode("utf-8") for key, _, _ in fields))
    ilst = atom("ilst", b"".join(atom(struct.pack(">I", i), data_atom(raw, dtype))
                                  for i, (_, raw, dtype) in enumerate(fields, 1))) if values else b""
    hdlr = atom("hdlr", b"\0" * 8 + b"mdta" + b"\0" * 12)
    return atom("meta", (b"\0" * 4 if fullbox else b"") + hdlr + (ilst + keys if reverse else keys + ilst))


class VideoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "真实元数据.mp4"

    def read(self, content, **kwargs):
        self.path.write_bytes(content)
        result = read_video_metadata(self.path, **kwargs)
        self.assertEqual(self.path.read_bytes(), content)
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(content).hexdigest())
        self.assertIsNone(result["intrinsics"])
        self.assertFalse(result["all_possible_camera_metadata_locations_exhausted"])
        json.dumps(result, allow_nan=False)
        return result

    def test_valid_tracks_both_versions_and_no_camera_fields(self):
        for version in (0, 1):
            with self.subTest(version=version):
                result = self.read(atom("ftyp", b"isom\0\0\0\0isommp42") + atom("moov", track(version)))
                self.assertEqual(result["status"], "fields_absent")
                tk = result["tracks"][0]["track_headers"][0]
                self.assertEqual((tk["track_id"], tk["presentation_width"], tk["presentation_height"]), (7, 720, 1280))
                self.assertEqual(tk["rotation_degrees"], 0)
                self.assertEqual(result["tracks"][0]["media_headers"][0]["timescale"], 600)
                self.assertAlmostEqual(result["tracks"][0]["media_headers"][0]["duration_seconds"], 185 / 30)
                self.assertTrue(result["source"]["unchanged_during_read"])

    def test_rotation_and_shear_not_guessed_as_rotation(self):
        rotated = self.read(atom("moov", track(rotation=True)))
        self.assertEqual(rotated["tracks"][0]["track_headers"][0]["rotation_degrees"], 90)
        sheared = self.read(atom("moov", track(shear=True)))
        self.assertIsNone(sheared["tracks"][0]["track_headers"][0]["rotation_degrees"])

    def test_iso_and_quicktime_keys_order_and_raw_focal_evidence(self):
        fields = [("com.vendor.FocalLengthIn35mmFormat", b"26", 1), ("sensor_width_mm", struct.pack(">d", 7.2), 24)]
        for fullbox in (True, False):
            for reverse in (True, False):
                result = self.read(atom("moov", track() + keyed_metadata(fields, fullbox=fullbox, reverse=reverse)))
                self.assertEqual(result["status"], "fields_present_insufficient")
                records = result["metadata_records"]
                self.assertEqual([x["value"] for x in records], ["26", 7.2])
                self.assertEqual(base64.b64decode(records[0]["raw_value_base64"]), b"26")
                self.assertTrue(all(not x["sufficient_for_intrinsics"] for x in result["camera_fields"]))

    def test_declared_key_without_value_is_not_absent(self):
        result = self.read(atom("moov", keyed_metadata([("FocalLength", b"", 1)], values=False)))
        self.assertEqual(result["status"], "fields_present_insufficient")
        self.assertEqual(result["metadata_records"], [])
        self.assertIsNone(result["camera_fields"][0]["value"])

    def test_freeform_key_and_multiple_localized_values(self):
        item = atom("----", atom("mean", b"\0" * 4 + b"com.camera") + atom("name", b"\0" * 4 + b"FocalLength")
                    + data_atom(b"4.5") + data_atom("4.5 mm".encode("utf-16-be"), 2))
        result = self.read(atom("moov", atom("udta", atom("meta", b"\0" * 4 + atom("ilst", item)))))
        self.assertEqual(result["status"], "fields_present_insufficient")
        self.assertEqual([x["value"] for x in result["metadata_records"] if x["key"] == "FocalLength"], ["4.5", "4.5 mm"])

    def test_copyright_and_unknown_vendor_blobs_are_preserved(self):
        raw = b"\0\0\0\0\x15\xc7{\"WXVer\":123}\0"
        result = self.read(atom("moov", atom("udta", atom("cprt", raw) + atom("abcd", b"\xff\x00\x01"))))
        self.assertEqual(result["status"], "fields_absent")
        self.assertEqual(result["metadata_records"][0]["value"], '{"WXVer":123}')
        self.assertEqual(base64.b64decode(result["metadata_records"][0]["raw_value_base64"]), raw)
        self.assertTrue(any("Vendor schema" in x["reason"] for x in result["unparsed_regions"]))

    def test_legacy_quicktime_text(self):
        result = self.read(atom("moov", atom("udta", atom(b"\xa9nam", struct.pack(">HH", 5, 0) + b"title"))))
        self.assertEqual(result["metadata_records"][0]["value"], "title")

    def test_mdat_is_not_searched_for_camera_strings(self):
        payload = b"FocalLength=123" * 100_000
        result = self.read(atom("moov", track()) + atom("mdat", payload))
        self.assertEqual(result["status"], "fields_absent")
        self.assertLess(result["metadata_bytes_read"], 1000)
        regions = [x for x in result["unparsed_regions"] if "/mdat@" in x["path"]]
        self.assertEqual(regions[0]["size"], len(payload))

    def test_extended_and_to_parent_end_atoms(self):
        result = self.read(atom("moov", track(), extended=True) + atom("free", b"padding", to_end=True))
        self.assertEqual(result["status"], "fields_absent")
        self.assertEqual(result["atoms"][0]["header_size"], 16)

    def test_uuid_retains_user_type(self):
        value = bytes(range(16))
        result = self.read(atom("moov", atom("uuid", value + b"vendor")))
        self.assertEqual(result["unparsed_regions"][0]["uuid_hex"], value.hex())

    def test_unresolved_index_does_not_claim_all_metadata_exhausted(self):
        result = self.read(atom("moov", atom("meta", b"\0" * 4 + atom("ilst", atom(b"\0\0\0\x05", data_atom(b"26"))))))
        self.assertEqual(result["status"], "fields_absent")
        self.assertTrue(result["warnings"])
        self.assertTrue(any("not mapped" in x["reason"] for x in result["unparsed_regions"]))

    def test_invalid_text_and_nonfinite_values_remain_raw(self):
        result = self.read(atom("moov", keyed_metadata([("FocalLength", b"\xff", 1), ("sensor_width", struct.pack(">d", float("nan")), 24)])))
        self.assertEqual(result["status"], "fields_present_insufficient")
        self.assertEqual([x["value"] for x in result["metadata_records"]], [None, None])
        self.assertEqual(len(result["unparsed_regions"]), 2)

    def test_missing_or_non_movie_source_is_unreadable(self):
        self.assertEqual(read_video_metadata(self.path)["status"], "unreadable")
        for content in (b"", b"garbage", atom("ftyp", b"isom\0\0\0\0")):
            self.assertEqual(self.read(content)["status"], "unreadable")

    def test_truncated_and_oversized_atom_headers(self):
        cases = [struct.pack(">I4s", 4, b"moov"), struct.pack(">I4s", 100, b"moov"),
                 struct.pack(">I4s", 1, b"moov"), struct.pack(">I4sQ", 1, b"moov", 2 ** 64 - 1),
                 atom("moov", b"abc"), atom("moov", struct.pack(">I4s", 100, b"udta")),
                 atom("moov", atom("uuid", b"short"))]
        for content in cases:
            with self.subTest(content=content.hex()):
                result = self.read(content)
                self.assertEqual(result["status"], "unreadable")
                self.assertFalse(result["scope_complete"])

    def test_truncated_known_payloads_and_zero_timescale(self):
        zero_mdhd = atom("mdhd", b"\0" * 24)
        for child in (atom("tkhd", b"\0" * 83), atom("mdia", zero_mdhd), atom("mdia", atom("hdlr", b"\0" * 11))):
            self.assertEqual(self.read(atom("moov", atom("trak", child)))["status"], "unreadable")

    def test_bad_keys_table_size_count_and_utf8(self):
        payloads = [b"\0" * 7, b"\0" * 4 + struct.pack(">I", 2 ** 32 - 1),
                    b"\0" * 4 + struct.pack(">II", 1, 7) + b"mdta",
                    b"\0" * 4 + struct.pack(">II", 1, 99) + b"mdta",
                    b"\0" * 4 + struct.pack(">II", 1, 9) + b"mdta\xff"]
        for payload in payloads:
            self.assertEqual(self.read(atom("moov", atom("meta", b"\0" * 4 + atom("keys", payload))))["status"], "unreadable")

    def test_duplicate_keys_and_truncated_data_atom(self):
        key = atom("keys", b"\0" * 8)
        for children in (key + key, atom("ilst", atom("test", atom("data", b"\0" * 7)))):
            self.assertEqual(self.read(atom("moov", atom("meta", b"\0" * 4 + children)))["status"], "unreadable")

    def test_resource_limits_and_oversized_vendor_skip(self):
        nested = atom("udta")
        for _ in range(8):
            nested = atom("moov", nested)
        self.assertEqual(self.read(nested, limits=MetadataReadLimits(max_depth=3))["status"], "unreadable")
        self.assertEqual(self.read(atom("moov", atom("free") * 4), limits=MetadataReadLimits(max_atoms=2))["status"], "unreadable")
        self.assertEqual(self.read(atom("moov", track()), limits=MetadataReadLimits(max_metadata_read_bytes=30))["status"], "unreadable")
        metadata = keyed_metadata([("FocalLength", b"a" * 1000, 1)])
        self.assertEqual(self.read(atom("moov", metadata), limits=MetadataReadLimits(max_payload_bytes=100))["status"], "unreadable")
        skipped = self.read(atom("moov", atom("udta", atom("test", b"x" * 1000))), limits=MetadataReadLimits(max_payload_bytes=100))
        self.assertEqual(skipped["status"], "fields_absent")
        self.assertTrue(any("exceeds" in x["reason"] for x in skipped["unparsed_regions"]))

    def test_invalid_limits_are_not_silently_replaced(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MetadataReadLimits(max_depth=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
