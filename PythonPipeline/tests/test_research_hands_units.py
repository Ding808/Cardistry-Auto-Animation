"""Actual licensed neutral MANO serialization; no video inference or accuracy claim."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cardcap.hand.research_hands_glb import PIPELINE, PLUGIN, export_research_hands


def binary_chunk(path):
    data = Path(path).read_bytes()
    json_size = struct.unpack_from('<I', data, 12)[0]
    offset = 20 + json_size
    length, kind = struct.unpack_from('<II', data, offset)
    assert kind == 0x004E4942
    return data[offset + 8:offset + 8 + length]


class ResearchHandsUnitTests(unittest.TestCase):
    def test_unit_declaration_changes_claims_without_changing_actual_geometry(self):
        if not (PIPELINE / 'models/MANO_RIGHT.pkl').is_file():
            self.skipTest('Licensed MANO asset not installed')
        with tempfile.TemporaryDirectory(prefix='cardistry-mesh-units-') as folder:
            base = Path(folder).resolve()
            self.assertFalse(base.is_relative_to(PLUGIN.resolve()))
            conditional = export_research_hands(output_dir=base/'conditional')
            metric = export_research_hands(output_dir=base/'explicit_metric', coordinate_units='ue_centimeters')
            self.assertIsNone(conditional['coordinates']['meters_per_mano_unit'])
            self.assertEqual(conditional['coordinates']['geometry_scale_factor'], 1.)
            self.assertEqual(conditional['coordinates']['coordinate_units'], 'conditional_ue_units')
            self.assertEqual(metric['coordinates']['meters_per_mano_unit'], 1.)
            self.assertEqual(metric['coordinates']['coordinate_units'], 'ue_centimeters')
            self.assertFalse(metric['provenance']['independent_metric_accuracy_verified'])
            for left, right in zip(conditional['outputs'], metric['outputs']):
                self.assertEqual(binary_chunk(left['path']), binary_chunk(right['path']))
                self.assertIsNone(left['hands']['left']['bounds_gltf_m'])
            with np.load(conditional['source_arrays']['path'], allow_pickle=False) as first, np.load(metric['source_arrays']['path'], allow_pickle=False) as second:
                self.assertEqual(first.files, second.files)
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(json.loads(Path(conditional['manifest_path']).read_text())['coordinates'], conditional['coordinates'])

    def test_unknown_coordinate_enum_rejected_before_asset_creation(self):
        with self.assertRaisesRegex(ValueError, 'coordinate_units'):
            export_research_hands(coordinate_units='unknown-but-pretend-metres')

    def test_explicit_external_output_required_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, 'explicit new output_dir'):
            export_research_hands()
        with self.assertRaisesRegex(ValueError, 'outside the installed plugin'):
            export_research_hands(output_dir=PLUGIN/'Content/forbidden-generated-mesh')

    def test_existing_output_is_preserved_before_loading_models(self):
        with tempfile.TemporaryDirectory(prefix='cardistry-existing-mesh-') as folder:
            path = Path(folder)
            marker = path/'existing.bin'
            marker.write_bytes(b'keep previous output')
            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite'):
                export_research_hands(output_dir=path)
            self.assertEqual(marker.read_bytes(), b'keep previous output')


if __name__ == '__main__':
    unittest.main()
