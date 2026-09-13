"""Independent 1.3 regression requirements; synthetic data, no model inference.

Codec tests encode real MP4/PNG fixtures but do not claim real hand or UE quality.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cardcap import editor_job as job
from cardcap.packet_contract import ContractError, validate_capture
from tests.test_packet_contract import MAPPING, fixture12, packet
from tests.test_editor_job import Fixture, write_json


def local_fixture():
    data = fixture12()
    data['format_version'] = '1.3'
    data['camera'].update(intrinsics=None, distortion=None, calibrated=False)
    data['provenance'].update(camera_intrinsics='unobservable',
                              camera_distortion='unobservable', coordinate_frame='per_hand_wrist_local')
    data['validity'].update(camera_intrinsics=False, camera_distortion=False, inter_hand_transform=False)
    frame = data['hands'][0]['frames'][0]
    frame['global_trans_cm'] = None
    frame['joint_positions_cm'] = [[0., 0., 0.]] + [[float(i), .2*i, -.1*i] for i in range(1, 21)]
    frame['validity']['global_translation'] = False
    return data


def shared_fixture():
    data = fixture12()
    data['format_version'] = '1.3'
    data['provenance']['coordinate_frame'] = 'shared_camera'
    data['validity']['inter_hand_transform'] = True
    data['hands'][0]['frames'][0]['validity']['global_translation'] = True
    return data


class LocalWorkerFixture(Fixture):
    """Fake UE orchestration with real PNGs; MP4 generation intentionally off."""
    def __init__(self, directory, *, tiny_side=None):
        super().__init__(directory, render=False)
        self.request['display_view_mode'] = 'per_hand_local'
        write_json(self.root / 'request.json', self.request)
        self.tiny_side = tiny_side

    def capture(self):
        data = super().capture()
        local = local_fixture()
        for key in ('format_version', 'camera', 'scale', 'provenance', 'validity'):
            data[key] = deepcopy(local[key])
        data['scale']['global_scale_factor_applied'] = 1.
        for hand in data['hands']:
            for frame in hand['frames']:
                frame.update(global_trans_cm=None, joint_positions_cm=deepcopy(local['hands'][0]['frames'][0]['joint_positions_cm']),
                             sample_kind='detected_model_observation', validity={'pose': True, 'global_translation': False})
        return data

    def reconstruct(self):
        super().reconstruct()
        manifest_path = job.expected_research_dir(self.plugin, self.root / 'pipeline/animation') / 'MANO_ResearchHands_Manifest.json'
        manifest = job.read_json(manifest_path)
        manifest['coordinates'] = {'meters_per_mano_unit': None, 'geometry_scale_factor': 1.,
                                   'coordinate_units': 'conditional_ue_units'}
        write_json(manifest_path, manifest)

    def rendered_frames(self, real_images=False):
        import cv2
        import numpy as np
        super().rendered_frames(real_images=True)
        render_path = self.root / 'UE_Frames/sequence_render.json'
        render = job.read_json(render_path)
        render['view_mode'] = 'per_hand_local'
        write_json(render_path, render)
        for i in range(3):
            for side, folder in (('left', ''), ('right', 'overview')):
                image = np.zeros((48, 64, 3), np.uint8)
                if side == self.tiny_side:
                    image[24, 32] = [180, 180, 180]
                else:
                    image[10:38, 22:42] = [120, 170, 200]
                (self.root / 'UE_Frames' / folder / f'frame_{i:06d}.png').write_bytes(cv2.imencode('.png', image)[1].tobytes())


class LocalPoseContractTests(unittest.TestCase):
    def rejected(self, data, path):
        before = deepcopy(data)
        with self.assertRaises(ContractError) as caught:
            validate_capture(data, MAPPING)
        self.assertIn(path, str(caught.exception))
        self.assertEqual(data, before)

    def test_local_pose_without_global_space_remains_consumable_and_unmodified(self):
        data = local_fixture()
        original = deepcopy(data)
        result = validate_capture(data, MAPPING)
        self.assertEqual(result['format_version'], '1.3')
        self.assertEqual(result['hand_sample_counts'], {'left': 1})
        self.assertFalse(result['measurement_truth_or_physical_identity_validated'])
        self.assertEqual(data, original)
        self.assertIsNone(data['camera']['intrinsics'])
        self.assertIsNone(data['camera']['distortion'])
        self.assertFalse(data['validity']['inter_hand_transform'])
        frame = data['hands'][0]['frames'][0]
        self.assertIsNone(frame['global_trans_cm'])
        self.assertFalse(frame['validity']['global_translation'])
        self.assertEqual(frame['joint_positions_cm'][0], [0, 0, 0])
        self.assertNotEqual(frame['joint_positions_cm'][1], [0, 0, 0])

    def test_local_unknown_translation_cannot_be_replaced_by_any_numeric_zero_or_offset(self):
        for value in ([0, 0, 0], [0., -0., 0.], [1, 2, 3], [0, 0, None]):
            with self.subTest(value=value):
                data = local_fixture()
                data['hands'][0]['frames'][0]['global_trans_cm'] = value
                self.rejected(data, '.global_trans_cm')

    def test_local_masks_cannot_claim_inter_hand_or_global_translation_knowledge(self):
        for path in (('validity', 'inter_hand_transform'),
                     ('hands', 0, 'frames', 0, 'validity', 'global_translation')):
            for value in (True, None, 0, 'false'):
                with self.subTest(path=path, value=value):
                    data = local_fixture()
                    target = data
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    self.rejected(data, path[-1])

    def test_new_masks_and_coordinate_frame_are_required_even_when_unknown(self):
        for path in (('validity', 'inter_hand_transform'), ('provenance', 'coordinate_frame'),
                     ('hands', 0, 'frames', 0, 'validity', 'global_translation')):
            with self.subTest(path=path):
                data = local_fixture()
                target = data
                for key in path[:-1]:
                    target = target[key]
                del target[path[-1]]
                self.rejected(data, path[-1])

    def test_local_origin_must_be_wrist_centered_not_a_disguised_shared_offset(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                data = local_fixture()
                data['hands'][0]['frames'][0]['joint_positions_cm'][0][axis] = .01
                self.rejected(data, '.joint_positions_cm[0]')

    def test_local_mode_rejects_camera_values_even_with_consistent_provenance_masks(self):
        for field, value in (('intrinsics', {'fx': 640, 'fy': 640, 'cx': 320, 'cy': 240}),
                             ('distortion', [0, 0, 0, 0, 0])):
            with self.subTest(field=field):
                data = local_fixture()
                data['camera'][field] = value
                key = 'camera_' + field
                data['validity'][key] = True
                data['provenance'][key] = 'inferred'
                self.rejected(data, '$.camera')

    def test_local_mode_cannot_publish_full_image_reprojection_including_perfect_zero(self):
        for error in (0, 1.2, 40):
            with self.subTest(error=error):
                data = local_fixture()
                data['quality']['mean_reprojection_error_px'] = error
                self.rejected(data, '$.quality.mean_reprojection_error_px')

    def test_coordinate_frame_typo_and_inconsistent_shared_claim_are_rejected(self):
        for name in ('per_hand_local', 'world', 'PER_HAND_WRIST_LOCAL', '', None, 'shared_camera'):
            with self.subTest(name=name):
                data = local_fixture()
                data['provenance']['coordinate_frame'] = name
                self.rejected(data, 'inter_hand_transform' if name == 'shared_camera' else 'coordinate_frame')

    def test_local_packets_are_rejected_even_if_packet_fields_are_valid(self):
        data = local_fixture()
        data['packets'] = [packet(unknown=True)]
        self.rejected(data, '$.packets')

    def test_shared_13_accepts_numeric_global_zero_but_requires_k_and_matching_mask(self):
        data = shared_fixture()
        data['hands'][0]['frames'][0]['global_trans_cm'] = [0, 0, 0]
        before = deepcopy(data)
        self.assertEqual(validate_capture(data, MAPPING)['status'], 'passed')
        self.assertEqual(data, before)
        data['camera']['intrinsics'] = None
        data['validity']['camera_intrinsics'] = False
        data['provenance']['camera_intrinsics'] = 'unobservable'
        self.rejected(data, '$.camera.intrinsics')

    def test_shared_13_cannot_carry_null_or_falsely_invalid_global_translation(self):
        for translation, mask, error in ((None, False, '.global_trans_cm'), ([0, 0, 1], False, '.validity.global_translation')):
            with self.subTest(translation=translation, mask=mask):
                data = shared_fixture()
                frame = data['hands'][0]['frames'][0]
                frame['global_trans_cm'] = translation
                frame['validity']['global_translation'] = mask
                self.rejected(data, error)

    def test_old_12_does_not_silently_gain_nullable_global_translation(self):
        data = fixture12()
        data['hands'][0]['frames'][0]['global_trans_cm'] = None
        self.rejected(data, '.global_trans_cm')
        # Adding new fields does not opt an old version into changed semantics.
        data = local_fixture()
        data['format_version'] = '1.2'
        self.rejected(data, '.global_trans_cm')


class LocalPosePreviewTests(unittest.TestCase):
    def test_disabling_mp4_still_checks_real_png_display_support_before_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalWorkerFixture(Path(temporary))
            self.assertEqual(fixture.worker().run(), 0)
            status = job.read_json(fixture.root / 'status.json')
            self.assertEqual(status['status'], 'succeeded')
            self.assertIsNone(status['result']['preview_video'])
            self.assertFalse((fixture.root / 'Preview').exists())
            report = job.read_json(fixture.root / 'UE_Frames/display_geometry_review.json')
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['raw_png_sha256']), 6)
            self.assertEqual(report['capture_sha256'], job.sha256(fixture.root / 'pipeline/animation/capture.cardcap.json'))
            self.assertEqual(len(fixture.calls), 4)

    def test_disabling_mp4_cannot_bypass_tiny_hand_gate_for_either_side(self):
        for side in ('left', 'right'):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as temporary:
                fixture = LocalWorkerFixture(Path(temporary), tiny_side=side)
                self.assertEqual(fixture.worker().run(), 1)
                status = job.read_json(fixture.root / 'status.json')
                self.assertEqual(status['status'], 'failed')
                self.assertIn('too small or invisible', status['error']['message'])
                self.assertFalse((fixture.root / 'job_result.json').exists())
                self.assertFalse((fixture.root / 'Preview').exists())
                self.assertEqual(len(fixture.calls), 3)  # Failed before reload or success publication.
                report = job.read_json(fixture.root / 'UE_Frames/display_geometry_review.json')
                self.assertEqual(report['status'], 'failed')

    def make_fixture(self, temporary, *, invisible_side=None, tiny_side=None):
        import cv2
        import numpy as np
        fixture = Fixture(Path(temporary))
        fixture.request['display_view_mode'] = 'per_hand_local'
        write_json(fixture.root / 'request.json', fixture.request)
        source = cv2.VideoWriter(str(fixture.video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
        self.assertTrue(source.isOpened())
        for i in range(3):
            source.write(np.full((48, 64, 3), [60, 100+i*10, 140], np.uint8))
        source.release()
        fixture.reconstruct()
        path = fixture.root / 'pipeline/animation/capture.cardcap.json'
        capture = job.read_json(path)
        capture['format_version'] = '1.3'
        template = local_fixture()
        capture['camera'] = template['camera']
        capture['scale'] = template['scale']
        capture['provenance'] = template['provenance']
        capture['validity'] = template['validity']
        for hand in capture['hands']:
            for frame in hand['frames']:
                frame.update(global_trans_cm=None, joint_positions_cm=deepcopy(template['hands'][0]['frames'][0]['joint_positions_cm']),
                             sample_kind='detected_model_observation', validity={'pose': True, 'global_translation': False})
        validate_capture(capture, fixture.mapping)
        write_json(path, capture)
        fixture.rendered_frames(real_images=True)
        render_path = fixture.root / 'UE_Frames/sequence_render.json'
        render = job.read_json(render_path)
        render['view_mode'] = 'per_hand_local'
        write_json(render_path, render)
        for i in range(3):
            for side, folder in (('left', ''), ('right', 'overview')):
                image = np.zeros((48, 64, 3), np.uint8)
                if side == invisible_side:
                    # A dark blank render must not pass from background intensity alone.
                    image[:] = 32
                elif side == tiny_side:
                    image[24, 32] = [180, 180, 180]
                else:
                    image[10:38, 22:42] = [120, 170, 200]
                png = fixture.root / 'UE_Frames' / folder / f'frame_{i:06d}.png'
                png.write_bytes(cv2.imencode('.png', image)[1].tobytes())
        return fixture

    def test_local_preview_real_codec_declares_separate_views_and_unknown_relative_placement(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            capture_path = fixture.root / 'pipeline/animation/capture.cardcap.json'
            before = job.sha256(capture_path)
            # Spy on labels while retaining actual rasterization/MP4 encoding.
            with patch.object(cv2, 'putText', wraps=cv2.putText) as draw:
                job.build_preview(fixture.root / 'request.json')
            strings = [call.args[1] for call in draw.call_args_list]
            self.assertIn('LEFT hand | independent wrist-local pose', strings)
            self.assertIn('RIGHT hand | independent wrist-local pose', strings)
            self.assertIn('Relative hand placement UNKNOWN | display_only local camera | no scene depth', strings)
            self.assertFalse(any('UE capture camera' in text or 'independent overview' in text for text in strings))
            report = job.read_json(fixture.root / 'Preview/preview.json')
            self.assertEqual(report['decoded_frames'], 3)
            self.assertEqual(report['resolution'], [3840, 720])
            self.assertEqual(report['view_mode'], 'per_hand_local')
            self.assertTrue(report['display_only'])
            self.assertFalse(report['inter_hand_transform_known'])
            self.assertFalse(report['geometry_changed'])
            self.assertEqual(job.sha256(capture_path), before)
            gate = job.read_json(fixture.root / 'Preview/display_geometry_review.json')
            self.assertEqual(gate['status'], 'passed')

    def test_local_preview_cannot_reuse_old_shared_render_even_with_readable_pngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            path = fixture.root / 'UE_Frames/sequence_render.json'
            render = job.read_json(path)
            render['view_mode'] = 'shared_camera'
            write_json(path, render)
            with self.assertRaisesRegex(ValueError, 'separate local hand previews'):
                job.build_preview(fixture.root / 'request.json')
            self.assertFalse((fixture.root / 'Preview/preview.json').exists())

    def test_each_hand_blank_or_one_pixel_render_fails_without_false_success_receipt(self):
        for side in ('left', 'right'):
            for mode in ('invisible_side', 'tiny_side'):
                with self.subTest(side=side, mode=mode), tempfile.TemporaryDirectory() as temporary:
                    fixture = self.make_fixture(temporary, **{mode: side})
                    with self.assertRaisesRegex(ValueError, 'too small or invisible'):
                        job.build_preview(fixture.root / 'request.json')
                    self.assertFalse((fixture.root / 'Preview/preview.json').exists())
                    gate = job.read_json(fixture.root / 'Preview/display_geometry_review.json')
                    self.assertEqual(gate['status'], 'failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
