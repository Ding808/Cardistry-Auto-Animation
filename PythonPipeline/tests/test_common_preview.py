"""Common-display orchestration and real codec tests using synthetic images only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cardcap import editor_job as job
from tests.test_editor_job import Fixture, write_json
from tests.test_local_pose_contract import LocalWorkerFixture


class CommonWorkerFixture(LocalWorkerFixture):
    def __init__(self, directory):
        super().__init__(directory)
        del self.request['display_view_mode']  # The default must require no extra input.
        write_json(self.root / 'request.json', self.request)

    def reconstruct(self):
        super().reconstruct()
        animation = self.root / 'pipeline/animation'
        self.display_path = animation / 'display_space.json'
        self.display = {
            'format_version': 'cardcap.display_space/1.0', 'display_only': True,
            'mode': 'assumed_common_camera', 'capture_sha256': job.sha256(animation / 'capture.cardcap.json'),
            'frame_count': 3, 'fps': 30., 'resolution': [64, 48],
            'display_assumed_focal_px': 1024., 'principal_point_px': [32., 24.],
            'neutral_hand_length_display_units': 20., 'coordinate_basis': 'ue_x_forward_y_right_z_up',
            'coordinate_units': 'conditional_ue_display_units',
            'frames': [{'frame': index, 'hands': {}, 'wrist_distance_over_hand_length': ratio,
                        'geometry_failed_checks': ['wrist_distance_over_hand_length'] if index == 1 else []}
                       for index, ratio in enumerate((.8, 1.8, .9))],
        }
        write_json(self.display_path, self.display)
        validation = job.read_json(animation / 'validation.json')
        validation.update(display_space_config=str(self.display_path), display_space_sha256=job.sha256(self.display_path))
        write_json(animation / 'validation.json', validation)

    def rendered_frames(self, real_images=False):
        super().rendered_frames(real_images=True)
        path = self.root / 'UE_Frames/sequence_render.json'
        render = job.read_json(path)
        render.update(view_mode='common_assumed_display', display_only=True, inter_hand_transform_known=False,
                      display_config_sha256=job.sha256(self.display_path))
        for index, frame in enumerate(render['frames']):
            frame.update(display_assumed_focal_px=1024.,
                         wrist_distance_over_hand_length=self.display['frames'][index]['wrist_distance_over_hand_length'])
        write_json(path, render)


class CommonPreviewTests(unittest.TestCase):
    def test_default_common_view_passes_sidecar_and_keeps_geometry_violations_reviewable(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CommonWorkerFixture(Path(temporary))
            self.assertEqual(fixture.worker().run(), 0)
            bake_args = fixture.calls[2][0]
            self.assertIn('-DisplayConfig=' + str(fixture.display_path), bake_args)
            report = job.read_json(fixture.root / 'UE_Frames/display_geometry_review.json')
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['geometry_failed_frames'], [1])
            self.assertFalse(report['geometry_gate_failures_block_display'])
            state = job.read_json(fixture.root / 'status.json')
            self.assertEqual(state['result']['display_view_mode'], 'common_space')
            self.assertFalse(state['result']['inter_hand_transform_known'])
            capture = job.read_json(fixture.root / 'pipeline/animation/capture.cardcap.json')
            self.assertIsNone(capture['camera']['intrinsics'])
            self.assertIsNone(capture['camera']['distortion'])
            self.assertIsNone(capture['scale']['meters_per_unit'])
            self.assertTrue(all(frame['global_trans_cm'] is None for hand in capture['hands'] for frame in hand['frames']))

    def test_local_option_does_not_pass_display_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalWorkerFixture(Path(temporary))
            self.assertEqual(fixture.worker().run(), 0)
            self.assertFalse(any(arg.startswith('-DisplayConfig=') for args, _ in fixture.calls for arg in args))

    def test_unavailable_crop_display_preserves_local_output_and_explains_default_common_failure(self):
        class UnavailableDisplayFixture(LocalWorkerFixture):
            def reconstruct(self):
                super().reconstruct()
                animation = self.root / 'pipeline/animation'
                validation = job.read_json(animation / 'validation.json')
                validation.update(display_space_status='unavailable', display_space_config=None,
                                  display_space_sha256=None,
                                  display_space_unavailable_reason='A recorded model crop convention is unavailable.')
                write_json(animation / 'validation.json', validation)
                self.source_capture_bytes = (animation / 'capture.cardcap.json').read_bytes()

        for view_mode in ('common_space', 'per_hand_local'):
            with self.subTest(view_mode=view_mode), tempfile.TemporaryDirectory() as temporary:
                fixture = UnavailableDisplayFixture(Path(temporary))
                if view_mode == 'common_space':
                    del fixture.request['display_view_mode']  # Exercise the actual default.
                write_json(fixture.root / 'request.json', fixture.request)
                self.assertEqual(fixture.worker().run(), 1 if view_mode == 'common_space' else 0)
                state = job.read_json(fixture.root / 'status.json')
                capture_path = fixture.root / 'pipeline/animation/capture.cardcap.json'
                self.assertEqual(capture_path.read_bytes(), fixture.source_capture_bytes)
                capture = job.read_json(capture_path)
                self.assertIsNone(capture['camera']['intrinsics'])
                self.assertIsNone(capture['camera']['distortion'])
                self.assertIsNone(capture['scale']['meters_per_unit'])
                self.assertTrue(all(frame['global_trans_cm'] is None for hand in capture['hands'] for frame in hand['frames']))
                self.assertFalse(any(arg.startswith('-DisplayConfig=') for args, _ in fixture.calls for arg in args))
                if view_mode == 'common_space':
                    self.assertEqual(state['status'], 'failed')
                    self.assertIn('Common-space preview is unavailable', state['error']['message'])
                    self.assertIn('Use Separate Local Preview', state['error']['message'])
                    self.assertIn('recorded model crop convention', state['error']['message'])
                    self.assertEqual(len(fixture.calls), 1)  # Stop before any UE operation.
                else:
                    self.assertEqual(state['status'], 'succeeded')
                    self.assertEqual(state['result']['display_view_mode'], 'per_hand_local')
                    self.assertEqual(len(fixture.calls), 4)

    def test_view_mode_rejects_unknown_or_nonstring_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            for value in ('shared_camera', '', None, True, []):
                with self.subTest(value=value):
                    fixture.request['display_view_mode'] = value
                    write_json(fixture.root / 'request.json', fixture.request)
                    with self.assertRaisesRegex(ValueError, 'display_view_mode'):
                        job.read_request(fixture.root / 'request.json')

    def make_codec_fixture(self, temporary):
        import cv2
        import numpy as np
        fixture = CommonWorkerFixture(Path(temporary))
        source = cv2.VideoWriter(str(fixture.video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
        self.assertTrue(source.isOpened())
        for index in range(3):
            source.write(np.full((48, 64, 3), [60, 100 + index * 10, 140], np.uint8))
        source.release()
        fixture.reconstruct()
        fixture.rendered_frames(real_images=True)
        return fixture

    def test_encoded_preview_shows_common_pair_and_framewise_assumptions_without_source_edits(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_codec_fixture(temporary)
            capture_path = fixture.root / 'pipeline/animation/capture.cardcap.json'
            before = capture_path.read_bytes()
            with patch.object(cv2, 'putText', wraps=cv2.putText) as draw:
                job.build_preview(fixture.root / 'request.json')
            labels = [call.args[1] for call in draw.call_args_list]
            self.assertIn('COMMON SPACE | assumed camera | NOT A MEASUREMENT', labels)
            self.assertIn('COMMON SPACE | independent overview | display only', labels)
            self.assertFalse(any('independent wrist-local pose' in label for label in labels))
            self.assertTrue(any('display_assumed_focal_px=1024.00 | wrists/L=1.800 | UNCALIBRATED | REVIEW: 1 geometry gate(s)' == label for label in labels))
            self.assertIn('left: detected', labels)
            self.assertIn('right: detected', labels)
            report = job.read_json(fixture.root / 'Preview/preview.json')
            self.assertEqual(report['decoded_frames'], 3)
            self.assertEqual(report['view_mode'], 'common_assumed_display')
            self.assertEqual(report['display_geometry_failed_frames'], [1])
            self.assertEqual(report['display_config_sha256'], job.sha256(fixture.display_path))
            self.assertTrue(report['display_only'])
            self.assertFalse(report['inter_hand_transform_known'])
            self.assertFalse(report['source_camera_or_metric_scale_changed'])
            self.assertEqual(capture_path.read_bytes(), before)

    def test_stale_binding_or_mismatched_render_annotation_is_rejected(self):
        for mutation in ('hash', 'focal', 'wrist'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_codec_fixture(temporary)
                path = fixture.root / 'UE_Frames/sequence_render.json'
                render = job.read_json(path)
                if mutation == 'hash':
                    render['display_config_sha256'] = '0' * 64
                elif mutation == 'focal':
                    render['frames'][0]['display_assumed_focal_px'] = 512.
                else:
                    render['frames'][0]['wrist_distance_over_hand_length'] = 3.
                write_json(path, render)
                with self.assertRaises(ValueError):
                    job.build_preview(fixture.root / 'request.json')
                self.assertFalse((fixture.root / 'Preview/preview.json').exists())

    def test_source_camera_values_cannot_be_smuggled_into_assumed_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_codec_fixture(temporary)
            path = fixture.root / 'pipeline/animation/capture.cardcap.json'
            capture = job.read_json(path)
            capture['camera']['intrinsics'] = {'fx': 1024, 'fy': 1024, 'cx': 32, 'cy': 24}
            write_json(path, capture)
            with self.assertRaisesRegex(ValueError, 'cannot replace source camera'):
                job.build_preview(fixture.root / 'request.json')

    def test_blank_common_render_fails_even_when_geometry_is_only_a_hypothesis(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_codec_fixture(temporary)
            path = fixture.root / 'UE_Frames/frame_000001.png'
            path.write_bytes(cv2.imencode('.png', np.zeros((48, 64, 3), np.uint8))[1].tobytes())
            with self.assertRaisesRegex(ValueError, 'too small or invisible'):
                job.build_preview(fixture.root / 'request.json')
            self.assertFalse((fixture.root / 'Preview/preview.json').exists())


if __name__ == '__main__':
    unittest.main()
