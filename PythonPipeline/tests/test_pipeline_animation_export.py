"""Mock orchestration only; no source video, model inference or real animation."""
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import time
import unittest

import numpy as np
import run_pipeline as pipeline


class PipelineAnimationExportTests(unittest.TestCase):
    def test_optional_animation_flag_calls_export_after_observations_default_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            for enabled in (False,True):
                with self.subTest(export_animation=enabled), ExitStack() as stack:
                    output = Path(temporary)/str(enabled)
                    argv = ['--video','synthetic-mock-only.mp4','--output',str(output),'--backend','wilor']
                    if enabled:
                        argv.append('--export-animation')
                    args = pipeline.parser().parse_args(argv)
                    spec = SimpleNamespace(id='primary',video=Path('synthetic-mock-only.mp4'),offset_seconds=0.)
                    view = SimpleNamespace(spec=spec,fps=30.,frame_count=1,width=8,height=8,
                        camera={'calibrated':False},metadata={'id':'primary'},
                        read=Mock(return_value=np.zeros((8,8,3),np.uint8)),close=Mock())
                    estimator = SimpleNamespace(backend_info={'test_stub':True},estimate=Mock(return_value=[]),close=Mock())
                    writer = SimpleNamespace(path=output/'overlay_00.mp4',write=Mock(),close=Mock())
                    export = Mock(return_value={'test_stub_only':True,'capture_path':str(output/'animation/capture.cardcap.json')})
                    module = ModuleType('cardcap.prepare_animation')
                    module.export_cardcap = export
                    stack.enter_context(patch.dict(sys.modules,{'cardcap.prepare_animation':module}))
                    for name,value in {'load_views':([spec],'primary'),'VideoView':view,'create_estimator':estimator,
                                       'sha256_file':'mock-test-no-real-file','OverlayWriter':writer,
                                       'measure_blur':(100.,False),'summarize_frames':{'primary':{'valid_frames_at_least_one_hand':1}},
                                       'verify_video':{'test_stub_only':True},'contact_sheet':None,
                                       'overlay_frame':np.zeros((8,8,3),np.uint8)}.items():
                        stack.enter_context(patch.object(pipeline,name,return_value=value))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(pipeline.run(args),0)
                    self.assertTrue((output/'hand_observations.json').exists())
                    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['status'],'completed')
                    if enabled:
                        export.assert_called_once_with(output/'hand_observations.json',output/'animation')
                        self.assertTrue(report['animation_export']['test_stub_only'])
                    else:
                        export.assert_not_called()
                        self.assertIsNone(report['animation_export'])
            unsupported = pipeline.parser().parse_args(['--video','synthetic-mock-only.mp4','--output',str(Path(temporary)/'reject'),'--export-animation'])
            with self.assertRaisesRegex(ValueError,'requires --backend wilor'):
                pipeline.run(unsupported)
            self.assertFalse((Path(temporary)/'reject').exists())


if __name__ == '__main__':
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineAnimationExportTests))
    report = {'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,
              'test_kind':'Mock orchestration, not real video inference or animation validation',
              'run_pipeline_sha256':hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest(),
              'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = Path(__file__).resolve().parents[1]/'evidence/bug001/one_command_flow_test.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
