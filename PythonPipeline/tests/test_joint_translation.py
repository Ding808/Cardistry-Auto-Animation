"""Synthetic solver mathematics and actual neutral MANO solid checks.

No test claims real-video pose/scale accuracy. No model or GPU inference runs.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import sys
import time
import unittest

import cv2
import numpy as np

from cardcap.hand.mano_assets import load_mano_arrays
from cardcap.solve import joint_translation as solver

ROOT = Path(__file__).resolve().parents[1]
METRICS = {}


def configuration(**overrides):
    value = json.loads((ROOT.parent / 'Config/HandSolve.json').read_text(encoding='utf-8'))
    value.update(window_frames=3, window_stride=2, max_function_evaluations=100)
    value.update(overrides)
    return value


def cube():
    vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float) * .01
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7], [0,1,5],[0,5,4],
                      [3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
    return vertices, faces


def fixture(n=5):
    index = np.arange(21)
    local = np.column_stack(((index % 3 - 1) * .016, (index // 3) * .012,
                             np.sin(index * .7) * .006))
    joints = np.broadcast_to(local, (n, 2, 21, 3)).copy()
    truth = np.broadcast_to(np.array([[-.12,.01,.65],[.14,-.02,.70]]), (n,2,3)).copy()
    truth += np.arange(n)[:,None,None] * np.array([.001,-.0005,.0008])
    camera = {'intrinsics': {'fx':850., 'fy':930., 'cx':640., 'cy':360.},
              'distortion': [.03,-.005,.001,-.002,0.]}
    k = np.array([[850.,0,640.],[0,930.,360.],[0,0,1.]])
    targets = project(joints, truth, k, camera['distortion'])
    vertices, faces = cube()
    vertices = np.broadcast_to(vertices, (n,2,8,3)).copy()
    return joints, truth, targets, np.ones((n,2),bool), camera, vertices, [faces, faces], k


def project(joints, translation, k, distortion):
    return np.array([[cv2.projectPoints(joints[i,s],np.zeros(3),translation[i,s],k,
                                        np.asarray(distortion))[0].reshape(21,2)
                      for s in range(2)] for i in range(len(joints))])


def expected_filled_translations(optimized, observed):
    """Independent np.interp oracle: final observed anchors, endpoint extension."""
    filled = np.empty_like(optimized)
    for side in range(2):
        anchors = np.flatnonzero(observed[:, side])
        for axis in range(3):
            filled[:, side, axis] = np.interp(np.arange(len(optimized)), anchors, optimized[anchors, side, axis])
    return filled


class JointTranslationTests(unittest.TestCase):
    def test_known_two_hand_translations_recovered_in_one_camera(self):
        joints, truth, targets, observed, camera, vertices, faces, k = fixture()
        initial = truth + np.array([.018,-.012,.075])
        source = initial.copy()
        actual, report = solver.solve_joint_translations(joints, initial, targets, observed, camera,
                                                        vertices_m=vertices, faces=faces, config=configuration())
        error = float(np.max(np.abs(actual - truth)))
        pixel_error = float(np.max(np.linalg.norm(project(joints,actual,k,camera['distortion'])-targets,axis=-1)))
        self.assertLess(error, 1e-6)
        self.assertLess(pixel_error, 1e-3)
        np.testing.assert_array_equal(initial, source)
        self.assertTrue(all(s['success'] for s in report['steps']))
        self.assertEqual([(s['start'],s['end_exclusive']) for s in report['steps'] if s['temporal']], [(0,3),(2,5)])
        METRICS['known_synthetic_translation'] = {'max_coordinate_error_m':error,'max_reprojection_error_px':pixel_error}

    def test_missing_target_pixels_are_not_measurements(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(3)
        observed[1,1] = False
        initial = truth + np.array([.013,-.008,.055])
        missing_nan = targets.copy()
        missing_nan[1,1] = np.nan
        missing_finite = targets.copy()
        missing_finite[1,1] = 1e12
        first, _ = solver.solve_joint_translations(joints,initial,missing_nan,observed,camera,
                                                  vertices_m=vertices,faces=faces,config=configuration())
        second, _ = solver.solve_joint_translations(joints,initial,missing_finite,observed,camera,
                                                   vertices_m=vertices,faces=faces,config=configuration())
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        # n=1 and n=2 have no eligible second difference; no boundary wraparound.
        for n in (1,2):
            _, report = solver.solve_joint_translations(joints[:n],truth[:n],targets[:n],np.ones((n,2),bool),camera,
                                                        vertices_m=vertices[:n],faces=faces,config=configuration())
            self.assertEqual(len(report['steps']), n+1)
        METRICS['missing_target'] = {'nan_vs_arbitrary_finite_solution_difference_m':0.,'short_window_lengths_checked':[1,2]}

    def test_missing_translations_use_final_optimized_observation_anchors(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(10)
        observed[:] = False
        observed[[2, 7], 0] = True
        observed[[0, 5, 9], 1] = True
        targets[~observed] = np.nan
        initial = truth + np.array([.013, -.008, .055])
        # Unobserved initial translations deliberately disagree with interpolation.
        initial[~observed] += np.array([.23, -.16, .3])
        result, _ = solver.solve_joint_translations(joints, initial, targets, observed, camera,
            vertices_m=vertices, faces=faces, config=configuration())
        expected = expected_filled_translations(result, observed)
        np.testing.assert_allclose(result, expected, rtol=0, atol=2e-15)
        self.assertGreater(float(np.max(np.abs(result[observed] - initial[observed]))), .001)
        self.assertTrue(np.isfinite(result).all())
        METRICS['final_anchor_fill'] = {'frames': 10, 'observed_anchors': [[2,7],[0,5,9]],
            'maximum_interpolation_or_hold_difference': float(np.max(np.abs(result-expected))),
            'compared_to_final_optimized_anchors': True}

    def test_long_endpoint_hold_and_single_observation_have_no_free_missing_translation(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(40)
        observed[:] = False
        observed[:5, 0] = True
        observed[2, 1] = True
        targets[~observed] = np.nan
        initial = truth + np.array([.013, -.008, .055])
        initial[~observed] += np.array([.5, -1.5, .2])
        result, report = solver.solve_joint_translations(joints, initial, targets, observed, camera,
            vertices_m=vertices, faces=faces, config=configuration(window_frames=5, window_stride=3))
        np.testing.assert_allclose(result, expected_filled_translations(result, observed), rtol=0, atol=2e-15)
        np.testing.assert_array_equal(result[5:,0], np.broadcast_to(result[4,0], (35,3)))
        np.testing.assert_array_equal(result[:,1], np.broadcast_to(result[2,1], (40,3)))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(all(step['success'] for step in report['steps']))
        METRICS['long_hold'] = {'frames':40,'left_trailing_hold_frames':35,
            'right_observation_frames':[2], 'right_constant_whole_take':True}

    def test_hand_without_any_observation_is_rejected_before_optimization(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(3)
        for sides in ((0,), (1,), (0,1)):
            mask = observed.copy()
            mask[:, sides] = False
            pixels = targets.copy()
            pixels[~mask] = np.nan
            with self.subTest(missing_sides=sides), patch.object(solver, 'least_squares') as optimize:
                with self.assertRaises(ValueError):
                    solver.solve_joint_translations(joints, truth, pixels, mask, camera,
                        vertices_m=vertices, faces=faces, config=configuration())
                optimize.assert_not_called()

    def test_near_plane_guard_includes_dependent_missing_pose(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(3)
        observed[1:,0] = False
        targets[~observed] = np.nan
        # The observed anchor pose itself is in front. A different missing pose
        # extends behind that anchor's camera plane and must not be exported.
        vertices[2,0,:,2] -= 1.
        initial = truth.copy()
        initial[2,0,2] = 2.  # Old free missing translation would conceal the issue.
        with patch.object(solver,'least_squares') as optimize:
            with self.assertRaisesRegex(ValueError,'near plane'):
                solver.solve_joint_translations(joints,initial,targets,observed,camera,
                    vertices_m=vertices,faces=faces,config=configuration())
            optimize.assert_not_called()

    def test_analytic_jacobian_soft_priors_and_window_boundaries(self):
        joints, _, _, observed, camera, vertices, faces, k = fixture(5)
        initial = np.broadcast_to(np.array([[.02,.01,.7],[.034,.013,.702]]),(5,2,3)).copy()
        # Equal x/y/z accelerations; actual cube intersection is nonzero.
        initial += (.0005*np.arange(5)**2)[:,None,None]
        targets = project(joints,initial,k,camera['distortion']) + .2
        observed[2,1] = False
        targets[2,1] = np.nan
        captured = []

        def inspect(fun, x0, *, jac, **kwargs):
            analytic = jac(x0)
            numeric = np.empty_like(analytic)
            step = 2e-6
            for column in range(len(x0)):
                change = np.zeros_like(x0)
                change[column] = step
                numeric[:,column] = (fun(x0+change)-fun(x0-change))/(2*step)
            np.testing.assert_allclose(analytic,numeric,rtol=3e-5,atol=2e-3)
            residual = fun(x0)
            captured.append({'residual':residual,'jacobian':analytic,
                             'max_jacobian_absolute_error':float(np.max(np.abs(analytic-numeric)))})
            return SimpleNamespace(x=x0.copy(),success=True,status=1,nfev=1,
                                   cost=float(.5*np.dot(residual,residual)),optimality=0.)

        with patch.object(solver,'least_squares',inspect):
            _, report = solver.solve_joint_translations(joints,initial,targets,observed,camera,
                                                        vertices_m=vertices,faces=faces,config=configuration())
        ratios = []
        for entry, detail in zip(captured,report['steps']):
            start,end = detail['start'],detail['end_exclusive']
            expected_dependent = set(range(start,end))
            # Right frame 2 is halfway between anchors 1 and 3. Either active
            # anchor must include its off-window volume and temporal dependency.
            if start <= 1 < end or start <= 3 < end:
                expected_dependent.add(2)
            self.assertEqual(detail['dependent_frames'], sorted(expected_dependent))
            self.assertEqual(len(detail['active_translation_anchors']),int(observed[start:end].sum()))
            self.assertEqual(entry['jacobian'].shape[1],int(observed[start:end].sum())*3)
            data_count = int(observed[start:end].sum())*42 + len(expected_dependent)
            if not detail['temporal']:
                self.assertEqual(len(entry['residual']),data_count)
                continue
            centers = sorted({c for f in expected_dependent for c in (f-1,f,f+1) if 1 <= c < 4})
            self.assertEqual(detail['temporal_centers'], centers)
            temporal = entry['residual'][data_count:].reshape(-1,3)
            self.assertEqual(len(temporal),len(centers)*2)
            self.assertLess(float(np.max(np.abs(temporal))),3.)  # Huber quadratic region.
            nonzero = np.abs(temporal[:,0]) > 1e-12
            # Linear missing-frame interpolation can have exactly zero acceleration.
            np.testing.assert_allclose(temporal[~nonzero], 0., atol=1e-10)
            ratio = temporal[nonzero,2]**2 / temporal[nonzero,0]**2
            np.testing.assert_allclose(ratio,3.,rtol=1e-10,atol=1e-10)
            ratios.extend(ratio.tolist())
        METRICS['jacobian_and_priors'] = {
            'max_absolute_jacobian_error':max(c['max_jacobian_absolute_error'] for c in captured),
            'depth_to_x_quadratic_cost_ratios':ratios,
            'checked':'nonzero lens distortion projection, real cube-volume gradient, overlapping windows with fixed outside neighbors',
            'note':'Threefold depth cost is the quadratic Huber region coefficient; Huber tails are not globally a constant threefold ratio.'}

    def test_actual_left_right_mano_solid_intersection_and_translation_invariance(self):
        # Actual official neutral arrays, synthetic overlapping placement; no video pose claim.
        models = [load_mano_arrays(ROOT/'models'/f'MANO_{side}.pkl') for side in ('LEFT','RIGHT')]
        vertices = np.stack([m['v_template'] for m in models])
        faces = [m['f'] for m in models]
        source_vertices = vertices.copy()
        volume = solver.IntersectionVolumes(vertices[None],faces)
        overlap = volume.volume_cm3(0,np.zeros(3))
        single = [s.volume()*1e6 for s in volume.solids[0]]
        self.assertGreater(overlap,0.)
        self.assertLessEqual(overlap,min(single)+1e-6)
        common = np.array([.4,-.3,.8])
        translated = solver.IntersectionVolumes((vertices+common)[None],faces).volume_cm3(0,np.zeros(3))
        swapped = solver.IntersectionVolumes(vertices[::-1][None],faces[::-1]).volume_cm3(0,np.zeros(3))
        self.assertAlmostEqual(overlap,translated,delta=1e-6)
        self.assertAlmostEqual(overlap,swapped,delta=1e-6)
        self.assertEqual(volume.volume_cm3(0,np.array([1.,0,0])),0.)
        np.testing.assert_array_equal(vertices,source_vertices)
        METRICS['actual_neutral_mano_solids'] = {'left_cm3':single[0],'right_cm3':single[1],
            'overlap_cm3':overlap,'common_translation_difference_cm3':abs(overlap-translated),
            'swap_difference_cm3':abs(overlap-swapped),'separated_cm3':0.,
            'meaning':'Actual MANO neutral geometry; synthetic placement, not video hand intersection accuracy.'}

    def test_dual_targets_validate_layout_and_only_include_finite_observed_hand_blocks(self):
        joints, truth, targets, observed, camera, vertices, faces, _ = fixture(2)
        observed[1,1] = False
        targets[1,1] = np.nan
        prior = targets.copy()
        prior[0,0,7,0] = np.nan  # One incomplete 21-point model block must be omitted.
        prior[1,1] = 1e12  # Finite but the hand was not observed: must also be omitted.
        initial = truth + np.array([.008,-.003,.025])
        cfg = configuration(model_projection_consistency_weight=1.)
        lengths = []
        real_least_squares = solver.least_squares

        def inspect(fun,x0,**kwargs):
            lengths.append(len(fun(x0)))
            return real_least_squares(fun,x0,**kwargs)

        with patch.object(solver,'least_squares',inspect):
            actual, _ = solver.solve_joint_translations(joints,initial,targets,observed,camera,
                vertices_m=vertices,faces=faces,config=cfg,model_projection_pixels=prior)
        # Each full observed model block adds 42 residuals; n=2 has no temporal rows.
        # Frame 0 right anchor also controls the held right frame 1 volume.
        self.assertEqual(lengths,[128,85,212])
        prior[1,1] = np.nan
        second, _ = solver.solve_joint_translations(joints,initial,targets,observed,camera,
            vertices_m=vertices,faces=faces,config=cfg,model_projection_pixels=prior)
        np.testing.assert_array_equal(actual,second)
        with self.assertRaisesRegex(ValueError,'same frame/hand/joint layout'):
            solver.solve_joint_translations(joints,initial,targets,observed,camera,
                vertices_m=vertices,faces=faces,config=cfg,model_projection_pixels=prior[:,:,:20])
        METRICS['dual_target_masks'] = {'per_pass_residual_counts':lengths,
            'unobserved_finite_vs_nan_solution_difference_m':0.,
            'partial_nonfinite_model_hand_block':'entire 21-point block omitted',
            'invalid_joint_layout_rejected':True,
            'model_points_are_independent_measurements':False}


if __name__ == '__main__':
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JointTranslationTests))
    directory = ROOT/'evidence/bug001-joint-translation'
    directory.mkdir(parents=True,exist_ok=True)
    report = {'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,
              'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'solver_source_sha256':hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
              'metrics':METRICS,'real_video_accuracy_validated':False}
    (directory/'tests_dual.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
