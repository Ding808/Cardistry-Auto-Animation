"""Joint two-hand translations in one camera, with real mesh-intersection volume."""
from collections import Counter
from pathlib import Path
import json
import time
import cv2
import numpy as np
import manifold3d
from scipy.optimize import least_squares


def closed_hand_solid(vertices, faces):
    """Close the model's wrist boundary for volume queries, leaving render geometry untouched."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    directed = [(int(a), int(b)) for tri in faces for a,b in zip(tri, np.roll(tri,-1))]
    counts = Counter(tuple(sorted(e)) for e in directed)
    if any(n > 2 for n in counts.values()):
        raise ValueError('Nonmanifold hand topology cannot define intersection volume')
    boundary = {a:b for a,b in directed if counts[tuple(sorted((a,b)))] == 1}
    new_faces, new_vertices = faces.tolist(), vertices.tolist()
    while boundary:
        start = next(iter(boundary))
        ring, node = [], start
        while node in boundary:
            ring.append(node)
            node = boundary.pop(node)
            if node == start:
                break
        if node != start or len(ring) < 3:
            raise ValueError('Open or branching hand boundary')
        center = len(new_vertices)
        new_vertices.append(vertices[ring].mean(axis=0).tolist())
        new_faces.extend([[ring[(i+1)%len(ring)], ring[i], center] for i in range(len(ring))])
    v, f = np.asarray(new_vertices, dtype=np.float64), np.asarray(new_faces, dtype=np.uint64)
    signed_volume = np.einsum('ij,ij->i', v[f[:,0]], np.cross(v[f[:,1]], v[f[:,2]])).sum()/6
    if signed_volume < 0:
        f = f[:, ::-1].copy()
    solid = manifold3d.Manifold(manifold3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f)))
    if solid.status() != manifold3d.Error.NoError or solid.volume() <= 0:
        raise ValueError(f'Invalid closed MANO solid: {solid.status()}')
    return solid


class IntersectionVolumes:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices)
        self.bounds = np.stack((self.vertices.min(axis=2), self.vertices.max(axis=2)), axis=2)
        self.solids = [[closed_hand_solid(pair[side], faces[side]) for side in range(2)] for pair in self.vertices]
        self.calls = 0

    def volume_cm3(self, frame, relative_right_translation):
        bounds = self.bounds[frame].copy()
        bounds[1] += relative_right_translation
        if np.any(np.minimum(bounds[0,1],bounds[1,1]) <= np.maximum(bounds[0,0],bounds[1,0])):
            return 0.0
        self.calls += 1
        volume = (self.solids[frame][0] ^ self.solids[frame][1].translate(relative_right_translation)).volume()
        if not np.isfinite(volume) or volume < -1e-12:
            raise ValueError('Invalid mesh Boolean intersection volume')
        return max(0.0, float(volume)) * 1e6


def solve_joint_translations(local_joints_m, initial_wrists_m, target_pixels, observed,
                             camera, *, vertices_m, faces, config=None, model_projection_pixels=None):
    """Input order N,left/right,joint21,xyz; output uses the same input geometry units.

    Only observed target pixels enter reprojection. Fixed shared-shape pose meshes
    define the geometry; translation variables are never applied to UE actors.
    """
    began = time.perf_counter()
    if camera.get('intrinsics') is None:
        raise ValueError('Joint global translation requires full-image camera intrinsics; local crop conventions are not sufficient')
    if config is None:
        config = json.loads((Path(__file__).resolve().parents[3]/'Config/HandSolve.json').read_text(encoding='utf-8'))
    cfg = dict(config)
    joints, initial, targets = map(lambda a: np.asarray(a, dtype=np.float64), (local_joints_m,initial_wrists_m,target_pixels))
    observed = np.asarray(observed, dtype=bool)
    model_targets = np.asarray(model_projection_pixels,dtype=np.float64) if model_projection_pixels is not None else None
    n = len(initial)
    if joints.shape != (n,2,21,3) or initial.shape != (n,2,3) or targets.shape != (n,2,21,2) or observed.shape != (n,2):
        raise ValueError('Invalid two-hand translation arrays')
    if not np.isfinite(joints).all() or not np.isfinite(initial).all() or not np.isfinite(targets[observed]).all():
        raise ValueError('Nonfinite observed translation inputs')
    if model_targets is not None and model_targets.shape != targets.shape:
        raise ValueError('Model projection prior must use the same frame/hand/joint layout')
    c = camera['intrinsics']
    k = np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    distortion = np.asarray(camera['distortion'],dtype=np.float64) if camera.get('distortion') is not None else np.zeros(5)
    volumes = IntersectionVolumes(vertices_m, faces)
    # A filled pose has no independent image-supported translation. Use the same
    # frame-linear interpolation / constant endpoints as the pose-fill policy,
    # throughout optimization (including every residual and its Jacobian).
    supports = [[None for _ in range(n)] for _ in range(2)]
    known_frames = []
    for side in range(2):
        known = np.flatnonzero(observed[:,side])
        if not len(known):
            raise ValueError('Each hand needs at least one observed translation anchor')
        known_frames.append(known)
        for frame in range(n):
            slot = int(np.searchsorted(known,frame))
            if slot == len(known):
                support = [(int(known[-1]),1.)]
            elif known[slot] == frame or slot == 0:
                support = [(int(known[slot]),1.)]
            else:
                before,after = int(known[slot-1]),int(known[slot])
                alpha = (frame-before)/(after-before)
                support = [(before,1.-alpha),(after,alpha)]
            supports[side][frame] = support

    def fill_translations(anchors):
        filled = anchors.copy()
        for side,known in enumerate(known_frames):
            for axis in range(3):
                filled[:,side,axis] = np.interp(np.arange(n),known,anchors[known,side,axis])
        return filled

    t = fill_translations(initial)
    lower = np.full_like(t, -np.inf)
    lower[:,:,2] = -np.minimum(joints[:,:,:,2].min(axis=2),np.asarray(vertices_m)[:,:,:,2].min(axis=2)) + 1e-4
    # Sufficient positivity bound for each convex-combination anchor, covering
    # the local posed vertices of every dependent frame, not only its own pose.
    anchor_lower = lower.copy()
    for side in range(2):
        for frame,support in enumerate(supports[side]):
            for anchor,_ in support:
                anchor_lower[anchor,side,2] = max(anchor_lower[anchor,side,2],lower[frame,side,2])
    if np.any(t[:,:,2] <= lower[:,:,2]) or np.any(t[:,:,2][observed] <= anchor_lower[:,:,2][observed]):
        raise ValueError('Initial camera translation places hand behind near plane')
    weights = np.sqrt([cfg['temporal_weight'],cfg['temporal_weight'],cfg['temporal_weight']*cfg['depth_temporal_weight_ratio']])*100
    projections_weight = np.sqrt(cfg['reprojection_weight'])
    model_weight = np.sqrt(cfg.get('model_projection_consistency_weight',0.))
    volume_weight = np.sqrt(cfg['intersection_volume_weight'])
    steps = []

    def optimize(start, end, temporal):
        nonlocal t
        active = [(frame,side) for frame in range(start,end) for side in range(2) if observed[frame,side]]
        columns = {key:index*3 for index,key in enumerate(active)}
        dependent_frames = sorted({frame for side in range(2) for frame,support in enumerate(supports[side])
                                   if any((anchor,side) in columns for anchor,_ in support)})
        temporal_centers = sorted({center for frame in dependent_frames for center in (frame-1,frame,frame+1)
                                   if 1 <= center < n-1}) if temporal else []
        detail = {'start':start,'end_exclusive':end,'temporal':temporal,
                  'active_translation_anchors':[list(key) for key in active],
                  'dependent_frames':dependent_frames,'temporal_centers':temporal_centers}
        if not active:
            steps.append(dict(detail,success=True,status=0,nfev=0,cost=0.,optimality=0.,
                              skipped_reason='No observed translation anchor in this window'))
            return
        width = len(active)*3
        outside = t.copy()
        cache = {}
        def compute(x):
            if 'x' in cache and np.array_equal(cache['x'],x):
                return cache['residual'],cache['jacobian']
            current = outside.copy()
            for index,(frame,side) in enumerate(active):
                current[frame,side] = x[index*3:index*3+3]
            current = fill_translations(current)
            values, rows = [], []
            def append(residual, entries):
                residual = np.atleast_1d(residual)
                block = np.zeros((len(residual),width))
                for frame,side,derivative in entries:
                    for anchor,weight in supports[side][frame]:
                        col = columns.get((anchor,side))
                        if col is not None:
                            block[:,col:col+3] += np.asarray(derivative).reshape(-1,3)*weight
                values.extend(residual.tolist())
                rows.extend(block)
            for frame in dependent_frames:
                for side in range(2):
                    if (frame,side) in columns:
                        projected, jac = cv2.projectPoints(joints[frame,side],np.zeros(3),current[frame,side],k,distortion)
                        append((projected.reshape(-1,2)-targets[frame,side]).ravel()*projections_weight,
                               [(frame,side,jac[:,3:6]*projections_weight)])
                        if model_targets is not None and model_weight and np.isfinite(model_targets[frame,side]).all():
                            append((projected.reshape(-1,2)-model_targets[frame,side]).ravel()*model_weight,
                                   [(frame,side,jac[:,3:6]*model_weight)])
                delta = current[frame,1]-current[frame,0]
                volume = volumes.volume_cm3(frame,delta)
                residual = np.sqrt(volume)*volume_weight
                gradient = np.zeros(3)
                if volume > 1e-10 and volume_weight:
                    epsilon = cfg['volume_gradient_step_m']
                    for axis in range(3):
                        change = np.eye(3)[axis]*epsilon
                        gradient[axis] = (volumes.volume_cm3(frame,delta+change)-volumes.volume_cm3(frame,delta-change))/(2*epsilon)
                    gradient *= volume_weight/(2*np.sqrt(volume))
                append([residual],[(frame,1,gradient),(frame,0,-gradient)])
            if temporal:
                for center in temporal_centers:
                    for side in range(2):
                        acceleration = (current[center-1,side]-2*current[center,side]+current[center+1,side])*weights
                        append(acceleration,[(center-1,side,np.diag(weights)),(center,side,-2*np.diag(weights)),(center+1,side,np.diag(weights))])
            cache.update(x=x.copy(),residual=np.asarray(values),jacobian=np.asarray(rows))
            return cache['residual'],cache['jacobian']
        x0 = np.concatenate([t[frame,side] for frame,side in active])
        solution = least_squares(lambda x:compute(x)[0],x0,jac=lambda x:compute(x)[1],
            bounds=(np.concatenate([anchor_lower[frame,side] for frame,side in active]),np.full_like(x0,np.inf)),method='trf',loss='huber',
            f_scale=cfg['huber_scale'],max_nfev=cfg['max_function_evaluations'],ftol=1e-7,xtol=1e-7,gtol=1e-7)
        if not np.isfinite(solution.x).all():
            raise RuntimeError('Joint translation solver returned nonfinite values')
        for index,(frame,side) in enumerate(active):
            t[frame,side] = solution.x[index*3:index*3+3]
        t = fill_translations(t)
        steps.append(dict(detail,success=bool(solution.success),status=int(solution.status),
                          nfev=int(solution.nfev),cost=float(solution.cost),optimality=float(solution.optimality)))

    initial_volumes = [volumes.volume_cm3(i,t[i,1]-t[i,0]) for i in range(n)]
    for i in range(n):
        optimize(i,i+1,False)
    per_frame_result = t.copy()
    window = min(cfg['window_frames'],n)
    starts = sorted(set(list(range(0,max(1,n-window+1),cfg['window_stride']))+[max(0,n-window)]))
    for start in starts:
        optimize(start,min(start+window,n),True)
    final_volumes = [volumes.volume_cm3(i,t[i,1]-t[i,0]) for i in range(n)]
    report = {'method':'scipy.optimize.least_squares', 'algorithm':'trf','loss':'huber','configuration':cfg,
              'unknowns':'Observed left/right wrist translations only; one shared shape, fixed per-frame rotations',
              'missing_translation_policy':'Linear interpolation between optimized observed anchors; constant nearest endpoint outside the observed span. Same mapping in residuals, analytic Jacobians, and returned geometry. Filled translations are not recovered observations.',
              'active_translation_anchor_count':int(observed.sum()),
              'reprojection_source':'Actual image detector landmarks when present; legacy model projections explicitly labeled by caller',
              'model_projection_consistency':'Optional same-image S2 model projection prior prevents image detector/model-pose disagreement being explained only by depth. Correlated prediction, not a second independent measurement or ground truth.',
              'distance_residual':'Removed: no assumed physical wrist-distance interval',
              'temporal_residual_units':'input model units times 100, second differences per source frame; conditional display units unless independently anchored',
              'intersection_residual':'sqrt(weight * closed posed-MANO mesh Boolean volume * 1e6); conditional display-unit volume unless independently anchored',
              'distortion_assumption':'ideal pinhole for projection when camera.distortion is null; not measured zero lens distortion',
              'intersection_geometry':'Original posed vertices and faces; wrist boundary capped for solid queries, render mesh unchanged',
              'initial_intersection_volume_cm3':initial_volumes,'final_intersection_volume_cm3':final_volumes,
              'intersection_boolean_calls':volumes.calls,'per_frame_pass':per_frame_result.tolist(),
              'steps':steps,'elapsed_seconds':time.perf_counter()-began,'independent_metric_accuracy_verified':False}
    return t,report
