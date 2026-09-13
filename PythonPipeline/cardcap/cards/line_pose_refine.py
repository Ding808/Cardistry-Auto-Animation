"""Local pose refinement from explicitly associated image edge segments.

Finite-difference Levenberg-Marquardt with endpoint Huber loss (delta=3 px).
This is a constrained fit within a supplied pose branch, not pose uniqueness,
physical corner verification, independent accuracy, or ambiguity resolution.
"""
from __future__ import annotations

import cv2
import numpy as np


HUBER_DELTA_PX = 3.
MAX_CORNER_MOVEMENT_PX = 40.
MAX_ROTATION_CHANGE_RAD = np.deg2rad(30.)
DEPTH_RATIO_BOUNDS = (.7, 1.3)
RANK_RELATIVE_TOLERANCE = 1e-8
FINITE_DIFFERENCE_STEP = 1e-6


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label + " must have the required shape and finite values")
    return result


def _huber(residuals, weights):
    absolute = np.abs(residuals)
    losses = np.where(absolute <= HUBER_DELTA_PX, .5*residuals**2,
                      HUBER_DELTA_PX*(absolute-.5*HUBER_DELTA_PX))
    robust = np.ones_like(residuals)
    outside = absolute > HUBER_DELTA_PX
    robust[outside] = HUBER_DELTA_PX/absolute[outside]
    return float(weights @ losses), np.sqrt(weights*robust)


def refine_pose_from_segments(initial_candidate, dimensions_m, camera_matrix,
                              distortion, segments, max_iterations=30,
                              corner_prior_sigma_px=None):
    """Fit segment endpoints to their associated projected infinite card lines.

``edge_index=i`` means local corner i -> (i+1)%4, where corners are
[-w/2,-h/2,0], [w/2,-h/2,0], [w/2,h/2,0], [-w/2,h/2,0]. Edge indices are
local model indices, not quad image indices or candidate.corner_indices.
Each input segment contains exactly two pixel points and a positive weight.
Weights multiply the two endpoint Huber losses; they are not pose confidence.
Only a zero-distortion image/camera model is supported. For a lens with nonzero
distortion, first undistort the pixels and supply the matching zero-distortion
camera matrix; this function does not perform that preprocessing. Residuals
use the infinite straight line through the two projected corner endpoints.

At least three distinct edges and weighted local Jacobian rank six are
required. Damping cannot supply missing rank. Translation increments are
normalized by the initial centre depth (metres) for rank/conditioning/LM.
Rejected fits do not expose a usable output pose. Residuals of the last valid
iterate may remain for diagnosis even when convergence was not reached.
Input dictionaries/arrays are not changed.

When corner_prior_sigma_px is supplied, add 0.5*sum((projected_corner_xy -
initial_corner_xy)**2 / sigma**2) to the weighted endpoint Huber data cost.
This is an explicit Gaussian prior on the initial four corner projections,
not additional measured edges. Rank and condition use only the real segment
endpoint Jacobian, before adding the prior or LM damping. None preserves the
unregularized fit. The prior can bias the result toward an imperfect initial
candidate and does not resolve planar/corner/normal ambiguities.
"""
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if corner_prior_sigma_px is not None and (type(corner_prior_sigma_px) not in (int, float)
            or not np.isfinite(corner_prior_sigma_px) or corner_prior_sigma_px <= 0):
        raise ValueError("corner_prior_sigma_px must be None or a finite positive pixel sigma")
    dimensions = _array(dimensions_m, (2,), "dimensions_m")
    if np.any(dimensions <= 0):
        raise ValueError("card dimensions must be positive metres")
    width, height = dimensions
    local = np.array([[-width/2, -height/2, 0.], [width/2, -height/2, 0.],
                      [width/2, height/2, 0.], [-width/2, height/2, 0.]])
    k = _array(camera_matrix, (3, 3), "camera_matrix")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0) or abs(k[0, 1])+abs(k[1, 0]) > 1e-12:
        raise ValueError("camera must have positive focal lengths, zero skew and bottom row [0,0,1]")
    d = np.asarray(distortion, dtype=np.float64)
    if d.ndim > 2 or (d.ndim == 2 and min(d.shape) != 1) or d.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError("distortion must contain 0/4/5/8/12/14 finite coefficients")
    d = d.reshape(-1)
    if np.any(d != 0):
        raise ValueError("line refinement requires zero-distortion pixels/camera; first undistort image segments and supply the matching camera matrix")
    initial = np.r_[_array(initial_candidate["rvec"], (3,), "initial rvec"),
                    _array(initial_candidate["tvec_m"], (3,), "initial tvec_m")]
    initial_r = cv2.Rodrigues(initial[:3])[0]
    if "rotation_matrix" in initial_candidate and not np.allclose(
            _array(initial_candidate["rotation_matrix"], (3, 3), "initial rotation_matrix"), initial_r, atol=1e-7, rtol=0):
        raise ValueError("initial rvec and rotation_matrix differ")
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")
    edge_indices, points, weights = [], [], []
    for index, segment in enumerate(segments):
        edge = segment["edge_index"]
        if type(edge) is not int or edge not in range(4):
            raise ValueError("edge_index must be an integer from 0 to 3")
        pixels = _array(segment["points_px"], (2, 2), f"segment {index} points_px")
        if np.linalg.norm(pixels[1]-pixels[0]) <= 1e-6:
            raise ValueError("segment endpoints must be distinct")
        weight = segment["weight"]
        if type(weight) not in (int, float) or not np.isfinite(weight) or weight <= 0:
            raise ValueError("segment weight must be finite and positive")
        edge_indices.extend([edge, edge]); points.extend(pixels); weights.extend([weight, weight])
    edges = np.asarray(edge_indices, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64)
    result = {"accepted": False, "status": "rejected", "rvec": None, "tvec_m": None, "rotation_matrix": None,
        "source_endpoint_residuals_px": None, "optimized_endpoint_residuals_px": None,
        "source_huber_cost": None, "optimized_huber_cost": None,
        "source_data_cost": None, "optimized_data_cost": None,
        "source_prior_cost": 0., "optimized_prior_cost": None,
        "source_total_cost": None, "optimized_total_cost": None,
        "source_jacobian_rank": None, "jacobian_rank": None, "condition_number": None,
        "jacobian_singular_values": None,
        "converged": False, "convergence_status": "not_started", "iterations": 0,
        "accepted_steps": 0, "rejected_trials": 0, "rejection_reasons": [],
        "distinct_model_edges": sorted(set(edge_indices)),
        "ambiguity_resolved": False, "physical_accuracy_measured": False,
        "measurement_label": "fitted_segment_endpoint_to_projected_infinite_line_signed_distance_px",
        "residual_order": "input segments in order, two endpoint residuals per segment",
        "huber_delta_px": HUBER_DELTA_PX,
        "jacobian_parameterization": "rvec radians, translations normalized by initial centre depth; positive-weight Huber IRLS Jacobian",
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "prior_strength": {"enabled": corner_prior_sigma_px is not None,
            "corner_sigma_px": corner_prior_sigma_px,
            "precision_per_coordinate_px_inverse_squared": 0. if corner_prior_sigma_px is None else 1./corner_prior_sigma_px**2,
            "reference": "initial candidate's four projected corners; not additional image measurements",
            "cost": "0.5 * sum((corner_xy - initial_corner_xy)**2 / sigma_px**2)",
            "prior_included_in_observation_rank_or_condition": False},
        "active_constraints": [],
        "active_constraint_near_bound_tolerances": {"corner_movement_px": .4, "centre_depth_ratio": .003, "rotation_degrees": .3, "positive_corner_depth_m": 1e-6},
        "local_limits": {"max_projected_corner_movement_px": MAX_CORNER_MOVEMENT_PX,
            "centre_depth_ratio": list(DEPTH_RATIO_BOUNDS), "max_rotation_change_degrees": 30.,
            "all_corners_positive_depth_required": True,
            "semantics": "protect a local initial branch; not image/metric accuracy bounds"},
        "local_constraint_diagnostics": None,
        "warnings": ["Planar, corner, normal and front/back ambiguities remain unresolved.",
                     "Fit residuals and local rank do not establish line identity, complete face, metric scale or independent accuracy."]}

    def project(parameters):
        r = cv2.Rodrigues(parameters[:3])[0]
        camera_points = local @ r.T + parameters[3:]
        pixels = cv2.projectPoints(local, parameters[:3], parameters[3:], k, d)[0].reshape(4, 2)
        return r, camera_points, pixels

    _, initial_camera, initial_pixels = project(initial)
    if initial[5] <= 1e-8 or np.any(initial_camera[:, 2] <= 1e-8) or not np.isfinite(initial_pixels).all():
        result["rejection_reasons"].append("initial_pose_has_nonpositive_depth_or_nonfinite_projection")
        return result
    parameter_scale = np.r_[np.ones(3), np.full(3, initial[5])]
    result["translation_parameter_scale_m"] = float(initial[5])
    if corner_prior_sigma_px is not None:
        result["warnings"].append("Initial-corner Gaussian prior is active; its stabilization is not new image evidence or independent accuracy.")

    def residual(parameters):
        _, _, projected = project(parameters)
        starts = projected[edges]
        vectors = projected[(edges+1) % 4]-starts
        lengths = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(projected).all() or np.any(lengths <= 1e-8):
            raise ValueError("projected_edge_is_degenerate")
        differences = points-starts
        return (vectors[:, 0]*differences[:, 1]-vectors[:, 1]*differences[:, 0])/lengths

    def prior_residual(parameters):
        if corner_prior_sigma_px is None:
            return np.zeros(0)
        return ((project(parameters)[2]-initial_pixels)/corner_prior_sigma_px).reshape(-1)

    def prior_cost(parameters):
        value = prior_residual(parameters)
        return float(.5 * (value @ value))

    def jacobian(parameters, function=residual):
        columns = []
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = FINITE_DIFFERENCE_STEP*parameter_scale[index]
            columns.append((function(parameters+delta)-function(parameters-delta))/(2*FINITE_DIFFERENCE_STEP))
        return np.column_stack(columns)

    def rank_information(parameters, current_residual):
        j = jacobian(parameters)
        _, root_weights = _huber(current_residual, weights)
        weighted = j*root_weights[:, None]
        singular = np.linalg.svd(weighted, compute_uv=False)
        rank = int(np.count_nonzero(singular > singular[0]*RANK_RELATIVE_TOLERANCE)) if singular.size and singular[0] > 0 else 0
        condition = float(singular[0]/singular[-1]) if rank == 6 else None
        return weighted, root_weights, rank, singular, condition

    def constraints(parameters):
        r, camera_points, projected = project(parameters)
        angle = float(np.arccos(np.clip((np.trace(r @ initial_r.T)-1.)/2., -1., 1.)))
        shift = float(np.max(np.linalg.norm(projected-initial_pixels, axis=1)))
        ratio = float(parameters[5]/initial[5])
        positive = bool(np.all(camera_points[:, 2] > 1e-8))
        diagnostics = {"max_projected_corner_movement_px": shift, "centre_depth_ratio": ratio,
                       "rotation_change_degrees": float(np.rad2deg(angle)), "all_corners_positive_depth": positive,
                       "minimum_corner_depth_m": float(np.min(camera_points[:, 2]))}
        valid = bool(np.isfinite(projected).all() and positive and shift <= MAX_CORNER_MOVEMENT_PX
                     and DEPTH_RATIO_BOUNDS[0] <= ratio <= DEPTH_RATIO_BOUNDS[1]
                     and angle <= MAX_ROTATION_CHANGE_RAD)
        return valid, diagnostics

    try:
        parameters = initial.copy()
        current_residual = residual(parameters)
        result["source_endpoint_residuals_px"] = current_residual.reshape(-1, 2).tolist()
        current_cost, _ = _huber(current_residual, weights)
        result["source_huber_cost"] = current_cost
        result["source_data_cost"] = current_cost
        current_prior_cost = prior_cost(parameters)
        current_total_cost = current_cost + current_prior_cost
        result["source_prior_cost"] = current_prior_cost
        result["source_total_cost"] = current_total_cost
        if len(set(edge_indices)) < 3:
            result["rejection_reasons"].append("fewer_than_three_distinct_model_edges")
            result["convergence_status"] = "insufficient_constraints"
            return result
        weighted_j, root_weights, rank, singular, condition = rank_information(parameters, current_residual)
        result.update(source_jacobian_rank=rank, jacobian_rank=rank,
                      jacobian_singular_values=singular.tolist(), condition_number=condition)
        if rank < 6:
            result["rejection_reasons"].append("local_projection_jacobian_rank_below_six")
            result["convergence_status"] = "insufficient_constraints"
            return result
        damping = 1e-3
        for iteration in range(max_iterations):
            result["iterations"] = iteration+1
            weighted_j, root_weights, rank, singular, condition = rank_information(parameters, current_residual)
            if rank < 6:
                result["rejection_reasons"].append("local_projection_jacobian_rank_below_six")
                result["convergence_status"] = "rank_lost"
                break
            gradient = weighted_j.T @ (root_weights*current_residual)
            normal = weighted_j.T @ weighted_j
            if corner_prior_sigma_px is not None:
                prior_j = jacobian(parameters, prior_residual)
                gradient += prior_j.T @ prior_residual(parameters)
                normal += prior_j.T @ prior_j
            if np.linalg.norm(gradient, ord=np.inf) <= 1e-7*(1+current_total_cost):
                result.update(converged=True, convergence_status="gradient_tolerance")
                break
            diagonal = np.maximum(np.diag(normal), 1e-12)
            accepted_step = False
            for _ in range(12):
                step = np.linalg.solve(normal+damping*np.diag(diagonal), -gradient)
                proposed = parameters+parameter_scale*step
                feasible, _ = constraints(proposed)
                proposed_residual = residual(proposed) if feasible else None
                proposed_cost = _huber(proposed_residual, weights)[0] if feasible else float("inf")
                proposed_prior_cost = prior_cost(proposed) if feasible else float("inf")
                proposed_total_cost = proposed_cost+proposed_prior_cost
                if feasible and proposed_total_cost <= current_total_cost:
                    improvement = current_total_cost-proposed_total_cost
                    parameters, current_residual, current_cost = proposed, proposed_residual, proposed_cost
                    current_prior_cost, current_total_cost = proposed_prior_cost, proposed_total_cost
                    result["accepted_steps"] += 1
                    damping = max(damping/3., 1e-12)
                    accepted_step = True
                    if np.linalg.norm(step, ord=np.inf) <= 1e-8 or (improvement <= 1e-10*(1+current_total_cost) and np.linalg.norm(step, ord=np.inf) <= 1e-5):
                        result.update(converged=True, convergence_status="local_step_cost_tolerance")
                    break
                damping = min(damping*10., 1e12)
                result["rejected_trials"] += 1
            if result["converged"]:
                break
            if not accepted_step:
                result["convergence_status"] = "no_feasible_decreasing_step"
                break
        else:
            result["convergence_status"] = "maximum_iterations"
        _, _, rank, singular, condition = rank_information(parameters, current_residual)
        feasible, constraint_diagnostics = constraints(parameters)
        result.update(jacobian_rank=rank, condition_number=condition,
                      jacobian_singular_values=singular.tolist(), local_constraint_diagnostics=constraint_diagnostics,
                      optimized_endpoint_residuals_px=current_residual.reshape(-1, 2).tolist(), optimized_huber_cost=current_cost,
                      optimized_data_cost=current_cost, optimized_prior_cost=current_prior_cost, optimized_total_cost=current_total_cost)
        active = result["active_constraints"]
        if constraint_diagnostics["max_projected_corner_movement_px"] >= MAX_CORNER_MOVEMENT_PX-.4:
            active.append("maximum_projected_corner_movement")
        if constraint_diagnostics["centre_depth_ratio"] <= DEPTH_RATIO_BOUNDS[0]+.003:
            active.append("minimum_centre_depth_ratio")
        if constraint_diagnostics["centre_depth_ratio"] >= DEPTH_RATIO_BOUNDS[1]-.003:
            active.append("maximum_centre_depth_ratio")
        if constraint_diagnostics["rotation_change_degrees"] >= 30.-.3:
            active.append("maximum_rotation_change")
        if constraint_diagnostics["minimum_corner_depth_m"] <= 1e-6:
            active.append("positive_corner_depth")
        if active:
            result["warnings"].append("A local branch guard is active or near active; numerical convergence can be constraint-limited and does not establish accuracy.")
        if rank < 6 and "local_projection_jacobian_rank_below_six" not in result["rejection_reasons"]:
            result["rejection_reasons"].append("local_projection_jacobian_rank_below_six")
        if not feasible:
            result["rejection_reasons"].append("outside_initial_local_branch_limits")
        if not result["converged"]:
            result["rejection_reasons"].append("optimization_did_not_converge")
        if not result["rejection_reasons"]:
            result.update(accepted=True, status="accepted", rvec=parameters[:3].tolist(),
                          tvec_m=parameters[3:].tolist(), rotation_matrix=cv2.Rodrigues(parameters[:3])[0].tolist())
        return result
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        result["rejection_reasons"].append("numerical_fit_failure: " + str(error))
        result["convergence_status"] = "numerical_failure"
        return result
