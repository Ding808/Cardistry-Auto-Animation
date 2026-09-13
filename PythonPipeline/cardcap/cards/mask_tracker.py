"""Reusable SAM2 mask-prompt propagation from any one reference frame.

Prompts describe visible regions, not verified physical packets. A reference
mask is replayed by SAM2 on its conditioning frame; its fixed presence score
is never exposed as a learned prediction. The two propagation directions
have separate tracking memories. No model loads or downloads occur on import.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Callable

from .segment import MODEL_SPECS, SegmentationCancelled, atomic_json, file_sha256, inspect_sam2_availability


def _int(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label}: expected integer >= {minimum}')
    return value


def load_mask_seed_inputs(source_run: Path, seed_file: Path, cancel_check=lambda: False):
    """Validate explicit sources and masks on CPU before loading any model."""
    import cv2
    import numpy as np
    source_run, seed_file = Path(source_run).resolve(), Path(seed_file).resolve()
    report_path, manifest_path = source_run / 'report.json', source_run / 'source_manifest.json'
    upstream = json.loads(report_path.read_text(encoding='utf-8'))
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    seed_config = json.loads(seed_file.read_text(encoding='utf-8'))
    if upstream.get('status') != 'completed':
        raise ValueError('Source image run must be completed')
    if seed_config.get('format_version') != 'cardcap.sam2_mask_seeds/1.0':
        raise ValueError('Unsupported mask seed format')
    if seed_config.get('events') != []:
        raise ValueError('Mask tracker does not execute physical lifecycle events')
    count = _int(source['frame_count'], 'frame_count', 1)
    if upstream['completed_frames'] != count:
        raise ValueError('Source run completed frame count differs')
    width, height = source['resolution']
    _int(width, 'width', 1); _int(height, 'height', 1)
    fps = source['fps']
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if file_sha256(Path(source['source_video'])) != source['source_video_sha256']:
        raise ValueError('Source video hash differs')
    if seed_config['source_video_sha256'] != source['source_video_sha256']:
        raise ValueError('Seeds belong to a different source video')
    seed_frame = _int(seed_config['frame'], 'seed_frame')
    if seed_frame >= count:
        raise ValueError('Seed frame is outside the source timeline')
    rows = source['frames']
    if len(rows) != count:
        raise ValueError('Source JPEG inventory must cover every frame')
    frame_directory = Path(source['frame_directory']).resolve()
    # The pinned SAM2 loader enumerates the entire directory, not this manifest.
    # Match its numeric filename order exactly so an added/unlisted JPEG cannot
    # shift model inputs while we label them with another frame's source hash.
    frame_names = [p.name for p in frame_directory.iterdir()
                   if p.suffix in ('.jpg', '.jpeg', '.JPG', '.JPEG')]
    try:
        numbers = [int(Path(name).stem) for name in frame_names]
        frame_names.sort(key=lambda name: int(Path(name).stem))
    except ValueError as error:
        raise ValueError('SAM2 JPEG filenames must have numeric stems') from error
    if len(set(numbers)) != len(numbers) or frame_names != [row['jpeg_file'] for row in rows]:
        raise ValueError('Actual SAM2 JPEG inventory/order differs from the source manifest')
    for index, row in enumerate(rows):
        if cancel_check():
            raise SegmentationCancelled('Cancelled while checking model input images')
        path = (frame_directory / row['jpeg_file']).resolve()
        if not path.is_relative_to(frame_directory) or row['frame'] != index or file_sha256(path) != row['jpeg_sha256']:
            raise ValueError('Reused source JPEG path/order/hash differs')
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (height, width, 3):
            raise ValueError('Reused source JPEG resolution differs from the source manifest')
    objects = seed_config['objects']
    if not isinstance(objects, list) or not objects:
        raise ValueError('At least one explicit mask seed is required')
    masks, ids = {}, set()
    for seed in objects:
        if cancel_check():
            raise SegmentationCancelled('Cancelled while checking seed masks')
        identity = seed['object_id']
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identity) or identity in masks:
            raise ValueError('Mask object IDs must be unique safe filename tokens')
        sam_id = _int(seed['sam2_object_id'], 'sam2_object_id', 1)
        if sam_id in ids or _int(seed['frame'], 'object.frame') != seed_frame:
            raise ValueError('SAM2 IDs must be unique and all seeds must share the reference frame')
        ids.add(sam_id)
        if any(not isinstance(seed.get(k), str) or not seed[k].strip() for k in ('provenance_kind', 'selection_provenance')):
            raise ValueError('Every seed requires explicit selection provenance')
        path = (seed_file.parent / seed['mask_file']).resolve()
        if not path.is_relative_to(seed_file.parent) or file_sha256(path) != seed['mask_sha256']:
            raise ValueError('Seed mask path/hash differs')
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.shape != (height, width) or not np.isin(decoded, [0, 255]).all():
            raise ValueError('Seed mask must be original-resolution binary PNG')
        area = int(np.count_nonzero(decoded))
        if area == 0 or area != _int(seed['input_area_px'], 'input_area_px', 1):
            raise ValueError('Seed mask must be nonempty with its actual recorded area')
        masks[identity] = decoded > 0
    provenance = {'source_run': str(source_run), 'source_report_sha256': file_sha256(report_path),
                  'source_manifest_sha256': file_sha256(manifest_path),
                  'seed_file': str(seed_file), 'seed_file_sha256': file_sha256(seed_file)}
    return source, seed_config, masks, provenance


def _overlap(first, second):
    import numpy as np
    intersection, union = int(np.count_nonzero(first & second)), int(np.count_nonzero(first | second))
    smaller = min(int(first.sum()), int(second.sum()))
    return {'intersection_px': intersection, 'union_px': union,
            'mask_iou': intersection / union if union else None,
            'intersection_over_smaller_mask': intersection / smaller if smaller else None,
            'is_lifecycle_event': False}


def run_mask_tracking(source_run: Path, seed_file: Path, output_dir: Path, *,
                      models_dir: Path | None = None, variant='tiny', cancel_file: Path | None = None,
                      progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Reuse verified JPEG inputs, copy explicit seeds, propagate and save evidence.

    A nonempty output is never overwritten. Cancellation is cooperative between
    model operations; partially written directed records remain for diagnosis.
    Incomplete runs are not resumable and never report a completed timeline.
    """
    import cv2
    import numpy as np
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Use a new output directory; existing runs are preserved')
    output_dir.mkdir(parents=True, exist_ok=True)
    cancellation = Path(cancel_file) if cancel_file else output_dir / 'cancel.flag'
    started = time.perf_counter()
    report = {'format_version': 'cardcap.sam2_segmentation_report/1.0', 'status': 'running',
              'tracking_kind': 'explicit_mask_seed_bidirectional_video', 'completed_frames': 0,
              'completed_directed_samples': 0, 'physical_packet_count': None,
              'automatic_instance_discovery': False, 'automatic_split_merge_inferred': False,
              'full_face_verification_performed': False, 'resume_supported': False,
              'code_sha256': file_sha256(Path(__file__))}
    torch = predictor = state = None
    directed, merged = [], []

    def check():
        if cancellation.exists():
            raise SegmentationCancelled('Cancelled between model operations')

    def progress(phase, **extra):
        value = {'status': report['status'], 'phase': phase, 'completed_frames': report['completed_frames'],
                 'completed_directed_samples': len(directed), 'elapsed_seconds': time.perf_counter() - started, **extra}
        atomic_json(output_dir / 'progress.json', value)
        if progress_callback:
            progress_callback(value)

    try:
        check()
        progress('validate_inputs')
        source, seeds, masks, provenance = load_mask_seed_inputs(source_run, seed_file, cancellation.exists)
        count, (width, height), seed_frame = source['frame_count'], source['resolution'], seeds['frame']
        objects = seeds['objects']
        sam_ids = [seed['sam2_object_id'] for seed in objects]
        report.update(input_provenance=provenance, frame_count_requested=count, prompted_object_count=len(objects),
                      seed_frame=seed_frame, merged_conditioning_frame_count=1, merged_model_propagation_frame_count=count - 1,
                      directed_sample_count_requested=count + int(seed_frame > 0))
        source['reused_derived_images_from'] = str(Path(source_run).resolve())
        source['reused_source_manifest_sha256'] = provenance['source_manifest_sha256']
        atomic_json(output_dir / 'source_manifest.json', source)
        seeds = deepcopy(seeds)
        for seed in seeds['objects']:
            origin = (Path(seed_file).resolve().parent / seed['mask_file']).resolve()
            target = output_dir / 'seed_masks' / (seed['object_id'] + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(origin.read_bytes())
            seed['mask_file'] = target.relative_to(output_dir).as_posix()
        atomic_json(output_dir / 'seeds_used.json', seeds)
        report.update(seed_config=str(output_dir / 'seeds_used.json'), seed_config_sha256=file_sha256(output_dir / 'seeds_used.json'))
        availability = inspect_sam2_availability(models_dir, variant=variant)
        report['backend'] = availability
        if not availability['structural_ready']:
            raise RuntimeError('SAM2 unavailable: ' + '; '.join(availability['blockers']))
        source_root = Path(availability['source']['path']).resolve()
        sys.path.insert(0, str(source_root))
        import torch as imported_torch
        import sam2
        from sam2.build_sam import build_sam2_video_predictor
        torch = imported_torch
        if Path(sam2.__file__).resolve().parent != source_root / 'sam2':
            raise RuntimeError('Another SAM2 package shadowed the reviewed source')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This configuration requires verified CUDA and BF16 support')
        torch.cuda.reset_peak_memory_stats()
        report['device'] = {'name': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
        overrides = ['++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true',
            '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05',
            '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98',
            '++model.binarize_mask_from_pts_for_mem_enc=true', '++model.fill_hole_area=0',
            '++model.compile_image_encoder=false', '++model.non_overlap_masks=false']
        report['configuration'] = {'variant': variant, 'precision': 'autocast_bfloat16', 'hydra_overrides': overrides,
            'apply_postprocessing': False, 'offload_video_to_cpu': True, 'offload_state_to_cpu': True,
            'non_overlap_masks': False, 'mask_threshold_logit': 0.0,
            'direction_isolation': 'reset_state before each direction; verified cleared object IDs and all per-object tracking output memories',
            'timeline_selection': 'reverse before seed, forward at/after seed; both conditioning records retained when seed>0',
            'seed_score_semantics': 'Supplied mask uses upstream fixed ±10 logits/presence; public learned presence is null on conditioning frame'}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            check(); progress('load_model')
            predictor = build_sam2_video_predictor(availability['config'], availability['checkpoint_path'], device='cuda', mode='eval',
                hydra_overrides_extra=overrides, apply_postprocessing=False, vos_optimized=False)
            if not predictor.use_mask_input_as_output_without_sam:
                raise RuntimeError('Unreviewed mask-conditioning behavior')
            check()
            availability['inference_attempted'] = True
            state = predictor.init_state(video_path=source['frame_directory'], offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
            directions = [('forward', False, count - seed_frame - 1)]
            if seed_frame > 0:
                directions.append(('reverse', True, seed_frame))
            with (output_dir / 'directed_samples.jsonl').open('w', encoding='utf-8') as stream:
                for direction, reverse, maximum in directions:
                    check()
                    predictor.reset_state(state)
                    if state['obj_ids'] or state['output_dict_per_obj'] or state['frames_tracked_per_obj']:
                        raise RuntimeError('Direction reset left previous object tracking state')
                    for seed in objects:
                        check()
                        predictor.add_new_mask(state, frame_idx=seed_frame, obj_id=seed['sam2_object_id'], mask=masks[seed['object_id']])
                    generator = predictor.propagate_in_video(state, start_frame_idx=seed_frame, max_frame_num_to_track=maximum, reverse=reverse)
                    samples = 0
                    while True:
                        check()
                        frame_started = time.perf_counter()
                        try:
                            frame, ids, logits = next(generator)
                        except StopIteration:
                            break
                        torch.cuda.synchronize()
                        inference_ms = (time.perf_counter() - frame_started) * 1000
                        expected = seed_frame - samples if reverse else seed_frame + samples
                        values = logits.detach().float().cpu().numpy()
                        if frame != expected or list(ids) != sam_ids or values.shape != (len(objects), 1, height, width) or not np.isfinite(values).all():
                            raise RuntimeError('Model output frame/object/logits invalid')
                        rows, current_masks = [], {}
                        for index, seed in enumerate(objects):
                            identity, sam_id = seed['object_id'], seed['sam2_object_id']
                            binary = values[index, 0] > 0
                            current_masks[identity] = binary
                            path = output_dir / 'directed_masks' / direction / identity / f'frame_{frame:06d}.png'
                            path.parent.mkdir(parents=True, exist_ok=True)
                            ok, encoded = cv2.imencode('.png', binary.astype(np.uint8) * 255)
                            if not ok:
                                raise RuntimeError('Cannot encode model mask')
                            path.write_bytes(encoded.tobytes())
                            nlabels, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
                            components = sorted([{'area_px': int(s[4]), 'bbox_xyxy_px': [int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3])]} for s in stats[1:]], key=lambda x: x['area_px'], reverse=True)
                            outputs = state['output_dict_per_obj'][state['obj_id_to_idx'][sam_id]]
                            current = outputs['cond_frame_outputs'].get(frame)
                            if current is None:
                                current = outputs['non_cond_frame_outputs'].get(frame)
                            raw = current.get('object_score_logits') if current is not None else None
                            score = None if raw is None else float(raw.detach().float().cpu().reshape(-1)[0])
                            area, conditioning = int(binary.sum()), frame == seed_frame
                            origin = 'supplied_mask_conditioning_reprojection' if conditioning else 'video_model_propagation'
                            row = {'object_id': identity, 'sam2_object_id': sam_id, 'mask_file': path.relative_to(output_dir).as_posix(),
                                'mask_sha256': file_sha256(path), 'area_px': area, 'empty': area == 0,
                                'component_count': nlabels - 1, 'components': components,
                                'mask_logit_min': float(values[index, 0].min()), 'mask_logit_max': float(values[index, 0].max()),
                                'raw_object_presence_logit': None if conditioning else score, 'upstream_object_presence_logit': score,
                                'raw_score_source': '_use_mask_as_output fixed input-derived score' if conditioning else 'pinned compact_current_out.object_score_logits',
                                'score_semantics': 'Input-derived constant, not learned confidence' if conditioning else 'Uncalibrated learned presence; not accuracy or physical identity',
                                'mask_source': origin, 'mask_generation_kind': origin,
                                'seed_indices': [index], 'seed_provenance_kind': seed['provenance_kind'],
                                'seed_candidate_mask_sha256': seed['mask_sha256'], 'seed_candidate_file_sha256': seed.get('candidate_file_sha256'),
                                'mask_quality_score': None, 'full_face_verified': False, 'physical_packet_identity': None,
                                'propagation_direction': direction, 'representation': 'visible_mask_not_verified_full_card_face'}
                            if conditioning:
                                row['input_seed_output_overlap'] = _overlap(masks[identity], binary)
                            rows.append(row)
                        overlaps = [{'object_ids': [a, b], **_overlap(current_masks[a], current_masks[b])} for a, b in combinations(current_masks, 2)]
                        record = {'frame': frame, 'source_frame': frame, 'timestamp_ms': round(frame / source['fps'] * 1000),
                            'time_seconds': frame / source['fps'], 'source_video_sha256': source['source_video_sha256'],
                            'source_jpeg_sha256': source['frames'][frame]['jpeg_sha256'], 'propagation_direction': direction,
                            'inference_ms': inference_ms, 'prompted_object_count': len(rows), 'physical_packet_count': None,
                            'inter_track_overlaps': overlaps, 'objects': rows}
                        stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()
                        directed.append(record); samples += 1
                        report['completed_directed_samples'] = len(directed)
                        progress('propagate', direction=direction, last_frame=frame)
                    if samples != maximum + 1:
                        raise RuntimeError('Propagation ended before completing a direction')
            predictor.reset_state(state)
        merged = sorted([r for r in directed if (r['frame'] < seed_frame and r['propagation_direction'] == 'reverse') or
                         (r['frame'] >= seed_frame and r['propagation_direction'] == 'forward')], key=lambda r: r['frame'])
        if [r['frame'] for r in merged] != list(range(count)) or len(directed) != report['directed_sample_count_requested']:
            raise RuntimeError('Merged timeline mismatch')
        (output_dir / 'frames.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in merged), encoding='utf-8')
        report['seed_direction_consistency'] = []
        seed_rows = [row for row in directed if row['frame'] == seed_frame]
        for index, seed in enumerate(objects):
            entry = {'object_id': seed['object_id'], 'direction_count': len(seed_rows), 'consistency_is_tracking_accuracy': False}
            if len(seed_rows) == 2:
                binaries = [cv2.imdecode(np.frombuffer((output_dir / row['objects'][index]['mask_file']).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE) > 0 for row in seed_rows]
                entry.update(direction_mask_overlap=_overlap(*binaries), different_pixels=int(np.count_nonzero(binaries[0] != binaries[1])))
            report['seed_direction_consistency'].append(entry)
        report.update(status='completed', completed_frames=count)
        availability['actual_inference_completed'] = True
    except SegmentationCancelled as error:
        report.update(status='cancelled', error=str(error))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        (output_dir / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if torch is not None and torch.cuda.is_available():
            report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        if predictor is not None and state is not None:
            try:
                predictor.reset_state(state)
            except Exception as error:
                report['cleanup_error'] = str(error)
        state = predictor = None
        report['elapsed_seconds'] = time.perf_counter() - started
        report['empty_mask_samples'] = sum(obj['empty'] for row in merged for obj in row['objects'])
        report['multi_component_samples'] = sum(obj['component_count'] > 1 for row in merged for obj in row['objects'])
        for name in ('frames', 'directed_samples'):
            path = output_dir / (name + '.jsonl')
            report[name + '_jsonl_sha256'] = file_sha256(path) if path.is_file() else None
        atomic_json(output_dir / 'report.json', report)
        progress('finished')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--variant', choices=MODEL_SPECS, default='tiny')
    parser.add_argument('--cancel-file', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        source, seeds, _, provenance = load_mask_seed_inputs(args.source_run, args.seeds)
        print(json.dumps({'valid_inputs': True, 'frames': source['frame_count'], 'objects': len(seeds['objects']),
                          'seed_frame': seeds['frame'], 'model_inference': False, 'provenance': provenance}, indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required when running inference')
    result = run_mask_tracking(args.source_run, args.seeds, args.output, models_dir=args.models_dir, variant=args.variant,
                               cancel_file=args.cancel_file)
    print(json.dumps({key: result.get(key) for key in ('status', 'completed_frames', 'elapsed_seconds', 'error')}, indent=2))
    return 0 if result['status'] == 'completed' else (130 if result['status'] == 'cancelled' else 3)


if __name__ == '__main__':
    raise SystemExit(main())
