"""Compare pairwise image motion directly from an explicit mask-tracking run.

Pair-exclusive feature regions prevent shared mask pixels from being counted
as evidence from two faces. These regions are not new segmentation outputs.
Image homographies alone never establish physical packet membership/events.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .draft_export import _bound, _inside, read_json, read_jsonl, sha, write_json
from .motion_relation import MotionParameters, compare_image_models, track_mask_points


def exclusive_pair_masks(first, second):
    """Retain original masks elsewhere; use distinct pixels only for features."""
    a, b = np.asarray(first), np.asarray(second)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError('Pair masks require the same2D pixel grid')
    a, b = a > 0, b > 0
    return (a & ~b).astype(np.uint8) * 255, (b & ~a).astype(np.uint8) * 255


def run_tracked_motion(tracking_run, output_dir, windows, parameters=MotionParameters()):
    run, out = Path(tracking_run).resolve(), Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve existing motion output; use a new directory')
    report_path, source_path = run / 'report.json', run / 'source_manifest.json'
    report, source = read_json(report_path), read_json(source_path)
    if report.get('status') != 'completed':
        raise ValueError('Tracking run must be completed')
    frames_path = _bound(run / 'frames.jsonl', report['frames_jsonl_sha256'], 'tracking timeline')
    seed_path = _bound(run / 'seeds_used.json', report['seed_config_sha256'], 'tracking seeds')
    seeds, rows = read_json(seed_path), read_jsonl(frames_path)
    count = source['frame_count']
    ids = [obj['object_id'] for obj in seeds['objects']]
    if len(ids) < 2 or len(set(ids)) != len(ids) or len(rows) != count or report['completed_frames'] != count:
        raise ValueError('Tracking object IDs or completed timeline inconsistent')
    if seeds['source_video_sha256'] != source['source_video_sha256']:
        raise ValueError('Seeds and original video differ')
    _bound(source['source_video'], source['source_video_sha256'], 'source video')
    normalized = []
    for window in windows:
        if len(window) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in window):
            raise ValueError('Each inclusive window needs two integer frame numbers')
        a, b = window
        if not 0 <= a < b < count:
            raise ValueError('Motion window is outside the source timeline')
        if any(not (b < p or a > q) for p, q in normalized):
            raise ValueError('Use nonoverlapping windows to avoid duplicate interval counts')
        normalized.append((a, b))
    if not normalized:
        raise ValueError('At least one motion window is required')
    start = time.perf_counter()
    images, masks, inputs = {}, {}, []
    for f in sorted({f for a, b in normalized for f in range(a, b + 1)}):
        row, src = rows[f], source['frames'][f]
        if row['frame'] != f or row['source_frame'] != f or src['frame'] != f or row['source_jpeg_sha256'] != src['jpeg_sha256']:
            raise ValueError('Source frame/JPEG association differs')
        if row['source_video_sha256'] != source['source_video_sha256']:
            raise ValueError('Track frame belongs to another source video')
        path = _bound(_inside(source['frame_directory'], src['jpeg_file']), src['jpeg_sha256'], 'source JPEG')
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2][::-1]) != source['resolution']:
            raise ValueError('Source image dimensions differ')
        images[f], masks[f] = image, {}
        refs = []
        for obj in row['objects']:
            identity = obj['object_id']
            if identity in masks[f] or identity not in ids:
                raise ValueError('Duplicate or unknown object in a frame')
            path = _bound(_inside(run, obj['mask_file']), obj['mask_sha256'], 'tracked mask')
            m = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            if m is None or m.shape != image.shape[:2] or not np.isin(m, [0, 255]).all() or np.count_nonzero(m) != obj['area_px']:
                raise ValueError('Mask grid, binary values or area differs')
            masks[f][identity] = m
            refs.append({'object_id': identity, 'path': str(path), 'sha256': obj['mask_sha256'], 'area_px': obj['area_px']})
        if set(masks[f]) != set(ids):
            raise ValueError('Missing track record; do not invent a mask')
        inputs.append({'frame': f, 'jpeg_sha256': src['jpeg_sha256'], 'masks': refs})
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for a, b in normalized:
        for f in range(a, b):
            gray0, gray1 = [cv2.cvtColor(images[t], cv2.COLOR_BGR2GRAY) for t in (f, f + 1)]
            pairs = []
            for x, y in combinations(ids, 2):
                first0, second0 = exclusive_pair_masks(masks[f][x], masks[f][y])
                first1, second1 = exclusive_pair_masks(masks[f + 1][x], masks[f + 1][y])
                tracks = {
                    x: track_mask_points(gray0, gray1, first0, first1, parameters),
                    y: track_mask_points(gray0, gray1, second0, second1, parameters),
                }
                pairs.append({'object_ids': [x, y], 'tracks': tracks,
                    'excluded_intersection_pixels': {
                        'source': int(np.count_nonzero((masks[f][x] > 0) & (masks[f][y] > 0))),
                        'target': int(np.count_nonzero((masks[f + 1][x] > 0) & (masks[f + 1][y] > 0)))},
                    'comparison': compare_image_models(tracks, parameters)})
            records.append({'source_frame': f, 'target_frame': f + 1,
                'time_seconds': f / source['fps'], 'pairs': pairs, 'physical_events': []})
    records_file = out / 'motion_relations.jsonl'
    records_file.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    by_frame = {r['source_frame']: r for r in records}
    videos = []
    width, height = source['resolution']
    banner = 300
    colors = [(220, 140, 20), (20, 210, 250), (190, 30, 190), (40, 220, 30)]
    for a, b in normalized:
        video = out / f'motion_{a:03}_{b:03}.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), source['fps'], (width, height + banner))
        if not writer.isOpened():
            raise RuntimeError('Motion video writer failed to open')
        try:
            for f in range(a, b + 1):
                body, head = images[f].copy(), np.full((banner, width, 3), 24, np.uint8)
                cv2.putText(head, f'Pair-exclusive image motion | original frame {f} | {f / source["fps"]:.3f}s',
                            (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 1, cv2.LINE_AA)
                for i, identity in enumerate(ids):
                    contour, _ = cv2.findContours(masks[f][identity], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(body, contour, -1, colors[i % len(colors)], 1)
                r = by_frame.get(f) if f < b else None
                if r:
                    for i, pair in enumerate(r['pairs']):
                        short_ids = '/'.join(str(ids.index(x) + 1) for x in pair['object_ids'])
                        label = pair['comparison']['image_motion_evidence']
                        points = '/'.join(str(t['accepted_correspondences']) for t in pair['tracks'].values())
                        text = f'{short_ids}: {label}; accepted points {points}'
                        cv2.putText(head, text, (14, 61 + 27 * i), cv2.FONT_HERSHEY_SIMPLEX, .47, (230, 230, 230), 1, cv2.LINE_AA)
                else:
                    cv2.putText(head, 'Window end; no outgoing interval evaluated', (14, 61), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
                cv2.putText(head, 'Mask intersections excluded ONLY for feature selection; original masks retained.', (14, 249), cv2.FONT_HERSHEY_SIMPLEX, .49, (210, 210, 210), 1)
                cv2.putText(head, 'Image motion is NOT physical packet membership; no split or merge inferred.', (14, 278), cv2.FONT_HERSHEY_SIMPLEX, .49, (210, 210, 210), 1)
                display = np.vstack((head, body))
                ok, png = cv2.imencode('.png', display)
                if not ok:
                    raise RuntimeError('Motion PNG encode failed')
                (out / f'frame_{f:06}.png').write_bytes(png.tobytes())
                writer.write(display)
        finally:
            writer.release()
        cap, decoded = cv2.VideoCapture(str(video)), 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if pixels.shape != (height + banner, width, 3):
                raise RuntimeError('Motion video decoded dimensions differ')
            decoded += 1
        cap.release()
        if decoded != b - a + 1:
            raise RuntimeError('Motion video is incomplete')
        videos.append({'window': [a, b], 'path': str(video), 'sha256': sha(video), 'decoded_frames': decoded})
    summary = []
    for x, y in combinations(ids, 2):
        selected = [p for r in records for p in r['pairs'] if p['object_ids'] == [x, y]]
        summary.append({'object_ids': [x, y], 'intervals': len(selected),
            'both_tracks_eligible_intervals': sum(all(t['eligible_for_model_comparison'] for t in p['tracks'].values()) for p in selected),
            'image_relation_counts': dict(Counter(p['comparison']['image_motion_evidence'] for p in selected))})
    result = {'schema': 'cardcap.tracked_pair_motion/1.0', 'status': 'completed',
        'tracking_report': str(report_path), 'tracking_report_sha256': sha(report_path),
        'source_manifest_sha256': sha(source_path), 'source_video_sha256': source['source_video_sha256'],
        'frames_jsonl_sha256': sha(frames_path), 'seeds_sha256': sha(seed_path),
        'windows_inclusive': normalized, 'source_intervals': len(records), 'pair_comparisons': sum(len(r['pairs']) for r in records),
        'pair_summaries': summary, 'parameters': asdict(parameters), 'inputs': inputs,
        'analysis_region': 'For each pair and frame, exclude the two original masks intersection from both feature-selection masks; never mutate or export these as segmented surfaces.',
        'physical_group_relation': 'unknown', 'physical_events': [], 'videos': videos,
        'motion_relations_sha256': sha(records_file), 'code_sha256': sha(__file__),
        'motion_math_code_sha256': sha(Path(__file__).with_name('motion_relation.py')),
        'elapsed_seconds': time.perf_counter() - start,
        'limitations': ['Pair-exclusive image features do not establish seed identity across occlusion or flips.',
                       'Small masks can lose enough pixels to be ineligible; absence is unknown, not merging.',
                       'Shared image homographies are not sufficient to identify one physical rigid group.',
                       'No3D pose, card dimensions, calibrated camera or independent metric measurement used.']}
    write_json(out / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, nargs=2, action='append', required=True)
    args = parser.parse_args()
    result = run_tracked_motion(args.tracking_run, args.output, args.window)
    print(json.dumps({k: result[k] for k in ('status', 'source_intervals', 'pair_comparisons', 'pair_summaries', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
