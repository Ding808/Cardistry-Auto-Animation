"""Original-frame overlays with explicit estimator and scale limitations."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
COLORS = {"left": (255, 190, 50), "right": (75, 210, 110)}


def _hand_caption(hand: dict) -> str:
    identity = hand.get("diagnostics", {}).get("hand_identity")
    if isinstance(identity, dict):
        native_side = identity.get("native_side", hand["side"])
        native_score = identity.get("native_handedness_confidence", hand["handedness_confidence"])
        caption = f'native {native_side} score {native_score:.3f} -> tracked {hand["side"]}'
        if identity.get("ambiguous", False):
            caption += " [AMBIGUOUS]"
        return caption
    return f'{hand["side"]}: label {hand["handedness_confidence"]:.3f}'


def overlay_frame(bgr: np.ndarray, record: dict, index: int, fps: float) -> np.ndarray:
    result = bgr.copy()
    height, width = result.shape[:2]
    # Keep the primary observed image intact below a compact informational band.
    band = result[:min(108, height)].copy()
    cv2.rectangle(band, (0, 0), (width, band.shape[0]), (18, 18, 18), -1)
    result[:band.shape[0]] = cv2.addWeighted(result[:band.shape[0]], 0.15, band, 0.85, 0)
    backend_note = (
        "WiLoR MANO reconstruction | MediaPipe hand boxes | metric scale unverified"
        if record.get("backend") == "wilor" else
        "MediaPipe world = hand-centered weak 3D | MANO unavailable | label score != pose accuracy"
    )
    labels = [
        f"CardistryCapture | Hand observations | frame {index} | {index/fps:.3f}s",
        backend_note,
    ]
    if not record["hands"]:
        labels.append("NO HAND DETECTED - frame retained for manual review")
    else:
        labels.append(" | ".join(_hand_caption(hand) for hand in record["hands"]))
    if record["blur_flag"]:
        labels[-1] += " | BLUR FLAG"
    font_scale = max(0.35, min(0.6, width / 2100))
    for i, label in enumerate(labels):
        cv2.putText(result, label, (12, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (225, 225, 235), 1, cv2.LINE_AA)
    for hand in record["hands"]:
        color = COLORS[hand["side"]]
        points = np.rint(hand["image_landmarks_px"]).astype(np.int64)
        # Avoid huge coordinates being passed to native drawing routines.
        points = np.clip(points, [-width, -height], [2 * width, 2 * height]).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(result, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
        for joint, point in enumerate(points):
            cv2.circle(result, tuple(point), 4 if joint in {4, 8, 12, 16, 20} else 3, color, -1, cv2.LINE_AA)
            if joint in {0, 4, 8, 12, 16, 20}:
                cv2.putText(result, str(joint), tuple(point + [5, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return result


class OverlayWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = path
        self.frames_written = 0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"No usable MP4 writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        self.writer.release()


def verify_video(path: Path, expected_count: int, fps: float) -> dict:
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"Generated overlay cannot be reopened: {path}")
    count = 0
    try:
        actual_fps = reader.get(cv2.CAP_PROP_FPS)
        size = [int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if frame is None or not frame.size:
                raise RuntimeError("Decoded empty overlay frame")
            count += 1
    finally:
        reader.release()
    if count != expected_count or abs(actual_fps - fps) > 0.01:
        raise RuntimeError(f"Overlay mismatch: {count}/{expected_count} frames, {actual_fps}/{fps} fps")
    return {"path": str(path), "decoded_frames": count, "fps": actual_fps, "resolution": size, "bytes": path.stat().st_size}


def contact_sheet(video: Path, output: Path, frame_indices: list[int], columns: int = 3) -> None:
    capture = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Contact sheet decode failed at frame {index}")
            tile = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"FRAME {index}", (8, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
    finally:
        capture.release()
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * 270, columns * 480, 3), 18, np.uint8)
    for i, tile in enumerate(tiles):
        sheet[(i // columns) * 270:(i // columns + 1) * 270,
              (i % columns) * 480:(i % columns + 1) * 480] = tile
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Contact sheet could not be saved: {output}")
