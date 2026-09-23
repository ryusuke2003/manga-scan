import math

import cv2
import numpy as np

from .page_contour import detect_page_quads
from .split import rotate_image
from .video import extract_frame

_FULL_ROI = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
_ROTATIONS = (0, 90, 180, 270)
_PREVIEW_WIDTH = 512


def _metadata_rotation(metadata):
    streams = (metadata.get("raw") or {}).get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    for side_data in stream.get("side_data_list") or []:
        value = side_data.get("rotation")
        if value is None:
            continue
        try:
            rotation = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rotation) and abs(rotation) > 0.1:
            return int(round(rotation)) % 360
    value = (stream.get("tags") or {}).get("rotate")
    if value is not None:
        try:
            rotation = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(rotation) and abs(rotation) > 0.1:
            return int(round(rotation)) % 360
    return None


def _resize_for_detection(image):
    if image.shape[1] <= _PREVIEW_WIDTH:
        return image
    scale = _PREVIEW_WIDTH / image.shape[1]
    return cv2.resize(
        image,
        (_PREVIEW_WIDTH, max(2, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _orientation_score(image):
    image = _resize_for_detection(image)
    height, width = image.shape[:2]
    aspect = width / max(height, 1)
    landscape = float(np.clip((aspect - 1.0) / 0.65, 0, 1))

    try:
        detection = detect_page_quads(
            image,
            _FULL_ROI,
            spine_ratio=0.5,
            min_confidence=0.0,
        )
        contour = float(detection["confidence"])
    except (ValueError, cv2.error):
        contour = 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    center = max(2, round(width * 0.12))
    x1 = max(0, width // 2 - center // 2)
    x2 = min(width, x1 + center)
    vertical = float(np.mean(sobel_x[:, x1:x2]))
    horizontal = float(np.mean(sobel_y[:, x1:x2]))
    gutter_verticality = vertical / max(vertical + horizontal, 1e-6)

    return 0.55 * contour + 0.30 * landscape + 0.15 * gutter_verticality


def _confidence(scores, best_rotation):
    best = float(scores[best_rotation])
    opposite = float(scores[(best_rotation + 180) % 360])
    orthogonal = max(
        float(scores[(best_rotation + 90) % 360]),
        float(scores[(best_rotation + 270) % 360]),
    )
    axis_margin = max(0.0, best - orthogonal)
    direction_margin = max(0.0, best - opposite)
    confidence = float(np.clip(0.35 + 0.35 * best + 0.8 * axis_margin, 0.35, 0.95))
    if direction_margin < 0.025:
        confidence = min(confidence, 0.62)
    if axis_margin < 0.04:
        confidence = min(confidence, 0.55)
    return round(confidence, 3)


def detect_video_rotation(path, metadata, first_frame, hwaccel="none"):
    """Return a best-effort extra rotation for frames already autorotated by FFmpeg."""
    display_rotation = _metadata_rotation(metadata)
    duration = float(metadata["duration"])
    frames = [first_frame]
    times = [0.0]
    for fraction in (0.35, 0.70):
        timestamp = min(max(duration * fraction, 0.0), max(0.0, duration - 0.001))
        if any(abs(timestamp - existing) < 0.05 for existing in times):
            continue
        try:
            frame = extract_frame(
                path,
                timestamp,
                width=_PREVIEW_WIDTH,
                hwaccel=hwaccel,
            )
        except (OSError, RuntimeError):
            continue
        frames.append(frame)
        times.append(round(timestamp, 3))

    totals = {rotation: 0.0 for rotation in _ROTATIONS}
    for frame in frames:
        for rotation in _ROTATIONS:
            totals[rotation] += _orientation_score(rotate_image(frame, rotation))

    scores = {
        rotation: round(totals[rotation] / len(frames), 4)
        for rotation in _ROTATIONS
    }
    best_score = max(scores.values())
    near_ties = [
        rotation for rotation in _ROTATIONS
        if best_score - scores[rotation] <= 0.015
    ]
    # Geometry is often invariant under 180-degree reversal. Use a stable
    # tie-breaker and expose the ambiguity through confidence instead of
    # pretending the page-shape heuristic can always determine "up".
    priority = (0, 90, 270, 180)
    best_rotation = next(rotation for rotation in priority if rotation in near_ties)
    suggested_rotation = best_rotation
    confidence = _confidence(scores, suggested_rotation)
    sideways_axis = suggested_rotation in (90, 270)
    opposite = (suggested_rotation + 180) % 360
    direction_margin = abs(float(scores[suggested_rotation]) - float(scores[opposite]))
    # Page shape and gutter direction cannot reliably distinguish a book from
    # the same book upside down. Small score gaps can come from a hand or desk
    # lighting, so ask for confirmation on either axis.
    direction_ambiguous = direction_margin < 0.06
    if direction_ambiguous:
        confidence = min(confidence, 0.55)
        # Page geometry identifies an axis more reliably than which end is up.
        # Keep the preview unchanged until the user confirms the direction.
        if not sideways_axis or first_frame.shape[0] > first_frame.shape[1]:
            best_rotation = 0
    # The geometry path is designed around three independent observations.
    # If one or both extra seeks fail, do not let a single cover/transition
    # frame suppress the setup warning with an overconfident score.
    if len(frames) == 2:
        confidence = min(confidence, 0.62)
    elif len(frames) == 1:
        confidence = min(confidence, 0.55)
    rotation_options = [best_rotation]
    if direction_ambiguous:
        rotation_options = [90, 270] if sideways_axis else [0, 180]
        if sideways_axis and first_frame.shape[0] > first_frame.shape[1]:
            rotation_options.insert(0, 0)
    result = {
        "rotation": best_rotation,
        "confidence": confidence,
        "source": "page_geometry",
        "scores": {str(rotation): scores[rotation] for rotation in _ROTATIONS},
        "sample_times": times,
        "sample_count": len(frames),
        "direction_ambiguous": direction_ambiguous,
        "requires_confirmation": direction_ambiguous,
        "rotation_options": rotation_options,
        "suggested_rotation": suggested_rotation,
    }
    if display_rotation is not None:
        result["display_rotation"] = display_rotation
        result["metadata_applied"] = True
    return result
