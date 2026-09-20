"""Suggest likely reference-spread frames before the user picks one manually."""

from __future__ import annotations

import math

import numpy as np

from .motion import motion_score
from .score import sharpness
from .split import rotate_image
from .spread_detect import detect_reference_spread
from .video import sample_frames

REFERENCE_CANDIDATE_SAMPLE_FPS = 2.0
REFERENCE_CANDIDATE_SEARCH_SECONDS = 45.0
REFERENCE_CANDIDATE_MIN_SEPARATION = 1.0


def _candidate_score(confidence, motion, sharpness_value, elapsed, window, turn_threshold):
    stillness = float(np.clip(1.0 - motion / max(float(turn_threshold), 1e-6), 0.0, 1.0))
    sharpness_score = float(np.clip(math.log1p(max(0.0, sharpness_value)) / 8.0, 0.0, 1.0))
    early = float(np.clip(1.0 - elapsed / max(window, 1e-6), 0.0, 1.0))
    return (
        0.58 * float(confidence)
        + 0.22 * stillness
        + 0.12 * sharpness_score
        + 0.08 * early
    )


def select_reference_candidates(records, limit=5, min_separation=REFERENCE_CANDIDATE_MIN_SEPARATION):
    """Pick strong, temporally distinct candidates while preserving score order."""

    if limit < 1:
        return []
    ranked = sorted(records, key=lambda item: (-item["score"], item["time"]))
    selected = []
    for record in ranked:
        if any(abs(record["time"] - existing["time"]) < min_separation for existing in selected):
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def scan_reference_candidates(source, metadata, cfg, *, start_time=0.0, limit=5):
    """Scan the early video for likely open-spread frames.

    This is intentionally lightweight: low-resolution frames are ranked by
    two-page geometry confidence, stillness, sharpness, and a small preference
    for earlier timestamps. It does not run hand detection and never replaces
    the user's final confirmation.
    """

    duration = float(metadata["duration"])
    start = float(np.clip(float(start_time), 0.0, max(0.0, duration - 0.001)))
    end = min(duration, start + REFERENCE_CANDIDATE_SEARCH_SECONDS)
    if end <= start:
        return []

    width_limit = min(640, int(cfg.analysis_width))
    source_width = int(metadata.get("display_width") or metadata.get("width") or width_limit)
    source_height = int(metadata.get("display_height") or metadata.get("height") or width_limit)
    width = min(width_limit, source_width)
    height = max(2, round(source_height * width / max(source_width, 1)))
    source_fps = float(metadata.get("fps") or REFERENCE_CANDIDATE_SAMPLE_FPS)
    fps = min(REFERENCE_CANDIDATE_SAMPLE_FPS, float(cfg.video_sample_fps), source_fps)
    fps = max(0.5, fps)

    threshold = max(0.4, min(0.65, float(cfg.page_contour_min_confidence) - 0.1))
    fallback_confidence = max(0.35, threshold - 0.08)
    previous = None
    records = []
    stream = sample_frames(source, fps, (width, height), cfg.hwaccel, start_time=start)
    try:
        for _index, timestamp, frame in stream:
            if timestamp > end:
                break
            displayed = rotate_image(frame, cfg.rotation)
            motion = motion_score(previous, displayed) if previous is not None else 1.0
            previous = displayed
            detection = detect_reference_spread(displayed, min_confidence=threshold)
            confidence = float(detection["confidence"])
            if not detection["detected"] and confidence < fallback_confidence:
                continue
            sharpness_value = sharpness(displayed)
            score = _candidate_score(
                confidence,
                motion,
                sharpness_value,
                timestamp - start,
                end - start,
                cfg.turn_threshold,
            )
            records.append(
                {
                    "time": round(float(timestamp), 3),
                    "score": round(float(score), 4),
                    "confidence": round(confidence, 4),
                    "motion": round(float(motion), 6),
                    "sharpness": round(float(sharpness_value), 3),
                    "detected": bool(detection["detected"]),
                    "stage": detection.get("stage"),
                    "_frame": displayed.copy(),
                }
            )
    finally:
        stream.close()

    return select_reference_candidates(records, limit=limit)
