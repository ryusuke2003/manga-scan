from __future__ import annotations

import math

import cv2
import numpy as np

from .perspective import validate_roi
from .temporal_alignment import estimate_frame_alignments, transform_normalized_quad


def _quad_area(quad):
    points = np.asarray(quad, dtype=np.float32)
    return abs(float(cv2.contourArea(points)))


def track_spread_roi(
    previous_image,
    current_image,
    previous_roi,
    reference_roi,
    *,
    max_step=0.08,
    max_total=0.16,
):
    """Track a trusted spread ROI into the next stable interval.

    The transform is accepted only when optical-flow homography is reliable and
    both per-step and cumulative movement remain conservative. The caller keeps
    the previous trusted ROI when tracking is unavailable.
    """
    previous = validate_roi(previous_roi)
    reference = validate_roi(reference_roi)
    max_step = float(max_step)
    max_total = float(max_total)
    if not math.isfinite(max_step) or not 0 < max_step <= 0.25:
        raise ValueError("max_step must be within 0..0.25")
    if not math.isfinite(max_total) or not max_step <= max_total <= 0.4:
        raise ValueError("max_total must be >= max_step and <= 0.4")

    alignments = estimate_frame_alignments([previous_image, current_image], anchor_index=1)
    alignment = alignments[0]
    if alignment.get("status") != "aligned" or alignment.get("matrix") is None:
        return {
            "tracked": False,
            "status": alignment.get("status", "unavailable"),
            "roi": previous.tolist(),
            "step_shift": 0.0,
            "total_shift": float(np.max(np.linalg.norm(previous - reference, axis=1))),
            "alignment": {key: value for key, value in alignment.items() if key != "matrix"},
        }

    try:
        tracked = transform_normalized_quad(
            previous,
            alignment["matrix"],
            previous_image.shape,
            current_image.shape,
        )
        tracked = validate_roi(tracked)
    except (ValueError, cv2.error):
        return {
            "tracked": False,
            "status": "invalid_transform",
            "roi": previous.tolist(),
            "step_shift": 0.0,
            "total_shift": float(np.max(np.linalg.norm(previous - reference, axis=1))),
            "alignment": {key: value for key, value in alignment.items() if key != "matrix"},
        }

    step_shift = float(np.max(np.linalg.norm(tracked - previous, axis=1)))
    total_shift = float(np.max(np.linalg.norm(tracked - reference, axis=1)))
    reference_area = max(_quad_area(reference), 1e-6)
    area_ratio = _quad_area(tracked) / reference_area

    if step_shift > max_step:
        status = "step_shift_too_large"
    elif total_shift > max_total:
        status = "total_shift_too_large"
    elif not 0.70 <= area_ratio <= 1.40:
        status = "area_change_too_large"
    else:
        status = "tracked"

    return {
        "tracked": status == "tracked",
        "status": status,
        "roi": (tracked if status == "tracked" else previous).tolist(),
        "step_shift": round(step_shift, 6),
        "total_shift": round(total_shift, 6),
        "area_ratio": round(float(area_ratio), 6),
        "alignment": {key: value for key, value in alignment.items() if key != "matrix"},
    }
