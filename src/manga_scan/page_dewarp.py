"""Conservative content-based page dewarping for book gutter curvature."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _to_gray(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy grayscale or multi-channel image")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image must be at least 2x2")
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("image must have 1, 3, or 4 channels")


def _validate_side(side):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")


def _horizontal_segments(image):
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(18, round(w * 0.05)),
        minLineLength=max(18, round(w * 0.07)),
        maxLineGap=max(4, round(w * 0.025)),
    )
    if lines is None:
        return []

    segments = []
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = map(int, raw)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < max(12, round(w * 0.05)):
            continue
        slope = dy / dx
        if abs(slope) > 0.35:
            continue
        length = math.hypot(dx, dy)
        midpoint = ((x1 + x2) / 2) / max(w - 1, 1)
        segments.append(
            {
                "points": [x1, y1, x2, y2],
                "x": float(midpoint),
                "slope": float(slope),
                "weight": float(length),
            }
        )
    return segments


def _weighted_fit(x, y, weights):
    design = np.column_stack([x, np.ones_like(x)])
    root_w = np.sqrt(weights)[:, None]
    coeff, *_ = np.linalg.lstsq(design * root_w, y * root_w[:, 0], rcond=None)
    fitted = design @ coeff
    return float(coeff[0]), float(coeff[1]), fitted


def estimate_page_curvature(image, side, *, min_segments=8):
    """Estimate gutter curvature from changing slopes of near-horizontal image edges.

    The varying slope component is modeled as a quadratic vertical displacement.
    Global page tilt is intentionally ignored so this only targets curvature.
    """

    _validate_side(side)
    if not isinstance(min_segments, int) or isinstance(min_segments, bool) or min_segments < 4:
        raise ValueError("min_segments must be an integer >= 4")

    gray = _to_gray(image)
    h, w = gray.shape[:2]
    segments = _horizontal_segments(image)
    result = {
        "detected": False,
        "confidence": 0.0,
        "spine_displacement": 0.0,
        "slope_gradient": 0.0,
        "segment_count": len(segments),
        "segments": [segment["points"] for segment in segments],
    }
    if len(segments) < min_segments:
        return result

    x = np.asarray([segment["x"] for segment in segments], dtype=np.float64)
    y = np.asarray([segment["slope"] for segment in segments], dtype=np.float64)
    weights = np.asarray([segment["weight"] for segment in segments], dtype=np.float64)
    span = float(np.ptp(x))
    if span < 0.28:
        return result

    gradient, intercept, fitted = _weighted_fit(x, y, weights)
    residual = y - fitted
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    tolerance = max(0.025, 3.0 * 1.4826 * mad)
    keep = np.abs(residual - median) <= tolerance

    if int(keep.sum()) >= min_segments and int(keep.sum()) < len(segments):
        x = x[keep]
        y = y[keep]
        weights = weights[keep]
        gradient, intercept, fitted = _weighted_fit(x, y, weights)
        span = float(np.ptp(x))

    y_mean = float(np.average(y, weights=weights))
    total = float(np.sum(weights * (y - y_mean) ** 2))
    error = float(np.sum(weights * (y - fitted) ** 2))
    r_squared = 1.0 - error / total if total > 1e-9 else 0.0

    count_score = min(1.0, len(x) / 20.0)
    span_score = float(np.clip((span - 0.28) / 0.5, 0.0, 1.0))
    fit_score = float(np.clip(r_squared, 0.0, 1.0))
    confidence = 0.30 * count_score + 0.30 * span_score + 0.40 * fit_score

    spine_displacement = 0.5 * gradient * w
    if abs(spine_displacement) < 0.75:
        confidence *= min(1.0, abs(spine_displacement) / 0.75)

    result.update(
        {
            "detected": bool(confidence >= 0.45 and abs(spine_displacement) >= 0.75),
            "confidence": round(float(confidence), 4),
            "spine_displacement": round(float(spine_displacement), 4),
            "slope_gradient": round(float(gradient), 6),
            "segment_count": int(len(x)),
            "fit_intercept": round(float(intercept), 6),
            "r_squared": round(float(r_squared), 4),
        }
    )
    return result


def dewarp_profile(shape, side, spine_displacement, *, max_strength=0.08):
    """Return the vertical source-sampling offset for each x coordinate."""

    _validate_side(side)
    h, w = shape[:2]
    if h < 2 or w < 2:
        raise ValueError("image must be at least 2x2")
    if not np.isfinite(spine_displacement):
        raise ValueError("spine_displacement must be finite")
    if not 0 <= float(max_strength) <= 0.25:
        raise ValueError("max_strength must be 0..0.25")

    limit = float(h) * float(max_strength)
    applied = float(np.clip(float(spine_displacement), -limit, limit))
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    outer = 0.0 if side == "left" else 1.0
    profile = applied * (x - outer) ** 2
    return profile.astype(np.float32), applied


def dewarp_page(
    image,
    side,
    spine_displacement,
    *,
    max_strength=0.08,
    interpolation=cv2.INTER_CUBIC,
):
    """Apply a bounded vertical remap while keeping the outer page edge anchored."""

    _to_gray(image)
    profile, applied = dewarp_profile(
        image.shape,
        side,
        spine_displacement,
        max_strength=max_strength,
    )
    h, w = image.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) + profile[None, :]
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, applied


def dewarp_page_auto(
    image,
    side,
    *,
    max_strength=0.08,
    min_confidence=0.45,
    min_segments=8,
):
    """Estimate and conservatively correct page curvature, or return an exact copy."""

    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")

    estimate = estimate_page_curvature(image, side, min_segments=min_segments)
    if not estimate["detected"] or estimate["confidence"] < min_confidence:
        return image.copy(), {**estimate, "applied": False, "applied_displacement": 0.0}

    corrected, applied = dewarp_page(
        image,
        side,
        estimate["spine_displacement"],
        max_strength=max_strength,
    )
    return corrected, {
        **estimate,
        "applied": True,
        "applied_displacement": round(float(applied), 4),
        "clamped": not np.isclose(applied, estimate["spine_displacement"]),
    }


def draw_dewarp_debug(image, estimate, side, *, max_strength=0.08):
    """Overlay detected line segments and the estimated remap profile."""

    _validate_side(side)
    gray = _to_gray(image)
    if image.ndim == 2:
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()

    for points in estimate.get("segments", []):
        x1, y1, x2, y2 = map(int, points)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 180, 255), 1, cv2.LINE_AA)

    displacement = float(estimate.get("spine_displacement", 0.0))
    profile, _ = dewarp_profile(canvas.shape, side, displacement, max_strength=max_strength)
    center = canvas.shape[0] // 2
    points = np.column_stack(
        [
            np.arange(canvas.shape[1], dtype=np.int32),
            np.clip(np.rint(center + profile), 0, canvas.shape[0] - 1).astype(np.int32),
        ]
    )
    cv2.polylines(canvas, [points], False, (60, 200, 80), 2, cv2.LINE_AA)
    return canvas
