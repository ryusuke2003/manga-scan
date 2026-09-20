"""Conservative left/right page contour detection inside a known spread ROI."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .perspective import pixel_quad, validate_roi


def _order_quad(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    # Keep TL, TR, BR, BL ordering in image coordinates.
    if pts[1, 0] < pts[3, 0]:
        pts = np.concatenate([pts[:1], pts[:0:-1]], axis=0)
    return pts


def _side_priors(reference_roi, shape, spine_ratio):
    q = pixel_quad(reference_roi, shape).astype(np.float32)
    top = q[0] * (1 - spine_ratio) + q[1] * spine_ratio
    bottom = q[3] * (1 - spine_ratio) + q[2] * spine_ratio
    return {
        "left": np.asarray([q[0], top, bottom, q[3]], dtype=np.float32),
        "right": np.asarray([top, q[1], q[2], bottom], dtype=np.float32),
    }


def _edge_map(image):
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("image must be HxW grayscale, BGR, or BGRA")

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    low = int(max(20, min(120, median * 0.5)))
    high = int(min(240, max(low + 30, median * 1.25)))
    adaptive = cv2.Canny(blur, low, high)
    fixed = cv2.Canny(blur, 30, 100)
    edges = cv2.bitwise_or(adaptive, fixed)
    return cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )


def _edge_support(edges, quad):
    outline = np.zeros_like(edges)
    cv2.polylines(
        outline,
        [np.round(quad).astype(np.int32)],
        True,
        255,
        2,
        cv2.LINE_AA,
    )
    count = int(np.count_nonzero(outline))
    if not count:
        return 0.0
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return float(np.count_nonzero((outline > 0) & (dilated > 0)) / count)


def _score_quad(quad, prior, edges):
    area = abs(float(cv2.contourArea(quad)))
    prior_area = abs(float(cv2.contourArea(prior)))
    if prior_area <= 1 or not 0.35 * prior_area <= area <= 1.03 * prior_area:
        return None

    intersection, _ = cv2.intersectConvexConvex(
        quad.astype(np.float32), prior.astype(np.float32)
    )
    coverage = float(intersection / prior_area)
    if coverage < 0.35:
        return None

    _, _, width, height = cv2.boundingRect(prior.astype(np.float32))
    diagonal = math.hypot(width, height)
    mean_corner_distance = float(
        np.mean(np.linalg.norm(quad - prior, axis=1)) / max(diagonal, 1.0)
    )
    corner_score = math.exp(-mean_corner_distance / 0.12)
    area_score = float(np.clip((coverage - 0.35) / 0.65, 0, 1))
    support = _edge_support(edges, quad)

    confidence = 0.50 * area_score + 0.35 * corner_score + 0.15 * support
    return float(np.clip(confidence, 0, 1))


def _quad_candidates(contour):
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter < 20:
        return []
    for epsilon in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return [_order_quad(approx.reshape(4, 2))]
    return []


def _detect_side(edges, prior):
    mask = np.zeros_like(edges)
    cv2.fillConvexPoly(mask, np.round(prior).astype(np.int32), 255)
    side_edges = cv2.bitwise_and(edges, mask)
    contours, _ = cv2.findContours(side_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_confidence = 0.0
    for contour in contours:
        for quad in _quad_candidates(contour):
            # Never expand beyond the side prior; this prevents desk pixels from being added.
            inside = all(
                cv2.pointPolygonTest(
                    prior.astype(np.float32), tuple(map(float, point)), False
                )
                >= 0
                for point in quad
            )
            if not inside:
                continue
            confidence = _score_quad(quad, prior, edges)
            if confidence is not None and confidence > best_confidence:
                best_quad = quad
                best_confidence = confidence
    return best_quad, best_confidence


def _normalize_quad(quad, shape):
    h, w = shape[:2]
    scale = np.asarray([max(w - 1, 1), max(h - 1, 1)], dtype=np.float32)
    normalized = np.asarray(quad, dtype=np.float32) / scale
    # A valid spread ROI can be close to validate_roi()'s minimum area. Splitting
    # it in half must not make an otherwise valid page fallback fail validation.
    if (
        normalized.shape != (4, 2)
        or not np.isfinite(normalized).all()
        or (normalized < 0).any()
        or (normalized > 1).any()
    ):
        raise ValueError("page quad must contain four finite [x,y] pairs in 0..1")
    return normalized.tolist()


def detect_page_quads(
    image,
    reference_roi,
    *,
    spine_ratio=0.5,
    min_confidence=0.5,
):
    """Detect left/right page quads within a known spread ROI.

    Returned quads are normalized TL,TR,BR,BL coordinates in the input image.
    Each side falls back to its reference split when no sufficiently confident
    contour is found, so callers can safely opt into detection without adding
    desk pixels.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image must be at least 2x2")
    validate_roi(reference_roi)
    if not 0.25 <= float(spine_ratio) <= 0.75:
        raise ValueError("spine_ratio must be 0.25..0.75")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")

    edges = _edge_map(image)
    priors = _side_priors(reference_roi, image.shape, float(spine_ratio))
    result = {}

    for side, prior in priors.items():
        detected_quad, confidence = _detect_side(edges, prior)
        detected = detected_quad is not None and confidence >= min_confidence
        selected = detected_quad if detected else prior
        result[side] = {
            "quad": _normalize_quad(selected, image.shape),
            "confidence": round(float(confidence), 4),
            "detected": bool(detected),
        }

    result["confidence"] = min(result["left"]["confidence"], result["right"]["confidence"])
    result["detected"] = result["left"]["detected"] and result["right"]["detected"]
    return result


def draw_page_quads(image, result):
    """Return a debug copy with detected/fallback page quads overlaid."""

    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy image")
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()

    h, w = canvas.shape[:2]
    scale = np.asarray([max(w - 1, 1), max(h - 1, 1)], dtype=np.float32)
    for side, color in (("left", (60, 180, 75)), ("right", (0, 140, 255))):
        data = result[side]
        quad = np.asarray(data["quad"], dtype=np.float32) * scale
        quad = np.round(quad).astype(np.int32)
        cv2.polylines(canvas, [quad], True, color, 2, cv2.LINE_AA)
        x, y = quad[0]
        label = f"{side} {data['confidence']:.2f}"
        if not data["detected"]:
            label += " fallback"
        cv2.putText(
            canvas,
            label,
            (int(x), max(16, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
