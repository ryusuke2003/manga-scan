"""Recover a photographed spread boundary on either side of an approximate ROI."""

from __future__ import annotations

import cv2
import numpy as np

from .perspective import validate_roi

_MAX_WORKING_SIDE = 512


def _resize(image):
    height, width = image.shape[:2]
    scale = min(1.0, _MAX_WORKING_SIDE / max(height, width))
    if scale == 1:
        return image
    return cv2.resize(
        image,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _adjust_quad(quad, amount):
    """Move all four edges by a fraction of the quad's own dimensions."""

    q = np.asarray(quad, dtype=np.float32)
    coordinates = (
        (-amount, -amount),
        (1 + amount, -amount),
        (1 + amount, 1 + amount),
        (-amount, 1 + amount),
    )
    points = [
        q[0] * (1 - u) * (1 - v)
        + q[1] * u * (1 - v)
        + q[2] * u * v
        + q[3] * (1 - u) * v
        for u, v in coordinates
    ]
    return np.clip(np.asarray(points, dtype=np.float32), 0, 1)


def _ordered_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    points = np.roll(points, -start, axis=0)
    if points[1, 0] < points[3, 0]:
        points = np.concatenate([points[:1], points[:0:-1]], axis=0)
    return points


def refine_spread_boundary(image, roi):
    """Return a page-aligned ROI only when image evidence supports the change.

    GrabCut receives a definite book seed in the center and a definite
    background seed outside a bounded search region. The fitted quadrilateral
    can move inward to remove desk or outward to recover a clipped page. An
    inward move is rejected when the original edge still lies on foreground;
    this avoids cutting artwork when the segmentation misses part of a page.
    """

    original = validate_roi(roi)
    info = {"refined": False, "status": "unavailable"}
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return original.tolist(), info

    working = _resize(image)
    if float(np.mean(np.std(working.reshape(-1, 3), axis=0))) < 8:
        return original.tolist(), info
    height, width = working.shape[:2]
    scale = np.asarray([width - 1, height - 1], dtype=np.float32)
    initial = np.rint(original * scale).astype(np.int32)
    inner = np.rint(_adjust_quad(original, -0.20) * scale).astype(np.int32)
    outer = np.rint(_adjust_quad(original, 0.12) * scale).astype(np.int32)
    labels = np.full((height, width), cv2.GC_BGD, np.uint8)
    cv2.fillConvexPoly(labels, outer, cv2.GC_PR_BGD)
    cv2.fillConvexPoly(labels, initial, cv2.GC_PR_FGD)
    cv2.fillConvexPoly(labels, inner, cv2.GC_FGD)
    if not np.any(labels == cv2.GC_BGD):
        # A generous coarse ROI can make the search polygon cover the frame.
        # Only frame corners outside the ROI are trusted as background seeds.
        margin = max(2, round(min(height, width) * 0.02))
        for ys, xs in (
            (slice(0, margin), slice(0, margin)),
            (slice(0, margin), slice(width - margin, width)),
            (slice(height - margin, height), slice(0, margin)),
            (slice(height - margin, height), slice(width - margin, width)),
        ):
            corner = labels[ys, xs]
            corner[corner == cv2.GC_PR_BGD] = cv2.GC_BGD
    if not np.any(labels == cv2.GC_BGD) or not np.any(labels == cv2.GC_FGD):
        return original.tolist(), info

    try:
        cv2.grabCut(
            working,
            labels,
            None,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return original.tolist(), info

    foreground = np.where(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original.tolist(), info
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    candidate = None
    for epsilon in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            candidate = _ordered_quad(approx.reshape(4, 2)) / scale
            break
    if candidate is None:
        return original.tolist(), info
    try:
        candidate = validate_roi(candidate)
    except ValueError:
        return original.tolist(), info

    initial_area = float(cv2.contourArea(original))
    candidate_area = float(cv2.contourArea(candidate))
    area_ratio = candidate_area / initial_area
    corner_shift = float(np.max(np.linalg.norm(candidate - original, axis=1)))
    if not 0.65 <= area_ratio <= 1.45 or corner_shift > 0.12:
        return original.tolist(), {**info, "status": "geometry_rejected"}

    outline = np.zeros((height, width), np.uint8)
    cv2.polylines(outline, [initial], True, 255, 3)
    edge_occupancy = float(np.mean(foreground[outline > 0] > 0))
    evidence = {
        "area_ratio": round(area_ratio, 4),
        "corner_shift": round(corner_shift, 4),
        "edge_occupancy": round(edge_occupancy, 4),
    }
    # Shrinking an edge already occupied by the book can clip artwork; growing
    # into an edge classified as desk can add background. Mixed-side errors
    # are common, so use the observed outline rather than area alone.
    if (area_ratio < 0.98 and edge_occupancy > 0.55) or (
        area_ratio > 1.02 and edge_occupancy < 0.55
    ):
        return original.tolist(), {**info, **evidence, "status": "edge_evidence_rejected"}

    candidate_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(candidate_mask, np.rint(candidate * scale).astype(np.int32), 255)
    fill_fraction = float(np.mean(foreground[candidate_mask > 0] > 0))
    if fill_fraction < 0.65:
        return original.tolist(), {**info, "status": "weak_foreground"}

    return candidate.tolist(), {
        "refined": True,
        "status": "refined",
        **evidence,
    }
