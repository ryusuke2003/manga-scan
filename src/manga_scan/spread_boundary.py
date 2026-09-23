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


def _nested_sheet_boundary(image, quad):
    """Find a page edge inside the visible edge of an underlying sheet.

    GrabCut treats stacked pages and covers as one foreground object. A long,
    nearly parallel inner edge with a consistent outer strip is evidence that
    the outer foreground edge belongs to the sheet underneath. The strip can
    be any color; the visible page side must still look like light paper.
    """

    height, width = image.shape[:2]
    pixels = np.asarray(quad, np.float32) * [width - 1, height - 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 8, 24)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        20,
        minLineLength=round(height * 0.25),
        maxLineGap=round(height * 0.07),
    )
    if lines is None:
        return quad, None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = pixels.copy()
    findings = []
    for side, indices, direction in (("left", (0, 3), 1), ("right", (1, 2), -1)):
        top, bottom = pixels[list(indices)]
        side_height = float(bottom[1] - top[1])
        if side_height < height * 0.25:
            continue
        side_angle = float(np.arctan2(bottom[0] - top[0], side_height))
        best = None
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = (float(value) for value in raw)
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            line_height = y2 - y1
            if line_height < side_height * 0.55:
                continue
            line_angle = float(np.arctan2(x2 - x1, line_height))
            if abs(line_angle - side_angle) > 0.18:
                continue
            low = max(y1, top[1] + side_height * 0.08)
            high = min(y2, bottom[1] - side_height * 0.08)
            if high - low < side_height * 0.45:
                continue
            ys = np.linspace(low, high, 25)
            xs = x1 + (x2 - x1) * (ys - y1) / line_height
            outer_xs = np.interp(ys, [top[1], bottom[1]], [top[0], bottom[0]])
            offsets = direction * (xs - outer_xs)
            if not 0.06 * width <= float(np.median(offsets)) <= 0.23 * width:
                continue
            if float(np.mean((offsets > 0.05 * width) & (offsets < 0.25 * width))) < 0.8:
                continue

            inner_colors = []
            outer_colors = []
            strip_colors = []
            for x, y, outer_x in zip(xs, ys, outer_xs):
                row = int(np.clip(round(y), 2, height - 3))
                column = int(np.clip(round(x), 12, width - 13))
                inside = column + direction * 8
                outside = column - direction * 8
                strip = int(np.clip(round((x + outer_x) / 2), 0, width - 1))
                inner_patch = lab[row - 2 : row + 3, inside - 3 : inside + 3]
                outer_patch = lab[row - 2 : row + 3, outside - 3 : outside + 3]
                inner_colors.append(np.median(inner_patch.reshape(-1, 3), axis=0))
                outer_colors.append(np.median(outer_patch.reshape(-1, 3), axis=0))
                strip_colors.append(lab[row, strip])
            inner_colors = np.asarray(inner_colors)
            outer_colors = np.asarray(outer_colors)
            strip_colors = np.asarray(strip_colors)
            contrast = float(np.linalg.norm(np.median(outer_colors - inner_colors, axis=0)))
            strip_consistency = float(
                np.mean(
                    (inner_colors[:, 0] > 180)
                    & (np.linalg.norm(outer_colors - strip_colors, axis=1) < 26)
                )
            )
            if contrast < 12.5 or strip_consistency < 0.75:
                continue
            score = contrast * strip_consistency * (high - low) / side_height
            if best is None or score > best[0]:
                best = (
                    score,
                    x1,
                    y1,
                    x2,
                    y2,
                    contrast,
                    strip_consistency,
                    float(np.median(offsets)),
                )
        if best is None:
            continue
        _, x1, y1, x2, y2, contrast, strip_consistency, offset = best
        for index in indices:
            y = pixels[index, 1]
            adjusted[index, 0] = np.clip(x1 + (x2 - x1) * (y - y1) / (y2 - y1), 0, width - 1)
        findings.append({
            "side": side,
            "contrast": round(contrast, 2),
            "strip_consistency": round(strip_consistency, 3),
            "offset_fraction": round(offset / width, 4),
        })
    if not findings:
        return quad, None
    refined = adjusted / [width - 1, height - 1]
    try:
        return validate_roi(refined), findings
    except ValueError:
        return quad, None


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
        # Keep the GMM initialization stable across re-renders of one frame.
        cv2.setRNGSeed(0)
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

    candidate, layered_sheets = _nested_sheet_boundary(working, candidate)

    initial_area = float(cv2.contourArea(original))
    candidate_area = float(cv2.contourArea(candidate))
    area_ratio = candidate_area / initial_area
    corner_shift = float(np.max(np.linalg.norm(candidate - original, axis=1)))
    max_corner_shift = 0.18 if layered_sheets else 0.12
    if not 0.65 <= area_ratio <= 1.45 or corner_shift > max_corner_shift:
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
    if (area_ratio < 0.98 and edge_occupancy > 0.55 and not layered_sheets) or (
        area_ratio > 1.02 and edge_occupancy < 0.55
    ):
        return original.tolist(), {**info, **evidence, "status": "edge_evidence_rejected"}
    if layered_sheets and area_ratio < 0.98 and edge_occupancy > 0.55:
        corrected_sides = {finding["side"] for finding in layered_sheets}
        for side, indices in (("left", (0, 3)), ("right", (1, 2))):
            moved = np.linalg.norm(candidate[list(indices)] - original[list(indices)], axis=1)
            if side not in corrected_sides and np.max(moved) > 0.025:
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
        **({"layered_sheets": layered_sheets} if layered_sheets else {}),
    }
