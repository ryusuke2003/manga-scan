"""Automatic single-cover quadrilateral detection."""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from .perspective import validate_roi


def _line_intersection(first, second):
    matrix = np.asarray(
        [
            [math.cos(first[1]), math.sin(first[1])],
            [math.cos(second[1]), math.sin(second[1])],
        ],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 0.15:
        return None
    return np.linalg.solve(matrix, np.asarray([first[0], second[0]], dtype=np.float64))


def _edge_support(distance, quad):
    height, width = distance.shape
    samples = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        count = max(12, round(float(np.linalg.norm(end - start)) / 4))
        points = np.linspace(start, end, count)
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        samples.extend(distance[y, x])
    return float(np.mean(np.asarray(samples) <= 3.0))


def _long_line_segments(gray, width, height):
    """Convert long LSD segments to Hough-style lines with a length score."""

    try:
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    except cv2.error:
        return []
    if detected is None:
        return []
    minimum_length = max(40.0, min(width, height) * 0.18)
    normalizer = max(float(width), float(height), 1.0)
    records = []
    for segment in detected.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, segment)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < minimum_length:
            continue
        normal_x = -dy / length
        normal_y = dx / length
        theta = math.atan2(normal_y, normal_x)
        rho = normal_x * x1 + normal_y * y1
        if theta < 0:
            theta += math.pi
            rho = -rho
        elif theta >= math.pi:
            theta -= math.pi
            rho = -rho
        length_ratio = min(1.0, length / normalizer)
        quality = 0.55 + 0.45 * length_ratio
        records.append((rho, theta, quality, length_ratio))
    return records


def _axis_lines(raw_lines, width, height):
    horizontal = []
    vertical = []
    for rho, theta, quality, length_ratio in sorted(
        raw_lines,
        key=lambda item: item[2],
        reverse=True,
    ):
        direction = (math.degrees(float(theta)) + 90) % 180
        if min(abs(direction), abs(direction - 180)) < 15:
            family = horizontal
            denominator = math.sin(float(theta))
            if abs(denominator) < 1e-6:
                continue
            position = (float(rho) - math.cos(float(theta)) * width / 2) / denominator
            limit = height
        elif abs(direction - 90) < 15:
            family = vertical
            denominator = math.cos(float(theta))
            if abs(denominator) < 1e-6:
                continue
            position = (float(rho) - math.sin(float(theta)) * height / 2) / denominator
            limit = width
        else:
            continue
        if not -0.08 * limit <= position <= 1.08 * limit:
            continue
        if len(family) >= 20:
            continue
        if any(abs(position - existing[2]) < 7 for existing in family):
            continue
        family.append(
            (
                float(rho),
                float(theta),
                float(position),
                float(quality),
                float(length_ratio),
            )
        )
        if len(horizontal) >= 20 and len(vertical) >= 20:
            break
    return horizontal, vertical


def _boundary_line_count(quad, width, height):
    """Count candidate sides that hug the image boundary."""
    return sum(
        (
            np.all(quad[[0, 3], 0] < width * 0.01),
            np.all(quad[[1, 2], 0] > width * 0.99),
            np.all(quad[[0, 1], 1] < height * 0.01),
            np.all(quad[[2, 3], 1] > height * 0.99),
        )
    )


def _intersection_with_side(start, end, side_start, side_end):
    """Return the crossing point and its position along a side."""

    direction = end - start
    side = side_end - side_start
    matrix = np.column_stack((direction, -side))
    if abs(float(np.linalg.det(matrix))) < 1e-5:
        return None
    _, fraction = np.linalg.solve(matrix, side_start - start)
    return side_start + fraction * side, float(fraction)


def _refine_cover_face_bottom(image, roi):
    """Follow the front cover when Hough selected the book block below it.

    A thick book can expose its page block below the front cover. Its straight
    bottom edge often outranks the cover edge in Hough voting. Require a long
    line anchored at the side of the existing detection and a perspective
    slope consistent with the top edge before moving either bottom corner.
    A strong color boundary can also correct the upper edge in this case.
    """

    scale = min(1.0, 1000 / max(image.shape[:2]))
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    height, width = working.shape[:2]
    quad = np.asarray(roi, dtype=np.float32) * [width - 1, height - 1]
    top_angle = math.degrees(
        math.atan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0])
    )
    bottom_angle = math.degrees(
        math.atan2(quad[2, 1] - quad[3, 1], quad[2, 0] - quad[3, 0])
    )
    if abs(top_angle) < 2 or abs(top_angle - bottom_angle) < 4:
        return roi

    gray = (
        working
        if working.ndim == 2
        else cv2.cvtColor(
            working,
            cv2.COLOR_BGRA2GRAY if working.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
        )
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(35, round(width * 0.045)),
        minLineLength=max(65, round(width * 0.13)),
        maxLineGap=max(12, round(width * 0.03)),
    )
    if segments is None:
        return roi

    page_width = float(np.linalg.norm(quad[2] - quad[3]))
    best = None
    for x1, y1, x2, y2 in segments[:, 0]:
        start = np.asarray([x1, y1], dtype=np.float64)
        end = np.asarray([x2, y2], dtype=np.float64)
        if start[0] > end[0]:
            start, end = end, start
        if np.linalg.norm(end - start) < page_width * 0.45:
            continue
        angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        if angle * top_angle <= 0 or not 5 <= abs(angle) <= 20:
            continue
        if abs(angle - top_angle) > 7:
            continue

        left = _intersection_with_side(start, end, quad[0], quad[3])
        right = _intersection_with_side(start, end, quad[1], quad[2])
        if left is None or right is None:
            continue
        left_point, left_fraction = left
        right_point, right_fraction = right
        if not (0.72 <= left_fraction <= 0.94 and 0.72 <= right_fraction <= 0.94):
            continue
        if abs(left_fraction - right_fraction) > 0.18:
            continue
        span = right_point[0] - left_point[0]
        if span <= 0 or abs(start[0] - left_point[0]) > span * 0.10:
            continue
        if end[0] < left_point[0] + span * 0.55:
            continue
        # Prefer the lowest supported cover edge when artwork supplies several
        # long, roughly parallel interior lines.
        score = (left_fraction + right_fraction) / 2
        if best is None or score > best[0]:
            best = (score, left_point, right_point)

    if best is None:
        return roi
    refined = quad.copy()
    refined[3], refined[2] = best[1:]

    if working.ndim == 3 and working.shape[2] == 3:
        lab = cv2.cvtColor(
            cv2.GaussianBlur(working, (11, 11), 0), cv2.COLOR_BGR2LAB
        )
        top_width = float(np.linalg.norm(quad[1] - quad[0]))
        best_top = None
        for x1, y1, x2, y2 in segments[:, 0]:
            start = np.asarray([x1, y1], dtype=np.float64)
            end = np.asarray([x2, y2], dtype=np.float64)
            if start[0] > end[0]:
                start, end = end, start
            if np.linalg.norm(end - start) < top_width * 0.40:
                continue
            angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
            if angle * top_angle <= 0 or not 5 <= abs(angle) <= 20:
                continue
            if abs(angle - top_angle) < 4:
                continue
            left = _intersection_with_side(start, end, quad[0], quad[3])
            right = _intersection_with_side(start, end, quad[1], quad[2])
            if left is None or right is None:
                continue
            left_point, left_fraction = left
            right_point, right_fraction = right
            if not (-0.03 <= left_fraction <= 0.18 and -0.03 <= right_fraction <= 0.18):
                continue
            span = right_point[0] - left_point[0]
            if span <= 0 or abs(start[0] - left_point[0]) > span * 0.10:
                continue
            if end[0] < left_point[0] + span * 0.55:
                continue
            xx = np.linspace(left_point[0] + span * 0.08, right_point[0] - span * 0.08, 40)
            yy = left_point[1] + (xx - left_point[0]) * (
                (right_point[1] - left_point[1]) / span
            )
            indices_x = np.clip(np.rint(xx).astype(int), 0, width - 1)
            above = lab[np.clip(np.rint(yy - 6).astype(int), 0, height - 1), indices_x]
            below = lab[np.clip(np.rint(yy + 6).astype(int), 0, height - 1), indices_x]
            contrast = float(
                np.mean(np.linalg.norm(above.astype(float) - below.astype(float), axis=1))
            )
            if contrast >= 35 and (best_top is None or contrast > best_top[0]):
                best_top = (contrast, left_point, right_point)
        if best_top is not None:
            refined[0], refined[1] = best_top[1:]

    normalized = refined / [width - 1, height - 1]
    try:
        return validate_roi(normalized).tolist()
    except ValueError:
        return roi


def detect_cover_quad_candidates(
    image,
    min_confidence=0.62,
    *,
    area_range=(0.06, 0.78),
    aspect_range=(0.35, 1.25),
    target_aspect=0.72,
    limit=8,
):
    """Rank Hough outline candidates as normalized TL/TR/BR/BL points.

    Keeping several geometrically distinct candidates lets callers verify them
    against stronger page evidence instead of committing to the strongest desk
    or background rectangle too early.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")
    area_min, area_max = map(float, area_range)
    aspect_min, aspect_max = map(float, aspect_range)
    target_aspect = float(target_aspect)
    if not 0 < area_min < area_max <= 1:
        raise ValueError("area_range must satisfy 0 < min < max <= 1")
    if not 0 < aspect_min < aspect_max or target_aspect <= 0:
        raise ValueError("aspect range/target must be positive")
    if int(limit) < 1:
        raise ValueError("limit must be at least 1")

    scale = min(1.0, 1000 / max(image.shape[:2]))
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    height, width = working.shape[:2]
    if working.ndim == 2:
        gray = working
    else:
        code = cv2.COLOR_BGRA2GRAY if working.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(working, code)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(20, min(100, median * 0.45)))
    high = int(min(220, max(low + 35, median * 1.2)))
    edges = cv2.bitwise_or(cv2.Canny(gray, low, high), cv2.Canny(gray, 30, 100))

    threshold = max(45, round(min(width, height) * 0.12))
    detected_lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold)
    raw_lines = []
    if detected_lines is not None:
        raw_lines.extend(
            (
                float(rho),
                float(theta),
                math.exp(-rank / 160),
                0.0,
            )
            for rank, (rho, theta) in enumerate(detected_lines[:, 0])
        )
    raw_lines.extend(_long_line_segments(gray, width, height))
    horizontal, vertical = _axis_lines(raw_lines, width, height)
    if len(horizontal) < 2 or len(vertical) < 2:
        return []

    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    candidates = []
    image_area = float(width * height)
    for first_horizontal, second_horizontal in itertools.combinations(horizontal, 2):
        top, bottom = sorted((first_horizontal, second_horizontal), key=lambda line: line[2])
        if bottom[2] - top[2] < height * 0.25:
            continue
        for first_vertical, second_vertical in itertools.combinations(vertical, 2):
            left, right = sorted((first_vertical, second_vertical), key=lambda line: line[2])
            if right[2] - left[2] < width * 0.12:
                continue
            points = [
                _line_intersection(top, left),
                _line_intersection(top, right),
                _line_intersection(bottom, right),
                _line_intersection(bottom, left),
            ]
            if any(point is None for point in points):
                continue
            quad = np.asarray(points, dtype=np.float32)
            if (quad < [-0.02 * width, -0.02 * height]).any() or (
                quad > [1.02 * width, 1.02 * height]
            ).any():
                continue
            area_ratio = abs(float(cv2.contourArea(quad))) / image_area
            if not area_min <= area_ratio <= area_max:
                continue
            lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            aspect = float((lengths[0] + lengths[2]) / max(lengths[1] + lengths[3], 1))
            if not aspect_min <= aspect <= aspect_max:
                continue
            area_score = min(area_ratio / 0.3, 1.0)
            aspect_score = math.exp(-abs(math.log(aspect / target_aspect)))
            rank_score = sum(line[3] for line in (top, bottom, left, right)) / 4
            segment_score = sum(line[4] for line in (top, bottom, left, right)) / 4
            boundary_lines = _boundary_line_count(quad, width, height)
            preliminary = (
                0.28 * area_score
                + 0.20 * aspect_score
                + 0.18 * rank_score
                + 0.06 * segment_score
                - 0.18 * boundary_lines
            )
            candidates.append((preliminary, quad))

    if not candidates:
        return []
    scored = []
    for preliminary, quad in sorted(candidates, key=lambda candidate: candidate[0], reverse=True)[:120]:
        support = _edge_support(distance, quad)
        scored.append((preliminary + 0.28 * support, support, quad))
    results = []
    scale_vector = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    for confidence, support, quad in sorted(scored, key=lambda item: item[0], reverse=True):
        normalized = np.clip(quad / scale_vector, 0, 1)
        try:
            roi = validate_roi(normalized).tolist()
        except ValueError:
            continue
        # Hough emits many near-identical line combinations. Do not let those
        # crowd genuinely different outline hypotheses out of the top-N set.
        if any(
            float(np.mean(np.linalg.norm(normalized - np.asarray(item["roi"]), axis=1)))
            < 0.025
            for item in results
        ):
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        results.append(
            {
                "detected": bool(confidence >= min_confidence and support >= 0.28),
                "confidence": round(confidence, 4),
                "support": round(float(support), 4),
                "roi": roi,
            }
        )
        if len(results) >= int(limit):
            break
    return results


def detect_cover_quad(
    image,
    min_confidence=0.62,
    *,
    area_range=(0.06, 0.78),
    aspect_range=(0.35, 1.25),
    target_aspect=0.72,
):
    """Find the strongest upright book-cover outline.

    This compatibility wrapper preserves the original single-result API. New
    spread detection uses :func:`detect_cover_quad_candidates` and verifies
    several hypotheses against the left/right page contours.
    """

    candidates = detect_cover_quad_candidates(
        image,
        min_confidence=min_confidence,
        area_range=area_range,
        aspect_range=aspect_range,
        target_aspect=target_aspect,
        limit=1,
    )
    if not candidates:
        return {"detected": False, "confidence": 0.0, "roi": None}
    best = candidates[0]
    if not best["detected"]:
        return {"detected": False, "confidence": best["confidence"], "roi": None}
    refined_roi = _refine_cover_face_bottom(image, best["roi"])
    if refined_roi != best["roi"]:
        best["roi"] = refined_roi
        best["face_refined"] = True
    return best
