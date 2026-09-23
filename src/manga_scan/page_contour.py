"""Conservative left/right page contour detection near a known spread ROI."""

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


def quad_edge_evidence(image, quad, occlusion_mask=None):
    """Measure real image-edge evidence for each side of a normalized quad.

    Support is measured against the full side length, so a hand covering part
    of an edge lowers confidence instead of making the remaining visible part
    look artificially strong. Visible-only support is kept for diagnostics.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    points = np.asarray(quad, dtype=np.float32)
    if (
        points.shape != (4, 2)
        or not np.isfinite(points).all()
        or (points < 0).any()
        or (points > 1).any()
    ):
        raise ValueError("quad must contain four normalized finite [x,y] points")

    height, width = image.shape[:2]
    pixel_points = np.rint(points * [width - 1, height - 1]).astype(np.int32)
    edges = _edge_map(image)
    radius = max(2, round(min(height, width) * 0.004))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    nearby_edges = cv2.dilate(edges, kernel, iterations=1)

    if occlusion_mask is None:
        occluded = np.zeros((height, width), np.uint8)
    else:
        occluded = np.asarray(occlusion_mask)
        if occluded.ndim != 2:
            raise ValueError("occlusion_mask must be a grayscale mask")
        if occluded.shape != (height, width):
            occluded = cv2.resize(
                occluded.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        occluded = (occluded > 0).astype(np.uint8) * 255
        occluded = cv2.dilate(occluded, np.ones((5, 5), np.uint8), iterations=1)

    evidence = {}
    edge_pairs = (
        ("top", 0, 1),
        ("right", 1, 2),
        ("bottom", 2, 3),
        ("left", 3, 0),
    )
    thickness = max(1, round(min(height, width) * 0.002))
    for name, first, second in edge_pairs:
        line = np.zeros((height, width), np.uint8)
        cv2.line(
            line,
            tuple(pixel_points[first]),
            tuple(pixel_points[second]),
            255,
            thickness,
            cv2.LINE_AA,
        )
        line_pixels = line > 0
        total = int(np.count_nonzero(line_pixels))
        if total == 0:
            evidence[name] = {
                "support": 0.0,
                "visible_support": 0.0,
                "occlusion": 0.0,
            }
            continue
        hidden = line_pixels & (occluded > 0)
        visible = line_pixels & ~hidden
        matched = visible & (nearby_edges > 0)
        support = float(np.count_nonzero(matched) / total)
        visible_count = int(np.count_nonzero(visible))
        visible_support = float(np.count_nonzero(matched) / max(1, visible_count))
        evidence[name] = {
            "support": round(support, 4),
            "visible_support": round(visible_support, 4),
            "occlusion": round(float(np.count_nonzero(hidden) / total), 4),
        }
    evidence["minimum_support"] = round(
        min(item["support"] for item in evidence.values() if isinstance(item, dict)),
        4,
    )
    return evidence


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


def _paper_outline(image, prior, side):
    """Track neutral paper against a colored desk within 5% of the reference.

    Unlike ink contours, this silhouette may extend outside the original ROI.
    Reject outlines made by the search window itself, and keep the spine fixed.
    Grayscale/colored pages fall back to ordinary edge detection.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    paper = ((hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 65)).astype(np.uint8) * 255
    kernel_size = max(3, round(min(h, w) * 0.016) | 1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((kernel_size,) * 2, np.uint8))
    search = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(search, np.rint(prior).astype(np.int32), 255)
    if np.mean(hsv[:, :, 1][search > 0] < 85) < 0.65:
        return None
    radius = max(2, round(max(h, w) * 0.05))
    search = cv2.dilate(search, np.ones((2 * radius + 1,) * 2, np.uint8))
    a, b = (prior[1], prior[2]) if side == "left" else (prior[0], prior[3])
    yy, xx = np.indices((h, w))
    cross = (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])
    search[cross < 0 if side == "left" else cross > 0] = 0
    contours, _ = cv2.findContours(paper & search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    quads = _quad_candidates(contour)
    if not quads:
        return None
    quad = quads[0]
    area = abs(cv2.contourArea(quad))
    prior_area = abs(cv2.contourArea(prior))
    intersection, _ = cv2.intersectConvexConvex(quad, prior)
    if not (0.85 < area / prior_area < 1.3 and intersection / prior_area > 0.8):
        return None
    if np.max(np.linalg.norm((quad - prior) / [w, h], axis=1)) > 0.10:
        return None

    # A neutral desk or missing boundary must not turn the dilated search ROI
    # into a fabricated paper edge. Require colored background along the real
    # silhouette, excluding the artificial spine and the source frame boundary.
    points = contour.reshape(-1, 2)
    distance = cv2.distanceTransform(search, cv2.DIST_L2, 5)
    spine_distance = np.abs(cross[points[:, 1], points[:, 0]]) / max(np.linalg.norm(b - a), 1)
    exterior = (
        (spine_distance > 3) & (points[:, 0] > 1) & (points[:, 0] < w - 2)
        & (points[:, 1] > 1) & (points[:, 1] < h - 2)
    )
    boundary = points[exterior]
    if len(boundary) < 10 or np.mean(distance[boundary[:, 1], boundary[:, 0]] < 2) > 0.1:
        return None
    silhouette = np.zeros((h, w), np.uint8)
    cv2.drawContours(silhouette, [contour], -1, 255, -1)
    ring = cv2.dilate(silhouette, np.ones((7, 7), np.uint8)) & ~silhouette & search
    if not np.any(ring) or np.mean(hsv[:, :, 1][ring > 0] >= 85) < 0.45:
        return None
    return quad, 0.85


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
    """Detect left/right page quads near a known spread ROI.

    Returned quads are normalized TL,TR,BR,BL coordinates in the input image.
    Each side falls back to its reference split when no sufficiently confident
    contour is found. A verified paper silhouette can follow small outward
    shifts; ordinary ink contours remain constrained to the reference.
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
        outline = _paper_outline(image, prior, side)
        if outline is not None:
            detected_quad, confidence = outline
        detected = detected_quad is not None and confidence >= min_confidence
        selected = detected_quad if detected else prior
        result[side] = {
            "quad": _normalize_quad(selected, image.shape),
            "confidence": round(float(confidence), 4),
            "detected": bool(detected),
            "touches_frame": bool(detected and np.any(
                (selected <= 1) | (selected >= np.asarray([image.shape[1], image.shape[0]]) - 2)
            )),
        }

    result["confidence"] = min(result["left"]["confidence"], result["right"]["confidence"])
    result["detected"] = result["left"]["detected"] and result["right"]["detected"]
    return result


def _quad_distance(first, second):
    return float(np.mean(np.linalg.norm(first - second, axis=1)))


def _consensus_inlier_indices(samples, max_corner_deviation, anchor_id=None):
    """Pick the largest mutually-close quad cluster, then trim residual outliers."""

    count = len(samples)
    if count <= 1:
        return list(range(count))

    quads = [sample["quad"] for sample in samples]
    distances = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            distance = _quad_distance(quads[i], quads[j])
            distances[i, j] = distance
            distances[j, i] = distance

    best = None
    best_key = None
    for i in range(count):
        cluster = np.flatnonzero(distances[i] <= max_corner_deviation).tolist()
        ids = {samples[index]["candidate_id"] for index in cluster}
        confidence = sum(samples[index]["confidence"] for index in cluster)
        key = (len(cluster), int(anchor_id in ids), confidence)
        if best_key is None or key > best_key:
            best = cluster
            best_key = key

    if len(best) < 3:
        return best

    cluster_quads = np.stack([quads[index] for index in best])
    median_quad = np.median(cluster_quads, axis=0)
    residuals = np.asarray(
        [_quad_distance(quads[index], median_quad) for index in best],
        dtype=np.float32,
    )
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    robust_limit = max(0.004, median_residual + 3.0 * 1.4826 * mad)
    robust_limit = min(float(max_corner_deviation), robust_limit)
    refined = [
        index
        for index, residual in zip(best, residuals)
        if float(residual) <= robust_limit + 1e-9
    ]
    return refined if len(refined) >= 2 else best


def consensus_page_quads(
    detections,
    *,
    min_confidence=0.5,
    max_corner_deviation=0.04,
    anchor_ids=None,
):
    """Combine page detections from several candidate frames.

    Each side is combined independently so one frame may contribute only the
    page boundary that is visible there. A RANSAC-like largest-cluster step and
    median/MAD trimming reject shifted outlines; the remaining corners are
    averaged with contour confidence as the weight. With a single usable
    detection this intentionally behaves like the existing one-frame path.
    """

    detections = list(detections)
    if not detections:
        raise ValueError("detections must contain at least one result")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")
    if not 0 < float(max_corner_deviation) <= 0.5:
        raise ValueError("max_corner_deviation must be 0..0.5")
    anchor_ids = anchor_ids or {}

    result = {}
    source_ids = [
        detection.get("candidate_id", index)
        for index, detection in enumerate(detections)
    ]

    for side in ("left", "right"):
        fallback_detection = next(
            (
                detection
                for index, detection in enumerate(detections)
                if detection.get("candidate_id", index) == anchor_ids.get(side)
            ),
            detections[0],
        )
        fallback = fallback_detection.get(side, {})
        fallback_quad = np.asarray(fallback.get("quad"), dtype=np.float32)
        if fallback_quad.shape != (4, 2) or not np.isfinite(fallback_quad).all():
            raise ValueError("each detection must contain valid left/right quads")

        samples = []
        for index, detection in enumerate(detections):
            data = detection.get(side, {})
            if not isinstance(data, dict) or not data.get("detected"):
                continue
            quad = np.asarray(data.get("quad"), dtype=np.float32)
            try:
                confidence = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                quad.shape != (4, 2)
                or not np.isfinite(quad).all()
                or (quad < 0).any()
                or (quad > 1).any()
                or not math.isfinite(confidence)
                or confidence <= 0
            ):
                continue
            samples.append(
                {
                    "candidate_id": detection.get("candidate_id", index),
                    "quad": quad,
                    "confidence": confidence,
                    "touches_frame": bool(data.get("touches_frame")),
                }
            )

        if not samples:
            result[side] = {
                "quad": fallback_quad.tolist(),
                "confidence": round(float(fallback.get("confidence", 0.0)), 4),
                "detected": False,
                "touches_frame": False,
                "consensus_count": 0,
                "consensus_candidate_ids": [],
                "consensus_outlier_ids": [],
            }
            continue

        inlier_indices = _consensus_inlier_indices(
            samples,
            float(max_corner_deviation),
            anchor_ids.get(side),
        )
        inliers = [samples[index] for index in inlier_indices]
        weights = np.asarray(
            [max(sample["confidence"], 1e-6) for sample in inliers],
            dtype=np.float64,
        )
        quads = np.stack([sample["quad"] for sample in inliers]).astype(np.float64)
        selected_quad = np.average(quads, axis=0, weights=weights).astype(np.float32)
        confidence = float(
            np.average(
                np.asarray([sample["confidence"] for sample in inliers], dtype=np.float64),
                weights=weights,
            )
        )
        detected = confidence >= float(min_confidence)
        if not detected:
            selected_quad = fallback_quad

        inlier_ids = [sample["candidate_id"] for sample in inliers]
        all_ids = [sample["candidate_id"] for sample in samples]
        result[side] = {
            "quad": selected_quad.tolist(),
            "confidence": round(confidence, 4),
            "detected": bool(detected),
            "touches_frame": bool(
                detected and any(sample["touches_frame"] for sample in inliers)
            ),
            "consensus_count": len(inliers),
            "consensus_candidate_ids": inlier_ids,
            "consensus_outlier_ids": [
                candidate_id for candidate_id in all_ids if candidate_id not in set(inlier_ids)
            ],
        }

    result["confidence"] = min(result["left"]["confidence"], result["right"]["confidence"])
    result["detected"] = result["left"]["detected"] and result["right"]["detected"]
    result["consensus"] = {
        "candidate_count": len(detections),
        "candidate_ids": source_ids,
    }
    return result


def spread_quad_from_page_quads(result):
    """Build one spread crop from the outer corners of two detected pages.

    The inner page corners are deliberately not used for warping. This keeps
    the photographed gutter intact while still letting each page boundary
    contribute to automatic outer-edge detection.
    """
    if not isinstance(result, dict) or not result.get("detected"):
        raise ValueError("both page quads must be detected")
    left = np.asarray(result.get("left", {}).get("quad"), dtype=np.float32)
    right = np.asarray(result.get("right", {}).get("quad"), dtype=np.float32)
    if left.shape != (4, 2) or right.shape != (4, 2):
        raise ValueError("page detection must contain left/right quads")
    spread = np.asarray([left[0], right[1], right[2], left[3]], dtype=np.float32)
    return validate_roi(spread).tolist()


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
        if data.get("consensus_count"):
            label += f" n={data['consensus_count']}"
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
