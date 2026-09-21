"""Align nearby video frames before comparing normalized page geometry."""

from __future__ import annotations

import cv2
import numpy as np

_MIN_TRACKED_POINTS = 12
_MIN_HOMOGRAPHY_INLIERS = 8
_MIN_INLIER_RATIO = 0.40
_MAX_FORWARD_BACKWARD_ERROR = 1.75
_MAX_REPROJECTION_ERROR = 3.0


def _gray(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


def _unavailable(frame_index, status, **details):
    return {
        "frame_index": int(frame_index),
        "status": status,
        "matrix": None,
        "tracked_points": int(details.pop("tracked_points", 0)),
        "inliers": int(details.pop("inliers", 0)),
        **details,
    }


def _estimate_to_anchor(anchor_gray, target_gray, anchor_points, frame_index):
    if target_gray.shape != anchor_gray.shape:
        return _unavailable(frame_index, "shape_mismatch")
    if anchor_points is None or len(anchor_points) < _MIN_TRACKED_POINTS:
        return _unavailable(frame_index, "insufficient_features")

    lk_options = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": 1e-4,
    }
    target_points, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
        anchor_gray,
        target_gray,
        anchor_points,
        None,
        **lk_options,
    )
    if target_points is None or forward_status is None:
        return _unavailable(frame_index, "optical_flow_failed")
    returned_points, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        target_gray,
        anchor_gray,
        target_points,
        None,
        **lk_options,
    )
    if returned_points is None or backward_status is None:
        return _unavailable(frame_index, "reverse_flow_failed")

    anchor_flat = anchor_points.reshape(-1, 2)
    target_flat = target_points.reshape(-1, 2)
    returned_flat = returned_points.reshape(-1, 2)
    valid = forward_status.reshape(-1).astype(bool)
    valid &= backward_status.reshape(-1).astype(bool)
    valid &= np.isfinite(target_flat).all(axis=1)
    valid &= np.isfinite(returned_flat).all(axis=1)
    valid &= np.linalg.norm(returned_flat - anchor_flat, axis=1) <= _MAX_FORWARD_BACKWARD_ERROR
    anchor_good = anchor_flat[valid]
    target_good = target_flat[valid]
    tracked_points = len(anchor_good)
    if tracked_points < _MIN_TRACKED_POINTS:
        return _unavailable(
            frame_index,
            "insufficient_tracks",
            tracked_points=tracked_points,
        )

    # Points were detected in the anchor and tracked into the target. Reverse
    # their order here so the resulting homography maps target -> anchor.
    matrix, inlier_mask = cv2.findHomography(
        target_good,
        anchor_good,
        cv2.RANSAC,
        _MAX_REPROJECTION_ERROR,
    )
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return _unavailable(
            frame_index,
            "homography_failed",
            tracked_points=tracked_points,
        )
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(tracked_points, 1)
    if inlier_count < _MIN_HOMOGRAPHY_INLIERS or inlier_ratio < _MIN_INLIER_RATIO:
        return _unavailable(
            frame_index,
            "insufficient_inliers",
            tracked_points=tracked_points,
            inliers=inlier_count,
            inlier_ratio=round(float(inlier_ratio), 4),
        )

    projected = cv2.perspectiveTransform(
        target_good[inliers].reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)
    reprojection_error = float(
        np.median(np.linalg.norm(projected - anchor_good[inliers], axis=1))
    )
    height, width = anchor_gray.shape[:2]
    hull = cv2.convexHull(anchor_good[inliers].astype(np.float32))
    coverage = abs(float(cv2.contourArea(hull))) / max(float(width * height), 1.0)
    if reprojection_error > _MAX_REPROJECTION_ERROR or coverage < 0.015:
        return _unavailable(
            frame_index,
            "unstable_homography",
            tracked_points=tracked_points,
            inliers=inlier_count,
            inlier_ratio=round(float(inlier_ratio), 4),
            reprojection_error=round(reprojection_error, 4),
            coverage=round(coverage, 4),
        )

    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    mapped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    area_ratio = abs(float(cv2.contourArea(mapped_corners))) / max(float(width * height), 1.0)
    margin = np.asarray([width, height], dtype=np.float32) * 0.35
    if (
        not 0.50 <= area_ratio <= 1.80
        or (mapped_corners < -margin).any()
        or (mapped_corners > np.asarray([width - 1, height - 1]) + margin).any()
    ):
        return _unavailable(
            frame_index,
            "implausible_homography",
            tracked_points=tracked_points,
            inliers=inlier_count,
            inlier_ratio=round(float(inlier_ratio), 4),
            reprojection_error=round(reprojection_error, 4),
            coverage=round(coverage, 4),
            area_ratio=round(area_ratio, 4),
        )

    matrix = matrix / matrix[2, 2] if abs(float(matrix[2, 2])) > 1e-9 else matrix
    return {
        "frame_index": int(frame_index),
        "status": "aligned",
        "matrix": matrix,
        "tracked_points": int(tracked_points),
        "inliers": inlier_count,
        "inlier_ratio": round(float(inlier_ratio), 4),
        "reprojection_error": round(reprojection_error, 4),
        "coverage": round(coverage, 4),
        "area_ratio": round(area_ratio, 4),
    }


def estimate_frame_alignments(images, anchor_index):
    """Estimate homographies that map every supplied frame into the anchor."""

    images = list(images)
    if not images:
        raise ValueError("images must contain at least one frame")
    if not 0 <= int(anchor_index) < len(images):
        raise ValueError("anchor_index must identify one supplied frame")
    grays = [_gray(image) for image in images]
    anchor_index = int(anchor_index)
    anchor_gray = grays[anchor_index]
    anchor_points = cv2.goodFeaturesToTrack(
        anchor_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    results = []
    for frame_index, target_gray in enumerate(grays):
        if frame_index == anchor_index:
            results.append(
                {
                    "frame_index": frame_index,
                    "status": "anchor",
                    "matrix": np.eye(3, dtype=np.float64),
                    "tracked_points": int(len(anchor_points)) if anchor_points is not None else 0,
                    "inliers": int(len(anchor_points)) if anchor_points is not None else 0,
                }
            )
        else:
            results.append(
                _estimate_to_anchor(
                    anchor_gray,
                    target_gray,
                    anchor_points,
                    frame_index,
                )
            )
    return results


def transform_normalized_quad(quad, matrix, source_shape, destination_shape):
    """Transform a normalized quad between pixel coordinate systems."""

    points = np.asarray(quad, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("quad must contain four finite [x,y] pairs")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3x3 homography")
    source_height, source_width = source_shape[:2]
    destination_height, destination_width = destination_shape[:2]
    source_scale = np.asarray(
        [max(source_width - 1, 1), max(source_height - 1, 1)],
        dtype=np.float32,
    )
    destination_scale = np.asarray(
        [max(destination_width - 1, 1), max(destination_height - 1, 1)],
        dtype=np.float32,
    )
    pixels = points * source_scale
    transformed = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    normalized = transformed / destination_scale
    if not np.isfinite(normalized).all() or (normalized < -0.05).any() or (normalized > 1.05).any():
        raise ValueError("transformed quad falls outside the destination frame")
    return np.clip(normalized, 0, 1).astype(np.float32)


def align_page_detection(detection, alignment, source_shape, anchor_shape):
    """Map one page detection into anchor coordinates when alignment is valid."""

    result = dict(detection)
    matrix = alignment.get("matrix")
    applied = matrix is not None and alignment.get("status") in ("anchor", "aligned")
    for side in ("left", "right"):
        original = dict(detection.get(side) or {})
        if applied:
            try:
                original["quad"] = transform_normalized_quad(
                    original.get("quad"),
                    matrix,
                    source_shape,
                    anchor_shape,
                ).tolist()
            except (ValueError, cv2.error):
                applied = False
                break
        result[side] = original
    if not applied:
        result = dict(detection)
        result["left"] = dict(detection.get("left") or {})
        result["right"] = dict(detection.get("right") or {})
    result["alignment_applied"] = bool(applied and alignment.get("status") == "aligned")
    result["alignment_status"] = alignment.get("status", "unavailable")
    return result


def alignment_summary(alignments):
    """Return JSON-safe diagnostics without exposing homography matrices."""

    frames = []
    for item in alignments:
        frames.append({key: value for key, value in item.items() if key != "matrix"})
    return {
        "attempted": max(0, len(frames) - 1),
        "aligned": sum(item["status"] == "aligned" for item in frames),
        "frames": frames,
    }
