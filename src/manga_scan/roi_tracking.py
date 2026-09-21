from __future__ import annotations

import math

import cv2
import numpy as np

from .perspective import pixel_quad, validate_roi
from .temporal_alignment import transform_normalized_quad

_MIN_TRACKED_POINTS = 12
_MIN_HOMOGRAPHY_INLIERS = 8
_MIN_INLIER_RATIO = 0.45
_MAX_FORWARD_BACKWARD_ERROR = 1.75
_MAX_REPROJECTION_ERROR = 3.0


def _gray(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _quad_area(quad):
    points = np.asarray(quad, dtype=np.float32)
    return abs(float(cv2.contourArea(points)))


def _book_homography(previous_image, current_image, previous_roi):
    previous_gray = _gray(previous_image)
    current_gray = _gray(current_image)
    if previous_gray.shape != current_gray.shape:
        return None, {"status": "shape_mismatch"}

    mask = np.zeros(previous_gray.shape, np.uint8)
    polygon = np.rint(pixel_quad(previous_roi, previous_image.shape)).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    # Avoid giving the outer paper/desk boundary disproportionate weight. The
    # interior manga lines are more representative of the book's own motion.
    kernel_size = max(3, int(round(min(previous_gray.shape) * 0.02)) | 1)
    inner_mask = cv2.erode(
        mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )
    if np.count_nonzero(inner_mask) < 0.2 * np.count_nonzero(mask):
        inner_mask = mask

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=inner_mask,
    )
    if points is None or len(points) < _MIN_TRACKED_POINTS:
        return None, {
            "status": "insufficient_features",
            "tracked_points": 0 if points is None else int(len(points)),
        }

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
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        **lk_options,
    )
    if current_points is None or forward_status is None:
        return None, {"status": "optical_flow_failed"}
    returned_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points,
        None,
        **lk_options,
    )
    if returned_points is None or backward_status is None:
        return None, {"status": "reverse_flow_failed"}

    previous_flat = points.reshape(-1, 2)
    current_flat = current_points.reshape(-1, 2)
    returned_flat = returned_points.reshape(-1, 2)
    valid = forward_status.reshape(-1).astype(bool)
    valid &= backward_status.reshape(-1).astype(bool)
    valid &= np.isfinite(current_flat).all(axis=1)
    valid &= np.isfinite(returned_flat).all(axis=1)
    valid &= (
        np.linalg.norm(returned_flat - previous_flat, axis=1)
        <= _MAX_FORWARD_BACKWARD_ERROR
    )
    previous_good = previous_flat[valid]
    current_good = current_flat[valid]
    tracked_points = len(previous_good)
    if tracked_points < _MIN_TRACKED_POINTS:
        return None, {
            "status": "insufficient_tracks",
            "tracked_points": int(tracked_points),
        }

    matrix, inlier_mask = cv2.findHomography(
        previous_good,
        current_good,
        cv2.RANSAC,
        _MAX_REPROJECTION_ERROR,
    )
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return None, {
            "status": "homography_failed",
            "tracked_points": int(tracked_points),
        }

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(tracked_points, 1)
    if inlier_count < _MIN_HOMOGRAPHY_INLIERS or inlier_ratio < _MIN_INLIER_RATIO:
        return None, {
            "status": "insufficient_inliers",
            "tracked_points": int(tracked_points),
            "inliers": inlier_count,
            "inlier_ratio": round(float(inlier_ratio), 4),
        }

    projected = cv2.perspectiveTransform(
        previous_good[inliers].reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)
    reprojection_error = float(
        np.median(np.linalg.norm(projected - current_good[inliers], axis=1))
    )
    hull = cv2.convexHull(previous_good[inliers].astype(np.float32))
    roi_pixels = pixel_quad(previous_roi, previous_image.shape).astype(np.float32)
    coverage = abs(float(cv2.contourArea(hull))) / max(
        abs(float(cv2.contourArea(roi_pixels))),
        1.0,
    )
    if reprojection_error > _MAX_REPROJECTION_ERROR or coverage < 0.03:
        return None, {
            "status": "unstable_homography",
            "tracked_points": int(tracked_points),
            "inliers": inlier_count,
            "inlier_ratio": round(float(inlier_ratio), 4),
            "reprojection_error": round(reprojection_error, 4),
            "coverage": round(coverage, 4),
        }

    matrix = matrix / matrix[2, 2] if abs(float(matrix[2, 2])) > 1e-9 else matrix
    return matrix, {
        "status": "aligned",
        "tracked_points": int(tracked_points),
        "inliers": inlier_count,
        "inlier_ratio": round(float(inlier_ratio), 4),
        "reprojection_error": round(reprojection_error, 4),
        "coverage": round(coverage, 4),
    }


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

    Features are restricted to the previous book ROI, so a static desk cannot
    dominate the transform when the book itself moves. The transform is accepted
    only when optical flow is reliable and both step/cumulative movement remain
    conservative.
    """
    previous = validate_roi(previous_roi)
    reference = validate_roi(reference_roi)
    max_step = float(max_step)
    max_total = float(max_total)
    if not math.isfinite(max_step) or not 0 < max_step <= 0.25:
        raise ValueError("max_step must be within 0..0.25")
    if not math.isfinite(max_total) or not max_step <= max_total <= 0.4:
        raise ValueError("max_total must be >= max_step and <= 0.4")

    matrix, alignment = _book_homography(
        previous_image,
        current_image,
        previous,
    )
    previous_total = float(np.max(np.linalg.norm(previous - reference, axis=1)))
    if matrix is None:
        return {
            "tracked": False,
            "status": alignment.get("status", "unavailable"),
            "roi": previous.tolist(),
            "step_shift": 0.0,
            "total_shift": previous_total,
            "alignment": alignment,
        }

    try:
        tracked = transform_normalized_quad(
            previous,
            matrix,
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
            "total_shift": previous_total,
            "alignment": alignment,
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
        "alignment": alignment,
    }
