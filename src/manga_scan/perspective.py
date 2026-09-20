import cv2
import numpy as np


def validate_roi(points):
    """Normalized coordinates in TL, TR, BR, BL order in displayed orientation."""
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("ROI must contain four [x,y] pairs in 0..1")
    edges = np.roll(q, -1, axis=0) - q
    cross = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if (cross <= 0).any() or cv2.contourArea(q) < 0.005:
        raise ValueError("ROI must be convex: top-left, top-right, bottom-right, bottom-left")
    if q[:2, 1].mean() >= q[2:, 1].mean() or q[[0, 3], 0].mean() >= q[[1, 2], 0].mean():
        raise ValueError("ROI corner order must be TL, TR, BR, BL")
    return q


def pixel_quad(points, shape):
    q = validate_roi(points)
    h, w = shape[:2]
    return q * [w - 1, h - 1]


def warp_roi(image, points):
    q = pixel_quad(points, image.shape).astype(np.float32)
    width = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    height = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    w, h = max(2, round(float(width)) + 1), max(2, round(float(height)) + 1)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    transform = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(image, transform, (w, h), flags=cv2.INTER_CUBIC)


def geometry_penalties(points, shape):
    q = pixel_quad(points, shape)
    e = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(e, axis=1)
    distortion = float(
        np.mean(
            np.abs(
                np.sum(e * np.roll(e, -1, axis=0), axis=1) / (lengths * np.roll(lengths, -1) + 1e-8)
            )
        )
    )
    # Opposing edge imbalance is only a geometry proxy, not physical page curvature.
    flatness = float(
        (
            abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2])
            + abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3])
        )
        / 2
    )
    return distortion, flatness
