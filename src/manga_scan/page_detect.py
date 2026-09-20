"""Conservative, opt-in contour refinement near the user's ROI."""

import cv2
import numpy as np

from .perspective import pixel_quad, validate_roi


def refine_quad(image, roi, max_shift=0.025):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ref = pixel_quad(roi, image.shape)
    h, w = image.shape[:2]
    best, best_distance = None, float("inf")
    for contour in contours:
        if cv2.contourArea(contour) < cv2.contourArea(ref.astype(np.float32)) * 0.85:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        # Match each detected corner to the reference, rejecting ambiguous assignments.
        indices = np.argmin(np.linalg.norm(ref[:, None, :] - pts[None, :, :], axis=2), axis=1)
        if len(set(indices)) != 4:
            continue
        q = pts[indices] / [w - 1, h - 1]
        try:
            validate_roi(q)
        except ValueError:
            continue
        distances = np.linalg.norm(q - roi, axis=1)
        # Never expand beyond the user's boundary: don't add desk pixels.
        inside = all(
            cv2.pointPolygonTest(ref.astype(np.float32), tuple(map(float, p)), False) >= 0
            for p in pts
        )
        if inside and distances.max() <= max_shift and distances.mean() < best_distance:
            best, best_distance = q, float(distances.mean())
    return (best.tolist(), True) if best is not None else (np.asarray(roi).tolist(), False)
