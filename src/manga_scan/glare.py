"""Conservative specular-glare detection for photographed manga pages."""

from __future__ import annotations

import cv2
import numpy as np

from .perspective import pixel_quad, validate_roi


def _gray_and_saturation(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy image")
    if image.ndim == 2:
        return image, np.zeros_like(image)
    if image.ndim == 3 and image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray, hsv[:, :, 1]
    raise ValueError("image must be HxW grayscale or BGR")


def _roi_mask(shape, roi):
    height, width = shape[:2]
    if roi is None:
        return np.full((height, width), 255, np.uint8)
    validate_roi(roi)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(pixel_quad(roi, shape)).astype(np.int32), 255)
    return mask


def detect_glare_mask(image, roi=None):
    """Return a conservative 0/255 mask for likely specular glare.

    The detector deliberately requires several signals at once:
    - very high luminance,
    - low saturation,
    - positive luminance contrast against a local background,
    - a spatially meaningful connected region.

    This avoids treating ordinary white paper as glare. Regions are then
    checked against a surrounding ring; small bright halos around manga lines
    are rejected unless the component is sufficiently broad.
    """

    gray, saturation = _gray_and_saturation(image)
    if min(gray.shape[:2]) < 8:
        return np.zeros(gray.shape[:2], np.uint8)

    roi_mask = _roi_mask(image.shape, roi)
    roi_values = gray[roi_mask > 0]
    if roi_values.size < 16:
        return np.zeros(gray.shape[:2], np.uint8)

    # Keep an absolute white floor so ordinary off-white paper is not enough,
    # while still adapting to HDR/brightly exposed footage.
    bright_threshold = max(242.0, float(np.percentile(roi_values, 96.0)))
    sigma = max(3.0, min(gray.shape[:2]) * 0.015)
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_delta = gray.astype(np.float32) - local_background.astype(np.float32)

    candidate = (
        (gray.astype(np.float32) >= bright_threshold)
        & (saturation <= 65)
        & (local_delta >= 8.0)
        & (roi_mask > 0)
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    roi_area = int(np.count_nonzero(roi_mask))
    min_area = max(12, round(roi_area * 0.00005))
    max_area = max(min_area, round(roi_area * 0.25))
    ring_radius = max(4, round(min(gray.shape[:2]) * 0.01))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ring_radius * 2 + 1, ring_radius * 2 + 1),
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)

    accepted = np.zeros_like(candidate)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        component = labels == label
        component_u8 = component.astype(np.uint8)
        distance = cv2.distanceTransform(component_u8, cv2.DIST_L2, 5)
        # Line-edge halos are usually only one pixel or two wide.
        if float(distance.max()) < 1.4 and area < min_area * 6:
            continue

        expanded = cv2.dilate(component_u8, ring_kernel, iterations=1) > 0
        ring = expanded & ~component & (roi_mask > 0)
        if np.count_nonzero(ring) < max(24, round(area * 0.15)):
            continue

        inside_brightness = float(np.median(gray[component]))
        ring_brightness = float(np.median(gray[ring]))
        contrast = inside_brightness - ring_brightness
        delta_median = float(np.median(local_delta[component]))
        ring_edge_density = float(np.mean(edges[ring] > 0))
        inside_edge_density = float(np.mean(edges[component] > 0))

        # A real reflection can erase lines/halftone, so the surrounding ring
        # often contains more edge energy than the blown-out interior. Strong
        # luminance contrast is also accepted for glare over otherwise blank
        # paper, where no surrounding line may exist.
        detail_loss = ring_edge_density >= inside_edge_density + 0.01
        if contrast < 8.0 or delta_median < 8.0:
            continue
        if contrast < 14.0 and not detail_loss:
            continue
        accepted[component] = 1

    if not np.any(accepted):
        return np.zeros(gray.shape[:2], np.uint8)

    margin = max(1, round(min(gray.shape[:2]) * 0.002))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (margin * 2 + 1, margin * 2 + 1),
    )
    accepted = cv2.dilate(accepted, kernel, iterations=1)
    accepted &= (roi_mask > 0).astype(np.uint8)
    return accepted.astype(np.uint8) * 255


def glare_overlap_fraction(mask, roi=None):
    """Return the fraction of the ROI covered by a glare mask."""

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("glare mask must be a grayscale array")
    roi_mask = _roi_mask(mask.shape, roi)
    total = int(np.count_nonzero(roi_mask))
    if total == 0:
        return 0.0
    return float(np.count_nonzero((mask > 127) & (roi_mask > 0)) / total)
