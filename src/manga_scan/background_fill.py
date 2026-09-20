"""Conservative cleanup of desk/background pixels outside detected pages."""

from __future__ import annotations

import cv2
import numpy as np


def detected_spread_mask(shape, detection):
    """Return a mask that protects both detected pages and the photographed gutter."""
    if not isinstance(detection, dict) or not detection.get("detected"):
        raise ValueError("both page quads must be detected")
    height, width = shape[:2]
    if height < 2 or width < 2:
        raise ValueError("image must be at least 2x2")

    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    quads = {}
    for side in ("left", "right"):
        quad = np.asarray(detection.get(side, {}).get("quad"), dtype=np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError("page detection must contain finite left/right quads")
        quads[side] = np.rint(quad * scale).astype(np.int32)

    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, quads["left"], 255)
    cv2.fillConvexPoly(mask, quads["right"], 255)

    # Keep the real photographed gutter instead of whitening the center seam.
    bridge = np.asarray(
        [
            quads["left"][1],
            quads["right"][0],
            quads["right"][3],
            quads["left"][2],
        ],
        dtype=np.int32,
    )
    if abs(cv2.contourArea(bridge.astype(np.float32))) > 1:
        cv2.fillConvexPoly(mask, bridge, 255)
    return mask


def _paper_fill_color(image, protected, target):
    radius = max(3, round(min(image.shape[:2]) * 0.015))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    inner_ring = protected & ~(cv2.erode(protected.astype(np.uint8), kernel) > 0)
    if not np.any(inner_ring):
        return target if image.ndim == 2 else np.asarray([target] * 3, np.uint8)

    if image.ndim == 2:
        values = image[inner_ring]
        bright = values[values >= 150]
        if bright.size >= 32:
            return np.uint8(round(float(np.median(bright))))
        return np.uint8(target)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    paper_like = inner_ring & (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 150)
    if np.count_nonzero(paper_like) >= 32:
        return np.median(image[paper_like], axis=0).astype(np.uint8)
    return np.asarray([target] * 3, np.uint8)


def fill_page_background(
    image,
    page_mask,
    *,
    mode="paper",
    paper_target=245,
    feather_px=4,
):
    """Fill only pixels outside a trusted page mask.

    preserve: keep the original background.
    paper: use a locally estimated paper color, falling back to paper_target.
    white: use pure white.

    A small protected margin and outward feather keep page-edge pixels untouched.
    """
    if mode not in ("preserve", "paper", "white"):
        raise ValueError("page background fill must be preserve, paper, or white")
    if not 0 <= int(paper_target) <= 255:
        raise ValueError("paper_target must be 0..255")
    if feather_px < 0:
        raise ValueError("feather_px must be nonnegative")

    mask = np.asarray(page_mask)
    if mask.shape != image.shape[:2]:
        raise ValueError("page mask must match image dimensions")
    protected = mask > 127
    info = {
        "mode": mode,
        "applied": False,
        "filled_fraction": 0.0,
        "fill_color": None,
    }
    if mode == "preserve" or not np.any(protected):
        return image.copy(), info

    margin = max(1, round(min(image.shape[:2]) * 0.003))
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (margin * 2 + 1, margin * 2 + 1),
    )
    protected_margin = cv2.dilate(protected.astype(np.uint8), margin_kernel) > 0
    background = ~protected_margin
    total_background = int(np.count_nonzero(background))
    if total_background == 0:
        return image.copy(), info

    if mode == "white":
        fill_color = np.uint8(255) if image.ndim == 2 else np.asarray([255] * 3, np.uint8)
    else:
        fill_color = _paper_fill_color(image, protected, int(paper_target))

    fill = np.empty_like(image)
    fill[...] = fill_color

    if feather_px == 0:
        alpha = background.astype(np.float32)
    else:
        distance = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(distance / float(max(1, feather_px)), 0.0, 1.0)
        alpha[~background] = 0.0

    if image.ndim == 3:
        alpha = alpha[:, :, None]
    mixed = image.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha
    result = np.clip(mixed, 0, 255).astype(np.uint8)

    changed = np.any(result != image, axis=2) if image.ndim == 3 else result != image
    info.update(
        applied=bool(np.any(changed)),
        filled_fraction=round(float(np.count_nonzero(changed)) / max(total_background, 1), 4),
        fill_color=(
            int(fill_color)
            if np.ndim(fill_color) == 0
            else [int(value) for value in np.asarray(fill_color).tolist()]
        ),
    )
    return result, info
