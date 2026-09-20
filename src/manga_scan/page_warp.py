"""Independent perspective correction for detected left/right manga pages."""

from __future__ import annotations

import cv2
import numpy as np


def validate_page_quad(points):
    """Validate normalized page corners in TL, TR, BR, BL order."""

    quad = np.asarray(points, dtype=np.float32)
    if (
        quad.shape != (4, 2)
        or not np.isfinite(quad).all()
        or (quad < 0).any()
        or (quad > 1).any()
    ):
        raise ValueError("page quad must contain four finite [x,y] pairs in 0..1")

    edges = np.roll(quad, -1, axis=0) - quad
    cross = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if (cross <= 0).any() or cv2.contourArea(quad) <= 1e-6:
        raise ValueError("page quad must be convex in TL, TR, BR, BL order")
    if (
        quad[:2, 1].mean() >= quad[2:, 1].mean()
        or quad[[0, 3], 0].mean() >= quad[[1, 2], 0].mean()
    ):
        raise ValueError("page quad corner order must be TL, TR, BR, BL")
    return quad


def pixel_page_quad(points, shape):
    """Convert a normalized page quad into image pixel coordinates."""

    quad = validate_page_quad(points)
    h, w = shape[:2]
    if h < 2 or w < 2:
        raise ValueError("image must be at least 2x2")
    return quad * np.asarray([w - 1, h - 1], dtype=np.float32)


def natural_page_size(points, shape):
    """Estimate a rectified page size from the longest opposing source edges."""

    quad = pixel_page_quad(points, shape)
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    return max(2, round(float(width)) + 1), max(2, round(float(height)) + 1)


def _validate_output_size(output_size):
    if output_size is None:
        return None
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or isinstance(output_size[0], bool)
        or isinstance(output_size[1], bool)
    ):
        raise ValueError("output_size must be (width, height)")
    width, height = output_size
    if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2:
        raise ValueError("output_size must contain integer dimensions >= 2")
    return width, height


def warp_page(image, quad, *, output_size=None, interpolation=cv2.INTER_CUBIC):
    """Rectify one page quad without first warping the whole spread."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy grayscale or multi-channel image")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image must be at least 2x2")

    src = pixel_page_quad(quad, image.shape).astype(np.float32)
    size = _validate_output_size(output_size) or natural_page_size(quad, image.shape)
    width, height = size
    dst = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=interpolation,
    )


def warp_detected_pages(image, detection, *, output_sizes=None, interpolation=cv2.INTER_CUBIC):
    """Warp left/right quads returned by detect_page_quads.

    The result retains explicit left and right keys and therefore does not
    impose reading order. Existing RTL/LTR ordering can remain a pipeline
    concern while each physical page is geometrically corrected independently.
    """

    if not isinstance(detection, dict):
        raise ValueError("detection must be a mapping with left/right page data")
    if output_sizes is not None and not isinstance(output_sizes, dict):
        raise ValueError("output_sizes must be a mapping keyed by left/right")

    pages = {}
    for side in ("left", "right"):
        data = detection.get(side)
        if not isinstance(data, dict) or "quad" not in data:
            raise ValueError(f"detection must contain {side}.quad")
        size = None if output_sizes is None else output_sizes.get(side)
        pages[side] = warp_page(
            image,
            data["quad"],
            output_size=size,
            interpolation=interpolation,
        )
    return pages
