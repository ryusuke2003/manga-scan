"""Automatic reference-spread detection from a full setup frame."""

from __future__ import annotations

import cv2
import numpy as np

from .cover_detect import detect_cover_quad
from .page_contour import detect_page_quads, spread_quad_from_page_quads


def detect_reference_spread(image, min_confidence=0.55):
    """Detect both pages from the full frame and return one outer spread ROI.

    The first stage finds a broad landscape book outline without relying on a
    user ROI. The second stage verifies that the outline really contains two
    page-like quads. If either stage is uncertain, callers should fall back to
    manual four-point selection.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")

    outline = detect_cover_quad(
        image,
        min_confidence=max(0.58, float(min_confidence)),
        area_range=(0.12, 0.96),
        aspect_range=(1.05, 3.2),
        target_aspect=1.75,
    )
    if not outline["detected"]:
        return {
            "detected": False,
            "confidence": round(float(outline["confidence"]), 4),
            "roi": None,
            "outline": outline,
            "pages": None,
            "stage": "outline",
        }

    pages = detect_page_quads(
        image,
        outline["roi"],
        spine_ratio=0.5,
        min_confidence=float(min_confidence),
    )
    confidence = min(float(outline["confidence"]), float(pages["confidence"]))
    if not pages["detected"]:
        return {
            "detected": False,
            "confidence": round(confidence, 4),
            "roi": None,
            "outline": outline,
            "pages": pages,
            "stage": "pages",
        }

    try:
        roi = spread_quad_from_page_quads(pages)
    except ValueError:
        return {
            "detected": False,
            "confidence": round(confidence, 4),
            "roi": None,
            "outline": outline,
            "pages": pages,
            "stage": "combined",
        }

    return {
        "detected": True,
        "confidence": round(confidence, 4),
        "roi": roi,
        "outline": outline,
        "pages": pages,
        "stage": "complete",
    }


def draw_reference_spread(image, detection):
    """Draw the detected outer spread and the two verified page quads."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    pages = detection.get("pages")
    if pages:
        h, w = canvas.shape[:2]
        scale = np.asarray([max(w - 1, 1), max(h - 1, 1)], dtype=np.float32)
        for side, color in (("left", (60, 180, 75)), ("right", (0, 140, 255))):
            quad = np.asarray(pages[side]["quad"], dtype=np.float32) * scale
            cv2.polylines(
                canvas,
                [np.rint(quad).astype(np.int32)],
                True,
                color,
                2,
                cv2.LINE_AA,
            )
    roi = detection.get("roi")
    if roi:
        h, w = canvas.shape[:2]
        quad = np.asarray(roi, dtype=np.float32) * [max(w - 1, 1), max(h - 1, 1)]
        cv2.polylines(
            canvas,
            [np.rint(quad).astype(np.int32)],
            True,
            (255, 220, 80),
            3,
            cv2.LINE_AA,
        )
    return canvas
