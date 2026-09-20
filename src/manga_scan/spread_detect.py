"""Automatic reference-spread detection from a full setup frame."""

from __future__ import annotations

import cv2
import numpy as np

from .cover_detect import detect_cover_quad
from .page_contour import detect_page_quads, spread_quad_from_page_quads


def _coarse_spread_priors():
    """Conservative full-frame priors for open books whose outer edge is obscured."""
    return [
        np.asarray([[0.04, 0.03], [0.86, 0.03], [0.86, 0.97], [0.04, 0.97]], np.float32),
        np.asarray([[0.09, 0.03], [0.91, 0.03], [0.91, 0.97], [0.09, 0.97]], np.float32),
        np.asarray([[0.14, 0.03], [0.96, 0.03], [0.96, 0.97], [0.14, 0.97]], np.float32),
        np.asarray([[0.15, 0.06], [0.85, 0.06], [0.85, 0.94], [0.15, 0.94]], np.float32),
    ]


def _detect_pages_from_priors(
    working,
    priors,
    min_confidence,
    *,
    confidence_scale=1.0,
    confidence_cap=None,
):
    best_failure = None
    successes = []
    for prior in priors:
        for spine_ratio in (0.46, 0.50, 0.54):
            try:
                pages = detect_page_quads(
                    working,
                    prior,
                    spine_ratio=spine_ratio,
                    min_confidence=float(min_confidence),
                )
            except ValueError:
                continue
            confidence = float(pages["confidence"]) * float(confidence_scale)
            if confidence_cap is not None:
                confidence = min(confidence, float(confidence_cap))
            if best_failure is None or confidence > best_failure[0]:
                best_failure = (confidence, pages)
            if not pages["detected"]:
                continue
            try:
                roi = spread_quad_from_page_quads(pages)
            except ValueError:
                continue
            successes.append((confidence, roi, pages))

    success = max(successes, key=lambda item: item[0]) if successes else None
    return success, best_failure


def detect_reference_spread(image, min_confidence=0.55):
    """Detect both pages from the full frame and return one outer spread ROI.

    Prefer a broad Hough-based book outline, then verify the left/right pages.
    If hands, page color, or curvature break that outer outline, retry from a
    small set of conservative centered priors and let the two page detectors
    establish the actual outer corners. Only a verified two-page result is
    accepted; otherwise callers still fall back to manual four-point selection.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")

    scale = min(1.0, 1000 / max(image.shape[:2]))
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    outline = detect_cover_quad(
        working,
        min_confidence=max(0.58, float(min_confidence)),
        area_range=(0.12, 0.96),
        aspect_range=(1.05, 3.2),
        target_aspect=1.75,
    )

    best_failure = None
    if outline["detected"]:
        outline_roi = np.asarray(outline["roi"], dtype=np.float32)
        center = outline_roi.mean(axis=0)
        outline_priors = [
            np.clip(center + (outline_roi - center) * expansion, 0, 1)
            for expansion in (1.0, 1.04, 1.08, 1.12)
        ]
        success, best_failure = _detect_pages_from_priors(
            working,
            outline_priors,
            min_confidence,
            confidence_cap=float(outline["confidence"]),
        )
        if success is not None:
            confidence, roi, pages = success
            return {
                "detected": True,
                "confidence": round(float(confidence), 4),
                "roi": roi,
                "outline": outline,
                "pages": pages,
                "stage": "complete",
                "source": "outline_pages",
            }

    coarse_success, coarse_failure = _detect_pages_from_priors(
        working,
        _coarse_spread_priors(),
        min_confidence,
        confidence_scale=0.94,
    )
    if coarse_success is not None:
        confidence, roi, pages = coarse_success
        return {
            "detected": True,
            "confidence": round(float(confidence), 4),
            "roi": roi,
            "outline": outline,
            "pages": pages,
            "stage": "complete",
            "source": "coarse_pages",
        }

    failures = [failure for failure in (best_failure, coarse_failure) if failure is not None]
    confidence, pages = max(failures, key=lambda item: item[0]) if failures else (0.0, None)
    return {
        "detected": False,
        "confidence": round(max(float(outline.get("confidence", 0.0)), float(confidence)), 4),
        "roi": None,
        "outline": outline,
        "pages": pages,
        "stage": "pages" if pages is not None else "outline",
        "source": None,
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
