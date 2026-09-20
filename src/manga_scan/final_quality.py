"""Conservative quality checks for fully rendered output pages."""

from __future__ import annotations

import cv2
import numpy as np

from .dedupe import compare, gray_thumb
from .score import glare_fraction

FINAL_QUALITY_REASONS = {
    "final_edge_crop_suspected",
    "final_dewarp_line_regression",
    "final_unresolved_finger",
    "final_finger_repair_residual",
    "final_background_fill_large",
    "final_glare_residual",
    "final_duplicate_suspected",
    "final_near_blank_white",
    "final_near_blank_black",
}


def _gray(image):
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("final quality checks expect grayscale or BGR images")


def _edge_line_metrics(image):
    """Find a suspicious near-continuous strong line at the final crop boundary."""
    gray = _gray(image)
    height, width = gray.shape[:2]
    if min(height, width) < 24:
        return {
            "edge_line_score": 0.0,
            "edge_line_fraction": 0.0,
            "edge_line_side": None,
        }

    band = max(2, round(min(height, width) * 0.012))
    reference_offset = band * 3
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    edges = {
        "left": np.max(gx[:, :band], axis=1),
        "right": np.max(gx[:, width - band :], axis=1),
        "top": np.max(gy[:band, :], axis=0),
        "bottom": np.max(gy[height - band :, :], axis=0),
    }
    references = {
        "left": np.max(gx[:, reference_offset : reference_offset + band], axis=1),
        "right": np.max(
            gx[:, max(0, width - reference_offset - band) : width - reference_offset],
            axis=1,
        ),
        "top": np.max(gy[reference_offset : reference_offset + band, :], axis=0),
        "bottom": np.max(
            gy[max(0, height - reference_offset - band) : height - reference_offset, :],
            axis=0,
        ),
    }

    best = (0.0, 0.0, None)
    for side in ("left", "right", "top", "bottom"):
        edge = edges[side]
        reference = references[side]
        if edge.size == 0 or reference.size == 0:
            continue
        threshold = max(45.0, float(np.percentile(reference, 90)) * 1.7)
        edge_fraction = float(np.mean(edge >= threshold))
        reference_fraction = float(np.mean(reference >= threshold))
        score = max(0.0, edge_fraction - reference_fraction)
        if score > best[0]:
            best = (score, edge_fraction, side)

    return {
        "edge_line_score": round(best[0], 4),
        "edge_line_fraction": round(best[1], 4),
        "edge_line_side": best[2],
    }


def _orthogonal_line_score(image):
    """Estimate how much long straight panel/text geometry survives in an image."""
    gray = _gray(image)
    height, width = gray.shape[:2]
    if min(height, width) < 40:
        return None

    scale = min(1.0, 640 / max(height, width))
    if scale < 1:
        gray = cv2.resize(
            gray,
            (max(20, round(width * scale)), max(20, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 55, 150)
    min_side = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(24, round(min_side * 0.08)),
        minLineLength=max(20, round(min_side * 0.18)),
        maxLineGap=max(3, round(min_side * 0.02)),
    )
    if lines is None:
        return 0.0

    total = 0.0
    count = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
        distance = min(angle, abs(90.0 - angle), abs(180.0 - angle))
        if distance <= 7.0:
            total += length
            count += 1
    if count < 2:
        return 0.0
    return float(total / max(1.0, height + width))


def _repair_residual(finger_repair):
    maximum = None
    for component in (finger_repair or {}).get("components", []):
        for donor in component.get("donors", []):
            value = donor.get("context_residual")
            if isinstance(value, (int, float)) and np.isfinite(value):
                maximum = float(value) if maximum is None else max(maximum, float(value))
    return maximum


def final_quality_checks(
    image,
    *,
    before_enhance=None,
    before_dewarp=None,
    dewarp=None,
    finger_repair=None,
    background_fill_fraction=None,
    white_normalization=False,
):
    """Inspect a fully rendered page and return review-only warnings.

    These checks never modify the page. They intentionally use conservative
    thresholds because a manga page can legitimately contain black borders,
    blank pages, white paper and long panel lines.
    """
    gray = _gray(image)
    reasons = []
    metrics = {}

    edge = _edge_line_metrics(image)
    metrics.update(edge)
    if edge["edge_line_fraction"] >= 0.72 and edge["edge_line_score"] >= 0.48:
        reasons.append("final_edge_crop_suspected")

    glare = float(glare_fraction(image))
    metrics["glare_fraction"] = round(glare, 6)
    if glare >= 0.003:
        reasons.append("final_glare_residual")

    mean = float(np.mean(gray))
    std = float(np.std(gray))
    white_fraction = float(np.mean(gray >= 250))
    black_fraction = float(np.mean(gray <= 5))
    metrics.update(
        mean_luma=round(mean, 3),
        luma_std=round(std, 3),
        white_fraction=round(white_fraction, 6),
        black_fraction=round(black_fraction, 6),
    )
    if white_fraction >= 0.985 and std <= 12.0:
        reasons.append("final_near_blank_white")
    if black_fraction >= 0.97 and std <= 12.0:
        reasons.append("final_near_blank_black")

    if before_enhance is not None:
        source_gray = _gray(before_enhance)
        if source_gray.shape != gray.shape:
            source_gray = cv2.resize(
                source_gray,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        delta = gray.astype(np.int16) - source_gray.astype(np.int16)
        whitened = (gray >= 245) & (source_gray < 238) & (delta >= 20)
        whitened_fraction = float(np.mean(whitened))
        metrics["whitened_fraction"] = round(whitened_fraction, 6)
        if white_normalization and whitened_fraction >= 0.30:
            reasons.append("final_background_fill_large")

    if background_fill_fraction is not None:
        background_fill_fraction = float(background_fill_fraction)
        metrics["background_fill_fraction"] = round(background_fill_fraction, 6)
        if background_fill_fraction >= 0.25:
            reasons.append("final_background_fill_large")

    repair = finger_repair or {}
    kinds = repair.get("occlusion_kinds")
    has_finger = not kinds or "finger" in kinds
    unresolved = (
        has_finger
        and (
            bool(repair.get("unresolved_mask"))
            or repair.get("status") == "incomplete"
        )
    )
    metrics["unresolved_finger"] = bool(unresolved)
    if unresolved:
        reasons.append("final_unresolved_finger")

    residual = _repair_residual(repair)
    metrics["finger_repair_max_residual"] = (
        None if residual is None else round(float(residual), 4)
    )
    if has_finger and residual is not None and residual >= 0.14:
        reasons.append("final_finger_repair_residual")

    if before_dewarp is not None and (dewarp or {}).get("applied"):
        before_score = _orthogonal_line_score(before_dewarp)
        after_score = _orthogonal_line_score(image)
        metrics["dewarp_line_score_before"] = (
            None if before_score is None else round(float(before_score), 4)
        )
        metrics["dewarp_line_score_after"] = (
            None if after_score is None else round(float(after_score), 4)
        )
        if (
            before_score is not None
            and after_score is not None
            and before_score >= 0.8
            and after_score <= before_score * 0.62
            and before_score - after_score >= 0.35
        ):
            reasons.append("final_dewarp_line_regression")

    return {
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": metrics,
    }


def adjacent_quality_check(a, b, config):
    """Check two informative final pages for suspicious near-duplication."""
    result = compare(a, b, config)
    informative = min(
        float(gray_thumb(a).std()),
        float(gray_thumb(b).std()),
    ) >= 8.0
    hash_limit = max(12, int(config.duplicate_hash_distance) * 3)
    suspect = (
        informative
        and result["ssim"] >= config.duplicate_suspect_ssim
        and result["hash_distance"] <= hash_limit
    )
    return {
        "suspect": bool(suspect),
        "duplicate": bool(result["duplicate"]),
        "ssim": round(float(result["ssim"]), 6),
        "hash_distance": int(result["hash_distance"]),
        "informative": bool(informative),
    }
