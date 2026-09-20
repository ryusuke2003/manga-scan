import math

import cv2
import numpy as np

from .perspective import geometry_penalties, warp_roi

_LOCAL_SHARPNESS_GRID = 3


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def local_sharpness_metrics(image, grid=_LOCAL_SHARPNESS_GRID):
    """Measure focus across the page instead of letting one sharp patch dominate.

    A manga page can contain a very sharp center and visibly blurred corners.
    Global Laplacian variance may still rate that frame highly, so calculate the
    same metric in a small grid and retain robust low-end statistics.
    """
    if not isinstance(grid, int) or grid < 1:
        raise ValueError("sharpness grid must be a positive integer")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Run Laplacian before splitting so artificial tile boundaries do not add
    # edge energy. Each tile then measures the variance of the real page edges.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    rows = np.array_split(laplacian, min(grid, gray.shape[0]), axis=0)
    values = []
    for row in rows:
        for tile in np.array_split(row, min(grid, gray.shape[1]), axis=1):
            if tile.size == 0:
                continue
            values.append(float(tile.var()))

    if not values:
        values = [float(laplacian.var())]

    values_array = np.asarray(values, dtype=np.float64)
    p10 = float(np.percentile(values_array, 10))
    median = float(np.median(values_array))
    worst = float(np.min(values_array))

    # Apply the log transform before combining values, matching the scale used
    # by the existing composite score while emphasizing weak regions.
    uniformity = (
        0.50 * math.log1p(max(0.0, p10))
        + 0.30 * math.log1p(max(0.0, median))
        + 0.20 * math.log1p(max(0.0, worst))
    ) / 8.0
    return {
        "sharpness_tiles": [round(float(value), 3) for value in values],
        "sharpness_median": median,
        "sharpness_p10": p10,
        "sharpness_worst": worst,
        "sharpness_uniformity": float(uniformity),
    }


def glare_fraction(image):
    """Estimate clipped specular highlights without treating plain white paper as glare."""
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        low_chroma = hsv[:, :, 1] <= 35
    else:
        value = image.astype(np.float32)
        low_chroma = np.ones(image.shape[:2], dtype=bool)

    # Glare should be a local bright anomaly. Uniform white paper therefore
    # contributes almost nothing even when it is close to 255.
    sigma = max(2.0, min(image.shape[:2]) * 0.025)
    local = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = (value >= 248) & low_chroma & ((value - local) >= 10)
    if not np.any(mask):
        return 0.0

    # Reject isolated compression noise while retaining small reflection spots.
    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    return float(np.mean(cleaned > 0))


def composite_score(metrics, config):
    # Keep the original absolute score stable. Candidate scoring v2 adds local
    # focus/glare through within-spread relative ranking in selection.py.
    return float(
        config.sharpness_weight * math.log1p(metrics["sharpness"]) / 8
        - config.motion_weight * min(1, metrics["motion"] / config.turn_threshold)
        - config.hand_overlap_weight * (metrics["hand_overlap"] or 0)
        - config.distortion_weight * metrics["distortion"]
        - config.flatness_weight * metrics["flatness_proxy"]
        - config.clipping_weight * metrics["clipping"]
        - config.exposure_weight * metrics["exposure"]
    )


def score_frame(image, roi, motion, hand_overlap, config):
    cropped = warp_roi(image, roi)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    distortion, flatness = geometry_penalties(roi, image.shape)
    # Saturated pixels are an imperfect proxy: black ink / white paper are legitimate.
    clipping = float(np.mean((gray <= 2) | (gray >= 253)))
    exposure = float(max(0, 55 - gray.mean()) / 55)
    metrics = {
        "sharpness": sharpness(cropped),
        **local_sharpness_metrics(cropped),
        "motion": float(motion),
        "hand_overlap": hand_overlap,
        "distortion": distortion,
        "flatness_proxy": flatness,
        "clipping": clipping,
        "exposure": exposure,
        "glare": glare_fraction(cropped),
    }
    metrics["score"] = composite_score(metrics, config)
    return metrics


def suspect_reasons(metrics, config, quad_ok=True):
    reasons = []
    if metrics["sharpness"] < config.suspect_sharpness:
        reasons.append("low_sharpness")
    if metrics["hand_overlap"] is None:
        reasons.append("hand_detection_disabled")
    elif metrics["hand_overlap"] > config.suspect_hand_overlap:
        reasons.append("hand_overlap")
    if metrics["motion"] > config.motion_threshold:
        reasons.append("high_motion")
    if metrics["distortion"] > 0.25 or not quad_ok:
        reasons.append("page_quad_uncertain")
    if metrics["exposure"] > 0.3:
        reasons.append("underexposed")
    return reasons
