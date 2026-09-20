import math

import cv2
import numpy as np

from .perspective import geometry_penalties, warp_roi


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def composite_score(metrics, config):
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
        "motion": float(motion),
        "hand_overlap": hand_overlap,
        "distortion": distortion,
        "flatness_proxy": flatness,
        "clipping": clipping,
        "exposure": exposure,
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
