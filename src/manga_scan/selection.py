import cv2
import numpy as np

from .score import composite_score, sharpness
from .split import rotate_image, split_spread


def page_quality_metrics(
    image,
    motion,
    hand_overlap,
    config,
    distortion=0.0,
    flatness_proxy=0.0,
    glare_overlap=0.0,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    metrics = {
        "sharpness": sharpness(image),
        "motion": float(motion),
        "hand_overlap": hand_overlap,
        "glare_overlap": float(glare_overlap),
        "distortion": float(distortion),
        "flatness_proxy": float(flatness_proxy),
        "clipping": float(np.mean((gray <= 2) | (gray >= 253))),
        "exposure": float(max(0, 55 - gray.mean()) / 55),
    }
    metrics["score"] = composite_score(metrics, config)
    return metrics


def score_candidate_pages(
    rectified,
    hand_mask,
    motion,
    config,
    geometry_metrics,
    hand_enabled,
    glare_mask=None,
):
    rectified = rotate_image(rectified, config.rotation)
    hand_mask = rotate_image(hand_mask, config.rotation)
    if glare_mask is not None:
        glare_mask = rotate_image(glare_mask, config.rotation)
    pages, spine = split_spread(
        rectified,
        config.spine_ratio,
        config.split_mode,
        config.gutter_fraction,
    )
    gutter = round(rectified.shape[1] * config.gutter_fraction / 2)
    left_end = max(1, spine - gutter)
    right_start = min(rectified.shape[1] - 1, spine + gutter)

    if hand_enabled:
        mask_pages = {
            "left": hand_mask[:, :left_end],
            "right": hand_mask[:, right_start:],
        }
        overlaps = {
            side: float(np.mean(mask_pages[side] > 127))
            for side in ("left", "right")
        }
    else:
        overlaps = {"left": None, "right": None}

    if glare_mask is not None:
        glare_pages = {
            "left": glare_mask[:, :left_end],
            "right": glare_mask[:, right_start:],
        }
        glare_overlaps = {
            side: float(np.mean(glare_pages[side] > 127))
            for side in ("left", "right")
        }
    else:
        glare_overlaps = {"left": 0.0, "right": 0.0}

    page_metrics = {}
    for side in ("left", "right"):
        page_metrics[side] = page_quality_metrics(
            pages[side],
            motion,
            overlaps[side],
            config,
            distortion=geometry_metrics["distortion"],
            flatness_proxy=geometry_metrics["flatness_proxy"],
            glare_overlap=glare_overlaps[side],
        )
    return page_metrics, spine


def choose_candidate_selection(records, mode):
    if not records:
        raise ValueError("No candidates to select")
    spread = max(records, key=lambda candidate: candidate["metrics"]["score"])
    if mode == "spread":
        selected_pages = {"left": spread["id"], "right": spread["id"]}
    elif mode == "per_page":
        selected_pages = {
            side: max(records, key=lambda candidate: candidate["page_metrics"][side]["score"])[
                "id"
            ]
            for side in ("left", "right")
        }
    else:
        raise ValueError(f"Unknown candidate selection mode: {mode}")
    return spread["id"], selected_pages
