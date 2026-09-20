import math

import cv2
import numpy as np

from .score import composite_score, glare_fraction, local_sharpness_metrics, sharpness
from .split import rotate_image, split_spread


def page_quality_metrics(
    image,
    motion,
    hand_overlap,
    config,
    distortion=0.0,
    flatness_proxy=0.0,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    metrics = {
        "sharpness": sharpness(image),
        **local_sharpness_metrics(image),
        "motion": float(motion),
        "hand_overlap": hand_overlap,
        "distortion": float(distortion),
        "flatness_proxy": float(flatness_proxy),
        "clipping": float(np.mean((gray <= 2) | (gray >= 253))),
        "exposure": float(max(0, 55 - gray.mean()) / 55),
        "glare": glare_fraction(image),
    }
    metrics["score"] = composite_score(metrics, config)
    return metrics


def score_candidate_pages(rectified, hand_mask, motion, config, geometry_metrics, hand_enabled):
    rectified = rotate_image(rectified, config.rotation)
    hand_mask = rotate_image(hand_mask, config.rotation)
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

    page_metrics = {}
    for side in ("left", "right"):
        page_metrics[side] = page_quality_metrics(
            pages[side],
            motion,
            overlaps[side],
            config,
            distortion=geometry_metrics["distortion"],
            flatness_proxy=geometry_metrics["flatness_proxy"],
        )
    return page_metrics, spine


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sharpness_value(metrics):
    value = metrics.get("sharpness_uniformity")
    if _finite(value):
        return float(value)
    sharp = metrics.get("sharpness")
    if _finite(sharp):
        return math.log1p(max(0.0, float(sharp))) / 8.0
    return None


def _percentile_scores(values, *, higher_is_better):
    """Return 0..1 relative ranks with ties sharing the same midpoint rank."""
    numeric = [float(value) if _finite(value) else None for value in values]
    present = [value for value in numeric if value is not None]
    if len(present) <= 1 or max(present) - min(present) <= 1e-12:
        return [0.5 if value is not None else None for value in numeric]

    ordered = sorted(present)
    result = []
    denominator = len(present) - 1
    for value in numeric:
        if value is None:
            result.append(None)
            continue
        lower = sum(other < value for other in ordered)
        equal = sum(other == value for other in ordered)
        rank = (lower + (equal - 1) / 2) / denominator
        result.append(float(rank if higher_is_better else 1.0 - rank))
    return result


def _relative_group(metrics_group):
    if not metrics_group:
        return

    sharpness_rank = _percentile_scores(
        [_sharpness_value(metrics) for metrics in metrics_group],
        higher_is_better=True,
    )
    motion_rank = _percentile_scores(
        [metrics.get("motion") for metrics in metrics_group],
        higher_is_better=False,
    )
    hand_rank = _percentile_scores(
        [metrics.get("hand_overlap") for metrics in metrics_group],
        higher_is_better=False,
    )
    glare_rank = _percentile_scores(
        [metrics.get("glare", 0.0) for metrics in metrics_group],
        higher_is_better=False,
    )
    base_rank = _percentile_scores(
        [metrics.get("score") for metrics in metrics_group],
        higher_is_better=True,
    )

    for index, metrics in enumerate(metrics_group):
        # Disabled hand detection yields None for every candidate. Do not make
        # that lower everybody's score; treat unavailable dimensions neutrally.
        components = {
            "sharpness": sharpness_rank[index],
            "motion": motion_rank[index],
            "hand_overlap": hand_rank[index],
            "glare": glare_rank[index],
            "base_score": base_rank[index],
        }
        weighted = (
            (0.35, components["sharpness"]),
            (0.15, components["motion"]),
            (0.20, components["hand_overlap"]),
            (0.10, components["glare"]),
            (0.20, components["base_score"]),
        )
        numerator = 0.0
        denominator = 0.0
        for weight, value in weighted:
            if value is None:
                continue
            numerator += weight * value
            denominator += weight
        metrics["selection_score"] = float(
            numerator / denominator if denominator else metrics.get("score", 0.0)
        )
        metrics["relative_quality"] = {
            key: None if value is None else round(float(value), 4)
            for key, value in components.items()
        }


def apply_relative_candidate_scores(records):
    """Add within-spread relative scores to spread and per-page metrics."""
    if not records:
        return records
    _relative_group([record["metrics"] for record in records])
    for side in ("left", "right"):
        _relative_group([record["page_metrics"][side] for record in records])
    return records


def _selection_value(metrics):
    value = metrics.get("selection_score")
    if _finite(value):
        return float(value)
    value = metrics.get("score")
    return float(value) if _finite(value) else float("-inf")


def choose_candidate_selection(records, mode):
    if not records:
        raise ValueError("No candidates to select")

    # New records gain relative v2 scores here. Legacy/minimal records that do
    # not contain enough metrics still work because the old absolute score is
    # retained as a fallback selection value.
    complete = all(
        "metrics" in candidate
        and all(side in candidate.get("page_metrics", {}) for side in ("left", "right"))
        for candidate in records
    )
    if complete:
        apply_relative_candidate_scores(records)

    spread = max(records, key=lambda candidate: _selection_value(candidate["metrics"]))
    if mode == "spread":
        selected_pages = {"left": spread["id"], "right": spread["id"]}
    elif mode == "per_page":
        selected_pages = {
            side: max(
                records,
                key=lambda candidate: _selection_value(candidate["page_metrics"][side]),
            )["id"]
            for side in ("left", "right")
        }
    else:
        raise ValueError(f"Unknown candidate selection mode: {mode}")
    return spread["id"], selected_pages
