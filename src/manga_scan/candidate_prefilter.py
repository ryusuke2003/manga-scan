"""Cheap candidate prefiltering before expensive contour/hand/glare analysis."""

from __future__ import annotations

import math

import cv2
import numpy as np

_PREFILTER_WIDTH = 128
_PREFILTER_HEIGHT = 96
_DUPLICATE_MEAN_ABS = 4.0
_DUPLICATE_MAX_BLOCK_ABS = 9.0
_DUPLICATE_MAX_SHIFT = 2.0


def _gray_thumb(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("candidate frames must be grayscale or BGR numpy arrays")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if min(gray.shape[:2]) < 2:
        raise ValueError("candidate frames are too small")
    thumb = cv2.resize(
        gray,
        (_PREFILTER_WIDTH, _PREFILTER_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.GaussianBlur(thumb, (3, 3), 0).astype(np.float32)


def _align_small_translation(reference, candidate):
    """Undo only tiny camera/book jitter for conservative duplicate grouping."""
    try:
        (dx, dy), response = cv2.phaseCorrelate(reference, candidate)
    except cv2.error:
        return candidate
    if (
        not np.isfinite([dx, dy, response]).all()
        or response < 0.15
        or abs(float(dx)) > _DUPLICATE_MAX_SHIFT
        or abs(float(dy)) > _DUPLICATE_MAX_SHIFT
    ):
        return candidate
    matrix = np.asarray(
        [[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        candidate,
        matrix,
        (_PREFILTER_WIDTH, _PREFILTER_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def near_identical_candidate_frames(first, second):
    """Return True only for frames safe to collapse before expensive analysis.

    The comparison is intentionally strict. A localized change such as a finger
    leaving the page should make at least one spatial block differ enough that
    the later frame remains a separate candidate.
    """
    left = _gray_thumb(first)
    right = _gray_thumb(second)
    right = _align_small_translation(left, right)

    # Remove only a small global exposure offset. Larger AE changes stay distinct.
    bias = float(np.median(left) - np.median(right))
    right = np.clip(right + np.clip(bias, -10.0, 10.0), 0, 255)
    delta = np.abs(left - right)
    mean_abs = float(delta.mean())

    block_means = []
    for ys in np.array_split(np.arange(delta.shape[0]), 4):
        for xs in np.array_split(np.arange(delta.shape[1]), 4):
            block_means.append(float(delta[np.ix_(ys, xs)].mean()))
    return (
        mean_abs <= _DUPLICATE_MEAN_ABS
        and max(block_means, default=0.0) <= _DUPLICATE_MAX_BLOCK_ABS
    )


def _coarse_score(sample):
    return math.log1p(max(0.0, float(sample.sharpness))) - 30.0 * max(
        0.0, float(sample.motion)
    )


def _representative_indices(frames):
    """Collapse only consecutive near-identical frames, keeping the later one."""
    if not frames:
        return [], []
    groups = [[0]]
    for index in range(1, len(frames)):
        previous = groups[-1][-1]
        if near_identical_candidate_frames(frames[previous], frames[index]):
            groups[-1].append(index)
        else:
            groups.append([index])
    representatives = [group[-1] for group in groups]
    return representatives, groups


def _groups_for_representatives(count, representatives):
    """Return groups whose last item is the representative kept for that group."""
    representatives = sorted(set(int(index) for index in representatives))
    if not representatives:
        return []
    groups = []
    start = 0
    for representative in representatives:
        groups.append(list(range(start, representative + 1)))
        start = representative + 1
    if start < count:
        groups[-1].extend(range(start, count))
    return groups


def _primary_indices(samples, representatives, limit):
    if not representatives:
        return []
    limit = max(1, min(int(limit), len(representatives)))
    if len(representatives) <= limit:
        return list(representatives)
    if limit == 1:
        return [representatives[-1]]

    # Always keep the latest distinct candidate so a hand that leaves late is
    # not lost. Fill earlier slots with the best low-motion/sharp candidate from
    # temporal bins covering the beginning and middle.
    latest = representatives[-1]
    earlier = representatives[:-1]
    bins = np.array_split(np.arange(len(earlier)), limit - 1)
    primary = []
    for group in bins:
        if not len(group):
            continue
        indices = [earlier[int(position)] for position in group]
        primary.append(max(indices, key=lambda index: (_coarse_score(samples[index]), index)))
    primary.append(latest)
    return sorted(set(primary))


def build_candidate_plan(samples, frames, primary_limit=3):
    """Plan staged evaluation from already-decoded low-resolution candidates."""
    samples = list(samples)
    frames = list(frames)
    if len(samples) != len(frames):
        raise ValueError("Candidate sample/frame count mismatch")

    # Small candidate pools are already cheap and are part of the established UI
    # behavior. Deduplicate only when there are enough candidates to save work.
    if len(samples) <= int(primary_limit):
        representatives = list(range(len(samples)))
        groups = [[index] for index in representatives]
    else:
        representatives, groups = _representative_indices(frames)
        if len(representatives) < min(int(primary_limit), len(samples)):
            # Keep temporal coverage even when a static spread looks nearly
            # identical for the whole interval.
            target = min(int(primary_limit), len(samples))
            for index in np.linspace(0, len(samples) - 1, target, dtype=int):
                if int(index) not in representatives:
                    representatives.append(int(index))
            representatives = sorted(representatives)

    # Padding temporal representatives can split a duplicate group. Rebuild the
    # diagnostics so each group's final index is exactly the representative that
    # survives; otherwise duplicate_groups would disagree with the actual plan.
    groups = _groups_for_representatives(len(samples), representatives)

    primary = _primary_indices(samples, representatives, primary_limit)
    primary_set = set(primary)
    representative_set = set(representatives)
    remaining = [index for index in representatives if index not in primary_set]
    suppressed = [
        index for index in range(len(samples)) if index not in representative_set
    ]
    return {
        "primary_indices": primary,
        "remaining_indices": remaining,
        "representative_indices": representatives,
        "duplicate_groups": groups,
        "duplicate_suppressed_indices": suppressed,
        "original_count": len(samples),
        "representative_count": len(representatives),
        "primary_count": len(primary),
        "duplicate_suppressed_count": len(suppressed),
    }


_STAGED_READABILITY_REASONS = frozenset(
    {
        "low_sharpness",
        "hand_overlap",
        "glare_overlap",
        "high_motion",
        "page_quad_uncertain",
        "underexposed",
    }
)


def staged_fallback_reasons(records, selection_mode):
    """Decide whether the cheap primary set failed without touching rescan policy."""
    if not records:
        return []

    if selection_mode == "per_page":
        triggered = set()
        for side in ("left", "right"):
            reasons_by_candidate = [
                set(record.get("page_suspect", {}).get(side, ()))
                & _STAGED_READABILITY_REASONS
                for record in records
            ]
            if reasons_by_candidate and all(reasons_by_candidate):
                for reasons in reasons_by_candidate:
                    triggered.update(reasons)
        return sorted(triggered)

    reasons_by_candidate = [
        set(record.get("suspect", ())) & _STAGED_READABILITY_REASONS
        for record in records
    ]
    if not reasons_by_candidate or not all(reasons_by_candidate):
        return []
    triggered = set()
    for reasons in reasons_by_candidate:
        triggered.update(reasons)
    return sorted(triggered)
