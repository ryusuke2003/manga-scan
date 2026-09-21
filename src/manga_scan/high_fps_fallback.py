from __future__ import annotations

import math

RECOVERABLE_CANDIDATE_REASONS = frozenset(
    {
        "low_sharpness",
        "hand_overlap",
        "glare_overlap",
        "high_motion",
    }
)


def fallback_reasons(records, selection_mode):
    """Return reasons only when every usable candidate is bad for a recoverable cause."""
    if not records:
        return []

    if selection_mode == "per_page":
        triggered = set()
        for side in ("left", "right"):
            side_reasons = [
                set(record.get("page_suspect", {}).get(side, ()))
                & RECOVERABLE_CANDIDATE_REASONS
                for record in records
            ]
            if side_reasons and all(side_reasons):
                triggered.update().update() if False else None
                for reasons in side_reasons:
                    triggered.update(reasons)
        return sorted(triggered)

    reasons_by_candidate = [
        set(record.get("suspect", ())) & RECOVERABLE_CANDIDATE_REASONS
        for record in records
    ]
    if not reasons_by_candidate or not all(reasons_by_candidate):
        return []
    triggered = set()
    for reasons in reasons_by_candidate:
        triggered.update(reasons)
    return sorted(triggered)


def best_low_motion_run(samples, motion_threshold, fps, min_stable_seconds):
    """Find a genuinely still high-fps run inside a missing-page turn window."""
    if not samples:
        return []

    minimum = max(2, int(math.ceil(float(fps) * float(min_stable_seconds))))
    runs = []
    current = []
    for sample in samples:
        motion = float(getattr(sample, "motion"))
        if motion <= float(motion_threshold):
            current.append(sample)
        else:
            if len(current) >= minimum:
                runs.append(current)
            current = []
    if len(current) >= minimum:
        runs.append(current)
    if not runs:
        return []

    def score(run):
        mean_motion = sum(float(getattr(sample, "motion")) for sample in run) / len(run)
        mean_sharpness = sum(float(getattr(sample, "sharpness")) for sample in run) / len(run)
        duration = float(getattr(run[-1], "time")) - float(getattr(run[0], "time"))
        return (duration, mean_sharpness, -mean_motion)

    return max(runs, key=score)
