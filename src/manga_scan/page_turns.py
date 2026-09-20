from __future__ import annotations


def _sample_value(sample, name):
    return getattr(sample, name) if hasattr(sample, name) else sample[name]


def _event_from_run(run):
    peak = max(run, key=lambda sample: _sample_value(sample, "motion"))
    return {
        "start": float(_sample_value(run[0], "time")),
        "end": float(_sample_value(run[-1], "time")),
        "peak_time": float(_sample_value(peak, "time")),
        "peak_motion": float(_sample_value(peak, "motion")),
        "sample_count": len(run),
    }


def detect_page_turn_events(samples, low_threshold, high_threshold, fps):
    """Detect distinct page-turn motion bursts from the low-resolution motion trace.

    A single physical page turn may dip below the high threshold for one frame,
    so nearby high-motion runs are merged unless the valley is quiet enough to
    plausibly represent a briefly visible page.
    """
    if len(samples) < 2:
        return []

    runs = []
    current = []
    # The first sampled frame has a synthetic motion=1.0 because there is no
    # previous frame. It must never become a page-turn event.
    for sample in samples[1:]:
        if _sample_value(sample, "motion") >= high_threshold:
            current.append(sample)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return []

    split_threshold = low_threshold + 0.55 * (high_threshold - low_threshold)
    min_separation = max(0.15, 1.5 / max(float(fps), 1.0))
    merged = [runs[0]]

    for run in runs[1:]:
        previous = merged[-1]
        gap_start = float(_sample_value(previous[-1], "time"))
        gap_end = float(_sample_value(run[0], "time"))
        between = [
            sample
            for sample in samples
            if gap_start < float(_sample_value(sample, "time")) < gap_end
        ]
        valley = min(
            (float(_sample_value(sample, "motion")) for sample in between),
            default=high_threshold,
        )

        if gap_end - gap_start < min_separation or valley > split_threshold:
            previous.extend(run)
        else:
            merged.append(run)

    events = []
    for index, run in enumerate(merged, 1):
        event = _event_from_run(run)
        event["id"] = f"turn_{index:04d}"
        events.append(event)
    return events


def _segment_bounds(segments):
    bounds = []
    for index, segment in enumerate(segments):
        if not segment:
            continue
        bounds.append(
            {
                "id": f"spread_{index + 1:04d}",
                "start": float(_sample_value(segment[0], "time")),
                "end": float(_sample_value(segment[-1], "time")),
            }
        )
    return bounds


def detect_missing_between_turns(
    samples,
    segments,
    events,
    low_threshold,
    high_threshold,
):
    """Find briefly visible pages that never became a stable interval.

    Every pair of adjacent page-turn events normally has one detected stable
    spread between them. If it does not, use the lowest-motion sample in that
    window as the review/add-frame candidate.
    """
    if len(events) < 2:
        return []

    bounds = _segment_bounds(segments)
    motion_limit = low_threshold + 0.65 * (high_threshold - low_threshold)
    candidates = []

    for left, right in zip(events, events[1:]):
        window_start = float(left["end"])
        window_end = float(right["start"])
        if window_end <= window_start:
            continue

        if any(
            left["peak_time"] < (segment["start"] + segment["end"]) / 2 < right["peak_time"]
            for segment in bounds
        ):
            continue

        visible = [
            sample
            for sample in samples
            if window_start < float(_sample_value(sample, "time")) < window_end
        ]
        if not visible:
            continue

        best = min(visible, key=lambda sample: _sample_value(sample, "motion"))
        best_motion = float(_sample_value(best, "motion"))
        if best_motion > motion_limit:
            # There was continuous motion rather than a briefly exposed page.
            # Staying conservative here avoids splitting one complicated turn
            # into a false missing-page warning.
            continue

        candidate_time = float(_sample_value(best, "time"))
        before = max(
            (segment for segment in bounds if segment["end"] <= candidate_time),
            key=lambda segment: segment["end"],
            default=None,
        )
        after = min(
            (segment for segment in bounds if segment["start"] >= candidate_time),
            key=lambda segment: segment["start"],
            default=None,
        )
        candidates.append(
            {
                "id": f"{left['id']}-{right['id']}",
                "time": candidate_time,
                "motion": best_motion,
                "window_start": window_start,
                "window_end": window_end,
                "left_turn": left["id"],
                "right_turn": right["id"],
                "before": before["id"] if before else None,
                "after": after["id"] if after else None,
                "reason": "no_stable_interval_between_turns",
            }
        )

    return candidates


def analyze_page_turns(samples, segments, low_threshold, high_threshold, fps):
    events = detect_page_turn_events(samples, low_threshold, high_threshold, fps)
    missing = detect_missing_between_turns(
        samples,
        segments,
        events,
        low_threshold,
        high_threshold,
    )
    return {
        "version": 2,
        "turn_count": len(events),
        "events": events,
        "missing_candidates": missing,
        "thresholds": {
            "motion": float(low_threshold),
            "turn": float(high_threshold),
        },
    }
