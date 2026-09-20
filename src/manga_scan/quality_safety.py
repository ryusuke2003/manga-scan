import math


_CANDIDATE_TRIGGER_REASONS = {
    "low_sharpness",
    "high_motion",
    "hand_overlap",
    "glare_overlap",
    "finger_repair_incomplete",
    "occlusion_repair_incomplete",
    "final_unresolved_finger",
    "final_finger_repair_residual",
    "final_glare_residual",
}

_CANDIDATE_RISK_REASONS = {
    "low_sharpness",
    "high_motion",
    "hand_overlap",
    "glare_overlap",
    "page_quad_uncertain",
    "underexposed",
    "source_frame_clipped",
    "page_contour_low_confidence",
}


def normalize_expected_page_count(value):
    if value in (None, ""):
        return None
    if type(value) is bool:
        raise ValueError("expected_page_count must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_page_count must be a positive integer") from exc
    if str(value).strip() not in (str(number), f"+{number}"):
        if not isinstance(value, int):
            raise ValueError("expected_page_count must be a positive integer")
    if not 1 <= number <= 10000:
        raise ValueError("expected_page_count must be 1..10000")
    return number


def physical_page_count(pages):
    total = 0
    output_items = 0
    for page in pages or []:
        if not page.get("enabled", True):
            continue
        output_items += 1
        total += 2 if page.get("side") == "spread" else 1
    return total, output_items


def refresh_page_count_check(manifest):
    expected = normalize_expected_page_count(manifest.get("expected_page_count"))
    manifest["expected_page_count"] = expected
    actual, output_items = physical_page_count(manifest.get("pages", []))
    difference = None if expected is None else actual - expected
    if expected is None:
        status = "unset"
    elif manifest.get("status") != "complete":
        status = "pending"
    elif difference == 0:
        status = "match"
    elif difference < 0:
        status = "short"
    else:
        status = "over"
    check = {
        "status": status,
        "expected": expected,
        "actual": actual,
        "output_items": output_items,
        "difference": difference,
    }
    manifest["page_count_check"] = check
    return check


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _selection_value(metrics):
    value = metrics.get("selection_score")
    if _finite(value):
        return float(value)
    value = metrics.get("score")
    return float(value) if _finite(value) else None


def _metric_value(metrics, *names):
    for name in names:
        value = metrics.get(name)
        if _finite(value):
            return float(value)
    return None


def _candidate_view(candidate, side):
    if side in ("left", "right"):
        metrics = (candidate.get("page_metrics") or {}).get(side) or {}
        suspects = (candidate.get("page_suspect") or {}).get(
            side,
            candidate.get("suspect", []),
        )
    else:
        metrics = candidate.get("metrics") or {}
        suspects = candidate.get("suspect", [])
    return metrics, set(suspects or [])


def _targeted_improvements(reasons, current, alternative):
    improvements = []

    if reasons & {"glare_overlap", "final_glare_residual", "occlusion_repair_incomplete"}:
        before = _metric_value(current, "glare_overlap", "glare")
        after = _metric_value(alternative, "glare_overlap", "glare")
        if before is not None and after is not None and before > 0:
            if after <= before * 0.70 and before - after >= 0.001:
                improvements.append("glare")

    if reasons & {
        "hand_overlap",
        "finger_repair_incomplete",
        "occlusion_repair_incomplete",
        "final_unresolved_finger",
        "final_finger_repair_residual",
    }:
        before = _metric_value(current, "hand_overlap")
        after = _metric_value(alternative, "hand_overlap")
        if before is not None and after is not None and before > 0:
            if after <= before * 0.70 and before - after >= 0.001:
                improvements.append("hand")

    if "low_sharpness" in reasons:
        before = _metric_value(current, "sharpness_uniformity", "sharpness")
        after = _metric_value(alternative, "sharpness_uniformity", "sharpness")
        if before is not None and after is not None and after >= before * 1.15:
            improvements.append("sharpness")

    if "high_motion" in reasons:
        before = _metric_value(current, "motion")
        after = _metric_value(alternative, "motion")
        if before is not None and after is not None and before > 0:
            if after <= before * 0.70:
                improvements.append("motion")

    return improvements


def _page_review_reasons(page):
    return set(page.get("suspect", [])) | set(
        (page.get("final_quality") or {}).get("reasons", [])
    )


def refresh_safe_fix_suggestions(manifest):
    spreads = {spread["id"]: spread for spread in manifest.get("spreads", [])}
    for page in manifest.get("pages", []):
        page.pop("safe_fix_suggestions", None)
        if not page.get("enabled", True):
            continue

        reasons = _page_review_reasons(page)
        trigger_reasons = reasons & _CANDIDATE_TRIGGER_REASONS
        if not trigger_reasons:
            continue

        spread = spreads.get(page.get("spread_id"))
        if not spread or len(spread.get("candidates", [])) < 2:
            continue

        side = page.get("side")
        metric_side = side if side in ("left", "right") else "spread"
        selected_id = page.get("candidate_id")
        if selected_id is None:
            if metric_side == "spread":
                selected_id = spread.get("selected")
            else:
                selected_id = (spread.get("selected_pages") or {}).get(
                    metric_side,
                    spread.get("selected"),
                )

        current = next(
            (
                candidate
                for candidate in spread.get("candidates", [])
                if candidate.get("id") == selected_id
            ),
            None,
        )
        if current is None:
            continue

        current_metrics, current_suspects = _candidate_view(current, metric_side)
        current_risks = current_suspects & _CANDIDATE_RISK_REASONS
        current_score = _selection_value(current_metrics)
        suggestions = []

        for candidate in spread.get("candidates", []):
            if candidate.get("id") == selected_id:
                continue
            alt_metrics, alt_suspects = _candidate_view(candidate, metric_side)
            alt_risks = alt_suspects & _CANDIDATE_RISK_REASONS

            # "Safe" means never introducing a new known capture risk.
            if not alt_risks.issubset(current_risks):
                continue

            improvements = _targeted_improvements(
                trigger_reasons,
                current_metrics,
                alt_metrics,
            )
            if len(alt_risks) < len(current_risks):
                improvements.append("risk_count")

            alt_score = _selection_value(alt_metrics)
            score_gain = None
            if current_score is not None and alt_score is not None:
                score_gain = alt_score - current_score
                if score_gain >= 0.08:
                    improvements.append("selection_score")

            improvements = list(dict.fromkeys(improvements))
            if not improvements:
                continue

            suggestions.append(
                {
                    "candidate_id": candidate["id"],
                    "side": None if metric_side == "spread" else metric_side,
                    "confidence": "high",
                    "improvements": improvements,
                    "score_gain": (
                        None if score_gain is None else round(float(score_gain), 4)
                    ),
                    "current_risks": sorted(current_risks),
                    "candidate_risks": sorted(alt_risks),
                }
            )

        suggestions.sort(
            key=lambda item: (
                len(item["candidate_risks"]),
                -len(item["improvements"]),
                -(item["score_gain"] if item["score_gain"] is not None else -1),
                item["candidate_id"],
            )
        )
        if suggestions:
            page["safe_fix_suggestions"] = suggestions[:2]

    return manifest


def refresh_review_safety(manifest):
    refresh_page_count_check(manifest)
    refresh_safe_fix_suggestions(manifest)
    return manifest
