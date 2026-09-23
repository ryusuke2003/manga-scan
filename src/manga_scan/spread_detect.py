"""Automatic reference-spread detection from a full setup frame."""

from __future__ import annotations

import cv2
import numpy as np

from .cover_detect import detect_cover_quad_candidates
from .motion import motion_score
from .page_contour import (
    consensus_page_quads,
    detect_page_quads,
    quad_edge_evidence,
    spread_quad_from_page_quads,
)
from .temporal_alignment import (
    align_page_detection,
    alignment_summary,
    estimate_frame_alignments,
    transform_normalized_quad,
)

REFERENCE_OUTLINE_CANDIDATES = 8
REFERENCE_CONSENSUS_MAX_CORNER_DEVIATION = 0.04
REFERENCE_AMBIGUITY_SCORE_MARGIN = 0.035
REFERENCE_PRIOR_SEARCH_WIDTH = 2
REFERENCE_PRIOR_SEARCH_STEPS = (0.10, 0.05)
REFERENCE_PRIOR_SEARCH_MAX_WIDTH = 600
REFERENCE_EDGE_MIN_SUPPORT = 0.22
REFERENCE_EDGE_HAND_OCCLUSION = 0.18
REFERENCE_UNCERTAIN_CONFIDENCE_CAP = 0.69


def _coarse_spread_priors():
    """Conservative full-frame priors for open books whose outer edge is obscured."""
    return [
        np.asarray([[0.04, 0.03], [0.86, 0.03], [0.86, 0.97], [0.04, 0.97]], np.float32),
        np.asarray([[0.04, 0.03], [0.96, 0.03], [0.96, 0.97], [0.04, 0.97]], np.float32),
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


def _working_image(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a numpy image")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image must be at least 32x32")
    scale = min(1.0, 1000 / max(image.shape[:2]))
    return (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )


def _working_mask(mask, shape):
    height, width = shape[:2]
    if mask is None:
        return np.zeros((height, width), np.uint8)
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("hand masks must be grayscale")
    if array.shape != (height, width):
        array = cv2.resize(
            array.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return (array > 0).astype(np.uint8) * 255


def _select_reference_search_frame(images, hand_masks, min_confidence):
    """Choose a nearby frame that exposes the spread boundary most clearly."""

    diagnostics = []
    for index, image in enumerate(images):
        mask = hand_masks[index]
        hand_fraction = float(np.count_nonzero(mask) / max(1, mask.size))
        hand_score = float(np.clip(1.0 - hand_fraction / 0.10, 0.0, 1.0))

        outlines = detect_cover_quad_candidates(
            image,
            min_confidence=max(0.35, float(min_confidence) - 0.20),
            area_range=(0.12, 0.96),
            aspect_range=(1.05, 3.2),
            target_aspect=1.75,
            limit=3,
        )
        outline_score = max(
            (float(item.get("confidence", 0.0)) for item in outlines),
            default=0.0,
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sharpness_score = float(np.clip(sharpness / 180.0, 0.0, 1.0))

        motions = []
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(images):
                try:
                    motions.append(motion_score(image, images[neighbor]))
                except ValueError:
                    pass
        frame_motion = float(np.median(motions)) if motions else 0.0
        stability_score = float(np.clip(1.0 - frame_motion / 0.04, 0.0, 1.0))

        score = (
            0.45 * outline_score
            + 0.25 * hand_score
            + 0.15 * stability_score
            + 0.15 * sharpness_score
        )
        diagnostics.append(
            {
                "index": index,
                "score": round(score, 4),
                "hand_fraction": round(hand_fraction, 4),
                "outline_score": round(outline_score, 4),
                "motion": round(frame_motion, 4),
                "sharpness": round(sharpness, 2),
            }
        )

    best = max(diagnostics, key=lambda item: item["score"])
    return int(best["index"]), diagnostics


def _priors_to_anchor(priors, search_index, anchor_index, alignments, frames):
    if search_index == anchor_index:
        return [np.asarray(prior, dtype=np.float32) for prior in priors]
    alignment = alignments[search_index]
    if alignment.get("status") != "aligned":
        return None
    transformed = []
    for prior in priors:
        try:
            transformed.append(
                transform_normalized_quad(
                    prior,
                    alignment["matrix"],
                    frames[search_index].shape,
                    frames[anchor_index].shape,
                )
            )
        except (ValueError, cv2.error):
            return None
    return transformed


def _outline_to_anchor(outline, search_index, anchor_index, alignments, frames):
    if outline is None or search_index == anchor_index:
        return outline
    transformed = _priors_to_anchor(
        [outline["roi"]],
        search_index,
        anchor_index,
        alignments,
        frames,
    )
    if not transformed:
        return None
    result = dict(outline)
    result["roi"] = transformed[0].tolist()
    return result


def _boundary_evidence(working_frames, pages, alignments, anchor_index, hand_masks):
    """Aggregate real outer-page edge evidence across aligned nearby frames."""

    anchor_quads = {
        side: np.asarray(pages[side]["quad"], dtype=np.float32)
        for side in ("left", "right")
    }
    per_frame = []
    for frame_index, frame in enumerate(working_frames):
        frame_quads = {
            side: quad.copy()
            for side, quad in anchor_quads.items()
        }
        alignment = alignments[frame_index]
        if frame_index != anchor_index:
            if alignment.get("status") != "aligned":
                continue
            try:
                anchor_to_frame = np.linalg.inv(alignment["matrix"])
                frame_quads = {
                    side: transform_normalized_quad(
                        quad,
                        anchor_to_frame,
                        working_frames[anchor_index].shape,
                        frame.shape,
                    )
                    for side, quad in anchor_quads.items()
                }
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                continue
        try:
            left = quad_edge_evidence(
                frame,
                frame_quads["left"],
                hand_masks[frame_index],
            )
            right = quad_edge_evidence(
                frame,
                frame_quads["right"],
                hand_masks[frame_index],
            )
        except ValueError:
            continue

        def combine(first, second):
            return {
                "support": min(first["support"], second["support"]),
                "visible_support": min(
                    first["visible_support"],
                    second["visible_support"],
                ),
                "occlusion": max(first["occlusion"], second["occlusion"]),
            }

        frame_edges = {
            "top": combine(left["top"], right["top"]),
            "right": right["right"],
            "bottom": combine(left["bottom"], right["bottom"]),
            "left": left["left"],
        }
        per_frame.append({"index": frame_index, "edges": frame_edges})

    edge_names = ("top", "right", "bottom", "left")
    aggregated = {}
    uncertain = []
    for edge in edge_names:
        samples = [
            {"index": item["index"], **item["edges"][edge]}
            for item in per_frame
        ]
        samples.sort(key=lambda item: item["support"], reverse=True)
        selected = samples[: min(2, len(samples))]
        if selected:
            support = float(np.mean([item["support"] for item in selected]))
            visible_support = float(
                np.mean([item["visible_support"] for item in selected])
            )
            occlusion = float(np.mean([item["occlusion"] for item in selected]))
        else:
            support = visible_support = occlusion = 0.0

        reasons = []
        if support < REFERENCE_EDGE_MIN_SUPPORT:
            reasons.append("weak_edge")
        if occlusion >= REFERENCE_EDGE_HAND_OCCLUSION:
            reasons.append("hand_occlusion")
        if reasons:
            uncertain.append(edge)
        aggregated[edge] = {
            "support": round(support, 4),
            "visible_support": round(visible_support, 4),
            "hand_occlusion": round(occlusion, 4),
            "reasons": reasons,
            "frames": selected,
        }

    minimum_support = min(aggregated[edge]["support"] for edge in edge_names)
    maximum_occlusion = max(
        aggregated[edge]["hand_occlusion"] for edge in edge_names
    )
    support_cap = 0.45 + 0.50 * min(1.0, minimum_support / 0.45)
    confidence_cap = min(0.95, support_cap)
    if uncertain:
        confidence_cap = min(confidence_cap, REFERENCE_UNCERTAIN_CONFIDENCE_CAP)
    return {
        "edges": aggregated,
        "minimum_support": round(float(minimum_support), 4),
        "maximum_hand_occlusion": round(float(maximum_occlusion), 4),
        "uncertain_edges": uncertain,
        "confidence_cap": round(float(confidence_cap), 4),
        "frame_count": len(per_frame),
    }


def _adjust_prior(roi, left=0.0, right=0.0, top=0.0, bottom=0.0):
    """Move a quad's four edges in its local bilinear coordinate system."""

    q = np.asarray(roi, dtype=np.float32)
    coordinates = (
        (-left, -top),
        (1 + right, -top),
        (1 + right, 1 + bottom),
        (-left, 1 + bottom),
    )
    points = []
    for u, v in coordinates:
        point = (
            q[0] * (1 - u) * (1 - v)
            + q[1] * u * (1 - v)
            + q[2] * u * v
            + q[3] * (1 - u) * v
        )
        points.append(point)
    adjusted = np.asarray(points, dtype=np.float32)
    if not np.isfinite(adjusted).all():
        raise ValueError("adjusted prior must remain finite")
    adjusted = np.clip(adjusted, 0, 1)
    if abs(float(cv2.contourArea(adjusted))) < 0.02:
        raise ValueError("adjusted prior is too small")
    return adjusted


def _outline_priors(roi):
    q = np.asarray(roi, dtype=np.float32)

    def extrapolate(left=0.0, right=0.0, top=0.0, bottom=0.0):
        return _adjust_prior(q, left=left, right=right, top=top, bottom=bottom)

    # Hough often locks onto an interior horizontal line while retaining the
    # real lower/side edges. Include asymmetric outward hypotheses so page
    # verification can recover that missing edge without accepting it blindly.
    return [
        q,
        extrapolate(left=0.06, right=0.12, top=0.12, bottom=0.12),
        extrapolate(left=0.06, right=0.16, top=0.38),
        extrapolate(left=0.06, right=0.16, bottom=0.38),
        extrapolate(left=0.10, right=0.18, top=0.24, bottom=0.24),
    ]


def _prior_key(prior):
    return tuple(np.rint(np.asarray(prior, dtype=np.float32).reshape(-1) * 1000).astype(int))


def _distinct_priors(priors, minimum_distance=0.012):
    selected = []
    for prior in priors:
        points = np.asarray(prior, dtype=np.float32)
        if any(
            float(np.mean(np.linalg.norm(points - existing, axis=1))) < minimum_distance
            for existing in selected
        ):
            continue
        selected.append(points)
    return selected


def _prior_search_frame(image):
    height, width = image.shape[:2]
    scale = min(1.0, REFERENCE_PRIOR_SEARCH_MAX_WIDTH / max(width, height))
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _score_prior(image, prior, min_confidence):
    try:
        pages = detect_page_quads(
            image,
            prior,
            spine_ratio=0.5,
            min_confidence=float(min_confidence),
        )
    except (ValueError, cv2.error):
        return 0.0
    left = float(pages["left"]["confidence"])
    right = float(pages["right"]["confidence"])
    detected_sides = int(pages["left"]["detected"]) + int(pages["right"]["detected"])
    return (
        0.62 * min(left, right)
        + 0.28 * ((left + right) / 2)
        + 0.05 * detected_sides
    )


def _prior_mutations(prior, step):
    mutations = [np.asarray(prior, dtype=np.float32)]
    for edge in ("left", "right", "top", "bottom"):
        for amount in (-float(step), float(step)):
            adjustment = {edge: amount}
            try:
                mutations.append(_adjust_prior(prior, **adjustment))
            except ValueError:
                continue
    return mutations


def _refine_priors(image, initial_priors, min_confidence):
    """Search inward/outward edge adjustments on the anchor frame."""

    search_image = _prior_search_frame(image)
    score_cache = {}
    evaluated = 0

    def ranked(priors):
        nonlocal evaluated
        records = []
        for prior in _distinct_priors(priors):
            key = _prior_key(prior)
            if key not in score_cache:
                score_cache[key] = _score_prior(search_image, prior, min_confidence)
                evaluated += 1
            records.append((score_cache[key], prior))
        records.sort(key=lambda item: item[0], reverse=True)
        return records

    initial_ranking = ranked(initial_priors)
    final_ranking = initial_ranking
    beam = final_ranking[:REFERENCE_PRIOR_SEARCH_WIDTH]
    for step in REFERENCE_PRIOR_SEARCH_STEPS:
        candidates = []
        for _score, prior in beam:
            candidates.extend(_prior_mutations(prior, step))
        final_ranking = ranked(candidates)
        beam = final_ranking[:REFERENCE_PRIOR_SEARCH_WIDTH]

    output = []
    for record in [
        *final_ranking[:1],
        *initial_ranking[:1],
        *final_ranking[1:],
    ]:
        _score, prior = record
        if any(
            float(np.mean(np.linalg.norm(prior - existing[1], axis=1))) < 0.012
            for existing in output
        ):
            continue
        output.append(record)
        if len(output) >= REFERENCE_PRIOR_SEARCH_WIDTH:
            break
    return [prior for _score, prior in output], {
        "evaluated": evaluated,
        "best_score": round(float(output[0][0]), 4) if output else 0.0,
        "beam_width": len(beam),
        "baseline_preserved": bool(
            initial_ranking
            and any(
                float(np.mean(np.linalg.norm(initial_ranking[0][1] - prior, axis=1)))
                < 0.012
                for _score, prior in output
            )
        ),
    }


def _proposal_from_prior_set(
    working_frames,
    priors,
    min_confidence,
    *,
    alignments,
    anchor_index,
    proposal_id,
    outline=None,
    source,
    local_search=None,
    hand_masks=None,
):
    detections = []
    failures = []
    for frame_index, working in enumerate(working_frames):
        frame_priors = priors
        alignment = alignments[frame_index]
        if alignment.get("status") == "aligned":
            try:
                anchor_to_frame = np.linalg.inv(alignment["matrix"])
                frame_priors = [
                    transform_normalized_quad(
                        prior,
                        anchor_to_frame,
                        working_frames[anchor_index].shape,
                        working.shape,
                    )
                    for prior in priors
                ]
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                frame_priors = priors
        success, failure = _detect_pages_from_priors(
            working,
            frame_priors,
            min_confidence,
            confidence_cap=float(outline["confidence"]) if outline else None,
        )
        if failure is not None:
            failures.append(failure)
        pages = success[2] if success is not None else failure[1] if failure is not None else None
        if pages is None:
            continue
        pages = dict(pages)
        pages["candidate_id"] = frame_index
        pages = align_page_detection(
            pages,
            alignment,
            working.shape,
            working_frames[anchor_index].shape,
        )
        detections.append(pages)

    minimum_frames = 2 if len(working_frames) > 1 else 1
    if not detections:
        failure_confidence = max((item[0] for item in failures), default=0.0)
        return None, failure_confidence

    pages = consensus_page_quads(
        detections,
        min_confidence=min_confidence,
        max_corner_deviation=REFERENCE_CONSENSUS_MAX_CORNER_DEVIATION,
        anchor_ids={"left": anchor_index, "right": anchor_index},
    )
    if not pages["detected"]:
        return None, float(pages["confidence"])
    if any(pages[side].get("consensus_count", 0) < minimum_frames for side in ("left", "right")):
        return None, float(pages["confidence"])
    try:
        roi = spread_quad_from_page_quads(pages)
    except ValueError:
        return None, float(pages["confidence"])
    if _has_frame_edge_background(working_frames[anchor_index], pages):
        return None, float(pages["confidence"])

    masks = hand_masks or [
        np.zeros(frame.shape[:2], np.uint8) for frame in working_frames
    ]
    boundary = _boundary_evidence(
        working_frames,
        pages,
        alignments,
        anchor_index,
        masks,
    )
    geometry_confidence = float(pages["confidence"])
    confidence = min(geometry_confidence, float(boundary["confidence_cap"]))

    temporal_support = min(
        pages["left"]["consensus_count"],
        pages["right"]["consensus_count"],
    ) / max(len(working_frames), 1)
    outline_confidence = float(outline["confidence"]) if outline else 0.62
    edge_quality = float(
        np.mean(
            [
                min(1.0, boundary["edges"][edge]["support"] / 0.55)
                for edge in ("top", "right", "bottom", "left")
            ]
        )
    )
    score = (
        0.56 * geometry_confidence
        + 0.18 * float(temporal_support)
        + 0.10 * outline_confidence
        + 0.16 * edge_quality
    )
    return {
        "proposal_id": proposal_id,
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 4),
        "geometry_confidence": round(float(geometry_confidence), 4),
        "boundary_evidence": boundary,
        "roi": roi,
        "outline": outline,
        "pages": pages,
        "source": source,
        "frame_support": min(
            pages["left"]["consensus_count"],
            pages["right"]["consensus_count"],
        ),
        "frame_count": len(working_frames),
        "local_search": local_search,
    }, float(confidence)


def _distinct_proposals(first, second):
    first_roi = np.asarray(first["roi"], dtype=np.float32)
    second_roi = np.asarray(second["roi"], dtype=np.float32)
    return float(np.mean(np.linalg.norm(first_roi - second_roi, axis=1))) >= 0.025


def _has_frame_edge_background(image, pages):
    """Reject a page candidate that is actually horizontal desk grain.

    The coarse search can split a narrow cover and the desk beside it into two
    convincing quads. Only inspect a side that reaches the image boundary: a
    real clipped page still has ink in both gradient directions, whereas the
    exposed wood in this case has almost exclusively horizontal grain.
    """

    height, width = image.shape[:2]
    clipped_sides = []
    for side in ("left", "right"):
        quad = np.asarray(pages[side]["quad"], dtype=np.float32)
        if np.any((quad <= 0.002) | (quad >= 0.998)):
            clipped_sides.append(quad)
    if not clipped_sides:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    for quad in clipped_sides:
        points = np.rint(quad * [width - 1, height - 1]).astype(np.int32)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        margin = max(5, round(min(height, width) * 0.025)) | 1
        mask = cv2.erode(mask, np.ones((margin, margin), np.uint8))
        interior = mask > 0
        if np.count_nonzero(interior) < 100:
            continue
        vertical_gradient = float(np.mean(gx[interior]))
        horizontal_gradient = float(np.mean(gy[interior]))
        if horizontal_gradient > 10 and vertical_gradient / horizontal_gradient < 0.45:
            return True
    return False


def detect_reference_spread_consensus(
    images,
    min_confidence=0.55,
    *,
    max_candidates=8,
    anchor_index=None,
    hand_masks=None,
):
    """Detect a spread by verifying top Hough candidates across nearby frames.

    The center frame supplies 5–10 outline hypotheses. Every hypothesis is
    reused as the prior on all supplied frames, and is accepted only when both
    page quads form a stable multi-frame consensus. A close, geometrically
    distinct runner-up is surfaced to the UI instead of being hidden behind a
    misleading high-confidence result.
    """

    images = list(images)
    if not images:
        raise ValueError("images must contain at least one frame")
    if not 0 <= float(min_confidence) <= 1:
        raise ValueError("min_confidence must be 0..1")
    if not 5 <= int(max_candidates) <= 10:
        raise ValueError("max_candidates must be 5..10")
    working_frames = [_working_image(image) for image in images]
    if anchor_index is None:
        anchor_index = len(working_frames) // 2
    if not 0 <= int(anchor_index) < len(working_frames):
        raise ValueError("anchor_index must identify one supplied frame")

    if hand_masks is None:
        working_masks = [
            np.zeros(frame.shape[:2], np.uint8) for frame in working_frames
        ]
    else:
        hand_masks = list(hand_masks)
        if len(hand_masks) != len(working_frames):
            raise ValueError("hand_masks must match the supplied frame count")
        working_masks = [
            _working_mask(mask, frame.shape)
            for mask, frame in zip(hand_masks, working_frames)
        ]

    search_index, frame_quality = _select_reference_search_frame(
        working_frames,
        working_masks,
        min_confidence,
    )
    alignments = estimate_frame_alignments(working_frames, int(anchor_index))
    if (
        search_index != int(anchor_index)
        and alignments[search_index].get("status") != "aligned"
    ):
        search_index = int(anchor_index)
    center = working_frames[search_index]
    alignment = alignment_summary(alignments)
    outlines = detect_cover_quad_candidates(
        center,
        min_confidence=max(0.42, float(min_confidence) - 0.12),
        area_range=(0.12, 0.96),
        aspect_range=(1.05, 3.2),
        target_aspect=1.75,
        limit=int(max_candidates),
    )

    proposals = []
    search_diagnostics = []
    best_failure = 0.0
    for index, outline in enumerate(outlines):
        refined_priors, search = _refine_priors(
            center,
            _outline_priors(outline["roi"]),
            min_confidence,
        )
        refined_priors = _priors_to_anchor(
            refined_priors,
            search_index,
            int(anchor_index),
            alignments,
            working_frames,
        )
        if not refined_priors:
            continue
        anchor_outline = _outline_to_anchor(
            outline,
            search_index,
            int(anchor_index),
            alignments,
            working_frames,
        )
        search_diagnostics.append({"proposal_id": f"hough_{index + 1}", **search})
        proposal, failure = _proposal_from_prior_set(
            working_frames,
            refined_priors,
            min_confidence,
            alignments=alignments,
            anchor_index=int(anchor_index),
            proposal_id=f"hough_{index + 1}",
            outline=anchor_outline,
            source="outline_pages_consensus" if len(images) > 1 else "outline_pages",
            local_search=search,
            hand_masks=working_masks,
        )
        best_failure = max(best_failure, failure)
        if proposal is not None:
            proposals.append(proposal)

    # Keep the established centered fallback as one explicit hypothesis. It is
    # especially useful when hands obscure the true outer Hough lines.
    coarse_priors, coarse_search = _refine_priors(
        center,
        _coarse_spread_priors(),
        min_confidence,
    )
    coarse_priors = _priors_to_anchor(
        coarse_priors,
        search_index,
        int(anchor_index),
        alignments,
        working_frames,
    )
    search_diagnostics.append({"proposal_id": "coarse", **coarse_search})
    coarse, failure = (None, 0.0)
    if coarse_priors:
        coarse, failure = _proposal_from_prior_set(
            working_frames,
            coarse_priors,
            min_confidence,
            alignments=alignments,
            anchor_index=int(anchor_index),
            proposal_id="coarse",
            source="coarse_pages_consensus" if len(images) > 1 else "coarse_pages",
            local_search=coarse_search,
            hand_masks=working_masks,
        )
    best_failure = max(best_failure, failure)
    if coarse is not None:
        proposals.append(coarse)

    proposals.sort(key=lambda item: item["score"], reverse=True)
    if not proposals:
        outline_confidence = max(
            (float(outline["confidence"]) for outline in outlines),
            default=0.0,
        )
        return {
            "detected": False,
            "confidence": round(max(outline_confidence, best_failure), 4),
            "roi": None,
            "outline": outlines[0] if outlines else {
                "detected": False, "confidence": 0.0, "roi": None,
            },
            "pages": None,
            "stage": "pages" if outlines or best_failure > 0 else "outline",
            "source": None,
            "ambiguous": False,
            "requires_confirmation": False,
            "candidate_count": len(outlines),
            "alignment": alignment,
            "search_frame_index": search_index,
            "frame_quality": frame_quality,
            "prior_search": search_diagnostics,
            "alternatives": [],
        }

    best = proposals[0]
    runner_up = next(
        (item for item in proposals[1:] if _distinct_proposals(best, item)),
        None,
    )
    ambiguous = bool(
        runner_up is not None
        and float(best["score"]) - float(runner_up["score"])
        <= REFERENCE_AMBIGUITY_SCORE_MARGIN
    )
    alternatives = [
        {
            "proposal_id": item["proposal_id"],
            "score": item["score"],
            "confidence": item["confidence"],
            "geometry_confidence": item.get("geometry_confidence"),
            "roi": item["roi"],
            "source": item["source"],
            "boundary_evidence": item.get("boundary_evidence"),
            "frame_support": item["frame_support"],
            "local_search": item["local_search"],
        }
        for item in proposals[:3]
    ]
    boundary_uncertain = bool(
        best.get("boundary_evidence", {}).get("uncertain_edges")
    )
    return {
        "detected": True,
        "confidence": best["confidence"],
        "geometry_confidence": best.get("geometry_confidence", best["confidence"]),
        "boundary_evidence": best.get("boundary_evidence"),
        "uncertain_edges": (
            best.get("boundary_evidence", {}).get("uncertain_edges", [])
        ),
        "roi": best["roi"],
        "outline": best["outline"] or {
            "detected": False, "confidence": 0.0, "roi": None,
        },
        "pages": best["pages"],
        "stage": "complete",
        "source": best["source"],
        "ambiguous": ambiguous,
        "requires_confirmation": ambiguous or boundary_uncertain,
        "candidate_count": len(outlines),
        "frame_support": best["frame_support"],
        "frame_count": best["frame_count"],
        "alignment": alignment,
        "search_frame_index": search_index,
        "frame_quality": frame_quality,
        "local_search": best["local_search"],
        "prior_search": search_diagnostics,
        "score_margin": round(
            float(best["score"]) - float(runner_up["score"]), 4
        ) if runner_up is not None else None,
        "alternatives": alternatives,
    }


def detect_reference_spread(image, min_confidence=0.55):
    """Detect both pages from the full frame and return one outer spread ROI.

    Prefer a broad Hough-based book outline, then verify the left/right pages.
    If hands, page color, or curvature break that outer outline, retry from a
    small set of conservative centered priors and let the two page detectors
    establish the actual outer corners. Only a verified two-page result is
    accepted; otherwise callers still fall back to manual four-point selection.
    """

    return detect_reference_spread_consensus(
        [image],
        min_confidence=min_confidence,
        max_candidates=REFERENCE_OUTLINE_CANDIDATES,
    )


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
        quad = np.rint(quad).astype(np.int32)
        uncertain = set(detection.get("uncertain_edges") or [])
        edges = (
            ("top", 0, 1),
            ("right", 1, 2),
            ("bottom", 2, 3),
            ("left", 3, 0),
        )
        for name, first, second in edges:
            color = (60, 60, 240) if name in uncertain else (255, 220, 80)
            cv2.line(
                canvas,
                tuple(quad[first]),
                tuple(quad[second]),
                color,
                3,
                cv2.LINE_AA,
            )
    return canvas
