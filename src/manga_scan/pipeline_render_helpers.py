"""Rendering helpers shared by the scan pipeline.

These functions are intentionally kept free of orchestration state. The small
wrappers in pipeline.py inject monkeypatchable detector dependencies so the
existing public/testing surface remains stable.
"""

import math

import cv2
import numpy as np

from .glare import detect_glare_mask
from .hand import boundary_finger_mask
from .page_contour import (
    consensus_page_quads,
    detect_page_quads,
    draw_page_quads,
    spread_quad_from_page_quads,
)
from .page_warp import warp_detected_pages
from .perspective import rotate_roi, validate_roi, warp_roi
from .split import rotate_image, spine_position, split_spread
from .spread_boundary import refine_spread_boundary
from .storage import save_image, write_json
from .temporal_alignment import (
    align_page_detection,
    alignment_summary,
    estimate_frame_alignments,
)

PAGE_CONTOUR_CONSENSUS_MAX_CORNER_DEVIATION = 0.04


def detect_spread_page_consensus(
    project,
    spread,
    cfg,
    anchor_ids=None,
    detect_page_quads_fn=detect_page_quads,
):
    """Estimate page quads from all saved candidates after temporal alignment.

    Each page side is aligned into the coordinate system of the candidate that
    will actually render that side. This keeps consensus useful when the book or
    camera drifts between candidate frames instead of treating the shift itself
    as a contour outlier.
    """

    anchor_ids = dict(anchor_ids or {})
    records = []
    images = []
    detections = []
    overrides = spread.get("roi_overrides", {})
    for record in spread.get("candidates", []):
        path = record.get("path")
        if not path:
            continue
        image = cv2.imread(str(project / path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        upright = rotate_image(image, cfg.rotation)
        override = overrides.get(str(record["id"]))
        source_roi = override if override is not None else record["roi"]
        roi = rotate_roi(source_roi, cfg.rotation).tolist()
        detection = detect_page_quads_fn(
            upright,
            roi,
            spine_ratio=spread.get("spine_ratio", cfg.spine_ratio),
            min_confidence=cfg.page_contour_min_confidence,
        )
        detection["candidate_id"] = record["id"]
        records.append(record)
        images.append(upright)
        detections.append(detection)

    if not detections:
        return None

    side_results = {}
    alignment_by_side = {}
    candidate_ids = [record["id"] for record in records]
    default_anchor = spread.get("selected", candidate_ids[0])
    for side in ("left", "right"):
        anchor_id = anchor_ids.get(side, default_anchor)
        try:
            anchor_index = candidate_ids.index(anchor_id)
        except ValueError:
            anchor_index = 0
            anchor_id = candidate_ids[0]

        alignments = estimate_frame_alignments(images, anchor_index)
        aligned = [
            align_page_detection(
                detection,
                alignment,
                image.shape,
                images[anchor_index].shape,
            )
            for detection, alignment, image in zip(detections, alignments, images)
        ]
        consensus = consensus_page_quads(
            aligned,
            min_confidence=cfg.page_contour_min_confidence,
            max_corner_deviation=PAGE_CONTOUR_CONSENSUS_MAX_CORNER_DEVIATION,
            anchor_ids={side: anchor_id},
        )
        side_results[side] = consensus[side]
        alignment_by_side[side] = {
            "anchor_id": anchor_id,
            **alignment_summary(alignments),
        }

    result = {
        "left": side_results["left"],
        "right": side_results["right"],
    }
    result["confidence"] = min(
        result["left"]["confidence"],
        result["right"]["confidence"],
    )
    result["detected"] = bool(
        result["left"]["detected"] and result["right"]["detected"]
    )
    result["consensus"] = {
        "candidate_count": len(detections),
        "candidate_ids": candidate_ids,
        "alignment_by_side": alignment_by_side,
    }
    return result


def rectify_spread_pages(
    project,
    image,
    rectified,
    chosen_roi,
    spread,
    cfg,
    page_detection=None,
    consensus_sides=None,
    manual_sides=None,
    detect_page_quads_fn=detect_page_quads,
):
    """Choose per-page perspective correction or the legacy spread fallback."""

    ratio = spread.get("spine_ratio", cfg.spine_ratio)
    manual_sides = list(manual_sides or [])
    force_manual_page = any(
        _page_override(spread, side).get("page_quad_mode") == "manual"
        for side in manual_sides
    )
    if (
        cfg.perspective_mode == "per_page" or force_manual_page
    ) and (not spread.get("manual_roi") or force_manual_page):
        detection_image = image
        if image.shape[1] > cfg.analysis_width:
            scale = cfg.analysis_width / image.shape[1]
            detection_image = cv2.resize(
                image,
                (cfg.analysis_width, max(2, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        detection = None
        if page_detection is None or consensus_sides:
            detection = detect_page_quads_fn(
                detection_image,
                chosen_roi,
                spine_ratio=ratio,
                min_confidence=cfg.page_contour_min_confidence,
            )
        if page_detection is not None:
            if consensus_sides:
                detection = dict(detection)
                for side in consensus_sides:
                    detection[side] = page_detection[side]
                detection["confidence"] = min(
                    detection["left"]["confidence"],
                    detection["right"]["confidence"],
                )
                detection["detected"] = (
                    detection["left"]["detected"] and detection["right"]["detected"]
                )
                detection["consensus"] = page_detection.get("consensus", {})
            else:
                detection = page_detection
        detection = _apply_manual_page_quads(detection, spread, manual_sides)
        spread["page_contours"] = detection
        debug_path = f"debug/page_contours/{spread['id']}.jpg"
        save_image(project / debug_path, draw_page_quads(image, detection))
        spread["page_contour_debug"] = debug_path

        if detection["detected"] or detection.get("manual_sides"):
            spread["spine_px"] = spine_position(rectified, ratio, cfg.split_mode)
            warped = warp_detected_pages(image, detection)
            if force_manual_page and cfg.perspective_mode != "per_page":
                fallback_sides, spine = split_spread(
                    rectified,
                    ratio,
                    cfg.split_mode,
                    cfg.gutter_fraction,
                )
                spread["spine_px"] = spine
                applied_manual_sides = [
                    side
                    for side in manual_sides
                    if _page_override(spread, side).get("page_quad_mode") == "manual"
                ]
                for side in applied_manual_sides:
                    fallback_sides[side] = warped[side]
                spread["perspective_mode_used"] = "mixed_manual"
                spread["manual_page_sides"] = applied_manual_sides
                return fallback_sides

            spread["perspective_mode_used"] = "per_page"
            spread.pop("manual_page_sides", None)
            return warped

        spread["perspective_mode_used"] = "spread_fallback"
        spread.pop("manual_page_sides", None)
        extra = spread.setdefault("extra_suspect", [])
        if "page_contour_low_confidence" not in extra:
            extra.append("page_contour_low_confidence")
    else:
        spread["perspective_mode_used"] = "spread"
        spread.pop("manual_page_sides", None)
        spread.pop("page_contours", None)
        spread.pop("page_contour_debug", None)

    sides, spine = split_spread(rectified, ratio, cfg.split_mode, cfg.gutter_fraction)
    spread["spine_px"] = spine
    return sides


def candidate_page_hand_mask(project, data, side, cfg, boundary_finger_mask_fn=boundary_finger_mask):
    chosen = data["chosen"]
    mask_path = chosen.get("hand_mask")
    if not mask_path:
        return None
    mask = cv2.imread(str(project / mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    source_height, source_width = data["source_frame_shape"][:2]
    full_mask = cv2.resize(
        mask,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    )
    rotated_mask = rotate_image(full_mask, cfg.rotation)
    state = data["state"]

    def supplement(page_mask):
        # Re-evaluate the final page boundary, including newly recovered pixels
        # outside an older candidate's ROI. Saved masks alone miss those fingers.
        page = data["sides"][side]
        extra = boundary_finger_mask_fn(page, [[0, 0], [1, 0], [1, 1], [0, 1]], cfg.hand_padding)
        return page_mask | extra

    perspective_mode_used = state.get("perspective_mode_used")
    page_uses_detected_quad = (
        perspective_mode_used == "per_page"
        or (
            perspective_mode_used == "mixed_manual"
            and side in set(state.get("manual_page_sides", []))
        )
    )
    if (
        page_uses_detected_quad
        and (state.get("page_contours") or {}).get("detected")
    ):
        output_sizes = {
            name: (page.shape[1], page.shape[0])
            for name, page in data["sides"].items()
        }
        return supplement(warp_detected_pages(
            rotated_mask,
            state["page_contours"],
            output_sizes=output_sizes,
            interpolation=cv2.INTER_NEAREST,
        )[side])

    rectified_mask = rotate_image(
        warp_roi(
            full_mask,
            chosen["roi"],
            interpolation=cv2.INTER_NEAREST,
        ),
        cfg.rotation,
    )
    spine = state.get("spine_px")
    if spine is None:
        spine = spine_position(
            data["rectified"],
            state.get("spine_ratio", cfg.spine_ratio),
            cfg.split_mode,
        )
    gutter = round(rectified_mask.shape[1] * cfg.gutter_fraction / 2)
    left_end = max(1, spine - gutter)
    right_start = min(rectified_mask.shape[1] - 1, spine + gutter)
    return supplement(
        rectified_mask[:, :left_end]
        if side == "left"
        else rectified_mask[:, right_start:]
    )


def _union_occlusion_masks(*masks):
    """Return a 0/255 union mask while preserving the existing mask contract."""
    available = [mask for mask in masks if mask is not None]
    if not available:
        return None
    shape = available[0].shape[:2]
    combined = np.zeros(shape, np.uint8)
    for mask in available:
        if mask.shape[:2] != shape:
            mask = cv2.resize(
                mask,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        combined[mask > 127] = 255
    return combined


def candidate_page_glare_mask(data, side, cfg):
    """Detect glare in final page coordinates used by repair alignment."""
    if not cfg.glare_repair:
        return None
    return detect_glare_mask(data["sides"][side])


def _page_override(spread, side):
    overrides = spread.get("page_overrides", {})
    override = overrides.get(side, {})
    return override if isinstance(override, dict) else {}


def _page_render_settings(spread, side, cfg):
    """Resolve project settings plus optional page-level Review overrides."""

    override = _page_override(spread, side)
    dewarp_override = override.get("dewarp")
    if dewarp_override is None:
        dewarp_enabled = (
            cfg.dewarp_mode != "off"
            and side not in set(spread.get("dewarp_disabled_sides", []))
        )
    else:
        dewarp_enabled = bool(dewarp_override)

    dewarp_mode = cfg.dewarp_mode if cfg.dewarp_mode != "off" else "auto"
    if not dewarp_enabled:
        dewarp_mode = "off"

    manual_quad = override.get("manual_quad")
    page_quad_mode = (
        "manual"
        if override.get("page_quad_mode") == "manual" and manual_quad is not None
        else "auto"
    )
    return {
        "dewarp": bool(dewarp_enabled),
        "dewarp_mode": dewarp_mode,
        "illumination_correction": bool(
            override.get("illumination_correction", cfg.illumination_correction)
        ),
        "white_normalization": bool(
            override.get("white_normalization", cfg.white_normalization)
        ),
        "page_quad_mode": page_quad_mode,
        "manual_quad": manual_quad if page_quad_mode == "manual" else None,
    }


def _apply_manual_page_quads(detection, spread, sides=None):
    """Overlay validated page-level manual quads on automatic detection."""

    if detection is None:
        return detection
    active_sides = set(sides or ("left", "right"))
    updated = dict(detection)
    manual_sides = []
    for side in ("left", "right"):
        if side not in active_sides:
            continue
        override = _page_override(spread, side)
        if override.get("page_quad_mode") != "manual":
            continue
        quad = validate_roi(override.get("manual_quad")).tolist()
        updated[side] = {
            **updated.get(side, {}),
            "quad": quad,
            "confidence": 1.0,
            "detected": True,
            "touches_frame": any(
                value <= 0.002 or value >= 0.998
                for point in quad
                for value in point
            ),
            "manual": True,
        }
        manual_sides.append(side)

    if manual_sides:
        updated["confidence"] = min(
            float(updated["left"].get("confidence", 0.0)),
            float(updated["right"].get("confidence", 0.0)),
        )
        updated["detected"] = bool(
            updated["left"].get("detected") and updated["right"].get("detected")
        )
        updated["manual_sides"] = manual_sides
    return updated


def _finger_donor_candidates(spread, side, selected_id):
    def rank(candidate):
        metrics = candidate.get("page_metrics", {}).get(side, candidate.get("metrics", {}))
        overlap = metrics.get("hand_overlap")
        glare_overlap = float(metrics.get("glare_overlap", metrics.get("glare", 0.0)) or 0.0)
        return (
            1.0 if overlap is None else float(overlap),
            glare_overlap,
            -float(metrics.get("selection_score", metrics.get("score", 0.0))),
        )

    return sorted(
        (candidate for candidate in spread["candidates"] if candidate["id"] != selected_id),
        key=rank,
    )


def _persist_finger_repair_component_debug(project, repair, stem):
    """Persist optional local-repair metadata without changing the core return contract."""
    if not isinstance(repair, dict) or "components" not in repair:
        return repair
    components = repair.get("components")
    if not components:
        return repair

    local_components = []
    applied_component_ids = set()
    max_shift = 0.0
    for component in components:
        if not isinstance(component, dict):
            continue
        component_id = component.get("component_id")
        for donor in component.get("donors", []):
            if not isinstance(donor, dict) or donor.get("method") != "local":
                continue
            dx = float(donor.get("dx", 0.0))
            dy = float(donor.get("dy", 0.0))
            local_components.append(
                {
                    "component_id": component_id,
                    "candidate_id": donor.get("candidate_id"),
                    "local_score": donor.get("local_score"),
                    "dx": dx,
                    "dy": dy,
                    "coverage": donor.get("coverage"),
                }
            )
            applied_component_ids.add(component_id)
            max_shift = max(max_shift, math.hypot(dx, dy))

    if local_components and "local_alignment" not in repair:
        repair["local_alignment"] = {
            "component_count": len(applied_component_ids),
            "max_shift_px": round(max_shift, 3),
            "components": local_components,
        }

    if repair.get("components_debug"):
        return repair
    path = f"debug/finger_repair/{stem}_components.json"
    payload = {"components": components}
    if repair.get("local_alignment"):
        payload["local_alignment"] = repair["local_alignment"]
    for key in ("component_count", "rejected_donors", "rejection_counts"):
        if key in repair:
            payload[key] = repair[key]
    write_json(project / path, payload)
    repair["components_debug"] = path
    return repair


def _whole_spread_geometry(source, record, spread, cfg, page_detection=None, detect_page_quads_fn=detect_page_quads):
    """Resolve one upright spread crop from manual or per-page outer corners."""
    upright = rotate_image(source, cfg.rotation)
    override = spread.get("roi_overrides", {}).get(str(record["id"]))
    reference = rotate_roi(override or record["roi"], cfg.rotation).tolist()
    if override is not None:
        return upright, reference, {
            "status": "manual",
            "candidate_id": record["id"],
            "roi": reference,
        }

    crop = {
        "status": "fallback",
        "candidate_id": record["id"],
        "roi": reference,
        "confidence": 0.0,
    }
    if not cfg.refine_quad:
        crop["status"] = "reference"
        return upright, reference, crop

    detection_image = upright
    if upright.shape[1] > cfg.analysis_width:
        scale = cfg.analysis_width / upright.shape[1]
        detection_image = cv2.resize(
            upright,
            (cfg.analysis_width, max(2, round(upright.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if (record.get("boundary_refinement") or {}).get("refined"):
        boundary_roi = reference
        boundary_info = {"refined": True, "status": "candidate_refined"}
    else:
        boundary_roi, boundary_info = refine_spread_boundary(detection_image, reference)
    crop["boundary_refinement"] = boundary_info
    if boundary_info["refined"]:
        crop["roi"] = boundary_roi
        crop["status"] = "auto_boundary"
    detection = page_detection
    if detection is None:
        detection = detect_page_quads_fn(
            detection_image,
            crop["roi"],
            spine_ratio=spread.get("spine_ratio", cfg.spine_ratio),
            min_confidence=cfg.page_contour_min_confidence,
        )
    crop["detection"] = detection
    crop["confidence"] = detection["confidence"]
    if detection["detected"]:
        try:
            crop["roi"] = spread_quad_from_page_quads(detection)
        except ValueError:
            crop["status"] = "fallback"
        else:
            crop["status"] = "auto_pages"
    return upright, crop["roi"], crop
