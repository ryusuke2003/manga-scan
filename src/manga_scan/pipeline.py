import csv
import hashlib
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .background_fill import detected_spread_mask, fill_page_background
from .config import Config
from .dedupe import compare
from .export import contact_sheets, export_cbz, export_pdf
from .final_quality import FINAL_QUALITY_REASONS, adjacent_quality_check, final_quality_checks
from .finger_repair import repair_finger_regions
from .glare import detect_glare_mask, glare_overlap_fraction
from .hand import HandDetector, boundary_finger_mask, temporal_transient_mask
from .motion import Sample, StableDetector, choose_candidates, motion_score
from .page_contour import (
    consensus_page_quads,
    detect_page_quads,
    draw_page_quads,
    spread_quad_from_page_quads,
)
from .page_detect import refine_quad
from .page_warp import warp_detected_pages
from .perspective import pixel_quad, rotate_roi, validate_roi, warp_roi
from .score import score_frame, sharpness, suspect_reasons
from .selection import choose_candidate_selection, score_candidate_pages
from .split import (
    auto_dewarp_page,
    dewarp_debug_grid,
    enhance_page,
    rotate_image,
    spine_position,
    split_spread,
)
from .storage import project_lock, read_manifest, save_image, save_manifest, write_json
from .video import extract_frame, sample_frames

LOG = logging.getLogger("manga_scan")

PAGE_CONTOUR_CONSENSUS_MAX_CORNER_DEVIATION = 0.04


def update(project, manifest, progress, message):
    manifest.update(progress=round(progress, 3), message=message)
    save_manifest(project, manifest)
    LOG.info("%3.0f%% %s", progress * 100, message)


def candidate(project, manifest, cfg, detector, spread_id, number, sample):
    # Only candidate timestamps seek back to the original video.
    image = extract_frame(manifest["source"], sample.time, cfg.analysis_width, cfg.hwaccel)
    roi, quad_ok = manifest["roi"], True
    if cfg.refine_quad:
        roi, quad_ok = refine_quad(image, roi, cfg.quad_max_shift)
    overlap, mask = detector.detect(image, roi)
    glare_mask = detect_glare_mask(image, roi)
    glare_overlap = glare_overlap_fraction(glare_mask, roi)
    metrics = score_frame(
        image,
        roi,
        sample.motion,
        overlap,
        cfg,
        glare_overlap=glare_overlap,
    )
    base = f"candidates/{spread_id}/candidate_{number:02d}"
    save_image(project / f"{base}.png", image)
    save_image(project / f"{base}_hand_mask.png", mask)
    save_image(project / f"{base}_glare_mask.png", glare_mask)
    rectified = warp_roi(image, roi)
    preview = f"{base}_spread.png"
    save_image(project / preview, rectified)
    review_preview = preview
    if cfg.rotation:
        review_preview = f"{base}_spread_review.png"
        save_image(project / review_preview, rotate_image(rectified, cfg.rotation))

    if cfg.candidate_selection_mode == "per_page":
        rectified_mask = (
            warp_roi(mask, roi, interpolation=cv2.INTER_NEAREST)
            if overlap is not None
            else np.zeros(rectified.shape[:2], np.uint8)
        )
        rectified_glare_mask = warp_roi(
            glare_mask,
            roi,
            interpolation=cv2.INTER_NEAREST,
        )
        page_metrics, _ = score_candidate_pages(
            rectified,
            rectified_mask,
            sample.motion,
            cfg,
            metrics,
            hand_enabled=overlap is not None,
            glare_mask=rectified_glare_mask,
        )
        page_suspect = {
            side: suspect_reasons(page_metrics[side], cfg, quad_ok)
            for side in ("left", "right")
        }
    else:
        page_metrics = {side: metrics.copy() for side in ("left", "right")}
        page_suspect = {
            side: suspect_reasons(metrics, cfg, quad_ok) for side in ("left", "right")
        }

    record = {
        "id": number,
        "time": sample.time,
        "path": f"{base}.png",
        "preview": preview,
        "review_preview": review_preview,
        "hand_mask": f"{base}_hand_mask.png",
        "glare_mask": f"{base}_glare_mask.png",
        "roi": roi,
        "metrics": metrics,
        "page_metrics": page_metrics,
        "page_suspect": page_suspect,
        "suspect": suspect_reasons(metrics, cfg, quad_ok),
    }
    write_json(project / f"{base}.json", record)
    return record


def _augment_temporal_hand_masks(project, records, cfg):
    """Supplement MediaPipe masks from transient same-spread candidate content."""
    if cfg.hand_backend != "mediapipe" or len(records) < 4:
        return records

    loaded = {}
    for record in records:
        image = cv2.imread(str(project / record["path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(project / record["hand_mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        loaded[record["id"]] = (image, mask)

    if len(loaded) < 4:
        return records

    for record in records:
        target = loaded.get(record["id"])
        if target is None:
            continue
        image, base_mask = target
        peers = [
            {"image": peer_image, "mask": peer_mask}
            for candidate_id, (peer_image, peer_mask) in loaded.items()
            if candidate_id != record["id"]
        ]
        temporal = temporal_transient_mask(
            image,
            record["roi"],
            peers,
            target_mask=base_mask,
            padding=cfg.hand_padding,
        )
        if not np.any(temporal):
            continue

        combined = cv2.bitwise_or(base_mask, temporal)
        mask_path = Path(record["hand_mask"])
        temporal_name = (
            mask_path.stem.replace("_hand_mask", "_temporal_hand_mask") + ".png"
        )
        temporal_path = str(mask_path.with_name(temporal_name))
        save_image(project / temporal_path, temporal)
        save_image(project / record["hand_mask"], combined)

        page = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(
            page,
            np.rint(pixel_quad(record["roi"], image.shape)).astype(np.int32),
            255,
        )
        page_pixels = page > 0
        overlap = float(
            np.count_nonzero((combined > 0) & page_pixels)
            / max(1, np.count_nonzero(page_pixels))
        )
        temporal_overlap = float(
            np.count_nonzero((temporal > 0) & page_pixels)
            / max(1, np.count_nonzero(page_pixels))
        )
        record["temporal_hand_mask"] = temporal_path
        record["temporal_hand_overlap"] = temporal_overlap

        metrics = score_frame(
            image,
            record["roi"],
            record["metrics"]["motion"],
            overlap,
            cfg,
            glare_overlap=record["metrics"].get("glare_overlap", 0.0),
        )
        quad_ok = "page_quad_uncertain" not in record.get("suspect", [])
        record["metrics"] = metrics

        if cfg.candidate_selection_mode == "per_page":
            rectified = warp_roi(image, record["roi"])
            rectified_mask = warp_roi(
                combined,
                record["roi"],
                interpolation=cv2.INTER_NEAREST,
            )
            glare_mask = None
            glare_path = record.get("glare_mask")
            if glare_path:
                glare_mask = cv2.imread(
                    str(project / glare_path),
                    cv2.IMREAD_GRAYSCALE,
                )
            rectified_glare_mask = (
                warp_roi(
                    glare_mask,
                    record["roi"],
                    interpolation=cv2.INTER_NEAREST,
                )
                if glare_mask is not None
                else None
            )
            page_metrics, _ = score_candidate_pages(
                rectified,
                rectified_mask,
                metrics["motion"],
                cfg,
                metrics,
                hand_enabled=True,
                glare_mask=rectified_glare_mask,
            )
            record["page_metrics"] = page_metrics
            record["page_suspect"] = {
                side: suspect_reasons(page_metrics[side], cfg, quad_ok)
                for side in ("left", "right")
            }
        else:
            record["page_metrics"] = {
                side: metrics.copy() for side in ("left", "right")
            }
            record["page_suspect"] = {
                side: suspect_reasons(metrics, cfg, quad_ok)
                for side in ("left", "right")
            }

        record["suspect"] = suspect_reasons(metrics, cfg, quad_ok)
        write_json(project / Path(record["path"]).with_suffix(".json"), record)
    return records


def _candidate_by_id(spread, candidate_id):
    return next(candidate for candidate in spread["candidates"] if candidate["id"] == candidate_id)


def _selected_candidate_id(spread, side):
    return (spread.get("selected_pages") or {}).get(side, spread["selected"])


def _join_physical_pages(physical_pages):
    height = min(page.shape[0] for page in physical_pages)
    resized = []
    for page in physical_pages:
        width = max(1, round(page.shape[1] * height / page.shape[0]))
        resized.append(cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA))
    width = min(page.shape[1] for page in resized)
    normalized = [
        page
        if page.shape[1] == width
        else cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA)
        for page in resized
    ]
    return np.concatenate(normalized, axis=1)


def selected_spread_preview(project, spread, cfg):
    selected = [_selected_candidate_id(spread, side) for side in ("left", "right")]
    if spread.get("output_layout", cfg.output_layout) == "spread":
        selected = [spread["selected"]] * 2
    if selected[0] == selected[1]:
        candidate_record = _candidate_by_id(spread, selected[0])
        preview = cv2.imread(str(project / candidate_record["preview"]))
        if preview is None:
            raise ValueError(f"Candidate preview missing: {candidate_record['preview']}")
        return rotate_image(preview, cfg.rotation)

    physical_pages = []
    for side, candidate_id in zip(("left", "right"), selected):
        candidate_record = _candidate_by_id(spread, candidate_id)
        rectified = cv2.imread(str(project / candidate_record["preview"]))
        if rectified is None:
            raise ValueError(f"Candidate preview missing: {candidate_record['preview']}")
        rectified = rotate_image(rectified, cfg.rotation)
        sides, _ = split_spread(
            rectified,
            spread.get("spine_ratio", cfg.spine_ratio),
            cfg.split_mode,
            cfg.gutter_fraction,
        )
        physical_pages.append(sides[side])
    return _join_physical_pages(physical_pages)


def render_cover(project, manifest):
    cover = manifest.get("cover") or {}
    if cover.get("status") != "ready":
        return None
    cfg = Config.from_dict(manifest["config"])
    image = extract_frame(manifest["source"], cover["time"], hwaccel=cfg.hwaccel)
    rectified = warp_roi(image, cover["roi"])
    manual_dewarp = cfg.dewarp_strength if cfg.dewarp_mode == "manual" else 0.0
    page_image = enhance_page(
        rectified,
        grayscale=cfg.grayscale,
        contrast=cfg.contrast,
        rotation=cfg.rotation,
        dewarp_strength=manual_dewarp,
        white_normalization=cfg.white_normalization,
        white_target=cfg.white_target,
        white_strength=cfg.white_strength,
        illumination_correction=cfg.illumination_correction,
        illumination_strength=cfg.illumination_strength,
    )
    final_quality = final_quality_checks(
        page_image,
        before_enhance=rotate_image(rectified, cfg.rotation),
        dewarp={
            "mode": "manual" if manual_dewarp else "off",
            "applied": bool(manual_dewarp),
            "strength": manual_dewarp,
        },
        white_normalization=cfg.white_normalization,
    )
    ext = "png" if cfg.image_format == "png" else "jpg"
    path = f"pages/cover.{ext}"
    save_image(project / path, page_image, cfg.jpeg_quality)
    h, w = page_image.shape[:2]
    thumb = cv2.resize(page_image, (max(1, round(w * min(1, 480 / h))), min(480, h)))
    preview = "pages/cover_thumb.jpg"
    save_image(project / preview, thumb)
    cover["path"] = path
    cover["preview"] = preview
    return {
        "id": "cover",
        "spread_id": "cover",
        "side": "cover",
        "path": path,
        "preview": preview,
        "enabled": True,
        "suspect": final_quality["reasons"],
        "final_quality": final_quality,
    }


def detect_spread_page_consensus(project, spread, cfg, anchor_ids=None):
    """Estimate page quads from all saved candidates for one spread.

    Candidate preview frames are already analysis-resolution images, so this
    adds no extra video seeks. Each frame is evaluated independently and the
    page_contour module rejects geometric outliers before combining corners.
    """

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
        detection = detect_page_quads(
            upright,
            roi,
            spine_ratio=spread.get("spine_ratio", cfg.spine_ratio),
            min_confidence=cfg.page_contour_min_confidence,
        )
        detection["candidate_id"] = record["id"]
        detections.append(detection)

    if not detections:
        return None
    return consensus_page_quads(
        detections,
        min_confidence=cfg.page_contour_min_confidence,
        max_corner_deviation=PAGE_CONTOUR_CONSENSUS_MAX_CORNER_DEVIATION,
        anchor_ids=anchor_ids,
    )


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
            detection = detect_page_quads(
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


def candidate_page_hand_mask(project, data, side, cfg):
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
        extra = boundary_finger_mask(page, [[0, 0], [1, 0], [1, 1], [0, 1]], cfg.hand_padding)
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


def _whole_spread_geometry(source, record, spread, cfg, page_detection=None):
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
    detection = page_detection
    if detection is None:
        detection = detect_page_quads(
            detection_image,
            reference,
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


def _render_whole_spread(project, manifest, spread, cfg):
    """Render one spread, using both page outlines without cutting the gutter."""
    for key in (
        "page_contours_by_side",
        "page_contour_debug_by_side",
        "perspective_mode_used_by_side",
        "spine_px_by_side",
    ):
        spread.pop(key, None)
    selected_id = spread["selected"]
    page_consensus = (
        detect_spread_page_consensus(
            project,
            spread,
            cfg,
            {"left": selected_id, "right": selected_id},
        )
        if cfg.refine_quad
        else None
    )
    cache = {}

    def load(candidate_id):
        if candidate_id in cache:
            return cache[candidate_id]
        record = _candidate_by_id(spread, candidate_id)
        source = extract_frame(manifest["source"], record["time"], hwaccel=cfg.hwaccel)
        upright, roi, crop = _whole_spread_geometry(
            source,
            record,
            spread,
            cfg,
            page_detection=page_consensus if candidate_id == selected_id else None,
        )
        page = warp_roi(upright, roi)
        background_mask = None
        detection = crop.get("detection")
        if crop.get("status") == "auto_pages" and detection and detection.get("detected"):
            detected_mask = detected_spread_mask(upright.shape, detection)
            background_mask = warp_roi(
                detected_mask,
                roi,
                interpolation=cv2.INTER_NEAREST,
            )
        hand_mask = None
        if cfg.hand_backend == "mediapipe" and record.get("hand_mask"):
            saved = cv2.imread(str(project / record["hand_mask"]), cv2.IMREAD_GRAYSCALE)
            if saved is not None:
                saved = cv2.resize(
                    saved, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                upright_mask = rotate_image(saved, cfg.rotation)
                hand_mask = warp_roi(
                    upright_mask, roi, interpolation=cv2.INTER_NEAREST
                )
                hand_mask |= boundary_finger_mask(
                    page, [[0, 0], [1, 0], [1, 1], [0, 1]], cfg.hand_padding
                )
        glare_mask = detect_glare_mask(page) if cfg.glare_repair else None
        repair_hand_mask = hand_mask if cfg.finger_repair else None
        occlusion_mask = _union_occlusion_masks(repair_hand_mask, glare_mask)
        cache[candidate_id] = {
            "page": page,
            "mask": occlusion_mask,
            "hand_mask": hand_mask,
            "glare_mask": glare_mask,
            "crop": crop,
            "upright": upright,
            "background_mask": background_mask,
        }
        return cache[candidate_id]

    chosen = _candidate_by_id(spread, spread["selected"])
    selected = load(chosen["id"])
    page, mask = selected["page"], selected["mask"]
    hand_mask = selected.get("hand_mask")
    glare_mask = selected.get("glare_mask")
    crop = selected["crop"]
    spread["whole_spread_crop"] = {key: value for key, value in crop.items() if key != "detection"}
    detection = crop.get("detection")
    if detection is not None:
        spread["page_contours"] = detection
        debug_path = f"debug/page_contours/{spread['id']}_whole.jpg"
        save_image(project / debug_path, draw_page_quads(selected["upright"], detection))
        spread["page_contour_debug"] = debug_path
    else:
        spread.pop("page_contours", None)
        spread.pop("page_contour_debug", None)
    source_path = f"selected/{spread['id']}.png"
    save_image(project / source_path, page)
    spread["path"] = source_path
    spread["perspective_mode_used"] = f"spread_{crop['status']}"
    repair = {"status": "disabled", "coverage": 0.0, "donors": []}
    if cfg.finger_repair or cfg.glare_repair:
        if mask is None:
            repair = {"status": "unavailable", "coverage": 0.0, "donors": []}
        elif not np.any(mask > 127):
            repair = {
                "status": "clean" if cfg.finger_repair else "disabled",
                "coverage": 1.0 if cfg.finger_repair else 0.0,
                "donors": [],
                "occlusion_kinds": [],
            }
        else:

            def donor_rank(record):
                metrics = record.get("metrics", {})
                overlap = metrics.get("hand_overlap")
                glare = float(metrics.get("glare_overlap", metrics.get("glare", 0.0)) or 0.0)
                return (
                    1.0 if overlap is None else float(overlap),
                    glare,
                    -float(metrics.get("selection_score", metrics.get("score", 0.0))),
                )

            def donors():
                records = sorted(spread["candidates"], key=donor_rank)
                for record in records:
                    if record["id"] == chosen["id"]:
                        continue
                    donor = load(record["id"])
                    if donor["mask"] is not None:
                        yield {
                            "candidate_id": record["id"],
                            "image": donor["page"],
                            "mask": donor["mask"],
                        }

            # repair_finger_regions remains the compatibility name; internally
            # it delegates to the generalized occlusion repair engine.
            page, repair, unresolved = repair_finger_regions(
                page,
                mask,
                donors(),
                min_coverage=cfg.finger_repair_min_coverage,
                fallback=cfg.finger_repair_fallback,
            )
            repair["occlusion_kinds"] = [
                kind
                for kind, kind_mask in (
                    ("finger", hand_mask if cfg.finger_repair else None),
                    ("glare", glare_mask),
                )
                if kind_mask is not None and np.any(kind_mask > 127)
            ]
            if np.any(mask):
                repair["target_mask"] = f"debug/finger_repair/{spread['id']}_whole_target.png"
                save_image(project / repair["target_mask"], mask)
            if glare_mask is not None and np.any(glare_mask > 127):
                repair["glare_mask"] = f"debug/finger_repair/{spread['id']}_whole_glare.png"
                save_image(project / repair["glare_mask"], glare_mask)
            if np.any(unresolved):
                repair["unresolved_mask"] = (
                    f"debug/finger_repair/{spread['id']}_whole_unresolved.png"
                )
                save_image(project / repair["unresolved_mask"], unresolved)
            repair = _persist_finger_repair_component_debug(
                project,
                repair,
                f"{spread['id']}_whole",
            )

    background_fill = {
        "mode": cfg.page_background_fill,
        "status": "preserve" if cfg.page_background_fill == "preserve" else "unavailable",
        "applied": False,
        "filled_fraction": 0.0,
        "fill_color": None,
    }
    background_mask = selected.get("background_mask")
    if cfg.page_background_fill != "preserve" and background_mask is not None:
        page, background_fill = fill_page_background(
            page,
            background_mask,
            mode=cfg.page_background_fill,
            paper_target=cfg.white_target,
        )
        background_fill["status"] = "applied" if background_fill["applied"] else "not_needed"
        background_fill["confidence"] = crop.get("confidence", 0.0)
        background_fill["mask"] = f"debug/background_fill/{spread['id']}_page_mask.png"
        save_image(project / background_fill["mask"], background_mask)

    # Single-page spine dewarping would distort the middle of a full spread.
    render_settings = _page_render_settings(spread, "spread", cfg)
    qa_before_enhance = page.copy()
    page = enhance_page(
        page,
        grayscale=cfg.grayscale,
        contrast=cfg.contrast,
        illumination_correction=render_settings["illumination_correction"],
        illumination_strength=cfg.illumination_strength,
        white_normalization=render_settings["white_normalization"],
        white_target=cfg.white_target,
        white_strength=cfg.white_strength,
    )
    background_fill_area = None
    if background_fill.get("applied") and background_mask is not None:
        background_fill_area = float(np.mean(background_mask <= 127))
    final_quality = final_quality_checks(
        page,
        before_enhance=qa_before_enhance,
        finger_repair=repair,
        background_fill_fraction=background_fill_area,
        white_normalization=render_settings["white_normalization"],
    )
    ext = "png" if cfg.image_format == "png" else "jpg"
    path = f"pages/{spread['id']}_whole.{ext}"
    preview = f"pages/{spread['id']}_whole_thumb.jpg"
    save_image(project / path, page, cfg.jpeg_quality)
    h, w = page.shape[:2]
    scale = min(1, 720 / max(h, w))
    save_image(
        project / preview, cv2.resize(page, (max(1, round(w * scale)), max(1, round(h * scale))))
    )
    suspect = list(
        dict.fromkeys(
            chosen.get("suspect", [])
            + [
                reason
                for reason in spread.get("extra_suspect", [])
                if reason != "page_contour_low_confidence"
            ]
        )
    )
    if crop["status"] == "fallback":
        suspect.append("page_contour_low_confidence")
    if detection and any(detection[side]["touches_frame"] for side in ("left", "right")):
        suspect.append("source_frame_clipped")
    if hand_mask is not None:
        suspect = [reason for reason in suspect if reason != "hand_overlap"]
        final_hand_overlap = float(np.mean(hand_mask > 127))
        if final_hand_overlap >= cfg.suspect_hand_overlap:
            suspect.append("hand_overlap")
    if glare_mask is not None:
        suspect = [reason for reason in suspect if reason != "glare_overlap"]
        final_glare_overlap = float(np.mean(glare_mask > 127))
        if final_glare_overlap >= cfg.suspect_glare_overlap:
            suspect.append("glare_overlap")
    if repair["status"] in ("clean", "complete"):
        if cfg.finger_repair:
            suspect = [reason for reason in suspect if reason != "hand_overlap"]
        if cfg.glare_repair:
            suspect = [reason for reason in suspect if reason != "glare_overlap"]
    elif repair["status"] in ("incomplete", "unavailable"):
        suspect.append("occlusion_repair_incomplete")
        if cfg.finger_repair:
            suspect.append("finger_repair_incomplete")
    suspect = list(dict.fromkeys(suspect + final_quality["reasons"]))
    spread["suspect"] = suspect
    manifest["pdf_stale"] = True
    return [
        {
            "id": f"{spread['id']}_whole",
            "spread_id": spread["id"],
            "side": "spread",
            "path": path,
            "preview": preview,
            "source": source_path,
            "candidate_id": chosen["id"],
            "candidate_time": chosen["time"],
            "enabled": not bool(spread.get("duplicate_of")),
            "suspect": suspect,
            "finger_repair": repair,
            "background_fill": background_fill,
            "final_quality": final_quality,
            "crop": spread["whole_spread_crop"],
            "dewarp": {"mode": "off", "status": "off", "applied": False},
            "render_settings": {
                **render_settings,
                "dewarp": False,
                "dewarp_mode": "off",
                "page_quad_mode": "auto",
                "manual_quad": None,
            },
        }
    ]


def render_spread(project, manifest, spread):
    cfg = Config.from_dict(manifest["config"])
    if spread.get("output_layout", cfg.output_layout) == "spread":
        return _render_whole_spread(project, manifest, spread, cfg)
    spread.pop("whole_spread_crop", None)
    selected_pages = spread.get("selected_pages") or {
        "left": spread["selected"],
        "right": spread["selected"],
    }
    spread["selected_pages"] = selected_pages
    same_candidate = selected_pages["left"] == selected_pages["right"]
    page_consensus = (
        detect_spread_page_consensus(project, spread, cfg, selected_pages)
        if cfg.perspective_mode == "per_page"
        else None
    )
    base_extra_suspect = [
        reason
        for reason in spread.get("extra_suspect", [])
        if reason != "page_contour_low_confidence"
    ]
    spread["extra_suspect"] = base_extra_suspect.copy()
    cache = {}

    def load_candidate(candidate_id):
        if candidate_id in cache:
            return cache[candidate_id]
        chosen = _candidate_by_id(spread, candidate_id)
        override = spread.get("roi_overrides", {}).get(str(candidate_id))
        if override is not None:
            chosen = {**chosen, "roi": override}
        source_image = extract_frame(manifest["source"], chosen["time"], hwaccel=cfg.hwaccel)
        rectified = rotate_image(warp_roi(source_image, chosen["roi"]), cfg.rotation)
        image = rotate_image(source_image, cfg.rotation)
        roi = rotate_roi(chosen["roi"], cfg.rotation)
        use_spread_state = (
            same_candidate and candidate_id == selected_pages["left"]
        )
        state = (
            spread
            if use_spread_state
            else {
                "id": f"{spread['id']}_candidate_{candidate_id:02d}",
                "spine_ratio": spread.get("spine_ratio", cfg.spine_ratio),
                "extra_suspect": [],
                "page_overrides": spread.get("page_overrides", {}),
            }
        )
        state["manual_roi"] = override is not None
        consensus_sides = [
            side
            for side in ("left", "right")
            if selected_pages[side] == candidate_id
        ]
        manual_override_sides = [
            side
            for side in consensus_sides
            if _page_override(spread, side).get("page_quad_mode") == "manual"
        ]
        if page_consensus is not None and consensus_sides:
            if manual_override_sides:
                sides = rectify_spread_pages(
                    project,
                    image,
                    rectified,
                    roi,
                    state,
                    cfg,
                    page_detection=page_consensus,
                    consensus_sides=consensus_sides,
                    manual_sides=manual_override_sides,
                )
            else:
                sides = rectify_spread_pages(
                    project,
                    image,
                    rectified,
                    roi,
                    state,
                    cfg,
                    page_detection=page_consensus,
                    consensus_sides=consensus_sides,
                )
        elif manual_override_sides:
            sides = rectify_spread_pages(
                project,
                image,
                rectified,
                roi,
                state,
                cfg,
                manual_sides=manual_override_sides,
            )
        else:
            sides = rectify_spread_pages(project, image, rectified, roi, state, cfg)
        cache[candidate_id] = {
            "chosen": chosen,
            "rectified": rectified,
            "sides": sides,
            "state": state,
            "source_frame_shape": source_image.shape,
        }
        return cache[candidate_id]

    selected_data = {
        side: load_candidate(selected_pages[side]) for side in ("left", "right")
    }

    if same_candidate:
        spread.pop("perspective_mode_used_by_side", None)
        spread.pop("spine_px_by_side", None)
        spread.pop("page_contours_by_side", None)
        spread.pop("page_contour_debug_by_side", None)
        selected_image = selected_data["left"]["rectified"]
    else:
        spread.pop("page_contours", None)
        spread.pop("page_contour_debug", None)
        spread["perspective_mode_used_by_side"] = {
            side: selected_data[side]["state"].get("perspective_mode_used", "spread")
            for side in ("left", "right")
        }
        modes = set(spread["perspective_mode_used_by_side"].values())
        spread["perspective_mode_used"] = modes.pop() if len(modes) == 1 else "mixed"
        spread["spine_px_by_side"] = {
            side: selected_data[side]["state"].get("spine_px")
            for side in ("left", "right")
        }
        spread["spine_px"] = spread["spine_px_by_side"]["left"]
        contours = {
            side: selected_data[side]["state"].get("page_contours")
            for side in ("left", "right")
            if selected_data[side]["state"].get("page_contours") is not None
        }
        if contours:
            spread["page_contours_by_side"] = contours
        else:
            spread.pop("page_contours_by_side", None)
        debug_paths = {
            side: selected_data[side]["state"].get("page_contour_debug")
            for side in ("left", "right")
            if selected_data[side]["state"].get("page_contour_debug")
        }
        if debug_paths:
            spread["page_contour_debug_by_side"] = debug_paths
        else:
            spread.pop("page_contour_debug_by_side", None)
        selected_image = _join_physical_pages(
            [
                selected_data["left"]["sides"]["left"],
                selected_data["right"]["sides"]["right"],
            ]
        )

    selected_path = f"selected/{spread['id']}.png"
    save_image(project / selected_path, selected_image)
    spread["path"] = selected_path

    selected_suspect = []
    for side in ("left", "right"):
        data = selected_data[side]
        chosen = data["chosen"]
        selected_suspect.extend(
            chosen.get("page_suspect", {}).get(side, chosen.get("suspect", []))
        )
        selected_suspect.extend(data["state"].get("extra_suspect", []))
    spread["extra_suspect"] = list(
        dict.fromkeys(
            base_extra_suspect
            + [
                reason
                for side in ("left", "right")
                for reason in selected_data[side]["state"].get("extra_suspect", [])
            ]
        )
    )
    spread["suspect"] = list(dict.fromkeys(selected_suspect + base_extra_suspect))

    pages = []
    order = ["right", "left"] if cfg.reading_order == "rtl" else ["left", "right"]
    ext = "png" if cfg.image_format == "png" else "jpg"
    for side in order:
        data = selected_data[side]
        render_settings = _page_render_settings(spread, side, cfg)
        chosen = data["chosen"]
        source_page = data["sides"][side]
        selected_source = f"selected/{spread['id']}_{side}.png"
        save_image(project / selected_source, source_page)

        finger_repair = {"status": "disabled", "coverage": 0.0, "donors": []}
        if cfg.finger_repair or cfg.glare_repair:
            target_hand_mask = (
                candidate_page_hand_mask(project, data, side, cfg)
                if cfg.finger_repair
                else None
            )
            target_glare_mask = candidate_page_glare_mask(data, side, cfg)
            target_mask = _union_occlusion_masks(target_hand_mask, target_glare_mask)
            if target_mask is None:
                finger_repair = {
                    "status": "unavailable",
                    "coverage": 0.0,
                    "donors": [],
                }
            elif np.any(target_mask > 127):
                target_mask_path = f"debug/finger_repair/{spread['id']}_{side}_target.png"
                save_image(project / target_mask_path, target_mask)

                def donor_pages():
                    for donor_record in _finger_donor_candidates(
                        spread,
                        side,
                        selected_pages[side],
                    )[:5]:
                        donor_data = load_candidate(donor_record["id"])
                        donor_hand_mask = (
                            candidate_page_hand_mask(project, donor_data, side, cfg)
                            if cfg.finger_repair
                            else None
                        )
                        donor_glare_mask = candidate_page_glare_mask(
                            donor_data, side, cfg
                        )
                        donor_mask = _union_occlusion_masks(
                            donor_hand_mask,
                            donor_glare_mask,
                        )
                        if donor_mask is None:
                            continue
                        yield {
                            "candidate_id": donor_record["id"],
                            "image": donor_data["sides"][side],
                            "mask": donor_mask,
                        }

                source_page, finger_repair, unresolved = repair_finger_regions(
                    source_page,
                    target_mask,
                    donor_pages(),
                    min_coverage=cfg.finger_repair_min_coverage,
                    fallback=cfg.finger_repair_fallback,
                )
                finger_repair["occlusion_kinds"] = [
                    kind
                    for kind, kind_mask in (
                        ("finger", target_hand_mask),
                        ("glare", target_glare_mask),
                    )
                    if kind_mask is not None and np.any(kind_mask > 127)
                ]
                finger_repair["target_mask"] = target_mask_path
                if target_glare_mask is not None and np.any(target_glare_mask > 127):
                    glare_path = f"debug/finger_repair/{spread['id']}_{side}_glare.png"
                    save_image(project / glare_path, target_glare_mask)
                    finger_repair["glare_mask"] = glare_path
                if np.any(unresolved):
                    unresolved_path = (
                        f"debug/finger_repair/{spread['id']}_{side}_unresolved.png"
                    )
                    save_image(project / unresolved_path, unresolved)
                    finger_repair["unresolved_mask"] = unresolved_path
                finger_repair = _persist_finger_repair_component_debug(
                    project,
                    finger_repair,
                    f"{spread['id']}_{side}",
                )
            else:
                finger_repair = {
                    "status": "clean" if cfg.finger_repair else "disabled",
                    "coverage": 1.0 if cfg.finger_repair else 0.0,
                    "donors": [],
                    "occlusion_kinds": [],
                }

        dewarp_mode = render_settings["dewarp_mode"]
        dewarp = {"mode": dewarp_mode, "applied": False, "status": "off"}
        manual_dewarp = 0.0
        qa_before_dewarp = None
        if dewarp_mode == "manual":
            manual_dewarp = cfg.dewarp_strength
            dewarp.update(
                applied=bool(manual_dewarp),
                status="applied" if manual_dewarp else "off",
                strength=manual_dewarp,
            )
            if manual_dewarp:
                qa_before_dewarp = enhance_page(
                    source_page,
                    grayscale=cfg.grayscale,
                    contrast=cfg.contrast,
                    rotation=0,
                    dewarp_strength=0.0,
                    white_normalization=render_settings["white_normalization"],
                    white_target=cfg.white_target,
                    white_strength=cfg.white_strength,
                    illumination_correction=render_settings["illumination_correction"],
                    illumination_strength=cfg.illumination_strength,
                )
        elif dewarp_mode == "auto":
            before = f"debug/dewarp/{spread['id']}_{side}_before.png"
            before_image = enhance_page(
                source_page,
                grayscale=cfg.grayscale,
                contrast=cfg.contrast,
                rotation=0,
                dewarp_strength=0.0,
                white_normalization=render_settings["white_normalization"],
                white_target=cfg.white_target,
                white_strength=cfg.white_strength,
                illumination_correction=render_settings["illumination_correction"],
                illumination_strength=cfg.illumination_strength,
            )
            save_image(project / before, before_image)
            qa_before_dewarp = before_image
            corrected, estimate = auto_dewarp_page(
                source_page,
                side,
                cfg.dewarp_max_strength,
                cfg.dewarp_min_confidence,
            )
            source_page = corrected
            dewarp.update(estimate)
            dewarp["before"] = before
            if estimate["strength"] > 0:
                grid = f"debug/dewarp/{spread['id']}_{side}_remap.png"
                save_image(
                    project / grid,
                    dewarp_debug_grid(
                        data["sides"][side].shape,
                        side,
                        estimate["strength"],
                        estimate.get("strength_profile"),
                    ),
                )
                dewarp["debug_grid"] = grid
        else:
            dewarp.update(status="disabled")

        qa_before_enhance = source_page.copy()
        page_image = enhance_page(
            source_page,
            grayscale=cfg.grayscale,
            contrast=cfg.contrast,
            rotation=0,
            dewarp_strength=manual_dewarp,
            white_normalization=render_settings["white_normalization"],
            white_target=cfg.white_target,
            white_strength=cfg.white_strength,
            illumination_correction=render_settings["illumination_correction"],
            illumination_strength=cfg.illumination_strength,
        )
        final_quality = final_quality_checks(
            page_image,
            before_enhance=qa_before_enhance,
            before_dewarp=qa_before_dewarp,
            dewarp=dewarp,
            finger_repair=finger_repair,
            white_normalization=render_settings["white_normalization"],
        )
        name = f"pages/{spread['id']}_{side}.{ext}"
        save_image(project / name, page_image, cfg.jpeg_quality)
        h, w = page_image.shape[:2]
        thumb = cv2.resize(page_image, (max(1, round(w * min(1, 480 / h))), min(480, h)))
        preview = f"pages/{spread['id']}_{side}_thumb.jpg"
        save_image(project / preview, thumb)

        page_suspect = list(
            dict.fromkeys(
                chosen.get("page_suspect", {}).get(side, chosen.get("suspect", []))
                + base_extra_suspect
                + data["state"].get("extra_suspect", [])
            )
        )
        if dewarp.get("status") == "low_confidence":
            page_suspect.append("dewarp_low_confidence")
        contours = data["state"].get("page_contours") or {}
        if contours.get(side, {}).get("touches_frame"):
            page_suspect.append("source_frame_clipped")
        if finger_repair["status"] in ("complete", "clean"):
            if cfg.finger_repair:
                page_suspect = [reason for reason in page_suspect if reason != "hand_overlap"]
            if cfg.glare_repair:
                page_suspect = [reason for reason in page_suspect if reason != "glare_overlap"]
        elif finger_repair["status"] in ("incomplete", "unavailable"):
            page_suspect.append("occlusion_repair_incomplete")
            if cfg.finger_repair:
                page_suspect.append("finger_repair_incomplete")
        page_suspect.extend(final_quality["reasons"])
        pages.append(
            {
                "id": f"{spread['id']}_{side}",
                "spread_id": spread["id"],
                "side": side,
                "path": name,
                "preview": preview,
                "source": selected_source,
                "candidate_id": selected_pages[side],
                "candidate_time": chosen["time"],
                "enabled": not bool(spread.get("duplicate_of")),
                "suspect": list(dict.fromkeys(page_suspect)),
                "finger_repair": finger_repair,
                "dewarp": dewarp,
                "render_settings": render_settings,
                "page_contour": {
                    "mode": render_settings["page_quad_mode"],
                    "quad": contours.get(side, {}).get("quad"),
                    "confidence": contours.get(side, {}).get("confidence"),
                    "detected": contours.get(side, {}).get("detected"),
                    "manual": bool(contours.get(side, {}).get("manual")),
                },
                "final_quality": final_quality,
            }
        )
    if cfg.finger_repair:
        incomplete = any(
            page.get("finger_repair", {}).get("status") in ("incomplete", "unavailable")
            for page in pages
        )
        if incomplete:
            spread["suspect"] = list(
                dict.fromkeys(spread.get("suspect", []) + ["finger_repair_incomplete"])
            )
        else:
            spread["suspect"] = [
                reason for reason in spread.get("suspect", []) if reason != "hand_overlap"
            ]
    manifest["pdf_stale"] = True
    return pages


def _refresh_adjacent_final_quality(project, manifest, cfg):
    """Refresh review-only duplicate warnings for the current enabled page order."""
    reason = "final_duplicate_suspected"
    if reason not in FINAL_QUALITY_REASONS:
        raise RuntimeError("final quality reason registry is incomplete")

    for page in manifest.get("pages", []):
        if "suspect" in page:
            page["suspect"] = [
                item for item in page.get("suspect", []) if item != reason
            ]
        quality = page.get("final_quality")
        if quality:
            quality["reasons"] = [
                item for item in quality.get("reasons", []) if item != reason
            ]
            quality.pop("adjacent_duplicate", None)

    previous = None
    previous_image = None
    for page in (item for item in manifest.get("pages", []) if item.get("enabled")):
        path = page.get("path")
        if not path:
            previous = None
            previous_image = None
            continue
        image = cv2.imread(str(project / path), cv2.IMREAD_COLOR)
        if image is None:
            previous = None
            previous_image = None
            continue
        if previous is not None and previous_image is not None:
            result = adjacent_quality_check(previous_image, image, cfg)
            if result["suspect"]:
                quality = page.setdefault(
                    "final_quality",
                    {"reasons": [], "metrics": {}},
                )
                quality["reasons"] = list(
                    dict.fromkeys(quality.get("reasons", []) + [reason])
                )
                quality["adjacent_duplicate"] = {
                    **result,
                    "other_page_id": previous["id"],
                }
                page["suspect"] = list(
                    dict.fromkeys(page.get("suspect", []) + [reason])
                )
        previous = page
        previous_image = image


def build_exports(project, manifest):
    cfg = Config.from_dict(manifest["config"])
    _refresh_adjacent_final_quality(project, manifest, cfg)
    paths = [project / p["path"] for p in manifest["pages"] if p["enabled"]]
    output = project / "output"
    pdf = output / "manga.pdf"
    cbz = output / "manga.cbz"
    next_pdf = output / "manga.next.pdf"
    next_cbz = output / "manga.next.cbz"
    try:
        export_pdf(
            paths,
            next_pdf,
            cfg.pdf_dpi,
            cfg.image_format,
            cfg.jpeg_quality,
        )
        export_cbz(paths, next_cbz)
        next_pdf.replace(pdf)
        next_cbz.replace(cbz)
    finally:
        next_pdf.unlink(missing_ok=True)
        next_cbz.unlink(missing_ok=True)

    manifest["pdf_stale"] = False
    manifest["pdf"] = "output/manga.pdf"
    manifest["cbz"] = "output/manga.cbz"
    contact_sheets(project, manifest["pages"])
    save_manifest(project, manifest)


def build_pdf(project, manifest):
    """Backward-compatible export entrypoint; now writes both PDF and CBZ."""
    build_exports(project, manifest)


_PAGE_HISTORY_LIMIT = 30


def _page_review_state(manifest):
    return {
        "order": [page["id"] for page in manifest.get("pages", [])],
        "disabled": [
            page["id"]
            for page in manifest.get("pages", [])
            if not page.get("enabled", True)
        ],
    }


def _page_history(manifest):
    history = manifest.setdefault("page_history", {})
    undo = history.setdefault("undo", [])
    redo = history.setdefault("redo", [])
    if not isinstance(undo, list) or not isinstance(redo, list):
        raise ValueError("Invalid page history")
    return history


def _push_page_history(manifest, before, label):
    history = _page_history(manifest)
    history["undo"].append({"label": label, "state": before})
    history["undo"] = history["undo"][-_PAGE_HISTORY_LIMIT:]
    history["redo"] = []


def _clear_page_history(manifest):
    manifest.pop("page_history", None)


def _restore_page_review_state(manifest, state):
    order = list(state.get("order") or [])
    current = {page["id"]: page for page in manifest.get("pages", [])}
    if len(order) != len(current) or set(order) != set(current):
        raise ValueError("Page history is no longer compatible with the current page set")

    disabled = set(state.get("disabled") or [])
    unknown_disabled = disabled - set(current)
    if unknown_disabled:
        raise ValueError("Page history contains unknown pages")

    manifest["pages"] = [current[page_id] for page_id in order]
    for page in manifest["pages"]:
        page["enabled"] = page["id"] not in disabled


def _apply_page_history(manifest, direction):
    history = manifest.get("page_history")
    if not history:
        return False
    if not isinstance(history, dict):
        raise ValueError("Invalid page history")
    undo = history.get("undo", [])
    redo = history.get("redo", [])
    if not isinstance(undo, list) or not isinstance(redo, list):
        raise ValueError("Invalid page history")
    source_name, target_name = (
        ("undo", "redo") if direction == "undo" else ("redo", "undo")
    )
    source = history.get(source_name, [])
    if not source:
        return False
    history.setdefault(target_name, [])

    entry = source.pop()
    current = _page_review_state(manifest)
    _restore_page_review_state(manifest, entry["state"])
    history[target_name].append({"label": entry.get("label", "ページ編集"), "state": current})
    history[target_name] = history[target_name][-_PAGE_HISTORY_LIMIT:]
    return True


def run(project, roi=None):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed. Use review edits or create a new project")
        cfg = Config.from_dict(manifest["config"])
        manifest["roi"] = validate_roi(roi if roi is not None else manifest["roi"]).tolist()
        reference = manifest.get("reference") or {}
        analysis_start = float(reference.get("time", 0)) if reference.get("confirmed") else 0.0
        detector = HandDetector(cfg)  # Fail before expensive analysis if hand support is missing.
        handler = logging.FileHandler(project / "debug/process.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)
        started = time.monotonic()
        try:
            manifest.update(status="processing", spreads=[], pages=[], pdf_stale=True)
            manifest.pop("error", None)
            cover_page = render_cover(project, manifest)
            if cover_page:
                manifest["pages"].append(cover_page)
            update(project, manifest, 0, "低解像度で動きを解析中")
            if cfg.hand_backend == "mediapipe":
                manifest["hand_model_sha256"] = hashlib.sha256(
                    Path(cfg.hand_model).read_bytes()
                ).hexdigest()
            h = manifest["metadata"]["display_height"]
            w = manifest["metadata"]["display_width"]
            width = min(cfg.analysis_width, w)
            size = (width, max(2, round(h * width / w)))
            fps = min(cfg.video_sample_fps, manifest["metadata"]["fps"] or cfg.video_sample_fps)
            manifest["analysis_fps"] = fps
            machine = StableDetector(cfg.stable_frames, cfg.motion_threshold, cfg.turn_threshold)
            segments, previous = [], None
            with (project / "debug/motion.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "time", "motion", "sharpness", "state"])
                for index, timestamp, frame in sample_frames(
                    manifest["source"], fps, size, cfg.hwaccel, start_time=analysis_start
                ):
                    cropped = warp_roi(frame, manifest["roi"])
                    motion = motion_score(previous, cropped) if previous is not None else 1.0
                    sample = Sample(index, timestamp, motion, sharpness(cropped))
                    complete = machine.push(sample)
                    if complete:
                        segments.append(complete)
                    writer.writerow([index, timestamp, motion, sample.sharpness, machine.state])
                    previous = cropped
                    if cfg.save_lowres:
                        save_image(project / f"frames_lowres/{index:08d}.jpg", frame)
                    if index % max(1, round(fps * 2)) == 0:
                        update(
                            project,
                            manifest,
                            min(
                                0.4,
                                0.4
                                * max(0.0, timestamp - analysis_start)
                                / max(0.001, manifest["metadata"]["duration"] - analysis_start),
                            ),
                            f"動き解析 {timestamp:.1f}s / {manifest['metadata']['duration']:.1f}s",
                        )
            tail = machine.finish()
            if tail:
                segments.append(tail)
            if not segments:
                raise ValueError(
                    "No stable intervals found. Hold pages longer, tune motion_threshold/stable_frames, or add frames manually"
                )
            write_json(
                project / "debug/intervals.json",
                [
                    {
                        "start": s[0].time,
                        "end": s[-1].time,
                        "sample_count": len(s),
                        "candidates": [
                            asdict(c) for c in choose_candidates(s, cfg.candidates_per_spread)
                        ],
                    }
                    for s in segments
                ],
            )
            previous_spreads = []
            gaps = np.diff([s[0].time for s in segments])
            typical_gap = float(np.median(gaps)) if len(gaps) else 0
            score_rows = []
            for i, segment in enumerate(segments):
                spread_id = f"spread_{i + 1:04d}"
                records = []
                for j, sample in enumerate(choose_candidates(segment, cfg.candidates_per_spread)):
                    records.append(
                        candidate(project, manifest, cfg, detector, spread_id, j, sample)
                    )
                    score_rows.append(
                        {
                            "spread": spread_id,
                            "candidate": j,
                            "time": sample.time,
                            **records[-1]["metrics"],
                            "left_score": records[-1]["page_metrics"]["left"]["score"],
                            "right_score": records[-1]["page_metrics"]["right"]["score"],
                            "left_sharpness": records[-1]["page_metrics"]["left"]["sharpness"],
                            "right_sharpness": records[-1]["page_metrics"]["right"]["sharpness"],
                            "left_hand_overlap": records[-1]["page_metrics"]["left"]["hand_overlap"],
                            "right_hand_overlap": records[-1]["page_metrics"]["right"]["hand_overlap"],
                            "left_glare_overlap": records[-1]["page_metrics"]["left"].get(
                                "glare_overlap", 0.0
                            ),
                            "right_glare_overlap": records[-1]["page_metrics"]["right"].get(
                                "glare_overlap", 0.0
                            ),
                        }
                    )
                _augment_temporal_hand_masks(project, records, cfg)
                selection_mode = cfg.candidate_selection_mode if cfg.output_layout == "split" else "spread"
                selected, selected_pages = choose_candidate_selection(records, selection_mode)

                # Candidate scoring v2 is relative to this spread, so the values
                # only exist after all candidates have been collected. Persist
                # them back into candidate JSON and the debug CSV for inspection.
                recent_rows = score_rows[-len(records):]
                for record, row in zip(records, recent_rows):
                    relative = record["metrics"].get("relative_quality", {})
                    row.update(
                        {
                            "score": record["metrics"].get("score"),
                            "hand_overlap": record["metrics"].get("hand_overlap"),
                            "left_score": record["page_metrics"]["left"].get("score"),
                            "right_score": record["page_metrics"]["right"].get("score"),
                            "left_hand_overlap": record["page_metrics"]["left"].get(
                                "hand_overlap"
                            ),
                            "right_hand_overlap": record["page_metrics"]["right"].get(
                                "hand_overlap"
                            ),
                            "selection_score": record["metrics"].get("selection_score"),
                            "relative_sharpness": relative.get("sharpness"),
                            "relative_motion": relative.get("motion"),
                            "relative_hand_overlap": relative.get("hand_overlap"),
                            "relative_glare": relative.get("glare"),
                            "relative_base_score": relative.get("base_score"),
                            "glare": record["metrics"].get("glare"),
                            "glare_overlap": record["metrics"].get("glare_overlap", 0.0),
                            "sharpness_median": record["metrics"].get("sharpness_median"),
                            "sharpness_p10": record["metrics"].get("sharpness_p10"),
                            "sharpness_worst": record["metrics"].get("sharpness_worst"),
                            "left_selection_score": record["page_metrics"]["left"].get(
                                "selection_score"
                            ),
                            "right_selection_score": record["page_metrics"]["right"].get(
                                "selection_score"
                            ),
                            "left_glare": record["page_metrics"]["left"].get("glare"),
                            "right_glare": record["page_metrics"]["right"].get("glare"),
                            "left_glare_overlap": record["page_metrics"]["left"].get(
                                "glare_overlap", 0.0
                            ),
                            "right_glare_overlap": record["page_metrics"]["right"].get(
                                "glare_overlap", 0.0
                            ),
                            "left_sharpness_p10": record["page_metrics"]["left"].get(
                                "sharpness_p10"
                            ),
                            "right_sharpness_p10": record["page_metrics"]["right"].get(
                                "sharpness_p10"
                            ),
                        }
                    )
                    write_json(project / Path(record["path"]).with_suffix(".json"), record)

                spread = {
                    "id": spread_id,
                    "start": segment[0].time,
                    "end": segment[-1].time,
                    "candidates": records,
                    "selected": selected,
                    "selected_pages": selected_pages,
                    "candidate_selection_mode": cfg.candidate_selection_mode,
                    "extra_suspect": [],
                }
                if i and typical_gap and gaps[i - 1] > typical_gap * cfg.interval_gap_factor:
                    spread["extra_suspect"].append("interval_gap")
                thumbnail = selected_spread_preview(project, spread, cfg)
                for prev_id, prev_thumb in reversed(previous_spreads[-cfg.dedupe_window :]):
                    match = compare(thumbnail, prev_thumb, cfg)
                    if match["suspect"]:
                        spread["duplicate_check"] = {"other": prev_id, **match}
                        spread["extra_suspect"].append("duplicate_suspected")
                    if match["duplicate"]:
                        spread["duplicate_of"] = prev_id
                        break
                if not spread.get("duplicate_of"):
                    previous_spreads.append((spread_id, thumbnail))
                    previous_spreads = previous_spreads[-cfg.dedupe_window :]
                pages = render_spread(project, manifest, spread)
                manifest["spreads"].append(spread)
                manifest["pages"].extend(pages)
                update(
                    project,
                    manifest,
                    0.4 + 0.55 * (i + 1) / len(segments),
                    f"候補評価・補正 {i + 1} / {len(segments)} 見開き",
                )
            with (project / "debug/scores.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            update(project, manifest, 0.97, "PDF / CBZを生成中")
            build_exports(project, manifest)
            manifest.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 2))
            update(project, manifest, 1, "完了 — 要確認ページを確認してください")
            return manifest
        except BaseException as exc:
            manifest.update(
                status="failed", error=str(exc), message=f"処理停止: {exc}", pdf_stale=True
            )
            save_manifest(project, manifest)
            LOG.exception("Processing failed")
            raise
        finally:
            detector.close()
            LOG.removeHandler(handler)
            handler.close()


def edit(project, action, **params):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        cfg = Config.from_dict(manifest["config"])
        if action == "export":
            manifest["message"] = "PDF / CBZを生成中です"
            save_manifest(project, manifest)
            try:
                build_exports(project, manifest)
            except Exception:
                manifest["message"] = "PDF / CBZの出力に失敗しました"
                save_manifest(project, manifest)
                raise
            manifest["message"] = "PDF / CBZを出力しました"
            save_manifest(project, manifest)
            return manifest
        if action in ("undo_page_edit", "redo_page_edit"):
            changed = _apply_page_history(
                manifest,
                "undo" if action == "undo_page_edit" else "redo",
            )
            if not changed:
                return manifest
        elif action == "toggle_page":
            before = _page_review_state(manifest)
            page = next(p for p in manifest["pages"] if p["id"] == params["page_id"])
            page["enabled"] = not page["enabled"]
            _push_page_history(manifest, before, "除外 / 復元")
        elif action == "move_page":
            index = next(
                i for i, p in enumerate(manifest["pages"]) if p["id"] == params["page_id"]
            )
            destination = max(
                0,
                min(len(manifest["pages"]) - 1, index + int(params["delta"])),
            )
            if destination == index:
                return manifest
            before = _page_review_state(manifest)
            manifest["pages"].insert(destination, manifest["pages"].pop(index))
            _push_page_history(manifest, before, "ページ並び替え")
        elif action == "reorder_pages":
            requested = params.get("page_ids")
            if not isinstance(requested, list) or not all(
                isinstance(page_id, str) for page_id in requested
            ):
                raise ValueError("page_ids must be a list of page IDs")
            current_ids = [page["id"] for page in manifest["pages"]]
            if len(requested) != len(current_ids) or len(set(requested)) != len(requested):
                raise ValueError("page_ids must contain every page exactly once")
            if set(requested) != set(current_ids):
                raise ValueError("page_ids do not match the current pages")
            if requested == current_ids:
                return manifest
            before = _page_review_state(manifest)
            pages = {page["id"]: page for page in manifest["pages"]}
            manifest["pages"] = [pages[page_id] for page_id in requested]
            _push_page_history(manifest, before, "ドラッグ並び替え")
        elif action == "page_settings":
            page = next(p for p in manifest["pages"] if p["id"] == params["page_id"])
            if page["side"] == "cover":
                raise ValueError("Cover page overrides are not supported")
            spread = next(s for s in manifest["spreads"] if s["id"] == page["spread_id"])
            side = page["side"]
            patch = params.get("settings")
            if not isinstance(patch, dict) or not patch:
                raise ValueError("Page settings must be a non-empty object")

            allowed = {
                "dewarp",
                "illumination_correction",
                "white_normalization",
                "page_quad_mode",
                "manual_quad",
            }
            unknown = set(patch) - allowed
            if unknown:
                raise ValueError(f"Unknown page settings: {sorted(unknown)}")

            override = dict(_page_override(spread, side))
            for name in ("dewarp", "illumination_correction", "white_normalization"):
                if name not in patch:
                    continue
                if type(patch[name]) is not bool:
                    raise ValueError(f"{name} must be a boolean")
                if side == "spread" and name == "dewarp":
                    raise ValueError("Dewarp is only available for split pages")
                override[name] = patch[name]

            if "manual_quad" in patch:
                if side not in ("left", "right"):
                    raise ValueError("Manual page contour is only available for split pages")
                override["manual_quad"] = validate_roi(patch["manual_quad"]).tolist()

            if "page_quad_mode" in patch:
                mode = patch["page_quad_mode"]
                if mode not in ("auto", "manual"):
                    raise ValueError("page_quad_mode must be auto or manual")
                if side not in ("left", "right"):
                    raise ValueError("Page contour mode is only available for split pages")
                if mode == "manual":
                    quad = patch.get("manual_quad", override.get("manual_quad"))
                    if quad is None:
                        raise ValueError("manual page contour requires manual_quad")
                    override["manual_quad"] = validate_roi(quad).tolist()
                    override["page_quad_mode"] = "manual"
                else:
                    override.pop("manual_quad", None)
                    override["page_quad_mode"] = "auto"

            spread.setdefault("page_overrides", {})[side] = override
            indices = [
                i
                for i, existing in enumerate(manifest["pages"])
                if existing["spread_id"] == spread["id"]
            ]
            replacements = {
                rendered["id"]: rendered
                for rendered in render_spread(project, manifest, spread)
            }
            for index in indices:
                old = manifest["pages"][index]
                new = replacements[old["id"]]
                new["enabled"] = old["enabled"]
                manifest["pages"][index] = new
        elif action == "output_layout":
            layout = params["layout"]
            if layout not in ("spread", "split"):
                raise ValueError("Output layout must be spread or split")
            spread = next(s for s in manifest["spreads"] if s["id"] == params["spread_id"])
            old_layout = spread.get("output_layout", cfg.output_layout)
            if layout == old_layout:
                return manifest
            old_pages = [p for p in manifest["pages"] if p["spread_id"] == spread["id"]]
            state = spread.setdefault("layout_page_state", {})
            state[old_layout] = [{"id": p["id"], "enabled": p["enabled"]} for p in old_pages]
            spread["output_layout"] = layout
            new_pages = render_spread(project, manifest, spread)
            restored = state.get(layout, [])
            enabled = {p["id"]: p["enabled"] for p in restored}
            for page in new_pages:
                page["enabled"] = enabled.get(page["id"], any(p["enabled"] for p in old_pages))
            order = {p["id"]: i for i, p in enumerate(restored)}
            new_pages.sort(key=lambda p: order.get(p["id"], len(order)))
            position = next(i for i, p in enumerate(manifest["pages"])
                            if p["spread_id"] == spread["id"])
            manifest["pages"] = [
                p for p in manifest["pages"] if p["spread_id"] != spread["id"]
            ]
            manifest["pages"][position:position] = new_pages
            _clear_page_history(manifest)
        elif action in (
            "select_candidate",
            "swap",
            "spine",
            "toggle_dewarp",
            "crop",
            "reset_crop",
        ):
            spread = next(s for s in manifest["spreads"] if s["id"] == params["spread_id"])
            layout = spread.get("output_layout", cfg.output_layout)
            if layout == "spread" and (
                action in ("swap", "spine", "toggle_dewarp")
                or (action == "select_candidate" and params.get("side") is not None)
            ):
                raise ValueError("This action requires split output")
            indices = [i for i, p in enumerate(manifest["pages"]) if p["spread_id"] == spread["id"]]
            if action == "swap":
                if len(indices) != 2:
                    raise ValueError("Expected two pages in this spread")
                before = _page_review_state(manifest)
                a, b = indices
                manifest["pages"][a], manifest["pages"][b] = (
                    manifest["pages"][b],
                    manifest["pages"][a],
                )
                _push_page_history(manifest, before, "左右の順番を入れ替え")
            else:
                if action in ("crop", "reset_crop"):
                    candidate_id = int(params["candidate_id"])
                    if candidate_id not in [c["id"] for c in spread["candidates"]]:
                        raise ValueError("Unknown candidate")
                    if action == "crop":
                        # The review editor displays the original frame upright.
                        roi = rotate_roi(params["roi"], (360 - cfg.rotation) % 360).tolist()
                        spread.setdefault("roi_overrides", {})[str(candidate_id)] = roi
                    else:
                        overrides = spread.get("roi_overrides", {})
                        overrides.pop(str(candidate_id), None)
                        if not overrides:
                            spread.pop("roi_overrides", None)
                elif action == "spine":
                    ratio = float(params["ratio"])
                    if not 0.25 <= ratio <= 0.75:
                        raise ValueError("Spine ratio must be 0.25..0.75")
                    spread["spine_ratio"] = ratio
                elif action == "toggle_dewarp":
                    side = params["side"]
                    if side not in ("left", "right"):
                        raise ValueError("Unknown page side")
                    disabled = set(spread.get("dewarp_disabled_sides", []))
                    if side in disabled:
                        disabled.remove(side)
                    else:
                        disabled.add(side)
                    spread["dewarp_disabled_sides"] = sorted(disabled)
                else:
                    selection = int(params["candidate_id"])
                    if selection not in [c["id"] for c in spread["candidates"]]:
                        raise ValueError("Unknown candidate")
                    side = params.get("side")
                    if side is None:
                        spread["selected"] = selection
                        spread["selected_pages"] = {"left": selection, "right": selection}
                    else:
                        if side not in ("left", "right"):
                            raise ValueError("Unknown page side")
                        selected_pages = spread.get("selected_pages") or {
                            "left": spread["selected"],
                            "right": spread["selected"],
                        }
                        spread["selected_pages"] = {**selected_pages, side: selection}
                replacements = {p["id"]: p for p in render_spread(project, manifest, spread)}
                for index in indices:
                    old = manifest["pages"][index]
                    new = replacements[old["id"]]
                    new["enabled"] = old["enabled"]
                    manifest["pages"][index] = new
        elif action == "add_frame":
            timestamp = float(params["time"])
            if (
                not math.isfinite(timestamp)
                or not 0 <= timestamp < manifest["metadata"]["duration"]
            ):
                raise ValueError("Timestamp outside video duration")
            validate_roi(manifest["roi"])
            spread_id = f"spread_{max([int(s['id'].split('_')[1]) for s in manifest['spreads']] + [0]) + 1:04d}"
            detector = HandDetector(cfg)
            try:
                rec = candidate(
                    project, manifest, cfg, detector, spread_id, 0, Sample(0, timestamp, 0, 0)
                )
            finally:
                detector.close()
            manual_reason = "manual_frame_motion_unmeasured"
            rec["suspect"] = list(dict.fromkeys(rec.get("suspect", []) + [manual_reason]))
            write_json(project / Path(rec["path"]).with_suffix(".json"), rec)
            spread = {
                "id": spread_id,
                "start": timestamp,
                "end": timestamp,
                "candidates": [rec],
                "selected": 0,
                "selected_pages": {"left": 0, "right": 0},
                "candidate_selection_mode": cfg.candidate_selection_mode,
                "extra_suspect": ["manual_frame", manual_reason],
            }
            pages = render_spread(project, manifest, spread)
            # Insert in chronological order; subsequent manual reordering is explicit.
            later = next((s for s in manifest["spreads"] if s["start"] > timestamp), None)
            position = next(
                (
                    i
                    for i, p in enumerate(manifest["pages"])
                    if later and p["spread_id"] == later["id"]
                ),
                len(manifest["pages"]),
            )
            manifest["pages"][position:position] = pages
            manifest["spreads"].append(spread)
            manifest["spreads"].sort(key=lambda s: s["start"])
            _clear_page_history(manifest)
        else:
            raise ValueError("Unknown review action")
        manifest["pdf_stale"] = True
        _refresh_adjacent_final_quality(project, manifest, cfg)
        save_manifest(project, manifest)
        return manifest
