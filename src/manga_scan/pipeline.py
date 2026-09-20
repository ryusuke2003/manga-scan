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
from .export import contact_sheets, export_pdf
from .finger_repair import repair_finger_regions
from .hand import HandDetector, boundary_finger_mask
from .motion import Sample, StableDetector, choose_candidates, motion_score
from .page_contour import detect_page_quads, draw_page_quads, spread_quad_from_page_quads
from .page_detect import refine_quad
from .page_warp import warp_detected_pages
from .perspective import rotate_roi, validate_roi, warp_roi
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
    metrics = score_frame(image, roi, sample.motion, overlap, cfg)
    base = f"candidates/{spread_id}/candidate_{number:02d}"
    save_image(project / f"{base}.png", image)
    save_image(project / f"{base}_hand_mask.png", mask)
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
        page_metrics, _ = score_candidate_pages(
            rectified,
            rectified_mask,
            sample.motion,
            cfg,
            metrics,
            hand_enabled=overlap is not None,
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
        "roi": roi,
        "metrics": metrics,
        "page_metrics": page_metrics,
        "page_suspect": page_suspect,
        "suspect": suspect_reasons(metrics, cfg, quad_ok),
    }
    write_json(project / f"{base}.json", record)
    return record


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
        "suspect": [],
    }


def rectify_spread_pages(project, image, rectified, chosen_roi, spread, cfg):
    """Choose per-page perspective correction or the legacy spread fallback."""

    ratio = spread.get("spine_ratio", cfg.spine_ratio)
    if cfg.perspective_mode == "per_page" and not spread.get("manual_roi"):
        detection_image = image
        if image.shape[1] > cfg.analysis_width:
            scale = cfg.analysis_width / image.shape[1]
            detection_image = cv2.resize(
                image,
                (cfg.analysis_width, max(2, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        detection = detect_page_quads(
            detection_image,
            chosen_roi,
            spine_ratio=ratio,
            min_confidence=cfg.page_contour_min_confidence,
        )
        spread["page_contours"] = detection
        debug_path = f"debug/page_contours/{spread['id']}.jpg"
        save_image(project / debug_path, draw_page_quads(image, detection))
        spread["page_contour_debug"] = debug_path

        if detection["detected"]:
            spread["perspective_mode_used"] = "per_page"
            spread["spine_px"] = spine_position(rectified, ratio, cfg.split_mode)
            return warp_detected_pages(image, detection)

        spread["perspective_mode_used"] = "spread_fallback"
        extra = spread.setdefault("extra_suspect", [])
        if "page_contour_low_confidence" not in extra:
            extra.append("page_contour_low_confidence")
    else:
        spread["perspective_mode_used"] = "spread"
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

    if (
        state.get("perspective_mode_used") == "per_page"
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


def _finger_donor_candidates(spread, side, selected_id):
    def rank(candidate):
        metrics = candidate.get("page_metrics", {}).get(side, candidate.get("metrics", {}))
        overlap = metrics.get("hand_overlap")
        return (
            1.0 if overlap is None else float(overlap),
            -float(metrics.get("score", 0.0)),
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


def _whole_spread_geometry(source, record, spread, cfg):
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
    cache = {}

    def load(candidate_id):
        if candidate_id in cache:
            return cache[candidate_id]
        record = _candidate_by_id(spread, candidate_id)
        source = extract_frame(manifest["source"], record["time"], hwaccel=cfg.hwaccel)
        upright, roi, crop = _whole_spread_geometry(source, record, spread, cfg)
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
        mask = None
        if cfg.hand_backend == "mediapipe" and record.get("hand_mask"):
            saved = cv2.imread(str(project / record["hand_mask"]), cv2.IMREAD_GRAYSCALE)
            if saved is not None:
                saved = cv2.resize(
                    saved, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                upright_mask = rotate_image(saved, cfg.rotation)
                mask = warp_roi(
                    upright_mask, roi, interpolation=cv2.INTER_NEAREST
                )
                mask |= boundary_finger_mask(
                    page, [[0, 0], [1, 0], [1, 1], [0, 1]], cfg.hand_padding
                )
        cache[candidate_id] = {
            "page": page,
            "mask": mask,
            "crop": crop,
            "upright": upright,
            "background_mask": background_mask,
        }
        return cache[candidate_id]

    chosen = _candidate_by_id(spread, spread["selected"])
    selected = load(chosen["id"])
    page, mask = selected["page"], selected["mask"]
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
    if cfg.finger_repair:
        if mask is None:
            repair = {"status": "unavailable", "coverage": 0.0, "donors": []}
        else:

            def donors():
                records = sorted(
                    spread["candidates"], key=lambda c: -c.get("metrics", {}).get("score", 0)
                )
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

            page, repair, unresolved = repair_finger_regions(
                page,
                mask,
                donors(),
                min_coverage=cfg.finger_repair_min_coverage,
                fallback=cfg.finger_repair_fallback,
            )
            if np.any(mask):
                repair["target_mask"] = f"debug/finger_repair/{spread['id']}_whole_target.png"
                save_image(project / repair["target_mask"], mask)
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
    page = enhance_page(
        page,
        grayscale=cfg.grayscale,
        contrast=cfg.contrast,
        illumination_correction=cfg.illumination_correction,
        illumination_strength=cfg.illumination_strength,
        white_normalization=cfg.white_normalization,
        white_target=cfg.white_target,
        white_strength=cfg.white_strength,
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
    if mask is not None:
        suspect = [reason for reason in suspect if reason != "hand_overlap"]
        final_hand_overlap = float(np.mean(mask > 127))
        if final_hand_overlap >= cfg.suspect_hand_overlap:
            suspect.append("hand_overlap")
    if repair["status"] in ("clean", "complete"):
        suspect = [reason for reason in suspect if reason != "hand_overlap"]
    elif repair["status"] in ("incomplete", "unavailable"):
        suspect.append("finger_repair_incomplete")
    suspect = list(dict.fromkeys(suspect))
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
            "crop": spread["whole_spread_crop"],
            "dewarp": {"mode": "off", "status": "off", "applied": False},
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
            }
        )
        state["manual_roi"] = override is not None
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
    disabled_sides = set(spread.get("dewarp_disabled_sides", []))
    for side in order:
        data = selected_data[side]
        chosen = data["chosen"]
        source_page = data["sides"][side]
        selected_source = f"selected/{spread['id']}_{side}.png"
        save_image(project / selected_source, source_page)

        finger_repair = {"status": "disabled", "coverage": 0.0, "donors": []}
        if cfg.finger_repair:
            target_mask = candidate_page_hand_mask(project, data, side, cfg)
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
                        donor_mask = candidate_page_hand_mask(
                            project, donor_data, side, cfg
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
                finger_repair["target_mask"] = target_mask_path
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
                    "status": "clean",
                    "coverage": 1.0,
                    "donors": [],
                }

        dewarp = {"mode": cfg.dewarp_mode, "applied": False, "status": "off"}
        manual_dewarp = 0.0
        if cfg.dewarp_mode == "manual":
            manual_dewarp = cfg.dewarp_strength
            dewarp.update(
                applied=bool(manual_dewarp),
                status="applied" if manual_dewarp else "off",
                strength=manual_dewarp,
            )
        elif cfg.dewarp_mode == "auto":
            if side in disabled_sides:
                dewarp.update(status="disabled")
            else:
                before = f"debug/dewarp/{spread['id']}_{side}_before.png"
                before_image = enhance_page(
                    source_page,
                    grayscale=cfg.grayscale,
                    contrast=cfg.contrast,
                    rotation=0,
                    dewarp_strength=0.0,
                    white_normalization=cfg.white_normalization,
                    white_target=cfg.white_target,
                    white_strength=cfg.white_strength,
                    illumination_correction=cfg.illumination_correction,
                    illumination_strength=cfg.illumination_strength,
                )
                save_image(project / before, before_image)
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

        page_image = enhance_page(
            source_page,
            grayscale=cfg.grayscale,
            contrast=cfg.contrast,
            rotation=0,
            dewarp_strength=manual_dewarp,
            white_normalization=cfg.white_normalization,
            white_target=cfg.white_target,
            white_strength=cfg.white_strength,
            illumination_correction=cfg.illumination_correction,
            illumination_strength=cfg.illumination_strength,
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
            page_suspect = [reason for reason in page_suspect if reason != "hand_overlap"]
        elif finger_repair["status"] in ("incomplete", "unavailable"):
            page_suspect.append("finger_repair_incomplete")
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


def build_pdf(project, manifest):
    cfg = Config.from_dict(manifest["config"])
    paths = [project / p["path"] for p in manifest["pages"] if p["enabled"]]
    export_pdf(paths, project / "output/manga.pdf", cfg.pdf_dpi, cfg.image_format, cfg.jpeg_quality)
    manifest["pdf_stale"] = False
    manifest["pdf"] = "output/manga.pdf"
    contact_sheets(project, manifest["pages"])
    save_manifest(project, manifest)


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
                        }
                    )
                selection_mode = cfg.candidate_selection_mode if cfg.output_layout == "split" else "spread"
                selected, selected_pages = choose_candidate_selection(records, selection_mode)
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
            update(project, manifest, 0.97, "PDFを生成中")
            build_pdf(project, manifest)
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
            manifest["message"] = "PDFを生成中です"
            save_manifest(project, manifest)
            try:
                build_pdf(project, manifest)
            except Exception:
                manifest["message"] = "PDFの出力に失敗しました"
                save_manifest(project, manifest)
                raise
            manifest["message"] = "PDFを出力しました"
            save_manifest(project, manifest)
            return manifest
        if action == "toggle_page":
            page = next(p for p in manifest["pages"] if p["id"] == params["page_id"])
            page["enabled"] = not page["enabled"]
        elif action == "move_page":
            index = next(i for i, p in enumerate(manifest["pages"]) if p["id"] == params["page_id"])
            destination = max(0, min(len(manifest["pages"]) - 1, index + int(params["delta"])))
            manifest["pages"].insert(destination, manifest["pages"].pop(index))
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
            manifest["pages"] = [p for p in manifest["pages"] if p["spread_id"] != spread["id"]]
            manifest["pages"][position:position] = new_pages
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
                a, b = indices
                manifest["pages"][a], manifest["pages"][b] = (
                    manifest["pages"][b],
                    manifest["pages"][a],
                )
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
        else:
            raise ValueError("Unknown review action")
        manifest["pdf_stale"] = True
        save_manifest(project, manifest)
        return manifest
