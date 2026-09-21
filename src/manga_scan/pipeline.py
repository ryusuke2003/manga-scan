import csv
import hashlib
import json
import logging
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import pipeline_render_helpers as render_helpers
from .config import Config
from .dedupe import compare
from .export import (
    contact_sheets,
    export_cbz,
    export_pdf,
    metadata_output_stem,
    normalize_book_metadata,
)
from .final_quality import FINAL_QUALITY_REASONS, adjacent_quality_check, final_quality_checks
from .finger_repair import repair_finger_regions
from .glare import detect_glare_mask, glare_overlap_fraction
from .hand import HandDetector, boundary_finger_mask, temporal_transient_mask
from .high_fps_fallback import best_low_motion_run, fallback_reasons
from .input_validation import load_bounded_rgb_image, validate_manifest_video
from .motion import Sample, StableDetector, choose_candidates, motion_score
from .page_contour import detect_page_quads
from .page_detect import refine_quad
from .page_turns import analyze_page_turns
from .perspective import pixel_quad, rotate_roi, validate_roi, warp_roi
from .processing_control import ProcessingCancelled, clear_cancel_request, raise_if_cancelled
from .roi_tracking import track_spread_roi
from .quality_safety import (
    normalize_expected_page_count,
    refresh_review_safety,
)
from .score import score_frame, sharpness, suspect_reasons
from .selection import choose_candidate_selection, score_candidate_pages
from .split import (
    auto_dewarp_page,
    dewarp_debug_grid,
    enhance_page,
    rotate_image,
    split_spread,
)
from .spread_render import render_whole_spread as _render_whole_spread_impl
from .storage import project_lock, read_manifest, save_image, save_manifest, write_json
from .video import extract_frame, sample_frames

_finger_donor_candidates = render_helpers._finger_donor_candidates
_page_override = render_helpers._page_override
_page_render_settings = render_helpers._page_render_settings
_persist_finger_repair_component_debug = render_helpers._persist_finger_repair_component_debug
_union_occlusion_masks = render_helpers._union_occlusion_masks
_whole_spread_geometry_impl = render_helpers._whole_spread_geometry
candidate_page_glare_mask = render_helpers.candidate_page_glare_mask
_candidate_page_hand_mask_impl = render_helpers.candidate_page_hand_mask
_detect_spread_page_consensus_impl = render_helpers.detect_spread_page_consensus
_rectify_spread_pages_impl = render_helpers.rectify_spread_pages

LOG = logging.getLogger("manga_scan")



def update(project, manifest, progress, message):
    manifest.update(progress=round(progress, 3), message=message)
    save_manifest(project, manifest)
    LOG.info("%3.0f%% %s", progress * 100, message)


def candidate(project, manifest, cfg, detector, spread_id, number, sample, base_roi=None):
    # Only candidate timestamps seek back to the original video.
    image = extract_frame(manifest["source"], sample.time, cfg.analysis_width, cfg.hwaccel)
    roi, quad_ok = (base_roi if base_roi is not None else manifest["roi"]), True
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
        "tracking_base_roi": (
            validate_roi(base_roi).tolist()
            if base_roi is not None
            else validate_roi(manifest["roi"]).tolist()
        ),
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
    return _detect_spread_page_consensus_impl(
        project,
        spread,
        cfg,
        anchor_ids=anchor_ids,
        detect_page_quads_fn=detect_page_quads,
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
    return _rectify_spread_pages_impl(
        project,
        image,
        rectified,
        chosen_roi,
        spread,
        cfg,
        page_detection=page_detection,
        consensus_sides=consensus_sides,
        manual_sides=manual_sides,
        detect_page_quads_fn=detect_page_quads,
    )


def candidate_page_hand_mask(project, data, side, cfg):
    return _candidate_page_hand_mask_impl(
        project,
        data,
        side,
        cfg,
        boundary_finger_mask_fn=boundary_finger_mask,
    )


def _whole_spread_geometry(source, record, spread, cfg, page_detection=None):
    return _whole_spread_geometry_impl(
        source,
        record,
        spread,
        cfg,
        page_detection=page_detection,
        detect_page_quads_fn=detect_page_quads,
    )


def _render_whole_spread(project, manifest, spread, cfg):
    return _render_whole_spread_impl(
        project,
        manifest,
        spread,
        cfg,
        detect_spread_page_consensus_fn=detect_spread_page_consensus,
        candidate_by_id_fn=_candidate_by_id,
        whole_spread_geometry_fn=_whole_spread_geometry,
        extract_frame_fn=extract_frame,
        boundary_finger_mask_fn=boundary_finger_mask,
        repair_finger_regions_fn=repair_finger_regions,
    )


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

    refresh_review_safety(manifest)


def _managed_previous_export(project, output, manifest_path, suffix):
    """Return a previous export only when it is a direct child of project/output."""
    if not manifest_path or not isinstance(manifest_path, str):
        return None
    relative = Path(manifest_path)
    if relative.is_absolute():
        return None
    candidate = (project / relative).resolve(strict=False)
    output_root = output.resolve(strict=False)
    if candidate.parent != output_root or candidate.suffix.lower() != suffix:
        return None
    return candidate


def build_exports(project, manifest):
    cfg = Config.from_dict(manifest["config"])
    _refresh_adjacent_final_quality(project, manifest, cfg)
    paths = [project / p["path"] for p in manifest["pages"] if p["enabled"]]
    output = project / "output"
    metadata = normalize_book_metadata(manifest.get("book_metadata"))
    stem = metadata_output_stem(metadata)
    pdf = output / f"{stem}.pdf"
    cbz = output / f"{stem}.cbz"
    next_pdf = output / f"{stem}.next.pdf"
    next_cbz = output / f"{stem}.next.cbz"
    previous_pdf = _managed_previous_export(project, output, manifest.get("pdf"), ".pdf")
    previous_cbz = _managed_previous_export(project, output, manifest.get("cbz"), ".cbz")
    try:
        export_pdf(
            paths,
            next_pdf,
            cfg.pdf_dpi,
            cfg.image_format,
            cfg.jpeg_quality,
            metadata,
        )
        export_cbz(paths, next_cbz, metadata)
        next_pdf.replace(pdf)
        next_cbz.replace(cbz)
        for previous, current in ((previous_pdf, pdf), (previous_cbz, cbz)):
            if previous and previous != current:
                previous.unlink(missing_ok=True)
    finally:
        next_pdf.unlink(missing_ok=True)
        next_cbz.unlink(missing_ok=True)

    manifest["pdf_stale"] = False
    manifest["pdf"] = str(pdf.relative_to(project))
    manifest["cbz"] = str(cbz.relative_to(project))
    contact_sheets(project, manifest["pages"])
    save_manifest(project, manifest)


def build_pdf(project, manifest):
    """Backward-compatible export entrypoint; now writes both PDF and CBZ."""
    build_exports(project, manifest)


_PAGE_HISTORY_LIMIT = 30


def _page_review_state(manifest, include_pages=False):
    pages = manifest.get("pages", [])
    state = {
        "order": [page["id"] for page in pages],
        "disabled": [page["id"] for page in pages if not page.get("enabled", True)],
    }
    if include_pages:
        state["pages"] = deepcopy(pages)
    return state


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
    snapshot = state.get("pages")
    if isinstance(snapshot, list):
        manifest["pages"] = deepcopy(snapshot)
        return

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
    entry_state = entry["state"]
    current = _page_review_state(
        manifest,
        include_pages=isinstance(entry_state.get("pages"), list),
    )
    _restore_page_review_state(manifest, entry_state)
    history[target_name].append({"label": entry.get("label", "ページ編集"), "state": current})
    history[target_name] = history[target_name][-_PAGE_HISTORY_LIMIT:]
    return True


def _interval_records(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("Invalid processing checkpoint intervals")
    return data


def _load_score_rows(project):
    path = Path(project) / "debug/scores.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_score_rows(project, rows):
    if not rows:
        return
    path = Path(project) / "debug/scores.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resume_tracking_state(project, manifest):
    spreads = manifest.get("spreads", [])
    if not spreads:
        return None, validate_roi(manifest["roi"]).tolist()
    for spread in reversed(spreads):
        candidates = spread.get("candidates", [])
        if not candidates:
            continue
        selected_id = spread.get("selected")
        record = next(
            (item for item in candidates if item.get("id") == selected_id),
            candidates[0],
        )
        image = cv2.imread(str(project / record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        roi = spread.get("tracked_roi", record.get("roi", manifest["roi"]))
        return image, validate_roi(roi).tolist()
    return None, validate_roi(manifest["roi"]).tolist()


def _resume_previous_spreads(project, manifest, cfg):
    previous = []
    for spread in manifest.get("spreads", []):
        if spread.get("duplicate_of"):
            continue
        thumbnail = selected_spread_preview(project, spread, cfg)
        previous.append((spread["id"], thumbnail))
    return previous[-cfg.dedupe_window :]


def _high_fps_window_samples(manifest, cfg, start, end, requested_fps):
    source_fps = float(manifest.get("metadata", {}).get("fps") or requested_fps)
    effective_fps = min(float(requested_fps), source_fps) if source_fps > 0 else float(requested_fps)
    analysis_fps = float(manifest.get("analysis_fps") or cfg.video_sample_fps)
    if effective_fps <= analysis_fps + 1e-6 or end <= start:
        return [], effective_fps

    height = int(manifest["metadata"]["display_height"])
    width = int(manifest["metadata"]["display_width"])
    analysis_width = min(cfg.analysis_width, width)
    size = (
        analysis_width,
        max(2, round(height * analysis_width / width)),
    )

    samples = []
    previous = None
    stream = sample_frames(
        manifest["source"],
        effective_fps,
        size,
        cfg.hwaccel,
        start_time=max(0.0, float(start)),
    )
    try:
        for index, timestamp, frame in stream:
            if timestamp > float(end) + (0.5 / effective_fps):
                break
            cropped = warp_roi(frame, manifest["roi"])
            motion = motion_score(previous, cropped) if previous is not None else 1.0
            samples.append(Sample(index, timestamp, motion, sharpness(cropped)))
            previous = cropped
    finally:
        stream.close()
    return samples, effective_fps


def _add_auto_high_fps_candidates(
    project,
    manifest,
    cfg,
    detector,
    spread_id,
    records,
    start,
    end,
    reasons,
    base_roi=None,
):
    samples, effective_fps = _high_fps_window_samples(
        manifest,
        cfg,
        start,
        end,
        cfg.auto_high_fps_fallback_fps,
    )
    if not samples:
        return [], effective_fps

    eligible = [sample for sample in samples if sample.motion <= cfg.turn_threshold]
    if not eligible:
        return [], effective_fps

    limit = max(3, min(6, int(cfg.candidates_per_spread)))
    picked = choose_candidates(eligible, limit)
    existing_times = [float(item["time"]) for item in records]
    duplicate_tolerance = 0.5 / effective_fps
    picked = [
        sample
        for sample in picked
        if all(abs(sample.time - current) > duplicate_tolerance for current in existing_times)
    ]
    if not picked:
        return [], effective_fps

    next_id = max((int(item["id"]) for item in records), default=-1) + 1
    added = []
    for offset, sample in enumerate(picked):
        raise_if_cancelled(project)
        record = candidate(
            project,
            manifest,
            cfg,
            detector,
            spread_id,
            next_id + offset,
            sample,
            base_roi=base_roi,
        )
        record["rescan"] = {
            "automatic": True,
            "trigger_reasons": list(reasons),
            "window_start": float(start),
            "window_end": float(end),
            "requested_fps": float(cfg.auto_high_fps_fallback_fps),
            "effective_fps": float(effective_fps),
        }
        added.append(record)
    records.extend(added)
    _augment_temporal_hand_masks(project, records, cfg)
    return added, effective_fps


def _recover_missing_segments_high_fps(project, manifest, cfg, motion_samples, segments, fps):
    analysis = analyze_page_turns(
        motion_samples,
        segments,
        cfg.motion_threshold,
        cfg.turn_threshold,
        fps,
    )
    if not cfg.auto_high_fps_fallback:
        return segments, analysis

    recovered = []
    for missing in list(analysis.get("missing_candidates", [])):
        raise_if_cancelled(project)
        samples, effective_fps = _high_fps_window_samples(
            manifest,
            cfg,
            float(missing["window_start"]),
            float(missing["window_end"]),
            cfg.auto_high_fps_fallback_fps,
        )
        stable = best_low_motion_run(
            samples,
            cfg.motion_threshold,
            effective_fps,
            cfg.auto_high_fps_min_stable_seconds,
        )
        if not stable:
            continue
        segments.append(stable)
        best = min(stable, key=lambda sample: sample.motion)
        recovered.append(
            {
                "id": missing["id"],
                "original_time": float(missing["time"]),
                "time": float(best.time),
                "start": float(stable[0].time),
                "end": float(stable[-1].time),
                "sample_count": len(stable),
                "requested_fps": float(cfg.auto_high_fps_fallback_fps),
                "effective_fps": float(effective_fps),
                "reason": "high_fps_stable_interval_recovered",
            }
        )

    if recovered:
        segments.sort(key=lambda segment: float(segment[0].time))
        analysis = analyze_page_turns(
            motion_samples,
            segments,
            cfg.motion_threshold,
            cfg.turn_threshold,
            fps,
        )
    analysis["high_fps_fallback"] = {
        "enabled": True,
        "requested_fps": float(cfg.auto_high_fps_fallback_fps),
        "min_stable_seconds": float(cfg.auto_high_fps_min_stable_seconds),
        "recovered_candidates": recovered,
    }
    return segments, analysis


def _score_row(spread_id, record):
    metrics = record["metrics"]
    left = record["page_metrics"]["left"]
    right = record["page_metrics"]["right"]
    relative = metrics.get("relative_quality", {})
    return {
        "spread": spread_id,
        "candidate": record["id"],
        "time": record["time"],
        **metrics,
        "left_score": left.get("score"),
        "right_score": right.get("score"),
        "left_sharpness": left.get("sharpness"),
        "right_sharpness": right.get("sharpness"),
        "left_hand_overlap": left.get("hand_overlap"),
        "right_hand_overlap": right.get("hand_overlap"),
        "left_glare_overlap": left.get("glare_overlap", 0.0),
        "right_glare_overlap": right.get("glare_overlap", 0.0),
        "selection_score": metrics.get("selection_score"),
        "relative_sharpness": relative.get("sharpness"),
        "relative_motion": relative.get("motion"),
        "relative_hand_overlap": relative.get("hand_overlap"),
        "relative_glare": relative.get("glare"),
        "relative_base_score": relative.get("base_score"),
        "glare": metrics.get("glare"),
        "glare_overlap": metrics.get("glare_overlap", 0.0),
        "sharpness_median": metrics.get("sharpness_median"),
        "sharpness_p10": metrics.get("sharpness_p10"),
        "sharpness_worst": metrics.get("sharpness_worst"),
        "left_selection_score": left.get("selection_score"),
        "right_selection_score": right.get("selection_score"),
        "left_glare": left.get("glare"),
        "right_glare": right.get("glare"),
        "left_sharpness_p10": left.get("sharpness_p10"),
        "right_sharpness_p10": right.get("sharpness_p10"),
    }


def run(project, roi=None):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed. Use review edits or create a new project")
        cfg = Config.from_dict(manifest["config"])
        validate_manifest_video(manifest, cfg, require_dimensions=True)
        previous_roi = manifest.get("roi")
        requested_roi = validate_roi(roi if roi is not None else previous_roi).tolist()
        checkpoint = manifest.get("processing_checkpoint") or {}
        interval_path = project / "debug/intervals.json"
        same_roi = previous_roi is not None and np.allclose(
            np.asarray(previous_roi, dtype=np.float32),
            np.asarray(requested_roi, dtype=np.float32),
            atol=1e-6,
        )
        resume = bool(
            manifest.get("status") == "cancelled"
            and same_roi
            and checkpoint.get("motion_analysis_complete")
            and interval_path.is_file()
        )
        manifest["roi"] = requested_roi
        reference = manifest.get("reference") or {}
        analysis_start = float(reference.get("time", 0)) if reference.get("confirmed") else 0.0
        detector = HandDetector(cfg)  # Fail before expensive analysis if hand support is missing.
        handler = logging.FileHandler(project / "debug/process.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)
        started = time.monotonic()
        try:
            manifest.pop("error", None)
            if cfg.hand_backend == "mediapipe":
                manifest["hand_model_sha256"] = hashlib.sha256(
                    Path(cfg.hand_model).read_bytes()
                ).hexdigest()

            if resume:
                interval_records = _interval_records(interval_path)
                completed_spreads = int(checkpoint.get("completed_spreads", 0))
                if not 0 <= completed_spreads <= len(interval_records):
                    raise ValueError("Invalid processing checkpoint")
                expected_ids = [
                    f"spread_{index + 1:04d}" for index in range(completed_spreads)
                ]
                current_ids = [
                    spread["id"] for spread in manifest.get("spreads", [])
                ]
                if current_ids != expected_ids:
                    raise ValueError(
                        "Processing checkpoint does not match completed spreads"
                    )
                manifest.update(
                    status="processing",
                    pdf_stale=True,
                    message="中断地点から再開しています",
                )
                previous_spreads = _resume_previous_spreads(
                    project,
                    manifest,
                    cfg,
                )
                tracking_image, tracking_roi = _resume_tracking_state(project, manifest)
                score_rows = _load_score_rows(project)
                update(
                    project,
                    manifest,
                    0.4 + 0.55 * completed_spreads / max(1, len(interval_records)),
                    f"中断地点から再開 {completed_spreads} / {len(interval_records)} 見開き",
                )
            else:
                manifest.update(
                    status="processing",
                    spreads=[],
                    pages=[],
                    pdf_stale=True,
                    page_turn_analysis=None,
                    processing_checkpoint={
                        "motion_analysis_complete": False,
                        "completed_spreads": 0,
                    },
                )
                _clear_page_history(manifest)
                cover_page = render_cover(project, manifest)
                if cover_page:
                    manifest["pages"].append(cover_page)
                update(project, manifest, 0, "低解像度で動きを解析中")
                h = manifest["metadata"]["display_height"]
                w = manifest["metadata"]["display_width"]
                width = min(cfg.analysis_width, w)
                size = (width, max(2, round(h * width / w)))
                fps = min(cfg.video_sample_fps, manifest["metadata"]["fps"] or cfg.video_sample_fps)
                manifest["analysis_fps"] = fps
                machine = StableDetector(
                    cfg.stable_frames,
                    cfg.motion_threshold,
                    cfg.turn_threshold,
                )
                segments, previous, motion_samples = [], None, []
                with (project / "debug/motion.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "time", "motion", "sharpness", "state"])
                    for index, timestamp, frame in sample_frames(
                        manifest["source"],
                        fps,
                        size,
                        cfg.hwaccel,
                        start_time=analysis_start,
                    ):
                        raise_if_cancelled(project)
                        cropped = warp_roi(frame, manifest["roi"])
                        motion = (
                            motion_score(previous, cropped)
                            if previous is not None
                            else 1.0
                        )
                        sample = Sample(
                            index,
                            timestamp,
                            motion,
                            sharpness(cropped),
                        )
                        motion_samples.append(sample)
                        complete = machine.push(sample)
                        if complete:
                            segments.append(complete)
                        writer.writerow(
                            [
                                index,
                                timestamp,
                                motion,
                                sample.sharpness,
                                machine.state,
                            ]
                        )
                        previous = cropped
                        if cfg.save_lowres:
                            save_image(
                                project / f"frames_lowres/{index:08d}.jpg",
                                frame,
                            )
                        if index % max(1, round(fps * 2)) == 0:
                            update(
                                project,
                                manifest,
                                min(
                                    0.4,
                                    0.4
                                    * max(0.0, timestamp - analysis_start)
                                    / max(
                                        0.001,
                                        manifest["metadata"]["duration"]
                                        - analysis_start,
                                    ),
                                ),
                                (
                                    f"動き解析 {timestamp:.1f}s / "
                                    f"{manifest['metadata']['duration']:.1f}s"
                                ),
                            )
                tail = machine.finish()
                if tail:
                    segments.append(tail)
                if not segments:
                    raise ValueError(
                        "No stable intervals found. Hold pages longer, tune "
                        "motion_threshold/stable_frames, or add frames manually"
                    )
                segments, page_turn_analysis = _recover_missing_segments_high_fps(
                    project,
                    manifest,
                    cfg,
                    motion_samples,
                    segments,
                    fps,
                )
                manifest["page_turn_analysis"] = page_turn_analysis
                write_json(project / "debug/page_turns.json", page_turn_analysis)
                interval_records = [
                    {
                        "start": segment[0].time,
                        "end": segment[-1].time,
                        "sample_count": len(segment),
                        "candidates": [
                            asdict(candidate_sample)
                            for candidate_sample in choose_candidates(
                                segment,
                                cfg.candidates_per_spread,
                            )
                        ],
                    }
                    for segment in segments
                ]
                write_json(interval_path, interval_records)
                manifest["processing_checkpoint"] = {
                    "motion_analysis_complete": True,
                    "completed_spreads": 0,
                }
                save_manifest(project, manifest)
                raise_if_cancelled(project)
                previous_spreads = []
                tracking_image = None
                tracking_roi = validate_roi(manifest["roi"]).tolist()
                score_rows = []
                completed_spreads = 0

            gaps = np.diff([item["start"] for item in interval_records])
            typical_gap = float(np.median(gaps)) if len(gaps) else 0
            for i in range(completed_spreads, len(interval_records)):
                raise_if_cancelled(project)
                interval = interval_records[i]
                candidate_samples = [
                    Sample(**sample) for sample in interval["candidates"]
                ]
                spread_id = f"spread_{i + 1:04d}"
                spread_roi = validate_roi(tracking_roi).tolist()
                roi_tracking = {
                    "tracked": False,
                    "status": "reference" if tracking_image is None else "previous_roi",
                    "roi": spread_roi,
                    "step_shift": 0.0,
                    "total_shift": float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(spread_roi, dtype=np.float32)
                                - np.asarray(manifest["roi"], dtype=np.float32),
                                axis=1,
                            )
                        )
                    ),
                }
                if cfg.roi_tracking and tracking_image is not None and candidate_samples:
                    tracking_probe = extract_frame(
                        manifest["source"],
                        candidate_samples[0].time,
                        cfg.analysis_width,
                        cfg.hwaccel,
                    )
                    roi_tracking = track_spread_roi(
                        tracking_image,
                        tracking_probe,
                        tracking_roi,
                        manifest["roi"],
                        max_step=cfg.roi_tracking_max_step,
                        max_total=cfg.roi_tracking_max_total,
                    )
                    spread_roi = roi_tracking["roi"]

                records = []
                for j, sample in enumerate(candidate_samples):
                    raise_if_cancelled(project)
                    records.append(
                        candidate(
                            project,
                            manifest,
                            cfg,
                            detector,
                            spread_id,
                            j,
                            sample,
                            base_roi=spread_roi,
                        )
                    )
                _augment_temporal_hand_masks(project, records, cfg)
                selection_mode = cfg.candidate_selection_mode if cfg.output_layout == "split" else "spread"
                selected, selected_pages = choose_candidate_selection(records, selection_mode)
                initial_selected = selected
                initial_selected_pages = dict(selected_pages)
                reasons = (
                    fallback_reasons(records, selection_mode)
                    if cfg.auto_high_fps_fallback
                    else []
                )
                added, effective_fps = ([], float(manifest.get("analysis_fps") or cfg.video_sample_fps))
                if reasons:
                    added, effective_fps = _add_auto_high_fps_candidates(
                        project,
                        manifest,
                        cfg,
                        detector,
                        spread_id,
                        records,
                        interval["start"],
                        interval["end"],
                        reasons,
                        base_roi=spread_roi,
                    )
                    if added:
                        selected, selected_pages = choose_candidate_selection(
                            records,
                            selection_mode,
                        )

                # Candidate scoring v2 is relative to the final candidate pool.
                # Persist both normal and automatically rescanned candidates.
                for record in records:
                    write_json(project / Path(record["path"]).with_suffix(".json"), record)
                    score_rows.append(_score_row(spread_id, record))

                spread = {
                    "id": spread_id,
                    "start": interval["start"],
                    "end": interval["end"],
                    "candidates": records,
                    "selected": selected,
                    "selected_pages": selected_pages,
                    "candidate_selection_mode": cfg.candidate_selection_mode,
                    "tracked_roi": validate_roi(spread_roi).tolist(),
                    "roi_tracking": roi_tracking,
                    "extra_suspect": [],
                }
                if reasons:
                    spread["auto_high_fps_fallback"] = {
                        "trigger_reasons": reasons,
                        "requested_fps": float(cfg.auto_high_fps_fallback_fps),
                        "effective_fps": float(effective_fps),
                        "added": len(added),
                        "candidate_ids": [record["id"] for record in added],
                        "selection_changed": (
                            selected != initial_selected
                            or selected_pages != initial_selected_pages
                        ),
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
                raise_if_cancelled(project)
                pages = render_spread(project, manifest, spread)
                selected_tracking = _candidate_by_id(spread, spread["selected"])
                next_tracking = cv2.imread(
                    str(project / selected_tracking["path"]),
                    cv2.IMREAD_COLOR,
                )
                if next_tracking is not None:
                    tracking_image = next_tracking
                    tracking_roi = validate_roi(
                        selected_tracking.get("roi", spread_roi)
                    ).tolist()
                manifest["spreads"].append(spread)
                manifest["pages"].extend(pages)
                _write_score_rows(project, score_rows)
                manifest["processing_checkpoint"] = {
                    "motion_analysis_complete": True,
                    "completed_spreads": i + 1,
                }
                update(
                    project,
                    manifest,
                    0.4 + 0.55 * (i + 1) / len(interval_records),
                    (
                        f"候補評価・補正 {i + 1} / "
                        f"{len(interval_records)} 見開き"
                    ),
                )
                raise_if_cancelled(project)

            raise_if_cancelled(project)
            update(project, manifest, 0.97, "PDF / CBZを生成中")
            build_exports(project, manifest)
            manifest.pop("processing_checkpoint", None)
            clear_cancel_request(project)
            manifest.update(
                status="complete",
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            refresh_review_safety(manifest)
            update(project, manifest, 1, "完了 — 要確認ページを確認してください")
            return manifest
        except ProcessingCancelled:
            clear_cancel_request(project)
            manifest.pop("error", None)
            manifest.update(
                status="cancelled",
                message="処理を中断しました。続きから再開できます",
                pdf_stale=True,
            )
            save_manifest(project, manifest)
            LOG.info("Processing cancelled at a safe checkpoint")
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


def import_external_page(project, image_path, page_id=None, display_name=None):
    """Add or replace one final page using a local external image."""
    project = Path(project).resolve()
    image_path = Path(image_path).expanduser().resolve(strict=True)
    if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("Unsupported image format")

    with project_lock(project):
        manifest = read_manifest(project)
        cfg = Config.from_dict(manifest["config"])
        rgb = load_bounded_rgb_image(image_path, cfg)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        before = _page_review_state(manifest, include_pages=True)
        imported = project / "source" / "external_pages"
        imported.mkdir(parents=True, exist_ok=True)
        serial = len(list(imported.glob("external_*"))) + 1
        original_path = imported / f"external_{serial:04d}.png"
        save_image(original_path, image)

        ext = ".jpg" if cfg.image_format == "jpeg" else ".png"
        output_path = project / "pages" / f"external_{serial:04d}{ext}"
        processed = image
        if cfg.grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        save_image(output_path, processed, quality=cfg.jpeg_quality)
        relative = str(output_path.relative_to(project))
        page = {
            "id": f"external_{serial:04d}",
            "spread_id": None,
            "side": "external",
            "enabled": True,
            "suspect": [],
            "path": relative,
            "preview": relative,
            "source": "external_image",
            "source_image": str(original_path.relative_to(project)),
            "external_name": display_name or image_path.name,
        }

        if page_id:
            index = next((i for i, item in enumerate(manifest["pages"]) if item["id"] == page_id), None)
            if index is None:
                raise ValueError("Unknown page")
            page["id"] = manifest["pages"][index]["id"]
            page["replaces"] = page_id
            manifest["pages"][index] = page
            label = "外部画像でページ差し替え"
        else:
            manifest["pages"].append(page)
            label = "外部画像ページを追加"

        _push_page_history(manifest, before, label)
        manifest["pdf_stale"] = True
        _refresh_adjacent_final_quality(project, manifest, cfg)
        save_manifest(project, manifest)
        return manifest



def _rescan_page_candidates(project, manifest, cfg, page_id, radius=1.0, requested_fps=60.0):
    """Add high-density candidates around one reviewed page without rescanning the book."""
    page = next((item for item in manifest.get("pages", []) if item["id"] == page_id), None)
    if page is None:
        raise ValueError("Unknown page")
    if page.get("side") in ("cover", "external") or not page.get("spread_id"):
        raise ValueError("High-fps rescan is only available for video-backed pages")

    spread = next(
        (item for item in manifest.get("spreads", []) if item["id"] == page["spread_id"]),
        None,
    )
    if spread is None:
        raise ValueError("Page spread is missing")

    radius = float(radius)
    requested_fps = float(requested_fps)
    if not math.isfinite(radius) or not 0.25 <= radius <= 3.0:
        raise ValueError("Rescan radius must be 0.25..3.0 seconds")
    if not math.isfinite(requested_fps) or not 10 <= requested_fps <= 120:
        raise ValueError("Rescan fps must be 10..120")

    selected_id = page.get("candidate_id")
    if selected_id is None:
        selected_pages = spread.get("selected_pages") or {}
        selected_id = selected_pages.get(page.get("side"), spread.get("selected"))
    selected = _candidate_by_id(spread, selected_id)
    center = float(selected["time"])

    source_fps = float(manifest.get("metadata", {}).get("fps") or requested_fps)
    effective_fps = min(requested_fps, source_fps) if source_fps > 0 else requested_fps
    duration = float(manifest["metadata"]["duration"])
    start = max(0.0, center - radius)
    end = min(duration, center + radius)

    height = int(manifest["metadata"]["display_height"])
    width = int(manifest["metadata"]["display_width"])
    analysis_width = min(cfg.analysis_width, width)
    size = (
        analysis_width,
        max(2, round(height * analysis_width / width)),
    )

    samples = []
    previous = None
    stream = sample_frames(
        manifest["source"],
        effective_fps,
        size,
        cfg.hwaccel,
        start_time=start,
    )
    try:
        for index, timestamp, frame in stream:
            if timestamp > end + (0.5 / effective_fps):
                break
            cropped = warp_roi(frame, manifest["roi"])
            motion = motion_score(previous, cropped) if previous is not None else 1.0
            samples.append(Sample(index, timestamp, motion, sharpness(cropped)))
            previous = cropped
    finally:
        stream.close()

    if not samples:
        raise ValueError("No frames found in the rescan window")

    limit = max(4, min(8, int(cfg.candidates_per_spread)))
    picked = choose_candidates(samples, limit)
    existing_times = [float(item["time"]) for item in spread.get("candidates", [])]
    duplicate_tolerance = 0.5 / effective_fps
    picked = [
        sample
        for sample in picked
        if all(abs(sample.time - current) > duplicate_tolerance for current in existing_times)
    ]

    added = []
    if picked:
        next_id = max((int(item["id"]) for item in spread.get("candidates", [])), default=-1) + 1
        detector = HandDetector(cfg)
        try:
            for offset, sample in enumerate(picked):
                record = candidate(
                    project,
                    manifest,
                    cfg,
                    detector,
                    spread["id"],
                    next_id + offset,
                    sample,
                    base_roi=spread.get("tracked_roi"),
                )
                record["rescan"] = {
                    "center_time": center,
                    "radius": radius,
                    "requested_fps": requested_fps,
                    "effective_fps": effective_fps,
                }
                added.append(record)
        finally:
            detector.close()

        spread["candidates"].extend(added)
        _augment_temporal_hand_masks(project, spread["candidates"], cfg)
        selection_mode = (
            spread.get("candidate_selection_mode", cfg.candidate_selection_mode)
            if spread.get("output_layout", cfg.output_layout) == "split"
            else "spread"
        )
        choose_candidate_selection(spread["candidates"], selection_mode)
        for record in spread["candidates"]:
            write_json(project / Path(record["path"]).with_suffix(".json"), record)

        indices = [
            index
            for index, existing in enumerate(manifest["pages"])
            if existing.get("spread_id") == spread["id"]
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

    spread["candidate_rescan"] = {
        "center_time": center,
        "radius": radius,
        "requested_fps": requested_fps,
        "effective_fps": effective_fps,
        "added": len(added),
        "candidate_ids": [item["id"] for item in added],
    }
    manifest["message"] = (
        f"{page_id}: 高fps再探索で候補を{len(added)}件追加しました"
        if added
        else f"{page_id}: 高fps再探索で新しい候補は見つかりませんでした"
    )
    return len(added)


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
        if action == "book_metadata":
            metadata = normalize_book_metadata(params.get("metadata"))
            if metadata == manifest.get("book_metadata", {}):
                return manifest
            manifest["book_metadata"] = metadata
            manifest["pdf_stale"] = True
            manifest["message"] = "書籍メタデータを更新しました。PDF / CBZを再出力してください"
            save_manifest(project, manifest)
            return manifest
        if action == "expected_page_count":
            expected = normalize_expected_page_count(params.get("expected_page_count"))
            if expected == manifest.get("expected_page_count"):
                return manifest
            manifest["expected_page_count"] = expected
            refresh_review_safety(manifest)
            manifest["message"] = (
                "期待ページ数を解除しました"
                if expected is None
                else f"期待ページ数を {expected}ページに設定しました"
            )
            save_manifest(project, manifest)
            return manifest
        if action == "rescan_candidates":
            validate_manifest_video(manifest, cfg, require_dimensions=True)
            _rescan_page_candidates(
                project,
                manifest,
                cfg,
                params["page_id"],
                radius=params.get("radius", 1.0),
                requested_fps=params.get("fps", 60.0),
            )
            manifest["pdf_stale"] = True
            _refresh_adjacent_final_quality(project, manifest, cfg)
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
            validate_manifest_video(manifest, cfg, require_dimensions=True)
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
            validate_manifest_video(manifest, cfg, require_dimensions=True)
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
            if action != "swap":
                validate_manifest_video(manifest, cfg, require_dimensions=True)
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
            validate_manifest_video(manifest, cfg, require_dimensions=True)
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
