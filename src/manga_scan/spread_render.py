"""Whole-spread rendering implementation used by pipeline.py."""

import cv2
import numpy as np

from .background_fill import detected_spread_mask, fill_page_background
from .final_quality import final_quality_checks
from .finger_repair import repair_finger_regions
from .glare import detect_glare_mask
from .hand import boundary_finger_mask
from .page_contour import draw_page_quads
from .perspective import warp_roi
from .pipeline_render_helpers import (
    _page_render_settings,
    _persist_finger_repair_component_debug,
    _union_occlusion_masks,
)
from .split import enhance_page, rotate_image
from .storage import save_image
from .video import extract_frame


def render_whole_spread(
    project,
    manifest,
    spread,
    cfg,
    *,
    detect_spread_page_consensus_fn,
    candidate_by_id_fn,
    whole_spread_geometry_fn,
    runtime_cache=None,
    extract_frame_fn=extract_frame,
    boundary_finger_mask_fn=boundary_finger_mask,
    repair_finger_regions_fn=repair_finger_regions,
):
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
        detect_spread_page_consensus_fn(
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
        record = candidate_by_id_fn(spread, candidate_id)
        source = extract_frame_fn(manifest["source"], record["time"], hwaccel=cfg.hwaccel)
        upright, roi, crop = whole_spread_geometry_fn(
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
            cached_runtime = (runtime_cache or {}).get(candidate_id) or {}
            saved = cached_runtime.get("hand_mask")
            if saved is None:
                saved = cv2.imread(
                    str(project / record["hand_mask"]),
                    cv2.IMREAD_GRAYSCALE,
                )
            if saved is not None:
                saved = cv2.resize(
                    saved, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                upright_mask = rotate_image(saved, cfg.rotation)
                hand_mask = warp_roi(
                    upright_mask, roi, interpolation=cv2.INTER_NEAREST
                )
                hand_mask |= boundary_finger_mask_fn(
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

    chosen = candidate_by_id_fn(spread, spread["selected"])
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
            page, repair, unresolved = repair_finger_regions_fn(
                page,
                mask,
                donors(),
                min_coverage=cfg.finger_repair_min_coverage,
                fallback=cfg.finger_repair_fallback,
                alignment_workers=cfg.processing_workers,
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
    if crop.get("boundary_refinement", {}).get("possible_inner_sheets"):
        suspect.append("page_quad_uncertain")
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
