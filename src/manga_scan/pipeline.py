import csv
import hashlib
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .dedupe import compare
from .export import contact_sheets, export_pdf
from .finger_repair import repair_finger_regions
from .hand import HandDetector
from .motion import Sample, StableDetector, choose_candidates, motion_score
from .page_contour import detect_page_quads, draw_page_quads
from .page_detect import refine_quad
from .page_warp import warp_detected_pages
from .perspective import validate_roi, warp_roi
from .score import score_frame, sharpness, suspect_reasons
from .selection import choose_candidate_selection, score_candidate_pages
from .split import auto_dewarp_page, dewarp_debug_grid, enhance_page, spine_position, split_spread
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
    save_image(project / f"{base}_spread.png", rectified)

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
        "preview": f"{base}_spread.png",
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
    if selected[0] == selected[1]:
        candidate_record = _candidate_by_id(spread, selected[0])
        preview = cv2.imread(str(project / candidate_record["preview"]))
        if preview is None:
            raise ValueError(f"Candidate preview missing: {candidate_record['preview']}")
        return preview

    physical_pages = []
    for side, candidate_id in zip(("left", "right"), selected):
        candidate_record = _candidate_by_id(spread, candidate_id)
        rectified = cv2.imread(str(project / candidate_record["preview"]))
        if rectified is None:
            raise ValueError(f"Candidate preview missing: {candidate_record['preview']}")
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
    if cfg.perspective_mode == "per_page":
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

    frame_height, frame_width = data["frame_shape"][:2]
    full_mask = cv2.resize(
        mask,
        (frame_width, frame_height),
        interpolation=cv2.INTER_NEAREST,
    )
    state = data["state"]
    if (
        state.get("perspective_mode_used") == "per_page"
        and (state.get("page_contours") or {}).get("detected")
    ):
        output_sizes = {
            name: (page.shape[1], page.shape[0])
            for name, page in data["sides"].items()
        }
        return warp_detected_pages(
            full_mask,
            state["page_contours"],
            output_sizes=output_sizes,
            interpolation=cv2.INTER_NEAREST,
        )[side]

    rectified_mask = warp_roi(
        full_mask,
        chosen["roi"],
        interpolation=cv2.INTER_NEAREST,
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
    return (
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


def render_spread(project, manifest, spread):
    cfg = Config.from_dict(manifest["config"])
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
        image = extract_frame(manifest["source"], chosen["time"], hwaccel=cfg.hwaccel)
        rectified = warp_roi(image, chosen["roi"])
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
        sides = rectify_spread_pages(project, image, rectified, chosen["roi"], state, cfg)
        cache[candidate_id] = {
            "chosen": chosen,
            "rectified": rectified,
            "sides": sides,
            "state": state,
            "frame_shape": image.shape,
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
                donors = []
                for donor_record in _finger_donor_candidates(
                    spread,
                    side,
                    selected_pages[side],
                )[:5]:
                    donor_data = load_candidate(donor_record["id"])
                    donor_mask = candidate_page_hand_mask(project, donor_data, side, cfg)
                    if donor_mask is None:
                        continue
                    donors.append(
                        {
                            "candidate_id": donor_record["id"],
                            "image": donor_data["sides"][side],
                            "mask": donor_mask,
                        }
                    )
                source_page, finger_repair, unresolved = repair_finger_regions(
                    source_page,
                    target_mask,
                    donors,
                    min_coverage=cfg.finger_repair_min_coverage,
                )
                finger_repair["target_mask"] = target_mask_path
                if np.any(unresolved):
                    unresolved_path = (
                        f"debug/finger_repair/{spread['id']}_{side}_unresolved.png"
                    )
                    save_image(project / unresolved_path, unresolved)
                    finger_repair["unresolved_mask"] = unresolved_path
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
                    rotation=cfg.rotation,
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
                        dewarp_debug_grid(data["sides"][side].shape, side, estimate["strength"]),
                    )
                    dewarp["debug_grid"] = grid

        page_image = enhance_page(
            source_page,
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
        if finger_repair["status"] == "complete":
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
                selected, selected_pages = choose_candidate_selection(
                    records, cfg.candidate_selection_mode
                )
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
            build_pdf(project, manifest)
            return manifest
        if action == "toggle_page":
            page = next(p for p in manifest["pages"] if p["id"] == params["page_id"])
            page["enabled"] = not page["enabled"]
        elif action == "move_page":
            index = next(i for i, p in enumerate(manifest["pages"]) if p["id"] == params["page_id"])
            destination = max(0, min(len(manifest["pages"]) - 1, index + int(params["delta"])))
            manifest["pages"].insert(destination, manifest["pages"].pop(index))
        elif action in ("select_candidate", "swap", "spine", "toggle_dewarp"):
            spread = next(s for s in manifest["spreads"] if s["id"] == params["spread_id"])
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
                if action == "spine":
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
            rec["suspect"].append("manual_frame_motion_unmeasured")
            spread = {
                "id": spread_id,
                "start": timestamp,
                "end": timestamp,
                "candidates": [rec],
                "selected": 0,
                "selected_pages": {"left": 0, "right": 0},
                "candidate_selection_mode": cfg.candidate_selection_mode,
                "extra_suspect": ["manual_frame"],
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
