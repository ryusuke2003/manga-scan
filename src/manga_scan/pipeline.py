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
from .hand import HandDetector
from .motion import Sample, StableDetector, choose_candidates, motion_score
from .page_contour import detect_page_quads, draw_page_quads
from .page_detect import refine_quad
from .page_warp import warp_detected_pages
from .perspective import validate_roi, warp_roi
from .score import score_frame, sharpness, suspect_reasons
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
    record = {
        "id": number,
        "time": sample.time,
        "path": f"{base}.png",
        "preview": f"{base}_spread.png",
        "hand_mask": f"{base}_hand_mask.png",
        "roi": roi,
        "metrics": metrics,
        "suspect": suspect_reasons(metrics, cfg, quad_ok),
    }
    write_json(project / f"{base}.json", record)
    return record


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


def render_spread(project, manifest, spread):
    cfg = Config.from_dict(manifest["config"])
    chosen = next(c for c in spread["candidates"] if c["id"] == spread["selected"])
    image = extract_frame(manifest["source"], chosen["time"], hwaccel=cfg.hwaccel)
    rectified = warp_roi(image, chosen["roi"])
    selected = f"selected/{spread['id']}.png"
    save_image(project / selected, rectified)
    spread["path"] = selected
    sides = rectify_spread_pages(
        project,
        image,
        rectified,
        chosen["roi"],
        spread,
        cfg,
    )
    spread["suspect"] = list(dict.fromkeys(chosen["suspect"] + spread.get("extra_suspect", [])))
    pages = []
    order = ["right", "left"] if cfg.reading_order == "rtl" else ["left", "right"]
    ext = "png" if cfg.image_format == "png" else "jpg"
    disabled_sides = set(spread.get("dewarp_disabled_sides", []))
    for side in order:
        source_page = sides[side]
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
                        dewarp_debug_grid(sides[side].shape, side, estimate["strength"]),
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
        # Review thumbnails avoid decoding all 4K pages in the browser.
        h, w = page_image.shape[:2]
        thumb = cv2.resize(page_image, (max(1, round(w * min(1, 480 / h))), min(480, h)))
        preview = f"pages/{spread['id']}_{side}_thumb.jpg"
        save_image(project / preview, thumb)
        page_suspect = spread["suspect"].copy()
        if dewarp.get("status") == "low_confidence":
            page_suspect.append("dewarp_low_confidence")
        pages.append(
            {
                "id": f"{spread['id']}_{side}",
                "spread_id": spread["id"],
                "side": side,
                "path": name,
                "preview": preview,
                "enabled": not bool(spread.get("duplicate_of")),
                "suspect": list(dict.fromkeys(page_suspect)),
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
                        }
                    )
                chosen = max(records, key=lambda c: c["metrics"]["score"])
                spread = {
                    "id": spread_id,
                    "start": segment[0].time,
                    "end": segment[-1].time,
                    "candidates": records,
                    "selected": chosen["id"],
                    "extra_suspect": [],
                }
                if i and typical_gap and gaps[i - 1] > typical_gap * cfg.interval_gap_factor:
                    spread["extra_suspect"].append("interval_gap")
                thumbnail = cv2.imread(str(project / chosen["preview"]))
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
                    spread["selected"] = selection
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
