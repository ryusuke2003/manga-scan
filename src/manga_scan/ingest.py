import math
import shutil
from pathlib import Path

import cv2

from .config import Config
from .cover_detect import detect_cover_quad
from .perspective import rotate_roi, validate_roi
from .rotation_detection import detect_video_rotation
from .split import rotate_image
from .spread_detect import detect_reference_spread, draw_reference_spread
from .storage import project_lock, read_manifest, save_image, save_manifest, write_json
from .video import extract_frame, probe


def create_project(video, project, config=None, copy_source=False):
    config = (config or Config()).validate()
    metadata = probe(video)
    project = Path(project).expanduser().resolve()
    if project.exists() and any(project.iterdir()):
        raise ValueError("Project directory must be empty; choose a new directory")
    first = extract_frame(metadata["path"])
    # A non-zero rotation was already a meaningful manual override before
    # auto-detection existed. Preserve that behavior for direct Config users.
    if config.auto_rotation and config.rotation:
        config.auto_rotation = False
    if config.auto_rotation:
        rotation_detection = detect_video_rotation(
            metadata["path"],
            metadata,
            first,
            hwaccel=config.hwaccel,
        )
        config.rotation = rotation_detection["rotation"]
    else:
        rotation_detection = {
            "rotation": config.rotation,
            "confidence": 1.0,
            "source": "manual",
            "scores": {str(config.rotation): 1.0},
            "sample_times": [0.0],
        }
    config.validate()
    rotation_detection["confirmed"] = not config.auto_rotation
    project.mkdir(parents=True, exist_ok=True)
    for folder in ("source", "frames_lowres", "candidates", "selected", "pages", "debug", "output"):
        (project / folder).mkdir(exist_ok=True)
    if copy_source:
        destination = project / "source" / ("video" + Path(metadata["path"]).suffix.lower())
        shutil.copy2(metadata["path"], destination)
        source = str(destination)
    else:
        source = metadata["path"]
    config.hand_model = str(Path(config.hand_model).expanduser().resolve())
    metadata["display_width"], metadata["display_height"] = first.shape[1], first.shape[0]
    save_image(project / "source/first_frame.png", first)
    save_image(
        project / "source/first_frame_preview.png",
        rotate_image(first, config.rotation),
    )
    warnings = (
        ["HDR input: MVP outputs 8-bit SDR without calibrated tone mapping; prefer SDR recording"]
        if metadata["hdr"]
        else []
    )
    if config.auto_rotation and rotation_detection["confidence"] < 0.65:
        warnings.append(
            "画像向きの自動判定に自信がありません。プレビューを確認し、必要なら向きを変更してください"
        )
    manifest = {
        "version": 2,
        "source": source,
        "metadata": metadata,
        "config": config.to_dict(),
        "rotation_detection": rotation_detection,
        "roi": None,
        "cover": {
            "status": "pending",
            "time": 0.0,
            "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png",
            "roi": None,
        },
        "reference": {
            "time": 0.0,
            "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png",
            "confirmed": False,
        },
        "status": "ready",
        "progress": 0,
        "message": "表紙を追加するか選んでください",
        "warnings": warnings,
        "spreads": [],
        "pages": [],
        "pdf_stale": True,
    }
    write_json(project / "source/video.json", metadata)
    write_json(project / "config.resolved.json", config.to_dict())
    save_manifest(project, manifest)
    return manifest


def _setup_time(manifest, value):
    timestamp = float(value)
    duration = float(manifest["metadata"]["duration"])
    if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
        raise ValueError("Timestamp outside video duration")
    return timestamp


def _detect_cover_for_rotation(image, rotation):
    displayed = rotate_image(image, rotation)
    detection = detect_cover_quad(displayed)
    detection["source"] = "auto"
    if detection["detected"]:
        roi = rotate_roi(detection["roi"], (-rotation) % 360).tolist()
        return detection, roi, "ready"
    return detection, None, "frame_selected"


def set_setup_frame(project, kind, time, confirm=False):
    if kind not in ("cover", "reference"):
        raise ValueError("Unknown setup frame kind")
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed")
        timestamp = _setup_time(manifest, time)
        cfg = Config.from_dict(manifest["config"])
        image = extract_frame(manifest["source"], timestamp, hwaccel=cfg.hwaccel)
        path = f"source/{kind}_frame.png"
        preview_path = f"source/{kind}_preview.png"
        save_image(project / path, image)
        save_image(project / preview_path, rotate_image(image, cfg.rotation))

        if kind == "cover":
            cover = manifest.setdefault("cover", {})
            detection = None
            roi = None
            status = "pending"
            if confirm:
                detection, roi, status = _detect_cover_for_rotation(image, cfg.rotation)
            cover.update(
                status=status,
                time=timestamp,
                frame=path,
                preview=preview_path,
                roi=roi,
                detection=detection,
            )
            manifest["message"] = (
                "表紙の外周を自動検出しました"
                if status == "ready"
                else "表紙の外周を自動検出できませんでした。4点で指定してください"
                if status == "frame_selected"
                else "表紙にするフレームを選んでください"
            )
        else:
            reference = manifest.setdefault("reference", {})
            reference_detection = None
            raw_roi = None
            debug_path = None
            if confirm:
                displayed = rotate_image(image, cfg.rotation)
                reference_detection = detect_reference_spread(
                    displayed,
                    min_confidence=cfg.page_contour_min_confidence,
                )
                reference_detection["source"] = "auto_pages"
                if reference_detection["detected"]:
                    raw_roi = rotate_roi(
                        reference_detection["roi"],
                        (-cfg.rotation) % 360,
                    ).tolist()
                debug_path = "source/reference_detection.png"
                save_image(
                    project / debug_path,
                    draw_reference_spread(displayed, reference_detection),
                )

            reference.update(
                time=timestamp,
                frame=path,
                preview=preview_path,
                confirmed=bool(confirm),
                detection=reference_detection,
                detection_preview=debug_path,
            )
            if confirm:
                rotation_detection = manifest.get("rotation_detection") or {}
                rotation_detection["confirmed"] = True
                manifest["rotation_detection"] = rotation_detection
                manifest["warnings"] = [
                    warning
                    for warning in manifest.get("warnings", [])
                    if not warning.startswith("画像向きの自動判定に自信がありません")
                ]
            manifest["roi"] = raw_roi
            manifest["message"] = (
                (
                    "見開き外周を自動検出しました。範囲を確認して抽出を開始してください"
                    if reference_detection and reference_detection["detected"]
                    else "見開き外周を自動検出できませんでした。4点で指定してください"
                )
                if confirm
                else "基準にする見開きフレームを選んでください"
            )
        save_manifest(project, manifest)
        return manifest


def _refresh_setup_previews(project, manifest, rotation):
    for key in ("cover", "reference"):
        item = manifest.get(key) or {}
        frame_path = item.get("frame")
        if not frame_path:
            continue
        frame = cv2.imread(str(project / frame_path))
        if frame is None:
            continue
        preview_path = item.get("preview") or f"source/{key}_preview.png"
        save_image(project / preview_path, rotate_image(frame, rotation))
        item["preview"] = preview_path


def _refresh_auto_cover_detection(project, manifest, cfg):
    cover = manifest.get("cover") or {}
    detection = cover.get("detection") or {}
    if detection.get("source") != "auto":
        return
    status = cover.get("status")
    # A frame_selected cover with an ROI means the user explicitly opened the
    # crop editor. Preserve that edit state instead of replacing it underneath them.
    if status not in ("ready", "frame_selected") or (
        status == "frame_selected" and cover.get("roi")
    ):
        return
    frame_path = cover.get("frame")
    if not frame_path:
        return
    frame = cv2.imread(str(project / frame_path))
    if frame is None:
        return

    detected, roi, status = _detect_cover_for_rotation(frame, cfg.rotation)
    cover["detection"] = detected
    cover["roi"] = roi
    cover["status"] = status
    manifest["message"] = (
        "基準にする見開きフレームを選んでください"
        if status == "ready"
        else "表紙の外周を自動検出できませんでした。4点で指定してください"
    )


def set_rotation(project, rotation):
    rotation = int(rotation)
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed")
        cfg = Config.from_dict(manifest["config"])
        cfg.rotation = rotation
        cfg.auto_rotation = False
        cfg.validate()
        manifest["config"] = cfg.to_dict()
        detection = manifest.setdefault("rotation_detection", {})
        detection.update(
            rotation=rotation,
            confidence=1.0,
            source="manual",
            confirmed=True,
        )
        manifest["warnings"] = [
            warning
            for warning in manifest.get("warnings", [])
            if not warning.startswith("画像向きの自動判定に自信がありません")
        ]
        _refresh_setup_previews(project, manifest, rotation)
        _refresh_auto_cover_detection(project, manifest, cfg)
        write_json(project / "config.resolved.json", cfg.to_dict())
        save_manifest(project, manifest)
        return manifest


def skip_cover(project):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed")
        cover = manifest.setdefault("cover", {})
        cover.update(status="skipped", roi=None)
        manifest["message"] = "基準にする見開きフレームを選んでください"
        save_manifest(project, manifest)
        return manifest


def set_cover_roi(project, roi):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed")
        cover = manifest.get("cover") or {}
        if cover.get("status") != "frame_selected":
            raise ValueError("Select a cover frame first")
        cover["roi"] = validate_roi(roi).tolist()
        cover["status"] = "ready"
        cover["detection"] = {"detected": False, "confidence": 0.0, "source": "manual"}
        manifest["message"] = "基準にする見開きフレームを選んでください"
        save_manifest(project, manifest)
        return manifest


def reopen_cover_roi(project):
    project = Path(project).resolve()
    with project_lock(project):
        manifest = read_manifest(project)
        if manifest["status"] == "complete":
            raise ValueError("Project already processed")
        cover = manifest.get("cover") or {}
        if cover.get("status") != "ready" or not cover.get("roi"):
            raise ValueError("No cover crop to edit")
        cover["status"] = "frame_selected"
        manifest["message"] = "表紙の外周を修正してください"
        save_manifest(project, manifest)
        return manifest
