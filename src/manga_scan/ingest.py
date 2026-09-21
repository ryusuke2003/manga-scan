import math
import shutil
from pathlib import Path

import cv2

from .config import Config
from .cover_detect import detect_cover_quad
from .input_validation import validate_video_collection, validate_video_metadata
from .manifest_migrations import CURRENT_MANIFEST_VERSION
from .perspective import rotate_roi, validate_roi
from .quality_safety import normalize_expected_page_count
from .reference_candidates import scan_reference_candidates
from .rotation_detection import detect_video_rotation
from .split import rotate_image
from .spread_detect import detect_reference_spread_consensus, draw_reference_spread
from .storage import project_lock, read_manifest, save_image, save_manifest, write_json
from .video import extract_frame, local_video, probe


def _concat_escape(path):
    return str(path).replace("'", "'\\''")


def _prepare_video_source(videos, project, copy_source, config=None):
    if isinstance(videos, (str, Path)):
        videos = [videos]
    if not isinstance(videos, (list, tuple)) or not videos:
        raise ValueError("Select at least one video")
    paths = [local_video(video) for video in videos]
    if config is not None:
        source_metadatas = []
        for path in paths:
            metadata = probe(path)
            validate_video_metadata(
            metadata,
            config,
            label=path.name,
            require_dimensions=True,
        )
            source_metadatas.append(metadata)
        validate_video_collection(source_metadatas, config)
    project.mkdir(parents=True, exist_ok=True)
    (project / "source").mkdir(exist_ok=True)

    source_files = []
    if copy_source:
        for index, path in enumerate(paths, 1):
            destination = project / "source" / f"video_{index:03d}{path.suffix.lower()}"
            shutil.copy2(path, destination)
            source_files.append(str(destination))
    else:
        source_files = [str(path) for path in paths]

    if len(source_files) == 1:
        return source_files[0], source_files

    list_path = project / "source/input.ffconcat"
    lines = ["ffconcat version 1.0"]
    for path in source_files:
        lines.append(f"file '{_concat_escape(Path(path).resolve())}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(list_path), source_files


def create_project(
    video,
    project,
    config=None,
    copy_source=False,
    expected_page_count=None,
):
    config = (config or Config()).validate()
    project = Path(project).expanduser().resolve()
    if project.exists() and any(project.iterdir()):
        raise ValueError("Project directory must be empty; choose a new directory")
    source, source_files = _prepare_video_source(
        video,
        project,
        copy_source,
        config,
    )
    metadata = probe(source)
    validate_video_metadata(
        metadata,
        config,
        label="Combined video",
        require_dimensions=True,
    )
    first = extract_frame(metadata["path"])
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
    rotation_detection["confirmed"] = (
        not config.auto_rotation
        or not rotation_detection.get("requires_confirmation", False)
    )
    for folder in ("source", "frames_lowres", "candidates", "selected", "pages", "debug", "output"):
        (project / folder).mkdir(exist_ok=True)
    config.hand_model = str(Path(config.hand_model).expanduser().resolve())
    metadata["display_width"], metadata["display_height"] = first.shape[1], first.shape[0]
    metadata["source_count"] = len(source_files)
    metadata["source_files"] = source_files
    save_image(project / "source/first_frame.png", first)
    save_image(project / "source/first_frame_preview.png", rotate_image(first, config.rotation))
    warnings = (
        ["HDR input: MVP outputs 8-bit SDR without calibrated tone mapping; prefer SDR recording"]
        if metadata["hdr"]
        else []
    )
    if len(source_files) > 1:
        warnings.append(f"{len(source_files)}本の動画を撮影順に連結して解析します")
    if config.auto_rotation and (
        rotation_detection["confidence"] < 0.65
        or rotation_detection.get("requires_confirmation")
    ):
        warnings.append(
            "画像向きの自動判定に自信がありません。プレビューを確認し、必要なら向きを変更してください"
        )
    manifest = {
        "version": CURRENT_MANIFEST_VERSION,
        "source": source,
        "sources": source_files,
        "metadata": metadata,
        "config": config.to_dict(),
        "expected_page_count": normalize_expected_page_count(expected_page_count),
        "rotation_detection": rotation_detection,
        "roi": None,
        "cover": {
            "status": "pending", "time": 0.0, "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png", "roi": None,
        },
        "reference": {
            "time": 0.0, "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png", "confirmed": False,
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


def _refresh_reference_candidates(project, manifest, cfg):
    reference = manifest.setdefault("reference", {})
    if reference.get("confirmed"):
        return
    cover = manifest.get("cover") or {}
    if cover.get("status") not in ("ready", "skipped"):
        return
    if not manifest.get("source") or not manifest.get("metadata"):
        reference["candidates"] = []
        return
    validate_video_metadata(manifest["metadata"], cfg)

    start_time = float(cover.get("time", 0.0)) if cover.get("status") == "ready" else 0.0
    try:
        candidates = scan_reference_candidates(
            manifest["source"],
            manifest["metadata"],
            cfg,
            start_time=start_time,
            limit=5,
        )
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        reference["candidates"] = []
        reference["candidate_search_error"] = str(exc)
        return

    reference.pop("candidate_search_error", None)
    directory = project / "source/reference_candidates"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    public = []
    for index, candidate in enumerate(candidates, 1):
        record = {key: value for key, value in candidate.items() if key != "_frame"}
        preview_path = f"source/reference_candidates/candidate_{index:02d}.jpg"
        save_image(project / preview_path, candidate["_frame"], quality=88)
        record["preview"] = preview_path
        public.append(record)
    reference["candidates"] = public


def _detect_cover_for_rotation(image, rotation):
    displayed = rotate_image(image, rotation)
    detection = detect_cover_quad(displayed)
    detection["source"] = "auto"
    if detection["detected"]:
        roi = rotate_roi(detection["roi"], (-rotation) % 360).tolist()
        return detection, roi, "ready"
    return detection, None, "frame_selected"


def _reference_consensus_frames(source, metadata, timestamp, image, cfg):
    """Load the selected frame and its ±0.5s neighbors in display orientation."""

    duration = float(metadata["duration"])
    samples = [rotate_image(image, cfg.rotation)]
    sample_times = [float(timestamp)]
    for offset in (-0.5, 0.5):
        sample_time = min(
            max(float(timestamp + offset), 0.0),
            max(0.0, duration - 0.001),
        )
        if any(abs(sample_time - existing) < 0.001 for existing in sample_times):
            continue
        try:
            sample = extract_frame(source, sample_time, hwaccel=cfg.hwaccel)
        except (OSError, RuntimeError):
            continue
        samples.append(rotate_image(sample, cfg.rotation))
        sample_times.append(sample_time)
    return samples, 0


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
        validate_video_metadata(manifest["metadata"], cfg)
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
            if confirm and status == "ready":
                _refresh_reference_candidates(project, manifest, cfg)
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
                consensus_frames, anchor_index = _reference_consensus_frames(
                    manifest["source"],
                    manifest["metadata"],
                    timestamp,
                    image,
                    cfg,
                )
                displayed = consensus_frames[anchor_index]
                reference_detection = detect_reference_spread_consensus(
                    consensus_frames,
                    min_confidence=cfg.page_contour_min_confidence,
                    anchor_index=anchor_index,
                )
                reference_detection["method"] = "auto_pages"
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
                if not rotation_detection.get("requires_confirmation"):
                    rotation_detection["confirmed"] = True
                manifest["rotation_detection"] = rotation_detection
                if rotation_detection.get("confirmed"):
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
            direction_ambiguous=False,
            requires_confirmation=False,
            rotation_options=[rotation],
        )
        manifest["warnings"] = [
            warning
            for warning in manifest.get("warnings", [])
            if not warning.startswith("画像向きの自動判定に自信がありません")
        ]
        _refresh_setup_previews(project, manifest, rotation)
        _refresh_auto_cover_detection(project, manifest, cfg)
        _refresh_reference_candidates(project, manifest, cfg)
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
        cfg = Config.from_dict(manifest["config"])
        _refresh_reference_candidates(project, manifest, cfg)
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
        cfg = Config.from_dict(manifest["config"])
        _refresh_reference_candidates(project, manifest, cfg)
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
