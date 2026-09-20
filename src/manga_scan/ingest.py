import math
import shutil
from pathlib import Path

from .config import Config
from .perspective import validate_roi
from .storage import project_lock, read_manifest, save_image, save_manifest, write_json
from .video import extract_frame, probe


def create_project(video, project, config=None, copy_source=False):
    config = (config or Config()).validate()
    metadata = probe(video)
    project = Path(project).expanduser().resolve()
    if project.exists() and any(project.iterdir()):
        raise ValueError("Project directory must be empty; choose a new directory")
    first = extract_frame(metadata["path"])
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
    warnings = (
        ["HDR input: MVP outputs 8-bit SDR without calibrated tone mapping; prefer SDR recording"]
        if metadata["hdr"]
        else []
    )
    manifest = {
        "version": 2,
        "source": source,
        "metadata": metadata,
        "config": config.to_dict(),
        "roi": None,
        "cover": {
            "status": "pending",
            "time": 0.0,
            "frame": "source/first_frame.png",
            "roi": None,
        },
        "reference": {
            "time": 0.0,
            "frame": "source/first_frame.png",
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
        save_image(project / path, image)

        if kind == "cover":
            cover = manifest.setdefault("cover", {})
            cover.update(status="frame_selected" if confirm else "pending", time=timestamp, frame=path)
            cover["roi"] = None
            manifest["message"] = (
                "表紙の外周を4点で指定してください"
                if confirm
                else "表紙にするフレームを選んでください"
            )
        else:
            reference = manifest.setdefault("reference", {})
            reference.update(time=timestamp, frame=path, confirmed=bool(confirm))
            manifest["roi"] = None
            manifest["message"] = (
                "見開きの外周を4点で指定してください"
                if confirm
                else "基準にする見開きフレームを選んでください"
            )
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
        manifest["message"] = "基準にする見開きフレームを選んでください"
        save_manifest(project, manifest)
        return manifest
