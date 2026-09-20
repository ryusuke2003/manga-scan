import shutil
from pathlib import Path

from .config import Config
from .storage import save_image, save_manifest, write_json
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
        "version": 1,
        "source": source,
        "metadata": metadata,
        "config": config.to_dict(),
        "roi": None,
        "status": "ready",
        "progress": 0,
        "message": "ROIを指定してください",
        "warnings": warnings,
        "spreads": [],
        "pages": [],
        "pdf_stale": True,
    }
    write_json(project / "source/video.json", metadata)
    write_json(project / "config.resolved.json", config.to_dict())
    save_manifest(project, manifest)
    return manifest
