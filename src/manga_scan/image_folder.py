import re
from pathlib import Path

import cv2

from .config import Config
from .input_validation import load_bounded_rgb_image
from .final_quality import final_quality_checks
from .manifest_migrations import CURRENT_MANIFEST_VERSION
from .quality_safety import normalize_expected_page_count, refresh_review_safety
from .split import enhance_page, rotate_image
from .storage import save_image, save_manifest, write_json

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def image_files(folder, max_files=None):
    folder = Path(folder).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Select an image folder")
    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    if not files:
        raise ValueError("No PNG / JPEG / WebP images found in the selected folder")
    if max_files is not None and len(files) > max_files:
        raise ValueError(
            f"Too many images in folder: {len(files)}; maximum is {max_files}"
        )
    return folder, files


def load_image(path, config):
    rgb = load_bounded_rgb_image(path, config)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _render_image_page(project, image, source_path, display_name, index, cfg):
    raw_relative = f"source/images/image_{index:04d}.png"
    save_image(project / raw_relative, image)

    before_enhance = rotate_image(image, cfg.rotation)
    processed = enhance_page(
        image,
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
    final_quality = final_quality_checks(
        processed,
        before_enhance=before_enhance,
        dewarp={"mode": "off", "status": "disabled", "applied": False},
        white_normalization=cfg.white_normalization,
    )

    ext = "jpg" if cfg.image_format == "jpeg" else "png"
    page_relative = f"pages/image_{index:04d}.{ext}"
    save_image(project / page_relative, processed, quality=cfg.jpeg_quality)
    height, width = processed.shape[:2]
    thumb_width = max(1, round(width * min(1, 480 / max(1, height))))
    thumb_height = min(480, height)
    preview_relative = f"pages/image_{index:04d}_thumb.jpg"
    save_image(
        project / preview_relative,
        cv2.resize(processed, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA),
        quality=88,
    )

    return {
        "id": f"image_{index:04d}",
        "spread_id": None,
        "side": "external",
        "enabled": True,
        "suspect": list(final_quality["reasons"]),
        "path": page_relative,
        "preview": preview_relative,
        "source": raw_relative,
        "source_image": raw_relative,
        "source_kind": "image_folder",
        "external_name": display_name,
        "original_path": str(source_path),
        "final_quality": final_quality,
    }


def create_image_folder_project(
    folder,
    project,
    config=None,
    expected_page_count=None,
):
    cfg = (config or Config()).validate()
    project = Path(project).expanduser().resolve()
    if project.exists() and any(project.iterdir()):
        raise ValueError("Project directory must be empty; choose a new directory")

    folder, files = image_files(folder, cfg.max_image_files)
    for name in ("source", "source/images", "pages", "debug", "output"):
        (project / name).mkdir(parents=True, exist_ok=True)

    # Phone/camera EXIF orientation is normalized per image. A manual rotation
    # setting is still honored, but video-only automatic rotation is not needed.
    cfg.auto_rotation = False
    cfg.validate()
    cfg.hand_model = str(Path(cfg.hand_model).expanduser().resolve())

    pages = [
        _render_image_page(
            project,
            load_image(path, cfg),
            path,
            path.name,
            index,
            cfg,
        )
        for index, path in enumerate(files, 1)
    ]

    expected = normalize_expected_page_count(expected_page_count)
    manifest = {
        "version": CURRENT_MANIFEST_VERSION,
        "source_type": "image_folder",
        "source": str(folder),
        "sources": [str(path) for path in files],
        "metadata": {
            "source_count": len(files),
            "source_files": [str(path) for path in files],
            "image_count": len(files),
        },
        "config": cfg.to_dict(),
        "expected_page_count": expected,
        "rotation_detection": {
            "rotation": cfg.rotation,
            "confidence": 1.0,
            "source": "image_exif",
            "confirmed": True,
            "scores": {str(cfg.rotation): 1.0},
        },
        "roi": None,
        "cover": {"status": "skipped"},
        "reference": {"confirmed": True},
        "status": "complete",
        "progress": 1,
        "message": f"静止画 {len(files)}枚を読み込みました — ページ順と要確認を確認してください",
        "warnings": [],
        "spreads": [],
        "pages": pages,
        "pdf_stale": True,
    }
    refresh_review_safety(manifest)
    write_json(project / "source/images.json", {
        "folder": str(folder),
        "files": [str(path) for path in files],
    })
    write_json(project / "config.resolved.json", cfg.to_dict())
    save_manifest(project, manifest)
    return manifest
