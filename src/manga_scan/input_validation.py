import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} is missing or invalid")
    return value


def validate_image_dimensions(width, height, config, label="Image"):
    width = _positive_int(width, f"{label} width")
    height = _positive_int(height, f"{label} height")
    pixels = width * height
    if width > config.max_image_dimension or height > config.max_image_dimension:
        raise ValueError(
            f"{label} is too large: {width}x{height}; "
            f"maximum dimension is {config.max_image_dimension}px"
        )
    if pixels > config.max_image_pixels:
        raise ValueError(
            f"{label} is too large: {pixels:,} pixels; "
            f"maximum is {config.max_image_pixels:,} pixels"
        )
    return width, height


def load_bounded_rgb_image(path, config):
    path = Path(path).expanduser().resolve(strict=True)
    try:
        with Image.open(path) as source:
            validate_image_dimensions(
                source.width,
                source.height,
                config,
                label=path.name or "Image",
            )
            normalized = ImageOps.exif_transpose(source)
            validate_image_dimensions(
                normalized.width,
                normalized.height,
                config,
                label=path.name or "Image",
            )
            return np.array(normalized.convert("RGB"), copy=True)
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Invalid or unsafe image: {path.name}") from exc


def validate_video_metadata(
    metadata,
    config,
    label="Video",
    *,
    require_dimensions=False,
):
    try:
        duration = float(metadata["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} metadata is missing or invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{label} duration is missing or invalid")
    if duration > config.max_video_duration_seconds:
        raise ValueError(
            f"{label} is too long: {duration:.1f}s; "
            f"maximum is {config.max_video_duration_seconds:.1f}s"
        )

    width_value = metadata.get("width", metadata.get("display_width"))
    height_value = metadata.get("height", metadata.get("display_height"))
    if width_value is None or height_value is None:
        if require_dimensions:
            raise ValueError(f"{label} dimensions are missing or invalid")
        return metadata
    try:
        width = int(width_value)
        height = int(height_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} dimensions are missing or invalid") from exc

    _positive_int(width, f"{label} width")
    _positive_int(height, f"{label} height")
    pixels = width * height
    if width > config.max_video_dimension or height > config.max_video_dimension:
        raise ValueError(
            f"{label} resolution is too large: {width}x{height}; "
            f"maximum dimension is {config.max_video_dimension}px"
        )
    if pixels > config.max_video_pixels:
        raise ValueError(
            f"{label} resolution is too large: {pixels:,} pixels/frame; "
            f"maximum is {config.max_video_pixels:,} pixels/frame"
        )
    return metadata


def validate_video_collection(metadatas, config):
    total = sum(float(metadata["duration"]) for metadata in metadatas)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Total video duration is missing or invalid")
    if total > config.max_video_duration_seconds:
        raise ValueError(
            f"Combined video duration is too long: {total:.1f}s; "
            f"maximum is {config.max_video_duration_seconds:.1f}s"
        )
    return total
