import pytest

from manga_scan.config import Config
from manga_scan.input_validation import (
    validate_image_dimensions,
    validate_video_collection,
    validate_video_metadata,
)
from manga_scan.pipeline import run
from manga_scan.storage import save_manifest


def test_default_input_limits_allow_high_resolution_camera_media():
    cfg = Config().validate()

    validate_image_dimensions(8064, 6048, cfg)
    validate_video_metadata(
        {"width": 7680, "height": 4320, "duration": 3600},
        cfg,
    )


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (12000, 6000, "maximum is"),
        (13000, 1000, "maximum dimension"),
    ],
)
def test_image_dimensions_reject_resource_exhaustion(width, height, message):
    cfg = Config().validate()

    with pytest.raises(ValueError, match=message):
        validate_image_dimensions(width, height, cfg)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"width": 9000, "height": 4320, "duration": 60}, "maximum dimension"),
        ({"width": 7680, "height": 6000, "duration": 60}, "pixels/frame"),
        ({"width": 3840, "height": 2160, "duration": 14401}, "too long"),
    ],
)
def test_video_metadata_rejects_oversized_inputs(metadata, message):
    cfg = Config().validate()

    with pytest.raises(ValueError, match=message):
        validate_video_metadata(metadata, cfg)


def test_combined_video_duration_is_bounded():
    cfg = Config().validate()
    parts = [
        {"width": 1920, "height": 1080, "duration": 8000},
        {"width": 1920, "height": 1080, "duration": 8000},
    ]

    with pytest.raises(ValueError, match="Combined video duration is too long"):
        validate_video_collection(parts, cfg)


def test_processing_revalidates_legacy_project_before_video_decode(tmp_path):
    project = tmp_path / "legacy"
    project.mkdir()
    cfg = Config(hand_backend="none", finger_repair=False)
    save_manifest(
        project,
        {
            "source": "/tmp/oversized.mp4",
            "metadata": {
                "width": 9000,
                "height": 4320,
                "display_width": 9000,
                "display_height": 4320,
                "duration": 30,
                "fps": 30,
            },
            "config": cfg.to_dict(),
            "status": "ready",
            "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "pages": [],
            "spreads": [],
        },
    )

    with pytest.raises(ValueError, match="maximum dimension"):
        run(project)
