import cv2
import numpy as np
import pytest

import manga_scan.rotation_detection as rotation_module
from manga_scan.config import Config
from manga_scan.ingest import set_rotation
from manga_scan.rotation_detection import detect_video_rotation
from manga_scan.storage import read_manifest, save_image, save_manifest, write_json


def _spread_like_image(height=240, width=400):
    image = np.full((height, width, 3), 225, np.uint8)
    cv2.rectangle(image, (18, 20), (194, 220), (35, 35, 35), 3)
    cv2.rectangle(image, (206, 20), (382, 220), (35, 35, 35), 3)
    cv2.line(image, (200, 18), (200, 222), (10, 10, 10), 5)
    for y in range(55, 195, 28):
        cv2.line(image, (45, y), (165, y), (70, 70, 70), 3)
        cv2.line(image, (235, y + 5), (355, y + 5), (70, 70, 70), 3)
    return image


def _metadata(duration=10.0):
    return {
        "duration": duration,
        "raw": {"streams": [{}]},
    }


def test_rotation_detection_trusts_ffmpeg_display_metadata(monkeypatch):
    metadata = {
        "duration": 10.0,
        "raw": {
            "streams": [
                {"side_data_list": [{"rotation": 90}]}
            ]
        },
    }
    monkeypatch.setattr(
        rotation_module,
        "extract_frame",
        lambda *_args, **_kwargs: pytest.fail("metadata path should not seek extra frames"),
    )

    result = detect_video_rotation("unused.mov", metadata, _spread_like_image())

    assert result["rotation"] == 0
    assert result["source"] == "video_metadata"
    assert result["confidence"] == 1.0
    assert result["display_rotation"] == 90


@pytest.mark.parametrize(
    "source_rotate_code",
    [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE],
)
def test_rotation_detection_recovers_landscape_axis(monkeypatch, source_rotate_code):
    upright = _spread_like_image()
    sideways = cv2.rotate(upright, source_rotate_code)
    monkeypatch.setattr(
        rotation_module,
        "extract_frame",
        lambda *_args, **_kwargs: sideways.copy(),
    )

    result = detect_video_rotation("unused.mov", _metadata(), sideways)

    assert result["rotation"] in (90, 270)
    assert result["source"] == "page_geometry"
    assert len(result["sample_times"]) == 3
    scores = result["scores"]
    assert max(scores["90"], scores["270"]) > max(scores["0"], scores["180"])


def test_manual_rotation_refreshes_setup_previews(tmp_path):
    project = tmp_path / "scan"
    (project / "source").mkdir(parents=True)
    frame = np.zeros((60, 120, 3), np.uint8)
    frame[:, :60] = (20, 40, 220)
    frame[:, 60:] = (220, 80, 20)
    save_image(project / "source/first_frame.png", frame)
    save_image(project / "source/first_frame_preview.png", frame)
    config = Config(auto_rotation=True, rotation=0, hand_backend="none")
    manifest = {
        "status": "ready",
        "config": config.to_dict(),
        "warnings": ["画像向きの自動判定に自信がありません。プレビューを確認してください"],
        "rotation_detection": {
            "rotation": 0,
            "confidence": 0.5,
            "source": "page_geometry",
            "confirmed": False,
        },
        "cover": {
            "status": "pending",
            "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png",
        },
        "reference": {
            "frame": "source/first_frame.png",
            "preview": "source/first_frame_preview.png",
            "confirmed": False,
        },
    }
    save_manifest(project, manifest)
    write_json(project / "config.resolved.json", config.to_dict())

    updated = set_rotation(project, 90)

    assert updated["config"]["rotation"] == 90
    assert updated["config"]["auto_rotation"] is False
    assert updated["rotation_detection"]["source"] == "manual"
    assert updated["rotation_detection"]["confirmed"] is True
    assert not updated["warnings"]
    preview = cv2.imread(str(project / "source/first_frame_preview.png"))
    assert preview.shape[:2] == (120, 60)
    assert read_manifest(project)["config"]["rotation"] == 90
