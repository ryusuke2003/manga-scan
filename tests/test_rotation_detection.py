import cv2
import numpy as np
import pytest

import manga_scan.ingest as ingest_module
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


def test_rotation_detection_keeps_metadata_but_checks_page_geometry(monkeypatch):
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
        lambda *_args, **_kwargs: _spread_like_image(),
    )

    result = detect_video_rotation("unused.mov", metadata, _spread_like_image())

    assert result["rotation"] == 0
    assert result["source"] == "page_geometry"
    assert result["confidence"] < 1.0
    assert result["display_rotation"] == 90
    assert result["metadata_applied"] is True


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

    assert result["rotation"] == 0
    assert result["suggested_rotation"] in (90, 270)
    assert result["source"] == "page_geometry"
    assert len(result["sample_times"]) == 3
    scores = result["scores"]
    assert max(scores["90"], scores["270"]) > max(scores["0"], scores["180"])
    assert result["direction_ambiguous"]
    assert result["requires_confirmation"]
    assert result["confidence"] < 1.0
    assert result["rotation_options"] == [0, 90, 270]


def test_rotation_detection_warns_when_extra_samples_fail(monkeypatch):
    first = _spread_like_image()

    def fail_seek(*_args, **_kwargs):
        raise RuntimeError("seek failed")

    monkeypatch.setattr(rotation_module, "extract_frame", fail_seek)
    monkeypatch.setattr(rotation_module, "_confidence", lambda *_args: 0.95)

    result = detect_video_rotation("unused.mov", _metadata(), first)

    assert result["sample_times"] == [0.0]
    assert result["sample_count"] == 1
    assert result["confidence"] == 0.55


def test_rotation_detection_offers_180_when_page_geometry_is_nearly_symmetric(monkeypatch):
    first = _spread_like_image()
    monkeypatch.setattr(rotation_module, "extract_frame", lambda *_args, **_kwargs: first)

    result = detect_video_rotation("unused.mov", _metadata(), first)

    assert result["rotation"] == 0
    assert result["requires_confirmation"]
    assert result["rotation_options"] == [0, 180]


def _rotation_project(tmp_path, cover):
    project = tmp_path / "scan"
    (project / "source").mkdir(parents=True)
    frame = np.zeros((60, 120, 3), np.uint8)
    save_image(project / "source/cover_frame.png", frame)
    save_image(project / "source/cover_preview.png", frame)
    save_image(project / "source/reference_frame.png", frame)
    save_image(project / "source/reference_preview.png", frame)
    config = Config(
        auto_rotation=True,
        rotation=0,
        hand_backend="none",
        finger_repair=False,
    )
    manifest = {
        "status": "ready",
        "config": config.to_dict(),
        "warnings": [],
        "rotation_detection": {
            "rotation": 0,
            "confidence": 0.5,
            "source": "page_geometry",
            "confirmed": False,
        },
        "cover": cover,
        "reference": {
            "frame": "source/reference_frame.png",
            "preview": "source/reference_preview.png",
            "confirmed": False,
        },
        "message": "基準にする見開きフレームを選んでください",
    }
    save_manifest(project, manifest)
    write_json(project / "config.resolved.json", config.to_dict())
    return project


def test_manual_rotation_redetects_automatic_cover(tmp_path, monkeypatch):
    old_roi = [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]]
    project = _rotation_project(
        tmp_path,
        {
            "status": "ready",
            "frame": "source/cover_frame.png",
            "preview": "source/cover_preview.png",
            "roi": old_roi,
            "detection": {"detected": True, "confidence": 0.8, "source": "auto"},
        },
    )
    detected_roi = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    seen = {}

    def detect(image):
        seen["shape"] = image.shape[:2]
        return {"detected": True, "confidence": 0.9, "roi": detected_roi}

    monkeypatch.setattr(ingest_module, "detect_cover_quad", detect)

    updated = set_rotation(project, 90)

    assert seen["shape"] == (120, 60)
    assert updated["cover"]["status"] == "ready"
    assert updated["cover"]["detection"]["source"] == "auto"
    expected = ingest_module.rotate_roi(detected_roi, 270).tolist()
    np.testing.assert_allclose(updated["cover"]["roi"], expected)


def test_manual_rotation_retries_previously_failed_auto_cover(tmp_path, monkeypatch):
    project = _rotation_project(
        tmp_path,
        {
            "status": "frame_selected",
            "frame": "source/cover_frame.png",
            "preview": "source/cover_preview.png",
            "roi": None,
            "detection": {"detected": False, "confidence": 0.3, "source": "auto"},
        },
    )
    detected_roi = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    monkeypatch.setattr(
        ingest_module,
        "detect_cover_quad",
        lambda _image: {"detected": True, "confidence": 0.9, "roi": detected_roi},
    )

    updated = set_rotation(project, 90)

    assert updated["cover"]["status"] == "ready"
    assert updated["cover"]["detection"]["source"] == "auto"
    assert updated["cover"]["roi"] is not None
    assert updated["message"] == "基準にする見開きフレームを選んでください"


def test_manual_rotation_preserves_user_edited_cover_crop(tmp_path, monkeypatch):
    roi = [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]]
    project = _rotation_project(
        tmp_path,
        {
            "status": "ready",
            "frame": "source/cover_frame.png",
            "preview": "source/cover_preview.png",
            "roi": roi,
            "detection": {"detected": False, "confidence": 0.0, "source": "manual"},
        },
    )
    monkeypatch.setattr(
        ingest_module,
        "detect_cover_quad",
        lambda _image: pytest.fail("manual crop must not be replaced"),
    )

    updated = set_rotation(project, 90)

    assert updated["cover"]["roi"] == roi
    assert updated["cover"]["detection"]["source"] == "manual"


def test_manual_rotation_returns_failed_auto_cover_to_manual_crop(tmp_path, monkeypatch):
    roi = [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]]
    project = _rotation_project(
        tmp_path,
        {
            "status": "ready",
            "frame": "source/cover_frame.png",
            "preview": "source/cover_preview.png",
            "roi": roi,
            "detection": {"detected": True, "confidence": 0.8, "source": "auto"},
        },
    )
    monkeypatch.setattr(
        ingest_module,
        "detect_cover_quad",
        lambda _image: {"detected": False, "confidence": 0.3, "roi": None},
    )

    updated = set_rotation(project, 90)

    assert updated["cover"]["status"] == "frame_selected"
    assert updated["cover"]["roi"] is None
    assert updated["cover"]["detection"]["source"] == "auto"
    assert "4点で指定" in updated["message"]


def test_manual_rotation_refreshes_setup_previews(tmp_path):
    project = tmp_path / "scan"
    (project / "source").mkdir(parents=True)
    frame = np.zeros((60, 120, 3), np.uint8)
    frame[:, :60] = (20, 40, 220)
    frame[:, 60:] = (220, 80, 20)
    save_image(project / "source/first_frame.png", frame)
    save_image(project / "source/first_frame_preview.png", frame)
    config = Config(
        auto_rotation=True,
        rotation=0,
        hand_backend="none",
        finger_repair=False,
    )
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
