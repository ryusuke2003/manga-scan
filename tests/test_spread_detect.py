import cv2
import numpy as np

from manga_scan.config import Config
from manga_scan.ingest import set_setup_frame
from manga_scan.perspective import rotate_roi
from manga_scan.spread_detect import detect_reference_spread
from manga_scan.storage import save_manifest


def _synthetic_spread():
    image = np.full((600, 1000, 3), (45, 85, 125), dtype=np.uint8)
    left = np.asarray([[80, 65], [490, 88], [475, 535], [60, 515]], dtype=np.int32)
    right = np.asarray([[510, 88], [930, 55], [950, 520], [525, 540]], dtype=np.int32)
    cv2.fillConvexPoly(image, left, (235, 235, 230))
    cv2.fillConvexPoly(image, right, (235, 235, 230))
    cv2.polylines(image, [left, right], True, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.rectangle(image, (125, 145), (420, 315), (35, 35, 35), 5)
    cv2.rectangle(image, (565, 135), (870, 350), (35, 35, 35), 5)
    return image


def test_detect_reference_spread_finds_two_pages_from_full_frame():
    result = detect_reference_spread(_synthetic_spread(), min_confidence=0.5)

    assert result["detected"]
    assert result["stage"] == "complete"
    assert result["pages"]["left"]["detected"]
    assert result["pages"]["right"]["detected"]
    assert result["confidence"] >= 0.5
    roi = np.asarray(result["roi"])
    assert roi.shape == (4, 2)
    assert roi[0, 0] < 0.15
    assert roi[1, 0] > 0.85


def test_detect_reference_spread_rejects_blank_frame():
    image = np.full((480, 800, 3), 180, dtype=np.uint8)

    result = detect_reference_spread(image)

    assert not result["detected"]
    assert result["roi"] is None
    assert result["stage"] == "outline"


def _occluded_mixed_color_spread():
    image = np.full((600, 1000, 3), (55, 95, 135), dtype=np.uint8)
    left = np.asarray([[105, 45], [495, 70], [480, 555], [90, 535]], dtype=np.int32)
    right = np.asarray([[515, 70], [900, 42], [930, 540], [520, 558]], dtype=np.int32)
    cv2.fillConvexPoly(image, left, (238, 238, 232))
    cv2.fillConvexPoly(image, right, (55, 75, 170))
    cv2.polylines(image, [left, right], True, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.rectangle(image, (585, 150), (820, 330), (210, 210, 205), 4)
    # Hands obscure parts of the lower outer boundary, like a real page turn.
    cv2.ellipse(image, (125, 500), (95, 70), -20, 0, 360, (150, 175, 215), -1)
    cv2.ellipse(image, (875, 490), (100, 85), 20, 0, 360, (145, 170, 210), -1)
    return image


def test_detect_reference_spread_uses_page_fallback_when_outer_outline_is_obscured(monkeypatch):
    monkeypatch.setattr(
        "manga_scan.spread_detect.detect_cover_quad",
        lambda *_args, **_kwargs: {"detected": False, "confidence": 0.24, "roi": None},
    )

    result = detect_reference_spread(_occluded_mixed_color_spread(), min_confidence=0.5)

    assert result["detected"]
    assert result["stage"] == "complete"
    assert result["source"] == "coarse_pages"
    assert result["pages"]["left"]["detected"]
    assert result["pages"]["right"]["detected"]
    roi = np.asarray(result["roi"])
    assert roi[0, 0] < 0.2
    assert roi[1, 0] > 0.8


def _setup_manifest(rotation):
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        rotation=rotation,
        auto_rotation=False,
    )
    return {
        "source": "/tmp/book.mp4",
        "metadata": {"duration": 10},
        "config": cfg.to_dict(),
        "rotation_detection": {
            "rotation": rotation,
            "confidence": 1.0,
            "source": "manual",
            "confirmed": True,
        },
        "roi": None,
        "cover": {"status": "skipped", "roi": None},
        "reference": {"confirmed": False},
        "status": "ready",
        "warnings": [],
        "pages": [],
        "spreads": [],
        "pdf_stale": True,
        "progress": 0,
        "message": "",
    }


def test_reference_confirmation_stores_auto_roi_in_source_coordinates(tmp_path, monkeypatch):
    frame = np.zeros((160, 300, 3), np.uint8)
    displayed_roi = [[0.10, 0.15], [0.90, 0.15], [0.90, 0.85], [0.10, 0.85]]
    save_manifest(tmp_path, _setup_manifest(90))
    monkeypatch.setattr(
        "manga_scan.ingest.extract_frame",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        "manga_scan.ingest.detect_reference_spread",
        lambda *_args, **_kwargs: {
            "detected": True,
            "confidence": 0.88,
            "roi": displayed_roi,
            "outline": {"detected": True, "confidence": 0.9, "roi": displayed_roi},
            "pages": {
                "detected": True,
                "confidence": 0.88,
                "left": {"quad": displayed_roi, "detected": True, "confidence": 0.9},
                "right": {"quad": displayed_roi, "detected": True, "confidence": 0.88},
            },
            "stage": "complete",
        },
    )

    manifest = set_setup_frame(tmp_path, "reference", 2.0, confirm=True)

    expected = rotate_roi(displayed_roi, 270)
    np.testing.assert_allclose(manifest["roi"], expected, atol=1e-6)
    assert manifest["reference"]["detection"]["detected"]
    assert manifest["reference"]["detection"]["confidence"] == 0.88
    assert manifest["reference"]["detection_preview"] == "source/reference_detection.png"
    assert (tmp_path / "source/reference_detection.png").is_file()
    assert "自動検出しました" in manifest["message"]


def test_reference_confirmation_falls_back_to_manual_points(tmp_path, monkeypatch):
    frame = np.zeros((160, 300, 3), np.uint8)
    save_manifest(tmp_path, _setup_manifest(0))
    monkeypatch.setattr(
        "manga_scan.ingest.extract_frame",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        "manga_scan.ingest.detect_reference_spread",
        lambda *_args, **_kwargs: {
            "detected": False,
            "confidence": 0.31,
            "roi": None,
            "outline": {"detected": False, "confidence": 0.31, "roi": None},
            "pages": None,
            "stage": "outline",
        },
    )

    manifest = set_setup_frame(tmp_path, "reference", 2.0, confirm=True)

    assert manifest["roi"] is None
    assert not manifest["reference"]["detection"]["detected"]
    assert "4点で指定" in manifest["message"]
