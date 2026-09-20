import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.perspective import warp_roi
from manga_scan.pipeline import rectify_spread_pages
from manga_scan.split import split_spread

REFERENCE = [[0.05, 0.07], [0.96, 0.07], [0.96, 0.94], [0.05, 0.94]]
LEFT = np.asarray([[90, 70], [490, 90], [470, 540], [70, 520]], dtype=np.float32)
RIGHT = np.asarray([[510, 90], [920, 60], [940, 520], [530, 545]], dtype=np.float32)


def synthetic_spread():
    image = np.full((600, 1000, 3), 45, dtype=np.uint8)
    cv2.fillConvexPoly(image, LEFT.astype(np.int32), (235, 235, 235))
    cv2.fillConvexPoly(image, RIGHT.astype(np.int32), (235, 235, 235))
    cv2.rectangle(image, (130, 130), (420, 300), (20, 20, 20), 6)
    cv2.rectangle(image, (560, 140), (860, 350), (20, 20, 20), 6)
    return image


def test_per_page_mode_connects_contour_detection_to_independent_warp(tmp_path):
    image = synthetic_spread()
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0001", "extra_suspect": []}
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)

    assert spread["perspective_mode_used"] == "per_page"
    assert spread["page_contours"]["detected"]
    assert spread["page_contours"]["left"]["detected"]
    assert spread["page_contours"]["right"]["detected"]
    assert "page_contour_low_confidence" not in spread["extra_suspect"]
    assert set(sides) == {"left", "right"}
    assert min(sides["left"].shape[:2]) > 100
    assert min(sides["right"].shape[:2]) > 100
    assert (tmp_path / spread["page_contour_debug"]).is_file()


def test_low_confidence_contours_fall_back_to_legacy_spread_split(tmp_path):
    image = np.full((400, 800, 3), 128, dtype=np.uint8)
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0002", "extra_suspect": []}
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)
    expected, expected_spine = split_spread(
        rectified,
        cfg.spine_ratio,
        cfg.split_mode,
        cfg.gutter_fraction,
    )

    assert spread["perspective_mode_used"] == "spread_fallback"
    assert not spread["page_contours"]["detected"]
    assert spread["spine_px"] == expected_spine
    assert "page_contour_low_confidence" in spread["extra_suspect"]
    np.testing.assert_array_equal(sides["left"], expected["left"])
    np.testing.assert_array_equal(sides["right"], expected["right"])
    assert (tmp_path / spread["page_contour_debug"]).is_file()


def test_default_spread_mode_is_pixel_compatible_and_skips_detection(tmp_path):
    image = synthetic_spread()
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0003", "extra_suspect": []}
    cfg = Config(hand_backend="none", finger_repair=False, perspective_mode="spread")
    expected, expected_spine = split_spread(
        rectified,
        cfg.spine_ratio,
        cfg.split_mode,
        cfg.gutter_fraction,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)

    assert spread["perspective_mode_used"] == "spread"
    assert spread["spine_px"] == expected_spine
    assert "page_contours" not in spread
    assert "page_contour_debug" not in spread
    np.testing.assert_array_equal(sides["left"], expected["left"])
    np.testing.assert_array_equal(sides["right"], expected["right"])


@pytest.mark.parametrize(
    "settings",
    [
        {"perspective_mode": "unknown"},
        {"page_contour_min_confidence": -0.1},
        {"page_contour_min_confidence": 1.1},
    ],
)
def test_per_page_integration_config_validation(settings):
    with pytest.raises(ValueError):
        Config.from_dict(settings)
