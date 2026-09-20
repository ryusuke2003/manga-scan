import cv2
import numpy as np
import pytest

from manga_scan.page_contour import detect_page_quads, draw_page_quads


REFERENCE = [[0.05, 0.07], [0.96, 0.07], [0.96, 0.94], [0.05, 0.94]]
LEFT = np.asarray([[90, 70], [490, 90], [470, 540], [70, 520]], dtype=np.float32)
RIGHT = np.asarray([[510, 90], [920, 60], [940, 520], [530, 545]], dtype=np.float32)


def synthetic_spread(page_value=235, desk_value=45):
    image = np.full((600, 1000, 3), desk_value, dtype=np.uint8)
    cv2.fillConvexPoly(image, LEFT.astype(np.int32), (page_value,) * 3)
    cv2.fillConvexPoly(image, RIGHT.astype(np.int32), (page_value,) * 3)

    # Large internal panels deliberately compete with the page boundary.
    ink = 20 if page_value > desk_value else 220
    cv2.rectangle(image, (130, 130), (420, 300), (ink,) * 3, 6)
    cv2.rectangle(image, (560, 140), (860, 350), (ink,) * 3, 6)
    return image


def normalized(points, shape=(600, 1000)):
    h, w = shape
    return np.asarray(points, dtype=np.float32) / [w - 1, h - 1]


@pytest.mark.parametrize(
    ("page_value", "desk_value"),
    [(235, 45), (35, 230)],
)
def test_detect_page_quads_finds_outer_pages_not_internal_panels(page_value, desk_value):
    image = synthetic_spread(page_value, desk_value)
    result = detect_page_quads(image, REFERENCE)

    assert result["detected"]
    assert result["confidence"] >= 0.5
    assert result["left"]["detected"]
    assert result["right"]["detected"]
    np.testing.assert_allclose(result["left"]["quad"], normalized(LEFT), atol=0.015)
    np.testing.assert_allclose(result["right"]["quad"], normalized(RIGHT), atol=0.015)


def test_detect_page_quads_falls_back_safely_when_no_edges_exist():
    image = np.full((400, 800, 3), 128, dtype=np.uint8)
    result = detect_page_quads(image, REFERENCE)

    assert not result["detected"]
    assert result["confidence"] == 0
    assert not result["left"]["detected"]
    assert not result["right"]["detected"]

    roi = np.asarray(REFERENCE, dtype=np.float32)
    top = (roi[0] + roi[1]) / 2
    bottom = (roi[3] + roi[2]) / 2
    expected_left = np.asarray([roi[0], top, bottom, roi[3]])
    expected_right = np.asarray([top, roi[1], roi[2], bottom])
    np.testing.assert_allclose(result["left"]["quad"], expected_left, atol=1e-6)
    np.testing.assert_allclose(result["right"]["quad"], expected_right, atol=1e-6)


def test_min_confidence_can_force_reference_fallback():
    image = synthetic_spread()
    detected = detect_page_quads(image, REFERENCE, min_confidence=0.5)
    fallback = detect_page_quads(image, REFERENCE, min_confidence=0.99)

    assert detected["detected"]
    assert not fallback["detected"]
    assert fallback["left"]["confidence"] > 0
    assert fallback["right"]["confidence"] > 0


def test_debug_overlay_marks_page_quads_without_modifying_input():
    image = synthetic_spread()
    original = image.copy()
    result = detect_page_quads(image, REFERENCE)

    overlay = draw_page_quads(image, result)

    np.testing.assert_array_equal(image, original)
    assert overlay.shape == image.shape
    assert np.count_nonzero(overlay != image) > 0


@pytest.mark.parametrize(
    ("spine_ratio", "min_confidence"),
    [(0.1, 0.5), (0.9, 0.5), (0.5, -0.1), (0.5, 1.1)],
)
def test_detect_page_quads_rejects_invalid_thresholds(spine_ratio, min_confidence):
    with pytest.raises(ValueError):
        detect_page_quads(
            synthetic_spread(),
            REFERENCE,
            spine_ratio=spine_ratio,
            min_confidence=min_confidence,
        )
