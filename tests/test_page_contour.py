import cv2
import numpy as np
import pytest

from manga_scan.page_contour import (
    consensus_page_quads,
    detect_page_quads,
    draw_page_quads,
    spread_quad_from_page_quads,
)

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


def test_small_valid_spread_roi_can_still_fall_back_per_page():
    image = np.full((400, 800, 3), 128, dtype=np.uint8)
    small_reference = [[0.45, 0.40], [0.56, 0.40], [0.56, 0.46], [0.45, 0.46]]

    result = detect_page_quads(image, small_reference)

    assert not result["detected"]
    assert len(result["left"]["quad"]) == 4
    assert len(result["right"]["quad"]) == 4


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


def test_paper_outline_tracks_outward_shift_without_cropping_edge_art():
    image = np.full((400, 700, 3), (40, 95, 155), np.uint8)
    cv2.rectangle(image, (95, 35), (605, 365), (220, 225, 230), -1)
    # Content outside the old crop must remain inside the new page quad.
    cv2.putText(image, "EDGE", (96, 100), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
    prior = [[.16, .12], [.84, .12], [.84, .88], [.16, .88]]
    result = detect_page_quads(image, prior)
    assert result["detected"]
    left = np.asarray(result["left"]["quad"]) * [699, 399]
    right = np.asarray(result["right"]["quad"]) * [699, 399]
    assert left[0, 0] < 100
    assert left[0, 1] < 40
    assert right[1, 0] > 600
    assert right[2, 1] > 360
    assert not result["left"]["touches_frame"]


def test_neutral_background_cannot_fabricate_an_expanded_paper_boundary():
    image = np.full((400, 700, 3), 220, np.uint8)
    result = detect_page_quads(image, REFERENCE)
    assert not result["detected"]


def test_paper_at_source_frame_edge_is_flagged():
    image = np.full((400, 700, 3), (40, 95, 155), np.uint8)
    cv2.rectangle(image, (95, 0), (605, 365), (220, 225, 230), -1)
    result = detect_page_quads(image, [[.16, .04], [.84, .04], [.84, .88], [.16, .88]])
    assert result["detected"]
    assert result["left"]["touches_frame"]
    assert result["right"]["touches_frame"]


def test_spread_quad_uses_only_outer_corners_from_both_pages():
    result = {
        "detected": True,
        "left": {"quad": [[.08, .10], [.48, .14], [.49, .88], [.06, .92]]},
        "right": {"quad": [[.51, .13], [.93, .08], [.96, .91], [.50, .87]]},
    }

    np.testing.assert_allclose(
        spread_quad_from_page_quads(result),
        [[.08, .10], [.93, .08], [.96, .91], [.06, .92]],
    )


def test_spread_quad_rejects_fallback_or_invalid_page_detection():
    with pytest.raises(ValueError, match="both page quads"):
        spread_quad_from_page_quads({"detected": False})
    with pytest.raises(ValueError, match="left/right quads"):
        spread_quad_from_page_quads({"detected": True, "left": {}, "right": {}})


def _page_detection(
    candidate_id,
    left_quad,
    right_quad,
    *,
    left_confidence=0.9,
    right_confidence=0.9,
    left_detected=True,
    right_detected=True,
):
    return {
        "candidate_id": candidate_id,
        "left": {
            "quad": np.asarray(left_quad, dtype=np.float32).tolist(),
            "confidence": left_confidence,
            "detected": left_detected,
            "touches_frame": False,
        },
        "right": {
            "quad": np.asarray(right_quad, dtype=np.float32).tolist(),
            "confidence": right_confidence,
            "detected": right_detected,
            "touches_frame": False,
        },
        "confidence": min(left_confidence, right_confidence),
        "detected": left_detected and right_detected,
    }


def test_consensus_page_quads_rejects_shifted_outlier():
    left = np.asarray([[.08, .10], [.48, .11], [.47, .90], [.07, .89]], np.float32)
    right = np.asarray([[.52, .11], [.92, .10], [.93, .89], [.53, .90]], np.float32)
    detections = []
    for candidate_id, shift in ((1, -0.002), (2, 0.0), (3, 0.002)):
        delta = np.asarray([shift, 0], np.float32)
        detections.append(
            _page_detection(candidate_id, left + delta, right + delta)
        )
    detections.append(
        _page_detection(
            99,
            left + np.asarray([0.12, 0], np.float32),
            right + np.asarray([0.12, 0], np.float32),
            left_confidence=0.99,
            right_confidence=0.99,
        )
    )

    result = consensus_page_quads(
        detections,
        min_confidence=0.5,
        max_corner_deviation=0.04,
        anchor_ids={"left": 2, "right": 2},
    )

    assert result["detected"]
    np.testing.assert_allclose(result["left"]["quad"], left, atol=0.003)
    np.testing.assert_allclose(result["right"]["quad"], right, atol=0.003)
    assert result["left"]["consensus_count"] == 3
    assert result["right"]["consensus_count"] == 3
    assert result["left"]["consensus_outlier_ids"] == [99]
    assert result["right"]["consensus_outlier_ids"] == [99]


def test_consensus_page_quads_can_recover_each_side_from_different_frames():
    left = np.asarray([[.08, .10], [.48, .11], [.47, .90], [.07, .89]], np.float32)
    right = np.asarray([[.52, .11], [.92, .10], [.93, .89], [.53, .90]], np.float32)
    fallback_left = left + np.asarray([0.03, 0], np.float32)
    fallback_right = right - np.asarray([0.03, 0], np.float32)

    result = consensus_page_quads(
        [
            _page_detection(
                1,
                left,
                fallback_right,
                left_detected=True,
                right_detected=False,
                right_confidence=0.2,
            ),
            _page_detection(
                2,
                fallback_left,
                right,
                left_detected=False,
                right_detected=True,
                left_confidence=0.2,
            ),
        ],
        min_confidence=0.5,
        anchor_ids={"left": 1, "right": 2},
    )

    assert result["detected"]
    np.testing.assert_allclose(result["left"]["quad"], left, atol=1e-6)
    np.testing.assert_allclose(result["right"]["quad"], right, atol=1e-6)
    assert result["left"]["consensus_candidate_ids"] == [1]
    assert result["right"]["consensus_candidate_ids"] == [2]
