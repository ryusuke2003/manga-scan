import cv2
import numpy as np

from manga_scan.cover_detect import (
    _boundary_line_count,
    _long_line_segments,
    detect_cover_quad,
)


def test_detect_cover_quad_finds_book_outline():
    image = np.full((720, 960, 3), (90, 145, 185), dtype=np.uint8)
    for y in range(30, 720, 45):
        cv2.line(image, (0, y), (959, y + 8), (75, 125, 165), 2)
    quad = np.asarray([[250, 70], [620, 90], [660, 660], [205, 640]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (55, 65, 145))
    cv2.polylines(image, [quad], True, (235, 235, 230), 6, cv2.LINE_AA)
    for y in range(160, 600, 70):
        cv2.line(image, (270, y), (610, y + 12), (30, 30, 40), 3)

    result = detect_cover_quad(image)

    assert result["detected"] is True
    assert result["confidence"] >= 0.62
    expected = quad.astype(np.float32) / [959, 719]
    np.testing.assert_allclose(result["roi"], expected, atol=0.035)


def test_detect_cover_quad_rejects_blank_frame():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)

    assert detect_cover_quad(image) == {
        "detected": False,
        "confidence": 0.0,
        "roi": None,
    }


def test_cover_boundary_penalty_counts_candidate_sides_not_all_corners():
    width, height = 1000, 800
    frame = np.asarray(
        [[0, 0], [999, 0], [999, 799], [0, 799]],
        dtype=np.float32,
    )
    inset = np.asarray(
        [[100, 80], [900, 80], [900, 720], [100, 720]],
        dtype=np.float32,
    )

    assert _boundary_line_count(frame, width, height) == 4
    assert _boundary_line_count(inset, width, height) == 0


def test_cover_detection_confidence_stays_in_probability_range():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (639, 479), (20, 20, 20), 6)

    result = detect_cover_quad(image)

    assert 0.0 <= result["confidence"] <= 1.0


def test_lsd_candidates_keep_long_segments_and_filter_short_content_lines():
    gray = np.full((300, 500), 180, np.uint8)
    cv2.line(gray, (30, 60), (470, 65), 20, 3, cv2.LINE_AA)
    cv2.line(gray, (220, 180), (250, 182), 20, 3, cv2.LINE_AA)

    segments = _long_line_segments(gray, width=500, height=300)

    assert segments
    assert max(segment[3] for segment in segments) > 0.8
    assert all(segment[3] * 500 >= 54 for segment in segments)
