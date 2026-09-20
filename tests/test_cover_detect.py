import cv2
import numpy as np

from manga_scan.cover_detect import detect_cover_quad


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
