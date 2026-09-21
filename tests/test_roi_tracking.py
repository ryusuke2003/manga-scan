import cv2
import numpy as np

from manga_scan.roi_tracking import track_spread_roi
from manga_scan.temporal_alignment import transform_normalized_quad


def _textured_frame():
    image = np.full((360, 540, 3), (45, 75, 110), np.uint8)
    for y in range(35, 335, 45):
        for x in range(35, 510, 55):
            color = ((x * 5) % 220 + 20, (y * 3) % 210 + 25, (x + y) % 190 + 35)
            cv2.circle(image, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.line(image, (x - 9, y + 10), (x + 11, y - 8), (235, 235, 235), 2)
    cv2.putText(image, "BOOK", (175, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 3)
    return image


def test_roi_tracking_follows_conservative_book_translation():
    previous = _textured_frame()
    matrix = np.asarray(
        [[1.0, 0.0, 28.0], [0.0, 1.0, 14.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    current = cv2.warpPerspective(
        previous,
        matrix,
        (previous.shape[1], previous.shape[0]),
        borderMode=cv2.BORDER_REFLECT101,
    )
    roi = np.asarray([[.12, .10], [.88, .10], [.88, .90], [.12, .90]], np.float32)

    result = track_spread_roi(
        previous,
        current,
        roi,
        roi,
        max_step=0.10,
        max_total=0.18,
    )
    expected = transform_normalized_quad(roi, matrix, previous.shape, current.shape)

    assert result["tracked"]
    assert result["status"] == "tracked"
    np.testing.assert_allclose(result["roi"], expected, atol=0.008)
    assert result["step_shift"] > 0


def test_roi_tracking_rejects_implausibly_large_cumulative_drift():
    previous = _textured_frame()
    matrix = np.asarray(
        [[1.0, 0.0, 55.0], [0.0, 1.0, 24.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    current = cv2.warpPerspective(
        previous,
        matrix,
        (previous.shape[1], previous.shape[0]),
        borderMode=cv2.BORDER_REFLECT101,
    )
    previous_roi = np.asarray([[.18, .12], [.94, .12], [.94, .92], [.18, .92]], np.float32)
    reference_roi = np.asarray([[.08, .10], [.84, .10], [.84, .90], [.08, .90]], np.float32)

    result = track_spread_roi(
        previous,
        current,
        previous_roi,
        reference_roi,
        max_step=0.12,
        max_total=0.14,
    )

    assert not result["tracked"]
    assert result["status"] in {"total_shift_too_large", "step_shift_too_large"}
    np.testing.assert_allclose(result["roi"], previous_roi, atol=1e-6)
