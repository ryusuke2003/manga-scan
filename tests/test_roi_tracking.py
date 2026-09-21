import cv2
import numpy as np

from manga_scan.roi_tracking import track_spread_roi
from manga_scan.temporal_alignment import transform_normalized_quad


def _textured_frame():
    rng = np.random.default_rng(42)
    image = np.full((360, 540, 3), (45, 75, 110), np.uint8)
    for index in range(90):
        x = int(rng.integers(28, 512))
        y = int(rng.integers(28, 332))
        radius = int(rng.integers(3, 10))
        value = int(rng.integers(45, 245))
        color = (
            min(255, value + (index * 13) % 31),
            max(0, value - (index * 7) % 29),
            min(255, value + (index * 5) % 23),
        )
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
        if index % 4 == 0:
            cv2.line(
                image,
                (max(0, x - 13), min(359, y + 9)),
                (min(539, x + 11), max(0, y - 8)),
                (235, 235, 235),
                2,
            )
    cv2.putText(image, "BOOK 73", (155, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 3)
    return image


def test_roi_tracking_follows_conservative_book_translation():
    previous = _textured_frame()
    matrix = np.asarray(
        [[1.0, 0.0, 18.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
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
        [[1.0, 0.0, 18.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]],
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
        max_step=0.08,
        max_total=0.12,
    )

    assert not result["tracked"]
    assert result["status"] == "total_shift_too_large"
    np.testing.assert_allclose(result["roi"], previous_roi, atol=1e-6)
