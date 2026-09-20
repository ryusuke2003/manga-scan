import cv2
import numpy as np
import pytest

from manga_scan.page_warp import (
    natural_page_size,
    validate_page_quad,
    warp_detected_pages,
    warp_page,
)

CANVAS_SHAPE = (520, 900)
LEFT_DST = np.asarray([[80, 65], [405, 95], [370, 465], [55, 440]], dtype=np.float32)
RIGHT_DST = np.asarray([[465, 90], [825, 55], [850, 450], [500, 470]], dtype=np.float32)


def page_pattern(seed):
    rng = np.random.default_rng(seed)
    page = np.full((240, 160, 3), 235, dtype=np.uint8)
    page[12:228, 12:148] = 220
    page[20:70, 20:70] = (20, 40, 200)
    page[20:70, 90:140] = (40, 190, 40)
    page[170:220, 20:70] = (180, 40, 40)
    page[170:220, 90:140] = (30, 30, 30)
    noise = rng.integers(-4, 5, page.shape, dtype=np.int16)
    return np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def normalized(points):
    h, w = CANVAS_SHAPE
    return (np.asarray(points, dtype=np.float32) / [w - 1, h - 1]).tolist()


def project_page(canvas, page, dst):
    h, w = page.shape[:2]
    src = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    projected = cv2.warpPerspective(page, matrix, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(
        np.full((h, w), 255, dtype=np.uint8),
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    canvas[mask > 0] = projected[mask > 0]


def synthetic_spread():
    canvas = np.full((*CANVAS_SHAPE, 3), 45, dtype=np.uint8)
    left = page_pattern(1)
    right = page_pattern(2)
    project_page(canvas, left, LEFT_DST)
    project_page(canvas, right, RIGHT_DST)
    detection = {
        "left": {"quad": normalized(LEFT_DST), "confidence": 0.9, "detected": True},
        "right": {"quad": normalized(RIGHT_DST), "confidence": 0.9, "detected": True},
    }
    return canvas, left, right, detection


def test_warp_page_recovers_full_rectangular_page():
    canvas, left, _, detection = synthetic_spread()

    corrected = warp_page(
        canvas,
        detection["left"]["quad"],
        output_size=(left.shape[1], left.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    assert corrected.shape == left.shape
    error = np.abs(corrected[3:-3, 3:-3].astype(int) - left[3:-3, 3:-3].astype(int))
    assert error.mean() < 8
    for y, x in ((30, 30), (30, 115), (195, 30), (195, 115)):
        np.testing.assert_allclose(corrected[y, x], left[y, x], atol=18)


def test_warp_detected_pages_corrects_left_and_right_independently():
    canvas, left, right, detection = synthetic_spread()

    pages = warp_detected_pages(
        canvas,
        detection,
        output_sizes={
            "left": (left.shape[1], left.shape[0]),
            "right": (right.shape[1], right.shape[0]),
        },
        interpolation=cv2.INTER_LINEAR,
    )

    assert set(pages) == {"left", "right"}
    assert pages["left"].shape == left.shape
    assert pages["right"].shape == right.shape
    np.testing.assert_allclose(pages["left"][35, 35], left[35, 35], atol=18)
    np.testing.assert_allclose(pages["right"][35, 35], right[35, 35], atol=18)


def test_natural_page_size_uses_longest_opposing_edges():
    quad = [[0.1, 0.1], [0.4, 0.1], [0.42, 0.8], [0.08, 0.8]]
    width, height = natural_page_size(quad, (201, 301, 3))

    assert width == pytest.approx(103, abs=2)
    assert height == pytest.approx(142, abs=2)


def test_grayscale_page_warp_is_supported():
    canvas, left, _, detection = synthetic_spread()
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    corrected = warp_page(
        gray_canvas,
        detection["left"]["quad"],
        output_size=(left.shape[1], left.shape[0]),
    )

    assert corrected.ndim == 2
    assert corrected.shape == left.shape[:2]


@pytest.mark.parametrize(
    "quad",
    [
        [[0, 0], [1, 1], [1, 0], [0, 1]],
        [[0.1, 0.1]] * 4,
        [[-0.1, 0.1], [0.4, 0.1], [0.4, 0.8], [0.1, 0.8]],
        [[0.1, 0.1], [0.4, 0.1], [float("nan"), 0.8], [0.1, 0.8]],
    ],
)
def test_validate_page_quad_rejects_invalid_geometry(quad):
    with pytest.raises(ValueError):
        validate_page_quad(quad)


@pytest.mark.parametrize("output_size", [(1, 100), (100, 1), (100.5, 200), "100x200"])
def test_warp_page_rejects_invalid_output_size(output_size):
    canvas, _, _, detection = synthetic_spread()
    with pytest.raises(ValueError):
        warp_page(canvas, detection["left"]["quad"], output_size=output_size)


def test_warp_detected_pages_requires_both_sides():
    canvas, _, _, detection = synthetic_spread()
    detection.pop("right")

    with pytest.raises(ValueError, match="right"):
        warp_detected_pages(canvas, detection)
