import cv2
import numpy as np
import pytest

from manga_scan.page_dewarp import (
    dewarp_page,
    dewarp_page_auto,
    dewarp_profile,
    draw_dewarp_debug,
    estimate_page_curvature,
)


def flat_page():
    image = np.full((360, 260), 245, dtype=np.uint8)
    for y in range(45, 330, 30):
        cv2.line(image, (12, y), (247, y), 25, 2, cv2.LINE_AA)
    cv2.rectangle(image, (25, 72), (105, 138), 80, 2)
    cv2.rectangle(image, (145, 192), (235, 278), 100, 2)
    return image


def curve_page(image, side, displacement):
    h, w = image.shape[:2]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    outer = 0.0 if side == "left" else 1.0
    profile = displacement * (x - outer) ** 2
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) - profile[None, :]
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def line_waviness(image, expected_y):
    xs = np.linspace(20, image.shape[1] - 21, 25).astype(int)
    ys = []
    for x in xs:
        lo = max(0, expected_y - 24)
        hi = min(image.shape[0], expected_y + 25)
        ys.append(lo + int(np.argmin(image[lo:hi, x])))
    return float(np.std(ys))


@pytest.mark.parametrize("side", ["left", "right"])
def test_auto_dewarp_estimates_and_reduces_quadratic_gutter_curve(side):
    source = flat_page()
    curved = curve_page(source, side, 14.0)

    estimate = estimate_page_curvature(curved, side)

    assert estimate["detected"]
    assert estimate["confidence"] >= 0.45
    assert estimate["spine_displacement"] == pytest.approx(14.0, abs=5.0)

    corrected, meta = dewarp_page_auto(curved, side)
    assert meta["applied"]
    assert line_waviness(corrected, 165) < line_waviness(curved, 165) * 0.6


def test_flat_page_falls_back_without_modifying_pixels():
    image = flat_page()

    corrected, meta = dewarp_page_auto(image, "left")

    assert not meta["applied"]
    np.testing.assert_array_equal(corrected, image)


def test_dewarp_clamps_requested_displacement():
    image = flat_page()

    corrected, applied = dewarp_page(image, "left", 80.0, max_strength=0.03)

    assert corrected.shape == image.shape
    assert applied == pytest.approx(image.shape[0] * 0.03)


@pytest.mark.parametrize("side", ["left", "right"])
def test_profile_is_zero_at_outer_edge_and_maximal_at_spine(side):
    profile, applied = dewarp_profile((200, 100), side, 12.0)

    assert applied == 12.0
    if side == "left":
        assert profile[0] == pytest.approx(0)
        assert profile[-1] == pytest.approx(12)
    else:
        assert profile[-1] == pytest.approx(0)
        assert profile[0] == pytest.approx(12)


def test_debug_overlay_is_non_destructive():
    image = curve_page(flat_page(), "left", 12.0)
    original = image.copy()
    estimate = estimate_page_curvature(image, "left")

    debug = draw_dewarp_debug(image, estimate, "left")

    np.testing.assert_array_equal(image, original)
    assert debug.shape == (*image.shape, 3)
    assert np.count_nonzero(debug[:, :, 1] != image) > 0


@pytest.mark.parametrize("side", ["middle", "", None])
def test_invalid_side_is_rejected(side):
    with pytest.raises(ValueError):
        dewarp_profile((100, 100), side, 5.0)


@pytest.mark.parametrize("strength", [-0.1, 0.3])
def test_invalid_max_strength_is_rejected(strength):
    with pytest.raises(ValueError):
        dewarp_profile((100, 100), "left", 5.0, max_strength=strength)


def test_invalid_confidence_threshold_is_rejected():
    with pytest.raises(ValueError):
        dewarp_page_auto(flat_page(), "left", min_confidence=1.1)
