import cv2
import numpy as np
import pytest

from manga_scan.background_fill import detected_spread_mask, fill_page_background


def _detection():
    return {
        "detected": True,
        "confidence": 0.9,
        "left": {
            "quad": [[0.08, 0.08], [0.48, 0.12], [0.48, 0.88], [0.06, 0.92]],
            "detected": True,
            "confidence": 0.92,
        },
        "right": {
            "quad": [[0.52, 0.12], [0.92, 0.07], [0.94, 0.93], [0.52, 0.88]],
            "detected": True,
            "confidence": 0.9,
        },
    }


def test_detected_spread_mask_keeps_pages_and_gutter():
    mask = detected_spread_mask((200, 320, 3), _detection())

    assert mask[100, 80] == 255
    assert mask[100, 240] == 255
    assert mask[100, 160] == 255
    assert mask[3, 3] == 0


def test_paper_fill_changes_only_page_exterior_and_keeps_gutter():
    image = np.full((200, 320, 3), (55, 95, 140), np.uint8)
    mask = detected_spread_mask(image.shape, _detection())
    image[mask > 0] = (226, 229, 233)
    image[90:110, 154:166] = (70, 70, 70)
    before = image.copy()

    filled, info = fill_page_background(image, mask, mode="paper", paper_target=245)

    assert info["applied"]
    assert info["mode"] == "paper"
    assert np.mean(filled[3, 3]) > np.mean(before[3, 3])
    np.testing.assert_array_equal(filled[100, 80], before[100, 80])
    np.testing.assert_array_equal(filled[100, 160], before[100, 160])


def test_white_fill_is_explicit_and_preserve_is_noop():
    image = np.full((120, 180, 3), (40, 90, 140), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (20, 15), (160, 105), 255, -1)

    preserved, preserve_info = fill_page_background(image, mask, mode="preserve")
    np.testing.assert_array_equal(preserved, image)
    assert not preserve_info["applied"]

    white, white_info = fill_page_background(image, mask, mode="white", feather_px=0)
    assert white_info["applied"]
    np.testing.assert_array_equal(white[0, 0], np.array([255, 255, 255], np.uint8))
    np.testing.assert_array_equal(white[60, 90], image[60, 90])


def test_background_fill_rejects_unknown_mode():
    image = np.zeros((30, 40, 3), np.uint8)
    mask = np.ones((30, 40), np.uint8) * 255
    with pytest.raises(ValueError, match="page background fill"):
        fill_page_background(image, mask, mode="paint")
