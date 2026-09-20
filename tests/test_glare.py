import cv2
import numpy as np

from manga_scan.glare import detect_glare_mask, glare_overlap_fraction


def _textured_page(height=220, width=320):
    image = np.full((height, width, 3), 205, np.uint8)
    for x in range(20, width - 20, 18):
        cv2.line(image, (x, 20), (x, height - 20), (80, 80, 80), 1)
    for y in range(25, height - 20, 24):
        cv2.line(image, (20, y), (width - 20, y), (110, 110, 110), 1)
    return image


def test_glare_detector_finds_broad_specular_region():
    image = _textured_page()
    cv2.ellipse(image, (160, 105), (42, 28), 0, 0, 360, (255, 255, 255), -1)

    mask = detect_glare_mask(image)

    assert mask.dtype == np.uint8
    assert mask.shape == image.shape[:2]
    assert np.count_nonzero(mask) > 600
    assert mask[105, 160] == 255
    assert glare_overlap_fraction(mask) > 0.005


def test_glare_detector_ignores_uniform_white_paper():
    image = np.full((220, 320, 3), 248, np.uint8)

    mask = detect_glare_mask(image)

    assert not np.any(mask)


def test_glare_detector_ignores_thin_white_art_lines():
    image = np.full((220, 320, 3), 90, np.uint8)
    cv2.line(image, (30, 110), (290, 110), (255, 255, 255), 1)

    mask = detect_glare_mask(image)

    assert not np.any(mask)


def test_glare_detector_respects_roi_and_overlap_fraction():
    image = _textured_page()
    cv2.circle(image, (260, 100), 30, (255, 255, 255), -1)
    roi = [[0.0, 0.0], [0.55, 0.0], [0.55, 1.0], [0.0, 1.0]]

    mask = detect_glare_mask(image, roi)

    assert not np.any(mask[:, 200:])
    assert glare_overlap_fraction(mask, roi) == 0.0
