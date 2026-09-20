import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.illumination import correct_illumination, estimate_illumination
from manga_scan.split import enhance_page


def _gradient_page(width=360, height=240):
    page = np.full((height, width), 235, np.float32)
    shade = np.linspace(0.58, 1.0, width, dtype=np.float32)
    return np.clip(page * shade[None, :], 0, 255).astype(np.uint8)


def _side_gap(image, margin=60):
    return abs(float(image[:, :margin].mean()) - float(image[:, -margin:].mean()))


def test_gradient_shadow_is_flattened_in_grayscale():
    image = _gradient_page()
    corrected = correct_illumination(image, strength=0.7)
    illumination = estimate_illumination(image)

    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8
    assert illumination.shape == image.shape
    assert illumination.dtype == np.float32
    assert _side_gap(corrected) < _side_gap(image) * 0.6


def test_color_correction_uses_luminance_without_chroma_shift():
    height, width = 200, 300
    page = np.full((height, width, 3), (210, 225, 238), np.float32)
    shade = np.linspace(0.65, 1.0, width, dtype=np.float32)[None, :, None]
    image = np.clip(page * shade, 0, 255).astype(np.uint8)

    corrected = correct_illumination(image, strength=0.8)
    before_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)

    before_gap = _side_gap(before_lab[:, :, 0], margin=50)
    after_gap = _side_gap(after_lab[:, :, 0], margin=50)
    chroma_shift = np.mean(
        np.abs(after_lab[:, :, 1:].astype(np.int16) - before_lab[:, :, 1:].astype(np.int16))
    )

    assert after_gap < before_gap * 0.4
    assert chroma_shift < 1.0


def test_black_ink_and_halftone_contrast_are_preserved():
    height, width = 240, 360
    page = np.full((height, width), 235, np.float32)
    page[70:140, 80:140] = 20
    checker = ((np.indices((60, 80)).sum(axis=0) % 2) * 80 + 100).astype(np.float32)
    page[150:210, 180:260] = checker
    shade = np.linspace(0.58, 1.0, width, dtype=np.float32)
    image = np.clip(page * shade[None, :], 0, 255).astype(np.uint8)

    corrected = correct_illumination(image, strength=0.7)

    assert float(corrected[80:130, 90:130].mean()) < 35
    before_texture = float(image[160:200, 190:250].std())
    after_texture = float(corrected[160:200, 190:250].std())
    assert after_texture >= before_texture * 0.9


def test_disabled_or_zero_strength_is_pixel_compatible():
    image = np.random.default_rng(7).integers(0, 256, (80, 120, 3), dtype=np.uint8)

    np.testing.assert_array_equal(correct_illumination(image, strength=0), image)
    np.testing.assert_array_equal(
        enhance_page(image, illumination_correction=False, illumination_strength=1.0),
        enhance_page(image),
    )


def test_enhance_page_integrates_illumination_correction():
    image = _gradient_page()
    corrected = enhance_page(
        image,
        illumination_correction=True,
        illumination_strength=0.8,
    )
    assert _side_gap(corrected) < _side_gap(image) * 0.5


@pytest.mark.parametrize(
    "data",
    [
        {"illumination_correction": "true"},
        {"illumination_strength": -0.1},
        {"illumination_strength": 1.1},
    ],
)
def test_config_rejects_invalid_illumination_settings(data):
    with pytest.raises(ValueError):
        Config.from_dict(data)


def test_illumination_rejects_unsupported_images():
    with pytest.raises(ValueError):
        correct_illumination(np.zeros((20, 20), dtype=np.float32))
    with pytest.raises(ValueError):
        correct_illumination(np.zeros((20, 20, 4), dtype=np.uint8))
