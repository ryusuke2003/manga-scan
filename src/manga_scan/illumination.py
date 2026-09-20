import cv2
import numpy as np

_MAX_MAP_SIDE = 512
_MIN_GAIN = 0.9
_MAX_GAIN = 1.4


def _validate_image(image):
    if image.dtype != np.uint8:
        raise ValueError("illumination correction expects uint8 images")
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] == 3:
        return
    raise ValueError("illumination correction expects grayscale or BGR images")


def _luminance(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _kernel_size(shape):
    minimum = min(shape)
    if minimum < 3:
        return 1
    size = max(3, int(round(minimum * 0.12)))
    if size % 2 == 0:
        size += 1
    maximum = min(81, minimum if minimum % 2 else minimum - 1)
    return max(3, min(size, maximum))


def estimate_illumination(image):
    """Estimate a smooth 0..255 illumination map without thresholding page content."""
    _validate_image(image)
    luminance = _luminance(image)
    height, width = luminance.shape

    scale = min(1.0, _MAX_MAP_SIDE / max(height, width))
    if scale < 1:
        small = cv2.resize(
            luminance,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = luminance

    kernel_size = _kernel_size(small.shape)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # Closing suppresses dark ink/halftone details before estimating the
        # low-frequency page illumination. It does not classify foreground.
        illumination = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
        sigma = max(1.0, kernel_size / 6)
        illumination = cv2.GaussianBlur(
            illumination, (0, 0), sigmaX=sigma, sigmaY=sigma
        )
    else:
        illumination = small.copy()

    if scale < 1:
        illumination = cv2.resize(
            illumination, (width, height), interpolation=cv2.INTER_LINEAR
        )
    return illumination.astype(np.float32, copy=False)


def correct_illumination(image, strength=0.7):
    """Normalize slow lighting/shadow variation while preserving local page detail."""
    _validate_image(image)
    if type(strength) not in (int, float) or not np.isfinite(strength):
        raise ValueError("illumination strength must be a finite number")
    if not 0 <= strength <= 1:
        raise ValueError("illumination strength must be between 0 and 1")
    if strength == 0:
        return image.copy()

    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0].astype(np.float32)
    else:
        lab = None
        luminance = image.astype(np.float32)

    illumination = estimate_illumination(image)
    reference = float(np.percentile(illumination, 90))
    gain = reference / np.maximum(illumination, 16.0)
    gain = np.clip(gain, _MIN_GAIN, _MAX_GAIN)
    gain = 1.0 + float(strength) * (gain - 1.0)
    corrected_luminance = np.clip(luminance * gain, 0, 255).astype(np.uint8)

    if lab is None:
        return corrected_luminance

    corrected_lab = lab.copy()
    corrected_lab[:, :, 0] = corrected_luminance
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
