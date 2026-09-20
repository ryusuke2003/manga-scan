"""Independent implementation of dHash and local-window SSIM."""

import cv2
import numpy as np


def gray_thumb(image, size=(256, 256)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def dhash(image):
    small = gray_thumb(image, (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def ssim(a, b):
    a, b = gray_thumb(a).astype(np.float64), gray_thumb(b).astype(np.float64)

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = np.maximum(0, blur(a * a) - mu_a * mu_a)
    var_b = np.maximum(0, blur(b * b) - mu_b * mu_b)
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    value = (
        (2 * mu_a * mu_b + c1)
        * (2 * cov + c2)
        / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
    )
    return float(value[5:-5, 5:-5].mean())


def compare(a, b, config):
    distance = (dhash(a) ^ dhash(b)).bit_count()
    similarity = ssim(a, b)
    # Also compare halves: a nearly blank spread must not hide a changed single page.
    halves = [
        ssim(x, y) for x, y in zip(np.array_split(a, 2, axis=1), np.array_split(b, 2, axis=1))
    ]
    informative = min(float(gray_thumb(a).std()), float(gray_thumb(b).std())) >= 8
    duplicate = (
        informative
        and distance <= config.duplicate_hash_distance
        and min(similarity, *halves) >= config.duplicate_ssim
    )
    suspect = similarity >= config.duplicate_suspect_ssim
    return {
        "duplicate": bool(duplicate),
        "suspect": bool(suspect),
        "hash_distance": distance,
        "ssim": similarity,
        "half_ssim": halves,
    }
