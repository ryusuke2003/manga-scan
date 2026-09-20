import cv2
import numpy as np

from .illumination import correct_illumination


def spine_position(image, ratio=0.5, mode="center"):
    h, w = image.shape[:2]
    center = int(round(w * ratio))
    if mode == "auto":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        profile = np.median(gray[h // 10 : max(h // 10 + 1, 9 * h // 10)], axis=0)
        radius = max(1, round(w * 0.04))
        lo, hi = max(1, center - radius), min(w - 1, center + radius)
        region = cv2.GaussianBlur(profile.astype(np.float32)[None, :], (9, 1), 0)[0]
        candidate = lo + int(np.argmin(region[lo:hi]))
        # A dark central panel can still fool this heuristic; center is the default.
        if float(np.median(region[lo:hi]) - region[candidate]) > 12:
            center = candidate
    return max(1, min(w - 1, center))


def split_spread(image, ratio=0.5, mode="center", gutter_fraction=0.0):
    spine = spine_position(image, ratio, mode)
    gutter = round(image.shape[1] * gutter_fraction / 2)
    left = image[:, : max(1, spine - gutter)].copy()
    right = image[:, min(image.shape[1] - 1, spine + gutter) :].copy()
    return {"left": left, "right": right}, spine


def enhance_page(
    image,
    grayscale=False,
    contrast=1.0,
    rotation=0,
    dewarp_strength=0.0,
    illumination_correction=False,
    illumination_strength=0.7,
):
    if dewarp_strength:
        # Symmetric cylindrical projection, user-controlled; never synthesizes pixels.
        h, w = image.shape[:2]
        theta = float(dewarp_strength)
        x = np.linspace(-1, 1, w, dtype=np.float32)
        map_x = np.tile(((np.sin(x * theta) / np.sin(theta) + 1) * (w - 1) / 2), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        image = cv2.remap(image, map_x.astype(np.float32), map_y, cv2.INTER_CUBIC)
    if illumination_correction and illumination_strength:
        image = correct_illumination(image, illumination_strength)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if contrast != 1:
        image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(
            np.uint8
        )
    if rotation:
        image = np.rot90(image, -(rotation // 90)).copy()
    return image
