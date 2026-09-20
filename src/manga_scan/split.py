import cv2
import numpy as np


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


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def normalize_white_background(image, target=245, strength=0.6):
    """Gently move bright, paper-like pixels toward white while protecting darker artwork."""
    if strength <= 0:
        return image.copy()

    if image.ndim == 2:
        lightness = image.astype(np.float32)
        candidate_floor = max(160.0, float(np.percentile(lightness, 70)))
        candidates = lightness >= candidate_floor
        if int(candidates.sum()) < max(32, round(lightness.size * 0.01)):
            return image.copy()
        white_level = float(np.percentile(lightness[candidates], 75))
        if white_level < 170:
            return image.copy()
        transition_start = max(150.0, white_level - 45.0)
        weight = _smoothstep(
            (lightness - transition_start) / max(1.0, white_level - transition_start)
        ) * float(strength)
        gain = min(1.18, max(1.0, float(target) / max(1.0, white_level)))
        brightened = np.clip(lightness * gain, 0, 255)
        return np.clip(lightness * (1.0 - weight) + brightened * weight, 0, 255).astype(
            np.uint8
        )

    # Keep the 3-channel Lab image as uint8. Converting it wholesale to float32
    # multiplies its memory footprint by four on full-resolution pages.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)

    # Build chroma with one reusable float32 scratch buffer instead of keeping
    # separate float a, b and chroma arrays alive at the same time.
    chroma = lab[:, :, 1].astype(np.float32)
    chroma -= 128.0
    np.square(chroma, out=chroma)
    scratch = lab[:, :, 2].astype(np.float32)
    scratch -= 128.0
    np.square(scratch, out=scratch)
    chroma += scratch
    del scratch
    np.sqrt(chroma, out=chroma)

    candidate_floor = max(160.0, float(np.percentile(lightness, 70)))
    candidates = (lightness >= candidate_floor) & (chroma <= 30.0)
    if int(candidates.sum()) < max(32, round(lightness.size * 0.01)):
        return image.copy()

    white_level = float(np.percentile(lightness[candidates], 75))
    if white_level < 170:
        return image.copy()

    transition_start = max(150.0, white_level - 45.0)
    weight = _smoothstep(
        (lightness - transition_start) / max(1.0, white_level - transition_start)
    ) * float(strength)

    gain = min(1.18, max(1.0, float(target) / max(1.0, white_level)))
    brightened = lightness.copy()
    brightened *= gain
    np.clip(brightened, 0, 255, out=brightened)
    brightened -= lightness
    brightened *= weight
    brightened += lightness
    lab[:, :, 0] = brightened

    # Reuse the chroma buffer as the final chroma correction weight.
    chroma *= -1.0
    chroma += 30.0
    chroma /= 20.0
    chroma_weight = _smoothstep(chroma)
    chroma_weight *= weight

    # Only neutralize low-chroma bright pixels. Process one channel at a time
    # so full-resolution float copies of both a and b are never resident together.
    for channel_index in (1, 2):
        channel = lab[:, :, channel_index].astype(np.float32)
        channel -= 128.0
        channel *= chroma_weight
        channel *= -1.0
        channel += lab[:, :, channel_index]
        np.clip(channel, 0, 255, out=channel)
        lab[:, :, channel_index] = channel

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_page(
    image,
    grayscale=False,
    contrast=1.0,
    rotation=0,
    dewarp_strength=0.0,
    white_normalization=False,
    white_target=245,
    white_strength=0.6,
):
    if dewarp_strength:
        # Symmetric cylindrical projection, user-controlled; never synthesizes pixels.
        h, w = image.shape[:2]
        theta = float(dewarp_strength)
        x = np.linspace(-1, 1, w, dtype=np.float32)
        map_x = np.tile(((np.sin(x * theta) / np.sin(theta) + 1) * (w - 1) / 2), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        image = cv2.remap(image, map_x.astype(np.float32), map_y, cv2.INTER_CUBIC)
    if white_normalization:
        image = normalize_white_background(image, white_target, white_strength)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if contrast != 1:
        image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(
            np.uint8
        )
    if rotation:
        image = np.rot90(image, -(rotation // 90)).copy()
    return image
