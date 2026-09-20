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


def rotate_image(image, rotation=0):
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if rotation == 0:
        return image
    return np.rot90(image, -(rotation // 90)).copy()


def split_spread(image, ratio=0.5, mode="center", gutter_fraction=0.0):
    spine = spine_position(image, ratio, mode)
    gutter = round(image.shape[1] * gutter_fraction / 2)
    left = image[:, : max(1, spine - gutter)].copy()
    right = image[:, min(image.shape[1] - 1, spine + gutter) :].copy()
    return {"left": left, "right": right}, spine


def _edge_peaks(profile, min_distance):
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 5 or float(profile.max()) <= 0:
        return np.array([], dtype=np.int32)
    smooth = cv2.GaussianBlur(profile[None, :], (0, 0), 1.4)[0]
    threshold = max(float(np.percentile(smooth, 72)), float(smooth.mean() + 0.35 * smooth.std()))
    candidates = np.flatnonzero(
        (smooth[1:-1] >= smooth[:-2])
        & (smooth[1:-1] > smooth[2:])
        & (smooth[1:-1] >= threshold)
    ) + 1
    if not len(candidates):
        return np.array([], dtype=np.int32)
    chosen = []
    for index in candidates[np.argsort(smooth[candidates])[::-1]]:
        if all(abs(int(index) - kept) >= min_distance for kept in chosen):
            chosen.append(int(index))
    return np.array(sorted(chosen), dtype=np.int32)


def _spacing_ratio(peaks, width, side):
    if len(peaks) < 6:
        return None
    spacing = np.diff(peaks).astype(np.float32)
    midpoints = (peaks[:-1] + peaks[1:]) / (2 * max(1, width - 1))
    if side == "right":
        inner = spacing[midpoints < 0.34]
        reference = spacing[(midpoints > 0.42) & (midpoints < 0.82)]
    else:
        inner = spacing[midpoints > 0.66]
        reference = spacing[(midpoints > 0.18) & (midpoints < 0.58)]
    if len(inner) < 2 or len(reference) < 2:
        return None
    reference_median = float(np.median(reference))
    if reference_median < 2:
        return None
    return float(np.median(inner) / reference_median)


def estimate_curvature(image, side, max_strength=0.25):
    """Estimate horizontal compression near the spine from repeated vertical edges."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    h, w = image.shape[:2]
    if h < 40 or w < 80:
        return {
            "strength": 0.0,
            "confidence": 0.0,
            "compression_ratio": None,
            "bands": 0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, 640 / w)
    if scale < 1:
        gray = cv2.resize(
            gray,
            (max(80, round(w * scale)), max(40, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gy, gx = gray.shape
    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    cap = float(np.percentile(gradient, 98))
    if cap <= 1:
        return {
            "strength": 0.0,
            "confidence": 0.0,
            "compression_ratio": None,
            "bands": 0,
        }
    gradient = np.minimum(gradient, cap)
    min_distance = max(4, round(gx * 0.008))
    ratios = []
    bands = ((0.08, 0.24), (0.24, 0.40), (0.40, 0.56), (0.56, 0.72), (0.72, 0.88))
    for start, end in bands:
        y0 = round(gy * start)
        y1 = max(round(gy * end), y0 + 1)
        profile = gradient[y0:y1].mean(axis=0)
        peaks = _edge_peaks(profile, min_distance)
        ratio = _spacing_ratio(peaks, gx, side)
        if ratio is not None and 0.35 <= ratio <= 1.65:
            ratios.append(ratio)

    if len(ratios) < 2:
        return {
            "strength": 0.0,
            "confidence": round(min(0.25, len(ratios) * 0.12), 4),
            "compression_ratio": round(float(np.median(ratios)), 4) if ratios else None,
            "bands": len(ratios),
        }

    ratio = float(np.median(ratios))
    mad = float(np.median(np.abs(np.asarray(ratios) - ratio)))
    coverage = min(1.0, len(ratios) / 4)
    consistency = max(0.0, min(1.0, 1 - mad / 0.18))
    confidence = coverage * consistency
    strength = min(float(max_strength), max(0.0, (0.96 - ratio) * 0.9))
    if strength < 0.015:
        strength = 0.0
    return {
        "strength": round(strength, 4),
        "confidence": round(confidence, 4),
        "compression_ratio": round(ratio, 4),
        "bands": len(ratios),
    }


def dewarp_page(image, side, strength):
    """Stretch the spine side conservatively while keeping image bounds unchanged."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    strength = float(strength)
    if strength <= 0:
        return image.copy()
    h, w = image.shape[:2]
    if w < 2:
        return image.copy()
    gamma = 1.0 + 3.0 * min(strength, 0.35)
    u = np.linspace(0, 1, w, dtype=np.float32)
    if side == "right":
        source_u = np.power(u, gamma)
    else:
        source_u = 1.0 - np.power(1.0 - u, gamma)
    map_x = np.tile(source_u * (w - 1), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def auto_dewarp_page(image, side, max_strength=0.25, min_confidence=0.6):
    estimate = estimate_curvature(image, side, max_strength)
    confidence = estimate["confidence"]
    strength = estimate["strength"]
    if confidence < min_confidence:
        return image.copy(), {**estimate, "applied": False, "status": "low_confidence"}
    if strength <= 0:
        return image.copy(), {**estimate, "applied": False, "status": "not_needed"}
    return dewarp_page(image, side, strength), {
        **estimate,
        "applied": True,
        "status": "applied",
    }


def dewarp_debug_grid(shape, side, strength):
    h, w = shape[:2]
    grid = np.full((h, w, 3), 245, dtype=np.uint8)
    for x in np.linspace(0, max(0, w - 1), 11, dtype=int):
        cv2.line(grid, (int(x), 0), (int(x), max(0, h - 1)), (90, 90, 90), 1)
    for y in np.linspace(0, max(0, h - 1), 9, dtype=int):
        cv2.line(grid, (0, int(y)), (max(0, w - 1), int(y)), (185, 185, 185), 1)
    return dewarp_page(grid, side, strength)


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
    if white_normalization:
        image = normalize_white_background(image, white_target, white_strength)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if contrast != 1:
        image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(
            np.uint8
        )
    image = rotate_image(image, rotation)
    return image
