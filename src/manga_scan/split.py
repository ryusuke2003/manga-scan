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


def _curvature_band_centers():
    return np.linspace(0.08, 0.92, 9, dtype=np.float32)


def _strength_from_ratio(ratio, max_strength):
    strength = min(float(max_strength), max(0.0, (0.96 - float(ratio)) * 0.9))
    return 0.0 if strength < 0.015 else strength


def _build_strength_profile(samples, max_strength):
    if len(samples) < 2:
        return []
    centers = _curvature_band_centers()
    sample_y = np.asarray([sample["y"] for sample in samples], dtype=np.float32)
    sample_strength = np.asarray(
        [_strength_from_ratio(sample["ratio"], max_strength) for sample in samples],
        dtype=np.float32,
    )
    interpolated = np.interp(centers, sample_y, sample_strength)
    padded = np.pad(interpolated, (1, 1), mode="edge")
    smoothed = 0.2 * padded[:-2] + 0.6 * padded[1:-1] + 0.2 * padded[2:]
    smoothed = np.clip(smoothed, 0.0, float(max_strength))
    return [
        {"y": round(float(y), 4), "strength": round(float(strength), 4)}
        for y, strength in zip(centers, smoothed)
    ]


def estimate_curvature(image, side, max_strength=0.25):
    """Estimate a vertical profile of horizontal compression near the page spine."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    h, w = image.shape[:2]
    empty = {
        "strength": 0.0,
        "mean_strength": 0.0,
        "profile_variation": 0.0,
        "strength_profile": [],
        "confidence": 0.0,
        "compression_ratio": None,
        "bands": 0,
    }
    if h < 40 or w < 80:
        return empty

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
        return empty
    gradient = np.minimum(gradient, cap)
    min_distance = max(4, round(gx * 0.008))

    samples = []
    centers = _curvature_band_centers()
    half_band = 0.065
    for center in centers:
        start = max(0.0, float(center) - half_band)
        end = min(1.0, float(center) + half_band)
        y0 = round(gy * start)
        y1 = max(round(gy * end), y0 + 1)
        profile = gradient[y0:y1].mean(axis=0)
        peaks = _edge_peaks(profile, min_distance)
        ratio = _spacing_ratio(peaks, gx, side)
        if ratio is not None and 0.35 <= ratio <= 1.65:
            samples.append({"y": float(center), "ratio": float(ratio)})

    if len(samples) < 2:
        ratios = [sample["ratio"] for sample in samples]
        return {
            **empty,
            "confidence": round(min(0.2, len(samples) * 0.1), 4),
            "compression_ratio": round(float(np.median(ratios)), 4) if ratios else None,
            "bands": len(samples),
        }

    ratios = np.asarray([sample["ratio"] for sample in samples], dtype=np.float32)
    ratio = float(np.median(ratios))
    coverage = min(1.0, len(samples) / len(centers))

    # Real books can curve more at one height than another, so distance from the
    # global median is not itself suspicious. What should reduce confidence is a
    # jagged profile where adjacent scanline bands disagree sharply; that pattern
    # is more likely to come from panels/text than from a physical book surface.
    if len(ratios) >= 3:
        adjacent_change = float(np.median(np.abs(np.diff(ratios))))
        smoothness = max(0.0, min(1.0, 1 - adjacent_change / 0.18))
    else:
        smoothness = 0.5
    confidence = min(1.0, coverage * (0.4 + 0.6 * smoothness))

    strength_profile = _build_strength_profile(samples, max_strength)
    strengths = np.asarray(
        [entry["strength"] for entry in strength_profile],
        dtype=np.float32,
    )
    peak_strength = float(strengths.max()) if strengths.size else 0.0
    mean_strength = float(strengths.mean()) if strengths.size else 0.0
    variation = float(strengths.max() - strengths.min()) if strengths.size else 0.0
    return {
        "strength": round(peak_strength, 4),
        "mean_strength": round(mean_strength, 4),
        "profile_variation": round(variation, 4),
        "strength_profile": strength_profile,
        "confidence": round(confidence, 4),
        "compression_ratio": round(ratio, 4),
        "bands": len(samples),
    }


def _row_strengths(height, strength_profile, fallback_strength=0.0):
    if height <= 0:
        return np.zeros(0, dtype=np.float32)
    if not strength_profile:
        return np.full(height, float(fallback_strength), dtype=np.float32)

    points = sorted(
        (
            max(0.0, min(1.0, float(entry["y"]))),
            max(0.0, min(0.35, float(entry["strength"]))),
        )
        for entry in strength_profile
    )
    ys = np.asarray([point[0] * max(1, height - 1) for point in points], dtype=np.float32)
    strengths = np.asarray([point[1] for point in points], dtype=np.float32)
    if len(points) == 1:
        return np.full(height, strengths[0], dtype=np.float32)

    rows = np.arange(height, dtype=np.float32)
    interpolated = np.interp(rows, ys, strengths).astype(np.float32)

    window = max(3, min(81, int(round(height * 0.05))))
    if window % 2 == 0:
        window += 1
    if window > height and height > 1:
        window = height if height % 2 else height - 1
    if window >= 3:
        kernel = cv2.getGaussianKernel(window, max(1.0, window / 4))[:, 0].astype(np.float32)
        pad = window // 2
        interpolated = np.convolve(
            np.pad(interpolated, (pad, pad), mode="edge"),
            kernel,
            mode="valid",
        ).astype(np.float32)
    return np.clip(interpolated, 0.0, 0.35)


def _remap_with_row_strengths(image, side, row_strengths):
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    h, w = image.shape[:2]
    if w < 2 or h < 1:
        return image.copy()
    row_strengths = np.asarray(row_strengths, dtype=np.float32)
    if row_strengths.shape != (h,):
        raise ValueError("row_strengths must contain one value per image row")
    if row_strengths.size == 0 or float(row_strengths.max()) <= 0:
        return image.copy()

    gamma = 1.0 + 3.0 * np.clip(row_strengths, 0.0, 0.35)
    u = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    gamma = gamma[:, None]
    if side == "right":
        source_u = np.power(u, gamma)
    else:
        source_u = 1.0 - np.power(1.0 - u, gamma)
    map_x = (source_u * (w - 1)).astype(np.float32, copy=False)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def dewarp_page(image, side, strength):
    """Apply the legacy constant-strength spine-side stretch."""
    strength = float(strength)
    if strength <= 0:
        return image.copy()
    return _remap_with_row_strengths(
        image,
        side,
        np.full(image.shape[0], min(strength, 0.35), dtype=np.float32),
    )


def dewarp_page_profile(image, side, strength_profile, fallback_strength=0.0):
    """Apply a height-varying 2D remap estimated from the photographed page."""
    return _remap_with_row_strengths(
        image,
        side,
        _row_strengths(image.shape[0], strength_profile, fallback_strength),
    )


def auto_dewarp_page(image, side, max_strength=0.25, min_confidence=0.6):
    estimate = estimate_curvature(image, side, max_strength)
    confidence = estimate["confidence"]
    strength = estimate["strength"]
    if confidence < min_confidence:
        return image.copy(), {**estimate, "applied": False, "status": "low_confidence"}
    if strength <= 0:
        return image.copy(), {**estimate, "applied": False, "status": "not_needed"}
    return dewarp_page_profile(
        image,
        side,
        estimate.get("strength_profile", []),
        fallback_strength=estimate.get("mean_strength", strength),
    ), {
        **estimate,
        "applied": True,
        "status": "applied",
    }


def dewarp_debug_grid(shape, side, strength, strength_profile=None):
    h, w = shape[:2]
    grid = np.full((h, w, 3), 245, dtype=np.uint8)
    for x in np.linspace(0, max(0, w - 1), 11, dtype=int):
        cv2.line(grid, (int(x), 0), (int(x), max(0, h - 1)), (90, 90, 90), 1)
    for y in np.linspace(0, max(0, h - 1), 9, dtype=int):
        cv2.line(grid, (0, int(y)), (max(0, w - 1), int(y)), (185, 185, 185), 1)
    if strength_profile:
        return dewarp_page_profile(grid, side, strength_profile, fallback_strength=strength)
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
