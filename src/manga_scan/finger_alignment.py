import cv2
import numpy as np

_MAX_ALIGNMENT_SIDE = 640
_MIN_ALIGNMENT_SCORE = 0.72
_MAX_TRANSLATION_FRACTION = 0.08
_MAX_ROTATION_DEGREES = 5.0
_MAX_LOCAL_SHIFT_FRACTION = 0.03
_LOCAL_CONTEXT_FRACTION = 0.06
_LOCAL_MASK_MARGIN_FRACTION = 0.006
_MIN_LOCAL_ALIGNMENT_SCORE = 0.55
_MAX_LOCAL_CONTEXT_RESIDUAL = 0.18
_MIN_LOCAL_CONTEXT_PIXELS = 80
_PHOTOMETRIC_GAIN_MIN = 0.90
_PHOTOMETRIC_GAIN_MAX = 1.10
_PHOTOMETRIC_BIAS_LIMIT = 12.0
_PHOTOMETRIC_MIN_IMPROVEMENT = 0.002
_PHOTOMETRIC_MIN_STD = 5.0


def _gray(image):
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("finger repair expects grayscale or BGR images")


def _binary_mask(mask, shape):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("finger repair masks must be grayscale arrays")
    height, width = shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Keep normalization idempotent. Internal alignment code intentionally
    # passes masks through this helper more than once, so an already-normalized
    # 0/1 mask must not be erased by the 8-bit 0/255 threshold.
    if mask.size and float(np.max(mask)) <= 1.0:
        return (mask > 0).astype(np.uint8)
    return (mask > 127).astype(np.uint8)


def _alignment_scale(shape):
    height, width = shape[:2]
    return min(1.0, _MAX_ALIGNMENT_SIDE / max(height, width))


def _global_alignment_residual(target_gray, donor_gray, target_mask, donor_mask):
    height, width = target_gray.shape[:2]
    if target_mask.shape != (height, width):
        target_mask = cv2.resize(
            target_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    if donor_mask.shape != (height, width):
        donor_mask = cv2.resize(
            donor_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    # Internal alignment masks may already be normalized to 0/1 while warped
    # masks use 0/255. Non-zero is the mask contract here, not a gray threshold.
    blocked = ((target_mask > 0) | (donor_mask > 0)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.dilate(blocked, kernel, iterations=1) == 0
    if np.count_nonzero(clean) < clean.size * 0.2:
        return None
    delta = np.abs(
        target_gray[clean].astype(np.float32)
        - donor_gray[clean].astype(np.float32)
    )
    return float(np.mean(delta) / 255.0)


def align_donor_page(target, donor, donor_mask, target_mask):
    """Align a donor page to the selected page using only existing image content."""
    if not isinstance(target, np.ndarray) or not isinstance(donor, np.ndarray):
        raise ValueError("target and donor must be numpy images")
    if target.ndim != donor.ndim or target.ndim not in (2, 3):
        raise ValueError("target and donor must have matching image dimensions")

    height, width = target.shape[:2]
    if min(height, width) < 8:
        return None

    donor = cv2.resize(donor, (width, height), interpolation=cv2.INTER_CUBIC)
    donor_mask = _binary_mask(donor_mask, target.shape)
    target_mask = _binary_mask(target_mask, target.shape)

    target_gray = _gray(target)
    donor_gray = _gray(donor)
    scale = _alignment_scale(target.shape)
    small_size = (
        max(8, round(width * scale)),
        max(8, round(height * scale)),
    )
    target_small = cv2.resize(target_gray, small_size, interpolation=cv2.INTER_AREA)
    donor_small = cv2.resize(donor_gray, small_size, interpolation=cv2.INTER_AREA)
    target_mask_small = cv2.resize(
        target_mask,
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )
    donor_mask_small = cv2.resize(
        donor_mask,
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )

    # The two hand masks live in different image coordinate systems before
    # alignment, so do not AND them into one ECC input mask. Neutralize each
    # image's own masked pixels instead, then align only the page appearance.
    kernel = np.ones((5, 5), np.uint8)
    target_mask_small = cv2.dilate(target_mask_small, kernel, iterations=1)
    donor_mask_small = cv2.dilate(donor_mask_small, kernel, iterations=1)
    target_clean = target_mask_small == 0
    donor_clean = donor_mask_small == 0
    if (
        np.count_nonzero(target_clean) < target_clean.size * 0.2
        or np.count_nonzero(donor_clean) < donor_clean.size * 0.2
    ):
        return None

    fill = int(
        round(
            (
                float(np.median(target_small[target_clean]))
                + float(np.median(donor_small[donor_clean]))
            )
            / 2
        )
    )
    target_for_ecc = target_small.copy()
    donor_for_ecc = donor_small.copy()
    target_for_ecc[~target_clean] = fill
    donor_for_ecc[~donor_clean] = fill

    # Page geometry has already been normalized by ROI / per-page perspective
    # correction. Residual alignment should therefore be only a small rotation
    # and translation. Refuse scale/shear instead of risking wrong manga pixels.
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        60,
        1e-5,
    )
    try:
        score, warp = cv2.findTransformECC(
            target_for_ecc,
            donor_for_ecc,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None

    if not np.isfinite(score) or score < _MIN_ALIGNMENT_SCORE or not np.isfinite(warp).all():
        return None

    small_width, small_height = small_size
    warp = warp.astype(np.float32, copy=True)
    angle = abs(float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))))
    if angle > _MAX_ROTATION_DEGREES:
        return None
    warp[0, 2] *= width / small_width
    warp[1, 2] *= height / small_height
    if (
        abs(float(warp[0, 2])) > width * _MAX_TRANSLATION_FRACTION
        or abs(float(warp[1, 2])) > height * _MAX_TRANSLATION_FRACTION
    ):
        return None

    aligned = cv2.warpAffine(
        donor,
        warp,
        (width, height),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    aligned_mask = cv2.warpAffine(
        (donor_mask * 255).astype(np.uint8),
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    # ECC can occasionally find a small but unnecessary warp when masked
    # regions differ. Prefer the unwarped donor unless the proposed global
    # transform actually improves clean-pixel agreement. This lets a genuine
    # page shift move the donor mask with the page without using a raw mask in
    # target coordinates as a safety backstop.
    identity_residual = _global_alignment_residual(
        target_gray,
        donor_gray,
        target_mask,
        donor_mask,
    )
    aligned_residual = _global_alignment_residual(
        target_gray,
        _gray(aligned),
        target_mask,
        aligned_mask,
    )
    if (
        identity_residual is not None
        and (
            aligned_residual is None
            or identity_residual <= aligned_residual + 0.002
        )
    ):
        return donor, (donor_mask * 255).astype(np.uint8), float(score)

    return aligned, aligned_mask, float(score)


def _dilate_mask(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _safe_clean_mask(mask, shape):
    binary = _binary_mask(mask, shape).astype(bool)
    margin = max(2, round(min(shape[:2]) * _LOCAL_MASK_MARGIN_FRACTION))
    return ~_dilate_mask(binary, margin)


def _component_records(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    records = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        records.append(
            {
                "id": label,
                "area": area,
                "mask": labels == label,
                "bbox": (
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ),
            }
        )
    return records


def _component_context(component, target_mask, donor_mask):
    height, width = target_mask.shape
    x, y, box_width, box_height = component["bbox"]
    radius = max(10, round(min(height, width) * _LOCAL_CONTEXT_FRACTION))
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + box_width + radius)
    y1 = min(height, y + box_height + radius)

    region = np.zeros((height, width), bool)
    region[y0:y1, x0:x1] = True

    margin = max(2, round(min(height, width) * _LOCAL_MASK_MARGIN_FRACTION))
    inner_margin = max(margin + 1, round(min(height, width) * 0.012))
    blocked_target = _dilate_mask(target_mask, margin)
    blocked_component = _dilate_mask(component["mask"], inner_margin)
    blocked_donor = _dilate_mask(donor_mask, margin)
    context = region & ~blocked_target & ~blocked_component & ~blocked_donor
    required = max(_MIN_LOCAL_CONTEXT_PIXELS, round(component["area"] * 0.5))
    if np.count_nonzero(context) < required:
        return None
    return context, (x0, y0, x1, y1)


def _warp_local_translation(image, mask, dx, dy):
    height, width = image.shape[:2]
    warp = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned_image = cv2.warpAffine(
        image,
        warp,
        (width, height),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    aligned_mask = cv2.warpAffine(
        (_binary_mask(mask, image.shape) * 255).astype(np.uint8),
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return aligned_image, aligned_mask


def _context_residual(target_gray, donor_gray, context):
    if np.count_nonzero(context) < _MIN_LOCAL_CONTEXT_PIXELS:
        return None
    delta = np.abs(
        target_gray[context].astype(np.float32)
        - donor_gray[context].astype(np.float32)
    )
    return float(np.mean(delta) / 255.0)


def _estimate_photometric_alignment(target, donor, context):
    """Fit a bounded scalar exposure correction from clean aligned context.

    The same gain/bias is applied to every channel so chroma relationships are
    preserved. The correction is only returned when it measurably improves the
    context residual; otherwise the donor is left untouched.
    """
    if np.count_nonzero(context) < _MIN_LOCAL_CONTEXT_PIXELS:
        return donor, {
            "applied": False,
            "gain": 1.0,
            "bias": 0.0,
            "residual_before": None,
            "residual_after": None,
        }

    target_gray = _gray(target).astype(np.float32)
    donor_gray = _gray(donor).astype(np.float32)
    x = donor_gray[context]
    y = target_gray[context]

    finite = np.isfinite(x) & np.isfinite(y)
    # Avoid clipped black/white pixels dominating the exposure fit. If the
    # page is mostly flat paper, fall back to all finite clean context pixels.
    unclipped = finite & (x > 5) & (x < 250) & (y > 5) & (y < 250)
    if np.count_nonzero(unclipped) >= _MIN_LOCAL_CONTEXT_PIXELS:
        x_fit = x[unclipped]
        y_fit = y[unclipped]
    else:
        x_fit = x[finite]
        y_fit = y[finite]

    if x_fit.size < _MIN_LOCAL_CONTEXT_PIXELS:
        return donor, {
            "applied": False,
            "gain": 1.0,
            "bias": 0.0,
            "residual_before": None,
            "residual_after": None,
        }

    x_centered = x_fit - float(np.mean(x_fit))
    variance = float(np.mean(x_centered * x_centered))
    if variance >= _PHOTOMETRIC_MIN_STD ** 2:
        covariance = float(
            np.mean(x_centered * (y_fit - float(np.mean(y_fit))))
        )
        gain = covariance / variance
    else:
        # On nearly uniform paper, gain is ill-conditioned; a simple offset
        # is safer and still removes the common AE brightness seam.
        gain = 1.0

    gain = float(np.clip(gain, _PHOTOMETRIC_GAIN_MIN, _PHOTOMETRIC_GAIN_MAX))
    bias = float(np.median(y_fit - gain * x_fit))
    bias = float(np.clip(bias, -_PHOTOMETRIC_BIAS_LIMIT, _PHOTOMETRIC_BIAS_LIMIT))

    residual_before = _context_residual(target_gray, donor_gray, context)
    corrected_gray = np.clip(donor_gray * gain + bias, 0, 255)
    residual_after = _context_residual(target_gray, corrected_gray, context)
    if (
        residual_before is None
        or residual_after is None
        or residual_after + _PHOTOMETRIC_MIN_IMPROVEMENT >= residual_before
    ):
        return donor, {
            "applied": False,
            "gain": 1.0,
            "bias": 0.0,
            "residual_before": residual_before,
            "residual_after": residual_before,
        }

    corrected = np.clip(donor.astype(np.float32) * gain + bias, 0, 255).astype(
        donor.dtype
    )
    return corrected, {
        "applied": True,
        "gain": gain,
        "bias": bias,
        "residual_before": residual_before,
        "residual_after": residual_after,
    }


def _validate_local_candidate(
    target,
    donor,
    donor_mask,
    target_mask,
    component,
    dx,
    dy,
    score,
    method,
):
    height, width = target.shape[:2]
    max_shift = max(2.0, min(height, width) * _MAX_LOCAL_SHIFT_FRACTION)
    if abs(float(dx)) > max_shift or abs(float(dy)) > max_shift:
        return None

    aligned_image, aligned_mask = _warp_local_translation(
        donor,
        donor_mask,
        float(dx),
        float(dy),
    )
    # A local warp must never turn pixels that were hand-masked in the
    # globally aligned donor into eligible manga pixels merely by moving the
    # mask away. Require clean pixels in both coordinate systems.
    source_clean = _safe_clean_mask(donor_mask, target.shape)
    clean = _safe_clean_mask(aligned_mask, target.shape) & source_clean
    context_data = _component_context(
        component,
        target_mask.astype(bool),
        (aligned_mask > 127),
    )
    if context_data is None:
        return None
    context, _ = context_data
    target_gray = _gray(target)
    donor_gray = _gray(aligned_image)
    raw_residual = _context_residual(target_gray, donor_gray, context)
    # Geometry remains the primary safety gate. Photometric correction must
    # never make a badly aligned donor look geometrically acceptable.
    if raw_residual is None or raw_residual > _MAX_LOCAL_CONTEXT_RESIDUAL:
        return None

    corrected_image, photometric = _estimate_photometric_alignment(
        target,
        aligned_image,
        context,
    )
    residual = photometric["residual_after"]
    if residual is None:
        residual = raw_residual

    usable = component["mask"] & clean
    clean_coverage = float(np.count_nonzero(usable) / component["area"])
    if np.count_nonzero(usable) < 8:
        return None

    bounded_score = max(0.0, min(1.0, float(score)))
    quality = (
        0.50 * clean_coverage
        + 0.30 * bounded_score
        + 0.20 * (1.0 - residual)
    )
    return {
        "image": corrected_image,
        "mask": aligned_mask,
        "clean": clean,
        "score": bounded_score,
        "dx": float(dx),
        "dy": float(dy),
        "residual": residual,
        "selection_residual": raw_residual,
        "raw_residual": raw_residual,
        "photometric": photometric,
        "clean_coverage": clean_coverage,
        "quality": quality,
        "method": method,
    }


def _align_local_component(
    target,
    donor,
    donor_mask,
    target_mask,
    component,
    global_score,
):
    """Refine one globally aligned donor using bounded translation only."""
    donor_mask = _binary_mask(donor_mask, target.shape)
    target_mask = _binary_mask(target_mask, target.shape)

    context_data = _component_context(
        component,
        target_mask.astype(bool),
        donor_mask.astype(bool),
    )
    if context_data is None:
        return None
    context, (x0, y0, x1, y1) = context_data

    best = _validate_local_candidate(
        target,
        donor,
        donor_mask,
        target_mask,
        component,
        0.0,
        0.0,
        global_score,
        "global",
    )

    target_gray = _gray(target)
    donor_gray = _gray(donor)
    target_patch = target_gray[y0:y1, x0:x1]
    donor_patch = donor_gray[y0:y1, x0:x1]
    context_patch = context[y0:y1, x0:x1]
    if (
        target_patch.shape[0] < 8
        or target_patch.shape[1] < 8
        or np.count_nonzero(context_patch) < _MIN_LOCAL_CONTEXT_PIXELS
    ):
        return best

    context_values = np.concatenate(
        (
            target_patch[context_patch].astype(np.float32),
            donor_patch[context_patch].astype(np.float32),
        )
    )
    fill = np.uint8(np.clip(round(float(np.median(context_values))), 0, 255))
    target_for_ecc = target_patch.copy()
    donor_for_ecc = donor_patch.copy()
    target_for_ecc[~context_patch] = fill
    donor_for_ecc[~context_patch] = fill
    input_mask = (context_patch.astype(np.uint8) * 255)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    try:
        local_score, warp = cv2.findTransformECC(
            target_for_ecc,
            donor_for_ecc,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=input_mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return best

    if (
        not np.isfinite(local_score)
        or local_score < _MIN_LOCAL_ALIGNMENT_SCORE
        or not np.isfinite(warp).all()
    ):
        return best

    local = _validate_local_candidate(
        target,
        donor,
        donor_mask,
        target_mask,
        component,
        float(warp[0, 2]),
        float(warp[1, 2]),
        float(local_score),
        "local",
    )
    if local is None:
        return best
    if best is None:
        return local

    # Prefer local refinement only when it measurably improves the surrounding
    # context, or exposes more clean donor pixels inside the finger component.
    if (
        local["selection_residual"] + 0.005 < best["selection_residual"]
        or local["clean_coverage"] > best["clean_coverage"] + 0.03
    ):
        return local
    return best
