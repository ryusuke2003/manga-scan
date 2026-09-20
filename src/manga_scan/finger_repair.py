import cv2
import numpy as np

_MAX_ALIGNMENT_SIDE = 640
_MIN_ALIGNMENT_SCORE = 0.55
_MAX_TRANSLATION_FRACTION = 0.18
_MIN_SCALE_DETERMINANT = 0.70
_MAX_SCALE_DETERMINANT = 1.35


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
    return (mask > 127).astype(np.uint8)


def _alignment_scale(shape):
    height, width = shape[:2]
    return min(1.0, _MAX_ALIGNMENT_SIDE / max(height, width))


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
    clean = ((target_mask == 0) & (donor_mask == 0)).astype(np.uint8)
    clean_small = cv2.resize(
        (clean * 255).astype(np.uint8),
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )
    clean_small = cv2.erode(clean_small, np.ones((3, 3), np.uint8), iterations=1)
    if np.count_nonzero(clean_small) < clean_small.size * 0.2:
        return None

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        60,
        1e-5,
    )
    try:
        score, warp = cv2.findTransformECC(
            target_small,
            donor_small,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=clean_small,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None

    if not np.isfinite(score) or score < _MIN_ALIGNMENT_SCORE or not np.isfinite(warp).all():
        return None

    small_width, small_height = small_size
    warp = warp.astype(np.float32, copy=True)
    warp[0, 2] *= width / small_width
    warp[1, 2] *= height / small_height
    determinant = float(np.linalg.det(warp[:, :2]))
    if not _MIN_SCALE_DETERMINANT <= determinant <= _MAX_SCALE_DETERMINANT:
        return None
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
    return aligned, aligned_mask, float(score)


def _blend_inside_mask(base, donor, mask):
    if not np.any(mask):
        return base
    blend_px = max(2.0, min(base.shape[:2]) * 0.004)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance / blend_px, 0, 1).astype(np.float32)
    if base.ndim == 3:
        alpha = alpha[:, :, None]
    mixed = base.astype(np.float32) * (1 - alpha) + donor.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def repair_finger_regions(target, target_mask, donors, min_coverage=0.9):
    """Fill detected finger pixels only from clean pixels in alternate frames."""
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")

    target_mask = _binary_mask(target_mask, target.shape)
    total = int(np.count_nonzero(target_mask))
    if total == 0:
        return target.copy(), {
            "status": "clean",
            "coverage": 1.0,
            "donors": [],
            "alignment_scores": [],
        }, np.zeros(target.shape[:2], np.uint8)

    result = target.copy()
    remaining = target_mask.astype(bool)
    donors_used = []
    alignment_scores = []

    for donor in donors:
        aligned = align_donor_page(
            target,
            donor["image"],
            donor["mask"],
            target_mask,
        )
        if aligned is None:
            continue
        aligned_image, aligned_mask, score = aligned
        donor_clean = aligned_mask <= 127
        donor_clean = cv2.erode(
            donor_clean.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        ).astype(bool)
        usable = remaining & donor_clean
        if np.count_nonzero(usable) < 8:
            continue

        result = _blend_inside_mask(result, aligned_image, usable)
        remaining[usable] = False
        donors_used.append(donor["candidate_id"])
        alignment_scores.append(round(score, 4))
        if not np.any(remaining):
            break

    unresolved = remaining.astype(np.uint8) * 255
    coverage = 1.0 - float(np.count_nonzero(remaining) / total)
    status = "complete" if coverage >= min_coverage else "incomplete"
    return result, {
        "status": status,
        "coverage": round(coverage, 4),
        "donors": donors_used,
        "alignment_scores": alignment_scores,
    }, unresolved
