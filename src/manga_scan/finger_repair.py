import cv2
import numpy as np

_MAX_ALIGNMENT_SIDE = 640
_MIN_ALIGNMENT_SCORE = 0.72
_MAX_TRANSLATION_FRACTION = 0.08
_MAX_ROTATION_DEGREES = 5.0


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


def conceal_unresolved_fingers(image, unresolved_mask, mode="paper"):
    """Conceal unresolved finger pixels without inventing manga content.

    paper: fill only connected regions whose surrounding ring looks like plain paper.
    white: explicitly fill every unresolved pixel with white.
    preserve: leave unresolved pixels untouched.
    """
    if mode not in ("preserve", "paper", "white"):
        raise ValueError("finger repair fallback must be preserve, paper, or white")

    unresolved = _binary_mask(unresolved_mask, image.shape).astype(bool)
    total = int(np.count_nonzero(unresolved))
    info = {
        "mode": mode,
        "applied": False,
        "filled_pixels": 0,
        "filled_fraction": 0.0,
        "components_filled": 0,
        "components_total": 0,
    }
    if total == 0 or mode == "preserve":
        return image.copy(), (unresolved.astype(np.uint8) * 255), info

    if mode == "white":
        fill = np.full_like(image, 255)
        concealed = _blend_inside_mask(image, fill, unresolved)
        info.update(
            applied=True,
            filled_pixels=total,
            filled_fraction=1.0,
            components_filled=1,
            components_total=1,
        )
        return concealed, np.zeros(image.shape[:2], np.uint8), info

    height, width = image.shape[:2]
    labels_input = unresolved.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(labels_input, connectivity=8)
    info["components_total"] = max(0, count - 1)

    gray = _gray(image)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else None
    radius = max(5, round(min(height, width) * 0.015))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )

    result = image.copy()
    filled = np.zeros((height, width), bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4 or area > height * width * 0.18:
            continue

        component = labels == label
        component_u8 = component.astype(np.uint8) * 255
        outer = cv2.dilate(component_u8, kernel, iterations=1) > 0
        # Ignore the immediate mask boundary. Canny naturally sees the
        # finger/paper transition there, but that edge says nothing about
        # whether the surrounding page itself contains artwork or text.
        inner_radius = max(2, radius // 3)
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inner_radius * 2 + 1, inner_radius * 2 + 1),
        )
        near_mask = cv2.dilate(component_u8, inner_kernel, iterations=1) > 0
        ring = outer & ~near_mask & ~unresolved
        ring_pixels = int(np.count_nonzero(ring))
        if ring_pixels < max(24, round(area * 0.12)):
            continue

        ring_gray = gray[ring]
        brightness = float(np.median(ring_gray))
        spread = float(np.std(ring_gray))
        edge_density = float(np.mean(edges[ring] > 0))
        bright_fraction = float(np.mean(ring_gray >= 170))
        if hsv is not None:
            saturation = float(np.median(hsv[:, :, 1][ring]))
            low_saturation_fraction = float(np.mean(hsv[:, :, 1][ring] <= 90))
        else:
            saturation = 0.0
            low_saturation_fraction = 1.0

        paper_like = (
            brightness >= 180
            and spread <= 30
            and edge_density <= 0.08
            and bright_fraction >= 0.75
            and saturation <= 80
            and low_saturation_fraction >= 0.70
        )
        if not paper_like:
            continue

        if image.ndim == 3:
            fill_color = np.median(image[ring], axis=0).astype(np.uint8)
            donor = np.empty_like(image)
            donor[...] = fill_color
        else:
            fill_value = np.uint8(round(float(np.median(ring_gray))))
            donor = np.full_like(image, fill_value)

        result = _blend_inside_mask(result, donor, component)
        filled |= component
        info["components_filled"] += 1

    filled_pixels = int(np.count_nonzero(filled))
    remaining = unresolved & ~filled
    info.update(
        applied=filled_pixels > 0,
        filled_pixels=filled_pixels,
        filled_fraction=round(filled_pixels / total, 4),
    )
    return result, (remaining.astype(np.uint8) * 255), info


def repair_finger_regions(
    target,
    target_mask,
    donors,
    min_coverage=0.9,
    fallback="preserve",
):
    """Fill detected finger pixels only from clean pixels in alternate frames."""
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if fallback not in ("preserve", "paper", "white"):
        raise ValueError("finger repair fallback must be preserve, paper, or white")

    target_mask = _binary_mask(target_mask, target.shape)
    total = int(np.count_nonzero(target_mask))
    if total == 0:
        return target.copy(), {
            "status": "clean",
            "coverage": 1.0,
            "donor_coverage": 1.0,
            "donors": [],
            "alignment_scores": [],
            "fallback": {
                "mode": fallback,
                "applied": False,
                "filled_pixels": 0,
                "filled_fraction": 0.0,
                "components_filled": 0,
                "components_total": 0,
            },
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

    donor_coverage = 1.0 - float(np.count_nonzero(remaining) / total)
    unresolved = remaining.astype(np.uint8) * 255
    result, unresolved, fallback_info = conceal_unresolved_fingers(
        result,
        unresolved,
        mode=fallback,
    )
    coverage = 1.0 - float(np.count_nonzero(unresolved) / total)
    status = "complete" if coverage >= min_coverage else "incomplete"
    return result, {
        "status": status,
        "coverage": round(coverage, 4),
        "donor_coverage": round(donor_coverage, 4),
        "donors": donors_used,
        "alignment_scores": alignment_scores,
        "fallback": fallback_info,
    }, unresolved
