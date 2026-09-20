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
    residual = _context_residual(target_gray, donor_gray, context)
    if residual is None or residual > _MAX_LOCAL_CONTEXT_RESIDUAL:
        return None

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
        "image": aligned_image,
        "mask": aligned_mask,
        "clean": clean,
        "score": bounded_score,
        "dx": float(dx),
        "dy": float(dy),
        "residual": residual,
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
        local["residual"] + 0.005 < best["residual"]
        or local["clean_coverage"] > best["clean_coverage"] + 0.03
    ):
        return local
    return best


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


def _local_alignment_metadata(components):
    local_components = []
    max_shift = 0.0
    for component in components:
        local_donors = [
            donor for donor in component.get("donors", [])
            if donor.get("method") == "local"
        ]
        if not local_donors:
            continue
        representative = max(
            local_donors,
            key=lambda donor: float(donor.get("dx", 0.0)) ** 2
            + float(donor.get("dy", 0.0)) ** 2,
        )
        dx = float(representative.get("dx", 0.0))
        dy = float(representative.get("dy", 0.0))
        shift = float(np.hypot(dx, dy))
        max_shift = max(max_shift, shift)
        local_components.append(
            {
                "component_id": component.get("component_id"),
                "donor_candidate_id": representative.get("candidate_id"),
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "score": representative.get("local_score"),
                "coverage": component.get("coverage", 0.0),
                "applied": True,
            }
        )
    if not local_components:
        return None
    return {
        "component_count": len(local_components),
        "max_shift_px": round(max_shift, 3),
        "components": local_components,
    }


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
            "components": [],
            "fallback": {
                "mode": fallback,
                "applied": False,
                "filled_pixels": 0,
                "filled_fraction": 0.0,
                "components_filled": 0,
                "components_total": 0,
            },
        }, np.zeros(target.shape[:2], np.uint8)

    # Keep the existing global page alignment as the first safety gate. Local
    # refinement is attempted only for donors that already match the page.
    aligned_donors = []
    for donor in donors:
        aligned = align_donor_page(
            target,
            donor["image"],
            donor["mask"],
            target_mask,
        )
        if aligned is None:
            continue
        aligned_image, aligned_mask, global_score = aligned
        aligned_donors.append(
            {
                "candidate_id": donor["candidate_id"],
                "image": aligned_image,
                "mask": aligned_mask,
                "global_score": float(global_score),
            }
        )

    result = target.copy()
    remaining = target_mask.astype(bool)
    donors_used = []
    alignment_scores = []
    components_info = []

    for component in _component_records(target_mask):
        original_pixels = int(np.count_nonzero(component["mask"]))
        component_used = set()
        component_donors = []

        while True:
            unresolved_component = remaining & component["mask"]
            unresolved_pixels = int(np.count_nonzero(unresolved_component))
            if unresolved_pixels < 8:
                break

            candidates = []
            for donor in aligned_donors:
                candidate_id = donor["candidate_id"]
                if candidate_id in component_used:
                    continue
                local = _align_local_component(
                    target,
                    donor["image"],
                    donor["mask"],
                    target_mask,
                    component,
                    donor["global_score"],
                )
                if local is None:
                    component_used.add(candidate_id)
                    continue

                # local["clean"] already requires the donor pixel to be clean
                # in both the globally aligned and locally refined donor masks.
                # The raw donor mask is still in pre-global coordinates, so
                # applying it here would reject valid pixels after translation.
                usable = unresolved_component & local["clean"]
                usable_pixels = int(np.count_nonzero(usable))
                if usable_pixels < 8:
                    component_used.add(candidate_id)
                    continue

                current_coverage = usable_pixels / unresolved_pixels
                quality = (
                    0.50 * current_coverage
                    + 0.30 * local["score"]
                    + 0.20 * (1.0 - local["residual"])
                )
                candidates.append((quality, usable_pixels, donor, local, usable))

            if not candidates:
                break

            _, usable_pixels, donor, local, usable = max(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            candidate_id = donor["candidate_id"]
            result = _blend_inside_mask(result, local["image"], usable)
            remaining[usable] = False
            component_used.add(candidate_id)

            if candidate_id not in donors_used:
                donors_used.append(candidate_id)
                alignment_scores.append(round(donor["global_score"], 4))

            component_donors.append(
                {
                    "candidate_id": candidate_id,
                    "method": local["method"],
                    "local_score": round(local["score"], 4),
                    "dx": round(local["dx"], 3),
                    "dy": round(local["dy"], 3),
                    "context_residual": round(local["residual"], 4),
                    "coverage": round(usable_pixels / original_pixels, 4),
                }
            )

        repaired_pixels = original_pixels - int(
            np.count_nonzero(remaining & component["mask"])
        )
        components_info.append(
            {
                "component_id": component["id"],
                "area": original_pixels,
                "coverage": round(repaired_pixels / original_pixels, 4),
                "donors": component_donors,
            }
        )

    donor_coverage = 1.0 - float(np.count_nonzero(remaining) / total)
    unresolved = remaining.astype(np.uint8) * 255
    result, unresolved, fallback_info = conceal_unresolved_fingers(
        result,
        unresolved,
        mode=fallback,
    )
    coverage = 1.0 - float(np.count_nonzero(unresolved) / total)
    status = "complete" if coverage >= min_coverage else "incomplete"
    metadata = {
        "status": status,
        "coverage": round(coverage, 4),
        "donor_coverage": round(donor_coverage, 4),
        "donors": donors_used,
        "alignment_scores": alignment_scores,
        "components": components_info,
        "fallback": fallback_info,
    }
    local_alignment = _local_alignment_metadata(components_info)
    if local_alignment is not None:
        metadata["local_alignment"] = local_alignment
    return result, metadata, unresolved

