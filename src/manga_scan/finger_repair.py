import cv2
import numpy as np

from .finger_alignment import (
    _align_local_component,
    _binary_mask,
    _component_records,
    _gray,
    align_donor_page,
)


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


def repair_occluded_regions(
    target,
    target_mask,
    donors,
    min_coverage=0.9,
    fallback="preserve",
):
    """Fill masked occlusions only from clean pixels in alternate frames.

    The mask may represent fingers, specular glare, or a union of both. Donor
    masks use the same contract, so a pixel is copied only when it is clean in
    the aligned donor as well.
    """
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
                    "context_residual_raw": (
                        None
                        if local.get("raw_residual") is None
                        else round(local["raw_residual"], 4)
                    ),
                    "photometric_applied": bool(
                        local.get("photometric", {}).get("applied")
                    ),
                    "photometric_gain": round(
                        float(local.get("photometric", {}).get("gain", 1.0)),
                        4,
                    ),
                    "photometric_bias": round(
                        float(local.get("photometric", {}).get("bias", 0.0)),
                        3,
                    ),
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


def repair_finger_regions(
    target,
    target_mask,
    donors,
    min_coverage=0.9,
    fallback="preserve",
):
    """Backward-compatible wrapper for the generalized occlusion repair engine."""

    return repair_occluded_regions(
        target,
        target_mask,
        donors,
        min_coverage=min_coverage,
        fallback=fallback,
    )

