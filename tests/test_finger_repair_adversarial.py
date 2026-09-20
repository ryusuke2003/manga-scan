import cv2
import numpy as np

from manga_scan.finger_repair import align_donor_page, repair_finger_regions


def _manga_page(height=220, width=320):
    page = np.full((height, width, 3), 238, np.uint8)
    cv2.rectangle(page, (18, 18), (width - 18, height - 18), (25, 25, 25), 3)
    cv2.line(page, (width // 2, 20), (width // 2, height - 20), (20, 20, 20), 4)
    for y in range(45, 170, 22):
        cv2.line(page, (205, y), (265, y), (55, 55, 55), 3)
    cv2.putText(
        page,
        "MANGA",
        (35, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    return page


def _mask(shape, rectangles):
    mask = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _shift(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )


def test_adversarial_shifted_panel_line_never_changes_target_outside_mask():
    """Issue #53: a few-pixel donor shift must never leak outside the target hand mask."""
    clean = _manga_page()
    target = clean.copy()
    target_mask = _mask(target.shape, [(145, 55, 190, 165)])
    target[target_mask > 0] = (45, 105, 190)
    before = target.copy()
    donor = _shift(clean, 4, -3)

    repaired, _metadata, _unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 4, "image": donor, "mask": np.zeros(target.shape[:2], np.uint8)}],
    )

    np.testing.assert_array_equal(repaired[target_mask == 0], before[target_mask == 0])


def test_adversarial_vertical_text_shift_never_changes_target_outside_mask():
    """Issue #53: locally shifted vertical/text-like strokes stay confined to the repair mask."""
    clean = _manga_page()
    target = clean.copy()
    target_mask = _mask(target.shape, [(195, 35, 278, 185)])
    target[target_mask > 0] = (35, 95, 180)
    before = target.copy()
    donor = _shift(clean, -3, 2)

    repaired, _metadata, _unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 2, "image": donor, "mask": np.zeros(target.shape[:2], np.uint8)}],
    )

    np.testing.assert_array_equal(repaired[target_mask == 0], before[target_mask == 0])


def test_adversarial_donor_occlusion_is_not_used_for_same_target_region():
    """Issue #53: donor pixels hidden by another finger must remain unavailable."""
    clean = _manga_page()
    target = clean.copy()
    target_mask = _mask(target.shape, [(120, 70, 185, 155)])
    target[target_mask > 0] = (35, 100, 190)
    donor_mask = cv2.dilate(target_mask, np.ones((15, 15), np.uint8), iterations=1)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 7, "image": clean, "mask": donor_mask}],
    )

    assert metadata["donors"] == []
    assert np.any(unresolved)
    np.testing.assert_array_equal(repaired, target)


def test_adversarial_unrelated_page_is_rejected():
    """Issue #53: high coverage is not useful if the donor is a different page."""
    target = _manga_page()
    donor = np.full_like(target, 238)
    cv2.line(donor, (20, 25), (300, 195), (15, 15, 15), 12)
    cv2.line(donor, (300, 25), (20, 195), (15, 15, 15), 12)
    zero = np.zeros(target.shape[:2], np.uint8)

    assert align_donor_page(target, donor, zero, zero) is None


def test_adversarial_large_shift_is_rejected():
    """Issue #53: a donor requiring a large shift must not be trusted."""
    target = _manga_page()
    donor = _shift(target, 45, 0)
    zero = np.zeros(target.shape[:2], np.uint8)

    assert align_donor_page(target, donor, zero, zero) is None


def test_adversarial_insufficient_clean_context_is_rejected():
    """Issue #53: alignment needs enough clean target/donor context."""
    target = _manga_page()
    donor = target.copy()
    target_mask = np.full(target.shape[:2], 255, np.uint8)
    donor_mask = np.full(target.shape[:2], 255, np.uint8)
    target_mask[10:35, 10:35] = 0
    donor_mask[10:35, 10:35] = 0

    assert align_donor_page(target, donor, donor_mask, target_mask) is None


def test_adversarial_multiple_regions_preserve_every_pixel_outside_union_mask():
    """Issue #53: multiple donors/components still cannot alter non-hand manga pixels."""
    clean = _manga_page()
    first = (55, 60, 105, 135)
    second = (215, 70, 270, 155)
    target_mask = _mask(clean.shape, [first, second])
    target = clean.copy()
    target[target_mask > 0] = (40, 100, 185)
    before = target.copy()

    donor_one_mask = _mask(clean.shape, [second])
    donor_two_mask = _mask(clean.shape, [first])
    repaired, _metadata, _unresolved = repair_finger_regions(
        target,
        target_mask,
        [
            {"candidate_id": 1, "image": clean, "mask": donor_one_mask},
            {"candidate_id": 3, "image": clean, "mask": donor_two_mask},
        ],
    )

    np.testing.assert_array_equal(repaired[target_mask == 0], before[target_mask == 0])
