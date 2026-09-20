import cv2
import numpy as np
import pytest

from manga_scan.finger_repair import (
    _component_records,
    _validate_local_candidate,
    repair_finger_regions,
)


def _textured_page(height=220, width=320):
    image = np.full((height, width, 3), 238, np.uint8)
    cv2.rectangle(image, (14, 14), (width - 14, height - 14), (30, 30, 30), 3)
    for x in range(35, width - 25, 28):
        cv2.line(image, (x, 25), (x - 12, height - 30), (75, 75, 75), 2)
    for y in range(42, height - 25, 30):
        cv2.line(image, (24, y), (width - 24, y + 8), (105, 105, 105), 2)
    cv2.putText(
        image,
        "LOCAL ALIGN",
        (58, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, (235, 155), 24, (125, 125, 125), 3)
    return image


def _mask(shape, rectangles):
    mask = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _local_deform(image, rectangle, dx, dy):
    height, width = image.shape[:2]
    shifted = cv2.warpAffine(
        image,
        np.asarray([[1, 0, dx], [0, 1, dy]], np.float32),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    x1, y1, x2, y2 = rectangle
    result = image.copy()
    result[y1:y2, x1:x2] = shifted[y1:y2, x1:x2]
    return result


def test_local_alignment_repairs_residual_component_shift():
    clean = _textured_page()
    target_mask = _mask(clean.shape, [(120, 82, 178, 148)])
    target = clean.copy()
    target[target_mask > 0] = (25, 95, 195)

    donor = _local_deform(clean, (90, 52, 210, 180), 5, -4)
    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 7, "image": donor, "mask": np.zeros(target.shape[:2], np.uint8)}],
        min_coverage=1.0,
    )

    assert metadata["status"] == "complete"
    assert metadata["donors"] == [7]
    assert not np.any(unresolved)
    donor_meta = metadata["components"][0]["donors"][0]
    assert donor_meta["method"] == "local"
    assert abs(donor_meta["dx"]) >= 1 or abs(donor_meta["dy"]) >= 1
    assert donor_meta["context_residual"] < 0.18
    assert np.mean(
        np.abs(repaired[target_mask > 0].astype(int) - clean[target_mask > 0].astype(int))
    ) < 12
    np.testing.assert_array_equal(repaired[target_mask == 0], target[target_mask == 0])


def test_one_component_can_be_completed_by_multiple_donors():
    clean = _textured_page()
    target_mask = _mask(clean.shape, [(105, 76, 205, 154)])
    target = clean.copy()
    target[target_mask > 0] = (30, 105, 205)

    donor_one_mask = _mask(clean.shape, [(105, 76, 145, 154)])
    donor_two_mask = _mask(clean.shape, [(165, 76, 205, 154)])
    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [
            {"candidate_id": 1, "image": clean, "mask": donor_one_mask},
            {"candidate_id": 2, "image": clean, "mask": donor_two_mask},
        ],
        min_coverage=1.0,
    )

    assert metadata["status"] == "complete"
    assert not np.any(unresolved)
    assert metadata["donors"] == [1, 2]
    component = metadata["components"][0]
    assert component["coverage"] == pytest.approx(1.0)
    assert {entry["candidate_id"] for entry in component["donors"]} == {1, 2}
    # Existing feathering intentionally preserves a narrow transition at
    # the target-mask boundary; the donor content should still dominate the
    # repaired component overall.
    assert np.mean(
        np.abs(repaired[target_mask > 0].astype(int) - clean[target_mask > 0].astype(int))
    ) < 5
    np.testing.assert_allclose(repaired[110, 155], clean[110, 155], atol=5)


def test_different_components_can_choose_different_donors():
    clean = _textured_page()
    left = (52, 70, 105, 132)
    right = (215, 82, 270, 148)
    target_mask = _mask(clean.shape, [left, right])
    target = clean.copy()
    target[target_mask > 0] = (35, 100, 205)

    donor_one_mask = _mask(clean.shape, [right])
    donor_two_mask = _mask(clean.shape, [left])
    _, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [
            {"candidate_id": 3, "image": clean, "mask": donor_one_mask},
            {"candidate_id": 4, "image": clean, "mask": donor_two_mask},
        ],
        min_coverage=1.0,
    )

    assert not np.any(unresolved)
    assert len(metadata["components"]) == 2
    used = [
        {entry["candidate_id"] for entry in component["donors"]}
        for component in metadata["components"]
    ]
    assert {3} in used
    assert {4} in used


def test_donor_occluding_same_region_is_not_used():
    clean = _textured_page()
    target_mask = _mask(clean.shape, [(112, 78, 195, 150)])
    target = clean.copy()
    target[target_mask > 0] = (25, 95, 195)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 8, "image": clean, "mask": target_mask.copy()}],
        min_coverage=1.0,
    )

    assert metadata["status"] == "incomplete"
    assert metadata["donors"] == []
    assert np.any(unresolved)
    np.testing.assert_array_equal(repaired, target)


def test_donor_mask_safety_margin_stays_unresolved():
    clean = _textured_page()
    target_mask = _mask(clean.shape, [(105, 75, 205, 155)])
    target = clean.copy()
    target[target_mask > 0] = (25, 95, 195)
    donor_mask = _mask(clean.shape, [(150, 75, 156, 155)])

    _, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 9, "image": clean, "mask": donor_mask}],
        min_coverage=1.0,
    )

    assert metadata["status"] == "incomplete"
    # The donor mask is deliberately narrow; the safety margin must keep
    # neighboring pixels unresolved as well instead of copying a finger edge.
    assert np.count_nonzero(unresolved[75:156, 147:160]) > 0


def test_unrelated_content_is_rejected_without_touching_target():
    clean = _textured_page()
    target_mask = _mask(clean.shape, [(110, 80, 190, 150)])
    target = clean.copy()
    target[target_mask > 0] = (25, 95, 195)

    unrelated = np.full_like(clean, 238)
    cv2.line(unrelated, (5, 5), (315, 215), (10, 10, 10), 14)
    cv2.line(unrelated, (315, 5), (5, 215), (10, 10, 10), 14)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 10, "image": unrelated, "mask": np.zeros(target.shape[:2], np.uint8)}],
        min_coverage=1.0,
    )

    assert metadata["donors"] == []
    assert np.any(unresolved)
    np.testing.assert_array_equal(repaired, target)


def test_local_shift_above_safety_limit_is_rejected():
    target = _textured_page()
    target_mask = _mask(target.shape, [(112, 78, 195, 150)])
    component = _component_records((target_mask > 127).astype(np.uint8))[0]
    zero = np.zeros(target.shape[:2], np.uint8)

    candidate = _validate_local_candidate(
        target,
        target,
        zero,
        target_mask,
        component,
        dx=40.0,
        dy=0.0,
        score=0.99,
        method="local",
    )

    assert candidate is None
