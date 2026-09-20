import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.finger_repair import align_donor_page, repair_finger_regions
from manga_scan.pipeline import candidate_page_hand_mask
from manga_scan.split import split_spread
from manga_scan.storage import save_image


def _page(height=180, width=260):
    image = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(image, (20, 20), (width - 20, height - 20), (40, 40, 40), 3)
    cv2.line(image, (35, 55), (width - 40, 70), (80, 80, 80), 4)
    cv2.circle(image, (width // 3, height // 2), 22, (120, 120, 120), -1)
    cv2.putText(
        image,
        "MANGA",
        (70, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return image


def _mask(shape, rectangles):
    mask = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def test_repair_replaces_only_detected_finger_region():
    clean = _page()
    target = clean.copy()
    target_mask = _mask(target.shape, [(95, 65, 145, 125)])
    target[target_mask > 0] = (25, 90, 190)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 2, "image": clean, "mask": np.zeros(target.shape[:2], np.uint8)}],
    )

    assert metadata["status"] == "complete"
    assert metadata["coverage"] == pytest.approx(1.0)
    assert metadata["donors"] == [2]
    assert not np.any(unresolved)
    assert np.mean(np.abs(repaired[90:105, 112:128].astype(int) - clean[90:105, 112:128])) < 2
    np.testing.assert_array_equal(repaired[target_mask == 0], target[target_mask == 0])


def test_repair_combines_multiple_donors_for_different_regions():
    clean = _page()
    target = clean.copy()
    first = (45, 55, 90, 110)
    second = (165, 70, 215, 125)
    target_mask = _mask(target.shape, [first, second])
    target[target_mask > 0] = (30, 110, 200)

    donor_one_mask = _mask(target.shape, [second])
    donor_two_mask = _mask(target.shape, [first])
    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [
            {"candidate_id": 1, "image": clean, "mask": donor_one_mask},
            {"candidate_id": 3, "image": clean, "mask": donor_two_mask},
        ],
    )

    assert metadata["status"] == "complete"
    assert metadata["donors"] == [1, 3]
    assert not np.any(unresolved)
    assert np.mean(np.abs(repaired[70:95, 58:78].astype(int) - clean[70:95, 58:78])) < 2
    assert np.mean(np.abs(repaired[85:110, 180:200].astype(int) - clean[85:110, 180:200])) < 2


def test_repair_keeps_original_when_every_donor_is_occluded():
    clean = _page()
    target = clean.copy()
    target_mask = _mask(target.shape, [(90, 60, 150, 125)])
    target[target_mask > 0] = (25, 90, 190)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [{"candidate_id": 4, "image": clean, "mask": target_mask.copy()}],
    )

    assert metadata["status"] == "incomplete"
    assert metadata["coverage"] == pytest.approx(0.0)
    assert metadata["donors"] == []
    assert np.any(unresolved)
    np.testing.assert_array_equal(repaired, target)


def test_alignment_recovers_small_candidate_translation():
    target = _page()
    matrix = np.float32([[1, 0, 5], [0, 1, -4]])
    donor = cv2.warpAffine(
        target,
        matrix,
        (target.shape[1], target.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    zero = np.zeros(target.shape[:2], np.uint8)

    aligned = align_donor_page(target, donor, zero, zero)

    assert aligned is not None
    aligned_image, aligned_mask, score = aligned
    assert score > 0.8
    assert np.mean(np.abs(aligned_image.astype(int) - target.astype(int))) < 8
    assert np.count_nonzero(aligned_mask[12:-12, 12:-12]) == 0


def test_candidate_hand_mask_uses_same_spread_page_coordinates(tmp_path):
    frame = _page(height=100, width=200)
    hand_mask = _mask(frame.shape, [(10, 20, 70, 80)])
    save_image(tmp_path / "candidate_mask.png", hand_mask)
    rectified = frame.copy()
    sides, spine = split_spread(rectified, 0.5, "center", 0.0)
    data = {
        "chosen": {
            "hand_mask": "candidate_mask.png",
            "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
        "source_frame_shape": frame.shape,
        "rectified": rectified,
        "sides": sides,
        "state": {
            "perspective_mode_used": "spread",
            "spine_px": spine,
            "spine_ratio": 0.5,
        },
    }
    cfg = Config(hand_backend="mediapipe")

    left = candidate_page_hand_mask(tmp_path, data, "left", cfg)
    right = candidate_page_hand_mask(tmp_path, data, "right", cfg)

    assert np.count_nonzero(left) > 0
    assert np.count_nonzero(right) == 0
    assert left.shape == sides["left"].shape[:2]
    assert right.shape == sides["right"].shape[:2]


def test_finger_repair_requires_mediapipe():
    with pytest.raises(ValueError, match="finger_repair requires"):
        Config.from_dict({"finger_repair": True, "hand_backend": "none"})
