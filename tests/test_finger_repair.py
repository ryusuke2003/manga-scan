import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.finger_repair import align_donor_page, repair_finger_regions
from manga_scan.pipeline import candidate_page_hand_mask
from manga_scan.split import rotate_image, split_spread
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


def test_paper_fallback_conceals_unresolved_finger_on_plain_paper():
    paper = np.full((180, 260, 3), (225, 228, 232), np.uint8)
    target = paper.copy()
    target_mask = _mask(target.shape, [(100, 70, 150, 125)])
    target[target_mask > 0] = (35, 95, 190)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [],
        fallback="paper",
    )

    assert metadata["status"] == "complete"
    assert metadata["donor_coverage"] == pytest.approx(0.0)
    assert metadata["fallback"]["mode"] == "paper"
    assert metadata["fallback"]["applied"]
    assert metadata["fallback"]["filled_fraction"] == pytest.approx(1.0)
    assert not np.any(unresolved)
    np.testing.assert_allclose(repaired[95, 125], paper[95, 125], atol=2)
    np.testing.assert_array_equal(repaired[target_mask == 0], target[target_mask == 0])


def test_paper_fallback_preserves_unresolved_region_near_artwork():
    target = np.full((180, 260, 3), 235, np.uint8)
    target_mask = _mask(target.shape, [(100, 70, 150, 125)])
    for x in range(70, 185, 8):
        cv2.line(target, (x, 45), (x, 150), (25, 25, 25), 3)
    target[target_mask > 0] = (35, 95, 190)
    before = target.copy()

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [],
        fallback="paper",
    )

    assert metadata["status"] == "incomplete"
    assert metadata["fallback"]["mode"] == "paper"
    assert not metadata["fallback"]["applied"]
    assert np.any(unresolved)
    np.testing.assert_array_equal(repaired, before)


def test_white_fallback_is_explicit_and_fills_all_unresolved_pixels():
    target = _page()
    target_mask = _mask(target.shape, [(95, 65, 145, 125)])
    target[target_mask > 0] = (25, 90, 190)

    repaired, metadata, unresolved = repair_finger_regions(
        target,
        target_mask,
        [],
        fallback="white",
    )

    assert metadata["status"] == "complete"
    assert metadata["fallback"]["mode"] == "white"
    assert metadata["fallback"]["applied"]
    assert not np.any(unresolved)
    np.testing.assert_array_equal(repaired[95, 120], np.array([255, 255, 255], np.uint8))


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


def test_alignment_rejects_unrelated_page_content():
    target = _page()
    donor = np.full_like(target, 235)
    cv2.line(donor, (10, 20), (250, 165), (20, 20, 20), 12)
    cv2.line(donor, (250, 20), (10, 165), (20, 20, 20), 12)
    zero = np.zeros(target.shape[:2], np.uint8)

    assert align_donor_page(target, donor, zero, zero) is None


def test_alignment_rejects_excessive_page_rotation():
    target = _page()
    center = (target.shape[1] / 2, target.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, 12, 1.0)
    donor = cv2.warpAffine(
        target,
        matrix,
        (target.shape[1], target.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    zero = np.zeros(target.shape[:2], np.uint8)

    assert align_donor_page(target, donor, zero, zero) is None


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


@pytest.mark.parametrize("rotation", [90, 270])
def test_candidate_hand_mask_follows_pre_split_rotation(tmp_path, rotation):
    frame = _page(height=100, width=200)
    hand_mask = _mask(frame.shape, [(12, 18, 65, 78)])
    save_image(tmp_path / "rotated_mask.png", hand_mask)

    rectified = rotate_image(frame, rotation)
    expected_mask = rotate_image(hand_mask, rotation)
    sides, spine = split_spread(rectified, 0.5, "center", 0.0)
    expected_sides, _ = split_spread(expected_mask, 0.5, "center", 0.0)
    data = {
        "chosen": {
            "hand_mask": "rotated_mask.png",
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
    cfg = Config(hand_backend="mediapipe", rotation=rotation)

    np.testing.assert_array_equal(
        candidate_page_hand_mask(tmp_path, data, "left", cfg),
        expected_sides["left"],
    )
    np.testing.assert_array_equal(
        candidate_page_hand_mask(tmp_path, data, "right", cfg),
        expected_sides["right"],
    )


def test_finger_repair_requires_mediapipe():
    with pytest.raises(ValueError, match="finger_repair requires"):
        Config.from_dict({"finger_repair": True, "hand_backend": "none"})


def test_finger_repair_fallback_rejects_unknown_mode():
    with pytest.raises(ValueError, match="finger_repair_fallback"):
        Config.from_dict({"finger_repair_fallback": "paint"})
