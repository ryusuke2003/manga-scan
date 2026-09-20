import cv2
import numpy as np

from manga_scan.hand import boundary_finger_mask, temporal_transient_mask

ROI = [[0, 0], [1, 0], [1, 1], [0, 1]]


def page():
    image = np.full((400, 300, 3), (205, 215, 220), np.uint8)
    cv2.rectangle(image, (15, 20), (270, 380), (30, 30, 30), 3)
    return image


def test_fingertip_without_landmarks_is_detected_at_page_edge():
    image = page()
    cv2.ellipse(image, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    mask = boundary_finger_mask(image, ROI)
    assert mask[340, 285] == 255
    assert not np.any(mask[40:280, 30:240])


def test_interior_color_and_monochrome_ink_are_not_fingertips():
    image = page()
    cv2.circle(image, (150, 200), 25, (75, 90, 140), -1)
    cv2.circle(image, (298, 340), 30, (25, 25, 25), -1)
    assert not np.any(boundary_finger_mask(image, ROI))


def test_color_page_and_long_book_cover_strip_are_not_fingertips():
    color = np.full((400, 300, 3), (75, 90, 140), np.uint8)
    assert not np.any(boundary_finger_mask(color, ROI))
    image = page()
    image[:, -15:] = (55, 65, 155)
    assert not np.any(boundary_finger_mask(image, ROI))


def test_temporal_transient_detects_edge_finger_missed_by_landmarks():
    clean = page()
    target = clean.copy()
    cv2.ellipse(target, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    zero = np.zeros(target.shape[:2], np.uint8)
    peers = [{"image": clean.copy(), "mask": zero.copy()} for _ in range(3)]

    mask = temporal_transient_mask(
        target,
        ROI,
        peers,
        target_mask=zero,
    )

    assert mask[340, 285] == 255
    assert not np.any(mask[40:260, 40:240])


def test_temporal_transient_ignores_stable_art_and_unconnected_center_change():
    stable = page()
    cv2.circle(stable, (150, 200), 28, (35, 35, 35), -1)
    zero = np.zeros(stable.shape[:2], np.uint8)
    stable_peers = [{"image": stable.copy(), "mask": zero.copy()} for _ in range(3)]
    assert not np.any(
        temporal_transient_mask(stable, ROI, stable_peers, target_mask=zero)
    )

    transient_center = stable.copy()
    cv2.circle(transient_center, (150, 200), 24, (75, 90, 140), -1)
    clean = page()
    center_peers = [{"image": clean.copy(), "mask": zero.copy()} for _ in range(3)]
    assert not np.any(
        temporal_transient_mask(
            transient_center,
            ROI,
            center_peers,
            target_mask=zero,
        )
    )


def test_temporal_transient_excludes_known_peer_hands_from_median():
    clean = page()
    target = clean.copy()
    cv2.ellipse(target, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    zero = np.zeros(target.shape[:2], np.uint8)

    occluded_peer = clean.copy()
    cv2.ellipse(occluded_peer, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    peer_mask = np.zeros(target.shape[:2], np.uint8)
    cv2.ellipse(peer_mask, (298, 340), (32, 27), 0, 0, 360, 255, -1)
    peers = [
        {"image": occluded_peer, "mask": peer_mask},
        {"image": clean.copy(), "mask": zero.copy()},
        {"image": clean.copy(), "mask": zero.copy()},
    ]

    mask = temporal_transient_mask(
        target,
        ROI,
        peers,
        target_mask=zero,
    )

    assert mask[340, 285] == 255


def test_disabled_hand_backend_does_not_enable_color_heuristic():
    from manga_scan.config import Config
    from manga_scan.hand import HandDetector

    image = page()
    cv2.ellipse(image, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    detector = HandDetector(Config(hand_backend="none", finger_repair=False))
    overlap, mask = detector.detect(image, ROI)
    assert overlap is None
    assert not np.any(mask)


def test_existing_empty_landmark_mask_is_supplemented_at_render_time(tmp_path):
    from manga_scan.config import Config
    from manga_scan.pipeline import candidate_page_hand_mask
    from manga_scan.storage import save_image

    image = page()
    cv2.ellipse(image, (298, 340), (30, 25), 0, 0, 360, (75, 90, 140), -1)
    spread = np.concatenate([page(), image], axis=1)
    save_image(tmp_path / "mask.png", np.zeros(spread.shape[:2], np.uint8))
    data = {
        "chosen": {"hand_mask": "mask.png", "roi": ROI},
        "source_frame_shape": spread.shape,
        "rectified": spread,
        "sides": {"left": page(), "right": image},
        "state": {"perspective_mode_used": "spread", "spine_px": 300},
    }
    mask = candidate_page_hand_mask(tmp_path, data, "right", Config())
    assert mask[340, 285] == 255
    assert not np.any(candidate_page_hand_mask(tmp_path, data, "left", Config()))
