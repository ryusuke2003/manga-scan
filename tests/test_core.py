from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.dedupe import compare, dhash, ssim
from manga_scan.hand import overlap_from_landmarks
from manga_scan.motion import Sample, StableDetector, choose_candidates, motion_score
from manga_scan.page_detect import refine_quad
from manga_scan.perspective import validate_roi, warp_roi
from manga_scan.score import composite_score, sharpness, suspect_reasons
from manga_scan.split import enhance_page, split_spread

ROI = [[0, 0], [1, 0], [1, 1], [0, 1]]


def pattern(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_motion_identical_and_moving():
    image = pattern()
    assert motion_score(image, image) == 0
    assert motion_score(np.zeros_like(image), np.full_like(image, 255)) == 1


def test_state_machine_consecutive_stability_and_eof():
    detector = StableDetector(3, 0.01, 0.03)
    scores = [0.2, 0, 0, 0.02, 0, 0, 0, 0.005, 0.2, 0, 0, 0]
    complete = []
    for i, value in enumerate(scores):
        segment = detector.push(Sample(i, i / 10, value, 100))
        if segment:
            complete.append(segment)
    assert [s.index for s in complete[0]] == [4, 5, 6, 7]
    assert [s.index for s in detector.finish()] == [9, 10, 11]
    assert detector.finish() == []


def test_short_intervals_not_accepted():
    detector = StableDetector(5, 0.01, 0.03)
    for i in range(4):
        assert detector.push(Sample(i, i / 10, 0, 100)) is None
    assert detector.finish() == []


def test_candidates_span_interval_including_late_hand_withdrawal():
    samples = [Sample(i, i / 10, 0, 500 if i < 10 else 100) for i in range(100)]
    chosen = choose_candidates(samples, 7)
    assert len(chosen) == 7
    assert chosen[-1].index >= 85
    assert len({s.index for s in chosen}) == 7


def test_dedupe_identical_changed_and_blank():
    a = pattern()
    b = pattern(2)
    cfg = Config()
    assert dhash(a) == dhash(a.copy())
    assert ssim(a, a) == pytest.approx(1)
    assert compare(a, a, cfg)["duplicate"]
    assert not compare(a, b, cfg)["duplicate"]
    blank = np.full_like(a, 255)
    match = compare(blank, blank, cfg)
    assert match["suspect"] and not match["duplicate"]


def test_one_changed_half_is_not_dropped():
    a = pattern()
    b = a.copy()
    b[:, :80] = pattern(2)[:, :80]
    assert not compare(a, b, Config())["duplicate"]


def test_roi_crop_removes_desk_and_preserves_corners():
    image = np.zeros((201, 301, 3), np.uint8)
    image[40:161, 60:241] = (40, 100, 220)
    cropped = warp_roi(image, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    assert cropped.shape == (121, 181, 3)
    np.testing.assert_array_equal(cropped[50, 50], [40, 100, 220])
    assert (cropped[:, :, 2] > 200).all()
    np.testing.assert_array_equal(warp_roi(image, ROI), image)


@pytest.mark.parametrize(
    "roi",
    [
        [[0, 0], [1, 1], [1, 0], [0, 1]],
        [[0, 0]] * 4,
        [[-1, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [float("nan"), 1], [0, 1]],
        [[1, 0], [1, 1], [0, 1], [0, 0]],
    ],
)
def test_invalid_roi(roi):
    with pytest.raises(ValueError):
        validate_roi(roi)


def test_quad_refinement_falls_back_on_missing_edges():
    roi, ok = refine_quad(np.full((200, 300, 3), 128, np.uint8), ROI)
    assert roi == ROI
    assert not ok


def test_split_no_pixels_lost_on_odd_width():
    image = pattern()[:, :159]
    pages, spine = split_spread(image)
    assert spine in (79, 80)
    np.testing.assert_array_equal(np.concatenate([pages["left"], pages["right"]], axis=1), image)


def test_auto_spine_and_correction():
    image = np.full((120, 200, 3), 230, np.uint8)
    image[:, 104:108] = 0
    _, spine = split_spread(image, mode="auto")
    assert 104 <= spine <= 107
    corrected = enhance_page(image, grayscale=True, rotation=90, dewarp_strength=0.2)
    assert corrected.shape == (200, 120)


def test_sharpness_and_score_penalties():
    image = pattern()
    assert sharpness(image) > sharpness(cv2.GaussianBlur(image, (9, 9), 3))
    cfg = Config()
    clean = dict(
        sharpness=300,
        motion=0.001,
        hand_overlap=0,
        distortion=0,
        flatness_proxy=0,
        clipping=0,
        exposure=0,
    )
    for field in ("motion", "hand_overlap", "distortion", "flatness_proxy", "clipping", "exposure"):
        assert composite_score(clean, cfg) > composite_score({**clean, field: 0.4}, cfg)
    assert composite_score(clean, cfg) > composite_score({**clean, "sharpness": 10}, cfg)
    assert "hand_detection_disabled" in suspect_reasons({**clean, "hand_overlap": None}, cfg)


def test_hand_union_intersection_only_on_page():
    def hand(x1, y1, x2, y2):
        return [SimpleNamespace(x=x, y=y) for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]

    roi = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    hands = [hand(0.3, 0.3, 0.5, 0.5)]
    single, _ = overlap_from_landmarks((101, 101), roi, hands, padding=0)
    doubled, _ = overlap_from_landmarks((101, 101), roi, hands * 2, padding=0)
    outside, _ = overlap_from_landmarks((101, 101), roi, [hand(0, 0, 0.1, 0.1)], padding=0)
    assert 0.15 < single < 0.19
    assert single == doubled
    assert outside == 0


@pytest.mark.parametrize(
    "data",
    [
        {"stable_frames": 0},
        {"jpeg_quality": 101},
        {"hand_backend": "cloud"},
        {"unknown": 1},
        {"motion_weight": float("nan")},
        {"stable_frames": 2.5},
        {"grayscale": "false"},
    ],
)
def test_config_rejects_invalid_values(data):
    with pytest.raises(ValueError):
        Config.from_dict(data)
