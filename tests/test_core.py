import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import manga_scan.split as split_module
from manga_scan.config import Config
from manga_scan.dedupe import compare, dhash, ssim
from manga_scan.hand import overlap_from_landmarks
from manga_scan.motion import Sample, StableDetector, choose_candidates, motion_score
from manga_scan.page_detect import refine_quad
from manga_scan.perspective import rotate_roi, validate_roi, warp_roi
from manga_scan.score import composite_score, sharpness, suspect_reasons
from manga_scan.split import (
    auto_dewarp_page,
    dewarp_debug_grid,
    dewarp_page_profile,
    enhance_page,
    estimate_curvature,
    normalize_white_background,
    rotate_image,
    split_spread,
)

ROI = [[0, 0], [1, 0], [1, 1], [0, 1]]


def pattern(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def synthetic_line_page():
    image = np.full((240, 480, 3), 240, dtype=np.uint8)
    for x in range(15, 470, 30):
        cv2.line(image, (x, 8), (x, 231), (25, 25, 25), 2)
    for y in range(30, 220, 45):
        cv2.line(image, (8, y), (471, y), (110, 110, 110), 1)
    return image


def compress_spine(image, side, strength):
    h, w = image.shape[:2]
    gamma = 1.0 + 3.0 * strength
    u = np.linspace(0, 1, w, dtype=np.float32)
    if side == "right":
        source_u = np.power(u, 1.0 / gamma)
    else:
        source_u = 1.0 - np.power(1.0 - u, 1.0 / gamma)
    map_x = np.tile(source_u * (w - 1), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def compress_spine_profile(image, side, strength_profile):
    h, w = image.shape[:2]
    ys = np.asarray([point["y"] * max(1, h - 1) for point in strength_profile], dtype=np.float32)
    strengths = np.asarray([point["strength"] for point in strength_profile], dtype=np.float32)
    row_strengths = np.interp(np.arange(h, dtype=np.float32), ys, strengths)
    gamma = 1.0 + 3.0 * row_strengths[:, None]
    u = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    if side == "right":
        source_u = np.power(u, 1.0 / gamma)
    else:
        source_u = 1.0 - np.power(1.0 - u, 1.0 / gamma)
    map_x = (source_u * (w - 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def test_motion_identical_and_moving():
    image = pattern()
    assert motion_score(image, image) == 0
    assert motion_score(np.zeros_like(image), np.full_like(image, 255)) == 1


def test_motion_v2_tolerates_moderate_auto_exposure_change():
    image = synthetic_line_page()
    changed = np.clip(image.astype(np.float32) * 0.90 + 15, 0, 255).astype(np.uint8)

    assert motion_score(image, changed) < Config().motion_threshold


def test_motion_v2_tolerates_focus_breathing():
    image = synthetic_line_page()
    softened = cv2.GaussianBlur(image, (0, 0), sigmaX=0.9, sigmaY=0.9)

    assert motion_score(image, softened) < Config().motion_threshold


def test_motion_v2_tolerates_small_jitter_with_ae_and_af_change():
    image = synthetic_line_page()
    height, width = image.shape[:2]
    shifted = cv2.warpAffine(
        image,
        np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    shifted = cv2.GaussianBlur(shifted, (0, 0), sigmaX=0.65, sigmaY=0.65)
    shifted = np.clip(shifted.astype(np.float32) * 0.94 + 9, 0, 255).astype(np.uint8)

    assert motion_score(image, shifted) < Config().motion_threshold


def test_motion_v2_still_detects_real_page_content_change():
    image = synthetic_line_page()
    changed = image.copy()
    cv2.rectangle(changed, (250, 28), (455, 212), (18, 18, 18), -1)
    cv2.line(changed, (270, 45), (430, 195), (245, 245, 245), 5)

    assert motion_score(image, changed) > Config().turn_threshold


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


def test_scan_style_defaults_and_hand_disabled_compatibility():
    cfg = Config().validate()
    assert cfg.reading_order == "rtl"
    assert cfg.image_format == "png"
    assert cfg.jpeg_quality == 92
    assert cfg.hand_backend == "mediapipe"
    assert cfg.finger_repair is True
    assert cfg.page_background_fill == "paper"
    assert cfg.candidate_selection_mode == "spread"
    assert cfg.grayscale is False
    assert cfg.refine_quad is True
    assert cfg.perspective_mode == "spread"
    assert cfg.page_contour_min_confidence == pytest.approx(0.55)
    assert cfg.split_mode == "auto"
    assert cfg.dewarp_mode == "auto"
    assert cfg.illumination_correction is True
    assert cfg.white_normalization is True
    assert cfg.auto_rotation is True
    assert cfg.hwaccel == ("videotoolbox" if sys.platform == "darwin" else "none")

    disabled = Config.from_dict({"hand_backend": "none"})
    assert disabled.hand_backend == "none"
    assert disabled.finger_repair is False
    assert disabled.page_background_fill == "preserve"


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


def test_dedupe_aligns_handheld_camera_shift():
    image = np.full((480, 720, 3), 245, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(100):
        x1, y1 = rng.integers([10, 10], [680, 440])
        x2 = min(710, x1 + int(rng.integers(8, 55)))
        y2 = min(470, y1 + int(rng.integers(8, 45)))
        shade = int(rng.integers(10, 190))
        cv2.rectangle(image, (x1, y1), (x2, y2), (shade,) * 3, 2)
    shifted = cv2.warpPerspective(
        image,
        np.float32([[1.01, 0.015, 18], [-0.01, 0.99, 12], [0.00002, -0.00003, 1]]),
        (720, 480),
        borderValue=(235, 235, 235),
    )

    result = compare(image, shifted, Config())

    assert result["duplicate"] is True
    assert result["alignment"]["available"] is True
    assert result["alignment"]["correlation"] >= 0.45


def test_aligned_match_on_only_one_half_is_not_dropped():
    left = synthetic_line_page()
    right = left.copy()
    rng = np.random.default_rng(4)
    right[:, right.shape[1] // 2 :] = rng.integers(
        0,
        255,
        right[:, right.shape[1] // 2 :].shape,
        dtype=np.uint8,
    )
    shifted = cv2.warpAffine(
        right,
        np.float32([[1, 0, 8], [0, 1, 5]]),
        (right.shape[1], right.shape[0]),
        borderValue=(240, 240, 240),
    )

    assert compare(left, shifted, Config())["duplicate"] is False


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


def test_rotate_before_split_uses_visual_left_and_right():
    upright = np.empty((60, 120, 3), np.uint8)
    upright[:, :60] = (20, 40, 220)
    upright[:, 60:] = (220, 80, 20)
    sideways = np.rot90(upright, 1).copy()

    oriented = rotate_image(sideways, 90)
    pages, spine = split_spread(oriented)

    assert oriented.shape == upright.shape
    assert spine == 60
    np.testing.assert_array_equal(pages["left"][30, 30], (20, 40, 220))
    np.testing.assert_array_equal(pages["right"][30, 30], (220, 80, 20))


def test_rotate_roi_keeps_tl_tr_br_bl_order():
    roi = [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]
    rotated = rotate_roi(roi, 90)
    np.testing.assert_allclose(
        rotated,
        [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]],
        atol=1e-6,
    )

    np.testing.assert_allclose(rotate_roi(rotated, 270), roi, atol=1e-6)


def test_explicit_legacy_rotation_disables_new_auto_detection_default():
    cfg = Config.from_dict({"rotation": 270})
    assert cfg.rotation == 270
    assert cfg.auto_rotation is False
    assert Config.from_dict({"rotation": 0}).auto_rotation is True
    assert Config.from_dict({}).auto_rotation is True


def test_auto_spine_and_correction():
    image = np.full((120, 200, 3), 230, np.uint8)
    image[:, 104:108] = 0
    _, spine = split_spread(image, mode="auto")
    assert 104 <= spine <= 107
    corrected = enhance_page(image, grayscale=True, rotation=90, dewarp_strength=0.2)
    assert corrected.shape == (200, 120)


@pytest.mark.parametrize("side", ["left", "right"])
def test_auto_curvature_dewarp_improves_synthetic_spine_compression(side):
    flat = synthetic_line_page()
    distorted = compress_spine(flat, side, 0.2)
    before = estimate_curvature(distorted, side)
    corrected, info = auto_dewarp_page(distorted, side, max_strength=0.25, min_confidence=0.6)
    after = estimate_curvature(corrected, side)

    assert before["confidence"] >= 0.6
    assert before["strength"] > 0
    assert info["applied"]
    assert corrected.shape == distorted.shape
    assert corrected.dtype == distorted.dtype
    assert after["compression_ratio"] is not None
    assert abs(after["compression_ratio"] - 1) < abs(before["compression_ratio"] - 1)


def test_curvature_confidence_rejects_jagged_scanline_measurements(monkeypatch):
    image = synthetic_line_page()
    ratios = iter([0.70, 1.08, 0.72, 1.05, 0.69, 1.10, 0.73, 1.04, 0.71])
    monkeypatch.setattr(split_module, "_spacing_ratio", lambda *args: next(ratios))

    estimate = estimate_curvature(image, "right", max_strength=0.25)

    assert estimate["bands"] == 9
    assert estimate["strength"] > 0
    assert estimate["confidence"] < 0.6


def test_curvature_confidence_rejects_single_large_outlier(monkeypatch):
    image = synthetic_line_page()
    ratios = iter([0.78, 0.79, 0.78, 0.79, 1.35, 0.79, 0.78, 0.79, 0.78])
    monkeypatch.setattr(split_module, "_spacing_ratio", lambda *args: next(ratios))

    estimate = estimate_curvature(image, "right", max_strength=0.25)

    assert estimate["bands"] == 9
    assert estimate["strength"] > 0
    assert estimate["confidence"] < 0.6


@pytest.mark.parametrize("side", ["left", "right"])
def test_profiled_dewarp_handles_height_varying_book_curve(side):
    flat = synthetic_line_page()
    profile = [
        {"y": 0.0, "strength": 0.05},
        {"y": 0.25, "strength": 0.11},
        {"y": 0.5, "strength": 0.24},
        {"y": 0.75, "strength": 0.15},
        {"y": 1.0, "strength": 0.07},
    ]
    distorted = compress_spine_profile(flat, side, profile)
    corrected = dewarp_page_profile(distorted, side, profile)

    before_error = np.mean(np.abs(distorted.astype(np.int16) - flat.astype(np.int16)))
    after_error = np.mean(np.abs(corrected.astype(np.int16) - flat.astype(np.int16)))
    assert after_error < before_error * 0.85
    assert corrected.shape == flat.shape


@pytest.mark.parametrize("side", ["left", "right"])
def test_auto_curvature_estimates_height_profile(side):
    flat = synthetic_line_page()
    distorted = compress_spine_profile(
        flat,
        side,
        [
            {"y": 0.0, "strength": 0.04},
            {"y": 0.5, "strength": 0.25},
            {"y": 1.0, "strength": 0.06},
        ],
    )
    corrected, info = auto_dewarp_page(
        distorted,
        side,
        max_strength=0.3,
        min_confidence=0.55,
    )

    assert info["applied"]
    assert len(info["strength_profile"]) == 9
    assert info["profile_variation"] >= 0.015
    assert info["strength"] == max(point["strength"] for point in info["strength_profile"])
    assert corrected.shape == distorted.shape


def test_auto_curvature_dewarp_falls_back_on_low_information_page():
    blank = np.full((160, 240, 3), 230, dtype=np.uint8)
    corrected, info = auto_dewarp_page(blank, "right")
    np.testing.assert_array_equal(corrected, blank)
    assert not info["applied"]
    assert info["status"] == "low_confidence"
    assert info["confidence"] == 0


def test_curvature_dewarp_keeps_bounds_without_holes():
    image = np.full((120, 150, 3), 180, dtype=np.uint8)
    profile = [
        {"y": 0.0, "strength": 0.05},
        {"y": 0.5, "strength": 0.25},
        {"y": 1.0, "strength": 0.08},
    ]
    corrected = dewarp_page_profile(image, "left", profile)
    grid = dewarp_debug_grid(image.shape, "left", 0.25, profile)
    assert corrected.shape == image.shape
    assert corrected.dtype == image.dtype
    assert grid.shape == image.shape
    assert corrected.min() == 180
    assert corrected.max() == 180
    assert np.isfinite(grid).all()


def test_white_normalization_brightens_paper_without_lifting_dark_art():
    image = np.full((120, 160, 3), (205, 218, 230), np.uint8)
    image[15:45, 15:55] = (20, 20, 20)
    image[60:90, 15:55] = (155, 155, 155)

    corrected = normalize_white_background(image, target=245, strength=0.8)

    paper_before = image[0, 0].astype(int)
    paper_after = corrected[0, 0].astype(int)
    assert paper_after.mean() > paper_before.mean()
    assert np.ptp(paper_after) < np.ptp(paper_before)
    assert np.abs(corrected[25, 25].astype(int) - image[25, 25].astype(int)).max() <= 2
    assert np.abs(corrected[75, 25].astype(int) - image[75, 25].astype(int)).max() <= 4


def test_white_normalization_matches_previous_color_math():
    image = np.full((96, 128, 3), (205, 218, 230), np.uint8)
    image[12:40, 12:48] = (25, 25, 25)
    image[48:78, 12:48] = (155, 155, 155)
    image[20:70, 70:110] = (80, 120, 210)

    def legacy(image, target=245, strength=0.6):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[:, :, 0]
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        chroma = np.sqrt(a * a + b * b)

        candidate_floor = max(160.0, float(np.percentile(lightness, 70)))
        candidates = (lightness >= candidate_floor) & (chroma <= 30.0)
        white_level = float(np.percentile(lightness[candidates], 75))
        transition_start = max(150.0, white_level - 45.0)
        x = np.clip(
            (lightness - transition_start) / max(1.0, white_level - transition_start),
            0.0,
            1.0,
        )
        weight = x * x * (3.0 - 2.0 * x) * float(strength)

        gain = min(1.18, max(1.0, float(target) / max(1.0, white_level)))
        brightened = np.clip(lightness * gain, 0, 255)
        lab[:, :, 0] = lightness * (1.0 - weight) + brightened * weight

        x = np.clip((30.0 - chroma) / 20.0, 0.0, 1.0)
        chroma_weight = weight * (x * x * (3.0 - 2.0 * x))
        lab[:, :, 1] = 128.0 + a * (1.0 - chroma_weight)
        lab[:, :, 2] = 128.0 + b * (1.0 - chroma_weight)
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    expected = legacy(image, strength=0.8)
    actual = normalize_white_background(image, strength=0.8)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1)


def test_white_normalization_grayscale_and_disabled_compatibility():
    image = np.full((80, 100), 220, np.uint8)
    image[20:60, 20:50] = 140

    corrected = normalize_white_background(image, target=245, strength=0.6)
    assert corrected[0, 0] > image[0, 0]
    assert corrected[30, 30] == image[30, 30]
    np.testing.assert_array_equal(
        enhance_page(image, white_normalization=False),
        image,
    )


def test_sharpness_and_score_penalties():
    image = pattern()
    assert sharpness(image) > sharpness(cv2.GaussianBlur(image, (9, 9), 3))
    cfg = Config()
    clean = dict(
        sharpness=300,
        motion=0.001,
        hand_overlap=0,
        glare_overlap=0,
        distortion=0,
        flatness_proxy=0,
        clipping=0,
        exposure=0,
    )
    for field in ("motion", "hand_overlap", "glare_overlap", "distortion", "flatness_proxy", "clipping", "exposure"):
        assert composite_score(clean, cfg) > composite_score({**clean, field: 0.4}, cfg)
    assert composite_score(clean, cfg) > composite_score({**clean, "sharpness": 10}, cfg)
    assert "hand_detection_disabled" in suspect_reasons({**clean, "hand_overlap": None}, cfg)
    assert "glare_overlap" in suspect_reasons({**clean, "glare_overlap": 0.2}, cfg)


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
        {"white_normalization": "true"},
        {"white_target": 199},
        {"white_strength": 1.1},
        {"dewarp_mode": "guess"},
        {"dewarp_max_strength": 0.5},
        {"dewarp_min_confidence": 1.1},
        {"page_background_fill": "paint"},
    ],
)
def test_config_rejects_invalid_values(data):
    with pytest.raises(ValueError):
        Config.from_dict(data)
