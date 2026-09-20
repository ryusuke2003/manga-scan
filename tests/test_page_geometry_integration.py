import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.perspective import warp_roi
from manga_scan.pipeline import (
    detect_spread_page_consensus,
    rectify_spread_pages,
    render_spread,
)
from manga_scan.split import split_spread

REFERENCE = [[0.05, 0.07], [0.96, 0.07], [0.96, 0.94], [0.05, 0.94]]
LEFT = np.asarray([[90, 70], [490, 90], [470, 540], [70, 520]], dtype=np.float32)
RIGHT = np.asarray([[510, 90], [920, 60], [940, 520], [530, 545]], dtype=np.float32)


def synthetic_spread():
    image = np.full((600, 1000, 3), 45, dtype=np.uint8)
    cv2.fillConvexPoly(image, LEFT.astype(np.int32), (235, 235, 235))
    cv2.fillConvexPoly(image, RIGHT.astype(np.int32), (235, 235, 235))
    cv2.rectangle(image, (130, 130), (420, 300), (20, 20, 20), 6)
    cv2.rectangle(image, (560, 140), (860, 350), (20, 20, 20), 6)
    return image


def test_per_page_mode_connects_contour_detection_to_independent_warp(tmp_path):
    image = synthetic_spread()
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0001", "extra_suspect": []}
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)

    assert spread["perspective_mode_used"] == "per_page"
    assert spread["page_contours"]["detected"]
    assert spread["page_contours"]["left"]["detected"]
    assert spread["page_contours"]["right"]["detected"]
    assert "page_contour_low_confidence" not in spread["extra_suspect"]
    assert set(sides) == {"left", "right"}
    assert min(sides["left"].shape[:2]) > 100
    assert min(sides["right"].shape[:2]) > 100
    assert (tmp_path / spread["page_contour_debug"]).is_file()


@pytest.mark.parametrize(
    ("rotation", "source_rotate_code", "source_roi"),
    [
        (
            180,
            cv2.ROTATE_180,
            [[0.04, 0.06], [0.95, 0.06], [0.95, 0.93], [0.04, 0.93]],
        ),
        (
            270,
            cv2.ROTATE_90_CLOCKWISE,
            [[0.06, 0.05], [0.93, 0.05], [0.93, 0.96], [0.06, 0.96]],
        ),
    ],
)
def test_render_spread_rotated_per_page_integration(
    tmp_path,
    monkeypatch,
    rotation,
    source_rotate_code,
    source_roi,
):
    upright = synthetic_spread()
    # Build the fixture independently from manga_scan's rotation helpers so a
    # bug in rotate_image()/rotate_roi() cannot cancel itself inside the test.
    source = cv2.rotate(upright, source_rotate_code)

    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected_spread = {"id": "expected", "extra_suspect": []}
    expected_cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )
    expected_sides = rectify_spread_pages(
        expected_dir,
        upright,
        warp_roi(upright, REFERENCE),
        REFERENCE,
        expected_spread,
        expected_cfg,
    )
    assert expected_spread["perspective_mode_used"] == "per_page"

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
        rotation=rotation,
        output_layout="split",
        reading_order="ltr",
        image_format="png",
        grayscale=False,
        dewarp_mode="off",
        illumination_correction=False,
        white_normalization=False,
    )
    manifest = {
        "source": "unused.mp4",
        "config": cfg.to_dict(),
        "pdf_stale": False,
    }
    spread = {
        "id": f"spread_rot_{rotation}",
        "candidates": [
            {
                "id": 0,
                "time": 1.0,
                "roi": source_roi,
                "metrics": {"score": 1.0},
                "page_suspect": {"left": [], "right": []},
                "suspect": [],
            }
        ],
        "selected": 0,
        "selected_pages": {"left": 0, "right": 0},
        "extra_suspect": [],
    }

    monkeypatch.setattr(
        "manga_scan.pipeline.extract_frame",
        lambda *_args, **_kwargs: source.copy(),
    )

    pages = render_spread(render_dir, manifest, spread)

    assert spread["perspective_mode_used"] == "per_page"
    assert spread["page_contours"]["detected"]
    assert spread["page_contours"]["left"]["detected"]
    assert spread["page_contours"]["right"]["detected"]
    assert "page_contour_low_confidence" not in spread["extra_suspect"]

    by_side = {page["side"]: page for page in pages}
    assert set(by_side) == {"left", "right"}
    for side in ("left", "right"):
        output = cv2.imread(str(render_dir / by_side[side]["path"]))
        assert output is not None
        assert output.shape == expected_sides[side].shape
        assert np.mean(
            np.abs(output.astype(np.int16) - expected_sides[side].astype(np.int16))
        ) < 1.0
        assert by_side[side]["candidate_id"] == 0

    assert (render_dir / spread["page_contour_debug"]).is_file()
    assert manifest["pdf_stale"]


def test_low_confidence_contours_fall_back_to_legacy_spread_split(tmp_path):
    image = np.full((400, 800, 3), 128, dtype=np.uint8)
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0002", "extra_suspect": []}
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)
    expected, expected_spine = split_spread(
        rectified,
        cfg.spine_ratio,
        cfg.split_mode,
        cfg.gutter_fraction,
    )

    assert spread["perspective_mode_used"] == "spread_fallback"
    assert not spread["page_contours"]["detected"]
    assert spread["spine_px"] == expected_spine
    assert "page_contour_low_confidence" in spread["extra_suspect"]
    np.testing.assert_array_equal(sides["left"], expected["left"])
    np.testing.assert_array_equal(sides["right"], expected["right"])
    assert (tmp_path / spread["page_contour_debug"]).is_file()


def test_explicit_spread_mode_is_pixel_compatible_and_skips_detection(tmp_path):
    image = synthetic_spread()
    rectified = warp_roi(image, REFERENCE)
    spread = {"id": "spread_0003", "extra_suspect": []}
    cfg = Config(hand_backend="none", finger_repair=False, perspective_mode="spread")
    expected, expected_spine = split_spread(
        rectified,
        cfg.spine_ratio,
        cfg.split_mode,
        cfg.gutter_fraction,
    )

    sides = rectify_spread_pages(tmp_path, image, rectified, REFERENCE, spread, cfg)

    assert spread["perspective_mode_used"] == "spread"
    assert spread["spine_px"] == expected_spine
    assert "page_contours" not in spread
    assert "page_contour_debug" not in spread
    np.testing.assert_array_equal(sides["left"], expected["left"])
    np.testing.assert_array_equal(sides["right"], expected["right"])


@pytest.mark.parametrize(
    "settings",
    [
        {"perspective_mode": "unknown"},
        {"page_contour_min_confidence": -0.1},
        {"page_contour_min_confidence": 1.1},
    ],
)
def test_per_page_integration_config_validation(settings):
    with pytest.raises(ValueError):
        Config.from_dict(settings)


def _fake_detection(candidate_id, shift=0.0, confidence=0.9):
    left = np.asarray(
        [[.08, .10], [.48, .11], [.47, .90], [.07, .89]],
        dtype=np.float32,
    )
    right = np.asarray(
        [[.52, .11], [.92, .10], [.93, .89], [.53, .90]],
        dtype=np.float32,
    )
    delta = np.asarray([shift, 0], dtype=np.float32)
    return {
        "candidate_id": candidate_id,
        "left": {
            "quad": (left + delta).tolist(),
            "confidence": confidence,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": (right + delta).tolist(),
            "confidence": confidence,
            "detected": True,
            "touches_frame": False,
        },
        "confidence": confidence,
        "detected": True,
    }


def test_detect_spread_page_consensus_uses_saved_candidate_frames(tmp_path, monkeypatch):
    candidates = []
    responses = {}
    for candidate_id, (pixel, shift, confidence) in enumerate(
        ((30, -0.002, 0.85), (60, 0.0, 0.9), (90, 0.002, 0.88), (120, 0.06, 0.99))
    ):
        path = f"candidate_{candidate_id}.png"
        assert cv2.imwrite(str(tmp_path / path), np.full((80, 120, 3), pixel, np.uint8))
        candidates.append({"id": candidate_id, "path": path, "roi": REFERENCE})
        responses[pixel] = _fake_detection(candidate_id, shift, confidence)

    def fake_detect(image, _roi, **_kwargs):
        return responses[int(image[0, 0, 0])]

    monkeypatch.setattr("manga_scan.pipeline.detect_page_quads", fake_detect)
    cfg = Config(hand_backend="none", finger_repair=False)
    spread = {"candidates": candidates, "spine_ratio": 0.5}

    result = detect_spread_page_consensus(
        tmp_path,
        spread,
        cfg,
        {"left": 1, "right": 1},
    )

    assert result["detected"]
    assert result["consensus"]["candidate_count"] == 4
    assert result["left"]["consensus_count"] == 3
    assert result["right"]["consensus_count"] == 3
    assert result["left"]["consensus_outlier_ids"] == [3]
    assert result["right"]["consensus_outlier_ids"] == [3]


def test_rectify_spread_pages_applies_consensus_only_to_selected_side(tmp_path):
    image = synthetic_spread()
    rectified = warp_roi(image, REFERENCE)
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        perspective_mode="per_page",
        page_contour_min_confidence=0.5,
    )
    baseline_state = {"id": "baseline", "extra_suspect": []}
    rectify_spread_pages(tmp_path, image, rectified, REFERENCE, baseline_state, cfg)
    baseline = baseline_state["page_contours"]

    consensus = {
        **baseline,
        "left": {
            **baseline["left"],
            "quad": (
                np.asarray(baseline["left"]["quad"], dtype=np.float32)
                + np.asarray([0.01, 0], dtype=np.float32)
            ).tolist(),
            "consensus_count": 3,
            "consensus_candidate_ids": [0, 1, 2],
            "consensus_outlier_ids": [],
        },
        "consensus": {"candidate_count": 3, "candidate_ids": [0, 1, 2]},
    }
    state = {"id": "consensus", "extra_suspect": []}

    rectify_spread_pages(
        tmp_path,
        image,
        rectified,
        REFERENCE,
        state,
        cfg,
        page_detection=consensus,
        consensus_sides=["left"],
    )

    np.testing.assert_allclose(
        state["page_contours"]["left"]["quad"],
        consensus["left"]["quad"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        state["page_contours"]["right"]["quad"],
        baseline["right"]["quad"],
        atol=1e-6,
    )
    assert state["page_contours"]["left"]["consensus_count"] == 3
