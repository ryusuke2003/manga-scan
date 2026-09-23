import cv2
import numpy as np

from manga_scan.config import Config
from manga_scan.ingest import _reference_consensus_frames, set_setup_frame
from manga_scan.perspective import rotate_roi
from manga_scan.spread_detect import (
    _adjust_prior,
    _refine_priors,
    _select_reference_search_frame,
    detect_reference_spread,
    detect_reference_spread_consensus,
)
from manga_scan.storage import save_manifest


def _synthetic_spread():
    image = np.full((600, 1000, 3), (45, 85, 125), dtype=np.uint8)
    left = np.asarray([[80, 65], [490, 88], [475, 535], [60, 515]], dtype=np.int32)
    right = np.asarray([[510, 88], [930, 55], [950, 520], [525, 540]], dtype=np.int32)
    cv2.fillConvexPoly(image, left, (235, 235, 230))
    cv2.fillConvexPoly(image, right, (235, 235, 230))
    cv2.polylines(image, [left, right], True, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.rectangle(image, (125, 145), (420, 315), (35, 35, 35), 5)
    cv2.rectangle(image, (565, 135), (870, 350), (35, 35, 35), 5)
    return image


def test_detect_reference_spread_finds_two_pages_from_full_frame():
    result = detect_reference_spread(_synthetic_spread(), min_confidence=0.5)

    assert result["detected"]
    assert result["stage"] == "complete"
    assert result["pages"]["left"]["detected"]
    assert result["pages"]["right"]["detected"]
    assert result["confidence"] >= 0.5
    roi = np.asarray(result["roi"])
    assert roi.shape == (4, 2)
    assert roi[0, 0] < 0.151
    assert roi[1, 0] > 0.849


def test_detect_reference_spread_rejects_blank_frame():
    image = np.full((480, 800, 3), 180, dtype=np.uint8)

    result = detect_reference_spread(image)

    assert not result["detected"]
    assert result["roi"] is None
    assert result["stage"] == "outline"


def test_adjust_prior_can_search_inward_as_well_as_outward():
    prior = np.asarray([[.1, .1], [.9, .1], [.9, .9], [.1, .9]], np.float32)

    inset = _adjust_prior(prior, top=-0.1, bottom=-0.1)
    outset = _adjust_prior(prior, top=0.1, bottom=0.1)

    assert inset[0, 1] > prior[0, 1]
    assert inset[3, 1] < prior[3, 1]
    assert outset[0, 1] < prior[0, 1]
    assert outset[3, 1] > prior[3, 1]


def test_prior_local_search_moves_edges_toward_stronger_page_detection(monkeypatch):
    initial = np.asarray([[.1, .1], [.9, .1], [.9, .9], [.1, .9]], np.float32)
    target = _adjust_prior(initial, top=-0.1, bottom=-0.1)

    def fake_detect(_image, prior, **_kwargs):
        error = float(np.mean(np.linalg.norm(np.asarray(prior) - target, axis=1)))
        confidence = max(0.0, 1.0 - 4.0 * error)
        side = {
            "quad": np.asarray(prior).tolist(),
            "confidence": confidence,
            "detected": confidence >= 0.5,
            "touches_frame": False,
        }
        return {
            "left": dict(side),
            "right": dict(side),
            "confidence": confidence,
            "detected": confidence >= 0.5,
        }

    monkeypatch.setattr("manga_scan.spread_detect.detect_page_quads", fake_detect)

    refined, diagnostics = _refine_priors(
        np.zeros((400, 800, 3), np.uint8),
        [initial],
        min_confidence=0.5,
    )

    initial_error = np.mean(np.linalg.norm(initial - target, axis=1))
    refined_error = np.mean(np.linalg.norm(refined[0] - target, axis=1))
    assert refined_error < initial_error
    assert any(np.allclose(prior, initial) for prior in refined)
    assert diagnostics["evaluated"] > 1
    assert diagnostics["best_score"] > 0.9
    assert diagnostics["baseline_preserved"]


def _occluded_mixed_color_spread():
    image = np.full((600, 1000, 3), (55, 95, 135), dtype=np.uint8)
    left = np.asarray([[105, 45], [495, 70], [480, 555], [90, 535]], dtype=np.int32)
    right = np.asarray([[515, 70], [900, 42], [930, 540], [520, 558]], dtype=np.int32)
    cv2.fillConvexPoly(image, left, (238, 238, 232))
    cv2.fillConvexPoly(image, right, (55, 75, 170))
    cv2.polylines(image, [left, right], True, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.rectangle(image, (585, 150), (820, 330), (210, 210, 205), 4)
    # Hands obscure parts of the lower outer boundary, like a real page turn.
    cv2.ellipse(image, (125, 500), (95, 70), -20, 0, 360, (150, 175, 215), -1)
    cv2.ellipse(image, (875, 490), (100, 85), 20, 0, 360, (145, 170, 210), -1)
    return image


def test_detect_reference_spread_uses_page_fallback_when_outer_outline_is_obscured(monkeypatch):
    monkeypatch.setattr(
        "manga_scan.spread_detect.detect_cover_quad_candidates",
        lambda *_args, **_kwargs: [],
    )

    result = detect_reference_spread(_occluded_mixed_color_spread(), min_confidence=0.5)

    assert result["detected"]
    assert result["stage"] == "complete"
    assert result["source"] == "coarse_pages"
    assert result["pages"]["left"]["detected"]
    assert result["pages"]["right"]["detected"]
    roi = np.asarray(result["roi"])
    assert roi[0, 0] < 0.2
    assert roi[1, 0] > 0.8


def _setup_manifest(rotation):
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        rotation=rotation,
        auto_rotation=False,
    )
    return {
        "source": "/tmp/book.mp4",
        "metadata": {"duration": 10},
        "config": cfg.to_dict(),
        "rotation_detection": {
            "rotation": rotation,
            "confidence": 1.0,
            "source": "manual",
            "confirmed": True,
        },
        "roi": None,
        "cover": {"status": "skipped", "roi": None},
        "reference": {"confirmed": False},
        "status": "ready",
        "warnings": [],
        "pages": [],
        "spreads": [],
        "pdf_stale": True,
        "progress": 0,
        "message": "",
    }


def test_reference_consensus_keeps_selected_frame_as_anchor_at_video_start(monkeypatch):
    selected = np.full((12, 20, 3), 17, np.uint8)
    neighbor = np.full((12, 20, 3), 31, np.uint8)
    extracted_times = []

    def fake_extract(_source, timestamp, **_kwargs):
        extracted_times.append(timestamp)
        return neighbor.copy()

    monkeypatch.setattr("manga_scan.ingest.extract_frame", fake_extract)

    frames, anchor_index = _reference_consensus_frames(
        "/tmp/book.mp4",
        {"duration": 10.0},
        0.0,
        selected,
        Config(rotation=0, auto_rotation=False),
    )

    assert anchor_index == 0
    np.testing.assert_array_equal(frames[anchor_index], selected)
    assert extracted_times == [0.25, 0.5]
    assert len(frames) == 3


def test_reference_search_frame_prefers_lower_hand_occlusion(monkeypatch):
    images = [
        np.full((120, 200, 3), 180, np.uint8),
        np.full((120, 200, 3), 180, np.uint8),
    ]
    masks = [
        np.zeros((120, 200), np.uint8),
        np.zeros((120, 200), np.uint8),
    ]
    masks[0][:, 120:] = 255
    monkeypatch.setattr(
        "manga_scan.spread_detect.detect_cover_quad_candidates",
        lambda *_args, **_kwargs: [{"confidence": 0.75}],
    )

    selected, diagnostics = _select_reference_search_frame(
        images,
        masks,
        min_confidence=0.5,
    )

    assert selected == 1
    assert diagnostics[1]["score"] > diagnostics[0]["score"]
    assert diagnostics[0]["hand_fraction"] > diagnostics[1]["hand_fraction"]


def test_reference_boundary_marks_persistently_hand_occluded_edge_uncertain():
    image = _synthetic_spread()
    hand = np.zeros(image.shape[:2], np.uint8)
    cv2.line(hand, (930, 55), (950, 520), 255, 110)

    result = detect_reference_spread_consensus(
        [image, image.copy(), image.copy()],
        min_confidence=0.5,
        anchor_index=1,
        hand_masks=[hand, hand.copy(), hand.copy()],
    )

    assert result["detected"]
    assert result["requires_confirmation"]
    assert "right" in result["uncertain_edges"]
    assert result["confidence"] <= 0.69
    assert result["geometry_confidence"] >= result["confidence"]
    assert (
        result["boundary_evidence"]["edges"]["right"]["hand_occlusion"]
        >= 0.18
    )


def test_reference_boundary_uses_clean_neighboring_frames_when_available():
    image = _synthetic_spread()
    hand = np.zeros(image.shape[:2], np.uint8)
    cv2.line(hand, (930, 55), (950, 520), 255, 110)
    clear = np.zeros_like(hand)

    result = detect_reference_spread_consensus(
        [image, image.copy(), image.copy()],
        min_confidence=0.5,
        anchor_index=1,
        hand_masks=[clear, hand, clear],
    )

    assert result["detected"]
    assert "right" not in result["uncertain_edges"]
    assert (
        result["boundary_evidence"]["edges"]["right"]["hand_occlusion"]
        < 0.18
    )


def test_reference_confirmation_stores_auto_roi_in_source_coordinates(tmp_path, monkeypatch):
    frame = np.zeros((160, 300, 3), np.uint8)
    displayed_roi = [[0.10, 0.15], [0.90, 0.15], [0.90, 0.85], [0.10, 0.85]]
    save_manifest(tmp_path, _setup_manifest(90))
    monkeypatch.setattr(
        "manga_scan.ingest.extract_frame",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        "manga_scan.ingest.detect_reference_spread_consensus",
        lambda *_args, **_kwargs: {
            "detected": True,
            "confidence": 0.88,
            "roi": displayed_roi,
            "outline": {"detected": True, "confidence": 0.9, "roi": displayed_roi},
            "pages": {
                "detected": True,
                "confidence": 0.88,
                "left": {"quad": displayed_roi, "detected": True, "confidence": 0.9},
                "right": {"quad": displayed_roi, "detected": True, "confidence": 0.88},
            },
            "stage": "complete",
        },
    )

    manifest = set_setup_frame(tmp_path, "reference", 2.0, confirm=True)

    expected = rotate_roi(displayed_roi, 270)
    np.testing.assert_allclose(manifest["roi"], expected, atol=1e-6)
    assert manifest["reference"]["detection"]["detected"]
    assert manifest["reference"]["detection"]["confidence"] == 0.88
    assert manifest["reference"]["detection_preview"] == "source/reference_detection.png"
    assert (tmp_path / "source/reference_detection.png").is_file()
    assert "自動検出しました" in manifest["message"]


def test_reference_confirmation_falls_back_to_manual_points(tmp_path, monkeypatch):
    frame = np.zeros((160, 300, 3), np.uint8)
    save_manifest(tmp_path, _setup_manifest(0))
    monkeypatch.setattr(
        "manga_scan.ingest.extract_frame",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        "manga_scan.ingest.detect_reference_spread_consensus",
        lambda *_args, **_kwargs: {
            "detected": False,
            "confidence": 0.31,
            "roi": None,
            "outline": {"detected": False, "confidence": 0.31, "roi": None},
            "pages": None,
            "stage": "outline",
        },
    )

    manifest = set_setup_frame(tmp_path, "reference", 2.0, confirm=True)

    assert manifest["roi"] is None
    assert not manifest["reference"]["detection"]["detected"]
    assert "4点で指定" in manifest["message"]


def test_reference_spread_consensus_requires_multiple_consistent_frames():
    image = _synthetic_spread()

    result = detect_reference_spread_consensus(
        [image, image.copy(), image.copy()],
        min_confidence=0.5,
    )

    assert result["detected"]
    assert result["frame_support"] == 3
    assert result["pages"]["left"]["consensus_count"] == 3
    assert result["pages"]["right"]["consensus_count"] == 3
    assert result["source"].endswith("_consensus")


def test_reference_spread_consensus_aligns_handheld_camera_motion():
    anchor = _synthetic_spread()
    previous = cv2.warpPerspective(
        anchor,
        np.asarray([[1, 0, -45], [0, 1, -20], [0, 0, 1]], np.float64),
        (anchor.shape[1], anchor.shape[0]),
        borderValue=(45, 85, 125),
    )
    following = cv2.warpPerspective(
        anchor,
        np.asarray([[1, 0, 42], [0, 1, 18], [0, 0, 1]], np.float64),
        (anchor.shape[1], anchor.shape[0]),
        borderValue=(45, 85, 125),
    )

    result = detect_reference_spread_consensus(
        [previous, anchor, following],
        min_confidence=0.5,
        anchor_index=1,
    )

    assert result["detected"]
    assert result["frame_support"] == 3
    assert result["alignment"]["aligned"] == 2
    assert all(
        frame["status"] in ("aligned", "anchor")
        for frame in result["alignment"]["frames"]
    )


def test_reference_spread_marks_close_distinct_candidates_ambiguous(monkeypatch):
    roi_a = [[0.05, 0.1], [0.9, 0.1], [0.9, 0.9], [0.05, 0.9]]
    roi_b = [[0.1, 0.1], [0.95, 0.1], [0.95, 0.9], [0.1, 0.9]]
    outlines = [
        {"detected": True, "confidence": 0.8, "roi": roi_a},
        {"detected": True, "confidence": 0.79, "roi": roi_b},
    ]
    monkeypatch.setattr(
        "manga_scan.spread_detect.detect_cover_quad_candidates",
        lambda *_args, **_kwargs: outlines,
    )

    def proposal(
        _frames,
        _priors,
        _confidence,
        *,
        alignments,
        anchor_index,
        proposal_id,
        outline=None,
        source,
        local_search=None,
        hand_masks=None,
    ):
        assert alignments[anchor_index]["status"] == "anchor"
        assert local_search is not None
        if proposal_id == "coarse":
            return None, 0.0
        roi = roi_a if proposal_id == "hough_1" else roi_b
        score = 0.80 if proposal_id == "hough_1" else 0.78
        pages = {
            "detected": True,
            "confidence": 0.8,
            "left": {"quad": roi, "detected": True, "confidence": 0.8},
            "right": {"quad": roi, "detected": True, "confidence": 0.8},
        }
        return {
            "proposal_id": proposal_id,
            "score": score,
            "confidence": 0.8,
            "roi": roi,
            "outline": outline,
            "pages": pages,
            "source": source,
            "frame_support": 3,
            "frame_count": 3,
            "local_search": local_search,
        }, 0.8

    monkeypatch.setattr("manga_scan.spread_detect._proposal_from_prior_set", proposal)
    image = np.zeros((80, 120, 3), np.uint8)

    result = detect_reference_spread_consensus([image, image, image], min_confidence=0.5)

    assert result["detected"]
    assert result["ambiguous"]
    assert result["requires_confirmation"]
    assert result["score_margin"] == 0.02
