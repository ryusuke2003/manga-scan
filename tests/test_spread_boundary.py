import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.motion import Sample
from manga_scan.pipeline import candidate, render_spread
from manga_scan.pipeline_render_helpers import _whole_spread_geometry
from manga_scan.spread_boundary import refine_spread_boundary

PAGE = np.array([[.15, .12], [.85, .10], [.87, .89], [.13, .91]], np.float32)
DESK_CROP = [[.08, .06], [.92, .04], [.94, .95], [.06, .97]]
CLIPPED_CROP = [[.21, .18], [.79, .16], [.81, .83], [.19, .85]]


def _image():
    image = np.full((600, 1000, 3), (48, 83, 112), np.uint8)
    cv2.fillConvexPoly(
        image,
        np.rint(PAGE * [999, 599]).astype(np.int32),
        (220, 223, 225),
    )
    return image


def _iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return float(intersection / (cv2.contourArea(a) + cv2.contourArea(b) - intersection))


@pytest.mark.parametrize("prior", [DESK_CROP, CLIPPED_CROP])
def test_boundary_moves_both_directions_to_page(prior):
    result, info = refine_spread_boundary(_image(), prior)

    assert info["refined"]
    assert _iou(result, PAGE) > .97
    assert _iou(result, PAGE) > _iou(prior, PAGE) + .15


def test_uniform_image_keeps_prior():
    result, info = refine_spread_boundary(np.full((300, 500, 3), 128, np.uint8), DESK_CROP)

    assert not info["refined"]
    np.testing.assert_allclose(result, DESK_CROP)


def test_candidate_saves_refined_crop_and_preview(tmp_path, monkeypatch):
    image = _image()
    monkeypatch.setattr("manga_scan.pipeline.extract_frame", lambda *_args: image.copy())
    monkeypatch.setattr("manga_scan.pipeline.refine_quad", lambda _image, roi, _shift: (roi, True))

    class Detector:
        def detect(self, frame, roi):
            return None, np.zeros(frame.shape[:2], np.uint8)

    record = candidate(
        tmp_path,
        {"source": "unused", "roi": CLIPPED_CROP},
        Config(hand_backend="none"),
        Detector(),
        "spread_0001",
        0,
        Sample(0, 1.0, 0.0, 0.0),
    )

    assert record["boundary_refinement"]["refined"]
    assert _iou(record["roi"], PAGE) > .97
    assert (tmp_path / record["preview"]).is_file()


def test_whole_spread_fallback_uses_boundary_but_manual_override_wins():
    image = _image()
    cfg = Config(refine_quad=True, analysis_width=500)
    record = {"id": 2, "roi": CLIPPED_CROP}
    uncertain = {"detected": False, "confidence": .2}
    spread = {"roi_overrides": {}}

    _, roi, crop = _whole_spread_geometry(
        image, record, spread, cfg, page_detection=uncertain
    )
    assert crop["status"] == "auto_boundary"
    assert _iou(roi, PAGE) > .97

    spread["roi_overrides"]["2"] = CLIPPED_CROP
    _, roi, crop = _whole_spread_geometry(
        image, record, spread, cfg, page_detection=uncertain
    )
    assert crop["status"] == "manual"
    np.testing.assert_allclose(roi, CLIPPED_CROP)


def _layered_image(cover_color=(236, 234, 229)):
    image = np.full((700, 400, 3), (40, 70, 100), np.uint8)
    cover = np.asarray([[0, 30], [310, 20], [245, 650], [0, 665]], np.float32)
    page = np.asarray([[0, 30], [255, 20], [190, 650], [0, 665]], np.float32)
    cv2.fillConvexPoly(image, cover.astype(np.int32), cover_color)
    cv2.fillConvexPoly(image, page.astype(np.int32), (210, 220, 218))
    cv2.rectangle(image, (40, 100), (175, 135), (60, 70, 70), 3)
    cv2.rectangle(image, (65, 350), (140, 425), (50, 55, 55), 3)
    cv2.ellipse(image, (290, 430), (45, 115), -15, 0, 360, (110, 155, 193), -1)
    return image, cover / [399, 699], page / [399, 699]


@pytest.mark.parametrize(
    "cover_color",
    [
        (236, 234, 229),  # white
        (45, 70, 170),  # red
        (150, 65, 45),  # navy
        (20, 20, 20),  # black
        (70, 195, 235),  # yellow
    ],
)
def test_colored_band_without_depth_evidence_keeps_outer_edge(cover_color):
    image, cover, page = _layered_image(cover_color)

    result, info = refine_spread_boundary(image, cover.tolist())

    assert info["refined"]
    assert info["possible_inner_sheets"][0]["side"] == "right"
    assert "layered_sheets" not in info
    assert _iou(result, cover) > .99


def _printed_margin_image(margin_color):
    image = np.full((700, 400, 3), (40, 70, 100), np.uint8)
    page = np.asarray([[0, 30], [310, 20], [245, 650], [0, 665]], np.float32)
    margin = np.asarray([[255, 20], [310, 20], [245, 650], [190, 650]], np.int32)
    cv2.fillConvexPoly(image, page.astype(np.int32), (210, 220, 218))
    cv2.fillConvexPoly(image, margin, margin_color)
    cv2.rectangle(image, (40, 100), (175, 135), (60, 70, 70), 3)
    inner = np.asarray([[0, 30], [255, 20], [190, 650], [0, 665]], np.float32)
    return image, page / [399, 699], inner / [399, 699]


@pytest.mark.parametrize("margin_color", [(236, 234, 229), (45, 70, 170), (20, 20, 20)])
def test_no_cover_printed_margin_is_not_clipped(margin_color):
    image, page, _ = _printed_margin_image(margin_color)

    result, info = refine_spread_boundary(image, page.tolist())

    assert info["refined"]
    assert "layered_sheets" not in info
    assert _iou(result, page) > .99


def test_uncertain_inner_contour_cannot_clip_printed_margin():
    image, page, inner = _printed_margin_image((45, 70, 170))
    detection = {
        "detected": True,
        "confidence": .9,
        "left": {"quad": [[0, .04], [.3, .04], [.3, .95], [0, .95]]},
        "right": {"quad": [[.3, .04], inner[1].tolist(), inner[2].tolist(), [.3, .95]]},
    }

    _, roi, crop = _whole_spread_geometry(
        image,
        {"id": 0, "roi": page.tolist()},
        {},
        Config(refine_quad=True),
        page_detection=detection,
    )

    assert crop["status"] == "auto_boundary"
    assert crop["boundary_refinement"]["possible_inner_sheets"]
    assert _iou(roi, page) > .99


def test_uncertain_inner_edge_is_flagged_in_rendered_page(tmp_path, monkeypatch):
    image, page, inner = _printed_margin_image((45, 70, 170))
    detection = {
        "detected": True,
        "confidence": .9,
        "left": {
            "quad": [[0, .04], [.3, .04], [.3, .95], [0, .95]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[.3, .04], inner[1].tolist(), inner[2].tolist(), [.3, .95]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
    }
    monkeypatch.setattr("manga_scan.pipeline.extract_frame", lambda *_args, **_kwargs: image.copy())
    monkeypatch.setattr("manga_scan.pipeline.detect_spread_page_consensus", lambda *_args: detection)
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        output_layout="spread",
        grayscale=False,
        illumination_correction=False,
        white_normalization=False,
        dewarp_mode="off",
    )
    spread = {
        "id": "layered_ambiguous",
        "selected": 0,
        "candidates": [{"id": 0, "time": 0.0, "roi": page.tolist(), "suspect": []}],
        "extra_suspect": [],
    }
    manifest = {"config": cfg.to_dict(), "source": "unused", "pdf_stale": False}

    result = render_spread(tmp_path, manifest, spread)[0]

    assert result["crop"]["status"] == "auto_boundary"
    assert "page_quad_uncertain" in result["suspect"]


def test_outer_page_contour_cannot_replace_inner_sheet_edge():
    image, cover, page = _layered_image()
    detection = {
        "detected": True,
        "confidence": .9,
        "left": {"quad": [[0, .05], [.3, .05], [.3, .95], [0, .95]]},
        "right": {"quad": [[.3, .05], cover[1].tolist(), cover[2].tolist(), [.3, .95]]},
    }

    _, roi, crop = _whole_spread_geometry(
        image,
        {
            "id": 0,
            "roi": page.tolist(),
            "boundary_refinement": {
                "refined": True,
                "layered_sheets": [{"side": "right"}],
            },
        },
        {},
        Config(refine_quad=True),
        page_detection=detection,
    )

    assert crop["status"] == "auto_boundary"
    assert _iou(roi, page) > .99


def test_panel_rule_does_not_masquerade_as_second_sheet():
    image = np.full((700, 400, 3), (40, 70, 100), np.uint8)
    page = np.asarray([[0, 30], [255, 20], [190, 650], [0, 665]], np.float32)
    cv2.fillConvexPoly(image, page.astype(np.int32), (210, 220, 218))
    cv2.rectangle(image, (40, 100), (200, 560), (35, 40, 40), 4)

    result, info = refine_spread_boundary(image, (page / [399, 699]).tolist())

    assert info["refined"]
    assert "layered_sheets" not in info
    assert _iou(result, page / [399, 699]) > .99
