import cv2
import numpy as np
import pytest

from manga_scan.config import Config
from manga_scan.motion import Sample
from manga_scan.pipeline import candidate
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
