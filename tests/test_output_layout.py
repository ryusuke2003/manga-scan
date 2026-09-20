import cv2
import numpy as np
import pytest

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.storage import read_manifest, save_image, save_manifest

ROI = [[0, 0], [1, 0], [1, 1], [0, 1]]


def fixture(tmp_path, monkeypatch, rotation=0):
    image = np.full((160, 300, 3), 220, np.uint8)
    image[:, :90] = (30, 80, 130)
    image[:, 180:] = (140, 60, 20)
    cv2.line(image, (5, 15), (295, 145), (15, 15, 15), 3)
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        rotation=rotation,
        grayscale=False,
        illumination_correction=False,
        white_normalization=False,
        page_background_fill="preserve",
    )
    spread = {
        "id": "spread_0001",
        "selected": 0,
        "extra_suspect": [],
        "selected_pages": {"left": 1, "right": 1},
        "candidates": [{"id": i, "time": float(i), "roi": ROI, "suspect": []} for i in range(2)],
    }
    manifest = {
        "config": cfg.to_dict(),
        "source": "unused",
        "spreads": [spread],
        "pages": [],
        "metadata": {"duration": 5},
    }
    monkeypatch.setattr(pipeline, "extract_frame", lambda *_a, **_k: image.copy())
    return image, manifest, spread


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_default_output_preserves_entire_spread_and_ignores_split_settings(
    tmp_path,
    monkeypatch,
    rotation,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch, rotation)
    assert Config().output_layout == "spread"
    manifest["config"]["gutter_fraction"] = 0.05
    manifest["config"]["candidate_selection_mode"] = "per_page"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Whole spread must not run the split-output rectifier")

    monkeypatch.setattr(pipeline, "rectify_spread_pages", forbidden)
    pages = pipeline.render_spread(tmp_path, manifest, spread)
    assert len(pages) == 1
    assert pages[0]["side"] == "spread"
    assert pages[0]["candidate_id"] == 0
    output = cv2.imread(str(tmp_path / pages[0]["path"]))
    np.testing.assert_array_equal(output, np.rot90(image, -(rotation // 90)))
    assert pages[0]["dewarp"]["mode"] == "off"


def test_whole_spread_auto_crop_uses_outer_corners_from_both_pages(tmp_path, monkeypatch):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    detected = {
        "detected": True,
        "confidence": .86,
        "left": {
            "quad": [[.08, .10], [.49, .12], [.49, .88], [.07, .90]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[.51, .12], [.92, .09], [.94, .91], [.51, .88]],
            "confidence": .86,
            "detected": True,
            "touches_frame": False,
        },
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detected)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]

    expected_roi = [[.08, .10], [.92, .09], [.94, .91], [.07, .90]]
    output = cv2.imread(str(tmp_path / page["path"]))
    expected = pipeline.warp_roi(image, expected_roi)
    np.testing.assert_allclose(output, expected, atol=1)
    assert page["crop"]["status"] == "auto_pages"
    assert page["crop"]["confidence"] == pytest.approx(.86)
    assert spread["perspective_mode_used"] == "spread_auto_pages"
    assert spread["page_contours"] == detected
    assert (tmp_path / spread["page_contour_debug"]).is_file()
    assert "page_contour_low_confidence" not in page["suspect"]


def test_whole_spread_background_fill_hides_desk_only_with_trusted_pages(
    tmp_path,
    monkeypatch,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"]["page_background_fill"] = "white"
    detected = {
        "detected": True,
        "confidence": .9,
        "left": {
            "quad": [[.08, .12], [.49, .14], [.49, .86], [.08, .88]],
            "confidence": .92,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[.51, .14], [.92, .10], [.94, .90], [.51, .86]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detected)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]
    output = cv2.imread(str(tmp_path / page["path"]))

    assert page["background_fill"]["status"] == "applied"
    assert page["background_fill"]["mode"] == "white"
    assert page["background_fill"]["confidence"] == pytest.approx(.9)
    assert page["background_fill"]["mask"]
    assert (tmp_path / page["background_fill"]["mask"]).is_file()
    page_mask = cv2.imread(
        str(tmp_path / page["background_fill"]["mask"]),
        cv2.IMREAD_GRAYSCALE,
    )
    background = (page_mask < 127).astype(np.uint8)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    assert distance[y, x] >= 4
    # A desk pixel well outside the protected page margin is concealed.
    assert np.all(output[y, x] >= 245)

    # The photographed gutter remains protected rather than being whitened.
    center = output.shape[1] // 2
    expected = pipeline.warp_roi(image, page["crop"]["roi"])
    np.testing.assert_allclose(
        output[output.shape[0] // 2, center],
        expected[expected.shape[0] // 2, center],
        atol=1,
    )


def test_whole_spread_background_fill_skips_uncertain_fallback(
    tmp_path,
    monkeypatch,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"]["page_background_fill"] = "white"
    detection = {
        "detected": False,
        "confidence": .2,
        "left": {"quad": ROI, "confidence": .8, "detected": True, "touches_frame": False},
        "right": {"quad": ROI, "confidence": .2, "detected": False, "touches_frame": False},
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detection)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]
    output = cv2.imread(str(tmp_path / page["path"]))

    np.testing.assert_array_equal(output, image)
    assert page["background_fill"]["status"] == "unavailable"
    assert not page["background_fill"]["applied"]


def test_whole_spread_auto_crop_falls_back_when_either_page_is_uncertain(
    tmp_path, monkeypatch,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    detection = {
        "detected": False,
        "confidence": .2,
        "left": {"quad": ROI, "confidence": .8, "detected": True, "touches_frame": False},
        "right": {"quad": ROI, "confidence": .2, "detected": False, "touches_frame": False},
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detection)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]

    output = cv2.imread(str(tmp_path / page["path"]))
    np.testing.assert_array_equal(output, image)
    assert page["crop"]["status"] == "fallback"
    assert "page_contour_low_confidence" in page["suspect"]


def test_whole_spread_invalid_combined_quad_falls_back_instead_of_failing(
    tmp_path, monkeypatch,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    detection = {
        "detected": True,
        "confidence": .9,
        "left": {
            "quad": [[.65, .10], [.45, .10], [.45, .90], [.10, .90]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[.55, .10], [.35, .10], [.90, .90], [.55, .90]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detection)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]

    output = cv2.imread(str(tmp_path / page["path"]))
    np.testing.assert_array_equal(output, image)
    assert page["crop"]["status"] == "fallback"
    assert "page_contour_low_confidence" in page["suspect"]


def test_layout_roundtrip_preserves_exclusions_order_and_cover(tmp_path, monkeypatch):
    _, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"]["output_layout"] = "split"
    manifest["config"]["perspective_mode"] = "spread"
    manifest["config"]["dewarp_mode"] = "off"
    manifest["pages"] = pipeline.render_spread(tmp_path, manifest, spread)
    manifest["pages"].reverse()  # User's left-before-right order.
    manifest["pages"][0]["enabled"] = False
    original = [(p["id"], p["enabled"]) for p in manifest["pages"]]
    cover = {"id": "cover", "spread_id": "cover", "side": "cover", "enabled": True}
    manifest["pages"].insert(0, cover)
    save_manifest(tmp_path, manifest)
    result = pipeline.edit(tmp_path, "output_layout", spread_id=spread["id"], layout="spread")
    assert [p["side"] for p in result["pages"]] == ["cover", "spread"]
    assert result["pages"][1]["enabled"]
    assert result["pdf_stale"]
    result = pipeline.edit(tmp_path, "select_candidate", spread_id=spread["id"], candidate_id=1)
    assert result["pages"][1]["candidate_id"] == 1
    result = pipeline.edit(tmp_path, "output_layout", spread_id=spread["id"], layout="split")
    assert result["pages"][0] == cover
    assert [(p["id"], p["enabled"]) for p in result["pages"][1:]] == original


def test_legacy_project_keeps_split_but_new_config_defaults_to_spread(tmp_path):
    save_manifest(tmp_path, {"config": {"hand_backend": "none"}, "pages": []})
    assert read_manifest(tmp_path)["config"]["output_layout"] == "split"
    assert Config.from_dict({"hand_backend": "none"}).output_layout == "spread"
    with pytest.raises(ValueError, match="output_layout"):
        Config.from_dict({"output_layout": "invalid"})


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_crop_applies_to_only_selected_candidate_in_display_orientation(
    tmp_path,
    monkeypatch,
    rotation,
):
    image, manifest, spread = fixture(tmp_path, monkeypatch, rotation)
    manifest["pages"] = pipeline.render_spread(tmp_path, manifest, spread)
    manifest["pages"][0]["enabled"] = False
    save_manifest(tmp_path, manifest)
    crop = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    result = pipeline.edit(tmp_path, "crop", spread_id=spread["id"], candidate_id=0, roi=crop)
    assert not result["pages"][0]["enabled"]
    assert result["spreads"][0]["candidates"][0]["roi"] == ROI
    output = cv2.imread(str(tmp_path / result["pages"][0]["path"]))
    upright = np.rot90(image, -(rotation // 90)).copy()
    expected = pipeline.warp_roi(upright, crop)
    np.testing.assert_allclose(output, expected, atol=1)
    assert "1" not in result["spreads"][0]["roi_overrides"]
    assert result["pages"][0]["crop"]["status"] == "manual"
    result = pipeline.edit(
        tmp_path, "reset_crop", spread_id=spread["id"], candidate_id=0
    )
    assert "roi_overrides" not in result["spreads"][0]
    assert result["pages"][0]["crop"]["status"] != "manual"
    with pytest.raises(ValueError, match="ROI"):
        pipeline.edit(tmp_path, "crop", spread_id=spread["id"], candidate_id=0, roi=[[0, 0]])


def test_whole_spread_flags_hand_added_by_expanded_auto_crop(tmp_path, monkeypatch):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    candidate_roi = [[.08, .08], [.88, .08], [.88, .92], [.08, .92]]
    spread["candidates"][0]["roi"] = candidate_roi
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[55:115, 270:296] = 255
    save_image(tmp_path / "expanded_hand_mask.png", mask)
    spread["candidates"][0]["hand_mask"] = "expanded_hand_mask.png"
    manifest["config"].update(hand_backend="mediapipe", finger_repair=False)

    detected = {
        "detected": True,
        "confidence": .9,
        "left": {
            "quad": [[.04, .06], [.49, .08], [.49, .92], [.04, .94]],
            "confidence": .9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[.51, .08], [.99, .06], [.99, .94], [.51, .92]],
            "confidence": .9,
            "detected": True,
            "touches_frame": True,
        },
    }
    monkeypatch.setattr(pipeline, "detect_page_quads", lambda *_a, **_k: detected)

    result = pipeline.render_spread(tmp_path, manifest, spread)[0]

    assert result["crop"]["status"] == "auto_pages"
    assert result["finger_repair"]["status"] == "disabled"
    assert "hand_overlap" in result["suspect"]


def test_whole_spread_keeps_finger_repair_active(tmp_path, monkeypatch):
    image, manifest, spread = fixture(tmp_path, monkeypatch)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[90:125, 220:250] = 255
    target = image.copy()
    target[mask > 0] = (90, 120, 175)
    save_image(tmp_path / "target_mask.png", mask)
    save_image(tmp_path / "donor_mask.png", np.zeros_like(mask))
    manifest["config"].update(hand_backend="mediapipe", finger_repair=True)
    spread["candidates"][0]["hand_mask"] = "target_mask.png"
    spread["candidates"][1]["hand_mask"] = "donor_mask.png"
    monkeypatch.setattr(
        pipeline,
        "extract_frame",
        lambda _source, time, **_kw: target.copy() if time == 0 else image.copy(),
    )
    result = pipeline.render_spread(tmp_path, manifest, spread)[0]
    assert result["finger_repair"]["status"] == "complete"
    assert result["finger_repair"]["donors"] == [1]
    output = cv2.imread(str(tmp_path / result["path"]))
    error = np.abs(output[100:115, 230:240].astype(float) - image[100:115, 230:240])
    assert error.mean() < 2
    np.testing.assert_array_equal(output[mask == 0], target[mask == 0])


def _review_detected_pages():
    return {
        "detected": True,
        "confidence": 0.9,
        "left": {
            "quad": [[0.06, 0.08], [0.48, 0.10], [0.48, 0.90], [0.06, 0.92]],
            "confidence": 0.9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": [[0.52, 0.10], [0.94, 0.08], [0.94, 0.92], [0.52, 0.90]],
            "confidence": 0.9,
            "detected": True,
            "touches_frame": False,
        },
    }


def test_page_settings_override_only_the_requested_split_page(tmp_path, monkeypatch):
    _, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"].update(
        output_layout="split",
        perspective_mode="per_page",
        dewarp_mode="auto",
        illumination_correction=True,
        white_normalization=True,
    )
    monkeypatch.setattr(
        pipeline,
        "detect_page_quads",
        lambda *_a, **_k: _review_detected_pages(),
    )
    manifest["pages"] = pipeline.render_spread(tmp_path, manifest, spread)
    save_manifest(tmp_path, manifest)

    result = pipeline.edit(
        tmp_path,
        "page_settings",
        page_id="spread_0001_right",
        settings={
            "dewarp": False,
            "illumination_correction": False,
            "white_normalization": False,
        },
    )

    right = next(page for page in result["pages"] if page["side"] == "right")
    left = next(page for page in result["pages"] if page["side"] == "left")
    assert right["render_settings"]["dewarp"] is False
    assert right["render_settings"]["illumination_correction"] is False
    assert right["render_settings"]["white_normalization"] is False
    assert right["dewarp"]["status"] == "disabled"

    assert left["render_settings"]["dewarp"] is True
    assert left["render_settings"]["illumination_correction"] is True
    assert left["render_settings"]["white_normalization"] is True
    assert result["spreads"][0]["page_overrides"]["right"] == {
        "dewarp": False,
        "illumination_correction": False,
        "white_normalization": False,
    }
    assert result["pdf_stale"] is True


def test_page_settings_can_switch_one_page_contour_between_manual_and_auto(
    tmp_path,
    monkeypatch,
):
    _, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"].update(
        output_layout="split",
        perspective_mode="per_page",
        dewarp_mode="off",
    )
    monkeypatch.setattr(
        pipeline,
        "detect_page_quads",
        lambda *_a, **_k: _review_detected_pages(),
    )
    manifest["pages"] = pipeline.render_spread(tmp_path, manifest, spread)
    save_manifest(tmp_path, manifest)
    manual_quad = [[0.56, 0.14], [0.91, 0.12], [0.92, 0.87], [0.55, 0.89]]

    result = pipeline.edit(
        tmp_path,
        "page_settings",
        page_id="spread_0001_right",
        settings={"page_quad_mode": "manual", "manual_quad": manual_quad},
    )

    right = next(page for page in result["pages"] if page["side"] == "right")
    left = next(page for page in result["pages"] if page["side"] == "left")
    assert right["page_contour"]["mode"] == "manual"
    assert right["page_contour"]["manual"] is True
    np.testing.assert_allclose(right["page_contour"]["quad"], manual_quad, atol=1e-6)
    assert left["page_contour"]["mode"] == "auto"
    assert left["page_contour"]["manual"] is False

    result = pipeline.edit(
        tmp_path,
        "page_settings",
        page_id="spread_0001_right",
        settings={"page_quad_mode": "auto"},
    )
    right = next(page for page in result["pages"] if page["side"] == "right")
    assert right["page_contour"]["mode"] == "auto"
    assert right["page_contour"]["manual"] is False
    assert "manual_quad" not in result["spreads"][0]["page_overrides"]["right"]


def test_page_settings_rejects_invalid_manual_quad(tmp_path, monkeypatch):
    _, manifest, spread = fixture(tmp_path, monkeypatch)
    manifest["config"]["output_layout"] = "split"
    manifest["pages"] = pipeline.render_spread(tmp_path, manifest, spread)
    save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="ROI"):
        pipeline.edit(
            tmp_path,
            "page_settings",
            page_id="spread_0001_right",
            settings={
                "page_quad_mode": "manual",
                "manual_quad": [[0, 0], [1, 0]],
            },
        )
