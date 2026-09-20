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
        pytest.fail("Whole spread must not split or detect separate pages")

    monkeypatch.setattr(pipeline, "rectify_spread_pages", forbidden)
    pages = pipeline.render_spread(tmp_path, manifest, spread)
    assert len(pages) == 1
    assert pages[0]["side"] == "spread"
    assert pages[0]["candidate_id"] == 0
    output = cv2.imread(str(tmp_path / pages[0]["path"]))
    np.testing.assert_array_equal(output, np.rot90(image, -(rotation // 90)))
    assert pages[0]["dewarp"]["mode"] == "off"


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
    with pytest.raises(ValueError, match="ROI"):
        pipeline.edit(tmp_path, "crop", spread_id=spread["id"], candidate_id=0, roi=[[0, 0]])


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
