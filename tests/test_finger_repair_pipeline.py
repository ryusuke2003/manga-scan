import json

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.storage import save_image

ROI = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _image():
    image = np.full((96, 192, 3), 224, np.uint8)
    cv2.line(image, (12, 18), (176, 78), (35, 35, 35), 3)
    cv2.rectangle(image, (24, 28), (74, 66), (80, 120, 170), 2)
    cv2.rectangle(image, (118, 24), (168, 70), (145, 80, 40), 2)
    return image


def _repair_metadata(status="complete", coverage=1.0, donor_coverage=1.0):
    return {
        "status": status,
        "coverage": coverage,
        "donor_coverage": donor_coverage,
        "donors": [1, 2],
        "alignment_scores": [0.9312, 0.8876],
        "components": [
            {
                "component_id": 1,
                "area": 480,
                "coverage": 1.0,
                "donors": [
                    {
                        "candidate_id": 1,
                        "method": "local",
                        "local_score": 0.96,
                        "dx": 2.0,
                        "dy": -1.0,
                        "context_residual": 0.031,
                        "coverage": 0.58,
                    },
                    {
                        "candidate_id": 2,
                        "method": "local",
                        "local_score": 0.91,
                        "dx": -1.0,
                        "dy": 1.0,
                        "context_residual": 0.044,
                        "coverage": 0.42,
                    },
                ],
            },
        ],
        "fallback": {
            "mode": "preserve",
            "applied": False,
            "filled_pixels": 0,
            "filled_fraction": 0.0,
            "components_filled": 0,
            "components_total": 0,
        },
    }


def _fixture(tmp_path, monkeypatch, *, output_layout="spread", finger_repair=True):
    (tmp_path / "debug").mkdir(exist_ok=True)
    image = _image()
    cfg = Config(
        output_layout=output_layout,
        hand_backend="mediapipe",
        finger_repair=finger_repair,
        finger_repair_fallback="preserve",
        refine_quad=False,
        perspective_mode="spread",
        dewarp_mode="off",
        illumination_correction=False,
        white_normalization=False,
        page_background_fill="preserve",
        grayscale=False,
    )
    target_mask = np.zeros(image.shape[:2], np.uint8)
    target_mask[38:58, 82:106] = 255
    clean_mask = np.zeros_like(target_mask)
    for candidate_id, mask in ((0, target_mask), (1, clean_mask), (2, clean_mask)):
        save_image(tmp_path / f"candidate_{candidate_id}_mask.png", mask)

    candidates = []
    for candidate_id, score in ((0, 1.0), (1, 0.9), (2, 0.8)):
        candidates.append(
            {
                "id": candidate_id,
                "time": float(candidate_id),
                "roi": ROI,
                "hand_mask": f"candidate_{candidate_id}_mask.png",
                "metrics": {"score": score, "hand_overlap": 0.0},
                "page_metrics": {
                    "left": {"score": score, "hand_overlap": 0.0},
                    "right": {"score": score, "hand_overlap": 0.0},
                },
                "page_suspect": {
                    "left": ["hand_overlap"],
                    "right": ["hand_overlap"],
                },
                "suspect": ["hand_overlap"],
            }
        )

    spread = {
        "id": "spread_0001",
        "selected": 0,
        "selected_pages": {"left": 0, "right": 0},
        "extra_suspect": [],
        "candidates": candidates,
    }
    manifest = {
        "config": cfg.to_dict(),
        "source": "unused",
        "spreads": [spread],
        "pages": [],
        "metadata": {"duration": 5},
        "pdf_stale": True,
    }
    monkeypatch.setattr(pipeline, "extract_frame", lambda *_a, **_k: image.copy())
    monkeypatch.setattr(
        pipeline,
        "boundary_finger_mask",
        lambda page, *_a, **_k: np.zeros(page.shape[:2], np.uint8),
    )
    return image, target_mask, manifest, spread


def _install_split_geometry(monkeypatch):
    def fake_rectify(_project, _image, rectified, _roi, state, _cfg):
        middle = rectified.shape[1] // 2
        state["perspective_mode_used"] = "spread"
        state["spine_px"] = middle
        return {
            "left": rectified[:, :middle].copy(),
            "right": rectified[:, middle:].copy(),
        }

    monkeypatch.setattr(pipeline, "rectify_spread_pages", fake_rectify)

    def fake_page_mask(_project, data, _side, _cfg):
        shape = data["sides"][_side].shape[:2]
        mask = np.zeros(shape, np.uint8)
        if data["chosen"]["id"] == 0:
            h, w = shape
            mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 255
        return mask

    monkeypatch.setattr(pipeline, "candidate_page_hand_mask", fake_page_mask)


def test_temporal_mask_augments_candidate_hand_score(tmp_path, monkeypatch):
    image = _image()
    cfg = Config(
        hand_backend="mediapipe",
        finger_repair=True,
        candidate_selection_mode="spread",
        refine_quad=False,
    )
    records = []
    for candidate_id in range(4):
        path = f"candidate_{candidate_id}.png"
        mask_path = f"candidate_{candidate_id}_hand_mask.png"
        save_image(tmp_path / path, image)
        save_image(tmp_path / mask_path, np.zeros(image.shape[:2], np.uint8))
        records.append(
            {
                "id": candidate_id,
                "path": path,
                "hand_mask": mask_path,
                "roi": ROI,
                "metrics": {
                    "motion": 0.0,
                    "sharpness": 1.0,
                    "hand_overlap": 0.0,
                    "distortion": 0.0,
                    "flatness_proxy": 0.0,
                    "clipping": 0.0,
                    "exposure": 0.0,
                    "score": 0.0,
                },
                "page_metrics": {},
                "page_suspect": {},
                "suspect": [],
            }
        )

    temporal = np.zeros(image.shape[:2], np.uint8)
    temporal[20:40, 0:18] = 255
    monkeypatch.setattr(
        pipeline,
        "temporal_transient_mask",
        lambda *_a, **_k: temporal.copy(),
    )

    pipeline._augment_temporal_hand_masks(tmp_path, records, cfg)

    for record in records:
        combined = cv2.imread(
            str(tmp_path / record["hand_mask"]),
            cv2.IMREAD_GRAYSCALE,
        )
        saved_temporal = cv2.imread(
            str(tmp_path / record["temporal_hand_mask"]),
            cv2.IMREAD_GRAYSCALE,
        )
        assert np.any(combined > 0)
        np.testing.assert_array_equal(saved_temporal, temporal)
        assert record["temporal_hand_overlap"] > 0
        assert record["metrics"]["hand_overlap"] > 0


def test_spread_output_preserves_local_repair_metadata_and_debug(tmp_path, monkeypatch):
    _, target_mask, manifest, spread = _fixture(
        tmp_path,
        monkeypatch,
        output_layout="spread",
    )
    manifest["config"]["processing_workers"] = 1
    clean_mask = np.zeros_like(target_mask)
    runtime_cache = {
        0: {"hand_mask": target_mask},
        1: {"hand_mask": clean_mask},
        2: {"hand_mask": clean_mask},
    }
    for candidate_id in (0, 1, 2):
        (tmp_path / f"candidate_{candidate_id}_mask.png").unlink()

    def fake_repair(target, _target_mask, donors, **_kwargs):
        donor_ids = [donor["candidate_id"] for donor in donors]
        assert donor_ids == [1, 2]
        assert _kwargs["alignment_workers"] == 1
        return target.copy(), _repair_metadata(), np.zeros(target.shape[:2], np.uint8)

    monkeypatch.setattr(pipeline, "repair_finger_regions", fake_repair)

    page = pipeline.render_spread(
        tmp_path,
        manifest,
        spread,
        runtime_cache=runtime_cache,
    )[0]
    repair = page["finger_repair"]

    assert page["side"] == "spread"
    assert repair["coverage"] == 1.0
    assert repair["donor_coverage"] == 1.0
    assert repair["donors"] == [1, 2]
    assert repair["alignment_scores"] == [0.9312, 0.8876]
    assert repair["components"][0]["component_id"] == 1
    assert [entry["candidate_id"] for entry in repair["components"][0]["donors"]] == [1, 2]
    assert repair["local_alignment"]["component_count"] == 1
    assert repair["local_alignment"]["max_shift_px"] == pytest.approx(
        round(float(np.hypot(2.0, -1.0)), 3)
    )
    assert [entry["candidate_id"] for entry in repair["local_alignment"]["components"]] == [1, 2]
    assert repair["fallback"]["mode"] == "preserve"
    assert repair["target_mask"].endswith("_whole_target.png")
    assert "unresolved_mask" not in repair
    assert repair["components_debug"].endswith("_whole_components.json")
    assert (tmp_path / repair["target_mask"]).is_file()
    assert (tmp_path / repair["components_debug"]).is_file()
    payload = json.loads((tmp_path / repair["components_debug"]).read_text())
    assert payload["components"] == repair["components"]
    assert payload["local_alignment"] == repair["local_alignment"]
    assert "hand_overlap" not in page["suspect"]
    assert "finger_repair_incomplete" not in page["suspect"]


def test_split_output_keeps_components_unresolved_and_multiple_donors(tmp_path, monkeypatch):
    _, _, manifest, spread = _fixture(tmp_path, monkeypatch, output_layout="split")
    _install_split_geometry(monkeypatch)
    image = _image()
    right_source = image[:, image.shape[1] // 2 :]

    def fake_repair(target, _target_mask, donors, **_kwargs):
        donor_ids = [donor["candidate_id"] for donor in donors]
        assert donor_ids == [1, 2]
        if np.array_equal(target, right_source):
            unresolved = np.zeros(target.shape[:2], np.uint8)
            unresolved[12:24, 20:32] = 255
            return (
                target.copy(),
                _repair_metadata(status="incomplete", coverage=0.7, donor_coverage=0.62),
                unresolved,
            )
        return target.copy(), _repair_metadata(), np.zeros(target.shape[:2], np.uint8)

    monkeypatch.setattr(pipeline, "repair_finger_regions", fake_repair)

    pages = pipeline.render_spread(tmp_path, manifest, spread)
    assert [page["side"] for page in pages] == ["right", "left"]

    incomplete = pages[0]
    repair = incomplete["finger_repair"]
    assert repair["status"] == "incomplete"
    assert repair["coverage"] == 0.7
    assert repair["donor_coverage"] == 0.62
    assert repair["donors"] == [1, 2]
    assert repair["alignment_scores"] == [0.9312, 0.8876]
    assert len(repair["components"]) == 1
    assert [entry["candidate_id"] for entry in repair["components"][0]["donors"]] == [1, 2]
    assert repair["local_alignment"]["component_count"] == 1
    assert [entry["candidate_id"] for entry in repair["local_alignment"]["components"]] == [1, 2]
    assert repair["target_mask"].endswith("_right_target.png")
    assert repair["unresolved_mask"].endswith("_right_unresolved.png")
    assert repair["components_debug"].endswith("_right_components.json")
    assert (tmp_path / repair["target_mask"]).is_file()
    assert (tmp_path / repair["unresolved_mask"]).is_file()
    assert (tmp_path / repair["components_debug"]).is_file()
    assert "finger_repair_incomplete" in incomplete["suspect"]

    complete = pages[1]
    assert complete["finger_repair"]["status"] == "complete"
    assert complete["finger_repair"]["donors"] == [1, 2]
    assert len(complete["finger_repair"]["components"]) == 1
    assert [
        entry["candidate_id"]
        for entry in complete["finger_repair"]["components"][0]["donors"]
    ] == [1, 2]
    assert complete["finger_repair"]["local_alignment"]["component_count"] == 1
    assert "hand_overlap" not in complete["suspect"]
    assert "finger_repair_incomplete" not in complete["suspect"]
    assert "finger_repair_incomplete" in spread["suspect"]


def test_legacy_repair_metadata_without_components_remains_compatible(
    tmp_path,
    monkeypatch,
):
    _, _, manifest, spread = _fixture(tmp_path, monkeypatch, output_layout="spread")

    def fake_repair(target, _target_mask, donors, **_kwargs):
        assert [donor["candidate_id"] for donor in donors] == [1, 2]
        metadata = _repair_metadata()
        metadata.pop("components")
        return target.copy(), metadata, np.zeros(target.shape[:2], np.uint8)

    monkeypatch.setattr(pipeline, "repair_finger_regions", fake_repair)

    page = pipeline.render_spread(tmp_path, manifest, spread)[0]
    repair = page["finger_repair"]

    assert repair["coverage"] == 1.0
    assert repair["donor_coverage"] == 1.0
    assert repair["donors"] == [1, 2]
    assert repair["alignment_scores"] == [0.9312, 0.8876]
    assert repair["fallback"]["mode"] == "preserve"
    assert "components" not in repair
    assert "local_alignment" not in repair
    assert "components_debug" not in repair


@pytest.mark.parametrize(
    ("output_layout", "expected_sides"),
    [
        ("spread", ["spread"]),
        ("split", ["right", "left"]),
    ],
)
def test_finger_repair_disabled_keeps_layout_and_pdf(
    tmp_path,
    monkeypatch,
    output_layout,
    expected_sides,
):
    _, _, manifest, spread = _fixture(
        tmp_path,
        monkeypatch,
        output_layout=output_layout,
        finger_repair=False,
    )
    if output_layout == "split":
        _install_split_geometry(monkeypatch)

    monkeypatch.setattr(
        pipeline,
        "repair_finger_regions",
        lambda *_a, **_k: pytest.fail("finger repair must stay disabled"),
    )

    pages = pipeline.render_spread(tmp_path, manifest, spread)
    assert [page["side"] for page in pages] == expected_sides
    assert all(page["finger_repair"]["status"] == "disabled" for page in pages)
    assert all("components_debug" not in page["finger_repair"] for page in pages)

    manifest["pages"] = pages
    pipeline.build_pdf(tmp_path, manifest)
    assert (tmp_path / "output/manga.pdf").is_file()
    assert len(PdfReader(tmp_path / "output/manga.pdf").pages) == len(expected_sides)
