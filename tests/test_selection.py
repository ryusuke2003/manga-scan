import cv2
import numpy as np
import pytest

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.selection import choose_candidate_selection, score_candidate_pages
from manga_scan.storage import save_manifest


def _record(candidate_id, spread_score, left_score, right_score):
    return {
        "id": candidate_id,
        "metrics": {"score": spread_score},
        "page_metrics": {
            "left": {"score": left_score},
            "right": {"score": right_score},
        },
    }


def test_per_page_selection_can_choose_different_candidate_ids():
    records = [
        _record(0, 9.0, 2.0, 8.0),
        _record(1, 8.0, 9.0, 3.0),
    ]

    selected, selected_pages = choose_candidate_selection(records, "per_page")

    assert selected == 0
    assert selected_pages == {"left": 1, "right": 0}


def test_spread_selection_keeps_one_candidate_for_both_pages():
    records = [
        _record(0, 4.0, 9.0, 1.0),
        _record(1, 6.0, 1.0, 9.0),
    ]

    selected, selected_pages = choose_candidate_selection(records, "spread")

    assert selected == 1
    assert selected_pages == {"left": 1, "right": 1}


def test_page_scoring_penalizes_hand_only_on_covered_side():
    image = np.full((100, 200, 3), 220, np.uint8)
    image[::4, :] = 30
    hand_mask = np.zeros((100, 200), np.uint8)
    hand_mask[:, :100] = 255
    cfg = Config(hand_overlap_weight=8.0)
    geometry = {"distortion": 0.0, "flatness_proxy": 0.0}

    metrics, spine = score_candidate_pages(
        image,
        hand_mask,
        motion=0.0,
        config=cfg,
        geometry_metrics=geometry,
        hand_enabled=True,
    )

    assert spine == 100
    assert metrics["left"]["hand_overlap"] == pytest.approx(1.0)
    assert metrics["right"]["hand_overlap"] == pytest.approx(0.0)
    assert metrics["right"]["score"] > metrics["left"]["score"]


def test_page_scoring_rotates_before_assigning_visual_sides():
    upright = np.full((80, 160, 3), 220, np.uint8)
    upright[::4, :] = 30
    upright_mask = np.zeros((80, 160), np.uint8)
    upright_mask[:, :80] = 255

    sideways = np.rot90(upright, 1).copy()
    sideways_mask = np.rot90(upright_mask, 1).copy()
    cfg = Config(rotation=90, hand_overlap_weight=8.0)

    metrics, spine = score_candidate_pages(
        sideways,
        sideways_mask,
        motion=0.0,
        config=cfg,
        geometry_metrics={"distortion": 0.0, "flatness_proxy": 0.0},
        hand_enabled=True,
    )

    assert spine == 80
    assert metrics["left"]["hand_overlap"] == pytest.approx(1.0)
    assert metrics["right"]["hand_overlap"] == pytest.approx(0.0)


def test_render_spread_rotates_before_left_right_split(tmp_path, monkeypatch):
    upright = np.empty((80, 160, 3), np.uint8)
    upright[:, :80] = (20, 40, 220)
    upright[:, 80:] = (220, 80, 20)
    sideways = np.rot90(upright, 1).copy()

    cfg = Config(
        hand_backend="none",
        rotation=90,
        dewarp_mode="off",
        perspective_mode="spread",
        image_format="png",
    )
    manifest = {"config": cfg.to_dict(), "source": "unused.mov"}
    spread = {
        "id": "spread_0001",
        "selected": 0,
        "candidates": [
            {
                "id": 0,
                "time": 1.0,
                "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "suspect": [],
                "page_suspect": {"left": [], "right": []},
            }
        ],
        "extra_suspect": [],
    }

    monkeypatch.setattr(
        pipeline,
        "extract_frame",
        lambda *args, **kwargs: sideways.copy(),
    )

    pages = pipeline.render_spread(tmp_path, manifest, spread)
    by_side = {page["side"]: page for page in pages}
    left = cv2.imread(str(tmp_path / by_side["left"]["path"]))
    right = cv2.imread(str(tmp_path / by_side["right"]["path"]))

    np.testing.assert_allclose(left[left.shape[0] // 2, left.shape[1] // 2], (20, 40, 220), atol=2)
    np.testing.assert_allclose(right[right.shape[0] // 2, right.shape[1] // 2], (220, 80, 20), atol=2)


def test_disabled_hand_detection_stays_none_per_page():
    image = np.full((80, 160, 3), 220, np.uint8)
    cfg = Config()
    metrics, _ = score_candidate_pages(
        image,
        np.zeros((80, 160), np.uint8),
        motion=0.0,
        config=cfg,
        geometry_metrics={"distortion": 0.0, "flatness_proxy": 0.0},
        hand_enabled=False,
    )
    assert metrics["left"]["hand_overlap"] is None
    assert metrics["right"]["hand_overlap"] is None


def test_review_can_change_only_one_page_candidate(tmp_path, monkeypatch):
    project = tmp_path / "book"
    project.mkdir()
    manifest = {
        "config": Config(candidate_selection_mode="per_page").to_dict(),
        "metadata": {"duration": 10.0},
        "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "spreads": [
            {
                "id": "spread_0001",
                "candidates": [{"id": 0}, {"id": 1}],
                "selected": 0,
                "selected_pages": {"left": 0, "right": 0},
            }
        ],
        "pages": [
            {
                "id": "spread_0001_left",
                "spread_id": "spread_0001",
                "side": "left",
                "enabled": True,
            },
            {
                "id": "spread_0001_right",
                "spread_id": "spread_0001",
                "side": "right",
                "enabled": True,
            },
        ],
        "pdf_stale": False,
    }
    save_manifest(project, manifest)

    def fake_render(_project, _manifest, spread):
        return [
            {
                "id": f"spread_0001_{side}",
                "spread_id": "spread_0001",
                "side": side,
                "enabled": True,
                "candidate_id": spread["selected_pages"][side],
            }
            for side in ("left", "right")
        ]

    monkeypatch.setattr(pipeline, "render_spread", fake_render)
    updated = pipeline.edit(
        project,
        "select_candidate",
        spread_id="spread_0001",
        candidate_id=1,
        side="left",
    )

    spread = updated["spreads"][0]
    assert spread["selected"] == 0
    assert spread["selected_pages"] == {"left": 1, "right": 0}
    pages = {page["side"]: page for page in updated["pages"]}
    assert pages["left"]["candidate_id"] == 1
    assert pages["right"]["candidate_id"] == 0
    assert updated["pdf_stale"]


@pytest.mark.parametrize("mode", ["unknown", "", "page"])
def test_config_rejects_unknown_candidate_selection_mode(mode):
    with pytest.raises(ValueError):
        Config.from_dict({"candidate_selection_mode": mode})
