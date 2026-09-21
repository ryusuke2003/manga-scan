import cv2
import numpy as np

from manga_scan.config import Config
from manga_scan.pipeline_render_helpers import detect_spread_page_consensus
from manga_scan.temporal_alignment import transform_normalized_quad


def _textured_frame():
    rng = np.random.default_rng(7)
    image = np.full((400, 800, 3), (48, 78, 112), np.uint8)
    for index in range(120):
        x = int(rng.integers(30, 770))
        y = int(rng.integers(30, 370))
        radius = int(rng.integers(3, 10))
        value = int(rng.integers(40, 245))
        cv2.circle(
            image,
            (x, y),
            radius,
            (value, min(255, value + 17), max(0, value - 11)),
            -1,
            cv2.LINE_AA,
        )
        if index % 5 == 0:
            cv2.line(
                image,
                (max(0, x - 14), min(399, y + 10)),
                (min(799, x + 12), max(0, y - 9)),
                (235, 235, 235),
                2,
            )
    cv2.putText(image, "MANGA 81", (270, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 3)
    return image


def _detection(left, right):
    return {
        "detected": True,
        "confidence": 0.9,
        "left": {
            "quad": np.asarray(left, np.float32).tolist(),
            "confidence": 0.9,
            "detected": True,
            "touches_frame": False,
        },
        "right": {
            "quad": np.asarray(right, np.float32).tolist(),
            "confidence": 0.9,
            "detected": True,
            "touches_frame": False,
        },
    }


def test_spread_contour_consensus_aligns_moved_candidates_to_selected_anchor(tmp_path):
    anchor = _textured_frame()
    anchor_to_neighbor = np.asarray(
        [[1.0, 0.004, 24.0], [-0.003, 1.0, 12.0], [0.000005, 0.0, 1.0]],
        np.float64,
    )
    neighbor = cv2.warpPerspective(
        anchor,
        anchor_to_neighbor,
        (anchor.shape[1], anchor.shape[0]),
        borderMode=cv2.BORDER_REFLECT101,
    )
    left = np.asarray([[.08, .10], [.48, .11], [.47, .90], [.07, .89]], np.float32)
    right = np.asarray([[.52, .11], [.92, .10], [.93, .89], [.53, .90]], np.float32)
    neighbor_left = transform_normalized_quad(
        left,
        anchor_to_neighbor,
        anchor.shape,
        neighbor.shape,
    )
    neighbor_right = transform_normalized_quad(
        right,
        anchor_to_neighbor,
        anchor.shape,
        neighbor.shape,
    )

    candidate_dir = tmp_path / "candidates" / "spread_0001"
    candidate_dir.mkdir(parents=True)
    cv2.imwrite(str(candidate_dir / "candidate_00.png"), neighbor)
    cv2.imwrite(str(candidate_dir / "candidate_01.png"), anchor)

    detections = iter(
        [
            _detection(neighbor_left, neighbor_right),
            _detection(left, right),
        ]
    )

    def fake_detect(_image, _roi, **_kwargs):
        return next(detections)

    spread = {
        "id": "spread_0001",
        "selected": 1,
        "candidates": [
            {
                "id": 0,
                "path": "candidates/spread_0001/candidate_00.png",
                "roi": [[.05, .05], [.95, .05], [.95, .95], [.05, .95]],
            },
            {
                "id": 1,
                "path": "candidates/spread_0001/candidate_01.png",
                "roi": [[.05, .05], [.95, .05], [.95, .95], [.05, .95]],
            },
        ],
    }
    cfg = Config(hand_backend="none", finger_repair=False)

    result = detect_spread_page_consensus(
        tmp_path,
        spread,
        cfg,
        anchor_ids={"left": 1, "right": 1},
        detect_page_quads_fn=fake_detect,
    )

    assert result["detected"]
    assert result["left"]["consensus_count"] == 2
    assert result["right"]["consensus_count"] == 2
    np.testing.assert_allclose(result["left"]["quad"], left, atol=0.01)
    np.testing.assert_allclose(result["right"]["quad"], right, atol=0.01)
    assert result["consensus"]["alignment_by_side"]["left"]["aligned"] == 1
    assert result["consensus"]["alignment_by_side"]["right"]["aligned"] == 1
