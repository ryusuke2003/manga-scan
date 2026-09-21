import cv2
import numpy as np

from manga_scan.page_contour import consensus_page_quads
from manga_scan.temporal_alignment import (
    align_page_detection,
    alignment_summary,
    estimate_frame_alignments,
    transform_normalized_quad,
)


def _textured_frame():
    image = np.full((360, 540, 3), (42, 72, 105), np.uint8)
    for y in range(35, 335, 45):
        for x in range(35, 510, 55):
            color = ((x * 3) % 220 + 25, (y * 5) % 210 + 30, (x + y) % 200 + 35)
            cv2.circle(image, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.line(image, (x - 10, y + 12), (x + 12, y - 9), (230, 230, 230), 2)
    cv2.putText(image, "MANGA", (145, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 3)
    return image


def _detection(candidate_id, left, right):
    return {
        "candidate_id": candidate_id,
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


def test_optical_flow_homography_maps_neighbor_back_to_anchor():
    anchor = _textured_frame()
    anchor_to_neighbor = np.asarray(
        [[1.0, 0.012, 24.0], [-0.008, 1.0, 15.0], [0.00002, -0.00001, 1.0]],
        np.float64,
    )
    neighbor = cv2.warpPerspective(anchor, anchor_to_neighbor, (anchor.shape[1], anchor.shape[0]))

    alignments = estimate_frame_alignments([neighbor, anchor], anchor_index=1)

    assert alignments[0]["status"] == "aligned"
    assert alignments[0]["inliers"] >= 8
    points = np.asarray([[[120.0, 90.0], [420.0, 285.0]]], np.float32)
    neighbor_points = cv2.perspectiveTransform(points, anchor_to_neighbor)
    recovered = cv2.perspectiveTransform(neighbor_points, alignments[0]["matrix"])
    np.testing.assert_allclose(recovered, points, atol=1.5)
    assert alignment_summary(alignments)["aligned"] == 1


def test_aligned_page_quads_form_consensus_after_camera_motion():
    shape = (400, 800, 3)
    left = np.asarray([[.08, .10], [.48, .11], [.47, .90], [.07, .89]], np.float32)
    right = np.asarray([[.52, .11], [.92, .10], [.93, .89], [.53, .90]], np.float32)
    anchor_to_neighbor = np.asarray(
        [[1, 0, 48], [0, 1, 24], [0, 0, 1]],
        np.float64,
    )
    neighbor_left = transform_normalized_quad(left, anchor_to_neighbor, shape, shape)
    neighbor_right = transform_normalized_quad(right, anchor_to_neighbor, shape, shape)
    anchor_detection = _detection(1, left, right)
    neighbor_detection = _detection(0, neighbor_left, neighbor_right)
    aligned_neighbor = align_page_detection(
        neighbor_detection,
        {"status": "aligned", "matrix": np.linalg.inv(anchor_to_neighbor)},
        shape,
        shape,
    )

    result = consensus_page_quads(
        [aligned_neighbor, anchor_detection],
        min_confidence=0.5,
        max_corner_deviation=0.04,
        anchor_ids={"left": 1, "right": 1},
    )

    assert aligned_neighbor["alignment_applied"]
    assert result["detected"]
    assert result["left"]["consensus_count"] == 2
    assert result["right"]["consensus_count"] == 2
    np.testing.assert_allclose(result["left"]["quad"], left, atol=1e-5)
    np.testing.assert_allclose(result["right"]["quad"], right, atol=1e-5)


def test_featureless_neighbor_falls_back_without_warping_detection():
    blank = np.full((180, 320, 3), 128, np.uint8)
    alignments = estimate_frame_alignments([blank, blank.copy()], anchor_index=0)
    left = [[.1, .1], [.48, .1], [.48, .9], [.1, .9]]
    right = [[.52, .1], [.9, .1], [.9, .9], [.52, .9]]
    detection = _detection(1, left, right)

    aligned = align_page_detection(detection, alignments[1], blank.shape, blank.shape)

    assert alignments[1]["status"] == "insufficient_features"
    assert not aligned["alignment_applied"]
    np.testing.assert_allclose(aligned["left"]["quad"], left)
    np.testing.assert_allclose(aligned["right"]["quad"], right)
