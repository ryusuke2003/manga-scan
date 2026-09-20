import json
from pathlib import Path

import pytest

from scripts.run_real_benchmark import (
    _repair_result_isolated,
    polygon_iou,
    validate_manifest,
)


def test_real_benchmark_manifest_is_valid():
    data = json.loads(Path("benchmarks/real/cases.json").read_text())
    validate_manifest(data)
    assert len(data["videos"]) == 3
    assert sum(len(video["samples"]) for video in data["videos"]) == 10
    assert sum(len(video.get("repair_pairs", [])) for video in data["videos"]) == 2


def test_polygon_iou_identical_and_disjoint():
    a = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    b = [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]
    assert polygon_iou(a, a) == 1.0
    assert polygon_iou(a, b) == 0.0


def test_real_benchmark_consensus_offsets_must_include_center_frame():
    data = json.loads(Path("benchmarks/real/cases.json").read_text())
    data["videos"][0]["samples"][0]["consensus_offsets"] = [-0.5, 0.5, 1.0]

    with pytest.raises(ValueError, match="must include 0"):
        validate_manifest(data)


def test_repair_worker_reports_python_errors_without_aborting(tmp_path):
    result = _repair_result_isolated(
        {"id": "video", "expected_rotation": 0},
        {"id": "pair", "target_time": 0.0, "donor_times": [1.0]},
        tmp_path / "missing.mov",
        tmp_path / "missing.task",
        timeout=10,
    )

    assert result["passed"] is False
    assert result["video_id"] == "video"
    assert result["pair_id"] == "pair"
    assert "Hand model missing" in result["error"]
