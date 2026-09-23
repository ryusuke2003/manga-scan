import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_real_benchmark import (
    _candidate_prefilter_result,
    _repair_result_isolated,
    polygon_iou,
    validate_manifest,
)
from scripts.run_recovery_speed_benchmark import _strict_failures


def test_real_benchmark_manifest_is_valid():
    data = json.loads(Path("benchmarks/real/cases.json").read_text())
    validate_manifest(data)
    assert len(data["videos"]) == 4
    assert sum(len(video["samples"]) for video in data["videos"]) == 15
    assert sum(len(video.get("repair_pairs", [])) for video in data["videos"]) == 2


def test_polygon_iou_identical_and_disjoint():
    a = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    b = [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]
    assert polygon_iou(a, a) == 1.0
    assert polygon_iou(a, b) == 0.0


def test_recovery_benchmark_requires_annotated_pages_even_if_legacy_missed():
    report = {
        "legacy_recoveries_lost": [],
        "annotated_page_coverage_legacy": {"page": False},
        "annotated_page_coverage_grouped": {"page": False},
    }
    assert _strict_failures(report) == ["annotated spreads were missed: page"]
    report["annotated_page_coverage_grouped"]["page"] = True
    assert _strict_failures(report) == []


def test_real_benchmark_consensus_offsets_must_include_center_frame():
    data = json.loads(Path("benchmarks/real/cases.json").read_text())
    data["videos"][0]["samples"][0]["consensus_offsets"] = [-0.5, 0.5, 1.0]

    with pytest.raises(ValueError, match="must include 0"):
        validate_manifest(data)


def test_real_benchmark_candidate_prefilter_preserves_changed_hand_frame(monkeypatch):
    base = np.full((120, 180, 3), 210, np.uint8)
    target = base.copy()
    target[55:115, 125:179] = 35

    def fake_extract_frame(_path, timestamp, width=None, hwaccel="none"):
        assert width == 480
        return target.copy() if timestamp == 10.0 else base.copy()

    monkeypatch.setattr("manga_scan.video.extract_frame", fake_extract_frame)
    result = _candidate_prefilter_result(
        {"id": "video", "expected_rotation": 0},
        {
            "id": "moving-hand",
            "target_time": 10.0,
            "donor_times": [11.0],
            "spread_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
        Path("unused.mov"),
    )

    assert result["passed"] is True
    assert result["donors"][0]["collapsed_as_near_identical"] is False


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
