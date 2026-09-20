import json
from pathlib import Path

from scripts.run_real_benchmark import polygon_iou, validate_manifest


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
