from types import SimpleNamespace

import cv2
import numpy as np

import manga_scan.pipeline as pipeline
from manga_scan.candidate_prefilter import (
    build_candidate_plan,
    near_identical_candidate_frames,
    staged_fallback_reasons,
)
from manga_scan.config import Config
from manga_scan.motion import Sample


def _sample(index, *, motion=0.005, focus=100.0):
    return Sample(index=index, time=float(index), motion=motion, sharpness=focus)


def _pattern(seed, shape=(120, 180, 3)):
    rng = np.random.default_rng(seed)
    image = np.full(shape, 190, np.uint8)
    for _ in range(12):
        x = int(rng.integers(5, shape[1] - 25))
        y = int(rng.integers(5, shape[0] - 20))
        cv2.rectangle(image, (x, y), (x + 18, y + 12), (30, 30, 30), -1)
    return image


def test_prefilter_collapses_only_near_identical_consecutive_frames():
    base = _pattern(1)
    same = base.copy()
    local_change = base.copy()
    cv2.rectangle(local_change, (145, 70), (178, 118), (40, 40, 40), -1)

    assert near_identical_candidate_frames(base, same)
    assert not near_identical_candidate_frames(base, local_change)


def test_candidate_plan_keeps_late_candidate_and_reduces_large_pool():
    frames = [
        _pattern(1),
        _pattern(1),
        _pattern(2),
        _pattern(2),
        _pattern(3),
        _pattern(3),
        _pattern(4),
    ]
    samples = [
        _sample(index, motion=0.004 + index * 0.0002, focus=100 + index)
        for index in range(len(frames))
    ]

    plan = build_candidate_plan(samples, frames, primary_limit=3)

    assert plan["original_count"] == 7
    assert plan["representative_count"] < plan["original_count"]
    assert plan["primary_count"] == 3
    assert plan["representative_indices"][-1] == 6
    assert 6 in plan["primary_indices"]
    assert plan["duplicate_suppressed_count"] > 0


def test_candidate_plan_preserves_small_candidate_pool():
    frames = [_pattern(index) for index in range(3)]
    samples = [_sample(index) for index in range(3)]

    plan = build_candidate_plan(samples, frames, primary_limit=3)

    assert plan["representative_indices"] == [0, 1, 2]
    assert plan["primary_indices"] == [0, 1, 2]
    assert plan["duplicate_suppressed_count"] == 0


def test_candidate_plan_duplicate_groups_match_padded_representatives():
    frame = _pattern(30)
    frames = [frame.copy() for _ in range(7)]
    samples = [_sample(index) for index in range(7)]

    plan = build_candidate_plan(samples, frames, primary_limit=3)

    assert len(plan["representative_indices"]) == 3
    assert [group[-1] for group in plan["duplicate_groups"]] == plan[
        "representative_indices"
    ]
    flattened = [index for group in plan["duplicate_groups"] for index in group]
    assert flattened == list(range(7))


def test_staged_fallback_is_independent_from_high_fps_policy():
    bad = [
        {"suspect": ["low_sharpness"]},
        {"suspect": ["hand_overlap"]},
    ]
    mixed = [
        {"suspect": ["low_sharpness"]},
        {"suspect": []},
    ]

    assert staged_fallback_reasons(bad, "spread") == [
        "hand_overlap",
        "low_sharpness",
    ]
    assert staged_fallback_reasons(mixed, "spread") == []


def test_staged_fallback_expands_for_readability_blockers():
    geometry_bad = [
        {"suspect": ["page_quad_uncertain"]},
        {"suspect": ["page_quad_uncertain"]},
        {"suspect": ["page_quad_uncertain"]},
    ]
    exposure_bad = [
        {"suspect": ["underexposed"]},
        {"suspect": ["underexposed"]},
        {"suspect": ["underexposed"]},
    ]

    assert staged_fallback_reasons(geometry_bad, "spread") == [
        "page_quad_uncertain"
    ]
    assert staged_fallback_reasons(exposure_bad, "spread") == ["underexposed"]


def test_staged_fallback_per_page_checks_each_side_for_readability():
    records = [
        {
            "page_suspect": {
                "left": ["underexposed"],
                "right": [],
            }
        },
        {
            "page_suspect": {
                "left": ["page_quad_uncertain"],
                "right": [],
            }
        },
    ]

    assert staged_fallback_reasons(records, "per_page") == [
        "page_quad_uncertain",
        "underexposed",
    ]


def test_adjacent_candidate_groups_share_one_decode(monkeypatch):
    calls = []

    def fake_extract_frames(source, times, size, hwaccel):
        calls.append((source, list(times), size, hwaccel))
        return [
            np.full((size[1], size[0], 3), index, np.uint8)
            for index, _time in enumerate(times)
        ]

    monkeypatch.setattr(pipeline, "extract_frames", fake_extract_frames)
    manifest = {
        "source": "/tmp/book.mp4",
        "metadata": {"display_width": 200, "display_height": 100},
    }
    cfg = SimpleNamespace(analysis_width=100, hwaccel="none")
    groups = [
        [_sample(0), _sample(1)],
        [_sample(2)],
    ]
    timings = {}

    decoded = pipeline._decode_candidate_frame_groups(
        manifest,
        cfg,
        groups,
        timings,
    )

    assert len(calls) == 1
    assert calls[0][1] == [0.0, 1.0, 2.0]
    assert [len(group) for group in decoded] == [2, 1]
    assert timings["candidate_decode"]["calls"] == 3
    assert timings["candidate_decode_batch"]["calls"] == 1


def test_staged_candidate_evaluation_expands_when_primary_set_is_bad(
    monkeypatch,
    tmp_path,
):
    samples = [_sample(index) for index in range(5)]
    frames = [_pattern(index + 10) for index in range(5)]
    calls = []

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        calls.append([sample.index for sample in batch_samples])
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": ["low_sharpness"],
                "page_suspect": {
                    "left": ["low_sharpness"],
                    "right": ["low_sharpness"],
                },
            }
            for offset, sample in enumerate(batch_samples)
        ]

    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)
    records, plan = pipeline._process_staged_candidate_batch(
        tmp_path,
        {},
        object(),
        object(),
        "spread_0001",
        samples,
        frames,
        selection_mode="spread",
    )

    assert len(calls) == 2
    assert len(calls[0]) == 3
    assert len(records) == plan["representative_count"]
    assert plan["fallback_evaluated"] is True
    assert plan["fallback_reasons"] == ["low_sharpness"]


def test_bad_primary_rechecks_candidates_suppressed_by_prefilter(
    monkeypatch,
    tmp_path,
):
    frame = _pattern(35)
    frames = [frame.copy() for _ in range(7)]
    samples = [_sample(index) for index in range(7)]
    calls = []

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        calls.append([sample.index for sample in batch_samples])
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": ["page_quad_uncertain"],
                "page_suspect": {
                    "left": ["page_quad_uncertain"],
                    "right": ["page_quad_uncertain"],
                },
            }
            for offset, sample in enumerate(batch_samples)
        ]

    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)
    records, plan = pipeline._process_staged_candidate_batch(
        tmp_path,
        {},
        object(),
        object(),
        "spread_0001",
        samples,
        frames,
        selection_mode="spread",
    )

    assert len(calls[0]) == 3
    assert sorted(calls[0] + calls[1]) == list(range(7))
    assert len(records) == 7
    assert plan["fallback_evaluated"] is True
    assert sorted(plan["fallback_indices"]) == sorted(calls[1])


def test_staged_candidate_evaluation_stops_after_readable_primary(
    monkeypatch,
    tmp_path,
):
    samples = [_sample(index) for index in range(5)]
    frames = [_pattern(index + 20) for index in range(5)]
    calls = []

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        calls.append([sample.index for sample in batch_samples])
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": [] if offset == 0 else ["low_sharpness"],
                "page_suspect": {"left": [], "right": []},
            }
            for offset, sample in enumerate(batch_samples)
        ]

    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)
    records, plan = pipeline._process_staged_candidate_batch(
        tmp_path,
        {},
        object(),
        object(),
        "spread_0001",
        samples,
        frames,
        selection_mode="spread",
    )

    assert len(calls) == 1
    assert len(records) == 3
    assert plan["fallback_evaluated"] is False
    assert plan["evaluated_count"] == 3



def test_readable_three_candidate_stage_keeps_one_hand_only_temporal_peer(
    monkeypatch,
    tmp_path,
):
    samples = [_sample(index) for index in range(5)]
    frames = [_pattern(index + 40) for index in range(5)]
    runtime_cache = {}

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": [],
                "page_suspect": {"left": [], "right": []},
            }
            for offset, sample in enumerate(batch_samples)
        ]

    class Detector:
        def __init__(self):
            self.calls = 0

        def detect(self, image, roi):
            self.calls += 1
            return 0.0, np.zeros(image.shape[:2], np.uint8)

    detector = Detector()
    cfg = SimpleNamespace(hand_backend="mediapipe")
    roi = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)

    records, plan = pipeline._process_staged_candidate_batch(
        tmp_path,
        {"roi": roi},
        cfg,
        detector,
        "spread_0001",
        samples,
        frames,
        selection_mode="spread",
        runtime_cache=runtime_cache,
    )

    assert len(records) == 3
    assert plan["fallback_evaluated"] is False
    assert plan["temporal_peer_index"] is not None
    assert detector.calls == 1
    assert len(runtime_cache["_temporal_peers"]) == 1


def test_temporal_hand_result_can_trigger_normal_candidate_fallback(
    monkeypatch,
    tmp_path,
):
    samples = [_sample(index) for index in range(5)]
    frames = [_pattern(index + 80) for index in range(5)]
    records = [
        {
            "id": record_id,
            "time": samples[sample_index].time,
            "suspect": ["hand_overlap"],
            "page_suspect": {
                "left": ["hand_overlap"],
                "right": ["hand_overlap"],
            },
        }
        for record_id, sample_index in enumerate((0, 2, 4))
    ]
    plan = {
        "primary_indices": [0, 2, 4],
        "fallback_evaluated": False,
        "fallback_reasons": [],
        "fallback_indices": [],
        "evaluated_count": 3,
    }
    calls = []

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        calls.append([sample.index for sample in batch_samples])
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": [],
                "page_suspect": {"left": [], "right": []},
            }
            for offset, sample in enumerate(batch_samples)
        ]

    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)
    runtime_cache = {"_temporal_peers": [{"image": frames[1], "mask": None}]}
    expanded, updated, added_ids = (
        pipeline._expand_staged_candidates_after_temporal(
            tmp_path,
            {},
            object(),
            object(),
            "spread_0001",
            samples,
            frames,
            records,
            plan,
            selection_mode="spread",
            runtime_cache=runtime_cache,
        )
    )

    assert calls == [[1, 3]]
    assert len(expanded) == 5
    assert updated["fallback_evaluated"] is True
    assert updated["post_temporal_fallback"] is True
    assert updated["fallback_indices"] == [1, 3]
    assert added_ids == [3, 4]
    assert "_temporal_peers" not in runtime_cache


def test_temporal_hand_augmentation_accepts_hand_only_extra_peer(
    monkeypatch,
    tmp_path,
):
    image = _pattern(70)
    mask = np.zeros(image.shape[:2], np.uint8)
    records = [
        {
            "id": index,
            "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "path": f"candidate_{index}.png",
            "hand_mask": f"candidate_{index}_hand.png",
        }
        for index in range(3)
    ]
    runtime_cache = {
        index: {"image": image.copy(), "hand_mask": mask.copy()}
        for index in range(3)
    }
    runtime_cache["_temporal_peers"] = [
        {"image": image.copy(), "mask": mask.copy()}
    ]
    calls = []

    def fake_temporal(target, roi, peers, target_mask=None, padding=0.015):
        calls.append(len(peers))
        return np.zeros(target.shape[:2], np.uint8)

    monkeypatch.setattr(pipeline, "temporal_transient_mask", fake_temporal)
    cfg = SimpleNamespace(hand_backend="mediapipe", hand_padding=0.015)

    pipeline._augment_temporal_hand_masks(
        tmp_path,
        records,
        cfg,
        runtime_cache=runtime_cache,
    )

    assert calls == [3, 3, 3]


def test_prefilter_compares_book_roi_instead_of_static_or_noisy_background(
    monkeypatch,
    tmp_path,
):
    base = _pattern(60, shape=(160, 220, 3))
    frames = []
    rng = np.random.default_rng(61)
    for _ in range(4):
        image = rng.integers(0, 255, base.shape, dtype=np.uint8)
        image[32:128, 44:176] = base[32:128, 44:176]
        frames.append(image)
    samples = [_sample(index) for index in range(4)]

    def fake_batch(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        batch_samples,
        batch_frames,
        *,
        start_id=0,
        **_kwargs,
    ):
        return [
            {
                "id": start_id + offset,
                "time": sample.time,
                "suspect": [],
                "page_suspect": {"left": [], "right": []},
            }
            for offset, sample in enumerate(batch_samples)
        ]

    monkeypatch.setattr(pipeline, "_process_candidate_batch", fake_batch)
    roi = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    _records, plan = pipeline._process_staged_candidate_batch(
        tmp_path,
        {"roi": roi},
        SimpleNamespace(hand_backend="none"),
        object(),
        "spread_0001",
        samples,
        frames,
        selection_mode="spread",
    )

    assert plan["representative_count"] == 3
    assert plan["duplicate_suppressed_count"] == 1


def _preview(seed):
    rng = np.random.default_rng(seed)
    image = np.full((64, 64), 220, np.uint8)
    for _ in range(18):
        x = int(rng.integers(2, 56))
        y = int(rng.integers(2, 56))
        cv2.rectangle(image, (x, y), (x + 6, y + 5), 30, -1)
    return image


def test_normal_intervals_premerge_before_heavy_evaluation():
    first = [_sample(index) for index in range(3)]
    second = [
        Sample(index=10 + index, time=3.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    third = [
        Sample(index=20 + index, time=7.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    page_a = _preview(1)
    page_b = _preview(2)
    previews = {
        **{sample.time: page_a for sample in first},
        **{sample.time: page_a for sample in second},
        **{sample.time: page_b for sample in third},
    }
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        candidates_per_spread=3,
    )

    records = pipeline._prepared_interval_records(
        [first, second, third],
        cfg,
        [],
        frame_previews=previews,
    )

    assert len(records) == 2
    assert records[0]["normal_premerge"]["source_intervals"] == 2
    assert len(records[0]["candidates"]) == 6
    assert records[1]["normal_premerge"]["source_intervals"] == 1


def test_normal_premerge_does_not_confuse_reused_high_fps_indices():
    first = [
        Sample(index=0 + index, time=10.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    recovered = [
        Sample(index=0 + index, time=13.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    page = _preview(5)
    # Only the original interval has previews. The recovered high-fps samples
    # intentionally reuse indices 0..2, as the production recovery path does.
    previews = {
        first[0].time: page,
        first[-1].time: page,
        0: page,
        2: page,
    }
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        candidates_per_spread=3,
    )

    records = pipeline._prepared_interval_records(
        [first, recovered],
        cfg,
        [],
        frame_previews=previews,
    )

    assert len(records) == 2
    assert all(
        record["normal_premerge"]["source_intervals"] == 1
        for record in records
    )


def test_normal_premerge_rejects_interval_with_changed_end_page():
    left = [
        Sample(index=index, time=float(index), motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    mixed = [
        Sample(index=10 + index, time=3.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    page_a = _preview(6)
    page_b = _preview(7)
    previews = {
        left[0].time: page_a,
        left[-1].time: page_a,
        mixed[0].time: page_a,
        mixed[-1].time: page_b,
    }
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        candidates_per_spread=3,
    )

    records = pipeline._prepared_interval_records(
        [left, mixed],
        cfg,
        [],
        frame_previews=previews,
    )

    assert len(records) == 2


def test_normal_premerge_rejects_small_local_page_change():
    first = [_sample(index) for index in range(3)]
    second = [
        Sample(index=10 + index, time=3.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    base = np.full((64, 64), 225, np.uint8)
    for y in range(8, 58, 8):
        cv2.line(base, (4, y), (28, y), 40, 1)
        cv2.line(base, (36, y), (60, y), 55, 1)
    for x in (8, 20, 40, 52):
        cv2.line(base, (x, 5), (x, 59), 90, 1)
    changed = base.copy()
    cv2.rectangle(changed, (44, 28), (47, 31), 10, -1)
    previews = {
        first[0].time: base,
        first[-1].time: base,
        second[0].time: changed,
        second[-1].time: changed,
    }
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        candidates_per_spread=3,
    )

    records = pipeline._prepared_interval_records(
        [first, second],
        cfg,
        [],
        frame_previews=previews,
    )

    assert len(records) == 2


def test_normal_interval_premerge_requires_both_page_halves_to_match():
    first = [_sample(index) for index in range(3)]
    second = [
        Sample(index=10 + index, time=3.0 + index, motion=0.005, sharpness=100.0)
        for index in range(3)
    ]
    page = _preview(3)
    changed = page.copy()
    changed[:, 32:] = _preview(4)[:, 32:]
    previews = {
        **{sample.time: page for sample in first},
        **{sample.time: changed for sample in second},
    }
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        candidates_per_spread=3,
    )

    records = pipeline._prepared_interval_records(
        [first, second],
        cfg,
        [],
        frame_previews=previews,
    )

    assert len(records) == 2
