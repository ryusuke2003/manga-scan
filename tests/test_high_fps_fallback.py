from manga_scan.high_fps_fallback import best_low_motion_run, fallback_reasons
from manga_scan.motion import Sample


def _record(*, suspect=(), left=(), right=()):
    return {
        "suspect": list(suspect),
        "page_suspect": {
            "left": list(left),
            "right": list(right),
        },
    }


def test_spread_fallback_requires_every_candidate_to_be_recoverably_bad():
    records = [
        _record(suspect=("low_sharpness",)),
        _record(suspect=("high_motion",)),
        _record(suspect=()),
    ]
    assert fallback_reasons(records, "spread") == []

    records[-1] = _record(suspect=("glare_overlap",))
    assert fallback_reasons(records, "spread") == [
        "glare_overlap",
        "high_motion",
        "low_sharpness",
    ]


def test_per_page_fallback_can_trigger_when_only_one_side_is_bad_across_all_candidates():
    records = [
        _record(left=("hand_overlap",), right=()),
        _record(left=("hand_overlap", "low_sharpness"), right=()),
        _record(left=("glare_overlap",), right=()),
    ]

    assert fallback_reasons(records, "per_page") == [
        "glare_overlap",
        "hand_overlap",
        "low_sharpness",
    ]


def test_fallback_ignores_reasons_that_high_fps_sampling_cannot_fix():
    records = [
        _record(suspect=("page_quad_uncertain",)),
        _record(suspect=("page_quad_uncertain",)),
    ]
    assert fallback_reasons(records, "spread") == []


def test_best_low_motion_run_requires_real_stable_duration_and_prefers_longest_run():
    samples = [
        Sample(0, 0.00, 1.0, 100),
        Sample(1, 0.05, 0.005, 100),
        Sample(2, 0.10, 0.006, 110),
        Sample(3, 0.15, 0.030, 100),
        Sample(4, 0.20, 0.004, 90),
        Sample(5, 0.25, 0.004, 95),
        Sample(6, 0.30, 0.005, 100),
        Sample(7, 0.35, 0.005, 105),
        Sample(8, 0.40, 0.030, 100),
    ]

    run = best_low_motion_run(samples, motion_threshold=0.012, fps=20, min_stable_seconds=0.18)
    assert [sample.index for sample in run] == [4, 5, 6, 7]

    assert best_low_motion_run(
        samples[:4],
        motion_threshold=0.012,
        fps=20,
        min_stable_seconds=0.18,
    ) == []
