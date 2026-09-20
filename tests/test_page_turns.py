from dataclasses import dataclass

from manga_scan.page_turns import analyze_page_turns, detect_page_turn_events


@dataclass
class MotionSample:
    index: int
    time: float
    motion: float
    sharpness: float = 100.0


def _samples(values, fps=10):
    return [
        MotionSample(index, index / fps, motion)
        for index, motion in enumerate(values)
    ]


def test_initial_synthetic_motion_is_not_a_turn():
    samples = _samples([1.0, 0.005, 0.006, 0.005])

    assert detect_page_turn_events(samples, 0.012, 0.025, 10) == []


def test_brief_page_between_two_turns_becomes_missing_candidate():
    samples = _samples(
        [
            1.0,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.030,
            0.050,
            0.040,
            0.017,
            0.014,
            0.016,
            0.040,
            0.050,
            0.030,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
        ]
    )
    segments = [samples[1:6], samples[15:20]]

    analysis = analyze_page_turns(samples, segments, 0.012, 0.025, 10)

    assert analysis["turn_count"] == 2
    assert len(analysis["missing_candidates"]) == 1
    candidate = analysis["missing_candidates"][0]
    assert candidate["time"] == 1.0
    assert candidate["motion"] == 0.014
    assert candidate["before"] == "spread_0001"
    assert candidate["after"] == "spread_0002"


def test_stable_spread_between_turns_is_not_missing():
    samples = _samples(
        [
            1.0,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.040,
            0.050,
            0.040,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.040,
            0.050,
            0.040,
            0.006,
            0.006,
        ]
    )
    segments = [samples[1:6], samples[9:14], samples[17:19]]

    analysis = analyze_page_turns(samples, segments, 0.012, 0.025, 10)

    assert analysis["turn_count"] == 2
    assert analysis["missing_candidates"] == []


def test_noisy_single_turn_is_merged_instead_of_double_counted():
    samples = _samples(
        [1.0, 0.006, 0.006, 0.006, 0.040, 0.050, 0.023, 0.040, 0.045, 0.006, 0.006]
    )

    events = detect_page_turn_events(samples, 0.012, 0.025, 10)

    assert len(events) == 1
    assert events[0]["peak_motion"] == 0.050
