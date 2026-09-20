import numpy as np

import manga_scan.reference_candidates as candidates_module
from manga_scan.config import Config
from manga_scan.ingest import skip_cover
from manga_scan.reference_candidates import (
    scan_reference_candidates,
    select_reference_candidates,
)
from manga_scan.storage import save_manifest


def test_select_reference_candidates_keeps_high_scores_temporally_distinct():
    records = [
        {"time": 1.0, "score": 0.90},
        {"time": 1.3, "score": 0.95},
        {"time": 3.0, "score": 0.80},
        {"time": 5.0, "score": 0.70},
    ]

    selected = select_reference_candidates(records, limit=3, min_separation=1.0)

    assert [item["time"] for item in selected] == [1.3, 3.0, 5.0]


def test_scan_reference_candidates_ranks_likely_still_spreads(monkeypatch):
    frames = []
    for index, (timestamp, confidence) in enumerate(
        [(0.0, 0.3), (1.0, 0.62), (2.0, 0.84), (3.0, 0.71), (5.0, 0.76)]
    ):
        frame = np.full((120, 200, 3), 120, dtype=np.uint8)
        frame[0, 0] = round(confidence * 100)
        frames.append((index, timestamp, frame))

    def fake_sample_frames(*_args, **_kwargs):
        yield from frames

    def fake_detect(image, min_confidence):
        confidence = float(image[0, 0, 0]) / 100
        return {
            "detected": confidence >= min_confidence,
            "confidence": confidence,
            "stage": "complete" if confidence >= min_confidence else "pages",
        }

    monkeypatch.setattr(candidates_module, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(candidates_module, "detect_reference_spread", fake_detect)
    monkeypatch.setattr(candidates_module, "sharpness", lambda _image: 250.0)

    cfg = Config(hand_backend="none", finger_repair=False, auto_rotation=False)
    result = scan_reference_candidates(
        "unused.mp4",
        {
            "duration": 8.0,
            "fps": 30.0,
            "display_width": 200,
            "display_height": 120,
        },
        cfg,
        limit=3,
    )

    assert len(result) == 3
    assert result[0]["time"] == 2.0
    assert result[0]["confidence"] == 0.84
    assert all(item["detected"] for item in result)
    assert all("_frame" in item for item in result)


def test_skip_cover_generates_reference_suggestion_previews(tmp_path, monkeypatch):
    cfg = Config(hand_backend="none", finger_repair=False, auto_rotation=False)
    manifest = {
        "version": 2,
        "source": "/tmp/book.mp4",
        "metadata": {
            "duration": 10.0,
            "fps": 30.0,
            "display_width": 300,
            "display_height": 180,
        },
        "config": cfg.to_dict(),
        "rotation_detection": {
            "rotation": 0,
            "confidence": 1.0,
            "source": "manual",
            "confirmed": True,
        },
        "roi": None,
        "cover": {"status": "pending", "time": 0.0, "roi": None},
        "reference": {"time": 0.0, "confirmed": False},
        "status": "ready",
        "warnings": [],
        "spreads": [],
        "pages": [],
        "pdf_stale": True,
        "progress": 0,
        "message": "",
    }
    save_manifest(tmp_path, manifest)
    monkeypatch.setattr(
        "manga_scan.ingest.scan_reference_candidates",
        lambda *_args, **_kwargs: [
            {
                "time": 2.4,
                "score": 0.91,
                "confidence": 0.82,
                "motion": 0.003,
                "sharpness": 240.0,
                "detected": True,
                "stage": "complete",
                "_frame": np.full((90, 160, 3), 180, dtype=np.uint8),
            }
        ],
    )

    updated = skip_cover(tmp_path)

    assert updated["cover"]["status"] == "skipped"
    assert len(updated["reference"]["candidates"]) == 1
    candidate = updated["reference"]["candidates"][0]
    assert candidate["time"] == 2.4
    assert "_frame" not in candidate
    assert candidate["preview"] == "source/reference_candidates/candidate_01.jpg"
    assert (tmp_path / candidate["preview"]).is_file()


def test_reference_suggestion_failure_does_not_block_manual_setup(tmp_path, monkeypatch):
    cfg = Config(hand_backend="none", finger_repair=False, auto_rotation=False)
    manifest = {
        "version": 2,
        "source": "/tmp/book.mp4",
        "metadata": {"duration": 10.0, "fps": 30.0, "display_width": 300, "display_height": 180},
        "config": cfg.to_dict(),
        "rotation_detection": {"rotation": 0, "confidence": 1.0, "source": "manual", "confirmed": True},
        "roi": None,
        "cover": {"status": "pending", "time": 0.0, "roi": None},
        "reference": {"time": 0.0, "confirmed": False},
        "status": "ready",
        "warnings": [],
        "spreads": [],
        "pages": [],
        "pdf_stale": True,
        "progress": 0,
        "message": "",
    }
    save_manifest(tmp_path, manifest)
    monkeypatch.setattr(
        "manga_scan.ingest.scan_reference_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )

    updated = skip_cover(tmp_path)

    assert updated["cover"]["status"] == "skipped"
    assert updated["reference"]["candidates"] == []
    assert updated["reference"]["candidate_search_error"] == "scan failed"
    assert "基準にする見開き" in updated["message"]
