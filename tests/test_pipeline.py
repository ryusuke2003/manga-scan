import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.ingest import create_project, reopen_cover_roi, set_cover_roi, set_setup_frame
from manga_scan.pipeline import edit, run
from manga_scan.processing_control import request_cancel
from manga_scan.storage import read_manifest
from manga_scan.video import extract_frame, extract_frames, probe, sample_frames

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
ROI = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "demo", Path(__file__).parents[1] / "scripts/make_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tmp_path = tmp_path_factory.mktemp("demo-video")
    return module.make_demo(tmp_path / "video with spaces.mp4")


def test_candidate_batch_preserves_input_order(monkeypatch, tmp_path):
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        processing_workers=3,
    )
    samples = [
        pipeline.Sample(index, float(index), 0.0, 100.0)
        for index in range(5)
    ]
    frames = [
        np.zeros((8, 8, 3), np.uint8)
        for _ in samples
    ]

    def fake_candidate(
        project,
        manifest,
        cfg,
        detector,
        spread_id,
        number,
        sample,
        **kwargs,
    ):
        return {"id": number, "time": sample.time}

    monkeypatch.setattr(pipeline, "candidate", fake_candidate)
    records = pipeline._process_candidate_batch(
        tmp_path,
        {},
        cfg,
        object(),
        "spread_0001",
        samples,
        frames,
        start_id=7,
    )

    assert [record["id"] for record in records] == [7, 8, 9, 10, 11]
    assert [record["time"] for record in records] == [0.0, 1.0, 2.0, 3.0, 4.0]


def _duplicate_fixture(spread_id, hand, focus=0.8, risks=(), *, root=None):
    spread = {
        "id": spread_id,
        "candidates": [{
            "id": 0,
            "metrics": {"hand_overlap": hand, "sharpness_uniformity": focus},
        }],
    }
    if root:
        spread["duplicate_of"] = root
    page = {
        "id": f"{spread_id}_spread",
        "spread_id": spread_id,
        "side": "spread",
        "candidate_id": 0,
        "enabled": root is None,
        "suspect": list(risks),
    }
    return spread, page


def test_later_duplicate_replaces_hand_covered_page_then_repairs_it():
    old, old_page = _duplicate_fixture(
        "spread_0001", 0.11,
        risks=("finger_repair_incomplete", "final_unresolved_finger"),
    )
    hand, hand_page = _duplicate_fixture(
        "spread_0002", 0.006,
        risks=("finger_repair_incomplete", "final_unresolved_finger"),
        root=old["id"],
    )
    clean, clean_page = _duplicate_fixture(
        "spread_0003", 0.008, root=old["id"],
    )
    manifest = {"spreads": [old], "pages": [old_page]}

    pipeline._promote_cleaner_duplicate(manifest, hand, [hand_page])
    assert not old_page["enabled"] and hand_page["enabled"]
    manifest["spreads"].append(hand)
    manifest["pages"].append(hand_page)
    pipeline._promote_cleaner_duplicate(manifest, clean, [clean_page])
    assert not hand_page["enabled"] and clean_page["enabled"]


def test_resume_dedupe_uses_cleaner_promoted_spread(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pipeline,
        "selected_spread_preview",
        lambda project, spread, cfg: spread["id"],
    )
    manifest = {
        "spreads": [
            {"id": "hand"},
            {"id": "clean", "duplicate_of": "hand"},
            {"id": "next"},
        ],
        "pages": [
            {"spread_id": "hand", "enabled": False},
            {"spread_id": "clean", "enabled": True},
            {"spread_id": "next", "enabled": True},
        ],
    }

    assert pipeline._resume_previous_spreads(tmp_path, manifest, Config()) == [
        ("hand", "clean"),
        ("next", "next"),
    ]


@pytest.mark.parametrize("hand,focus,risks", [
    (0.01, 0.5, ()),
    (0.01, 0.8, ("final_edge_crop_suspected",)),
    (0.01, 0.8, ("final_unresolved_finger",)),
    (None, 0.8, ()),
])
def test_duplicate_promotion_rejects_new_quality_risk(hand, focus, risks):
    old, old_page = _duplicate_fixture("spread_0001", 0.1)
    newer, newer_page = _duplicate_fixture(
        "spread_0002", hand, focus, risks, root=old["id"],
    )
    manifest = {"spreads": [old], "pages": [old_page]}

    pipeline._promote_cleaner_duplicate(manifest, newer, [newer_page])

    assert old_page["enabled"] and not newer_page["enabled"]


def test_manual_duplicate_selection_blocks_later_auto_promotion():
    old, old_page = _duplicate_fixture("spread_0001", 0.1)
    newer, newer_page = _duplicate_fixture("spread_0002", 0.0, root=old["id"])
    manifest = {
        "spreads": [old],
        "pages": [old_page],
        "manual_duplicate_groups": [old["id"]],
    }

    pipeline._promote_cleaner_duplicate(manifest, newer, [newer_page])

    assert old_page["enabled"] and not newer_page["enabled"]


def test_manual_duplicate_group_marking_and_undo_state():
    old, old_page = _duplicate_fixture("spread_0001", 0.1)
    newer, newer_page = _duplicate_fixture("spread_0002", 0.0, root=old["id"])
    manifest = {"spreads": [old, newer], "pages": [old_page, newer_page]}
    before = pipeline._page_review_state(manifest)

    pipeline._protect_manual_duplicate_group(manifest, newer)
    assert manifest["manual_duplicate_groups"] == [old["id"]]
    pipeline._restore_page_review_state(manifest, before)
    assert manifest["manual_duplicate_groups"] == []


def test_end_to_end_dedupe_review_pdf(video, tmp_path):
    project = tmp_path / "book"
    cfg = Config(output_layout="split", hand_backend="none", finger_repair=False,
                 analysis_width=480, candidates_per_spread=3)
    create_project(video, project, cfg)
    manifest = run(project, ROI)
    assert manifest["status"] == "complete"
    assert len(manifest["spreads"]) == 4
    assert len(manifest["pages"]) == 8
    assert all("final_quality" in page for page in manifest["pages"])
    assert all("reasons" in page["final_quality"] for page in manifest["pages"])
    assert sum(p["enabled"] for p in manifest["pages"]) == 6
    assert manifest["spreads"][2]["duplicate_of"] == "spread_0002"
    assert [p["side"] for p in manifest["pages"]][:2] == ["right", "left"]
    assert len(PdfReader(project / "output/manga.pdf").pages) == 6
    with zipfile.ZipFile(project / "output/manga.cbz") as archive:
        assert archive.namelist() == [f"{i:03d}.png" for i in range(1, 7)]
        enabled = [page for page in manifest["pages"] if page["enabled"]]
        assert archive.read("001.png") == (project / enabled[0]["path"]).read_bytes()
    assert manifest["cbz"] == "output/manga.cbz"
    assert (project / "debug/motion.csv").is_file()
    assert (project / "debug/scores.csv").is_file()
    assert (project / "debug/contact_sheet.jpg").is_file()
    performance = json.loads((project / "debug/performance.json").read_text())
    assert performance == manifest["performance"]
    assert performance["candidate_decode"]["calls"] > 0
    assert "PERF candidate_decode" in (project / "debug/process.log").read_text()
    assert not list((project / "frames_lowres").iterdir())
    page = manifest["pages"][0]["id"]
    manifest = edit(project, "toggle_page", page_id=page)
    assert manifest["pdf_stale"]
    manifest = edit(project, "select_candidate", spread_id="spread_0001", candidate_id=1)
    assert not manifest["pages"][0]["enabled"]  # Candidate switch preserves user's exclusion.
    manifest = edit(project, "swap", spread_id="spread_0001")
    assert manifest["pages"][0]["side"] == "left"
    edit(project, "export")
    assert len(PdfReader(project / "output/manga.pdf").pages) == 5
    with zipfile.ZipFile(project / "output/manga.cbz") as archive:
        assert len(archive.namelist()) == 5
    manifest = edit(project, "add_frame", time=0.8)
    assert len(manifest["pages"]) == 10
    manual_spread = next(
        spread
        for spread in manifest["spreads"]
        if "manual_frame" in spread.get("extra_suspect", [])
    )
    manual_reason = "manual_frame_motion_unmeasured"
    assert "manual_frame" in manual_spread["suspect"]
    assert manual_reason in manual_spread["suspect"]
    candidate = manual_spread["candidates"][0]
    assert manual_reason in candidate["suspect"]
    assert manual_reason in manual_spread["extra_suspect"]
    candidate_json = json.loads(
        (project / Path(candidate["path"]).with_suffix(".json")).read_text()
    )
    assert manual_reason in candidate_json["suspect"]
    manual_pages = [
        page for page in manifest["pages"] if page["spread_id"] == manual_spread["id"]
    ]
    assert len(manual_pages) == 2
    assert all(manual_reason in page["suspect"] for page in manual_pages)
    assert read_manifest(project)["pdf_stale"]
    with pytest.raises(ValueError, match="already processed"):
        run(project, ROI)


def test_scan_tracks_roi_between_stable_spreads(video, tmp_path, monkeypatch):
    project = tmp_path / "tracked-book"
    cfg = Config(
        output_layout="spread",
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=2,
        roi_tracking=True,
    )
    create_project(video, project, cfg)

    calls = []

    def fake_track(previous_image, current_image, previous_roi, reference_roi, **_kwargs):
        calls.append((previous_image.shape, current_image.shape))
        shifted = [
            [min(0.99, float(x) + 0.01), float(y)]
            for x, y in previous_roi
        ]
        return {
            "tracked": True,
            "status": "tracked",
            "roi": shifted,
            "step_shift": 0.01,
            "total_shift": 0.01 * len(calls),
            "area_ratio": 1.0,
            "alignment": {"status": "aligned", "inliers": 20},
        }

    monkeypatch.setattr(pipeline, "track_spread_roi", fake_track)
    manifest = run(project, ROI)

    assert len(calls) == len(manifest["spreads"]) - 1
    assert manifest["spreads"][0]["roi_tracking"]["status"] == "reference"
    assert all(
        spread["roi_tracking"]["tracked"]
        for spread in manifest["spreads"][1:]
    )
    for spread in manifest["spreads"][1:]:
        assert spread["tracked_roi"] == spread["candidates"][0]["tracking_base_roi"]


def test_page_scoped_high_fps_rescan_adds_candidates_without_switching_selection(video, tmp_path):
    project = tmp_path / "rescan-book"
    cfg = Config(
        output_layout="spread",
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=3,
    )
    create_project(video, project, cfg)
    initial = run(project, ROI)

    page = next(page for page in initial["pages"] if page["side"] == "spread")
    spread = next(item for item in initial["spreads"] if item["id"] == page["spread_id"])
    selected_before = spread["selected"]
    candidate_count_before = len(spread["candidates"])

    rescanned = edit(
        project,
        "rescan_candidates",
        page_id=page["id"],
        radius=0.5,
        fps=60,
    )
    updated = next(item for item in rescanned["spreads"] if item["id"] == spread["id"])
    rescan = updated["candidate_rescan"]

    assert rescan["requested_fps"] == 60
    assert rescan["effective_fps"] == pytest.approx(30)
    assert rescan["added"] > 0
    assert len(updated["candidates"]) == candidate_count_before + rescan["added"]
    assert updated["selected"] == selected_before
    assert set(rescan["candidate_ids"]).issubset({item["id"] for item in updated["candidates"]})
    assert all(
        item.get("rescan", {}).get("center_time") == pytest.approx(page["candidate_time"])
        for item in updated["candidates"]
        if item["id"] in rescan["candidate_ids"]
    )
    assert rescanned["pdf_stale"] is True
    refreshed = next(item for item in rescanned["pages"] if item["id"] == page["id"])
    assert (project / refreshed["path"]).is_file()


def test_automatic_high_fps_fallback_adds_candidates_only_for_triggered_spread(
    video,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "auto-rescan-book"
    cfg = Config(
        output_layout="spread",
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=3,
        auto_high_fps_fallback=True,
        auto_high_fps_fallback_fps=60,
    )
    create_project(video, project, cfg)

    calls = 0

    def force_first_spread(records, selection_mode):
        nonlocal calls
        calls += 1
        return ["low_sharpness"] if calls == 1 else []

    monkeypatch.setattr(pipeline, "fallback_reasons", force_first_spread)
    manifest = run(project, ROI)

    first = manifest["spreads"][0]
    info = first["auto_high_fps_fallback"]
    assert info["trigger_reasons"] == ["low_sharpness"]
    assert info["effective_fps"] == pytest.approx(30)
    assert info["added"] > 0
    assert info["candidate_ids"]
    added = [
        record
        for record in first["candidates"]
        if record["id"] in info["candidate_ids"]
    ]
    assert added
    assert all(record["rescan"]["automatic"] is True for record in added)
    assert all(
        record["rescan"]["trigger_reasons"] == ["low_sharpness"]
        for record in added
    )
    assert all(
        "auto_high_fps_fallback" not in spread
        for spread in manifest["spreads"][1:]
    )


def test_missing_page_high_fps_fallback_only_recovers_a_real_stable_run(
    tmp_path,
    monkeypatch,
):
    values = [
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
    samples = [
        pipeline.Sample(index, index / 10, motion, 100.0)
        for index, motion in enumerate(values)
    ]
    segments = [samples[1:6], samples[15:20]]
    recovered_samples = [
        pipeline.Sample(100 + index, 0.90 + index * 0.04, 0.005, 120.0)
        for index in range(6)
    ]

    monkeypatch.setattr(
        pipeline,
        "_high_fps_window_samples",
        lambda manifest, cfg, start, end, requested_fps: (recovered_samples, 25.0),
    )
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        auto_high_fps_fallback=True,
        auto_high_fps_min_stable_seconds=0.18,
    )
    updated, analysis = pipeline._recover_missing_segments_high_fps(
        tmp_path,
        {"analysis_fps": 10},
        cfg,
        samples,
        segments,
        10,
    )

    assert len(updated) == 3
    assert analysis["missing_candidates"] == []
    recovered = analysis["high_fps_fallback"]["recovered_candidates"]
    assert len(recovered) == 1
    assert recovered[0]["reason"] == "high_fps_stable_interval_recovered"
    assert recovered[0]["effective_fps"] == pytest.approx(25.0)
    assert recovered[0]["sample_count"] == 6


def test_cancelled_processing_resumes_after_completed_spread(video, tmp_path, monkeypatch):
    project = tmp_path / "resume-book"
    cfg = Config(
        output_layout="split",
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=2,
    )
    create_project(video, project, cfg)

    original_update = pipeline.update
    requested = False

    def cancel_after_first_spread(current_project, manifest, progress, message):
        nonlocal requested
        original_update(current_project, manifest, progress, message)
        if not requested and message.startswith("候補評価・補正 1 /"):
            requested = True
            request_cancel(current_project)

    monkeypatch.setattr(pipeline, "update", cancel_after_first_spread)
    cancelled = pipeline.run(project, ROI)

    assert cancelled["status"] == "cancelled"
    assert cancelled["processing_checkpoint"]["motion_analysis_complete"] is True
    assert cancelled["processing_checkpoint"]["completed_spreads"] == 1
    assert len(cancelled["spreads"]) == 1
    first_candidate = project / cancelled["spreads"][0]["candidates"][0]["path"]
    first_mtime = first_candidate.stat().st_mtime_ns

    monkeypatch.setattr(pipeline, "update", original_update)
    resumed = pipeline.run(project, ROI)

    assert resumed["status"] == "complete"
    assert len(resumed["spreads"]) == 4
    assert "processing_checkpoint" not in resumed
    assert first_candidate.stat().st_mtime_ns == first_mtime
    assert (project / "output/manga.pdf").is_file()
    assert (project / "output/manga.cbz").is_file()


def test_sampling_and_seeking_use_presentation_time(video):
    metadata = probe(video)
    assert metadata["fps"] == 30
    assert metadata["duration"] == pytest.approx(6, abs=0.1)
    frames = list(sample_frames(video, 10, (240, 160)))
    assert len(frames) == 60
    assert frames[-1][1] == 5.9
    late = list(sample_frames(video, 10, (240, 160), start_time=1.6))
    assert late[0][1] == pytest.approx(1.6)
    assert late[-1][1] == pytest.approx(5.9)
    assert extract_frame(video, 1.8).shape == (320, 480, 3)


def test_batch_candidate_extraction_preserves_requested_order(video):
    frames = extract_frames(video, [1.8, 2.4, 1.8], (240, 160), "none")
    assert len(frames) == 3
    assert all(frame.shape == (160, 240, 3) for frame in frames)
    assert cv2.norm(frames[0], frames[2], cv2.NORM_INF) == 0

    individual = extract_frame(video, 1.8, 240, "none")
    assert individual.shape == frames[0].shape
    assert cv2.norm(frames[0], individual, cv2.NORM_INF) == 0


def test_optional_cover_and_reference_time(video, tmp_path):
    project = tmp_path / "cover-book"
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=3,
        dewarp_mode="auto",
        output_layout="split",
    )
    create_project(video, project, cfg)

    manifest = set_setup_frame(project, "cover", 0.2, confirm=True)
    assert manifest["cover"]["status"] == "ready"
    assert manifest["cover"]["detection"]["detected"] is True
    assert (project / "source/cover_frame.png").is_file()
    detected_roi = manifest["cover"]["roi"]
    manifest = reopen_cover_roi(project)
    assert manifest["cover"]["status"] == "frame_selected"
    assert manifest["cover"]["roi"] == detected_roi
    manifest = set_cover_roi(project, detected_roi)
    assert manifest["cover"]["status"] == "ready"
    assert manifest["cover"]["detection"]["source"] == "manual"

    manifest = set_setup_frame(project, "reference", 1.6, confirm=True)
    assert manifest["reference"]["confirmed"]
    assert (project / "source/reference_frame.png").is_file()

    manifest = run(project, ROI)
    assert len(manifest["spreads"]) == 3
    assert manifest["spreads"][0]["start"] >= 1.6
    assert manifest["pages"][0]["side"] == "cover"
    assert len(manifest["pages"]) == 7
    original_cover = (project / manifest["pages"][0]["path"]).read_bytes()
    corrected = edit(project, "cover_crop", roi=[
        [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8],
    ])
    assert corrected["cover"]["detection"]["source"] == "manual"
    assert corrected["pdf_stale"] is True
    assert corrected["pages"][0]["id"] == "cover"
    assert (project / corrected["pages"][0]["path"]).read_bytes() != original_cover
    spread_pages = [page for page in manifest["pages"] if page["side"] != "cover"]
    assert all(page["dewarp"]["mode"] == "auto" for page in spread_pages)
    assert all("strength_profile" in page["dewarp"] for page in spread_pages)
    assert all((project / page["dewarp"]["before"]).is_file() for page in spread_pages)
    assert all(
        "debug_grid" not in page["dewarp"] or (project / page["dewarp"]["debug_grid"]).is_file()
        for page in spread_pages
    )
    manifest = edit(project, "toggle_dewarp", spread_id="spread_0001", side="right")
    right = next(
        page
        for page in manifest["pages"]
        if page["spread_id"] == "spread_0001" and page["side"] == "right"
    )
    assert right["dewarp"]["status"] == "disabled"
    assert "right" in manifest["spreads"][0]["dewarp_disabled_sides"]
    assert len(PdfReader(project / "output/manga.pdf").pages) == 5


def test_bad_roi_and_missing_model_fail_early(video, tmp_path):
    project = tmp_path / "book"
    create_project(video, project, Config(hand_model=str(tmp_path / "absent.task")))
    with pytest.raises(ValueError, match="Hand model missing"):
        run(project, ROI)
    assert read_manifest(project)["status"] == "ready"


@pytest.mark.parametrize("fps", [60, 120, 240])
def test_high_fps_does_not_increase_analysis_frame_count(video, tmp_path, fps):
    output = tmp_path / f"fps{fps}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-t",
            "1",
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(output),
        ],
        check=True,
    )
    assert probe(output)["fps"] == fps
    samples = list(sample_frames(output, 10, (120, 80)))
    assert len(samples) == 10
    assert samples[-1][1] == 0.9


def test_candidate_review_preview_uses_configured_rotation(video, tmp_path):
    project = tmp_path / "rotated-review"
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        analysis_width=480,
        candidates_per_spread=3,
        candidate_selection_mode="per_page",
        rotation=90,
    )
    create_project(video, project, cfg)
    manifest = run(project, ROI)

    candidate = manifest["spreads"][0]["candidates"][0]
    assert candidate["review_preview"] != candidate["preview"]

    raw = cv2.imread(str(project / candidate["preview"]))
    review = cv2.imread(str(project / candidate["review_preview"]))
    assert raw is not None
    assert review is not None
    expected = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
    assert review.shape == expected.shape
    assert cv2.norm(review, expected, cv2.NORM_INF) == 0


def test_rotation_metadata_shared_by_preview_and_analysis(video, tmp_path):
    output = tmp_path / "rotated.mov"
    help_text = subprocess.run(
        ["ffmpeg", "-h", "full"], capture_output=True, text=True, check=True
    ).stdout
    if "-display_rotation" in help_text:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(video),
            "-c",
            "copy",
            str(output),
        ]
    else:  # Older supported FFmpeg releases use the rotate metadata tag.
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(output),
        ]
    subprocess.run(
        command,
        check=True,
    )
    frame = extract_frame(output)
    assert frame.shape == (480, 320, 3)
    project = tmp_path / "rotated-project"
    manifest = create_project(output, project, Config(hand_backend="none", finger_repair=False))
    assert manifest["metadata"]["display_width"] == 320
    assert manifest["metadata"]["display_height"] == 480
    assert manifest["rotation_detection"]["source"] == "page_geometry"
    assert manifest["rotation_detection"]["confidence"] < 1.0
    assert manifest["rotation_detection"]["display_rotation"] == 90
    assert manifest["rotation_detection"]["metadata_applied"] is True
    preview = cv2.imread(str(project / manifest["reference"]["preview"]))
    assert preview.shape == frame.shape
    samples = sample_frames(output, 10, (160, 240))
    try:
        assert next(samples)[2].shape == (240, 160, 3)
    finally:
        samples.close()


def test_vfr_sampling_and_4k_candidate(video, tmp_path):
    # Nonuniform PTS; selection and seek remain in presentation seconds.
    output = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "select='if(lt(t,3),not(mod(n,2)),not(mod(n,3)))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(output),
        ],
        check=True,
    )
    samples = list(sample_frames(output, 10, (120, 80)))
    assert len(samples) >= 58
    assert extract_frame(output, 4.0, width=240).shape == (160, 240, 3)
    output4k = tmp_path / "4k.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-t",
            "0.2",
            "-vf",
            "scale=3840:2160",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(output4k),
        ],
        check=True,
    )
    assert extract_frame(output4k).shape == (2160, 3840, 3)
    frames = list(sample_frames(output4k, 10, (384, 216)))
    assert len(frames) == 2
    assert frames[0][2].shape == (216, 384, 3)


def test_default_spread_output_pdf_cover_and_manual_add(video, tmp_path):
    project = tmp_path / "whole-book"
    cfg = Config(hand_backend="none", finger_repair=False, analysis_width=480,
                 candidates_per_spread=3, candidate_selection_mode="per_page")
    create_project(video, project, cfg)
    set_setup_frame(project, "cover", 0.2, confirm=True)
    manifest = run(project, ROI)
    assert manifest["config"]["output_layout"] == "spread"
    assert len(manifest["pages"]) == 5  # One cover + four complete spreads.
    assert [p["side"] for p in manifest["pages"]] == ["cover"] + ["spread"] * 4
    assert all("final_quality" in page for page in manifest["pages"])
    assert len(PdfReader(project / "output/manga.pdf").pages) == 4  # One duplicate excluded.
    first = manifest["pages"][1]
    image = cv2.imread(str(project / first["path"]))
    assert image.shape[1] > image.shape[0]
    manifest = edit(project, "toggle_page", page_id=first["id"])
    manifest = edit(project, "select_candidate", spread_id="spread_0001", candidate_id=1)
    assert not manifest["pages"][1]["enabled"]
    manifest = edit(project, "add_frame", time=.8)
    assert len(manifest["pages"]) == 6
    manual = next(s for s in manifest["spreads"] if "manual_frame" in s.get("extra_suspect", []))
    assert [p["side"] for p in manifest["pages"] if p["spread_id"] == manual["id"]] == ["spread"]
    edit(project, "export")
    assert len(PdfReader(project / "output/manga.pdf").pages) == 4
