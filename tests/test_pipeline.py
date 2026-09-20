import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfReader

from manga_scan.config import Config
from manga_scan.ingest import create_project, set_cover_roi, set_setup_frame
from manga_scan.pipeline import edit, run
from manga_scan.storage import read_manifest
from manga_scan.video import extract_frame, probe, sample_frames

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


def test_end_to_end_dedupe_review_pdf(video, tmp_path):
    project = tmp_path / "book"
    cfg = Config(hand_backend="none", analysis_width=480, candidates_per_spread=3)
    create_project(video, project, cfg)
    manifest = run(project, ROI)
    assert manifest["status"] == "complete"
    assert len(manifest["spreads"]) == 4
    assert len(manifest["pages"]) == 8
    assert sum(p["enabled"] for p in manifest["pages"]) == 6
    assert manifest["spreads"][2]["duplicate_of"] == "spread_0002"
    assert [p["side"] for p in manifest["pages"]][:2] == ["right", "left"]
    assert len(PdfReader(project / "output/manga.pdf").pages) == 6
    assert (project / "debug/motion.csv").is_file()
    assert (project / "debug/scores.csv").is_file()
    assert (project / "debug/contact_sheet.jpg").is_file()
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
    manifest = edit(project, "add_frame", time=0.8)
    assert len(manifest["pages"]) == 10
    assert "manual_frame" in manifest["spreads"][1]["suspect"]
    assert read_manifest(project)["pdf_stale"]
    with pytest.raises(ValueError, match="already processed"):
        run(project, ROI)


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


def test_optional_cover_and_reference_time(video, tmp_path):
    project = tmp_path / "cover-book"
    cfg = Config(
        hand_backend="none",
        analysis_width=480,
        candidates_per_spread=3,
        dewarp_mode="auto",
    )
    create_project(video, project, cfg)

    manifest = set_setup_frame(project, "cover", 0.2, confirm=True)
    assert manifest["cover"]["status"] == "frame_selected"
    assert (project / "source/cover_frame.png").is_file()
    manifest = set_cover_roi(project, ROI)
    assert manifest["cover"]["status"] == "ready"

    manifest = set_setup_frame(project, "reference", 1.6, confirm=True)
    assert manifest["reference"]["confirmed"]
    assert (project / "source/reference_frame.png").is_file()

    manifest = run(project, ROI)
    assert len(manifest["spreads"]) == 3
    assert manifest["spreads"][0]["start"] >= 1.6
    assert manifest["pages"][0]["side"] == "cover"
    assert len(manifest["pages"]) == 7
    spread_pages = [page for page in manifest["pages"] if page["side"] != "cover"]
    assert all(page["dewarp"]["mode"] == "auto" for page in spread_pages)
    assert all((project / page["dewarp"]["before"]).is_file() for page in spread_pages)
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
    manifest = create_project(output, project, Config(hand_backend="none"))
    assert manifest["metadata"]["display_width"] == 320
    assert manifest["metadata"]["display_height"] == 480
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
