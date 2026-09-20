from pathlib import Path

import cv2
import numpy as np

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.ingest import _prepare_video_source
from manga_scan.storage import read_manifest, save_image, save_manifest


def _project_manifest(project):
    cfg = Config(hand_backend="none", finger_repair=False)
    (project / "pages").mkdir(parents=True, exist_ok=True)
    (project / "source").mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 3,
        "source": "/tmp/book.mp4",
        "metadata": {"duration": 10, "display_width": 100, "display_height": 120, "fps": 30},
        "config": cfg.to_dict(),
        "status": "complete",
        "progress": 1,
        "message": "done",
        "warnings": [],
        "spreads": [],
        "pages": [],
        "pdf_stale": False,
    }
    save_manifest(project, manifest)
    return manifest


def test_prepare_video_source_builds_ordered_concat_manifest(tmp_path):
    first = tmp_path / "part-1.mp4"
    second = tmp_path / "part-2.mov"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    project = tmp_path / "project"

    source, sources = _prepare_video_source([first, second], project, copy_source=False)

    concat = Path(source)
    assert concat.name == "input.ffconcat"
    assert sources == [str(first.resolve()), str(second.resolve())]
    lines = concat.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert str(first.resolve()) in lines[1]
    assert str(second.resolve()) in lines[2]


def test_prepare_video_source_keeps_single_video_direct(tmp_path):
    video = tmp_path / "book.mp4"
    video.write_bytes(b"video")

    source, sources = _prepare_video_source(video, tmp_path / "project", copy_source=False)

    assert source == str(video.resolve())
    assert sources == [str(video.resolve())]


def test_external_page_can_be_added_replaced_and_undone(tmp_path, monkeypatch):
    project = tmp_path / "project"
    original = np.full((80, 60, 3), 210, dtype=np.uint8)
    replacement = np.zeros((90, 70, 3), dtype=np.uint8)
    external = tmp_path / "phone.jpg"
    assert cv2.imwrite(str(external), replacement)

    manifest = _project_manifest(project)
    save_image(project / "pages/original.png", original)
    manifest["pages"] = [{
        "id": "page_0001",
        "spread_id": "spread_0001",
        "side": "spread",
        "enabled": True,
        "suspect": [],
        "path": "pages/original.png",
    }]
    save_manifest(project, manifest)
    monkeypatch.setattr(pipeline, "_refresh_adjacent_final_quality", lambda *_args, **_kwargs: None)

    added = pipeline.import_external_page(project, external)
    assert len(added["pages"]) == 2
    assert added["pages"][-1]["source"] == "external_image"
    assert (project / added["pages"][-1]["path"]).is_file()
    assert added["pdf_stale"] is True

    pipeline.edit(project, "undo_page_edit")
    restored = read_manifest(project)
    assert [page["id"] for page in restored["pages"]] == ["page_0001"]

    replaced = pipeline.import_external_page(project, external, "page_0001")
    assert len(replaced["pages"]) == 1
    assert replaced["pages"][0]["id"] == "page_0001"
    assert replaced["pages"][0]["source"] == "external_image"
    assert replaced["pages"][0]["external_name"] == "phone.jpg"

    pipeline.edit(project, "undo_page_edit")
    restored = read_manifest(project)
    assert restored["pages"][0]["path"] == "pages/original.png"
    assert restored["pages"][0].get("source") != "external_image"
