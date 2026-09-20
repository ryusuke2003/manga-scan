import cv2
import numpy as np

from manga_scan.config import Config
from manga_scan.image_folder import create_image_folder_project, image_files
from manga_scan.storage import read_manifest


def _write_image(path, value):
    image = np.full((90, 70, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_image_folder_uses_natural_filename_order(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    _write_image(folder / "10.jpg", 100)
    _write_image(folder / "2.jpg", 120)
    _write_image(folder / "1.jpg", 140)
    nested = folder / "nested"
    nested.mkdir()
    _write_image(nested / "3.jpg", 160)

    _, files = image_files(folder)

    assert [path.name for path in files] == ["1.jpg", "2.jpg", "10.jpg"]


def test_create_image_folder_project_builds_reviewable_pages(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    _write_image(folder / "page-1.jpg", 80)
    _write_image(folder / "page-2.png", 160)
    project = tmp_path / "project"
    cfg = Config(
        hand_backend="none",
        finger_repair=False,
        auto_rotation=False,
        illumination_correction=False,
        white_normalization=False,
    )

    manifest = create_image_folder_project(
        folder,
        project,
        cfg,
        expected_page_count=3,
    )

    assert manifest["source_type"] == "image_folder"
    assert manifest["status"] == "complete"
    assert [page["external_name"] for page in manifest["pages"]] == [
        "page-1.jpg",
        "page-2.png",
    ]
    assert all(page["source_kind"] == "image_folder" for page in manifest["pages"])
    assert all((project / page["path"]).is_file() for page in manifest["pages"])
    assert all((project / page["source_image"]).is_file() for page in manifest["pages"])
    assert manifest["page_count_check"]["status"] == "short"
    assert manifest["page_count_check"]["difference"] == -1
    assert read_manifest(project)["expected_page_count"] == 3
