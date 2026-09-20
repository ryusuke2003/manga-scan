import numpy as np
from PIL import Image
from pypdf import PdfReader

from manga_scan.config import Config
from manga_scan.export import export_pdf
from manga_scan.pipeline import edit
from manga_scan.storage import read_manifest, save_manifest


def test_png_pdf_preserves_pixels_and_page_ratio(tmp_path):
    array = np.random.default_rng(1).integers(0, 256, (80, 51, 3), dtype=np.uint8)
    path = tmp_path / "page.png"
    Image.fromarray(array).save(path)
    export_pdf([path, path], tmp_path / "book.pdf", dpi=300)
    reader = PdfReader(tmp_path / "book.pdf")
    assert len(reader.pages) == 2
    page = reader.pages[0]
    assert abs(float(page.mediabox.width / page.mediabox.height) - 51 / 80) < 1e-6
    embedded = list(page.images)[0].image.convert("RGB")
    np.testing.assert_array_equal(np.array(embedded), array)
    assert page.extract_text() == ""


def test_jpeg_embedded_without_second_recompression(tmp_path):
    path = tmp_path / "page.jpg"
    Image.new("RGB", (81, 120), "#123456").save(path, quality=87)
    export_pdf([path], tmp_path / "book.pdf", image_format="jpeg", jpeg_quality=20)
    embedded = list(PdfReader(tmp_path / "book.pdf").pages[0].images)[0]
    assert embedded.data == path.read_bytes()


def test_export_fails_without_destroying_previous_pdf(tmp_path):
    output = tmp_path / "old.pdf"
    output.write_bytes(b"previous")
    import pytest

    with pytest.raises(Exception):
        export_pdf([tmp_path / "missing.png"], output)
    assert output.read_bytes() == b"previous"


def test_review_export_persists_completion_message(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (40, 60), "white").save(pages / "page.png")
    config = Config(hand_backend="none", finger_repair=False).to_dict()
    manifest = {
        "source": "/tmp/book.mp4",
        "status": "complete",
        "config": config,
        "pages": [
            {
                "id": "page-1",
                "spread_id": "spread-1",
                "side": "spread",
                "path": "pages/page.png",
                "enabled": True,
                "suspect": [],
            }
        ],
        "spreads": [],
        "pdf_stale": True,
        "progress": 1,
        "message": "完了",
    }
    save_manifest(tmp_path, manifest)

    result = edit(tmp_path, "export")
    persisted = read_manifest(tmp_path)

    assert result["pdf_stale"] is False
    assert result["pdf"] == "output/manga.pdf"
    assert result["message"] == "PDFを出力しました"
    assert persisted["message"] == "PDFを出力しました"
    assert (tmp_path / "output/manga.pdf").is_file()
