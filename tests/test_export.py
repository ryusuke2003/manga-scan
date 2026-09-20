import numpy as np
from PIL import Image
from pypdf import PdfReader

from manga_scan.export import export_pdf


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
