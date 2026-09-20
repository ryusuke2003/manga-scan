import zipfile

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

import manga_scan.pipeline as pipeline
from manga_scan.config import Config
from manga_scan.export import export_cbz, export_pdf, metadata_output_stem
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


def test_pdf_embeds_book_metadata(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(path)
    metadata = {
        "title": "テスト漫画 1",
        "author": "漫画 太郎",
        "series": "テスト漫画",
        "volume": "1",
        "publisher": "Example Press",
        "language": "ja",
    }

    export_pdf([path], tmp_path / "book.pdf", metadata=metadata)

    info = PdfReader(tmp_path / "book.pdf").metadata
    assert info.title == "テスト漫画 1"
    assert info.author == "漫画 太郎"
    assert "テスト漫画" in info.subject
    assert "Vol. 1" in info.subject
    assert "Example Press" in info.subject
    assert "language:ja" in info.get("/Keywords", "")


def test_cbz_writes_comicinfo_when_metadata_exists(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(path)
    metadata = {
        "title": "テスト漫画 1",
        "author": "漫画 太郎",
        "series": "テスト漫画",
        "volume": "1",
        "publisher": "Example Press",
        "language": "ja",
    }

    output = export_cbz([path], tmp_path / "book.cbz", metadata)

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["001.png", "ComicInfo.xml"]
        xml = archive.read("ComicInfo.xml").decode()
        assert "<Title>テスト漫画 1</Title>" in xml
        assert "<Writer>漫画 太郎</Writer>" in xml
        assert "<Series>テスト漫画</Series>" in xml
        assert "<Number>1</Number>" in xml
        assert "<Publisher>Example Press</Publisher>" in xml
        assert "<LanguageISO>ja</LanguageISO>" in xml
        assert "<PageCount>1</PageCount>" in xml


def test_metadata_output_stem_is_safe_and_readable():
    assert metadata_output_stem({"title": " 漫画 / 第1巻 "}) == "漫画 _ 第1巻"
    assert metadata_output_stem({}) == "manga"


def test_jpeg_embedded_without_second_recompression(tmp_path):
    path = tmp_path / "page.jpg"
    Image.new("RGB", (81, 120), "#123456").save(path, quality=87)
    export_pdf([path], tmp_path / "book.pdf", image_format="jpeg", jpeg_quality=20)
    embedded = list(PdfReader(tmp_path / "book.pdf").pages[0].images)[0]
    assert embedded.data == path.read_bytes()


def test_cbz_preserves_page_bytes_order_and_uses_store_mode(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (31, 47), "#123456").save(first)
    Image.new("RGB", (29, 43), "#abcdef").save(second, quality=83)

    output = export_cbz([second, first], tmp_path / "book.cbz")

    with zipfile.ZipFile(output) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == ["001.jpg", "002.png"]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
        assert archive.read("001.jpg") == second.read_bytes()
        assert archive.read("002.png") == first.read_bytes()


def test_cbz_export_fails_without_destroying_previous_archive(tmp_path):
    output = tmp_path / "old.cbz"
    output.write_bytes(b"previous")
    with pytest.raises(Exception):
        export_cbz([tmp_path / "missing.png"], output)
    assert output.read_bytes() == b"previous"


def test_export_fails_without_destroying_previous_pdf(tmp_path):
    output = tmp_path / "old.pdf"
    output.write_bytes(b"previous")
    with pytest.raises(Exception):
        export_pdf([tmp_path / "missing.png"], output)
    assert output.read_bytes() == b"previous"


def test_combined_export_failure_keeps_previous_pdf_and_cbz(tmp_path, monkeypatch):
    pages = tmp_path / "pages"
    output = tmp_path / "output"
    pages.mkdir()
    output.mkdir()
    Image.new("RGB", (40, 60), "white").save(pages / "page.png")
    (output / "manga.pdf").write_bytes(b"previous-pdf")
    (output / "manga.cbz").write_bytes(b"previous-cbz")

    config = Config(hand_backend="none", finger_repair=False).to_dict()
    save_manifest(
        tmp_path,
        {
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
            "pdf": "output/manga.pdf",
            "cbz": "output/manga.cbz",
            "pdf_stale": True,
            "progress": 1,
            "message": "完了",
        },
    )

    monkeypatch.setattr(
        pipeline,
        "export_cbz",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cbz failed")),
    )

    with pytest.raises(RuntimeError, match="cbz failed"):
        edit(tmp_path, "export")

    assert (output / "manga.pdf").read_bytes() == b"previous-pdf"
    assert (output / "manga.cbz").read_bytes() == b"previous-cbz"
    persisted = read_manifest(tmp_path)
    assert persisted["pdf_stale"] is True
    assert persisted["message"] == "PDF / CBZの出力に失敗しました"


def test_metadata_edit_marks_exports_stale_and_uses_title_for_next_export(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (40, 60), "white").save(pages / "page.png")
    config = Config(hand_backend="none", finger_repair=False).to_dict()
    save_manifest(
        tmp_path,
        {
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
            "pdf_stale": False,
            "pdf": "output/manga.pdf",
            "cbz": "output/manga.cbz",
            "progress": 1,
            "message": "完了",
        },
    )

    edited = edit(
        tmp_path,
        "book_metadata",
        metadata={"title": "私の漫画 / 1", "author": "作者", "language": "ja"},
    )
    assert edited["book_metadata"]["title"] == "私の漫画 / 1"
    assert edited["pdf_stale"] is True

    exported = edit(tmp_path, "export")
    assert exported["pdf"] == "output/私の漫画 _ 1.pdf"
    assert exported["cbz"] == "output/私の漫画 _ 1.cbz"
    assert (tmp_path / exported["pdf"]).is_file()
    assert (tmp_path / exported["cbz"]).is_file()


def test_export_cleanup_never_deletes_paths_outside_output(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (40, 60), "white").save(pages / "page.png")
    outside_pdf = tmp_path.parent / f"{tmp_path.name}-outside.pdf"
    outside_cbz = tmp_path.parent / f"{tmp_path.name}-outside.cbz"
    outside_pdf.write_bytes(b"keep-pdf")
    outside_cbz.write_bytes(b"keep-cbz")
    config = Config(hand_backend="none", finger_repair=False).to_dict()
    save_manifest(
        tmp_path,
        {
            "source": "/tmp/book.mp4",
            "status": "complete",
            "config": config,
            "book_metadata": {"title": "安全な出力"},
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
            "pdf": f"../{outside_pdf.name}",
            "cbz": str(outside_cbz),
            "progress": 1,
            "message": "完了",
        },
    )

    try:
        exported = edit(tmp_path, "export")
        assert exported["pdf"] == "output/安全な出力.pdf"
        assert exported["cbz"] == "output/安全な出力.cbz"
        assert outside_pdf.read_bytes() == b"keep-pdf"
        assert outside_cbz.read_bytes() == b"keep-cbz"
    finally:
        outside_pdf.unlink(missing_ok=True)
        outside_cbz.unlink(missing_ok=True)


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
    assert result["cbz"] == "output/manga.cbz"
    assert result["message"] == "PDF / CBZを出力しました"
    assert persisted["message"] == "PDF / CBZを出力しました"
    assert (tmp_path / "output/manga.pdf").is_file()
    assert (tmp_path / "output/manga.cbz").is_file()
