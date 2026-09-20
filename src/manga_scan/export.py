import math
import re
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw
from reportlab.pdfgen.canvas import Canvas


BOOK_METADATA_FIELDS = ("title", "author", "series", "volume", "publisher", "language")


def normalize_book_metadata(metadata):
    """Validate and normalize optional book metadata stored in the manifest."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("book metadata must be an object")
    unknown = set(metadata) - set(BOOK_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"Unknown book metadata fields: {sorted(unknown)}")

    normalized = {}
    for field in BOOK_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"book metadata {field} must be a string")
        value = " ".join(value.split())
        if any(ord(char) < 32 for char in value):
            raise ValueError(f"book metadata {field} contains control characters")
        if not value:
            continue
        limit = 32 if field == "language" else (64 if field == "volume" else 200)
        if len(value) > limit:
            raise ValueError(f"book metadata {field} is too long")
        normalized[field] = value
    return normalized


def metadata_output_stem(metadata):
    """Return a portable file stem while keeping Japanese titles readable."""
    metadata = normalize_book_metadata(metadata)
    stem = metadata.get("title") or metadata.get("series") or "manga"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    # Keep room for ".next.cbz" and filesystem bookkeeping. macOS filenames
    # are byte-limited, so truncating by Unicode code points is not sufficient
    # for Japanese titles or emoji.
    encoded = stem.encode("utf-8")
    if len(encoded) > 180:
        stem = encoded[:180].decode("utf-8", errors="ignore").rstrip(" .")
    return stem or "manga"


def comicinfo_xml(metadata, page_count):
    """Build ComicInfo.xml for CBZ readers without adding a dependency."""
    metadata = normalize_book_metadata(metadata)
    root = ET.Element("ComicInfo")
    mapping = (
        ("title", "Title"),
        ("series", "Series"),
        ("volume", "Number"),
        ("author", "Writer"),
        ("publisher", "Publisher"),
        ("language", "LanguageISO"),
    )
    for field, tag in mapping:
        if field in metadata:
            ET.SubElement(root, tag).text = metadata[field]
    ET.SubElement(root, "PageCount").text = str(int(page_count))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def export_pdf(paths, output, dpi=300, image_format="png", jpeg_quality=92, metadata=None):
    """PNG pixels are lossless Flate; JPEG inputs are embedded without recompression.

    'jpeg' converts lossless inputs once at the requested quality. Existing JPEG
    files are always passed through, avoiding a second generation of artifacts.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("No enabled pages to export")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pdf")
    try:
        with tempfile.TemporaryDirectory(prefix="manga-pdf-") as work:
            canvas = Canvas(str(temporary), pageCompression=1)
            metadata = normalize_book_metadata(metadata)
            canvas.setTitle(metadata.get("title") or "Manga Scan")
            if metadata.get("author"):
                canvas.setAuthor(metadata["author"])
            subject = " / ".join(
                part
                for part in (
                    metadata.get("series"),
                    f'Vol. {metadata["volume"]}' if metadata.get("volume") else None,
                    metadata.get("publisher"),
                )
                if part
            )
            if subject:
                canvas.setSubject(subject)
            if metadata.get("language"):
                canvas.setKeywords(f'language:{metadata["language"]}')
            canvas.setCreator("manga-scan-local (no OCR)")
            for i, path in enumerate(paths):
                path = Path(path)
                with Image.open(path) as im:
                    w, h = im.size
                    if image_format == "jpeg" and im.format != "JPEG":
                        jpeg = Path(work) / f"{i}.jpg"
                        im.convert("RGB").save(jpeg, quality=jpeg_quality, subsampling=0)
                        path = jpeg
                size = (w * 72 / dpi, h * 72 / dpi)
                canvas.setPageSize(size)
                canvas.drawImage(str(path), 0, 0, width=size[0], height=size[1])
                canvas.showPage()
            canvas.save()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def export_cbz(paths, output, metadata=None):
    """Store rendered page bytes in manifest order without recompression."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("No enabled pages to export")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.cbz")
    digits = max(3, len(str(len(paths))))
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for index, path in enumerate(paths, 1):
                suffix = path.suffix.lower()
                if suffix not in (".png", ".jpg", ".jpeg"):
                    raise ValueError(f"Unsupported CBZ page format: {path}")
                archive.write(
                    path,
                    arcname=f"{index:0{digits}d}{suffix}",
                    compress_type=zipfile.ZIP_STORED,
                )
            metadata = normalize_book_metadata(metadata)
            if metadata:
                archive.writestr(
                    "ComicInfo.xml",
                    comicinfo_xml(metadata, len(paths)),
                    compress_type=zipfile.ZIP_STORED,
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def contact_sheets(project, pages, per_sheet=80):
    """Paginated so a very long book cannot allocate an unbounded tall bitmap."""
    project = Path(project)
    debug = project / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pages), per_sheet):
        batch = pages[offset : offset + per_sheet]
        sheet = Image.new("RGB", (5 * 180, math.ceil(len(batch) / 5) * 260), "#e8e5df")
        draw = ImageDraw.Draw(sheet)
        for i, page in enumerate(batch):
            with Image.open(project / page["path"]) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((164, 218))
                x, y = (i % 5) * 180 + 8, (i // 5) * 260 + 8
                sheet.paste(thumb, (x, y))
            color = "#bd3826" if page.get("suspect") else "#283e37"
            draw.text((x, y + 224), f"{offset + i + 1}  {page['side']}", fill=color)
            if not page.get("enabled", True):
                draw.text((x, y + 238), "EXCLUDED", fill="#bd3826")
        name = (
            "contact_sheet.jpg"
            if offset == 0
            else f"contact_sheet_{offset // per_sheet + 1:03d}.jpg"
        )
        sheet.save(debug / name, quality=90)
