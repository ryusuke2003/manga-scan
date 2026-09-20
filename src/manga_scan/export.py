import math
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.pdfgen.canvas import Canvas


def export_pdf(paths, output, dpi=300, image_format="png", jpeg_quality=92):
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
            canvas.setTitle("Manga Scan")
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
