import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2

from .manifest_migrations import migrate_manifest


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_manifest(project):
    manifest = json.loads((Path(project) / "manifest.json").read_text())
    return migrate_manifest(manifest)


def save_manifest(project, manifest):
    write_json(Path(project) / "manifest.json", manifest)


@contextmanager
def project_lock(project):
    with (Path(project) / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This project is busy in another process") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def save_image(path, image, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = (
        [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    )
    ok, encoded = cv2.imencode(path.suffix, image, options)
    if not ok:
        raise RuntimeError(f"Image encoding failed: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)
