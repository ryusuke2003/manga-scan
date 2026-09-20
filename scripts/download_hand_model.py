#!/usr/bin/env python3
"""Explicit setup-only download; the scanner itself never downloads anything."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main():
    parser = argparse.ArgumentParser(
        description="Download the official MediaPipe hand model during setup"
    )
    parser.add_argument("--output", default="models/hand_landmarker.task")
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="Expected model digest")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise SystemExit(f"Already exists: {destination}; move it first to download again")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024 or len(data) < 1024:
        raise SystemExit("Unexpected model size")
    digest = hashlib.sha256(data).hexdigest()
    if args.sha256 and digest != args.sha256:
        raise SystemExit("SHA256 mismatch")
    # A .task bundle is a ZIP containing TFLite models.
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if not any(n.endswith(".tflite") for n in archive.namelist()):
            raise SystemExit("Invalid model bundle")
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    destination.with_suffix(".json").write_text(
        json.dumps({"url": URL, "sha256": digest}, indent=2) + "\n"
    )
    print(f"Saved {destination}\nSHA256 {digest}")


if __name__ == "__main__":
    main()
