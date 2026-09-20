#!/usr/bin/env python3
"""Generate original synthetic manga-like panels; no copyrighted test assets."""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def page(seed, w=480, h=320):
    rng = np.random.default_rng(seed)
    image = np.full((h, w, 3), (75, 93, 105), np.uint8)
    image[32:288, 48:432] = 242
    for offset in (48, 240):
        for row in range(2):
            x1, y1 = offset + 12, 44 + row * 117
            cv2.rectangle(image, (x1, y1), (x1 + 166, y1 + 101), (30, 30, 30), 2)
            for _ in range(12):
                x, y = rng.integers(x1 + 7, x1 + 157), rng.integers(y1 + 7, y1 + 92)
                cv2.circle(image, (int(x), int(y)), int(rng.integers(3, 16)), (40, 40, 40), 1)
            cv2.putText(
                image,
                f"{seed}-{row}",
                (x1 + 9, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
    cv2.line(image, (239, 32), (239, 287), (100, 100, 100), 2)
    return image


def make_demo(output, fps=30):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "480x320",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    rng = np.random.default_rng(123)
    try:
        for n, seed in enumerate((1, 2, 2, 3)):
            frame = page(seed)
            for _ in range(round(fps * 1.2)):
                proc.stdin.write(frame.tobytes())
            if n < 3:
                for _ in range(round(fps * 0.4)):
                    moving = rng.integers(0, 255, frame.shape, dtype=np.uint8)
                    proc.stdin.write(moving.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError("Demo video encoding failed")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="projects/demo-input.mp4")
    parser.add_argument("--fps", type=int, choices=(30, 60, 120, 240), default=30)
    args = parser.parse_args()
    print(make_demo(args.output, args.fps))
    print("ROI: [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]")
