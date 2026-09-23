"""FFmpeg subprocess boundary: local files only, display rotation respected."""

import json
import logging
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


def check_tools():
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} not found. On Mac: brew install ffmpeg")


def local_video(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in (".mov", ".mp4", ".ffconcat"):
        raise ValueError("Select a local .mov or .mp4 file")
    return path


def _ffmpeg_input(path):
    path = local_video(path)
    if path.suffix.lower() == ".ffconcat":
        return [
            "-f",
            "concat",
            "-safe",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(path),
        ]
    return ["-protocol_whitelist", "file,pipe", "-i", str(path)]


def _ffprobe_input(path):
    path = local_video(path)
    if path.suffix.lower() == ".ffconcat":
        return [
            "-f",
            "concat",
            "-safe",
            "0",
            "-protocol_whitelist",
            "file,pipe",
        ]
    return ["-protocol_whitelist", "file,pipe"]


def probe(path):
    check_tools()
    path = local_video(path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            *_ffprobe_input(path),
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    raw = json.loads(result.stdout)
    if not raw.get("streams"):
        raise ValueError("No video stream found")
    stream = raw["streams"][0]

    def rate(value):
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            return 0.0

    duration = float(stream.get("duration", raw.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration is missing or invalid")
    transfer = stream.get("color_transfer", "unknown")
    return {
        "path": str(path),
        "width": stream["width"],
        "height": stream["height"],
        "duration": duration,
        "fps": rate(stream.get("avg_frame_rate", "0/1")),
        "nominal_fps": rate(stream.get("r_frame_rate", "0/1")),
        "codec": stream.get("codec_name"),
        "color_transfer": transfer,
        "hdr": transfer in ("smpte2084", "arib-std-b67"),
        "raw": raw,
    }


def input_args(path, hwaccel="none", time=None):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hwaccel != "none":
        args += ["-hwaccel", hwaccel]
    if time is not None:
        if not math.isfinite(time) or time < 0:
            raise ValueError("Timestamp must be finite and >= 0")
        args += ["-ss", f"{time:.8f}"]
    return args + [
        *_ffmpeg_input(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ]


def extract_frame(path, time=0, width=None, hwaccel="none"):
    path = local_video(path)
    args = input_args(path, hwaccel, time)
    if width:
        args += ["-vf", f"scale={int(width)}:-1"]
    args += ["-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"]
    result = subprocess.run(args, capture_output=True, timeout=120)
    if result.returncode or not result.stdout:
        if hwaccel != "none":
            LOG.warning("Hardware decode failed; retrying frame on CPU")
            return extract_frame(path, time, width, "none")
        raise RuntimeError(
            f"Cannot extract frame at {time:.3f}s: {result.stderr.decode(errors='replace')[-2000:]}"
        )
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("FFmpeg returned an invalid frame")
    return frame


def read_exact(stream, size):
    chunks, remaining = [], size
    while remaining:
        part = stream.read(remaining)
        if not part:
            break
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def extract_frames(path, times, size, hwaccel="none"):
    """Extract several presentation timestamps through one FFmpeg decode process.

    Requests are decoded in timestamp order from the earliest requested frame.
    The selected frames are returned in the caller's original order. Hardware
    failures retry once on CPU, and an incomplete batch falls back to the
    existing per-frame seek path for correctness.
    """
    path = local_video(path)
    requested = []
    for value in times:
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamps must be finite and >= 0")
        requested.append(timestamp)
    if not requested:
        return []

    try:
        width, height = (int(size[0]), int(size[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError("size must be a (width, height) pair") from None
    if width <= 0 or height <= 0:
        raise ValueError("size must contain positive dimensions")

    unique_times = sorted(set(requested))
    start_time = unique_times[0]
    offsets = [timestamp - start_time for timestamp in unique_times]
    select_terms = ["isnan(prev_selected_t)"]
    select_terms.extend(
        f"gte(t\\,{offset:.8f})*lt(prev_selected_t\\,{offset:.8f})"
        for offset in offsets[1:]
    )
    filters = (
        "setpts=PTS-STARTPTS,"
        f"select={'+'.join(select_terms)},"
        f"scale={width}:{height}"
    )
    args = input_args(path, hwaccel, start_time if start_time else None)
    args += [
        "-vf",
        filters,
        "-frames:v",
        str(len(unique_times)),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    frame_size = width * height * 3
    frames = []
    code = 0
    message = ""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err)
        try:
            for _ in unique_times:
                data = read_exact(proc.stdout, frame_size)
                if not data:
                    break
                if len(data) != frame_size:
                    break
                frames.append(
                    np.frombuffer(data, np.uint8).reshape(height, width, 3)
                )
            code = proc.wait(timeout=30)
            err.seek(0)
            message = err.read().decode(errors="replace")[-2000:]
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()

    if code or len(frames) != len(unique_times):
        if hwaccel != "none":
            LOG.warning("Hardware batch decode failed; retrying frames on CPU")
            return extract_frames(path, requested, size, "none")

        LOG.warning(
            "Batch frame extraction returned %d/%d frames; falling back to individual seeks: %s",
            len(frames),
            len(unique_times),
            message,
        )
        frames = []
        for timestamp in unique_times:
            frame = extract_frame(path, timestamp, width, "none")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)

    by_time = dict(zip(unique_times, frames))
    return [by_time[timestamp] for timestamp in requested]


def sample_frames(path, fps, size, hwaccel="none", start_time=0.0):
    """Bounded rawvideo pipe. Samples on the presentation timeline, not frame indices.

    Inter-frame codecs still decode intervening frames inside FFmpeg; only sampled,
    scaled frames cross into Python. No full-resolution video array is retained.
    """
    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError("Start time must be finite and >= 0")
    w, h = size
    args = input_args(local_video(path), hwaccel, start_time if start_time else None)
    args += [
        "-vf",
        f"setpts=PTS-STARTPTS,fps=fps={fps}:start_time=0:round=near,scale={w}:{h}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    index = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                data = read_exact(proc.stdout, w * h * 3)
                if not data:
                    break
                if len(data) != w * h * 3:
                    raise RuntimeError("Truncated FFmpeg frame")
                yield index, start_time + index / fps, np.frombuffer(data, np.uint8).reshape(h, w, 3)
                index += 1
            code = proc.wait(timeout=30)
            if code or index == 0:
                err.seek(0)
                message = err.read().decode(errors="replace")[-2000:]
                if hwaccel != "none" and index == 0:
                    LOG.warning("Hardware decode failed; retrying analysis on CPU")
                    yield from sample_frames(path, fps, size, "none", start_time)
                else:
                    raise RuntimeError(f"FFmpeg analysis failed: {message}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
