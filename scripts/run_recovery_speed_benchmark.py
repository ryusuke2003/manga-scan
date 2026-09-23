#!/usr/bin/env python3
"""Compare legacy and grouped high-fps recovery on a short real-video window."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga_scan.config import Config  # noqa: E402
from manga_scan.motion import Sample, StableDetector, motion_score  # noqa: E402
from manga_scan.perspective import warp_roi  # noqa: E402
from manga_scan.pipeline import _recover_missing_segments_high_fps  # noqa: E402
from manga_scan.score import sharpness  # noqa: E402
from manga_scan.video import probe, sample_frames  # noqa: E402


def _scan_window(path, roi, cfg, start, end, metadata):
    fps = min(cfg.video_sample_fps, metadata["fps"])
    width = min(cfg.analysis_width, metadata["width"])
    size = (width, max(2, round(metadata["height"] * width / metadata["width"])))
    machine = StableDetector(cfg.stable_frames, cfg.motion_threshold, cfg.turn_threshold)
    samples, segments, previews = [], [], {}
    previous = None
    stream = sample_frames(path, fps, size, cfg.hwaccel, start_time=start)
    try:
        for index, timestamp, frame in stream:
            if timestamp > end:
                break
            cropped = warp_roi(frame, roi)
            sample = Sample(index, timestamp,
                            motion_score(previous, cropped) if previous is not None else 1.0,
                            sharpness(cropped))
            samples.append(sample)
            previews[index] = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
                                         (64, 64), interpolation=cv2.INTER_AREA)
            complete = machine.push(sample)
            if complete:
                segments.append(complete)
            previous = cropped
    finally:
        stream.close()
    tail = machine.finish()
    if tail:
        segments.append(tail)
    return fps, samples, segments, previews


def _anchor_hits(anchors, segments, merged=()):
    return {item["id"]: (
                any(segment[0].time - 0.15 <= item["time"]
                    <= segment[-1].time + 0.15 for segment in segments)
                or any(window["start"] - 0.15 <= item["time"]
                       <= window["end"] + 0.15 for window in merged)
            )
            for item in anchors}


def _strict_failures(report):
    failures = []
    if report["legacy_recoveries_lost"]:
        failures.append("legacy recovery windows were lost")
    missed = [key for key, found in report["annotated_page_coverage_grouped"].items()
              if not found]
    if missed:
        failures.append("annotated spreads were missed: " + ", ".join(missed))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="desk-portrait-6479")
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--order", choices=("legacy-first", "grouped-first"),
                        default="legacy-first")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    cases = json.loads((ROOT / "benchmarks/real/cases.json").read_text())
    case = next((item for item in cases["videos"] if item["id"] == args.video_id), None)
    if case is None:
        parser.error(f"unknown video id: {args.video_id}")
    path = ROOT / "benchmarks/real/videos" / case["filename"]
    if not path.is_file():
        parser.error(f"local benchmark video missing: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != case["sha256"]:
        parser.error(f"benchmark video SHA-256 mismatch: {path}")
    end = min(args.start + args.duration, float(case["duration"]))
    if not 0 <= args.start < end:
        parser.error("start/duration must select a nonempty window within the video")
    anchors = [item for item in case["samples"]
               if item["expect_spread"] and args.start <= item["time"] <= end]
    if not anchors:
        parser.error("selected window contains no annotated spread anchors")
    roi = anchors[0]["spread_quad"]
    cfg = Config()
    metadata = probe(path)
    analysis_started = time.monotonic()
    fps, samples, segments, previews = _scan_window(path, roi, cfg, args.start, end, metadata)
    analysis_seconds = time.monotonic() - analysis_started
    manifest = {"source": str(path), "roi": roi, "analysis_fps": fps,
                "metadata": {"fps": metadata["fps"], "display_width": metadata["width"],
                             "display_height": metadata["height"]}}
    def run_variant(grouped):
        started = time.monotonic()
        result = _recover_missing_segments_high_fps(
            ROOT, manifest, cfg, samples, list(segments), fps,
            frame_previews=previews if grouped else None,
            use_batch=grouped,
        )
        return result, time.monotonic() - started

    if args.order == "legacy-first":
        (legacy_segments, legacy), legacy_seconds = run_variant(False)
        (new_segments, current), grouped_seconds = run_variant(True)
    else:
        (new_segments, current), grouped_seconds = run_variant(True)
        (legacy_segments, legacy), legacy_seconds = run_variant(False)
    old_ids = {item["id"] for item in legacy["high_fps_fallback"]["recovered_candidates"]}
    new_info = current["high_fps_fallback"]
    retained_ids = {item["id"] for item in new_info["recovered_candidates"]}
    retained_ids.update(item["id"] for item in new_info["merged_windows"])
    old_hits = _anchor_hits(anchors, legacy_segments)
    new_hits = _anchor_hits(anchors, new_segments, new_info["merged_windows"])
    report = {
        "video_id": args.video_id, "start": args.start, "end": end,
        "order": args.order,
        "original_intervals": len(segments),
        "lowres_analysis_seconds": round(analysis_seconds, 3),
        "legacy_seconds": round(legacy_seconds, 3),
        "grouped_seconds": round(grouped_seconds, 3),
        "speedup": round(legacy_seconds / max(grouped_seconds, 0.001), 2),
        "legacy_analysis_total_seconds": round(analysis_seconds + legacy_seconds, 3),
        "grouped_analysis_total_seconds": round(analysis_seconds + grouped_seconds, 3),
        "legacy_recovered": len(old_ids),
        "legacy_decode_groups": legacy["high_fps_fallback"]["decode_groups"],
        "grouped_decode_groups": new_info["decode_groups"],
        "grouped_recovered": len(new_info["recovered_candidates"]),
        "grouped_merged": len(new_info["merged_windows"]),
        "legacy_recoveries_lost": sorted(old_ids - retained_ids),
        "annotated_page_coverage_legacy": old_hits,
        "annotated_page_coverage_grouped": new_hits,
    }
    report["strict_failures"] = _strict_failures(report)
    output = ROOT / "benchmarks/real/reports/recovery_speed_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output.write_text(report_text)
    case_output = output.with_name(
        f"recovery_speed_{args.video_id}_{args.start:g}_{end:g}_{args.order}.json"
    )
    case_output.write_text(report_text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report["strict_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
