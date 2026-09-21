#!/usr/bin/env python3
"""Run local-only real-camera benchmark cases without committing manga media."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_ROTATIONS = {0, 90, 180, 270}
_DEFAULT_CONSENSUS_OFFSETS = (-0.5, 0.0, 0.5)
_CONSENSUS_MAX_CORNER_DEVIATION = 0.04


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polygon_iou(a, b) -> float:
    qa = np.asarray(a, dtype=np.float32)
    qb = np.asarray(b, dtype=np.float32)
    if qa.shape != (4, 2) or qb.shape != (4, 2):
        raise ValueError("polygon_iou expects two 4x2 quads")
    area_a = abs(float(cv2.contourArea(qa)))
    area_b = abs(float(cv2.contourArea(qb)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(qa, qb)
    union = area_a + area_b - float(intersection)
    return 0.0 if union <= 0 else float(np.clip(intersection / union, 0.0, 1.0))


def _validate_quad(quad, label: str) -> None:
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{label}: expected four finite [x,y] pairs")
    if (points < 0).any() or (points > 1).any():
        raise ValueError(f"{label}: coordinates must be normalized to 0..1")
    if abs(float(cv2.contourArea(points))) <= 1e-5:
        raise ValueError(f"{label}: quad area is too small")
    first = np.roll(points, -1, axis=0) - points
    second = np.roll(first, -1, axis=0)
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if np.any(cross <= 0):
        raise ValueError(f"{label}: quad must be convex TL,TR,BR,BL")


def validate_manifest(data: dict) -> None:
    if data.get("version") != 1:
        raise ValueError("real benchmark manifest version must be 1")
    videos = data.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("real benchmark manifest needs a non-empty videos array")
    seen_video_ids = set()
    for video in videos:
        video_id = video.get("id")
        if not isinstance(video_id, str) or not video_id or video_id in seen_video_ids:
            raise ValueError("video ids must be unique non-empty strings")
        seen_video_ids.add(video_id)
        digest = video.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{video_id}: sha256 must be 64 lowercase hex characters")
        if video.get("expected_rotation") not in _ROTATIONS:
            raise ValueError(f"{video_id}: expected_rotation must be 0/90/180/270")
        duration = float(video.get("duration", 0))
        if duration <= 0:
            raise ValueError(f"{video_id}: duration must be positive")
        sample_ids = set()
        for sample in video.get("samples", []):
            sample_id = sample.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"{video_id}: sample ids must be unique non-empty strings")
            sample_ids.add(sample_id)
            timestamp = float(sample.get("time", -1))
            if not 0 <= timestamp < duration:
                raise ValueError(f"{video_id}/{sample_id}: time must be within the video")
            expect_spread = sample.get("expect_spread")
            if type(expect_spread) is not bool:
                raise ValueError(f"{video_id}/{sample_id}: expect_spread must be boolean")
            if expect_spread:
                if "spread_quad" not in sample:
                    raise ValueError(f"{video_id}/{sample_id}: positive case needs spread_quad")
                _validate_quad(sample["spread_quad"], f"{video_id}/{sample_id}.spread_quad")
                for key in ("min_reference_iou", "min_page_iou"):
                    value = float(sample.get(key, 0))
                    if not 0 <= value <= 1:
                        raise ValueError(f"{video_id}/{sample_id}: {key} must be 0..1")
                offsets = sample.get("consensus_offsets", _DEFAULT_CONSENSUS_OFFSETS)
                if not isinstance(offsets, (list, tuple)) or len(offsets) < 3:
                    raise ValueError(
                        f"{video_id}/{sample_id}: consensus_offsets needs at least 3 values"
                    )
                try:
                    offsets = [float(value) for value in offsets]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{video_id}/{sample_id}: consensus_offsets must be numeric"
                    ) from exc
                if not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets):
                    raise ValueError(
                        f"{video_id}/{sample_id}: consensus_offsets must be finite and unique"
                    )
                if not any(abs(value) <= 1e-9 for value in offsets):
                    raise ValueError(
                        f"{video_id}/{sample_id}: consensus_offsets must include 0"
                    )
                if any(not 0 <= timestamp + value < duration for value in offsets):
                    raise ValueError(
                        f"{video_id}/{sample_id}: consensus frame must be within the video"
                    )
        repair_ids = set()
        for pair in video.get("repair_pairs", []):
            pair_id = pair.get("id")
            if not isinstance(pair_id, str) or not pair_id or pair_id in repair_ids:
                raise ValueError(f"{video_id}: repair pair ids must be unique non-empty strings")
            repair_ids.add(pair_id)
            _validate_quad(pair.get("spread_quad"), f"{video_id}/{pair_id}.spread_quad")
            target = float(pair.get("target_time", -1))
            donors = pair.get("donor_times")
            if not 0 <= target < duration or not isinstance(donors, list) or not donors:
                raise ValueError(f"{video_id}/{pair_id}: invalid target/donor times")
            if any(not 0 <= float(value) < duration for value in donors):
                raise ValueError(f"{video_id}/{pair_id}: donor time must be within the video")


def _find_video(directory: Path, spec: dict, hash_cache: dict[Path, str]) -> Path:
    expected = spec["sha256"]
    direct = directory / spec["filename"]
    candidates = [direct] if direct.is_file() else []
    candidates += [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in (".mp4", ".mov") and path != direct
    ]
    for path in candidates:
        digest = hash_cache.setdefault(path, sha256_file(path))
        if digest == expected:
            return path
    raise FileNotFoundError(
        f"Missing media for {spec['id']}: place {spec['filename']} "
        f"(sha256 {expected}) in {directory}"
    )


def _draw_quad(image, quad, color, label):
    if quad is None:
        return
    h, w = image.shape[:2]
    points = np.asarray(quad, np.float32) * [max(w - 1, 1), max(h - 1, 1)]
    points = np.rint(points).astype(np.int32)
    cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
    x, y = points[0]
    cv2.putText(
        image,
        label,
        (int(x), max(18, int(y) - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _sample_result(video_id, sample, displayed):
    from manga_scan.page_contour import detect_page_quads, spread_quad_from_page_quads
    from manga_scan.score import sharpness
    from manga_scan.spread_detect import detect_reference_spread

    result = {
        "video_id": video_id,
        "sample_id": sample["id"],
        "time": sample["time"],
        "tags": sample.get("tags", []),
        "sharpness": round(sharpness(displayed), 3),
        "checks": {},
    }
    expected_spread = bool(sample["expect_spread"])
    reference = detect_reference_spread(displayed, min_confidence=0.55)
    reference_check = {
        "expected": expected_spread,
        "detected": bool(reference["detected"]),
        "confidence": reference["confidence"],
        "stage": reference.get("stage"),
        "source": reference.get("source"),
    }
    if not expected_spread:
        reference_check["passed"] = not reference["detected"]
        result["checks"]["reference_spread"] = reference_check
        return result, reference.get("roi"), None

    expected_quad = sample["spread_quad"]
    reference_iou = polygon_iou(reference["roi"], expected_quad) if reference.get("roi") else 0.0
    reference_check.update(
        iou=round(reference_iou, 4),
        min_iou=sample["min_reference_iou"],
        passed=bool(reference["detected"] and reference_iou >= sample["min_reference_iou"]),
    )
    result["checks"]["reference_spread"] = reference_check

    page = detect_page_quads(
        displayed,
        expected_quad,
        spine_ratio=0.5,
        min_confidence=0.55,
    )
    page_quad = None
    if page["detected"]:
        try:
            page_quad = spread_quad_from_page_quads(page)
        except ValueError:
            page_quad = None
    page_iou = polygon_iou(page_quad, expected_quad) if page_quad is not None else 0.0
    result["checks"]["page_contour"] = {
        "detected": bool(page["detected"]),
        "confidence": page["confidence"],
        "left_confidence": page["left"]["confidence"],
        "right_confidence": page["right"]["confidence"],
        "iou": round(page_iou, 4),
        "min_iou": sample["min_page_iou"],
        "passed": bool(page["detected"] and page_iou >= sample["min_page_iou"]),
    }
    return result, reference.get("roi"), page_quad


def _page_consensus_result(video_spec, sample, path):
    from manga_scan.page_contour import (
        consensus_page_quads,
        detect_page_quads,
        spread_quad_from_page_quads,
    )
    from manga_scan.split import rotate_image
    from manga_scan.temporal_alignment import (
        align_page_detection,
        alignment_summary,
        estimate_frame_alignments,
    )
    from manga_scan.video import extract_frame

    offsets = [
        float(value)
        for value in sample.get("consensus_offsets", _DEFAULT_CONSENSUS_OFFSETS)
    ]
    detections = []
    displayed_frames = []
    times = []
    anchor_id = None
    for candidate_id, offset in enumerate(offsets):
        timestamp = float(sample["time"]) + offset
        source = extract_frame(path, timestamp)
        displayed = rotate_image(source, video_spec["expected_rotation"])
        displayed_frames.append(displayed)
        detection = detect_page_quads(
            displayed,
            sample["spread_quad"],
            spine_ratio=0.5,
            min_confidence=0.55,
        )
        detection["candidate_id"] = candidate_id
        detections.append(detection)
        times.append(round(timestamp, 3))
        if abs(offset) <= 1e-9:
            anchor_id = candidate_id

    alignments = estimate_frame_alignments(displayed_frames, anchor_id)
    detections = [
        align_page_detection(
            detection,
            alignments[index],
            displayed_frames[index].shape,
            displayed_frames[anchor_id].shape,
        )
        for index, detection in enumerate(detections)
    ]
    consensus = consensus_page_quads(
        detections,
        min_confidence=0.55,
        max_corner_deviation=_CONSENSUS_MAX_CORNER_DEVIATION,
        anchor_ids={"left": anchor_id, "right": anchor_id},
    )
    consensus_quad = None
    if consensus["detected"]:
        try:
            consensus_quad = spread_quad_from_page_quads(consensus)
        except ValueError:
            consensus_quad = None
    consensus_iou = (
        polygon_iou(consensus_quad, sample["spread_quad"])
        if consensus_quad is not None
        else 0.0
    )
    check = {
        "detected": bool(consensus["detected"]),
        "confidence": consensus["confidence"],
        "left_confidence": consensus["left"]["confidence"],
        "right_confidence": consensus["right"]["confidence"],
        "left_consensus_count": consensus["left"]["consensus_count"],
        "right_consensus_count": consensus["right"]["consensus_count"],
        "offsets": offsets,
        "times": times,
        "alignment": alignment_summary(alignments),
        "iou": round(consensus_iou, 4),
        "min_iou": sample["min_page_iou"],
        "passed": bool(
            consensus["detected"] and consensus_iou >= sample["min_page_iou"]
        ),
    }
    return check, consensus_quad


def _reference_consensus_result(video_spec, sample, path):
    from manga_scan.split import rotate_image
    from manga_scan.spread_detect import detect_reference_spread_consensus
    from manga_scan.video import extract_frame

    offsets = [
        float(value)
        for value in sample.get("consensus_offsets", _DEFAULT_CONSENSUS_OFFSETS)
    ]
    displayed_frames = [
        rotate_image(
            extract_frame(path, float(sample["time"]) + offset),
            video_spec["expected_rotation"],
        )
        for offset in offsets
    ]
    anchor_index = next(
        index for index, offset in enumerate(offsets) if abs(offset) <= 1e-9
    )
    detection = detect_reference_spread_consensus(
        displayed_frames,
        min_confidence=0.55,
        anchor_index=anchor_index,
    )
    quad = detection.get("roi")
    iou = polygon_iou(quad, sample["spread_quad"]) if quad is not None else 0.0
    return {
        "detected": bool(detection["detected"]),
        "confidence": detection["confidence"],
        "source": detection.get("source"),
        "candidate_count": detection.get("candidate_count", 0),
        "frame_support": detection.get("frame_support", 0),
        "ambiguous": bool(detection.get("ambiguous")),
        "score_margin": detection.get("score_margin"),
        "alignment": detection.get("alignment"),
        "local_search": detection.get("local_search"),
        "prior_search": detection.get("prior_search"),
        "offsets": offsets,
        "iou": round(iou, 4),
        "min_iou": sample["min_reference_iou"],
        "passed": bool(
            detection["detected"] and iou >= sample["min_reference_iou"]
        ),
    }, quad


def _repair_result(video_spec, pair, path, model_path):
    from manga_scan.config import Config
    from manga_scan.finger_repair import repair_finger_regions
    from manga_scan.hand import HandDetector
    from manga_scan.perspective import warp_roi
    from manga_scan.split import rotate_image
    from manga_scan.video import extract_frame

    config = Config(hand_model=str(model_path), hand_backend="mediapipe", finger_repair=True)
    detector = HandDetector(config)
    rotation = video_spec["expected_rotation"]
    roi = pair["spread_quad"]
    try:
        target_source = rotate_image(extract_frame(path, pair["target_time"]), rotation)
        _target_overlap, target_source_mask = detector.detect(target_source, roi)
        target = warp_roi(target_source, roi)
        target_mask = warp_roi(target_source_mask, roi, interpolation=cv2.INTER_NEAREST)
        donors = []
        for index, timestamp in enumerate(pair["donor_times"], start=1):
            donor_source = rotate_image(extract_frame(path, timestamp), rotation)
            _donor_overlap, donor_source_mask = detector.detect(donor_source, roi)
            donors.append(
                {
                    "candidate_id": index,
                    "image": warp_roi(donor_source, roi),
                    "mask": warp_roi(
                        donor_source_mask,
                        roi,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                }
            )
        repaired, metadata, unresolved = repair_finger_regions(
            target,
            target_mask,
            donors,
            min_coverage=0.0,
            fallback="preserve",
        )
    finally:
        detector.close()

    masked = target_mask > 127
    changed = np.any(repaired != target, axis=2) if target.ndim == 3 else repaired != target
    outside_changed = int(np.count_nonzero(changed & ~masked))
    target_fraction = float(np.mean(masked))
    unresolved_fraction = float(np.count_nonzero(unresolved) / max(1, np.count_nonzero(masked)))
    passed = (
        target_fraction >= float(pair.get("min_target_mask_fraction", 0))
        and float(metadata.get("donor_coverage", 0.0)) >= float(pair.get("min_donor_coverage", 0))
        and outside_changed == 0
    )
    return {
        "video_id": video_spec["id"],
        "pair_id": pair["id"],
        "target_time": pair["target_time"],
        "donor_times": pair["donor_times"],
        "tags": pair.get("tags", []),
        "target_mask_fraction": round(target_fraction, 6),
        "donor_coverage": metadata.get("donor_coverage", 0.0),
        "coverage": metadata.get("coverage", 0.0),
        "unresolved_fraction": round(unresolved_fraction, 4),
        "outside_mask_changed_pixels": outside_changed,
        "donors": metadata.get("donors", []),
        "local_alignment": metadata.get("local_alignment"),
        "passed": bool(passed),
    }


def _repair_error_result(video_spec, pair, error):
    return {
        "video_id": video_spec["id"],
        "pair_id": pair["id"],
        "target_time": pair["target_time"],
        "donor_times": pair["donor_times"],
        "tags": pair.get("tags", []),
        "passed": False,
        "error": error,
    }


def _repair_worker(connection, video_spec, pair, path, model_path):
    """Keep native MediaPipe failures from aborting the full benchmark run."""
    try:
        result = _repair_result(video_spec, pair, Path(path), Path(model_path))
        connection.send({"ok": True, "result": result})
    except Exception as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _repair_result_isolated(video_spec, pair, path, model_path, timeout=180):
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_repair_worker,
        args=(writer, video_spec, pair, str(path), str(model_path)),
        name=f"real-benchmark-{video_spec['id']}-{pair['id']}",
    )
    process.start()
    writer.close()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(5)
        reader.close()
        return _repair_error_result(
            video_spec,
            pair,
            f"MediaPipe worker exceeded the {timeout}s timeout",
        )

    payload = None
    try:
        if reader.poll():
            payload = reader.recv()
    except EOFError:
        payload = None
    finally:
        reader.close()

    if payload and payload.get("ok"):
        return payload["result"]
    if payload:
        return _repair_error_result(video_spec, pair, payload["error"])

    exitcode = process.exitcode
    if exitcode is not None and exitcode < 0:
        try:
            reason = signal.Signals(-exitcode).name
        except ValueError:
            reason = f"signal {-exitcode}"
        error = (
            f"MediaPipe worker terminated by {reason}. On macOS, run the benchmark "
            "from a session with Metal/GPU service access."
        )
    else:
        error = f"MediaPipe worker exited without a result (exit code {exitcode})"
    return _repair_error_result(video_spec, pair, error)


def _summarize(report):
    checks = Counter()
    passes = Counter()
    if report["videos"]:
        checks["media"] = len(report["videos"])
        passes["media"] = sum(int(item["metadata_passed"]) for item in report["videos"])
    for sample in report["samples"]:
        for name, check in sample["checks"].items():
            checks[name] += 1
            passes[name] += int(bool(check.get("passed")))
    if report["rotations"]:
        checks["rotation"] = len(report["rotations"])
        passes["rotation"] = sum(int(item["passed"]) for item in report["rotations"])
    if report["repairs"]:
        checks["finger_repair"] = len(report["repairs"])
        passes["finger_repair"] = sum(int(item["passed"]) for item in report["repairs"])
    summary = {
        name: {"passed": passes[name], "total": total}
        for name, total in sorted(checks.items())
    }
    summary["all_required_passed"] = all(passes[name] == total for name, total in checks.items())
    return summary


def run(
    manifest_path: Path,
    videos_dir: Path,
    output: Path,
    debug_dir: Path,
    with_hands: bool,
    hand_model: Path,
):
    from manga_scan.rotation_detection import detect_video_rotation
    from manga_scan.split import rotate_image
    from manga_scan.video import extract_frame, probe

    data = json.loads(manifest_path.read_text())
    validate_manifest(data)
    videos_dir = videos_dir.resolve()
    hash_cache = {}
    report = {
        "manifest": str(manifest_path),
        "videos_dir": str(videos_dir),
        "videos": [],
        "rotations": [],
        "samples": [],
        "repairs": [],
    }
    debug_dir.mkdir(parents=True, exist_ok=True)

    for spec in data["videos"]:
        path = _find_video(videos_dir, spec, hash_cache)
        metadata = probe(path)
        media_ok = (
            int(metadata["width"]) == int(spec["width"])
            and int(metadata["height"]) == int(spec["height"])
            and abs(float(metadata["duration"]) - float(spec["duration"])) <= 0.25
        )
        report["videos"].append(
            {
                "id": spec["id"],
                "path": str(path),
                "sha256": hash_cache[path],
                "width": metadata["width"],
                "height": metadata["height"],
                "duration": round(float(metadata["duration"]), 6),
                "metadata_passed": bool(media_ok),
            }
        )

        first = extract_frame(path, 0.0)
        rotation = detect_video_rotation(path, metadata, first)
        report["rotations"].append(
            {
                "video_id": spec["id"],
                "expected": spec["expected_rotation"],
                "actual": rotation["rotation"],
                "confidence": rotation["confidence"],
                "source": rotation["source"],
                "scores": rotation.get("scores"),
                "requires_confirmation": bool(rotation.get("requires_confirmation")),
                "rotation_options": rotation.get("rotation_options"),
                "passed": bool(
                    rotation["rotation"] == spec["expected_rotation"]
                    or (
                        rotation.get("requires_confirmation")
                        and spec["expected_rotation"] in rotation.get("rotation_options", [])
                    )
                ),
            }
        )

        for sample in spec.get("samples", []):
            source = extract_frame(path, sample["time"])
            displayed = rotate_image(source, spec["expected_rotation"])
            sample_result, reference_quad, page_quad = _sample_result(spec["id"], sample, displayed)
            consensus_quad = None
            reference_consensus_quad = None
            if sample["expect_spread"]:
                consensus_check, consensus_quad = _page_consensus_result(spec, sample, path)
                sample_result["checks"]["page_contour_consensus"] = consensus_check
                reference_consensus_check, reference_consensus_quad = (
                    _reference_consensus_result(spec, sample, path)
                )
                sample_result["checks"]["reference_spread_consensus"] = (
                    reference_consensus_check
                )
            report["samples"].append(sample_result)
            canvas = displayed.copy()
            if sample.get("spread_quad") is not None:
                _draw_quad(canvas, sample["spread_quad"], (70, 210, 70), "expected")
            _draw_quad(canvas, reference_quad, (255, 170, 0), "reference")
            _draw_quad(canvas, page_quad, (60, 80, 255), "pages")
            _draw_quad(canvas, consensus_quad, (210, 80, 210), "consensus")
            _draw_quad(
                canvas,
                reference_consensus_quad,
                (220, 220, 40),
                "reference consensus",
            )
            name = f"{spec['id']}__{sample['id']}.jpg"
            cv2.imwrite(str(debug_dir / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if with_hands:
            for pair in spec.get("repair_pairs", []):
                report["repairs"].append(
                    _repair_result_isolated(spec, pair, path, hand_model)
                )

    report["summary"] = _summarize(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _print_summary(report):
    print(f"real benchmark: {len(report['videos'])} videos, {len(report['samples'])} samples")
    for name, result in report["summary"].items():
        if name == "all_required_passed":
            continue
        print(f"  {name:24s} {result['passed']:>2}/{result['total']:<2} passed")
    if not report["repairs"]:
        print("  finger_repair            skipped (use --with-hands)")
    overall = "PASS" if report["summary"]["all_required_passed"] else "FAIL"
    print(f"  {'overall':24s} {overall}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/real/cases.json"))
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(os.environ.get("MANGA_SCAN_REAL_BENCHMARK_DIR", "benchmarks/real/videos")),
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/real/reports/latest.json"))
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("benchmarks/real/reports/debug"),
    )
    parser.add_argument("--with-hands", action="store_true")
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a benchmark check fails",
    )
    args = parser.parse_args(argv)
    try:
        report = run(
            args.manifest,
            args.videos_dir,
            args.output,
            args.debug_dir,
            args.with_hands,
            args.hand_model,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"real benchmark error: {exc}", file=sys.stderr)
        return 2
    _print_summary(report)
    print(f"report: {args.output}")
    print(f"debug:  {args.debug_dir}")
    return 1 if args.strict and not report["summary"]["all_required_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
