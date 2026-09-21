# Real-camera benchmark

This benchmark exercises manga-scan against real phone video while keeping the copyrighted manga footage out of Git.

The repository stores only:

- source filename, dimensions, duration and SHA-256
- hand-annotated spread quads at selected timestamps
- scenario tags such as hand occlusion, tight crop, curvature and rotation
- optional same-spread target/donor timestamps for finger-repair checks

`benchmarks/real/videos/` and generated reports are intentionally ignored.

## Initial corpus

The first corpus contains three videos and ten annotated frames:

- `IMG_6481.mp4`: landscape desk capture, curved gutter, moving fingers, one frame touching the right edge
- `IMG_6474.mp4`: portrait file that needs 270-degree clockwise correction, tight handheld framing and strong curvature
- `IMG_6479.mp4`: portrait desk capture with a cover negative case and increasingly heavy hand occlusion

Two same-spread pairs are also defined for optional MediaPipe/finger-repair evaluation.

## Put the local videos in place

Copy the exact source videos into:

```text
benchmarks/real/videos/
  IMG_6474.MOV
  IMG_6479.MOV
  IMG_6481.MOV
```

The runner verifies SHA-256 before using a file. A renamed file is also accepted when its hash matches.

You can instead keep the videos elsewhere:

```bash
MANGA_SCAN_REAL_BENCHMARK_DIR=/path/to/private/videos \
  python scripts/run_real_benchmark.py
```

## Run geometry + rotation benchmark

```bash
python scripts/run_real_benchmark.py
```

This evaluates:

1. automatic video rotation
2. reference-spread detection, including the cover false-positive case
3. left/right page-contour detection from a manually annotated coarse spread ROI
4. page-contour consensus across the center frame and frames 0.5 seconds before/after,
   after optical-flow/RANSAC homography alignment into the center-frame coordinates
5. top-8 Hough reference candidates verified by the same aligned left/right page consensus
6. polygon IoU against the manually annotated real page boundary
7. sharpness as a reported diagnostic value

The default consensus window is `[-0.5, 0.0, 0.5]` seconds. A positive sample
can override it with `consensus_offsets`; the list must contain at least three
unique offsets including `0`, and every resulting timestamp must stay inside
the video. This mirrors the production behavior of combining page boundaries
from multiple candidate frames while keeping the single-frame metric visible.
When feature tracking or homography validation fails, that frame safely falls
back to its original coordinates instead of applying an unreliable warp.
Reference-outline ranking also combines Hough lines with long OpenCV Line
Segment Detector results, so short panel/text rules are less likely to outrank
page-length edges.
Each outline is then refined on the center frame with a bounded beam search
that can move every edge inward or outward. Only the strongest refined priors
are evaluated across the neighboring frames.

Results are written to:

```text
benchmarks/real/reports/latest.json
benchmarks/real/reports/debug/*.jpg
```

Debug images overlay:

- `expected`: hand annotation
- `reference`: `detect_reference_spread()` result
- `pages`: outer spread reconstructed from `detect_page_quads()`
- `consensus`: outer spread reconstructed from neighboring page detections
- `reference consensus`: top Hough/coarse candidate selected from neighboring frames

Use `--strict` when you intentionally want benchmark failures to return a non-zero exit code. The real-media benchmark is not part of normal CI because the source videos are private/local-only.

## Optional finger-repair benchmark

Install the normal hand support and hand model first, then run:

```bash
python scripts/run_real_benchmark.py --with-hands
```

The initial repair pairs use different moments from the same physical spread. The benchmark records:

- target hand-mask fraction
- donor coverage and final coverage
- unresolved fraction
- donor IDs/local-alignment metadata
- number of changed pixels outside the target hand mask

The safety invariant is strict: pixels outside the target mask must remain bit-identical.

MediaPipe's macOS wheel initializes a Metal context even when its inference
delegate is explicitly set to CPU. Run `--with-hands` from a normal interactive
macOS session with GPU service access. The benchmark isolates each repair pair
in a child process so a native MediaPipe/Metal abort is recorded as a failed
repair check instead of terminating the entire benchmark without a report.

This is deliberately not treated as a pixel-perfect restoration benchmark yet: there is no committed clean copyrighted ground-truth image. Coverage and safety are measured now; later corpus versions can add locally stored clean references without publishing the media.

## Annotation policy

Do not generate expected quads from the production detector. The point is to catch regressions and detector mistakes, so expected geometry must be reviewed against the real frame itself.

When adding a case:

- prefer stable frames but include hard cases intentionally
- annotate the visible white page boundary, not the red cover or desk
- store quads in `TL, TR, BR, BL` order using normalized coordinates
- lower IoU thresholds only for a documented hard condition such as frame clipping or large hand occlusion
- add descriptive tags so reports can later be grouped by failure mode
