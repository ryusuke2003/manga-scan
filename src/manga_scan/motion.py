from dataclasses import dataclass

import cv2
import numpy as np

_MOTION_MAX_SIDE = 512
# Motion is about page movement, not fine focus breathing. A modest low-pass
# still preserves page-turn structure while making small AF changes much less dominant.
_MOTION_BLUR_SIGMA = 1.8
_PHOTOMETRIC_MIN_SPAN = 18.0
_PHOTOMETRIC_GAIN_MIN = 0.82
_PHOTOMETRIC_GAIN_MAX = 1.22
_PHOTOMETRIC_BIAS_LIMIT = 28.0
_PHASE_MIN_RESPONSE = 0.20
_PHASE_MAX_SHIFT_PIXELS = 6.0
_PHASE_MAX_SHIFT_FRACTION = 0.02


def _motion_gray(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("motion images must be grayscale or BGR numpy arrays")
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError("motion images must be grayscale or BGR")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    if min(height, width) < 2:
        raise ValueError("motion images are too small")
    scale = min(1.0, _MOTION_MAX_SIDE / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    image = image.astype(np.float32, copy=False)
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=_MOTION_BLUR_SIGMA,
        sigmaY=_MOTION_BLUR_SIGMA,
    )


def _photometric_match(reference, current):
    """Remove moderate global AE gain/offset changes without hiding real scene changes."""
    ref_low, ref_mid, ref_high = np.percentile(reference, (10, 50, 90))
    cur_low, cur_mid, cur_high = np.percentile(current, (10, 50, 90))
    ref_span = float(ref_high - ref_low)
    cur_span = float(cur_high - cur_low)
    if ref_span < _PHOTOMETRIC_MIN_SPAN or cur_span < _PHOTOMETRIC_MIN_SPAN:
        return current

    gain = float(
        np.clip(
            ref_span / max(cur_span, 1e-6),
            _PHOTOMETRIC_GAIN_MIN,
            _PHOTOMETRIC_GAIN_MAX,
        )
    )
    bias = float(
        np.clip(
            float(ref_mid) - gain * float(cur_mid),
            -_PHOTOMETRIC_BIAS_LIMIT,
            _PHOTOMETRIC_BIAS_LIMIT,
        )
    )
    return np.clip(current * gain + bias, 0, 255).astype(np.float32)


def _gradient_structure(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 95))
    if scale < 4.0:
        return None
    return np.clip(magnitude / scale, 0, 1).astype(np.float32)


def _align_micro_translation(reference, current):
    """Undo only high-confidence camera/book jitter of a few pixels.

    Larger shifts are intentionally kept as motion because they may be page turns
    or meaningful book movement.
    """
    ref_structure = _gradient_structure(reference)
    cur_structure = _gradient_structure(current)
    if ref_structure is None or cur_structure is None:
        return current

    height, width = reference.shape[:2]
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(
            ref_structure,
            cur_structure,
            window,
        )
    except cv2.error:
        return current

    if not np.isfinite([dx, dy, response]).all() or response < _PHASE_MIN_RESPONSE:
        return current

    max_shift = min(
        _PHASE_MAX_SHIFT_PIXELS,
        max(2.0, min(height, width) * _PHASE_MAX_SHIFT_FRACTION),
    )
    if abs(float(dx)) > max_shift or abs(float(dy)) > max_shift:
        return current

    matrix = np.asarray(
        [[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        current,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def motion_score(previous, current):
    """AE/AF-tolerant frame motion score inside the rectified ROI.

    v2 keeps the existing 0..1-like score scale while reducing false motion
    caused by moderate auto-exposure changes, focus breathing and a few pixels
    of camera/book jitter. Real page turns and larger movement are intentionally
    left uncorrected.
    """
    if previous.shape[:2] != current.shape[:2]:
        raise ValueError("motion images must have matching dimensions")

    reference = _motion_gray(previous)
    candidate = _motion_gray(current)

    # Uniform/near-uniform frames do not provide enough structure to distinguish
    # AE from a genuine scene replacement. Preserve the conservative raw result.
    candidate = _photometric_match(reference, candidate)
    candidate = _align_micro_translation(reference, candidate)

    return float(np.mean(np.abs(reference - candidate)) / 255.0)


@dataclass
class Sample:
    index: int
    time: float
    motion: float
    sharpness: float


class StableDetector:
    """Hysteresis state machine. Only consecutive low-motion frames become candidates."""

    def __init__(self, stable_frames, low_threshold, high_threshold):
        self.required = stable_frames
        self.low = low_threshold
        self.high = high_threshold
        self.state = "turning"
        self.pending = []
        self.segment = []

    def push(self, sample):
        complete = None
        if sample.motion <= self.low:
            self.pending.append(sample)
            if self.state == "stable":
                self.segment.append(sample)
                self.pending.clear()
            elif len(self.pending) >= self.required:
                self.state = "stable"
                self.segment = self.pending
                self.pending = []
        else:
            self.pending.clear()
            if self.state == "stable":
                # Medium motion pauses candidates; only high motion separates spreads.
                if sample.motion >= self.high:
                    complete = self.segment
                    self.segment = []
                    self.state = "turning"
            elif sample.motion >= self.high:
                self.state = "turning"
        return complete

    def finish(self):
        segment = self.segment if self.state == "stable" else []
        self.segment, self.pending = [], []
        self.state = "turning"
        return segment


def choose_candidates(samples, limit):
    """One locally sharp, still candidate per time bin; covers late hand withdrawal."""
    if not samples:
        return []
    bins = np.array_split(np.arange(len(samples)), min(limit, len(samples)))
    return [
        max((samples[int(i)] for i in group), key=lambda s: np.log1p(s.sharpness) - 30 * s.motion)
        for group in bins
    ]
