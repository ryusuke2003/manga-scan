from dataclasses import dataclass

import cv2
import numpy as np


def motion_score(previous, current):
    """Normalized blurred grayscale difference, measured inside the rectified ROI."""

    def gray(im):
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(im, (5, 5), 0)

    a, b = gray(previous), gray(current)
    return float(cv2.absdiff(a, b).mean() / 255)


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
