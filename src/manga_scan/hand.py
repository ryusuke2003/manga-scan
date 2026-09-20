from pathlib import Path

import cv2
import numpy as np

from .perspective import pixel_quad


def boundary_finger_mask(image, roi, padding=0.015):
    """Supplement landmarks with edge-connected skin on monochrome pages.

    This is deliberately limited to small reddish components on otherwise
    neutral pages. It is not a general skin classifier for colored artwork.
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w), np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return result
    if max(h, w) > 960:
        scale = 960 / max(h, w)
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        mask = boundary_finger_mask(small, roi, padding)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    page = np.zeros_like(result)
    cv2.fillConvexPoly(page, np.rint(pixel_quad(roi, image.shape)).astype(np.int32), 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if np.mean(hsv[:, :, 1][page > 0] < 85) < 0.75:
        return result
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 175))
        & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] > 35)
        & (ycc[:, :, 1] > 140) & (ycc[:, :, 2] > 100)
    ).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Break thin colored cover strips that otherwise join a fingertip to the
    # whole book outline; also discard colored noise along printed ink.
    size = max(3, round(min(h, w) * 0.02) | 1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    band = page & ~cv2.erode(page, np.ones((9, 9), np.uint8), borderValue=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    page_area = np.count_nonzero(page)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < page_area * 0.0003:
            continue
        component = labels == index
        overlap = np.count_nonzero(component & (page > 0)) / max(1, page_area)
        if not 0.0003 <= overlap <= 0.12 or not np.any(component & (band > 0)):
            continue
        # Long book-cover/desk strips are not fingertips.
        if stats[index, cv2.CC_STAT_WIDTH] > w * 0.6:
            continue
        if stats[index, cv2.CC_STAT_HEIGHT] > h * 0.6:
            continue
        result[component] = 255
    radius = max(1, round(min(h, w) * padding))
    return cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))


def overlap_from_landmarks(shape, roi, hands, padding=0.015):
    """Union of padded landmark hulls / page ROI area. A conservative mask proxy."""
    h, w = shape[:2]
    page_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(page_mask, np.rint(pixel_quad(roi, shape)).astype(np.int32), 255)
    hand_mask = np.zeros_like(page_mask)
    for hand in hands:
        points = np.array([[p.x * (w - 1), p.y * (h - 1)] for p in hand], np.float32)
        if len(points) >= 3 and np.isfinite(points).all():
            points = np.clip(points, [0, 0], [w - 1, h - 1]).astype(np.int32)
            cv2.fillConvexPoly(hand_mask, cv2.convexHull(points), 255)
    radius = round(min(h, w) * padding)
    if radius:
        hand_mask = cv2.dilate(
            hand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        )
    overlap = float(
        np.count_nonzero(cv2.bitwise_and(hand_mask, page_mask))
        / max(1, np.count_nonzero(page_mask))
    )
    return overlap, hand_mask


class HandDetector:
    def __init__(self, config):
        self.config = config
        self.detector = None
        if config.hand_backend == "none":
            return
        model = Path(config.hand_model).expanduser()
        if not model.is_file():
            raise ValueError(
                f"Hand model missing: {model}. Run scripts/download_hand_model.py during setup, "
                "or explicitly use hand_backend='none' (all results will be flagged)."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install hand support: pip install -e '.[hands]'") from exc
        if mp.__version__ != "0.10.35":
            raise RuntimeError("Use the tested MediaPipe version: pip install 'mediapipe==0.10.35'")
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model.resolve()), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, image, roi):
        if self.detector is None:
            return None, np.zeros(image.shape[:2], np.uint8)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        )
        _, mask = overlap_from_landmarks(
            image.shape, roi, result.hand_landmarks, self.config.hand_padding
        )
        mask |= boundary_finger_mask(image, roi, self.config.hand_padding)
        page = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(page, np.rint(pixel_quad(roi, image.shape)).astype(np.int32), 255)
        overlap = np.count_nonzero(mask & page) / max(1, np.count_nonzero(page))
        return float(overlap), mask

    def close(self):
        if self.detector is not None:
            self.detector.close()
