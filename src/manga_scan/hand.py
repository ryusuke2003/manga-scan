from pathlib import Path

import cv2
import numpy as np

from .perspective import pixel_quad


_TEMPORAL_MAX_ALIGNMENT_SIDE = 640
_TEMPORAL_MIN_PEERS = 3
_TEMPORAL_MIN_ALIGNMENT_SCORE = 0.70
_TEMPORAL_MAX_SHIFT_FRACTION = 0.025
_TEMPORAL_DIFF_THRESHOLD = 18.0
_TEMPORAL_PEER_MAD_THRESHOLD = 8.0
_TEMPORAL_MIN_COMPONENT_FRACTION = 0.0002
_TEMPORAL_MAX_COMPONENT_FRACTION = 0.15
_TEMPORAL_EDGE_BAND_FRACTION = 0.025


def _page_mask(shape, roi):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(pixel_quad(roi, shape)).astype(np.int32), 255)
    return mask


def _resize_mask(mask, shape):
    h, w = shape[:2]
    if mask is None:
        return np.zeros((h, w), np.uint8)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _temporal_residual(target_gray, peer_gray, page_mask, target_mask, peer_mask):
    blocked = ((target_mask > 0) | (peer_mask > 0)).astype(np.uint8)
    blocked = cv2.dilate(blocked, np.ones((5, 5), np.uint8), iterations=1)
    clean = (page_mask > 0) & (blocked == 0)
    if np.count_nonzero(clean) < max(64, np.count_nonzero(page_mask) * 0.2):
        return None
    delta = np.abs(
        target_gray[clean].astype(np.float32)
        - peer_gray[clean].astype(np.float32)
    )
    return float(np.mean(delta) / 255.0)


def _align_temporal_peer(target, roi, target_mask, peer, peer_mask):
    """Align one same-spread peer with a bounded translation-only ECC warp."""
    h, w = target.shape[:2]
    if peer.shape[:2] != (h, w):
        peer = cv2.resize(peer, (w, h), interpolation=cv2.INTER_CUBIC)
    target_mask = _resize_mask(target_mask, target.shape)
    peer_mask = _resize_mask(peer_mask, target.shape)
    page = _page_mask(target.shape, roi)

    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    peer_gray = cv2.cvtColor(peer, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, _TEMPORAL_MAX_ALIGNMENT_SIDE / max(h, w))
    small_size = (max(8, round(w * scale)), max(8, round(h * scale)))
    target_small = cv2.resize(target_gray, small_size, interpolation=cv2.INTER_AREA)
    peer_small = cv2.resize(peer_gray, small_size, interpolation=cv2.INTER_AREA)
    page_small = cv2.resize(page, small_size, interpolation=cv2.INTER_NEAREST)
    target_mask_small = cv2.resize(
        target_mask, small_size, interpolation=cv2.INTER_NEAREST
    )
    peer_mask_small = cv2.resize(
        peer_mask, small_size, interpolation=cv2.INTER_NEAREST
    )

    kernel = np.ones((5, 5), np.uint8)
    target_blocked = cv2.dilate(
        (target_mask_small > 0).astype(np.uint8), kernel, iterations=1
    )
    peer_blocked = cv2.dilate(
        (peer_mask_small > 0).astype(np.uint8), kernel, iterations=1
    )
    page_pixels = page_small > 0
    target_clean = page_pixels & (target_blocked == 0)
    peer_clean = page_pixels & (peer_blocked == 0)
    required = max(64, round(np.count_nonzero(page_pixels) * 0.2))
    if np.count_nonzero(target_clean) < required or np.count_nonzero(peer_clean) < required:
        return None

    values = np.concatenate(
        (
            target_small[target_clean].astype(np.float32),
            peer_small[peer_clean].astype(np.float32),
        )
    )
    fill = np.uint8(np.clip(round(float(np.median(values))), 0, 255))
    target_for_ecc = target_small.copy()
    peer_for_ecc = peer_small.copy()
    target_for_ecc[~target_clean] = fill
    peer_for_ecc[~peer_clean] = fill

    input_mask = cv2.erode(
        page_small,
        np.ones((5, 5), np.uint8),
        iterations=1,
        borderValue=0,
    )
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    try:
        score, warp = cv2.findTransformECC(
            target_for_ecc,
            peer_for_ecc,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=input_mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None

    if not np.isfinite(score) or score < _TEMPORAL_MIN_ALIGNMENT_SCORE:
        return None
    warp = warp.astype(np.float32, copy=True)
    warp[0, 2] *= w / small_size[0]
    warp[1, 2] *= h / small_size[1]
    max_shift = max(2.0, min(h, w) * _TEMPORAL_MAX_SHIFT_FRACTION)
    if abs(float(warp[0, 2])) > max_shift or abs(float(warp[1, 2])) > max_shift:
        return None

    aligned = cv2.warpAffine(
        peer,
        warp,
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    aligned_mask = cv2.warpAffine(
        peer_mask,
        warp,
        (w, h),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    identity_residual = _temporal_residual(
        target_gray,
        peer_gray,
        page,
        target_mask,
        peer_mask,
    )
    aligned_residual = _temporal_residual(
        target_gray,
        cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
        page,
        target_mask,
        aligned_mask,
    )
    if (
        identity_residual is not None
        and (
            aligned_residual is None
            or identity_residual <= aligned_residual + 0.002
        )
    ):
        return peer, peer_mask

    return aligned, aligned_mask


def temporal_transient_mask(image, roi, peers, target_mask=None, padding=0.015):
    """Find edge-connected content that is transient across same-spread frames.

    MediaPipe remains the primary detector. This conservative temporal detector
    only supplements it: peer frames are aligned to the target, known hand
    pixels are neutralized, and a component must either touch the page boundary
    or connect to an existing hand mask before it can be returned.
    """
    result = np.zeros(image.shape[:2], np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return result

    target_mask = _resize_mask(target_mask, image.shape)
    aligned_peers = []
    aligned_masks = []
    for peer in peers:
        peer_image = peer.get("image")
        if not isinstance(peer_image, np.ndarray) or peer_image.ndim != 3:
            continue
        aligned = _align_temporal_peer(
            image,
            roi,
            target_mask,
            peer_image,
            peer.get("mask"),
        )
        if aligned is None:
            continue
        aligned_image, aligned_mask = aligned
        aligned_peers.append(aligned_image)
        aligned_masks.append(aligned_mask > 0)

    if len(aligned_peers) < _TEMPORAL_MIN_PEERS:
        return result

    stack = np.stack(aligned_peers).astype(np.float32)
    invalid = np.stack(aligned_masks)
    valid_count = np.count_nonzero(~invalid, axis=0)
    image_mask = np.repeat(invalid[:, :, :, None], 3, axis=3)
    masked_stack = np.ma.array(stack, mask=image_mask)
    median = np.ma.median(masked_stack, axis=0).data
    median[valid_count == 0] = image[valid_count == 0]
    median = np.clip(median, 0, 255).astype(np.uint8)

    gray_stack = np.stack(
        [cv2.cvtColor(peer, cv2.COLOR_BGR2GRAY) for peer in aligned_peers]
    ).astype(np.float32)
    masked_gray = np.ma.array(gray_stack, mask=invalid)
    gray_median = np.ma.median(masked_gray, axis=0)
    deviations = np.ma.abs(masked_gray - gray_median)
    peer_mad = np.ma.median(deviations, axis=0).filled(255.0)

    target_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    median_lab = cv2.cvtColor(median, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.max(np.abs(target_lab - median_lab), axis=2)

    page = _page_mask(image.shape, roi)
    candidate = (
        (delta >= _TEMPORAL_DIFF_THRESHOLD)
        & (peer_mad <= _TEMPORAL_PEER_MAD_THRESHOLD)
        & (valid_count >= 2)
        & (page > 0)
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    h, w = image.shape[:2]
    page_area = max(1, int(np.count_nonzero(page)))
    edge_radius = max(3, round(min(h, w) * _TEMPORAL_EDGE_BAND_FRACTION))
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_radius * 2 + 1, edge_radius * 2 + 1),
    )
    inner = cv2.erode(page, edge_kernel, iterations=1, borderValue=0)
    boundary_band = (page > 0) & (inner == 0)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        fraction = area / page_area
        if not _TEMPORAL_MIN_COMPONENT_FRACTION <= fraction <= _TEMPORAL_MAX_COMPONENT_FRACTION:
            continue
        component = labels == label
        touches_known = np.any(component & (target_mask > 0))
        if not touches_known and not np.any(component & boundary_band):
            continue
        if not touches_known:
            if stats[label, cv2.CC_STAT_WIDTH] > w * 0.6:
                continue
            if stats[label, cv2.CC_STAT_HEIGHT] > h * 0.6:
                continue
        result[component] = 255

    radius = max(1, round(min(h, w) * padding))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    return cv2.dilate(result, kernel, iterations=1)


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
