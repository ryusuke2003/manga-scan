"""Conservative duplicate detection with camera-motion tolerant alignment."""

import cv2
import numpy as np


def gray_thumb(image, size=(256, 256)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def dhash(image):
    small = gray_thumb(image, (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def ssim(a, b):
    a, b = gray_thumb(a).astype(np.float64), gray_thumb(b).astype(np.float64)

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = np.maximum(0, blur(a * a) - mu_a * mu_a)
    var_b = np.maximum(0, blur(b * b) - mu_b * mu_b)
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    value = (
        (2 * mu_a * mu_b + c1)
        * (2 * cov + c2)
        / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
    )
    return float(value[5:-5, 5:-5].mean())


def _alignment_defaults():
    return {
        "available": False,
        "feature_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "match_balance": 0.0,
        "inlier_balance": 0.0,
        "inliers_by_half": [0, 0],
        "coverage_x": 0.0,
        "coverage_y": 0.0,
        "overlap": 0.0,
        "correlation": 0.0,
        "duplicate": False,
        "suspect": False,
    }


def _feature_gray(image, max_side=640):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def feature_alignment(a, b):
    """Match the same photographed spread despite translation/perspective drift.

    The conservative duplicate decision requires broad feature support on both
    halves. A match confined to one physical page is only marked as suspect,
    because the other page may genuinely have changed.
    """
    result = _alignment_defaults()
    left = _feature_gray(a)
    right = _feature_gray(b)
    if min(*left.shape[:2], *right.shape[:2]) < 48:
        return result

    orb = cv2.ORB_create(
        nfeatures=1800,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=12,
    )
    left_points, left_desc = orb.detectAndCompute(left, None)
    right_points, right_desc = orb.detectAndCompute(right, None)
    if left_desc is None or right_desc is None:
        return result

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(right_desc, left_desc, k=2)
    matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    result["feature_matches"] = len(matches)
    if len(matches) < 8:
        return result

    source = np.float32([right_points[item.queryIdx].pt for item in matches])
    target = np.float32([left_points[item.trainIdx].pt for item in matches])
    left_height, left_width = left.shape[:2]
    right_height, right_width = right.shape[:2]
    source_left = int(np.count_nonzero(source[:, 0] < right_width / 2))
    source_right = len(matches) - source_left
    target_left = int(np.count_nonzero(target[:, 0] < left_width / 2))
    target_right = len(matches) - target_left
    match_balance = min(source_left, source_right, target_left, target_right) / len(matches)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        return result

    accepted = inlier_mask.ravel().astype(bool)
    source = source[accepted]
    target = target[accepted]
    inliers = len(source)
    if inliers < 4:
        return result

    source_on_left = source[:, 0] < right_width / 2
    target_on_left = target[:, 0] < left_width / 2
    left_inliers = int(np.count_nonzero(source_on_left & target_on_left))
    right_inliers = int(np.count_nonzero(~source_on_left & ~target_on_left))
    inlier_balance = min(left_inliers, right_inliers) / inliers

    coverage_x = min(
        float(np.ptp(source[:, 0])) / right_width,
        float(np.ptp(target[:, 0])) / left_width,
    )
    coverage_y = min(
        float(np.ptp(source[:, 1])) / right_height,
        float(np.ptp(target[:, 1])) / left_height,
    )

    warped = cv2.warpPerspective(right, homography, (left_width, left_height))
    overlap_mask = cv2.warpPerspective(
        np.full(right.shape, 255, dtype=np.uint8),
        homography,
        (left_width, left_height),
        flags=cv2.INTER_NEAREST,
    ) > 0
    overlap = float(overlap_mask.mean())
    correlation = 0.0
    if np.count_nonzero(overlap_mask) >= 256:
        left_values = cv2.GaussianBlur(left, (3, 3), 0).astype(np.float32)[overlap_mask]
        right_values = cv2.GaussianBlur(warped, (3, 3), 0).astype(np.float32)[overlap_mask]
        if float(left_values.std()) >= 1.0 and float(right_values.std()) >= 1.0:
            correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0

    inlier_ratio = inliers / len(matches)
    feature_density = inliers / max(1, min(len(left_points), len(right_points)))
    suspect = (
        inliers >= 40
        and feature_density >= 0.025
        and inlier_ratio >= 0.35
        and coverage_x >= 0.25
        and coverage_y >= 0.30
        and overlap >= 0.45
        and correlation >= 0.30
    )
    duplicate = (
        inliers >= 60
        and feature_density >= 0.04
        and inlier_ratio >= 0.45
        and min(left_inliers, right_inliers) >= 20
        and inlier_balance >= 0.08
        and coverage_x >= 0.40
        and coverage_y >= 0.45
        and overlap >= 0.55
        and correlation >= 0.45
    )
    # A hand may cover a large fraction of one page and depress global pixel
    # correlation. Numerous geometrically consistent matches on *both* pages
    # still establish that the visible content is the same. Count RANSAC
    # inliers per half; pre-RANSAC matches can misleadingly look balanced when
    # only one physical page is actually unchanged.
    occluded_duplicate = (
        inliers >= 120
        and inlier_ratio >= 0.40
        and min(left_inliers, right_inliers) >= 40
        and inlier_balance >= 0.20
        and coverage_x >= 0.58
        and coverage_y >= 0.65
        and overlap >= 0.75
        and correlation >= 0.25
    )
    result.update(
        available=True,
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 6),
        match_balance=round(match_balance, 6),
        inlier_balance=round(inlier_balance, 6),
        inliers_by_half=[left_inliers, right_inliers],
        coverage_x=round(coverage_x, 6),
        coverage_y=round(coverage_y, 6),
        overlap=round(overlap, 6),
        correlation=round(correlation, 6),
        duplicate=bool(duplicate or occluded_duplicate),
        suspect=bool(suspect or duplicate or occluded_duplicate),
    )
    return result


def compare(a, b, config):
    distance = (dhash(a) ^ dhash(b)).bit_count()
    similarity = ssim(a, b)
    # Also compare halves: a nearly blank spread must not hide a changed single page.
    halves = [
        ssim(x, y) for x, y in zip(np.array_split(a, 2, axis=1), np.array_split(b, 2, axis=1))
    ]
    informative = min(float(gray_thumb(a).std()), float(gray_thumb(b).std())) >= 8
    direct_duplicate = (
        informative
        and distance <= config.duplicate_hash_distance
        and min(similarity, *halves) >= config.duplicate_ssim
    )
    alignment = (
        _alignment_defaults()
        if direct_duplicate or not informative
        else feature_alignment(a, b)
    )
    duplicate = direct_duplicate or (informative and alignment["duplicate"])
    suspect = (
        duplicate
        or similarity >= config.duplicate_suspect_ssim
        or (informative and alignment["suspect"])
    )
    return {
        "duplicate": bool(duplicate),
        "suspect": bool(suspect),
        "hash_distance": distance,
        "ssim": similarity,
        "half_ssim": halves,
        "alignment": alignment,
    }
