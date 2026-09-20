import cv2
import numpy as np

from manga_scan.config import Config
from manga_scan.final_quality import adjacent_quality_check, final_quality_checks


def manga_page(height=320, width=240):
    image = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(image, (18, 20), (width - 20, height - 22), (45, 45, 45), 2)
    cv2.line(image, (30, 90), (width - 35, 90), (55, 55, 55), 2)
    cv2.line(image, (30, 185), (width - 35, 185), (55, 55, 55), 2)
    cv2.putText(
        image,
        "MANGA",
        (42, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return image


def test_final_quality_clean_page_is_quiet():
    image = manga_page()
    quality = final_quality_checks(image, before_enhance=image.copy())
    assert quality["reasons"] == []


def test_final_quality_flags_residual_glare():
    image = manga_page()
    cv2.circle(image, (120, 130), 20, (255, 255, 255), -1)
    quality = final_quality_checks(image)
    assert "final_glare_residual" in quality["reasons"]
    assert quality["metrics"]["glare_fraction"] > 0


def test_final_quality_flags_unresolved_finger_and_high_repair_residual():
    image = manga_page()
    repair = {
        "status": "incomplete",
        "unresolved_mask": "debug/unresolved.png",
        "components": [
            {
                "component_id": 1,
                "donors": [
                    {
                        "candidate_id": 2,
                        "context_residual": 0.16,
                    }
                ],
            }
        ],
    }
    quality = final_quality_checks(image, finger_repair=repair)
    assert "final_unresolved_finger" in quality["reasons"]
    assert "final_finger_repair_residual" in quality["reasons"]


def test_final_quality_flags_large_white_normalization_change():
    before = manga_page()
    before[40:280, 35:205] = 210
    after = before.copy()
    after[40:280, 35:205] = 250

    quality = final_quality_checks(
        after,
        before_enhance=before,
        white_normalization=True,
    )

    assert "final_background_fill_large" in quality["reasons"]
    assert quality["metrics"]["whitened_fraction"] >= 0.30


def test_final_quality_flags_nearly_blank_white_and_black_pages():
    white = np.full((240, 180, 3), 255, np.uint8)
    black = np.zeros((240, 180, 3), np.uint8)

    assert "final_near_blank_white" in final_quality_checks(white)["reasons"]
    assert "final_near_blank_black" in final_quality_checks(black)["reasons"]


def test_final_quality_flags_long_crop_boundary_line():
    image = manga_page()
    image[:, :4] = 245
    cv2.line(image, (5, 0), (5, image.shape[0] - 1), (0, 0, 0), 3)

    quality = final_quality_checks(image)

    assert "final_edge_crop_suspected" in quality["reasons"]
    assert quality["metrics"]["edge_line_side"] == "left"


def test_adjacent_duplicate_check_requires_informative_content():
    cfg = Config()
    page = manga_page()
    same = page.copy()
    blank = np.full_like(page, 255)

    assert adjacent_quality_check(page, same, cfg)["suspect"] is True
    assert adjacent_quality_check(blank, blank.copy(), cfg)["suspect"] is False


def test_dewarp_regression_warns_when_straight_structure_is_destroyed():
    before = manga_page(360, 260)
    for x in range(30, 240, 35):
        cv2.line(before, (x, 25), (x, 335), (35, 35, 35), 2)

    # Deliberately destroy the straight-line evidence while preserving enough
    # image content to avoid the near-blank checks.
    after = cv2.GaussianBlur(before, (0, 0), 12)
    cv2.circle(after, (130, 180), 45, (80, 80, 80), -1)

    quality = final_quality_checks(
        after,
        before_dewarp=before,
        dewarp={"applied": True},
    )

    assert "final_dewarp_line_regression" in quality["reasons"]
