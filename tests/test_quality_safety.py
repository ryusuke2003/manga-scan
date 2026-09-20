import pytest

from manga_scan.quality_safety import (
    normalize_expected_page_count,
    refresh_page_count_check,
    refresh_safe_fix_suggestions,
)


def test_expected_page_count_counts_spreads_as_two_physical_pages():
    manifest = {
        "status": "complete",
        "expected_page_count": 4,
        "pages": [
            {"id": "spread", "side": "spread", "enabled": True},
            {"id": "single", "side": "external", "enabled": True},
            {"id": "excluded", "side": "left", "enabled": False},
            {"id": "cover", "side": "cover", "enabled": True},
        ],
    }

    check = refresh_page_count_check(manifest)

    assert check == {
        "status": "match",
        "expected": 4,
        "actual": 4,
        "output_items": 3,
        "difference": 0,
    }


def test_expected_page_count_reports_short_and_waits_until_complete():
    manifest = {
        "status": "processing",
        "expected_page_count": 10,
        "pages": [{"id": "a", "side": "spread", "enabled": True}],
    }
    assert refresh_page_count_check(manifest)["status"] == "pending"

    manifest["status"] = "complete"
    check = refresh_page_count_check(manifest)
    assert check["status"] == "short"
    assert check["difference"] == -8


@pytest.mark.parametrize("value", [0, -1, 1.5, "1.5", True, 10001])
def test_expected_page_count_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_expected_page_count(value)


def test_safe_fix_suggests_only_better_existing_candidate_for_flagged_page():
    manifest = {
        "pages": [{
            "id": "s_left",
            "spread_id": "s",
            "side": "left",
            "enabled": True,
            "candidate_id": 0,
            "suspect": ["glare_overlap"],
            "final_quality": {"reasons": ["final_glare_residual"]},
        }],
        "spreads": [{
            "id": "s",
            "selected": 0,
            "selected_pages": {"left": 0, "right": 0},
            "candidates": [
                {
                    "id": 0,
                    "suspect": [],
                    "page_suspect": {"left": ["glare_overlap"], "right": []},
                    "metrics": {"score": 0.5},
                    "page_metrics": {
                        "left": {
                            "score": 0.5,
                            "selection_score": 0.45,
                            "glare_overlap": 0.02,
                            "hand_overlap": 0.0,
                        },
                        "right": {"score": 0.5},
                    },
                },
                {
                    "id": 1,
                    "suspect": [],
                    "page_suspect": {"left": [], "right": []},
                    "metrics": {"score": 0.7},
                    "page_metrics": {
                        "left": {
                            "score": 0.7,
                            "selection_score": 0.72,
                            "glare_overlap": 0.002,
                            "hand_overlap": 0.0,
                        },
                        "right": {"score": 0.6},
                    },
                },
                {
                    "id": 2,
                    "suspect": [],
                    "page_suspect": {"left": ["hand_overlap"], "right": []},
                    "metrics": {"score": 0.8},
                    "page_metrics": {
                        "left": {
                            "score": 0.8,
                            "selection_score": 0.80,
                            "glare_overlap": 0.001,
                            "hand_overlap": 0.03,
                        },
                        "right": {"score": 0.6},
                    },
                },
            ],
        }],
    }

    refresh_safe_fix_suggestions(manifest)

    suggestions = manifest["pages"][0]["safe_fix_suggestions"]
    assert [item["candidate_id"] for item in suggestions] == [1]
    assert suggestions[0]["side"] == "left"
    assert "glare" in suggestions[0]["improvements"]
    assert "risk_count" in suggestions[0]["improvements"]


def test_safe_fix_does_not_suggest_unflagged_page():
    manifest = {
        "pages": [{
            "id": "s_whole",
            "spread_id": "s",
            "side": "spread",
            "enabled": True,
            "candidate_id": 0,
            "suspect": [],
            "final_quality": {"reasons": []},
        }],
        "spreads": [{
            "id": "s",
            "selected": 0,
            "candidates": [
                {"id": 0, "metrics": {"selection_score": 0.3}, "suspect": []},
                {"id": 1, "metrics": {"selection_score": 0.9}, "suspect": []},
            ],
        }],
    }

    refresh_safe_fix_suggestions(manifest)

    assert "safe_fix_suggestions" not in manifest["pages"][0]
