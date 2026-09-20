import json

import pytest

from manga_scan.config import Config
from manga_scan.manifest_migrations import CURRENT_MANIFEST_VERSION, migrate_manifest
from manga_scan.storage import read_manifest


def legacy_manifest(version=2, **config):
    return {
        "version": version,
        "source": "/tmp/book.mp4",
        "metadata": {"duration": 10},
        "config": {
            "hand_backend": "mediapipe",
            "rotation": 0,
            "dewarp_strength": 0.0,
            **config,
        },
        "spreads": [],
        "pages": [],
    }


def test_v1_migration_preserves_pre_feature_rendering_semantics():
    source = legacy_manifest(version=1)
    migrated = migrate_manifest(source)

    assert migrated["version"] == CURRENT_MANIFEST_VERSION
    assert migrated["config"]["output_layout"] == "split"
    assert migrated["config"]["candidate_selection_mode"] == "spread"
    assert migrated["config"]["perspective_mode"] == "spread"
    assert migrated["config"]["page_background_fill"] == "preserve"
    assert migrated["config"]["finger_repair"] is False
    assert migrated["config"]["glare_repair"] is False
    assert migrated["config"]["glare_overlap_weight"] == 0.0
    assert migrated["config"]["illumination_correction"] is False
    assert migrated["config"]["white_normalization"] is False
    assert migrated["config"]["auto_rotation"] is False
    assert migrated["config"]["dewarp_mode"] == "off"
    # Migration is pure; callers decide when a migrated manifest is persisted.
    assert source["version"] == 1
    assert "output_layout" not in source["config"]
    Config.from_dict(migrated["config"])


def test_v2_migration_preserves_explicit_newer_settings():
    source = legacy_manifest(
        version=2,
        output_layout="spread",
        candidate_selection_mode="per_page",
        perspective_mode="per_page",
        page_background_fill="paper",
        finger_repair=True,
        glare_repair=True,
        finger_repair_fallback="paper",
        glare_overlap_weight=6.0,
        suspect_glare_overlap=0.01,
        illumination_correction=True,
        white_normalization=True,
        auto_rotation=True,
        dewarp_mode="auto",
    )

    migrated = migrate_manifest(source)

    for key, value in source["config"].items():
        assert migrated["config"][key] == value
    assert migrated["version"] == CURRENT_MANIFEST_VERSION
    Config.from_dict(migrated["config"])


def test_v2_legacy_dewarp_strength_maps_to_manual_mode():
    migrated = migrate_manifest(legacy_manifest(version=2, dewarp_strength=0.2))
    assert migrated["config"]["dewarp_mode"] == "manual"


def test_current_manifest_is_returned_as_an_equal_copy():
    source = legacy_manifest(version=CURRENT_MANIFEST_VERSION, output_layout="spread")
    migrated = migrate_manifest(source)
    assert migrated == source
    assert migrated is not source
    assert migrated["config"] is not source["config"]


@pytest.mark.parametrize("version", [0, CURRENT_MANIFEST_VERSION + 1, "2"])
def test_manifest_migration_rejects_unsupported_versions(version):
    source = legacy_manifest(version=2)
    source["version"] = version
    with pytest.raises(ValueError, match="manifest version"):
        migrate_manifest(source)


def test_read_manifest_migrates_in_memory_without_rewriting_disk(tmp_path):
    raw = legacy_manifest(version=2)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))

    loaded = read_manifest(tmp_path)

    assert loaded["version"] == CURRENT_MANIFEST_VERSION
    assert loaded["config"]["output_layout"] == "split"
    assert json.loads(path.read_text())["version"] == 2
