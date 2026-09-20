"""Versioned project-manifest migrations.

Project manifests are persisted user data. Keep compatibility decisions here,
rather than letting current Config defaults silently change old projects.
"""

from __future__ import annotations

from copy import deepcopy

CURRENT_MANIFEST_VERSION = 3
OLDEST_MANIFEST_VERSION = 1


def _config(manifest):
    config = manifest.setdefault("config", {})
    if not isinstance(config, dict):
        raise ValueError("manifest config must be an object")
    return config


def _migrate_v1_to_v2(manifest):
    """Preserve the original split-output project semantics."""
    config = _config(manifest)
    # Version 1 predates whole-spread output. Existing projects were always
    # rendered as independent left/right pages.
    config.setdefault("output_layout", "split")
    manifest["version"] = 2
    return manifest


def _migrate_v2_to_v3(manifest):
    """Freeze behavior for settings introduced during the long-lived v2 era.

    Version 2 remained current while several opt-in algorithms were added.
    A missing key therefore means that the project was created before that
    behavior existed; it must not inherit a newer Config default on rerender.
    """
    config = _config(manifest)

    # Layout / candidate / geometry features that did not exist in early v2.
    config.setdefault("output_layout", "split")
    config.setdefault("candidate_selection_mode", "spread")
    config.setdefault("perspective_mode", "spread")
    config.setdefault("page_background_fill", "preserve")

    # Old projects had no multi-frame occlusion repair or glare scoring.
    config.setdefault("finger_repair", False)
    config.setdefault("glare_repair", False)
    config.setdefault("finger_repair_fallback", "preserve")
    config.setdefault("glare_overlap_weight", 0.0)
    config.setdefault("suspect_glare_overlap", 1.0)

    # Illumination and white-background normalization were added later.
    config.setdefault("illumination_correction", False)
    config.setdefault("white_normalization", False)

    # Auto rotation did not exist in early v2. Preserve the stored numeric
    # rotation exactly instead of reinterpreting zero as "auto".
    config.setdefault("auto_rotation", False)

    # Before dewarp_mode existed, dewarp_strength directly controlled the
    # legacy cylindrical correction.
    if "dewarp_mode" not in config:
        strength = float(config.get("dewarp_strength", 0.0))
        config["dewarp_mode"] = "manual" if strength > 0 else "off"

    manifest["version"] = 3
    return manifest


_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_manifest(manifest):
    """Return a current-version copy of *manifest* without rewriting disk.

    Missing version is treated as version 1 for pre-versioned/test fixtures.
    Future versions are rejected so an older binary never guesses how to read
    data written by a newer one.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")

    migrated = deepcopy(manifest)
    version = migrated.get("version", OLDEST_MANIFEST_VERSION)
    if type(version) is not int:
        raise ValueError("manifest version must be an integer")
    if version < OLDEST_MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {version}")
    if version > CURRENT_MANIFEST_VERSION:
        raise ValueError(
            f"manifest version {version} is newer than supported "
            f"version {CURRENT_MANIFEST_VERSION}"
        )

    while version < CURRENT_MANIFEST_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"no migration registered for manifest version {version}")
        migrated = migration(migrated)
        version = migrated.get("version")
        if type(version) is not int:
            raise ValueError("manifest migration produced an invalid version")

    return migrated
