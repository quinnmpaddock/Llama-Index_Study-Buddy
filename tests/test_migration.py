"""Tests for legacy summaries/ migration and data storage."""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from services.community import CommunityService


class TestMigrateLegacySummaries:
    """Tests for the _migrate_legacy_summaries function in app.py."""

    def test_migrate_copies_files_when_target_empty(self, tmp_path):
        """Legacy summaries/ files are copied to data/default/summaries/."""
        from app import _migrate_legacy_summaries

        # Create a fake legacy summaries directory
        legacy_dir = tmp_path / "summaries"
        legacy_dir.mkdir(parents=True)

        # Create target data dir (but empty default/summaries)
        data_dir = tmp_path / "data"

        # Write some files to legacy dir
        (legacy_dir / "community_summaries_2026-04-08_150308.json").write_text(
            json.dumps({"1": "test summary"})
        )
        (legacy_dir / "entity_info_2026-04-08_150308.json").write_text(
            json.dumps({"entity1": [1]})
        )
        (legacy_dir / "current.json").write_text(
            json.dumps({"version": "2026-04-08_150308"})
        )

        # Call migration with explicit legacy_dir (avoids hardcoded path resolution)
        _migrate_legacy_summaries(data_dir=data_dir, _legacy_dir=legacy_dir)

        # Verify migration happened
        target_summaries = data_dir / "default" / "summaries"
        assert target_summaries.exists(), f"Target summaries dir should exist at {target_summaries}"
        assert (target_summaries / "community_summaries_2026-04-08_150308.json").exists()
        assert (target_summaries / "entity_info_2026-04-08_150308.json").exists()
        assert (target_summaries / "current.json").exists()

        # Verify content was copied correctly
        with open(target_summaries / "community_summaries_2026-04-08_150308.json") as f:
            assert json.load(f) == {"1": "test summary"}

    def test_migrate_skips_when_target_has_data(self, tmp_path):
        """Migration should be skipped when target already has data."""
        from app import _migrate_legacy_summaries

        # Create target data directory with existing data
        data_dir = tmp_path / "data"
        target_summaries = data_dir / "default" / "summaries"
        target_summaries.mkdir(parents=True)
        (target_summaries / "existing_file.json").write_text("{}")

        # Create legacy dir (should be ignored since target has data)
        legacy_dir = tmp_path / "summaries"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "other_file.json").write_text("{}")

        _migrate_legacy_summaries(data_dir=data_dir, _legacy_dir=legacy_dir)

        # Target should still have only the original file (migration skipped)
        files_after = list(target_summaries.iterdir())
        assert len(files_after) == 1
        assert (target_summaries / "existing_file.json").exists()
        # Legacy file should NOT have been copied
        assert not (target_summaries / "other_file.json").exists()

    def test_community_service_default_storage(self, tmp_path):
        """CommunityService stores and retrieves data in default/summaries paths."""
        data_dir = tmp_path / "data"
        svc = CommunityService(data_dir=str(data_dir))

        # Save summaries
        svc.save_summaries(
            community_summaries={"1": "alpha community", "2": "beta"},
            entity_info={"entityA": [1, 2], "entityB": [2]},
            version="2026-04-13_120000",
        )

        # Verify files exist in default/summaries path
        assert (data_dir / "default" / "summaries" / "community_summaries_2026-04-13_120000.json").exists()
        assert (data_dir / "default" / "summaries" / "entity_info_2026-04-13_120000.json").exists()

        # Load and verify
        summaries, entities = svc.load_summaries_and_entity_info()
        assert 1 in summaries
        assert "entityA" in entities

    def test_community_service_list_versions(self, tmp_path):
        """list_versions returns versions stored in default/summaries."""
        data_dir = tmp_path / "data"
        svc = CommunityService(data_dir=str(data_dir))

        svc.save_summaries({"1": "s1"}, {"e1": [1]}, version="v_default")

        current, versions = svc.list_versions()
        assert current is not None
        assert current["version"] == "v_default"
        assert len(versions) == 1

    def test_community_service_cleanup(self, tmp_path):
        """cleanup_versions removes old versions."""
        data_dir = tmp_path / "data"
        svc = CommunityService(data_dir=str(data_dir))

        # Create multiple versions
        for v in ["v_old", "v_mid", "v_new"]:
            svc.save_summaries({"1": f"s_{v}"}, {"e1": [1]}, version=v)

        deleted, kept = svc.cleanup_versions(keep=2)
        assert len(deleted) >= 2  # entity_info + community_summaries for v_old