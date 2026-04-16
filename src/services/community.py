"""Community summary persistence and versioning service.

Handles loading, saving, listing, and cleaning up community summary
snapshots.  Works with both the legacy ``summaries/`` directory and
the ``data/default/summaries/`` path.
"""

import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root (sibling of src/) — services/ is two levels deep: src/services/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SUMMARIES_DIR = str(_PROJECT_ROOT / "summaries")


class CommunityService:
    """Manage community summary files and versions."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialise the service.

        Parameters
        ----------
        data_dir:
            Root data directory.  If *None*, the legacy ``summaries/``
            path is used for backward compatibility.
        """
        self.data_dir = data_dir  # may be None → use legacy path

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _summaries_dir(self) -> str:
        """Return the summaries directory.

        If *data_dir* is set, uses ``<data_dir>/default/summaries/``.
        Otherwise falls back to the legacy ``summaries/`` directory.
        """
        if self.data_dir:
            path = os.path.join(self.data_dir, "default", "summaries")
        else:
            path = _DEFAULT_SUMMARIES_DIR

        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Snapshot helpers (extracted from app.py)
    # ------------------------------------------------------------------

    def find_most_recent_snapshot(self) -> Optional[str]:
        """Scan for the most recent *complete* snapshot pair.

        A complete snapshot has both ``community_summaries_<version>.json``
        and ``entity_info_<version>.json``.

        Returns the version string (e.g. ``2026-04-08_150308``) or
        ``None`` if no complete pair exists.
        """
        summaries_dir = self._summaries_dir()

        entity_files = glob.glob(
            os.path.join(summaries_dir, "entity_info_*.json")
        )
        summary_files = glob.glob(
            os.path.join(summaries_dir, "community_summaries_*.json")
        )

        if not entity_files or not summary_files:
            return None

        def _parse_timestamp(filepath: str) -> Optional[datetime]:
            name = os.path.basename(filepath)
            # Remove extension and split
            parts = name.replace(".json", "").split("_")
            # parts for "entity_info_2026-04-08_150308":
            #   ['entity', 'info', '2026-04-08', '150308']
            # parts for "community_summaries_2026-04-08_150308":
            #   ['community', 'summaries', '2026-04-08', '150308']
            if len(parts) >= 4:
                date_str = parts[-2]
                time_str = parts[-1]
                try:
                    return datetime.strptime(
                        f"{date_str}_{time_str}", "%Y-%m-%d_%H%M%S"
                    )
                except ValueError:
                    return None
            return None

        entity_versions: Dict[datetime, str] = {}
        for f in entity_files:
            ts = _parse_timestamp(f)
            if ts:
                entity_versions[ts] = f

        summary_versions: Dict[datetime, str] = {}
        for f in summary_files:
            ts = _parse_timestamp(f)
            if ts:
                summary_versions[ts] = f

        common_timestamps = set(entity_versions.keys()) & set(
            summary_versions.keys()
        )

        if not common_timestamps:
            return None

        most_recent = max(common_timestamps)
        return most_recent.strftime("%Y-%m-%d_%H%M%S")

    def load_summaries_and_entity_info(
        self,
    ) -> Tuple[Dict, Dict]:
        """Load community summaries and entity info.

        Priority:
        1. ``current.json`` pointer (explicit pin)
        2. Most recent complete snapshot (auto-detect)
        3. Legacy un-versioned files (``community_summaries.json`` /
           ``entity_info.json``)

        Returns ``(community_summaries, entity_info)`` — either as loaded
        data or empty dicts when nothing is found.
        """
        summaries_dir = self._summaries_dir()
        current_path = os.path.join(summaries_dir, "current.json")

        def _load_version(version: str, source: str = "versioned"):
            summary_file = os.path.join(
                summaries_dir, f"community_summaries_{version}.json"
            )
            entity_file = os.path.join(
                summaries_dir, f"entity_info_{version}.json"
            )

            if not (os.path.exists(summary_file) and os.path.exists(entity_file)):
                return None, None

            with open(summary_file, "r", encoding="utf-8") as f:
                raw_summaries = json.load(f)
            with open(entity_file, "r", encoding="utf-8") as f:
                entity_info = json.load(f)

            community_summaries = {int(k): v for k, v in raw_summaries.items()}
            logger.info(
                "Loaded %d community summaries (%s: %s).",
                len(community_summaries),
                source,
                version,
            )
            logger.info("Loaded %d entity mappings.", len(entity_info))
            return community_summaries, entity_info

        # 1. Try current.json
        if os.path.exists(current_path):
            logger.info("Found current.json pointer...")
            with open(current_path, "r", encoding="utf-8") as f:
                current_info = json.load(f)
            version = current_info.get("version")
            if version:
                summaries, entities = _load_version(version, source="pinned")
                if summaries is not None:
                    return summaries, entities
                logger.warning(
                    "current.json points to version %s, but files not found. "
                    "Falling back to scan.",
                    version,
                )

        # 2. Scan for most recent snapshot
        most_recent_version = self.find_most_recent_snapshot()
        if most_recent_version:
            logger.info(
                "Auto-detected most recent snapshot: %s", most_recent_version
            )
            summaries, entities = _load_version(
                most_recent_version, source="auto-detected"
            )
            if summaries is not None:
                return summaries, entities

        # 3. Fall back to legacy files
        legacy_summaries = os.path.join(summaries_dir, "community_summaries.json")
        legacy_entity = os.path.join(summaries_dir, "entity_info.json")
        if os.path.exists(legacy_summaries) and os.path.exists(legacy_entity):
            logger.info("Loading legacy summary files...")
            with open(legacy_summaries, "r", encoding="utf-8") as f:
                raw_summaries = json.load(f)
            with open(legacy_entity, "r", encoding="utf-8") as f:
                entity_info = json.load(f)
            community_summaries = {int(k): v for k, v in raw_summaries.items()}
            logger.info(
                "Loaded %d community summaries from legacy files.",
                len(community_summaries),
            )
            logger.info("Loaded %d entity mappings.", len(entity_info))
            return community_summaries, entity_info

        # Nothing found
        logger.warning(
            "No summary files found in %s. Starting with empty knowledge graph.",
            summaries_dir,
        )
        return {}, {}

    # ------------------------------------------------------------------
    # Save / persist
    # ------------------------------------------------------------------

    def save_summaries(
        self,
        community_summaries: Dict,
        entity_info: Dict,
        version: Optional[str] = None,
    ) -> str:
        """Save community summaries and entity info to disk.

        Returns the version string used for the file names.
        """
        summaries_dir = self._summaries_dir()

        if version is None:
            version = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Write versioned files
        summary_path = os.path.join(
            summaries_dir, f"community_summaries_{version}.json"
        )
        entity_info_path = os.path.join(
            summaries_dir, f"entity_info_{version}.json"
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(community_summaries, f, indent=4)
        logger.info("Community summaries saved to %s", summary_path)

        with open(entity_info_path, "w", encoding="utf-8") as f:
            json.dump(entity_info, f, indent=4)
        logger.info("Entity info saved to %s", entity_info_path)

        # Update current.json pointer
        current_path = os.path.join(summaries_dir, "current.json")
        current_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "files": {
                "community_summaries": f"community_summaries_{version}.json",
                "entity_info": f"entity_info_{version}.json",
            },
            "stats": {
                "total_entities": len(entity_info),
                "total_communities": len(community_summaries),
            },
        }
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(current_info, f, indent=4)
        logger.info("Current version updated to %s", version)

        return version

    # ------------------------------------------------------------------
    # Version listing / retrieval
    # ------------------------------------------------------------------

    def list_versions(
        self,
    ) -> Tuple[Optional[dict], List[dict]]:
        """Return ``(current_version_info, list_of_version_dicts)``.

        Each version dict has keys ``version``, ``filename``, ``modified``,
        ``size_bytes``.
        """
        summaries_dir = self._summaries_dir()

        pattern = os.path.join(summaries_dir, "community_summaries_*.json")
        summary_files = glob.glob(pattern)

        versions: List[dict] = []
        for f in summary_files:
            filename = os.path.basename(f)
            parts = filename.replace("community_summaries_", "").replace(
                ".json", ""
            )
            stat = os.stat(f)
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            versions.append(
                {
                    "version": parts,
                    "filename": filename,
                    "modified": mtime,
                    "size_bytes": stat.st_size,
                }
            )

        # Sort newest first
        versions.sort(key=lambda x: x["version"], reverse=True)

        # Current version
        current = None
        current_path = os.path.join(summaries_dir, "current.json")
        if os.path.exists(current_path):
            with open(current_path, "r", encoding="utf-8") as f:
                current = json.load(f)

        return current, versions

    def get_current_version(self) -> Optional[dict]:
        """Return the content of ``current.json`` or ``None``."""
        current_path = os.path.join(
            self._summaries_dir(), "current.json"
        )
        if not os.path.exists(current_path):
            return None
        with open(current_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_version(
        self,
        version: str,
    ) -> Optional[dict]:
        """Return ``{version, community_summaries, entity_info}`` or ``None``."""
        summaries_dir = self._summaries_dir()
        summary_file = os.path.join(
            summaries_dir, f"community_summaries_{version}.json"
        )
        entity_file = os.path.join(
            summaries_dir, f"entity_info_{version}.json"
        )

        if not os.path.exists(summary_file):
            return None

        result: dict = {"version": version}
        with open(summary_file, "r", encoding="utf-8") as f:
            result["community_summaries"] = json.load(f)
        if os.path.exists(entity_file):
            with open(entity_file, "r", encoding="utf-8") as f:
                result["entity_info"] = json.load(f)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_versions(
        self,
        keep: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """Delete old summary versions, keeping *keep* newest.

        Returns ``(deleted_filenames, kept_filenames)``.
        Also updates ``current.json`` if the current version is deleted.
        """
        if keep < 1:
            raise ValueError("Must keep at least 1 version")

        summaries_dir = self._summaries_dir()

        pattern = os.path.join(summaries_dir, "community_summaries_*.json")
        summary_files = glob.glob(pattern)

        # Build (version, filepath) pairs and sort newest first
        versions: List[Tuple[str, str]] = []
        for f in summary_files:
            filename = os.path.basename(f)
            version = filename.replace("community_summaries_", "").replace(
                ".json", ""
            )
            versions.append((version, f))
        versions.sort(key=lambda x: x[0], reverse=True)

        to_delete = versions[keep:]
        to_keep = versions[:keep]

        deleted: List[str] = []
        for _version, filepath in to_delete:
            summary_file = filepath
            entity_file = filepath.replace(
                "community_summaries_", "entity_info_"
            )

            try:
                os.remove(summary_file)
                deleted.append(os.path.basename(summary_file))
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", summary_file, exc)

            if os.path.exists(entity_file):
                try:
                    os.remove(entity_file)
                    deleted.append(os.path.basename(entity_file))
                except OSError as exc:
                    logger.warning("Failed to delete %s: %s", entity_file, exc)

        # Patch current.json if it was pointing to a deleted version
        current_path = os.path.join(summaries_dir, "current.json")
        if os.path.exists(current_path):
            with open(current_path, "r", encoding="utf-8") as f:
                current_data = json.load(f)

            current_version = current_data.get("version", "")
            deleted_versions = {v for v, _ in to_delete}

            if current_version in deleted_versions:
                if to_keep:
                    newest_version = to_keep[0][0]
                    new_current = {
                        "version": newest_version,
                        "created_at": datetime.now().isoformat(),
                        "files": {
                            "community_summaries": f"community_summaries_{newest_version}.json",
                            "entity_info": f"entity_info_{newest_version}.json",
                        },
                    }
                    # Load stats if possible
                    entity_file = os.path.join(
                        summaries_dir,
                        f"entity_info_{newest_version}.json",
                    )
                    if os.path.exists(entity_file):
                        with open(entity_file, "r", encoding="utf-8") as f:
                            entity_data = json.load(f)
                            new_current["stats"] = {
                                "total_entities": len(entity_data),
                                "total_communities": 0,
                            }
                    with open(current_path, "w", encoding="utf-8") as f:
                        json.dump(new_current, f, indent=4)
                else:
                    os.remove(current_path)

        kept_files = [os.path.basename(f) for _, f in to_keep]
        return deleted, kept_files