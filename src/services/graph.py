"""Graph entity and community query service.

Provides read-only access to entity & community data that lives in
``app.state`` attributes.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _make_summary_preview(summary: str, max_len: int = 100) -> str:
    """Create a meaningful preview from a community summary by stripping
    the intro sentence."""
    text = summary.strip()

    # Find the end of the first sentence
    first_sentence_end = -1
    for i, char in enumerate(text):
        if char == ".":
            if i + 1 >= len(text) or text[i + 1] in " \n":
                first_sentence_end = i + 1
                break

    if first_sentence_end > 0:
        remaining = text[first_sentence_end:].lstrip()
        if remaining:
            text = remaining

    if len(text) <= max_len:
        return text

    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    last_space = truncated.rfind(" ")

    if last_period > max_len * 0.5:
        text = text[: last_period + 1]
    elif last_space > max_len * 0.5:
        text = text[:last_space]

    return text[:max_len].strip() + "..."


class GraphService:
    """Read-only service for entity & community lookups."""

    def __init__(self, state=None):
        """Initialise with a reference to the app state object.

        Parameters
        ----------
        state:
            The ``app.state`` object (or any object with attributes
            ``entity_info`` and ``community_summaries``).  Can be
            ``None`` for testing; call :meth:`attach_state` later.
        """
        self._state = state

    def attach_state(self, state) -> None:
        """(Re-)attach an app-state reference after initialization."""
        self._state = state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_entity_info(self) -> dict:
        if self._state is None or not hasattr(self._state, "entity_info"):
            return {}
        return self._state.entity_info

    def _get_community_summaries(self) -> dict:
        if self._state is None or not hasattr(self._state, "community_summaries"):
            return {}
        return self._state.community_summaries

    # ------------------------------------------------------------------
    # Entity endpoints
    # ------------------------------------------------------------------

    def search_entities(
        self,
        query: Optional[str],
        limit: int = 50,
    ) -> dict:
        """Search entities by name (case-insensitive).

        Returns ``{"entities": [...], "total": int}``.
        """
        entity_info = self._get_entity_info()

        if query:
            q_lower = query.lower()
            matches = [
                {"name": name, "communities": list(set(communities))}
                for name, communities in entity_info.items()
                if q_lower in name.lower()
            ]
            matches.sort(key=lambda x: (x["name"].lower() != q_lower, len(x["name"])))
        else:
            matches = [
                {"name": name, "communities": list(set(communities))}
                for name, communities in entity_info.items()
            ]
            matches.sort(key=lambda x: x["name"].lower())

        return {"entities": matches[:limit], "total": len(matches)}

    def get_entity(self, name: str) -> Optional[dict]:
        """Get entity details by name (case-insensitive).

        Returns ``{"name": ..., "communities": [...]}`` or ``None``.
        """
        entity_info = self._get_entity_info()

        for entity_name, communities in entity_info.items():
            if entity_name.lower() == name.lower():
                return {"name": entity_name, "communities": list(set(communities))}
        return None

    # ------------------------------------------------------------------
    # Community endpoints
    # ------------------------------------------------------------------

    def list_communities(self) -> dict:
        """List all communities with entity counts.

        Returns ``{"communities": [...], "total": int}``.
        """
        summaries = self._get_community_summaries()
        entity_info = self._get_entity_info()

        # Build per-community entity lists
        community_entities: Dict[int, List[str]] = {}
        for entity_name, communities in entity_info.items():
            for comm_id in communities:
                if comm_id not in community_entities:
                    community_entities[comm_id] = []
                community_entities[comm_id].append(entity_name)

        communities = [
            {
                "id": int(comm_id_str),
                "entity_count": len(set(community_entities.get(int(comm_id_str), []))),
                "summary_preview": _make_summary_preview(summaries.get(comm_id_str, "")),
            }
            for comm_id_str in sorted(summaries.keys(), key=int)
        ]

        return {"communities": communities, "total": len(communities)}

    def get_community(self, id: int) -> Optional[dict]:
        """Get community details by ID.

        Returns ``{"id": ..., "summary": ..., "entity_count": ...}``
        or ``None`` if not found.
        """
        summaries = self._get_community_summaries()
        summary = summaries.get(str(id))
        if summary is None:
            return None

        entity_info = self._get_entity_info()
        entity_count = sum(1 for communities in entity_info.values() if id in communities)

        return {"id": id, "summary": summary, "entity_count": entity_count}

    def get_community_entities(
        self, id: int
    ) -> Optional[dict]:
        """Get entities belonging to a community.

        Returns ``{"community_id": ..., "entities": [...], "total": ...}``
        or ``None`` if the community doesn't exist.
        """
        summaries = self._get_community_summaries()
        if str(id) not in summaries:
            return None

        entity_info = self._get_entity_info()
        entities = sorted(set(name for name, communities in entity_info.items() if id in communities))

        return {"community_id": id, "entities": entities, "total": len(entities)}