"""Query service — thin wrapper around the GraphRAGQueryEngine.

Delegates to ``app.state.engine`` for backward compatibility.
"""

import logging

logger = logging.getLogger(__name__)


class QueryService:
    """Service that forwards queries to the active query engine."""

    def __init__(self, state=None):
        """Initialise with a reference to the app state object.

        Parameters
        ----------
        state:
            The ``app.state`` object (or any object with attributes
            ``engine`` and ``summaries_loaded``).  Can be ``None``;
            call :meth:`attach_state` later.
        """
        self._state = state

    def attach_state(self, state) -> None:
        """(Re-)attach an app-state reference after initialization."""
        self._state = state

    def query(
        self,
        query_str: str,
        similarity_top_k: int = 20,
    ) -> dict:
        """Execute an async-compatible query against the knowledge graph.

        Returns a dict with keys ``answer``, ``communities_consulted``,
        ``entities_found`` on success.  Raises ``RuntimeError`` if the
        engine is not ready.

        .. note::
            This is a *synchronous* helper — the actual LLM call should
            be awaited in the route handler via
            ``engine.acustom_query()``.
        """
        if self._state is None or not hasattr(self._state, "engine"):
            raise RuntimeError("Engine not initialized")

        if not getattr(self._state, "summaries_loaded", False):
            raise RuntimeError(
                "No data ingested. Run 'sb ingest <directory>' first."
            )

        # Update similarity_top_k on the engine instance
        self._state.engine.similarity_top_k = similarity_top_k

        return {
            "engine": self._state.engine,
            "query_str": query_str,
        }

    @staticmethod
    def format_response(response) -> dict:
        """Format a ``Response`` object from the query engine into a dict."""
        return {
            "answer": response.response,
            "communities_consulted": response.metadata.get(
                "communities_consulted", []
            ),
            "entities_found": response.metadata.get("entities_found", []),
        }