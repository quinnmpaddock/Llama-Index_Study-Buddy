"""Pydantic models for structured KG extraction with Instructor.

These models define the expected JSON schema for LLM extraction output.
When Instructor is enabled (``use_instructor=True`` in config), the LLM
response is validated against these models with automatic retries.

When Instructor is disabled, the legacy ``parse_fn`` / ``extract_json``
path is used instead.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A single entity extracted from text."""

    entity_name: str = Field(
        ...,
        description="Canonical name of the entity, capitalised.",
    )
    entity_type: str = Field(
        ...,
        description=(
            "Category of the entity. Must be one of: "
            "Person, Organization, Technology, Location, Event, "
            "Concept, Document, Product, Award, Other."
        ),
    )
    entity_description: str = Field(
        ...,
        description="Comprehensive description of the entity's attributes and activities.",
    )


class Relationship(BaseModel):
    """A directed relationship between two entities."""

    source_entity: str = Field(
        ...,
        description="Name of the source entity, exactly as listed in the entities list.",
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity, exactly as listed in the entities list.",
    )
    relation: str = Field(
        ...,
        description=(
            "Specific, descriptive verb phrase for the relationship. "
            "Prefer concrete verbs like 'developed', 'acquired', 'won' "
            "over vague ones like 'is related to' or 'involves'."
        ),
    )
    relationship_description: str = Field(
        ...,
        description="Brief explanation of why these entities are related.",
    )


class ExtractionResult(BaseModel):
    """Structured output of entity-relationship extraction.

    This is the top-level model that the LLM must produce.
    """

    entities: List[Entity] = Field(
        default_factory=list,
        description="All entities identified in the text.",
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="All relationships among the identified entities.",
    )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_tuples(
        self,
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
        """Convert to the tuple format expected by ``GraphRAGExtractor``.

        Returns
        -------
        (entities, relationships)
            entities    : list of (name, type, description)
            relationships : list of (source, target, relation, description)
        """
        entity_tuples = [
            (e.entity_name, e.entity_type, e.entity_description)
            for e in self.entities
        ]
        relationship_tuples = [
            (
                r.source_entity,
                r.target_entity,
                r.relation,
                r.relationship_description,
            )
            for r in self.relationships
        ]
        return entity_tuples, relationship_tuples