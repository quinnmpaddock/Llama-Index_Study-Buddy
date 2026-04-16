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

    This is the top-level model that the LLM must produce for
    single-pass extraction (the default path).
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


# ------------------------------------------------------------------
# Two-pass extraction models
# ------------------------------------------------------------------


class EntitiesOnlyResult(BaseModel):
    """Pass 1 of two-pass extraction: entities only.

    Used when ``use_two_pass=True``.  The LLM produces only entities
    in the first pass.  Relationships are extracted in a separate
    second pass that receives the entity list as context.
    """

    entities: List[Entity] = Field(
        default_factory=list,
        description="All entities identified in the text.",
    )

    def to_tuples(
        self,
    ) -> list[tuple[str, str, str]]:
        """Convert entities to the tuple format for LlamaIndex.

        Returns
        -------
        list of (name, type, description)
        """
        return [
            (e.entity_name, e.entity_type, e.entity_description)
            for e in self.entities
        ]

    def format_for_relationship_prompt(self) -> str:
        """Format entities as a human-readable list for the relationship prompt.

        Returns a string like::

            - Margaret Hamilton (Person): Software engineer...
            - MIT Draper Laboratory (Organization): The laboratory...
        """
        lines = []
        for e in self.entities:
            lines.append(f"- {e.entity_name} ({e.entity_type}): {e.entity_description}")
        return "\n".join(lines)


class RelationshipsOnlyResult(BaseModel):
    """Pass 2 of two-pass extraction: relationships only.

    The LLM receives the entity list and the original text, and
    produces only relationships referencing those entities.
    """

    relationships: List[Relationship] = Field(
        default_factory=list,
        description="All relationships among the provided entities, supported by the text.",
    )

    def to_tuples(
        self,
    ) -> list[tuple[str, str, str, str]]:
        """Convert relationships to the tuple format for LlamaIndex.

        Returns
        -------
        list of (source, target, relation, description)
        """
        return [
            (
                r.source_entity,
                r.target_entity,
                r.relation,
                r.relationship_description,
            )
            for r in self.relationships
        ]