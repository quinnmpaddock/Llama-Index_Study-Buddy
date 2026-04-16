"""Tests for KG extraction Pydantic models and Instructor integration."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.extraction_models import Entity, ExtractionResult, Relationship


# ------------------------------------------------------------------
# Pydantic model tests
# ------------------------------------------------------------------


class TestEntityModel:
    def test_valid_entity(self):
        e = Entity(
            entity_name="Albert Einstein",
            entity_type="Person",
            entity_description="A theoretical physicist.",
        )
        assert e.entity_name == "Albert Einstein"
        assert e.entity_type == "Person"
        assert e.entity_description == "A theoretical physicist."

    def test_entity_from_dict(self):
        e = Entity(**{
            "entity_name": "Nobel Prize",
            "entity_type": "Award",
            "entity_description": "A prestigious award.",
        })
        assert e.entity_name == "Nobel Prize"

    def test_entity_requires_all_fields(self):
        with pytest.raises(Exception):
            Entity(entity_name="X")  # missing entity_type, entity_description


class TestRelationshipModel:
    def test_valid_relationship(self):
        r = Relationship(
            source_entity="Albert Einstein",
            target_entity="Theory of Relativity",
            relation="developed",
            relationship_description="Einstein developed the theory of relativity.",
        )
        assert r.source_entity == "Albert Einstein"
        assert r.relation == "developed"

    def test_relationship_from_dict(self):
        r = Relationship(**{
            "source_entity": "A",
            "target_entity": "B",
            "relation": "created",
            "relationship_description": "A created B.",
        })
        assert r.source_entity == "A"

    def test_relationship_requires_all_fields(self):
        with pytest.raises(Exception):
            Relationship(source_entity="A")  # missing fields


class TestExtractionResult:
    def test_empty_result(self):
        result = ExtractionResult()
        assert result.entities == []
        assert result.relationships == []

    def test_result_with_entities_and_relationships(self):
        entity = Entity(
            entity_name="Einstein",
            entity_type="Person",
            entity_description="Physicist.",
        )
        rel = Relationship(
            source_entity="Einstein",
            target_entity="Relativity",
            relation="developed",
            relationship_description="Einstein developed relativity.",
        )
        result = ExtractionResult(entities=[entity], relationships=[rel])
        assert len(result.entities) == 1
        assert len(result.relationships) == 1

    def test_result_from_dict(self):
        data = {
            "entities": [
                {
                    "entity_name": "Ada Lovelace",
                    "entity_type": "Person",
                    "entity_description": "First computer programmer.",
                }
            ],
            "relationships": [
                {
                    "source_entity": "Ada Lovelace",
                    "target_entity": "Analytical Engine",
                    "relation": "programmed",
                    "relationship_description": "Ada wrote algorithms for the Analytical Engine.",
                }
            ],
        }
        result = ExtractionResult(**data)
        assert len(result.entities) == 1
        assert result.entities[0].entity_name == "Ada Lovelace"
        assert len(result.relationships) == 1
        assert result.relationships[0].relation == "programmed"

    def test_result_defaults_to_empty(self):
        result = ExtractionResult(entities=[], relationships=[])
        assert result.entities == []
        assert result.relationships == []

    # ------------------------------------------------------------------
    # to_tuples() conversion
    # ------------------------------------------------------------------

    def test_to_tuples_basic(self):
        result = ExtractionResult(
            entities=[
                Entity(
                    entity_name="Einstein",
                    entity_type="Person",
                    entity_description="Physicist.",
                ),
                Entity(
                    entity_name="Relativity",
                    entity_type="Concept",
                    entity_description="A theory.",
                ),
            ],
            relationships=[
                Relationship(
                    source_entity="Einstein",
                    target_entity="Relativity",
                    relation="developed",
                    relationship_description="Einstein developed relativity.",
                ),
            ],
        )
        entity_tuples, rel_tuples = result.to_tuples()
        assert len(entity_tuples) == 2
        assert entity_tuples[0] == ("Einstein", "Person", "Physicist.")
        assert entity_tuples[1] == ("Relativity", "Concept", "A theory.")
        assert len(rel_tuples) == 1
        assert rel_tuples[0] == (
            "Einstein",
            "Relativity",
            "developed",
            "Einstein developed relativity.",
        )

    def test_to_tuples_empty(self):
        result = ExtractionResult()
        entity_tuples, rel_tuples = result.to_tuples()
        assert entity_tuples == []
        assert rel_tuples == []

    def test_to_tuples_compatible_with_parse_fn(self):
        """Verify to_tuples() produces the same format as parse_fn().

        This is critical: the tuple format must match what GraphRAGExtractor
        expects when creating EntityNode and Relation objects.
        """
        # Simulate LLM JSON output
        llm_json = {
            "entities": [
                {
                    "entity_name": "Quantum Mechanics",
                    "entity_type": "Concept",
                    "entity_description": "Branch of physics.",
                },
            ],
            "relationships": [
                {
                    "source_entity": "Quantum Mechanics",
                    "target_entity": "Physics",
                    "relation": "is a branch of",
                    "relationship_description": "Quantum mechanics is a branch of physics.",
                },
            ],
        }

        # Path 1: parse_fn (legacy)
        from services.ingestion import parse_fn
        legacy_entities, legacy_rels = parse_fn(json.dumps(llm_json))

        # Path 2: ExtractionResult.to_tuples() (Instructor)
        result = ExtractionResult(**llm_json)
        instructor_entities, instructor_rels = result.to_tuples()

        # Both paths must produce identical tuples
        assert legacy_entities == list(instructor_entities)
        assert legacy_rels == list(instructor_rels)


# ------------------------------------------------------------------
# GraphRAGExtractor integration tests
# ------------------------------------------------------------------


class TestGraphRAGExtractorInstructorMode:
    """Test the Instructor-based extraction path in GraphRAGExtractor.

    These tests avoid importing llama_index directly (which fails on NixOS
    due to libstdc++ issues).  Instead they mock the llama_index
    dependencies and test just our validation logic.
    """

    @pytest.fixture()
    def _mock_llama_index(self):
        """Mock llama_index.core imports so we can import GraphRAGExtractor
        without pulling in numpy's C extensions."""
        import types
        mock_modules = {}

        # Create mock modules for llama_index.core subpackages
        for mod_name in [
            "llama_index.core",
            "llama_index.core.async_utils",
            "llama_index.core.graph_stores.types",
            "llama_index.core.indices.property_graph.utils",
            "llama_index.core.llms",
            "llama_index.core.prompts",
            "llama_index.core.prompts.default_prompts",
            "llama_index.core.schema",
        ]:
            mock_modules[mod_name] = types.ModuleType(mod_name)

        # Add necessary attributes to mock modules
        mock_modules["llama_index.core"].Settings = MagicMock()
        mock_modules["llama_index.core"].async_utils = mock_modules["llama_index.core.async_utils"]
        mock_modules["llama_index.core.async_utils"].run_jobs = MagicMock()
        mock_modules["llama_index.core.graph_stores.types"].KG_NODES_KEY = "_kg_nodes"
        mock_modules["llama_index.core.graph_stores.types"].KG_RELATIONS_KEY = "_kg_relations"
        mock_modules["llama_index.core.graph_stores.types"].EntityNode = MagicMock
        mock_modules["llama_index.core.graph_stores.types"].Relation = MagicMock
        mock_modules["llama_index.core.indices.property_graph.utils"].default_parse_triplets_fn = lambda x: ([], [])
        mock_modules["llama_index.core.llms"].LLM = MagicMock
        mock_modules["llama_index.core.prompts"].PromptTemplate = MagicMock
        mock_modules["llama_index.core.prompts.default_prompts"].DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = "default"
        mock_modules["llama_index.core.schema"].BaseNode = MagicMock
        mock_modules["llama_index.core.schema"].MetadataMode = MagicMock()
        mock_modules["llama_index.core.schema"].MetadataMode.LLM = "llm"

        # TransformComponent is the base class for GraphRAGExtractor
        # We create a simple stand-in that stores kwargs as attributes
        class MockTransformComponent:
            __class_name__ = "TransformComponent"
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_modules["llama_index.core.schema"].TransformComponent = MockTransformComponent

        with patch.dict(sys.modules, mock_modules):
            yield

    def test_instructor_mode_requires_client(self, _mock_llama_index):
        """use_instructor=True without client should raise ValueError."""
        # Force reimport with mocked dependencies
        if "core.extractor" in sys.modules:
            del sys.modules["core.extractor"]

        from core.extractor import GraphRAGExtractor

        with pytest.raises(ValueError, match="instructor_client must be provided"):
            GraphRAGExtractor(use_instructor=True, instructor_client=None)

    def test_legacy_mode_default(self, _mock_llama_index):
        """Default should be legacy mode (use_instructor=False)."""
        if "core.extractor" in sys.modules:
            del sys.modules["core.extractor"]

        from core.extractor import GraphRAGExtractor

        extractor = GraphRAGExtractor(parse_fn=lambda x: ([], []))
        assert extractor.use_instructor is False
        assert extractor.instructor_client is None

    def test_instructor_mode_with_client(self, _mock_llama_index):
        """use_instructor=True with a client should initialize correctly."""
        if "core.extractor" in sys.modules:
            del sys.modules["core.extractor"]

        from core.extractor import GraphRAGExtractor

        mock_client = MagicMock()
        extractor = GraphRAGExtractor(
            use_instructor=True,
            instructor_client=mock_client,
            instructor_model_name="test-model",
            parse_fn=lambda x: ([], []),
        )
        assert extractor.use_instructor is True
        assert extractor.instructor_client is mock_client
        assert extractor.instructor_model_name == "test-model"


class TestCreateInstructorClient:
    """Test the create_instructor_client helper."""

    def test_create_client_imports_instructor(self):
        """Should import instructor lazily."""
        from core.extractor import _import_instructor

        # instructor may or may not be installed; just verify it's callable
        assert callable(_import_instructor)

    def test_create_client_imports_openai(self):
        """Should import openai lazily."""
        from core.extractor import _import_openai

        assert callable(_import_openai)


class TestExtractionResultEdgeCases:
    """Test edge cases in ExtractionResult."""

    def test_extra_fields_in_entities(self):
        """JSON with extra fields should still parse (Pydantic v2 strips by default)."""
        data = {
            "entities": [
                {
                    "entity_name": "Test",
                    "entity_type": "Concept",
                    "entity_description": "A test.",
                    "extra_field": "should be ignored",
                }
            ],
            "relationships": [],
        }
        result = ExtractionResult(**data)
        assert result.entities[0].entity_name == "Test"
        # extra_field is not present on the model
        assert not hasattr(result.entities[0], "extra_field")

    def test_missing_relationships_defaults_empty(self):
        """JSON missing 'relationships' key should default to empty list."""
        data = {
            "entities": [
                {
                    "entity_name": "Test",
                    "entity_type": "Concept",
                    "entity_description": "A test.",
                }
            ],
        }
        result = ExtractionResult(**data)
        assert result.relationships == []

    def test_json_round_trip(self):
        """ExtractionResult should survive JSON serialisation round-trip."""
        result = ExtractionResult(
            entities=[
                Entity(
                    entity_name="Python",
                    entity_type="Technology",
                    entity_description="A programming language.",
                ),
            ],
            relationships=[],
        )
        json_str = result.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)
        assert restored.entities[0].entity_name == "Python"
        assert restored.entities[0].entity_type == "Technology"