"""Tests for the KG extraction evaluation harness.

These tests verify:
1. Golden case loading and parsing
2. Normalisation and matching logic
3. Metric computation (precision, recall, F1)
4. False positive / false negative identification
5. Aggregate evaluation across multiple cases

These tests DO NOT require an LLM — they test the harness logic itself.
LLM-based evaluation is done separately via scripts or manual runs.
"""

import json
import math
from pathlib import Path

import pytest

from tests.eval.harness import (
    AggregateResult,
    EvaluationHarness,
    EvaluationResult,
    GoldenCase,
    MetricResult,
    _compute_metrics,
    entity_match_key,
    normalise,
    relationship_match_key,
)


# ------------------------------------------------------------------
# Normalisation tests
# ------------------------------------------------------------------


class TestNormalise:
    def test_case_folding(self):
        assert normalise("Margaret Hamilton") == "margaret hamilton"

    def test_whitespace_collapse(self):
        # Note: "MIT" gets alias-expanded to "massachusetts institute of technology"
        assert normalise("  MIT   Draper  Laboratory  ") == (
            "massachusetts institute of technology draper laboratory"
        )

    def test_alias_expansion(self):
        assert normalise("She worked at MIT") == "she worked at massachusetts institute of technology"

    def test_alias_whole_word_only(self):
        # "emit" should NOT have "mit" replaced
        assert "massachusetts" not in normalise("emit signal")

    def test_mixed_aliases(self):
        result = normalise("AI and ML are subfields of CS")
        assert "artificial intelligence" in result
        assert "machine learning" in result


class TestEntityMatchKey:
    def test_basic(self):
        assert entity_match_key("Margaret Hamilton", "Person") == (
            "margaret hamilton::person"
        )

    def test_case_insensitive(self):
        assert entity_match_key("MARGARET HAMILTON", "PERSON") == (
            entity_match_key("margaret hamilton", "person")
        )

    def test_whitespace_tolerant(self):
        assert entity_match_key("  Margaret  Hamilton  ", "  Person  ") == (
            entity_match_key("Margaret Hamilton", "Person")
        )


class TestRelationshipMatchKey:
    def test_basic(self):
        key = relationship_match_key("Hamilton", "Apollo", "developed software for")
        assert key == "hamilton→apollo:developed software for"

    def test_case_insensitive(self):
        assert relationship_match_key("HAMILTON", "APOLLO", "DEVELOPED") == (
            relationship_match_key("hamilton", "apollo", "developed")
        )

    def test_direction_matters(self):
        key1 = relationship_match_key("A", "B", "created")
        key2 = relationship_match_key("B", "A", "created")
        assert key1 != key2  # direction should matter


# ------------------------------------------------------------------
# Metric computation tests
# ------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_match(self):
        metrics, fp, fn = _compute_metrics(
            {"a::person", "b::org"},
            {"a::person", "b::org"},
            "entities",
            ["a::person", "b::org"],
            ["a::person", "b::org"],
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.matched_count == 2
        assert fp == []
        assert fn == []

    def test_no_match(self):
        metrics, fp, fn = _compute_metrics(
            {"x::concept"},
            {"a::person", "b::org"},
            "entities",
            ["x::concept"],
            ["a::person", "b::org"],
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.matched_count == 0
        assert len(fp) == 1
        assert len(fn) == 2

    def test_partial_match(self):
        metrics, fp, fn = _compute_metrics(
            {"a::person", "c::concept"},
            {"a::person", "b::org"},
            "entities",
            ["a::person", "c::concept"],
            ["a::person", "b::org"],
        )
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(0.5)
        assert metrics.matched_count == 1

    def test_superset_prediction(self):
        """Predicted more than golden — precision < 1 but recall = 1."""
        metrics, _, _ = _compute_metrics(
            {"a::person", "b::org", "c::concept"},
            {"a::person", "b::org"},
            "entities",
            ["a::person", "b::org", "c::concept"],
            ["a::person", "b::org"],
        )
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.recall == 1.0
        assert metrics.matched_count == 2

    def test_empty_predicted(self):
        metrics, fp, fn = _compute_metrics(
            set(),
            {"a::person"},
            "entities",
            [],
            ["a::person"],
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_empty_golden(self):
        """If golden is empty and predicted is non-empty, precision = 0."""
        metrics, _, _ = _compute_metrics(
            {"a::person"},
            set(),
            "entities",
            ["a::person"],
            [],
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0  # nothing to recall

    def test_both_empty(self):
        metrics, fp, fn = _compute_metrics(
            set(), set(), "entities", [], []
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert fp == []
        assert fn == []


# ------------------------------------------------------------------
# GoldenCase tests
# ------------------------------------------------------------------


class TestGoldenCase:
    def test_load_hamilton(self, tmp_path):
        """Golden case loading should parse JSON correctly."""
        golden_data = {
            "text": "Margaret Hamilton led the team.",
            "entities": [
                {
                    "entity_name": "Margaret Hamilton",
                    "entity_type": "Person",
                    "entity_description": "Software engineer.",
                }
            ],
            "relationships": [
                {
                    "source_entity": "Margaret Hamilton",
                    "target_entity": "Team",
                    "relation": "led",
                    "relationship_description": "Margaret Hamilton led the team.",
                }
            ],
        }
        golden_file = tmp_path / "hamilton.json"
        golden_file.write_text(json.dumps(golden_data), encoding="utf-8")

        # Monkeypatch the golden dir
        import tests.eval.harness as h
        original_dir = h._GOLDEN_DIR
        h._GOLDEN_DIR = tmp_path
        try:
            case = GoldenCase.load("hamilton")
            assert case.key == "hamilton"
            assert "Margaret Hamilton" in case.text
            assert len(case.entities) == 1
            assert case.entities[0] == ("Margaret Hamilton", "Person", "Software engineer.")
            assert len(case.relationships) == 1
            assert case.relationships[0] == (
                "Margaret Hamilton", "Team", "led", "Margaret Hamilton led the team."
            )
        finally:
            h._GOLDEN_DIR = original_dir

    def test_from_dict(self):
        data = {
            "text": "Test text",
            "entities": [
                {"entity_name": "A", "entity_type": "Person", "entity_description": "Desc A"},
            ],
            "relationships": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relation": "knows",
                    "relationship_description": "A knows B",
                },
            ],
        }
        case = GoldenCase.from_dict("test", data)
        assert case.key == "test"
        assert case.text == "Test text"
        assert case.entities == [("A", "Person", "Desc A")]
        assert case.relationships == [("A", "B", "knows", "A knows B")]

    def test_load_nonexistent(self, tmp_path):
        import tests.eval.harness as h
        original_dir = h._GOLDEN_DIR
        h._GOLDEN_DIR = tmp_path
        try:
            with pytest.raises(FileNotFoundError, match="Golden case not found"):
                GoldenCase.load("nonexistent")
        finally:
            h._GOLDEN_DIR = original_dir


# ------------------------------------------------------------------
# Full harness evaluation tests
# ------------------------------------------------------------------


class TestEvaluationHarness:
    def _make_hamilton_result(self):
        """Create a near-perfect prediction for the Hamilton golden case."""
        return EvaluationHarness().evaluate(
            predicted_entities=[
                ("Margaret Hamilton", "Person", "Software engineer who led the software engineering division."),
                ("MIT Draper Laboratory", "Organization", "The laboratory where Hamilton worked."),
                ("Apollo Program", "Event", "NASA's human spaceflight program."),
                ("Software Engineering", "Concept", "A term coined by Hamilton."),
                ("Barack Obama", "Person", "President who awarded the Medal of Freedom."),
            ],
            predicted_relationships=[
                ("Margaret Hamilton", "MIT Draper Laboratory", "led division at", "Hamilton led the division."),
                ("Margaret Hamilton", "Apollo Program", "developed software for", "Hamilton developed flight software."),
                ("Margaret Hamilton", "Software Engineering", "coined term", "Hamilton coined the term."),
            ],
            golden_key="hamilton",
        )

    def test_evaluate_hamilton_perfect(self):
        """Perfect match should give P=R=F1=1.0."""
        harness = EvaluationHarness()
        golden = GoldenCase.load("hamilton")

        result = harness.evaluate(
            predicted_entities=golden.entities,
            predicted_relationships=golden.relationships,
            golden_key="hamilton",
        )
        assert result.entity_metrics.precision == 1.0
        assert result.entity_metrics.recall == 1.0
        assert result.entity_metrics.f1 == 1.0
        assert result.relationship_metrics.precision == 1.0
        assert result.relationship_metrics.recall == 1.0

    def test_evaluate_hamilton_partial(self):
        """Partial match — some entities missed."""
        result = self._make_hamilton_result()
        # We predicted 5 entities, golden has 6 — missing "Presidential Medal of Freedom"
        assert result.entity_metrics.golden_count == 6
        # Should have reasonable recall
        assert result.entity_metrics.recall > 0.5

    def test_evaluate_empty_prediction(self):
        """Empty prediction should give zero recall and F1."""
        harness = EvaluationHarness()
        result = harness.evaluate(
            predicted_entities=[],
            predicted_relationships=[],
            golden_key="hamilton",
        )
        assert result.entity_metrics.recall == 0.0
        assert result.entity_metrics.precision == 0.0
        assert result.entity_metrics.f1 == 0.0
        assert result.relationship_metrics.recall == 0.0

    def test_summary_output(self):
        result = self._make_hamilton_result()
        summary = result.summary()
        assert "hamilton" in summary
        assert "Entities:" in summary or "entities" in summary

    def test_false_positives_negatives(self):
        harness = EvaluationHarness()
        result = harness.evaluate(
            predicted_entities=[
                ("NONEXISTENT ENTITY", "Concept", "Made up."),
                ("Margaret Hamilton", "Person", "Software engineer."),
            ],
            predicted_relationships=[
                ("NONEXISTENT SOURCE", "NONEXISTENT TARGET", "fake relation", "Made up."),
            ],
            golden_key="hamilton",
        )
        # Should have at least one false positive
        assert len(result.entity_false_positives) >= 1
        # Should have false negatives (most golden entities not predicted)
        assert len(result.entity_false_negatives) >= 1

    def test_normalisation_in_matching(self):
        """Case/whitespace differences should be tolerated."""
        harness = EvaluationHarness()

        # Predict entities with different casing
        result = harness.evaluate(
            predicted_entities=[
                ("MARGARET HAMILTON", "person", "Software engineer."),
                ("mit draper laboratory", "ORGANIZATION", "The lab."),
            ],
            predicted_relationships=[
                ("Margaret Hamilton", "MIT Draper Laboratory", "led division at", "Led the div."),
            ],
            golden_key="hamilton",
        )
        # These should be normalised to match the golden data
        # At minimum, the two entities we predicted should match
        assert result.entity_metrics.matched_count >= 2


class TestAggregateResult:
    def test_summary_format(self):
        agg = AggregateResult(
            num_cases=3,
            entity_precision=0.85,
            entity_recall=0.78,
            entity_f1=0.81,
            relationship_precision=0.72,
            relationship_recall=0.65,
            relationship_f1=0.68,
        )
        summary = agg.summary()
        assert "3 cases" in summary
        assert "85.00%" in summary
        assert "78.00%" in summary