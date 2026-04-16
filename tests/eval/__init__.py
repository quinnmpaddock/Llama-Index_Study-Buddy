"""Evaluation harness for KG extraction quality.

Provides programmatic evaluation of extraction output against golden test sets.
See ``harness.py`` for the main EvaluationHarness class.

Usage::

    from tests.eval import EvaluationHarness, GoldenCase

    harness = EvaluationHarness()
    result = harness.evaluate(predicted_entities, predicted_rels, "hamilton")
    print(result.summary())
"""

from tests.eval.harness import (
    AggregateResult,
    EvaluationHarness,
    EvaluationResult,
    GoldenCase,
    MetricResult,
    entity_match_key,
    normalise,
    relationship_match_key,
)

__all__ = [
    "EvaluationHarness",
    "GoldenCase",
    "EvaluationResult",
    "AggregateResult",
    "MetricResult",
    "normalise",
    "entity_match_key",
    "relationship_match_key",
]