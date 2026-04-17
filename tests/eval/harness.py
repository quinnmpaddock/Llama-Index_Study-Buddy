"""Evaluation harness for KG extraction quality.

Provides:
- Golden test set loading and parsing
- Quality metrics: precision, recall, F1 for entities and relationships
- Normalized matching (case, whitespace, partial match)
- Aggregate evaluation across multiple golden cases

Usage::

    from tests.eval.harness import EvaluationHarness

    harness = EvaluationHarness()
    results = harness.evaluate(
        predicted_entities=[(name, type, desc), ...],
        predicted_relationships=[(src, tgt, rel, desc), ...],
        golden_key="hamilton",
    )
    print(results.summary())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_ALIASES: Dict[str, str] = {
    "us": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "mit": "massachusetts institute of technology",
    "nasa": "national aeronautics and space administration",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "llm": "large language model",
    "kg": "knowledge graph",
}


def normalise(text: str) -> str:
    """Normalise text for fuzzy comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Expand common aliases
    for abbr, full in _ALIASES.items():
        # Whole-word match only
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


def _stem_verb(relation: str) -> str:
    """Normalise a relation verb for fuzzy matching.

    Takes the first word of the relation to match on the core verb:
    "developed theory of" → "developed"
    "manufactures battery cells at" → "manufactures"
    "co-developed" → "co-developed"
    """
    relation = normalise(relation)
    return relation.split()[0] if relation else relation


def entity_match_key(name: str, entity_type: str) -> str:
    """Create a match key for entity comparison.

    Uses normalised ``name::type`` so that "Margaret Hamilton" / "Person"
    matches "margaret hamilton" / "person".
    """
    return f"{normalise(name)}::{normalise(entity_type)}"


def entity_name_key(name: str) -> str:
    """Create a name-only match key for entity comparison (ignoring type)."""
    return normalise(name)


def relationship_match_key(
    source: str, target: str, relation: str
) -> str:
    """Create a match key for relationship comparison.

    Uses normalised ``source→target:relation`` so direction matters
    but whitespace/casing differences are tolerated.
    """
    return f"{normalise(source)}→{normalise(target)}:{normalise(relation)}"


def relationship_fuzzy_key(
    source: str, target: str, relation: str
) -> str:
    """Create a fuzzy match key for relationship comparison.

    Uses stemmed verb (first word only) and allows either direction:
    ``A→B:developed`` matches ``B→A:developed``.
    """
    src = normalise(source)
    tgt = normalise(target)
    verb = _stem_verb(relation)
    # Normalise direction: always put alphabetically smaller name first
    if src > tgt:
        src, tgt = tgt, src
    return f"{src}↔{tgt}:{verb}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Precision, recall, and F1 for a single category."""
    category: str
    precision: float
    recall: float
    f1: float
    predicted_count: int
    golden_count: int
    matched_count: int

    def summary(self) -> str:
        return (
            f"{self.category}: P={self.precision:.2%} R={self.recall:.2%} "
            f"F1={self.f1:.2%} ({self.matched_count}/{self.golden_count} golden, "
            f"{self.predicted_count} predicted)"
        )


@dataclass
class EvaluationResult:
    """Full evaluation result for a single golden case.

    Provides three-tier scoring:
    - ``entity_metrics``: strict match on (name, type)
    - ``entity_name_metrics``: match on name only (ignoring type)
    - ``relationship_metrics``: strict match on (source, target, relation)
    - ``rel_fuzzy_metrics``: fuzzy match (stemmed verb, bidirectional)
    """
    golden_key: str
    entity_metrics: MetricResult
    relationship_metrics: MetricResult
    # Two-tier: name-only matching (ignoring entity type)
    entity_name_metrics: Optional[MetricResult] = None
    # Fuzzy relationship matching (stemmed verb, bidirectional)
    rel_fuzzy_metrics: Optional[MetricResult] = None
    entity_false_positives: List[str] = field(default_factory=list)
    entity_false_negatives: List[str] = field(default_factory=list)
    relationship_false_positives: List[str] = field(default_factory=list)
    relationship_false_negatives: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation: {self.golden_key} ===",
            self.entity_metrics.summary(),
        ]
        if self.entity_name_metrics:
            lines.append(f"  (name-only: P={self.entity_name_metrics.precision:.2%} "
                         f"R={self.entity_name_metrics.recall:.2%} "
                         f"F1={self.entity_name_metrics.f1:.2%})")
        lines.append(self.relationship_metrics.summary())
        if self.rel_fuzzy_metrics:
            lines.append(f"  (rel-fuzzy: P={self.rel_fuzzy_metrics.precision:.2%} "
                         f"R={self.rel_fuzzy_metrics.recall:.2%} "
                         f"F1={self.rel_fuzzy_metrics.f1:.2%})")
        if self.entity_false_positives:
            lines.append(f"  Entity FP: {self.entity_false_positives[:5]}")
        if self.entity_false_negatives:
            lines.append(f"  Entity FN: {self.entity_false_negatives[:5]}")
        if self.relationship_false_positives:
            lines.append(f"  Rel FP: {self.relationship_false_positives[:5]}")
        if self.relationship_false_negatives:
            lines.append(f"  Rel FN: {self.relationship_false_negatives[:5]}")
        return "\n".join(lines)


@dataclass
class AggregateResult:
    """Aggregate metrics across multiple golden cases.

    Provides two-tier scoring:
    - entity_*: strict match on (name, type)
    - entity_name_*: match on name only (ignoring type)
    - relationship_*: strict match on (source, target, relation)
    - rel_fuzzy_*: fuzzy match (stemmed verb, bidirectional)
    """
    num_cases: int
    entity_precision: float
    entity_recall: float
    entity_f1: float
    relationship_precision: float
    relationship_recall: float
    relationship_f1: float
    # Two-tier: name-only matching
    entity_name_precision: float = 0.0
    entity_name_recall: float = 0.0
    entity_name_f1: float = 0.0
    # Fuzzy relationship matching
    rel_fuzzy_precision: float = 0.0
    rel_fuzzy_recall: float = 0.0
    rel_fuzzy_f1: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Aggregate ({self.num_cases} cases) ===",
            f"  Entities (name+type): P={self.entity_precision:.2%} "
            f"R={self.entity_recall:.2%} F1={self.entity_f1:.2%}",
            f"  Entities (name-only):  P={self.entity_name_precision:.2%} "
            f"R={self.entity_name_recall:.2%} F1={self.entity_name_f1:.2%}",
            f"  Rels (strict):         P={self.relationship_precision:.2%} "
            f"R={self.relationship_recall:.2%} F1={self.relationship_f1:.2%}",
            f"  Rels (fuzzy):          P={self.rel_fuzzy_precision:.2%} "
            f"R={self.rel_fuzzy_recall:.2%} F1={self.rel_fuzzy_f1:.2%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden test set
# ---------------------------------------------------------------------------

_GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass
class GoldenCase:
    """A golden test case with known-correct entities and relationships."""
    key: str
    text: str
    entities: List[Tuple[str, str, str]]  # (name, type, description)
    relationships: List[Tuple[str, str, str, str]]  # (source, target, relation, desc)

    @classmethod
    def from_dict(cls, key: str, data: dict) -> "GoldenCase":
        entities = [
            (e["entity_name"], e["entity_type"], e["entity_description"])
            for e in data.get("entities", [])
        ]
        relationships = [
            (
                r["source_entity"],
                r["target_entity"],
                r["relation"],
                r["relationship_description"],
            )
            for r in data.get("relationships", [])
        ]
        return cls(
            key=key,
            text=data["text"],
            entities=entities,
            relationships=relationships,
        )

    @classmethod
    def load(cls, key: str) -> "GoldenCase":
        """Load a golden case by key from the ``golden/`` directory."""
        path = _GOLDEN_DIR / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Golden case not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(key, data)

    @classmethod
    def load_all(cls) -> List["GoldenCase"]:
        """Load all golden cases from the ``golden/`` directory."""
        cases = []
        for path in sorted(_GOLDEN_DIR.glob("*.json")):
            key = path.stem
            data = json.loads(path.read_text(encoding="utf-8"))
            cases.append(cls.from_dict(key, data))
        return cases


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _compute_metrics(
    predicted_keys: set[str],
    golden_keys: set[str],
    category: str,
    predicted_raw: list[str],
    golden_raw: list[str],
) -> tuple[MetricResult, list[str], list[str]]:
    """Compute precision, recall, F1 from match-key sets."""
    matched = predicted_keys & golden_keys
    matched_count = len(matched)

    precision = matched_count / len(predicted_keys) if predicted_keys else 0.0
    recall = matched_count / len(golden_keys) if golden_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    false_positives = [k for k in predicted_raw if k not in golden_keys]
    false_negatives = [k for k in golden_raw if k not in predicted_keys]

    metrics = MetricResult(
        category=category,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=len(predicted_keys),
        golden_count=len(golden_keys),
        matched_count=matched_count,
    )
    return metrics, false_positives, false_negatives


class EvaluationHarness:
    """Evaluate KG extraction quality against golden test sets."""

    def evaluate(
        self,
        predicted_entities: List[Tuple[str, str, str]],
        predicted_relationships: List[Tuple[str, str, str, str]],
        golden_key: str,
    ) -> EvaluationResult:
        """Compare predicted extraction against a golden case.

        Parameters
        ----------
        predicted_entities : list of (name, type, description)
        predicted_relationships : list of (source, target, relation, description)
        golden_key : str
            Key to load from the golden/ directory (e.g. "hamilton").

        Returns
        -------
        EvaluationResult with entity and relationship metrics.
        """
        golden = GoldenCase.load(golden_key)

        # Entity matching: compare on (normalised_name, normalised_type)
        predicted_entity_keys = {
            entity_match_key(name, etype)
            for name, etype, _ in predicted_entities
        }
        golden_entity_keys = {
            entity_match_key(name, etype)
            for name, etype, _ in golden.entities
        }
        # Keep raw keys for FP/FN reporting
        predicted_entity_raw = [
            entity_match_key(name, etype)
            for name, etype, _ in predicted_entities
        ]
        golden_entity_raw = [
            entity_match_key(name, etype)
            for name, etype, _ in golden.entities
        ]

        e_metrics, e_fp, e_fn = _compute_metrics(
            predicted_entity_keys,
            golden_entity_keys,
            "entities",
            predicted_entity_raw,
            golden_entity_raw,
        )

        # Two-tier: name-only matching (ignoring entity type)
        predicted_name_keys = {normalise(name) for name, _, _ in predicted_entities}
        golden_name_keys = {normalise(name) for name, _, _ in golden.entities}
        predicted_name_raw = [normalise(name) for name, _, _ in predicted_entities]
        golden_name_raw = [normalise(name) for name, _, _ in golden.entities]
        e_name_metrics, _, _ = _compute_metrics(
            predicted_name_keys,
            golden_name_keys,
            "entities_name_only",
            predicted_name_raw,
            golden_name_raw,
        )

        # Relationship matching: compare on (normalised_source, normalised_target, normalised_relation)
        predicted_rel_keys = {
            relationship_match_key(src, tgt, rel)
            for src, tgt, rel, _ in predicted_relationships
        }
        golden_rel_keys = {
            relationship_match_key(src, tgt, rel)
            for src, tgt, rel, _ in golden.relationships
        }
        predicted_rel_raw = [
            relationship_match_key(src, tgt, rel)
            for src, tgt, rel, _ in predicted_relationships
        ]
        golden_rel_raw = [
            relationship_match_key(src, tgt, rel)
            for src, tgt, rel, _ in golden.relationships
        ]

        r_metrics, r_fp, r_fn = _compute_metrics(
            predicted_rel_keys,
            golden_rel_keys,
            "relationships",
            predicted_rel_raw,
            golden_rel_raw,
        )

        # Fuzzy relationship matching: stemmed verb + bidirectional
        predicted_fuzzy_keys = {
            relationship_fuzzy_key(src, tgt, rel)
            for src, tgt, rel, _ in predicted_relationships
        }
        golden_fuzzy_keys = {
            relationship_fuzzy_key(src, tgt, rel)
            for src, tgt, rel, _ in golden.relationships
        }
        predicted_fuzzy_raw = [
            relationship_fuzzy_key(src, tgt, rel)
            for src, tgt, rel, _ in predicted_relationships
        ]
        golden_fuzzy_raw = [
            relationship_fuzzy_key(src, tgt, rel)
            for src, tgt, rel, _ in golden.relationships
        ]

        r_fuzzy_metrics, _, _ = _compute_metrics(
            predicted_fuzzy_keys,
            golden_fuzzy_keys,
            "relationships_fuzzy",
            predicted_fuzzy_raw,
            golden_fuzzy_raw,
        )

        return EvaluationResult(
            golden_key=golden_key,
            entity_metrics=e_metrics,
            entity_name_metrics=e_name_metrics,
            relationship_metrics=r_metrics,
            rel_fuzzy_metrics=r_fuzzy_metrics,
            entity_false_positives=e_fp,
            entity_false_negatives=e_fn,
            relationship_false_positives=r_fp,
            relationship_false_negatives=r_fn,
        )

    def evaluate_all(
        self,
        extract_fn,
    ) -> AggregateResult:
        """Evaluate an extraction function against all golden cases.

        Parameters
        ----------
        extract_fn : callable
            Function that takes a text string and returns
            (entities, relationships) where:
            - entities: list of (name, type, description)
            - relationships: list of (source, target, relation, description)

        Returns
        -------
        AggregateResult with mean metrics across all golden cases.
        """
        cases = GoldenCase.load_all()
        if not cases:
            raise FileNotFoundError(f"No golden cases found in {_GOLDEN_DIR}")

        results: list[EvaluationResult] = []
        for case in cases:
            predicted_entities, predicted_rels = extract_fn(case.text)
            result = self.evaluate(
                predicted_entities,
                predicted_rels,
                golden_key=case.key,
            )
            results.append(result)

        # Average metrics across cases
        n = len(results)
        avg_ep = sum(r.entity_metrics.precision for r in results) / n
        avg_er = sum(r.entity_metrics.recall for r in results) / n
        avg_ef = sum(r.entity_metrics.f1 for r in results) / n
        avg_enp = sum(r.entity_name_metrics.precision for r in results if r.entity_name_metrics) / n
        avg_enr = sum(r.entity_name_metrics.recall for r in results if r.entity_name_metrics) / n
        avg_enf = sum(r.entity_name_metrics.f1 for r in results if r.entity_name_metrics) / n
        avg_rp = sum(r.relationship_metrics.precision for r in results) / n
        avg_rr = sum(r.relationship_metrics.recall for r in results) / n
        avg_rf = sum(r.relationship_metrics.f1 for r in results) / n
        avg_rfp = sum(r.rel_fuzzy_metrics.precision for r in results if r.rel_fuzzy_metrics) / n
        avg_rfr = sum(r.rel_fuzzy_metrics.recall for r in results if r.rel_fuzzy_metrics) / n
        avg_rff = sum(r.rel_fuzzy_metrics.f1 for r in results if r.rel_fuzzy_metrics) / n

        return AggregateResult(
            num_cases=n,
            entity_precision=avg_ep,
            entity_recall=avg_er,
            entity_f1=avg_ef,
            entity_name_precision=avg_enp,
            entity_name_recall=avg_enr,
            entity_name_f1=avg_enf,
            relationship_precision=avg_rp,
            relationship_recall=avg_rr,
            relationship_f1=avg_rf,
            rel_fuzzy_precision=avg_rfp,
            rel_fuzzy_recall=avg_rfr,
            rel_fuzzy_f1=avg_rff,
        )